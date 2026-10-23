# %%%% CELL 0 [code] len=6328
# Lab v4 inference kernel (T4, no internet). One code cell = this file; our code + weights come from one dataset.
# Dataset layout (made by package.py, flat): *.py, <member>.pt, <member>.json, members.json, parity_ref.npz
# Steps: parity gate on the visible test studies (vs local predictions from the SAME cache contract) -> all test
# studies through knee_v4_prep -> view slots -> geometry -> members (fp16) -> weighted mean -> submission.csv.
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Combined kernels: KERNEL_T0 = notebook start (a global when run as a cell, an env var when run as a subprocess),
# V4_OUT = where the predictions go (default: the submission itself); <V4_OUT>.status.json tells whether it is complete.
T0 = float(globals().get("KERNEL_T0") or os.environ.get("KERNEL_T0") or time.time())
TIME_LIMIT_S = float(os.environ.get("V4_TIME_LIMIT_S", 8.3 * 3600))
OUT = Path(os.environ.get("V4_OUT", "/kaggle/working/submission.csv"))
COMP = next(p for p in (Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
                        Path("/kaggle/input/rsna-knee-abnormality-detection"))
            if (p / "test_series.csv").is_file())
PKG = next(p.parent for pattern in ("*/members.json", "datasets/*/*/members.json")  # never walk the DICOM trees
           for p in Path("/kaggle/input").glob(pattern))
sys.path.insert(0, str(PKG))
from knee_v4_prep import process_series  # noqa: E402
import infer_core as IC  # noqa: E402
import torch  # noqa: E402

SPEC = json.loads((PKG / "members.json").read_text())
GEOMS = sorted({IC.geometry_key(m["geometry"]) for m in SPEC["members"]})
HOST = pd.read_csv(COMP / "test_series.csv", dtype=str)
print(f"package {PKG} | members {[m['name'] for m in SPEC['members']]} | geometries {GEOMS} | cpus {os.cpu_count()}", flush=True)


def prep_study(study_uid):
    """CPU worker: all series of one study -> view slots rendered for every needed geometry."""
    try:
        rows, jpegs = [], {}
        for series_uid in HOST.loc[HOST.StudyInstanceUID == study_uid, "SeriesInstanceUID"]:
            try:
                jp, meta = process_series(COMP / "test_series" / study_uid / series_uid)
            except Exception as error:  # a broken series is dropped, the study still gets predicted
                print(f"  [series fail] {series_uid[-12:]} {type(error).__name__}: {error}", flush=True)
                continue
            rows.append({"StudyInstanceUID": study_uid, "SeriesInstanceUID": series_uid, **meta})
            jpegs[series_uid] = jp
        if not rows:
            return study_uid, None, "no readable series"
        return study_uid, IC.study_views(rows, jpegs, HOST, GEOMS), ""
    except Exception as error:
        return study_uid, None, f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=3)}"


def load_members():
    members = []
    for m in SPEC["members"]:
        member = IC.Member(PKG / f"{m['name']}.json", PKG / f"{m['name']}.pt", device="cuda")
        members.append((m, member))
    return members


def predict(members, slots_by_geom, tta):
    total, wsum = np.zeros(12), 0.0
    for spec, member in members:
        slots = slots_by_geom[member.geom]
        if all(s is None for s in slots):
            continue
        p = member.predict(IC.study_batch(slots, member.k, member.canonical), mirror_tta=tta and member.tta_ok)[0]
        total += spec.get("weight", 1.0) * p
        wsum += spec.get("weight", 1.0)
    return total / wsum if wsum else None


def run(pool, study_uids, members, tta):
    global TIME_GUARD_HIT
    preds, failures = {}, []
    for i, (uid, slots, err) in enumerate(pool.imap(prep_study, study_uids), 1):
        p = predict(members, slots, tta) if slots is not None else None
        if p is None:
            failures.append((uid, err or "no views"))
        else:
            preds[uid] = p
        if i % 100 == 0 or i == len(study_uids):
            el = time.time() - T0
            print(f"  {i}/{len(study_uids)} studies, fails {len(failures)}, {el / 60:.1f} min", flush=True)
        if time.time() - T0 > TIME_LIMIT_S:
            print("TIME GUARD: remaining studies get the fallback row", flush=True)
            TIME_GUARD_HIT = True
            break
    return preds, failures


TIME_GUARD_HIT = False


def main():
    pool = mp.get_context("fork").Pool(os.cpu_count())  # forked before CUDA is initialised; workers stay on CPU
    members = load_members()
    tta = bool(SPEC.get("mirror_tta", False))
    test = pd.read_csv(COMP / "test.csv", dtype=str)
    # Parity gate: studies whose local predictions (training-time store path) were packaged.
    ref = np.load(PKG / "parity_ref.npz", allow_pickle=False)
    ref_uids = [u for u in ref["study_uids"].astype(str) if u in set(test.StudyInstanceUID)]
    if ref_uids:
        got, fails = run(pool, ref_uids, members, False)
        assert not fails, f"parity studies failed: {fails}"
        want = ref["probabilities"][[list(ref["study_uids"].astype(str)).index(u) for u in ref_uids]]
        have = np.stack([got[u] for u in ref_uids])
        diff = float(np.abs(have - want).max())
        print(f"PARITY on {len(ref_uids)} studies: max |dp| = {diff:.5f}", flush=True)
        assert diff < 0.02, "parity gate failed: kernel preprocessing differs from training"
    preds, failures = run(pool, list(test.StudyInstanceUID), members, tta)
    pool.close()
    prior = np.array(SPEC.get("fallback", [0.5] * 12), float)
    rows = [[uid, *preds.get(uid, prior)] for uid in test.StudyInstanceUID]
    sub = pd.DataFrame(rows, columns=["StudyInstanceUID", *IC.LABELS])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT.with_name(OUT.name + ".tmp"), index=False)
    os.replace(OUT.with_name(OUT.name + ".tmp"), OUT)
    status = {"rows": len(sub), "fallback_rows": len(test) - len(preds), "time_guard": TIME_GUARD_HIT,
              "failures": [f[0] for f in failures], "minutes_since_t0": round((time.time() - T0) / 60, 1)}
    Path(str(OUT) + ".status.json").write_text(json.dumps(status))
    print(f"submission {sub.shape}, fallback rows {len(test) - len(preds)}, failures {failures[:5]}, "
          f"{(time.time() - T0) / 60:.1f} min", flush=True)


main()

