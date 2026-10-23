#!/usr/bin/env python3
# s6f_driver.py [eeshsaxena V224-FLIP-S6F; not in any public notebook]. Written by build_v224s6f.py from s6f_driver_src.py.
# The B path ONLY of nartaa/rsna-knee-s6fall-solo (public, .950 solo): every "VERBATIM" block below is copied byte for byte from
# its version 2 source (sha256 6e87350d..., the run whose public commit output is our parity reference; its code tree, model and
# worker arguments are the ones version 1 scored with, each file checked against the same sha256 manifest). The builder proves
# each block equal to the pulled source (ast segment, byte for byte). NOT included: remove_submission(), the 8 h hang guard
# (_watchdog / arm_guard / faulthandler backstop), component A, the blend, the parity mode, and every "HARD ERROR" of a B failure.
# Our parts (marked [s6f]): paths, the argument parser, the environment-isolation probe, run_b_s6f (= nartaa's run_b with a
# deadline that cuts instead of raising, partial coverage allowed, a killed part read from its predictions.csv), main().
# This process never touches /kaggle/working/submission.csv: it writes only under --out (default /kaggle/working/s6f) and --tmp.
# The notebook cell that starts it owns the hard deadline and kills every S6F process group at it. No em dashes.
import argparse, hashlib, json, os, shutil, signal, subprocess, sys, tarfile, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd

# ---- [s6f] paths (nartaa: KROOT /kaggle, OUT /kaggle/working/k1, TMP /tmp/k1); set by configure() --------------------------
KROOT = Path("/kaggle")
INPUT, WORK = KROOT / "input", KROOT / "working"
OUT = WORK / "s6f"
TMP = Path("/tmp/s6f/work")
CODE_MANIFEST_SHA256 = "__CODE_MANIFEST_SHA256__"   # [s6f] code_manifest.json of the scored run (nartaa's commit receipt)
R7_ROOTS = ("/tmp/probe", "/kaggle/temp/probe")     # [s6f] R7's code / wheel dirs (V224 cell 6): never on our path

# ---- VERBATIM constants of nartaa's kernel (B path) -------------------------------------------------------------------------
# @@VERBATIM COMP
# @@VERBATIM DS_SLUG
# @@VERBATIM HELIOS_SLUG
# @@VERBATIM EXTRAS_FILE
# @@VERBATIM LAB
# @@VERBATIM B_MODEL
# @@VERBATIM ANATOMY_FILE
# @@VERBATIM PAYLOAD_FILE
# @@VERBATIM SOURCE_SHA
# @@VERBATIM B_RECEIPT_SOURCES
# @@VERBATIM CODE_SHA
# @@VERBATIM B_GEOMETRY
# @@VERBATIM B_AMP
# @@VERBATIM B_WORKERS
# @@VERBATIM EXIT_GRACE_S
# @@VERBATIM B_DEADLINE_MARGIN_S
# @@VERBATIM B_RETRY_MIN_S
# @@VERBATIM HEARTBEAT_S
MODE = "submit"                  # [s6f] nartaa's MODE value (prepare_inputs reads it; parity mode is not ported)

T0 = time.time()
_LOG = None
_CHILDREN = {}                   # pgid -> name
_RECEIPT = {"tag": "s6f", "stages": {}, "gpu": {}}


# ---- VERBATIM helpers ---------------------------------------------------------------------------------------------------------
# @@VERBATIM K1Error
# @@VERBATIM K1Timeout
# @@VERBATIM log
# @@VERBATIM sha256_file
# @@VERBATIM save_json
# @@VERBATIM tail
# @@VERBATIM gpu_snapshot
# @@VERBATIM mem_available_mb
# @@VERBATIM group_rss_mb
# @@VERBATIM kill_group
# @@VERBATIM kill_all_children
# @@VERBATIM stage
# @@VERBATIM stage_done
# @@VERBATIM _size
# @@VERBATIM run_group

# ---- VERBATIM environment and inputs (prepare_inputs, probe_b_stack and their helpers) ---------------------------------------
# @@VERBATIM PROBE
# @@VERBATIM probe_env
# @@VERBATIM find_mount
# @@VERBATIM safe_rel
# @@VERBATIM extract_payload
# @@VERBATIM LAUNCHER
# @@VERBATIM B_PROBE
# @@VERBATIM probe_b_stack
# @@VERBATIM write_launcher
# @@VERBATIM Inputs
# @@VERBATIM prepare_inputs

# ---- VERBATIM component B helpers (run_b's building blocks) -------------------------------------------------------------------
# @@VERBATIM b_cmd
# @@VERBATIM split_parts
# @@VERBATIM b_env
# @@VERBATIM collect_part
# @@VERBATIM progress_lines
# @@VERBATIM b_done


# ---- [s6f] ours ---------------------------------------------------------------------------------------------------------------
def configure(kaggle_root=None, out=None, tmp=None):
    """[s6f] paths (tests point them at a temp dir)."""
    global KROOT, INPUT, WORK, OUT, TMP
    if kaggle_root is not None:
        KROOT = Path(kaggle_root)
        INPUT, WORK = KROOT / "input", KROOT / "working"
        OUT = WORK / "s6f"
    if out is not None:
        OUT = Path(out)
    if tmp is not None:
        TMP = Path(tmp)


ENV2 = r'''
import importlib, json, os, sys
out = {"executable": sys.executable, "sys_path": list(sys.path), "PYTHONPATH": os.environ.get("PYTHONPATH"),
       "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"), "MPLBACKEND": os.environ.get("MPLBACKEND"), "mods": {}}
for m in ("torch", "timm", "pydicom", "numpy", "pandas", "scipy", "cv2", "PIL", "safetensors", "setuptools",
          "pylibjpeg", "libjpeg", "openjpeg"):
    try:
        mod = importlib.import_module(m)
        out["mods"][m] = [str(getattr(mod, "__version__", "?")), os.path.dirname(os.path.dirname(os.path.abspath(mod.__file__)))]
    except Exception as e:
        out["mods"][m] = ["MISSING", type(e).__name__]
print("S6F_ENV2 " + json.dumps(out), flush=True)
'''


def env_isolation():
    """[s6f] what the S6F processes import, from where: a fresh interpreter with THIS process's environment (= what the B
    probe and the B workers inherit). Fails if an R7 dir (/tmp/probe/..) or /kaggle/working is on the import path."""
    r = subprocess.run([sys.executable, "-c", ENV2], capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL,
                       cwd=str(TMP))
    line = [l for l in r.stdout.splitlines() if l.startswith("S6F_ENV2 ")]
    if r.returncode != 0 or not line:
        raise K1Error("environment isolation probe failed rc %d: %s %s" % (r.returncode, r.stdout[-800:], r.stderr[-800:]))
    e = json.loads(line[0][len("S6F_ENV2 "):])
    bad = [p for p in e["sys_path"] + [sys.executable] if str(p).startswith(R7_ROOTS + (str(WORK),))]
    e["r7_or_working_on_path"] = bad
    e["driver_sys_path"] = list(sys.path)
    if bad or any(str(p).startswith(R7_ROOTS) for p in sys.path):
        raise K1Error("an R7 / working dir is on the S6F import path: %s" % bad)
    return e


def collect_part_csv(pdir, part):
    """[s6f] a part that has no receipt (its worker group was killed, or it crashed): the studies its compact_infer already wrote
    to predictions.csv (one row per scored study, flushed, '%.6f' probabilities) count as scored; z = logit(p) clipped to the
    6-decimal grid. Integrity: the 12 columns in label order, the row uid in the part, finite in [0, 1]."""
    z = {}
    p = pdir / "predictions.csv"
    if not p.is_file():
        return z
    try:
        pc = pd.read_csv(p, dtype={"StudyInstanceUID": str})
    except Exception:
        return z
    if list(pc.columns) != ["StudyInstanceUID"] + LAB:
        return z
    want = set(part)
    for _, r in pc.iterrows():
        u = r["StudyInstanceUID"]
        v = np.asarray([r[l] for l in LAB], np.float64)
        if u in want and np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all():
            v = np.clip(v, 5e-7, 1 - 5e-7)
            z[u] = np.log(v / (1 - v))
    return z


def run_b_s6f(I, tag, ids, split, stop_at):
    """[s6f] nartaa's run_b with these changes only (everything a scored study goes through is unchanged: split_parts, b_cmd,
    b_env, the launcher, compact_infer, collect_part's checks, z = mean logit, P = sigmoid(z), the B_probs.csv format):
    - budget: compact_infer --deadline = stop_at - B_DEADLINE_MARGIN_S (no NEW study after it), run_group timeout = stop_at - now;
    - a deadline, a failed / stalled / killed part, a retry that runs out of time: never an error. Studies without logits are
      'unscored' (NaN rows) and coverage = scored / N is reported; the notebook blend decides (>= 98 % or the S6F canary);
    - a part without a receipt: its predictions.csv rows (collect_part_csv);
    - the one retry of non-deadline fallbacks runs only if B_RETRY_MIN_S remain before stop_at;
    - no fail cap, no hard error."""
    def deadline_named(fb):
        return [u for u, v in fb.items() if "deadline" in str(v).lower()]
    root = OUT / tag
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    n_gpu = max(1, int(_RECEIPT["env"].get("n_gpu", 1)))
    parts = split_parts(ids, I.cost, n_gpu)
    t_start = time.time()
    deadline = stop_at - B_DEADLINE_MARGIN_S
    specs, pdirs = [], []
    for gi, part in enumerate(parts):
        pdir = root / ("part%d" % gi)
        pdir.mkdir()
        (pdir / "ids.json").write_text(json.dumps(part))
        specs.append({"name": "%s.part%d" % (tag, gi), "cmd": b_cmd(I, split, pdir / "ids.json", pdir, TMP / ("%s_tmp%d" % (tag, gi)), deadline),
                      "env": b_env(gi, None), "cwd": str(TMP), "log": pdir / "log.txt", "done_when": (lambda d=pdir: b_done(d))})
        pdirs.append(pdir)
    (root / "argv_part0.json").write_text(json.dumps(specs[0]["cmd"]))
    log("[%s] %d studies over %d GPU process(es) %s, stop_at +%ds, compact_infer deadline +%ds" % (
        tag, len(ids), len(parts), [len(p) for p in parts], stop_at - t_start, deadline - t_start))
    run_err = None
    try:
        runs = run_group(specs, max(30.0, stop_at - time.time()), tag)
    except K1Error as e:                               # K1Timeout included: the surviving parts' outputs are still read
        runs, run_err = None, repr(e)[:3000]
        log("[%s] run_group: %s" % (tag, repr(e)[:600]))
    z, fallback, receipts, sources = {}, {}, [], []
    for pdir, part in zip(pdirs, parts):
        if (pdir / "receipt.json").is_file():
            try:
                zz, fb, rec = collect_part(pdir, part, I)
                src = "logits"
            except Exception as e:
                zz, fb, rec, src = {}, {u: "collect: %r" % (e,) for u in part}, {}, "collect_failed: %r" % (e,)
        else:
            zz = collect_part_csv(pdir, part)
            fb, rec, src = {u: "no receipt (part killed or crashed)" for u in part if u not in zz}, {}, "predictions_csv"
        z.update(zz); fallback.update(fb)
        receipts.append({k: rec.get(k) for k in ("studies", "scored", "seconds", "amp", "gpu", "torch", "timm", "stats", "timed_out",
                                                 "anchor_methods", "per_study")})
        receipts[-1]["progress"] = progress_lines(pdir / "log.txt")
        sources.append(src)
    timed = [i for i, rc in enumerate(receipts) if rc.get("timed_out") is not False]
    retry = None
    redo = sorted(u for u in fallback if u not in z and u not in deadline_named(fallback))
    left = stop_at - time.time()
    if redo and left >= B_RETRY_MIN_S and not timed:
        log("[%s] %d fallback studies %s; %.0fs left before stop_at: one retry" % (tag, len(redo), dict(list(fallback.items())[:5]), left))
        rdir = root / "retry"
        rdir.mkdir()
        (rdir / "ids.json").write_text(json.dumps(redo))
        rdeadline = time.time() + max(left - B_DEADLINE_MARGIN_S, left * 0.5)
        rerr = None
        try:
            rruns = run_group([{"name": tag + ".retry", "cmd": b_cmd(I, split, rdir / "ids.json", rdir, TMP / (tag + "_tmp_retry"), rdeadline),
                                "env": b_env(0, None), "cwd": str(TMP), "log": rdir / "log.txt",
                                "done_when": lambda: b_done(rdir)}], max(30.0, stop_at - time.time()), tag)
        except K1Error as e:
            rruns, rerr = None, repr(e)[:2000]
        if (rdir / "receipt.json").is_file():
            try:
                zz, fb2, rec2 = collect_part(rdir, redo, I)
            except Exception as e:
                zz, fb2, rec2 = {}, {u: "collect: %r" % (e,) for u in redo}, {}
        else:
            zz = collect_part_csv(rdir, redo)
            fb2, rec2 = {u: "retry: no receipt" for u in redo if u not in zz}, {}
        z.update(zz)
        retry = {"ids": redo, "first_reasons": {u: fallback.get(u) for u in redo}, "still_failing": fb2, "seconds": rec2.get("seconds"),
                 "timed_out": rec2.get("timed_out"), "runs": rruns, "error": rerr}
        for u in redo:
            if u in zz:
                fallback.pop(u, None)
            else:
                fallback[u] = fb2.get(u, fallback.get(u))
    elif redo:
        log("[%s] %d fallback studies, %.0fs left before stop_at: no retry" % (tag, len(redo), left))
    missing = [u for u in ids if u not in z]
    failed = {u: str(fallback.get(u, "no logits"))[:300] for u in missing}
    nan12 = np.full(len(LAB), np.nan)
    Z = np.stack([z.get(u, nan12) for u in ids])
    P = 1.0 / (1.0 + np.exp(-Z))
    ok = np.array([u in z for u in ids])
    nonfinite = int((~np.isfinite(P[ok])).any(axis=1).sum()) if ok.any() else 0
    df = pd.DataFrame({"StudyInstanceUID": ids})
    for j, l in enumerate(LAB):
        df[l] = P[:, j]
    for j, l in enumerate(LAB):
        df["z|" + l] = Z[:, j]
    df.to_csv(root / ("%s_probs.csv" % tag), index=False, float_format="%.17g")
    secs = round(time.time() - t_start, 1)
    per_gpu = [rc.get("seconds") / max(1, rc.get("scored") or 0) for rc in receipts if rc.get("seconds") and rc.get("scored")]
    st = [rc.get("stats") or {} for rc in receipts]
    rows = sum(int(s_.get("rows") or 0) for s_ in st)
    info = {"seconds": secs, "stop_at": stop_at, "ci_deadline": deadline, "parts": [len(p) for p in parts], "runs": runs,
            "run_error": run_err, "sources": sources, "timed_out_parts": timed, "receipts": receipts, "retry": retry,
            "n": len(ids), "scored": int(ok.sum()), "coverage": float(ok.mean()) if len(ids) else 0.0,
            "nonfinite_scored_rows": nonfinite, "unscored": dict(list(failed.items())[:50]), "n_unscored": len(failed),
            "microbench": {"s_per_study_per_gpu_max": round(max(per_gpu), 3) if per_gpu else None,
                           "gpu_s_per_study": round(sum(float(s_.get("gpu_s") or 0) for s_ in st) / rows, 3) if rows else None,
                           "decode_s_per_study_per_worker": round(sum(float(s_.get("decode_s") or 0) for s_ in st) / rows, 3) if rows else None,
                           "windows_per_study": round(sum(float(s_.get("windows") or 0) for s_ in st) / rows, 1) if rows else None,
                           "wait_s_per_study": round(sum(float(s_.get("wait_s") or 0) for s_ in st) / rows, 3) if rows else None},
            "probs_sha256": sha256_file(root / ("%s_probs.csv" % tag))}
    save_json(root / ("%s_info.json" % tag), info)
    log("[%s] done: %d / %d studies scored (coverage %.4f), %.0fs, timed-out parts %s, retry %s, sources %s" % (
        tag, int(ok.sum()), len(ids), info["coverage"], secs, timed, None if retry is None else len(retry["ids"]), sources))
    return P, Z, info, failed


def main(argv=None):
    """[s6f] env probe -> inputs (nartaa's prepare_inputs + our code-manifest pin) -> B stack probe -> B. Exit 0 with
    s6f_driver_receipt.json {"ok": true} when B_probs.csv was written (any coverage); exit 3 with {"ok": false, "error"} else."""
    global _LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-at", type=float, required=True, help="epoch s: every S6F process ends by then (cell kills later)")
    ap.add_argument("--nb-t0", type=float, default=0.0, help="the notebook's NB_T0 (receipt ages)")
    ap.add_argument("--kaggle-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tmp", default=None)
    a = ap.parse_args(argv)
    configure(a.kaggle_root, a.out, a.tmp)
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    for stale in ("s6f_driver_receipt.json",):
        try:
            (OUT / stale).unlink()
        except FileNotFoundError:
            pass
    _LOG = open(OUT / "s6f_driver_log.txt", "a")
    age = lambda: round((time.time() - a.nb_t0) / 3600.0, 4) if a.nb_t0 else None
    _RECEIPT.update(ok=False, stop_at=a.stop_at, nb_t0=a.nb_t0, start_age_h=age(), pid=os.getpid(),
                    pgid=os.getpgrp() if hasattr(os, "getpgrp") else None, code_manifest_pin=CODE_MANIFEST_SHA256)
    rc = 0
    try:
        log("S6F_DRIVER start | python %s | stop_at in %.0fs | model %s %s... | out %s | tmp %s" % (
            sys.version.split()[0], a.stop_at - time.time(), B_MODEL["file"], B_MODEL["sha256"][:12], OUT, TMP))
        stage("env")
        env = probe_env()
        _RECEIPT["env"] = env
        _RECEIPT["env_isolation"] = env_isolation()
        log("S6F_ENV", json.dumps(env))
        log("S6F_ENV2", json.dumps({k: v for k, v in _RECEIPT["env_isolation"].items() if k in ("PYTHONPATH", "mods", "r7_or_working_on_path")}))
        if not env.get("cuda_available") or env.get("n_gpu", 0) < 1:
            raise K1Error("CUDA is not available")
        stage_done("env")
        stage("inputs")
        I = prepare_inputs()
        if I.manifest_sha256 != CODE_MANIFEST_SHA256:
            raise K1Error("code_manifest.json %s... is not the scored run's %s..." % (I.manifest_sha256[:16], CODE_MANIFEST_SHA256[:16]))
        stage_done("inputs")
        stage("B_probe")
        _RECEIPT["B_probe"] = probe_b_stack(I)
        log("K1_BPROBE", json.dumps(_RECEIPT["B_probe"]))
        stage_done("B_probe")
        stage("B")
        _P, _Z, b_info, b_failed = run_b_s6f(I, "B", I.test_ids, "test", a.stop_at)
        _RECEIPT["B"] = {k: v for k, v in b_info.items() if k != "receipts"}
        _RECEIPT["B"]["per_part"] = [{k: rc_.get(k) for k in ("studies", "scored", "seconds", "stats", "timed_out", "amp", "timm", "torch")}
                                     for rc_ in b_info["receipts"]]
        stage_done("B", coverage=b_info["coverage"])
        _RECEIPT["ok"] = True
    except BaseException as e:
        kill_all_children()
        _RECEIPT["ok"] = False
        _RECEIPT["error"] = repr(e)[:3000]
        _RECEIPT["error_trace"] = traceback.format_exc()[-4000:]          # receipt file only (kept out of the kernel log)
        log("S6F_DRIVER_FAILED %s" % repr(e)[:600])
        rc = 3
    finally:
        _RECEIPT["end_age_h"] = age()
        _RECEIPT["total_seconds"] = round(time.time() - T0, 1)
        save_json(OUT / "s6f_driver_receipt.json", _RECEIPT)
        log("S6F_DRIVER_DONE %s" % json.dumps({"ok": _RECEIPT["ok"], "seconds": _RECEIPT["total_seconds"],
                                               "coverage": (_RECEIPT.get("B") or {}).get("coverage"),
                                               "stages": {k: v.get("seconds") for k, v in _RECEIPT["stages"].items()}}))
    return rc


if __name__ == "__main__":
    sys.exit(main())
