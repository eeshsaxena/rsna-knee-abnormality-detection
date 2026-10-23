"""Item 11: our leg, run by the notebook's runner cell as a child process (its own CUDA context).

    python leg_main.py --out _ours.csv --report _leg_child_report.json [--deadline EPOCH_SECONDS]

members.json beside this file lists the checkpoints: {"members": [{"ckpt": "<file>", "weight": 1.0}, ...]}. Every checkpoint has a
<ckpt>.meta.json (arch, img, laterality, lat_mm, geometry, tta_offsets, ckpt_sha256; written by rsna/ckpt_meta.py) and is checked against the
sha256 in it. Members that share a pack recipe (geometry, laterality, lat_mm) are scored from the same packs, so a second member of the same
geometry costs scoring time only. A study's value is the weighted mean of the probabilities of the members that scored it. A study that no
member scored is left out of the output, so the blend keeps its public rank. At the deadline the loop stops taking studies and the file holds
what was scored. pack_submit.refuse() stops any path that names the gold sets; nothing here reads a label.

Exit codes: 0 with at least one study scored, 3 with none, anything else is an uncaught error."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pack_submit as P  # noqa: E402

ID = "StudyInstanceUID"


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_key(meta: dict) -> tuple:
    """Members with the same key read the same packs."""
    lat = bool(meta["laterality"])
    return (meta["geometry"], lat, float(meta["lat_mm"]) if lat else None)


def member_offsets(meta: dict) -> tuple:
    """Slice-offset passes, pack5 only (the Fine Spacing windows are all scored already); same rule as pack_submit.run_submit."""
    if meta.get("tta_offsets") and meta["geometry"] == "pack5":
        return tuple(float(x) for x in meta["tta_offsets"])
    return (0.0,)


def load_members(folder: str) -> list[dict]:
    with open(os.path.join(folder, "members.json"), encoding="utf-8") as f:
        spec = json.load(f)
    members = []
    for i, item in enumerate(spec["members"]):
        path = os.path.join(folder, item["ckpt"])
        if not os.path.isfile(path):
            raise FileNotFoundError(f"member {i}: {item['ckpt']} is not in {folder}")
        meta = P.ckpt_meta(path)
        digest = sha256_of(path)
        if meta.get("ckpt_sha256") and meta["ckpt_sha256"] != digest:
            raise ValueError(f"member {i}: {item['ckpt']} does not match the sha256 in its meta file")
        weight = float(item.get("weight", 1.0))
        if not weight > 0:
            raise ValueError(f"member {i}: weight must be positive")
        members.append({"i": i, "name": item["ckpt"], "path": path, "meta": meta, "sha256": digest, "weight": weight,
                        "offsets": member_offsets(meta), "n_scored": 0, "n_errors": 0})
    if not members:
        raise ValueError("members.json lists no member")
    return members


def lower_oom_priority(path: str = "/proc/self/oom_score_adj", value: str = "1000") -> bool:
    """If the machine runs out of memory the kernel should end this child (its pool workers inherit the setting), not the notebook."""
    try:
        with open(path, "w") as f:
            f.write(value)
        return True
    except OSError:
        return False


def pick_device(torch) -> tuple[str, list]:
    """cuda with the most free memory (the public pipeline may still hold some of either card), else cpu."""
    if not torch.cuda.is_available():
        return "cpu", []
    free = []
    for i in range(torch.cuda.device_count()):
        try:
            free.append(round(torch.cuda.mem_get_info(i)[0] / 1024 ** 3, 2))
        except Exception:
            free.append(0.0)
    best = int(np.argmax(free)) if free else 0
    torch.cuda.set_device(best)
    return "cuda", free


def score_safe(model, packs, masks, device, member, geo, empty_cache):
    """One member, one batch. A failed batch is retried one study at a time before any study is given up."""
    meta = member["meta"]
    kw = dict(fp16=(device == "cuda"), img=int(meta["img"]), k_eval=geo["k_eval"], bounds=geo["bounds"], offsets=member["offsets"])
    if meta.get("norm_mean"):                                  # item 10: a checkpoint trained with its timm normalisation (needs the newer packer)
        kw["norm"] = P.norm_of(meta)
    chunk = meta.get("score_studies")                          # item 9: a large model is scored a few studies at a time (needs the newer packer)
    try:
        if chunk and hasattr(P, "score_chunks"):
            probs, _ = P.score_chunks(model, packs, masks, device, chunk, **kw)
        else:
            probs, _ = P.score_batch(model, packs, masks, device, **kw)
        return [np.asarray(p, np.float64) for p in probs], [None] * len(packs)
    except Exception:
        empty_cache()
    out, errs = [], []
    for pack, mask in zip(packs, masks):
        try:
            probs, _ = P.score_batch(model, [pack], [mask], device, **kw)
            out.append(np.asarray(probs[0], np.float64))
            errs.append(None)
        except Exception as exc:
            empty_cache()
            out.append(None)
            errs.append(f"score_{type(exc).__name__}")
    return out, errs


def serve(members, comp=None, img_root=None, deadline=None, workers=4, batch=4, pool_factory=None, device=None, now=time.time):
    """Returns (frame of the scored studies in test order, report)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    import torch
    t_all = time.perf_counter()
    comp = P.find_comp(comp)
    if comp is None:
        raise RuntimeError("test.csv not found")
    P.refuse(comp)
    test = pd.read_csv(os.path.join(comp, "test.csv"), dtype={ID: str})
    series = pd.read_csv(os.path.join(comp, "test_series.csv"), dtype={ID: str, "SeriesInstanceUID": str})
    grouped = {k: v for k, v in series.groupby(ID, sort=False)}
    uids = test[ID].astype(str).tolist()
    img_root = P.find_image_root(comp, uids, img_root)
    if img_root is not None:
        P.refuse(img_root)
    header = P.header_of(os.path.join(comp, "sample_submission.csv"))
    gpu_free = []
    if device is None:
        device, gpu_free = pick_device(torch)
    empty_cache = torch.cuda.empty_cache if device == "cuda" else (lambda: None)
    code_dir = HERE
    os.environ["PYTHONPATH"] = code_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    ctx = mp.get_context("spawn")
    make_pool = pool_factory or (lambda: ProcessPoolExecutor(max_workers=workers, mp_context=ctx))

    groups: dict = {}
    for m in members:
        groups.setdefault(pack_key(m["meta"]), []).append(m)
    parts: dict[str, list] = {}                         # uid -> [(probabilities, weight)]
    group_reports, pack_errors = [], {}
    peak_rss = P._tree_rss_gb()
    deadline_hit = False
    t_load = time.perf_counter()
    models = {m["i"]: P._load_model(m["path"], device, m["meta"]["arch"]) for m in members}
    load_s = time.perf_counter() - t_load

    for key, group in groups.items():
        geometry, _lat, lat_mm = key
        geo = P.GEOMETRIES[geometry]
        t_group = time.perf_counter()
        if deadline is not None and now() >= deadline:
            deadline_hit = True
            group_reports.append({"key": list(key), "members": [m["i"] for m in group], "skipped": "deadline", "seconds": 0.0})
            continue
        jobs, errors = [], {}
        for uid in uids:
            records = grouped.get(uid)
            if img_root is None or records is None:
                errors[uid] = "no_series"
            else:
                jobs.append((uid, records.to_dict("records"), img_root, lat_mm, geometry))
        n_packed = 0
        with make_pool() as pool:
            for results in P.iter_job_batches(pool, jobs, fn=P.pack_study_job, size=batch):
                if deadline is not None and now() >= deadline:
                    deadline_hit = True
                    try:
                        pool.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                    break
                good = [r for r in results if r["error"] is None]
                for r in results:
                    if r["error"] is not None:
                        errors[r["uid"]] = r["error"]
                if good:
                    packs, masks = [r["pack"] for r in good], [r["mask"] for r in good]
                    for m in group:
                        probs, errs = score_safe(models[m["i"]], packs, masks, device, m, geo, empty_cache)
                        for r, p, e in zip(good, probs, errs):
                            if p is None:
                                m["n_errors"] += 1
                                errors.setdefault(r["uid"], e)
                            else:
                                parts.setdefault(r["uid"], []).append((p, m["weight"]))
                                m["n_scored"] += 1
                    n_packed += len(good)
                for r in results:
                    r["pack"] = None
                    r["mask"] = None
                rss = P._tree_rss_gb()
                if rss is not None:
                    peak_rss = rss if peak_rss is None else max(peak_rss, rss)
        for uid, why in errors.items():
            pack_errors[why] = pack_errors.get(why, 0) + 1
        group_reports.append({"key": list(key), "members": [m["i"] for m in group], "n_jobs": len(jobs), "n_packed": n_packed,
                              "seconds": round(time.perf_counter() - t_group, 1)})
        if deadline_hit:
            break

    rows = []
    for uid in uids:
        got = parts.get(uid)
        if not got:
            continue
        weights = np.array([w for _, w in got], np.float64)
        values = np.stack([p for p, _ in got])
        mean = (values * weights[:, None]).sum(0) / weights.sum()
        rows.append([uid] + [float(x) for x in mean])
    frame = pd.DataFrame(rows, columns=[ID] + P.FINDINGS).reindex(columns=list(header))
    report = {
        "ok": len(rows) > 0,
        "n_test": len(uids),
        "n_scored": len(rows),
        "n_missing": len(uids) - len(rows),
        "partial": bool(deadline_hit),
        "deadline_hit": bool(deadline_hit),
        "errors_by_reason": pack_errors,
        "header_match": list(frame.columns) == list(header),
        "device": device,
        "gpu_free_gb_at_start": gpu_free,
        "fp16": device == "cuda",
        "members": [{"i": m["i"], "ckpt": m["name"], "sha256": m["sha256"], "arch": m["meta"]["arch"], "geometry": m["meta"]["geometry"],
                     "laterality": m["meta"]["laterality"], "tta_offsets": list(m["offsets"]), "weight": m["weight"],
                     "row": m["meta"].get("row"), "n_scored": m["n_scored"], "n_errors": m["n_errors"]} for m in members],
        "groups": group_reports,
        "load_s": round(load_s, 3),
        "runtime_s": round(time.perf_counter() - t_all, 3),
        "peak_rss_gb": None if peak_rss is None else round(peak_rss, 3),
        "comp": comp,
        "img_root": img_root,
        "torch": getattr(torch, "__version__", None),
        "gpu_peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3) if device == "cuda" else None,
        "gold": "not read",
    }
    return frame, report


def main(argv=None, pool_factory=None) -> int:
    import multiprocessing as mp
    if mp.current_process().name != "MainProcess":
        return 1
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--deadline", type=float, default=None, help="epoch seconds; the loop stops taking studies at this time")
    ap.add_argument("--members-dir", default=HERE)
    ap.add_argument("--comp", default=None, help="for a local run; on Kaggle the competition folder is found")
    ap.add_argument("--img-root", default=None, help="for a local run")
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    oom_adj = lower_oom_priority()
    members = load_members(args.members_dir)
    frame, report = serve(members, args.comp, args.img_root, args.deadline, workers=args.workers, device=args.device,
                          pool_factory=pool_factory)
    report["deadline"] = args.deadline
    report["oom_score_adj_set"] = oom_adj
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    frame.to_csv(tmp, index=False)
    os.replace(tmp, args.out)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps({k: report[k] for k in ("ok", "n_test", "n_scored", "partial", "runtime_s", "load_s", "device", "peak_rss_gb")}, default=str))
    return 0 if report["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
