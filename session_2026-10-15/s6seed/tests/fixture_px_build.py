"""S6 px cache builder (BOX_PLAN_1015 S3 prototype): raw DICOMs -> the compact_train pixel cache (px.u8.npy, meta.f32.npy,
index.json) for the S6fall recipe: size 320, views v4, slices 3, phys 3.5 mm, eval stride 1, SKM-TEA knee anchors (anatomy.pt
5904dd16), NO side model (nartaa's scored S6fall serves without one: argv_part0 has no --side-model).

Every study goes through compact_infer 34f433fe CStudies.rows() = native render -> geometry bank -> anchors ->
px_cache.StudySlices.build, the same path CStudies.__getitem__ serves at inference, so the training rows are the serve rows by
construction (S4 PX_PARITY checks it end to end on gold58).

Parallel: --procs spawn workers, worker i renders uids[i::procs] into <out>/partNNN (raw px / meta + part.json, resumable: a
restarted worker skips finished studies and truncates a half-written tail); --merge concatenates the parts into the trainer's
--px directory.  Records s/study and the worker's peak RSS per study (S4: sets the build's process count).
CPU only unless --gpus is given.  Writes only under --out (and --tmp)."""
import argparse, hashlib, json, os, resource, shutil, sys, time
from pathlib import Path

META_COLS = ["series_idx", "slot", "plane", "view_id", "rel_pos", "log_spacing", "extent_130", "fluid", "fatsat", "slice_order",
             "n_slices", "real_series_idx"]


def worker(a, wi, uids):
    if a.gpus:
        g = a.gpus.split(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = g[wi % len(g)]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for p in (a.src, a.code):
        sys.path.insert(0, p)
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    import compact_infer as CI
    S = a.size
    part = Path(a.out) / ("part%03d" % wi)
    part.mkdir(parents=True, exist_ok=True)
    jp = part / "part.json"
    st = json.load(open(jp)) if jp.exists() else {"done": {}, "blocks": {}, "errors": {}, "rows": 0, "anchor": {}, "sides": {}, "sec": {},
                                                 "rss_mb": {}}
    fpx = open(part / "px.u8", "ab")
    fm = open(part / "meta.f32", "ab")
    fpx.truncate(st["rows"] * S * S)                     # a crash between the data write and the json write: drop the tail
    fm.truncate(st["rows"] * len(META_COLS) * 4)
    fpx.seek(0, 2)
    fm.seek(0, 2)
    ds = CI.CStudies(a.data, a.split, uids, str(Path(a.tmp) / ("w%03d" % wi)), a.code, S, eval_stride=1, views="v4", slices=3,
                     anchor_weights=a.anchor, phys_mm=3.5)
    ds._setup()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    n_new = 0
    for uid in uids:
        if uid in st["done"] or uid in st["errors"]:
            continue
        ts = time.time()
        try:
            item, sides = ds.rows(uid)
            if item.get("error"):
                raise ValueError(item["error"])
            px = np.ascontiguousarray(item["px"].numpy())
            meta = np.ascontiguousarray(item["meta"].numpy().astype(np.float32))
            if px.dtype != np.uint8 or px.shape[1:] != (S, S) or meta.shape != (len(px), len(META_COLS)):
                raise ValueError("unexpected px %s %s / meta %s" % (px.dtype, px.shape, meta.shape))
            cur = st["rows"]
            fpx.write(px.tobytes())
            fm.write(meta.tobytes())
            fpx.flush()
            fm.flush()
            blocks = []
            for b in item["blocks"]:
                b = dict(b)
                b["views"] = [[int(v), cur + int(off)] for v, off in b["views"]]      # rows local to this part
                blocks.append(b)
            st["done"][uid] = [cur, int(len(px)), len(blocks)]
            st["blocks"][uid] = blocks
            st["rows"] = cur + int(len(px))
            st["sides"][uid] = sorted(str(s) for s in sides)
        except Exception as exc:
            st["errors"][uid] = repr(exc)[:400]
        st["sec"][uid] = round(time.time() - ts, 2)
        st["rss_mb"][uid] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
        n_new += 1
        tmp = str(jp) + ".tmp"
        json.dump(st, open(tmp, "w"))
        os.replace(tmp, jp)
        if n_new % 10 == 0 or n_new == 1:
            el = time.time() - t0
            print("worker %d: %d new, %d done, %d errors, %.1f s/study, peak RSS %.0f MB" % (
                wi, n_new, len(st["done"]), len(st["errors"]), el / n_new, st["rss_mb"][uid]), flush=True)
    print("worker %d finished: %d done, %d errors" % (wi, len(st["done"]), len(st["errors"])), flush=True)


def merge(a):
    import numpy as np
    S = a.size
    out = Path(a.out)
    parts = sorted(p for p in out.glob("part*") if (p / "part.json").exists())
    sts = [json.load(open(p / "part.json")) for p in parts]
    N = sum(s["rows"] for s in sts)
    px = np.lib.format.open_memmap(out / "px.u8.npy", mode="w+", dtype=np.uint8, shape=(N, S, S))
    meta = np.lib.format.open_memmap(out / "meta.f32.npy", mode="w+", dtype=np.float32, shape=(N, len(META_COLS)))
    index = {"order": [], "n_upper": N, "done": {}, "blocks": {}, "cursor": N, "errors": {}, "removed_series": 0, "meta_cols": META_COLS,
             "config": {"size": S, "local_mm": 100.0, "views": "v4", "slices": 3, "phys_mm": 3.5, "eval_stride": 1,
                        "anchors": "SKM-TEA anatomy.pt via compact_infer 34f433fe CStudies.rows (knee_anchor_bank_v2)",
                        "side_model": None, "builder": "px_build_s6.py", "rows": "one per (series, view, slice), physical order"},
             "timing": {}}
    off = 0
    for p, s in zip(parts, sts):
        n = s["rows"]
        px[off:off + n] = np.fromfile(p / "px.u8", dtype=np.uint8, count=n * S * S).reshape(n, S, S)
        meta[off:off + n] = np.fromfile(p / "meta.f32", dtype=np.float32, count=n * len(META_COLS)).reshape(n, len(META_COLS))
        for u, (st_, k, nb) in s["done"].items():
            index["done"][u] = [st_ + off, k, nb]
            index["blocks"][u] = [dict(b, views=[[v, r + off] for v, r in b["views"]]) for b in s["blocks"][u]]
            index["order"].append(u)
        index["errors"].update(s["errors"])
        index["timing"].update({u: [s["sec"].get(u), s["rss_mb"].get(u)] for u in s.get("sec", {})})
        off += n
    px.flush()
    meta.flush()
    json.dump(index, open(out / "index.json", "w"))
    secs = [v[0] for v in index["timing"].values() if v[0] is not None]
    rss = [v[1] for v in index["timing"].values() if v[1] is not None]
    print("merged %d parts: %d studies, %d rows (%.1f GB), %d errors | s/study mean %.1f max %.1f | peak RSS max %.0f MB" % (
        len(parts), len(index["done"]), N, N * S * S / 1e9, len(index["errors"]),
        sum(secs) / max(1, len(secs)), max(secs or [0]), max(rss or [0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="competition root: train.csv, train_series.csv, train_series/<study>/<series>/*.dcm")
    ap.add_argument("--split", default="train")
    ap.add_argument("--code", help="dir holding the mainline 'training' package (cloud_code)")
    ap.add_argument("--src", help="dir holding compact_infer.py 34f433fe, compact_train.py 14767c7a, px_cache.py, ...")
    ap.add_argument("--anchor", help="anatomy.pt (sha256 5904dd16...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--gpus", default="", help="comma list of GPU ids for the anchors (empty = CPU)")
    ap.add_argument("--uids", default=None, help="file with one study uid per line (default: every study of <data>/<split>.csv)")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        return merge(a)
    for need, val in (("--data", a.data), ("--code", a.code), ("--src", a.src), ("--anchor", a.anchor)):
        if not val:
            raise SystemExit(need + " is required to build")
    a.tmp = a.tmp or str(Path(a.out) / "_tmp")
    if a.uids:
        uids = [l.strip() for l in open(a.uids) if l.strip()]
    else:
        import csv
        uids = sorted({r["StudyInstanceUID"] for r in csv.DictReader(open(os.path.join(a.data, a.split + ".csv"), encoding="utf-8"))})
    os.makedirs(a.out, exist_ok=True)
    json.dump({"argv": sys.argv, "n_uids": len(uids), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
              open(os.path.join(a.out, "build_args.json"), "w"), indent=1)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=worker, args=(a, i, uids[i::a.procs])) for i in range(a.procs)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print("all workers exited: %s" % [p.exitcode for p in ps], flush=True)
    shutil.rmtree(a.tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
