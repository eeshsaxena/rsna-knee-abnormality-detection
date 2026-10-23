"""Build the px384_v4 training cache (px_cache.py format, R7 views v4, sidefix applied in the same pass) from raw DICOMs,
through compact_infer.CStudies = heliosli's R7 inference path and his own sidefix-cache builder:
  lean native render -> geometry bank -> image side check (side_cls.pt) -> SKM-TEA knee anchors (anatomy.pt) -> px_cache.StudySlices.build
Parallel: --procs worker processes, each renders a disjoint slice of the study list into its own part (raw px / meta files + json);
--merge concatenates the parts into <out>/px.u8.npy, meta.f32.npy, index.json (the trainer's --px dir).
Resumable: a part's json lists the studies it finished; a restarted worker skips them.
Example (rental, 4 GPUs for anchors):
  python px_build_all.py --data /data/rsna --code code --src src --anchor w/anatomy.pt --side w/side_cls.pt --out /nvme/px384_v4 --procs 16 --gpus 0,1,2,3
  python px_build_all.py --out /nvme/px384_v4 --merge"""
import argparse, hashlib, json, os, shutil, sys, time
from pathlib import Path

META_COLS = ["series_idx", "slot", "plane", "view_id", "rel_pos", "log_spacing", "extent_130", "fluid", "fatsat", "slice_order",
             "n_slices", "real_series_idx"]
S = 384


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
    part = Path(a.out) / ("part%03d" % wi)
    part.mkdir(parents=True, exist_ok=True)
    jp = part / "part.json"
    st = json.load(open(jp)) if jp.exists() else {"done": {}, "blocks": {}, "errors": {}, "rows": 0, "side": {}, "anchor": {}}
    fpx = open(part / "px.u8", "ab")
    fm = open(part / "meta.f32", "ab")
    # a crash between the data write and the json write leaves extra bytes: truncate back to the json's row count
    fpx.truncate(st["rows"] * S * S)
    fm.truncate(st["rows"] * len(META_COLS) * 4)
    fpx.seek(0, 2)
    fm.seek(0, 2)
    ds = CI.CStudies(a.data, a.split, uids, str(Path(a.tmp) / ("w%03d" % wi)), a.code, S, eval_stride=1, views="v4", slices=3,
                     anchor_weights=a.anchor, phys_mm=3.5, side_model=a.side, side_threshold=2.0, side_max_x=0.0, side_position_mm=20.0)
    ds._setup()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    n_new = 0
    for uid in uids:
        if uid in st["done"] or uid in st["errors"]:
            continue
        key = hashlib.sha256(uid.encode()).hexdigest()[:24]
        work = Path(a.tmp) / ("w%03d" % wi) / ("b_" + key[:16])
        try:
            note = ds._native_and_geometry(uid, work)
            ds.builder.cache = str(work)
            study = json.load(open(work / "studies" / (key + ".json"), encoding="utf-8"))
            rec = ds.bank.record(uid, study, geo_dir=str(work / "geometry"))
            (work / "anchors").mkdir(exist_ok=True)
            json.dump(rec, open(work / "anchors" / (ds.KB.study_key(uid) + ".json"), "w"))
            ds.builder.anchors = str(work / "anchors")
            item = ds.builder.build(uid)
            if item.get("error"):
                raise ValueError(item["error"])
            px = item["px"].numpy()
            meta = item["meta"].numpy().astype(np.float32)
            cur = st["rows"]
            fpx.write(np.ascontiguousarray(px).tobytes())
            fm.write(np.ascontiguousarray(meta).tobytes())
            fpx.flush()
            fm.flush()
            for b in item["blocks"]:
                b["views"] = [[v, cur + off] for v, off in b["views"]]          # rows local to this part
            st["done"][uid] = [cur, int(len(px)), len(item["blocks"])]
            st["blocks"][uid] = item["blocks"]
            st["rows"] = cur + int(len(px))
            st["side"][uid] = note
            st["anchor"][uid] = rec.get("method")
        except Exception as exc:
            st["errors"][uid] = repr(exc)[:400]
        finally:
            shutil.rmtree(work, ignore_errors=True)
        n_new += 1
        tmp = str(jp) + ".tmp"
        json.dump(st, open(tmp, "w"))
        os.replace(tmp, jp)
        if n_new % 10 == 0:
            el = time.time() - t0
            print("worker %d: %d new, %d done, %d errors, %.1f s/study" % (wi, n_new, len(st["done"]), len(st["errors"]), el / n_new), flush=True)
    print("worker %d finished: %d done, %d errors" % (wi, len(st["done"]), len(st["errors"])), flush=True)


def merge(a):
    import numpy as np
    out = Path(a.out)
    parts = sorted(p for p in out.glob("part*") if (p / "part.json").exists())
    sts = [json.load(open(p / "part.json")) for p in parts]
    N = sum(s["rows"] for s in sts)
    px = np.lib.format.open_memmap(out / "px.u8.npy", mode="w+", dtype=np.uint8, shape=(N, S, S))
    meta = np.lib.format.open_memmap(out / "meta.f32.npy", mode="w+", dtype=np.float32, shape=(N, len(META_COLS)))
    index = {"order": [], "n_upper": N, "done": {}, "blocks": {}, "cursor": N, "errors": {}, "removed_series": 0, "meta_cols": META_COLS,
             "config": {"size": S, "local_mm": 100.0, "policy": "series_companded", "views": "v4",
                        "anchors": "SKM-TEA anchors rebuilt per study (knee_anchor_bank_v2 via compact_infer.CStudies)",
                        "sidefix": "side_cls.pt image side check applied in the render", "view_recipe": "v4",
                        "duplicates": "collapse_exact", "rows": "one per (series, view, slice), physical order", "gold_first": 0}}
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
        off += n
    px.flush()
    meta.flush()
    json.dump(index, open(out / "index.json", "w"))
    print("merged %d parts: %d studies, %d rows (%.1f GB), %d errors" % (len(parts), len(index["done"]), N, N * S * S / 1e9,
                                                                         len(index["errors"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--code")
    ap.add_argument("--src")
    ap.add_argument("--anchor")
    ap.add_argument("--side")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--gpus", default="", help="comma list of GPU ids for anchors / side model (empty = CPU)")
    ap.add_argument("--uids", default=None, help="file with one study uid per line (default: every study of <data>/<split>.csv)")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        return merge(a)
    a.tmp = a.tmp or str(Path(a.out) / "_tmp")
    if a.uids:
        uids = [l.strip() for l in open(a.uids) if l.strip()]
    else:
        import pandas as pd
        uids = sorted(pd.read_csv(os.path.join(a.data, a.split + ".csv"), usecols=["StudyInstanceUID"], dtype=str)["StudyInstanceUID"].unique())
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=worker, args=(a, i, uids[i::a.procs])) for i in range(a.procs)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print("all workers exited:", [p.exitcode for p in ps])


if __name__ == "__main__":
    main()
