"""Box CPU test: raw DICOM -> px384_v4 rows for a few studies through the R7 inference / sidefix path
(compact_infer.CStudies: lean native render -> geometry bank -> image side check (side_cls.pt) -> SKM-TEA anchors
(anatomy.pt) -> px_cache.StudySlices.build, views v4, 384 px).  Writes only under --out; no GPU (caller hides it)."""
import argparse, hashlib, json, os, shutil, sys, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--code", required=True, help="dir holding the mainline 'training' package")
ap.add_argument("--src", required=True, help="dir holding compact_infer.py / px_cache.py / ...")
ap.add_argument("--anchor", required=True)
ap.add_argument("--side", required=True)
ap.add_argument("--uids", required=True, help="comma list")
ap.add_argument("--tmp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--save-px", action="store_true")
a = ap.parse_args()
for p in (a.src, a.code):
    sys.path.insert(0, p)
import numpy as np, torch
torch.set_num_threads(a.threads)
import compact_infer as CI
uids = [u for u in a.uids.split(",") if u]
os.makedirs(a.out, exist_ok=True)
ds = CI.CStudies(a.data, "train", uids, a.tmp, a.code, 384, eval_stride=1, views="v4", slices=3, anchor_weights=a.anchor,
                 phys_mm=3.5, side_model=a.side, side_threshold=2.0, side_max_x=0.0, side_position_mm=20.0)
t_setup = time.perf_counter()
ds._setup()
torch.set_num_threads(a.threads)
t_setup = time.perf_counter() - t_setup
res = []
for uid in uids:
    key = hashlib.sha256(uid.encode()).hexdigest()[:24]
    work = Path(a.tmp) / ("t_" + key[:16])
    sdir = Path(a.data) / "train_series" / uid
    dcm = list(sdir.glob("*/*.dcm"))
    dbytes = sum(p.stat().st_size for p in dcm)
    t0 = time.perf_counter()
    try:
        note = ds._native_and_geometry(uid, work)
        t1 = time.perf_counter()
        ds.builder.cache = str(work)
        study = json.load(open(work / "studies" / (key + ".json"), encoding="utf-8"))
        rec = ds.bank.record(uid, study, geo_dir=str(work / "geometry"))
        (work / "anchors").mkdir(exist_ok=True)
        json.dump(rec, open(work / "anchors" / (ds.KB.study_key(uid) + ".json"), "w"))
        ds.builder.anchors = str(work / "anchors")
        t2 = time.perf_counter()
        item = ds.builder.build(uid)
        t3 = time.perf_counter()
        px = item["px"].numpy()
        nb = sum(int(np.prod(np.load(s["path"], mmap_mode="r").shape)) * 2 for s in study["series"])
        r = {"uid": uid, "ok": True, "n_dicom": len(dcm), "dicom_MB": round(dbytes / 2**20, 1), "n_series": len(study["series"]),
             "native_MB": round(nb / 2**20, 1), "rows": int(len(px)), "px_MB": round(px.nbytes / 2**20, 1),
             "blocks": [[b["plane"], b["n_slices"], [v for v, _ in b["views"]]] for b in item["blocks"]],
             "removed_dup_series": item.get("removed"), "side": note, "anchor": {"method": rec.get("method"), "why": rec.get("why", [])[:3]},
             "sec_native_geom_side": round(t1 - t0, 2), "sec_anchor": round(t2 - t1, 2), "sec_render": round(t3 - t2, 2),
             "sec_total": round(t3 - t0, 2)}
        if a.save_px:
            np.save(os.path.join(a.out, key + "_px.npy"), px)
            np.save(os.path.join(a.out, key + "_meta.npy"), item["meta"].numpy())
            json.dump(item["blocks"], open(os.path.join(a.out, key + "_blocks.json"), "w"))
    except Exception as exc:
        import traceback
        r = {"uid": uid, "ok": False, "error": repr(exc)[:500], "trace": traceback.format_exc()[-1500:], "sec_total": round(time.perf_counter() - t0, 2)}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    res.append(r)
    print(json.dumps(r), flush=True)
ok = [r for r in res if r["ok"]]
summ = {"setup_s": round(t_setup, 1), "n_ok": len(ok), "n": len(res), "threads": a.threads}
if ok:
    for k in ("sec_total", "sec_native_geom_side", "sec_anchor", "sec_render", "rows", "px_MB", "dicom_MB", "n_series"):
        summ["mean_" + k] = round(float(np.mean([r[k] for r in ok])), 2)
print("SUMMARY " + json.dumps(summ), flush=True)
json.dump({"summary": summ, "studies": res}, open(os.path.join(a.out, "px_build_test.json"), "w"), indent=1)
