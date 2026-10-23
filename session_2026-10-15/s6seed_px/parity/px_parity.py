"""PX_PARITY, train side (10-15, BOX_PLAN_1015 S4): the px_build_s6 cache read the way the TRAINER reads it at evaluation
(compact_train 14767c7a: Studies(px, mode 'eval', eval_stride 1, phys_mm 3.5, slices 3) -> collate -> centre crop 288 -> Net, the
same steps as compact_train.predict), compared with the SERVE side = compact_infer 34f433fe main() end to end on the raw DICOMs
(ci_cpu_launcher.py, --dump-logits + K1_PX_DUMP window hashes), same model file, fp32 on CPU.
Checks per study: window count, uint8 window bytes (sha256), meta6 bytes (sha256), then |logit(train side) - logit(serve side)|.
PASS = every study scored on both sides, bytes equal, max |dlogit| <= --tol (1e-3)."""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np
import torch
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--px", required=True)
ap.add_argument("--models", required=True)
ap.add_argument("--arm", default="E_prior")
ap.add_argument("--backbone", required=True)
ap.add_argument("--crop", type=int, default=288)
ap.add_argument("--slices", type=int, default=3)
ap.add_argument("--phys-mm", type=float, default=3.5)
ap.add_argument("--eval-stride", type=int, default=1)
ap.add_argument("--ref", default=[], nargs="*", help="compact_infer output dirs (each: logits.json, predictions.csv, receipt.json, pxdump/)")
ap.add_argument("--train-cache", default=None, help="json of train-side results: written when absent (with no --ref: train side only), "
                "read when present (compare only, no forward)")
ap.add_argument("--uids", default=None, help="restrict to these uids (one per line or csv with StudyInstanceUID)")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--tol", type=float, default=1e-3)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)
sys.path.insert(0, os.path.abspath(a.src))
import compact_train as CT
assert os.path.dirname(os.path.abspath(CT.__file__)) == os.path.abspath(a.src)
src_sha = {n: hashlib.sha256(open(os.path.join(a.src, n), "rb").read()).hexdigest()[:8] for n in ("compact_train.py", "compact_infer.py", "px_cache.py")}

# serve side
ref_z, ref_p, ref_px, receipts = {}, {}, {}, []
for d in a.ref:
    ref_z.update(json.load(open(os.path.join(d, "logits.json"))))
    import csv
    for r in csv.DictReader(open(os.path.join(d, "predictions.csv"), encoding="utf-8")):
        ref_p[r["StudyInstanceUID"]] = [float(r[L]) for L in CT.LABELS]
    for f in glob.glob(os.path.join(d, "pxdump", "px_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            ref_px[r["uid"]] = r
    receipts.append(json.load(open(os.path.join(d, "receipt.json"))))
fallback = {u: e for rc in receipts for u, e in rc.get("fallback", {}).items()}

# train side
index = json.load(open(os.path.join(a.px, "index.json")))
uids = sorted(index["done"])
if a.uids:
    txt = open(a.uids).read().splitlines()
    want = [r.split(",")[0] for r in txt[1:]] if txt and txt[0].startswith("StudyInstanceUID") else [l.strip() for l in txt if l.strip()]
    uids = sorted(set(want))
fs = sorted(glob.glob(os.path.join(a.models, a.arm + "_s*_f*.pt")))
assert fs, "no model file"
models = []
for p in fs:
    sd = torch.load(p, map_location="cpu", weights_only=True)
    spatial = any(k == "sp_q" or k.endswith(".sp_q") for k in sd)
    m = CT.Net(a.backbone, None, freeze_blocks=16, crop=a.crop, state_spec=None, in_chans=a.slices, spatial=spatial)
    m.load_state_dict(sd, strict=True)
    models.append(m.eval().requires_grad_(False))
present = [u for u in uids if u in index["done"]]
ds = CT.Studies(a.px, present, 64, "eval", 0, eval_stride=a.eval_stride, train_sampling="uniform", cover_stride=3, px_extra=None,
                slices=a.slices, slice_jitter=0.0, phys_mm=a.phys_mm, px_override=None, extent_meta=False)
cache = json.load(open(a.train_cache)) if a.train_cache and os.path.exists(a.train_cache) else None
new_cache = {}
rows, worst = [], 0.0
t0 = time.time()
for i, u in enumerate(present):
    if cache is not None:
        c = cache[u]
        img = None
        rec = {"uid": u, "n": c["n"], "img_sha256": c["img_sha256"], "meta_sha256": c["meta_sha256"]}
        zs = torch.tensor(c["zs"], dtype=torch.float32)
    else:
        img, meta, _ = ds[i]
        rec = {"uid": u, "n": int(len(img)), "img_sha256": hashlib.sha256(img.numpy().tobytes()).hexdigest(),
               "meta_sha256": hashlib.sha256(meta.numpy().astype(np.float32).tobytes()).hexdigest()}
        img_b, meta_b, mask_b, _ = CT.collate([(img, meta, 0)])
        S = img_b.shape[-1]
        off = (S - a.crop) // 2
        with torch.no_grad():
            zs = torch.stack([m(img_b[..., off:off + a.crop, off:off + a.crop], meta_b, mask_b, chunk=CT.PREDICT_CHUNK)[0].float()
                              for m in models])
        new_cache[u] = dict(rec, zs=zs.tolist(), win_sha1_10=[hashlib.sha1(w.numpy().tobytes()).hexdigest()[:10] for w in img])
    r = ref_px.get(u)
    rec["ref_n"] = None if r is None else r.get("n")
    rec["bytes_equal"] = bool(r is not None and r.get("n") == rec["n"] and r.get("img_sha256") == rec["img_sha256"]
                              and r.get("meta_sha256") == rec["meta_sha256"])
    if r is not None and not rec["bytes_equal"] and r.get("win_sha1_10") and r.get("n") == rec["n"]:
        mine = (new_cache.get(u) or cache[u])["win_sha1_10"]
        rec["windows_differing"] = int(sum(x != y for x, y in zip(mine, r["win_sha1_10"])))
    z = zs.mean(0)
    zr = ref_z.get(u)
    if zr is not None:
        zr_ = torch.tensor(zr, dtype=torch.float32)
        d = float((zs - zr_).abs().max())
        rec["max_dlogit"] = d
        rec["max_dprob_csv"] = float(np.max(np.abs(torch.sigmoid(z).numpy() - np.array(ref_p[u])))) if u in ref_p else None
        worst = max(worst, d)
    else:
        rec["max_dlogit"] = None
        rec["ref_fallback"] = fallback.get(u)
    rec["z_train_side"] = [round(float(x), 6) for x in z]
    rows.append(rec)
    print(json.dumps({k: rec[k] for k in ("uid", "n", "ref_n", "bytes_equal", "max_dlogit")}), flush=True)
if a.train_cache and cache is None:
    json.dump(new_cache, open(a.train_cache, "w"))
    print("train-side cache written: %s (%d studies, %.0f s)" % (a.train_cache, len(new_cache), time.time() - t0), flush=True)
missing_px = [u for u in uids if u not in index["done"]]
n_scored = sum(r["max_dlogit"] is not None for r in rows)
n_bytes = sum(r["bytes_equal"] for r in rows)
ok = (not missing_px and n_scored == len(uids) and n_bytes == len(uids) and worst <= a.tol)
summ = {"verdict": "PASS" if ok else "FAIL", "studies": len(uids), "px_missing": len(missing_px), "scored_both_sides": n_scored,
        "bytes_equal": n_bytes, "max_abs_dlogit": worst, "tol": a.tol,
        "models": [os.path.basename(p) for p in fs], "model_sha256": [hashlib.sha256(open(p, "rb").read()).hexdigest() for p in fs],
        "backbone": a.backbone, "geometry": {"crop": a.crop, "slices": a.slices, "phys_mm": a.phys_mm, "eval_stride": a.eval_stride},
        "src_sha8": src_sha, "serve_receipt_sha": [rc.get("sha256") for rc in receipts][:1], "serve_amp": sorted({rc.get("amp") for rc in receipts}),
        "serve_fallback": fallback, "windows_total": int(sum(r["n"] for r in rows)), "train_side_seconds": round(time.time() - t0, 1)}
json.dump({"summary": summ, "studies": rows, "px_missing": missing_px}, open(a.out, "w"), indent=1)
print("PX_PARITY %s %d/%d bytes-equal, %d/%d scored both sides, max |dlogit| %.3g (tol %.0e), backbone %s" % (
    summ["verdict"], n_bytes, len(uids), n_scored, len(uids), worst, a.tol, a.backbone), flush=True)
