"""Pipeline-test inputs for compact_train.py from the 5 studies rendered by px_build_test.py --save-px:
  px cache (px_cache.py format) with R replicas of every study under synthetic UIDs (so 4 hash folds all get studies),
  targets CSV (E_lrprior arm columns from our llm_labels_v4_blend soft labels), cell-weight CSV, and SYNTHETIC state targets
  (random valid categories / binaries from the R7 state spec; pipeline test only, never for a real run)."""
import argparse, csv, hashlib, json, os, glob
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--test-px", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--spec", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--replicas", type=int, default=6)
a = ap.parse_args()
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
META_COLS = ["series_idx", "slot", "plane", "view_id", "rel_pos", "log_spacing", "extent_130", "fluid", "fatsat", "slice_order", "n_slices", "real_series_idx"]
res = json.load(open(os.path.join(a.test_px, "px_build_test.json")))
ok = [r for r in res["studies"] if r["ok"]]
lab = {r["StudyInstanceUID"]: r for r in csv.DictReader(open(a.labels, encoding="utf-8"))}
spec = json.load(open(a.spec))
px_dir = os.path.join(a.out, "px")
os.makedirs(px_dir, exist_ok=True)
pxs, metas, blocks = [], [], {}
cur = 0
src_rows = {}
for r in ok:
    key = hashlib.sha256(r["uid"].encode()).hexdigest()[:24]
    p = np.load(os.path.join(a.test_px, key + "_px.npy")); m = np.load(os.path.join(a.test_px, key + "_meta.npy"))
    b = json.load(open(os.path.join(a.test_px, key + "_blocks.json")))
    for bb in b:
        bb["views"] = [[v, cur + off] for v, off in bb["views"]]
    src_rows[r["uid"]] = (cur, len(p), b)
    pxs.append(p); metas.append(m); cur += len(p)
px = np.concatenate(pxs); meta = np.concatenate(metas).astype(np.float32)
np.save(os.path.join(px_dir, "px.u8.npy"), px); np.save(os.path.join(px_dir, "meta.f32.npy"), meta)
done, uids, src_of = {}, [], {}
for rep in range(a.replicas):
    for u, (st, n, b) in src_rows.items():
        fu = u if rep == 0 else "%s.%d" % (u, rep)
        done[fu] = [st, n, len(b)]; blocks[fu] = b; uids.append(fu); src_of[fu] = u
index = {"order": uids, "n_upper": int(len(px)), "done": done, "blocks": blocks, "cursor": int(len(px)), "errors": {}, "removed_series": 0,
         "meta_cols": META_COLS, "config": {"size": int(px.shape[-1]), "local_mm": 100.0, "policy": "series_companded", "views": "v4",
         "anchors": "rebuilt per study (CStudies path)", "duplicates": "collapse_exact", "rows": "one per (series, view, slice), physical order",
         "note": "pipeline test: replicas share rows"}}
json.dump(index, open(os.path.join(px_dir, "index.json"), "w"))
rng = np.random.default_rng(0)
base = {l: float(spec["base"][l]) for l in LABELS}
gold_u = uids[-1]                                     # one synthetic gold study (blank targets): exercises the gold path
with open(os.path.join(a.out, "targets.csv"), "w", newline="", encoding="utf-8") as fh, \
     open(os.path.join(a.out, "cell_weights.csv"), "w", newline="", encoding="utf-8") as fw:
    cols = ["StudyInstanceUID", "is_gold", "review_holdout", "fold"] + [l + "|" + s for l in LABELS for s in ("prior", "lr_prior", "strict")]
    w = csv.DictWriter(fh, cols); w.writeheader()
    ww = csv.DictWriter(fw, ["StudyInstanceUID"] + [l + "|w" for l in LABELS]); ww.writeheader()
    for u in uids:
        g = u == gold_u
        row = {"StudyInstanceUID": u, "is_gold": int(g), "review_holdout": 0, "fold": 0}
        wr = {"StudyInstanceUID": u}
        for l in LABELS:
            q = min(0.98, max(0.02, float(lab[src_of[u]][l])))
            lr = (q / (1 - q)) / (base[l] / (1 - base[l]))
            row[l + "|prior"] = "" if g else "%.6f" % q
            row[l + "|lr_prior"] = "" if g else "%.6f" % lr
            row[l + "|strict"] = "" if g else "%d" % (q >= 0.5)
            wr[l + "|w"] = "1.0"
        w.writerow(row); ww.writerow(wr)
bins = [(b, l) for b, labs in spec["binaries"].items() for l in labs]
with open(os.path.join(a.out, "state_targets_SYNTHETIC.csv"), "w", newline="", encoding="utf-8") as fh:
    cols = ["StudyInstanceUID"] + [l + "|state" for l in LABELS] + [l + "|sw" for l in LABELS] + [l + "|" + b for b, l in bins]
    w = csv.DictWriter(fh, cols); w.writeheader()
    for u in uids:
        row = {"StudyInstanceUID": u}
        for l in LABELS:
            row[l + "|state"] = int(rng.integers(0, len(spec["categories"][l])))
            row[l + "|sw"] = float(rng.choice([1.0, 0.7, 0.4]))
        for b, l in bins:
            row[l + "|" + b] = int(rng.integers(0, 2))
        w.writerow(row)
print("studies", len(uids), "rows", len(px), "px GB %.2f" % (px.nbytes / 2**30), "gold", gold_u)
