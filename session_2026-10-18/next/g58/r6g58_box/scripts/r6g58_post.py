"""Post-process the box R6 gold58 serve (10-18, Claude): worker predictions.csv -> g58/R6_gold58_serve.csv (StudyInstanceUID + 12
labels, R7OOF.csv p| order, gold58 row order) + receipt + per-label gold58 AUC of R6 vs R7 OOF (and S6fall n42gpu for context).
  python r6g58_post.py <fetched dir with worker0/ and r6g58_receipt.json> <g58 dir>"""
import sys, json, hashlib, csv
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr

F, D = sys.argv[1], sys.argv[2]
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
assert n == 58
r7cols = [c for c in pd.read_csv(D + "/R7OOF.csv", nrows=0).columns if c.startswith("p|")]
assert r7cols == ["p|" + l for l in L], r7cols
P = pd.read_csv(F + "/worker0/predictions.csv", dtype={"StudyInstanceUID": str})
assert list(P.columns[:13]) == ["StudyInstanceUID"] + L, list(P.columns)
assert P.StudyInstanceUID.is_unique and set(P.StudyInstanceUID) == set(ids), (len(P), len(set(P.StudyInstanceUID) & set(ids)))
P = P.set_index("StudyInstanceUID").loc[ids, L]
V = P.values.astype(float)
assert np.isfinite(V).all() and (V >= 0).all() and (V <= 1).all()
# raw strings of the worker file (keep its %.9g text exactly)
with open(F + "/worker0/predictions.csv", encoding="utf-8", newline="") as fh:
    rd = csv.reader(fh); hdr = next(rd); raw = {r[0]: r for r in rd}
out = D + "/R6_gold58_serve.csv"
with open(out, "w", encoding="utf-8", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["StudyInstanceUID"] + L)
    for u in ids:
        r = raw[u]
        wr.writerow([u] + [r[hdr.index(l)] for l in L])
chk = pd.read_csv(out, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
assert np.array_equal(chk, V)

R7 = pd.read_csv(D + "/R7OOF.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, r7cols].values
S6 = pd.read_csv(D + "/n42gpu_gold58.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


rows = []
for j, c in enumerate(L):
    rows.append({"label": c, "pos": int(Y[:, j].sum()), "R6": round(auc(V[:, j], Y[:, j]), 4), "R7oof": round(auc(R7[:, j], Y[:, j]), 4),
                 "S6fall": round(auc(S6[:, j], Y[:, j]), 4), "rho_R6_R7": round(float(spearmanr(V[:, j], R7[:, j])[0]), 3)})
T = pd.DataFrame(rows)
mac = {k: round(float(T[k].mean()), 4) for k in ("R6", "R7oof", "S6fall")}
# paired bootstrap of macro AUC R6 - R7 (studies resampled, predictions fixed)
rng = np.random.default_rng(1018)
d = []
for _ in range(2000):
    idx = rng.integers(0, n, n)
    if any(Y[idx, j].min() == Y[idx, j].max() for j in range(12)):
        continue
    d.append(np.mean([auc(V[idx, j], Y[idx, j]) - auc(R7[idx, j], Y[idx, j]) for j in range(12)]))
d = np.array(d)
boot = {"n": int(len(d)), "mean": round(float(d.mean()), 4), "q05": round(float(np.quantile(d, .05)), 4),
        "q95": round(float(np.quantile(d, .95)), 4), "P(R6>R7)": round(float((d > 0).mean()), 3)}
print(T.to_string(index=False))
print("MACRO", json.dumps(mac), "R6-R7 boot", json.dumps(boot))
box = json.load(open(F + "/r6g58_receipt.json", encoding="utf-8"))
wrec = json.load(open(F + "/worker0/receipt.json", encoding="utf-8"))
rec = {"what": "R6 bundle fe055406 served on gold58 (58 train studies, gold_in_training 0) exactly as the c4 kernel cells 7-9 "
               "(compact_infer 8401812c, 4 folds, flip TTA forced, kernel WORKER_ARGS exec'd from cells 7+8), box V100 fp16",
       "csv": out.replace("\\", "/"), "csv_sha256": hashlib.sha256(open(out, "rb").read()).hexdigest(),
       "rows": n, "label_order": L, "auc_gold58": rows, "macro": mac, "boot_macro_R6_minus_R7": boot,
       "box_receipt": box, "worker_receipt": wrec}
json.dump(rec, open(D + "/R6_gold58_serve_receipt.json", "w", encoding="utf-8"), indent=1)
print("WROTE", out, rec["csv_sha256"][:16])
