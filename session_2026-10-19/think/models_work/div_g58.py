# div_g58.py (10-19 think lens "other model types"): read-only numpy on existing gold58 prediction files.
# Question: which model FAMILY adds rank diversity to c4 on gold58 (the only expert-labelled set we hold)?
# Members: c4 legs (r384, r224, R6, S6fall), our own (s43, R7oof, OAI-free ConvNeXt fleet mean, RadImageNet crop
# fine-tunes), and Dread's public competition-only Raptor arms (maxspan v5, native384 v8, finespacing v9, parent proxy).
# CAVEAT: gold58 cannot tune weights (A19 lesson); it is used here only to rank FAMILIES by diversity, never to pick w.
import sys, json, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
G, W = sys.argv[1], sys.argv[2]  # G = session_2026-10-18/next/g58, W = this dir
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
g = pd.read_csv(G + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
def rd(path, cols=L):
    d = pd.read_csv(path, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    miss = [i for i in ids if i not in d.index]
    if miss: raise SystemExit("%s missing %d gold58 ids" % (path, len(miss)))
    return d.loc[ids, cols].values.astype(float)
M = {}
M["r384"] = np.load(G + "/gold58_r384_kernelserve_cpu.npy"); M["r224"] = np.load(G + "/gold58_r224_kernelserve.npy")
M["R6"] = rd(G + "/R6_gold58_serve.csv"); M["S6fall"] = rd(G + "/n42gpu_gold58.csv"); M["s43"] = rd(G + "/s43_gold58.csv")
M["R7oof"] = rd(G + "/R7OOF.csv", ["p|" + l for l in L])
for k in ["T_maxspan_v5", "T_native384_v8", "T_finespacing_v9", "T_raptor_proxy"]: M[k[2:]] = rd(W + "/" + k + ".csv")
M["fleetOF25"] = rd(W + "/gold_fleetOF_mean.csv")
z = np.load(W + "/proxy_crop_gold74_preds.npz", allow_pickle=True); zi = {s: i for i, s in enumerate(z["ids"])}
assert list(z["labels"]) == L, z["labels"]
for k in ["finetune_crop_f1_radimagenet", "finetune_cropaug_f1_radimagenet", "finetune_cropaug_mixSA_f1"]:
    M[k.replace("finetune_", "")] = z[k][[zi[i] for i in ids]].astype(float)
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    p = y.sum(); q = len(y) - p; r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
mac = lambda P: np.array([auc(P[:, j], Y[:, j]) for j in range(12)])
def c4(extra=None, wx=0.0):
    out = np.zeros((n, 12))
    for j, c in enumerate(L):
        rr = .75 * pct(M["r384"][:, j]) + .25 * pct(M["r224"][:, j]); w = C4.get(c, .5)
        b = w * rr + (1 - w) * pct(M["R6"][:, j]); f = .8 * pct(b) + .2 * pct(M["S6fall"][:, j])
        if extra is not None: f = (1 - wx) * pct(f) + wx * pct(extra[:, j])
        out[:, j] = f
    return out
rng = np.random.default_rng(20261019); Wm = []
while len(Wm) < 2000:
    w = np.bincount(rng.integers(0, n, n), minlength=n)
    if all(w[Y[:, j] == 1].sum() > 0 and w[Y[:, j] == 0].sum() > 0 for j in range(12)): Wm.append(w)
Wm = np.array(Wm, float)
def boot(P):
    A = np.zeros((len(Wm), 12))
    for j in range(12):
        pos, neg = Y[:, j] == 1, Y[:, j] == 0
        Cm = (P[pos, j][:, None] > P[neg, j][None, :]) + .5 * (P[pos, j][:, None] == P[neg, j][None, :])
        wp, wn = Wm[:, pos], Wm[:, neg]; A[:, j] = np.einsum('bi,ij,bj->b', wp, Cm, wn) / (wp.sum(1) * wn.sum(1))
    return A.mean(1)
base = c4(); bm = mac(base); bb = boot(base)
rho = lambda a, b: float(np.mean([spearmanr(a[:, j], b[:, j])[0] for j in range(12)]))
# residual diversity: Spearman of X with the c4 output's rank residual is not defined for ranks; use 1 - rho(X, c4)
res = {"c4_macro": round(float(bm.mean()), 4), "members": {}}
print("c4 gold58 macro %.4f" % bm.mean())
print("%-26s %6s | %5s %5s %5s %5s %5s | %8s %5s | %8s %5s" % ("member", "AUC", "c4", "R6", "r384", "S6f", "R7", "d@.10", "P>0", "d@.20", "P>0"))
for k, X in M.items():
    a = mac(X); r = {t: rho(X, M[t]) for t in ["R6", "r384", "S6fall", "R7oof"]}; rc = rho(X, base)
    row = dict(auc=round(float(a.mean()), 4), per_label=dict(zip(L, np.round(a, 3).tolist())), rho_c4=round(rc, 3), rho={t: round(v, 3) for t, v in r.items()})
    for wx in (.1, .2):
        P = c4(X, wx); d = boot(P) - bb; row["d%.2f" % wx] = round(float(mac(P).mean() - bm.mean()), 5); row["P%.2f" % wx] = round(float((d > 0).mean()), 3)
    res["members"][k] = row
    print("%-26s %.4f | %.3f %.3f %.3f %.3f %.3f | %+.5f %.2f | %+.5f %.2f" % (k, row["auc"], rc, r["R6"], r["r384"], r["S6fall"], r["R7oof"], row["d0.10"], row["P0.10"], row["d0.20"], row["P0.20"]))
json.dump(res, open(W + "/div_g58.json", "w"), indent=1)
