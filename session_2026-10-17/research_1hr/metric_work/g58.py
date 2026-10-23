# g58.py: per-label gold58 analysis of the HW blend structure with the components we hold on gold58.
# Components (gold58, 58 studies): R7 OOF (proxy for "ours" = V224 on the R6 bundle), Raptor crop320 (serve geometry, public
# checkpoint, selected on gold58 -> optimistic), nartaa S6fall (serve path). No r224 and no R6 gold preds exist.
import sys, json, numpy as np, pandas as pd
from scipy.stats import rankdata, norm
D = sys.argv[1]
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
TABLE = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35}
g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int)
r7 = pd.read_csv(D + "/R7OOF.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
R7 = r7.loc[ids, ["p|" + l for l in L]].values
RA = np.load(D + "/gold58_N_crop320_k24.npy"); RF = np.load(D + "/gold58_N_full384_k24.npy")
s6 = pd.read_csv(D + "/nartaa_gold58.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
S6 = s6.loc[ids, L].values
n = len(ids)
def pct(v): return rankdata(v) / len(v)
def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
def final(idx, wr_default=0.5, w_s6=0.2, table=TABLE):
    # ranks recomputed inside the subsample idx, as the kernel ranks over the scored set
    out = np.zeros((len(idx), 12))
    for j, c in enumerate(L):
        rr, ro = pct(RA[idx, j]), pct(R7[idx, j])
        wr = table.get(c, wr_default)
        st3 = wr * rr + (1 - wr) * ro
        out[:, j] = (1 - w_s6) * pct(st3) + w_s6 * pct(S6[idx, j])
    return out
def per_label(P, idx):
    return np.array([auc(P[:, j], Y[idx, j]) for j in range(12)])
allidx = np.arange(n)
res = {}
prev = Y.mean(0)
print("gold58 positives:", dict(zip(L, Y.sum(0).tolist())))
comp = {"R7(ours proxy)": R7, "Raptor crop320": RA, "Raptor full384": RF, "S6fall nartaa": S6}
tab = pd.DataFrame({k: per_label(v, allidx) for k, v in comp.items()}, index=L)
tab["HW-like blend"] = per_label(final(allidx), allidx)
tab["wr45"] = per_label(final(allidx, wr_default=0.45), allidx)
tab["s615"] = per_label(final(allidx, w_s6=0.15), allidx)
tab["uniform(R6-like)"] = per_label(final(allidx, table={}), allidx)
print(tab.round(4).to_string()); print("macro:", tab.mean().round(4).to_dict())
# weight curves (macro + per-label) on gold58
for name, kw in (("W_RAPTOR on 7 non-table", [dict(wr_default=w) for w in (0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7)]),
                 ("W_S6F", [dict(w_s6=w) for w in (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)])):
    print(name, [round(per_label(final(allidx, **k), allidx).mean(), 5) for k in kw])
# paired bootstrap of the probe deltas at n=58 (resample studies; ranks recomputed per resample)
rng = np.random.default_rng(0); B = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
d_wr, d_s6, d_un = [], [], []
for b in range(B):
    idx = rng.integers(0, n, n)
    if (Y[idx].sum(0) == 0).any() or (Y[idx].sum(0) == n).any():
        continue
    base = per_label(final(idx), idx)
    d_wr.append(per_label(final(idx, wr_default=0.45), idx) - base)
    d_s6.append(per_label(final(idx, w_s6=0.15), idx) - base)
    d_un.append(per_label(final(idx, table={}), idx) - base)
d_wr, d_s6, d_un = map(np.array, (d_wr, d_s6, d_un))
pt = {"wr45": tab["wr45"].values - tab["HW-like blend"].values, "s615": tab["s615"].values - tab["HW-like blend"].values,
      "uniform": tab["uniform(R6-like)"].values - tab["HW-like blend"].values}
out = {"n_boot": len(d_wr), "prev_gold58": dict(zip(L, prev.round(4).tolist())), "pos_gold58": dict(zip(L, Y.sum(0).tolist())), "table": tab.round(5).to_dict()}
for nm, d in (("wr45", d_wr), ("s615", d_s6), ("uniform", d_un)):
    m = d.mean(1)
    out[nm] = {"point_macro": float(pt[nm].mean()), "boot_sd_macro_n58": float(m.std()), "P_boot_gt0": float((m > 0).mean()),
               "P_boot_eq0": float((m == 0).mean()), "per_label_point": dict(zip(L, pt[nm].round(5).tolist())),
               "per_label_sd_n58": dict(zip(L, d.std(0).round(5).tolist()))}
    print(nm, "point %+.5f | boot sd(n58) %.5f | P>0 %.2f | P==0 %.2f" % (pt[nm].mean(), m.std(), (m > 0).mean(), (m == 0).mean()))
    print("   per-label point:", " ".join("%+.4f" % x for x in pt[nm]))
    print("   per-label sd   :", " ".join("%.4f" % x for x in d.std(0)))
# within-class rank correlations (latent error correlation proxy) between components, averaged over labels
def wc(a, b):
    rs = []
    for j in range(12):
        for cls in (0, 1):
            m = Y[:, j] == cls
            if m.sum() >= 8:
                rs.append(np.corrcoef(rankdata(a[m, j]), rankdata(b[m, j]))[0, 1])
    return float(np.mean(rs))
out["within_class_rho"] = {"R7-Raptor": wc(R7, RA), "R7-S6": wc(R7, S6), "Raptor-S6": wc(RA, S6), "Raptor320-384": wc(RA, RF)}
print("within-class rank corr:", {k: round(v, 3) for k, v in out["within_class_rho"].items()})
json.dump(out, open(sys.argv[3] if len(sys.argv) > 3 else "g58_out.json", "w"), indent=1)
