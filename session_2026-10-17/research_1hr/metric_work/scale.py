# scale.py: paired-delta bootstrap on gold58 for the comparisons that decide picks/probes, scaled to public 390 / private 910
# under two prevalence scenarios (gold58 enriched vs train-report gpt56sol), plus Hanley SE per label and sign-agreement math.
import sys, json, numpy as np, pandas as pd
from scipy.stats import rankdata, norm
D = sys.argv[1]; B = int(sys.argv[2])
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
TABLE = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35}
g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int)
r7 = pd.read_csv(D + "/R7OOF.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
R7 = r7.loc[ids, ["p|" + l for l in L]].values
RA = np.load(D + "/gold58_N_crop320_k24.npy")
S6 = pd.read_csv(D + "/nartaa_gold58.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
n = len(ids)
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
def final(idx, wr_default=0.5, w_s6=0.2, table=TABLE):
    out = np.zeros((len(idx), 12))
    for j, c in enumerate(L):
        wr = table.get(c, wr_default)
        st3 = wr * pct(RA[idx, j]) + (1 - wr) * pct(R7[idx, j])
        out[:, j] = (1 - w_s6) * pct(st3) + w_s6 * pct(S6[idx, j])
    return out
pl = lambda P, idx: np.array([auc(P[:, j], Y[idx, j]) for j in range(12)])
CFG = {"HW": dict(), "clone(heliosli,noS6)": dict(w_s6=0.0), "R6pick(uniform+S6)": dict(table={}),
       "wr45": dict(wr_default=0.45), "s615": dict(w_s6=0.15), "wr55": dict(wr_default=0.55), "s625": dict(w_s6=0.25)}
PAIRS = [("HW", "clone(heliosli,noS6)"), ("R6pick(uniform+S6)", "clone(heliosli,noS6)"), ("HW", "R6pick(uniform+S6)"),
         ("wr45", "HW"), ("s615", "HW"), ("wr55", "HW"), ("s625", "HW")]
rng = np.random.default_rng(1); D_ = {p: [] for p in PAIRS}
for b in range(B):
    idx = rng.integers(0, n, n)
    if (Y[idx].sum(0) < 2).any() or (Y[idx].sum(0) > n - 2).any():
        continue
    v = {k: pl(final(idx, **kw), idx) for k, kw in CFG.items()}
    for a, c in PAIRS:
        D_[(a, c)].append(v[a] - v[c])
pt = {k: pl(final(np.arange(n), **kw), np.arange(n)) for k, kw in CFG.items()}
pos_g = Y.sum(0); neg_g = n - pos_g
PREV = {"gold58": pos_g / n, "report": np.array([.204,.147,.394,.151,.359,.257,.450,.593,.126,.247,.168,.068])}
def factor(prev, N):   # per-label variance factor vs gold58 for a paired AUC difference (Hanley-style 1/P + 1/Q)
    P = prev * N; Q = N - P
    return (1 / P + 1 / Q) / (1 / pos_g + 1 / neg_g)
res = {"n_boot": len(D_[PAIRS[0]])}
print("n_boot", res["n_boot"])
for (a, c), d in D_.items():
    d = np.array(d); C = np.cov(d.T); m = (pt[a] - pt[c]).mean()
    row = {"gold58_point": float(m), "sd_n58": float(d.mean(1).std())}
    for sc, prev in PREV.items():
        for nm, N in (("pub", 390), ("priv", 910)):
            f = np.sqrt(factor(prev, N)); Cs = C * np.outer(f, f)
            row[f"sd_{nm}_{sc}"] = float(np.sqrt(Cs.sum()) / 12)
    # top labels by variance contribution (gold58 scale)
    contrib = np.diag(C) / np.diag(C).sum() if np.diag(C).sum() > 0 else np.zeros(12)
    row["top_var_labels"] = [L[i] + " %.0f%%" % (100 * contrib[i]) for i in np.argsort(-contrib)[:4]]
    row["per_label_point"] = dict(zip(L, (pt[a] - pt[c]).round(4).tolist()))
    res[f"{a} - {c}"] = row
    print(f"{a:20s} - {c:22s} gold58 {m:+.4f} sd58 {row['sd_n58']:.4f} | pub sd g/r {row['sd_pub_gold58']:.5f}/{row['sd_pub_report']:.5f}"
          f" | priv sd g/r {row['sd_priv_gold58']:.5f}/{row['sd_priv_report']:.5f} | top {row['top_var_labels']}")
# Hanley-McNeil SE of a single label AUC at A on N studies, both prevalence scenarios
def hanley(A, P, Q):
    q1 = A / (2 - A); q2 = 2 * A * A / (1 + A)
    return np.sqrt((A * (1 - A) + (P - 1) * (q1 - A * A) + (Q - 1) * (q2 - A * A)) / (P * Q))
A_lab = pt["HW"]  # gold58 HW-like per-label AUC as the level
res["hanley"] = {}
for sc, prev in PREV.items():
    for nm, N in (("pub", 390), ("priv", 910)):
        P = prev * N; Q = N - P; se = hanley(A_lab, P, Q)
        res["hanley"][f"{nm}_{sc}"] = {"pos": dict(zip(L, P.round(0).tolist())), "se": dict(zip(L, se.round(4).tolist())),
                                       "macro_se_indep": float(np.sqrt((se ** 2).sum()) / 12)}
        print(f"Hanley {nm} {sc}: macro SE (indep labels) {np.sqrt((se**2).sum())/12:.4f} | per-label SE",
              " ".join(f"{l[:6]}:{s:.3f}" for l, s in zip(L, se)))
json.dump(res, open(sys.argv[3], "w"), indent=1)
