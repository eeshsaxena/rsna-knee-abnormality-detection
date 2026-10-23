# ours_g58.py (10-19 think lens "our unused assets"): read-only numpy on existing gold58 prediction files.
# c4 recipe: rr = .75 pct(r384) + .25 pct(r224); b = w_c rr + (1-w_c) pct(R6); final = .8 pct(b) + .2 pct(S6fall)
# Members we can test locally: s43 (our S6 seed), R7 OOF (proxy for "another ConvNeXt of the same family"; LB read R7 late = practically negative).
import sys, json, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
D = sys.argv[1]
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(D + "/" + f, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
R6 = rd("R6_gold58_serve.csv"); S6 = rd("n42gpu_gold58.csv"); S6cpu = rd("nartaa_gold58.csv"); S43 = rd("s43_gold58.csv")
R7 = pd.read_csv(D + "/R7OOF.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, ["p|" + l for l in L]].values
R384 = np.load(D + "/gold58_r384_kernelserve_cpu.npy"); R224 = np.load(D + "/gold58_r224_kernelserve.npy")
pct = lambda v: rankdata(v) / len(v)


def auc(s, y):
    p = y.sum(); q = len(y) - p
    if p == 0 or q == 0: return np.nan
    r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def final(ours=R6, s6=S6, A=0.25, ws6=0.2, table=C4, extra=None, wx=0.0, x_pos="end", ours_mix=None, wom=0.0, s6_mix=None, v6=0.0, labels=None):
    out = np.zeros((n, 12))
    for j, c in enumerate(L):
        rr = (1 - A) * pct(R384[:, j]) + A * pct(R224[:, j]); w = table.get(c, 0.5)
        o = pct(ours[:, j])
        if ours_mix is not None and (labels is None or c in labels): o = (1 - wom) * o + wom * pct(ours_mix[:, j])
        b = w * rr + (1 - w) * o
        s = pct(s6[:, j])
        if s6_mix is not None and (labels is None or c in labels): s = (1 - v6) * s + v6 * pct(s6_mix[:, j])
        f = (1 - ws6) * pct(b) + ws6 * s
        if extra is not None and (labels is None or c in labels): f = (1 - wx) * pct(f) + wx * pct(extra[:, j])
        out[:, j] = f
    return out


mac = lambda P: np.array([auc(P[:, j], Y[:, j]) for j in range(12)])
rng = np.random.default_rng(20261019); B = 3000
Wm = []
while len(Wm) < B:
    w = np.bincount(rng.integers(0, n, n), minlength=n)
    if all((w[Y[:, j] == 1].sum() > 0) and (w[Y[:, j] == 0].sum() > 0) for j in range(12)): Wm.append(w)
Wm = np.array(Wm, float)


def boot_auc(P):
    A = np.zeros((len(Wm), 12))
    for j in range(12):
        pos, neg = Y[:, j] == 1, Y[:, j] == 0
        Cm = (P[pos, j][:, None] > P[neg, j][None, :]) + 0.5 * (P[pos, j][:, None] == P[neg, j][None, :])
        wp, wn = Wm[:, pos], Wm[:, neg]
        A[:, j] = np.einsum('bi,ij,bj->b', wp, Cm, wn) / (wp.sum(1) * wn.sum(1))
    return A


res = {}
comp = {"r384": R384, "r224": R224, "R6": R6, "S6fall(gpu)": S6, "S6fall(cpu)": S6cpu, "s43": S43, "R7oof": R7}
res["components"] = {k: dict(macro=round(float(np.nanmean(mac(v))), 4), per_label=dict(zip(L, np.round(mac(v), 3).tolist()))) for k, v in comp.items()}
base = final(); bm = mac(base); res["c4_macro"] = float(bm.mean()); res["c4_per_label"] = dict(zip(L, np.round(bm, 4).tolist()))
print("c4 gold58 macro %.5f" % bm.mean())
for k, v in res["components"].items(): print("  %-12s %.4f | %s" % (k, v["macro"], " ".join("%s:%.3f" % (l[:6], a) for l, a in v["per_label"].items())))
# correlation of each leg with the c4 output (per-label Spearman, mean) and with each other
legs = {"r384": R384, "r224": R224, "R6": R6, "S6fall": S6, "s43": S43, "R7oof": R7, "c4": base}
rho = {a: {b: round(float(np.mean([spearmanr(legs[a][:, j], legs[b][:, j])[0] for j in range(12)])), 3) for b in legs} for a in legs}
res["spearman_mean"] = rho
print("Spearman (per-label mean):")
for a in legs: print("  %-7s %s" % (a, " ".join("%s:%.2f" % (b, rho[a][b]) for b in legs)))
bb = boot_auc(base)
CFG = {}
for v in (0.1, 0.2, 0.33, 0.5): CFG["S6slot s43 %.2f" % v] = dict(s6_mix=S43, v6=v)
for wx in (0.05, 0.1, 0.2): CFG["s43 end %.2f" % wx] = dict(extra=S43, wx=wx)
for wom in (0.1, 0.2, 0.33, 0.5): CFG["ours R6+R7 %.2f" % wom] = dict(ours_mix=R7, wom=wom)
for wx in (0.05, 0.1, 0.2): CFG["R7 end %.2f" % wx] = dict(extra=R7, wx=wx)
CFG["S6 cpu instead of gpu"] = dict(s6=S6cpu)
res["cfg"] = {}
for k, kw in CFG.items():
    P = final(**kw); pt = mac(P) - bm; d = (boot_auc(P) - bb).mean(1)
    top = np.argsort(-np.abs(pt))[:3]
    res["cfg"][k] = dict(point=float(pt.mean()), sd=float(d.std()), p_gt0=float((d > 0).mean()), per_label=dict(zip(L, np.round(pt, 4).tolist())))
    print("%-22s %+.5f sd %.5f P>0 %.3f | %s" % (k, pt.mean(), d.std(), (d > 0).mean(), ", ".join("%s %+.4f" % (L[j], pt[j]) for j in top if pt[j] != 0)))
# quadratic-model curvature of "add a member at the end": second difference on gold58 (label-free part = how much ranks move)
def rank_move(extra, wx):
    P = final(extra=extra, wx=wx)
    return float(np.mean([np.mean(np.abs(pct(P[:, j]) - pct(base[:, j]))) for j in range(12)]))
res["rank_move"] = {k: {str(w): rank_move(v, w) for w in (0.05, 0.1, 0.2)} for k, v in (("s43", S43), ("R7oof", R7), ("R6", R6), ("r384", R384))}
print("mean |rank shift| of the final when a member is appended at w:", json.dumps(res["rank_move"]))
json.dump(res, open(sys.argv[2], "w"), indent=1)
