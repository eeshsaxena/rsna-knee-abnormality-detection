# r7mix_g58.py: robustness of "ours = (1-v) R6 + v R7" inside the c4 recipe on gold58 (read-only numpy on existing preds).
# Variants: rank mix (pct) and probability mix (the A14 cell's form: per-study mean of probabilities, here weighted).
# Checks: curve over v, paired bootstrap, rerank bootstrap, split-half direction stability, leave-one-label-out.
import sys, json, numpy as np, pandas as pd
from scipy.stats import rankdata
D = sys.argv[1]
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(D + "/" + f, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
R6 = rd("R6_gold58_serve.csv"); S6 = rd("n42gpu_gold58.csv")
r7 = pd.read_csv(D + "/R7OOF.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids]
R7 = r7[["p|" + l for l in L]].values
print("R7 fold of gold58 rows:", r7.r7_fold.value_counts().to_dict())
R384 = np.load(D + "/gold58_r384_kernelserve_cpu.npy"); R224 = np.load(D + "/gold58_r224_kernelserve.npy")


def pct(v): return rankdata(v) / len(v)


def auc(s, y):
    p = y.sum(); q = len(y) - p
    r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def final(v=0.0, mode="rank", idx=None):
    idx = np.arange(n) if idx is None else idx
    out = np.zeros((len(idx), 12))
    for j, c in enumerate(L):
        rr = .75 * pct(R384[idx, j]) + .25 * pct(R224[idx, j]); w = C4.get(c, 0.5)
        if mode == "rank": o = (1 - v) * pct(R6[idx, j]) + v * pct(R7[idx, j])
        else: o = pct((1 - v) * R6[idx, j] + v * R7[idx, j])
        b = w * rr + (1 - w) * pct(o)
        out[:, j] = .8 * pct(b) + .2 * pct(S6[idx, j])
    return out


def mac(P, idx=None):
    yy = Y if idx is None else Y[idx]
    return np.array([auc(P[:, j], yy[:, j]) for j in range(12)])


base = mac(final())
rng = np.random.default_rng(11); idxs = []
while len(idxs) < 1500:
    ix = rng.integers(0, n, n)
    if all(0 < Y[ix, j].sum() < n for j in range(12)): idxs.append(ix)
bb = np.array([mac(final(idx=ix), ix) for ix in idxs])
out = {}
for mode in ("rank", "prob"):
    for v in (0.1, 0.15, 0.2, 0.25, 0.3, 0.5):
        P = final(v, mode); pt = mac(P) - base
        d = np.array([(mac(final(v, mode, ix), ix) - b0).mean() for ix, b0 in zip(idxs, bb)])
        lolo = [float(np.delete(pt, j).mean()) for j in range(12)]
        out["%s %.2f" % (mode, v)] = dict(point=float(pt.mean()), p_rerank=float((d > 0).mean()), sd=float(d.std()),
                                          lolo_min=min(lolo), lolo_max=max(lolo), per_label=dict(zip(L, np.round(pt, 4).tolist())))
        print("%s v %.2f: %+.5f  P>0(rerank) %.3f sd %.5f | leave-one-label-out min %+.5f max %+.5f | top %s" % (
            mode, v, pt.mean(), (d > 0).mean(), d.std(), min(lolo), max(lolo),
            ", ".join("%s %+.4f" % (L[j], pt[j]) for j in np.argsort(-np.abs(pt))[:3])))
# split-half direction stability (A: 29 studies, B: the other 29), 800 splits: does the sign on half A predict half B?
agree = {}; held = {}
rng2 = np.random.default_rng(5)
for mode, v in (("rank", 0.2), ("prob", 0.2), ("rank", 0.1), ("prob", 0.5)):
    a_ = []; h_ = []
    k = 0
    while k < 800:
        perm = rng2.permutation(n); A, Bh = perm[:29], perm[29:]
        if any(Y[A, j].sum() in (0, len(A)) or Y[Bh, j].sum() in (0, len(Bh)) for j in range(12)): continue
        dA = (mac(final(v, mode, A), A) - mac(final(0, mode, A), A)).mean()
        dB = (mac(final(v, mode, Bh), Bh) - mac(final(0, mode, Bh), Bh)).mean()
        a_.append(np.sign(dA) == np.sign(dB)); h_.append(dB if dA > 0 else np.nan); k += 1
    agree["%s %.2f" % (mode, v)] = float(np.mean(a_)); held["%s %.2f" % (mode, v)] = float(np.nanmean(h_))
    print("split-half %s %.2f: sign agreement %.3f | held-out mean gain when half A says + : %+.5f" % (mode, v, np.mean(a_), np.nanmean(h_)))
out["split_half_agree"] = agree; out["split_half_heldout_if_A_pos"] = held
json.dump(out, open(sys.argv[2], "w"), indent=1)
