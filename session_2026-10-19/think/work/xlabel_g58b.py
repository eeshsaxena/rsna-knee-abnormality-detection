# xlabel_g58b.py: robustness of Synovitis <- Effusion on gold58: per-leg, split-half, and the cross-AUC rows.
import numpy as np, pandas as pd
from scipy.stats import rankdata
D = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
R6 = pd.read_csv(D + "/R6_gold58_serve.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
R7 = pd.read_csv(D + "/R7OOF.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, ["p|" + l for l in L]].values
R384 = np.load(D + "/gold58_r384_kernelserve_cpu.npy"); R224 = np.load(D + "/gold58_r224_kernelserve.npy")
S6 = pd.read_csv(D + "/n42gpu_gold58.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    p = y.sum(); q = len(y) - p
    r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
def c4(ours):
    P = np.zeros((n, 12))
    for j, c in enumerate(L):
        rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); w = C4.get(c, .5)
        b = w * rr + (1 - w) * pct(ours[:, j]); P[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
    return P
P = c4(R6)
js, je = L.index('Synovitis'), L.index('Effusion')
print('y:Synovitis row (AUC of each c4 column for expert Synovitis):')
print({L[h]: round(auc(P[:, h], Y[:, js]), 3) for h in range(12)})
print('y:Effusion row:', {L[h]: round(auc(P[:, h], Y[:, je]), 3) for h in range(12)})
for name, M in [('R6', R6), ('R7oof', R7), ('r384', R384), ('r224', R224), ('S6fall', S6), ('c4', P), ('c4(R7)', c4(R7))]:
    s, e = M[:, js], M[:, je]
    row = ['%s syn %.3f eff->syn %.3f' % (name, auc(s, Y[:, js]), auc(e, Y[:, js]))]
    for a in (.1, .2, .3, .5, 1.0):
        row.append('a%.1f %+.4f' % (a, auc(pct(s) + a * pct(e), Y[:, js]) - auc(s, Y[:, js])))
    print(' | '.join(row))
# split-half direction agreement at a=.2 on c4
rng = np.random.default_rng(3); agree = 0; tot = 0; both_pos = 0
for _ in range(2000):
    perm = rng.permutation(n); A, B = perm[:29], perm[29:]
    if not (0 < Y[A, js].sum() < 29 and 0 < Y[B, js].sum() < 29): continue
    dA = auc(pct(P[A, js]) + .2 * pct(P[A, je]), Y[A, js]) - auc(P[A, js], Y[A, js])
    dB = auc(pct(P[B, js]) + .2 * pct(P[B, je]), Y[B, js]) - auc(P[B, js], Y[B, js])
    tot += 1; agree += (np.sign(dA) == np.sign(dB)); both_pos += (dA > 0 and dB > 0)
print('split-half a=.2: sign agree %.3f, both halves positive %.3f (n=%d)' % (agree / tot, both_pos / tot, tot))
# the 5 Synovitis positives without Effusion and the 13 Effusion positives without Synovitis: where do they rank?
s1 = pct(P[:, js]) + .2 * pct(P[:, je])
r0 = rankdata(P[:, js]); r1 = rankdata(s1)
for i in range(n):
    if Y[i, js] != Y[i, je]:
        print('syn %d eff %d  rank syn-only %4.0f -> mixed %4.0f   (eff pct %.2f)' % (Y[i, js], Y[i, je], r0[i], r1[i], pct(P[:, je])[i]))
# per-half sign distribution at several a
for a in (.1, .2, .3):
    pos = neg = zero = 0
    for _ in range(4000):
        perm = rng.permutation(n); A = perm[:29]
        if not (0 < Y[A, js].sum() < 29): continue
        d = auc(pct(P[A, js]) + a * pct(P[A, je]), Y[A, js]) - auc(P[A, js], Y[A, js])
        pos += d > 1e-12; neg += d < -1e-12; zero += abs(d) <= 1e-12
    t = pos + neg + zero
    print('half-sample a=%.1f: P(d>0) %.3f  P(d<0) %.3f  P(d=0) %.3f' % (a, pos / t, neg / t, zero / t))
