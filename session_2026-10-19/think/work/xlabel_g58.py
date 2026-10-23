# xlabel_g58.py (10-19 think/forum lens, read-only): does a second label's c4 score help rank a target label on gold58?
# Builds the c4 blend on gold58 from the existing leg predictions (same recipe as session_2026-10-18/next/quick_r6_slate.py),
# then for target t and helper h: s' = pct(c4_t) + a * pct(c4_h), AUC vs expert labels; paired bootstrap P(delta > 0).
import numpy as np, pandas as pd, json, sys
from scipy.stats import rankdata
D = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
R6 = pd.read_csv(D + "/R6_gold58_serve.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
R384 = np.load(D + "/gold58_r384_kernelserve_cpu.npy"); R224 = np.load(D + "/gold58_r224_kernelserve.npy")
S6 = pd.read_csv(D + "/n42gpu_gold58.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    p = y.sum(); q = len(y) - p
    r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
P = np.zeros((n, 12))
for j, c in enumerate(L):
    rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); w = C4.get(c, .5)
    b = w * rr + (1 - w) * pct(R6[:, j]); P[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
base = [auc(P[:, j], Y[:, j]) for j in range(12)]
print('c4-on-gold58 per-label:', {L[j]: round(base[j], 4) for j in range(12)}, 'macro %.4f' % np.mean(base))
# cross-AUC matrix: AUC of c4 score of label h for expert label t
M = np.array([[auc(P[:, h], Y[:, t]) for h in range(12)] for t in range(12)])
pd.set_option('display.width', 250)
print(pd.DataFrame(M, index=['y:' + l for l in L], columns=['s:' + l[:6] for l in L]).round(3))
rng = np.random.default_rng(11); idxs = []
while len(idxs) < 2000:
    ix = rng.integers(0, n, n)
    if all(0 < Y[ix, j].sum() < n for j in range(12)): idxs.append(ix)
out = {}
for t, h in [('Synovitis', 'Effusion'), ('Effusion', 'Synovitis'), ('Contusion', 'ACL'), ('Fracture', 'Contusion'), ('ACL', 'Contusion'),
             ("Baker's", 'Effusion'), ('Medial OA', 'Medial Meniscus'), ('Lateral OA', 'Lateral Meniscus'), ('PF OA', 'Medial OA')]:
    jt, jh = L.index(t), L.index(h)
    for a in (0.1, 0.2, 0.3, 0.5):
        s1 = pct(P[:, jt]) + a * pct(P[:, jh])
        d0 = auc(s1, Y[:, jt]) - base[jt]
        bd = np.array([auc(s1[ix], Y[ix, jt]) - auc(P[ix, jt], Y[ix, jt]) for ix in idxs])
        out['%s<-%s@%.1f' % (t, h, a)] = (round(d0, 4), round(float((bd > 0).mean()), 3), round(float(bd.std()), 4))
        print('%-10s <- %-15s a=%.1f  dAUC %+.4f  P>0 %.3f  sd %.4f  (macro %+.5f)' % (t, h, a, d0, (bd > 0).mean(), bd.std(), d0 / 12))
json.dump(out, open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/work/xlabel_g58.json', 'w'), indent=1)
