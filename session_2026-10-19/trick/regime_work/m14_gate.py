# m14_gate.py: kernel-natural gate values for the OA-decor build (final c4 output, after the failed-row fill), a = 0 / .1 / .2:
#   for t in (Medial OA, Lateral OA, PF OA): out_t = pct(f_t) - a * (pct(f_o1) + pct(f_o2)) / 2   (f = the final c4 column, BEFORE any change)
# a = 0 must equal c4 ranks exactly; a = .1 / .2 give the gold58 replay targets. Other 9 columns untouched.
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
F = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); b = w * rr + (1 - w) * pct(R6[:, j]); F[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
OA = [L.index(c) for c in ('Medial OA', 'Lateral OA', 'PF OA')]
def decor(F, a):
    O = F.copy(); U = np.stack([pct(F[:, t]) for t in OA], 1)
    for k, t in enumerate(OA):
        o = [m for m in range(3) if m != k]; O[:, t] = U[:, k] - a * (U[:, o[0]] + U[:, o[1]]) / 2
    return O
for a in (0., .1, .2):
    O = decor(F, a); per = np.array([auc(O[:, j], Y[:, j]) for j in range(12)])
    same = all((pd.Series(O[:, j]).rank().values == pd.Series(F[:, j]).rank().values).all() for j in range(12))
    print('a %.1f gold58 macro %.5f | OA AUCs %s | ranks identical to c4: %s | changed OA rank pairs %s' % (a, per.mean(), np.round(per[OA], 4).tolist(), same,
          [int((np.sign(np.subtract.outer(O[:, t], O[:, t])) != np.sign(np.subtract.outer(F[:, t], F[:, t]))).sum() // 2) for t in OA]))
