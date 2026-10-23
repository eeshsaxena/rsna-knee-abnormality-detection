# m11_null.py (research only, CPU): is the OA-trio decorrelation gain OA-specific? Nulls on c4 gold58:
#  (1) each OA label minus a * mean pct of a RANDOM PAIR of non-OA labels (all 36 pairs);
#  (2) each OA label minus a * mean pct of all 9 non-OA labels (global abnormality, negative sign);
#  (3) the same within-group decorrelation applied to the other anatomical groups (menisci, ligaments, fluid trio, bone pair);
#  plus the report-OOF control for (3) with R7 OOF.
import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G, R
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
P = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); b = w * rr + (1 - w) * pct(R6[:, j]); P[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
ix = lambda names: [L.index(c) for c in names]
OA = ix(['Medial OA', 'Lateral OA', 'PF OA']); NON = [j for j in range(12) if j not in OA]


def sub(X, y, targets, helpers_of, a):
    d = 0.
    for t in targets:
        hs = helpers_of(t); s = pct(X[:, t]) - a * np.mean([pct(X[:, h]) for h in hs], 0)
        d += auc(s, y[:, t]) - auc(X[:, t], y[:, t])
    return d / 12


for a in (.1, .2):
    oa = sub(P, Y, OA, lambda t: [h for h in OA if h != t], a)
    nul = np.array([sub(P, Y, OA, lambda t, p=p: list(p), a) for p in itertools.combinations(NON, 2)])
    glob = sub(P, Y, OA, lambda t: NON, a)
    print('a %.1f: OA-trio decor %+.5f | random non-OA pair nulls (36): mean %+.5f sd %.5f max %+.5f, n >= OA-trio %d | minus global non-OA %+.5f' % (
        a, oa, nul.mean(), nul.std(), nul.max(), (nul >= oa).sum(), glob))
GROUPS = {'menisci': ['Medial Meniscus', 'Lateral Meniscus'], 'ligaments': ['ACL', 'MCL'], 'fluid': ['Effusion', 'Synovitis', "Baker's"],
          'bone': ['Contusion', 'Fracture'], 'OA': ['Medial OA', 'Lateral OA', 'PF OA']}
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
lab = pd.read_csv(R + '/_labels/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[L]
idx = [u for u in r7.index if u not in set(g.StudyInstanceUID) and u in lab.index]
XO = r7.loc[idx, ['p|' + c for c in L]].values; YO = (lab.loc[idx].fillna(0).values >= .5).astype(int)
for name, gl in GROUPS.items():
    gi = ix(gl)
    print('%-9s within-group decor: gold58 c4 a.1 %+.5f a.2 %+.5f | report OOF (R7, sol56) a.1 %+.5f a.2 %+.5f' % (
        name, sub(P, Y, gi, lambda t, gi=gi: [h for h in gi if h != t], .1), sub(P, Y, gi, lambda t, gi=gi: [h for h in gi if h != t], .2),
        sub(XO, YO, gi, lambda t, gi=gi: [h for h in gi if h != t], .1), sub(XO, YO, gi, lambda t, gi=gi: [h for h in gi if h != t], .2)))
