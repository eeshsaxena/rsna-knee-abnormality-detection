# m13_full_nested.py (research only, CPU): FULLY nested version of the group decorrelation finding. On the inner half choose the
# anatomical group (menisci / ligaments / fluid / bone / OA), the sign (decorrelate = -1 or borrow = +1) and a in {.1, .2} by inner macro
# delta; apply the winner (if its inner delta > 0) to the outer half. Variant B: apply EVERY (group, sign, a=.1) whose inner delta > 0.
# This also selects the group and sign, so it is honest about the m5 / m8 looks that led to the OA trio.
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
P = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); b = w * rr + (1 - w) * pct(R6[:, j]); P[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
GROUPS = {'menisci': ['Medial Meniscus', 'Lateral Meniscus'], 'ligaments': ['ACL', 'MCL'], 'fluid': ['Effusion', 'Synovitis', "Baker's"],
          'bone': ['Contusion', 'Fracture'], 'OA': ['Medial OA', 'Lateral OA', 'PF OA']}
GI = {k: [L.index(c) for c in v] for k, v in GROUPS.items()}


def gd(ix, grp, sgn, a):
    gi = GI[grp]; d = 0.
    for t in gi:
        o = [h for h in gi if h != t]
        s = pct(P[ix, t]) + sgn * a * np.mean([pct(P[ix, h]) for h in o], 0)
        d += auc(s, Y[ix, t]) - auc(P[ix, t], Y[ix, t])
    return d / 12


ok = lambda ix: all(0 < Y[ix, j].sum() < len(ix) for j in range(12))
rng = np.random.default_rng(99); A, B, picks = [], [], {}; cnt = 0
while cnt < 1000:
    perm = rng.permutation(n); h1, h2 = perm[:29], perm[29:]
    if not (ok(h1) and ok(h2)): continue
    cnt += 1
    for inner, outer in [(h1, h2), (h2, h1)]:
        cands = [(gd(inner, gname, s, a), gname, s, a) for gname in GI for s in (-1, 1) for a in (.1, .2)]
        best = max(cands)
        if best[0] > 0:
            A.append(gd(outer, best[1], best[2], best[3])); picks[best[1:]] = picks.get(best[1:], 0) + 1
        else:
            A.append(0.)
        B.append(sum(gd(outer, gname, s, .1) for (v, gname, s, a) in cands if a == .1 and v > 0))
for name, v in (('A best (group, sign, a)', A), ('B all positive (group, sign) at a .1', B)):
    v = np.array(v); print('%-38s outer macro delta mean %+.5f sd %.5f P>0 %.3f P<0 %.3f' % (name, v.mean(), v.std(), (v > 0).mean(), (v < 0).mean()))
print('inner picks (group, sign, a):', sorted(picks.items(), key=lambda z: -z[1])[:8])
