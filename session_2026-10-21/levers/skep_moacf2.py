# skep_moacf2.py (10-21 levers, Claude, SKEPTIC of MOACF, part 2). RESEARCH ONLY, local CPU, NO OAI, no box. Writes skep_moacf2.log.
# (A) Does the MOACF ruler gain depend on E-OA (base no-EOA / pw25 / pw30)? v4 labels (all rows known), REP and EB_v4 (self-consistent v4).
# (B) Member co-firing on gold58 (label-free): Spearman rho(MOA, Cont / Frac / ACL) for Raptor 384, Raptor 224, R6, S6fall, c4 blend, and
#     the train-OOF rulers on train (R7, s1044, s1045) for scale. Raptor has no OOF, so this is the only look at whether it shares the coupling.
# (C) Where the ruler gain comes from: share of the MOA AUC gain from (MOA- acute-high) rows vs other rows, R7 / 3mean, v4, pw30 base.
import json, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from rg_lib import (L, J, AB, OA, eoa, near_pd, copula_cells, weights_from_target, load_all, pct, wauc_multi, tet_xy_fast, LD)
Bd = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/'
log = open(Bd + 'skep_moacf2.log', 'w')


def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()


D = load_all(); X, tr3 = D['X'], D['tr3']
g, g58 = LD.gold(); ids58 = g58.StudyInstanceUID.tolist(); E58 = g58[L].values.astype(float)
REL = json.load(open(Bd + 'regime_dlt.json'))['rel']['nontrio']
C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
trio = lambda i, j: L[i] in OA and L[j] in OA


def nest(A, Bm, C):
    V = np.column_stack([C4W.get(c, .5) * pct(A[:, j]) + (1 - C4W.get(c, .5)) * pct(Bm[:, j]) for j, c in enumerate(L)])
    return np.column_stack([.8 * pct(V[:, j]) + .2 * pct(C[:, j]) for j in range(12)])


RAWR = dict(X)
for p in itertools.permutations(['R7', 's1044', 's1045']):
    RAWR['N:' + '/'.join(x[:2] if x == 'R7' else x[-2:] for x in p)] = nest(X[p[0]], X[p[1]], X[p[2]])
lab4 = LD.labels('v4'); Y4 = lab4.reindex(tr3)[L].values.astype(float); Y4g = lab4.reindex(ids58)[L].values.astype(float)


def tetm(Yx):
    T = np.eye(12)
    for i in range(12):
        for j in range(i + 1, 12): T[i, j] = T[j, i] = tet_xy_fast(Yx[:, i], Yx[:, j])
    return T


Tr = tetm(Y4); Dm = np.zeros((12, 12))
for i in range(12):
    for j in range(i + 1, 12): Dm[i, j] = Dm[j, i] = tet_xy_fast(E58[:, i], E58[:, j]) - tet_xy_fast(Y4g[:, i], Y4g[:, j])
mD = np.mean([Dm[i, j] for i in range(12) for j in range(i + 1, 12) if not trio(i, j)])
De = np.array([[0 if i == j else (Dm[i, j] if trio(i, j) else mD + REL * (Dm[i, j] - mD)) for j in range(12)] for i in range(12)])
cols = OA + ['Contusion', 'Fracture']; jc = [J[c] for c in cols]
Ys = Y4[:, jc].astype(int); yt = Ys[:, 0]
W = [np.ones(len(Ys)), weights_from_target(Ys, copula_cells(Ys.mean(0), near_pd(Tr[np.ix_(jc, jc)] + De[np.ix_(jc, jc)])))]


def moacf(B):
    c = B[:, J['Medial OA']].copy()
    c = pct(pct(c) - .05 * pct(B[:, J['Contusion']])); c = pct(pct(c) - .05 * pct(B[:, J['Fracture']]))
    return c


P('(A) MOACF gain vs E-OA base, v4 labels, label AUC x1e4 [REP | EB_v4]; also base MOA AUC (REP)')
for base in ('none', 'pw25', 'pw30'):
    out = {}
    for rn, F in RAWR.items():
        B = F if base == 'none' else eoa(F, .25 if base == 'pw25' else .3, 'pw')
        b0 = wauc_multi(B[:, J['Medial OA']], yt, W); b1 = wauc_multi(moacf(B), yt, W)
        out[rn] = ((b1 - b0) * 1e4, b0[0])
    pl = ' '.join('%s %+5.1f|%+5.1f (%.4f)' % (k, out[k][0][0], out[k][0][1], out[k][1]) for k in ('R7', 's1044', 's1045', '3mean'))
    ne = np.array([out[k][0] for k in out if k.startswith('N:')])
    P('base %-4s: %s || nests REP %+5.1f..%+5.1f EB_v4 %+5.1f..%+5.1f' % (base, pl, ne[:, 0].min(), ne[:, 0].max(), ne[:, 1].min(), ne[:, 1].max()))

P('')
P('(B) label-free co-firing, Spearman rho of MOA with Cont / Frac / ACL')
G = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58'
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids58, L].values
MEM = {}
try:
    MEM['R6'] = rd('R6_gold58_serve.csv'); MEM['S6fall'] = rd('n42gpu_gold58.csv')
    MEM['Raptor384'] = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); MEM['Raptor224'] = np.load(G + '/gold58_r224_kernelserve.npy')
    rr = np.column_stack([.75 * pct(MEM['Raptor384'][:, j]) + .25 * pct(MEM['Raptor224'][:, j]) for j in range(12)])
    MEM['Raptor(.75/.25)'] = rr
    MEM['c4(no EOA)'] = np.column_stack([.8 * pct(C4W.get(c, .5) * rr[:, j] + (1 - C4W.get(c, .5)) * pct(MEM['R6'][:, j])) + .2 * pct(MEM['S6fall'][:, j]) for j, c in enumerate(L)])
except Exception as e:
    P('member load failed:', repr(e))
MEM['c4 kernel (pw30 path, gold58)'] = None
c4k = D['c4']
try:
    MEM['c4 kernel (pw30 path, gold58)'] = np.asarray(c4k)[:, :12].astype(float) if not hasattr(c4k, 'loc') else c4k.loc[ids58, L].values.astype(float)
except Exception as e:
    P('c4 kernel output not usable:', repr(e)); MEM.pop('c4 kernel (pw30 path, gold58)')
for k, M in MEM.items():
    if M is None: continue
    r = [spearmanr(M[:, J['Medial OA']], M[:, J[o]]).correlation for o in ('Contusion', 'Fracture', 'ACL')]
    P('gold58 %-30s MOA~Cont %+.2f  MOA~Frac %+.2f  MOA~ACL %+.2f' % (k, *r))
for k in ('R7', 's1044', 's1045'):
    M = X[k]; Mg = None
    r = [spearmanr(M[:, J['Medial OA']], M[:, J[o]]).correlation for o in ('Contusion', 'Fracture', 'ACL')]
    P('train  %-30s MOA~Cont %+.2f  MOA~Frac %+.2f  MOA~ACL %+.2f' % (k + ' OOF', *r))
try:
    r7 = LD.r7(); M = r7.reindex(ids58)[['p|' + c for c in L]].values.astype(float)
    if not np.isnan(M).any():
        r = [spearmanr(M[:, J['Medial OA']], M[:, J[o]]).correlation for o in ('Contusion', 'Fracture', 'ACL')]
        P('gold58 %-30s MOA~Cont %+.2f  MOA~Frac %+.2f  MOA~ACL %+.2f' % ('R7 OOF', *r))
except Exception as e:
    P('R7 gold58 failed', repr(e))

P('')
P('(C) where the gain sits (v4, REP, pw30 base): rank moves of MOA- knees with high acute score vs MOA+ knees')
for rn in ('R7', '3mean'):
    B = eoa(RAWR[rn], .3, 'pw'); c0 = pct(B[:, J['Medial OA']]); c1 = moacf(B)
    ac = np.maximum(pct(B[:, J['Contusion']]), pct(B[:, J['Fracture']]))
    hi = ac > .8
    for nm, m in (('MOA+ acute-hi', (yt == 1) & hi), ('MOA+ acute-lo', (yt == 1) & ~hi), ('MOA- acute-hi', (yt == 0) & hi), ('MOA- acute-lo', (yt == 0) & ~hi)):
        P('%-6s %-14s n %4d mean pct move %+.4f' % (rn, nm, m.sum(), (c1[m] - c0[m]).mean()))
P('SKEP_MOACF2 DONE')
