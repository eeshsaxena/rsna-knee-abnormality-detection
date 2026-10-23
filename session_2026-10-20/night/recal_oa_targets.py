# recal_oa_targets.py (10-20 night, Claude, TRACK 1). RESEARCH ONLY, local CPU, no OAI. Writes recal_oa_targets.log.
# The two killed OA-target directions that the gap-matched OAI read (recal_box_gapmatch.log) does not kill: LOA <- MM (decorrelate,
# EX C12 / census) and LOA += LM (borrow, EX C14 / C13 half), measured on the ACTUAL best-id base: c4 gold58 rebuild -> pw30 E-OA block
# (A_OA .3, pair weights), then the extra line. Rulers: gold58 c4 (paired bootstrap 2000, split-half 1000), R7 OOF on gold74 (pw30 applied
# to R7's OA columns), report OOF (R7 vs report v2, should lose for the decorrelation if the coupling is report-inherited), read noise (R7
# OOF + sol56 labels, 390 / 910) on the pw30-applied R7 base. Also c4's coupling (rho_ns, q, gap) on the pw30 base.
import sys
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
import census_lib as CL
from census_lib import auc, pct, L
import load as LD
OUTD = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/night/'
log = open(OUTD + 'recal_oa_targets.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()
rng = np.random.default_rng(2110)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
J = {c: i for i, c in enumerate(L)}; OA = ['Medial OA', 'Lateral OA', 'PF OA']
W_PW = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}


def pw(F, a=.3):
    G = F.copy(); U = {c: pct(F[:, J[c]]) for c in OA}
    for c in OA:
        o = [k for k in OA if k != c]; w = [W_PW[tuple(sorted((c, k), key=OA.index))] for k in o]
        G[:, J[c]] = pct(U[c] - a / 2.0 * (w[0] * U[o[0]] + w[1] * U[o[1]]))
    return G


def extra(F, t, o, a):
    G = F.copy(); G[:, J[t]] = pct(pct(F[:, J[t]]) - a * pct(F[:, J[o]])); return G


D = CL.data(); Y = D['Y58']; n = len(Y)
B30 = pw(D['C4'])
P('pw30 base gold58 macro %.5f (a21 gate: .94336)' % np.mean([auc(B30[:, j], Y[:, j]) for j in range(12)]))
BI = [rng.integers(0, n, n) for _ in range(2000)]
SPL = [rng.permutation(n) for _ in range(1000)]
for t, o in (('Lateral OA', 'Medial Meniscus'), ('Lateral OA', 'Lateral Meniscus'), ('Medial OA', 'Medial Meniscus')):
    jt = J[t]; y = Y[:, jt]
    zt, zo = ns(B30[:, jt]), ns(B30[:, J[o]]); ct = np.corrcoef(zt, y)[0, 1]
    rho = np.corrcoef(zt, zo)[0, 1]; q = np.corrcoef(zo, y)[0, 1] / ct
    gb = []
    for ii in BI[:1000]:
        yy = y[ii]; a1 = np.corrcoef(zt[ii], yy)[0, 1]; gb.append(np.corrcoef(zt[ii], zo[ii])[0, 1] - np.corrcoef(zo[ii], yy)[0, 1] / a1)
    P('== %s <- %s on pw30 base: c4 rho_ns %.3f q %.3f gap %+.3f (boot sd %.3f, robust %.3f) | base AUC %.4f' % (
        t, o, rho, q, rho - q, np.std(gb), (np.percentile(gb, 84) - np.percentile(gb, 16)) / 2, auc(B30[:, jt], y)))
    for a in (-.15, -.1, -.05, .05, .1, .15, .2):
        G = extra(B30, t, o, a); d = auc(G[:, jt], y) - auc(B30[:, jt], y)
        bb = np.array([auc(G[ii, jt], y[ii]) - auc(B30[ii, jt], y[ii]) for ii in BI])
        sh = []
        for p_ in SPL:
            h1, h2 = p_[:29], p_[29:]
            if y[h1].min() == y[h1].max() or y[h2].min() == y[h2].max(): continue
            d1 = auc(G[h1, jt], y[h1]) - auc(B30[h1, jt], y[h1]); d2 = auc(G[h2, jt], y[h2]) - auc(B30[h2, jt], y[h2]); sh.append(np.sign(d1) == np.sign(d2) and d1 != 0)
        npos = int(y.sum()); flips = d * npos * (n - npos)
        P('   a %+.2f gold58 d %+.4f (macro %+.2fe-4) boot sd %.4f P>0 %.2f split-half same-sign %.2f | net pair flips %+.1f of %d' % (
            a, d, d / 12 * 1e4, bb.std(), (bb > 0).mean(), np.mean(sh), flips, npos * (n - npos)))
# R7 OOF on gold74 and report OOF, pw30 applied to R7's OA columns
for nm, F, Yr in (('R7 x gold74', D['R7_74'], D['Y74']), ('R7 x report v2 (train - gold74)', D['F7'], D['Yv2']), ('3-model mean x report v2', D['M3'], D['Y3v2'])):
    ok = ~np.isnan(F[:, [J[c] for c in OA]]).any(1); F = pw(F[ok]); Yr = Yr[ok]
    for t, o in (('Lateral OA', 'Medial Meniscus'), ('Lateral OA', 'Lateral Meniscus')):
        jt = J[t]; m = ~np.isnan(Yr[:, jt]) & ~np.isnan(F[:, J[o]]); y = Yr[m, jt].astype(int)
        s = ' '.join('a%+.2f %+.4f' % (a, auc(extra(F[m], t, o, a)[:, jt], y) - auc(F[m, jt], y)) for a in (-.1, -.05, .05, .1))
        P('%-32s %s <- %s (n %d, pos %d): %s' % (nm, t, o, m.sum(), y.sum(), s))
# read noise on the pw30-applied R7 OOF base (macro units)
r7 = LD.r7(); g, _ = LD.gold()
lab = pd.read_csv(LD.R + '/_labels/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[L]
ids = [u for u in r7.index if u not in set(g.StudyInstanceUID) and u in lab.index]
X = r7.loc[ids, ['p|' + c for c in L]].values.astype(float); Yn = (lab.loc[ids].fillna(0).values >= .5).astype(int)
for t, o, a in (('Lateral OA', 'Medial Meniscus', .1), ('Lateral OA', 'Lateral Meniscus', -.1), ('Lateral OA', 'Lateral Meniscus', -.05)):
    out = {}
    for size in (390, 910):
        v = []
        for _ in range(400):
            ix = rng.choice(len(ids), size, replace=False); F = pw(X[ix]); G = extra(F, t, o, a)
            v.append((auc(G[:, J[t]], Yn[ix, J[t]]) - auc(F[:, J[t]], Yn[ix, J[t]])) / 12)
        out[size] = np.std(v)
    P('read noise %s <- %s a %+.2f on pw30 base: public (390) %.2fe-4, private (910) %.2fe-4 macro' % (t, o, a, out[390] * 1e4, out[910] * 1e4))
P('RECAL_OA_TARGETS DONE')
