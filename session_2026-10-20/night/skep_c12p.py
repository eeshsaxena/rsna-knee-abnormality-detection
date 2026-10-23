# skep_c12p.py (10-20 night, Claude, SKEPTIC on C12' = LOA <- MM a .05 on pw30). RESEARCH ONLY, local CPU, no OAI.
# Writes skep_c12p.log. Checks: (A) gold58 placebo / multiplicity for the +.0058, (B) R7 on gold58 vs the 16 extra gold74 knees,
# (C) expert vs report LOA-MM coupling (paired on gold74 knees and train-wide), (D) measured read noise at a .05,
# (E) report-world gap for R7 (is the co-firing report-inherited?).
import sys
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
import census_lib as CL
from census_lib import auc, pct, L, tet
import load as LD
OUTD = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/night/'
log = open(OUTD + 'skep_c12p.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()
rng = np.random.default_rng(20201)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
J = {c: i for i, c in enumerate(L)}; OA = ['Medial OA', 'Lateral OA', 'PF OA']
W_PW = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}
T, O = J['Lateral OA'], J['Medial Meniscus']


def pw(F, a=.3):
    G = F.copy(); U = {c: pct(F[:, J[c]]) for c in OA}
    for c in OA:
        o = [k for k in OA if k != c]; w = [W_PW[tuple(sorted((c, k), key=OA.index))] for k in o]
        G[:, J[c]] = pct(U[c] - a / 2.0 * (w[0] * U[o[0]] + w[1] * U[o[1]]))
    return G


def ext(F, t, o, a):
    G = F.copy(); G[:, t] = pct(pct(F[:, t]) - a * pct(F[:, o])); return G


def ext_v(F, t, v, a):
    G = F.copy(); G[:, t] = pct(pct(F[:, t]) - a * pct(v)); return G


def logor(a, b):
    a = np.asarray(a, int); b = np.asarray(b, int)
    c11 = ((a == 1) & (b == 1)).sum() + .5; c10 = ((a == 1) & (b == 0)).sum() + .5; c01 = ((a == 0) & (b == 1)).sum() + .5; c00 = ((a == 0) & (b == 0)).sum() + .5
    lo = np.log(c11 * c00 / (c10 * c01)); se = np.sqrt(1 / c11 + 1 / c10 + 1 / c01 + 1 / c00)
    return lo, se, int(c11 - .5), int(a.sum()), int(b.sum()), len(a)


D = CL.data(); Y = D['Y58']; n = len(Y)
B = pw(D['C4']); y = Y[:, T]; base = auc(B[:, T], y)
d0 = {a: auc(ext(B, T, O, a)[:, T], y) - base for a in (-.15, -.1, -.05, .05, .1, .15, .2)}
P('== (A) gold58 c4 pw30 base, LOA AUC %.4f, pos %d of %d' % (base, y.sum(), n))
P('LOA <- MM d by a:', ' '.join('a%+.2f %+.4f' % (a, v) for a, v in d0.items()))
P('-- A1 LOA <- each partner at a .05 / a .10 (label d):')
for c in L:
    if c == 'Lateral OA': continue
    P('   %-18s a.05 %+.4f  a.10 %+.4f' % (c, auc(ext(B, T, J[c], .05)[:, T], y) - base, auc(ext(B, T, J[c], .10)[:, T], y) - base))
P('-- A2 all 132 directions on pw30 base at a .05 (label d):')
allv = []
for t in range(12):
    for o in range(12):
        if t == o: continue
        allv.append((auc(ext(B, t, o, .05)[:, t], Y[:, t]) - auc(B[:, t], Y[:, t]), L[t], L[o]))
v = np.array([x[0] for x in allv])
rk = int((v > d0[.05]).sum()) + 1
P('   mean %+.4f sd %.4f | #>=+.0058: %d of 132 | LOA<-MM rank %d | top5: %s' % (v.mean(), v.std(), int((v >= d0[.05] - 1e-9).sum()), rk,
  '; '.join('%s<-%s %+.4f' % (x[1], x[2], x[0]) for x in sorted(allv, reverse=True)[:5])))
P('-- A3 placebo: LOA <- random-permuted MM (rank-noise partner), 4000 draws')
pl05, plmax = [], []
grid = (-.15, -.1, -.05, .05, .1, .15, .2)
for _ in range(4000):
    vv = rng.permutation(B[:, O])
    ds = [auc(ext_v(B, T, vv, a)[:, T], y) - base for a in grid]
    pl05.append(ds[3]); plmax.append(max(ds[3:]))
pl05 = np.array(pl05); plmax = np.array(plmax)
P('   a .05: mean %+.4f sd %.4f P(>= +.0058) %.3f | max over a in {.05..,.2}: P(>= +.0058) %.3f | MM partner z vs placebo a.05 %.2f' % (
    pl05.mean(), pl05.std(), (pl05 >= d0[.05] - 1e-9).mean(), (plmax >= d0[.05] - 1e-9).mean(), (d0[.05] - pl05.mean()) / pl05.std()))
# (B) R7 gold58 vs extra 16
P('== (B) R7 OOF, pw30 applied: gold58 vs extra gold74 knees')
ids74 = LD.gold()[0].StudyInstanceUID.tolist()
F74 = D['R7_74']; Y74 = D['Y74']
in58 = np.array([u in set(D['ids']) for u in ids74])
ok = ~np.isnan(F74).any(1) & ~np.isnan(Y74[:, T])
for nm, m in (('gold74 all', ok), ('gold58 part', ok & in58), ('extra16 part', ok & ~in58)):
    F = pw(F74[m]); yy = Y74[m, T].astype(int)
    if yy.min() == yy.max(): P('   %s: no LOA variation' % nm); continue
    P('   %-12s n %d pos %d | a.05 %+.4f a.10 %+.4f' % (nm, m.sum(), yy.sum(), auc(ext(F, T, O, .05)[:, T], yy) - auc(F[:, T], yy),
                                                     auc(ext(F, T, O, .10)[:, T], yy) - auc(F[:, T], yy)))
# c4 on gold58 vs R7 on gold58 same knees
F = pw(D['R7_58']); m = ~np.isnan(F).any(1)
P('   R7 on gold58 knees (n %d): a.05 %+.4f | c4 a.05 %+.4f' % (m.sum(), auc(ext(F[m], T, O, .05)[:, T], y[m]) - auc(F[m, T], y[m]), d0[.05]))
# (C) coupling
P('== (C) LOA-MM label coupling: log OR (se) | tet | n11 / nLOA / nMM / n')
g = D['Y74']; g = g[~np.isnan(g[:, T]) & ~np.isnan(g[:, O])].astype(int)
P('   expert gold74     : lor %+.2f (%.2f) tet %+.2f | %s' % (*logor(g[:, T], g[:, O])[:2], tet(g[:, T], g[:, O]), logor(g[:, T], g[:, O])[2:]))
P('   expert gold58     : lor %+.2f (%.2f) tet %+.2f | %s' % (*logor(Y[:, T], Y[:, O])[:2], tet(Y[:, T], Y[:, O]), logor(Y[:, T], Y[:, O])[2:]))
for src in ('v2', 'v4', 'sol56'):
    lab = LD.labels(src)
    r = lab.reindex(ids74)[[L[T], L[O]]].values
    mm = ~np.isnan(r).any(1) & ~np.isnan(D['Y74'][:, [T, O]]).any(1)
    gg = D['Y74'][mm].astype(int)
    P('   report %-5s gold74 knees (n %d): lor %+.2f (%.2f) tet %+.2f | expert same knees lor %+.2f tet %+.2f | agree LOA %.2f MM %.2f' % (
        src, mm.sum(), *logor(r[mm, 0], r[mm, 1])[:2], tet(r[mm, 0].astype(int), r[mm, 1].astype(int)), logor(gg[:, T], gg[:, O])[0],
        tet(gg[:, T], gg[:, O]), (r[mm, 0] == gg[:, T]).mean(), (r[mm, 1] == gg[:, O]).mean()))
    allg = set(ids74); tr = [u for u in lab.index if u not in allg]
    rr = lab.loc[tr, [L[T], L[O]]].values; mm = ~np.isnan(rr).any(1)
    P('   report %-5s train (n %d): lor %+.2f (%.2f) tet %+.2f | prev LOA %.3f MM %.3f' % (src, mm.sum(), *logor(rr[mm, 0], rr[mm, 1])[:2],
      tet(rr[mm, 0].astype(int), rr[mm, 1].astype(int)), rr[mm, 0].mean(), rr[mm, 1].mean()))
# trio reference: expert vs report coupling for LOA-MOA, LOA-PFOA (the pair where the public confirmed an over-coupling)
for c in ('Medial OA', 'PF OA', 'Lateral Meniscus'):
    lab = LD.labels('v2'); r = lab.reindex(ids74)[[L[T], c]].values; mm = ~np.isnan(r).any(1) & ~np.isnan(D['Y74'][:, [T, J[c]]]).any(1); gg = D['Y74'][mm].astype(int)
    P('   ref LOA-%-16s gold74: report v2 tet %+.2f vs expert tet %+.2f (n %d)' % (c, tet(r[mm, 0].astype(int), r[mm, 1].astype(int)), tet(gg[:, T], gg[:, J[c]]), mm.sum()))
# (E) report-world gap for R7 (train, v2)
P('== (E) report-world coupling gap, R7 OOF pw30-applied, train minus gold74, v2 labels')
F7 = pw(D['F7']); Yv = D['Yv2']; mm = ~np.isnan(Yv[:, T]) & ~np.isnan(F7).any(1)
zt, zo = ns(F7[mm, T]), ns(F7[mm, O]); yy = Yv[mm, T]
rho = np.corrcoef(zt, zo)[0, 1]; q = np.corrcoef(zo, yy)[0, 1] / np.corrcoef(zt, yy)[0, 1]
P('   R7 report world: rho_ns %.3f q %.3f gap %+.3f (n %d)' % (rho, q, rho - q, mm.sum()))
F7r = D['F7']; zt0, zo0 = ns(F7r[mm, T]), ns(F7r[mm, O])
P('   R7 raw (no pw) rho_ns %.3f' % np.corrcoef(zt0, zo0)[0, 1])
zt, zo = ns(B[:, T]), ns(B[:, O])
P('   c4 pw30 gold58: rho_ns %.3f q(expert) %.3f' % (np.corrcoef(zt, zo)[0, 1], np.corrcoef(zo, y)[0, 1] / np.corrcoef(zt, y)[0, 1]))
# (D) read noise at a .05 (R7 OOF + sol56, 390 / 910) on pw30-applied R7
r7 = LD.r7(); g_, _ = LD.gold()
lab = pd.read_csv(LD.R + '/_labels/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[L]
ids = [u for u in r7.index if u not in set(g_.StudyInstanceUID) and u in lab.index]
X = r7.loc[ids, ['p|' + c for c in L]].values.astype(float); Yn = (lab.loc[ids].fillna(0).values >= .5).astype(int)
for a in (.05, .10):
    out = {}
    for size in (390, 910):
        vv = []
        for _ in range(600):
            ix = rng.choice(len(ids), size, replace=False); F = pw(X[ix]); G = ext(F, T, O, a)
            vv.append((auc(G[:, T], Yn[ix, T]) - auc(F[:, T], Yn[ix, T])) / 12)
        out[size] = (np.mean(vv), np.std(vv))
    P('== (D) read noise LOA<-MM a %.2f: public 390 sd %.2fe-4 (mean %+.2f), private 910 sd %.2fe-4 (mean %+.2f) macro' % (
        a, out[390][1] * 1e4, out[390][0] * 1e4, out[910][1] * 1e4, out[910][0] * 1e4))
P('SKEP_C12P DONE')
