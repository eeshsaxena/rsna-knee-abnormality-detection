# regime_dlt.py (10-21 levers, Claude, REGIME track step 3). RESEARCH ONLY, local CPU, NO OAI. Writes regime_dlt.log + regime_dlt.json.
# Problem with raking train labels to the RAW gold74 coupling (regime_sweep G74 / I_FT): gold74 is a different case mix (less OA, more
# fracture / ACL), and the models' own co-firing moves with the case mix (CLAIMED census.md sec 5: R7 MOA-Cont +.01 train vs -.49 gold74).
# Raking imposes the gold case-mix coupling on train-population predictions and can manufacture a gap. The population-free quantity is
# the SAME-KNEE regime difference: on the gold58 knees (expert-read; the 16 extras are report-derived, CLAIMED memory 10-03) compare the
# expert tet with the report-v2 tet. Target for the train population: tet_DLT = tet_rep(train) + gamma x [tet_exp(g58) - tet_rep(g58)].
# (1) per pair: tet_rep train, tet_rep g58 (same knees), tet_exp g58 / g74, Delta; R7 OOF co-firing rho_ns on train vs gold58 vs gold74
#     (population drift of the MODEL), c4 rho_ns on gold58.
# (2) split-half reliability of the Delta map across the 66 pairs (random halves of gold58, Spearman-Brown to 58), and of the
#     population drift (report tet g58 - train).
# (3) trio calibration under DLT(gamma): which gamma reproduces the five public E-OA reads per ruler.
import json
import numpy as np
from rg_lib import (L, J, AB, OA, PUB, SH, eoa, wauc, tet_xy, near_pd, copula_cells, weights_from_target, load_all, ns, LD)
OUT = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/regime_dlt'
log = open(OUT + '.log', 'w')


def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()


D = load_all(); X, Y, Y74, c4 = D['X'], D['Y'], D['Y74'], D['c4']
g, g58 = LD.gold(); ids58 = g58.StudyInstanceUID.tolist(); ids74 = g.StudyInstanceUID.tolist()
E58 = g58[L].values.astype(float); lv2 = LD.labels('v2'); lv4 = LD.labels('v4')
R58 = lv2.reindex(ids58)[L].values.astype(float); R58v4 = lv4.reindex(ids58)[L].values.astype(float)
r7 = LD.r7(); R7c = ['p|' + c for c in L]
R7g58 = r7.loc[ids58, R7c].values.astype(float); R7g74 = r7.loc[ids74, R7c].values.astype(float)
rho = lambda F, i, j: float(np.corrcoef(ns(F[:, i]), ns(F[:, j]))[0, 1])
pairs = {}
for i in range(12):
    for j in range(i + 1, 12):
        e = dict(rep_tr=tet_xy(Y[:, i], Y[:, j]), rep_g58=tet_xy(R58[:, i], R58[:, j]), rep4_g58=tet_xy(R58v4[:, i], R58v4[:, j]),
                 exp_g58=tet_xy(E58[:, i], E58[:, j]), exp_g74=tet_xy(Y74[:, i], Y74[:, j]),
                 n_rep_g58=int((~np.isnan(R58[:, i]) & ~np.isnan(R58[:, j])).sum()),
                 r7_tr=rho(X['R7'], i, j), r7_g58=rho(R7g58, i, j), r7_g74=rho(R7g74, i, j), c4_g58=rho(c4, i, j))
        e['delta'] = e['exp_g58'] - e['rep_g58']; e['delta4'] = e['exp_g58'] - e['rep4_g58']
        e['popdrift_lab'] = e['rep_g58'] - e['rep_tr']; e['popdrift_r7'] = e['r7_g58'] - e['r7_tr']
        pairs[L[i] + ' | ' + L[j]] = e
P('== (1) same-knee regime difference Delta = expert tet - report-v2 tet on gold58 (n per pair = non-NaN v2), vs raw gold74 - train gap')
P('%-10s %6s %6s %6s %6s | %6s %6s | %6s %6s %6s %6s | %s' % ('pair', 'repTr', 'repG58', 'expG58', 'expG74', 'DELTA', 'D(v4)', 'r7Tr', 'r7G58', 'r7G74', 'c4G58', 'n'))
for k, e in sorted(pairs.items(), key=lambda kv: kv[1]['delta']):
    a, b = k.split(' | ')
    P('%-10s %+6.2f %+6.2f %+6.2f %+6.2f | %+6.2f %+6.2f | %+6.2f %+6.2f %+6.2f %+6.2f | %d' % (
        AB[a] + '-' + AB[b], e['rep_tr'], e['rep_g58'], e['exp_g58'], e['exp_g74'], e['delta'], e['delta4'], e['r7_tr'], e['r7_g58'], e['r7_g74'],
        e['c4_g58'], e['n_rep_g58']))
ks = list(pairs)
A = lambda f: np.array([pairs[k][f] for k in ks])
P('')
P('corr across 66 pairs: raw gap (expG74 - repTr) vs DELTA %.2f | popdrift labels (repG58 - repTr) vs popdrift R7 (r7G58 - r7Tr) %.2f | '
  'DELTA v2 vs v4 %.2f | c4G58 - expG58 vs DELTA %.2f' % (np.corrcoef(A('exp_g74') - A('rep_tr'), A('delta'))[0, 1],
                                                           np.corrcoef(A('popdrift_lab'), A('popdrift_r7'))[0, 1], np.corrcoef(A('delta'), A('delta4'))[0, 1],
                                                           np.corrcoef(A('c4_g58') - A('exp_g58'), A('delta'))[0, 1]))
# (2) split-half reliability
rng = np.random.default_rng(2110)
sh_d, sh_p = [], []
for _ in range(300):
    pp = rng.permutation(58); h1, h2 = pp[:29], pp[29:]
    d1 = np.array([tet_xy(E58[h1, J[k.split(' | ')[0]]], E58[h1, J[k.split(' | ')[1]]]) - tet_xy(R58[h1, J[k.split(' | ')[0]]], R58[h1, J[k.split(' | ')[1]]]) for k in ks])
    d2 = np.array([tet_xy(E58[h2, J[k.split(' | ')[0]]], E58[h2, J[k.split(' | ')[1]]]) - tet_xy(R58[h2, J[k.split(' | ')[0]]], R58[h2, J[k.split(' | ')[1]]]) for k in ks])
    sh_d.append(np.corrcoef(d1, d2)[0, 1])
r = float(np.mean(sh_d)); sb = 2 * r / (1 + r) if r > -1 else float('nan')
P('(2) split-half reliability of the DELTA map across 66 pairs: mean r %.3f (P(r>0) %.2f) -> Spearman-Brown to n 58: %.3f' % (r, np.mean(np.array(sh_d) > 0), sb))
P('    => EB shrink factor for a pair DELTA ~ %.2f (reliability); non-trio pairs only below' % sb)
sh_n = []
nt = [k for k in ks if not (k.split(' | ')[0] in OA and k.split(' | ')[1] in OA)]
for _ in range(300):
    pp = rng.permutation(58); h1, h2 = pp[:29], pp[29:]
    f = lambda h, k: tet_xy(E58[h, J[k.split(' | ')[0]]], E58[h, J[k.split(' | ')[1]]]) - tet_xy(R58[h, J[k.split(' | ')[0]]], R58[h, J[k.split(' | ')[1]]])
    sh_n.append(np.corrcoef([f(h1, k) for k in nt], [f(h2, k) for k in nt])[0, 1])
r2 = float(np.mean(sh_n)); P('    non-trio 63 pairs: mean r %.3f -> SB %.3f' % (r2, 2 * r2 / (1 + r2)))

# (3) trio calibration under DLT(gamma)
jOA = [J[c] for c in OA]; m = ~np.isnan(Y[:, jOA]).any(1); Ym = Y[m][:, jOA].astype(int)
Trep = np.eye(3); Dl = np.zeros((3, 3))
for x in range(3):
    for y in range(x + 1, 3):
        k = OA[x] + ' | ' + OA[y]; Trep[x, y] = Trep[y, x] = pairs[k]['rep_tr']; Dl[x, y] = Dl[y, x] = pairs[k]['delta']
P('')
P('(3) trio DLT: report train tet %s | same-knee DELTA %s (v4 %s)' % (
    ' '.join('%.2f' % pairs[OA[x] + ' | ' + OA[y]]['rep_tr'] for x, y in ((0, 1), (0, 2), (1, 2))),
    ' '.join('%+.2f' % pairs[OA[x] + ' | ' + OA[y]]['delta'] for x, y in ((0, 1), (0, 2), (1, 2))),
    ' '.join('%+.2f' % pairs[OA[x] + ' | ' + OA[y]]['delta4'] for x, y in ((0, 1), (0, 2), (1, 2)))))
pubv = np.array([PUB[k] for k in SH]); GR = [0., .5, 1., 1.25, 1.5, 1.75, 2., 2.5, 3.]
calg = {}
for mn, F in X.items():
    Fm = F[m]; G = {k: eoa(Fm, a, md) for k, (a, md) in SH.items()}
    def gains(w):
        return np.array([sum(wauc(G[k][:, J[c]], Ym[:, ii], w) - wauc(Fm[:, J[c]], Ym[:, ii], w) for ii, c in enumerate(OA)) / 12 * 1e4 for k in SH])
    cur = {}
    for gm in GR:
        R = near_pd(Trep + gm * Dl); w = weights_from_target(Ym, copula_cells(Ym.mean(0), R)); cur[gm] = gains(w)
        P('  %-6s gamma %.2f: tet %s -> %s | sse %.1f' % (mn, gm, ' '.join('%.2f' % R[x, y] for x, y in ((0, 1), (0, 2), (1, 2))),
                                                         ' '.join('%+.1f' % v for v in cur[gm]), ((cur[gm] - pubv) ** 2).sum()))
    fine = np.linspace(0, 3, 301); gf = np.array([[np.interp(x, GR, [cur[q][i] for q in GR]) for i in range(5)] for x in fine])
    sse = ((gf - pubv) ** 2).sum(1); calg[mn] = float(fine[np.argmin(sse)])
    P('  %-6s -> gamma_m %.2f (sse %.1f)' % (mn, calg[mn], sse.min()))
json.dump(dict(pairs=pairs, rel=dict(all=sb, nontrio=2 * r2 / (1 + r2)), gamma=calg), open(OUT + '.json', 'w'), indent=1)
P('REGIME_DLT DONE')
