# regime_cal.py (10-21 levers, Claude, REGIME track step 1). RESEARCH ONLY, local CPU, NO OAI (OAI tet values are read from the
# aggregate JSON already on the laptop: session_2026-10-19/ex/box_coupling.log JSON block). Writes regime_cal.log + regime_cal.json.
# (1) all 66 pairwise couplings: report-v2 tet (train tr3), gold74 tet (+.5), gold74 log OR + se, OAI B1 tet (36 pairs), c4 label-free
#     rho_ns on gold58 (raw c4 and pw30), OOF model rho_ns on tr3 (pw30 base).
# (2) trio-implied common shrink lambda: gold74 trio log OR shifted by k x se (k -0.83 = P_FT gap-matched, -1.21 = P_AREA, -0.17 = c4-rho
#     label-side; CLAIMED night/recal.md sec 0.2), converted to tetrachoric at gold74 marginals; lambda = tet_implied / tet_gold74.
# (3) per-ruler calibration: E-OA trio gains of each OOF ruler (R7, s1044, s1045, 3mean) with report labels raked (Gaussian copula,
#     report marginals, tet = lambda x gold74) over a lambda grid, vs the five public reads; lambda_m = least-squares fit.
import json
import numpy as np
from rg_lib import (L, J, AB, OA, PUB, SH, eoa, wauc, tet_matrix, tet_counts, near_pd, lor_to_tet, copula_cells,
                    weights_from_target, weights_ipf_g74, load_all, counts_xy, ns)
OUT = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/regime_cal'
log = open(OUT + '.log', 'w')


def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()


D = load_all(); X, Y, Y74, c4 = D['X'], D['Y'], D['Y74'], D['c4']
P('rows tr3 %d | gold74 %d | gold58 %d' % (len(D['tr3']), len(Y74), len(D['Y58'])))
T74 = tet_matrix(Y74); Trep = tet_matrix(Y)
oai = json.loads(open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/box_coupling.log').read().split('JSON_BEGIN')[1].strip().split('\n')[0])
c4pw = eoa(c4, .3, 'pw')
rho = lambda F, i, j: float(np.corrcoef(ns(F[:, i]), ns(F[:, j]))[0, 1])
pairs = {}
for i in range(12):
    for j in range(i + 1, 12):
        a, b = L[i], L[j]; cs = counts_xy(Y74[:, i], Y74[:, j]) + .5
        lor = float(np.log(cs[3] * cs[0] / (cs[1] * cs[2]))); se = float(np.sqrt((1 / cs).sum()))
        csr = counts_xy(Y[:, i], Y[:, j]) + .5; lorr = float(np.log(csr[3] * csr[0] / (csr[1] * csr[2])))
        ko = a + ' | ' + b if (a + ' | ' + b) in oai else (b + ' | ' + a if (b + ' | ' + a) in oai else None)
        e = dict(tet74=T74[i, j], lor74=lor, se74=se, n74=int(cs.sum() - 2), tetrep=Trep[i, j], lorrep=lorr, nrep=int(csr.sum() - 2),
                 tetoai=(oai[ko]['tet'] if ko else None), rho_c4=rho(c4, i, j), rho_c4pw=rho(c4pw, i, j),
                 rho_oof={m: rho(eoa(F, .3, 'pw'), i, j) for m, F in X.items()})
        pairs[a + ' | ' + b] = e
P('')
P('== (1) couplings: tet report v2 (tr3) / gold74 (+.5) / OAI B1 | c4 gold58 rho_ns raw / pw30 | OOF rho_ns pw30 base R7 s1044 s1045 3mean')
for k, e in sorted(pairs.items(), key=lambda kv: -(kv[1]['tetrep'] - kv[1]['tet74'])):
    a, b = k.split(' | ')
    P('%-10s rep %+.2f (n %4d) g74 %+.2f (lor %+.2f se %.2f) oai %s | c4 %+.2f/%+.2f | oof %s | rep-g74 %+.2f c4pw-g74 %+.2f' % (
        AB[a] + '-' + AB[b], e['tetrep'], e['nrep'], e['tet74'], e['lor74'], e['se74'],
        ('%+.2f' % e['tetoai']) if e['tetoai'] is not None else '  na ', e['rho_c4'], e['rho_c4pw'],
        ' '.join('%+.2f' % v for v in e['rho_oof'].values()), e['tetrep'] - e['tet74'], e['rho_c4pw'] - e['tet74']))

# (2) trio-implied lambda
P('')
P('== (2) trio-implied common shrink (k x se of the gold74 log OR -> tet at gold74 marginals)')
LAM = {}
for kk, nm in ((-0.17, 'c4rho'), (-0.5, 'k-0.5'), (-0.83, 'FT'), (-1.0, 'k-1.0'), (-1.21, 'AREA'), (-1.5, 'k-1.5')):
    rs = []
    for a, b in (('Medial OA', 'Lateral OA'), ('Medial OA', 'PF OA'), ('Lateral OA', 'PF OA')):
        e = pairs[a + ' | ' + b]; i, j = J[a], J[b]
        p1 = np.nanmean(Y74[:, i]); p2 = np.nanmean(Y74[:, j])
        ti = lor_to_tet(p1, p2, e['lor74'] + kk * e['se74']); t0 = lor_to_tet(p1, p2, e['lor74'])
        rs.append((ti, t0, ti / t0))
    LAM[nm] = float(np.mean([r[2] for r in rs]))
    P('  k %+.2f (%-5s): trio tet implied %s (gold74 %s) -> lambda %.3f (pairs %s)' % (
        kk, nm, ' '.join('%.2f' % r[0] for r in rs), ' '.join('%.2f' % r[1] for r in rs), LAM[nm], ' '.join('%.2f' % r[2] for r in rs)))
P('  report / gold74 trio tet ratio: %s' % ' '.join('%.2f' % (pairs[p]['tetrep'] / pairs[p]['tet74']) for p in
                                                   ('Medial OA | Lateral OA', 'Medial OA | PF OA', 'Lateral OA | PF OA')))

# (3) per-ruler calibration on the E-OA trio
P('')
P('== (3) E-OA trio gains (e-4 macro vs raw ruler) under Gaussian-copula raking, tet = lambda x gold74; public %s' % PUB)
jOA = [J[c] for c in OA]; m = ~np.isnan(Y[:, jOA]).any(1); Ym = Y[m][:, jOA].astype(int)
R74 = near_pd(T74[np.ix_(jOA, jOA)])
GRID = [1.3, 1.15, 1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.]
cal = {}
pubv = np.array([PUB[k] for k in SH])
for mn, F in X.items():
    Fm = F[m]; base = {c: None for c in OA}
    G = {k: eoa(Fm, a, md) for k, (a, md) in SH.items()}
    def gains(w):
        return np.array([sum(wauc(G[k][:, J[c]], Ym[:, ii], w) - wauc(Fm[:, J[c]], Ym[:, ii], w) for ii, c in enumerate(OA)) / 12 * 1e4 for k in SH])
    w1 = np.ones(len(Ym)); g_rep = gains(w1)
    wi = weights_ipf_g74(Ym, OA, Y74); g_ipf = gains(wi)
    P('  %-6s REP %s | IPF-g74 %s' % (mn, ' '.join('%+.1f' % v for v in g_rep), ' '.join('%+.1f' % v for v in g_ipf)))
    curve = {}
    for lam in GRID:
        R = np.eye(3) + lam * (R74 - np.eye(3)); tgt = copula_cells(Ym.mean(0), R)
        w = weights_from_target(Ym, tgt); g = gains(w); curve[lam] = g.tolist()
        P('    lambda %.2f: %s | ratio pub/ruler %s | sse %.1f' % (lam, ' '.join('%+.1f' % v for v in g),
                                                                   ' '.join('%.2f' % (p / v) if v > 0 else ' neg' for p, v in zip(pubv, g)), ((g - pubv) ** 2).sum()))
    lams = np.array(GRID); sse = np.array([((np.array(curve[l]) - pubv) ** 2).sum() for l in GRID])
    # refine by linear interpolation of gains on a fine grid
    fine = np.linspace(0, 1.3, 131); gf = np.array([[np.interp(x, lams[::-1], np.array([curve[l][q] for l in GRID])[::-1]) for q in range(5)] for x in fine])
    ssef = ((gf - pubv) ** 2).sum(1); lm = float(fine[np.argmin(ssef)])
    # scale-free fit: best lambda if the ruler is allowed a free amplitude s (fits shape only)
    shp = []
    for q, x in enumerate(fine):
        g = gf[q]; s = float((g @ pubv) / max(g @ g, 1e-9)); shp.append(((s * g - pubv) ** 2).sum() if s > 0 else 1e9)
    lms = float(fine[int(np.argmin(shp))])
    cal[mn] = dict(rep=g_rep.tolist(), ipf=g_ipf.tolist(), curve={str(k): v for k, v in curve.items()}, lam_fit=lm, sse_min=float(ssef.min()),
                   lam_shape=lms)
    P('    -> lambda_m (level fit) %.2f (sse %.1f) | lambda (shape-only fit, free amplitude) %.2f' % (lm, ssef.min(), lms))
json.dump(dict(pairs=pairs, lam=LAM, cal=cal, T74=T74.tolist(), Trep=Trep.tolist()), open(OUT + '.json', 'w'), indent=1)
P('REGIME_CAL DONE')
