# regime_dlt_sweep.py (10-21 levers, Claude, REGIME track step 4). RESEARCH ONLY, local CPU, NO OAI. Writes regime_dlt_sweep.log/.json.
# Same 132 directions x a (+.05, +.10, -.05, -.10) on the pw30 base of the four RSNA train OOF rulers as regime.py, but the expert regime
# is now POPULATION-MATCHED: target tet for the train population = report-v2 train tet + D, where D is the same-knee expert-minus-report
# tet difference on gold58 (regime_dlt.json), in four strengths fixed before computing:
#   COP0 : D = 0 (copula control; must track REP)
#   GEN  : trio pairs raw same-knee D; every other pair the mean non-trio D (one generic "experts couple less" shift)
#   EB   : trio raw D; others meanD + rel x (D - meanD), rel = split-half reliability of the D map (regime_dlt.log sec 2)
#   RAW  : every pair its raw same-knee D (mostly noise outside the trio; reported, not trusted)
#   EBg  : EB x gamma_m, gamma_m = the ruler's public calibration on the trio under DLT (regime_dlt.log sec 3)
# plus REP (unraked). Raking set S = {t, o} (+ OA trio when t or o is OA), Gaussian copula with the subset's report marginals.
import json, itertools, time
import numpy as np
from rg_lib import (L, J, AB, OA, eoa, near_pd, copula_cells, weights_from_target, load_all, pct, wauc_multi)
B = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/'
OUT = B + 'regime_dlt_sweep'
log = open(OUT + '.log', 'w')


def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()


dl = json.load(open(B + 'regime_dlt.json')); cal = json.load(open(B + 'regime_cal.json'))
REL = dl['rel']['nontrio']; GAM = dl['gamma']
Trep = np.array(cal['Trep'])
Dm = np.zeros((12, 12))
for k, e in dl['pairs'].items():
    a, b = k.split(' | '); Dm[J[a], J[b]] = Dm[J[b], J[a]] = e['delta']
trio = lambda i, j: L[i] in OA and L[j] in OA
nt = [(i, j) for i in range(12) for j in range(i + 1, 12) if not trio(i, j)]
meanD = float(np.mean([Dm[i, j] for i, j in nt]))
Dgen = np.zeros((12, 12)); Deb = np.zeros((12, 12))
for i in range(12):
    for j in range(12):
        if i == j: continue
        if trio(i, j): Dgen[i, j] = Deb[i, j] = Dm[i, j]
        else: Dgen[i, j] = meanD; Deb[i, j] = meanD + REL * (Dm[i, j] - meanD)
P('rel (non-trio) %.3f | mean non-trio D %+.3f | gamma_m %s' % (REL, meanD, {k: round(v, 2) for k, v in GAM.items()}))
D = load_all(); X, Y = D['X'], D['Y']
BASE = {m: eoa(F, .3, 'pw') for m, F in X.items()}
AV = (.05, .10, -.05, -.10)
REGS = ['REP', 'COP0', 'GEN', 'EB', 'RAW', 'EBg']
MODELS = list(X.keys())


def target_w(Ysub, jc, Dmat, g=1.):
    R = near_pd(Trep[np.ix_(jc, jc)] + g * Dmat[np.ix_(jc, jc)])
    return weights_from_target(Ysub, copula_cells(Ysub.mean(0), R))


t0 = time.time(); RES = {}
for t, o in itertools.permutations(L, 2):
    cols = list(dict.fromkeys(([c for c in OA] if (t in OA or o in OA) else []) + [t, o]))
    jc = [J[c] for c in cols]; m_ = ~np.isnan(Y[:, jc]).any(1); Ysub = Y[m_][:, jc].astype(int); yt = Ysub[:, cols.index(t)]
    Z = np.zeros((12, 12))
    Wc = {'REP': np.ones(len(Ysub)), 'COP0': target_w(Ysub, jc, Z), 'GEN': target_w(Ysub, jc, Dgen), 'EB': target_w(Ysub, jc, Deb),
          'RAW': target_w(Ysub, jc, Dm)}
    d = {'n': int(m_.sum()), 'res': {}}
    for mn in MODELS:
        W = dict(Wc); W['EBg'] = target_w(Ysub, jc, Deb, GAM[mn])
        Bm = BASE[mn][m_]; bt = Bm[:, J[t]]; Ws = [W[r] for r in REGS]
        b0 = wauc_multi(bt, yt, Ws)
        res = np.zeros((len(REGS), len(AV)))
        for ia, a in enumerate(AV):
            res[:, ia] = wauc_multi(pct(pct(bt) - a * pct(Bm[:, J[o]])), yt, Ws) - b0
        for ir, r in enumerate(REGS): d['res'][mn + '|' + r] = res[ir].tolist()
    RES[t + ' <- ' + o] = d
P('sweep done in %.0f s' % (time.time() - t0))
json.dump(dict(AV=AV, REGS=REGS, MODELS=MODELS, RES=RES, meanD=meanD, REL=REL, GAM=GAM), open(OUT + '.json', 'w'))
old = json.load(open(B + 'regime_sweep.json'))['RES']


def short(k):
    t, o = k.split(' <- '); return AB[t] + '<-' + AB[o]


rows = []
for k, d in RES.items():
    for ia, a in enumerate(AV):
        cells = {r: np.array([d['res'][mn + '|' + r][ia] for mn in MODELS]) for r in REGS}
        cells['G74'] = np.array([old[k]['res'][mn + '|G74'][ia] for mn in MODELS])
        rows.append((k, a, cells))
R2 = REGS + ['G74']
ap = lambda c: bool((c > 0).all())
fmt = lambda c: '%s%+6.1f/%+6.1f' % ('+' if (c > 0).all() else ('-' if (c < 0).all() else ' '), c[3] * 1e4, c.min() * 1e4)
P('')
P('cells: sign over 4 rulers (+ all > 0, - all < 0) then 3mean / min-over-rulers, label AUC x1e4 (macro = /12)')
P('%-14s %5s ' % ('direction', 'a') + ' '.join('%-17s' % r for r in R2))
cat = {'A_robust_even_REP': [], 'B_EB_and_RAW': [], 'C_implied_only': []}
for k, a, c in rows:
    if ap(c['REP']) and ap(c['EB']) and ap(c['EBg']): cat['A_robust_even_REP'].append((k, a, c))
    elif ap(c['EB']) and ap(c['RAW']) and ap(c['EBg']): cat['B_EB_and_RAW'].append((k, a, c))
    elif (ap(c['GEN']) or ap(c['EBg']) or ap(c['EB'])) and not ap(c['REP']): cat['C_implied_only'].append((k, a, c))
for nm, lst in cat.items():
    P('-- %s (%d cells)' % (nm, len(lst)))
    for k, a, c in sorted(lst, key=lambda r: -r[2]['EB'][3])[:45]:
        P('%-14s %+5.2f ' % (short(k), a) + ' '.join('%-17s' % fmt(c[r]) for r in R2) + ' | n %d' % RES[k]['n'])
P('')
for r in R2:
    P('regime %-5s: all-4-positive decorrelation %3d / 264, borrow %3d / 264' % (
        r, sum(1 for k, a, c in rows if ap(c[r]) and a > 0), sum(1 for k, a, c in rows if ap(c[r]) and a < 0)))
P('REGIME_DLT_SWEEP DONE')
