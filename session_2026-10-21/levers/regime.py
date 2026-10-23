# regime.py (10-21 levers, Claude, REGIME track step 2). RESEARCH ONLY, local CPU, NO OAI. Writes regime_sweep.log + regime_sweep.json.
# For all 132 directions t <- o (66 pairs) and a in (+.05, +.10, -.05, -.10): out_t = pct(pct(B_t) - a pct(B_o)) on the pw30 base
# B = eoa(F, .3, 'pw') of each RSNA train OOF ruler F (R7, s1044, s1045, 3mean; n 3,467 non-gold), scored as the change in the weighted
# AUC of label t against report-v2 labels raked to a target expert regime. Raking set S = {t, o} plus the OA trio whenever t or o is an
# OA label (the base already mixes the trio). Regimes (fixed BEFORE computing, from regime_cal.log):
#   REP   : unraked report labels (mechanism control: a decorrelation that only fixes report-inherited coupling must LOSE here)
#   IPF   : gold74 joint (+.5) IPF-raked to the subset marginals (adj_c12 / stack RAK pattern), lambda 1
#   G74   : Gaussian copula, report marginals, tet = gold74 tet (lambda 1)
#   I_FT  : tet = .713 x gold74 (trio-implied label shift -0.83 sd, the P_FT gap-matched reading; regime_cal sec 2)
#   I_AR  : tet = .568 x gold74 (-1.21 sd, P_AREA)
#   RC    : tet = lambda_m x gold74 with lambda_m the ruler's own public calibration (R7 .57, s1044 .82, s1045 .77, 3mean .99)
#   A_FT  : sensitivity: every pair's gold74 log OR shifted by -0.83 x its own se (not a common shrink), copula
import json, itertools, time
import numpy as np
from rg_lib import (L, J, AB, OA, eoa, wauc, near_pd, lor_to_tet, copula_cells, weights_from_target, weights_ipf_g74, load_all,
                    counts_xy, pct)
OUT = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/regime_sweep'
log = open(OUT + '.log', 'w')


def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()


cal = json.load(open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/regime_cal.json'))
T74 = np.array(cal['T74'])
LAMM = {m: cal['cal'][m]['lam_fit'] for m in cal['cal']}
LFT, LAR = cal['lam']['FT'], cal['lam']['AREA']
P('lambda: FT %.3f AREA %.3f RC %s' % (LFT, LAR, {k: round(v, 2) for k, v in LAMM.items()}))
D = load_all(); X, Y, Y74 = D['X'], D['Y'], D['Y74']
BASE = {m: eoa(F, .3, 'pw') for m, F in X.items()}
# per-pair k-shifted tet (A_FT)
TA = np.eye(12)
for i in range(12):
    for j in range(i + 1, 12):
        cs = counts_xy(Y74[:, i], Y74[:, j]) + .5; lor = np.log(cs[3] * cs[0] / (cs[1] * cs[2])); se = np.sqrt((1 / cs).sum())
        TA[i, j] = TA[j, i] = lor_to_tet(np.nanmean(Y74[:, i]), np.nanmean(Y74[:, j]), lor - .83 * se)
AV = (.05, .10, -.05, -.10)
REGS = ['REP', 'IPF', 'G74', 'I_FT', 'I_AR', 'RC', 'A_FT']
MODELS = list(X.keys())
WC = {}


def get_w(cols, reg, m, Ysub):
    key = (tuple(cols), reg if reg != 'RC' else 'RC' + m)
    if key in WC: return WC[key]
    jc = [J[c] for c in cols]
    if reg == 'REP': w = np.ones(len(Ysub))
    elif reg == 'IPF': w = weights_ipf_g74(Ysub, cols, Y74)
    else:
        if reg == 'A_FT': R = near_pd(TA[np.ix_(jc, jc)])
        else:
            lam = {'G74': 1., 'I_FT': LFT, 'I_AR': LAR}.get(reg, LAMM.get(m))
            R0 = near_pd(T74[np.ix_(jc, jc)]); R = np.eye(len(jc)) + lam * (R0 - np.eye(len(jc)))
        w = weights_from_target(Ysub, copula_cells(Ysub.mean(0), R))
    WC[key] = w; return w


t0 = time.time()
RES = {}
for t, o in itertools.permutations(L, 2):
    cols = list(dict.fromkeys(([c for c in OA] if (t in OA or o in OA) else []) + [t, o]))
    jc = [J[c] for c in cols]; m_ = ~np.isnan(Y[:, jc]).any(1); Ysub = Y[m_][:, jc].astype(int); yt = Ysub[:, cols.index(t)]
    d = {'n': int(m_.sum()), 'cols': cols, 'res': {}}
    for mn in MODELS:
        B = BASE[mn][m_]; bt = B[:, J[t]]
        outs = {a: pct(pct(bt) - a * pct(B[:, J[o]])) for a in AV}
        for reg in REGS:
            w = get_w(cols, reg, mn, Ysub); b0 = wauc(bt, yt, w)
            d['res'][mn + '|' + reg] = [wauc(outs[a], yt, w) - b0 for a in AV]
    RES[t + ' <- ' + o] = d
P('sweep done in %.0f s, %d directions' % (time.time() - t0, len(RES)))
json.dump(dict(AV=AV, REGS=REGS, MODELS=MODELS, RES=RES), open(OUT + '.json', 'w'))

# ---- summary: sign consistency across rulers per regime (label AUC units)
P('')
P('== sign-consistency table (label AUC delta; + = all 4 rulers > 0; numbers = 3mean delta x1e4 label / min over 4 rulers x1e4)')
hdr = '%-14s %5s ' % ('direction', 'a') + ' '.join('%-17s' % r for r in REGS)
P(hdr)
rows = []
for k, d in RES.items():
    for ia, a in enumerate(AV):
        cells = {}
        for reg in REGS:
            v = np.array([d['res'][mn + '|' + reg][ia] for mn in MODELS]); cells[reg] = v
        allpos = {reg: bool((cells[reg] > 0).all()) for reg in REGS}
        rows.append((k, a, cells, allpos))
# flagged categories
CP = [r for r in rows if r[3]['G74'] and r[3]['IPF'] and r[3]['I_FT'] and r[3]['RC']]
IO = [r for r in rows if (r[3]['I_FT'] or r[3]['I_AR'] or r[3]['RC']) and not (r[3]['G74'] and r[3]['IPF'])]
fmt = lambda c: '%s%+6.1f/%+6.1f' % ('+' if (c > 0).all() else ('-' if (c < 0).all() else ' '), c[3] * 1e4, c.min() * 1e4)


def short(k):
    t, o = k.split(' <- '); return AB[t] + '<-' + AB[o]


P('-- CP: all 4 rulers > 0 at IPF, G74, I_FT and RC (%d direction x a cells)' % len(CP))
for k, a, cells, ap in sorted(CP, key=lambda r: -r[2]['G74'].mean()):
    P('%-14s %+5.2f ' % (short(k), a) + ' '.join('%-17s' % fmt(cells[r]) for r in REGS) + ' | n %d' % RES[k]['n'])
P('-- IO: all 4 > 0 only at a lambda < 1 regime (I_FT / I_AR / RC), not at both gold74 regimes (%d cells)' % len(IO))
for k, a, cells, ap in sorted(IO, key=lambda r: -r[2]['I_FT'].mean())[:60]:
    P('%-14s %+5.2f ' % (short(k), a) + ' '.join('%-17s' % fmt(cells[r]) for r in REGS) + ' | n %d' % RES[k]['n'])
# per-regime counts of all-positive cells, split by decorrelation / borrow
P('')
for reg in REGS:
    nd = sum(1 for r in rows if r[3][reg] and r[1] > 0); nb = sum(1 for r in rows if r[3][reg] and r[1] < 0)
    P('regime %-5s: all-4-positive cells decorrelation %3d / 264, borrow %3d / 264' % (reg, nd, nb))
P('REGIME_SWEEP DONE')
