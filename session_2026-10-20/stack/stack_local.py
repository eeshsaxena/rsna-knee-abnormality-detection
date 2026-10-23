# stack_local.py (10-20 ~22:45 IST, Claude, STACK inventory). RESEARCH ONLY, local CPU, NO OAI, nothing written outside stack/.
# Pairwise interactions among the E >= 0 candidate components on top of pw30, on three local rulers:
#   G58  = gold58 (n 58) c4 kernel path from the member serve files (INFO only, never a selection ruler);
#   RAK  = RSNA train OOF (R7 / s1044 / s1045) in c4-shaped nests, report-v2 labels raked by IPF to the gold74 EXPERT joint
#          (OA labels: 32-cell (MOA, LOA, PF, MM, LM) joint +.5; sensitivity 16-cell (MOA, LOA, PF, MM); MCL: (MCL, MM) 2x2;
#          other labels unraked = report OOF); same rows / weights for every variant, so differences are paired;
#   REP  = the same nests on raw report-v2 labels (w = 1): mechanism control (decorrelation MUST lose here).
# Components (one change each on pw30 = c4 + A21 pw E-OA a .3):
#   pw25   A_OA .3 -> .25 (cell 18)                         lomm05 LOA = pct(pct(LOA) - .05 pct(MM)) after E-OA (cell 18)
#   S6up   W_S6F .2 -> .3 on the OA trio only (cell 18)       SUBH   E-OA subtracts pct(2/3 pct(R6) + 1/3 pct(S6fall)) of the
#   RPF    W_RAPTOR_C['PF OA'] .5 -> .4 (cell 14)                    other compartments instead of the final's (cell 18)
#   BOR    MOA = pct(pct(MOA) + .1 pct(MM)); LOA = pct(pct(LOA) + .1 pct(LM)) after E-OA (C13; its LOA half = killed C14)
#   BORm   MOA half of BOR only (MOA += .1 MM)               MCLmm  MCL = pct(pct(MCL) - .1 pct(MM)) (cell 18)
#   E1     soft-OR: every member pct ** 2 before mixing (cells 14 + 18)
# Stack order inside cell 18 when combined: stage 3 -> S6 blend -> E-OA (pw / SUBH) -> lomm05 -> BOR -> MCLmm.
# Writes stack/stack_local.log and stack/stack_local.json. Aggregates only.
import sys, json, itertools, time
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
import load as LD
T0 = time.time()
OUT = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/stack_local'
log = open(OUT + '.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
L = LD.L; J = {c: i for i, c in enumerate(L)}; G = LD.G
OA = ['Medial OA', 'Lateral OA', 'PF OA']
pct = lambda v: rankdata(v) / len(v)
C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
W_PW = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}
BASE = dict(a=.3, mode='pw', e1=1., ws6_oa=.2, wr_pf=.5, sub='lin', loamm=0., bor_m=0., bor_l=0., mclmm=0.)
COMP = {'pw25': dict(a=.25), 'lomm05': dict(loamm=.05), 'S6up': dict(ws6_oa=.3), 'SUBH': dict(sub='high'), 'RPF': dict(wr_pf=.4),
        'BOR': dict(bor_m=.1, bor_l=.1), 'MCLmm': dict(mclmm=.1), 'E1': dict(e1=2.)}
EXTRA = {'BORm': dict(bor_m=.1), 'c4': dict(a=0.), 'eoa10': dict(a=.1, mode='sym'), 'eoa20': dict(a=.2, mode='sym'),
         'eoa30': dict(a=.3, mode='sym'), 'pw20': dict(a=.2), 'pw35': dict(a=.35), 'RPF35': dict(wr_pf=.35), 'S6up4': dict(ws6_oa=.4),
         'lomm10': dict(loamm=.1)}
PUB = {'eoa10': 5.9, 'eoa20': 10.3, 'eoa30': 9.6, 'pw20': 10.4, 'pw30': 11.3}   # public vs c4, e-4 (CLAIMED val/implied2.md)
f32 = lambda x: np.asarray(x, np.float64).astype(np.float32).astype(np.float64)


def pipeline(M, cfg):
    """M: rap = [(share, X)], r6 = X, s6 = X (n x 12 raw member scores). Kernel order and float32 casts as c4 cells 14 / 18."""
    e1 = cfg['e1']; Pw = lambda v: pct(v) ** e1; n = M['r6'].shape[0]
    V = np.zeros((n, 12))
    for j, c in enumerate(L):
        wr = cfg['wr_pf'] if c == 'PF OA' else C4W.get(c, .5)
        rr = sum(s * Pw(X[:, j]) for s, X in M['rap'])
        V[:, j] = wr * rr + (1 - wr) * Pw(M['r6'][:, j])
    V = f32(V); F = np.zeros_like(V)
    for j, c in enumerate(L):
        ws = cfg['ws6_oa'] if c in OA else .2
        F[:, j] = (1 - ws) * Pw(V[:, j]) + ws * Pw(M['s6'][:, j])
    F = f32(F); out = F.copy()
    a = cfg['a']
    if a:
        U = {c: pct(F[:, J[c]]) for c in OA}
        if cfg['sub'] == 'high':
            H = {c: pct(2 / 3 * pct(M['r6'][:, J[c]]) + 1 / 3 * pct(M['s6'][:, J[c]])) for c in OA}
        else:
            H = U
        for c in OA:
            o = [k for k in OA if k != c]
            w = [W_PW[tuple(sorted((c, k), key=OA.index))] for k in o] if cfg['mode'] == 'pw' else [1., 1.]
            out[:, J[c]] = f32(pct(U[c] - a / 2.0 * (w[0] * H[o[0]] + w[1] * H[o[1]])))
    def line(t, o, b):
        out[:, J[t]] = f32(pct(pct(out[:, J[t]]) - b * pct(out[:, J[o]])))
    if cfg['loamm']:
        line('Lateral OA', 'Medial Meniscus', cfg['loamm'])
    if cfg['bor_m']:
        line('Medial OA', 'Medial Meniscus', -cfg['bor_m'])
    if cfg['bor_l']:
        line('Lateral OA', 'Lateral Meniscus', -cfg['bor_l'])
    if cfg['mclmm']:
        line('MCL', 'Medial Meniscus', cfg['mclmm'])
    return out


def variants():
    V = {'pw30': dict(BASE)}
    for k, d in list(COMP.items()) + list(EXTRA.items()):
        c = dict(BASE); c.update(d); V[k] = c
    for x, y in itertools.combinations(COMP, 2):
        c = dict(BASE); c.update(COMP[x]); c.update(COMP[y]); V[x + '+' + y] = c
    return V


VAR = variants()
PAIRS = list(itertools.combinations(COMP, 2))


def wauc(s, y, w):
    o = np.argsort(s, kind='mergesort'); s, y, w = s[o], y[o], w[o]
    wp = w * y; wn = w * (1 - y)
    _, inv = np.unique(s, return_inverse=True)
    gp = np.bincount(inv, wp); gn = np.bincount(inv, wn); cn = np.cumsum(gn) - gn
    return float((gp * (cn + .5 * gn)).sum() / (wp.sum() * wn.sum()))


def rake(T, margins, it=300):
    T = T.copy(); k = T.ndim
    for _ in range(it):
        for ax in range(k):
            cur = T.sum(axis=tuple(i for i in range(k) if i != ax)); sh = [1] * k; sh[ax] = 2
            T = T * (np.array([1 - margins[ax], margins[ax]]) / cur).reshape(sh)
    return T / T.sum()


def rk_weights(Ysub, cols, Y74):
    jc = [J[c] for c in cols]; ok = ~np.isnan(Y74[:, jc]).any(1); Yg = Y74[ok][:, jc].astype(int)
    T = np.zeros((2,) * len(cols)); obs = np.zeros((2,) * len(cols))
    for r in Yg: T[tuple(r)] += 1
    for r in Ysub: obs[tuple(r)] += 1
    obs /= obs.sum(); tgt = rake(T + .5, Ysub.mean(0)); w = (tgt / obs)[tuple(Ysub.T)]
    return w / w.mean()


def evaluate(name, outs, Yfull, WT, nb, rng, macro_div=12.):
    """outs: variant -> n x 12. WT: label -> (mask, w) (w over masked rows). Returns per-variant macro delta vs pw30 (e-4) + boot."""
    base = outs['pw30']; n = base.shape[0]
    changed = {k: [j for j in range(12) if not np.array_equal(outs[k][:, j], base[:, j])] for k in outs}
    pt = {}
    for k in outs:
        d = 0.
        for j in changed[k]:
            m, w = WT[L[j]]; y = Yfull[m, j]
            d += wauc(outs[k][m, j], y, w) - wauc(base[m, j], y, w)
        pt[k] = d / macro_div * 1e4
    bs = {k: [] for k in outs}
    for _ in range(nb):
        ii = rng.integers(0, n, n); cache = {}
        for k in outs:
            d = 0.
            for j in changed[k]:
                m, w = WT[L[j]]; sel = ii[m[ii]]
                wi = WMAP[name][L[j]][sel]; y = Yfull[sel, j]
                if (j, 'b') not in cache:
                    cache[(j, 'b')] = wauc(base[sel, j], y, wi)
                d += wauc(outs[k][sel, j], y, wi) - cache[(j, 'b')]
            bs[k].append(d / macro_div * 1e4)
    return pt, {k: np.array(v) for k, v in bs.items()}, changed


WMAP = {}


def report(tag, pt, bs, extra_ref=True):
    P('  -- %s singles (e-4 macro vs pw30; boot sd, P>0) --' % tag)
    for k in list(COMP) + ['BORm', 'RPF35', 'S6up4', 'lomm10', 'pw20', 'pw35']:
        P('    %-8s %+7.2f  sd %5.2f  P>0 %.2f' % (k, pt[k], bs[k].std(), (bs[k] > 0).mean()))
    P('  -- %s E-OA ladder vs c4 (calibration; public in parentheses) --' % tag)
    lad = {k: pt[k] - pt['c4'] for k in ('eoa10', 'eoa20', 'eoa30', 'pw20', 'pw30')}
    P('    ' + ' | '.join('%s %+.2f (%.1f, ratio %s)' % (k, v, PUB[k], ('%.2f' % (PUB[k] / v)) if v > 0 else 'n/a') for k, v in lad.items()))
    P('  -- %s pairwise: delta(A+B), interaction I = d(AB) - d(A) - d(B) with boot sd and P(I>0) --' % tag)
    res = {}
    for x, y in PAIRS:
        k = x + '+' + y; I = bs[k] - bs[x] - bs[y]; ip = pt[k] - pt[x] - pt[y]
        res[k] = dict(d=pt[k], I=ip, sd=float(I.std()), P=float((I > 0).mean()), dsum=pt[x] + pt[y])
        P('    %-14s d %+7.2f  sum %+7.2f  I %+6.2f (sd %4.2f, P>0 %.2f)' % (k, pt[k], pt[x] + pt[y], ip, I.std(), (I > 0).mean()))
    return res, lad


summary = {}
# ============================== G58 ==============================
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}); g58 = g[g.source == 'original_gold58'].reset_index(drop=True)
ids = g58.StudyInstanceUID.tolist(); Y58 = g58[L].values.astype(float)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values.astype(float)
M58 = dict(rap=[(.75, np.load(G + '/gold58_r384_kernelserve_cpu.npy')), (.25, np.load(G + '/gold58_r224_kernelserve.npy'))],
           r6=rd('R6_gold58_serve.csv'), s6=rd('n42gpu_gold58.csv'))
outs = {k: pipeline(M58, c) for k, c in VAR.items()}
mac = lambda o: np.mean([wauc(o[:, j], Y58[:, j], np.ones(len(ids))) for j in range(12)])
P('== G58 parity: c4 %.5f (target .94113) | pw30 %.5f (target .94336) | pw25 %.5f (.94368) | lomm05 %.5f (.94384) | E1+pw30 %.5f (.94347)' % (
    mac(outs['c4']), mac(outs['pw30']), mac(outs['pw25']), mac(outs['lomm05']), mac(outs['E1'])))
m = np.ones(len(ids), bool)
WMAP['G58'] = {c: np.ones(len(ids)) for c in L}
WT = {c: (m, np.ones(len(ids))) for c in L}
rng = np.random.default_rng(7)
pt, bs, ch = evaluate('G58', outs, Y58, WT, 2000, rng)
summary['G58'] = dict(pt=pt, pairs=None)
P('== G58 (n 58, INFO ONLY) ==')
summary['G58']['pairs'], summary['G58']['ladder'] = report('G58', pt, bs)

# ============================== OOF nests ==============================
d0g = g; Y74 = d0g[L].values.astype(float); allg = set(d0g.StudyInstanceUID)
r7 = LD.r7(); s44, _ = LD.s6seed(1044); s45, _ = LD.s6seed(1045); lv2 = LD.labels('v2')
tr = [i for i in r7.index if i not in allg and i in lv2.index]; tr3 = [i for i in tr if i in s44.index and i in s45.index]
X = {'R7': r7.loc[tr3, ['p|' + c for c in L]].values.astype(float), 's1044': s44.loc[tr3, L].values.astype(float),
     's1045': s45.loc[tr3, L].values.astype(float)}
Y = lv2.loc[tr3, L].values.astype(float); n = len(tr3)
P('\nOOF rows %d (non-gold train with R7 + s1044 + s1045 OOF and v2 labels)' % n)
OAX = OA + ['Medial Meniscus', 'Lateral Meniscus']; OA4 = OA + ['Medial Meniscus']
def mkWT(cols_for):
    WT = {}; WM = {}
    for c in L:
        cols = cols_for(c)
        if cols is None:
            m = ~np.isnan(Y[:, J[c]]); w = np.ones(m.sum())
        else:
            jc = [J[x] for x in cols]; m = ~np.isnan(Y[:, jc]).any(1); w = rk_weights(Y[m][:, jc].astype(int), cols, Y74)
        WT[c] = (m, w); full = np.full(n, np.nan); full[m] = w; WM[c] = full
    return WT, WM
RAK32 = lambda c: OAX if c in OA else (['MCL', 'Medial Meniscus'] if c == 'MCL' else None)
RAK16 = lambda c: OA4 if c in OA else (['MCL', 'Medial Meniscus'] if c == 'MCL' else None)
RAW = lambda c: None
NESTS = {'N_A R7|s1044|s1045': ('R7', 's1044', 's1045'), 'N_B s1044|R7|s1045': ('s1044', 'R7', 's1045'),
         'N_C s1045|s1044|R7': ('s1045', 's1044', 'R7')}
YF = np.nan_to_num(Y, nan=0.)
for nn, (ra, r6, s6) in NESTS.items():
    M = dict(rap=[(1., X[ra])], r6=X[r6], s6=X[s6])
    outs = {k: pipeline(M, c) for k, c in VAR.items()}
    for rn, fn, nb in (('RAK32', RAK32, 200), ('RAK16', RAK16, 0), ('REP', RAW, 100)):
        WT, WM = mkWT(fn); WMAP[rn] = WM
        rng = np.random.default_rng(11)
        if nb == 0:
            # point estimates only (sensitivity)
            base = outs['pw30']; pt = {}
            for k in outs:
                d = 0.
                for j in range(12):
                    if np.array_equal(outs[k][:, j], base[:, j]): continue
                    mm, w = WT[L[j]]; d += wauc(outs[k][mm, j], YF[mm, j], w) - wauc(base[mm, j], YF[mm, j], w)
                pt[k] = d / 12 * 1e4
            P('\n== %s | %s (point only) ==' % (nn, rn))
            P('    singles: ' + ' | '.join('%s %+.2f' % (k, pt[k]) for k in list(COMP) + ['BORm']))
            P('    ladder vs c4: ' + ' | '.join('%s %+.2f (pub %.1f)' % (k, pt[k] - pt['c4'], PUB[k]) for k in PUB))
            P('    I: ' + ' | '.join('%s %+.2f' % (x + '+' + y, pt[x + '+' + y] - pt[x] - pt[y]) for x, y in PAIRS))
            summary['%s|%s' % (nn, rn)] = dict(pt=pt, pairs={x + '+' + y: dict(I=pt[x + '+' + y] - pt[x] - pt[y], d=pt[x + '+' + y]) for x, y in PAIRS})
            continue
        pt, bs, ch = evaluate(rn, outs, YF, WT, nb, rng)
        P('\n== %s | %s (boot %d) == [%.0f s]' % (nn, rn, nb, time.time() - T0))
        pr, lad = report(rn, pt, bs)
        summary['%s|%s' % (nn, rn)] = dict(pt=pt, pairs=pr, ladder=lad)
json.dump(summary, open(OUT + '.json', 'w'), indent=1, default=float)
P('STACK_LOCAL DONE %.0f s' % (time.time() - T0))
