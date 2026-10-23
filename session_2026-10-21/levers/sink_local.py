# sink_local.py (10-21 ~09:50 IST, Claude, SINK track). RESEARCH ONLY, local CPU, NO OAI, writes only under session_2026-10-21/levers.
# Kitchen-sink stack measurement on top of each plausible A22 winner (pw30, pw25, spl, swide, salt).
# Rulers:
#   G58   gold58 (n 58) c4 kernel path from the member serve files (INFO ONLY, never a selection ruler)
#   RAK   RSNA train OOF (R7 / s1044 / s1045) in three c4-shaped nests, report-v2 labels raked by IPF to the gold74 EXPERT joint
#         (OA: 32-cell (MOA, LOA, PF, MM, LM); MCL: (MCL, MM) 2x2); other labels unraked (= report OOF).
#         Regimes: g74 = gold74 couplings; s<x> = the 3 OA-trio pair log ORs shifted by x sd of the gold74 log OR (meniscus pairs
#         kept); sM<x> = trio AND OA-meniscus pairs shifted. PUB = the shift whose E-OA ladder best matches the 5 public reads.
#   REP   the same nests on raw report-v2 labels (mechanism control: an expert-regime decorrelation MUST lose here)
# Bases: pw30 (scored), pw25 (A_OA .25), spl (pw25 + LOA <- MM .05), swide (spl + MCL <- MM .1), salt (pw25 + MCL <- MM .1).
# Extras (near-miss inventory; one change each on a base):
#   CVX  convex E-OA exponent p 2 at the base dose (levers/build_convex.py spec)   CVXd  p 2 at base dose - .05 (levers "twin")
#   SUBH E-OA subtracts pct(2/3 pct(R6) + 1/3 pct(S6fall)) of the other compartments (stack/inventory.md spec)
#   S6up W_S6F .2 -> .3 on the OA trio      RPF W_RAPTOR_C['PF OA'] .5 -> .4      E1 soft-OR (member pct ** 2)
#   BOR  MOA += .1 MM, LOA += .1 LM (C13)   C6p LM <- MM .05 (night C6')          C14 LOA += .1 LM (night C14)
# Aggregates only. Usage: python -B sink_local.py [NB]
import sys, json, itertools, time, hashlib
import numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.optimize import brentq
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
import load as LD
T0 = time.time()
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 150
OUT = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/sink_local'
log = open(OUT + '.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
L = LD.L; J = {c: i for i, c in enumerate(L)}; G = LD.G
OA = ['Medial OA', 'Lateral OA', 'PF OA']; OAX = OA + ['Medial Meniscus', 'Lateral Meniscus']
pct = lambda v: rankdata(v) / len(v)
C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
W_PW = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}
f32 = lambda x: np.asarray(x, np.float64).astype(np.float32).astype(np.float64)
LOMM = ('Lateral OA', 'Medial Meniscus', .05); MCLMM = ('MCL', 'Medial Meniscus', .1)
ROOT = dict(a=.3, mode='pw', p=1., sub='lin', e1=1., ws6_oa=.2, wr_pf=.5, lines=())
BASES = {'pw30': dict(), 'pw25': dict(a=.25), 'spl': dict(a=.25, lines=(LOMM,)), 'swide': dict(a=.25, lines=(LOMM, MCLMM)),
         'salt': dict(a=.25, lines=(MCLMM,))}
EXTRAS = {'CVX': dict(p=2.), 'CVXd': dict(p=2., da=-.05), 'SUBH': dict(sub='high'), 'S6up': dict(ws6_oa=.3), 'RPF': dict(wr_pf=.4),
          'E1': dict(e1=2.), 'BOR': dict(lines=(('Medial OA', 'Medial Meniscus', -.1), ('Lateral OA', 'Lateral Meniscus', -.1))),
          'C6p': dict(lines=(('Lateral Meniscus', 'Medial Meniscus', .05),)), 'C14': dict(lines=(('Lateral OA', 'Lateral Meniscus', -.1),))}
PAIRX = ['CVX', 'CVXd', 'SUBH', 'S6up', 'RPF', 'C6p', 'C14', 'BOR']
PAIR_BASES = ['pw30', 'pw25', 'spl', 'swide', 'salt']
LADDER = {'c4': dict(a=0.), 'eoa10': dict(a=.1, mode='sym'), 'eoa20': dict(a=.2, mode='sym'), 'eoa30': dict(a=.3, mode='sym'),
          'pw20': dict(a=.2)}
PUB = {'eoa10': 5.9, 'eoa20': 10.3, 'eoa30': 9.6, 'pw20': 10.4, 'pw30': 11.3}   # public vs c4, e-4 (CLAIMED val/implied2.md)


def mk(base, extras=()):
    c = dict(ROOT); c.update({k: v for k, v in BASES[base].items()}); c['lines'] = tuple(c['lines'])
    for x in extras:
        d = dict(EXTRAS[x]); da = d.pop('da', 0.); ln = d.pop('lines', ())
        c.update(d); c['a'] = round(c['a'] + da, 6); c['lines'] = c['lines'] + tuple(ln)
    return c


VAR = {}
for k, d in LADDER.items():
    c = dict(ROOT); c.update(d); VAR[k] = c
for b in BASES:
    VAR[b] = mk(b)
    for x in EXTRAS:
        VAR[b + '+' + x] = mk(b, (x,))
for b in PAIR_BASES:
    for x, y in itertools.combinations(PAIRX, 2):
        if {x, y} == {'CVX', 'CVXd'}:
            continue
        VAR[b + '+' + x + '+' + y] = mk(b, (x, y))


def pipeline(M, cfg):
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
    F = f32(F); out = F.copy(); a = cfg['a']
    if a:
        U = {c: pct(F[:, J[c]]) for c in OA}
        H = {c: pct(2 / 3 * pct(M['r6'][:, J[c]]) + 1 / 3 * pct(M['s6'][:, J[c]])) for c in OA} if cfg['sub'] == 'high' else U
        S = H if cfg['p'] == 1 else {c: H[c] ** cfg['p'] for c in OA}
        for c in OA:
            o = [k for k in OA if k != c]
            w = [W_PW[tuple(sorted((c, k), key=OA.index))] for k in o] if cfg['mode'] == 'pw' else [1., 1.]
            out[:, J[c]] = f32(pct(U[c] - a / 2.0 * (w[0] * S[o[0]] + w[1] * S[o[1]])))
    for t, o, b in cfg['lines']:
        out[:, J[t]] = f32(pct(pct(out[:, J[t]]) - b * pct(out[:, J[o]])))
    return out


def wauc(s, y, w):
    o = np.argsort(s, kind='mergesort'); s, y, w = s[o], y[o], w[o]
    wp = w * y; wn = w * (1 - y)
    _, inv = np.unique(s, return_inverse=True)
    gp = np.bincount(inv, wp); gn = np.bincount(inv, wn); cn = np.cumsum(gn) - gn
    return float((gp * (cn + .5 * gn)).sum() / (wp.sum() * wn.sum()))


def or_cells(p1, p2, lor):
    lo = max(0., p1 + p2 - 1) + 1e-12; hi = min(p1, p2) - 1e-12
    g = lambda p11: np.log(p11) + np.log(1 - p1 - p2 + p11) - np.log(p1 - p11) - np.log(p2 - p11) - lor
    p11 = brentq(g, lo, hi)
    return np.array([[1 - p1 - p2 + p11, p2 - p11], [p1 - p11, p11]])


def rake(T, margins, it=300):
    T = T.copy(); k = T.ndim
    for _ in range(it):
        for ax in range(k):
            cur = T.sum(axis=tuple(i for i in range(k) if i != ax)); sh = [1] * k; sh[ax] = 2
            T = T * (np.array([1 - margins[ax], margins[ax]]) / cur).reshape(sh)
    return T / T.sum()


def pair_marg(T, i, j):
    return T.sum(axis=tuple(k for k in range(T.ndim) if k not in (i, j)))


def shift_target(tgt, T74, cols, s_trio, s_men, it=300):
    """IPF the raked gold74 joint so that each OA-trio pair (and, with s_men, each OA-meniscus pair) has log OR = its gold74 log OR
    + s * se (se from the gold74 2x2 + .5), all other pair 2x2 margins and the 1-D margins kept."""
    if not s_trio and not s_men:
        return tgt
    k = len(cols); tg = {}
    for i, j in itertools.combinations(range(k), 2):
        ci, cj = cols[i], cols[j]
        c74 = pair_marg(T74, i, j).ravel() + .5
        lor74 = np.log(c74[0] * c74[3] / (c74[1] * c74[2])); se = np.sqrt((1 / c74).sum())
        cur = pair_marg(tgt, i, j); p1 = cur[1, :].sum(); p2 = cur[:, 1].sum()
        lc = np.log(cur[0, 0] * cur[1, 1] / (cur[0, 1] * cur[1, 0]))
        if ci in OA and cj in OA:
            s = s_trio
        elif (ci in OA) != (cj in OA):
            s = s_men
        else:
            s = 0.
        tg[(i, j)] = or_cells(p1, p2, lc + s * se)
    q = tgt.copy()
    for _ in range(it):
        for (i, j), t in tg.items():
            cur = pair_marg(q, i, j); sh = [1] * k; sh[i] = 2; sh[j] = 2
            q = q * (t / cur).reshape(sh)
        q /= q.sum()
    return q


# ============================== data ==============================
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}); Y74 = g[L].values.astype(float); allg = set(g.StudyInstanceUID)
g58 = g[g.source == 'original_gold58'].reset_index(drop=True); ids = g58.StudyInstanceUID.tolist(); Y58 = g58[L].values.astype(float)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values.astype(float)
M58 = dict(rap=[(.75, np.load(G + '/gold58_r384_kernelserve_cpu.npy')), (.25, np.load(G + '/gold58_r224_kernelserve.npy'))],
           r6=rd('R6_gold58_serve.csv'), s6=rd('n42gpu_gold58.csv'))
r7 = LD.r7(); s44, _ = LD.s6seed(1044); s45, _ = LD.s6seed(1045); lv2 = LD.labels('v2')
tr = [i for i in r7.index if i not in allg and i in lv2.index]; tr3 = [i for i in tr if i in s44.index and i in s45.index]
X = {'R7': r7.loc[tr3, ['p|' + c for c in L]].values.astype(float), 's1044': s44.loc[tr3, L].values.astype(float),
     's1045': s45.loc[tr3, L].values.astype(float)}
Y = lv2.loc[tr3, L].values.astype(float); n = len(tr3); YF = np.nan_to_num(Y, nan=0.)
NESTS = {'N_A': ('R7', 's1044', 's1045'), 'N_B': ('s1044', 'R7', 's1045'), 'N_C': ('s1045', 's1044', 'R7')}
COLS = {c: (OAX if c in OA else ['MCL', 'Medial Meniscus']) for c in OA + ['MCL']}
MASK = {c: ~np.isnan(Y[:, [J[x] for x in COLS[c]]]).any(1) for c in COLS}


def table74(cols):
    jc = [J[c] for c in cols]; ok = ~np.isnan(Y74[:, jc]).any(1); T = np.zeros((2,) * len(cols))
    for r in Y74[ok][:, jc].astype(int): T[tuple(r)] += 1
    return T


def weights(regime):
    """label -> (mask, w over masked rows) for the regime (None = REP raw)."""
    WT = {}
    for c in L:
        if regime is None or c not in COLS:
            m = ~np.isnan(Y[:, J[c]]); WT[c] = (m, np.ones(m.sum())); continue
        cols = COLS[c]; m = MASK[c]; jc = [J[x] for x in cols]; Ys = Y[m][:, jc].astype(int)
        obs = np.zeros((2,) * len(cols))
        for r in Ys: obs[tuple(r)] += 1
        obs /= obs.sum(); T74 = table74(cols); tgt = rake(T74 + .5, Ys.mean(0))
        if c in OA:
            tgt = shift_target(tgt, T74, cols, regime[0], regime[1])
        w = (tgt / obs)[tuple(Ys.T)]; WT[c] = (m, w / w.mean())
    return WT


def colkeys(outs):
    """unique column store: (j, key) -> array; var -> list of 12 keys."""
    store, vk = {}, {}
    for k, o in outs.items():
        ks = []
        for j in range(12):
            h = hashlib.md5(o[:, j].tobytes()).hexdigest(); store[(j, h)] = o[:, j]; ks.append(h)
        vk[k] = ks
    return store, vk


def deltas(store, vk, WT, YY, ii=None):
    """macro delta vs pw30 (e-4) for every variant, one (re)sample ii of rows (None = all)."""
    au = {}
    for (j, h), col in store.items():
        m, w = WT[L[j]]
        if ii is None:
            au[(j, h)] = wauc(col[m], YY[m, j], w)
        else:
            sel = ii[m[ii]]
            ww = WMAPS[id(WT)][L[j]][sel]
            au[(j, h)] = wauc(col[sel], YY[sel, j], ww)
    b = vk['pw30']
    return {k: sum(au[(j, ks[j])] - au[(j, b[j])] for j in range(12) if ks[j] != b[j]) / 12 * 1e4 for k, ks in vk.items()}


WMAPS = {}


def full_w(WT):
    WM = {}
    for c, (m, w) in WT.items():
        f = np.full(len(m), np.nan); f[m] = w; WM[c] = f
    WMAPS[id(WT)] = WM


def summarize(tag, pt, bs):
    """bs: list of dicts. Returns dict of mean/sd/P for singles, marginals, interactions."""
    A = {k: np.array([d[k] for d in bs]) for k in pt} if bs else None
    R = {'pt': pt}
    st = lambda x: (float(x.mean()), float(x.std()), float((x > 0).mean()))
    lad = {k: pt[k] - pt['c4'] for k in PUB}
    R['ladder'] = lad
    P('  [%s] ladder vs c4: ' % tag + ' | '.join('%s %+.2f (pub %.1f)' % (k, v, PUB[k]) for k, v in lad.items()) +
      ' | SSE vs pub %.1f' % sum((lad[k] - PUB[k]) ** 2 for k in PUB))
    P('  [%s] bases vs pw30: ' % tag + ' | '.join('%s %+.2f' % (b, pt[b]) for b in BASES) +
      ' | LOMM on pw25 %+.2f, on salt %+.2f | MCLMM on pw25 %+.2f, on spl %+.2f' % (pt['spl'] - pt['pw25'], pt['swide'] - pt['salt'],
                                                                               pt['salt'] - pt['pw25'], pt['swide'] - pt['spl']))
    R['marg'] = {}
    for b in BASES:
        row = []
        for x in EXTRAS:
            k = b + '+' + x; mpt = pt[k] - pt[b]
            if A is not None:
                mm, sd, pp = st(A[k] - A[b]); R['marg'][k] = (mpt, sd, pp); row.append('%s %+.2f (%.2f, P %.2f)' % (x, mpt, sd, pp))
            else:
                R['marg'][k] = (mpt, None, None); row.append('%s %+.2f' % (x, mpt))
        P('  [%s] marg on %-5s: ' % (tag, b) + ' | '.join(row))
    R['I'] = {}
    for b in PAIR_BASES:
        row = []
        for x, y in itertools.combinations(PAIRX, 2):
            if {x, y} == {'CVX', 'CVXd'}:
                continue
            k = b + '+' + x + '+' + y; ipt = pt[k] - pt[b + '+' + x] - pt[b + '+' + y] + pt[b]
            if A is not None:
                I = A[k] - A[b + '+' + x] - A[b + '+' + y] + A[b]; mm, sd, pp = st(I); R['I'][k] = (ipt, sd, pp)
                row.append('%s*%s %+.2f (%.2f, %.2f)' % (x, y, ipt, sd, pp))
            else:
                R['I'][k] = (ipt, None, None); row.append('%s*%s %+.2f' % (x, y, ipt))
        P('  [%s] I on %-5s: ' % (tag, b) + ' | '.join(row))
    return R


SUMM = {}
# ============================== G58 (INFO) ==============================
outs = {k: pipeline(M58, c) for k, c in VAR.items()}
mac = lambda o: np.mean([wauc(o[:, j], Y58[:, j], np.ones(len(ids))) for j in range(12)])
P('== G58 parity: c4 %.5f (.94113) | pw30 %.5f (.94336) | pw25 %.5f (.94368) | pw30+lomm05 %.5f (.94384) | spl %.5f (.94417) | '
  'swide %.5f (.94492) | salt %.5f (.94444) | pw25p2 %.5f (.94291) | pw20p2 [pw25+CVXd] %.5f (.94328)' % (
      mac(outs['c4']), mac(outs['pw30']), mac(outs['pw25']), mac(pipeline(M58, dict(ROOT, lines=(LOMM,)))), mac(outs['spl']),
      mac(outs['swide']), mac(outs['salt']), mac(outs['pw25+CVX']), mac(outs['pw25+CVXd'])))
WT58 = {c: (np.ones(len(ids), bool), np.ones(len(ids))) for c in L}; full_w(WT58)
store, vk = colkeys(outs)
pt = deltas(store, vk, WT58, Y58)
rng = np.random.default_rng(7)
bs = [deltas(store, vk, WT58, Y58, rng.integers(0, len(ids), len(ids))) for _ in range(1000)]
P('\n== G58 (n 58, INFO ONLY; boot 1000) ==')
SUMM['G58'] = summarize('G58', pt, bs)

# ============================== RAK regime scan (point estimates, ladder fit) ==============================
P('\n== RAK regime scan: E-OA ladder vs c4 (e-4) per nest, OA-trio shift s (sd of gold74 log OR) ==')
NOUT = {}
for nn, (ra, r6, s6) in NESTS.items():
    M = dict(rap=[(1., X[ra])], r6=X[r6], s6=X[s6]); NOUT[nn] = {k: pipeline(M, c) for k, c in VAR.items()}
SCAN = {}
GRID = [-0.83, -0.4, 0., 0.4, 0.8, 1.2, 1.6, 2.0]
for s in GRID:
    WT = weights((s, 0.)); row = []; sse = 0.; lads = []
    for nn in NESTS:
        st_, vk_ = colkeys({k: NOUT[nn][k] for k in list(LADDER) + ['pw30']})
        d = deltas(st_, vk_, WT, YF); lad = {k: d[k] - d['c4'] for k in PUB}; lads.append(lad)
        sse += sum((lad[k] - PUB[k]) ** 2 for k in PUB)
    mlad = {k: np.mean([l[k] for l in lads]) for k in PUB}
    # amplitude-free shape score: best single factor f minimising SSE
    v = np.array([mlad[k] for k in PUB]); p_ = np.array([PUB[k] for k in PUB]); f = (v @ p_) / (v @ v)
    SCAN[s] = dict(sse=sse / 3, lad=mlad, f=f, sse_shape=float(((f * v - p_) ** 2).sum()))
    P('  s %+5.2f | mean ladder ' % s + ' '.join('%s %+.2f' % (k, mlad[k]) for k in PUB) +
      ' | SSE(level) %.1f | best factor %.2f, SSE(shape) %.2f | eoa30-eoa20 %+.2f (pub -0.7) | pw30-pw20 %+.2f (pub +0.9)' % (
          sse / 3, f, SCAN[s]['sse_shape'], mlad['eoa30'] - mlad['eoa20'], mlad['pw30'] - mlad['pw20']))
# fine fit on level SSE (quadratic interpolation around the grid min)
sg = np.array(GRID); se_ = np.array([SCAN[s]['sse'] for s in GRID]); i0 = int(np.argmin(se_))
if 0 < i0 < len(GRID) - 1:
    c2 = np.polyfit(sg[i0 - 1:i0 + 2], se_[i0 - 1:i0 + 2], 2); SPUB = float(-c2[1] / (2 * c2[0]))
else:
    SPUB = float(sg[i0])
SPUB = round(SPUB, 2)
P('  -> PUB regime (level fit to the 5 public reads): s* = %+.2f' % SPUB)
SUMM['scan'] = {str(k): v for k, v in SCAN.items()}; SUMM['SPUB'] = SPUB

# ============================== RAK / REP with bootstrap ==============================
REG = [('g74', (0., 0.)), ('PUB', (SPUB, 0.)), ('PUBM', (SPUB, SPUB)), ('s-0.83', (-0.83, 0.)), ('REP', None)]
for rn, reg in REG:
    WT = weights(reg); full_w(WT)
    for nn in NESTS:
        store, vk = colkeys(NOUT[nn]); pt = deltas(store, vk, WT, YF)
        rng = np.random.default_rng(11)
        nb = NB if rn in ('g74', 'PUB', 'REP') else max(NB // 3, 40)
        bs = [deltas(store, vk, WT, YF, rng.integers(0, n, n)) for _ in range(nb)]
        P('\n== %s | %s (boot %d) [%.0f s] ==' % (rn, nn, nb, time.time() - T0))
        SUMM['%s|%s' % (rn, nn)] = summarize('%s %s' % (rn, nn), pt, bs)
json.dump(SUMM, open(OUT + '.json', 'w'), indent=0, default=float)
P('SINK_LOCAL DONE %.0f s' % (time.time() - T0))
