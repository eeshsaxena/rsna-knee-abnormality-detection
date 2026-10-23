# rakpub_skep.py (10-21, Claude, SKEPTIC of the RAK-PUB ruler, track sink). RESEARCH ONLY, local CPU, NO OAI, no box.
# Reads sink_local.py (exec of its data/function part, its own log redirected); writes only rakpub_skep.log / rakpub_skep_tmp.log.
# Checks:
#   1. units: the T32 regime s is measured from the 32-cell (+.5 per cell) smoothed gold74 joint; the OAI ruler / night recal shift
#      is measured from the pairwise gold74 (+.5 per pair cell) log OR. Translate T32 s into the pairwise convention.
#   2. "no free factor": shift route vs factor route (g74 x f) vs T2W shift route, leave-one-rung-out (LORO) prediction of the
#      5 public rungs, and sensitivity of s* to the c4 position (common offset) of the joint-refit reads.
#   3. INFO values of the A22 components (pw25 - pw30, lomm05 on pw25, MCLmm on pw25, CVX on pw25) under each calibration route.
#   4. target sampling of the parts the ladder does not calibrate: LOA-MM pair +-1 se, Dirichlet over the gold74 32-cell table
#      (lomm05, pw25, CVX) and over the gold74 (MCL, MM) 2x2 (MCLmm). Target draws only, no row bootstrap.
# Aggregates only.
import itertools, time
import numpy as np
T0 = time.time()
D = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/'
src = open(D + 'sink_local.py', encoding='utf-8').read().split('SUMM = {}')[0]
src = src.replace("log = open(OUT + '.log', 'w')", "log = open('%srakpub_skep_tmp.log', 'w')" % D)
assert "rakpub_skep_tmp.log" in src
ns = {}; exec(src, ns)
pipeline, wauc, rake, or_cells, pair_marg, shift_target, mk, ROOT, LADDER = (ns[k] for k in (
    'pipeline', 'wauc', 'rake', 'or_cells', 'pair_marg', 'shift_target', 'mk', 'ROOT', 'LADDER'))
L, J, OA, OAX, COLS, MASK, Y, YF, X, n, NESTS, g, table74 = (ns[k] for k in (
    'L', 'J', 'OA', 'OAX', 'COLS', 'MASK', 'Y', 'YF', 'X', 'n', 'NESTS', 'g', 'table74'))
LOG = open(D + 'rakpub_skep.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()
PUB = {'eoa10': 5.9, 'eoa20': 10.3, 'eoa30': 9.6, 'pw20': 10.4, 'pw30': 11.3}
RK = list(PUB)
lor = lambda c: float(np.log(c[0, 0] * c[1, 1] / (c[0, 1] * c[1, 0])))

# ---------------- 1. units ----------------
P('== 1. regime units: trio pair log ORs of each target vs the pairwise gold74 (+.5) log OR, in gold74 se units ==')
cols = OAX; m = MASK['Medial OA']; jc = [J[x] for x in cols]; Ys = Y[m][:, jc].astype(int); marg = Ys.mean(0)
obs = np.zeros((2,) * 5)
for r in Ys: obs[tuple(r)] += 1
obs /= obs.sum()
T74 = table74(cols); Y74 = g[L].values.astype(float); Y58 = g[g.source == 'original_gold58'][L].values.astype(float)
def tabg(Yg, cc):
    jj = [J[c] for c in cc]; ok = ~np.isnan(Yg[:, jj]).any(1); T = np.zeros((2,) * len(cc))
    for r in Yg[ok][:, jj].astype(int): T[tuple(r)] += 1
    return T
T58 = tabg(Y58, cols)
P('  32-cell complete rows: gold74 %d, gold58 %d, train reports %d' % (T74.sum(), T58.sum(), m.sum()))
tgt0 = rake(T74 + .5, marg)
PAIRS = list(itertools.combinations(range(5), 2))
nm = lambda i: cols[i].replace('Medial ', 'M').replace('Lateral ', 'L').replace(' Meniscus', 'men').replace(' OA', 'OA').replace('PF OA', 'PF')
hdr = []
for s in (0., .23):
    tg = shift_target(tgt0, T74, cols, s, 0.); row = []; seff = []
    for i, j in PAIRS:
        c74 = pair_marg(T74, i, j) + .5; l74 = lor(c74); se = float(np.sqrt((1 / c74).sum()))
        lt = lor(pair_marg(tg, i, j)); lo = lor(pair_marg(obs, i, j)); e = (lt - l74) / se
        if cols[i] in OA and cols[j] in OA: seff.append(e)
        row.append('%s-%s g74 %.2f (se %.2f) T32 %.2f -> %+.2f sd | rep %.2f (%+.1f sd)' % (nm(i), nm(j), l74, se, lt, e, lo, (lo - l74) / se))
    P('  T32 @ s %+.2f: trio mean s_eff (pairwise convention) %+.2f  [%s]' % (s, np.mean(seff), ', '.join('%+.2f' % x for x in seff)))
    for r_ in row: P('     ' + r_)
# gold58 pair LORs in gold74 units (what T58 / T58_2W s means)
se58 = []
for i, j in PAIRS:
    if cols[i] in OA and cols[j] in OA:
        c74 = pair_marg(T74, i, j) + .5; c58 = pair_marg(T58, i, j) + .5
        se58.append((lor(c58) - lor(c74)) / np.sqrt((1 / c74).sum()))
P('  gold58 trio pair LOR vs gold74, gold74 se units: %s (mean %+.2f)' % (', '.join('%+.2f' % x for x in se58), np.mean(se58)))
# T58 32-cell smoothing offset
tg58 = rake(T58 + .5, marg); off58 = []
for i, j in PAIRS:
    if cols[i] in OA and cols[j] in OA:
        c74 = pair_marg(T74, i, j) + .5
        off58.append((lor(pair_marg(tg58, i, j)) - lor(c74)) / np.sqrt((1 / c74).sum()))
P('  T58 (32-cell +.5 on gold58) @ s 0 trio s_eff vs gold74 pairwise: %s (mean %+.2f)' % (', '.join('%+.2f' % x for x in off58), np.mean(off58)))

# ---------------- weights for each route ----------------
def w_oa(kind, s, T=None, men=None):
    """OA-column weights. kind 'T32' (sink_local RAK) or 'T2W' (pairwise, gold74 +.5 pair LOR + s se on the trio).
    men = optional dict {(i, j): extra shift in se} for single OA-meniscus pairs (T32 only)."""
    T = T74 if T is None else T
    if kind == 'T32':
        tg = shift_target(rake(T + .5, marg), T, cols, s, 0.)
        if men:
            tgts = {}
            for i, j in PAIRS:
                cur = pair_marg(tg, i, j); c74 = pair_marg(T, i, j) + .5; se = np.sqrt((1 / c74).sum())
                tgts[(i, j)] = or_cells(cur[1, :].sum(), cur[:, 1].sum(), lor(cur) + men.get((i, j), 0.) * se)
            q = tg.copy()
            for _ in range(300):
                for (i, j), t in tgts.items():
                    cur = pair_marg(q, i, j); sh = [1] * 5; sh[i] = 2; sh[j] = 2; q = q * (t / cur).reshape(sh)
                q /= q.sum()
            tg = q
    else:
        q = obs.copy(); tgts = {}
        for i, j in PAIRS:
            c = pair_marg(T, i, j) + .5; sh_ = s if (cols[i] in OA and cols[j] in OA) else 0.
            tgts[(i, j)] = or_cells(marg[i], marg[j], lor(c) + sh_ * np.sqrt((1 / c).sum()))
        for _ in range(300):
            for (i, j), t in tgts.items():
                cur = pair_marg(q, i, j); shp = [1] * 5; shp[i] = 2; shp[j] = 2; q = q * (t / np.maximum(cur, 1e-15)).reshape(shp)
            q /= q.sum()
        tg = q
    w = (tg / obs)[tuple(Ys.T)]
    return w / w.mean()

mM = MASK['MCL']; jM = [J['MCL'], J['Medial Meniscus']]; YsM = Y[mM][:, jM].astype(int)
obsM = np.zeros((2, 2))
for r in YsM: obsM[tuple(r)] += 1
obsM /= obsM.sum(); TM74 = tabg(Y74, ['MCL', 'Medial Meniscus'])
P('  gold74 (MCL, MM) 2x2 = %s (MCL+ %d) | train reports MCL+ share %.3f' % (TM74.astype(int).tolist(), TM74[1].sum(), YsM[:, 0].mean()))
def w_mcl(TM=None):
    TM = TM74 if TM is None else TM
    tg = rake(TM + .5, YsM.mean(0)); w = (tg / obsM)[tuple(YsM.T)]; return w / w.mean()

VAR = {k: dict(ROOT, **d) for k, d in LADDER.items()}
VAR.update({'pw30': mk('pw30'), 'pw25': mk('pw25'), 'spl': mk('spl'), 'salt': mk('salt'), 'pw25+CVX': mk('pw25', ('CVX',)),
            'pw30+CVX': mk('pw30', ('CVX',))})
OUTS = {nn: {k: pipeline(dict(rap=[(1., X[ra])], r6=X[r6], s6=X[s6]), c) for k, c in VAR.items()} for nn, (ra, r6, s6) in NESTS.items()}
OAJ = [J[c] for c in OA]; jMCL = J['MCL']

def deltas(nn, woa, wm=None):
    """macro delta vs pw30 (e-4) per variant on the OA trio (+ MCL if wm given)."""
    O = OUTS[nn]; b = O['pw30']; res = {}; cache = {}
    for k, o in O.items():
        d = 0.
        for j in OAJ:
            if np.array_equal(o[:, j], b[:, j]): continue
            if j not in cache: cache[j] = wauc(b[m, j], YF[m, j], woa)
            d += wauc(o[m, j], YF[m, j], woa) - cache[j]
        if wm is not None and not np.array_equal(o[:, jMCL], b[:, jMCL]):
            if 'M' not in cache: cache['M'] = wauc(b[mM, jMCL], YF[mM, jMCL], wm)
            d += wauc(o[mM, jMCL], YF[mM, jMCL], wm) - cache['M']
        res[k] = d / 12 * 1e4
    return res

def ladder(D_):
    return {k: D_[k] - D_['c4'] for k in RK}

def nest_mean(fn):
    rs = [fn(nn) for nn in NESTS]
    return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}, rs

# ---------------- 2. calibration routes, LORO ----------------
P('\n== 2. calibration routes on the 5 joint-refit public rungs (nest mean) ==')
SG = np.round(np.arange(-1.0, 0.81, 0.05), 3)
CURVE = {}
for kind in ('T32', 'T2W'):
    for s in SG:
        w = w_oa(kind, s); dm, _ = nest_mean(lambda nn: deltas(nn, w)); CURVE[(kind, s)] = dm
P('  [%.0f s] curves done' % (time.time() - T0))
pv = np.array([PUB[k] for k in RK])
def fit_shift(kind, keep, target=pv):
    best = None
    for s in SG:
        v = np.array([ladder(CURVE[(kind, s)])[k] for k in RK]); e = ((v - target)[keep] ** 2).sum()
        if best is None or e < best[1]: best = (s, e, v)
    return best
def fit_factor(kind, s0, keep, target=pv):
    v = np.array([ladder(CURVE[(kind, s0)])[k] for k in RK]); f = (v[keep] @ target[keep]) / (v[keep] @ v[keep])
    return f, ((f * v - target)[keep] ** 2).sum(), f * v
allk = np.ones(5, bool)
ROUTES = {}
for kind in ('T32', 'T2W'):
    s, e, v = fit_shift(kind, allk); ROUTES['%s shift' % kind] = dict(s=s, f=1., sse=e, v=v)
for kind, s0, lab in (('T32', 0., 'T32 s0 x f (g74 x f)'), ('T2W', 0., 'T2W s0 x f'), ('T2W', -0.85, 'T2W s-.85 x f (OAI regime)')):
    f, e, v = fit_factor(kind, s0, allk); ROUTES[lab] = dict(s=s0, f=f, sse=e, v=v, kind=kind)
for lab, R in ROUTES.items():
    P('  %-28s s %+.2f f %.2f | ladder %s | SSE %.2f' % (lab, R['s'], R['f'], ' '.join('%+.2f' % x for x in R['v']), R['sse']))
# LORO
P('  LORO (fit on 4 rungs, predict the 5th; residual = predicted - public):')
for lab in ROUTES:
    res = []
    for h in range(5):
        keep = allk.copy(); keep[h] = False
        if lab.endswith('shift'):
            s, e, v = fit_shift(lab.split()[0], keep)
        else:
            R = ROUTES[lab]; f, e, v = fit_factor(R['kind'], R['s'], keep)
        res.append(v[h] - pv[h])
    res = np.array(res); P('    %-28s residuals %s | LORO RMSE %.2f' % (lab, ' '.join('%+.2f' % x for x in res), np.sqrt((res ** 2).mean())))
# public-read uncertainty: joint refit [p10, p90] vs c4 (implied2 sec 0.2) -> c4 position common offset
for off in (-1.0, +1.0):
    s, e, v = fit_shift('T32', allk, pv + off); s2, e2, _ = fit_shift('T2W', allk, pv + off)
    P('  c4 position offset %+.1f e-4 on all rungs: T32 s* %+.2f (SSE %.1f) | T2W s* %+.2f (SSE %.1f)' % (off, s, e, s2, e2))
# LB-only-ish: pw30 - eoa20 +0.5 instead of +1.0 (implied2 LB-only)
alt = pv.copy(); alt[4] = pv[1] + 0.5
s, e, v = fit_shift('T32', allk, alt); P('  pw30 = eoa20 + 0.5 (LB-only gap): T32 s* %+.2f (SSE %.1f)' % (s, e))

# ---------------- 3. INFO values per route ----------------
P('\n== 3. A22 INFO values by calibration route (e-4 vs pw30 / marginal; nest mean [min, max]) ==')
wm0 = w_mcl()
def info(kind, s, f):
    w = w_oa(kind, s); _, rs = nest_mean(lambda nn: deltas(nn, w, wm0))
    out = {}
    for lab, fn in (('pw25-pw30', lambda r: r['pw25']), ('lomm05@pw25', lambda r: r['spl'] - r['pw25']),
                    ('MCLmm@pw25', lambda r: r['salt'] - r['pw25']), ('CVX@pw25', lambda r: r['pw25+CVX'] - r['pw25']),
                    ('CVX@pw30', lambda r: r['pw30+CVX'])):
        v = np.array([fn(r) for r in rs]) * (f if lab != 'MCLmm@pw25' else 1.)
        out[lab] = (v.mean(), v.min(), v.max())
    return out
for lab, R in ROUTES.items():
    kind = R.get('kind', lab.split()[0]); I = info(kind, R['s'], R['f'])
    P('  %-28s ' % lab + ' | '.join('%s %+.2f [%+.2f, %+.2f]' % (k, *v) for k, v in I.items()))

# ---------------- 4. target sampling ----------------
P('\n== 4. gold74 target sampling (T32 @ s* of sec 2; MCL 2x2) ==')
sT = ROUTES['T32 shift']['s']
iLM = cols.index('Lateral OA'); iMM = cols.index('Medial Meniscus'); pr = (min(iLM, iMM), max(iLM, iMM))
c = pair_marg(T74, *pr) + .5
P('  gold74 (LOA, MM) pair +.5 = %s, LOR %+.2f se %.2f | train reports LOR %+.2f' % (c.tolist(), lor(c), np.sqrt((1 / c).sum()), lor(pair_marg(obs, *pr))))
for dz in (-1., 1.):
    w = w_oa('T32', sT, men={pr: dz}); dm, rs = nest_mean(lambda nn: deltas(nn, w))
    P('  LOA-MM LOR %+.0f se: lomm05@pw25 %+.2f [%s] | pw25-pw30 %+.2f | CVX@pw25 %+.2f' % (
        dz, dm['spl'] - dm['pw25'], ', '.join('%+.2f' % (r['spl'] - r['pw25']) for r in rs), dm['pw25'], dm['pw25+CVX'] - dm['pw25']))
rng = np.random.default_rng(2121); B = 60
acc = {k: [] for k in ('lomm', 'pw25', 'cvx', 'mcl', 'sfit')}
for b in range(B):
    Td = rng.dirichlet((T74 + .5).ravel()).reshape(T74.shape) * T74.sum(); Td = np.maximum(Td - .5, 0)
    # refit s on the ladder for this draw (coarse grid), as the ruler would
    best = None
    for s in np.arange(-0.4, 0.81, 0.1):
        w = w_oa('T32', s, T=Td); dm, _ = nest_mean(lambda nn: deltas(nn, w))
        e = ((np.array([ladder(dm)[k] for k in RK]) - pv) ** 2).sum()
        if best is None or e < best[1]: best = (s, e, dm)
    dm = best[2]; acc['sfit'].append(best[0])
    acc['lomm'].append(dm['spl'] - dm['pw25']); acc['pw25'].append(dm['pw25']); acc['cvx'].append(dm['pw25+CVX'] - dm['pw25'])
    TMd = rng.dirichlet((TM74 + .5).ravel()).reshape(2, 2) * TM74.sum(); TMd = np.maximum(TMd - .5, 0)
    wmd = w_mcl(TMd); dmm, _ = nest_mean(lambda nn: deltas(nn, w_oa('T32', sT), wmd)); acc['mcl'].append(dmm['salt'] - dmm['pw25'])
    if b % 20 == 19: P('  [%d draws, %.0f s]' % (b + 1, time.time() - T0))
st = lambda x: '%+.2f sd %.2f [%+.2f, %+.2f] P>0 %.2f' % (np.mean(x), np.std(x), np.percentile(x, 10), np.percentile(x, 90), (np.array(x) > 0).mean())
P('  Dirichlet(gold74 + .5) target draws, s refit per draw (B %d, nest mean, no row bootstrap):' % B)
P('    refit s*: %s' % st(acc['sfit']))
P('    pw25-pw30 %s' % st(acc['pw25'])); P('    lomm05@pw25 %s' % st(acc['lomm'])); P('    CVX@pw25 %s' % st(acc['cvx']))
P('    MCLmm@pw25 (MCL 2x2 draws) %s' % st(acc['mcl']))
P('RAKPUB_SKEP DONE %.0f s' % (time.time() - T0))
