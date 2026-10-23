# sink_cvx_skep.py (10-21 ~10:40 IST, Claude, SINK track). RESEARCH ONLY, local CPU, NO OAI. Skeptic for CVX (convex E-OA, p 2),
# the one extra that survives sink_local.py on RAK-PUB. Question: does the RAK gain ride on the gold74 3-way OA structure (74 knees,
# 16 of them report-read) rather than on the pairwise expert couplings?
# Targets for the OA columns (MOA, LOA, PF, MM, LM), each at its own ladder-fitted trio shift s* (refit per target):
#   T32    gold74 32-cell joint (+.5) raked to the train margins (= sink_local RAK)
#   T2W    pairwise-only: IPF from the TRAIN-REPORT joint to the gold74 pairwise log ORs (+ s* se on the trio), train 1-D margins
#          (higher-order structure = the train reports', so a 3-way gold74 quirk cannot help)
#   T58    gold58-only 32-cell joint (the 16 report-read extra rows dropped)
#   T58_2W gold58 pairwise-only
#   RAKu   T32 with Dirichlet target draws x row bootstrap
# Plus: gold74 trio 3-way check (observed all-three-positive count vs the pairwise max-entropy prediction) and leave-one-gold-study-out
# influence on the CVX gain (T32, nest N_A, point). Aggregates only. Writes sink_cvx_skep.log.
import sys, itertools, time
import numpy as np, pandas as pd
T0 = time.time()
src = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/sink_local.py', encoding='utf-8').read()
src = src.split('SUMM = {}')[0].replace("log = open(OUT + '.log', 'w')", "log = open(OUT + '_skeptmp.log', 'w')")
ns = {}; exec(src, ns)
pipeline, wauc, rake, or_cells, pair_marg = ns['pipeline'], ns['wauc'], ns['rake'], ns['or_cells'], ns['pair_marg']
mk, ROOT, LADDER = ns['mk'], ns['ROOT'], ns['LADDER']
L, J, OA, OAX, COLS, MASK, Y, YF, X, n, NESTS, g = (ns[k] for k in ('L', 'J', 'OA', 'OAX', 'COLS', 'MASK', 'Y', 'YF', 'X', 'n', 'NESTS', 'g'))
LOG = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/sink_cvx_skep.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()
PUB = {'eoa10': 5.9, 'eoa20': 10.3, 'eoa30': 9.6, 'pw20': 10.4, 'pw30': 11.3}
VAR = {k: dict(ROOT, **d) for k, d in LADDER.items()}
VAR.update({'pw30': mk('pw30'), 'pw25': mk('pw25'), 'spl': mk('spl'), 'pw30+CVX': mk('pw30', ('CVX',)), 'pw25+CVX': mk('pw25', ('CVX',)),
            'spl+CVX': mk('spl', ('CVX',)), 'pw30+CVXd': mk('pw30', ('CVXd',)), 'pw25+CVXd': mk('pw25', ('CVXd',))})
VAR['pw25+P15'] = dict(mk('pw25'), p=1.5); VAR['pw25+P3'] = dict(mk('pw25'), p=3.)
SHOW = ['pw25', 'spl', 'pw30+CVX', 'pw25+CVX', 'spl+CVX', 'pw30+CVXd', 'pw25+CVXd', 'pw25+P15', 'pw25+P3']
g58 = g[g.source == 'original_gold58']
GSETS = {'g74': g[L].values.astype(float), 'g58': g58[L].values.astype(float)}


def tab(Yg, cols):
    jc = [J[c] for c in cols]; ok = ~np.isnan(Yg[:, jc]).any(1); T = np.zeros((2,) * len(cols))
    for r in Yg[ok][:, jc].astype(int): T[tuple(r)] += 1
    return T


def target(kind, Yg, s, T=None):
    """joint target over OAX for the OA columns' masked train rows; kind '32' or '2W'."""
    cols = OAX; m = MASK['Medial OA']; jc = [J[x] for x in cols]; Ys = Y[m][:, jc].astype(int)
    obs = np.zeros((2,) * 5)
    for r in Ys: obs[tuple(r)] += 1
    obs /= obs.sum(); marg = Ys.mean(0)
    Tg = tab(Yg, cols) if T is None else T
    if kind == '32':
        tgt = rake(Tg + .5, marg)
        tgt = ns['shift_target'](tgt, Tg, cols, s, 0.)
    else:
        q = obs.copy(); tg = {}
        for i, j in itertools.combinations(range(5), 2):
            c = pair_marg(Tg, i, j).ravel() + .5; lor = np.log(c[0] * c[3] / (c[1] * c[2])); se = np.sqrt((1 / c).sum())
            sh = s if (cols[i] in OA and cols[j] in OA) else 0.
            tg[(i, j)] = or_cells(marg[i], marg[j], lor + sh * se)
        for _ in range(300):
            for (i, j), t in tg.items():
                cur = pair_marg(q, i, j); shp = [1] * 5; shp[i] = 2; shp[j] = 2
                q = q * (t / np.maximum(cur, 1e-15)).reshape(shp)
            q /= q.sum()
        tgt = q
    w = (tgt / obs)[tuple(Ys.T)]
    return m, w / w.mean()


def WT_for(kind, Yg, s, T=None):
    WT = {}
    for c in L:
        if c in OA:
            WT[c] = target(kind, Yg, s, T)
        else:
            mm = ~np.isnan(Y[:, J[c]]); WT[c] = (mm, np.ones(mm.sum()))
    return WT


def dl(outs, WT, ii=None):
    b = outs['pw30']; res = {}
    cache = {}
    for k, o in outs.items():
        d = 0.
        for j in [J[c] for c in OA]:
            if np.array_equal(o[:, j], b[:, j]):
                continue
            m, w = WT[L[j]]
            if ii is None:
                sel = np.where(m)[0]; ww = w
            else:
                full = np.full(n, np.nan); full[m] = w; sel = ii[m[ii]]; ww = full[sel]
            if (j, 'b') not in cache:
                cache[(j, 'b')] = wauc(b[sel, j], YF[sel, j], ww)
            d += wauc(o[sel, j], YF[sel, j], ww) - cache[(j, 'b')]
        res[k] = d / 12 * 1e4
    return res


OUTS = {nn: {k: pipeline(dict(rap=[(1., X[ra])], r6=X[r6], s6=X[s6]), c) for k, c in VAR.items()} for nn, (ra, r6, s6) in NESTS.items()}

# ---- 0. gold trio 3-way structure ----
P('== 0. gold trio 3-way check (counts are local aggregates): all-three-OA-positive observed vs pairwise max-entropy prediction ==')
for gn, Yg in GSETS.items():
    T3 = tab(Yg, OA); N3 = T3.sum(); obs3 = T3[1, 1, 1]
    q = np.full((2, 2, 2), 1 / 8.)
    for _ in range(500):
        for i, j in itertools.combinations(range(3), 2):
            cur = pair_marg(q, i, j); t = pair_marg(T3 + .5 / 8, i, j); t = t / t.sum(); shp = [1] * 3; shp[i] = 2; shp[j] = 2
            q = q * (t / cur).reshape(shp)
        q /= q.sum()
    P('  %s n %d | all-3+ observed %d | pairwise max-ent predicts %.1f | exactly-2+ observed %d vs %.1f | P(LOA+ | MOA+ PF+) %.2f vs P(LOA+ | MOA+) %.2f' % (
        gn, N3, obs3, q[1, 1, 1] * N3, T3[1, 1, 0] + T3[1, 0, 1] + T3[0, 1, 1], (q[1, 1, 0] + q[1, 0, 1] + q[0, 1, 1]) * N3,
        T3[1, 1, 1] / max(T3[1, :, 1].sum(), 1), T3[1, 1, :].sum() / max(T3[1].sum(), 1)))
Ytr = Y[MASK['Medial OA']][:, [J[c] for c in OA]].astype(int); T3 = np.zeros((2, 2, 2))
for r in Ytr: T3[tuple(r)] += 1
P('  train reports n %d | all-3+ share %.3f | P(LOA+ | MOA+ PF+) %.2f vs P(LOA+ | MOA+) %.2f' % (T3.sum(), T3[1, 1, 1] / T3.sum(), T3[1, 1, 1] / T3[1, :, 1].sum(), T3[1, 1, :].sum() / T3[1].sum()))

# ---- 1. per target: ladder fit, then the CVX family at s* ----
P('\n== 1. per target: trio shift s* fitted to the public ladder (grid), then deltas vs pw30 (e-4) at s* (nest mean [min, max]) ==')
RES = {}
for tn, (kind, gk) in {'T32': ('32', 'g74'), 'T2W': ('2W', 'g74'), 'T58': ('32', 'g58'), 'T58_2W': ('2W', 'g58')}.items():
    best = None
    for s in (-0.2, 0., 0.2, 0.4, 0.6):
        WT = WT_for(kind, GSETS[gk], s); lads = []
        for nn in NESTS:
            d = dl({k: OUTS[nn][k] for k in list(LADDER) + ['pw30']}, WT); lads.append({k: d[k] - d['c4'] for k in PUB if k != 'pw30'} | {'pw30': -d['c4']})
        ml = {k: np.mean([l_[k] for l_ in lads]) for k in PUB}; sse = sum((ml[k] - PUB[k]) ** 2 for k in PUB)
        P('  %-7s s %+.1f ladder %s | SSE %.1f' % (tn, s, ' '.join('%s %+.2f' % (k, ml[k]) for k in PUB), sse))
        if best is None or sse < best[1]:
            best = (s, sse)
    s = best[0]; WT = WT_for(kind, GSETS[gk], s); rows = {k: [] for k in SHOW}
    for nn in NESTS:
        d = dl(OUTS[nn], WT)
        for k in SHOW: rows[k].append(d[k])
    RES[tn] = {k: np.mean(v) for k, v in rows.items()}
    P('  %-7s @ s* %+.1f (SSE %.1f): ' % (tn, s, best[1]) + ' | '.join('%s %+.2f [%+.2f,%+.2f]' % (k, np.mean(v), min(v), max(v)) for k, v in rows.items()))
    P('          CVX marginals: on pw25 %+.2f | on spl %+.2f | on pw30 %+.2f | CVXd on pw30 %+.2f | CVXd on pw25 %+.2f | p1.5 on pw25 %+.2f | p3 on pw25 %+.2f' % (
        RES[tn]['pw25+CVX'] - RES[tn]['pw25'], RES[tn]['spl+CVX'] - RES[tn]['spl'], RES[tn]['pw30+CVX'], RES[tn]['pw30+CVXd'],
        RES[tn]['pw25+CVXd'] - RES[tn]['pw25'], RES[tn]['pw25+P15'] - RES[tn]['pw25'], RES[tn]['pw25+P3'] - RES[tn]['pw25']))

# ---- 2. RAK-u (T32 Dirichlet target x row bootstrap) at the T32 s* ----
P('\n== 2. RAK-u: T32 at s* +0.2 with Dirichlet(gold74 32-cell + .5) target draws x row bootstrap (B 150 per nest) ==')
T32 = tab(GSETS['g74'], OAX); rng = np.random.default_rng(23)
for nn in NESTS:
    acc = {k: [] for k in ('pw25+CVX', 'spl+CVX', 'pw30+CVX', 'pw25', 'spl', 'cvx25', 'cvxspl')}
    sub = {k: OUTS[nn][k] for k in ('pw30', 'pw25', 'spl', 'pw25+CVX', 'spl+CVX', 'pw30+CVX')}
    for b in range(150):
        Td = rng.dirichlet((T32 + .5).ravel()).reshape(T32.shape) * T32.sum()
        WT = WT_for('32', GSETS['g74'], 0.2, T=np.maximum(Td - .5, 0)); ii = rng.integers(0, n, n)
        d = dl(sub, WT, ii)
        for k in ('pw25+CVX', 'spl+CVX', 'pw30+CVX', 'pw25', 'spl'): acc[k].append(d[k])
        acc['cvx25'].append(d['pw25+CVX'] - d['pw25']); acc['cvxspl'].append(d['spl+CVX'] - d['spl'])
    A = {k: np.array(v) for k, v in acc.items()}
    st = lambda x: '%+.2f [%+.2f, %+.2f] P>0 %.2f' % (x.mean(), np.percentile(x, 10), np.percentile(x, 90), (x > 0).mean())
    P('  %s | CVX on pw25 %s | CVX on spl %s | CVX on pw30 %s | pw25 %s' % (nn, st(A['cvx25']), st(A['cvxspl']), st(A['pw30+CVX']), st(A['pw25'])))

# ---- 3. leave-one-gold-study-out (T32, s +0.2, nest N_A, point) ----
P('\n== 3. leave-one-gold-study-out influence on CVX on pw25 (T32 @ s +0.2, N_A, point, e-4) ==')
Yg = GSETS['g74']; sub = {k: OUTS['N_A'][k] for k in ('pw30', 'pw25', 'pw25+CVX')}
full = dl(sub, WT_for('32', Yg, .2)); base = full['pw25+CVX'] - full['pw25']; loo = []
for i in range(len(Yg)):
    Yi = np.delete(Yg, i, axis=0); d = dl(sub, WT_for('32', Yi, .2)); loo.append(d['pw25+CVX'] - d['pw25'])
loo = np.array(loo); infl = base - loo; src_ = g.source.values; o = np.argsort(-infl)
P('  full %+.2f | LOO min %+.2f max %+.2f | > 0 in %d / %d | top-5 study influences %s | share of the gain from the top 5 %.0f %% | extra16 total influence %+.2f' % (
    base, loo.min(), loo.max(), (loo > 0).sum(), len(loo), ' '.join('%+.2f' % x for x in infl[o[:5]]), 100 * infl[o[:5]].sum() / base if base else 0,
    infl[src_ != 'original_gold58'].sum()))
P('SINK_CVX_SKEP DONE %.0f s' % (time.time() - T0))
