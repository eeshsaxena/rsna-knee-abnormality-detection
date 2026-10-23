# sink_value_mc.py (10-21 ~12:00 IST, Claude, SINK track). MODEL only (arithmetic on stated inputs; no data read).
# 10-22 kitchen-sink (KS) / conservative (CONS) entries on top of the A22 slate, conditional on which A22 entry becomes the best-id W.
# World: stack/skeptic/skep_value_mc.py world() VERBATIM (its source is exec'd up to its first report line, log redirected) for pw25
# (val/implied2 abc3 conditional on the A21 reads), lomm05 (scenario prior), MCLmm (prior variants), I(pw25, lomm05), read noise, gap.
# Its component reads are recovered exactly from its entries (lomm05 = entry 'lomm05', MCLmm = 'S_safe', I = S_pl* - pw25 - lomm05).
# Added here (separate RNG, so world()'s draws are untouched; asserted):
#   CVX = convex E-OA exponent p 2 (sink_local / sink_cvx_skep / box_sink, sink.md sec 2). Population effect on a pw25-family base mu25 and
#   on the pw30 base mu30 (corr .8); I(CVX, lomm05) on the spl / swide bases; MCLmm x CVX = 0 exactly (disjoint columns).
#   CVX priors (mu25 / mu30, e-4):  ruler N(+1.5, .9) / N(+0.9, 1.0) | primary N(+0.5, 1.0) / N(+0.2, 1.1) | lskep (levers skeptic)
#   N(+0.2, .8) / N(-0.1, .9) | zero | neg (gold58-led) N(-1.0, 1.0) / N(-1.2, 1.0). I(CVX, lomm05) N(-0.2, .3) (neg prior: N(-0.8, .4)).
#   CVX read sd public / private: 2.6 / 1.5 on pw25-family, 3.4 / 1.9 on pw30 (MEASURED sink_read_noise.log), corr(c25, c30) .93,
#   corr with the lomm05 read +.14 (MEASURED); corr with the pw25 read -.38 (MEASURED) used only in sensitivity S2.
# R1: A = argmax public over {pw30 = 0, pw25, spl, swide, salt} plus the 10-22 entries; value = A's private (e-4 vs pw30).
# KS / CONS by W: pw30 -> KS cvx_lomm30 (pw30 + lomm05 + CVX), CONS cvx_pw30 | pw25 -> KS cvx_spl, CONS cvx_pw25 | spl -> cvx_spl (both) |
#   salt -> KS cvx_swide, CONS cvx_salt | swide -> cvx_swide (both). r6pw20 is B, never A (left out).
import numpy as np, io, itertools, json
SRC = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/skeptic/skep_value_mc.py'
LOGP = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/sink_value_mc.log'
src = open(SRC, encoding='utf-8').read(); src = src[:src.index("P('MODEL skep_value_mc.py")]
src = src.replace("OUT = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/skeptic/skep_value_mc.log', 'w')", "OUT = io.StringIO()")
assert 'OUT = io.StringIO()' in src
g = {'io': io}; exec(compile(src, SRC, 'exec'), g); world, slate, N = g['world'], g['slate'], g['N']
LG = open(LOGP, 'w')
def P(s=''):
    print(s, flush=True); LG.write(s + '\n')
CVXP = {'ruler': ((1.5, .9), (0.9, 1.0), (-0.2, .3)), 'primary': ((0.5, 1.0), (0.2, 1.1), (-0.2, .3)), 'lskep': ((0.2, .8), (-0.1, .9), (-0.2, .3)),
        'zero': ((0., 0.), (0., 0.), (0., 0.)), 'neg': ((-1.0, 1.0), (-1.2, 1.0), (-0.8, .4))}
A22 = ['pw25', 'spl', 'swide', 'salt']
KS = {'pw30': 'cvx_lomm30', 'pw25': 'cvx_spl', 'spl': 'cvx_spl', 'salt': 'cvx_swide', 'swide': 'cvx_swide'}
CONS = {'pw30': 'cvx_pw30', 'pw25': 'cvx_pw25', 'spl': 'cvx_spl', 'salt': 'cvx_salt', 'swide': 'cvx_swide'}
WN = ['pw30'] + A22


def world2(prior, cvx, gapm=(.75, 1.8), lomm='stack', seed=2110, s2=False, sd_scale=1.0):
    W = world(prior, gapm, lomm, seed); pub, pri, mu = W['pub'], W['pri'], W['mu']
    # recover component reads exactly from world()'s entries
    c = {}
    for side, D in (('pub', pub), ('pri', pri)):
        c[side] = dict(pw25=D['pw25'], lomm05=D['lomm05'], mcl=D['S_safe'], I=D['S_pl*'] - D['pw25'] - D['lomm05'])
        assert np.allclose(D['S_wide'], c[side]['pw25'] + c[side]['lomm05'] + c[side]['mcl'] + c[side]['I'])
        assert np.allclose(D['S_alt'], c[side]['pw25'] + c[side]['mcl'])
    rng = np.random.default_rng(seed + 7)
    (m25, s25), (m30, s30), (mi, si) = CVXP[cvx]
    z1, z2, z3 = rng.standard_normal((3, N))
    mu25 = m25 + s25 * z1; mu30 = m30 + s30 * (.8 * z1 + .6 * z2); Icl = mi + si * z3
    zl = (pub['lomm05'] - mu['lomm05']) / 1.00; zlq = (pri['lomm05'] - mu['lomm05']) / .62
    z25 = (pub['pw25'] - .09) / .80
    def err(sd_a, sd_b, zcorr):
        u1, u2 = rng.standard_normal((2, N))
        if s2:
            ea = .14 * zcorr[0] - .38 * zcorr[1] + np.sqrt(max(1 - .14 ** 2 - .38 ** 2, .01)) * u1
        else:
            ea = .14 * zcorr[0] + np.sqrt(1 - .14 ** 2) * u1
        eb = .93 * ea + np.sqrt(1 - .93 ** 2) * u2
        return sd_a * sd_scale * ea, sd_b * sd_scale * eb
    ep25, ep30 = err(2.6, 3.4, (zl, z25)); eq25, eq30 = err(1.5, 1.9, (zlq, np.zeros(N)))
    for side, e25, e30 in (('pub', ep25, ep30), ('pri', eq25, eq30)):
        k = c[side]; D = pub if side == 'pub' else pri
        c25 = mu25 + e25; c30 = mu30 + e30
        D['spl'] = D['S_pl*']; D['swide'] = D['S_wide']; D['salt'] = D['S_alt']; D['pw30'] = np.zeros(N)
        D['cvx_pw30'] = c30
        D['cvx_lomm30'] = k['lomm05'] + c30 + Icl
        D['cvx_pw25'] = k['pw25'] + c25
        D['cvx_spl'] = k['pw25'] + k['lomm05'] + k['I'] + c25 + Icl
        D['cvx_salt'] = k['pw25'] + k['mcl'] + c25
        D['cvx_swide'] = k['pw25'] + k['lomm05'] + k['I'] + k['mcl'] + c25 + Icl
    W['mu']['cvx25'] = mu25; W['mu']['cvx30'] = mu30
    return W


def pick(W, ents):
    PU = np.column_stack([W['pub'][e] for e in ents]); PR = np.column_stack([W['pri'][e] for e in ents])
    a = PU.argmax(1); r = np.arange(N)
    return a, PR[r, a], PU[r, a]


def run(prior, cvx, lomm='stack', gapm=(.75, 1.8), s2=False, sd_scale=1.0, detail=False):
    W = world2(prior, cvx, gapm, lomm, s2=s2, sd_scale=sd_scale); gap = W['gap']
    a0, pr0, pu0 = pick(W, WN); Wi = np.array(WN)[a0]
    res = dict(base=dict(E=pr0.mean(), p960=(pu0 > gap).mean(), Aneg=(pr0 < 0).mean()))
    for pol in ('CONS', 'KS', 'BOTH'):
        pr = pr0.copy(); pu = pu0.copy(); isnew = np.zeros(N, bool)
        for w in WN:
            m = Wi == w
            if not m.any():
                continue
            ex = [CONS[w]] if pol == 'CONS' else [KS[w]] if pol == 'KS' else sorted({CONS[w], KS[w]})
            ents = WN + ex; PU = np.column_stack([W['pub'][e][m] for e in ents]); PR = np.column_stack([W['pri'][e][m] for e in ents])
            a = PU.argmax(1); r = np.arange(m.sum()); pr[m] = PR[r, a]; pu[m] = PU[r, a]; isnew[m] = a >= len(WN)
        res[pol] = dict(E=pr.mean(), inc=pr.mean() - pr0.mean(), p960=(pu > gap).mean(), Aneg=(pr < 0).mean(), pnew=isnew.mean(),
                        byW={w: dict(P=(Wi == w).mean(), inc=(pr - pr0)[Wi == w].mean(), p960_0=(pu0 > gap)[Wi == w].mean(), p960=(pu > gap)[Wi == w].mean(),
                                     pnew=isnew[Wi == w].mean(), Enew=((pr - pr0)[(Wi == w) & isnew]).mean() if (isnew & (Wi == w)).any() else 0.)
                             for w in WN})
    if detail:
        res['W'] = W; res['Wi'] = Wi
    return res


P('MODEL sink_value_mc.py | N %d | e-4 macro vs pw30 57425970 | world() = stack/skeptic/skep_value_mc.py verbatim + CVX (sink.md) | R1 over {pw30, pw25, spl, swide, salt} + 10-22 entries' % N)
P('CVX priors (mu on pw25-family / mu on pw30 / I(CVX, lomm05)): ' + ' | '.join('%s %s' % (k, v) for k, v in CVXP.items()))

# ---- 1. per W, primary (MCLmm skep2, lomm05 stack, CVX primary) ----
R0 = run('skep2', 'primary')
P('\n== 1. PRIMARY (MCLmm skep2, lomm05 stack prior, CVX primary): A22 best-id W, and the 10-22 increment of E private of A by policy ==')
P('   A22 alone: E priv A %+.2f | P(.960 shown) %.2f | P(A < pw30) %.2f' % (R0['base']['E'], R0['base']['p960'], R0['base']['Aneg']))
for pol in ('CONS', 'KS', 'BOTH'):
    r = R0[pol]
    P('   %-4s: E %+.2f (incr %+.2f) | P(.960) %.2f -> %.2f | P(A<pw30) %.2f | P(10-22 entry becomes A) %.2f' % (pol, r['E'], r['inc'], R0['base']['p960'], r['p960'], r['Aneg'], r['pnew']))
P('   by W (P(W) | incr CONS / KS / BOTH | P(.960) A22 -> CONS / KS / BOTH | P(new = A) CONS / KS | E gain when new = A, CONS / KS):')
for w in WN:
    b = R0['CONS']['byW'][w]; k = R0['KS']['byW'][w]; bo = R0['BOTH']['byW'][w]
    P('   W %-6s P %.2f | %+.2f / %+.2f / %+.2f | %.2f -> %.2f / %.2f / %.2f | %.2f / %.2f | %+.2f / %+.2f   [CONS %s, KS %s]' % (
        w, b['P'], b['inc'], k['inc'], bo['inc'], b['p960_0'], b['p960'], k['p960'], bo['p960'], b['pnew'], k['pnew'], b['Enew'], k['Enew'], CONS[w], KS[w]))

# ---- 2. robustness grid ----
P('\n== 2. increment of E private of A (e-4) by prior: rows = CVX prior, columns = MCLmm prior x lomm05 prior; cell = CONS / KS / BOTH ==')
COLS = [('skep2', 'stack'), ('skep2', 'skeptic'), ('stack', 'stack'), ('zero', 'stack'), ('zero', 'skeptic')]
P('   %-8s | ' % 'CVX' + ' | '.join('%-24s' % ('MCL %s, lomm %s' % c) for c in COLS))
GRID = {}
for cv in CVXP:
    cells = []
    for mp, lp in COLS:
        r = run(mp, cv, lp); GRID[(cv, mp, lp)] = r
        cells.append('%+.2f / %+.2f / %+.2f' % (r['CONS']['inc'], r['KS']['inc'], r['BOTH']['inc']))
    P('   %-8s | ' % cv + ' | '.join('%-24s' % x for x in cells))
P('   P(.960 shown): A22 alone vs + CONS / KS / BOTH, by CVX prior (MCLmm skep2, lomm05 stack):')
for cv in CVXP:
    r = GRID[(cv, 'skep2', 'stack')]
    P('   %-8s %.2f -> %.2f / %.2f / %.2f | P(A < pw30) %.2f -> %.2f / %.2f / %.2f' % (cv, r['base']['p960'], r['CONS']['p960'], r['KS']['p960'], r['BOTH']['p960'],
      r['base']['Aneg'], r['CONS']['Aneg'], r['KS']['Aneg'], r['BOTH']['Aneg']))

# ---- 3. sensitivities ----
P('\n== 3. sensitivities (MCLmm skep2, lomm05 stack): S1 LB-only gap; S2 CVX read error corr -.38 with the pw25 read; S3 CVX read sd x 1.3; S4 x .7 ==')
for nm, kw in (('S1 gap LB-only', dict(gapm=(.5, 1.3))), ('S2 corr pw25 -.38', dict(s2=True)), ('S3 read sd x1.3', dict(sd_scale=1.3)), ('S4 read sd x.7', dict(sd_scale=.7))):
    row = []
    for cv in ('primary', 'lskep', 'zero'):
        r = run('skep2', cv, **kw); row.append('%s %+.2f / %+.2f / %+.2f (P960 %.2f->%.2f)' % (cv, r['CONS']['inc'], r['KS']['inc'], r['BOTH']['inc'], r['base']['p960'], r['KS']['p960']))
    P('   %-18s ' % nm + ' | '.join(row))

# ---- 4. by-W robustness: CONS increment conditional on W under each CVX prior ----
P('\n== 4. CONS / KS increment conditional on W, by CVX prior (MCLmm skep2, lomm05 stack) ==')
for cv in CVXP:
    r = GRID[(cv, 'skep2', 'stack')]
    P('   %-8s ' % cv + ' | '.join('%s %+.2f/%+.2f' % (w, r['CONS']['byW'][w]['inc'], r['KS']['byW'][w]['inc']) for w in WN))
P('   same under lomm05 skeptic N(0, .9):')
for cv in CVXP:
    r = GRID[(cv, 'skep2', 'skeptic')]
    P('   %-8s ' % cv + ' | '.join('%s %+.2f/%+.2f' % (w, r['CONS']['byW'][w]['inc'], r['KS']['byW'][w]['inc']) for w in WN))

# ---- 5. what the 10-22 entry would show: E public / private of the stack itself vs W, primary ----
R = run('skep2', 'primary', detail=True); W, Wi = R['W'], R['Wi']
P('\n== 5. the stack itself, primary prior: E public / private vs W (e-4) [p10, p90], P(public > W), P(stack shows .960 | W does not) ==')
q = lambda x: '%+.2f [%+.2f, %+.2f]' % (x.mean(), np.percentile(x, 10), np.percentile(x, 90))
for w in WN:
    m = Wi == w
    for lab, X in (('CONS', CONS[w]), ('KS', KS[w])):
        if lab == 'KS' and KS[w] == CONS[w]:
            continue
        dpu = W['pub'][X][m] - W['pub'][w][m]; dpr = W['pri'][X][m] - W['pri'][w][m]
        nshow = W['pub'][w][m] <= W['gap'][m]
        P('   W %-6s %-4s %-12s public %s | private %s | P(pub > W) %.2f | P(.960 | W not) %.2f' % (w, lab, X, q(dpu), q(dpr), (dpu > 0).mean(),
          (W['pub'][X][m][nshow] > W['gap'][m][nshow]).mean() if nshow.any() else float('nan')))
json.dump({'%s|%s|%s' % k: {p: {kk: vv for kk, vv in v[p].items() if kk != 'byW'} for p in ('base', 'CONS', 'KS', 'BOTH')} for k, v in GRID.items()},
          open(LOGP.replace('.log', '.json'), 'w'), indent=0, default=float)
P('SINK_VALUE_MC DONE')
LG.close()
