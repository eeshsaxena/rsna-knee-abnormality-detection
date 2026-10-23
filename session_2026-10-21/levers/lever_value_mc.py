# lever_value_mc.py (10-21 ~12:00 IST, Claude, LEVERS synthesis). MODEL only (arithmetic on stated inputs; no data read).
# One optional A-eligible 10-22 entry on top of the A22 slate, conditional on the A22 best-id W: MOACF(W) vs CONS(W) (CVX) vs both.
# World: stack/skeptic/skep_value_mc.py world() VERBATIM (exec'd up to its first report line, as sink_value_mc.py does). Component
# reads recovered exactly from its entries (asserted). Added with separate RNGs (world()'s draws untouched):
#   MOACF: one population effect mu_m for every base (MOA column only; disjoint from lomm05 / MCLmm / dose, so no interaction; the ruler
#     gain is base-independent: regime_sens nests +38..+45 label on pw30 / pw25 / spl / swide, MEASURED). Read sd public / private
#     1.8 / 1.15 (rulers, MEASURED moacf_noise.log); sensitivity 1.5 / 1.0 (c4 tighter, GUESS as skep_moacf_r1.py). Read errors of the
#     MOACF entry independent of the other reads (GUESS: different column), corr .95 across bases (same MOA moves).
#     Priors (e-4): skep = skep_moacf_r1.py base mixture (w .65 lognormal median .55 sdlog .6 | w .25 N(-.1, .05) | w .10 N(-.5, .3));
#     optim / pess = its optim / pess mixtures; regime = N(+0.8, 1.37) (regime.md [-1.0, +2.5] as p10 / p90); zero; flip = N(-0.5, .5).
#   CVX: as sink_value_mc.py (mu25 / mu30 corr .8, I(CVX, lomm05), read sd 2.6 / 1.5 and 3.4 / 1.9, corr with lomm05 read +.14), priors:
#     rk = rakpub-skeptic corrected N(+0.3, 1.1) / N(+0.1, 1.1) (rakpub_skep vote) | lskep N(+0.2, .8) / N(-0.1, .9) | primary
#     N(+0.5, 1.0) / N(+0.2, 1.1) | zero.
# R1: A = argmax public over {pw30, pw25, spl, swide, salt} + the 10-22 entries; value = A's private (e-4 vs pw30).
# H2 view: when W in {pw30, pw25}, H2 makes A = pw20 and a 10-22 entry has no A role; only the stack-class rows count ("stack-only").
import numpy as np, io
SRC = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/skeptic/skep_value_mc.py'
LOGP = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/lever_value_mc.log'
src = open(SRC, encoding='utf-8').read(); src = src[:src.index("P('MODEL skep_value_mc.py")]
src = src.replace("OUT = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/skeptic/skep_value_mc.log', 'w')", "OUT = io.StringIO()")
assert 'OUT = io.StringIO()' in src
g = {'io': io}; exec(compile(src, SRC, 'exec'), g); world, N = g['world'], g['N']
LG = open(LOGP, 'w')
def P(s=''):
    print(s, flush=True); LG.write(s + '\n')
WN = ['pw30', 'pw25', 'spl', 'swide', 'salt']; STACK = {'spl', 'swide', 'salt'}
CVXP = {'rk': ((0.3, 1.1), (0.1, 1.1), (-0.2, .3)), 'lskep': ((0.2, .8), (-0.1, .9), (-0.2, .3)), 'primary': ((0.5, 1.0), (0.2, 1.1), (-0.2, .3)),
        'zero': ((0., 0.), (0., 0.), (0., 0.))}
MIX = {'skep': ((.65, .25, .10), .55), 'optim': ((.75, .18, .07), .8), 'pess': ((.5, .35, .15), .4)}


def mo_prior(name, rng):
    if name in MIX:
        w, med = MIX[name]; k = rng.choice(3, N, p=w)
        return np.where(k == 0, med * np.exp(.6 * rng.standard_normal(N)), np.where(k == 1, -.1 + .05 * rng.standard_normal(N), -.5 + .3 * rng.standard_normal(N)))
    if name == 'regime':
        return .8 + 1.37 * rng.standard_normal(N)
    if name == 'flip':
        return -.5 + .5 * rng.standard_normal(N)
    return np.zeros(N)


def world3(moa, cvx, mcl='skep2', lomm='stack', gapm=(.75, 1.8), seed=2110, msd=(1.8, 1.15), rho0=False):
    W = world(mcl, gapm, lomm, seed); pub, pri, mu = W['pub'], W['pri'], W['mu']
    c = {}
    for side, D in (('pub', pub), ('pri', pri)):
        c[side] = dict(pw25=D['pw25'], lomm05=D['lomm05'], mcl=D['S_safe'], I=D['S_pl*'] - D['pw25'] - D['lomm05'])
        assert np.allclose(D['S_wide'], c[side]['pw25'] + c[side]['lomm05'] + c[side]['mcl'] + c[side]['I'])
        assert np.allclose(D['S_alt'], c[side]['pw25'] + c[side]['mcl'])
    rng = np.random.default_rng(seed + 7); rm = np.random.default_rng(seed + 11)
    (m25, s25), (m30, s30), (mi, si) = CVXP[cvx]
    z1, z2, z3 = rng.standard_normal((3, N))
    mu25 = m25 + s25 * z1; mu30 = m30 + s30 * (.8 * z1 + .6 * z2); Icl = mi + si * z3
    zl = (pub['lomm05'] - mu['lomm05']) / 1.00; zlq = (pri['lomm05'] - mu['lomm05']) / .62
    def err(sd_a, sd_b, zc):
        u1, u2 = rng.standard_normal((2, N)); ea = .14 * zc + np.sqrt(1 - .14 ** 2) * u1; eb = .93 * ea + np.sqrt(1 - .93 ** 2) * u2
        return sd_a * ea, sd_b * eb
    ep25, ep30 = err(2.6, 3.4, zl); eq25, eq30 = err(1.5, 1.9, zlq)
    mum = mo_prior(moa, rm); mum_q = mo_prior(moa, rm) if rho0 else mum   # rho0: private population effect independent of public (skeptic_wc rhoA0 analogue)
    if rho0:
        y1, y2 = rng.standard_normal((2, N)); mu25q = m25 + s25 * y1; mu30q = m30 + s30 * (.8 * y1 + .6 * y2)
    else:
        mu25q, mu30q = mu25, mu30
    def merr(sd):
        base = rm.standard_normal(N); return {b: sd * (.95 * base + np.sqrt(1 - .95 ** 2) * rm.standard_normal(N)) for b in WN}
    em_pub, em_pri = merr(msd[0]), merr(msd[1])
    for side, e25, e30, em, M, M25, M30 in (('pub', ep25, ep30, em_pub, mum, mu25, mu30), ('pri', eq25, eq30, em_pri, mum_q, mu25q, mu30q)):
        k = c[side]; D = pub if side == 'pub' else pri
        c25 = M25 + e25; c30 = M30 + e30
        D['spl'] = D['S_pl*']; D['swide'] = D['S_wide']; D['salt'] = D['S_alt']; D['pw30'] = np.zeros(N)
        D['cvx_pw30'] = c30; D['cvx_pw25'] = k['pw25'] + c25
        D['cvx_spl'] = k['pw25'] + k['lomm05'] + k['I'] + c25 + Icl
        D['cvx_salt'] = k['pw25'] + k['mcl'] + c25
        D['cvx_swide'] = k['pw25'] + k['lomm05'] + k['I'] + k['mcl'] + c25 + Icl
        for b in WN:
            D['mo_' + b] = D[b] + M + em[b]
    W['mu']['moacf'] = mum
    return W


def run(moa, cvx, **kw):
    W = world3(moa, cvx, **kw); gap = W['gap']
    PU = np.column_stack([W['pub'][e] for e in WN]); PR = np.column_stack([W['pri'][e] for e in WN]); r = np.arange(N)
    a0 = PU.argmax(1); pr0 = PR[r, a0]; pu0 = PU[r, a0]; Wi = np.array(WN)[a0]
    res = dict(base=dict(E=pr0.mean(), p960=(pu0 > gap).mean(), Aneg=(pr0 < 0).mean()), PW={w: (Wi == w).mean() for w in WN})
    for pol, ex in (('MOACF', lambda w: ['mo_' + w]), ('CVX', lambda w: ['cvx_' + w]), ('BOTH', lambda w: ['mo_' + w, 'cvx_' + w])):
        pr = pr0.copy(); pu = pu0.copy(); new = np.zeros(N, bool)
        for w in WN:
            m = Wi == w; ents = WN + ex(w)
            Pu = np.column_stack([W['pub'][e][m] for e in ents]); Pr = np.column_stack([W['pri'][e][m] for e in ents]); a = Pu.argmax(1); rr = np.arange(m.sum())
            pr[m] = Pr[rr, a]; pu[m] = Pu[rr, a]; new[m] = a >= len(WN)
        d = pr - pr0; st = np.isin(Wi, list(STACK))
        res[pol] = dict(inc=d.mean(), inc_stack=(d * st).mean(), p960=(pu > gap).mean(), Aneg=(pr < 0).mean(), Aneg_stack=np.where(st, pr, pr0).__lt__(0).mean(),
                        pnew=new.mean(), worse=((d < 0) & new).mean(), byW={w: d[Wi == w].mean() for w in WN})
    return res


P('MODEL lever_value_mc.py | N %d | e-4 macro private vs pw30 57425970 | world() = stack/skeptic/skep_value_mc.py verbatim (MCLmm skep2, lomm05 stack unless stated)' % N)
P('policies: one 10-22 entry MOACF(W), or CONS(W) = W + CVX, or BOTH (two separate entries); R1 over A22 + entries. "stack-only" = entry made only when W in {spl, swide, salt} (H2).')
P('\n== 1. increment of E private of A, all W / stack-only (H2) | P(10-22 entry becomes A) | P(A < pw30) A22 -> with entry | P(.960 shown) A22 -> with entry ==')
for moa, cvx in (('skep', 'rk'), ('optim', 'rk'), ('pess', 'rk'), ('regime', 'rk'), ('zero', 'zero'), ('flip', 'lskep'), ('skep', 'lskep'), ('skep', 'primary')):
    R = run(moa, cvx)
    P('MOACF prior %-7s CVX prior %-8s | A22 alone E %+.2f P(.960) %.2f P(A<pw30) %.2f' % (moa, cvx, R['base']['E'], R['base']['p960'], R['base']['Aneg']))
    for pol in ('MOACF', 'CVX', 'BOTH'):
        x = R[pol]
        P('   %-5s incr %+.2f / stack-only %+.2f | P(new = A) %.2f | P(new = A and worse) %.2f | P(A<pw30) -> %.2f (stack-only %.2f) | P(.960) -> %.2f | by W %s' % (
            pol, x['inc'], x['inc_stack'], x['pnew'], x['worse'], x['Aneg'], x['Aneg_stack'], x['p960'], ' '.join('%s %+.2f' % (w, x['byW'][w]) for w in WN)))
P('\n== 2. sensitivities, MOACF prior skep / CVX rk: MOACF read sd 1.5 / 1.0 (c4 tighter); lomm05 skeptic prior; MCLmm zero ==')
for nm, kw in (('rho0 pub/pri indep', dict(rho0=True)), ('msd 1.5/1.0', dict(msd=(1.5, 1.0))), ('lomm05 skeptic', dict(lomm='skeptic')), ('MCLmm zero', dict(mcl='zero')), ('gap LB-only', dict(gapm=(.5, 1.3)))):
    R = run('skep', 'rk', **kw)
    P('   %-15s ' % nm + ' | '.join('%s %+.2f / stack-only %+.2f' % (p, R[p]['inc'], R[p]['inc_stack']) for p in ('MOACF', 'CVX', 'BOTH')))
P('\n== 3. rho0 (entry component private effect independent of its public effect) under other priors ==')
for moa, cvx in (('optim', 'primary'), ('pess', 'lskep'), ('regime', 'rk'), ('flip', 'lskep')):
    R = run(moa, cvx, rho0=True)
    P('   MOACF %-7s CVX %-8s ' % (moa, cvx) + ' | '.join('%s %+.2f / stack-only %+.2f (P new = A and worse %.2f)' % (p, R[p]['inc'], R[p]['inc_stack'], R[p]['worse']) for p in ('MOACF', 'CVX', 'BOTH')))
P('LEVER_VALUE_MC DONE')
LG.close()
