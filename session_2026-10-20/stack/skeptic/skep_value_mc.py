# skep_value_mc.py (10-21 ~01:00 IST, Claude, STACK SKEPTIC round). MODEL only (arithmetic on stated inputs; no data read).
# Re-derives the stack values of stack/kernels/stack_value_mc.py with a skeptic MCLmm prior built from MEASURED curves, plus the R1
# private-cost decomposition and a virtual (NOT built) S_pl = pw25 + lomm05 as a reference.
#
# MCLmm skeptic prior ("skep2"), MODEL on MEASURED inputs (stack/skeptic/skep_mcl_curve_gold58.log, skep_mcl_relabel.log):
#   state s (posterior on the hidden expert MCL~MM coupling; gold58 = the ONLY expert-labelled gold, HANDOFF 1795):
#     measured .25: theta N(-1.06, .85) | psi N(+1.53, 1.09)     (gold58 as measured; psi by report MM, as the train emulation drops)
#     mechanism .50: theta prior N(+0.2, .8) x data -> N(-0.39, .58) | psi prior N(+1.0, 1.0) x data -> N(+1.24, .74)
#     skeptic  .25: theta prior N(+0.9 = report, .6) x data -> N(+0.25, .49) | psi prior N(0, .7) x data -> N(+0.45, .59)
#   emulation e: reweight (inventory RAK, curve over theta) .5 | relabel (drop mechanism, curve over psi, f .5) .5
#   population p: train-like | gold58-like (density ratio C .05); tied to the lomm05 scenario (S1 -> train, S3 -> gold, S2 -> 50/50)
#   x RAK->public factor F ~ U(.45, .85)
# Everything else as stack_value_mc.py PRIMARY (pw25 = val/implied2 abc3 conditional on A21 reads; lomm05 scenario prior; I(pw25, lomm05)
# N(+.16, .12); read sd public / private pw25 1.70 / 1.00, lomm05 1.00 / .62, MCLmm 2.40 / 1.50, corr(pw25, lomm05) -.2; gap joint
# lognormal median .75 p90 1.8, LB-only median .5 p90 1.3).
import numpy as np, itertools
OUT = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/skeptic/skep_value_mc.log', 'w')


def P(s=''):
    print(s, flush=True); OUT.write(s + '\n')


N = 400000
TH = np.array([-1.5, -1.25, -0.95, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.91])
RW = {'train': np.array([5.10, 4.30, 3.27, 2.54, 1.62, 0.68, -0.26, -1.18, -2.07, -2.91, -3.42]),
      'gold': np.array([3.70, 2.55, 1.10, 0.10, -1.18, -2.49, -3.81, -5.12, -6.42, -7.69, -8.48])}
PS = np.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
RL = {'train': np.array([-5.14, -3.03, -1.95, -0.91, 0.02, 0.82, 1.47, 1.97, 2.34]),
      'gold': np.array([-10.26, -4.89, -2.14, 0.49, 2.90, 5.00, 6.76, 8.15, 9.20])}


def ip(x, xs, ys):
    y = np.interp(x, xs, ys); lo, hi = x < xs[0], x > xs[-1]
    y[lo] = ys[0] + (x[lo] - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0]); y[hi] = ys[-1] + (x[hi] - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return y


STATES = {'measured': (.25, (-1.06, .85), (1.53, 1.09)), 'mechanism': (.50, (-0.39, .58), (1.24, .74)), 'skeptic': (.25, (0.25, .49), (0.45, .59))}


def mcl_skep2(rng, popg):
    """popg: bool array, True = gold58-like population."""
    st = rng.choice(3, N, p=[v[0] for v in STATES.values()]); em = rng.random(N) < .5   # True = relabel
    th_m = np.array([v[1][0] for v in STATES.values()])[st]; th_s = np.array([v[1][1] for v in STATES.values()])[st]
    ps_m = np.array([v[2][0] for v in STATES.values()])[st]; ps_s = np.array([v[2][1] for v in STATES.values()])[st]
    th = rng.normal(th_m, th_s); ps = rng.normal(ps_m, ps_s); F = rng.uniform(.45, .85, N)
    g_rw = np.where(popg, ip(th, TH, RW['gold']), ip(th, TH, RW['train'])); g_rl = np.where(popg, ip(ps, PS, RL['gold']), ip(ps, PS, RL['train']))
    return F * np.where(em, g_rl, g_rw), st, em


ENT = {'pw25': ['pw25'], 'lomm05': ['lomm05'], 'S_safe': ['MCLmm'], 'S_alt': ['pw25', 'MCLmm'], 'S_wide': ['pw25', 'lomm05', 'MCLmm'],
       'S_mid': ['lomm05', 'MCLmm'], 'S_pl*': ['pw25', 'lomm05']}
STACKS = ['S_safe', 'S_alt', 'S_wide', 'S_mid']
q = lambda x: '%+.2f [%+.2f, %+.2f]' % (x.mean(), np.percentile(x, 10), np.percentile(x, 90))


def world(prior, gapm=(.75, 1.8), lomm='stack', seed=2110):
    rng = np.random.default_rng(seed)
    sc = rng.choice(3, N, p=[.40, .35, .25])
    def scen(means, sds, hc, s=sc):
        return hc * (np.array(means)[s] + np.array(sds)[s] * rng.standard_normal(N))
    mu = {}
    mu['lomm05'] = scen([1.10, -.48, -.26], [1.2, 1.2, 1.6], .85) if lomm == 'stack' else rng.normal(0., .9, N)
    popg = np.where(sc == 0, False, np.where(sc == 2, True, rng.random(N) < .5))
    extra = {}
    if prior == 'stack':
        mu['MCLmm'] = scen([2.30, 0., 1.30], [1.0, .8, 1.2], .75)
    elif prior == 'pessimistic':
        sc2 = rng.choice(3, N, p=[.25, .50, .25]); mu['MCLmm'] = scen([2.30, 0., 1.30], [1.0, .8, 1.2], .60, sc2)
    elif prior == 'skeptic-null':
        mu['MCLmm'] = rng.normal(0., 1.0, N)
    elif prior == 'recal':
        mu['MCLmm'] = rng.normal(-.2, 1.0, N)
    elif prior == 'skep2':
        mu['MCLmm'], extra['st'], extra['em'] = mcl_skep2(rng, popg)
    elif prior == 'zero':
        mu['MCLmm'] = np.zeros(N)
    elif prior == 'neg':
        mu['MCLmm'] = rng.normal(-.5, 1.0, N)
    extra['popg'] = popg
    I = rng.normal(.16, .12, N) if lomm == 'stack' else rng.normal(0., .12, N)   # lomm05 skeptic: the super-additivity rides on the C12' mechanism
    sp, sq, rho = {'pw25': 1.70, 'lomm05': 1.00, 'MCLmm': 2.40}, {'pw25': 1.00, 'lomm05': .62, 'MCLmm': 1.50}, -.20
    def errs(sd):
        z1, z2, z3 = rng.standard_normal((3, N))
        return {'pw25': sd['pw25'] * z1, 'lomm05': sd['lomm05'] * (rho * z1 + np.sqrt(1 - rho ** 2) * z2), 'MCLmm': sd['MCLmm'] * z3}
    ep, eq = errs(sp), errs(sq)
    med, p90 = gapm
    gap = np.exp(rng.normal(np.log(med), (np.log(p90) - np.log(med)) / 1.2816, N))
    z1, z2, z3 = rng.standard_normal((3, N)); rho25 = .17
    pub25 = .09 + .80 * z1; mu['pw25'] = .50 + 1.45 * (rho25 * z1 + np.sqrt(1 - rho25 ** 2) * z2); pri25 = mu['pw25'] + 1.32 * z3
    def ent(parts, e, x25):
        return sum((x25 if p == 'pw25' else mu[p] + e[p]) for p in parts) + (I if {'pw25', 'lomm05'} <= set(parts) else 0.)
    pub = {k: ent(v, ep, pub25) for k, v in ENT.items()}; pri = {k: ent(v, eq, pri25) for k, v in ENT.items()}
    return dict(pub=pub, pri=pri, mu=mu, gap=gap, I=I, extra=extra)


def slate(W, es):
    PU = np.column_stack([np.zeros(N)] + [W['pub'][e] for e in es]); PR = np.column_stack([np.zeros(N)] + [W['pri'][e] for e in es])
    a = PU.argmax(1); r = np.arange(N)
    return dict(a=a, priv=PR[r, a], pub=PU[r, a], p960=(PU[r, a] > W['gap']).mean(), show=PU[r, a] > W['gap'])


P('MODEL skep_value_mc.py | N %d | e-4 macro vs pw30 57425970 | [p10, p90] | R1: A = max public among {pw30, A probes}; r6pw20 = B, never A' % N)
P('S_pl* = pw25 + lomm05 is VIRTUAL (not built), shown as the MCLmm-free reference for S_wide.')

# ---- 1. the MCLmm prior itself ----
W = world('skep2'); m = W['mu']['MCLmm']; st, em, pg = W['extra']['st'], W['extra']['em'], W['extra']['popg']
P('\n== 1. MCLmm population effect under the skeptic prior skep2 (MODEL on MEASURED curves) ==')
P('  overall %s | sd %.2f | P>0 %.2f' % (q(m), m.std(), (m > 0).mean()))
for i, sn in enumerate(STATES):
    for e_, en in ((False, 'reweight'), (True, 'relabel')):
        for p_, pn in ((False, 'train-like'), (True, 'gold58-like')):
            k = (st == i) & (em == e_) & (pg == p_)
            P('  %-9s %-8s %-11s %s P>0 %.2f' % (sn, en, pn, q(m[k]), (m[k] > 0).mean()))
for pr in ('stack', 'pessimistic', 'skeptic-null', 'recal'):
    mm = world(pr)['mu']['MCLmm']; P('  ref prior %-12s %s | sd %.2f | P>0 %.2f' % (pr, q(mm), mm.std(), (mm > 0).mean()))

# ---- 2. per entry, standalone ----
PRI = ['stack', 'pessimistic', 'skeptic-null', 'recal', 'skep2', 'zero', 'neg']
RES = {}
for gapn, gapm in (('joint', (.75, 1.8)), ('LB-only', (.5, 1.3)), ('brief 1-2e-4', (1.4, 2.5))):
    for lomm in ('stack', 'skeptic'):
        if gapn != 'joint' and lomm == 'skeptic':
            continue
        tag = 'gap %s, lomm05 prior %s' % (gapn, lomm)
        P('\n== 2/3. %s ==' % tag)
        for pr in PRI:
            W = world(pr, gapm, lomm); base = slate(W, ['pw25']); b2 = slate(W, ['pw25', 'lomm05'])
            show = pr in ('stack', 'skep2', 'zero') and gapn == 'joint' and lomm == 'stack'
            if show:
                P('-- prior %s | {pw25}: E priv A %+.2f, P(.960) %.2f | NIGHT {pw25, lomm05}: %+.2f, P(.960) %.2f' % (pr, base['priv'].mean(), base['p960'], b2['priv'].mean(), b2['p960']))
                P('   %-7s | %-24s | %-8s | %-7s | %-24s | %-8s | %-9s | %-6s | %-7s | %-22s | %-22s | %-9s' % (
                    'entry', 'E public [p10,p90]', 'P(pub>0)', 'P(.960)', 'E private [p10,p90]', 'P(pri>0)', 'incr{pw25}', 'P(X=A)', 'P(A<0)',
                    'E MCLmm | X=A (uncond)', 'E priv | X=A shows .960', 'incr NIGHT'))
            for X in ['lomm05', 'S_pl*'] + STACKS:
                s1 = slate(W, ['pw25', X]); s2 = slate(W, ['pw25', 'lomm05', X]) if X != 'lomm05' else b2
                isA = s1['a'] == 2; inc = s1['priv'].mean() - base['priv'].mean(); inc2 = s2['priv'].mean() - b2['priv'].mean()
                RES[(gapn, lomm, pr, X)] = dict(inc=inc, inc2=inc2, p960=s1['p960'], pA=isA.mean(), pAneg=(s1['priv'] < 0).mean())
                if show:
                    mcl = W['mu']['MCLmm']; hasm = 'MCLmm' in ENT[X]
                    c960 = isA & s1['show']
                    P('   %-7s | %-24s | %.2f     | %.2f    | %-24s | %.2f     | %+.2f     | %.2f   | %.2f    | %-22s | %-22s | %+.2f' % (
                        X, q(W['pub'][X]), (W['pub'][X] > 0).mean(), (W['pub'][X] > W['gap']).mean(), q(W['pri'][X]), (W['pri'][X] > 0).mean(), inc,
                        isA.mean(), (s1['priv'] < 0).mean(),
                        ('%+.2f (%+.2f)' % (mcl[isA].mean(), mcl.mean())) if hasm else 'n/a',
                        '%+.2f (P %.2f)' % (s1['priv'][c960].mean(), c960.mean()) if c960.any() else 'n/a', inc2))

# ---- 4. robustness tables ----
P('\n== 4. increment of X added to {pw25} (r6pw20 B) by MCLmm prior: stack / pessimistic / skeptic-null / recal / skep2 / zero / neg N(-.5, 1) (gap joint, lomm05 stack) ==')
for X in ['lomm05', 'S_pl*'] + STACKS:
    v = [RES[('joint', 'stack', pr, X)]['inc'] for pr in PRI]; w = [RES[('joint', 'stack', pr, X)]['inc2'] for pr in PRI]
    P('   %-7s {pw25}: %s | min(first 5) %+.2f || over NIGHT {pw25, lomm05}: %s' % (X, ' / '.join('%+.2f' % x for x in v), min(v[:5]), ' / '.join('%+.2f' % x for x in w)))
P('   same, lomm05 skeptic prior N(0, .9) with I(pw25, lomm05) N(0, .12):')
for X in ['lomm05', 'S_pl*'] + STACKS:
    v = [RES[('joint', 'skeptic', pr, X)]['inc'] for pr in PRI]
    P('   %-7s {pw25}: %s' % (X, ' / '.join('%+.2f' % x for x in v)))
P('   P(.960 shown) of the {pw25, X} slate, joint gap / LB-only gap, prior skep2 vs stack:')
for X in ['lomm05', 'S_pl*'] + STACKS:
    P('   %-7s skep2 %.2f / %.2f / %.2f | stack %.2f / %.2f / %.2f | zero %.2f   (joint / LB-only / brief-gap)' % (X, RES[('joint', 'stack', 'skep2', X)]['p960'], RES[('LB-only', 'stack', 'skep2', X)]['p960'],
      RES[('brief 1-2e-4', 'stack', 'skep2', X)]['p960'], RES[('joint', 'stack', 'stack', X)]['p960'], RES[('LB-only', 'stack', 'stack', X)]['p960'],
      RES[('brief 1-2e-4', 'stack', 'stack', X)]['p960'], RES[('joint', 'stack', 'zero', X)]['p960']))

# ---- 5. 5-slot sets ----
P('\n== 5. A-probe sets (r6pw20 B + pw25 + up to 3 more; * = virtual, not built), E private of A by prior stack / pess / null / recal / skep2 / zero / neg; ranked by skep2 ==')
rows = []
Ws = {pr: world(pr) for pr in PRI}
for k in range(0, 4):
    for c in itertools.combinations(['lomm05', 'S_pl*'] + STACKS, k):
        es = ['pw25'] + list(c); vals = []; extra = None
        for pr in PRI:
            s = slate(Ws[pr], es); vals.append(s['priv'].mean())
            if pr == 'skep2':
                extra = [s['p960'], (s['priv'] < 0).mean()]
            if pr == 'zero':
                extra.append(s['p960'])
        rows.append((vals[4], '{%s}' % ', '.join(es), vals, extra))
rows.sort(reverse=True)
def showrow(v, nm, vals, ex_):
    P('   %-34s %s | worst(all 7) %+.2f | skep2 P(.960) %.2f P(A<0) %.2f | zero P(.960) %.2f' % (nm, ' / '.join('%+.2f' % x for x in vals), min(vals), ex_[0], ex_[1], ex_[2]))
for r_ in rows[:12]:
    showrow(*r_)
P('   ranked by worst case over all 7 priors (top 8):')
for r_ in sorted(rows, key=lambda r: -min(r[2]))[:8]:
    showrow(*r_)
P('   references:')
for r_ in rows:
    if r_[1] in ('{pw25}', '{pw25, lomm05}', '{pw25, S_wide}', '{pw25, lomm05, S_wide}', '{pw25, lomm05, S_alt, S_wide}'):
        showrow(*r_)

# ---- 6. R1 private cost of adopting S_wide by public noise ----
P(chr(10) + '== 6. R1 cost decomposition, slate {pw25, X} (gap joint, lomm05 stack): where the MCLmm stacks win or lose private vs the MCLmm-free twin ==')
for pr in ('stack', 'skep2', 'skeptic-null', 'recal', 'zero', 'neg'):
    W = Ws[pr]; mcl = W['mu']['MCLmm']; base = slate(W, ['pw25'])
    for X, twin in (('S_wide', 'S_pl*'), ('S_alt', None)):
        s1 = slate(W, ['pw25', X]); isA = s1['a'] == 2; bad = isA & (mcl < 0)
        line = '   prior %-12s %-6s: P(X=A) %.2f | P(X=A & MCLmm<0) %.2f, E[MCLmm | that] %+.2f, its contribution %+.2f | P(X=A & MCLmm>0) %.2f, contribution %+.2f | net E[MCLmm x 1(X=A)] %+.2f' % (
            pr, X, isA.mean(), bad.mean(), mcl[bad].mean() if bad.any() else 0., (mcl * bad).mean(), (isA & (mcl > 0)).mean(), (mcl * (isA & (mcl > 0))).mean(), (mcl * isA).mean())
        if twin:
            st_ = slate(W, ['pw25', twin]); line += ' | slate vs twin {pw25, %s}: %+.2f' % (twin, s1['priv'].mean() - st_['priv'].mean())
        P(line)
P('SKEP_VALUE_MC DONE')
OUT.close()
