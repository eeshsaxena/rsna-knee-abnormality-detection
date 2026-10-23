# stack_value_mc.py (10-20 ~23:25 IST, Claude, STACK BUILD round). MODEL only (arithmetic on stated inputs; no data read).
# For each built stack X: expected public delta vs pw30, P(display .960), expected private, and its value as an EXTRA slate entry under R1
# (A = API best-id = max public among pw30 and the A probes; r6pw20 is B and never A, so it does not enter A) next to r6pw20 + pw25:
#   increment(X) = E[private of A | A probes {pw25, X}] - E[private of A | {pw25}]   (e-4 macro vs pw30).
# Logic = night/night_slate_mc.py (argmax of public among {pw30 = 0, entries}, value = the pick's true effect) + stack/stack_mc.py
# (component structure: an entry's public / private delta = sum over its components of (mu_c + read error_c) + I(pw25, lomm05); a
# component's read error is SHARED by every entry that contains it, because it is the same column change on the same 390 / 910 knees).
# Inputs (sources):
#   mu pw25 ~ N(+0.40, 0.80)                                                    (stack/inventory.md sec 2, MODEL; VAL / implied2 private +0.56)
#   mu lomm05, mu MCLmm: shared scenario draw S1 train-like / S2 extra co-firing carries signal / S3 gold-like, weights .40 / .35 / .25;
#     lomm05 means +1.10 / -0.48 / -0.26, sd 1.2 / 1.2 / 1.6, x .85; MCLmm +2.30 / 0 / +1.30, sd 1.0 / 0.8 / 1.2, x .75 (stack/stack_mc.py)
#   MCLmm prior variants (stack/stack_mc_sens.py): pessimistic (own scenario draw .25 / .50 / .25, x .60), skeptic-null N(0, 1), recal N(-.2, 1)
#   I(pw25, lomm05) ~ N(+0.16, 0.12)                                            (stack/inventory.md sec 4)
#   pw25 (PRIMARY) as val/implied2.md abc3 primary: public vs pw30 N(+.09, .80), population N(+.50, 1.45), private noise 1.32, corr .17;
#   read sd public / private (e-4): pw25 1.70 / 1.00, lomm05 1.00 / 0.62, MCLmm 2.40 / 1.50, corr(pw25, lomm05 read errors) -0.20,
#     other pairs 0  (MEASURED stack/kernels/read_noise.log, R7 OOF, 1,000 paired subsets; the inventory used 1.4 / 1.01 / 2.5 x .655 -> SENS row)
#   gap pw30 -> .960 display: lognormal median 0.75, p90 1.8 (joint, CLAIMED val/implied2.md); LB-only median 0.5, p90 1.3 (SENS)
import numpy as np, itertools, sys
OUT = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/kernels/stack_value_mc.log', 'w')
def P(s=''):
    print(s, flush=True); OUT.write(s + '\n')
N = 400000
ENT = {'pw25': ['pw25'], 'lomm05': ['lomm05'], 'S_safe': ['MCLmm'], 'S_alt': ['pw25', 'MCLmm'], 'S_wide': ['pw25', 'lomm05', 'MCLmm'],
       'S_mid': ['lomm05', 'MCLmm']}
STACKS = ['S_safe', 'S_alt', 'S_wide', 'S_mid']
q = lambda x: '%+.2f [%+.2f, %+.2f]' % (x.mean(), np.percentile(x, 10), np.percentile(x, 90))


def world(prior, noise, gapm, pwm='implied2', seed=1021):
    rng = np.random.default_rng(seed)
    sc = rng.choice(3, N, p=[.40, .35, .25])
    def scen(means, sds, hc, s=sc):
        return hc * (np.array(means)[s] + np.array(sds)[s] * rng.standard_normal(N))
    mu = {'pw25': rng.normal(.40, .80, N), 'lomm05': scen([1.10, -.48, -.26], [1.2, 1.2, 1.6], .85)}
    if prior == 'stack':
        mu['MCLmm'] = scen([2.30, 0., 1.30], [1.0, .8, 1.2], .75)
    elif prior == 'pessimistic':
        sc2 = rng.choice(3, N, p=[.25, .50, .25]); mu['MCLmm'] = scen([2.30, 0., 1.30], [1.0, .8, 1.2], .60, sc2)
    elif prior == 'skeptic-null':
        mu['MCLmm'] = rng.normal(0., 1.0, N)
    elif prior == 'recal':
        mu['MCLmm'] = rng.normal(-.2, 1.0, N)
    I = rng.normal(.16, .12, N)
    if noise == 'measured':
        sp, sq, rho = {'pw25': 1.70, 'lomm05': 1.00, 'MCLmm': 2.40}, {'pw25': 1.00, 'lomm05': .62, 'MCLmm': 1.50}, -.20
    else:   # inventory inputs
        sp = {'pw25': 1.4, 'lomm05': 1.01, 'MCLmm': 2.5}; sq = {k: v * np.sqrt(390 / 910.) for k, v in sp.items()}; rho = 0.
    def errs(sd):
        z1, z2, z3 = rng.standard_normal((3, N))
        return {'pw25': sd['pw25'] * z1, 'lomm05': sd['lomm05'] * (rho * z1 + np.sqrt(1 - rho ** 2) * z2), 'MCLmm': sd['MCLmm'] * z3}
    ep, eq = errs(sp), errs(sq)
    med, p90 = gapm
    gap = np.exp(rng.normal(np.log(med), (np.log(p90) - np.log(med)) / 1.2816, N))
    if pwm == 'implied2':
        # pw25 component as val/implied2.md (abc3 primary, conditional on the A21 public reads): public vs pw30 N(+0.09, 0.80), population
        # N(+0.50, 1.45), private noise sd 1.32 (PRIV +0.56 [-1.97, +3.04]); corr(public, population) .17, fitted so that the R1 value of the
        # slate {pw25} = E[pop x 1(pub > 0)] reproduces implied2's +0.37 (P(move) .55 vs implied2 .56).
        z1, z2, z3 = rng.standard_normal((3, N)); rho25 = .17
        pub25 = .09 + .80 * z1; mu['pw25'] = .50 + 1.45 * (rho25 * z1 + np.sqrt(1 - rho25 ** 2) * z2); pri25 = mu['pw25'] + 1.32 * z3
    else:
        pub25 = mu['pw25'] + ep['pw25']; pri25 = mu['pw25'] + eq['pw25']
    def ent(parts, e, x25):
        return sum((x25 if p == 'pw25' else mu[p] + e[p]) for p in parts) + (I if {'pw25', 'lomm05'} <= set(parts) else 0.)
    pub = {k: ent(v, ep, pub25) for k, v in ENT.items()}; pri = {k: ent(v, eq, pri25) for k, v in ENT.items()}
    pop = {k: sum(mu[p] for p in v) + (I if {'pw25', 'lomm05'} <= set(v) else 0.) for k, v in ENT.items()}
    return pub, pri, pop, gap


def slate(W, es):
    pub, pri, pop, gap = W
    PU = np.column_stack([np.zeros(N)] + [pub[e] for e in es]); PR = np.column_stack([np.zeros(N)] + [pri[e] for e in es])
    PO = np.column_stack([np.zeros(N)] + [pop[e] for e in es])
    a = PU.argmax(1); r = np.arange(N)
    return dict(a=a, priv=PR[r, a], pop=PO[r, a], pub=PU[r, a], p960=(PU[r, a] > gap).mean())


PRIORS = ['stack', 'pessimistic', 'skeptic-null', 'recal']
P('MODEL stack_value_mc.py | N %d per world | units e-4 macro vs pw30 57425970 | intervals [p10, p90]' % N)
P('A = max public among {pw30, A probes} (R1); r6pw20 = B, never A. "increment" = E[private of A] with X added to {pw25} minus with {pw25} alone.')
RES = {}
CFG = [('PRIMARY', 'measured', (.75, 1.8), 'implied2', 'PRIMARY: pw25 as val/implied2 (conditional on the A21 reads); lomm05 / MCLmm read noise MEASURED; joint gap'),
       ('SENS_A', 'measured', (.75, 1.8), 'inventory', 'SENS A: pw25 as stack_mc (mu N(+.40, .80), fresh read sd 1.70 / 1.00, MEASURED); rest as primary'),
       ('SENS_B', 'inventory', (.75, 1.8), 'inventory', 'SENS B: stack/stack_mc.py inputs exactly (pw25 read 1.4, lomm05 1.01, MCLmm 2.5, private x .655)'),
       ('SENS_C', 'measured', (.5, 1.3), 'implied2', 'SENS C: as primary with the LB-only gap (median 0.5, p90 1.3)')]
for cfg, noise, gapm, pwm, tagw in CFG:
    P(chr(10) + '==== %s ====' % tagw)
    for prior in PRIORS:
        W = world(prior, noise, gapm, pwm); pub, pri, pop, gap = W
        base = slate(W, ['pw25']); b2 = slate(W, ['pw25', 'lomm05'])
        P('-- MCLmm prior: %s | baseline {pw25}: E priv of A %+.2f, P(move) %.2f, P(.960 shown) %.2f | NIGHT {pw25, lomm05}: %+.2f (increment of lomm05 %+.2f), P(.960) %.2f' % (
            prior, base['priv'].mean(), (base['a'] > 0).mean(), base['p960'], b2['priv'].mean(), b2['priv'].mean() - base['priv'].mean(), b2['p960']))
        show = prior == 'stack' or cfg == 'PRIMARY'
        if show:
            P('   %-7s | %-24s | %-8s | %-8s | %-24s | %-8s | %-11s | %-8s | %-8s | %-8s | %s' % (
                'entry', 'E public vs pw30 [read]', 'P(pub>0)', 'P(.960)', 'E private [p10,p90]', 'P(pri>0)', 'incr {pw25}', 'P(X = A)', 'P(A<0)', 'P(.960)', 'incr {pw25,lomm05}'))
        for X in ['pw25', 'lomm05'] + STACKS:
            s1 = slate(W, ['pw25', X]) if X != 'pw25' else base
            s2 = slate(W, ['pw25', 'lomm05', X]) if X not in ('pw25', 'lomm05') else b2
            RES[(cfg, prior, X)] = dict(inc=s1['priv'].mean() - base['priv'].mean(), inc2=s2['priv'].mean() - b2['priv'].mean(),
                                        pub=pub[X].mean(), pri=pri[X].mean(), p960=(pub[X] > gap).mean(), pa=(s1['a'] == 2).mean() if X != 'pw25' else (base['a'] == 1).mean())
            if show:
                P('   %-7s | %-24s | %.2f     | %.2f     | %-24s | %.2f     | %+.2f       | %.2f     | %.2f     | %.2f     | %+.2f' % (
                    X, q(pub[X]), (pub[X] > 0).mean(), (pub[X] > gap).mean(), q(pri[X]), (pri[X] > 0).mean(),
                    s1['priv'].mean() - base['priv'].mean(), RES[(cfg, prior, X)]['pa'], (s1['priv'] < 0).mean(), s1['p960'],
                    s2['priv'].mean() - b2['priv'].mean()))
        if cfg in ('PRIMARY', 'SENS_A'):
            rows = []
            for k in range(0, 4):     # 5 slots = r6pw20 (B) + pw25 + up to 3 more A probes
                for c in itertools.combinations(['lomm05'] + STACKS, k):
                    nm = '{pw25%s}' % ''.join(', ' + x for x in c); s = slate(W, ['pw25'] + list(c))
                    rows.append((s['priv'].mean(), nm, s['p960'], (s['priv'] < 0).mean())); RES[(cfg, prior, 'slate', nm)] = (s['priv'].mean(), s['p960'], (s['priv'] < 0).mean())
            rows.sort(reverse=True)
            P('   top 5 A-probe sets: ' + ' | '.join('%s %+.2f (P.960 %.2f, P(A<0) %.2f)' % (nm, v, p9, pn) for v, nm, p9, pn in rows[:5]))

for cfg in ('PRIMARY', 'SENS_A', 'SENS_B', 'SENS_C'):
    P(chr(10) + '==== %s robustness: increment of X added to {pw25} (r6pw20 B) | and to {pw25, lomm05}; priors stack / pessimistic / skeptic-null / recal ====' % cfg)
    for X in STACKS + ['lomm05']:
        v = [RES[(cfg, pr, X)]['inc'] for pr in PRIORS]; w = [RES[(cfg, pr, X)]['inc2'] for pr in PRIORS]
        P('   %-7s incr over {pw25}: %s | min %+.2f || over {pw25, lomm05}: %s | min %+.2f' % (X, ' / '.join('%+.2f' % x for x in v), min(v), ' / '.join('%+.2f' % x for x in w), min(w)))
for cfg in ('PRIMARY', 'SENS_A'):
    P(chr(10) + '==== %s 5-slot sets (r6pw20 + pw25 + up to 3 more): E private of A by prior stack / pessimistic / skeptic-null / recal, ranked by worst case ====' % cfg)
    names = sorted({k[3] for k in RES if len(k) == 4 and k[0] == cfg})
    tab = sorted(((min(RES[(cfg, pr, 'slate', nm)][0] for pr in PRIORS), nm) for nm in names), reverse=True)
    for mn, nm in tab[:10]:
        P('   %-38s %s | worst %+.2f | P(.960) stack prior %.2f | P(A<0) stack prior %.2f' % (nm, ' / '.join('%+.2f' % RES[(cfg, pr, 'slate', nm)][0] for pr in PRIORS), mn,
          RES[(cfg, 'stack', 'slate', nm)][1], RES[(cfg, 'stack', 'slate', nm)][2]))
P('STACK_VALUE_MC DONE')
OUT.close()
