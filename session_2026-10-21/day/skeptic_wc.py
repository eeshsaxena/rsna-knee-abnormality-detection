# skeptic_wc.py (10-21 ~09:50 IST, Claude, DAY SKEPTIC track). MODEL only. RESEARCH ONLY: local CPU arithmetic, no OAI rows, no box,
# no Kaggle / LB call, nothing written outside session_2026-10-21/day. Run with `python -B`.
#
# Question: is R1 (A = API best-id over 5 near-identical A22 entries) a winner's-curse trap for private? Is B = r6pw20 right in every
# outcome? Is any 10-22 entry worth its private risk?
#
# World = readtable_mc.py sections 1-2 VERBATIM (same seeds, so the base run reproduces readtable_base.log), plus three skeptic knobs
# that readtable / plan1022 do not have:
#   rhoS=  corr of each stack component's PRIVATE population effect with its PUBLIC one in the SITE scenario (weight .2; readtable = 1).
#          A site split with unseen sites is exactly where a public-selected lomm05 / MCLmm effect need not carry over.
#   rhoA=  same corr in ALL scenarios (a global "public 390 != private 910 population" stress; default 1).
#   nf=    multiplier on the component read-noise sds (public AND private; lomm05 1.00 / .62, MCLmm 2.40 / 1.50 in readtable).
#   pf=    shrink of the component prior around its mean (true effects smaller than the priors say; R1 then selects on noise).
# plus lomm= / mcl= / dose_mc args as readtable_mc.py.
# Output: per best-id class: public margin of W over pw30 vs its private margin (the curse), P(A private < pw25 / pw20), policies
# R1 / R1two / PF / H1 / H2 / H2s / pw25fix / R1+P1 / H2+P1, and B alternatives (rule, R6, HW, pw20, W).
import sys, os, io, json, numpy as np
HERE = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/day'
DW = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/private/dose_work'
SKV = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/skeptic/skep_value_mc.py'
SHK = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/private/work/curve_base.json'
MINE = ('lomm', 'mcl', 'i20', 'rL', 'rhoS', 'rhoA', 'nf', 'pf', 'tag2')
own = dict(a.split('=', 1) for a in sys.argv[1:] if a.split('=', 1)[0] in MINE)
sys.argv = ['dose_mc.py'] + [a for a in sys.argv[1:] if a.split('=', 1)[0] not in MINE]
sys.dont_write_bytecode = True
sys.path.insert(0, DW)
import dose_mc as D
LOMM = own.get('lomm', 'stack'); MCLP = own.get('mcl', 'skep2'); I20 = float(own.get('i20', 2.0)); RL = float(own.get('rL', .9))
RHOS = float(own.get('rhoS', 1.0)); RHOA = float(own.get('rhoA', 1.0)); NF = float(own.get('nf', 1.0)); PF = float(own.get('pf', 1.0))
TAG = D.TAG + ('_' + own['tag2'] if 'tag2' in own else '')
LOG = open(HERE + '/skeptic_wc_%s.log' % TAG, 'w')


def P(s=''):
    print(s, flush=True); LOG.write(s + '\n')


# ---------------- 1. dose world: verbatim readtable_mc.py section 1 ----------------
F = D.fit(); rng = np.random.default_rng(D.SEED + 7)
Z = np.load(D.PWK + '/post_base_report.npz'); ENT = list(Z['ent']); e = {k: i for i, k in enumerate(ENT)}
PV = Z['priv'].astype(float); M0 = PV.shape[0]; N = M0 * D.NREP; ib = np.tile(np.arange(M0), D.NREP)
OUR = {k: PV[ib, e[k]] for k in ('c4', 'R6u', 'HW')}
j = rng.choice(len(F['w']), N, p=F['w'] / F['w'].sum())
sh, s, rc4 = F['sh'][j], F['s'][j], F['rc4'][j]; G = {t: F['G'][t][j] for t in D.SH}; P0 = {t: F['P0'][t][j] for t in D.SH}
x20 = F['x20'][j]
SDP = np.sqrt(390 / 910); nc = F['Ls'].shape[0]
Z1 = (rng.standard_normal((N, nc)) @ F['Ls'].T) * SDP; Z2 = (rng.standard_normal((N, nc)) @ F['Ls'].T) * SDP
NP = {t: Z1[:, 3 * k_:3 * k_ + 3].sum(1) for k_, t in enumerate(D.SH)}; NV2 = {t: Z2[:, 3 * k_:3 * k_ + 3].sum(1) for k_, t in enumerate(D.SH)}
del Z1, Z2
kc = rng.normal(D.K_MU, D.K_SD, N) + rng.normal(0, .05, N); epsx = D.EPS * rng.standard_normal(N)
uR6 = np.clip(rng.normal(.10, .07, N), .01, .4)
pubg = kc * P0['PW20'] + epsx + .8 * (G['PW20'] - P0['PW20']) + .6 * F['sdpub']['PW20'] * rng.standard_normal(N)
disp = np.where(pubg > (1 - uR6) * 1e-3, 959, np.where(pubg < -uR6 * 1e-3, 957, 958))
dS = rng.normal(0, D.SITE_D, N); rcS = rng.normal(0, D.SITE_RC, N); sS = np.exp(rng.normal(0, D.SITE_S, N))
SCEN = {'S0': dict(d=0.0), 'C+': dict(d=.5), 'C-': dict(d=-.5), 'SITE': dict(d=dS, rc=rcS, sf=sS, nsc=D.SITE_N)}
MIX = {'S0': .6, 'C+': .1, 'C-': .1, 'SITE': .2}

# ---------------- 2. stack components: verbatim readtable_mc.py section 2 (same rS stream) ----------------
src = open(SKV, encoding='utf-8').read(); src = src[:src.index("P('MODEL skep_value_mc.py")]
src = src.replace("OUT = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/skeptic/skep_value_mc.log', 'w')", "OUT = io.StringIO()")
assert 'OUT = io.StringIO()' in src
gv = {'io': io}; exec(compile(src, SKV, 'exec'), gv); gv['N'] = N
rS = np.random.default_rng(2110)
sc = rS.choice(3, N, p=[.40, .35, .25])
if LOMM == 'stack':
    muL = .85 * (np.array([1.10, -.48, -.26])[sc] + np.array([1.2, 1.2, 1.6])[sc] * rS.standard_normal(N)); I = rS.normal(.16, .12, N)
elif LOMM == 'skeptic':
    muL = rS.normal(0., .9, N); I = rS.normal(0., .12, N)
else:
    muL = rS.normal(-.9, .9, N); I = rS.normal(0., .12, N)
popg = np.where(sc == 0, False, np.where(sc == 2, True, rS.random(N) < .5))
if MCLP == 'skep2':
    muM = gv['mcl_skep2'](rS, popg)[0]
elif MCLP == 'zero':
    muM = np.zeros(N)
else:
    muM = .75 * (np.array([2.30, 0., 1.30])[sc] + np.array([1.0, .8, 1.2])[sc] * rS.standard_normal(N))
e25 = ((G['PW25'] - P0['PW25']) - (G['PW30'] - P0['PW30'])) * 1e4; z25 = (e25 - e25.mean()) / e25.std()
zL = rS.standard_normal(N); eLp = 1.00 * (-.2 * z25 + np.sqrt(1 - .04) * zL)
eLp_o = 1.00 * (RL * (eLp / 1.00) + np.sqrt(1 - RL ** 2) * rS.standard_normal(N))
eMp = 2.40 * rS.standard_normal(N)
eLq = .62 * rS.standard_normal(N); eLq_o = .62 * (RL * eLq / .62 + np.sqrt(1 - RL ** 2) * rS.standard_normal(N))
eMq = 1.50 * rS.standard_normal(N)

# ---------------- 2b. skeptic knobs (separate RNG; with defaults the world equals readtable's) ----------------
rK = np.random.default_rng(5150)
mL, mM, mI = muL.mean(), muM.mean(), I.mean(); sL, sM, sI = muL.std(), max(muM.std(), 1e-9), I.std()
if PF != 1.0:
    muL = mL + PF * (muL - mL); muM = mM + PF * (muM - mM); I = mI + PF * (I - mI)
if NF != 1.0:
    eLp, eLp_o, eMp, eLq, eLq_o, eMq = [NF * x for x in (eLp, eLp_o, eMp, eLq, eLq_o, eMq)]
zq = rK.standard_normal((3, N))


def privcomp(rho):          # private population effects with corr rho to the public ones (same marginal prior moments)
    if rho >= 1.0:
        return muL, muM, I
    c = np.sqrt(1 - rho ** 2)
    return (mL + rho * (muL - mL) + c * PF * sL * zq[0], mM + rho * (muM - mM) + c * PF * sM * zq[1] * (MCLP != 'zero'),
            mI + rho * (I - mI) + c * PF * sI * zq[2])


# 10-22 L dose step (plan1022_mc.py GUESS curve: mu(r) = r mu05 - kap r (r - 1), noise and I scale with r), r = 1.6 (a .08)
kap = np.maximum(0., np.random.default_rng(909).normal(.36, .2, N)); R8 = 1.6

PUB = {'pw30': G['PW30'] * 1e4, 'pw25': G['PW25'] * 1e4, 'pw20': G['PW20'] * 1e4}
PUB['spl'] = PUB['pw25'] + muL + I + eLp
PUB['swide'] = PUB['spl'] + muM + eMp
PUB['salt'] = PUB['pw25'] + muM + eMp
PUB['p30L'] = PUB['pw30'] + muL + eLp_o
PUB['smid'] = PUB['p30L'] + muM + eMp
PUB['ssafe'] = PUB['pw30'] + muM + eMp
PUB['spl08'] = PUB['pw25'] + R8 * muL - kap * R8 * (R8 - 1) + R8 * I + R8 * eLp
PUB['swide08'] = PUB['spl08'] + muM + eMp
WIN = ['pw30', 'pw25', 'spl', 'swide', 'salt']
W = np.argmax(np.column_stack([PUB[k] for k in WIN]), 1)
W2 = np.argmax(np.column_stack([PUB['pw30'], PUB['pw25']]), 1)
pubW = np.column_stack([PUB[k] for k in WIN])[np.arange(N), W]
pos = lambda x: x20 + (x - G['S20'] * 1e4) * .1
show960 = pos(pubW) >= 1.0
STK = W >= 2
CLS = [('ALL', np.ones(N, bool))] + [(WIN[i], W == i) for i in range(5)] + [('stack', STK), ('stack&960', STK & show960), ('stack&no960', STK & ~show960)]
# 10-22 class entries (plan1022 P1): pw30 class -> p30L, ssafe, smid; spl -> spl08; swide -> swide08; pw25 / salt -> none
P1E = {0: ['p30L', 'ssafe', 'smid'], 1: [], 2: ['spl08'], 3: ['swide08'], 4: []}

# ---------------- 3. curves (readtable_mc.py) ----------------
CV = np.load(D.P2 + '/curves_base.npz'); g19 = CV['grid'] * 1e4
c19 = lambda k, x: np.interp(x, g19, CV[k], left=float(CV[k][0]), right=float(CV[k][-1]))
CB = json.load(open(SHK)); ms = np.array(sorted(float(k) for k in CB)); key = {float(k): k for k in CB}


def cshk(name, x):
    y = np.array([CB[key[m]][name] for m in ms]); slope = (y[-1] - y[-2]) / (ms[-1] - ms[-2])
    out = np.interp(x, ms, y, left=0.0, right=float(y[-1])); hi = x > ms[-1]; out[hi] = np.minimum(1.0, y[-1] + slope * (x[hi] - ms[-1]))
    out[x < ms[0]] = 0.0
    return out


def met(x):
    return np.column_stack([x, c19('silver', x), cshk('P(r<=316)', x), cshk('P(r<=633)', x), cshk('P(r<=150)', x)])


MN = ['E', 'silv_step', 'silv_smooth', 'bronze', 'r150']
acc = {}


def add(k, v, wg):
    for cn, cm in CLS:
        if cm.any():
            acc.setdefault((k, cn), 0.)
            acc[(k, cn)] = acc[(k, cn)] + wg * v[cm].mean(0)


for sn, scd in SCEN.items():
    pop = F['popf'](sh + scd['d'], rc4 + scd.get('rc', 0.0), s * scd.get('sf', 1.0)); nsc = scd.get('nsc', 1.0)
    qL, qM, qI = privcomp(RHOS if sn == 'SITE' else RHOA)
    X = {}
    for t, nm in (('PW20', 'pw20'), ('PW25', 'pw25'), ('PW30', 'pw30')):
        X[nm] = (OUR['c4'] + pop[t] + nsc * NP[t]) * 1e4
    X['R6'] = OUR['R6u'] * 1e4; X['HW'] = OUR['HW'] * 1e4
    X['r6pw20'] = (OUR['R6u'] + kc * pop['PW20'] + epsx + nsc * (D.RHO * NP['PW20'] + np.sqrt(1 - D.RHO ** 2) * NV2['PW20'])) * 1e4
    X['spl'] = X['pw25'] + qL + qI + nsc * eLq
    X['swide'] = X['spl'] + qM + nsc * eMq
    X['salt'] = X['pw25'] + qM + nsc * eMq
    X['p30L'] = X['pw30'] + qL + nsc * eLq_o
    X['smid'] = X['p30L'] + qM + nsc * eMq
    X['ssafe'] = X['pw30'] + qM + nsc * eMq
    X['spl08'] = X['pw25'] + R8 * qL - kap * R8 * (R8 - 1) + R8 * qI + nsc * R8 * eLq
    X['swide08'] = X['spl08'] + qM + nsc * eMq
    r = np.arange(N)
    AW = np.column_stack([X[k] for k in WIN])[r, W]
    A2 = np.column_stack([X['pw30'], X['pw25']])[r, W2]
    Bv = np.where(disp >= 958, X['r6pw20'], X['R6'])
    # R1 + P1: argmax public over A22 + the class entries
    AP1 = AW.copy()
    for wi, extra in P1E.items():
        if not extra:
            continue
        mk = W == wi; names = WIN + extra
        aa = np.column_stack([PUB[n][mk] for n in names]).argmax(1)
        AP1[mk] = np.column_stack([X[n][mk] for n in names])[np.arange(mk.sum()), aa]
    POL = {'R1': AW, 'R1two {pw30,pw25}': A2, 'PF pw20': X['pw20'], 'H1': np.where(W == 0, X['pw20'], AW),
           'H2': np.where(W <= 1, X['pw20'], AW), 'H2s (stack only if .960)': np.where((W <= 1) | ~show960, X['pw20'], AW),
           'pw25 fixed': X['pw25'], 'pw30 fixed': X['pw30'], 'R1+P1': AP1, 'H2+P1': np.where(W <= 1, X['pw20'], AP1)}
    wg = MIX[sn]
    for pn, A in POL.items():
        add('A|' + pn, (A - X['pw30'])[:, None], wg)
        add('P|' + pn, met(np.maximum(A, Bv)), wg)
        add('lt25|' + pn, (A < X['pw25']).astype(float)[:, None], wg)
        add('lt20|' + pn, (A < X['pw20']).astype(float)[:, None], wg)
    # B alternatives
    for bn, bv in (('rule', Bv), ('R6', X['R6']), ('HW', X['HW']), ('r6pw20 always', X['r6pw20']), ('pw20', X['pw20']), ('pw25', X['pw25'])):
        add('B|R1|' + bn, met(np.maximum(AW, bv)), wg)
        add('B|H2|' + bn, met(np.maximum(np.where(W <= 1, X['pw20'], AW), bv)), wg)
    for bn, bv in (('rule', Bv), ('R6', X['R6']), ('HW', X['HW']), ('W (A22 best-id)', AW), ('pw25', X['pw25'])):
        add('B|PF|' + bn, met(np.maximum(X['pw20'], bv)), wg)
    add('curse', np.column_stack([pubW - PUB['pw30'], AW - X['pw30'], pubW - PUB['pw25'], AW - X['pw25']]), wg)

# ---------------- 4. report ----------------
P('MODEL skeptic_wc.py | tag %s | dose_mc prox %s sh %s s %s | lomm05 %s, MCLmm %s | knobs rhoS %.2f rhoA %.2f nf %.2f pf %.2f | N %d' % (
    TAG, D.PROX, D.PSH, D.PS, LOMM, MCLP, RHOS, RHOA, NF, PF, N))
P('Sanity (base knobs must equal readtable_base.log): P(best-id) %s | P(.960) %.3f | R1 pair %s | PF pair %s' % (
    ' / '.join('%.3f' % (W == i).mean() for i in range(5)), show960.mean(),
    ' '.join('%.3f' % v for v in acc[('P|R1', 'ALL')][[0, 1, 2]]), ' '.join('%.3f' % v for v in acc[('P|PF pw20', 'ALL')][[0, 1, 2]])))
P('\n== CURSE by best-id class: E[pub W - pub pw30] vs E[priv W - priv pw30] (e-4), retention = priv / pub; same vs pw25 ==')
for cn, cm in CLS:
    c = acc[('curse', cn)]
    P('  %-12s P %.3f | vs pw30: pub %+5.2f priv %+5.2f ret %5.2f | vs pw25: pub %+5.2f priv %+5.2f | P(A<pw25) %.2f P(A<pw20) %.2f' % (
        cn, cm.mean(), c[0], c[1], c[1] / c[0] if abs(c[0]) > 1e-6 else float('nan'), c[2], c[3], acc[('lt25|R1', cn)][0], acc[('lt20|R1', cn)][0]))
P('\n== POLICIES (A rule; B = A22 rule). E(A) - pw30 | pair E | silver step / smooth | bronze | r150 || P(A<pw25) P(A<pw20) ==')
for cn, cm in CLS:
    P('  class %s (P %.3f)' % (cn, cm.mean()))
    for pn in ('R1', 'R1two {pw30,pw25}', 'PF pw20', 'H1', 'H2', 'H2s (stack only if .960)', 'pw25 fixed', 'pw30 fixed', 'R1+P1', 'H2+P1'):
        a = acc[('A|' + pn, cn)][0]; p = acc[('P|' + pn, cn)]; b = acc[('P|R1', cn)]
        P('     %-26s %+6.2f | %+6.2f | %.3f / %.3f | %.3f | %.3f || %.2f %.2f || vs R1 silver %+5.1f / %+5.1f pts, E(A) %+5.2f' % (
            pn, a, p[0], p[1], p[2], p[3], p[4], acc[('lt25|' + pn, cn)][0], acc[('lt20|' + pn, cn)][0],
            100 * (p[1] - b[1]), 100 * (p[2] - b[2]), a - acc[('A|R1', cn)][0]))
P('\n== B ALTERNATIVES (pair E | silver step / smooth | bronze | r150; d vs B rule in pts) ==')
for cn, cm in CLS:
    P('  class %s (P %.3f)' % (cn, cm.mean()))
    for A in ('R1', 'H2', 'PF'):
        base = acc[('B|%s|rule' % A, cn)]
        for bn in [k[0].split('|', 2)[2] for k in acc if k[0].startswith('B|%s|' % A) and k[1] == cn]:
            p = acc[('B|%s|%s' % (A, bn), cn)]
            P('     A=%-3s B=%-16s %+6.2f | %.3f / %.3f | %.3f | %.3f || d silver %+5.1f / %+5.1f, bronze %+5.1f, r150 %+5.1f' % (
                A, bn, p[0], p[1], p[2], p[3], p[4], 100 * (p[1] - base[1]), 100 * (p[2] - base[2]), 100 * (p[3] - base[3]), 100 * (p[4] - base[4])))
json.dump({'%s||%s' % k: [float(x) for x in np.atleast_1d(v)] for k, v in acc.items()}, open(HERE + '/skeptic_wc_%s.json' % TAG, 'w'))
P('SKEPTIC_WC DONE')
LOG.close()
