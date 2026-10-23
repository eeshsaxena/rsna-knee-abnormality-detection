# plan1022_mc.py (10-21 ~09:00 IST, Claude, plan1022 track). MODEL only. RESEARCH ONLY: local CPU, no OAI rows, no box, no LB writes,
# nothing pushed / submitted. Question: given each A22 outcome, which 10-22 entries (if any) raise the final PRIVATE expectation of the
# pair {A, B} and the public rank?
# Built from the existing MC scripts, unchanged in substance:
#   * E-OA + our side + R6 transfer + field curves + coupling scenarios: private/dose_work/dose_mc.py fit() and the draw construction of
#     private/skeptic_work/skep_dose.py (same seeds), incl. the step curve (val/picks2_work/curves_base.npz) and the smooth curve
#     (private/work/curve_base.json).
#   * Stack components lomm05 (L = LOA <- MM a .05) and MCLmm (M = MCL <- MM a .1), their interaction I(pw25, L), read noises and priors:
#     stack/skeptic/skep_value_mc.py world() (exec'd verbatim, N = 400000 = dose N, seed 2110), paired with the dose draws at random
#     (independence, as both source scripts assume).
# Entries (public raw AUC vs c4 = G, private margin X in e-4):
#   A22 set: pw30 (scored), pw25, spl = pw25 + L + I, swide = spl + M, salt = pw25 + M. pw20 / eoa* are below pw30 by construction (O3).
#   10-22 candidates: p30L = pw30 + L (night/c12 build, gated), ssafe = pw30 + M (stack S_safe build), smid = pw30 + L + M (S_mid build),
#     spl08 / spl03 = pw25 + L at a .08 / .03 (UNBUILT; GUESS dose curve: mu(r) = r mu05 - kap r (r - 1), r = a / .05,
#     kap ~ N(.36, .2)+ from NIGHT +0.41 at .05 vs recal +0.1 at .1; noise and I scale with r), R6pw25 (UNBUILT B-side, R6 + pw .25).
# A = R1 argmax of public over the entered set. B = r6pw20 rule of Amendment 22 (R6pw20 if display >= .958 else R6); B option R6pw25.
# .960 shown: position of the best in the .959 band = x20 + (G_best - G_eoa20) * 1e3 >= 1 (abc3 frozen-ruler convention).
import sys, os, io, json, itertools, numpy as np
args = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
LOMM = args.pop('lomm', 'stack'); MPRI = args.pop('mpri', 'skep2'); TAG = args.pop('tag', 'base')
KAPM = float(args.pop('kapm', .36)); KAPS = float(args.pop('kaps', .2))
R = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20'
OUTD = os.path.dirname(os.path.abspath(__file__))
sys.argv = ['dose_mc.py'] + ['%s=%s' % kv for kv in args.items()]
sys.path.insert(0, R + '/private/dose_work')
import dose_mc as D
LOG = open(os.path.join(OUTD, 'plan1022_mc_%s.log' % TAG), 'w')
def P(s=''):
    print(s, flush=True); LOG.write(s + '\n')

# ---------------- E-OA + our side (skep_dose.py construction, same seeds) ----------------
F = D.fit(); rng = np.random.default_rng(D.SEED + 7)
Z = np.load(D.PWK + '/post_base_report.npz'); ENTN = list(Z['ent']); e = {k: i for i, k in enumerate(ENTN)}
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
zpub = rng.standard_normal(N)
pubg = kc * P0['PW20'] + epsx + .8 * (G['PW20'] - P0['PW20']) + .6 * F['sdpub']['PW20'] * zpub
disp20 = np.where(pubg > (1 - uR6) * 1e-3, 959, np.where(pubg < -uR6 * 1e-3, 957, 958))
dS = rng.normal(0, D.SITE_D, N); rcS = rng.normal(0, D.SITE_RC, N); sS = np.exp(rng.normal(0, D.SITE_S, N))
# R6pw25 (new, separate RNG so the draws above equal skep_dose's): picks2 convention k_X = common + N(0, .05), eps corr .8
r2 = np.random.default_rng(4242)
k25 = kc + r2.normal(0, .05, N); eps25 = D.EPS * (.8 * epsx / D.EPS + .6 * r2.standard_normal(N))
pubg25 = k25 * P0['PW25'] + eps25 + .8 * (G['PW25'] - P0['PW25']) + .6 * F['sdpub']['PW25'] * (.9 * zpub + np.sqrt(1 - .81) * r2.standard_normal(N))
disp25 = np.where(pubg25 > (1 - uR6) * 1e-3, 959, np.where(pubg25 < -uR6 * 1e-3, 957, 958))

# ---------------- stack components (skep_value_mc.py world(), verbatim) ----------------
SRC = R + '/stack/skeptic/skep_value_mc.py'
src = open(SRC, encoding='utf-8').read(); src = src[:src.index("P('MODEL skep_value_mc.py")]
src = src.replace("OUT = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/skeptic/skep_value_mc.log', 'w')", "OUT = io.StringIO()")
assert 'OUT = io.StringIO()' in src and 'N = 400000' in src and N == 400000
g = {'io': io}; exec(compile(src, SRC, 'exec'), g)
W = g['world'](MPRI, (.75, 1.8), LOMM)
muL = W['mu']['lomm05']; Lpub = W['pub']['lomm05']; Lpri = W['pri']['lomm05']       # entry 'lomm05' = pw30 + L, no I
Mpub = W['pub']['S_safe']; Mpri = W['pri']['S_safe']; I = W['I']                     # entry 'S_safe' = pw30 + M
eLp, eLq = Lpub - muL, Lpri - muL
r3 = np.random.default_rng(909); kap = np.maximum(0., r3.normal(KAPM, KAPS, N))
def Ldose(r):       # GUESS dose curve for L at a = .05 r (pub, pri, I part), e-4
    m = r * muL - kap * r * (r - 1); return m + r * eLp, m + r * eLq, r * I
L08 = Ldose(1.6); L03 = Ldose(.6)

# ---------------- public reads (raw, vs c4) ----------------
E4 = 1e-4
PUB = {'pw30': G['PW30'], 'pw25': G['PW25'], 'spl': G['PW25'] + E4 * (Lpub + I), 'swide': G['PW25'] + E4 * (Lpub + I + Mpub),
       'salt': G['PW25'] + E4 * Mpub, 'p30L': G['PW30'] + E4 * Lpub, 'ssafe': G['PW30'] + E4 * Mpub, 'smid': G['PW30'] + E4 * (Lpub + Mpub),
       'spl08': G['PW25'] + E4 * (L08[0] + L08[2]), 'spl03': G['PW25'] + E4 * (L03[0] + L03[2]),
       'swide08': G['PW25'] + E4 * (L08[0] + L08[2] + Mpub)}
A22 = ['pw30', 'pw25', 'spl', 'swide', 'salt']
def r1(names):
    M_ = np.stack([PUB[n] for n in names], 1); return M_.argmax(1)
a22 = r1(A22); best22 = np.choose(a22, [PUB[n] for n in A22])
marg = (best22 - PUB['pw30']) * 1e4                       # observable via the frozen ruler (winner vs old pw30 position)
pos22 = x20 + (best22 - G['S20']) * 1e3
show22 = pos22 >= 1
BINS = [('m0', marg <= 0), ('m<.5', (marg > 0) & (marg <= .5)), ('m.5-1.5', (marg > .5) & (marg <= 1.5)), ('m>1.5', marg > 1.5)]
CLASSES = [('ALL', np.ones(N, bool))] + [(A22[i], a22 == i) for i in range(5)]
for i in range(1, 5):
    for bn, bm in BINS[1:]:
        CLASSES.append(('%s|%s' % (A22[i], bn), (a22 == i) & bm))
CLASSES += [('show960', show22), ('no960', ~show22), ('r6pw20<958', disp20 < 958)]

# ---------------- private, per scenario ----------------
SCEN = {'S0': dict(d=0.0), 'C+': dict(d=.5), 'C-': dict(d=-.5), 'SITE': dict(d=dS, rc=rcS, sf=sS, nsc=D.SITE_N)}
MIX = {'S0': .6, 'C+': .1, 'C-': .1, 'SITE': .2}
CV = np.load(D.P2 + '/curves_%s.npz' % D.FS); g19 = CV['grid'] * 1e4
def c19(k, x): return np.interp(x - D.FSHIFT * 1e4, g19, CV[k], left=float(CV[k][0]), right=float(CV[k][-1]))
CB = json.load(open(R + '/private/work/curve_base.json')); ms = np.array(sorted(float(k) for k in CB)); key = {float(k): k for k in CB}
def cshk(name, x):
    y = np.array([CB[key[m]][name] for m in ms]); slope = (y[-1] - y[-2]) / (ms[-1] - ms[-2])
    out = np.interp(x, ms, y, left=0.0, right=float(y[-1])); hi = x > ms[-1]; out[hi] = np.minimum(1.0, y[-1] + slope * (x[hi] - ms[-1]))
    out[x < ms[0]] = 0.0; return out

OPTS = {'LOCK': [], '+p30L': ['p30L'], '+ssafe': ['ssafe'], '+smid': ['smid'], '+p30L+smid': ['p30L', 'smid'],
        '+p30L+ssafe+smid': ['p30L', 'ssafe', 'smid'], '+spl08': ['spl08'], '+spl03': ['spl03'], '+spl08+p30L': ['spl08', 'p30L'],
        '+spl08+p30L+smid': ['spl08', 'p30L', 'smid'], '+swide08': ['swide08'], '+spl08+swide08': ['spl08', 'swide08']}
BOPT = ['B=r6pw20', 'B=R6pw25']
acc = {}
for sn, sc in SCEN.items():
    pop = F['popf'](sh + sc['d'], rc4 + sc.get('rc', 0.0), s * sc.get('sf', 1.0)); nsc = sc.get('nsc', 1.0)
    X = {}
    for t in ('PW20', 'PW25', 'PW30'): X[D.NAME[t]] = (OUR['c4'] + pop[t] + nsc * NP[t]) * 1e4
    raw20 = (OUR['R6u'] + kc * pop['PW20'] + epsx + nsc * (D.RHO * NP['PW20'] + np.sqrt(1 - D.RHO ** 2) * NV2['PW20'])) * 1e4
    raw25 = (OUR['R6u'] + k25 * pop['PW25'] + eps25 + nsc * (D.RHO * NP['PW25'] + np.sqrt(1 - D.RHO ** 2) * NV2['PW25'])) * 1e4
    R6 = OUR['R6u'] * 1e4
    B = {'B=r6pw20': np.where(disp20 >= 958, raw20, R6)}
    B['B=R6pw25'] = np.where(disp25 >= 958, raw25, B['B=r6pw20'])
    PRI = {'pw30': X['pw30'], 'pw25': X['pw25'], 'spl': X['pw25'] + Lpri + I, 'swide': X['pw25'] + Lpri + I + Mpri, 'salt': X['pw25'] + Mpri,
           'p30L': X['pw30'] + Lpri, 'ssafe': X['pw30'] + Mpri, 'smid': X['pw30'] + Lpri + Mpri,
           'spl08': X['pw25'] + L08[1] + L08[2], 'spl03': X['pw25'] + L03[1] + L03[2], 'swide08': X['pw25'] + L08[1] + L08[2] + Mpri}
    for on, extra in OPTS.items():
        names = A22 + extra; a = r1(names); Apri = np.choose(a, [PRI[n] for n in names]); Apub = np.choose(a, [PUB[n] for n in names])
        show = (x20 + (Apub - G['S20']) * 1e3) >= 1; dpub = (Apub - PUB['pw30']) * 1e4
        for bo in BOPT:
            pair = np.maximum(Apri, B[bo])
            for cn, cm in CLASSES:
                w = cm.mean()
                if w == 0: continue
                q = dict(P=w, EA=(Apri - X['pw30'])[cm].mean(), PAlt=((Apri - X['pw30'])[cm] < 0).mean(), Epair=pair[cm].mean(),
                         s19=c19('silver', pair[cm]).mean(), er19=c19('erank', pair[cm]).mean(), sS=cshk('P(r<=316)', pair[cm]).mean(),
                         brS=cshk('P(r<=633)', pair[cm]).mean(), r150S=cshk('P(r<=150)', pair[cm]).mean(), r100S=cshk('P(r<=100)', pair[cm]).mean(),
                         p960=show[cm].mean(), dpub=dpub[cm].mean(), newA=(a[cm] >= 5).mean())
                acc.setdefault((on, bo, cn), {})[sn] = q
    # POLICIES: entries chosen by the A22 winner (observable). elig = A-eligible 10-22 entries; pub = public-only 10-22 entries (never A).
    POL = {'P1 refine(R1)':      {1: ([], []), 2: (['spl08'], []), 3: (['swide08'], []), 4: ([], []), 0: (['p30L', 'ssafe', 'smid'], [])},
           'P1b refine, no W30 push': {1: ([], []), 2: (['spl08'], []), 3: (['swide08'], []), 4: ([], []), 0: ([], [])},
           'P2 public-max(R1)':  {1: ([], []), 2: (['spl08', 'p30L', 'smid'], []), 3: (['swide08', 'smid', 'p30L'], []), 4: (['ssafe', 'smid'], []), 0: (['p30L', 'ssafe', 'smid'], [])},
           'P3 hybrid':          {1: ([], []), 2: (['spl08'], ['p30L', 'smid']), 3: (['swide08'], ['smid', 'p30L']), 4: ([], ['ssafe', 'smid']), 0: ([], ['p30L', 'ssafe', 'smid'])},
           'P4 A-freeze public': {1: ([], []), 2: ([], ['spl08', 'p30L', 'smid']), 3: ([], ['swide08', 'smid', 'p30L']), 4: ([], ['ssafe', 'smid']), 0: ([], ['p30L', 'ssafe', 'smid'])}}
    for pn, rule in POL.items():
        Apri = np.zeros(N); Apub = np.zeros(N); nA = np.zeros(N, bool)
        for wi, (el, po) in rule.items():
            mk = a22 == wi
            if not mk.any(): continue
            na = A22 + el; aa = r1(na)[mk]
            Apri[mk] = np.choose(aa, [PRI[n][mk] for n in na]); nA[mk] = aa >= 5
            Apub[mk] = np.max(np.stack([PUB[n][mk] for n in A22 + el + po], 1), 1)
        show = (x20 + (Apub - G['S20']) * 1e3) >= 1; dpub = (Apub - PUB['pw30']) * 1e4; pair = np.maximum(Apri, B['B=r6pw20'])
        for cn, cm in [c for c in CLASSES if c[0] in ('ALL', 'pw30', 'pw25', 'spl', 'swide', 'salt', 'no960', 'show960')]:
            acc.setdefault((pn, 'B=r6pw20', cn), {})[sn] = dict(P=cm.mean(), EA=(Apri - X['pw30'])[cm].mean(), PAlt=((Apri - X['pw30'])[cm] < 0).mean(),
                Epair=pair[cm].mean(), s19=c19('silver', pair[cm]).mean(), er19=c19('erank', pair[cm]).mean(), sS=cshk('P(r<=316)', pair[cm]).mean(),
                brS=cshk('P(r<=633)', pair[cm]).mean(), r150S=cshk('P(r<=150)', pair[cm]).mean(), r100S=cshk('P(r<=100)', pair[cm]).mean(),
                p960=show[cm].mean(), dpub=dpub[cm].mean(), newA=nA[cm].mean())
    # ERROR value (class ALL only): A22 minus one entry (B = r6pw20), and B = R6 (r6pw20 ERROR, full A22)
    for dn, names, bo, bv in [('-' + x, [n for n in A22 if n != x], 'B=r6pw20', B['B=r6pw20']) for x in A22[1:]] + [('LOCK', A22, 'B=R6', R6)]:
        a = r1(names); Apri = np.choose(a, [PRI[n] for n in names]); Apub = np.choose(a, [PUB[n] for n in names])
        show = (x20 + (Apub - G['S20']) * 1e3) >= 1; pair = np.maximum(Apri, bv)
        acc.setdefault((dn, bo, 'ALL'), {})[sn] = dict(P=1.0, EA=(Apri - X['pw30']).mean(), PAlt=((Apri - X['pw30']) < 0).mean(), Epair=pair.mean(),
            s19=c19('silver', pair).mean(), er19=c19('erank', pair).mean(), sS=cshk('P(r<=316)', pair).mean(), brS=cshk('P(r<=633)', pair).mean(),
            r150S=cshk('P(r<=150)', pair).mean(), r100S=cshk('P(r<=100)', pair).mean(), p960=show.mean(), dpub=((Apub - PUB['pw30']) * 1e4).mean(), newA=0.0)

def mixed(on, bo, cn):
    d = acc[(on, bo, cn)]; return {k: sum(MIX[sn] * d[sn][k] for sn in MIX) for k in d['S0']}

P('MODEL plan1022_mc.py tag %s | lomm prior %s | MCLmm prior %s | kap N(%.2f, %.2f)+ | dose args %s | N %d | scenario MIX %s' % (TAG, LOMM, MPRI, KAPM, KAPS, args, N, MIX))
P('A22 outcome shares (public argmax over pw30 pw25 spl swide salt): ' + ' | '.join('%s %.3f' % (A22[i], (a22 == i).mean()) for i in range(5)) +
  ' | P(.960 shown after A22) %.3f | r6pw20 display 957/958/959 %.3f/%.3f/%.3f | R6pw25 display %.3f/%.3f/%.3f' % (
      show22.mean(), *[(disp20 == c).mean() for c in (957, 958, 959)], *[(disp25 == c).mean() for c in (957, 958, 959)]))
P('Units: EA = E[private A - private pw30] e-4; pair = max(A, B) margin e-4; s19 / er19 = step curve silver / E[rank]; sS / brS / r150S = smooth '
  'silver / bronze / P(r<=150); p960 = P(final best shows .960); dpub = E[public best - pw30] e-4; newA = P(a 10-22 entry becomes A).')
HDR = '   %-22s %-9s | EA %-6s PA<0 | Epair  s19   er19 | sS    brS   r150S | p960  dpub  newA'
for cn, cm in CLASSES:
    P('\n== class %s (P %.3f) ==' % (cn, cm.mean()))
    base = {bo: mixed('LOCK', bo, cn) for bo in BOPT}
    for on in OPTS:
        for bo in BOPT:
            if (on, bo, cn) not in acc: continue
            m = mixed(on, bo, cn); b0 = base['B=r6pw20']
            P('   %-22s %-9s | %+6.2f %.2f | %+6.2f %.3f %5.0f | %.3f %.3f %.3f | %.3f %+5.2f %.2f || vs LOCK/B=r6pw20: EA %+5.2f s19 %+5.1f sS %+5.1f brS %+5.1f r150S %+5.1f er19 %+5.1f p960 %+5.1f pts' % (
                on, bo, m['EA'], m['PAlt'], m['Epair'], m['s19'], m['er19'], m['sS'], m['brS'], m['r150S'], m['p960'], m['dpub'], m['newA'],
                m['EA'] - b0['EA'], 100 * (m['s19'] - b0['s19']), 100 * (m['sS'] - b0['sS']), 100 * (m['brS'] - b0['brS']), 100 * (m['r150S'] - b0['r150S']),
                m['er19'] - b0['er19'], 100 * (m['p960'] - b0['p960'])))
# scenario detail for the main lines (ALL)
P('\n== scenario detail, class ALL ==')
for on in ('LOCK', '+p30L', '+smid', '+spl08', '+p30L+smid'):
    for bo in BOPT:
        d = acc[(on, bo, 'ALL')]
        P('   %-14s %-9s ' % (on, bo) + ' | '.join('%s EA %+5.2f sS %.3f s19 %.3f' % (sn, d[sn]['EA'], d[sn]['sS'], d[sn]['s19']) for sn in SCEN))
json.dump({'%s|%s|%s' % k: v for k, v in acc.items()}, open(os.path.join(OUTD, 'plan1022_mc_%s.json' % TAG), 'w'))
P('PLAN1022_MC DONE')
LOG.close()
