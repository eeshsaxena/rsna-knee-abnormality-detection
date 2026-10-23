# surface_mc.py (10-21 ~19:30 IST, Claude, LASTDAY surface track). MODEL. RESEARCH ONLY: local CPU arithmetic on public-LB counts
# (reads_lb.json / lb_reads.log, MEASURED) and priors copied from earlier notes (CLAIMED, named below). No OAI data, no box, no Kaggle call.
#
# Public response surface of the E-OA family + stack components, fitted to OUR censored public reads, then used to predict untried
# neighbours of swide 57464326 (A, .960). Units: e-4 macro AUC. y = (unrounded public - .9580) x 1e4, display TRUNCATES
# (.958 = [0, 10), .959 = [10, 20), .960 = [20, 30)); if Kaggle rounds instead, every y shifts by -5 and no difference changes.
#
# Model (all public values are population + public-sample noise on the same 390 studies):
#   band density inside a display band: f(x) ~ exp(-k x), x in [0, 1) of the band, k ~ U(.2, .8) (MEASURED band count ratios .959/.960/.961
#     .40-.60 per band; variant k -> 0 = uniform).
#   eoa10, eoa20: joint multinomial ruler on the SAME 65 frozen .959 teams at 05:52 / 05:53 UTC 10-20 (29 below eoa10, 32 between, 4 above
#     eoa20; MEASURED lb_reads.log), sampled exactly as Dirichlet(30, 33, 5) on the G scale, importance-corrected to a flat prior on y.
#   pw30: O2 frozen cross-board ruler (the 4 frozen teams above eoa20: 2 below pw30, 2 above at 13:20 UTC; MEASURED) sampled exactly as
#     Gq ~ Beta(3, 3) on the eoa20 -> .960 gap, times the 03:11 UTC 10-21 ruler of the 34 .959 teams NOT frozen since 05:53 (2 above /
#     32 below pw30; MEASURED).
#   pw curve near a .2-.3 (A = a / .1): public pw(A) = y30 + (v + w)(A - 3) - c (A - 3)^2 + rho_A, A in {2, 2.5}:
#     v = population slope at A 3 ~ N(-2.0, 1.5), c = population curvature ~ logN(ln 2.1, .4) (both from the implied2 population means
#     pw20 -0.06 / pw25 +0.50 / pw35 -1.83 vs pw30, CLAIMED val/implied2.md sec 3; sd widened to cover the skeptic variants sec 8.1);
#     w = public-noise slope ~ N(0, 2.8) (paired public sd pw25 - pw30 1.4 per .05, CLAIMED implied2 sec 8.1; linear in a as the
#     m12_lbnoise.log sd 2.1 -> 4.2e-4 from a .1 to .2, CLAIMED); rho_A ~ N(0, .3) non-linearity roughness (GUESS).
#   eoa30: eoa20 + (eoa20 - eoa10) - 2 cs + rho, cs ~ logN(ln 2.0, .5) (GUESS; only enters through its .959 display and eoa30 < pw30).
#   Orders / displays (MEASURED): pw20, eoa30 < pw30; pw20, pw25, eoa30 in [10, 20); swide in [20, 30) with 4 below / 46 above in .960
#     (12:03 UTC 10-21). r6pw20 / spl / salt are S6F canaries = unobserved.
#   Stack (stack/skeptic/skep_value_mc.py world(), CLAIMED, priors copied verbatim): swide = pw25 + L + I + M, with
#     L = muL + eLp (lomm05 on pw30; muL stack 3-scenario prior x .85; eLp sd 1.00, corr -.2 with pw25's public noise),
#     I = I(pw25, lomm05) ~ N(.16, .12) (population, no read noise), M = muM + eMp (MCLmm .1; muM skep2 prior; eMp sd 2.40).
#   Dose curves for re-doses (r = dose / built dose): population mu(r) = r mu1 - kap r (r - 1); public noise r e1 + eta,
#     eta ~ N(0, (tau sd1 |r - 1|)^2), tau .3; I scales with r (plan1022 convention); lomm I(a) = I x (.3 - a) / .05 across E-OA doses.
#     kapL ~ N(.30, .12)+ (MEASURED here from night/adj_c12.log: raked and raw LOA <- MM curves at .05 / .10 give curvature .18-.41
#     macro e-4 in all 5 nests and on raw labels); kapM = .80 x U(.45, .85) (stack/stack_mcl.log RAK N_A curve .05 / .10 / .15 / .20
#     -> curvature .78 RAK units, raw-report curve .93, times the skep2 RAK -> public factor F).
#   MOACF on swide (Medial OA column, disjoint): Q = muQ + eQp, eQp sd 1.8 public (MEASURED levers/moacf_noise.log), muQ skeptic mixture
#     (levers/lever_value_mc.py mo_prior 'skep', CLAIMED).
# Private: E of the population delta vs swide (private noise is independent of public, so it has mean 0 given the public reads), and the
#   R1 policy increment E[popdelta x 1(public X > public swide)] (entering X, A = the better public of swide and X).
import sys, io, json, numpy as np
HERE = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/lastday/work'
SKV = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/skeptic/skep_value_mc.py'
args = dict(a.split('=', 1) for a in sys.argv[1:])
g = lambda k, d: type(d)(args.get(k, d))
TAG = g('tag', 'base'); N = g('N', 2000000); SEED = g('seed', 1021)
KLO, KHI = g('klo', .2), g('khi', .8)
VMU, VSD, CMED, CSDL = g('vmu', -2.0), g('vsd', 1.5), g('cmed', 2.1), g('csdl', .4)
WSD, RSD = g('wsd', 2.8), g('rsd', .3)
LOMM = g('lomm', 'stack'); MCLP = g('mcl', 'skep2'); MOA = g('moa', 'skep')
KAPL_MU, KAPL_SD = g('kapl', .30), g('kapl_sd', .12); KAPM_MULT = g('kapm', .80); TAU = g('tau', .3)
LSD, MSD, QSD = g('lsd', 1.00), g('msd', 2.40), g('qsd', 1.8)
SWRULER = g('swr', 'on'); A22 = g('a22', 'on'); RHO0 = g('rho0', 'off')
LOG = open(HERE + '/surface_%s.log' % TAG, 'w')


def P(s=''):
    print(s, flush=True); LOG.write(s + '\n')


rng = np.random.default_rng(SEED)
# ---- band density helpers (x in [0,1) of a band; returns CDF, inverse, density) ----
k = rng.uniform(KLO, KHI, N) if KHI > 0 else np.zeros(N)
kk = np.maximum(k, 1e-6)
Gf = lambda x: (1 - np.exp(-kk * x)) / (1 - np.exp(-kk))
Gi = lambda u: -np.log(1 - u * (1 - np.exp(-kk))) / kk
fd = lambda x: kk * np.exp(-kk * x) / (1 - np.exp(-kk))
logw = np.zeros(N)

# ---- eoa10 / eoa20: Dirichlet(30, 33, 5) on G (exact multinomial posterior under a flat prior on G) ----
D = rng.dirichlet([30, 33, 5], N)
G10 = D[:, 0]; G20 = D[:, 0] + D[:, 1]
x10, x20 = Gi(G10), Gi(G20)
logw -= np.log(fd(x10)) + np.log(fd(x20))          # flat prior on y instead of on G
y10, y20 = 10 + 10 * x10, 10 + 10 * x20
# ---- pw30: O2 Beta(3,3) on the eoa20 -> .960 gap (exact), + the 03:11 non-frozen ruler (2 above / 32 below) ----
Gq = rng.beta(3, 3, N); G30 = G20 + Gq * (1 - G20); x30 = Gi(np.clip(G30, 0, 1 - 1e-12))
logw += np.log(1 - G20) - np.log(fd(x30))          # Jacobian to a flat prior on y30
logw += 32 * np.log(np.clip(G30, 1e-300, 1)) + 2 * np.log(np.clip(1 - G30, 1e-300, 1))
y30 = 10 + 10 * x30
gap = 20 - y30                                     # pw30 -> .960, e-4
# ---- pw curve ----
v = rng.normal(VMU, VSD, N); c = np.exp(np.log(CMED) + CSDL * rng.standard_normal(N)); w = rng.normal(0, WSD, N)
r20, r25 = rng.normal(0, RSD, (2, N))
y_pw20 = y30 - (v + w) - c + r20
y_pw25 = y30 - .5 * (v + w) - .25 * c + r25
cs = np.exp(np.log(2.0) + .5 * rng.standard_normal(N)); y_e30 = y20 + (y20 - y10) - 2 * cs + rng.normal(0, .3, N)
ok = (y_pw20 >= 10) & (y_pw20 < y30) & (y_e30 >= 10) & (y_e30 < y30)
# shape link (same dose a .2, different pair weights): public pw20 - eoa20 = population (+.45..+.95 in 18/18 skeptic variants, CLAIMED
# implied2 sec 8.0) + paired public noise sd 1.0 (CLAIMED implied2 sec 8.1) -> N(.7, sqrt(.25^2 + 1^2)); soft observation
LINK = g('link', 'on'); DL_MU, DL_SD = g('dl_mu', .7), g('dl_sd', 1.03)
if LINK == 'on':
    logw += -.5 * ((y_pw20 - y20 - DL_MU) / DL_SD) ** 2
# public noise of pw25 - pw30 (for the lomm05 read-noise correlation), standardised
n25 = -.5 * w + r25; z25 = n25 / np.sqrt((.5 * WSD) ** 2 + RSD ** 2)

# ---- stack components: priors from skep_value_mc.py world() (verbatim constants; mcl_skep2 exec'd from its source prefix) ----
src = open(SKV, encoding='utf-8').read(); src = src[:src.index("P('MODEL skep_value_mc.py")]
src = src.replace("OUT = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/skeptic/skep_value_mc.log', 'w')", "OUT = io.StringIO()")
assert 'OUT = io.StringIO()' in src
gv = {'io': io}; exec(compile(src, SKV, 'exec'), gv); gv['N'] = N
rS = np.random.default_rng(SEED + 1)
sc = rS.choice(3, N, p=[.40, .35, .25])
if LOMM == 'stack':
    muL = .85 * (np.array([1.10, -.48, -.26])[sc] + np.array([1.2, 1.2, 1.6])[sc] * rS.standard_normal(N)); I = rS.normal(.16, .12, N)
else:                                               # lomm05 skeptic prior (skep_value_mc 'skeptic')
    muL = rS.normal(0., .9, N); I = rS.normal(0., .12, N)
popg = np.where(sc == 0, False, np.where(sc == 2, True, rS.random(N) < .5))
if MCLP == 'skep2':
    muM = gv['mcl_skep2'](rS, popg)[0]
elif MCLP == 'zero':
    muM = np.zeros(N)
elif MCLP == 'stack':
    muM = .75 * (np.array([2.30, 0., 1.30])[sc] + np.array([1.0, .8, 1.2])[sc] * rS.standard_normal(N))
elif MCLP == 'neg':
    muM = rS.normal(-.5, 1.0, N)
eLp = LSD * (-.2 * z25 + np.sqrt(1 - .04) * rS.standard_normal(N))
eMp = MSD * rS.standard_normal(N)
kapL = np.maximum(0., rS.normal(KAPL_MU, KAPL_SD, N)); kapM = KAPM_MULT * rS.uniform(.45, .85, N)
etaL = rS.standard_normal((2, N)); etaM = rS.standard_normal((2, N))
# MOACF
MIX = {'skep': ((.65, .25, .10), .55), 'optim': ((.75, .18, .07), .8), 'pess': ((.5, .35, .15), .4)}
if MOA in MIX:
    wq, med = MIX[MOA]; kq = rS.choice(3, N, p=wq)
    muQ = np.where(kq == 0, med * np.exp(.6 * rS.standard_normal(N)), np.where(kq == 1, -.1 + .05 * rS.standard_normal(N), -.5 + .3 * rS.standard_normal(N)))
elif MOA == 'regime':
    muQ = rS.normal(.8, 1.37, N)
elif MOA == 'flip':
    muQ = rS.normal(-.5, .5, N)
else:
    muQ = np.zeros(N)
eQp = QSD * rS.standard_normal(N)

L05 = muL + eLp; M10 = muM + eMp
T = L05 + I + M10
y_sw = y_pw25 + T
# ---- A22 observations ----
if A22 == 'on':
    ok &= (y_pw25 >= 10) & (y_pw25 < 20) & (y_sw >= 20) & (y_sw < 30)
    if SWRULER == 'on':
        Gs = Gf(np.clip((y_sw - 20) / 10, 0, 1))
        logw += 4 * np.log(np.clip(Gs, 1e-300, 1)) + 46 * np.log(np.clip(1 - Gs, 1e-300, 1))
else:
    ok &= (y_pw25 >= 10)                            # pre-A22 check: pw25 would at least show .959 (as abc3's floor convention)
logw = np.where(ok, logw, -np.inf)
wt = np.exp(logw - logw[ok].max()); wt /= wt.sum()
ess = 1 / (wt ** 2).sum()


def wq(x, ps=(.1, .5, .9)):
    o = np.argsort(x); cw = np.cumsum(wt[o]); return [float(x[o][min(np.searchsorted(cw, p), len(x) - 1)]) for p in ps]


def S(x):
    m = float((wt * x).sum()); q = wq(x); return m, q


def fmt(x):
    m, q = S(x); return '%+.2f [%+.2f, %+.2f]' % (m, q[0], q[2])


P('MODEL surface_mc.py tag %s | N %d | ESS %.0f (accepted %.3f) | args %s' % (TAG, N, ess, ok.mean(), args))
P('== 1. fitted public values (y = (unrounded - .9580) x 1e4; truncation display) ==')
for nm, x in (('eoa10', y10), ('eoa20', y20), ('eoa30', y_e30), ('pw20', y_pw20), ('pw25', y_pw25), ('pw30', y30), ('swide', y_sw)):
    m, q = S(x); P('  %-6s %.5f [%.5f, %.5f]' % (nm, .958 + m * 1e-4, .958 + q[0] * 1e-4, .958 + q[2] * 1e-4))
P('  pw30 -> .960 gap %s | pw25 - pw30 %s P(>0) %.2f | pw30 - eoa20 %s | pw20 - pw30 %s | swide - .960 %s' % (
    fmt(gap), fmt(y_pw25 - y30), (wt * (y_pw25 > y30)).sum(), fmt(y30 - y20), fmt(y_pw20 - y30), fmt(y_sw - 20)))
P('  stack public T = L + I + M %s | L05 pub %s | M10 pub %s | muL %s | muM %s | I %s' % (fmt(T), fmt(L05), fmt(M10), fmt(muL), fmt(muM), fmt(I)))
P('  E-OA population: v %s c %s w %s | pop pw25 - pw30 %s' % (fmt(v), fmt(c), fmt(w), fmt(-.5 * v - .25 * c)))

# ---- 2. candidates: public delta vs swide, population (private) delta vs swide ----
def Ldose(r, j):            # public / population of the L (+ I) part at lomm dose .05 r on the pw25 base, MINUS the built one
    pub = (r - 1) * (muL + eLp + I) - kapL * r * (r - 1) + TAU * LSD * abs(r - 1) * etaL[j]
    pop = (r - 1) * (muL + I) - kapL * r * (r - 1)
    return pub, pop


def Mdose(r, j):
    pub = (r - 1) * (muM + eMp) - kapM * r * (r - 1) + TAU * MSD * abs(r - 1) * etaM[j]
    pop = (r - 1) * muM - kapM * r * (r - 1)
    return pub, pop


C = {}
C['swide@.30 (= smid: pw30 + L + M)'] = ((y30 - y_pw25) - I, (.5 * v + .25 * c) - I)
C['swide@.20 (pw20 + L + M, I x2)'] = ((y_pw20 - y_pw25) + I, (-.5 * v - .75 * c) + I)
C['MCLmm .05 on swide'] = Mdose(.5, 0)
C['MCLmm .15 on swide'] = Mdose(1.5, 1)
C['lomm .03 on swide'] = Ldose(.6, 0)
C['lomm .08 on swide (swide08)'] = Ldose(1.6, 1)
C['MOACF .05 on swide'] = (muQ + eQp, muQ)
C['spl (swide minus MCLmm)'] = (-M10, -muM)
C['salt (swide minus lomm05)'] = (-(L05 + I), -(muL + I))
C['pw25 (swide minus both)'] = (-T, -(muL + I + muM))
a, b = C['lomm .08 on swide (swide08)'], C['MOACF .05 on swide']; C['swide08 + MOACF'] = (a[0] + b[0], a[1] + b[1])
a, b = C['MCLmm .15 on swide'], C['MOACF .05 on swide']; C['MCLmm .15 + MOACF'] = (a[0] + b[0], a[1] + b[1])
a, b = C['lomm .08 on swide (swide08)'], C['MCLmm .15 on swide']; C['lomm .08 + MCLmm .15'] = (a[0] + b[0], a[1] + b[1])
if RHO0 == 'on':            # skeptic: the private population effect of the CHANGE is independent of anything public (prior-only)
    P('  rho0: population deltas re-drawn from the prior (not updated by public reads)')
P('\n== 2. candidates vs swide (e-4; [p10, p90]) ==')
P('  %-34s | %-24s | %-8s | %-24s | %-8s | %-10s | %s' % ('candidate', 'public delta', 'P(pub>0)', 'population (private E)', 'P(pop>0)', 'R1 incr', 'P(shows .961)'))
RES = {}
for nm, (pub, pop) in C.items():
    if RHO0 == 'on':
        perm = rng.permutation(N); pop = pop[perm]           # break the public-private link, keep the prior marginal
    win = pub > 0
    inc = float((wt * pop * win).sum())
    p961 = float((wt * ((y_sw + pub) >= 30)).sum())
    m, q = S(pub); mp, qp = S(pop)
    RES[nm] = dict(pub=m, pub10=q[0], pub90=q[2], Ppub=float((wt * win).sum()), pop=mp, pop10=qp[0], pop90=qp[2], Ppop=float((wt * (pop > 0)).sum()),
                   R1inc=inc, P961=p961)
    P('  %-34s | %-24s | %.2f     | %-24s | %.2f     | %+.2f      | %.3f' % (nm, fmt(pub), RES[nm]['Ppub'], fmt(pop), RES[nm]['Ppop'], inc, p961))
json.dump(dict(tag=TAG, args=args, ess=float(ess), res=RES), open(HERE + '/surface_%s.json' % TAG, 'w'), indent=1)
