# pair_mc.py (10-22 ~09:30 IST, Claude, PUBLIC-PUSH track). MODEL. LOCAL CPU only. Reuses the fitted public response surface of
# session_2026-10-21/lastday/work/surface_mc.py VERBATIM (sections 0-1: censored public reads + priors; exec'd from its source, only the
# HERE / log path changed and the candidate section cut), then scores PAIRS of 10-22 public-push entries on the PUBLIC objective only:
#   P(at least one entry's public > swide), with and without the S6F canary (each entry clean w.p. PC, independent; CLAIMED CANARY_1021 .55),
#   E[public teams passed] by the better clean entry (band ruler: 46 .960 teams above swide, 28 in .961, same exp(-k x) density).
# Candidates (public delta vs swide, e-4):
#   Q05 MOACF .05 (surface: muQ + eQp), L08 lomm .08, M15 MCLmm .15, L08M15, L08Q05, M15Q05 (surface dose curves),
#   CVX = CONS(swide): mu25 ~ N(.3, 1.1) + I(CVX, lomm05) ~ N(-.2, .3) (levers/lever_value_mc.py CVXP 'rk', CLAIMED) + public read noise
#        sd CSD (2.62 MEASURED work/pair_noise.log on the swide base), corr -.36 with pw25 - pw30 public noise and +.20 with L08 noise
#        (MEASURED pair_noise.log), ~0 with MOACF (+.03 MEASURED).
# Read-noise cross-correlations used: Q05~L08 +.10, Q05~M15 ~0, Q05~CVX +.03 (MEASURED pair_noise.log); the surface model already makes
# L08 / M15 / combos share the fitted L05 / M10 posteriors, so their public deltas are correlated through the fit (as they should be).
import sys, io, json, itertools, numpy as np
SRC = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/lastday/work/surface_mc.py'
HERE = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-22/final/work'
argv = sys.argv[1:]
a2 = dict(a.split('=', 1) for a in argv)
NAB = float(a2.pop('nab', 48)); N961 = float(a2.pop('n961', 31)); PC = float(a2.pop('pc', .55)); CSD = float(a2.pop('csd', 2.62)); CVXP = a2.pop('cvx', 'rk'); QL = float(a2.pop('ql', .10))
src = open(SRC, encoding='utf-8').read()
src = src[:src.index('# ---- 2. candidates')]
src = src.replace("HERE = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/lastday/work'", "HERE = %r" % HERE)
src = src.replace("LOG = open(HERE + '/surface_%s.log' % TAG, 'w')", "LOG = open(HERE + '/pair_%s.log' % TAG, 'w')")
assert "HERE = %r" % HERE in src and "/pair_%s.log" in src
sys.argv = [SRC] + ['%s=%s' % kv for kv in a2.items()]
g = {'__name__': 'surface'}; exec(compile(src, SRC, 'exec'), g)
P, S, fmt, wt, N = g['P'], g['S'], g['fmt'], g['wt'], g['N']
rS = np.random.default_rng(g['SEED'] + 7)
muL, eLp, I, muM, eMp, kapL, kapM, etaL, etaM = (g[k] for k in ('muL', 'eLp', 'I', 'muM', 'eMp', 'kapL', 'kapM', 'etaL', 'etaM'))
TAU, LSD, MSD, muQ, eQp, z25, y_sw, kk = (g[k] for k in ('TAU', 'LSD', 'MSD', 'muQ', 'eQp', 'z25', 'y_sw', 'kk'))
def Ldose(r, j):
    return (r - 1) * (muL + eLp + I) - kapL * r * (r - 1) + TAU * LSD * abs(r - 1) * etaL[j]
def Mdose(r, j):
    return (r - 1) * (muM + eMp) - kapM * r * (r - 1) + TAU * MSD * abs(r - 1) * etaM[j]
Q05 = muQ + eQp
L08 = Ldose(1.6, 1); M15 = Mdose(1.5, 1)
# small extra read-noise correlation Q05 ~ L08 (+.10 MEASURED): add a shared term to the L08 increment's own eta
L08 = L08 + QL * .73 * (eQp / g['QSD'])
CV = {'rk': ((.3, 1.1), (-.2, .3)), 'lskep': ((.2, .8), (-.2, .3)), 'primary': ((.5, 1.0), (-.2, .3)), 'zero': ((0, 0), (0, 0)), 'g58': ((-1.5, 1.1), (-.2, .3))}
(m25, s25), (mi, si) = CV[CVXP]
zl = (L08 - L08.mean()) / L08.std()
eC = CSD * (-.36 * z25 + .20 * zl + np.sqrt(1 - .36 ** 2 - .20 ** 2) * rS.standard_normal(N))
CVX = rS.normal(m25, s25, N) + rS.normal(mi, si, N) + eC
C = {'Q05 MOACF(swide)': Q05, 'L08 swide lomm .08': L08, 'M15 swide MCLmm .15': M15, 'CVX CONS(swide)': CVX,
     'L08M15 lomm .08 + MCLmm .15': L08 + M15, 'L08Q05 swide08 + MOACF': L08 + Q05, 'M15Q05 MCLmm .15 + MOACF': M15 + Q05}
# rank ruler: teams passed when public moves from y_sw to y_sw + d (46 .960 teams above swide, 28 .961 teams; density exp(-k x) per band)
Gf = lambda x: (1 - np.exp(-kk * x)) / (1 - np.exp(-kk))
xs = np.clip((y_sw - 20) / 10, 0, 1); Gs = Gf(xs)
def passed(d):
    yn = y_sw + np.maximum(d, 0); x1 = np.clip((yn - 20) / 10, 0, 1)
    in960 = NAB * (Gf(x1) - Gs) / np.maximum(1 - Gs, 1e-9)
    x2 = np.clip((yn - 30) / 10, 0, 1); return in960 + N961 * Gf(x2) * (yn >= 30)
clean = {k: rS.random(N) < PC for k in C}
P('\nPAIR MC | PC %.2f | CVX prior %s | CSD %.2f | QL %.2f | surface args %s' % (PC, CVXP, CSD, QL, a2))
P('== singles: public delta vs swide [p10, p90] | P(pub > swide) | P(clean and >) | E teams passed (with canary) | E teams | clean')
RES = {'single': {}, 'pair': {}}
for k, x in C.items():
    p1 = float((wt * (x > 0)).sum()); tp = passed(x); et = float((wt * tp).sum())
    pc1 = float((wt * ((x > 0) & clean[k])).sum()); etc = float((wt * tp * clean[k]).sum())
    RES['single'][k] = dict(pub=S(x)[0], P=p1, Pc=pc1, Eteams=et, Eteams_c=etc)
    P('  %-30s %-24s | %.3f | %.3f | %5.2f | %5.2f' % (k, fmt(x), p1, pc1, etc, et))
P('== pairs (sorted by P(>=1 clean beats swide)); corr = weighted corr of the two public deltas')
rows = []
for (ka, xa), (kb, xb) in itertools.combinations(C.items(), 2):
    pn = float((wt * ((xa > 0) | (xb > 0))).sum())
    wa = (xa > 0) & clean[ka]; wb = (xb > 0) & clean[kb]; pcl = float((wt * (wa | wb)).sum())
    best = np.maximum(np.where(clean[ka], xa, -1e9), np.where(clean[kb], xb, -1e9))
    et = float((wt * passed(best)).sum())
    ma, mb = (wt * xa).sum(), (wt * xb).sum()
    cr = float((wt * (xa - ma) * (xb - mb)).sum() / np.sqrt((wt * (xa - ma) ** 2).sum() * (wt * (xb - mb) ** 2).sum()))
    rows.append((pcl, pn, et, cr, ka.split()[0], kb.split()[0]))
for pcl, pn, et, cr, a, b in sorted(rows, reverse=True):
    RES['pair']['%s + %s' % (a, b)] = dict(P_noCanary=pn, P_canary=pcl, Eteams_c=et, corr=cr)
    P('  %-7s + %-7s | P(>=1 > swide) %.3f | with canary %.3f | E teams %5.2f | corr %+.2f' % (a, b, pn, pcl, et, cr))
json.dump(dict(args=argv, res=RES), open(HERE + '/pair_%s.json' % g['TAG'], 'w'), indent=1)
