# lbstrat_eb.py (10-20 ~16:20 IST, levers/LB-strategy lens, Claude). MODEL on MEASURED inputs, local CPU only, no OAI data.
# Empirical-Bayes fit of the "knob" probe class (one-line weight / A224 / S6fall / table moves read on the LB 10-17 .. 10-19) from
# OUR OWN LB reads, with the joint public noise of every atomic move from lbstrat_noise.npz (gold58 bootstrap, kernel path).
#   theta_x = sigma_x * (m + t * z_x)       true (population) public-scale effect of atomic move x, sigma_x = its public noise sd
#   pub_x   = theta_x + e_x, e ~ N(0, Sigma_pub) (joint, same 390 public studies)
#   stacks are EXACT sums of single-label atoms (each label's AUC depends only on its own column), so c4 - a25 = fr65 + pf50 +
#   bk50 + lm40, c2max - c4 = fr100 + pf10, c2 - c4 = fr90 + pf10, c3max - c4 = fr100 + pf10 + lat10 (population AND sample).
# Observations (MEASURED order from the API best-id; magnitudes MODEL from the in-band rulers, cited in lbstrat.md):
#   hw3, s615, wr45 < 0 (vs HW, base unchanged); a25 - HW ~ N(1.5, .5) e-4; c4 - a25 ~ N(2.3, .8) e-4 (picks.md ruler);
#   fr65, pf50, a15 (each vs a25) < c4 - a25 (c4 is the max of the parallel A18 reads); c4a15 - c4 < 0;
#   -T < fr100 - c4 < 0 and -T < pf10 - c4 < 0 (display .958); fr100 + pf10, fr90 + pf10, fr100 + pf10 + lat10 < -T (display .957),
#   T = 10 * u_c4 e-4 with u_c4 ~ N(.76, .07) (implied.md).
# Output: posterior over (m, t); derived hit rate, corr(public, private), R1 value and winner's curse of a NEW knob-class probe,
# posterior private value of the accepted knob moves a25 and the c4 stack (= the optimism baked into our public score).
import sys, numpy as np
from scipy.stats import norm
OUT = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/levers/'
reg = sys.argv[1] if len(sys.argv) > 1 else 'g'          # 'g' gold-prevalence noise (primary) or 'r' report-prevalence noise
c4mag = float(sys.argv[2]) if len(sys.argv) > 2 else 2.3  # c4 - a25 public magnitude (e-4); sensitivity 1.0 (implied.md u-ruler)
Z = np.load(OUT + 'lbstrat_noise.npz'); names = list(Z['names']); Cp = Z['cov_pub_' + reg] * 1e8; Cv = Z['cov_priv_' + reg] * 1e8
ATOMS = ['hw3-HW', 's615-HW', 'wr45-HW', 'a25-HW', 'fr65-a25', 'pf50-a25', 'bk50-a25', 'lm40-a25', 'a15-a25', 'c4a15-c4',
         'fr100-c4', 'fr90-c4', 'pf10-c4', 'lat10-c4']
ix = [names.index(a) for a in ATOMS]; S = Cp[np.ix_(ix, ix)]; sig = np.sqrt(np.diag(S)); k = {a: i for i, a in enumerate(ATOMS)}
Sv = Cv[np.ix_(ix, ix)]; r_pv = float(np.median(np.diag(Sv) / np.diag(S)))
Lc = np.linalg.cholesky(S + 1e-9 * np.eye(len(ATOMS)))
rng = np.random.default_rng(1020)
N = 60000
zN = rng.standard_normal((N, len(ATOMS))); eN = rng.standard_normal((N, len(ATOMS))) @ Lc.T; uN = rng.normal(.76, .07, N)


def lik(m, t, ret=False):
    th = sig[None, :] * (m + t * zN); P = th + eN
    g = lambda a: P[:, k[a]]
    T = 10 * uN
    c4s = g('fr65-a25') + g('pf50-a25') + g('bk50-a25') + g('lm40-a25')
    ok = ((g('hw3-HW') < 0) & (g('s615-HW') < 0) & (g('wr45-HW') < 0) & (g('fr65-a25') < c4s) & (g('pf50-a25') < c4s) & (g('a15-a25') < c4s)
          & (g('c4a15-c4') < 0) & (g('fr100-c4') < 0) & (g('fr100-c4') > -T) & (g('pf10-c4') < 0) & (g('pf10-c4') > -T)
          & (g('fr100-c4') + g('pf10-c4') < -T) & (g('fr90-c4') + g('pf10-c4') < -T) & (g('fr100-c4') + g('pf10-c4') + g('lat10-c4') < -T))
    w = ok * norm.pdf(g('a25-HW'), 1.5, .5) * norm.pdf(c4s, c4mag, .8)
    if ret:
        return w, th, P
    return w.mean()


ms = np.round(np.arange(-1.6, 1.01, .1), 2); ts = np.round(np.arange(0.0, 2.01, .1), 2)
LK = np.array([[lik(m, t) for t in ts] for m in ms])
post = LK / LK.sum()
pm = post.sum(1); pt = post.sum(0)
lines = []
pr = lambda s: (print(s), lines.append(s))
pr('lbstrat_eb | noise regime %s | c4 - a25 magnitude %.1f e-4 | atoms %d | private/public variance ratio %.3f' % (reg, c4mag, len(ATOMS), r_pv))
pr('atom public sd (e-4): ' + ', '.join('%s %.2f' % (a.split('-')[0], s) for a, s in zip(ATOMS, sig)))
Em = (pm * ms).sum(); Et = (pt * ts).sum()
cdf = lambda p, x, q: x[np.searchsorted(np.cumsum(p), q)]
pr('posterior m (mean effect, sigma units): mean %+.2f  p10/50/90 %+.2f %+.2f %+.2f' % (Em, cdf(pm, ms, .1), cdf(pm, ms, .5), cdf(pm, ms, .9)))
pr('posterior t (effect spread, sigma units): mean %.2f  p10/50/90 %.2f %.2f %.2f' % (Et, cdf(pt, ts, .1), cdf(pt, ts, .5), cdf(pt, ts, .9)))
pr('P(m < 0) = %.3f' % pm[ms < 0].sum())
# derived quantities, averaged over the (m, t) posterior; a NEW knob-class probe with public sd sigma, private sd sqrt(r) sigma
D = {}
for i, m in enumerate(ms):
    for j, t in enumerate(ts):
        s = np.sqrt(1 + t * t); a = m / s
        hit = norm.cdf(a)
        corr = t * t / np.sqrt((1 + t * t) * (t * t + r_pv)) if t > 0 else 0.0
        val = m * norm.cdf(a) + (t * t / s) * norm.pdf(a)                     # E[theta 1(pub > 0)] / sigma (R1 adopts on a win)
        wc = (1 / s) * norm.pdf(a) / max(norm.cdf(a), 1e-12)                  # E[pub - theta | pub > 0] / sigma
        # P(theta > 0 | pub > 0): bivariate normal
        rho = t / s if t > 0 else 0.0
        from scipy.stats import multivariate_normal as mvn
        p_both = mvn.cdf([m / max(t, 1e-9) if t > 0 else (1e9 if m > 0 else -1e9), a], mean=[0, 0], cov=[[1, rho], [rho, 1]]) if t > 0 else (hit if m > 0 else 0.0)
        for kk, v in (('hit', hit), ('corr', corr), ('val', val), ('wc', wc), ('ppos_given_win', p_both / max(hit, 1e-12))):
            D[kk] = D.get(kk, 0) + post[i, j] * v
pr('NEW knob-class probe (per unit of its public sd sigma):')
pr('  P(reads positive on R1) %.3f | corr(public delta, private delta) %.3f | P(true > 0 | read positive) %.3f' % (D['hit'], D['corr'], D['ppos_given_win']))
pr('  E[private gain under R1] %+.3f sigma | winner\'s curse E[public - true | win] %.3f sigma' % (D['val'], D['wc']))
for s_ in (2.0, 3.7, 5.0):
    pr('    sigma %.1fe-4: E[private gain under R1] %+.2fe-4, public optimism of a win %.2fe-4' % (s_, D['val'] * s_, D['wc'] * s_))
# posterior of the ACCEPTED knob moves (a25, c4 stack): private expectation vs the public magnitudes
acc = {'a25': 0., 'c4stack': 0., 'pub_a25': 0., 'pub_c4': 0., 'priv_le0_a25': 0., 'priv_le0_c4': 0., 'W': 0.}
rngp = np.random.default_rng(7)
for i, m in enumerate(ms):
    for j, t in enumerate(ts):
        if post[i, j] < 1e-4:
            continue
        w, th, P = lik(m, t, ret=True)
        if w.sum() <= 0:
            continue
        ww = w / w.sum() * post[i, j]
        th_a = th[:, k['a25-HW']]; th_c = th[:, k['fr65-a25']] + th[:, k['pf50-a25']] + th[:, k['bk50-a25']] + th[:, k['lm40-a25']]
        pa = P[:, k['a25-HW']]; pc = P[:, k['fr65-a25']] + P[:, k['pf50-a25']] + P[:, k['bk50-a25']] + P[:, k['lm40-a25']]
        sda = np.sqrt(Sv[k['a25-HW'], k['a25-HW']]); ic = [k[a] for a in ('fr65-a25', 'pf50-a25', 'bk50-a25', 'lm40-a25')]
        sdc = np.sqrt(Sv[np.ix_(ic, ic)].sum())
        priv_a = th_a + rngp.standard_normal(N) * sda; priv_c = th_c + rngp.standard_normal(N) * sdc
        acc['a25'] += (ww * th_a).sum(); acc['c4stack'] += (ww * th_c).sum(); acc['pub_a25'] += (ww * pa).sum(); acc['pub_c4'] += (ww * pc).sum()
        acc['priv_le0_a25'] += (ww * (priv_a <= 0)).sum(); acc['priv_le0_c4'] += (ww * (priv_c <= 0)).sum(); acc['W'] += ww.sum()
for kk in acc:
    if kk != 'W': acc[kk] /= acc['W']
pr('ACCEPTED knob moves: a25 public %.2f -> population (private expectation) %+.2f e-4, P(private <= 0) %.2f' % (acc['pub_a25'], acc['a25'], acc['priv_le0_a25']))
pr('                     c4 stack public %.2f -> population %+.2f e-4, P(private <= 0) %.2f' % (acc['pub_c4'], acc['c4stack'], acc['priv_le0_c4']))
pr('                     knob optimism baked into the public chain HW -> c4: %.2f e-4' % (acc['pub_a25'] + acc['pub_c4'] - acc['a25'] - acc['c4stack']))
open(OUT + 'lbstrat_eb_%s_%s.log' % (reg, ('%.1f' % c4mag).replace('.', 'p')), 'w').write('\n'.join(lines) + '\n')
