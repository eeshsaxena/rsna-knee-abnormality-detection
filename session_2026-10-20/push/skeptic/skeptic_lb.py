# skeptic_lb.py (10-20 push, skeptic pass, Claude). MODEL checks of implied.md's LB arithmetic, on public LB counts only
# (counts re-verified by re-running work/lbpos.py). No OAI data, no network.
#  1. Re-derive G10 / G20 / gap-to-.960 with the same Poisson-density model, then vary the one weak input: the density just
#     below the .959 floor (implied: k' = k + U(0, 1.5); the full .958 band, 1,341 teams, would need k' ~ 4.3).
#  2. Median of the ratio vs ratio of the medians.
#  3. prior_check sensitivity: how 'R5 / ladder rejected at P .02-.03' moves with the read-noise scale and correlation.
import numpy as np
from scipy.stats import chi2
from scipy.optimize import brentq
rng = np.random.default_rng(2020); B = 200000
W = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push/work/'
r = dict(strip=23, a=29, b=32, c=4, n960=44, n958=1341)


def run(kp_mode):
    n959 = r['a'] + r['b'] + r['c']
    m959 = rng.gamma(n959 + 1, size=B); m960 = rng.gamma(r['n960'] + 1, size=B); k = np.maximum(np.log(m959 / m960), 1e-6)
    D = rng.dirichlet([r['a'] + 1, r['b'] + 1, r['c'] + 1], size=B); F10 = D[:, 0]; F20 = D[:, 0] + D[:, 1]
    inv = lambda F_: -np.log(1 - F_ * (1 - np.exp(-k))) / k
    x10 = inv(F10); x20 = inv(F20); lam0 = m959 * k / (1 - np.exp(-k))
    if kp_mode == 'implied':
        kp = k + rng.uniform(0, 1.5, B)
    elif kp_mode == 'flat':
        kp = np.full(B, 1e-6)
    elif kp_mode == 'fit958':
        # k' such that lam0 * (exp(k') - 1) / k' = 1,341 (whole .958 band as one exponential)
        lam = float(np.median(lam0)); kk = brentq(lambda z: lam * (np.exp(z) - 1) / z - r['n958'], .01, 20); kp = np.full(B, kk)
    else:
        kp = np.full(B, float(kp_mode))
    ms = rng.gamma(r['strip'] + 1, size=B)
    w = np.log1p(kp * ms / lam0) / kp
    return dict(w=w, x10=x10, x20=x20, G10=w + x10, G20=w + x20, gap=1 - x20, kp=kp)


q = lambda v: '%.2f [%.2f, %.2f]' % (np.median(v), np.quantile(v, .1), np.quantile(v, .9))
print('== 1. ALL-teams ruler, counts-k, varying only the density below the .959 floor (band units = 1e-3 macro)')
for m in ('flat', 'implied', 2.5, 'fit958'):
    s = run(m)
    print('  k\' %-8s (median k\' %.2f): 1-u_c4 %s | G10 %s | G20 %s | gap to .960 %s | G20/G10 median %.2f, ratio of medians %.2f' % (
        m, np.median(s['kp']), q(s['w']), q(s['G10']), q(s['G20']), q(s['gap']), np.median(s['G20'] / s['G10']), np.median(s['G20']) / np.median(s['G10'])))
P = np.load(W + 'lb_post_pooled.npz')
print('  implied pooled file: G10 %s | G20 %s | median(G20/G10) %.2f | ratio of medians %.2f' % (q(P['G10']), q(P['G20']), np.median(P['G20'] / P['G10']), np.median(P['G20']) / np.median(P['G10'])))

print('\n== 3. prior_check sensitivity (pooled LB posterior; Gaussian public noise of (S10, S20) at n 390)')
G = np.stack([P['G10'], P['G20']], 1) * 1e-3
H = {'R5 revision': (.0002, -.00005), 'skeptic ladder': (.0003, .0001), 'skeptic_eoa': (.0005, .0007), 'ABC mean': (.00064, .00097)}
for scale, rho in ((1.0, .94), (1.25, .94), (1.5, .94), (1.0, .85), (1.0, .98), (1.5, .85)):
    sd = np.array([.00025, .00052]) * scale
    C = np.array([[sd[0] ** 2, rho * sd[0] * sd[1]], [rho * sd[0] * sd[1], sd[1] ** 2]]); Ci = np.linalg.inv(C)
    row = []
    for k_, m in H.items():
        rr = G - np.array(m); m2 = np.einsum('ij,jk,ik->i', rr, Ci, rr)
        row.append('%s M2 %.1f P %.3f' % (k_, np.median(m2), np.median(chi2.sf(m2, 2))))
    print('  noise x%.2f corr %.2f | %s' % (scale, rho, ' | '.join(row)))
# simple 1-D check: eoa20 alone vs the R5 / ladder population values
for scale in (1.0, 1.5):
    sd20 = .00052 * scale
    print('  1-D eoa20 only, noise x%.1f: z vs R5 (-.00005) %.1f, vs ladder (+.0001) %.1f, vs skeptic_eoa (+.0007) %.1f' % (
        scale, (np.median(G[:, 1]) + .00005) / sd20, (np.median(G[:, 1]) - .0001) / sd20, (np.median(G[:, 1]) - .0007) / sd20))
