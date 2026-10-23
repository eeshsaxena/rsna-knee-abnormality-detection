# lb_post2.py (val 10-20): public-LB posterior refit with the A21 read. MODEL on MEASURED counts (lbpos2.py, push/work/lbpos.py).
# Same exponential in-band density model as push/work/lb_post.py (pooled ALL / F18 / F19b rulers), now saving k, plus:
#  (1) frozen cross-board ruler: the 4 .959 teams above eoa20 at 05:53 UTC are all frozen at 13:20 UTC; exactly 2 of them are
#      below pw30 (E2 r133 / r134 -> A21 r141 / r143) and 2 above (r131 / r132 -> r137 / r139). Given x20 and k, those 4 are iid
#      on (x20, 1) with density ~ exp(-k x), so P(order | x_pw30) = 6 G^2 (1 - G)^2, G = truncated CDF at x_pw30.
#      LB-only posterior for pw30: G ~ Beta(3, 3) (uniform-in-quantile prior, the Dirichlet-equivalent of lb_post.py).
#  (2) cross-check: A21 board alone (3 above / 68 below inside .959, 45 in .960), as abc_update.py does.
#  (3) cross-check: newcomers into .959 since 05:53 (1 above pw30, 15 below) as iid band draws (sensitivity only).
import numpy as np, json
rng = np.random.default_rng(17); B = 300000
RUL = {'ALL': dict(strip=23, a=29, b=32, c=4, n960=44), 'F18': dict(strip=7, a=10, b=11, c=3, n960=11),
       'F19b': dict(strip=18, a=23, b=26, c=3, n960=34)}
def run(r, n):
    n959 = r['a'] + r['b'] + r['c']; m959 = rng.gamma(n959 + 1, size=n); m960 = rng.gamma(r['n960'] + 1, size=n)
    k = np.maximum(np.log(m959 / m960), 1e-6)
    D = rng.dirichlet([r['a'] + 1, r['b'] + 1, r['c'] + 1], size=n); F10 = D[:, 0]; F20 = D[:, 0] + D[:, 1]
    inv = lambda F: -np.log(1 - F * (1 - np.exp(-k))) / k
    x10 = inv(F10); x20 = inv(F20); lam0 = m959 * k / (1 - np.exp(-k)); kp = k + rng.uniform(0, 1.5, size=n)
    ms = rng.gamma(r['strip'] + 1, size=n); w = np.log1p(kp * ms / lam0) / kp
    return dict(k=k, w=w, x10=x10, x20=x20)
P = {}
for nm in RUL:
    s = run(RUL[nm], B // 3)
    for kk, v in s.items(): P.setdefault(kk, []).append(v)
P = {kk: np.concatenate(v) for kk, v in P.items()}
k, x20 = P['k'], P['x20']
# (1) frozen ruler, LB-only: G ~ Beta(3,3) on the truncated density over (x20, 1)
G = rng.beta(3, 3, size=len(k)); e0 = np.exp(-k * x20); e1 = np.exp(-k)
xp = -np.log(e0 - G * (e0 - e1)) / k; d = xp - x20
P['xp'] = xp; P['d'] = d; P['G10'] = P['w'] + P['x10']; P['G20'] = P['w'] + x20; P['gap960'] = 1 - x20; P['gap960p'] = 1 - xp
q = lambda v: '%.2f [%.2f, %.2f]' % (np.median(v), np.quantile(v, .1), np.quantile(v, .9))
print('POOLED (band units = 1e-3): 1-u_c4 %s | x10 %s | x20 %s | G10 %s | G20 %s | to .960 from eoa20 %s' % (
    q(P['w']), q(P['x10']), q(x20), q(P['G10']), q(P['G20']), q(P['gap960'])))
print('FROZEN RULER (LB-only, Beta(3,3)): x_pw30 %s | pw30 - eoa20 %s | pw30 to .960 %s' % (q(xp), q(d), q(1 - xp)))
print('   d / gap960(eoa20) %s' % q(d / (1 - x20)))
for nm, (lo, hi) in (('trunc', (.958, .959)), ('round', (.9575, .9585))):
    print('absolute (%s display): c4 %s | eoa10 %s | eoa20 %s | pw30 %s' % ((nm,) + tuple(
        '%.5f [%.5f, %.5f]' % (np.median(v), np.quantile(v, .1), np.quantile(v, .9)) for v in (
            lo + (1 - P['w']) / 1e3, hi + P['x10'] / 1e3, hi + x20 / 1e3, hi + xp / 1e3))))
# (2) A21 board alone (abc_update.py method): 3 above / 68 below in .959; .960 n 45
rng2 = np.random.default_rng(4); n = 200000
kA = np.maximum(np.log(rng2.gamma(72, size=n) / rng2.gamma(46, size=n)), 1e-3); F = rng2.beta(68 + 1, 3 + 1, size=n)
xA = -np.log(1 - F * (1 - np.exp(-kA))) / kA; dA = xA - rng2.choice(x20, n)
print('A21 BOARD ALONE: x_pw30 %s | pw30 - eoa20 (vs E2 x20) %s' % (q(xA), q(dA)))
# (3) newcomers sensitivity: 16 teams entered .959 since E2 (1 above pw30, 15 below), iid on (0,1) with the band density
GA = (1 - np.exp(-k * xp)) / (1 - np.exp(-k)); lw = 15 * np.log(GA) + np.log(1 - GA); wN = np.exp(lw - lw.max())
o = np.argsort(d); cw = np.cumsum(wN[o]) / wN.sum(); qq = [d[o][np.searchsorted(cw, p)] for p in (.1, .5, .9)]
print('FROZEN RULER + NEWCOMERS: pw30 - eoa20 %.2f [%.2f, %.2f] | x20 %.2f' % (qq[1], qq[0], qq[2], (wN * x20).sum() / wN.sum()))
np.savez('lb_post2.npz', **P)
json.dump({kk: [float(np.median(v)), float(np.quantile(v, .1)), float(np.quantile(v, .9))] for kk, v in P.items()}, open('lb_post2.json', 'w'), indent=1)
