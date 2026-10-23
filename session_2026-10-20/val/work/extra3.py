# extra3.py: (a) P(display .960) for candidates if pw30's distance to .960 is taken from the curve-free LB-only posterior (lb_post2.py)
# instead of the joint posterior (bracket only; candidate increments over pw30 from the primary joint posterior, paired independently).
# (b) population-optimal shape among the candidate set under the primary posterior. MODEL.
import numpy as np, abc3
R = abc3.run(tag='primary-arrays', M_=600000, quiet=True)
A = R['_arr']; w = A['w']; tot = w.sum(); G = A['G']; Pop = A['Pop']; x20 = A['x20']
LB = np.load('lb_post2.npz'); rng = np.random.default_rng(3)
gapLB = 1 - LB['xp']                          # pw30 to .960, LB-only (band units)
idx = rng.choice(len(w), 400000, p=w / tot); g = rng.choice(gapLB, 400000)
gapJ = 1 - (x20 + (G['PW30'] - G['S20']) * 1e3)
print('pw30 to .960 (e-4): joint %.2f [%.2f, %.2f] | LB-only %.2f [%.2f, %.2f]' % tuple(
    [np.median(gapJ[idx]) * 10, np.quantile(gapJ[idx], .1) * 10, np.quantile(gapJ[idx], .9) * 10, np.median(g) * 10, np.quantile(g, .1) * 10, np.quantile(g, .9) * 10]))
for v in ('PW25', 'M20_LP30', 'LP30', 'PW35', 'PW40', 'PW45', 'S40', 'S25', 'LPW30'):
    inc = (G[v] - G['PW30'])[idx] * 1e3
    print('  %-9s P(.960) joint %.2f | with LB-only gap %.2f | P(.961) LB-only %.3f' % (v, np.mean(inc >= gapJ[idx]), np.mean(inc >= g), np.mean(inc >= g + 1)))
names = ['S20', 'S25', 'S30', 'PW20', 'PW25', 'PW30', 'PW35', 'PW40', 'M20_LP30', 'LP30', 'LPW30', 'LP35']
P = np.stack([Pop[v] for v in names], 1); bi = np.argmax(P, 1)
print('population-optimal (private-relevant) shape, posterior mass:', ' '.join('%s %.2f' % (n, (w * (bi == i)).sum() / tot) for i, n in enumerate(names)))
GG = np.stack([G[v] for v in names], 1); bg = np.argmax(GG, 1)
print('public-optimal shape (this public sample), posterior mass:', ' '.join('%s %.2f' % (n, (w * (bg == i)).sum() / tot) for i, n in enumerate(names)))
