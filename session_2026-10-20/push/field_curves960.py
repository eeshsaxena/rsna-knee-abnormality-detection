# field_curves960.py (10-20 push): r5/field_curves.py with two extra rank thresholds: r130 (today's .960+ public band size) and r167 (projected .960+ band at the deadline, field_mc t1_bands). MODEL / GUESS field params unchanged.
# "near-gold" thresholds, from the 10-19 think/picks field model (field_mc.py head, reused verbatim via exec; MODEL / GUESS params).
# Writes curves_<FS>.npz into r5 (nothing is written into picks_work). CPU only.
import sys, os, numpy as np
PW = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/picks_work'
OUT = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push'
FS = sys.argv[1] if len(sys.argv) > 1 else 'base'
NDRAW = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
src = open(PW + '/field_mc.py').read(); cut = src.index('idx = rng.choice')
cwd = os.getcwd(); os.chdir(PW)
sys.argv = [sys.argv[0], 'base_report', FS, '2000']
G = {'__name__': 'field_head'}
exec(compile(src[:cut], 'field_mc_head', 'exec'), G)       # builds MU, SD, OTHER, P, silver, bronze, gold, nF, rng
os.chdir(cwd)
MU, SD, OTHER, P, nF = G['MU'], G['SD'], G['OTHER'], G['P'], G['nF']
silver, bronze, gold = G['silver'], G['bronze'], G['gold']
rng = np.random.default_rng(2010)
grid = np.linspace(-6e-4, 3.0e-3, 181)
K = {'gold': gold, 'r50': 50, 'r100': 100, 'r130': 130, 'r167': 167, 'r200': 200, 'silver': silver, 'bronze': bronze}
acc = {k: np.zeros(grid.size) for k in K}; erank = np.zeros(grid.size); n = 0
for i in range(0, NDRAW, 1000):
    y = MU + SD * rng.standard_normal((1000, nF)) + OTHER * (P['lin_sd'] * rng.standard_normal((1000, 1)))
    y.sort(axis=1)
    above = nF - np.stack([np.searchsorted(y[j], grid, side='right') for j in range(1000)])   # (1000, grid)
    r = 1 + above + np.where(grid < 0, P['n_copy'], 0)[None, :]
    for k, thr in K.items(): acc[k] += (r <= thr).sum(0)
    erank += np.minimum(r, 2000).sum(0); n += 1000
cur = {k: v / n for k, v in acc.items()}
np.savez(OUT + f'/curves_{FS}.npz', grid=grid, erank=erank / n, **cur, thr=np.array([K[k] for k in K]), names=np.array(list(K)))
xs = grid[np.argmax(cur['silver'] >= .5)]
print(f'FIELD {FS}: N {P["N"]} gold r{gold} silver r{silver} bronze r{bronze} | silver x* {xs*1e4:.2f}e-4')
for x in (0, 2e-4, 4e-4, 5e-4, 6e-4, 8e-4, 1e-3, 1.5e-3, 2e-3, 3e-3):
    j = np.argmin(abs(grid - x))
    print('  x %+.1fe-4: ' % (grid[j] * 1e4) + ' '.join('%s %.3f' % (k, cur[k][j]) for k in K))
