"""b_model3.py: does nartaa's second public checkpoint (N2 = .945 r224 crop, same OAI training) add to F+N(+G)? MODEL;
r(N,N2) unmeasured (GUESS .92-.96). Also K10 (320 set) and runtime-safe variants. Laptop, light."""
import numpy as np
from scipy.stats import norm
def d(a): return np.sqrt(2) * norm.ppf(a)
def bk(s, R, w):
    w = np.asarray(w, float); dv = np.array([d(x) for x in s]); return float(norm.cdf((w @ dv) / np.sqrt(2 * (w @ R @ w))))
def psd(R): return np.linalg.eigvalsh(R).min() >= -1e-9
# legs F, N, G, N2
for rFN in (.89, .92):
    for rNN2 in (.92, .94, .96):
        rFN2 = rFN - .01; rGN2 = .838 - .01
        R = np.array([[1, rFN, .84, rFN2], [rFN, 1, .838, rNN2], [.84, .838, 1, rGN2], [rFN2, rNN2, rGN2, 1]])
        if not psd(R): print('npsd', rFN, rNN2); continue
        S = (.944, .949, .929, .945)
        grid = [(f, round(1 - f - g - n2, 3), g, n2) for f in np.arange(.2, .51, .05) for g in (0, .05, .1, .15)
                for n2 in np.arange(0, .41, .05) if 1 - f - g - n2 >= .2]
        b = max((bk(S, R, w), w) for w in grid); b0 = max((bk(S, R, w), w) for w in grid if w[3] == 0)
        print(f'r_FN {rFN} r(N,N2) {rNN2}: best with N2 {b[0]:.4f} at F/N/G/N2 {tuple(round(x, 2) for x in b[1])} | without N2 '
              f'{b0[0]:.4f} at {tuple(round(x, 2) for x in b0[1])} | N2 adds {b[0] - b0[0]:+.4f}')
