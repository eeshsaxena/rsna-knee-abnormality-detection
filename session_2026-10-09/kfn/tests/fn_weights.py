#!/usr/bin/env python3
"""fn_weights.py (kfn, MODEL; same macro binormal as k9n/k9n_weights.py blend2): 2-leg F/N rank-blend weight for FN.
F .944 (K7i SOLO; K9 SOLO not read), N_true .949 / .9475 (winner's curse), F SOLO .942-.946, r_FN .86-.94 (not measured).
Prints the per-scenario optimal N weight and the regret of W_N = .60 (the research's FN weight) and of .50 / .70."""
import numpy as np
from scipy.stats import norm
d = lambda a: np.sqrt(2) * norm.ppf(a)
def b2(a, b, r, W):
    return norm.cdf(((1 - W) * d(a) + W * d(b)) / np.sqrt(2 * ((1 - W) ** 2 + W ** 2 + 2 * W * (1 - W) * r)))
ws = np.linspace(0, 1, 1001)
rows, reg = [], {0.5: [], 0.6: [], 0.7: []}
for F in (.942, .944, .946):
    for N in (.9475, .949):
        for r in (.86, .88, .90, .92, .94):
            v = b2(F, N, r, ws); i = v.argmax()
            rows.append((F, N, r, ws[i], v[i]))
            for w in reg:
                reg[w].append(v[i] - b2(F, N, r, w))
wo = np.array([x[3] for x in rows])
print(f'{len(rows)} scenarios: optimal W_N {wo.min():.2f}-{wo.max():.2f}, median {np.median(wo):.2f}')
for w, rg in reg.items():
    print(f'W_N {w:.2f}: max regret {max(rg):.5f}, mean {np.mean(rg):.5f}')
c = [x for x in rows if x[0] == .944 and x[1] == .949 and x[2] == .92][0]
print(f'central (F .944, N .949, r .92): optimal W_N {c[3]:.2f} AUC {c[4]:.4f}; at .60 {b2(.944, .949, .92, .6):.4f}')
