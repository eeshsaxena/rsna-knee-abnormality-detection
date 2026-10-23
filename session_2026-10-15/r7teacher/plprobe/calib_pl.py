# calib_pl.py: calibrate the sim's error correlations to MEASURED/CLAIMED LB anchors (population, large N):
#   (a) rho_RR (R7 vs Raptor errors): V224 blend of components at ~.949-.950 must land near public .956 (MEASURED display;
#       r384 solo .949 and r224 .945 CLAIMED by their authors; R7 solo GUESS .950).
#   (b) rho_F (fleet vs both): F (solo GUESS .944) at f .2 > f 0 (MEASURED: HFR7-RR beat HR7-RR) and f .3 < f .2
#       (MEASURED 10-14: RR-F30 57163771 did not become team best).
import sys, numpy as np
from scipy.stats import rankdata
from scipy.special import ndtri
PREV = np.array([.278, .174, .533, .254, .367, .247, .452, .605, .132, .253, .294, .071])
GOLD_N = np.array([.964, .943, .982, .896, .981, .776, .860, .973, .829, .976, .937, .958])
LEVEL = .949 + .5 * (GOLD_N - GOLD_N.mean())
rng = np.random.default_rng(1)
def auc(s, y):
    r = rankdata(s); p = y.sum(); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * (len(y) - p))
pct = lambda x: rankdata(x) / len(x)
def run(rrr, r34, rf, fA=.944 - .949, n=60000):
    C = np.full((4, 4), rrr); np.fill_diagonal(C, 1); C[1, 2] = C[2, 1] = r34; C[3, :3] = C[:3, 3] = rf
    L = np.linalg.cholesky(C)
    res = {k: [] for k in ('R7', 'r384', 'mix', 'V224', 'f10', 'f20', 'f30', 'F')}
    for c in range(12):
        A = np.array([LEVEL[c] + .0005, LEVEL[c] - .0005, LEVEL[c] - .0045, LEVEL[c] + fA])
        al = np.sqrt(2) * ndtri(np.clip(A, .6, .995))
        y = (rng.random(n) < PREV[c]).astype(int)
        S = al[None, :] * y[:, None] + rng.standard_normal((n, 4)) @ L.T
        P = np.column_stack([pct(S[:, m]) for m in range(4)])
        mix = .65 * P[:, 1] + .35 * P[:, 2]; H = .5 * mix + .5 * P[:, 0]; pH = pct(H)
        res['R7'].append(auc(P[:, 0], y)); res['r384'].append(auc(P[:, 1], y)); res['mix'].append(auc(mix, y)); res['F'].append(auc(P[:, 3], y))
        res['V224'].append(auc(H, y))
        for f in (.1, .2, .3):
            res['f%d' % (f * 100)].append(auc((1 - f) * pH + f * P[:, 3], y))
    return {k: np.mean(v) for k, v in res.items()}
print('rho_RR r384~r224 rho_F | R7 r384 mix V224 | F | f10-0 f20-0 f30-0 (x1e4)')
for rrr in (.80, .85, .88):
    for r34 in (.93,):
        for rf in (.70, .75, .80, .85):
            m = run(rrr, r34, rf)
            print(f'{rrr:.2f} {r34:.2f} {rf:.2f} | {m["R7"]:.4f} {m["r384"]:.4f} {m["mix"]:.4f} {m["V224"]:.4f} | {m["F"]:.4f} | '
                  f'{(m["f10"]-m["V224"])*1e4:+.1f} {(m["f20"]-m["V224"])*1e4:+.1f} {(m["f30"]-m["V224"])*1e4:+.1f}', flush=True)
