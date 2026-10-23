"""b_model2.py: 4-leg (P,F,N,G) blends with r_PG .89 (fitted from the goodpjw2008 P+G sweep), r_GN .838 (N+G cluster),
r_PF .945, r_PN .945; r_FN and r_FG scanned around the one-factor estimate (r_FN ~.89, r_FG ~.84). MODEL. Laptop, light."""
import itertools
import numpy as np
from scipy.stats import norm
def d(a): return np.sqrt(2) * norm.ppf(a)
def bk(s, R, w):
    w = np.asarray(w, float); dv = np.array([d(x) for x in s]); return float(norm.cdf((w @ dv) / np.sqrt(2 * (w @ R @ w))))
def psd(R): return np.linalg.eigvalsh(R).min() >= -1e-9
def R4(rFN, rFG, rPF=.945, rPN=.945, rPG=.89, rGN=.838):
    return np.array([[1, rPF, rPN, rPG], [rPF, 1, rFN, rFG], [rPN, rFN, 1, rGN], [rPG, rFG, rGN, 1]])
S = (.943, .944, .949, .929)
C = {
    'K9N-a  .10/.30/.60/0': (.10, .30, .60, 0),
    'MR     .05/.35/.60/0': (.05, .35, .60, 0),
    'FN     0/.40/.60/0': (0, .40, .60, 0),
    'FNG    0/.35/.50/.15': (0, .35, .50, .15),
    'FNG    0/.30/.55/.15': (0, .30, .55, .15),
    'FNG    0/.35/.55/.10': (0, .35, .55, .10),
    'PFNG   .05/.30/.50/.15': (.05, .30, .50, .15),
    'N+G cluster 0/0/.838/.162': (0, 0, .838, .162),
}
grid = [(p, f, round(1 - p - f - g, 3), g) for p in (0, .05, .10) for f in np.arange(.15, .56, .05) for g in np.arange(0, .31, .05)
        if 1 - p - f - g >= .2]
print(f'{"candidate":28s} ' + ' '.join(f'FN{rFN:.2f}/FG{rFG:.2f}' for rFN in (.86, .89, .92, .94) for rFG in (.84, .91)))
for nm, w in C.items():
    row = []
    for rFN in (.86, .89, .92, .94):
        for rFG in (.84, .91):
            R = R4(rFN, rFG)
            row.append(f'{bk(S, R, w):.4f}' if psd(R) else '  npsd')
    print(f'{nm:28s} ' + ' '.join(f'{x:>12s}' for x in row))
print('\nbest 4-leg weights (P,F,N,G) per scenario (.05 grid, P<=.10):')
for rFN in (.86, .89, .92, .94):
    for rFG in (.84, .88, .91):
        R = R4(rFN, rFG)
        if not psd(R): print(f'  r_FN {rFN} r_FG {rFG}: not PSD'); continue
        v = [(bk(S, R, w), w) for w in grid]; b = max(v)
        b3 = max((bk(S, R, w), w) for w in grid if w[3] == 0)
        print(f'  r_FN {rFN} r_FG {rFG}: best {b[0]:.4f} at {tuple(round(x, 2) for x in b[1])} | best without G {b3[0]:.4f} at '
              f'{tuple(round(x, 2) for x in b3[1])} | G adds {b[0] - b3[0]:+.4f}')
# K9N-a's read as an r_FN detector: modelled raw value minus .0005 optimism, by r_FN, at F .944/.945/.946
print('\nK9N-a (.10/.30/.60) modelled DISPLAY-equivalent (raw - .0005) by r_FN and F SOLO:')
for F in (.943, .944, .945, .946):
    print(f'  F {F}: ' + ' '.join(f'rFN {r:.2f}->{bk((.943, F, .949, .929), R4(r, .88), (.10, .30, .60, 0)) - .0005:.4f}'
                                for r in (.84, .86, .88, .90, .92, .94, .96) if psd(R4(r, .88))))
