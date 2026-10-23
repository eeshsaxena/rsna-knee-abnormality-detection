"""k9n_weights_stress.py: MODEL/GUESS stress of the shipped triples vs N's TRUE strength (winner's curse / public-only
OAI gain). Central r_PF .945, r_PN .925 (p949_model.py points), r_FN .90 / .92. Same binormal as k9n_weights.py."""
import numpy as np
from scipy.stats import norm
def d(a): return np.sqrt(2) * norm.ppf(a)
def b3(s, R, w):
    w = np.asarray(w, float); dv = np.array([d(x) for x in s]); return norm.cdf((w @ dv) / np.sqrt(2 * (w @ R @ w)))
C = {'.10/.30/.60': (.10, .30, .60), '.15/.25/.60': (.15, .25, .60), '.30/.30/.40': (.30, .30, .40),
     '.40/.30/.30': (.40, .30, .30), 'P+F .50/.50/0 (K9 main)': (.5, .5, 0), 'P+N .28/0/.72': (.28, 0, .72)}
out = []
for rFN in (.90, .92):
    R = np.array([[1, .945, .925], [.945, 1, rFN], [.925, rFN, 1]])
    out.append(f'r_FN {rFN}: N_true ->   ' + '  '.join(f'{x:.4f}' for x in (.944, .946, .9475, .949)))
    for k, w in C.items():
        out.append(f'  {k:26s} ' + '  '.join(f'{b3((.943, .944, nt), R, w):.4f}' for nt in (.944, .946, .9475, .949)))
print('\n'.join(out)); open(__file__.replace('.py', '.out'), 'w').write('\n'.join(out) + '\n')
