"""Can we recover which DINO / Rad fold model held out each gold58 study? In-sample members fit the gold labels
(.96-.99 AUC), the held-out fold member should show clearly the largest per-study loss. If the argmax fold agrees
across the 4 DINO seed-runs, the split is shared and the inference is validated."""
import numpy as np
from collections import Counter
from inject_common import *
D = load_all(); Y = D['Y']; n = len(Y)


def loss(p):
    p = np.clip(np.asarray(p, np.float64), 1e-4, 1 - 1e-4)
    return -(Y * np.log(p) + (1 - Y) * np.log(1 - p)).mean(1)          # per study


runs = sorted(set(D['dino_run']))
arg = {}
for r in runs:
    idx = [i for i, rr in enumerate(D['dino_run']) if rr == r]
    folds = [D['dino_fold'][i] for i in idx]
    L = np.stack([loss(D['dino'][i]) for i in idx])                      # (5, n)
    o = np.argsort(-L, axis=0)
    a = np.array([folds[k] for k in o[0]])
    ratio = L[o[0], np.arange(n)] / L[o[1], np.arange(n)]
    arg[r] = a
    print(f'{r:24s} argmax-fold counts {sorted(Counter(a.tolist()).items())}  max/2nd loss ratio median {np.median(ratio):.2f} '
          f'p10 {np.quantile(ratio, .1):.2f}  member per-study loss median {np.median(L):.3f}')
A = np.stack([arg[r] for r in runs])
agree = (A == A[0]).all(0)
print('studies where all 4 DINO runs agree on the held-out fold:', int(agree.sum()), '/', n)
maj = np.array([Counter(A[:, s].tolist()).most_common(1)[0] for s in range(n)])
print('majority size distribution:', sorted(Counter(maj[:, 1].tolist()).items()))
for k in ('rad_e10_raw', 'rad_e13_raw', 'rad_legacy_second_raw'):
    L = np.stack([loss(h) for h in D[k]])
    o = np.argsort(-L, axis=0); ratio = L[o[0], np.arange(n)] / L[o[1], np.arange(n)]
    print(f'{k:22s} argmax-head counts {sorted(Counter(o[0].tolist()).items())} ratio median {np.median(ratio):.2f} '
          f'p10 {np.quantile(ratio, .1):.2f}; agreement with DINO majority fold: {(o[0] == maj[:, 0]).mean():.2f}')
    print('   per-head gold58 AUC', [round(mauc(Y, h), 3) for h in D[k]])
L = np.stack([loss(f) for f in D['a5_raw']]); o = np.argsort(-L, axis=0)
print('a5 folds argmax counts', sorted(Counter(o[0].tolist()).items()), 'agree w DINO maj', (o[0] == maj[:, 0]).mean().round(2),
      'per-fold AUC', [round(mauc(Y, f), 3) for f in D['a5_raw']])
print('DINO per-member gold58 AUC', [round(mauc(Y, m), 3) for m in D['dino']])
np.save('oof_dino_fold_majority.npy', maj)
