#!/usr/bin/env python
# recompute_flrm.py (2026-10-14, Claude subagent): standalone recompute of the V224-FLIP-RM commit blend. numpy + pandas only, its
# own average-tie rank (counting), no gate / builder / scipy code. Run from the project root. No em dashes.
#   usage: python session_2026-10-15/v224rm/recompute_flrm.py OUT [FLIP_OUT]
import sys
import numpy as np, pandas as pd

O = sys.argv[1]
F = sys.argv[2] if len(sys.argv) > 2 else 'session_2026-10-14/v224/out_v224flip_v1'
S = pd.read_csv('_data/sample_submission.csv', dtype=str)
ids, L = S.StudyInstanceUID.tolist(), list(S.columns[1:])
rd = lambda d, f: pd.read_csv(f'{d}/{f}', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values.astype(np.float64)


def avg_rank(v):
    v = np.asarray(v)
    return np.array([(v < x).sum() + ((v == x).sum() + 1) / 2.0 for x in v]) / len(v)


avg, pl, mi = rd(O, 'raptor224_raw.csv'), rd(O, 'raptor224_raw_plain.csv'), rd(O, 'raptor224_raw_mirror.csv')
print('r224 avg vs float32 .5 (plain + mirror), max', np.abs(avg - (np.float32(.5) * (pl.astype(np.float32) + mi.astype(np.float32)))).max())
r224 = np.stack([np.argsort(np.argsort(avg[:, j], kind='stable'), kind='stable') / (len(ids) - 1) for j in range(12)], 1)
print('raptor224_sub vs her rankpct(avg), max', np.abs(r224 - rd(O, 'raptor224_sub.csv')).max())
raw, ours = rd(O, 'raptor_raw.csv'), rd(O, 'ours_sub.csv')
exp = np.stack([.5 * (.65 * avg_rank(raw[:, j]) + .35 * avg_rank(r224[:, j])) + .5 * avg_rank(ours[:, j]) for j in range(12)], 1)
sub = pd.read_csv(f'{O}/submission.csv', dtype={'StudyInstanceUID': str})
assert sub.StudyInstanceUID.tolist() == ids and list(sub.columns[1:]) == L
d0 = float(np.abs(exp - sub[L].values).max())
print('submission vs standalone recompute (V224 recipe, no re-rank), max', d0)
d = np.abs(sub[L].values - rd(F, 'submission.csv'))
print('vs V224-FLIP commit: cells differing', int((d > 1e-6).sum()), 'of', d.size, '; columns', [L[j] for j in range(12) if d[:, j].max() > 1e-6])
print('constant columns', [c for c in L if sub[c].nunique() == 1])
print('RECOMPUTE_FLRM', 'PASS' if d0 <= 1e-6 else 'FAIL')
