"""Reviewer's independent recompute (scipy rankdata, no builder code): V224 stage 3 from the stage csvs, then the HF blend."""
import sys, numpy as np, pandas as pd
from scipy.stats import rankdata
out, wf = sys.argv[1], float(sys.argv[2])
S = pd.read_csv('_data/sample_submission.csv', dtype=str); ids = S.StudyInstanceUID.tolist(); L = list(S.columns[1:])
rd = lambda f: pd.read_csv(f'{out}/{f}', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values.astype(np.float64)
pct = lambda v: rankdata(v, method='average') / len(v)
raw, r224, ours = rd('raptor_raw.csv'), rd('raptor224_sub.csv'), rd('ours_sub.csv')
v224 = np.stack([.5 * (.65 * pct(raw[:, j]) + .35 * pct(r224[:, j])) + .5 * pct(ours[:, j]) for j in range(12)], 1).astype(np.float32)
sv = rd('submission_V224.csv')
print('V224 recompute vs submission_V224.csv max', np.abs(v224 - sv).max())
F = rd('submission_fleetsolo.csv')
z = np.load(f'{out}/_fleet_0.npz'); z1 = np.load(f'{out}/_fleet_1.npz')
P = np.concatenate([z['P'], z1['P']], 1).astype(np.float64); sids = np.concatenate([z['sids'], z1['sids']]).astype(str)
Fm = pd.DataFrame(P.mean(0), index=sids, columns=L).loc[ids].values
print('fleet npz mean vs fleetsolo max', np.abs(Fm - F).max(), 'members', P.shape[0])
fin = np.stack([(1 - wf) * pct(sv[:, j]) + wf * pct(Fm[:, j]) for j in range(12)], 1)
sub = pd.read_csv(f'{out}/submission.csv', dtype={'StudyInstanceUID': str})
assert sub.StudyInstanceUID.tolist() == ids and list(sub.columns[1:]) == L
print('final recompute vs submission.csv max', np.abs(fin - sub[L].values).max())
print('constant columns', [c for c in L if sub[c].nunique() == 1])
if 'raptor224_raw.csv' in __import__('os').listdir(out):
    a = rd('raptor224_raw.csv'); p = rd('raptor224_raw_plain.csv'); m = rd('raptor224_raw_mirror.csv')
    print('r224 avg == 0.5(p+m) max', np.abs(a - (0.5 * (p.astype(np.float32) + m.astype(np.float32))).astype(np.float64)).max(),
          '; r224 sub == pct(avg) max', np.abs(np.stack([pct(a[:, j]) for j in range(12)], 1) - r224).max())
