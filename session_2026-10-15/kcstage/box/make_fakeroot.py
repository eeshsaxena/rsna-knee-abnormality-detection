# make_fakeroot.py (10-15, Claude subagent, kcstage): a competition-shaped TEST root on the box made of RSNA TRAIN studies
# (symlinks into ~/eda_data, read only), for CPU runs of the KC driver / serve path. RSNA-only.
#   python make_fakeroot.py --out ~/kcstage_1015/fakeroot_n24 --n 24 [--weak 3 --absent 2] [--seed 1015] [--ids a,b,c]
# Picks first the requested number of weak-zone studies and SagPD-absent studies (from the side kernel's train table
# ~/kc_store/side_train.csv), then random ones; only studies present in ~/eda_data/.../train_series.
import os, json, argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=24)
ap.add_argument('--weak', type=int, default=3)
ap.add_argument('--absent', type=int, default=2)
ap.add_argument('--seed', type=int, default=1015)
ap.add_argument('--ids', default='')
ap.add_argument('--src', default=os.path.expanduser('~/eda_data/rsna-knee-abnormality-detection'))
ap.add_argument('--side', default=os.path.expanduser('~/kc_store/side_train.csv'))
A = ap.parse_args()
assert not os.path.exists(A.out), A.out
have = set(os.listdir(A.src + '/train_series'))
side = pd.read_csv(A.side, dtype={'StudyInstanceUID': str, 'tag': str})
side = side[side.StudyInstanceUID.isin(have)].sort_values('StudyInstanceUID')
rng = np.random.default_rng(A.seed)
if A.ids:
    pick = A.ids.split(',')
else:
    weak = side[side.weak == 1].StudyInstanceUID.tolist()
    absent = side[(side.weak == 0) & (side.s1_n == 0)].StudyInstanceUID.tolist()
    pick = list(rng.permutation(weak)[:A.weak]) + list(rng.permutation(absent)[:A.absent])
    rest = [s for s in rng.permutation(side.StudyInstanceUID.tolist()) if s not in pick]
    pick += rest[:A.n - len(pick)]
os.makedirs(A.out + '/test_series')
ser = pd.read_csv(A.src + '/train_series.csv', dtype=str)
ser[ser.StudyInstanceUID.isin(pick)].to_csv(A.out + '/test_series.csv', index=False)
pd.DataFrame({'StudyInstanceUID': sorted(pick)}).to_csv(A.out + '/test.csv', index=False)
pd.read_csv(A.src + '/sample_submission.csv', nrows=0).to_csv(A.out + '/sample_submission.csv', index=False)
for s in pick:
    os.symlink(A.src + '/train_series/' + s, A.out + '/test_series/' + s)
info = side[side.StudyInstanceUID.isin(pick)][['StudyInstanceUID', 'side', 'rule', 'weak', 'geo_ok', 'serve', 's1_n']]
json.dump({'n': len(pick), 'weak': int(info.weak.sum()), 'sagpd_absent': int((info.s1_n == 0).sum()), 'ids': sorted(pick)},
          open(A.out + '/fakeroot.json', 'w'), indent=1)
print('FAKEROOT', A.out, len(pick), 'weak', int(info.weak.sum()), 'sagpd_absent', int((info.s1_n == 0).sum()))
