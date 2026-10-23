# side_pass_check.py (10-15, Claude subagent, kcstage): run the side kernel's own study() (kc_side_kernel.py sha 2979c191, as the
# KC driver does: fork Pool(P), one study per task) on a fake test root of TRAIN studies and compare every column with the side
# kernel's Kaggle output for the same studies (~/kc_store/side_train.csv = kcrop/out_kc_side_v1/kc_side_train.csv, sha ecc5cd26).
# Also times the pass (s / study wall at P procs). Box CPU only.
#   python side_pass_check.py --root FAKEROOT --code CODE --out side.csv [--procs 4]
import os, sys, json, time, argparse, importlib.util
import multiprocessing as mp
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True); ap.add_argument('--code', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--procs', type=int, default=4)
ap.add_argument('--ref', default=os.path.expanduser('~/kc_store/side_train.csv'))
A = ap.parse_args()
os.environ['KC_ROOT'] = A.root
os.environ['KC_OUT'] = os.path.dirname(os.path.abspath(A.out))
spec = importlib.util.spec_from_file_location('kc_side_kernel', os.path.join(A.code, 'kc_side_kernel.py'))
KS = importlib.util.module_from_spec(spec); sys.modules['kc_side_kernel'] = KS; spec.loader.exec_module(KS)
ser = pd.read_csv(A.root + '/test_series.csv')
ser['StudyInstanceUID'] = ser['StudyInstanceUID'].astype(str); ser['SeriesInstanceUID'] = ser['SeriesInstanceUID'].astype(str)
SER = {k: v.to_dict('records') for k, v in ser.groupby('StudyInstanceUID')}
jobs = [(s, SER[s], A.root + '/test_series') for s in sorted(SER)]
t0 = time.time()
with mp.get_context('fork').Pool(A.procs) as pool:
    res = list(pool.imap_unordered(KS.study, jobs, chunksize=1))
wall = time.time() - t0
df = pd.DataFrame(res).sort_values('StudyInstanceUID').reset_index(drop=True)
first = ['StudyInstanceUID', 'mean_x', 'n', 'sag_nx', 'tag', 'side', 'rule']
df = df[first + [c for c in df.columns if c not in first]]
df.to_csv(A.out, index=False)
mine = pd.read_csv(A.out, dtype={'StudyInstanceUID': str, 'tag': str}).set_index('StudyInstanceUID')
ref = pd.read_csv(A.ref, dtype={'StudyInstanceUID': str, 'tag': str}).set_index('StudyInstanceUID').loc[mine.index]
diff = {}
for c in mine.columns:
    if c not in ref.columns:
        diff[c] = 'missing in ref'; continue
    a, b = mine[c], ref[c]
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        x, y = a.to_numpy(np.float64), b.to_numpy(np.float64)
        bad = ~((np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= 1e-9 * np.maximum(1, np.abs(y))))
    else:
        bad = (a.fillna('<NA>').astype(str) != b.fillna('<NA>').astype(str)).to_numpy()
    if bad.any():
        diff[c] = int(bad.sum())
cols_ref_only = [c for c in ref.columns if c not in mine.columns]
r = {'n': len(df), 'procs': A.procs, 'wall_s': round(wall, 2), 's_per_study_wall': round(wall / max(len(df), 1), 4),
     'columns': len(mine.columns), 'diff_columns': diff, 'ref_only_columns': cols_ref_only,
     'files_read': int(df.nread.sum()), 'serve': df.serve.value_counts().to_dict(), 'rules': df.rule.value_counts().to_dict(),
     'PASS': not diff and not cols_ref_only}
print('SIDE_PASS_CHECK ' + json.dumps(r, default=str))
