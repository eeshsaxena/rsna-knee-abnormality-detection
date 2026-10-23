# make_dread_labels.py -- BOX_PLAN_1015 C5: the dread soft-label table in the fleet trainer's label format, for the KC
# dread all-data members (s2003, s2004, and s2002 if the S read is not positive). RSNA-only; box CPU.
#   python make_dread_labels.py --src ~/jump2/labels_in/r2_llm_soft.csv --ref ~/r7t_labels/real/labels_r7t_b.csv
#          --folds ~/rsna_train/folds5.csv --out ~/kc/labels/labels_dread_fleet.csv
# Asserts: src md5 96e6723f (dreaddevelopment/rsna-knee-labels); the same 12 label columns in the same order as T_b; every
# folds5 pool study (fold >= 0) covered; gold58 absent; values finite in [0, 1]. Output = StudyInstanceUID + the 12 labels
# (src rows unchanged, sorted by StudyInstanceUID); refuses to overwrite an existing output.
import os, sys, hashlib, argparse, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True); ap.add_argument('--ref', required=True)
ap.add_argument('--folds', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--gold', default=os.path.expanduser('~/rsna_train/gold74_eval_set.csv'))
A = ap.parse_args()
m = hashlib.md5(open(A.src, 'rb').read()).hexdigest()
assert m.startswith('96e6723f'), f'src md5 {m} != 96e6723f'
assert not os.path.exists(A.out), f'{A.out} exists (never overwritten)'
src = pd.read_csv(A.src, dtype={'StudyInstanceUID': str})
ref = pd.read_csv(A.ref, dtype={'StudyInstanceUID': str}, nrows=5)
L = [c for c in ref.columns if c != 'StudyInstanceUID']
assert len(L) == 12 and [c for c in src.columns if c != 'StudyInstanceUID'][:12] == L, (list(src.columns), L)
fo = pd.read_csv(A.folds, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
pool = set(fo.index[fo >= 0])
g = pd.read_csv(A.gold, dtype={'StudyInstanceUID': str})
g58 = set(g.StudyInstanceUID[g.source == 'original_gold58'])
assert src.StudyInstanceUID.is_unique
have = set(src.StudyInstanceUID)
miss = pool - have
assert not miss, f'{len(miss)} folds5 pool studies missing from the dread table'
assert not (have & g58), f'{len(have & g58)} gold58 studies present in the dread table'
V = src[L].to_numpy(np.float64)
assert np.isfinite(V).all() and V.min() >= 0 and V.max() <= 1, (np.nanmin(V), np.nanmax(V))
out = src[['StudyInstanceUID'] + L].sort_values('StudyInstanceUID')
out.to_csv(A.out, index=False)
rec = dict(src=A.src, src_md5=m, rows=len(out), pool=len(pool), pool_covered=len(pool & have), extra_rows=len(have - pool),
           gold58_present=0, out=A.out, out_md5=hashlib.md5(open(A.out, 'rb').read()).hexdigest(),
           pos_rate={l: round(float((out[l] >= .5).mean()), 4) for l in L})
json.dump(rec, open(A.out + '.receipt.json', 'w'), indent=1)
print(json.dumps(rec))
