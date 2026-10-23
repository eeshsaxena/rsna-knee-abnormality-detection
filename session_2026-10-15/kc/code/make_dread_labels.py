# make_dread_labels.py -- BOX_PLAN_1015 C5: the dread soft-label table in the fleet trainer's label format, for the KC
# dread all-data members (s2003, s2004, and s2002 if the S read is not positive). RSNA-only; box CPU.
#   python make_dread_labels.py --src ~/jump2/labels_in/r2_llm_soft.csv --ref ~/r7t_labels/real/labels_r7t_b.csv
#          --folds ~/rsna_train/folds5.csv --out ~/kc/labels/labels_dread_fleet.csv
# Asserts: src md5 96e6723f (dreaddevelopment/rsna-knee-labels); the same 12 label columns in the same order as T_b; every
# folds5 pool study (fold >= 0) covered; gold58 absent; values finite in [0, 1]. Output = StudyInstanceUID + the 12 labels
# (src rows unchanged, sorted by StudyInstanceUID); refuses to overwrite an existing output.
# KC-R1 (10-15 review fix): every gold74 id missing from the src (the 58 gold58 studies) is appended with NaN in all 12
# labels, after asserting it is held out (fold < 0 in --folds; gold74 is also the --exclude list of every KC line). The
# trainer never trains a held id and masks NaN targets; its gold74 eval reads gold_tab labels but requires the id in the
# label table (gold_ids = gold_tab ids in the store AND in lab), so without these rows a dread member's 'gold74' (log,
# KC-9 epochs/gold_*.csv, A6 G1 floor, G2) covered only 16 of 74 studies. Output = 4,407 rows.
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
gold_all = list(g.StudyInstanceUID)                                               # KC-R1
add = sorted(s for s in gold_all if s not in have)
not_held = [s for s in add if not (s in fo.index and fo[s] < 0)]
assert not not_held, f'{len(not_held)} gold74 ids to append are not held out (fold < 0) in {A.folds}'
pad = pd.DataFrame({'StudyInstanceUID': add, **{l: np.nan for l in L}})
out = pd.concat([src[['StudyInstanceUID'] + L], pad], ignore_index=True).sort_values('StudyInstanceUID')
assert out.StudyInstanceUID.is_unique and set(gold_all) <= set(out.StudyInstanceUID)
out.to_csv(A.out, index=False)
rec = dict(src=A.src, src_md5=m, rows=len(out), src_rows=len(src), pool=len(pool), pool_covered=len(pool & have),
           extra_rows=len(have - pool), gold58_labelled=0, gold74_nan_rows_added=len(add),
           gold74_in_table=int(sum(s in set(out.StudyInstanceUID) for s in gold_all)), out=A.out,
           out_md5=hashlib.md5(open(A.out, 'rb').read()).hexdigest(),
           pos_rate={l: round(float((out[l].dropna() >= .5).mean()), 4) for l in L})
json.dump(rec, open(A.out + '.receipt.json', 'w'), indent=1)
print(json.dumps(rec))
