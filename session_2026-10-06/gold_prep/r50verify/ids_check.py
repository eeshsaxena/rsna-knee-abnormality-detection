# ids_check.py (gold_prep/r50verify, CPU, pandas only): replays fleet_train.py lines 129-151 with the ARMED argv
# (real ~/fleet_store, --min_cover default .99, fold 0, --exclude gold74) and checks gold leakage + val alignment.
import os, glob, numpy as np, pandas as pd
H = os.path.expanduser('~'); R = f'{H}/rsna_train'; STORE = f'{H}/fleet_store'
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
lab = pd.read_csv(f'{R}/labels_mix50_logit.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
folds = pd.read_csv(f'{R}/folds5.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
g74 = pd.read_csv(f'{R}/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
excl = set(g74.StudyInstanceUID)
held = set(folds.index[folds < 0]) | excl
have = {s for s in lab.index if os.path.exists(f'{STORE}/{s}.npz')}
ids = [s for s in lab.index if s in folds.index and s in have]
tr = [s for s in ids if s not in held and folds[s] != 0]
va = [s for s in ids if s not in held and folds[s] == 0]
n_want = sum(1 for s in lab.index if s in folds.index and s not in held)
print(f'labels {len(lab)} | store has {len(have)} | held {len(held)} (fold<0 {int((folds < 0).sum())}, gold74 {len(excl)}) '
      f'| train {len(tr)} | val {len(va)} | cover {(len(tr) + len(va)) / n_want:.4f} vs min_cover .99 -> '
      f'{"PASS" if len(tr) + len(va) >= 0.99 * n_want else "FAIL"}')
print('assert disjoint train/val vs held:', not ((set(tr) | set(va)) & held), '| train/val overlap:', len(set(tr) & set(va)))
print('it/epoch (bs 2, drop_last):', len(tr) // 2, '| 28.9 X check:', round(len(tr) // 2 / 60, 2))
# gold leakage: gold58 (kmimic ids) and gold74 never in train/val
g58 = set(pd.read_csv(f'{H}/kmimic/ids/gold58.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID)
src58 = set(g74.StudyInstanceUID[g74.source == 'original_gold58'])
print('gold58 file n', len(g58), '| gold74 source==original_gold58 n', len(src58), '| equal', g58 == src58,
      '| gold58 in train', len(g58 & set(tr)), 'in val', len(g58 & set(va)), '| gold74 in train', len(excl & set(tr)),
      'in val', len(excl & set(va)), '| gold74 dup ids', int(g74.StudyInstanceUID.duplicated().sum()))
print('gold74 sources', g74.source.value_counts().to_dict(), '| gold74 in folds5', len(excl & set(folds.index)),
      '| their folds', folds.reindex(sorted(excl)).value_counts(dropna=False).to_dict())
# label sanity: mix50 values are probabilities (BCE-with-logits targets must be in [0, 1])
v = lab[T].to_numpy(np.float64)
print('mix50 min', np.nanmin(v), 'max', np.nanmax(v), '| NaN cells', int(np.isnan(v).sum()), '| cols', list(lab.columns) == T)
v4 = pd.read_csv(f'{R}/llm_labels_v4_blend.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
print('eval_labels covers val:', set(va) <= set(v4.index), '| v4 cols ok', all(t in v4.columns for t in T))
# val alignment vs the existing fold-0 OOFs (J1 TTR224 mix50, TTR288 in progress)
for p in [f'{H}/fleet_runs/f0_s1043_ttr_mix50/oof_f0_s1043_raw.csv', f'{H}/fleet_runs/f0_s1044_ttr288mix50/oof_f0_s1044_raw.csv']:
    if os.path.exists(p):
        o = pd.read_csv(p, dtype={'StudyInstanceUID': str})
        print(os.path.basename(os.path.dirname(p)), '| OOF rows', len(o), '| same set as R50 va', set(o.StudyInstanceUID) == set(va),
              '| same order', o.StudyInstanceUID.tolist() == va)
# decode probe = sorted(ids)[:4] (ids include held studies): must equal the 4 shipped parity npz
par = sorted(os.path.basename(f)[len('parity_'):-4] for f in glob.glob(f'{H}/gold_prep/r50/k5code/parity_*.npz'))
print('decode probe ids == shipped parity ids:', sorted(ids)[:4] == par)
# gold eval set (logged only): gold_ids = gold74 in store and in labels
gi = [s for s in g74.StudyInstanceUID if s in have and s in lab.index]
print('gold eval ids', len(gi), '| gold58 among them', len([s for s in gi if s in src58]))
