# pick_store.py (gold_prep/r50, CPU): tiny symlinked store for the R50 probe dry run.
# 16 train (fold != 0), 6 val (fold 0; 3 with DICOMs in the kmimic mirror, label variety vs v4), 3 gold74 (2 gold58).
import os, numpy as np, pandas as pd
H = os.path.expanduser('~'); R = f'{H}/rsna_train'
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
lab = pd.read_csv(f'{R}/labels_mix50_logit.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
v4 = pd.read_csv(f'{R}/llm_labels_v4_blend.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
fo = pd.read_csv(f'{R}/folds5.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
g = pd.read_csv(f'{R}/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
mir = set(os.listdir(f'{H}/kmimic/kaggle/input/rsna-knee-abnormality-detection/train_series'))
held = set(fo.index[fo < 0]) | set(g.index)
ok = [s for s in lab.index if s in fo.index and s not in held and os.path.exists(f'{H}/fleet_store/{s}.npz')]
rng = np.random.RandomState(7)
va_m = [s for s in ok if fo[s] == 0 and s in mir]
va_o = [s for s in ok if fo[s] == 0 and s not in mir]
best = None
for _ in range(400):
    va = list(rng.choice(va_m, 3, replace=False)) + list(rng.choice(va_o, 3, replace=False))
    Y = v4.loc[va, T].to_numpy() >= 0.5
    n = int(((Y.mean(0) > 0) & (Y.mean(0) < 1)).sum())
    if best is None or n > best[0]:
        best = (n, va)
nvar, va = best
tr = list(rng.choice([s for s in ok if fo[s] != 0], 16, replace=False))
g58 = [s for s in g.index if g.loc[s, 'source'] == 'original_gold58']
gold = list(rng.choice(g58, 2, replace=False)) + [s for s in g.index if g.loc[s, 'source'] != 'original_gold58'][:1]
for s in tr + va + gold:
    d = f'{H}/gold_prep/r50/store/{s}.npz'
    if not os.path.lexists(d):
        os.symlink(f'{H}/fleet_store/{s}.npz', d)
pd.DataFrame({'StudyInstanceUID': tr, 'role': 'train'}).to_csv(f'{H}/gold_prep/r50/pick_train.csv', index=False)
pd.DataFrame({'StudyInstanceUID': va, 'role': 'val', 'in_mirror': [s in mir for s in va]}).to_csv(f'{H}/gold_prep/r50/pick_val.csv', index=False)
pd.DataFrame({'StudyInstanceUID': gold, 'role': 'gold', 'source': [g.loc[s, 'source'] for s in gold]}).to_csv(f'{H}/gold_prep/r50/pick_gold.csv', index=False)
print('train', len(tr), '| val', len(va), 'labels with both classes vs v4:', nvar, '| val in mirror', [s[-6:] for s in va if s in mir],
      '| gold', [(s[-6:], g.loc[s, 'source']) for s in gold], '| store files', len(os.listdir(f'{H}/gold_prep/r50/store')))
