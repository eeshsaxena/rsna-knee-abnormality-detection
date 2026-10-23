# make_tiny.py -- tiny label subset for the end-to-end trainer parity runs: 16 TRAIN studies (fold in 0,1,3,4) + 8 VAL
# studies (fold 2), none held out (fold < 0 or gold74), all in the store. Rows copied verbatim from labels_mix50_logit.csv.
import os, pandas as pd, numpy as np
H = os.path.expanduser('~')
lab = pd.read_csv(f'{H}/rsna_train/labels_mix50_logit.csv', dtype={'StudyInstanceUID': str})
folds = pd.read_csv(f'{H}/rsna_train/folds5.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
excl = set(pd.read_csv(f'{H}/rsna_train/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID)
ev = pd.read_csv(f'{H}/rsna_train/llm_labels_v4_blend.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
ok = lab[lab.StudyInstanceUID.map(lambda s: s in folds.index and folds[s] >= 0 and s not in excl
                                  and os.path.exists(f'{H}/fleet_store/{s}.npz'))]
f = ok.StudyInstanceUID.map(folds)
tr = ok[f != 2].sort_values('StudyInstanceUID').iloc[:16]
va_pool = ok[(f == 2) & ok.StudyInstanceUID.isin(ev.index)].sort_values('StudyInstanceUID')
# pick 8 val studies so that most targets have both classes under the eval labels (AUC defined)
T = [c for c in lab.columns if c != 'StudyInstanceUID']
best, rng = None, np.random.RandomState(0)
for _ in range(400):
    cand = va_pool.iloc[rng.choice(len(va_pool), 8, replace=False)]
    Y = ev.loc[cand.StudyInstanceUID, T].to_numpy() >= 0.5
    n = int(((Y.mean(0) > 0) & (Y.mean(0) < 1)).sum())
    if best is None or n > best[0]:
        best = (n, cand)
va = best[1]
out = pd.concat([tr, va])
os.makedirs(f'{H}/fast_build/tiny', exist_ok=True)
out.to_csv(f'{H}/fast_build/tiny/labels_tiny.csv', index=False)
print('tiny labels', len(out), '| train', len(tr), '| val', len(va), '| val targets with both classes', best[0], '/', len(T))
