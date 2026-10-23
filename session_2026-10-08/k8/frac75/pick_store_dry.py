# pick_store_dry.py (k8/frac75, CPU, seconds): tiny SYMLINKED store for the frac75 argv dry runs (same idea as
# gold_prep/r50/pick_store.py). 16 train studies (folds 2-4), 6 fold-0 val + 6 fold-1 val (picked for label variety vs
# steven_v4, so most of the 12 val AUCs are defined in both the fold-0 and fold-1 runs), 3 gold74 (2 gold58 + 1 extra).
# Every pick is a non-held study present in ~/fleet_store and in labels_frac75_logit.csv. Writes ~/novel/frac75/dry only.
import os, numpy as np, pandas as pd
H = os.path.expanduser('~'); R = f'{H}/rsna_train'; D = f'{H}/novel/frac75/dry'
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
lab = rd(f'{H}/novel/frac75/labels_frac75_logit.csv'); v4 = rd(f'{R}/llm_labels_v4_blend.csv')
fo = rd(f'{R}/folds5.csv')['fold']; g = rd(f'{R}/gold74_eval_set.csv')
held = set(fo.index[fo < 0]) | set(g.index)
ok = [s for s in lab.index if s in fo.index and s not in held and os.path.exists(f'{H}/fleet_store/{s}.npz')]
rng = np.random.RandomState(75)
def pick_val(f):
    pool = [s for s in ok if fo[s] == f]; best = None
    for _ in range(400):
        va = list(rng.choice(pool, 6, replace=False))
        n = int(((lambda Y: (Y.mean(0) > 0) & (Y.mean(0) < 1))(v4.loc[va, T].to_numpy() >= .5)).sum())
        if best is None or n > best[0]: best = (n, va)
    return best
n0, va0 = pick_val(0); n1, va1 = pick_val(1)
tr = list(rng.choice([s for s in ok if fo[s] in (2, 3, 4)], 16, replace=False))
g58 = [s for s in g.index if g.loc[s, 'source'] == 'original_gold58']
gold = list(rng.choice(g58, 2, replace=False)) + [s for s in g.index if g.loc[s, 'source'] != 'original_gold58'][:1]
os.makedirs(f'{D}/store', exist_ok=True)
for s in tr + va0 + va1 + gold:
    d = f'{D}/store/{s}.npz'
    if not os.path.lexists(d):
        os.symlink(f'{H}/fleet_store/{s}.npz', d)
rows = [(s, 'train', fo[s]) for s in tr] + [(s, 'val0', 0) for s in va0] + [(s, 'val1', 1) for s in va1] + [(s, 'gold', fo.get(s, -9)) for s in gold]
pd.DataFrame(rows, columns=['StudyInstanceUID', 'role', 'fold']).to_csv(f'{D}/picks.csv', index=False)
fr = lab.loc[tr + va0 + va1, 'Fracture']; mx = rd(f'{R}/labels_mix50_logit.csv').loc[tr + va0 + va1, 'Fracture']
print(f'train {len(tr)} | val f0 {len(va0)} ({n0}/12 labels with both classes vs v4) | val f1 {len(va1)} ({n1}/12) | gold {len(gold)} '
      f'| store links {len(os.listdir(D + "/store"))} | Fracture frac75-mix50 on picks: mean {float((fr - mx).mean()):+.4f} max|d| {float((fr - mx).abs().max()):.4f}')
