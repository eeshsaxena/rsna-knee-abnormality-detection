#!/usr/bin/env python3
"""kfn_gold_sanity.py (box, CPU, read-only; kfn 2026-10-09 = k9n_gold_sanity.py with kfn paths): label-order / pipeline
sanity of the N path on gold studies from the A0no_gold12 run dir(s) (CPU fp32 child). Per-label AUC + macro, the N-only
submission.csv AUC (must equal N's: rank-invariant), and a control with the columns permuted. NOT a CV read (the author
reused gold58 for selection; the model saw all training data).
  ~/miniconda3/envs/kmimic/bin/python ~/kfn_test/kfn/box/kfn_gold_sanity.py ~/kfn_test/runs/A0no_gold12_<HHMM>
"""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
lab = pd.read_csv(os.path.expanduser('~/kfn_test/ids/gold_labels.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
rows, subs = {}, {}
for d in sys.argv[1:]:
    z = np.load(f'{d}/working/_n949_probs.npz')
    meta = json.loads(str(z['meta']))
    assert meta['labels'] == L and not meta['fallback'] and not meta['empty'], meta
    for s, p in zip(z['ids'], z['P']):
        rows[str(s)] = p.astype(np.float64)
    sb = pd.read_csv(f'{d}/working/submission.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    for s in sb.index:
        subs[s] = sb.loc[s, L].to_numpy(np.float64)
ids = sorted(rows)
P = np.stack([rows[s] for s in ids]); S = np.stack([subs[s] for s in ids]); Y = lab.loc[ids, L].to_numpy(np.float64)
aucs, auc_sub, shuf = {}, {}, {}
perm = np.random.default_rng(0).permutation(12)
for j, t in enumerate(L):
    if 0 < Y[:, j].sum() < len(ids):
        aucs[t] = roc_auc_score(Y[:, j], P[:, j]); auc_sub[t] = roc_auc_score(Y[:, j], S[:, j])
        shuf[t] = roc_auc_score(Y[:, j], P[:, perm[j]])
print(f'N on {len(ids)} gold studies (CPU fp32 child): per-label AUC ' + ', '.join(f'{k} {v:.3f}' for k, v in aucs.items()))
m, ms, mc = np.mean(list(aucs.values())), np.mean(list(auc_sub.values())), np.mean(list(shuf.values()))
print(f'macro AUC over {len(aucs)} labels: N probs {m:.4f} | the N-only submission.csv {ms:.4f} (must be equal) | '
      f'control (columns permuted) {mc:.4f}')
ok = m >= 0.80 and abs(m - ms) < 1e-12
print(f'KFNGOLD {"PASS" if ok else "FAIL"} (sanity bar macro >= .80 and submission AUC == N AUC; NOT a CV read)')
