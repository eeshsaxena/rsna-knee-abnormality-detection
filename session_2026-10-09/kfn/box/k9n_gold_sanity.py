#!/usr/bin/env python3
"""k9n_gold_sanity.py (box, CPU, read-only; K9N 2026-10-08): label-order / pipeline sanity of the N stage on gold
studies. Reads the child npz of the A0n_gold12_* runs (CPU fp32 box child, 12 + 12 gold58 studies) and the gold labels;
prints per-label AUC (labels with both classes) and the macro. NOT a CV read: the author reused gold58 for selection and
the model saw all training data. A broken label order / reader / mirror would sit near .5; a sane pipeline far above.
  ~/miniconda3/envs/kmimic/bin/python ~/k9n_test/k9n/box/k9n_gold_sanity.py ~/k9n_test/runs/A0n_gold12_a_* ~/k9n_test/runs/A0n_gold12_b_*
"""
import sys, json, glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
lab = pd.read_csv(f'{sys.argv[0].rsplit("/k9n/", 1)[0]}/ids/gold_labels.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
rows = {}
for d in sys.argv[1:]:
    z = np.load(f'{d}/working/_n949_probs.npz')
    meta = json.loads(str(z['meta']))
    assert meta['labels'] == L and not meta['fallback'] and not meta['empty'], meta
    for s, p in zip(z['ids'], z['P']):
        rows[str(s)] = p.astype(np.float64)
ids = sorted(rows)
P = np.stack([rows[s] for s in ids]); Y = lab.loc[ids, L].to_numpy(np.float64)
aucs = {}
for j, t in enumerate(L):
    if 0 < Y[:, j].sum() < len(ids):
        aucs[t] = roc_auc_score(Y[:, j], P[:, j])
shuf = {}
rng = np.random.default_rng(0)
perm = rng.permutation(12)
for j, t in enumerate(L):                                           # label-order control: a wrong column mapping
    if 0 < Y[:, j].sum() < len(ids):
        shuf[t] = roc_auc_score(Y[:, j], P[:, perm[j]])
print(f'N on {len(ids)} gold studies (CPU fp32 child): per-label AUC ' + ', '.join(f'{k} {v:.3f}' for k, v in aucs.items()))
print(f'macro AUC over {len(aucs)} labels {np.mean(list(aucs.values())):.4f} | control (columns permuted) '
      f'{np.mean(list(shuf.values())):.4f}')
ok = np.mean(list(aucs.values())) >= 0.80
print(f'K9NGOLD {"PASS" if ok else "FAIL"} (sanity bar macro >= .80; NOT a CV read)')
