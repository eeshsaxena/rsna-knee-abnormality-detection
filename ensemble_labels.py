#!/usr/bin/env python3
"""Soft rule x Qwen label fusion (the +0.06 lever from the label-pipeline analysis).
Per-label weighted average p = w_r*rule + w_q*qwen (both binary -> 3-level soft target).
Gold-58 rows are identical in both files (kept as gold), so they pass through unchanged.
Output: _data/pseudo_labels_v1.csv (soft targets for the image model)."""
import pandas as pd, numpy as np

RULE = r"C:\Users\eeshs\Downloads\rsna-knee\_data\pseudo_labels_rulev0.csv"
QWEN = r"C:\Users\eeshs\Downloads\rsna-knee\_out_qwen_full\pseudo_labels_qwen.csv"
TRAIN = r"C:\Users\eeshs\Downloads\rsna-knee\_data\train.csv"
OUT = r"C:\Users\eeshs\Downloads\rsna-knee\_data\pseudo_labels_v1.csv"

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
          'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

# per-label (w_rule, w_qwen) from the label-pipeline analysis
W = {
    'ACL': (.5, .5), 'MCL': (.5, .5), 'Medial Meniscus': (.5, .5),
    'Lateral Meniscus': (.5, .5), 'Fracture': (.5, .5),
    'Lateral OA': (.35, .65), 'Effusion': (.35, .65), 'Synovitis': (.35, .65),
    "Baker's": (.35, .65), 'Contusion': (.35, .65),
    'Medial OA': (.2, .8),
    'PF OA': (.65, .35),   # Qwen ~ random here; rule better
}

r = pd.read_csv(RULE).set_index('StudyInstanceUID')
q = pd.read_csv(QWEN).set_index('StudyInstanceUID')
tr = pd.read_csv(TRAIN).set_index('StudyInstanceUID')
ids = r.index
q = q.reindex(ids); tr = tr.reindex(ids)
gold_mask = tr['ACL'].notna().values
print(f"studies {len(ids)} | gold {gold_mask.sum()} | rule cols ok {all(l in r for l in LABELS)} "
      f"| qwen cols ok {all(l in q for l in LABELS)}")

out = pd.DataFrame(index=ids)
for lab in LABELS:
    wr, wq = W[lab]
    # clip extractor outputs to [0,1] (Qwen occasionally emitted 2, violating the 0/1 spec)
    rv = np.clip(r[lab].values.astype(float), 0, 1)
    qv = np.clip(q[lab].values.astype(float), 0, 1)
    soft = np.clip(wr * rv + wq * qv, 0, 1)
    # gold rows: use the true gold label exactly
    out[lab] = np.where(gold_mask, tr[lab].values, soft)
out = out.reset_index()
out.to_csv(OUT, index=False)

# report: pseudo-label positive rate (>0.5) on the 4349, and value spread
rep = out[~gold_mask]
print("\nsoft-label mean per label (4349 report-only):")
for lab in LABELS:
    vals = rep[lab].values
    print(f"  {lab:18s} mean={vals.mean():.3f}  frac>0.5={np.mean(vals>0.5):.3f}  "
          f"distinct={sorted(set(np.round(vals,2)))[:5]}")
print(f"\nwrote {OUT}  shape {out.shape}")
