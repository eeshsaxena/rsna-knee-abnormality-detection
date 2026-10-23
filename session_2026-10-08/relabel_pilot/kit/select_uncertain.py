#!/usr/bin/env python3
"""select_uncertain.py (gold_prep/relabel, 2026-10-06): the "active labelling" subset for the scale-up batch, i.e. the
train studies where an image-aware relabel can change a target label. Pure label-file arithmetic, no gold labels used
(gold74 rows are dropped: they stay held out). Rule fixed here, before the pilot:

  a study is UNCERTAIN if, for any of the 6 targets,
    (D) text and image teacher disagree: |mix50 - v4| > --dis (default .30), or
    (S) the report is silent on the finding (v4 == .25, the extractor's "not mentioned" level) AND the training label is
        ambiguous: mix50 in (--lo, --hi) (default .35-.65) -- only for the "silent" labels Effusion, Synovitis,
        Fracture, Contusion.

  python select_uncertain.py            -> uncertain_ids.csv (StudyInstanceUID, reasons) + counts per reason/label
"""
import os, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
H = os.path.expanduser
DS = '~/Downloads/rsna-knee/session_2026-10-05/kaggle_train/ds/rsna-knee-fleet-train'
T6 = ['Effusion', 'Synovitis', 'Fracture', 'Contusion', 'PF OA', 'Lateral OA']
SILENT_T = ['Effusion', 'Synovitis', 'Fracture', 'Contusion']


def first_existing(*c):
    for x in c:
        if x and os.path.exists(H(x)):
            return H(x)
    return H(c[0])


ap = argparse.ArgumentParser()
ap.add_argument('--mix50', default=None)
ap.add_argument('--v4', default=None)
ap.add_argument('--gold', default=None)
ap.add_argument('--dis', type=float, default=0.30)
ap.add_argument('--lo', type=float, default=0.35)
ap.add_argument('--hi', type=float, default=0.65)
ap.add_argument('--out', default=os.path.join(HERE, 'uncertain_ids.csv'))
A = ap.parse_args()
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
m = rd(first_existing(A.mix50, '~/rsna_train/labels_mix50_logit.csv', f'{DS}/labels_mix50_logit.csv'))
v = rd(first_existing(A.v4, '~/rsna_train/llm_labels_v4_blend.csv', f'{DS}/llm_labels_v4_blend.csv'))
g = rd(first_existing(A.gold, '~/rsna_train/gold74_eval_set.csv', f'{DS}/gold74_eval_set.csv'))
m = m.drop(index=g.index, errors='ignore'); v = v.loc[m.index]
D = (m[T6] - v[T6]).abs() > A.dis
S = pd.DataFrame(False, index=m.index, columns=T6)
for t in SILENT_T:
    S[t] = np.isclose(v[t], 0.25) & (m[t] > A.lo) & (m[t] < A.hi)
unc = D.any(axis=1) | S.any(axis=1)
why = [';'.join([f'D:{t}' for t in T6 if D.at[s, t]] + [f'S:{t}' for t in T6 if S.at[s, t]]) for s in m.index[unc]]
out = pd.DataFrame({'StudyInstanceUID': m.index[unc], 'reasons': why})
out.to_csv(A.out, index=False)
print(f'train studies (gold74 dropped) {len(m)} | uncertain {len(out)} ({len(out) / len(m):.0%}) -> {A.out}')
print('  D disagreement per label', {t: int(D[t].sum()) for t in T6}, '| any', int(D.any(axis=1).sum()))
print('  S silent+ambiguous per label', {t: int(S[t].sum()) for t in SILENT_T}, '| any', int(S.any(axis=1).sum()))
