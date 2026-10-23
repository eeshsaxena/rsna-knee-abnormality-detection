# xlabel_oof.py: Synovitis <- Effusion borrowing on R7 OOF (4,407 train studies, 5-fold OOF) scored against report-derived labels.
# Second, larger ruler than gold58 (report regime, not expert regime). Read-only, local files only.
import numpy as np, pandas as pd
from scipy.stats import rankdata
R = 'C:/Users/eeshs/Downloads/rsna-knee/'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
oof = pd.read_csv(R + 'session_2026-10-18/next/g58/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
g = pd.read_csv(R + 'session_2026-10-18/next/g58/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
gold_ids = set(g.StudyInstanceUID)
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    p = y.sum(); q = len(y) - p
    r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
for name, f in [('v4_blend', '_labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv'),
                ('gpt56sol', '_labels/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv'),
                ('blendjev_sol', 'session_2026-10-13/raptor/vast/remote/labels_raptor_rsna_blendjev_sol.csv')]:
    try:
        lab = pd.read_csv(R + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    except Exception as e:
        print(name, 'ERR', e); continue
    cols = [c for c in L if c in lab.columns]
    if len(cols) < 12: print(name, 'cols', list(lab.columns)[:20]); continue
    ids = [i for i in oof.index if i in lab.index and i not in gold_ids]
    for t, h in [('Synovitis', 'Effusion')]:
        y = lab.loc[ids, t].values.astype(float)
        mask = ~np.isnan(y)
        # binarise soft labels at .5 (report regime); also report the prevalence
        yb = (y[mask] >= 0.5).astype(int)
        s = oof.loc[ids, 'p|' + t].values[mask]; e = oof.loc[ids, 'p|' + h].values[mask]
        base = auc(s, yb)
        row = ['%s n=%d prev %.3f  syn %.4f  eff->syn %.4f' % (name, mask.sum(), yb.mean(), base, auc(e, yb))]
        for a in (.1, .2, .3, .5):
            row.append('a%.1f %+.4f' % (a, auc(pct(s) + a * pct(e), yb) - base))
        print(' | '.join(row))
        # conditional: among effusion-positive reports, synovitis prevalence
        ye = (lab.loc[ids, 'Effusion'].values[mask] >= 0.5)
        print('   report P(syn|eff) %.3f  P(syn|no eff) %.3f' % (yb[ye].mean(), yb[~ye].mean()))
g58 = g[g.source == 'original_gold58']
ys, ye = g58['Synovitis'].values, g58['Effusion'].values
print('gold58 expert P(syn|eff) %.3f  P(syn|no eff) %.3f' % (ys[ye == 1].mean(), ys[ye == 0].mean()))
