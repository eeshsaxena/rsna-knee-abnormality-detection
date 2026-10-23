# xlabel_gap.py: where does the expert label coupling (gold58) differ most from the report-label coupling (v4_blend >= .5)?
# Lift(t|h) = P(t=1 | h=1) - P(t=1 | h=0). Mechanism filter for cross-label borrowing; then gold58 c4 test for the top gaps.
import numpy as np, pandas as pd
R = 'C:/Users/eeshs/Downloads/rsna-knee/'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
g = pd.read_csv(R + 'session_2026-10-18/next/g58/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
g58 = g[g.source == 'original_gold58']; G = g58[L].values.astype(int)
lab = pd.read_csv(R + '_labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
lab = lab[~lab.index.isin(set(g.StudyInstanceUID))]
Rp = (lab[L].values >= 0.5).astype(int)
def lift(Y, t, h):
    a = Y[Y[:, h] == 1, t]; b = Y[Y[:, h] == 0, t]
    return (a.mean() if len(a) else np.nan) - (b.mean() if len(b) else np.nan), len(a)
rows = []
for t in range(12):
    for h in range(12):
        if t == h: continue
        le, nh = lift(G, t, h); lr, _ = lift(Rp, t, h)
        rows.append((le - lr, L[t], L[h], le, lr, nh, G[:, t].sum()))
rows.sort(key=lambda r: -r[0])
print('target <- helper | expert lift | report lift | gap | n helper+ (gold) | n target+ (gold)')
for r in rows[:12]:
    print('%-16s <- %-16s  %+.2f  %+.2f  gap %+.2f  nh %d  nt %d' % (r[1], r[2], r[3], r[4], r[0], r[5], r[6]))
print('prevalence gold vs report:', {l: (round(G[:, j].mean(), 2), round(Rp[:, j].mean(), 2)) for j, l in enumerate(L)})
