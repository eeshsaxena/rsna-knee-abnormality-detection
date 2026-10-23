# g3_regime_strata.py (10-20 night, Track 3, Claude). RESEARCH ONLY, local CPU, competition data only.
# Where do expert (gold58) and report labels disagree for LM / Fracture? Split by expert acute-trauma context
# (ACL or Contusion or Fracture or MCL expert-positive) and by expert degenerative context (any OA label positive).
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G, R, labels
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58.set_index('StudyInstanceUID')[L]
rep = {s: labels(s).reindex(ids) for s in ('v2', 'v4', 'sol56')}
tr = (Y[['ACL', 'Contusion', 'MCL']].max(1) == 1).values
trF = (Y[['ACL', 'Contusion', 'MCL', 'Fracture']].max(1) == 1).values
deg = (Y[['Medial OA', 'Lateral OA', 'PF OA']].max(1) == 1).values
for lab, strata in (('Lateral Meniscus', {'trauma(ACL|Cont|MCL)': tr, 'no trauma': ~tr, 'degen(any OA)': deg, 'no OA': ~deg}),
                    ('Fracture', {'trauma(ACL|Cont|MCL)': tr, 'no trauma': ~tr, 'degen(any OA)': deg, 'no OA': ~deg})):
    y = Y[lab].values
    print('==', lab)
    for sn, m in strata.items():
        s = '  %-22s n %2d expert prev %.2f |' % (sn, m.sum(), y[m].mean())
        for src, d in rep.items():
            v = d[lab].values; ok = m & ~np.isnan(v)
            agree = (v[ok] == y[ok]).mean() if ok.any() else np.nan
            fn = ((v == 0) & (y == 1) & ok).sum(); fp = ((v == 1) & (y == 0) & ok).sum()
            s += ' %s agree %.2f (rep-miss %d, rep-extra %d, known %d) |' % (src, agree, fn, fp, ok.sum())
        print(s)
# same strata, but trauma defined from the REPORT side (v4 ACL/Contusion/MCL) to avoid conditioning on expert labels
v4 = rep['v4']; trR = (v4[['ACL', 'Contusion', 'MCL']].max(1) == 1).values
y = Y['Lateral Meniscus'].values; v = v4['Lateral Meniscus'].values
for nm, m in (('report-trauma', trR), ('report-no-trauma', ~trR)):
    print('LM by report-side trauma %-17s n %2d expert prev %.2f report prev %.2f agree %.2f' % (nm, m.sum(), y[m].mean(), v[m].mean(), (v[m] == y[m]).mean()))
# train-wide report coupling for context (report v2 YES/NO): P(LM | ACL) vs P(LM | no ACL)
lab = labels('v2'); a = lab['ACL']; l = lab['Lateral Meniscus']; ok = a.notna() & l.notna()
print('train report v2: P(LM tear | ACL tear) %.3f (n %d) vs P(LM | no ACL) %.3f (n %d)' % (l[ok & (a == 1)].mean(), (ok & (a == 1)).sum(), l[ok & (a == 0)].mean(), (ok & (a == 0)).sum()))
yA = Y['ACL'].values
print('gold58 expert: P(LM | ACL) %.3f (n %d) vs P(LM | no ACL) %.3f (n %d)' % (y[yA == 1].mean(), (yA == 1).sum(), y[yA == 0].mean(), (yA == 0).sum()))
g74 = g.set_index('StudyInstanceUID')[L].dropna(subset=['ACL', 'Lateral Meniscus'])
print('gold74: P(LM | ACL) %.3f (n %d) vs P(LM | no ACL) %.3f (n %d)' % (g74.loc[g74.ACL == 1, 'Lateral Meniscus'].mean(), (g74.ACL == 1).sum(), g74.loc[g74.ACL == 0, 'Lateral Meniscus'].mean(), (g74.ACL == 0).sum()))
