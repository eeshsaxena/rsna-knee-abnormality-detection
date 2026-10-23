# xlabel2.py: WHY the cross-label stacker gains in CV vs report labels but not on gold58. Three decompositions:
#  (a) model transfer: apply the R7-fit stacker to R7's OWN gold58 rows (same model, no R7 -> R6 shift);
#  (b) reader noise: fit on v2 verdicts, score the held-out folds against two other report readers (v4_blend, gpt56sol);
#  (c) regime: label co-occurrence (phi) in report labels vs gold58, for the stacker's top helper pairs.
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from load import *
d = r7(); g, g58 = gold(); gid = g58.StudyInstanceUID.tolist(); allg = set(g.StudyInstanceUID)
tr = d.loc[[i for i in d.index if i not in allg]]
lv2 = labels('v2').reindex(tr.index); lv4 = labels('v4').reindex(tr.index); l56 = labels('sol56').reindex(tr.index)
Z = np.clip(tr[['z|' + c for c in L]].values.astype(float), -16.1, 16.1)
# R7 deployed logits use stacker z_out for Effusion / Baker's
Z[:, L.index('Effusion')] = np.clip(tr['fluid|Effusion|z_out'].values.astype(float), -16.1, 16.1)
Z[:, L.index("Baker's")] = np.clip(tr["stk|Baker's|z_out"].values.astype(float), -16.1, 16.1)
Zg = np.clip(d.loc[gid, ['z|' + c for c in L]].values.astype(float), -16.1, 16.1)
Zg[:, L.index('Effusion')] = np.clip(d.loc[gid, 'fluid|Effusion|z_out'].values.astype(float), -16.1, 16.1)
Zg[:, L.index("Baker's")] = np.clip(d.loc[gid, "stk|Baker's|z_out"].values.astype(float), -16.1, 16.1)
mu, sd = Z.mean(0), Z.std(0); S = lambda X: (X - mu) / sd
Yg = g58[L].values.astype(int)
rows = []
for j, c in enumerate(L):
    y2 = lv2[c].values; m = ~np.isnan(y2); X = S(Z); kf = KFold(5, shuffle=True, random_state=0)
    pc = np.full(len(y2), np.nan)
    idx = np.where(m)[0]
    for a, b in kf.split(idx):
        mdl = LogisticRegression(C=0.1, max_iter=3000).fit(X[idx[a]], y2[idx[a]].astype(int)); pc[idx[b]] = mdl.decision_function(X[idx[b]])
    own = X[:, j]
    r = {'label': c}
    for nm, lab in (('v2', lv2[c].values), ('v4', lv4[c].values), ('sol56', l56[c].values)):
        mm = m & ~np.isnan(lab)   # held-out predictions exist only on v2-known rows
        r[nm + '_own'] = auc(own[mm], lab[mm].astype(int)); r[nm + '_x'] = auc(pc[mm], lab[mm].astype(int))
    full = LogisticRegression(C=0.1, max_iter=3000).fit(X[m], y2[m].astype(int))
    r['g58_R7own'] = auc(Zg[:, j], Yg[:, j]); r['g58_R7x'] = auc(full.decision_function(S(Zg)), Yg[:, j])
    rows.append(r)
df = pd.DataFrame(rows).set_index('label')
for nm in ('v2', 'v4', 'sol56', 'g58_R7'):
    a, b = (nm + '_own', nm + '_x') if nm != 'g58_R7' else ('g58_R7own', 'g58_R7x')
    df[nm + '_gain'] = df[b] - df[a]
pd.set_option('display.width', 250)
print(df.round(4).to_string())
print('macro gains:', {k: round(df[k].mean(), 4) for k in df.columns if k.endswith('_gain')})
# (c) co-occurrence phi: report (v4 >= .5, non-gold train) vs gold58 expert
Rp = lv4.loc[tr.index, L].values; Gp = Yg
phi = lambda A, i, k: np.corrcoef(A[:, i], A[:, k])[0, 1]
print('pair | phi report(v4) | phi gold58 | phi of R7 OOF z (train) | phi of R7 z (gold58)')
for t, h in [('ACL', 'Lateral OA'), ('ACL', 'MCL'), ('ACL', 'Contusion'), ('Contusion', 'Fracture'), ('Contusion', 'PF OA'), ('MCL', 'Medial Meniscus'),
             ('MCL', 'Fracture'), ('Lateral Meniscus', 'PF OA'), ('Lateral Meniscus', 'MCL'), ('Synovitis', 'Effusion'), ('Synovitis', 'Fracture'),
             ('Fracture', 'Medial OA'), ('Effusion', 'Synovitis'), ('Lateral Meniscus', 'Lateral OA'), ('Lateral Meniscus', 'Medial OA')]:
    i, k = L.index(t), L.index(h)
    print('%-16s ~ %-16s %+.3f  %+.3f  %+.3f  %+.3f' % (t, h, phi(Rp, i, k), phi(Gp, i, k), np.corrcoef(Z[:, i], Z[:, k])[0, 1], np.corrcoef(Zg[:, i], Zg[:, k])[0, 1]))
df.to_csv('xlabel2.csv')
