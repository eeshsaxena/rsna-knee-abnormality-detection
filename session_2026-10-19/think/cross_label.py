# cross_label.py (10-19, Claude, research only): does a cross-label stacker (each label's logit regressed on all 12 logits of the
# same model) add ranking information? Fit on R7 OOF z (4,349 non-gold train studies) vs LLM report verdicts (YES / NO), 5-fold CV;
# then the R7-fitted stacker is applied unchanged to R6's deployed logits on gold58 (held out, radiologist labels).
import json, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
R = 'C:/Users/eeshs/Downloads/rsna-knee'
G = R + '/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


d = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
g58 = g[g.source == 'original_gold58'].reset_index(drop=True)
lab = pd.read_csv(R + '/_labels/rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
tr = d.loc[~d.index.isin(set(g.StudyInstanceUID))]
lab = lab.reindex(tr.index)
# R7's deployed logit per label: logit of its p (Effusion / Baker's p are stacker outputs); clip saturated cells
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
Xtr = lg(tr[['p|' + c for c in L]].values)
# R6 deployed logits on gold58 (fold-mean flip z; stacker z_out for Baker's / Effusion)
W0 = G + '/r6g58_box/run_b/worker0'
dump = json.load(open(W0 + '/dump.json'))
ids = g58.StudyInstanceUID.astype(str).tolist(); Y = g58[L].values.astype(int)
Z6 = np.array([np.mean(np.array(dump[u]['z'], np.float64), axis=0) for u in ids])
st = pd.read_csv(W0 + '/state_stacker.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
fl = pd.read_csv(W0 + '/fluid_stacker.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
Z6[:, L.index("Baker's")] = st.loc[ids, "Baker's|z_out"].values
Z6[:, L.index('Effusion')] = fl.loc[ids, 'Effusion|z_out'].values
Z6 = np.clip(Z6, -16.1, 16.1)   # same clip as lg on train
Xtr = np.clip(Xtr, -16.1, 16.1)
mu, sd = Xtr.mean(0), Xtr.std(0)
S = lambda X: (X - mu) / sd
res = {}
print('label | CV own-only | CV cross | gain | gold58 R6 own | gold58 R6 cross | gain | top cross coefs')
for j, c in enumerate(L):
    v = lab[c + '__verdict'].values; m = np.isin(v, ['YES', 'NO']); y = (v[m] == 'YES').astype(int); X = S(Xtr[m])
    kf = KFold(5, shuffle=True, random_state=0); po = np.zeros(len(y)); pc = np.zeros(len(y))
    for a, b in kf.split(X):
        po[b] = X[b, j]   # own logit only = identity (a 1-feature LR is monotone)
        mc = LogisticRegression(C=0.1, max_iter=2000).fit(X[a], y[a]); pc[b] = mc.decision_function(X[b])
    cv_o, cv_c = auc(po, y), auc(pc, y)
    mfull = LogisticRegression(C=0.1, max_iter=2000).fit(X, y)
    g_o = auc(Z6[:, j], Y[:, j]); g_c = auc(mfull.decision_function(S(Z6)), Y[:, j])
    co = mfull.coef_[0] / abs(mfull.coef_[0][j]); top = sorted([(L[k], round(co[k], 2)) for k in range(12) if k != j], key=lambda t: -abs(t[1]))[:3]
    res[c] = dict(cv_own=cv_o, cv_cross=cv_c, g58_own=g_o, g58_cross=g_c, top=top)
    print('%-17s | %.4f | %.4f | %+.4f | %.4f | %.4f | %+.4f | %s' % (c, cv_o, cv_c, cv_c - cv_o, g_o, g_c, g_c - g_o, top))
mo = np.mean([r['cv_cross'] - r['cv_own'] for r in res.values()]); mg = np.mean([r['g58_cross'] - r['g58_own'] for r in res.values()])
print('macro CV gain %+.4f | macro gold58 gain (R7-fit stacker on R6) %+.4f' % (mo, mg))
json.dump(res, open(R + '/session_2026-10-19/think/cross_label.json', 'w'), indent=1, default=str)
