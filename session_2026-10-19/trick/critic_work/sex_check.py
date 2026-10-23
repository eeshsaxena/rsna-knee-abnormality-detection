import pandas as pd, numpy as np, os
from scipy.stats import rankdata, norm
st = pd.read_csv(os.environ['TEMP'] + '/hdr_study.csv', dtype=str).set_index('StudyInstanceUID')
st = st[st.sex.isin(['M', 'F'])]
sex = (st.sex == 'F').astype(float)
tr = pd.read_csv('_data/train.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
L = [c for c in tr.columns if c != 'Report']
gold = tr.dropna(subset=L)
r7 = pd.read_csv('session_2026-10-18/next/g58/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
v2 = pd.read_csv('_labels/rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
def fit(x, y):
    X = np.c_[np.ones(len(y)), x]; b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1]); g = X.T @ (y - p)
        b += np.linalg.solve(H, g)
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return b, se
def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
for ruler, ids, Y in (('report v2, non-gold', st.index.difference(gold.index), v2),
                      ('gold58 expert', st.index.intersection(gold.index), gold)):
    ids = ids.intersection(r7.index).intersection(Y.index)
    print(f'== {ruler}: n {len(ids)}  female {sex[ids].mean():.2f}')
    zs = []
    for c in L:
        y = (Y.loc[ids, c].astype(float) >= .5).astype(float).values
        if y.sum() < 5 or y.sum() > len(y) - 5: continue
        z = r7.loc[ids, 'z|' + c].astype(float).values
        b, se = fit(np.c_[z, sex[ids].values], y)
        a0 = auc(z, y)
        prevF = y[sex[ids].values == 1].mean(); prevM = y[sex[ids].values == 0].mean()
        zs.append(b[2] / se[2])
        print(f'{c:17s} pos {int(y.sum()):3d} prev F/M {prevF:.2f}/{prevM:.2f}  R7 AUC {a0:.3f}  sex|R7 coef {b[2]:+.2f} (z {b[2]/se[2]:+.2f})')
    zs = np.array(zs); print(f'  sum z^2 {np.sum(zs**2):.1f} on {len(zs)} df (chi2 .95 ~ {len(zs)+1.645*np.sqrt(2*len(zs)):.1f})')
