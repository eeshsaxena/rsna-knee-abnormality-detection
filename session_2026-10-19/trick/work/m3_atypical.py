# m3_atypical.py (10-19 trick lens, Claude, research only, CPU): do R7 (R6 family) and S6 seeds (S6fall family) degrade
# differently on acquisition-atypical train studies (a within-train stand-in for unseen sites)? OOF, report labels.
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
import load as LD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
L = LD.L; R = LD.R; B = R + '/session_2026-10-19/trick/box'
rng = np.random.default_rng(0)
sf = pd.read_csv(B + '/store_feats.csv').set_index('StudyInstanceUID')
ct = pd.read_csv('series_comp_local.csv', index_col=0)
feat = sf.drop(columns=[c for c in sf.columns if c.endswith('_present') or c == 'err'], errors='ignore')
feat = feat.join(ct[['Sag1','Sag0','Cor1','Cor0','Axi1','Axi0']])
feat = feat.fillna(feat.median())
X = StandardScaler().fit_transform(np.log1p(feat.clip(lower=0)))
nn = NearestNeighbors(n_neighbors=21).fit(X)
d, _ = nn.kneighbors(X)
atyp = pd.Series(d[:, 1:].mean(1), index=feat.index)          # mean distance to 20 nearest other studies
lang = pd.read_csv(R + '/_data/_reports_lang.csv', usecols=['StudyInstanceUID','lang']).set_index('StudyInstanceUID').lang
r7 = LD.r7()[['p|' + c for c in L]]; r7.columns = L
s44, _ = LD.s6seed(1044); s45, _ = LD.s6seed(1045)
g, _ = LD.gold(); gold_ids = set(g.StudyInstanceUID)
print('atypicality by lang (median):', atyp.groupby(lang.reindex(atyp.index).values).median().round(2).to_dict())
for src in ['v2', 'v4']:
    lab = LD.labels(src)
    ids = sorted(set(r7.index) & set(s44.index) & set(s45.index) & set(lab.index) - gold_ids)
    a = atyp.reindex(ids).values
    S6 = (s44.loc[ids, L].rank(pct=True).values + s45.loc[ids, L].rank(pct=True).values) / 2
    R7 = r7.loc[ids, L].rank(pct=True).values
    Y = lab.loc[ids, L].values
    for q in [0.9, 0.8]:
        hi = a >= np.quantile(a, q)
        def gaps(idx):
            out = []
            for j in range(12):
                y = Y[idx, j]; ok = ~np.isnan(y); h = hi[idx][ok]; yy = y[ok].astype(int)
                if yy[h].sum() < 5 or (1 - yy[h]).sum() < 5: continue
                # pooled AUC on the atypical stratum vs the typical stratum, per member; then (R7 - S6) difference of drops
                ar = LD.auc(R7[idx][ok][h, j], yy[h]) - LD.auc(R7[idx][ok][~h, j], yy[~h])
                as_ = LD.auc(S6[idx][ok][h, j], yy[h]) - LD.auc(S6[idx][ok][~h, j], yy[~h])
                out.append((ar, as_))
            o = np.array(out); return o[:, 0].mean(), o[:, 1].mean(), len(o)
        base = gaps(np.arange(len(ids)))
        bs = []
        for b in range(300):
            idx = rng.integers(0, len(ids), len(ids)); r = gaps(idx); bs.append(r[0] - r[1])
        bs = np.array(bs)
        print('%s top %d%% atypical (n %d): AUC drop atyp-typ R7 %+.4f  S6 %+.4f  (R7-S6 %+.4f, boot sd %.4f, P(R7 drops less) %.2f, labels %d)' % (
            src, round((1 - q) * 100), hi.sum(), base[0], base[1], base[0] - base[1], bs.std(), (bs > 0).mean(), base[2]))
atyp.rename('atyp').to_csv('atypicality.csv')
