# m4_offset_align.py (10-19 trick lens, Claude, research only, CPU): label-free test-time site-offset alignment between members.
# (a) consensus alignment (each member's per-cluster mean pct -> the weighted mean of members): MODEL says ~no-op for a linear blend;
# (b) align R7's per-cluster offsets to S6's (= "trust one family's site offsets"); clusters = language (true site proxy) or
#     observable clusters (KMeans on store + series features, available at test time). OOF, report labels, paired bootstrap.
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
import load as LD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
L = LD.L; R = LD.R; B = R + '/session_2026-10-19/trick/box'
rng = np.random.default_rng(1)
lang = pd.read_csv(R + '/_data/_reports_lang.csv', usecols=['StudyInstanceUID','lang']).set_index('StudyInstanceUID').lang
sf = pd.read_csv(B + '/store_feats.csv').set_index('StudyInstanceUID')
ct = pd.read_csv('series_comp_local.csv', index_col=0)
feat = sf.drop(columns=[c for c in sf.columns if c.endswith('_present') or c == 'err'], errors='ignore').join(ct[['Sag1','Sag0','Cor1','Cor0','Axi1','Axi0']])
feat = feat.fillna(feat.median())
Xf = StandardScaler().fit_transform(np.log1p(feat.clip(lower=0)))
km = pd.Series(KMeans(12, n_init=5, random_state=0).fit_predict(Xf), index=feat.index)
r7 = LD.r7()[['p|' + c for c in L]]; r7.columns = L
s44, _ = LD.s6seed(1044); s45, _ = LD.s6seed(1045)
g, _ = LD.gold(); gold_ids = set(g.StudyInstanceUID)

def macro(P, Y, idx):
    a = []
    for j in range(12):
        y = Y[idx, j]; ok = ~np.isnan(y)
        a.append(LD.auc(P[idx][ok, j], y[ok].astype(int)))
    return np.mean(a)

def align(Pm, cl, target):
    out = Pm.copy()
    for k in np.unique(cl):
        m = cl == k
        out[m] += target[m].mean(0) - Pm[m].mean(0)
    return out

for src in ['v2', 'v4']:
    lab = LD.labels(src)
    ids = sorted(set(r7.index) & set(s44.index) & set(s45.index) & set(lab.index) - gold_ids)
    Y = lab.loc[ids, L].values
    pr7 = r7.loc[ids, L].rank(pct=True).values
    ps6 = ((s44.loc[ids, L].rank(pct=True) + s45.loc[ids, L].rank(pct=True)) / 2).values
    base = .5 * pr7 + .5 * ps6
    for cname, cl in [('language', lang.reindex(ids).values), ('obs-kmeans12', km.reindex(ids).values)]:
        cons = .5 * align(pr7, cl, base) + .5 * align(ps6, cl, base)
        r7_to_s6 = .5 * align(pr7, cl, ps6) + .5 * ps6
        s6_to_r7 = .5 * pr7 + .5 * align(ps6, cl, pr7)
        n = len(ids); full = np.arange(n)
        b0 = macro(base, Y, full)
        for nm, P in [('consensus', cons), ('R7->S6 offsets', r7_to_s6), ('S6->R7 offsets', s6_to_r7)]:
            d = macro(P, Y, full) - b0
            bs = []
            for _ in range(200):
                idx = rng.integers(0, n, n); bs.append(macro(P, Y, idx) - macro(base, Y, idx))
            bs = np.array(bs)
            print('%s %-13s %-15s base %.4f  d %+.5f (boot sd %.5f, P>0 %.2f)' % (src, cname, nm, b0, d, bs.std(), (bs > 0).mean()))
