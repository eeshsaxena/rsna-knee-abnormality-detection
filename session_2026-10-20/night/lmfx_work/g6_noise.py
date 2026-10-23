# g6_noise.py (10-20 night, Track 3, Claude). RESEARCH ONLY, local CPU. Public / private read noise of the two best per-label
# candidates (the r5/noise_corr.py method): sd of the MACRO delta over 400 random subsamples of 390 / 910 non-gold train studies,
# R7 OOF vs sol56 report labels (all 12 labels defined). The sd is the read noise, not the expected effect.
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G, labels
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
g, _ = gold(); gid = set(g.StudyInstanceUID)
lab = labels('sol56'); ii = [u for u in lab.index.intersection(r7.index) if u not in gid]
F = r7.loc[ii, ['p|' + c for c in L]].values; Y = lab.loc[ii, L].values
ok = ~np.isnan(Y).any(1); F, Y = F[ok], Y[ok].astype(int); n = len(Y)
jL, jF = L.index('Lateral Meniscus'), L.index('Fracture')
def lm_shrink(Fs):
    f = pct(Fs[:, jL]); return .5 + (f - .5) * (1 - .4 * pct(Fs[:, L.index('ACL')]))
def fr_deg(Fs):
    return pct(Fs[:, jF]) - .1 * np.mean([pct(Fs[:, L.index(c)]) for c in ('Medial OA', 'Lateral OA', 'PF OA')], 0)
rng = np.random.default_rng(7)
for nm, fn, j in (('LM ACL-ctx shrink k.4', lm_shrink, jL), ('Frac <- DEG a.1', fr_deg, jF)):
    for m in (390, 910):
        ds = []
        for _ in range(400):
            ix = rng.choice(n, m, replace=False); y = Y[ix, j]
            if 0 < y.sum() < m: ds.append((auc(fn(F[ix]), y) - auc(F[ix, j], y)) / 12)
        ds = np.array(ds); print('%-24s subsample %d: macro delta mean %+.5f sd %.5f' % (nm, m, ds.mean(), ds.std()))
print('n pool', n)
