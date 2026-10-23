# E-OA (a .1 / .2) on every gold58 prediction set we hold locally: is the gain general across report-trained models?
import glob, sys, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, G
B = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/ours_work/boxpreds'
g = pd.read_csv('g58_labels.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID'); ids = g.index
OA = ['Medial OA', 'Lateral OA', 'PF OA']; Y = g[OA].values.astype(int)
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p; return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
def dec(F, a):
    U = np.stack([pct(F[:, k]) for k in range(3)], 1); O = U.copy()
    for k in range(3):
        o = [m for m in range(3) if m != k]; O[:, k] = U[:, k] - a * (U[:, o[0]] + U[:, o[1]]) / 2
    return O
sets = {}
rd = lambda f: pd.read_csv(f, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
sets['c4 final'] = rd('c4_g58_final.csv')[OA]
sets['R6'] = rd(G + '/R6_gold58_serve.csv')[OA]; sets['S6fall'] = rd(G + '/n42gpu_gold58.csv')[OA]
for nm, f in (('r384', 'gold58_r384_kernelserve_cpu.npy'), ('r224', 'gold58_r224_kernelserve.npy')):
    sets[nm] = pd.DataFrame(np.load(G + '/' + f), columns=L, index=ids)[OA]
r7 = rd(G + '/R7OOF.csv')[['p|' + c for c in OA]].set_axis(OA, axis=1); sets['R7 OOF'] = r7.loc[r7.index.intersection(ids)]
def foldmean(pat):
    fs = sorted(glob.glob(pat))
    if not fs: return None
    D = [rd(f) for f in fs]; D = [d.loc[d.index.intersection(ids), OA] for d in D]
    if min(len(d) for d in D) < 50: return None
    return sum(D) / len(D), len(fs)
for tag, pat in (('TTR s1044 frac75 (OAI teacher recipe)', B + '/r7t_labels/gold_preds/f*_s1044_ttr_frac75__raw__A.csv'),
                 ('TTR s1047 r7tb', B + '/r7t_labels/gold_preds/f*_s1047_ttr_r7tb__raw__A.csv'),
                 ('TTR s1048 r7ta', B + '/r7t_labels/gold_preds/f*_s1048_ttr_r7ta__raw__A.csv'),
                 ('KC s2001 ep12', B + '/fleet_runs/f0_s2001_kc256_r7tb/epochs/gold_*.csv'),
                 ('KC s2011 ep12', B + '/fleet_runs/f0_s2011_kc256_ev2s_r7tb/epochs/gold_*.csv'),
                 ('KC dread s2002-2007', B + '/fleet_runs/fA_s200*_kc256_dread*/epochs/gold_*.csv'),
                 ('TTR s1043 frac75oaiT123 (OAI-trained)', B + '/k14_test/gold/preds/f*_s1043_ttr_frac75oaiT123v00__raw__A.csv')):
    r = foldmean(pat)
    if r is not None:
        sets['%s [%d f]' % (tag, r[1])] = r[0]
print('%-44s %4s %18s %24s %24s' % ('set', 'n', 'rho ML/MPF/LPF', 'a .1 OA-mean (per label)', 'a .2 OA-mean (per label)'))
for k, d in sets.items():
    if d is None: print(k, 'n/a'); continue
    d = d.loc[ids.intersection(d.index)]; y = g.loc[d.index, OA].values.astype(int); F = d.values.astype(float)
    base = np.array([auc(F[:, j], y[:, j]) for j in range(3)]); rho = spearmanr(F).correlation
    out = []
    for a in (.1, .2):
        O = dec(F, a); dd = np.array([auc(O[:, j], y[:, j]) for j in range(3)]) - base
        out.append('%+.4f (%s)' % (dd.mean(), '/'.join('%+.3f' % x for x in dd)))
    print('%-44s %4d %18s %24s %24s' % (k[:44], len(d), '%.2f/%.2f/%.2f' % (rho[0, 1], rho[0, 2], rho[1, 2]), out[0], out[1]))
