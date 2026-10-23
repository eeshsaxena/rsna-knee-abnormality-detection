"""labx.py (jump2, 2026-10-05): label tables, quality vs parent-likeness. CPU only, read-only inputs.
For each target table: gold58 macro AUC (vs radiologist labels in train.csv) and target-level parent-likeness
(mean per-label Spearman vs the parent's Raptor arms over all non-gold train studies; plus within-class latent r on gold58).
Also simple blends that drop the shared public-label channel. Output ~/jump2/labx_out.json.
"""
import os, json, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
H = os.path.expanduser
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
def rd(p):
    d = pd.read_csv(H(p), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID'); return d[T].astype(float)
L = {'steven_v4': '~/rsna_train/llm_labels_v4_blend.csv', 'mix50_logit': '~/rsna_train/labels_mix50_logit.csv',
     'gated75_logit': '~/rsna_train/labels_gated75_logit.csv', 'gpt56sol': '~/rsna_train/labels_llm_gpt56sol.csv',
     'qwen7b': '~/rsna_train/pseudo_labels_qwen7b.csv', 'pseudo_v1': '~/rsna_train/pseudo_labels_v1.csv',
     'FINAL2': '~/rsna_train/FINAL2_labels_with_confidence.csv', 'report_v2': '~/rsna_train/report_labels_v2.csv',
     'widedense_teacher': '~/rsna_train/teacher_train_probs.csv'}
LT = {k: rd(p) for k, p in L.items()}
RP = rd('~/fleet/teacher_v2_raptorproxy_shard0.csv')
g = pd.read_csv(H('~/kmimic/ids/gold58.csv'), dtype={'StudyInstanceUID': str})
g58 = g.loc[g.subset == 'gold58', 'StudyInstanceUID'].tolist()
tr = pd.read_csv(H('~/kmimic/kaggle/input/rsna-knee-abnormality-detection/train.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
G = tr.loc[g58, T].astype(float).values >= 0.5
def rk(df, ids): return np.column_stack([rankdata(df.loc[ids, t].values) / len(ids) for t in T])
def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
def gold(df):
    keep = [i for i, u in enumerate(g58) if u in df.index]
    S = df.loc[[g58[i] for i in keep], T].values; Gk = G[keep]
    return float(np.nanmean([auc(Gk[:, j], S[:, j]) for j in range(12)]))
ids = sorted(set.intersection(*[set(v.index) for v in LT.values()]) & set(RP.index) - set(g58))
def sp(a, b): return float(np.mean([spearmanr(a[:, j], b[:, j]).correlation for j in range(12)]))
# composite independent-noise tables (rank means), built here only for measurement
def rmean(names):
    allid = sorted(set.intersection(*[set(LT[n].index) for n in names]))
    return pd.DataFrame(np.mean([rk(LT[n], allid) for n in names], axis=0), index=allid, columns=T)
LT['indep4 = rank-mean(gpt56sol,qwen7b,FINAL2,report_v2)'] = rmean(['gpt56sol', 'qwen7b', 'FINAL2', 'report_v2'])
LT['v4+indep4 (rank-mean of 5)'] = rmean(['steven_v4', 'gpt56sol', 'qwen7b', 'FINAL2', 'report_v2'])
LT['gpt56sol+report_v2'] = rmean(['gpt56sol', 'report_v2'])
out = {}
print(f'{len(ids)} non-gold train studies for parent-likeness; gold58 n={len(g58)}')
print('table | gold58 macro AUC | sp vs raptorproxy (train) | sp vs steven_v4 (train)')
v4r = rk(LT['steven_v4'], ids); rpr = rk(RP, ids)
for k, v in LT.items():
    a = rk(v, ids); out[k] = dict(gold58=gold(v), sp_rp=sp(a, rpr), sp_v4=sp(a, v4r))
    print(f'  {k:52s} {out[k]["gold58"]:.4f} | {out[k]["sp_rp"]:.3f} | {out[k]["sp_v4"]:.3f}')
print(f'  raptorproxy (parent arm, for reference)              {gold(RP):.4f} | 1.000 | {sp(rpr, v4r):.3f}')
json.dump(out, open(H('~/jump2/labx_out.json'), 'w'), indent=1)
