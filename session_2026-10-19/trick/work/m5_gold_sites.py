# m5_gold_sites.py (10-19 trick lens, Claude, research only, CPU): c4 members on gold58 (expert labels) split by site proxy
# (English vs non-English report; acquisition-atypical half vs typical half). Noise-level by construction (n 58); reported for completeness.
import numpy as np, pandas as pd
from scipy.stats import rankdata
D = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58'; R = 'C:/Users/eeshs/Downloads/rsna-knee'
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(D + "/" + f, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
R6 = rd("R6_gold58_serve.csv"); S6 = rd("n42gpu_gold58.csv")
R384 = np.load(D + "/gold58_r384_kernelserve_cpu.npy"); R224 = np.load(D + "/gold58_r224_kernelserve.npy")
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    p = y.sum(); q = len(y) - p
    if p < 2 or q < 2: return np.nan
    r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
def c4(table=C4, ws6=.2):
    out = np.zeros((n, 12))
    for j, c in enumerate(L):
        rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); w = table.get(c, .5)
        b = w * rr + (1 - w) * pct(R6[:, j]); out[:, j] = (1 - ws6) * pct(b) + ws6 * pct(S6[:, j])
    return out
RAP = np.column_stack([.75 * pct(R384[:, j]) + .25 * pct(R224[:, j]) for j in range(12)])
M = {'Raptor(r384+r224)': RAP, 'R6': R6, 'S6fall': S6, 'c4': c4(), 'c4 uniform .5 table': c4(table={})}
lang = pd.read_csv(R + '/_data/_reports_lang.csv', usecols=['StudyInstanceUID','lang']).set_index('StudyInstanceUID').lang.reindex(ids).values
at = pd.read_csv('atypicality.csv', index_col=0).atyp.reindex(ids).values
groups = {'en': lang == 'en', 'non-en': lang != 'en', 'typical half': at <= np.median(at), 'atypical half': at > np.median(at)}
def macro(P, m):
    return np.nanmean([auc(P[m, j], Y[m, j]) for j in range(12)])
rng = np.random.default_rng(2)
print('n per group', {k: int(v.sum()) for k, v in groups.items()})
rows = {}
for nm, P in M.items():
    rows[nm] = {k: macro(P, m) for k, m in groups.items()}
    rows[nm]['all'] = macro(P, np.ones(n, bool))
tab = pd.DataFrame(rows).T; tab['nonen-en'] = tab['non-en'] - tab['en']; tab['atyp-typ'] = tab['atypical half'] - tab['typical half']
print(tab.round(4).to_string())
# bootstrap: is any member's (non-en minus en) gap different from c4's?
for nm in ['Raptor(r384+r224)', 'R6', 'S6fall']:
    bs = []
    for _ in range(1000):
        i = rng.integers(0, n, n); m_ = {k: v[i] for k, v in groups.items()}
        P = M[nm][i]; C = M['c4'][i]; Yb = Y[i]
        def mac(PP, mm): return np.nanmean([auc(PP[mm, j], Yb[mm, j]) for j in range(12)])
        bs.append((mac(P, m_['non-en']) - mac(P, m_['en'])) - (mac(C, m_['non-en']) - mac(C, m_['en'])))
    bs = np.array(bs)
    print('%-18s (non-en - en) gap minus c4 gap: mean %+.4f sd %.4f' % (nm, np.nanmean(bs), np.nanstd(bs)))
