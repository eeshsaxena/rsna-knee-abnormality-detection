# label_sources.py: every report-label table on the box scored on gold58 (target level), and simple multi-source
# averages (Raymond's "multiple weak-label sources"). CPU, read-only.
import os, glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr
H = os.path.expanduser
T = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
gold = pd.read_csv(H('~/rsna_train/gold74_eval_set.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
G = [s for s in gold.index if gold.loc[s, 'source'] == 'original_gold58']; Y = gold.loc[G, T].astype(int)
tabs = {}
for f in ['llm_labels_v4_blend', 'labels_llm_gpt56sol', 'pseudo_labels_qwen7b', 'pseudo_labels_v1', 'report_labels_v2',
          'FINAL2_labels_with_confidence', 'labels_mix50_logit', 'labels_gated75_logit', 'labels_mixSA_logit', 'teacher_train_probs']:
    p = H(f'~/rsna_train/{f}.csv')
    try:
        d = pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
        if not all(l in d.columns for l in T): print(f, 'missing label cols', list(d.columns)[:6]); continue
        d = d[T].apply(pd.to_numeric, errors='coerce')
        cov = d.reindex(G).notna().all(1).mean(); n = len(d)
        tabs[f] = d; X = d.reindex(G)
        ok = X.notna().all(1).values
        a = np.mean([roc_auc_score(Y[l].values[ok], X[l].values[ok]) for l in T])
        print(f'{f:32s} rows {n:5d} gold58 coverage {cov:.2f} AUC {a:.4f}')
    except Exception as e:
        print(f, 'ERR', str(e)[:80])
def rk(d): return d.rank(pct=True)
def sc(X): return np.mean([roc_auc_score(Y[l], X[l]) for l in T])
srcs = [k for k in ['llm_labels_v4_blend', 'labels_llm_gpt56sol', 'pseudo_labels_qwen7b', 'report_labels_v2'] if k in tabs and tabs[k].reindex(G).notna().all(1).all()]
print('multi-source rank means over', srcs)
import itertools
for r in range(2, len(srcs) + 1):
    for c in itertools.combinations(srcs, r):
        X = sum(rk(tabs[k].reindex(G)) for k in c) / len(c)
        print(f'  {" + ".join(k[:14] for k in c):70s} {sc(X):.4f}')
# agreement between sources on all train rows
ids = sorted(set.intersection(*[set(tabs[k].dropna().index) for k in srcs]))
for a, b in itertools.combinations(srcs, 2):
    print(f'rho {a[:16]} ~ {b[:16]}: {np.nanmean([spearmanr(tabs[a].loc[ids, l], tabs[b].loc[ids, l]).correlation for l in T]):.3f}')
