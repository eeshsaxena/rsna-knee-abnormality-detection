#!/usr/bin/env python3
"""FAST lever search on cached DINOv2 embeddings (no re-embed). Pools per study,
fits per-label logistic regression, evals gold-58 macro-AUC. Sweeps label subsets,
pooling, and hard/soft to find what moves the number. Runs in seconds."""
import glob, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

D = r"C:\Users\eeshs\Downloads\rsna-knee"
LABELS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
        'Effusion','Synovitis',"Baker's",'Contusion','Fracture']
z=np.load(glob.glob(D+r"\kernels\_out_full9\shards\emb_cache.npz")[0], allow_pickle=True)
emb, meta, gids = z['emb'].astype(np.float32), z['meta'], np.array(z['study_ids'])
by={}
for row,(si,pid,fb) in enumerate(meta): by.setdefault(int(si),[]).append(row)

def pool(mode):
    X=np.zeros((len(gids), emb.shape[1]), np.float32)
    for si,rows in by.items():
        e=emb[rows]
        X[si]= e.mean(0) if mode=='mean' else e.max(0) if mode=='max' else np.concatenate([e.mean(0),e.max(0)])[:emb.shape[1]]
    return X

rule=pd.read_csv(D+r"\_data\pseudo_labels_rulev0.csv").set_index('StudyInstanceUID').reindex(gids)
qwen=pd.read_csv(D+r"\_out_qwen_full\pseudo_labels_qwen.csv").set_index('StudyInstanceUID').reindex(gids)
soft=pd.read_csv(D+r"\_data\pseudo_labels_v1.csv").set_index('StudyInstanceUID').reindex(gids)
tr=pd.read_csv(D+r"\_data\train.csv").set_index('StudyInstanceUID').reindex(gids)
lang=pd.read_csv(D+r"\_data\_reports_lang.csv").set_index('StudyInstanceUID').reindex(gids)
lc=[c for c in lang.columns if 'lang' in c.lower()][0]; L=lang[lc].fillna('unk').values
gold=tr['ACL'].notna().values; gi=np.where(gold)[0]; ri=np.where(~gold)[0]
Yg=tr[LABELS].values.astype(float)

def evalcfg(name, Xtr_idx, Xall, ycol_src, hard=True):
    aucs=[]
    for l in LABELS:
        y=ycol_src[l].values.astype(float)[Xtr_idx]
        yb=(y>0.5).astype(int) if hard else y
        m=(~np.isnan(yb));
        if len(set(np.round(yb[m]).astype(int)))<2: aucs.append(np.nan); continue
        clf=LogisticRegression(max_iter=200,C=1.0)
        clf.fit(Xall[Xtr_idx][m], np.round(yb[m]).astype(int))
        p=clf.predict_proba(Xall[gi])[:,1]
        yt=Yg[gi][:,LABELS.index(l)]
        aucs.append(roc_auc_score(yt,p) if len(set(yt))>1 else np.nan)
    macro=np.nanmean(aucs)
    print(f"{name:38s} macroAUC={macro:.4f}")
    return macro, aucs

Xmean=pool('mean'); Xmax=pool('max')
print("=== POOLING / SUBSET / LABEL sweep (gold-58) ===")
evalcfg("mean-pool | all report | soft-hard", ri, Xmean, soft)
evalcfg("max-pool  | all report | soft-hard", ri, Xmax, soft)
# high-confidence: rule & qwen agree per label handled inside via soft in {0,1}; approximate with agree mask
agree = (rule[LABELS].values==qwen[LABELS].clip(0,1).values)
# english-heavy subset
for langset,nm in [(['en'],'en'),(['en','es','nl','de','fr'],'en+eu')]:
    idx=ri[np.isin(L[ri], langset)]
    if len(idx)>200: evalcfg(f"mean|{nm}({len(idx)})|soft-hard", idx, Xmean, soft)
# hard vs soft on Qwen alone / rule alone
evalcfg("mean-pool | all report | QWEN hard", ri, Xmean, qwen)
evalcfg("mean-pool | all report | RULE hard", ri, Xmean, rule)
print("\n=== per-label AUC (mean-pool, all report, soft) ===")
_,a=evalcfg("mean|all|soft (per-label below)", ri, Xmean, soft)
for l,au in zip(LABELS,a): print(f"  {l:18s} {au:.3f}")
