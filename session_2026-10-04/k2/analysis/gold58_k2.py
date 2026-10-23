#!/usr/bin/env python3
"""gold58 local LB for K2: parent (box mimic, real) vs fleet solo (K2 serve path, real DICOMs) vs P+F rank blends.
gold58 = train.csv image labels; fleet member never trained on them (held out), parent public models may have."""
import sys, glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, rankdata
work = sys.argv[1]; par_csv = sys.argv[2]; tr_csv = sys.argv[3]
T = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
P = pd.read_csv(par_csv, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
F = pd.read_csv(f'{work}/submission_fleetsolo.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
tr = pd.read_csv(tr_csv, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
ids = [s for s in F.index if s in tr.index and not np.isnan(tr.loc[s, 'ACL'])]
Y = tr.loc[ids, T].values.astype(float); Pv = P.loc[ids, T].values; Fv = F.loc[ids, T].values
def rp(a): return np.column_stack([rankdata(a[:, j]) / len(a) for j in range(a.shape[1])])
def mauc(X):
    return float(np.mean([roc_auc_score(Y[:, j], X[:, j]) for j in range(12) if len(set(Y[:, j])) > 1]))
def per(X): return [roc_auc_score(Y[:, j], X[:, j]) if len(set(Y[:, j])) > 1 else np.nan for j in range(12)]
print(f'n {len(ids)} | positives per label {Y.sum(0).astype(int).tolist()}')
print(f'parent {mauc(Pv):.4f} | fleet solo {mauc(Fv):.4f} | rho(P,F) mean {np.nanmean([spearmanr(Pv[:,j],Fv[:,j]).correlation for j in range(12)]):.3f}')
for w in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]:
    B = rp((1 - w) * rp(Pv) + w * rp(Fv)); print(f'  W {w:.2f}: {mauc(B):.4f}  ({mauc(B)-mauc(Pv):+.4f})')
for f in sorted(glob.glob(f'{work}/submission_w*.csv')):
    X = pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, T].values
    print(f'  file {f.split("/")[-1]}: {mauc(X):.4f}')
pp, pf = per(Pv), per(Fv); B30 = rp(0.7 * rp(Pv) + 0.3 * rp(Fv)); pb = per(B30)
print('label            parent  fleet   W.30')
for j, t in enumerate(T): print(f'{t:16s} {pp[j]:.3f}   {pf[j]:.3f}   {pb[j]:.3f}')
# bootstrap the W.30 - parent delta (study resampling)
rng = np.random.default_rng(0); d = []
for _ in range(2000):
    i = rng.integers(0, len(ids), len(ids)); Yb = Y[i]
    vals = []
    for j in range(12):
        if len(set(Yb[:, j])) > 1: vals.append(roc_auc_score(Yb[:, j], B30[i, j]) - roc_auc_score(Yb[:, j], Pv[i, j]))
    d.append(np.mean(vals))
d = np.array(d); print(f'W.30 - parent bootstrap: mean {d.mean():+.4f}  90% CI [{np.quantile(d,.05):+.4f}, {np.quantile(d,.95):+.4f}]  P(>0) {(d>0).mean():.2f}')
