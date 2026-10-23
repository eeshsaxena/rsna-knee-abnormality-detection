"""gold58: every parent arm (box mimic run, real) vs the K2 fleet member. AUC, rho vs parent, rho vs fleet."""
import glob, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
tr = pd.read_csv(os.path.expanduser('~/kmimic/kaggle/input/rsna-knee-abnormality-detection/train.csv'),
                 dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
R = os.path.expanduser('~/kmimic/runs/gold58_gpu')
F = pd.read_csv(os.path.expanduser('~/k2_test/runs/C_gold58/working/submission_fleetsolo.csv'),
                dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
ids = [s for s in F.index if s in tr.index and not np.isnan(tr.loc[s, 'ACL'])]
Y = tr.loc[ids, T].values.astype(float); Fv = F.loc[ids, T].values


def mauc(X): return float(np.mean([roc_auc_score(Y[:, j], X[:, j]) for j in range(12)]))
def rho(A, B): return float(np.nanmean([spearmanr(A[:, j], B[:, j]).correlation for j in range(12)]))


files = sorted(glob.glob(R + '/localLB_parts/*.csv'))
for wd in ['_raptor.csv', '_raptor_public.csv', '_coat_arm.csv', '_coat_repairv1.csv',
           'submission_checkpoint_rank_mean.csv', 'raptor_input_before_coat.csv']:
    files.append(R + '/kaggle/working/' + wd)
files.append(R + '/localLB_parent_preds.csv')
P = pd.read_csv(R + '/localLB_parent_preds.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, T].values
print(f'{"arm":40s} gold58  rhoParent rhoFleet')
for f in files:
    try:
        d = pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
        X = d.loc[ids, T].values.astype(float)
        print(f'{os.path.basename(f)[:40]:40s} {mauc(X):.4f}  {rho(X, P):.3f}     {rho(X, Fv):.3f}')
    except Exception as e:
        print(f'{os.path.basename(f)[:40]:40s} ERR {str(e)[:60]}')
print(f'{"FLEET f0 mix50":40s} {mauc(Fv):.4f}  {rho(Fv, P):.3f}')
