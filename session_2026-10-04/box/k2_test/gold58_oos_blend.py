"""gold58 blend value of the fleet into the parent's OUT-OF-SAMPLE arms only.
The parent's DINOv2 stack / RadImageNet e10-e13 arms score .96-.996 on gold58 (trained on these studies' gold labels),
so the parent's own gold58 (.951) cannot judge a blend. Raptor (Dread excluded the 58) and CoAt arms look OOS."""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
tr = pd.read_csv(os.path.expanduser('~/kmimic/kaggle/input/rsna-knee-abnormality-detection/train.csv'),
                 dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
R = os.path.expanduser('~/kmimic/runs/gold58_gpu')
F = pd.read_csv(os.path.expanduser('~/k2_test/runs/C_gold58/working/submission_fleetsolo.csv'),
                dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
ids = [s for s in F.index if s in tr.index and not np.isnan(tr.loc[s, 'ACL'])]
Y = tr.loc[ids, T].values.astype(float); Fv = F.loc[ids, T].values


def load(f): return pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, T].values.astype(float)
def rp(a): return np.column_stack([rankdata(a[:, j]) / len(a) for j in range(a.shape[1])])
def per(X): return np.array([roc_auc_score(Y[:, j], X[:, j]) for j in range(12)])


bases = {'raptor_coat_side (Raptor+CoAt, OOS)': R + '/localLB_parts/raptor_coat_side_rank.csv',
         'raptor_public (4 Raptor arms)': R + '/localLB_parts/raptor_public_rank.csv',
         'coat_family': R + '/localLB_parts/coat_family_rank.csv',
         'raptor_arm maxspan-v5': R + '/localLB_parts/raptor_arm_maxspan-v5_prob.csv'}
rng = np.random.default_rng(0); boot = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
for name, f in bases.items():
    Bv = load(f); b0 = per(Bv).mean()
    line = f'{name:38s} base {b0:.4f} |'
    for w in [0.15, 0.25, 0.35, 0.5]:
        line += f' W{w:.2f} {per(rp((1 - w) * rp(Bv) + w * rp(Fv))).mean() - b0:+.4f}'
    X = rp(0.75 * rp(Bv) + 0.25 * rp(Fv)); d = []
    for i in boot:
        Yb = Y[i]; v = [roc_auc_score(Yb[:, j], X[i, j]) - roc_auc_score(Yb[:, j], Bv[i, j]) for j in range(12) if len(set(Yb[:, j])) > 1]
        d.append(np.mean(v))
    d = np.array(d)
    print(line + f' | W.25 boot 90% [{np.quantile(d, .05):+.4f}, {np.quantile(d, .95):+.4f}] P>0 {(d > 0).mean():.2f}')
