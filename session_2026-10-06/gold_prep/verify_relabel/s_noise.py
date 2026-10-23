# How noisy are G1 and S on gold58? Use existing score sets as stand-ins for the VLM arms.
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
SP='C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad'
DS='C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-05/kaggle_train/ds/rsna-knee-fleet-train'
rd=lambda p: pd.read_csv(p,dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
g=rd(f'{DS}/gold74_eval_set.csv'); g58=g[g.source=='original_gold58']; ids=g58.index
T=['Effusion','Synovitis','Fracture','Contusion','PF OA','Lateral OA']
Y=(g58[T]>=.5).to_numpy()
mix=rd(f'{DS}/labels_mix50_logit.csv').loc[ids,T].to_numpy(float)
v4=rd(f'{DS}/llm_labels_v4_blend.csv').loc[ids,T].to_numpy(float)
rap=rd(f'{SP}/teacher_v2_out/raptor_proxy_v5v8.csv').reindex(ids)[T].to_numpy(float)
print('raptor rows on gold58', int(np.isfinite(rap).all(1).sum()))
lg=lambda p: np.log(np.clip(p,1e-4,1-1e-4)/(1-np.clip(p,1e-4,1-1e-4)))
bl=lambda P: 1/(1+np.exp(-(0.5*lg(mix)+0.5*lg(np.clip(P,.02,.98)))))
def aucs(P,i): return np.array([roc_auc_score(Y[i,j],P[i,j]) for j in range(6)])
rng=np.random.default_rng(0); n=len(ids)
for name,P in (('v4 as PI',v4),('raptor as PI',rap)):
    full=np.mean(aucs(bl(P),np.arange(n))-aucs(mix,np.arange(n)))
    s=[np.mean(aucs(bl(P),i)-aucs(mix,i)) for i in (rng.integers(0,n,n) for _ in range(1000))]
    print(f'{name}: S = {full:+.4f}, bootstrap SE(S) {np.std(s):.4f}, 90% [{np.percentile(s,5):+.4f},{np.percentile(s,95):+.4f}]')
# G1-type: Effusion AUC(raptor) - AUC(v4) as an image-vs-text stand-in
d=[aucs(rap,i)[0]-aucs(v4,i)[0] for i in (rng.integers(0,n,n) for _ in range(1000))]
print(f'Effusion raptor - v4 = {aucs(rap,np.arange(n))[0]-aucs(v4,np.arange(n))[0]:+.3f}, SE {np.std(d):.3f}')
