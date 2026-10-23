# (4) macro gain of a per-study fallback when one member silently degrades on X% of studies.
# Members: TTR s1044 OOF (slot-picker Raptor-recipe proxy) and R7 OOF (set-model R6-family proxy); labels report-derived.
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
B='C:/Users/eeshs/Downloads/rsna-knee/'
LAB=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
J=pd.read_pickle('JT.pkl')
Y=(pd.read_csv(B+'session_2026-10-05/jump2/labels_in/jev_hybrid_binary.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').reindex(J.index)[LAB]>0.5).astype(float).values
pct=lambda x: pd.Series(x).rank(pct=True).values
RT=np.column_stack([pct(J['t|'+c].values) for c in LAB]); RO=np.column_stack([pct(J['p|'+c].values) for c in LAB])
N=len(J); rng=np.random.default_rng(3)
W=0.5
def macro(S): return np.mean([roc_auc_score(Y[:,j],S[:,j]) for j in range(12)])
base=macro(W*RT+(1-W)*RO)
print('clean blend macro %.4f | TTR %.4f | R7 %.4f'%(base,macro(RT),macro(RO)))
rows=[]
for X in [0.005,0.01,0.02,0.05,0.10]:
    for sev,desc in [(1.0,'uninformative'),(0.5,'half-informative'),('const','constant output')]:
        gs=[];ls=[]
        for rep in range(6):
            bad=rng.random(N)<X
            RTd=RT.copy()
            for j in range(12):
                if sev=='const':
                    raw=J['t|'+LAB[j]].values.copy(); raw[bad]=np.median(raw); RTd[:,j]=pct(raw)   # degraded member returns one value
                else:
                    noise=rng.random(N)
                    RTd[bad,j]=(1-sev)*RT[bad,j]+sev*noise[bad]
            deg=W*RTd+(1-W)*RO
            fb=deg.copy(); fb[bad]=RO[bad]                       # per-study fallback: degraded rows take the other member's rank
            gs.append(macro(fb)-macro(deg)); ls.append(base-macro(deg))
        rows.append(dict(X=X,degradation=desc,loss_vs_clean=np.mean(ls),fallback_gain=np.mean(gs),sd=np.std(gs)))
T=pd.DataFrame(rows); print(T.to_string(float_format=lambda v:'%+.5f'%v)); T.to_csv('fallback_gain_sim.csv',index=False)
