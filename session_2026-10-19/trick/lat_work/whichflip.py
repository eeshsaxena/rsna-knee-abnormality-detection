import numpy as np, pandas as pd, json
G='C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58'
OUT='C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/trick/lat_work/'
L=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
g=pd.read_csv(G+"/gold74_eval_set.csv"); g=g[g.source=="original_gold58"].reset_index(drop=True); ids=g.StudyInstanceUID.astype(str).tolist()
lg=lambda p: np.log(np.clip(p,1e-7,1-1e-7)/(1-np.clip(p,1e-7,1-1e-7)))
dump=json.load(open(G+'/r6g58_box/run_b/worker0/dump.json'))
R6p=np.array([np.mean(dump[i]['z_plain'],0) for i in ids])
S6=lg(pd.read_csv(G+"/n42gpu_gold58.csv").set_index("StudyInstanceUID").loc[ids,L].values)
R7=pd.read_csv(G+"/R7OOF.csv").set_index("StudyInstanceUID").loc[ids,["z|"+l for l in L]].values
kc=pd.read_csv('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-15/kcrop/out_kc_side_v1/kc_side_train.csv').set_index('StudyInstanceUID').loc[ids]
rs=pd.read_csv(OUT+'report_side.csv').set_index('StudyInstanceUID').loc[ids]
def asym(M): Z=(M-M.mean(0))/M.std(0); return np.column_stack([Z[:,2]-Z[:,3],Z[:,4]-Z[:,5]])
a6,aS,a7=asym(R6p),asym(S6),asym(R7)
score=(a6*aS).sum(1)   # <0 = R6 plain pass M/L opposite to S6
o=np.argsort(score)
for i in o[:6]:
    print(ids[i][-12:], 'R6.S6 %.2f R6.R7 %.2f S6.R7 %.2f' % (score[i], (a6[i]*a7[i]).sum(), (aS[i]*a7[i]).sum()), 'x %.1f tag %s rep %s' % (kc.mean_x.iloc[i], kc.tag.iloc[i], rs.rep_side.iloc[i]), 'asymR6', a6[i].round(2), 'asymS6', aS[i].round(2))
