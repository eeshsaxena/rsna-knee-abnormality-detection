import numpy as np, pandas as pd, json
from scipy.stats import rankdata
G='C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58'
L=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
g=pd.read_csv(G+"/gold74_eval_set.csv"); g=g[g.source=="original_gold58"].reset_index(drop=True); ids=g.StudyInstanceUID.astype(str).tolist()
lg=lambda p: np.log(np.clip(p,1e-7,1-1e-7)/(1-np.clip(p,1e-7,1-1e-7)))
R6=lg(pd.read_csv(G+"/R6_gold58_serve.csv").set_index("StudyInstanceUID").loc[ids,L].values)
S6=lg(pd.read_csv(G+"/n42gpu_gold58.csv").set_index("StudyInstanceUID").loc[ids,L].values)
R7=pd.read_csv(G+"/R7OOF.csv").set_index("StudyInstanceUID").loc[ids,["z|"+l for l in L]].values
R384=lg(np.load(G+"/gold58_r384_kernelserve_cpu.npy"))
def zs(M): return (M-M.mean(0))/M.std(0)
A={'R6':zs(R6),'S6':zs(S6),'R7':zs(R7),'r384':zs(R384)}
# per study: asymmetry vector (MedMen-LatMen, MedOA-LatOA) per member
asym={k:np.column_stack([v[:,2]-v[:,3],v[:,4]-v[:,5]]) for k,v in A.items()}
rows=[]
for i,s in enumerate(ids):
    r={'sid':s[-12:]}
    for k in asym: r[k+'_men']=round(asym[k][i,0],2); r[k+'_oa']=round(asym[k][i,1],2)
    rows.append(r)
df=pd.DataFrame(rows)
# flag studies where R6's asymmetry has the opposite sign of S6/R7/r384 in both pairs with magnitude
for a,b in [('R6','S6'),('R6','r384'),('S6','r384'),('R6','R7')]:
    opp=(np.sign(asym[a])!=np.sign(asym[b])).all(1)&(np.abs(asym[a]).min(1)>0.5)&(np.abs(asym[b]).min(1)>0.5)
    print(a,'vs',b,'both pairs opposite & |asym|>.5:',int(opp.sum()), [ids[i][-12:] for i in np.where(opp)[0]])
print(df.to_string())
