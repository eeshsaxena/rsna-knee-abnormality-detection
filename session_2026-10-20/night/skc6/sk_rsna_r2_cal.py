# skeptic C6' calibration check of the RSNA-R2 ruler on the trio (LOA <- PFOA, MOA <- LOA) and the other meniscus direction.
# Local only. Same recipe as sk_rsna_r2.py. Public-implied c4 trio gap .128 (CLAIMED val/implied2.md); box D5 at k0 LOA<-PFOA gap .193.
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
from scipy.optimize import brentq
R='C:/Users/eeshs/Downloads/rsna-knee/'
oof=pd.read_csv(R+'session_2026-10-18/next/g58/R7OOF.csv').set_index('StudyInstanceUID')
g=pd.read_csv(R+'session_2026-10-18/next/g58/gold74_eval_set.csv').set_index('StudyInstanceUID')
rep=pd.read_csv(R+'_labels/rsna-knee-llm-labels/report_labels_v2.csv').set_index('StudyInstanceUID')
ids=[i for i in oof.index if i not in g.index and i in rep.index]
pct=lambda v: rankdata(v)/len(v); ns=lambda v: norm.ppf((rankdata(v)-.5)/len(v))
def auc(s,y):
    r=rankdata(s); p=y.sum(); q=len(y)-p; return (r[y==1].sum()-p*(p+1)/2)/(p*q)
def or_cells(p1,p2,lor):
    lo=max(0.,p1+p2-1)+1e-9; hi=min(p1,p2)-1e-9
    f=lambda p11: np.log(p11)+np.log(1-p1-p2+p11)-np.log(p1-p11)-np.log(p2-p11)-lor
    p11=brentq(f,lo,hi); return np.array([1-p1-p2+p11,p2-p11,p1-p11,p11])
def g74cells(a_,b_):
    gg=g[[a_,b_]].dropna(); k=(gg[a_].astype(int)*2+gg[b_].astype(int)).values; return np.bincount(k,minlength=4).astype(float)
rng=np.random.default_rng(7); B=200; AV=(.05,.1,.2)
for t,o in [('Lateral OA','PF OA'),('Medial OA','Lateral OA'),('PF OA','Lateral OA'),('Medial Meniscus','Lateral Meniscus'),('Lateral Meniscus','Medial Meniscus')]:
    a_,b_=t,o
    Ft=oof.loc[ids,'p|'+t].values.astype(float); Fo=oof.loc[ids,'p|'+o].values.astype(float)
    ya=(rep.loc[ids,a_]>=.5).astype(int).values; yb=(rep.loc[ids,b_]>=.5).astype(int).values
    key=ya*2+yb; po=np.bincount(key,minlength=4)/len(key)
    cs=g74cells(a_,b_)+.5; lor=np.log(cs[3]*cs[0]/(cs[1]*cs[2])); lsd=np.sqrt((1/cs).sum())
    rl=np.log((((key==3).sum()+.5)*((key==0).sum()+.5))/(((key==1).sum()+.5)*((key==2).sum()+.5)))
    print('== %s <- %s | report log OR %+.2f | gold74 log OR %+.2f (sd %.2f) | g74 cells %s'%(t,o,rl,lor,lsd,(cs-.5).astype(int).tolist()))
    outs={x: pct(Ft)-x*pct(Fo) for x in AV}
    for k in (-1.,0.,1.):
        tg=or_cells(ya.mean(),yb.mean(),lor+k*lsd); w=(tg/np.maximum(po,1e-12))[key]; p=w/w.sum()
        Rr,Q,D=[],[],{x:[] for x in AV}
        for _ in range(B):
            ii=rng.choice(len(ids),len(ids),p=p); y=ya[ii]
            zt=ns(Ft[ii]); zo=ns(Fo[ii]); ct=np.cov(zt,y)[0,1]
            Rr.append(np.corrcoef(zt,zo)[0,1]); Q.append(np.cov(zo,y)[0,1]/ct); b0=auc(Ft[ii],y)
            for x in AV: D[x].append(auc(outs[x][ii],y)-b0)
        print('  k %+.1f | rho %.3f q %.3f gap %+.3f | '%(k,np.mean(Rr),np.mean(Q),np.mean(Rr)-np.mean(Q))+' '.join('a%+.2f %+.4f'%(x,np.mean(D[x])) for x in AV))
