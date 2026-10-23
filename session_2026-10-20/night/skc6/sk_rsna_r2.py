# skeptic C6' (10-20 night, Claude). Local only, no OAI. RSNA-population ruler for LM <- MM: R7 OOF on non-gold train studies,
# report v2 labels (>= .5), 2x2 (MM, LM) cells reweighted to gold74 expert log OR shifted by k sd (train marginals kept), exactly the
# box recipe (rho_ns, q = cov(zo,y)/cov(zt,y), gap, dAUC_t). Plus the raw report regime. Question: at RSNA composition, does the a .05
# re-dose gain at the gaps this population produces, and does dAUC-at-matched-gap agree with the OAI box curve?
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
from scipy.optimize import brentq
R='C:/Users/eeshs/Downloads/rsna-knee/'
oof=pd.read_csv(R+'session_2026-10-18/next/g58/R7OOF.csv').set_index('StudyInstanceUID')
g=pd.read_csv(R+'session_2026-10-18/next/g58/gold74_eval_set.csv').set_index('StudyInstanceUID')
rep=pd.read_csv(R+'_labels/rsna-knee-llm-labels/report_labels_v2.csv').set_index('StudyInstanceUID')
ids=[i for i in oof.index if i not in g.index and i in rep.index]
Ft=oof.loc[ids,'p|Lateral Meniscus'].values.astype(float); Fo=oof.loc[ids,'p|Medial Meniscus'].values.astype(float)
ya=(rep.loc[ids,'Medial Meniscus']>=.5).astype(int).values; yb=(rep.loc[ids,'Lateral Meniscus']>=.5).astype(int).values
pct=lambda v: rankdata(v)/len(v); ns=lambda v: norm.ppf((rankdata(v)-.5)/len(v))
def auc(s,y):
    r=rankdata(s); p=y.sum(); q=len(y)-p; return (r[y==1].sum()-p*(p+1)/2)/(p*q)
def or_cells(p1,p2,lor):
    lo=max(0.,p1+p2-1)+1e-9; hi=min(p1,p2)-1e-9
    f=lambda p11: np.log(p11)+np.log(1-p1-p2+p11)-np.log(p1-p11)-np.log(p2-p11)-lor
    p11=brentq(f,lo,hi); return np.array([1-p1-p2+p11,p2-p11,p1-p11,p11])
key=ya*2+yb; po=np.bincount(key,minlength=4)/len(key)
cs=np.array([25,13,19,16],float)+.5; lor=np.log(cs[3]*cs[0]/(cs[1]*cs[2])); lsd=np.sqrt((1/cs).sum())
rl=np.log((((key==3).sum()+.5)*((key==0).sum()+.5))/(((key==1).sum()+.5)*((key==2).sum()+.5)))
print('n %d | report MM prev %.3f LM prev %.3f | report log OR %+.2f | gold74 expert log OR %+.2f (sd %.2f)'%(len(ids),ya.mean(),yb.mean(),rl,lor,lsd))
AV=(-.05,.05,.1); rng=np.random.default_rng(1020); B=300
outs={x: pct(Ft)-x*pct(Fo) for x in AV}
def run(w,lab):
    p=w/w.sum(); ess=1/(p**2).sum(); Rr,Q,D=[],[],{x:[] for x in AV}
    for _ in range(B):
        ii=rng.choice(len(ids),len(ids),p=p); y=yb[ii]
        zt=ns(Ft[ii]); zo=ns(Fo[ii]); ct=np.cov(zt,y)[0,1]
        Rr.append(np.corrcoef(zt,zo)[0,1]); Q.append(np.cov(zo,y)[0,1]/ct)
        b0=auc(Ft[ii],y)
        for x in AV: D[x].append(auc(outs[x][ii],y)-b0)
    print('%-22s ess %4d | rho %.3f q %.3f gap %+.3f | '%(lab,ess,np.mean(Rr),np.mean(Q),np.mean(Rr)-np.mean(Q))+
          ' '.join('a%+.2f %+.4f (sd %.4f, P>0 %.2f)'%(x,np.mean(D[x]),np.std(D[x]),np.mean(np.array(D[x])>0)) for x in AV))
run(np.ones(len(ids)),'raw report regime')
for k in (-2.,-1.5,-1.,-.5,0.,.5,1.):
    tg=or_cells(ya.mean(),yb.mean(),lor+k*lsd); w=(tg/np.maximum(po,1e-12))[key]; run(w,'gold74 OR k %+.1f'%k)
# also: reweight to gold58 expert prevalence + gold74 OR (enriched composition)
pm=g.loc[g.source=='original_gold58','Medial Meniscus'].mean(); pl=g.loc[g.source=='original_gold58','Lateral Meniscus'].mean()
for k in (-1.,0.):
    tg=or_cells(pm,pl,lor+k*lsd); w=(tg/np.maximum(po,1e-12))[key]; run(w,'g58 prev, OR k %+.1f'%k)
