import numpy as np
from scipy.stats import norm
def mu(a): return np.sqrt(2)*norm.ppf(a)
def blend(a1,a2,rho,w):
    m=(1-w)*mu(a1)+w*mu(a2); v=(1-w)**2+w**2+2*w*(1-w)*rho
    return norm.cdf(m/np.sqrt(2*v))
def nblend(aucs,ws,R):  # multi-member, equal-variance binormal
    aucs=np.array(aucs); ws=np.array(ws)/np.sum(ws); m=ws@mu(aucs); v=ws@R@ws
    return norm.cdf(m/np.sqrt(2*v))
rhos=np.linspace(0.6,0.999,800)
def implied(p,a,reads):
    sse=[sum((blend(p,a,r,w)-b)**2 for w,b in reads) for r in rhos]; return rhos[int(np.argmin(sse))]
print('crop K24 reads (.942@.15,.941@.25) implied rho by assumed served solo:')
for a in (0.900,0.905,0.910,0.915,0.920,0.924): print('  solo',a,'rho',round(implied(.943,a,[(0.15,.942),(0.25,.941)]),3))
print('Aastik fleet (+0.002 at W .5/.6) implied rho by solo:')
for a in (0.925,0.928,0.931,0.934,0.937): print('  solo',a,'rho',round(implied(.943,a,[(0.5,.945),(0.6,.945)]),3))
print('Tian v09r (.942@.1,.941@.2) rho', round(implied(.942,.927,[(0.1,.942),(0.2,.941)]),3))
# seed averaging check vs Tian MEASURED: 0.927+0.927 rho .952 -> 0.930 measured
print('2 seeds .927 rho .952 ->', round(nblend([.927,.927],[1,1],np.array([[1,.952],[.952,1]])),4), '(MEASURED 0.930)')
print('c03 pair .932 + R34 .921 rho .86, 1/3 weight ->', round(nblend([.932,.921],[2,1],np.array([[1,.86],[.86,1]])),4),'(MEASURED 0.932)')
# own-ensemble build-up: N seeds of family with single solo a, seed rho rs; F2 family solo a2, cross rho rc
def fam(a,n,rs):
    R=np.full((n,n),rs); np.fill_diagonal(R,1); return nblend([a]*n,[1]*n,R)
for a in (0.925,0.929,0.933):
    for n in (1,2,3,5,6):
        print(f'F1 single {a} x{n} seeds (rho .95): {fam(a,n,0.95):.4f}')
def two_fam(a1,n1,a2,n2,rs,rc,w2):
    n=n1+n2; R=np.full((n,n),rc); R[:n1,:n1]=rs; R[n1:,n1:]=rs; np.fill_diagonal(R,1)
    ws=[(1-w2)/n1]*n1+[w2/n2]*n2; return nblend([a1]*n1+[a2]*n2,ws,R)
print('two families: F1 x6 + F2 x4 (seed rho .95)')
for a1,a2 in ((0.929,0.929),(0.929,0.925),(0.933,0.930)):
    for rc in (0.86,0.90,0.93):
        best=max((two_fam(a1,6,a2,4,.95,rc,w),w) for w in np.linspace(0,1,21))
        print(f'  F1 {a1} F2 {a2} cross-rho {rc}: best {best[0]:.4f} at wF2 {best[1]:.2f} | F1 alone {fam(a1,6,.95):.4f}')
print('parent blend of own ensemble (parent .943):')
for own in (0.933,0.936,0.938,0.940,0.942,0.945):
    row=[]
    for r in (0.80,0.84,0.88,0.91):
        ws=np.linspace(0,1,101); b=[blend(.943,own,r,w) for w in ws]; i=int(np.argmax(b)); row.append(f'r{r}: W{ws[i]:.2f} +{b[i]-.943:.4f}')
    print('  own',own,' | '.join(row))
