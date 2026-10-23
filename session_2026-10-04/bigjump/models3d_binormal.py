# Equal-variance binormal blend model (same family as the brief's closed form):
# score_i | class ~ N(+-d_i/2, 1), within-class correlation matrix R; best linear blend
# has d* = sqrt(d' R^-1 d); AUC = Phi(d/sqrt2). Also reports the optimal weights.
import numpy as np
from scipy.stats import norm
def d(a): return np.sqrt(2)*norm.ppf(a)
def A(dd): return norm.cdf(dd/np.sqrt(2))
def blend(aucs, R):
    dv=np.array([d(a) for a in aucs]); R=np.array(R,float)
    w=np.linalg.solve(R,dv); ds=np.sqrt(dv@w); return A(ds), w/w.sum()
P=0.943
print('calibration (brief): F .935@.80 ->', round(blend([P,.935],[[1,.8],[.8,1]])[0],4))
print('calibration vs MEASURED: crop .924@.80 ->', round(blend([P,.924],[[1,.8],[.8,1]])[0],4), '(measured 0.941-0.942 at W .15-.25)')
print('calibration vs MEASURED: Tian .927@.835 ->', round(blend([P,.927],[[1,.835],[.835,1]])[0],4), '(measured 0.942 at b .10, 0.941 at b .20)')
print()
print('Two-way P + V (video member alone):')
for v in (.910,.915,.920,.925,.930,.935):
    row=[]
    for r in (.65,.70,.75,.80,.85):
        row.append('%.4f'%blend([P,v],[[1,r],[r,1]])[0])
    print(' V=%.3f'%v,' rho .65/.70/.75/.80/.85:',' '.join(row))
print()
print('Three-way P + F + V (F = ConvNeXt fleet):')
for F,rPF in ((.930,.82),(.935,.80),(.935,.85)):
    for V,rPV in ((.915,.72),(.925,.72),(.925,.78),(.935,.75)):
        for rFV in (.75,.85):
            R=[[1,rPF,rPV],[rPF,1,rFV],[rPV,rFV,1]]
            a2,_=blend([P,F],[[1,rPF],[rPF,1]])
            a3,w=blend([P,F,V],R)
            print(' F %.3f@%.2f  V %.3f@%.2f rFV %.2f | P+F %.4f  P+F+V %.4f  (+%.4f)  w=%s'%(F,rPF,V,rPV,rFV,a2,a3,a3-a2,np.round(w,2)))
