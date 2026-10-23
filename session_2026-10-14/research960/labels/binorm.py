# MODEL: equal-variance binormal blend arithmetic (single "average label" approximation)
from scipy.stats import norm
import numpy as np
def d(a): return np.sqrt(2)*norm.ppf(a)
def A(dd): return norm.cdf(dd/np.sqrt(2))
def blend(aB,aF,r,w):
    dB,dF=d(aB),d(aF)
    return A(((1-w)*dB+w*dF)/np.sqrt((1-w)**2+w**2+2*w*(1-w)*r))
def best(aB,aF,r):
    ws=np.linspace(0,1,1001); v=[blend(aB,aF,r,w) for w in ws]; i=int(np.argmax(v)); return ws[i],v[i]
aB=.9565
print("Base B (HR7-RR) AUC", aB)
for aF in [.944,.946,.948,.950,.952,.954]:
    for r in [.80,.85,.88,.90,.92]:
        w,v=best(aB,aF,r)
        g20=blend(aB,aF,r,.2)-aB; g30=blend(aB,aF,r,.3)-aB
        print(f"F {aF:.3f} r {r:.2f}: w* {w:.2f} gain* {v-aB:+.4f} | w.2 {g20:+.4f} w.3 {g30:+.4f}")
