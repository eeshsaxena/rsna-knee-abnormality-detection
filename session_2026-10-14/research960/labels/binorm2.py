# MODEL: fleet SOLO strength needed for a blend gain on top of HR7-RR (equal-variance binormal, one "average label").
# Optimal linear blend of two scores with separations dB, dF and within-class correlation r:
#   d*^2 = (dB^2 - 2 r dB dF + dF^2) / (1 - r^2)   (valid while w* in [0,1])
from scipy.stats import norm
from scipy.optimize import brentq
import numpy as np
d=lambda a: np.sqrt(2)*norm.ppf(a); A=lambda x: norm.cdf(x/np.sqrt(2))
def blend(aB,aF,r,w): return A(((1-w)*d(aB)+w*d(aF))/np.sqrt((1-w)**2+w**2+2*w*(1-w)*r))
def best(aB,aF,r):
    dB,dF=d(aB),d(aF); ds=np.sqrt(max(dB**2,(dB**2-2*r*dB*dF+dF**2)/(1-r**2)))
    wF=(dF-r*dB); wB=(dB-r*dF); w=max(0,wF)/(max(0,wF)+max(0,wB)) if (max(0,wF)+max(0,wB))>0 else 0
    return w, A(ds)-aB
aB=.9565
for g20 in [.0001,.0002,.0004]:
    r=brentq(lambda r: blend(aB,.945,r,.2)-aB-g20, .5,.999)
    w,g=best(aB,.945,r)
    print(f"calib: F SOLO .945, gain at w .2 {g20:+.4f} -> r {r:.3f}; w* {w:.2f} gain* {g:+.4f}")
    for target in [.001,.002,.004]:
        aF=brentq(lambda a: best(aB,a,r)[1]-target, .946,.9999)
        print(f"   gain* {target:+.3f} needs F SOLO {aF:.4f} (+{aF-.945:.4f}), w* {best(aB,aF,r)[0]:.2f}")
    for r2 in [r-.02, r-.04]:
        print(f"   r {r2:.3f}: F .945 gain* {best(aB,.945,r2)[1]:+.4f} | F .948 gain* {best(aB,.948,r2)[1]:+.4f}")
print("\nPrediction for the pending reads (gain over HR7-RR base):")
for g20 in [.0001,.0002,.0004]:
    r=brentq(lambda r: blend(aB,.945,r,.2)-aB-g20, .5,.999)
    print(f"  r {r:.3f}: w.2 {blend(aB,.945,r,.2)-aB:+.5f}  w.3 {blend(aB,.945,r,.3)-aB:+.5f}  w.4 {blend(aB,.945,r,.4)-aB:+.5f}")
for aF,r in [(.948,.90),(.950,.90),(.946,.88)]:
    print(f"  alt F {aF} r {r}: w.2 {blend(aB,aF,r,.2)-aB:+.5f} w.3 {blend(aB,aF,r,.3)-aB:+.5f} w.4 {blend(aB,aF,r,.4)-aB:+.5f}")
