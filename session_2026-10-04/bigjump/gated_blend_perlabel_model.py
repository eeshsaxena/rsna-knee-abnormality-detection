# MODEL: per-label binormal; student gains land where the gated target gains (gold58 per-label target deltas x transfer).
import numpy as np
from scipy.stats import norm
d=lambda a: np.sqrt(2)*norm.ppf(np.clip(a,1e-6,1-1e-6)); auc=lambda x: norm.cdf(x/np.sqrt(2))
def blend(P,M,R,w): return np.mean([auc(((1-w)*d(p)+w*d(m))/np.sqrt((1-w)**2+w**2+2*w*(1-w)*r)) for p,m,r in zip(P,M,R)])
def best(P,M,R):
    W=np.linspace(0,.8,161); v=[blend(P,M,R,w) for w in W]; i=int(np.argmax(v)); return v[i]-np.mean(P),W[i]
gp=np.array([.978,.993,.966,.870,.984,.832,.834,.975,.830,.971,.927,.935])
PA=np.clip(.943+(gp-gp.mean())*0.6,.5,.995); PB=np.full(12,.943)
# gold58 target-level per-label deltas of gated .75/0 vs v4 (measured today), and of flat mix50 (Raptor proxy)
dg=np.array([0,0,.001,.011,.001,.007,.018,0,.038,.024,.008,.114]); dm=np.array([.001,.027,.026,.047,.050,.003,-.006,.091,.051,.027,.047,.101])
for S0 in (.92,.93):
  for tr in (0.28,0.5):
    for nm,P in (("A gold-like parent",PA),("B flat parent",PB)):
        M0=np.full(12,S0)
        base=best(P,M0,np.full(12,.80)); g=best(P,np.clip(M0+tr*dg,0,.995),np.full(12,.816)); m=best(P,np.clip(M0+tr*dm,0,.995),np.full(12,.857))
        print(f"S0 {S0} transfer {tr} {nm:18s} | pure v4 +{base[0]:.4f} | gated (solo {S0+tr*dg.mean():.4f}, rho .816) +{g[0]:.4f} | mix50 (solo {S0+tr*dm.mean():.4f}, rho .857) +{m[0]:.4f}")
