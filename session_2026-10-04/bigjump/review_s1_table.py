# S1 (K2 fleetsolo, single F1 fold-0 member) -> P(displayed >= .950 by 10-12). MODEL + GUESS inputs.
# final own = S1_true + gain; gain KILL +.008 / PASS +.012 (P(PASS) .30); S1 read SD .0047 -> S1_true ~ N(S1, .0047),
# extra gain uncertainty SD .003; rho_eff ~ 50/50 U(.80,.88) / U(.84,.92); blend read noise SD .0012; best W known.
import numpy as np
from scipy.stats import norm
rng=np.random.default_rng(1)
def mu(a): return np.sqrt(2)*norm.ppf(a)
ws=np.linspace(0,1,51)
def bestv(own,r):
    m=(1-ws)[:,None]*mu(0.943)+ws[:,None]*mu(own)[None,:]
    v=(1-ws)[:,None]**2+ws[:,None]**2+2*(ws*(1-ws))[:,None]*r[None,:]
    return norm.cdf(m/np.sqrt(2*v)).max(0)
n=20000
for s1 in (0.915,0.918,0.922,0.925,0.928,0.931,0.934):
    st=rng.normal(s1,0.0047,n)
    gain=np.where(rng.random(n)<0.30,0.012,0.008)+rng.normal(0,0.003,n)
    own=np.clip(st+gain,0.90,0.975)
    r=np.where(rng.random(n)<0.5,rng.uniform(0.80,0.88,n),rng.uniform(0.84,0.92,n))
    out=bestv(own,r)+rng.normal(0,0.0012,n)
    print(f"S1 {s1:.3f}: P(>=.950) {np.mean(out>=0.9495):.2f}  P(>=.947) {np.mean(out>=0.9465):.2f}  median {np.median(out):.4f}")
