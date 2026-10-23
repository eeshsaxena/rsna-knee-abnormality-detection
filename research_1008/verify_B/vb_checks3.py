"""verify_B checks 3: display distribution of K9N-a and F+N (P-free) under an r_FN prior, r_PN/r_PF uncertainty,
model optimism U(0,.001) and N/F/P display rounding. MODEL/GUESS. Light numpy."""
import numpy as np
from scipy.stats import norm
rng=np.random.default_rng(7); M=40000
d=lambda a: np.sqrt(2)*norm.ppf(a)
def bk(s,R,w):
    w=np.asarray(w,float); dv=np.array([d(x) for x in s]); return norm.cdf((w@dv)/np.sqrt(2*(w@R@w)))
for mu in (.89,.905,.92):
    outa=[];outf=[]
    for _ in range(M//10):
        rFN=np.clip(rng.normal(mu,.025),.84,.97); rPN=rng.uniform(.925,.965); rPF=rng.uniform(.93,.96)
        N=rng.uniform(.9485,.9495); F=rng.uniform(.9435,.9445); P=rng.uniform(.9425,.9435)
        R=np.array([[1,rPF,rPN],[rPF,1,rFN],[rPN,rFN,1]])
        if np.min(np.linalg.eigvalsh(R))<=0: continue
        opt=rng.uniform(0,.001)
        outa.append(bk((P,F,N),R,(.10,.30,.60))-opt); outf.append(bk((P,F,N),R,(0,.40,.60))-opt)
    for nm,o in (('K9N-a',np.array(outa)),('F+N',np.array(outf))):
        disp=np.round(o,3)
        print(f'r_FN~N({mu},.025) {nm}: mean true {o.mean():.4f} sd {o.std():.4f} | P(disp>=.952) {np.mean(disp>=.952):.2f} .951 {np.mean(np.isclose(disp,.951)):.2f} .950 {np.mean(np.isclose(disp,.950)):.2f} <=.949 {np.mean(disp<=.949):.2f} | P(true>.9500) {np.mean(o>.95):.2f}')
