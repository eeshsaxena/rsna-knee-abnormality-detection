# Closed-form (equal-variance binormal) scenario numbers for BIG_JUMP_PLAN.md. MODEL only.
import numpy as np
from scipy.stats import norm
def mu(a): return np.sqrt(2)*norm.ppf(a)
def nblend(aucs,ws,R):
    aucs=np.array(aucs,float); ws=np.array(ws,float)/np.sum(ws); return norm.cdf((ws@mu(aucs))/np.sqrt(2*(ws@R@ws)))
def blend(p,a,r,w): return nblend([p,a],[1-w,w],np.array([[1,r],[r,1]]))
def best(p,a,r):
    ws=np.linspace(0,1,101); b=[blend(p,a,r,w) for w in ws]; i=int(np.argmax(b)); return ws[i],b[i]
def fams(groups, within, cross):
    # groups: list of (solo, n, weight); within-family rho per group; cross rho between groups
    aucs=[];ws=[];fam=[]
    for gi,(a,n,w) in enumerate(groups):
        aucs+= [a]*n; ws+=[w/n]*n; fam+=[gi]*n
    N=len(aucs); R=np.empty((N,N))
    for i in range(N):
        for j in range(N):
            R[i,j]=1 if i==j else (within[fam[i]] if fam[i]==fam[j] else cross)
    return nblend(aucs,ws,R)
TTA=0.0015  # MEASURED-analog +0.0016 in a blend (Tian), applied as a flat add to the own solo
print("Own-ensemble build-up (MODEL; member solos are GUESS inputs)")
kill=fams([(0.929,6,0.65),(0.926,3,0.35)],[0.93,0.95],0.90)
print(f"KILL  F1x6 (.929, within .93 incl. 3 label variants) + EV2Sx3 (.926, cross .90): {kill:.4f} +TTA {kill+TTA:.4f}")
pc=fams([(0.934,5,0.55),(0.929,4,0.25),(0.926,3,0.20)],[0.95,0.93,0.95],0.90)
print(f"PASS-central TTRx5 (.934) + F1x4 (.929) + EV2Sx3 (.926), cross .90: {pc:.4f} +TTA {pc+TTA:.4f}")
pu=fams([(0.938,5,0.6),(0.929,4,0.2),(0.926,3,0.2)],[0.95,0.93,0.95],0.88)
print(f"PASS-upper   TTRx5 (.938) + F1x4 + EV2Sx3, cross .88: {pu:.4f} +TTA {pu+TTA:.4f}")
print()
print("Parent .943 + own at best W, by effective rho (gold58 rho + .04/.07/.10):")
for name,own in (("KILL",kill+TTA),("PASS-central",pc+TTA),("PASS-upper",pu+TTA)):
    row=[]
    for r in (0.80,0.84,0.88):
        w,b=best(0.943,own,r); row.append(f"rho {r:.2f}: W {w:.2f} -> {b:.4f} (+{b-0.943:.4f})")
    print(f"  {name:13s} own {own:.4f} | "+" | ".join(row))
print()
print("Own solo needed for a displayed 0.950 (public 0.9495+) at the best W:")
for r in (0.78,0.80,0.82,0.84,0.86,0.88,0.90):
    for a in np.arange(0.925,0.960,0.0005):
        if best(0.943,a,r)[1]>=0.9495: print(f"  eff rho {r:.2f}: own solo >= {a:.4f}"); break
print()
print("Fixed W .30 vs W* (cost of a mis-set W), own .938 rho .84:")
for w in (0.25,0.30,0.35,0.40,0.45,0.50): print(f"  W {w:.2f}: {blend(0.943,0.938,0.84,w):.4f}")
