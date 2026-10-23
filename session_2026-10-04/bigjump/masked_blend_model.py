# MODEL (equal-variance binormal, per label): flat global W vs headroom-masked W for an own member.
import numpy as np
from scipy.stats import norm
def d(a): return np.sqrt(2)*norm.ppf(a)
def auc(x): return norm.cdf(x/np.sqrt(2))
def blend(ap,am,rho,w):
    dp,dm=d(ap),d(am); return auc(((1-w)*dp+w*dm)/np.sqrt((1-w)**2+w**2+2*w*(1-w)*rho))
weak=[3,5,6,8,10,11]  # LatMen, LatOA, PFOA, Synovitis, Contusion, Fracture
def scen(parent,member,rho,label):
    parent=np.array(parent); member=np.array(member)
    W=np.linspace(0,0.8,81)
    flat=[np.mean([blend(parent[j],member[j],rho,w) for j in range(12)]) for w in W]
    i=int(np.argmax(flat))
    mask=[np.mean([blend(parent[j],member[j],rho,w) if j in weak else parent[j] for j in range(12)]) for w in W]
    k=int(np.argmax(mask))
    # pre-registered (not optimised) weights
    f30=np.mean([blend(parent[j],member[j],rho,.30) for j in range(12)])
    m45=np.mean([blend(parent[j],member[j],rho,.45) if j in weak else parent[j] for j in range(12)])
    print(f"{label:58s} parent {parent.mean():.4f} member {member.mean():.4f} | flat best {flat[i]-parent.mean():+.4f}@{W[i]:.2f} flat.30 {f30-parent.mean():+.4f} | mask best {mask[k]-parent.mean():+.4f}@{W[k]:.2f} mask.45 {m45-parent.mean():+.4f}")
# parent per-label: A = gold58-like spread rescaled to macro .943; B = flat .943
gp=np.array([.978,.993,.966,.870,.984,.832,.834,.975,.830,.971,.927,.935])
A=np.clip(.943+(gp-gp.mean())*0.6,0.5,0.995); B=np.full(12,.943)
for solo in [.925,.935,.945]:
    for rho in [.75,.80,.85]:
        # member: same relative profile as the crop ensemble (better on weak labels), rescaled to the solo macro
        cp=np.array([.923,.904,.852,.830,.968,.864,.864,.919,.790,.948,.923,.876])
        mA=np.clip(solo+(cp-cp.mean())*0.6,0.5,0.995); mB=np.full(12,solo)
        scen(A,mA,rho,f"A gold-like spread | member {solo} rho {rho}")
        scen(B,mB,rho,f"B flat parent      | member {solo} rho {rho}")
