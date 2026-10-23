# Implied effective rho of measured parent+member blends under the equal-variance binormal model.
import numpy as np
from scipy.stats import norm
def mu(a): return np.sqrt(2)*norm.ppf(a)
def blend(a1,a2,rho,w):
    m=(1-w)*mu(a1)+w*mu(a2); v=(1-w)**2+w**2+2*w*(1-w)*rho
    return norm.cdf(m/np.sqrt(2*v))
rhos=np.linspace(0.60,0.999,400)
# (label, parent_disp, member_disp, [(w, blend_disp)...], gold58 rho if known)
pts=[('Tian v09r 0.927 (Raptor-distilled) on 0.942 fork',0.942,0.927,[(0.10,0.942),(0.20,0.941)],0.835),
     ('Tian c02 arm ~0.913 on 0.942 fork',0.942,0.913,[(0.10,0.942),(0.20,0.939)],None),
     ('Eesh cnx 0.894 on 0.943 parent',0.943,0.894,[(0.50,0.930)],None),
     ('Eesh cnx 0.894 on 0.942 fast base',0.942,0.894,[(0.50,0.930),(0.40,0.935)],None),
     ('Eesh crop K24-served (solo ~0.915-0.924) on 0.943',0.943,0.920,[(0.15,0.942),(0.25,0.941)],0.81),
     ('Eesh mixSA crop K24 (solo ~0.915) on 0.943',0.943,0.915,[(0.25,0.940)],0.84),
     ('Aastik fleet ~0.928 (CLAIM solo) on 0.943 s74',0.943,0.928,[(0.50,0.945),(0.60,0.945)],None),
     ('LeQuangCanh own 0.934 + public 0.936 (CLAIM)',0.936,0.934,[(0.50,0.937),(0.75,0.937)],None)]
print('label | implied rho range (all reads within display rounding, parent/member true = displayed +-0.0005)')
for lab,p,a,reads,g in pts:
    ok=[]
    for r in rhos:
        good=False
        for dp in (-0.0005,0,0.0005):
            for da in (-0.0005,0,0.0005):
                if all(abs(blend(p+dp,a+da,r,w)-(b))<=0.0005+1e-9 for w,b in reads): good=True
        if good: ok.append(r)
    rng=(min(ok),max(ok)) if ok else None
    # point estimate: least squares
    sse=[sum((blend(p,a,r,w)-b)**2 for w,b in reads) for r in rhos]
    rbest=rhos[int(np.argmin(sse))]
    print(f'{lab:55s} | range {None if rng is None else (round(rng[0],3),round(rng[1],3))} | LS {rbest:.3f} | gold58 rho {g}')
# what the model says at gold58 rho for the two cases with known rho
print('model at gold58 rho: v09r W.2 ->', round(blend(.942,.927,.835,.2)-.942,4), '| crop W.25 ->', round(blend(.943,.92,.81,.25)-.943,4))
# forward table with effective rho = gold58 rho + delta
for d in (0.0,0.04,0.07,0.10):
    print('delta_rho',d)
    for a2,rg in ((0.930,0.78),(0.935,0.78),(0.940,0.78),(0.945,0.78),(0.950,0.75)):
        r=min(rg+d,0.995); ws=np.linspace(0,1,101); b=[blend(.943,a2,r,w) for w in ws]; i=int(np.argmax(b))
        print(f'   solo {a2} gold-rho {rg} eff {r:.2f}: W* {ws[i]:.2f} blend {b[i]:.4f} (+{b[i]-.943:.4f}) | at W.35 {blend(.943,a2,r,.35):.4f}')
