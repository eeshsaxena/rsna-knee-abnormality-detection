# MODEL check of jump2/blend_scen.py: K2 rho range under display rounding, and a K3 consistency test.
import numpy as np
from scipy.stats import norm
def mu(a): return np.sqrt(2)*norm.ppf(a)
def auc_lin(mus, C, w):
    w=np.asarray(w,float); m=w@mus; v=w@C@w
    return norm.cdf(m/np.sqrt(2*v))
def blend2(P,S,rho,W):
    C=np.array([[1,rho],[rho,1]]); return auc_lin(np.array([mu(P),mu(S)]),C,[1-W,W])
# 1) K2: parent shown .943, solo shown .932, blend W.30 shown .943 -> rho range over rounding boxes
grid=np.linspace(0.80,0.99,191)
ok=[]
for rho in grid:
    hit=False
    for P in np.linspace(.9425,.94349,6):
        for S in np.linspace(.9315,.93249,6):
            b=blend2(P,S,rho,.30)
            if .9425<=b<.9435: hit=True
    if hit: ok.append(rho)
print('K2-consistent rho_eff (any P in [.9425,.9435), S in [.9315,.9325)): %.3f .. %.3f'%(min(ok),max(ok)))
okc=[r for r in grid if .9425<=blend2(.943,.932,r,.30)<.9435]
print('K2-consistent rho_eff at P=.943,S=.932 exactly: %.3f .. %.3f'%(min(okc),max(okc)))
# best-W lift at the K2 rho range for solo .932 (what K2 implies the max lift of THAT member is)
for rho in [min(ok),0.91,max(ok)]:
    ws=np.linspace(0,1,201); b=[blend2(.943,.932,rho,w) for w in ws]; i=int(np.argmax(b))
    print(f'  rho {rho:.3f}: best W {ws[i]:.2f}, lift {b[i]-.943:+.4f}')
# 2) K3: parent + mean(F1 mix50 [.932], TTR v4 [.923]) at W .25 shown .943.
#    r_FT = member-member within-class corr ~ .92-.94 (Spearman proxy). rho_FP from K2.
print()
print('K3 test: which rho_TP (TTR v4 vs parent) is consistent with blend W.25 shown .943?')
for rFP in [0.91,0.92]:
    for rFT in [0.92,0.94]:
        cons=[]
        for rTP in np.linspace(0.70,0.99,59):
            C=np.array([[1,rFP,rTP],[rFP,1,rFT],[rTP,rFT,1]])
            if np.min(np.linalg.eigvalsh(C))<=0: continue
            mus=np.array([mu(.943),mu(.932),mu(.923)])
            b=auc_lin(mus,C,[.75,.125,.125])
            solo=auc_lin(mus,C,[0,.5,.5])
            if .9425<=b<.9435: cons.append((rTP,solo,b))
        if cons:
            print(f'  rFP {rFP} rFT {rFT}: rho_TP in [{cons[0][0]:.3f},{cons[-1][0]:.3f}] ; implied mean-member solo {cons[0][1]:.4f}')
        else:
            print(f'  rFP {rFP} rFT {rFT}: none')
# 3) what a "decorrelated" member at rho .84 / .80 would have needed to show K3-like null
print()
for rho in [0.80,0.84,0.88]:
    for S in [0.923,0.926,0.930]:
        print(f'  solo {S} rho {rho}: W.25 blend {blend2(.943,S,rho,.25):.4f}  best lift {max(blend2(.943,S,rho,w) for w in np.linspace(0,1,101))-.943:+.4f}')
