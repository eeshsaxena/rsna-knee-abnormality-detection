# MODEL: equal-variance binormal, same formula as blend_scen.py
import numpy as np
from scipy.stats import norm
def mu(a): return np.sqrt(2)*norm.ppf(a)
def blend(a1,a2,rho,w):
    m=(1-w)*mu(a1)+w*mu(a2); v=(1-w)**2+w**2+2*w*(1-w)*rho
    return norm.cdf(m/np.sqrt(2*v))
def best(a1,a2,rho):
    ws=np.linspace(0,1,101); b=[blend(a1,a2,rho,w) for w in ws]; i=int(np.argmax(b)); return ws[i], b[i]
P=.943
print('A. parent .943 + own fleet: true blend at W .15/.20/.30, best W')
for s in [.933,.936,.939,.942]:
    for r in [.90,.92,.94]:
        bw,bb=best(P,s,r)
        print(f' s {s:.3f} r {r:.2f}: W.15 {blend(P,s,r,.15):.4f} W.20 {blend(P,s,r,.20):.4f} W.30 {blend(P,s,r,.30):.4f} | W* {bw:.2f} {bb:.4f}')
print('B. merge: partner own-model fleet vs our parent .943 (best W)')
for q in [.945,.947,.949]:
    for r in [.75,.80,.85,.90]:
        bw,bb=best(P,q,r)
        print(f' partner {q:.3f} r {r:.2f}: W*(partner) {bw:.2f} -> {bb:.4f}')
print('C. rho needed for .958 with partner .949 at best W')
for r in np.arange(.60,.80,.02):
    bw,bb=best(P,.949,r); print(f' r {r:.2f}: {bb:.4f}')
print('D. single member (J1) W .20 and W .50 true blend vs solo s and r')
for s in [.929,.931,.933,.935,.937]:
    print(f' s {s:.3f}: ' + ' | '.join(f'r {r:.2f} W.20 {blend(P,s,r,.20):.4f} W.50 {blend(P,s,r,.50):.4f} W* {best(P,s,r)[0]:.2f}' for r in [.88,.91,.94]))
print('E. 5f fleet with TTA/M1, s range, W .20/.25/.30')
for s in [.934,.936,.938,.940]:
    print(f' s {s:.3f}: ' + ' | '.join(f'r {r:.2f} W.20 {blend(P,s,r,.20):.4f} W.25 {blend(P,s,r,.25):.4f} W.30 {blend(P,s,r,.30):.4f}' for r in [.90,.92,.94]))
