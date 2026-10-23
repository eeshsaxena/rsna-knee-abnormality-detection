# MODEL: equal-variance binormal blend of parent (a1) and a fleet member (a2) at correlation rho.
import numpy as np
from scipy.stats import norm
def mu(a): return np.sqrt(2)*norm.ppf(a)
def blend(a1,a2,rho,w):
    m=(1-w)*mu(a1)+w*mu(a2); v=(1-w)**2+w**2+2*w*(1-w)*rho
    return norm.cdf(m/np.sqrt(2*v))
def best(a1,a2,rho):
    ws=np.linspace(0,1,101); b=[blend(a1,a2,rho,w) for w in ws]; i=int(np.argmax(b)); return ws[i], b[i]-a1
P=.943
print('implied rho_eff from K2: solo .932, W .30, blend displayed .943 (true in [.9425,.9435))')
for rho in [.80,.84,.86,.88,.90,.92,.94]:
    print(f'  rho {rho:.2f}: blend@W.30 = {blend(P,.932,rho,.30):.4f}')
print()
print('scenarios: best-W lift over parent .943')
print('solo vs rho   ' + '  '.join(f'{r:.2f}' for r in [.80,.84,.88,.90,.92]))
for s in [.932,.936,.940,.944,.948,.950]:
    print(f'  {s:.3f}     ' + '  '.join(f'{best(P,s,r)[1]:+.4f}' for r in [.80,.84,.88,.90,.92]))
