"""verify_B checks 2: consistency of r_GN (cluster) vs goodpjw 'N + readers @.30 = .948'. Light numpy."""
import numpy as np
from scipy.stats import norm
d=lambda a: np.sqrt(2)*norm.ppf(a)
def b2(a,b,r,W):
    v=(1-W)**2+W**2+2*W*(1-W)*r
    return norm.cdf(((1-W)*d(a)+W*d(b))/np.sqrt(2*v))
print('A. N .949 + G .929 at global w, by r_GN (cluster calibration point is w .162 -> .9500)')
for r in (.80,.84,.88,.90):
    print('  r_GN',r,' '.join(f'w{w:.2f}:{b2(.949,.929,r,w):.4f}' for w in (.10,.162,.20,.30)))
print('B. N .949 + readers mix (Raptor .933 + G .929 averaged, solo GUESS .933-.937) at .30 -> measured .948 display [.9475,.9485)')
for rd in (.933,.935,.937):
    print('  readers',rd,' '.join(f'r{r:.2f}:{b2(.949,rd,r,.30):.4f}' for r in (.84,.86,.88,.90,.92,.94)))
print('C. F .944 with N at F weight .30/.40 by r_FN (P-free)')
for r in (.86,.88,.90,.92,.94):
    print('  r_FN',r,' '.join(f'W{w:.2f}:{b2(.949,.944,r,w):.4f}' for w in (.30,.40)))
print('D. If F behaves like the readers mix: implied r(N,readers) band, then F .944 at that r')
