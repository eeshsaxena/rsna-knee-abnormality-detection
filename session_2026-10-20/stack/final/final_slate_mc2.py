# final_slate_mc2.py (10-21 ~00:20 IST, Claude). MODEL only. Same model as final_slate_mc.py (skep_value_mc.py world()/slate() verbatim).
# Question: does pw25 still pull weight once spl (= pw25 + lomm05) is in the slate? Compares slates with and without pw25.
import numpy as np, io
SRC = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/skeptic/skep_value_mc.py'
LOGP = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/final/final_slate_mc2.log'
src = open(SRC, encoding='utf-8').read(); src = src[:src.index("P('MODEL skep_value_mc.py")]
src = src.replace("OUT = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/skeptic/skep_value_mc.log', 'w')", "OUT = io.StringIO()")
assert 'OUT = io.StringIO()' in src
g = {'io': io}; exec(compile(src, SRC, 'exec'), g); world, slate = g['world'], g['slate']
L = open(LOGP, 'w')
def P(s=''):
    print(s, flush=True); L.write(s + '\n')
P('MODEL final_slate_mc2.py | E private of A vs pw30 (e-4), P(A<0), P(.960) joint | lomm05 prior stack / skeptic | MCLmm prior skep2 / zero')
for lomm in ('stack', 'skeptic'):
    for pr in ('skep2', 'zero'):
        W = world(pr, (.75, 1.8), lomm)
        for es in (['S_pl*'], ['pw25', 'S_pl*'], ['S_pl*', 'S_wide'], ['pw25', 'S_pl*', 'S_wide']):
            s = slate(W, es)
            P('   lomm05 %-7s MCLmm %-5s %-26s E %+.2f | P(A<0) %.2f | P(.960) %.2f' % (lomm, pr, '{%s}' % ', '.join(es), s['priv'].mean(), (s['priv'] < 0).mean(), s['p960']))
P('FINAL_SLATE_MC2 DONE'); L.close()
