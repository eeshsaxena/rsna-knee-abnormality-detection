# final_slate_mc3.py (10-21 ~00:25 IST, Claude). MODEL only. Same model as final_slate_mc.py (skep_value_mc.py world()/slate() verbatim).
# Stress: lomm05 in an "OAI-only" world (population mean shifted to -0.9e-4, the figure PRIVATE_1020 sec 5 quotes from lomm05's own
# skeptic), I(pw25, lomm05) tied to N(0, .12) (lomm05 skeptic branch), then every entry containing lomm05 shifted by the same constant
# on public and private (a population shift moves both reads). Also prints p10 of A's private for the private-first lens.
import numpy as np, io
SRC = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/skeptic/skep_value_mc.py'
LOGP = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/final/final_slate_mc3.log'
src = open(SRC, encoding='utf-8').read(); src = src[:src.index("P('MODEL skep_value_mc.py")]
src = src.replace("OUT = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/skeptic/skep_value_mc.log', 'w')", "OUT = io.StringIO()")
assert 'OUT = io.StringIO()' in src
g = {'io': io}; exec(compile(src, SRC, 'exec'), g); world, slate, ENT = g['world'], g['slate'], g['ENT']
L = open(LOGP, 'w')
def P(s=''):
    print(s, flush=True); L.write(s + '\n')
SL = (['pw25'], ['pw25', 'lomm05'], ['S_pl*'], ['pw25', 'S_pl*'], ['pw25', 'S_pl*', 'S_wide'], ['pw25', 'S_pl*', 'S_alt', 'S_wide'])
P('MODEL final_slate_mc3.py | E private of A vs pw30 (e-4) [p10] | P(A<0) | P(.960) joint')
for shift, lab in ((0., 'lomm05 skeptic N(0, .9)'), (-.9, 'lomm05 OAI-only: skeptic N(0, .9) shifted -0.9')):
    for pr in ('skep2', 'zero'):
        W = world(pr, (.75, 1.8), 'skeptic')
        for k, parts in ENT.items():
            if 'lomm05' in parts:
                W['pub'][k] = W['pub'][k] + shift; W['pri'][k] = W['pri'][k] + shift
        for es in SL:
            s = slate(W, es)
            P('   %-46s MCLmm %-5s %-34s E %+.2f [p10 %+.2f] | P(A<0) %.2f | P(.960) %.2f' % (lab, pr, '{%s}' % ', '.join(es), s['priv'].mean(), np.percentile(s['priv'], 10), (s['priv'] < 0).mean(), s['p960']))
P('FINAL_SLATE_MC3 DONE'); L.close()
