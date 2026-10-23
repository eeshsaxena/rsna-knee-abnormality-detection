# bma_abs.py: absolute pair metrics per A under the bma.py weightings (MODEL; weights GUESS). Writes bma_abs.out.
import json, numpy as np
exec(open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/lastday/bma.py', encoding='utf-8').read().split("out = open(")[0])
out = open(HERE + '/bma_abs.out', 'w')
def P(s=''):
    print(s); out.write(s + '\n')
P('bma_abs (MODEL). cols: silver step / silver smooth | r150 step / r150 smooth | bronze | E e-4 (pub-scale = .95802 + E x 1e-4)')
for wn in ('A22 priors', 'optimist', 'flat', 'levers', 'skeptic', 'EB-heavy'):
    wl, wm = LW[wn]
    pw = {k: wl[k[0]] * wm[k[1]] for k in R}; post = {k: pw[k] * EV['%s|%s' % k]['ev'] for k in R}; s1 = sum(post.values()); post = {k: v / s1 for k, v in post.items()}
    stk = sum(post[k] * R[k]['decomp']['priv'][5] for k in R)
    P('\n== %s (posterior private stack E %+.2f) ==' % (wn, stk))
    for B in ('R6', 'r6pw20 rule', 'mix0.55', 'none (A alone)'):
        for a in AL:
            v = sum(post[k] * np.array(R[k]['pairs']['%s|%s' % (a, B)]) for k in R)
            P('  B=%-14s A=%-6s %.3f / %.3f | %.3f / %.3f | %.3f | %+6.2f (%.5f)' % (B, a, v[1], v[3], v[2], v[5], v[4], v[0], .95802 + v[0] * 1e-4))
out.close()
