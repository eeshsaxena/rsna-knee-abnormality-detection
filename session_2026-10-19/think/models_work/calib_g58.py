# calib_g58.py: gold58 delta for the two LB-POSITIVE member moves (S6fall .2 on, A224 .25 on), to calibrate div_g58 deltas.
import sys, numpy as np
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2]]
exec(open('div_g58.py').read().split('base = c4()')[0])   # reuse loaders + helpers only (our own file)
def blend(ws6=.2, A=.25):
    out = np.zeros((n, 12))
    for j, c in enumerate(L):
        rr = (1 - A) * pct(M["r384"][:, j]) + A * pct(M["r224"][:, j]); w = C4.get(c, .5)
        b = w * rr + (1 - w) * pct(M["R6"][:, j]); out[:, j] = (1 - ws6) * pct(b) + ws6 * pct(M["S6fall"][:, j])
    return out
ref = blend(); rb = boot(ref); rm = mac(ref).mean()
for name, kw in [("no S6fall (ws6 0)", dict(ws6=0)), ("S6fall .3", dict(ws6=.3)), ("S6fall .15", dict(ws6=.15)), ("no A224 (A 0)", dict(A=0)), ("A224 .15", dict(A=.15))]:
    P = blend(**kw); d = boot(P) - rb
    print("%-20s c4 -> variant gold58 %+.5f  P(variant>c4) %.2f" % (name, mac(P).mean() - rm, (d > 0).mean()))
