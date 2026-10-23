# lbstrat_fill.py (10-20 ~16:45 IST, Claude): value of ONE pw25 fill-in conditional on the A21 outcome AND on whether the A21 winner
# already shows .960. MODEL only; same ABC (push/work/abc_local2.run, box aggregates only, no OAI rows).
import sys, numpy as np
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push/work/')
import abc_local2 as A
A21 = ['S20', 'S30', 'PW20', 'PW30']; NM = dict(S20='eoa20', S30='eoa30', PW20='pw20', PW30='pw30')
out = []
pr = lambda s: (print(s), out.append(s))
for prox, psh, ps in (('PFT', 'N01', 'logN'), ('PAREA', 'N01', 'logN'), ('PFT', 'flat', 'logU')):
    o = A.run(prox, 'D5', psh, ps, M_=100000, quiet=True); a = o['_arr']; G, Pop, w, iv, gap = a['G'], a['Pop'], a['w'], a['iv'], a['gap']
    w = w / w.sum(); G4 = np.stack([G[:, iv[v]] for v in A21], 1); P4 = np.stack([Pop[:, iv[v]] for v in A21], 1)
    win = np.argmax(G4, 1); g4 = G4.max(1); p4 = P4[np.arange(len(win)), win]; s20 = G[:, iv['S20']]
    shown = (g4 - s20) > gap
    pr('== %s/%s/%s' % (prox, psh, ps))
    for c in ('PW25', 'S25', 'PW35'):
        gc = G[:, iv[c]]; pc = Pop[:, iv[c]]
        for j in range(4):
            for sh in (True, False):
                m = (win == j) & (shown == sh); t = (w * m).sum()
                if t < .005: continue
                E = lambda x: float((w * m * x).sum() / t)
                mv = gc > g4; gain = np.where(mv, pc - p4, 0.0); sh5 = (np.maximum(g4, gc) - s20) > gap
                pr('  %-5s winner %-5s shows .960 %-5s (P %.3f): P(moves) %.2f | E[private gain R1] %+.2fe-4 | P(.960 after) %.2f (+%.2f) | cand pop - winner pop %+.2fe-4' % (
                    NM.get(c, c), NM[A21[j]], sh, t, E(mv), E(gain) * 1e4, E(sh5), E(sh5) - E(shown), E(pc - p4) * 1e4))
open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/levers/lbstrat_fill.log', 'w').write('\n'.join(out) + '\n')
