# lbstrat_override.py (10-20 ~16:15 IST, Claude): P(override target population > R1 pick population | A21 outcome), same ABC as lbstrat_a21.py.
# MODEL only; box aggregates via push/work/abc_local2.run; no OAI rows.
import sys, numpy as np
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push/work/')
import abc_local2 as A
A21 = ['S20', 'S30', 'PW20', 'PW30']; NM = dict(S20='eoa20', S30='eoa30', PW20='pw20', PW30='pw30')
out = []
pr = lambda s: (print(s), out.append(s))
for prox, psh, ps in (('PFT', 'N01', 'logN'), ('PAREA', 'N01', 'logN'), ('PFT', 'flat', 'logU'), ('PFT', 'N01', 'logU'), ('PAREA', 'flat', 'logU')):
    o = A.run(prox, 'D5', psh, ps, M_=100000, quiet=True); a = o['_arr']; G, Pop, w, iv = a['G'], a['Pop'], a['w'], a['iv']
    w = w / w.sum(); G4 = np.stack([G[:, iv[v]] for v in A21], 1); win = np.argmax(G4, 1)
    pr('== %s/%s/%s' % (prox, psh, ps))
    for j, alt in ((0, 'PW20'), (1, 'PW30'), (2, 'PW20'), (3, 'PW30'), (1, 'S20'), (3, 'PW20')):
        m = win == j; t = (w * m).sum()
        if t < .01: continue
        d = Pop[:, iv[alt]] - Pop[:, iv[A21[j]]]
        if A21[j] == alt: continue
        Ed = (w * m * d).sum() / t; Pp = (w * m * (d > 0)).sum() / t
        # private: add private-sample noise of the difference
        E = A.load('box_abc2_%s.log' % prox); C = np.array(E['cov']['D5'])
        v = C[iv[alt], iv[alt]] + C[iv[A21[j]], iv[A21[j]]] - 2 * C[iv[alt], iv[A21[j]]]
        priv = d + np.random.default_rng(3).standard_normal(len(d)) * np.sqrt(v * 390 / 910)
        Pv = (w * m * (priv > 0)).sum() / t
        pr('  outcome %-5s (P %.2f): %s - %s population %+.2fe-4, P(pop > 0) %.2f, P(private > 0) %.2f' % (NM[A21[j]], t, NM[alt], NM[A21[j]], Ed * 1e4, Pp, Pv))
open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/levers/lbstrat_override.log', 'w').write('\n'.join(out) + '\n')
