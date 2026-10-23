# abc_joint.py: joint read probabilities for the A21 slate (eoa30, pw20, pw30 read in parallel; R1 = max unrounded) from abc_local.run.
import numpy as np, abc_local as A
rows = []
for prox in ('PFT', 'PAREA'):
    for psh in ('N01', 'flat'):
        for ps in ('logN', 'logU'):
            o = A.run(prox, 'D5', psh, ps, quiet=True); a = o['_arr']; G, w, gap, iv = a['G'], a['w'], a['gap'], a['iv']; t = w.sum()
            inc = {v: G[:, iv[v]] - G[:, iv['S20']] for v in ('S30', 'PW20', 'PW30', 'S25')}
            mx = np.max(np.stack([inc['S30'], inc['PW20'], inc['PW30']], 1), 1)
            P = lambda x: float((w * x).sum() / t)
            best = np.argmax(np.stack([np.zeros_like(mx), inc['S30'], inc['PW20'], inc['PW30']], 1), 1)
            bm = [P(best == k) for k in range(4)]
            # expected rank change: ~4.5 teams per 1e-4 near the .959/.960 boundary (density 45-55 per 1e-3, MODEL)
            print('%-5s %-4s %-4s | P(best-id moves) %.2f | P(any shows .960) %.2f | E[max inc | move] %+.2fe-4 | best: eoa20 %.2f eoa30 %.2f pw20 %.2f pw30 %.2f | P(.960) eoa30 %.2f pw20 %.2f pw30 %.2f (eoa25 %.2f)' % (
                prox, psh, ps, P(mx > 0), P(mx > gap), (w * mx * (mx > 0)).sum() / max(1e-12, (w * (mx > 0)).sum()) * 1e4, *bm,
                P(inc['S30'] > gap), P(inc['PW20'] > gap), P(inc['PW30'] > gap), P(inc['S25'] > gap)))
