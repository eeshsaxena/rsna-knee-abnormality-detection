# slate_cond.py (10-20 ~15:40 IST, push synthesis, Claude). MODEL only, local CPU. Reuses work/abc_local2.run, which uses
# ONLY box AGGREGATES already on the laptop (per-regime mean curves + pooled noise covariance from box_abc2_*.log). No OAI row,
# no box contact, nothing written except this script's stdout (redirected to slate_cond.log).
# Question: for each possible A21 outcome (which of eoa20 / eoa30 / pw20 / pw30 the API best-id lands on), which untested E-OA
# shape(s) would be worth a 10-21 slot, judged by
#   P(cand becomes best-id | outcome), E[private gain of R1 over {A21 + cand}] (population value; private-sample noise is
#   independent of the public read), P(cand population > A21 winner population), and the cosmetic P(shows .960).
# Also: the best PAIR of untested shapes per outcome (two slots), and the oracle headroom per outcome.
import sys, itertools, numpy as np
W = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push/work/'
sys.path.insert(0, W)
import abc_local2 as A

A21 = ['S20', 'S30', 'PW20', 'PW30']
NAME = {'S20': 'eoa20', 'S30': 'eoa30', 'PW20': 'pw20', 'PW30': 'pw30'}
CANDS = ['S15', 'S25', 'S35', 'S40', 'PW15', 'PW25', 'PW35', 'PW40']
GRID = ['S10', 'S15', 'S20', 'S25', 'S30', 'S35', 'S40', 'PW15', 'PW20', 'PW25', 'PW30', 'PW35', 'PW40']
for prox, psh, ps in (('PFT', 'N01', 'logN'), ('PAREA', 'N01', 'logN'), ('PFT', 'flat', 'logU')):
    o = A.run(prox, 'D5', psh, ps, M_=100000, quiet=True); a = o['_arr']
    G, Pop, w, gap, iv = a['G'], a['Pop'], a['w'], a['gap'], a['iv']
    col = lambda v: iv[v]
    G4 = np.stack([G[:, col(v)] for v in A21], 1); P4 = np.stack([Pop[:, col(v)] for v in A21], 1)
    win = np.argmax(G4, 1); pop4 = P4[np.arange(len(win)), win]; g4 = G4.max(1)
    s20 = G[:, col('S20')]
    show4 = g4 - s20 > gap
    best_pop = np.max(np.stack([Pop[:, col(v)] for v in GRID], 1), 1)
    print('\n== %s | sh %s | s %s | ESS %.0f' % (prox, psh, ps, o['ess']))
    for k, v in enumerate(A21):
        m = win == k; ww = w * m; t = ww.sum()
        if t / w.sum() < .01:
            continue
        E = lambda x: float((ww * x).sum() / t)
        print('  A21 winner %-5s P %.2f | P(.960 shown) %.2f | winner pop vs eoa20 %+.2fe-4 | oracle headroom over winner %+.2fe-4' % (
            NAME[v], t / w.sum(), E(show4), E(pop4 - Pop[:, col('S20')]) * 1e4, E(best_pop - pop4) * 1e4))
        rows = []
        for c in CANDS:
            gc = G[:, col(c)]; pc = Pop[:, col(c)]
            mv = gc > g4; pop5 = np.where(mv, pc, pop4); show5 = np.maximum(g4, gc) - s20 > gap
            rows.append((c, E(mv), E(pop5 - pop4) * 1e4, E(pc - pop4) * 1e4, E(pc > pop4), E(show5) - E(show4)))
        rows.sort(key=lambda r: -r[2])
        for c, pm, eg, dpop, ppos, d960 in rows:
            print('     + %-5s P(moves) %.2f | E[private gain under R1] %+.2fe-4 | cand pop - winner pop %+.2fe-4, P(>0) %.2f | P(.960) +%.2f' % (
                c, pm, eg, dpop, ppos, d960))
        # best pair of untested shapes (2 slots)
        best = None
        for c1, c2 in itertools.combinations(CANDS, 2):
            g5 = np.stack([g4, G[:, col(c1)], G[:, col(c2)]], 1); p5 = np.stack([pop4, Pop[:, col(c1)], Pop[:, col(c2)]], 1)
            pp = p5[np.arange(len(g5)), np.argmax(g5, 1)]; eg = E(pp - pop4) * 1e4
            if best is None or eg > best[2]:
                best = (c1, c2, eg, E(np.maximum(g4, g5.max(1)) - s20 > gap) - E(show4))
        print('     best PAIR %s + %s: E[private gain under R1] %+.2fe-4 | P(.960) +%.2f' % best)

# --- appended 10-20 ~15:55: exact values for the two specific pairs named in PUSH_1020 sec 2.2 (same seeds / runs as above) ---
print('\n== specific pairs')
for prox, psh, ps in (('PFT', 'N01', 'logN'), ('PAREA', 'N01', 'logN'), ('PFT', 'flat', 'logU')):
    o = A.run(prox, 'D5', psh, ps, M_=100000, quiet=True); a = o['_arr']
    G, Pop, w, gap, iv = a['G'], a['Pop'], a['w'], a['gap'], a['iv']
    G4 = np.stack([G[:, iv[v]] for v in A21], 1); P4 = np.stack([Pop[:, iv[v]] for v in A21], 1)
    win = np.argmax(G4, 1); pop4 = P4[np.arange(len(win)), win]; g4 = G4.max(1); s20 = G[:, iv['S20']]
    for k, c1, c2 in ((0, 'PW25', 'S25'), (1, 'PW25', 'PW35')):
        m = win == k; ww = w * m; t = ww.sum(); E = lambda x: float((ww * x).sum() / t)
        g5 = np.stack([g4, G[:, iv[c1]], G[:, iv[c2]]], 1); p5 = np.stack([pop4, Pop[:, iv[c1]], Pop[:, iv[c2]]], 1)
        pp = p5[np.arange(len(g5)), np.argmax(g5, 1)]
        print('  %s | winner %s | pair %s + %s: E[private gain under R1] %+.2fe-4 | P(.960) +%.2f' % (
            prox + '/' + psh + '/' + ps, NAME[A21[k]], c1, c2, E(pp - pop4) * 1e4, E(np.maximum(g4, g5.max(1)) - s20 > gap) - E(g4 - s20 > gap)))
