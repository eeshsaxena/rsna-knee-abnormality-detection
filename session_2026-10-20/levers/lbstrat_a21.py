# lbstrat_a21.py (10-20 ~16:35 IST, levers/LB-strategy lens, Claude). MODEL only, local CPU. Reuses push/work/abc_local2.run, which uses
# ONLY box AGGREGATES already on the laptop (per-regime mean curves + pooled noise covariance, box_abc2_*.log). No OAI row, no box contact.
# Questions:
#  (1) winner's curse of the E-OA climb: E[public - population] for eoa20 (the current best) and for the A21 R1 winner;
#  (2) public-private correlation of 1-parameter E-OA moves (vs c4, and the A21 alternatives vs eoa20);
#  (3) R1 (A = max public) vs a posterior rule that uses only what we will SEE after the reads (which entry is the best-id, which
#      entries display .960) -> private cost of R1 = E[pop(posterior pick)] - E[pop(R1 pick)];
#  (4) value of 0 / 1 / 2 extra E-OA fill-ins under R1 and under the posterior rule, per A21 outcome (complements slate_cond.log).
import sys, itertools, numpy as np
W = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push/work/'
sys.path.insert(0, W)
import abc_local2 as A

A21 = ['S20', 'S30', 'PW20', 'PW30']
NAME = {'S10': 'eoa10', 'S20': 'eoa20', 'S30': 'eoa30', 'PW20': 'pw20', 'PW30': 'pw30', 'PW25': 'pw25', 'S25': 'eoa25', 'PW35': 'pw35', 'PW15': 'pw15'}
sdp = np.sqrt(390 / 910)
out = []
pr = lambda s: (print(s), out.append(s))
for prox, psh, ps in (('PFT', 'N01', 'logN'), ('PAREA', 'N01', 'logN'), ('PFT', 'flat', 'logU')):
    o = A.run(prox, 'D5', psh, ps, M_=100000, quiet=True); a = o['_arr']
    G, Pop, w, gap, iv = a['G'], a['Pop'], a['w'], a['gap'], a['iv']
    E = A.load('box_abc2_%s.log' % prox); C = np.array(E['cov']['D5'])
    w = w / w.sum(); Ew = lambda x, m=None: float((w * x).sum()) if m is None else float((w * m * x).sum() / (w * m).sum())
    col = lambda v: G[:, iv[v]]; pop = lambda v: Pop[:, iv[v]]
    rng = np.random.default_rng(5)
    pr('\n== %s | sh %s | s %s | ESS %.0f' % (prox, psh, ps, o['ess']))
    # (1) winner's curse
    pr('  (1) winner\'s curse (public increment over c4 minus population increment, e-4):')
    pr('      eoa10 %.2f | eoa20 %.2f   (eoa20 public %+.2f, population %+.2f)' % (
        Ew(col('S10') - pop('S10')) * 1e4, Ew(col('S20') - pop('S20')) * 1e4, Ew(col('S20')) * 1e4, Ew(pop('S20')) * 1e4))
    G4 = np.stack([col(v) for v in A21], 1); P4 = np.stack([pop(v) for v in A21], 1)
    win = np.argmax(G4, 1); gw = G4.max(1); pw_ = P4[np.arange(len(win)), win]
    pr('      A21 R1 winner: public %+.2f, population %+.2f -> optimism %.2f (of which over eoa20: public %+.2f vs population %+.2f)' % (
        Ew(gw) * 1e4, Ew(pw_) * 1e4, Ew(gw - pw_) * 1e4, Ew(gw - col('S20')) * 1e4, Ew(pw_ - pop('S20')) * 1e4))
    # (2) public-private correlation of 1-parameter moves (population + private noise vs public read), posterior-weighted
    pr('  (2) corr(public delta, private delta) under the posterior:')
    for x, y in (('S10', None), ('S20', None), ('S30', None), ('PW30', None), ('S30', 'S20'), ('PW20', 'S20'), ('PW30', 'S20'), ('PW25', 'PW30'), ('S25', 'S20')):
        if y is None:
            g_ = col(x); p_ = pop(x); v = C[iv[x], iv[x]]
        else:
            g_ = col(x) - col(y); p_ = pop(x) - pop(y); v = C[iv[x], iv[x]] + C[iv[y], iv[y]] - 2 * C[iv[x], iv[y]]
        priv = p_ + rng.standard_normal(len(p_)) * np.sqrt(v) * sdp
        mg, mp = Ew(g_), Ew(priv)
        cov = Ew((g_ - mg) * (priv - mp)); cr = cov / np.sqrt(Ew((g_ - mg) ** 2) * Ew((priv - mp) ** 2))
        psign = Ew(((g_ > 0) == (priv > 0)).astype(float))
        pr('      %-12s corr %.2f | P(private sign = public sign) %.2f | public sd %.2f, population sd %.2f (e-4)' % (
            NAME.get(x, x) + ('-' + NAME.get(y, y) if y else '-c4'), cr, psign, np.sqrt(Ew((g_ - mg) ** 2)) * 1e4, np.sqrt(Ew((p_ - Ew(p_)) ** 2)) * 1e4))
    # (3) R1 vs posterior rule on observables: argmax identity x which A21 entries display .960 (G_k - G_S20 > gap)
    show = np.stack([(col(v) - col('S20')) > gap for v in A21], 1)
    key = win * 16 + (show * (1 << np.arange(4))).sum(1)
    # winner magnitude tercile (in-band rank read) as a third feature
    qs = np.quantile(gw - col('S20'), [1 / 3, 2 / 3]); terc = np.digitize(gw - col('S20'), qs)
    key2 = key * 3 + terc
    def rule_value(kk):
        pick = np.empty(len(kk), int)
        for c in np.unique(kk):
            m = kk == c
            mu = [Ew(P4[:, j], m) for j in range(4)]
            pick[m] = int(np.argmax(mu))
        return pick
    pk1 = rule_value(key); pk2 = rule_value(key2)
    pop_post1 = P4[np.arange(len(pk1)), pk1]; pop_post2 = P4[np.arange(len(pk2)), pk2]
    oracle = P4.max(1)
    pr('  (3) A pick among {eoa20, eoa30, pw20, pw30}, population increment over eoa20 (e-4):')
    pr('      always eoa20 %+.2f | always pw20 %+.2f | always pw30 %+.2f | R1 %+.2f | posterior(id + .960 flags) %+.2f | posterior(+ winner tercile) %+.2f | oracle %+.2f' % (
        0.0, Ew(pop('PW20') - pop('S20')) * 1e4, Ew(pop('PW30') - pop('S20')) * 1e4, Ew(pw_ - pop('S20')) * 1e4,
        Ew(pop_post1 - pop('S20')) * 1e4, Ew(pop_post2 - pop('S20')) * 1e4, Ew(oracle - pop('S20')) * 1e4))
    pr('      P(posterior rule overrides R1) %.2f / %.2f; private sd of the pick difference small (same E-OA family)' % (Ew((pk1 != win).astype(float)), Ew((pk2 != win).astype(float))))
    for j, v in enumerate(A21):
        m = win == j
        if (w * m).sum() < .01: continue
        mu = [Ew(P4[:, i] - pop('S20'), m) * 1e4 for i in range(4)]
        pr('      outcome %-5s P %.2f | E[pop - eoa20 | outcome]: %s | R1 keeps %s' % (NAME[v], (w * m).sum(), ' '.join('%s %+.2f' % (NAME[A21[i]], mu[i]) for i in range(4)), NAME[v]))
    # (4) extra fill-ins: value under R1 and under the posterior rule (observables: argmax over the enlarged set)
    pr('  (4) extra E-OA fill-ins (population gain of the A pick over the A21-only R1 pick, e-4; P(.960 shown)):')
    base_show = Ew(((gw - col('S20')) > gap).astype(float))
    for extra in ([], ['PW25'], ['S25'], ['PW25', 'S25'], ['PW25', 'PW35'], ['PW15', 'PW25']):
        S = A21 + extra; Gs = np.stack([col(v) for v in S], 1); Ps = np.stack([pop(v) for v in S], 1)
        wn = np.argmax(Gs, 1); r1 = Ps[np.arange(len(wn)), wn]
        kk = wn
        pick = np.empty(len(kk), int)
        for c in np.unique(kk):
            m = kk == c; mu = [Ew(Ps[:, j], m) for j in range(len(S))]; pick[m] = int(np.argmax(mu))
        po = Ps[np.arange(len(pick)), pick]
        sh = Ew(((Gs.max(1) - col('S20')) > gap).astype(float))
        pr('      + %-12s R1 %+.2f | posterior(id) %+.2f | P(.960) %.2f (%+.2f)' % ('+'.join(NAME[e] for e in extra) or 'nothing', Ew(r1 - pw_) * 1e4, Ew(po - pw_) * 1e4, sh, sh - base_show))
open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/levers/lbstrat_a21.log', 'w').write('\n'.join(out) + '\n')
