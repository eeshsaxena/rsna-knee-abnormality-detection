# skeptic_abc.py (10-20 push, skeptic pass, Claude). MODEL only, local CPU, reuses work/abc_local2.run (box AGGREGATES only:
# per-regime mean curves and noise covariance already on the laptop; no OAI row is used or needed).
# Questions:
#  1. pw25 as a 10-21 probe AFTER the A21 reads: P(it becomes the API best-id), E[private gain of R1 over {A21 + pw25} minus R1
#     over {A21}], and the cosmetic P(.960) increment. (Private sample noise is independent of the public read, so the private
#     expectation of an R1 choice is its population value.)
#  2. How well do the reads identify the hidden OA regime (posterior vs prior sd), and how often would the A21 reads push the
#     regime posterior mean to <= -1 sd (the implied.md sec 6 trigger for LOA <- MM)?
#  3. The unconditional G20 / G10 ratio (implied quotes 1.69, the median over 'valid' inversions only).
import sys, numpy as np
W = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push/work/'
sys.path.insert(0, W)
import abc_local2 as A

LB = np.load(W + 'lb_post_pooled.npz'); R = LB['G20'] / LB['G10']
print('== 3. G20/G10 over the pooled LB posterior: median %.2f [p10 %.2f, p90 %.2f] (implied.md quotes 1.69 [1.34, 1.97] = valid-inversion subset)' % (
    np.median(R), np.quantile(R, .1), np.quantile(R, .9)))

A21 = ['S20', 'S30', 'PW20', 'PW30']
for prox, psh, ps in (('PFT', 'N01', 'logN'), ('PAREA', 'N01', 'logN'), ('PFT', 'flat', 'logU')):
    o = A.run(prox, 'D5', psh, ps, M_=60000, quiet=True); a = o['_arr']
    G, Pop, w, gap, iv, sh = a['G'], a['Pop'], a['w'], a['gap'], a['iv'], a['sh']; t = w.sum()
    E = lambda x: float((w * x).sum() / t)
    col = lambda v: iv[v]
    G4 = np.stack([G[:, col(v)] for v in A21], 1); P4 = np.stack([Pop[:, col(v)] for v in A21], 1)
    c4 = np.argmax(G4, 1); pop4 = P4[np.arange(len(c4)), c4]
    print('\n== %s | sh %s | s %s | ESS %.0f' % (prox, psh, ps, o['ess']))
    # noise correlations of the candidate with its neighbours (box covariance)
    E_ = A.load('box_abc2_%s.log' % ('PFT' if prox == 'PFT' else 'PAREA')); Cv = np.array(E_['cov']['D5']); VN = E_['VN']
    ix = {v: VN.index(v) for v in VN}
    def ncorr(u, v): return Cv[ix[u], ix[v]] / np.sqrt(Cv[ix[u], ix[u]] * Cv[ix[v], ix[v]])
    def nsd(u, v): return np.sqrt(Cv[ix[u], ix[u]] + Cv[ix[v], ix[v]] - 2 * Cv[ix[u], ix[v]])
    print('  public-noise corr PW25 with PW20 / PW30 / S20: %.3f / %.3f / %.3f | sd(PW25-PW20) %.2fe-4, sd(PW25-PW30) %.2fe-4' % (
        ncorr('PW25', 'PW20'), ncorr('PW25', 'PW30'), ncorr('PW25', 'S20'), nsd('PW25', 'PW20') * 1e4, nsd('PW25', 'PW30') * 1e4))
    best_pop = np.max(np.stack([Pop[:, col(v)] for v in ('S10', 'S15', 'S20', 'S25', 'S30', 'S35', 'S40', 'PW15', 'PW20', 'PW25', 'PW30', 'PW35', 'PW40')], 1), 1)
    print('  headroom: E[pop(best shape on the grid) - pop(R1 over A21)] %+.2fe-4' % (E(best_pop - pop4) * 1e4))
    for cand in ('PW25', 'S25', 'PW35'):
        g5 = G[:, col(cand)]; p5 = Pop[:, col(cand)]
        mv = g5 > G4.max(1); pop5 = np.where(mv, p5, pop4)
        show4 = G4.max(1) - G[:, col('S20')] > gap; show5 = np.maximum(G4.max(1), g5) - G[:, col('S20')] > gap
        print('  + %-4s: P(becomes best-id) %.2f | E[private gain of R1 incl. it] %+.2fe-4 (P>0 given move %.2f; E[gain | move] %+.2fe-4) | P(.960) A21 %.2f -> %.2f' % (
            cand, E(mv), E(pop5 - pop4) * 1e4, E((pop5 - pop4 > 0) & mv) / max(E(mv), 1e-9), E((pop5 - pop4) * mv) / max(E(mv), 1e-9) * 1e4,
            E(show4), E(show5)))
        if cand == 'PW25':
            for k, v in enumerate(A21):
                m = c4 == k
                if E(m) < .01: continue
                ww = w * m; tt = ww.sum()
                print('      if A21 winner = %-4s (P %.2f): P(pw25 moves) %.2f | E[private gain] %+.2fe-4' % (
                    v, E(m), float((ww * mv).sum() / tt), float((ww * (pop5 - pop4)).sum() / tt) * 1e4))
    # 2. regime identification
    m_sh = E(sh); sd_sh = np.sqrt(E((sh - m_sh) ** 2))
    X = np.column_stack([np.ones(len(sh))] + [G[:, col(v)] - G[:, col('S20')] for v in ('S30', 'PW20', 'PW30')])
    sw = np.sqrt(w / t); beta = np.linalg.lstsq(X * sw[:, None], sh * sw, rcond=None)[0]; fit = X @ beta
    res_sd = np.sqrt(E((sh - fit) ** 2))
    print('  regime sh: prior sd %.2f -> posterior now mean %+.2f sd %.2f | after A21 (linear preposterior): sd of posterior mean %.2f, residual sd %.2f, P(post mean <= -1) %.3f, P(<= -0.5) %.3f' % (
        np.std(np.clip(np.random.default_rng(0).normal(0, 1, 100000), -2.5, 1)) if psh == 'N01' else np.std(np.random.default_rng(0).uniform(-2.5, 1, 100000)),
        m_sh, sd_sh, np.sqrt(E((fit - m_sh) ** 2)), res_sd, E(fit <= -1), E(fit <= -.5)))
