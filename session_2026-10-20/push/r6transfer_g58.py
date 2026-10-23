# r6transfer_g58.py (10-20 push, bpick lens; RESEARCH ONLY, local CPU, gold58 = RSNA competition data, no OAI).
# Question: how much of c4's E-OA effect transfers to the R6 entry (57248084), and how correlated are the two E-OA deltas?
# Reuses ex_r6eoa_gate.py's machinery verbatim (exec of its head: the kernels' OWN cell-18 code with the 9 E-OA lines inserted in
# memory, stage 3 rebuilt from the gold58 serves with each kernel's cell-14 parameters). Builds nothing, writes no kernel.
import numpy as np, pandas as pd
from scipy.stats import spearmanr
SRC = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/ex_r6eoa_gate.py'
s = open(SRC).read(); head = s[:s.index('res = {}')]
G = {'__name__': 'gate_head'}; exec(compile(head, 'gate_head', 'exec'), G)
run, auc, L, OA, Y, n = G['run'], G['auc'], G['L'], G['OA'], G['Y'], G['n']
outs = {}
for base in ('c4', 'R6'):
    for a in (None, 0.1, 0.2, 0.3):
        _, _, o = run(base, a); outs[(base, a)] = o
rk = lambda v: pd.Series(v).rank(pct=True).values
print('== OA column agreement, c4 vs R6 entry (gold58, final outputs, a 0)')
for c in OA:
    print('  %-10s Spearman c4 vs R6 %.4f' % (c, spearmanr(outs[('c4', None)][c], outs[('R6', None)][c])[0]))
print('== E-OA rank change agreement (o_a - o_0 in pct ranks), c4 vs R6')
for a in (0.1, 0.2, 0.3):
    r = [spearmanr(rk(outs[('c4', a)][c]) - rk(outs[('c4', None)][c]), rk(outs[('R6', a)][c]) - rk(outs[('R6', None)][c]))[0] for c in OA]
    print('  a %.1f: ' % a + ' '.join('%s %.3f' % (c, x) for c, x in zip(OA, r)))
rng = np.random.default_rng(1020); B = 4000
def macro_oa(o, ix):
    t = 0.
    for c in OA:
        j = L.index(c); y = Y[ix, j]
        if 0 < y.sum() < len(y): t += auc(o[c].values[ix], y)
    return t / 12
print('== paired bootstrap of the macro E-OA deltas (OA labels / 12), gold58 n %d, B %d' % (n, B))
for a in (0.1, 0.2, 0.3):
    dc, dr = [], []
    for _ in range(B):
        ix = rng.integers(0, n, n)
        dc.append(macro_oa(outs[('c4', a)], ix) - macro_oa(outs[('c4', None)], ix))
        dr.append(macro_oa(outs[('R6', a)], ix) - macro_oa(outs[('R6', None)], ix))
    dc, dr = np.array(dc), np.array(dr)
    full_c = macro_oa(outs[('c4', a)], np.arange(n)) - macro_oa(outs[('c4', None)], np.arange(n))
    full_r = macro_oa(outs[('R6', a)], np.arange(n)) - macro_oa(outs[('R6', None)], np.arange(n))
    slope = np.cov(dc, dr)[0, 1] / dc.var()
    print('  a %.1f: c4 %+.5f (sd %.5f) R6 %+.5f (sd %.5f) | ratio of full-sample deltas %.2f | boot corr %.3f | slope R6 on c4 %.2f | '
          'sd(R6 - c4) %.5f | P(R6 delta > 0) %.2f' % (a, full_c, dc.std(), full_r, dr.std(), full_r / full_c, np.corrcoef(dc, dr)[0, 1], slope,
                                                        (dr - dc).std(), (dr > 0).mean()))
