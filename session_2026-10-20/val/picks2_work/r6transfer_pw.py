# r6transfer_pw.py (10-20 val Track D; RESEARCH ONLY, local CPU, gold58 = RSNA competition data, no OAI; info only).
# How much of the pw-shape E-OA effect transfers from c4 (the A entries) to the R6 entry 57248084, and how R6+pw30 compares with R6e20.
# Reuses ex_r6eoa_gate.py's head (each kernel's OWN cell-18 code run on the gold58 serves). The E-OA block is the LAST transform on the
# OA columns, so sym / pw are applied post hoc to the unpatched outputs; check: post-hoc sym .2 == in-cell sym .2 (A20 block).
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
SRC = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/ex_r6eoa_gate.py'
s = open(SRC).read(); head = s[:s.index('res = {}')]
G = {'__name__': 'gate_head'}; exec(compile(head, 'gate_head', 'exec'), G)
run, auc, L, OA, Y, n = G['run'], G['auc'], G['L'], G['OA'], G['Y'], G['n']
pct = lambda v: rankdata(v) / len(v)
W = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}   # build_a21.py pw weights
def eoa(o, a, mode):
    o = o.copy(); U = {c: pct(o[c].values.astype(np.float64)) for c in OA}
    for c in OA:
        oth = [k for k in OA if k != c]
        if mode == 'sym': o[c] = pct(U[c] - a * (U[oth[0]] + U[oth[1]]) / 2.0).astype(np.float32)
        else:
            w = [W[tuple(sorted((c, k), key=OA.index))] for k in oth]
            o[c] = pct(U[c] - a / 2.0 * (w[0] * U[oth[0]] + w[1] * U[oth[1]])).astype(np.float32)
    return o
base = {b: run(b, None)[2] for b in ('c4', 'R6')}
for b in ('c4', 'R6'):
    _, _, o_in = run(b, 0.2); o_ph = eoa(base[b], 0.2, 'sym')
    print('[check] %s post-hoc sym .2 == in-cell A20 block: max |diff| %.2e' % (b, max(np.abs(o_in[c].values - o_ph[c].values).max() for c in L)))
V = [('sym', .1), ('sym', .2), ('sym', .3), ('pw', .2), ('pw', .25), ('pw', .3), ('pw', .35), ('pw', .4)]
out = {(b, m, a): eoa(base[b], a, m) for b in ('c4', 'R6') for m, a in V}
def macro(o, ix):
    t = 0.
    for j, c in enumerate(L):
        y = Y[ix, j]
        if 0 < y.sum() < len(y): t += auc(o[c].values[ix], y)
    return t / 12
full = np.arange(n); m0 = {b: macro(base[b], full) for b in ('c4', 'R6')}
print('gold58 macro (kernel path): c4 %.5f R6 %.5f (R6 - c4 %+.5f)' % (m0['c4'], m0['R6'], m0['R6'] - m0['c4']))
rng = np.random.default_rng(10201); B = 3000; BI = [rng.integers(0, n, n) for _ in range(B)]
bm = {k: np.array([macro(o, ix) for ix in BI]) for k, o in list(out.items()) + [(('c4', 'base', 0), base['c4']), (('R6', 'base', 0), base['R6'])]}
print('%-8s | %-9s %-9s | ratio | boot corr | slope | sd(R6d - c4d) | P(R6 delta > 0)' % ('variant', 'c4 delta', 'R6 delta'))
for m, a in V:
    dc_f = macro(out[('c4', m, a)], full) - m0['c4']; dr_f = macro(out[('R6', m, a)], full) - m0['R6']
    dc = bm[('c4', m, a)] - bm[('c4', 'base', 0)]; dr = bm[('R6', m, a)] - bm[('R6', 'base', 0)]
    print('%-3s %.2f | %+.5f %+.5f | %.2f | %.3f | %.2f | %.5f | %.2f' % (m, a, dc_f, dr_f, dr_f / dc_f, np.corrcoef(dc, dr)[0, 1], np.cov(dc, dr)[0, 1] / dc.var(), (dr - dc).std(), (dr > 0).mean()))
print('== pair-relevant gold58 comparisons (macro, info only; n 58)')
def cmp(k1, k2, lab):
    f = macro(out[k1], full) - macro(out[k2], full); d = bm[k1] - bm[k2]
    print('  %-34s %+.5f (boot sd %.5f, P>0 %.2f)' % (lab, f, d.std(), (d > 0).mean()))
cmp(('R6', 'pw', .3), ('R6', 'sym', .2), 'R6pw30 - R6e20')
cmp(('c4', 'pw', .3), ('c4', 'sym', .2), 'pw30 - eoa20 (c4 base)')
cmp(('R6', 'pw', .3), ('c4', 'pw', .3), 'R6pw30 - pw30 (B - A)')
cmp(('R6', 'sym', .2), ('c4', 'pw', .3), 'R6e20 - pw30 (B - A)')
cmp(('c4', 'sym', .2), ('c4', 'pw', .3), 'eoa20 - pw30 (B - A)')
print('== per-OA-column AUC (full gold58)')
for k in [('c4', 'pw', .3), ('R6', 'pw', .3), ('c4', 'sym', .2), ('R6', 'sym', .2)]:
    print('  %-14s ' % ('%s %s %.2f' % k) + ' '.join('%s %.4f' % (c, auc(out[k][c].values, Y[:, L.index(c)])) for c in OA))
for b in ('c4', 'R6'): print('  %-14s ' % (b + ' base') + ' '.join('%s %.4f' % (c, auc(base[b][c].values, Y[:, L.index(c)])) for c in OA))
print('== E-OA rank-change agreement c4 vs R6 (pw .3): ' + ' '.join('%s %.3f' % (c, spearmanr(pct(out[('c4', 'pw', .3)][c]) - pct(base['c4'][c]), pct(out[('R6', 'pw', .3)][c]) - pct(base['R6'][c]))[0]) for c in OA))
