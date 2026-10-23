# stacker_g58.py (10-19 trick audit, Claude, research only): does R6's internal report-fitted Baker's STATE stacker / Effusion
# FLUID stacker cost score in the expert (gold) regime? gold58 (R6 served on the box exactly as c4 cells 7-9, dump.json = per-fold
# flip-TTA logits; gold not in R6 training or stacker fit, receipt gold_in_training 0). c4 nested replay (load.py conventions).
import sys, json; sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
import numpy as np, pandas as pd
from load import *
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
B = G + '/r6g58_box/run_b/worker0/'
dump = json.load(open(B + 'dump.json'))
Z = np.array([np.mean(dump[s]['z'], 0) for s in ids])          # fold-mean flip-TTA logits (the stackers' z feature)
st = pd.read_csv(B + 'state_stacker.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids]
fl = pd.read_csv(B + 'fluid_stacker.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids]
jb, je = L.index("Baker's"), L.index('Effusion')
print('check served Baker p == state p_out: %.2e ; Effusion == fluid p_out: %.2e' % (np.abs(R6[:, jb] - st["Baker's|p_out"].values).max(),
      np.abs(R6[:, je] - fl['Effusion|p_out'].values).max()))
print('fluid status on gold58:', fl['status'].value_counts().to_dict())


def c4(R6m, ix=None):
    ix = np.arange(n) if ix is None else ix; P = np.zeros((len(ix), 12))
    for j, c in enumerate(L):
        w = C4.get(c, .5)
        rr = .75 * pct(R384[ix, j]) + .25 * pct(R224[ix, j])
        b = w * rr + (1 - w) * pct(R6m[ix, j])
        P[:, j] = .8 * pct(b) + .2 * pct(S6[ix, j])
    return P


mac = lambda P, yy: np.mean([auc(P[:, j], yy[:, j]) for j in range(12)])
V = {}
V['c4'] = R6
x = R6.copy(); x[:, jb] = Z[:, jb]; V['Baker raw z'] = x
x = R6.copy(); x[:, je] = Z[:, je]; V['Effusion raw z'] = x
x = R6.copy(); x[:, jb] = Z[:, jb]; x[:, je] = Z[:, je]; V['both raw z'] = x
rng = np.random.default_rng(11); idxs = []
while len(idxs) < 4000:
    ix = rng.integers(0, n, n)
    if all(0 < Y[ix, j].sum() < n for j in range(12)): idxs.append(ix)
P0 = c4(R6); B0 = np.array([[auc(P0[ix, j], Y[ix, j]) for j in (jb, je)] for ix in idxs])
print('c4 replay gold58 macro %.5f | Baker %.4f Effusion %.4f' % (mac(P0, Y), auc(P0[:, jb], Y[:, jb]), auc(P0[:, je], Y[:, je])))
for k, M in V.items():
    if k == 'c4': continue
    P = c4(M); Bb = np.array([[auc(P[ix, j], Y[ix, j]) for j in (jb, je)] for ix in idxs]); d = (Bb - B0).sum(1) / 12
    print('%-15s macro d %+.5f (Baker %+.4f, Effusion %+.4f) boot sd %.5f P>0 %.3f' % (k, mac(P, Y) - mac(P0, Y),
          auc(P[:, jb], Y[:, jb]) - auc(P0[:, jb], Y[:, jb]), auc(P[:, je], Y[:, je]) - auc(P0[:, je], Y[:, je]), d.std(), (d > 0).mean()))
# R6 alone, per component
print('R6 alone Baker: served (stacker) %.4f | raw z %.4f | state p_model %.4f' % (auc(R6[:, jb], Y[:, jb]), auc(Z[:, jb], Y[:, jb]), auc(st["Baker's|p_model"].values, Y[:, jb])))
print('R6 alone Effusion: served (stacker) %.4f | raw z %.4f' % (auc(R6[:, je], Y[:, je]), auc(Z[:, je], Y[:, je])))
# Effusion: missing-path studies
miss = fl['status'].values != 'ok'
print('Effusion missing-path studies %d (pos %d); stacker z_out mean miss %.2f vs ok %.2f' % (miss.sum(), Y[miss, je].sum(),
      fl['Effusion|z_out'].values[miss].mean(), fl['Effusion|z_out'].values[~miss].mean()))
# R7 family on the same gold rows (its own stacker, fit on its own OOF pool) and on gold74
r7d = r7()
for nm, gg in (('gold58', g58), ('gold74', g)):
    u = gg.StudyInstanceUID.tolist(); yb = gg["Baker's"].values.astype(int); ye = gg['Effusion'].values.astype(int)
    q = r7d.loc[u]
    print('R7 %s Baker: p (served = stacker) %.4f | z raw %.4f | stk z_out %.4f || Effusion: p %.4f | z raw %.4f | fluid z_out %.4f' % (
        nm, auc(q["p|Baker's"].values, yb), auc(q["z|Baker's"].values, yb), auc(q["stk|Baker's|z_out"].values, yb),
        auc(q['p|Effusion'].values, ye), auc(q['z|Effusion'].values, ye), auc(q['fluid|Effusion|z_out'].values, ye)))
