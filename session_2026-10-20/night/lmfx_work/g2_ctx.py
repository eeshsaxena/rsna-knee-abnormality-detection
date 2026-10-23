# g2_ctx.py (10-20 night, Track 3, Claude). RESEARCH ONLY, local CPU, competition data only.
# Context-dependent reliability ("heteroscedastic shrink") for Lateral Meniscus and Fracture, read on three local rulers:
#   (i) gold58 c4 (descriptive; n 58), (ii) R7 OOF on gold74 (n 73, mostly gold58), (iii) R7 OOF on non-gold train vs
#   report_labels_v2 / sol56 (report regime, n ~3,800). Transform (deployable on the final c4 columns, 0 runtime):
#   ctx  = mean pct of the context columns (LM: ACL, Contusion, Fracture = acute-trauma context; Fracture: Contusion)
#   out  = .5 + (pct(f) - .5) * (1 - k * ctx)      k in {.2, .4, .6}; a monotone map inside any fixed context value.
# Also the additive decor out = pct(f) - a * ctx (a .1 / .2) for comparison, and a placebo context (Baker's + Medial OA + PF OA).
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G, R, labels
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); n = len(ids); Y = g58[L].values.astype(int)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
C4 = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4W.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j])
    C4[:, j] = .8 * pct(w * rr + (1 - w) * pct(R6[:, j])) + .2 * pct(S6[:, j])
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
F7 = r7[['p|' + c for c in L]].set_axis(L, axis=1)
CTX = {'Lateral Meniscus': {'trauma(ACL,Cont,Frac)': ['ACL', 'Contusion', 'Fracture'], 'ACL': ['ACL'],
                            'placebo(Bak,MOA,PF)': ["Baker's", 'Medial OA', 'PF OA']},
       'Fracture': {'Contusion': ['Contusion'], 'DEG(MOA,LOA,PF)': ['Medial OA', 'Lateral OA', 'PF OA'],
                    'placebo(Bak,MM,Syn)': ["Baker's", 'Medial Meniscus', 'Synovitis']}}
TR = [('shrink', k) for k in (.2, .4, .6)] + [('decor', a) for a in (.1, .2)]


def apply(F, lab, ctx, kind, k):
    f = pct(F[:, L.index(lab)]); cx = np.mean([pct(F[:, L.index(c)]) for c in ctx], 0)
    return .5 + (f - .5) * (1 - k * cx) if kind == 'shrink' else f - k * cx


def read(F, Yv, tag, B=0, seed=0):
    out = []
    for lab, cs in CTX.items():
        j = L.index(lab); y = Yv[:, j]; ok = ~np.isnan(y); y = y[ok].astype(int); Fk = F[ok]
        base = auc(Fk[:, j], y)
        for cn, ctx in cs.items():
            for kind, k in TR:
                o = apply(Fk, lab, ctx, kind, k); d = auc(o, y) - base
                s = ''
                if B:
                    rng = np.random.default_rng(seed); bs = []
                    for _ in range(B):
                        ix = rng.integers(0, len(y), len(y))
                        if 0 < y[ix].sum() < len(ix): bs.append(auc(o[ix], y[ix]) - auc(Fk[ix, j], y[ix]))
                    bs = np.array(bs); s = ' sd %.4f P>0 %.2f' % (bs.std(), (bs > 0).mean())
                out.append((tag, lab, cn, kind, k, len(y), int(y.sum()), base, d, s))
    for r in out:
        print('[%s] %-16s ctx %-22s %-6s %.1f | n %4d pos %3d base %.4f | dAUC(label) %+.4f%s' % r)


read(C4, Y.astype(float), 'gold58 c4', B=1000)
read(R6, Y.astype(float), 'gold58 R6', B=0)
read(S6, Y.astype(float), 'gold58 S6fall', B=0)
g74 = g.set_index('StudyInstanceUID'); i74 = g74.index.intersection(F7.index)
read(F7.loc[i74].values, g74.loc[i74, L].values.astype(float), 'gold74 R7oof', B=500)
gold_ids = set(g.StudyInstanceUID)
for src in ('v2', 'sol56'):
    lab = labels(src); ii = [u for u in lab.index.intersection(F7.index) if u not in gold_ids]
    read(F7.loc[ii].values, lab.loc[ii, L].values, 'trainOOF R7 vs ' + src, B=200)
