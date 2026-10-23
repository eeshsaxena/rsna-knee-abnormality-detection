# m7_gap_nested.py (10-19 trick lens, research only, CPU): NESTED test of the regime-transform THEORY, with no peeking at the outer
# half's predictions: on the inner half, measure the LABEL-LEVEL coupling gap per ordered pair (t, h):
#     gap = phi_report(t, h) - phi_expert(t, h)       (report = reader >= .5 on the SAME inner studies; expert = gold58 labels)
# Report-trained models inherit the report coupling, so where gap > thr the theory says "subtract the helper" and where gap < -thr
# "add the helper". Per target take the single helper with the largest |gap| (if > thr), transform the FINAL c4 output
#     s_t' = pct(s_t) - sign(gap) * a * pct(s_h)
# and score the macro delta vs c4 on the OUTER half only. 2-fold CV repeated (1000 random halvings x 2 directions).
# Variants: reader v4blend / sol56; thr .2 / .3; a .05 / .1; plus "global" = the same rule with the gap computed on ALL 58 (in-sample,
# for reference) and "fixed LatOA<-PF" = only the host-mechanism pair.
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G, R
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(ids)[L].astype(float).values
R6 = rd(G + '/R6_gold58_serve.csv'); S6 = rd(G + '/n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
P = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); b = w * rr + (1 - w) * pct(R6[:, j]); P[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
LB = R + '/_labels/'
REP = {'v4blend': (np.nan_to_num(rd(LB + 'rsna-knee-llm-report-labels/llm_labels_v4_blend.csv'), nan=0) >= .5).astype(int),
       'sol56': (np.nan_to_num(rd(LB + 'rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv'), nan=0) >= .5).astype(int)}


def phimat(X):
    X = X.astype(float); sd = X.std(0); C = np.corrcoef(X.T); C[np.ix_(sd == 0, sd == 0)] = 0
    C[sd == 0, :] = 0; C[:, sd == 0] = 0; return C


def rule(ix, rep, thr):
    gap = phimat(rep[ix]) - phimat(Y[ix]); np.fill_diagonal(gap, 0); sel = {}
    for t in range(12):
        h = int(np.argmax(np.abs(gap[t])))
        if abs(gap[t, h]) > thr: sel[t] = (h, np.sign(gap[t, h]))
    return sel


def delta(ix, sel, a):
    d = 0.
    for t, (h, sg) in sel.items():
        d += auc(pct(P[ix, t]) - sg * a * pct(P[ix, h]), Y[ix, t]) - auc(P[ix, t], Y[ix, t])
    return d / 12


ok = lambda ix: all(0 < Y[ix, j].sum() < len(ix) for j in range(12))
full = np.arange(n)
for rname, rep in REP.items():
    for thr in (.2, .3):
        sel = rule(full, rep, thr)
        print('[%s thr %.1f] in-sample rule picks: %s' % (rname, thr, {L[t]: (L[h], '-' if s > 0 else '+') for t, (h, s) in sel.items()}))
        print('   in-sample macro delta a.05 %+.5f a.10 %+.5f' % (delta(full, sel, .05), delta(full, sel, .1)))
rng = np.random.default_rng(42); res = {}; cnt = 0
while cnt < 1000:
    perm = rng.permutation(n); h1, h2 = perm[:29], perm[29:]
    if not (ok(h1) and ok(h2)): continue
    cnt += 1
    for inner, outer in [(h1, h2), (h2, h1)]:
        for rname, rep in REP.items():
            for thr in (.2, .3):
                sel = rule(inner, rep, thr)
                for a in (.05, .1):
                    res.setdefault((rname, thr, a), []).append(delta(outer, sel, a))
        jl, jp = L.index('Lateral OA'), L.index('PF OA')
        for a in (.05, .1):
            res.setdefault(('fixed LatOA<- -PF', '-', a), []).append(delta(outer, {jl: (jp, 1.)}, a))
print('NESTED outer-half macro delta vs c4 (2-fold CV x 1000):')
for k, v in res.items():
    v = np.array(v); print('   reader %-18s thr %s a %.2f: mean %+.5f sd %.5f P>0 %.3f' % (k[0], k[1], k[2], v.mean(), v.std(), (v > 0).mean()))
