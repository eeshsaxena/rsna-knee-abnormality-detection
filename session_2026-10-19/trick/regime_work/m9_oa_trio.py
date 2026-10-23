# m9_oa_trio.py: the gap rule restricted to the OA trio (mechanism: reports use "tricompartmental" / facet-vs-compartment wording that
# couples the three OA labels; the expert rubric scores each compartment separately with a strict high-grade threshold).
# Nested 2-fold CV x 1000 as in m7: inner half picks, per OA target, the OA helper with the largest |gap| (> thr), outer half scores.
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/trick/regime_work')
src = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/trick/regime_work/m7_gap_nested.py').read()
exec(src[:src.index('ok = lambda')])
OA = [L.index(c) for c in ('Medial OA', 'Lateral OA', 'PF OA')]
def rule_oa(ix, rep, thr):
    gap = phimat(rep[ix]) - phimat(Y[ix]); sel = {}
    for t in OA:
        cand = [(abs(gap[t, h]), h, np.sign(gap[t, h])) for h in OA if h != t]
        m = max(cand)
        if m[0] > thr: sel[t] = (m[1], m[2])
    return sel
ok = lambda ix: all(0 < Y[ix, j].sum() < len(ix) for j in range(12))
full = np.arange(n); rep = REP['sol56']
for thr in (.1, .2):
    sel = rule_oa(full, rep, thr); print('in-sample thr %.1f picks %s  d a.05 %+.5f a.10 %+.5f' % (thr, {L[t]: (L[h], '-' if s > 0 else '+') for t, (h, s) in sel.items()}, delta(full, sel, .05), delta(full, sel, .1)))
# fixed full decorrelation: each OA label minus a * mean pct of the other two
def decor(ix, a):
    d = 0.
    for t in OA:
        o = [h for h in OA if h != t]
        d += auc(pct(P[ix, t]) - a * .5 * (pct(P[ix, o[0]]) + pct(P[ix, o[1]])), Y[ix, t]) - auc(P[ix, t], Y[ix, t])
    return d / 12
print('in-sample fixed OA decorrelation: ' + '  '.join('a%.2f %+.5f' % (a, decor(full, a)) for a in (.05, .1, .2)))
rng = np.random.default_rng(7); res = {}; cnt = 0
while cnt < 1000:
    perm = rng.permutation(n); h1, h2 = perm[:29], perm[29:]
    if not (ok(h1) and ok(h2)): continue
    cnt += 1
    for inner, outer in [(h1, h2), (h2, h1)]:
        for thr in (.1, .2):
            sel = rule_oa(inner, rep, thr)
            for a in (.05, .1):
                res.setdefault(('gap-rule', thr, a), []).append(delta(outer, sel, a))
        for a in (.05, .1):
            res.setdefault(('fixed decor', '-', a), []).append(decor(outer, a))
for k, v in res.items():
    v = np.array(v); print('NESTED %-12s thr %s a %.2f: mean %+.5f sd %.5f P>0 %.3f P<0 %.3f' % (k[0], k[1], k[2], v.mean(), v.std(), (v > 0).mean(), (v < 0).mean()))
