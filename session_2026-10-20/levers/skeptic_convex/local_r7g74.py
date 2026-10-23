# skeptic_convex/local_r7g74.py (10-20 skeptic of the convex E-OA proposal). Competition data only.
# R7 OOF (report-trained single model, out of fold) x gold74 EXPERT labels: linear vs p2 E-OA at several a, sym and pw.
# Split: gold58 part (labels shared with the c4 gold58 replay) and the 16 extra studies (fresh labels).
import numpy as np, pandas as pd
from scipy.stats import rankdata
D = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58/'
OA = ['Medial OA', 'Lateral OA', 'PF OA']
PW = {(0, 1): .876, (0, 2): .708, (1, 2): 1.416}
pct = lambda v: rankdata(v) / len(v)
rng = np.random.default_rng(7)

def auc(s, y):
    m = ~np.isnan(y); s = s[m]; y = y[m].astype(int)
    p = y.sum(); q = len(y) - p
    if p == 0 or q == 0: return np.nan
    r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)

def eoah(U, a, p=1., pw=False):
    S = U ** p; out = np.empty_like(U)
    for t in range(3):
        o = [k for k in range(3) if k != t]
        ww = [PW[tuple(sorted((t, k)))] for k in o] if pw else [1., 1.]
        out[:, t] = U[:, t] - a * (ww[0] * S[:, o[0]] + ww[1] * S[:, o[1]]) / 2.
    return out

g = pd.read_csv(D + 'gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
o = pd.read_csv(D + 'R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
# percentiles on the FULL train OOF (as a deployed model would rank within a test-sized population)
Ufull = np.stack([pct(o['p|' + c].values) for c in OA], 1)
Udf = pd.DataFrame(Ufull, index=o.index, columns=OA)

def oamean(S, Y):
    v = [auc(S[:, k], Y[:, k]) for k in range(3)]
    return np.nanmean(v)

for subset in ('gold74', 'gold58', 'extra16'):
    gg = g if subset == 'gold74' else (g[g.source == 'original_gold58'] if subset == 'gold58' else g[g.source != 'original_gold58'])
    ids = gg.StudyInstanceUID.tolist(); Y = gg[OA].values.astype(float)
    # recompute pct within the subset (deployment-like: pct within the scored set) AND with full-train pct
    for ref in ('within', 'trainpct'):
        U = np.stack([pct(o.loc[ids, 'p|' + c].values) for c in OA], 1) if ref == 'within' else Udf.loc[ids].values
        base = oamean(U, Y)
        rows = []
        for pw in (False, True):
            for p in (1., 2.):
                rows.append('%s p%.0f a.1/.2/.25/.3 %s' % ('pw' if pw else 'sym', p,
                            np.round([(oamean(eoah(U, a, p, pw), Y) - base) * 3 / 12 for a in (.1, .2, .25, .3)], 5).tolist()))
        print('%-8s %-8s n %d prev %s base OA-mean %.4f | macro deltas: %s' % (subset, ref, len(ids), np.nansum(Y, 0).tolist(), base, ' | '.join(rows)))
    # paired bootstrap p2 - lin at a .2 (sym, pw), within-subset pct fixed
    U = np.stack([pct(o.loc[ids, 'p|' + c].values) for c in OA], 1)
    d = {k: [] for k in ('sym.2', 'pw.2', 'sym.25', 'pw.25')}
    for b in range(3000):
        ix = rng.integers(0, len(ids), len(ids))
        for k in d:
            pw = k.startswith('pw'); a = float(k[-3:] if k.endswith('.25') else k[-2:])
            d[k].append((oamean(eoah(U, a, 2., pw)[ix], Y[ix]) - oamean(eoah(U, a, 1., pw)[ix], Y[ix])) * 3 / 12)
    print('%-8s paired boot p2 - lin (macro): ' % subset + ' | '.join('%s %+.5f (sd %.5f P>0 %.2f)' % (k, np.nanmean(v), np.nanstd(v), np.nanmean(np.array(v) > 0)) for k, v in d.items()))
