# tie_pure.py (10-19 trick audit, Claude, research only): PURE float32-saturation tie effect of the heliosli compact family.
# R7 OOF (non-gold train) raw fold-mean logit z per label -> p32 = float32 sigmoid (what compact_infer writes, MEASURED == R6 gold58
# serve to 1.2e-7) vs z itself. Report-regime labels (v2 / v4 / sol56). Also inside a 2-leg .5/.5 rank blend with a REAL partner
# (our S6 seed s1044 OOF, same train studies).
import sys; sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
import numpy as np, pandas as pd
from scipy.stats import rankdata
from load import *
d = r7(); g, g58 = gold()
d = d.loc[~d.index.isin(set(g.StudyInstanceUID))]
sig32 = lambda z: (1 / (1 + np.exp(-z.astype(np.float32)))).astype(np.float32)
try:
    s44, _ = s6seed(1044)
except Exception as e:
    s44 = None; print('no s1044', e)
for src in ('v2', 'v4', 'sol56'):
    lab = labels(src).reindex(d.index)
    tot1, tot2, tot3 = [], [], []
    rows = []
    for c in L:
        z = d['z|' + c].values.astype(np.float64); p = sig32(z).astype(np.float64)
        y = lab[c].values; m = np.isfinite(y)
        yy = y[m].astype(int); zz = z[m]; pp = p[m]
        if yy.sum() < 5: continue
        top = pp >= 1 - 1e-6
        a_p, a_z = auc(pp, yy), auc(zz, yy)
        tot1.append(a_z - a_p)
        line = '%-17s n %4d pos %4d | top-sat %4d (neg %3d) | AUC p32 %.5f z %.5f d %+.6f' % (c, m.sum(), yy.sum(), top.sum(), (top & (yy == 0)).sum(), a_p, a_z, a_z - a_p)
        if s44 is not None:
            q = s44.reindex(d.index)[c].values[m]; mm = np.isfinite(q)
            bp = .5 * pct(pp[mm]) + .5 * pct(q[mm]); bz = .5 * pct(zz[mm]) + .5 * pct(q[mm])
            db = auc(bz, yy[mm]) - auc(bp, yy[mm]); tot2.append(db)
            line += ' | blend w/ s1044 d %+.6f' % db
        rows.append(line)
    print('== labels', src); print('\n'.join(rows))
    print('macro single-leg z - p32: %+.6f | blend: %+.6f' % (np.mean(tot1), np.mean(tot2) if tot2 else np.nan))
# hidden-size expectation: fraction of cells saturated on the full R7 OOF
for c in L:
    z = d['z|' + c].values; p = sig32(z)
    print('%-17s frac p32 >= 1-1e-6: %.3f  frac p32 == 1: %.3f  frac tied (any value) %.3f' % (c, (p >= 1 - 1e-6).mean(), (p == 1).mean(),
          1 - len(np.unique(p)) / len(p)))
