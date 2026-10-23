# skeptic_facet_rubric2.py: paired bootstrap of the lateral minus medial facet AUC AFTER removing the model's PF output
# (residual on PF with the slope fitted on the label-negative pool), plus the residual AUCs' own bootstrap. Competition data only.
import sys, io, contextlib, numpy as np, pandas as pd
LEV = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/levers'
src = open(LEV + '/skeptic_facet_rubric.py', encoding='utf-8').read().split("print('== (1)")[0]
with contextlib.redirect_stdout(io.StringIO()):
    exec(src)
rng = np.random.default_rng(7)
for m in M:
    out = {}
    for side, lab in SIDE.items():
        cls = CLS[side]; neg = labneg(cls.index, lab); ii = neg.intersection(U[m].index)
        pool = labneg(U[m].index.intersection(pd.Index(list(SS))), lab).difference(pd.Index(list(gold)))
        bb = np.polyfit(U[m].loc[pool, 'PF OA'].values, U[m].loc[pool, lab].values, 1)[0]
        out[side] = (ii, (cls.loc[ii] == 'facet_only').astype(int).values, U[m].loc[ii, lab].values - bb * U[m].loc[ii, 'PF OA'].values)
    P = out['lateral'][0].union(out['medial'][0]); pos = {u: i for i, u in enumerate(P)}
    idx = {s: np.array([pos[u] for u in out[s][0]]) for s in out}
    a0 = {s: auc(out[s][2], out[s][1]) for s in out}
    bs = []
    for _ in range(2000):
        cnt = np.bincount(rng.integers(0, len(P), len(P)), minlength=len(P)); r = {}
        for s in out:
            y = np.repeat(out[s][1], cnt[idx[s]]); x = np.repeat(out[s][2], cnt[idx[s]])
            r[s] = auc(x, y) if 0 < y.sum() < len(y) else np.nan
        bs.append((r['lateral'], r['medial']))
    bs = np.array(bs); d = bs[:, 0] - bs[:, 1]; d = d[np.isfinite(d)]
    print(f"{m:6s} PF-residual facet AUC lateral {a0['lateral']:.3f} (P>.5 {np.nanmean(bs[:,0] > .5):.2f}) | medial {a0['medial']:.3f} "
          f"| lat - med {a0['lateral'] - a0['medial']:+.3f} sd {d.std():.3f} P(>0) {np.mean(d > 0):.2f}")
