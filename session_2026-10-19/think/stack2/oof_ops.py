# oof_ops.py (10-19 stacking lens, Claude, research only, CPU): the SECOND RULER for blend operators.
# Members with true out-of-fold predictions on the same train studies: R7 (heliosli compact family, the R6 sibling; p and z) and our
# S6-recipe seeds s1044 (5 folds), s1045 (4 folds), s1043 (3 folds). Gold studies are excluded by construction (S6 OOF has none).
# Labels: report regime (v4_blend >= .5 on all studies; v2 YES/NO with UNK dropped). Paired study bootstrap of macro deltas.
import sys, json, numpy as np, pandas as pd
from scipy.stats import rankdata, norm
from load import *
src = sys.argv[1] if len(sys.argv) > 1 else 'v4'
d7 = r7(); s44, _ = s6seed(1044); s45, _ = s6seed(1045); s43, _ = s6seed(1043)
lab = labels(src)
ids = [i for i in s44.index if i in d7.index and i in lab.index]
ids3 = [i for i in ids if i in s45.index]
nsc = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
zst = lambda v: (v - v.mean()) / (v.std() + 1e-12)


def op_apply(op, X, w):
    w = np.asarray(w, float); w = w / w.sum(); k = len(X)
    if op == 'rank': return sum(w[i] * pct(X[i]) for i in range(k))
    if op == 'logit': return sum(w[i] * lg(X[i]) for i in range(k))
    if op == 'prob': return sum(w[i] * X[i] for i in range(k))
    if op == 'zlogit': return sum(w[i] * zst(lg(X[i])) for i in range(k))
    if op == 'nscore': return sum(w[i] * nsc(X[i]) for i in range(k))
    if op == 'geo': return sum(w[i] * np.log(pct(X[i])) for i in range(k))
    if op == 'pow2': return sum(w[i] * pct(X[i]) ** 2 for i in range(k))
    if op == 'pow4': return sum(w[i] * pct(X[i]) ** 4 for i in range(k))
    if op == 'invgeo': return -sum(w[i] * np.log(1 - pct(X[i]) + .5 / len(X[i])) for i in range(k))
    if op == 'median': return np.median(np.stack([pct(x) for x in X]), 0)
    if op == 'max': return np.max(np.stack([pct(x) for x in X]), 0)
    if op == 'min': return np.min(np.stack([pct(x) for x in X]), 0)
    raise ValueError(op)


def run(name, ids_, members, w, ops, B=400, seed=0):
    Y = lab.loc[ids_, L].values
    M = [m.loc[ids_, L].values if isinstance(m, pd.DataFrame) else m(ids_) for m in members]
    rng = np.random.default_rng(seed); n = len(ids_)
    boots = [rng.integers(0, n, n) for _ in range(B)]
    res = {}
    def score(P):
        a = []; ab = np.zeros((B, 12))
        for j in range(12):
            y = Y[:, j]; ok = ~np.isnan(y); yy = y[ok].astype(int); s = P[ok, j]
            a.append(auc(s, yy))
            idx_ok = np.where(ok)[0]; pos = {v: i for i, v in enumerate(idx_ok)}
            for b, ix in enumerate(boots):
                ix2 = ix[ok[ix]]; ab[b, j] = auc(P[ix2, j], y[ix2].astype(int))
        return np.array(a), ab
    singles = {}
    for mi, X in enumerate(M):
        singles[mi] = np.mean([auc(X[~np.isnan(Y[:, j]), j], Y[~np.isnan(Y[:, j]), j].astype(int)) for j in range(12)])
    base_op = ops[0]
    P0 = np.column_stack([op_apply(base_op, [X[:, j] for X in M], w) for j in range(12)]); a0, ab0 = score(P0)
    res[base_op] = dict(macro=float(a0.mean()), per=dict(zip(L, map(float, a0))))
    print('[%s] n=%d members single macro %s | base op %s macro %.5f' % (name, n, {k: round(v, 4) for k, v in singles.items()}, base_op, a0.mean()))
    for op in ops[1:]:
        P = np.column_stack([op_apply(op, [X[:, j] for X in M], w) for j in range(12)]); a, ab = score(P)
        dd = (ab - ab0).mean(1); per = a - a0
        npos = int((per > 0).sum()); nneg = int((per < 0).sum())
        res[op] = dict(macro=float(a.mean()), d=float(a.mean() - a0.mean()), sd=float(dd.std()), P_pos=float((dd > 0).mean()),
                       labels_up=npos, labels_down=nneg, per=dict(zip(L, map(float, per))))
        print('  %-7s d %+.5f sd %.5f P>0 %.3f  labels +%d/-%d  biggest %s' % (op, a.mean() - a0.mean(), dd.std(), (dd > 0).mean(), npos, nneg,
              sorted([(L[k], round(per[k], 4)) for k in range(12)], key=lambda t: -abs(t[1]))[:3]))
    return res


OPS = ['rank', 'logit', 'prob', 'zlogit', 'nscore', 'geo', 'pow2', 'pow4', 'invgeo']
R7p = lambda ii: d7.loc[ii, ['p|' + c for c in L]].values
R7z = lambda ii: 1 / (1 + np.exp(-np.clip(d7.loc[ii, ['z|' + c for c in L]].values.astype(float), -40, 40)))
out = {'labels': src, 'n2': len(ids), 'n3': len(ids3)}
out['R7p+s44 .5/.5'] = run('R7p+s44 .5/.5', ids, [R7p, s44], [.5, .5], OPS)
out['R7p+s44 .8/.2'] = run('R7p+s44 .8/.2', ids, [R7p, s44], [.8, .2], OPS)
out['R7z+s44 .5/.5'] = run('R7z+s44 .5/.5', ids, [R7z, s44], [.5, .5], ['rank', 'logit', 'zlogit'])
out['R7p+s44+s45 eq'] = run('R7p+s44+s45 eq', ids3, [R7p, s44, s45], [1, 1, 1], OPS + ['median', 'max', 'min'])
out['R7p+s44+s45 .5/.25/.25'] = run('R7p+s44+s45 .5/.25/.25', ids3, [R7p, s44, s45], [.5, .25, .25], OPS)
json.dump(out, open('oof_ops_%s.json' % src, 'w'), indent=1)
