# rg_lib.py (10-21 levers, Claude, REGIME track). RESEARCH ONLY, local CPU, NO OAI. Shared helpers for regime.py:
# loaders (RSNA train OOF R7 / s1044 / s1045, report-v2 labels, gold74 expert labels, c4 gold58 final output), transforms (E-OA sym / pw,
# directional line t <- o), weighted AUC, tetrachoric, and Gaussian-copula target tables for raking report labels to an expert regime
# whose pairwise tetrachoric matrix is lambda x gold74 (lambda < 1 = experts couple less).
import sys
import numpy as np
from scipy.stats import norm, rankdata, multivariate_normal
from scipy.optimize import brentq
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
import load as LD  # noqa: E402

L = LD.L
J = {c: i for i, c in enumerate(L)}
AB = {"ACL": "ACL", "MCL": "MCL", "Medial Meniscus": "MM", "Lateral Meniscus": "LM", "Medial OA": "MOA", "Lateral OA": "LOA",
      "PF OA": "PF", "Effusion": "Eff", "Synovitis": "Syn", "Baker's": "Bak", "Contusion": "Cont", "Fracture": "Frac"}
OA = ['Medial OA', 'Lateral OA', 'PF OA']
W_PW = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}
PUB = {'eoa10': 5.9, 'eoa20': 10.3, 'eoa30': 9.6, 'pw20': 10.4, 'pw30': 11.3}   # CLAIMED val/implied2.md joint refit, e-4 vs c4
SH = {'eoa10': (.1, 'sym'), 'eoa20': (.2, 'sym'), 'eoa30': (.3, 'sym'), 'pw20': (.2, 'pw'), 'pw30': (.3, 'pw')}
pct = lambda v: rankdata(v) / len(v)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))


def eoa(F, a, mode):
    G = F.copy(); U = {c: pct(F[:, J[c]]) for c in OA}
    for c in OA:
        o = [k for k in OA if k != c]
        w = [W_PW[tuple(sorted((c, k), key=OA.index))] for k in o] if mode == 'pw' else [1., 1.]
        G[:, J[c]] = pct(U[c] - a / 2.0 * (w[0] * U[o[0]] + w[1] * U[o[1]]))
    return G


def line_col(F, t, o, a):
    return pct(pct(F[:, J[t]]) - a * pct(F[:, J[o]]))


def wauc(s, y, w):
    o = np.argsort(s, kind='mergesort'); s, y, w = s[o], y[o], w[o]
    wp = w * y; wn = w * (1 - y)
    _, inv = np.unique(s, return_inverse=True)
    gp = np.bincount(inv, wp); gn = np.bincount(inv, wn); cn = np.cumsum(gn) - gn
    return float((gp * (cn + .5 * gn)).sum() / (wp.sum() * wn.sum()))


def tet_counts(c00, c01, c10, c11, sm=.5):
    c00, c01, c10, c11 = c00 + sm, c01 + sm, c10 + sm, c11 + sm; N = c00 + c01 + c10 + c11
    h = norm.ppf(1 - (c11 + c10) / N); k = norm.ppf(1 - (c11 + c01) / N); p11 = c11 / N
    f = lambda r: multivariate_normal.cdf([-h, -k], mean=[0, 0], cov=[[1, r], [r, 1]]) - p11
    if f(-.995) > 0: return -.995
    if f(.995) < 0: return .995
    return float(brentq(f, -.995, .995, xtol=1e-5))


def tet_xy(a, b):
    m = ~(np.isnan(a) | np.isnan(b)); a = a[m].astype(int); b = b[m].astype(int)
    return tet_counts(((a == 0) & (b == 0)).sum(), ((a == 0) & (b == 1)).sum(), ((a == 1) & (b == 0)).sum(), ((a == 1) & (b == 1)).sum())


def counts_xy(a, b):
    m = ~(np.isnan(a) | np.isnan(b)); a = a[m].astype(int); b = b[m].astype(int)
    return np.array([((a == 0) & (b == 0)).sum(), ((a == 0) & (b == 1)).sum(), ((a == 1) & (b == 0)).sum(), ((a == 1) & (b == 1)).sum()], float)


def tet_matrix(Y):
    T = np.eye(Y.shape[1])
    for i in range(Y.shape[1]):
        for j in range(i + 1, Y.shape[1]):
            T[i, j] = T[j, i] = tet_xy(Y[:, i], Y[:, j])
    return T


def near_pd(R, eps=1e-3):
    w, V = np.linalg.eigh(R); w = np.maximum(w, eps); R2 = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(R2)); R2 = R2 / np.outer(d, d); np.fill_diagonal(R2, 1.); return R2


def lor_to_tet(p1, p2, lor):
    lo = max(0., p1 + p2 - 1) + 1e-9; hi = min(p1, p2) - 1e-9
    g = lambda p11: np.log(p11) + np.log(1 - p1 - p2 + p11) - np.log(p1 - p11) - np.log(p2 - p11) - lor
    p11 = brentq(g, lo, hi); N = 1e6
    c = np.array([1 - p1 - p2 + p11, p2 - p11, p1 - p11, p11]) * N
    return tet_counts(*c, sm=0.)


_CACHE = {}


def copula_cells(marg, R):
    """2^k cell probabilities (pattern index = sum b_i 2^(k-1-i), first label most significant) for a Gaussian copula."""
    k = len(marg); tau = norm.ppf(1 - np.asarray(marg))
    key = (tuple(np.round(marg, 5)), tuple(np.round(R, 5).ravel()))
    if key in _CACHE: return _CACHE[key]
    out = np.zeros(2 ** k)
    for idx in range(2 ** k):
        b = [(idx >> (k - 1 - i)) & 1 for i in range(k)]
        s = np.array([-1. if bi else 1. for bi in b]); u = s * tau
        C = R * np.outer(s, s)
        if k == 1:
            out[idx] = norm.cdf(u[0])
        else:
            out[idx] = multivariate_normal.cdf(u, mean=np.zeros(k), cov=C, abseps=1e-7, releps=1e-6, maxpts=200000 * k)
    out = np.maximum(out, 1e-12); out /= out.sum(); _CACHE[key] = out
    return out


def rake_ipf(T, margins, it=300):
    T = T.copy(); k = T.ndim
    for _ in range(it):
        for ax in range(k):
            cur = T.sum(axis=tuple(i for i in range(k) if i != ax)); sh = [1] * k; sh[ax] = 2
            T = T * (np.array([1 - margins[ax], margins[ax]]) / cur).reshape(sh)
    return T / T.sum()


def weights_from_target(Ysub, tgt_flat):
    k = Ysub.shape[1]; idx = (Ysub * (2 ** np.arange(k - 1, -1, -1))).sum(1).astype(int)
    obs = np.bincount(idx, minlength=2 ** k).astype(float); obs /= obs.sum()
    wc = np.where(obs > 0, tgt_flat / np.maximum(obs, 1e-12), 0.)
    w = wc[idx]; return w / w.mean()


def weights_ipf_g74(Ysub, cols, Y74):
    jc = [J[c] for c in cols]; ok = ~np.isnan(Y74[:, jc]).any(1); Yg = Y74[ok][:, jc].astype(int)
    T = np.zeros((2,) * len(cols))
    for r in Yg: T[tuple(r)] += 1
    tgt = rake_ipf(T + .5, Ysub.mean(0))
    return weights_from_target(Ysub, tgt.ravel())


def load_all():
    from cbase import load
    d0 = load(); g74 = d0['g74']; allg = set(g74.StudyInstanceUID)
    Y74 = g74[L].values.astype(float)
    r7 = LD.r7(); R7c = ['p|' + c for c in L]
    s44, _ = LD.s6seed(1044); s45, _ = LD.s6seed(1045); lv2 = LD.labels('v2')
    tr = [i for i in r7.index if i not in allg and i in lv2.index]
    tr3 = [i for i in tr if i in s44.index and i in s45.index]
    X = {'R7': r7.loc[tr3, R7c].values.astype(float), 's1044': s44.loc[tr3, L].values.astype(float),
         's1045': s45.loc[tr3, L].values.astype(float)}
    X['3mean'] = np.column_stack([np.mean([pct(X[k][:, j]) for k in ('R7', 's1044', 's1045')], 0) for j in range(12)])
    Y = lv2.loc[tr3, L].values.astype(float)
    return dict(X=X, Y=Y, Y74=Y74, c4=d0['c4'], Y58=d0['Y58'].astype(float), tr3=tr3, members=d0['members'])


def wauc_multi(s, y, Ws):
    """weighted AUC of one score vector against one label vector for several weight vectors (same rows)."""
    o = np.argsort(s, kind='mergesort'); so = s[o]; yo = y[o]
    _, inv = np.unique(so, return_inverse=True); out = []
    for w in Ws:
        wo = w[o]; wp = wo * yo; wn = wo * (1 - yo)
        gp = np.bincount(inv, wp); gn = np.bincount(inv, wn); cn = np.cumsum(gn) - gn
        out.append(float((gp * (cn + .5 * gn)).sum() / (wp.sum() * wn.sum())))
    return np.array(out)


from scipy.special import owens_t as _owt
_RG = np.linspace(-.995, .995, 801)


def bvn_upper_grid(h, k, r):
    """P(Z1 > h, Z2 > k) for correlation array r (Owen's T; vectorized over r)."""
    def Phi2(x, y, rr):   # P(Z1 <= x, Z2 <= y)
        s = np.sqrt(1 - rr ** 2)
        eps = 1e-12
        x_ = x if abs(x) > eps else eps; y_ = y if abs(y) > eps else eps
        a1 = (y_ - rr * x_) / (x_ * s); a2 = (x_ - rr * y_) / (y_ * s)
        bb = 0.5 if (x_ * y_ < 0) else 0.0
        return 0.5 * (norm.cdf(x_) + norm.cdf(y_)) - _owt(x_, a1) - _owt(y_, a2) - bb
    return 1 - norm.cdf(h) - norm.cdf(k) + Phi2(h, k, r)


def tet_fast(c00, c01, c10, c11, sm=.5):
    c00, c01, c10, c11 = c00 + sm, c01 + sm, c10 + sm, c11 + sm; N = c00 + c01 + c10 + c11
    h = norm.ppf(1 - (c11 + c10) / N); k = norm.ppf(1 - (c11 + c01) / N); p11 = c11 / N
    g = bvn_upper_grid(h, k, _RG)
    if p11 <= g[0]: return -.995
    if p11 >= g[-1]: return .995
    return float(np.interp(p11, g, _RG))


def tet_xy_fast(a, b):
    m = ~(np.isnan(a) | np.isnan(b)); a = a[m].astype(int); b = b[m].astype(int)
    return tet_fast(((a == 0) & (b == 0)).sum(), ((a == 0) & (b == 1)).sum(), ((a == 1) & (b == 0)).sum(), ((a == 1) & (b == 1)).sum())


_ZMC = {}


def copula_cells_mc(marg, R, n=400000, seed=11):
    k = len(marg)
    if k not in _ZMC: _ZMC[k] = np.random.default_rng(seed).standard_normal((n, k))
    Lc = np.linalg.cholesky(R); Xs = _ZMC[k] @ Lc.T; tau = norm.ppf(1 - np.asarray(marg))
    idx = ((Xs > tau) * (2 ** np.arange(k - 1, -1, -1))).sum(1)
    out = np.bincount(idx, minlength=2 ** k).astype(float) + 1e-9
    return out / out.sum()
