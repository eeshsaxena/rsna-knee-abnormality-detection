# census_lib.py (10-20 push, Claude): helpers shared by census_d.py / census_e.py (same definitions as census.py).
import sys, functools
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from cbase import load, L, AB, pct
import load as LD
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def spear(a, b): return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


TG = np.linspace(-.995, .995, 3981); I0 = int(np.argmin(np.abs(TG))); DT = np.diff(TG)


@functools.lru_cache(maxsize=None)
def tet_counts(c00, c01, c10, c11):
    c00, c01, c10, c11 = c00 + .5, c01 + .5, c10 + .5, c11 + .5; N = c00 + c01 + c10 + c11
    pa = (c10 + c11) / N; pb = (c01 + c11) / N; p11 = c11 / N
    h = norm.ppf(1 - pa); k = norm.ppf(1 - pb); s = 1 - TG ** 2
    f = np.exp(-(h * h - 2 * TG * h * k + k * k) / (2 * s)) / (2 * np.pi * np.sqrt(s))
    C = np.concatenate([[0.], np.cumsum((f[1:] + f[:-1]) / 2 * DT)]); C -= C[I0]
    return float(np.interp(p11, pa * pb + C, TG))


def tet(a, b):
    a = np.asarray(a, int); b = np.asarray(b, int)
    return tet_counts(int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum()))


def K(s):
    s = np.asarray(s); return (s[:, None] > s[None, :]).astype(float) + .5 * (s[:, None] == s[None, :])


def tr_score(F, t, o, a): return pct(pct(F[:, t]) - a * pct(F[:, o]))


def wdelta(Dm, y, W):
    Wp = W * y; Wn = W * (1 - y); den = Wp.sum(1) * Wn.sum(1)
    num = np.einsum('si,ij,sj->s', Wp, Dm, Wn)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def data():
    d = load(); g74 = d['g74']; allg = set(g74.StudyInstanceUID)
    r7 = LD.r7(); R7cols = ['p|' + c for c in L]
    s44, _ = LD.s6seed(1044); s45, _ = LD.s6seed(1045); lv2 = LD.labels('v2'); lv4 = LD.labels('v4')
    tr = [i for i in r7.index if i not in allg and i in lv2.index]
    tr3 = [i for i in tr if i in s44.index and i in s45.index]
    M3 = np.column_stack([np.mean([pct(X[:, j]) for X in (r7.loc[tr3, R7cols].values.astype(float), s44.loc[tr3, L].values, s45.loc[tr3, L].values)], 0)
                          for j in range(12)])
    return dict(CUR=d['eoa20'], C4=d['c4'], Y58=d['Y58'], Y74=g74[L].values.astype(float), R7_74=r7.reindex(d['ids74'])[R7cols].values.astype(float),
                R7_58=r7.reindex(d['ids'])[R7cols].values.astype(float), F7=r7.loc[tr, R7cols].values.astype(float), Yv2=lv2.loc[tr, L].values,
                Yv4=lv4.reindex(tr)[L].values, M3=M3, Y3v2=lv2.loc[tr3, L].values, Y3v4=lv4.reindex(tr3)[L].values, members=d['members'], ids=d['ids'])
