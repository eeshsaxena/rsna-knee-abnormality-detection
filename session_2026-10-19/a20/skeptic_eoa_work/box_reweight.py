# Skeptic transfer check (10-19). Read-only; prints AGGREGATES ONLY (no ids, no rows). Piped via stdin, writes nothing.
# Question: if OAI expert OA labels were coupled at the RSNA expert level (gold74 phi ~.24-.40), does E-OA still gain?
# Method: importance-resample OAI knees by their 3-label OA cell (model predictions given labels kept as observed).
#  R1: match gold74 8-cell joint distribution (smoothed +.5), marginals AND coupling move to gold74.
#  R2: tilt grid w = exp(theta * #pos pairs + beta.Y), beta solved so marginals stay at OAI values; isolates coupling.
import os, numpy as np, pandas as pd
from scipy.stats import rankdata, norm
OA = ['Medial OA', 'Lateral OA', 'PF OA']
COMP = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'],
        'PF OA': ['mp', 'lp', 'amf', 'alf']}
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
def decor(F, a):
    U = np.stack([pct(F[:, k]) for k in range(3)], 1); O = U.copy()
    for k in range(3):
        o = [m for m in range(3) if m != k]; O[:, k] = U[:, k] - a * (U[:, o[0]] + U[:, o[1]]) / 2
    return O
def proxy(src, name):
    s = pd.read_csv(src, dtype={'kid': str}).set_index('kid')
    Y = pd.DataFrame(index=s.index, columns=OA, dtype=float)
    for lab, regs in COMP.items():
        c = s[['m_cm_' + r for r in regs]].astype(float).values
        area = np.floor(c + 1e-6); ft = np.round((c - area) * 10); fin = np.isfinite(c)
        pos = ((ft >= 1) & fin) if name == 'P_FT' else ((area >= 2) & fin)
        y = pos.any(1).astype(float); y[~fin.any(1)] = np.nan; Y[lab] = y
    return Y.dropna()
P = pd.read_csv(os.path.expanduser('~/oai_build/teacher/teacher_D5_mean_v00.csv'), dtype={'StudyInstanceUID': str})
P = P.drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')[OA].dropna()
G74 = {'000': 33, '001': 12, '010': 3, '011': 2, '100': 4, '101': 8, '110': 2, '111': 7}  # gold74 complete (71), local aggregate
cells = [format(i, '03b') for i in range(8)]
def ns(v): return norm.ppf((rankdata(v) - .5) / len(v))
def evalw(F, Y, w, tag, B=300, seed=0):
    rng = np.random.default_rng(seed); n = len(Y); p = w / w.sum()
    ess = 1 / (p ** 2).sum()
    D = {.1: [], .2: [], .3: [], .5: []}; PH = []; RH = []; Q = []
    for b in range(B):
        ii = rng.choice(n, n, p=p); y = Y[ii]; f = F[ii]
        if (y.sum(0) < 5).any(): continue
        PH.append(np.corrcoef(y.T)[np.triu_indices(3, 1)])
        Z = np.stack([ns(f[:, k]) for k in range(3)], 1); RH.append(np.corrcoef(Z.T)[np.triu_indices(3, 1)])
        # q_k = mean_j corr(Z_j, y_k) / corr(Z_k, y_k)  (cross-predictiveness; theory: E-OA helps iff q < rho)
        qq = []
        for k in range(3):
            o = [m for m in range(3) if m != k]; ck = np.corrcoef(Z[:, k], y[:, k])[0, 1]
            qq.append(np.mean([np.corrcoef(Z[:, j], y[:, k])[0, 1] for j in o]) / ck)
        Q.append(qq)
        base = np.array([auc(f[:, k], y[:, k]) for k in range(3)])
        for a in D:
            O = decor(f, a); D[a].append(np.mean([auc(O[:, k], y[:, k]) for k in range(3)]) - base.mean())
    PH = np.mean(PH, 0); RH = np.mean(RH, 0); Q = np.mean(Q, 0)
    s = ' | '.join('a %.1f %+.5f (sd %.5f, P>0 %.2f)' % (a, np.mean(v), np.std(v), np.mean(np.array(v) > 0)) for a, v in D.items())
    print('[%s] ESS %.0f | label phi %s | model rho(ns) %s | q %s | OA-mean dAUC: %s' % (
        tag, ess, np.round(PH, 3).tolist(), np.round(RH, 3).tolist(), np.round(Q, 3).tolist(), s))
for name in ('P_FT', 'P_AREA'):
    Yd = proxy(os.path.expanduser('~/oai_build/oai_label_sources.csv'), name)
    ids = Yd.index.intersection(P.index); F = P.loc[ids].values.astype(float); Y = Yd.loc[ids].values.astype(int)
    key = np.array([''.join(map(str, r)) for r in Y])
    po = pd.Series(key).value_counts(normalize=True).reindex(cells).fillna(0).values
    print('[%s] n %d OAI marginals %s' % (name, len(Y), np.round(Y.mean(0), 3).tolist()))
    evalw(F, Y, np.ones(len(Y)), name + ' R0 unweighted')
    pg = np.array([G74[c] + .5 for c in cells]); pg = pg / pg.sum()
    wc = pg / np.maximum(po, 1e-9); w = wc[[cells.index(k) for k in key]]
    evalw(F, Y, w, name + ' R1 gold74-cell-matched')
    if name != 'P_FT': continue
    pairs = Y[:, 0] * Y[:, 1] + Y[:, 0] * Y[:, 2] + Y[:, 1] * Y[:, 2]
    tgt = Y.mean(0)
    for th in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        beta = np.zeros(3)
        for _ in range(200):
            w = np.exp(th * pairs + Y @ beta); cur = (w[:, None] * Y).sum(0) / w.sum()
            beta += np.log(tgt / (1 - tgt)) - np.log(cur / (1 - cur))
        evalw(F, Y, w, name + ' R2 tilt theta %.1f' % th, B=200)
