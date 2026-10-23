# box_ladder_transfer_blend.py: same as box_ladder_transfer.py but for the equal pct-blend of D5 + s1044 + s1045 (c4-like multi-family blend). (10-20 skeptic of idea 8, Claude). Read-only, AGGREGATES ONLY, writes nothing on the box.
# Question: does the E-OA "robust across 5 MOAKS definitions" claim survive at RSNA expert coupling, i.e. after the skeptic_eoa
# R1 reweight (OAI knees importance-resampled to the gold74 8-cell OA joint), for each definition x each OAI-free model?
# Also R2 tilt (coupling only, OAI marginals kept) at theta 1.0 / 1.5.
# Pre-stated read: robustness at the operative regime holds if R1 a .1 > 0 with P >= .9 in >= 13 of 15 cells AND a .2 > 0 with
# P >= .9 in >= 12 of 15 cells. Otherwise the raw ladder does not support the size/sign claim beyond skeptic_eoa D5.
import os, numpy as np, pandas as pd
from scipy.stats import rankdata, norm
H = os.path.expanduser('~/oai_build')
OA = ['Medial OA', 'Lateral OA', 'PF OA']
COMP = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'], 'PF OA': ['mp', 'lp', 'amf', 'alf']}
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
def decor(F, a):
    U = np.stack([pct(F[:, k]) for k in range(3)], 1); O = U.copy()
    for k in range(3):
        o = [m for m in range(3) if m != k]; O[:, k] = U[:, k] - a * (U[:, o[0]] + U[:, o[1]]) / 2
    return O
def ns(v): return norm.ppf((rankdata(v) - .5) / len(v))
AL = (.1, .2, .3)
def evalw(F, Y, w, B=200, seed=0):
    rng = np.random.default_rng(seed); n = len(Y); p = w / w.sum(); ess = 1 / (p ** 2).sum()
    D = {a: [] for a in AL}; PH = []; RH = []; Q = []
    for b in range(B):
        ii = rng.choice(n, n, p=p); y = Y[ii]; f = F[ii]
        if (y.sum(0) < 5).any() or (y.sum(0) > n - 5).any(): continue
        PH.append(np.corrcoef(y.T)[np.triu_indices(3, 1)].mean())
        Z = np.stack([ns(f[:, k]) for k in range(3)], 1); RH.append(np.corrcoef(Z.T)[np.triu_indices(3, 1)].mean())
        qq = []
        for k in range(3):
            o = [m for m in range(3) if m != k]; ck = np.corrcoef(Z[:, k], y[:, k])[0, 1]
            qq.append(np.mean([np.corrcoef(Z[:, j], y[:, k])[0, 1] for j in o]) / ck)
        Q.append(np.mean(qq))
        base = np.mean([auc(f[:, k], y[:, k]) for k in range(3)])
        for a in AL:
            O = decor(f, a); D[a].append(np.mean([auc(O[:, k], y[:, k]) for k in range(3)]) - base)
    s = ' '.join('a%.1f %+.4f(sd %.4f,P %.2f)' % (a, np.mean(v), np.std(v), np.mean(np.array(v) > 0)) for a, v in D.items())
    return 'ESS %4.0f phi %.2f rho %.3f q %.3f gap %+.3f | %s' % (ess, np.mean(PH), np.mean(RH), np.mean(Q), np.mean(RH) - np.mean(Q), s)
rd = lambda f: pd.read_csv(f, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')[OA]
P = {'D5': rd(H + '/teacher/teacher_D5_mean_v00.csv'), 's1044': rd(H + '/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv'),
     's1045': rd(H + '/teacher2/f0_s1045_ttr288frac75__raw__A__gpu.csv')}
S = pd.read_csv(H + '/oai_label_sources.csv', dtype={'kid': str}).set_index('kid')
common = S.index
for k in P: common = common.intersection(P[k].dropna().index)
S = S.loc[common]
lev = {}; fin_all = np.ones(len(S), bool)
for c, regs in COMP.items():
    cc = S[['m_cm_' + r for r in regs]].astype(float).values; fin_all &= np.isfinite(cc).all(1)
    area = np.floor(cc + 1e-6); ft = np.round((cc - area) * 10)
    for nm, v in {'area1+': (area >= 1).any(1), 'area2+': (area >= 2).any(1), 'FT': (ft >= 1).any(1),
                  'area2&FT': ((area >= 2) & (ft >= 1)).any(1), 'FT2+': (ft >= 2).any(1)}.items():
        lev.setdefault(nm, {})[c] = v.astype(int)
ok = fin_all; ids = common[ok]; print('n %d' % ok.sum())
PB = {'blend3': pd.DataFrame(np.mean([np.column_stack([pct(P[k].loc[ids, c].values) for c in OA]) for k in P], 0), index=ids, columns=OA)}
G74 = {'000': 33, '001': 12, '010': 3, '011': 2, '100': 4, '101': 8, '110': 2, '111': 7}
cells = [format(i, '03b') for i in range(8)]
pg = np.array([G74[c] + .5 for c in cells]); pg = pg / pg.sum()
for nm, Yd in lev.items():
    Y = np.column_stack([Yd[c][ok] for c in OA])
    key = np.array([''.join(map(str, r)) for r in Y])
    po = pd.Series(key).value_counts(normalize=True).reindex(cells).fillna(0).values
    print('== %s prev %s | OAI cell min frac %.4f (cell %s)' % (nm, '/'.join('%.2f' % v for v in Y.mean(0)), po.min(), cells[int(np.argmin(po))]))
    wc = pg / np.maximum(po, 1e-9); w1 = wc[[cells.index(k) for k in key]]
    pairs = Y[:, 0] * Y[:, 1] + Y[:, 0] * Y[:, 2] + Y[:, 1] * Y[:, 2]; tgt = Y.mean(0)
    WT = {}
    for th in (1.0, 1.5):
        beta = np.zeros(3)
        for _ in range(300):
            w = np.exp(th * pairs + Y @ beta); cur = (w[:, None] * Y).sum(0) / w.sum()
            beta += np.log(tgt / (1 - tgt)) - np.log(cur / (1 - cur))
        WT[th] = w
    for k, Pk in PB.items():
        F = Pk.loc[ids, OA].values.astype(float)
        print('  %-5s R0   %s' % (k, evalw(F, Y, np.ones(len(Y)), B=100)))
        print('  %-5s R1g74 %s' % (k, evalw(F, Y, w1)))
        for th, w in WT.items():
            print('  %-5s tilt%.1f %s' % (k, th, evalw(F, Y, w)))
