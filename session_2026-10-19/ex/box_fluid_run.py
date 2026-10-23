# box_fluid (10-20 ex, Claude). Read-only; prints AGGREGATES ONLY (n, prevalence, phi, rho, AUC deltas, P, ESS). Never writes
# a file, never prints an id, a row or a per-knee value. Piped via ssh stdin into python -I. "G74F = {...}" (gold74 fluid
# 3-label cell counts + pairwise 2x2 counts, local aggregates) is filled in locally by make_fluid.py.
#
# CANDIDATE fluid: group (Effusion, Synovitis, Baker's), each out_t = pct(f_t) - a * mean(pct(f_o)) over the other two (FL3),
# and the pairwise Effusion <-> Synovitis only version (ES2: E <- S and S <- E at a, Baker's untouched).
# a in {-.1 (control), .1, .2}. pct over the full OAI prediction population (2,581 knees, deployment-like), AUC per label on
# its labelled knees, paired bootstrap over knees with >= 1 fluid label.
# Binarisations: B1 = map primary (Eff MOAKS grade >= 2, Hoffa Syn >= 1), B2 (Eff >= 1, Syn >= 2), B3 = prevalence-matched to
# RSNA gold74 (Eff >= 1 [OAI .57 vs gold74 .65], Syn >= 1 [.58 vs .49]); Baker's = 1 in all.
# Criterion 2 (skeptic method: importance-resample knees by label cell, predictions given labels kept as observed):
#   FL3 on the complete-case set (all 3 fluid labels): raw_cc, R1 (gold74 8-cell joint, +.5 smoothed), R2 (IPF: OAI
#   marginals + gold74 pairwise log OR = coupling only), R2hi / R2lo (gold74 log OR +/- 1 sd).
#   ES2 (and the E, S components of FL3) on the E+S set: raw, R1 (gold74 4-cell), R2, R2hi, R2lo.
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
from scipy.optimize import brentq
H = os.path.expanduser('~/oai_build')
G74F = {"cells": {"000": 16, "001": 3, "010": 4, "011": 1, "100": 13, "101": 2, "110": 16, "111": 12}, "pairs": {"Effusion | Synovitis": [20, 5, 17, 31], "Effusion | Baker's": [20, 4, 29, 14], "Synovitis | Baker's": [29, 5, 20, 13]}, "n_complete": 67}
FL = ['Effusion', 'Synovitis', "Baker's"]
MODELS = {'D5mean_s1043': H + '/teacher/teacher_D5_mean_v00.csv', 's1044_mix50': H + '/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv',
          's1045_288': H + '/teacher2/f0_s1045_ttr288frac75__raw__A__gpu.csv'}
pct = lambda v: rankdata(v) / len(v)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 400
RB = int(sys.argv[2]) if len(sys.argv) > 2 else 300
AS = (-.1, .1, .2)


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def phi(a, b):
    return float(np.corrcoef(a.astype(float), b.astype(float))[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan')


T = pd.read_csv(H + '/oai_labels_T123.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
P = {}
for k, f in MODELS.items():
    P[k] = pd.read_csv(f, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')[FL]
common = T.index
for k in P: common = common.intersection(P[k].dropna().index)
thr = lambda c, x: np.where(T.loc[common, c].notna(), (T.loc[common, c] >= x - 1e-6).astype(float), np.nan)
BIN = {'B1': {'Effusion': thr('Effusion', .8), 'Synovitis': thr('Synovitis', 1 / 3), "Baker's": thr("Baker's", 1.)},
       'B2': {'Effusion': thr('Effusion', .3), 'Synovitis': thr('Synovitis', 2 / 3), "Baker's": thr("Baker's", 1.)},
       'B3': {'Effusion': thr('Effusion', .3), 'Synovitis': thr('Synovitis', 1 / 3), "Baker's": thr("Baker's", 1.)}}
N = len(common)
print('OAI knees with all 3 model predictions and a T123 row: %d' % N)
for bn, Yb in BIN.items():
    Ym = np.stack([Yb[c] for c in FL], 1); cc = np.isfinite(Ym).all(1)
    print('%s coverage/prev %s | complete-case n %d | phi E-S %+.3f E-B %+.3f S-B %+.3f (pairwise-complete)' % (
        bn, {c: (int(np.isfinite(Yb[c]).sum()), round(float(np.nanmean(Yb[c])), 3)) for c in FL}, cc.sum(),
        *[phi(Ym[np.isfinite(Ym[:, i]) & np.isfinite(Ym[:, j]), i], Ym[np.isfinite(Ym[:, i]) & np.isfinite(Ym[:, j]), j])
          for i, j in ((0, 1), (0, 2), (1, 2))]))
for k in P:
    F = P[k].loc[common, FL].values.astype(float); Z = np.stack([ns(F[:, i]) for i in range(3)], 1); C = np.corrcoef(Z.T)
    print('model %s rho(ns) on all %d: E-S %+.3f E-B %+.3f S-B %+.3f' % (k, N, C[0, 1], C[0, 2], C[1, 2]))
sys.stdout.flush()


def transforms(F):
    U = np.stack([pct(F[:, i]) for i in range(3)], 1); out = {}
    for a in AS:
        O = U.copy()
        for i in range(3):
            o = [m for m in range(3) if m != i]; O[:, i] = U[:, i] - a * (U[:, o[0]] + U[:, o[1]]) / 2
        out['FL3 a%+.1f' % a] = O
    for a in AS:
        O = U.copy(); O[:, 0] = U[:, 0] - a * U[:, 1]; O[:, 1] = U[:, 1] - a * U[:, 0]
        out['ES2 a%+.1f' % a] = O
    return out


def or_cells(p1, p2, lor):
    lo = max(0., p1 + p2 - 1) + 1e-9; hi = min(p1, p2) - 1e-9
    g = lambda p11: np.log(p11) + np.log(1 - p1 - p2 + p11) - np.log(p1 - p11) - np.log(p2 - p11) - lor
    p11 = brentq(g, lo, hi)
    return np.array([[1 - p1 - p2 + p11, p2 - p11], [p1 - p11, p11]])   # [a][b]


PAIRS = [(0, 1), (0, 2), (1, 2)]
PK = {(0, 1): 'Effusion | Synovitis', (0, 2): "Effusion | Baker's", (1, 2): "Synovitis | Baker's"}


def g74_lor(i, j, shift):
    c = np.array(G74F['pairs'][PK[(i, j)]], float) + .5     # [c00, c01, c10, c11], first label first
    return np.log(c[3] * c[0] / (c[1] * c[2])) + shift * np.sqrt((1 / c).sum())


def w_fl3(Yc, scheme):
    """Weights on complete-case knees (Yc n x 3 int) for the 3-label fluid cell."""
    n = len(Yc); key = Yc[:, 0] * 4 + Yc[:, 1] * 2 + Yc[:, 2]; p0 = np.bincount(key, minlength=8) / n
    if scheme == 'raw_cc': return np.ones(n)
    if scheme == 'R1':
        pg = np.array([G74F['cells'][format(i, '03b')] + .5 for i in range(8)], float); pg /= pg.sum()
        return (pg / np.maximum(p0, 1e-12))[key]
    shift = {'R2': 0., 'R2hi': 1., 'R2lo': -1.}[scheme]
    cells = np.array([[(c >> (2 - i)) & 1 for i in range(3)] for c in range(8)]); marg = Yc.mean(0); q = p0.copy()
    tg = {(i, j): or_cells(marg[i], marg[j], g74_lor(i, j, shift)) for i, j in PAIRS}
    for _ in range(500):
        for (i, j), t in tg.items():
            for a in (0, 1):
                for b in (0, 1):
                    m = (cells[:, i] == a) & (cells[:, j] == b); s = q[m].sum()
                    if s > 0: q[m] *= t[a, b] / s
        q /= q.sum()
    return np.where(p0 > 0, q / np.maximum(p0, 1e-12), 0)[key]


def w_es(Ye, scheme):
    n = len(Ye); key = Ye[:, 0] * 2 + Ye[:, 1]; p0 = np.bincount(key, minlength=4) / n
    if scheme == 'raw': return np.ones(n)
    c = np.array(G74F['pairs'][PK[(0, 1)]], float) + .5
    if scheme == 'R1':
        t = c / c.sum()
    else:
        shift = {'R2': 0., 'R2hi': 1., 'R2lo': -1.}[scheme]
        t = or_cells(Ye[:, 0].mean(), Ye[:, 1].mean(), g74_lor(0, 1, shift)).ravel()
    return (t / np.maximum(p0, 1e-12))[key]


def wphi(Y, w, cols):
    p = w / w.sum(); ii = np.random.default_rng(0).choice(len(Y), 20000, p=p); C = np.corrcoef(Y[ii][:, cols].T)
    return C


RES = {}
for bn, Yb in BIN.items():
    Ym = np.stack([Yb[c] for c in FL], 1); has = np.isfinite(Ym); anyl = has.any(1)
    RES[bn] = {}
    for mk, Pm in P.items():
        F = Pm.loc[common, FL].values.astype(float); TR = transforms(F); R = {}
        # ---- criterion 1: raw OAI, all labelled knees, paired bootstrap over knees with >= 1 fluid label ----
        base = np.array([auc(F[has[:, i], i], Ym[has[:, i], i]) for i in range(3)])
        pt = {t: np.array([auc(O[has[:, i], i], Ym[has[:, i], i]) for i in range(3)]) - base for t, O in TR.items()}
        idx = np.where(anyl)[0]; rng = np.random.default_rng(11); BD = {t: [] for t in TR}
        for _ in range(BOOT):
            ii = idx[rng.integers(0, len(idx), len(idx))]; ok = True; b0 = []
            for i in range(3):
                h = has[ii, i]; y = Ym[ii[h], i]
                if y.sum() < 5 or (len(y) - y.sum()) < 5: ok = False
            if not ok: continue
            b0 = np.array([auc(F[ii[has[ii, i]], i], Ym[ii[has[ii, i]], i]) for i in range(3)])
            for t, O in TR.items():
                d = np.array([auc(O[ii[has[ii, i]], i], Ym[ii[has[ii, i]], i]) for i in range(3)]) - b0
                BD[t].append(d)
        R['raw_all'] = {t: {'per': pt[t].tolist(), 'P_FL3sum': float(np.mean(np.array(BD[t]).sum(1) > 0)),
                            'P_ESsum': float(np.mean(np.array(BD[t])[:, :2].sum(1) > 0)),
                            'P_per': np.mean(np.array(BD[t]) > 0, 0).tolist(), 'nb': len(BD[t])} for t in TR}
        # theory: q_t = mean_o corr(ns f_o, y_t) / corr(ns f_t, y_t), vs rho to the others
        Z = np.stack([ns(F[:, i]) for i in range(3)], 1); qd = {}
        for i in range(3):
            h = has[:, i]; ct = np.corrcoef(Z[h, i], Ym[h, i])[0, 1]
            qd[FL[i]] = {FL[o]: float(np.corrcoef(Z[h, o], Ym[h, i])[0, 1] / ct) for o in range(3) if o != i}
            qd[FL[i]]['rho'] = {FL[o]: float(np.corrcoef(Z[h, i], Z[h, o])[0, 1]) for o in range(3) if o != i}
        R['q'] = qd
        # ---- criterion 2a: FL3 on complete-case knees, label-cell reweighting to gold74 fluid coupling ----
        cc = has.all(1); Yc = Ym[cc].astype(int); Fc = {t: O[cc] for t, O in TR.items()}; F0 = F[cc]; ncc = cc.sum()
        R['cc'] = {}
        for sn in ('raw_cc', 'R1', 'R2', 'R2hi', 'R2lo'):
            w = w_fl3(Yc, sn); p = w / w.sum(); ess = 1 / (p ** 2).sum(); C = wphi(Yc, w, [0, 1, 2])
            rng = np.random.default_rng(7); D = {t: [] for t in TR}
            for _ in range(RB):
                ii = rng.choice(ncc, ncc, p=p); y = Yc[ii]
                if (y.sum(0) < 5).any() or (y.sum(0) > ncc - 5).any(): continue
                b0 = np.array([auc(F0[ii, i], y[:, i]) for i in range(3)])
                for t in TR: D[t].append(np.array([auc(Fc[t][ii, i], y[:, i]) for i in range(3)]) - b0)
            R['cc'][sn] = {'ess': float(ess), 'phi': [float(C[0, 1]), float(C[0, 2]), float(C[1, 2])], 'prev': (p @ Yc).tolist(),
                           't': {t: {'per': np.mean(v, 0).tolist(), 'sum3': float(np.mean(np.sum(v, 1))),
                                     'P3': float(np.mean(np.sum(v, 1) > 0)), 'sum2': float(np.mean(np.array(v)[:, :2].sum(1))),
                                     'P2': float(np.mean(np.array(v)[:, :2].sum(1) > 0))} for t, v in D.items()}}
        # ---- criterion 2b: E+S set, E-S cell reweighting (ES2 and the E, S components of FL3) ----
        es = has[:, 0] & has[:, 1]; Ye = Ym[es][:, :2].astype(int); Fe = {t: O[es] for t, O in TR.items()}; F0e = F[es]; ne = es.sum()
        R['es'] = {}
        for sn in ('raw', 'R1', 'R2', 'R2hi', 'R2lo'):
            w = w_es(Ye, sn); p = w / w.sum(); ess = 1 / (p ** 2).sum(); C = wphi(Ye, w, [0, 1])
            rng = np.random.default_rng(8); D = {t: [] for t in TR}
            for _ in range(RB):
                ii = rng.choice(ne, ne, p=p); y = Ye[ii]
                if (y.sum(0) < 5).any() or (y.sum(0) > ne - 5).any(): continue
                b0 = np.array([auc(F0e[ii, i], y[:, i]) for i in range(2)])
                for t in TR: D[t].append(np.array([auc(Fe[t][ii, i], y[:, i]) for i in range(2)]) - b0)
            R['es'][sn] = {'ess': float(ess), 'phi': float(C[0, 1]), 'prev': (p @ Ye).tolist(),
                           't': {t: {'per': np.mean(v, 0).tolist(), 'sum2': float(np.mean(np.sum(v, 1))),
                                     'P2': float(np.mean(np.sum(v, 1) > 0))} for t, v in D.items()}}
        RES[bn][mk] = R
        # ---- print ----
        print('[%s | %s] n per label %s | base AUC %s' % (bn, mk, has.sum(0).tolist(), np.round(base, 4).tolist()))
        for t in TR:
            v = R['raw_all'][t]
            print('   raw_all %-10s macro FL3-sum %+.5f (P %.2f) ES-sum %+.5f (P %.2f) | per E/S/B %s (P %s)' % (
                t, sum(v['per']) / 12, v['P_FL3sum'], sum(v['per'][:2]) / 12, v['P_ESsum'], np.round(v['per'], 4).tolist(),
                np.round(v['P_per'], 2).tolist()))
        print('   q (cross-predictiveness / own) and rho(ns): %s' % json.dumps({k2: {k3: (round(v3, 3) if not isinstance(v3, dict) else
              {k4: round(v4, 3) for k4, v4 in v3.items()}) for k3, v3 in v2.items()} for k2, v2 in qd.items()}))
        for sn, v in R['cc'].items():
            print('   cc %-6s ESS %4.0f phi E-S %+.2f E-B %+.2f S-B %+.2f prev %s | %s' % (
                sn, v['ess'], *v['phi'], np.round(v['prev'], 3).tolist(), ' | '.join('%s %+.5f (P %.2f; per %s)' % (
                    t, x['sum3'] / 12, x['P3'], np.round(x['per'], 4).tolist()) for t, x in v['t'].items() if t.startswith('FL3'))))
        for sn, v in R['es'].items():
            print('   es %-6s ESS %4.0f phi E-S %+.2f prev %s | %s' % (
                sn, v['ess'], v['phi'], np.round(v['prev'], 3).tolist(), ' | '.join('%s ES %+.5f (P %.2f)' % (
                    t, x['sum2'] / 12, x['P2']) for t, x in v['t'].items())))
        sys.stdout.flush()
print('JSON_BEGIN'); print(json.dumps(RES)); print('JSON_END')
