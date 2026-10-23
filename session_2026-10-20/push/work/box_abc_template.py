# box_abc.py (10-20 push, Claude): public-size (n 390) OAI resamples under a coupling continuum; ABC on the public LB reads.
# Read-only analysis. Prints AGGREGATES ONLY (means, sds, correlations, regression coefficients, posterior summaries of
# resample-level AUC deltas). Never writes a file, never prints an id, a row or a per-knee value.
# Piped: ssh training@192.168.10.30 '~/miniconda3/envs/rsna/bin/python -I - P_FT B h' < box_abc_run.py
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
from scipy.optimize import brentq
H = os.path.expanduser('~/oai_build')
OA = ['Medial OA', 'Lateral OA', 'PF OA']
COMP = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'],
        'PF OA': ['mp', 'lp', 'amf', 'alf']}
MODELS = {'D5': H + '/teacher/teacher_D5_mean_v00.csv', 's1044': H + '/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv',
          's1045': H + '/teacher2/f0_s1045_ttr288frac75__raw__A__gpu.csv'}
G74P = {(0, 1): [45, 5, 12, 9], (0, 2): [36, 14, 6, 16], (1, 2): [37, 20, 5, 9]}
GAPW = {(0, 1): .26, (0, 2): .21, (1, 2): .42}; mg = np.mean(list(GAPW.values())); PW = {k: v / mg for k, v in GAPW.items()}
PROX = sys.argv[1] if len(sys.argv) > 1 else 'P_FT'
NPUB = 390
BREG = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
HBW = float(sys.argv[3]) if len(sys.argv) > 3 else 4e-5
SHIFTS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
LBD = np.array(__LBDRAWS__)   # rows (G10, G20, gap960) in macro units, from the public LB posterior (lb_post.py)
VN = ['S05', 'S10', 'S15', 'S20', 'S25', 'S30', 'S40', 'S50', 'PW10', 'PW20', 'PW30', 'PW40', 'S35', 'PW15', 'PW25', 'PW35', 'PW50']


def oth(t):
    return [m for m in range(3) if m != t]


def pct(v):
    return rankdata(v) / len(v)


def bcols(t):
    o1, o2 = oth(t); out = [(0., 0.)]
    for v in VN:
        a = int(v[-2:]) / 100.
        if v.startswith('S'):
            out.append((a / 2, a / 2))
        else:
            out.append((a / 2 * PW[tuple(sorted((t, o1)))], a / 2 * PW[tuple(sorted((t, o2)))]))
    return np.array(out)


BC = [bcols(t) for t in range(3)]


def auc_cols(S, y):
    R = rankdata(S, axis=0); p = y.sum(); q = len(y) - p
    return (R[y == 1].sum(0) - p * (p + 1) / 2) / (p * q)


def proxy(name):
    s = pd.read_csv(H + '/oai_label_sources.csv', dtype={'kid': str}).set_index('kid')
    Y = pd.DataFrame(index=s.index, columns=OA, dtype=float)
    for lab, regs in COMP.items():
        c = s[['m_cm_' + r for r in regs]].astype(float).values
        area = np.floor(c + 1e-6); ft = np.round((c - area) * 10); fin = np.isfinite(c)
        pos = ((ft >= 1) & fin) if name == 'P_FT' else ((area >= 2) & fin)
        y = pos.any(1).astype(float); y[~fin.any(1)] = np.nan; Y[lab] = y
    return Y.dropna()


def or_cells(p1, p2, lor):
    lo = max(0., p1 + p2 - 1) + 1e-9; hi = min(p1, p2) - 1e-9
    g = lambda p11: np.log(p11) + np.log(1 - p1 - p2 + p11) - np.log(p1 - p11) - np.log(p2 - p11) - lor
    p11 = brentq(g, lo, hi)
    return np.array([[1 - p1 - p2 + p11, p2 - p11], [p1 - p11, p11]])


def weights(Y, sh):
    n = len(Y); k3 = Y[:, 0] * 4 + Y[:, 1] * 2 + Y[:, 2]; p0 = np.bincount(k3, minlength=8) / n
    if sh is None:
        return np.ones(n)
    cells = np.array([[(c >> (2 - i)) & 1 for i in range(3)] for c in range(8)]); marg = Y.mean(0); q = p0.copy(); tg = {}
    for (i, j), c in G74P.items():
        c = np.array(c, float) + .5; lor = np.log(c[3] * c[0] / (c[1] * c[2])) + sh * np.sqrt((1 / c).sum())
        tg[(i, j)] = or_cells(marg[i], marg[j], lor)
    for _ in range(500):
        for (i, j), t in tg.items():
            for a in (0, 1):
                for b in (0, 1):
                    m = (cells[:, i] == a) & (cells[:, j] == b); s = q[m].sum()
                    if s > 0:
                        q[m] *= t[a, b] / s
        q /= q.sum()
    return np.where(p0 > 0, q / np.maximum(p0, 1e-12), 0)[k3]


def one(U, Y, ii):
    y = Y[ii]; Ui = U[ii]; d = np.zeros(len(VN))
    for t in range(3):
        o1, o2 = oth(t)
        O = Ui[:, [t]] - Ui[:, [o1]] * BC[t][None, :, 0] - Ui[:, [o2]] * BC[t][None, :, 1]
        a = auc_cols(O, y[:, t]); d += a[1:] - a[0]
    Z = norm.ppf((rankdata(Ui, axis=0) - .5) / len(ii)); rz = np.corrcoef(Z.T); q = []
    for k in range(3):
        ck = np.corrcoef(Z[:, k], y[:, k])[0, 1]
        q.append(np.mean([np.corrcoef(Z[:, j], y[:, k])[0, 1] for j in oth(k)]) / ck)
    ph = np.corrcoef(y.T)
    return d / 12, rz[np.triu_indices(3, 1)].mean(), float(np.mean(q)), ph[np.triu_indices(3, 1)].mean()


def wquant(x, w, qs=(.1, .5, .9)):
    o = np.argsort(x); cw = np.cumsum(w[o]) / w.sum()
    return [float(x[o][min(len(x) - 1, np.searchsorted(cw, q_))]) for q_ in qs], float((x * w).sum() / w.sum())


def main():
    Yd = proxy(PROX); P = {}
    for k, f in MODELS.items():
        P[k] = pd.read_csv(f, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')[OA]
    ids = Yd.index
    for k in P:
        ids = ids.intersection(P[k].dropna().index)
    Y = Yd.loc[ids].values.astype(int); n = len(Y)
    UU = {k: np.stack([pct(P[k].loc[ids].values[:, c].astype(float)) for c in range(3)], 1) for k in P}
    print('ABC | proxy %s | n %d | NPUB %d | B/regime %d | h %.1e | LB draws %d' % (PROX, n, NPUB, BREG, HBW, len(LBD)), flush=True)
    regs = [('raw', None)] + [('R3p%+.1f' % s, s) for s in SHIFTS]
    SIM = []
    rng = np.random.default_rng(5)
    i10, i20, i30 = VN.index('S10'), VN.index('S20'), VN.index('S30')
    for m in MODELS:
        for rn, sh in regs:
            w = weights(Y, sh); p = w / w.sum(); D = []; R_ = []
            for b in range(BREG):
                ii = rng.choice(n, NPUB, p=p)
                if (Y[ii].sum(0) < 5).any() or ((NPUB - Y[ii].sum(0)) < 5).any():
                    continue
                d, rho, q, phi = one(UU[m], Y, ii); D.append(d); R_.append((rho, q, phi))
            D = np.array(D); R_ = np.array(R_); SIM.append((m, rn, D, R_))
            c = np.corrcoef(D[:, [i10, i20, i30]].T)
            print('[%s | %s] B %d | mean S10 %+.5f S20 %+.5f S30 %+.5f S40 %+.5f PW20-S20 %+.5f PW30-S30 %+.5f | sd S10 %.5f S20 %.5f S30 %.5f sd(S30-S20) %.5f | corr 10/20 %.2f 10/30 %.2f 20/30 %.2f | rho %.3f q %.3f gap %.3f phi %.3f' % (
                m, rn, len(D), D[:, i10].mean(), D[:, i20].mean(), D[:, i30].mean(), D[:, VN.index('S40')].mean(),
                (D[:, VN.index('PW20')] - D[:, i20]).mean(), (D[:, VN.index('PW30')] - D[:, i30]).mean(),
                D[:, i10].std(), D[:, i20].std(), D[:, i30].std(), (D[:, i30] - D[:, i20]).std(), c[0, 1], c[0, 2], c[1, 2],
                R_[:, 0].mean(), R_[:, 1].mean(), (R_[:, 0] - R_[:, 1]).mean(), R_[:, 2].mean()), flush=True)
    # aggregate export: per (model, regime) mean of [VN..., rho, q, phi] and the pooled within-regime covariance per model
    EXP = {'VN': VN + ['rho', 'q', 'phi'], 'mean': {}, 'cov': {}, 'n': {}}
    for m in MODELS:
        devs = []
        for sm in SIM:
            if sm[0] != m:
                continue
            X = np.concatenate([sm[2], sm[3]], 1); mu = X.mean(0)
            EXP['mean'][m + '|' + sm[1]] = [float(v) for v in mu]; EXP['n'][m + '|' + sm[1]] = int(len(X)); devs.append(X - mu)
        dv = np.concatenate(devs); EXP['cov'][m] = np.cov(dv.T).tolist()
    print('JSON_BEGIN'); print(json.dumps(EXP)); print('JSON_END', flush=True)
    for m in MODELS:
        D = np.concatenate([s[2] for s in SIM if s[0] == m]); X = D[:, [i10, i20]]
        for v in ('S30', 'S40', 'PW20', 'PW30'):
            yv = D[:, VN.index(v)]; b = np.linalg.lstsq(X, yv, rcond=None)[0]; res = yv - X @ b
            print('REG %s %s = %.3f S10 + %.3f S20 | resid sd %.5f | R2 %.3f' % (m, v, b[0], b[1], res.std(), 1 - res.var() / yv.var()))
    SG = np.exp(np.linspace(np.log(.5), np.log(2.), 15)); J = LBD
    Dall = np.concatenate([s[2] for s in SIM]); Rall = np.concatenate([s[3] for s in SIM])
    lab = np.concatenate([[k] * len(s[2]) for k, s in enumerate(SIM)]); cnt = np.bincount(lab); pw_ = 1. / cnt[lab]
    g10 = Dall[:, i10]; g20 = Dall[:, i20]
    W = np.zeros((len(Dall), len(SG), len(J)), dtype=np.float32)
    for si, s in enumerate(SG):
        a = (s * g10[:, None] - J[None, :, 0]) / HBW; b = (s * g20[:, None] - J[None, :, 1]) / HBW
        W[:, si, :] = (np.exp(-.5 * (a * a + b * b)) * pw_[:, None]).astype(np.float32)
    tot = float(W.sum(dtype=np.float64)); W64sq = float((W.astype(np.float64) ** 2).sum())
    print('ABC total weight %.3e | ESS %.0f' % (tot, tot ** 2 / W64sq))
    wi = W.sum((1, 2), dtype=np.float64); ws = W.sum((0, 2), dtype=np.float64)
    print('posterior s (amplitude vs OAI model):', ' '.join('%.2f:%.3f' % (s, x / tot) for s, x in zip(SG, ws)))
    for k, sm in enumerate(SIM):
        print('posterior mass %s | %s: %.3f' % (sm[0], sm[1], wi[lab == k].sum() / tot))
    for nm, col in (('rho', 0), ('q', 1), ('label phi', 2)):
        qs, mu = wquant(Rall[:, col], wi); print('implied sample %s: mean %.3f, p10/50/90 %s' % (nm, mu, np.round(qs, 3).tolist()))
    qs, mu = wquant(Rall[:, 0] - Rall[:, 1], wi); print('implied sample gap rho-q: mean %.3f, p10/50/90 %s' % (mu, np.round(qs, 3).tolist()))
    for v in ('S05', 'S10', 'S15', 'S25', 'S30', 'S40', 'S50', 'PW10', 'PW20', 'PW30', 'PW40'):
        inc = Dall[:, VN.index(v)] - g20
        X = inc[:, None] * SG[None, :]
        Ws = W.sum(2, dtype=np.float64)
        m = float((Ws * X).sum() / tot); sd = float(np.sqrt((Ws * (X - m) ** 2).sum() / tot))
        p960 = float((W * (X[:, :, None] > J[None, None, :, 2])).sum(dtype=np.float64) / tot)
        ppos = float((Ws * (X > 0)).sum() / tot)
        qq, _ = wquant(X.ravel(), Ws.ravel())
        print('PRED %-5s public increment over eoa20: mean %+.5f sd %.5f p10/50/90 %s | P(>0) %.2f | P(shows .960) %.2f' % (
            v, m, sd, np.round(qq, 5).tolist(), ppos, p960), flush=True)
    grid = ['S05', 'S10', 'S15', 'S20', 'S25', 'S30', 'S40', 'S50']; G = Dall[:, [VN.index(v) for v in grid]]
    best = np.argmax(G, 1); bm = np.bincount(best, weights=wi, minlength=len(grid)) / tot
    print('public-optimal symmetric a (posterior):', ' '.join('%s:%.2f' % (g, x) for g, x in zip(grid, bm)))
    print('regime posterior -> POPULATION increments (regime means x s; what a large set would show, no public-sample noise):')
    for v in ('S10', 'S30', 'S40', 'PW20', 'PW30'):
        acc2 = 0.; tot2 = 0.
        for k, sm in enumerate(SIM):
            mu = (sm[2][:, VN.index(v)] - sm[2][:, i20]).mean(); wk = W[lab == k].sum((0, 2), dtype=np.float64)
            acc2 += (wk * SG * mu).sum(); tot2 += wk.sum()
        print('   POP %s - S20: %+.5f' % (v, acc2 / tot2))
    acc2 = np.zeros(2); tot2 = 0.
    for k, sm in enumerate(SIM):
        mu = np.array([sm[2][:, i10].mean(), sm[2][:, i20].mean()]); wk = W[lab == k].sum((0, 2), dtype=np.float64)
        acc2 += (wk[None, :] * SG[None, :] * mu[:, None]).sum(1); tot2 += wk.sum()
    print('   POP S10 / S20 (x s): %+.5f / %+.5f' % tuple(acc2 / tot2))


main()
