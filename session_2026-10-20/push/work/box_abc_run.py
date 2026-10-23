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
LBD = np.array([[0.0005378, 0.0012751, 0.0001003], [0.0006373, 0.0010639, 0.0001612], [0.0006706, 0.001167, 4.89e-05], [0.0006381, 0.0010919, 4.03e-05], [0.0004301, 0.0011358, 0.0001642], [0.0005387, 0.0010354, 7.32e-05], [0.0007278, 0.0012433, 7.7e-05], [0.0008494, 0.0009876, 3.38e-05], [0.0005661, 0.0011786, 3.88e-05], [0.0004943, 0.0008565, 0.0002882], [0.0008089, 0.0011482, 8.86e-05], [0.0005409, 0.000922, 0.0002265], [0.000616, 0.0011942, 5.55e-05], [0.0006127, 0.0010169, 0.0002557], [0.0006851, 0.0011553, 9.81e-05], [0.000577, 0.0014324, 0.000296], [0.0007424, 0.0009397, 0.0001014], [0.0005737, 0.0012564, 5.14e-05], [0.0006306, 0.0011893, 0.0001141], [0.0006542, 0.0011544, 0.0001134], [0.0005432, 0.0010838, 0.0001449], [0.0002612, 0.0009431, 0.0003022], [0.0006233, 0.0011023, 8.62e-05], [0.000536, 0.0011495, 7.87e-05], [0.000767, 0.001224, 0.0001821], [0.0003079, 0.0008726, 0.0002703], [0.000552, 0.0010491, 3.24e-05], [0.0004768, 0.0009237, 0.0001234], [0.0006063, 0.0011286, 7.34e-05], [0.0005949, 0.0013495, 8.42e-05], [0.0007231, 0.0009274, 0.0001085], [0.000684, 0.0012547, 6.26e-05], [0.0005086, 0.0011971, 3.65e-05], [0.0005434, 0.0012365, 4.64e-05], [0.0003489, 0.0009339, 0.0002137], [0.0008063, 0.0012319, 7.92e-05], [0.0006261, 0.0012396, 6.57e-05], [0.0006, 0.0013328, 2.85e-05], [0.0006708, 0.0010276, 0.0002263], [0.0005434, 0.0010921, 0.0001496], [0.0008257, 0.0012189, 0.0001324], [0.0004919, 0.001186, 0.0001067], [0.0004745, 0.0012218, 0.000112], [0.0005977, 0.0010962, 0.0001336], [0.000463, 0.0009303, 0.0002759], [0.0007933, 0.0010831, 0.0001415], [0.0004521, 0.0010414, 0.0001139], [0.0003784, 0.001006, 9.03e-05], [0.0003917, 0.0010603, 0.0002582], [0.0006552, 0.0012058, 0.000237], [0.0005705, 0.0011276, 9.35e-05], [0.0003785, 0.0009962, 0.0001525], [0.0007107, 0.0012141, 5.94e-05], [0.0005773, 0.0011023, 0.0001841], [0.0007957, 0.0011339, 2.55e-05], [0.0007309, 0.0007411, 0.0003043], [0.0008262, 0.0010854, 9.13e-05], [0.0006158, 0.0011304, 8.88e-05], [0.0005485, 0.0013899, 0.0001107], [0.0006939, 0.0011891, 0.0001064], [0.0006866, 0.0012415, 8.33e-05], [0.0004019, 0.0008981, 0.0002027], [0.0006021, 0.0011759, 7.72e-05], [0.0007938, 0.0010319, 5.23e-05], [0.0006286, 0.0010329, 4.75e-05], [0.0013482, 0.0011422, 0.0002595], [0.0006038, 0.0011509, 0.0001186], [0.0002926, 0.0009936, 3.8e-05], [0.000644, 0.0012245, 0.0001163], [0.0004225, 0.0011423, 0.0002006], [0.0007213, 0.0010807, 0.0001453], [0.0007505, 0.001384, 5.71e-05], [0.0005914, 0.000983, 0.0001998], [0.0007145, 0.0012909, 0.0001473], [0.0005891, 0.0011705, 4.88e-05], [0.0006516, 0.0013278, 0.0001038], [0.0004934, 0.0010463, 0.0001159], [0.000469, 0.0011093, 5.86e-05], [0.0005055, 0.0011467, 3.35e-05], [0.0008362, 0.0011223, 0.0001324], [0.0003211, 0.0007937, 0.00028], [0.0006308, 0.0012143, 5.02e-05], [0.0005232, 0.0010341, 0.0001535], [0.0007249, 0.0009907, 0.0001175], [0.000691, 0.0009978, 6.74e-05], [0.0006325, 0.0011455, 6.69e-05], [0.0006942, 0.0009865, 0.0001337], [0.000757, 0.0011954, 5.4e-05], [0.0007156, 0.0011154, 0.0001403], [0.000525, 0.0011615, 8.58e-05], [0.0005739, 0.0009687, 0.0001134], [0.0006255, 0.0011343, 9.69e-05], [0.0005138, 0.001211, 0.0001402], [0.0009845, 0.0012828, 4.3e-05], [0.0006666, 0.0010734, 5.85e-05], [0.0007799, 0.0012392, 4.73e-05], [0.0009495, 0.0012642, 4.1e-05], [0.0005258, 0.0011323, 7.45e-05], [0.0007489, 0.001231, 1.5e-05], [0.0004048, 0.001105, 5.07e-05]])   # rows (G10, G20, gap960) in macro units, from the public LB posterior (lb_post.py)
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
