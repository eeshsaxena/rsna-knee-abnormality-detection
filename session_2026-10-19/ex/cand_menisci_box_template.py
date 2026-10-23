# cand_menisci_box (10-20 ex, Claude). Read-only; prints AGGREGATES ONLY (n, prevalence, phi, rho, AUC deltas, P, ESS, q, a*).
# Never writes a file, never prints an id, a row or a per-knee value. Piped via ssh stdin into python -I.
# "G74P = {...}" (gold74 pairwise 2x2 counts [c00, c01, c10, c11], first-named label first; local aggregates) is filled in
# locally by cand_menisci_make.py.
#
# CANDIDATE (pre-registered in cand_menisci.md before this ran): menisci decorrelation
#   MM' = pct(f_MM) - a * pct(f_LM),  LM' = pct(f_LM) - a * pct(f_MM),  a in {.1, .2}; control a = -.1.
#   macro delta = (dAUC_MM + dAUC_LM) / 12.  a in {.05, .15} printed as DESCRIPTIVE only (not used for the verdict).
# Labels: B1 (primary) tear = T123 == 1 (MOAKS morph 2-8 or posterior-root tear); B2 = T123 >= .2 (adds intrasubstance signal);
#   B3 = tear without maceration (morph 2-5 or root tear = 1; knees whose only lesion is morph 6-8 dropped; code map = GUESS).
# Coupling-matched reweighting (skeptic method; knees resampled by label cell, predictions given labels kept as observed):
#   R1    : (MM, LM) 4-cell joint = gold74 (+.5 smoothing); marginals AND coupling move to gold74
#   R2    : OAI marginals + gold74 MM-LM log OR (coupling only)
#   R2hi / R2lo : gold74 log OR +/- 1 sd
#   R2m   : 5-label IPF (MOA, LOA, PF = P_FT proxy; MM, LM = B1): every pairwise 2x2 at OAI marginals + gold74 log OR
#   R2m_hi: R2m with every log OR + 1 sd
import os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, norm
from scipy.optimize import brentq
H = os.path.expanduser('~/oai_build')
OA = ['Medial OA', 'Lateral OA', 'PF OA']; MEN = ['Medial Meniscus', 'Lateral Meniscus']; LAB = OA + MEN
COMP = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'],
        'PF OA': ['mp', 'lp', 'amf', 'alf']}
MODELS = {'D5mean_s1043': H + '/teacher/teacher_D5_mean_v00.csv', 's1044_mix50': H + '/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv',
          's1045_288': H + '/teacher2/f0_s1045_ttr288frac75__raw__A__gpu.csv'}
pct = lambda v: rankdata(v) / len(v)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
BOOT = int(os.environ.get('BOOT', '2000')); RB = int(os.environ.get('RBOOT', '400'))
APRE = (-.1, .1, .2); ADES = (.05, .15); ALL_A = APRE + ADES


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def phi(a, b):
    return float(np.corrcoef(a.astype(float), b.astype(float))[0, 1])


T = pd.read_csv(H + '/oai_labels_T123.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
S = pd.read_csv(H + '/oai_label_sources.csv', dtype={'kid': str}).set_index('kid').reindex(T.index)
Y = pd.DataFrame(index=T.index)
for c in OA:
    cc = S[['m_cm_' + r for r in COMP[c]]].astype(float).values
    area = np.floor(cc + 1e-6); ft = np.round((cc - area) * 10); fin = np.isfinite(cc)
    y = ((ft >= 1) & fin).any(1).astype(float); y[~fin.any(1)] = np.nan; Y[c] = y
B2 = pd.DataFrame(index=T.index); B3 = pd.DataFrame(index=T.index)
for c, s in (('Medial Meniscus', 'mm'), ('Lateral Meniscus', 'lm')):
    Y[c] = np.where(T[c].notna(), (T[c] >= 1 - 1e-6).astype(float), np.nan)
    B2[c] = np.where(T[c].notna(), (T[c] >= .2 - 1e-6).astype(float), np.nan)
    mo = S[['m_%s%s_morph' % (s, r) for r in ('ah', 'body', 'ph')]].astype(float).values
    pr = S['m_%spr_tear' % s].astype(float).values
    tear = ((mo >= 2) & (mo <= 5)).any(1) | (pr == 1)
    mac_only = ((mo >= 6) & (mo <= 8)).any(1) & ~tear
    b3 = np.where(tear, 1., 0.); b3[mac_only] = np.nan; b3[T[c].isna().values] = np.nan
    B3[c] = b3
P = {}
for k, f in MODELS.items():
    P[k] = pd.read_csv(f, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')[LAB]
ids = Y.dropna().index
for k in P: ids = ids.intersection(P[k].dropna().index)
Yv = Y.loc[ids, LAB].values.astype(int); n = len(ids)
YM = Yv[:, 3:5]
print('n %d (knees with OA P_FT + both menisci + all 3 models) | B1 prevalence MM %.3f LM %.3f | B1 phi MM-LM %+.3f' % (
    n, YM[:, 0].mean(), YM[:, 1].mean(), phi(YM[:, 0], YM[:, 1])))
b2 = B2.loc[ids].values; b3d = B3.loc[ids].dropna()
print('B2 prevalence MM %.3f LM %.3f phi %+.3f | B3 n %d prevalence MM %.3f LM %.3f phi %+.3f' % (
    b2[:, 0].mean(), b2[:, 1].mean(), phi(b2[:, 0], b2[:, 1]), len(b3d), b3d.values[:, 0].mean(), b3d.values[:, 1].mean(),
    phi(b3d.values[:, 0], b3d.values[:, 1])))
sys.stdout.flush()


def men_tr(F2, a):
    U = pct(F2[:, 0]); V = pct(F2[:, 1])
    return np.stack([U - a * V, V - a * U], 1)


def deltas(F2, Y2, outs, ii=None):
    if ii is None: ii = np.arange(len(Y2))
    y = Y2[ii]; b = np.array([auc(F2[ii, k], y[:, k]) for k in range(2)])
    return {a: np.array([auc(O[ii, k], y[:, k]) for k in range(2)]) - b for a, O in outs.items()}


def or_cells(p1, p2, lor):
    lo = max(0., p1 + p2 - 1) + 1e-9; hi = min(p1, p2) - 1e-9
    g = lambda p11: np.log(p11) + np.log(1 - p1 - p2 + p11) - np.log(p1 - p11) - np.log(p2 - p11) - lor
    p11 = brentq(g, lo, hi)
    return np.array([[1 - p1 - p2 + p11, p2 - p11], [p1 - p11, p11]])


def g74_counts(a, b):
    k = a + ' | ' + b
    if k in G74P: return np.array(G74P[k], float)
    return np.array(G74P[b + ' | ' + a], float)[[0, 2, 1, 3]]


def two_cell_w(target4):
    key = YM[:, 0] * 2 + YM[:, 1]; po = np.bincount(key, minlength=4) / n
    return (target4 / np.maximum(po, 1e-12))[key]


def ipf_weights(shift):
    key = (Yv * (2 ** np.arange(5))).sum(1)
    p0 = np.bincount(key, minlength=32) / n; q = p0.copy(); cells = np.array([[(c >> i) & 1 for i in range(5)] for c in range(32)])
    marg = Yv.mean(0); tg = {}
    for i in range(5):
        for j in range(i + 1, 5):
            c = g74_counts(LAB[i], LAB[j]) + .5
            lor = np.log(c[3] * c[0] / (c[1] * c[2])) + shift * np.sqrt((1 / c).sum())
            tg[(i, j)] = or_cells(marg[i], marg[j], lor)
    for _ in range(300):
        for (i, j), t in tg.items():
            for a in (0, 1):
                for b in (0, 1):
                    m = (cells[:, i] == a) & (cells[:, j] == b); s = q[m].sum()
                    if s > 0: q[m] *= t[a, b] / s
        q /= q.sum()
    return np.where(p0 > 0, q / np.maximum(p0, 1e-12), 0)[key]


c = g74_counts('Medial Meniscus', 'Lateral Meniscus') + .5
lor = np.log(c[3] * c[0] / (c[1] * c[2])); lsd = np.sqrt((1 / c).sum())
p1, p2 = YM.mean(0)
print('gold74 MM-LM counts %s | log OR %+.3f sd %.3f | OAI B1 log OR %+.3f' % (
    (c - .5).astype(int).tolist(), lor, lsd,
    np.log((((YM[:, 0] == 1) & (YM[:, 1] == 1)).sum() + .5) * (((YM[:, 0] == 0) & (YM[:, 1] == 0)).sum() + .5) /
           ((((YM[:, 0] == 0) & (YM[:, 1] == 1)).sum() + .5) * (((YM[:, 0] == 1) & (YM[:, 1] == 0)).sum() + .5)))))
SCH = {'R1': two_cell_w(c / c.sum()), 'R2': two_cell_w(or_cells(p1, p2, lor).ravel()),
       'R2lo': two_cell_w(or_cells(p1, p2, lor - lsd).ravel()), 'R2hi': two_cell_w(or_cells(p1, p2, lor + lsd).ravel()),
       'R2m': ipf_weights(0.), 'R2m_hi': ipf_weights(1.)}
for nm, w in SCH.items():
    p = w / w.sum(); ii = np.random.default_rng(0).choice(n, 40000, p=p); yy = Yv[ii]
    print('scheme %-6s ESS %5.0f | weighted prevalence MM %.3f LM %.3f | phi MM-LM %+.3f MM-LOA %+.3f MM-MOA %+.3f LM-LOA %+.3f' % (
        nm, 1 / (p ** 2).sum(), yy[:, 3].mean(), yy[:, 4].mean(), phi(yy[:, 3], yy[:, 4]), phi(yy[:, 3], yy[:, 1]),
        phi(yy[:, 3], yy[:, 0]), phi(yy[:, 4], yy[:, 1])))
sys.stdout.flush()

for mk, Pm in P.items():
    F2 = Pm.loc[ids, MEN].values.astype(float)
    rs = spearmanr(F2[:, 0], F2[:, 1]).correlation
    Z = np.stack([ns(F2[:, 0]), ns(F2[:, 1])], 1); rz = np.corrcoef(Z.T)[0, 1]
    th = []
    for t, o in ((0, 1), (1, 0)):
        ct = np.corrcoef(Z[:, t], YM[:, t])[0, 1]; co = np.corrcoef(Z[:, o], YM[:, t])[0, 1]; q = co / ct
        th.append('%s q %+.3f (rho-q %+.3f, a* %+.3f)' % (MEN[t][:6], q, rz - q, (rz - q) / (1 - rz * q)))
    print('[%s] model Spearman MM-LM %+.3f, rho_ns %+.3f | %s' % (mk, rs, rz, ' | '.join(th)))
    outs = {a: men_tr(F2, a) for a in ALL_A}
    # raw, B1 primary, full paired bootstrap
    d0 = deltas(F2, YM, outs); rng = np.random.default_rng(11); BD = {a: [] for a in ALL_A}
    for _ in range(BOOT):
        ii = rng.integers(0, n, n); dd = deltas(F2, YM, outs, ii)
        for a in ALL_A: BD[a].append(dd[a].sum())
    base = [auc(F2[:, k], YM[:, k]) for k in range(2)]
    print('[%s | raw B1] base AUC MM %.4f LM %.4f' % (mk, *base))
    for a in ALL_A:
        v = np.array(BD[a])
        print('[%s | raw B1] a %+.2f%s macro %+.5f | per label MM %+.4f LM %+.4f | boot sd(macro) %.5f P>0 %.3f' % (
            mk, a, '' if a in APRE else ' (descr)', d0[a].sum() / 12, d0[a][0], d0[a][1], v.std() / 12, (v > 0).mean()))
    # proxies B2, B3 (raw, smaller bootstrap)
    for pn, Ydf in (('B2', B2.loc[ids]), ('B3', B3.loc[ids].dropna())):
        jj = np.array([ids.get_loc(x) for x in Ydf.index]); Fp = F2[jj]; Yp = Ydf.values.astype(int)
        outp = {a: men_tr(Fp, a) for a in APRE}; dp = deltas(Fp, Yp, outp)
        rng = np.random.default_rng(12); BP = {a: [] for a in APRE}
        for _ in range(500):
            ii = rng.integers(0, len(Yp), len(Yp)); dd = deltas(Fp, Yp, outp, ii)
            for a in APRE: BP[a].append(dd[a].sum())
        print('[%s | raw %s n %d] ' % (mk, pn, len(Yp)) + ' | '.join('a %+.1f macro %+.5f (MM %+.4f LM %+.4f, P>0 %.2f)' % (
            a, dp[a].sum() / 12, dp[a][0], dp[a][1], np.mean(np.array(BP[a]) > 0)) for a in APRE))
    # coupling-matched reweighting (B1)
    for sn, w in SCH.items():
        p = w / w.sum(); rng = np.random.default_rng(13); BW = {a: [] for a in APRE}; PL = {a: [] for a in APRE}
        for _ in range(RB):
            ii = rng.choice(n, n, p=p); dd = deltas(F2, YM, outs, ii)
            for a in APRE: BW[a].append(dd[a].sum()); PL[a].append(dd[a])
        print('[%s | %s] ' % (mk, sn) + ' | '.join('a %+.1f macro %+.5f (MM %+.4f LM %+.4f, P>0 %.2f)' % (
            a, np.mean(BW[a]) / 12, np.mean(PL[a], 0)[0], np.mean(PL[a], 0)[1], np.mean(np.array(BW[a]) > 0)) for a in APRE))
    sys.stdout.flush()
print('DONE')
