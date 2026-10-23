# box_oaigrid.py (10-20 ~19:00 IST val TRACK A, Claude). Box only, read-only; prints AGGREGATES ONLY (n, ESS, means, sds,
# probabilities, argmax summaries, likelihood sums). Never prints an id, a row or a per-knee value; writes nothing on the box.
# cov / lab_cov: covariance of the bootstrap draws (macro (NF*NA) / per-label (3*NF*NA) flattened), units 1e-8.
# Piped: ssh training@192.168.10.30 '~/miniconda3/envs/rsna/bin/python -I - P_FT BK BP' < box_oaigrid.py
# What it does:
#  c4D5 proxy = c4's real OA structure with D5 (OAI-free) in the Raptor slot, R6 + S6fall SERVED OAI predictions:
#     per OA label k: V = WR_k pct(D5) + (1 - WR_k) pct(R6); C = .8 pct(V) + .2 pct(S6fall); WR = .5 / .2 / .5
#     (effective Raptor .40 / .16 / .40, R6 .40 / .64 / .40, S6fall .20; ROUND5.md A.3). All pct are weighted CDFs under the
#     regime (+ bootstrap) weights, i.e. the population analogue of the test-set pct in the kernel.
#  E-OA families on U = pct(C): out_t = U_t - a/2 * sum_o w(t,o) U_o
#     sym  w = 1; pw  w(M,L) .876, w(M,PF) .708, w(L,PF) 1.416 (build_a21.py); LP2  w(L,PF) 2, Medial pairs 0 (Medial
#     untouched; L and PF each subtract a x partner, the sym total weight); LPpw  w(L,PF) 1.416, Medial pairs 0 (= LP2 at .708 a)
#  Regimes: raw OAI; R1g74 (gold74 8-cell joint, skeptic R1); MIX lambda (8-cell joint (1-l) OAI + l gold74); R3p shift
#     (gold74 pairwise log OR shifted by s sd, OAI marginals kept; box_abc_run.py family).
#  Per regime: exact weighted population curve; OAI-size paired Poisson(1) bootstrap (BK draws); public-size (n 390)
#     multinomial resamples (BP draws) -> P(public ordering) and an ABC likelihood against the LB posterior draws (G10, G20,
#     gap960; push/work/box_abc_run.py LBD) with amplitude s log-uniform [.5, 2].
import os, sys, json, time
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

HM = os.path.expanduser('~')
OA = ['Medial OA', 'Lateral OA', 'PF OA']
COMP = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'],
        'PF OA': ['mp', 'lp', 'amf', 'alf']}
G74 = np.array([33, 12, 3, 2, 4, 8, 2, 7], float)       # gold74 8-cell joint, cells 000..111 (M, L, PF bits), local aggregate
G74P = {(0, 1): [45, 5, 12, 9], (0, 2): [36, 14, 6, 16], (1, 2): [37, 20, 5, 9]}
WR = np.array([.5, .2, .5])
FAM = {'sym': {(0, 1): 1., (0, 2): 1., (1, 2): 1.}, 'pw': {(0, 1): .876, (0, 2): .708, (1, 2): 1.416},
       'LP2': {(0, 1): 0., (0, 2): 0., (1, 2): 2.}, 'LPpw': {(0, 1): 0., (0, 2): 0., (1, 2): 1.416}}
FN = list(FAM); NF = len(FN)
AG = np.round(np.arange(.025, .7001, .025), 3); NA = len(AG)
PROX = sys.argv[1] if len(sys.argv) > 1 else 'P_FT'
BK = int(sys.argv[2]) if len(sys.argv) > 2 else 300
BP = int(sys.argv[3]) if len(sys.argv) > 3 else 600
ONLY = sys.argv[4].split(',') if len(sys.argv) > 4 else None
NPUB = 390; HBW = 4e-5; SG = np.exp(np.linspace(np.log(.5), np.log(2.), 15))
SH = [-3., -2.5, -2., -1.75, -1.5, -1.25, -1., -.75, -.5, -.25, 0., .5, 1.]
MIXL = [.25, .5, .75]
LBD = None  # replaced below
LBD = np.array([[0.0005378, 0.0012751, 0.0001003], [0.0006373, 0.0010639, 0.0001612], [0.0006706, 0.001167, 4.89e-05], [0.0006381, 0.0010919, 4.03e-05], [0.0004301, 0.0011358, 0.0001642], [0.0005387, 0.0010354, 7.32e-05], [0.0007278, 0.0012433, 7.7e-05], [0.0008494, 0.0009876, 3.38e-05], [0.0005661, 0.0011786, 3.88e-05], [0.0004943, 0.0008565, 0.0002882], [0.0008089, 0.0011482, 8.86e-05], [0.0005409, 0.000922, 0.0002265], [0.000616, 0.0011942, 5.55e-05], [0.0006127, 0.0010169, 0.0002557], [0.0006851, 0.0011553, 9.81e-05], [0.000577, 0.0014324, 0.000296], [0.0007424, 0.0009397, 0.0001014], [0.0005737, 0.0012564, 5.14e-05], [0.0006306, 0.0011893, 0.0001141], [0.0006542, 0.0011544, 0.0001134], [0.0005432, 0.0010838, 0.0001449], [0.0002612, 0.0009431, 0.0003022], [0.0006233, 0.0011023, 8.62e-05], [0.000536, 0.0011495, 7.87e-05], [0.000767, 0.001224, 0.0001821], [0.0003079, 0.0008726, 0.0002703], [0.000552, 0.0010491, 3.24e-05], [0.0004768, 0.0009237, 0.0001234], [0.0006063, 0.0011286, 7.34e-05], [0.0005949, 0.0013495, 8.42e-05], [0.0007231, 0.0009274, 0.0001085], [0.000684, 0.0012547, 6.26e-05], [0.0005086, 0.0011971, 3.65e-05], [0.0005434, 0.0012365, 4.64e-05], [0.0003489, 0.0009339, 0.0002137], [0.0008063, 0.0012319, 7.92e-05], [0.0006261, 0.0012396, 6.57e-05], [0.0006, 0.0013328, 2.85e-05], [0.0006708, 0.0010276, 0.0002263], [0.0005434, 0.0010921, 0.0001496], [0.0008257, 0.0012189, 0.0001324], [0.0004919, 0.001186, 0.0001067], [0.0004745, 0.0012218, 0.000112], [0.0005977, 0.0010962, 0.0001336], [0.000463, 0.0009303, 0.0002759], [0.0007933, 0.0010831, 0.0001415], [0.0004521, 0.0010414, 0.0001139], [0.0003784, 0.001006, 9.03e-05], [0.0003917, 0.0010603, 0.0002582], [0.0006552, 0.0012058, 0.000237], [0.0005705, 0.0011276, 9.35e-05], [0.0003785, 0.0009962, 0.0001525], [0.0007107, 0.0012141, 5.94e-05], [0.0005773, 0.0011023, 0.0001841], [0.0007957, 0.0011339, 2.55e-05], [0.0007309, 0.0007411, 0.0003043], [0.0008262, 0.0010854, 9.13e-05], [0.0006158, 0.0011304, 8.88e-05], [0.0005485, 0.0013899, 0.0001107], [0.0006939, 0.0011891, 0.0001064], [0.0006866, 0.0012415, 8.33e-05], [0.0004019, 0.0008981, 0.0002027], [0.0006021, 0.0011759, 7.72e-05], [0.0007938, 0.0010319, 5.23e-05], [0.0006286, 0.0010329, 4.75e-05], [0.0013482, 0.0011422, 0.0002595], [0.0006038, 0.0011509, 0.0001186], [0.0002926, 0.0009936, 3.8e-05], [0.000644, 0.0012245, 0.0001163], [0.0004225, 0.0011423, 0.0002006], [0.0007213, 0.0010807, 0.0001453], [0.0007505, 0.001384, 5.71e-05], [0.0005914, 0.000983, 0.0001998], [0.0007145, 0.0012909, 0.0001473], [0.0005891, 0.0011705, 4.88e-05], [0.0006516, 0.0013278, 0.0001038], [0.0004934, 0.0010463, 0.0001159], [0.000469, 0.0011093, 5.86e-05], [0.0005055, 0.0011467, 3.35e-05], [0.0008362, 0.0011223, 0.0001324], [0.0003211, 0.0007937, 0.00028], [0.0006308, 0.0012143, 5.02e-05], [0.0005232, 0.0010341, 0.0001535], [0.0007249, 0.0009907, 0.0001175], [0.000691, 0.0009978, 6.74e-05], [0.0006325, 0.0011455, 6.69e-05], [0.0006942, 0.0009865, 0.0001337], [0.000757, 0.0011954, 5.4e-05], [0.0007156, 0.0011154, 0.0001403], [0.000525, 0.0011615, 8.58e-05], [0.0005739, 0.0009687, 0.0001134], [0.0006255, 0.0011343, 9.69e-05], [0.0005138, 0.001211, 0.0001402], [0.0009845, 0.0012828, 4.3e-05], [0.0006666, 0.0010734, 5.85e-05], [0.0007799, 0.0012392, 4.73e-05], [0.0009495, 0.0012642, 4.1e-05], [0.0005258, 0.0011323, 7.45e-05], [0.0007489, 0.001231, 1.5e-05], [0.0004048, 0.001105, 5.07e-05]])   # rows (G10, G20, gap960) in macro units, from the public LB posterior (lb_post.py)


def oth(t):
    return [m for m in range(3) if m != t]


def ia(a):
    return int(np.argmin(np.abs(AG - a)))


CO = []
for t in range(3):
    o1, o2 = oth(t); rows = [(0., 0.)]
    for f in FN:
        for a in AG:
            rows.append((a / 2 * FAM[f][tuple(sorted((t, o1)))], a / 2 * FAM[f][tuple(sorted((t, o2)))]))
    CO.append(np.array(rows))


def wpct(v, w):
    o = np.argsort(v, kind='mergesort'); vs = v[o]; ws = w[o]
    u, start = np.unique(vs, return_index=True)
    gw = np.add.reduceat(ws, start); below = np.concatenate([[0.], np.cumsum(gw)[:-1]])
    g = np.searchsorted(u, v)
    return (below[g] + .5 * gw[g]) / w.sum()


def c4d5_U(R6, S6, D5, w):
    U = np.empty_like(R6)
    for k in range(3):
        V = WR[k] * wpct(D5[:, k], w) + (1 - WR[k]) * wpct(R6[:, k], w)
        C = .8 * wpct(V, w) + .2 * wpct(S6[:, k], w)
        U[:, k] = wpct(C, w)
    return U


def wauc_cols(S, y, w):
    o = np.argsort(S, axis=0)
    yo = y[o]; wo = w[o]
    nw = wo * (1. - yo); pw_ = wo * yo
    cn = np.cumsum(nw, axis=0) - nw
    return (pw_ * cn).sum(0) / (pw_.sum(0) * nw.sum(0))


def evalD(U, Y, w):
    D = np.empty((3, NF, NA)); base = np.empty(3)
    for t in range(3):
        o1, o2 = oth(t)
        S = U[:, [t]] - U[:, [o1]] * CO[t][None, :, 0] - U[:, [o2]] * CO[t][None, :, 1]
        a = wauc_cols(S, Y[:, t].astype(float), w)
        base[t] = a[0]; D[t] = (a[1:] - a[0]).reshape(NF, NA)
    return D, base


def wcorr(x, y, w):
    p = w / w.sum(); mx = (p * x).sum(); my = (p * y).sum()
    return (p * (x - mx) * (y - my)).sum() / np.sqrt((p * (x - mx) ** 2).sum() * (p * (y - my) ** 2).sum())


def coup(U, Y, w):
    Z = norm.ppf(np.clip(U, 1e-6, 1 - 1e-6))
    rho = np.mean([wcorr(Z[:, i], Z[:, j], w) for i, j in ((0, 1), (0, 2), (1, 2))])
    q = np.mean([np.mean([wcorr(Z[:, j], Y[:, k], w) for j in oth(k)]) / wcorr(Z[:, k], Y[:, k], w) for k in range(3)])
    phi = np.mean([wcorr(Y[:, i].astype(float), Y[:, j].astype(float), w) for i, j in ((0, 1), (0, 2), (1, 2))])
    phip = [wcorr(Y[:, i].astype(float), Y[:, j].astype(float), w) for i, j in ((0, 1), (0, 2), (1, 2))]
    return float(rho), float(q), float(phi), [float(x) for x in phip]


def cells_of(Y):
    return Y[:, 0] * 4 + Y[:, 1] * 2 + Y[:, 2]


def w_cell(cell, target):
    p0 = np.bincount(cell, minlength=8) / len(cell)
    return np.where(p0 > 0, target / np.maximum(p0, 1e-12), 0.)[cell]


def or_cells(p1, p2, lor):
    lo = max(0., p1 + p2 - 1) + 1e-9; hi = min(p1, p2) - 1e-9
    g = lambda p11: np.log(p11) + np.log(1 - p1 - p2 + p11) - np.log(p1 - p11) - np.log(p2 - p11) - lor
    p11 = brentq(g, lo, hi)
    return np.array([[1 - p1 - p2 + p11, p2 - p11], [p1 - p11, p11]])


def w_r3p(Y, sh):
    n = len(Y); k3 = cells_of(Y); p0 = np.bincount(k3, minlength=8) / n
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


def proxy(name):
    s = pd.read_csv(HM + '/oai_build/oai_label_sources.csv', dtype={'kid': str}).set_index('kid')
    Y = pd.DataFrame(index=s.index, columns=OA, dtype=float)
    for lab, regs in COMP.items():
        c = s[['m_cm_' + r for r in regs]].astype(float).values
        area = np.floor(c + 1e-6); ft = np.round((c - area) * 10); fin = np.isfinite(c)
        pos = ((ft >= 1) & fin) if name == 'P_FT' else ((area >= 2) & fin)
        y = pos.any(1).astype(float); y[~fin.any(1)] = np.nan; Y[lab] = y
    return Y.dropna()


def rdp(p):
    d = pd.read_csv(p, dtype={'StudyInstanceUID': str})
    return d.drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')


def r7(x):
    return np.round(np.asarray(x, float), 7).tolist()


def main():
    t0 = time.time()
    Yd = proxy(PROX)
    r6 = rdp(HM + '/r5_oai/run_r6/worker0/predictions.csv')[OA].astype(float)
    s6 = rdp(HM + '/r5_oai/s6_r5/out/predictions.csv')[OA].astype(float)
    d5 = rdp(HM + '/oai_build/teacher/teacher_D5_mean_v00.csv')[OA].astype(float)
    r6 = r6[r6.index.str.startswith('oai_')].dropna()
    ids = Yd.index.intersection(r6.index).intersection(s6.dropna().index).intersection(d5.dropna().index)
    ids = ids.sort_values()
    Y = Yd.loc[ids].values.astype(int); R6 = r6.loc[ids].values; S6 = s6.loc[ids].values; D5 = d5.loc[ids].values
    n = len(Y); cell = cells_of(Y); one = np.ones(n)
    print('OAIGRID | proxy %s | common knees %d | prev M/L/PF %s | BK %d BP %d NPUB %d | families %s | a grid %.3f..%.3f (%d)' % (
        PROX, n, np.round(Y.mean(0), 3).tolist(), BK, BP, NPUB, FN, AG[0], AG[-1], NA), flush=True)
    # sanity: ties and wauc vs rank AUC, unweighted
    U0 = c4d5_U(R6, S6, D5, one)
    from sklearn.metrics import roc_auc_score
    D0, b0 = evalD(U0, Y, one)
    sk = [roc_auc_score(Y[:, t], U0[:, t]) for t in range(3)]
    print('SANITY ties in U columns %s | wauc-vs-sklearn base max |diff| %.2e | raw c4D5 base AUC M/L/PF %s | raw sym OA-mean a .1 / .2 %+.4f / %+.4f (skeptic c4D5 R0: +.0060 / +.0103)' % (
        [int(n - len(np.unique(U0[:, t]))) for t in range(3)], max(abs(b0[t] - sk[t]) for t in range(3)), np.round(b0, 4).tolist(),
        D0[:, 0, ia(.1)].mean(), D0[:, 0, ia(.2)].mean()), flush=True)
    pg = (G74 + .5) / (G74 + .5).sum(); p0 = np.bincount(cell, minlength=8) / n
    REG = [('raw', one), ('R1g74', w_cell(cell, pg))]
    REG += [('MIX%.2f' % l, w_cell(cell, (1 - l) * p0 + l * pg)) for l in MIXL]
    REG += [('R3p%+.2f' % s, w_r3p(Y, s)) for s in SH]
    if ONLY:
        REG = [r for r in REG if r[0] in ONLY]
    rng = np.random.default_rng(20261020)
    i10, i20, i30 = ia(.1), ia(.2), ia(.3); fs, fp = FN.index('sym'), FN.index('pw')
    EXP = {'proxy': PROX, 'n': n, 'FN': FN, 'AG': AG.tolist(), 'reg': {}}
    for rn, w in REG:
        tr = time.time()
        w = w / w.mean(); ess = w.sum() ** 2 / (w ** 2).sum(); p = w / w.sum()
        U = c4d5_U(R6, S6, D5, w); Dp, bp = evalD(U, Y, w); rho, q, phi, phip = coup(U, Y, w)
        Mp = Dp.sum(0) / 12
        # OAI-size paired Poisson bootstrap (R1g74 also with Dirichlet target uncertainty)
        variants = [('knee', None)] + ([('knee+tgt', 'dir')] if rn == 'R1g74' else [])
        BOOT = {}
        for vn, mode in variants:
            DB = []
            for b in range(BK):
                pois = rng.poisson(1., n).astype(float)
                if mode == 'dir':
                    wb = w_cell(cell, rng.dirichlet(G74 + .5)) * pois
                else:
                    wb = w * pois
                Ub = c4d5_U(R6, S6, D5, wb); Db, _ = evalD(Ub, Y, wb); DB.append(Db)
            DB = np.array(DB); MB = DB.sum(1) / 12
            M0 = np.concatenate([np.zeros((len(MB), NF, 1)), MB], 2)
            am = M0.argmax(2)                                   # index into [0, AG...]
            ref = MB[:, fp, i30][:, None, None]
            BOOT[vn] = dict(mean=r7(MB.mean(0)), sd=r7(MB.std(0)), sd_adj=r7((MB[:, :, 1:] - MB[:, :, :-1]).std(0)),
                            mean_adj=r7((MB[:, :, 1:] - MB[:, :, :-1]).mean(0)),
                            sd_vs_pw30=r7((MB - ref).std(0)), p_gt_pw30=r7((MB > ref).mean(0)),
                            sd_pw_minus_sym=r7((MB[:, fp] - MB[:, fs]).std(0)),
                            argmax_counts=[np.bincount(am[:, f], minlength=NA + 1).tolist() for f in range(NF)],
                            lab_sd=r7((DB).std(0)), B=len(MB), cov=np.round(np.cov(MB.reshape(len(MB), -1).T) * 1e8, 5).tolist(),
                            lab_cov=np.round(np.cov(DB.reshape(len(DB), -1).T) * 1e8, 5).tolist())
        # public-size resamples
        DPs = []
        for b in range(BP):
            ii = rng.choice(n, NPUB, p=p); Ys = Y[ii]
            if (Ys.sum(0) < 5).any() or ((NPUB - Ys.sum(0)) < 5).any():
                continue
            o1 = np.ones(NPUB); Us = c4d5_U(R6[ii], S6[ii], D5[ii], o1); Ds, _ = evalD(Us, Ys, o1); DPs.append(Ds.sum(0) / 12)
        DPs = np.array(DPs); J = len(DPs)
        S10, S20, S30 = DPs[:, fs, i10], DPs[:, fs, i20], DPs[:, fs, i30]; PW20, PW30 = DPs[:, fp, i20], DPs[:, fp, i30]
        oc = np.stack([S10 > 0, S20 > S10, PW30 > S20, PW30 > PW20, PW30 > S30], 1)
        ordj = oc.all(1).astype(float)
        a_ = (SG[None, :, None] * S10[:, None, None] - LBD[None, None, :, 0]) / HBW
        b_ = (SG[None, :, None] * S20[:, None, None] - LBD[None, None, :, 1]) / HBW
        K = np.exp(-.5 * (a_ * a_ + b_ * b_))                          # (J, S, L)
        nod = (SG[None, :, None] * (PW30 - S20)[:, None, None] < LBD[None, None, :, 2]).astype(float)
        KJ = K * ordj[:, None, None] * nod
        Wjs = KJ.sum(2)                                                # (J, S)
        Wt = Wjs.sum()
        inc = SG[None, :, None, None] * (DPs - PW30[:, None, None])[:, None]       # (J, S, NF, NA): public increment over pw30
        pp_mean = (Wjs[:, :, None, None] * inc).sum((0, 1)) / max(Wt, 1e-300)
        pp_sq = (Wjs[:, :, None, None] * inc ** 2).sum((0, 1)) / max(Wt, 1e-300)
        pp_pos = (Wjs[:, :, None, None] * (inc > 0)).sum((0, 1)) / max(Wt, 1e-300)
        p960 = np.zeros((NF, NA))
        if Wt > 0:
            lift = SG[None, :, None, None] * (DPs - S20[:, None, None])[:, None]   # (J, S, NF, NA) increment over eoa20
            for f in range(NF):
                for k in range(NA):
                    p960[f, k] = (KJ * (lift[:, :, f, k][:, :, None] > LBD[None, None, :, 2])).sum() / Wt
        s_post = (Wjs.sum(0) / max(Wt, 1e-300))
        RR = dict(ess=float(ess), rho=rho, q=q, gap=rho - q, phi=phi, phi_pairs=phip, base_auc=r7(bp), pop=r7(Mp), pop_lab=r7(Dp),
                  boot=BOOT,
                  pub=dict(J=J, mean=r7(DPs.mean(0)), sd=r7(DPs.std(0)), sd_vs_pw30=r7((DPs - PW30[:, None, None]).std(0)),
                           sd_adj=r7((DPs[:, :, 1:] - DPs[:, :, :-1]).std(0)), cov=np.round(np.cov(DPs.reshape(J, -1).T) * 1e8, 5).tolist(),
                           p_order_each=r7(oc.mean(0)), p_order_all=float(ordj.mean()),
                           L_mag=float(K.mean()), L_mag_order=float((K * ordj[:, None, None]).mean()), L_joint=float(KJ.mean()),
                           W_joint=float(Wt), s_post=r7(s_post), pp_mean=r7(pp_mean), pp_sd=r7(np.sqrt(np.maximum(pp_sq - pp_mean ** 2, 0))),
                           pp_pos=r7(pp_pos), p960=r7(p960)))
        EXP['reg'][rn] = RR
        am_pop = [float(AG[np.argmax(Mp[f])]) if Mp[f].max() > 0 else 0. for f in range(NF)]
        bk = BOOT['knee']
        print('[%s %-9s] ESS %4.0f phi %.3f (ML %.3f MPF %.3f LPF %.3f) rho %.3f q %.3f gap %+.3f | pop macro e-4 sym .1/.2/.3/.4 %s pw .2/.3/.4 %s LP2 .2/.3/.4 %s | argmax sym/pw/LP2/LPpw %s | P(order) %.3f each %s | L_mag %.3e L_joint %.3e | %.0fs' % (
            PROX, rn, ess, phi, *phip, rho, q, rho - q,
            np.round(Mp[fs, [ia(.1), i20, i30, ia(.4)]] * 1e4, 2).tolist(), np.round(Mp[fp, [i20, i30, ia(.4)]] * 1e4, 2).tolist(),
            np.round(Mp[FN.index('LP2'), [i20, i30, ia(.4)]] * 1e4, 2).tolist(), am_pop, ordj.mean(), np.round(oc.mean(0), 3).tolist(),
            K.mean(), KJ.mean(), time.time() - tr), flush=True)
    print('JSON_BEGIN'); print(json.dumps(EXP)); print('JSON_END', flush=True)
    print('DONE %.0fs' % (time.time() - t0))


main()
