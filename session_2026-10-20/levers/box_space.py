# box_space.py (10-20 levers, Claude): output-only transform families AFTER E-OA, measured on OAI expert MOAKS reads with
# OAI-free members (R6 + S6fall served on the OAI store rebuild, ~/r5_oai; D5 / s1044 / s1045 fleet teacher files).
# Read-only. Prints AGGREGATES ONLY (AUC deltas, sd, P, rho, q, gap, ESS, n, prevalence, argmax). Never writes a file,
# never prints an id, a row or a per-knee value. Piped via ssh stdin:
#   ssh training@192.168.10.30 '~/miniconda3/envs/rsna/bin/python -I - JOB B' < box_space.py
# Role mapping (MEASURED in r5/r6_oai.md sec 4.3 / 11): on OAI under gold74 coupling R6 / S6fall are the LOW-gap members
# (gap +.006 / -.004) and the fleet D5 / s1044 / s1045 the HIGH-gap ones (D5 +.080). On RSNA gold58 the c4 Raptor slot is the
# lower-gap member (r384 / r224 ~.05 CLAIMED) and R6 / S6fall the higher-gap ones (.09 / .13). So "LOW in the Raptor slot" is
# the role-matched proxy for the question "after E-OA, more or less Raptor OA weight?"; "HIGH in the Raptor slot" (c4D5) is
# the role-reversed control: a gap mechanism must flip the sign there.
# JOB F1 = Raptor-slot weight x E-OA interaction (2-member blends + c4-shaped nest, role-matched and role-reversed).
# JOB F2 = transform families on top of FINAL E-OA: regime-aware member E-OA, probit space, agreement gating, convex /
#          concave subtraction, cross-member subtraction, own-label disagreement term. Envelopes vs FINAL.
import os, sys, glob, time
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

T0 = time.time()
H = os.path.expanduser('~/oai_build'); R5 = os.path.expanduser('~/r5_oai')
OA = ['Medial OA', 'Lateral OA', 'PF OA']
COMP = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'],
        'PF OA': ['mp', 'lp', 'amf', 'alf']}
G74OA = {'000': 33, '001': 12, '010': 3, '011': 2, '100': 4, '101': 8, '110': 2, '111': 7}
G74P = {(0, 1): [45, 5, 12, 9], (0, 2): [36, 14, 6, 16], (1, 2): [37, 20, 5, 9]}
G74 = np.array([G74OA[format(i, '03b')] for i in range(8)], float)
WR = np.array([.5, .2, .5])
JOB = sys.argv[1] if len(sys.argv) > 1 else 'F1'
B = int(sys.argv[2]) if len(sys.argv) > 2 else 150
rng = np.random.default_rng(2010)


def wpct(v, w):
    o = np.argsort(v, kind='mergesort'); vs = v[o]; ws = w[o]
    u, start = np.unique(vs, return_index=True)
    gw = np.add.reduceat(ws, start); below = np.concatenate([[0.], np.cumsum(gw)[:-1]])
    g = np.searchsorted(u, v)
    return (below[g] + .5 * gw[g]) / w.sum()


def WP(X, w):
    return np.stack([wpct(X[:, k], w) for k in range(X.shape[1])], 1)


def wauc(s, y, w):
    u, inv = np.unique(s, return_inverse=True)
    pw = np.bincount(inv, weights=w * (y == 1), minlength=len(u)); nw = np.bincount(inv, weights=w * (y == 0), minlength=len(u))
    cn = np.cumsum(nw) - nw
    return float((pw * (cn + .5 * nw)).sum() / (pw.sum() * nw.sum()))


def oam(S, Y, w):
    return float(np.mean([wauc(S[:, k], Y[:, k], w) for k in range(3)]))


def per(S, Y, w):
    return np.array([wauc(S[:, k], Y[:, k], w) for k in range(3)])


def wcorr(a, b, p):
    ma = (p * a).sum(); mb = (p * b).sum()
    return float((p * (a - ma) * (b - mb)).sum() / np.sqrt((p * (a - ma) ** 2).sum() * (p * (b - mb) ** 2).sum()))


def coupling(F, Y, w):
    p = w / w.sum(); Z = norm.ppf(np.clip(WP(F, w), 1e-6, 1 - 1e-6))
    rho = np.mean([wcorr(Z[:, i], Z[:, j], p) for i, j in ((0, 1), (0, 2), (1, 2))])
    q = np.mean([np.mean([wcorr(Z[:, j], Y[:, k].astype(float), p) for j in range(3) if j != k]) / wcorr(Z[:, k], Y[:, k].astype(float), p)
                 for k in range(3)])
    return rho, q, rho - q


# ---------------- labels (float parse; skeptic showed 0 mismatches vs the string parse) ----------------
def labels(proxy):
    S = pd.read_csv(H + '/oai_label_sources.csv', dtype={'kid': str}).set_index('kid')
    Y = pd.DataFrame(index=S.index, columns=OA, dtype=float)
    for c in OA:
        cc = S[['m_cm_' + r for r in COMP[c]]].astype(float).values
        area = np.floor(cc + 1e-6); ft = np.round((cc - area) * 10); fin = np.isfinite(cc)
        pos = ((ft >= 1) & fin) if proxy == 'P_FT' else ((area >= 2) & fin)
        y = pos.any(1).astype(float); y[~fin.any(1)] = np.nan; Y[c] = y
    return Y.dropna()


def rdp(p):
    d = pd.read_csv(p, dtype={'StudyInstanceUID': str})
    return d.drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')


def members():
    P = {'R6': rdp(R5 + '/run_r6/worker0/predictions.csv'), 'S6': rdp(R5 + '/s6_r5/out/predictions.csv'),
         'D5': rdp(H + '/teacher/teacher_D5_mean_v00.csv'),
         's1044': rdp(H + '/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv'),
         's1045': rdp(H + '/teacher2/f0_s1045_ttr288frac75__raw__A__gpu.csv')}
    P = {k: v[OA].astype(float) for k, v in P.items()}
    P['R6'] = P['R6'][P['R6'].index.str.startswith('oai_')]
    P['S6'] = P['S6'][P['S6'].index.str.startswith('oai_')]
    return P


# ---------------- weighting schemes ----------------
def or_cells(p1, p2, lor):
    lo = max(0., p1 + p2 - 1) + 1e-9; hi = min(p1, p2) - 1e-9
    g = lambda p11: np.log(p11) + np.log(1 - p1 - p2 + p11) - np.log(p1 - p11) - np.log(p2 - p11) - lor
    p11 = brentq(g, lo, hi)
    return np.array([[1 - p1 - p2 + p11, p2 - p11], [p1 - p11, p11]])


def cells_of(Y):
    return (Y[:, 0] * 4 + Y[:, 1] * 2 + Y[:, 2]).astype(int)


def r1w(cell, pg, bw=None):
    bw = np.ones(len(cell)) if bw is None else bw
    po = np.bincount(cell, weights=bw, minlength=8) / bw.sum()
    return bw * (pg / np.maximum(po, 1e-12))[cell]


def r3p(Y, sh):
    n = len(Y); k3 = cells_of(Y); p0 = np.bincount(k3, minlength=8) / n
    cells = np.array([[(c >> (2 - i)) & 1 for i in range(3)] for c in range(8)]); marg = Y.mean(0)
    q = p0.copy(); tg = {}
    for (i, j), c in G74P.items():
        c = np.array(c, float) + .5; lor = np.log(c[3] * c[0] / (c[1] * c[2])) + sh * np.sqrt((1 / c).sum())
        tg[(i, j)] = or_cells(marg[i], marg[j], lor)
    for _ in range(500):
        for (i, j), t in tg.items():
            for a in (0, 1):
                for b in (0, 1):
                    m = (cells[:, i] == a) & (cells[:, j] == b); s = q[m].sum()
                    if s > 0: q[m] *= t[a, b] / s
        q /= q.sum()
    return np.where(p0 > 0, q / np.maximum(p0, 1e-12), 0)[k3]


def schemes(Y):
    cell = cells_of(Y); pg0 = (G74 + .5) / (G74 + .5).sum()
    return {'R1': r1w(cell, pg0), 'R3p': r3p(Y, 0.), 'R3p_lo': r3p(Y, -1.), 'R3p_hi': r3p(Y, 1.)}


def draws(Y, nb):
    """knee Poisson(1) x target Dirichlet(gold74 + .5) draws of R1 weights."""
    cell = cells_of(Y)
    for _ in range(nb):
        yield r1w(cell, rng.dirichlet(G74 + .5), rng.poisson(1., len(Y)).astype(float))


# ---------------- transforms ----------------
def eoa(F, a, w, space='pct', sub=None, gate=None, h=None):
    """FINAL E-OA family on a 3-col OA score F. space pct / probit; sub = 3-col matrix whose other-label columns are
    subtracted (default: F itself); gate = per-knee multiplier on a (mean 1); h = monotone map applied to the subtracted
    pct (convex / concave subtraction)."""
    U = WP(F, w)
    Sb = U if sub is None else WP(sub, w)
    if h is not None:
        Sb = h(Sb)
    if space == 'probit':
        U = norm.ppf(np.clip(U, 1e-4, 1 - 1e-4)); Sb = norm.ppf(np.clip(Sb, 1e-4, 1 - 1e-4))
    g = np.ones(len(U)) if gate is None else gate
    out = np.empty_like(U)
    for t in range(3):
        o = [k for k in range(3) if k != t]
        out[:, t] = U[:, t] - a * g * (Sb[:, o[0]] + Sb[:, o[1]]) / 2.0
    return out


def blend2(L, Hh, wl, w):
    return wl * WP(L, w) + (1 - wl) * WP(Hh, w)


def c4nest(RS, R6s, S6s, m, w):
    """c4 cell 14 + 18 OA structure: V = m WR pct(RS) + (1 - m WR) pct(R6s); final = .8 pct(V) + .2 pct(S6s)."""
    wr = np.clip(m * WR, 0, 1)
    V = wr * WP(RS, w) + (1 - wr) * WP(R6s, w)
    return .8 * WP(V, w) + .2 * WP(S6s, w)


def st(x):
    x = np.asarray(x); return '%+.5f (sd %.5f P %.2f)' % (x.mean(), x.std(), (x > 0).mean())


# ---------------- setup ----------------
P = members()
YD = {px: labels(px) for px in ('P_FT', 'P_AREA')}
ids = YD['P_FT'].index.intersection(YD['P_AREA'].index)
for k in P:
    ids = ids.intersection(P[k].dropna().index)
ids = sorted(ids)
F = {k: P[k].loc[ids].values.astype(np.float64) for k in P}
YY = {px: YD[px].loc[ids].values.astype(int) for px in YD}
n = len(ids)
print('JOB %s | n %d common knees | prevalence P_FT %s P_AREA %s | B %d' % (
    JOB, n, np.round(YY['P_FT'].mean(0), 3).tolist(), np.round(YY['P_AREA'].mean(0), 3).tolist(), B), flush=True)
Y = YY['P_FT']; W = schemes(Y)
for sn, w in W.items():
    p = w / w.sum()
    print('  scheme %-6s ESS %.0f | label phi %.3f' % (sn, 1 / (p ** 2).sum(), np.mean([wcorr(Y[:, i].astype(float), Y[:, j].astype(float), p)
                                                                                   for i, j in ((0, 1), (0, 2), (1, 2))])), flush=True)
for k in F:
    s = ' | '.join('%s rho %.3f q %.3f gap %+.3f' % ((sn,) + coupling(F[k], Y, W[sn])) for sn in ('R1', 'R3p_lo'))
    print('  member %-6s AUC(R1) %s | %s' % (k, np.round(per(F[k], Y, W['R1']), 4).tolist(), s), flush=True)

AL = [0., .1, .2, .3]


def job_f1():
    # ---- 2-member blends: Raptor-slot weight wl on the LOW member, E-OA FINAL ----
    PAIRS = [('R6', 'D5'), ('S6', 'D5'), ('R6', 's1044'), ('R6', 's1045'), ('D5', 'R6')]   # last = role-reversed control
    WL = np.round(np.arange(0, 1.01, .1), 2)
    for (lo, hi) in PAIRS:
        for sn in ('R1', 'R3p', 'R3p_lo', 'R3p_hi'):
            w = W[sn]; G = np.zeros((len(WL), len(AL)))
            for i, wl in enumerate(WL):
                Fb = blend2(F[lo], F[hi], wl, w)
                for j, a in enumerate(AL):
                    G[i, j] = oam(eoa(Fb, a, w), Y, w)
            best = [float(WL[np.argmax(G[:, j])]) for j in range(len(AL))]
            # local slope around wl .4: A(.55) - A(.25) at each a (interpolated on the .1 grid: .5/.6 and .2/.3 means)
            sl = [(G[5, j] + G[6, j]) / 2 - (G[2, j] + G[3, j]) / 2 for j in range(len(AL))]
            gap = coupling(blend2(F[lo], F[hi], .4, w), Y, w)
            print('F1 pair LOW=%-5s HIGH=%-5s %-6s | blend.4 gap %+.3f | argmax wl by a %s %s | slope(.25->.55) by a %s | '
                  'E-OA gain at wl .4 by a %s' % (lo, hi, sn, gap[2], AL, best, np.round(sl, 5).tolist(),
                                                 np.round([(G[4, j] + G[4, j]) / 2 - G[4, 0] for j in range(len(AL))], 5).tolist()), flush=True)
        # P_AREA point estimates under R1
        w = W['R1']; Ya = YY['P_AREA']; wa = r1w(cells_of(Ya), (G74 + .5) / (G74 + .5).sum())
        G = np.zeros((len(WL), len(AL)))
        for i, wl in enumerate(WL):
            Fb = blend2(F[lo], F[hi], wl, wa)
            for j, a in enumerate(AL):
                G[i, j] = oam(eoa(Fb, a, wa), Ya, wa)
        sl = [(G[5, j] + G[6, j]) / 2 - (G[2, j] + G[3, j]) / 2 for j in range(len(AL))]
        print('F1 pair LOW=%-5s HIGH=%-5s P_AREA R1 | argmax wl by a %s | slope by a %s' % (
            lo, hi, [float(WL[np.argmax(G[:, j])]) for j in range(len(AL))], np.round(sl, 5).tolist()), flush=True)
        # bootstrap of the interaction I(a) = slope(a) - slope(0), R1 knee x target
        Ib = {a: [] for a in (.2, .3)}; Sb0 = []
        for wb in draws(Y, B):
            g = {}
            for wl in (.25, .55):
                Fb = blend2(F[lo], F[hi], wl, wb)
                for a in (0., .2, .3):
                    g[(wl, a)] = oam(eoa(Fb, a, wb), Y, wb)
            s0 = g[(.55, 0.)] - g[(.25, 0.)]; Sb0.append(s0)
            for a in (.2, .3):
                Ib[a].append((g[(.55, a)] - g[(.25, a)]) - s0)
        print('F1 pair LOW=%-5s HIGH=%-5s R1 boot | slope(.25->.55) at a 0 %s | interaction I(.2) %s | I(.3) %s' % (
            lo, hi, st(Sb0), st(Ib[.2]), st(Ib[.3])), flush=True)

    # ---- c4-shaped nest: Raptor slot multiplier m on WR (OA), E-OA FINAL ----
    NEST = [('ROLE R6|D5|s1044', 'R6', 'D5', 's1044'), ('ROLE S6|D5|s1045', 'S6', 'D5', 's1045'),
            ('ROLE R6|s1044|S6x', 'R6', 's1044', 's1045'), ('REVERSED c4D5 D5|R6|S6', 'D5', 'R6', 'S6')]
    MS = [0., .5, 1., 1.5, 2.]
    for nm, rs, r6s, s6s in NEST:
        for sn in ('R1', 'R3p', 'R3p_lo', 'R3p_hi'):
            w = W[sn]; G = np.zeros((len(MS), len(AL))); PL = {}
            for i, m in enumerate(MS):
                Fb = c4nest(F[rs], F[r6s], F[s6s], m, w)
                for j, a in enumerate(AL):
                    S = eoa(Fb, a, w); G[i, j] = oam(S, Y, w)
                    if m in (.5, 1.5) and a in (0., .2):
                        PL[(m, a)] = per(S, Y, w)
            gap = coupling(c4nest(F[rs], F[r6s], F[s6s], 1., w), Y, w)
            d = [G[3, j] - G[1, j] for j in range(len(AL))]
            plab = (PL[(1.5, .2)] - PL[(.5, .2)]) - (PL[(1.5, 0.)] - PL[(.5, 0.)])
            print('F1 nest %-24s %-6s | gap(m1) %+.3f | argmax m by a %s | D = A(m1.5) - A(m.5) by a %s | I(.2) per label M/L/PF %s | '
                  'E-OA gain at m1 %s' % (nm, sn, gap[2], [MS[int(np.argmax(G[:, j]))] for j in range(len(AL))], np.round(d, 5).tolist(),
                                          np.round(plab, 5).tolist(), np.round(G[2] - G[2, 0], 5).tolist()), flush=True)
        Ib = {a: [] for a in (.2, .3)}; D0 = []
        for wb in draws(Y, B):
            g = {}
            for m in (.5, 1.5):
                Fb = c4nest(F[rs], F[r6s], F[s6s], m, wb)
                for a in (0., .2, .3):
                    g[(m, a)] = oam(eoa(Fb, a, wb), Y, wb)
            d0 = g[(1.5, 0.)] - g[(.5, 0.)]; D0.append(d0)
            for a in (.2, .3):
                Ib[a].append((g[(1.5, a)] - g[(.5, a)]) - d0)
        print('F1 nest %-24s R1 boot | D(a 0) %s | I(.2) %s | I(.3) %s' % (nm, st(D0), st(Ib[.2]), st(Ib[.3])), flush=True)


def job_f2():
    AGRID = [0., .05, .1, .15, .2, .25, .3, .35, .4, .5]
    CFG = [('pair R6|D5 wl.4', 'pair', 'R6', 'D5', None), ('pair S6|D5 wl.4', 'pair', 'S6', 'D5', None),
           ('nest R6|D5|s1044', 'nest', 'R6', 'D5', 's1044'), ('REVERSED nest D5|R6|S6', 'nest', 'D5', 'R6', 'S6')]
    sq = lambda U: U ** 2
    sr = lambda U: np.sqrt(U)

    def build(kind, lo, hi, x, w):
        if kind == 'pair':
            return blend2(F[lo], F[hi], .4, w)
        return c4nest(F[lo], F[hi], F[x], 1., w)

    def highonly(kind, lo, hi, x, w, a):
        # regime-aware: E-OA on the HIGH member(s) before they enter the blend; LOW (Raptor-slot) member untouched
        if kind == 'pair':
            return .4 * WP(F[lo], w) + .6 * WP(eoa(F[hi], a, w), w)
        return c4nest(F[lo], eoa(F[hi], a, w), eoa(F[x], a, w), 1., w)

    def lowonly(kind, lo, hi, x, w, a):
        if kind == 'pair':
            return .4 * WP(eoa(F[lo], a, w), w) + .6 * WP(F[hi], w)
        return c4nest(eoa(F[lo], a, w), F[hi], F[x], 1., w)

    def gate(kind, lo, hi, x, w, sign):
        # per-knee disagreement of the LOW and HIGH members on the OTHER compartments, ranked; sign +1: stronger when they agree
        UL = WP(F[lo], w); UH = WP(F[hi], w)
        g = np.zeros((len(UL), 3))
        for t in range(3):
            o = [k for k in range(3) if k != t]
            g[:, t] = np.abs(UL[:, o] - UH[:, o]).mean(1)
        r = wpct(g.mean(1), w)
        return (1.5 - r) if sign > 0 else (.5 + r)

    FAM = {
        'FINAL': lambda k, lo, hi, x, w, a: eoa(build(k, lo, hi, x, w), a, w),
        'HIGHONLY': lambda k, lo, hi, x, w, a: highonly(k, lo, hi, x, w, a),
        'LOWONLY(ctl)': lambda k, lo, hi, x, w, a: lowonly(k, lo, hi, x, w, a),
        'PROBIT': lambda k, lo, hi, x, w, a: eoa(build(k, lo, hi, x, w), a, w, space='probit'),
        'GATE_AGREE': lambda k, lo, hi, x, w, a: eoa(build(k, lo, hi, x, w), a, w, gate=gate(k, lo, hi, x, w, +1)),
        'GATE_DISAGREE': lambda k, lo, hi, x, w, a: eoa(build(k, lo, hi, x, w), a, w, gate=gate(k, lo, hi, x, w, -1)),
        'CONVEX(u^2)': lambda k, lo, hi, x, w, a: eoa(build(k, lo, hi, x, w), a, w, h=sq),
        'CONCAVE(sqrt)': lambda k, lo, hi, x, w, a: eoa(build(k, lo, hi, x, w), a, w, h=sr),
        'SUB_HIGH': lambda k, lo, hi, x, w, a: eoa(build(k, lo, hi, x, w), a, w, sub=F[hi]),
        'SUB_LOW': lambda k, lo, hi, x, w, a: eoa(build(k, lo, hi, x, w), a, w, sub=F[lo]),
    }
    for cn, kind, lo, hi, x in CFG:
        for sn in ('R1', 'R3p', 'R3p_lo', 'R3p_hi'):
            w = W[sn]; base = oam(build(kind, lo, hi, x, w), Y, w); best = {}
            line = []
            for fn, f in FAM.items():
                cur = [oam(f(kind, lo, hi, x, w, a), Y, w) - base if a else 0. for a in AGRID]
                j = int(np.argmax(cur)); best[fn] = AGRID[j]
                line.append('%s best a %.2f env %+.5f [a.1 %+.5f a.2 %+.5f a.3 %+.5f]' % (fn, AGRID[j], cur[j], cur[2], cur[4], cur[6]))
            print('F2 %-22s %-6s base %.4f | ' % (cn, sn, base) + ' || '.join(line), flush=True)
            if sn == 'R1':
                BEST = dict(best)
        # paired bootstrap (R1 knee x target) of each family at its full-sample best a minus FINAL at its best a
        diffs = {fn: [] for fn in FAM if fn != 'FINAL'}
        for wb in draws(Y, B):
            ref = oam(FAM['FINAL'](kind, lo, hi, x, wb, BEST['FINAL']), Y, wb)
            for fn in diffs:
                diffs[fn].append(oam(FAM[fn](kind, lo, hi, x, wb, BEST[fn]), Y, wb) - ref if BEST[fn] else
                                 oam(build(kind, lo, hi, x, wb), Y, wb) - ref)
        print('F2 %-22s R1 boot vs FINAL(a %.2f) | ' % (cn, BEST['FINAL']) +
              ' | '.join('%s(a %.2f) %s' % (fn, BEST[fn], st(v)) for fn, v in diffs.items()), flush=True)


PW = {(0, 1): .876, (0, 2): .708, (1, 2): 1.416}


def eoah(F, a, w, h, pw=False):
    """E-OA with a monotone map h on the SUBTRACTED other-compartment pct (h = None: the kernel's linear E-OA), optional A21
    pair weights."""
    U = WP(F, w); Sb = U if h is None else h(U); out = np.empty_like(U)
    for t in range(3):
        o = [k for k in range(3) if k != t]
        ww = [PW[tuple(sorted((t, k)))] for k in o] if pw else [1., 1.]
        out[:, t] = U[:, t] - a * (ww[0] * Sb[:, o[0]] + ww[1] * Sb[:, o[1]]) / 2.0
    return out


HF = {'lin': None, 'p1.5': lambda U: U ** 1.5, 'p2': lambda U: U ** 2, 'p3': lambda U: U ** 3,
      'hinge.3': lambda U: np.maximum(U - .3, 0) / .7, 'hinge.5': lambda U: np.maximum(U - .5, 0) / .5}
AG3 = np.round(np.arange(0, .501, .025), 3)


def job_f3():
    CF = {'N1 R6|D5|s1044': lambda w: c4nest(F['R6'], F['D5'], F['s1044'], 1., w),
          'N2 S6|D5|s1045': lambda w: c4nest(F['S6'], F['D5'], F['s1045'], 1., w),
          'N3 R6|s1044|s1045': lambda w: c4nest(F['R6'], F['s1044'], F['s1045'], 1., w),
          'N4 REV D5|R6|S6': lambda w: c4nest(F['D5'], F['R6'], F['S6'], 1., w),
          'N5 fleet s1044|D5|s1045': lambda w: c4nest(F['s1044'], F['D5'], F['s1045'], 1., w),
          'FLAT3 D5,s1044,s1045': lambda w: (WP(F['D5'], w) + WP(F['s1044'], w) + WP(F['s1045'], w)) / 3,
          'R6+S6 .8/.2': lambda w: .8 * WP(F['R6'], w) + .2 * WP(F['S6'], w)}
    for k in ('R6', 'S6', 'D5', 's1044', 's1045'):
        CF['single ' + k] = (lambda kk: (lambda w: WP(F[kk], w)))(k)
    # ---- 1. envelopes per h, all configs x schemes x proxies ----
    tally = {h: [] for h in HF if h != 'lin'}
    for px in ('P_FT', 'P_AREA'):
        Yp = YY[px]; Wp = {'R1': r1w(cells_of(Yp), (G74 + .5) / (G74 + .5).sum()), 'R3p': r3p(Yp, 0.), 'R3p_lo': r3p(Yp, -1.),
                          'R3p_hi': r3p(Yp, 1.)}
        for cn, f in CF.items():
            for sn, w in Wp.items():
                Fb = f(w); base = oam(Fb, Yp, w); env = {}
                for h, hf in HF.items():
                    cur = np.array([oam(eoah(Fb, a, w, hf), Yp, w) - base if a else 0. for a in AG3])
                    j = int(np.argmax(cur)); env[h] = (cur[j], AG3[j], cur[8], cur[12])
                for h in tally:
                    tally[h].append(env[h][0] - env['lin'][0])
                print('F3 env %-6s %-24s %-6s | lin env %+.5f (a %.3f; a.2 %+.5f a.3 %+.5f) | ' % (px, cn, sn, *env['lin']) +
                      ' | '.join('%s %+.5f (a %.3f) d %+.5f' % (h, env[h][0], env[h][1], env[h][0] - env['lin'][0]) for h in HF if h != 'lin'),
                      flush=True)
    print('F3 TALLY over %d config x scheme x proxy cells: env(h) - env(lin): ' % len(tally['p2']) +
          ' | '.join('%s mean %+.5f min %+.5f >0 in %d' % (h, np.mean(v), np.min(v), int((np.array(v) > 1e-6).sum())) for h, v in tally.items()),
          flush=True)
    # ---- 2. held-out a selection (split halves stratified by 8-cell), R1 P_FT ----
    cell = cells_of(Y)
    for cn in ('N1 R6|D5|s1044', 'N3 R6|s1044|s1045', 'N5 fleet s1044|D5|s1045', 'FLAT3 D5,s1044,s1045', 'single D5', 'single R6'):
        res = {h: [] for h in HF}
        for sp in range(20):
            half = np.zeros(n, bool)
            for c in range(8):
                ix = np.where(cell == c)[0]; rng.shuffle(ix); half[ix[:len(ix) // 2]] = True
            for tr, te in ((half, ~half), (~half, half)):
                pg = (G74 + .5) / (G74 + .5).sum()
                wtr = np.zeros(n); wtr[tr] = r1w(cell[tr], pg); wte = np.zeros(n); wte[te] = r1w(cell[te], pg)
                # each half is scored on its own knees only (pct recomputed inside the half)
                Xtr = {k: F[k][tr] for k in F}; Xte = {k: F[k][te] for k in F}
                for h, hf in HF.items():
                    best, ba = -9, 0.
                    Fb_tr = _cf_sub(cn, Xtr, wtr[tr]); b0 = oam(Fb_tr, Y[tr], wtr[tr])
                    for a in AG3:
                        v = oam(eoah(Fb_tr, a, wtr[tr], hf), Y[tr], wtr[tr]) - b0 if a else 0.
                        if v > best: best, ba = v, a
                    Fb_te = _cf_sub(cn, Xte, wte[te])
                    res[h].append(oam(eoah(Fb_te, ba, wte[te], hf), Y[te], wte[te]) - oam(Fb_te, Y[te], wte[te]))
        lin = np.array(res['lin'])
        print('F3 heldout %-24s R1 P_FT 20 splits x 2 | lin held-out gain %+.5f | ' % (cn, lin.mean()) +
              ' | '.join('%s %+.5f vs lin %s' % (h, np.mean(res[h]), st(np.array(res[h]) - lin)) for h in HF if h != 'lin'), flush=True)
    # ---- 3. paired bootstrap at full-sample a*, N1 / N3 / FLAT3, R1 knee x target ----
    for cn in ('N1 R6|D5|s1044', 'N3 R6|s1044|s1045', 'FLAT3 D5,s1044,s1045'):
        w = W['R1']; Fb = CF[cn](w); base = oam(Fb, Y, w); ast = {}
        for h, hf in HF.items():
            cur = [oam(eoah(Fb, a, w, hf), Y, w) - base if a else 0. for a in AG3]; ast[h] = AG3[int(np.argmax(cur))]
        D = {h: [] for h in HF if h != 'lin'}
        for wb in draws(Y, B):
            Fbb = CF[cn](wb); ref = oam(eoah(Fbb, ast['lin'], wb, None), Y, wb)
            for h in D:
                D[h].append(oam(eoah(Fbb, ast[h], wb, HF[h]), Y, wb) - ref)
        print('F3 boot %-24s R1 (a* lin %.3f) | ' % (cn, ast['lin']) + ' | '.join('%s(a %.3f) %s' % (h, ast[h], st(v)) for h, v in D.items()),
              flush=True)
    # ---- 4. on top of the A21 pair weights (pw) and at the in-flight strengths ----
    for cn in ('N1 R6|D5|s1044', 'N3 R6|s1044|s1045', 'N5 fleet s1044|D5|s1045', 'FLAT3 D5,s1044,s1045'):
        for sn in ('R1', 'R3p_lo'):
            w = W[sn]; Fb = CF[cn](w); base = oam(Fb, Y, w)
            row = []
            for pw in (False, True):
                for h in ('lin', 'p1.5', 'p2'):
                    cur = [oam(eoah(Fb, a, w, HF[h], pw), Y, w) - base for a in (.2, .25, .3)]
                    row.append('%s%s a.2/.25/.3 %s' % ('pw-' if pw else 'sym-', h, np.round(cur, 5).tolist()))
            print('F3 inflight %-24s %-6s | ' % (cn, sn) + ' | '.join(row), flush=True)
    # ---- 5. Raptor-slot multiplier fine grid x E-OA (lin / p2), A(m) - A(m 1) ----
    MS = [0., .25, .5, .75, 1., 1.25, 1.5, 2.]
    for nm, rs, r6s, s6s in (('N1', 'R6', 'D5', 's1044'), ('N2', 'S6', 'D5', 's1045'), ('N3', 'R6', 's1044', 's1045'), ('N4 REV', 'D5', 'R6', 'S6')):
        for sn in ('R1', 'R3p_lo'):
            w = W[sn]
            for h in ('lin', 'p2'):
                rows = []
                for a in (0., .2, .3):
                    A = np.array([oam(eoah(c4nest(F[rs], F[r6s], F[s6s], m, w), a, w, HF[h]), Y, w) for m in MS])
                    rows.append('a %.1f: %s' % (a, np.round(A - A[MS.index(1.)], 5).tolist()))
                print('F3 mgrid %-6s %-6s %-3s m %s | ' % (nm, sn, h, MS) + ' | '.join(rows), flush=True)
            # PF-only multiplier
            for a in (0., .2, .3):
                vals = []
                for mpf in (.2, .5, 1., 1.5):
                    wr = np.array([.5, .2, .5 * mpf]); V = wr * WP(F[rs], w) + (1 - wr) * WP(F[r6s], w)
                    Fb = .8 * WP(V, w) + .2 * WP(F[s6s], w); vals.append(per(eoah(Fb, a, w, None), Y, w))
                vals = np.array(vals); print('F3 PFonly %-6s %-6s a %.1f | PF Raptor share x (.2,.5,1,1.5): PF AUC - at x1 %s | OA-mean - at x1 %s' % (
                    nm, sn, a, np.round(vals[:, 2] - vals[2, 2], 5).tolist(), np.round(vals.mean(1) - vals[2].mean(), 5).tolist()), flush=True)


def _cf_sub(cn, X, w):
    if cn.startswith('N1'):
        return c4nest(X['R6'], X['D5'], X['s1044'], 1., w)
    if cn.startswith('N3'):
        return c4nest(X['R6'], X['s1044'], X['s1045'], 1., w)
    if cn.startswith('N5'):
        return c4nest(X['s1044'], X['D5'], X['s1045'], 1., w)
    if cn.startswith('FLAT3'):
        return (WP(X['D5'], w) + WP(X['s1044'], w) + WP(X['s1045'], w)) / 3
    return WP(X[cn.split()[1]], w)


def job_f4():
    """S6-slot OA weight x E-OA (role-matched: Raptor slot R6 = low gap, R6 slot s1045, S6 slot D5 = the highest gap, as
    S6fall is on gold58), and Raptor m .75 x convex p2 additivity."""
    def nest(rs, r6s, s6s, m, ws6, w):
        wr = np.clip(m * WR, 0, 1); V = wr * WP(F[rs], w) + (1 - wr) * WP(F[r6s], w)
        return (1 - ws6) * WP(V, w) + ws6 * WP(F[s6s], w)
    for nm, rs, r6s, s6s in (('ROLE R6|s1045|D5', 'R6', 's1045', 'D5'), ('ROLE S6|s1044|D5', 'S6', 's1044', 'D5'),
                             ('REV D5|s1045|R6', 'D5', 's1045', 'R6')):
        for sn in ('R1', 'R3p_lo'):
            w = W[sn]; rows = []
            for a in (0., .2, .3):
                A = [oam(eoah(nest(rs, r6s, s6s, 1., ws, w), a, w, None), Y, w) for ws in (.1, .2, .3, .4)]
                rows.append('a %.1f: %s' % (a, np.round(np.array(A) - A[1], 5).tolist()))
            print('F4 S6slot %-18s %-6s ws6 (.1,.2,.3,.4) minus ws6 .2 | ' % (nm, sn) + ' | '.join(rows), flush=True)
        Ib = {a: [] for a in (.2, .3)}; D0 = []
        for wb in draws(Y, B):
            g = {(ws, a): oam(eoah(nest(rs, r6s, s6s, 1., ws, wb), a, wb, None), Y, wb) for ws in (.1, .3) for a in (0., .2, .3)}
            d0 = g[(.3, 0.)] - g[(.1, 0.)]; D0.append(d0)
            for a in (.2, .3):
                Ib[a].append((g[(.3, a)] - g[(.1, a)]) - d0)
        print('F4 S6slot %-18s R1 boot | D(ws .1 -> .3, a 0) %s | I(.2) %s | I(.3) %s' % (nm, st(D0), st(Ib[.2]), st(Ib[.3])), flush=True)
    # additivity: Raptor m .75 and convex p2 (role-matched N1 / N3), R1 and R3p_lo, at a .2 / .25
    p2 = lambda U: U ** 2
    for nm, rs, r6s, s6s in (('N1', 'R6', 'D5', 's1044'), ('N3', 'R6', 's1044', 's1045')):
        for sn in ('R1', 'R3p_lo'):
            w = W[sn]; out = []
            for a in (.2, .25):
                v = {(m, h): oam(eoah(c4nest(F[rs], F[r6s], F[s6s], m, w), a, w, hf, True), Y, w) for m in (1., .75) for h, hf in (('lin', None), ('p2', p2))}
                out.append('a %.2f pw: m.75 %+.5f p2 %+.5f both %+.5f (sum %+.5f)' % (a, v[(.75, 'lin')] - v[(1., 'lin')], v[(1., 'p2')] - v[(1., 'lin')],
                                                                                    v[(.75, 'p2')] - v[(1., 'lin')], v[(.75, 'lin')] + v[(1., 'p2')] - 2 * v[(1., 'lin')]))
            print('F4 additivity %-4s %-6s | ' % (nm, sn) + ' | '.join(out), flush=True)


def job_f5():
    """Public-size read noise of a convex twin: 390-knee draws (multinomial on R1 weights, i.e. a gold74-joint population), paired
    deltas of the built candidates vs their linear twins / the in-flight shapes. OA-mean and macro (x 3/12)."""
    p2 = lambda U: U ** 2
    SCH = sys.argv[3] if len(sys.argv) > 3 else 'R1'; pr = W[SCH] / W[SCH].sum()
    for cn, f in (('N1 R6|D5|s1044', lambda w: c4nest(F['R6'], F['D5'], F['s1044'], 1., w)),
                  ('N3 R6|s1044|s1045', lambda w: c4nest(F['R6'], F['s1044'], F['s1045'], 1., w)),
                  ('N5 fleet', lambda w: c4nest(F['s1044'], F['D5'], F['s1045'], 1., w)),
                  ('FLAT3', lambda w: (WP(F['D5'], w) + WP(F['s1044'], w) + WP(F['s1045'], w)) / 3)):
        D = {k: [] for k in ('pw25p2-pw30', 'pw25p2-pw25', 'pw20p2-pw20', 'eoa20p2-eoa20', 'pw30-eoa20', 'pw20-eoa20')}
        for _ in range(B):
            ix = rng.choice(n, 390, p=pr); Fb = f(np.ones(n))[ix]; Yb = Y[ix]; one = np.ones(390)
            v = {'pw25p2': oam(eoah(Fb, .25, one, p2, True), Yb, one), 'pw20p2': oam(eoah(Fb, .2, one, p2, True), Yb, one),
                 'eoa20p2': oam(eoah(Fb, .2, one, p2, False), Yb, one), 'pw30': oam(eoah(Fb, .3, one, None, True), Yb, one),
                 'pw25': oam(eoah(Fb, .25, one, None, True), Yb, one), 'pw20': oam(eoah(Fb, .2, one, None, True), Yb, one),
                 'eoa20': oam(eoah(Fb, .2, one, None, False), Yb, one)}
            for k in D:
                x, y = k.split('-'); D[k].append(v[x] - v[y])
        print('F5 n390 %-18s %s population, %d draws, macro units (OA-mean x 3/12) | ' % (cn, SCH, B) +
              ' | '.join('%s %+.5f sd %.5f P>0 %.2f' % (k, np.mean(d) * .25, np.std(d) * .25, np.mean(np.array(d) > 0)) for k, d in D.items()), flush=True)


if JOB == 'F5':
    job_f5()
elif JOB == 'F1':
    job_f1()
elif JOB == 'F4':
    job_f4()
elif JOB == 'F2':
    job_f2()
elif JOB == 'F3':
    job_f3()
print('DONE %s %.0f s' % (JOB, time.time() - T0), flush=True)
