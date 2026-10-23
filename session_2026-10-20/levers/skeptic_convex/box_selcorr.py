# skeptic_convex/box_selcorr.py (10-20 skeptic of convex E-OA). Read-only, CPU, AGGREGATES ONLY (means, sd, P, counts).
# Never writes a file, never prints an id, a row or a per-knee value. Piped via ssh stdin:
#   ssh training@192.168.10.30 'nice -n 5 ~/miniconda3/envs/rsna/bin/python -I - S1 25 R1 P_FT' < box_selcorr.py
# Question: the convex family was picked as the best of ~13 output-transform families on ALL 1,704 OAI knees, and the
# proposer's "held-out" test only held out the choice of a. Here the WHOLE search (family + a) is run on one stratified half
# and the winner is scored on the other half (selection-corrected). Also a fixed-p2 held-out row for comparison.
import os, sys, time
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
JOB = sys.argv[1] if len(sys.argv) > 1 else 'S1'
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
SCH = sys.argv[3] if len(sys.argv) > 3 else 'R1'
PX = sys.argv[4] if len(sys.argv) > 4 else 'P_FT'
rng = np.random.default_rng(20201)


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


def scheme_w(Y):
    if SCH == 'R1':
        return r1w(cells_of(Y), (G74 + .5) / (G74 + .5).sum())
    return r3p(Y, {'R3p': 0., 'R3p_lo': -1., 'R3p_hi': 1.}[SCH])


def eoa(Fb, a, w, space='pct', sub=None, gate=None, h=None):
    U = WP(Fb, w)
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


def c4nest(RS, R6s, S6s, w):
    V = WR * WP(RS, w) + (1 - WR) * WP(R6s, w)
    return .8 * WP(V, w) + .2 * WP(S6s, w)


CFG = [('pair R6|D5', 'pair', 'R6', 'D5', None), ('pair S6|D5', 'pair', 'S6', 'D5', None),
       ('N1 R6|D5|s1044', 'nest', 'R6', 'D5', 's1044'), ('N2 S6|D5|s1045', 'nest', 'S6', 'D5', 's1045'),
       ('N3 R6|s1044|s1045', 'nest', 'R6', 's1044', 's1045'), ('N5 s1044|D5|s1045', 'nest', 's1044', 'D5', 's1045')]


def build(X, c, w):
    _, kind, lo, hi, x = c
    if kind == 'pair':
        return .4 * WP(X[lo], w) + .6 * WP(X[hi], w)
    return c4nest(X[lo], X[hi], X[x], w)


def gate(X, c, w, sign):
    _, kind, lo, hi, x = c
    UL = WP(X[lo], w); UH = WP(X[hi], w)
    g = np.zeros((len(UL), 3))
    for t in range(3):
        o = [k for k in range(3) if k != t]
        g[:, t] = np.abs(UL[:, o] - UH[:, o]).mean(1)
    r = wpct(g.mean(1), w)
    return (1.5 - r) if sign > 0 else (.5 + r)


def highonly(X, c, w, a):
    _, kind, lo, hi, x = c
    if kind == 'pair':
        return .4 * WP(X[lo], w) + .6 * WP(eoa(X[hi], a, w), w)
    return c4nest(X[lo], eoa(X[hi], a, w), eoa(X[x], a, w), w)


FAM = {
    'lin': lambda X, c, w, a: eoa(build(X, c, w), a, w),
    'p1.5': lambda X, c, w, a: eoa(build(X, c, w), a, w, h=lambda U: U ** 1.5),
    'p2': lambda X, c, w, a: eoa(build(X, c, w), a, w, h=lambda U: U ** 2),
    'p3': lambda X, c, w, a: eoa(build(X, c, w), a, w, h=lambda U: U ** 3),
    'hinge.3': lambda X, c, w, a: eoa(build(X, c, w), a, w, h=lambda U: np.maximum(U - .3, 0) / .7),
    'hinge.5': lambda X, c, w, a: eoa(build(X, c, w), a, w, h=lambda U: np.maximum(U - .5, 0) / .5),
    'sqrt': lambda X, c, w, a: eoa(build(X, c, w), a, w, h=np.sqrt),
    'probit': lambda X, c, w, a: eoa(build(X, c, w), a, w, space='probit'),
    'gate_agree': lambda X, c, w, a: eoa(build(X, c, w), a, w, gate=gate(X, c, w, +1)),
    'gate_dis': lambda X, c, w, a: eoa(build(X, c, w), a, w, gate=gate(X, c, w, -1)),
    'sub_high': lambda X, c, w, a: eoa(build(X, c, w), a, w, sub=X[c[3]]),
    'sub_low': lambda X, c, w, a: eoa(build(X, c, w), a, w, sub=X[c[2]]),
    'highonly': lambda X, c, w, a: highonly(X, c, w, a),
}
CONVEX = {'p1.5', 'p2', 'p3', 'hinge.3', 'hinge.5'}
AG = np.round(np.arange(0, .501, .05), 3)

P = members()
YD = labels(PX)
ids = YD.index
for k in P:
    ids = ids.intersection(P[k].dropna().index)
ids = sorted(ids)
F = {k: P[k].loc[ids].values.astype(np.float64) for k in P}
Y = YD.loc[ids].values.astype(int)
n = len(ids)
print('JOB %s | n %d | proxy %s | scheme %s | splits %d x 2 | families %d | a grid %s' % (JOB, n, PX, SCH, NS, len(FAM), AG.tolist()), flush=True)


def envs(X, Yh, w):
    """per family: per-config best a and env (gain over a 0) on this half."""
    out = {f: [] for f in FAM}
    for c in CFG:
        base = oam(build(X, c, w), Yh, w)
        for f, fn in FAM.items():
            cur = np.array([oam(fn(X, c, w, a), Yh, w) - base if a else 0. for a in AG])
            j = int(np.argmax(cur)); out[f].append((AG[j], cur[j]))
    return out


def score(X, Yh, w, f, alist):
    """per config: gain of family f at the given a (per config) over the base."""
    g = []
    for c, a in zip(CFG, alist):
        base = oam(build(X, c, w), Yh, w)
        g.append(oam(FAM[f](X, c, w, a), Yh, w) - base if a else 0.)
    return np.array(g)


cell = cells_of(Y)
sel_count = {f: 0 for f in FAM}
ins_adv, ho_adv, ho_p2, ho_p2_ins, ho_sel_is_cvx, rank_p2 = [], [], [], [], [], []
for sp in range(NS):
    half = np.zeros(n, bool)
    for c8 in range(8):
        ix = np.where(cell == c8)[0]; rng.shuffle(ix); half[ix[:len(ix) // 2]] = True
    for tr, te in ((half, ~half), (~half, half)):
        Xtr = {k: F[k][tr] for k in F}; Xte = {k: F[k][te] for k in F}
        wtr = scheme_w(Y[tr]); wte = scheme_w(Y[te])
        E = envs(Xtr, Y[tr], wtr)
        m = {f: np.mean([e for _, e in E[f]]) for f in FAM}
        adv = {f: m[f] - m['lin'] for f in FAM}
        fstar = max(FAM, key=lambda f: adv[f]); sel_count[fstar] += 1
        order = sorted(FAM, key=lambda f: -adv[f]); rank_p2.append(order.index('p2') + 1)
        ins_adv.append(adv[fstar])
        lin_te = score(Xte, Y[te], wte, 'lin', [a for a, _ in E['lin']])
        sel_te = score(Xte, Y[te], wte, fstar, [a for a, _ in E[fstar]])
        p2_te = score(Xte, Y[te], wte, 'p2', [a for a, _ in E['p2']])
        ho_adv.append(float(np.mean(sel_te - lin_te))); ho_p2.append(float(np.mean(p2_te - lin_te))); ho_p2_ins.append(adv['p2'])
        ho_sel_is_cvx.append(fstar in CONVEX)
    print('  split %d done %.0f s' % (sp + 1, time.time() - T0), flush=True)


def st(x):
    x = np.asarray(x, float); return '%+.5f (sd %.5f P>0 %.2f)' % (x.mean(), x.std(), (x > 0).mean())


print('S1 %s %s | selected family counts over %d half-searches: %s' % (PX, SCH, 2 * NS, {f: c for f, c in sel_count.items() if c}), flush=True)
print('S1 %s %s | P(winner is a convex-type shape) %.2f | rank of p2 among %d families: mean %.1f, median %.0f, P(rank 1) %.2f' % (
    PX, SCH, np.mean(ho_sel_is_cvx), len(FAM), np.mean(rank_p2), np.median(rank_p2), np.mean(np.array(rank_p2) == 1)), flush=True)
print('S1 %s %s | OA-mean, mean over 6 configs | winner in-sample adv vs lin %s | winner HELD-OUT adv vs lin %s' % (PX, SCH, st(ins_adv), st(ho_adv)), flush=True)
print('S1 %s %s | fixed p2 (family pre-chosen, a held out) | in-sample adv %s | held-out adv %s' % (PX, SCH, st(ho_p2_ins), st(ho_p2)), flush=True)
print('S1 %s %s | macro units (x 3/12): winner held-out %+.2fe-4 | fixed p2 held-out %+.2fe-4 | shrink winner in->out %.2f' % (
    PX, SCH, np.mean(ho_adv) * 2.5e3, np.mean(ho_p2) * 2.5e3, np.mean(ho_adv) / max(np.mean(ins_adv), 1e-9)), flush=True)
print('DONE %.0f s' % (time.time() - T0), flush=True)
