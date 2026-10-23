# box_sink.py (10-21 ~10:05 IST, Claude, SINK track). Box CPU only, read-only, piped via ssh stdin into python -I:
#   ssh training@192.168.10.30 'OMP_NUM_THREADS=4 nice -n 5 ~/miniconda3/envs/rsna/bin/python -I - B' < box_sink.py
# Prints AGGREGATES ONLY (means / sd / P over weighted full-size resamples of macro AUC deltas). Never writes a file, never prints an
# id, a row or a per-knee value. OAI-free members: R6 + S6fall served on OAI (~/r5_oai), fleet D5 / s1044 / s1045.
# Question: on each plausible A22 winner base (pw30 / pw25 / spl = pw25 + LOA <- MM .05; swide / salt equal spl / pw25 here, no MCL
# on OAI), do the OA-touching near-miss extras (CVX = convex E-OA p 2 at the base dose, CVXd = p 2 at base dose - .05, SUBH) add,
# and how do they interact with each other and with the INFO extras (S6up, RPF, BOR, C6p = LM <- MM .05, C14 = LOA += .1 LM, E1)?
# Labels: OA trio P_FT (MOAKS cartilage full-thickness >= 1 in the compartment) or P_AREA (area >= 2); MM / LM = T123 >= 1.
# Regimes (5-label IPF on 10 pairwise 2x2 tables, OAI marginals): g74 = every pair at gold74 log OR (+.5);
# pub = trio pairs shifted -0.83 sd (public gap-matched regime, CLAIMED night/recal.md), meniscus pairs at gold74; raw = OAI (P_FT).
# The gold74 2x2 counts below are local aggregates (same as stack/box_stack.py).
import os, sys, itertools, time
import numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.optimize import brentq
T0 = time.time()
B = int(sys.argv[1]) if len(sys.argv) > 1 else 40
H = os.path.expanduser('~/oai_build'); R5 = os.path.expanduser('~/r5_oai')
C5 = ['Medial OA', 'Lateral OA', 'PF OA', 'Medial Meniscus', 'Lateral Meniscus']
OA = C5[:3]
COMPM = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'], 'PF OA': ['mp', 'lp', 'amf', 'alf']}
G2 = {(0, 1): [45, 5, 12, 9], (0, 2): [36, 14, 6, 16], (0, 3): [33, 16, 4, 18], (0, 4): [36, 14, 7, 15], (1, 2): [37, 20, 5, 9],
      (1, 3): [30, 26, 7, 7], (1, 4): [39, 18, 3, 11], (2, 3): [26, 16, 12, 18], (2, 4): [27, 15, 17, 14], (3, 4): [25, 13, 19, 16]}
TRIO = {(0, 1), (0, 2), (1, 2)}
C4W = np.array([.5, .2, .5, .5, .4])
WPW = {(0, 1): .876, (0, 2): .708, (1, 2): 1.416}
rng = np.random.default_rng(2110)
pct = lambda v: rankdata(v) / len(v)


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def or_cells(p1, p2, lor):
    lo = max(0., p1 + p2 - 1) + 1e-9; hi = min(p1, p2) - 1e-9
    g = lambda p11: np.log(p11) + np.log(1 - p1 - p2 + p11) - np.log(p1 - p11) - np.log(p2 - p11) - lor
    p11 = brentq(g, lo, hi)
    return np.array([[1 - p1 - p2 + p11, p2 - p11], [p1 - p11, p11]])


def weights(Y, sh_t):
    n = len(Y); key = (Y * (2 ** np.arange(4, -1, -1))).sum(1).astype(int); p0 = np.bincount(key, minlength=32) / n
    if sh_t is None:
        return np.ones(n)
    cells = np.array([[(c >> (4 - i)) & 1 for i in range(5)] for c in range(32)]); marg = Y.mean(0); q = p0.copy(); tg = {}
    for (i, j), c in G2.items():
        c = np.array(c, float) + .5; s = sh_t if (i, j) in TRIO else 0.
        tg[(i, j)] = or_cells(marg[i], marg[j], np.log(c[3] * c[0] / (c[1] * c[2])) + s * np.sqrt((1 / c).sum()))
    for _ in range(400):
        for (i, j), t in tg.items():
            for a in (0, 1):
                for b in (0, 1):
                    m = (cells[:, i] == a) & (cells[:, j] == b); s = q[m].sum()
                    if s > 0:
                        q[m] *= t[a, b] / s
        q /= q.sum()
    return np.where(p0 > 0, q / np.maximum(p0, 1e-12), 0)[key]


def rdp(p):
    d = pd.read_csv(p, dtype={'StudyInstanceUID': str})
    d = d[d['StudyInstanceUID'].str.startswith('oai_')].drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
    return d[C5].astype(float)


MEM = {'R6': rdp(R5 + '/run_r6/worker0/predictions.csv'), 'S6': rdp(R5 + '/s6_r5/out/predictions.csv'),
       'D5': rdp(H + '/teacher/teacher_D5_mean_v00.csv'), 's1044': rdp(H + '/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv'),
       's1045': rdp(H + '/teacher2/f0_s1045_ttr288frac75__raw__A__gpu.csv')}
T = pd.read_csv(H + '/oai_labels_T123.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
S = pd.read_csv(H + '/oai_label_sources.csv', dtype={'kid': str}).set_index('kid')


def labels(proxy):
    Yd = pd.DataFrame(index=T.index)
    for c in OA:
        cc = S.reindex(T.index)[['m_cm_' + r for r in COMPM[c]]].astype(float).values
        area = np.floor(cc + 1e-6); ft = np.round((cc - area) * 10); fin = np.isfinite(cc)
        pos = ((ft >= 1) & fin) if proxy == 'P_FT' else ((area >= 2) & fin)
        y = pos.any(1).astype(float); y[~fin.any(1)] = np.nan; Yd[c] = y
    for c in ('Medial Meniscus', 'Lateral Meniscus'):
        Yd[c] = np.where(T[c].notna(), (T[c] >= 1 - 1e-6).astype(float), np.nan)
    return Yd.dropna()


LOMM = (1, 3, .05)
ROOT = dict(a=.3, mode='pw', p=1., e1=1., ws6=.2, wr_pf=.5, sub='lin', lines=())
BASES = {'pw30': dict(), 'pw25': dict(a=.25), 'spl': dict(a=.25, lines=(LOMM,))}
EXTRAS = {'CVX': dict(p=2.), 'CVXd': dict(p=2., da=-.05), 'SUBH': dict(sub='high'), 'S6up': dict(ws6=.3), 'RPF': dict(wr_pf=.4),
          'E1': dict(e1=2.), 'BOR': dict(lines=((0, 3, -.1), (1, 4, -.1))), 'C6p': dict(lines=((4, 3, .05),)), 'C14': dict(lines=((1, 4, -.1),))}
PAIRS = [('CVX', 'SUBH'), ('CVXd', 'SUBH'), ('CVX', 'S6up'), ('CVX', 'RPF'), ('CVX', 'C14'), ('CVX', 'BOR'), ('CVX', 'E1'),
         ('CVXd', 'S6up'), ('CVXd', 'RPF'), ('SUBH', 'S6up'), ('SUBH', 'RPF'), ('SUBH', 'C14')]


def mk(base, extras=()):
    c = dict(ROOT); c.update(BASES[base]); c['lines'] = tuple(c['lines'])
    for x in extras:
        d = dict(EXTRAS[x]); da = d.pop('da', 0.); ln = d.pop('lines', ())
        c.update(d); c['a'] = round(c['a'] + da, 6); c['lines'] = c['lines'] + tuple(ln)
    return c


VAR = {'c4': dict(ROOT, a=0.), 'eoa10': dict(ROOT, a=.1, mode='sym'), 'eoa20': dict(ROOT, a=.2, mode='sym'),
       'eoa30': dict(ROOT, a=.3, mode='sym'), 'pw20': dict(ROOT, a=.2)}
for b in BASES:
    VAR[b] = mk(b)
    for x in EXTRAS:
        VAR[b + '+' + x] = mk(b, (x,))
    for x, y in PAIRS:
        VAR[b + '+' + x + '+' + y] = mk(b, (x, y))


def pipeline(R, R6s, S6s, cfg):
    e1 = cfg['e1']; Pw = lambda v: pct(v) ** e1
    wr = C4W.copy(); wr[2] = cfg['wr_pf']
    V = np.column_stack([wr[j] * Pw(R[:, j]) + (1 - wr[j]) * Pw(R6s[:, j]) for j in range(5)]).astype(np.float32).astype(np.float64)
    F = np.column_stack([(1 - (cfg['ws6'] if j < 3 else .2)) * Pw(V[:, j]) + (cfg['ws6'] if j < 3 else .2) * Pw(S6s[:, j])
                         for j in range(5)]).astype(np.float32).astype(np.float64)
    out = F.copy(); a = cfg['a']
    if a:
        U = [pct(F[:, t]) for t in range(3)]
        Hs = [pct(2 / 3 * pct(R6s[:, t]) + 1 / 3 * pct(S6s[:, t])) for t in range(3)] if cfg['sub'] == 'high' else U
        Ss = Hs if cfg['p'] == 1 else [h ** cfg['p'] for h in Hs]
        for t in range(3):
            o = [k for k in range(3) if k != t]
            w = [WPW[tuple(sorted((t, k)))] for k in o] if cfg['mode'] == 'pw' else [1., 1.]
            out[:, t] = pct(U[t] - a / 2. * (w[0] * Ss[o[0]] + w[1] * Ss[o[1]]))
    for t, o, b in cfg['lines']:
        out[:, t] = pct(pct(out[:, t]) - b * pct(out[:, o]))
    return out


NESTS = {'N1 R6|D5|s1044': ('R6', 'D5', 's1044'), 'N3 R6|s1044|s1045': ('R6', 's1044', 's1045'), 'N2 S6|D5|s1045': ('S6', 'D5', 's1045'),
         'N4 R6|s1045|D5': ('R6', 's1045', 'D5'), 'c4D5 D5|R6|S6': ('D5', 'R6', 'S6')}
PUB = {'eoa10': 5.9, 'eoa20': 10.3, 'eoa30': 9.6, 'pw20': 10.4, 'pw30': 11.3}
for proxy in ('P_FT', 'P_AREA'):
    Yd = labels(proxy); ids = Yd.index
    for k in MEM:
        ids = ids.intersection(MEM[k].dropna().index)
    ids = sorted(ids); Y = Yd.loc[ids, C5].values.astype(int); n = len(ids)
    Fm = {k: MEM[k].loc[ids, C5].values for k in MEM}
    print('PROXY %s | n %d | prevalence %s' % (proxy, n, np.round(Y.mean(0), 3).tolist()), flush=True)
    REG = [('g74', 0.), ('pub', -0.83)] + ([('raw', None)] if proxy == 'P_FT' else [])
    for rn, sh in REG:
        w = weights(Y, sh); p = w / w.sum(); ess = 1 / (p ** 2).sum()
        print(' REGIME %s ess %.0f' % (rn, ess), flush=True)
        nb = B if rn != 'raw' else max(B // 2, 10)
        for nn, (ra, r6, s6) in NESTS.items():
            acc = {k: [] for k in VAR}
            for _ in range(nb):
                ii = rng.choice(n, n, p=p); y = Y[ii]
                if (y.sum(0) < 10).any():
                    continue
                outs = {k: pipeline(Fm[ra][ii], Fm[r6][ii], Fm[s6][ii], c) for k, c in VAR.items()}
                base = [auc(outs['pw30'][:, j], y[:, j]) for j in range(5)]
                for k, o in outs.items():
                    acc[k].append(sum(auc(o[:, j], y[:, j]) - base[j] for j in range(5) if not np.array_equal(o[:, j], outs['pw30'][:, j])) / 12 * 1e4)
            A = {k: np.array(v) for k, v in acc.items()}
            st = lambda x: '%+.2f (sd %.2f P %.2f)' % (x.mean(), x.std(), (x > 0).mean())
            lad = {k: (A[k] - A['c4']).mean() for k in PUB if k != 'pw30'}; lad['pw30'] = -A['c4'].mean()
            print('  %s %s %s | n_boot %d | ladder vs c4 %s | bases vs pw30: pw25 %s | spl %s' % (proxy, rn, nn, len(A['pw30']),
                  ' '.join('%s %+.2f' % (k, v) for k, v in lad.items()), st(A['pw25']), st(A['spl'])), flush=True)
            for b in BASES:
                print('  %s %s %s | marg on %s | %s' % (proxy, rn, nn, b, ' | '.join('%s %s' % (x, st(A[b + '+' + x] - A[b])) for x in EXTRAS)), flush=True)
                print('  %s %s %s | I on %s | %s' % (proxy, rn, nn, b, ' | '.join('%s*%s %s' % (x, y, st(A[b + '+' + x + '+' + y] - A[b + '+' + x] - A[b + '+' + y] + A[b]))
                                                                            for x, y in PAIRS)), flush=True)
print('BOX_SINK DONE %.0f s' % (time.time() - T0), flush=True)
