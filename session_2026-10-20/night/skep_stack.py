# skep_stack.py (10-20 night, Claude, SKEPTIC on the recal probe "stack C6' + C12' on pw30"). RESEARCH ONLY, local CPU, NO OAI rows
# (the only OAI numbers used are the aggregate curves already on disk in recal_box_gapmatch.log). Writes skep_stack.log.
# Questions:
#  A. Is LM <- MM "the same kind" as E-OA? E-OA works because reports couple the OA trio MORE than experts. Compare report-v2 vs
#     expert coupling for MM-LM and LOA-MM (train report n ~2.9k; gold74 expert; PAIRED report vs expert on the gold74 knees),
#     with the trio pairs as the positive control.
#  B. Is c4's gold58 MM-LM co-firing (.312) representative of RSNA? R7 OOF rho on train (n ~2.9k) vs R7 on gold58 / gold74;
#     c4 rho bootstrap sd on gold58. Report-label gap (R7 rho - q_v2) on train, and the report OOF delta of LM <- MM / LOA <- MM at a .05.
#  C. Jensen: integrate the box a .05 curves (3 proxies, aggregates) over an honest gap distribution instead of reading at the centre.
import sys, re
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
import census_lib as CL
from census_lib import auc, pct, L, tet
import load as LD
from cbase import load
OUTD = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/night/'
log = open(OUTD + 'skep_stack.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()
rng = np.random.default_rng(20261020)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
J = {c: i for i, c in enumerate(L)}
OA = ['Medial OA', 'Lateral OA', 'PF OA']
W_PW = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}


def pw(F, a=.3):
    G = F.copy(); U = {c: pct(F[:, J[c]]) for c in OA}
    for c in OA:
        o = [k for k in OA if k != c]; w = [W_PW[tuple(sorted((c, k), key=OA.index))] for k in o]
        G[:, J[c]] = pct(U[c] - a / 2.0 * (w[0] * U[o[0]] + w[1] * U[o[1]]))
    return G


D = CL.data(); d0 = load()
ids74 = d0['ids74']; ids58 = d0['ids']
lv2 = LD.labels('v2')
Y74 = D['Y74']; Y58 = D['Y58']
R74v2 = lv2.reindex(ids74)[L].values
PAIRS = [('Medial Meniscus', 'Lateral Meniscus'), ('Lateral OA', 'Medial Meniscus'),
         ('Medial OA', 'Lateral OA'), ('Medial OA', 'PF OA'), ('Lateral OA', 'PF OA')]


def tet_ok(a, b):
    m = ~np.isnan(a) & ~np.isnan(b); return tet(a[m].astype(int), b[m].astype(int)), int(m.sum())


def boot_tet(a, b, B=400, idx=None):
    m = ~np.isnan(a) & ~np.isnan(b); a, b = a[m].astype(int), b[m].astype(int); n = len(a)
    return np.array([tet(a[ii], b[ii]) for ii in (rng.integers(0, n, n) for _ in range(B))])


P('== A. label side: report v2 vs expert coupling (tetrachoric; sd = bootstrap)')
for x, y in PAIRS:
    jx, jy = J[x], J[y]
    tr_t, tr_n = tet_ok(D['Yv2'][:, jx], D['Yv2'][:, jy]); tr_b = boot_tet(D['Yv2'][:, jx], D['Yv2'][:, jy], B=200)
    g_t, g_n = tet_ok(Y74[:, jx], Y74[:, jy]); g_b = boot_tet(Y74[:, jx], Y74[:, jy])
    # paired on gold74 knees that have report v2 verdicts for both labels
    m = ~np.isnan(R74v2[:, jx]) & ~np.isnan(R74v2[:, jy])
    rp = tet(R74v2[m, jx].astype(int), R74v2[m, jy].astype(int)); ep = tet(Y74[m, jx].astype(int), Y74[m, jy].astype(int))
    n = int(m.sum()); idx = np.where(m)[0]; dd = []
    for _ in range(400):
        ii = idx[rng.integers(0, n, n)]
        dd.append(tet(R74v2[ii, jx].astype(int), R74v2[ii, jy].astype(int)) - tet(Y74[ii, jx].astype(int), Y74[ii, jy].astype(int)))
    dd = np.array(dd)
    P('%-34s | report train tet %+.3f (sd %.3f, n %d) | expert gold74 tet %+.3f (sd %.3f, n %d) | PAIRED on gold74 (n %d): report %+.3f expert %+.3f, diff %+.3f (sd %.3f, P(report>expert) %.2f)' % (
        x + ' x ' + y, tr_t, tr_b.std(), tr_n, g_t, g_b.std(), g_n, n, rp, ep, rp - ep, dd.std(), (dd > 0).mean()))

P('')
P('== B. model side: normal-score Spearman rho (label-free) and report-label gap')
C4 = D['C4']; B30 = pw(C4); F7 = D['F7']; R58 = D['R7_58']; R74 = D['R7_74']; M3 = D['M3']
def rho(F, a, b): return float(np.corrcoef(ns(F[:, J[a]]), ns(F[:, J[b]]))[0, 1])
def rho_boot(F, a, b, B=1000):
    n = len(F); za, zb = ns(F[:, J[a]]), ns(F[:, J[b]])
    return np.array([np.corrcoef(za[ii], zb[ii])[0, 1] for ii in (rng.integers(0, n, n) for _ in range(B))])
for a, b, base in (('Lateral Meniscus', 'Medial Meniscus', 'c4'), ('Lateral OA', 'Medial Meniscus', 'pw30'), ('Medial OA', 'Lateral OA', 'c4')):
    Fc = C4 if base == 'c4' else B30
    rb = rho_boot(Fc, a, b)
    ok74 = ~np.isnan(R74).any(1); ok58 = ~np.isnan(R58).any(1)
    P('%-34s | c4(%s) gold58 rho %.3f (boot sd %.3f) | R7 OOF: train(non-gold, n %d) %.3f, gold58 %.3f, gold74 %.3f | 3-model mean train(n %d) %.3f' % (
        a + ' ~ ' + b, base, rho(Fc, a, b), rb.std(), len(F7), rho(F7, a, b), rho(R58[ok58], a, b), rho(R74[ok74], a, b), len(M3), rho(M3, a, b)))


def gap_q(F, Y, t, o):
    m = ~np.isnan(Y[:, J[t]]); y = Y[m, J[t]]; zt, zo = ns(F[m, J[t]]), ns(F[m, J[o]])
    return np.corrcoef(zt, zo)[0, 1], np.cov(zo, y)[0, 1] / np.cov(zt, y)[0, 1]


def gap_boot(F, Y, t, o, B=300):
    m = ~np.isnan(Y[:, J[t]]); F = F[m]; Y = Y[m]; n = len(F); out = []
    for _ in range(B):
        ii = rng.integers(0, n, n); r, q = gap_q(F[ii], Y[ii], t, o); out.append(r - q)
    return np.array(out)


P('')
for t, o in (('Lateral Meniscus', 'Medial Meniscus'), ('Lateral OA', 'Medial Meniscus'), ('Medial OA', 'Lateral OA'), ('Lateral OA', 'PF OA')):
    r, q = gap_q(F7, D['Yv2'], t, o); gb = gap_boot(F7, D['Yv2'], t, o)
    r3, q3 = gap_q(M3, D['Y3v2'], t, o)
    r74, q74 = gap_q(R74, Y74, t, o) if not np.isnan(R74).any() else gap_q(R74[~np.isnan(R74).any(1)], Y74[~np.isnan(R74).any(1)], t, o)
    P('%-34s | REPORT labels: R7 train rho %.3f q_v2 %.3f gap %+.3f (sd %.3f) | 3-mean rho %.3f q %.3f gap %+.3f | EXPERT gold74 R7: rho %.3f q %.3f gap %+.3f' % (
        t + ' <- ' + o, r, q, r - q, gb.std(), r3, q3, r3 - q3, r74, q74, r74 - q74))

P('')
P('== B2. report OOF delta of the exact probe lines (decorrelate t <- o at a), R7 on train v2 and 3-model mean v2; paired bootstrap')


def dline(F, Y, t, o, a, B=500):
    m = ~np.isnan(Y[:, J[t]]); F = F[m]; y = Y[m, J[t]]
    g = pct(pct(F[:, J[t]]) - a * pct(F[:, J[o]])); base = F[:, J[t]]
    d = auc(g, y) - auc(base, y); n = len(y); bb = []
    for _ in range(B):
        ii = rng.integers(0, n, n)
        if y[ii].min() == y[ii].max(): continue
        bb.append(auc(g[ii], y[ii]) - auc(base[ii], y[ii]))
    bb = np.array(bb); return d, bb.std(), (bb > 0).mean(), n


for t, o in (('Lateral Meniscus', 'Medial Meniscus'), ('Lateral OA', 'Medial Meniscus'), ('Lateral OA', 'PF OA')):
    for a in (.05, .1):
        F7b = pw(F7) if t == 'Lateral OA' and o != 'PF OA' else F7
        M3b = pw(M3) if t == 'Lateral OA' and o != 'PF OA' else M3
        d1 = dline(F7b, D['Yv2'], t, o, a); d3 = dline(M3b, D['Y3v2'], t, o, a)
        P('%-34s a %.2f %s| R7 v2 d %+.4f (sd %.4f, P>0 %.2f, n %d) | 3-mean v2 d %+.4f (sd %.4f, P>0 %.2f, n %d)' % (
            t + ' <- ' + o, a, '(pw30-applied base) ' if F7b is not F7 else '', *d1, *d3))

P('')
P('== C. Jensen: box a .05 curve (aggregates on disk) integrated over the gap distribution')
txt = open(OUTD + 'recal_box_gapmatch.log').read()
def curve(block_title):
    blk = txt.split('== ' + block_title)[1].split('\n== ')[0]
    pts = []
    for m in re.finditer(r'(D5|s1044|s1045)\s+k ([+-][\d.]+) .*?gap ([+-][\d.]+) \| dAUC_t a-0.10 ([+-][\d.]+) a-0.05 ([+-][\d.]+) a\+0.05 ([+-][\d.]+) a\+0.10 ([+-][\d.]+)', blk):
        pts.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(6)), float(m.group(7))))
    return pts


def interp_vec(pts, G, col=3):
    res = []
    for mdl in ('D5', 's1044', 's1045'):
        pp = sorted([(p[2], p[col]) for p in pts if p[0] == mdl]); x = np.array([p[0] for p in pp]); yv = np.array([p[1] for p in pp])
        sl = (yv[1] - yv[0]) / (x[1] - x[0])
        v = np.interp(G, x, yv); v = np.where(G < x[0], yv[0] + (G - x[0]) * sl, v); v = np.where(G > x[-1], yv[-1], v); res.append(v)
    return np.mean(res, 0), np.array(res)


AMP = .86  # recal's OAI -> public amplitude (gap-matched)
for title, centres in (('Lateral Meniscus <- Medial Meniscus', [('recal Route O kq .43', .207, None), ('Route O kq .9', .117, None), ('gold58 direct', .084, None)]),
                       ('Lateral OA <- Medial Meniscus', [('Route O pw30 base', .110, None), ('gold58 direct pw30', .224, None)])):
    pts = curve(title)
    P('-- %s: a .05 per-label curve, mean of 3 proxies, at gaps:' % title)
    for g in (.0, .05, .08, .10, .12, .15, .19, .25):
        v, per = interp_vec(pts, np.array([g])); P('   gap %.2f -> %+.5f label (%s)' % (g, v[0], ' / '.join('%+.4f' % p[0] for p in per)))
    for nm, c, _ in centres:
        P('   centre %-22s gap %.3f: plug-in %+.5f label = %+.2fe-4 macro x%.2f' % (nm, c, interp_vec(pts, np.array([c]))[0][0], interp_vec(pts, np.array([c]))[0][0] / 12 * 1e4 * AMP, AMP))
    for sd in (.06, .09, .12, .15):
        for nm, c, _ in centres[:2]:
            G = rng.normal(c, sd, 200000); v, _ = interp_vec(pts, G)
            P('   gap ~ N(%.3f, %.2f) [%s]: E gain %+.5f label = %+.2fe-4 macro x%.2f | P(gain>0) %.2f' % (c, sd, nm, v.mean(), v.mean() / 12 * 1e4 * AMP, AMP, (v > 0).mean()))
P('SKEP_STACK DONE')
