# stack_rak2.py (10-20 ~23:15 IST, Claude, STACK inventory). RESEARCH ONLY, local CPU, NO OAI. Follow-up to stack_local.py.
# (1) Raked-OOF ruler WITH gold74 target uncertainty: each draw = Dirichlet(gold74 joint + .5) target x row bootstrap, for the
#     column-compatible candidates (pw25, lomm05, MCLmm, E1) and their pair / triple stacks, in the 3 c4-shaped OOF nests, plus the
#     E-OA ladder in the same draw (draw-level public / ruler calibration ratio).
# (2) MCL <- X placebo partners (a .1 on pw30), each raked on its own (MCL, X) gold74 2x2: is MM special, and does the gain track
#     the report-minus-expert coupling gap (the mechanism)?
# (3) Label-free co-firing rho_ns(MCL, MM) and (LOA, MM) of the pw30 output: c4 on gold58 vs the nests on train.
# Writes stack/stack_rak2.log. Aggregates only.
import sys, itertools, time
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack')
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
import load as LD
T0 = time.time()
src = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/stack_local.py', encoding='utf-8').read()
ns_ = {}; exec(src.split('summary = {}')[0].replace("log = open(OUT + '.log', 'w')", "log = open(OUT + '_rak2_tmp.log', 'w')"), ns_)
pipeline, wauc, rake, BASE = ns_['pipeline'], ns_['wauc'], ns_['rake'], ns_['BASE']
OUT = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/stack_rak2.log'
log = open(OUT, 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
L = LD.L; J = {c: i for i, c in enumerate(L)}; G = LD.G
OA = ['Medial OA', 'Lateral OA', 'PF OA']; OAX = OA + ['Medial Meniscus', 'Lateral Meniscus']
pct = lambda v: rankdata(v) / len(v); nsc = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
PUB = {'eoa10': 5.9, 'eoa20': 10.3, 'eoa30': 9.6, 'pw20': 10.4, 'pw30': 11.3}
C = {'pw25': dict(a=.25), 'lomm05': dict(loamm=.05), 'MCLmm': dict(mclmm=.1), 'E1': dict(e1=2.)}
STK = {'pw25+lomm05': ['pw25', 'lomm05'], 'lomm05+MCLmm': ['lomm05', 'MCLmm'], 'pw25+MCLmm': ['pw25', 'MCLmm'],
       'pw25+lomm05+MCLmm': ['pw25', 'lomm05', 'MCLmm'], 'E1+lomm05+MCLmm': ['E1', 'lomm05', 'MCLmm'],
       'E1+pw25+lomm05+MCLmm': ['E1', 'pw25', 'lomm05', 'MCLmm'], 'pw25+E1': ['pw25', 'E1'], 'lomm05+E1': ['lomm05', 'E1']}
LAD = {'c4': dict(a=0.), 'eoa10': dict(a=.1, mode='sym'), 'eoa20': dict(a=.2, mode='sym'), 'eoa30': dict(a=.3, mode='sym'), 'pw20': dict(a=.2)}
VAR = {'pw30': dict(BASE)}
for k, d in list(C.items()) + list(LAD.items()):
    c = dict(BASE); c.update(d); VAR[k] = c
for k, parts in STK.items():
    c = dict(BASE)
    for p_ in parts: c.update(C[p_])
    VAR[k] = c

g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}); Y74 = g[L].values.astype(float); allg = set(g.StudyInstanceUID)
r7 = LD.r7(); s44, _ = LD.s6seed(1044); s45, _ = LD.s6seed(1045); lv2 = LD.labels('v2')
tr = [i for i in r7.index if i not in allg and i in lv2.index]; tr3 = [i for i in tr if i in s44.index and i in s45.index]
X = {'R7': r7.loc[tr3, ['p|' + c for c in L]].values.astype(float), 's1044': s44.loc[tr3, L].values.astype(float),
     's1045': s45.loc[tr3, L].values.astype(float)}
Y = lv2.loc[tr3, L].values.astype(float); n = len(tr3); YF = np.nan_to_num(Y, nan=0.)


def table(cols):
    jc = [J[c] for c in cols]; ok = ~np.isnan(Y74[:, jc]).any(1); T = np.zeros((2,) * len(cols))
    for r in Y74[ok][:, jc].astype(int): T[tuple(r)] += 1
    return T


def wts(cols, m, Tp):
    jc = [J[c] for c in cols]; Ys = Y[m][:, jc].astype(int); obs = np.zeros((2,) * len(cols))
    for r in Ys: obs[tuple(r)] += 1
    obs /= obs.sum(); tgt = rake(Tp, Ys.mean(0), it=150); w = (tgt / obs)[tuple(Ys.T)]
    return w / w.mean()


COLS = {c: (OAX if c in OA else ['MCL', 'Medial Meniscus']) for c in OA + ['MCL']}
MASK = {c: ~np.isnan(Y[:, [J[x] for x in COLS[c]]]).any(1) for c in COLS}
TAB = {c: table(COLS[c]) for c in COLS}
NESTS = {'N_A R7|s1044|s1045': ('R7', 's1044', 's1045'), 'N_B s1044|R7|s1045': ('s1044', 'R7', 's1045'),
         'N_C s1045|s1044|R7': ('s1045', 's1044', 'R7')}
NB = 200
rng = np.random.default_rng(31)
g58 = g[g.source == 'original_gold58'].reset_index(drop=True); ids = g58.StudyInstanceUID.tolist()
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values.astype(float)
M58 = dict(rap=[(.75, np.load(G + '/gold58_r384_kernelserve_cpu.npy')), (.25, np.load(G + '/gold58_r224_kernelserve.npy'))],
           r6=rd('R6_gold58_serve.csv'), s6=rd('n42gpu_gold58.csv'))
o58 = pipeline(M58, dict(BASE))
rho = lambda F, a, b: float(np.corrcoef(nsc(F[:, J[a]]), nsc(F[:, J[b]]))[0, 1])
P('(3) label-free co-firing on the pw30 output: c4 gold58 MCL~MM %.3f | LOA~MM %.3f | MCL~MOA %.3f' % (
    rho(o58, 'MCL', 'Medial Meniscus'), rho(o58, 'Lateral OA', 'Medial Meniscus'), rho(o58, 'MCL', 'Medial OA')))
SUMM = {}
for nn, (ra, r6, s6) in NESTS.items():
    M = dict(rap=[(1., X[ra])], r6=X[r6], s6=X[s6]); outs = {k: pipeline(M, c) for k, c in VAR.items()}
    P('\n==== %s | train co-firing pw30 output: MCL~MM %.3f | LOA~MM %.3f | MCL~MOA %.3f' % (nn, rho(outs['pw30'], 'MCL', 'Medial Meniscus'),
      rho(outs['pw30'], 'Lateral OA', 'Medial Meniscus'), rho(outs['pw30'], 'MCL', 'Medial OA')))
    base = outs['pw30']
    ch = {k: [j for j in range(12) if not np.array_equal(outs[k][:, j], base[:, j])] for k in outs}
    acc = {k: [] for k in outs}
    for b in range(NB):
        ii = rng.integers(0, n, n); W = {}
        TD = {}
        for c in COLS:
            key = tuple(COLS[c])
            if key not in TD:
                TD[key] = rng.dirichlet((TAB[c] + .5).ravel()).reshape(TAB[c].shape)
            wf = np.full(n, np.nan); wf[MASK[c]] = wts(COLS[c], MASK[c], TD[key]); W[c] = wf
        bc = {}
        for k in outs:
            d = 0.
            for j in ch[k]:
                c = L[j]
                if c in W:
                    sel = ii[MASK[c][ii]]; w = W[c][sel]
                else:
                    sel = ii[~np.isnan(Y[ii, j])]; w = np.ones(len(sel))
                if j not in bc:
                    bc[j] = wauc(base[sel, j], YF[sel, j], w)
                d += wauc(outs[k][sel, j], YF[sel, j], w) - bc[j]
            acc[k].append(d / 12 * 1e4)
    A = {k: np.array(v) for k, v in acc.items()}
    lad = {k: A[k] - A['c4'] for k in PUB}   # pw30 vs c4 = -(c4 delta)
    lad['pw30'] = -A['c4']
    ratio = np.array([PUB['pw30'] / v if v > 0 else np.nan for v in lad['pw30']])
    st = lambda x: '%+.2f [%+.2f, %+.2f] P>0 %.2f' % (x.mean(), np.percentile(x, 10), np.percentile(x, 90), (x > 0).mean())
    P('  ladder vs c4 (mean): ' + ' | '.join('%s %+.2f (pub %.1f)' % (k, lad[k].mean(), PUB[k]) for k in PUB) +
      ' | draw-level pub/ruler at pw30: median %.2f [p10 %.2f, p90 %.2f]' % (np.nanmedian(ratio), np.nanpercentile(ratio, 10), np.nanpercentile(ratio, 90)))
    for k in list(C) + list(STK):
        P('  %-22s %s' % (k, st(A[k])))
    for k, parts in STK.items():
        I = A[k] - sum(A[p_] for p_ in parts)
        P('  I(%-20s) %s' % (k, st(I)))
    SUMM[nn] = {k: (float(A[k].mean()), float((A[k] > 0).mean())) for k in A}

# (2) MCL <- X placebo partners
P('\n(2) MCL <- X a .1 on pw30, each raked on its own (MCL, X) gold74 2x2 (+.5), point estimates, label units (MCL AUC delta)')
def lor(T):
    c = T.ravel() + .5; return np.log(c[0] * c[3] / (c[1] * c[2]))
for nn, (ra, r6, s6) in NESTS.items():
    M = dict(rap=[(1., X[ra])], r6=X[r6], s6=X[s6]); B0 = pipeline(M, dict(BASE)); jm = J['MCL']; rows = []
    for Xp in [c for c in L if c != 'MCL']:
        cols = ['MCL', Xp]; m = ~np.isnan(Y[:, [J[x] for x in cols]]).any(1); T = table(cols)
        w = wts(cols, m, (T + .5) / (T + .5).sum())
        G1 = B0.copy(); G1[:, jm] = pct(pct(B0[:, jm]) - .1 * pct(B0[:, J[Xp]]))
        dr = wauc(G1[m, jm], YF[m, jm], w) - wauc(B0[m, jm], YF[m, jm], w)
        d0 = wauc(G1[m, jm], YF[m, jm], np.ones(m.sum())) - wauc(B0[m, jm], YF[m, jm], np.ones(m.sum()))
        Ys = Y[m][:, [J[x] for x in cols]].astype(int); Tr = np.zeros((2, 2))
        for r in Ys: Tr[tuple(r)] += 1
        rows.append((Xp, dr, d0, lor(Tr) - lor(T)))
    rows.sort(key=lambda r: -r[1])
    P('  %s: ' % nn + ' | '.join('%s raked %+.4f raw %+.4f (lorgap %+.2f)' % (x[:5], a, b, gp) for x, a, b, gp in rows))
    arr = np.array([[r[1], r[3]] for r in rows])
    P('    MM rank %d of 11 | corr(raked gain, report-minus-expert lor gap) %.2f' % (1 + [r[0] for r in rows].index('Medial Meniscus'),
      np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]))
P('STACK_RAK2 DONE %.0f s' % (time.time() - T0))
