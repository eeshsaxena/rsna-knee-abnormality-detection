# stack_mcl.py (10-20 ~23:40 IST, Claude, STACK). RESEARCH ONLY, local CPU, NO OAI. Robustness of MCLmm (MCL = pct(pct(MCL) - .1 pct(MM))
# on pw30) on the raked-OOF ruler: (a) 3-way rakings that also hold a confounder at its gold74 expert coupling (ACL, Contusion, MOA,
# Effusion), Dirichlet target draws x row bootstrap; (b) dose curve a .05 / .1 / .15 / .2 (2x2 raking, point); (c) gold74 expert read
# with whatever OOF members cover the 74 gold studies (INFO, n 74, 11 MCL+); (d) gold58 c4 kernel path dose curve (INFO).
# Writes stack/stack_mcl.log. Aggregates only.
import sys, time
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
import load as LD
T0 = time.time()
src = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/stack_local.py', encoding='utf-8').read()
ns_ = {}; exec(src.split('summary = {}')[0].replace("log = open(OUT + '.log', 'w')", "log = open(OUT + '_mcl_tmp.log', 'w')"), ns_)
pipeline, wauc, rake, BASE = ns_['pipeline'], ns_['wauc'], ns_['rake'], ns_['BASE']
log = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/stack_mcl.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
L = LD.L; J = {c: i for i, c in enumerate(L)}; G = LD.G; pct = lambda v: rankdata(v) / len(v)
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}); Y74 = g[L].values.astype(float); allg = set(g.StudyInstanceUID)
r7 = LD.r7(); s44, _ = LD.s6seed(1044); s45, _ = LD.s6seed(1045); lv2 = LD.labels('v2')
tr = [i for i in r7.index if i not in allg and i in lv2.index]; tr3 = [i for i in tr if i in s44.index and i in s45.index]
X = {'R7': r7.loc[tr3, ['p|' + c for c in L]].values.astype(float), 's1044': s44.loc[tr3, L].values.astype(float),
     's1045': s45.loc[tr3, L].values.astype(float)}
Y = lv2.loc[tr3, L].values.astype(float); n = len(tr3); YF = np.nan_to_num(Y, nan=0.)
jm = J['MCL']
def table(cols):
    jc = [J[c] for c in cols]; ok = ~np.isnan(Y74[:, jc]).any(1); T = np.zeros((2,) * len(cols))
    for r in Y74[ok][:, jc].astype(int): T[tuple(r)] += 1
    return T
def wts(cols, m, Tp):
    jc = [J[c] for c in cols]; Ys = Y[m][:, jc].astype(int); obs = np.zeros((2,) * len(cols))
    for r in Ys: obs[tuple(r)] += 1
    obs /= obs.sum(); tgt = rake(Tp, Ys.mean(0), it=200); w = (tgt / obs)[tuple(Ys.T)]
    return w / w.mean()
NESTS = {'N_A': ('R7', 's1044', 's1045'), 'N_B': ('s1044', 'R7', 's1045'), 'N_C': ('s1045', 's1044', 'R7')}
rng = np.random.default_rng(41)
P('(a) MCLmm a .1 on pw30, macro e-4, Dirichlet target x row bootstrap (150 draws); raking columns as listed')
for nn, (ra, r6, s6) in NESTS.items():
    M = dict(rap=[(1., X[ra])], r6=X[r6], s6=X[s6]); B0 = pipeline(M, dict(BASE)); c = dict(BASE); c['mclmm'] = .1; B1 = pipeline(M, c)
    row = []
    for cols in (['MCL', 'Medial Meniscus'], ['MCL', 'Medial Meniscus', 'ACL'], ['MCL', 'Medial Meniscus', 'Contusion'],
                 ['MCL', 'Medial Meniscus', 'Medial OA'], ['MCL', 'Medial Meniscus', 'Effusion'], ['MCL', 'Medial Meniscus', 'ACL', 'Contusion']):
        m = ~np.isnan(Y[:, [J[x] for x in cols]]).any(1); idx = np.where(m)[0]; T = table(cols); d = []
        for _ in range(150):
            Tp = rng.dirichlet((T + .5).ravel()).reshape(T.shape); w = wts(cols, m, Tp)
            ii = rng.integers(0, len(idx), len(idx)); sel = idx[ii]; ws = w[ii]
            d.append((wauc(B1[sel, jm], YF[sel, jm], ws) - wauc(B0[sel, jm], YF[sel, jm], ws)) / 12 * 1e4)
        d = np.array(d); row.append('%s %+.2f [%+.2f, %+.2f] P>0 %.2f' % ('+'.join(x[:4] for x in cols[1:]), d.mean(), np.percentile(d, 10), np.percentile(d, 90), (d > 0).mean()))
    P('  %s: ' % nn + ' | '.join(row))
P('(b) dose curve, (MCL, MM) raking point estimates, macro e-4: a .05 / .10 / .15 / .20 (raw report in parentheses)')
for nn, (ra, r6, s6) in NESTS.items():
    M = dict(rap=[(1., X[ra])], r6=X[r6], s6=X[s6]); B0 = pipeline(M, dict(BASE)); cols = ['MCL', 'Medial Meniscus']
    m = ~np.isnan(Y[:, [J[x] for x in cols]]).any(1); T = table(cols); w = wts(cols, m, (T + .5) / (T + .5).sum()); one = np.ones(m.sum())
    out = []
    for a in (.05, .1, .15, .2):
        c = dict(BASE); c['mclmm'] = a; B1 = pipeline(M, c)
        out.append('%.2f %+.2f (%+.2f)' % (a, (wauc(B1[m, jm], YF[m, jm], w) - wauc(B0[m, jm], YF[m, jm], w)) / 12 * 1e4,
                                          (wauc(B1[m, jm], YF[m, jm], one) - wauc(B0[m, jm], YF[m, jm], one)) / 12 * 1e4))
    P('  %s: %s' % (nn, ' | '.join(out)))
# (c) gold74 expert read with OOF members on the gold studies
gid = g.StudyInstanceUID.tolist(); cov = {k: sum(i in d.index for i in gid) for k, d in (('R7', r7), ('s1044', s44), ('s1045', s45))}
P('(c) OOF coverage of the 74 gold studies: %s' % cov)
YG = g[L].values.astype(float); okm = ~np.isnan(YG[:, jm])
def expert_read(F, tag):
    out = []
    for a in (.05, .1, .2):
        G1 = F.copy(); G1[:, jm] = pct(pct(F[:, jm]) - a * pct(F[:, J['Medial Meniscus']]))
        y = YG[okm, jm]; d = wauc(G1[okm, jm], y, np.ones(okm.sum())) - wauc(F[okm, jm], y, np.ones(okm.sum())); bs = []
        for _ in range(2000):
            ii = rng.integers(0, okm.sum(), okm.sum()); yy = y[ii]
            if 0 < yy.sum() < len(yy):
                bs.append(wauc(G1[okm][ii, jm], yy, np.ones(len(ii))) - wauc(F[okm][ii, jm], yy, np.ones(len(ii))))
        out.append('a %.2f %+.4f label (P>0 %.2f)' % (a, d, np.mean(np.array(bs) > 0)))
    P('  %s x gold74 (n %d, MCL+ %d): %s' % (tag, okm.sum(), int(YG[okm, jm].sum()), ' | '.join(out)))
if cov['R7'] == 74:
    expert_read(r7.loc[gid, ['p|' + c for c in L]].values.astype(float), 'R7 OOF')
if min(cov.values()) == 74:
    Xg = {'R7': r7.loc[gid, ['p|' + c for c in L]].values.astype(float), 's1044': s44.loc[gid, L].values.astype(float), 's1045': s45.loc[gid, L].values.astype(float)}
    for nn, (ra, r6, s6) in NESTS.items():
        expert_read(pipeline(dict(rap=[(1., Xg[ra])], r6=Xg[r6], s6=Xg[s6]), dict(BASE)), 'nest ' + nn)
# (d) gold58 c4 kernel path dose curve (INFO)
g58 = g[g.source == 'original_gold58'].reset_index(drop=True); ids = g58.StudyInstanceUID.tolist(); Y58 = g58[L].values.astype(float)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values.astype(float)
M58 = dict(rap=[(.75, np.load(G + '/gold58_r384_kernelserve_cpu.npy')), (.25, np.load(G + '/gold58_r224_kernelserve.npy'))],
           r6=rd('R6_gold58_serve.csv'), s6=rd('n42gpu_gold58.csv'))
B0 = pipeline(M58, dict(BASE)); one = np.ones(len(ids)); out = []
for a in (.05, .1, .15, .2):
    c = dict(BASE); c['mclmm'] = a; B1 = pipeline(M58, c)
    out.append('%.2f %+.4f label' % (a, wauc(B1[:, jm], Y58[:, jm], one) - wauc(B0[:, jm], Y58[:, jm], one)))
P('(d) gold58 c4 kernel path (INFO): ' + ' | '.join(out))
P('STACK_MCL DONE %.0f s' % (time.time() - T0))
