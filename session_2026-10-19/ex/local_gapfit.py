# local_gapfit.py (10-20 ex): MODEL extrapolation of the symmetric E-OA optimum to c4.
# (1) From the box aggregates (box_curve_*.log JSON; per model x scheme: mean rho, mean q, symmetric curve), fit
#     inc(a) = delta(S a) - delta(S .1) against gap = rho - q (OAI-free models, reweighted schemes, P_FT and P_AREA).
# (2) Bootstrap c4's gap on gold58 (and R7 on gold58 / gold74) and push it through the fit -> P(eoa20 > eoa10), P(eoa15 > eoa10).
import json, numpy as np, pandas as pd
from scipy.stats import rankdata, norm
E = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/'
R = {}
for j in ('D5', 's1044', 's1045', 'ENS3', 'FOLDS', 'AREA'):
    s = open(E + 'box_curve_%s.log' % j, encoding='utf-8').read(); R[j] = json.loads(s.split('JSON_BEGIN')[1].split('JSON_END')[0])
pts = []
for j, d in R.items():
    for m, ss in d['res'].items():
        for s, r in ss.items():
            if s == 'raw': continue
            gap = np.mean(r['desc']['rho']) - np.mean(r['desc']['q'])
            c = {a: r['S%.2f' % a]['macro']['m'] for a in (.05, .1, .15, .2, .25, .3)}
            pts.append(dict(src=j, m=m, s=s, prox=d['proxy'], gap=gap, d1=c[.1], i15=c[.15] - c[.1], i2=c[.2] - c[.1], i05=c[.05] - c[.1],
                            best=max(c, key=c.get)))
P = pd.DataFrame(pts)
print(P.groupby('prox').size().to_dict(), 'points')
for prox in ('P_FT', 'P_AREA', None):
    Q = P if prox is None else P[P.prox == prox]
    for y in ('i15', 'i2', 'd1'):
        b = np.polyfit(Q.gap, Q[y], 1); r = np.corrcoef(Q.gap, Q[y])[0, 1]
        res = Q[y] - np.polyval(b, Q.gap); print('%-6s %-3s = %+.5f + %+.5f * gap  (r %.2f, resid sd %.5f, n %d) | zero at gap %.3f' % (
            prox or 'both', y, b[1], b[0], r, res.std(), len(Q), -b[1] / b[0]))
    print('   best a by gap bin:', Q.assign(bin=pd.cut(Q.gap, [-1, .02, .04, .06, .07, .08, .1, 1])).groupby('bin', observed=True).best.agg(lambda x: x.value_counts().to_dict()).to_dict())
FIT = {y: np.polyfit(P[P.prox == 'P_FT'].gap, P[P.prox == 'P_FT'][y], 1) for y in ('i15', 'i2')}
FITA = {y: np.polyfit(P[P.prox == 'P_AREA'].gap, P[P.prox == 'P_AREA'][y], 1) for y in ('i15', 'i2')}

# gold58 c4 gap bootstrap
R0 = 'C:/Users/eeshs/Downloads/rsna-knee'; G = R0 + '/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
OA = ['Medial OA', 'Lateral OA', 'PF OA']
pct = lambda v: rankdata(v) / len(v)
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
g58 = g[g.source == 'original_gold58'].reset_index(drop=True); ids = g58.StudyInstanceUID.tolist(); n58 = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
C4 = np.zeros((n58, 12))
for j, c in enumerate(L):
    w = C4W.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j])
    b = w * rr + (1 - w) * pct(R6[:, j]); C4[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
F7 = r7[['p|' + c for c in OA]].set_axis(OA, axis=1)
g74 = g.dropna(subset=OA)
SETS = {'gold58 c4': (C4[:, [L.index(c) for c in OA]], g58[OA].values.astype(int)),
        'gold58 R7': (F7.loc[ids].values, g58[OA].values.astype(int)),
        'gold74 R7': (F7.loc[g74.StudyInstanceUID].values, g74[OA].values.astype(int))}


def gap(F, Y):
    n = len(Y); Z = np.stack([norm.ppf((rankdata(F[:, k]) - .5) / n) for k in range(3)], 1); rz = np.corrcoef(Z.T)
    q = []
    for k in range(3):
        o = [m for m in range(3) if m != k]; ck = np.corrcoef(Z[:, k], Y[:, k])[0, 1]
        q.append(np.mean([np.corrcoef(Z[:, j], Y[:, k])[0, 1] for j in o]) / ck)
    return np.mean(rz[np.triu_indices(3, 1)]) - np.mean(q), np.mean(rz[np.triu_indices(3, 1)]), np.mean(q)


rng = np.random.default_rng(0)
for nm, (F, Y) in SETS.items():
    g0, r0, q0 = gap(F, Y); bs = []
    for _ in range(2000):
        ii = rng.integers(0, len(Y), len(Y))
        if (Y[ii].sum(0) < 3).any(): continue
        bs.append(gap(F[ii], Y[ii])[0])
    bs = np.array(bs)
    pr = {}
    for tag, fit in (('P_FT', FIT), ('P_AREA', FITA)):
        for y in ('i15', 'i2'):
            pred = np.polyval(fit[y], bs); pr[tag + ' ' + y] = (np.polyval(fit[y], g0), np.mean(pred > 0))
    print('[%s] rho %.3f q %.3f gap %.3f (boot sd %.3f, 80%% [%.3f, %.3f]) | predicted inc over S.1 (point, P>0 over gap uncertainty): %s' % (
        nm, r0, q0, g0, bs.std(), np.quantile(bs, .1), np.quantile(bs, .9),
        ' | '.join('%s %+.5f (%.2f)' % (k, v[0], v[1]) for k, v in pr.items())))
