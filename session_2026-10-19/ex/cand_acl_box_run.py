# cand_acl_box (10-20 ex, Claude). Read-only; prints AGGREGATES ONLY (n, prevalence, phi, ESS, AUC deltas, P, rho, q).
# Never writes a file, never prints an id, a row or a per-knee value. Piped via ssh stdin into python -I.
# "G74P = {...}" (gold74 pairwise 2x2 counts among the 7 labels) and "G74J4 = [...]" (gold74 4-label joint cells, ACL/MM/LM/Eff)
# are local RSNA aggregates filled in by cand_acl_make.py.
#
# Candidate acl: out_ACL = pct(f_ACL) - a * mean(pct(f_o) for o in group), a in {-.1, .1, .2}; groups LM (only ACL pair with
# consistent excess coupling in coupling_map.md), MM (mixed), MM+LM, Eff (deficit; reference), MM+LM+Eff (the task group).
# Labels (B1, as box_coupling): ACL T123 >= .5 (partial + complete; B2 = complete only); menisci tear T123 = 1;
# Effusion grade >= 2 (T123 >= .8); OA trio = P_FT (MOAKS full-thickness cartilage proxy, as oai_eoa_check.py).
# Weighting (knee resampling by label cell; predictions given labels kept as observed = the skeptic method):
#   raw      : OAI as is (paired bootstrap)
#   R2x4     : IPF on the 4-label cell (ACL, MM, LM, Eff): every pairwise 2x2 at OAI marginals + gold74 log OR
#   R2x7     : IPF on the 7-label cell (+ MOA, LOA, PFOA): also moves ACL-OA from OAI's degenerative +.5 tet to gold74's
#              -.38 / +.04 / -.46. PRIMARY criterion-2 scheme (OA is the main confounder of a meniscus score vs ACL in OAI)
#   R2x7_hi  : R2x7 with the 6 ACL-pair log ORs + 1 sd (stress against decorrelation)
#   R2x7_lo  : R2x7 with the 6 ACL-pair log ORs - 1 sd
#   R2x7_hiI : R2x7 with only the involved ACL-{MM, LM, Eff} log ORs + 1 sd
#   R1j4     : gold74 4-label joint (16 cells, +.5) over the OAI 4-label cell (moves marginals too: ACL prevalence -> ~.39)
import os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
from scipy.optimize import brentq
H = os.path.expanduser('~/oai_build')
G74P = {"ACL | Medial Meniscus": [24, 20, 14, 15], "ACL | Lateral Meniscus": [26, 19, 18, 11], "ACL | Effusion": [19, 26, 7, 22], "ACL | Medial OA": [26, 17, 24, 5], "ACL | Lateral OA": [34, 8, 23, 6], "ACL | PF OA": [20, 24, 22, 7], "Medial Meniscus | Lateral Meniscus": [25, 13, 19, 16], "Medial Meniscus | Effusion": [17, 21, 8, 27], "Medial Meniscus | Medial OA": [33, 4, 16, 18], "Medial Meniscus | Lateral OA": [30, 7, 26, 7], "Medial Meniscus | PF OA": [26, 12, 16, 18], "Lateral Meniscus | Effusion": [18, 26, 8, 22], "Lateral Meniscus | Medial OA": [36, 7, 14, 15], "Lateral Meniscus | Lateral OA": [39, 3, 18, 11], "Lateral Meniscus | PF OA": [27, 17, 15, 14], "Effusion | Medial OA": [20, 5, 30, 17], "Effusion | Lateral OA": [23, 2, 34, 12], "Effusion | PF OA": [16, 10, 26, 21], "Medial OA | Lateral OA": [45, 5, 12, 9], "Medial OA | PF OA": [36, 14, 6, 16], "Lateral OA | PF OA": [37, 20, 5, 9]}
G74J4 = [9, 2, 3, 4, 5, 1, 1, 0, 7, 7, 7, 5, 3, 4, 9, 6]
L7 = ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Effusion', 'Medial OA', 'Lateral OA', 'PF OA']
OA = ['Medial OA', 'Lateral OA', 'PF OA']
COMP = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'],
        'PF OA': ['mp', 'lp', 'amf', 'alf']}
MODELS = {'D5mean_s1043': H + '/teacher/teacher_D5_mean_v00.csv', 's1044_mix50': H + '/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv',
          's1045_288': H + '/teacher2/f0_s1045_ttr288frac75__raw__A__gpu.csv'}
GROUPS = {'LM': ['Lateral Meniscus'], 'MM': ['Medial Meniscus'], 'MM+LM': ['Medial Meniscus', 'Lateral Meniscus'],
          'Eff': ['Effusion'], 'MM+LM+Eff': ['Medial Meniscus', 'Lateral Meniscus', 'Effusion']}
AS = (-.1, .1, .2)
pct = lambda v: rankdata(v) / len(v)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
BR = int(os.environ.get('BOOT', '1000')); BW = int(os.environ.get('RBOOT', '400'))


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


T = pd.read_csv(H + '/oai_labels_T123.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
S = pd.read_csv(H + '/oai_label_sources.csv', dtype={'kid': str}).set_index('kid').reindex(T.index)
thr = lambda c, x: np.where(T[c].notna(), (T[c] >= x - 1e-6).astype(float), np.nan)
Y = pd.DataFrame(index=T.index)
Y['ACL'] = thr('ACL', .5); Y['ACL_B2'] = thr('ACL', 1.)
Y['Medial Meniscus'] = thr('Medial Meniscus', 1.); Y['Lateral Meniscus'] = thr('Lateral Meniscus', 1.)
Y['Effusion'] = thr('Effusion', .8)
for c in OA:
    cc = S[['m_cm_' + r for r in COMP[c]]].astype(float).values
    area = np.floor(cc + 1e-6); ft = np.round((cc - area) * 10); fin = np.isfinite(cc)
    y = ((ft >= 1) & fin).any(1).astype(float); y[~fin.any(1)] = np.nan; Y[c] = y
FC = ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Effusion']
P = {k: pd.read_csv(f, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')[FC]
     for k, f in MODELS.items()}
ids = Y[L7 + ['ACL_B2']].dropna().index
for k in P: ids = ids.intersection(P[k].dropna().index)
n = len(ids); Yv = Y.loc[ids, L7].values.astype(int); yB2 = Y.loc[ids, 'ACL_B2'].values.astype(int)
print('cohort: OAI knees with ACL, MM, LM, Eff, OA-trio labels and all 3 model predictions: n %d' % n)
print('prevalence %s | ACL_B2 (complete) %.3f (pos %d) | ACL B1 pos %d' % (
    dict(zip(L7, np.round(Yv.mean(0), 3).tolist())), yB2.mean(), yB2.sum(), Yv[:, 0].sum()))


def or_cells(p1, p2, lor):
    lo = max(0., p1 + p2 - 1) + 1e-9; hi = min(p1, p2) - 1e-9
    g = lambda p11: np.log(p11) + np.log(1 - p1 - p2 + p11) - np.log(p1 - p11) - np.log(p2 - p11) - lor
    p11 = brentq(g, lo, hi)
    return np.array([[1 - p1 - p2 + p11, p2 - p11], [p1 - p11, p11]])   # [a][b]


def g74c(i, j):
    k = L7[i] + ' | ' + L7[j]
    if k in G74P: return np.array(G74P[k], float)
    return np.array(G74P[L7[j] + ' | ' + L7[i]], float)[[0, 2, 1, 3]]


def ipf(sub, shift=None, iters=400):
    shift = shift or {}
    m = len(sub); Ys = Yv[:, sub]; key = (Ys * (2 ** np.arange(m))).sum(1)
    p0 = np.bincount(key, minlength=2 ** m) / n; q = p0.copy()
    cells = np.array([[(c >> i) & 1 for i in range(m)] for c in range(2 ** m)])
    marg = Ys.mean(0); tg = {}
    for a in range(m):
        for b in range(a + 1, m):
            c = g74c(sub[a], sub[b]) + .5
            lor = np.log(c[3] * c[0] / (c[1] * c[2])) + shift.get((sub[a], sub[b]), 0.) * np.sqrt((1 / c).sum())
            tg[(a, b)] = (or_cells(marg[a], marg[b], lor), lor)
    for _ in range(iters):
        for (a, b), (t, _l) in tg.items():
            for u in (0, 1):
                for v in (0, 1):
                    mm = (cells[:, a] == u) & (cells[:, b] == v); s = q[mm].sum()
                    if s > 0: q[mm] *= t[u, v] / s
        q /= q.sum()
    err = []
    for (a, b), (t, lor) in tg.items():
        pc = [q[(cells[:, a] == u) & (cells[:, b] == v)].sum() for u in (0, 1) for v in (0, 1)]
        err.append(abs(np.log(pc[3] * pc[0] / (pc[1] * pc[2])) - lor))
    w = np.where(p0 > 0, q / np.maximum(p0, 1e-12), 0)[key]
    return w, max(err)


def r1j4():
    key = (Yv[:, :4] * (2 ** np.arange(4))).sum(1); p0 = np.bincount(key, minlength=16) / n
    pg = np.array(G74J4, float) + .5; pg /= pg.sum()
    return (pg / np.maximum(p0, 1e-12))[key]


ACLP = [(0, j) for j in range(1, 7)]; INV = [(0, 1), (0, 2), (0, 3)]
SCH = {'raw': (np.ones(n), 0.)}
SCH['R2x4'] = ipf([0, 1, 2, 3])
SCH['R2x7'] = ipf(list(range(7)))
SCH['R2x7_hi'] = ipf(list(range(7)), {p: 1. for p in ACLP})
SCH['R2x7_lo'] = ipf(list(range(7)), {p: -1. for p in ACLP})
SCH['R2x7_hiI'] = ipf(list(range(7)), {p: 1. for p in INV})
SCH['R1j4'] = (r1j4(), 0.)


def wphi(w, i, j):
    p = w / w.sum(); a = Yv[:, i]; b = Yv[:, j]
    ma = (p * a).sum(); mb = (p * b).sum(); cov = (p * a * b).sum() - ma * mb
    return cov / np.sqrt(ma * (1 - ma) * mb * (1 - mb))


for nm, (w, err) in SCH.items():
    p = w / w.sum()
    print('scheme %-9s ESS %5.0f (max |log OR err| %.3f) | ACL prev %.3f | weighted phi ACL-MM %+.3f ACL-LM %+.3f ACL-Eff %+.3f '
          'ACL-MOA %+.3f ACL-LOA %+.3f ACL-PF %+.3f | MM-LM %+.3f LM-LOA %+.3f MM-MOA %+.3f' % (
              nm, 1 / (p ** 2).sum(), err, (p * Yv[:, 0]).sum(), *[wphi(w, 0, j) for j in range(1, 7)], wphi(w, 1, 2), wphi(w, 2, 5),
              wphi(w, 1, 4)))
sys.stdout.flush()


def wcorr(x, y, p):
    mx = (p * x).sum(); my = (p * y).sum()
    return ((p * (x - mx) * (y - my)).sum()) / np.sqrt((p * (x - mx) ** 2).sum() * (p * (y - my) ** 2).sum())


def run(yt, tag, schemes):
    for mk, Pm in P.items():
        F = {c: Pm.loc[ids, c].values.astype(float) for c in FC}
        U = {c: pct(F[c]) for c in FC}
        outs = {(g, a): U['ACL'] - a * np.mean([U[o] for o in grp], 0) for g, grp in GROUPS.items() for a in AS}
        zt = ns(F['ACL']); zg = {g: ns(np.mean([U[o] for o in grp], 0)) for g, grp in GROUPS.items()}
        base_full = auc(F['ACL'], yt)
        for sn in schemes:
            w = SCH[sn][0]; p = w / w.sum(); rng = np.random.default_rng(11); D = {k: [] for k in outs}; BB = []
            nb = BR if sn == 'raw' else BW
            for _ in range(nb):
                ii = rng.integers(0, n, n) if sn == 'raw' else rng.choice(n, n, p=p); y = yt[ii]
                if y.sum() < 5 or y.sum() > n - 5: continue
                b0 = auc(F['ACL'][ii], y); BB.append(b0)
                for k, v in outs.items(): D[k].append(auc(v[ii], y) - b0)
            ct = wcorr(zt, yt.astype(float), p)
            th = {g: (wcorr(zt, zg[g], p), wcorr(zg[g], yt.astype(float), p) / ct) for g in GROUPS}
            pt = {k: (auc(v, yt) - base_full) for k, v in outs.items()} if sn == 'raw' else None
            for g in GROUPS:
                s = ' | '.join('a %+.1f %+.4f (P>0 %.2f)' % (a, (pt[(g, a)] if pt else np.mean(D[(g, a)])), np.mean(np.array(D[(g, a)]) > 0))
                               for a in AS)
                print('[%s | %-12s | %-9s] base %.4f | ACL <- %-10s rho %+.3f q %+.3f | %s' % (
                    tag, mk, sn, base_full if pt else np.mean(BB), g, th[g][0], th[g][1], s))
            sys.stdout.flush()


run(Yv[:, 0], 'ACL B1', list(SCH))
run(yB2, 'ACL B2', ['raw', 'R2x7'])
print('DONE (aggregates only; nothing written on the box)')
