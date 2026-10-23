# box_followup (10-19 ex, Claude). Read-only; prints AGGREGATES ONLY (AUC deltas, P, ESS, phi). Never writes a file, never
# prints an id, a row or a per-knee value. Piped via ssh stdin. "G74P = {...}" (gold74 pairwise 2x2 counts, local aggregates) and
# "G74OA = {...}" (gold74 3-label OA cells, from skeptic box_reweight.py) are filled in locally by make_followup.py.
# Labels: OA trio = P_FT (MOAKS full-thickness cartilage proxy, as oai_eoa_check.py), menisci = tear (T123 == 1).
# Weighting schemes (knee resampling by label cell; predictions given labels kept as observed):
#   raw      : OAI as is
#   R1oa3    : skeptic R1, the 3-label OA cell matched to gold74 (71 complete), menisci free
#   R2m      : IPF on the 5-label cell (MOA, LOA, PF, MM, LM): every pairwise 2x2 set to OAI marginals + gold74 log OR (coupling only)
#   R2m_hi   : as R2m with every gold74 log OR + 1 sd (more expert coupling; stress test against decorrelation)
#   R2m_lo   : as R2m with every gold74 log OR - 1 sd (less expert coupling; stress test against borrowing)
# Transforms (single model, OA columns only; menisci untouched):
#   EOA(a)   : out_t = pct(f_t) - a * mean(pct(f_o1), pct(f_o2)) for the OA trio
#   +LM(b)   : then LOA = pct(LOA) - b * pct(f_MM)            (pair LOA <- MM, excess-coupling decorrelation)
#   +BOR(c)  : then MOA = pct(MOA) + c * pct(f_MM), LOA = pct(LOA) + c * pct(f_LM)  (same-compartment meniscus borrow)
#   pair-only variants without EOA.
import os, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.optimize import brentq
H = os.path.expanduser('~/oai_build')
G74P = {"ACL | Medial Meniscus": [24, 20, 14, 15], "ACL | Lateral Meniscus": [26, 19, 18, 11], "ACL | Medial OA": [26, 17, 24, 5], "ACL | Lateral OA": [34, 8, 23, 6], "ACL | PF OA": [20, 24, 22, 7], "ACL | Effusion": [19, 26, 7, 22], "ACL | Synovitis": [22, 23, 15, 13], "ACL | Baker's": [28, 13, 21, 5], "Medial Meniscus | Lateral Meniscus": [25, 13, 19, 16], "Medial Meniscus | Medial OA": [33, 4, 16, 18], "Medial Meniscus | Lateral OA": [30, 7, 26, 7], "Medial Meniscus | PF OA": [26, 12, 16, 18], "Medial Meniscus | Effusion": [17, 21, 8, 27], "Medial Meniscus | Synovitis": [22, 16, 14, 20], "Medial Meniscus | Baker's": [30, 5, 18, 13], "Lateral Meniscus | Medial OA": [36, 7, 14, 15], "Lateral Meniscus | Lateral OA": [39, 3, 18, 11], "Lateral Meniscus | PF OA": [27, 17, 15, 14], "Lateral Meniscus | Effusion": [18, 26, 8, 22], "Lateral Meniscus | Synovitis": [25, 18, 12, 18], "Lateral Meniscus | Baker's": [33, 7, 16, 11], "Medial OA | Lateral OA": [45, 5, 12, 9], "Medial OA | PF OA": [36, 14, 6, 16], "Medial OA | Effusion": [20, 30, 5, 17], "Medial OA | Synovitis": [28, 21, 8, 14], "Medial OA | Baker's": [42, 6, 7, 11], "Lateral OA | PF OA": [37, 20, 5, 9], "Lateral OA | Effusion": [23, 34, 2, 12], "Lateral OA | Synovitis": [32, 24, 3, 11], "Lateral OA | Baker's": [44, 10, 5, 6], "PF OA | Effusion": [16, 26, 10, 21], "PF OA | Synovitis": [25, 17, 12, 18], "PF OA | Baker's": [33, 7, 16, 10], "Effusion | Synovitis": [20, 5, 17, 31], "Effusion | Baker's": [20, 4, 29, 14], "Synovitis | Baker's": [29, 5, 20, 13]}
G74OA = {"000": 33, "001": 12, "010": 3, "011": 2, "100": 4, "101": 8, "110": 2, "111": 7}
OA = ['Medial OA', 'Lateral OA', 'PF OA']; MEN = ['Medial Meniscus', 'Lateral Meniscus']; LAB = OA + MEN
COMP = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'],
        'PF OA': ['mp', 'lp', 'amf', 'alf']}
MODELS = {'D5mean_s1043': H + '/teacher/teacher_D5_mean_v00.csv', 's1044_mix50': H + '/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv',
          's1045_288': H + '/teacher2/f0_s1045_ttr288frac75__raw__A__gpu.csv'}
pct = lambda v: rankdata(v) / len(v)
B = int(os.environ.get('BOOT', '300'))


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


T = pd.read_csv(H + '/oai_labels_T123.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
S = pd.read_csv(H + '/oai_label_sources.csv', dtype={'kid': str}).set_index('kid').reindex(T.index)
Y = pd.DataFrame(index=T.index)
for c in OA:
    cc = S[['m_cm_' + r for r in COMP[c]]].astype(float).values
    area = np.floor(cc + 1e-6); ft = np.round((cc - area) * 10); fin = np.isfinite(cc)
    y = ((ft >= 1) & fin).any(1).astype(float); y[~fin.any(1)] = np.nan; Y[c] = y
for c in MEN:
    Y[c] = np.where(T[c].notna(), (T[c] >= 1 - 1e-6).astype(float), np.nan)
P = {}
for k, f in MODELS.items():
    P[k] = pd.read_csv(f, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')[LAB]
ids = Y.dropna().index
for k in P: ids = ids.intersection(P[k].dropna().index)
Yv = Y.loc[ids, LAB].values.astype(int); n = len(ids)
print('n %d | prevalence %s' % (n, dict(zip(LAB, np.round(Yv.mean(0), 3).tolist()))))
key = (Yv * (2 ** np.arange(5))).sum(1)


def or_cells(p1, p2, lor):
    lo = max(0., p1 + p2 - 1) + 1e-9; hi = min(p1, p2) - 1e-9
    g = lambda p11: np.log(p11) + np.log(1 - p1 - p2 + p11) - np.log(p1 - p11) - np.log(p2 - p11) - lor
    p11 = brentq(g, lo, hi)
    return np.array([[1 - p1 - p2 + p11, p2 - p11], [p1 - p11, p11]])   # [a][b]


def ipf_weights(shift):
    p0 = np.bincount(key, minlength=32) / n; q = p0.copy(); cells = np.array([[(c >> i) & 1 for i in range(5)] for c in range(32)])
    marg = Yv.mean(0); tg = {}
    for i in range(5):
        for j in range(i + 1, 5):
            c = np.array(G74P[LAB[i] + ' | ' + LAB[j]] if (LAB[i] + ' | ' + LAB[j]) in G74P else
                         np.array(G74P[LAB[j] + ' | ' + LAB[i]])[[0, 2, 1, 3]], float) + .5
            lor = np.log(c[3] * c[0] / (c[1] * c[2])) + shift * np.sqrt((1 / c).sum())
            tg[(i, j)] = or_cells(marg[i], marg[j], lor)
    for _ in range(300):
        for (i, j), t in tg.items():
            for a in (0, 1):
                for b in (0, 1):
                    m = (cells[:, i] == a) & (cells[:, j] == b); s = q[m].sum()
                    if s > 0: q[m] *= t[a, b] / s
        q /= q.sum()
    w = np.where(p0 > 0, q / np.maximum(p0, 1e-12), 0)[key]
    return w


def r1_oa3():
    k3 = Yv[:, 0] * 4 + Yv[:, 1] * 2 + Yv[:, 2]
    po = np.bincount(k3, minlength=8) / n
    pg = np.array([G74OA[format(i, '03b')] + .5 for i in range(8)]); pg /= pg.sum()
    return (pg / np.maximum(po, 1e-12))[k3]


SCHEMES = {'raw': np.ones(n), 'R1oa3': r1_oa3(), 'R2m': ipf_weights(0.), 'R2m_hi': ipf_weights(1.), 'R2m_lo': ipf_weights(-1.)}
for nm, w in SCHEMES.items():
    p = w / w.sum(); ph = np.corrcoef(Yv[np.random.default_rng(0).choice(n, 20000, p=p)].T)
    print('scheme %-7s ESS %5.0f | weighted phi MOA-LOA %+.3f MOA-PF %+.3f LOA-PF %+.3f MM-LOA %+.3f MM-MOA %+.3f LM-LOA %+.3f MM-LM %+.3f LM-PF %+.3f' % (
        nm, 1 / (p ** 2).sum(), ph[0, 1], ph[0, 2], ph[1, 2], ph[3, 1], ph[3, 0], ph[4, 1], ph[3, 4], ph[4, 2]))


def eoa(U, a):
    O = U.copy()
    for k in range(3):
        o = [m for m in range(3) if m != k]; O[:, k] = U[:, k] - a * (U[:, o[0]] + U[:, o[1]]) / 2
    return O


def transforms(F):
    U = np.stack([pct(F[:, k]) for k in range(5)], 1); out = {}
    for a in (.1, .15, .2):
        out['EOA%.2f' % a] = eoa(U, a)[:, :3]
    for a in (.1, .2):
        E = eoa(U, a)[:, :3]
        X = E.copy(); X[:, 1] = pct(E[:, 1]) - .1 * U[:, 3]; out['EOA%.2f+LM.1' % a] = X
        X = E.copy(); X[:, 0] = pct(E[:, 0]) + .1 * U[:, 3]; X[:, 1] = pct(E[:, 1]) + .1 * U[:, 4]; out['EOA%.2f+BOR.1' % a] = X
        X = E.copy(); X[:, 0] = pct(E[:, 0]) + .1 * U[:, 3]; X[:, 1] = pct(E[:, 1]) - .1 * U[:, 3] + .1 * U[:, 4]; out['EOA%.2f+LM.1+BOR.1' % a] = X
    X = U[:, :3].copy(); X[:, 1] = U[:, 1] - .1 * U[:, 3]; out['pair LOA<-MM .1'] = X
    X = U[:, :3].copy(); X[:, 0] = U[:, 0] + .1 * U[:, 3]; X[:, 1] = U[:, 1] + .1 * U[:, 4]; out['borrow MOA<-MM, LOA<-LM .1'] = X
    X = U[:, :3].copy(); X[:, 0] = U[:, 0] + .2 * U[:, 3]; X[:, 1] = U[:, 1] + .2 * U[:, 4]; out['borrow MOA<-MM, LOA<-LM .2'] = X
    X = U[:, :3].copy(); X[:, 0] = U[:, 0] - .1 * U[:, 3]; X[:, 1] = U[:, 1] - .1 * U[:, 4]; out['control: decor MOA<-MM, LOA<-LM .1'] = X
    return out


RES = {}
for mk, Pm in P.items():
    F = Pm.loc[ids, LAB].values.astype(float); TR = transforms(F); RES[mk] = {}
    for sn, w in SCHEMES.items():
        p = w / w.sum(); rng = np.random.default_rng(7); D = {t: [] for t in TR}; I = {t: [] for t in TR}
        for _ in range(B):
            ii = np.arange(n) if (sn == 'raw' and _ == 0 and False) else rng.choice(n, n, p=p); y = Yv[ii, :3]
            if (y.sum(0) < 5).any(): continue
            base = np.array([auc(F[ii, k], y[:, k]) for k in range(3)])
            ref = None
            per = {}
            for t, O in TR.items():
                per[t] = np.array([auc(O[ii, k], y[:, k]) for k in range(3)]) - base
            for t in TR:
                D[t].append(per[t].sum())                     # sum over the 3 OA labels; macro = /12
                I[t].append(per[t].sum() - per['EOA0.10'].sum())
        RES[mk][sn] = {t: {'sum': float(np.mean(D[t])), 'P': float(np.mean(np.array(D[t]) > 0)), 'inc_vs_EOA.1': float(np.mean(I[t])),
                           'P_inc': float(np.mean(np.array(I[t]) > 0))} for t in TR}
        print('[%s | %s] ' % (mk, sn) + ' | '.join('%s macro %+.5f (P %.2f; vs EOA.1 %+.5f P %.2f)' % (
            t, v['sum'] / 12, v['P'], v['inc_vs_EOA.1'] / 12, v['P_inc']) for t, v in RES[mk][sn].items()), flush=True)
# borrow / pair directions under pair-specific OR matching are covered by R2m (5-label IPF) above.
print('JSON_BEGIN'); print(json.dumps(RES)); print('JSON_END')
