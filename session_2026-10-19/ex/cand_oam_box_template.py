# cand_oam_box (10-20 ex, Claude). CANDIDATE oa_meniscus. Read-only; prints AGGREGATES ONLY (n, prevalence, phi, rho, AUC
# deltas, P, ESS). Never writes a file, never prints an id, a row or a per-knee value. Piped via ssh stdin into python -I.
# "G74P = {...}" (gold74 pairwise 2x2 counts) and "G74C5 = [...]" (gold74 5-label joint cell counts, 70 complete studies) are
# local RSNA aggregates filled in by make_cand_oam.py.
#
# Candidate (same-compartment degenerative coupling), applied ON TOP of E-OA(e), e in {.1, .2}:
#   E      = E-OA(e) on the OA trio: E_t = pct(f_t) - e * mean(pct(f_o)) over the other two OA members
#   MOA'   = pct(E_MOA) - a * pct(f_MM)      LOA' = pct(E_LOA) - a * pct(f_LM)       (OA side, "OAs")
#   MM'    = pct(f_MM)  - a * pct(f_MOA)     LM'  = pct(f_LM)  - a * pct(f_LOA)      (meniscus side, "MENs", raw partner)
#   MENsE  : meniscus side with the E-OA'd partner, MM' = pct(f_MM) - a * pct(E_MOA) (sensitivity)
#   a in {.1, .2} (decorrelation, the candidate) and {-.1, -.2} (negative control = borrow).
#   Increment = sum over the 4 touched labels of [AUC(candidate) - AUC(E-OA(e) alone)]; macro = /12. Because partners are raw
#   percentiles, the columns are independent, so BOTH = OAs + MENs exactly, per bootstrap draw.
# Labels B1 (primary): OA trio = P_FT (MOAKS m_cm_* full-thickness proxy, as oai_eoa_check.py), menisci = tear (T123 == 1).
#        B2 (sensitivity): OA = T123 composite >= .5, menisci >= .2 (incl. intrasubstance signal).
# Schemes (importance resampling by label cell; predictions given labels kept as observed):
#   raw | R1oa3 (skeptic: 3-label OA cell -> gold74) | R1m5 (5-label cell -> gold74 joint, +.5 smoothed, noisy: 70 studies)
#   R2m (IPF: OAI marginals + gold74 log OR on all 10 pairs) | R2m_hi / R2m_lo (all log OR +/- 1 sd)
#   R2m_schi / R2m_sclo (only the 2 same-compartment pairs MOA-MM, LOA-LM shifted +/- 1 sd; the candidate's key uncertainty)
import os, json
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm, spearmanr
from scipy.optimize import brentq
H = os.path.expanduser('~/oai_build')
OA = ['Medial OA', 'Lateral OA', 'PF OA']; MEN = ['Medial Meniscus', 'Lateral Meniscus']; LAB = OA + MEN
SH = ['MOA', 'LOA', 'PF', 'MM', 'LM']
COMP = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'],
        'PF OA': ['mp', 'lp', 'amf', 'alf']}
MODELS = {'D5mean_s1043': H + '/teacher/teacher_D5_mean_v00.csv', 's1044_mix50': H + '/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv',
          's1045_288': H + '/teacher2/f0_s1045_ttr288frac75__raw__A__gpu.csv'}
pct = lambda v: rankdata(v) / len(v)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
B = int(os.environ.get('BOOT', '400'))
ES = (.1, .2); AS = (.1, .2, -.1, -.2)
PARTNER = {0: 3, 1: 4, 3: 0, 4: 1}   # MOA<-MM, LOA<-LM, MM<-MOA, LM<-LOA


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


T = pd.read_csv(H + '/oai_labels_T123.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
S = pd.read_csv(H + '/oai_label_sources.csv', dtype={'kid': str}).set_index('kid').reindex(T.index)
Y1 = pd.DataFrame(index=T.index); Y2 = pd.DataFrame(index=T.index)
thr = lambda c, x: np.where(T[c].notna(), (T[c] >= x - 1e-6).astype(float), np.nan)
for c in OA:
    cc = S[['m_cm_' + r for r in COMP[c]]].astype(float).values
    area = np.floor(cc + 1e-6); ft = np.round((cc - area) * 10); fin = np.isfinite(cc)
    y = ((ft >= 1) & fin).any(1).astype(float); y[~fin.any(1)] = np.nan; Y1[c] = y; Y2[c] = thr(c, .5)
for c in MEN:
    Y1[c] = thr(c, 1.); Y2[c] = thr(c, .2)
P = {}
for k, f in MODELS.items():
    P[k] = pd.read_csv(f, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')[LAB]


def or_cells(p1, p2, lor):
    lo = max(0., p1 + p2 - 1) + 1e-9; hi = min(p1, p2) - 1e-9
    g = lambda p11: np.log(p11) + np.log(1 - p1 - p2 + p11) - np.log(p1 - p11) - np.log(p2 - p11) - lor
    p11 = brentq(g, lo, hi)
    return np.array([[1 - p1 - p2 + p11, p2 - p11], [p1 - p11, p11]])


def g74pair(i, j):
    k = LAB[i] + ' | ' + LAB[j]
    return np.array(G74P[k], float) if k in G74P else np.array(G74P[LAB[j] + ' | ' + LAB[i]], float)[[0, 2, 1, 3]]


def ipf_weights(Yv, shift, only=None):
    n = len(Yv); key = (Yv * (2 ** np.arange(5))).sum(1)
    p0 = np.bincount(key, minlength=32) / n; q = p0.copy(); cells = np.array([[(c >> i) & 1 for i in range(5)] for c in range(32)])
    marg = Yv.mean(0); tg = {}
    for i in range(5):
        for j in range(i + 1, 5):
            c = g74pair(i, j) + .5
            sh = shift if (only is None or (i, j) in only) else 0.
            lor = np.log(c[3] * c[0] / (c[1] * c[2])) + sh * np.sqrt((1 / c).sum())
            tg[(i, j)] = or_cells(marg[i], marg[j], lor)
    for _ in range(400):
        for (i, j), t in tg.items():
            for a in (0, 1):
                for b in (0, 1):
                    m = (cells[:, i] == a) & (cells[:, j] == b); s = q[m].sum()
                    if s > 0: q[m] *= t[a, b] / s
        q /= q.sum()
    return np.where(p0 > 0, q / np.maximum(p0, 1e-12), 0)[key]


def r1_oa3(Yv):
    k3 = Yv[:, 0] * 4 + Yv[:, 1] * 2 + Yv[:, 2]; po = np.bincount(k3, minlength=8) / len(Yv)
    pg = np.array([G74OA[format(i, '03b')] + .5 for i in range(8)]); pg /= pg.sum()
    return (pg / np.maximum(po, 1e-12))[k3]


def r1_m5(Yv):
    key = (Yv * (2 ** np.arange(5))).sum(1); po = np.bincount(key, minlength=32) / len(Yv)
    pg = np.array(G74C5, float) + .5; pg /= pg.sum()
    return (pg / np.maximum(po, 1e-12))[key]


def eoa(U, e):
    O = U.copy()
    for k in range(3):
        o = [m for m in range(3) if m != k]; O[:, k] = U[:, k] - e * (U[:, o[0]] + U[:, o[1]]) / 2
    return O


def build(F):
    """Returns dict of score columns: base U, E-OA(e) OA cols, and candidate cols keyed (mode, e, a, col)."""
    U = np.stack([pct(F[:, k]) for k in range(5)], 1); C = {'U': U}
    for e in ES:
        E = eoa(U, e); C[('E', e)] = E
        for a in AS:
            for k in (0, 1):
                C[('OAs', e, a, k)] = pct(E[:, k]) - a * U[:, PARTNER[k]]
            for k in (3, 4):
                C[('MENsE', e, a, k)] = U[:, k] - a * pct(E[:, PARTNER[k]])
    for a in AS:
        for k in (3, 4):
            C[('MENs', a, k)] = U[:, k] - a * U[:, PARTNER[k]]
    return C


def run(Yall, tag, scheme_names):
    ids = Yall.dropna().index
    for k in P: ids = ids.intersection(P[k].dropna().index)
    Yv = Yall.loc[ids, LAB].values.astype(int); n = len(ids)
    print('\n==== %s: n %d | prevalence %s' % (tag, n, dict(zip(SH, np.round(Yv.mean(0), 3).tolist()))), flush=True)
    ALL = {'raw': lambda: np.ones(n), 'R1oa3': lambda: r1_oa3(Yv), 'R1m5': lambda: r1_m5(Yv), 'R2m': lambda: ipf_weights(Yv, 0.),
           'R2m_hi': lambda: ipf_weights(Yv, 1.), 'R2m_lo': lambda: ipf_weights(Yv, -1.),
           'R2m_schi': lambda: ipf_weights(Yv, 1., only={(0, 3), (1, 4)}), 'R2m_sclo': lambda: ipf_weights(Yv, -1., only={(0, 3), (1, 4)})}
    SCH = {k: ALL[k]() for k in scheme_names}
    for nm, w in SCH.items():
        p = w / w.sum(); yy = Yv[np.random.default_rng(0).choice(n, 40000, p=p)]; ph = np.corrcoef(yy.T)
        print('scheme %-8s ESS %5.0f | prev %s | phi MOA-MM %+.3f LOA-LM %+.3f MOA-LOA %+.3f MOA-PF %+.3f LOA-PF %+.3f MM-LOA %+.3f LM-MOA %+.3f MM-LM %+.3f' % (
            nm, 1 / (p ** 2).sum(), np.round(yy.mean(0), 3).tolist(), ph[0, 3], ph[1, 4], ph[0, 1], ph[0, 2], ph[1, 2], ph[3, 1], ph[4, 0], ph[3, 4]))
    OUT = {}
    for mk, Pm in P.items():
        F = Pm.loc[ids, LAB].values.astype(float); C = build(F); U = C['U']
        # coupling + theory diagnostic on raw: rho(ns f_t, ns f_o) vs q = corr(ns f_o, y_t) / corr(ns f_t, y_t) (decor helps iff q < rho)
        Z = np.stack([ns(F[:, k]) for k in range(5)], 1)
        dg = []
        for t, o in PARTNER.items():
            rho = np.corrcoef(Z[:, t], Z[:, o])[0, 1]; q = np.corrcoef(Z[:, o], Yv[:, t])[0, 1] / np.corrcoef(Z[:, t], Yv[:, t])[0, 1]
            dg.append('%s<-%s rho %.3f q %.3f' % (SH[t], SH[o], rho, q))
        print('[%s] model diag (raw): %s | spearman MOA-MM %.3f LOA-LM %.3f' % (mk, ' | '.join(dg), spearmanr(F[:, 0], F[:, 3])[0],
                                                                                 spearmanr(F[:, 1], F[:, 4])[0]), flush=True)
        OUT[mk] = {}
        for sn, w in SCH.items():
            p = w / w.sum(); rng = np.random.default_rng(11)
            REC = {'EOA': {e: [] for e in ES}}
            keys = [('OAs', e, a) for e in ES for a in AS] + [('MENs', a) for a in AS] + [('MENsE', e, a) for e in ES for a in AS]
            for kk in keys: REC[kk] = []
            for _ in range(B):
                ii = rng.choice(n, n, p=p); y = Yv[ii]
                if (y.sum(0) < 5).any() or (y.sum(0) > n - 5).any(): continue
                b0 = np.array([auc(U[ii, k], y[:, k]) for k in range(5)])
                for e in ES:
                    E = C[('E', e)]; be = np.array([auc(E[ii, k], y[:, k]) for k in range(3)])
                    REC['EOA'][e].append((be - b0[:3]).sum())
                    for a in AS:
                        REC[('OAs', e, a)].append([auc(C[('OAs', e, a, k)][ii], y[:, k]) - be[k] for k in (0, 1)])
                        REC[('MENsE', e, a)].append([auc(C[('MENsE', e, a, k)][ii], y[:, k]) - b0[k] for k in (3, 4)])
                for a in AS:
                    REC[('MENs', a)].append([auc(C[('MENs', a, k)][ii], y[:, k]) - b0[k] for k in (3, 4)])
            R = {}
            for e in ES:
                v = np.array(REC['EOA'][e]) / 12; R['EOA%.1f' % e] = (float(v.mean()), float((v > 0).mean()))
            for a in AS:
                m = np.array(REC[('MENs', a)]) / 12
                for e in ES:
                    o = np.array(REC[('OAs', e, a)]) / 12; me = np.array(REC[('MENsE', e, a)]) / 12
                    both = o.sum(1) + m.sum(1); bothE = o.sum(1) + me.sum(1); med = o[:, 0] + m[:, 0]; lat = o[:, 1] + m[:, 1]
                    R['e%.1f a%+.1f' % (e, a)] = {
                        'BOTH': (float(both.mean()), float((both > 0).mean())), 'BOTH_Epartner': (float(bothE.mean()), float((bothE > 0).mean())),
                        'OAside': (float(o.sum(1).mean()), float((o.sum(1) > 0).mean())), 'MENside': (float(m.sum(1).mean()), float((m.sum(1) > 0).mean())),
                        'medial_pair': (float(med.mean()), float((med > 0).mean())), 'lateral_pair': (float(lat.mean()), float((lat > 0).mean())),
                        'per_label_x12': dict(zip(['MOA', 'LOA', 'MM', 'LM'], np.round(np.r_[o.mean(0), m.mean(0)] * 12, 5).tolist()))}
            OUT[mk][sn] = R
            print('[%s | %s | %s] nboot %d | E-OA .1 %+.5f (P %.2f) E-OA .2 %+.5f (P %.2f)' % (
                tag, mk, sn, len(REC['EOA'][.1]), *R['EOA0.1'], *R['EOA0.2']))
            for e in ES:
                for a in AS:
                    r = R['e%.1f a%+.1f' % (e, a)]
                    print('   on E-OA %.1f, a %+.1f: BOTH inc macro %+.5f (P %.2f) | Epartner %+.5f (P %.2f) | OAside %+.5f (P %.2f) | MENside %+.5f (P %.2f) | medial %+.5f (P %.2f) lateral %+.5f (P %.2f) | per-label AUC %s' % (
                        e, a, *r['BOTH'], *r['BOTH_Epartner'], *r['OAside'], *r['MENside'], *r['medial_pair'], *r['lateral_pair'], r['per_label_x12']), flush=True)
    return OUT


RES = {'B1': run(Y1, 'B1 primary (OA P_FT, menisci tear)', ['raw', 'R1oa3', 'R1m5', 'R2m', 'R2m_hi', 'R2m_lo', 'R2m_schi', 'R2m_sclo'])}
RES['B2'] = run(Y2, 'B2 sensitivity (OA T123 >= .5, menisci >= .2)', ['raw', 'R2m'])
print('JSON_BEGIN'); print(json.dumps(RES)); print('JSON_END')
