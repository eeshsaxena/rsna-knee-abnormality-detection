# box_global (10-20 ex, Claude). Read-only; prints AGGREGATES ONLY (n, prevalence, AUC deltas, P, ESS, phi, rho). Never writes a
# file, never prints an id, a row or a per-knee value. Piped via ssh stdin into python -I. "G74P = {...}" (gold74 pairwise 2x2
# counts over the 9 OAI-covered labels, local aggregates from local_coupling.json) is filled in locally by make_global.py.
#
# CANDIDATE "global factor removal": every OAI-covered label minus a * mean pct of the other covered labels (a "general
# abnormality" factor); compared with the group-wise E-OA, and applied only outside the OA trio on top of E-OA .1.
# Transforms (U = pct of the 12 prediction columns over the OAI knees; AUC is invariant to the outer pct of the deployed form):
#   EOA a      : OA trio only, out_t = U_t - a * mean(U of the 2 other OA labels)
#   G9 a       : each of the 9 covered labels, out_t = U_t - a * mean(U of the other 8 covered labels)
#   G12 a      : each of the 9 covered labels, out_t = U_t - a * mean(U of the other 11 prediction columns) (deployed form would
#                also move MCL / Contusion / Fracture; those cannot be scored on OAI)
#   E1+N9 a    : OA trio = EOA .1; each NON-OA covered label (ACL, MM, LM, Eff, Syn, Bak) out_t = U_t - a * mean(U of the other 8)
#   E1+N12 a   : as E1+N9 with the mean over the other 11 prediction columns
# Labels: B1 binarisation exactly as box_coupling_template.py (ACL >= .5; menisci tear = 1; OA = P_FT MOAKS full-thickness
#   cartilage proxy; Effusion >= .8 (grade >= 2); Synovitis (Hoffa) >= 1/3; Baker's 1). B2 = alternative thresholds, raw only.
# Score: per-label AUC on the knees where that label exists; "macro" = sum of the 9 per-label deltas / 12; "core" = same over the
#   7 labels without ACL (degenerative, rare on OAI) and Synovitis (Hoffa; the OAI-free models' Syn output ~ their Eff output).
# Weighting schemes (knee resampling, predictions given labels kept as observed):
#   raw    : OAI as is
#   R1oa3  : skeptic R1 (3-label OA cell matched to gold74 71 complete; others free)
#   R9     : raking over ALL 36 pairwise 2x2 tables of the 9 labels (knees where both labels exist): OAI marginals + gold74 log
#            odds ratio (+.5 smoothed); coupling only. R9hi / R9lo: every gold74 log OR +/- 1 sd (stress tests).
import os, sys, json, itertools
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from scipy.optimize import brentq
H = os.path.expanduser('~/oai_build')
G74P = {"ACL | Medial Meniscus": [24, 20, 14, 15], "ACL | Lateral Meniscus": [26, 19, 18, 11], "ACL | Medial OA": [26, 17, 24, 5], "ACL | Lateral OA": [34, 8, 23, 6], "ACL | PF OA": [20, 24, 22, 7], "ACL | Effusion": [19, 26, 7, 22], "ACL | Synovitis": [22, 23, 15, 13], "ACL | Baker's": [28, 13, 21, 5], "Medial Meniscus | Lateral Meniscus": [25, 13, 19, 16], "Medial Meniscus | Medial OA": [33, 4, 16, 18], "Medial Meniscus | Lateral OA": [30, 7, 26, 7], "Medial Meniscus | PF OA": [26, 12, 16, 18], "Medial Meniscus | Effusion": [17, 21, 8, 27], "Medial Meniscus | Synovitis": [22, 16, 14, 20], "Medial Meniscus | Baker's": [30, 5, 18, 13], "Lateral Meniscus | Medial OA": [36, 7, 14, 15], "Lateral Meniscus | Lateral OA": [39, 3, 18, 11], "Lateral Meniscus | PF OA": [27, 17, 15, 14], "Lateral Meniscus | Effusion": [18, 26, 8, 22], "Lateral Meniscus | Synovitis": [25, 18, 12, 18], "Lateral Meniscus | Baker's": [33, 7, 16, 11], "Medial OA | Lateral OA": [45, 5, 12, 9], "Medial OA | PF OA": [36, 14, 6, 16], "Medial OA | Effusion": [20, 30, 5, 17], "Medial OA | Synovitis": [28, 21, 8, 14], "Medial OA | Baker's": [42, 6, 7, 11], "Lateral OA | PF OA": [37, 20, 5, 9], "Lateral OA | Effusion": [23, 34, 2, 12], "Lateral OA | Synovitis": [32, 24, 3, 11], "Lateral OA | Baker's": [44, 10, 5, 6], "PF OA | Effusion": [16, 26, 10, 21], "PF OA | Synovitis": [25, 17, 12, 18], "PF OA | Baker's": [33, 7, 16, 10], "Effusion | Synovitis": [20, 5, 17, 31], "Effusion | Baker's": [20, 4, 29, 14], "Synovitis | Baker's": [29, 5, 20, 13]}
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
L9 = ["ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's"]
OA = ['Medial OA', 'Lateral OA', 'PF OA']
CORE = [c for c in L9 if c not in ('ACL', 'Synovitis')]
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
S = pd.read_csv(H + '/oai_label_sources.csv', dtype={'kid': str}).set_index('kid')
B1 = pd.DataFrame(index=T.index); B2 = pd.DataFrame(index=T.index)
thr = lambda c, x: np.where(T[c].notna(), (T[c] >= x - 1e-6).astype(float), np.nan)
B1['ACL'] = thr('ACL', .5); B2['ACL'] = thr('ACL', 1.)
for c in ('Medial Meniscus', 'Lateral Meniscus'):
    B1[c] = thr(c, 1.); B2[c] = thr(c, .2)
for c in OA:
    cc = S.reindex(T.index)[['m_cm_' + r for r in COMP[c]]].astype(float).values
    area = np.floor(cc + 1e-6); ft = np.round((cc - area) * 10); fin = np.isfinite(cc)
    y = ((ft >= 1) & fin).any(1).astype(float); y[~fin.any(1)] = np.nan
    B1[c] = y; B2[c] = thr(c, .5)
B1['Effusion'] = thr('Effusion', .8); B2['Effusion'] = thr('Effusion', .3)
B1['Synovitis'] = thr('Synovitis', 1 / 3); B2['Synovitis'] = thr('Synovitis', 2 / 3)
B1["Baker's"] = thr("Baker's", 1.); B2["Baker's"] = thr("Baker's", 1.)
P = {}
for k, f in MODELS.items():
    d = pd.read_csv(f, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
    P[k] = d[L]
common = T.index
for k in P: common = common.intersection(P[k].dropna().index)
n = len(common)
Y1 = B1.loc[common, L9].values; Y2 = B2.loc[common, L9].values
OBS = np.isfinite(Y1)
print('knees %d | B1 coverage/prev %s' % (n, {c: (int(OBS[:, j].sum()), round(float(np.nanmean(Y1[:, j])), 3)) for j, c in enumerate(L9)}))
sys.stdout.flush()
J9 = [L.index(c) for c in L9]; JOA = [L.index(c) for c in OA]


def transforms(F):
    U = np.stack([pct(F[:, k]) for k in range(12)], 1)
    S9 = U[:, J9].sum(1); S12 = U.sum(1); out = {}
    def eoa(a):
        X = U.copy(); so = U[:, JOA].sum(1)
        for j in JOA: X[:, j] = U[:, j] - a * (so - U[:, j]) / 2
        return X
    for a in (.1, .2):
        out['EOA %.1f' % a] = eoa(a)
    for a in (-.1, .1, .2):
        X = U.copy()
        for j in J9: X[:, j] = U[:, j] - a * (S9 - U[:, j]) / 8
        out['G9 %+.1f' % a] = X
        X = U.copy()
        for j in J9: X[:, j] = U[:, j] - a * (S12 - U[:, j]) / 11
        out['G12 %+.1f' % a] = X
    E1 = eoa(.1)
    for a in (-.1, .1, .2):
        X = E1.copy()
        for j in J9:
            if j not in JOA: X[:, j] = U[:, j] - a * (S9 - U[:, j]) / 8
        out['E1+N9 %+.1f' % a] = X
    for a in (.1, .2):
        X = E1.copy()
        for j in J9:
            if j not in JOA: X[:, j] = U[:, j] - a * (S12 - U[:, j]) / 11
        out['E1+N12 %+.1f' % a] = X
    return U, out


# ---------------- coupling-matched weights ----------------
def or_cells(p1, p2, lor):
    lo = max(0., p1 + p2 - 1) + 1e-9; hi = min(p1, p2) - 1e-9
    g = lambda p11: np.log(p11) + np.log(1 - p1 - p2 + p11) - np.log(p1 - p11) - np.log(p2 - p11) - lor
    p11 = brentq(g, lo, hi)
    return np.array([[1 - p1 - p2 + p11, p2 - p11], [p1 - p11, p11]])


def target_lor(i, j, shift):
    c = np.array(G74P[L9[i] + ' | ' + L9[j]], float) + .5      # [00, 01, 10, 11], L9[i] first
    return np.log(c[3] * c[0] / (c[1] * c[2])) + shift * np.sqrt((1 / c).sum())


def rake9(shift, sweeps=400):
    w = np.ones(n); Yb = np.nan_to_num(Y1).astype(int)
    tg = {}
    for i, j in itertools.combinations(range(9), 2):
        m = OBS[:, i] & OBS[:, j]
        tg[(i, j)] = (m, or_cells(Yb[m, i].mean(), Yb[m, j].mean(), target_lor(i, j, shift)))
    for _ in range(sweeps):
        for (i, j), (m, t) in tg.items():
            W = w[m].sum()
            for a in (0, 1):
                for b in (0, 1):
                    sel = m & (Yb[:, i] == a) & (Yb[:, j] == b); cur = w[sel].sum()
                    if cur > 0: w[sel] *= t[a, b] * W / cur
    # convergence report: weighted log OR vs target, over the 36 pairs
    dev = []
    for (i, j), (m, t) in tg.items():
        tab = np.array([[w[m & (Yb[:, i] == a) & (Yb[:, j] == b)].sum() for b in (0, 1)] for a in (0, 1)])
        dev.append(np.log(tab[1, 1] * tab[0, 0] / (tab[0, 1] * tab[1, 0])) - np.log(t[1, 1] * t[0, 0] / (t[0, 1] * t[1, 0])))
    dev = np.abs(dev)
    return w, float(dev.mean()), float(dev.max())


def r1_oa3():
    io = [L9.index(c) for c in OA]; m = OBS[:, io].all(1); Yb = np.nan_to_num(Y1).astype(int)
    k3 = Yb[:, io[0]] * 4 + Yb[:, io[1]] * 2 + Yb[:, io[2]]
    po = np.bincount(k3[m], minlength=8) / m.sum()
    G74OA = {'000': 33, '001': 12, '010': 3, '011': 2, '100': 4, '101': 8, '110': 2, '111': 7}
    pg = np.array([G74OA[format(i, '03b')] + .5 for i in range(8)]); pg /= pg.sum()
    w = np.where(m, (pg / np.maximum(po, 1e-12))[k3], 1.0)
    return w


def wphi(w, i, j):
    m = OBS[:, i] & OBS[:, j]; x = Y1[m, i]; y = Y1[m, j]; ww = w[m] / w[m].sum()
    mx = (ww * x).sum(); my = (ww * y).sum(); cov = (ww * (x - mx) * (y - my)).sum()
    return cov / np.sqrt((ww * (x - mx) ** 2).sum() * (ww * (y - my) ** 2).sum())


SCHEMES = {'raw': (np.ones(n), None), 'R1oa3': (r1_oa3(), None)}
for nm, sh in (('R9', 0.), ('R9hi', 1.), ('R9lo', -1.)):
    w, dm, dx = rake9(sh); SCHEMES[nm] = (w, (dm, dx))
pairs = list(itertools.combinations(range(9), 2))
for nm, (w, cv) in SCHEMES.items():
    p = w / w.sum(); ph = np.array([wphi(w, i, j) for i, j in pairs])
    oa = [k for k, (i, j) in enumerate(pairs) if L9[i] in OA and L9[j] in OA]
    noa = [k for k in range(len(pairs)) if k not in oa]
    print('scheme %-5s ESS %5.0f | raking |dlogOR| mean/max %s | weighted label phi: mean over 36 pairs %+.3f, OA-trio 3 %+.3f, other 33 %+.3f' % (
        nm, 1 / (p ** 2).sum(), ('%.3f/%.3f' % cv) if cv else '-', ph.mean(), ph[oa].mean(), ph[noa].mean()))
sys.stdout.flush()

# ---------------- model coupling (aggregate) ----------------
for mk, Pm in P.items():
    F = Pm.loc[common, L].values.astype(float)
    R = spearmanr(F[:, J9]).correlation
    rr = np.array([R[i, j] for i, j in pairs])
    oa = [k for k, (i, j) in enumerate(pairs) if L9[i] in OA and L9[j] in OA]
    print('[%s] model Spearman over the 9 covered labels: mean 36 pairs %+.3f, OA-trio %+.3f, other 33 %+.3f' % (
        mk, rr.mean(), rr[oa].mean(), np.delete(rr, oa).mean()))
sys.stdout.flush()

# ---------------- evaluation ----------------
RES = {}
for mk, Pm in P.items():
    F = Pm.loc[common, L].values.astype(float)
    U, TR = transforms(F); RES[mk] = {}
    chg = {t: [k for k, j in enumerate(J9) if not np.allclose(X[:, j], U[:, j])] for t, X in TR.items()}
    for sn, (w, _) in list(SCHEMES.items()) + [('raw_B2', (np.ones(n), None))]:
        Y = Y2 if sn == 'raw_B2' else Y1
        p = w / w.sum(); rng = np.random.default_rng(11)
        D = {t: [] for t in TR}; C = {t: [] for t in TR}; PL = {t: [] for t in TR}
        BB = B if sn in ('raw', 'R9', 'raw_B2') else max(150, B // 2)
        for b in range(BB):
            ii = rng.choice(n, n, p=p)
            subs = []
            ok = True
            for k in range(9):
                s = ii[np.isfinite(Y[ii, k])]; y = Y[s, k].astype(int)
                if y.sum() < 5 or (len(y) - y.sum()) < 5: ok = False; break
                subs.append((s, y))
            if not ok: continue
            base = [auc(U[s, J9[k]], y) for k, (s, y) in enumerate(subs)]
            for t, X in TR.items():
                d = np.zeros(9)
                for k in chg[t]:
                    s, y = subs[k]; d[k] = auc(X[s, J9[k]], y) - base[k]
                D[t].append(d.sum() / 12); C[t].append(sum(d[L9.index(c)] for c in CORE) / 12); PL[t].append(d)
        R0 = {}
        for t in TR:
            dd = np.array(D[t]); cc = np.array(C[t]); inc = dd - np.array(D['EOA 0.1']); incc = cc - np.array(C['EOA 0.1'])
            R0[t] = {'macro': float(dd.mean()), 'P': float((dd > 0).mean()), 'core': float(cc.mean()), 'Pcore': float((cc > 0).mean()),
                     'inc': float(inc.mean()), 'Pinc': float((inc > 0).mean()), 'inc_core': float(incc.mean()), 'Pinc_core': float((incc > 0).mean()),
                     'per': np.round(np.mean(PL[t], 0), 4).tolist(), 'nb': len(dd)}
        RES[mk][sn] = R0
        print('[%s | %s] boots %d' % (mk, sn, R0['EOA 0.1']['nb']))
        for t, v in R0.items():
            print('   %-12s macro %+.5f (P %.2f) core %+.5f (P %.2f) | vs EOA.1 %+.5f (P %.2f) core %+.5f (P %.2f) | per %s' % (
                t, v['macro'], v['P'], v['core'], v['Pcore'], v['inc'], v['Pinc'], v['inc_core'], v['Pinc_core'],
                v['per']))
        sys.stdout.flush()
print('per-label order: %s' % L9)
print('JSON_BEGIN'); print(json.dumps(RES)); print('JSON_END')
