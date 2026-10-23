# box_curve_ph.py (10-20 ex, Claude): POST-HOC follow-up of box_curve.py (oracle showed Lateral OA wants its weight on PF OA,
# not Medial OA, in every model and both proxies). NOT pre-registered; reported as post-hoc. Same read-only, aggregates-only rules.
# box_curve.py header follows.
# box_curve.py (10-20 ex, Claude): E-OA curve + asymmetric refinements on OAI, coupling-matched to gold74. Read-only analysis.
# Prints AGGREGATES ONLY (AUC deltas, sd, P, ESS, phi, rho, q, n, prevalence). Never writes a file, never prints an id, a row
# or a per-knee value. Piped via ssh stdin:  ssh training@192.168.10.30 '~/miniconda3/envs/rsna/bin/python -I - JOB' < box_curve.py
# JOB in {D5, s1044, s1045, ENS3, FOLDS, AREA}. Pre-registration: cand_eoa_curve.md sec 0.
# Transform: U = within-set percentiles; out_t = U_t - b1 * U_o1 - b2 * U_o2 (o1 < o2 = the other two OA compartments).
# Symmetric E-OA(a) = b1 = b2 = a/2. Each label's AUC depends only on its own (b1, b2), so multi-label variants are a
# per-label column choice and the OA-trio delta on a resample is the sum of per-label deltas.
import os, sys, glob, json
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
from scipy.optimize import brentq

H = os.path.expanduser('~/oai_build')
OA = ['Medial OA', 'Lateral OA', 'PF OA']
COMP = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'],
        'PF OA': ['mp', 'lp', 'amf', 'alf']}
MODELS = {'D5': H + '/teacher/teacher_D5_mean_v00.csv', 's1044': H + '/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv',
          's1045': H + '/teacher2/f0_s1045_ttr288frac75__raw__A__gpu.csv'}
FOLDGLOB = H + '/teacher/f{k}_s1043_ttr_frac75__raw__A__shard*of4.csv'
# gold74 aggregates (local, no OAI content): pairwise 2x2 counts [a0b0, a0b1, a1b0, a1b1] and 3-label OA cells (M/L/PF bits)
G74P = {(0, 1): [45, 5, 12, 9], (0, 2): [36, 14, 6, 16], (1, 2): [37, 20, 5, 9]}
G74OA = {'000': 33, '001': 12, '010': 3, '011': 2, '100': 4, '101': 8, '110': 2, '111': 7}
# gold74 excess-gap pair weights (coupling map: LOA-PF .42, MOA-LOA .26, MOA-PF .21), weight = gap / mean gap
GAP = {(0, 1): .26, (0, 2): .21, (1, 2): .42}; mg = np.mean(list(GAP.values())); PW = {k: v / mg for k, v in GAP.items()}
GRID = [0, .025, .05, .075, .1, .125, .15]
pct = lambda v: rankdata(v) / len(v)
JOB = sys.argv[1] if len(sys.argv) > 1 else 'D5'
BSCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0   # smoke test: small fraction of B


def oth(t):
    return [m for m in range(3) if m != t]


# per-label columns: name -> {t: (b1, b2)}
COLS = {}
for i, b1 in enumerate(GRID):
    for j, b2 in enumerate(GRID):
        COLS['g%d%d' % (i, j)] = {t: (b1, b2) for t in range(3)}
COLS['ctrl'] = {t: (-.05, -.05) for t in range(3)}
for a in (.1, .2):
    COLS['PW%02d' % round(a * 100)] = {t: tuple(a / 2 * PW[tuple(sorted((t, o)))] for o in oth(t)) for t in range(3)}
for a in (.1, .15, .2):
    k = round(a * 100)
    COLS['Lpf%02d' % k] = {0: (0., 0.), 1: (0., a), 2: (0., 0.)}      # used for label 1 only: Lateral <- PF only, weight a
    COLS['PFl%02d' % k] = {0: (0., 0.), 1: (0., 0.), 2: (0., a)}      # used for label 2 only: PF <- Lateral only, weight a
CN = list(COLS)


def sym(a):
    k = int(round(a / .05)); return 'g%d%d' % (k, k)


VAR = {}
for a in (.05, .1, .15, .2, .25, .3):
    VAR['S%.2f' % a] = [sym(a)] * 3
VAR['S-0.10'] = ['ctrl'] * 3
for nm, v in {'LP10': (0, .1, .1), 'LP15': (0, .15, .15), 'LP20': (0, .2, .2), 'RAMP': (.05, .1, .15), 'RAMP2': (.1, .15, .2),
              'PFH': (.1, .1, .2), 'PFH2': (.05, .1, .2), 'LH': (.1, .2, .1), 'MH_ctrl': (.2, .1, .1)}.items():
    VAR[nm] = [sym(x) for x in v]
VAR['PW10'] = ['PW10'] * 3; VAR['PW20'] = ['PW20'] * 3
for a in (.1, .15, .2):
    k = round(a * 100)
    VAR['PH_L%02d' % k] = [sym(a), 'Lpf%02d' % k, sym(a)]            # symmetric a, but Lateral's whole weight a on PF
    VAR['PH_PAIR%02d' % k] = ['g00', 'Lpf%02d' % k, 'PFl%02d' % k]     # only the LOA-PF pair decorrelated, weight a each way
    VAR['PH_PAIRh%02d' % k] = [sym(a), 'Lpf%02d' % k, 'PFl%02d' % k]   # Medial symmetric a; L and PF only against each other


def auc_cols(S, y):
    # S (m, k) scores, y (m,) binary -> (k,) AUCs
    R = rankdata(S, axis=0); p = y.sum(); q = len(y) - p
    return (R[y == 1].sum(0) - p * (p + 1) / 2) / (p * q)


def outcols(U, ii):
    # per label t: (m, ncol) matrix of transformed scores on resample ii
    out = []
    for t in range(3):
        o1, o2 = oth(t); B1 = np.array([COLS[c][t][0] for c in CN]); B2 = np.array([COLS[c][t][1] for c in CN])
        out.append(U[ii, t][:, None] - U[ii, o1][:, None] * B1[None] - U[ii, o2][:, None] * B2[None])
    return out


def deltas(U, Y, p, B, seed, idx=None):
    # returns D (B', 3, ncol): per-label AUC(col) - AUC(untransformed) on weighted resamples
    base_col = CN.index('g00'); rng = np.random.default_rng(seed); pool = np.arange(len(Y)) if idx is None else idx
    pp = p[pool] / p[pool].sum(); D = []
    for _ in range(B):
        ii = pool[rng.choice(len(pool), len(pool), p=pp)]; y = Y[ii]
        if (y.sum(0) < 5).any() or ((len(ii) - y.sum(0)) < 5).any():
            continue
        O = outcols(U, ii)
        A_ = [auc_cols(O[t], y[:, t]) for t in range(3)]
        D.append(np.stack([A_[t] - A_[t][base_col] for t in range(3)]))
    return np.array(D)


def vsum(D, cols):
    return sum(D[:, t, CN.index(cols[t])] for t in range(3))


def stat(x):
    return {'m': float(np.mean(x)), 'sd': float(np.std(x)), 'P': float(np.mean(x > 0))}


def proxy(name):
    s = pd.read_csv(H + '/oai_label_sources.csv', dtype={'kid': str}).set_index('kid')
    Y = pd.DataFrame(index=s.index, columns=OA, dtype=float)
    for lab, regs in COMP.items():
        c = s[['m_cm_' + r for r in regs]].astype(float).values
        area = np.floor(c + 1e-6); ft = np.round((c - area) * 10); fin = np.isfinite(c)
        pos = ((ft >= 1) & fin) if name == 'P_FT' else ((area >= 2) & fin)
        y = pos.any(1).astype(float); y[~fin.any(1)] = np.nan; Y[lab] = y
    return Y.dropna()


def or_cells(p1, p2, lor):
    lo = max(0., p1 + p2 - 1) + 1e-9; hi = min(p1, p2) - 1e-9
    g = lambda p11: np.log(p11) + np.log(1 - p1 - p2 + p11) - np.log(p1 - p11) - np.log(p2 - p11) - lor
    p11 = brentq(g, lo, hi)
    return np.array([[1 - p1 - p2 + p11, p2 - p11], [p1 - p11, p11]])


def schemes(Y):
    n = len(Y); k3 = Y[:, 0] * 4 + Y[:, 1] * 2 + Y[:, 2]; p0 = np.bincount(k3, minlength=8) / n
    pg = np.array([G74OA[format(i, '03b')] + .5 for i in range(8)]); pg /= pg.sum()
    out = {'raw': np.ones(n), 'R1oa3': (pg / np.maximum(p0, 1e-12))[k3]}
    cells = np.array([[(c >> (2 - i)) & 1 for i in range(3)] for c in range(8)]); marg = Y.mean(0)
    for nm, sh in (('R3p', 0.), ('R3p_hi', 1.), ('R3p_lo', -1.)):
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
        out[nm] = np.where(p0 > 0, q / np.maximum(p0, 1e-12), 0)[k3]
    return out


def describe(U, Y, w, tag):
    p = w / w.sum(); rng = np.random.default_rng(99); ii = rng.choice(len(Y), 30000, p=p)
    y = Y[ii]; Z = np.stack([norm.ppf(np.clip(U[ii, k], 1e-4, 1 - 1e-4)) for k in range(3)], 1)
    ph = np.corrcoef(y.T); rz = np.corrcoef(Z.T)
    q = []
    for k in range(3):
        ck = np.corrcoef(Z[:, k], y[:, k])[0, 1]; q.append(np.mean([np.corrcoef(Z[:, j], y[:, k])[0, 1] for j in oth(k)]) / ck)
    return {'ESS': float(1 / (p ** 2).sum()), 'prev': np.round(y.mean(0), 3).tolist(),
            'phi': [round(ph[0, 1], 3), round(ph[0, 2], 3), round(ph[1, 2], 3)],
            'rho': [round(rz[0, 1], 3), round(rz[0, 2], 3), round(rz[1, 2], 3)], 'q': np.round(q, 3).tolist()}


def report(D, tag):
    res = {}
    S1 = vsum(D, VAR['S0.10']); S2 = vsum(D, VAR['S0.20'])
    for nm, cols in VAR.items():
        x = vsum(D, cols)
        res[nm] = {'macro': stat(x / 12), 'inc1': stat((x - S1) / 12), 'inc2': stat((x - S2) / 12),
                   'per': [float(D[:, t, CN.index(cols[t])].mean()) for t in range(3)]}
    # per-label oracle over the 49 grid columns
    gi = [c for c in CN if c.startswith('g')]
    best = []
    for t in range(3):
        mm = np.array([D[:, t, CN.index(c)].mean() for c in gi]); c = gi[int(np.argmax(mm))]
        best.append(c)
    x = vsum(D, best)
    res['ORACLE'] = {'macro': stat(x / 12), 'inc1': stat((x - S1) / 12), 'inc2': stat((x - S2) / 12),
                     'per': [float(D[:, t, CN.index(best[t])].mean()) for t in range(3)],
                     'b': [COLS[best[t]][t] for t in range(3)]}
    # per-label symmetric curve (AUC units)
    res['curve_per_label'] = {('%.2f' % a): [float(D[:, t, CN.index(sym(a))].mean()) for t in range(3)] for a in (.05, .1, .15, .2, .25, .3)}
    res['curve_per_label']['-0.10'] = [float(D[:, t, CN.index('ctrl')].mean()) for t in range(3)]
    res['nboot'] = int(len(D))
    print('[%s] B %d' % (tag, len(D)))
    for nm in list(VAR) + ['ORACLE']:
        r = res[nm]; flag = ' PASS' if (r['inc1']['m'] > r['inc1']['sd'] and r['inc2']['m'] > r['inc2']['sd']) else ''
        print('   %-8s macro %+.5f (sd %.5f P %.2f) | inc vs S.1 %+.5f (sd %.5f P %.2f) | vs S.2 %+.5f (sd %.5f P %.2f) | per-label AUC %s%s%s' % (
            nm, r['macro']['m'], r['macro']['sd'], r['macro']['P'], r['inc1']['m'], r['inc1']['sd'], r['inc1']['P'],
            r['inc2']['m'], r['inc2']['sd'], r['inc2']['P'], np.round(r['per'], 4).tolist(),
            (' b ' + str([tuple(round(v, 3) for v in b) for b in r['b']])) if nm == 'ORACLE' else '', flag), flush=True)
    return res


def split_half(U, Y, w, B, seed):
    rng = np.random.default_rng(11); perm = rng.permutation(len(Y)); A, Bh = perm[: len(Y) // 2], perm[len(Y) // 2:]
    out = {}
    for nmA, nmB in ((A, Bh), (Bh, A)):
        DA = deltas(U, Y, w, B, seed, nmA); gi = [c for c in CN if c.startswith('g')]
        best = [gi[int(np.argmax([DA[:, t, CN.index(c)].mean() for c in gi]))] for t in range(3)]
        DB = deltas(U, Y, w, B, seed + 1, nmB)
        x = vsum(DB, best); S1 = vsum(DB, VAR['S0.10']); S2 = vsum(DB, VAR['S0.20'])
        key = 'AtoB' if nmA is A else 'BtoA'
        out[key] = {'b': [COLS[best[t]][t] for t in range(3)], 'inc1': stat((x - S1) / 12), 'inc2': stat((x - S2) / 12)}
    return out


def load(p):
    return pd.read_csv(p, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')[OA]


def main():
    P = {k: load(f) for k, f in MODELS.items()}
    for k in range(5):
        fs = sorted(glob.glob(FOLDGLOB.format(k=k)))
        P['f%d' % k] = pd.concat([pd.read_csv(f, dtype={'StudyInstanceUID': str}) for f in fs]).drop_duplicates(
            'StudyInstanceUID').set_index('StudyInstanceUID')[OA]
    prox = 'P_AREA' if JOB == 'PH_AREA' else 'P_FT'
    Yd = proxy(prox); ids = Yd.index
    for k in P: ids = ids.intersection(P[k].dropna().index)
    Y = Yd.loc[ids].values.astype(int); n = len(Y)
    UU = {k: np.stack([pct(P[k].loc[ids].values[:, c].astype(float)) for c in range(3)], 1) for k in P}
    UU['ENS3'] = np.stack([pct((UU['D5'][:, c] + UU['s1044'][:, c] + UU['s1045'][:, c]) / 3) for c in range(3)], 1)
    W = schemes(Y)
    print('JOB %s | proxy %s | n %d | prevalence %s' % (JOB, prox, n, np.round(Y.mean(0), 3).tolist()))
    if JOB.startswith('PH_FT'):
        grp = {'PH_FT_A': ['D5', 's1044', 's1045'], 'PH_FT_B': ['ENS3', 'f0', 'f1'], 'PH_FT_C': ['f2', 'f3', 'f4']}[JOB]
        mods, sch, B, sh = grp, ['raw', 'R1oa3', 'R3p', 'R3p_hi', 'R3p_lo'], 300, False
    else:
        mods, sch, B, sh = ['D5', 's1044', 's1045', 'ENS3'], ['raw', 'R1oa3', 'R3p'], 200, False
    ALL = {}
    for m in mods:
        ALL[m] = {}
        for si, s in enumerate(sch):
            d = describe(UU[m], Y, W[s], m + ' ' + s)
            print('[%s | %s | %s] ESS %.0f | prev %s | label phi ML/MPF/LPF %s | model rho(ns) %s | q %s' % (
                m, s, prox, d['ESS'], d['prev'], d['phi'], d['rho'], d['q']), flush=True)
            D = deltas(UU[m], Y, W[s], max(5, int(B * BSCALE)), 1000 + si)
            r = report(D, '%s | %s | %s' % (m, s, prox)); r['desc'] = d
            if sh and s in ('R1oa3', 'R3p'):
                r['split'] = split_half(UU[m], Y, W[s], max(5, int(200 * BSCALE)), 2000 + si)
                for kk, v in r['split'].items():
                    print('   split-half ORACLE %s: b %s | inc vs S.1 %+.5f (sd %.5f P %.2f) | vs S.2 %+.5f (sd %.5f P %.2f)' % (
                        kk, [tuple(round(x, 3) for x in b) for b in v['b']], v['inc1']['m'], v['inc1']['sd'], v['inc1']['P'],
                        v['inc2']['m'], v['inc2']['sd'], v['inc2']['P']), flush=True)
            ALL[m][s] = r
    print('JSON_BEGIN'); print(json.dumps({'job': JOB, 'proxy': prox, 'n': n, 'res': ALL})); print('JSON_END')


main()
