# box_mech.py (10-20 more, Claude): E-OA ensemble mechanics on OAI with OAI-free stand-in members. Read-only analysis.
# Prints AGGREGATES ONLY (AUC deltas, sd, P, ESS, phi, rho, q, n, prevalence, a*). Never writes a file, never prints an id,
# a row or a per-knee value. Piped via ssh stdin:
#   ssh training@192.168.10.30 '~/miniconda3/envs/rsna/bin/python -I - JOB [BSCALE]' < box_mech.py
# JOB L12 = lens 1 (E1 pow2 x E-OA) + lens 2 (placement FINAL / V224 / MEMBER) on 4 stand-in configs.
# JOB L3  = lens 3 (a* vs number of blended members, 120 flat blends of 7 single models).
# Pre-registration: mechanics.md sec 0. E-OA as the kernel: U = pct(col); out_t = pct(U_t - a (U_o1 + U_o2) / 2).
import os, sys, glob, json, itertools, time
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
from scipy.optimize import brentq

T0 = time.time()
H = os.path.expanduser('~/oai_build')
L9 = ["ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's"]
OA = ['Medial OA', 'Lateral OA', 'PF OA']
IOA = [L9.index(c) for c in OA]
INON = [j for j in range(9) if j not in IOA]
COMP = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'],
        'PF OA': ['mp', 'lp', 'amf', 'alf']}
FOLDGLOB = H + '/teacher/f{k}_s1043_ttr_frac75__raw__A__shard*of4.csv'
SINGLE = {'s1044': H + '/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv', 's1045': H + '/teacher2/f0_s1045_ttr288frac75__raw__A__gpu.csv',
          'D5': H + '/teacher/teacher_D5_mean_v00.csv'}
G74P = {(0, 1): [45, 5, 12, 9], (0, 2): [36, 14, 6, 16], (1, 2): [37, 20, 5, 9]}
G74OA = {'000': 33, '001': 12, '010': 3, '011': 2, '100': 4, '101': 8, '110': 2, '111': 7}
C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4}
WV = np.array([C4W.get(c, .5) for c in L9])
AG = [.05, .1, .15, .2, .25, .3]
JOB = sys.argv[1] if len(sys.argv) > 1 else 'L12'
BSCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0


def pctc(X):
    X = np.asarray(X, np.float64)
    return rankdata(X, axis=0) / X.shape[0]


def eoa(U3, a):
    U = pctc(U3); out = np.empty_like(U)
    for t in range(3):
        o = [k for k in range(3) if k != t]
        out[:, t] = pctc(U[:, t] - a * (U[:, o[0]] + U[:, o[1]]) / 2.0)
    return out


def auc_cols(S, y):
    R = rankdata(S, axis=0); p = y.sum(); q = len(y) - p
    return (R[y == 1].sum(0) - p * (p + 1) / 2) / (p * q)


def wauc(s, y, w):
    u, inv = np.unique(s, return_inverse=True)
    pw = np.bincount(inv, weights=w * (y == 1), minlength=len(u)); nw = np.bincount(inv, weights=w * (y == 0), minlength=len(u))
    cn = np.cumsum(nw) - nw
    return float((pw * (cn + .5 * nw)).sum() / (pw.sum() * nw.sum()))


def stat(x):
    x = np.asarray(x); return {'m': float(np.mean(x)), 'sd': float(np.std(x)), 'P': float(np.mean(x > 0))}


def fmt(s):
    return '%+.5f (sd %.5f P %.2f)' % (s['m'], s['sd'], s['P'])


# ---------------- labels ----------------
def labels():
    T = pd.read_csv(H + '/oai_labels_T123.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    S = pd.read_csv(H + '/oai_label_sources.csv', dtype={'kid': str}).set_index('kid')
    Y = pd.DataFrame(index=T.index, columns=L9, dtype=float)
    thr = lambda c, x: np.where(T[c].notna(), (T[c] >= x - 1e-6).astype(float), np.nan)
    Y['ACL'] = thr('ACL', .5)
    for c in ('Medial Meniscus', 'Lateral Meniscus'):
        Y[c] = thr(c, 1.)
    Sx = S.reindex(T.index)
    for c in OA:
        cc = Sx[['m_cm_' + r for r in COMP[c]]].astype(float).values
        area = np.floor(cc + 1e-6); ft = np.round((cc - area) * 10); fin = np.isfinite(cc)
        y = ((ft >= 1) & fin).any(1).astype(float); y[~fin.any(1)] = np.nan; Y[c] = y
    Y['Effusion'] = thr('Effusion', .8); Y['Synovitis'] = thr('Synovitis', 1 / 3); Y["Baker's"] = thr("Baker's", 1.)
    return Y


def load(f):
    return pd.read_csv(f, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')[L9]


def members():
    P = {}
    for k in range(5):
        fs = sorted(glob.glob(FOLDGLOB.format(k=k)))
        P['f%d' % k] = pd.concat([pd.read_csv(f, dtype={'StudyInstanceUID': str}) for f in fs]).drop_duplicates(
            'StudyInstanceUID').set_index('StudyInstanceUID')[L9]
    for k, f in SINGLE.items():
        P[k] = load(f)
    return P


# ---------------- coupling-matched weights (as box_curve.py) ----------------
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


def wcorr(a, b, p):
    ma = (p * a).sum(); mb = (p * b).sum()
    return float((p * (a - ma) * (b - mb)).sum() / np.sqrt((p * (a - ma) ** 2).sum() * (p * (b - mb) ** 2).sum()))


def coupling(U3, Y, w):
    # weighted normal-score coupling of a 3-column OA score set: rho (3 pairs), q (3 labels), gap = mean rho - mean q
    p = w / w.sum(); Z = norm.ppf(np.clip(pctc(U3) - .5 / len(U3), 1e-6, 1 - 1e-6))
    rho = [wcorr(Z[:, 0], Z[:, 1], p), wcorr(Z[:, 0], Z[:, 2], p), wcorr(Z[:, 1], Z[:, 2], p)]
    q = []
    for k in range(3):
        ck = wcorr(Z[:, k], Y[:, k].astype(float), p)
        q.append(np.mean([wcorr(Z[:, j], Y[:, k].astype(float), p) for j in range(3) if j != k]) / ck)
    return {'rho': rho, 'q': q, 'gap': float(np.mean(rho) - np.mean(q)), 'ESS': float(1 / (p ** 2).sum())}


# ---------------- blends ----------------
def nested(M, e, place, a):
    P = {k: pctc(M[k]) for k in M}
    if place == 'MEMBER' and a:
        for k in P:
            P[k][:, IOA] = eoa(P[k][:, IOA], a)
    rr = .75 * P['r384'] ** e + .25 * P['r224'] ** e
    V = WV * rr + (1 - WV) * P['R6'] ** e
    rb = pctc(V)
    if place == 'V224' and a:
        rb[:, IOA] = eoa(rb[:, IOA], a)
    out = .8 * rb ** e + .2 * P['S6'] ** e
    if place == 'FINAL' and a:
        out[:, IOA] = eoa(out[:, IOA], a)
    return out


def flat(Ms, e, place, a):
    P = [pctc(X) for X in Ms]
    if place == 'MEMBER' and a:
        for X in P:
            X[:, IOA] = eoa(X[:, IOA], a)
    out = np.mean([X ** e for X in P], 0)
    if place == 'FINAL' and a:
        out[:, IOA] = eoa(out[:, IOA], a)
    return out


def setup():
    Yd = labels(); P = members()
    ids = Yd.index[Yd[OA].notna().all(1)]
    for k in P:
        ids = ids.intersection(P[k].dropna().index)
    ids = sorted(ids)
    Y = Yd.loc[ids].values; F = {k: P[k].loc[ids].values.astype(np.float64) for k in P}
    return Y, F


def job_l12():
    Y, F = setup(); n = len(Y); Yoa = Y[:, IOA].astype(int); Ynon = Y[:, INON]
    print('JOB L12 | n %d | OA prevalence %s | non-OA coverage/prev %s' % (
        n, np.round(Yoa.mean(0), 3).tolist(),
        {L9[j]: (int(np.isfinite(Y[:, j]).sum()), round(float(np.nanmean(Y[:, j])), 3)) for j in INON}), flush=True)
    pm = lambda ks: np.mean([F[k] for k in ks], 0)
    CFG = {'ROT1': {'R6': pm(['f0', 'f1', 'f2']), 'r384': F['s1044'], 'r224': F['s1045'], 'S6': pm(['f3', 'f4'])},
           'ROT2': {'R6': F['s1044'], 'r384': pm(['f0', 'f1', 'f2']), 'r224': pm(['f3', 'f4']), 'S6': F['s1045']},
           'ROT3': {'R6': F['s1045'], 'r384': pm(['f0', 'f1', 'f2']), 'r224': pm(['f3', 'f4']), 'S6': F['s1044']},
           'FLAT3': [F['D5'], F['s1044'], F['s1045']]}
    W = schemes(Yoa)
    B = max(5, int(400 * BSCALE)); ALL = {}
    for cn, M in CFG.items():
        isflat = cn == 'FLAT3'; places = ['FINAL', 'MEMBER'] if isflat else ['FINAL', 'V224', 'MEMBER']
        f = (lambda e, pl, a: flat(M, e, pl, a)) if isflat else (lambda e, pl, a: nested(M, e, pl, a))
        cols = [(e, 'NONE', 0.) for e in (1, 2)] + [(e, pl, a) for e in (1, 2) for pl in places for a in AG]
        S = {c: f(*c) for c in cols}
        CI = {c: i for i, c in enumerate(cols)}
        Soa = np.stack([S[c][:, IOA] for c in cols], 2)          # (n, 3, ncol)
        base = CI[(1, 'NONE', 0.)]
        res = {}
        # member and blend coupling (raw + R1oa3), e1 and e2 base
        cp = {}
        for s in ('raw', 'R1oa3', 'R3p'):
            cp[s] = {'blend_e1': coupling(S[(1, 'NONE', 0.)][:, IOA], Yoa, W[s]), 'blend_e2': coupling(S[(2, 'NONE', 0.)][:, IOA], Yoa, W[s])}
            mem = M if isflat else M
            it = enumerate(mem) if isflat else mem.items()
            cp[s]['members'] = {str(k): coupling(v[:, IOA], Yoa, W[s]) for k, v in it}
            print('[%s | %s] blend e1 rho %s q %s gap %.3f | e2 gap %.3f | member gaps %s' % (
                cn, s, np.round(cp[s]['blend_e1']['rho'], 3).tolist(), np.round(cp[s]['blend_e1']['q'], 3).tolist(),
                cp[s]['blend_e1']['gap'], cp[s]['blend_e2']['gap'], {k: round(v['gap'], 3) for k, v in cp[s]['members'].items()}), flush=True)
        for si, s in enumerate(('raw', 'R1oa3', 'R3p', 'R3p_hi')):
            p = W[s] / W[s].sum(); rng = np.random.default_rng(5000 + si); D = []
            for _ in range(B):
                ii = rng.choice(n, n, p=p); y = Yoa[ii]
                if (y.sum(0) < 5).any() or ((n - y.sum(0)) < 5).any():
                    continue
                D.append(np.stack([auc_cols(Soa[ii, t, :], y[:, t]) for t in range(3)]))   # (3, ncol)
            D = np.array(D); M3 = D.sum(1) / 12.                # (B, ncol) OA-trio macro contribution
            d = lambda c: M3[:, CI[c]] - M3[:, base]
            inc = lambda c1, c0: M3[:, CI[c1]] - M3[:, CI[c0]]
            r = {'B': int(len(D)), 'base_auc': D[:, :, base].mean(0).tolist(),
                 'curve': {'%d|%s|%.2f' % c: stat(d(c)) for c in cols if c[1] != 'NONE'}}
            r['E1_oa'] = stat(d((2, 'NONE', 0.)))
            r['lens1'] = {}
            for a in AG:
                g1 = d((1, 'FINAL', a)); g2 = inc((2, 'FINAL', a), (2, 'NONE', 0.))
                r['lens1']['%.2f' % a] = {'eoa_gain_e1': stat(g1), 'eoa_gain_e2': stat(g2), 'interaction': stat(g2 - g1),
                                          'stack_inc': stat(inc((2, 'FINAL', a), (1, 'FINAL', a)))}
            r['lens2'] = {}
            for pl in places[1:]:
                r['lens2'][pl] = {}
                for a in AG:
                    r['lens2'][pl]['%.2f' % a] = {'same_a': stat(inc((1, pl, a), (1, 'FINAL', a))),
                                                  'vs_F10': stat(inc((1, pl, a), (1, 'FINAL', .1))),
                                                  'vs_F20': stat(inc((1, pl, a), (1, 'FINAL', .2)))}
            # best a per placement (e1) and for e2 FINAL
            r['best'] = {}
            for e, pl in [(1, x) for x in places] + [(2, 'FINAL')]:
                mm = [d((e, pl, a)).mean() for a in AG]; r['best']['%d|%s' % (e, pl)] = [AG[int(np.argmax(mm))], float(np.max(mm))]
            res[s] = r
            print('[%s | %s | P_FT] B %d | base OA AUC %s' % (cn, s, len(D), np.round(r['base_auc'], 4).tolist()), flush=True)
            print('   E1 (pow2) alone, OA trio macro: %s' % fmt(r['E1_oa']))
            for a in (.1, .2):
                l1 = r['lens1']['%.2f' % a]
                print('   a %.2f | E-OA gain e1 %s | e2 %s | interaction %s | STACK (E1+E-OA) - (E-OA) %s%s' % (
                    a, fmt(l1['eoa_gain_e1']), fmt(l1['eoa_gain_e2']), fmt(l1['interaction']), fmt(l1['stack_inc']),
                    ' >SD' if l1['stack_inc']['m'] > l1['stack_inc']['sd'] else ''))
            for e, pl in [(1, x) for x in places] + [(2, 'FINAL')]:
                print('   curve e%d %-6s %s | best a %.2f' % (e, pl, ' '.join('%.2f:%+.5f' % (a, r['curve']['%d|%s|%.2f' % (e, pl, a)]['m']) for a in AG),
                                                         r['best']['%d|%s' % (e, pl)][0]))
            for pl in places[1:]:
                for a in AG:
                    l2 = r['lens2'][pl]['%.2f' % a]
                    print('   place %-6s a %.2f | vs FINAL same a %s | vs FINAL .1 %s%s | vs FINAL .2 %s%s' % (
                        pl, a, fmt(l2['same_a']), fmt(l2['vs_F10']), ' >SD' if l2['vs_F10']['m'] > l2['vs_F10']['sd'] else '',
                        fmt(l2['vs_F20']), ' >SD' if l2['vs_F20']['m'] > l2['vs_F20']['sd'] else ''), flush=True)
        # non-OA labels, E1 effect, raw OAI, plain bootstrap over knees (each label on its covered subset)
        S1 = S[(1, 'NONE', 0.)]; S2 = S[(2, 'NONE', 0.)]; rng = np.random.default_rng(77); DN = []
        full = []
        for jj, j in enumerate(INON):
            m = np.isfinite(Y[:, j]); y = Y[m, j].astype(int)
            full.append(auc_cols(np.stack([S1[m, j], S2[m, j]], 1), y))
        for _ in range(B):
            ii = rng.integers(0, n, n); row = []
            for j in INON:
                jj = ii[np.isfinite(Y[ii, j])]; y = Y[jj, j].astype(int)
                if y.sum() < 5 or (len(y) - y.sum()) < 5:
                    row.append(np.nan); continue
                a2 = auc_cols(np.stack([S1[jj, j], S2[jj, j]], 1), y); row.append(a2[1] - a2[0])
            DN.append(row)
        DN = np.array(DN)
        nonoa = {L9[j]: dict(stat(DN[:, k][np.isfinite(DN[:, k])]), full=float(full[k][1] - full[k][0]), base=float(full[k][0]))
                 for k, j in enumerate(INON)}
        tot = np.nansum(DN, 1) / 12.
        print('[%s | raw | non-OA] E1 per label (AUC): %s | 6-label macro contribution %s' % (
            cn, {k: (round(v['m'], 4), round(v['sd'], 4)) for k, v in nonoa.items()}, fmt(stat(tot))), flush=True)
        ALL[cn] = {'coupling': cp, 'oa': res, 'nonoa': nonoa, 'nonoa_macro': stat(tot)}
        print('   (%.0f s)' % (time.time() - T0), flush=True)
    print('JSON_BEGIN'); print(json.dumps({'job': 'L12', 'n': n, 'res': ALL})); print('JSON_END')


def job_l3():
    Y, F = setup(); n = len(Y); Yoa = Y[:, IOA].astype(int)
    W = schemes(Yoa); singles = ['f0', 'f1', 'f2', 'f3', 'f4', 's1044', 's1045']
    AGR = [round(x, 3) for x in np.arange(0, .4001, .025)]
    print('JOB L3 | n %d | OA prevalence %s' % (n, np.round(Yoa.mean(0), 3).tolist()), flush=True)
    combos = [c for k in (1, 2, 3, 4, 5, 7) for c in itertools.combinations(singles, k)]
    U = {}
    for c in combos:
        U[c] = np.mean([pctc(F[k][:, IOA]) for k in c], 0)
    OUT = {}
    for s in ('R1oa3', 'R3p', 'R3p_hi', 'raw'):
        w = W[s]; rows = []
        for c in combos:
            cp = coupling(U[c], Yoa, w)
            base = [wauc(U[c][:, t], Yoa[:, t], w) for t in range(3)]
            curve = []
            for a in AGR:
                Ua = eoa(U[c], a) if a else U[c]
                curve.append(sum(wauc(Ua[:, t], Yoa[:, t], w) - base[t] for t in range(3)) / 12.)
            curve = np.array(curve); ib = int(np.argmax(curve))
            # parabolic refinement of a* around the grid max
            if 0 < ib < len(AGR) - 1:
                y0, y1, y2 = curve[ib - 1], curve[ib], curve[ib + 1]; den = y0 - 2 * y1 + y2
                astar = AGR[ib] + (.025 * .5 * (y0 - y2) / den if den < 0 else 0.)
            else:
                astar = AGR[ib]
            rows.append({'combo': list(c), 'k': len(c), 'fold_only': all(x.startswith('f') for x in c),
                         'rho': float(np.mean(cp['rho'])), 'q': float(np.mean(cp['q'])), 'gap': cp['gap'], 'astar': float(astar),
                         'd10': float(curve[AGR.index(.1)]), 'd20': float(curve[AGR.index(.2)]), 'dmax': float(curve.max()),
                         'base_mean_auc': float(np.mean(base))})
        df = pd.DataFrame(rows)
        print('[%s] per-k summary (mean over blends; sd in brackets):' % s)
        for (k, fo), g in df.groupby(['k', 'fold_only']):
            print('   k %d %-5s n_blends %3d | rho %.3f q %.3f gap %.3f (%.3f) | a* %.3f (%.3f) | d.1 %+.5f d.2 %+.5f d.2-d.1 %+.5f (%.5f) | base AUC %.4f' % (
                k, 'folds' if fo else 'mixed', len(g), g.rho.mean(), g.q.mean(), g.gap.mean(), g.gap.std() if len(g) > 1 else 0,
                g.astar.mean(), g.astar.std() if len(g) > 1 else 0, g.d10.mean(), g.d20.mean(), (g.d20 - g.d10).mean(),
                (g.d20 - g.d10).std() if len(g) > 1 else 0, g.base_mean_auc.mean()), flush=True)
        for k, g in df.groupby('k'):
            print('   k %d ALL n_blends %3d | gap %.3f | a* %.3f | d.2-d.1 %+.5f' % (k, len(g), g.gap.mean(), g.astar.mean(), (g.d20 - g.d10).mean()))
        X = np.column_stack([np.ones(len(df)), df.gap.values, df.k.values])
        for yv in ('astar',):
            bcoef = np.linalg.lstsq(X, df[yv].values, rcond=None)[0]
            b1 = np.linalg.lstsq(X[:, :2], df[yv].values, rcond=None)[0]
            print('   OLS %s = %.3f + %.3f gap + %.4f k   | gap only: %.3f + %.3f gap | corr(gap, a*) %.3f corr(k, a*) %.3f corr(k, gap) %.3f' % (
                yv, bcoef[0], bcoef[1], bcoef[2], b1[0], b1[1], np.corrcoef(df.gap, df[yv])[0, 1], np.corrcoef(df.k, df[yv])[0, 1],
                np.corrcoef(df.k, df.gap)[0, 1]))
        Xd = np.column_stack([np.ones(len(df)), df.gap.values])
        bd = np.linalg.lstsq(Xd, (df.d20 - df.d10).values, rcond=None)[0]
        print('   OLS d.2-d.1 = %+.5f + %+.5f gap  (zero crossing gap %.3f)' % (bd[0], bd[1], -bd[0] / bd[1] if bd[1] else float('nan')), flush=True)
        OUT[s] = rows
    # bootstrap of the per-k mean of d(.2) - d(.1), R1oa3 and R3p (weighted resample, unweighted AUC, as box_curve)
    BOOTK = max(5, int(100 * BSCALE)); BK = {}
    for si, s in enumerate(('R1oa3', 'R3p')):
        p = W[s] / W[s].sum(); rng = np.random.default_rng(900 + si); acc = []
        pre = {c: (U[c], eoa(U[c], .1), eoa(U[c], .2)) for c in combos}
        for _ in range(BOOTK):
            ii = rng.choice(n, n, p=p); y = Yoa[ii]; row = {}
            for c in combos:
                u0, u1, u2 = pre[c]; dd = 0.
                for t in range(3):
                    A_ = auc_cols(np.stack([u0[ii, t], u1[ii, t], u2[ii, t]], 1), y[:, t]); dd += A_[2] - A_[1]
                row[c] = dd / 12.
            acc.append(row)
        BK[s] = {}
        for k in (1, 2, 3, 4, 5, 7):
            for fo in (True, False):
                cs = [c for c in combos if len(c) == k and all(x.startswith('f') for x in c) == fo]
                if not cs: continue
                v = np.array([np.mean([r[c] for c in cs]) for r in acc]); BK[s]['%d|%s' % (k, fo)] = stat(v)
                print('[%s] boot per-k mean d.2-d.1 k %d %-5s %s' % (s, k, 'folds' if fo else 'mixed', fmt(stat(v))), flush=True)
        # k7 minus k1 (all), paired
        v = np.array([np.mean([r[c] for c in combos if len(c) == 7]) - np.mean([r[c] for c in combos if len(c) == 1]) for r in acc])
        BK[s]['k7_minus_k1'] = stat(v); print('[%s] boot (k7 - k1 mean) of d.2-d.1 %s' % (s, fmt(stat(v))), flush=True)
    print('JSON_BEGIN'); print(json.dumps({'job': 'L3', 'n': n, 'rows': OUT, 'boot': BK})); print('JSON_END')
    print('(%.0f s)' % (time.time() - T0))


if JOB == 'L12':
    job_l12()
elif JOB == 'L3':
    job_l3()
