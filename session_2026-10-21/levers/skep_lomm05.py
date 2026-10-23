# skep_lomm05.py (10-21 levers, Claude, SKEPTIC of the "lomm05 PROMISING" proposal from regime.md). RESEARCH ONLY, local CPU, NO OAI,
# NO box. Writes skep_lomm05.log (aggregates only). Question: does the regime track's EB support for LOA = pct(pct(LOA) - .05 pct(MM))
# survive (a) splitting EB into its generic (mean non-trio D) and pair-specific parts, (b) a different report-label extraction for the
# SAME population-matched construction (v4, sol56 instead of v2), (c) a gold-like case mix (density ratio on R7 OOF logits),
# (d) an LOA-target placebo set under each regime?
# Ruler machinery = regime_sens.py (cell cols MOA, LOA, PF, MM; four plain rulers + six c4-shaped nests on the pw30 base).
import sys, json, itertools
import numpy as np
from rg_lib import (L, J, AB, OA, eoa, near_pd, copula_cells, weights_from_target, load_all, pct, wauc_multi, tet_xy_fast, tet_fast, LD)
from scipy.special import expit
Bd = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/'
log = open(Bd + 'skep_lomm05.log', 'w')


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()


D = load_all(); X, Y74 = D['X'], D['Y74']; tr3 = D['tr3']
g, g58 = LD.gold(); ids58 = g58.StudyInstanceUID.tolist(); E58 = g58[L].values.astype(float)
REL = json.load(open(Bd + 'regime_dlt.json'))['rel']['nontrio']
C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
trio = lambda i, j: L[i] in OA and L[j] in OA


def nest(A, Bm, C):
    V = np.column_stack([C4W.get(c, .5) * pct(A[:, j]) + (1 - C4W.get(c, .5)) * pct(Bm[:, j]) for j, c in enumerate(L)])
    return np.column_stack([.8 * pct(V[:, j]) + .2 * pct(C[:, j]) for j in range(12)])


RUL = {m: eoa(F, .3, 'pw') for m, F in X.items()}
for p in itertools.permutations(['R7', 's1044', 's1045']):
    RUL['N:' + '/'.join(x[:2] if x == 'R7' else x[-2:] for x in p)] = eoa(nest(X[p[0]], X[p[1]], X[p[2]]), .3, 'pw')
PL = ['R7', 's1044', 's1045', '3mean']; NE = [k for k in RUL if k.startswith('N:')]


def wtet(a, b, w=None):
    m = ~(np.isnan(a) | np.isnan(b)); a = a[m].astype(int); b = b[m].astype(int)
    w = np.ones(len(a)) if w is None else w[m]
    c = [w[(a == x) & (b == y)].sum() for x, y in ((0, 0), (0, 1), (1, 0), (1, 1))]
    s = len(a) / max(sum(c), 1e-9); return tet_fast(*[ci * s for ci in c])


def mats(Ytr, Yg58, wtr=None):
    Trep = np.eye(12); Dm = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            Trep[i, j] = Trep[j, i] = wtet(Ytr[:, i], Ytr[:, j], wtr)
            Dm[i, j] = Dm[j, i] = tet_xy_fast(E58[:, i], E58[:, j]) - tet_xy_fast(Yg58[:, i], Yg58[:, j])
    mD = np.mean([Dm[i, j] for i in range(12) for j in range(i + 1, 12) if not trio(i, j)])
    return Trep, Dm, mD


def Dfam(Dm, mD, kind):
    """non-trio D used in the target; trio always raw D."""
    f = {'EB': lambda d: mD + REL * (d - mD), 'GEN': lambda d: mD, 'SPEC': lambda d: REL * (d - mD), 'SPEC0': lambda d: REL * d,
         'TRIO': lambda d: 0., 'RAW': lambda d: d}[kind]
    return np.array([[0 if i == j else (Dm[i, j] if trio(i, j) else f(Dm[i, j])) for j in range(12)] for i in range(12)])


T74 = np.eye(12)
for i in range(12):
    for j in range(i + 1, 12): T74[i, j] = T74[j, i] = tet_xy_fast(Y74[:, i], Y74[:, j])


def wobs_target(Ys, tgt, wpop):
    k = Ys.shape[1]; idx = (Ys * (2 ** np.arange(k - 1, -1, -1))).sum(1).astype(int)
    obs = np.bincount(idx, weights=wpop, minlength=2 ** k); obs /= obs.sum()
    wc = np.where(obs > 0, tgt / np.maximum(obs, 1e-12), 0.); w = wpop * wc[idx]; return w / w.mean()


def run_cell(t, parts, Yt, Mset, regs, wpop=None, extra_cols=None):
    cols = list(dict.fromkeys(([c for c in OA] if (t in OA or any(o in OA for o, _ in parts)) else []) + [t] + [o for o, _ in parts]))
    jc = [J[c] for c in cols]
    m_ = ~np.isnan(Yt[:, jc]).any(1); Ys = Yt[m_][:, jc].astype(int); yt = Ys[:, cols.index(t)]
    wp = np.ones(m_.sum()) if wpop is None else wpop[m_] / wpop[m_].mean()
    Tr, Dm, mD = Mset
    W = []
    for r in regs:
        if r == 'REP': W.append(wp.copy()); continue
        M = T74[np.ix_(jc, jc)] if r == 'G74' else (Tr + Dfam(Dm, mD, r))[np.ix_(jc, jc)]
        marg = (wp[:, None] * Ys).sum(0) / wp.sum()
        W.append(wobs_target(Ys, copula_cells(marg, near_pd(M)), wp))
    res = {}
    for rn, Bm in RUL.items():
        B_ = Bm[m_]; col = B_[:, J[t]].copy()
        for o, a in parts: col = pct(pct(col) - a * pct(B_[:, J[o]]))
        res[rn] = (wauc_multi(col, yt, W) - wauc_multi(B_[:, J[t]], yt, W)) * 1e4
    return res, len(Ys), cols


def fmt(res, regs, ir):
    pl = [res[k][ir] for k in PL]; ne = np.array([res[k][ir] for k in NE])
    return '%s || nests %+6.1f / %+6.1f / %+6.1f%s' % (' '.join('%+6.1f' % v for v in pl), ne.min(), ne.mean(), ne.max(), '  ALL+' if min(min(pl), ne.min()) > 0 else '')


# ---------------- label sources ----------------
SRC = {}
for s in ('v2', 'v4', 'sol56'):
    lab = LD.labels(s)
    Ytr = lab.reindex(tr3)[L].values.astype(float); Yg = lab.reindex(ids58)[L].values.astype(float)
    SRC[s] = (Ytr, Yg)
    P('label source %-5s: train rows with label %d / %d, gold58 rows with label %d / 58' % (s, int((~np.isnan(Ytr).all(1)).sum()), len(tr3), int((~np.isnan(Yg).all(1)).sum())))

REGS = ['REP', 'EB', 'GEN', 'SPEC', 'SPEC0', 'TRIO', 'RAW', 'G74']
LOAMM = ('Lateral OA', [('Medial Meniscus', .05)])
P('\nREL (non-trio reliability) %.3f; units label AUC x1e4 (macro = /12); rulers R7 s1044 s1045 3mean || six c4-shaped nests min / mean / max' % REL)
P('regimes: REP report; EB = trio raw D + nontrio mD + REL (D - mD) [regime.md]; GEN = nontrio mD only; SPEC = nontrio REL (D - mD) only;')
P('         SPEC0 = nontrio REL D (no centring); TRIO = trio raw D, nontrio report coupling; RAW = all raw D; G74 = raw gold74 joint')
SUMM = {}
for s in ('v2', 'v4', 'sol56'):
    for mode in ('dropUNK', 'UNK0'):
        Ytr, Yg = SRC[s]
        if mode == 'UNK0': Ytr, Yg = np.nan_to_num(Ytr, nan=0.), np.nan_to_num(Yg, nan=0.)
        Ms = mats(Ytr, Yg)
        Tr, Dm, mD = Ms; i, j = J['Lateral OA'], J['Medial Meniscus']
        P('\n[%s %s] mean nontrio D %+.3f | LOA-MM: train report tet %.3f, gold58 report %.3f, gold58 expert %.3f, D %+.3f -> EB target %.3f '
          '(GEN %.3f, SPEC %.3f, RAW %.3f, G74 %.3f) | MOA-MM rep %.3f EB %.3f | PF-MM rep %.3f EB %.3f' % (
              s, mode, mD, Tr[i, j], tet_xy_fast(Yg[:, i], Yg[:, j]), tet_xy_fast(E58[:, i], E58[:, j]), Dm[i, j], Tr[i, j] + Dfam(Dm, mD, 'EB')[i, j],
              Tr[i, j] + mD, Tr[i, j] + Dfam(Dm, mD, 'SPEC')[i, j], Tr[i, j] + Dm[i, j], T74[i, j],
              Tr[J['Medial OA'], j], Tr[J['Medial OA'], j] + Dfam(Dm, mD, 'EB')[J['Medial OA'], j], Tr[J['PF OA'], j], Tr[J['PF OA'], j] + Dfam(Dm, mD, 'EB')[J['PF OA'], j]))
        res, n, cols = run_cell(*LOAMM, Ytr, Ms, REGS)
        for ir, r in enumerate(REGS):
            P('  LOA<-MM.05 %-5s %-7s n %4d %-5s: %s' % (s, mode, n, r, fmt(res, REGS, ir)))
            SUMM[(s, mode, r)] = {k: float(res[k][ir]) for k in res}

# ---------------- LOA-target placebo set under REP / EB / GEN for each label source ----------------
P('\nPLACEBO: LOA <- X a .05 for all 11 X, dropUNK; 3mean / R7 / nest mean (label x1e4); MM rank by nest mean')
for s in ('v2', 'v4', 'sol56'):
    Ytr, Yg = SRC[s]; Ms = mats(Ytr, Yg); rows = []
    for o in [c for c in L if c != 'Lateral OA']:
        res, n, _ = run_cell('Lateral OA', [(o, .05)], Ytr, Ms, ['REP', 'EB', 'GEN'])
        rows.append((o, n, [(res['3mean'][k], res['R7'][k], np.mean([res[q][k] for q in NE])) for k in range(3)]))
    for k, r in enumerate(['REP', 'EB', 'GEN']):
        srt = sorted(rows, key=lambda z: -z[2][k][2]); rank = [z[0] for z in srt].index('Medial Meniscus') + 1
        pm = np.mean([z[2][k][2] for z in rows if z[0] != 'Medial Meniscus'])
        P('  %-5s %-3s: MM rank %d / 11 (nest mean MM %+.1f, placebo mean %+.1f, excess %+.1f) | %s' % (
            s, r, rank, [z[2][k][2] for z in rows if z[0] == 'Medial Meniscus'][0], pm, [z[2][k][2] for z in rows if z[0] == 'Medial Meniscus'][0] - pm,
            ' '.join('%s %+.1f/%+.1f/%+.1f' % (AB[z[0]], *z[2][k]) for z in srt)))

# ---------------- gold-like case mix: density ratio on R7 OOF logits (as stack/skeptic/skep_mcl_curve.py) ----------------
from sklearn.linear_model import LogisticRegression
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
r7 = LD.r7(); gid58 = ids58
Ftr = lg(r7.loc[tr3, ['p|' + c for c in L]].values.astype(float)); Fg = lg(r7.loc[gid58, ['p|' + c for c in L]].values.astype(float))
mu_, sg_ = Ftr.mean(0), Ftr.std(0)
Xd = np.vstack([(Ftr - mu_) / sg_, (Fg - mu_) / sg_]); yd = np.r_[np.zeros(len(Ftr)), np.ones(len(Fg))]
i, j = J['Lateral OA'], J['Medial Meniscus']


def wcorr(a, b, w):
    a = a - np.average(a, weights=w); b = b - np.average(b, weights=w)
    return np.average(a * b, weights=w) / np.sqrt(np.average(a * a, weights=w) * np.average(b * b, weights=w))


c4 = D['c4']
P('\nGOLD-LIKE case mix (density ratio gold58 vs train on R7 OOF logits). c4 gold58 LOA~MM rank corr (pw30 base) %.3f; c4 base %.3f' % (
    np.corrcoef(pct(eoa(c4, .3, 'pw')[:, i]), pct(eoa(c4, .3, 'pw')[:, j]))[0, 1], np.corrcoef(pct(c4[:, i]), pct(c4[:, j]))[0, 1]))
for C in (.05, .3):
    clf = LogisticRegression(C=C, max_iter=3000).fit(Xd, yd, sample_weight=np.r_[np.ones(len(Ftr)) * len(Fg) / len(Ftr), np.ones(len(Fg))])
    pr = clf.predict_proba((Ftr - mu_) / sg_)[:, 1]; wdr = pr / (1 - pr); wdr = np.minimum(wdr, np.percentile(wdr, 99.5)); wdr /= wdr.mean()
    ess = wdr.sum() ** 2 / (wdr ** 2).sum()
    cof = {k: (np.corrcoef(pct(RUL[k][:, i]), pct(RUL[k][:, j]))[0, 1], wcorr(pct(RUL[k][:, i]), pct(RUL[k][:, j]), wdr)) for k in ('R7', '3mean', NE[0])}
    P('  C %.2f: ESS %.0f of %d | ruler LOA~MM co-firing train -> gold-like: %s | R7 OOF on gold58 itself %.3f' % (
        C, ess, len(wdr), ' '.join('%s %.3f->%.3f' % (k, *v) for k, v in cof.items()),
        np.corrcoef(pct(Fg[:, i]), pct(Fg[:, j]))[0, 1]))
    for s in ('v2', 'v4'):
        Ytr, Yg = SRC[s]; Ms = mats(Ytr, Yg, wdr)
        P('    [%s gold-like] weighted train report LOA-MM tet %.3f -> EB target %.3f' % (s, Ms[0][i, j], Ms[0][i, j] + Dfam(Ms[1], Ms[2], 'EB')[i, j]))
        regs = ['REP', 'EB', 'GEN', 'SPEC', 'G74']
        res, n, _ = run_cell(*LOAMM, Ytr, Ms, regs, wpop=wdr)
        for ir, r in enumerate(regs):
            P('    LOA<-MM.05 %-3s gold-like C%.2f %-4s: %s' % (s, C, r, fmt(res, regs, ir)))
            SUMM[(s, 'gold%.2f' % C, r)] = {k: float(res[k][ir]) for k in res}

json.dump({'|'.join(k): v for k, v in SUMM.items()}, open(Bd + 'skep_lomm05.json', 'w'), indent=1)
P('SKEP_LOMM05 DONE')
