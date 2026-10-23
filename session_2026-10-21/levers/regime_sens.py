# regime_sens.py (10-21 levers, Claude, REGIME track step 6). RESEARCH ONLY, local CPU, NO OAI. Writes regime_sens.log.
# Sensitivity of the short-listed cells (regime_boot.log) to: (a) the ruler SHAPE: six c4-shaped nests built from the three OOF models
# (each permutation of R7 / s1044 / s1045 in the Raptor / R6 / S6fall slots, c4 per-label weights, .8 / .2 S6F blend, then pw30 E-OA, the
# kernel order), next to the plain rulers; (b) report UNKNOWN handling: v2 NaN dropped (as everywhere above) vs UNKNOWN = 0 on all rows,
# with the same-knee D map and the report train tet recomputed the same way; (c) a stacked MOA line (Cont .05 then Frac .05).
# Regimes: REP, EB (trio raw D, others meanD + rel (D - meanD)), RAW, G74 (raw gold74 tet, copula); label AUC x1e4; macro = /12.
import json, itertools
import numpy as np
from rg_lib import (L, J, AB, OA, eoa, near_pd, copula_cells, weights_from_target, load_all, pct, wauc_multi, tet_xy_fast, tet_matrix, LD)
Bd = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/'
log = open(Bd + 'regime_sens.log', 'w')


def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()


D = load_all(); X, Y, Y74 = D['X'], D['Y'], D['Y74']
g, g58 = LD.gold(); ids58 = g58.StudyInstanceUID.tolist(); E58 = g58[L].values.astype(float)
R58 = LD.labels('v2').reindex(ids58)[L].values.astype(float)
REL = json.load(open(Bd + 'regime_dlt.json'))['rel']['nontrio']
C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
trio = lambda i, j: L[i] in OA and L[j] in OA


def nest(A, Bm, C):
    V = np.column_stack([C4W.get(c, .5) * pct(A[:, j]) + (1 - C4W.get(c, .5)) * pct(Bm[:, j]) for j, c in enumerate(L)])
    return np.column_stack([.8 * pct(V[:, j]) + .2 * pct(C[:, j]) for j in range(12)])


RUL = {m: eoa(F, .3, 'pw') for m, F in X.items()}
for p in itertools.permutations(['R7', 's1044', 's1045']):
    RUL['N:' + '/'.join(x[:2] if x == 'R7' else x[-2:] for x in p)] = eoa(nest(X[p[0]], X[p[1]], X[p[2]]), .3, 'pw')


def regime_mats(Ytr, Yg58):
    Trep = tet_matrix_fast(Ytr); Dm = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            Dm[i, j] = Dm[j, i] = tet_xy_fast(E58[:, i], E58[:, j]) - tet_xy_fast(Yg58[:, i], Yg58[:, j])
    mD = np.mean([Dm[i, j] for i in range(12) for j in range(i + 1, 12) if not trio(i, j)])
    Deb = np.array([[0 if i == j else (Dm[i, j] if trio(i, j) else mD + REL * (Dm[i, j] - mD)) for j in range(12)] for i in range(12)])
    return Trep, Dm, Deb, mD


def tet_matrix_fast(Yx):
    T = np.eye(12)
    for i in range(12):
        for j in range(i + 1, 12): T[i, j] = T[j, i] = tet_xy_fast(Yx[:, i], Yx[:, j])
    return T


T74 = tet_matrix_fast(Y74)
MODES = {'dropUNK': (Y, R58), 'UNK0': (np.nan_to_num(Y, nan=0.), np.nan_to_num(R58, nan=0.))}
MATS = {k: regime_mats(*v) for k, v in MODES.items()}
for k, (Tr, Dm, De, mD) in MATS.items():
    P('mode %-7s: mean non-trio D %+.3f | trio D %s' % (k, mD, ' '.join('%+.2f' % Dm[J[a], J[b]] for a, b in (('Medial OA', 'Lateral OA'), ('Medial OA', 'PF OA'), ('Lateral OA', 'PF OA')))))
CELLS = [('Medial OA', [('Fracture', .05)]), ('Medial OA', [('Fracture', .10)]), ('Medial OA', [('Contusion', .05)]), ('Medial OA', [('Contusion', .10)]),
         ('Medial OA', [('Contusion', .05), ('Fracture', .05)]), ('Fracture', [("Baker's", .05)]), ("Baker's", [('Fracture', .05)]),
         ("Baker's", [('ACL', .05)]), ('Lateral OA', [('Medial Meniscus', .05)]), ('MCL', [('Medial Meniscus', .10)]), ('ACL', [('PF OA', -.05)]),
         ('Lateral OA', [('Contusion', .05)]), ('PF OA', [('Contusion', .05)])]
REGS = ['REP', 'EB', 'RAW', 'G74']
P('')
P('cell | mode | n | regime: R7 s1044 s1045 3mean || six c4-shaped nests (min / mean / max) ; label AUC x1e4')
for t, parts in CELLS:
    cols = list(dict.fromkeys(([c for c in OA] if (t in OA or any(o in OA for o, _ in parts)) else []) + [t] + [o for o, _ in parts]))
    jc = [J[c] for c in cols]
    for mode, (Yt, _) in MODES.items():
        Tr, Dm, De, mD = MATS[mode]
        m_ = ~np.isnan(Yt[:, jc]).any(1); Ys = Yt[m_][:, jc].astype(int); yt = Ys[:, cols.index(t)]
        W = {'REP': np.ones(len(Ys))}
        for r, M in (('EB', Tr[np.ix_(jc, jc)] + De[np.ix_(jc, jc)]), ('RAW', Tr[np.ix_(jc, jc)] + Dm[np.ix_(jc, jc)]), ('G74', T74[np.ix_(jc, jc)])):
            W[r] = weights_from_target(Ys, copula_cells(Ys.mean(0), near_pd(M)))
        Ws = [W[r] for r in REGS]; res = {}
        for rn, Bm in RUL.items():
            B_ = Bm[m_]; col = B_[:, J[t]].copy()
            for o, a in parts: col = pct(pct(col) - a * pct(B_[:, J[o]]))
            res[rn] = (wauc_multi(col, yt, Ws) - wauc_multi(B_[:, J[t]], yt, Ws)) * 1e4
        nm = AB[t] + '<-' + '+'.join('%s%.2f' % (AB[o], a) for o, a in parts)
        for ir, r in enumerate(REGS):
            pl = [res[k][ir] for k in ('R7', 's1044', 's1045', '3mean')]; ne = np.array([res[k][ir] for k in res if k.startswith('N:')])
            P('%-22s %-7s n %4d %-3s: %s || nests %+6.1f / %+6.1f / %+6.1f%s' % (nm, mode, len(Ys), r, ' '.join('%+6.1f' % v for v in pl), ne.min(), ne.mean(), ne.max(),
                                                                          '  ALL+' if min(min(pl), ne.min()) > 0 else ''))
P('REGIME_SENS DONE')
