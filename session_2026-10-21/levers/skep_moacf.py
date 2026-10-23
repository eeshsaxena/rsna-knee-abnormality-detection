# skep_moacf.py (10-21 levers, Claude, SKEPTIC of MOACF). RESEARCH ONLY, local CPU, NO OAI, no box. Writes skep_moacf.log.
# Question: is the MOACF ruler gain (MOA <- Cont .05 then MOA <- Frac .05 on the pw30 base) a v2-report-labeler artefact?
# (1) REP on three report label sources (v2 / v4 / sol56), dropUNK and UNK0, 4 plain rulers + 6 c4-shaped nests.
# (2) EB regime rebuilt from the v4 labels (train tet + same-knee D map on gold58 from v4), next to the v2 EB of regime_sens.py.
# (3) Placebo field for the MOA target: every partner at a .05 under REP (v2 / v4 / sol56) and EB (v2 / v4), 3mean and R7 rulers.
import json, itertools
import numpy as np
from rg_lib import (L, J, AB, OA, eoa, near_pd, copula_cells, weights_from_target, load_all, pct, wauc_multi, tet_xy_fast, LD)
Bd = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/'
log = open(Bd + 'skep_moacf.log', 'w')


def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()


D = load_all(); X, tr3, Y74 = D['X'], D['tr3'], D['Y74']
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

LAB = {}
for src in ('v2', 'v4', 'sol56'):
    lab = LD.labels(src)
    Ytr = lab.reindex(tr3)[L].values.astype(float)
    Yg = lab.reindex(ids58)[L].values.astype(float)
    LAB[src] = (Ytr, Yg)
    P('labels %-5s: train rows %d, MOA known %d, Cont known %d, Frac known %d | gold58 Cont known %d Frac known %d' % (
        src, len(Ytr), (~np.isnan(Ytr[:, J['Medial OA']])).sum(), (~np.isnan(Ytr[:, J['Contusion']])).sum(), (~np.isnan(Ytr[:, J['Fracture']])).sum(),
        (~np.isnan(Yg[:, J['Contusion']])).sum(), (~np.isnan(Yg[:, J['Fracture']])).sum()))


def tetm(Yx):
    T = np.eye(12)
    for i in range(12):
        for j in range(i + 1, 12): T[i, j] = T[j, i] = tet_xy_fast(Yx[:, i], Yx[:, j])
    return T


def regime_mats(Ytr, Yg58):
    Trep = tetm(Ytr); Dm = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            Dm[i, j] = Dm[j, i] = tet_xy_fast(E58[:, i], E58[:, j]) - tet_xy_fast(Yg58[:, i], Yg58[:, j])
    mD = np.mean([Dm[i, j] for i in range(12) for j in range(i + 1, 12) if not trio(i, j)])
    Deb = np.array([[0 if i == j else (Dm[i, j] if trio(i, j) else mD + REL * (Dm[i, j] - mD)) for j in range(12)] for i in range(12)])
    return Trep, Dm, Deb, mD


MATS = {}
for src in ('v2', 'v4'):
    for mode in ('dropUNK', 'UNK0'):
        Ytr, Yg = LAB[src]
        if mode == 'UNK0': Ytr, Yg = np.nan_to_num(Ytr, nan=0.), np.nan_to_num(Yg, nan=0.)
        MATS[(src, mode)] = regime_mats(Ytr, Yg)
        Tr, Dm, De, mD = MATS[(src, mode)]
        P('%s %-7s: mean non-trio D %+.3f | MOA-Cont train tet %+.2f D %+.2f EBshift %+.2f | MOA-Frac train tet %+.2f D %+.2f EBshift %+.2f' % (
            src, mode, mD, Tr[J['Medial OA'], J['Contusion']], Dm[J['Medial OA'], J['Contusion']], De[J['Medial OA'], J['Contusion']],
            Tr[J['Medial OA'], J['Fracture']], Dm[J['Medial OA'], J['Fracture']], De[J['Medial OA'], J['Fracture']]))


def run(t, parts, src, mode, regs, rulers=None):
    Ytr = LAB[src][0]
    if mode == 'UNK0': Ytr = np.nan_to_num(Ytr, nan=0.)
    cols = list(dict.fromkeys(([c for c in OA] if (t in OA or any(o in OA for o, _ in parts)) else []) + [t] + [o for o, _ in parts]))
    jc = [J[c] for c in cols]
    m_ = ~np.isnan(Ytr[:, jc]).any(1); Ys = Ytr[m_][:, jc].astype(int); yt = Ys[:, cols.index(t)]
    W = {}
    for r in regs:
        if r == 'REP':
            W[r] = np.ones(len(Ys))
        else:
            Tr, Dm, De, mD = MATS[(r.split('_')[1], mode)]
            W[r] = weights_from_target(Ys, copula_cells(Ys.mean(0), near_pd(Tr[np.ix_(jc, jc)] + De[np.ix_(jc, jc)])))
    Ws = [W[r] for r in regs]; res = {}
    for rn, Bm in RUL.items():
        if rulers and rn not in rulers: continue
        B_ = Bm[m_]; col = B_[:, J[t]].copy()
        for o, a in parts: col = pct(pct(col) - a * pct(B_[:, J[o]]))
        res[rn] = (wauc_multi(col, yt, Ws) - wauc_multi(B_[:, J[t]], yt, Ws)) * 1e4
    return len(Ys), res


MOACF = [('Contusion', .05), ('Fracture', .05)]
P('')
P('(1) MOACF REP by label source | label AUC x1e4 | R7 s1044 s1045 3mean || nests min / mean / max')
for src in ('v2', 'v4', 'sol56'):
    for mode in ('dropUNK', 'UNK0'):
        for nm, parts in (('MOA<-Cont.05', MOACF[:1]), ('MOA<-Frac.05', MOACF[1:]), ('MOACF', MOACF)):
            n, res = run('Medial OA', parts, src, mode, ['REP'])
            pl = [res[k][0] for k in ('R7', 's1044', 's1045', '3mean')]; ne = np.array([res[k][0] for k in res if k.startswith('N:')])
            P('%-5s %-7s %-13s n %4d REP: %s || nests %+6.1f / %+6.1f / %+6.1f%s' % (src, mode, nm, n, ' '.join('%+6.1f' % v for v in pl), ne.min(), ne.mean(), ne.max(),
                                                                      '  ALL+' if min(min(pl), ne.min()) > 0 else ''))

P('')
P('(2) MOACF EB built from v2 vs v4 (both evaluated on v2 train labels, the rulers trained on v2-like report labels) | label AUC x1e4')
for mode in ('dropUNK', 'UNK0'):
    for nm, parts in (('MOA<-Cont.05', MOACF[:1]), ('MOA<-Frac.05', MOACF[1:]), ('MOACF', MOACF)):
        n, res = run('Medial OA', parts, 'v2', mode, ['EB_v2', 'EB_v4'])
        for ir, r in enumerate(['EB_v2', 'EB_v4']):
            pl = [res[k][ir] for k in ('R7', 's1044', 's1045', '3mean')]; ne = np.array([res[k][ir] for k in res if k.startswith('N:')])
            P('v2lab %-7s %-13s n %4d %-5s: %s || nests %+6.1f / %+6.1f / %+6.1f' % (mode, nm, n, r, ' '.join('%+6.1f' % v for v in pl), ne.min(), ne.mean(), ne.max()))
P('    and v4 labels raked to EB_v4 (self-consistent v4 world):')
for mode in ('dropUNK', 'UNK0'):
    for nm, parts in (('MOA<-Cont.05', MOACF[:1]), ('MOA<-Frac.05', MOACF[1:]), ('MOACF', MOACF)):
        n, res = run('Medial OA', parts, 'v4', mode, ['EB_v4'])
        pl = [res[k][0] for k in ('R7', 's1044', 's1045', '3mean')]; ne = np.array([res[k][0] for k in res if k.startswith('N:')])
        P('v4lab %-7s %-13s n %4d EB_v4: %s || nests %+6.1f / %+6.1f / %+6.1f' % (mode, nm, n, ' '.join('%+6.1f' % v for v in pl), ne.min(), ne.mean(), ne.max()))

P('')
P('(3) placebo field, MOA <- o at a .05, UNK0, 3mean / R7 | REP v2, REP v4, REP sol56, EB_v2 (v2 lab), EB_v4 (v4 lab)')
for o in L:
    if o in OA: continue
    out = []
    for src, regs in (('v2', ['REP', 'EB_v2']), ('v4', ['REP', 'EB_v4']), ('sol56', ['REP'])):
        n, res = run('Medial OA', [(o, .05)], src, 'UNK0', regs, rulers=('R7', '3mean'))
        out.append((src, regs, res))
    s = ' | '.join('%s %s %+5.1f/%+5.1f' % (src, r, res['3mean'][ir], res['R7'][ir]) for src, regs, res in out for ir, r in enumerate(regs))
    P('MOA<-%-5s %s' % (AB[o], s))
P('SKEP_MOACF DONE')
