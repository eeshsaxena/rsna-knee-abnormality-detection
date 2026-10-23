# skep_lomm05_cal.py (10-21 levers, Claude, lomm05 SKEPTIC). Local CPU, NO OAI. Fairness check for the label-source finding of skep_lomm05.log:
# (1) does the v4 / sol56 ruler calibrate to the public E-OA reads differently from the v2 one (if v4 under-reads E-OA, its smaller lomm05
#     read would need a larger public factor)? E-OA pw30 vs the no-E-OA base, trio AUC summed / 12 (macro e-4), same raking cell
#     (MOA, LOA, PF, MM), regimes REP / EB / G74, per label source; public pw30 vs c4 = 11.3 (CLAIMED val/implied2.md).
# (2) gold58 agreement of each report-label source with the EXPERT LOA and MM labels (aggregates only).
import json, itertools
import numpy as np
from rg_lib import L, J, OA, eoa, near_pd, copula_cells, load_all, pct, wauc_multi, tet_xy_fast, LD
exec(open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/skep_lomm05.py', encoding='utf-8').read().split('# ---------------- label sources')[0].replace("log = open(Bd + 'skep_lomm05.log', 'w')", "log = open(Bd + 'skep_lomm05_cal.log', 'w')"))
RUL0 = dict(X)
for p in itertools.permutations(['R7', 's1044', 's1045']):
    RUL0['N:' + '/'.join(x[:2] if x == 'R7' else x[-2:] for x in p)] = nest(X[p[0]], X[p[1]], X[p[2]])
cols = ['Medial OA', 'Lateral OA', 'PF OA', 'Medial Meniscus']; jc = [J[c] for c in cols]
P('(1) E-OA pw30 vs no-E-OA on the rulers, trio AUC sum / 12 (macro e-4); public 11.3. rulers R7 s1044 s1045 3mean | nest mean; ratio public / nest mean')
for s in ('v2', 'v4', 'sol56'):
    lab = LD.labels(s); Ytr = lab.reindex(tr3)[L].values.astype(float); Yg = lab.reindex(ids58)[L].values.astype(float)
    Ms = mats(Ytr, Yg); Tr, Dm, mD = Ms
    m_ = ~np.isnan(Ytr[:, jc]).any(1); Ys = Ytr[m_][:, jc].astype(int)
    for r in ('REP', 'EB', 'G74'):
        if r == 'REP': W = np.ones(len(Ys))
        else:
            M = T74[np.ix_(jc, jc)] if r == 'G74' else (Tr + Dfam(Dm, mD, 'EB'))[np.ix_(jc, jc)]
            W = wobs_target(Ys, copula_cells(Ys.mean(0), near_pd(M)), np.ones(len(Ys)))
        d = {}
        for k in RUL:
            d[k] = sum((wauc_multi(RUL[k][m_][:, J[c]], Ys[:, cols.index(c)], [W])[0] - wauc_multi(RUL0[k][m_][:, J[c]], Ys[:, cols.index(c)], [W])[0]) for c in OA) / 12 * 1e4
        nm = np.mean([d[k] for k in NE])
        P('  %-5s %-3s: %s | nests %+.2f | public / nest %.2f' % (s, r, ' '.join('%+6.2f' % d[k] for k in PL), nm, 11.3 / nm if abs(nm) > 1e-6 else float('nan')))
P('\n(2) gold58: report label vs EXPERT label (n with both, sensitivity, specificity, kappa)')
E = g58[L].values.astype(float)
for c in ('Lateral OA', 'Medial Meniscus', 'Medial OA', 'PF OA'):
    row = []
    for s in ('v2', 'v4', 'sol56'):
        r_ = LD.labels(s).reindex(ids58)[c].values.astype(float); e = E[:, J[c]]; ok = ~(np.isnan(r_) | np.isnan(e)); a, b = r_[ok].astype(int), e[ok].astype(int)
        tp = ((a == 1) & (b == 1)).sum(); fn = ((a == 0) & (b == 1)).sum(); tn = ((a == 0) & (b == 0)).sum(); fp = ((a == 1) & (b == 0)).sum(); n = ok.sum()
        po = (tp + tn) / n; pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2; kap = (po - pe) / (1 - pe)
        row.append('%s n %d sens %.2f spec %.2f kappa %.2f' % (s, n, tp / max(tp + fn, 1), tn / max(tn + fp, 1), kap))
    P('  %-15s %s' % (c, ' | '.join(row)))
P('SKEP_LOMM05_CAL DONE')
