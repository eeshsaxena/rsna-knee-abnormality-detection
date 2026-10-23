# read_noise.py (10-20 ~23:22 IST, Claude, STACK BUILD round). LOCAL CPU, NO OAI. Paired read noise of each stack component vs pw30 on a
# public-size (390) and private-size (910) subset, method of night/skep_c12p.py (D): R7 OOF (non-gold) with pw30 E-OA applied, report labels,
# 1,000 subsets without replacement; the SAME subsets for every component, so the cross-component correlation of read errors is measured too.
# Two label sets: sol56 (as skep_c12p) and report v2; plus a gold-like MCL-prevalence variant (subsets drawn with MCL+ share = gold74 11/74).
# Prints aggregates only; writes stack/kernels/read_noise.log.
import sys
import numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
from census_lib import auc, pct, L
import load as LD
OUT = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/kernels/read_noise.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n')
rng = np.random.default_rng(21021)
J = {c: i for i, c in enumerate(L)}; OA = ['Medial OA', 'Lateral OA', 'PF OA']
W_PW = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}
def pw(F, a):
    G = F.copy(); U = {c: pct(F[:, J[c]]) for c in OA}
    for c in OA:
        o = [k for k in OA if k != c]; w = [W_PW[tuple(sorted((c, k), key=OA.index))] for k in o]
        G[:, J[c]] = pct(U[c] - a / 2.0 * (w[0] * U[o[0]] + w[1] * U[o[1]]))
    return G
def line(F, t, o, a):
    G = F.copy(); G[:, J[t]] = pct(pct(F[:, J[t]]) - a * pct(F[:, J[o]])); return G
r7 = LD.r7(); g_, _ = LD.gold(); gold = set(g_.StudyInstanceUID)
LABS = {'sol56': pd.read_csv(LD.R + '/_labels/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[L],
        'v2': LD.labels('v2')}
for nm, lab in LABS.items():
    if 'StudyInstanceUID' in getattr(lab, 'columns', []):
        lab = lab.set_index('StudyInstanceUID')
    lab = lab[L]
    ids = [u for u in r7.index if u not in gold and u in lab.index]
    X = r7.loc[ids, ['p|' + c for c in L]].values.astype(float); Y = (lab.loc[ids].fillna(0).values >= .5).astype(int)
    prev = Y[:, J['MCL']].mean()
    P('== labels %s: n %d non-gold R7 OOF studies, MCL prevalence %.3f (gold74 expert 11/74 = .149)' % (nm, len(ids), prev))
    pos = np.where(Y[:, J['MCL']] == 1)[0]; neg = np.where(Y[:, J['MCL']] == 0)[0]
    for mode in ('random', 'gold-like MCL prevalence'):
        for size in (390, 910):
            D = {k: [] for k in ('pw25', 'lomm05', 'MCLmm')}
            for _ in range(1000):
                if mode == 'random':
                    ix = rng.choice(len(ids), size, replace=False)
                else:
                    k1 = int(round(size * 11 / 74)); ix = np.concatenate([rng.choice(pos, k1, replace=False), rng.choice(neg, size - k1, replace=False)])
                F30 = pw(X[ix], .3); y = Y[ix]
                mac = lambda G, cols: sum(auc(G[:, J[c]], y[:, J[c]]) for c in cols) / 12
                F25 = pw(X[ix], .25)
                D['pw25'].append((mac(F25, OA) - mac(F30, OA)) * 1e4)
                D['lomm05'].append((mac(line(F30, 'Lateral OA', 'Medial Meniscus', .05), ['Lateral OA']) - mac(F30, ['Lateral OA'])) * 1e4)
                D['MCLmm'].append((mac(line(F30, 'MCL', 'Medial Meniscus', .1), ['MCL']) - mac(F30, ['MCL'])) * 1e4)
            A = np.array([D[k] for k in D])
            C = np.corrcoef(A)
            P('  %-26s n %3d | sd e-4: pw25 %.2f lomm05 %.2f MCLmm %.2f | mean: %+.2f %+.2f %+.2f | corr pw25~lomm05 %+.2f pw25~MCLmm %+.2f lomm05~MCLmm %+.2f' % (
                mode, size, *A.std(1), *A.mean(1), C[0, 1], C[0, 2], C[1, 2]))
P('READ_NOISE DONE')
OUT.close()
