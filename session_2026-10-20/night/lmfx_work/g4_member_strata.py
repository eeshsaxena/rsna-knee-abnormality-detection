# g4_member_strata.py (10-20 night, Track 3, Claude). RESEARCH ONLY, local CPU, competition data only. Descriptive (gold58 n 58).
# Member AUC within co-label strata: Fracture by expert Contusion and by expert degenerative context; LM by expert trauma / ACL.
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); n = len(ids); Y = g58[L].values.astype(int)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
C4 = np.zeros((n, 12)); RAP = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4W.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); RAP[:, j] = rr
    C4[:, j] = .8 * pct(w * rr + (1 - w) * pct(R6[:, j])) + .2 * pct(S6[:, j])
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
R7 = r7.loc[ids, ['p|' + c for c in L]].values
M = {'R6': R6, 'S6fall': S6, 'Raptor': RAP, 'R7oof': R7, 'c4': C4}
col = lambda c: Y[:, L.index(c)]
trauma = (col('ACL') | col('Contusion') | col('MCL')).astype(bool); deg = (col('Medial OA') | col('Lateral OA') | col('PF OA')).astype(bool)
cont = col('Contusion').astype(bool)
for lab, strata in (('Fracture', {'all': np.ones(n, bool), 'Contusion+': cont, 'Contusion-': ~cont, 'any OA+': deg, 'no OA': ~deg}),
                    ('Lateral Meniscus', {'all': np.ones(n, bool), 'trauma+ (ACL|Cont|MCL)': trauma, 'trauma-': ~trauma, 'ACL+': col('ACL').astype(bool), 'ACL-': ~col('ACL').astype(bool), 'any OA+': deg, 'no OA': ~deg})):
    j = L.index(lab); y = Y[:, j]; print('==', lab)
    for sn, m in strata.items():
        s = '  %-24s n %2d pos %2d |' % (sn, m.sum(), y[m].sum())
        for k, F in M.items():
            a = auc(F[m, j], y[m]) if 0 < y[m].sum() < m.sum() else np.nan
            s += ' %s %.3f' % (k, a)
        print(s)
# Fracture: does Raptor's edge over R6 come from the Contusion+ stratum? paired bootstrap within stratum
rng = np.random.default_rng(4); j = L.index('Fracture'); y = Y[:, j]
for sn, m in (('Contusion+', cont), ('Contusion-', ~cont)):
    ii = np.where(m)[0]; ds = []
    for _ in range(2000):
        b = rng.choice(ii, len(ii))
        if 0 < y[b].sum() < len(b): ds.append(auc(RAP[b, j], y[b]) - auc(R6[b, j], y[b]))
    ds = np.array(ds); print('Fracture Raptor - R6 within %s: %+.3f (boot sd %.3f, P>0 %.2f)' % (sn, auc(RAP[m, j], y[m]) - auc(R6[m, j], y[m]), ds.std(), (ds > 0).mean()))
