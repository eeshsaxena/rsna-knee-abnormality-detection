# moa_check.py (10-21 levers, Claude). RESEARCH ONLY, local, NO OAI. Direct expert-label reads (INFO, n small) and a ceiling check for the
# MOA <- Contusion / Fracture lines on the pw30 base: (1) c4 final output on gold58 (kernel path, cbase) and R7 OOF on gold58 / gold74 / the
# 16 extras, MOA AUC change with paired bootstrap P; (2) base MOA AUC of each train-OOF ruler under REP / EB weights vs c4 gold58, to scale
# ruler gains by remaining misranked mass. Writes moa_check.log.
import json
import numpy as np
from scipy.stats import rankdata
from rg_lib import L, J, OA, eoa, pct, load_all, LD, near_pd, copula_cells, weights_from_target, wauc_multi
out = open('moa_check.log', 'w')
def P(s): print(s); out.write(s + '\n')
def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p; return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
D = load_all(); c4pw = eoa(D['c4'], .3, 'pw'); Y58 = D['Y58']; Y74 = D['Y74']
g, g58 = LD.gold(); ids74 = g.StudyInstanceUID.tolist(); ids58 = g58.StudyInstanceUID.tolist(); ex = [i for i in ids74 if i not in set(ids58)]
r7 = LD.r7(); c = ['p|' + x for x in L]
R = {'R7 g58': (eoa(r7.loc[ids58, c].values.astype(float), .3, 'pw'), g58[L].values.astype(float)),
     'R7 g74': (eoa(r7.loc[ids74, c].values.astype(float), .3, 'pw'), Y74),
     'R7 extra16 (report-read labels)': (eoa(r7.loc[ex, c].values.astype(float), .3, 'pw'), g.set_index('StudyInstanceUID').loc[ex, L].values.astype(float)),
     'c4 g58 (kernel path)': (c4pw, Y58.astype(float))}
TR = {'Cont .05': [('Contusion', .05)], 'Cont .10': [('Contusion', .10)], 'Frac .05': [('Fracture', .05)], 'Frac .10': [('Fracture', .10)],
      'Cont .05 + Frac .05': [('Contusion', .05), ('Fracture', .05)], 'placebo Eff .05': [('Effusion', .05)], 'placebo ACL .05': [('ACL', .05)]}
rng = np.random.default_rng(5)
jm = J['Medial OA']
for nm, (F, Yx) in R.items():
    y = Yx[:, jm]; ok = ~np.isnan(y); F = F[ok]; y = y[ok].astype(int)
    P('== %s: n %d, MOA positives %d, base MOA AUC %.4f' % (nm, len(y), y.sum(), auc(F[:, jm], y)))
    for tn, parts in TR.items():
        col = F[:, jm].copy()
        for o, a in parts: col = pct(pct(col) - a * pct(F[:, J[o]]))
        d = auc(col, y) - auc(F[:, jm], y); bs = []
        for _ in range(2000):
            ii = rng.integers(0, len(y), len(y))
            if y[ii].sum() in (0, len(ii)): continue
            bs.append(auc(col[ii], y[ii]) - auc(F[ii, jm], y[ii]))
        P('   MOA <- %-20s d %+.4f (label) = %+.2fe-4 macro | boot P>0 %.2f, P<0 %.2f' % (tn, d, d / 12 * 1e4, np.mean(np.array(bs) > 0), np.mean(np.array(bs) < 0)))
# (2) ceiling: base MOA AUC of rulers on train under REP and EB weights (trio + Cont + Frac raking, dropUNK rows)
cal = json.load(open('regime_cal.json')); dl = json.load(open('regime_dlt.json')); REL = dl['rel']['nontrio']
Trep = np.array(cal['Trep']); Dm = np.zeros((12, 12))
for k, e in dl['pairs'].items():
    a, b = k.split(' | '); Dm[J[a], J[b]] = Dm[J[b], J[a]] = e['delta']
trio = lambda i, j: L[i] in OA and L[j] in OA
mD = np.mean([Dm[i, j] for i in range(12) for j in range(i + 1, 12) if not trio(i, j)])
De = np.array([[0 if i == j else (Dm[i, j] if trio(i, j) else mD + REL * (Dm[i, j] - mD)) for j in range(12)] for i in range(12)])
Y = D['Y']; cols = OA + ['Contusion']; jc = [J[x] for x in cols]; m_ = ~np.isnan(Y[:, jc]).any(1); Ys = Y[m_][:, jc].astype(int)
W = [np.ones(len(Ys)), weights_from_target(Ys, copula_cells(Ys.mean(0), near_pd(Trep[np.ix_(jc, jc)] + De[np.ix_(jc, jc)])))]
P('')
P('(2) ceiling: base MOA AUC (pw30) of train-OOF rulers, rows with trio + Contusion known: REP / EB')
for mn, F in D['X'].items():
    Bm = eoa(F, .3, 'pw')[m_]; b = wauc_multi(Bm[:, jm], Ys[:, 0], W)
    P('   %-6s REP %.4f EB %.4f (misranked mass %.3f)' % (mn, b[0], b[1], 1 - b[1]))
