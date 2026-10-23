# local_compose.py (10-19 ex): the box_followup transforms on (i) gold58 c4 final (m14_gate rebuild; consistency only, gold58 is
# NOT a selection ruler), (ii) R7 OOF on gold58 (expert labels), (iii) R7 OOF on train vs report_labels_v2 (the mechanism says
# decorrelation should LOSE there; borrow should not gain either if reports already carry the same-compartment coupling).
import numpy as np, pandas as pd
from scipy.stats import rankdata
R = 'C:/Users/eeshs/Downloads/rsna-knee'
G = R + '/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
LAB = ['Medial OA', 'Lateral OA', 'PF OA', 'Medial Meniscus', 'Lateral Meniscus']
pct = lambda v: rankdata(v) / len(v)


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def eoa(U, a):
    O = U.copy()
    for k in range(3):
        o = [m for m in range(3) if m != k]; O[:, k] = U[:, k] - a * (U[:, o[0]] + U[:, o[1]]) / 2
    return O


def transforms(F):
    U = np.stack([pct(F[:, k]) for k in range(5)], 1); out = {}
    for a in (.1, .15, .2):
        out['EOA%.2f' % a] = eoa(U, a)[:, :3]
    for a in (.1, .2):
        E = eoa(U, a)[:, :3]
        X = E.copy(); X[:, 1] = pct(E[:, 1]) - .1 * U[:, 3]; out['EOA%.2f+LM.1' % a] = X
        X = E.copy(); X[:, 0] = pct(E[:, 0]) + .1 * U[:, 3]; X[:, 1] = pct(E[:, 1]) + .1 * U[:, 4]; out['EOA%.2f+BOR.1' % a] = X
        X = E.copy(); X[:, 0] = pct(E[:, 0]) + .1 * U[:, 3]; X[:, 1] = pct(E[:, 1]) - .1 * U[:, 3] + .1 * U[:, 4]; out['EOA%.2f+LM.1+BOR.1' % a] = X
    X = U[:, :3].copy(); X[:, 1] = U[:, 1] - .1 * U[:, 3]; out['pair LOA<-MM .1'] = X
    X = U[:, :3].copy(); X[:, 0] = U[:, 0] + .1 * U[:, 3]; X[:, 1] = U[:, 1] + .1 * U[:, 4]; out['borrow MOA<-MM, LOA<-LM .1'] = X
    X = U[:, :3].copy(); X[:, 0] = U[:, 0] + .2 * U[:, 3]; X[:, 1] = U[:, 1] + .2 * U[:, 4]; out['borrow MOA<-MM, LOA<-LM .2'] = X
    X = U[:, :3].copy(); X[:, 0] = U[:, 0] - .1 * U[:, 3]; X[:, 1] = U[:, 1] - .1 * U[:, 4]; out['control: decor MOA<-MM, LOA<-LM .1'] = X
    return out


def evaluate(F, Y, tag, B=2000):
    TR = transforms(F); n = len(Y)
    base = np.array([auc(F[:, k], Y[:, k]) for k in range(3)])
    pt = {t: (np.array([auc(O[:, k], Y[:, k]) for k in range(3)]) - base) for t, O in TR.items()}
    rng = np.random.default_rng(0); BI = {t: [] for t in TR}
    if B:
        for _ in range(B):
            ii = rng.integers(0, n, n); y = Y[ii, :3]
            if (y.sum(0) < 2).any() or (y.sum(0) > n - 2).any(): continue
            b0 = np.array([auc(F[ii, k], y[:, k]) for k in range(3)])
            dd = {t: np.array([auc(O[ii, k], y[:, k]) for k in range(3)]) - b0 for t, O in TR.items()}
            for t in TR: BI[t].append(dd[t].sum() - dd['EOA0.10'].sum())
    print('[%s] n %d base OA AUC %s' % (tag, n, np.round(base, 4).tolist()))
    for t, d in pt.items():
        inc = d.sum() - pt['EOA0.10'].sum()
        print('   %-36s macro %+.5f per-label %s | vs EOA.1 %+.5f%s' % (t, d.sum() / 12, np.round(d, 4).tolist(), inc / 12,
              (' (boot P>0 %.2f)' % np.mean(np.array(BI[t]) > 0)) if B else ''))


g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
g58 = g[g.source == 'original_gold58'].reset_index(drop=True); ids = g58.StudyInstanceUID.tolist(); n58 = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
C4 = np.zeros((n58, 12))
for j, c in enumerate(L):
    w = C4W.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j])
    b = w * rr + (1 - w) * pct(R6[:, j]); C4[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
Y58 = g58[LAB].values.astype(int)
evaluate(C4[:, [L.index(c) for c in LAB]], Y58, 'gold58 c4 final')
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
F7 = r7[['p|' + c for c in LAB]].set_axis(LAB, axis=1)
evaluate(F7.loc[ids].values, Y58, 'gold58 R7 OOF')
F7t = F7[~F7.index.isin(g.StudyInstanceUID)]
rep = pd.read_csv(R + '/_labels/rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
V = pd.DataFrame({c: np.where(rep[c + '__verdict'] == 'YES', 1.0, np.where(rep[c + '__verdict'] == 'NO', 0.0, np.nan)) for c in LAB}, index=rep.index)
cm = F7t.index.intersection(V.dropna().index)
evaluate(F7t.loc[cm].values, V.loc[cm].values.astype(int), 'train R7 OOF vs report v2 (all 5 YES/NO)', B=300)
