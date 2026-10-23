# cand_menisci_local.py (10-20 ex, Claude). RESEARCH ONLY, local CPU, prints aggregates.
# Menisci decorrelation MM' = pct(f_MM) - a pct(f_LM), LM' = pct(f_LM) - a pct(f_MM); macro delta = (dMM + dLM) / 12.
#   (i)   gold58 c4 final (m14_gate rebuild, parity macro .94102): consistency only, gold58 is NOT a selection ruler
#   (ii)  R7 OOF on gold58 and on gold74 (expert labels; report-trained heliosli family)
#   (iii) R7 OOF on train (minus gold74) vs report_labels_v2 YES/NO (mechanism check: should LOSE)
#   plus the linear-Gaussian theory read: rho_ns, q = corr(z_o, y_t) / corr(z_t, y_t), a* = (rho - q) / (1 - rho q).
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, norm
R = 'C:/Users/eeshs/Downloads/rsna-knee'
G = R + '/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
MEN = ['Medial Meniscus', 'Lateral Meniscus']
pct = lambda v: rankdata(v) / len(v)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
A = (-.1, .05, .1, .15, .2)


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def men_tr(F2, a):
    U = pct(F2[:, 0]); V = pct(F2[:, 1])
    return np.stack([U - a * V, V - a * U], 1)


def theory(F2, Y2):
    Z = np.stack([ns(F2[:, 0]), ns(F2[:, 1])], 1); rz = np.corrcoef(Z.T)[0, 1]; out = [rz]
    for t, o in ((0, 1), (1, 0)):
        ct = np.corrcoef(Z[:, t], Y2[:, t])[0, 1]; co = np.corrcoef(Z[:, o], Y2[:, t])[0, 1]; q = co / ct
        out += [q, (rz - q) / (1 - rz * q)]
    return np.array(out)   # rho, q_MM, a*_MM, q_LM, a*_LM


def evaluate(F2, Y2, tag, B=2000, seed=0):
    n = len(Y2); outs = {a: men_tr(F2, a) for a in A}
    base = np.array([auc(F2[:, k], Y2[:, k]) for k in range(2)])
    d0 = {a: np.array([auc(O[:, k], Y2[:, k]) for k in range(2)]) - base for a, O in outs.items()}
    rng = np.random.default_rng(seed); BD = {a: [] for a in A}; TH = []
    for _ in range(B):
        ii = rng.integers(0, n, n); y = Y2[ii]
        if (y.sum(0) < 2).any() or (y.sum(0) > n - 2).any(): continue
        b0 = np.array([auc(F2[ii, k], y[:, k]) for k in range(2)])
        for a, O in outs.items(): BD[a].append((np.array([auc(O[ii, k], y[:, k]) for k in range(2)]) - b0).sum())
        if len(TH) < 500: TH.append(theory(F2[ii], y))
    th = theory(F2, Y2); ths = np.std(TH, 0)
    print('[%s] n %d prev MM %.3f LM %.3f | label phi %+.3f | model Spearman %+.3f | base AUC MM %.4f LM %.4f' % (
        tag, n, *Y2.mean(0), np.corrcoef(Y2.T)[0, 1], spearmanr(F2[:, 0], F2[:, 1]).correlation, *base))
    print('   theory: rho_ns %+.3f (sd %.3f) | MM q %+.3f a* %+.3f (sd %.3f) | LM q %+.3f a* %+.3f (sd %.3f)' % (
        th[0], ths[0], th[1], th[2], ths[2], th[3], th[4], ths[4]))
    for a in A:
        v = np.array(BD[a])
        print('   a %+.2f%s macro %+.5f | MM %+.4f LM %+.4f | boot sd(macro) %.5f P>0 %.3f' % (
            a, '' if a in (-.1, .1, .2) else ' (descr)', d0[a].sum() / 12, d0[a][0], d0[a][1], v.std() / 12, (v > 0).mean()))


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
Y58 = g58[L].values.astype(int)
print('c4 rebuild gold58 macro %.5f (m14_gate.log: .94102)' % np.mean([auc(C4[:, j], Y58[:, j]) for j in range(12)]))
jm = [L.index(c) for c in MEN]
evaluate(C4[:, jm], Y58[:, jm], 'gold58 c4 final')
for nm, M in (('R6', R6), ('S6fall', S6)):
    evaluate(M[:, jm], Y58[:, jm], 'gold58 %s (c4 member, descr)' % nm, B=500)

r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
F7 = r7[['p|' + c for c in MEN]].set_axis(MEN, axis=1)
evaluate(F7.loc[ids].values, Y58[:, jm], 'gold58 R7 OOF')
g74 = g.set_index('StudyInstanceUID')[MEN].dropna(); i74 = g74.index.intersection(F7.index)
evaluate(F7.loc[i74].values, g74.loc[i74].values.astype(int), 'gold74 R7 OOF')
F7t = F7[~F7.index.isin(g.StudyInstanceUID)]
rep = pd.read_csv(R + '/_labels/rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
V = pd.DataFrame({c: np.where(rep[c + '__verdict'] == 'YES', 1.0, np.where(rep[c + '__verdict'] == 'NO', 0.0, np.nan)) for c in MEN}, index=rep.index)
cm = F7t.index.intersection(V.dropna().index)
evaluate(F7t.loc[cm].values, V.loc[cm].values.astype(int), 'train R7 OOF vs report v2 (both YES/NO)', B=1000)
V0 = V.fillna(0.0); cm0 = F7t.index.intersection(V0.index)
evaluate(F7t.loc[cm0].values, V0.loc[cm0].values.astype(int), 'train R7 OOF vs report v2 (UNK = 0)', B=1000)
