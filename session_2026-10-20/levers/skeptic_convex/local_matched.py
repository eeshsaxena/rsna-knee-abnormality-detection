# skeptic_convex/local_matched.py (10-20 skeptic of convex E-OA). Competition data only.
# Is "p2 removes more report coupling than linear at a .2" a mechanism signal or just a strength difference?
# Compare p2 vs linear at MATCHED decorrelation strength: the label-free output coupling rho_out (mean pairwise
# normal-score correlation of the 3 transformed OA columns). Mechanism predicts: at matched rho_out, p2 loses MORE on
# report labels and gains MORE on expert labels. Rulers: R7 OOF x report labels (train minus gold74, n 4333);
# R7 OOF x gold74 expert (n 74, INFO); c4 x gold58 expert (n 58, INFO).
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
D = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58/'
LBL = 'C:/Users/eeshs/Downloads/rsna-knee/_labels/rsna-knee-llm-labels/report_labels_v2.csv'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
OA = ['Medial OA', 'Lateral OA', 'PF OA']; IOA = [L.index(c) for c in OA]
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5}
pct = lambda v: rankdata(v) / len(v)
pctc = lambda X: np.stack([pct(X[:, j]) for j in range(X.shape[1])], 1)
AG = np.round(np.arange(0, .601, .01), 3)


def auc(s, y):
    m = ~np.isnan(y); s = s[m]; y = y[m].astype(int)
    p = y.sum(); q = len(y) - p
    r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def eoah(U, a, p=1.):
    S = U ** p; out = np.empty_like(U)
    for t in range(3):
        o = [k for k in range(3) if k != t]
        out[:, t] = U[:, t] - a * (S[:, o[0]] + S[:, o[1]]) / 2.
    return out


def rho_out(S):
    Z = norm.ppf((rankdata(S, axis=0) - .5) / len(S)); C = np.corrcoef(Z.T)
    return np.mean([C[0, 1], C[0, 2], C[1, 2]])


def oam(S, Y):
    return np.mean([auc(S[:, k], Y[:, k]) for k in range(3)])


def curve(U, Y, p):
    b = oam(U, Y)
    return np.array([rho_out(eoah(U, a, p)) for a in AG]), np.array([oam(eoah(U, a, p), Y) - b for a in AG])


def matched(U, Y, name, scale=3 / 12):
    r1, d1 = curve(U, Y, 1.); r2, d2 = curve(U, Y, 2.)
    print('%s | rho_out at a 0 %.3f | lin a .2 -> rho %.3f, p2 a .2 -> rho %.3f, p2 a .15 -> rho %.3f' % (
        name, r1[0], r1[20], r2[20], r2[15]))
    rows = []
    for a_l in (.1, .15, .2, .25, .3):
        j = int(round(a_l * 100)); target = r1[j]
        # p2 a giving the same rho_out (rho decreasing in a): interpolate on the p2 curve
        o = np.argsort(r2); a_p2 = float(np.interp(target, r2[o], AG[o])); dp2 = float(np.interp(target, r2[o], d2[o]))
        rows.append('lin a %.2f (rho %.3f): lin %+.5f | p2 matched a %.3f: %+.5f | p2-lin %+.5f' % (
            a_l, target, d1[j] * scale, a_p2, dp2 * scale, (dp2 - d1[j]) * scale))
    print('   ' + '\n   '.join(rows))


R7 = pd.read_csv(D + 'R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
RL = pd.read_csv(LBL, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
g = pd.read_csv(D + 'gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
ix = R7.index.intersection(RL.index).difference(g.StudyInstanceUID)
F7 = R7.loc[ix, ['p|' + c for c in OA]].values.astype(float); YR = RL.loc[ix, OA].values.astype(float)
ok = np.isfinite(F7).all(1) & np.isfinite(YR).all(1); F7 = F7[ok]; YR = (YR[ok] >= .5).astype(float)
print('units: macro (OA-mean delta x 3/12)')
matched(pctc(F7), YR, 'R7 OOF x REPORT labels, train minus gold74, n %d' % len(F7))
gi = g.StudyInstanceUID.tolist(); U74 = pctc(R7.loc[gi, ['p|' + c for c in OA]].values.astype(float)); Y74 = g[OA].values.astype(float)
matched(U74, Y74, 'R7 OOF x gold74 EXPERT (INFO), n 74')
# c4 gold58
g58 = g[g.source == 'original_gold58'].reset_index(drop=True); ids = g58.StudyInstanceUID.tolist(); Y58 = g58[OA].values.astype(float)
R6 = pd.read_csv(D + 'R6_gold58_serve.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R384 = np.load(D + 'gold58_r384_kernelserve_cpu.npy'); R224 = np.load(D + 'gold58_r224_kernelserve.npy')
S6 = pd.read_csv(D + 'n42gpu_gold58.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
RAP = .75 * pctc(R384) + .25 * pctc(R224)
C4F = np.zeros((58, 3))
for jj, c in enumerate(OA):
    j = L.index(c); w = C4[c]; b = w * RAP[:, j] + (1 - w) * pct(R6[:, j]); C4F[:, jj] = .8 * pct(b) + .2 * pct(S6[:, j])
matched(pctc(C4F), Y58, 'c4 x gold58 EXPERT (INFO), n 58')
