# cand_oam_local.py (10-20 ex, Claude). CANDIDATE oa_meniscus on RSNA-side data (local CPU, no OAI data):
#  (i)   gold58 c4 final (m14_gate rebuild, same recipe as local_compose.py; consistency only, gold58 is NOT a selection ruler)
#  (ii)  R7 OOF on gold58 and on gold74 (expert labels; consistency)
#  (iii) R7 OOF on train (gold74 removed) vs report_labels_v2 YES/NO (all 5 labels known). Mechanism check: a decorrelation that
#        removes REPORT-inherited excess coupling should LOSE here; if reports do not over-couple the pair there is no mechanism.
# Transform (identical to cand_oam_box_template.py): E = E-OA(e) on the OA trio, then
#   MOA' = pct(E_MOA) - a pct(f_MM), LOA' = pct(E_LOA) - a pct(f_LM), MM' = pct(f_MM) - a pct(f_MOA), LM' = pct(f_LM) - a pct(f_LOA).
# Increment = sum over touched labels of AUC(candidate) - AUC(E-OA(e) alone); macro = /12. Paired bootstrap P>0.
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
R = 'C:/Users/eeshs/Downloads/rsna-knee'
G = R + '/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
LAB = ['Medial OA', 'Lateral OA', 'PF OA', 'Medial Meniscus', 'Lateral Meniscus']; SH = ['MOA', 'LOA', 'PF', 'MM', 'LM']
PARTNER = {0: 3, 1: 4, 3: 0, 4: 1}
ES = (.1, .2); AS = (.1, .2, -.1, -.2)
pct = lambda v: rankdata(v) / len(v)


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def eoa(U, e):
    O = U.copy()
    for k in range(3):
        o = [m for m in range(3) if m != k]; O[:, k] = U[:, k] - e * (U[:, o[0]] + U[:, o[1]]) / 2
    return O


def cols(F):
    U = np.stack([pct(F[:, k]) for k in range(5)], 1); C = {'U': U}
    for e in ES:
        E = eoa(U, e); C[('E', e)] = E
        for a in AS:
            for k in (0, 1): C[('OAs', e, a, k)] = pct(E[:, k]) - a * U[:, PARTNER[k]]
    for a in AS:
        for k in (3, 4): C[('MENs', a, k)] = U[:, k] - a * U[:, PARTNER[k]]
    return C


def incs(C, Y, ii):
    """Per-label increments over E-OA(e) for every (e, a), on rows ii. Labels with NaN are dropped per label."""
    out = {}
    U = C['U']
    def A(s, k):
        y = Y[ii, k]; m = ~np.isnan(y); yy = y[m].astype(int)
        if yy.sum() < 2 or yy.sum() > len(yy) - 2: return np.nan
        return auc(s[ii][m], yy)
    b0 = {k: A(U[:, k], k) for k in (3, 4)}
    men = {a: np.array([A(C[('MENs', a, k)], k) - b0[k] for k in (3, 4)]) for a in AS}
    for e in ES:
        E = C[('E', e)]; be = {k: A(E[:, k], k) for k in (0, 1)}
        bu = np.array([A(U[:, k], k) for k in range(3)]); bE = np.array([A(E[:, k], k) for k in range(3)])
        out[('EOA', e)] = np.nansum(bE - bu)
        for a in AS:
            o = np.array([A(C[('OAs', e, a, k)], k) - be[k] for k in (0, 1)])
            out[(e, a)] = np.r_[o, men[a]]   # MOA, LOA, MM, LM
    return out


def evaluate(F, Y, tag, B=2000, seed=0):
    C = cols(F); n = len(Y); full = incs(C, Y, np.arange(n))
    U = C['U']
    base = [auc(U[~np.isnan(Y[:, k]), k], Y[~np.isnan(Y[:, k]), k].astype(int)) for k in range(5)]
    rho = [spearmanr(F[:, 0], F[:, 3])[0], spearmanr(F[:, 1], F[:, 4])[0]]
    print('\n[%s] n %d | prev %s | base AUC %s | model rho MOA-MM %.3f LOA-LM %.3f | E-OA .1 macro %+.5f, .2 %+.5f' % (
        tag, n, np.round(np.nanmean(Y, 0), 3).tolist(), np.round(base, 4).tolist(), rho[0], rho[1], full[('EOA', .1)] / 12, full[('EOA', .2)] / 12))
    rng = np.random.default_rng(seed); BS = {k: [] for k in full if k[0] != 'EOA'}
    for _ in range(B):
        ii = rng.integers(0, n, n); d = incs(C, Y, ii)
        if any(np.isnan(v).any() for k, v in d.items() if k[0] != 'EOA'): continue
        for k in BS: BS[k].append(d[k])
    res = {}
    for e in ES:
        for a in AS:
            v = full[(e, a)]; bs = np.array(BS[(e, a)])
            both = bs.sum(1); oa = bs[:, :2].sum(1); mn = bs[:, 2:].sum(1); med = bs[:, 0] + bs[:, 2]; lat = bs[:, 1] + bs[:, 3]
            res[(e, a)] = v
            print('   on E-OA %.1f, a %+.1f: BOTH %+.5f (P %.2f) | OAside %+.5f (P %.2f) | MENside %+.5f (P %.2f) | medial %+.5f (P %.2f) lateral %+.5f (P %.2f) | per-label AUC MOA %+.4f LOA %+.4f MM %+.4f LM %+.4f (nboot %d)' % (
                e, a, v.sum() / 12, (both > 0).mean(), v[:2].sum() / 12, (oa > 0).mean(), v[2:].sum() / 12, (mn > 0).mean(),
                (v[0] + v[2]) / 12, (med > 0).mean(), (v[1] + v[3]) / 12, (lat > 0).mean(), *v, len(bs)))
    return res


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
Y58all = g58[L].values.astype(float)
mac = np.mean([auc(C4[:, j], Y58all[:, j].astype(int)) for j in range(12)])
print('gold58 c4 rebuild 12-label macro %.5f (parity target m14_gate.log .94102)' % mac)
Y58 = g58[LAB].values.astype(float)
evaluate(C4[:, [L.index(c) for c in LAB]], Y58, 'gold58 c4 final')
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
F7 = r7[['p|' + c for c in LAB]].set_axis(LAB, axis=1)
evaluate(F7.loc[ids].values, Y58, 'gold58 R7 OOF')
i74 = [s for s in g.StudyInstanceUID if s in F7.index]
evaluate(F7.loc[i74].values, g.set_index('StudyInstanceUID').loc[i74, LAB].values.astype(float), 'gold74 R7 OOF (NaN labels dropped per label)')
F7t = F7[~F7.index.isin(g.StudyInstanceUID)]
rep = pd.read_csv(R + '/_labels/rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
V = pd.DataFrame({c: np.where(rep[c + '__verdict'] == 'YES', 1.0, np.where(rep[c + '__verdict'] == 'NO', 0.0, np.nan)) for c in LAB}, index=rep.index)
cm = F7t.index.intersection(V.dropna().index)
evaluate(F7t.loc[cm].values, V.loc[cm].values.astype(float), 'train R7 OOF vs report v2 (all 5 YES/NO)', B=300)
cm2 = F7t.index.intersection(V.index)
evaluate(F7t.loc[cm2].values, V.loc[cm2].values.astype(float), 'train R7 OOF vs report v2 (UNK dropped per label)', B=300)
