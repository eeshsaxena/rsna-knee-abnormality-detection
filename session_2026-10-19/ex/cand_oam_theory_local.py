# Theory diagnostic on RSNA-side sets (local): rho = corr(ns f_t, ns f_o) vs q = corr(ns f_o, y_t) / corr(ns f_t, y_t).
# Decorrelation t <- o helps iff q < rho (MODEL, as skeptic box_reweight.py). Bootstrap sd of (rho - q) on gold58 / gold74.
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
exec(open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/cand_oam_local.py', encoding='utf-8').read().split("g = pd.read_csv(G + '/gold74_eval_set.csv'")[0])
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
def diag(F, Y, tag, B=1000):
    def one(ii):
        out = []
        for t, o in PARTNER.items():
            m = ~np.isnan(Y[ii, t]); f = F[ii][m]; y = Y[ii, t][m]
            zt, zo = ns(f[:, t]), ns(f[:, o]); rho = np.corrcoef(zt, zo)[0, 1]
            q = np.corrcoef(zo, y)[0, 1] / np.corrcoef(zt, y)[0, 1]; out.append((rho, q))
        return np.array(out)
    full = one(np.arange(len(Y))); rng = np.random.default_rng(0); bs = []
    for _ in range(B):
        ii = rng.integers(0, len(Y), len(Y))
        try:
            r = one(ii)
            if np.isfinite(r).all(): bs.append(r[:, 0] - r[:, 1])
        except Exception: pass
    bs = np.array(bs)
    print('[%s] ' % tag + ' | '.join('%s<-%s rho %.3f q %.3f (rho-q %+.3f, P>0 %.2f)' % (SH[t], SH[o], full[i, 0], full[i, 1], full[i, 0] - full[i, 1], (bs[:, i] > 0).mean())
                                     for i, (t, o) in enumerate(PARTNER.items())))
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
Y58 = g58[LAB].values.astype(float)
diag(C4[:, [L.index(c) for c in LAB]], Y58, 'gold58 c4')
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
F7 = r7[['p|' + c for c in LAB]].set_axis(LAB, axis=1)
diag(F7.loc[ids].values, Y58, 'gold58 R7')
i74 = [s for s in g.StudyInstanceUID if s in F7.index]
diag(F7.loc[i74].values, g.set_index('StudyInstanceUID').loc[i74, LAB].values.astype(float), 'gold74 R7')
F7t = F7[~F7.index.isin(g.StudyInstanceUID)]
rep = pd.read_csv(R + '/_labels/rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
V = pd.DataFrame({c: np.where(rep[c + '__verdict'] == 'YES', 1.0, np.where(rep[c + '__verdict'] == 'NO', 0.0, np.nan)) for c in LAB}, index=rep.index)
cm = F7t.index.intersection(V.dropna().index)
diag(F7t.loc[cm].values, V.loc[cm].values.astype(float), 'train R7 vs report v2', B=200)
