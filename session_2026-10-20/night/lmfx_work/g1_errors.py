# g1_errors.py (10-20 night, Track 3 LM / Fracture, Claude). RESEARCH ONLY, local CPU, competition data only (no OAI).
# gold58 error analysis for Lateral Meniscus and Fracture: member AUCs, discordant-pair decomposition, co-label strata,
# report-reader agreement, R7 state heads. gold58 is descriptive here, never a selection ruler.
import sys, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G, R, labels
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); n = len(ids); Y = g58[L].values.astype(int)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
C4 = np.zeros((n, 12)); RAP = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4W.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); RAP[:, j] = rr
    b = w * rr + (1 - w) * pct(R6[:, j]); C4[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
print('c4 rebuild gold58 macro %.5f (expect .94102)' % np.mean([auc(C4[:, j], Y[:, j]) for j in range(12)]))
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
R7 = r7.loc[ids, ['p|' + c for c in L]].values
rep = {s: labels(s).reindex(ids) for s in ('v2', 'v4', 'sol56')}
M = {'R6': R6, 'S6fall': S6, 'r384': R384, 'r224': R224, 'Raptor(.75/.25)': RAP, 'c4': C4, 'R7oof': R7}
rng = np.random.default_rng(20)
BI = [ix for ix in (rng.integers(0, n, n) for _ in range(3000))]

for lab in ('Lateral Meniscus', 'Fracture'):
    j = L.index(lab); y = Y[:, j]; P = y.sum()
    print('\n' + '=' * 30, lab, 'pos %d / %d' % (P, n))
    for k, F in M.items():
        a = auc(F[:, j], y); bs = [auc(F[ix, j], y[ix]) for ix in BI if 0 < y[ix].sum() < n]
        print('  %-16s AUC %.4f (boot sd %.4f)  Spearman vs c4 %.3f' % (k, a, np.std(bs), spearmanr(F[:, j], C4[:, j]).correlation))
    for s, d in rep.items():
        v = d[lab].values; ok = ~np.isnan(v)
        if ok.sum() and 0 < y[ok].sum() < ok.sum():
            tp = ((v == 1) & (y == 1) & ok).sum(); fn = ((v == 0) & (y == 1) & ok).sum(); fp = ((v == 1) & (y == 0) & ok).sum(); tn = ((v == 0) & (y == 0) & ok).sum()
            print('  report %-6s known %d: TP %d FN %d FP %d TN %d | sens %.2f spec %.2f | UNK among pos %d neg %d' % (
                s, ok.sum(), tp, fn, fp, tn, tp / max(tp + fn, 1), tn / max(tn + fp, 1), ((~ok) & (y == 1)).sum(), ((~ok) & (y == 0)).sum()))
    # paired member differences (bootstrap)
    for a_, b_ in (('Raptor(.75/.25)', 'R6'), ('S6fall', 'R6'), ('r384', 'r224'), ('c4', 'Raptor(.75/.25)')):
        d0 = auc(M[a_][:, j], y) - auc(M[b_][:, j], y)
        ds = [auc(M[a_][ix, j], y[ix]) - auc(M[b_][ix, j], y[ix]) for ix in BI if 0 < y[ix].sum() < n]
        print('  paired %s - %s: %+.4f (sd %.4f, P>0 %.2f)' % (a_, b_, d0, np.std(ds), np.mean(np.array(ds) > 0)))
    # discordant-pair decomposition on c4
    s = C4[:, j]; pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    disc = np.array([[1.0 if s[q] > s[p] else (.5 if s[q] == s[p] else 0.) for q in neg] for p in pos])   # pos x neg
    tot = disc.sum(); print('  c4 discordant pairs %.1f of %d (AUC %.4f)' % (tot, len(pos) * len(neg), 1 - tot / (len(pos) * len(neg))))
    fpos = disc.sum(1); fneg = disc.sum(0)
    o = np.argsort(-fpos); cum = np.cumsum(fpos[o]) / tot
    print('  positives: top-3 / top-5 carry %.0f%% / %.0f%% of discordance; positives with 0 discordance %d' % (100 * cum[2], 100 * cum[4], (fpos == 0).sum()))
    o2 = np.argsort(-fneg); cum2 = np.cumsum(fneg[o2]) / tot
    print('  negatives: top-3 / top-5 carry %.0f%% / %.0f%% of discordance; negatives with 0 discordance %d' % (100 * cum2[2], 100 * cum2[4], (fneg == 0).sum()))
    st = [c for c in r7.columns if c.startswith('state|%s|' % lab)]
    S = r7.loc[ids, st]; S.columns = [c.split('|')[2] for c in st]
    def row(i, kind, dval):
        co = [L[k][:4] for k in range(12) if Y[i, k] and k != j]
        return dict(kind=kind, disc=dval, idx=i, c4=round(pct(s)[i], 2), R6=round(pct(R6[:, j])[i], 2), S6=round(pct(S6[:, j])[i], 2),
                    r384=round(pct(R384[:, j])[i], 2), r224=round(pct(R224[:, j])[i], 2), R7=round(pct(R7[:, j])[i], 2),
                    v2=rep['v2'][lab].values[i], v4=rep['v4'][lab].values[i], r7state=S.iloc[i].idxmax(), r7sp=round(S.iloc[i].max(), 2),
                    co='+'.join(co))
    rows = [row(pos[k], 'POS-low', fpos[k]) for k in o[:8]] + [row(neg[k], 'NEG-high', fneg[k]) for k in o2[:8]]
    print(pd.DataFrame(rows).to_string(index=False))
    # co-label strata: c4 within-stratum AUC and positives' mean pct by co-label
    print('  co-label strata (c4 pct of positives / negatives when co-label = 1 vs 0; within-stratum AUC)')
    for k, c in enumerate(L):
        if k == j: continue
        x = Y[:, k]; out = []
        for v in (1, 0):
            m = x == v
            a = auc(s[m], y[m]) if 0 < y[m].sum() < m.sum() else np.nan
            out.append('%d: npos %2d nneg %2d posmean %.2f negmean %.2f AUC %s' % (v, y[m].sum(), (1 - y[m]).sum(),
                       pct(s)[m & (y == 1)].mean() if (m & (y == 1)).any() else np.nan, pct(s)[m & (y == 0)].mean() if (m & (y == 0)).any() else np.nan,
                       '%.3f' % a if a == a else ' n/a'))
        print('   %-16s | %s | %s' % (c, out[0], out[1]))
    # R7 state heads: per-state AUC on gold58 (descriptive)
    print('  R7 OOF state heads on gold58 (AUC of each state prob vs expert; mean prob pos / neg)')
    for c in S.columns:
        v = S[c].values; print('   %-38s AUC %.3f  mean pos %.3f neg %.3f' % (c, auc(v, y), v[y == 1].mean(), v[y == 0].mean()))
    print('   binary p|%s AUC %.3f' % (lab, auc(R7[:, j], y)))
