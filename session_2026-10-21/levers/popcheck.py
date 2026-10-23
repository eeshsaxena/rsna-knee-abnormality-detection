# popcheck.py (10-21 levers, Claude). RESEARCH ONLY, local, NO OAI. Is gold58 a random-like sample of the train image population?
# Label-free test on R7 OOF predictions (R7 covers both): statistic S = sum over 66 pairs of (rho_ns(sample) - rho_ns(train))^2 and the
# per-label mean percentile of the predictions; null = 1,000 random 58-study subsets of the non-gold train rows (tr3). Writes popcheck.log.
import numpy as np
from rg_lib import L, AB, load_all, ns, LD, pct
D = load_all(); g, g58 = LD.gold(); ids58 = g58.StudyInstanceUID.tolist()
r7 = LD.r7(); c = ['p|' + x for x in L]
Xtr = r7.loc[D['tr3'], c].values.astype(float); Xg = r7.loc[ids58, c].values.astype(float)
allX = np.vstack([Xtr, Xg]); P_ = np.column_stack([pct(allX[:, j]) for j in range(12)])   # common percentile scale
Ptr, Pg = P_[:len(Xtr)], P_[len(Xtr):]
def rhos(M):
    Z = np.column_stack([ns(M[:, j]) for j in range(12)]); C = np.corrcoef(Z.T); return C[np.triu_indices(12, 1)]
rt = rhos(Ptr); mt = Ptr.mean(0)
rg = rhos(Pg); Sg = ((rg - rt) ** 2).sum(); Mg = ((Pg.mean(0) - mt) ** 2).sum()
rng = np.random.default_rng(7); Sn, Mn = [], []
for _ in range(1000):
    ii = rng.choice(len(Ptr), 58, replace=False); Sn.append(((rhos(Ptr[ii]) - rt) ** 2).sum()); Mn.append(((Ptr[ii].mean(0) - mt) ** 2).sum())
Sn = np.array(Sn); Mn = np.array(Mn)
out = open('popcheck.log', 'w')
def P(s): print(s); out.write(s + '\n')
P('R7 co-firing structure: S(gold58) %.3f vs null median %.3f [p90 %.3f, p99 %.3f]; P(null >= gold58) %.3f' % (Sg, np.median(Sn), np.quantile(Sn, .9), np.quantile(Sn, .99), (Sn >= Sg).mean()))
P('R7 mean predicted percentile per label: M(gold58) %.4f vs null median %.4f [p99 %.4f]; P %.3f' % (Mg, np.median(Mn), np.quantile(Mn, .99), (Mn >= Mg).mean()))
P('per-label mean percentile gold58 vs train (0.5 = same): ' + ' '.join('%s %.2f' % (AB[x], v) for x, v in zip(L, Pg.mean(0))))
iu = np.triu_indices(12, 1); big = np.argsort(-np.abs(rg - rt))[:10]
P('largest pair drifts (R7 rho gold58 - train): ' + ' | '.join('%s-%s %+.2f' % (AB[L[iu[0][k]]], AB[L[iu[1][k]]], rg[k] - rt[k]) for k in big))
