# local_e1_diversity.py (10-20 more, Claude): RESEARCH ONLY, local CPU. Control for lens 1: is E1's report-OOF gain a
# member-DIVERSITY effect? Compare E1 (flat mean of pct^2 vs pct) on the report OOF ruler for (a) R7 + s1044 + s1045
# (cross-family, the THINK_1019 ruler) and (b) s1044 + s1045 only (same S6 family, like the OAI stand-ins), with the mean
# between-member Spearman per label as the diversity measure. Writes only its own .log.
import sys, numpy as np
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import *
from scipy.stats import spearmanr
d7 = r7(); s44, _ = s6seed(1044); s45, _ = s6seed(1045)
ii = [i for i in s45.index if i in s44.index and i in d7.index]
A = d7.loc[ii, ['p|' + c for c in L]].values.astype(float); B = s44.loc[ii, L].values.astype(float); C = s45.loc[ii, L].values.astype(float)
N = len(ii); rng = np.random.default_rng(41); boots = [rng.integers(0, N, N) for _ in range(300)]
pc = lambda X: np.column_stack([pct(X[:, j]) for j in range(12)])
PA, PB, PC = pc(A), pc(B), pc(C)
def sp(X, Y): return np.mean([spearmanr(X[:, j], Y[:, j]).correlation for j in range(12)])
print('n %d | mean per-label Spearman: R7-s44 %.3f R7-s45 %.3f s44-s45 %.3f' % (N, sp(A, B), sp(A, C), sp(B, C)))
for src in ('v4', 'v2', 'sol56'):
    lab = labels(src).loc[ii, L].values.astype(float)
    def mac(S, ix):
        out = []
        for j in range(12):
            m = ~np.isnan(lab[ix, j]); y = lab[ix, j][m].astype(int)
            if 5 <= y.sum() <= len(y) - 5: out.append(auc(S[ix, j][m], y))
        return np.mean(out) if False else np.sum(out) / 12.
    for nm, Ps in (('R7+s44+s45', [PA, PB, PC]), ('s44+s45', [PB, PC]), ('R7+s44', [PA, PB])):
        S1 = np.mean(Ps, 0); S2 = np.mean([p ** 2 for p in Ps], 0)
        f = mac(S2, np.arange(N)) - mac(S1, np.arange(N)); bd = np.array([mac(S2, ix) - mac(S1, ix) for ix in boots])
        print('%-5s %-11s E1 flat pow2 vs exp1 %+.5f boot sd %.5f P>0 %.2f' % (src, nm, f, bd.std(), (bd > 0).mean()), flush=True)
