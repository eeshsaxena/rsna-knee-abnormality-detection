# oof_norr.py (10-19 skeptic lens, research only, CPU): does the OOF-ruler soft-OR gain survive in the kernel-natural build
# (intermediate family mix NOT re-ranked in either arm, so exponent 1 == the shipped rank recipe)? Same members / labels as
# stack2/nested_oof.py (R7 OOF + s44 + s45; v4 / v2 / sol56 report readers). Paired bootstrap over studies.
import sys, numpy as np
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import *
d7 = r7(); s44, _ = s6seed(1044); s45, _ = s6seed(1045)
ii = [i for i in s45.index if i in s44.index and i in d7.index]
A = d7.loc[ii, ['p|' + c for c in L]].values; B = s44.loc[ii, L].values; C = s45.loc[ii, L].values
PA = np.column_stack([pct(A[:, j]) for j in range(12)]); PB = np.column_stack([pct(B[:, j]) for j in range(12)])
PC = np.column_stack([pct(C[:, j]) for j in range(12)])


def nest(e, wR, rerank):
    fam = .5 * PB ** e + .5 * PC ** e
    if rerank: fam = np.column_stack([pct(fam[:, j]) for j in range(12)]) ** e
    return wR * PA ** e + (1 - wR) * fam


rng = np.random.default_rng(3); N = len(ii); boots = [rng.integers(0, N, N) for _ in range(300)]
print('n studies', N)
for src in ('v4', 'v2', 'sol56'):
    lab = labels(src).loc[ii, L].values
    def macro(S, ix):
        out = []
        for j in range(12):
            m = ~np.isnan(lab[ix, j]); y = lab[ix, j][m].astype(int)
            if 0 < y.sum() < len(y): out.append(auc(S[ix, j][m], y))
        return np.mean(out)
    for wR in (.5, .7):
        for rerank in (False, True):
            b0, b1 = nest(1, wR, rerank), nest(2, wR, rerank)
            full = macro(b1, np.arange(N)) - macro(b0, np.arange(N))
            bd = np.array([macro(b1, ix) - macro(b0, ix) for ix in boots])
            print('%-5s R7 %.1f rerank=%-5s pow2 vs exp1 (same structure): %+.5f  boot sd %.5f P>0 %.2f' % (src, wR, rerank, full, bd.std(), (bd > 0).mean()))
