# G5b: per-label and bootstrap detail of the member-level DEF effects on gold58 (same inputs as g5_harm.py)
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/more/work')
exec(open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/more/work/g5_harm.py').read().split("C4 = {")[0].split("for k, (b, d) in M.items():")[0])
rng = np.random.default_rng(3); bs = [rng.integers(0, n, n) for _ in range(2000)]
def bm(S, ix): return np.mean([auc(S[ix, j], Y[ix, j]) for j in range(12) if 0 < Y[ix, j].sum() < len(ix)])
for k, (b, d) in M.items():
    per = np.array([auc(d[:, j], Y[:, j]) - auc(b[:, j], Y[:, j]) for j in range(12)])
    dd = np.array([bm(d, ix) - bm(b, ix) for ix in bs])
    print('%-5s member macro delta %+.4f boot sd %.4f P(<0) %.2f | per label: %s' % (k, per.mean(), dd.std(), (dd < 0).mean(),
          ' '.join('%s %+.4f' % (L[j][:6], per[j]) for j in range(12) if abs(per[j]) > 1e-9)))
fb = pd.read_csv(W + 'box/r6_def_fluid_stacker.csv', dtype={'StudyInstanceUID': str})
print('R6 DEF fluid stacker columns:', list(fb.columns)[:12])
