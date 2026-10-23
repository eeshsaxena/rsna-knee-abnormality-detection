# pl_sim_b4.py: paired-bootstrap noise of gold58 deltas (target vs frontier point; student vs student) -> gate widths. CPU.
import os, numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.special import logit, expit
exec(open(os.path.expanduser('~/jump2/pl_sim_b3.py')).read().split('rows = []')[0])   # reuse loaders/helpers only
rng = np.random.default_rng(1); BOOT = [rng.integers(0, 58, 58) for _ in range(2000)]
def pb(X, Z):
    d = []
    for i in BOOT:
        v = [fauc(Y[i, j], X[n:, j][i]) - fauc(Y[i, j], Z[n:, j][i]) for j in range(12) if 0 < Y[i, j].mean() < 1]
        d.append(np.mean(v))
    d = np.array(d); return f'{d.mean():+.4f} sd {d.std():.4f} P>0 {(d > 0).mean():.3f}'
S_tm = own(['TTRmix']); S_em = own(['TTRmix', 'F1mix'])
w = W(.75, .5, .5)
R2_tm = (1 - w) * B4 + w * S_tm; R2_em = (1 - w) * B4 + w * S_em
F75 = .25 * B4 + .75 * BM; F100 = BM
print('R2 target (TTRmix UNK.75 NO.5 YES.5) vs frontier b=.75 :', pb(R2_tm, F75), '| vs mix50:', pb(R2_tm, F100))
print('R2 target (Emix   UNK.75 NO.5 YES.5) vs frontier b=.75 :', pb(R2_em, F75), '| vs mix50:', pb(R2_em, F100))
print('frontier b=.5 vs mix50 (Dread .25 vs .50):', pb(.5 * B4 + .5 * BM, F100))
P = {k: ST[k].values for k in ST}
for a, b in [('F1mix', 'TTRmix'), ('F1v4', 'TTRv4'), ('F1gated', 'F1v4'), ('F1mix', 'F1v4'), ('TTRmix', 'TTRv4'), ('F1mix', 'F1gated')]:
    print(f'student {a:7s} - {b:7s}:', pb(P[a], P[b]))
