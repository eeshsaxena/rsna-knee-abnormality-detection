# pl_sim_b5.py: the 10-04 BIG_JUMP 'DSD-combo' (own UNK+NO .75 + Raptor YES .25) re-scored with the REAL fold-0 fleet as
# the own teacher (10-04 used gold-selected crop R50s), plus its F0 kill test. CPU, read-only.
import os, numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.special import logit, expit
exec(open(os.path.expanduser('~/jump2/pl_sim_b3.py')).read().split('rows = []')[0])
RA = rap.loc[ROWS].values; La = lg(rap.loc[v4.index].values); Lv = lg(v4.values)
RAm = expit((lg(RA) - La.mean(0)) / La.std(0) * Lv.std(0) + Lv.mean(0))      # logit-matched over all 4407, as build_gated
F = {b: (1 - b) * B4 + b * BM for b in [0, .25, .5, .625, .75, .875, 1.0]}
Ff = sorted([(ws(X), g58(X)) for X in F.values()])
def line(w): return np.interp(w, [a for a, _ in Ff], [b for _, b in Ff])
for nm, keys in [('TTRv4', ['TTRv4']), ('Ev4', ['TTRv4', 'F1v4'])]:
    S = own(keys); sil = np.isin(VV, ['UNK', 'NO']); yes = VV == 'YES'
    own_only = np.where(sil, .25 * B4 + .75 * S, B4)
    combo = np.where(sil, .25 * B4 + .75 * S, .75 * B4 + .25 * RAm)
    combo5 = np.where(sil, .25 * B4 + .75 * S, .5 * B4 + .5 * RAm)
    for t, X in [('own UNK+NO .75 (DSD-own)', own_only), ('DSD-combo (+Raptor YES .25)', combo), ('DSD-combo (+Raptor YES .5)', combo5)]:
        w_ = ws(X); g = g58(X)
        print(f'{nm:6s} {t:30s} gold58 {g:.4f} (v4 {g58(B4):.4f}, mix50 {g58(BM):.4f}) ws {w_:.3f} rho {rho(X):.3f} | line@ws {line(w_):.4f} diff {g - line(w_):+.4f}')
print('10-04 plan F0 kill rule: own UNK+NO .75 must gain >= +0.008 over v4; DSD-combo must be >= mix50 gold58 at lower within-class rho')
