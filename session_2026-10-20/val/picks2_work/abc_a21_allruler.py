# abc_a21_allruler.py (10-20 val Track D): abc_a21.py primary (PFT, N01, logN) with the LB kernel restricted to the ALL-teams ruler
# draws of lb_post_pooled.npz (first third; eoa20 gap to .960 0.8e-4 [0.4, 1.4] instead of the pooled 1.0 [0.5, 2.5]). Sensitivity only.
import numpy as np, abc_a21 as M
A = M.A
LB = np.load('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push/work/lb_post_pooled.npz'); n3 = len(LB['G10']) // 3
r = np.random.default_rng(12); J = r.choice(n3, 400, replace=False)
A.LBG = np.stack([LB['G10'][J], LB['G20'][J], LB['gap960'][J]], 1) * 1e-3
rng = np.random.default_rng(1022); tag = 'PFT_N01_logN_ALLruler'
D = M.collect('PFT', 'N01', 'logN')
M.show(M.summ(D, D['w'], tag + ' PRE-A21', rng)); w1 = D['w'] * M.a21_like(D); M.show(M.summ(D, w1, tag + ' POST-A21', rng))
p = w1 / w1.sum(); j = rng.choice(len(p), 200000, p=p)
np.savez(M.OUT + 'abc_a21_draws_%s.npz' % tag, Pop=D['Pop'][j], G=D['G'][j], gap=D['gap'][j], C=D['C'], VARS=np.array(M.VARS),
         pre_Pop=D['Pop'][rng.choice(len(D['w']), 200000, p=D['w'] / D['w'].sum())])
