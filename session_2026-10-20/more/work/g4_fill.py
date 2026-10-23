# G4: failed-row fill in c4 stage 3 (cell 14 l.75-77): a study one member failed takes the OTHER member's percentile as is
# (F0, c4). Alternatives that are exact no-ops at 0 failures: F1 quantile map of that percentile onto the blend's own distribution
# (rank-neutral), F2 regression imputation of the missing member's percentile from the present one (E[r_missing | r_present] by
# 20-bin means over the scored rows; label-free, test-only), F3 0.5. Members: TTR s1044 OOF (Raptor-recipe slot model) and R7 OOF
# (R6 family), 4,333 train studies, c4 per-label Raptor weights. Labels: 3 report-derived sets (mechanism ruler, not selection).
import pandas as pd, numpy as np, sys
from sklearn.metrics import roc_auc_score
B = 'C:/Users/eeshs/Downloads/rsna-knee/'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
J = pd.read_pickle(B + 'session_2026-10-19/trick/mp_work/JT.pkl')
labs = {'jev_bin': B + 'session_2026-10-05/jump2/labels_in/jev_hybrid_binary.csv',
        'v4blend': B + '_labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv'}
Ys = {k: (pd.read_csv(v, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(J.index)[L] > 0.5).astype(float).values for k, v in labs.items()}
WR = {'Medial OA': .5, 'Lateral OA': .2, 'PF OA': .5, 'Synovitis': .8, "Baker's": .5, 'Lateral Meniscus': .4, 'Fracture': .65}
pct = lambda v: pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values
def pct_on(v, ok):
    r = np.full(len(v), np.nan); r[ok] = pct(np.asarray(v, np.float64)[ok]); return r
def cond_mean(x_ok, y_ok, x_q, nb=20):                       # E[y | x] by quantile bins of x over the scored rows
    e = np.quantile(x_ok, np.linspace(0, 1, nb + 1)); e[0], e[-1] = -np.inf, np.inf
    bi = np.clip(np.searchsorted(e, x_ok, side='right') - 1, 0, nb - 1); mu = np.array([y_ok[bi == k].mean() if (bi == k).any() else .5 for k in range(nb)])
    return mu[np.clip(np.searchsorted(e, x_q, side='right') - 1, 0, nb - 1)]
TA = J[['t|' + c for c in L]].values; TB = J[['p|' + c for c in L]].values; N = len(J)
def blend(fa, fb, mode):
    out = np.zeros((N, 12))
    for j, c in enumerate(L):
        ra, rb = pct_on(TA[:, j], ~fa), pct_on(TB[:, j], ~fb); w = WR.get(c, .5); b = w * ra + (1 - w) * rb
        ok = ~fa & ~fb
        for miss, pres, rp, rm, wp in ((fb & ~fa, 'a', ra, rb, w), (fa & ~fb, 'b', rb, ra, 1 - w)):
            if not miss.any(): continue
            if mode == 'F0': b[miss] = rp[miss]
            elif mode == 'F1': b[miss] = np.quantile(b[ok], rp[miss])
            elif mode == 'F2': b[miss] = wp * rp[miss] + (1 - wp) * cond_mean(rp[ok], rm[ok], rp[miss])
            elif mode == 'F3': b[miss] = .5
        b[fa & fb] = .5; out[:, j] = b
    return out
def macro(S, Y): return np.mean([roc_auc_score(Y[:, j], S[:, j]) for j in range(12)])
clean = blend(np.zeros(N, bool), np.zeros(N, bool), 'F0')
for k, Y in Ys.items(): print('%s clean c4-shaped macro %.5f' % (k, macro(clean, Y)))
rng = np.random.default_rng(7); rows = []
for X in (0.005, 0.01, 0.02, 0.05):
    for who in ('B (R6) fails', 'A (Raptor) fails'):
        for rep in range(20):
            f = rng.random(N) < X; fa, fb = (np.zeros(N, bool), f) if who.startswith('B') else (f, np.zeros(N, bool))
            S = {m: blend(fa, fb, m) for m in ('F0', 'F1', 'F2', 'F3')}
            for k, Y in Ys.items():
                base = macro(S['F0'], Y)
                rows.append(dict(X=X, who=who, labels=k, F1=macro(S['F1'], Y) - base, F2=macro(S['F2'], Y) - base, F3=macro(S['F3'], Y) - base,
                                 loss_F0=macro(clean, Y) - base))
            # exact no-op check at 0 failures is structural (the miss masks are empty); verify once
T = pd.DataFrame(rows)
G = T.groupby(['X', 'who', 'labels'])[['loss_F0', 'F1', 'F2', 'F3']].agg(['mean']).round(6)
print(G.to_string())
P = T.groupby(['X', 'who'])[['F1', 'F2']].agg(lambda s: (s > 0).mean()).round(2)
print('\nP(delta > 0) over reps x label sets:\n' + P.to_string())
z = blend(np.zeros(N, bool), np.zeros(N, bool), 'F2'); print('\nno-op at 0 failures (F1/F2 identical to F0):', bool((z == clean).all()))
T.to_csv(B + 'session_2026-10-20/more/work/g4_fill.csv', index=False)
