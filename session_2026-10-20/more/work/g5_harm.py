# G5: harm of a definition-following hidden CSV (DEF flags) to c4 on gold58 (train studies, expert labels; no OAI), and what
# G-FLAG restores (G-FLAG(DEF) == BASE flags on every row by construction, g2_exposure.py). Members rerun on the box V100:
# Raptor r384 / r224 (GPU, BASE and DEF), R6 (DEF; BASE = run_b serve). S6fall NOT rerun (BASE in both arms, so the DEF harm
# is a LOWER bound: S6fall reads the flags like R6).
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G
W = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/more/work/'
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
def npy(tag, var):
    a = np.load(W + 'box/rap_%s_%s_raw.npy' % (var, tag)); i = pd.read_csv(W + 'box/rap_%s_%s_ids.csv' % (var, tag), dtype=str).StudyInstanceUID.tolist()
    return pd.DataFrame(a, index=i, columns=L).loc[ids].values
rd = lambda f: pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
M = {'r384': (npy('r384', 'base'), npy('r384', 'def')), 'r224': (npy('r224', 'base'), npy('r224', 'def')),
     'R6': (rd(G + '/R6_gold58_serve.csv'), rd(W + 'box/r6_def_predictions.csv'))}
S6 = rd(G + '/n42gpu_gold58.csv')
F = pd.read_csv(W + 'g2_def_flags.csv', dtype=str)
chg = F[(F.F_def != F.F_base) | (F.Fat_def != F.F_base)].StudyInstanceUID.unique()
aff = np.array([u in set(chg) for u in ids]); print('gold58 studies with >= 1 DEF-changed row: %d / %d' % (aff.sum(), n))
cpu = np.load(G + '/gold58_r384_kernelserve_cpu.npy')
print('sanity r384 GPU base vs CPU kernel serve: min per-label Spearman %.4f' % min(pd.Series(M['r384'][0][:, j]).corr(pd.Series(cpu[:, j]), method='spearman') for j in range(12)))
mac = lambda S: np.mean([auc(S[:, j], Y[:, j]) for j in range(12)])
for k, (b, d) in M.items():
    un = ~aff
    rho = [pd.Series(b[aff, j]).corr(pd.Series(d[aff, j]), method='spearman') for j in range(12)]
    print('%-5s macro base %.4f def %.4f (%+.4f) | unaffected max|dp| %.2e | affected: mean|dp| %.4f, per-label Spearman mean %.3f min %.3f'
          % (k, mac(b), mac(d), mac(d) - mac(b), np.abs(b[un] - d[un]).max(), np.abs(b[aff] - d[aff]).mean(), np.mean(rho), np.min(rho)))
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
def c4(r384, r224, r6, s6=S6):
    out = np.zeros((n, 12))
    for j, c in enumerate(L):
        w = C4.get(c, .5); rr = .75 * pct(r384[:, j]) + .25 * pct(r224[:, j]); b = w * rr + (1 - w) * pct(r6[:, j]); out[:, j] = .8 * pct(b) + .2 * pct(s6[:, j])
    return out
B = c4(M['r384'][0], M['r224'][0], M['R6'][0])
arms = {'DEF all members (S6fall base)': c4(M['r384'][1], M['r224'][1], M['R6'][1]),
        'DEF Raptor only': c4(M['r384'][1], M['r224'][1], M['R6'][0]),
        'DEF R6 only': c4(M['r384'][0], M['r224'][0], M['R6'][1])}
rng = np.random.default_rng(11); bs = [rng.integers(0, n, n) for _ in range(2000)]
def bmac(S, ix):
    v = [auc(S[ix, j], Y[ix, j]) for j in range(12) if 0 < Y[ix, j].sum() < len(ix)]; return np.mean(v)
print('\nc4 gold58 macro BASE (= G-FLAG under DEF) %.5f' % mac(B))
for k, S in arms.items():
    per = np.array([auc(S[:, j], Y[:, j]) - auc(B[:, j], Y[:, j]) for j in range(12)])
    dd = np.array([bmac(S, ix) - bmac(B, ix) for ix in bs])
    print('%-30s macro %.5f delta %+.5f | boot sd %.5f P(delta<0) %.2f | worst labels %s' % (k, mac(S), mac(S) - mac(B), dd.std(), (dd < 0).mean(),
          ', '.join('%s %+.4f' % (L[j], per[j]) for j in np.argsort(per)[:3])))
# rank perturbation of the final c4 output on affected studies
S = arms['DEF all members (S6fall base)']
print('final c4 output, affected studies: per-label Spearman(base, def) mean %.3f min %.3f; mean |pct shift| %.3f'
      % (np.mean([pd.Series(B[aff, j]).corr(pd.Series(S[aff, j]), method='spearman') for j in range(12)]),
         np.min([pd.Series(B[aff, j]).corr(pd.Series(S[aff, j]), method='spearman') for j in range(12)]),
         np.abs(np.column_stack([pct(B[:, j]) for j in range(12)]) - np.column_stack([pct(S[:, j]) for j in range(12)]))[aff].mean()))
