# G5c: c4-level per-label deltas of the DEF arms on gold58, and which studies drive R6's Effusion change
import sys, numpy as np, pandas as pd
src = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/more/work/g5_harm.py').read()
exec(src.split("rng = np.random.default_rng(11)")[0].replace("for k, (b, d) in M.items():", "for k, (b, d) in []:"))
for k, S in arms.items():
    per = np.array([auc(S[:, j], Y[:, j]) - auc(B[:, j], Y[:, j]) for j in range(12)])
    print('%-30s %s' % (k, ' '.join('%s %+.4f' % (L[j][:6], per[j]) for j in range(12))))
e = L.index('Effusion'); b6, d6 = M['R6']
F2 = pd.read_csv(W + 'g2_def_flags.csv', dtype=str)
nofs = set(F2[(F2.Anatomical_Plane == 'Sagittal')].groupby('StudyInstanceUID').F_base.apply(lambda s: (s == '1').sum() == 0).loc[lambda s: s].index)
ch = np.abs(d6[:, e] - b6[:, e]) > 1e-6
print('R6 Effusion changed on %d studies; of them no-FS-sagittal (miss path in BASE): %d; expert Effusion of those: %s'
      % (ch.sum(), sum(u in nofs for u in np.array(ids)[ch]), Y[ch, e].tolist()))
print('their R6 Effusion pct base -> def:', [(round(float(pct(b6[:, e])[i]), 3), round(float(pct(d6[:, e])[i]), 3)) for i in np.flatnonzero(ch)])
print('c4 Effusion AUC base %.4f | DEF-all %.4f | DEF-R6-only %.4f' % (auc(B[:, e], Y[:, e]), auc(arms['DEF all members (S6fall base)'][:, e], Y[:, e]), auc(arms['DEF R6 only'][:, e], Y[:, e])))
# drivers: leave-one-study-out contribution to the R6 Effusion AUC change
base_auc, def_auc = auc(b6[:, e], Y[:, e]), auc(d6[:, e], Y[:, e])
contrib = []
for i in np.flatnonzero(ch):
    d_ = d6[:, e].copy(); d_[i] = b6[i, e]            # undo the DEF change for study i only
    contrib.append((def_auc - auc(d_, Y[:, e]), ids[i] in nofs, int(Y[i, e])))
contrib.sort(key=lambda t: -abs(t[0]))
print('R6 Effusion AUC base %.4f def %.4f; top single-study contributions (delta, no-FS-sag?, expert y):' % (base_auc, def_auc), [(round(c, 4), f, y) for c, f, y in contrib[:6]])
