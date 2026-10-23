# r6_dump_audit.py (10-19, Claude, research only): R6 gold58 dump (box run_b = exact c4 cells 7-9 incl. flip TTA) audit:
# (1) p == float32 sigmoid(fold-mean z)? (2) mirror swap correctness for M/L labels, (3) saturation ties.
import json, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
R = 'C:/Users/eeshs/Downloads/rsna-knee'
B = R + '/session_2026-10-18/next/g58/r6g58_box/run_b/worker0/'
L = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
d = json.load(open(B + 'dump.json'))
p = pd.read_csv(B + 'predictions.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
g = pd.read_csv(R + '/session_2026-10-18/next/g58/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
ids = [s for s in p.index]
Z = np.array([d[s]['z'] for s in ids]); ZP = np.array([d[s]['z_plain'] for s in ids]); ZM = np.array([d[s]['z_mirror_raw'] for s in ids])
print('shapes', Z.shape)
zf, zp, zm = Z.mean(1), ZP.mean(1), ZM.mean(1)
sig = lambda x: 1 / (1 + np.exp(-x))
P = p[L].values
for j, c in enumerate(L):
    s32 = sig(zf[:, j].astype(np.float32)).astype(np.float32)
    print('%-17s |p - f32 sig(mean z)| max %.2e | n p>=1-1e-6 %2d | n p==1 %d | z range %.1f..%.1f' % (
        c, np.abs(P[:, j] - s32).max(), (P[:, j] >= 1 - 1e-6).sum(), (P[:, j] == 1).sum(), zf[:, j].min(), zf[:, j].max()))
y = g.loc[ids, L].values.astype(int)
def auc(s, yy):
    r = rankdata(s); a = yy.sum(); b = len(yy) - a
    return (r[yy == 1].sum() - a * (a + 1) / 2) / (a * b)
perm = [0, 1, 3, 2, 5, 4, 6, 7, 8, 9, 10, 11]
print('\nmirror audit (fold-mean z over 4 folds), gold58 AUC: plain | mirror swapped | mirror unswapped | kernel z | spearman(plain, mirror swapped) vs (plain, mirror unswapped)')
for j, c in enumerate(L):
    ms, mu = zm[:, perm[j]], zm[:, j]
    print('%-17s %.4f | %.4f | %.4f | %.4f | rho sw %.3f un %.3f' % (c, auc(zp[:, j], y[:, j]), auc(ms, y[:, j]), auc(mu, y[:, j]), auc(zf[:, j], y[:, j]),
          spearmanr(zp[:, j], ms)[0], spearmanr(zp[:, j], mu)[0]))
# kernel z check: z == (plain + mirror swapped)/2 except MCL plain
k = (ZP + ZM[:, :, perm]) / 2; k[:, :, 1] = ZP[:, :, 1]
print('kernel z reconstruct maxdiff (non-stacker cols):', np.abs(k - Z).max())
# saturation: p-rank vs z-rank AUC per label on gold58 (R6 alone)
print('\nR6 alone gold58: AUC(p as served) vs AUC(fold-mean z) [stacker cols use stacker p]')
for j, c in enumerate(L):
    print('%-17s p %.4f z %.4f diff %+.4f  ties in p: %d' % (c, auc(P[:, j], y[:, j]), auc(zf[:, j], y[:, j]), auc(zf[:, j], y[:, j]) - auc(P[:, j], y[:, j]),
          len(P) - len(np.unique(P[:, j]))))
