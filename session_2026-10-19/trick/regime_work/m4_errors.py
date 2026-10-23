# m4_errors.py (research only): who does c4 mis-rank on gold58 for the low-AUC labels, and what do the report readers / R7 states say
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G, R
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(ids)[L].values
R6 = rd(G + '/R6_gold58_serve.csv'); S6 = rd(G + '/n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
P = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); b = w * rr + (1 - w) * pct(R6[:, j]); P[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
v4 = rd(R + '/_labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv')
d = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids]
tr = pd.read_csv(R + '/_data/train.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids]
for lab in ['Lateral OA', 'Synovitis', 'PF OA', 'Fracture']:
    j = L.index(lab)
    print('\n==== %s  c4 gold58 AUC %.4f  pos %d' % (lab, auc(P[:, j], Y[:, j]), Y[:, j].sum()))
    pos = np.where(Y[:, j] == 1)[0]; neg = np.where(Y[:, j] == 0)[0]
    # pairwise misorders contributed per study
    mis_neg = {i: int((P[pos, j] < P[i, j]).sum()) for i in neg}; mis_pos = {i: int((P[neg, j] > P[i, j]).sum()) for i in pos}
    worst = sorted(mis_neg.items(), key=lambda t: -t[1])[:5] + sorted(mis_pos.items(), key=lambda t: -t[1])[:5]
    sc = [c for c in d.columns if c.startswith('state|%s|' % lab)]
    for i, m in worst:
        if m == 0: continue
        st = d.iloc[i][sc].values.astype(float); top = np.argsort(-st)[:2]
        print('exp %d misorders %2d c4 %.2f | v4 %s=%.2f PFOA=%.2f LatOA=%.2f Eff=%.2f Syn=%.2f | expert PF %d Eff %d | R7 top states %s | rep: %s' % (
            Y[i, j], m, P[i, j], lab, v4[i, j], v4[i, 6], v4[i, 5], v4[i, 7], v4[i, 8], Y[i, 6], Y[i, 7],
            ['%s %.2f' % (sc[k].split('|')[2], st[k]) for k in top], str(tr.Report.iloc[i])[:260].replace('\n', ' ')))
