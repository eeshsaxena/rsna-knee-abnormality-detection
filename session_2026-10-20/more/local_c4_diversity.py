# local_c4_diversity.py (10-20 more): mean per-label Spearman between c4's real members on gold58 (n 58; context for E1's
# diversity dependence). Writes only its .log.
import sys, numpy as np, pandas as pd, itertools
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import *
from scipy.stats import spearmanr
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist()
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values.astype(float)
M = {'R6': rd('R6_gold58_serve.csv'), 'S6': rd('n42gpu_gold58.csv'), 'r384': np.load(G + '/gold58_r384_kernelserve_cpu.npy').astype(float),
     'r224': np.load(G + '/gold58_r224_kernelserve.npy').astype(float), 'R7': r7().loc[ids, ['p|' + c for c in L]].values.astype(float),
     's43': rd('s43_gold58.csv')}
for a, b in itertools.combinations(M, 2):
    print('gold58 mean per-label Spearman %-5s - %-5s %.3f' % (a, b, np.nanmean([spearmanr(M[a][:, j], M[b][:, j]).correlation for j in range(12)])))
