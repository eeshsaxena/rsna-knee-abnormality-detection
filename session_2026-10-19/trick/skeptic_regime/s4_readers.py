# s4_readers.py: are the 'independent' report readers on gold58 actually distinct for the OA labels (binarized >= .5)?
import numpy as np, pandas as pd
G = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58'; R = 'C:/Users/eeshs/Downloads/rsna-knee'
OA = ['Medial OA', 'Lateral OA', 'PF OA']
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}); ids = g[g.source == 'original_gold58'].StudyInstanceUID.tolist()
src = {'sol56': R + '/_labels/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv', 'v4blend': R + '/_labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv',
       'jevhyb': R + '/session_2026-10-05/jump2/labels_in/jev_hybrid_scores.csv', 'v2api': R + '/_labels/rsna-knee-llm-labels/report_labels_v2.csv'}
B = {k: (pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(ids)[OA].astype(float) >= .5).astype(int) for k, p in src.items()}
ks = list(B)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        print('%-8s vs %-8s OA cells differing: %d of %d' % (ks[i], ks[j], int((B[ks[i]].values != B[ks[j]].values).sum()), B[ks[i]].size))
