# m15_samestudy.py: SAME 58 studies, SAME c4 predictions, two label regimes: expert (gold58) vs report readers (>= .5).
# If OA decor helps under expert labels and hurts under report labels on the same studies, the gain is a label-regime effect.
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/trick/regime_work')
src = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/trick/regime_work/m14_gate.py').read()
exec(src[:src.index('for a in (0.')])
from load import R
LB = R + '/_labels/'
rdl = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(ids)[L].astype(float).values
REG = {'expert': Y.astype(float), 'sol56': rdl(LB + 'rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv'),
       'v4blend': rdl(LB + 'rsna-knee-llm-report-labels/llm_labels_v4_blend.csv'), 'jevhyb': rdl(R + '/session_2026-10-05/jump2/labels_in/jev_hybrid_scores.csv'),
       'v2api': rdl(LB + 'rsna-knee-llm-labels/report_labels_v2.csv')}
for name, Z in REG.items():
    out = []
    for a in (.1, .2):
        O = decor(F, a); d = 0.
        for t in OA:
            m = ~np.isnan(Z[:, t]); y = (Z[m, t] >= .5).astype(int)
            d += auc(O[m, t], y) - auc(F[m, t], y)
        out.append(d / 12)
    print('%-8s labels on the same 58 studies: OA-decor macro delta a.1 %+.5f a.2 %+.5f' % (name, out[0], out[1]))
