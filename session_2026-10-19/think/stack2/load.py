# load.py (10-19 stacking lens, Claude, research only): shared loaders for the OOF ruler (train studies, report labels) and gold58.
import glob, numpy as np, pandas as pd
from scipy.stats import rankdata
R = 'C:/Users/eeshs/Downloads/rsna-knee'
G = R + '/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


pct = lambda v: rankdata(v) / len(v)
lg = lambda p, e=1e-7: np.log(np.clip(p, e, 1 - e) / (1 - np.clip(p, e, 1 - e)))


def s6seed(seed, kind='ema'):
    fs = sorted(glob.glob(R + '/session_2026-10-*/kaggle_frac75*/out_*/f*_s%d_ttr_frac75/oof_f*_s%d_%s.csv' % (seed, seed, kind)))
    d = pd.concat([pd.read_csv(f, dtype={'StudyInstanceUID': str}) for f in fs]).drop_duplicates('StudyInstanceUID')
    return d.set_index('StudyInstanceUID')[L], fs


def r7():
    d = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    return d


def gold():
    g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
    return g, g[g.source == 'original_gold58'].reset_index(drop=True)


def labels(src='v2'):
    """Report-regime labels on train studies, as a DataFrame of 1 / 0 / NaN (NaN = unknown / dropped)."""
    if src == 'v2':
        lab = pd.read_csv(R + '/_labels/rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
        out = pd.DataFrame(index=lab.index)
        for c in L:
            v = lab[c + '__verdict']; out[c] = np.where(v == 'YES', 1.0, np.where(v == 'NO', 0.0, np.nan))
        return out
    f = {'v4': '/_labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv',
         'sol56': '/_labels/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv'}[src]
    lab = pd.read_csv(R + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    return (lab[L] >= 0.5).astype(float).where(lab[L].notna())
