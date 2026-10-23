# syn_site2.py (10-20 night, Claude). LOCAL, OOF train (non-gold), aggregates. Is the language effect in the Syn head a Syn-specific
# site prior, or just smaller effusions in those sites? Compare Syn and Eff head ranks among v2 Effusion-YES studies by language group,
# and the Syn-head residual after regressing on the Eff-head rank (OLS on rank-pct), by language group.
import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import s6seed
R = 'C:/Users/eeshs/Downloads/rsna-knee'; G = R + '/session_2026-10-18/next/g58'
pct = lambda v: rankdata(v) / len(v)
g74 = set(pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID)
lang = pd.read_csv(R + '/_data/_reports_lang.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['lang']
v2 = pd.read_csv(R + '/_labels/rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
mods = {'R7': r7[['p|Synovitis', 'p|Effusion']].set_axis(['Synovitis', 'Effusion'], axis=1)}
for s_ in (1044, 1045):
    mods['s%d' % s_] = s6seed(s_)[0][['Synovitis', 'Effusion']]
LOW = ['bg', 'hr', 'tr', 'nl']
for k, P in mods.items():
    P = P[~P.index.isin(g74)].dropna()
    df = pd.DataFrame({'syn': pct(P.Synovitis.values), 'eff': pct(P.Effusion.values)}, index=P.index)
    df['low'] = lang.reindex(df.index).isin(LOW).values
    df['effyes'] = (v2['Effusion__verdict'].reindex(df.index) == 'YES').values
    df['synyes'] = (v2['Synovitis__verdict'].reindex(df.index) == 'YES').values
    b = np.polyfit(df.eff, df.syn, 1); df['res'] = df.syn - np.polyval(b, df.eff)
    for nm, m in (('all', np.ones(len(df), bool)), ('v2 Eff YES', df.effyes.values), ('v2 Eff YES & v2 Syn not YES', (df.effyes & ~df.synyes).values)):
        a_, h_ = df[m & df.low.values], df[m & ~df.low.values]
        print('[%s] %-28s LOW n %4d syn %.3f eff %.3f res %+.3f | HIGH n %4d syn %.3f eff %.3f res %+.3f | res gap LOW-HIGH %+.3f' % (
            k, nm, len(a_), a_.syn.mean(), a_.eff.mean(), a_.res.mean(), len(h_), h_.syn.mean(), h_.eff.mean(), h_.res.mean(), a_.res.mean() - h_.res.mean()))
