# local_facet.py (10-20 levers, Claude; research only, CPU, competition data only): model-level facet confusion.
# Among EN non-gold train studies whose LLM labels say Lateral OA = 0 (v4 and jev both < .5), is the OOF Lateral OA output higher
# when the report describes lateral PATELLAR facet / trochlea chondral damage only (no lateral TF chondral sentence) than when it
# describes other PF chondral damage with no lateral mention? Same for Medial OA with the medial facet. Mann-Whitney AUC + boot.
import re, sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, G, R, s6seed
ns = {}
exec(open('census_lang.py', encoding='utf-8').read().split('CATS = []')[0], ns)
SS, pos_hit = ns['SS'], ns['pos_hit']
tr = pd.read_csv(R + '/_data/train.csv', dtype={'StudyInstanceUID': str}); gold = set(tr.loc[tr['ACL'].notna(), 'StudyInstanceUID'])
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
M = {'R7': r7[['p|' + c for c in L]].set_axis(L, axis=1), 's1044': s6seed(1044)[0], 's1045': s6seed(1045)[0]}
v4 = pd.read_csv(R + '/_labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
jev = pd.read_csv(R + '/session_2026-10-05/jump2/labels_in/jev_hybrid_scores.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
CH = r"(chondr|cartilag)"
rng = np.random.default_rng(1020)
for side, lab in (('lateral', 'Lateral OA'), ('medial', 'Medial OA')):
    other = 'medial' if side == 'lateral' else 'lateral'
    FAC = side + r" (patellar )?facet|" + side + r" patell|" + side + r" trochle"
    TF = side + r" (femoral condyle|tibial plateau|compartment|femorotibial|tibiofemoral|femoro-tibial|condyle|plateau)"
    cls = {}
    for u, ss in SS.items():
        if u in gold: continue
        f = any(re.search(FAC, s) and re.search(CH, s) for s in ss)
        t = any(re.search(TF, s) and pos_hit(s, CH) for s in ss)
        p = any(re.search(r"(patell|trochle|femoropatellar|patellofemoral|retropatellar)", s) and pos_hit(s, CH) for s in ss)
        anyside = any(re.search(side, s) and re.search(CH, s) for s in ss)
        if f and not t: cls[u] = 'facet_only'
        elif p and not anyside: cls[u] = 'PF_no_side'
    cls = pd.Series(cls)
    neg = cls.index[(v4.reindex(cls.index)[lab] < .5) & (jev.reindex(cls.index)[lab] < .5)]
    for m, D in M.items():
        ii = neg.intersection(D.dropna().index)
        s = pd.Series(pct(D.loc[D.dropna().index, lab].values), index=D.dropna().index).loc[ii].values
        y = (cls.loc[ii] == 'facet_only').astype(int).values
        bs = [auc(s[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(1000)) if 0 < y[i].sum() < len(y)]
        print(f'{lab:10s} {m:6s} label-negative studies: facet_only n {y.sum()} vs PF_no_side n {(1 - y).sum()} | '
              f'mean pct {s[y == 1].mean():.3f} vs {s[y == 0].mean():.3f} | AUC {auc(s, y):.3f} (boot sd {np.std(bs):.3f}, P>.5 {np.mean(np.array(bs) > .5):.2f})')
