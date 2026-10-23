# local_subclass.py (10-20 levers / rubric lens, Claude; research only, CPU, competition data only, no OAI)
# (1) Non-monotone rubric-negative subclasses on RSNA train OOF (report text, EN, regex): how high do OOF members rank them,
#     and the perfect-demotion bound on the rubric-mapped ruler. Subclasses: ACL mucoid / degeneration (no tear), ACL low-grade /
#     sprain, MCL chronic / thickened, MCL grade 1 / sprain, Effusion small, Baker's small (for contrast).
# (2) Facet leakage behind E-OA pw: P(Lateral OA label | lateral PATELLAR facet chondral only) vs base rates.
import re, sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, G, R, s6seed
ns = {}
src = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/levers/census_lang.py', encoding='utf-8').read()
exec(src.split('rows = []')[0], ns)
CATS, SS = ns['CATS'], ns['SS']
tr = pd.read_csv(R + '/_data/train.csv', dtype={'StudyInstanceUID': str})
gold_ids = set(tr.loc[tr['ACL'].notna(), 'StudyInstanceUID'])
U = [u for u in SS if u not in gold_ids]
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
M = {'R7': r7[['p|' + c for c in L]].set_axis(L, axis=1)}
for sd in (1044, 1045):
    M['s%d' % sd] = s6seed(sd)[0]
ids = pd.Index(U)
for k in M:
    ids = ids.intersection(M[k].dropna().index)
M['mean3'] = sum(pd.DataFrame({c: pct(M[k].loc[ids, c].values) for c in L}, index=ids) for k in ('R7', 's1044', 's1045')) / 3
cat = {(lab, c): np.array([bool(fn(SS[u])) for u in ids]) for lab, c, fn, rub in CATS}


def get(lab, name):
    return cat[(lab, name)]

SPEC = [
    ('ACL', ['complete / full / rupture', 'high-grade partial'], ['mucoid / degeneration / thickening, no tear'], 'mucoid / degeneration'),
    ('ACL', ['complete / full / rupture', 'high-grade partial'], ['low-grade / sprain / grade 1'], 'low-grade / sprain'),
    ('MCL', ['grade 3 / complete / high-grade'], ['chronic / old / thickened / Pellegrini'], 'chronic / thickened'),
    ('MCL', ['grade 3 / complete / high-grade'], ['grade 1 / sprain / periligamentous edema only'], 'grade 1 / sprain'),
    ('Effusion', ['moderate', 'large / massive / marked'], ['small / trace / minimal / mild only'], 'small'),
    ("Baker's", ['moderate / large'], ['small / tiny / minimal only'], 'small'),
]
print('non-gold EN train studies with OOF:', len(ids))
print('label | subclass g1 | model | n0 n1 n2 | AUC(g2 vs g0+g1) | AUC(g2 vs g0) | AUC(g2 vs g1) | AUC(g1 vs g0) | bound | headroom')
for lab, poscats, negcats, nm in SPEC:
    g2 = np.any([get(lab, c) for c in poscats], 0)
    g1 = np.any([get(lab, c) for c in negcats], 0) & ~g2
    # clean negatives: no category of this label fires at all
    anyc = np.any([cat[(l, c)] for (l, c) in cat if l == lab], 0)
    g0 = ~anyc
    for m in ('R7', 's1044', 's1045', 'mean3'):
        s = M[m].loc[ids, lab].values
        k = np.where(g2, 2, np.where(g1, 1, np.where(g0, 0, -1)))
        sel = k >= 0; kk = k[sel]; ss = s[sel]
        n0, n1, n2 = (kk == 0).sum(), (kk == 1).sum(), (kk == 2).sum()
        base = auc(ss, (kk == 2).astype(int))
        a20 = auc(ss[kk != 1], (kk[kk != 1] == 2).astype(int)); a21 = auc(ss[kk != 0], (kk[kk != 0] == 2).astype(int))
        a10 = auc(ss[kk != 2], (kk[kk != 2] == 1).astype(int))
        bound = (n0 * a20 + n1) / (n0 + n1)
        print(f'{lab:9s} | {nm:22s} | {m:6s} | {n0:4d} {n1:4d} {n2:4d} | {base:.4f} | {a20:.4f} | {a21:.4f} | {a10:.4f} | {bound:.4f} | {bound - base:+.4f}')

# (2) facet leakage
print('\nFACET LEAKAGE (EN, all studies incl. gold): P(label >= .5) by lateral-cartilage text class')
LATF = r"lateral (patellar )?facet|lateral patell|lateral trochle"
CH = r"(chondr|cartilag)"
LATTF = r"lateral (femoral condyle|tibial plateau|compartment|femorotibial|tibiofemoral|femoro-tibial|condyle|plateau)"
pos_hit = ns['pos_hit']
cls = {}
for u, ss in SS.items():
    f = any(re.search(LATF, s) and re.search(CH, s) for s in ss)
    t = any(re.search(LATTF, s) and pos_hit(s, CH) for s in ss)
    p = any(re.search(r"(patell|trochle|femoropatellar|patellofemoral|retropatellar)", s) and pos_hit(s, CH) for s in ss)
    cls[u] = 'facet only' if f and not t else ('lateral TF' if t else ('PF other (no lateral)' if p else 'no lateral / PF chondral'))
cls = pd.Series(cls)
for src_name, path in (('v4', R + '/_labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv'),
                       ('jev', R + '/session_2026-10-05/jump2/labels_in/jev_hybrid_scores.csv'),
                       ('sol56', R + '/_labels/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv')):
    lab = pd.read_csv(path, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
    t = pd.DataFrame({'cls': cls, 'LOA': (lab.reindex(cls.index)['Lateral OA'] >= .5).astype(float),
                      'PF': (lab.reindex(cls.index)['PF OA'] >= .5).astype(float)})
    print(src_name, t.groupby('cls').agg(n=('LOA', 'size'), LOA=('LOA', 'mean'), PF=('PF', 'mean')).round(3).to_dict('index'))
# same classes, OOF models: mean pct of the Lateral OA output by class (non-gold)
t = pd.DataFrame({'cls': cls.reindex(ids), 'R7_LOA': pct(M['R7'].loc[ids, 'Lateral OA'].values), 'mean3_LOA': M['mean3'].loc[ids, 'Lateral OA'].values})
print('OOF Lateral OA mean pct by class:', t.groupby('cls').mean().round(3).to_dict('index'))
