# box_facet_skeptic.py (10-20 levers, skeptic of "facet confusion supports E-OA pw"; Claude). BOX ONLY, read-only, CPU.
# Prints AGGREGATES ONLY (n, AUC, deltas, sd, P). Never prints an id, a row or a per-knee value; writes nothing.
# Independent expert ruler for the facet mechanism: OAI MOAKS cartilage subregions (area.ft codes; COMP as box_coupling_run.py).
#  T1 side contrast: knees whose lateral TF is CLEAN (all 5 lateral TF subregions area 0) and with PF damage on exactly one side:
#     lateral PF only (lp / alf damaged, mp / amf area 0) vs medial PF only. Does the LOA output rank lateral-PF-only higher?
#     Medial analogue with a clean medial TF. Then the same with the model's PF output removed (residual, slope from the pool).
#  T2 RSNA-style contrast: clean lateral TF; lateral-PF-damaged (any mp / amf allowed) vs PF damaged with lateral PF area 0.
#  T3 the decision on the same ruler (raw, unweighted): pw20 - sym20 and pw30 - sym30 AUC on LOA / MOA / PF, P_FT and P_AREA.
import os
import numpy as np, pandas as pd
from scipy.stats import rankdata

H = os.path.expanduser('~')
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
       "Baker's", 'Contusion', 'Fracture']
OA = ['Medial OA', 'Lateral OA', 'PF OA']
COMP = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'],
        'PF OA': ['mp', 'lp', 'amf', 'alf']}
PFL, PFM = ['lp', 'alf'], ['mp', 'amf']
W_PW = {('Medial OA', 'Lateral OA'): .876, ('Medial OA', 'PF OA'): .708, ('Lateral OA', 'PF OA'): 1.416}
B = int(os.environ.get('BOOT', '1000'))
rng = np.random.default_rng(1020)
pct = lambda v: rankdata(v) / len(v)


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def rdp(p):
    d = pd.read_csv(p, dtype={'StudyInstanceUID': str})
    return d.drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')


S = pd.read_csv(H + '/oai_build/oai_label_sources.csv', dtype={'kid': str}).set_index('kid')
MOD = {}
for k, p in (('R6', '/r5_oai/run_r6/worker0/predictions.csv'), ('S6fall', '/r5_oai/s6_r5/out/predictions.csv'),
             ('D5', '/oai_build/teacher/teacher_D5_mean_v00.csv'),
             ('s1044', '/oai_build/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv'),
             ('s1045', '/oai_build/teacher2/f0_s1045_ttr288frac75__raw__A__gpu.csv')):
    try:
        d = rdp(H + p)[OA].astype(float).dropna(); MOD[k] = d[d.index.isin(S.index)]
    except Exception as e:
        print('missing', k, type(e).__name__)
regs = sum(COMP.values(), [])
C = S[['m_cm_' + r for r in regs]].astype(float)
C = C[np.isfinite(C.values).all(axis=1)]
area = pd.DataFrame(np.floor(C.values + 1e-6), index=C.index, columns=regs)
ft = pd.DataFrame(np.round((C.values - area.values) * 10), index=C.index, columns=regs)
common = C.index
for k in MOD: common = common.intersection(MOD[k].index)
print('knees with all 14 cartilage subregions and all %d models: %d' % (len(MOD), len(common)))
A, F = area.loc[common], ft.loc[common]
Y = {'P_FT': pd.DataFrame({c: (F[COMP[c]] >= 1).any(axis=1).astype(int) for c in OA}),
     'P_AREA': pd.DataFrame({c: (A[COMP[c]] >= 2).any(axis=1).astype(int) for c in OA})}
print('prevalence P_FT %s | P_AREA %s' % ({c: round(float(Y['P_FT'][c].mean()), 3) for c in OA}, {c: round(float(Y['P_AREA'][c].mean()), 3) for c in OA}))
U = {k: pd.DataFrame({c: pct(MOD[k].loc[common, c].values) for c in OA}, index=common) for k in MOD}


def boot_auc(s, y):
    n = len(y); a = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < n: a.append(auc(s[i], y[i]))
    a = np.array(a); return a.std(), (a > .5).mean()


for dmg_name, dmg in (('any loss (area>=1)', lambda R: (A[R] >= 1).any(axis=1)), ('deep (area>=2 or FT)', lambda R: ((A[R] >= 2) | (F[R] >= 1)).any(axis=1))):
    print('\n=== PF damage definition: %s' % dmg_name)
    for side, lab, own, other in (('lateral', 'Lateral OA', PFL, PFM), ('medial', 'Medial OA', PFM, PFL)):
        clean = (A[COMP[lab]] == 0).all(axis=1)
        ownd = dmg(own); othz = (A[other] == 0).all(axis=1); othd = dmg(other); ownz = (A[own] == 0).all(axis=1)
        # T1: own-side PF only vs other-side PF only
        g1 = clean & ownd & othz; g0 = clean & othd & ownz
        # T2: own-side PF damaged (other side free) vs PF damaged with own side area 0
        h1 = clean & ownd; h0 = clean & othd & ownz
        for tname, a1, a0 in (('T1 own-PF-only vs other-PF-only', g1, g0), ('T2 own-PF-damaged vs other-PF-only', h1, h0)):
            sel = (a1 | a0).values; y = a1.values[sel].astype(int)
            if y.sum() < 5 or (1 - y).sum() < 5:
                print('  %s %s: too few (n %d / %d)' % (lab, tname, y.sum(), (1 - y).sum())); continue
            row = []
            for k in MOD:
                s = U[k].loc[common[sel], lab].values; pf = U[k].loc[common[sel], 'PF OA'].values
                pool = clean.values  # slope of own-OA output on PF output over all clean-own-TF knees
                bb = np.polyfit(U[k].loc[common[pool], 'PF OA'].values, U[k].loc[common[pool], lab].values, 1)[0]
                a_raw = auc(s, y); sd, P = boot_auc(s, y); a_res = auc(s - bb * pf, y); sdr, Pr = boot_auc(s - bb * pf, y)
                row.append('%s raw %.3f (sd %.3f, P>.5 %.2f) PF-resid %.3f (P>.5 %.2f) PFout %.3f/%.3f' % (k, a_raw, sd, P, a_res, Pr, pf[y == 1].mean(), pf[y == 0].mean()))
            print('  %-10s %s | n %d vs %d' % (lab, tname, y.sum(), (1 - y).sum()))
            for r in row: print('      ' + r)

print('\n=== T3 the decision (raw, unweighted OAI): AUC delta pw - sym at a .2 / .3 (bootstrap P(pw > sym))')
for k in MOD:
    for yn, YY in Y.items():
        out = []
        for a in (.2, .3):
            T = {}
            for mode in ('sym', 'pw'):
                T[mode] = {t: U[k][t].values - a / 2 * sum((1.0 if mode == 'sym' else W_PW.get((t, o), W_PW.get((o, t)))) * U[k][o].values for o in OA if o != t) for t in OA}
            d = {t: auc(T['pw'][t], YY[t].values) - auc(T['sym'][t], YY[t].values) for t in OA}
            n = len(common); dm = []
            for _ in range(min(B, 400)):
                i = rng.integers(0, n, n)
                dm.append(np.mean([auc(T['pw'][t][i], YY[t].values[i]) - auc(T['sym'][t][i], YY[t].values[i]) for t in OA]))
            dm = np.array(dm)
            out.append('a %.1f: LOA %+.5f MOA %+.5f PF %+.5f | OA-mean %+.5f (sd %.5f, P %.2f) | 12-label macro %+.6f' % (
                a, d['Lateral OA'], d['Medial OA'], d['PF OA'], np.mean(list(d.values())), dm.std(), (dm > 0).mean(), np.sum(list(d.values())) / 12))
        print('  %-7s %-6s %s' % (k, yn, ' || '.join(out)))
