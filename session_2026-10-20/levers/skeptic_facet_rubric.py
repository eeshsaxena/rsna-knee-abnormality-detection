# skeptic_facet_rubric.py (10-20 levers, skeptic of "facet confusion supports E-OA pw", rubric angle; Claude)
# Research only, CPU, competition data only (train reports + OOF + LLM labels), no OAI, nothing written except stdout.
# Checks: (1) reproduce local_facet.log; (2) PF-severity confound (PF label / PF output by group); (3) LOA facet AUC stratified
# by the model's own PF output, and within PF-label strata; (4) direct side contrast (lateral-facet-only vs medial-facet-only,
# both OA-side labels negative); (5) paired bootstrap of the lateral minus medial facet AUC; (6) does pw remove more of the
# lateral facet excess than sym at a .2 / .3 (the actual decision), and how big is pw - sym on the LOA output at all;
# (7) negation sensitivity of the facet regex; (8) gold74 expert labels for facet-only reports (count only).
import re, sys, numpy as np, pandas as pd
LEV = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/levers'
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, G, R, s6seed
import io, contextlib
ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(LEV + '/census_lang.py', encoding='utf-8').read().split('CATS = []')[0], ns)
SS, pos_hit = ns['SS'], ns['pos_hit']
tr = pd.read_csv(R + '/_data/train.csv', dtype={'StudyInstanceUID': str}); gold = set(tr.loc[tr['ACL'].notna(), 'StudyInstanceUID'])
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
M = {'R7': r7[['p|' + c for c in L]].set_axis(L, axis=1), 's1044': s6seed(1044)[0], 's1045': s6seed(1045)[0]}
v4 = pd.read_csv(R + '/_labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
jev = pd.read_csv(R + '/session_2026-10-05/jump2/labels_in/jev_hybrid_scores.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
CH = r"(chondr|cartilag)"
OA = ['Medial OA', 'Lateral OA', 'PF OA']
W_PW = {('Medial OA', 'Lateral OA'): .876, ('Medial OA', 'PF OA'): .708, ('Lateral OA', 'PF OA'): 1.416}


def w(t, o, mode):
    return 1.0 if mode == 'sym' else W_PW.get((t, o), W_PW.get((o, t)))


def classify(side, neg_aware=False):
    FAC = side + r" (patellar )?facet|" + side + r" patell|" + side + r" trochle"
    TF = side + r" (femoral condyle|tibial plateau|compartment|femorotibial|tibiofemoral|femoro-tibial|condyle|plateau)"
    cls = {}
    for u, ss in SS.items():
        if u in gold: continue
        f = any(re.search(FAC, s) and (pos_hit(s, CH) if neg_aware else re.search(CH, s)) for s in ss)
        t = any(re.search(TF, s) and pos_hit(s, CH) for s in ss)
        p = any(re.search(r"(patell|trochle|femoropatellar|patellofemoral|retropatellar)", s) and pos_hit(s, CH) for s in ss)
        anyside = any(re.search(side, s) and re.search(CH, s) for s in ss)
        if f and not t: cls[u] = 'facet_only'
        elif p and not anyside: cls[u] = 'PF_no_side'
    return pd.Series(cls)


def labneg(ids, lab):
    return ids[(v4.reindex(ids)[lab] < .5).values & (jev.reindex(ids)[lab] < .5).values]


rng = np.random.default_rng(20261020)
U = {m: pd.DataFrame({c: pct(D.dropna()[c].values) for c in L}, index=D.dropna().index) for m, D in M.items()}
CLS = {s: classify(s) for s in ('lateral', 'medial')}
CLSN = {s: classify(s, True) for s in ('lateral', 'medial')}
SIDE = {'lateral': 'Lateral OA', 'medial': 'Medial OA'}

print('== (1) reproduce + (2) PF-severity confound + (7) negation sensitivity')
for side, lab in SIDE.items():
    cls = CLS[side]; neg = labneg(cls.index, lab)
    for m in M:
        ii = neg.intersection(U[m].index); y = (cls.loc[ii] == 'facet_only').astype(int).values
        s = U[m].loc[ii, lab].values; pf = U[m].loc[ii, 'PF OA'].values
        pfl = (v4.reindex(ii)['PF OA'] >= .5).values
        print(f'{lab:10s} {m:6s} n {y.sum()} vs {(1-y).sum()} | AUC {lab} {auc(s, y):.3f} | AUC PF-output {auc(pf, y):.3f} '
              f'| mean pct PF out {pf[y==1].mean():.3f} vs {pf[y==0].mean():.3f} | PF label rate {pfl[y==1].mean():.3f} vs {pfl[y==0].mean():.3f}')
    clsn = CLSN[side]; negn = labneg(clsn.index, lab)
    ii = negn.intersection(U['R7'].index); y = (clsn.loc[ii] == 'facet_only').astype(int).values
    print(f'   negation-aware facet regex (R7): n {y.sum()} vs {(1-y).sum()} | AUC {auc(U["R7"].loc[ii, lab].values, y):.3f}')

print('\n== (3) LOA / MOA facet AUC controlling the model\'s own PF output (5 strata of PF pct, stratum-weighted AUC) and PF label')
for side, lab in SIDE.items():
    cls = CLS[side]; neg = labneg(cls.index, lab)
    for m in M:
        ii = neg.intersection(U[m].index); y = (cls.loc[ii] == 'facet_only').astype(int).values
        s = U[m].loc[ii, lab].values; pf = U[m].loc[ii, 'PF OA'].values
        q = np.quantile(pf, [.2, .4, .6, .8]); k = np.searchsorted(q, pf)
        num = den = 0.
        for b in range(5):
            sel = k == b
            if 0 < y[sel].sum() < sel.sum():
                pq = y[sel].sum() * (sel.sum() - y[sel].sum()); num += auc(s[sel], y[sel]) * pq; den += pq
        # residual on PF output: slope fitted on the whole label-negative OOF pool (not the contrast set)
        pool = labneg(U[m].index.intersection(pd.Index(list(SS))), lab).difference(pd.Index(list(gold)))
        bb = np.polyfit(U[m].loc[pool, 'PF OA'].values, U[m].loc[pool, lab].values, 1)[0]
        res = s - bb * pf
        pfl = (v4.reindex(ii)['PF OA'] >= .5).values
        a_pos = auc(s[pfl], y[pfl]) if 0 < y[pfl].sum() < pfl.sum() else np.nan
        a_neg = auc(s[~pfl], y[~pfl]) if 0 < y[~pfl].sum() < (~pfl).sum() else np.nan
        print(f'{lab:10s} {m:6s} raw {auc(s, y):.3f} | PF-stratified {num/den:.3f} | residual on PF (slope {bb:.2f}) {auc(res, y):.3f} '
              f'| within PF-label+ {a_pos:.3f} (n {y[pfl].sum()}/{(1-y[pfl]).sum()}) | within PF-label- {a_neg:.3f} (n {y[~pfl].sum()}/{(1-y[~pfl]).sum()})')

print('\n== (4) direct side contrast: lateral-facet-only vs medial-facet-only (both OA-side labels negative, no TF chondral either side)')
LF = r"lateral (patellar )?facet|lateral patell|lateral trochle"; MF = r"medial (patellar )?facet|medial patell|medial trochle"
LT = r"lateral (femoral condyle|tibial plateau|compartment|femorotibial|tibiofemoral|femoro-tibial|condyle|plateau)"
MT = LT.replace('lateral', 'medial')
grp = {}
for u, ss in SS.items():
    if u in gold: continue
    lf = any(re.search(LF, s) and re.search(CH, s) for s in ss); mf = any(re.search(MF, s) and re.search(CH, s) for s in ss)
    lt = any(re.search(LT, s) and pos_hit(s, CH) for s in ss); mt = any(re.search(MT, s) and pos_hit(s, CH) for s in ss)
    if lt or mt: continue
    if lf and not mf: grp[u] = 1
    elif mf and not lf: grp[u] = 0
grp = pd.Series(grp)
gi = labneg(labneg(grp.index, 'Lateral OA'), 'Medial OA')
for m in M:
    ii = gi.intersection(U[m].index); y = grp.loc[ii].values
    lo, mo, pf = (U[m].loc[ii, c].values for c in ('Lateral OA', 'Medial OA', 'PF OA'))
    print(f'{m:6s} n lat {y.sum()} med {(1-y).sum()} | AUC(LOA out: lat>med) {auc(lo, y):.3f} | AUC(MOA out: med>lat) {auc(mo, 1-y):.3f} '
          f'| AUC(LOA-MOA: lat>med) {auc(lo-mo, y):.3f} | PF out lat {pf[y==1].mean():.3f} med {pf[y==0].mean():.3f}')

print('\n== (5) paired bootstrap: lateral facet AUC minus medial facet AUC (studies resampled jointly), and stratified version')
for m in M:
    out = {}
    for side, lab in SIDE.items():
        cls = CLS[side]; neg = labneg(cls.index, lab); ii = neg.intersection(U[m].index)
        out[side] = (ii, (cls.loc[ii] == 'facet_only').astype(int).values, U[m].loc[ii, lab].values)
    pool = out['lateral'][0].union(out['medial'][0]); pos = {u: i for i, u in enumerate(pool)}
    d0 = auc(out['lateral'][2], out['lateral'][1]) - auc(out['medial'][2], out['medial'][1])
    ds = []
    idxL = np.array([pos[u] for u in out['lateral'][0]]); idxM = np.array([pos[u] for u in out['medial'][0]])
    for _ in range(1000):
        cnt = np.bincount(rng.integers(0, len(pool), len(pool)), minlength=len(pool))
        a = []
        for side, idx in (('lateral', idxL), ('medial', idxM)):
            _, y, s = out[side]; rep = cnt[idx]
            yy = np.repeat(y, rep); ss_ = np.repeat(s, rep)
            a.append(auc(ss_, yy) if 0 < yy.sum() < len(yy) else np.nan)
        ds.append(a[0] - a[1])
    ds = np.array(ds); ds = ds[np.isfinite(ds)]
    print(f'{m:6s} lateral - medial facet AUC {d0:+.3f} | boot sd {ds.std():.3f} | P(>0) {np.mean(ds > 0):.2f} | 90% CI {np.quantile(ds, .05):+.3f} .. {np.quantile(ds, .95):+.3f}')

print('\n== (6) the decision: facet AUC on the transformed LOA / MOA output, c4-like a .2 / .3, sym vs pw (lower = more facet excess removed)')
for side, lab in SIDE.items():
    cls = CLS[side]; neg = labneg(cls.index, lab)
    for m in M:
        ii = neg.intersection(U[m].index); y = (cls.loc[ii] == 'facet_only').astype(int).values
        row = [f'{lab:10s} {m:6s} raw {auc(U[m].loc[ii, lab].values, y):.3f}']
        for a in (.2, .3):
            for mode in ('sym', 'pw'):
                tt = U[m].loc[ii, lab].values - a / 2 * sum(w(lab, o, mode) * U[m].loc[ii, o].values for o in OA if o != lab)
                row.append(f'{mode}{int(a*100)} {auc(tt, y):.3f}')
        print(' | '.join(row))

print('\n== (6b) how different are pw and sym on the LOA ranking at all (all OOF studies; Spearman of the transformed LOA, and share of')
print('   studies whose LOA rank moves by > 5 pct points); and the share of facet-only studies in EN train (hidden-size scaling)')
for m in M:
    D = U[m]
    for a in (.2, .3):
        t_s = D['Lateral OA'] - a / 2 * (D['Medial OA'] + D['PF OA'])
        t_p = D['Lateral OA'] - a / 2 * (.876 * D['Medial OA'] + 1.416 * D['PF OA'])
        ps, pp = pct(t_s.values), pct(t_p.values)
        print(f'{m:6s} a {a:.1f}: Spearman(pw, sym) LOA {np.corrcoef(ps, pp)[0,1]:.4f} | share |rank move| > .05: {np.mean(np.abs(ps-pp) > .05):.3f}')
n_lat = (CLS['lateral'] == 'facet_only').sum(); n_en = len([u for u in SS if u not in gold])
print(f'lateral facet-only share of EN non-gold reports {n_lat}/{n_en} = {n_lat/n_en:.3f}; at private n 910 (CLAIMED implied.md) about {n_lat/n_en*910:.0f} knees')

print('\n== (8) gold74 expert labels on facet-only reports (count only; gold is NOT used to select)')
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
for side, lab in SIDE.items():
    FAC = side + r" (patellar )?facet|" + side + r" patell|" + side + r" trochle"
    TF = side + r" (femoral condyle|tibial plateau|compartment|femorotibial|tibiofemoral|femoro-tibial|condyle|plateau)"
    hits = [u for u in g.index if u in SS and any(re.search(FAC, s) and re.search(CH, s) for s in SS[u]) and not any(re.search(TF, s) and pos_hit(s, CH) for s in SS[u])]
    print(f'{lab}: gold74 EN studies {sum(u in SS for u in g.index)}, facet-only {len(hits)}, expert {lab} positives among them {int(g.loc[hits, lab].sum()) if hits else 0}, expert PF OA positives {int(g.loc[hits, "PF OA"].sum()) if hits else 0}')
