# syn_local.py (10-20 night, Claude). Synovitis deep dive, LOCAL CPU ONLY, RSNA competition data only (no OAI).
# Read-only inputs: gold58 / gold74 expert labels, member serves on gold58 (session_2026-10-18/next/g58), Dread / fleet member
# CSVs (session_2026-10-19/think/models_work), R7 OOF (incl. state heads), train.csv report text, LLM report labels.
# Prints aggregates only (counts, prevalences, AUCs, deltas, bootstrap P). Writes nothing except its own stdout log.
import sys, re, json
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, norm
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
import cbase

R = 'C:/Users/eeshs/Downloads/rsna-knee'
G = R + '/session_2026-10-18/next/g58'
MW = R + '/session_2026-10-19/think/models_work'
L = cbase.L
J = {c: i for i, c in enumerate(L)}
SYN, EFF, BAK = J['Synovitis'], J['Effusion'], J["Baker's"]
pct = lambda v: rankdata(v) / len(v)
rng = np.random.default_rng(20261020)
BOOT = 2000


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    r = rankdata(s); p = y.sum(); q = len(y) - p
    if p == 0 or q == 0:
        return np.nan
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def tetra(a, b):
    a = np.asarray(a, int); b = np.asarray(b, int)
    from scipy.stats import multivariate_normal
    pa, pb = a.mean(), b.mean(); p11 = (a & b).mean()
    ta, tb = norm.ppf(1 - pa), norm.ppf(1 - pb)
    best, br = 9, 0
    for r in np.linspace(-.98, .98, 197):
        m = multivariate_normal([0, 0], [[1, r], [r, 1]])
        p = 1 - norm.cdf(ta) - norm.cdf(tb) + m.cdf([ta, tb])
        if abs(p - p11) < best:
            best, br = abs(p - p11), r
    return br


d = cbase.load()
g58 = d['g58']; ids = d['ids']; Y = d['Y58']; n = len(ids)
g74 = d['g74']; ids74 = d['ids74']
print('=' * 30, 'PART 1. Expert label structure (gold58 n %d; gold74 n %d, extra 16 = our reads, not expert)' % (n, len(g74)))
ys, ye, yb = Y[:, SYN], Y[:, EFF], Y[:, BAK]
print('gold58 prevalence: Syn %.3f (%d) Eff %.3f (%d) Bak %.3f (%d)' % (ys.mean(), ys.sum(), ye.mean(), ye.sum(), yb.mean(), yb.sum()))
ct = pd.crosstab(pd.Series(ye, name='Eff'), pd.Series(ys, name='Syn'))
print('gold58 Eff x Syn counts (rows Eff 0/1, cols Syn 0/1):', ct.values.tolist())
print('P(Syn|Eff=1) %.3f  P(Syn|Eff=0) %.3f  P(Eff|Syn=1) %.3f  tet %.2f' % (ys[ye == 1].mean(), ys[ye == 0].mean(), ye[ys == 1].mean(), tetra(ys, ye)))
anyoa = (Y[:, [J['Medial OA'], J['Lateral OA'], J['PF OA']]].sum(1) > 0).astype(int)
acute = (Y[:, [J['ACL'], J['Contusion'], J['Fracture']]].sum(1) > 0).astype(int)
for nm, h in (('Bak', yb), ('anyOA', anyoa), ('MM', Y[:, J['Medial Meniscus']]), ('acute(ACL|Cont|Frac)', acute)):
    print('  Syn x %-22s P(Syn|h=1) %.3f (n %d)  P(Syn|h=0) %.3f (n %d)  tet %.2f' % (nm, ys[h == 1].mean(), h.sum(), ys[h == 0].mean(), (h == 0).sum(), tetra(ys, h)))
nab = Y.sum(1) - ys
print('  Syn by number of OTHER expert positives: ' + ' | '.join('%s: %d/%d' % (k, ys[(nab >= lo) & (nab <= hi)].sum(), ((nab >= lo) & (nab <= hi)).sum())
                                                                 for k, lo, hi in (('0-1', 0, 1), ('2-3', 2, 3), ('4-5', 4, 5), ('6+', 6, 99))))
# Syn-positive without (moderate+) effusion: what else do they have (counts only)
m = (ys == 1) & (ye == 0)
print('  Syn+ Eff- (n %d): other-label positives summed over these studies: %s' % (m.sum(), {L[j]: int(Y[m, j].sum()) for j in range(12) if j not in (SYN, EFF) and Y[m, j].sum() > 0}))
m = (ys == 0) & (ye == 1)
print('  Syn- Eff+ (n %d): other-label positives: %s' % (m.sum(), {L[j]: int(Y[m, j].sum()) for j in range(12) if j not in (SYN, EFF) and Y[m, j].sum() > 0}))
Y74 = g74[L].values
ok = np.isfinite(Y74[:, SYN]) & np.isfinite(Y74[:, EFF])
a74, b74 = Y74[ok, SYN].astype(int), Y74[ok, EFF].astype(int)
print('gold74 (n %d with both): Syn %.3f Eff %.3f P(Syn|Eff=1) %.3f P(Syn|Eff=0) %.3f tet %.2f' % (ok.sum(), a74.mean(), b74.mean(), a74[b74 == 1].mean(), a74[b74 == 0].mean(), tetra(a74, b74)))
ex = (g74.source != 'original_gold58').values & ok
print('  extra16 only (our reads): n %d Syn %d Eff %d' % (ex.sum(), int(Y74[ex, SYN].sum()), int(Y74[ex, EFF].sum())))

print('=' * 30, 'PART 2. Reports: how the text and the LLM labels assign Synovitis (train n 4,407)')
tr = pd.read_csv(R + '/_data/train.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
lang = pd.read_csv(R + '/_data/_reports_lang.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['lang']
txt = tr.Report.fillna('').str.lower()
PAT = {
    'syn_term': r'(?<!teno)(?<!teno-)(synovit|sinovit|synovialit|синовит|υμενίτ|υμενιτ|reizsynov)',
    'synovial_thick': r'(synovial (thicken|hypertroph|prolifer)|thicken\w* (of the )?synovi|engrosamiento sinovial|sinovial engros|synovialverdick|verdickt\w* synov|sinovyal (kal|kalınla)|sinoviyal kal|épaississement synovial|synoviale verdikking|zadebljan\w* sinovij|πάχυνση\w* (του )?(αρθρικού )?υμέν|синовиална хипертроф|синовиално задебел|hipertrofia sinovial|synovialisproliferation|synovialenprolif)',
    'hoffa': r'(hoff|fat pad|fat-pad|grasa de hoffa|cuerpo adiposo|corps adipeux|fettkörper|yağ yastı)',
    'effusion': r'(effusion|derrame|erguss|efüzyon|efuzyon|izljev|συλλογ|излив|vocht|hydrops|épanchement|epanchement|líquido articular|liquido articular|sıvı|tekućin|течност)',
}
F = pd.DataFrame({k: txt.str.contains(p, regex=True) for k, p in PAT.items()})
F['lang'] = lang.reindex(F.index).fillna('na')
v2 = pd.read_csv(R + '/_labels/rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
v4 = pd.read_csv(R + '/_labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
F['v2_syn'] = v2['Synovitis__verdict'].reindex(F.index)
F['v2_eff'] = v2['Effusion__verdict'].reindex(F.index)
F['v4_syn'] = v4['Synovitis'].reindex(F.index)
F['v4_eff'] = v4['Effusion'].reindex(F.index)
gold_mask = F.index.isin(ids)
rows = []
for lg, sub in [('ALL', F)] + list(F.groupby('lang')):
    rows.append(dict(lang=lg, n=len(sub), syn_term=sub.syn_term.mean(), syn_thick=sub.synovial_thick.mean(), hoffa=sub.hoffa.mean(),
                     effusion=sub.effusion.mean(), v2_YES=(sub.v2_syn == 'YES').mean(), v2_NO=(sub.v2_syn == 'NO').mean(),
                     v2_UNK=(~sub.v2_syn.isin(['YES', 'NO'])).mean(), v4_mean=sub.v4_syn.mean(), v4_ge5=(sub.v4_syn >= .5).mean(),
                     v2eff_YES=(sub.v2_eff == 'YES').mean(),
                     P_v2synYES_given_v2effYES=((sub.v2_syn == 'YES') & (sub.v2_eff == 'YES')).sum() / max(1, (sub.v2_eff == 'YES').sum())))
print(pd.DataFrame(rows).set_index('lang').round(3).to_string())
# among reports that mention a synovitis term: share with v2 YES
st = F[F.syn_term]
print('reports with a synovitis term: %d; v2 YES %.3f NO %.3f UNK %.3f (negated mentions such as "no synovitis" count as a mention)' % (
    len(st), (st.v2_syn == 'YES').mean(), (st.v2_syn == 'NO').mean(), (~st.v2_syn.isin(['YES', 'NO'])).mean()))
nt = F[~F.syn_term]
print('reports WITHOUT a synovitis term: %d; v2 YES %.3f (YES from thickening / Hoffa wording) | of these, with thickening wording %d, Hoffa wording %d' % (
    len(nt), (nt.v2_syn == 'YES').mean(), int(nt.synovial_thick.sum()), int(nt.hoffa.sum())))
# gold58 subset: wording vs expert label
Fg = F.loc[ids]
for nm, msk in (('expert Syn+', ys == 1), ('expert Syn-', ys == 0)):
    s = Fg[msk]
    print('gold58 %s (n %d): syn_term %.2f syn_thick %.2f hoffa %.2f effusion %.2f | v2 YES %.2f NO %.2f UNK %.2f | v4 mean %.3f | langs %s' % (
        nm, len(s), s.syn_term.mean(), s.synovial_thick.mean(), s.hoffa.mean(), s.effusion.mean(), (s.v2_syn == 'YES').mean(),
        (s.v2_syn == 'NO').mean(), (~s.v2_syn.isin(['YES', 'NO'])).mean(), s.v4_syn.mean(), s.lang.value_counts().to_dict()))
print('gold58 AUC vs expert Syn: v4 Syn %.3f | v4 Eff %.3f | v4 Syn + .5 v4 Eff (rank) %.3f | syn_term %.3f | hoffa %.3f' % (
    auc(Fg.v4_syn, ys), auc(Fg.v4_eff, ys), auc(pct(Fg.v4_syn.values) + .5 * pct(Fg.v4_eff.values), ys), auc(Fg.syn_term.astype(float), ys),
    auc(Fg.hoffa.astype(float), ys)))
print('gold58 language mix: %s | gold58 report Syn (v2 YES) %.3f vs train %.3f' % (Fg.lang.value_counts().to_dict(), (Fg.v2_syn == 'YES').mean(), (F.v2_syn == 'YES').mean()))

print('=' * 30, 'PART 3. Member heads vs expert Synovitis (gold58)')
rd = lambda f: pd.read_csv(f, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
M = {}
M['R6 (c4, OAI-free)'] = d['members']['R6']
M['S6fall (c4, OAI-free)'] = d['members']['S6']
M['Raptor r384 (c4, OAI Hoffa>=1 Syn)'] = d['members']['R384']
M['Raptor r224 (c4, OAI)'] = d['members']['R224']
M['Raptor N crop320'] = np.load(G + '/gold58_N_crop320_k24.npy')
M['Raptor N full384'] = np.load(G + '/gold58_N_full384_k24.npy')
M['s43 (ours S6 seed)'] = rd(G + '/s43_gold58.csv').loc[ids, L].values
r7 = rd(G + '/R7OOF.csv')
M['R7 OOF (report-trained)'] = r7.loc[ids, ['p|' + c for c in L]].values
for f, nm in (('T_maxspan_v5', 'Dread maxspan v5 (CoAtNet, no OAI)'), ('T_native384_v8', 'Dread native384 v8 (no OAI)'),
              ('T_finespacing_v9', 'Dread finespacing v9 (no OAI)')):
    M[nm] = rd(MW + '/%s.csv' % f).loc[ids, L].values
M['fleetOF mean (ours, OAI-free)'] = rd(MW + '/gold_fleetOF_mean.csv').loc[ids, L].values
Rmix = np.stack([.75 * pct(d['members']['R384'][:, j]) + .25 * pct(d['members']['R224'][:, j]) for j in range(12)], 1)
M['c4 Raptor mix (.75/.25)'] = Rmix
M['c4 FINAL (= pw30 Syn/Eff/Bak cols)'] = d['c4']
idx = []
while len(idx) < BOOT:
    ii = rng.integers(0, n, n)
    if 0 < ys[ii].sum() < n and 0 < ye[ii].sum() < n:
        idx.append(ii)


def bootd(base, new, y):
    b = np.array([auc(new[ii], y[ii]) - auc(base[ii], y[ii]) for ii in idx])
    return auc(new, y) - auc(base, y), b.std(), (b > 0).mean()


rows = []
for k, P in M.items():
    s, e, b = P[:, SYN], P[:, EFF], P[:, BAK]
    best = max(((auc(P[:, j], ys), L[j]) for j in range(12) if j != SYN))
    r = dict(member=k, Syn_head=auc(s, ys), Eff_head=auc(e, ys), Bak_head=auc(b, ys), best_other=f'{best[1]} {best[0]:.3f}',
             SynHead_for_goldEff=auc(s, ye), EffHead_for_goldEff=auc(e, ye), rho_SynEff=spearmanr(s, e).correlation,
             SynHead_inEffpos=auc(s[ye == 1], ys[ye == 1]), EffHead_inEffpos=auc(e[ye == 1], ys[ye == 1]),
             SynHead_inEffneg=auc(s[ye == 0], ys[ye == 0]), EffHead_inEffneg=auc(e[ye == 0], ys[ye == 0]))
    for a in (.1, .2, .3):
        dd, sd, P0 = bootd(pct(s), pct(s) + a * pct(e), ys)
        r['E2_a%.1f' % a] = dd; r['P_a%.1f' % a] = P0
    rows.append(r)
T = pd.DataFrame(rows).set_index('member')
pd.set_option('display.width', 250)
print(T[['Syn_head', 'Eff_head', 'Bak_head', 'best_other', 'SynHead_for_goldEff', 'EffHead_for_goldEff', 'rho_SynEff']].round(3).to_string())
print(T[['SynHead_inEffpos', 'EffHead_inEffpos', 'SynHead_inEffneg', 'EffHead_inEffneg']].round(3).to_string())
print('(strata: gold Eff+ n 35 = 22 Syn+ / 13 Syn-; gold Eff- n 23 = 5 Syn+ / 18 Syn-)')
print(T[['E2_a0.1', 'P_a0.1', 'E2_a0.2', 'P_a0.2', 'E2_a0.3', 'P_a0.3']].round(4).to_string())
print('E2 sign count over the %d single members (excl. composites) at a .1 / .2: %d / %d positive' % (
    len(T) - 2, int((T['E2_a0.1'].iloc[:-2] > 0).sum()), int((T['E2_a0.2'].iloc[:-2] > 0).sum())))

print('=' * 30, 'PART 4. R7 state heads (report-trained, OOF) vs expert Synovitis: which report concept matches the experts?')
st_cols = [c for c in r7.columns if c.startswith('state|Synovitis|') or c.startswith('state|Effusion|')]
for gname, gid, yv in (('gold58', ids, ys), ('gold74', [i for i, o in zip(ids74, ok) if o], a74)):
    S = r7.loc[gid]
    out = {c.replace('state|', ''): round(auc(S[c].values, yv), 3) for c in st_cols}
    out['p|Synovitis'] = round(auc(S['p|Synovitis'].values, yv), 3); out['p|Effusion'] = round(auc(S['p|Effusion'].values, yv), 3)
    sy = S[[c for c in st_cols if c.startswith('state|Synovitis|') and not c.endswith('|none')]].sum(1)
    out['Syn 1-P(none)'] = round(auc(sy.values, yv), 3)
    ef = S[['state|Effusion|small', 'state|Effusion|small_to_moderate', 'state|Effusion|present_unspecified', 'state|Effusion|moderate', 'state|Effusion|large']].sum(1)
    out['Eff 1-P(none_or_trace)'] = round(auc(ef.values, yv), 3)
    print(gname, 'n %d pos %d' % (len(yv), int(yv.sum())), json.dumps(out))
# the same states vs expert EFFUSION, for contrast
S = r7.loc[ids]
print('gold58 vs expert Effusion:', json.dumps({c.replace('state|', ''): round(auc(S[c].values, ye), 3) for c in st_cols}))
# report-OOF ruler for reference: R7 Syn <- Eff on report labels (v2 YES/NO), non-gold train
tr_ids = [i for i in r7.index if i not in set(ids74)]
lab = v2['Synovitis__verdict'].reindex(tr_ids); okk = lab.isin(['YES', 'NO']).values
yy = (lab[okk] == 'YES').astype(int).values; P7 = r7.loc[np.array(tr_ids)[okk]]
s7, e7 = pct(P7['p|Synovitis'].values), pct(P7['p|Effusion'].values)
print('report-OOF ruler (R7 OOF vs v2 Syn YES/NO, n %d, prev %.3f): base %.4f | E2 a .1 %+.4f a .2 %+.4f | Eff head alone %.4f' % (
    len(yy), yy.mean(), auc(s7, yy), auc(s7 + .1 * e7, yy) - auc(s7, yy), auc(s7 + .2 * e7, yy) - auc(s7, yy), auc(e7, yy)))
