# m1_regime.py (10-19 trick lens, research only, CPU): expert (gold58) vs report-derived label regimes on the SAME 58 studies.
# Per label: expert prevalence; each report reader's prevalence (>= .5), sens / spec vs expert, AUC of the reader's soft score vs
# expert; P(expert+ | reader "not addressed"); and the c4 members' + c4 output gold58 AUC for context.
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G, R
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)

def rd(path):
    d = pd.read_csv(path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    d = d.reindex(ids)[L].astype(float)
    if d.isna().any().any(): print('NaN cells in', path.split('/')[-1], d.isna().sum().to_dict())
    return d.values

LB = R + '/_labels/'
readers = {
    'v2api': None,  # report_labels_v2 (our labeler): value + verdict
    'v4blend': LB + 'rsna-knee-llm-report-labels/llm_labels_v4_blend.csv',
    'full': LB + 'rsna-knee-llm-report-labels/llm_labels_full.csv',
    'sol56': LB + 'rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv',
    'jevhyb': R + '/session_2026-10-05/jump2/labels_in/jev_hybrid_scores.csv',
    'blendjev_sol': R + '/session_2026-10-13/raptor/vast/remote/labels_raptor_rsna_blendjev_sol.csv',
}
v2 = pd.read_csv(LB + 'rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(ids)
print('v2 missing gold rows:', int(v2[L[0]].isna().sum()))
v2[L] = v2[L].fillna(.28); v2[[c + '__verdict' for c in L]] = v2[[c + '__verdict' for c in L]].fillna('UNK')
S = {'v2api': v2[L].astype(float).values}
V2verd = v2[[c + '__verdict' for c in L]].values
for k, p in readers.items():
    if p: S[k] = rd(p)

# not-addressed masks per reader (where definable)
NA = {'v2api': V2verd == 'UNK', 'v4blend': np.isclose(S['v4blend'], .25), 'full': np.isclose(S['full'], .5)}

C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
R6 = rd(G + '/R6_gold58_serve.csv'); S6 = rd(G + '/n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4P = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j])
    b = w * rr + (1 - w) * pct(R6[:, j]); C4P[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
M = {'R6': R6, 'r384': R384, 'r224': R224, 'S6fall': S6, 'c4': C4P}
print('c4 gold58 macro %.5f (expect .94102)' % np.mean([auc(C4P[:, j], Y[:, j]) for j in range(12)]))

rows = []
for j, c in enumerate(L):
    r = {'label': c, 'exp_pos': int(Y[:, j].sum()), 'exp_prev': round(Y[:, j].mean(), 3)}
    for k, s in S.items():
        b = (s[:, j] >= .5).astype(int)
        tp = ((b == 1) & (Y[:, j] == 1)).sum(); fn = ((b == 0) & (Y[:, j] == 1)).sum()
        tn = ((b == 0) & (Y[:, j] == 0)).sum(); fp = ((b == 1) & (Y[:, j] == 0)).sum()
        r[k + '_prev'] = round(b.mean(), 3); r[k + '_sens'] = round(tp / max(tp + fn, 1), 2); r[k + '_spec'] = round(tn / max(tn + fp, 1), 2)
        ok = ~np.isnan(s[:, j]); r[k + '_auc'] = round(auc(s[ok, j], Y[ok, j]), 3)
    for k, m in NA.items():
        mm = m[:, j]; r[k + '_NAfrac'] = round(mm.mean(), 2)
        r[k + '_P(exp+|NA)'] = round(Y[mm, j].mean(), 2) if mm.sum() else None
    for k, m in M.items():
        r['auc_' + k] = round(auc(m[:, j], Y[:, j]), 4)
    rows.append(r)
T = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 80)
print(T[['label', 'exp_pos', 'exp_prev', 'v2api_prev', 'v4blend_prev', 'sol56_prev', 'jevhyb_prev']].to_string(index=False))
print()
print(T[['label', 'v2api_sens', 'v2api_spec', 'v4blend_sens', 'v4blend_spec', 'sol56_sens', 'sol56_spec', 'jevhyb_sens', 'jevhyb_spec']].to_string(index=False))
print()
print(T[['label', 'v2api_auc', 'v4blend_auc', 'full_auc', 'sol56_auc', 'jevhyb_auc', 'blendjev_sol_auc', 'auc_R6', 'auc_r384', 'auc_r224', 'auc_S6fall', 'auc_c4']].to_string(index=False))
print()
print(T[['label', 'v2api_NAfrac', 'v2api_P(exp+|NA)', 'v4blend_NAfrac', 'v4blend_P(exp+|NA)', 'full_NAfrac', 'full_P(exp+|NA)']].to_string(index=False))
mac = {k: round(float(T[k].mean()), 4) for k in T.columns if k.endswith('_auc') or k.startswith('auc_')}
print('MACRO', json.dumps(mac))
# correlation: label-level regime gap vs c4 gold58 AUC shortfall
gap = 1 - T[['v4blend_auc', 'sol56_auc', 'v2api_auc']].mean(1)
print('Spearman(per-label report-reader AUC gap, c4 AUC gap) = %.3f' % pd.Series(gap.values).corr(pd.Series(1 - T['auc_c4'].values), method='spearman'))
T.to_csv('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/trick/regime_work/m1_regime.csv', index=False)
