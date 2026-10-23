# m2_states.py (10-19 trick lens, research only, CPU): does a RUBRIC-mapped score from the heliosli-family STATE heads rank the
# EXPERT labels better than the binary head?  Mechanism (MEASURED m1_regime.log): experts use stricter severity thresholds than
# reports for Effusion (moderate+), MCL (high-grade tear), ACL (>50 % fibres), OA (high-grade), Contusion (impact edema), Baker's
# (moderate+). Report-trained binary heads saturate for "small" and "large" alike; the state heads separate them.
# The weights below were written from the rubric text (forum 733343) BEFORE any gold58 number for them was computed.
# rule: state literally meets the rubric -> 1; key criterion unknown / compatible with both -> .5; literally fails -> 0.
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G, R
W = {
 'ACL': {'normal': 0, 'signal_or_degeneration': 0, 'low_grade_sprain_or_partial': 0, 'partial_tear_unspecified': .5, 'tear_unspecified': 1, 'complete_tear': 1},
 'MCL': {'normal': 0, 'periligamentous_edema_or_grade1': 0, 'partial_tear_or_grade2': .5, 'chronic_or_healed_change': 0, 'high_grade_partial_tear': 1, 'complete_tear': 1},
 'Medial Meniscus': {'normal': 0, 'degeneration_no_tear': 0, 'abnormal_tear_status_unstated': .5, 'tear': 1, 'macerated_or_destroyed': 1},
 'Lateral Meniscus': {'normal': 0, 'degeneration_no_tear': 0, 'abnormal_tear_status_unstated': .5, 'tear': 1, 'macerated_or_destroyed': 1},
 'Medial OA': {'normal': 0, 'low_grade_chondral': 0, 'chondral_unspecified': .5, 'high_grade_size_unstated': 1, 'high_grade_large': 1, 'no_cartilage_detail': .5},
 'Lateral OA': {'normal': 0, 'low_grade_chondral': 0, 'chondral_unspecified': .5, 'high_grade_size_unstated': 1, 'high_grade_large': 1, 'no_cartilage_detail': .5},
 'PF OA': {'normal': 0, 'low_grade_chondral': 0, 'chondral_unspecified': .5, 'high_grade_size_unstated': 1, 'high_grade_large': 1, 'no_cartilage_detail': .5},
 'Effusion': {'none_or_trace': 0, 'small': 0, 'small_to_moderate': .5, 'present_unspecified': .5, 'moderate': 1, 'large': 1},
 'Synovitis': {'none': 0, 'other_synovial_finding': .5, 'fat_pad_inflammation': .5, 'synovitis': 1, 'synovial_thickening': 1},
 "Baker's": {'none': 0, 'small': 0, 'present_unspecified': .5, 'moderate': 1, 'large': 1, 'ruptured_or_leaking': 1},
 'Contusion': {'none': 0, 'degenerative_or_reactive_edema': 0, 'nonspecific_marrow_edema': .5, 'fracture_associated_edema_only': 0, 'traumatic_marrow_edema': 1, 'contusion_or_bruise': 1},
 'Fracture': {'none': 0, 'trabecular_microfracture_only': 0, 'insufficiency_or_stress_fracture': 1, 'avulsion_fracture': 1, 'impaction_or_osteochondral_fracture': 1, 'fracture_age_unstated': .5, 'acute_fracture': 1, 'old_or_healed_fracture': 0},
}
STRICT = ['ACL', 'MCL', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', "Baker's", 'Contusion']  # m1: expert prevalence < report prevalence
d = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
for c in L:
    cols = [k for k in d.columns if k.startswith('state|%s|' % c)]
    assert set(k.split('|')[2] for k in cols) == set(W[c]), (c, cols)
def scores(df):
    B = df[['p|' + c for c in L]].values
    Sr = np.stack([sum(df['state|%s|%s' % (c, s)].values * w for s, w in W[c].items()) for c in L], 1)
    Sn = np.stack([1 - df['state|%s|%s' % (c, list(W[c])[0])].values for c in L], 1)  # 1 - P(first = normal / none)
    return B, Sr, Sn
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
B, Sr, Sn = scores(d.loc[ids])
mix = lambda a, b: .5 * pct(a) + .5 * pct(b)
rows = []
for j, c in enumerate(L):
    rows.append({'label': c, 'pos': int(Y[:, j].sum()), 'bin': auc(B[:, j], Y[:, j]), 'rubric': auc(Sr[:, j], Y[:, j]),
                 'mix.5': auc(mix(B[:, j], Sr[:, j]), Y[:, j]), '1-P(normal)': auc(Sn[:, j], Y[:, j])})
T = pd.DataFrame(rows).set_index('label')
T['d_rub'] = T.rubric - T.bin; T['d_mix'] = T['mix.5'] - T.bin; T['d_1mn'] = T['1-P(normal)'] - T.bin
pd.set_option('display.width', 200)
print('R7 OOF on gold58 (expert labels): binary head vs rubric-mapped state score')
print(T.round(4).to_string())
print('macro: bin %.4f rubric %.4f mix %.4f 1-P(normal) %.4f' % tuple(T[k].mean() for k in ['bin', 'rubric', 'mix.5', '1-P(normal)']))
print('STRICT-label mean delta rubric %+.4f mix %+.4f | other labels rubric %+.4f mix %+.4f' % (
    T.loc[STRICT, 'd_rub'].mean(), T.loc[STRICT, 'd_mix'].mean(), T.drop(STRICT).d_rub.mean(), T.drop(STRICT).d_mix.mean()))
# paired bootstrap on gold58
rng = np.random.default_rng(1019); out = {'rub': [], 'mix': []}
for _ in range(3000):
    ix = rng.integers(0, n, n)
    if any(Y[ix, j].min() == Y[ix, j].max() for j in range(12)): continue
    b = np.mean([auc(B[ix, j], Y[ix, j]) for j in range(12)])
    out['rub'].append(np.mean([auc(Sr[ix, j], Y[ix, j]) for j in range(12)]) - b)
    out['mix'].append(np.mean([auc(mix(B[ix, j], Sr[ix, j]), Y[ix, j]) for j in range(12)]) - b)
for k, v in out.items():
    v = np.array(v); print('gold58 boot macro d_%s mean %+.4f sd %.4f P>0 %.3f' % (k, v.mean(), v.std(), (v > 0).mean()))

# CONTROL: report-regime OOF (non-gold train studies), labels = reader >= .5 with "not addressed" dropped where definable
nongold = d.index[~d.index.isin(set(g.StudyInstanceUID))]
Bo, Sro, Sno = scores(d.loc[nongold])
LBp = R + '/_labels/'
for name, path, na in [('v4blend', LBp + 'rsna-knee-llm-report-labels/llm_labels_v4_blend.csv', .25),
                       ('sol56', LBp + 'rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv', None),
                       ('jevhyb', R + '/session_2026-10-05/jump2/labels_in/jev_hybrid_scores.csv', None)]:
    lab = pd.read_csv(path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(nongold)[L].astype(float).values
    r = []
    for j, c in enumerate(L):
        ok = ~np.isnan(lab[:, j])
        if na is not None: ok &= ~np.isclose(lab[:, j], na)
        y = (lab[ok, j] >= .5).astype(int)
        r.append((c, auc(Bo[ok, j], y), auc(Sro[ok, j], y), auc(mix(Bo[ok, j], Sro[ok, j]), y)))
    R_ = pd.DataFrame(r, columns=['label', 'bin', 'rubric', 'mix']).set_index('label')
    print('OOF report-regime control [%s] n~%d: d_rub %s | macro bin %.4f rub %+.4f mix %+.4f' % (
        name, len(nongold), np.round((R_.rubric - R_.bin).values, 4).tolist(), R_.bin.mean(), (R_.rubric - R_.bin).mean(), (R_['mix'] - R_.bin).mean()))
T.to_csv('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/trick/regime_work/m2_states_g58.csv')
