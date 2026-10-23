# transfer.py (research_1008 dimension C, 2026-10-08). Light CPU, laptop.
# Measured TARGET -> STUDENT transfer on expert gold58: for every pair of students that differ (mainly) in their
# training labels, per-label delta of the student gold58 AUC vs the per-label delta of the TARGET gold58 AUC.
# Student gold preds were copied read-only from the box (~/jump2/gold74_preds, ~/novel/gold74_preds,
# ~/novel/k9gold/preds, ~/novel/frac75/gold74_preds). Gold scoring: masked (NaN expert cells dropped).
import os, json
import numpy as np, pandas as pd
from scipy.stats import rankdata

H = os.path.dirname(os.path.abspath(__file__)); BP = H + '/box_preds'
R = 'C:/Users/eeshs/Downloads/rsna-knee'
DS = R + '/session_2026-10-05/kaggle_train/ds/rsna-knee-fleet-train'
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
SH = ['ACL', 'MCL', 'MedMen', 'LatMen', 'MedOA', 'LatOA', 'PFOA', 'Eff', 'Syn', 'Bak', 'Cont', 'Frac']
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
gold = rd(DS + '/gold74_eval_set.csv'); g58 = gold.index[gold.source == 'original_gold58']
lab = {'v4': rd(DS + '/llm_labels_v4_blend.csv'), 'mix50': rd(DS + '/labels_mix50_logit.csv'),
       'gated75': rd(R + '/session_2026-10-04/bigjump/labels_gated75_logit.csv'),
       'frac75': rd(R + '/session_2026-10-08/k8/frac75/labels_frac75_logit.csv')}
lab = {k: v.reindex(g58)[T] for k, v in lab.items()}


def load(*paths):
    """mean of member probs (prob-mean, as the K-builders serve) over the given files"""
    return sum(rd(os.path.join(BP, p)).reindex(g58)[T] for p in paths) / len(paths)


S = {
    'TTR_v4_f0 (s1042)': load('jump2/f0_s1042_ttr__raw.csv'),
    'TTR_mix50_f0 J1 (s1043)': load('jump2/f0_s1043_ttr_mix50__raw.csv'),
    'cnxt_s42_v4': load('jump2/f0_s42_v4__raw.csv'),
    'cnxt_s42_mix50': load('jump2/f0_s42__raw.csv'),
    'cnxt_s42_gated75': load('jump2/f0_s42_gated__raw.csv'),
    'TTR_frac75_f0 (Kaggle E17)': load('frac75/f0_s1043_ttr_frac75__raw__A.csv'),
    'set A 5x mix50 s1043': load(*[f'novel/f{f}_s1043_ttr_mix50__raw__A.csv' for f in range(5)]),
    'set D 5x frac75 s1043': load(*[f'k9gold/f{f}_s1043_ttr_frac75__raw__A.csv' for f in range(5)]),
    'set B4 4x mix50 s1044': load(*[f'novel/f{f}_s1044_ttr_mix50__raw__A.csv' for f in range(4)]),
    'set A4 4x mix50 s1043': load(*[f'novel/f{f}_s1043_ttr_mix50__raw__A.csv' for f in range(4)]),
}
# pairs: (name, student_new, student_old, target_new, target_old)
PAIRS = [
    ('TTR f0: mix50(J1) - v4(s1042)  [labels + seed]', 'TTR_mix50_f0 J1 (s1043)', 'TTR_v4_f0 (s1042)', 'mix50', 'v4'),
    ('cnxt s42 f0: mix50 - v4  [labels only]', 'cnxt_s42_mix50', 'cnxt_s42_v4', 'mix50', 'v4'),
    ('cnxt s42 f0: gated75 - v4  [labels only]', 'cnxt_s42_gated75', 'cnxt_s42_v4', 'gated75', 'v4'),
    ('cnxt s42 f0: mix50 - gated75  [labels only]', 'cnxt_s42_mix50', 'cnxt_s42_gated75', 'mix50', 'gated75'),
    ('TTR f0: frac75 probe - J1  [Fracture col + E17 schedule]', 'TTR_frac75_f0 (Kaggle E17)', 'TTR_mix50_f0 J1 (s1043)', 'frac75', 'mix50'),
    ('5-fold: set D frac75 - set A mix50  [Fracture col]', 'set D 5x frac75 s1043', 'set A 5x mix50 s1043', 'frac75', 'mix50'),
    ('4-fold NULL: set B4 s1044 - set A4 s1043  [seed only]', 'set B4 4x mix50 s1044', 'set A4 4x mix50 s1043', 'mix50', 'mix50'),
]


def aucs(Y, P):
    out = []
    for c in T:
        y = Y[c].values; p = P[c].values; ok = np.isfinite(y)
        yb = y[ok] >= .5; n1 = yb.sum(); n0 = len(yb) - n1
        r = rankdata(p[ok]); out.append((r[yb].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return np.array(out)


Y = gold.loc[g58, T]
rng = np.random.default_rng(0); B = 2000; boot = rng.integers(0, len(g58), size=(B, len(g58)))
res = {}
print('per-label gold58 (n 58, masked). dS = student delta, dT = target delta, SE = paired bootstrap SE of dS')
allx, ally, allw, alltag = [], [], [], []
for name, sn, so, tn, to in PAIRS:
    dS = aucs(Y, S[sn]) - aucs(Y, S[so]); dT = aucs(Y, lab[tn]) - aucs(Y, lab[to])
    bs = np.array([aucs(Y.iloc[b], S[sn].iloc[b]) - aucs(Y.iloc[b], S[so].iloc[b]) for b in boot])
    se = bs.std(0); mse = bs.mean(1).std()
    print(f'\n== {name}')
    print('       ' + ' '.join(f'{s:>7s}' for s in SH) + '   macro')
    print('dT     ' + ' '.join(f'{x:+7.4f}' for x in dT) + f' {dT.mean():+7.4f}')
    print('dS     ' + ' '.join(f'{x:+7.4f}' for x in dS) + f' {dS.mean():+7.4f}  (SE {mse:.4f})')
    print('SE(dS) ' + ' '.join(f'{x:7.4f}' for x in se))
    big = np.abs(dT) >= .01
    if big.sum() >= 2:
        w = 1 / se[big] ** 2
        slope = float((w * dT[big] * dS[big]).sum() / (w * dT[big] ** 2).sum())
        slope_se = float(1 / np.sqrt((w * dT[big] ** 2).sum()))
        print(f'WLS slope through 0 on labels with |dT| >= .01 (n {big.sum()}): {slope:+.3f} (SE {slope_se:.3f}); '
              f'macro ratio dS/dT {dS.mean() / dT.mean() if abs(dT.mean()) > 1e-9 else float("nan"):+.3f}')
        allx += list(dT[big]); ally += list(dS[big]); allw += list(1 / se[big] ** 2); alltag += [name] * int(big.sum())
    elif big.sum() == 1:
        j = int(np.where(big)[0][0])
        print(f'single changed label {SH[j]}: ratio dS/dT {dS[j] / dT[j]:+.3f} (SE {se[j] / abs(dT[j]):.3f}); '
              f'null SD of dS on the {12 - 1} unchanged labels {np.std(np.delete(dS, j)):.4f}')
        allx.append(dT[j]); ally.append(dS[j]); allw.append(1 / se[j] ** 2); alltag.append(name)
    else:
        print(f'null pair: SD of dS over 12 labels {dS.std():.4f}; mean |dS| {np.abs(dS).mean():.4f}')
    res[name] = dict(dT=dT.tolist(), dS=dS.tolist(), se=se.tolist(), macro_dS=float(dS.mean()), macro_se=float(mse))
x, yv, w = map(np.array, (allx, ally, allw))
slope = float((w * x * yv).sum() / (w * x ** 2).sum()); sse = float(1 / np.sqrt((w * x ** 2).sum()))
print(f'\nPOOLED WLS slope (all pairs, labels with |dT| >= .01, n {len(x)}): {slope:+.3f} (SE {sse:.3f})')
# same pooled slope, single-member pairs only vs ensemble pair
for keep, nm in ((lambda t: not t.startswith('5-fold'), 'single-member pairs'), (lambda t: t.startswith('5-fold'), '5-fold set pair')):
    m = np.array([keep(t) for t in alltag])
    if m.sum():
        s2 = float((w[m] * x[m] * yv[m]).sum() / (w[m] * x[m] ** 2).sum()); e2 = float(1 / np.sqrt((w[m] * x[m] ** 2).sum()))
        print(f'  {nm}: slope {s2:+.3f} (SE {e2:.3f}, n {m.sum()})')
# split by direction: moves TOWARD the image teacher vs others
json.dump(dict(pairs=res, pooled_slope=slope, pooled_se=sse), open(H + '/transfer.json', 'w'), indent=1)
print('wrote transfer.json')
