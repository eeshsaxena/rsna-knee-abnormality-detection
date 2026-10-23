"""oofx.py (jump2, 2026-10-05): fold-0 OOF member/ensemble/parent-likeness analysis. CPU only, read-only inputs.
Inputs: ~/fleet_runs/*/oof_*.csv (5 finished fold-0 runs), ~/rsna_train label tables, ~/fleet/teacher_v2_*_shard0.csv
(parent's Raptor arms on train studies). Output: ~/jump2/oofx_out.json + stdout log.
Metric: macro AUC vs steven_v4 (>=.5) on the 867 fold-0 val studies.
r measures:
  sp   = mean per-label Spearman (signal + noise)
  rB   = within-class Spearman (pooled Fisher z over the two v4 classes), converted to Pearson (latent r)
  reff = effective binormal r fitted to MEASURED rank blends at W .1-.5 (same construction as proxy_fit's r_fit)
"""
import os, json, itertools, numpy as np, pandas as pd
from scipy.stats import rankdata, norm, spearmanr
from scipy.optimize import minimize_scalar

H = os.path.expanduser
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
FR = '~/fleet_runs/'
M = {'F1mix50': FR + 'f0_s42/oof_f0_s42_raw.csv', 'F1mix50_ema': FR + 'f0_s42/oof_f0_s42_ema.csv',
     'F1v4': FR + 'f0_s42_v4/oof_f0_s42_raw.csv', 'F1v4_ema': FR + 'f0_s42_v4/oof_f0_s42_ema.csv',
     'F1gated': FR + 'f0_s42_gated/oof_f0_s42_raw.csv', 'F1gated_ema': FR + 'f0_s42_gated/oof_f0_s42_ema.csv',
     'TTRv4_raw': FR + 'f0_s1042_ttr/oof_f0_s1042_raw.csv', 'TTRv4': FR + 'f0_s1042_ttr/oof_f0_s1042_ema.csv',
     'TTRmix50': FR + 'f0_s1043_ttr_mix50/oof_f0_s1043_raw.csv', 'TTRmix50_ema': FR + 'f0_s1043_ttr_mix50/oof_f0_s1043_ema.csv'}
BEST = ['F1mix50', 'F1v4', 'F1gated', 'TTRv4', 'TTRmix50']
ARMS = {'raptorproxy': '~/fleet/teacher_v2_raptorproxy_shard0.csv', 'native384_v8': '~/fleet/teacher_v2_native384_v8_shard0.csv',
        'maxspan_v5': '~/fleet/teacher_v2_maxspan_v5_shard0.csv', 'widedense_teacher': '~/rsna_train/teacher_train_probs.csv'}


def rd(p):
    d = pd.read_csv(H(p), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    return d[T].astype(float)


D = {k: rd(v) for k, v in M.items()}
A = {k: rd(v) for k, v in ARMS.items()}
ids = sorted(set.intersection(*[set(d.index) for d in D.values()]) & set.intersection(*[set(a.index) for a in A.values()]))
lab = rd('~/rsna_train/llm_labels_v4_blend.csv')
Y = lab.loc[ids, T].values >= 0.5
N = len(ids); prev = Y.mean(0)
X = {k: d.loc[ids, T].values for k, d in D.items()}
XA = {k: a.loc[ids, T].values for k, a in A.items()}


def fastauc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def per(Z): return np.array([fastauc(Y[:, j], Z[:, j]) for j in range(12)])
def rp(a): return np.column_stack([rankdata(a[:, j]) / len(a) for j in range(a.shape[1])])
def dv(a): return np.sqrt(2) * norm.ppf(np.clip(a, 0.02, 0.998))


def sp(a, b): return float(np.mean([spearmanr(a[:, j], b[:, j]).correlation for j in range(12)]))


def rB(a, b):
    out = []
    for j in range(12):
        zs, ws = [], []
        for cls in (False, True):
            m = Y[:, j] == cls
            if m.sum() >= 8:
                zs.append(np.arctanh(np.clip(spearmanr(a[m, j], b[m, j]).correlation, -.999, .999))); ws.append(m.sum() - 3)
        out.append(2 * np.sin(np.pi * np.tanh(np.average(zs, weights=ws)) / 6))
    return float(np.mean(out))


WS = [0.1, 0.2, 0.3, 0.4, 0.5]


def blend_pred(pa, pb, r, w):
    m = (1 - w) * dv(pa) + w * dv(pb); v = (1 - w) ** 2 + w ** 2 + 2 * w * (1 - w) * r
    return float(norm.cdf(m / np.sqrt(2 * v)).mean())


def reff(a, b):
    """effective r of (a, b) fitted to measured macro AUC of global rank blends at WS (vs v4 labels)."""
    pa, pb = per(a), per(b); Ra, Rb = rp(a), rp(b)
    meas = [float(per((1 - w) * Ra + w * Rb).mean()) for w in WS]
    f = lambda r: sum((blend_pred(pa, pb, r, w) - m) ** 2 for w, m in zip(WS, meas))
    o = minimize_scalar(f, bounds=(-0.5, 0.9999), method='bounded')
    return float(o.x), meas


res = dict(n=N, members={}, arms={}, pairs={}, ens={}, labels={})
print(f'n={N} fold-0 val studies (also covered by the parent-arm files); AUC vs steven_v4 (>=.5); prevalence mean {prev.mean():.3f}')
for k, a in XA.items():
    res['arms'][k] = float(per(a).mean()); print(f'  ARM {k:18s} AUCvsV4 {res["arms"][k]:.4f}')
print('member | AUCvsV4 | sp/rB/reff vs raptorproxy | sp vs native384_v8, maxspan_v5, widedense_teacher')
for k in X:
    au = float(per(X[k]).mean())
    s_rp = sp(X[k], XA['raptorproxy']); b_rp = rB(X[k], XA['raptorproxy'])
    e_rp, _ = reff(X[k], XA['raptorproxy']) if k in BEST else (float('nan'), None)
    extra = {an: sp(X[k], XA[an]) for an in ('native384_v8', 'maxspan_v5', 'widedense_teacher')}
    res['members'][k] = dict(auc=au, sp_rp=s_rp, rB_rp=b_rp, reff_rp=e_rp, **{'sp_' + a: v for a, v in extra.items()})
    print(f'  {k:13s} {au:.4f} | {s_rp:.3f} / {b_rp:.3f} / {e_rp:.3f} | ' + ' '.join(f'{v:.3f}' for v in extra.values()))

print('\npairs among best-tag members: sp | rB | reff (fitted to measured rank blends)')
for a, b in itertools.combinations(BEST, 2):
    s = sp(X[a], X[b]); r = rB(X[a], X[b]); e, meas = reff(X[a], X[b])
    res['pairs'][f'{a}|{b}'] = dict(sp=s, rB=r, reff=e, blend_meas=meas)
    print(f'  {a:9s} {b:9s} sp {s:.3f} rB {r:.3f} reff {e:.3f} | blend W.1-.5 ' + ' '.join(f'{m:.4f}' for m in meas))
for a, b in [('F1mix50', 'F1mix50_ema'), ('TTRmix50', 'TTRmix50_ema'), ('TTRv4', 'TTRv4_raw')]:
    s = sp(X[a], X[b]); r = rB(X[a], X[b]); e, meas = reff(X[a], X[b])
    res['pairs'][f'{a}|{b}'] = dict(sp=s, rB=r, reff=e)
    print(f'  {a:9s} {b:12s} sp {s:.3f} rB {r:.3f} reff {e:.3f} (same run raw/ema)')

print('\nequal-weight rank-mean ensembles of best-tag members: AUCvsV4 | best single | mean single | sp/rB vs raptorproxy | reff vs raptorproxy')
for k in range(1, len(BEST) + 1):
    for S in itertools.combinations(BEST, k):
        E = np.mean([rp(X[m]) for m in S], axis=0)
        au = float(per(E).mean()); bs = max(res['members'][m]['auc'] for m in S); ms = float(np.mean([res['members'][m]['auc'] for m in S]))
        s_rp = sp(E, XA['raptorproxy']); b_rp = rB(E, XA['raptorproxy']); e_rp, _ = reff(E, XA['raptorproxy'])
        res['ens']['+'.join(S)] = dict(auc=au, best=bs, mean=ms, sp_rp=s_rp, rB_rp=b_rp, reff_rp=e_rp)
        print(f'  {"+".join(S):42s} {au:.4f} | {bs:.4f} | {ms:.4f} | {s_rp:.3f}/{b_rp:.3f} | {e_rp:.3f}')

# label tables themselves: Spearman vs the parent's Raptor arms on all non-gold train studies (target-level parent-likeness)
L = {'steven_v4': '~/rsna_train/llm_labels_v4_blend.csv', 'mix50_logit': '~/rsna_train/labels_mix50_logit.csv',
     'gated75_logit': '~/rsna_train/labels_gated75_logit.csv', 'gpt56sol': '~/rsna_train/labels_llm_gpt56sol.csv',
     'qwen7b': '~/rsna_train/pseudo_labels_qwen7b.csv', 'pseudo_v1': '~/rsna_train/pseudo_labels_v1.csv',
     'FINAL2': '~/rsna_train/FINAL2_labels_with_confidence.csv'}
LT = {}
for k, p in L.items():
    try:
        LT[k] = rd(p)
    except Exception as e:
        print('label read fail', k, e)
allids = sorted(set.intersection(*[set(v.index) for v in LT.values()]) & set(A['raptorproxy'].index))
print(f'\nlabel tables vs parent Raptor arms, {len(allids)} train studies: sp vs raptorproxy | sp vs native384_v8 | sp vs steven_v4')
for k, v in LT.items():
    a = v.loc[allids, T].values
    s1 = sp(a, A['raptorproxy'].loc[allids, T].values); s2 = sp(a, A['native384_v8'].loc[allids, T].values)
    s3 = sp(a, LT['steven_v4'].loc[allids, T].values)
    res['labels'][k] = dict(sp_rp=s1, sp_v8=s2, sp_v4=s3)
    print(f'  {k:14s} {s1:.3f} | {s2:.3f} | {s3:.3f}')
json.dump(res, open(H('~/jump2/oofx_out.json'), 'w'), indent=1)
print('wrote ~/jump2/oofx_out.json')
