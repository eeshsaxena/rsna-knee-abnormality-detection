"""vbm_formcheck.py (verify_blendmath, 2026-10-05). CPU only, read-only inputs, writes ~/jump2/vbm/.
Tests the MACRO single-d binormal model used by lbmath.py against MEASURED rank-blend curves on fold-0 OOF
(867 val studies, labels = steven_v4 >= .5). For each pair (stronger 'P', weaker 'F'):
  measured macro AUC of (1-W)*rank(P) + W*rank(F) for W = 0..1
  macro model: d from macro AUC, one r fitted to the W .3 read (K2-style) -> predicted b(.5), b(.6), drop .5->.6
  per-label model: per-label d, one r fitted to W .3 -> same predictions
Also dumps per-label AUCs (OOF vs v4, gold74 vs radiologist) to size heterogeneity."""
import os, json, numpy as np, pandas as pd
from scipy.stats import rankdata, norm
from scipy.optimize import brentq
H = os.path.expanduser
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
FR = '~/fleet_runs/'
M = {'F1m': FR + 'f0_s42/oof_f0_s42_raw.csv', 'F1v4': FR + 'f0_s42_v4/oof_f0_s42_raw.csv',
     'F1g': FR + 'f0_s42_gated/oof_f0_s42_raw.csv', 'TTRv4': FR + 'f0_s1042_ttr/oof_f0_s1042_ema.csv',
     'TTRm': FR + 'f0_s1043_ttr_mix50/oof_f0_s1043_raw.csv',
     'raptorproxy': '~/fleet/teacher_v2_raptorproxy_shard0.csv', 'maxspan_v5': '~/fleet/teacher_v2_maxspan_v5_shard0.csv',
     'native384_v8': '~/fleet/teacher_v2_native384_v8_shard0.csv'}
def rd(p):
    d = pd.read_csv(H(p), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID'); return d[T].astype(float)
D = {k: rd(v) for k, v in M.items()}
ids = sorted(set.intersection(*[set(d.index) for d in D.values()]))
Y = rd('~/rsna_train/llm_labels_v4_blend.csv').loc[ids, T].values >= .5
X = {k: d.loc[ids, T].values for k, d in D.items()}
def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def per(Z): return np.array([auc(Y[:, j], Z[:, j]) for j in range(12)])
def rp(a): return np.column_stack([rankdata(a[:, j]) / len(a) for j in range(12)])
def dv(a): return np.sqrt(2) * norm.ppf(np.clip(a, 1e-4, 1 - 1e-4))
def bl(dP, dF, r, W):
    v = (1 - W) ** 2 + W ** 2 + 2 * W * (1 - W) * r; return norm.cdf(((1 - W) * dP + W * dF) / np.sqrt(2 * v))
WS = np.round(np.arange(0, 1.0001, .1), 1)
out = {'n': len(ids), 'pairs': {}, 'perlabel': {}}
print('n', len(ids))
for k in X: out['perlabel'][k] = per(X[k]).round(4).tolist()
print('per-label OOF AUC vs v4:'); [print(f'  {k:12s}', ' '.join(f'{a:.3f}' for a in out['perlabel'][k]), f'| macro {np.mean(out["perlabel"][k]):.4f} sd {np.std(out["perlabel"][k]):.3f}') for k in X]
PAIRS = [('TTRm', 'F1v4'), ('TTRm', 'TTRv4'), ('F1m', 'F1v4'), ('F1m', 'TTRv4'), ('TTRm', 'F1g'),
         ('raptorproxy', 'TTRm'), ('raptorproxy', 'F1m'), ('raptorproxy', 'TTRv4'), ('maxspan_v5', 'TTRv4'), ('native384_v8', 'TTRv4'), ('native384_v8', 'F1v4')]
print('\npair | measured b(W) W=0..1 | fit@W.3 macro r -> pred b.5 b.6 drop | per-label r -> pred | measured drop .5->.6 | W* measured')
for p, f in PAIRS:
    RP, RF = rp(X[p]), rp(X[f])
    meas = np.array([per((1 - w) * RP + w * RF).mean() for w in WS])
    aP, aF = per(X[p]), per(X[f]); mP, mF = aP.mean(), aF.mean()
    def fit(fun, target, W):
        g = lambda r: fun(r, W) - target
        lo, hi = -0.9, 0.9999
        try: return brentq(g, lo, hi)
        except ValueError: return np.nan
    mac = lambda r, W: bl(dv(mP), dv(mF), r, W)
    pl = lambda r, W: bl(dv(aP), dv(aF), r, W).mean()
    b3 = meas[3]
    rm_ = fit(mac, b3, .3); rp_ = fit(pl, b3, .3)
    pm = [mac(rm_, w) for w in (.5, .6)]; pp = [pl(rp_, w) for w in (.5, .6)]
    # also fit macro r to W .5 read, then predict W .6 (the aastik-type test)
    rm5 = fit(mac, meas[5], .5); pm6 = mac(rm5, .6)
    rec = dict(meas=meas.round(5).tolist(), mP=mP, mF=mF, r_mac_w3=rm_, r_pl_w3=rp_, pred_mac=pm, pred_pl=pp,
               r_mac_w5=rm5, pred_mac6_from5=pm6, meas_drop=float(meas[5] - meas[6]))
    out['pairs'][f'{p}|{f}'] = rec
    print(f'{p:12s}/{f:6s} P {mP:.4f} F {mF:.4f} | ' + ' '.join(f'{m:.4f}' for m in meas) +
          f' | mac r {rm_:.3f} -> .5 {pm[0]:.4f} .6 {pm[1]:.4f} (err .5 {pm[0]-meas[5]:+.4f}, .6 {pm[1]-meas[6]:+.4f})'
          f' | pl r {rp_:.3f} -> err .5 {pp[0]-meas[5]:+.4f} .6 {pp[1]-meas[6]:+.4f}'
          f' | drop meas {meas[5]-meas[6]:+.4f} mac(from W.5 fit) {meas[5]-pm6:+.4f} | W* {WS[meas.argmax()]:.1f}')
# gold74 per-label (vs radiologist labels) for the 4-5 members with gold74 preds
g = pd.read_csv(H('~/rsna_train/gold74_eval_set.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
G = g[T].astype(float).values >= .5
print('\ngold74 positives per label:', G.sum(0).tolist())
import glob
for fpath in sorted(glob.glob(H('~/jump2/gold74_preds/*.csv'))):
    k = os.path.basename(fpath)[:-4]; d = pd.read_csv(fpath, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[g.index, T].values
    a = np.array([auc(G[:, j], d[:, j]) if 0 < G[:, j].sum() < len(G) else np.nan for j in range(12)])
    out['perlabel']['g74_' + k] = a.round(4).tolist()
    print(f'  g74 {k:28s}', ' '.join(f'{x:.3f}' for x in a), f'| macro {np.nanmean(a):.4f}')
print('gold74 source counts:', g['source'].value_counts().to_dict())
os.makedirs(H('~/jump2/vbm'), exist_ok=True)
json.dump(out, open(H('~/jump2/vbm/vbm_formcheck.json'), 'w'), indent=1, default=float)
