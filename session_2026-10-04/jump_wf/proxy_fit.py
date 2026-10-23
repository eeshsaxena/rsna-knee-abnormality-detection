"""proxy_fit.py : effective (blend-fitted) correlation on gold58 vs the latent (Spearman-implied) correlation.

For each ordered pair (base, member) the macro-level binormal r_eff that best fits the MEASURED gold58 rank-blend curve
(W .10/.15/.25/.35/.50) is compared with rA (latent r from proxy_gold58_out.json). Offset r_eff - rA is summarised and
cross-validated (leave-one-base-out). Also prints r_eff of our members (fleet, crop, widedense) vs the OOS parent sides,
which is what the LB transfer uses. Box, CPU, read-only inputs. Writes proxy_fit_out.json.
"""
import os, json, numpy as np, pandas as pd
from scipy.stats import rankdata, norm
from sklearn.metrics import roc_auc_score

H = os.path.expanduser
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
G = json.load(open(H('~/k2_test/analysis_wf/proxy_gold58_out.json')))
tr = pd.read_csv(H('~/kmimic/kaggle/input/rsna-knee-abnormality-detection/train.csv'),
                 dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
R = H('~/kmimic/runs/gold58_gpu'); P = R + '/localLB_parts/'
F = pd.read_csv(H('~/k2_test/runs/C_gold58/working/submission_fleetsolo.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
ids = [s for s in F.index if s in tr.index and not np.isnan(tr.loc[s, 'ACL'])]
Y = tr.loc[ids, T].values.astype(float)


def lc(f):
    return pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, T].values.astype(float)


def rp(a): return np.column_stack([rankdata(a[:, j]) / len(a) for j in range(a.shape[1])])
def mac(X): return float(np.mean([roc_auc_score(Y[:, j], X[:, j]) for j in range(12)]))


A = {}
for nm in ['raptor_arm_maxspan-v5_prob', 'raptor_arm_maxspan-v5-reverse_prob', 'raptor_arm_native384-v8_prob',
           'raptor_arm_native384dense-v10_prob', 'raptor_public_rank', 'raptor_coat_side_rank', 'coat_family_rank',
           'coat_d4_prob', 'coat_global96_prob', 'coat_repairv1_prob', 'coat_resgated_prob', 'a5_foldmean_prob']:
    A[nm] = lc(P + nm + '.csv')
A['parent_final'] = lc(R + '/localLB_parent_preds.csv')
A['fleet_f0_mix50'] = F.loc[ids, T].values.astype(float)
z = np.load(H('~/k2_test/analysis_wf/proxy_crop_gold74_preds.npz'), allow_pickle=True)
zid = {str(s): i for i, s in enumerate(z['ids'])}; zi = [zid[s] for s in ids]
for k, nm in [('finetune_cropaug_f1_radimagenet', 'crop_cropaug_f1'), ('finetune_cropaug_f2_radimagenet', 'crop_cropaug_f2'),
              ('finetune_crop_f1_radimagenet', 'crop_crop_f1'), ('finetune_cropaug_mixSA_f1', 'crop_mixSA_f1')]:
    A[nm] = z[k][zi].astype(float)
A['crop_K1_ens3'] = rp(rp(A['crop_cropaug_f1']) + rp(A['crop_cropaug_f2']) + rp(A['crop_crop_f1']))
A['dread_widedense'] = lc(H('~/rsna_train/teacher_train_probs.csv'))
MAC = {k: mac(v) for k, v in A.items()}
WS = [0.10, 0.15, 0.25, 0.35, 0.50]


def dv(a): return np.sqrt(2) * norm.ppf(a)


def bl(aB, aM, r, w):
    v = (1 - w) ** 2 + w ** 2 + 2 * w * (1 - w) * r
    return norm.cdf(((1 - w) * dv(aB) + w * dv(aM)) / np.sqrt(2 * v))


GR = np.linspace(0.0, 0.999, 1000)


def curve(b, m): return [mac(rp((1 - w) * rp(A[b]) + w * rp(A[m]))) - MAC[b] for w in WS]


def fit(b, m, meas):
    sse = np.zeros_like(GR)
    for w, d in zip(WS, meas): sse += (bl(MAC[b], MAC[m], GR, w) - MAC[b] - d) ** 2
    return float(GR[int(np.argmin(sse))])


def rAof(a, b):
    k = f'{a}|{b}' if f'{a}|{b}' in G['pairs'] else f'{b}|{a}'
    return G['pairs'][k]['rA'] if k in G['pairs'] else np.nan


vset = ['raptor_arm_maxspan-v5_prob', 'raptor_arm_maxspan-v5-reverse_prob', 'raptor_arm_native384-v8_prob',
        'raptor_arm_native384dense-v10_prob', 'coat_d4_prob', 'coat_global96_prob', 'coat_repairv1_prob',
        'coat_resgated_prob', 'a5_foldmean_prob', 'raptor_public_rank', 'coat_family_rank', 'raptor_coat_side_rank',
        'fleet_f0_mix50', 'crop_K1_ens3', 'dread_widedense']
rows = []
for b in vset:
    for m in vset:
        if b == m: continue
        meas = curve(b, m)
        rows.append(dict(base=b, member=m, aB=MAC[b], aM=MAC[m], rA=rAof(b, m), r_fit=fit(b, m, meas), meas=meas))
V = pd.DataFrame(rows)
V['off'] = V.r_fit - V.rA
inner = V[(V.r_fit > 0.005) & (V.r_fit < 0.995)]
print(f'pairs {len(V)}, with interior fit {len(inner)}')
print(f'offset r_fit - rA: mean {inner.off.mean():+.3f} median {inner.off.median():+.3f} sd {inner.off.std():.3f}')
b1 = np.polyfit(inner.rA, inner.r_fit, 1)
print(f'linear: r_fit = {b1[0]:.3f} * rA + {b1[1]:.3f}')
# noise-ratio form: (1 - r_fit) = k (1 - rA)
kk = float(np.median((1 - inner.r_fit) / (1 - inner.rA)))
print(f'noise-ratio form: (1 - r_fit) = {kk:.3f} * (1 - rA)  (median)')
# leave-one-base-out CV of the offset model
err0, err1, err2 = [], [], []
for b in vset:
    tr_ = inner[inner.base != b]; te_ = V[V.base == b]
    off = tr_.off.mean(); k2 = float(np.median((1 - tr_.r_fit) / (1 - tr_.rA)))
    for r in te_.itertuples():
        for w, d in zip(WS, r.meas):
            err0.append(bl(r.aB, r.aM, r.rA, w) - r.aB - d)
            err1.append(bl(r.aB, r.aM, min(r.rA + off, .999), w) - r.aB - d)
            err2.append(bl(r.aB, r.aM, 1 - k2 * (1 - r.rA), w) - r.aB - d)
for nm, e in [('rA (no offset)', err0), ('rA + offset (LOBO)', err1), ('noise-ratio (LOBO)', err2)]:
    e = np.array(e); print(f'  {nm:22s} MAE {np.abs(e).mean():.4f} bias {e.mean():+.4f} rmse {np.sqrt((e ** 2).mean()):.4f}')
# the members we transfer to LB: r_fit vs the OOS sides and vs the (contaminated) parent
out = dict(offset_mean=float(inner.off.mean()), offset_sd=float(inner.off.std()), lin=b1.tolist(), noise_ratio_k=kk,
           cv=dict(mae_noofs=float(np.abs(err0).mean()), mae_ofs=float(np.abs(err1).mean()), mae_ratio=float(np.abs(err2).mean()),
                   bias_noofs=float(np.mean(err0)), bias_ofs=float(np.mean(err1))), members={})
print('\nmember r_fit (macro, gold58) vs reference arms [rA in brackets]:')
refs = ['raptor_coat_side_rank', 'raptor_public_rank', 'coat_family_rank', 'raptor_arm_maxspan-v5_prob', 'parent_final']
for m in ['fleet_f0_mix50', 'crop_K1_ens3', 'crop_cropaug_f1', 'crop_mixSA_f1', 'crop_crop_f1', 'dread_widedense']:
    line = f'  {m:18s} ({MAC[m]:.4f})'; out['members'][m] = {}
    for b in refs:
        meas = curve(b, m); rf = fit(b, m, meas); ra = rAof(m, b)
        out['members'][m][b] = dict(r_fit=rf, rA=ra, meas=meas, aB=MAC[b], aM=MAC[m])
        line += f' | {b[:14]} {rf:.3f} [{ra:.3f}]'
    print(line)
V.drop(columns=['meas']).to_csv(H('~/k2_test/analysis_wf/proxy_fit_pairs.csv'), index=False)
json.dump(out, open(H('~/k2_test/analysis_wf/proxy_fit_out.json'), 'w'), indent=1)
print('wrote proxy_fit_out.json, proxy_fit_pairs.csv')
