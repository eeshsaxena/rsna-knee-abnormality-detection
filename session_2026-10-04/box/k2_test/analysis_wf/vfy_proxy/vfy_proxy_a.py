"""vfy_proxy_a.py : independent re-check of the lb-proxy-model gold58 numbers (box, CPU only, read-only inputs).
Own code (does not import the proxy scripts). Writes nothing except stdout (redirected to a log under analysis_wf/vfy_proxy).
 1. id alignment: fleet serve ids filtered by non-NaN ACL vs ~/kmimic/ids/gold58.csv; parent / arm files cover them
 2. macro AUCs of the key arms
 3. the 4 context curves (per-column rank blend, fleet weight W), measured deltas
 4. a third latent-r estimator: within-class Pearson of normal scores (pooled Fisher z), vs the stored rA
 5. own least-squares macro r_fit (bounded scalar minimiser) for the 4 curves
"""
import os, json, numpy as np, pandas as pd
from scipy.stats import rankdata, norm
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

H = os.path.expanduser
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
tr = pd.read_csv(H('~/kmimic/kaggle/input/rsna-knee-abnormality-detection/train.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
g58 = pd.read_csv(H('~/kmimic/ids/gold58.csv'), dtype={'StudyInstanceUID': str})
g58ids = set(g58.StudyInstanceUID)
F = pd.read_csv(H('~/k2_test/runs/C_gold58/working/submission_fleetsolo.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
ids_agent = [s for s in F.index if s in tr.index and not np.isnan(tr.loc[s, 'ACL'])]
print('1. ids: agent filter n', len(ids_agent), '| gold58.csv n', len(g58ids), '| equal sets', set(ids_agent) == g58ids,
      '| gold58.csv subset values', g58.subset.unique().tolist())
print('   train rows with non-NaN ACL:', int(tr.ACL.notna().sum()))
ids = sorted(g58ids)
Y = tr.loc[ids, T].values.astype(float)
print('   any NaN label in gold58:', bool(np.isnan(Y).any()), '| positives per label', dict(zip(T, Y.sum(0).astype(int).tolist())))
P = H('~/kmimic/runs/gold58_gpu/localLB_parts/')


def load(f):
    d = pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    miss = [s for s in ids if s not in d.index]
    assert not miss, (f, len(miss))
    assert not d.index.duplicated().any(), f
    return d.loc[ids, T].values.astype(float)


A = {n: load(P + n + '.csv') for n in ['raptor_arm_maxspan-v5_prob', 'raptor_public_rank', 'coat_family_rank', 'raptor_coat_side_rank']}
A['parent_final'] = load(H('~/kmimic/runs/gold58_gpu/localLB_parent_preds.csv'))
A['fleet'] = load(H('~/k2_test/runs/C_gold58/working/submission_fleetsolo.csv'))
A['dread_widedense'] = load(H('~/rsna_train/teacher_train_probs.csv'))


def mac(X): return float(np.mean([roc_auc_score(Y[:, j], X[:, j]) for j in range(12)]))
def rk(X): return np.column_stack([rankdata(X[:, j]) / X.shape[0] for j in range(12)])


MAC = {k: mac(v) for k, v in A.items()}
print('\n2. macro AUC gold58:', {k: round(v, 4) for k, v in MAC.items()})

print('\n3. context curves (fleet weight W, per-column rank blend), measured delta vs base:')
WS = [0.10, 0.15, 0.25, 0.35, 0.50]
curves = {}
for b in ['raptor_arm_maxspan-v5_prob', 'raptor_public_rank', 'coat_family_rank', 'raptor_coat_side_rank']:
    d = [mac((1 - w) * rk(A[b]) + w * rk(A['fleet'])) - MAC[b] for w in WS]
    curves[b] = d
    print(f'   {b:28s} ' + ' '.join(f'W{w:.2f} {x:+.4f}' for w, x in zip(WS, d)))
# K2-style blend into the (contaminated) parent for reference only
d = [mac((1 - w) * rk(A['parent_final']) + w * rk(A['fleet'])) - MAC['parent_final'] for w in (0.10, 0.30)]
print('   parent_final (contaminated)  W.10 %+.4f W.30 %+.4f' % tuple(d))


def nscore(x):
    return norm.ppf((rankdata(x) - 0.5) / len(x))


def rC(a, b):
    out = []
    for j in range(12):
        zs, ws = [], []
        for c in (0, 1):
            m = Y[:, j] == c
            if m.sum() >= 5:
                rr = np.corrcoef(nscore(a[m, j]), nscore(b[m, j]))[0, 1]
                zs.append(np.arctanh(np.clip(rr, -.999, .999))); ws.append(m.sum() - 3)
        out.append(np.tanh(np.average(zs, weights=ws)))
    return float(np.mean(out))


G = json.load(open(H('~/k2_test/analysis_wf/proxy_gold58_out.json')))
print('\n4. latent r of fleet vs refs: stored rA / stored rB / my normal-score within-class rC')
for b in ['raptor_arm_maxspan-v5_prob', 'raptor_public_rank', 'coat_family_rank', 'raptor_coat_side_rank', 'parent_final', 'dread_widedense']:
    gk = 'fleet_f0_mix50|' + b if 'fleet_f0_mix50|' + b in G['pairs'] else b + '|fleet_f0_mix50'
    pr = G['pairs'].get(gk, {})
    print(f'   {b:28s} rA {pr.get("rA", float("nan")):.3f} rB {pr.get("rB", float("nan")):.3f} rC {rC(A["fleet"], A[b]):.3f}')


def dv(a): return np.sqrt(2) * norm.ppf(a)


def bl(aB, aM, r, w):
    v = (1 - w) ** 2 + w ** 2 + 2 * w * (1 - w) * r
    return norm.cdf(((1 - w) * dv(aB) + w * dv(aM)) / np.sqrt(2 * v))


print('\n5. own macro r_fit for the 4 curves (bounded minimiser on [0, .999]):')
for b, d in curves.items():
    f = lambda r: sum((bl(MAC[b], MAC['fleet'], r, w) - MAC[b] - x) ** 2 for w, x in zip(WS, d))
    o = minimize_scalar(f, bounds=(0, 0.999), method='bounded')
    pred = [bl(MAC[b], MAC['fleet'], o.x, w) - MAC[b] for w in WS]
    print(f'   {b:28s} r_fit {o.x:.3f} | resid ' + ' '.join(f'{p - x:+.4f}' for p, x in zip(pred, d)))
