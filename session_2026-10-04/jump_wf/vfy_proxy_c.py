"""vfy_proxy_c.py : independent re-check of the LB-side MODEL numbers + crop/dread gold58 r_fit + prior sensitivity.
Box, CPU, read-only inputs, prints only.
"""
import os, numpy as np, pandas as pd
from scipy.stats import rankdata, norm
from scipy.optimize import minimize_scalar, brentq
from sklearn.metrics import roc_auc_score

H = os.path.expanduser
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]


def dv(a): return np.sqrt(2) * norm.ppf(a)


def bl(P, a, r, w):
    v = (1 - w) ** 2 + w ** 2 + 2 * w * (1 - w) * r
    return norm.cdf(((1 - w) * dv(P) + w * dv(a)) / np.sqrt(2 * v))


# ---- 1. crop / dread r_fit on gold58 (own code)
tr = pd.read_csv(H('~/kmimic/kaggle/input/rsna-knee-abnormality-detection/train.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
ids = sorted(pd.read_csv(H('~/kmimic/ids/gold58.csv'), dtype={'StudyInstanceUID': str}).StudyInstanceUID)
Y = tr.loc[ids, T].values.astype(float)
P = H('~/kmimic/runs/gold58_gpu/localLB_parts/')
ld = lambda f: pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, T].values.astype(float)
A = {n: ld(P + n + '.csv') for n in ['raptor_public_rank', 'raptor_coat_side_rank']}
A['parent_final'] = ld(H('~/kmimic/runs/gold58_gpu/localLB_parent_preds.csv'))
A['dread'] = ld(H('~/rsna_train/teacher_train_probs.csv'))
z = np.load(H('~/k2_test/analysis_wf/proxy_crop_gold74_preds.npz'), allow_pickle=True)
zid = {str(s): i for i, s in enumerate(z['ids'])}; zi = [zid[s] for s in ids]
print('npz keys', [k for k in z.files])
rk = lambda X: np.column_stack([rankdata(X[:, j]) / X.shape[0] for j in range(12)])
for k, nm in [('finetune_cropaug_f1_radimagenet', 'cropaug_f1'), ('finetune_cropaug_f2_radimagenet', 'cropaug_f2'), ('finetune_crop_f1_radimagenet', 'crop_f1')]:
    A[nm] = z[k][zi].astype(float)
A['K1ens3'] = rk(rk(A['cropaug_f1']) + rk(A['cropaug_f2']) + rk(A['crop_f1']))
mac = lambda X: float(np.mean([roc_auc_score(Y[:, j], X[:, j]) for j in range(12)]))
MAC = {k: mac(v) for k, v in A.items()}
print('gold58 macro:', {k: round(v, 4) for k, v in MAC.items()})
WS = [0.10, 0.15, 0.25, 0.35, 0.50]
for m in ['K1ens3', 'cropaug_f1', 'dread']:
    line = f'  r_fit {m:10s}'
    for b in ['raptor_coat_side_rank', 'parent_final', 'raptor_public_rank']:
        d = [mac((1 - w) * rk(A[b]) + w * rk(A[m])) - MAC[b] for w in WS]
        o = minimize_scalar(lambda r: sum((bl(MAC[b], MAC[m], r, w) - MAC[b] - x) ** 2 for w, x in zip(WS, d)), bounds=(0, .999), method='bounded')
        line += f' | {b[:16]} {o.x:.3f}'
    print(line)

# ---- 2. spot checks of the LB-side tables
print('\nspot checks (parent .943 true):')
for s, r in [(0.925, 0.90), (0.915, 0.94), (0.935, 0.97), (0.905, 0.90)]:
    print(f'  BLEND W.30 solo {s} r {r}: {bl(0.943, s, r, 0.30):.4f}')
WG = np.linspace(0, 1, 2001)
for r, s in [(0.80, 0.938), (0.90, 0.9475), (0.85, 0.9435), (0.94, 0.950)]:
    b = bl(0.943, s, r, WG); i = b.argmax(); print(f'  need-table: r {r} solo {s}: W* {WG[i]:.2f} lift {b[i] - 0.943:+.4f}')


# inversion cells (own brute force: any parent in [.9425,.9435], solo +-.0005, read +-.0005)
def inv(s, read):
    RG = np.linspace(0.3, 0.9999, 3000); ok = np.zeros_like(RG, bool)
    for Pp in np.linspace(0.9425, 0.9435, 11):
        for ss in np.linspace(s - 0.0005, s + 0.0005, 11):
            ok |= np.abs(bl(Pp, ss, RG, 0.30) - read) <= 0.0005
    return (RG[ok].min(), RG[ok].max()) if ok.any() else (np.nan, np.nan)


for s, x in [(0.915, 0.940), (0.925, 0.940), (0.925, 0.943), (0.915, 0.943), (0.925, 0.942), (0.935, 0.943)]:
    lo, hi = inv(s, x); print(f'  inversion solo {s} read {x}: r [{lo:.3f}, {hi:.3f}]')

# Tian / crop C3 anchors
for nm, a, reads, PG in [('Tian .927 on .942', 0.927, [(0.10, 0.942), (0.20, 0.941)], (0.9415, 0.9425)),
                         ('C3 K1 at solo .924', 0.924, [(0.25, 0.942)], (0.9425, 0.9435)),
                         ('C3 K1 at solo .928', 0.928, [(0.25, 0.942)], (0.9425, 0.9435))]:
    RG = np.linspace(0.3, 0.9999, 3000); ok = np.zeros_like(RG, bool)
    for Pp in np.linspace(*PG, 11):
        for aa in np.linspace(a - 0.0005, a + 0.0005, 5):
            g = np.ones_like(RG, bool)
            for w, d in reads: g &= np.abs(bl(Pp, aa, RG, w) - d) <= 0.0005
            ok |= g
    print(f'  anchor {nm}: r [{RG[ok].min():.3f}, {RG[ok].max():.3f}]')

# ---- 3. prior predictive: agent prior vs alternatives
rng = np.random.default_rng(11); n = 400000
Pp = rng.uniform(0.9425, 0.9435, n)
def pp(name, s, r):
    b = np.round(bl(Pp, s, r, 0.30), 3)
    print(f'  {name:62s} median {np.median(bl(Pp, s, r, 0.30)):.4f} P>=.943 {(b >= 0.943).mean():.2f} P>=.942 {(b >= 0.942).mean():.2f} P<=.939 {(b <= 0.939).mean():.2f}')
print('\nprior predictive of the K2 BLEND W.30 read:')
s_ag = np.clip(rng.normal(0.922, 0.008, n), 0.89, 0.95)
r_ag = np.where(rng.random(n) < 0.5, rng.uniform(0.93, 0.98, n), rng.uniform(0.88, 0.93, n))
pp('agent: solo N(.922,.008), r 50% U[.93,.98] / 50% U[.88,.93]', s_ag, r_ag)
s_w = np.clip(rng.normal(0.922, 0.013, n), 0.88, 0.96)
pp('wider solo N(.922,.013), same r', s_w, r_ag)
pp('solo N(.922,.008), r U[.86,.93] (gold58 r_fit, no LB inflation)', s_ag, rng.uniform(0.86, 0.93, n))
pp('solo N(.922,.008), r U[.92,.97] (Tian-analog only)', s_ag, rng.uniform(0.92, 0.97, n))
