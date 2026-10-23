"""proxy_oof867.py : member-to-member correlation of our fleet on the fold-0 val studies (867, all members OOS there).
Read-only inputs (~/fleet_runs/*/oof_*.csv = each run's best-epoch OOF so far; TTR and gated are still training).
Labels = steven_v4 LLM labels (>= .5). Reports AUC vs v4, Spearman, latent r (MC inversion rA and within-class rB),
and validates the binormal ENSEMBLE formula (k equal members, latent r) against measured rank-mean ensembles.
Writes ~/k2_test/analysis_wf/proxy_oof867_out.json.
"""
import os, json, numpy as np, pandas as pd
from scipy.stats import rankdata, norm, spearmanr

H = os.path.expanduser
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
M = {'mix50_raw': '~/fleet_runs/f0_s42/oof_f0_s42_raw.csv', 'mix50_ema': '~/fleet_runs/f0_s42/oof_f0_s42_ema.csv',
     'v4_raw': '~/fleet_runs/f0_s42_v4/oof_f0_s42_raw.csv', 'v4_ema': '~/fleet_runs/f0_s42_v4/oof_f0_s42_ema.csv',
     'gated_raw(ep<=3)': '~/fleet_runs/f0_s42_gated/oof_f0_s42_raw.csv',
     'ttr_raw(ep<=8)': '~/fleet_runs/f0_s1042_ttr/oof_f0_s1042_raw.csv',
     'ttr_ema(ep<=8)': '~/fleet_runs/f0_s1042_ttr/oof_f0_s1042_ema.csv'}
D = {k: pd.read_csv(H(v), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID') for k, v in M.items()}
ids = sorted(set.intersection(*[set(d.index) for d in D.values()]))
lab = pd.read_csv(H('~/rsna_train/llm_labels_v4_blend.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
Y = (lab.loc[ids, T].values >= 0.5)
X = {k: d.loc[ids, T].values.astype(float) for k, d in D.items()}
N = len(ids); prev = Y.mean(0)


def fastauc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def per(A): return np.array([fastauc(Y[:, j], A[:, j]) for j in range(12)])
def rp(a): return np.column_stack([rankdata(a[:, j]) / len(a) for j in range(a.shape[1])])
def dval(a): return np.sqrt(2) * norm.ppf(np.clip(a, 0.02, 0.998))


rng = np.random.default_rng(7); NMC = 12000
Z1, Z2, U = rng.standard_normal(NMC), rng.standard_normal(NMC), rng.random(NMC)
RG = np.linspace(-0.5, 0.995, 46)
def rank01(x): return x.argsort().argsort().astype(float) / len(x)


def rA(a, b, pa, pb):
    out = []
    for j in range(12):
        y = U < prev[j]; r1 = rank01(dval(pa[j]) * y + Z1)
        c = np.maximum.accumulate(np.array([np.corrcoef(r1, rank01(dval(pb[j]) * y + r * Z1 + np.sqrt(1 - r * r) * Z2))[0, 1] for r in RG]))
        out.append(float(np.interp(spearmanr(a[:, j], b[:, j]).correlation, c, RG)))
    return np.array(out)


def rB(a, b):
    out = []
    for j in range(12):
        zs, ws = [], []
        for cls in (False, True):
            m = Y[:, j] == cls
            if m.sum() >= 8:
                zs.append(np.arctanh(np.clip(spearmanr(a[m, j], b[m, j]).correlation, -.999, .999))); ws.append(m.sum() - 3)
        out.append(2 * np.sin(np.pi * np.tanh(np.average(zs, weights=ws)) / 6))
    return np.array(out)


PL = {k: per(v) for k, v in X.items()}
res = dict(n=N, members={k: float(PL[k].mean()) for k in X}, pairs={}, ens_check=[])
print(f'n={N} fold-0 val studies; AUC vs steven_v4 (>=.5)')
for k in X: print(f'  {k:18s} {PL[k].mean():.4f}')
keys = list(X)
print('pair | Spearman | rA | rB')
for i, a in enumerate(keys):
    for b in keys[i + 1:]:
        sp = float(np.mean([spearmanr(X[a][:, j], X[b][:, j]).correlation for j in range(12)]))
        ra = rA(X[a], X[b], PL[a], PL[b]); rb = rB(X[a], X[b])
        res['pairs'][f'{a}|{b}'] = dict(spearman=sp, rA=float(ra.mean()), rB=float(rb.mean()))
        print(f'  {a:18s} {b:18s} sp {sp:.3f} rA {ra.mean():.3f} rB {rb.mean():.3f}')


# ensemble formula check: equal-weight rank mean of 2 members vs binormal prediction with the pair's latent r
def ens_pred(pa, pb, r):
    dm = (dval(pa) + dval(pb)) / 2; v = (2 + 2 * r) / 4
    return norm.cdf(dm / np.sqrt(2 * v))


print('2-member rank-mean ensembles: measured vs binormal (per label, latent r)')
for a, b in [('mix50_raw', 'v4_raw'), ('mix50_raw', 'ttr_raw(ep<=8)'), ('v4_raw', 'ttr_raw(ep<=8)'),
             ('mix50_raw', 'gated_raw(ep<=3)'), ('mix50_raw', 'mix50_ema')]:
    E = rp(rp(X[a]) + rp(X[b])); meas = float(per(E).mean())
    ra = rA(X[a], X[b], PL[a], PL[b]); rb = rB(X[a], X[b])
    pa_ = float(ens_pred(PL[a], PL[b], ra).mean()); pb_ = float(ens_pred(PL[a], PL[b], rb).mean())
    best = max(PL[a].mean(), PL[b].mean())
    res['ens_check'].append(dict(a=a, b=b, measured=meas, pred_rA=pa_, pred_rB=pb_, best_single=float(best)))
    print(f'  {a}+{b}: measured {meas:.4f} | pred rA {pa_:.4f} rB {pb_:.4f} | best single {best:.4f} mean {(PL[a].mean() + PL[b].mean()) / 2:.4f}')
json.dump(res, open(H('~/k2_test/analysis_wf/proxy_oof867_out.json'), 'w'), indent=1)
print('wrote proxy_oof867_out.json')
