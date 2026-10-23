"""Verifier step 1 (independent of inject_common.py): check the agent's recovered DINO / Rad held-out folds against
ground truth shipped in the input datasets (DINO oof.npz in rsna-knee-weights, Rad v52_oof.csv / v52_e11_oof.csv with a
fold column). CPU only, read only on the inputs, writes only into ~/k2_test/analysis_wf/verify_inner/."""
import json, numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]
H = '/home/training'
R = H + '/kmimic/runs/gold58_gpu'; W = R + '/kaggle/working'
INP = H + '/kmimic/kaggle/input'
ids = pd.read_csv(R + '/kaggle/input/rsna-knee-abnormality-detection/test.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
g58 = pd.read_csv(H + '/kmimic/ids/gold58.csv', dtype={'StudyInstanceUID': str})
print('test.csv ids == gold58 id set:', set(ids) == set(g58.StudyInstanceUID), len(ids))
tr = pd.read_csv(INP + '/rsna-knee-abnormality-detection/train.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
Y = tr.loc[ids, T].to_numpy(float); assert not np.isnan(Y).any()
n = len(ids)


def mauc(Y, X):
    return float(np.mean([roc_auc_score(Y[:, j], X[:, j]) for j in range(12)]))


def npz(path, key='values'):
    z = np.load(path, allow_pickle=False); pos = {str(s): i for i, s in enumerate(z['study_uids'])}
    v = z[key]; ix = [pos[s] for s in ids]
    return v[..., ix, :] if v.ndim == 3 else v[ix]


man = json.load(open(INP + '/rsna-knee-weights/manifest.json'))
mem = man['members']
dino = np.stack([npz(W + f'/diagnostics/dino_{m["id"]}.npz') for m in mem]).astype(np.float64)
folds = np.array([m['fold'] for m in mem]); runs = np.array([m['source_run'] for m in mem])
print('dino stack shape', dino.shape, 'value range', dino.min().round(4), dino.max().round(4))

# ---- agent's recovery, re-implemented: per run, fold member with the max per-study log loss; majority over runs
def loss(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return -(Y * np.log(p) + (1 - Y) * np.log(1 - p)).mean(1)
A = []
for r in sorted(set(runs)):
    ix = np.where(runs == r)[0]
    L = np.stack([loss(dino[i]) for i in ix]); A.append(folds[ix][np.argmax(L, 0)])
A = np.stack(A)
agent_fold = np.array([Counter(A[:, s].tolist()).most_common(1)[0][0] for s in range(n)])
print('agent-method: all 4 runs agree on', int((A == A[0]).all(0).sum()), 'of', n)
saved = np.load(H + '/k2_test/analysis_wf/oof_dino_fold_majority.npy')[:, 0]
print('agent-method fold == agent saved majority file:', bool((saved == agent_fold).all()))

# ---- ground truth 1: DINO oof.npz (true out-of-fold predictions of the public training run)
z = np.load(INP + '/rsna-knee-weights/oof.npz', allow_pickle=False)
pos = {str(s): i for i, s in enumerate(z['ids'])}
ix = [pos[s] for s in ids]
oof = z['pred'][ix].astype(np.float64); gm = z['gold_mask'][ix]; yd = z['y_derived'][ix]
print('oof.npz: targets', list(z['targets']) == T, '| gold_mask on gold58', int(gm.sum()), '/', n,
      '| y_derived == train.csv gold labels', bool((yd == Y).all()), '| oof range', oof.min().round(4), oof.max().round(4))
print(f'TRUE OOF (oof.npz) gold58 macro AUC {mauc(Y, oof):.4f}')
# which fold's members reproduce oof.npz for each study?  try (a) 4-run prob mean, (b) each single run, (c) logit mean
D = {}
for k in range(5):
    sel = np.where(folds == k)[0]
    D[('mean4', k)] = np.abs(dino[sel].mean(0) - oof).mean(1)
    lg = np.log(np.clip(dino[sel], 1e-6, 1 - 1e-6) / (1 - np.clip(dino[sel], 1e-6, 1 - 1e-6))).mean(0)
    D[('logit4', k)] = np.abs(1 / (1 + np.exp(-lg)) - oof).mean(1)
    for r in sorted(set(runs)):
        i = np.where((folds == k) & (runs == r))[0][0]
        D[(r, k)] = np.abs(dino[i] - oof).mean(1)
for key in ['mean4', 'logit4'] + sorted(set(runs)):
    M = np.stack([D[(key, k)] for k in range(5)])            # (5, n)
    o = np.sort(M, 0)
    print(f'  match oof.npz with {key:20s}: median best dist {np.median(o[0]):.5f}, median 2nd {np.median(o[1]):.5f}, '
          f'max best {o[0].max():.5f}')
M = np.stack([D[('mean4', k)] for k in range(5)]); true_fold = np.argmin(M, 0)
ratio = np.sort(M, 0)[1] / np.maximum(np.sort(M, 0)[0], 1e-12)
print('truth fold (argmin dist to oof.npz, 4-run mean) counts', sorted(Counter(true_fold.tolist()).items()),
      '| 2nd/best dist ratio min', ratio.min().round(2), 'median', np.median(ratio).round(2))
print('AGENT DINO fold == truth fold on', int((agent_fold == true_fold).sum()), 'of', n,
      '| on the 54 unanimous studies:', int(((agent_fold == true_fold) & (A == A[0]).all(0)).sum()), 'of', int((A == A[0]).all(0).sum()))
bad = np.where(agent_fold != true_fold)[0]
print('mismatched studies (idx, agent, truth, run votes):', [(int(s), int(agent_fold[s]), int(true_fold[s]), A[:, s].tolist()) for s in bad])
# per-member AUC on its own (truth) held-out studies vs the rest: confirms in-sample vs OOS
for k in range(5):
    sel = np.where(folds == k)[0]; ho = true_fold == k
    if ho.sum() >= 5:
        def sub_auc(X, m):
            vals = [roc_auc_score(Y[m, j], X[m, j]) for j in range(12) if 0 < Y[m, j].sum() < m.sum()]
            return np.mean(vals)
        print(f'  fold {k}: n_heldout {int(ho.sum()):2d}  mean4 AUC on held-out {sub_auc(dino[sel].mean(0), ho):.3f}  on rest {sub_auc(dino[sel].mean(0), ~ho):.3f}')
dino_oof_agent = np.stack([dino[folds == agent_fold[s], s].mean(0) for s in range(n)])
dino_oof_true = np.stack([dino[folds == true_fold[s], s].mean(0) for s in range(n)])
print(f'DINO-oof AUC: agent folds {mauc(Y, dino_oof_agent):.4f} | truth folds {mauc(Y, dino_oof_true):.4f} | oof.npz {mauc(Y, oof):.4f}')

# ---- ground truth 2: Rad OOF csvs with a fold column
for name, f in [('v52_oof (e9/e10 family)', INP + '/rsna-knee-e9-radimagenet-heads-v15/v52_oof.csv'),
                ('v52_e11_oof (e11/e13 family)', INP + '/rsna-knee-e11-diverse-heads-v20/v52_e11_oof.csv')]:
    d = pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids]
    print(f'{name}: is_gold all {bool((d.is_gold == 1).all())}; fold counts {sorted(Counter(d.fold.tolist()).items())}; '
          f'oof AUC gold58 {mauc(Y, d[T].to_numpy(float)):.4f}')
    D[name] = d.fold.to_numpy()
print('rad v52 fold == v52_e11 fold on', int((D['v52_oof (e9/e10 family)'] == D['v52_e11_oof (e11/e13 family)']).sum()), 'of', n)
print('rad fold == DINO truth fold on', int((D['v52_oof (e9/e10 family)'] == true_fold).sum()), 'of', n)
rad = {k: npz(W + f'/diagnostics/{k}.npz').astype(np.float64) for k in ('rad_e10_raw', 'rad_e13_raw', 'rad_legacy_second_raw')}
Lrad = sum(np.stack([loss(h) for h in rad[k]]) for k in rad)
agent_rfold = np.argmax(Lrad, 0)
for name in ['v52_oof (e9/e10 family)', 'v52_e11_oof (e11/e13 family)']:
    rf = D[name]
    print(f'AGENT Rad head == {name} fold on', int((agent_rfold == rf).sum()), 'of', n)
    # head index <-> fold check: for each head h, AUC on studies with fold==h vs rest
    for k in ('rad_e10_raw', 'rad_e13_raw'):
        row = []
        for h in range(5):
            m = rf == h
            vals_ho = [roc_auc_score(Y[m, j], rad[k][h][m, j]) for j in range(12) if 0 < Y[m, j].sum() < m.sum()]
            vals_r = [roc_auc_score(Y[~m, j], rad[k][h][~m, j]) for j in range(12) if 0 < Y[~m, j].sum() < (~m).sum()]
            row.append(f'h{h}:{np.mean(vals_ho):.2f}/{np.mean(vals_r):.2f}')
        print(f'    {k} AUC on fold==h / rest:', ' '.join(row))
    # match oof csv to a head: which head is closest per study
    d = pd.read_csv(INP + ('/rsna-knee-e9-radimagenet-heads-v15/v52_oof.csv' if 'e10' in name else '/rsna-knee-e11-diverse-heads-v20/v52_e11_oof.csv'),
                    dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, T].to_numpy(float)
    for k in ('rad_e10_raw', 'rad_e13_raw', 'rad_legacy_second_raw'):
        dist = np.stack([np.abs(rad[k][h] - d).mean(1) for h in range(5)])
        print(f'    nearest {k} head to this oof csv == fold column on', int((np.argmin(dist, 0) == rf).sum()), 'of', n,
              f'(median best dist {np.median(dist.min(0)):.4f})')
np.savez('verify_folds.npz', agent_fold=agent_fold, true_fold=true_fold, agent_rfold=agent_rfold,
         rad_fold_v52=D['v52_oof (e9/e10 family)'], rad_fold_e11=D['v52_e11_oof (e11/e13 family)'])
print('wrote verify_folds.npz')
