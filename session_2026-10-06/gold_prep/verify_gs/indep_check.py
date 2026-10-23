#!/usr/bin/env python3
# indep_check.py (2026-10-06, Claude, adversarial verify of gold_prep/gates). CPU ONLY, read only on runs.
# Independent re-implementation (does NOT import score_run): own Mann-Whitney AUC, own val-set rebuild from the
# trainer's code path, scipy Spearman, own bootstrap. Prints numbers to compare with score_run / blend_plan outputs.
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '2')
import glob, json, re, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

H = os.path.expanduser('~'); FR = f'{H}/fleet_runs'
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
v4 = rd(f'{H}/rsna_train/llm_labels_v4_blend.csv')
folds = rd(f'{H}/rsna_train/folds5.csv')['fold']
g74 = rd(f'{H}/rsna_train/gold74_eval_set.csv')
g58 = set(pd.read_csv(f'{H}/kmimic/ids/gold58.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID)
px = rd(f'{H}/gold_prep/gates/raptor_proxy_v5v8.csv')
store = {os.path.basename(p)[:-4] for p in glob.glob(f'{H}/fleet_store/*.npz')}
print(f'v4 {v4.shape} dup {v4.index.duplicated().sum()} | folds {len(folds)} dup {folds.index.duplicated().sum()} '
      f'| g74 {len(g74)} | g58 {len(g58)} g58<=g74 {g58 <= set(g74.index)} | proxy {px.shape} dup {px.index.duplicated().sum()} '
      f'NaN {int(px[T].isna().sum().sum())} | store {len(store)}')
print(f'gold74 in folds: {len(set(g74.index) & set(folds.index))}, their folds: '
      f'{folds[folds.index.isin(g74.index)].value_counts().to_dict()}')
print(f'proxy overlaps gold74: {len(set(px.index) & set(g74.index))}')


def mw_auc(y, s):
    r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro(P, ids, tab, f32=True):
    Y = tab.loc[ids, T].to_numpy(np.float64); a = []
    for j in range(12):
        ok = np.isfinite(Y[:, j]) & np.isfinite(P[:, j]); y = Y[ok, j] >= .5
        if ok.sum() and 0 < y.mean() < 1:
            a.append(mw_auc(y, P[ok, j]))
    return float(np.mean(a)), len(a)


def closing(log):
    ln = [l.strip() for l in open(log, errors='replace') if l.strip()]
    try:
        d = json.loads(ln[-1]); return d
    except Exception:
        return None


labsets = {}
def labs(name):
    if name not in labsets:
        labsets[name] = rd(f'{H}/rsna_train/{name}')
    return labsets[name]


RUNS = sys.argv[1:] or ['f0_s1043_ttr_mix50', 'f0_s42', 'f0_s42_gated', 'f0_s1042_ttr', 'f0_s142_f2v4', 'f0_s42_v4',
                        'f1_s1043_ttr_mix50', 'f2_s1043_ttr_mix50', 'f3_s1043_ttr_mix50', 'f4_s1043_ttr_mix50',
                        'f1_s1042_ttr', 'f2_s1042_ttr', 'f3_s1042_ttr']
TRAIN_LAB = {'f0_s42_gated': 'labels_gated75_logit.csv', 'f0_s1042_ttr': 'llm_labels_v4_blend.csv',
             'f0_s142_f2v4': 'llm_labels_v4_blend.csv', 'f0_s42_v4': 'llm_labels_v4_blend.csv'}
best = {}
for run in RUNS:
    d = f'{FR}/{run}'; c = closing(d + '.log')
    if not c or 'raw' not in c:
        print(f'{run}: no closing line'); continue
    m = re.match(r'f(\d+)_s(\d+)', run); F, S = int(m.group(1)), int(m.group(2))
    res = {}
    for tag in ('raw', 'ema'):
        p = f'{d}/oof_f{F}_s{S}_{tag}.csv'
        D64 = rd(p)[T]
        a64 = macro(D64.to_numpy(np.float64), list(D64.index), v4)[0]
        a32 = macro(D64.to_numpy(np.float32).astype(np.float64), list(D64.index), v4)[0]
        sk = np.mean([roc_auc_score(v4.loc[D64.index, c_] >= .5, D64[c_].astype(np.float32)) for c_ in T])
        res[tag] = (a64, a32, sk, D64)
    # val ids per the trainer: ids = lab.index & folds & store, va = not held & fold == F (ORDER = lab order)
    tl = TRAIN_LAB.get(run, 'labels_mix50_logit.csv'); lab = labs(tl)
    held = set(folds.index[folds < 0]) | set(g74.index)
    ids = [s for s in lab.index if s in folds.index and s in store]
    va = [s for s in ids if s not in held and folds[s] == F]
    oof_ids = list(res['raw'][3].index)
    bt = 'raw' if c['raw'] >= c['ema'] else 'ema'
    print(f'{run:22s} best {bt} | closing raw {c["raw"]:.12f} ema {c["ema"]:.12f}')
    for tag in ('raw', 'ema'):
        a64, a32, sk, _ = res[tag]
        print(f'   {tag}: MW f64 diff {a64 - c[tag]:+.2e} | MW f32 diff {a32 - c[tag]:+.2e} | sklearn f32 diff {sk - c[tag]:+.2e}')
    print(f'   val ids == trainer rebuild (ordered): {oof_ids == va} | set equal {set(oof_ids) == set(va)} '
          f'(n {len(oof_ids)} vs {len(va)}) | gold74 in OOF {len(set(oof_ids) & set(g74.index))} | gold58 in OOF '
          f'{len(set(oof_ids) & g58)} | raw/ema index equal {list(res["ema"][3].index) == oof_ids}')
    best[run] = res[bt][3].astype(np.float32)
    if F == 0 or True:
        ids_p = [s for s in oof_ids if s in px.index]
        rho = [spearmanr(best[run].loc[ids_p, c_], px.loc[ids_p, c_]).correlation for c_ in T]
        print(f'   scipy Spearman vs proxy {np.mean(rho):.4f} (n {len(ids_p)})')

J1 = 'f0_s1043_ttr_mix50'
if J1 in best:
    for run in best:
        if run == J1 or not run.startswith('f0_'):
            continue
        rho = [spearmanr(best[run].loc[best[J1].index, c_], best[J1][c_]).correlation for c_ in T]
        print(f'Spearman {run} vs J1: {np.mean(rho):.4f}')
    # independent 2-member blends + bootstrap (1000, seed 12345)
    ids = list(best[J1].index); Y = v4.loc[ids, T].to_numpy(np.float64)
    print(f'v4 NaN cells on fold-0 val: {int(np.isnan(Y).sum())}')
    Yb = Y >= .5
    R = {k: np.column_stack([rankdata(v.loc[ids, c_]) / len(ids) for c_ in T]) for k, v in best.items() if k.startswith('f0_')}
    def mac(M, ix=None):
        ix = np.arange(len(ids)) if ix is None else ix
        out = []
        for j in range(12):
            y = Yb[ix, j]
            if 0 < y.mean() < 1:
                out.append(mw_auc(y, M[ix, j]))
        return np.mean(out)
    rng = np.random.default_rng(12345); BI = [rng.integers(0, len(ids), len(ids)) for _ in range(1000)]
    base = mac(R[J1])
    print(f'J1 solo (rank) {base:.6f}')
    for k in R:
        if k == J1:
            continue
        M = (R[J1] + R[k]) / 2; bl = mac(M)
        d = np.array([mac(M, ix) - mac(R[J1], ix) for ix in BI])
        print(f'  J1+{k:22s} blend {bl:.4f} d {bl - base:+.4f} SE {d.std(ddof=1):.4f} P>0 {(d > 0).mean():.3f} '
              f'95% [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]')
