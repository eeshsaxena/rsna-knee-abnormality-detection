# build_pl_tm5.py (jump2, 2026-10-05): round-2 pseudo-label TARGET from the 5-fold TTR mix50 OOF + gate T. CPU only.
# Pre-registered (pseudolabel.md sec 4), no per-label tuning:
#   S    = TTR mix50 seed-1043 OOF, raw best checkpoint per fold (~/fleet_runs/f{k}_s1043_ttr_mix50/oof_f{k}_s1043_raw.csv);
#          logits standardised WITHIN each fold (removes fold calibration offsets), pooled, then mapped onto
#          steven_v4's logit mean/std over the non-gold rows (build_gated 'logit_match' convention).
#   w    = 0.75 on report-UNK cells, 0.50 on report-NO, 0.50 on report-YES (verdict = report_labels_v2 __verdict, missing = UNK)
#   y    = (1 - w) * steven_v4 + w * S'          gold74 rows keep steven_v4 (they are --exclude'd from training anyway)
# Gate T (target level): gold58 AUC of y on the 58 gold rows (S on gold rows = mean z of the fold models' gold74 predictions,
#   ~/jump2/gold74_preds/f{k}_s1043_ttr_mix50__raw.csv from gold_infer.py) >= mix50 - 0.005, AND within-verdict Spearman vs
#   the Raptor proxy over the non-gold rows <= mix50's.
# Usage: python build_pl_tm5.py            -> needs all 5 folds; writes ~/jump2/labels_pl_tm5_v1.csv
#        python build_pl_tm5.py --dry      -> uses whatever folds exist, prints gate T on those rows, writes nothing
import os, sys, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from scipy.special import logit, expit

H = os.path.expanduser
DRY = '--dry' in sys.argv
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
W_UNK, W_NO, W_YES = 0.75, 0.50, 0.50
OUT = H('~/jump2/labels_pl_tm5_v1.csv')
rd = lambda p: pd.read_csv(H(p), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
lg = lambda a: logit(np.clip(np.asarray(a, float), 1e-4, 1 - 1e-4))
v4 = rd('~/rsna_train/llm_labels_v4_blend.csv')
mix50 = rd('~/rsna_train/labels_mix50_logit.csv')[T]
rap = rd('~/fleet/teacher_v2_raptorproxy_shard0.csv')[T]
folds = rd('~/rsna_train/folds5.csv')['fold']
gold = rd('~/rsna_train/gold74_eval_set.csv')
G58 = [s for s in gold.index if gold.loc[s, 'source'] == 'original_gold58']
r2 = rd('~/rsna_train/report_labels_v2.csv')
VER = pd.DataFrame({l: r2[l + '__verdict'] for l in T}).reindex(v4.index).fillna('UNK')
NONGOLD = folds.index[folds >= 0]
assert not set(NONGOLD) & set(gold.index)

parts, have = [], []
for k in range(5):
    p = H(f'~/fleet_runs/f{k}_s1043_ttr_mix50/oof_f{k}_s1043_raw.csv')
    if not os.path.exists(p):
        print(f'fold {k}: OOF missing ({p})'); continue
    o = rd(p)[T]
    assert set(o.index) == set(folds.index[folds == k]), f'fold {k} OOF rows != folds5 fold {k}'
    L = lg(o.values); mu, sd = L.mean(0), L.std(0)
    parts.append(pd.DataFrame((L - mu) / sd, index=o.index, columns=T)); have.append((k, mu, sd))
if len(have) < 5 and not DRY:
    sys.exit(f'only folds {[k for k, _, _ in have]} present: rerun when all 5 OOFs exist, or use --dry')
Z = pd.concat(parts); assert not Z.index.duplicated().any()
rows = list(Z.index)
LV = lg(v4.loc[NONGOLD, T].values); MU_V, SD_V = LV.mean(0), LV.std(0)
zpool = Z.values; zpool = (zpool - zpool.mean(0)) / zpool.std(0)
S = pd.DataFrame(expit(zpool * SD_V + MU_V), index=rows, columns=T)


def wmat(idx):
    V = VER.loc[idx, T].values
    return np.where(V == 'UNK', W_UNK, np.where(V == 'NO', W_NO, W_YES)).astype(float)


def target(idx, Sx):
    w = wmat(idx); b = v4.loc[idx, T].values
    return pd.DataFrame((1 - w) * b + w * Sx.loc[idx, T].values, index=idx, columns=T)


Y = target(rows, S)
# gold rows: mean z over the available fold models' gold predictions (each z-scored with that fold's OOF stats)
zg = []
for k, mu, sd in have:
    gpth = H(f'~/jump2/gold74_preds/f{k}_s1043_ttr_mix50__raw.csv')
    if os.path.exists(gpth):
        zg.append((lg(rd(gpth).loc[G58, T].values) - mu) / sd)
    else:
        print(f'gold preds for fold {k} missing ({gpth}): run gold_infer.py on its raw best checkpoint')
assert zg, 'no gold predictions at all'
NGM = len(zg); zg = np.mean(zg, 0)
ref = Z.values  # pooled-z normalisation constants from the train rows
Sg = pd.DataFrame(expit((zg - ref.mean(0)) / ref.std(0) * SD_V + MU_V), index=G58, columns=T)
Yg = target(G58, Sg)
Yt = gold.loc[G58, T].astype(float).values


def fauc(y, s):
    r = rankdata(s); y = y > .5; n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def g58(X): return float(np.mean([fauc(Yt[:, j], np.asarray(X)[:, j]) for j in range(12)]))


def ws(X, idx):
    Vv = VER.loc[idx]; R = rap.loc[idx, T].values; Xv = np.asarray(X); num = den = 0.0
    for j, l in enumerate(T):
        for v in ['YES', 'UNK', 'NO']:
            m = (Vv[l] == v).values
            if m.sum() >= 30 and np.unique(Xv[m, j]).size > 3:
                r = spearmanr(Xv[m, j], R[m, j]).correlation
                if np.isfinite(r): num += r * m.sum(); den += m.sum()
    return num / den


def rho(X, idx):
    R = rap.loc[idx, T].values; Xv = np.asarray(X)
    return float(np.mean([spearmanr(Xv[:, j], R[:, j]).correlation for j in range(12)]))


g_t, g_m, g_v = g58(Yg.values), g58(mix50.loc[G58].values), g58(v4.loc[G58, T].values)
w_t, w_m = ws(Y.values, rows), ws(mix50.loc[rows].values, rows)
r_t, r_m = rho(Y.values, rows), rho(mix50.loc[rows].values, rows)
print(f'folds used {[k for k, _, _ in have]} | train rows {len(rows)} | fold models with gold preds {NGM}')
print('NOTE: a fold whose run is still training has a PARTIAL OOF (best epoch so far); judge gate T only on finished folds')
print(f'gold58 target AUC: pl_tm5 {g_t:.4f} | mix50 {g_m:.4f} | v4 {g_v:.4f}')
print(f'within-verdict Spearman vs Raptor proxy (train rows): pl_tm5 {w_t:.4f} | mix50 {w_m:.4f}')
print(f'mean per-label Spearman vs Raptor proxy (train rows): pl_tm5 {r_t:.4f} | mix50 {r_m:.4f}')
ok = (g_t >= g_m - 0.005) and (w_t <= w_m)
print('GATE T:', 'PASS' if ok else 'FAIL', f'(needs gold58 >= {g_m - 0.005:.4f} and ws <= {w_m:.4f})')
if DRY:
    print('--dry: nothing written'); sys.exit(0)
outdf = v4[T].copy()
outdf.loc[rows, T] = Y.values
assert outdf.notna().all().all() and ((outdf.values >= 0) & (outdf.values <= 1)).all()
outdf.reset_index().to_csv(OUT, index=False)
print('wrote', OUT, outdf.shape, '| changed rows', len(rows), '| gold74 rows = steven_v4 (excluded from training)')
