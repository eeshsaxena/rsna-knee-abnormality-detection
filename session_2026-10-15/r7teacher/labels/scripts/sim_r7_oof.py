# sim_r7_oof.py (r7teacher/labels, 2026-10-14; A1 version): writes a SIMULATED R7-OOF teacher file in the exact format the
# label builder expects from the real R7 out-of-fold run, so the whole chain can be dry-run before the real file exists.
# Source of the simulated values: the K14 fleet member set DT (TTR224 frac75 + OAI T123v00, s1043): its 5-fold OOF on
# the non-gold pool rows and the 5-fold mean of its gold74 pass-A predictions on the gold rows. DT is NOT one of the
# OAI-free sets that form the real fleet_oof component, so the simulated T_b has three distinct sources like the real one.
# The file is tagged teacher_tag = SIM:... on every row; the builder names its outputs SIM_* and the queue gate refuses
# any SIM table. A few pool rows are dropped on purpose (--drop) to exercise the builder's missing-row path.
#
# PREREG_TEACHER Amendment A1 FX6 (realistic logit magnitudes): with --absz_ref (r7_absz_ref44.json = per-label sorted |z| of
#   the 44 real R7 hash-fold rows, make_absz_ref44.py), the simulated logits are DT's logits with DT's order kept and their
#   magnitudes quantile-mapped per label onto the real |z| distribution:
#       l = logit(DT prob); F = average-rank CDF of |l| over all rows; |z| = interp(F, [0, (i + .5) / 44 ..., 1], [0, ref..., 1.1 max])
#       z = sign(l) |z|   (strictly monotone in l, so every rank-based number of the pre-A1 dry run is unchanged)
#   and the file carries z|<label> (float32, %.9g) as the real one does. p| = float32 sigmoid(z) for the 10 non-stacked labels;
#   for Effusion / Baker's p| imitates the R7 stackers (a different function of more features): sigmoid(.8 z + N(0, s)) with
#   --stacker_sd s / --stacker_seed, so a builder that still read p| for those labels would change with the seed (A1 FX2 test).
#   The pre-A1 DT stand-in never reached the 1e-4 clip; this one ties ~55 % of the cells there, like the real rows.
# L0 negative controls (A1 FX3): --leak_folds K [K ...] --leak L adds L x (2 v4 - 1) to the logits of the non-gold pool rows
#   whose R7 hash fold is in K (what a fold model that trained on those studies' report-derived targets would look like).
#
# TEACHER FILE FORMAT = job 1's merged r7_oof.csv (session_2026-10-15/r7teacher/merge_sessions.py): StudyInstanceUID,
#   r7_fold (int = int(sha1(uid)[:8], 16) % 4, the hash fold whose model scored the study = the fold that did NOT train on it;
#   gold studies carry their hash fold too: gold58 was never in R7's pool, so any fold model is out-of-fold there), shard, job,
#   precision ('full' when logits were dumped), p|<label> x 12 (per-study PROBABILITIES, the R7 bundle's final output),
#   z|<label> x 12 raw fold logits and fluid / stk / state extras (ignored by the builder). The builder also accepts plain
#   <label> columns (pre-A1 SIM only). This simulation writes r7_fold, precision, p|<label>, z|<label> (with --absz_ref) and
#   teacher_tag ('SIM:...'; the real file has no tag column).
import argparse, hashlib, json, os
import numpy as np, pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata

H = os.path.expanduser('~')
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
STACKED = ('Effusion', "Baker's")
ap = argparse.ArgumentParser()
ap.add_argument('--oof', required=True)          # collect_fleet_oof.py oof_<set>.csv (pool rows)
ap.add_argument('--gold_mean', required=True)    # collect_fleet_oof.py gold_<name>_mean.csv (gold74 rows)
ap.add_argument('--out', required=True)
ap.add_argument('--tag', default='SIM:K14-DT')
ap.add_argument('--drop', type=int, default=3)
ap.add_argument('--absz_ref', default='')        # A1 FX6: r7_absz_ref44.json -> realistic z| columns
ap.add_argument('--stacker_sd', type=float, default=1.0)
ap.add_argument('--stacker_seed', type=int, default=11)
ap.add_argument('--leak_folds', type=int, nargs='*', default=[])
ap.add_argument('--leak', type=float, default=0.0)
ap.add_argument('--v4', default=f'{H}/rsna_train/llm_labels_v4_blend.csv')
A = ap.parse_args()
assert A.tag.startswith('SIM:'), 'a simulated teacher must carry a SIM: tag'
assert not A.leak_folds or (A.absz_ref and A.leak > 0), '--leak_folds needs --absz_ref and --leak > 0'
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
o = rd(A.oof)[T]; g = rd(A.gold_mean)[T]
assert not set(o.index) & set(g.index)
rng = np.random.default_rng(7)
drop = set(rng.choice(np.array(sorted(o.index)), A.drop, replace=False)) if A.drop else set()
o = o.drop(index=sorted(drop))
hf = lambda u: int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4
d = pd.concat([o, g])
fold = np.array([hf(u) for u in d.index])
info = ''
if A.absz_ref:
    ref = json.load(open(A.absz_ref)); assert ref['n'] == 44, ref['n']
    Z = np.zeros((len(d), 12))
    for j, c in enumerate(T):
        l = logit(np.clip(d[c].to_numpy(np.float64), 1e-6, 1 - 1e-6))
        F = (rankdata(np.abs(l)) - 0.5) / len(l)
        r = np.asarray(ref['abs_z'][c], np.float64); assert len(r) == 44 and (np.diff(r) > 0).all()
        xs = np.concatenate([[0.0], (np.arange(44) + 0.5) / 44, [1.0]]); ys = np.concatenate([[0.0], r, [1.1 * r[-1]]])
        Z[:, j] = np.sign(l) * np.interp(F, xs, ys)
    if A.leak_folds:
        v4 = rd(A.v4)[T].reindex(d.index)
        lk = np.isin(fold, A.leak_folds) & ~d.index.isin(g.index)
        Z[lk] += A.leak * (2 * v4.to_numpy(np.float64)[lk] - 1)
        info += f' | LEAK +{A.leak} x (2 v4 - 1) on {int(lk.sum())} pool rows of hash folds {A.leak_folds}'
    Z32 = Z.astype(np.float32)
    P = expit(Z32.astype(np.float64)).astype(np.float32)
    srng = np.random.default_rng(A.stacker_seed)
    for c in STACKED:                                          # stacker-like p| (not a function of z| alone)
        j = T.index(c)
        P[:, j] = expit(0.8 * Z32[:, j].astype(np.float64) + srng.normal(0, A.stacker_sd, len(d))).astype(np.float32)
    out = pd.DataFrame(index=d.index)
    out['r7_fold'] = fold; out['precision'] = 'full'
    for j, c in enumerate(T):
        out['p|' + c] = ['%.9g' % x for x in P[:, j]]
    for j, c in enumerate(T):
        out['z|' + c] = ['%.9g' % x for x in Z32[:, j]]
    tie = float((np.abs(Z32[~d.index.isin(g.index)]) > np.log((1 - 1e-4) / 1e-4)).mean())
    info += f' | z| realistic (A1 FX6): pool cells beyond the 1e-4 clip {tie:.3f} (44 real rows {ref["clip_tie_share_all"]:.3f}); stacker-like p| on {list(STACKED)} sd {A.stacker_sd} seed {A.stacker_seed}'
else:
    out = d.rename(columns={c: 'p|' + c for c in T}); out.insert(0, 'precision', 'full'); out.insert(0, 'r7_fold', fold)
    out = out[['r7_fold', 'precision'] + ['p|' + c for c in T]]
out['teacher_tag'] = A.tag
out.index.name = 'StudyInstanceUID'
out.to_csv(A.out)
print(f'wrote {A.out}: {len(out)} rows ({len(o)} pool rows, {len(g)} gold rows, {len(drop)} pool rows dropped on purpose) '
      f'| hash-fold counts {np.bincount(fold, minlength=4).tolist()} | tag {A.tag}{info}')
