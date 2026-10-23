# build_teacher_labels.py (r7teacher/labels, 2026-10-14): builds the PRE-REGISTERED student label tables of
# PREREG_TEACHER.md from (v4 report labels, our OAI-free fleet OOF, the R7 out-of-fold teacher). CPU, seconds.
# Read only on its inputs; writes only --out_dir.
#
#   T_a = .5 v4 + .5 M(R7)                      (mix50 structure with the R7 OOF as the image teacher)
#   T_b = .4 v4 + .3 M(fleet) + .3 M(R7)        (PRIMARY)
# M(x) = per-label logit match of x onto v4 (== build_frac75.py / build_mixSA.py logit_match: z-score of logit(clip(x,
# 1e-4)) rescaled to v4's logit mean and std), with the statistics taken on the non-gold pool rows where x exists.
# Mixing is a convex combination in PROBABILITY space (== mix50 / frac75, MEASURED reproduction to 1e-12 on 10-07).
# Pre-registered fallback: if the teacher file looks like per-label ranks (max KS distance to Uniform(0,1) over the
# 12 labels < .02), M(R7) is an ECDF quantile map onto v4's pool marginal instead of the logit match.
# A source missing for a study is dropped for that study and the remaining weights renormalised; the teacher may miss
# at most 1 % of the pool rows (else exit 2: the R7 run is incomplete).
# Outputs (names prefixed SIM_ for a simulated teacher): labels_r7t_a.csv, labels_r7t_b.csv (header + rows in v4's order,
# LF, repr floats), teacher_labels_receipt.json (sha256 / md5 of every input and output), teacher_labels_diag.txt.
#
# usage (box): python build_teacher_labels.py --teacher R7OOF.csv --teacher_tag R7OOF --fleet_oof oof_fleetOF_mean.csv
#              --fleet_gold gold_fleetOF_mean.csv --out_dir DIR [--prereg PREREG_TEACHER.md --prereg_sha PREREG_TEACHER.sha256]
import argparse, hashlib, json, os, sys, time
import numpy as np, pandas as pd
from scipy.special import logit, expit
from scipy.stats import spearmanr, kstest

H = os.path.expanduser('~')
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
TABLES = {'a': {'v4': .5, 'r7': .5}, 'b': {'v4': .4, 'fleet': .3, 'r7': .3}}     # PREREG_TEACHER sec 3 (frozen)
PRIMARY = 'b'
CLIP = 1e-4; MISSING_MAX = 0.01; RANK_KS = 0.02
ap = argparse.ArgumentParser()
ap.add_argument('--v4', default=f'{H}/rsna_train/llm_labels_v4_blend.csv')
ap.add_argument('--frac75', default=f'{H}/novel/frac75/labels_frac75_logit.csv')
ap.add_argument('--folds', default=f'{H}/rsna_train/folds5.csv')
ap.add_argument('--gold', default=f'{H}/rsna_train/gold74_eval_set.csv')
ap.add_argument('--train_csv', default=f'{H}/rsna_train/train.csv')
ap.add_argument('--fleet_oof', required=True); ap.add_argument('--fleet_gold', default='')
ap.add_argument('--teacher', required=True); ap.add_argument('--teacher_tag', required=True)
ap.add_argument('--out_dir', required=True); ap.add_argument('--store', default=f'{H}/fleet_store')
ap.add_argument('--prereg', default=''); ap.add_argument('--prereg_sha', default='')
A = ap.parse_args()
REAL = not A.teacher_tag.startswith('SIM:')
PFX = '' if REAL else 'SIM_'
os.makedirs(A.out_dir, exist_ok=True)
L = []


def say(s=''):
    print(s, flush=True); L.append(s)


def digest(p, algo):
    h = hashlib.new(algo)
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
say(f'build_teacher_labels {time.strftime("%Y-%m-%d %H:%M:%S")} | teacher_tag {A.teacher_tag} | {"REAL" if REAL else "SIMULATION"}')
rec = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), teacher_tag=A.teacher_tag, real=REAL, tables=TABLES,
           primary=PRIMARY, clip=CLIP, inputs={})
for k in ('v4', 'frac75', 'folds', 'gold', 'train_csv', 'fleet_oof', 'fleet_gold', 'teacher'):
    p = getattr(A, k)
    if p:
        rec['inputs'][k] = dict(path=p, sha256=digest(p, 'sha256'), md5=digest(p, 'md5'))
        say(f'input {k:10s} md5 {rec["inputs"][k]["md5"]}  {p}')
# ---- 0. pre-registration guard (REAL teacher only): the PREREG file must exist and match its recorded sha256
if REAL:
    assert A.prereg and A.prereg_sha, 'a REAL teacher build needs --prereg and --prereg_sha (PREREG_TEACHER sec 7)'
    want = open(A.prereg_sha).read().split()[0]; got = digest(A.prereg, 'sha256')
    assert want == got, f'PREREG sha256 mismatch: file {got} != recorded {want}'
    rec['prereg_sha256'] = got; say(f'PREREG_TEACHER sha256 {got[:16]}.. OK')
    assert A.fleet_gold, 'a REAL build needs --fleet_gold (the label gate scores T_b on gold58 with the fleet component)'

# ---- 1. base tables
v4 = rd(A.v4); assert list(v4.columns) == T and len(v4) == 4407 and v4.index.is_unique and not v4.isna().any().any()
ids = v4.index
folds = rd(A.folds)['fold'].reindex(ids); assert folds.notna().all()
gold = rd(A.gold); assert len(gold) == 74
g58 = [s for s in gold.index if gold.loc[s, 'source'] == 'original_gold58']; assert len(g58) == 58
gset = set(gold.index)
pool = [s for s in ids if folds[s] >= 0 and s not in gset]
say(f'v4 {len(v4)} rows | gold74 {len(gold)} (gold58 {len(g58)}) | non-gold pool {len(pool)} (the rows a student trains / validates on)')
assert not any('oai' in s.lower() for s in ids), 'OAI ids in v4'
# official labels: train.csv labelled rows must be exactly the gold58 rows with identical values (the gate's expert ruler)
tr = rd(A.train_csv)
lab = tr[T].dropna(how='all')
same_ids = sorted(lab.index) == sorted(g58)
same_vals = same_ids and np.allclose(lab.loc[g58, T].values.astype(float), gold.loc[g58, T].values.astype(float), equal_nan=True)
say(f'official train.csv labelled rows {len(lab)}: ids == gold74 original_gold58 {same_ids}; values identical {same_vals}')
rec['official_eq_gold58'] = bool(same_vals)

# ---- 2. sources
te = rd(A.teacher)
# job-1 format (merge_sessions.py r7_oof.csv): probabilities in 'p|<label>' columns (+ 'z|' logits, fluid / state extras,
# 'precision'); every study, gold included, is scored by its hash-fold model (gold58 was never in R7's pool, so any fold is
# out-of-fold there). Plain '<label>' columns (sim_r7_oof.py) are accepted too.
if not all(c in te.columns for c in T) and all('p|' + c in te.columns for c in T):
    te = te.rename(columns={'p|' + c: c for c in T})
    say('teacher file in job-1 layout: using the p|<label> probability columns')
if 'n_models' not in te.columns:
    te['n_models'] = 1
if 'precision' in te.columns:
    say(f'teacher precision column: {te["precision"].value_counts().to_dict()}')
if 'teacher_tag' in te.columns:
    tags = set(te['teacher_tag'].astype(str))
    assert tags == {A.teacher_tag}, f'teacher file tags {tags} != --teacher_tag {A.teacher_tag}'
assert te.index.is_unique, 'duplicate study ids in the teacher file'
assert set(te.index) <= set(ids), f'{len(set(te.index) - set(ids))} teacher ids not in v4'
assert all(c in te.columns for c in T), 'teacher file lacks label columns'
tv = te[T].astype(float)
assert np.isfinite(tv.values).all() and (tv.values >= 0).all() and (tv.values <= 1).all(), 'teacher values must be probabilities in [0, 1]'
miss = [s for s in pool if s not in te.index]
say(f'teacher rows {len(te)} | pool rows missing {len(miss)} ({len(miss) / len(pool):.2%}, max {MISSING_MAX:.0%}) | gold74 rows present '
    f'{sum(s in te.index for s in gold.index)}/74 (gold58 {sum(s in te.index for s in g58)}/58)')
rec['teacher_rows'] = len(te); rec['teacher_missing_pool'] = len(miss)
if len(miss) > MISSING_MAX * len(pool):
    say(f'STOP: the teacher misses {len(miss)} pool rows (> {MISSING_MAX:.0%}): the R7 run is incomplete'); sys.exit(2)
if 'r7_fold' in te.columns:
    hf = lambda u: int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4
    one = te.index[te['n_models'].astype(int) == 1]
    bad = [u for u in one if int(te.loc[u, 'r7_fold']) != hf(u)]
    g58_single = [u for u in one if u in set(g58)]           # gold58 was never in R7's pool (gold_in_training 0): any single
    say(f'R7 hash-fold cross-check: {len(one)} single-model rows, {len(bad)} with r7_fold != sha1(uid)[:8] % 4; '   # fold model is
        f'gold58 rows single-model {len(g58_single)} / 4-fold mean {sum(1 for u in g58 if u in te.index) - len(g58_single)}')  # out-of-fold
    assert not bad, 'teacher fold assignment is not the R7 hash split'
    rec['hash_fold_check'] = dict(single=len(one), mismatches=0)
fl = rd(A.fleet_oof)[T]
assert sorted(fl.index) == sorted(pool), 'fleet OOF rows != the non-gold pool'
assert np.isfinite(fl.values).all() and (fl.values >= 0).all() and (fl.values <= 1).all()
flg = rd(A.fleet_gold)[T].reindex(gold.index) if A.fleet_gold else None
fleet = pd.concat([fl, flg]) if flg is not None else fl
fr = rd(A.frac75)[T].reindex(ids)

# ---- 3. teacher map: logit match (default) or ECDF (pre-registered fallback for rank-like files)
ks = [kstest(tv.loc[[s for s in pool if s in tv.index], c].values, 'uniform').statistic for c in T]
MAP = 'ecdf' if max(ks) < RANK_KS else 'lm'
say(f'teacher marginal vs Uniform(0,1): KS min {min(ks):.3f} max {max(ks):.3f} -> map = {MAP} '
    f'({"rank-like file: ECDF quantile map" if MAP == "ecdf" else "probabilities: logit match onto v4"})')
rec['map'] = MAP; rec['ks_uniform'] = dict(zip(T, [round(float(k), 4) for k in ks]))


def match(src, name):
    """per-label map of src (DataFrame over some ids) onto v4, statistics from the pool rows present in src."""
    out = pd.DataFrame(index=src.index, columns=T, dtype=float); st = {}
    pr = [s for s in pool if s in src.index]
    for c in T:
        a = src[c].values.astype(float); ap_ = src.loc[pr, c].values.astype(float); r = v4.loc[pr, c].values.astype(float)
        if MAP == 'lm' or name == 'fleet':
            la = logit(np.clip(a, CLIP, 1 - CLIP)); lap = logit(np.clip(ap_, CLIP, 1 - CLIP)); lr = logit(np.clip(r, CLIP, 1 - CLIP))
            out[c] = expit((la - lap.mean()) / lap.std() * lr.std() + lr.mean())
            st[c] = dict(src_mu=round(float(lap.mean()), 4), src_sd=round(float(lap.std()), 4), v4_mu=round(float(lr.mean()), 4),
                         v4_sd=round(float(lr.std()), 4))
        else:                                                    # ECDF quantile map onto v4's pool marginal
            srt = np.sort(ap_); q = np.searchsorted(srt, a, side='right') / len(srt)
            out[c] = np.quantile(r, np.clip(q, 0, 1))
            st[c] = dict(ecdf=True)
    return out, st


r7m, st_r7 = match(tv, 'r7'); flm, st_fl = match(fleet, 'fleet')
rec['map_stats'] = dict(r7=st_r7, fleet=st_fl)
src = {'v4': v4[T].astype(float), 'fleet': flm, 'r7': r7m}

# ---- 4. tables
outs = {}
for tn, W in TABLES.items():
    num = pd.DataFrame(0.0, index=ids, columns=T); den = pd.Series(0.0, index=ids)
    for k, w in W.items():
        have = src[k].index.intersection(ids)
        num.loc[have] += w * src[k].loc[have].values; den.loc[have] += w
    tab = num.div(den, axis=0)
    renorm = int((den < sum(W.values()) - 1e-12).sum()); renorm_pool = int((den.loc[pool] < sum(W.values()) - 1e-12).sum())
    assert np.isfinite(tab.values).all() and (tab.values > 0).all() and (tab.values < 1).all(), tn
    p = f'{A.out_dir}/{PFX}labels_r7t_{tn}.csv'
    with open(p, 'w', newline='\n') as fh:
        fh.write(','.join(['StudyInstanceUID'] + T) + '\n')
        for s, row in zip(ids, tab.values):
            fh.write(s + ',' + ','.join(repr(float(x)) for x in row) + '\n')
    back = rd(p)                                            # trainer-style read (pandas C parser, string ids)
    assert list(back.columns) == T and (back.index == ids).all() and not back.isna().any().any()
    assert float(np.max(np.abs(back.values - tab.values))) < 1e-12
    outs[tn] = back
    rec[f'table_{tn}'] = dict(path=p, sha256=digest(p, 'sha256'), md5=digest(p, 'md5'), weights=W, rows_renormalised=renorm,
                              pool_rows_renormalised=renorm_pool, primary=(tn == PRIMARY))
    say(f'WROTE {p} | md5 {rec[f"table_{tn}"]["md5"]} | weights {W} | rows renormalised {renorm} (pool {renorm_pool})')

# ---- 5. trainer compatibility (fleet_train.py reads --labels with this exact logic)
if os.path.isdir(A.store):
    for tn, back in outs.items():
        have = {s for s in back.index if os.path.exists(f'{A.store}/{s}.npz')}
        idt = [s for s in back.index if s in folds.index and s in have]
        n_want = sum(1 for s in back.index if folds[s] >= 0 and s not in gset)
        n_got = sum(1 for s in idt if folds[s] >= 0 and s not in gset)
        say(f'trainer check {tn}: store covers {n_got}/{n_want} usable studies (min_cover .99: {"OK" if n_got >= .99 * n_want else "FAIL"})')
        assert n_got >= .99 * n_want

# ---- 6. diagnostics (REPORT ONLY; PREREG_TEACHER sec 5 leg L1 is the gold58 target AUC)
from sklearn.metrics import roc_auc_score


def auc_masked(P, idx):
    Y = gold.loc[idx, T].values.astype(float); X = P.loc[idx, T].values.astype(float); a = []
    for j in range(12):
        ok = np.isfinite(Y[:, j]) & np.isfinite(X[:, j]); yb = Y[ok, j] >= .5
        if ok.sum() and 0 < yb.mean() < 1:
            a.append(roc_auc_score(yb, X[ok, j]))
    return float(np.mean(a)), len(a)


diag = {}
cands = {'v4': v4, 'frac75': fr, 'R7_teacher': tv, 'T_a': outs['a'], 'T_b': outs['b']}
if flg is not None:
    cands['fleet'] = flg
for nm, idx in (('gold58', g58), ('gold74', list(gold.index))):
    row = {}
    for k, P in cands.items():
        ii = [s for s in idx if s in P.index]
        if len(ii) == len(idx):
            row[k] = round(auc_masked(P, ii)[0], 4)
    diag[f'target_auc_{nm}'] = row
    say(f'L1 target macro AUC vs expert labels, {nm} (gold rows: R7 = its out-of-fold hash-fold model, fleet = 25-member mean, a best case for the fleet): '
        + ' | '.join(f'{k} {v:.4f}' for k, v in row.items()))
pr_ = [s for s in pool if s in tv.index]
Yv = (v4.loc[pr_, T].values >= .5)
diag['r7_vs_v4_auc_pool'] = round(float(np.mean([roc_auc_score(Yv[:, j], tv.loc[pr_, T].values[:, j]) for j in range(12)])), 4)
diag['fleet_vs_v4_auc_pool'] = round(float(np.mean([roc_auc_score(Yv[:, j], fl.loc[pr_, T].values[:, j]) for j in range(12)])), 4)
say(f'teacher OOF macro AUC vs v4 (>= .5) on pool rows: R7 {diag["r7_vs_v4_auc_pool"]:.4f} | fleet {diag["fleet_vs_v4_auc_pool"]:.4f}')
sp = lambda X, Y: float(np.mean([spearmanr(X[:, j], Y[:, j]).correlation for j in range(12)]))
P_ = {k: P.loc[pr_, T].values.astype(float) for k, P in (('v4', v4), ('frac75', fr), ('fleet', fl), ('R7', tv), ('T_a', outs['a']), ('T_b', outs['b']))}
pairs = [('R7', 'fleet'), ('R7', 'v4'), ('fleet', 'v4'), ('T_a', 'frac75'), ('T_b', 'frac75'), ('T_a', 'T_b'), ('T_a', 'R7'), ('T_b', 'R7'), ('T_b', 'fleet')]
diag['spearman_pool'] = {f'{a}~{b}': round(sp(P_[a], P_[b]), 4) for a, b in pairs}
say('mean per-label Spearman on pool rows: ' + ' | '.join(f'{k} {v}' for k, v in diag['spearman_pool'].items()))
diag['mean_abs_change_vs_frac75'] = {k: round(float(np.abs(P_[k] - P_['frac75']).mean()), 4) for k in ('T_a', 'T_b')}
say(f'mean |T - frac75| on pool cells: {diag["mean_abs_change_vs_frac75"]}')
prev = {}
for k in ('v4', 'frac75', 'T_a', 'T_b'):
    X = P_[k]; prev[k] = {c: (round(float(X[:, j].mean()), 3), round(float((X[:, j] >= .5).mean()), 3)) for j, c in enumerate(T)}
diag['prevalence_pool_mean_share05'] = prev
say('pool target mean / share >= .5 per label:')
for j, c in enumerate(T):
    say(f'  {c:17s} ' + ' | '.join(f'{k} {prev[k][c][0]:.3f}/{prev[k][c][1]:.3f}' for k in prev))
rec['diag'] = diag
# ---- 7. LABEL GATE (PREREG_TEACHER sec 4): a table may be trained only if its gold58 target AUC is not clearly below
# frac75's (the labels of every current fleet set): AUC58(T) - AUC58(frac75) >= -0.010. Primary T_b; fallback T_a; else STOP.
L1_BAR = -0.010
t58 = diag['target_auc_gold58']
lg = {tn: (t58.get(f'T_{tn}', float('nan')) - t58['frac75']) for tn in TABLES}
passes = {tn: bool(np.isfinite(d) and d >= L1_BAR) for tn, d in lg.items()}
train = PRIMARY if passes[PRIMARY] else next((tn for tn in TABLES if passes[tn]), None)
rec['label_gate'] = dict(bar=L1_BAR, delta_vs_frac75=lg, passes=passes,
                         train_table=(os.path.basename(rec[f'table_{train}']['path']) if train else None),
                         train_md5=(rec[f'table_{train}']['md5'] if train else None))
say(f'LABEL_GATE {"SIM " if not REAL else ""}| ' + ' | '.join(f'T_{tn} - frac75 {d:+.4f} ({"PASS" if passes[tn] else "FAIL"})' for tn, d in lg.items())
    + f' | bar >= {L1_BAR} | TRAIN_TABLE={rec["label_gate"]["train_table"] or "NONE (STOP: no student)"}')
json.dump(rec, open(f'{A.out_dir}/{PFX}teacher_labels_receipt.json', 'w'), indent=1, default=str)
open(f'{A.out_dir}/{PFX}teacher_labels_diag.txt', 'w').write('\n'.join(L) + '\n')
say(f'TEACHER_LABELS_OK {A.teacher_tag} | primary T_{PRIMARY} {rec["table_" + PRIMARY]["md5"]} | receipt {A.out_dir}/{PFX}teacher_labels_receipt.json')
