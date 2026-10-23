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
# PREREG_TEACHER Amendment A1 (10-14, before any full R7-OOF byte existed; PLAN_TEACHER sec 1 FX1-FX3):
#   FX1/FX2 the R7 teacher enters M(R7) through its RAW fold logits `z|<label>` (all 12 labels, UNCLIPPED): the logit match
#           z-scores them (that sets the temperature) and rescales onto v4's logit mean / SD. The old path took logit(clip(p,
#           1e-4)), which tied 55.5 % of the 44 real hash-fold cells at the clip (MEASURED) and turned M(R7) near-binary; and
#           Effusion / Baker's p| are the stackers' outputs (fit on all 4,344 OOF rows incl. the scored study), z| is not.
#           A REAL teacher file without z| columns is refused. v4 and the fleet keep logit(clip(x, 1e-4)) (unchanged).
#           The ECDF fallback branch is unchanged (it reads p|). Diagnostics use z| (rank-identical, no float32 saturation).
#   FX3     L0 per-fold leak gate (new, before the tables): macro AUC of z| vs confident v4 cells (v4 <= .1 -> 0, >= .9 -> 1)
#           per R7 hash fold on the non-gold pool rows (labels valid in all 4 folds). PASS iff max - min < .020. Else: the
#           fold farthest from the median of the other three is the outlier; if the other three then span < .020, that ONE
#           fold's non-gold pool rows lose their R7 component (T_b renormalises to v4 + fleet there, T_a to v4; missing cap
#           .27 for this branch only; gold rows keep R7: gold58 was never in R7's pool); otherwise STOP (exit 4, no tables).
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
L0_BAR, L0_LO, L0_HI, MISSING_MAX_L0 = 0.020, 0.1, 0.9, 0.27      # A1 FX3 (frozen)
NONSTACKED = [c for c in T if c not in ('Effusion', "Baker's")]  # p| == sigmoid(z|) only for these (R7 stackers on the other 2)
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
           primary=PRIMARY, clip=CLIP, amendment='A1', inputs={})


def finish(code=0):
    json.dump(rec, open(f'{A.out_dir}/{PFX}teacher_labels_receipt.json', 'w'), indent=1, default=str)
    open(f'{A.out_dir}/{PFX}teacher_labels_diag.txt', 'w').write('\n'.join(L) + '\n')
    if code:
        sys.exit(code)

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
    _txt = open(A.prereg, encoding='utf-8').read(); _me = digest(os.path.abspath(__file__), 'md5')
    assert 'Amendment A1' in _txt and _me in _txt, f'A1: this builder (md5 {_me}) is not the code frozen in the PREREG amendment'
    rec['builder_md5'] = _me; say(f'builder md5 {_me} listed in the PREREG (A1 code freeze) OK')
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
# A1 FX1/FX2: the raw fold logits z|<label> (float32, exact in job 1's file) are the teacher's values for M(R7), all 12 labels
HAS_Z = all('z|' + c in te.columns for c in T)
if HAS_Z:
    tz = te[['z|' + c for c in T]].astype(float); tz.columns = T
    assert np.isfinite(tz.values).all(), 'non-finite z| logits in the teacher file'
    dz = float(np.abs(expit(tz[NONSTACKED].values) - tv[NONSTACKED].values).max())
    say(f'teacher logits: z|<label> present for all 12 labels -> M(R7) uses the RAW fold logits, unclipped (A1 FX1/FX2) | '
        f'max |p - sigmoid(z)| over the 10 non-stacked labels {dz:.2e}')
    assert dz < 1e-6, 'p| != sigmoid(z|) on the non-stacked labels: columns misaligned?'
    rec['teacher_space'] = 'z| raw fold logits (A1)'; rec['p_vs_sigmoid_z_max'] = dz
else:
    assert not REAL, 'A1 (FX1): a REAL teacher file must carry the raw fold logits z|<label> for all 12 labels'
    tz = pd.DataFrame(logit(np.clip(tv.values, CLIP, 1 - CLIP)), index=tv.index, columns=T)
    say('teacher logits: NO z| columns (SIM plain layout): logit(clip(p, 1e-4)) as before A1')
    rec['teacher_space'] = 'logit(clip(p)) (SIM plain layout, pre-A1 path)'
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

# ---- 2b. L0 per-fold leak gate (PREREG_TEACHER Amendment A1, FX3; before any table is written)
from sklearn.metrics import roc_auc_score
hf = lambda u: int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4
pr0 = [s for s in pool if s in tz.index]
V4c = v4.loc[pr0, T].values.astype(float); Z0 = tz.loc[pr0, T].values; fk = np.array([hf(s) for s in pr0])
conf = (V4c <= L0_LO) | (V4c >= L0_HI); Y0 = V4c >= L0_HI


def l0_label_aucs(rows, w=None):
    out = {}
    for j, c in enumerate(T):
        ok = rows & conf[:, j] & (True if w is None else w > 0)
        if ok.sum() and 0 < Y0[ok, j].mean() < 1:
            out[c] = float(roc_auc_score(Y0[ok, j], Z0[ok, j], sample_weight=None if w is None else w[ok]))
    return out


per = {k: l0_label_aucs(fk == k) for k in range(4)}
LAB0 = [c for c in T if all(c in per[k] for k in range(4))]
auc0 = np.array([np.mean([per[k][c] for c in LAB0]) for k in range(4)])
rng0 = np.random.default_rng(14)
se0 = []
for k in range(4):                                                    # row bootstrap within the fold (report only)
    idx = np.flatnonzero(fk == k); bb = []
    for _ in range(200):
        w = np.bincount(rng0.choice(idx, len(idx), replace=True), minlength=len(pr0)).astype(float)
        a = l0_label_aucs(fk == k, w); bb.append(np.mean([a[c] for c in LAB0 if c in a]))
    se0.append(float(np.std(bb)))
spread = float(auc0.max() - auc0.min())
dev = np.array([abs(auc0[k] - np.median(np.delete(auc0, k))) for k in range(4)])
out_k = int(np.argmax(dev)); rest = np.delete(auc0, out_k)
L0V = 'PASS' if spread < L0_BAR else ('DROP' if float(rest.max() - rest.min()) < L0_BAR else 'STOP')
rec['l0'] = dict(bar=L0_BAR, conf=[L0_LO, L0_HI], labels=LAB0, auc_per_fold=[round(float(x), 5) for x in auc0],
                 se_boot_per_fold=[round(x, 5) for x in se0], n_rows=[int((fk == k).sum()) for k in range(4)],
                 n_conf_cells=[int(conf[fk == k].sum()) for k in range(4)], spread=round(spread, 5), verdict=L0V,
                 outlier_fold=(out_k if L0V != 'PASS' else None),
                 outlier_delta=(round(float(auc0[out_k] - np.median(rest)), 5) if L0V != 'PASS' else None))
say(f'L0 per-fold leak gate (A1 FX3): macro AUC of R7 z| vs confident v4 cells (<= {L0_LO} / >= {L0_HI}), non-gold pool rows, '
    f'{len(LAB0)} labels: ' + ' | '.join(f'f{k} {auc0[k]:.4f} (SE {se0[k]:.4f}, n {int((fk == k).sum())})' for k in range(4))
    + f' | spread {spread:.4f} (bar < {L0_BAR}) -> L0 {L0V}'
    + (f' (outlier fold {out_k}, {auc0[out_k] - np.median(rest):+.4f} vs the median of the other three)' if L0V != 'PASS' else ''))
if L0V == 'STOP':
    rec['label_gate'] = dict(bar=None, train_table=None, train_md5=None, stop='L0: 2 or more outlier folds')
    say('L0 STOP: two or more R7 folds disagree (PREREG_TEACHER A1 FX3): no table is written, the R7-teacher line stops')
    finish(4)
if L0V == 'DROP':
    drop_rows = [s for s in pr0 if hf(s) == out_k]
    tv = tv.drop(index=drop_rows); tz = tz.drop(index=drop_rows)
    miss2 = [s for s in pool if s not in tv.index]
    rec['l0']['dropped_rows'] = len(drop_rows); rec['teacher_missing_pool_after_l0'] = len(miss2)
    say(f'L0 DROP: R7 component removed for the {len(drop_rows)} non-gold pool rows of hash fold {out_k} (tables renormalise there; '
        f'gold rows keep R7) | pool rows without R7 now {len(miss2)} ({len(miss2) / len(pool):.2%}, branch cap {MISSING_MAX_L0:.0%})')
    if len(miss2) > MISSING_MAX_L0 * len(pool):
        say('STOP: missing rows above the L0-branch cap'); rec['label_gate'] = dict(train_table=None, stop='L0 cap'); finish(4)
assert (tv.index == tz.index).all()
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


def match(src, name, zsrc=None):
    """per-label map of src (DataFrame over some ids) onto v4, statistics from the pool rows present in src.
    zsrc (A1 FX1/FX2): the source's raw logits on the same ids; used unclipped instead of logit(clip(src))."""
    out = pd.DataFrame(index=src.index, columns=T, dtype=float); st = {}
    pr = [s for s in pool if s in src.index]
    assert zsrc is None or (zsrc.index == src.index).all()
    for c in T:
        a = src[c].values.astype(float); ap_ = src.loc[pr, c].values.astype(float); r = v4.loc[pr, c].values.astype(float)
        if MAP == 'lm' or name == 'fleet':
            if zsrc is not None:
                la = zsrc[c].values.astype(float); lap = zsrc.loc[pr, c].values.astype(float)
            else:
                la = logit(np.clip(a, CLIP, 1 - CLIP)); lap = logit(np.clip(ap_, CLIP, 1 - CLIP))
            lr = logit(np.clip(r, CLIP, 1 - CLIP))
            out[c] = expit((la - lap.mean()) / lap.std() * lr.std() + lr.mean())
            st[c] = dict(src_mu=round(float(lap.mean()), 4), src_sd=round(float(lap.std()), 4), v4_mu=round(float(lr.mean()), 4),
                         v4_sd=round(float(lr.std()), 4), src_space='z' if zsrc is not None else 'logit(clip(p))')
        else:                                                    # ECDF quantile map onto v4's pool marginal
            srt = np.sort(ap_); q = np.searchsorted(srt, a, side='right') / len(srt)
            out[c] = np.quantile(r, np.clip(q, 0, 1))
            st[c] = dict(ecdf=True)
    return out, st


r7m, st_r7 = match(tv, 'r7', tz if HAS_Z else None); flm, st_fl = match(fleet, 'fleet')
# A1 tie report: share of pool cells the pre-A1 clip would have tied at 1e-4 / 1 - 1e-4, and the tie share of M(R7) now
_pz = tz.loc[[s for s in pool if s in tz.index]]
clip_tie = {c: round(float((np.abs(_pz[c].values) > logit(1 - CLIP)).mean()), 4) for c in T}
m_tie = {c: round(1 - len(np.unique(r7m.loc[_pz.index, c].values)) / len(_pz), 4) for c in T}
rec['tie_report'] = dict(pre_a1_clip_tie_share=clip_tie, pre_a1_clip_tie_share_all=round(float(np.mean(list(clip_tie.values()))), 4),
                         m_r7_tie_share=m_tie)
say(f'A1 tie report (pool cells): the pre-A1 clip would tie {np.mean(list(clip_tie.values())):.1%} of R7 cells '
    f'(per label {clip_tie}); M(R7) tie share now max {max(m_tie.values()):.2%}')
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
    v4_only_pool = int((den.loc[pool] <= W.get('v4', 0) + 1e-12).sum())                     # A1 report (T_a after an L0 DROP)
    # A1: [0, 1], not (0, 1): a row whose only remaining source is v4 (T_a after an L0 DROP) carries v4's hard 0 / 1 cells,
    # which the soft-BCE trainer takes as it does in every v4-label run
    assert np.isfinite(tab.values).all() and (tab.values >= 0).all() and (tab.values <= 1).all(), tn
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
    rec[f'table_{tn}']['pool_rows_v4_only'] = v4_only_pool
    say(f'WROTE {p} | md5 {rec[f"table_{tn}"]["md5"]} | weights {W} | rows renormalised {renorm} (pool {renorm_pool}; pool rows = v4 alone {v4_only_pool})')

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
cands = {'v4': v4, 'frac75': fr, 'R7_teacher': tz, 'T_a': outs['a'], 'T_b': outs['b']}      # A1: R7 = z| (rank-identical)
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
diag['r7_vs_v4_auc_pool'] = round(float(np.mean([roc_auc_score(Yv[:, j], tz.loc[pr_, T].values[:, j]) for j in range(12)])), 4)
diag['fleet_vs_v4_auc_pool'] = round(float(np.mean([roc_auc_score(Yv[:, j], fl.loc[pr_, T].values[:, j]) for j in range(12)])), 4)
say(f'teacher OOF macro AUC vs v4 (>= .5) on pool rows: R7 {diag["r7_vs_v4_auc_pool"]:.4f} | fleet {diag["fleet_vs_v4_auc_pool"]:.4f}')
sp = lambda X, Y: float(np.mean([spearmanr(X[:, j], Y[:, j]).correlation for j in range(12)]))
P_ = {k: P.loc[pr_, T].values.astype(float) for k, P in (('v4', v4), ('frac75', fr), ('fleet', fl), ('R7', tz), ('T_a', outs['a']), ('T_b', outs['b']))}
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
say(f'TEACHER_LABELS_OK {A.teacher_tag} | primary T_{PRIMARY} {rec["table_" + PRIMARY]["md5"]} | L0 {rec["l0"]["verdict"]} | '
    f'receipt {A.out_dir}/{PFX}teacher_labels_receipt.json')
finish(0)
