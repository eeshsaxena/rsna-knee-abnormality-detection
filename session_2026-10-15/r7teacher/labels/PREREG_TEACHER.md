# PREREG_TEACHER: student label tables and student gate for the R7-teacher program

Written 2026-10-14 ~13:15 IST, frozen ~13:55 IST by the label-design job (workflow R7-TEACHER, job "label design + student kits").
No em dashes. Numbers are tagged MEASURED / MODEL / GUESS.

**Status at writing (MEASURED):** no R7 out-of-fold file exists anywhere (job 1 is still building the private R7-OOF
notebook; its commit test covers ~8 studies and is plumbing only). This job has read NO R7-OOF value. The whole chain was
dry-run with a SIMULATED teacher (K14 member set DT, see sec 9). Every weight, bar and rule below is frozen by the sha256
of this file (`PREREG_TEACHER.sha256`, copied to the box as `~/r7t_labels/PREREG_TEACHER.sha256`).
`build_teacher_labels.py` refuses a REAL build unless this file's sha256 matches the recorded one.

## 1. Question

Does a student from our own TTR family, trained on labels that mix our report labels with the R7 bundle's TRUE
out-of-fold predictions (each RSNA train study scored only by the R7 fold model that did not train on it, i.e. a
distillation of heliosli's private Opus-graded report-state targets), add to the V224 base beyond our current F slot
(K14 fleet)? The decisive measure is one public LB read (API best-id move, sec 7). The offline gates below are guards
against broken, degenerate or cloned students. They are not quality claims: gold58 has 58 studies (paired SE of
set-vs-set differences ~.003, MEASURED K14 gate .0033), and every OOF ruler is biased (sec 8).

## 2. Inputs (frozen; md5 MEASURED on the box 10-14)

| Input | Path (box) | md5 |
|---|---|---|
| v4 report labels | `~/rsna_train/llm_labels_v4_blend.csv` | 44874d8ded7699d356b935ca8d7e8df7 |
| frac75 (labels of every current fleet set except C) | `~/novel/frac75/labels_frac75_logit.csv` | ad7582b817243f8f22e8bcfdd7e3f84a |
| folds5 (student split; gold74 = held out) | `~/rsna_train/folds5.csv` | 9c5117e06e5c6ecc77aabbeeca5db44e |
| gold74 eval set (58 original + 16 re-reads) | `~/rsna_train/gold74_eval_set.csv` | a0ff70f7af1d458b9cfe9509e6122740 |
| official train.csv (58 labelled rows == gold74 original_gold58, values identical: MEASURED) | `~/rsna_train/train.csv` | 3868da29af79ebde4f91b3a97372c095 |
| fleet OOF = equal prob-mean of the OAI-FREE sets C, D, E, E2, D2 (raw-best OOF, 5 folds each, 4,333 pool rows) | `~/r7t_labels/fleet/oof_fleetOF_mean.csv` | 3479835c4056e86a74511ad68e9abb48 |
| fleet gold = equal mean of the same 25 members' gold74 pass-A predictions (D2's 5 computed 10-14, CPU) | `~/r7t_labels/fleet/gold_fleetOF_mean.csv` | ba7956c85c6ec57d08b9e3cc18a2203b |
| R7-OOF teacher | `~/r7t_labels/real/R7OOF.csv` (copied from the Kaggle output) | recorded in the build receipt |

Fleet sets (MEASURED): C = TTR288 mix50 s1044, D = TTR224 frac75 s1043, E = TTR288 frac75 s1045, E2 = TTR288 frac75 s1046,
D2 = TTR224 frac75 s1044. DT (OAI rows) and L13 (OAI rows) are EXCLUDED on purpose: the student tables and every rented
run stay OAI-free (OAI ruling risk, thread 748276; no OAI-derived numbers leave the box). D3 (4/5 folds) is excluded.

**R7-OOF file format (required) = job 1's merged `r7_oof.csv`** (`session_2026-10-15/r7teacher/merge_sessions.py`):
`StudyInstanceUID`, `r7_fold` (= `int(sha1(uid)[:8], 16) % 4`, the hash fold whose model scored the study, i.e. the fold
that did NOT train on it), `p|<label>` x 12 (per-study PROBABILITIES in [0, 1], the R7 bundle's final per-study output),
plus `z|` logits, fluid / state extras and `precision` (all ignored). Plain `<label>` columns and an optional `n_models`
(1 or 4) are accepted too. Every study, gold included, is scored by its hash-fold model: gold58 was never in R7's pool
(manifest `gold_in_training 0`, pool 4,344 of 4,403 feature studies, MEASURED; job 1's `prove_fold_map.log` 23/23 PASS),
so any single fold model is out-of-fold on gold58. The 16 gold74 re-read studies are in R7's pool (MODEL); they are
excluded from student training either way. At most 1 % of the 4,333 non-gold pool rows may be missing (else STOP:
R7 run incomplete). The builder checks `r7_fold` against the hash for every single-model row (0 mismatches required).

**Frozen code (md5, laptop copy == box copy `~/r7t_labels/`, MEASURED):** build_teacher_labels.py 0d032723,
student_gate.py a08b8ec1, collect_fleet_oof.py 059f79e4, labels_gate.sh 65442ebb, sim_r7_oof.py 922fb205;
trainer `~/fast_build/code2/fleet_train.py` 3e62434a. A code change after this file's sha is recorded needs an amendment
written before any REAL R7-OOF value is read.

## 3. The tables (frozen)

```
T_a = .5 v4 + .5 M(R7)                    (mix50 structure, R7 OOF as the image teacher)
T_b = .4 v4 + .3 M(fleet) + .3 M(R7)      PRIMARY
```

- `M(x)` = per-label logit match of x onto v4: z-score of `logit(clip(x, 1e-4, 1 - 1e-4))` rescaled to v4's logit mean
  and SD, statistics on the non-gold pool rows where x exists. This is the exact method that built mix50 and frac75
  (MEASURED: build_frac75.py reproduces all 12 mix50 columns to 1e-12). It keeps every teacher on v4's marginal
  calibration, so the tables stay comparable to frac75 (the fleet's labels) and the soft-BCE trainer sees familiar
  base rates.
- **Mixing space: PROBABILITY space** (convex combination of v4 and the matched teachers), as mix50 / frac75.
  Not logit space: the fleet's whole history (mix50 +.009 solo for TianK003, CLAIMED; frac75 PASS, MEASURED) is
  prob-space mixing; changing two things at once would make a null result unreadable.
- **Pre-registered fallback:** if the teacher file looks like per-label ranks (max over the 12 labels of the KS
  distance to Uniform(0, 1) < .02), `M(R7)` is an ECDF quantile map onto v4's pool marginal instead.
- A source missing for a study is dropped for that study and the remaining weights are renormalised.
- Uniform weights over labels (no per-label weights: public-overfit trap, RESEARCH_960 sec 5).
- Why T_b is primary: it keeps the student anchored on our own image family (diversity against the R7 half of V224)
  while moving ~.3 of the target onto R7's image-informed estimate of heliosli's labels; total image-teacher weight .6
  (T_a .5). T_a is the cleaner "mix50 with a better teacher" and is the fallback / second student.
- Builder: `scripts/build_teacher_labels.py` (outputs `labels_r7t_a.csv`, `labels_r7t_b.csv`, receipt, diag).

## 4. Label gate (before any student training)

`L1`: target macro AUC of the table vs the official labels on gold58 (masked convention of fleet_train.macro_auc), on the
gold rows where R7 = its hash-fold model (out-of-fold) and fleet = the 25-member mean (a best case for the fleet, stated).

- A table may be trained only if `AUC58(T) - AUC58(frac75) >= -0.010`.
- Train T_b if it passes; else T_a if it passes; else **STOP** (no student; report).
- Reported, not gating: gold74 target AUC, R7-vs-v4 OOF AUC, Spearman matrix, prevalence per label.
- The receipt's `label_gate.train_table` is what the box queue gate (`labels_gate.sh`) requires.

## 5. Student training (recipe frozen)

- Recipe = our box TTR224 RSNA-only fold recipe, token for token (fleet_train.py md5 3e62434a; `--model ttr --aug heavy
  --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs 21 --save_epochs 6-21 --workers 10 --store
  ~/fleet_store --folds folds5 --exclude gold74 --eval_labels v4 --gold gold74 --grad_ckpt 0`), only `--labels`,
  `--seed`, `--fold`, `--out` differ.
- T_b student: seed 1047, out `~/fleet_runs/f{F}_s1047_ttr_r7tb`. T_a student: seed 1048, out `f{F}_s1048_ttr_r7ta`.
  Fresh seeds so the kernel member keys `f{F}_s{seed}` never collide with D / DT (s1043) or D2 / L13 (s1044).
- Checkpoint selection as every fleet set: raw best by OOF AUC vs v4 (T3 rule: never select on the training targets,
  which would reward copying the teacher). Members = the 5 raw-best checkpoints.
- Where: box queue (`kits/box`, sheets `queue_r7tb.txt` / `queue_r7ta.txt`, gated by `labels_gate.sh`) and/or the rented
  RSNA-only kit (`kits/vast_r7t`, same argv). Both are prepared, neither is launched by this job.

## 6. Student gate (after 5/5 folds; `scripts/student_gate.py`, box CPU)

S = the student's 5 members (equal mean); D family = D (s1043) and D2 (s1044), the same recipe on frac75 (two seeds, so
the reference is not one seed's luck); F = K14 fleet (C + DT + E + E2, 20 members, the F slot of our LB kernels);
N = nartaa Raptor r384 SWA gold58 predictions in serving geometry (`~/raptor_store/run_s42/serve/...r384_swa.pt.npy`).
rho = mean per-label Spearman on the 4,333 non-gold pool rows (OOF); rk = per-label percentile rank.

| Leg | Rule | Role |
|---|---|---|
| G0 integrity | 5/5 folds closed; raw-best ckpt + OOF per fold; OOF ids == folds5 val ids minus gold74; ckpt args fold / seed / 21 ep / ttr 224 / `train_labels` == table name / `sel_labels` == v4; table md5 == build receipt | bug guard |
| G1 gold58 | `AUC58(S) - mean(AUC58(D), AUC58(D2)) >= -0.005` | expert non-inferiority |
| G2 uptake | `rho(S, R7) - rho_ref >= 0.10 x (1 - rho_ref)` with `rho_ref = mean(rho(D, R7), rho(D2, R7))` (the student closes >= 10 % of the D family's Spearman gap to R7) AND OOF AUC(S vs v4) >= .85 | sanity: the teacher was learned, report signal kept |
| G3 diversity | `rho(S, R7) <= rho(D, D2)` (MEASURED rho(D, D2) = .9625 on the pool) | not an R7 clone |
| G4 blend | on gold58: base = .5 rk(R7) + .5 rk(N); `AUC58(.8 rk(base) + .2 rk(.5 F + .5 S)) - AUC58(.8 rk(base) + .2 rk(F)) >= -0.002` | the LB use, proxied |

VERDICT PASS iff G0-G4 all pass. Everything else the script prints (REPLACE variant, per-label deltas, SEs) is report only.
The T_a student, if trained, is gated the same way.

## 7. LB use (one change per entry; reads = API best-id moves)

- Candidate entry: the current best V224-F kernel line with the F slot = **.5 K14 + .5 S** (prob-space mean of the two
  sets' member means inside F), w_f unchanged (the 10-14 rule). Nothing else changes.
- It enters pick B only if its entry becomes the team best (API best-id), per RESEARCH_960 sec 6 component rule.
- If it does: one follow-up entry with F = S alone (REPLACE) is allowed. If it does not: the student line stops for the
  LB (no weight probing of the .5 split, no w_f retune).
- A T_a student enters only after the T_b read, as its own entry.
- Needs (main session, not this job): a new private Kaggle dataset with the 5 student checkpoints after an OAI-id scan,
  a kernel build from the V224-F builder, a commit test, a runtime check (5 more members in F), a submit slot.

## 8. Biases and leakage (stated, not corrected)

- gold58 legs: no training leakage (gold74 is `--exclude`d from every student and fleet run; gold58 never in R7 training).
  Selection leakage is possible on both sides: R7's author may have watched gold58 during training (unknown), and frac75's
  Dread half is gold58 selection-leaked (10-03 note). Both L1 numbers are therefore optimistic; L1 is a guard only.
- OOF vs v4 is biased AGAINST a teacher-trained student; OOF vs R7 is biased TOWARD it. Both are used only as sanity
  floors (G2), never as a quality score.
- Second-order teacher leakage: R7 fold model h(s) trained on the heliosli labels of studies that are in the student's
  validation fold; the fleet OOF of a training study comes from a fleet model that trained on the student's validation
  fold. The student never sees a validation image, so information can only flow through other studies' teacher values
  (no study-level memorisation path). Expected effect on the student's OOF numbers: negligible (MODEL). Not corrected.
- Split mismatch (folds5 vs R7's 4 hash folds) needs no alignment: each teacher value is already out-of-fold for its own
  study; the student keeps folds5 so it is comparable to every fleet set and its OOF lines up with theirs.
- The simulated dry run (sec 9) says nothing about R7; its numbers describe our own fleet only.

## 9. Dry run with a simulated teacher (MEASURED 10-14, box CPU; says nothing about R7)

- SIM teacher = K14 member DT (5-fold OOF on the pool, 5-fold mean on gold74) written in job 1's r7_oof.csv layout by
  `sim_r7_oof.py` (`p|<label>`, `r7_fold` = hash for every row), 3 pool rows dropped on purpose. Builder: job-1 layout
  parsed, hash-fold check 4,404 rows / 0 mismatches, 3 rows renormalised, trainer-compatible (store 4,333 / 4,333).
  The same teacher in the plain layout gives byte-identical T_a (md5 8ec090db) and T_b (without fleet gold) tables.
- SIM L1 gold58 target AUC: v4 .8927, frac75 .9300, teacher (DT) .9235, fleet .9217, T_a .9309, T_b .9316
  (SIM LABEL_GATE: both PASS, train T_b). gold74: v4 .9113, frac75 .9381, T_a .9380, T_b .9371.
- Trainer: the T_b and T_a sheet argv ran on the real `fleet_train.py` (md5 3e62434a) on CPU (31 studies, 1 epoch):
  rc 0 both; ckpt args fold 1 / seed 1047 / 21 ep / ttr 224 / train_labels = the table name / sel v4 / grad_ckpt False.
- Student gate: stand-in E2 (not trained on the teacher): G0 P, G1 P (-.0004), **G2 FAIL** (delta -.0026 vs bar +.0047),
  G3 P, G4 P: the uptake leg rejects a student that did not learn the teacher. Stand-in DT (= the teacher itself): G2 P,
  **G3 FAIL** (rho 1.000 > .9625): the clone leg fires. Reference numbers (our fleet only): rho(D, D2) = .9625; fleet set
  pairs .955-.964; fleet OOF AUC vs v4 .886-.890 per set; gold58 F (K14) .9212, N .9229, OAI-free fleet .9217.

## 10. Time and cost (MODEL unless tagged)

- Box V100: 3.52 h per TTR224 gc0 21-ep fold (MEASURED, 10.07 min/ep) -> 5 folds ~17.6 h + ~40 min CPU gold inference
  for the gate (MEASURED 450 s per checkpoint). The box is on L13 s1044 f1-f4 until ~10-15 10:25 (MODEL, 5.9 h/fold).
- Rented 4 x RTX 4090 (RSNA-only, kit `kits/vast_r7t`): fold 1.68 h mid / 2.1 h worst (offers_rank SPEED 4090 = 2.1 x
  V100, MODEL). 5 folds = 2 rounds: 4.1 h mid / 4.9 h worst incl. 0.45 h setup + 0.25 h fetch; at $1.898/h (MEASURED
  10-13 rate) = **$7.7 mid / $9.3 worst + ~$0.3 data-in**. Both tables (10 folds, 3 rounds): 5.7 / 7.0 h, $10.9 / $13.3.
  Credit $17.23 (MEASURED 10-14), floor $2.
- Wall clock to an LB read after the R7-OOF file lands (MODEL): label build + gate 10 min; rental 5 h or box 18 h; student
  gate 45 min; dataset + kernel + commit ~1.5 h; hidden scoring ~5-7 h.

## 11. Stop rules

- Label gate STOP (sec 4), student gate FAIL (sec 6), or a non-moving LB read (sec 7) ends the line for the LB.
- No re-weighting of T_a / T_b, no new table, no bar change after any REAL R7-OOF value has been read.
- Training freeze 10-20 12:00 IST: a student that cannot finish 5/5 folds before it is not used.

## Amendment A1 (written 2026-10-14 14:55 IST, BEFORE any full R7-OOF byte exists; PLAN_TEACHER sec 1 FX1-FX6)

Base: the text above this heading is byte-identical to the file frozen at 13:50 IST (sha256
5d25ff950a89b0b33d8789af6319f2af75d9c91031953da90ff8938cfab2db0e; check: head -c 14577 PREREG_TEACHER.md | sha256sum).
What had been seen when A1 was written (MEASURED): only the 44 format-check rows of job 1 (commit v1: 8 OOF rows; proof v2:
the hash-fold model's logits of 36 proof studies). They exposed the clip bug and gave the |z| magnitudes for the simulation
(FX6). No full R7-OOF run had started. A1 is a method fix, not a value-tuned choice. Tables, mixing weights, M() for v4 and
the fleet, seeds (1047 T_b / 1048 T_a), the training recipe, the label gate bar and the G1, G2, G3, G4 bars are UNCHANGED.

- **FX1 + FX2 (teacher values).** M(R7) uses the RAW fold logits `z|<label>` of the R7-OOF file for all 12 labels, UNCLIPPED:
  z-scored on the non-gold pool rows where R7 exists, rescaled to v4's logit mean / SD (v4 keeps logit(clip(v4, 1e-4))).
  Reason: logit(clip(p, 1e-4)) tied 55.5 % of the 44 real hash-fold cells at the clip (ACL 82 %, Fracture 70 %; MEASURED),
  making M(R7) near-binary; and `p|Effusion` / `p|Baker's` are the stackers' outputs (fit on all 4,344 OOF rows incl. the
  scored study), `z|` is not. A REAL teacher file without `z|` columns is refused; `p| == sigmoid(z|)` within 1e-6 on the 10
  non-stacked labels is asserted (column alignment). The ECDF fallback branch is unchanged. M(fleet) is unchanged.
- **FX3 (L0 per-fold leak gate, new, before any table).** Macro AUC of R7 `z|` vs confident v4 cells (v4 <= .1 -> 0,
  v4 >= .9 -> 1) per R7 hash fold on the non-gold pool rows, over the labels valid in all 4 folds. PASS iff max - min < .020.
  Else the outlier = the fold farthest from the median of the other three; if those three span < .020: DROP = that fold's
  non-gold pool rows lose their R7 component (T_b renormalises to .4/.7 v4 + .3/.7 fleet there, T_a to v4 alone; the
  missing cap becomes 27 % for this branch only; gold rows keep R7 because gold58 was never in R7's pool); otherwise STOP
  (exit 4, no table, the line ends). Tables may hold v4's hard 0 / 1 cells on rows whose only source is v4 (asserted range
  [0, 1]). `on_r7oof.sh` prints the L0 line and stops on exit 4.
- **FX4 (checkpoint selection on the student's own table).** The student queue lines pass `--eval_labels` = the student's
  own table (T_b: `~/r7t_labels/real/labels_r7t_b.csv`), so the raw-best checkpoint and its OOF follow the AUC vs that table on
  held-out validation rows (rewards generalising the teacher; validation rows are never trained on, so it cannot reward
  copying). Replaces sec 5's "raw best by OOF AUC vs v4". G0 requires ckpt args `sel_labels == train_labels ==` the table name.
- **FX2b (gate ruler, consistency with FX1/FX2).** The pool-row R7 ruler of G2 / G3 / G3' is `z|` when present (what the
  student was taught; rank-identical to sigmoid(z) on the 10 non-stacked labels). G4 (the LB proxy) keeps `p|`, the deployed
  R7 output. G2 / G3 bars unchanged.
- **FX5 (G3' diversity ceiling, gates the LB entry).** d = rho(S, R7) - rho_ref (rho_ref = mean(rho(D, R7), rho(D2, R7)),
  as in G2). The entry is skipped if d > 0.5 x (1 - rho_ref) (the student closes more than half of the D family's gap to R7;
  threshold GUESS). Consistent with G2's floor d >= 0.10 x (1 - rho_ref) (a flat d > .02 would contradict it). Reported, not
  gating: d, the gap share, rho(S, OAI-free fleet OOF mean), rho(S, K14 OOF mean C + DT + E + E2). VERDICT (A1) = PASS iff
  G0, G1, G2, G3, G3', G4 all pass. Recorded, not corrected (second-order chain leak, review 1): R6 state fusion, plr7 cell
  weights, all-row stackers and holdout_fold None in heliosli's bundle can leak up to about 1e-3 into train-row rulers
  (OOF-vs-R7 numbers; L1 is on gold58), 0 on the hidden test (MODEL / GUESS).
- **FX6 (realistic SIM dry run).** The simulated teacher = DT's logits with DT's order kept and the per-label |z| magnitudes
  quantile-mapped onto the 44 real rows (`r7_absz_ref44.json`), stacker-like `p|` on Effusion / Baker's. It reaches the clip on
  55.6 % of pool cells (real rows 55.5 %).
- **Code freeze (A1):** each REAL build asserts that this file contains `Amendment A1` and the builder's own md5; the student
  gate does the same in G0 (REAL runs). Any later code change needs Amendment A2 before any REAL R7-OOF value is read.

Frozen code and sheets (md5, laptop `session_2026-10-15/r7teacher/labels/` == box `~/r7t_labels/`):

| file | md5 |
|---|---|
| build_teacher_labels.py | a7f8823d7216fb0a55d5fc7821607c78 |
| student_gate.py | 8c0187906013dc611725d2507755e78f |
| sim_r7_oof.py | 0e61f7cc45bc2827c7b27ccee8a0d7f7 |
| r7_absz_ref44.json (+ make_absz_ref44.py 20998d21a72f4682b66142ac2bf03297) | 9a7ce040280cd448dcf854a608803cc6 |
| make_sheets.sh | 772f4be0a235db7c6d18911dc75bfbb9 |
| on_r7oof.sh | 4d434327aa3ee219946d4bc50f0dba4a |
| dry_run_student.sh | 50e1d57d9d679f641d34b948be2decbc |
| labels_gate.sh (unchanged) | 65442ebb6d26313bf21a6c956b92b983 |
| collect_fleet_oof.py (unchanged) | 059f79e42fffb960c5e167b18578b351 |
| sheets/queue_r7tb.txt (T_b, seed 1047, `--eval_labels` = the T_b table) | 6196c8e48ff15687f524a4c180a4bda6 |
| sheets/queue_r7ta.txt (T_a, seed 1048) | 7b95041e616597b5d9b4c73f6b97561a |
| sheets/queue_r7tb_SIMCHECK.txt / queue_r7ta_SIMCHECK.txt | 49b6f846845a448191dcad0f9cd68aaf / 300faa2de7412d4a5e46ec828fc85ccf |
| trainer ~/fast_build/code2/fleet_train.py (unchanged) | 3e62434adea4fd8971af7c96e0465773 |

A1 dry run (box CPU, cores 34-39, nice 19; GPU and queue untouched; logs `~/r7t_labels/a1_dryrun.log`,
`a1_recheck_builder.log`; MEASURED, describes our own fleet only, never R7):
- Pre-A1 path reproduced: the A1 builder on the pre-A1 SIM file gives the pre-A1 tables byte for byte (8ec090db / 8fbf5787).
- Realistic SIM: L0 PASS (fold AUCs .9345 / .9362 / .9350 / .9337, spread .0025, bootstrap SE ~.004 per fold). L1 gold58:
  T_a - frac75 -.0003, T_b +.0013 (LABEL_GATE PASS, train T_b); the PRE-A1 builder on the same teacher: T_a -.0042, T_b +.0002.
  M(R7) tie share now <= .03 % (pre-A1 clip: 55.6 % of cells tied). Tables T_a d2502485, T_b dbfa4a52.
- FX2: changing p| of Effusion / Baker's (stacker seed) leaves both tables byte-identical.
- L0 negative controls: an in-fold leak on hash fold 2 -> fold AUC +.038 -> DROP (1,116 rows renormalised, 25.8 % < 27 %,
  label gate still PASS); leaks on folds 1 + 2 -> STOP, exit 4, no table.
- REAL-path guards: no z| -> refused; PREREG without this builder's md5 -> refused; wrong sha -> refused; correct -> PASS.
- Trainer argv on the real fleet_train.py (CPU, 31 studies, 1 epoch): T_b rc 0, T_a rc 0; ckpt args fold 1, seed 1047 / 1048,
  21 ep, ttr 224, grad_ckpt False, sel_labels == train_labels == the table name.
- Student gate stand-ins: E2 (did not learn the teacher) G2 FAIL, G3' PASS (d -.0026); DT (= the teacher) G3 FAIL and
  G3' FAIL (d +.0470 > ceiling .0235). Both VERDICT FAIL as designed (SIM, not real verdicts).

Open item for the main session (not changed by A1): `labels_gate.sh` passes only the table the label gate chose, so a T_a
student cannot be queued behind T_b with the current gate (its lines would be SKIPPED). Training T_a as a second student
needs Amendment A2 (e.g. a T_a verdict = "T_a passes L1 on its own") written before its lines are installed.

## Amendment A2 (written 2026-10-14 16:05 IST, while the full R7-OOF run is RUNNING on Kaggle; no byte of it downloaded or read)

Trigger: plan change from the main session at 15:56 (Eesh wants his Vast credit used): the **T_b student (seed 1047) trains on a
rented 4 x RTX 4090** (RSNA-only, ~$15 cap, run by a separate rental agent; sec 5 already allowed "the rented RSNA-only kit,
same argv"), and the **box trains the T_a student (seed 1048)** ahead of the L13 filler, with the 5 T_b box lines kept as
comments (#HOLD) as the fallback if the rental fails. Tables, weights, seeds, recipe, A1 (FX1-FX6) and every gate bar are
UNCHANGED. Changes:

- **Which tables may be trained (replaces "train T_b if it passes, else T_a"):** each table is trainable iff it passes the L1
  bar on its own (AUC58(T) - AUC58(frac75) >= -.010) and L0 is PASS or DROP. `labels_gate.sh` now PASSes a table iff the REAL
  receipt carries the frozen PREREG sha256, L0 is PASS / DROP, `label_gate.passes[a|b]` is true for THAT table, and the file md5
  equals the receipt's `table_a` / `table_b` md5 (was: "the label gate chose this table"; identical for T_b). The receipt's
  `train_table` (the primary choice) is unchanged and still names T_b when it passes.
- **Publication for the rental:** `finalize_labels.sh` (box) runs `labels_gate.sh` for both tables right after the REAL build
  and copies the passing tables + the receipt to `~/r7t_labels/final/`; it writes `~/r7t_labels/final/LABELS_READY` (sha256 +
  md5 of each table, PREREG sha256, L0 verdict, label gate deltas) LAST and atomically, only if T_b passes (else
  `LABELS_STOP` with the reason). The rental agent starts only on LABELS_READY and uses the T_b table byte-identical to
  `final/labels_r7t_b.csv`.
- **Rental argv = the A1 box sheet line:** `kits/vast_r7t/remote/argv_template.txt` now carries `--eval_labels` = the T_b table
  (A1 FX4; `box_ops.sh argv-check 0-4`: IDENTICAL to `queue_r7tb.txt`, MEASURED 16:00) and the kit's ckpt-args check expects
  `sel_labels ==` the table. (kit.env's cosmetic queue-name template was left as is.)
- **Box queue:** `install_r7ta_top.sh` puts the 5 T_a lines (sheet queue_r7ta.txt, A1 md5 7b95041e) ahead of the L13 lines and
  the 5 T_b lines as `#HOLD(...)` comments (un-hold only if the T_b rental fails), via `qctl.sh top` only, after labels_gate
  PASSes T_a. The student gate (A1) gates each student separately; LB use unchanged: S = T_b enters first (PLAN sec 7 /
  PREREG_10-14 Amendment 4 text); a T_a student enters only after the T_b read, same form, only if its own G0-G3' + G4 pass.

Frozen code added / changed by A2 (md5; laptop == box): labels_gate.sh 6f15f41451824b15250ff56ab082773e (A1: 65442ebb),
finalize_labels.sh ff4d1ec9d8414d4bdd3ac4477ee4168a (new), install_r7ta_top.sh a921eabb83b40b9a540a17670348a4be (new),
rental kit argv_template.txt 185108e9c432134093b43a3eb560fd4e and box_ops.sh 16544ecdb2b6be9ce9d8b5c526666880. Every A1 md5
above is unchanged (build_teacher_labels.py a7f8823d7216fb0a55d5fc7821607c78, student_gate.py 8c0187906013dc611725d2507755e78f).
Tests (box CPU, isolated temp HOME, SIM values under a REAL tag, deleted after; MEASURED): T_b PASS and T_a PASS on the null SIM
teacher and on the 1-fold-leak SIM (L0 DROP); a SIM table name, a tampered T_a file, `passes.b = False` and an L0 STOP receipt
all FAIL; finalize_labels.sh wrote LABELS_READY with both tables (md5 re-checked) and refuses a second run.
