# R7-TEACHER: label design + student kits (RESULT, 10-14 IST)

Job: label design and student kits for the R7-teacher program (workflow R7-TEACHER). No em dashes.
Tags: MEASURED (we ran or read it), MODEL (computed from MEASURED inputs), GUESS. Nothing here trained on the GPU,
rented anything, pushed a kernel, submitted, or created a dataset.

## 0. Bottom line

- **Ready the moment the R7-OOF file exists.** One box command (`on_r7oof.sh <merged r7_oof.csv>`) builds the
  pre-registered student label tables, runs the label gate and writes the verdict that the box queue lines and the rental
  kit both require. The full chain (fleet OOF collection -> simulated teacher in job 1's exact file layout -> label build
  -> trainer argv on the real trainer (CPU) -> student gate) ran end to end on a SIMULATED teacher (K14 member DT).
- **Pre-registered before any R7-OOF value existed** (`PREREG_TEACHER.md`, sha256 in `PREREG_TEACHER.sha256`, copied to
  the box). Primary table **T_b = .4 v4 + .3 M(fleet OOF) + .3 M(R7 OOF)**, fallback **T_a = .5 v4 + .5 M(R7 OOF)**;
  M = per-label logit match onto v4 (the exact mix50 / frac75 method), convex mix in probability space. Fleet component =
  OAI-FREE sets only (C, D, E, E2, D2), so the student and any rental stay OAI-free.
- **Student = our TTR224 fold recipe, token for token**, seed 1047 (T_b) / 1048 (T_a), 5 folds on folds5.
  Box: ~17.6 h for 5 folds (MEASURED 3.52 h/fold). Rented 4 x RTX 4090: ~4.1 h mid / 4.9 h worst, **~$7.5-9.3**
  (MODEL from MEASURED 10-13 offers); both tables on one 4 x 4090: ~5.7-7.0 h, $10.6-12.9. Credit $17.23, kit cap $15.
- **Honest value:** the student only matters if it moves the API best-id when it replaces half of the F slot of the
  V224-F line (PREREG sec 7). Expected public effect is small (GUESS +0 to +.0005; .960 still needs +.0034 or more).
  This is the "own family on image-informed pseudo labels" lever the top teams use, at the scale we can afford.
- **PREREG frozen 13:50 IST, sha256 `5d25ff950a89b0b3...`** (laptop and box copies identical; code md5s inside). Written
  before any R7-OOF value existed; `build_teacher_labels.py` refuses a REAL build without a matching sha (tested).
- **Format aligned with job 1:** the builder reads job 1's merged `r7_oof.csv` as is (`p|<label>` columns, `r7_fold` =
  hash for every study, gold58 scored by its hash-fold model, which is out-of-fold because gold58 was never in R7's pool).
- **One blocker:** the R7-OOF file itself (job 1; Eesh's OK for the multi-hour Kaggle run). Everything after it is
  prepared and tested; nothing is launched, installed, rented, pushed or submitted.

## 1. Files

| Path (laptop: `session_2026-10-15/r7teacher/labels/`; box: `~/r7t_labels/`) | What |
|---|---|
| `PREREG_TEACHER.md` + `.sha256` | tables, label gate, student recipe, student gate, LB use, biases, stop rules |
| `scripts/collect_fleet_oof.py` | per-set OOF (5 folds, folds5 val ids asserted) + fleet mean + gold74 mean + receipts; refuses OAI sets |
| `scripts/sim_r7_oof.py` | simulated teacher in job 1's `r7_oof.csv` layout (p&#124;label columns, r7_fold = hash), tag SIM: |
| `scripts/build_teacher_labels.py` | the builder (T_a, T_b), all checks, diagnostics, LABEL_GATE; REAL builds need the PREREG sha |
| `scripts/labels_gate.sh` | queue-gate adapter: PASS only for a REAL, PREREG-matching, label-gate-chosen, md5-unchanged table |
| `scripts/student_gate.py` | PREREG sec 6 gate (G0-G4) incl. CPU gold inference of the student (`--infer`) |
| `kits/box/make_sheets.sh`, `kits/box/sheets/queue_r7t{b,a}.txt` (box `~/r7t_labels/sheets/`) | 5 queue lines each (gated, NOT installed) |
| `kits/box/on_r7oof.sh` | the one command to run when the merged R7-OOF file lands (build + label gate + next commands) |
| `kits/box/dry_run_student.sh` | CPU dry run of the student argv on the real trainer |
| `kits/vast_r7t/` (+ `RUNBOOK.md`) | rented RSNA-only kit (port of vastS1044 + a BOX-side fetch loop), not launched |
| box `fleet/` | `oof_fleetOF_mean.csv` (md5 3479835c), `gold_fleetOF_mean.csv`, per-set OOF, receipt |
| box `sim/` | SIM teacher, SIM tables + receipt + diag |
| `dryrun/` | copies of the dry-run logs, receipts and gate output |

## 2. What was measured today

All on the box CPU (nice 19, cores 26-39), GPU and queue untouched (L13 f1 kept 14.0-14.3 min/epoch, MEASURED).

| Step | Result (MEASURED) |
|---|---|
| Fleet OOF (C, D, E, E2, D2; 25 members) | 4,333 pool rows each, ids == folds5 val ids; OOF AUC vs v4 C .8904, D .8860, E .8871, E2 .8887, D2 .8860; pairwise Spearman .955-.964, rho(D, D2) .9625; fleet mean md5 3479835c |
| D2 gold74 preds (were missing) | 5 x CPU pass A, 427-556 s each; gold58 per member .9093-.9175; OAI-free fleet gold58 .9217 (K14 .9212) |
| Official labels | train.csv's 58 labelled rows == gold74 original_gold58, values identical |
| SIM teacher (K14 member DT, job-1 layout) | 4,404 rows, 3 pool rows dropped on purpose, hash-fold check 0 mismatches |
| SIM label build | T_a md5 8ec090db, T_b md5 8fbf5787; 3 rows renormalised; trainer-compatible 4,333 / 4,333; identical tables from the plain and job-1 layouts |
| SIM L1 gold58 target AUC | v4 .8927, frac75 .9300, teacher .9235, fleet .9217, T_a .9309, T_b .9316 -> SIM LABEL_GATE PASS, train T_b |
| Trainer argv (real fleet_train.py, CPU, 31 studies, 1 epoch) | T_b rc 0, T_a rc 0; ckpt args fold / seed 1047 / 21 ep / ttr 224 / train_labels = table name / grad_ckpt False |
| Queue sheets `--check` | 5 lines, 0 BAD per sheet; real sheets WOULD-DEFER (labels missing) |
| REAL code path (isolated copy, SIM values under a REAL tag, then deleted) | PREREG sha OK -> build -> LABEL_GATE -> labels_gate PASS -> next commands; a wrong sha -> STOP; a different teacher file -> REFUSE |
| labels_gate.sh | FAIL on a SIM table, on a missing receipt, on a SIM receipt |
| Student gate, stand-in E2 (not trained on the teacher) | G0 P, G1 P (-.0004), **G2 FAIL** (uptake -.0026 < +.0047), G3 P, G4 P (d -.0000, SE .0006) |
| Student gate, stand-in DT (= the teacher) | G2 P (+.0470), **G3 FAIL** (rho 1.000 > .9625): the clone leg fires |
| Vast kit, box side | helpers synced to `~/vastR7T`; paths OK except the REAL table; argv-check 5/5 IDENTICAL; parity.py with the SIM table OK; ckpt-args reference check: 0 unexpected key diffs on 3 existing runs |

The SIM numbers describe our own fleet only (DT is a sibling of the fleet), never R7.
One design change made during the dry run, before the freeze: G2 became relative (close >= 10 % of the D family's
Spearman gap to R7) instead of an absolute +.02, so a genuine student is not rejected when R7 happens to be close to our
fleet; the negative control still fails it.

## 3. Label design (PREREG sec 3-4)

- **Why these two tables.** mix50 (.5 v4 + .5 logit-matched image teacher) is our proven label recipe (frac75 = mix50 with
  Fracture .75; every TTR set trains on it). The only change in T_a is the teacher: heliosli's R7 out-of-fold predictions
  (image model trained on Opus-graded report states, true OOF) instead of Dread's in-sample Raptor probabilities. T_b
  keeps .3 of our own fleet OOF in the target (self-distillation of our image family) so the student stays partly
  anchored on our family: diversity against the R7 half of V224 is what an F-slot member needs.
- **Probability space, logit-matched teachers.** Continuity with mix50 / frac75 (one change at a time).
- **Leakage / bias** (PREREG sec 8): gold74 is excluded from every student and fleet run and gold58 was never in R7's pool,
  so gold58 legs have no training leakage; OOF vs v4 is biased against the student, OOF vs R7 toward it, so both are
  sanity floors only. Second-order teacher leakage through other studies' teacher values is stated, not corrected.
- **Label gate:** a table trains only if its gold58 target AUC is within .010 of frac75's; T_b first, then T_a, else STOP.

## 4. Student kits

### 4a. Box (V100), queue lines (NOT installed)
Box sheet `~/r7t_labels/sheets/queue_r7tb.txt`, line for fold F (F = 0..4):
```
R7Tb S1047 fF (TTR224 s1047 R7-teacher T_b, 21 ep, gc0; PREREG_TEACHER) | ~/fleet_runs/fF_s1047_ttr_r7tb | needs PASS from ~/r7t_labels/real/labels_gate_labels_r7t_b.verdict wait 120 via bash ~/r7t_labels/labels_gate.sh labels_r7t_b.csv | --model ttr --aug heavy --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs 21 --seed 1047 --save_epochs 6-21 --workers 10 --store ~/fleet_store --labels ~/r7t_labels/real/labels_r7t_b.csv --folds ~/rsna_train/folds5.csv --exclude ~/rsna_train/gold74_eval_set.csv --eval_labels ~/rsna_train/llm_labels_v4_blend.csv --gold ~/rsna_train/gold74_eval_set.csv --fold F --out ~/fleet_runs/fF_s1047_ttr_r7tb --grad_ckpt 0
```
`queue_runner.sh --check`: 5 lines, 0 BAD per sheet (MEASURED); today each would DEFER (labels file missing) and its gate
reads MISSING, so even an installed sheet cannot start before the REAL build passes. Install (main session) with
`for F in 4 3 2 1 0; do bash ~/queue/qctl.sh top "$(sed -n "$((F + 1))p" ~/r7t_labels/sheets/queue_r7tb.txt)"; done`
(puts f0 first, ahead of the L13 filler seed). Install only AFTER `on_r7oof.sh` printed PASS: the gate's `via` runs once per runner, so an early FAIL verdict would stick. Time: 5 x 3.52 h = ~17.6 h (MEASURED rate) + ~40 min CPU gold inference.

### 4b. Rented 4 x RTX 4090 (RSNA-only), `kits/vast_r7t/`
Port of `session_2026-10-09/vastS1044` (box push with OAI-path refusal + md5 parity, parity.py box vs instance, lanes,
laptop watcher, on-instance self-destroy guard) with: folds 0-4, seed 1047, the gated label table (the push refuses a SIM
table, a table mentioning "oai", or a table without a PASS verdict), offers with 2-8 GPUs, cap $15 / floor $2, and a
**BOX-side fetch loop** (lesson 10-10 / Raptor 10-13): the box pulls + verifies each finished fold and writes
FINAL_FETCH_OK, so a dead laptop session cannot lose the run. Commands and estimates: `kits/vast_r7t/RUNBOOK.md`.

| Plan | Wall (MODEL) | Cost (MODEL, 10-13 offers MEASURED) |
|---|---|---|
| 4 x 4090, T_b 5 folds (2 rounds) | 4.1 h mid / 4.9 h worst | $7.5-7.7 / $9.1-9.3 + ~$0.1-0.3 data-in |
| 2 x 4090, T_b 5 folds (3 rounds) | 5.7 / 7.0 h | $5.5-9.4 mid by host |
| 4 x 4090, T_b + T_a (two kit copies, 10 folds) | 5.7 / 7.0 h | $10.6 / $12.9 |

## 5. Time line to an LB read (MODEL)

R7-OOF lands (job 1; needs Eesh's OK for the multi-hour Kaggle run) -> `on_r7oof.sh` 1 min -> rental 5 h or box 18 h ->
`student_gate.py --infer` ~45 min -> main session: OAI-id scan, new private dataset with the 5 student checkpoints,
V224-F kernel with F = .5 K14 + .5 S, commit test, runtime check -> submit -> ~5-7 h hidden scoring.
With the R7-OOF on 10-15 evening: rental route read ~10-16 evening; box route ~10-17. Both before the 10-20 freeze.

## 6. Open items for the main session

1. Eesh's OK for job 1's full R7-OOF Kaggle run (the only blocker).
2. Choose the route: rental (fast, ~$8-10, needs Eesh's spend OK + Vast 2FA) or box (free, ~18 h, preempts the L13 filler).
3. After the student gate: dataset + kernel are main-session work (this job never creates datasets or kernels).
