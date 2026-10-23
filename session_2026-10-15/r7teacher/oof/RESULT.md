# R7-OOF teacher notebook: RESULT (10-14 IST, subagent of the R7-TEACHER workflow). No em dashes.

Location: `session_2026-10-15/r7teacher/oof/` (this agent's subdir).

Tags: MEASURED (we ran it / read it), MODEL (computed from MEASURED inputs), GUESS, CLAIMED (third party wrote it).

## 0. Bottom line

- **Fold mapping PROVEN.** R7's trainer (sha / md5 = the bundle pins) assigns `fold = int(sha1(uid).hexdigest()[:8], 16) % 4`, a pure
  function of the study id, and fold k's model trains on every other fold (MEASURED code + manifest). Held-out fold of all 4,407 train
  studies: `r7_fold_map.csv`, counts [1043, 1146, 1137, 1081]; gold58 and 5 other non-pool studies were never trained by any fold.
  Empirical check on 36 non-gold studies run through all 4 fold models: the hash-fold model is the outlier in 20 / 36 (null 9; p 9e-5)
  and the least aligned with confident report labels (fold-shuffle p .0003), for every fold (MEASURED). The bet is alive.
- **Notebook built and commit-tested:** private `eeshsaxena/rsna-r7-oof` reuses the R7 asset cell verbatim and runs R7's own
  `compact_infer.py` unchanged, one held-out fold model per study, plain pass, resumable shards, deadline guard, full-precision
  `r7_oof.csv` (12 p, 12 raw logits, stacker terms, 84 state read-outs). Commit v1 (8 studies): 8 / 8 scored, each by exactly its hash
  fold, values finite in [0, 1], all checks PASS, same model + same study in two separate runs: max |dz| 0 / 0 / 0 / 0.003 (MEASURED).
  Commit v2 (proof-only, 36 studies) was added because 4 proof studies could not settle the mapping.
- **Time:** in steady state a study costs ~17 s of single-process CPU (decode + render + anchors) and 2.6 s of T4 per fold model
  (MEASURED); the earlier "37-50 s decode" was decode running beside the compile check. Full 4,407-study OOF on 2 x T4: **~6.2 h
  (5.3-8.7 h) MODEL**, recommended as 2 sessions of ~3 h (shards built, `kernel_r7oof_full_s0of2/` ready). A 4 x 4090 rental computes it
  in < 1 h (MODEL / GUESS) but needs ~550 GB of train DICOMs moved first (Eesh's own token or hours of CPU-kernel tarring): not faster overall.
- **Not launched (needs Eesh's explicit OK):** the full run, ~6-9 GPU-hours of his Kaggle quota, one GPU slot at a time.

## 1. Fold mapping: PROVEN (code + manifest + bundle files; empirical check in sec 3)

`prove_fold_map.py` (log `prove_fold_map.log`, 23/23 PASS) on local copies whose hashes equal the bundle manifest pins:

- The trainer we read is the one the bundle pins: `compact_train.py` sha256 04df7e2e... = manifest `src_sha256`, md5 8622009e = manifest
  `trainer_md5` (MEASURED). The R7 notebook embeds the same file (its asset cell asserts both).
- The fold rule is one line, assigned once in the whole trainer (MEASURED):
  `pool_fold = np.array([int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % K for u in pool])`, `K = a.folds`.
  It is a pure function of the StudyInstanceUID: pool membership, targets and row order do not change a study's fold.
- Fold k's training set: `tr = np.flatnonzero(pool_fold != k)` (holdout_fold None); its eval set `ho = pool_fold == k`; the file is
  `"%s_s%d_f%d.pt" % (arm, seed, k)` (MEASURED code). Manifest: folds 4, holdout_fold null, seeds 1, seed_base 0, models
  `E_lrprior_s0_f0..f3.pt`, include_review_holdout true (review rows are in the pool, same hash), warm_start / reeval_run / px_extra null,
  gold_in_training 0, pool 4,344, gold 58, feature (px) studies 4,403 (MEASURED manifest).
- Gold studies are never in the pool (`assert not (set(pool) & set(gold))`), so any fold is out-of-fold for the 58 gold studies; the
  same holds for the 5 non-gold train studies outside the pool (4 without px rows + 1 blanked report; MODEL from 4,407 - 4,403 and
  4,403 - 4,344 - 58).
- The bundle's two stackers were fit on the run's OOF file `oof_E_lrprior_s0.npy` (md5 aeca67f8) with 4,344 pool rows and 4-fold
  cross-fitting (MEASURED json): heliosli's own OOF exists and is exactly what this notebook reconstructs (single held-out fold model per
  study, eval stride 1), up to decode differences between his training cache and the inference decode.
- Held-out fold of every train study: `r7_fold_map.csv` (4,407 rows: uid, r7_fold, is_gold, n_series, shard_of_2/3/4).
  Counts over all 4,407: **[1043, 1146, 1137, 1081]**; gold [14, 11, 19, 14]; non-gold [1029, 1135, 1118, 1067] (MEASURED).
- Limit of the proof: the manifest stores fold 0's run_config only (`only_folds "0"`); folds 1-3 were separate runs of the same trainer.
  Their K = 4 and hash rule are implied by the file names and the single trainer, not printed. The commit's proof test (sec 3) checks
  each fold empirically.

## 2. The notebook `eeshsaxena/rsna-r7-oof` (private; builder `build_r7oof.py`)

Inputs: competition data (train split), `heliosli/rsna-knee-compact-r7cnxfsss` (R7 bundle), `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`
(code + wheels + anatomy.pt); internet off; 2 x T4; the R7 notebook's pinned docker image. No Raptor, no nartaa dataset.

| cell | content |
|---|---|
| 0 | markdown (purpose, credits) |
| 1 | config baked by the builder (mode, shard index / count, explicit uids, skip list, time budget, chunk, loader workers, proof uids); `R7OOF_<KEY>` env vars (json) override; sets `NB_T0` (the R7 cell counts from it) |
| 2 | **the R7 notebook's asset cell verbatim** (heliosli/0-957-latest-dataset-score-in-one-click cell 4 = varshitha V224 cell 6; text sha256 cb168cfe..., asserted by the builder): SOURCES + sha asserts (compact_infer 8401812c, compact_train 04df7e2e, px_cache, side_common, knee_anchor_bank_v2, probe_*), mainline code sha asserts, pinned wheels (timm 1.0.25 / pydicom 3.0.1), bundle copy with per-file sha256, anatomy.pt sha, WORKER_ARGS built from the manifest (eval stride 1, views v4, side check, compile, fluid + state stackers, NO flip because manifest flip_tta = false) |
| 3 | controller: asserts the fold rule line and the train/holdout line inside the embedded trainer text, folds 4 / holdout None / models f0-f3, no `--flip-tta`; reads train.csv (4,407), selects the shard (`md5(uid)[:8] % shard_count`, independent of the sha1 fold rule) minus the skip list, groups studies by held-out fold, splits each fold into chunks of <= `chunk`, and runs `compact_infer.py` **unchanged** once per chunk with `--models <folder holding only E_lrprior_s0_f{k}.pt>` (the worker globs `<arm>_s*_f*.pt`), `--split train`, `--deadline` = session start + budget, the R7 WORKER_ARGS unchanged, plus `--dump-logits --dump-state` (read-outs only). Same per-worker command and env as R7's `probe_dual.py` (which itself cannot run train: it needs sample_submission ids). Two GPU threads pull jobs; a job starts only if >= `min_job_start_min` remain; the worker stops after the study in flight at the deadline and still writes all its files. Optional proof jobs: the unchanged 4-model path on a few studies. CPU busy / load sampled every 60 s. |
| 4 | merge + checks -> `r7_oof.csv`: uid, r7_fold, shard, job, precision, `p|<12>` (non-stacked classes = the worker's float32 sigmoid written %.9g = exact; Effusion / Baker's = the stackers' p_out at %.17g), `z|<12>` raw fold logits (float32, exact), fluid / state stacker terms, 84 state read-out columns (`state|class|category`, `bin|binary|class`); `r7_oof_status.csv` (every requested study: ok / failure reason / not reached); `r7_oof_receipt.json` (checks, per-job timings, CPU samples); `r7_proof.csv`. Checks: receipt models == [fold k file] per job, every row's fold == sha1 rule, no study twice, all p finite in [0, 1], p == sigmoid(z) (non-stacked), p == stacker p_out (stacked). |

Resume: a later session is built with `--skip-from <earlier r7_oof.csv ...>` (studies with full-precision rows are skipped; failed or
unreached ones run again). `merge_sessions.py` joins sessions (fold re-check, conflict check, missing list).

Logic tested before any GPU use (box CPU, `mock/`): cells 1/3/4 with a fake worker that writes the real worker's file formats:
commit mode 8/8 and proof 4/4; a 2,248-study shard; a deadline cut (657 scored, rest "not started") + resume with the skip list;
3-session merge with 0 conflicts (MEASURED, mock only).

## 3. Commit tests (v1: OOF path on 8 train studies; v2: proof-only, 36 studies)

**Commit v1** (`kernel_r7oof_commit`, pushed 07:42 UTC, COMPLETE 07:55; notebook 706 s; output `out_commit_v1/`): 8 non-gold train
studies, 2 per hash fold (5-6 series, md5 order), plus the unchanged 4-model path on 4 of them (1 per fold). All MEASURED:

- 8 / 8 scored, 0 fallbacks, 2 per fold; every OOF worker receipt lists exactly one model file and it is the study's hash fold
  (`E_lrprior_s0_f{sha1(uid)[:8] % 4}.pt`); no flip block in any receipt (plain pass); compile active, gate "compiled == eager within
  0.001" in every worker; side check ran on all studies (0 flips); anchors skm_tea_joint / avg2 / slab_prior.
- In-notebook checks 16/16 PASS and the independent local re-check `verify_commit.py` ALL PASS (`verify_commit_v1.log`): one fold
  model per study and it is the hash fold, all 96 probabilities finite and in [0, 1], `r7_oof.csv` logits == worker dump, non-stacked
  p == predictions.csv exactly, stacked p within 4.5e-10 of predictions.csv (%.17g vs %.9g), p == sigmoid(z) within 7.8e-8 (float32).
- **Deterministic per study:** the same fold model on the same study in two separate worker processes (the 1-model OOF job and the
  4-model proof job, each decoding the study itself): max |dz| = 0 on 3 of 4 studies and 0.0030 logit on the 4th (compiled kernels
  for different batch layouts; the compile gate's own tolerance is 1e-3 in probability). Run-to-run noise is far below anything a
  student can use.
- Probabilities are very confident (the E_lrprior arm: |z| mostly 5-16, e.g. p 1.000 / 0.000): a student should use them as ranks /
  logits with a temperature, not as raw soft targets (MODEL).
- **Proof on 4 studies (empirical fold check):** test A (the held-out model is the outlier among the 4) 0 / 4, which carries no
  information here: fold models differ by 1.5-3.9 logit RMS on every study whether or not they trained on it (MEASURED), so the outlier
  test has no power. Test B (margin toward our confident LLM label, rank of the hash-fold model among the 4; null mean 1.5): mean
  **1.00**, rank counts [15, 7, 5, 5] over 32 cells, all 4 studies below 1.5; per model, its rank when it is the hash fold vs when it
  trained on the study: f0 1.10 vs 1.82, f1 0.20 vs 1.44, f2 1.36 vs 1.52, f3 0.83 vs 1.89 (MEASURED). Direction is right for every fold,
  but 4 studies cannot prove it (fold-shuffle p .085, the minimum possible with 4 studies is ~.04). Hence commit v2.

**Commit v2 = proof-only** (`kernel_r7oof_proof`, as pushed in `pushed/v2/`; pushed 07:59 UTC, COMPLETE 08:11; notebook 615 s;
output `out_proof_v2/`): the unchanged 4-model R7 path on 32 new non-gold train studies (8 per hash fold) + 4 gold controls.
36 / 36 scored, 0 fallbacks. `proof_analysis.py` (logs `proof_analysis_v2.log`, `proof_analysis_v1v2.log`), all MEASURED:

| test (non-gold studies) | v2 (32 studies) | v1 + v2 (36) | null |
|---|---|---|---|
| A: hash-fold model is the outlier of the 4 (largest RMS logit distance to the other three) | **20 / 32**, binomial p 8e-6 | 20 / 36, p 9e-5 | 1 / 4 per study |
| A per fold (v1 + v2) | f0 6/9, f1 4/9, f2 4/9, f3 6/9 | | 2.25 / 9 |
| mean RMS distance: hash-fold model vs the other models | | 2.93 vs 2.14 logit | equal |
| B: margin toward a confident LLM label (>= .9 / <= .1), rank of the hash-fold model among 4 | mean 1.30 over 246 cells | 1.26 over 278 cells | 1.50 |
| C: same statistic, null = hash folds shuffled across the same studies (keeps each model's calibration) | p .0016 | **p .0003** | |
| C per model: its rank when it is the hash fold vs when it trained on the study | f0 1.02 vs 1.41, f1 1.33 vs 1.71, f2 1.25 vs 1.53, f3 1.54 vs 1.63 | f0 1.03 vs 1.45, f1 1.25 vs 1.68, f2 1.27 vs 1.53, f3 1.48 vs 1.66 | equal |

Reading: on a study it trained on, a fold model agrees with the other three trained models and leans further toward the report label;
the model named by `sha1(uid)[:8] % 4` is the odd one out and the least label-aligned, for every one of the 4 folds. That is what the
mapping predicts and what a wrong mapping cannot produce. Gold controls (never trained by any fold) carry no such signal
(fold-shuffle p .33 vs true labels; 4 studies only). **Mapping: proven by code, manifest and data. The bet is alive.**

## 4. Time per study and the full 4,407-study run

MEASURED inputs (2 x T4 Kaggle session: 4 vCPU, 31.3 GB RAM, torch 2.11.0+cu128, timm 1.0.25, fp16):

| quantity | value | source |
|---|---|---|
| notebook setup to ASSETS_READY | 19-20 s | v1, v2 logs |
| compile check per worker run (the R7 gate: 6 shapes, compiled vs eager) | cold 220 s (1 model) / 262-273 s (4 models); warm, inductor cache from an earlier run in the session: 81-85 s / 118-122 s | v1, v2 receipts |
| loader time per study, steady state (after the compile; v2 studies 3-18, n = 32) | decode mean 11.6 s (median ~9.3, range 7-28), render ~2.9 s, anchors ~2.4 s: **~17 s per study per loader process** | v2 receipts |
| loader time per study while a compile check runs beside it | 41-64 s (this is why the 10-13 R7 commit showed 37-50 s) | v1 receipts |
| GPU per study | **1 fold model 2.55 s** (1.8-3.4, n = 8); 4 fold models 11.4 s (n = 36) | v1, v2 |
| 4-model path throughput, steady | 15 studies / 209 s per GPU = 6.95 s per study on 2 GPUs (GPU-bound); CPU busy .89-.91 | v2 |
| cold read of one study's DICOMs | 39.5 MB/s (51 MB in 1.3 s): decode is CPU-bound, not I/O-bound | v1 IO bench |

MODEL for the OOF path (1 fold model per study, 2 GPU workers x 2 loader processes = 4 decoders on 4 vCPUs): the GPU needs 2.6 s per
study, the loaders ~17 s per study per process, so the run is CPU-bound at about 4 vCPU / (17 s loader + ~2-4 s main-process CPU) per
study = **~5 s per study (range 4.3-7 s)**.

| run | studies | wall on 2 x T4 (MODEL) |
|---|---|---|
| everything in one session | 4,407 | **~6.2 h central (5.3-8.7 h)** + ~6 min setup / compile |
| 2 shards (`shard_of_2`, recommended) | 2,248 / 2,159 | ~3.1 h / ~3.0 h each (2.7-4.4 h), budget 8.5 h each: large margin |
| 3 shards | 1,448-1,500 | ~2.1 h each |

GPU quota: ~6-9 h of T4x2 session time in total (MODEL), on top of the ~0.4 h already used by the two commits (MEASURED: 706 s + 615 s
of notebook time plus boot). Chunk default 600 (one worker run per fold per 2-shard session: 4 runs, 2 per GPU, ~5 min of compile
overhead per session). Every run is deadline-safe (stop after the study in flight, all files written) and resumable (`--skip-from`).

**Rented 4 x RTX 4090 (RSNA data only):** compute is small (MODEL/GUESS: ~17 CPU-s per study at Kaggle-vCPU speed over >= 32 vCPU
= ~40 min, faster cores less; GPU ~1 s per study per 4090, ~20 min over 4 GPUs: ~0.5-1 h total). The binding part is data access:
the R7 path needs the raw train DICOMs, ~550 GB (MODEL: 4,407 x ~121 MB, box sample of 5), which reach a rental only by (a) Eesh
placing his own kaggle.json there and running `kaggle competitions download` (we never handle his token; GUESS 1-2 h at 100-300 MB/s,
plus unzip, >= 1.2 TB disk), or (b) the token-free route used for the 10-13 Raptor corpus: private Kaggle CPU kernels tar the DICOMs
into kernel outputs that the rental fetches by signed URL (~28 x 19 GB parts here vs 33 x 1 GB then: hours of CPU-kernel prep).
Cost ~$1.9/h (10-13 offer, MEASURED) x 3-4 h = ~$6-8 (MODEL) of the $17 credit. Wall time ~3-5 h with setup: **not faster than one
Kaggle session and riskier**; Kaggle is the recommended route. The rental would also need the R7 bundle + code dataset (public Kaggle
datasets, same access path) and the pinned timm 1.0.25 / pydicom 3.0.1 wheels.

## 5. How to run the full job (needs Eesh's explicit OK: multi-hour GPU on his quota)

Nothing below has been launched. Each step uses one of the 2 concurrent GPU slots on Eesh's account for hours.

1. Check slots: `kaggle kernels list --mine --sort-by dateRun` + `kaggle kernels status <ref>` for the top rows; at most 1 other GPU
   session may be running.
2. Session A (already built: `kernel_r7oof_full_s0of2/`, shard 0 of 2, 2,248 studies, budget 8.5 h, chunk 600, no proof jobs):
   `kaggle kernels push -p kernel_r7oof_full_s0of2` (same slug `eeshsaxena/rsna-r7-oof`; each push = a new version). Expected ~3.1 h (MODEL).
3. When it finishes: `kaggle kernels output eeshsaxena/rsna-r7-oof -p out_full_s0` (download BEFORE the next push: `kernels output`
   returns the latest version only). Run `python verify_commit.py out_full_s0` (same checks at scale).
4. Session B: `python build_r7oof.py --mode full --shard-index 1 --shard-count 2 --skip-from out_full_s0/r7_oof.csv` (the skip list
   only matters if A spilled over; shard 1 is disjoint), push, download. Repeat for any further shard. A session cut by the deadline is simply re-run with `--skip-from` all earlier csvs.
5. `python merge_sessions.py r7_oof_ALL.csv out_full_s0 out_full_s1 ...` -> coverage report + `r7_oof_ALL_missing.txt`; studies that
   fail twice stay missing (fall back to our own labels for those rows).
6. Two sessions can run at the same time only if no other agent needs a GPU slot (2 concurrent max): then wall time halves. Use a
   second slug for the second shard (`--slug rsna-r7-oof-b`), so the two runs are separate kernels.

## 6. Risks and caveats

1. **Teacher quality = one fold model, not the 4-fold ensemble.** Each OOF value comes from a single ConvNeXt-small fold model, the same
   object heliosli's stackers were fit on. His own numbers for exactly this OOF vs his explicit report ruler (CLAIMED, bundle
   state_stacker.json `auc_baseline`): ACL .985, MCL .976 (28 pos), Med Men .958, Lat Men .946, Med OA .965, Lat OA .929, PF OA .937,
   Effusion .980 (.982 with the fluid stacker), Synovitis .940, Baker's .890 (.936 with the state stacker), Contusion .953, Fracture .966.
   These are against his own labels, so they are an upper-side view of how well it ranks "his truth"; the gap between that truth and
   the hidden labels is unknown.
2. **Small leaks inside the deployed outputs (MODEL: negligible, recorded so they can be removed).** (a) The Effusion fluid stacker
   (4 coefficients) and Baker's state stacker (7) were fit on all 4,344 OOF rows including the study being scored; `r7_oof.csv` keeps the
   raw fold logit `z|<class>` and the stacker inputs, so a student can use the raw logit instead. (b) The head bias was initialised from
   pool base rates (all folds). (c) Same-patient studies (both knees, repeat scans) can sit in different folds; the hash is per study,
   not per patient. Not checkable without DICOM PatientID; it would make the teacher slightly optimistic on those studies only.
3. **Decode differences.** heliosli's OOF came from his training pixel cache (px384_v4 + sidefix); ours from the inference decode
   (`compact_infer.CStudies`, which he also used to build the sidefix cache). Small row-level differences are possible (GPU vs CPU anchors).
4. **Stacked classes use single-fold features.** The stackers were fit on single-fold OOF features (good match), but at test time R7
   applies them to the 4-fold mean; for a teacher table the OOF match is the right one.
5. **Folds 1-3 run configs are not public** (sec 1 limit). The empirical proof test covers it per fold.
6. **Kaggle constraints.** At most 2 concurrent GPU sessions on Eesh's account (other agents push commits too); a full run blocks one
   slot for hours. Kaggle weekly GPU quota (~30 h, GUESS for this account) is shared with the commit tests (RESEARCH_960 sec 2 D13
   budgets ~2.5-3 h for those).
7. **Timing on the commit is pessimistic per study** (compile checks run alongside the first decodes); the projection uses the
   steady-state part, see sec 4.

## 7. Files (all under `session_2026-10-15/r7teacher/oof/`; run every command from that folder)

| file | what |
|---|---|
| `prove_fold_map.py`, `prove_fold_map.log` | step 1 proof (23/23 PASS) |
| `r7_fold_map.csv` | 4,407 train studies: held-out R7 fold, gold flag, series count, session shards of 2 / 3 / 4 |
| `build_r7oof.py` | notebook builder (`--mode commit` / `--mode full --shard-index --shard-count --skip-from --proof-n --budget-h --chunk --slug`) |
| `cells/c3_controller.py`, `cells/c4_merge.py` | the two cells we wrote (cells 1 and 2 come from the builder / the R7 notebook) |
| `kernel_r7oof_commit/` | commit-test kernel (pushed as version 1; the local rebuild differs from v1 only in the build timestamp and one controller line, `if C['uids'] is not None`, needed for proof mode, no effect in commit mode) |
| `kernel_r7oof_proof/`, `pushed/v2/` | proof-only kernel (version 2; `pushed/v2` = pulled from Kaggle, identical cells except the timestamp) |
| `kernel_r7oof_full_s0of2/`, `kernel_r7oof_full_s1of2_parallel/` | full-run kernels, NOT pushed (shard 0 on slug rsna-r7-oof; shard 1 on slug rsna-r7-oof-b if both run at once) |
| `out_proof_v2/`, `proof_analysis.py`, `proof_analysis_v2.log`, `proof_analysis_v1v2.log` | proof commit output + the empirical fold tests |
| `out_commit_v1/` | its downloaded output (r7_oof.csv, r7_proof.csv, receipts, per-job logs and dumps) |
| `verify_commit.py`, `verify_commit_v1.log` | independent local re-check of the output + timing printout |
| `merge_sessions.py` | joins several sessions' r7_oof.csv, coverage + missing list |
| `mock/` | fake worker + harness used to test the controller / merge logic on the box CPU |
| `src/` | provenance copies: R7 notebook (pull 10-13), compact_train / compact_infer / probe_dual, R7 manifest, stacker jsons, state spec |
| `push_commit_v1.log`, `push_proof_v2.log`, `poll_commit.sh`, `poll_commit.log` | push + status trail |
