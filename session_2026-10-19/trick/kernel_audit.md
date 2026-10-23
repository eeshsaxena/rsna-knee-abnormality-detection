# kernel_audit.md: end-to-end audit of c4 (57343435, public .958) for silent bugs and lossy steps

10-19 ~21:30 IST, Claude subagent. RESEARCH + MEASUREMENT ONLY: nothing built, pushed, committed, submitted or created; no GPU,
training or box job; no ssh used; no public code pulled or run. No PREREG / HANDOFF / PROGRESS / scheduled-task edits.
Tags: MEASURED (computed here or read from our own files, file named), CLAIMED (a note or third party says so, source named),
MODEL (arithmetic on stated assumptions), GUESS (judgement).

Audited: `session_2026-10-18/a18/kernel_v224r6a25c4j/rsna_sub_v224r6a25c4j.ipynb` (19 cells; cells dumped to the scratchpad
and the 7 embedded R6 sources extracted with `ast.literal_eval`, the S6F driver decoded from its base64; nothing executed).
Line numbers below are within each cell's source (cell N, l.X), or within the embedded file (compact_infer.py l.X, the file
the kernel writes to /tmp/probe/probe_src). Commit outputs: `session_2026-10-18/a18/out_v224r6a25c4j_v1/`.
Scripts and logs of this audit: `session_2026-10-19/trick/work/audit/` (r6_dump_audit, tie_pure, stacker_g58, member_g58).

## 1. Verdict
**The hypothesis "c4 has a silent bug or a lossy step costing score" is not supported. Every check came back clean or ~0.**
- No column, label or ID misalignment in any member (MEASURED, sec 3).
- The R6 flip TTA is correct (MEASURED: mirror outputs agree with the plain pass only after the Medial/Lateral swap).
- The float32 saturation of R6's probabilities (up to 18 % of studies tied at the top) costs about 1e-5 macro (MEASURED).
- The two port-fidelity references we hold match: S6fall to the author to 1.8e-3, rank rho .9998. R6 sigmoid(z) equals the served p to 1.2e-7 (MEASURED).
- All 54 committed c4-lineage runs (68 logs incl. 14 local mocks) have 0 fallbacks, 0 OOM, 0 compile or decoder fallbacks, and no unintended canary. These runs only see the 3 visible studies (MEASURED).
- The one real "lossy step" is R6's report-fitted Baker's state stacker. It costs R6 alone -.011 Baker's AUC on gold58. Inside c4 it is
  worth -.00015 macro on gold58 (P>0 .23), i.e. removing it does not help (MEASURED).
- Two genuine but practically dormant defects exist (sec 4: Raptor blank-study path, fallback-row spread). Each matters only if a
  hidden study fails, and nothing indicates any does (MODEL: expected gain ~0).
- **Nothing here is a buildable trick worth >= +2e-4. Verdict DEAD for this lens.** The rank slip is the crowd moving, not a
  pipeline leak (GUESS, consistent with THINK_1019 sec 1).

## 2. Pipeline as built (MEASURED, code read)
| stage | cell | what | output | failure rule |
|---|---|---|---|---|
| J decoders | 2 | pylibjpeg cp313 wheels on PYTHONPATH after a self-test + test-data preflight | dec_receipt.json | any failure: unchanged base, DEC_FALLBACK |
| Raptor r384 | 4-5 | nartaa CoAtNet, 94 windows, 320 crop of the 384 render, plain + anatomical mirror (no label swap), fp16 | raptor_raw.csv (probs), raptor_sub.csv (ordinal ranks) | a study that raises stays 0.5 in all 12 (cell 4 l.536, l.570) |
| R6 | 7-9 | heliosli compact ConvNeXt-S x4, v4 views, flip TTA forced (cell 8), fluid stacker (Effusion), state stacker (Baker's) | ours_sub.csv via probe_dual (%.9g) | unscored study = column mean (probe_dual l.128-138); cell 14 re-detects it from worker predictions.csv |
| r224 | 11-12 | nartaa 224 / 0.8 crop, plain only | raptor224_sub.csv (ordinal ranks) | skipped / failed: Fracture canary (cell 15) |
| stage 3 | 14 | per label `wr*rr + (1-wr)*ro`, rr = .75 pct(r384 raw) + .25 pct(r224 rank), each over Raptor's non-failed rows | submission_v224flip.csv (float32) | Raptor-failed rows take R6 rank, R6-failed take Raptor rank, both 0.5 |
| S6fall | 16-18 | nartaa E_prior_s42_fall via k1_b_infer (heliosli compact_infer 34f433fe, NO flip), own subprocess, guard 5.75 h | s6f/B/B_probs.csv (NaN = unscored) | coverage < .98 or rho floor: ACL canary |
| final | 18 | .8 pct(stage 3) + .2 pct(S6fall over its scored rows; unscored rows take the base rank) | submission.csv | |

## 3. Checks that came back clean
### 3.1 ID / column / label alignment (MEASURED)
- Cell 14 l.20-25: every frame is checked against test.csv ids, then `set_index('StudyInstanceUID').loc[test_ids, labels]`, so
  alignment is by name. The label list comes from sample_submission.csv.
- Cell 18 l.55-61: B_probs.csv is checked for duplicate or missing ids and indexed by name.
- probe_dual l.103-138 writes in the sample_submission order and reads worker rows by header name.
- Commit run: Baker's (apostrophe) present in all 6 CSVs. scored 3 / 3 for each member (the `r[L]` KeyError path would have dropped rows).
- Label ORDER inside each model, gold58 per-label AUC (member_g58.log): a permuted column would show ~.5; none does.

| label | r384 | r224 | R6 | S6fall ours | S6fall nartaa |
|---|---|---|---|---|---|
| ACL | .9645 | .9767 | .9963 | .9890 | .9890 |
| MCL | .9433 | .9705 | .9637 | .9660 | .9660 |
| Medial Meniscus | .9820 | .9760 | .9712 | .9603 | .9603 |
| Lateral Meniscus | .8957 | .8609 | .9155 | .9354 | .9354 |
| Medial OA | .9814 | .9736 | .9907 | .9814 | .9814 |
| Lateral OA | .7756 | .7853 | .8646 | .8511 | .8511 |
| PF OA | .8597 | .8507 | .9163 | .9022 | .9022 |
| Effusion | .9727 | .9665 | .9453 | .9491 | .9491 |
| Synovitis | .8292 | .7826 | .8590 | .8268 | .8268 |
| Baker's | .9764 | .9565 | .9692 | .9801 | .9801 |
| Contusion | .9352 | .9379 | .9595 | .9339 | .9339 |
| Fracture | .9583 | .9431 | .8875 | .9014 | .9000 |
| macro | .9228 | .9150 | .9366 | .9314 | .9313 |
No duplicate rows, no all-0.5 rows in any gold58 serve. r224 / r384 npy row order == gold58 order (MEASURED).

### 3.2 R6 flip TTA correctness (MEASURED, r6_dump_audit.log, box gold58 run_b dump = exact c4 cells 7-9)
- Mean over the 4 folds of the per-fold z: `z = (z_plain + z_mirror_raw[perm]) / 2`, MCL from the plain pass; reconstruction maxdiff 9.5e-7.
- Spearman(plain, mirror) on gold58:

| pair | swapped (as served) | unswapped | AUC plain / mirror-swapped / mirror-unswapped |
|---|---|---|---|
| Medial Meniscus | .991 | .344 | .970 / .962 / .668 |
| Lateral Meniscus | .989 | .320 | .923 / .912 / .595 |
| Medial OA | .989 | .829 | .986 / .992 / .864 |
| Lateral OA | .988 | .798 | .872 / .855 / .780 |
- The swap (perm [0,1,3,2,5,4,...]) is right. Without it the M/L labels would collapse. MCL's mirror pass is the LCL side (AUC .807), and
  it is correctly kept from the plain pass only.
- The flip axis is right: a wrong axis would decorrelate every label, and ACL / PF OA / Contusion rho is .996-.999.
- Flip on vs off for R6 alone on gold58 is ~0 net: MM +.0012, LM -.0075, MOA +.0047, LOA -.0077, Syn +.0024, Bak -.0018.
  Flip is a variance reducer here, not a gold58-visible gain (MEASURED).
- Raptor's mirror has no label swap (cell 4 l.270-283). levers.md (CLAIMED, 10-17) says Raptor trains on un-canonicalised L and R
  knees. Note the sagittal "mirror" only reverses the 3 channels of each window: windows keep their slot position, so it is a partial
  mirror. nartaa claims +.001 LB for it (CLAIMED, thread 746792). Not testable without a GPU; low value (sec 6).

### 3.3 float32 / fp16 saturation and ties
- R6 probabilities are `torch.sigmoid(zmean)` in float32 (compact_infer l.1172); MEASURED |p - f32 sigmoid(fold-mean z)| <= 1.2e-7.
  Every z > 16.6 becomes exactly 1.0, and z in [15, 16.6] falls on ~17 float32 levels below 1.
- Size (MEASURED, tie_pure.log, R7 OOF 4,349 non-gold studies, same worker family): p >= 1 - 1e-6 in 17.9 % of studies (Medial Meniscus),
  12.3 % (PF OA), 12.8 % (Effusion raw), 8.6 % (Medial OA). Tied cells 25 % for MM.
- Cost, pure tie effect, p32 vs z (non-stacker z for every label):

| ruler | single R6-family leg (macro z - p32) | .5/.5 rank blend with a real partner (our S6 s1044 OOF) |
|---|---|---|
| v2 | -4e-6 | -1.2e-5 |
| v4 | +7e-6 | -0e-6 |
| sol56 | +6e-6 | -1e-6 |
- Gold58, R6 alone: identical AUC for all 10 non-stacker labels. The top ties are all positives (MEASURED).
- **Closed: ~1e-5.** This confirms the earlier tie_loss.py closure with the stacker confound removed. The old -.0005 compared stacker p
  with raw z on Effusion / Baker's.
- Raptor logits come out of fp16 autocast. Its ordinal `rankpct` (argsort().argsort(), cell 4 l.352) only matters for exact ties, and
  r224 ties are label-independent, so the AUC effect is 0 (CLAIMED levers.md 10-17, agreed, MODEL).
- Stage-3 float32 output re-ranked in cell 18: distinct float64 blend values are >= ~1e-4 apart at n ~1,300, far above the float32
  step of 6e-8 (MODEL).

### 3.4 Failed rows, canaries, timeouts (MEASURED from logs)
- Scanned all 68 kernel logs (54 real commits + 14 local mocks) under session_2026-10-15..19 (`rsna-sub-*.log`). Markers searched:
  BLEND_FALLBACK, FALLBACK (, CANARY_FIRED, R224_SKIPPED / TIMEOUT / UNUSABLE, S6F_SKIPPED / CANARY, COMPILE_FALLBACK,
  DEC_FALLBACK, J_UNAVAILABLE, WORKER_FAILED, Traceback, stack_fallback, OutOfMemory.
- Hits: only the intended control canaries (v224fctl) and the two mock canary tests. c4 commit: raptor_failed 0, ours_failed 0,
  R6 fallback 0 (both workers), S6F coverage 1.0, rho_mean .78, J DEC_OK, compile active (|dp| 2.1e-4 vs eager).
- Limits: commits only see the 3 visible studies, so hidden fallback counts are unobservable. Bounds: S6F skipped or < 98 % covered
  would fire the ACL canary (-.03), which c4's .958 rules out. R6 > 50 % unscored would raise. Hidden latency <= 6h33 < every guard
  (S6F start 5.75 h is checked at its start; R6 deadline 8.3 h).
- The box run_a of R6 on gold58 lost 21 of 58 studies to CUDA OOM: `Tried to allocate 1.06 GiB ... 799 MiB free ... this process
  has 7.39 GiB`, i.e. the V100 was shared with another job. run_b, alone on the GPU, scored 58 / 58.
  - On Kaggle each R6 worker owns a 15 GB T4 (CUDA_VISIBLE_DEVICES per worker, stages sequential, nvidia-smi 0 MiB after Raptor), so
    the same peak fits (MODEL). This is the only plausible hidden R6 failure mode found.
  - Its cost would be "Raptor-only rank for those rows" (sec 4.2), not a crash.

### 3.5 Port fidelity
- S6fall: our serve (n42gpu) vs nartaa's published gold58: max |d| 1.83e-3, min per-label Spearman .99982, macro .9314 vs .9313
  (MEASURED). The commit B_probs p == sigmoid(z) (MEASURED, values in the commit CSV).
- R6: box gold58 serve = exact c4 cells 7-9 (receipt pins cell7 / cell8 sha, bundle and source shas). Gold58 macro .9366 vs R7 OOF .9193
  (MEASURED, R6_gold58_serve_receipt.json). Parity vs heliosli's 0-9576 commit output on the visible studies was checked when R6 was
  ported (CLAIMED, HANDOFF 10-15/16). gold not in R6 training (receipt gold_in_training 0).
- Raptor r384 / r224: the commit gate reproduces HW v1 raptor_sub / raptor224_sub at maxdiff 0 and raw at 6.9e-5 (MEASURED,
  gate_v224r6a25c4j_v1.txt). No independent author gold58 file for Raptor exists locally. The kernel-serve gold58 files come
  from the kernel's own cells: r384 on CPU fp32 instead of T4 fp16, r224 on GPU (make_runs.py).

### 3.6 Decoders
- JPEG Lossless was the only silent-drop syntax: Raptor zero-fills, R6 / S6fall drop the series. The J cell fixes it, and outputs on
  uncompressed data are byte-identical with and without it. Hidden exposure is <= 2 %, likely ~0: train 24,371 / 24,371 series are
  uncompressed (MEASURED earlier, session_2026-10-15/jpeg/RESULT.md).
- J preflight probes only the first and last file of each series dir (cell 2 l.96-104), but the decoders are installed globally once it
  passes, so mid-series JPEG files decode too (MEASURED code).

## 4. Defects found (real, but dormant: expected gain ~0)
### 4.1 Raptor blank-study path is not flagged as a failure (cell 4 l.463-468 + l.248-257; cell 14 l.37)
- A study whose preprocessing raises gets a zero volume and mask, but is NOT marked failed.
- `_eval_centers` then falls back to `valid = arange(3)`, so all 94 windows are window 1, and the CoAtNet returns a constant garbage vector,
  not 0.5.
- `rap_fail = (raw == 0.5).all(1)` therefore misses it, and stage 3 blends a garbage Raptor rank instead of R6-only.
- The same applies to a study whose every slice fails to decode. The J cell removes the known cause.
- Frequency on hidden: unknown, likely ~0 (no FAILED line in any commit or gold58 run; MEASURED logs).
- Fix if ever rebuilt: mark `mask.sum() == 0` rows as 0.5 in cell 4 (one line). Expected gain ~0 (MODEL). Not worth a slot.

### 4.2 Fallback rows sit at the extremes (cell 14 l.76-77)
- A failed row takes ONE model's percentile (`b[our_fail] = rr`), while normal rows hold a 2-model average, which has a smaller spread.
- So an R6-failed study that Raptor ranks at .99 lands above every blended study. A mild over-extremisation of fallback rows.
- Harmless at 0 failures (all evidence).
- A fix (map rr through the empirical CDF of b on the non-failed rows) only matters if R6 OOMs on hidden studies (sec 3.4). ~0 (MODEL).

### 4.3 Stale documentation (no score effect)
- Markdown cells 0, 1, 10 and 13 still describe the R7 bundle and A224 .35.
- The cell 14 l.4 comment ("Medial OA / Lateral OA / PF OA 0.50") contradicts l.11.
- Private kernel only, cosmetic.

## 5. The one real lossy step: R6's report-fitted Baker's state stacker (and the Effusion fluid stacker)
- **Mechanism (MEASURED, session_2026-10-15/r6v1/raw/state_stacker.json).** Baker's z_out = -0.96 + 0.14 z + a coefficient per
  report-language state: none -2.31, small -0.95, present_unspecified +1.29, moderate +5.08, large +2.58, ruptured +5.33.
  - It was adopted on report rulers (explicit +.038, corrected +.0077, cross-fitted).
  - It encodes how REPORTS grade cysts (a "small" cyst ranks well below an unspecified one). Expert image labels (the hidden test regime)
    should not care.
- **Gold58, single model (MEASURED, stacker_g58.log):**

| | served (stacker) | raw flip-TTA z |
|---|---|---|
| R6 Baker's | .9692 | .9801 (+.011) |
| R7 Baker's (its own stacker) | .9384 | .9710 (+.033) |
| R6 Effusion (fluid stacker) | .9453 | .9478 (+.0025) |
| R7 Effusion | .9416 | .9366 (-.005) |
- **Inside c4 (MEASURED, nested replay on the real gold58 serves; 4,000 bootstraps; Baker's 12 pos):**

| R6 change | c4 macro delta | boot sd | P > 0 |
|---|---|---|---|
| Baker's raw z | -.00015 (Baker's -.0018 = one pair flip) | .00044 | .23 |
| Effusion raw z | +.00000 | .00026 | .30 |
| both | -.00015 | .00054 | .31 |
- c4 already blends R6 Baker's with r384 (.976) and S6fall (.980). At a .981 ceiling with 12 positives, one pair is .0018.
- **Not a trick:** a mechanism plus single-model support, but zero blend support. Do not spend a slot (GUESS: P(true private gain > 0) ~.45).
- **Side fact (MEASURED, R7 OOF + train_series + _reports_lang):** the fluid stacker's "missing" path (no fluid-sensitive sagittal series,
  b_miss 2.30 vs a mean .64*lv of 1.15 on present studies) is effectively a German-site offset.
  - German studies: 93.5 % missing. Every other language: <= 2 %. All train studies have all 3 planes.
  - Visible test study ...10047035 takes this path: R6 puts its Effusion last of 3, S6fall first.
  - If a NEW hidden site also lacks a fluid-sensitive sagittal series, R6 gives it a report-learned +1.1 logit bump on Effusion. Direction
    in the expert regime: unknown (GUESS). Site offsets were already LB/gold-negative (closed 10-07), so this is a risk note, not a lever.

## 6. GPU measurements that could settle what is left (specified, NOT run)
1. **Hidden-like R6 failure rate on T4** (sec 3.4 / 4.2).
   - Run the exact c4 cells 7-9 on a Kaggle T4 x2 kernel over ~400 train studies, the largest by DICOM file count (they set peak
     memory), with no other stage.
   - Read: fallback reasons, peak `torch.cuda.max_memory_allocated` per study (add one print), windows per study.
   - ~1.5-2 h T4.
   - Value: only if it shows OOM > 0 %. Then the fix is `--chunk 96` or a CPU-side mirror copy, plus the 4.2 rank mapping. Prior
     P(OOM > 0) ~.05 (GUESS).
2. **Raptor mirror semantics** (sec 3.2): plain vs mirror vs label-swapped mirror on gold58 + 300 OOF-free train studies,
   cell 4 with arm_probs split by pass. ~1 h V100. Value low: nartaa's +.001 LB claim already covers it.

## 7. What this means for Eesh
- The kernel is doing what it says: no hidden leak, no misalignment, no broken TTA, no lossy numerics worth more than 1e-5.
- The slipping rank is the field catching up at .959. It is not c4 leaking score.
- "The missing trick" is not inside this notebook. That agrees with THINK_1019's conclusion: stop, or spend one slot on a near-zero-mean
  probe, and select A = best-id, B = R6 by hand.
