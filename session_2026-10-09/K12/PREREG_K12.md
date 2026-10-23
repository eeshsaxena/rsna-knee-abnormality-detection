# PREREG K12: does the OAI 224 set D-OAI REPLACE set D (or join it) in the FN11 fleet?

Written 2026-10-09 ~22:40 IST by Claude (subagent), BEFORE any D-OAI fold 1-4 checkpoint, log line, OOF or gold
prediction was read (f1 has been training since 19:12; f2-f4 follow). Do not edit after the first D-OAI f1-f4 read; append
a dated "RESULTS" section at the bottom. The sha256 of this file as written is recorded in `K12/PREREG_K12.sha256`.
No em dashes. Tags: MEASURED, MODEL, GUESS.

**Known at writing (disclosure).** Only fold 0 of D-OAI exists: the T12v00 probe `~/fleet_runs/f0_s1043_ttr_frac75oaiT12v00`
(17 epochs, grad-ckpt on; gate file `~/oai_build/out/gate2_T12_f0_s1043_ttr_frac75oaiT12v00.txt`, read 10-09 ~20:55):
raw best ep 14, cv .887764 vs its paired control D f0 .883777 (oof12 +.0040); single-member gold58 .9142 vs D f0 .9163,
gold74 .9082 vs .9083; ckpt args inspected (keys model / args / cv / epoch; decode_probe = 4 RSNA ids). The FN11 gate
numbers (C, D, E, E2) are known. Nothing of folds 1-4 has been seen (not even file names).

## 1. Candidate and question

- **D-OAI** = `f{F}_s1043_ttr_frac75oaiT12v00`, F = 0..4: TTR224, seed 1043, the set-D recipe, trained on the RSNA train
  folds + 2,581 OAI V00 knees with T12 labels (`labels_frac75oai_T12v00.csv`, union `~/union_store_t12v00`; OAI rows are
  fold 9 = never validation). Only labels + data differ from D (same seed, ttr, aug, heads).
- Base = **FN11 fleet B = C + D + E + E2** (20 members, gate PASS 10-09 20:41).
- **R (REPLACE)** = C + D-OAI + E + E2 (20 members). **A (ADD)** = C + D + D-OAI + E + E2 (25 members).
- K12 kernels are OAI-trained in two places (N and D-OAI): N pick only, never the clean pick (unchanged policy).

## 2. Fold 0 (17 epochs) vs folds 1-4 (21 epochs): decided now

Fold 0 is used **as is** (the probe's raw best checkpoint, ep 14 of 17). Reasons: set D, which D-OAI replaces, already mixes
17 / 21 / 22-epoch folds (D f0 / f1 are 17-epoch Kaggle runs); the f0 probe is the gated, paired checkpoint; a 21-epoch
re-run would cost ~6.3 h of the single V100 that the OAI 288 chain needs. The per-fold minimum epoch count is 14 (the D
rule). Grad checkpointing changes memory only (same model_sha, asserted by the picker).

## 3. Per-fold acceptance (mechanical, before any fleet read)

A fold is ACCEPTED only if all hold (`k12/box/k5_pick.py`, the K11 picker; box runs, so the K5 OOF-vs-ckpt mtime rule is
ENFORCED):
- every K10 picker check (closing line, one raw + one ema best file, cv == closing line, OOF rescored == ckpt cv to 1e-12,
  model_sha f141aeed136dfc01, ttr_size 224, fold / seed 1043, train_labels `labels_frac75oai_T12v00.csv`, sel labels v4,
  amp fp16, epochs header == ckpt args, min epochs 14);
- **K12-OAI-CLEAN**: the OOF csv has no `oai_` id and its id set == set D's OOF ids for that fold (RSNA val rows only);
  the checkpoint dict has exactly the keys model / args / cv / epoch, `model` holds tensors only, `decode_probe` keys are
  RSNA UIDs only, and no string anywhere in the dict (recursively, incl. nested containers) contains `oai_` followed by a
  digit, no container with > 64 string entries other than the model state, no pandas / numpy table object. Any hit = the
  fold FAILS loud and nothing of it leaves the box.
- **Sanity bar** D-OAI_F >= D_F - .005 (D = the paired same-seed control; D f0 .883777, f1 .890967, f2 .890903, f3 .876911,
  f4 .892889). No override.
- K12 needs **all 5 folds accepted** (it replaces a whole 5-fold set); otherwise "K12 NOT GATED" and FN11 stays the vehicle.

## 4. The gate (RSNA rows only; never anything on OAI rows; `k12/box/k12_gate.py`, sha256 in K12_RESULT.md)

Fleet = equal member mean of RAW probabilities (as the K5 worker). Same metrics as K11GATE:
- **G1 OOF**: fold-matched OOF macro AUC vs `~/rsna_train/llm_labels_v4_blend.csv` on the 5 x ~867 RSNA val rows (4,333;
  asserted disjoint from gold74 and free of `oai_` ids), masked macro AUC `tta_bench.macro_auc`; a fleet's row = the equal
  mean of its sets' fold-F rows.
- **G2 gold58**: pass A CPU fp32 predictions (`~/novel/gold_infer5.py`, sha256 sidecars), equal mean of ALL fleet members,
  masked macro AUC on the original_gold58 rows of `gold74_eval_set.csv` (held out of every run incl. the OAI unions).
- R passes iff dOOF(R - B) >= 0.0000 AND d58(R - B) >= -0.0005. A passes iff dOOF(A - B) >= 0.0000 AND d58(A - B) >= -0.0005.
- **VERDICT**:
  1. R passes and A does not -> **K12 = R (REPLACE)**.
  2. Both pass -> **R**, unless A beats R by a margin on both: OOF(A) - OOF(R) >= +.0005 AND AUC58(A) - AUC58(R) >= 0, then
     **K12 = A (ADD)** (25 members; the margin pays for the 5 extra members and the extra runtime).
  3. Only A passes -> **K12 = A (ADD)**.
  4. Neither passes -> **no K12**; FN11 stays the vehicle; stop after recording.
- Reported, never gating: gold74 deltas, paired study bootstrap SE (B 2000, seed 59), per-label deltas (OAI-supervised T12
  labels PF OA / Lateral OA / Synovitis / Medial OA / Effusion / Baker's singled out), D-OAI vs D per fold (OOF and gold58),
  mean per-label Spearman D-OAI~D (fold-matched).
- No gate on the F .40 / N .60 blend (N is in-sample on train and gold rows; FN11 PREREG sec 3). OAI rows are never scored.

## 5. Runtime (hidden rerun; rule: slow-case end <= 0.65 x 9 h = 5.85 h)

`k11/tests/runtime_k11.py` MODEL: R = 5 x 224 + 15 x 288 + N = FN11 = 3.08 h slow; A = 10 x 224 + 15 x 288 + N = 3.47 h slow.
Both PASS (decided now; a recheck is in K12_RESULT.md). The two-dataset merge adds seconds (symlinks + 5 sha256 of 107 MB).

## 6. Build rules fixed now

- Weights dataset `eeshsaxena/rsna-knee-cnxt-fleet-k12w` holds ONLY the 5 D-OAI raw best checkpoints (re-keyed
  `f{F}_s1043f75oai_raw_ttr224-...`) + a sha256 manifest + dataset-metadata.json. No code, no OAI data / images / labels /
  ids. The kernel attaches the FN11 dataset `-k11w` (unchanged) + `-k12w` + the N weights; a small K12-MERGE cell links
  both into one fleet directory (/tmp) whose manifest == the build's expected merged manifest, and checks every D-OAI file
  sha256 == the k12w manifest, fail loud.
- Kernels `eeshsaxena/rsna-sub-k12fn` (F .40 / N .60) and `eeshsaxena/rsna-sub-k12fn-n70` (F .30 / N .70); the member
  list is R or A per sec 4. Everything else byte-identical to the A0-tested FN11 twins except the merge cell, the fleet
  dataset path line, MEMBERS / TTA, TAG and naming (diff file in the build).
- No Kaggle create / version / push / submit by this agent. Box: CPU only, CPUs 34-39, nice 19, writes only `~/k12_test`;
  D-OAI checkpoints are copied to the laptop only after K12-OAI-CLEAN passed on the box.

## RESULTS (append below, dated; do not edit the sections above)

### RESULTS 2026-10-10 19:36 IST (gate run by the resumed subagent's detached box job; recorded by Claude, main session 19:45)

- **Acceptance (sec 3):** K12COLLECT FINAL 19:20. DO 5/5 ACCEPTED, all K12-OAI-CLEAN PASS (5/5). Raw best cv:

  | fold | cv | epoch | vs D | bar |
  |---|---|---|---|---|
  | f0 | .887764 | 14/17 | +.0040 | |
  | f1 | .892041 | 21/21 | +.0011 | |
  | f2 | .889377 | 18/21 | -.0015 | .885903 |
  | f3 | .878381 | 15/21 | +.0015 | |
  | f4 | .888938 | 17/21 | -.0040 | .887889 |

- **G1 OOF** (4,333 RSNA val rows): B .89171, R .89223, A .89211. dOOF R - B +.00052 (PASS), A - B +.00040 (PASS),
  A - R -.00012. Spearman DO~D .9577.
- **G2 gold58:** B .9219, R .9201, A .9212. d58 R - B **-.0018** (SE .0014, P>0 .095) FAIL (bar -.0005); A - B
  **-.0007** (SE .0014) FAIL; A - R +.0011.
- **VERDICT (sec 4 rule 4): NO K12. FN11 stays the vehicle.** Report-only: gold74 R - B -.0006, A - B +.0003.
- Report-only per-label pattern, DO vs D:
  - gold58: OAI-supervised labels UP (PF OA +.019, Baker's +.025, Lateral OA +.012); unsupervised labels DOWN (Lateral
    Meniscus -.021, Fracture -.019, MCL -.009, Contusion -.008, Effusion -.009).
  - OOF R - B: Lateral OA +.0021, PF OA +.0028, Effusion +.0014, Baker's +.0013, ACL -.0009, Contusion -.0010.
  - Full report: box `~/k12_test/gold/k12_gate_20261010_1936.txt` (+ json).
