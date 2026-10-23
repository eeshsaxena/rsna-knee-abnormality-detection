# PREREG FN11: does set E2 (TTR288 frac75 seed 1046) join the parent-free fleet?

Written 2026-10-09 ~16:50 IST by Claude (subagent), BEFORE any E2 checkpoint was final, picked, inferred or scored.
Do not edit after the first E2 gate run; append a dated "RESULTS" section at the bottom instead. The sha256 of this
file as written is recorded in `FN11_RESULT.md`. No em dashes. Tags: MEASURED, MODEL, GUESS.

**Disclosure.** At 16:47 IST, before writing this file, `~/modal_e2/status.sh` printed the trainer's per-epoch val lines
of E2 f0-f3 at epochs 9-11 of 21 (raw val AUC vs llm_labels_v4 in the .87-.89 range, as for every TTR288 run at that
point). No E2 run had finished: no best checkpoint, no OOF file, no receipt and no gold prediction existed. Every number
used below (bars, baselines) comes from the already-shipped sets C, D and E (k10 collect, 10-09 07:05).

## 1. What is decided

FN11 = the FN10-3 kernel (parent-free, rank blend P/F/N 0/.40/.60 per label, strict fail-loud) with set E2 added to the
fleet: F = C (288 mix50 s1044) + D (224 frac75 s1043) + E (288 frac75 s1045) + E2 (288 frac75 s1046). Inside F every
member has equal weight (raw-probability mean over members and A+B passes, exactly as the K5 fleet worker); F and N
are combined only by the per-label percentile rank blend. N-weight twin FN11-N70 (P/F/N 0/.30/.70) is the same fleet;
it needs no separate gate (it rides on the same E2 decision; its slot follows the Saturday N70 reads).

E2 = `~/fleet_runs/f{F}_s1046_ttr288frac75`, trained on Modal L40S with the box queue line of set E token for token
except `--seed 1046 --fold F --out` (MODAL_E2.md). Members are the RAW best checkpoint of each fold (K8-RAW, like E).

## 2. Per-fold acceptance of E2 (mechanical, before any fleet read)

A fold is ACCEPTED only if all of these hold (picker `k11/box/k5_pick.py` = the K10 picker + K11-MREC):
- Every K10 set-E picker check: closing line `{"raw": .., "ema": ..}`, exactly one raw and one ema best file, ckpt cv ==
  closing line (1e-12), OOF rescored vs `llm_labels_v4_blend.csv` == ckpt cv (1e-12), model_sha f141aeed136dfc01,
  ttr_size buffer == args == 288, fold / seed 1046, train_labels `labels_frac75_logit.csv`, sel labels v4, amp fp16, decode
  probe npz present, epochs header == ckpt args == 21, min epochs 16.
- **K11-MREC** (Modal receipt, replaces the Vast receipt of E): `~/fleet_runs/modal_e2_<run>_receipt.json` exists with
  run == the folder name, verify "OK" (place_fold.sh md5 of every fetched file == the Modal volume manifest),
  best_raw / best_ema / top_oof / epoch_oof == 1 / 1 / 2 / 32, closing_line[:120] == the log's last line[:120], both
  ckpts listed with args "OK" and the picked file among them. With a valid receipt the OOF-vs-ckpt mtime rule is
  reported, not enforced (a copied run, as the Vast and Kaggle folds).
- **Sanity bar** E2_F >= E_F - .005, E_F = set E's picked raw cv of the same fold (E is the same recipe, other seed):
  f0 .884325, f1 .887201, f2 .886670, f3 .873837, f4 .886362. Basis (MEASURED, K10 runbook sec 1): fold-matched seed
  swaps at a fixed recipe have SD ~.0016-.0020, so a seed-vs-seed difference has SD ~.0025 and .005 is ~2 SD; a broken
  fold (historically -.01 or worse) fails. No override: a fold under its bar does not ship.
- A fold that fails a check or its bar is EXCLUDED (never repaired by hand). E2 is gated at all only with >= 4 accepted
  folds; with <= 3 the result is "E2 NOT GATED" and FN10-3 stays the vehicle.

## 3. The gate (decision; one script, `k11/box/k11_gate.py`, sha256 in FN11_RESULT.md)

Base fleet B0 = C + D + E (15 members, == FN10-3, bytes == the k10w upload). Candidate B1 = B0 + the accepted E2 folds.

- **G1, OOF (non-gold rows):** dOOF = macroAUC_OOF(B1) - macroAUC_OOF(B0) >= 0.0000.
  OOF = fold-matched: each val study of fold F is scored by the fold-F member of each set (the picked RAW checkpoint's
  OOF csv, `~/k10_test/oof` for C / D / E, the E2 picker copy for E2); a fleet's OOF row = the equal mean of its sets'
  fold-F rows. Labels `~/rsna_train/llm_labels_v4_blend.csv`; masked macro AUC = `tta_bench.macro_auc` (as
  k10_gold_read.py). Rows = the val studies of the folds where E2 has an accepted member (paired: B0 and B1 on the same
  rows), intersected over sets; asserted disjoint from gold74 (every run used `--exclude gold74_eval_set.csv`).
  Caveat (stated now): every member's epoch was selected on these labels on its own val fold, so each set's OOF is
  optimistic by a similar amount; G1 guards against a broken or harmful set, it is not an unbiased gain estimate.
- **G2, gold58 (expert labels, held out of every run):** d58 = AUC58(B1) - AUC58(B0) >= -0.0005.
  Pass A predictions (offset 0, no mirror; `~/novel/gold_infer5.py` CPU fp32, sha256 sidecars as k10_gold_read.py),
  fleet = equal mean of ALL members of the fleet (15 for B0; 15 + n_E2 for B1), masked macro AUC on the
  `original_gold58` rows of `gold74_eval_set.csv`. C / D predictions come from the read-only caches used by the K10 read;
  E from `~/k10_test/gold/preds` (sidecar == ckpt sha256); E2 from new predictions with sidecars.
- **VERDICT:** E2 JOINS (build FN11 FINAL) iff G1 AND G2. Otherwise E2 does NOT join: FN10-3 stays the vehicle, no FN11
  dataset or kernel is created, and the work stops after recording the result.
- **Reported, never gating:** gold74 delta; paired study bootstrap SE (B 2000, seed 59) of d58 and d74; per-label deltas
  (Fracture singled out); per-set and per-fold OOF macro AUC; E2 vs E per fold; mean per-label Spearman E2~E (fold-matched,
  gold74 and OOF).

**Why no gate on the F .40 / N .60 blend on gold or train rows.** N's predictions on train and gold rows are not
out-of-sample: the public N (nartaa CoAtNet) was trained on all RSNA training studies (the gold74 studies are training
studies) and its author selected on gold58 (KFN_RUNBOOK sec 2: "NOT a CV read"; kfn_gold_sanity.py), and only 12 gold58
N predictions exist (A0no_gold12). A gold-row blend would be dominated by N's in-sample fit and could not see the fleet
change, so the suggested "F.40/N.60 on gold rows drops <= .0005" criterion is replaced by the fleet-only G1 + G2. The
fleet enters the submission only through its per-label ranks, and G1 / G2 are rank statistics (AUC) of that same fleet.

## 4. Only 4 E2 folds (fold 4 later or never)

- FN11 cannot be submitted before 10-11 (all five 10-10 slots are taken by PREREG_10-10 Amendment 2), so waiting for f4
  is free until 10-10 midday.
- When f0-f3 are accepted and f4 is not final: run the gate on 4 folds ("E2-4 read": G1 on the val rows of folds 0-3,
  G2 with B1 = 19 members). It is PROVISIONAL. On PASS the 19-member kit is built and A0-tested as a TEST build
  (`DO_NOT_SUBMIT_INTERIM.txt`) so the pipeline is proven before f4 lands.
- If f4 is accepted by **10-10 12:00 IST**: the 5-fold read (B1 = 20 members) is the decision; FN11 ships E2 5/5. The
  4-fold read is info only.
- If f4 is not final by 10-10 12:00 IST, or f4 fails a check or its bar: the E2-4 read is the decision and FN11 ships E2
  as a 4-member set (19 members), built FINAL from the same members.
- No mixing: the read that decides is the read of exactly the composition that ships.

## 5. Runtime (hidden rerun 2 x T4, 9 h hard limit; decided before building)

- Rule: the MODEL slow-case end of the shipped kernel must be <= 0.65 x 9 h = **5.85 h** (>= 35 % of the 9 h limit
  unused). Model = `k10/tests/runtime_k10.py` (MEASURED T4 microbench of the K5 / K7i commits) extended by the extra 288
  members; cross-checked against the scored FN hidden run (submit -> COMPLETE <= 1.2 h, 10-09) and the FN commit
  MICROBENCH (TTR288 474-488 ms/member/study).
- If it does not fit: drop set C folds one at a time, lowest picked cv first (C f3 .882181, then f0 .890920, ...), until
  it fits; each drop is recorded. (MODEL expectation: FN11 ~2.3 h central / ~3.1 h slow, so no drop.)

## 6. D2 (optional, only if 224 frac75 s1044 f1-f4 land from Vast)

Same structure, decided now: base = the fleet after the E2 decision; D2 joins iff G1 and G2 (same bars) hold for
base + D2 vs base; D2 needs 5/5 accepted folds (f0 Kaggle receipt, f1-f4 Vast receipts), bar D2_F >= D_F - .005 (D f0
.883777, f1 .890967, f2 .890903, f3 .876911, f4 .892889). It ships under NEW slugs (dataset `rsna-knee-cnxt-fleet-k11w2`,
kernel `rsna-sub-k11fn-d2`), never as a new version of an FN11 dataset that a pending FN11 commit or submission reads.

## 7. Fixed regardless of the result

No Kaggle dataset create / version, no kernel push, no submission by this agent. `eeshsaxena/rsna-knee-cnxt-fleet` is
never versioned. Box work is CPU only (CUDA hidden, nice 19, ionice idle, taskset to CPUs 34-39), writes only
`~/k11_test`; the V100, `~/queue`, `~/fleet_runs` (read only) and `~/oai_*` are never touched. E2 is RSNA-only (no OAI).

## RESULTS (append below, dated; do not edit the sections above)

### 2026-10-09 18:40 IST: E2-4 PROVISIONAL read (folds 0-3; f4 launched on Modal ~18:05, not final)
- Per-fold acceptance (k11_merge.py): E2 f0 .889364 (bar .884325), f1 .891364 (.887201), f2 .893418 (.886670), f3 .880323
  (.873837): all ACCEPTED; every picker check + K11-MREC passed (TK11PICK 9/9 on f0).
- k11_gate.py (sha256 8bfa6d70..., recorded before E2 landed in GATE_SCRIPTS_pre_E2.sha256), log `logs/k11_gate_20261009_1840.txt`:
  - G1 OOF (folds 0-3, 3,467 val studies): B0 .89037, B1 .89083, dOOF **+.00046 PASS** (bar >= 0).
  - G2 gold58 pass A: B0 .9220 (15 members), B1 .9219 (19 members), d58 **-.0001 PASS** (bar >= -.0005; SE .0010).
  - Info: d74 +.0003 (SE .0008); Fracture d58 -.0028; E2 - E gold58 +.0045 (SE .0041); Spearman E2~E .966 (gold74),
    .966 (OOF).
  - **VERDICT E2 JOINS (provisional)**. Decision read = the 5-fold read if f4 is accepted by 10-10 12:00 IST (sec 4).
- Info only, earlier: the 1-fold read at 17:56 (f0 only, NOT GATED by rule).

### 2026-10-09 20:41 IST: DECISION read, E2 5/5 (f4 placed 20:24, before the 10-10 12:00 cutoff: sec 4 -> the 5-fold read decides)
- Per-fold acceptance: f4 .894199 (bar .886362) ACCEPTED; f0-f3 as above. E2 5/5 FINAL (k11_merge K11MERGE OK 20 members).
- k11_gate.py (same sha256 8bfa6d70...), log `logs/k11_gate_20261009_2041.txt`:
  - G1 OOF (5 folds, 4,333 val studies): B0 .89121, B1 .89171, dOOF **+.00049 PASS**.
  - G2 gold58 pass A: B0 .9220 (15), B1 .9219 (20), d58 **-.0001 PASS** (SE .0013, bar >= -.0005).
  - Info: d74 -.0002 (SE .0009); Fracture d58 -.0028; per-set gold58 E2 .9199 vs E .9163 (E2 - E +.0035, SE .0039);
    Spearman E2~E .966 (gold74) / .965 (OOF); E2 - E OOF per fold +.0000 / -.0008 / +.0018 / +.0015 / +.0028.
  - **VERDICT E2 JOINS. FN11 = C + D + E + E2 (20 members) is built FINAL.**
