# Compute schedule to 10-21: V100 lanes + Kaggle GPU (endgame3, lens = compute)

Written 2026-10-03 ~23:30 IST. Research only: nothing was submitted, pushed, versioned, posted or run on the box.
Every command below is for Eesh to run. Tags: **MEASURED** (log, LB or file), **ESTIMATE** (computed from
benchmarks or measured rates), **GUESS**.

`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`.
Times are IST unless marked UTC. The UTC day (5 submission slots) starts at 05:30 IST.

---

## 0. Bottom line

1. **The V100 is not the binding constraint. The T4 window after the parent is.** The parent runs about
   7.7-8.05 h (its own 8 h budget caps it). That leaves 33-56 min to an 8.6 h process-age wall. The fleet's
   decode alone takes about 15-20 min (MEASURED rate). So the ConvNeXt fleet can serve **about 5 members**, and the
   crop stage about 8 min more. At the top of the P range even that is a coin flip with the current cell constants.
2. **Cut the fleet plan from 10 members (~85 V100-h, `fleet2/SCHEDULE.md`) to 5.** Members: F1 s43, s44, s45
   plus EV2S s143, s144 if family 2 passes G1/G1c; otherwise F1 only. That is ~45-50 V100-h, and the core is
   done by **10-06 ~14:00**. The ~40 V100-h freed go to a pre-registered reserve: all-data seeds that replace the
   fold-0 members, a seed-soup test, crop seeds, and merge support.
3. **Hidden-rerun outputs never come back to us.** The scored run's `fleet_receipt.json` is not visible; only the
   3-study commit's is. A skipped or fallen-back fleet scores exactly like the parent (0.943). So every new runtime
   configuration needs a **`submission_fleetsolo.csv` canary read**: ~0.92 means the stage ran on the hidden set;
   0.943 means it was skipped. The canary is also the G2 solo read and the input for the slot lens's
   "solo read + offline rho -> W" rule.
4. **The ConvNeXt fleet cell needs two laptop changes before its first commit.**
   - Port the crop cell's multi-file writer, so one commit gives W .25/.30/.45, a family subset and fleetsolo.
   - Add a commit-only T4 microbenchmark (ms per member per study).

   Without the port, each W is one Kaggle commit and G2 needs a separate kernel.
5. **Kaggle GPU: commits only, about 3-5.5 T4-h for the rest of the comp.** This week that is about 2-3 h of the
   ~17 h left. The 458-study Kaggle local-LB run (2.5-3 h) is dropped. Parent-on-gold58 runs on the box mimic.
   The fallback, if the mimic is not producing by 10-05 18:00, is a gold58-only Kaggle run (~0.5 h).
6. **Label decision (D1).**
   - Provisional at epoch 4: mix50 ep4 ~01:55, v4 ep4 ~03:35-05:20. Kill one run early only if it is dominated
     on both criteria.
   - Final at v4 ep10, ~10:45 10-04, with both runs complete. The ep10 information is worth more than the early
     lane: the V100 has slack, and the loser can never ship anyway (same `f0_s42` key).
7. **Overnight lanes run only label-independent, pre-queued jobs**, so no lane idles while Eesh sleeps: crop v4
   all-data R50 s0 goes behind f0_s42 tonight, and crop s1 goes behind s43.
8. **Final kernels.**
   - Safe: commit by 10-19 23:00 IST, read on the 10-20 UTC slots, keep 10-21 as reserve, lock picks
     10-21 23:30 IST.
   - Absolute latest: commit by 10-20 22:00 IST, submit at 10-21 05:30 IST, with no reserve.
   - Better: have both picks already scored by **10-14**. With a capped fleet, nothing after 10-07 adds members,
     so 10-16 to 10-20 is only for merge-driven changes.

---

## 1. What binds (in order)

| Resource | Supply | Demand under this plan | Binding? |
|---|---|---|---|
| T4 time after the parent, inside the 9 h hidden rerun | 33-56 min to an 8.6 h wall (P 7.67-8.05 h) | cnxt fleet ~3 + max(15-20, N x 3-4.5) min, plus crop K=8 ~5-10 min | **YES**: caps the cnxt fleet at ~5 members, stage order matters |
| Submission slots | 5 per UTC day, shared with SwaplY | 3-5 per day 10-04 to 10-07, then 1-3 per day | Tight only 10-04 to 10-06 |
| Eesh's hands (relay, packing, commits, gates) | ~09:00-01:00 IST (GUESS) | 2-4 actions per day | Sets the decision times below |
| V100 (one, 2 lanes, ~1.1x aggregate) | ~2 x 24 lane-h per day to 10-18 | core ~45-50 alone-h, then reserve | No (slack from 10-06) |
| Kaggle GPU quota | ~17 h to the 10-10 reset, 30 h per week after (verify the reset day on the quota page; the playbook says "until 10-10") | ~0.4 h per commit, ~8-12 commits total | No |

---

## 2. The runtime fit (why 5 members, why the canary)

**Parent time P on the hidden set.**
- MEASURED outcomes: 1 crop member at K=24 (old cell, about 42 min) fits, 2 members (about 80 min) time out
  (`cropfleet/README.md` sec 4). So 7.67 < P <= 8.3.
- The parent's own `TIME_BUDGET = 8 h` cuts the CoAt children at 8 h (`boxmimic/INVENTORY.md` sec 7). So P <= ~8.05.
- Working range: **P = 7.7-8.05 h.**

**Fleet decode floor (MEASURED).**
- The CPU prep kernel ran the same DICOM-to-140mm-320px-JPEG decode at 2,204 studies in 24.9 min on 4 vCPU
  (`SP/fleet_prep_out/rsna-fleet-prep-s0.log`: "done 24.9 min"; s1: 2,203 in 23.5 min). That is 0.68 s/study.
- For ~1,322 hidden studies, that is about **15 min**. Call it 15-20 min, since the GPU workers share the same 4 vCPUs.

**Per-member GPU cost (ESTIMATE).**
- timm RTX 3090 AMP: convnext_tiny 1,942 img/s at 288 px, scaled to 320 px gives ~1,570 img/s
  (`SP/fleet2/benchmark-infer-amp-nhwc-pt240-cu124-rtx3090.csv`).
- Adjustments: x0.7 for chunk-48 batches; T4 at 1/3 to 1/4.5 of a 3090.
- Result: 240-370 img/s. 96 images per study gives 0.26-0.40 s per member per study. With 661 studies per T4,
  **g = 3-4.5 min per member**.
- EV2S is ~0.91x of that.
- To be replaced by the C2 commit microbenchmark (Sec 7).

**Fleet cell as coded (`round3/fleet/fleet_blend_cell.py:49-53`).**
- left = (9.0 - age - 0.35) h, and the fleet requires left > 30 min.
- Worker budget = left - 15 min. The worker valve keeps a prefix at x0.9.

| P (h) | worker budget as coded | members that fit (decode-bound) | crop K=8 after it (8.6 h wall)? |
|---|---|---|---|
| 7.70 | 42 min | 7-11 | yes (~25 min left) |
| 7.85 | 33 min | 5-8 | yes (~16 min left) |
| 8.00 | 24 min | 4-6 (floor 18-23 min: tight) | barely (~8-13 min left) |
| 8.05 | 21 min | coin flip (floor ~= budget) | no |
| >= 8.15 | fleet skipped (left < 30 min) | 0 | no |

**Consequences**
- **Ship at most 5 cnxt members.** Up to about N = 5 the stage is decode-bound, so members 1-5 cost little
  extra; beyond that every member adds 3-4.5 min.
- **Stage order in any combined kernel: cnxt fleet first, crop fleet second.** The crop cell already has an 8.6 h
  wall with an 8 min minimum start (`crop_fleet_cell.py:37-39`) and is all-or-abort. Crop first would push the
  cnxt cell past its 30 min assert whenever P >= ~7.95.
- **Retune the cnxt cell to the crop cell's convention** (laptop, review, then one commit):
  - one 8.6 h process-age wall;
  - start if at least 20 min are left;
  - shrink N with the valve instead of skipping.

  As coded, it throws away a feasible fleet at P 8.05-8.15.
- **Canary per configuration.** Submit `submission_fleetsolo.csv` once for each new runtime shape:
  1. crop-only K=8 on 10-04;
  2. cnxt N=1 on 10-04;
  3. cnxt N=5 on 10-06;
  4. combined on ~10-10.

  If a canary reads 0.943, the stage is being skipped on the hidden set. Then apply the runtime cuts in Sec 9.
- **Lighter host (only if a canary fails).** The Hisernberg anchor drops the Repair-v1 child: ~1 h, MEASURED
  2.9 s/study on gold58 (`hisern_kernel/BUILD.md`). It would widen the window to ~1.5 h, but the anchor is unscored.
  It costs one commit (C7) and one C1-anchor read before it could host anything.

---

## 3. Assumptions for the lane schedule

- A1: f0_s42 (mix50_logit) started 21:07 IST 10-03. f0_s42_v4 started ~22:45. **If v4 actually auto-launched at
  ~00:30, add +1.75 h to every Lane 2 time before 10-05.**
- A2: the shared epoch is 72 min (65-80, includes val/gold; MEASURED range from the box). Alone is ~40 min (shared / 1.8).
- A3, alone time per job (ESTIMATE). Shared wall time is 1.8x alone.

  | Job | Alone | Shared |
  |---|---|---|
  | F1 all-data seed (10 ep) | 9 h | 16 h |
  | EV2S fold-0 | 6.3 h | 11.5 h |
  | EV2S all-data | 8.5 h | 15.3 h |
  | crop R50 all-data, v4 labels, K=8 shards (MEASURED 9.5 min/ep in fold mode, ~2.8 h; x1.25 for all data) | ~3.5 h | ~6 h |
  | box mimic, parent on gold58 (58 x ~19 s at 2xT4 speed, plus setup) | 0.5-1 h | n/a |

- A4: the box mimic needs about 9 GB of free GPU memory. Its shim maps two "T4" devices onto one V100, a child pair
  takes ~4-5 GB, plus the parent's contexts. So it runs next to at most ONE training job, after a `nvidia-smi`
  check. A guaranteed both-lanes-free window exists at ~10-06 14:00.
- A5: anything needing Eesh is placed between 09:00 and 01:00 IST. Overnight, lanes only run jobs queued in
  advance (the `~/fleet/auto_fold0.sh` pattern: wait for a PID, then launch).

---

## 4. Hour-level schedule, 10-03 23:00 to 10-07 (V100 lanes + Kaggle + reads)

| IST | Lane 1 | Lane 2 | Eesh / laptop / Kaggle | Gate |
|---|---|---|---|---|
| **10-03 23:00** | f0_s42 mix50 (ep ~1.5) | f0_s42_v4 (ep ~0) | Tonight (each 5-15 min): (a) scp `teacher_v2_out/teacher_v2_native384_v8_shard0.csv` + `fleet_g1b.py` to the box; (b) queue on L1 "after f0_s42 exits: crop v4 all-data R50 s0" (`box_finetune_aug_mix50.py --labels llm_labels_v4_blend.csv --strat_labels llm_labels_v4_blend.csv --fold -1 --seed 0`, `_last.pt` rule of LANE_A_RUNBOOK); (c) push the boxmimic gold58 DICOM pack (CPU kernel, no GPU quota) so the data is ready by morning; (d) **C1** commit `rsna-sub-cropfleet` once the members-manifest dataset version is "ready" (0.3-0.5 T4-h) | |
| 10-04 01:55 | mix50 ep4 line | | (optional) `python fleet_g1b.py <v8 csv>` on the box | provisional G1/G1b |
| 03:35 (05:20 if A1 late) | | v4 ep4 line | **Early-kill rule** (pre-registered): kill a run only if it is BOTH >= 0.005 lower on gold58 (mean of raw/EMA, last 2 epochs) AND >= 0.03 higher in rho vs the v8 arm. Otherwise both run to ep10 | D1-provisional |
| 05:30 | | | 10-04 UTC slots open. First read the 10-03 statuses: any ERROR on crop25/v9crop25/crop15/mixsacrop25 = runtime alarm (P near the top). After the C1 log shows `RESULT OK fleet blended: 3 member(s)`, submit C1 `submission.csv` (W .25) + `submission_fleetsolo.csv` (crop canary + solo) | |
| 09:05 | mix50 ep10 done; crop s0 auto-starts (09:05 to ~15:05) | v4 ep8 | scp the OOF csvs; laptop starts the cnxt cell port (multi-file + microbench + margin retune), see Sec 7 | |
| **10:45** | crop s0 | **v4 ep10 done** | **D1 final** (Sec 5); P = fold-0 peak epoch of the winner from the ckpt | G1, G1b |
| 11:00 | crop s0 | **F1 all-data s43** (winner labels, `--epochs 10 --stop_epoch P`), 11:00 to ~03:00 10-05 | Pack fv1 = the winner's f0 ckpt (+ parity npz). Dataset `rsna-knee-cnxt-fleet` v1 | |
| ~13:30 | | | **C2** commit fleet kernel (fv1, ported cell). Check `RESULT OK` + microbench lines | |
| ~14:30 | | | Submit fv1 W .30 + fleetsolo (cnxt canary + G2). 1 slot spare for the crop W follow-up per the 10-03 reads | |
| 15:05 | crop s0 done; **mimic slot** if `nvidia-smi` shows >= 9 GB free next to s43 (parent on gold58 + crop K8/K24 gold bench), else skip | s43 | Pack crop s0 (hold for s1) | |
| ~16:05 (15:05 if no mimic) | **EV2S fold-0 s142** (`fleet2/code`, `--f2 1`, **winner labels**), to ~03:35 10-05 | s43 | | |
| **10-05 03:00** | EV2S f0 | s43 done; **crop s1** auto-starts (queued), 03:00 to 09:00 | | |
| 03:35 | EV2S f0 done; **F1 s44** auto-starts (queued; kill if G2/G3 fail), to ~19:35 | crop s1 | | |
| 05:30 | | | 10-05 UTC slots | |
| **09:00** | s44 | crop s1 done | Read fv1 (G2 solo/canary, G3), crop v1 (W, canary), EV2S (G1, G1c via `g1c_corr.py`). Decide | G2, G3, G1c |
| 09:15 | s44 | EV2S all-data **s143** if G1 + G1c pass (to ~00:20 10-06); else crop cnxt224-v4 all-data (~6 h) then reserve | Pack fv2 = {s43, f0_s42, f0_s142} + crop v2 = {v4 all-data s0, s1, crop_f1}. **C3** crop v2 + **C4** fv2 (files: W grid, F1-only subset = paired family-2 read, fleetsolo) | |
| ~13:00 | | | Submit 10-05 UTC: fv2 best W + F1-only paired + crop v2 main + 1-2 W follow-ups | |
| 18:00 | | | **Mimic deadline.** No `run_check ok` parent gold58 preds by now -> C6 = gold58-only local-LB Kaggle run (~0.5 T4-h; `localLB/build_kernel.py` needs a subset filter, ~5 lines) | |
| 19:35 | s44 done; **reserve R1a: F1 s45** (replaces f0_s42 in the cap-5 fleet), to ~11:35 10-06 | s143 | | |
| **10-06 00:20** | s45 | s143 done; **reserve R1b: EV2S s144** (replaces f0_s142; if family 2 failed, F1 s46), to ~13:50 | | |
| ~14:00 | **Both lanes free.** Guaranteed mimic window if not done yet (1 h). Then reserve list (Sec 6) | | Pack **fv3 = 5 all-data** (s43, s143, s44, s144, s45); **C5**; submit W .25/.35/.45 + fleetsolo (5-member canary) on 10-06 UTC (before 05:30 10-07) | canary N=5 |
| 10-07 | reserve | reserve | Merge pitch gets the fleet numbers (playbook sec 5 timing) | |

**If D1 slips to Eesh's morning** (no early kill), nothing is lost: both fold-0 runs finish by ~10:45 anyway.
**If family 2 fails G1 or G1c,** its all-data rows become F1 seeds (s45, s46). The cap stays at 5.

---

## 5. D1: the label decision (mix50_logit vs pure steven_v4)

- **Inputs**, per run, at ep4 (provisional) and ep10 (final):
  - G1: val AUC vs steven_v4 >= 0.87;
  - gold58 (bug guard and solo proxy);
  - G1b: mean per-label Spearman vs the parent's real v8 arm on the 867 fold-0 val studies <= 0.85
    (`SP/fleet_g1b.py SP/teacher_v2_out/teacher_v2_native384_v8_shard0.csv`).
- **Bias to correct.** `fleet_g1b.py` feeds AUC-vs-steven_v4 into `best_lift` as the solo. That favours the v4 run,
  which is scored against the same noisy labels it trained on.
  - For the lift comparison, use gold58 as the solo: mean of raw and EMA over the last 3 epochs, since a single
    eval has 0.005-0.008 noise.
  - Use rho vs the v8 arm as the decorrelation input.
  - Keep AUC-vs-v4 only as the G1 floor.
- **Rule.** Pick the variant with the higher closed-form best-W lift (slots_tables A1).
  - If the lifts are within 0.0005, pick **v4**: the decorrelation prior from the binormal model and the
    10-03 21:50 crop result.
  - If both runs have rho > 0.85, pick v4 regardless.
- **The loser never ships.** Both runs carry the key `f0_s42`, and the worker/pack tool refuses duplicate keys.
  After D1 its value is analysis only.
- **Every later run uses the D1 winner's labels.** That includes EV2S: `fleet2/SCHEDULE.md` sec 4c hard-codes
  `labels_mix50_logit.csv`. Crop retrains always use steven_v4, independent of D1.

---

## 6. Reserve lanes, 10-06 14:00 to 10-18 (pre-registered priority; run only what the reads justify)

| # | Job | V100 (alone) | Trigger | Why |
|---|---|---|---|---|
| R1 | (scheduled above) F1 s45 + EV2S s144 replace the two fold-0 members | 9 + 8.5 h | fleet G3 read >= 0.943 | Same cap of 5, stronger members (all-data ~ a 5-fold ensemble, LEDGER:39) |
| R2 | Seed-soup test: fold-0 s46 of the winner, then soup(f0_s42, f0_s46) vs ensemble on fold-0 OOF | ~7 h | fleet is lifting AND the microbench + canary say N > 5 does not fit | If soup is within 0.002 OOF of the ensemble (GUESS that it holds: same in22k init, 10 epochs at lr 1e-4), each of the 5 T4 slots can carry a soup of 2-3 seeds, i.e. more ensemble per T4 minute |
| R3 | More crop v4 all-data seeds (R50 s2, cnxt224-v4) | 3.5 h each | crop v2 read >= crop v1 | Cheap at inference (K=8 crop stage ~5-10 min for 3-4 members) |
| R4 | Merge-partner folds/seeds | as needed | merge agreed (10-13 to 10-16) | Any partner leg must fit the same 33-56 min window or replace a stage |
| R5 | F1 s47+ / EV2S s145+ | 9 h each | only if the C2 microbench + the N=5 canary show the window holds N >= 7 | Otherwise wasted: the valve trims them silently |
| R6 | CoAtNet R6 retry | ~7 h | only if written into the playbook before G2 reads (skeptic) | The most parent-like member we could build |
| cut | R50 family-2 (pick B) | n/a | not run | T4 cap; 1.69x memory; same arch as the parent's Rad heads and the crop leg |

**Third concurrent job.** Run one only for a crop R50 job, and only when `nvidia-smi dmon -s um -c 60` shows SM
below ~85% with 2 jobs running and more than 6 GB free (`fleet2/SCHEDULE.md` sec 3 rule).

---

## 7. Kaggle GPU ledger (commits only; submissions do not use quota)

| # | When (IST) | Kernel | T4-h | One commit gives |
|---|---|---|---|---|
| C1 | 10-03 23:30 or 10-04 07:00 | `rsna-sub-cropfleet` (v4 legacy members cropaug_f1/f2 + crop_f1, K=8) | 0.3-0.5 | 6 files: W .25 main, .15/.20/.35/.45, fleetsolo |
| C2 | 10-04 ~13:30 | parent + cnxt fleet fv1 (ported cell) | 0.3-0.5 | W .25/.30/.45 + fleetsolo + T4 microbench log |
| C3 | 10-05 ~11:00 | cropfleet v2 (all-data v4 s0/s1 + crop_f1) | 0.3-0.5 | 6 files |
| C4 | 10-05 ~11:30 | fv2 (s43, f0_s42, f0_s142) | 0.3-0.5 | W grid + F1-only subset + fleetsolo |
| C5 | 10-06 ~15:00 | fv3 (5 all-data) | 0.3-0.5 | W grid + fleetsolo (N=5 canary) |
| (C6) | 10-05 18:00, only if the mimic is not producing | local-LB gold58-only parent | ~0.5 | parent gold58 preds for W decisions |
| (C7) | only if a canary reads 0.943 | `rsna-sub-hisern-ext` (held) | 0.3-0.5 | 5 files; C1 anchor = host without Repair-v1 |
| C8 | ~10-10 (after the reset) | combined final-shape kernel: parent, then cnxt, then crop, retuned margins, per-stage solos | 0.3-0.5 | combos + per-stage canaries |
| C9-C11 | 10-19 18:00-23:00 | final kernels (<= 3) + 1 spare re-commit | 1.2-2.0 | finals |
| **Total** | | | **~3-5.5** | (teacher-v2 remainder extra) |

**Week totals.**
- Week to 10-10: C1-C5 (+C6/C7) = 1.5-3.5 h of ~17 h.
- Week 10-10 to 10-17: C8 + spares, under 2 h.
- 10-17 to 10-22: about 2 h.

SwaplY's own quota is a reserve; never needed under this plan.

**Laptop work that must land before C2 (10-04 morning, CPU only).** Same review standard as `round3/FLEET_REVIEW.md`:
1. Port the crop cell's multi-file writer into `fleet_blend_cell.py`:
   - tmp + `os.replace`, `submission.csv` last;
   - the parent copied to every file on any failure;
   - files `submission_w25/w30/w45.csv`, a member-subset file, and `submission_fleetsolo.csv`.
2. Commit-run microbenchmark: when there are 3 or fewer test studies, time each member over 5 forwards of one
   study per GPU and log ms/member/study. Adds ~1 min. `fleet_worker.py` ships in the dataset; `model_sha` is untouched.
3. Margin retune per Sec 2 (8.6 h process-age wall, start if at least 20 min are left, valve instead of skip).
4. Re-run the round3 dry suite (S1-S5) and the fleet2 t5 serving test.

---

## 8. Reads calendar (compact; slot allocation is the slots lens's job)

| UTC day (starts 05:30 IST) | Reads |
|---|---|
| 10-04 | C1 W .25 + crop fleetsolo canary; C1 W follow-up per the 10-03 crop reads; fv1 W .30 + cnxt fleetsolo canary |
| 10-05 | fv1 W follow-up; fv2 best W + F1-only paired (family-2 value); crop v2 main |
| 10-06 | fv3 W .25/.35/.45 + fleetsolo (N=5 canary); crop v2 follow-up |
| 10-07 to 10-14 | confirmations, the combined kernel (C8) canary + best combos, merge-driven reads. **Goal: both final picks already scored by 10-14** |
| 10-15 to 10-19 | merge-driven reads only |
| 10-20 | final candidates (from C9-C11) |
| 10-21 | reserve: re-reads/hedges only; lock picks 23:30 IST (18:00 UTC) |
| 10-22 | nothing new; deadline 10-23 05:29 IST |

---

## 9. Final packaging: latest safe times (work backwards from the picks)

Scoring latency for our kernels is about 8.6 h of rerun, plus up to ~8 h of queue (MEASURED by Eesh on 09-24),
so 9-17 h after submit.

| Step | SAFE (keeps 10-21 as reserve) | ABSOLUTE LATEST (no reserve) |
|---|---|---|
| Last start of a NEW member recipe or family (it needs its own read first) | 10-16 09:00 | 10-17 06:00 |
| Last start of a same-recipe seed | 10-18 12:00 alone (10-17 22:00 if shared) | 10-19 10:00 alone |
| Checkpoints off the box | 10-19 10:00 | 10-20 08:00 |
| Pack + laptop `verify` (pack_fleet_ds / pack_member) | 10-19 12:00 | 10-20 11:00 |
| Dataset versions "ready" (push the kernel only after "ready", else it attaches the old version) | 10-19 15:00 | 10-20 15:00 |
| Final kernel commits done + `RESULT OK` log checked | 10-19 23:00 | 10-20 22:00 |
| Submit final candidates | 10-20 05:30 (00:00 UTC) | 10-21 05:30 (00:00 UTC) |
| Scored by (worst case) | 10-20 ~23:00 | 10-21 ~22:30 |
| Picks locked | 10-21 23:30 (18:00 UTC) | 10-23 05:29 (deadline) |

**Runtime rules for any final kernel.**
- At most 5 cnxt members plus 3 crop members.
- Stage order: cnxt first, then crop.
- Retuned margins.
- Its exact runtime shape (same N, same stages) must already have a canary read that was not 0.943.
- A final kernel that differs from a scored one only by same-recipe seed swaps is acceptable. A new stage or
  family is not.

---

## 10. What to cut first if the fleet misses its gates

**Already cut by this plan (no gate needed):**
- R50 family-2;
- F1/EV2S seeds beyond 5 members;
- the 458-study Kaggle local-LB run;
- fleet dataset versions that keep fold-0 members once all-data seeds exist.

**On a gate miss, in this order:**

| Miss | Cut | Keep |
|---|---|---|
| G1b: rho > 0.85 for the chosen target | switch to the other target (v4) for every later run | everything else |
| G1: both variants < 0.87 val at ep10 | do not start s43 (Lane 2 goes to crop s1 early) | EV2S f0 as the pre-registered family switch (R6 says switch family before tuning). If EV2S also < 0.87, Lane F is closed |
| G2: fv1 fleetsolo < 0.915 | kill s43/s44 at once; no F1 all-data | EV2S only if its G1 >= 0.87 and G1c PASS; else close Lane F |
| G3: fv1 W .30 <= 0.942 and fv2 best W <= 0.942 | stop all F1 seeds; at most one W .20 read, then close | |
| G1c: EV2S rho vs F1 > 0.90 | EV2S all-data seeds (their slots go to F1 s45/s46) | EV2S f0 for analysis only |
| Canary: a fleetsolo reads 0.943 (stage skipped on the hidden set) | in order: (1) crop stage out of the cnxt kernel (separate picks); (2) retuned margins if not shipped yet; (3) N <= 3; (4) C7 lighter-host read | |

**If Lane F closes:**
- The V100 goes to R3 (crop seeds) and R4 (merge).
- The final kernel becomes parent + crop fleet (K=8, ~8 min).
- Kaggle GPU drops to 1-2 more commits.

**Protected in every branch:**
- crop v4 all-data s0/s1 and their reads;
- parent-on-gold58 (box mimic or C6);
- the 10-14 "picks already scored" target.

---

## 11. Corrections to existing plans (found while building this)

1. `fleet2/SCHEDULE.md` sec 2-3 budgets 10 members / ~85 V100-h and says "the binding limit is the T4 inference
   budget (about 10 members...)". That rests on aastik's s75, whose base is faster than our 7.7-8.05 h parent. For
   our parent the decode floor alone uses most of the window: cap at 5.
2. `fleet2/SCHEDULE.md` sec 5 says "After every rerun, read `fleet_receipt.json`". The hidden rerun's outputs are not
   returned; only the 3-study commit's are. Use fleetsolo canary reads.
3. `fleet2/SCHEDULE.md` sec 4c EV2S/R50 commands use `labels_mix50_logit.csv`. Use the D1 winner.
4. `SP/fleet_g1b.py` uses AUC-vs-steven_v4 as the solo in `best_lift`, which biases D1 toward the v4 run. Use gold58
   (multi-epoch mean) as the solo.
5. `round3/fleet/fleet_blend_cell.py`:
   - one W per kernel and no fleetsolo (wastes commits; G2 had no kernel);
   - `left > 30 min` with a 0.35 h margin skips a feasible fleet at P 8.05-8.15;
   - stacking it after the crop cell makes the skip likely.

   Fixes are in Sec 2 and Sec 7.
6. The playbook sec 4 G0 (45 min/epoch alone) passes on the measured shared rate. No budget variant (16 triplets,
   256 px) is needed. Do not change train/serve geometry now.
