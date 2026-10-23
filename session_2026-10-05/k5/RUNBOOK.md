> **K3 note (10-04 22:00):** this file is the K2 runbook, copied unchanged from `SP/k2`. For K3 (two members, strict main, TTR solo) use `K3_RUNBOOK.md` in this folder.

# K2 RUNBOOK: fold-0 ConvNeXt member on the 0.943 parent, one commit, several submittable files

Written 2026-10-04 ~03:45 IST, finalised ~04:45 IST after the review fixes (`FIX_LOG.md`). Nothing was pushed,
created, versioned or submitted. Eesh does every `kaggle` write.
`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`,
`PY` = `C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe`, `B` = `training@192.168.10.30`
(`ssh -o BatchMode=yes`). Box python for the scripts here = `~/miniconda3/envs/kmimic/bin/python`.

## 0. What K2 is

K2 = the 0.943 parent (`SP/kernels/repro_0946`, 31 cells, sha256 `35547f8a...`, unchanged) + one appended cell, the K2
fleet stage. One commit writes:

| File | Content | Exists when |
|---|---|---|
| `submission.csv` | P+F rank blend at W_MAIN .30 | always: P+F if the stage was clean, else the parent's bytes (fail-safe, as requested) |
| `submission_w30.csv` | identical to `submission.csv` when clean | **only if the stage was clean** (the fail-loud twin of the main file) |
| `submission_w20.csv`, `submission_w40.csv` | P+F at W .20 / .40 | only if clean |
| `submission_fleetsolo.csv` | the fleet alone (raw-prob mean, parent row order) | only if clean |

"Clean" = every worker exited 0 with the full member count (`n_keep == M`), no worker hit the wall, every gate passed
(coverage > 98 %, views/study >= 3, parent~fleet Spearman >= .30 on >= 50 studies, finite, in [0, 1], parent row order)
and every write completed. Otherwise none of the four fail-loud files exists (stale copies are deleted first, a
partial write deletes what it wrote) and `submission.csv` is the parent's bytes. A slot submitted on a missing file
errors instead of silently scoring 0.943.

Other `submission*` names in the output are NOT K2's and are never submitted: the parent's own
`submission_checkpoint_rank_mean.csv` (top level, plus `g96_input/` and `d4_input/` copies; unchanged 0.943 parent
behaviour) and, in the combined kernel, the K1 crop cell's sandbox files, which the CF writer renames
`_k2_crop/k1out*.csv` (C6) so none of them looks submittable.

Optional combined kernel (K3 in ENDGAME_V3, `--combined`): + 3 cells = crop setup, the verbatim K1 crop cell (sha
`049033d6002cf9fc`) in a sandbox, the CF writer. Extra files, each fail-loud on the stages it needs:

| File | Content | Exists when |
|---|---|---|
| `submission_cf.csv` | rank_pct(.60 r(P) + .15 r(C) + .25 r(F)) | fleet clean AND crop ok |
| `submission_c15.csv` | rank_pct(.85 r(P) + .15 r(C)) | crop ok |
| `submission_cropsolo.csv` | crop leg alone (raw-prob mean of the crop members) | crop ok |

Both stages share one wall: kernel start (= earliest of the kernel process start and the parent's `T0`) + 8.6 h. The
wall is the only time guard: the fleet stage starts only with >= 20 min left and then runs to completion or to the
wall (no 150 min cap any more, FIX_LOG F6); the crop stage starts only with >= 10 min left (a start does not guarantee
completion, section 6; a doomed start only leaves the C/CF files absent, nothing runs after it). Only if no kernel
clock is readable at all (not expected: the pinned parent sets `T0` in cell 5) the fleet stage gets 40 min from its
own start, so a parent at its 8.0 h budget still ends < 9 h. A crop failure never touches a fleet file (sha256-checked
in the smoke and on the box).

**CF definition: rank-averaged legs, raw-prob mean only inside a leg.** Why not one raw-prob mean over all members:
(1) the legs are different families trained on different targets (crop R50 MIL heads on steven_v4; ConvNeXt 4-view on
the D1 labels), so their per-label probability scales differ, and a pooled raw mean would weight each leg by member
count x probability spread (3 crop members vs 1 fleet member = 3:1 before scale), an uncontrolled, label-dependent
weight; (2) P is itself a per-label rank file and every read we have or will have (K1 w15/w25, K2 w30) is a rank blend
of P with ONE leg, so a 3-way rank blend reduces exactly to those reads when a weight is 0 and the reads map straight
onto CF weights (the A3 closed form in `endgame3/slots_tables.txt` is this rank-space model); (3) AUC is per-label
rank-invariant, so rank space loses nothing. Inside a leg the raw mean stays (both verified cells do it; same-family
evidence: aastik's 6 models 0.927 raw vs 0.924 rank-mean). Weights .60/.15/.25 = `risk.md`'s CF and rule R6 (crop .15,
crop + fleet <= .40); A3 puts the optimum at wC .16 / wF .27 for C .925 / F .930 / rho .80.

## 1. Files

| Path (`SP/k2/`) | Role |
|---|---|
| `build_k2.py` | checkpoints -> preflight -> `build/upload/rsna-knee-cnxt-fleet` (dataset folder + metadata, NEW dataset `eeshsaxena/rsna-knee-cnxt-fleet`) -> `build/kernel/rsna-sub-k2-fleet` (+ `rsna-sub-k2-fleetcrop` with `--combined`) -> `build/BUILD_SUMMARY.json` + the kaggle commands (printed, never run) |
| `make_k2_kernel.py` | the patched `make_parent_fleet_kernel.py`: sha-pinned parent, W literals (`K2_CFG` / `K2C_CFG`), K1 crop cell pinned, metadata, read-back |
| `cells/k2_fleet_cell.py` | the ported fleet stage (round3 B1-B8 kept; P1-P8 in its header) |
| `cells/k2_crop_pre_cell.py`, `cells/k2_crop_post_cell.py` | combined mode: sandbox + manifest pin + shared wall; CF writer (+ C6 sandbox rename) |
| `code/` | dataset code: `fleet_common.py`, `fleet_model.py`, `fleet_model_f2.py` = byte copies of fleet2 = the box trainer's `~/fleet` (sha `c112dac9...`, `2a3b176d...`); `fleet_worker.py` = fleet2 worker + K2 patch (K2a-c decode timing, commit-only microbench, `loop_min`; K2d cv2 1 thread; K2e per-study decode timeout; predictions unchanged); `pack_fleet_ds.py` = fleet2 pack tool |
| `k2_check.py` | independent checker of any K2 output dir (smoke, box, downloaded Kaggle commit output); `--kaggle_commit` also gates on the receipt (parity, decode probe, microbench, 2 GPUs) |
| `box/d1_pick.py` | box, read-only: D1 label rule + tag rule + G1 -> the one checkpoint to ship (exit 0 final / 2 no K2 / 3 provisional) |
| `box/k2_box_stage.py`, `box/make_overlay.sh`, `box/BOX_TEST.md` | box tests (CPU, GPU stage, full notebook via `run_parent.py`) |
| `tests/smoke_k2.py`, `tests/run_scenario.py`, `tests/smoke_transcript.txt` | laptop CPU smoke |
| `_real/store/` | the 4 parity npz (`~/fleet_store/<decode_probe sid>.npz`; the same 4 for both fold-0 runs) |
| `FIX_LOG.md` | the review issues, verdicts and what changed |

## 2. Evidence so far (10-04)

- Laptop CPU smoke after the fixes: `K2 SMOKE PASS` (`tests/smoke_transcript.txt`, 04:09-04:15, 5.2 min): K2 worker == fleet2 worker (P max abs 0.0, 24 studies);
  K2e decode timeout (one study hangs 30 s, timeout 1 s: that study skipped, the other 23 rows max abs 0.0);
  22 cell scenarios pass (16 fleet-only incl. the new blank / views2 / noclock gates, 6 combined incl. the C6
  sandbox rename); `build_k2.py` on the real interim checkpoint + the one-tag-per-run refusal.
  `k2_check.py --kaggle_commit` PASS 27/27 on the smoke's 3-study commit output and FAIL (no microbench) on the
  24-study one, as intended.
- `build_k2.py` on the REAL interim box member (`f0_s42_ema_cnxt320x24_cv8183_best.pt`, mix50): contract and
  model_sha `fa35880fca963652` match `k2/code`, pack OK, 117.7 MB / 11 files, both kernels built (32 and 35 cells,
  cells 0-30 == parent).
- Box CPU stage test A0 on 2 real fold-0 val studies with the real interim member and the real K1 crop members (Cv1:
  cropaug_f1, cropaug_f2, crop_f1), Kaggle's cv2 4.14: 03:22 (pre-fix) `BOXSTAGE PASS 35/35`, fleet solo vs the
  trainer's own OOF max abs 1.3e-4; 04:15 (post-fix: patched worker K2d/K2e, patched cells) `BOXSTAGE PASS 36/36`,
  same OOF max abs 1.34e-4 and `submission_fleetsolo.csv` byte-identical to the pre-fix run (K2d leaves the decode
  unchanged with Kaggle's cv2), sandbox holds only `k1out*.csv` (`box/box_cpu_transcript_fix.txt`).
- `d1_pick.py` (fixed) on the box 04:07 against the live runs: exit 3 `PROVISIONAL` (both runs mid-training), exact cv
  from the checkpoints (v4 raw .84025, file name says 8402); on a symlinked copy with the closing json lines added it
  exits 0 with a `SHIP` line, and it crashes (exit 1, never ships) when a best file is missing after the json line.
- Box full-notebook path prepared (`run_parent.py --prepare-only` via the overlay): 0 unrewritten `/kaggle` in the 4
  appended cells, both datasets linked. Not run (needs an idle V100).

## 3. Which checkpoint (rule, mechanical)

```bash
ssh -o BatchMode=yes training@192.168.10.30 'CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/kmimic/bin/python -W ignore ~/k2_test/k2/box/d1_pick.py; echo EXIT $?'
```

Exit 0 + one `SHIP <path>` line = final. Exit 3 `PROVISIONAL` = do not act, re-run in 5 min. Exit 2 = both runs fail
G1: no K2 today. The rule it applies (ENDGAME_V3 sec 4 D1 + its [verify] correction, sec 5, fleet2/SCHEDULE 4c):

1. **Tag per run:** EMA, unless the raw `_best.pt` has the higher cv (val AUC vs steven_v4 at its best epoch, read
   exactly from the checkpoint, not from the truncated file name). Never both tags (the pack tool and the worker
   refuse two files of one run).
2. **Label variant (D1):** per run, solo = gold58 mean over raw AND EMA of the last 3 epochs; rho = mean per-label
   Spearman of the chosen tag's OOF vs the parent's real Raptor v8 arm on the 867 fold-0 val studies; lift = binormal
   best-W lift over .943. **mix50 ships only if** its lift beats v4's by > .001 AND its gold58 edge > .012 AND its rho is
   not higher; both rho > .85 -> v4; anything else -> **v4** (the measured decorrelation prior). The loser never ships.
3. **G1 floor** for the shipped tag at its best epoch (the epoch stored in the checkpoint): val AUC vs steven_v4 >= .87
   and gold58 >= .87. Winner fails and the other run passes -> ship the other. Both fail -> **no K2** (EV2S f0 becomes
   the family switch, ENDGAME G1).
4. **Final** = BOTH trainers have printed their closing `{"raw": .., "ema": ..}` line (fleet_train.py prints it only
   after the last `save()`) and logged 10 epochs. The last EMA eval + save land ~4.6 min after the last `ep 9 raw` line
   and `save()` deletes the old best file before writing the new one, so 10 logged epochs alone is not final. Until
   then the script exits 3 and prints no `SHIP` line.

At 04:07 (provisional, mix50 ep 5 / v4 ep 4 done): mix50 raw cv .8742 gold58 .900 rho .873, v4 raw cv .8403 gold58 .836
rho .811; both lifts 0 at these solos; v4 fails G1 at ep 4 -> the provisional pick is mix50 raw. Do not act on it.

Expected finish (from the logs, ~70-74 min per epoch incl. eval): mix50 (`f0_s42`) ~09:35 IST, v4 (`f0_s42_v4`)
~10:30 IST. T below = the v4 closing json line.

## 4. Tonight, from "fold-0 done" to "submit"

| IST (T = v4 done, ~10:30) | Step | Who | Gate (stop if it fails) |
|---|---|---|---|
| before T (optional, 6 min) | `$PY -W ignore $SP/k2/tests/smoke_k2.py`, only if anything in `k2/` changed after this runbook | Claude | `K2 SMOKE PASS` |
| T+0 (2 min) | `d1_pick.py` (section 3) | Claude/Eesh | exit 0 and a `SHIP` line (exit 3: wait 5 min, re-run; exit 2: no K2 today) |
| T+2 (2 min) | `mkdir -p "$SP/k2/ckpt"` then the `scp` line `d1_pick.py` prints (`scp -o BatchMode=yes $B:fleet_runs/<run>/<file> "$SP/k2/ckpt/"`). Parity npz are already in `$SP/k2/_real/store` | Claude | file size 112,823,466 B; exactly one `.pt` in `$SP/k2/ckpt` |
| T+4 (1 min) | `$PY -W ignore $SP/k2/build_k2.py --ckpt "$SP/k2/ckpt/<file>" --store "$SP/k2/_real/store" --out "$SP/k2/build" --combined` (add `--overwrite` if `build/` exists) | Claude | the `member` line shows the `SHIP` run's labels (`labels_mix50_logit.csv` or `llm_labels_v4_blend.csv`), tag, epoch, cv, sha `fa35880fca963652`; `upload folder OK ... 11 files`; two kernels built; `BUILD_SUMMARY.json` written |
| T+5 (5 min, parallel) | box A0 CPU test with the shipped member (`box/BOX_TEST.md` sections 0 + A0, `RUN`/`TAG` from the `SHIP` line) | Claude | `BOXSTAGE PASS`; fleet solo vs trainer OOF max abs < 2e-2 (interim member: 1.3e-4). A1 GPU gold10 instead only if >= 6 GB free |
| T+5 (5-15 min) | `kaggle datasets create -p "$SP/k2/build/upload/rsna-knee-cnxt-fleet"` then `kaggle datasets status eeshsaxena/rsna-knee-cnxt-fleet` until `ready`. Windows CLI upload bug: pre-create the `%TEMP%\.kaggle\uploads\C_\Users\...` staging dirs (LANE_A_RUNBOOK) | **Eesh** | status `ready` before the push (a push before that mounts nothing: the commit log then says `fleet dataset not attached`) |
| T+20 (1 min) | `kaggle kernels push -p "$SP/k2/build/kernel/rsna-sub-k2-fleet"` (fleet-only K2; NOT the combined kernel today) | **Eesh** | push accepted; **write down V** from `Kernel version V successfully pushed` (V > 1 after any re-push) |
| T+20 to T+40 | commit run: parent on the 3 public test studies (~4 min fixed, 242 s measured) + fleet stage (~1-3 min) + queue | Kaggle | `kaggle kernels status eeshsaxena/rsna-sub-k2-fleet` = complete |
| T+40 (5 min) | `kaggle kernels output eeshsaxena/rsna-sub-k2-fleet -p "$SP/k2/build/out_k2v<V>"`; `$PY $SP/k2/k2_check.py --work "$SP/k2/build/out_k2v<V>" --expect_fleet clean --kaggle_commit`; read the log (below) | Claude | `K2CHECK PASS` for THIS version V (output dir is per version; no push between this check and the submit) |
| T+45 (~11:15) to 21:30 at the latest | submit (section 5) | **Eesh** | slots left in the UTC day (05:30-05:29 IST, shared with SwaplY) |
| submit + ~6 h | reads -> G2 (fleetsolo) / G3 (main), ENDGAME sec 4 | Claude | |

Commit-log landmarks (`out_k2v<V>/rsna-sub-k2-fleet.log`):

```
[k2-fleet ..] RESULT OK fleet stage clean: 1 member(s) ['f0_s42_<tag>_cnxt320x24_cv<NNNN>_best.pt'], 3 studies, 0 median-filled, views/study >=3.00
[k2-fleet] submission.csv / submission_fleetsolo.csv / submission_w30.csv / submission_w20.csv / submission_w40.csv   rows=3 nan=0   (none ABSENT)
[fleet 0] decode probe vs ckpt (train-side decode): match 4 mismatch 0          (and the same for [fleet 1])
[fleet 0] PARITY re-encode vs stored train bytes: {... 'frac_gt8': <=0.01, ... 'mismatch': []}   (Kaggle has train_series)
[fleet 0] MICROBENCH 5 forwards/member on cuda: ms/member/study {...}           (one per GPU: G-T input g)
```

The `[k2-fleet]` lines are printed by the notebook process and are required. The `[fleet g]` lines are printed by the
Popen workers to the inherited fd 1; nothing in this project has yet shown child-process output in a Kaggle log. If
they are missing, read the same facts from `fleet_receipt.json` (also echoed in the `[k2-fleet] receipt` line):
`parity[g].decode_match == 4` and `decode_mismatch == 0` for every g; `parity[0].mismatch == []` and
`parity[0].frac_gt8 <= 0.01`; `bench` has one non-empty dict per GPU; `n_keep == [M, M]`; `n_fail == [0, 0]`.
`k2_check.py --kaggle_commit` checks exactly these, so the gates stay: RESULT OK, no ABSENT file, `K2CHECK PASS`.

With 3 studies the W files usually print `(byte-identical to parent)`: per-label ranks of 3 rows coincide with the
parent's. Expected; the gates above are what count.
`RESULT FLEET STAGE NOT CLEAN` -> do not submit; the reason is on that line: `fleet dataset not attached` = pushed
before `ready` (re-push once `ready`; the re-push is version V+1, use that V everywhere below); `worker rc [1, ..]` =
read the `[fleet g]` traceback above it (contract / model_sha / timm); `rc [4, ..]` = PARITY FAIL (Kaggle decode
differs from training: stop, investigate); `only N min before the wall` cannot happen on a 3-study commit.

## 5. What to submit (2 slots, Wave B of ENDGAME_V3 10-04)

```bash
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k2-fleet -v <V> -f <FILE> -m "K2 P+F W.30 f0 <labels> <tag> ep<E>"
# V = the version whose commit output passed K2CHECK. Never a version whose log says `fleet dataset not attached`:
# its rerun scores submission.csv as the parent with no error, and its fail-loud files error.
```

| K1 filename test (K1 `submission_fleetsolo.csv`, Wave A) | Slot 1 (main, G3) | Slot 2 (solo, G2) |
|---|---|---|
| scored | `submission_w30.csv` (fail-loud main: an error = the fleet did not finish in that rerun, never a silent 0.943) | `submission_fleetsolo.csv` |
| still pending | `submission.csv` (scores under either filename branch) | `submission_fleetsolo.csv` (canary for the stage; also settles the filename question) |
| errored / CLI refused (Branch S) | `submission.csv` only; an exact .943 then carries ENDGAME E2's haircut | none (the solo read needs the Branch S light notebook, not built) |

Only these names. Never `submission_checkpoint_rank_mean.csv` (the parent's own intermediate) or anything under a
sub-folder.

W: the main is .30 (binormal model + gold58 proxies put best W ~.25-.35 for a ~.92-.93 member; one read resolves only
~+.003 vs ~0). `w20` / `w40` are for a later single read by rule only (G3: d <= -1 with a fine solo -> one `w20`
read), never a W curve. The combined kernel waits for 10-05 (ENDGAME: push K3 `cf` / `c` only after K2 has at least
one scored file); it is already built by `--combined` and runs the same way (`kaggle kernels push -p
"$SP/k2/build/kernel/rsna-sub-k2-fleetcrop"`, then `k2_check.py ... --expect_fleet clean --expect_crop ok
--kaggle_commit`; its log also needs `[cropfleet ..] manifest ...: 3 member(s) [...]` with the intended crop version
and `[k2-crop ..] RESULT crop ok=True fleet clean=True | C files: written | CF file: written`). The crop members
dataset is attached unpinned: the commit uses whatever version is current at push time.

## 6. Timing budget inside the hidden rerun

Parent P: <= ~6 h (measured end-to-end LB turnaround); ENDGAME/compute.md's older model says 7.7-8.05 h, and the
parent's own hard cap is 8.0 h (TIME_BUDGET) + a few minutes of writing. N = 1,322 studies is a placeholder;
everything below scales linearly with it.

Fleet stage (2 workers x 4 decode threads, 4 vCPU):
- setup ~1-2 min (torch/timm import, checkpoint load, shard-0 parity re-encode of 4 train studies).
- decode D: 0.68 s/study MEASURED by the CPU prep kernel (4 processes, cv2 1 thread; the worker now also sets cv2 to 1
  thread, K2d) -> 15 min. Central 15-20 min; a slow-storage session (per-file latency x2) ~30 min.
- GPU per T4 (ESTIMATE): 661 studies x N x 0.26-0.40 s = 3-4.4 min (N = 1), 6-9 min (N = 2). Per-worker decode is
  ~1.36 s/study vs GPU 0.26-0.80 s, so 1 or 2 members take the same wall time (decode-bound up to N = 3).
- Fleet stage total: 17-22 min central, ~32 min slow.

Crop stage K=8, 3 members (combined kernel only): 0.43 s/study MEASURED (K1 commit, 3 studies) -> 9.5 min + ~0.5 min
load = ~10 min (~20 min slow). The 10 min start floor does not guarantee completion.

| P | Left at the 8.6 h wall | Fleet-only K2 | Combined (fleet then crop) |
|---|---|---|---|
| 6.0 h (measured turnaround) | 156 min | done by <= 6.6 h | done by <= 6.9 h (27-52 min) |
| 7.7 h (ENDGAME low) | 54 min | fits, also slow | central fits; slow (52 min) borderline: C/CF absent at worst |
| 8.0-8.05 h (parent cap) | 33-36 min | central fits; slow (~32 min) borderline: fleet files absent at worst, `submission.csv` = parent | crop not started or cut: C/CF absent, fleet files fine |

Every "absent" above is the designed fail-loud outcome (a slot on that file errors), never a wrong score. Kaggle's 9 h
limit is never at risk: workers are killed 5 min after the 8.6 h wall and nothing heavy runs after that.

GPU memory: the parent's heavy model work runs in child processes that exit, so both T4s are near-empty when the
stage starts (the reviewer read cuda:0 19 MB allocated / 63 MB reserved, cuda:1 10 / 23 MB from the parent's `[phase]`
lines; not re-read here); a worker needs ~1.5-2.5 GB per T4 (ESTIMATE).

## 7. Gaps (known, not closed tonight)

1. **CUDA paths are unverified until the 3-study commit**: fp16 worker on a T4, two-GPU sharding, the microbench on
   CUDA, the crop stage after the workers. Everything else ran (laptop CPU smoke; box CPU on real data). The A1 GPU
   stage test (box, >= 6 GB free) can cover CUDA/fp16 on the V100 before the push if a training gap appears.
2. **A missing selected file -> error** is ENDGAME's GUESS about Kaggle; the K1 Wave A fleetsolo read settles whether
   non-default names score at all (section 5 table).
3. **D1's paired-bootstrap SD is a constant** (.012 ~ 1.5 x .008) in `d1_pick.py`: the training logs hold macro gold58
   per epoch, not per-study gold58 predictions. The [verify] rule's "rho not higher on gold58 too" is likewise
   approximated by the 867-study rho only.
4. **No ETA abort of the fleet stage** (combined mode could free time for the crop stage by stopping a fleet that
   cannot finish); irrelevant at P <= ~6 h. A stuck decode no longer blocks a worker (K2e skips that one study).
5. **Box full notebook (`run_parent.py`, level B) not run** (GPU busy since 10-03 21:07); prepared only.
6. **Parity re-encode is skipped in every box test** (only 1 of the 4 parity studies has DICOMs on the box); it runs
   on Kaggle and in the laptop smoke (fake store). `k2_check.py --kaggle_commit` makes it a hard gate on the commit.
7. **One member, by rule.** Shipping both label variants needs an explicit re-key (`--ckpt PATH=f0_s42v4`); ENDGAME D1
   says the loser never ships, so `build_k2.py` refuses a second fold-0/seed-42 member without it. If it is ever
   wanted: scp the second variant into its own folder (`$SP/k2/ckpt_v4/`): both runs name their files
   `f0_s42_<tag>_cnxt320x24_cv<NNNN>_best.pt`, so two files with the same tag and cv would overwrite each other.
8. `kaggle kernels output` must return `_k2_parent.csv`, `_k2_stage_fleet.json` and `fleet_receipt.json` for
   `k2_check.py`; if the CLI skips files it cannot read, use the log landmarks alone (the receipt line in the log
   carries the same parity / bench / n_keep facts).
9. **K2d (cv2 1 thread) cannot be proven output-identical on the laptop** (the smoke uses a PIL cv2 shim); the box A0
   with Kaggle's cv2 4.14 and the commit's PARITY re-encode (exit 4 on a mismatch) are its guards.
