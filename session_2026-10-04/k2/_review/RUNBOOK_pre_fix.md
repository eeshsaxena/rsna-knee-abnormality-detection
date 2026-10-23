# K2 RUNBOOK: fold-0 ConvNeXt member on the 0.943 parent, one commit, several submittable files

Written 2026-10-04 ~03:45 IST. Nothing was pushed, created, versioned or submitted. Eesh does every `kaggle` write.
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

Optional combined kernel (K3 in ENDGAME_V3, `--combined`): + 3 cells = crop setup, the verbatim K1 crop cell (sha
`049033d6002cf9fc`) in a sandbox, the CF writer. Extra files, each fail-loud on the stages it needs:

| File | Content | Exists when |
|---|---|---|
| `submission_cf.csv` | rank_pct(.60 r(P) + .15 r(C) + .25 r(F)) | fleet clean AND crop ok |
| `submission_c15.csv` | rank_pct(.85 r(P) + .15 r(C)) | crop ok |
| `submission_cropsolo.csv` | crop leg alone (raw-prob mean of the crop members) | crop ok |

Both stages share one wall: kernel start (= earliest of the kernel process start and the parent's `T0`) + 8.6 h. The
fleet stage starts only with >= 20 min left; the crop stage only with >= 10 min left. A crop failure never touches a
fleet file (sha256-checked in the smoke and on the box).

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
| `cells/k2_crop_pre_cell.py`, `cells/k2_crop_post_cell.py` | combined mode: sandbox + manifest pin + shared wall; CF writer |
| `code/` | dataset code: `fleet_common.py`, `fleet_model.py`, `fleet_model_f2.py` = byte copies of fleet2 = the box trainer's `~/fleet` (sha `c112dac9...`, `2a3b176d...`); `fleet_worker.py` = fleet2 worker + K2 patch (decode timing, commit-only microbench, `loop_min`; predictions unchanged); `pack_fleet_ds.py` = fleet2 pack tool |
| `k2_check.py` | independent checker of any K2 output dir (smoke, box, downloaded Kaggle commit output) |
| `box/d1_pick.py` | box, read-only: D1 label rule + tag rule + G1 -> the one checkpoint to ship |
| `box/k2_box_stage.py`, `box/make_overlay.sh`, `box/BOX_TEST.md` | box tests (CPU, GPU stage, full notebook via `run_parent.py`) |
| `tests/smoke_k2.py`, `tests/run_scenario.py`, `tests/smoke_transcript.txt` | laptop CPU smoke |
| `_real/store/` | the 4 parity npz (`~/fleet_store/<decode_probe sid>.npz`; the same 4 for both fold-0 runs) |

## 2. Evidence so far (10-04 03:00-03:35)

- Laptop CPU smoke `K2 SMOKE PASS` in 4.2 min (`tests/smoke_transcript.txt`): K2 worker == fleet2 worker (P max abs 0.0,
  same NaN pattern, 24 studies); microbench + decode timing present only when asked; 19 cell scenarios pass: normal,
  3-study commit (microbench from both workers), stub ok, low time (18 min left: nothing launched), worker exit 3,
  time valve (1/3 kept), hard stop, hang (killed after the wall + grace), stale files from an earlier run, injected
  disk-full on the 3rd file (all 4 fail-loud files gone, parent restored), dataset missing, no parent file, broken
  parent; combined: normal (8 files, CF / C / P+F all equal an independent recomputation, fleet files untouched),
  stub ok, fleet fails + crop ok (C files only), crop dataset missing (crop disarmed), crop start floor not met, stale
  CF files. Exceptions never escape a cell.
- `build_k2.py` on the REAL interim box member (`f0_s42_ema_cnxt320x24_cv8183_best.pt`, ep 4, mix50): contract and
  model_sha `fa35880fca963652` match `k2/code`, pack OK, 117.7 MB / 11 files, both kernels built (32 and 35 cells,
  cells 0-30 == parent).
- Box CPU stage test (A0, `box/box_cpu_transcript.txt`): combined notebook's appended cells on 2 real fold-0 val
  studies with the real interim member and the real K1 crop members (Cv1: cropaug_f1, cropaug_f2, crop_f1), Kaggle's
  cv2 4.14: `BOXSTAGE PASS 35/35`; **fleet solo vs the trainer's own OOF on the same studies: max abs 1.3e-4** (serve
  path DICOM -> decode -> worker == training path store -> eval); decode probe match 4/4; crop probes d32 <= 1.8e-7.
- Box full-notebook path prepared (`run_parent.py --prepare-only` via the overlay): 0 unrewritten `/kaggle` in the 4
  appended cells, both datasets linked. Not run (needs an idle V100).

## 3. Which checkpoint (rule, mechanical)

```bash
ssh -o BatchMode=yes training@192.168.10.30 'CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/kmimic/bin/python ~/k2_test/k2/box/d1_pick.py'
```

It prints one `SHIP <path>` line. The rule it applies (ENDGAME_V3 sec 4 D1 + its [verify] correction, sec 5,
fleet2/SCHEDULE 4c):

1. **Tag per run:** EMA, unless the raw `_best.pt` has the higher cv (val AUC vs steven_v4 at its best epoch).
   Never both tags (the pack tool and the worker refuse two files of one run).
2. **Label variant (D1):** per run, solo = gold58 mean over raw AND EMA of the last 3 epochs; rho = mean per-label
   Spearman of the chosen tag's OOF vs the parent's real Raptor v8 arm on the 867 fold-0 val studies; lift = binormal
   best-W lift over .943. **mix50 ships only if** its lift beats v4's by > .001 AND its gold58 edge > .012 AND its rho is
   not higher; both rho > .85 -> v4; anything else -> **v4** (the measured decorrelation prior). The loser never ships.
3. **G1 floor** for the shipped tag at its best epoch: val AUC vs steven_v4 >= .87 and gold58 >= .87. Winner fails and
   the other run passes -> ship the other. Both fail -> **no K2** (EV2S f0 becomes the family switch, ENDGAME G1).
4. It warns `provisional` until both runs have 10 logged epochs. Ship only on the final call (~10:45 IST).

At 03:30 (provisional, ep 4/5): mix50 raw gold58 .900 / rho .873, v4 raw gold58 .836 / rho .811; both lifts 0 at
these solos; v4 failed G1 at ep 4. Do not act on it.

## 4. Tonight, from "fold-0 done" to "submit"

Times assume f0_s42 (mix50) reaches ep 10 ~09:30 and f0_s42_v4 ~10:45 IST (03:00 logs: ~70 min/epoch).

| IST (T = v4 ep 10, ~10:45) | Step | Who | Gate (stop if it fails) |
|---|---|---|---|
| before T (optional, 5 min) | `$PY $SP/k2/tests/smoke_k2.py` re-run if anything in `k2/` changed | Claude | `K2 SMOKE PASS` |
| T+0 (2 min) | `d1_pick.py` (section 3) | Claude/Eesh | no `NOT final`; a `SHIP` line (else no K2 today) |
| T+2 (2 min) | copy the shipped file: `scp -o BatchMode=yes $B:fleet_runs/<run>/<file> "$SP/k2/ckpt/"` (the line `d1_pick.py` prints; `mkdir -p $SP/k2/ckpt` first). Parity npz are already in `$SP/k2/_real/store` | Claude | file size 112,823,466 B |
| T+4 (1 min) | `$PY -W ignore $SP/k2/build_k2.py --ckpt "$SP/k2/ckpt/<file>" --store "$SP/k2/_real/store" --out "$SP/k2/build" --combined` | Claude | preflight line shows the expected run, tag, epoch, `train_labels`, sha `fa35880fca963652`; `upload folder OK ... 11 files`; two kernels built; `BUILD_SUMMARY.json` written |
| T+5 (4 min, parallel) | box A0 CPU test with the shipped member (`box/BOX_TEST.md` sections 0 + A0) | Claude | `BOXSTAGE PASS`; OOF max abs < 2e-2 (tonight's interim: 1.3e-4). A1 GPU gold10 instead only if >= 6 GB free |
| T+5 (5-15 min) | `kaggle datasets create -p "$SP/k2/build/upload/rsna-knee-cnxt-fleet"` then `kaggle datasets status eeshsaxena/rsna-knee-cnxt-fleet` until `ready`. Windows CLI upload bug: pre-create the `%TEMP%\.kaggle\uploads\C_\Users\...` staging dirs (LANE_A_RUNBOOK) | **Eesh** | status `ready` before the push (a push before that mounts nothing: the commit log then says `fleet dataset not attached`) |
| T+20 (1 min) | `kaggle kernels push -p "$SP/k2/build/kernel/rsna-sub-k2-fleet"` (fleet-only K2; NOT the combined kernel today) | **Eesh** | push accepted, version 1 |
| T+20 to T+40 | commit run: parent on the 3 public test studies (~4 min fixed, 242 s measured) + fleet stage (~1-3 min) + queue | Kaggle | `kaggle kernels status eeshsaxena/rsna-sub-k2-fleet` = complete |
| T+40 (5 min) | `kaggle kernels output eeshsaxena/rsna-sub-k2-fleet -p "$SP/k2/build/out_k2v1"`; `$PY $SP/k2/k2_check.py --work "$SP/k2/build/out_k2v1" --expect_fleet clean`; read the log (below) | Claude | `K2CHECK PASS` and every log line below |
| T+45 (~11:30) to 21:30 at the latest | submit (section 5) | **Eesh** | slots left in the UTC day (05:30-05:29 IST, shared with SwaplY) |
| submit + ~6 h | reads -> G2 (fleetsolo) / G3 (main), ENDGAME sec 4 | Claude | |

Commit-log landmarks (`out_k2v1/rsna-sub-k2-fleet.log`), all required:

```
[fleet 0] decode probe vs ckpt (train-side decode): match 4 mismatch 0          (and the same for [fleet 1])
[fleet 0] PARITY re-encode vs stored train bytes: {... 'frac_gt8': <=0.01, ... 'mismatch': []}   (Kaggle has train_series)
[fleet 0] MICROBENCH 5 forwards/member on cuda: ms/member/study {...}           (one per GPU: G-T input g)
[k2-fleet ..] RESULT OK fleet stage clean: 1 member(s) ['f0_s42_<tag>_cnxt320x24_cv<NNNN>_best.pt'], 3 studies, 0 median-filled, views/study >=3.00
[k2-fleet] submission.csv / submission_fleetsolo.csv / submission_w30.csv / submission_w20.csv / submission_w40.csv   rows=3 nan=0   (none ABSENT)
```

With 3 studies the W files usually print `(byte-identical to parent)`: per-label ranks of 3 rows coincide with the
parent's. Expected; the gates are the RESULT line, no ABSENT file and `K2CHECK PASS`.
`RESULT FLEET STAGE NOT CLEAN` -> do not submit; the reason is on that line: `fleet dataset not attached` = pushed
before `ready` (re-push); `worker rc [1, ..]` = read the `[fleet g]` traceback above it (contract / model_sha / timm);
`rc [4, ..]` = PARITY FAIL (Kaggle decode differs from training: stop, investigate); `only N min before the wall`
cannot happen on a 3-study commit.

## 5. What to submit (2 slots, Wave B of ENDGAME_V3 10-04)

```bash
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k2-fleet -v 1 -f <FILE> -m "K2 P+F W.30 f0 <labels> <tag> ep<E>"
```

| K1 filename test (K1 `submission_fleetsolo.csv`, Wave A) | Slot 1 (main, G3) | Slot 2 (solo, G2) |
|---|---|---|
| scored | `submission_w30.csv` (fail-loud main: an error = the fleet did not finish in that rerun, never a silent 0.943) | `submission_fleetsolo.csv` |
| still pending | `submission.csv` (scores under either filename branch) | `submission_fleetsolo.csv` (canary for the stage; also settles the filename question) |
| errored / CLI refused (Branch S) | `submission.csv` only; an exact .943 then carries ENDGAME E2's haircut | none (the solo read needs the Branch S light notebook, not built) |

W: the main is .30 (binormal model + gold58 proxies put best W ~.25-.35 for a ~.92-.93 member; one read resolves only
~+.003 vs ~0). `w20` / `w40` are for a later single read by rule only (G3: d <= -1 with a fine solo -> one `w20`
read), never a W curve. The combined kernel waits for 10-05 (ENDGAME: push K3 `cf` / `c` only after K2 has at least
one scored file); it is already built by `--combined` and runs the same way (`kaggle kernels push -p
"$SP/k2/build/kernel/rsna-sub-k2-fleetcrop"`; its log also needs `[cropfleet ..] manifest ...: 3 member(s) [...]` with
the intended crop version and `[k2-crop ..] RESULT crop ok=True fleet clean=True | C files: written | CF file:
written`). The crop members dataset is attached unpinned: the commit uses whatever version is current at push time.

## 6. Timing budget inside the hidden rerun

Parent <= ~6 h (measured end-to-end LB turnaround). Fleet stage, 1 member, ~1,322 studies: ~3 + max(D 15-20, N x g)
min ~ 20-25 min (decode 0.68 s/study on 4 vCPU MEASURED; g from the commit microbench). Crop stage K=8, 3 members:
~5-10 min. Wall 8.6 h: >= 2 h of margin for the fleet-only K2, ~2 h for the combined kernel. The fleet stage is not
ETA-aborted (P6): if it ever cannot finish, it runs to the wall, its files stay absent and the crop stage gets no time.

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
   cannot finish); irrelevant at P <= ~6 h.
5. **Box full notebook (`run_parent.py`, level B) not run** (GPU busy since 10-03 21:07); prepared only.
6. **Parity re-encode is skipped in every box test** (only 1 of the 4 parity studies has DICOMs on the box); it runs
   on Kaggle and in the laptop smoke (fake store).
7. **One member, by rule.** Shipping both label variants needs an explicit re-key (`--ckpt PATH=f0_s42v4`); ENDGAME D1
   says the loser never ships, so `build_k2.py` refuses a second fold-0/seed-42 member without it.
8. `kaggle kernels output` must return `_k2_parent.csv`, `_k2_stage_fleet.json` and `fleet_receipt.json` for
   `k2_check.py`; if the CLI skips files it cannot read, use the log landmarks alone.
