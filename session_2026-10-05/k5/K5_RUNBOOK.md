# K5 RUNBOOK: the 5-fold TTR mix50 set served with TTA A+B (+ optional M1, pass A only), three kernels, one dataset version

Written 2026-10-05 10:20-12:00 IST by Claude. Nothing was created, versioned, pushed or submitted on Kaggle. The
operator (Eesh) runs every `kaggle` command below. No em dashes in this file. Tags: **MEASURED** (log, file, read),
**MODEL** (arithmetic on measured inputs), **GUESS** (judgement).

`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`,
`PY` = `C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe`, `B` = `training@192.168.10.30`
(`ssh -o BatchMode=yes`, append `< /dev/null`). Box steps are CPU only (CUDA hidden, nice 19) except the one GPU
parity check (sec 6), which ran under `flock ~/jump2/gpu.lock` capped at 4.0 GiB. Nothing here touches the training
chain, `~/fleet_runs`, `~/fleet_store`, `~/bigjump_build` or `~/rsna_train` (read only); k2 / k3 / k4 were not edited.

## 0. What K5 is

K5 = K4 (parent + the raw-prob mean of the TTR mix50 fold members, f{0..4}_s1043_ttr_mix50) with **per-member TTA**
(BUILD.md item B6): every TTR member is served **A+B** = mean of pass A and pass B probabilities, pass B = the same
decoded study through `mirror_views` + slice offset **+HALF, HALF = 0.5 - 1e-3** (tta_bench.py FIX-1). M1 (F1 mix50
f0_s42 raw), only with `--with_m1`, is served **pass A only**. Source of the policy (MEASURED, `~/jump2/tta/run1`,
tta_bench over the worker path, 867 fold-0 val + 58 gold58 studies): J1 (TTR mix50 f0) A+B - A = **+.0020 val
(P 1.00), gold58 +.0015 -> PASS**; M1 A+B - A = +.0014 val, gold58 **-.0013 -> FAIL**.

| Kernel | Slug | MAIN_MODE | submission.csv | Members (K2_CFG MEMBERS) |
|---|---|---|---|---|
| K5-main | `eeshsaxena/rsna-sub-k5-fleet` | strict | rank_pct(.80 r(P) + .20 r(F)) | the set (TTR AB [, M1 A]) |
| K5-solo | `eeshsaxena/rsna-sub-k5-solo` | solo | F alone (raw probs, parent row order) | the set |
| K5-w50 (optional) | `eeshsaxena/rsna-sub-k5-w50` | strict | rank_pct(.50 r(P) + .50 r(F)) | the set |

- **The set** = ONE checkpoint per FINISHED run `f{0..4}_s1043_ttr_mix50` (tag with the higher own val cv, tie -> ema),
  picked by `box/k5_pick.py`: folds 1-2 from the box chain, folds 3-4 from the Kaggle training notebook
  (`kaggle_train/RUNBOOK.md` step 6 copies them into the `~/fleet_runs` placeholders + the run receipt to
  `~/fleet_runs/kaggle_f34_run_receipt.json`). 1-4 folds = an **INTERIM** build, 5 = **FINAL**.
- F = raw-prob mean over members of each member's served row (TTR: (A + B) / 2; M1: A).
- All kernels read ONE new version of `eeshsaxena/rsna-knee-cnxt-fleet`: 5 code files (`fleet_common.py`,
  `fleet_model.py`, `fleet_model_f2.py`, `fleet_model_ttr.py` byte-identical to K3/K4 = box `~/bigjump_build/code`,
  model_sha `f141aeed136dfc01` unchanged; `fleet_worker.py` = the **K5 worker**, sha256[:16] `21effe04f052c18d`),
  the N members, `fleet_manifest.json` (sources: sha256[:16] + **`tta`** per member), 4 `parity_<sid>.npz`,
  `dataset-metadata.json` = **11 + N files** (N=1: 117.7 MB, J1 + M1: 225.3 MB, 5 TTR: ~548 MB, 5 TTR + M1: ~656 MB).
- Cell 31 = the **K5 cell** = the K3/K4 cell + ONE additive block `K5a` (TTA gate + receipt, incl. K5b) + one call
  line; `build_k5.py` asserts that stripping the block gives the K3 cell byte for byte. **K5b**: every K5 kernel carries
  its own expectation `K2_CFG['TTA']` (the build's map) and the cell fails loud unless the attached manifest's tta
  equals it: a K5 kernel that mounts the K4 dataset version (the SAME J1 bytes, sha256 `dbd5bf6a..`, so the K3c sha
  check cannot tell) errors instead of silently serving pass A only. Parent cells 0-30 = repro_0946
  (sha-pinned), kernel metadata == the proven K2 kernel (asserted). Strict and solo: on ANY failure `submission.csv`
  is DELETED (the slot errors, never a silent 0.943).

## 1. Files (`SP/k5/`, a copy of `SP/k4/` without its build / ckpt_old dirs; k4, k3, k2 untouched)

| Path | Role |
|---|---|
| `code/fleet_worker.py` | **K5 worker**: K5-TTA per-member mode from `sources[name]['tta']` (`A` default, `AB`), one decode per study, pass B = `sample_triplets(stack, 24, HALF)` + `mirror_views`, served row `(pA + pB) / 2` (float32, == tta_bench `A+B`); K5-REC npz key `tta` (modes, half, rows, forwards per member per pass, microbench passes) + `PA`/`PB` only when a member is AB; K5-MB microbench times every served pass; K5-VALVE cost-aware valve (disabled by the cell, P6). Any other mode -> exit 1 |
| `code/pack_fleet_ds.py` | `--tta A,AB,...` (one per member) writes `sources[name]['tta']`; without it the manifest is exactly as before |
| `cells/k2_fleet_cell.py` | K5 cell: `_k2_tta` gate after the hard-stop gate: K5b kernel expectation `K2_CFG['TTA']` == the attached manifest's tta (when the kernel has one), every worker's modes == the manifest's, AB members ran pass B on every written row, A members never, HALF == .499; `rec['tta']` (+ `tta_kernel`) in `fleet_receipt.json`; log line `[k2-fleet ..] TTA modes ...`. A pre-K5 worker with an AB manifest fails loud |
| `make_k2_kernel.py` | + `--tta NAME=MODE,..` -> `K2_CFG['TTA']` (K5b; only when given, older kernels unchanged) |
| `build_k2.py` | + `--tta NAME=MODE,...` (every member, refused otherwise), manifest tta verified, K5 worker required when any AB, map kept for `--kernel_only`, passed to every kernel (K5b) |
| `build_k5.py` | ONE command: upload + 3 kernels; `--tta_ttr AB` (default) / `--tta_m1 A` (default), `--with_m1`, `--no_w50`; out `k5/build` (5 folds) or `k5/build_i<N>`, `+_m1`; writes `K5_BUILD.json` (members, `tta`, `expect_tta`, worker / cell / notebook sha) |
| `k2_check.py` | K4 gate + `--expect_tta NAME=A\|AB,..` or `--expect_tta <K5_BUILD.json>`: kernel `K2_CFG['TTA']` == expected (K5b), receipt modes == expected for exactly the members that ran, HALF, rows == studies - n_fail - n_blank, pass A count == rows per member, pass B count == rows (AB) / 0 (A), with `--kaggle_commit` microbench passes 2 / 1; with `--expect_manifest` also receipt modes == manifest tta |
| `k5_collect.sh` | laptop: = k4_collect with `~/k5_test`, `k5/`, `box/k5_pick.py`; env `K5_MIN_EPOCHS`, `K5_PICK_ARGS` pass through |
| `box/k5_pick.py` | = k4_pick + **K5-E** (E from the log header, == ckpt args epochs; last-epoch lines `ep E-1`; E < 16 FAILS unless `--min_epochs`), **K5-OOF** (the picked OOF rescored exactly as the trainer must reproduce the ckpt cv to 1e-12, every run), **K5-KAG** (fold listed in the Kaggle run receipt: mtime rule replaced by ckpt + OOF sha256 == receipt; `--kaggle_folds` without a receipt), **K5-ROOT** (`--extra_root`) |
| `k5_stage_box.sh` | laptop: stage a build on the box (`~/k5_test/ds`, `~/k5_test/kernel`, `~/k5_test/K5_BUILD.json`), "DS IDENTICAL", A0 ids |
| `box/k5_box_launch.sh` | box A0 x3 (CPU) on the K5 notebooks, `--expect_tta ~/k5_test/K5_BUILD.json` |
| `box/k2_box_stage.py` | + `--expect_tta`; per-member OOF parity on the PASS A row (`PA`); K5 npz identity (AB: P == (PA + PB) / 2 exactly, A: P == PA, PB NaN); microbench passes |
| `box/k5_parity.py` | the K5 parity tests (sec 6): `--mode cpu` / `--mode gpu` (under flock, 4 GiB cap) |
| `box/t_k5_pick.py` | 10 simulated Kaggle-filled runs through k5_pick.py (sec 6) |
| `k5_auto.sh` | laptop: steps 1-3 in one command (`K5_WITH_M1=1` for the M1 variant); last line `K5AUTO READY <dir> members <list> tta <map>` |
| `tests/smoke_k2.py`, `tests/run_scenario.py` | laptop smoke = every K3 scenario + K5 (sec 6) |
| `build_i1/`, `build_i1_m1/` | INTERIM builds: J1 AB; M1 A + J1 AB (built 10-05, A0 on the box) |
| `box/transcripts_k5/` | today's logs: parity CPU / GPU (+ reports), A0 x3 for both builds, t_k5_pick, k5_auto, smoke |
| `_dev/` | dev helpers that generated k5_pick / k2_box_stage / smoke patches and the K2e / D re-runs (not shipped) |
| `_from_k4/` | the K4/K3-era scripts that came with the copy (they target `SP/k4` and `~/k4_test`): reference only, never run from k5 |

## 2. Operator steps (identical for an interim and the final build; only the build dir differs)

Git Bash on the laptop unless marked PowerShell. **Shortcut for steps 1-3:** `bash "$SP/k5/k5_auto.sh"` (or
`K5_WITH_M1=1 bash ...`); it must end `K5AUTO READY <build dir> members <list> tta <map>`; set `BD` and go to step 4.

```bash
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe; B=training@192.168.10.30
# 0. where are the folds? (final = the run's .log ends with {"raw": .., "ema": ..}); Kaggle folds 3-4: after
#    kaggle_train/RUNBOOK.md step 6 (files in the ~/fleet_runs placeholders + ~/fleet_runs/kaggle_f34_run_receipt.json)
ssh -o BatchMode=yes $B 'for F in 0 1 2 3 4; do echo "f$F: $(tail -n 1 ~/fleet_runs/f${F}_s1043_ttr_mix50.log 2>/dev/null | cut -c1-90)"; done; ls -l ~/fleet_runs/kaggle_f34_run_receipt.json' < /dev/null

# 1. collect                       gate: last line "K5COLLECT OK <N> member(s) folds [...]"
bash "$SP/k5/k5_collect.sh"
#    exit 3 = nothing final; exit 1 = STOP (the FAIL line names the check). Every member line shows "source box|kaggle"
#    and "OOF rescored <cv>" (== its cv). A Kaggle fold with E < 16 FAILS: operator's call (kaggle_train RUNBOOK sec 5),
#    accept with K5_MIN_EPOCHS=<E> bash ...; Kaggle files staged elsewhere: K5_PICK_ARGS="--extra_root <dir>" bash ...

# 2. build (~20 s per member)      gate: "K5 BUILD OK FINAL (5/5 folds)" (or INTERIM), "expect_tta: ..." (every TTR
#    member =AB, M1 =A)
$PY -W ignore "$SP/k5/build_k5.py"                      # --with_m1 to add M1 (pass A); --overwrite; --no_w50
BD="$SP/k5/build"                                        # interim: "$SP/k5/build_i<N>"; with M1: ..._m1
MEM=$($PY -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['members']))" "$BD/K5_BUILD.json" | tr -d '[:cntrl:]')

# 3. box A0 (CPU; J1 AB 2.5 min, J1 AB + M1 5.5 min MEASURED; 5 AB members (+ M1) ~12-20 min: AB doubles TTR forwards)
bash "$SP/k5/k5_stage_box.sh" "$BD"                      # gate: "DS IDENTICAL (<11+N> files)", "tta {...}", "K5STAGE OK"
ssh -o BatchMode=yes $B 'bash ~/k5_test/k5/box/k5_box_launch.sh all' < /dev/null
ssh -o BatchMode=yes $B 'cd ~/k5_test/runs; for n in A0_main A0_solo A0_w50; do f=$(ls -t ${n}_*.log | head -1); echo "== $f"; grep -E "^FAIL|BOXSTAGE|TTA modes" $f | cut -c1-200; done' < /dev/null
#    PASS = every one "BOXSTAGE PASS k/k", no FAIL line, and "[k2-fleet ..] TTA modes {...}" == K5_BUILD.json tta.
```

```powershell
# 4. dataset version (PowerShell). Windows kaggle CLI staging-dir bug: pre-create the staging folder first.
$UP = "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k5\build\upload\rsna-knee-cnxt-fleet"   # interim / m1: ...\k5\build_i<N>[_m1]\upload\...
New-Item -ItemType Directory -Force -Path (Join-Path "$env:TEMP\.kaggle\uploads" ($UP -replace ':', '_')) | Out-Null
(Get-ChildItem $UP).Count                                   # 11 + N
kaggle datasets version -p $UP -m "K5: TTR mix50 s1043 folds <list> served A+B (+ M1 pass A) + K5 TTA worker"
# 5. wait for ready + check the NEW version's file list BEFORE any push
kaggle datasets status eeshsaxena/rsna-knee-cnxt-fleet      # repeat until: ready
kaggle datasets files  eeshsaxena/rsna-knee-cnxt-fleet      # 11 + N files: every member of K5_BUILD.json, fleet_model_ttr.py
# 6. push (write each version V down)
kaggle kernels push -p "$env:TEMP\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k5\build\kernel\rsna-sub-k5-fleet"
kaggle kernels push -p "$env:TEMP\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k5\build\kernel\rsna-sub-k5-solo"
kaggle kernels push -p "$env:TEMP\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k5\build\kernel\rsna-sub-k5-w50"     # optional
# 7. commit runs (~20-40 min incl. queue), then download each version's output to its own dir
kaggle kernels status eeshsaxena/rsna-sub-k5-fleet          # until complete (same for -solo, -w50)
kaggle kernels output eeshsaxena/rsna-sub-k5-fleet -p "$env:TEMP\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k5\build\out_main_v<V>"
kaggle kernels output eeshsaxena/rsna-sub-k5-solo  -p "$env:TEMP\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k5\build\out_solo_v<V>"
kaggle kernels output eeshsaxena/rsna-sub-k5-w50   -p "$env:TEMP\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k5\build\out_w50_v<V>"
```

`$env:TEMP` = `C:\Users\eeshs\AppData\Local\Temp`. For an interim / M1 build replace `\k5\build\` by
`\k5\build_i<N>[_m1]\` in steps 4, 6 and 7.

```bash
# 8. gate each commit (Git Bash). All must end "K2CHECK PASS".
MAN="$BD/upload/rsna-knee-cnxt-fleet/fleet_manifest.json"
$PY "$SP/k5/k2_check.py" --work "$BD/out_main_v<V>" --expect_fleet clean --kaggle_commit --expect_mode strict --expect_w_main 0.2 --expect_members "$MEM" --expect_manifest "$MAN" --expect_tta "$BD/K5_BUILD.json"
$PY "$SP/k5/k2_check.py" --work "$BD/out_solo_v<V>" --expect_fleet clean --kaggle_commit --expect_mode solo                     --expect_members "$MEM" --expect_manifest "$MAN" --expect_tta "$BD/K5_BUILD.json"
$PY "$SP/k5/k2_check.py" --work "$BD/out_w50_v<V>"  --expect_fleet clean --kaggle_commit --expect_mode strict --expect_w_main 0.5 --expect_members "$MEM" --expect_manifest "$MAN" --expect_tta "$BD/K5_BUILD.json"
```

Covered by the gate: everything K4 covered (manifest names + sha256[:16] == this build, mode, W_MAIN, members, twins,
independent blend recomputation, 2 GPUs, decode probe, shard-0 PARITY re-encode, microbench per GPU, n_fail = n_blank
= 0, schema) **plus the K5 TTA lines**: receipt TTA modes == `K5_BUILD.json` tta for exactly the members that ran and
== the attached manifest's `tta`, HALF .499, rows == 3 (the commit's studies) - n_fail - n_blank, per member pass A
count == rows and pass B count == rows (AB) / 0 (A), microbench passes 2 (AB) / 1 (A). **K5c** (added by the
adversarial verify, 10-05): the same gate also reads the worker npz files the kernel leaves in its output
(`_fleet_0.npz`, `_fleet_1.npz`; the download must keep them): one per GPU, every AB member `P == (PA + PB) / 2`
exactly on its written rows with `PB != PA`, every A member `P == PA` with pass B never run, and
`submission_fleetsolo.csv` == the members' raw mean of those `P` rows. So the gate checks the served numbers, not only
the receipt's counters. Three `K5c` PASS lines per commit; a missing npz is a FAIL under `--kaggle_commit`.

### 2.1 The TTA lines the commit log must show (each worker prints its own `[fleet 0]` / `[fleet 1]` copy)

```
[fleet g] members: <M1>[KneeFleetNet], f0_s1043_raw_...[KneeFleetNetTTR], ...          (one [..TTR] per TTR member)
[fleet g] TTA modes <M1>=A, f0_s1043_raw_ttr224-320x24_cv8835_best.pt=AB, ... | AB = mean of pass A and pass B probs, pass B = mirror_views + slice offset +0.499
[fleet g] MICROBENCH 5 rounds/member on cuda: ms/member/study over all served passes (passes {<TTR>: 2, ..., <M1>: 1}) {...}
[fleet g] TTA forwards {<TTR>: {'A': n_g, 'B': n_g}, ..., <M1>: {'A': n_g, 'B': 0}} over n_g rows written (modes {...})
[k2-fleet HH:MM:SS] TTA modes {... AB ..., <M1>: 'A'} | half 0.499 | 3 rows | forwards per member {... {'A': 3, 'B': 3} ...}
[k2-fleet HH:MM:SS] RESULT OK fleet stage clean: N member(s) [...]
```

n_0 + n_1 = 3 on the commit (shards of the 3 public studies). Stop signs (K4 ones still apply): `RESULT FLEET STAGE NOT
CLEAN` / `submission.csv ABSENT`; `worker TTA modes ... != manifest` (a worker read another manifest); `manifest asks
TTA ... but no worker reports TTA (pre-K5 worker in the dataset?)` (an old dataset version is mounted: push again
after `ready`); `manifest tta must be A or AB` (worker rc 1); `TTA forward counts ... do not match` (worker bug: stop);
a `TTA modes ... | pass A only (no TTA)` line on a K5 kernel = the manifest lost its tta (wrong build dir uploaded);
`kernel expects TTA {...} but the attached manifest says {...}` (K5b) = a dataset version without the K5 manifest is
mounted (typically the K4 version, pushed before the K5 version was `ready`): push again after `ready` (version V+1).

```powershell
# 9. submit (only versions that passed step 8; the UTC day ends 05:30 IST)
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k5-fleet -v <V> -f submission.csv -m "K5 strict P+F W.20: TTR mix50 s1043 folds <list> A+B TTA (<N> members)"
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k5-solo  -v <V> -f submission.csv -m "K5 SOLO: TTR mix50 s1043 folds <list> A+B TTA"
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k5-w50   -v <V> -f submission.csv -m "K5 strict P+F W.50: TTR mix50 s1043 folds <list> A+B TTA"   # optional
```

## 3. Runtime check (hidden rerun, 2 x T4, 9 h; MODEL on MEASURED inputs)

Inputs. MEASURED K3 commit receipt (`SP/k3/build/out_main_v1/fleet_receipt.json`): TTR 141.9 / 147.6 ms/member/study
fp16 on the two T4s (pass A), M1 280.1 / 291.3, decode 1.376 s/study/thread; parent ~6 h, ~661 studies per T4.
MEASURED today (box V100, shared with the fold-1 trainer, fp16, `k5_parity.py --mode gpu` microbench, 3 rounds):
J1 pass A 158.2 ms -> **J1 A+B 294.9 ms (x1.86)**, M1 (A) 287-308 ms. MEASURED (box Xeon, 1 thread, 20 studies):
pass B input build (sample_triplets +HALF on 4 views, np.stack, mirror_views) **49.5 ms/study median**, max 156 ms
(pass A input 24.3 ms). Below, an AB member costs 2 x its T4 pass-A time (x2.0, conservative vs the x1.86 measured).

| Item (per T4, ~661 studies) | K4: 5 TTR (A) | **K5: 5 TTR (AB)** | K5 + M1 (A) |
|---|---|---|---|
| GPU: N x ms x 661 (slower T4) | 5 x 147.6 -> 8.1 min | 5 x 295.2 -> **16.3 min** | + 291.3 -> **19.5 min** |
| pass B input build on the worker main thread (661 x 0.05-0.1 s) | 0 | 0.6-1.1 min | 0.6-1.1 min |
| decode per worker (member-independent, unchanged) | 3.8 min ideal, ~7.6 at 2x CPU oversubscription; K3 conservative central 15-20 min, slow 32 min | same | same |
| stage, serial upper bound (decode + GPU + pass B build) | ~23-28 / ~40 min | **~32-37 central, ~50 slow** | ~36-41 / ~53 min |
| kernel end at a 6 h parent | ~6.4-6.7 h | **~6.55-6.65 h, slow 6.85 h** | ~6.6-6.7 h, slow 6.9 h |
| margin to the 8.6 h wall | >= 1.9 h | **>= 1.75 h** | >= 1.7 h |
| GPU memory per T4 | ~3-4.5 GB of 15 | same + one 28 MiB uint8 input (passes run one after the other) | ~3.5-5 GB |

- **Fits comfortably.** AB doubles only the TTR forward cost; decode is per study and shared by every member and both
  passes (one decode, pass B reuses the decoded stacks). Serial upper bound slow case with M1 ~53 min, so the parent
  may run to ~7.7 h before the wall bites (the stage starts only with >= 20 min left; a stage that misses the 8.6 h
  wall leaves NO `submission.csv` in strict / solo = an error, never a silent parent score).
- The worker overlaps decode (4 threads, the next 32-study chunk is submitted before the current chunk's forwards) with
  the GPU, so the real stage is nearer max(decode, GPU) + overhead, ~20-25 min central (MODEL). With AB the GPU share
  (16-20 min) becomes comparable to decode; neither is near the wall.
- Commit run (3 studies, microbench 5 rounds with 2 passes per TTR member): K4 MODEL ~2-3 min -> K5 ~3-4 min.
- Box CPU A0 (MEASURED, irrelevant to T4 speed): J1 AB + M1 A, 2 studies, fleet stage 5.4-5.6 min per notebook
  (3 notebooks in parallel); J1 AB only: see sec 6.

## 4. Reads (what changes vs K4)

`SP/k4/K4_RUNBOOK.md` sec 4-5 (fleetsim_k4) stay the reference for member strength and W. TTA shifts the fleet, not the
blend math: per member MEASURED on J1 (fold 0 val, 867 studies) A+B - A = +.0020 val macro AUC (P(>0) 1.00, per label
all but Synovitis up, Contusion +.0075, MCL +.0052), gold58 +.0015 (P .76). GUESS: on a 5-fold mean the TTA gain is
smaller than on one member (folds already average out part of the same variance), so the K4 tables shift by at most
~+.001 member-solo AUC and ~+.0002 true lift at W .20. Read the K5 SOLO as "TTR mix50 set with A+B"; K5 vs K4 SOLO
(if both are submitted) is the only clean TTA read on the LB, and only the 3rd decimal moves.

## 5. Interim vs final

Folds: fold 1 at epoch 9 of 24 at 10:45 IST (MEASURED log, 0.37-0.38 s/it), fold 2 after it on the box (Kaggle
RUNBOOK estimate: fold 2 ends ~19:00-22:00 IST), folds 3-4 on Kaggle (the kernel ends 10.3-11.4 h after it starts
running, kaggle_train RUNBOOK sec 4). The FINAL K5 (5 folds) is shippable 10-06 morning: collect + build + stage + A0
~25-40 min (A0 with 5 AB members on CPU is the long pole), dataset version + ready ~10-20 min, commits ~20-40 min,
gates ~5 min: **submit ~1.5-2 h after the last fold's closing line**. Interim kits already built and A0-tested today:
`build_i1` (J1 AB) and `build_i1_m1` (M1 A + J1 AB); a 1-member K5 differs from K4-i1 only by TTA (the K5 vs K4 SOLO
pair would be a direct LB read of A+B on J1, 3rd decimal only).

## 6. Evidence (2026-10-05, all MEASURED; transcripts in `box/transcripts_k5/`)

**Worker parity on the box** (`box/k5_parity.py`, real DICOMs from the box mirror, real J1 + M1 checkpoints, kmimic env):
- (a) A-only, 2 fold-0 val studies (the K3/K4 A0 pair), J1 + M1, CPU: K5 worker with no `tta` key and with explicit
  `tta A` == the old K4 worker (`~/bigjump_build/code/fleet_worker.py`, md5 `f27d3f1c..`): **P max|diff| 0.0**, NV /
  names / sids / n_keep / n_fail / n_blank equal; the A-only npz has no PA/PB (only the extra `tta` record).
- (a+) mixed J1 AB + M1 A, same 2 studies, CPU: M1 rows == old M1 rows (0.0), J1 pass A (PA) == old J1 rows (0.0),
  J1 P == (PA + PB) / 2 exactly in float32, max|PB - PA| .064.
- (b) J1 AB on 20 fold-0 val studies with DICOMs, CPU fp32: worker A+B vs **tta_bench.py run through its own CLI on the
  same 20 studies** (a folds csv with only these 20 as fold 0; `--passes A,B --device cpu`): **A+B 0.0, pass A 0.0,
  pass B 0.0** (gate 2e-4). J1 pass A vs its trainer OOF 1.2e-3 (CPU fp32 vs GPU fp16 trainer); A+B moves a
  probability by up to .113 vs pass A on these studies.
- GPU check (ONE run, `flock ~/jump2/gpu.lock`, every worker capped at 4.0 GiB via a runner; 12.5 GB free before;
  fold-1 trainer untouched): new worker (M1 A + J1 AB) vs old worker (M1 + J1) on the 20 studies, fp16 autocast:
  M1 rows 0.0, J1 pass A 0.0; **J1 A+B / A / B vs tta_bench run1 (`~/jump2/tta/run1/tta_passes.npz`, GPU fp16):
  0.0 / 0.0 / 0.0**; GPU fp16 A+B vs CPU fp32 tta_bench A+B 8.6e-4 (precision note, not a gate). V100 microbench:
  J1 158.2 ms (A) -> 294.9 ms (A+B, x1.86), M1 287-308 ms.
- (c) TTA record: mixed run modes {M1: A, J1: AB}, rows 2, forwards {M1: A2 B0, J1: A2 B2}, half .499, bench passes
  {M1: 1, J1: 2}; 20-study run rows 20, forwards A20 B20. `K5PARITY CPU PASS 11/11`, `K5PARITY GPU PASS 4/4`.

**Box A0** (`k5_box_launch.sh all`, CPU, the 3 K5 notebooks, real DICOMs, 2 fold-0 val studies). First round (before
K5b): i1_m1 35/35 x3, i1 33/33 x3 (`*_1041_*`, `*_1048_*` logs). Final round on the K5b rebuilds (`*_1124_*`, `*_1130_*`):
- `build_i1_m1` (M1 A + J1 AB): main / solo / w50 each **BOXSTAGE PASS 36/36** (fleet stage 5.0 min); receipt
  `TTA modes {M1: A, J1: AB} | half 0.499 | 2 rows`; per-member OOF parity on pass A rows M1 5.3e-4, J1 3.07e-4 (== K4);
  mean of the members' pass A rows vs mean of the 2 OOFs 3.8e-4; fleet solo == raw mean of the served rows 1.1e-16;
  K5 npz identity exact; microbench passes {M1: 1, J1: 2}.
- `build_i1` (J1 AB): main / solo / w50 each **BOXSTAGE PASS 34/34** (2.3-2.4 min).
- `k2_check.py --expect_tta <K5_BUILD.json>` (+ mode / W / members / manifest) on the i1 outputs: main 35/35, solo
  34/34, w50 35/35 PASS; i1_m1 main with the M1 A + J1 AB expectation 33/33 PASS. Negative controls FAIL as designed:
  J1=A expected on the AB receipt (kernel TTA, receipt modes and forward counts fail), solo output checked as strict
  (31/32); before K5b also M1=AB expected (26/28).
- **K5b negative** (`NEG_k4ds_1125.log`): the K5-i1 main notebook (expects J1 AB) on the staged **K4** dataset (same J1
  bytes `dbd5bf6a..`, K4 worker, manifest without tta): `RESULT FLEET STAGE NOT CLEAN (STRICT: submission.csv DELETED
  ...): kernel expects TTA {J1: 'AB'} but the attached manifest says {J1: 'A'}`, every output ABSENT (BOXSTAGE 7/7 on
  the fail-loud outcome). Without K5b this run would have been clean with pass A only.

**Picker** (`box/k5_pick.py`): on the real runs fold 0 PICK (J1 raw, OOF rescored cv 0.8835985889396399 == ckpt, diff
0.0), folds 1-4 SKIP. `box/t_k5_pick.py` on simulated Kaggle-filled runs built from copies of the J1 run:
**TK5PICK PASS 10/10** (receipt run picked with mtime rule skipped + sha256 checked; no receipt -> FAIL naming
`--kaggle_receipt`; `--kaggle_folds` -> PICK; wrong receipt sha -> FAIL; wrong OOF under the raw name -> FAIL on the
rescored cv; E 18 -> PICK with epochs 18; header 24 with ep-17 lines -> SKIP; E 12 -> FAIL, with `--min_epochs 12`
PICK; empty placeholder under `--root` + run under `--extra_root` -> PICK).

**Builds** (laptop, final K5b kits): `build_i1` 117.7 MB 12 files, notebooks sha256[:16] main `c3656a25e84ab227`,
solo `c4922acadd04ee55`, w50 `577c33278f1bf51c`; `build_i1_m1` 225.3 MB 13 files, main `e3f18ebf561eacda`, solo
`19b6445c7aa29793`, w50 `b00ecd43ee906368`, manifest `tta` {M1: A, J1: AB}; every K2_CFG carries `TTA` == the build's
map; shipped worker sha256[:16] `21effe04f052c18d`, cell `e37f0604997cd40f`; frozen files == K3; cell == K3 + K5a
(strip check). Builds are deterministic (a rebuild of i1 before K5b gave identical notebook / manifest / worker / cell
sha256). Backward compatibility: the K5 `k2_check.py`
on a K4 A0 output (no TTA in its receipt) passes 27/27 (K4's 26 + "TTA modes == manifest", all A).

**Laptop smoke** (`tests/smoke_k2.py` on the final K5 kit: K5 worker, K5 cell incl. K5b, K5 build_k2 / pack /
make_k2_kernel; tiny 64 px members, CPU; `tests/smoke_run_k5.log` + `smoke_transcript.txt`, copies in
`box/transcripts_k5/*_final*`; 29.9 min, the laptop was thermally throttling; the first run before K5b is in
`tests/smoke_run_k5_first.log`):
- **44/44 scenarios PASS**: all **38 old ones** (the K3 list, unchanged); those that pass `--expect_manifest` carry one
  more check than in K3 (receipt TTA modes == manifest tta): fleet_normal 28/28, fleet_commit3 26/26, comb_normal 45/45,
  strict_ttr 32/32, strict_ttr_all 27/27, ttrsolo_ok 31/31. **6 new**: strict_ttr_tta 40/40 (f1 A + TTR AB, clean,
  `--expect_tta`, P == (PA + PB) / 2 in each worker npz), ttrsolo_tta 38/38, strict_tta_oldworker 12/12 (AB manifest +
  the K3 worker: `manifest asks TTA ... but no worker reports TTA`, submission.csv DELETED), strict_tta_bad 12/12 (mode
  `ABM`: worker rc [1, 1], DELETED), strict_tta_expect_ok 39/39 (K5b kernel expectation == manifest, clean),
  strict_tta_expect_k4ds 12/12 (K5b kernel on a manifest without tta: `kernel expects TTA ...`, DELETED). C2 PASS.
- A / A5 / B / B5 worker checks PASS: K5 worker A-only == the K3/K4 worker on family 1 + TTR (24 studies, P max|diff|
  0.0), == fleet2 and the K2 kit worker on family 1 (0.0); f1 A + TTR AB: TTR pass A == the A-only run (0.0), f1 rows
  == the A-only run (0.0), P == (PA + PB) / 2, record rows 24 / forwards A24 B24; AB microbench passes {f1: 1, ttr: 2}.
- D + D5 PASS: build_k2.py on the real M1 + M2 (+ 5 refusals); `build_k2.py --tta M1=A,M2=AB` writes the manifest tta,
  a partial `--tta` map is refused. (In the first run D5 failed on a harness bug, `--w_main 0.20` with the default
  variants 0.20,0.40 = two weights on one file name, refused by make_k2_kernel as designed; fixed.)
- **The one FAIL: K2e decode-timeout wall-time bound** (logic PASS: slow study skipped, decode_timeout 1, n_fail 1, row
  NaN, NV -1, other 23 rows max|diff| 0.0; but 123 s > the 60 s bound; 88 s in the first run). Not the K5 change:
  re-timed back to back on the throttled laptop, the K3 worker takes 94-115 s on the same slow-study run (K5 worker
  110-123 s) and the K3 worker took 123 s even WITHOUT the slow study (29 s for the same run earlier): the laptop speed
  swung ~4x (the K2 microbench of the same tiny member read 384 ms on 10-04 and 2290-2370 ms here).
- `k5_auto.sh` end to end (J1 interim, before K5b): collect -> build reused (current) -> stage -> A0 x3 BOXSTAGE PASS
  33/33 -> `K5AUTO READY` in 2 min 52 s (`box/transcripts_k5/k5_auto_i1.log`); after K5b the same A0 was re-run by hand
  (34/34 x3, above).

**Adversarial verify** (10-05 12:44-13:40 IST; box scratch `~/k5_adv` only; transcripts in
`box/transcripts_k5/adv_verify/`):
- `k5_parity.py --mode cpu` re-run in a fresh dir: **K5PARITY CPU PASS 11/11** (A-only == old worker 0.0 with and
  without a tta key; mixed M1 A + J1 AB: M1 0.0, J1 PA 0.0, P == (PA + PB) / 2; J1 A+B / A / B vs tta_bench's own CLI on
  the 20 studies 0 / 0 / 0). `--mode gpu` re-run (under `flock ~/jump2/gpu.lock`, 4.0 GiB cap, trainer untouched):
  **PASS 4/4**, J1 A+B / A / B vs tta_bench run1 0 / 0 / 0; V100 J1 A+B / A = x1.99 this time (x1.86 earlier).
- Independent reference (`indep_passB.py`: own triplet sampler and own flips, no `mirror_views`, no tta_bench input
  code, from the worker's own DICOM decode, J1 fp32 CPU): worker PA / PB / P max|diff| **0.0 / 0.0 / 0.0** on 20 studies
  (2 absent views, 6 views with 24 slices, 16-32 slices). Every wrong variant differs from PB by >= .045: mirror only
  .193, offset only .257, exact .5 then mirror .045, mirror then offset .111, -HALF then mirror .110.
- `k2_check.py` gained **K5c** (sec 2 step 8): on the 6 A0 outputs (i1, i1_m1 x main / solo / w50) PASS 32/32 each;
  negative controls FAIL as designed (npz P = PA under an AB receipt; PB = PA = P; npz missing under `--kaggle_commit`;
  J1 = A expected on the AB output). The box kit copy `~/k5_test/k5/k2_check.py` was synced to the laptop file.
- Laptop smoke re-run on the kit with K5c: **K5 SMOKE PASS in 11.1 min, 44/44 scenarios + every A / B / B5 / C2 / D /
  D5 check, incl. K2e (30 s, laptop cool)**; strict_ttr_tta 43/43, ttrsolo_tta 41/41, strict_tta_expect_ok 42/42 (+3 K5c
  checks each). `box/transcripts_k5/adv_verify/smoke_adv.log`.

## 7. Gaps / caveats

1. **2 x T4 CUDA with AB is untested until the commit** (the GPU path ran on the box V100 with fp16 autocast, exact vs
   tta_bench run1; K3 ran 2 members on 2 T4 cleanly). `k2_check.py --kaggle_commit --expect_tta` gates it.
2. **The final 5-fold A0 has not run** (folds 1-4 do not exist yet). The box has DICOMs for fold 0 + gold58 only, so
   folds 1-4 get no member-specific OOF parity on the box (same as K4): they are covered by sha256, the picker's
   model_sha / ttr / fold / OOF-rescore checks, the decode probe, the npz identities and the TTA forward counts.
3. **Kaggle-filled folds 3-4 have not been picked for real**; only simulated (sec 6). If the operator copies them
   somewhere other than the `~/fleet_runs` placeholders, pass `K5_PICK_ARGS="--extra_root <dir>"`; without the run
   receipt next to the runs the mtime rule applies and will most likely FAIL (download / scp mtimes): copy the
   receipt (Kaggle RUNBOOK step 6 does) or pass `--kaggle_folds 3,4`.
4. **The time valve is not family-interleaved** (BUILD.md B6 wording). Member order is unchanged on purpose (the
   non-TTA output must stay byte-identical, and order feeds the float sum of the fleet mean); the valve is now
   forward-cost aware, but the K2 cell disables it anyway (P6, budget 1e12), so it cannot cut a K5 fleet.
5. TTA gain on a 5-fold mean is a GUESS (sec 4): the +.0020 val is one member, one fold.
6. Laptop smoke: the K2e wall-time bound (60 s) failed on the throttled laptop in the build run (logic passed); the
   adversarial-verify re-run on a cool laptop passed it (30 s) with every other check (sec 6).
7. Box scratch left behind (ours, removable): `~/k5_test/pick_sim` (~2 GB of checkpoint copies for the picker
   simulation), `~/k5_test/parity` (FLEET_DIRs of symlinks + npz), `~/k5_test/runs`; `~/k5_adv` (adversarial
   verify: parity re-runs, indep_passB, tampered A0 copies under `neg/`).
8. `k2_check.py --expect_tta` runs its TTA and K5c lines only when the fleet stage is clean; a not-clean commit is
   caught by `--expect_fleet clean` (always passed in step 8), never pass `--expect_tta` without it.
