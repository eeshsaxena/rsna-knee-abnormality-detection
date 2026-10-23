# K4 RUNBOOK: the 5-fold TTR mix50 set (f0..f4_s1043_ttr_mix50), three kernels, one dataset version

Written 2026-10-05 09:15-10:00 IST by Claude. Nothing was created, versioned, pushed or submitted on Kaggle. The
operator (Eesh) runs every `kaggle` command below. No em dashes in this file. Tags: **MEASURED** (log, file, read),
**MODEL** (binormal model on measured reads, `analysis/fleetsim_k4.py`), **GUESS** (judgement).

`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`,
`PY` = `C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe`, `B` = `training@192.168.10.30`
(`ssh -o BatchMode=yes`). Every box step below is CPU only (CUDA hidden, nice 19) and never touches the GPU, the
training chain, `~/fleet_runs`, `~/fleet_store`, `~/bigjump_build` or `~/rsna_train` (read only).

## 0. What K4 is

| Kernel | Slug | MAIN_MODE | submission.csv | Members (K2_CFG MEMBERS) |
|---|---|---|---|---|
| K4-main | `eeshsaxena/rsna-sub-k4-fleet` | strict | rank_pct(.80 r(P) + .20 r(F)), F = raw-prob mean of the set | the set |
| K4-solo | `eeshsaxena/rsna-sub-k4-solo` | solo | F alone (raw probs, parent row order) | the set |
| K4-w50 (optional) | `eeshsaxena/rsna-sub-k4-w50` | strict | rank_pct(.50 r(P) + .50 r(F)) | the set |

- **The set** = ONE checkpoint per FINISHED run `f{0..4}_s1043_ttr_mix50` (TTR, mix50 labels, seed 1043; J1 = fold 0).
  Tag rule as K3: the tag whose best checkpoint has the higher own val cv (exact, from the checkpoint, must equal the
  trainer's closing line), tie -> ema. 1-4 finished folds = an **INTERIM** build, 5 = **FINAL**.
- **M1 (F1 mix50 f0_s42) is OUT by default** (K4 measures the TTR mix50 set alone). `build_k4.py --with_m1` adds it
  to the upload and to every kernel's member list (tested: builds, 2 members, 225.3 MB).
- All kernels read ONE new version of `eeshsaxena/rsna-knee-cnxt-fleet`: 5 code files (byte copies of K3 = box
  `~/bigjump_build/code`: worker `d5edacbe`, `fleet_model_ttr.py` `25b41631`, `fleet_common.py` `c112dac9`,
  `fleet_model.py` `2a3b176d`, `fleet_model_f2.py` `bec2c230`), the N members, `fleet_manifest.json` (+ `sources`
  sha256[:16]), 4 `parity_<sid>.npz`, `dataset-metadata.json` = **11 + N files** (N=1: 117.7 MB, N=2: ~225 MB,
  N=5: ~548 MB). `kaggle datasets version` REPLACES the whole file set.
- Cell 31 = the K3 cell **byte-identical** (`build_k4.py` asserts it), parent cells 0-30 = repro_0946 (sha-pinned),
  kernel metadata (sources, GPU, internet, machine) == the proven K2 kernel (asserted). Strict and solo: on ANY
  failure `submission.csv` is DELETED (the slot errors, never a silent 0.943). Only `submission.csv` is scored.

## 1. Files (`SP/k4/`, a copy of `SP/k3/` without build/ckpt/smoke dirs; k3, k2 and the box run dirs were not edited)

| Path | Role |
|---|---|
| `k4_auto.sh` | laptop: steps 1-3 in ONE command (collect -> build if the set changed -> stage -> A0 x3 -> wait -> verdict); last line `K4AUTO READY <build dir> members <list>` (MEASURED 2 min 18 s end to end with J1, 10-05 09:42) |
| `k4_collect.sh` | laptop: sync box tools -> box pick + copy -> scp members, sha256 laptop == box source -> parity npz sha == `~/fleet_store` -> stale `ckpt/f*_s1043_*` moved to `ckpt_old/` -> `members.json` |
| `box/k4_pick.py` | box picker (generalises `k3_pick_ttr.py` to f0..f4): FINAL = closing `{"raw","ema"}` line + last-epoch lines; checks one best per tag, cv == closing line, OOF mtime 0..5 s after ckpt mtime (VERIFY 10-05: was whole-second equality, ~1.3% false FAIL per pick; save() writes the OOF ~13 ms after the ckpt), fleet_model_ttr, model_sha `f141aeed136dfc01`, ttr 224, fold/seed, train labels `labels_mix50_logit.csv`, sel labels `llm_labels_v4_blend.csv`, fp16, probe npz present; `--copy` -> `~/k4_test/{ckpt,oof,members.json}`. Exit 0 / 1 (check failed) / 3 (nothing final) |
| `members.json` | the set: fold, tag, epoch (1-based), cv, other-tag cv, file, sha256, decode_probe, OOF copy; + `skipped` folds with reason |
| `build_k4.py` | ONE command: upload folder + 3 kernels via `build_k2.py` (unchanged); default out `k4/build` (5 folds) or `k4/build_i<N>` (interim) (+ `_m1`); writes `K4_BUILD.json` (members, kernels, notebook sha) |
| `k4_stage_box.sh` | laptop: stage a build on the box (`~/k4_test/ds`, `~/k4_test/kernel`), whole-folder sha256 box == laptop ("DS IDENTICAL"), A0 study list |
| `box/k4_ids.py` | box: A0 studies = 2 val studies of each member's own fold (from its OOF) with DICOMs on the box (fold 0 = the K3 `two.csv` pair). VERIFY 10-05: the box DICOM mirror holds fold 0 + gold58 only (458 studies, folds 1-4: 0), so folds 1-4 add no study (NOTE line) instead of the old assert, which made every set with a fold 1-4 member FAIL at stage |
| `box/k4_box_launch.sh` | box: A0 CPU tests (`a0_main`, `a0_solo`, `a0_w50`, `all`), per-member OOF parity for every manifest member whose OOF holds an A0 study (fold 0: J1, M1); fold 1-4 members get a NOTE (stderr) and are covered by the npz mean identity, decode probe, n_fail/n_blank, then the Kaggle commit |
| `box/k2_box_stage.py` | K3 harness + K4: member OOFs on disjoint folds no longer fail the mean-of-OOF check (skipped, noted); new check "fleet solo == raw mean of every worker member row" (all studies, < 1e-5) |
| `box/k4_rehearsal5.sh` | box: 5-TTR-member rehearsal of the built K4-main notebook with real checkpoints that exist today (re-keyed f0..f4_s1043); test only, never shipped |
| `k2_check.py` | K3 gate + `--expect_w_main` (W .20 main vs W .50 kernel cannot be mixed up at submit time) |
| `analysis/fleetsim_k4.py` / `.out` | interim-vs-final model + pre-registered read tables (prior/rejection block copied from `jump2/fleetsim.py`) |
| `build_i1/` | INTERIM build with J1 only (built + box-tested 10-05 09:2x) |
| `box/transcripts_k4/` | today's box logs (A0 x3, rehearsal x2, box members.json) |

## 2. Operator steps (identical for an interim and the final build; only the build dir differs)

Git Bash on the laptop unless marked PowerShell. **Shortcut for steps 1-3:** `bash "$SP/k4/k4_auto.sh"` (exit 3 =
nothing final; must end `K4AUTO READY <build dir> members <list>`; then set `BD` to that dir and continue at step 4).

```bash
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe; B=training@192.168.10.30
# 0. where is the chain? (fold F is final once its log ends with {"raw": .., "ema": ..})
ssh -o BatchMode=yes $B 'tail -n 3 ~/fleet_runs/auto_folds.log; for F in 1 2 3 4; do echo "f$F: $(tail -n 1 ~/fleet_runs/f${F}_s1043_ttr_mix50.log 2>/dev/null | cut -c1-90)"; done'

# 1. collect (~10 s + ~30 s per new 108 MB member)          gate: last line "K4COLLECT OK <N> member(s) folds [...]"
bash "$SP/k4/k4_collect.sh"
#    exit 3 = no run final; exit 1 = STOP (a FAIL line names the check). INTERIM is printed when N < 5.

# 2. build (~20 s per member)    gate: "K4 BUILD OK FINAL (5/5 folds)" (or "INTERIM (N/5 folds: [...])") and
#    one "member f<F>_s1043_<tag>_..." line per fold with the epoch / cv that k4_collect printed
$PY -W ignore "$SP/k4/build_k4.py"                      # add --overwrite to rebuild the same dir; --no_w50 to skip K4-w50
BD="$SP/k4/build"                                        # interim: BD="$SP/k4/build_i<N>"
MEM=$($PY -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['members']))" "$BD/K4_BUILD.json" | tr -d '[:cntrl:]')

# 3. box A0 (CPU; 1 member ~3 min, 5 members ~10-15 min)
bash "$SP/k4/k4_stage_box.sh" "$BD"                      # gate: "DS IDENTICAL (<11+N> files)" and "K4STAGE OK"
ssh -o BatchMode=yes $B 'bash ~/k4_test/k4/box/k4_box_launch.sh all' < /dev/null
ssh -o BatchMode=yes $B 'cd ~/k4_test/runs; for n in A0_main A0_solo A0_w50; do f=$(ls -t ${n}_*.log | head -1); echo "== $f"; grep -E "^FAIL|BOXSTAGE" $f | cut -c1-200; done'
#    PASS = every one "BOXSTAGE PASS k/k" and no FAIL line. k = 25 for ANY TTR set (23 + J1 parity + npz mean;
#    folds 1-4 have no DICOMs on the box, so only J1 gets a per-member OOF parity line; MEASURED 25/25 with a
#    simulated 2-member set, VERIFY 10-05), 27 with --with_m1 (+ M1 parity + mean-of-OOFs). Must show
#    "fleet solo == raw mean of the N worker member rows" with N = all members, and "keep N/N" in the log.
```

```powershell
# 4. dataset version (PowerShell). Windows kaggle CLI staging-dir bug: pre-create the staging folder first.
$UP = "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k4\build\upload\rsna-knee-cnxt-fleet"   # interim: ...\k4\build_i<N>\upload\...
New-Item -ItemType Directory -Force -Path (Join-Path "$env:TEMP\.kaggle\uploads" ($UP -replace ':', '_')) | Out-Null
(Get-ChildItem $UP).Count                                   # 11 + N  (16 for the final set)
kaggle datasets version -p $UP -m "K4: TTR mix50 s1043 folds <list> (<tags>) + bigjump worker"
#    a "No such file or directory" under .kaggle\uploads -> create exactly that folder and re-run. ~548 MB: minutes.
# 5. wait for ready + check the NEW version's file list BEFORE any push (an expired token fails silently)
kaggle datasets status eeshsaxena/rsna-knee-cnxt-fleet      # repeat until: ready
kaggle datasets files  eeshsaxena/rsna-knee-cnxt-fleet      # 11 + N files: every f<F>_s1043_*_best.pt of K4_BUILD.json, fleet_model_ttr.py
# 6. push (each prints "Kernel version V successfully pushed": write V down; new slugs V = 1, the final after an interim = V 2)
kaggle kernels push -p "$env:TEMP\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k4\build\kernel\rsna-sub-k4-fleet"
kaggle kernels push -p "$env:TEMP\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k4\build\kernel\rsna-sub-k4-solo"
kaggle kernels push -p "$env:TEMP\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k4\build\kernel\rsna-sub-k4-w50"     # optional
# 7. commit runs (~20-40 min incl. queue), then download each version's output to its own dir
kaggle kernels status eeshsaxena/rsna-sub-k4-fleet          # until complete (same for -solo, -w50)
kaggle kernels output eeshsaxena/rsna-sub-k4-fleet -p "$env:TEMP\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k4\build\out_main_v<V>"
kaggle kernels output eeshsaxena/rsna-sub-k4-solo  -p "$env:TEMP\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k4\build\out_solo_v<V>"
kaggle kernels output eeshsaxena/rsna-sub-k4-w50   -p "$env:TEMP\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k4\build\out_w50_v<V>"
```

`$env:TEMP` on this laptop = `C:\Users\eeshs\AppData\Local\Temp`. For an interim build replace `\k4\build\` by
`\k4\build_i<N>\` in steps 4, 6 and 7.

```bash
# 8. gate each commit (Git Bash; SP, PY, BD as above). All must end "K2CHECK PASS".
MEM=$($PY -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['members']))" "$BD/K4_BUILD.json" | tr -d '[:cntrl:]')
MAN="$BD/upload/rsna-knee-cnxt-fleet/fleet_manifest.json"
$PY "$SP/k4/k2_check.py" --work "$BD/out_main_v<V>" --expect_fleet clean --kaggle_commit --expect_mode strict --expect_w_main 0.2 --expect_members "$MEM" --expect_manifest "$MAN"
$PY "$SP/k4/k2_check.py" --work "$BD/out_solo_v<V>" --expect_fleet clean --kaggle_commit --expect_mode solo                     --expect_members "$MEM" --expect_manifest "$MAN"
$PY "$SP/k4/k2_check.py" --work "$BD/out_w50_v<V>"  --expect_fleet clean --kaggle_commit --expect_mode strict --expect_w_main 0.5 --expect_members "$MEM" --expect_manifest "$MAN"
```

Covered by the gate: attached manifest names AND per-member sha256[:16] == this build (proves the commit ran these
bytes, not an older same-name file), mode, W_MAIN, member names and count, `submission.csv` == its twin (main:
`submission_w20.csv`; solo: `submission_fleetsolo.csv`; w50: `submission_w50.csv`), every wXX == an independent
recomputation, 2 GPUs, decode probe match >= 1 / mismatch 0 per worker, shard-0 PARITY re-encode (frac_gt8 <= .01,
no mismatch), one microbench per GPU, n_fail = n_blank = 0, schema / row order / finite / [0, 1].
Commit log must show `RESULT OK fleet stage clean: N member(s) [...]` and `[fleet g] members: ...[KneeFleetNetTTR] x N`.
Stop signs: `RESULT FLEET STAGE NOT CLEAN` or `submission.csv ABSENT`; `MEMBERS [...] not in the fleet manifest` = the
old dataset version was mounted (push again after `ready`, version V+1); `worker rc [1, ..]` (traceback above it);
`rc [4, ..]` = PARITY FAIL (Kaggle decode differs: stop).

```powershell
# 9. submit (only versions that passed step 8; the UTC day ends 05:30 IST)
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k4-fleet -v <V> -f submission.csv -m "K4 strict P+F W.20: TTR mix50 s1043 folds <list> (<N> members)"
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k4-solo  -v <V> -f submission.csv -m "K4 SOLO: TTR mix50 s1043 folds <list>"
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k4-w50   -v <V> -f submission.csv -m "K4 strict P+F W.50: TTR mix50 s1043 folds <list>"   # optional 3rd slot
```

Already-committed kernel versions keep their pinned dataset version, so an interim submission stays valid after the
final dataset version replaces the files; the final kernels are new versions (V 2) of the same slugs.

## 3. Runtime check (hidden rerun, 2 x T4, 9 h; MODEL on MEASURED inputs)

MEASURED (K3 commit receipt `SP/k3/build/out_main_v1/fleet_receipt.json`): TTR 141.9 / 147.6 ms/member/study fp16 on
the two T4s, M1 280.1 / 291.3; decode 1.376 s/study/thread. Parent ~6 h and ~661 studies per T4 (K3 RUNBOOK sec 7).

| Item | 5 TTR members | 5 TTR + M1 (`--with_m1`) |
|---|---|---|
| GPU per T4: N x 0.148 s x 661 | **8.2 min** | 11.4 min |
| decode per worker (member-independent) | 3.8 min ideal (661 / 4 threads x 1.376 s), ~7.6 min at 2x CPU oversubscription; K3 conservative central 15-20 min, slow 32 min | same |
| stage, serial upper bound (decode + GPU) | **~23-28 min central, ~40 min slow** | ~27-31 / ~44 min |
| kernel end at a 6 h parent | **~6.4-6.7 h** of the 8.6 h wall (>= 1.9 h margin) | ~6.5-6.8 h |
| GPU memory per T4 | ~3-4.5 GB (K3 2-member estimate 2.5-4 GB + 4 more resident ConvNeXt-T, GUESS) of 15 GB | ~3.5-5 GB |

Fits easily: the stage needs < 45 min even slow, so the parent may run to ~7.8 h before the wall bites (the stage
only starts with >= 20 min left; a stage that misses the 8.6 h wall leaves NO `submission.csv` in strict / solo = an
error, never a silent parent score). Commit run (3 public studies): K3 workers took 62 s for 2 members (MEASURED),
K4 with 5 ~2-3 min (MODEL). Box CPU rehearsal (MEASURED, irrelevant to T4 speed): 5 members, 2 studies, 7.2 min.

## 4. Pre-registered reads (MODEL, `analysis/fleetsim_k4.out`; display = public LB, 3 decimals)

Posterior after K2 + K3: TTR mix50 member solo .929/.933/.937 (10/50/90), r_LB .908/.925/.939; fold-fold r_mm
.957/.962/.965 (prior GUESS .94-.965). Expected reads:

| Set / kernel | display 10/50/90 | P(<=.942) / P(.943) / P(.944) / P(>=.945) | E[display] | true lift vs parent (median) |
|---|---|---|---|---|
| 5f SOLO | .932/.936/.939 | n/a | .9357 | n/a |
| **5f K4-main W .20** | .943/.943/.944 | .07 / .51 / .39 / .03 | **.9434** | +.0004 |
| 5f K4-w50 W .50 | .941/.942/.944 | .57 / .22 / .13 / .07 | .9423 | -.0008 |
| 2f (J1 + fold 1) W .20 | .943/.943/.944 | .08 / .52 / .37 / .03 | .9433 | +.0003 |
| 1f (J1 only) W .20 | .943/.943/.944 | .09 / .56 / .33 / .02 | .9433 | +.0003 |

**How to read each slot** (5f; the 2f / 1f rows of `fleetsim_k4.out` are within .0002):
- **SOLO pins member strength** (s_Tm sd .0030 -> .0003-.0004). 5f SOLO <= .932 -> the 5-fold W .20 true lift is
  ~-.0002 (the mix50 TTR set adds nothing over K2); .934-.936 -> +.0001 to +.0004; >= .938 -> +.0008 or more
  (display .944 likely). The member-solo GUESS .9325 in `jump2/blendmath.md` is replaced by this read.
- **W .20 (score slot)**: .944 (P .39) = out of the .943 tie, true lift ~+.0009; .943 (P .51) = +.0001 (K4 re-measures
  K2/K3); .942 (P .07) = the set hurts (lift ~-.0006): stop adding mix50 members.
- **W .50 (information slot)**: in this model the W .50 read tracks fleet STRENGTH more than r (r is tied to the K2
  calibration): .941 -> fleet solo ~.933, .942 -> ~.935, .943 -> ~.937, .944 -> ~.939. With the SOLO read in hand, use
  `jump2/blendmath.md` sec 7 for r (assumes fleet solo .936; shift r by about +-.03 per +-.003 of solo): .944 -> r
  .885-.926 (fleet route alive for H2-type mixes); .943 -> r .906-.948; .942 -> r .927-.969; <= .941 -> r >= .948
  (the mix50 fleet is a parent copy: stop adding mix50 members).
- **Pairing** (blendmath sec 7): best expected score = W .20; most information = W .50; either paired with SOLO pins
  the fleet's r to about +-.025.

## 5. Interim K4 today vs waiting for 5 folds (MODEL unless tagged)

Chain timing: fold 1 at epoch 5 of 24 at 09:38 (MEASURED log), 0.48-0.53 s/it while the GPU is shared (0.36 s/it
alone, MEASURED); fold 1 final ~13:30-15:30 IST, fold 4 ~04:00-08:00 IST 10-06 depending on GPU sharing (MODEL).
K4 kit turnaround after a fold lands: collect + build + stage + A0 ~10-20 min (MEASURED pieces today: collect 9 s,
build 22 s for 1 member, stage 4 s, A0 2 min for 1 member / 7 min for 5); then dataset version + ready ~10-20 min,
commits ~20-40 min, gates ~5 min: **submit ~1-1.5 h after the fold's closing line**.

| Option | ready to submit | W .20 E[display] | P(.944+) | what its SOLO pins |
|---|---|---|---|---|
| **K4-i1 (J1 only, `build_i1/`, already built + A0 PASS)** | now (~10:30-11:00 after the Kaggle steps) | .9433 | .35 | member strength (sd .003 -> ~.0004) |
| K4-i2 (J1 + fold 1) | ~15:00-17:00 IST | .9433 | .40 | same + a 2-fold mean |
| K4 final (5 folds) | ~05:30-09:30 IST 10-06 | .9434 | .42 | same |

- **Score value of waiting is ~zero:** 5f minus 2f true lift at W .20 = +.0000 (10/50/90 .0000/.0000/.0001); the two
  W .20 displays agree in 96 % of draws. Closed-form sensitivity at central values: for fold-fold r_mm .90 / .94 /
  .975 the 5f - 2f gap at W .20 is +.00012 / +.00007 / +.00003 (at W .50: +.0007 / +.0004 / +.0002, so a W .50 read
  of an interim set is NOT the same measurement as the final W .50 read).
- **Information value of an interim is nearly all of K4's:** an interim SOLO pins the TTR mix50 member strength as
  well as the 5f SOLO does (sd .0030 -> .0003 vs .0004) and shrinks the uncertainty of the final W .20 lift as much
  (.0006 -> .0004); an interim W .50 shrinks it to .0002, same as the final W .50.
- **Decision value (GUESS, the main argument for an interim):** reads come back ~6 h after submission. An interim SOLO
  read today (~17:00-23:00 IST) says whether the TTR mix50 member is strong (SOLO >= .935) or not (<= .932) while
  folds 2-4 still have ~15 h of V100 time ahead; `jump2/blendmath.md` sec 8 already says more mix50 members beyond
  the 5 folds have ~zero value, and a weak read would argue for moving the box to TTR v4 folds (the clean
  self-teacher path) after fold 2 instead of fold 4. The chain itself is the operator's call (PAUSE_CHAIN).
- **Slots:** an interim today and a final before 05:30 IST 10-06 come out of the same UTC day; the final most likely
  lands after 05:30 IST (new day). Suggested pairing (GUESS): interim = K4-main W .20 + K4-solo; final = K4-main W .20
  + K4-w50 (the SOLO is then already known), or final SOLO again if the interim SOLO surprised.

## 6. Evidence (2026-10-05, all MEASURED)

- **Picker on the real runs** (`box/k4_pick.py`): fold 0 FINAL, raw .883599 (epoch 17) > ema .882400 (epoch 24) ->
  `f0_s1043_raw_ttr224-320x24_cv8835_best.pt`, sha256 `dbd5bf6a20fa0683`, fleet_model_ttr `f141aeed136dfc01`, ttr
  224, OOF mtime == ckpt mtime; folds 1-4 SKIPPED (fold 1 not finished, 2-4 not started). Exit 0.
- **Collect**: laptop sha256 == box source; the 4 probe npz in `k4/_real/store` == `~/fleet_store` (sha256).
- **Build `build_i1/`**: 117.7 MB, 12 files; notebooks sha256[:16] main `9636e4896e73518a`, solo `96e9eb54d2837f2c`,
  w50 `b4e76971babc4cad` (deterministic: identical after a rebuild); code == K3 == box bigjump; cell 31 == K3 cell.
- **Box A0** (CPU, kmimic env, real DICOMs, the 2 fold-0 val studies of K3 `two.csv`): main, solo and w50 each
  `BOXSTAGE PASS 25/25`; J1 vs its trainer OOF max|diff| 3.07e-4; fleet solo == npz member mean 1.1e-16; decode probe
  match 4 mismatch 0; attached manifest sha == build. `k2_check.py` with `--expect_mode --expect_w_main
  --expect_members --expect_manifest`: PASS 28/28 on all three; negative controls fail as designed (solo output
  checked as strict 25/26; w50 output checked as W .20 27/28).
- **5-member rehearsal** (`box/k4_rehearsal5.sh`, built K4-main notebook with 5 real TTR checkpoints: J1 raw/ema, TTR
  v4 raw/ema, J1 epoch-20 ema): with their original names the worker REFUSED them (`duplicate fold/seed members (raw
  AND ema?)`, strict deleted `submission.csv`) = the one-tag-per-run guard works; re-keyed to the K4 pattern
  f0..f4_s1043: `BOXSTAGE PASS 30/30`, `keep 5/5`, member filter with 5 names, 5-entry manifest sha == attached,
  per-member OOF parity 2.7e-4 .. 4.8e-4, fleet solo vs mean of the 5 OOFs 1.9e-4, npz mean identity 6.3e-17.
- **Builder refusals / options**: a stale extra `f1_s1043_*` file in `k4/ckpt` -> refused; `--with_m1` -> builds
  ([M1, J1], 225.3 MB, 13 files).
- Transcripts: `box/transcripts_k4/` (A0 x3, rehearsal x2, box members.json).

## 7. Gaps / caveats

1. **2 x T4 CUDA with 5 TTR members is untested until the commit** (K3 ran 2 members on 2 T4 cleanly; CPU paths ran
   here with 1 and 5 members). `k2_check.py --kaggle_commit` gates it.
2. **The final 5-fold A0 has not run** (folds 1-4 do not exist yet). VERIFY 10-05: the box has DICOMs for fold 0 +
   gold58 only, so step 3 runs all N members on the 2 fold-0 studies: per-member OOF parity for J1 only; folds 1-4
   are covered by sha256, the picker's model_sha / ttr / fold checks, the decode probe, the npz mean identity and
   n_fail/n_blank on the box, and by `k2_check --kaggle_commit` on Kaggle. No member-specific parity for folds 1-4.
3. **Parity re-encode is skipped on the box** (DICOMs for 1 of the 4 probe studies only); it runs on the Kaggle
   commit (shard 0, exit 4 on mismatch).
4. **Fold ETAs are uncertain**: fold 1 ran at 0.48-0.53 s/it this morning (TTA bench, then another ~8.9 GB GPU
   process from another workflow) vs 0.36 alone.
5. Fold runs re-save their best file under the SAME name when cv improves in the 5th digit; the picker only reads a
   run after its closing line, and every copy is checked by sha256, never by name.
6. Laptop smoke (`tests/smoke_k2.py`) not re-run: cell and worker are byte-identical to K3; the only gate change
   (`--expect_w_main`) is an added optional check, exercised on the box.
7. `fleetsim_k4` treats fold members as exchangeable with r_mm from a GUESS prior; the gold74 J1-vs-fold-1 r_mm
   measurement (`jump2/blendmath.md` 10.2) refines it once fold 1 lands. Every number in sec 4-5 is MODEL.
