# K7 RUNBOOK: K5 + the teammate's seed-1044 folds (+ TTR288 fold 0 if it passes), all A+B, strict W .50

Written 2026-10-06 21:40-23:30 IST by Claude. Nothing was created, versioned, pushed or submitted on Kaggle. The
operator (Eesh) runs every `kaggle` command below. No em dashes in this file. Tags: **MEASURED**, **MODEL**, **GUESS**.

`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`,
`PY` = `C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe`, `B` = `training@192.168.10.30`
(`ssh -o BatchMode=yes`, append `< /dev/null`). Box steps are CPU only (CUDA hidden, nice 19).

## 0. What K7 is

| set | runs | trained on | ttr_size | shipped key |
|---|---|---|---|---|
| A | `f{0..4}_s1043_ttr_mix50` (the K5 members; folds 3-4 Kaggle-filled) | box + eeshsaxena Kaggle | 224 | `f{F}_s1043` |
| B | `f{0..4}_s1044_ttr_mix50` | teammate swapiy's Kaggle (`rsna-train-ttr-s1044-{f01,f23,f4}`) | 224 | `f{F}_s1044` |
| C (optional) | `f0_s1044_ttr288mix50` | box | 288 | **`f0_s1044t288`** (re-keyed, see sec 3) |

- mix50 labels only (settled), every member TTR (`fleet_model_ttr`, model_sha `f141aeed136dfc01`), every member A+B TTA.
- Set C ships only when its picked val cv (best of raw / ema, vs v4 selection labels) is **>= .8876** (J1 .8836 + .004,
  the same bar as the box gate `auto_after_ttr288.sh`). `K7_WITH_C=0` excludes it by hand.
- F = raw-prob mean over ALL members (each member 1/N; 11 members: set A 5/11, B 5/11, C 1/11).

| Kernel | Slug | MAIN_MODE | submission.csv |
|---|---|---|---|
| K7-main | `eeshsaxena/rsna-sub-k7-fleet` | strict | rank_pct(.50 r(P) + .50 r(F)), variants w40 / w60 |
| K7-solo | `eeshsaxena/rsna-sub-k7-solo` | solo | F alone |

Both read ONE new version of `eeshsaxena/rsna-knee-cnxt-fleet` (11 + N files). Worker = the **K5 worker, byte for byte**
(sha256[:16] `21effe04f052c18d`, asserted by build_k7.py); cell = the K5 cell (asserted); frozen model files == K3.

## 1. Files (`SP/k7/`, a copy of `SP/k6`; k5 / k6 untouched)

| Path | Role |
|---|---|
| `k7_collect.sh` | laptop: sync tools to `~/$K7T/k5`, one picker run per set (A, B with `--require_receipt` + the 3 s1044 receipts, C with `--ttr_size 288 --no_kaggle_receipt`), merge, fetch + sha-verify checkpoints, parity npz, `k7/members.json`. Unfinished runs are listed as `NOT FINISHED set X ...`. Last line `K7COLLECT OK N member(s): A a/5, B b/5, C c`. Env: `K7T` (box area, default `k7_test`), `K7_ROOT_A/B/C`, `K7_WITH_C`, `K7_C_FOLDS` (default 0), `K7_C_BAR` (.8876), `K5_MIN_EPOCHS` (16), `K7_LAPTOP_DIR` |
| `box/k5_pick.py` | K7 picker = K6 picker + **K7-TTR** `--ttr_size`, **K7-REC** repeatable `--kaggle_receipt` (a receipt counts only if its cfg SEED == `--seed` and its `runs[F]` == this run's folder) + `--require_receipt`, **K7-KEY** recipe-aware stale removal (`--copy` never deletes the other set's fold-0 s1044 pick) + OOF copy `oof_f0_s1044t288_<tag>_cv<NNNN>.csv` for non-224 sets; records carry `suffix` |
| `box/k7_merge.py` | box: merge `members_{A,B,C}.json`; set C bar; refuses duplicate (seed, fold, ttr_size) / duplicate files; `set_c` verdict |
| `build_k7.py` | ONE command: upload + 2 kernels (strict W .50, solo). Member identity (seed, fold, ttr_size); non-224 members re-keyed `PATH=f{F}_s{S}t{N}`; asserts worker == K5 worker, cell == K5 cell; out `k7/build` (A 5/5 and B 5/5) or `k7/build_i<N>`; `K5_BUILD.json` picks carry `staged`, `set`, `ttr_size`, `suffix`, `oof` |
| `build_k2.py` | K6 copy + **K7-KEY**: the preflight run identity of a TTR member includes args ttr_size |
| `k7_stage_box.sh` | laptop: stage a build on the box (`~/$K7T/ds`, `kernel`, `K5_BUILD.json`), copies checkpoints under their shipped name via K5_BUILD.json picks, "DS IDENTICAL", A0 ids |
| `box/k5_box_launch.sh` | box A0 (CPU) on `rsna_sub_k7_fleet` / `rsna_sub_k7_solo`; OOF map via K5_BUILD.json staged names; `K7T` env |
| `box/k5_ids.py` | A0 studies picked PER FOLD (shared by the fold's members), so A f0, B f0 and C f0 all get per-member OOF parity on the same 2 studies |
| `box/t_k7_pick.py` | picker + merge tests on the simulation root (sec 6) |
| `tests/t_k7_conflict.py` | laptop: the B f0 / C f0 conflict tests (worker on tiny real TTR members + build_k2 / build_k7 on the real checkpoints) |
| `tests/smoke_k2.py`, `run_scenario.py` | the K5 laptop smoke, run on the k7 kit |
| `_dev/k7_sim_setup.py` | builds the SIMULATION root `~/k7_sim/root` (test only) |
| `build_sim/`, `_sim/` | the simulated 11-member build and its collect area (TEST ONLY: **never version or push build_sim**) |

## 2. Operator steps

### 2.1 Fetch the teammate's outputs (PowerShell), one notebook at a time as each completes

```powershell
kaggle kernels status swapiy/rsna-train-ttr-s1044-f01      # (and -f23, -f4) until: complete
$KO = "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k7\kaggle_s1044"
kaggle kernels output swapiy/rsna-train-ttr-s1044-f01 -p "$KO\out_f01" --page-size 200 --file-pattern '^(?!.*epochs/.*\.pt$)'
kaggle kernels output swapiy/rsna-train-ttr-s1044-f23 -p "$KO\out_f23" --page-size 200 --file-pattern '^(?!.*epochs/.*\.pt$)'
kaggle kernels output swapiy/rsna-train-ttr-s1044-f4  -p "$KO\out_f4"  --page-size 200 --file-pattern '^(?!.*epochs/.*\.pt$)'
# if a download crawls for > 10 min (vendored _xx_env/ trees, 10-05 gotcha), stop it and use
#   --file-pattern '^(?!_[a-z0-9]+_env/)(?!.*epochs/.*\.pt$)'
python "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\kaggle_train\verify_outputs.py" "$KO\out_f01"
```

`verify_outputs.py` exit 0 = DONE and every present file == the receipt sha256 (`epochs/*.pt` are reported ABSENT on
purpose). Read `plan E` per notebook: f01 / f23 expect E ~22, **f4 expects E ~14-15** (7.0 h budget). The picker FAILS a
run with E < 16 unless the operator accepts it: run the collect with `K5_MIN_EPOCHS=14` (MODEL: a 14-15 epoch cosine is
a slightly weaker member; set A fold 4 peaked at epoch 15/22; GUESS: accept, the 2nd-seed gain is small either way).
Exit 2 (PARTIAL / ABORTED) = do not copy that notebook's folds.

### 2.2 Copy to the box (Git Bash). Receipt naming (decided): `~/fleet_runs/kaggle_s1044_<tag>_run_receipt.json`, tag = f01 / f23 / f4

```bash
KO=$SP/k7/kaggle_s1044; B=training@192.168.10.30
for x in "f01:0 1" "f23:2 3" "f4:4"; do t=${x%%:*}; O=$KO/out_$t
  [ -f "$O/run_receipt.json" ] || { echo "no $O (not fetched yet)"; continue; }
  for F in ${x#*:}; do R=f${F}_s1044_ttr_mix50
    ssh -o BatchMode=yes $B "[ ! -e ~/fleet_runs/$R ] && [ ! -e ~/fleet_runs/$R.log ]" < /dev/null || { echo "STOP: ~/fleet_runs/$R exists"; break 2; }
    ssh -o BatchMode=yes $B "mkdir ~/fleet_runs/$R" < /dev/null
    scp -q -o BatchMode=yes "$O/$R/"*_best.pt "$O/$R/"oof_*.csv $B:fleet_runs/$R/
    scp -q -o BatchMode=yes "$O/$R.log" $B:fleet_runs/$R.log
  done
  scp -q -o BatchMode=yes "$O/run_receipt.json" $B:fleet_runs/kaggle_s1044_${t}_run_receipt.json
  ssh -o BatchMode=yes $B "mkdir -p ~/kt_test" < /dev/null
  scp -q -o BatchMode=yes "$SP/kaggle_train/verify_outputs.py" $B:kt_test/verify_outputs.py
  ssh -o BatchMode=yes $B "~/miniconda3/envs/rsna/bin/python ~/kt_test/verify_outputs.py ~/fleet_runs --receipt ~/fleet_runs/kaggle_s1044_${t}_run_receipt.json | tail -n 4" < /dev/null
done
ssh -o BatchMode=yes $B 'for F in 0 1 2 3 4; do echo "f$F: $(tail -n 1 ~/fleet_runs/f${F}_s1044_ttr_mix50.log 2>/dev/null | cut -c1-80)"; done; ls -l ~/fleet_runs/kaggle_s1044_*_run_receipt.json' < /dev/null
```

Every copied fold must show `OK` lines (the `epochs/*.pt` entries stay ABSENT, by design) and a closing
`{"raw": ..., "ema": ...}` log line. Only the two `*_best.pt`, the two OOFs and the log per fold are copied (the picker
needs nothing else). Set C needs no copy (it is a box run).

### 2.3 Collect, build, box A0 (Git Bash)

```bash
# collect                  gate: last line "K7COLLECT OK <N> member(s): A 5/5, B <b>/5, C <0|1>"
K5_MIN_EPOCHS=16 bash "$SP/k7/k7_collect.sh"      # 14 if the f4 notebook ran E 14-15 (sec 2.1); K7_WITH_C=0 to drop C
#   exit 1 = STOP (the FAIL line names the check); "NOT FINISHED set X f.._s.._...: <reason>" lists every run left out;
#   "set C: INCLUDED / EXCLUDED: <why> / NOT FINAL". Set C is never a reason to stop: a failed C check excludes C.
# build (~30 s)             gate: "K7 BUILD OK FINAL (...)" or "INTERIM", "expect_tta: ..." (every member =AB)
$PY -W ignore "$SP/k7/build_k7.py"
BD="$SP/k7/build"                                   # interim: "$SP/k7/build_i<N>"
MEM=$($PY -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['members']))" "$BD/K5_BUILD.json" | tr -d '[:cntrl:]')
# box A0 (CPU, 2 fold-0 studies, ~30-40 min with 11 members)
bash "$SP/k7/k7_stage_box.sh" "$BD"                 # gate: "DS IDENTICAL (<11+N> files)", "K7STAGE OK"
ssh -o BatchMode=yes $B 'bash ~/k7_test/k5/box/k5_box_launch.sh all' < /dev/null
ssh -o BatchMode=yes $B 'cd ~/k7_test/runs; for n in A0_main A0_solo; do f=$(ls -t ${n}_*.log | head -1); echo "== $f"; grep -E "^FAIL|BOXSTAGE|TTA modes|members:" $f | cut -c1-220; done' < /dev/null
#   PASS = both "BOXSTAGE PASS k/k", no FAIL line; the members line shows N x [KneeFleetNetTTR] incl. f0_s1044t288_... if C is in.
```

### 2.4 Dataset version, push, gates, submit

```powershell
# dataset version (PowerShell). Windows kaggle CLI staging-dir bug: pre-create the staging folder first.
$UP = "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k7\build\upload\rsna-knee-cnxt-fleet"   # interim: ...\k7\build_i<N>\upload\...
New-Item -ItemType Directory -Force -Path (Join-Path "$env:TEMP\.kaggle\uploads" ($UP -replace ':', '_')) | Out-Null
(Get-ChildItem $UP).Count                                   # 11 + N (11 members: 22)
kaggle datasets version -p $UP -m "K7: TTR mix50 s1043 x5 + s1044 x<b> (+ TTR288 f0) A+B, K5 worker"
kaggle datasets status eeshsaxena/rsna-knee-cnxt-fleet      # repeat until: ready (1.2 GB: allow longer than K5's 548 MB)
kaggle datasets files  eeshsaxena/rsna-knee-cnxt-fleet      # 11 + N files, every name of K5_BUILD.json members (incl. f0_s1044t288_...)
kaggle kernels push -p "$env:TEMP\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k7\build\kernel\rsna-sub-k7-fleet"
kaggle kernels push -p "$env:TEMP\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k7\build\kernel\rsna-sub-k7-solo"
kaggle kernels status eeshsaxena/rsna-sub-k7-fleet          # until complete (same for -solo); write the version V down
kaggle kernels output eeshsaxena/rsna-sub-k7-fleet -p "$env:TEMP\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k7\build\out_main_v<V>"
kaggle kernels output eeshsaxena/rsna-sub-k7-solo  -p "$env:TEMP\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k7\build\out_solo_v<V>"
```

```bash
# gates (Git Bash): each must end "K2CHECK PASS"
MAN="$BD/upload/rsna-knee-cnxt-fleet/fleet_manifest.json"
$PY "$SP/k7/k2_check.py" --work "$BD/out_main_v<V>" --expect_fleet clean --kaggle_commit --expect_mode strict --expect_w_main 0.5 --expect_members "$MEM" --expect_manifest "$MAN" --expect_tta "$BD/K5_BUILD.json"
$PY "$SP/k7/k2_check.py" --work "$BD/out_solo_v<V>" --expect_fleet clean --kaggle_commit --expect_mode solo                     --expect_members "$MEM" --expect_manifest "$MAN" --expect_tta "$BD/K5_BUILD.json"
```

The commit log must show `[fleet g] members: ... [KneeFleetNetTTR]` once per member (incl. `f0_s1044t288_...` when C is
in), `TTA modes ...=AB` for every member, and the microbench line: K7 adds ~1.65x for the TTR288 member (sec 4).

```powershell
# submit (only versions that passed the gate; UTC day ends 05:30 IST). Main first; solo is the optional read.
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k7-fleet -v <V> -f submission.csv -m "K7 strict P+F W.50: TTR mix50 s1043 x5 + s1044 x<b> (+ TTR288 f0) A+B (<N> members)"
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k7-solo  -v <V> -f submission.csv -m "K7 SOLO: <N> TTR mix50 members A+B"
```

Timeline (MODEL): last notebook complete -> fetch ~5-10 min per notebook -> copy + verify ~5 min -> collect ~6 min
(downloads the new checkpoints, ~108 MB each) -> build ~1 min -> A0 ~30-40 min (CPU, 11 members) -> dataset version +
ready ~15-25 min (1.2 GB) -> commits ~25-40 min -> gates ~5 min: **~1.5-2 h**. To hit ~1 h, run the dataset version and
the pushes WHILE the A0 runs and gate the submit on A0 PASS + K2CHECK PASS (the A0 then only blocks the submit).

## 3. The set B / set C conflict and its fix (K7-KEY)

Set B fold 0 and set C fold 0 are both fold 0, seed 1044. The trainer names files `f{F}_s{S}_{tag}_{recipe}_cv{NNNN}_best.pt`
(B: `f0_s1044_raw_ttr224-320x24_...`, C: `f0_s1044_raw_ttr288-320x24_...`). Where the collision bites:

| where | old rule | effect on K7 | fix |
|---|---|---|---|
| **worker** `code/fleet_worker.py` lines 109-110: `fs_key = ['_'.join(basename.split('_')[:2]) ...]`; `assert len(fs_key) == len(set(fs_key)), 'duplicate fold/seed members (raw AND ema?)'` | key = `f{F}_s{S}` | B f0 and C f0 both `f0_s1044`: worker exit 1, fleet stage not clean | **worker NOT changed** (stays the K5 worker that scored .946). build_k7 re-keys every non-224 member with the existing P-TAG mechanism: `f0_s1044` -> `f0_s1044t288` (file name only, bytes identical). Keys become `f0_s1044` / `f0_s1044t288`; fold grouping (`f0`) unchanged |
| `build_k2.py` preflight | run = (fold, seed, train_labels, backbone, model_mod); same fold+seed needs an explicit re-key | B f0 and C f0 = "raw AND ema of one run" | run identity + args ttr_size for TTR members (all older kits: every TTR member 224, so the same accept / refuse outcome) |
| `build_k6.py` | members keyed (seed, fold) | duplicate | build_k7 keys (seed, fold, ttr_size) |
| picker `--copy` | removes every `DEST/ckpt/f{F}_s{S}_*_best.pt` other than its pick | picking C deletes B's f0 checkpoint from the shared dest | remove only same-recipe files; non-224 OOF copies get a `t{N}` key |
| picker receipts | `--kaggle_receipt` single + default f34 receipt | a seed-1044 fold 3/4 would be checked against the seed-1043 f34 receipt (sha mismatch FAIL); a s1044 fold-0 receipt could vouch for the TTR288 box run | receipt used only if SEED == `--seed` and `runs[F]` == the run folder |

A true duplicate is still refused at every layer: the picker returns one pick per run; k7_merge refuses duplicate
(seed, fold, ttr_size) or file; build_k7 refuses duplicate members / keys; build_k2 refuses the same run twice even when
re-keyed (run identity); pack_fleet_ds P-RUN refuses identical ckpt args even when re-keyed; the worker refuses a
repeated key. Model math, model_sha (`f141aeed136dfc01`), the worker and the cell are unchanged.

Serving at 288: the worker builds `KneeFleetNetTTR(ttr_size=a['ttr_size'])` from the checkpoint args and asserts
`int(m.ttr_size_t) == int(a['ttr_size']) == m.ttr_size` after loading (W-TTR / TTR-6); the TTR288 checkpoint carries
args 288 and buffer 288 (MEASURED, sec 6).

## 4. Runtime (hidden rerun, 2 x T4, 8.6 h wall; MODEL on MEASURED inputs)

Inputs (MEASURED): K5 / K6 commit microbench, TTR224 A+B **285-332 ms/member/study** on T4 (slower T4 ~325-332); decode
1.30-1.62 s/study/thread; parent ~6 h; ~661 studies per T4. TTR288 cost: x1.65 by pixel count (given); the A0 CPU
microbench measured the real TTR288 member at 38.0 s vs 20.4-29.2 s (median 21.1) for the TTR224 members, **x1.8** on a
loaded CPU. Below a TTR288 member costs x1.8 -> ~600 ms A+B on T4 (conservative).

| per T4, ~661 studies | K5 (5 x 224) | **K7 (10 x 224 + 1 x 288)** | 15 members (10 x 224 + 5 x 288) |
|---|---|---|---|
| ms/study, all members A+B | 5 x 335 = 1.7 s | 10 x 335 + 600 = **3.95 s** | 10 x 335 + 5 x 600 = 6.35 s |
| GPU minutes | 18.5 | **43.5** | 70 |
| stage serial upper bound (GPU + decode 15-32 min + pass B build ~1 min) | 35-52 | **~60 central, ~77 slow** | ~86 central, ~103 slow |
| kernel end at a 6.0 h parent | 6.6-6.9 h | **~7.0 h, slow ~7.3 h** | ~7.45 h, slow ~7.7 h |
| margin to the 8.6 h wall | >= 1.7 h | **>= 1.3 h** | >= 0.9 h |

- **11 members fit comfortably; no cut needed.** The parent could run to ~7.2 h before an 11-member stage (slow case)
  reaches the wall; the stage starts only with >= 20 min left and strict / solo never write a silent parent score.
- GPU memory: all members are resident (fp32 weights ~108 MB each, ~1.2 GB for 11) + one study's activations; fine on 15 GB.
- If a later K-kit grows past ~13 members (e.g. a 5-fold TTR288 set), the cut in order: (1) drop the weaker 2nd-seed
  folds of the 224 set (MEASURED: more seeds of one recipe add only ~+.0005-.001), (2) serve TTR288 members pass A only
  (halves their cost; but TTA was +.002 on J1), (3) never shorten the wall or the 20-min start floor.

## 5. Reads (GUESS / MODEL)

K7 vs K5 w50 (.946): set B is the same recipe with another seed (MEASURED lesson: +.0005-.001 solo), set C is a
higher-resolution member of the same family at 1/11 weight. Expected main W .50 ~.946-.947 (MODEL), solo ~.943-.945.
Display rounding is +-.0005, so only a .947+ read is a clear gain.

## 6. Evidence (2026-10-06 21:45-23:00 IST, MEASURED; transcripts in `box/transcripts_k7/`)

**Simulation inputs** (`_dev/k7_sim_setup.py`, box `~/k7_sim/root`, ours): set B sim = the 5 set A runs (both tags),
ckpt args seed 1043 -> 1044 (weights untouched), files renamed, 3 simulated receipts
`kaggle_s1044_{f01,f23,f4}_run_receipt.json` (cfg SEED 1044, runs, per-file sha256); set C sim = cp -p of the LIVE
TTR288 run's best ckpts at epoch 15 (raw cv .889075, ema .883306; args ttr_size 288, buffer 288, seed 1044, fold 0) +
its log with a SIMULATED closing line. Nothing under `~/fleet_runs` was written.

1. **Collect** (`K7T=k7_simtest K7_ROOT_B=~/k7_sim/root K7_ROOT_C=~/k7_sim/root K7_LAPTOP_DIR=k7/_sim bash k7_collect.sh`):
   set A 5 PICK (identical to K5), set B 5 PICK (all `source kaggle`, receipts f01 / f01 / f23 / f23 / f4, OOF rescore
   diff 0.0), set C 1 PICK (`ttr_size_t 288`, OOF copy `oof_f0_s1044t288_raw_cv8890.csv`), `set C: INCLUDED`,
   **K7COLLECT OK 11 member(s): A 5/5, B 5/5, C 1 (FINAL A+B)** (`collect_sim2.log`; the first try died on a LAN
   reset mid-scp; the re-run moved the partial file aside and re-fetched it).
2. **Build** (`build_k7.py --members_json _sim/members.json ... --out k7/build_sim`): **K7 BUILD OK FINAL (11 members)**,
   upload 1193.7 MB / 22 files, worker sha256[:16] `21effe04f052c18d` (== K5), pack log
   `member f0_s1044t288_raw_ttr288-320x24_cv8890_best.pt ... ttr_size 288`, every member `AB` (`build_sim.log`).
3. **Box A0** (CPU, `K7T=k7_simtest`, 2 fold-0 studies shared by A f0 / B f0 / C f0): main and solo each
   **BOXSTAGE PASS 40/40**, `RESULT OK fleet stage clean: 11 member(s)`, 11 x `[KneeFleetNetTTR]`, TTA forwards A2 B2
   for all 11, K5c npz identity exact, fleet solo == raw mean of the 11 member rows (1.1e-16). Per-member pass A vs its
   trainer OOF: A f0 3.07e-4, B f0 3.07e-4, **C f0 (TTR288) 4.11e-4** (vs the 288 trainer's own OOF, so the worker built
   it at 288 and the buffer assert passed). Fleet stage 39.4 min on the loaded box CPU (load 27-41 from other jobs; the
   trainer kept 0.45 s/it). `k2_check.py --expect_tta K5_BUILD.json` on both outputs: **K2CHECK PASS 38/38 (main),
   37/37 (solo)**.
4. **Picker / merge tests** (`box/t_k7_pick.py`): **TK7PICK PASS 11/11** (T1 the K6 command line picks exactly the
   K6 set A members and OOF names; T2 B via 3 receipts; T3 no receipt + `--require_receipt` FAIL; T4 the seed-1043 f34
   receipt never vouches for s1044 folds 3-4; T5 a s1044 fold-0 Kaggle receipt never vouches for the TTR288 box run;
   T6 C with ttr_size 224 FAIL; T7 B and C fold-0 picks coexist in one dest, stale same-recipe file removed; T8 merge:
   11 with C, C below the bar or `--with_c 0` excluded, **a true duplicate (B f0 offered again) K7MERGE FAIL**).
5. **Conflict tests** (`tests/t_k7_conflict.py`, laptop CPU): **TK7CONFLICT PASS 15/15**. Two tiny REAL TTR members with
   the same weights, fold 0 seed 1044, ttr 48 and 56 (the 288 analogue): the shipped worker (== K5 worker) serves both
   after the re-key (rows finite, rows differ), **each member's P / PA / PB == its rows when served alone (max|diff|
   0.0)**; refused: pack without re-key, pack of the same run twice even when re-keyed (P-RUN), worker on a manifest
   naming B0 twice, worker on B0 + C0 under the same `f0_s1044` key, worker on C0 with args ttr_size 48 vs buffer 56.
   On the REAL checkpoints: build_k2 preflight accepts B f0 + C f0 re-keyed, refuses them un-re-keyed and the same B f0
   twice; build_k7 refuses a duplicate member; only C is re-keyed; set A alone keeps its keys and names.
6. **Old behaviour byte-identical**: k7 `build_k2.py` on the K5 members with the K5 arguments reproduces the K5 (.946)
   build exactly: upload folder 16/16 files sha256-identical to `k5/build/upload`, notebooks `rsna_sub_k5_fleet`
   `816a4d6be5881a4f` and `rsna_sub_k5_w50` `dc09b2e196110070` == the K5 ones, kernel metadata identical
   (`regress_k5.log`, `k7/_regress_k5`).
7. **Laptop smoke** (`tests/smoke_k2.py` on the k7 kit): **K5 SMOKE PASS in 12.2 min, 44/44 scenarios** + every A / B /
   B5 / C2 / D / D5 check incl. K2e (`smoke_run_k7.log`). A first run stopped at staging because the k6 copy lacked the
   two crop cell files the smoke's combined notebooks need; copied from k5 (byte-identical), then PASS.

## 7. Gaps / caveats

1. **Set B is simulated** (set A weights relabelled seed 1044). The real teammate outputs have not been seen; their
   notebook code is the f34 kit with SEED 1044 (`RUN = f{F}_s1044_ttr_mix50`, receipt cfg SEED + runs), and the picker's
   receipt match was written against that format.
2. **f4 will most likely need `K5_MIN_EPOCHS=14`** (E ~14-15 by design of its 7 h budget). Operator's call (sec 2.1);
   without it the collect FAILS on f4 and stops (by design). Until f4 lands, an interim build of B folds 0-3 needs no flag.
3. **Set C's real run is unfinished** (epoch ~17 of 24 at 22:45). The sim used its epoch-15 best (.8891). The real pick
   reads the closing line (~00:00-00:45); the .8876 bar decides. If the box gate passes, `auto_after_ttr288.sh` trains
   TTR288 folds 1-4 (`f{1..4}_s1044_ttr288mix50`); `K7_C_FOLDS=0,1,..` can add them later, but .8876 is a fold-0 bar
   (fold 3 is a harder split), so set `K7_C_BAR` deliberately; 15 members still fit (sec 4) with ~0.9 h margin.
4. **T4 GPU path of a TTR288 member is untested until the commit** (CPU path parity-checked vs its trainer OOF, 4.1e-4).
   The commit's `--kaggle_commit` gate + its microbench line cover it.
5. Dataset version is **1.19 GB** (K5: 548 MB): upload / `ready` takes longer; budget it inside the 1 h target.
6. Box A0 with 11 members took **39 min** (about half is the commit-style microbench, 5 rounds x 22 forwards on CPU).
   To ship in ~1 h, run the dataset version + pushes while the A0 runs; submit only after A0 PASS + K2CHECK PASS.
7. Scratch left (ours, removable; not deleted here): box `~/k7_sim` (2.7 GB: sim root + picker test dests) and
   `~/k7_simtest` (2.4 GB: sim A0 area); laptop `k7/_sim` (11 ckpts), `k7/build_sim` (TEST build, **never version it**),
   `k7/_regress_k5`, `k7/tests/_smoke`, `k7/tests/_k7c`. The real flow uses `~/k7_test` and `k7/build`, both fresh.
