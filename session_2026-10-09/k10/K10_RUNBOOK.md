# K10 RUNBOOK: set E (TTR288 frac75 s1045) kits: K10 / K10-solo / K10N-a / FN10 / FN10-3

Written 2026-10-09 03:00-05:00 IST by Claude (subagent), per `Downloads/rsna-knee/RESEARCH_TO_TOP_10-08.md` (L2, L3, sec 3.5,
sec 5, sec 6: "a new set must REPLACE, not add" with the parent) and `PREREG_10-10.md`. Nothing was created, versioned,
pushed or submitted on Kaggle (no Kaggle call at all). Box work: CPU only (CUDA hidden, nice 19, ionice idle, taskset to 6
CPUs 34-39 in total), writes only `~/k10_test`. No em dashes. Tags: MEASURED, MODEL, GUESS.

`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`,
`PY` = `C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe`, `B` = `training@192.168.10.30`
(`ssh -o BatchMode=yes`, append `< /dev/null`). The kit is `SP/k10` (copies of `SP/k9`, `SP/k9n`, `SP/kfn`; those kits are
untouched). Everything below the line "SIMULATION" was tested on simulated E members only (sec 5).

## 0. What K10 is

| set | runs | trained on | ttr | labels | E | shipped key |
|---|---|---|---|---|---|---|
| E (new) | `f{0..4}_s1045_ttr288frac75` | f0 box V100 (ends ~07:05 10-09); f1-f4 Vast 4 x RTX 3090 (fetched into the EMPTY box placeholders ~06:50 by `vast320/vast_fetch.sh`, receipts `~/fleet_runs/vast320_f{F}_s1045_ttr288frac75_receipt.json`) | 288 | frac75 | 21 | `f{F}_s1045t288f75` (new; asserted absent from every older manifest) |
| D | `f{0..4}_s1043_ttr_frac75` | Kaggle (K9 picks, byte-identical, asserted) | 224 | frac75 | 17-22 | `f{F}_s1043f75` (== K9) |
| C | `f{0..4}_s1044_ttr288mix50` | box (K9 picks, byte-identical) | 288 | mix50 | 24 | `f{F}_s1044t288` (== K9), FN10-3 only |

| kernel | slug | dataset (NEW slug) | cells | submission.csv | runtime MODEL (central / slow) |
|---|---|---|---|---|---|
| K10 main | `eeshsaxena/rsna-sub-k10-fleet` | `eeshsaxena/rsna-knee-cnxt-fleet-k10` (E + D, 10 members, 21 files, 1086 MB) | parent 0-30 + K10-DS fleet cell 31 | rank_pct(.50 r(P) + .50 r(F)), STRICT (w40 / w60 variants) | 6.97 / 7.27 h (== K9) |
| K10 SOLO | `eeshsaxena/rsna-sub-k10-solo` | same | same, MAIN_MODE solo | F alone | == K9 SOLO |
| K10N-a | `eeshsaxena/rsna-sub-k10n-a` | same + `nartaa/rsna-knee-publication-swa-weights-20261007` | K10 main + cell 32 (the K9N N stage, byte-identical) | rank_pct(.10 r(P) + .30 r(F) + .60 r(N)) (K9N-a weights) | 7.40 / 7.95 h (== K9N-a) |
| FN10 | `eeshsaxena/rsna-sub-k10fn` | `-k10` + N weights | md / FN-IDS / fleet (E + D) / N (== kfn FN cells) | rank_pct(.40 r(F) + .60 r(N)), no parent | 1.45 / 2.00 h (== FN) |
| FN10-3 | `eeshsaxena/rsna-sub-k10fn3` | `eeshsaxena/rsna-knee-cnxt-fleet-k10w` (C + D + E, 15 members, 26 files, ~1.6 GB) + N weights | same, 15-member fleet | rank_pct(.40 r(F) + .60 r(N)) | 1.88 / 2.54 h |

- **Inference cap** (research sec 3.1 / 6): every parent kernel (K10, K10 SOLO, K10N-a) runs exactly 5 x TTR224 (D) +
  5 x TTR288 (E): E REPLACES C, so the composition and the runtime equal K9 / K9N-a (asserted by the build and
  `tests/t_k10_build.py` T7). Three sets exist ONLY parent-free (FN10-3): a parent + 3 sets would end ~8.49 h slow
  (MODEL, `tests/runtime_k10.py`), so `build_k10.py --kit k10w` writes the 15-member upload folder and NO kernel.
- **K10-DS (new dataset slugs).** The K9 version of `eeshsaxena/rsna-knee-cnxt-fleet` is never versioned or attached by
  any K10 kernel. The fleet cell hard-codes its mount path, so cell 31 = the K5 cell (`e37f0604997cd40f`) with its three
  `rsna-knee-cnxt-fleet` strings replaced by the kit slug (`cells/k10_fleet_cell.py` 74dc5bde..., `cells/k10w_fleet_cell.py`
  6e9de1bd...; `make_k10_tools.py`, exact counted substitutions, reverse substitution == the K5 cell). Everything else is
  the K9 build path byte for byte (build_k2 / make_k2_kernel / k2_check / worker / pack tool asserted == k9 / k5 / k7 / k3).
  Negative test: a K10 notebook with the dataset mounted under the K9 slug fails loud ("fleet dataset not attached").
- **Gates.** K10 main / SOLO: `k2_check.py` (unchanged). K10N-a: `k10n_check.py` (= k9n_check with the kernel key / TAG
  K10N-a; generated). FN10 / FN10-3: `fn10_check.py` (= fn_check unchanged, slug taken from the FN_BUILD.json).

## 1. Set E acceptance (pre-registered 10-09 ~03:40, before any E checkpoint was picked)

- **Picker** (`box/k5_pick.py`, K10 copy): every K9 check (closing line, both best files, cv == closing line, OOF rescored
  == ckpt cv to 1e-12, model_sha f141aeed136dfc01, ttr_size buffer == args == 288, fold / seed 1045, train_labels frac75,
  selection labels v4, amp fp16, decode probe npz, epochs header == ckpt args) + **K10-VREC**: folds 1-4 must carry their
  vast320 receipt (run == source_run == the folder, mode canonical, verify "OK" = md5 of every copied file == instance,
  best_raw 1, best_ema 1, top_oof 2, closing_line == the log's last line, the picked ckpt listed "args OK"). With a valid
  receipt the OOF-vs-ckpt mtime rule is reported, not enforced (a copied run, as Kaggle folds). Fold 0 (box) keeps the
  mtime rule. Override only if a fold was retrained elsewhere: `K10_E_VAST_FOLDS="1,3"` etc.
- **Min epochs** 16 for E (like K9 set C; every E fold is planned at 21), D keeps 14 (K9), C 16.
- **Sanity bar** E_F >= C_F - .004 (C_F = set C's picked raw cv, same fold): f0 .886920, f1 .891223, f2 .889559,
  f3 .878181, f4 .891564. Justification (MEASURED components, box closing lines): fold-matched seed swaps at the same
  recipe +.0028 / -.0009 / +.0003 / +.0003 (B s1044 - A s1043, TTR224 mix50 f0-f3) and +.0028 (s1044 - s1043 TTR224 frac75
  f0): SD ~.0016-.0020; label swap frac75 - mix50 same seed (D - A) +.0002 / -.0017 / +.0002 / +.0006 / +.0001 (worst
  -.0017); 21 vs 24 epochs no measurable loss (B 21 ep vs A 24 ep, mean +.0006). So a healthy E fold sits at C_F - .0003
  +- ~.0022 (MODEL) and .004 is ~1.7 SD below it; a broken fold (historically -.01 or worse) fails. D bars = A_F - .004,
  C bars = A_F - .002 (K9, unchanged).
- **KNOWN RISK:** C f0 / f3 are seed-lucky (C - A +.0073 / +.0059 vs +.0026-.0028 elsewhere), and the 320 frac75 s1045
  21-ep probe (same seed / labels / epochs) read .8877 on f0 (= C f0 - .0032). A healthy E f0 can fall under .886920
  (P ~.2-.35, MODEL: if 288 reads like the 320 probe, E f0 sits only ~.0008 above its bar). **Pre-registered override** (k10_merge prints it per failing E fold): accept with
  `K10_E_BARS="F:<cv>"` ONLY IF cv >= C_F - .007 AND last-3-epoch raw val mean >= best - .003 AND all 21 epochs logged.
  Otherwise the fold does not ship (sec 6).

## 2. Final steps (main session, from ~06:50; times MODEL)

### 2.0 Before anything

The simulation A0 queues finished at 06:17 (`QUEUE q1 DONE` 05:44, `QUEUE q2 DONE` 06:17): nothing of this kit runs on
the box. C and D checkpoints are already copied to `~/k10_test/ckpt` (03:40 dry run); the box tools, including
`k10_gold_read.py` and `k10_gold_bg.sh`, are in `~/k10_test/k10/box/`.

```bash
B=training@192.168.10.30
ssh -o BatchMode=yes $B 'tail -n 1 ~/k10_test/a0sim/runs/q1_*.log ~/k10_test/a0sim/runs/q2_*.log; ls ~/k10_test/ckpt | wc -l' < /dev/null   # QUEUE ... DONE x2, 10
```

### 2.1 Verify the 5 E folds (closing lines + Vast receipts) (~06:50-07:10)

```bash
ssh -o BatchMode=yes $B 'for F in 0 1 2 3 4; do r=f${F}_s1045_ttr288frac75; echo "f$F: $(tail -n 1 ~/fleet_runs/$r.log 2>/dev/null | cut -c1-90) | files $(ls ~/fleet_runs/$r 2>/dev/null | wc -l) | receipt $(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(r[\"mode\"], r[\"verify\"])" ~/fleet_runs/vast320_${r}_receipt.json 2>/dev/null || echo none)"; done' < /dev/null
#   every fold: last line {"raw": ..., "ema": ...}; f1-f4: "receipt canonical OK"; f0: "receipt none" (box run).
#   A fold without its closing line = not finished: wait (Vast watcher / box queue). Never touch ~/fleet_runs.
```

### 2.2 Gold predictions for E (start as soon as folds land; ~12-20 min per TTR288 fold on 6 CPUs, MODEL from the MEASURED 61 min for one fold at 1 shared thread)

```bash
ssh -o BatchMode=yes $B 'bash ~/k10_test/k10/box/k10_gold_bg.sh infer' < /dev/null      # detached inside the script
ssh -o BatchMode=yes $B 'tail -n 3 $(ls -t ~/k10_test/gold/infer_*.log | head -1)' < /dev/null
#   infers every FINISHED E fold missing from ~/k10_test/gold/preds (+ a .sha256 sidecar per prediction); C / D / A
#   predictions are reused read-only from ~/novel/*. Last line "K10GOLD PARTIAL ... INFO ONLY" until E 5/5.
#   Run it again after f0 closes (~07:05). It uses CPUs 34-39 (NTHR 6): do not overlap it with the A0 queues of 2.5
#   unless the CPU budget allows (both pin to 34-39, so together they still use at most 6 CPUs, only slower).
#   (k10_gold_bg.sh is synced by k10_collect.sh; it is already on the box since the 03:40 dry run.)
```

### 2.3 Collect (~2-3 min; C / D are already staged in ~/k10_test/ckpt from the 03:40 dry run)

```bash
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
bash "$SP/k10/k10_collect.sh" 2>&1 | tee "$SP/k10/logs/collect_final_$(date +%H%M).log" | tail -n 25
#   gate (last line): "K10COLLECT OK 15 member(s): E 5/5, D 5/5, C 5/5  (FINAL)"   (no "[SIMULATION ...]" tail)
#   exit 3 = a set not 5/5 (wait). exit 1 = STOP: read the FAIL / K10MERGE lines. An E fold under its bar prints
#   "PRE-REGISTERED OVERRIDE CHECK E fF ... ELIGIBLE | NOT ELIGIBLE" (sec 1): only ELIGIBLE folds may be accepted:
#   K10_E_BARS="F:<cv>" bash "$SP/k10/k10_collect.sh"   (write the override into HANDOFF)
#   Every E f1-f4 line must say "Vast-filled run (.../vast320_..._receipt.json: verify OK ...)".
```

### 2.4 Pre-push gold read (decision; ~2 min once 2.2 has all 5 E folds)

```bash
ssh -o BatchMode=yes $B 'bash ~/k10_test/k10/box/k10_gold_bg.sh final' < /dev/null      # after 2.3 (needs ~/k10_test/members.json)
ssh -o BatchMode=yes $B 'tail -n 9 $(ls -t ~/k10_test/gold/final_*.log | head -1)' < /dev/null
#   last line "K10GOLD FINAL | K10 TIE RULE ON|OFF: gold58 pass A K10 - K9 ... ; Fracture ... | E - C ... |
#   COMPOSITION: K10W - max(K9, K10) ... -> FN10-3 PREFERRED | FN10 | K9 FN"  -> write it into HANDOFF.
#   Pre-registered in the script header (sha256 176f675f...eef0): ON when d58(K10 - K9) > -.003 AND Fracture >= 0;
#   FN10-3 preferred when K10W - max(K9, K10) > -.003. It never blocks a push.
```

### 2.5 Build (~3 min) + laptop tests + box A0 (CPU 6 cores, two queues, ~2.5-3 h: the simulation took 2.8 h)

```bash
cd "$SP/k10"
$PY -B -W ignore build_k10.py --kit k10  2>&1 | tail -n 3     # gate: "K10 BUILD OK [k10] FINAL (10 members: E 5/5 + D 5/5: [...])"
$PY -B -W ignore build_k10.py --kit k10w 2>&1 | tail -n 3     # gate: "K10 BUILD OK [k10w] FINAL (15 members: C 5/5 + D 5/5 + E 5/5 ...)"
$PY -B build_k10n.py 2>&1 | tail -n 2                         # gate: "K10N BUILD OK FINAL"   (out k10/build_n)
$PY -B build_k10fn.py --kit fn10   2>&1 | tail -n 2           # gate: "K10FN BUILD OK FINAL [fn10]"   (out k10/build_fn)
$PY -B build_k10fn.py --kit fn10_3 2>&1 | tail -n 2           # gate: "K10FN BUILD OK FINAL [fn10_3]" (out k10/build_fn3)
$PY -B -W ignore tests/t_k10_build.py --root "$SP/k10" --real | tail -n 1     # "TK10BUILD PASS n/n"
ls "$SP"/k10/build*/DO_NOT_* 2>/dev/null                      # must print nothing
bash "$SP/k10/k10_stage_box.sh" "$SP/k10" k10_test/a0 k10_test     # gates: "DS IDENTICAL rsna-knee-cnxt-fleet-k10 (21 files)",
                                                                   # "... -k10w (26 files)", "K10STAGE OK"
ssh -o BatchMode=yes $B 'A=k10_test/a0 bash ~/k10_test/a0/k10/box/k10_box_launch.sh q1 q2' < /dev/null
# results (each must end PASS):
ssh -o BatchMode=yes $B 'cd ~/k10_test/a0/runs; cat q1_*.log q2_*.log; for f in $(ls -t *_[0-9][0-9][0-9][0-9].log | grep -v "^q[12]_"); do echo "== $f $(grep -hE "^(BOXSTAGE|K9NBOX|KFNBOX) " $f | cut -c1-60)"; done' < /dev/null
```
- PASS = `neg_k9slug` BOXSTAGE PASS (fleet stage NOT clean "not attached", submission.csv absent); `main` / `solo`
  BOXSTAGE PASS with `RESULT OK fleet stage clean: 10 member(s)` (10 x KneeFleetNetTTR incl. 5 x `f{F}_s1045t288f75_...`,
  all AB, decode probe 40/40, per-member OOF parity D f0 and E f0 < 2e-2); `n_chain` K9NBOX PASS (k10n_check part A =
  every K9 gate via k2_check + part B); `n_neg_*` K9NBOX PASS (N NOT clean, submission.csv absent); `fn_chain` /
  `fn3_chain` KFNBOX PASS (15 members for fn3); `fn*_neg_*` KFNBOX PASS; `n_stub` / `fn_stub` PASS.
- If a job FAILS: read its log (`~/k10_test/a0/runs/<job>_<HHMM>.log`, `runs/<job>_<HHMM>/report.json`); do not push
  that kernel. `touch ~/k10_test/a0/STOP` ends the queues after the current job.

### 2.6 Dataset CREATE (new slugs; Eesh runs every kaggle command; any time after 2.5 PASS; no GPU quota)

The K9 dataset `eeshsaxena/rsna-knee-cnxt-fleet` must NOT be versioned until every Saturday 10-10 commit and submission
that reads it (K9N-a, FN, K9 SOLO, K9 main, N-only) has scored. K10 never touches it: these are two NEW datasets.

```powershell
$K10 = "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k10"
foreach ($d in "build", "build_w", "build_n", "build_fn", "build_fn3") { Get-ChildItem "$K10\$d" -Filter "DO_NOT_*" }   # must list nothing
$UP  = "$K10\build\upload\rsna-knee-cnxt-fleet-k10"
$UPW = "$K10\build_w\upload\rsna-knee-cnxt-fleet-k10w"
foreach ($u in $UP, $UPW) { New-Item -ItemType Directory -Force -Path (Join-Path "$env:TEMP\.kaggle\uploads" ($u -replace ':', '_')) | Out-Null }   # Windows CLI staging bug
(Get-ChildItem $UP).Count; (Get-ChildItem $UPW).Count        # 21, 26
Get-Content "$UP\dataset-metadata.json"                       # id eeshsaxena/rsna-knee-cnxt-fleet-k10
kaggle datasets create -p $UP                                 # NEW dataset (private by default; never --public)
kaggle datasets create -p $UPW                                # NEW dataset (FN10-3 only)
kaggle datasets status eeshsaxena/rsna-knee-cnxt-fleet-k10    # until: ready (~1.09 GB)
kaggle datasets status eeshsaxena/rsna-knee-cnxt-fleet-k10w   # until: ready (~1.6 GB)
kaggle datasets files  eeshsaxena/rsna-knee-cnxt-fleet-k10    # 21 files incl. f{0..4}_s1045t288f75_raw_ttr288-... + f{0..4}_s1043f75_raw_ttr224-...
kaggle datasets files  eeshsaxena/rsna-knee-cnxt-fleet-k10w   # 26 files (+ f{0..4}_s1044t288_...)
kaggle datasets files  eeshsaxena/rsna-knee-cnxt-fleet        # unchanged K9 version: 21 files, s1043f75 + s1044t288 (NOT versioned)
```
- If `create` says the slug exists (it should not): STOP, do not `version` blindly; compare `kaggle datasets files` with
  the local folder first.

### 2.7 10-11 (or later): push, gate, submit (after the 10-10 05:30 quota reset and the Saturday commits)

Order by value (research sec 3.6): FN10 (or FN10-3 if the gold read says FN10-3 PREFERRED), K10N-a, then K10 main / K10
SOLO only if the clean pick needs them. Quota (MODEL): ~0.1-0.25 h per parent-free commit, ~0.5 h per parent commit.

```powershell
$K10 = "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k10"
kaggle quota
kaggle kernels push -p "$K10\build_fn\kernel\rsna-sub-k10fn"
kaggle kernels push -p "$K10\build_fn3\kernel\rsna-sub-k10fn3"
kaggle kernels push -p "$K10\build_n\kernel\rsna-sub-k10n-a"
kaggle kernels push -p "$K10\build\kernel\rsna-sub-k10-fleet"
kaggle kernels push -p "$K10\build\kernel\rsna-sub-k10-solo"
kaggle kernels status eeshsaxena/rsna-sub-k10fn                # until complete; write each version V down
kaggle kernels output eeshsaxena/rsna-sub-k10fn   -p "$K10\build_fn\out_v<V>"  --page-size 200
kaggle kernels output eeshsaxena/rsna-sub-k10fn3  -p "$K10\build_fn3\out_v<V>" --page-size 200
kaggle kernels output eeshsaxena/rsna-sub-k10n-a  -p "$K10\build_n\out_v<V>"   --page-size 200
kaggle kernels output eeshsaxena/rsna-sub-k10-fleet -p "$K10\build\out_main_v<V>" --page-size 200
kaggle kernels output eeshsaxena/rsna-sub-k10-solo  -p "$K10\build\out_solo_v<V>" --page-size 200
```
- Never `--timeout` on a submission kernel. Instant ERROR with an empty log = quota refusal: push again later.

Gates (Git Bash; each must end PASS):
```bash
K10=$SP/k10; BD=$K10/build; BW=$K10/build_w
MEM=$($PY -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['members']))" "$BD/K5_BUILD.json" | tr -d '[:cntrl:]')
MEMW=$($PY -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['members']))" "$BW/K5_BUILD.json" | tr -d '[:cntrl:]')
MAN=$BD/upload/rsna-knee-cnxt-fleet-k10/fleet_manifest.json; MANW=$BW/upload/rsna-knee-cnxt-fleet-k10w/fleet_manifest.json
$PY $K10/fn10_check.py --work $K10/build_fn/out_v<V>  --fn_build $K10/build_fn/FN_BUILD.json  --kernel fn --expect_n clean --expect_fleet clean --kaggle_commit --expect_members "$MEM"  --expect_manifest "$MAN"  --expect_tta "$BD/K5_BUILD.json" --expect_n_ranks log949   # FNCHECK PASS
$PY $K10/fn10_check.py --work $K10/build_fn3/out_v<V> --fn_build $K10/build_fn3/FN_BUILD.json --kernel fn --expect_n clean --expect_fleet clean --kaggle_commit --expect_members "$MEMW" --expect_manifest "$MANW" --expect_tta "$BW/K5_BUILD.json" --expect_n_ranks log949   # FNCHECK PASS
$PY $K10/k10n_check.py --work $K10/build_n/out_v<V> --k9n_build $K10/build_n/K10N_BUILD.json --tag a --expect_n clean --expect_fleet clean --kaggle_commit --expect_members "$MEM" --expect_manifest "$MAN" --expect_tta "$BD/K5_BUILD.json" --expect_n_ranks log949   # K10NCHECK PASS
$PY $K10/k2_check.py --work $BD/out_main_v<V> --expect_fleet clean --kaggle_commit --expect_mode strict --expect_w_main 0.5 --expect_members "$MEM" --expect_manifest "$MAN" --expect_tta "$BD/K5_BUILD.json"   # K2CHECK PASS
$PY $K10/k2_check.py --work $BD/out_solo_v<V> --expect_fleet clean --kaggle_commit --expect_mode solo --expect_members "$MEM" --expect_manifest "$MAN" --expect_tta "$BD/K5_BUILD.json"                       # K2CHECK PASS
```
- In every commit log: `[k2-fleet ...] RESULT OK fleet stage clean: 10 member(s)` (15 for FN10-3) with
  `f{F}_s1045t288f75_raw_ttr288-...` members, `TTA modes ...=AB`, MICROBENCH ~466-532 ms per 288 member; N kernels: two
  `[n949-child] [weights] sha256 7e5315da...` lines and `RESULT OK N stage clean`. If only the log949 rank check fails:
  < 2e-3 gap = fp16 noise (accept by hand), else STOP (KFN_RUNBOOK sec 2.3).

Submit (only after the gate PASS; UTC day = 05:30-05:29 IST):
```powershell
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k10fn  -v <V> -f submission.csv -m "FN10 parent-free P/F/N 0/.40/.60: E 288 frac75 s1045 x5 + D 224 frac75 s1043 x5 + public .949 CoAtNet (nartaa weights)"
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k10fn3 -v <V> -f submission.csv -m "FN10-3 parent-free P/F/N 0/.40/.60: C + D + E (15 members) + public .949 CoAtNet (nartaa weights)"
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k10n-a -v <V> -f submission.csv -m "K10N-a P/F/N .10/.30/.60: parent + E + D (10 members) + public .949 CoAtNet (nartaa weights)"
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k10-fleet -v <V> -f submission.csv -m "K10 strict P+F W.50: TTR288 frac75 s1045 x5 + TTR224 frac75 s1043 x5 raw A+B (clean, no N)"
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k10-solo  -v <V> -f submission.csv -m "K10 SOLO: E 288 frac75 + D 224 frac75 (10 members)"
```
- Rule context (unchanged): FN10 / FN10-3 / K10N-a attach the OAI-trained N weights; the clean pick stays N-free (K10 main
  or K9 main / KF) unless the host OKs OAI-trained weights.
- FN10-3 is the first 15-member fleet on a T4 (K7i ran 11): read its commit log's MICROBENCH line (expect ~466-532 ms per
  288 member, ~300-426 per 224 member) and `n_fail 0`; the runtime model (sec 0) assumes the per-member cost is additive.
- No K10 KF twin is built (not asked); if the clean pick needs one: `build_k2.py --kernel_only --slug rsna-sub-k10-kf
  --main_mode strict --outer_flat 0.60 ...` against `k10/build` plus the K10-DS patch, exactly as `build_k10.py` patches
  main / solo (not done, not tested).

## 3. Reads (pre-registered here, before any K10 read; MODEL)

| read | expected | rule |
|---|---|---|
| K10 SOLO vs K9 SOLO | -.001 to +.002 (E - C set swap; gold / OOF read 2.4 informs) | K10 SOLO <= K9 SOLO - .003 = audit set E before any further K10 use |
| FN10 vs FN | same display (+.0000 to +.0003, a one-set swap is ~.14x its set-level change) | adopt FN10 as the vehicle only if >= FN; tie -> K10 tie rule (2.4) |
| FN10-3 vs best of FN / FN10 | +.0001 to +.0005 | FN10-3 is the vehicle if >= the best 2-set FN read (more members, same family); within .001 prefer it when 2.4 said PREFERRED |
| K10N-a vs K9N-a | same display | parent-free first (PREREG R1): K10N-a matters only if R1 said the parent adds signal |
| K10 main vs K9 main | -.001 to +.002 | clean-pick candidate; within .001 prefer the KF twin of the final fleet (research sec 5) |

## 4. Files (`SP/k10/`)

| path | role |
|---|---|
| `k10_collect.sh` | laptop: sync tools to `~/k10_test/k10`, pick E / D / C, `k10_merge.py`, fetch + sha-verify (C / D copied from `SP/k9/ckpt` when the bytes match), parity npz, `members.json` (env: K10T, K10_ROOT_*, K10_INTERIM, K10_*_MARGIN / _BARS / _MIN_EPOCHS, K10_E_VAST_FOLDS, K10_LAPTOP_DIR) |
| `box/k5_pick.py` | K9 picker + K10-VREC (Vast receipts) |
| `box/k10_merge.py` | merge E + D + C, bars (E = C - .004), C / D == K9 bytes, Vast receipt rule, pre-registered override print |
| `box/k10_gold_read.py` | pre-push gold58 / gold74 read (sec 2.4); sha256 176f675f3f770328d47d8c95fc5095d47c70ef08968e6e7d56cecfa4eeddeef0 |
| `build_k10.py` | `--kit k10` (upload `-k10` + K10 main + SOLO), `--kit k10w` (upload `-k10w` only) |
| `build_k10n.py`, `build_k10fn.py` | K10N-a; FN10 / FN10-3 (`--kit fn10 / fn10_3`); `--test`, `--compare` |
| `make_k10_tools.py` | generates `cells/k10*_fleet_cell.py`, `k10n_check.py`, `box/k10_box_stage.py`, `box/k10n_box_stage.py`, `box/k10fn_box_stage.py` (counted substitutions; `--check`) |
| `fn10_check.py`, `k10n_check.py`, `k2_check.py` (== k9), `k9n_check.py` / `fn_check.py` (== k9n / kfn) | gates |
| `k10_stage_box.sh`, `box/k10_box_launch.sh` | box A0 staging (`~/k10_test/a0` real, `~/k10_test/a0sim` simulation) + the two queues |
| `_dev/k10_sim_setup.py`, `box/t_k10_pick.py`, `tests/t_k10_build.py`, `tests/runtime_k10.py` | simulation + tests |
| `_sim/` | the SIMULATION collect + builds (E = set C weights relabelled): **TEST ONLY, never create / version / push / submit** (every build there carries DO_NOT_PUSH_SIMULATION.txt or DO_NOT_PUSH_TEST.txt) |

## 5. Evidence (SIMULATION, 10-09; E = set C fold weights relabelled s1045 / frac75 / 21 ep, folds 1-4 with simulated Vast receipts)

Logs: laptop `SP/k10/tests/*.log`, `SP/k10/_sim/collect_sim.log`, `SP/k10/logs/`, box copies `SP/k10/tests/box/a0sim/`;
box `~/k10_test/{simtest, a0sim/runs, sim/gold, pickt}`.

| test | where | result |
|---|---|---|
| Simulated collect (`K10T=k10_test/simtest K10_ROOT_*=~/k10_test/sim/root K10_LAPTOP_DIR=k10/_sim bash k10_collect.sh`) | box + laptop | **K10COLLECT OK 15 member(s): E 5/5, D 5/5, C 5/5 (FINAL) [SIMULATION root ...]**; E f0 source box (mtime rule), E f1-f4 source vast (receipts); C / D picks == K9 sha256; C / D copied from SP/k9/ckpt |
| Real-root dry run (03:40, `bash k10_collect.sh`) | box | D 5/5 + C 5/5 OK (== K9 bytes, bars as K9), E 0/5 -> **K10COLLECT NOT FINAL, exit 3** (as intended); C / D now pre-staged in `~/k10_test/ckpt` |
| `box/t_k10_pick.py` (picker K10-VREC + merge) | box | **TK10PICK PASS 20/20**: valid receipts (f0 box, f1-f4 vast); missing / verify-failed / wrong closing line / wrong run receipts each FAIL with the right why; no `--vast_receipt_folds` -> box mtime rule; `--min_epochs 22` fails E 21; merge FINAL 15 with E bars C - .004; forced E f0 under its bar -> FAIL + the pre-registered override check printed; D bytes != K9 FAIL; receipt on an unlisted fold FAIL; listed fold without receipt FAIL; E with mix50 labels refused; E 4/5 -> rc 3 / INTERIM with --allow_interim; picker failure propagates |
| `tests/t_k10_build.py` (laptop, no model) | laptop | **TK10BUILD PASS 52/52**: generated tools == make(); K10-DS cells reverse to the K5 cell; K10 main / SOLO cells 0-30 == K9-main byte for byte, cell 31 = K10 cell, K2_CFG == K9's except MEMBERS / TTA / mode, metadata == K9's with only the slug changed; refusals (no k10 flag, set A identity, wrong set label, ema, stale ckpt, E key seen before, D key with other bytes, E 4/5 without do_not_submit); k10w = upload only (no kernel), E / D == K10 bytes; K10N-a cells 0-31 == K10 main, cell 32 == K9N-a's after the config line, config == K9N-a except TAG; FN10 / FN10-3 == the kfn FN kernel except the K10 parts (FN-IDS, N cell, metadata keys); --compare rebuilds byte-identical; non-final builds refused without --test; fn10_check slug logic; inference cap (every parent kernel = 5 x 224 + 5 x 288, no C); no notebook can mount the K9 slug; 5 new slugs |
| `tests/runtime_k10.py` | laptop | K10 6.97 / 7.27 h, K10N-a 7.40 / 7.95 h, FN10 1.45 / 2.00 h, FN10-3 1.88 / 2.54 h (central / slow; parent + 3 sets would be 8.49 h slow) |
| `k10_gold_read.py` on the simulation (E preds = C preds, sidecars) | box | **K10GOLD SIMULATION FINAL | K10 TIE RULE ON** (K10 - K9 +.0000, Fracture +.0000: E == C by construction) | COMPOSITION K10W - max -.0002 -> FN10-3 PREFERRED; one stale sidecar -> "MISSING predictions" + PARTIAL / no verdict; an --out outside ~/k10_test refused |
| A0 `neg_k9slug` (K10 main with the dataset under the K9 slug) | box CPU | **BOXSTAGE PASS 8/8**: "fleet dataset not attached ... rsna-knee-cnxt-fleet-k10", submission.csv deleted |
| A0 `main` (K10 main, real 10-member E(sim) + D fleet, 2 fold-0 studies) | box CPU | **BOXSTAGE PASS 40/40**, RESULT OK fleet stage clean: 10 member(s), all AB, decode probe 40/40, per-member OOF parity D f0 6.1e-4, E f0 3.1e-4 (bar 2e-2), stage 43.3 min (3 CPUs) |
| A0 `fn_chain` (FN10: FN-IDS + real fleet + N) | box CPU | **KFNBOX PASS 96/96**, fleet clean 10 members, N stage clean (child 9.2 min, 0 fallback), P/F/N 0/.40/.60; **cross-kit: FN10 submission_fleetsolo.csv == K10 main's, max diff 0.0** |
| A0 `n_chain` (K10N-a cells 31 + 32: real fleet + N) | box CPU | **K9NBOX PASS 86/86**: part A = every K9 gate via k2_check (strict, W .50, the 10 members, manifest bytes, TTA AB; the K8-INT line passes only as "[--allow_interim: TEST USE ONLY]" because the build is a simulation); part B N config W3 .10/.30/.60, TAG K10N-a, submission.csv == its pfn_103060 twin; N clean (0 fallback); **cross-kit: fleetsolo == K10 main's (0.0), N csv == FN10's (0.0)** |
| A0 `solo` (K10 SOLO, real fleet) | box CPU | **BOXSTAGE PASS 40/40**, fleet clean 10 members, per-member parity D f0 6.1e-4 / E f0 3.1e-4, fleet solo == raw mean of the 10 member rows (1.1e-16) |
| A0 `fn3_chain` (FN10-3: FN-IDS + real 15-member C + D + E fleet + N, dataset `-k10w`) | box CPU | **KFNBOX PASS 96/96**: RESULT OK fleet stage clean: 15 member(s), the workers ran exactly the 15, manifest bytes == local k10w build, N clean (0 fallback); fleet 68 min on 3 shared CPUs |
| A0 `n_stub` (K10N-a N stage, 4 studies, stub fleet) | box CPU | **K9NBOX PASS 46/46** (mirror unit, P == (plain + mirror) / 2, 3-leg blends == numpy reference) |
| A0 `fn_stub` (FN10 N stage, 4 studies, stub fleet) | box CPU | **KFNBOX PASS 55/55** |
| **A0 total** | box CPU, 2 queues on CPUs 34-36 / 37-39, 03:28-06:17 | **15/15 jobs PASS** (5 kernels: K10 main, K10 SOLO, K10N-a, FN10, FN10-3; 9 negatives incl. the K9-slug mount; 2 N stubs). Copies: `SP/k10/tests/box/a0sim/` |

What the simulation cannot show: real E weights (E = set C weights relabelled; the real E checkpoints are checked by the
picker at the collect and by the real A0 of 2.5, whose fold-0 parity covers the real E f0); T4 fp16 numerics and runtime
(commit gates + MODEL); the hidden rerun.
| `k10_gold_read.py --infer` on one simulated E fold (f2) | box CPU, NTHR 1 | prediction written + `.sha256` sidecar == the ckpt; == the cached set C f2 prediction (max diff 0.0, same weights); 61 min at 1 shared thread |
| A0 negatives K10N-a (stub fleet): `fleet_fail`, `child_sha`, `missing_weights` | box CPU | **K9NBOX PASS 15/15 each** (N NOT clean, submission.csv absent, why names the cause, no child left) |
| A0 negatives FN10 / FN10-3: `fn_neg_fleet_missing`, `fn3_neg_fleet_missing` (dataset not attached), `fn_neg_fleet_fail`, `fn_neg_child_crash` | box CPU | **KFNBOX PASS 33/33, 33/33, 19/19, 19/19** |

## 6. Fallbacks

- An E fold never lands (Vast failure): `vast_fetch.sh --partial` keeps partial output aside; the fold goes back to the
  box queue (~5.8 h). K10 waits; FN (K9) stays the vehicle. Never build with E < 5/5 except `K10_INTERIM=1` TEST builds
  (marked DO_NOT_SUBMIT_INTERIM, refused by k2_check / the builders).
- An E fold FAILS a picker check or its bar and is NOT ELIGIBLE for the override: no K10; record it; the 288 frac75
  family re-runs that fold only if the box has a free slot (research 3.2 filler).
- The gold read says OFF: K10 kernels may still be pushed for the information, but get no tie preference.
- A K10 A0 job FAILS: do not push that kernel; the others are independent (separate notebooks, shared datasets).
