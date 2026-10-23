# K9 RUNBOOK: parent + [5 x TTR288 mix50 s1044 (set C) + 5 x TTR224 frac75 s1043 (set D)], equal members, raw best, TTA A+B, W .50

Written 2026-10-08 04:30-06:00 IST by Claude (NOVEL_PLAN_10-07 sec 4 row 10-09, sec 5; frac75 gate PASS 04:26). Nothing
was created, versioned, pushed or submitted on Kaggle. The operator (Eesh) runs every `kaggle` command below. No em dashes
in this file. Tags: **MEASURED**, **MODEL**, **GUESS**.

`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`,
`PY` = `C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe`, `B` = `training@192.168.10.30`
(`ssh -o BatchMode=yes`, append `< /dev/null`). Box steps are CPU only (CUDA hidden, nice 19). Box writes: `~/k9_test`
(kit area + simulation) and `~/novel/k9gold` (gold read) only; `~/fleet_runs` is written only by the sec 2.1 copy of the
kmrsuman outputs into their EMPTY placeholders (the frac75b RUNBOOK step that was always planned).

## 0. What K9 is

| set | runs | trained on | ttr | train labels | E | shipped key |
|---|---|---|---|---|---|---|
| C | `f{0..4}_s1044_ttr288mix50` | box chain `~/auto_after_ttr288.sh` (f4 ends ~09:45 10-08) | 288 | mix50 | 24 | `f{F}_s1044t288` |
| D | `f{0,1}_s1043_ttr_frac75` | Kaggle swapiy `rsna-train-ttr-frac75-f01` (receipt `kaggle_frac75_f01_run_receipt.json`) | 224 | frac75 | 17 | `f{F}_s1043f75` |
| D | `f{2,3}_s1043_ttr_frac75`, `f4_s1043_ttr_frac75` | Kaggle kmrsuman `-f23`, `-f4x` (receipts `kaggle_frac75_{f23,f4x}_run_receipt.json`; ~09:00) | 224 | frac75 | ~21 | `f{F}_s1043f75` |
| A | the K5 mix50 s1043 members | | | | | **REPLACED by D** (refused by build_k9.py / k9_merge.py) |

- Equal members: fleet F = raw-prob mean of the 10 members (each 1/10). RAW best checkpoints (picker `--tag raw`). Every
  member TTR (`fleet_model_ttr`, model_sha `f141aeed136dfc01`), every member A+B TTA. Worker = the K5 worker byte for byte
  (`21effe04f052c18d`), cell = the K5 cell (`e37f0604997cd40f`), KF cell = the K8 KF cell (`637eab6239433539`), frozen model
  files == K3, pack tool == K7 (all asserted by build_k9.py).
- **K9-KEY.** Member identity = (seed, fold, ttr_size, train_labels). A frac75 member would ship as
  `f{F}_s1043_raw_ttr224-...` = the K5 set A key of that fold; build_k9.py re-keys set D to `f{F}_s1043f75` (labels tag)
  and set C to `f{F}_s1044t288` (pack_fleet_ds.py P-TAG, file name only). No K9 key equals any K5/K7/K8 key (asserted).
- **Set acceptance:** FINAL only at C 5/5 AND D 5/5 (closing line + every picker check) and every fold >= its sanity bar.
  Bars (reference = set A raw cv of the same fold, the K5 members): C = A - .002 (as K8); **D = A - .004**: the frac75
  folds are E 17 / ~21 instead of 24 and are selected on OOF vs v4 labels while their Fracture target moved toward the
  teacher, so a healthy fold reads ~.000 to .002 below its mix50 twin (MEASURED f0 +.0002, f1 -.0016; paired fold SE
  ~.0014). .004 = the worst healthy gap seen + ~1.7 SE: a normal fold passes, a diverged or mislabelled one (-.01 or worse)
  fails. Override a single fold only by hand: `K9_D_BARS="F:<bar>"`.
- **Min epochs:** set D is collected with `K5_MIN_EPOCHS=14` (Eesh approved 10-06; covers a kmrsuman fold shortened to
  E 14-16 by a slow T4; E 17 already passes the default 16). Set C keeps 16.

| Kernel | Slug | MAIN_MODE | cell 31 | submission.csv |
|---|---|---|---|---|
| K9-main | `eeshsaxena/rsna-sub-k9-fleet` | strict | K5 cell | rank_pct(.50 r(P) + .50 r(F)), P = parent (cell-28 map); w40 / w60 variants |
| K9-solo | `eeshsaxena/rsna-sub-k9-solo` | solo | K5 cell | F alone |
| K9-KF | `eeshsaxena/rsna-sub-k9-kf` | strict | KF cell | rank_pct(.50 r(P_flat) + .50 r(F)), P_flat = parent rebuilt with flat outer CoAt weight .60 |

All three read ONE new version of `eeshsaxena/rsna-knee-cnxt-fleet` (21 files, 1086 MB: 5 code + 10 members + manifest +
metadata + 4 parity npz). Runtime = K8 (same 5 x 224 + 5 x 288 composition): stage ~58 min central / ~76 min slow per T4,
margin to the 8.6 h wall >= 1.33 h (`SP/k8/tests/runtime_k8.py`, MEASURED microbench inputs).

## 1. Quota decision (read before pushing anything)

- MEASURED 10-07 20:15 (`kaggle quota`): **2.52 h GPU left until 10-10 05:30 IST**; ~0.5 h per commit (MODEL). K8 (3) +
  K9 (3) = ~3 h: they do not both fit.
- **K8 and K9 cannot share today anyway:** both version the SAME dataset slug `eeshsaxena/rsna-knee-cnxt-fleet` with
  different files. Never version the dataset while a commit or a submission that reads it is still running or pending
  (whether a pending rerun reads the version it was committed with or the latest one is not documented; do not test it
  with a slot).
- **Commit refusal risk (open):** swapiy saw instant ERROR + empty log at < 12 h free without `--timeout`; Eesh's own K7i
  commits started fine at ~10 h free; at 2.52 h nobody has tried. `--timeout` is NOT added to submission kernels: the CLI
  (2.2.4) documents it only as "Limit the run time of a kernel to the given number of seconds"; nothing in the CLI or docs
  says the competition rerun ignores it, and a cap the quota allows (< 2.5 h) would kill the ~7 h hidden rerun if it
  carried over. So `--timeout` cannot help a submission kernel here either way.

| option | today (10-08) | after the 10-10 05:30 reset | quota today |
|---|---|---|---|
| A: K9 only | K9 main + SOLO + KF | nothing needed for K9 | ~1.5 h of 2.52 (~1 h spare) |
| **B: K9 main + SOLO (RECOMMENDED)** | K9 main + SOLO | KF twin of whichever fleet is final (K9-KF if K9 holds) | ~1.0 h of 2.52 (~1.5 h spare) |

**Recommendation: B.** The plan's final-pick logic (sec 5): pick 1 = best public read, with the tie rule favouring the
newest fleet whose gates passed (K9: frac75 gate PASS 04:26); pick 2 = the **KF twin of the final fleet**. Which fleet is
final is only known after the K9 reads (~tonight): K9 is kept only if K9 SOLO >= the K8 SOLO - .002 sanity bound (K8 SOLO
will not exist, so use **K7i SOLO .944 - .002 = .942**, the strongest scored SOLO) and its pre-push gold read keeps the tie
rule ON. A K9-KF committed today is wasted if K9 fails that bound, while deferring it costs nothing (full quota from 10-10,
deadline 10-22, KF slot plumbing still validated 12 days early). B also keeps ~1.5 h spare for one slow or retried commit.
Choose A only if Eesh wants the first real KF commit before the weekend (still fits, ~1 h spare). Do not spend today's
quota on K8 (it is only the fallback, sec 2.8).

## 2. Operator steps (MODEL timeline: kmrsuman ~09:00, TTR288 f4 ~09:45, collect + build ~10:00, A0 + dataset + 2 commits in parallel ~1-1.2 h, gates + submit ~11:30-12:30)

### 2.0 Status checks (any time)

```bash
B=training@192.168.10.30
ssh -o BatchMode=yes $B 'for F in 0 1 2 3 4; do echo "C f$F: $(tail -n 1 ~/fleet_runs/f${F}_s1044_ttr288mix50.log 2>/dev/null | cut -c1-70)"; echo "D f$F: $(tail -n 1 ~/fleet_runs/f${F}_s1043_ttr_frac75.log 2>/dev/null | cut -c1-70)"; done; ls ~/fleet_runs/kaggle_frac75_*_run_receipt.json' < /dev/null
#   every fold must end {"raw": ..., "ema": ...}. Never touch the chain, the queue runner or another ~/fleet_runs run.
```

```powershell
kaggle kernels status kmrsuman/rsna-train-ttr-frac75-f23
kaggle kernels status kmrsuman/rsna-train-ttr-frac75-f4x
```

### 2.1 kmrsuman f23 + f4x: fetch, verify, copy into the EMPTY placeholders (= frac75b RUNBOOK sec 5-6, verbatim)

```powershell
$KT = "C:\Users\eeshs\Downloads\rsna-knee\session_2026-10-07\kaggle_frac75b"
foreach ($S in "f23", "f4x") {
  $OUT = "$KT\out_$S"
  kaggle kernels output kmrsuman/rsna-train-ttr-frac75-$S -p $OUT --page-size 200 --file-pattern '^(?!_[a-z0-9]+_env/)(?!.*epochs/.*\.pt$)'
  python "$KT\verify_outputs.py" $OUT          # expect status DONE, plan E=21 (or 22), 0 mismatch, 12 absent, exit 0
}
```

```bash
KT=/c/Users/eeshs/Downloads/rsna-knee/session_2026-10-07/kaggle_frac75b; B=training@192.168.10.30
ssh -o BatchMode=yes $B 'test -e ~/fleet_runs/f0_s1044_ttr_frac75 || mkdir ~/fleet_runs/f0_s1044_ttr_frac75' < /dev/null
for X in "f23 f2_s1043_ttr_frac75 f3_s1043_ttr_frac75" "f4x f4_s1043_ttr_frac75 f0_s1044_ttr_frac75"; do
  set -- $X; S=$1; shift; OUT=$KT/out_$S
  for R in "$@"; do
    ssh -o BatchMode=yes $B "test -d ~/fleet_runs/$R && [ -z \"\$(ls -A ~/fleet_runs/$R)\" ] && [ ! -e ~/fleet_runs/$R.log ]" < /dev/null \
      || { echo "STOP: ~/fleet_runs/$R is not an empty placeholder"; break 2; }
    scp -r "$OUT/$R/"* $B:fleet_runs/$R/ && scp "$OUT/$R.log" $B:fleet_runs/$R.log
  done
  scp "$OUT/run_receipt.json" $B:fleet_runs/kaggle_frac75_${S}_run_receipt.json
  ssh -o BatchMode=yes $B "~/miniconda3/envs/rsna/bin/python ~/kt_frac75b/verify_outputs.py ~/fleet_runs --receipt ~/fleet_runs/kaggle_frac75_${S}_run_receipt.json | tail -n 2" < /dev/null
done
```

- If a notebook ERRORs / is PARTIAL: its D folds never land, D < 5/5, K9 cannot ship today: go to sec 2.8 (K8 fallback).

### 2.2 Incremental gold predictions (CPU; start each as its folds land; ~13 min per 224 fold, ~15-20 per 288 fold)

```bash
ssh -o BatchMode=yes $B 'cd ~/novel/k9gold && CUDA_VISIBLE_DEVICES= NTHR=8 nohup nice -n 19 ionice -c3 ~/miniconda3/envs/kmimic/bin/python -W ignore k9_gold_read.py --infer --members none > infer_$(date +%H%M).log 2>&1 &' < /dev/null
#   last line "K9GOLD PARTIAL ... INFO ONLY" until C 5/5 + D 5/5. D f0/f1 (k9gold/preds) and set A (gold74_preds) are
#   cached; C f1-f3 were inferred into ~/novel/k8gold/preds at 04:40 (reused read-only). Remaining: D f2-f4, C f4.
```

### 2.3 Collect (~3 min; fetches C f4 + D f2-f4, the other 6 are already on the laptop from the 05:05 interim collect)

```bash
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
bash "$SP/k9/k9_collect.sh"
#   gate (last line): "K9COLLECT OK 10 member(s): C 5/5, D 5/5  (FINAL)"   (no "[SIMULATION ...]" tail)
#   exit 3 = a set is not 5/5 yet (a fold has no closing line / no copy): wait or sec 2.8. exit 1 = STOP: the FAIL /
#   K9MERGE line names the check ("below their sanity bar" = read that run's log; accept only by hand K9_D_BARS="F:<bar>").
#   Every D fold must print "Kaggle-filled run (.../kaggle_frac75_<f01|f23|f4x>_run_receipt.json): ... sha256 == receipt".
```

### 2.4 Pre-push gold read (required; ~2 min if 2.2 ran, else ~1 h): K9 set vs K8 set

```bash
ssh -o BatchMode=yes $B 'cd ~/novel/k9gold && CUDA_VISIBLE_DEVICES= NTHR=8 nice -n 19 ionice -c3 ~/miniconda3/envs/kmimic/bin/python -W ignore k9_gold_read.py --infer 2>&1 | tail -n 9' < /dev/null
#   last line "K9GOLD FINAL | K9 TIE RULE ON|OFF: gold58 pass A K9 - K8 ... ; Fracture K9 - K8 ... | K8 read: C - A ..."
#   -> write it into HANDOFF. Pre-registered (script header): ON when d58(K9 - K8) > -.003 AND gold58 Fracture K9 - K8 >= 0.
#   It never blocks the push; it decides whether the sec 5 tie rule may favour K9 for pick 1.
```

### 2.5 Build (~30 s) + box A0 (CPU, ~50-70 min on the loaded box, window A0_AGE_H 7.0 = 1.6 h; run 2.6 in parallel)

```bash
$PY -W ignore "$SP/k9/build_k9.py"
#   gate: "K9 BUILD OK FINAL (10 members: C 5/5, D 5/5: ['D:f0_s1043f75', ..., 'C:f4_s1044t288'])"; dir k9/build (never
#   build_i<N> / build_sim, which carry DO_NOT_SUBMIT_INTERIM.txt / DO_NOT_PUSH_SIMULATION.txt)
BD="$SP/k9/build"
MEM=$($PY -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['members']))" "$BD/K5_BUILD.json" | tr -d '[:cntrl:]')
bash "$SP/k9/k9_stage_box.sh" "$BD"                  # gate: "DS IDENTICAL (21 files)", "K9STAGE OK"
ssh -o BatchMode=yes $B 'bash ~/k9_test/k5/box/k5_box_launch.sh all; bash ~/k9_test/k5/box/k5_box_launch.sh kfneg' < /dev/null
ssh -o BatchMode=yes $B 'cd ~/k9_test/runs; for n in A0_main A0_solo A0_kf A0_kfneg_perturb A0_kfneg_missing; do f=$(ls -t ${n}_*.log | head -1); echo "== $f"; grep -E "^FAIL|BOXSTAGE|RESULT|KF outer" $f | cut -c1-200; done' < /dev/null
#   PASS = A0_main / A0_solo / A0_kf each "BOXSTAGE PASS k/k" + "RESULT OK fleet stage clean: 10 member(s)" (A0_kf also
#   "KF outer map flattened to 0.6"); both kfneg "BOXSTAGE PASS 10/10" with RESULT NOT CLEAN naming the KF check.
```

### 2.6 Dataset version + push (PowerShell; while the A0 runs; submit only after A0 PASS + K2CHECK PASS)

```powershell
$K9B = "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k9\build"
Test-Path "$K9B\DO_NOT_SUBMIT_INTERIM.txt"; Test-Path "$K9B\DO_NOT_PUSH_SIMULATION.txt"   # both must be False
$UP = "$K9B\upload\rsna-knee-cnxt-fleet"
New-Item -ItemType Directory -Force -Path (Join-Path "$env:TEMP\.kaggle\uploads" ($UP -replace ':', '_')) | Out-Null   # Windows CLI staging bug
(Get-ChildItem $UP).Count                                   # 21
kaggle datasets version -p $UP -m "K9: TTR288 mix50 s1044 x5 + TTR224 frac75 s1043 x5 (raw, A+B), K5 worker"
kaggle datasets status eeshsaxena/rsna-knee-cnxt-fleet      # until: ready (1.09 GB)
kaggle datasets files  eeshsaxena/rsna-knee-cnxt-fleet      # 21 files incl. f{0..4}_s1043f75_raw_ttr224-... and f{0..4}_s1044t288_raw_ttr288-...
kaggle quota                                                # GPU left; each commit ~0.5 h
kaggle kernels push -p "$K9B\kernel\rsna-sub-k9-fleet"
kaggle kernels push -p "$K9B\kernel\rsna-sub-k9-solo"
# option A only (else after the 10-10 05:30 reset, and before any later dataset version): 
#   kaggle kernels push -p "$K9B\kernel\rsna-sub-k9-kf"
kaggle kernels status eeshsaxena/rsna-sub-k9-fleet          # until complete (same for -solo); write each version V down
#   status error within ~2 min with an EMPTY log = Kaggle refused the session (low quota): do NOT retry with --timeout
#   (sec 1); push again after the 10-10 05:30 reset.
kaggle kernels output eeshsaxena/rsna-sub-k9-fleet -p "$K9B\out_main_v<V>" --page-size 200
kaggle kernels output eeshsaxena/rsna-sub-k9-solo  -p "$K9B\out_solo_v<V>" --page-size 200
# (KF, when pushed: kaggle kernels output eeshsaxena/rsna-sub-k9-kf -p "$K9B\out_kf_v<V>" --page-size 200, and
#  Test-Path "$K9B\out_kf_v<V>\diagnostics\btkd_parent_after_rad.npz" must be True)
```

### 2.7 Gates (Git Bash; each must end "K2CHECK PASS"), then submit

```bash
MAN="$BD/upload/rsna-knee-cnxt-fleet/fleet_manifest.json"
$PY "$SP/k9/k2_check.py" --work "$BD/out_main_v<V>" --expect_fleet clean --kaggle_commit --expect_mode strict --expect_w_main 0.5 --expect_members "$MEM" --expect_manifest "$MAN" --expect_tta "$BD/K5_BUILD.json"
$PY "$SP/k9/k2_check.py" --work "$BD/out_solo_v<V>" --expect_fleet clean --kaggle_commit --expect_mode solo                     --expect_members "$MEM" --expect_manifest "$MAN" --expect_tta "$BD/K5_BUILD.json"
# KF (when pushed): ... --work "$BD/out_kf_v<V>" ... --expect_mode strict --expect_w_main 0.5 ... --expect_outer_flat 0.6
```

- Main / solo must show `PASS no KF flat parent ...`; never pass `--allow_interim`. The commit log must show
  `[fleet g] members:` with 10 x `[KneeFleetNetTTR]` (5 x `f{F}_s1043f75_...`, 5 x `f{F}_s1044t288_...`), `TTA modes ...=AB`
  for all 10, and the MICROBENCH line (~270-430 ms per 224 member, ~466-532 ms per 288 member).

```powershell
# submit (UTC day = 05:30 IST to 05:29 IST next day)
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k9-fleet -v <V> -f submission.csv -m "K9 strict P+F W.50: TTR288 mix50 s1044 x5 + TTR224 frac75 s1043 x5 raw A+B (10 members)"
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k9-solo  -v <V> -f submission.csv -m "K9 SOLO: 10 TTR members A+B (C 288 mix50 + D 224 frac75)"
# option A / after the reset: kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k9-kf -v <V> -f submission.csv -m "K9-KF strict P_flat(.60)+F W.50 (final pick 2 candidate)"
```

### 2.8 Fallback: set D not 5/5 (a kmrsuman notebook failed) or a D fold FAILS its checks

K8 = set A + set C (`SP/k8/K8_RUNBOOK.md` sec 2.2-2.3, verified 10-08, see its sec 7b). Same quota logic: K8 main + SOLO
today, KF after the reset. If TTR288 f4 has no closing line, neither K8 nor K9 is FINAL: submit nothing today.

## 3. Reads (pre-registered)

| read | expected [MODEL / GUESS] | rule |
|---|---|---|
| K9 SOLO | .943-.946 | **>= .942** (K7i SOLO .944 - .002; stands in for the plan's "K8 SOLO - .002") keeps K9 as the final-kit candidate; **<= .941 = build problem: audit before any use** |
| K9 main (W .50, map parent) | .946-.948 | pick-1 candidate: K9 is pick 1 when it reads >= best other - .001 AND sec 2.4 said K9 TIE RULE ON (plan sec 5) |
| K9-KF (when submitted) | ~ main - .001 | .944-.946 = expected; <= .943 = investigate the rebuild before using it as pick 2 |

Final picks (plan sec 5): pick 1 = best public read under the tie rule; pick 2 = the KF twin of the final fleet (K9-KF if
K9 holds, else KF of the fallback fleet). Both must be scored submissions; push the KF twin before any later dataset
version replaces the K9 files.

## 4. Files (`SP/k9/`, a copy of `SP/k8` sources; k7 / k8 untouched except the fixes listed in K8_RUNBOOK sec 7b)

| Path | Role |
|---|---|
| `k9_collect.sh` | laptop: sync tools to `~/$K9T/k5`, picker set C (as K8) + set D (`--train_labels labels_frac75_logit.csv --key_tag f75 --require_receipt` + the 3 frac75 receipts, `K5_MIN_EPOCHS` 14), `k9_merge.py`, fetch + sha-verify checkpoints, parity npz, `k9/members.json`. Env: `K9T` (k9_test), `K9_ROOT_C/D`, `K9_INTERIM`, `K9_C/D_MARGIN`, `K9_C/D_BARS`, `K9_C/D_MIN_EPOCHS`, `K9_LAPTOP_DIR` |
| `box/k5_pick.py` | K8 picker + **K9-REC** (a receipt whose cfg SEED is a per-fold map, as the kmrsuman f23 / f4x notebooks write it, is read per fold; the K8 copy crashed with TypeError on it) + **K9-KEY** `--key_tag` (OOF copy name `oof_f{F}_s1043f75_...`) |
| `box/k9_merge.py` | merge C + D; 5/5 rule for both, per-fold bars vs set A (C .002, D .004), refuses set A / B members, ema picks, duplicates (seed, fold, ttr, labels) |
| `build_k9.py` | ONE command: upload + 3 kernels; sets by (seed, ttr, train_labels), K9-KEY keys, refuses set A / B, asserts K5 worker / K5 cell / K8 KF cell / frozen files / pack tool; writes DO_NOT_* markers for sim / interim builds; K5_BUILD.json `k8: true` (so k2_check's K8-INT guard applies unchanged) + `k9: true` |
| `k9_stage_box.sh`, `box/k5_box_launch.sh`, `box/k5_ids.py` | A0 on the box (K9 notebooks, `~/k9_test`) |
| `box/k9_gold_read.py` | the pre-push gold58 / gold74 read (sec 2.4), box copy `~/novel/k9gold/k9_gold_read.py` |
| `build_k2.py`, `make_k2_kernel.py`, `make_kf_cell.py`, `k2_check.py`, `code/`, `cells/`, `box/k2_box_stage.py` | byte-identical to the k8 copies (asserted by `tests/regress_k9.py` K9-SAME) |
| `_dev/k9_sim_setup.py`, `box/t_k9_pick.py`, `tests/t_k9_build.py`, `tests/t_k9_kf.py`, `tests/regress_k9.py` | simulation root + tests (sec 5) |
| `build_sim/` | SIMULATED build (D f2-f4 + C f4 = set A weights relabelled): **TEST ONLY, never version / push** |
| `build_i6/` | INTERIM build from the real runs at 05:05 (C f0-f3, D f0-f1): **never version / push / submit** |

## 5. Evidence (2026-10-08, MEASURED; logs in `SP/k9/tests/`, `SP/k9/_sim/`, `SP/k9/_interim/`, box `~/k9_test/`)

1. **Picker / merge** (`box/t_k9_pick.py`, box CPU, sim root + real runs): **TK9PICK PASS 13/13** (`~/k9_test/sim/t_k9_pick_v2.log`):
   T1 set D 5 raw frac75 picks, receipts f01/f01/f23/f23/f4x, `f75` OOF names; **T2 the unchanged K8 picker crashes on the
   kmrsuman-format receipt (`TypeError: int() ... not 'dict'`)**; T3 receipt order f4x first still gives D f0 to f01;
   T4 no frac75 receipt + `--require_receipt` FAILS all folds; T5 a wrong receipt sha256 FAILS that fold only; T6 mix50
   labels refused both ways; T7 E 17 < `--min_epochs 18` FAILS f0/f1, 16 and 14 pass; T8 set C beside set D; T9 merge
   FINAL 10, D bars = A - .004; T10 `--d_margin 0` FAILS D f1 (.8910 < .8926), the override is recorded; T11 set A
   member / ema / duplicate refused; T12 real runs C 4/5 D 2/5: rc 3, interim with `--allow_interim`.
   The first run found a bar-rounding bug (`round(A - m, 6)` moved a bar up by < 5e-7 and failed folds sitting exactly on
   it, T10b): fixed in k9_merge.py and k8_merge.py (`.bak_0850`), re-run PASS.
2. **Simulated collect + build**: `K9T=k9_test/simtest K9_ROOT_C/D=~/k9_test/sim/root K9_LAPTOP_DIR=k9/_sim bash k9_collect.sh`
   -> **K9COLLECT OK 10 member(s): C 5/5, D 5/5 (FINAL) [SIMULATION root ...]** (C f0-f3 REAL, C f4 sim; D f0/f1 REAL with
   the real f01 receipt, D f2-f4 sim with simulated kmrsuman-format receipts); `build_k9.py --out k9/build_sim` -> **K9 BUILD
   OK SIMULATION ... FINAL (10 members)**, upload 1086.1 MB / 21 files, kernels fleet `6c81de50`, solo `ec8cc7d2`, kf `6e706b0f`.
3. **Laptop**: `make_kf_cell.py --check` OK (`637eab62`); **TK9BUILD PASS 12/12** (set A / B / ema / duplicate / interim /
   foreign labels / set-label mismatch refused; every key tagged; KF cell == K8 KF cell; the sim build's 3 notebooks carry
   the right cell, MEMBERS and OUTER_FLAT; 5 fold groups of 2; k2_check's K8-INT line FAILS the sim build);
   **TK9KF PASS 34/34** (the K8 KF suite on the K9 kernels); **REGRESS_K9 PASS 8/8** (K5 and K7i rebuilt byte-identical
   with the k9 tools; shared tools == k8).
4. **Box A0 on the simulated build** (CPU, 2 fold-0 studies): see sec 5b.
5. **INTERIM real collect + build** (05:05): `K9_INTERIM=1 bash k9_collect.sh` -> **K9COLLECT OK 6 member(s): C 4/5, D 2/5
   (INTERIM ...)** on the real `~/fleet_runs` (D f0/f1 sha256 == the real f01 receipt); `build_k9.py` -> `k9/build_i6`
   with `DO_NOT_SUBMIT_INTERIM.txt`; `k2_check --expect_tta build_i6/K5_BUILD.json` FAILS ("an INTERIM build is NEVER
   submitted"), as intended.
6. **Gold**: D f0 pass A gold58 .9163 / gold74 .9083 (== the gate read), D f1 gold58 .9111 / gold74 .9086
   (`~/novel/k9gold/preds`); C f1-f3 inferred into `~/novel/k8gold/preds` (04:40 job, done 05:36).
   PARTIAL reads, INFO ONLY (not fold-matched, NOT the decision read; `~/novel/k9gold/k9_gold_read_20261008_0548.txt`,
   `~/novel/k8gold/k8_gold_read_20261008_0537.txt`): K8 read C{0-3} - A gold58 -.0015 (SE .0037), gold74 +.0006;
   fold-matched D{0,1} vs A{0,1}: gold58 .9169 vs .9194, gold74 .9115 vs .9138 (Fracture up, MCL / Synovitis / LatOA down,
   as in the gate read); K9 - K8 (C 4 + D 2 vs A 5 + C 4) gold58 -.0005 (SE .0019), gold74 +.0010.

## 5b. Box A0 on the simulated build (10-08, box CPU, 2 fold-0 studies; D f0 / C f0 REAL, members otherwise as sec 5.2)

| run | result |
|---|---|
| A0_kfneg_perturb / A0_kfneg_missing (05:00) | BOXSTAGE PASS 10/10 each (KF check fails loud before any worker, submission.csv absent) |
| A0_main / solo / kf, first run 05:00 (`--age_h 7.9`) | stopped by me at 05:25 (pid groups 829662-4, ours): the identical K8 run hit the 0.7 h CPU wall at 47.0 min (K8_RUNBOOK sec 7b), this one would have too; launcher fixed (`A0_AGE_H`, default 7.0) |
| A0_main / solo / kf, re-run 05:25 (`A0_AGE_H` 7.0) | **BOXSTAGE PASS 40/40, 40/40, 44/44**, `RESULT OK fleet stage clean: 10 member(s)` each, 10 x KneeFleetNetTTR incl. 5 x `f{F}_s1043f75_...`, all AB, decode probe 40/40, KF map flattened (rebuild 0.0); per-member pass A vs its trainer OOF: **D f0 (real Kaggle frac75) 6.1e-4**, C f0 3.1e-4 (bar 2e-2); OOF mean parity 2.5e-4; stage 50.6-51.1 min (the K8 twin run: 40/40, 40/40, 44/44, 46.7-50.9 min; logs `~/k{8,9}_test/simtest/runs/A0_*_0525.log`) |

## 6. Gaps / caveats

1. **D f2-f4 are simulated in the A0** (set A weights relabelled frac75); the real kmrsuman checkpoints are checked by the
   picker (closing line, cv == closing line, OOF rescored == cv to 1e-12, model_sha, ttr_size, fold / seed, labels, amp,
   decode probe, receipt sha256) when they land. The real kmrsuman receipt format is assumed from the notebook source
   (`nb_src/c01_lib.py`: `cfg=CFG`, SEED per trainer); if the collect FAILS on a receipt, read the FAIL line before anything.
2. **A0 per-member OOF parity covers fold 0 only** (box DICOM mirror = fold 0 + gold): D f0 and C f0 get parity; folds 1-4
   are covered by the npz identities, the decode probe and the commit gate.
3. **KF has never run on Kaggle** (first real KF commit = K9-KF or K8-KF); its K2CHECK needs the FULL output (diagnostics/).
4. The quota / refusal behaviour at 2.52 h is unknown (sec 1). If refused: push after the reset.
5. The interim guard lives in k2_check (`--expect_tta <K5_BUILD.json>`) and in the DO_NOT_* marker files.
6. Disk: `k9/build_sim` + `k9/build_i6` ~2.9 GB, `k9/_sim/ckpt` 1.1 GB on the laptop; box `~/k9_test` ~5 GB. All removable
   after K9 is submitted (never `~/fleet_runs`).
