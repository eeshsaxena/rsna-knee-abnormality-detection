# RUNBOOK (Eesh): frac75 folds 0 + 1 on a teammate's Kaggle GPU (`rsna-train-ttr-frac75-f01`)

Written 2026-10-07 by Claude. Nothing here was done on Kaggle: no dataset was created or versioned, nothing was pushed,
shared or submitted. Every Kaggle step below is yours. No em dashes.

## 0. What and why

- **The run.** One Kaggle notebook trains the frac75 Fracture-target variant of TTR224 on folds 0 and 1, one fold per T4, in
  parallel. The recipe is the box mix50 recipe of J1 (`~/fleet_runs/f0_s1043_ttr_mix50`) with ONE change:
  `--labels labels_frac75_logit.csv`. Seed is 1043, so fold 0 is paired with J1.
- **Fold 0** is the frac75 probe (plan Q1). It is scored on the box with the pre-registered gate.
- **Fold 1** saves the V100 the Q2 fold-1 job (~4 h).
- **On PASS** the V100 queue only has to train folds 2-4.
- **Sized for a teammate with ~10 h of weekly quota.** The training budget is 8.4 h and the hard stop is 9.2 h, so the
  whole session ends by ~9.3 h. Measured basis (Kaggle T4 x2 receipts on the box: `~/fleet_runs/kaggle_f34_run_receipt.json`,
  `kaggle_s1044_f01_run_receipt.json`, `kaggle_s1044_f23_run_receipt.json`):
  - setup + store + checks + probe end at 0.139-0.144 h;
  - 0.78-0.81 s/it;
  - 27.4-28.3 min per epoch with both evals;
  - the slowest run overran its epoch estimate by 5.6 %;
  - finalize takes under 1 min.

| | expected | bound |
|---|---|---|
| E (epochs) | **17** at 0.80-0.81 s/it; 18 at <= ~0.785 s/it; 16 at 0.85; WARNING below 16 | min(24, floor((8.4 h - elapsed - 2 min) / (1733 x s/it x 1.21))) |
| `--save_epochs` | (E-2)-E (last 3 epochs: ckpt + OOF) | |
| training ends | T0 + 8.1 h (E 17) to 8.35 h (E 18); 8.9 h even with the f34 overrun | hard wall T0 + 9.2 h, then quiet-wait <= 2 min + kill grace <= 1 min + finalize |
| session ends | ~T0 + 8.1-8.9 h | < T0 + 9.3 h, plus a few minutes of output saving (~1.8 GB) |
| GPU quota used | ~8.3-9.0 h | ~9.4 h (so the teammate needs >= 10 h) |

- **E vs J1's 24 epochs.** E ~17 makes the gate a slightly asymmetric pairing. Read it with
  `PREREG_KAGGLE_ADDENDUM.md` (in `$SP\kaggle_frac75\`; box copy `~/kt_frac75/`):
  - PASS stays PASS, and a wide FAIL stays FAIL.
  - A miss by < .003 on every failing gate is INCONCLUSIVE-NARROW. Your call: tie-break with the registered 24-epoch
    box probe, or accept FAIL.

```
SP = C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad
KT = $SP\kaggle_frac75          (archive without ds\: Downloads\rsna-knee\session_2026-10-07\kaggle_frac75)
  ds_labels\                          NEW dataset eeshsaxena/rsna-knee-frac75-labels (labels_frac75_logit.csv + metadata)
  ds\rsna-knee-fleet-train\           the EXISTING dataset; rebuilt byte-identical: do NOT create a new version
  kernel\rsna-train-ttr-frac75-f01\   kernel folder, id eeshsaxena/... (only if you ever run it yourself)
  kernel_teammate\...                 the same notebook, id YOUR_KAGGLE_USERNAME/... (in the teammate pack)
  PREREG_KAGGLE_ADDENDUM.md           how the E ~17 Kaggle probe is scored and read (written before the run)
  verify_outputs.py                   checks downloaded outputs against run_receipt.json
  box\                                score_frac75_kaggle.py, tiebreak_edit.sh, queue_lines.txt + their tests (box: ~/kt_frac75/box/)
  nb_src\ build.py dry\               notebook source, builder, box dry tests (not uploaded)
Teammate pack: Downloads\rsna-knee\friend_frac75_pack\ (+ .zip): README_TEAMMATE.md, TODO.md, this runbook, notebook, cli\
labels csv: sha256 5036425cd0d64f27f08acfe6865d668ecf830e11562ed8921c63afa782c2c8c0, md5 ad7582b817243f8f22e8bcfdd7e3f84a (= PREREG)
B = training@192.168.10.30   (ssh -o BatchMode=yes $B '...' < /dev/null)
```

## 1. Before anything is shared (rule check + no duplicate work)

1. **Team.** The teammate must ALREADY be on the competition team "Eesh saxena". Check the Team tab. Rules forbid
   sharing code or data outside a team, so share nothing before that.
2. **Quota.** The teammate's weekly GPU quota must show **>= 10 h**.
3. **Box.** The box must not have started its own frac75 fold-0 probe (queue Q1). Check:
   ```bash
   ssh -o BatchMode=yes $B 'ls -ld ~/fleet_runs/f[01]_s1043_ttr_frac75* 2>&1; grep -h "LAUNCH Q1\|REFUSED Q1" ~/fleet_runs/queue_runner.log 2>/dev/null | tail -n 2' < /dev/null
   ```
   - Nothing listed: go on.
   - A `LAUNCH Q1` line, or a non-empty `f0_s1043_ttr_frac75`: fold 0 is already training on the V100, so do NOT push
     this notebook as is. Ask Claude for a folds 1 + 2 build (one line in `nb_src/c01_lib.py` + rebuild + dry test).
     See also sec 9.

## 2. Create the labels dataset (PowerShell, ~2 min)

```powershell
$UP = "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\kaggle_frac75\ds_labels"
# Windows kaggle CLI staging-dir bug: pre-create %TEMP%\.kaggle\uploads\<path with ':' -> '_'> first
New-Item -ItemType Directory -Force -Path (Join-Path "$env:TEMP\.kaggle\uploads" ($UP -replace ':', '_')) | Out-Null
(Get-ChildItem $UP).Count                                   # 2 (dataset-metadata.json + labels_frac75_logit.csv)
(Get-FileHash "$UP\labels_frac75_logit.csv").Hash           # 5036425CD0D64F27F08ACFE6865D668ECF830E11562ED8921C63AFA782C2C8C0
kaggle datasets create -p $UP                               # private by default
kaggle datasets status eeshsaxena/rsna-knee-frac75-labels   # repeat until: ready
kaggle datasets files eeshsaxena/rsna-knee-frac75-labels    # 1 file: labels_frac75_logit.csv (~1.33 MB)
```

- If the CLI prints `No such file or directory` for a path under `.kaggle\uploads`, create exactly that folder and run
  the create again.
- Leave `eeshsaxena/rsna-knee-fleet-train` alone. It is unchanged; the build asserts byte identity with the published
  files.
- The notebook re-checks the csv against its baked sha256 within its first minute. A wrong file aborts before any GPU
  work.

## 3. Share the inputs and hand over the pack (web UI)

1. Share these four items with the teammate's username. On each page: Settings / Sharing / add user / **Can view**.
   - dataset `eeshsaxena/rsna-knee-fleet-train`
   - dataset `eeshsaxena/rsna-knee-frac75-labels` (new)
   - notebook `eeshsaxena/rsna-fleet-prep-s0` (its output = half of the image store)
   - notebook `eeshsaxena/rsna-fleet-prep-s1`
2. Send `Downloads\rsna-knee\friend_frac75_pack.zip`. The teammate follows `README_TEAMMATE.md` (`TODO.md` has the
   joint checklist).
3. The teammate keeps the slug `rsna-train-ttr-frac75-f01` and shares the notebook back with `eeshsaxena` (Can view).

## 4. When the teammate's run shows `running` (= T0)

1. **Write down T0.** It should end by ~T0 + 8.1-8.9 h, never after T0 + 9.3 h (+ output saving). Example: running at
   16:00 IST -> complete ~00:10-00:55 IST, at the latest ~01:20 IST.
2. **Reserve the two out dirs on the box right away**, with EMPTY placeholders. Then the V100 queue never trains Q1 or Q2
   fold 1 itself: the runner marks a line whose out dir exists `#REFUSED`. Only do this if sec 1 step 3 showed nothing.
   ```bash
   ssh -o BatchMode=yes $B 'for F in 0 1; do d=~/fleet_runs/f${F}_s1043_ttr_frac75; if [ -e $d ] || [ -e $d.log ]; then echo "EXISTS $d (stop, read sec 9)"; else mkdir $d && echo "placeholder $d"; fi; done' < /dev/null
   ```
3. **What the queue does meanwhile**, depending on the frac75 verdict file `~/queue/gates/frac75.verdict`:
   - If the queue reaches the frac75 lines before the verdict exists: Q1 and Q2 f1 are refused, and Q2 f2 runs its gate
     command once. That command uses the ORIGINAL scorer and cannot score a Kaggle run, so it reports not-scorable. The
     queue holds 30 min, defers f2-f4 and moves on to Q3 (no idle GPU).
   - Once you write the verdict (sec 7), the deferred f2-f4 run next in queue order.

## 5. Monitor

```powershell
kaggle kernels status TEAMMATE/rsna-train-ttr-frac75-f01     # queued -> running -> complete | error
```

| status | meaning / action |
|---|---|
| `error` at once, **EMPTY log** | Kaggle refused the session (quota, or 2 GPU sessions already running on that account). Do not retry blindly: ask for a screenshot of the quota bar. Retry only with >= 10 h left and no other GPU session. Placeholders: see sec 8 |
| `error` within ~30 min | an input check failed: the last `[kt ...] ABORT` line says why (table in sec 8). Fix it; the teammate re-runs only if the quota still shows >= 9.5 h |
| `error` later | unexpected: fetch the log (sec 6) and ask Claude |
| `complete` | sec 6 |

## 6. Fetch and verify (PowerShell)

```powershell
$OUT = "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\kaggle_frac75\out_v1"
kaggle kernels output TEAMMATE/rsna-train-ttr-frac75-f01 -p $OUT --page-size 200 --file-pattern '^(?!_[a-z0-9]+_env/)(?!.*epochs/.*\.pt$)'
python "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\kaggle_frac75\verify_outputs.py" $OUT
```

- The pattern drops the 12 per-epoch checkpoints (~1.35 GB) and any vendored `_xxx_env/` dir. It keeps:
  - the 4 best checkpoints (2 per fold, ~113 MB each);
  - the best and per-epoch OOF csvs (`epochs/oof_*_ep{E-2..E}.csv`: the gate's schedule-end read needs them);
  - the logs, `run_receipt.json` and `probe/`.
- Expected output:
  - `status DONE`;
  - `plan E=17` (or 18) and `save_epochs=15-17` (or 16-18);
  - `0 mismatch, 12 absent` (the 12 absent = the skipped `epochs/*.pt`); exit 0.
- Exit 2 = PARTIAL / ABORTED: read the warnings. Fold 0 is scorable only if `f0_s1043_ttr_frac75.log` ends with
  `{"raw": ..., "ema": ...}`.
- Exit 1 = a file differs or a ckpt is UNUSABLE: fetch again before anything else.

## 7. Box: copy, verify, score the gate, act (Git Bash)

```bash
KT=/c/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad/kaggle_frac75
OUT=$KT/out_v1; B=training@192.168.10.30
# (if the sec 4 placeholders were never made and sec 1 step 3 is still clean, make them now with the sec 4 command)
for F in 0 1; do R=f${F}_s1043_ttr_frac75
  ssh -o BatchMode=yes $B "test -d ~/fleet_runs/$R && [ -z \"\$(ls -A ~/fleet_runs/$R)\" ] && [ ! -e ~/fleet_runs/$R.log ]" < /dev/null \
    || { echo "STOP: ~/fleet_runs/$R is not an empty placeholder (sec 9)"; break; }
  scp -r "$OUT/$R/"* $B:fleet_runs/$R/
  scp "$OUT/$R.log" $B:fleet_runs/$R.log
done
scp "$OUT/run_receipt.json" $B:fleet_runs/kaggle_frac75_f01_run_receipt.json
ssh -o BatchMode=yes $B '~/miniconda3/envs/rsna/bin/python ~/kt_frac75/verify_outputs.py ~/fleet_runs --receipt ~/fleet_runs/kaggle_frac75_f01_run_receipt.json | tail -n 3; for F in 0 1; do tail -n 1 ~/fleet_runs/f${F}_s1043_ttr_frac75.log; done' < /dev/null
```

- Result: `~/fleet_runs/f{0,1}_s1043_ttr_frac75/` + `.log` + `kaggle_frac75_f01_run_receipt.json`. These are the same
  names the queue, `argv_frac75.sh` and the scorers use.
- Checkpoint names equal set A's (`f{F}_s1043_raw_ttr224-...`), and their args record `epochs E`, `save_epochs (E-2)-E`
  and `train_labels labels_frac75_logit.csv`.

**7a. Score the gate.** CPU only, nice 19, ~12-15 min; gold_infer5 runs on the 74 gold studies.

```bash
ssh -o BatchMode=yes $B 'CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/rsna/bin/python ~/kt_frac75/box/score_frac75_kaggle.py > ~/kt_frac75/gate_run.log 2>&1; echo rc $?; grep -E "KAGGLE ADDENDUM:|^G[123] |FRAC75 GATE|schedule-end|internal control|KAGGLE ADDENDUM READ" ~/kt_frac75/gate_run.log' < /dev/null
```

- What the wrapper is: `score_frac75_kaggle.py` = the pinned pre-registered `~/novel/frac75/score_frac75.py`, run
  unchanged except that ckpt args `epochs` / `save_epochs` may differ from J1's.
- Bars are unchanged: G1 >= .8774, G2 >= .912, G3 >= .9062.
- It writes the registered outputs `~/novel/frac75/{gold74_preds,out}/` (gate json `gate_f0_s1043_ttr_frac75.json`).
- `rc 0` = PASS, `1` = FAIL, `2` = integrity error (read the INTEGRITY line), `3` = not scorable (no closing line).
- Then act on the last line, `KAGGLE ADDENDUM READ`.

**7b. PASS or FAIL: write the queue verdict.** This re-runs the same scorer through the queue adapter: ~1 min, the gold
preds are cached.

```bash
ssh -o BatchMode=yes $B 'F75_SCORER=$HOME/kt_frac75/box/score_frac75_kaggle.py bash ~/queue/gates/frac75_gate.sh > ~/kt_frac75/gate_queue.log 2>&1; cat ~/queue/gates/frac75.verdict; bash ~/queue/queue_runner.sh --check | grep -E "frac75|CHECK"' < /dev/null
```

- **PASS:**
  - Q1 and Q2 f1 are refused (dirs exist), so nothing is duplicated.
  - Q2 f2, f3, f4 (21 ep, gc0, ~3.7-4.0 h each, ~12 h) launch as soon as the V100 is free. If the queue runner is not
    started yet, start it per `~/queue/QUEUE_RUNBOOK.md` sec 2.
  - Q4 then takes its frac75 lines.
  - **K9** (plan sec 4) = parent + [5x TTR288 mix50 + 5x TTR224 frac75]:
    - the builder needs per-set `train_labels` + a labels tag in the member key (`build_k7.py` l.79 asserts mix50);
    - folds 0/1 are E ~17 members, so collect them with `K5_MIN_EPOCHS=14` (approved 10-06).
- **FAIL:**
  - Q2 f2-f4 become `#SKIP`; Q3 (TTR320 probe) runs next, then the Q4 mix50 lines.
  - Close target re-mixing as registered: no re-run, no other seed or bar.
  - The Kaggle f0/f1 stay in `~/fleet_runs` as a record. They are not members.

**7c. INCONCLUSIVE-NARROW** (every failing gate within .003 of its bar). Write NO verdict from the Kaggle run, and decide:

- **Accept FAIL:**
  ```bash
  ssh -o BatchMode=yes $B 'echo "FAIL frac75 probe: Kaggle read INCONCLUSIVE-NARROW, accepted as FAIL by Eesh $(date +%F)" > ~/queue/gates/frac75.verdict' < /dev/null
  ```
- **Tie-break on the V100** with the registered 24-epoch probe (J1 argv, gc1, ~4.6-4.8 h) in a NEW out dir. Its
  original-scorer PASS / FAIL decides folds 2-4.
  ```bash
  ssh -o BatchMode=yes $B 'EDITOR=$HOME/kt_frac75/box/tiebreak_edit.sh bash ~/queue/qctl.sh edit && bash ~/queue/qctl.sh top "$(grep "^Q1b " ~/kt_frac75/box/queue_lines.txt)" && bash ~/queue/qctl.sh check | grep -E "frac75|CHECK"' < /dev/null
  ```
  - `tiebreak_edit.sh`, run under the queue lock, points every frac75 gate `via` at
    `F75_RUN=/home/training/fleet_runs/f0_s1043_ttr_frac75_b24`.
  - `Q1b` (= `argv_frac75.sh` fold 0, out renamed `_b24`) goes on top.
  - When Q1b ends, Q2 f2 runs that gate command once and the verdict follows. If the runner had already used its one gate
    command earlier, run it by hand after Q1b ends:
    `F75_RUN=$HOME/fleet_runs/f0_s1043_ttr_frac75_b24 bash ~/queue/gates/frac75_gate.sh`.
  - On PASS, the K9 fold-0 member is the b24 run, and the Kaggle f0 is a spare.

## 8. If the Kaggle run fails or is refused

- **Remove the EMPTY placeholders** (`rmdir` refuses non-empty dirs):
  `ssh -o BatchMode=yes $B 'rmdir ~/fleet_runs/f0_s1043_ttr_frac75 ~/fleet_runs/f1_s1043_ttr_frac75' < /dev/null`.
- **Restore Q1 and Q2 f1 only if the runner already refused them.** Check with
  `bash ~/queue/qctl.sh show | grep "REFUSED.*frac75 f[01]"`. If it did, re-add them (Q2 f1 first, then Q1, so Q1 is on
  top; qctl refuses a duplicate of a still-active line, so this is safe either way):
  ```bash
  ssh -o BatchMode=yes $B 'L=~/kt_frac75/box/queue_lines.txt; bash ~/queue/qctl.sh top "$(grep "^Q2 frac75 f1 " $L)" && bash ~/queue/qctl.sh top "$(grep "^Q1 frac75 f0 " $L)"' < /dev/null
  ```
- **Abort reasons** (receipt `status`, also the last `[kt ...] ABORT` line of the notebook log):

| status contains | meaning / fix |
|---|---|
| `labels_frac75_logit.csv not found` | the labels dataset is not attached: add input `eeshsaxena/rsna-knee-frac75-labels` |
| `sha256 ... != baked` | the attached labels csv is not the box file: re-create the dataset from `ds_labels\` (sec 2) |
| `found at N paths with DIFFERENT bytes` | two different label datasets attached: keep only `eeshsaxena/rsna-knee-frac75-labels` |
| `code dataset ... found []` | `eeshsaxena/rsna-knee-fleet-train` not attached or not shared yet |
| `prep tar parts: missing` | prep notebook outputs not attached: add the OUTPUT of `rsna-fleet-prep-s0` and `-s1` |
| `only X GB free` | scratch disk too small: ask Claude (`MIN_FREE_GB` in `nb_src/c01_lib.py`) |
| `tar sha256 mismatch`, `store names differ`, `store files differ` | corrupt mount or changed prep outputs: run again once; if it repeats, stop |
| `parity with the box failed`, `model check failed`, `trainer header differs` | Kaggle image drift: stop and ask Claude (the log has the diff) |
| `fewer than 2 GPUs` / `GPU(s) visible, need 2` | accelerator was not GPU T4 x2 |

## 9. The box already started its own frac75 fold-0 probe (no duplicate work)

- **Never stop the box probe.** It is the registered probe (24 epochs, exact J1 pairing). The queue scores it with the
  ORIGINAL scorer when Q2 f2 reaches the head, and that verdict decides.
- **Kaggle fold 0 = a spare, kept out of the collectors' way:**
  ```bash
  ssh -o BatchMode=yes $B 'mkdir -p ~/fleet_runs/kaggle_extra/f0_s1043_ttr_frac75_kgl' < /dev/null
  scp -r "$OUT/f0_s1043_ttr_frac75/"* $B:fleet_runs/kaggle_extra/f0_s1043_ttr_frac75_kgl/
  scp "$OUT/f0_s1043_ttr_frac75.log" $B:fleet_runs/kaggle_extra/f0_s1043_ttr_frac75_kgl.log
  ```
  - Optional report-only read, which never touches the verdict:
    `score_frac75_kaggle.py --run ~/fleet_runs/kaggle_extra/f0_s1043_ttr_frac75_kgl --tag kaggle_f0_report`.
- **Kaggle fold 1 is still useful** as long as Q2 f1 has not started. Q2 f1 cannot start before the probe ends and the
  gate passes. Make `~/fleet_runs/f1_s1043_ttr_frac75` an empty dir and copy fold 1 into it as in sec 7 (F=1 only). Q2
  f1 is then refused, which saves ~4 h of V100.
- **If the teammate has not pushed yet:** do not push the f01 notebook. Ask Claude for a folds 1 + 2 build.

## 10. Never

- Version `eeshsaxena/rsna-knee-fleet-train`.
- Edit `~/novel/frac75/PREREG.md` or its bars.
- Use `LAUNCH=1 launch_frac75.sh` while `~/queue/queue_runner.sh` lives.
- Kill a box job.
- Pick a run whose log lacks the closing line.

## 11. Verification done, ALL PASS (read 16:02 IST; ran 2026-10-07, box CPU, `CUDA_VISIBLE_DEVICES=` nice 19, under `~/kt_frac75/` only)

- `dry_src/kt_dry.py` (log `~/kt_frac75/dry_run_v1.log`): DRY TEST 74/74 PASS (DRYEXIT 0), incl. both mount layouts, CRLF/missing-wheel faults, and 4 labels-file aborts (missing, flipped, two different copies, mix50 instead of frac75)
- `dry_src/kt_dry_g.py` (log `~/kt_frac75/dry_run_v1_g.log`): PHASE G 8/8 PASS (verify_outputs: complete run exit 0, watchdog run exit 2, tampered run exit 1)
- `dry_src/kt_smoke_real.py` (log `~/kt_frac75/smoke_real.log`): SMOKE 19/19 PASS (the real fleet_train.py, frac75 labels, seed 1043, rc 0; args differ from J1 only in amp/epochs/fold/save_epochs/train_labels)
- `box/test_score_kaggle.py` (log `~/kt_frac75/test_score_kaggle.log`): SCORE-KAGGLE TESTS 22/22 PASS (gate wrapper end to end; J1-as-probe control correctly reads FAIL)
- `box/test_queue_lines.sh` (log `~/kt_frac75/test_queue_lines.log`): QUEUE-LINES TESTS 10/10 PASS (tie-break and restore queues pass queue_runner.sh --check, 0 BAD)
