# auto_f2probe.sh: unattended family-2 (F2) decorrelation probe

Built and dry-tested 2026-10-05 ~13:05 IST. **Not started.** The operator starts it. It implements JUMP_PLAN_10-05 sec 2 action 4 + sec 3 and HANDOFF TASK 4.

The launcher is installed on the box as `~/auto_f2probe.sh`, sha256 `39ba8f90ed11c051...`. It is byte-identical to `SP/f2probe/auto_f2probe.sh` and to the tested copy `~/f2probe_test/auto_f2probe.sh`.

## Start (operator, any time before fold 2 ends ~18:05)

```bash
setsid nohup bash ~/auto_f2probe.sh > /dev/null 2>&1 < /dev/null &
tail -f ~/fleet_runs/auto_f2probe.log
```

Status only (launches nothing, writes nothing, safe at any time): `bash ~/auto_f2probe.sh --check`.

The exact trainer argv it would use: `bash ~/auto_f2probe.sh --print-args ev2s` (or `r50`).

To turn off the EV2S watchdog fallback (step 5 below), start it with `F2P_WATCHDOG_FALLBACK=0 setsid nohup bash ~/auto_f2probe.sh ...`. No other `F2P_*` variable is for the operator; they exist for the dry test.

## What it does

1. **Wait.** Every 120 s it checks four things:
   - the fast chain is finished: `~/fleet_runs/auto_folds_fast.log` has a `chain done` line after its last `chain start` line, OR no `auto_folds_fast.sh` bash is alive;
   - the old chain `auto_folds.sh` is not alive;
   - `~/fleet_runs/PAUSE_CHAIN` does not exist;
   - `nvidia-smi` lists no python compute app. An nvidia-smi error counts as busy, as in the chain.

   All four must hold on two checks 120 s apart.
2. **Lock.** It takes `~/jump2/gpu.lock` without blocking. This is the lock `auto_round2.sh` uses. It holds the lock until the probe exits, and the trainer inherits it, so round 2 can never co-launch.
3. **Preflight (G-1).** It runs:
   `cd ~/fleet2_code && ~/miniconda3/envs/rsna/bin/python f2_preflight.py --store ~/fleet_store --n_train 3466 --cases convnext_tiny.fb_in22k_ft_in1k:1e-4 tf_efficientnetv2_s.in21k_ft_in1k:2e-4 resnet50.a1_in1k:3e-4`
   - It takes a few GPU minutes and writes nothing. Its output goes to `~/fleet_runs/f2probe_preflight_<MMDD_HHMMSS>.log`.
   - The launcher reads the last line, `PREFLIGHT <backbone>=<PASS|FAIL|OOM|ERROR> ...`.
4. **Choose.**
   - EV2S PASS: `tf_efficientnetv2_s.in21k_ft_in1k`, seed 142, lr 2e-4.
   - Else R50 PASS: `resnet50.a1_in1k`, seed 242, lr 3e-4.
   - Else ABORT. Nothing is launched.
5. **Launch the fold-0 probe alone**, in the foreground, and log its rc:
   ```
   cd ~/bigjump_build/code && ~/miniconda3/envs/rsna/bin/python fleet_train.py --f2 1 --backbone <B> --lr <LR> --seed <S> \
     --freeze_bn 1 --channels_last 1 --store ~/fleet_store --labels ~/rsna_train/llm_labels_v4_blend.csv \
     --folds ~/rsna_train/folds5.csv --fold 0 --exclude ~/rsna_train/gold74_eval_set.csv \
     --eval_labels ~/rsna_train/llm_labels_v4_blend.csv --gold ~/rsna_train/gold74_eval_set.csv \
     --epochs 10 --workers 10 --out ~/fleet_runs/f0_s<S>_f2v4 > ~/fleet_runs/f0_s<S>_f2v4.log 2>&1
   ```
   - It refuses if that out dir or its `.log` already exists.
   - Right before launch it re-checks the pause file, GPU python and the deadline.
   - If EV2S dies with the F2-T5 fp16 watchdog (rc 2 plus a `!! F2-T5` line) and R50 passed the same preflight, it launches the R50 probe once, with the same guards.

**Hard rules:**
- It never kills or signals anything. It only reads ps, pgrep and nvidia-smi.
- It gives up without launching once **2026-10-06 12:00 IST** is reached.
- A second copy refuses to start; the instance lock is `~/.auto_f2probe.instance`.

## Recipe check (why the command differs slightly from HANDOFF TASK 4)

- **`--lr` is added (the one real fix).**
  - The HANDOFF command has no `--lr`, so the trainer's default 1e-4 would apply. That is the ConvNeXt LR.
  - The reviewed recipe (`fleet2/SCHEDULE.md` 4c, `FAMILY2.md`) and the preflight's own cases use **EV2S 2e-4** and **R50 3e-4**.
  - Without the fix, the probe would have run at a different LR than the one the preflight checked.
- **`--freeze_bn 1 --channels_last 1`** are the trainer defaults. They are passed explicitly; the args recorded in the ckpt are the same.
- **Everything else stays at the trainer default, as in the reviewed recipe:**
  - head_lr 1e-3, wd 0.05, ema 0.999;
  - bs 2 x accum 4;
  - drop_path 0.1, dropout 0.2;
  - `--model f1 --aug light` (no BJ opt-in path), 10 epochs;
  - grad ckpt on: the bigjump F2 path hard-codes it, and that trainer has no `--grad_ckpt` flag.
- **No `--init_ckpt`:** ImageNet timm init, per the plan.
- **R50 keeps bs 2:** it runs alone. The `--bs 1 --accum 8` rule is only for sharing the GPU.
- **Verified on the box:**
  - the timm weights for all three backbones are cached;
  - `fleet_model_f2.py`, `fleet_model.py` and `fleet_common.py` are byte-identical in `~/bigjump_build/code` and `~/fleet2_code`, so the preflight builds the same model the trainer trains;
  - `--n_train 3466` = the "train 3466" of the v4 fold-0 runs;
  - neither out dir exists.

## Expected timeline (fold 2 ending ~18:05)

| When | Event |
|---|---|
| ~18:05 | Chain logs `chain done` |
| +2-4 min | Two all-clear checks |
| +a few min | Preflight finishes |
| ~18:15 | LAUNCH |
| ~00:00-01:00 10-06 | Probe exits (5-6 h is a GUESS; F2 was never timed) |

If the box runs folds 3-4 instead (placeholders removed), the chain ends ~02:10 and the probe starts ~02:15. That is still well before the 10-06 12:00 deadline.

## Log lines (`~/fleet_runs/auto_f2probe.log`, prefix `[auto-f2probe MM-DD HH:MM]`)

| Line | Meaning |
|---|---|
| `start pid N \| deadline ... \| trainer ... sha ... \| preflight ... sha ...` | Launcher started; config and code fingerprints. |
| `plan A (EV2S PASS): cd ... fleet_train.py ...` / `plan B (else R50 PASS): ...` | The two exact commands it may run. |
| `WARNING: .../fleet_model_f2.py differs from ...` | The trainer and preflight model sources differ. Not expected today; the launcher continues. Look before trusting the preflight. |
| `note: <out> (or its .log) already exists, so choice X would be refused` | Early warning for one candidate. |
| `REFUSE: both probe out dirs already exist` | Exit 3 at start. Nothing to do. |
| `waiting: <reasons>` | Not yet. Reasons are any of: `fast chain running (pid N), no 'chain done' yet;` `old chain auto_folds.sh alive (pid N);` `.../PAUSE_CHAIN present;` `python on GPU: N;` `nvidia-smi error (counted as busy);`. Logged when the reason changes, and once an hour while unchanged. |
| `all clear once (chain finished: done-line yes/no, bash alive/gone; ...); confirming in 120s` | First clear check. A second one follows 120 s later. |
| `waiting: GPU lock ~/jump2/gpu.lock is held by another launcher` | Another launcher (e.g. auto_round2.sh) holds the lock. Retries every 120 s. |
| `GPU lock taken; all clear on two checks 120s apart` | Committed: the preflight starts now. |
| `backing off inside the lock: <reasons>` | Something appeared between the check and the lock. Lock released, waiting again. |
| `preflight: start, GPU mem used/total ... \| cd ~/fleet2_code && ... > <pflog>` | Preflight running. |
| `preflight: exited rc=N after S s` | Preflight finished. |
| `preflight: GPU ... \| free X / Y GB before start` | The preflight's own memory line. |
| `preflight: [PASS\|FAIL\|OOM\|ERROR] backbone=... fp32_amax=... skipped_after_first=... peak_reserved_gb=... s_per_step=... proj_train_min_per_epoch=...` | One summary per case. `fp32_amax` < 16384 is needed; the ConvNeXt case is the G0 timing reference. |
| `preflight: verdicts EV2S=X R50=Y \| PREFLIGHT ...` | The parsed decision line. `missing` = no PREFLIGHT line, i.e. the preflight crashed. |
| `choose: EV2S passed -> ...` / `choose: EV2S <verdict>, R50 passed -> ...` | Backbone picked. |
| `ABORT: no family-2 backbone passed the preflight (...); nothing launched. Read <pflog>` | Exit 4. Plan sec 2 kill gate: no F2. The operator falls back to the TTR v4 folds. |
| `ABORT: preflight pid N still running after 2700 s; ... NOT killing it` | Exit 6. A hung preflight is left alone; it keeps the lock until it exits. |
| `launch of X deferred: <reasons>` + `GPU lock released; waiting again (choice X kept, no new preflight)` | A GPU python or PAUSE_CHAIN appeared after the preflight. It retries later without re-running the preflight. |
| `REFUSE: <out> (or <out>.log) already exists; not launching X` | Exit 3. Never overwrites a run. |
| `LAUNCH X: <backbone> seed S lr LR fold 0, alone on the GPU (mem used/total ...) \| cd ... > <out>.log` | Probe started. Watch `<out>.log`. |
| `probe X exited rc=N after M min \| last line: ...` | Probe finished. rc 0 ends with the `{"raw": .., "ema": ..}` line. |
| `done: X probe finished OK (<out>). Next (operator): gate F2-G ...` | Exit 0. |
| `EV2S stopped by the F2-T5 fp16 watchdog ...; fallback: R50 passed the same preflight, launching it once` | The EV2S fp16 failure that the preflight missed. R50 is launched next (`LAUNCH r50 ...`). |
| `EV2S stopped by the F2-T5 fp16 watchdog; no fallback (...)` | Exit 2. Fallback off, or R50 did not pass. |
| `FAILED: X probe exited rc=N; nothing else launched. Operator: read <out>.log` | Exit 1. A crash other than the watchdog (e.g. OOM, data error). |
| `GIVE UP: deadline 2026-10-06 12:00 +0530 reached ...` | Exit 5. Nothing was launched. |
| `another auto_f2probe.sh is already running ...` | A second copy was started. It exits 1; the first copy continues. |

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | Probe ran OK |
| 1 | Probe failed, or second instance |
| 2 | EV2S watchdog with no fallback, or bad setup |
| 3 | Out dir exists |
| 4 | Preflight: no PASS |
| 5 | Deadline |
| 6 | Preflight hung |

## Files it writes

- `~/fleet_runs/auto_f2probe.log`
- `~/fleet_runs/f2probe_preflight_*.log`
- the probe's own `~/fleet_runs/f0_s{142|242}_f2v4/` and its `.log` (written by the trainer)
- the empty lock file `~/.auto_f2probe.instance`

Nothing else.

## Operator controls

- **Hold before the launch:** `touch ~/fleet_runs/PAUSE_CHAIN`. This also holds the fast chain if fold 2 has not started yet. Remove the file to release.
- **Cancel before the launch:** stop the launcher bash yourself (it has no child while waiting).
- **After LAUNCH:** stopping the launcher does not stop the trainer. The trainer keeps the GPU lock until it exits.

## After the probe: gate F2-G (JUMP_PLAN sec 3, HANDOFF TASK 4 step 3)

Quick reads:

```
grep -E "^F2 backbone|F2-T5|TRAIN-ONLY|val AUC" ~/fleet_runs/f0_s142_f2v4.log
```

Use `f0_s242_f2v4.log` if R50 ran. The `ep 0 TRAIN-ONLY` line is the G0 timing.

PASS needs all four, against TTR v4 fold 0 `~/fleet_runs/f0_s1042_ttr`:
- (a) OOF AUC vs steven_v4 >= .8714;
- (b) gold58 >= .868;
- (c) mean per-label Spearman vs the Raptor proxy on the 867 fold-0 val studies <= .855;
- (d) Spearman vs the TTR v4 OOF <= .90.

PASS -> F2 folds 1-4. FAIL -> TTR v4 folds 1-4 on the fast trainer. Queue the next GPU job before the probe ends; the launcher queues nothing after the probe.

## Dry tests (CPU only, no GPU, nothing under ~/fleet_runs, ~/bigjump_build, ~/fleet2_code, ~/rsna_train, ~/jump2 written)

Kit on the box: `~/f2probe_test/`. Copy here: `SP/f2probe/test/`. Outputs: `test/evidence/`.

**1. `bash ~/f2probe_test/test/run_tests.sh`: 133/133 PASS, 26 scenarios.**
- **Fakes:**
  - `nvidia-smi` via PATH override (idle / busy / error);
  - a fake python. It emulates `f2_preflight.py` by printing fixtures in the preflight's real output format, and `fleet_train.py` by logging its exact argv and cwd and returning rc 0 / 1 / 2 (watchdog);
  - fake chain and old-chain bash processes;
  - a fake chain log.
- **Fixtures:** two are the REAL preflight output (`fleet2/tests/out/t7_preflight_cpu.log`, same `f2_preflight.py` sha 8e4da307 as the box), as LF and as CRLF. The rest are GPU-style variants of the same format: all PASS; EV2S FAIL + R50 OOM; EV2S ERROR; R50 FAIL; a traceback with no PREFLIGHT line.
- **Scenarios covered:**
  - waits on the fold python and the live chain;
  - waits on "chain done" while a GPU python remains;
  - chain gone without a done line;
  - a stale "chain done" before a new "chain start";
  - PAUSE_CHAIN before the preflight and during it;
  - an nvidia-smi error;
  - the old chain alive;
  - gpu.lock held by another holder;
  - a second instance;
  - out dir exists, only its log exists, both exist;
  - the deadline (and the default = 2026-10-06 12:00 IST);
  - every preflight verdict combination and a missing line;
  - a preflight hang (aborts, the process is not killed and keeps the lock until it ends on its own);
  - a GPU python that appears during the preflight (deferred, then launched without a second preflight);
  - the watchdog fallback, the fallback turned off, the fallback with R50 FAIL, and a non-watchdog crash;
  - `--check` writes nothing;
  - `--print-args` with production defaults equals the HANDOFF TASK 4 command plus `--lr` and the two explicit defaults.
- **Side-effect checks:** `~/fleet_runs` entries unchanged; `~/jump2/gpu.lock` size and mtime unchanged.

**2. `bash ~/f2probe_test/test/real_trainer_header.sh`: 15/15 PASS.**
- **How it runs:**
  - the exact production argv from `--print-args ev2s|r50` goes to the REAL `~/bigjump_build/code/fleet_train.py`;
  - only `--store` (an 8-study symlinked store) and `--out` (scratch) are swapped;
  - `--device cpu --min_cover 0 --epochs 0` is appended;
  - CUDA is hidden, HF is offline, no bytecode is written.
- **What it showed:**
  - all 16 flags are in `--help`;
  - the real labels, folds and gold csvs are read and the store contract passes;
  - the real model is built from the cached weights;
  - the F2 line reads exactly `F2 backbone tf_efficientnetv2_s.in21k_ft_in1k | name tag ev2s- | norm (0.5, 0.5, 0.5)/(0.5, 0.5, 0.5) | freeze_bn 1 | init timm | lr 0.0002/0.001 | channels_last 1`, and for R50 `name tag r50- | norm (0.485, ...)/(0.229, ...) | ... | lr 0.0003/0.001`;
  - no BJ opt-in path;
  - the F2-T3 guard refuses `--seed 42`;
  - `~/bigjump_build/code` is unchanged (files, sizes, mtimes, sha256).

**Independent verification (10-05 ~13:20, verifier).** No change to the launcher (sha256 still `39ba8f90ed11c051...`).
- Re-ran both suites on the box: 133/133 and 15/15 again (`test/evidence/verify_run_tests_rerun.out`).
- Added `test/adv_tests.sh` (box: `~/f2probe_test/test/adv_tests.sh`): **59/59 PASS** (`test/evidence/verify_adv_tests.out`):
  - the production pgrep regexes, taken from the launcher, vs 15 real-looking cmdlines (no match on the launcher itself, `tail -f` of the chain log, `vim`, `bash -c cat`, pgrep's own argv, `auto_round2.sh`), plus a live pgrep that finds exactly pid 404321;
  - the `chain done` awk on the chain's real `say()` format, incl. a fold line whose tail text ends in "chain done", the early-failure re-run lines, done-then-new-start and a missing log;
  - the verdict parser on prefix keys, absent keys, CRLF;
  - deadline passing during the preflight (no launch), EV2S OOM + R50 PASS (R50), ConvNeXt-only line (abort), PREFLIGHT line beats the per-case JSON, rc 1 with a full line (warns, goes by the line), a GPU python appearing between the first all-clear and the confirmation (resets, needs two fresh clears), empty nvidia-smi output, a missing chain log with a live chain (waits), EV2S out dir present with R50 PASS (refuses, never switches silently), rc 2 without the F2-T5 line (no fallback).
- Live `--check` at ~13:15 matched the box (chain 404321 alive, no done line, 1 python on GPU, lock free, both out dirs free). The trainer, preflight model and common files are byte-identical between `~/bigjump_build/code` and `~/fleet2_code`; all three timm weights are in the HF cache.
- FYI: another session's k5 GPU jobs also use `~/jump2/gpu.lock` (`flock -w 300`). While the probe holds the lock they wait 300 s and then fail; before the launch, the launcher waits for them (lock held or python on GPU).

**3. `bash ~/auto_f2probe.sh --check` on the real box (13:05).** It sees the fast chain pid 404321 alive with no done line, python on GPU 1 (fold 1), no pause, the lock free, both out dirs free and the deadline not reached. It wrote nothing.
