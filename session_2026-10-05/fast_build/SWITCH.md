# SWITCH.md: TTR mix50 folds 2-4 on the fast trainer (--grad_ckpt 0)

Written 2026-10-05 ~10:05 IST by the fast-path workstream. NOTHING has been switched: the running chain
(~/auto_folds.sh, pid 311059, fold 1 = pid 311068) is untouched. The operator does the switch below.

Labels: MEASURED = observed on the box today; MODEL = arithmetic from measured numbers; GUESS = judgement.

## 0. TL;DR

* `~/fast_build/code2/fleet_train.py` = the bigjump trainer + ONE opt-in flag `--grad_ckpt {1,0}` (default 1).
  Every other file in code2 is byte-identical to `~/bigjump_build/code` (cmp). Diff: `~/fast_build/fleet_train.diff`.
* `--grad_ckpt 0` changes speed and memory only. MEASURED: weights, EMA, OOF, cv and losses are BITWISE identical to
  `--grad_ckpt 1` in every check (unit test in both cudnn modes, and two end-to-end trainer runs).
* Speed (MEASURED, A/B interleaved, 3 reps, real recipe, sharing the GPU with fold 1): train step time ratio
  gc0/gc1 = 0.786 (prefetched batches) and 0.777 (live DataLoader). Eval time unchanged (ratio 1.017).
* Fold time (MODEL): 24 x (10.33 x 0.78 + 2.19) min = ~4.1 h instead of 5.0 h (J1 MEASURED 5.0 h).
  Folds 2-4: ~12.3 h instead of ~15.0 h, i.e. done ~02:00 10-06 instead of ~04:45 10-06 if fold 2 starts ~13:45.
* Memory (MEASURED): the gc0 trainer process peaks at 9072 MiB (nvidia-smi, incl. CUDA context; allocator peak
  8186 MiB alloc / 8632 MiB reserved). Alone on the 16384 MiB V100 (+313 MiB gnome-remote-desktop): ~7.0 GB headroom.
* New chain script: `~/auto_folds_fast.sh` (folds 2 3 4, --grad_ckpt 0, same guards + 2 extra safety guards). Not started.

## 1. The change

```
+ap.add_argument('--grad_ckpt', type=int, default=1, choices=(0, 1))   # FB-1
- ... grad_ckpt=True).to(dev)                    (3 constructor calls: TTR, family 1, family 2)
+ ... grad_ckpt=bool(A.grad_ckpt)).to(dev)
+if not A.grad_ckpt:                              # informational args key, only on the opt-in path
+    args.update(grad_ckpt=False)
+    print(f'FB-1 opt-in: grad_ckpt 0 (no activation checkpointing) | model_sha {args["model_sha"]}', flush=True)
```
(plus a 5-line header comment FB-1).

Default path = no executed statement changes its value: without the flag `A.grad_ckpt == 1`, so (a) argparse has
one more option (no RNG, no I/O); (b) each constructor receives `grad_ckpt=bool(1)`, i.e. `True`, the same value as
the old literal, and the constructor only uses it in `if grad_ckpt: self.cnn.set_grad_checkpointing(True)`;
(c) the `if not A.grad_ckpt:` block does not run, so `args` and stdout are unchanged. MEASURED: pristine copy of the
old trainer vs code2 default, same tiny run: weights (raw, EMA, epochs 1 and 2), OOF csvs, cv, logged losses and
ckpt args all identical (args diff = []).

Why `--grad_ckpt 0` does not change the math: timm 1.0.29 `ConvNeXtStage.forward` runs `checkpoint_seq(blocks)` only
when `grad_checkpointing` is set, otherwise `blocks(x)`; torch 2.5.1 non-reentrant checkpoint (timm default
use_reentrant False) with `preserve_rng_state=True` recomputes the identical forward with the stashed CPU+CUDA RNG
state (same drop_path masks) and restores the RNG afterwards, so forward values, gradients and the global RNG stream
(DataLoader base seed, dropout masks) are the same. Only activation memory and recompute time differ.

model_sha (MEASURED, recomputed from code2): family 1 `fa35880fca963652`, TTR `f141aeed136dfc01` = J1 ckpt args.
Ckpt args with --grad_ckpt 0 gain only `grad_ckpt: False`. Checked consumers: worker loader (reads named keys),
pack_fleet_ds (args signature only for duplicate detection; fold differs anyway), k3 build_k2 / k3_pick_ttr (named
keys only). None compares whole args across folds.

## 2. Evidence (all MEASURED, logs in ~/fast_build/logs2/)

| check | result |
|---|---|
| Unit parity, 2 real store studies (all 4 views), heavy aug drawn once, bs 2, accum 1, 4 applied optimizer steps at full recipe LR, drop_path .2 / dropout .3 active, cudnn.benchmark True (trainer mode) | losses, grad norms, logits: identical; final raw + EMA weights BITWISE equal (max abs diff 0.0; update size 3.8e-3); same state_dict keys (`parity_bench.log`) |
| Same, deterministic cudnn | bitwise equal again (`parity_det.log`) |
| End-to-end: the real trainer, recipe flags, 16 train + 8 val studies, 2 epochs (4 optimizer steps at accum 4, 32 micro-steps), save_epochs 1-2 | ref (old trainer) == code2 default == code2 --grad_ckpt 0: raw + EMA weights of epochs 1 and 2 bitwise equal, OOF max diff 0.0, cv equal; args diff old vs default = [], default vs gc0 = [grad_ckpt] (`compare_tiny.log`; run in deterministic cudnn mode) |
| Worker strict-load: the loader block of ~/bigjump_build/code/fleet_worker.py exec'd verbatim (model_sha + ttr_size_t asserts, strict key check) | 6 ckpts saved by the gc0 path (top-level best raw/ema, epochs/ep2 raw/ema, 2 unit ckpts): all load, 199 keys, keys and shapes == J1, loaded tensors == ckpt tensors (`worker_check.log`, WCHK_ALL_OK True) |
| Speed A/B (`speed_ab.log`), real recipe step (bs 2, accum 4, heavy aug, 16 slices, EMA, AdamW, clip, fp16), 30 optimizer steps = 120 micro-steps per segment, order AB BA AB, while fold 1 trains on the same GPU | prefetched real batches: gc1 95.9 s, gc0 75.4 s, ratio 0.786 (per rep .777 .784 .797). Live DataLoader (10 workers): gc1 96.2 s, gc0 74.7 s, ratio 0.777 (.756 .792 .783), data wait <= 0.08 s per segment. Eval forward (no_grad, 24 slices): ratio 1.017 (no change) |
| DataLoader alone (CPU, 10 workers, heavy aug, while fold 1's loader also runs) | 7.72 batches/s; gc0 alone needs ~3.6 batches/s (MODEL) -> ~2.1x margin |
| Peak memory gc0 | real trainer (tiny run): alloc 8186 MiB, reserved 8632 MiB, nvidia-smi process 9072 MiB. Speed bench: reserved 8518 MiB, nvidia-smi 8912 MiB. The 9.8 GiB test cap was never binding (reserved 1.4 GB below it). gc1 for reference: reserved 2608 MiB, nvidia-smi ~3.0 GB |

Contention cost (MEASURED): during the 20 min speed bench fold 1 slowed from 0.36 to ~0.6-0.75 s/it (~8-10 min of
fold-1 time lost). The other checks took ~6 min.

## 3. Expected gain (MODEL)

* J1 (fold 0, same recipe, alone): train-only 10.33 min/epoch, epoch total 12.52 min (eval raw + EMA + saves 2.19 min).
* gc0: 10.33 x 0.78 + 2.19 = 10.2 min/epoch -> 24 epochs ~4.1 h per fold (vs 5.0 h). Range with ratio .75-.85
  (GUESS for alone vs shared GPU): 3.95-4.4 h.
* Fold 1 ETA (MODEL): ~13:45 IST (it was at ep 5 at 09:54, 12.5 min/epoch alone from now on). Check:
  `grep -c TRAIN-ONLY ~/fleet_runs/f1_s1043_ttr_mix50.log` (24 = training done; the run then prints `{"raw":..,"ema":..}`).
* Folds 2-4: ~12.3 h vs ~15.0 h. Saves ~2.7 h of the night.

## 4. Switch procedure (operator)

The new chain `~/auto_folds_fast.sh` waits before each fold while `~/fleet_runs/PAUSE_CHAIN` exists, while any python
holds the GPU, AND while the old chain (`/bin/bash /home/training/auto_folds.sh`) is alive; it skips existing fold
dirs, logs to `~/fleet_runs/auto_folds_fast.log`, writes the SAME out dirs / logs / ckpt names as the old chain
(`~/fleet_runs/f{2,3,4}_s1043_ttr_mix50[.log]`). If a gc0 fold fails within 45 min (e.g. CUDA OOM at start) its dir
and log are renamed `..._fastfail_<mmdd_HHMM>` (never deleted) and that fold is re-run once with `--grad_ckpt 1`
(= the old trainer bit for bit). Dry-run with a stub trainer (MEASURED): waits on the live old chain and the live GPU
python; fallback, skip and launch paths behave as described. Verify fix 10-05: the fallback re-run now waits on the
same 3 guards (PAUSE_CHAIN, GPU python, old chain) and re-checks that the out dir is absent, and an nvidia-smi error
counts as "GPU busy" (see section 7).

**Step 0 (now, any time before fold 1 ends):**
```
touch ~/fleet_runs/PAUSE_CHAIN
```
Safe while fold 1 runs: the old chain only reads it between folds.

**Step 1: wait until fold 1 has exited.** All of:
```
tail -2 ~/fleet_runs/auto_folds.log          # "fold 1: exited rc=0 ({"raw": ..., "ema": ...})"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv   # no python left
ls ~/fleet_runs/f1_s1043_ttr_mix50/          # *_raw_*_best.pt, *_ema_*_best.pt, oof_*.csv, epochs/
```
If rc != 0: stop here and look at the fold-1 log before switching.

**Step 2: start the new chain** (it cannot launch yet: PAUSE_CHAIN and the live old chain both hold it):
```
setsid nohup bash ~/auto_folds_fast.sh > /dev/null 2>&1 < /dev/null &
tail -3 ~/fleet_runs/auto_folds_fast.log
# expect: "chain start (pid N): folds 2 3 4 with --grad_ckpt 0; old chain pids: [311059 ]"
#         "fold 2: waiting (PAUSE_CHAIN=yes, gpu python=0, old chain=[311059 ])"
```

**Step 3: stop the OLD chain by PID, only after these checks:**
```
P=$(pgrep -f '^(/usr)?(/bin/)?(ba)?sh ([^ ]*/)?auto_folds\.sh$'); echo "$P"   # exactly one PID (311059)
tr '\0' ' ' < /proc/$P/cmdline; echo        # /bin/bash /home/training/auto_folds.sh
ps --ppid $P -o pid,etime,cmd               # ONLY "sleep 120" (the pause loop), NO python
tail -1 ~/fleet_runs/auto_folds.log         # "fold 1: exited rc=0 ..."  (NOT "fold 2: launching")
```
All four true -> `kill $P` (SIGTERM to that bash only; its `sleep 120` child ends by itself within 2 min), then
`kill -0 $P 2>/dev/null && echo STILL-ALIVE || echo gone`. If `ps --ppid` shows a python child, the old chain has
already started fold 2 on the old trainer: do NOT kill anything; let fold 2 finish (outputs valid) and repeat
steps 0-4 at the fold-2/3 boundary. The new chain stays parked behind the old-chain guard meanwhile and will skip
fold 2 because its dir exists.

**Step 4: release:**
```
rm ~/fleet_runs/PAUSE_CHAIN
```
Within 2 min:
```
tail -2 ~/fleet_runs/auto_folds_fast.log    # "fold 2: launching (grad_ckpt 0) -> /home/training/fleet_runs/f2_s1043_ttr_mix50"
head -4 ~/fleet_runs/f2_s1043_ttr_mix50.log # "BJ opt-in: model ttr ... model_sha f141aeed136dfc01"
                                            # "FB-1 opt-in: grad_ckpt 0 (no activation checkpointing) | model_sha f141aeed136dfc01"
```
After ~15 min: `grep "/1733" ~/fleet_runs/f2_s1043_ttr_mix50.log | tail -2` should read ~0.27-0.29 s/it (MODEL; was
0.36 alone), `nvidia-smi` python ~9.0-9.1 GB. After the first epoch: `ep 0 TRAIN-ONLY ~8 min`.

**Optional variant B (nobody at the box when fold 1 ends; avoids GPU idle time between fold 1 and the switch).**
Deviates from step 3's "pause loop" condition on purpose: do steps 2 and 3 NOW, while fold 1 trains. Start the new
chain (it waits on the fold-1 python and on the old chain), then kill the old chain's bash PID after the same
cmdline check; `ps --ppid $P` will show the fold-1 python (pid 311068): kill ONLY the bash PID, never the python.
SIGTERM to the bash parent does not reach its child (MEASURED on this box with a dummy script: the script's bash
died at once, its running child kept running re-parented to PID 1, and no later line of the script ran): fold 1
keeps training and writes its ckpts, OOF and log normally; only the "fold 1: exited" line in auto_folds.log is lost
(k3_pick_ttr reads the fold log, not that file). Skip step 0 (or `rm ~/fleet_runs/PAUSE_CHAIN` right after the
kill). The new chain launches fold 2 within 2 min of fold 1's python leaving the GPU.

## 5. Rollback

* Back to the checkpointed trainer mid-way: `touch ~/fleet_runs/PAUSE_CHAIN`, let the current fold finish (its
  outputs are valid: same math), check the new chain is in its pause loop (`pgrep -af auto_folds_fast.sh`: use
  only the PID whose cmdline is exactly `bash /home/training/auto_folds_fast.sh`, since any shell that merely mentions
  the name is listed too; `ps --ppid <pid>` shows only `sleep 120` or `sleep 60`, no python), kill that PID, `rm ~/fleet_runs/PAUSE_CHAIN`, start
  `setsid nohup bash ~/auto_folds.sh > /dev/null 2>&1 < /dev/null &` (it skips existing fold dirs).
* Early gc0 failure: automatic (rename + one `--grad_ckpt 1` re-run, which waits on PAUSE_CHAIN / GPU python / old
  chain like a normal launch), visible in auto_folds_fast.log. To stop a fold on purpose, `touch ~/fleet_runs/PAUSE_CHAIN`
  FIRST, then stop the trainer: otherwise a kill within 45 min of launch is read as an early failure and re-run.
* A gc0 fold that dies after 45+ min: logged, chain moves on (same as the old chain); the partial dir stays.

## 6. Risks and notes

* GPU budget for OTHER workflows changes during folds 2-4: the fold trainer holds ~9.1 GB instead of ~3.0 GB, so
  free memory drops from ~12.8 GB to ~7.0 GB. Any other GPU job in that window must cap itself at <= 4.5 GiB
  (set_per_process_memory_fraction ~0.28) and gate on free memory; gates/caps that assume "trainer = 3 GB" are wrong
  after the switch. The trainer's caching allocator keeps its peak pool, so a late-arriving job is far more likely to
  OOM itself than the trainer (GUESS), but do not test that.
* Sharing the GPU slows the fold roughly 2x for the duration (MEASURED today with gc1: 0.36 -> 0.6-0.8 s/it).
* The speed ratio was measured on a shared GPU; the alone ratio may differ a little (GUESS range .75-.85).
* Long-run memory growth from fragmentation over 24 epochs without a cap: GUESS < 0.5 GB; headroom is ~7 GB.
* Artifacts of this workstream: ~/fast_build/code2 (trainer), ~/fast_build/code_ref (pristine copy used as the
  reference), ~/fast_build/{parity,tiny} (test ckpts, ~2.3 GB, deletable), scripts parity_unit.py, compare_tiny.py,
  worker_load_check.py, speed_ab.py, loader_rate.py, cap_run.py, fbcommon.py, gpu_queue.sh, auto_folds_fast.sh (copy
  installed at ~/auto_folds_fast.sh).

## 7. Adversarial verification (10-05 ~10:15 IST, separate verifier)

STATUS: VERIFIED (verdict GO_WITH_FIXES; the two script fixes below are applied and dry-run tested)

MEASURED by the verifier (own scripts in ~/fbverify, not the builder's harness):
* Independent parity (`~/fbverify/gc_parity2.py`, GPU under `flock ~/jump2/gpu.lock`, 6 GiB cap, after the TTA bench
  exited): KneeFleetNetTTR from code2, grad_ckpt True vs False, trainer-mode cudnn.benchmark, fp16 autocast + GradScaler
  + clip + AdamW + EMA, drop_path .2 / dropout .3 active, bs 1, one view absent, 16 of 24 slices via pos, random uint8
  input (seed 777), 5 steps: losses equal, raw and EMA state_dicts bitwise equal, CPU and CUDA global RNG states equal
  afterwards (DataLoader base seeds unchanged). Non-vacuous: a pre-hook on stages[1].blocks[0] fired 4x per step with
  grad_ckpt (forward + recompute, 2 chunks) vs 2x without. Peak reserved 2128 vs 4658 MiB at bs 1.
* Ckpt args (`~/fbverify/args_cmp.py`, CPU): J1 vs fold-1 ep6 (old trainer) differ only in fold; builder's tiny ref vs
  tiny gc0 differ only in grad_ckpt (absent vs False); state_dict keys, shapes and dtypes equal to J1; model_sha
  f141aeed136dfc01 in all. Consumers (fleet_worker loader, pack_fleet_ds, build_k2 preflight, k3_pick_ttr) read named
  keys only; pack's whole-args signature is duplicate detection and is per fold anyway.
* Chain command line: old and new trainer invocations expand to the identical string except the trailing
  `--grad_ckpt 0` (same labels, folds5, gold74 exclude, eval labels, seed, out dirs). code2 differs from
  ~/bigjump_build/code only in fleet_train.py; code_ref == bigjump fleet_train.py (md5 4079332e...).
* Old-chain signals: bash 311059 has no controlling TTY (setsid), ignores HUP/INT/QUIT, does not catch TERM; the
  python 311068 ignores HUP. `kill 311059` therefore ends only the bash (variant B is safe). The anchored pgrep returns
  exactly 311059 and cannot match `bash -c ...` wrappers or auto_folds_fast.sh.

Fixes applied (both copies ~/auto_folds_fast.sh and ~/fast_build/auto_folds_fast.sh, md5 16f7ba2c...; previous version
kept as ~/fast_build/auto_folds_fast.sh.pre_verify):
1. The early-failure fallback used to wait only for "no GPU python": it ignored PAUSE_CHAIN and the old-chain guard, so
   an operator who killed a fold within 45 min to free the GPU would see it relaunched at once. It now waits on all 3
   guards and skips if the out dir reappeared.
2. `gpu_python` returned 0 ("free") when nvidia-smi errored; it now returns 99 ("busy").
Stub dry-run of the fixed script in a fake HOME (MEASURED): waits on PAUSE, GPU python, old chain and an nvidia-smi
error in turn; a gc0 failure is moved aside, the fallback waits while PAUSE_CHAIN is present and runs --grad_ckpt 1
after release; existing fold dir skipped; next fold launched with --grad_ckpt 0; all calls from cwd code2.

Still open (operator / orchestrator):
* Every workflow's GPU rule still says "the running trainer (3 GB)". After the switch it is ~9.1 GB: side jobs must
  cap at <= 4.5 GiB, and the one at real risk is a side job that starts in the first ~2 min after a fold launch (the
  trainer has not reached its peak yet; the fallback then re-runs the fold with grad_ckpt 1 after the side job ends).
* A gc0 fold that dies after 45 min is not retried (same as the old chain): the partial dir blocks a re-run until it
  is renamed.
* ~/fast_build/parity and ~/fast_build/tiny hold fake f2_s1043_* ckpts with a valid model_sha: delete them (or never
  point a packer at them) before any fold-2 packing.
