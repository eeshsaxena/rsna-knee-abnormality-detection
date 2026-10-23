# RUNBOOK: TTR mix50 folds 3 + 4 on Kaggle (rsna-train-ttr-f34)

Goal: train TTR mix50 folds 3 and 4 (seed 1043) on Kaggle T4 x2, one fold per GPU, in parallel, with the exact box
recipe, so the 5-fold set and the full own-OOF exist tonight instead of ~03:00 tomorrow. The V100 box keeps folds 1-2.

```
SP = C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad
KT = $SP\kaggle_train
  ds\rsna-knee-fleet-train\          dataset eeshsaxena/rsna-knee-fleet-train (16 entries incl. dataset-metadata.json, ~119 MB)
  kernel\rsna-train-ttr-f34\         kernel eeshsaxena/rsna-train-ttr-f34 (notebook + kernel-metadata.json)
  verify_outputs.py                  checks downloaded outputs against the kernel's run_receipt.json
  nb_src\ build.py dry\              notebook source cells, builder, box dry test (not uploaded)
```

## What the notebook does (and what makes it abort BEFORE any training)

| step | what | aborts when |
|---|---|---|
| 1 inputs | finds the code dataset and the 12 store tars under `/kaggle/input` (any of `/kaggle/input/<slug>/`, `/kaggle/input/{datasets,notebooks}/<owner>/<slug>/`); sha256 of every dataset file vs `kt_manifest.json`; the 6 code files == box `~/bigjump_build/code`; the 4 csvs == box `~/rsna_train` (a Kaggle CRLF rewrite is tolerated: LF copy must match, warning) | dataset not attached / changed, a tar part missing |
| 2 env | versions, GPUs, CPUs, RAM, disk; timm < 1.0 or missing: bundled timm 1.0.29 wheel, then PyPI | fewer than 2 GPUs, GPU memory already in use |
| 3 store | free disk >= 20 GB at the roomiest of `/tmp`, `/kaggle/tmp`, `/kaggle/temp`, streams the 12 tars to `<that>/fleet_store` hashing each tar (== prep manifests), then: 4407 npz, names sha == box `~/fleet_store`, 24 sampled npz byte-identical to the box (2 per tar), every labelled study present | any mismatch |
| 4 parity | `pixel_sha 6f49c74fc48f3345`, `tensor_sha 08ec55bc155be309`, `select_sha 91bb8a65594a25bd`, `jpeg_fp 34819a6207119a3b`, `model_sha f141aeed136dfc01` (all == the J1 ckpt args); the J1 ckpt's 4-study decode probe re-decoded here; KneeFleetNetTTR built here has the 199 state_dict keys/shapes/dtypes of the J1 ckpt and its 180 backbone tensors == the box's own `convnext_tiny.fb_in22k_ft_in1k` safetensors (bundled, loaded offline from an HF cache: no download); fold counts == box (fold 3 and 4: train 3467, val 866, held 74, 1733 it/epoch); one-thread CPU cost of the heavy train path vs the box (0.46 s/study) | any mismatch |
| 5 probe | both trainers at once with the real recipe (fold 3 GPU0, fold 4 GPU1, 3 workers each) until each prints `it 400` (timeout 25 min), then killed; steady s/it = window it 200 -> 400; CPU busy %, GPU util %, train studies/s, verdict CPU- or GPU-bound. The trainers' own first lines (contract, counts, model_sha) are checked against the box | a trainer exits early (OOM, error: log tail printed), header mismatch |
| 6 train | `E = min(24, floor((10.3 h - elapsed - 2 min) / (1733 x s/it x 1.21)))` (1.21 = eval share measured in the J1 log: 10.1 -> 12.3 min/epoch); `--save_epochs (E-1)-E`; WARNING if E < 16; launch, watch every 30 s, status line every 15 min; hard wall at 11.4 h since kernel start: waits (max 2 min) until no ckpt/OOF write is in flight (the trainer deletes the old best before writing the new one in place), then SIGTERM to each trainer's process group, SIGKILL after 60 s; then receipt (every output file: bytes, sha256, ckpt loads + keys == box) | trainer header mismatch on the real run |

Command per fold (box recipe, only E / save_epochs / workers / paths differ):
`python fleet_train.py --model ttr --aug heavy --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs E
--seed 1043 --save_epochs (E-1)-E --workers 3 --store /tmp/fleet_store --labels .../labels_mix50_logit.csv --folds
.../folds5.csv --exclude .../gold74_eval_set.csv --eval_labels .../llm_labels_v4_blend.csv --gold .../gold74_eval_set.csv
--fold F --out /kaggle/working/fF_s1043_ttr_mix50` (log: `/kaggle/working/fF_s1043_ttr_mix50.log`).

Outputs (`/kaggle/working`): `f3_s1043_ttr_mix50/` and `f4_s1043_ttr_mix50/` (best raw + ema ckpt `fF_s1043_{raw,ema}_ttr224-320x24_cv####_best.pt`, `oof_fF_s1043_{raw,ema}.csv`, `epochs/` = last 2 epochs' raw+ema ckpt + OOF), their `.log` files, `run_receipt.json`, `probe/probe_f{3,4}.log`. About 1.4 GB in total.

## 1. Create the dataset (PowerShell, ~5 min)

```powershell
$UP = "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\kaggle_train\ds\rsna-knee-fleet-train"
# Windows kaggle CLI staging-dir bug: pre-create %TEMP%\.kaggle\uploads\<path with ':' -> '_'> first
New-Item -ItemType Directory -Force -Path (Join-Path "$env:TEMP\.kaggle\uploads" ($UP -replace ':', '_')) | Out-Null
(Get-ChildItem $UP).Count                                   # 16
kaggle datasets create -p $UP                               # private by default
kaggle datasets status eeshsaxena/rsna-knee-fleet-train     # repeat until: ready
kaggle datasets files eeshsaxena/rsna-knee-fleet-train      # 15 files incl. convnext_tiny...safetensors (114 MB) and timm-1.0.29-py3-none-any.whl.bin
```

If the CLI prints `No such file or directory` for a path under `.kaggle\uploads`, create exactly that folder and re-run.
Do not push the kernel before `ready` (a commit that mounts nothing aborts in step 1: `code dataset ... found []`).

## 2. Push the kernel (PowerShell)

```powershell
kaggle kernels push -p "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\kaggle_train\kernel\rsna-train-ttr-f34"
```

`kernel-metadata.json`: private, GPU on, `machine_shape NvidiaTeslaT4` (= T4 x2), internet ON (only for fallbacks:
pretrained weights and timm are bundled), `dataset_sources [eeshsaxena/rsna-knee-fleet-train]`, `kernel_sources
[eeshsaxena/rsna-fleet-prep-s0, eeshsaxena/rsna-fleet-prep-s1]`, no competition attached. Write down the time the
status turns `running` (= T0 below). Uses ~10.5-11.5 h of the weekly GPU quota: BEFORE pushing, check the remaining weekly GPU quota on kaggle.com (Settings / quota bar) is >= 12 h. If the quota runs out mid-run Kaggle stops the session without the receipt/finalize step and the outputs may be lost. If the push fails on the 2 concurrent
GPU batch sessions limit, wait for a running GPU kernel to finish and push again.

## 3. Stop the box chain from ALSO training folds 3-4 (Git Bash, right after the push)

`~/auto_folds.sh` skips a fold whose out dir exists. Create EMPTY placeholders now (must exist before fold 2 ends; fold 1
started 08:25 and runs at 13-21 min/epoch today (J1 alone: 12.3), so fold 2 ends roughly 19:00-22:00 IST; do not wait):

```bash
B=training@192.168.10.30
ssh $B 'mkdir ~/fleet_runs/f3_s1043_ttr_mix50 ~/fleet_runs/f4_s1043_ttr_mix50 && ls -ld ~/fleet_runs/f[34]_s1043_ttr_mix50 && tail -n 2 ~/fleet_runs/auto_folds.log'
```

After fold 2 the chain logs `fold 3: ... exists, skipping`, `fold 4: ... exists, skipping`, `chain done`: the V100 is
then idle, so queue the next box job before fold 2 ends.

**K4 collector rule:** pick a run only if its log ends with the trainer's closing line `{"raw": ..., "ema": ...}`.
The placeholders have no `.log`; a Kaggle run cut by the watchdog has no closing line either (the receipt says
`PARTIAL`): neither may be picked automatically. Check: `tail -n 1 ~/fleet_runs/fF_s1043_ttr_mix50.log`.

## 4. Monitor

```powershell
kaggle kernels status eeshsaxena/rsna-train-ttr-f34        # queued -> running -> complete | error
```

A batch run shows nothing else until it ends (log and outputs are downloadable only afterwards). Timeline from T0:
store + checks ~5-15 min, probe ~10-25 min, then training. It plans to END by T0 + 10.3 h (earlier if 24 epochs fit:
that needs s/it <= ~0.6 per trainer; the expected T4 x2 + 4 vCPU rate is ~0.8-1.2 s/it, i.e. E ~15-22). Hard wall
T0 + 11.4 h, then ~5-10 min of output upload. Example: running at 10:45 IST -> complete ~21:00-21:15 IST, never after
~22:20 IST.

`error` within the first ~40 min = an abort before training (step 5 tells why); `error` later is unexpected.
`complete` = read the receipt (`status` DONE or PARTIAL, `plan.E`, `warnings`).

## 5. Fetch the outputs (PowerShell)

```powershell
$OUT = "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\kaggle_train\out_v1"
kaggle kernels output eeshsaxena/rsna-train-ttr-f34 -p $OUT --page-size 200
# smaller (skips the 8 per-epoch ckpt snapshots, ~0.9 GB; the CLI applies re.search to paths like f3_s1043_ttr_mix50/epochs/x.pt):
# kaggle kernels output eeshsaxena/rsna-train-ttr-f34 -p $OUT --page-size 200 --file-pattern '^(?!.*epochs/.*\.pt$)'
python "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\kaggle_train\verify_outputs.py" $OUT
```

`verify_outputs.py` prints status, E, s/it, CPU/GPU utilisation, warnings, per fold `complete / ended / best val`,
and checks every file against the receipt's sha256 (exit 0 = DONE and all present files match; 2 = PARTIAL or
ABORTED; 1 = a file differs, or a ckpt the kernel could not load (UNUSABLE)). The notebook log is `$OUT\rsna-train-ttr-f34.log`.

Decide: DONE with E >= 16 -> use as fold 3/4 members. E < 16 (warning) or PARTIAL -> operator's call (valid best-so-far
ckpts, but a shorter cosine than folds 0-2's 24 epochs). Reference: J1 fold 0 best val raw .8836 / ema .8824.

## 6. Copy to the box (Git Bash)

```bash
KT=/c/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad/kaggle_train
OUT=$KT/out_v1
B=training@192.168.10.30
for F in 3 4; do R=f${F}_s1043_ttr_mix50
  ssh $B "test -d ~/fleet_runs/$R && [ -z \"\$(ls -A ~/fleet_runs/$R)\" ] && [ ! -e ~/fleet_runs/$R.log ]" \
    || { echo "STOP: ~/fleet_runs/$R is not an empty placeholder"; break; }
  scp -r "$OUT/$R/"* $B:fleet_runs/$R/
  scp "$OUT/$R.log" $B:fleet_runs/$R.log
done
scp "$OUT/run_receipt.json" $B:fleet_runs/kaggle_f34_run_receipt.json
scp "$KT/verify_outputs.py" $B:kt_test/verify_outputs.py
ssh $B '~/miniconda3/envs/rsna/bin/python ~/kt_test/verify_outputs.py ~/fleet_runs --receipt ~/fleet_runs/kaggle_f34_run_receipt.json; for F in 3 4; do tail -n 1 ~/fleet_runs/f${F}_s1043_ttr_mix50.log; done'
```

Result on the box: `~/fleet_runs/fF_s1043_ttr_mix50/` (same file names as the box runs) and
`~/fleet_runs/fF_s1043_ttr_mix50.log` next to it, so the existing pickers and K4 tools read folds 0-4 the same way.
Checkpoint `args` record `epochs E` and `save_epochs (E-1)-E` (box folds: 24 and 6-24).

## 7. If the Kaggle run fails or is rejected

- Chain has NOT reached fold 3 yet (`tail ~/fleet_runs/auto_folds.log` still shows fold 1 or 2): remove the empty
  placeholders and the chain trains folds 3-4 itself after fold 2:
  `ssh $B 'rmdir ~/fleet_runs/f3_s1043_ttr_mix50 ~/fleet_runs/f4_s1043_ttr_mix50'` (`rmdir` refuses non-empty dirs).
- Chain already logged `chain done`: remove the placeholders as above, then re-run the chain (it skips folds 1-2
  because their dirs exist, waits for a free GPU, trains 3 then 4):
  `ssh $B 'nohup bash ~/auto_folds.sh > /dev/null 2>&1 < /dev/null &'`

Abort reasons (receipt `status`, also the last `[kt ...] ABORT` line of the notebook log):

| status contains | meaning / fix |
|---|---|
| `code dataset ... found []` | dataset not ready or not attached: wait for `ready`, push again |
| `prep tar parts: missing` | prep kernel outputs not mounted: check `kernel_sources`, both prep kernels still have their outputs |
| `only X GB free` | the largest of /tmp, /kaggle/tmp, /kaggle/temp has < 20 GB. The run needs ~13 GB (store 11.6 + outputs 1.4): if X >= 14, set `MIN_FREE_GB=_env('KT_MIN_FREE_GB', 14)` in notebook cell 1 (`nb_src/c01_lib.py`, then `python build.py`) and push again |
| `tar sha256 mismatch`, `store names differ`, `store files differ` | a prep output changed or a corrupt mount: push again; if it repeats, the prep kernels were re-run (do not train on a different store) |
| `parity with the box failed` | `jpeg_fp`: Kaggle image cv2 changed (decode would differ from the box: stop); others: code/data drift |
| `model check failed` | timm on the image builds a different ConvNeXt: the log has the key diff |
| `probe: trainer exited early` | the trainer crashed (OOM on T4, missing package): log tails printed above it |
| `trainer header differs` | the trainer itself printed another contract / counts / model_sha |

## 8. Dry test (done on the box CPU, 2026-10-05)

`dry/kt_dry.py` runs the notebook's code cells unchanged as one script under a fake root (`KT_ROOT`, a dummy trainer
with the real log format; no GPU, read-only on `~/fleet_store`, `~/fleet_runs`, `~/rsna_train`): unit tests (s/it
parser on the real J1 log + a fake log, header checks, E arithmetic, process-group SIGTERM/SIGKILL, tar member guard),
A full flow on a 2-tar subset of 10 box store files in both mount layouts (real decode probe, real model + pretrained
check), B watchdog with a ~60 s wall (fold 3 SIGTERM, fold 4 SIGKILL, best-so-far kept, E < 16 warning), C corrupted
tar, D wrong tensor_sha, E wrong val count (all abort before training), F CRLF csv + missing wheel (warnings, run OK).

Review pass (2026-10-05 ~10:50 IST): added (a) a bounded quiet-wait before the hard-wall SIGTERM (the trainer deletes the
old best ckpt and then writes the new one in place, so a kill inside a save would leave no usable best ckpt), (b) tolerance
for the code dataset being visible at 2 mount paths with an identical manifest (it used to abort), (c) /dev/shm free
space in the env report (warning if < 2 GB), (d) verify_outputs.py flags a ckpt the kernel could not load (UNUSABLE,
exit 1). Re-run on the box: `dry_run_v3.log` 57/57, `dry_run_v3_g.log` 6/6, plus `smoke_real.log` 7/7: the REAL
fleet_train.py launched with the notebook's own recipe_args() (+ `--device cpu --min_cover 0 --time_epochs 1`, a
14-file store, the bundled HF cache offline) passes header_problems(), and its progress, eval and closing lines parse.
It writes the best and epochs/ ckpts with the box keys, and OOF rows equal the val ids.
