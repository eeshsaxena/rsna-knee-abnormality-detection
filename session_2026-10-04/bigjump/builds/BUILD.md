# BIG JUMP builds B1-B5 (BIG_JUMP_PLAN sec 11), 10-04: reviewed and fixed

Built 2026-10-04 04:50-06:00 IST by Claude. Review fixes applied 06:10-06:50 (`FIX_LOG.md`; pre-fix version
`BUILD_pre_fix.md`). Nothing was submitted, pushed, versioned or posted. No running process was touched and no GPU
was used. Box writes: `~/bigjump_build/` (code, tests, outputs) and `~/rsna_train/labels_gated75_logit.csv` (B3).
Every launch and every `kill` below is for Eesh to run.

`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`,
`BLD` = `SP/bigjump/builds`, `B` = `training@192.168.10.30`, `PYB` = `~/miniconda3/envs/rsna/bin/python` (box: torch
2.5.1+cu121, timm 1.0.29, cv2 4.14.0; ConvNeXt-T weights in the HF cache).

## 0. GO / NO-GO for TTR-A fold-0

**GO**, if two steps come first:
1. **Stop `~/auto_after_fleet.sh` before f0_s42 mix50 ends** (sec 3.0). Otherwise, when mix50 ends it runs the parent
   gold58, then `f2_preflight` and an EV2S s142 launch in lane 1, which collides with TTR.
2. **Run the parent gold58 by hand at mix50's closing line** (sec 3.1, ~20-25 min).

Then launch TTR with the guarded command (sec 3.3). The trigger is the parent's exit (~10:00-10:15), not 09:45 on the
clock.

**Alternative (TTR first):** if mix50 ends late, launch TTR at mix50's end, ~09:45, with the same guarded command. Then
run the parent at v4's end (~10:48). TTR plus the parent fit in 16 GB (TTR ~3-5 + parent <= 8.9). The cost is a gold58
parent read ~1 h later. G-T0 stays clean either way.

The TTR seed is now **1042** (trainer BJ-5: TTR owns seeds 1000-1999) and the output is `~/fleet_runs/f0_s1042_ttr`.
The labels are pure steven_v4, as the plan says for the recipe test. TTR `model_sha` is **`f141aeed136dfc01`**.

## 1. What was built

| Item | File (`BLD/code/`, staged byte-identical at `~/bigjump_build/code/`) | What |
|---|---|---|
| B1 | `fleet_model_ttr.py` | `KneeFleetNetTTR` = `KneeFleetNet` + one change in `encode()`: uint8 -> float [0,1] -> `F.interpolate(224, bilinear, antialias)` -> ImageNet norm. fp32 resize, pass-through if already 224. **TTR-6**: persistent int64 buffer `ttr_size_t`, so a TTR state_dict can no longer strict-load into a plain KneeFleetNet. `fleet_model.py` untouched. **model_sha `f141aeed136dfc01`** (family 1 stays `fa35880fca963652`, family 2 `ba3e1f85e38128bf`) |
| B1 | `fleet_worker.py` | K2 worker + W-TTR: `model_mod == 'fleet_model_ttr'` builds `KneeFleetNetTTR(ttr_size=args)`. The sha is checked like family 2, and `ttr_size_t` must equal args `ttr_size`. Family-1/2 path unchanged (predictions == K2 worker, max diff 0) |
| B2 | `fleet_train.py` | fleet2 trainer + opt-in BJ-1..5: `--model ttr [--ttr_size 224]`, `--aug heavy`, `--drop_path`, `--dropout`, `--save_epochs a-b`; slot subset (`--train_slices`) taken before the per-view aug in the heavy path; **BJ-5 `--model ttr` needs `--seed` 1000-1999**. **No flag = byte-identical run** |
| B2 | `fleet_aug.py` | the heavy TRAIN path (imported only under `--aug heavy`). Coins first: mirror .5, 16-of-24 slots, view drop .2. Then per view: decode, slice jitter U(-1,1), mirror-remapped slot subset, affine (rot +-15, scale .85-1.15, shift +-8 %), gamma .75-1.35 / gain .85-1.15 / bias +-15, noise sigma U(0,6) p .3, CoarseDropout 1-3 holes of 8-16 % (same holes every slice) p .3 |
| B4 | `pack_fleet_ds.py` | K2 pack tool + P-TTR (ships `fleet_model_ttr.py`) + **P-TAG `--tag K1,K2,...`** (one entry per member; `=` keeps the key; KEY = the member's own `f{fold}_s{seed}` + `[A-Za-z0-9]*`) + P-RUN (identical ckpt args = one run, refused even if re-keyed). Manifest `sources` = original name, run folder, sha256[:16], model_mod, model_sha. **FIX-5: no local paths.** The worker reads only `checkpoints` |
| B5 | `tta_bench.py` | G-TTA bench through the worker path. Passes A / M / P(+HALF) / N(-HALF) / B(mirror + HALF), with **HALF = 0.5 - 1e-3 (FIX-1)**. Variants are built from the passes that ran (none, mirror, offset, A+B, A+B lesion-max, all5, raw+EMA). Macro AUC vs v4 on fold-0 val + gold58; paired bootstrap; gate line; `--oof` parity; `--fp_check`. **GPU guard**: refuses below `--min_free_gb` 4, caps itself at `--max_gpu_gb` 3. `--max_minutes` cut |
| gate | `gt2_read.py` | G-T2 read: exact OOF macro AUC (best raw/EMA) TTR vs f0_s42_v4, per-label floor, gold58 direction; PASS / MIDDLE / KILL; exit 3 while a run is unfinished |
| G-P0 | `aug_dump.py` | PNG grid of heavy-aug draws for one study, mirror check, light vs heavy loader cost |
| opt. | `ttr_preflight.py` | optional 2-4 min GPU check of TTR memory + s/step. **FIX-6**: default `--cases ttr`. It refuses below need + 1.5 GB headroom (6.5 GB free for TTR, 9 GB if `f1` is added) and caps itself at free - 1.5 GB. Writes nothing |
| copies | `fleet_common.py`, `fleet_model.py`, `fleet_model_f2.py` | md5 == `~/fleet`, `~/fleet2_code`, `k2/code` |

Also: `BLD/FIX_LOG.md` (review findings, verdicts, changes), `BLD/PATCH.diff` (trainer vs fleet2, worker vs K2, pack
vs K2, model vs pre-fix), `BLD/tests/` (laptop suite `run_all.sh`, `transcript.txt`), `BLD/box/` (box suite
`run_box_tests.sh`, transcript `BLD/tests/out/box_run2.txt`), aug PNGs `BLD/tests/out/box_aug/`.

md5 (laptop == box `~/bigjump_build/code`, checked after staging):

| file | md5 |
|---|---|
| fleet_train.py | `4079332e503e9c40ca7c74b62430feb6` |
| fleet_model_ttr.py | `bcdf7d153f339962b97d2736d8baed08` |
| fleet_aug.py | `4e7311c2f4c6f94acfb1a1abe1bfc12f` (unchanged) |
| fleet_worker.py | `f27d3f1cd305db5602154a18d85e3ede` |
| pack_fleet_ds.py | `bc65dea9510a71ede83f3bffc123c7ea` |
| tta_bench.py | `176851b1d241b54a283a1e85e2669e30` |
| gt2_read.py | `e0d21e012acf77082a26c0e9125a5e5b` |
| ttr_preflight.py | `f8ddcd120ca6ee21ab13fee9d9fba7cc` |
| fleet_common.py | `658ab8103acdf6e4aec029a0f6ee5959` (contract, unchanged) |
| fleet_model.py | `897f83b272fbb144a4d3deab773d9064` (contract, unchanged) |

## 2. Test results (re-run after the fixes)

All tests were re-run after the fixes, 06:24-06:47. Every test passes (laptop 7/7 rc 0; box BOXCHECK, BOXPARITY and
BOXGP0 PASS).

**Laptop CPU** (`BLD/tests/run_all.sh`; 64 px synthetic store, convnext_atto, cv2 PIL shim; ~11 min; transcript
`BLD/tests/transcript.txt`, pre-fix copy `BLD/tests_pre_fix/transcript.txt`):

- **t_b1_model**
  - Contract files are identical in all 4 trees.
  - The model_sha registry is identical in worker, pack and tta_bench: f1 `fa35880fca963652`, f2 `ba3e1f85e38128bf`,
    **TTR `f141aeed136dfc01`**.
  - TTR with ttr_size == input size == KneeFleetNet: eval, train logits and 144 grads all max diff 0; the state_dict
    keys are the same plus `ttr_size_t`.
  - encode == manual interpolate + norm + cnn (diff 0). A 320 px input reaches the backbone at 224. Antialias is
    active.
  - **(e) TTR-6**: `ttr_size_t` stays int64 224 in raw and EMA after 5 real `AveragedModel` updates. Strict load into
    KneeFleetNet is refused; TTR -> TTR strict load works.
- **t_b2_default_parity**: with no opt-in flag the run is bit-identical to round3/fleet (`~/fleet`) AND fleet2
  (`~/fleet2_code`). Cases: fold-0 2 ep, all-data stop_epoch, T6 `--train_slices 16`, `--f2 1` (resnet18). Compared:
  ckpt names, all state_dict tensors `torch.equal`, ckpt args, cv/epoch, OOF bytes and metric log lines.
- **t_b2_heavy_equiv**:
  - The heavy path with pixel aug off == default `mirror_views(full)[:, pos]` on 480 cases, K 16/24.
  - View drop forced on 120 cases. Sagittal mirror reverses slot depth and channels.
  - Aug is deterministic; holes are identical on all 48 slice-channels.
- **t_b2_ttr_train**:
  - TTR fold-0 3 ep and all-data 3 ep at **seed 1042**: names `f0_s1042_{raw,ema}_ttr224-64x24_*` and
    `f-1_s1042_*`, plus `epochs/` snapshots and per-epoch OOFs. Per-epoch ckpt cv == log; top-level best == its
    epoch snapshot; args right (sha `f141aeed136dfc01`).
  - Guards refuse `--model ttr --f2 1`, `--save_epochs 3-1` and **`--model ttr --seed 42` (BJ-5)**.
  - **TTR-6** on real trainer ckpts: int64 224, strict load into KneeFleetNet refused.
  - Pack `--tag f0_s1042ttr,=,=,=`; the manifest `sources` have the FIX-5 keys only.
  - The bigjump worker serves 2 TTR + 2 F1 members; F1 predictions == K2 worker (max diff 0). The K2 worker refuses a
    TTR member.
- **t_b4_pack_tag**:
  - Same-name pair: the K2 pack tool and the bigjump pack without `--tag` refuse it; `--tag =,f0_s42v4` packs both
    (bytes == sources).
  - **FIX-5**: `sources` = original / run_dir / sha256[:16] (== the bytes) / model_mod / model_sha, with no ':' or '/'.
  - All 6 bad re-keys are refused.
  - **K2 notebook k2_check PASS 22/22 on 24 studies and on the 3-study commit cut, n_keep [2, 2]**. K2 worker swap:
    fleetsolo byte-identical. Real pair: `build_k2.py ... =f0_s42v4` gives 2 members, 225.3 MB / 12 files (local only).
- **t_b5_tta_bench**:
  - Pass A == bigjump worker P (max diff 0, TTR and F1 members).
  - Pass B == independent `mirror_views(sample_triplets(+HALF))` (0).
  - All variants + gate.
  - Trainer OOF vs pass A at chunk 32 and 48: 0.
  - **(6) HALF**: n=24 P/N centres 0..23 (exact .5 gives `0,2,2,4,4,..`); every other n 1-32 is the same as .5,
    except the N pass's last slot on even n.
  - **(7) `--passes A,B`**: none / A+B / A+B lesion-max + raw+EMA A+B + 2 gate lines, all == the 5-pass run.
  - **(8)** a `--max_minutes` cut does not crash and is flagged PARTIAL.
- **t_gt2_saveep**:
  - `--save_epochs` does not change training (tensors, cv, OOF bytes; args differ only in `save_epochs`).
  - gt2_read score == recomputed; verdict == rule; exit 3 PROVISIONAL without a closing line.
- **ttr_preflight CPU smoke** (real parity npz, convnext_atto, `--no_pretrained --bs 1 --steps 3`): default case `ttr`
  only, finite losses. An unknown case is refused. The GPU guard lines were not executed (no GPU in this pass).

**Box CPU** (`bash ~/bigjump_build/box/run_box_tests.sh 3`, `CUDA_VISIBLE_DEVICES=`, nice 10, real cv2 4.14;
06:36-06:46; transcript `BLD/tests/out/box_run2.txt`). All 16 staged files have md5 == laptop.

- **box_check PASS**:
  - Contract files == `~/fleet` and `~/fleet2_code`.
  - model_sha in the box env: f1 `fa35880fca963652`, f2 `ba3e1f85e38128bf`, **TTR `f141aeed136dfc01`** (pinned).
  - **B3**: `~/rsna_train/labels_gated75_logit.csv` md5 `a813b986aa09b6fc4736ff493fbaa918` (== laptop); 4407 ids;
    finite, in [0,1]; report-ADDRESSED cells == steven_v4 exactly (38,427 cells); report-UNK 27.3 %, all changed
    (mean |diff| .135); fold-0 train 3466 / val 867.
- **box_parity PASS**: the 7 default-path comparisons are bit-identical in the box env. The TTR path runs there (seed
  1042, sha `f141aeed136dfc01`).
- **box_gp0 (G-P0) PASS**:
  - Setup: 4 real K2 parity studies (`~/fleet_store`, steven_v4, 2 train / 2 val), REAL pretrained ConvNeXt-T, the
    exact TTR flags + `--seed 1042`, 1 epoch CPU fp32 (50 s).
  - Ckpt `f0_s1042_{raw,ema}_ttr224-320x24_*` with sha `f141aeed136dfc01` and `ttr_size_t` 224.
  - **Trainer path vs worker path (tta_bench pass A): max |diff| 0.0 for raw and EMA at chunk 32 AND 48.**
  - Mirror check PASS; aug PNGs in `BLD/tests/out/box_aug/`.
  - Real-data tta_bench smoke on the interim family-1 member (3 val + 3 gold58 x 5 passes): pass A vs its GPU fp16
    trainer OOF max |diff| 2.1e-4 (fp16 noise).
- **gt2_read on the LIVE logs** (pre-fix run, format dry run): parses both real logs; OOF macro AUC == the log
  values; exit 3 PROVISIONAL. gt2_read changed only in comments since then.

## 3. TTR-A fold-0 launch (lane 1, plan sec 3.1; labels = pure steven_v4)

### 3.0 Free lane 1 first (Eesh, NOW, at the latest before mix50 ends ~09:05-09:35)

The armed scheduler is `~/auto_after_fleet.sh` (PID 241241 per the review, running since 01:54). Its loop waits while
both fleet jobs are alive, and then runs, in order:
1. the parent gold58 (`--min-free-gb 8`);
2. `~/fleet2_code/f2_preflight.py`;
3. `nohup fleet_train.py --f2 1 ... --seed 142`, which is EV2S in lane 1.

Started next to TTR and v4, step 3 is ~16-18 GB of demand on a 16 GB card, so it can push f0_s42_v4 (the G-T2
reference and a K2 member) into CUDA OOM. Plan sec 5 has EV2S on 10-05. Claude did not read this script (box access
in this pass is limited to copy / md5 / CPU tests), so check it first.

(a) Look. This is read-only: it shows the PID's command, the parent command, the log tail and GPU memory.

```bash
ssh -o BatchMode=yes training@192.168.10.30 'ps -o pid,etime,args -p 241241 | cut -c1-160; grep -n "run_parent\|f2_preflight\|fleet_train" ~/auto_after_fleet.sh | cut -c1-260; tail -n 5 ~/auto_after_fleet.log; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader' </dev/null
```

(b) Stop it. The command kills PID 241241 only if its command line contains `auto_after_fleet`; otherwise it kills
nothing and lists any matching process. Never use `pkill -f`. Never edit the running script: bash reads it while it
executes.

```bash
ssh -o BatchMode=yes training@192.168.10.30 'if ps -o args= -p 241241 | grep -q auto_after_fleet; then kill 241241; sleep 2; if ps -p 241241 >/dev/null; then echo STILL RUNNING; else echo "[sched] stopped by Eesh $(date) (TTR-A takes lane 1; EV2S s142 moves to 10-05)" | tee -a ~/auto_after_fleet.log; fi; else echo "241241 is not auto_after_fleet.sh: nothing killed"; ps -ef | grep -v grep | grep auto_after_fleet; fi' </dev/null
```

If the log shows the scheduler already moved on:
- If the parent is running, stopping the scheduler leaves the parent running to completion, which is wanted. Skip 3.1.
- If `f2_preflight` or the EV2S job has started, stop that process by its own PID (`kill <pid>`) before 3.3. Your call.

### 3.1 Parent gold58 by hand (at mix50's closing `{"raw": .., "ema": ..}` line)

v4 alone leaves ~8.9 GB free, and the parent needs 8. The command below is the scheduler's own, as quoted by the
review. Compare it with the `grep` output of 3.0 (a) before running it.

```bash
ssh -o BatchMode=yes training@192.168.10.30 'tail -n 1 ~/fleet_runs/f0_s42.log; nvidia-smi --query-gpu=memory.free --format=csv,noheader' </dev/null
ssh -o BatchMode=yes training@192.168.10.30 'nohup ~/miniconda3/envs/kmimic/bin/python ~/kmimic/run_parent.py --ids ~/kmimic/ids/gold58.csv --out ~/kmimic/runs/gold58_gpu --device-mode gpu --gpus 0 --min-free-gb 8 --quiet >> ~/auto_after_fleet.log 2>&1 < /dev/null & echo PID $!' </dev/null
```

It takes ~20-25 min. Launch TTR only after this PID has exited (`ps -p <PID>`).

### 3.2 Pre-checks (just before the launch)

```bash
ssh -o BatchMode=yes training@192.168.10.30 'cd ~/bigjump_build/code && md5sum fleet_train.py fleet_model_ttr.py fleet_aug.py fleet_model.py fleet_common.py; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader' </dev/null
#   expect 4079332e..., bcdf7d15..., 4e7311c2..., 897f83b2..., 658ab810...
# optional (2-4 GPU-min, writes nothing, TTR case only; refuses itself below 6.5 GB free, caps itself at free - 1.5 GB):
ssh -o BatchMode=yes training@192.168.10.30 'cd ~/bigjump_build/code && ~/miniconda3/envs/rsna/bin/python ttr_preflight.py --store ~/fleet_store' </dev/null
```

The preflight is optional: G-T0 plus `nvidia-smi` after epoch 1 measure the same thing. Add `--cases f1,ttr` only
when >= 9 GB is free.

### 3.3 Launch (guarded)

The command refuses to start if there is more than one GPU process, if < 8000 MiB is free, if the scheduler is still
alive, or if the output dir exists:

```bash
ssh -o BatchMode=yes training@192.168.10.30 'NP=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); echo "gpu procs $NP free $FREE MiB"; if [ "$NP" -gt 1 ] || [ "$FREE" -lt 8000 ] || ps -p 241241 >/dev/null || [ -e ~/fleet_runs/f0_s1042_ttr ]; then echo NOT LAUNCHING; exit 1; fi; cd ~/bigjump_build/code && nohup ~/miniconda3/envs/rsna/bin/python fleet_train.py --model ttr --aug heavy --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs 24 --seed 1042 --save_epochs 6-24 --workers 10 --store ~/fleet_store --labels ~/rsna_train/llm_labels_v4_blend.csv --folds ~/rsna_train/folds5.csv --fold 0 --exclude ~/rsna_train/gold74_eval_set.csv --eval_labels ~/rsna_train/llm_labels_v4_blend.csv --gold ~/rsna_train/gold74_eval_set.csv --out ~/fleet_runs/f0_s1042_ttr > ~/fleet_runs/f0_s1042_ttr.log 2>&1 < /dev/null & echo PID $!' </dev/null
```

Write down the PID. G-T0 / G-T1 kills use `kill <PID>`, never `pkill -f`.

The run relies on these trainer defaults (unchanged): AdamW lr 1e-4 backbone / 1e-3 head, wd .05, 1-epoch warmup +
cosine over `--epochs 24`, fp16 + GradScaler, bs 2 x accum 4 (433 optimizer steps/epoch), clip 1.0, OOF on the 867
every epoch (raw and EMA), gold74 logged but never selected.

`--save_epochs 6-24` only adds raw+EMA snapshots and per-epoch OOFs under `~/fleet_runs/f0_s1042_ttr/epochs/`
(19 x 2 x ~113 MB = 4.3 GB; 7.2 TB free). It is proven not to change training.

Do not edit `~/bigjump_build/code/fleet_model_ttr.py` after launch, because any edit changes model_sha. B6 work goes in
a new folder.

First lines (~2 min after launch):

```bash
ssh -o BatchMode=yes training@192.168.10.30 'head -n 3 ~/fleet_runs/f0_s1042_ttr.log; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader' </dev/null
```
```
labels 4407 | store has 4407 | held-out (fold<0 or --exclude) 74 | train 3466 | val 867 | all_data False | device cuda
BJ opt-in: model ttr (ttr_size 224) | aug heavy | drop_path 0.2 dropout 0.3 | train_slices 16 | ema 0.9995 | epochs 24 | save_epochs (6, 24) | name tag ttr224- | model_sha f141aeed136dfc01
```

Checkpoints are named `f0_s1042_{raw,ema}_ttr224-320x24_cv<NNNN>_best.pt` (~113 MB). The member key `f0_s1042` is
unique, so the TTR member needs no `--tag` when packed.

### 3.4 Expected memory and time (V100 16 GB shared with ONE family-1 job)

These figures hold only with the scheduler stopped and no third job. K2 A1 and G-TTA inflate the epoch times while
they run.

- **Memory: ~3-5 GB in nvidia-smi (ESTIMATE).**
  - Family 1 measured ~7.1 GB per job (14.2/16 GB with two jobs, HANDOFF 23:45).
  - TTR keeps ~0.33x the activations: 224^2 x 16 slots vs 320^2 x 24, i.e. 128 images/step x ~6.4 MB vs 192 x ~13 MB
    of checkpoint-boundary fp16 activations (fleet2 SCHEDULE method).
  - Same 28.6 M params: weights + grads + AdamW + EMA ~0.65 GB, plus context, allocator and cuDNN slack.
  - Next to one F1 job: ~10-12 of 16 GB. Never as a 3rd job while two F1 jobs run (14.2 GB used).
  - `ttr_preflight.py` or nvidia-smi after epoch 1 replaces the estimate.
- **Per-epoch time: ~22-30 min shared (central ~25), ~12-17 min alone (MODEL).**
  - Family 1 measured ~70-74 min/epoch shared incl. eval (~63 train + ~9 eval of raw+EMA on 941 studies), ~36-44 alone.
  - TTR train compute is x0.33 and eval (24 slots at 224) x0.49, plus fixed per-step overheads that do not shrink.
  - **24 epochs = ~9-12 h wall shared.** A launch at ~10:00-10:15 puts the last epoch at ~19:00-22:30, the plan's
    G-T2 window.
- **Loader is not the bottleneck (MEASURED, box CPU).**
  - One TRAIN item costs 0.35 s heavy (16 slots) vs 0.31 s light (24 slots) per worker process (12 real studies, cv2
    threads 0 as in the loader).
  - 10 workers supply ~28 studies/s; the GPU needs ~2-3 studies/s.

### 3.5 Gates on this run

```bash
# G-T0 (after epoch 1, ~25-35 min after launch): epoch wall = the 'epoch X min' on the 'ep 0 ema' line
#   PASS <= 32 min shared (<= 20 alone)
ssh -o BatchMode=yes training@192.168.10.30 'grep -m1 TRAIN-ONLY ~/fleet_runs/f0_s1042_ttr.log; grep -E "^ep 0 (raw|ema)" ~/fleet_runs/f0_s1042_ttr.log; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader' </dev/null
#   FAIL -> kill THIS run only (kill <PID>) and relaunch the 3.3 command with --epochs 18 and a new
#   --out/log name (e.g. f0_s1042_ttr18; the guard refuses an existing dir)
# G-T1 (after epoch 3): gate on max(raw, ema), in practice the RAW line: 'ep 2 raw' val AUC vs llm_labels_v4_blend.csv
#   >= .80, else kill and debug. Do NOT gate on ema: at --ema 0.9995 and 433 optimizer steps/epoch the EMA still holds
#   ~52 % of the step-1 weights (random head) after epoch 3 (0.9995^1298), ~27 % at epoch 6. Family 1 at the default
#   0.999 was at ~27 % already by epoch 3. The early --save_epochs EMA snapshots lag the same way.
ssh -o BatchMode=yes training@192.168.10.30 'grep -E "^ep 2 (raw|ema)" ~/fleet_runs/f0_s1042_ttr.log' </dev/null
```

**G-T2 read** after the TTR closing json line, ~19:00-22:30. f0_s42_v4 is the reference. Runs on CPU, read-only:

```bash
ssh -o BatchMode=yes training@192.168.10.30 'CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/rsna/bin/python ~/bigjump_build/code/gt2_read.py --ttr ~/fleet_runs/f0_s1042_ttr --ref ~/fleet_runs/f0_s42_v4 --eval_labels ~/rsna_train/llm_labels_v4_blend.csv --json ~/fleet_runs/gt2_read.json; echo EXIT $?' </dev/null
```

What it prints:
- both curves (val and gold58 per epoch and tag);
- each tag's best epoch and the exact OOF macro AUC;
- per-label deltas;
- then `G-T2 VERDICT: PASS | MIDDLE | KILL`.

Exit 0 means final. Exit 3 means PROVISIONAL: a run has no closing line yet, or the val sets differ.

Rule (plan sec 4):
- PASS = TTR best >= ref best + .008 AND no label below -.02 AND gold58 not lower;
- KILL = TTR best <= ref best;
- MIDDLE otherwise.

Selection note: TTR picks from 48 candidates vs the ref's 20, worth ~+.001-.002 (GUESS). The per-epoch OOFs in
`epochs/` allow a top-3-epoch mean as a less selection-biased cross-check.

### 3.6 When f0_s42_v4 ends (~10:48)

Run these in order, with TTR still running alone and ~11 GB free:
1. `d1_pick.py` (K2 RUNBOOK sec 3/4, CPU).
2. K2 A0 (CPU) / A1 (GPU, needs >= 6 GB free).
3. If the alternative order of sec 0 was used, the parent gold58 (sec 3.1 command).
4. Then the lane 2 launch (sec 4).

Do not start the lane 2 job while A1 or the parent is on the GPU.

## 4. Other commands enabled by B3-B5

**Lane 2 F1-gated fold-0** (plan sec 3.2; B3 labels; default path, so `~/bigjump_build/code` == `~/fleet` bit for bit).
It keeps seed 42 on purpose: same shuffle and aug stream as f0_s42_v4, so G-L is a paired label test. Guarded the
same way, it refuses with > 1 GPU process or < 8000 MiB free:

```bash
ssh -o BatchMode=yes training@192.168.10.30 'NP=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); echo "gpu procs $NP free $FREE MiB"; if [ "$NP" -gt 1 ] || [ "$FREE" -lt 8000 ] || [ -e ~/fleet_runs/f0_s42g ]; then echo NOT LAUNCHING; exit 1; fi; cd ~/bigjump_build/code && nohup ~/miniconda3/envs/rsna/bin/python fleet_train.py --store ~/fleet_store --labels ~/rsna_train/labels_gated75_logit.csv --folds ~/rsna_train/folds5.csv --fold 0 --exclude ~/rsna_train/gold74_eval_set.csv --eval_labels ~/rsna_train/llm_labels_v4_blend.csv --gold ~/rsna_train/gold74_eval_set.csv --epochs 10 --seed 42 --out ~/fleet_runs/f0_s42g > ~/fleet_runs/f0_s42g.log 2>&1 < /dev/null & echo PID $!' </dev/null
```

**B4 pack** of the fold-0 runs (laptop, after scp of each run's chosen file into its own folder). The three F1 runs
share `f0_s42`; TTR's `f0_s1042` is already unique:

```bash
$PY $SP/bigjump/builds/code/pack_fleet_ds.py --out <ds dir> --store <dir with the parity npz> --tag =,f0_s42v4,f0_s42g,= <mix50 ckpt> <v4 ckpt> <gated ckpt> <ttr ckpt>
```

The K2 kit's `build_k2.py --ckpt PATH=NEWKEY` still works for family-1 pairs. It does not know TTR members; that is
B6/B7. All-data TTR seeds (plan sec 3.3) become **1001-1005** (BJ-5).

**B5 G-TTA.**
- The offsets are +-(0.5 - 1e-3), not +-0.5 (FIX-1, sample_triplets' half-to-even tie on 24-slice stacks). B6's
  worker pass B must use the same HALF.
- GPU guard: the bench refuses below 4 GB free and caps itself at 3 GB, so it can only OOM itself.
- It is still a third job that slows the trainers, so do not run it during TTR's epoch 1 (G-T0).

Smoke on ~10 studies first:

```bash
ssh -o BatchMode=yes training@192.168.10.30 'cd ~/bigjump_build/code && ~/miniconda3/envs/rsna/bin/python tta_bench.py --store ~/fleet_store --folds ~/rsna_train/folds5.csv --fold 0 --exclude ~/rsna_train/gold74_eval_set.csv --eval_labels ~/rsna_train/llm_labels_v4_blend.csv --gold ~/rsna_train/gold74_eval_set.csv --n 10 --out ~/bigjump_build/tta_n10 --oof ~/fleet_runs/f0_s42/oof_f0_s42_raw.csv ~/fleet_runs/f0_s42/oof_f0_s42_ema.csv -- ~/fleet_runs/f0_s42/f0_s42_raw_*_best.pt ~/fleet_runs/f0_s42/f0_s42_ema_*_best.pt' </dev/null
```

The gate run, cheapest form: A and B only, with a time cap. The gate line says PARTIAL if the cap cuts it.

```bash
ssh -o BatchMode=yes training@192.168.10.30 'cd ~/bigjump_build/code && ~/miniconda3/envs/rsna/bin/python tta_bench.py --store ~/fleet_store --folds ~/rsna_train/folds5.csv --fold 0 --exclude ~/rsna_train/gold74_eval_set.csv --eval_labels ~/rsna_train/llm_labels_v4_blend.csv --gold ~/rsna_train/gold74_eval_set.csv --passes A,B --max_minutes 20 --max_gpu_gb 3 --min_free_gb 4 --out ~/bigjump_build/tta_AB --oof ~/fleet_runs/f0_s42/oof_f0_s42_raw.csv ~/fleet_runs/f0_s42/oof_f0_s42_ema.csv -- ~/fleet_runs/f0_s42/f0_s42_raw_*_best.pt ~/fleet_runs/f0_s42/f0_s42_ema_*_best.pt' </dev/null
# -> tta_summary.json + 'G-TTA ... -> PASS (serve A+B) | FAIL (serve pass A only)'. The full 5-pass run is ~45-75 min.
```

## 5. Not done here / caveats

- **B6/B7 are 10-05 items.** B6 is worker pass B + the family-interleaved valve; pass B must use HALF = 0.5 - 1e-3.
  B7 is the K3 builder. The bigjump worker already loads TTR members; the K2 kit (worker, build_k2.py) refuses them by
  design.
- **The scheduler was not read by Claude.** The 3.0 / 3.1 commands come from the review and from HANDOFF, and they
  check themselves: the PID's command line is checked, and the parent command can be compared with `grep`.
- **The GPU guards were not executed** in `tta_bench.py` (`--min_free_gb` / `--max_gpu_gb`) and `ttr_preflight.py`
  (need + headroom, cap), because this pass used no GPU. Their CPU paths ran. If a guard misfires, it refuses or
  OOMs itself; it cannot touch another process.
- **Memory and per-epoch time** above are MODEL/ESTIMATE. G-T0 at epoch 1 is the measurement.
- **`fleet_model_ttr.py` is frozen at model_sha `f141aeed136dfc01`** from the launch on. The pre-fix sha
  `dd2dd29b6718600e` was never used by a real run.
- **fp32 vs fp16.** The CPU parity tests prove trainer path == worker path in fp32. The GPU fp16 path differs by
  ~1e-4 (as in K2 A0: 1.3e-4), which `tta_bench --oof` reports on the first GPU run.
- **The K2 kit changed under the first build.** At 05:32 another session patched `k2/cells/k2_fleet_cell.py`,
  `k2/build_k2.py` and `k2/make_k2_kernel.py`. This build never wrote into `k2/`; t_b4 used the patched kit and
  passed. `k2/code/` is unchanged (worker md5 `7c61ff420b8f81779ba61e151c1fad6f`, the base of the bigjump worker).
