# BIG JUMP builds B1-B5 (BIG_JUMP_PLAN sec 11), 10-04

Built 2026-10-04 04:50-06:00 IST by Claude. Nothing was submitted, pushed, versioned or posted. No running process
was touched, no GPU was used. Box writes: `~/bigjump_build/` (code, tests, outputs) and
`~/rsna_train/labels_gated75_logit.csv` (B3). Every launch below is for Eesh to run.

`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`,
`BLD` = `SP/bigjump/builds`, `B` = `training@192.168.10.30`, `PYB` = `~/miniconda3/envs/rsna/bin/python` (box: torch
2.5.1+cu121, timm 1.0.29, cv2 4.14.0; ConvNeXt-T weights in the HF cache).

## 1. What was built

| Item | File (`BLD/code/`, staged byte-identical at `~/bigjump_build/code/`) | What |
|---|---|---|
| B1 | `fleet_model_ttr.py` | `KneeFleetNetTTR` = `KneeFleetNet` + one change in `encode()`: uint8 -> float [0,1] -> `F.interpolate(224, bilinear, antialias)` -> ImageNet norm. fp32 resize, pass-through if already 224. `fleet_model.py` untouched. **model_sha `dd2dd29b6718600e`** (family 1 stays `fa35880fca963652`, family 2 `ba3e1f85e38128bf`) |
| B1 | `fleet_worker.py` | K2 worker + W-TTR: `model_mod == 'fleet_model_ttr'` builds `KneeFleetNetTTR(ttr_size=args)`, sha checked like family 2. Family-1/2 path unchanged (predictions == K2 worker, max diff 0) |
| B2 | `fleet_train.py` | fleet2 trainer + opt-in BJ-1..4: `--model ttr [--ttr_size 224]`, `--aug heavy`, `--drop_path`, `--dropout`, `--save_epochs a-b`; slot subset (`--train_slices`) before the per-view aug in the heavy path. **No flag = byte-identical run** |
| B2 | `fleet_aug.py` | the heavy TRAIN path (imported only under `--aug heavy`): coins first (mirror .5, 16-of-24 slots, view drop .2), then per view: decode, slice jitter U(-1,1), mirror-remapped slot subset, affine rot +-15 / scale .85-1.15 / shift +-8 %, gamma .75-1.35 / gain .85-1.15 / bias +-15, noise sigma U(0,6) p .3, CoarseDropout 1-3 holes 8-16 % (same holes every slice) p .3 |
| B4 | `pack_fleet_ds.py` | K2 pack tool + P-TTR (ships `fleet_model_ttr.py`) + **P-TAG `--tag K1,K2,...`** (one entry per member, `=` keeps the key; KEY = the member's own `f{fold}_s{seed}` + `[A-Za-z0-9]*`, e.g. `f0_s42v4`) + P-RUN (identical ckpt args = one run, refused even if re-keyed). Manifest gains `sources` (worker reads only `checkpoints`) |
| B5 | `tta_bench.py` | G-TTA bench through the worker path (worker member build + worker input on store npz): passes A / M / P(+.5) / N(-.5) / B(mirror + .5); variants none, mirror, offset, A+B, A+B lesion-max, all5, raw+EMA; macro AUC vs v4 on fold-0 val + gold58; paired bootstrap; gate line; `--oof` parity; `--fp_check` (CUDA) |
| gate | `gt2_read.py` | G-T2 read: exact OOF macro AUC (best raw/EMA) TTR vs f0_s42_v4, per-label floor, gold58 direction; PASS / MIDDLE / KILL; exit 3 while a run is unfinished |
| G-P0 | `aug_dump.py` | PNG grid of heavy-aug draws for one study, mirror check, light vs heavy loader cost |
| opt. | `ttr_preflight.py` | 2-4 min GPU check (memory + s/step, F1 reference in the same window). Writes nothing |
| copies | `fleet_common.py`, `fleet_model.py`, `fleet_model_f2.py` | md5 == `~/fleet`, `~/fleet2_code`, `k2/code` |

Also: `BLD/PATCH.diff` (trainer vs fleet2, worker vs K2, pack vs K2), `BLD/tests/` (laptop suite, `run_all.sh`,
`transcript.txt`), `BLD/box/` (box suite `run_box_tests.sh`, box transcript `BLD/tests/out/box_run1.txt`), aug PNGs
`BLD/tests/out/box_aug/`.

md5 (laptop == box `~/bigjump_build/code`): fleet_train.py `f8b5739dc2e120bb8592b14db47b0be7`, fleet_model_ttr.py
`0a5f3ba4e6779d21712f55d1dc8d9317`, fleet_aug.py `4e7311c2f4c6f94acfb1a1abe1bfc12f`, fleet_worker.py
`f1301ea8511977b8339bb011eb5fd2d7`, pack_fleet_ds.py `2b5e3a45bc01875de07fa93ccaae290d`, tta_bench.py
`0c8816a0007c605ca456a239beaaf172`, gt2_read.py `28f62fcc5b599b9f897d9f4075c6db74`, fleet_common.py
`658ab8103acdf6e4aec029a0f6ee5959`, fleet_model.py `897f83b272fbb144a4d3deab773d9064`.

## 2. Test results

Laptop CPU (`BLD/tests/run_all.sh`, 64 px synthetic store, convnext_atto, cv2 PIL shim; all 7 rc 0, ~11 min;
transcript `BLD/tests/transcript.txt`, t_gt2_saveep re-run once after a harness path fix):

- **t_b1_model**: contract files identical in all 4 trees; model_sha registry identical in worker / pack / tta_bench
  (f1 `fa35880fca963652`, f2 unchanged, TTR distinct); TTR with ttr_size == input size == KneeFleetNet (eval, train
  logits, 144 grads: max diff 0, same state_dict keys); encode == manual interpolate+norm+cnn (0); 320 -> 224 seen by
  the backbone; antialias active.
- **t_b2_default_parity**: no opt-in flag -> bit-identical to round3/fleet (`~/fleet`) AND fleet2 (`~/fleet2_code`) on
  fold-0 2 ep, all-data stop_epoch, T6 `--train_slices 16`, and `--f2 1` (resnet18): ckpt names, all state_dict
  tensors `torch.equal`, ckpt args ==, cv/epoch ==, OOF bytes ==, metric log lines ==.
- **t_b2_heavy_equiv**: heavy path with pixel aug off == default `mirror_views(full)[:, pos]` on 480 cases (228
  mirrored), K 16/24; forced view drop (120 cases); sagittal mirror reverses slot depth + channels; aug deterministic,
  identity cfg returns input, holes identical on all 48 slice-channels.
- **t_b2_ttr_train**: TTR fold-0 3 ep + all-data 3 ep through the trainer: names `f0_s42_{raw,ema}_ttr224-64x24_*`,
  `epochs/` snapshots + per-epoch OOF, per-epoch ckpt cv == log, top-level best == its epoch snapshot (tensors, OOF
  bytes), args (model_mod, ttr_size 224, aug heavy, drop_path .2, dropout .3, ema .9995, train_slices 16, sha); guards
  refuse `--model ttr --f2 1` and `--save_epochs 3-1`; bigjump worker serves 2 TTR + 2 F1 members (re-keyed with
  `--tag`), F1 predictions == K2 worker (max diff 0); K2 worker refuses a TTR member (rc 1, parent restored).
- **t_b4_pack_tag**: two runs, IDENTICAL file names: K2 pack tool and bigjump pack without `--tag` refuse;
  `--tag =,f0_s42v4` packs both (bytes == sources); bad keys (fold, seed, `_` suffix, count, raw+ema of one run, same
  file twice) refused. **K2 notebook (k2/make_k2_kernel.py, fleet-only) on the re-keyed 2-member dataset: k2_check
  PASS 22/22 on 24 studies and on the 3-study commit cut, n_keep [2, 2]; with the K2 worker swapped in (what
  build_k2.py ships) fleetsolo byte-identical.** Real contract: `k2/build_k2.py --ckpt <real interim> --ckpt
  <same-name v4 variant>=f0_s42v4` -> 2 members, upload 225 MB / 12 files, kernel built.
- **t_b5_tta_bench**: store rebuilt from the fake competition DICOMs exactly as the prep kernel; tta_bench pass A ==
  bigjump worker P (max diff 0, TTR and F1 members); pass B == independent recompute (0); all variants + gate; trainer
  OOF vs tta_bench pass A at chunk 32 and 48: max diff 0.
- **t_gt2_saveep**: `--save_epochs` does not change training (same tensors / cv / OOF bytes; args differ only in
  `save_epochs`); gt2_read score == recomputed macro AUC, verdict == rule, exit 3 PROVISIONAL without a closing line.

Box CPU (`bash ~/bigjump_build/box/run_box_tests.sh`, `CUDA_VISIBLE_DEVICES=`, nice 10, real cv2 4.14):

- **box_check PASS**: contract files == `~/fleet` and `~/fleet2_code`; model_sha in the box env f1 `fa35880fca963652`,
  f2 `ba3e1f85e38128bf`, TTR `dd2dd29b6718600e`.
- **B3 PASS**: `~/rsna_train/labels_gated75_logit.csv` md5 `a813b986aa09b6fc4736ff493fbaa918` (== laptop); 4407 ids ==
  llm_labels_v4_blend; finite, in [0,1]; report-ADDRESSED cells == steven_v4 exactly (38,427 cells, max diff 0);
  report-UNK cells 27.3 %, all changed (mean |diff| .135); fold-0 contract train 3466 / val 867.
- **box_parity PASS**: same 7 parity comparisons as the laptop, in the box env (real cv2), all bit-identical; the TTR
  path runs there.
- **box_gp0 (G-P0) PASS** on the 4 real K2 parity studies (`~/fleet_store`, steven_v4, 2 train / 2 val), REAL
  pretrained ConvNeXt-T, the exact TTR flags, 1 epoch CPU fp32 (51 s): ckpt `f0_s42_{raw,ema}_ttr224-320x24_*`,
  model_sha `dd2dd29b6718600e` (!= family 1), args model_mod/ttr_size/aug/train_slices right; **trainer path vs worker
  path (tta_bench pass A) predictions (sigmoid of the logits): max |diff| 0.0 for raw and EMA, at chunk 32 AND 48**, all finite; mirror check PASS
  (sagittal slot k == slot 23-k with channels reversed, axial/coronal W-flipped); aug PNGs look anatomically sane
  (`BLD/tests/out/box_aug/`: rotation, scale, intensity, the same dropout holes on every slice); real-data tta_bench
  smoke on the interim family-1 member (3 val + 3 gold58 x 5 passes, CPU 323 s): pass A vs its GPU fp16 trainer OOF
  max |diff| 2.1e-4 (fp16 noise, K2 A0 saw 1.3e-4).
- **gt2_read on the LIVE logs** (format dry run, read-only, mix50 run as "--ttr" vs f0_s42_v4 at epoch 6): parses both
  real logs, OOF macro AUC == the log values (.8748 / .8603), 867 == 867 val studies, prints PROVISIONAL, exit 3.

## 3. TTR-A fold-0 launch (lane 1, plan sec 3.1; labels = pure steven_v4)

Pre-checks (any time before):

```bash
ssh -o BatchMode=yes training@192.168.10.30 'cd ~/bigjump_build/code && md5sum fleet_train.py fleet_model_ttr.py fleet_aug.py fleet_model.py fleet_common.py; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv' </dev/null
#   expect f8b5739d..., 0a5f3ba4..., 4e7311c2..., 897f83b2..., 658ab810...; launch only with >= 6 GB free
# optional, 2-4 GPU-minutes, writes nothing (peak memory + s/step of TTR vs F1 in the same contention window):
ssh -o BatchMode=yes training@192.168.10.30 'cd ~/bigjump_build/code && ~/miniconda3/envs/rsna/bin/python ttr_preflight.py --store ~/fleet_store' </dev/null
```

Launch (when f0_s42 mix50 has printed its closing `{"raw": .., "ema": ..}` line, ~09:35; it then shares the V100 with
f0_s42_v4 until ~10:30, later with lane 2):

```bash
ssh -o BatchMode=yes training@192.168.10.30 'cd ~/bigjump_build/code && nohup ~/miniconda3/envs/rsna/bin/python fleet_train.py --model ttr --aug heavy --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs 24 --seed 42 --save_epochs 6-24 --workers 10 --store ~/fleet_store --labels ~/rsna_train/llm_labels_v4_blend.csv --folds ~/rsna_train/folds5.csv --fold 0 --exclude ~/rsna_train/gold74_eval_set.csv --eval_labels ~/rsna_train/llm_labels_v4_blend.csv --gold ~/rsna_train/gold74_eval_set.csv --out ~/fleet_runs/f0_s42_ttr > ~/fleet_runs/f0_s42_ttr.log 2>&1 < /dev/null & echo PID $!' </dev/null
```

Recipe defaults this relies on (unchanged trainer defaults): AdamW lr 1e-4 backbone / 1e-3 head, wd .05, 1-epoch
warmup + cosine over `--epochs 24`, fp16 + GradScaler, bs 2 x accum 4, clip 1.0, OOF on the 867 every epoch (raw and
EMA), gold74 logged never selected. `--save_epochs 6-24` only adds raw+EMA snapshots and per-epoch OOFs under
`~/fleet_runs/f0_s42_ttr/epochs/` (19 x 2 x 112.8 MB = 4.3 GB; 7.2 TB free); it is proven not to change training.

First lines to check (~2 min after launch):

```
labels 4407 | store has 4407 | held-out (fold<0 or --exclude) 74 | train 3466 | val 867 | all_data False | device cuda
BJ opt-in: model ttr (ttr_size 224) | aug heavy | drop_path 0.2 dropout 0.3 | train_slices 16 | ema 0.9995 | epochs 24 | save_epochs (6, 24) | name tag ttr224- | model_sha dd2dd29b6718600e
```

Checkpoints: `f0_s42_{raw,ema}_ttr224-320x24_cv<NNNN>_best.pt` (112.8 MB; key `f0_s42` collides with family-1 f0_s42:
pack it with `--tag ...,f0_s42ttr`).

### Expected memory and time (V100 16 GB shared with ONE family-1 job)

- **Memory: ~3-5 GB in nvidia-smi (ESTIMATE).** Family 1 measured ~7.1 GB per job (14.2/16 GB with two jobs, HANDOFF
  23:45). TTR keeps ~0.33x the activations (224^2 x 16 slots vs 320^2 x 24: 128 images/step x ~6.4 MB vs 192 x ~13 MB
  checkpoint-boundary fp16 activations, fleet2 SCHEDULE method), same 28.6 M params (weights + grads + AdamW + EMA
  ~0.65 GB), plus context/allocator/cuDNN slack. Next to one F1 job: ~10-12 of 16 GB. Never as a 3rd job while two F1
  jobs run (14.2 GB used). `ttr_preflight.py` or nvidia-smi after epoch 1 replaces the estimate.
- **Per-epoch time: ~22-30 min shared (central ~25), ~12-17 min alone (MODEL).** Family 1 measured ~70-74 min/epoch
  incl. eval shared (~63 train + ~9 eval of raw+EMA on 941 studies), ~36-44 alone. TTR train compute x0.33, eval
  (24 slots at 224) x0.49, plus fixed per-step overheads that do not shrink. **24 epochs = ~9-12 h wall shared**:
  launch ~09:40 -> last epoch ~19:00-22:00 (the plan's G-T2 window).
- **Loader is not the bottleneck (MEASURED, box CPU):** one TRAIN item costs 0.35 s heavy (16 slots) vs 0.31 s light
  (24 slots) per worker process (12 real studies, cv2 threads 0 as in the loader). 10 workers supply ~28 studies/s; the
  GPU needs ~2-3 studies/s.

### Gates on this run

```bash
# G-T0 (after epoch 1, ~25-35 min): epoch wall = the 'epoch X min' on the 'ep 0 ema' line; PASS <= 32 min shared (<= 20 alone)
ssh -o BatchMode=yes training@192.168.10.30 'grep -m1 TRAIN-ONLY ~/fleet_runs/f0_s42_ttr.log; grep -E "^ep 0 (raw|ema)" ~/fleet_runs/f0_s42_ttr.log' </dev/null
#   FAIL -> kill THIS run only (kill <PID printed at launch>, never pkill -f) and relaunch the same command with --epochs 18
# G-T1 (after epoch 3): 'ep 2 raw/ema' val AUC vs llm_labels_v4_blend.csv >= .80, else kill and debug
ssh -o BatchMode=yes training@192.168.10.30 'grep -E "^ep 2 (raw|ema)" ~/fleet_runs/f0_s42_ttr.log' </dev/null
```

**G-T2 read** (after the TTR closing json line, ~20:00-22:00; f0_s42_v4 is the reference; CPU, read-only):

```bash
ssh -o BatchMode=yes training@192.168.10.30 'CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/rsna/bin/python ~/bigjump_build/code/gt2_read.py --ttr ~/fleet_runs/f0_s42_ttr --ref ~/fleet_runs/f0_s42_v4 --eval_labels ~/rsna_train/llm_labels_v4_blend.csv --json ~/fleet_runs/gt2_read.json; echo EXIT $?' </dev/null
```

Prints both curves (val, gold58 per epoch and tag), each tag's best epoch, the exact OOF macro AUC, per-label deltas,
then `G-T2 VERDICT: PASS | MIDDLE | KILL`. Exit 0 = final; exit 3 = PROVISIONAL (a run has no closing line yet, or the
val sets differ). Rule (plan sec 4): PASS = TTR best >= ref best + .008 AND no label below -.02 AND gold58 not lower;
KILL = TTR best <= ref best; MIDDLE otherwise. Selection note: TTR picks from 48 candidates vs 20, worth ~+.001-.002
(GUESS); the per-epoch OOFs in `epochs/` allow a top-3-epoch mean as a less selection-biased cross-check.

## 4. Other commands enabled by B3-B5

Lane 2 F1-gated fold-0 (plan sec 3.2; B3 labels; default path, so `~/bigjump_build/code` == `~/fleet` bit for bit):

```bash
ssh -o BatchMode=yes training@192.168.10.30 'cd ~/bigjump_build/code && nohup ~/miniconda3/envs/rsna/bin/python fleet_train.py --store ~/fleet_store --labels ~/rsna_train/labels_gated75_logit.csv --folds ~/rsna_train/folds5.csv --fold 0 --exclude ~/rsna_train/gold74_eval_set.csv --eval_labels ~/rsna_train/llm_labels_v4_blend.csv --gold ~/rsna_train/gold74_eval_set.csv --epochs 10 --seed 42 --out ~/fleet_runs/f0_s42g > ~/fleet_runs/f0_s42g.log 2>&1 < /dev/null & echo PID $!' </dev/null
```

B4 pack of the fold-0 runs (laptop, after scp of each run's chosen file into its own folder):

```bash
$PY $SP/bigjump/builds/code/pack_fleet_ds.py --out <ds dir> --store <dir with the parity npz> --tag =,f0_s42v4,f0_s42g,f0_s42ttr <mix50 ckpt> <v4 ckpt> <gated ckpt> <ttr ckpt>
```

(The K2 kit's `build_k2.py --ckpt PATH=NEWKEY` still works for family-1 pairs; it does not know TTR members: B6/B7.)

B5 G-TTA (only when a lane is free, SM < 85 %; ~10 studies first):

```bash
ssh -o BatchMode=yes training@192.168.10.30 'cd ~/bigjump_build/code && ~/miniconda3/envs/rsna/bin/python tta_bench.py --store ~/fleet_store --folds ~/rsna_train/folds5.csv --fold 0 --exclude ~/rsna_train/gold74_eval_set.csv --eval_labels ~/rsna_train/llm_labels_v4_blend.csv --gold ~/rsna_train/gold74_eval_set.csv --n 10 --out ~/bigjump_build/tta_n10 --oof ~/fleet_runs/f0_s42/oof_f0_s42_raw.csv ~/fleet_runs/f0_s42/oof_f0_s42_ema.csv -- ~/fleet_runs/f0_s42/f0_s42_raw_*_best.pt ~/fleet_runs/f0_s42/f0_s42_ema_*_best.pt' </dev/null
# then the same without --n (867 val + 58 gold58) -> tta_summary.json + 'G-TTA ... -> PASS (serve A+B) | FAIL (serve pass A only)'
```

## 5. Not done here / caveats

- B6 (worker pass B + family-interleaved valve) and B7 (K3 builder) are 10-05 items. The bigjump worker already loads
  TTR members; the K2 kit (worker, build_k2.py) refuses them by design.
- Memory and per-epoch time above are MODEL/ESTIMATE; G-T0 at epoch 1 is the measurement.
- `fleet_model_ttr.py` is frozen: any edit changes model_sha `dd2dd29b6718600e` and the worker refuses the TTR ckpts.
- The CPU parity tests prove trainer path == worker path in fp32; the GPU fp16 path differs by ~1e-4 (as K2 A0: 1.3e-4),
  which `tta_bench --oof` reports on the first GPU run.
- The K2 kit changed under this build: at 05:32 another session patched `k2/cells/k2_fleet_cell.py`,
  `k2/build_k2.py` and `k2/make_k2_kernel.py` (`.preSolo` backups left next to them) and re-ran its smoke. This build
  never wrote into `k2/`. The final t_b4 run (05:41) used the patched kit and passed (k2_check 22/22 x3, build_k2 real
  pair OK). `k2/code/` is unchanged (worker md5 `7c61ff420b8f81779ba61e151c1fad6f`, the base of the bigjump worker).
