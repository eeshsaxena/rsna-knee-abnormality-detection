# MODAL_E2: TTR288 frac75 seed 1046 on Modal (2026-10-09)

**Status: DONE. All 5 folds (f0-f4, seed 1046) trained on Modal L40S, fetched md5-verified and placed on the box with
VERIFY_OK receipts. `/fleet_store` deleted from the volume at 20:24 IST. Billed $0.00 (free credit).**

**COST CORRECTION (21:57 IST): the real E2 cost is $27.47 by the per-app billing report (smoke $0.72, check $0.01,
f0-3 $21.39, f4 $5.35, fetches ~$0.01). `modal billing summary` now shows $28.71 metered for the month. The $25.05
quoted at 20:25 was a LAGGING summary reading: during the run the summary trailed real usage by ~1-2 h. Consequence: the
fold-4 rule check at 18:00 read $20.28, but the per-app report puts f0-3 + smoke at ~$22.1, just over the $22.0 line.
The month total by the summary ($28.71) is now above the $28.5 cap, and ~$1.3 of a $30 free credit remains.**

## What was trained

- Family: E2 = TTR288 frac75, **seed 1046**, folds 0-4 (f0-f3 in one parallel launch, f4 after the budget check).
- Recipe: the box queue line `Q4 TTR288 s1045 f0 frac75` (`~/queue/queue.txt`, `#RUN 10-09 01:15`), token for token.
  Only 3 of the 40 tokens change: `--seed 1046`, `--fold F`, `--out /vol/runs/fF_s1046_ttr288frac75`.
  `~/modal_e2/make_template.sh` builds `meta/argv_template.txt` from that queue line and fails if anything else differs;
  `e2_app.py::_argv` re-asserts 40 tokens, seed, fold, out, epochs 21, ttr 288, grad_ckpt 0, workers 10 and both label files.
- Template: `--model ttr --ttr_size 288 --aug heavy --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995
  --epochs 21 --save_epochs 6-21 --workers 10 --seed 1046 --fold F --store @R@/fleet_store
  --labels @R@/novel/frac75/labels_frac75_logit.csv --folds @R@/rsna_train/folds5.csv
  --exclude @R@/rsna_train/gold74_eval_set.csv --eval_labels @R@/rsna_train/llm_labels_v4_blend.csv
  --gold @R@/rsna_train/gold74_eval_set.csv --out @OUT@ --grad_ckpt 0`.
- Environment in the container: `HF_HOME=<inputs>/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4` (as on Vast); one GPU.

## Setup

| item | value |
|---|---|
| Modal | workspace/profile `eeshsaxena`, CLI 1.6.1 at `~/modalenv` on the box (token never printed) |
| App | `rsna-e2` (`~/modal_e2/e2_app.py`; laptop copy in `session_2026-10-09/modal_e2/`) |
| Volume | `rsna-e2`: `/code2` `/novel/frac75` `/rsna_train` `/hf` `/fleet_store` `/meta` (inputs), `/runs` (outputs) |
| Image | `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` (torch 2.5.1+cu121, torchvision 0.20.1, Python 3.11.10) + box pins: numpy 1.26.4, pandas 3.0.5, scikit-learn 1.9.0, scipy 1.17.1, timm 1.0.29, huggingface_hub 0.36.2, safetensors 0.8.0, PyYAML 6.0.3, pillow 12.3.0; opencv-python-headless 4.14.0.94 with `--no-deps` (as on the box and Vast) |
| Weights | ConvNeXt-T `timm/convnext_tiny.fb_in22k_ft_in1k` rev bc48a87f from the box HF cache (`~/.cache/huggingface/hub/...`, blob sha256 77b619e1..), loaded offline |

What was uploaded (nothing else): `~/fast_build/code2` (11 .py, no `__pycache__`), `~/novel/frac75/labels_frac75_logit.csv`,
`~/rsna_train/{folds5,gold74_eval_set,llm_labels_v4_blend}.csv`, the HF snapshot above, `~/fleet_store` (4407 competition
npz, 11.66 GB) and `meta/` (argv template, md5 manifests, box parity json, the Vast kit's `parity.py`).
`stage_and_upload.sh` and `upload_store_sharded.py` refuse any source that resolves under `~/oai_*` or `~/union_store`,
any symlink, and any store entry that is not one of the 4407 regular `*.npz` files. **No OAI data was uploaded.**

Upload: a single `modal volume put` stream ran at ~5 MB/s (one TCP stream to us-east), so it was stopped and replaced by
8 parallel SDK uploaders over disjoint shards: 4407 files, 11.66 GB in 348 s (33.5 MB/s).

Parity (`modal run e2_app.py::check`, CPU only, 3 min): md5 of all 4407 store files on the volume = the box manifest
(`~/vast320/store_md5.txt`, signature re-checked), md5 of all 18 code/csv/HF files identical, and the Vast kit's
`parity.py` gives IDENTICAL `contract`, `select_sha`, `jpeg_fp`, `model_sha` (f141aeed136dfc01), `n`, `fold_counts`
(3466/867 train/val for f0-f2, 3467/866 for f3-f4), 4-study `decode_probe`, `backbone_sha` (31fa48e9af913136),
`state_keys` (199) and `pretrained_cfg`. Every training container re-copies the inputs to local disk and re-checks all
4407 + 18 md5s before it starts (7-15 s copy, 4 s md5). The first loss line (`ep 0 it 0 loss 0.6816` for f0) matched in
all three smoke containers.

## GPU choice (smoke, 1000 iterations of the real f0 argv on each GPU, in parallel)

| GPU (Modal $/h) | steady s/it (it 200-1000) | GPU util | CPU used | train min/epoch | fold h (MODEL) | $/fold (GPU+CPU+mem) |
|---|---|---|---|---|---|---|
| H100 80GB ($3.95) | 0.120 | 97 % | 6.4 vCPU | 3.5 | 1.56 | ~6.7 |
| **L40S ($1.95)** | **0.176** | 100 % | 4.4 vCPU | 5.1 | 2.28 | **~5.1** |
| A100-SXM4-40GB ($2.10) | 0.216 | 94 % | 5.1 vCPU | 6.2 | 2.79 | ~6.7 |
| ref: RTX 3090 (Vast, 10-09) | 0.28 | | | 8.2 | 3.6 | |
| ref: box V100-FHHL | ~0.46 | | | 13.4 | 5.8-6.0 | |

MODEL: 1733 it/epoch; eval + saves = 0.27 x train (measured on the 3090 and the V100, confirmed on L40S: 4.7 train /
5.9 total min); 21 epochs + ~1.5 min container start and staging. Non-GPU cost measured from the smoke bill:
~$0.25/h CPU + ~$0.05/h memory per container. A10G and L4 were not measured: at public speed ratios they cost
$5.6-7.4 per fold. The smoke cost $0.72 and the CPU parity check $0.01.

Per-fold limits: `gpu="L40S"`, `timeout=9540` s (2.65 h, a 16 % margin over the 2.28 h MODEL), `cpu=(4, 8)`,
`memory=(6 GiB, 16 GiB)`, `retries=0`, `max_containers=5`.

## Budget

- 5 folds on L40S = 5 x ~$5.1 = ~$25.5, plus $0.73 already spent = ~$26.3: over the $25 hard target, so **only folds 0-3
  were launched** (expected ~$20.8 in total).
  The coordinator then approved fold 4 under a rule (launch only if metered <= $22.0 after f0-3, worst case <= $28.5);
  it was met ($20.28) and fold 4 ran (see Results).
- Worst case for the 4 launched folds: 4 x 2.65 h x $2.25/h (a hung trainer bills GPU + the CPU/memory request)
  = $23.9 + $0.73 = $24.6. At the CPU/memory LIMITS for the whole timeout it would be $26.7; the measured use is about
  half the limits.
- Fold 4 on Modal L40S would add ~$5.2 (total ~$26), still inside the ~$30/month free credit. One command once approved:
  `E2_FOLDS=4 setsid nohup ~/modalenv/bin/modal run --detach /home/training/modal_e2/e2_app.py::train > ~/modal_e2/logs/train_f4.out 2>&1 < /dev/null &`
- Remaining credit: the CLI exposes only the month's metered cost (`modal billing summary`), not the credit balance.
  Metered before this job: $0.00 (so ~$30 of free credit at the start).

## Timeline (IST, 2026-10-09)

| time | event |
|---|---|
| 14:12 | workspace empty: no apps, no volumes, $0.00 metered this month |
| 14:56 | `~/modal_e2` created; argv template built from the queue line (40 tokens, 3 changed) |
| 14:57-15:09 | upload: small files 24 s; store single-stream at ~5 MB/s, switched at 15:03 to 8 shards (348 s, 33.5 MB/s) |
| 15:12-15:15 | image build + CPU parity check: everything IDENTICAL |
| 15:15-15:21 | 3-GPU smoke (H100, L40S, A100-40GB) |
| 15:23:30 | `modal run --detach e2_app.py::train`, folds 0-3 on L40S (app ap-gQe22pPNrWaeyJJy89mYy3) |
| 15:26-15:34 | containers start: f0 15:25:57, f1 15:28:37, f2 15:31:43, f3 15:34:24 (Modal staggers GPU starts by ~3 min) |
| 16:00 | f0 5.9 min/epoch (ETA ~17:31); f1-f3 6.6-6.7 min/epoch at ep 0 (ETA ~17:50-17:55); timeouts 18:05-18:13 |
| ~17:30-18:00 | laptop network dropped twice; the detached app was unaffected |
| 17:33 / 17:49 / 17:52 / 17:54 | f0 / f1 / f2 / f3 finished (train 127.7 / 140.9 / 141.0 / 140.2 min) |
| 17:38 | box-side `auto_e2.sh` started (setsid): fetch + place each fold as it finishes, then the fold-4 rule |
| 17:41 / 17:54 / 17:56 / 17:58 | f0 / f1 / f2 / f3 fetched (231 MB each, ~90-105 s) and placed: VERIFY_OK |
| 18:00:56 | metered $20.28 -> fold 4 launched on L40S, timeout 9900 s; container started 18:01:22 |
| 20:22 | f4 finished (train 139.8 min) |
| 20:24 | f4 fetched (231 MB, 95 s) and placed: VERIFY_OK; `/fleet_store` deleted from the volume; summary then read $25.05 (lagging) |
| 21:57 | billing settled: per-app report $27.47, summary $28.71; the D2 f3 request (cap $28.5) NOT run: no headroom |

## Results

Box paths: `~/fleet_runs/fF_s1046_ttr288frac75/` (1 raw + 1 ema `*_best.pt`, `oof_fF_s1046_{raw,ema}.csv`,
`epochs/oof_*_ep6..21.csv` = 32 csvs; no `epochs/*.pt`) + `~/fleet_runs/fF_s1046_ttr288frac75.log`, receipt
`~/fleet_runs/modal_e2_fF_s1046_ttr288frac75_receipt.json`, md5 manifest `~/fleet_runs/modal_e2_fF_s1046_ttr288frac75_manifest.json`.
Every fold: 37 files, every md5 identical to the volume, closing line present, both ckpts `args OK` (fold, seed 1046,
ttr_size 288, epochs 21, frac75 train labels, steven_v4 selection labels, grad_ckpt False, model_sha f141aeed136dfc01,
heavy aug, 16 train slices, dp .2 / do .3, ema .9995, save_epochs 6-21).

| fold | best raw val (steven_v4) @ epoch | gold58 / gold74 at that epoch | best ema @ epoch (gold58) | closing raw / ema | train min | L40S host | cost |
|---|---|---|---|---|---|---|---|
| f0 | .8894 @ ep17 | .9170 / .9126 | .8873 @ ep21 (.9162) | .8894 / .8873 | 127.7 (5.9 min/ep) | 49 GB, driver 580 | ~$4.5 |
| f1 | .8914 @ ep19 | .9150 / .9086 | .8906 @ ep21 (.9152) | .8914 / .8906 | 140.9 (6.5 min/ep) | 46 GB, driver 610 | ~$5.0 |
| f2 | .8934 @ ep18 | .9159 / .9098 | .8927 @ ep21 (.9157) | .8934 / .8927 | 141.0 (6.5 min/ep) | 46 GB, driver 610 | ~$5.0 |
| f3 | .8803 @ ep16 | .9140 / .9091 | .8781 @ ep21 (.9149) | .8803 / .8781 | 140.2 (6.5 min/ep) | 46 GB, driver 610 | ~$5.0 |
| f4 | .8942 @ ep15 | .9151 / .9092 | .8927 @ ep21 (.9147) | .8942 / .8927 | 139.8 (6.5 min/ep) | 46 GB, driver 610 | ~$4.8 |

Seed 1045 for comparison (closing raw / ema): f0 .8893 / .8875 (box), f1 .8922 / .8912, f2 .8917 / .8906,
f3 .8788 / .8778, f4 .8914 / .8903 (Vast). The s1046 folds land within +-.002 of s1045 (f2 +.0017, f3 +.0015).

Per-fold cost = container hours x the measured blended rate ($19.55 for 9.22 container-hours = ~$2.12/h: L40S $1.95 +
CPU ~4.4 cores + memory). Measured in the containers: 4.0-4.6 vCPU average, GPU util ~93 % over the whole run
(eval included).

Metered after folds 0-3 (`modal billing summary`, 18:00 IST, 3 min after the last container stopped): **$20.28**
(smoke + parity check $0.73, folds 0-3 $19.55). Fold-4 rule: $20.28 <= $22.0, so fold 4 was launched with
timeout = min(9900 s, (28.5 - 20.28 - 0.30) / 2.456 $/h) = 9900 s (2.75 h); worst case $20.28 + 2.75 h x $2.456/h
(L40S + the CPU and memory LIMITS) = $27.03 <= $28.5. That bound was computed on the lagging $20.28; with the per-app
report's ~$22.1 it would have been ~$28.9. **Final cost: $27.47 by the per-app report, $28.71 by `modal billing summary`
(21:57 IST)**; fold 4 alone = $5.35 (2.35 container-hours). Per-fold costs from the report: f0-3 $21.39 for 9.25
container-hours, about $2.31/h blended (f0 ~$4.9, f1-f3 ~$5.4 each; the ~$4.5 / ~$5.0 in the table were from the lagging summary).

5-fold s1046 means: best raw .8897, closing raw .8897 / ema .8883 (s1045: closing raw .8887 / ema .8875).

Left on Modal: volume `rsna-e2` keeps `/runs` (all 5 folds incl. `epochs/*.pt`, ~3.6 GB per fold, ~18 GB) plus the small
inputs (`/code2 /novel /rsna_train /hf /meta`, ~120 MB); storage bills ~$0.09/GB-month (~$1.6/month while kept).
Delete with `~/modalenv/bin/modal volume rm -r rsna-e2 /runs` (or `modal volume delete rsna-e2`) once the epoch
checkpoints are no longer wanted. No app is running.

## Files

- Box: `~/modal_e2/` (`e2_app.py`, `make_template.sh`, `stage_and_upload.sh`, `upload_store_sharded.py`, `status.sh`,
  `place_fold.sh`, `auto_e2.sh`, `meta/`, `logs/`: `auto_e2.log`, `f4_decision.txt`, `fetch_fF.out`, `place_fF.out`, `train_f0-3.out`, `train_f4.out`).
- Laptop: `Downloads/rsna-knee/session_2026-10-09/modal_e2/` (copies of the scripts + argv template + box parity json).
- Progress: `ssh training@192.168.10.30 'bash ~/modal_e2/status.sh'` (per-fold last epoch line from the volume logs;
  the logs are committed every ~10 min) or `~/modal_e2/logs/train_f0-3.out` (live stream of all 4 containers).
- Fetch one fold (box): `~/modalenv/bin/modal run /home/training/modal_e2/e2_app.py::fetch --run fF_s1046_ttr288frac75 --dest /home/training/modal_e2/dl`
  then `bash ~/modal_e2/place_fold.sh F` (moves into `~/fleet_runs/`, re-checks md5 + closing line + ckpt args, writes
  `~/fleet_runs/modal_e2_fF_s1046_ttr288frac75_receipt.json`).
