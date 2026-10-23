# LIGHTNING_D2: D2 (TTR224 frac75 seed 1044) fold 3 on Lightning AI (2026-10-09)

**Status: BLOCKED, needs Eesh.** The GPU job was refused at creation (19:52:58 IST):

> HTTP 403: "Free-tier users must have a verified payment method before starting GPU compute"

- **What does not work:** no GPU of any kind can start on this account without a card. I did not add a card and will not.
- **What is ready:** everything up to the GPU launch is DONE and verified.
- **Spend:** 0.238 credits. Balance 4.762.
- **Nothing running:** both Studios are stopped and there are no running jobs.
- **Placeholders:** no f3 or f4 placeholder was created on the box, since no training started.

## Account facts (billing API, 18:58 IST)

- Lightning user `eeshsaxena`, teamspace `default-project`, org `eeshsaxena-org`.
- Balance **5.0 credits, not 15**: user, teamspace and org balances are all 5.0, with 0 spent.
- Phone-verified 2026-10-09 13:21 UTC.
- Profile flag `eligible_for_credit_card_unlock = True`: the rest of the free credits, and GPU access, are card-gated.
- Free CPU Studio `scratch-studio-devbox` (lightning-public-prod, 4 vCPU, 15 GB RAM):
  - Required-balance reason `SUFFICIENT_FREE_STUDIO`, so it costs 0 credits.
  - Never change its auto-sleep: the SDK warns that this converts it from free to paid.

### Single-GPU prices (credits/h = $/h)

| GPU | Cloud account | On-demand | Spot | vCPU |
|---|---|---|---|---|
| L40S | lightning-nebius-prod | **2.14** | none | 16 |
| L40S | lightning-public-prod (AWS g6e.4xlarge) | 3.54 | 2.51 | 16 |
| A100-40GB | lightning-lambda-prod | 2.19 | none | |
| L4 | gcp (g2-standard-8) | 0.79 | 0.70 | 8 |
| L4 | AWS (g6.4xlarge) | 1.68 | 1.00 | 16 |
| T4 | gcp (n1-standard-8) | 0.69 | 0.38 | 8 |
| T4 | AWS (g4dn.2xlarge) | 1.04 | 0.64 | 8 |
| H100 | Nebius / Lambda | 5.68 / 4.68 | none | |

- **Concurrency:** not testable (GPU refused).

## Decisions

- **Option B (coordinator, 19:05):**
  - Train f3 only, on Nebius L40S on-demand (no spot).
  - Abort and keep the outputs if the projected total spend exceeds 4.3 credits.
  - Stop at the closing line.
  - Create the placeholder only once training has started.
- **Estimated f3 cost on Nebius L40S:** ~1.4 h of training + ~7 min of billed job setup, so ~3.2-3.4 credits including the 0.24 already spent.
  - This scales Modal's L40S TTR288 timing (0.176 s/it, 2.1-2.35 h per fold) by the measured box 224/288 ratio of 0.6.
- **Studio jobs must run in their Studio's own cloud account** (SDK check).
  - The free Studio is in lightning-public-prod (AWS only): L40S there costs 3.54 on-demand, which is over the cap, and spot was excluded.
  - So I created the paid Studio **`rsna-d2-nb` in lightning-nebius-prod**. It ran on Nebius CPU-4 at 0.40/h only for setup and is now stopped.

## What was uploaded (nothing else)

- `~/fast_build/code2` (12 *.py, no `__pycache__`).
- `~/novel/frac75/labels_frac75_logit.csv`.
- `~/rsna_train/{folds5,gold74_eval_set,llm_labels_v4_blend}.csv`.
- The ConvNeXt-T HF snapshot rev bc48a87f (sha256 77b619e1.. checked).
- `~/fleet_store`: 4407 competition npz, 11.66 GB, as 6 tar shards.
- `meta/`: argv template, md5 manifests, box parity json, `parity.py`.

`stage.sh` refuses:
- any source that resolves under an `oai` or `union_store` path;
- any symlink;
- any OAI name in the store;
- any store entry that is not one of the 4407 regular npz files.

**No OAI data was uploaded.**

## Recipe

- argv = the Vast kit's `remote/argv_template.txt`. On the box, `box_ops.sh argv-check 3` = **IDENTICAL** to the queue line `Q2 frac75 f3` except `--seed 1044` and `--out` (38 tokens, 2 changed).
- Template: `--model ttr --aug heavy --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs 21 --seed 1044 --save_epochs 6-21 --workers 10 --store @R@/fleet_store --labels @R@/novel/frac75/labels_frac75_logit.csv --folds @R@/rsna_train/folds5.csv --exclude @R@/rsna_train/gold74_eval_set.csv --eval_labels @R@/rsna_train/llm_labels_v4_blend.csv --gold @R@/rsna_train/gold74_eval_set.csv --fold 3 --out @R@/fleet_runs/f3_s1044_ttr_frac75 --grad_ckpt 0`.
- Environment: `HF_HOME=<R>/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4`, as on Vast and Modal, plus `PYTHONDONTWRITEBYTECODE=1` so nothing is written into the snapshot dir.
- `train_job.sh` re-asserts: 38 tokens, seed 1044, fold, epochs 21, save_epochs 6-21, model ttr, gc0, workers 10, 16 slices, out name, both label files.
- **Note on the Kaggle f0 s1044 ckpt:** it says `epochs 22, save_epochs 20-22`. The D2 recipe used here is the queue line's 21 / 6-21, as in the task and the Vast kit.
  - `fetch_place.sh` checks the args like the Vast verify does: every arg except fold, epochs, save_epochs and grad_ckpt must equal the Kaggle f0 ckpt.

## Parity (all IDENTICAL)

| check | free Studio (AWS, 0 credits) | Nebius Studio rsna-d2-nb | dry job on Nebius CPU-4 |
|---|---|---|---|
| tar md5 (7) | OK | OK | |
| store md5 4407/4407 vs box manifest | IDENTICAL (33 s) | IDENTICAL (35 s) | IDENTICAL (13 s, 8 procs) |
| code/csv/HF md5 (18) | IDENTICAL | IDENTICAL | IDENTICAL |
| parity.py TTR224 vs box json | contract, select_sha 91bb8a65, jpeg_fp 34819a62, model_sha f141aeed, n, fold_counts (f3 3467/866), decode_probe, backbone_sha 31fa48e9, state_keys 199, pretrained_cfg: all IDENTICAL | same, all IDENTICAL | |
| venv pins vs box | identical | identical | torch 2.5.1+cu121 imports |

**venv:** `uv venv` Python 3.11.15 (box 3.11.16; Modal ran 3.11.10 with identical parity).
- torch 2.5.1+cu121 and torchvision 0.20.1+cu121 from the PyTorch cu121 index.
- numpy 1.26.4, pandas 3.0.5, scikit-learn 1.9.0, scipy 1.17.1, timm 1.0.29, huggingface_hub 0.36.2, safetensors 0.8.0, PyYAML 6.0.3, pillow 12.3.0.
- opencv-python-headless 4.14.0.94 `--no-deps`.
- cudnn 90100, as on the box.

## Timeline (IST, 2026-10-09)

| time | event |
|---|---|
| 18:56-19:00 | account explored: 5.0 credits, card-gated; reported to main |
| 19:05 | coordinator: option B (f3 only, Nebius L40S, abort above 4.3) |
| 19:03-19:06 | box parity TTR224 (8 s), argv-check f3 IDENTICAL, `stage.sh` (37 s); env build on the free Studio (1 min) |
| 19:07-19:12 | upload to the free Studio: small + 6 shards (single stream 26 MB/s, 5 parallel ~60 MB/s) |
| 19:13-19:15 | free Studio: unpack + md5 + CPU parity, all IDENTICAL |
| 19:14 | `rsna-d2-nb` created in lightning-nebius-prod; uploads went straight to its drive while it was stopped (no cost) |
| 19:20:51 | Nebius CPU-4 start requested; Running 19:26:37 (6 min to start) |
| 19:27 | first prep failed: CRLF line endings from a Windows-side patch (fixed, scripts are LF) |
| 19:29-19:34 | Nebius: env + unpack + md5 + parity, all IDENTICAL |
| 19:34:38 | dry job `rsna-d2-dry1` (Nebius CPU-4): snapshot 2m06s, machine 33 s, **setup 6m41s (billed)**, script 17 s, ended 19:48:05, cost 0.071 |
| ~19:50 | `rsna-d2-nb` auto-slept (idle) |
| 19:52:31 | GPU job `rsna-d2-f3-s1044` submit (Nebius L40S, trainer cap 6100 s, PRE 0.26) |
| 19:52:58 | **refused, 403: verified payment method required for GPU compute** |
| 19:54:30 | free Studio stopped too; zero running machines |

## Credits

- Used: **0.238**. Nebius CPU Studio ~25 min, about 0.167; dry job 0.071; the free Studio costs 0.
- Balance: **4.762**.

## Results per fold

- f3: not trained. No best raw val or gold58, because no GPU was allowed.
- f4: not attempted (option B).

## Lightning job facts (for the next attempt)

- **Job filesystem:**
  - A Studio job sees the Studio snapshot at `/teamspace/studios/this_studio`, including the venv under `/system/conda/...`.
  - `/teamspace/jobs/<job>` is read-only.
  - The job ARTIFACTS are the files created or changed under `this_studio`, uploaded when the command ends (teamspace drive `jobs/<job>/...`).
  - `Teamspace.download_folder("jobs/<job>/ld2_out", dir)` fetches them (tested on the dry job).
- **What the job script does (`train_job.sh`):**
  - The trainer writes to a local root, `/tmp/ldR`: symlinks to the snapshot inputs plus a real `fleet_runs`.
  - Only the fetch set goes to `this_studio/ld2_out` (every 5 min, and once more at the end): log, `*_best.pt`, `oof_*.csv`, `epochs/oof_*.csv`. `epochs/*.pt` is never copied.
  - Then it writes the md5 manifest and exits, so the machine ends right after the closing line.
- **Billing:** machine setup (snapshot restore, ~6.7 min) is billed. Budget ~7-8 min of GPU time per job for setup and teardown.
- **Studio venv trap:** Lightning Studios set `UV_LINK_MODE=symlink` with the cache under `/system/conda`.
  - `uv cache clean` emptied the first venv: torch imported as an empty namespace.
  - `setup_env.sh` now forces `UV_LINK_MODE=copy` and never cleans the cache.
- **Auto-sleep:** a paid CPU Studio auto-sleeps when idle, which stops its spend. The Studio does not need to be running for uploads or (expected) for job submission.
- **In-job cost guard (every 60 s):** projected total = PRE + 2.14 x (0.13 setup + elapsed + remaining + 0.05).
  - Remaining comes from the completed-epoch times (ema line "epoch X min"), or in epoch 0 from the interval s/it between log lines, since the trainer prints a cumulative mean.
  - It aborts (TERM the trainer, copy the outputs, exit) above 4.3, or above 4.73 on the epoch-0 estimate.
  - Unit-tested on a box log.
- **Trainer backstop:** `timeout 6100 s`.

## To resume (only after Eesh adds a verified payment method, his call)

1. **Pre-launch checks:**
   - `~/lightenv/bin/python ~/lightning_d2/ld2.py balance` (a card may also unlock the card-gated credits, in which case option A, f3 + f4, fits).
   - `... ld2.py studios`: expect both Studios stopped, no jobs.
2. **Launch f3:**
   ```
   ~/lightenv/bin/python ~/lightning_d2/ld2.py submit gpu rsna-d2-f3-s1044 6100 <PRE>
   ```
   - PRE = credits spent so far + 0.02.
   - Trainer cap = (4.3 - PRE - 2.14 x 0.18) / 2.14 h.
   - For f4, train_job.sh takes the fold as its 2nd argument: add a fold argument to `submit` and use a new job name.
3. **Placeholder:** once `ld2.py joblogs rsna-d2-f3-s1044 20` shows `LAUNCH` and iteration lines, run `mkdir ~/fleet_runs/f3_s1044_ttr_frac75` (EMPTY placeholder).
4. **Monitor:**
   - `ld2.py job rsna-d2-f3-s1044` (status, `total_cost`);
   - `ld2.py joblogs rsna-d2-f3-s1044 30` (`COST_GUARD` lines show the projection);
   - `ld2.py balance` after the first hour.
5. **When it is Completed:** `bash ~/lightning_d2/fetch_place.sh 3 rsna-d2-f3-s1044`. It downloads, checks md5 against the in-job manifest, checks the closing line, the counts (1+1 best, 2 top oof, 32 epoch oof) and the ckpt args, moves the run into the empty placeholder and writes `~/fleet_runs/lightning_d2_f3_s1044_ttr_frac75_receipt.json`.
6. **Finish:** `ld2.py studios`, expecting zero running.

## Files

- **Box `~/lightning_d2/`:**
  - `ld2.py`: SDK helper (balance, studios, exec, bg, upload, start_cpu, stop, submit, job, joblogs, jobstop, artifacts).
  - `stage.sh`, `fetch_place.sh`.
  - `studio/{setup_env,unpack_check,train_job}.sh`.
  - `meta/box_parity_ttr224.json`.
  - `logs/`, including `timeline.txt`.
  - `stage/`: 12 GB of tars, deletable.
- **Laptop:** `session_2026-10-09/lightning_d2/`, same scripts.
- **Lightning:**
  - Studio `rsna-d2-nb` (Nebius, stopped): `~/rsna` (inputs, md5-verified), the venv, `~/rsna_d2/{kit,meta,PARITY_OK}`.
  - Studio `scratch-studio-devbox` (free, stopped): a verified copy of `~/rsna`. Its venv is broken by the cache clean and is not used.
  - Job `rsna-d2-dry1` (completed).
- Credentials (`~/.lightning/credentials.json`) were never printed, copied or moved; long tokens are redacted in all tool output.
