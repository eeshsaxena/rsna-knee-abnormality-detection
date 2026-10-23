# VAST_RAPTOR: RSNA-only Raptor seeds on a rented GPU (path b). No em dashes.

Status: **LAUNCHED 10-13 20:57 IST** (main 20:10: Eesh added $25 and renewed 2FA; no wait for the box smoke, the rental runs its
own bf16 smoke first). 4 seeds (43-46) training in parallel on instance 55762630 (4x RTX 4090, Taiwan, $1.898/h).

## Rules (main 10-13)
- RSNA ONLY: the rental receives code (train_raptor.py, raptor_ref.py, unpack_d96.py, setup / lanes scripts), the two RSNA csvs
  (`labels_raptor_rsna_blendjev_sol.csv` md5 f1fa9d51, `gold74_eval_set.csv` md5 a0ff70f7) from the LAPTOP kit, and the RSNA
  d96 corpus straight from the Kaggle prep-kernel outputs (signed URLs; the same bytes as `~/raptor_store/rsna`: every volume is
  sha256-checked against the Kaggle receipts, which equal the box copy). The box NEVER sends data to the rental (it only pulls
  results and touches one status file), so no OAI byte can reach it.
- Fail-loud OAI checks: kit csvs grepped for `oai_` before renting; on the instance `setup.sh` asserts that no file name, index id,
  path or label cell under /workspace/rsna mentions OAI (and that the index has exactly 4,407 RSNA studies); the trainer runs
  with `--forbid_oai 1` (asserts no OAI id / path / label file; tested on the box: passes on the RSNA index, fails on OAI).
- Budget: hard total cap $23 incl. data-in and failed attempts, ~$2 credit buffer (config.sh CAP_USD 23, FLOOR_USD 2,
  RESERVE_USD 1). Offers: verified, not China, reliability >= .98, >= 8 effective vCPU per GPU, >= 500 Mbps down, CUDA >= 12.1.
  Searches: 2 narrow queries, limit 25 each.
- Seeds 43 / 44 / 45 (box run = 42), same recipe minus OAI rows (4,349 RSNA rows), 16 epochs, bf16 autocast (recipe dtype).
- Fetch from the BOX (lesson 10-10): `box_fetch.sh` (setsid nohup on the box) pulls each finished seed with sha256 verification
  into `~/raptor_store/rented/s<seed>/`; partial pulls from guard deadline - 45 min; then FINAL_FETCH_OK on the instance, whose
  on-instance guard self-destroys 15 min later or at the hard budget deadline. Laptop `vast_watch.sh` = backup destroy only.

## Kit (`session_2026-10-13/raptor/vast/`)
| File | Role |
|---|---|
| config.sh | settings + helpers (vast_raw wrapper from vastL5, state.env) |
| offers_rank.py | filters + MODEL ranking (s/step from the box smoke's V100 s/step / GUESS speed ratio; 3090 = 1.95x MEASURED 10-09) |
| vast_launch.sh | `--search [V100_SPS]`, `--dry-run OFFER`, `OFFER`: rent -> guard -> box key -> kit + fresh URLs -> setup -> lanes -> box fetch loop -> backup watcher; logs measured s/step 15 min after the lanes start |
| remote/setup.sh | pip (timm 1.0.29), 33 tar parts (sha = manifests), unpack (sha = receipts), OAI assertion, pretrained weights |
| remote/lanes.sh | one lane per GPU, seeds round-robin, deadline-aware, SHA256SUMS + status/s<seed>.rc |
| remote/guard.sh | hard-deadline / FINAL_FETCH_OK self-destroy (vastL5 guard, Raptor paths) |
| box_fetch.sh | BOX-side pull + verify loop |
| vast_watch.sh, vast_destroy.sh | laptop backup destroy + spend record |

## One-command launch (after main confirms credit >= $20 and 2FA)
1. `bash vast_launch.sh --search <V100 s/step from the smoke>`  (read the table; pick the cheapest `fits=yes`, DC preferred)
2. `bash vast_launch.sh --dry-run <id> <sps>` then `bash vast_launch.sh <id> <sps>`

## Plan numbers (MODEL, to be replaced by measurements)
- Per seed: 4,349 x 16 = 69,584 study-steps. 4090 ~0.27 s/step (if the V100 measures .65) -> ~5.3 h; A100 ~0.23 -> ~4.5 h.
- 3 x 4090 (3 seeds parallel) ~$1.0-1.3/h x ~6.5 h = ~$7-9; 2 x 4090 (2 rounds) ~$0.7/h x ~12 h = ~$8.5; 1 x A100 80GB
  (3 rounds) ~$1.0-1.4/h x ~15 h: exceeds MAX_RUN_H 12 h, so only 2 seeds.

## Launch / spend / results log
- 20:12 credit $25.44 (2FA OK). Searches: 2 x limit 25. Note: raw search JSON has `verified: null` (the query enforces it); the ranker was fixed to accept it.
- 20:14 attempt 1: offer 52380868, 1x A100 SXM4 40 GB, California, $0.501/h (cheapest MODEL cost $7.7 for 3 sequential seeds).
  Instance 55758866 ran but sshd never came up: the container was still in `apt-get update` (mirror.yandex.ru) after 20 min.
  Destroyed 20:37 (the CLI now needs `-y`; vast_destroy.sh fixed). Spend $0.19.
- 20:38 attempt 2: offer 42318231, **4x RTX 4090, Taiwan, $1.898/h**, rel .998, 112 vCPU, 252 GB RAM, 1.8 Gbps (MODEL: 4 seeds in one
  round, ~$12.6). Instance **55762630** up in ~3 min; guard self-test http 200; hard self-destroy 10-14 07:57 IST (budget cap $22.8 - reserve).
  Bug fixed live: the instance's authorized_keys had no trailing newline, so the box key was glued onto the account key line
  (box ssh "Permission denied"); repaired on the instance, launch script now prepends a newline and drops empty lines.
- 20:44-20:53 setup: pip OK (torch 2.5.1, timm 1.0.29, bf16 True); 33 tars (32 GB) in 7.7 min; PARTS_VERIFIED 33/33; unpack
  4,407/4,407 sha OK (49 s); **OAI_ASSERT_OK**; pretrained blob sha 91570c7a... = the box's HF blob.
- 20:54-20:57 **RENTAL SMOKE OK** (bf16, 600 steps, GPU 0): 0.109 s/step, peak mem 9.2 GB, 0 nonfinite, 0 skipped, gold58 eval 5 s,
  SWA + strict serve-load at 320 OK.
- 20:57 lanes: seeds 43 / 44 / 45 / 46 on GPUs 0-3; measured **0.108-0.112 s/step** in parallel (GPU 97-99 %), i.e. 2.5x faster
  than modelled. Per seed 69,584 steps ~2.1 h + 16 evals -> **ETA ~23:10 IST**. Box fetch loop pid 646436 started; laptop backup watcher started.
- Projected spend: ~3.0 h x $1.898 + data-in ~$0.31 + attempt 1 $0.19 = **~$6.2** (cap $23).
- 23:25-00:22 all 4 seeds rc 0; SV gold58 scored on the rental (sv_watch.sh); box fetch loop pulled each seed sha-verified
  (~7 MB/s, 5.5 GB per seed incl. epoch ckpts); FINAL_FETCH_OK 00:22.
- **00:37 instance 55762630 self-destroyed by the on-instance guard** (FINAL_FETCH_OK + 15 min; `vastai show instances` = []).
- **Final spend: $8.21** (credit $25.44 -> $17.23): attempt 1 $0.19 + instance 55762630 ~4 h at $1.898/h + storage / data-in.
  Remaining credit $17.23 (cap left vs $23: ~$14.8).
- Gate: RAPTOR_RENTED_GATE PASS (RAPTOR_RENTED_RESULTS.md). Kernel seeds k = 2 (s46, s45), dataset folder `raptor/ds_raptor_seeds/`.
