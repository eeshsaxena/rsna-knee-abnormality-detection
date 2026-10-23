# Vast offers (TTR288 frac75 f1-f4 since 2026-10-09; the 320 snapshot is kept below)

## 2026-10-09 01:15 IST re-rank for 4 TTR288 folds (V100 fold = 5.8 h; `--v100_fold_h 5.8`)

| offer | gpu | n | $/h | ver | dc | where | vCPU | RAM | rounds | fold h (MODEL) | run h mid/worst | $ mid/worst |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 36643047/49 (same UK machine, the slice id rotates) | RTX 4090 | 2 | 0.913 | V | DC | United Kingdom | 40 | 126 | 2 | 1.93 | 4.8/5.7 | 4.39/5.27 |
| 29390213 | RTX 3090 | 4 | 1.096 | V | DC | Finland | 32 | 252 | 1 | 3.31 | 4.2/5.0 | 4.65/5.56 |
| 36643048 | RTX 4090 | 4 | 1.766 | V | DC | United Kingdom | 80 | 252 | 1 | 1.93 | 2.8/3.3 | 5.04/5.90 |
| 46407649 | RTX 4090 | 4 | 1.567 | V | -- | Japan | 64 | 252 | 1 | 1.93 | 2.8/3.3 | 4.48/5.24 |

Choice: **2x RTX 4090, UK, verified datacenter, $0.913/h**. Lanes: GPU0 runs f1 then f3; GPU1 runs f2 then f4.
- The box's own f0 ends ~07:05, so every row above completes the 5-fold set at about the same time. The cheapest
  verified datacenter row therefore wins.
- 4x 3090 Finland was the runner-up. It was rejected because 32 vCPU for 4 loaders risks CPU starvation.


Offers are volatile (ids appear and vanish within hours). Re-rank right before the launch with
`bash vast_launch.sh --search --folds "2 3 4"` (same query, same ranker, `offers_rank.py`). Raw JSON of this snapshot:
`logs/o23.json`, `logs/o3f.json`, `logs/o4f.json`.

## Filters used

`vastai search offers -n` with: rentable, 2 to 4 GPUs of {RTX 4090, RTX 3090, RTX A5000, RTX A6000, L40, L40S,
A100 PCIe/SXM4, RTX 6000 Ada}, gpu_ram >= 20 GB, cuda_vers >= 12.1, disk_space >= 150 GB, inet_down >= 500 Mb/s,
reliability >= 0.98, priced with 80 GB of storage (`--storage 80`). `$/h` below is `dph_total` (GPU + that storage).
Costs also add the host's per-GB internet price for ~12 GB in and ~2 GB out (a few cents on these hosts).

Rule for "fits": worst-case run <= 7.0 h (it must end before the box's own fold f1, ~08:50 IST 10-09), worst-case
cost <= $8.10 (cap $8.50 minus the $0.40 fetch/destroy reserve) and enough CPU for the heavy-aug loader
(>= 12 vCPU per busy 4090/A100/L40-class GPU, >= 8 per 3090/A-series GPU). Rank: fits, then verified datacenter,
verified, deverified datacenter, others; then worst-case cost.

## Speed MODEL (per fold, vs our V100)

Anchor (MEASURED on the box, the running f0 probe): V100-FHHL 16 GB, thermally capped at ~870 MHz, 0.27 s/it on
bs 1 x 64 slice images of 320 px, TRAIN-ONLY 15.5 min/epoch, ~20-21 min/epoch with the raw+ema evals, 21 epochs =
7.3 h per fold. That is ~480 img/s in 224-px equivalents, i.e. about 55-65 % of a full-clock V100.

| GPU | speed vs our V100 (MODEL) | fold h mid | fold h worst (x0.8 for CPU/IO contention) |
|---|---|---|---|
| RTX 4090 | 3.0x | 2.4 | 3.0 |
| A100 SXM4 / PCIe | 3.3x / 3.0x | 2.2 / 2.4 | 2.8 / 3.0 |
| L40S / RTX 6000 Ada | 2.9x / 2.8x | 2.5 / 2.6 | 3.1 / 3.3 |
| RTX 3090 / RTX A6000 | 1.75x | 4.2 | 5.2 |
| RTX A5000 | 1.4x | 5.2 | 6.5 |

Ratios come from public ConvNeXt-T AMP (fp16) training throughput comparisons, not from a run of ours: treat
them as +-25 %. The trainer prints s/it after 200 iterations, so the real number is known ~5 min into training;
`vast_watch.sh` logs it and the lanes use the measured duration before starting a second fold on a GPU.

Run hours = 0.6 h setup (boot, 11.7 GB box -> instance copy, pip, md5 + parity) + rounds x fold h + 0.3 h fetch
and destroy. CPU (MEASURED on the box): the heavy-aug loader costs 0.40-0.46 core-seconds per study on a Xeon
Silver 4114, so a 4090 at ~12 it/s needs ~3-4 modern cores plus one for the main process.

## Plan A: Vast trains 3 folds (f2, f3, f4), the box trains f1 (the 5-fold set ~1 day sooner)

| offer | gpu | n | $/h | ver | dc | where | vCPU | RAM GB | down Mb/s | rel | cuda | rounds | fold h (MODEL) | run h mid/worst | $ mid/worst | fits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 29390213 | RTX 3090 | 4 | 1.096 | V | DC | Finland, FI | 32 | 252 | 694 | 0.999 | 12.8 | 1 | 4.17 | 5.1/6.1 | 5.59/6.73 | yes |
| 53298353 | RTX 3090 | 4 | 0.642 | V | -- | British Columbia, CA | 32 | 252 | 1363 | 0.985 | 13.3 | 1 | 4.17 | 5.1/6.1 | 3.31/3.98 | yes |
| 47360551 | RTX 3090 | 4 | 0.764 | V | -- | California, US | 128 | 252 | 789 | 0.988 | 13.2 | 1 | 4.17 | 5.1/6.1 | 3.97/4.77 | yes |
| 53860966 | RTX 4090 | 2 | 0.796 | V | -- | Brazil, BR | 36 | 63 | 616 | 0.981 | 12.8 | 2 | 2.43 | 5.8/7.0 | 4.63/5.59 | yes |
| 38922997 | RTX 4090 | 2 | 0.815 | V | -- | Taiwan, TW | 38 | 189 | 754 | 0.998 | 12.8 | 2 | 2.43 | 5.8/7.0 | 4.83/5.83 | yes |
| 38923002 | RTX 4090 | 4 | 1.615 | V | -- | Taiwan, TW | 76 | 378 | 754 | 0.998 | 12.8 | 1 | 2.43 | 3.3/3.9 | 5.52/6.50 | yes |
| 54822069 | RTX 4090 | 3 | 1.662 | V | -- | Norway, NO | 55 | 81 | 874 | 0.995 | 13.0 | 1 | 2.43 | 3.3/3.9 | 5.73/6.74 | yes |
| 36643031 | RTX 4090 | 2 | 1.393 | V | DC | United Kingdom, GB | 40 | 126 | 7943 | 0.989 | 13.0 | 2 | 2.43 | 5.8/7.0 | 8.03/9.72 | NO (cost) |

Only verified datacenter offer that fits Plan A at snapshot time: **29390213**. No verified 3-GPU datacenter offer
existed; the 3-GPU 4090 offers were non-datacenter (Norway).

## Plan B: Vast trains 2 folds (f3, f4), the box trains f1 and f2 (~15 h sooner)

| offer | gpu | n | $/h | ver | dc | where | vCPU | RAM GB | rel | rounds | run h mid/worst | $ mid/worst |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 36643031 | RTX 4090 | 2 | 1.393 | V | DC | United Kingdom, GB | 40 | 126 | 0.989 | 1 | 3.3/3.9 | 4.68/5.53 |
| 29390213 | RTX 3090 | 4 | 1.096 | V | DC | Finland, FI | 32 | 252 | 0.999 | 1 | 5.1/6.1 | 5.59/6.73 |
| 38922997 | RTX 4090 | 2 | 0.815 | V | -- | Taiwan, TW | 38 | 189 | 0.998 | 1 | 3.3/3.9 | 2.85/3.35 |

## Recommendation

1. **Plan A on 29390213** (4x RTX 3090, Finland, verified datacenter, reliability 0.999, 32 vCPU, 252 GB RAM,
   $1.096/h): f2, f3, f4 on GPUs 0-2 in one round (GPU 3 comes with the offer and stays idle). MODEL rental
   5.1 h ($5.59), worst 6.1 h ($6.73). Vast finishes ~06:50-07:50 IST, before the box's f1 (~08:50), so the 5-fold
   320 set is complete ~08:50 IST 10-09 instead of ~06:45 IST 10-10 (box alone): ~22 h sooner.
2. If 29390213 is gone at launch: the top verified datacenter row of `--search` that fits; if none fits Plan A,
   either a verified non-datacenter Plan A row (e.g. 38922997 2x 4090 Taiwan, two rounds, $4.83/$5.83, or
   54822069 3x 4090 Norway, one round, $5.73/$6.74), or Plan B on a verified datacenter row (36643031 UK 2x 4090,
   $4.68/$5.53, set ~16:10 IST 10-09). Datacenter vs non-datacenter is a data-rule judgement for the main session /
   Eesh: competition data is wiped and the instance destroyed at the end either way, but a datacenter host is the
   lower-exposure choice.
3. Avoid: unverified/deverified hosts, < 48 GB RAM where avoidable, and offers whose vCPU is below the rule (the
   loader would starve the GPUs and the MODEL hours would not hold).

Every number above is MODEL except the V100 anchor and the CPU cost (MEASURED on the box).
