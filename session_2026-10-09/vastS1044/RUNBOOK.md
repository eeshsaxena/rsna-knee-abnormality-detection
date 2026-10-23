# vastS1044: TTR224 frac75 SEED 1044 folds 1-4 on Vast (status + how to retry)

Kit = copy of `session_2026-10-08/vast320` with its own `state.env`, box dir `~/vastS1044`, box known_hosts
`~/.ssh/known_hosts_vastS1044` (L5's files are never touched). Hard cap $2.30 per run.

## 2026-10-09 attempt: ABORTED, no folds trained

- Instance 54991838 (offer 53871122): 2x RTX 3090, Jiangsu CN, verified, $0.321/h with 45 GB.
  Rented 14:11:23 IST; wiped and destroyed 15:17:12 IST. Cost about $0.35. Credit went from $3.31 to $2.73; about
  $0.23 of that drop was L5.
- **Cause 1 (kit bug, fixed).** A sed cleanup turned the line continuation in `config.sh` `ix()` into a literal `$`.
  Every laptop-to-instance ssh then failed, so the launch died at its ssh step and the instance sat idle until the
  session restart. `ix()` is now byte-identical to vast320's.
- **Cause 2 (CN network).** PyPI wheels arrived truncated (sha256 mismatch). The aliyun mirror ran at about 140 KB/s,
  and a Sweden test download at 0.8 MB/s. S2 was still not done at 15:15. Training could not start before about
  15:45-16:00, so round 2 would have ended near 20:50, past the $2.30 time cap (20:13). f1-f4 could not fit, so the
  instance was destroyed, per the coordinator's rule.
- **Box:** no `f{1..4}_s1044_ttr_frac75` placeholder was ever created; only the Kaggle f0 exists.
- **L5:** never touched; its watcher was left running.

## Market facts (10-09 13:39-14:10)

- **Cheapest acceptable offer:** about $0.32-0.36/h for a verified 2x3090 with >= 8 vCPU/GPU and >= 500 Mbps.
  Data-in adds $0.2-0.3 (16 GB at $0.0156-0.02/GB).
- **MODEL fold:** 2.4 h on a 3090. The box V100 MEASURED 3.52 h; the ~1.9 min eval per epoch is CPU-bound.
  4 folds in 2 rounds: mid case about $2.0-2.2.
- **When $2.30 fits:** only on a non-CN offer at about <= $0.34/h. Such offers appeared twice
  (Missouri 54888729, California 52756389) and were gone within minutes.

## 2026-10-09 15:22 IST: polling (coordinator rules)

- `poll_launch.sh` (background) runs until 10-10 02:00 IST. It accepts a verified 2x RTX 3090 offer at <= $0.34/h
  (`dph_total`) with reliability >= 0.98, NOT CN, >= 8 vCPU/GPU and >= 500 Mbps.
- **Run cap $2.30.** A failed attempt's cost is booked in `logs/run_spent.txt` and lowers the next cap.
- **Budget** = min(cap, credit - $0.15 - L5 hold-back). The hold-back is L5 $/h to 17:15 + $0.10, applied only
  while L5 lives.
- **Launch rule:** it launches even if only 2 folds fit (the lanes skip f3/f4). It then runs setup, starts training
  and exits.
- **`raise_budget.sh`** (called by the watcher once L5 is gone): lifts the budget to
  min(cap, launch credit - L5 since launch - $0.15). It moves the watcher cap, the on-instance guard deadline (the
  guard is restarted) and the lane deadline (lane scripts restarted, trainers untouched).
- **`vast_watch.sh`** (background) is the 2FA keepalive before a launch. While `logs/poll.active` exists it idles
  between attempts. It finalises THIS run if the credit nears L5's stop line while L5 lives.
- **Attempt 1:** 55012605 (offer 51737581, US) was stuck in docker pull, destroyed after 20 min at ~$0.06.
  The cost is booked in `logs/run_spent.txt` and the offer is in `logs/offer_blacklist.txt`. L5 itself was destroyed
  17:10:39.
- **Placeholders (coordinator 18:45):** `vast_setup.sh` S6 reserves only the round-1 folds (f1, f2).
  `vast_watch.sh` creates the f3/f4 placeholder only when a lane actually starts that fold. Skipped folds stay
  unreserved (for another free GPU later).
- **After a session restart:** re-run `bash vast_watch.sh` and `bash poll_launch.sh` (both with Bash
  run_in_background). The poller only adds a launch if no live instance is recorded. If one IS recorded, run
  `vast_launch.sh` (resume), `vast_setup.sh` and `vast_train.sh` by hand.

## 2026-10-09 23:23 IST: REVISED PLAN (coordinator), ONE GPU, sequential folds

- **2FA:** the session expired ~23:10 and Eesh renewed it ~23:2x. No instance on the account; credit $2.2498.
- **Search quota exhausted.** Vast answers every search with
  `429 "Daily search row quota exhausted"`. The wide 2-GPU polling (limit 400 every ~50 s) used up the daily row
  quota at an unknown time between 18:43 (last good search) and 21:02. So the "no acceptable offer" lines after that
  were BLIND.
- **Poller fixes:** `poll_launch.sh` now uses a narrow query (`num_gpus=1`, `dph_total<=0.25`, GPU list,
  `cpu_cores_effective>=8`) with `--limit 25` every 2 min. On a 429 it backs off 15 min. `vast_launch.sh` re-checks
  the offer with the same narrow query (`LAUNCH_QUERY`/`LAUNCH_LIMIT`).
- **Offer:** ONE verified, non-CN RTX 3090 / 3090 Ti / 4090 / A5000 / A6000 / A40 at <= $0.25/h incl. storage, with
  >= 8 vCPU, >= 500 Mbps and rel >= 0.98.
- **Budget:** min(run cap $2.10 - failed attempts in `logs/run_spent_1gpu.txt`, credit - $0.15).
- **Fold order:** sequential on GPU 0: f1 -> f2 -> f3 -> f4.
  - lane.sh starts the next fold only if (measured previous fold x 1.03; x 1.10 for f4) ends 6 min before the
    watcher cap, which is budget-derived: (budget - $0.03 - data-in) / $/h - 0.15 h.
  - Skipped folds stay unreserved.
- **MODEL fold times:** 3090 2.39 h, 4090 1.68 h, A5000 2.93 h. At $0.21/h on a 3090 that is ~3 folds
  (f1-f3 ~7.7 h, ~$1.85).
- **Placeholders:** setup reserves only f1. vast_watch.sh creates the f2/f3/f4 placeholder only when the lane starts
  that fold, and fetches + verifies each finished fold at once (incremental fetch). It destroys the instance on
  finish or cap.
- **Polling window:** until 10-10 12:00 IST.

Exact relaunch (Git Bash, this folder; both with Bash run_in_background):
```
touch logs/poll.active; bash vast_watch.sh >> logs/watch_run.out 2>&1
touch logs/poll.active; bash poll_launch.sh >> logs/poll_run.out 2>&1      # default END 10-10 12:00, MAXDPH 0.25, PFOLDS "1 2 3 4"
```
Kill only verified PIDs: `ps -ef | grep -E "[b]ash poll_launch.sh|[b]ash vast_watch.sh"`, then check
`readlink /proc/PID/cwd` is this folder.

## Retry

1. `bash vast_watch.sh` (Bash run_in_background). This runs the 2FA keepalive and becomes the watcher at launch.
2. `bash poll_launch.sh 60`. It polls every 30 s and launches the first 4-fold-feasible non-CN offer under the cap
   and lane-admission rule, with an L5-safety check.
3. `bash vast_setup.sh`. It covers S1-S5 parity and then creates the EMPTY placeholders. For a slow PyPI, set
   `PIP_MIRROR=<index url>`.
4. `bash vast_train.sh`. The watcher fetches, verifies (ckpt args vs the Kaggle f0 s1044 ckpt), wipes and destroys.

Alternative without Vast: `session_2026-10-09/kaggle_frac75c` (teammate Kaggle notebooks for exactly these folds).
