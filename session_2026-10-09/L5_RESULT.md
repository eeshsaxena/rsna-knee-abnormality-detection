# L5 result: all-24-slot TTR288 probe (f0, s1045, frac75, 21 ep) vs Q4 f0

**Status: PENDING.** This section was written 2026-10-09 ~11:35 IST, before the L5 run started (pre-registration).

## Gate (pre-registered, RESEARCH_TO_TOP_10-08.md L5)

PASS iff both clauses hold:

1. last-3 raw val mean (epochs 19-21) of L5 >= Q4 f0 last-3 + .002
2. gold58 not lower than Q4 f0 by more than .003

Field definitions:

- "val" is the trainer's `val AUC vs llm_labels_v4_blend.csv` on the `ep N raw` lines, the same field `g320_gate.py` reads.
- The primary reading of clause 2 is gold58 at the best-raw-val epoch. That is the `g320_gate.py` convention, and it is
  the epoch whose `*_best.pt` ships.
- The secondary reading is the last-3 gold58 mean. It is reported only. If the two readings disagree, the verdict is
  flagged "reading-sensitive".
- Both logs need 21 raw epoch lines and the closing line, else the verdict is INVALID.
- Scorer: `session_2026-10-09/vastL5/score_l5.py`.

## Control: Q4 f0 (`~/fleet_runs/f0_s1045_ttr288frac75`, 16 train slots, box V100), MEASURED

| metric | value | L5 bar |
|---|---|---|
| last-3 raw val mean | .8888 (.88877) | >= .8908 (.89077) |
| best raw val | .8893 @ep15 | |
| gold58 @best | .9133 | >= .9103 |
| last-3 gold58 (secondary) | .9152 | >= .9122 |
| closing line | raw .8893 / ema .8875 | |

The only argv difference between the two runs is `--train_slices 0` vs `16`, plus the out dir.

## Run (Vast, kit `session_2026-10-09/vastL5/`, a copy of `vast320` retargeted to L5)

- **Instance:** 54973344 (label rsna-ttr288frac75s24-1009112135), offer 45598880.
  - 1x RTX 3090 24 GB in Poland: verified, not a datacenter, 12 vCPU, 46 GB RAM, rel .998.
  - Price: $0.211/h with 60 GB storage included.
  - No verified datacenter offer fit. The one DC 3090 (CZ, $0.303/h) was gone by launch time. The other DC offers
    (L40S, 4090, A100 at $0.82-1.10/h) have a worst case of $3.60-4.94, which is over the $3.10 spend line.
- **Money:** credit at start $4.009, so the budget is min($3.50, credit - $0.50) = $3.50.
  - MODEL cost: 6.1 h / $1.40 mid, 7.3 h / $1.67 worst.
  - Watcher cap: 20:21 IST (9.0 h). On-instance guard self-destroy: 20:51 IST (self-test http 200).
- **Timeline:**
  - Rented 11:21 IST.
  - Setup 11:24-11:41. The store copied at 15 MB/s. 4407/4407 store md5 identical, and every parity.py hash and count
    was identical: contract, select, jpeg_fp, model_sha, fold counts (3466/867), decode probe, backbone.
- **Box placeholder:** `~/fleet_runs/f0_s1045_ttr288frac75s24` was created EMPTY at 11:41, after setup was verified.
  The runner shows `L57 WOULD-REFUSE(out exists)`. queue.txt was not edited.
- **Training:** started 11:41 IST. The argv is token for token the L57 queue line (argv-check IDENTICAL, 40 tokens);
  the log header shows `train_slices 0`.
  - MEASURED 0.44 s/it at GPU util 99%, CPU load 2 of 12.
  - **GPU memory: nvidia-smi shows 20.7 GB.** This counts the caching allocator's pool, so it is an upper bound on what
    the run needs. It is still above the V100's 16 GB, so this argv (bs 2, gc0) may not fit on the box. Not tested.
  - ETA: the fold ends ~16:55 IST, then fetch + destroy.

## RESULT (scored 2026-10-09 17:14 IST, laptop, `score_l5.py` on both box logs)

**VERDICT: FAIL.** 24 train slots do not help at 288; close the slot-count lever.

| metric | Q4 f0 (16 slots) | L5 f0 (24 slots) | delta | bar |
|---|---|---|---|---|
| last-3 raw val | .8888 | .8864 | -.0024 | >= +.002: NOT MET |
| best raw val | .8893 @ep15 | .8866 @ep18 | -.0027 | |
| gold58 @best | .9133 | .9092 | -.0041 | >= -.003: NOT MET |
| last-3 gold58 (secondary) | .9152 | .9085 | -.0067 | NOT MET |
| gold74 @best | .9099 | .9046 | -.0053 | |

- Instance 54973344 destroyed 17:10:39 IST after FETCH + VERIFY into `~/fleet_runs/f0_s1045_ttr288frac75s24` (receipt `~/fleet_runs/vastL5_f0_s1045_ttr288frac75s24_receipt.json`). Spend ~$1.73; credit $2.27 after.
- The checkpoint stays on the box unused (not a fleet member).
