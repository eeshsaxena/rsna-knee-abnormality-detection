# PREREG_RAPTOR_KERNELS (written 10-13 ~21:20 IST, before any rented-seed gold58 number was read and before any public read of a
# kernel containing our seeds). No em dashes.

## Kernels (builder `raptor/kaggle/build_hr7rs.py`, stage-3 proof `raptor/kaggle/sim_stage3_rs.py` A-D PASS on both)
- **HR7-RS** = HR7-RR v2 (`hr7rr/kernel_hr7rr`, sha 18cb5a27) + stage 1c (our seeds) + stage-3 Raptor_mix with three members.
- **HFR7-RS** = HFR7-RR v1 (`hr7rr/kernel_hfr7rr`, sha ca3da2fb) + the same two changes (fleet F at .2 and HF cell unchanged).
- Stage 1c: the r384 scored source with only the checkpoint preamble / ARMS / per-file sha / mean-of-seeds raw dump changed; same
  serve path (d96 fastread, 320 crop, 94 windows, anatomical mirror, fp16), own process, RS_TIMEOUT_S = 60 min cap,
  RAPTOR_SEEDS_FALLBACK on any failure = the parent submission byte for byte (sim cases B, C).
- Seeds: the rented seeds that PASS `PREREG_RAPTOR_RENTED.md` (all of its legs), weights = their swa.pt unchanged, dataset
  `eeshsaxena/rsna-knee-raptor-seeds` (private, RSNA-only weights; main creates it after the gate). "ours" = the mean of their
  probabilities.

## Blend (fixed now)
- Raptor_mix = per-label average-tie percentile ranks, **r384 .45 / r224 .25 / ours .30**, renormalised over the members available
  for a study (r224 or our seeds failed on it); r384-failed studies unchanged from the parent rule. Then the unchanged stage 3
  (R7 .5 / Raptor_mix .5, OA per-class .5) and, for HFR7-RS, the unchanged HF cell (F at .2).
- If the r224 stage falls back (RR_FALLBACK), the mix is r384 .45 / ours .30 renormalised (sim case D).

## Commit gate before any submit (main submits)
- Commit run on the 3 visible studies: RS_STAGE_DONE and RAPTOR_SEEDS_MIX_OK printed (no RAPTOR_SEEDS_FALLBACK), RR_MIX_OK,
  NOTEBOOK_COMPLETE; submission == an independent recompute from the commit's own raw csvs (rr_check-style); for HFR7-RS also
  HCHECK PASS (h_check.py --kind hf --wf 0.2 --members 20).
- Runtime: stage-1c seconds on the commit x (hidden studies / 3) is NOT a valid scale (prep and model load dominate on 3 rows);
  the hidden add-on is estimated from per-study GPU time in the stage log; HFR7-RS total hidden must be <= 6.5 h by that estimate.

## Reads
- Public reads follow the 10-13 band-position rule (a new submission shows only if it beats the team best). One kernel change per
  submission: HR7-RS is read against HR7-RR, HFR7-RS against HFR7-RR.

## AMENDMENT 1 (10-13 23:30 IST, still before any rented-seed gold58 number was read)
- Hidden-runtime constraint on stage 1c: every seed is one more arm of 94 x 2 mirrored windows per study. MODEL from the r384 stage
  (~25 min hidden for one arm incl. ~5-8 min DICOM prep on 4 CPUs, ~1,200 studies): ~15-18 min GPU per arm -> 4 arms ~70-80 min,
  above the 60 min stage cap and the 6.5 h HFR7 total. Rule (fixed now): the kernel carries the k best passing seeds by their
  training-geometry SWA gold58 (the trainer's "=== SWA" line), k = the largest number whose estimated hidden stage-1c time is
  <= 45 min, estimated as prep 8 min + k x (per-arm hidden GPU minutes), the per-arm figure = 18 min unless a commit-log
  measurement (per-study arm time on T4 x 1,200) gives a different number. Default k = 2. RS_TIMEOUT_S stays 60 min.
- "ours" = the mean of the probabilities of those k seeds; weights r384 .45 / r224 .25 / ours .30 unchanged.
