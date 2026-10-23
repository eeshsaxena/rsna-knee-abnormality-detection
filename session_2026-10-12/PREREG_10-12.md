# PREREG 10-12: public-blend lineup (H = heliosli Raptor + S6 rank blend; HF = H + our fleet F)

Written 2026-10-12 ~08:30 IST by Claude, BEFORE any of these kernels was pushed or submitted. No em dashes. Append results
in a separate RESULTS file.

## Why (what was seen first)

- LB 07:35 (5,797 teams): we are .951 rank 788. Public notebooks at .954 fill ranks 159-695; .955 = r110-158; .956 = r85-109.
  Medal cuts by rule: silver ~r290, bronze ~r580; ties rank by submission time, so a new .954 lands ~r695 (no medal).
- `heliosli/blend-mine-in-private-lb-luck-to-all-upvoters` (Apache 2.0, posted 10-11): Best Score **0.956 (V1)**. It is a
  per-label rank blend Raptor .5 (the SAME nartaa .949 checkpoint as our N, sha 7e5315da...) + heliosli's ConvNeXt-small "S6"
  4-fold bundle .5 (datasets heliosli/rsna-knee-compact-r6cnxfsss v2 bundle 209a32c8b50a6b1c, heliosli/rsna-orthofoundation-
  fold0-epoch7-20261004). Our FN11 (N .6 + our F .4) = .951, so S6 is a much better partner for N than our F.
- Commit log (3 test studies): Raptor stage 50 s; S6 workers 49-74 s/study (decode-dominated); S6 controller deadline = notebook
  start + 8.3 h with fallback to Raptor ranks for unscored studies. **Hidden runtime unknown** (could be near the limit).

## Lineup (in order; each gated before submission)

1. **HV** = the heliosli notebook V2 verbatim (sha256 63e92116...), kernel `eeshsaxena/rsna-sub-hv`, same 3 datasets,
   GPU T4 x2, no internet. Purpose: (a) take the .956 display now; (b) MEASURE the hidden wall time (submit -> COMPLETE) to
   size the HF build. Commit gate: kernel COMPLETE, `NOTEBOOK_COMPLETE` printed, submission.csv 13 cols, 3 rows, all in [0,1],
   blend_receipt ours_failed 0 and raptor_failed 0.
2. **HF** (built only after HV's hidden wall time is known) = HV + our fleet stage F inserted AFTER the Raptor stage and BEFORE
   the S6 stage (S6 keeps its NB_T0 + 8.3 h deadline, so the notebook cannot overrun), final per-label rank blend
   **(1 - wF) x H + wF x F**, H = HV's own 2-way blend unchanged. F = our fleet members chosen by runtime (set D first, then E,
   C, E2 as time allows; RSNA-only weights from `eeshsaxena/rsna-knee-cnxt-fleet-k11w`). If F did not complete, the kernel
   writes H unchanged and prints HF_FALLBACK (no silent mixing).
   - Variants: **HF20 (wF .20)** and **HF30 (wF .30)**. Rationale: FN11 used F .40 against N alone; H is stronger than N, so a
     smaller F share. No offline gate is possible (Raptor and S6 are in-sample on train and gold); LB decides between HV, HF20,
     HF30. No weight fishing beyond these two values.
   - Runtime rule: HF is built only if (HV hidden wall + F slow estimate) <= 8.0 h; otherwise HF uses fewer members, or is
     dropped.
3. Final picks (10-22) consider the OAI ruling (thread 748276): H and HF both contain OAI-trained Raptor (as N in FN11).

## Not allowed
- Submitting HF variants other than wF .20 / .30 without a new prereg.
- Changing anything inside H (Raptor / S6 code, weights, the .5 / .5 rule) in HF.
