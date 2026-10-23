# A20 validation pre-registration (written 10-19 ~23:05 IST, BEFORE any of the results below were seen)
Eesh 10-19 ~23:00: "do offline validation, use uni box and my own gpu if you like, then push the 2 if they are strong".
Entries: eoa20 (c4 + E-OA a .2, kernel_v224r6c4eoa20j sha d40f43a4) and eoa10 (a .1, kernel_v224r6c4eoa10j sha c1242740).

## Validation checks
- V1 REPLICATION (box GPU): oai_eoa_check.py on 2 more OAI-free RSNA-only models on the 2,399 OAI expert-read knees:
  f0_s1044_ttr_mix50 raw (different seed + labels) and f0_s1045_ttr288frac75 raw (288 px). Teacher 1 (s1043 frac75, 5 folds) = SUPPORT already.
- V2 TRANSFER (independent skeptic agent): estimate of the hidden-set delta when labels are coupled at the RSNA expert level
  (gold58/74 OA phi ~.24-.40) and with c4's own OA model coupling.

## "Strong" rule (fixed now)
- PUSH BOTH (eoa20 + eoa10): V1 gives OA-mean delta > 0 with P>0 >= .90 at BOTH a .1 and a .2 for BOTH new models (primary proxy P_FT),
  AND V2 does not estimate a negative hidden delta at a .2.
- PUSH eoa10 ONLY: V1 as above, but V2 estimates a .2 negative (over-decorrelation at RSNA coupling) while a .1 stays >= 0.
- HOLD (push nothing, report to Eesh): V1 fails on either model at a .1, or V2 estimates a .1 negative.
- Submits: each pushed entry is submitted only after 05:30 IST 10-20 AND on probe_check_eoa.py PASS 17/17. Read R1 vs c4 57343435.

## RESULT (10-19 ~23:35 IST)
- V1 REPLICATION: SUPPORT for both new OAI-free models (trick/oai_check/g1_replication_box.log), primary P_FT:
  - f0_s1044_ttr_mix50: a .1 +.00498, a .2 +.00859 (P>0 1.000, 3/3 proxies, a -.1 control -.00581).
  - f0_s1045_ttr288frac75: a .1 +.00531, a .2 +.00929 (P>0 1.000, 3/3 proxies, a -.1 control -.00610).
- V2 TRANSFER (skeptic_eoa.md): sign positive, size shrinks at RSNA expert coupling (phi ~.32): expected hidden macro a .1 ~+.0005 (80% [-.0002, +.0012], P ~.80), a .2 ~+.0007 ([-.0007, +.0021], P ~.70). Not negative at a .2.
- RULE -> **PUSH BOTH** (eoa20 + eoa10).
