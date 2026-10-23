# implied2.md: public-implied model refit with all five E-OA reads + the A21 board (10-20 ~19:40 IST, Claude). TRACK B, RESEARCH ONLY.

Nothing was submitted, pushed, committed, versioned or downloaded. No dataset, no GPU job, no scheduled task. PREREG, HANDOFF,
PROGRESS, a20 and a21 were only read. Box use: 3 read-only CPU python runs piped over ssh stdin (`work/box/box_perlabel.py`:
a B 20 smoke, then P_FT and P_AREA at B 3000/regime), printing aggregates only (per (model, regime) means of resample-level
per-label AUC deltas, rho / q / phi means, pooled within-regime covariance). No OAI id, row or per-knee value left the box,
nothing was written there, no OAI CSV was printed. Everything else is local on public LB snapshots and those aggregates.
Tags: MEASURED (computed now, file named), CLAIMED (older note, file named), MODEL (arithmetic on stated assumptions),
GUESS (judgement). Units: macro AUC, "e-4" = 1e-4 macro, "band" = .001 of display. Intervals are [p10, p90]. gold58 is info only.

## 0. Bottom line

1. **What the A21 board says (MEASURED counts, `work/lbpos2.py` -> `lbpos2.log`; MODEL conversion, `work/lb_post2.py`).**
   - pw30 57425970 is the best-id at r140 (.959, 3 teams above it inside .959; .960 = r92-136).
   - Frozen cross-board ruler: the 4 .959 teams that sat above eoa20 at 05:53 UTC all have the same best submission at
     13:20 UTC. Exactly 2 of them are now below pw30 (05:53 r133 / r134 -> r141 / r143) and 2 above (r131 / r132 -> r137 / r139).
     So pw30 sits about halfway between eoa20 and the .960 line: quantile .49 [.24, .75] of that gap.
   - **pw30 - eoa20 = +0.5e-4 [0.2, 1.2] (LB-only) / +1.0e-4 [0.3, 1.9] (joint refit).**
   - **pw30 to .960 = 0.5e-4 [0.2, 1.3] (LB-only) / 0.75e-4 median, mean 0.9 [0.3, 1.8] (joint).** Two independent cross-checks
     agree with the LB-only number: A21 board alone +0.4e-4 [-0.3, +1.8]; adding the 16 non-frozen .959 teams as iid band draws +0.5e-4 [0.2, 1.3].
   - Pre-read calibration (MODEL, same refit code with the A21 data removed): median forecast pw30 - eoa20 was +2.1e-4; the read
     sits at the ~30th percentile (P(<= +0.5e-4) .29, P(<= +1.0e-4) .35). A modest downward surprise, not a model rejection.
2. **Unrounded public scores (joint refit, primary; display truncates; subtract .0005 if it rounds; gains do not depend on it):**

   | entry | unrounded (trunc) | in-band position | to .960 (e-4) | vs pw30 (e-4) |
   |---|---|---|---|---|
   | c4 57343435 | .95878 [.95867, .95888] | .78 of .958 | 12.3 [11.2, 13.3] | -11.3 [-12.8, -9.7] |
   | eoa10 57413380 | .95937 [.95926, .95946] | .37 of .959 | 6.3 [5.4, 7.4] | -5.4 [-6.3, -4.5] |
   | eoa30 57425223 | .95974 [.95954, .95990] | .74 | 2.6 [1.0, 4.6] | -1.7 [-3.4, -0.3] |
   | eoa20 57413368 | .95981 [.95966, .95992] | .81 | 1.9 [0.8, 3.4] | -1.0 [-1.9, -0.3] |
   | pw20 57425269 | .95982 [.95968, .95993] | .82 | 1.8 [0.7, 3.2] | -0.9 [-1.9, -0.2] |
   | **pw30 57425970** | **.95991 [.95982, .95997]** | **.91** | **0.9 [0.3, 1.8]** | 0 |

   - Censored orderings (MODEL): P(eoa30 > eoa20) .30 (variants .24-.59), P(pw20 > eoa20) .56 (.50-.62),
     P(pw20 > eoa30) .69. eoa30 - eoa20 -0.7e-4 [-2.3, +0.7]; pw20 - eoa20 +0.1e-4 [-0.9, +1.0].
   - Every variant's means are within +-.00005 of these (`work/ranges3.log`).
   - The joint refit puts eoa20 lower in its band (.81 [.66, .92]) than the LB-only fit (.90 [.75, .95]) because the curve
     prior wants pw30 - eoa20 > +0.5e-4. Both sit inside the LB-only interval; the frozen F18 ruler alone gave .81 (CLAIMED
     push/implied.md sec 1.2). Read the .960 distances as 0.5 to 0.9e-4 for pw30.
3. **What the refit changes (MODEL, `work/abc3.py` -> `abc3_primary.log`).**
   - Coupling regime unchanged: shift -0.17 sd [-1.0, +0.6] (was -0.16), c4 gap .128 [.081, .176], q_hidden .69, label phi .29.
   - Amplitude s down 9%: .82 [.58, 1.07] (was .90).
   - All three A21 entries came in ~0.7-1.0e-4 under their pre-read means (pw30 +2.1 -> +1.0, pw20 +0.8 -> +0.1,
     eoa30 +0.2 -> -0.7 over eoa20). That is a common shortfall along the a-slope (public noise corr .94-.98 across a), not a
     shape-specific failure.
   - Private-optimal shape (population argmax, posterior mass): PW20 .32, PW25 .30, PW30 .23, PW35 .12, PW40 .03
     (`work/extra3.log`). So pw30 is near, not at, the private optimum.
4. **Asked candidates vs pw30 (primary, P_FT; [p10, p90]; variant range of the mean in parentheses; full table sec 3):**

   | shape | public vs pw30 (e-4) | P(> pw30) | P(shows .960) | P(shows .958) | private vs pw30 (e-4) | P(private > pw30) |
   |---|---|---|---|---|---|---|
   | pw35 | -1.6 [-3.5, +0.2] (-2.2..-1.2) | .12 | .04 | .00 | -1.9 [-4.7, +0.9] (-2.5..-1.2) | .19 |
   | pw40 | -5.1 [-8.8, -1.6] (-6.3..-3.8) | .03 | .01 | .09 | -5.7 [-11.6, -0.1] (-6.6..-3.9) | .10 |
   | pw45 | -10.2 [-16.3, -4.4] (-12.2..-7.8) | .01 | .01 | **.58** | -11.1 [-20.5, -2.1] (-12.7..-8.0) | .06 |
   | eoa40 (S40) | -7.0 [-11.5, -2.8] (-7.6..-5.0) | .01 | .00 | .26 | -7.9 [-13.9, -2.1] (-8.4..-4.8) | .04 |
   | LP30 | -1.7 [-6.0, +2.6] (P_AREA -7.7) | .32 (P_AREA .01) | .23 (.00) | .01 (.34) | -0.5 [-5.3, +4.3] (P_FT -1.4..-0.2; P_AREA -7.3) | .45 (.01) |
   | ref: pw25 | +0.1 [-0.9, +1.1] (-0.2..+0.2) | .56 | .22 (.29 LB-only gap) | .00 | +0.6 [-2.0, +3.0] (+0.1..+1.0) | .61 |
   | ref: M20_LP30 | +0.1 [-1.7, +1.8] (P_AREA -2.3) | .51 | .29 (.34) | .00 | +0.3 [-1.7, +2.4] (P_AREA -1.9) | .59 |

   - LP30 = Medial OA untouched, Lateral OA and PF OA at symmetric a .3 (the 10-19 pre-registered F2 "LP" family, CLAIMED
     session_2026-10-19/ex/cand_eoa_curve.md). M20_LP30 = Medial sym .2 + LP30. If "LP30" means something else, edit
     `shape()` in `work/abc3.py` and re-run (40 s).
   - pw40 / pw45 / eoa40 are past the optimum in every variant; pw45 and eoa40 risk a .958 display. pw35 is mildly negative.
   - LP30 is a proxy coin flip. P_FT says Medial OA carries ~0 of the E-OA gain at a .3, so sparing it is cheap. P_AREA says
     Medial carries about half, so sparing it costs ~7e-4. gold58 (info only) sides with P_FT: LP30 would score .94366 there,
     +.0003 over pw30 (MODEL arithmetic on the per-label AUCs in a21/gate_a21_offline.log).
5. **Winner's curse (MODEL, sec 4).**
   - Picking the public max of the five E-OA entries overstates the winner's gain over c4 by +1.9e-4 (variants 1.5-3.0).
     For seven entries it is +2.3e-4 (adding pw25 + pw35) or +3.3e-4 (adding pw40 + LP30); for 3 entries +1.3e-4.
     Five INDEPENDENT entries with the same per-entry noise (sd ~5.5e-4 vs c4) would give ~6.5e-4: the correlation cuts the
     curse ~3.5x.
   - Regret (population-best member minus the pick): 0.9e-4 for 5 entries, 1.0-1.3e-4 for 7. The pick is the population-best
     member only half the time (5 entries) or ~30% (7).
   - Realised on our reads: pw30 is +11.3e-4 [9.7, 12.8] over c4 publicly but +10.0e-4 [0.8, 19.2] in expectation
     privately. That is a discount of 1.4e-4 (~12%; variants 0.7-2.8), almost all common-mode. Against eoa20 the discount is
     only +0.1e-4: public +1.0, private +0.9 [-2.6, +4.4], P(private pw30 > eoa20) .58. P(private pw30 > c4) .86 (.80-.92).
6. **10-21 slates under R1 (best-id = max unrounded), conditional on today's reads (MODEL, sec 5):**
   - Best simple slates:
     - pw25 alone: private of the pick +0.4e-4 vs keeping pw30 (variants +0.1..+0.7), P(.960) .22.
     - pw25 + M20_LP30: +0.35e-4 (+0.1..+0.6), P(.960) .40.
   - The asked five (pw35, pw40, pw45, eoa40, LP30) has negative private value under R1: -0.3e-4 (-0.5..-0.15).
     - It moves the best-id 41% of the time, mostly on public noise.
     - P(private down) .24 vs P(up) .17.
   - P(any entry shows .960) is ~.2-.3 for one good shape, ~.4 for two, ~.5 for five (each ~+.05 under the LB-only gap).
     A .960 display is worth ~3-4 places (r140 -> ~r137; CLAIMED push/implied.md sec 4 for the ~3-5 places per 1e-4).

## 1. Public LB inputs

### 1.1 Counts (MEASURED, `work/lbpos2.py` -> `lbpos2.log`)
- Snapshots: `lbprobe/api_lb_1020_eoaread2.json` (05:53 UTC, eoa20 r135) and `api_lb_1020_a21read.json` (13:20 UTC = 18:50 IST,
  pw30 r140). The c4 / eoa10 reads and their 04:17 / 05:52 snapshots are as in push/implied.md (CLAIMED).
- Bands at 13:20: .961 r67-91 (25), .960 r92-136 (45), .959 r137-208 (72), .958 r209-1590 (1,382).
  At 05:53: .960 r87-130 (44), .959 r131-196 (66).
- Board movement between the two reads:
  - 312 teams changed best submission and 17 are new.
  - 16 teams in .959 have a new best submission (9 came up from .955-.958, 7 improved inside .959): 1 above pw30 (r138,
    up from .958), 15 below.
  - 3 moved .959 -> .960 and 2 moved .958 -> .960.
- Frozen teams (same best submission at both reads):
  - In .959: 55. Of these, 2 are above eoa20 and above pw30, 2 above eoa20 but below pw30, and 51 below both.
  - In .960 / .961: 34 / 19, all above both.
  - No frozen team contradicts pw30 > eoa20.

### 1.2 Conversion (MODEL, `work/lb_post2.py` -> `lb_post2.log`, `lb_post2.npz`)
- Same exponential in-band density as push/work/lb_post.py (pooled ALL / F18 / F19b rulers), now saving k per draw. It
  reproduces implied.md: 1 - u_c4 .24 [.14, .35], x10 .37 [.26, .47], x20 .90 [.75, .95], eoa20 to .960 1.0e-4 [0.5, 2.5].
- Frozen ruler likelihood: given x20 and k, the 4 teams above eoa20 are iid on (x20, 1) with density ~exp(-kx). So
  P(observed order | x_pw30) = 6 G^2 (1 - G)^2, where G is the truncated CDF at x_pw30. LB-only posterior: G ~ Beta(3, 3).

| estimator | x_pw30 | pw30 - eoa20 (e-4) | pw30 to .960 (e-4) |
|---|---|---|---|
| frozen ruler, LB-only | .95 [.87, .98] | +0.5 [0.2, 1.2] | 0.5 [0.2, 1.3] |
| A21 board alone (3 above / 68 below, abc_update.py method) | .94 [.89, .97] | +0.4 [-0.3, +1.8] | |
| frozen ruler + the 16 non-frozen .959 teams as iid band draws (sensitivity) | | +0.5 [0.2, 1.3] | |
| joint refit (abc3 primary; curve prior + ruler) | .91 [.82, .97] | +1.0 [0.3, 1.9] | 0.75 median, mean 0.9 [0.3, 1.8] |

## 2. Refit (MODEL, `work/abc3.py`; box aggregates `work/box/perlabel_{PFT,PAREA}.log`)

### 2.1 What is new vs push/work/abc_local2.py
- **Per-label curves.**
  - New box run with per-label (Medial / Lateral / PF) AUC deltas for S05-S50 and PW10-PW50, plus the pooled 57-dim noise
    covariance at n 390.
  - Models: D5 (primary) and s1044. Regimes: raw and R3p -2 ... +2 sd (now to +2).
  - So PW45, S45, LP*, LPW* and M20_LP30 are exact column sums, not interpolations.
  - Matches the old totals (D5 R3p+0: S20 +.00032 vs +.00031, S30 -.00025 vs -.00026; MEASURED).
- **Observations.**
  - O1: kernel on (G10, G20) as before.
  - O2: frozen-ruler likelihood on pw30 - eoa20 with each LB draw's own x20, k.
  - O3: R1 argmax (pw30 > eoa20, pw20, eoa30, eoa10).
  - O4: the pw20 / eoa30 .959 displays (band floor).
- **Prior.** Regime prior N(0, 1) clipped to [-2.5, +2.0] (was +1.0).
- **Outputs.**
  - Private draws: population + independent noise x sqrt(390/910).
  - A fresh public replicate for the winner's-curse sims.
- **Sample size.** 600k sims (primary) and 360k (variants); ESS 2,750 primary, 1,000-2,200 variants.
- **Pre-read check** (O1 only) reproduces implied.md within ~0.4e-4: PW30 - eoa20 +2.1 (implied.md +1.7), PW20 +0.8 (+0.9),
  PW25 +2.0 over eoa20 (+1.9). The small shifts come from the wider regime grid and the fresh box run.

### 2.2 Posterior (primary P_FT / D5 / N(0,1) / logN(0, .35); variants in `abc3_variants.log`, `ranges3.log`)

| quantity | primary | variant range of the mean |
|---|---|---|
| regime shift (sd of gold74 log OR) | -0.17 [-1.0, +0.6] | -0.51 (rho_c4 .78) .. +0.18 (P_AREA) |
| amplitude s | .82 [.58, 1.07] | .69 (logU) .. .87 |
| c4 gap (rho - q) | .128 [.081, .176] | .076 (P_AREA) .. .163 (s1044) |
| q_hidden / label phi | .69 / .29 | .66-.74 / .25-.34 |
| Medial amplitude ratio (variant "Medial free", prior logN(0, .5)) | posterior = prior (1.00 [.51, 1.60]) | not identified by the reads |

- Dropping O2 while keeping the argmax (variant "no O2") would put pw30 at +3.8e-4 over eoa20 and above .960. That
  contradicts its .959 display, so the frozen ruler is the binding piece of information.

## 3. Candidate table (primary; `abc3_primary.log`; variant ranges `ranges3.log`; e-4 vs pw30)

| shape | public mean [p10, p90] | P(> pw30) | P(.960) joint / LB-only gap | private mean [p10, p90] | P(priv > pw30) | population mean |
|---|---|---|---|---|---|---|
| pw25 | +0.09 [-0.94, +1.10] | .56 (.43-.62) | .22 / .29 | +0.56 [-1.97, +3.04] | .61 (.50-.69) | +0.50 |
| M20_LP30 | +0.06 [-1.73, +1.84] | .51 (P_AREA .04) | .29 / .34 | +0.33 [-1.74, +2.40] | .59 (P_AREA .11) | +0.27 |
| pw20 (built) | -0.93 [-1.90, -0.16] | 0 | 0 | +0.06 [-4.50, +4.57] | .50 (.42-.57) | -0.06 |
| eoa25 (S25) | -0.75 [-2.05, +0.50] | .23 | .08 / .10 | -0.63 [-3.65, +2.30] | .39 | -0.69 |
| eoa20 (built) | -1.02 [-1.90, -0.31] | 0 | 0 | -0.81 [-5.70, +3.94] | .42 (.34-.46) | -0.91 |
| LPW30 | -1.36 [-5.02, +2.48] | .32 (P_AREA .01) | .22 / .25 | -0.28 [-4.35, +3.62] | .47 (P_AREA .01) | -0.38 |
| LP30 | -1.66 [-5.96, +2.57] | .32 (P_AREA .01) | .23 / .25 | -0.47 [-5.31, +4.29] | .45 (P_AREA .01) | -0.61 |
| pw35 | -1.63 [-3.48, +0.16] | .12 (.06-.22) | .04 / .06 | -1.89 [-4.68, +0.90] | .19 (.11-.32) | -1.83 |
| eoa30 (built) | -1.72 [-3.40, -0.29] | 0 | 0 | -1.76 [-3.59, +0.06] | .11 | -1.77 |
| pw40 | -5.14 [-8.80, -1.58] | .03 | .01 / .02 | -5.71 [-11.62, -0.06] | .10 | -5.59 |
| eoa40 (S40) | -7.01 [-11.47, -2.77] | .01 | .00 / .01 | -7.92 [-13.90, -2.06] | .04 | -7.81 |
| pw45 | -10.17 [-16.29, -4.38] | .01 | .01 / .01 | -11.08 [-20.47, -2.11] | .06 | -10.90 |

- P(display .958): pw45 .58 (.39-.73), eoa40 .26 (.13-.35), pw40 .09, LP30 .01 (P_AREA .34). P(.961) ~0 for everything.
- The public column is conditional on the realised public noise (the same 390 knees), so it is much tighter than the private
  column for shapes close to pw30.
- Private for the pw ladder is P_FT / P_AREA robust. Medial-sparing shapes (LP*, LPW*, M20_LP30) depend on the proxy.

## 4. Winner's curse (MODEL, `abc3.py` WC block; posterior-predictive fresh public sample + private sample)

| set picked by public max | overstatement of the winner's gain vs c4 (e-4) | regret vs population-best member | P(pick = pop-best) | E private of pick minus fixed pw30 |
|---|---|---|---|---|
| A21 five (eoa10, eoa20, eoa30, pw20, pw30) | +1.9 [-5.2, +10.7] (1.5-3.0) | 0.9 (0.7-1.2) | .50 | +0.0 (-0.0..+0.6) |
| seven (+pw25, pw35) | +2.3 (1.8-3.6) | 1.0 (0.8-1.4) | .31 | +0.05 (-0.0..+0.6) |
| seven (+pw40, LP30) | +3.3 (2.5-4.9) | 1.3 (1.0-1.7) | .29 | -0.4 (-0.4..+0.3) |
| three (eoa20, pw20, pw30) | +1.3 (1.1-2.1) | 0.55 (0.4-0.75) | .60 | +0.3 (+0.2..+0.9) |

- Reading:
  - A fixed entry's overstatement is 0 in expectation. Selection adds ~2e-4 for 5-7 E-OA entries, scaling with how far up
    the noisy high-a tail the set reaches.
  - Selection among E-OA entries buys ~nothing privately. Adding high-a or Medial-sparing shapes to the pool lowers the
    expected private of the R1 pick.
- Realised on today's data (posterior, not a replicate):
  - pw30 public over c4 +11.3e-4. Its expected population gain is +10.0e-4, so the discount is 1.4e-4 (variants 0.7-2.8).
  - Against eoa20 the overstatement is +0.1e-4 (variants -0.2..+0.7).
  - Hence: the absolute E-OA gain carries a ~12% common-mode haircut. The ordering among E-OA entries carries almost none.

## 5. 10-21 slates under R1, conditional on today's reads (MODEL, `abc3_primary.log` SLATE lines)

| slate added to pw30 | P(best-id moves) | public gain of best-id (e-4) | private of pick vs pw30 (e-4) | variants | P(up) / P(down) | P(best shows .960) |
|---|---|---|---|---|---|---|
| pw25 | .56 | +0.37 [0, 1.10] | +0.37 [-1.11, +2.51] | +0.13..+0.67 | .35 / .21 | .22 (.11-.26) |
| M20_LP30 | .51 | +0.58 [0, 1.84] | +0.15 [-1.09, +1.66] | -0.06..+0.23 | .30 / .22 | .29 (.01-.30) |
| pw25 + M20_LP30 | .74 | +0.76 [0, 1.89] | +0.35 [-1.46, +2.44] | +0.07..+0.58 | .45 / .29 | .40 (.11-.43) |
| pw25 + pw35 | .62 | +0.43 [0, 1.19] | +0.28 [-1.59, +2.51] | +0.04..+0.53 | .37 / .26 | .26 (.14-.31) |
| pw25 + LP30 | .67 | +0.92 [0, 2.57] | +0.14 [-2.42, +2.98] | -0.20..+0.36 | .38 / .29 | .38 (.12-.43) |
| asked five: pw35 pw40 pw45 eoa40 LP30 | .41 | +0.75 [0, 2.60] | -0.32 [-2.95, +1.44] | -0.47..-0.15 | .17 / .24 | .26 (.02-.31) |
| pw25 M20_LP30 LP30 LPW30 pw35 | .80 | +1.18 [0, 2.93] | +0.13 [-2.48, +2.82] | -0.17..+0.28 | .44 / .36 | .49 (.15-.54) |
| pw35 + pw40 | .13 | +0.09 [0, 0.20] | -0.17 | -0.27..-0.09 | .04 / .09 | .05 (.02-.09) |

- MODEL reading:
  - pw25 is the one shape that is positive on public, private and robustness at once.
  - Every extra entry raises P(.960) by ~.05-.1 but costs private expectation unless it is itself near the population
    optimum (pw20-pw25).
  - The asked high-a ladder is the worst use of slots: negative private value, small P(.960).
- Any 10-21 build needs Eesh's go plus a PREREG amendment (not touched here).

## 6. Caveats
- **Curve family = OAI proxy.** OAI-free report-trained D5 on 2,399 MOAKS-read knees, coupling-reweighted, resampled at
  n 390. The P_FT / P_AREA split on Medial OA is the largest unresolved model uncertainty. It only matters for Medial-sparing
  shapes.
- **Medial ceiling.** c4's Medial OA is near ceiling (gold58 .989). The free-Medial-amplitude variant shows the reads cannot
  identify a Medial-specific amplitude: posterior = prior, and every number moves by <= 0.3e-4. So the implied.md eoa30 haircut
  (GUESS) is neither confirmed nor refuted. The read itself (eoa30 below pw30, P(eoa30 > eoa20) .30) is in its direction.
- **Joint vs LB-only position of eoa20** (.81 vs .90): P(.960) for a candidate is ~.05 higher under the LB-only gap
  (`extra3.log`).
- **Band-density model** as in push/implied.md (exponential within band, Poisson teams). The 16 non-frozen .959 teams are used only
  as a sensitivity: they are not iid band draws (7 were already in .959).
- **R1 exactness** (best-id = max unrounded) is assumed, as in decision.md.
- **Private.** Private noise scaling sqrt(390/910) assumes AUC noise ~ 1/sqrt(n) and a random public / private split
  (CLAIMED flat_733491).

## 7. Files (all under `session_2026-10-20/val/work/`)
- `lbpos2.py` -> `lbpos2.log` (A21 counts, frozen cross-board ruler).
- `lb_post2.py` -> `lb_post2.log`, `lb_post2.npz`, `lb_post2.json` (LB posterior + ruler).
- `box/box_perlabel.py` -> `box/perlabel_PFT.log`, `box/perlabel_PAREA.log` (box aggregates only).
- `abc3.py` -> `abc3_primary.log/.json` (pre-read check + primary), `abc3_variants.log/.json` (13 variants incl. an MC seed check).
- `summarize3.py` -> `summarize3.log`.
- `ranges3.py` -> `ranges3.log`.
- `extra3.py` -> `extra3.log` (LB-only-gap P(.960), optimal-shape mass).

## 8. Skeptic (10-20 ~20:05 IST, Claude; val SKEPTIC track, RESEARCH ONLY)

Target: refute (a) the predicted optimum (Track A `oaigrid.md`: a* ~.225-.25 for sym / pw; this file: private-best mass PW20 .32,
PW25 .30, PW30 .23) and (b) any step beyond pw30. LOCAL ONLY: a re-analysis of the box AGGREGATES already on disk (JSON block of
`val/box_oaigrid_{PFT,PAREA}.log`: per-regime population curves, OAI-size bootstrap covariance, n 390 public-resample mean and
covariance). No ssh, no OAI row / id / per-knee value read or printed, nothing pushed, committed, submitted or registered.
Files: `val/skeptic_work/skeptic_val.py` -> `skeptic_val.log`; `lingauss_shape.py` -> `lingauss_shape.log`.
Units: public e-4 (OAI units x the regime's posterior amplitude s). "Variants" = 9 posteriors (prior N(0,1) / N(-1.2,1) / flat x
likelihood Track-A joint / joint + A21 frozen ruler / ruler with s log-uniform [.2, 4]) x 2 proxies (P_FT, P_AREA) = 18.

### 8.0 Verdicts

| entry | verdict | reason (tag) |
|---|---|---|
| pw35 | **DOWNGRADE** | population pw35 - pw30 < 0 in 18/18 variants: -0.55 to -2.09, P(>) .06-.32 (MODEL on MEASURED aggregates). The only corner where it wins: flat prior + wide s + private labels LESS coupled by .5 sd, and then by +0.4 to +0.5 (sec 8.4). gold58 +.00034 (P .64) is info only. |
| pw40 | **DOWNGRADE** | -2.2 to -5.7, P(>) .02-.21 (MODEL) |
| pw45 | **DOWNGRADE (strong)** | -5.0 to -11.6, P(>) <= .13; P(.958 display) .58 (sec 3) |
| eoa40 | **DOWNGRADE (strong)** | -3.6 to -7.1, P(>) <= .11 |
| LP30 (10-19 F2 definition) | **DOWNGRADE** | proxy sign flip (P_FT private -0.5, P_AREA -7.3; sec 3). Track A's LP2 / LPpw (different transforms, sec 8.2) are -1.5 to -4.8 in 18/18 variants, P(>) <= .35 |
| M20_LP30 | **DOWNGRADE** | adds ~0 private next to pw25 (slate +0.35 vs pw25 alone +0.37, sec 5); P_AREA private -1.9; Medial-sparing shapes flip sign across the cartilage proxies (CLAIMED session_2026-10-19/ex/cand_eoa_curve.md line 70). It buys P(.960 display) only |
| pw25 | **KEEP, low value, rationale downgraded** | "pw25 beats pw30" is prior-driven: population pw25 - pw30 -0.27 to +0.90 across variants (P .41-.76). Kept because it is adjacent (public paired sd vs pw30 1.4), its R1 value is +0.06 to +0.67 in the B / picks2 variants (CLAIMED sec 5, picks2.md sec 1.2), never negative, and it is the minimax-regret dose under coupling shifts (sec 8.4) |
| R6pw30 (B) | **DOWNGRADE** | duplicates A's a .3 dose tail (agrees with picks2.md sec 0) |
| R6pw20 (B) | **KEEP** | a real dose hedge: pw20 - pw30 swings to +0.8..+3.3 if private labels are .5 sd more coupled; pw20 > eoa20 in population in 18/18 variants (mean +0.45 to +0.95). Its edge over R6e20 is model-only (gold58 no preference) |
| R6e20 (B alt) | KEEP as substitute | same dose, sym shape, ~0.2e-4 behind in the OAI families |
| B fallback pw20 57425269 | KEEP | as picks2.md |
| Override R1 to A = pw20 (picks2.md sec 8, info) | not supported | population pw20 - pw30 changes sign with the prior: -1.26 to +0.94 |

### 8.1 Is the regime fit informative? (MEASURED aggregates, MODEL arithmetic; `skeptic_val.log` sec 0, 1, 4)
- **Re-implementation check.** A Gaussian version of Track A's ABC on (S10, S20, S30, PW20, PW30), using the logged n 390
  resample mean and covariance, reproduces the logged P(order) within .02 and L_joint within ~25% wherever the posterior has mass.
- **Read noise vs the steps.** Public paired sd at the MAP regimes (public units): pw30 - eoa20 **2.8-3.0**, eoa30 - eoa20
  2.4-2.5, eoa20 - eoa10 2.1, pw35 - pw30 1.5, pw25 - pw30 1.4, pw20 - eoa20 1.0. The LB-only pw30 - eoa20 step (+0.5 [0.2, 1.2])
  is ~0.2 sd. So every individual public ordering among the E-OA entries is weak, as suspected. Track A's likelihood does carry
  this noise, which is why its k posterior is wide (sd .41-.56).
- **What locates the regime.** The order-only likelihood is monotone and saturates (P(order) .11 at k -1.0 -> .77 at k <= -2.5);
  alone it gives a* .35-.375 under a flat prior, but that is the grid floor (k -3), not data. The (G10, G20) magnitudes with the
  amplitude prior pull k to ~-1.2..-1.7, and the single binary fact "pw30 did not show .960" removes k <= -2.5. Adding the A21
  frozen ruler (Beta(3,3) position in the eoa20 -> .960 gap) moves the k mean by <= .02 and a* not at all: Track A's "not .960" already held it.
- **Posterior-predictive check.** Under Track A's fit, public pw30 - eoa20 is +1.06 [0.29, 1.95]; LB-only says +0.5 [0.2, 1.2].
  Slightly high, inside noise (the same "modest downward surprise" as sec 0.1). The pw / sym double difference
  (pw30 - eoa30) - (pw20 - eoa20) is predicted +0.7..+0.9 [-0.3, +2.0]; nothing on the LB contradicts it (eoa30 / pw20 censored).
- **Prior and amplitude sensitivity** (pw optimum = argmax of the posterior-mixed population curve; P(a* >= .35) from the
  per-regime argmax; ranges = joint / ruler likelihood; last column P_FT population pw25 - pw30 with P(>), from the joint row at
  s [.5, 2] and the ruler row at s [.2, 4]):

  | prior on k | s prior | P_FT a* | P_FT P(a* >= .35) | P_AREA a* | P_AREA P(a* >= .35) | P_FT pw25 - pw30 |
  |---|---|---|---|---|---|---|
  | N(0,1) (Track A) | [.5, 2] | .225 | .02 | .25 | .01-.02 | +0.87 (.76) |
  | N(0,1) | [.2, 4] | .225 | .04-.05 | .25 | .03-.04 | +0.63 (.68) |
  | N(-1.2, 1) | [.5, 2] | .25 | .05 | .25 | .04-.05 | +0.29 (.59) |
  | N(-1.2, 1) | [.2, 4] | .275 | .12-.15 | .275 | .08-.11 | +0.04 (.52) |
  | flat | [.5, 2] | .275 | .11 | .25 | .05-.06 | +0.07 (.51) |
  | flat | [.2, 4] | .30 | .23-.29 | .275 | .11-.15 | -0.27 (.41) |

  N(-1.2, 1) is the N(0,1) prior re-centred where Track A itself says c4D5 has to sit to mimic c4 (oaigrid.md sec 3.1: k is an
  effective shape parameter at rho .66, not hidden coupling). A prior centred on 0 therefore pulls a* down by about one .025 step.
- **Reading.** REFUTED as a data result: "pw25 is expected to beat pw30". It is the N(0,1) answer; likelihood-led variants put
  the optimum at .25-.30, i.e. pw30 at, not past, the optimum. NOT REFUTED: "nothing at .35+". The mixed-curve optimum never
  exceeds .30 in the 30 prior x likelihood x proxy rows that use the magnitudes (`skeptic_val.log` sec 1, incl. mag-only), and
  P(a* >= .35) <= .29 in all of them. Only the 6 order-only rows reach .30-.375, the grid-floor effect above.

### 8.2 Multiplicity across tracks and families
- Track A scored 4 families x 28 doses x 18 regimes, this file 17 candidates x 13 variants, picks2 33 sensitivity rows, all on
  the same OAI labels, the same D5 stand-in and the same 5 LB reads. A / B agreement is therefore not independent confirmation.
- Stable across tracks: the high-dose side. Population pw35 / pw40 / pw45 - pw30 is -2.1 / -5.5 / -10.4 (A, N01, P_FT) vs
  -1.8 / -5.6 / -10.9 here. Not stable: the fine order among pw20 / pw25 / pw30 / eoa20 (pw20 - pw30 -1.3..+0.9;
  eoa20 - pw30 -2.0..+0.1). Do not spend decision weight on that order.
- "pw25 is the only untested grid point expected to beat pw30" is a max over ~100 correlated model means; its margin
  (+0.16, sd .58) is .3 sd and changes sign under a flat prior with a wide s prior.
- **LP is defined differently in the two tracks.** Track A's LP2 / LPpw: Lateral and PF subtract only each other (a post-hoc
  PH-type transform). This file's LP30 = the 10-19 pre-registered F2 shape: Medial output untouched, Lateral and PF taken from the
  sym .3 transform, so they still subtract Medial (`work/abc3.py` shape(); CLAIMED cand_eoa_curve.md line 25). The two "LP loses"
  verdicts are about different transforms; the LP30 Eesh named is this file's.

### 8.3 Proxy validity (MODEL `lingauss_shape.log`; CLAIMED where named)
- **rho mismatch** (c4D5 .66 and D5 .78 vs c4 ~.82; R6 / S6fall coupling collapse on OAI, CLAIMED ROUND5.md A.4). In the
  linear-Gaussian picture at a matched optimum, the early shape barely depends on rho (g(.1)/g(.2) .623 at rho .66 vs .610 at
  .82, a* .25), so a shape fit like Track A's identifies a* whatever the rho. But past the optimum a higher rho falls FASTER:
  g(.4)/g(.3) .43 (rho .66) vs .29 (rho .82); g(.45)/g(.3) -.12 vs -.47. If c4's true coupling is above the proxy's, over-dosing
  costs more than A or B predict. This strengthens the pw40 / pw45 / eoa40 downgrades.
- This file's gap matching (sh_eq) could understate c4's a* by ~11% (formula a* .33 at D5 rho .776 vs .37 at c4 rho .816, same
  gap .128): .25 -> ~.28. Not enough to reach .35. The formula overstates the measured OAI AUC argmax (.29 vs .175-.225 at the
  P_FT MAP), so it is used for direction only.
- **D5 for Raptor**: not testable (Raptor is OAI-trained). If Raptor's OA heads are more coupled than D5's, same direction as the
  rho point (GUESS).
- **Per-label sign conflict** (OAI: PF OA loses from the first step; gold58: PF gains up to .4). Only the macro shape transfers;
  Medial-sparing and LP verdicts do not. Info only: gold58 c4 pw .35 -> .45 falls -9.9e-4 macro and the proxy (P_FT MAP)
  -9.2e-4 public units, so the high-dose fall-off magnitude agrees.

### 8.4 Private cost of over-decorrelation if private labels are more coupled (MODEL, `skeptic_val.log` sec 3)
Private regime = posterior regime + d (d in sd of gold74 log OR; + = more coupled). Population X - pw30, public units.
Cells: P_FT under N(0,1) x Track-A joint / P_FT under flat x ruler, s [.2, 4]; parentheses: P_AREA, same two posteriors.

| d | pw20 | pw25 | pw35 | pw40 | pw45 (P_FT) | best pw a (P_FT) |
|---|---|---|---|---|---|---|
| -0.50 | -1.6 / -3.2 (-2.6 / -3.8) | -0.4 / -1.3 (-0.6 / -1.4) | -0.7 / **+0.5** (-0.5 / +0.4) | -2.6 / +0.1 (-2.3 / -0.3) | -6.0 / -1.6 | .30 / .35 |
| 0 | +0.9 / -1.3 (+0.3 / -1.1) | +0.9 / -0.3 (+0.8 / 0.0) | -2.1 / -0.6 (-2.1 / -1.0) | -5.5 / -2.2 (-5.6 / -3.3) | -10.4 / -5.0 | .225 / .30 |
| +0.50 | +3.3 / +0.8 (+2.9 / +1.3) | +2.1 / +0.7 (+2.2 / +1.3) | -3.4 / -1.7 (-3.6 / -2.4) | -8.2 / -4.5 (-8.7 / -6.2) | -14.6 / -8.5 | .175 / .225 |
| +1.00 | +5.6 / +2.7 | +3.3 / +1.7 | -4.8 / -2.8 | -10.9 / -6.7 | -18.7 / -12.0 | .125 / .175 |

- **Asymmetry.** The corner most favourable to stepping up (flat, wide s, private .5 sd LESS coupled) gives pw35 +0.4..+0.5 and
  pw40 -0.3..+0.1. The mirror corner (.5 sd MORE coupled) gives pw35 -1.7..-3.6 and pw40 -4.5..-8.7. Within the same posterior the pw35 downside is 3-6x its upside, and under N(0,1) there is no upside at all.
- **Worst-case regret over d in [-.5, +.5]** across the 4 proxy x posterior rows: pw25 1.8, pw30 3.3, pw20 4.2, pw35 6.7. pw25 is
  the robust dose, for a robustness reason, not because its expected public score is higher.
- How big d can be (MODEL / GUESS): under a random split (CLAIMED flat_733491) the private sample's label coupling differs from
  the population by ~.2-.25 sd of d (phi sampling sd ~.026 at n 910, dphi / dk ~.116), and that part is already inside this
  file's private noise. d = +-.5 needs a non-random split or label-protocol drift.

### 8.5 What would change the verdicts
- A 10-21 public read of pw25 or pw35 against pw30 cannot resolve population differences below ~1.4-1.5e-4 (paired sd), so a
  "win" by either is mostly noise. Under R1 that makes pw35 / pw40 slightly negative privately (sec 5; picks2.md sec 4.3) and
  pw25 ~0 to slightly positive.
- Only direct evidence on the hidden labels' coupling or per-label E-OA response would reopen a >= .35; nothing available gives it.
- Any 10-21 build still needs Eesh's go plus a PREREG amendment (not touched here).
