# implied.md: what the eoa10 / eoa20 public reads imply (10-20 ~15:05 IST, Claude). RESEARCH ONLY.

Nothing was submitted, pushed, committed, versioned or downloaded. No dataset, no GPU job, no scheduled-task edit. PREREG,
HANDOFF, PROGRESS, a20 and a21 were only read. Box use: 4 read-only CPU python runs piped over ssh stdin
(`work/box_abc_run.py`, smoke + P_FT + P_AREA, then an extended grid), printing aggregates only (means, sds, covariances,
regression coefficients, posterior summaries of AUC deltas over 390-knee resamples). No OAI id, row or per-knee value left the
box, nothing was written there, and no OAI CSV was printed. All other work is local on public LB snapshots and box aggregates.
Tags: MEASURED (computed now, file named), CLAIMED (older note, file named), MODEL (arithmetic on stated assumptions),
GUESS (judgement). Units: macro AUC; "e-4" = 1e-4 macro; "band" = .001 of display score.

## 0. Bottom line

1. **Unrounded public scores (MODEL from MEASURED in-band ranks; median [p10, p90]):**

   | entry | read | in-band position | unrounded if display truncates | if display rounds |
   |---|---|---|---|---|
   | c4 57343435 | r217, 23 teams above it inside .958 (04:17 UTC) | u .76 [.65, .86] of .958 | .95876 [.95865, .95886] | .95826 |
   | eoa10 57413380 | r167, 36 above / 29 below inside .959 (05:52) | x .37 [.26, .47] of .959 | .95937 [.95926, .95947] | .95887 |
   | eoa20 57413368 | r135, 4 above / 61 below inside .959 (05:53) | x .90 [.75, .95] of .959 | .95990 [.95975, .95995] | .95940 |

   - Public gains: **eoa10 - c4 = +6.1e-4 [4.4, 7.8]; eoa20 - c4 = +11.3e-4 [9.3, 12.7]; eoa20 - eoa10 = +5.1e-4 [4.0, 6.0]**
     (ratio G20 / G10 1.69 [1.34, 1.97]). The gains do not depend on the rounding convention.
   - **eoa20 is ~1.0e-4 [0.5, 2.5] below a .960 display.** From c4 a .960 display needs +1.24e-3.
2. **Implied coupling (MODEL, sec 2): the public expert OA labels are at about gold74 coupling, at most mildly less coupled.**
   - Implied c4 gap (rho - q) .127 [.077, .182]; q_hidden .70 [.63, .76] (gold58 .711); label phi .29 [.20, .40] (gold74 .32).
   - That is a shift of -0.16 sd [-1.0, +0.8] of the gold74 log OR (flat regime prior: -0.4 to -0.5 sd; P_AREA proxy: ~0).
   - Not the raw-OAI regime. The big public gains come mostly from c4's large excess coupling (gap ~.12-.13, above the
     skeptic's .096) plus a modest positive public draw.
   - The pre-read pessimists are 2.3 sigma low: R5's revision (eoa20 ~0) and the skeptic ladder (eoa20 +.0001) fit the reads
     at P ~.02-.03, 9-11x less likely than skeptic_eoa (+.0005 / +.0007) (MODEL, `prior_check.log`).
3. **Public-optimal a ~ .25** (posterior mass S20 .29, S25 .35, S30 .19, S35 .10; MODEL). Pair weighting adds ~+1e-4 at a .2-.3.
4. **A21 forecast (public increment over eoa20; primary model, [range over 8 prior x proxy variants]):**

   | entry | mean | P(> eoa20) | P(shows .960) | P(it is the new best-id) |
   |---|---|---|---|---|
   | eoa30 (sym .3) | -0.3e-4 [-0.3, +1.8] | .48 [.46, .78] | **.32 [.31, .64]** | .11 [.11, .36] |
   | pw20 | +0.9e-4 [+0.2, +1.2] | .75 [.59, .79] | **.45 [.28, .45]** | .26 [.13, .26] |
   | pw30 | +1.7e-4 [+1.1, +2.6] | .72 [.64, .80] | **.57 [.52, .68]** | .53 [.41, .60] |
   | any of the three | | P(best-id moves) **.89** [.88, .93] | P(any shows .960) **.70** [.67, .79] | eoa20 stays best .11 |

   GUESS haircut: c4's Medial OA is near ceiling (gold58 .989) and the D5 curve family cannot show that; shade eoa30 down by
   ~.05-.10 in every P. A .960 display alone is worth only ~4-9 places (r135 -> ~r126-131 on the 05:53 board).
5. **Private (MODEL, population curve + private-sample noise):**
   - vs c4: eoa10 +6.4e-4 (P > c4 .99), eoa20 +9.7e-4 (.95), eoa30 +8.6e-4 (.82), pw20 +10.6e-4 (.96), pw30 +10.6e-4 (.87).
   - vs eoa20: eoa10 -3.3e-4 (P .15), eoa30 -1.1e-4 (P .41), pw20 +1.0e-4 (P .85), pw25 +1.5e-4 (P .76), pw30 +0.9e-4 (P .59).
   - So the R1 best-id rule costs at most ~1e-4 privately whichever A21 entry wins.
6. **Re-scoring the EX / census candidates at the implied regime (sec 6):** verdicts unchanged. The regime is near gold74, so
   only LOA <- MM (and weakly PF <- LM) improve, by ~+.00005-.0001 macro on top of E-OA at -0.2 to -0.5 sd. Every KILL stays
   KILL. MCL <- MM still has no ruler.
7. **Best untested next shape for 10-21 (MODEL): pw25** (pair-weighted, a .25).
   - Public +1.9e-4 over eoa20 [+1.2, +2.1], P(> eoa20) .84 [.75, .86], P(shows .960) .64 [.53, .68].
   - Private +1.5e-4 (P .76).
   - It sits between pw20 and pw30, so its variance is lower. Decide after the A21 reads with `work/abc_update.py` (sec 7).

## 1. LB reads -> unrounded scores

### 1.1 Counts (MEASURED: `work/lbpos.py`, `work/frozen_multi.py`)
- Snapshots: `forum/api_lb_1020T0417.json` (c4 r217), `lbprobe/api_lb_1020_eoaread1.json` (05:52, eoa10 r167) and
  `api_lb_1020_eoaread2.json` (05:53, eoa20 r135).
- Between 05:52 and 05:53 only our row changed above .955 (the one other change was a .954 team near r2,100). So the .959 split
  29 / 32 / 4 (below eoa10 / between / above eoa20) is exact on one board.
- Bands at 05:53: .961+ 86 teams, .960 r87-130 (44), .959 r131-196 (66 incl. us), .958 r197-1537 (1,341).
- No exact-tie copy runs near the boundary (`runs_near.py`): runs >= 4 occur at the random rate, so a smooth density is
  reasonable there.

Zone counts. Frozen = best submissionId unchanged in every snapshot since the cutoff:

| ruler | .958 above c4 | .959 below eoa10 | eoa10..eoa20 | .959 above eoa20 | .960 |
|---|---|---|---|---|---|
| all teams | 23 | 29 | 32 | 4 | 44 |
| frozen since 10-17 16:21 | 4 | 7 | 7 | 2 | 6 |
| frozen since 10-18 12:52 | 7 | 10 | 11 | 3 | 11 |
| frozen since 10-19 10:41 | 13 | 15 | 20 | 3 | 27 |
| frozen since 10-19 20:17 | 18 | 23 | 26 | 3 | 34 |
| frozen since 10-16 08:45 | 0 | 4 | 3 | 1 | 2 (too sparse; reported only) |

### 1.2 Conversion (MODEL, `work/lb_post.py`)
- Teams near the boundary are a Poisson process with density A exp(-k x) above the .959 floor. k comes from the
  .959 : .960 counts.
- Below the floor the density grows faster: exp(k' |x|), with k' = k + U(0, 1.5), because .958 holds ~1,300 teams.
- Our positions come from a Dirichlet over the others above / between / below.

| ruler (k from counts) | 1 - u_c4 | x10 | x20 | G10 (e-4) | G20 (e-4) | to .960 (e-4) |
|---|---|---|---|---|---|---|
| all teams | .26 | .39 | .92 | 6.5 [5.3, 8.0] | 11.7 [10.7, 12.9] | 0.8 [0.4, 1.4] |
| frozen 10-18 | .19 | .32 | .81 | 5.2 [3.6, 7.2] | 10.0 [8.3, 11.8] | 1.9 [1.0, 3.3] |
| frozen 10-19b | .25 | .39 | .92 | 6.4 [5.1, 8.0] | 11.6 [10.6, 12.9] | 0.8 [0.4, 1.5] |
| flat density (all teams), reference | .36 | .44 | .93 | 8.0 | 12.9 | 0.7 |
| **pooled (all / F18 / F19b)** | **.24 [.14, .35]** | **.37 [.26, .47]** | **.90 [.75, .95]** | **6.1 [4.4, 7.8]** | **11.3 [9.3, 12.7]** | **1.0 [0.5, 2.5]** |

- Consistency: the 10-19 picks.md ruler put c4 at ~.9587-.9589 under truncation (CLAIMED). We now get .95876 [.95865, .95886].
- Rounding vs truncation is still unresolved (density.md sec 4). It shifts every absolute value by .0005 and changes no gain.

## 2. Implied coupling (MODEL)

### 2.1 Skeptic theory, direct inversion (`work/gapfit_pub.py`)
- The skeptic's linear maps (skeptic_eoa.md sec 4, x1.1 for c4) give gap .103 from G10 and .120 from G20 (CLAIMED maps, MODEL
  inversion).
- Refitted on the 40 OAI P_FT curve points (`ex/box_curve_*.log`), each a linear in gap (r .98-.99):

  | a | delta(a) = intercept + slope x gap |
  |---|---|
  | .1 | -.00027 + .00710 gap |
  | .2 | -.00102 + .01504 gap |
  | .3 | -.00240 + .02401 gap |

  Inversions on these fits:
  - With amplitude s = 1.0 / 1.1: gap .141 / .133.
  - With s free: gap .18, s .63 (wide).
- Population S20 / S10 ratio by gap: 1.11 at .10, 1.51 at .14, 1.68 at .18, 1.83 at .25. The observed 1.69 [1.34, 1.97]
  alone would say gap ~.15-.18. But the ratio is noisy, because the public-sample noise is slope-like (corr .95 between the
  a .1 and a .2 deltas).

### 2.2 ABC with real expert-label noise at n 390 (primary; `work/box_abc_run.py` -> `box_abc2_{PFT,PAREA}.log`, `work/abc_local2.py`)
- **Box (aggregates only):**
  - 3 OAI-free models x 8 coupling regimes: raw, and R3p with the gold74 pairwise log OR shifted by -2 ... +1 sd.
  - 1,200 (P_FT) / 600 (P_AREA) weighted resamples of 390 knees each.
  - Exported per regime: the mean of 17 transform deltas (S05-S50, PW10-PW50) plus rho, q and phi, and the pooled
    within-regime noise covariance.
  - Measured noise at n 390 (D5, gold74 regime): sd S10 .00025, S20 .00052, S30 .00083, S30 - S20 .00034;
    corr(S10, S20) .94.
- **Local:**
  - c4's population curve = s x the D5 curve at the regime whose D5 gap equals c4's gap.
  - c4 gap = rho_c4 (N(.816, .035)) - q(regime). c4's q equals D5's at gold74 (.711 vs .712).
  - The public sample = curve + noise drawn from the box covariance.
  - Weighting: a kernel on the LB posterior draws of (G10, G20). The .960 margin is joint with the same draw.

| prior (regime / amplitude s) | proxy | regime shift (sd) | c4 gap | q_hidden | label phi | s | eoa30 P(.960) | pw20 | pw30 |
|---|---|---|---|---|---|---|---|---|---|
| **N(0,1) / logN(0, .35) (primary)** | P_FT | **-0.16 [-1.0, +0.8]** | **.127 [.077, .182]** | **.70 [.63, .76]** | **.29 [.20, .40]** | .90 | .32 | .45 | .57 |
| N(0,1) / log-uniform [.4, 2.5] | P_FT | -0.22 | .130 | .69 | .29 | .80 | .46 | .40 | .63 |
| flat [-2.5, 1] / logN | P_FT | -0.41 | .138 | .68 | .27 | .88 | .36 | .44 | .61 |
| flat / log-uniform | P_FT | -0.53 | .148 | .67 | .25 | .75 | .51 | .40 | .68 |
| N(0,1) / logN | P_AREA | +0.09 | .080 | .74 | .33 | .92 | .49 | .30 | .52 |
| flat / log-uniform | P_AREA | -0.26 | .099 | .71 | .30 | .79 | .64 | .28 | .62 |
| primary, noise x1.25 / x0.8 | P_FT | -0.10 / -0.20 | .121 / .130 | .70 / .69 | .30 / .29 | .9 | .33 / .32 | .44 / .43 | .55 / .60 |
| primary, rho_c4 .78 | P_FT | -0.48 | .117 | .67 | .26 | .91 | .31 | .43 | .56 |
| primary, model s1044 | P_FT | -0.15 | .160 | .66 | .29 | .91 | .29 | .50 | .50 |

- Sensitivity, not primary: the box-side ABC lets the amplitude also scale the noise and caps s at .5. It pushes coupling much
  lower: sample phi .13 [.05, .23] (P_FT), .19 (P_AREA). Even so, its A21 forecast is the same: eoa30 P(.960) .48,
  pw20 .42, pw30 .68 (`box_abc_PFT.log`).
- Read: every variant puts the hidden OA coupling between ~1 sd below gold74 and gold74 itself.
  - The answer to "are the public labels less coupled than gold74?" is: a little (point -0.2 to -0.5 sd), not dramatically.
  - That makes the gap the main story: c4's blend raises rho to ~.82 while q stays ~.70.
  - That is the skeptic's mechanism (skeptic_eoa.md sec 2) with a gap ~.03 larger than its .096 point.

### 2.3 Which pre-read model the reads support (`work/prior_check.py`, MODEL)
Gaussian noise at n 390 as above, averaged over the LB posterior:

| expectation (population eoa10, eoa20) | Mahalanobis^2 | tail P | relative likelihood |
|---|---|---|---|
| R5 revision (+.0002, -.00005) | 7.6 | .023 | 1 |
| skeptic ladder (+.0003, +.0001) | 7.3 | .026 | 1.5 |
| skeptic_eoa (+.0005, +.0007) | 2.3 | .32 | 9 |
| ABC posterior mean (+.00064, +.00097) | 2.0 | .36 | 11 |
| gold58 c4 x .5 (+.0007, +.0012) | 1.4 | .50 | 13 |

- R5's "OAI is pessimistic for real members" caveat (r5/ROUND5.md) was right in direction. Its size revision was not.

## 3. A21 forecast details (`work/abc_joint.log`, `abc_local2.log`)
- Public increments over eoa20, primary (P_FT, N01, logN), [p10, p90] in e-4:

  | entry | median | [p10, p90] |
  |---|---|---|
  | eoa25 | +0.6 | [-1.3, +2.3] |
  | eoa30 | -0.1 | [-4.0, +3.2] |
  | eoa35 | -2.0 | [-8.5, +3.3] |
  | pw20 | +0.9 | [-0.8, +2.6] |
  | pw25 | +1.9 | [-0.6, +4.3] |
  | pw30 | +1.8 | [-2.2, +5.6] |
  | pw35 | +0.3 | [-6.2, +5.8] |
  | pw40 | -3.2 | [-13, +4.9] |

- The symmetric a .3 is past the population optimum for the primary prior. Pair weighting keeps Medial OA's total weight
  at ~.79 a, so PW30 holds up better.
  - Same direction on gold58 c4 (CLAIMED `a21/gate_a21_offline.log`): pw30 .94336 vs sym30 .94313; pw20 .94378 vs
    eoa20 .94360.
- Joint read (R1 = max unrounded of eoa20, eoa30, pw20, pw30), primary [8 variants]:
  - P(best-id moves) .89 [.88, .93].
  - P(at least one shows .960) .70 [.67, .79].
  - E[gain of the new best | move] +2.9e-4 [2.8, 3.6].
  - Winner: pw30 .53 [.41, .60], pw20 .26 [.13, .26], eoa30 .11 [.11, .36], eoa20 .11 [.07, .12].
- A no-move read (eoa20 stays best-id) has P ~.1. It would point to the ceiling / overshoot side (gap < .1) and argue for
  a <= .2 on private.

## 4. What .960 needs (MODEL, `work/rank_map.log`, 05:53 board, no drift)

| gain over eoa20 | rank | P(shows .960) |
|---|---|---|
| 0 | 135 | 0 |
| +0.5e-4 | 132 | .11 |
| +1.0e-4 | 131 | .48 |
| +1.5e-4 | 129 | .71 |
| +2.0e-4 | 126 | .83 |
| +3.0e-4 | 121 | .95 |
| +5.0e-4 | 112 | 1.00 |

- So "unrounded >= .960" (truncation) or ">= .9595" (rounding) both mean ~+1.0e-4 [0.5, 2.5] over eoa20.
- Near the boundary each +1e-4 is worth ~3-5 places. The display step itself is not a jump.
- Board drift: .959+ inflow is ~12-25 per day (CLAIMED forum/LB_10-20.md). It costs ~3-6 places by the evening read
  regardless.
- .961 needs ~+1.1e-3 over eoa20. No E-OA shape gets there (max public posterior p90 ~+6e-4).

## 5. Private and picks (MODEL; noise scaled by sqrt(390/910))
- The population regime posterior is shared with private (random split, same reader process; CLAIMED flat_733491).
  The public set (390) is far more informative about OA coupling than gold74 (71). So the private expectation moves with the
  public read: eoa20 vs c4 +9.7e-4 (P .95), not the +.0007 / ~0 of the pre-read models.
- Private-optimal shape is pw20-pw25 (+1.0-1.5e-4 over eoa20). eoa30 is -1.1e-4 (P > eoa20 .41) under the primary prior,
  +0.7 to +1.4e-4 under P_AREA.
- The R1 rule (A = API best-id) therefore picks something within ~1e-4 of the private optimum in every A21 outcome. This
  gives no reason to deviate from A15 (A = best-id, B = R6). It does strengthen the case that A should be an E-OA entry, not c4.

## 6. Re-scoring EX / census candidates at the implied regime (MODEL; `work/rescore_pairs.log`)
- Method: interpolate the OAI 3-model mean single-label delta (a .1) between R2 (gold74 log OR, sh 0), R2lo (-1 sd) and
  R2hi (+1 sd), using `ex/box_coupling.log` (CLAIMED aggregates).
- Caveat: the implied shift was measured on the OA trio only. Applying it to other pairs assumes one common
  "experts couple less" factor (GUESS).
- Readout at sh -0.2 (primary) / -0.5 (flat prior), AUC units:

| candidate | R2 (0) | sh -0.2 | sh -0.5 | R2lo (-1) | status at implied regime |
|---|---|---|---|---|---|
| LOA <- PF / PF <- LOA / MOA <- LOA (E-OA members) | +.0019 / +.0019 / +.0013 | +.0024 / +.0022 / +.0016 | +.0030 / +.0026 / +.0021 | +.0041 / +.0033 / +.0028 | in flight (E-OA) |
| **LOA <- MM** (standalone) | +.0014 | +.0023 | +.0035 | +.0056 | best non-trio pair. On top of E-OA (box_followup.log): +.00003 at gold74 (E-OA .2), +.00027 at -1 sd. So **~+.00005 (sh -0.2) to ~+.00015 (sh -0.5) macro** |
| PF <- LM | +.0003 | +.0008 | +.0015 | +.0028 | weak; failed transfer for s1044 at gold74. Overlaps PF <- LOA |
| PF <- ACL / Eff / Syn, MOA <- ACL / Eff, Syn <- ACL | positive | positive | positive | positive | still OAI population artifacts (coupling_map sec 0.4: c4 already anti-couples ACL-PF on gold58; OAI Syn = Eff). Stay KILL |
| menisci MM <-> LM (C6) | -.0002 macro | | | -.0002 to -.0004 macro | KILL (gold74 couples menisci MORE than OAI) |
| fluid FL3 (C7) | -.0008 | | | -.0003 to -.0005 | KILL |
| OA-meniscus (C9), global (C11) | negative | | | negative (R2m_lo, R9lo) | KILL |
| MCL <- MM (expert2) | no ruler | | | | DEAD for a slot (fails the specificity null) |

- Verdict: the implied regime is too close to gold74 to revive anything. The only pair that scales with lower coupling is
  LOA <- MM, worth ~+.00005-.00015 on top of E-OA.
  - That is half the .960 margin at best, with P(> 0) ~.6 (GUESS).
  - Not a slot unless the A21 reads come back strongly in favour of low coupling, e.g. pw30 or eoa30 well above
    pw20 and showing .960. In that case re-run `abc_update.py`. If the posterior shift is <= -1 sd, LOA <- MM at b .1 on top
    of the best shape becomes ~+.0003 (MODEL, R2m_lo).

## 7. After the A21 reads: update tool for the 10-21 slate (`work/abc_update.py`, read-only)
- Inputs: a fresh API snapshot (fetch_api_lb.py), which A21 entry the best-id maps to, and the 3-decimal display of each
  A21 entry (from our own submissions list).
- What it does:
  - Converts the best's in-band rank into its unrounded gain over eoa20.
  - Re-weights the ABC by that gain, by the argmax and by every display band.
  - Prints, for S25 / S30 / S35 / PW20-PW50: P(> best), P(shows .960 / .961) and the population (private) increments.
- Tested on a hypothetical read (pw30 best at r129, .960; others .959): delta +1.6e-4 [0.9, 3.1]; next best untested
  pw25 (P > best .33, P(.960) .74); private pw25 +1.6e-4 over eoa20.
- Pre-read MODEL ranking of untested 10-21 shapes (public mean over eoa20 / P(.960)):

  | shape | public mean over eoa20 | P(.960) |
  |---|---|---|
  | **pw25** | +1.9e-4 | .64 |
  | eoa25 | +0.5e-4 | .37 |
  | pw35 | +0.0e-4 | .43 |
  | eoa35 | -2.4e-4 | .24 |
  | pw40 | -3.7e-4 | .25 |

- Any 10-21 build needs Eesh's go plus a PREREG amendment (not touched here).

## 8. Caveats
- **Curve family and noise are OAI proxies.**
  - OAI-free report-trained models on 2,399 MOAKS-read knees, reweighted to RSNA coupling, resampled at the public size.
  - c4 is an RSNA blend with near-ceiling Medial OA (gold58 .989). The Gaussian check (`ceiling_check.log`) says a higher
    base AUC does not bend the curve faster (curvature ratio .64 -> .70 as AUC .84 -> .92).
  - But gold58 c4 does lose Medial OA at a .3 (-.0062, a few rank swaps). Hence the eoa30 haircut (GUESS).
- **Amplitude s is weakly identified.** Two priors are reported; log-uniform raises P(.960) for the a .3 entries by ~.05-.15.
- **LB density.**
  - The model is an exponential in-band density, continuous through the .959 floor.
  - The F17 / F18 rulers (2-3 of ~16-24 frozen .959 teams above eoa20) put eoa20 lower (x .78-.81, 1.9e-4 from .960).
  - The all-teams ruler puts it higher (.92, 0.8e-4). The pooled interval carries both.
- **R1 exactness** (the best-id is the max unrounded) is assumed, as in decision.md.
- **Public read noise is already in the model.** The large slope-like public noise (corr .94-.98 across a) is why the a .3
  points cannot be pinned below sd ~3e-4 from two reads.
- **Selection.** E-OA was first spotted on gold58. The public reads are independent of that selection, which is why they
  move the posterior this much.

## 9. Files (all in `session_2026-10-20/push/work/`)
- **LB:**
  - `snaps.py` (snapshot table)
  - `lbpos.py`, `frozen_multi.py`, `runs_near.py` (counts)
  - `lb_post.py` -> `lb_post.json`, `lb_post_pooled.npz` (positions)
  - `rank_map.py` -> `rank_map.log`
- **Theory inversion:** `gapfit_pub.py` -> `gapfit_pub.log`, `gapfit_pub.json`.
- **Box (aggregates only):**
  - `box_abc_template.py` + `make_box_abc.py` -> `box_abc_run.py`
  - logs `box_abc_smoke.log`, `box_abc_PFT.log`, `box_abc_PAREA.log`, `box_abc2_PFT.log`, `box_abc2_PAREA.log`
- **Local ABC:**
  - `abc_local.py` -> `abc_local.log`, `abc_local.json`
  - `abc_local2.py` (extended grid) -> `abc_local2.log`, `abc_local2.json`
  - `abc_joint.py` -> `abc_joint.log`
  - `prior_check.py` -> `prior_check.log`
  - `ceiling_check.py` -> `ceiling_check.log`
- **Re-score:** `rescore_pairs.py` -> `rescore_pairs.log`, `rescore_pairs.json`.
- **Post-read tool:** `abc_update.py`.
