# oaigrid.md: E-OA families (sym / pw / LP) on the c4D5 OAI proxy, three coupling regimes (10-20 ~19:00-19:40 IST, Claude)

RESEARCH + OFFLINE VALIDATION ONLY. Nothing submitted, pushed, committed, versioned or downloaded; no dataset, no GPU job, no
scheduled task. PREREG, HANDOFF, PROGRESS, a20 and a21 were only read. Box use: read-only CPU python piped over ssh stdin
(`box_oaigrid.py`: 1 smoke run, 2 full runs P_FT / P_AREA; two of my own full runs were killed at regime 3 and relaunched to add
the covariance export). Printed AGGREGATES ONLY (AUC-delta means, sds, covariances over resamples, likelihood sums,
argmax counts). No OAI id, row or per-knee value left the box; nothing was written there. Local: `oaigrid_post.py`,
`oaigrid_post2.py` on those aggregates. The other files in `val/` (build_val, gate_val, picks2_work, sim, work, slate) are not
from this track.

Tags: MEASURED (box aggregate, log named), MODEL (likelihood / posterior arithmetic on stated assumptions), CLAIMED (older
note, file named), GUESS (judgement).
Units: **macro e-4** = (sum of the 3 OA label AUC deltas) / 12 x 1e4, measured on OAI ("OAI units"). "Public units" = OAI
units x amplitude s (posterior s .64-.71, sec 3.1), comparable with the LB reads (G10 ~ +6.1, G20 ~ +11.3 e-4, implied.md).

## 0. Bottom line

1. **Fitted coupling regime (MODEL, sec 3.1).** On the R3p continuum (gold74 pairwise log OR shifted by k sd, OAI
   marginals), the reads c4 < eoa10 < eoa20 < pw30, pw20 < pw30, eoa30 < pw30 (pw30 not .960) plus the (G10, G20) magnitudes
   give:
   - P_FT: **k = -1.33 (sd .48), MAP -1.25** -> label phi .16, c4D5 rho .661, q .510, **gap .151**.
   - P_AREA: **k = -1.01 (sd .41), MAP -1.00** -> phi .215, rho .659, q .564, **gap .095**.
   - Both end-points are rejected: gold74 (R1) P(order) .000 / .006; raw OAI fits 3x (P_FT) / 13x (P_AREA) worse than the peak.
2. **Optimum a per family at the fitted regime** (posterior-mixed population curve; p10/50/90 over regime posterior x knee
   bootstrap; MODEL on MEASURED curves):

   | family | P_FT N01 prior | P_FT flat prior | P_AREA N01 | P_AREA flat | P(a* >= .35) | P(a* >= .40) |
   |---|---|---|---|---|---|---|
   | sym | **.225** [.125, .30] | .25 [.175, .35] | **.25** [.175, .325] | .275 [.175, .35] | .03-.15 | .00-.03 |
   | pw | **.225** [.15, .30] | .275 [.175, .35] | **.25** [.175, .325] | .275 [.175, .35] | .04-.16 | .01-.04 |
   | LP2 (L-PF only, w 2) | **.20** [.15, .275] | .25 [.175, .325] | **.20** [.15, .275] | .225 [.175, .30] | .01-.07 | .00-.01 |
   | LPpw (L-PF only, w 1.416) | **.30** [.20, .40] | .35 [.25, .45] | **.30** [.225, .40] | .325 [.25, .425] | .29-.56 | .14-.31 |

   LPpw at a equals LP2 at .708 a, so the two LP rows are one family: optimum coefficient on the partner ~.20-.25.
3. **Curve shape near .3-.45 (MEASURED at the MAP regime; OAI units; paired sd [OAI-size knee bootstrap | public n 390])**:
   concave everywhere, every step down is larger than the last.

   | family | P_FT .30 / .35 / .40 / .45 | P_FT steps .30->.35, .35->.40, .40->.45 | P_AREA .30 / .35 / .40 / .45 | P_AREA steps |
   |---|---|---|---|---|
   | sym | +7.7 / +3.7 / -2.1 / -10.6 | -4.0 [1.0 / 2.1], -5.8 [1.1 / 2.2], -8.5 [1.1 / 2.2] | +11.8 / +8.7 / +3.7 / -4.2 | -3.0 [1.0 / 2.1], -5.0 [1.1 / 2.2], -7.9 [1.2 / 2.5] |
   | pw | +9.7 / +6.1 / +0.5 / -7.6 | -3.7 [1.0 / 2.2], -5.6 [1.1 / 2.2], -8.1 [1.2 / 2.4] | +12.5 / +9.5 / +4.3 / -4.0 | -3.0 [1.0 / 2.2], -5.2 [1.2 / 2.4], -8.4 [1.3 / 2.6] |
   | LP2 | +6.4 / +3.2 / -1.2 / -7.9 | -3.1 [0.9 / 1.9], -4.4 [0.9 / 2.0], -6.8 [1.0 / 2.1] | +7.4 / +3.9 / -1.1 / -8.8 | -3.5 [0.9 / 2.0], -5.0 [1.0 / 2.1], -7.7 [1.1 / 2.3] |
   | LPpw | +8.8 / +8.2 / +7.2 / +5.3 | -0.6 [0.6 / 1.3], -0.9 [0.6 / 1.3], -1.9 [0.6 / 1.4] | +9.4 / +8.9 / +8.1 / +6.3 | -0.6 [0.6 / 1.3], -0.8 [0.7 / 1.4], -1.8 [0.7 / 1.4] |

   - Each .05 step past .3 is ~3-7 OAI-size sd negative for sym / pw / LP2, but only ~1.4-3.8 sd at public size.
4. **Next-entry implications (MODEL, sec 4-5):**
   - Public, over pw30: **pw25 +0.16e-4 (sd .58), P(> pw30) .60, P(shows .960) .24** (flat prior +0.08, .54, .20; P_AREA
     +0.17, .61, .24). It is the only grid point with a positive mean. pw35 -1.5 (P .06), pw40 -4.4 (P .00), eoa25 -0.7 (P .14-.21),
     every LP variant -3.6 to -9.4 (P <= .04).
   - Private (population): **pw25 - pw30 +0.1 to +1.0e-4 public units, P .52-.76**; pw20 vs pw30 is a coin flip
     (P .39-.64); **pw20 > eoa20 P .88-.98**; eoa30 < pw30 (P .02-.24); pw35 < pw30 (P .07-.21).
   - **No a >= .35 step in any family on this evidence, and no LP entry.** LP is the weakest result here, though: its loss
     comes from PF OA, the label where OAI and gold58 disagree in sign (sec 3.4).

## 1. Setup (MEASURED `box_oaigrid_PFT.log`, `box_oaigrid_PAREA.log`)
- **Knees:** 1,704 OAI knees with served R6 + served S6fall + D5 + MOAKS labels (the R6 prefix). Prevalence M / L / PF:
  P_FT .216 / .195 / .433; P_AREA .470 / .368 / .748.
- **c4D5 proxy** (c4 cells 14 + 18 structure, CLAIMED skeptic_r6oai.py / ROUND5.md A.3): per OA label
  V = wr pct(D5) + (1 - wr) pct(R6), C = .8 pct(V) + .2 pct(S6fall), wr .5 / .2 / .5. Effective weights: D5-as-Raptor
  .40 / .16 / .40, R6 .40 / .64 / .40, S6fall .20. Every pct is a weighted CDF under the regime (and bootstrap) weights.
- **E-OA:** out_t = U_t - a/2 sum_o w(t,o) U_o on U = pct(C).
  - sym: w = 1.
  - pw: w(M,L) .876, w(M,PF) .708, w(L,PF) 1.416 (build_a21.py).
  - **LP2:** w(L,PF) = 2 and Medial pairs 0, so Medial is untouched and L / PF each subtract a x partner (the sym total weight).
  - **LPpw:** w(L,PF) 1.416 and Medial pairs 0 (pw with the Medial pairs zeroed; = LP2 at .708 a).
  - a grid .025-.70 in .025 steps.
- **Regimes:**
  - raw OAI;
  - R1g74 (gold74 8-cell joint, the skeptic's R1);
  - MIX l (8-cell joint (1 - l) OAI + l gold74, l .25 / .5 / .75);
  - R3p k (gold74 pairwise log OR + k sd, OAI marginals; push/work/box_abc_run.py family), k from -3 to +1.
- **Per regime:**
  - the exact weighted population curve;
  - a 400-draw paired Poisson(1) knee bootstrap (R1 also with Dirichlet(gold74 + .5) target draws);
  - 2,000 public-size resamples (n 390, from the regime's knee distribution) for the ordering and ABC likelihood.
- **Sanity checks:**
  - It reproduces the skeptic's c4D5: raw OA-mean a .1 / .2 +.0060 / +.0104 (skeptic +.0060 / +.0103); R1 macro +0.62 / -3.64 e-4
    = OA-mean +.00025 / -.00146 (skeptic +.0003 / -.0014).
  - Weighted AUC vs sklearn: max |diff| 7e-6.
  - U columns hold 89-104 tied values (arithmetic collisions of the .8 / .2 pct blend, which the kernel shares); the tie
    effect is <= 1e-5 AUC.

## 2. Raw OAI and gold74 (R1) curves (MEASURED, macro e-4, OAI units)

| regime / family | .10 | .20 | .25 | .30 | .35 | .40 | .45 | .50 | .60 | optimum (pop; knee p10-p90) |
|---|---|---|---|---|---|---|---|---|---|---|
| P_FT raw sym | +15.0 | +25.9 | +29.5 | +31.6 | +32.2 | +31.4 | +28.0 | +22.7 | +5.4 | .325 (.30-.375) |
| P_FT raw pw | +15.1 | +26.1 | +29.7 | +31.9 | +32.8 | +31.9 | +28.7 | +23.4 | +4.6 | .35 (.30-.375) |
| P_FT raw LP2 | +9.5 | +15.2 | +16.5 | +16.8 | +15.7 | +13.5 | +9.1 | +3.1 | -15.5 | .275 (.25-.325) |
| P_FT raw LPpw | +7.2 | +12.3 | +14.2 | +15.7 | +16.5 | +16.9 | +16.5 | +15.6 | +11.6 | .40 (.325-.45) |
| P_AREA raw sym | +13.8 | +24.7 | +28.7 | +31.3 | +32.3 | +31.3 | +27.8 | +21.8 | -0.1 | .35 (.325-.375) |
| P_AREA raw pw | +12.9 | +22.7 | +25.7 | +27.5 | +27.3 | +25.4 | +20.7 | +12.5 | -13.1 | .325 (.275-.375) |
| P_AREA raw LP2 | +5.2 | +7.1 | +6.0 | +3.9 | -0.2 | -5.8 | -14.0 | -24.7 | -53.3 | .20 (.15-.225) |
| P_AREA raw LPpw | +4.2 | +6.5 | +7.1 | +7.0 | +6.1 | +4.9 | +2.6 | -0.7 | -9.5 | .275 (.225-.325) |
| P_FT R1 sym | +0.6 | -3.6 | -8.4 | -14.7 | -23.3 | -33.5 | -47.2 | -63.8 | -106.8 | .05 (.025-.125) |
| P_FT R1 pw | +1.4 | -2.4 | -6.9 | -13.1 | -21.5 | -31.6 | -44.9 | -61.1 | -105.3 | .075 (.025-.125) |
| P_FT R1 LP2 | +2.1 | -0.3 | -3.3 | -7.8 | -13.4 | -21.0 | -31.1 | -43.5 | -75.1 | .125 (.05-.15) |
| P_FT R1 LPpw | +1.9 | +1.9 | +0.7 | -1.0 | -3.2 | -6.0 | -9.8 | -13.9 | -25.8 | .175 (.075-.20) |
| P_AREA R1 sym | +2.6 | +0.6 | -2.8 | -8.2 | -15.4 | -25.6 | -38.2 | -53.9 | -95.7 | .125 (.075-.15) |
| P_AREA R1 pw | +3.0 | +1.1 | -2.4 | -7.5 | -15.2 | -25.3 | -38.2 | -54.5 | -99.1 | .10 (.075-.15) |
| P_AREA R1 LP2 | +2.6 | +1.2 | -1.3 | -5.3 | -10.8 | -18.5 | -28.3 | -40.8 | -74.3 | .10 (.075-.175) |
| P_AREA R1 LPpw | +2.1 | +2.3 | +2.0 | +0.7 | -1.3 | -3.7 | -7.3 | -11.4 | -23.0 | .15 (.10-.25) |

- Coupling: raw P_FT phi .040 (LPF -.008), c4D5 rho .639, q .404, gap .235; R1 phi .322, rho .683, q .629, gap .053.
- With gold74 target uncertainty added (R1 knee + Dirichlet), the R1 optimum spreads to p90 .175 (sym / pw), .20 (LP2), .30
  (LPpw) on P_FT, and to .225 / .225 / .325 on P_AREA.
- pw beats sym slightly in every regime except P_AREA raw. LP2 / LPpw trail sym / pw except at R1, where they lose least.
- R1 is incompatible with the public reads in this proxy: P(all orderings) .000 (P_FT) / .006 (P_AREA); S20 > S10 alone holds
  in only .07 / .21 of public-size draws.
- Paired knee-bootstrap sd of a single point (OAI size): ~2.7 e-4 at a .2 and ~4.3 at a .3. Public-size sd (n 390): 5.7 at .2 and
  9.0 at .3. That is unpaired; the paired step sds are in sec 0.3.

## 3. Fitted regime (iii)

### 3.1 How it was fitted (MODEL on MEASURED resamples; `oaigrid_post.log` sec 2)
- Each regime gets 2,000 public-size (n 390) resamples. The likelihood multiplies three pieces, as in push/work/box_abc_run.py:
  - an ABC kernel (h 4e-5) of (s S10, s S20) on the 100 LB-posterior draws of (G10, G20), with s log-uniform [.5, 2];
  - the indicator [S10 > 0, S20 > S10, PW30 > S20, PW30 > PW20, PW30 > S30];
  - the indicator s (PW30 - S20) < gap960, because pw30 did not display .960.
- Priors on k: N(0, 1) (primary, as in implied.md) or flat on [-3, 1].

| k (sd) | phi | rho | q | gap | P(order all) | L_mag | L_joint | P_AREA P(order) | P_AREA L_joint |
|---|---|---|---|---|---|---|---|---|---|
| -3.00 | -.03 | .627 | .334 | .293 | .77 | 7.1e-4 | 1.3e-5 | .64 | 1e-22 |
| -2.50 | .03 | .638 | .391 | .247 | .77 | 4.0e-3 | 1.8e-4 | .65 | 5.3e-6 |
| -2.00 | .08 | .649 | .445 | .204 | .67 | 8.4e-3 | 5.5e-4 | .65 | 1.5e-4 |
| -1.75 | .11 | .654 | .471 | .183 | .51 | 9.0e-3 | **6.3e-4** | .62 | 3.3e-4 |
| -1.50 | .14 | .659 | .496 | .163 | .38 | 8.3e-3 | 5.6e-4 | .55 | 5.0e-4 |
| -1.25 | .17 | .663 | .520 | .143 | .23 | 6.8e-3 | 4.3e-4 | .44 | 6.2e-4 |
| -1.00 | .20 | .668 | .543 | .124 | .11 | 4.5e-3 | 2.2e-4 | .30 | **6.3e-4** |
| -0.75 | .23 | .672 | .566 | .106 | .045 | 2.7e-3 | 1.2e-4 | .19 | 3.8e-4 |
| -0.50 | .26 | .676 | .587 | .089 | .018 | 1.4e-3 | 3.2e-5 | .07 | 1.9e-4 |
| 0 (gold74 pairs) | .31 | .683 | .626 | .057 | .001 | 2.0e-4 | 7.6e-7 | .009 | 1.7e-5 |
| +1.00 | .41 | .695 | .693 | .002 | .000 | 2.5e-12 | 0 | .001 | 6e-15 |
| raw | .04 | .639 | .404 | .235 | .54 | 4.7e-3 | 2.1e-4 | .05 | 4.9e-5 |
| MIX .25 | .12 | .651 | .468 | .183 | .40 | 9.5e-3 | 6.5e-4 | .07 | 1.7e-4 |
| MIX .50 | .19 | .662 | .526 | .136 | .18 | 6.7e-3 | 4.1e-4 | .07 | 2.4e-4 |

(phi / rho / q / gap / L columns are P_FT.)

**Posterior:**

| proxy | prior x likelihood | k mean (sd) | MAP | phi | c4D5 gap |
|---|---|---|---|---|---|
| P_FT | **N01 x joint** | **-1.33 (.48)** | -1.25 | .160 | .151 |
| P_FT | flat x joint | -1.68 (.50) | -2.00 | .120 | .178 |
| P_FT | N01 x order only | -1.60 (.56) | -2.00 | .129 | .173 |
| P_AREA | **N01 x joint** | **-1.01 (.41)** | -1.00 | .215 | .095 |
| P_AREA | flat x joint | -1.21 (.45) | -1.00 | .191 | .111 |

- The amplitude s posterior (public gain per unit c4D5 OAI gain) is .67 (P_FT) and .71 (P_AREA).
- What pulls where:
  - The ordering alone keeps pushing toward low coupling: it is monotone in k.
  - The (G10, G20) magnitudes peak at k ~ -1.75 (P_FT) and -1.25 (P_AREA).
  - "pw30 not .960" cuts the k <= -2.5 tail (at -3.0, L_joint drops 40x from L_mag & order).
- Population-mean ordering (no public noise) holds only at k <= -1.75 (P_FT) and k <= -1.25 (P_AREA). At the P_FT MAP, pw30
  sits below eoa20 in population (9.7 vs 10.6), so part of the public pw30 > eoa20 is a public-sample draw (MODEL).
- **Read the regime as an effective shape, not literal hidden phi** (MODEL / GUESS):
  - implied.md fitted D5 alone (rho ~.78) and got k -0.16 (CLAIMED).
  - c4D5 has rho .66 on OAI, so it needs lower label coupling (k ~ -1.0 to -1.3) to reproduce the same public shape. In the
    linear-Gaussian picture, the optimum b ~ (rho - q) / ((1 + rho) / 2 - rho q) is what the reads pin down.
  - The fit is therefore on the right quantity (curve shape) for choosing a, but its phi should not be quoted as the hidden
    label coupling.

### 3.2 Posterior-mixed population curves (N01 x joint; OAI units, macro e-4)

| family | .10 | .20 | .25 | .30 | .35 | .40 | .45 | .50 | .60 | argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| P_FT sym | +8.2 | +11.8 | +11.7 | +9.7 | +6.0 | +0.7 | -7.5 | -18.1 | -47.0 | .225 |
| P_FT pw | +8.9 | +13.1 | +13.0 | +11.6 | +8.3 | +3.1 | -4.5 | -14.8 | -44.9 | .225 |
| P_FT LP2 | +6.8 | +9.5 | +9.1 | +7.6 | +4.7 | +0.5 | -6.0 | -14.5 | -37.9 | .20 |
| P_FT LPpw | +5.2 | +8.4 | +9.2 | +9.5 | +9.1 | +8.3 | +6.6 | +4.4 | -2.4 | .30 |
| P_AREA sym | +8.1 | +12.8 | +13.3 | +12.3 | +9.4 | +4.5 | -3.3 | -14.3 | -47.2 | .25 |
| P_AREA pw | +8.6 | +13.5 | +14.2 | +13.1 | +10.1 | +5.1 | -3.1 | -15.0 | -48.7 | .25 |
| P_AREA LP2 | +6.3 | +9.6 | +9.1 | +7.7 | +4.3 | -0.6 | -8.2 | -18.5 | -46.0 | .20 |
| P_AREA LPpw | +5.0 | +8.1 | +9.2 | +9.6 | +9.1 | +8.4 | +6.7 | +4.0 | -4.0 | .30 |

- Flat-prior mixes sit about 1 step higher: argmax sym .25, pw .275, LP2 .25, LPpw .35 (P_FT); .275 / .25 / .225 / .30 (P_AREA).
- Medial-untouched sym / pw (L and PF as in sym / pw, derived from the per-label deltas at the MAP) peak at only ~+6.4 / +8.0
  (P_FT, a .175 / .20) and ~+4.1 / +6.4 (P_AREA). On OAI, Medial carries about 1/3-1/2 of the gain.

### 3.3 Shape near .3-.45 (MEASURED at the MAP and its neighbours)
- The MAP rows are in sec 0.3.
- **Neighbours (pw, OAI units, steps .30->.35 / .35->.40 / .40->.45):**

  | proxy | regime | pw at .30 | steps | pw optimum (knee p10-p90) |
  |---|---|---|---|---|
  | P_FT | k -1.50 | +14.7 | -2.6 / -4.5 / -6.9 | .25 (.20-.275) |
  | P_FT | k -1.00 | +4.6 | -4.6 / -6.7 / -9.1 | .175 (.15-.225) |
  | P_AREA | k -1.25 | +17.8 | -1.9 / -4.0 / -7.0 | .275 (.225-.30) |
  | P_AREA | k -0.75 | +7.4 | -4.1 / -6.3 / -9.6 | .20 (population argmax) |

- **pw vs sym at the same a** (paired, P_FT MAP): pw - sym at .30 +2.0 (sd 0.9); at .35 +2.3; at .40 +2.5. P_AREA: +0.6 (sd 1.0)
  at .30.
- **Paired vs pw30 at the P_FT MAP:** pw35 -3.7 (1.0), pw40 -9.3 (2.0), pw45 -17.3 (3.1); sym30 -2.0 (0.9); LP2 at .30 -3.4 (2.4);
  LPpw at .30 / .40 -1.0 (2.7) / -2.5 (2.4).

### 3.4 Per-label structure, which does not match gold58 (MEASURED here; gold58 is info only)
Label AUC deltas (e-4, not macro) at .30 / .35 / .40 / .45:

| proxy, family | Medial OA | Lateral OA | PF OA |
|---|---|---|---|
| P_FT MAP, pw | +49 / +43 / +33 / +18 | +119 / +115 / +103 / +79 | -52 / -86 / -130 / -188 |
| P_AREA MAP, pw | +110 / +120 / +128 / +129 | +131 / +133 / +124 / +106 | -92 / -139 / -200 / -283 |
| P_FT MAP, sym | +39 / +24 / -3 / -43 | +105 / +103 / +97 / +81 | -51 / -83 / -118 / -165 |

- On OAI, PF OA loses from the first step in every fitted regime, and Lateral OA carries the gain.
- **gold58 c4 shows the opposite** (task info): PF OA rises to .4 and Lateral OA falls past .35.
- So the per-label split does not transfer. Candidate causes (GUESS): the DESS swap and the R6 / S6fall coupling collapse
  (ROUND5.md A.4), OAI PF prevalence .43-.75, and MOAKS-to-RSNA mapping for PF.
- Consequences:
  - The macro curve shape is the usable output.
  - The LP verdict ("LP loses") rests on OAI's PF loss and is the least trustworthy result here.
  - The Medial ceiling on hidden (gold58 Medial ~.989, CLAIMED implied.md) cannot be represented: c4D5 Medial base AUC on OAI
    is .90 (P_FT) / .85 (P_AREA).

## 4. Posterior predictive public increment over pw30 (MODEL; public units e-4; N01 x joint, [flat])
Conditioned on the public draw that produced the reads, so the noise correlation with eoa10 / eoa20 / pw30 is used.

| entry | P_FT mean (sd) | P(> pw30) | P(shows .960) | P_AREA mean | P(> pw30) | P(.960) |
|---|---|---|---|---|---|---|
| **pw25** | **+0.16 (0.58)** [+0.08] | **.60** [.54] | **.24** [.20] | +0.17 | .61 | .24 |
| eoa25 | -0.77 (0.77) | .14 | .04 | -0.62 | .21 | .05 |
| pw35 | -1.52 (1.07) | .06 | .02 | -1.70 | .05 | .02 |
| eoa35 | -2.98 (1.56) | .00 | .00 | -2.59 | .02 | .01 |
| pw40 | -4.39 (2.21) | .00 | .00 | -4.98 | .00 | .00 |
| pw45 / pw50 | -8.6 / -14.7 | .00 | .00 | -10.3 / -18.0 | .00 | .00 |
| LP2 .20 / .25 / .30 / .35 / .40 | -4.0 / -3.8 / -4.4 / -5.8 / -8.3 | .01-.04 | <= .02 | -3.8 / -3.6 / -4.3 / -6.1 / -9.4 | .00-.04 | <= .02 |
| LPpw .30 / .40 / .50 | -3.9 / -4.1 / -6.0 | .01-.04 | <= .02 | -3.7 / -3.9 / -6.4 | .01-.03 | <= .02 |

## 5. Population (private-relevant) comparisons of built / submitted entries (MODEL; `oaigrid_post2.log`)
The table gives the public-units mean, with P(X > Y) over regime posterior x knee bootstrap in parentheses. Private-sample
noise is not added.

| pair | P_FT N01 | P_FT flat | P_AREA N01 | P_AREA flat |
|---|---|---|---|---|
| pw25 - pw30 | +0.96 (.76) | +0.10 (.52) | +0.83 (.74) | +0.23 (.58) |
| pw25 - pw20 | -0.10 (.48) | +0.72 (.72) | +0.50 (.65) | +1.02 (.77) |
| pw25 - eoa20 | +0.79 (.72) | +1.54 (.87) | +1.04 (.76) | +1.56 (.85) |
| pw20 - pw30 | +1.06 (.64) | -0.62 (.39) | +0.33 (.56) | -0.79 (.40) |
| pw20 - eoa20 | +0.88 (.98) | +0.82 (.97) | +0.54 (.88) | +0.54 (.89) |
| eoa20 - pw30 | +0.17 (.51) | -1.44 (.28) | -0.21 (.48) | -1.33 (.33) |
| eoa25 - pw30 | +0.09 (.51) | -0.75 (.29) | +0.18 (.56) | -0.40 (.42) |
| eoa30 - pw30 | -1.28 (.02) | -1.21 (.03) | -0.50 (.24) | -0.48 (.24) |
| pw35 - pw30 | -2.17 (.07) | -1.15 (.21) | -2.05 (.08) | -1.37 (.19) |
| eoa10 - eoa20 | -2.40 (.16) | -3.78 (.06) | -3.28 (.06) | -4.15 (.03) |
| LP2 .20 - pw25 | -2.36 (.22) | -4.12 (.09) | -3.28 (.11) | -4.42 (.06) |
| LPpw .30 - pw25 | -2.32 (.21) | -3.98 (.09) | -3.25 (.10) | -4.32 (.06) |

- Robust on OAI: pw20 > eoa20 (.88-.98), eoa30 < pw30 (.02-.24), pw35 < pw30 (.07-.21).
- Not resolved: the ordering among pw20 / pw25 / pw30 / eoa20. All lie within ~1e-4 and P is .39-.76.

## 6. Caveats
1. **c4D5 is not c4.**
   - Raptor (~.32 of c4's OA output) is OAI-trained and unmeasurable, so D5 stands in.
   - R6 and S6fall lose coupling on OAI (.55 vs .78-.85 on gold58; CLAIMED ROUND5.md A.4), so c4D5 rho is .64-.70 here vs ~.82
     implied for c4.
   - The fitted regime absorbs this into an effective shape (sec 3.1). The shape transfer is first-order only.
2. **Per-label sign conflict with gold58** (sec 3.4). It caps confidence in LP and in any Medial-specific reading.
3. **Inputs to the read likelihood are CLAIMED:**
   - LB-posterior draws (G10, G20, gap960) from push/work/box_abc_run.py LBD (lb_post.py) and the 05:53 board;
   - public n 390;
   - pw30 enters only as "above eoa20 / pw20 / eoa30, not .960". Its in-band position (4th in .959 at 18:50 after other
     teams moved) was not converted to an unrounded value.
4. **Coarse grid:** k steps are .25 sd in the bulk (.5 at the ends); the a grid is .025. The posterior on k is coarse; the
   argmax resolution is .025.
5. **Public-size noise:** n 390 resamples are drawn from the same 1,704 OAI knees (with replacement), so they slightly
   understate public-sample diversity.
