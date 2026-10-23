
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
