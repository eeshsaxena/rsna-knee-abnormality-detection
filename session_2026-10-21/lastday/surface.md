# surface.md: public response surface of the E-OA family, fitted to our censored reads, and predictions for untried neighbours of swide (10-21 ~20:15 IST, Claude, LASTDAY surface track)

RESEARCH + LOCAL ONLY. Nothing was submitted, pushed, committed, built into a kernel, downloaded or turned into a dataset. No box use at
all (no OAI row, id or per-knee value was read or printed; OAI enters only through earlier box AGGREGATES quoted from older notes), no
GPU, no scheduled-task edit. PREREG, HANDOFF, PROGRESS and the live folders were only read. Everything written is under
`session_2026-10-21/lastday/`.
**Tags.** MEASURED = computed now from files on disk (file named). CLAIMED = older note (named). MODEL = arithmetic on stated inputs
(`work/surface_mc.py`). GUESS = judgement. **Units.** e-4 = 1e-4 macro AUC. y = (unrounded public - .9580) x 1e4, display assumed to
TRUNCATE (.959 = [.9590, .9600)); if Kaggle rounds, every level shifts by -.0005 and no difference changes. Intervals are [p10, p90].
gold58 is not used anywhere.

## 0. Answer

**No.** Under the primary fit, none of the asked untried combinations is predicted to beat swide 57464326 publicly with P >= .6 AND a
non-negative private expectation (MODEL, `work/surface_base.log`, 25 variants in `work/summ.out`).

| candidate (vs swide) | public delta | P(public > swide) | private E (population delta) | P(pop > 0) | R1 increment | variants passing both bars | verdict |
|---|---|---|---|---|---|---|---|
| swide@.30 (= smid = pw30 + L + M) | -0.20 [-0.70, +0.36] | .30 | -0.55 [-1.44, +0.35] | .22 | -0.14 | 0 / 25 (private < 0 in 25 / 25) | DEAD |
| swide@.20 (pw20 + L + M) | -0.75 [-1.46, -0.11] | .06 | -0.53 [-1.67, +0.55] | .28 | -0.01 | 0 / 25 | DEAD |
| MCLmm .05 on swide | -0.23 [-1.29, +0.82] | .40 | -0.01 [-1.12, +0.97] | .53 | +0.10 | 0 / 25 | DEAD |
| MCLmm .15 on swide | -0.02 [-1.07, +1.05] | .48 | -0.25 [-1.23, +0.85] | .35 | -0.00 | 2 / 25 (no ruler; linear dose); private < 0 in 22 / 25 | DEAD |
| lomm .03 on swide | -0.19 [-0.94, +0.56] | .37 | -0.14 [-0.76, +0.47] | .39 | +0.08 | 0 / 25 | DEAD |
| lomm .08 on swide (swide08) | +0.11 [-1.03, +1.24] | .56 | +0.03 [-0.89, +0.96] | .52 | +0.23 | 4 / 25 (low / zero dose curvature, no ruler, no link) | WEAK |
| MOACF .05 on swide (built: moacfswide) | +0.33 [-2.08, +2.71] | .57 | +0.35 [-0.18, +1.01] | .66 | +0.26 | 2 / 25 (MOACF prior regime / optim) | WEAK (closest single change) |
| swide08 + MOACF (two changes) | +0.44 [-2.21, +3.11] | .58 | +0.38 [-0.77, +1.54] | .66 | +0.36 | 7 / 25 (all optimistic corners) | WEAK |
| MCLmm .15 + MOACF | +0.31 [-2.32, +2.93] | .56 | +0.10 [-1.10, +1.40] | .52 | +0.16 | 5 / 25 | WEAK |
| lomm .08 + MCLmm .15 | +0.10 [-0.56, +0.78] | .57 | -0.21 [-1.41, +0.99] | .41 | -0.06 | 3 / 25 | DEAD (private < 0) |

- "R1 increment" = E[population delta x 1(public X > public swide)]: the change in A's private expectation from entering X when A = the
  better public of swide and X. "Private E" = what switching to X unconditionally would do.
- **Nothing can show .961** (P .000 for every row): swide sits at ~.96008, 9e-4 below .961. A public win is invisible on display and
  worth a few places inside the .960 band (GUESS: ~5-7 teams per 1e-4 near the band bottom, 51 teams in .960 MEASURED).
- **The E-OA dose is done.** swide@.30 / @.20 lose publicly (P .30 / .06) and privately in every variant. MCLmm and lomm re-doses
  ride the public noise already seen: P(public) .37-.56, private E -0.25..+0.03.
- **The only private-positive single change is MOACF(swide)**, P .57 (range .40-.64), R1 +0.26e-4. Times P(clean run) ~.55
  (CLAIMED canary/CANARY_1021.md), that is ~+0.14e-4: about 0 silver points (CLAIMED LEVERS_1021 sec 2.4 slope .5 pt per 1e-4).
  It is already built and gated (LEVERS_1021 sec 3.1: moacfswide, nb e85ceb04). It is the same optional lottery LEVERS_1021
  recommended, not a new lever, and it does not clear the P >= .6 bar.
- swide08 (+ MOACF) is not built. LEVERS_1021 sec 4.2 says no lomm .08. This fit agrees: lomm .08 alone is a coin flip on public
  (.56) at ~0 private (+0.03; -0.08 if the public lomm05 signal does not transfer, `rho0`).
- Recommendation (GUESS): no new 10-22 entry on the strength of this surface. If Eesh wants one A-eligible lottery, MOACF(swide) as
  built. A stays swide 57464326; this file changes neither B nor the r6pw20 resubmit question (bpick.md).

## 1. Dataset (MEASURED; `reads.csv`, `work/lb_reads.py` -> `work/lb_reads.log`, `work/ruler_counts.py` -> `work/ruler_counts.log`)

| entry | sid | a / shape | lomm / MCLmm | display | snapshot (UTC) | rank, band, in-band above / below | order facts |
|---|---|---|---|---|---|---|---|
| c4 | 57343435 | 0 | 0 / 0 | .958 | 10-20 04:17 | r217, .958 r194-1530, 23 / 1313 | base |
| eoa10 | 57413380 | .10 sym | 0 / 0 | .959 | 10-20 05:52 | r167, .959 r131-196, 36 / 29 | > c4 |
| eoa20 | 57413368 | .20 sym | 0 / 0 | .959 | 10-20 05:53 | r135, same band, 4 / 61 | > eoa10; the same 65 frozen teams split 29 below eoa10 / 32 between / 4 above eoa20 |
| eoa30 | 57425223 | .30 sym | 0 / 0 | .959 | 10-20 13:20 | censored | < pw30 |
| pw20 | 57425269 | .20 pw | 0 / 0 | .959 | 10-20 13:20 | censored | < pw30 |
| pw30 | 57425970 | .30 pw | 0 / 0 | .959 | 10-20 13:20 / 10-21 03:11 | r140, .959 r137-208, 3 / 68; r146, r143-219, 3 / 73 | > eoa20, pw20, eoa30. Of the 4 frozen teams above eoa20: 2 below pw30, 2 above. 34 .959 teams new since 05:53: 2 above, 32 below |
| pw25 | 57463742 | .25 pw | 0 / 0 | .959 | 10-21 12:03 | censored | < swide; vs pw30 censored (best-id jumped to swide) |
| swide | 57464326 | .25 pw | .05 / .10 | .960 | 10-21 12:03 | r146, .960 r100-150, 46 / 4 | > pw25, pw30; 3 of the 4 below are frozen .960 teams since 03:11 |
| spl / salt / r6pw20 | 57464296 / 57464911 / 57463719 | .25 / .25 / R6 + .20 | .05 / 0, 0 / .1 | .916 | | S6F canary | unobserved |

- Only two orderings are open inside the family: pw25 vs pw30, and how swide's +L +M splits between L and M.
- So the surface is identified by priors along exactly the directions the asked candidates move in. That is the honest reason
  nothing reaches P .6.

## 2. Model (MODEL `work/surface_mc.py`; importance sampling, N 2,000,000, ESS ~8,700)

- **Band rulers.** Inside a display band, rival teams are iid with density ~exp(-k x) on [0, 1).
  - k ~ U(.2, .8), from the MEASURED count ratios .959 / .960 / .961 = 80 / 51 / 28; variant uniform.
  - eoa10 / eoa20: the exact multinomial posterior Dirichlet(30, 33, 5) on the 65 frozen teams, importance-corrected to a flat prior
    on y.
  - pw30: Beta(3, 3) on the eoa20 -> .960 gap (the 2 / 2 frozen split), times the 34 new teams (2 above / 32 below).
  - swide: G^4 (1 - G)^46 in .960.
- **pw curve near a .2-.3 (A = a / .1).** public pw(A) = y30 + (v + w)(A - 3) - c (A - 3)^2 + rho_A.
  - Population slope v ~ N(-2.0, 1.5) and curvature c ~ logN(ln 2.1, .4). These fit the implied2 population means pw20 -0.06 /
    pw25 +0.50 / pw35 -1.83 vs pw30 (CLAIMED val/implied2.md sec 3), with the sd widened over its skeptic variants (sec 8.1).
  - Public noise slope w ~ N(0, 2.8): paired public sd pw25 - pw30 1.4 per .05 (CLAIMED implied2 sec 8.1). The noise is linear in
    a, as in m12_lbnoise.log (sd 2.1e-4 at a .1 -> 4.2e-4 at .2, CLAIMED).
  - Roughness rho ~ N(0, .3) (GUESS).
- **Shape link** (soft observation): public pw20 - eoa20 ~ N(+.7, 1.03). The population part is +.45..+.95 in 18 / 18 skeptic
  variants and the paired public sd is 1.0 (both CLAIMED implied2 sec 8.0-8.1).
- **eoa30** = eoa20 + (eoa20 - eoa10) - 2 cs + noise, cs ~ logN(ln 2, .5) (GUESS). It enters only through its .959 display and
  eoa30 < pw30.
- **Stack** (priors copied verbatim from stack/skeptic/skep_value_mc.py world(), CLAIMED). swide = pw25 + L + I + M, where:
  - L = muL + eLp: lomm05 on pw30; stack 3-scenario prior x .85; read sd 1.00; corr -.2 with pw25's public noise;
  - I(pw25, lomm05) ~ N(.16, .12);
  - M = muM + eMp: MCLmm .1; skep2 prior; read sd 2.40.
- **Re-dose curves.** For r = new dose / built dose:
  - population mu(r) = r mu1 - kap r (r - 1); public noise r e1 + eta, eta sd .3 x read sd x |r - 1|; I scales with r
    (plan1022 convention); I(a) = I x (.3 - a) / .05 across E-OA doses, so I(.30) = 0 and I(.20) = 2 I.
  - **kapL ~ N(.30, .12)+, MEASURED from night/adj_c12.log.** Fitting the quadratic to the raked and raw LOA <- MM curves at .05 /
    .10 gives curvature .18, .24, .29, .31, .41 (5 raked nests) and ~.29-.37 (raw labels). The curvature does not depend on the
    target coupling.
  - **kapM = .8 x U(.45, .85), MEASURED from stack/stack_mcl.log.** RAK N_A at .05 / .10 / .15 / .20 = +1.95 / +3.15 / +4.19 /
    +4.74 fits s 3.93, c .78. The raw-report curve gives c .93. Both are multiplied by the skep2 RAK -> public factor.
- **MOACF.** Q = muQ + eQp, read sd 1.8 public (MEASURED levers/moacf_noise.log). muQ uses the LEVERS skeptic mixture
  (levers/lever_value_mc.py, CLAIMED).
- **Private.** Private sample noise is independent of the public one, so the private expectation = the posterior mean of the
  population delta.

## 3. Fit (MODEL `work/surface_base.log`; cross-checks CLAIMED)

| entry | unrounded public [p10, p90] | cross-check |
|---|---|---|
| eoa10 | .95938 [.95930, .95945] | implied2 .95937 |
| eoa20 | .95987 [.95981, .95992] | implied2 .95981 (joint) / LB-only higher |
| pw20 | .95985 [.95976, .95992] | implied2 .95982 |
| pw30 | .95993 [.95990, .95997] | implied2 .95991 |
| pw25 | .95994 [.95988, .95999] | (new) |
| swide | .96008 [.96004, .96012] | bpick uniform ruler ~.96010 |

- pw30 -> .960 gap +0.66 [+0.34, +1.02] (implied2: LB-only 0.5, joint 0.75).
- **pw25 - pw30 public +0.03 [-0.51, +0.52], P(pw25 > pw30) .58.** Before the A22 read the same model says +0.17, P .65
  (implied2 +0.09, P .56). The .959 display of pw25 trims the upper tail.
- swide - .960 = +0.77 [+0.38, +1.21]. Stack public T = L + I + M = +1.39 [+0.75, +2.11] (bpick: +2.03 with the dose world's
  lower pw25 and a uniform ruler; the uniform variant here gives +1.47).
- The split is the prior's: L05 public +0.50 [-1.35, +2.32], M10 public +0.73 [-1.14, +2.67]. Population posteriors: muL +0.37,
  muM +0.29, I +0.16.
- Population pw25 - pw30 = +0.39 [-0.48, +1.26]. pw25 is the private-better dose, as before.

## 4. Robustness (MODEL `work/summ.out`, 25 variants)

- **Variants:** seed; uniform band density; no swide ruler; no shape link; lomm05 skeptic prior; MCLmm prior stack / zero / neg;
  re-dose curvature high / low / zero (linear); tau .6; E-OA curvature flat / steep; noise slope sd 2.0 / 4.0; roughness .8; lomm
  read sd 1.3; MOACF prior regime / optim / pess / flip; MOACF read sd 1.5; rho0 (the private effect of the change is independent
  of everything public).
- **swide@.30 / @.20:** private E < 0 in 25 / 25 (-0.77..-0.29 / -0.89..-0.35).
  - swide@.30's public P reaches .69 only without the pw20-eoa20 link. Without it the pw20 position is prior-only and the fitted
    public slope gets steep.
  - Its private E stays -0.32 even there. Either way it fails.
- **MCLmm .15:** private E < 0 in 22 / 25 (>= 0 only with no swide ruler, the optimistic MCLmm stack prior, or a linear dose curve).
- **lomm .08:** P .43-.68. It passes only with dose curvature <= .15 (low or zero), no swide ruler, or no link. It is
  private-negative under the lomm05 skeptic prior (-0.23), high curvature (-0.16), rho0 (-0.08) and the MCLmm stack prior (-0.07).
- **MOACF:** P .56-.58 under every knob that is not its own prior. Under its priors: regime .64, optim .62 (the only passes), pess
  .52, flip .40. With read sd 1.5 it is still .58.
- **swide08 + MOACF:** P .43-.65, passing in 7 / 25, all optimistic corners (MOACF regime / optim, linear or low curvature, no ruler,
  no link, MCLmm neg). It is .54 under the lomm05 skeptic prior and .55 at high curvature.

## 5. Why nothing clears the bar (MODEL reading)

1. **E-OA dose.** swide@.30 vs swide is pw30 - pw25 - I(pw25, lomm05).
   - pw25 vs pw30 is the one E-OA ordering no read resolved. Our fit gives P(pw25 > pw30) .58, set by the curvature prior and
     nudged by pw25's .959 display.
   - The super-additive I (+.16) tilts it further toward swide. Population pw25 > pw30 (+0.39), so going back to .30 costs
     privately in every variant.
   - swide@.20 loses the pw20 - pw25 public step (pw20 sits ~0.9 below pw30).
2. **Stack re-doses.**
   - The A22 read only says L + I + M ~ +1.4 public. spl and salt canaried, so the L / M split is the prior's.
   - Under linear noise scaling, raising a dose extrapolates the public noise already seen, minus curvature. The public P
     therefore hovers near .5, while the private E stays ~0 (lomm .08) or negative (MCLmm .15). That is the "re-select on noise
     already seen" pattern LEVERS_1021 sec 4.2 warned about, now quantified.
   - MCLmm's public read sd (2.4) is twice its prior sd, so most of its share of the +1.4 is noise.
3. **MOACF** is column-disjoint and so untouched by the swide selection. Its public P is set by its prior mean (+0.35) against a
   read sd of 1.8: Phi(.35 / 1.87) ~ .57. Only a more optimistic prior lifts it over .6.
4. **Two-change combos** add two weak positives: P ~.58, private ~+0.38 for swide08 + MOACF. They still miss .6 in the base. The
   read could not tell which change helped, and lomm .08 is unbuilt.

## 6. Caveats

- **Band rulers** assume rival teams are iid inside a band. The pw30 rulers partly overlap (frozen 2 / 2 split + new teams), and
  dropping the swide ruler moves the stack T from +1.4 to +3.3 (variant noswr). The ruler is real data, so it stays in.
- **The E-OA curvature and the pw20-eoa20 link are box-proxy priors** (CLAIMED implied2), not public data. They decide pw25 vs
  pw30. The private verdict on swide@.30 / .20 holds without them.
- **Re-dose curvature** comes from report / raked rulers (MEASURED in earlier logs); its transfer to expert labels is GUESS.
  Linear noise scaling with tau .3 is GUESS; tau .6 changes nothing material.
- **P(clean) ~.55** for any 10-22 entry (CLAIMED CANARY_1021). A canaried entry reads .916, so it can never become A. That costs
  no private value but roughly halves every R1 increment above.
- Additivity across columns is exact for MCLmm / MOACF (disjoint columns, CLAIMED stack/inventory.md). For lomm x E-OA it rests on
  I(a) linear in dose (GUESS, plan1022 convention).

## 7. Files (all under `session_2026-10-21/lastday/`)

- `reads.csv`: the dataset (one row per read; canary rows marked unobserved).
- `work/lb_reads.py` -> `work/lb_reads.log`, `work/reads_lb.json`: our row, band and frozen rulers per read snapshot (read-only).
- `work/ruler_counts.py` -> `work/ruler_counts.log`: non-frozen team counts for the pw30 / swide rulers (read-only).
- `work/surface_mc.py` -> `work/surface_<tag>.log / .json` (base, preA22 and 25 variants); `work/run_var.sh` runs the variants.
- `work/summ.py` -> `work/summ.out`: per-candidate variant table with PASS flags.
