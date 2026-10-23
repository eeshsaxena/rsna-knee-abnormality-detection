# cand_eoa_curve.md: E-OA curve and asymmetric refinements (RESEARCH ONLY, 10-20)

Nothing built, pushed, committed or submitted. No dataset, no GPU job, no scheduled task. PREREG, HANDOFF, PROGRESS and the
a20 folder untouched. Box runs are read-only CPU python piped over ssh stdin, print aggregates only, write nothing on the box.
Tags: MEASURED (computed in this session), CLAIMED (from an earlier file, source named), MODEL (arithmetic on stated
assumptions), GUESS (judgement).

## 0. Pre-registration (written 2026-10-19 18:45 UTC, before any number of this run was seen)

**Question.** Does any E-OA variant beat the two in-flight shapes, symmetric a .1 (eoa10) and a .2 (eoa20), once OAI is
reweighted to RSNA-expert OA coupling?

**Data.** OAI n 2,399 knees, label proxy P_FT (MOAKS full-thickness cartilage per compartment, as `oai_eoa_check.py`;
primary). P_AREA is a robustness read only. Models: the 3 OAI-free report-trained sets D5mean_s1043, s1044_mix50 and
s1045_288 (gating). Two informational extras, NOT gating: ENS3 (mean of the 3 models' percentiles, an ensemble like c4) and
the 5 single D5 folds (to relate the optimum to model OA rho).

**Transform.** U = within-set percentiles. For target t with the other two OA compartments o1, o2:
out_t = U_t - b_t,o1 * U_o1 - b_t,o2 * U_o2. Each label's AUC depends only on its own (b_t,o1, b_t,o2), so any
multi-label variant is a per-label choice, and its OA-trio delta on a resample is the sum of the per-label deltas.
Symmetric E-OA(a) = b = a/2 for every pair.

**Pre-specified family.**
- F1 symmetric curve: a in {-.1 (control), .05, .1, .15, .2, .25, .3}.
- F2 per-target a (aM, aL, aPF), b = a_t/2 for both others:
  - Lateral and PF only: LP10 (0, .1, .1), LP15 (0, .15, .15), LP20 (0, .2, .2).
  - PF more than Medial: RAMP (.05, .1, .15), RAMP2 (.1, .15, .2), PFH (.1, .1, .2), PFH2 (.05, .1, .2).
  - Lateral heavy: LH (.1, .2, .1).
  - Control, Medial heavy: MH (.2, .1, .1). The hypothesis predicts MH does worse than S.1 / S.2.
- F3 pair-weighted by the gold74 excess gap (coupling map: LOA-PF .42, MOA-LOA .26, MOA-PF .21; weight = gap / mean gap
  = 1.416 / .876 / .708): b_t,o = (a/2) * w(t,o), PW10 (a .1) and PW20 (a .2).
- ORACLE: per model and scheme, the per-label argmax over b_t,o1, b_t,o2 in {0, .025, ..., .15}. Upper bound only. Not
  eligible on its own. A split-half check (choose on one random half, evaluate on the other) is reported as its honesty test.

**Weighting schemes** (knee importance resampling by label cell; predictions given labels kept as observed):
- raw: OAI as is.
- R1oa3: skeptic method. The 3-label OA cell is matched to gold74 (71 complete; +.5 smoothing). Moves coupling AND marginals.
- R3p: 3-label IPF. Each OA pair's 2x2 table is set to the OAI marginals and the gold74 pairwise log OR (+.5 smoothing).
  Moves coupling only, pair-specific. This matters for asymmetric variants.
- R3p_hi / R3p_lo: as R3p with every gold74 log OR +1 / -1 sd (stress, not gating).
- Bootstrap: 400 weighted resamples per scheme (same indices across models, so cross-model reads are paired).

**Decision rule (pre-specified; the task's bar).** A refinement (any F1 a other than .1 / .2, any F2, F3) is reported as a
CANDIDATE only if, on P_FT, for ALL 3 gating models and under BOTH R1oa3 and R3p, its paired increment over S.1 AND over
S.2 is > 0 by more than the bootstrap sd of that increment. Then it must also pass:
- (1) raw OAI SUPPORT for all 3 models: delta vs untransformed > 0, P>0 >= .90, and the S(-.1) control < 0.
- (2) delta vs untransformed > 0 under R1oa3 and R3p for all 3 models.
- (3) gold58 c4 not clearly against: its increment over S.1 on gold58 c4 must not be < 0 with bootstrap P>0 < .10.
  gold58 is a code and consistency check, NOT a selection ruler.
- Mechanism check: on R7 OOF vs report labels (train) the refinement should LOSE versus untransformed.

**Verdict mapping.**
- SUPPORT: at least one refinement is a CANDIDATE and passes (1), (2) and (3).
- INCONCLUSIVE: a refinement beats both S.1 and S.2 by > sd on all 3 models under one scheme only, or on 2 of 3 models
  under both.
- KILL: otherwise. In that case the curve is reported as a map for choosing between eoa10 and eoa20 only.

**Reported regardless.** The full symmetric curve per model and scheme, per-label curves, per-model best symmetric a, the
ENS3 and fold rho relation (MODEL extrapolation to c4 OA rho ~.82, CLAIMED from skeptic_eoa.md), gold58 c4, gold58 and
gold74 R7 OOF, and R7 OOF vs report labels.

## 1. Verdict

**INCONCLUSIVE by the pre-registered mapping. In practice: no refinement is worth a slot.**
- No pre-specified refinement clears the bar (beat S.1 AND S.2 by > 1 boot sd, all 3 models, under R1oa3 AND R3p).
- The best one is PW10 (the gap-weighted shape at a .1). It clears the bar on 2 of 3 models under both schemes, which maps to
  INCONCLUSIVE. It fails on D5 only because D5 prefers a higher a (PW10 vs S.2 for D5: +.00001 / +.00006, sd .00011).
- PW10 passes criteria (1), (2) and (3), and the mechanism check. But its gain over S.1 is +.00004 to +.00006 macro
  (MEASURED). That is about 1/20 of a public LB step.
- The asymmetric "Medial vs Lateral/PF" shapes flip sign between the two OAI cartilage proxies (sec 4), so they are not robust.
- The useful output is the map in sec 2 and sec 5. eoa10 and eoa20 bracket the expected optimum for c4 (~.15, MODEL). Which one
  is better is close to a coin flip, and the expected difference is +.0001 to +.0005 macro.

All OAI numbers are MEASURED on the box (read-only, aggregates only), n 2,399, P_FT unless stated. Macro = OA-trio AUC sum / 12.
Bootstrap: 400 weighted resamples per model and scheme (300 in post-hoc runs, 200 for P_AREA and the folds).

## 2. Symmetric curve: macro delta vs untransformed (MEASURED, `box_curve_{D5,s1044,s1045,ENS3}.log`)

Weighted label phi ML / MPF / LPF: raw .07 / .07 / -.02; R1oa3 .36 / .38 / .22; R3p .37 / .36 / .23; R3p_hi .48 / .43 / .32;
R3p_lo .22 / .26 / .11. ESS 1,794 (R1oa3), 1,965 (R3p), 1,680 (hi), 2,233 (lo).

| model | scheme | a -.1 | .05 | .1 | .15 | .2 | .25 | .3 | best |
|---|---|---|---|---|---|---|---|---|---|
| D5 | raw | -.00165 | +.00080 | +.00156 | +.00226 | +.00289 | +.00343 | +.00384 | .3 |
| s1044 | raw | -.00145 | +.00066 | +.00124 | +.00175 | +.00214 | +.00244 | +.00263 | .3 |
| s1045 | raw | -.00153 | +.00070 | +.00133 | +.00186 | +.00232 | +.00266 | +.00283 | .3 |
| D5 | R1oa3 | -.00060 | +.00025 | +.00041 | **+.00048** | +.00044 | +.00026 | -.00010 | .15 |
| s1044 | R1oa3 | -.00034 | **+.00008** | +.00003 | -.00012 | -.00042 | -.00085 | -.00142 | .05 |
| s1045 | R1oa3 | -.00037 | **+.00009** | +.00008 | -.00005 | -.00032 | -.00075 | -.00141 | .05 |
| ENS3 | R1oa3 | -.00052 | +.00016 | **+.00026** | +.00026 | +.00010 | -.00019 | -.00066 | .1 |
| D5 | R3p | -.00056 | +.00023 | +.00037 | **+.00041** | +.00035 | +.00015 | -.00021 | .15 |
| s1044 | R3p | -.00031 | **+.00006** | -.00001 | -.00016 | -.00047 | -.00092 | -.00150 | .05 |
| s1045 | R3p | -.00032 | **+.00006** | +.00002 | -.00014 | -.00043 | -.00089 | -.00156 | .05 |
| D5 | R3p_hi | -.00022 | **+.00005** | -.00002 | -.00019 | -.00048 | -.00092 | -.00154 | .05 |
| s1044 / s1045 | R3p_hi | +.00007 / +.00008 | -.00014 / -.00015 | -.00041 / -.00040 | -.00079 / -.00080 | -.00134 / -.00134 | | | (none) |
| D5 | R3p_lo | -.00097 | +.00045 | +.00082 | +.00111 | +.00130 | **+.00139** | +.00132 | .25 |
| s1044 / s1045 | R3p_lo | | | +.00047 / +.00053 | **+.00057 / +.00064** | +.00053 / +.00064 | | | .15-.2 |

Reads (MEASURED):
- The R1oa3 rows reproduce the coupling-map R1oa3 numbers for a .1 / .15 / .2 to within .00001 (code parity). R3p (3-label
  pairwise IPF) is new here and agrees with R1oa3 to within .00015 at every a.
- The negative control a -.1 loses everywhere except R3p_hi for the two single folds (+.00007 / +.00008). There, more expert
  coupling makes positive borrowing break even.
- After reweighting, the optimum tracks the model's excess coupling (sec 5). The 5-fold mean D5 peaks at .15. The single folds
  peak at .05 and are about 0 at .1.
- ENS3, the mean of the 3 models, is NOT more coupled than D5: rho .76 / .74 / .76 under R1oa3. Averaging different recipes
  does not raise OA coupling the way c4's blend does, so ENS3 is not a c4 proxy.

## 3. Pre-specified refinements (MEASURED; `merge_curve.out` sec 3 and 3b)

Each count is the number of the 3 gating models where the variant beats S.1 AND S.2 by > 1 boot sd.

| variant (aM, aL, aPF) | R1oa3 | R3p | R3p_hi | R3p_lo | raw | rule |
|---|---|---|---|---|---|---|
| S .05 / .15 / .25 / .3 | 0 / 0 / 0 / 0 | 1 / 0 / 0 / 0 | 3 / 0 / 0 / 0 | 0 / 0 / 1 / 0 | 0 / 0 / 3 / 3 | no |
| LP10 / LP15 / LP20 (Medial untouched) | 1 / 1 / 0 | 1 / 1 / 1 | 3 / 1 / 0 | 0 | 0 | no |
| RAMP / RAMP2 / PFH / PFH2 | 1 / 1 / 1 / 1 | 1 / 1 / 2 / 2 | 2 / 0 / 0 / 1 | 0-1 | 0 | no |
| LH (.1, .2, .1) | 0 | 0 | 0 | 1 | 0 | no |
| MH_ctrl (.2, .1, .1) | 0 | 0 | 0 | 0 | 0 | no (control behaves) |
| **PW10** (gap-weighted, a .1) | **2** | **2** | 3 | 0 | 0 | **INCONCLUSIVE-level** |
| PW20 | 1 | 0 | 0 | 3 | 2 | no |
| ORACLE (per-label argmax, upper bound) | 3 | 3 | 3 | 3 | 3 | not eligible; split-half below |

**PW10 detail.** Increment over S.1, macro, mean / sd:
- R1oa3: +.00004 / .00002, +.00005 / .00002, +.00006 / .00002.
- R3p: +.00004 / .00002, +.00005 / .00002, +.00006 / .00003.
- It is > 0 in all 8 OAI models under R1oa3 and R3p.

PW10 against the three criteria (MEASURED):
- (1) raw: +.00157 / +.00125 / +.00136, P 1.00. Control a -.1 is -.00165 / -.00145 / -.00153. PASS.
- (2) R1oa3: +.00045 / +.00008 / +.00014 (P 1.00 / .77 / .91). R3p: +.00041 / +.00004 / +.00008. PASS on the point estimate,
  but weak for the single folds.
- (3) gold58 c4: identical ranking to S.1 (increment .00000), so it is not against. PASS.
- Mechanism check: against report labels (R7 OOF, train n 2,668) it loses -.00064 vs untransformed, the same as S.1. PASS.

**PW20** beats S.2 in all 8 models under R1oa3 and R3p: +.00011 to +.00012 mean, about 2 to 3 sd. It inherits S.2's level
problem for the single folds.

**ORACLE split-half (honesty test).** The choice is made on one random half of the knees and evaluated on the other.

| model | scheme | out-of-sample increment over S.1 (A to B / B to A) | over S.2 |
|---|---|---|---|
| D5 | R1oa3 | +.00017 (P .85) / -.00003 (P .40) | +.00007 / -.00000 |
| D5 | R3p | +.00022 (P .93) / -.00009 (P .28) | +.00015 / +.00002 |
| ENS3 | R3p | +.00001 / -.00012 | +.00013 / +.00017 |

- The honest upper bound for any shape refinement is about +.0001 macro over S.1.
- The oracle's per-label choices differ across models, with one stable feature: Lateral OA takes its weight from PF OA, not
  from Medial OA. b(L<-PF) is .1 to .15 and b(L<-M) is 0 to .05 in almost every model, scheme and proxy. That led to the post-hoc
  run in sec 6.

## 4. Per-label curves: the Medial vs PF asymmetry depends on the proxy (MEASURED)

Per-label AUC delta at a .1 / .2 under R1oa3 (AUC units):

| model | proxy | Medial OA | Lateral OA | PF OA |
|---|---|---|---|---|
| D5 | P_FT | +.0003 / -.0013 | +.0025 / +.0034 | +.0022 / +.0032 |
| s1044 | P_FT | -.0002 / -.0027 | +.0017 / +.0019 | -.0011 / -.0042 |
| s1045 | P_FT | -.0012 / -.0046 | +.0002 / -.0021 | +.0020 / +.0030 |
| D5 | P_AREA | +.0027 / +.0046 | +.0015 / +.0018 | +.0004 / -.0004 |
| s1044 | P_AREA | +.0025 / +.0040 | +.0003 / +.0001 | -.0008 / -.0042 |
| s1045 | P_AREA | +.0013 / +.0014 | -.0005 / -.0028 | +.0006 / -.0004 |

Reads:
- Under P_FT (full-thickness), decorrelating Medial OA hurts. Under P_AREA (area >= 10 %, any depth), Medial OA gains the most.
- So "Lateral and PF only" helps under P_FT and hurts under P_AREA:
  - LP20 vs S.2, R1oa3: P_FT +.00011 / +.00022 / +.00039; P_AREA -.00038 / -.00033 / -.00011.
  - MH_ctrl flips the other way.
- The RSNA rubric (">50 % thickness loss, ~1 cm+") sits between the two proxies, so OAI cannot fix the direction (MODEL).
- The RSNA expert sets lean towards P_FT, but are small (MEASURED):
  - Medial OA at a .1 / .2: gold74 R7 -.0019 / -.0067; gold58 R7 -.0016 / -.0039; gold58 c4 +.0016 / .0000 (AUC .989, at
    ceiling).
  - PF OA gains in all three: +.0025 to +.0180.

## 5. eoa10 vs eoa20 for c4: where c4 sits on the map (MODEL from MEASURED pieces; `local_gapfit.log`)

The excess coupling a model carries is gap = mean model OA rho (normal scores) minus mean q, where q is the
cross-predictiveness of the other compartments' scores for each label (skeptic's definition).

Fit across 31 points (OAI-free models x reweighted schemes, P_FT):
- inc(.15 vs .1) = -.00032 + .00418 x gap (r .94); zero at gap .076.
- inc(.2 vs .1) = -.00078 + .00865 x gap (r .93); zero at gap .090.
- delta(.1 vs none) = -.00030 + .00767 x gap (r .96); zero at gap .040.
- The P_AREA fit (8 points) crosses zero earlier: .15 at gap .036, .2 at gap .049.

Best a by gap bin (OAI, MEASURED):

| gap | best symmetric a |
|---|---|
| <= .06 | .05 |
| .06 to .07 | .1 |
| .07 to .08 | .15 |
| > .1 | .2 |

Where the RSNA prediction sets sit (MEASURED gap, boot sd from 2,000 resamples), pushed through the fit:

| set | rho | q | gap (sd) | .15 vs .1, P_FT (P>0) | .2 vs .1, P_FT (P>0) | .2 vs .1, P_AREA fit (P>0) |
|---|---|---|---|---|---|---|
| gold58 c4 | .813 | .711 | .102 (.068) | +.00011 (.60) | +.00010 (.52) | +.00052 (.76) |
| gold58 R7 | .753 | .681 | .072 (.072) | -.00002 (.45) | -.00016 (.37) | +.00023 (.62) |
| gold74 R7 | .756 | .721 | .036 (.064) | -.00017 (.24) | -.00047 (.17) | -.00013 (.38) |

Reads:
- c4's point gap (.10) is above every OAI-free model's gap at gold74 coupling (R1oa3 / R3p: max .08, D5 and fold 0; only
  the R3p_lo stress points reach > .1). That puts c4's MODEL optimum at
  about .15 to .2, with a wide band (gap 80 % interval .02 to .19).
- c4 E-OA vs none, MODEL from the P_FT fit:
  - a .1: +.00048, P(>0) about .82.
  - a .2: +.00058, P(>0) about .70.
  - These match the skeptic's +.0005 / +.0007 and P .80 / .70 (CLAIMED, skeptic_eoa.md).
- Direct RSNA reads of .2 vs .1 lean towards .2 but are noisy (MEASURED; consistency only):
  - gold58 c4: +.00097 (P .92).
  - gold58 R7: +.00152 (P .92).
  - gold74 R7: +.00040 (P .64).
- Stress: at +1 sd expert coupling (R3p_hi), .2 vs .1 is negative in all 8 OAI models (mean -.00066).
- Bottom line (GUESS): eoa20 vs eoa10 is roughly 55/45 in favour of eoa20 for c4. The expected difference is +.0001 to +.0005,
  invisible against LB noise. An eoa15 would be the hedge (P > eoa10 about .60 to .83, MODEL) but is worth about +.0001, so
  not worth a slot.

## 6. POST-HOC (not pre-registered): Lateral OA decorrelated against PF OA only (`box_curve_PH_*.log`, `merge_ph.out`)

The variants were suggested by the oracle, so this section cannot produce a SUPPORT.
- PH_L(a): symmetric a, but Lateral OA's whole weight a goes on PF OA.
- PH_PAIR(a): only the LOA-PF pair, weight a each way; Medial untouched.
- PH_PAIRh(a): Medial symmetric at a, plus L and PF decorrelated only against each other.

Results (MEASURED):
- **PH_L10 vs S.1** is the one proxy-robust shape change. It is > 0 in every model and both proxies under R1oa3 and R3p:
  - P_FT: +.00005 / +.00007 / +.00002; the 6 other models average +.00005 (6 / 6 > 0).
  - P_AREA: +.00011 / +.00016 / +.00014.
  - Size: +.00002 to +.00022 macro.
- **PH_PAIRh15 and PH_L15** pass the full bar under P_AREA (3/3 under R1oa3 and R3p) but fail under P_FT (single folds < S.1).
  This is proxy-dependent again.
- **PH_PAIR10** passes on 2 of 3 models under P_FT for both schemes (D5 fails vs S.2), and 1 of 3 under P_AREA.
- **RSNA consistency** (`local_curve_ph.log`):
  - PH_PAIR20 is the best read on gold58 R7 (+.00295; +.00217 vs S.1, P .96) and gold74 R7 (+.00162; +.00097 vs S.2, P .85).
  - On gold58 c4 it reads +.00185 vs S.2's +.00239.
  - Against report labels it loses (-.00145), as the mechanism predicts.
  - The PAIR family's RSNA reads come mostly from leaving Medial OA alone. Section 4 shows that choice is proxy-fragile on OAI.
- **Read (GUESS):** the size of any shape refinement is +.0001 to +.0003. That is not a reason to spend one of the last slots.

## 7. Criteria table for the verdict

| criterion | PW10 (best pre-specified) | ORACLE | LP20 / RAMP2 / PFH |
|---|---|---|---|
| refinement bar (all 3 models, R1oa3 and R3p, vs S.1 AND S.2 > sd) | 2/3 and 2/3 | 3/3 and 3/3 (in-sample; split-half fails for D5 and ENS3) | <= 2/3 |
| (1) raw SUPPORT + control | yes | yes | yes |
| (2) > 0 after reweighting, all 3 models | yes (point) | yes | no (s1044 < 0) |
| (3) gold58 c4 not against | yes (= S.1) | n/a | yes |
| report-label OOF loses | yes (-.00064) | n/a | yes |
| proxy-robust (P_AREA) | yes (+.00003 to +.00005 vs S.1) | per-label choice flips | no (flips) |

## 8. Files (all in this folder)

- **Pre-registration:** `cand_eoa_curve.md` sec 0, written before the first number was seen.
- **Box scripts (read-only, piped):**
  - `box_curve.py`, jobs D5 / s1044 / s1045 / ENS3 / FOLDS / AREA, logs `box_curve_<JOB>.log`.
  - `box_curve_ph.py` (post-hoc), jobs PH_FT_A/B/C and PH_AREA.
- **Merges:**
  - `merge_curve.py` > `merge_curve.out`.
  - `merge_ph.py` > `merge_ph.out`.
- **Local RSNA checks:**
  - `local_curve.py` > `local_curve.log` (gold58 c4 rebuild, parity macro .94102; gold58 and gold74 R7 OOF; R7 vs report labels).
  - `local_curve_ph.py` > `local_curve_ph.log`.
- **Gap fit and c4 extrapolation:** `local_gapfit.py` > `local_gapfit.log`.
- **Governance:**
  - Every box run printed aggregates only (AUC deltas, sd, P, ESS, phi, rho, q, n, prevalence) and wrote nothing on the box.
  - No OAI id, row or per-knee value left the box. No GPU job.
  - Nothing built, pushed, committed or submitted.
