# skeptic_eoa.md: adversarial check of E-OA (OA decorrelation) before the 10-19/20 decision. RESEARCH ONLY.

Nothing was pushed, committed, submitted or built; no dataset; no GPU job. Box use: one read-only CPU python run piped via stdin
(nothing written on the box), printing aggregates only (no id, row or per-knee value left the box).
Work files: `a20/skeptic_eoa_work/` (`box_reweight.py` + `.log`, `local_q.py` + `.log`).
Tags: MEASURED (computed now), MODEL (arithmetic on stated assumptions), GUESS (judgement).

## 0. Verdict
- **Refuted: FALSE on the sign, TRUE on the size.** The OAI result (+.0029 macro at a .2) does not carry over. When OAI is
  reweighted so its expert labels are coupled like gold74's, the gain falls 5 to 7 times, to +.0004 to +.0006 macro. It turns
  negative at a .2 once label coupling is about 1 sd above the gold58 estimate.
- Hidden macro delta, c4 + E-OA (MODEL):
  - **a .1: about +.0005, 80 % interval [-.0002, +.0012], P(delta > 0) about .80**
  - **a .2: about +.0007, 80 % interval [-.0007, +.0021], P(delta > 0) about .70**
- a .1 has about the same mean, half the spread and a higher P > 0. If one entry gets built, a .1 has the better risk shape.
  a .2 is only better if hidden OA coupling is at or below gold58's.

## 1. Transfer: does E-OA survive RSNA-level label coupling? (task 1)
Latent (tetrachoric) label correlation among the 3 OA compartments (MEASURED):

| label set | latent corr ML / MPF / LPF |
|---|---|
| OAI P_FT | .12 / .13 / -.01 |
| gold58 | .53 / .58 / .47 |
| gold74 (71 complete) | .60 / .61 / .41 |
| reports v2 (n 2,713) | .83 / .80 / .77 |

Phi: gold58 .32 / .37 / .28; gold74 .38 / .40 / .24.

**Theory (MODEL, Gaussian, equicorrelated scores).** Let rho be the model's OA-OA correlation. Let
q = corr(f_other, Y_k) / corr(f_k, Y_k), the "cross-predictiveness" of the other compartments' scores for label k.
- out_k = f_k - a*(f_j + f_l)/2 raises the latent correlation with Z_k by the factor
  (1 - a q) / sqrt(1 - 2 a rho + a^2 (1 + rho) / 2).
- That factor is above 1 only if q < rho.
- Break-even a = 2 (rho - q) / ((1 + rho)/2 - q^2).
- So E-OA works only when the model couples the OA trio more than the labels justify.
- Label phi does not enter directly. It enters through q.

**Box check (MEASURED, `box_reweight.log`).** OAI teacher_D5_mean_v00, n 2,399. Knees are importance-resampled by their 3-label OA
cell; predictions given labels are left untouched. 300 weighted resamples (200 for tilts). Values are OA-mean dAUC; macro = /4.

| population | label phi (mean) | model rho | q | a .1 | a .2 | a .3 | a .5 |
|---|---|---|---|---|---|---|---|
| R0 OAI as is (P_FT) | .04 | .758 | .51 | +.0062 | +.0115 | +.0153 | +.0162 |
| tilt theta .5 | .16 | .771 | .60 | +.0041 | +.0070 | +.0082 | +.0024 |
| tilt theta 1.0 | .29 | .784 | .69 | +.0019 | +.0023 | +.0006 | -.0125 |
| **R1 gold74 cells (P_FT)** | **.32** | **.785** | **.71** | **+.0016 (P 1.0)** | **+.0017 (P .98)** | -.0002 | -.0141 |
| **R1 gold74 cells (P_AREA)** | **.32** | **.797** | **.74** | **+.0019 (P 1.0)** | **+.0026 (P 1.0)** | +.0011 | -.0134 |
| tilt theta 1.5 | .41 | .794 | .77 | -.0001 (P .44) | -.0020 (P 0) | -.0065 | -.0267 |

- Tilt theta >= 2 is degenerate: the marginal solver drifts and label phi falls back. Ignore those rows.
- What the table shows:
  - At gold74 coupling, E-OA still gains on OAI: +.0004 to +.0006 macro, P >= .98 within OAI.
  - The optimum moves from a >= .5 down to about .15, and break-even falls to a ~.3.
  - Break-even label phi (mean) is ~.36 for a .2 and ~.40 for a .1. The gold58 point estimate of mean phi is .32, with sd about .09
    at n 58 (MODEL).
  - So P(hidden coupling < break-even) is ~.67 at a .2 and ~.81 at a .1, scored with the OAI teacher's coupling.
- Theory vs measurement: the Gaussian theory overstates the OAI gain about 1.4x at a .1 and about 2x at a .2. The empirical
  break-even q is ~.012 (a .1) and ~.027 (a .2) below theory. I apply those corrections below.

## 2. Does c4 couple the OA trio as much as the OAI teacher? (task 2) It couples MORE. That works against refutation.
MEASURED (`local_q.log`, fam rerun):

| prediction set | data | rho | q | gap rho - q |
|---|---|---|---|---|
| c4 final (rebuilt through the c4 notebook's cell-18 `_s6f_blend`) | gold58 | .816 (normal scores; Spearman .85 / .82 / .81) | .714 (boot sd .083; P(q < rho) .94) | .10 |
| R6 | gold58 | | | .09 |
| S6fall | gold58 | | | .13 |
| r384 | gold58 | | | .05 |
| r224 | gold58 | | | .05 |
| R7 OOF | gold58 | | | .07 |
| teacher recipe s1044 | gold58 | | | .03 |
| OAI-trained s1043 | gold58 | | | .00 |
| R7 OOF | train, n 4,406 | .75 (gold58 also .75, so gold58 rho is representative) | | |
| R7 OOF vs report v2 (n 2,713) | train | .78 | .84 | negative: E-OA loses -.0026 (a .1) / -.0072 (a .2), as the theory predicts |

- Across the 7 gold58 sets, corr(gap, a .2 delta) = .59.
- Why c4's gap is bigger: blending averages away member-specific noise. That raises rho, while q (a covariance ratio) stays put.
  So an ensemble has more room for E-OA than any single member. That is why c4 is the largest gold58 gainer.
- This is real and not a selection artifact: rho does not depend on the labels.

## 3. Does the hidden rubric reward coupling? (task 3) No.
- flat_733343, Overview: "Osteoarthritis, or cartilage loss, is assessed separately in each of these three compartments". The
  Lateral and PF criteria are "the same criteria applied to" that compartment. Borderline cases are graded negative.
- "Labels are assigned at the level of the whole examination, for a single knee" means study level, not a global OA grade.
- flat_733491: the host says report wording does not determine the labels, and that test labels used the same 2-reader +
  adjudicator image review.
- Nothing in either text makes hidden OA a global knee property. The only coupling in the hidden labels is the real disease
  coupling, estimated at latent ~.5-.6 from gold58/gold74. That coupling is exactly what shrinks the gain in sec 1.
- Residual risk: the gold58 coupling estimate itself has sampling noise. That risk is already inside the intervals.

## 4. Hidden-set estimate (MODEL)
- **Gap estimate.** The hidden gap is g = rho_c4 - q_hidden.
  - rho_c4 = .816 +- .035.
  - q_hidden = .72 +- .06. Two sources agree on it:
    - gold58 direct: .714 +- .083.
    - OAI at gold74 coupling: .71 to .74, with dq/dphi ~ .6, so +-.055 from phi uncertainty.
  - So g ~ N(.096, .069).
- **Break-even gap**, empirically corrected theory at rho .816: .025 (a .1) and .056 (a .2).
- **Mapping from gap to delta**, taken from the OAI tilt curve (macro, roughly linear near the gap range):
  - a .1: delta = .0075 (g - .029)
  - a .2: delta = .0158 (g - .055)
  - Multiply by about 1.1 for c4's higher base OA AUCs (binormal sensitivity, MODEL).
  - Add finite-test noise of about +-.00013 (a .1) / +-.00025 (a .2) for ~1,300 studies.
- **Result:**
  - a .1: E about +.0005, sd about .0005, P > 0 about .82
  - a .2: E about +.0007, sd about .0011, P > 0 about .72
- I round P down to .80 / .70 for two unmodelled risks:
  - pf10 (a PF OA column move) read negative on the public LB today, and E-OA's gold58 noise correlates .37-.64 with it. That LB
    read is display-level noise, so it is weak evidence.
  - The OA triple was first spotted on gold58 (the skeptic-1 selection charge).

## 5. Where the case for E-OA is strong and where it is weak
- Strong:
  - The mechanism has the right sign in three regimes: OAI expert (+), gold58 expert (+), report OOF (-).
  - It holds on an independent expert set at the RSNA coupling level (P_FT and P_AREA both positive).
  - c4's measured coupling exceeds the teacher's.
- Weak:
  - The magnitude: +.0004 to +.0006 macro on OAI at RSNA coupling, not the +.0025 gold58 headline or the +.0029 raw-OAI figure.
  - Sensitivity at a .2: one sd more coupling in the hidden labels (mean phi ~.41) flips a .2 to -.0005 macro, while a .1 sits at 0.
  - The ~20 pair swaps behind the gold58 c4 number are noise around a smaller true effect, not the effect itself.
