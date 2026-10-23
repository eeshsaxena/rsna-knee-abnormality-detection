# cand_menisci.md: Medial Meniscus <-> Lateral Meniscus decorrelation (10-20 night, Claude). RESEARCH + MEASUREMENT ONLY.

Tags: MEASURED = computed tonight (log named); CLAIMED = older note says so; MODEL = arithmetic on stated assumptions;
GUESS = judgement. MM / LM = medial / lateral meniscus. "Macro" = 12-label macro AUC = (dAUC_MM + dAUC_LM) / 12.

## 0. Pre-registration (written before cand_menisci_box_run.py was run)

Honesty note: the directional single-pair rows for MM <- LM and LM <- MM already existed in `box_coupling.log` (coupling map,
10-19), and they were negative on OAI at a .1 for all 3 models. So this pre-registration was written before THIS run, not
before any related number was seen. The rule is copied from the task's three criteria and the E-OA rule in `oai_eoa_check.py`.

Transform: MM' = pct(f_MM) - a * pct(f_LM), LM' = pct(f_LM) - a * pct(f_MM).
Tested a: .1 and .2; control a = -.1. a = .05 / .15 are printed as descriptive only and cannot be selected.

- Criterion 1 (OAI, B1 tear labels, n about 2,399, paired bootstrap 2,000): SUPPORT at a if macro delta > 0 AND P>0 >= .90 for
  ALL 3 OAI-free models, AND the a -.1 control is < 0 for all 3. KILL if delta <= 0 or P>0 < .5 for 2+ of the 3 models.
  Else INCONCLUSIVE.
- Criterion 2 (transfer): at the supported a, the delta stays > 0 for all 3 models under R2 (OAI marginals + gold74 MM-LM log
  OR) and R1 (gold74 4-cell joint). R2m (5-label IPF) is a second read; R2hi / R2lo are stress tests, reported, not gated.
- Criterion 3 (gold58 c4 consistency, NOT a ruler): c4 macro delta not clearly negative (bootstrap P>0 not < .10).
- Mechanism check: R7 OOF vs report_labels_v2 should LOSE (delta < 0).
- B2 (adds intrasubstance signal) and B3 (tear without maceration) are sensitivity reads, not gates.
- Overall: SUPPORT only if 1 + 2 + 3 pass; KILL if 1 is KILL; else INCONCLUSIVE.

## 1. Answer

**Verdict: KILL.** Criterion 1 fails for all 3 OAI-free models at both tested a, and criterion 2 fails harder (MEASURED).

- **OAI expert tear labels:** the symmetric menisci decorrelation LOSES at a .1 and at a .2 for every model.
  - The negative-a control loses too.
  - So the OAI optimum sits at a of about 0 to .05. Descriptive a .05: +.00000 / -.00001 / -.00012 macro.
- **Reweighting to RSNA coupling makes it worse:** -.0005 to -.0009 macro at a .1, P>0 <= .03 for all 3 models.
  - The reason is that RSNA experts couple the two menisci MORE than OAI experts do (gold74 tet +.18 +- .17 vs OAI -.01).
  - At gold74 +1 sd coupling the sign flips: the a -.1 BORROW gains (+.00013 / +.00017 / +.00039).
- **gold58 c4 is mildly positive at a .1:** +.00057, P>0 .75.
  - All of it comes from Lateral Meniscus (+.0068, one or two rank swaps in n 58).
  - It is negative at a .2 (-.00090).
  - Not clearly against, but gold58 is not a ruler and cannot rescue a criterion-1 KILL.
- **Report-label OOF loses at a .1** (-.00033, P>0 .000), as the mechanism predicts.
  - But the a -.1 control also loses there (-.00012 / -.00021), so the check does not discriminate for this pair.
- **Expected hidden macro** (MODEL):
  - about -.0002 at a .1 and about -.001 at a .2
  - P(gain > 0 at a .1) about .25 (GUESS)
- **Do not build.** No a is worth a slot.

## 2. OAI, criterion 1 (MEASURED, `cand_menisci_box.log`, n 2,399 knees, paired bootstrap 2,000)

**Labels.** B1 (primary) = tear, T123 == 1 (MOAKS morph 2-8 or root tear).
- Prevalence: MM .411, LM .206.
- Expert phi MM-LM -.006 (tet -.01).

**Model coupling.** Spearman MM-LM: D5 .124, s1044 .135, s1045 .098.
- The gap (model minus expert tet) is about +.11 to +.15, the same size as c4's gold74 gap of +.16.
- So raw OAI is the gap-matched test (MODEL).

| model | base AUC MM / LM | a -.1 (control) | a .1 (P>0) | a .2 (P>0) | a .05 descr (P>0) | a .15 descr |
|---|---|---|---|---|---|---|
| D5 mean s1043 | .9312 / .9298 | -.00046 | -.00015 (.065) | -.00089 (.000) | +.00000 (.53) | -.00045 |
| s1044 mix50 | .9262 / .9223 | -.00042 | -.00017 (.044) | -.00093 (.000) | -.00001 (.40) | -.00049 |
| s1045 288 | .9369 / .9285 | -.00023 | -.00040 (.000) | -.00137 (.000) | -.00012 (.02) | -.00083 |

- Per label at a .1: MM -.0012 / -.0013 / -.0022, LM -.0006 / -.0008 / -.0025.
- This reproduces the directional rows of the coupling map (`box_coupling.log`): both directions lose at a .1.

**Read: KILL by the pre-registered rule.** Delta <= 0 and P>0 < .5 for all 3 models, at a .1 and at a .2.
- The control is negative as well.
- So the AUC surface is concave with its top near a = 0 to .05. This is not a pass with a missing control.

**Sensitivity labels** (MEASURED, 500 boots). Macro at a -.1 / .1 / .2:

| proxy | D5 | s1044 | s1045 |
|---|---|---|---|
| B3 tear without maceration (n 2,011; morph code map GUESS) | -.00045 / -.00023 / -.00108 | -.00046 / -.00020 / -.00103 | -.00017 / -.00049 / -.00157 |
| B2 tear OR intrasubstance signal (MM prev .805) | -.00106 / +.00059 / +.00056 | -.00099 / +.00051 / +.00052 | -.00091 / +.00037 / +.00020 |

- B3 agrees with B1.
- **B2 points the other way:** +.0004 to +.0006 at a .1 (P>0 1.00, driven by MM, +.005). It is not the hidden regime and
  not a gate:
  - B2 counts intrasubstance signal as positive. `build_oai_labels.py` itself codes that as "not a tear", and no tear rubric
    counts it.
  - gold74 MM prevalence (.48) matches tear semantics (B1 .41), not B2 (.81).
  - The RSNA rubric wording is a GUESS.

**Linear-Gaussian theory** (MEASURED inputs; MODEL formula a* = (rho - q) / (1 - rho q)):

| quantity | OAI value | a* |
|---|---|---|
| rho_ns | .12-.15 | |
| q_MM | .08-.09 | .03-.05 |
| q_LM | .02-.03 | .10-.13 |

- The theory says LM <- MM should gain near a .1, yet LM loses there for all 3 models. So the theory overstates the gain for
  this pair.
- MM's a* is already below .1.

## 3. Transfer, criterion 2 (MEASURED, coupling-matched reweighting, 400 weighted resamples)

- gold74 MM-LM 2x2 [c00, c01, c10, c11] = [25, 13, 19, 16] (n 73).
- gold74 log OR +.47 +- .48; OAI B1 log OR -.03.

Macro, each cell a -.1 / .1 / .2:

| scheme | weighted phi MM-LM | ESS | D5 | s1044 | s1045 |
|---|---|---|---|---|---|
| R1 (gold74 4-cell joint) | .113 | 1,847 | -.00016 / -.00050 / -.00166 | -.00012 / -.00052 / -.00170 | +.00010 / -.00078 / -.00218 |
| R2 (OAI marg + gold74 OR) | .098 | 2,375 | -.00016 / -.00046 / -.00155 | -.00011 / -.00049 / -.00160 | +.00009 / -.00073 / -.00206 |
| R2lo (gold74 OR - 1 sd) | -.009 | 2,399 | -.00044 / -.00017 / -.00093 | -.00040 / -.00019 / -.00097 | -.00021 / -.00041 / -.00140 |
| R2hi (gold74 OR + 1 sd) | .192 | 2,310 | +.00013 / -.00076 / -.00218 | +.00017 / -.00079 / -.00223 | +.00039 / -.00105 / -.00272 |
| R2m (5-label IPF incl. OA trio) | .094 | 1,644 | -.00010 / -.00050 / -.00164 | -.00005 / -.00059 / -.00176 | +.00021 / -.00086 / -.00226 |
| R2m_hi (5-label, OR + 1 sd) | .193 | 1,060 | +.00018 / -.00079 / -.00225 | +.00023 / -.00092 / -.00242 | +.00052 / -.00121 / -.00293 |

- P>0 at a .1 and a .2 is <= .03 in every reweighted row for every model (.00 except R2lo a .1).
- **Read: fails criterion 2 at every tested a.** At RSNA expert coupling, the direction that helps is the opposite one
  (borrowing). This matches the coupling map's reading that the MM-LM excess is "not removable".

**Caveat (MODEL):** this reweighting is harsher than c4's situation.
- It keeps the OAI-free models' low MM-LM coupling (.12) while moving the labels to gold74. That shrinks the model-minus-expert
  gap to about 0.
- c4's situation is rho .34 with a gap of +.16.
- The gap-matched read is raw OAI (sec 2), which also loses. Both reads agree in sign.

## 4. gold58 c4 consistency, criterion 3 (MEASURED, `cand_menisci_local.log`, c4 rebuild parity macro .94102)

| model on gold58 (n 58) | Spearman MM-LM | a -.1 | a .1 (MM / LM; P>0) | a .2 (P>0) |
|---|---|---|---|---|
| c4 final | .338 | -.00072 | +.00057 (+.0000 / +.0068; .75) | -.00090 (.27) |
| R6 (member, descr) | .343 | -.00077 | -.00020 (-.0024 / +.0000; .36) | -.00131 (.18) |
| S6fall (member, descr) | .322 | -.00006 | +.00001 (-.0024 / +.0025; .52) | -.00090 (.20) |
| R7 OOF | .374 | -.00082 | +.00036 (.66) | +.00001 (.49) |
| R7 OOF on gold74 (n 73) | .279 | -.00064 | -.00016 (.37) | -.00051 (.31) |

- gold58 expert phi MM-LM .120 (tet .18).
- c4 theory: a*_MM -.03 (sd .12), a*_LM +.10 (sd .14). This is uninformative at n 58.
- **Read: c4 is not clearly against a .1, but not supportive either.**
  - The bootstrap sd is .00086 against a mean of +.00057.
  - The gain is one label and one or two rank swaps.
  - c4's members do not share it: R6 is negative, S6 is zero.
  - R7 on the larger gold74 set is slightly negative.

## 5. Report-label OOF, mechanism check (MEASURED)

R7 OOF on train minus gold74, vs report_labels_v2. Report phi MM-LM .269 (tet .45); R7 Spearman .444.

| labels | n | a -.1 | a .1 (P>0) | a .2 |
|---|---|---|---|---|
| YES/NO only | 3,847 | -.00012 | -.00033 (.000) | -.00113 |
| UNK = 0 | 4,333 | -.00021 | -.00025 (.000) | -.00096 |

- It loses at a .1, as the mechanism predicts.
- But the a -.1 control loses too, so the report regime is also at its optimum near a = 0.
- So the check does not separate "inherited report coupling" from "no usable coupling" for this pair.

## 6. Why this pair fails while the OA trio works (MODEL / GUESS)

- **OA trio:** a gap of +.2 to +.76, big enough for a .1 to land inside the gain region.
  - Models couple at rho .69-.82.
  - Experts couple at tet -.01 to .13 (OAI) and .39-.61 (gold74).
- **Menisci:** the excess is only +.13 to +.16.
  - The models' MM-LM coupling is weak: .10-.14 for the OAI-free models, .34 for c4.
  - The experts' own coupling on RSNA is .18 tet.
  - The measured AUC optimum on OAI is a of about 0 to .05 (sec 2). So a .1 overshoots and a .2 overshoots badly.
  - Any gain at a .05 is below +.0001 macro on every read. Even if real, it is not worth a slot (GUESS).
- **The inheritance signature exists but is small.** R7 train reproduces the report coupling (.44 vs report tet .45). That is
  small next to the real expert coupling of the two menisci on RSNA.

## 7. Governance and files

- **Box:** two read-only CPU runs piped over ssh stdin (smoke BOOT 30; full BOOT 2,000 / RBOOT 400).
  - Nothing was written on the box.
  - Printed aggregates only: n, prevalence, phi, rho, q, AUC deltas, P, ESS.
  - No OAI id, row or per-knee value left the box.
  - The only data sent TO the box were gold74 aggregate 2x2 counts embedded in the script.
  - No GPU job (box GPU 0 % before the run).
- **Nothing else touched:** nothing built, pushed, committed or submitted; no dataset, scheduled task, PREREG, HANDOFF,
  PROGRESS or a20 file.
- **Files** (`session_2026-10-19/ex/`):

  | step | script | output |
  |---|---|---|
  | box run | `cand_menisci_box_template.py` + `cand_menisci_make.py` -> `cand_menisci_box_run.py` | `cand_menisci_box.log` |
  | local run | `cand_menisci_local.py` | `cand_menisci_local.log` |
  | note | | this file |
