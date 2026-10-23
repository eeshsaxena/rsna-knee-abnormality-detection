# cv_shrink: cross-validated shrunk Raptor-weight table around c4 (10-18, Claude)

> **OURS LEG = R7 OOF (FALLBACK), NOT R6.** `g58/R6_gold58_serve.csv` did not appear during the 120 min poll (22:55 to 00:57 IST).
> Every number below uses `R7OOF.csv` as the stand-in for R6, the same proxy all earlier gold58 screens used. If the R6 serve file
> lands later, rerun the four commands at the bottom; the script reads either `L` or `p|L` columns.

Tags: MEASURED = computed here on gold58. MODEL = derived from a stated model. GUESS = judgement.
Read-only work: nothing was submitted or pushed, a19/, PREREG files and scheduled tasks were not touched, the box GPU was not used. All of it ran on the laptop CPU.

## TL;DR

1. **A shrunk 12-label table adds almost nothing over c4 (MEASURED).** The best fully honest setting is ALL labels at lambda .5:
   held-out macro **+0.00014, P>0 .54** (train 29 / test 29, 2000 splits), and **+0.00025, P>0 .53** with a 39/19 split.
   lambda 1 (the unshrunk pick) loses: -0.00067 (P>0 .45).
   MODEL extrapolation to training on all 58: lambda* about .45 to .5, gain about +0.0003. That is too small to matter on the LB.
2. **Using direction agreement to choose the subset fails when done honestly (MEASURED).** NEST_DIR75 (the agreement computed inside the
   training half only) comes out negative at every lambda (lambda .5: -0.00016, P>0 .14). The positive version, FULL_DIR75 = {Medial Meniscus, Fracture},
   +0.00165 at lambda 1, is leaky: its subset was chosen on all 58, held-out halves included.
   A label can have a stable direction and still be wrong. Synovitis has the second-highest nested agreement (.56) and the worst held-out result (-0.00095, P<0 .61).
3. **Fracture is the only label whose move holds up when its weight is chosen on the training half (MEASURED).**
   Fracture alone at its train-half pick (lambda 1) scores **+0.00138 held-out, P>0 .78, P<0 only .05**. The gain rises steadily with lambda,
   so 1.0 beats .9 beats .8. Choosing Fracture as the label to move is still leaky. The honest procedure "move the single best train-half label"
   picks Fracture 42% of the time at n_train 29 and 65% at n_train 39, so its held-out goes from -0.00023 to +0.00018.
   MODEL/GUESS: at n_train 58 it picks Fracture every time (Fracture's in-sample gain is .031 per label, the runner-up's .012).
4. **The PF OA move has no honest support (MEASURED).** PF OA alone at its train-half pick: +0.00001 (P>0 .43, P<0 .35) at 29/29,
   and +0.00027 (P>0 .42, P<0 .24) at 39/19. The fixed pf10 table shows +0.00098 on the B halves, and all of that comes from selection.
5. **A19 slate ranking from this CV (MODEL/GUESS): fr100 >= c2max > fr90 >= c2 >> pf10.** c2max and c2 are fr100 and fr90 plus a coin-flip PF OA move.
   pf10 alone has an expected value of about 0. The table I would actually ship if trained on all 58 is **fr100**, the c4 table with Fracture 1.0. See "Full-data table".

## Setup (MEASURED unless tagged)

- Blend = the c4 kernel 57343435: per label rr = .75 pct(r384) + .25 pct(r224); b = w_c rr + (1 - w_c) pct(ours); final = .8 pct(b) + .2 pct(S6fall).
  Components: r384 = `gold58_r384_kernelserve_cpu.npy`, r224 = `gold58_r224_kernelserve.npy`, S6fall = `n42gpu_gold58.csv`, ours = `R7OOF.csv` (fallback).
  The vectorised scorer matches `screen_a25.final` exactly: HW base macro .935733 in both, and the c4 base gives .937673.
  Floating-point order matters because it changes which values tie: pct is computed before the weights are applied, as in the screens.
- Repeated random splits of the 58 studies. Each split must leave both classes of every label in both halves.
  On half A, per label: w_pick = argmax AUC over the grid 0, .05, ..., 1 (ties go to the value nearest c4). The table is w = c4 + lambda (w_pick - c4) on labels in S, and c4 elsewhere.
  Score on half B, with pct ranks recomputed inside B, against c4 on the same B. Macro gain = mean over the 12 labels (an unchanged label contributes 0).
- Main run: 29/29, 2000 splits. Sensitivity runs: 39/19 (1500 splits), a second seed (1000 splits), and an HW base for calibration (2000 splits).
  The nested direction agreement uses 40 half-subsamples of half A. The full-data agreement uses 400 half-splits of the 58.
- "P>0" is the fraction of splits with a positive held-out gain. It is not a p-value: every split comes from the same 58 studies.
  The Monte Carlo SE of a mean is about .00005. The seed rerun moved the means by at most .00009 (MEASURED).

## 1. Shrinkage curve (c4 base, ours R7, 29/29, 2000 splits) (MEASURED; last two columns MODEL)

| subset rule S | mean abs(S) | lambda .25 | lambda .5 | lambda .75 | lambda 1 | MODEL lambda* at 29 | MODEL lambda* at 58 (gain) |
|---|---|---|---|---|---|---|---|
| ALL (honest) | 12 | +0.00006 (P>0 .52) | **+0.00014 (.54)** | -0.00013 (.50) | -0.00067 (.45) | .31 | .48 (+0.00025) |
| GAIN005: in-sample gain >= .005 (honest) | 8.1 | +0.00006 (.51) | +0.00013 (.54) | -0.00015 (.49) | -0.00068 (.45) | .30 | .47 (+0.00023) |
| TOP3: biggest 3 train-half gains (honest) | 3 | +0.00008 (.50) | +0.00009 (.55) | -0.00009 (.51) | -0.00042 (.48) | .32 | .49 (+0.00018) |
| TOP2 (honest) | 2 | +0.00006 (.45) | +0.00006 (.54) | -0.00006 (.51) | -0.00031 (.48) | .32 | .48 (+0.00013) |
| TOP1 (honest) | 1 | +0.00003 (.32) | +0.00002 (.43) | -0.00008 (.45) | -0.00023 (.45) | .28 | .44 (+0.00007) |
| NEST_DIR60: agreement in half A >= .60 (honest) | 1.8 | -0.00004 (.35) | -0.00015 (.38) | -0.00037 (.37) | -0.00070 (.34) | .07 | .13 (0) |
| NEST_DIR75 (honest) | 0.5 | -0.00007 (.12) | -0.00016 (.14) | -0.00030 (.14) | -0.00049 (.14) | 0 | 0 |
| NEST_DIR90 (honest) | 0.06 | -0.00003 | -0.00005 | -0.00009 | -0.00013 | 0 | 0 |
| FULL_DIR75 = {MM, Fracture} (LEAKY subset) | 2 | +0.00036 (.54) | +0.00087 (.77) | +0.00134 (.83) | +0.00165 (.85) | 1 | 1 |
| FIX_FR = {Fracture} (LEAKY label, honest w) | 1 | +0.00029 (.43) | +0.00068 (.67) | +0.00108 (.74) | **+0.00138 (.78; P<0 .05)** | 1 | 1 |
| FIX_FRPF = {Fracture, PF OA} (LEAKY label) | 2 | +0.00035 (.55) | +0.00073 (.72) | +0.00110 (.77) | +0.00140 (.79; P<0 .17) | 1 | 1 |
| FIX_PF = {PF OA} (LEAKY label) | 1 | +0.00006 (.30) | +0.00005 (.38) | +0.00002 (.39) | +0.00001 (.43; P<0 .35) | .5 | .67 (+0.00007) |

MODEL used for the last two columns: the train-half pick deviation is d = delta* + noise, with noise variance proportional to 1/n_train.
The held-out gain is then G(lambda) = a lambda - b lambda^2. a and b are fitted on the 9-point lambda curve, and the noise part of b is rescaled to n_train 58.
Check (MODEL vs MEASURED): the 29/29 fit predicts ALL at n_train 39 should peak at lambda .38 with +0.00020, and fall to -0.00033 at lambda 1. The 39/19 run measured a peak at lambda .375 to .5 of +0.00025 to +0.00027, and -0.00058 at lambda 1. The model is roughly right and slightly generous at large lambda. Fitting the 39/19 curve instead gives lambda* .45 at n_train 58, with +0.00032.
Curves that are nearly linear in lambda (Fracture, whose picks sit at the 1.0 boundary) break the model and are left unscaled.

### Best lambda / subset

- Best honest rule: **ALL at lambda .5**: +0.00014 (P>0 .54) at 29/29 and +0.00025 (P>0 .53) at 39/19 (MEASURED). MODEL at n_train 58: about +0.0003.
  Choosing the best of about 50 (rule, lambda) cells on the same 58 adds some optimism of its own. GUESS: about +0.0001.
- Best subset overall: {Fracture}, lambda 1 (+0.00138, P>0 .78). It is honest in w but leaky in which label was chosen. The justification for it is in section 3.

### Bagged picks (an attempt to reduce pick noise) (MEASURED; 29/29, 2000 splits, 30 bootstrap resamples of half A)

| rule | lambda .25 | lambda .5 | lambda .75 | lambda 1 | MODEL lambda* at 58 (gain) |
|---|---|---|---|---|---|
| ALL, plain pick (reference) | +0.00006 (.52) | +0.00014 (.54) | -0.00013 (.50) | -0.00067 (.45) | .48 (+0.00025) |
| BAGCURVE_ALL: argmax of the mean bootstrap AUC curve | +0.00015 (.54) | **+0.00027 (.56)** | -0.00005 (.51) | -0.00079 (.44) | .51 (+0.00042) |
| BAGMEAN_ALL: mean of the bootstrap picks | -0.00001 (.49) | +0.00002 (.52) | +0.00000 (.51) | -0.00007 (.51) | (flat, about 0) |

Bagging the curve doubles the small honest gain, but it is still about +0.0003 to +0.0004 (MODEL at 58), and Fracture supplies more than all of it. At lambda .5 Fracture contributes +0.00096 and the other 11 labels net -0.00069.
Its full-data table at lambda .5 changes 11 labels: MCL .6, MM .75, LM .35, Medial OA .65, Lateral OA .1, PF .25, Effusion .75, Synovitis .65, Baker's .75, Contusion .375, Fracture .825. In-sample gain +0.0022.
Medial OA flips direction compared with the plain pick (.65 vs .35), which shows how jagged these curves are. Not recommended.

### Why Fracture leans Raptor (MEASURED, component AUCs on gold58 from `screen_c4b.json`)

Fracture (18 positives): r384 .958, r224 .943, ours = R7 OOF .860, S6fall .901. PF OA (21 positives): r384 .860, r224 .851, R7 .888, S6fall .902.
The Fracture gap between the Raptor legs and R7 is about .1 AUC, which is why the pick goes to w = 1.
**This is the main R6 risk.** If R6's Fracture is much better than R7's on these studies, the Fracture push is worth less. The R6 serve file would settle it.

## 2. Per-label decomposition: each label moved alone to its train-half pick (c4 base, 29/29) (MEASURED)

Macro contribution of the held-out gain, which is the per-label AUC gain divided by 12. "full pick" and "full agree" come from all 58. "nested agree" is the mean agreement inside a training half.

| label | c4 | full pick (in-sample gain, per label) | full agree | nested agree | lambda .5 | lambda 1 | P>0 / P<0 at lambda 1 |
|---|---|---|---|---|---|---|---|
| **Fracture** | .65 | 1.00 (+.0306) | .88 | .57 | +0.00068 | **+0.00138** | .78 / .05 |
| Lateral OA | .20 | 0.05 (+.0039) | .71 | .20 | +0.00013 | +0.00030 | .34 / .03 |
| Medial Meniscus | .50 | 0.85 (+.0084) | .78 | .46 | +0.00019 | +0.00026 | .44 / .13 |
| Baker's | .50 | 1.00 (+.0073) | .63 | .38 | +0.00009 | +0.00006 | .20 / .12 |
| PF OA | .50 | 0.10 (+.0103) | .70 | .47 | +0.00005 | +0.00002 | .43 / .35 |
| Effusion | .50 | 1.00 (+.0118) | .67 | .44 | +0.00010 | -0.00005 | .28 / .26 |
| ACL | .50 | 0.35 (+.0006) | .40 | .23 | -0.00004 | -0.00006 | .04 / .13 |
| Medial OA | .50 | 0.35 (+.0023) | .23 | .24 | -0.00004 | -0.00012 | .04 / .19 |
| MCL | .50 | 0.70 (+.0045) | .31 | .22 | -0.00007 | -0.00015 | .08 / .20 |
| Contusion | .50 | 0.20 (+.0061) | .54 | .40 | -0.00022 | -0.00051 | .17 / .52 |
| Lateral Meniscus | .40 | 0.35 (+.0019) | .49 | .42 | -0.00037 | -0.00085 | .15 / .56 |
| Synovitis | .80 | 0.70 (+.0012) | .68 | .56 | -0.00037 | -0.00095 | .23 / .61 |

Reading: the ALL rule nets out to about 0 because Fracture's +0.0014 and small gains on Lateral OA, MM and Baker's are cancelled by losses on Synovitis, Lateral Meniscus and Contusion.
The full-data pick and its agreement do not tell the two groups apart (MEASURED). Synovitis (.68), Effusion (.67) and PF OA (.70) all have high full agreement and held-out results of zero or worse.
Lateral OA down is small but rarely hurts (P<0 .03 at 29/29, .006 at 39/19). It is a possible low-risk add-on of about +0.0003. GUESS: low priority, because the curve is jagged (.1 is worse than c4 in sample and .05 is better).

## 3. Is Fracture real or the winner of 12 labels? (MEASURED + MODEL)

- Fracture's weight chosen on half A and scored on half B: +0.00138 macro (per-label AUC +.017), P<0 only .05 (29/29). At 39/19: +0.00187, P<0 .05 (MEASURED).
  The 11 other labels have honest-in-w values between -0.00095 and +0.00030, with a spread of about .0004. Fracture sits about 3.5 to 4 spreads above their mean.
  MODEL: the maximum of 12 pure-noise labels is expected about 1.6 spreads above the mean. That spread also contains real differences between labels, so the yardstick is conservative.
- Winner's curse on Fracture's own value is small. When half A ranks Fracture first, its held-out is +0.00122. Without that condition it is +0.00138 (MEASURED).
- The honest TOP1 procedure becomes positive as n_train grows. It picks Fracture in 42.5% of splits at 29 and 64.6% at 39 (MEASURED).
  Its held-out moves from -0.00023 to +0.00018 (MEASURED). On all 58 the procedure picks Fracture outright.
  MODEL: the expected value at 58 is close to the honest-in-w Fracture value of about +0.0012 to +0.0014, less a small leftover winner's curse.
- Train-half Fracture picks: 1.0 in 31%, .95 in 25%, .9 in 16%, and .65 (= c4) in only 6% (29/29). At 39/19: 1.0 in 43%, .95 in 32%, .9 in 15% (MEASURED).

## 4. A19 slate entries: fixed-table held-out (LEAKY) vs honest analogue (c4 base, ours R7) (MEASURED)

"Fixed on B" scores the exact slate table on the same B halves. It is leaky because the table was chosen on all 58, B included.
The honest analogue picks the same label(s) with w chosen on half A (honest in w, leaky in label).

| entry | table vs c4 | fixed on B, 29/29 (LEAKY) | fixed on B, 39/19 (LEAKY) | honest analogue at lambda 1, 29/29 | 39/19 | selection optimism (fixed - analogue) |
|---|---|---|---|---|---|---|
| c2max | Fr 1.0, PF .1 | +0.00331 (P>0 .96) | +0.00346 (.86) | FIX_FRPF +0.00140 (.79) | +0.00214 (.76) | ~+0.0019 |
| c2 | Fr .9, PF .1 | +0.00284 (.95) | +0.00297 (.84) | FIX_FRPF at lambda .875: +0.00127 (.78) | | ~+0.0016 |
| fr100 | Fr 1.0 | +0.00233 (.92) | +0.00236 (.72) | FIX_FR **+0.00138 (.78)** | +0.00187 (.68) | ~+0.0010 |
| fr90 | Fr .9 | +0.00187 (.90) | +0.00187 (.69) | FIX_FR at lambda .875: +0.00124 (.76) | +0.00165 | ~+0.0006 |
| pf10 | PF .1 | +0.00098 (.745) | +0.00110 (.61) | FIX_PF **+0.00001 (.43; P<0 .35)** | +0.00027 (.42) | ~+0.0010 (all of it) |
| (best honest table) | ALL at lambda .5 | | | +0.00014 (.54) | +0.00025 (.53) | |

Paired on the same splits (MEASURED; the comparison is biased toward the leaky slate): ALL at lambda .5 minus fr100 = -0.00219 (P>0 .20), and ALL at lambda .5 minus c2max = -0.00317 (P>0 .12).
No honest rule beats any slate entry, but the slate side of each comparison is inflated by selection.
The fair comparison is the honest-analogue column. There, fr100 and c2max tie (+0.00138 vs +0.00140), because the PF OA half of c2max contributes +0.00001.

## 5. Calibration: HW base (where the LB says the HW -> c4 table move was POSITIVE) (MEASURED)

The same CV run from the HW table (a25, S6F .2) instead of c4:

- Honest ALL: lambda .25 +0.00046 (P>0 .61), lambda .5 +0.00086 (.66), lambda .75 +0.00099 (.66), lambda 1 +0.00076 (.62). Honest TOP3 at lambda .75: +0.00069 (.66).
- The fixed c4 table on B (LEAKY): +0.00177 (P>0 .89).

So the honest CV did detect value in re-picking from HW, about +0.0009, and the LB agreed with the one table move made there.
From the c4 base the same honest rule finds about +0.00014. That is roughly 6x less, so c4 has already captured almost all the honestly recoverable table gain except Fracture.
GUESS: this is one calibration point, not a validation. The global knobs (S6fall .3 and .15) were gold58-positive at P>0 .7 to .8 and then LB-negative.
Gains under about .001 that exist only in leaky gold58 reads should be treated as noise. The Fracture signal is held-out, rises with lambda, and has P<0 .05, so it is the most robust read in this set.

## Full-data table

Applying each rule to all 58 (pick on 58; agreement from 400 half-splits of 58) (MEASURED; in-sample gains are optimistic):

| rule at lambda | changes vs c4 | in-sample gain |
|---|---|---|
| ALL at lambda .5 (best honest rule) | ACL .425, MCL .6, MM .675, LM .375, Medial OA .425, Lateral OA .125, PF OA .3, Effusion .75, Synovitis .75, Baker's .75, Contusion .35, Fracture .825 | +0.0024 |
| ALL at lambda .25 | ACL .4625, MCL .55, MM .5875, LM .3875, MOA .4625, LOA .1625, PF .4, Eff .625, Syn .775, Bak .625, Con .425, Fr .7375 | +0.00085 |
| TOP1 at lambda 1 (= procedure at n 58) | Fracture 1.0 | +0.00255 |
| DIR75 at lambda 1 (full agreement >= .75) | Medial Meniscus .85, Fracture 1.0 | +0.00325 |

**The table I would actually submit if trained on all 58: c4 with Fracture 1.0 (= fr100, already built as `v224r6c4fr100j`).** (MODEL/GUESS)
- The ALL at lambda .5 table is the formal winner among honest rules, but it is worth only about +0.0003 at n 58 (MODEL), and it changes 12 labels.
  Its value comes from Fracture .825; the other 11 changes net out to about 0 or below (section 2).
- Fracture at lambda 1 beats lambda .875 and .75 at both split ratios (MEASURED), and the train-half picks pile up at .9 to 1.0. That supports 1.0 over .9.
- Optional low-risk add-on, not on the slate and not recommended over the slate as is: Lateral OA .05 to .1. Honest-in-w +0.0003, P<0 .03 (MEASURED), but jagged in sample (GUESS).

## Recommendation for tomorrow's A19 slate (MODEL/GUESS)

- Expected LB gain vs c4, honest-ish (Fracture label taken as given): fr100 about +0.0012 to +0.0014. c2max about the same, with extra variance from PF OA.
  fr90 about +0.0011 to +0.0012. c2 about the same as fr90. pf10 about 0 (+/- 0.0003).
- If the slots are scarce, the order is **fr100, then c2max (tests whether PF OA adds anything on the LB), then fr90**. Drop pf10 if a slot is needed for something else.
  The CV says pf10 is a coin flip, so its LB read mostly measures noise.
  One argument for keeping pf10 (GUESS): it is the only entry that isolates PF OA, so c2max minus fr100 and pf10 would cross-check each other.
- None of this beats c4 by more than about LB noise for a 12-label shrink, so do not spend a slot on the ALL at lambda .5 table.

## Caveats

- **Ours = R7 OOF, not R6 (fallback).** The R6 serve file never arrived. Labels where R6 and R7 differ could shift their picks. Fracture's weight depends on how good the ours leg is on Fracture.
- c4 itself was partly chosen on these 58 (PF .5, Baker's .5, LM .4, Fracture .65). B halves therefore favour c4 a little, which makes every deviation look slightly worse (MODEL). This is pessimistic for the shrink and for Fracture alike.
- Training on 29 (or 39) studies makes picks noisier than training on 58. The MODEL columns correct for this under a simple noise model, and the 39/19 run agrees with it.
- The P>0 values are split fractions on one fixed set of 58 studies. They do not reflect how much the 58 themselves vary relative to the LB population.
  The real uncertainty is larger: the LB resubmit noise on this comp is unknown here, and GUESS puts it at a similar order to these gains.
- Grid ties go to the value nearest c4, which leaves many labels unchanged in many splits (many P=0 cells). That is conservative.

## Files (all in session_2026-10-18/next/)

- `cv_shrink.py`: the CV. Usage: `python cv_shrink.py g58 <ours.csv> <reps> <out.json> [train_frac=.5] [K_inner=40] [seed] [base=c4|hw]`
- `cv_shrink_report.py`: markdown summary plus MODEL extrapolation. `python cv_shrink_report.py <json>...`
- Outputs: `cv_shrink_R7_half.json/.log` (main), `cv_shrink_R7_2to1.json/.log`, `cv_shrink_R7_hwbase.json/.log`, `cv_shrink_R7_half_bag.json/.log` (bagged picks; arg 9 = 30). Its non-bag rows are identical to the main run, which checks reproducibility.
- Rerun with R6 when it exists (about 25 min of laptop CPU in total):
  `python cv_shrink.py g58 g58/R6_gold58_serve.csv 2000 cv_shrink_R6_half.json 0.5 40 20261018 c4`,
  then the same with `0.6667` (file name `_2to1`, 1500 reps) and with base `hw`.
