# census.md: Track 1 coupling census, all 66 label pairs (10-20 ~11:45-14:00 IST, Claude). RESEARCH + MEASUREMENT ONLY.

Nothing was built, pushed, committed or submitted. No dataset, no download, no GPU job. No scheduled task, PREREG, HANDOFF,
PROGRESS, session_2026-10-19/a20 or session_2026-10-20/a21 file was touched (the c4, a20 eoa20 and a21 kernels were only read, to
exec their cell-18 code on gold58 in a temp dir, as `a21/gate_a21_offline.py` does). Everything was written under `push/`.
- **OAI governance.** No OAI analysis was run tonight. Every OAI number below is CLAIMED from aggregate logs already on the laptop
  (`session_2026-10-19/ex/box_coupling.log` JSON block, `box_followup.log`; `session_2026-10-20/more/box_inspect.log` for the
  absence of MCL / Contusion / Fracture grades). The box was touched once: a read-only ssh check (`echo ok`, `nvidia-smi`,
  `ls ~/oai_build` file names). No OAI row, id or value was read or moved.
- **Tags.** MEASURED = computed tonight, file named. CLAIMED = an older note or log says so, file named. MODEL = arithmetic on
  stated inputs. GUESS = judgement.
- **Units.** "d" = single-label AUC delta. "Macro" = 12-label macro AUC = d / 12.
- **Bases.** CUR = the scored best-id eoa20 final output on gold58 (gold58 macro .94360, parity with the a21 gate). Its 9
  non-OA columns are byte-equal to c4 57343435 (.94113, parity). Model coupling below is c4; for OA pairs the eoa20 value is
  shown in brackets when it differs.

## 0. Answer

1. **Census result (MEASURED, sec 2).** Outside the OA trio, the only coupling excess that is larger than chance is **MCL's**.
   - 5 of the 66 gold58 gaps exceed 2 bootstrap sd, against 3.0 expected by chance: MCL-Bak (z 4.1), MCL-MOA (2.9), MCL-MM
     (2.6), MCL-Eff (2.1) and Cont-Frac (2.1). All five are excess (model couples more than experts); no deficit reaches 2 sd.
   - The MCL head couples positively with degenerative findings (c4 rho +.25 with MM) where experts couple negatively (gold74
     tet -.31). This fits the rubric (MCL = high-grade ACUTE tear only).
   - Largest consistent excesses (min of gold58 and gold74 gap): MCL-MM +.56, Bak-Frac +.40, MCL-Bak +.39, Cont-Frac +.34,
     MCL-Eff +.33, MCL-Cont +.30, LM-Syn +.27, MCL-MOA +.24. Largest deficits: LOA-Eff -.30, MOA-Bak -.23, Syn-Cont -.22.
2. **No non-OA candidate is robust. None is worth a public probe on the evidence.** gold58 cannot tell a good decorrelation
   from a bad one (MEASURED, sec 3):
   - **Mechanism test on held-out rows.** Across the 120 non-OA-internal directions, the coupling excess measured on one half of
     gold58 does not predict the decorrelation gain on the other half: mean Spearman +.001 (P(>0) .53 over 400 splits). The
     theory gap (rho - q) gives -.009 (P .48).
     - In-sample the same correlations are +.32 and +.63. That difference is overfit.
     - On the report-OOF ruler (n 4,333) the excess does predict the delta (+.30). So the logic is sound; the 58-study
       instrument is too weak to use it.
   - **Selection CV.** Picking the best mechanism-consistent direction on 29 studies and scoring it on the other 29 gives
     d -.0019 (P(>0) .22). Picking on 46 and scoring on 12 gives -.0034 (P .04).
   - **Noise level.** The best observed direction (+.0118) is below the median best of 120 pure-noise directions (+.0194).
   - **Few studies carry each effect.** Every shortlisted effect rests on 2-3 positives (the top 2 carry 65 % or more of the delta).
   - **Independent ruler.** MCL / Contusion / Fracture have no OAI grade. Where OAI labels exist it kills the 9-label survivors:
     - LM <- MM and MM <- LOA lose on OAI.
     - LOA <- MM is about 0 on top of E-OA .2 (raw +.00008 / +.00003 / -.00004 macro; skeptic R1oa3 -.0003 to -.0005).
3. **Ranked candidate list** (sec 4 and 6):
   - **E** = expected hidden macro. It is the same for public and private; only the read noise differs (390 vs 910 studies).
   - **P** = P(true effect > 0) under a 50/50 mixture of an optimistic "mechanism" prior and the selection-CV prior (MODEL).
   - **Probe** = whether the candidate is worth ONE public LB probe.

   | rank | candidate (on the A21 winner) | gold58 d (P boot) | why it is not robust | E macro (mixture; mech / cv prior) | P(>0) | probe? |
   |---|---|---|---|---|---|---|
   | 1 | MCL <- MM, a .1 | +.0091 (.79) | only above-chance excess; 3 of 9 positives carry it; split-half .33; no OAI; report OOF loses (predicted, non-discriminating) | +2.8e-5 (+1.2e-4 / -1.1e-4) | .56 | lottery only |
   | 2 | Frac <- MOA, a .1 | +.0118 (.89) | excess +.16 is not above chance; acute <- DEG family failed its specificity null (CLAIMED expert2); extra16 -.025 (4 pos) | +1.9e-5 (+1.2e-4 / -1.2e-4) | .54 | no |
   | 3 | MCL <- mean of other 11 (general factor), a .1 | +.0023 (.40) | coherent with the MCL row, but P below .5 at a .1 | +2.8e-5 | .57 | no |
   | 4 | LOA <- Eff, borrow a -.1 | +.0116 (.90) | the deficit is a gold58 artefact (R7 train rho .33, report .39, expert .39, c4 on gold58 .07); base-dependent +.0097 to +.0155; 83 % from 2 positives; report OOF loses | +2.0e-5 | .55 | no |
   | 5 | MCL <- Syn / Cont / Frac / MOA, a .1 | +.0045 to +.0091 (.59-.74) | same MCL factor; not separable from rank 1 | about 0 | .49-.50 | no |
   | - | MM <- LOA, LM <- MM, LOA <- MM | +.0048 / +.0075 / +.0039 | KILLED by OAI (independent; CLAIMED 10-19 logs) | <= 0 | <= .4 (GUESS) | no |
   | - | every Synovitis, Contusion and ACL direction | <= +.0060 | sec 5 | about 0 or < 0 | about .5 or less | no |

4. **If Eesh still wants one spare 10-21 slot used** (NOT recommended; needs his go, a PREREG amendment, a build and a gate,
   none of which was done):
   - **The least-bad ticket is MCL <- MM at a .1**, stacked on the A21 best-id (one line in cell 18, MCL column only).
   - E[hidden macro] +3e-5 (MODEL, range -1e-4 to +1.2e-4 by prior), P(>0) about .56.
   - Paired public read noise sd 2.5e-4 macro (MEASURED, R7 OOF + sol56 subsample of 390). That is about 2x the optimistic
     effect, so the read cannot confirm it. P(it becomes the API best-id) about .54, mostly noise.
   - E[private gain of probe + R1] +6e-5 (MODEL). Nearly all of that is R1's option value under an assumed prior spread, which
     every zero-mean change shows (+4.5e-5 to +9.5e-5 across the list). So it does not single this candidate out.
   - The 2-label combo MCL <- MM + Frac <- MOA: E +4.6e-5, P .57, public noise sd 4.4e-4 (MODEL). Same verdict.

## 1. Method (MEASURED unless tagged; scripts in `push/`)

**Data.**
- gold58 expert labels (58 studies).
- gold74 = gold58 + 16 manual_verified studies. The 16 are our own reads under the gold convention, not expert reads
  (CLAIMED trick/label_regime.md). gold74 has NaNs, so n is 65-74 per pair.
- c4 member serves exist only on gold58, so all c4 transforms are scored on gold58.
- R7 OOF exists on all 74, so R7 is the second model on gold74.
- Report regime: report_labels_v2 (YES / NO, UNK dropped) and v4_blend.
- Report-OOF models:
  - R7 OOF on train minus gold74 (n 4,333).
  - Rank mean of R7 + s1044 + s1045 (n 3,467).

**Coupling.**
- Expert: phi and tetrachoric on gold74 (bootstrap sd from 300 draws), and tetrachoric on gold58. The fast grid tetrachoric
  matches the old brentq version to 1e-4 on 6 test tables.
- Model: Spearman of the final c4 / eoa20 output on gold58. R7 Spearman on gold74 and train. s1044 on train.
- Report: v2 / v4 tetrachoric on train.
- Gap g58 = model rho - gold58 tet on the same 58 studies (paired bootstrap of 1,000). Gap g74 = model rho - gold74 tet.
- "Consistent" = min of the two when both are > 0, max when both are < 0, else mixed.

**Directional transform** (132 directions, on CUR): out_t = pct(pct(f_t) - a pct(f_o)), with a in {-.2, -.1, -.05, .05, .1,
.2}. Positive a decorrelates; negative a borrows.
- **Theory.** z = normal scores, rho = corr(z_t, z_o), q = corr(z_o, y_t) / corr(z_t, y_t). Decorrelation helps at small a iff
  rho > q. a* = (c_t rho - c_o) / (c_t - c_o rho). The theory gap rho - q was bootstrapped.
- **Rulers.** Paired bootstrap of 2,000 and split-half sign agreement (1,000 random 29 / 29 splits) on gold58 c4; R7 on gold74
  and on gold58.
- **Report-OOF control** (mechanism: decorrelating a report-inherited coupling LOSES there): R7 vs v2 / v4 and the 3-model
  mean vs v2 / v4.
- **OAI.** 3 OAI-free models from the 10-19 box run (CLAIMED): raw, R2 (OR-matched to gold74) and R1 (cell-matched).

**Honesty checks.**
- Count of gaps beyond 2 sd vs chance.
- Pure-noise maximum over 120 directions.
- Empirical Bayes across directions.
- Selection CV (29 / 29 and 46 / 12, 1,000 splits each).
- Held-out mechanism test (400 splits).
- Leave-one-out influence and per-positive contributions.
- Base dependence (c4, eoa20 and the in-flight eoa30 / pw20 / pw30 bases).
- Per-member replication (R6, S6fall, Raptor, R7).

**Read noise** (`census_d.py` part 4, the `r5/noise_corr.py` method): sd of the macro delta over 400 subsamples of 390 / 910
non-gold train studies, R7 OOF vs sol56 labels.

## 2. The census (66 pairs, sorted by |consistent excess|; MEASURED `census.log`, `census_pairs.csv`)

**How to read it.**
- **Bold z** marks a gold58 gap beyond 2 bootstrap sd.
- "report - expert" = train report tet - gold74 expert tet. The two come from different populations (see the caveat after the
  table).

| # | pair | gold74 n (pos a, b) | gold74 tet +- sd | gold58 tet | c4 rho g58 (eoa20) | R7 rho g74 / train | report v2 tet | gap g58 (sd, P>0) | gap g74 | consistent | report - expert |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | MCL-MM | 73 (11,35) | -0.31 +- 0.20 | -0.34 | +0.25 | +0.12 / +0.33 | +0.31 | +0.59 (0.23, 1.00) **z 2.6** | +0.56 | +0.56 | +0.62 |
| 2 | Bak-Frac | 67 (18,21) | -0.32 +- 0.20 | -0.27 | +0.13 | +0.06 / +0.15 | +0.04 | +0.40 (0.20, 0.98) | +0.45 | +0.40 | +0.37 |
| 3 | MCL-Bak | 67 (10,18) | -0.30 +- 0.21 | -0.47 | +0.09 | -0.03 / +0.04 | +0.16 | +0.56 (0.14, 1.00) **z 4.1** | +0.39 | +0.39 | +0.46 |
| 4 | Cont-Frac | 72 (26,22) | +0.40 +- 0.17 | +0.49 | +0.84 | +0.75 / +0.65 | +0.82 | +0.34 (0.17, 0.99) **z 2.1** | +0.44 | +0.34 | +0.43 |
| 5 | MCL-Eff | 74 (11,48) | -0.03 +- 0.22 | -0.09 | +0.30 | +0.23 / +0.32 | +0.32 | +0.38 (0.19, 0.97) **z 2.1** | +0.33 | +0.33 | +0.36 |
| 6 | LOA-Eff | 71 (14,46) | +0.39 +- 0.18 | +0.37 | +0.08 (+0.07) | +0.15 / +0.33 | +0.39 | -0.30 (0.20, 0.06) | -0.32 | -0.30 | -0.00 |
| 7 | MCL-Cont | 73 (11,26) | +0.30 +- 0.21 | +0.21 | +0.60 | +0.47 / +0.50 | +0.43 | +0.39 (0.23, 0.95) | +0.30 | +0.30 | +0.12 |
| 8 | LM-Syn | 73 (30,36) | +0.27 +- 0.16 | +0.25 | +0.54 | +0.39 / +0.40 | +0.45 | +0.29 (0.16, 0.97) | +0.27 | +0.27 | +0.18 |
| 9 | MCL-MOA | 72 (11,22) | -0.39 +- 0.17 | -0.54 | -0.16 (-0.15) | -0.23 / +0.17 | +0.25 | +0.39 (0.14, 1.00) **z 2.9** | +0.24 | +0.24 | +0.64 |
| 10 | LOA-PFOA | 71 (14,29) | +0.39 +- 0.19 | +0.45 | +0.81 (+0.68) | +0.78 / +0.73 | +0.77 | +0.23 (0.18, 0.91) | +0.29 | +0.23 | +0.38 |
| 11 | MOA-Bak | 66 (18,17) | +0.71 +- 0.12 | +0.70 | +0.48 (+0.47) | +0.37 / +0.27 | +0.32 | -0.23 (0.15, 0.07) | -0.24 | -0.23 | -0.39 |
| 12 | Syn-Cont | 72 (36,25) | +0.23 +- 0.18 | +0.13 | -0.09 | +0.08 / +0.30 | +0.44 | -0.22 (0.20, 0.16) | -0.31 | -0.22 | +0.21 |
| 13 | MM-LOA | 70 (33,14) | +0.05 +- 0.20 | +0.02 | +0.31 (+0.27) | +0.27 / +0.41 | +0.42 | +0.25 (0.20, 0.89) | +0.21 | +0.21 | +0.37 |
| 14 | LOA-Bak | 65 (11,16) | +0.52 +- 0.19 | +0.63 | +0.36 (+0.33) | +0.24 / +0.22 | +0.18 | -0.30 (0.15, 0.03) | -0.19 | -0.19 | -0.34 |
| 15 | MCL-Frac | 72 (11,22) | +0.12 +- 0.23 | +0.24 | +0.42 | +0.49 / +0.37 | +0.31 | +0.19 (0.21, 0.80) | +0.31 | +0.19 | +0.20 |
| 16 | LM-Bak | 67 (27,18) | +0.40 +- 0.18 | +0.46 | +0.22 | +0.05 / +0.18 | +0.23 | -0.24 (0.18, 0.09) | -0.19 | -0.19 | -0.17 |
| 17 | ACL-PFOA | 73 (29,31) | -0.46 +- 0.15 | -0.41 | -0.60 (-0.63) | -0.50 / -0.20 | +0.10 | -0.22 (0.17, 0.10) | -0.18 | -0.18 | +0.55 |
| 18 | ACL-LM | 74 (29,30) | -0.06 +- 0.18 | -0.06 | +0.12 | +0.02 / +0.23 | +0.42 | +0.17 (0.17, 0.85) | +0.18 | +0.17 | +0.49 |
| 19 | Eff-Frac | 72 (47,22) | +0.27 +- 0.19 | +0.26 | +0.44 | +0.40 / +0.36 | +0.41 | +0.18 (0.18, 0.83) | +0.17 | +0.17 | +0.14 |
| 20 | MM-Eff | 73 (35,48) | +0.36 +- 0.15 | +0.35 | +0.19 | +0.15 / +0.33 | +0.35 | -0.17 (0.14, 0.13) | -0.17 | -0.17 | -0.00 |
| 21 | ACL-LOA | 71 (29,14) | +0.04 +- 0.21 | -0.08 | -0.30 (-0.24) | -0.30 / -0.02 | +0.32 | -0.17 (0.21, 0.23) | -0.28 | -0.17 | +0.28 |
| 22 | MCL-Syn | 73 (11,36) | -0.06 +- 0.22 | -0.20 | +0.11 | +0.26 / +0.35 | +0.15 | +0.30 (0.24, 0.89) | +0.16 | +0.16 | +0.21 |
| 23 | MOA-LOA | 71 (21,14) | +0.59 +- 0.16 | +0.51 | +0.85 (+0.75) | +0.77 / +0.83 | +0.83 | +0.24 (0.19, 0.90) | +0.16 | +0.16 | +0.24 |
| 24 | MOA-Frac | 71 (21,22) | -0.40 +- 0.18 | -0.39 | -0.22 | -0.17 / +0.18 | +0.17 | +0.16 (0.17, 0.82) | +0.17 | +0.16 | +0.57 |
| 25 | MM-LM | 73 (35,29) | +0.18 +- 0.17 | +0.18 | +0.34 | +0.27 / +0.43 | +0.45 | +0.15 (0.16, 0.81) | +0.15 | +0.15 | +0.27 |
| 26 | MCL-PFOA | 73 (11,31) | -0.38 +- 0.18 | -0.41 | -0.22 (-0.23) | -0.19 / +0.09 | +0.18 | +0.18 (0.22, 0.80) | +0.15 | +0.15 | +0.56 |
| 27 | LM-MOA | 72 (29,22) | +0.56 +- 0.15 | +0.61 | +0.43 (+0.41) | +0.34 / +0.43 | +0.43 | -0.19 (0.15, 0.11) | -0.15 | -0.15 | -0.13 |
| 28 | MCL-LM | 74 (11,30) | +0.08 +- 0.21 | +0.09 | -0.05 | -0.05 / +0.22 | +0.23 | -0.14 (0.22, 0.27) | -0.14 | -0.14 | +0.15 |
| 29 | ACL-MCL | 74 (29,11) | +0.37 +- 0.21 | +0.22 | +0.50 | +0.54 / +0.38 | +0.45 | +0.28 (0.21, 0.91) | +0.13 | +0.13 | +0.07 |
| 30 | MOA-Syn | 71 (22,35) | +0.30 +- 0.17 | +0.38 | +0.52 (+0.50) | +0.30 / +0.44 | +0.70 | +0.12 (0.18, 0.72) | +0.20 | +0.12 | +0.40 |
| 31 | Eff-Syn | 73 (48,36) | +0.62 +- 0.13 | +0.59 | +0.74 | +0.70 / +0.83 | +0.82 | +0.16 (0.13, 0.87) | +0.12 | +0.12 | +0.20 |
| 32 | LM-Cont | 73 (30,26) | -0.33 +- 0.18 | -0.30 | -0.19 | -0.14 / +0.15 | +0.31 | +0.11 (0.17, 0.75) | +0.14 | +0.11 | +0.64 |
| 33 | LM-Eff | 74 (30,48) | +0.23 +- 0.17 | +0.23 | +0.33 | +0.26 / +0.30 | +0.34 | +0.10 (0.16, 0.72) | +0.10 | +0.10 | +0.11 |
| 34 | Eff-Cont | 73 (47,26) | +0.21 +- 0.18 | +0.18 | +0.31 | +0.20 / +0.30 | +0.45 | +0.13 (0.17, 0.77) | +0.10 | +0.10 | +0.24 |
| 35 | PFOA-Frac | 71 (29,22) | -0.29 +- 0.17 | -0.31 | -0.21 (-0.20) | -0.11 / +0.15 | +0.05 | +0.11 (0.17, 0.75) | +0.09 | +0.09 | +0.35 |
| 36 | MOA-PFOA | 72 (22,30) | +0.61 +- 0.14 | +0.56 | +0.82 (+0.71) | +0.80 / +0.73 | +0.80 | +0.14 (0.15, 0.86) | +0.09 | +0.09 | +0.18 |
| 37 | LM-PFOA | 73 (29,31) | +0.15 +- 0.17 | +0.19 | +0.32 (+0.27) | +0.33 / +0.30 | +0.31 | +0.08 (0.17, 0.67) | +0.13 | +0.08 | +0.16 |
| 38 | LOA-Syn | 70 (14,35) | +0.48 +- 0.18 | +0.56 | +0.45 (+0.41) | +0.27 / +0.39 | +0.61 | -0.16 (0.17, 0.17) | -0.07 | -0.07 | +0.13 |
| 39 | MM-Frac | 71 (34,22) | +0.05 +- 0.19 | +0.11 | -0.03 | +0.00 / +0.20 | +0.07 | -0.14 (0.18, 0.24) | -0.07 | -0.07 | +0.03 |
| 40 | MM-Bak | 66 (31,18) | +0.48 +- 0.16 | +0.50 | +0.41 | +0.28 / +0.22 | +0.31 | -0.09 (0.15, 0.27) | -0.07 | -0.07 | -0.17 |
| 41 | ACL-Eff | 74 (29,48) | +0.29 +- 0.17 | +0.38 | +0.23 | +0.10 / +0.10 | +0.43 | -0.16 (0.15, 0.15) | -0.07 | -0.07 | +0.13 |
| 42 | MCL-LOA | 71 (11,14) | +0.01 +- 0.25 | -0.13 | -0.19 (-0.18) | -0.21 / +0.14 | +0.15 | -0.06 (0.24, 0.43) | -0.19 | -0.06 | +0.14 |
| 43 | LOA-Frac | 70 (13,22) | -0.30 +- 0.20 | -0.23 | -0.19 (-0.17) | -0.18 / +0.15 | +0.12 | +0.06 (0.22, 0.59) | +0.13 | +0.06 | +0.42 |
| 44 | MM-Syn | 72 (34,36) | +0.25 +- 0.17 | +0.30 | +0.36 | +0.25 / +0.37 | +0.49 | +0.05 (0.17, 0.61) | +0.10 | +0.05 | +0.24 |
| 45 | MM-MOA | 71 (34,22) | +0.67 +- 0.13 | +0.63 | +0.54 (+0.58) | +0.53 / +0.58 | +0.66 | -0.05 (0.14, 0.31) | -0.09 | -0.05 | -0.01 |
| 46 | ACL-Bak | 67 (26,18) | -0.22 +- 0.19 | -0.13 | -0.09 | -0.31 / -0.06 | +0.14 | +0.05 (0.21, 0.61) | +0.13 | +0.05 | +0.36 |
| 47 | ACL-Syn | 73 (28,36) | -0.07 +- 0.18 | -0.13 | -0.03 | +0.07 / +0.13 | +0.33 | +0.10 (0.20, 0.68) | +0.04 | +0.04 | +0.40 |
| 48 | PFOA-Bak | 66 (26,17) | +0.37 +- 0.18 | +0.39 | +0.44 (+0.43) | +0.34 / +0.30 | +0.23 | +0.04 (0.19, 0.61) | +0.06 | +0.04 | -0.14 |
| 49 | ACL-MOA | 72 (29,22) | -0.38 +- 0.16 | -0.42 | -0.37 (-0.34) | -0.45 / -0.07 | +0.28 | +0.07 (0.16, 0.72) | +0.04 | +0.04 | +0.66 |
| 50 | Syn-Frac | 71 (36,22) | +0.08 +- 0.19 | +0.07 | +0.11 | +0.31 / +0.44 | +0.51 | +0.04 (0.21, 0.57) | +0.03 | +0.03 | +0.43 |
| 51 | PFOA-Eff | 73 (31,47) | +0.09 +- 0.19 | +0.15 | +0.06 | +0.15 / +0.26 | +0.34 | -0.09 (0.16, 0.28) | -0.03 | -0.03 | +0.25 |
| 52 | LM-LOA | 71 (29,14) | +0.61 +- 0.15 | +0.65 | +0.56 (+0.58) | +0.60 / +0.55 | +0.60 | -0.07 (0.17, 0.29) | -0.03 | -0.03 | -0.01 |
| 53 | MOA-Eff | 72 (22,47) | +0.27 +- 0.19 | +0.12 | +0.10 (+0.10) | +0.13 / +0.38 | +0.45 | -0.02 (0.19, 0.44) | -0.18 | -0.02 | +0.18 |
| 54 | LM-Frac | 72 (30,22) | -0.11 +- 0.18 | -0.01 | +0.01 | +0.01 / +0.28 | +0.17 | +0.02 (0.19, 0.58) | +0.12 | +0.02 | +0.28 |
| 55 | Syn-Bak | 67 (33,18) | +0.44 +- 0.18 | +0.34 | +0.32 | +0.16 / +0.14 | +0.48 | -0.02 (0.18, 0.41) | -0.12 | -0.02 | +0.04 |
| 56 | ACL-Cont | 73 (29,26) | +0.57 +- 0.13 | +0.56 | +0.54 | +0.47 / +0.41 | +0.48 | -0.02 (0.13, 0.40) | -0.03 | -0.02 | -0.09 |
| 57 | ACL-Frac | 72 (29,22) | +0.30 +- 0.17 | +0.41 | +0.29 | +0.31 / +0.22 | +0.37 | -0.12 (0.15, 0.19) | -0.02 | -0.02 | +0.07 |
| 58 | PFOA-Cont | 72 (30,26) | -0.44 +- 0.16 | -0.47 | -0.50 (-0.48) | -0.40 / -0.03 | +0.20 | -0.01 (0.18, 0.51) | -0.05 | -0.01 | +0.64 |
| 59 | MM-Cont | 72 (34,26) | -0.11 +- 0.19 | -0.06 | -0.13 | -0.25 / +0.15 | +0.26 | -0.07 (0.18, 0.38) | -0.01 | -0.01 | +0.37 |
| 60 | ACL-MM | 73 (29,35) | +0.10 +- 0.19 | +0.24 | +0.21 | -0.04 / +0.13 | +0.35 | -0.03 (0.14, 0.42) | +0.11 | mixed | +0.25 |
| 61 | PFOA-Syn | 72 (30,35) | +0.29 +- 0.18 | +0.45 | +0.47 (+0.45) | +0.33 / +0.33 | +0.64 | -0.00 (0.17, 0.47) | +0.16 | mixed | +0.35 |
| 62 | LOA-Cont | 70 (13,26) | -0.39 +- 0.21 | -0.45 | -0.45 (-0.41) | -0.36 / +0.02 | +0.29 | +0.03 (0.20, 0.61) | -0.03 | mixed | +0.68 |
| 63 | MM-PFOA | 72 (34,30) | +0.33 +- 0.15 | +0.07 | +0.23 (+0.18) | +0.34 / +0.32 | +0.38 | +0.11 (0.17, 0.72) | -0.15 | mixed | +0.05 |
| 64 | MOA-Cont | 71 (21,26) | -0.49 +- 0.16 | -0.57 | -0.51 | -0.49 / +0.01 | +0.33 | +0.06 (0.15, 0.70) | -0.02 | mixed | +0.83 |
| 65 | Bak-Cont | 67 (18,23) | -0.01 +- 0.20 | -0.13 | -0.02 | -0.08 / +0.07 | +0.12 | +0.11 (0.20, 0.72) | -0.01 | mixed | +0.14 |
| 66 | Eff-Bak | 67 (43,18) | +0.28 +- 0.20 | +0.10 | +0.21 | +0.27 / +0.20 | +0.49 | +0.11 (0.19, 0.69) | -0.07 | mixed | +0.21 |

**Reading the census** (MEASURED unless tagged):

- **Population caveat.** gold58 is not test-like. It is trauma-enriched:
  - Its studies sit at the 66th-72nd train percentile of R7 scores for ACL, MCL, LM, Eff, Syn, Contusion and Fracture.
  - Expert Fracture prevalence is .31 against a train report rate of .15; ACL is .41 against .30.
  - Model coupling moves with population. R7 MOA-Cont is +.01 on train and -.49 on gold74; R7 MCL-MOA is +.17 on train and
    -.23 on gold74.
  - So the "report - expert" column mixes population with regime. On the SAME gold74 studies, R7 already matches the expert
    for most Contusion / OA pairs.
  - The hidden test is presumably train-like (GUESS). Its expert couplings for the trauma labels are unknown.
- **Mean gap.** The mean gold58 gap is +.064 over the 66 pairs and +.057 outside the OA trio. Attenuation from model noise
  should make it negative for a calibrated model (MODEL). So the models carry some general positive co-firing that experts do
  not. MCL carries most of it: its mean consistent excess over its 11 partners is +.21, the next is Fracture at +.13
  (`census_genfactor.csv`).
- **OA trio on the eoa20 base.** The OA trio's excess is already partly removed: LOA-PFOA .81 -> .68, MOA-LOA .85 -> .75,
  MOA-PFOA .82 -> .71. None of the three is beyond 2 sd any more.
- **Deficits are inflated by attenuation.** A noisy model under-couples by construction (MODEL). The directional theory
  (rho vs q) corrects for this. It confirms a borrow signal only for LOA <- Eff (q .57 vs rho .08) and weakly for MOA <- Bak.

## 3. Why gold58 cannot pick a non-OA candidate (MEASURED `census.log`, `census_d.log`, `census_e.log`)

| check | result | reading |
|---|---|---|
| gaps beyond 2 sd | 5 of 66 (chance 3.0); 4 of the 5 are MCL pairs | MCL is the one coherent signal |
| mean gold58 d over 120 directions | a +.1: -.0008 (40 % > 0); a -.1: -.0018 (34 % > 0) | a random .1 perturbation costs AUC |
| best observed vs pure-noise best | +.0118 vs median +.0194 (90th pct +.0260) for 120 independent noise draws | the top is not remarkable |
| empirical Bayes | observed sd across directions .0046 < mean bootstrap se .0070, so tau = 0 | weak bound only: the directions share positives, so their noise is correlated |
| mechanism on held-out rows | Spearman(half-A excess, half-B d) +.001 (P .53); theory gap -.009 (P .48). In-sample +.32 / +.63 | the excess has no held-out predictive power at n 29 |
| same test, report OOF n 4,333 | Spearman(report excess, report d) +.30 | the logic works at large n |
| selection CV 29 / 29 | top1 any -.0020 (P .22); top1 mechanism-consistent -.0019 (.22); top3 -.0022 (.27); all mechanism-consistent -.0021 (.12) | selecting by gold58 loses on fresh expert rows |
| selection CV 46 / 12 | -.0031 / -.0034 / -.0027 / -.0021 (P .05 / .04 / .12 / .29) | same |
| report-OOF control | mean d over 120 directions, R7 vs v2: a +.1 -.0043, a -.1 +.0008. 3-model mean: -.0019 / -.0013 | every decorrelation loses on reports, so "loses on reports" does not discriminate (as expert2 found) |

- **Most frequent picks in the CV:** LOA <- Eff, Frac <- MOA, MCL <- MM, MOA <- Bak.
- **Their held-out half on 29 / 29:** +.0033, +.0028, +.0032 and -.0019. Picks like MCL <- Eff (-.0078) and LOA <- MM (-.0047)
  drag the procedure below zero.
- A specific pick's held-out half largely restates its full-sample value. So these numbers are not independent support.

## 4. Candidates (pairs outside the OA trio with |consistent excess| >= .15; a = +.1 decorrelate if excess > 0, -.1 borrow if < 0)

MEASURED `census_d.log` / `census_candidates.csv`; OAI CLAIMED (10-19 `box_coupling.log`, 3 OAI-free models). Top 12 by
gold58 d plus the OAI-tested ones; the full 46 rows are in the CSV.

| direction | excess | pos | theory rho-q (P) | d .05 / .1 / .2 | P boot | split-half same / both + | R7 g74 / g58 | report R7 v2 / 3-mean v2 | OAI d (3 models) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Frac <- MOA | +.16 | 18 | +.31 (.98) | +.0049 / +.0118 / +.0181 | .89 | .59 / .58 | +.0023 / +.0042 | -.0030 / -.0015 | no label | WEAK |
| LOA <- Eff (borrow) | -.30 | 11 | +.49 (1.00) | +.0039 / +.0116 / +.0242 | .90 | .48 / .46 | +.0094 / +.0135 | -.0010 / -.0025 | -.0016 / +.0005 / -.0024 (not a valid test: OAI models over-couple LOA-Eff) | WEAK; deficit is a gold58 artefact |
| MCL <- MM | +.56 | 9 | +.42 (1.00) | +.0045 / +.0091 / +.0159 | .79 | .33 / .27 | +.0101 / +.0068 | -.0077 / -.0040 | no label | WEAK (best census support) |
| MCL <- Syn | +.16 | 9 | +.33 (.99) | +.0045 / +.0091 / +.0068 | .74 | .32 / .26 | -.0043 / .0000 | -.0049 / -.0011 | no label | WEAK |
| LM <- MM | +.15 | 23 | +.08 (.74) | +.0019 / +.0075 / +.0006 | .83 | .53 / .51 | -.0008 / +.0037 | -.0043 / -.0017 | -.0006 / -.0008 / -.0025; R2 -.0030 / -.0031 / -.0051 | KILLED (OAI) |
| MCL <- Cont | +.30 | 9 | +.25 (.94) | .0000 / +.0068 / +.0091 | .66 | .21 / .12 | +.0043 / +.0068 | -.0021 / -.0007 | no label | WEAK |
| MOA <- Bak (borrow) | -.23 | 15 | +.24 (.99) | +.0031 / +.0062 / +.0016 | .60 | .07 / .00 | +.0041 / +.0008 | +.0006 / -.0005 | -.0017 / +.0026 / -.0010 (OAI MOA-Bak tet .01 vs gold74 .71: invalid) | DEAD (1 positive carries it; drop-one min -.0017) |
| MCL <- Frac | +.19 | 9 | +.04 (.58) | +.0045 / +.0057 / +.0023 | .63 | .16 / .12 | +.0029 / +.0023 | -.0048 / -.0003 | no label | WEAK |
| LM <- Bak (borrow) | -.19 | 23 | +.14 (.85) | +.0031 / +.0056 / +.0062 | .74 | .44 / .42 | +.0072 / +.0025 | +.0020 / -.0003 | -.0015 / -.0022 / +.0018 | WEAK |
| MM <- LOA | +.21 | 26 | -.02 (.41) | +.0024 / +.0048 / -.0036 | .80 | .37 / .30 | -.0019 / -.0006 | -.0023 / -.0012 | -.0008 / -.0020 / -.0020; R2 -.0010 / -.0022 / -.0022 | KILLED (OAI + theory) |
| MCL <- Eff | +.33 | 9 | +.14 (.76) | +.0045 / +.0045 / -.0057 | .57 | .16 / .07 | +.0022 / +.0023 | -.0056 / -.0013 | no label | DEAD |
| MCL <- MOA | +.24 | 9 | +.30 (.98) | -.0023 / +.0045 / +.0045 | .59 | .26 / .11 | -.0014 / -.0045 | -.0089 / -.0052 | no label | DEAD |
| LOA <- MM | +.21 | 11 | +.26 (.91) | +.0058 / +.0039 / +.0058 | .62 | .25 / .19 | +.0006 / +.0097 | -.0013 / +.0012 | +.0027 / +.0019 / +.0023; on top of E-OA .2: raw +.00008 / +.00003 / -.00004 macro, R1oa3 -.0003 / -.0004 / -.0005 | DEAD on top of eoa20 (CLAIMED box_followup.log) |

**Robustness of the top 4** (MEASURED `census_e.log`):

- **Per member** (same 58 labels, so this is not independent evidence). The effect appears in every member:

  | direction | R6 | S6fall | Raptor | R7 |
  |---|---|---|---|---|
  | Frac <- MOA | +.0097 | +.0083 | +.0090 | +.0042 |
  | LOA <- Eff | +.0097 | +.0203 | +.0077 | +.0135 |
  | MCL <- MM | +.0113 | +.0045 | +.0091 | +.0068 |

  It is a property of how these few positives are ranked, not of c4's blend.
- **Influence.** The share of the delta carried by the top-2 positives:

  | direction | top-2 share | positives contributing + / - / 0 | drop-one min |
  |---|---|---|---|
  | Frac <- MOA | 65 % | 8 / 2 / 8 | +.0064 |
  | LOA <- Eff | 83 % | 3 / 0 / 8 | |
  | MCL <- MM | 75 % | 3 / 0 / 6 | |
  | MCL <- Syn | 100 % | | |

- **Base dependence.**
  - Non-OA targets are identical on c4, eoa20, eoa30, pw20 and pw30.
  - Frac <- MOA: +.0104 to +.0118.
  - LOA <- Eff: +.0097 (pw20) to +.0155 (pw30). It depends on which A21 entry wins.
- **Extra 16 gold74 studies** (our reads, R7 only, 2-4 positives): MCL <- MM .0000, Frac <- MOA -.0250, LOA <- Eff .0000.
  Uninformative.

## 5. Requested families (MEASURED `census_dirs.csv`; d at a +.1 unless stated)

- **MCL <- MM.**
  - It is the largest and most consistent excess of the 66 (gold58 +.59, gold74 +.56, z 2.6). It is part of a coherent MCL
    pattern: decorrelating MCL helps for 8 of 11 helpers at +.1 (MM, Syn +.0091; Cont +.0068; Frac +.0057; Eff, LOA, MOA
    +.0045), and borrowing (a -.1) hurts for 9 of 11.
  - The theory agrees (rho .19 vs q -.24, a* .40).
  - The general-factor version (MCL minus the mean of the other 11) gives +.0023 at a .1 (P .40) and +.0091 at a .2.
  - It is still 9 positives with no independent ruler. R7 vs report loses -.0077, as the mechanism predicts but without
    discriminating.
  - Verdict: WEAK, the best of the census. Not robust.
- **Contusion pairs.**
  - Cont-Frac has an excess of +.34 (z 2.1), but q is about rho in both directions: Cont <- Frac -.0027 (P .17), Frac <- Cont
    +.0021 (P .63). DEAD, as in expert2 sec 3.3.
  - Syn-Cont deficit -.22: the borrows Syn <- Cont +.0012 (a -.1) and Cont <- Syn +.0020 (a -.1) are noise.
  - Contusion target: the best are Cont <- LOA / PFOA +.0040 (P .70 / .68), with excess 0. Report OOF loses -.013.
  - c4 already anti-couples Contusion with OA as the experts do (rows 58, 62, 64). DEAD.
- **Fracture pairs.**
  - Frac <- MOA +.0118 is the best in the census by d. Next come Frac <- Syn +.0090 (excess only +.03) and Frac <- MM +.0062
    (excess -.07). So Fracture also "improves" when you subtract labels with NO excess.
  - Frac <- Bak (excess +.40) gives only +.0035.
  - General factor: +.0021 (P .57).
  - Same specificity problem as expert2's acute <- DEG (DEG ranked 175 of 252 helper sets, CLAIMED). WEAK at best.
- **ACL pairs.**
  - ACL is at AUC .989 on gold58, so there is little room. Every ACL-target direction is <= +.0025.
  - ACL-PFOA / ACL-LOA deficits (c4 MORE anti-coupled than experts, -.18 / -.17) give borrow d +.0026 (PFOA <- ACL) and +.0019
    (LOA <- ACL). PFOA <- ACL a -.1 loses on OAI (-.0114 / -.0087 / -.0104), but OAI ACL is degenerative, so it is not a valid
    test.
  - LM <- ACL (excess +.17): +.0031 (P .70); OAI raw -.0009 / -.0033 / -.0027.
  - DEAD.
- **Synovitis pairs.**
  - LM-Syn has an excess of +.27 (P .97), yet the theory says decorrelation HURTS: LM <- Syn -.0062. The borrow LM <- Syn
    (a -.1) gives +.0075 (P .95), opposite to the coupling story, so it is noise or a ranking quirk. Syn <- LM +.0030.
  - MCL-Syn: the MCL factor again (+.0091).
  - Synovitis target: the best is Syn <- MOA +.0060 (P .76). Syn <- Eff (a -.1) +.0036 (P .76) is the killed E2.
  - OAI is invalid for every Synovitis pair (OAI Syn = Hoffa; OAI-free Syn = Eff).
  - DEAD.

## 6. Posterior, P(>0), expected public / private, value of one probe (MODEL on MEASURED inputs; `census_d.log`, `census_f.log`)

**Priors** for a single-label effect at |a| .1 on hidden expert labels (label-AUC units):
- **mech** N(+.001, .002) (GUESS): a mechanism-consistent pair without an independent ruler, set below E-OA's per-label hidden
  estimate.
- **cv** N(-.002, .002) (MEASURED prior mean): the selection-CV held-out mean.
- **mixture**: 50 / 50 of the two, reweighted by the marginal likelihood of the gold58 observation.

**Likelihood.** gold58 d with its bootstrap sd. The posterior weight on the data is only .04-.17, so the prior dominates.

**Columns.**
- n_pub / n_priv = paired read-noise sd (macro) on 390 / 910 studies.
- Pmove = P(the probe becomes the API best-id).
- Probe+R1 = E[private macro gain] of probing then following R1.

| candidate | n_pub / n_priv | E mech / cv / mixture | P(>0) mech / cv / mixture | P(private delta > 0) mixture | Pmove mixture | probe+R1 mixture |
|---|---|---|---|---|---|---|
| MCL <- MM | 2.5e-4 / 1.5e-4 | +1.2e-4 / -1.1e-4 / +2.8e-5 | .77 / .24 / .56 | .55 | .54 | +6.4e-5 |
| Frac <- MOA | 3.6e-4 / 2.2e-4 | +1.2e-4 / -1.2e-4 / +1.9e-5 | .77 / .23 / .54 | .53 | .52 | +4.9e-5 |
| LOA <- Eff (a -.1) | 1.9e-4 / 1.1e-4 | +1.2e-4 / -1.2e-4 / +2.0e-5 | .77 / .23 / .55 | .54 | .53 | +7.0e-5 |
| MCL <- mean11 | 1.4e-4 / 9.3e-5 | +1.1e-4 / -9.2e-5 / +2.8e-5 | .76 / .27 / .57 | .56 | .55 | +7.1e-5 |
| MCL <- Syn / Cont / Frac (MOA similar, not run) | 2.7-2.8e-4 / 1.5-1.6e-4 | about +1e-4 / -1.4e-4 / about 0 | .73-.74 / .20 / .49-.50 | .49-.50 | .49-.50 | +4.5 to +4.9e-5 |
| combo MCL <- MM + Frac <- MOA | 4.4e-4 / 2.6e-4 | mixture +4.6e-5 | mixture .57 | .55 | .54 | +8.6e-5 |
| MM <- LOA (OAI-killed; posterior ignores OAI) | 1.4e-4 / 8.4e-5 | +1.4e-4 / -7.3e-5 / +6.4e-5 | .81 / .32 / .64 | | | dominated by the OAI loss |

- **Honest summary.** E[hidden macro] about 0 to +3e-5 for every live candidate (range -1.2e-4 to +1.2e-4 by prior), with
  P(>0) about .5-.57.
- **The read cannot tell.** Public read noise is 1.4-3.6e-4: 1.2-3x the optimistic (mech) effect and 5-19x the mixture estimate.
- **Probe+R1 does not discriminate.** It is +4.5 to +9.5e-5 for every row. That is R1's generic option value for a zero-mean
  change under the assumed prior spread, so it does not single out any candidate.

## 7. Probe verdicts

**Robust enough for ONE public LB probe: NONE.** The bar is an independent ruler, survival of the controls, and P(>0) >= .7.

**Lottery ticket only, if a 10-21 slot would otherwise go unused and Eesh explicitly asks: MCL <- MM at a .1** on top of the
A21 best-id. It has the only above-chance census signal, a rubric mechanism, a coherent MCL-wide pattern, replication across
4 members, and theory P 1.00. Its weaknesses: 9 positives (3 carry it), split-half .33, and no ruler. It needs his go, a
PREREG amendment, a build following the `build_a21.py` pattern and an offline gate; none was done here.

**Not recommended, not even as a ticket:**
- **Frac <- MOA.** The specificity null failed and the extra16 check went the wrong way.
- **LOA <- Eff.** The deficit is a gold58 artefact and the effect is base-dependent.
- **Everything OAI-killed:** LM <- MM, MM <- LOA, LOA <- MM on eoa20.
- **All Synovitis, Contusion and ACL directions.**

**What would change this:** any expert-graded external set with MCL / Fracture labels, or the host answering on the
Synovitis criterion (737155). Neither is expected before 10-22.

## 8. Files (all in session_2026-10-20/push/)

| file | what it is |
|---|---|
| `cbase.py` | exact cell-18 rebuild of c4 / eoa20 on gold58 (parity .94113 / .94360) |
| `census_lib.py` | shared helpers |
| `census.py` -> `census.log`, `census_pairs.csv`, `census_dirs.csv`, `census_cv.json` | 66-pair census, 132 directions, EB, selection CV |
| `census_d.py` -> `census_d.log`, `census_candidates.csv`, `census_genfactor.csv`, `census_posterior.csv`, `census_mech.json` | candidates, held-out mechanism test, general factor, read noise, posterior |
| `census_e.py` -> `census_e.log` | z-count, noise max, base dependence (incl. the a21 bases), extra16, influence |
| `census_f.py` -> `census_f.log` | mixture posterior and the combo |
| `census_table.md` | the rendered sec 2 table |

## 9. Skeptic (10-20 ~15:20 IST, Claude, skeptic pass). RESEARCH ONLY.

Nothing was built, pushed, committed or submitted. No dataset, download, GPU job or box contact. No OAI data: every OAI number
here is CLAIMED from the files named in the sections above. PREREG, HANDOFF, PROGRESS, a20 and a21 were not touched (the a20
eoa20 notebook and the local R6e20 notebook were only read). Inputs: this file, `implied.md`, `bpick.md`, `e1build.md`.
Scripts and logs are in `push/skeptic/` (sec 9.5). Tags as in the header.

### 9.0 Verdicts, one line per candidate proposed for a public probe or a slot

| # | candidate (source) | verdict | reason |
|---|---|---|---|
| 1 | MCL <- MM, a .1, on the A21 best-id (sec 7 lottery ticket) | **KILL** | Under a correct null its gap is p .057, the smallest of 66, with 0 of 66 pairs below .0455 (the census z 2.6-4.1 came from bootstrap sds that collapse on empty 2x2 cells). It fails the specificity null the census applied to Fracture: MCL <- Bak (gap +.56) LOSES -.0034, MCL <- LOA (gap -.06) gains +.0045. The +.0091 is 4 pair flips of 441, and there is no ruler. Under the selection-CV prior, probe + R1 is -0.7 to -1.7e-5. |
| 2 | Frac <- MOA, a .1 | **KILL** | MOA-Frac gap p > .2 under the same null. Fracture gains from helpers with no excess (Frac <- Syn, gap +.04, +.0090; Frac <- MM, gap -.14, +.0063). 8.5 pair flips of 720; extra16 -.025. |
| 3 | MCL <- mean of the other 11, a .1 | **KILL** | +.0023 (1 pair flip) with the census definition, +.0091 (4 flips) if the mean is re-ranked first. The evidence changes size with a forking-path choice; bootstrap P .40. |
| 4 | LOA <- Eff, borrow a -.1 | **KILL** | The deficit exists only on gold58 (c4 .08 vs R7 train .33, report .39, gold74 expert .39), so on a train-like hidden set the borrow over-couples. 6 pair flips of 517, 83 % from 2 positives, base-dependent. |
| 5 | MCL <- Syn / Cont / Frac / MOA | **KILL** | Same non-specific MCL pattern as row 1: Spearman(gap, d) over MCL's 11 helpers is .34 (n 11). |
| 6 | combo MCL <- MM + Frac <- MOA | **KILL** | Two KILLs. Its E > 0 comes only from the GUESS "mech" prior. |
| 7 | LM <- MM, MM <- LOA | **KILL** (unchanged) | They lose on OAI (CLAIMED 10-19 logs). |
| 8 | LOA <- MM on top of the E-OA best (implied sec 6, armed if the regime posterior reaches <= -1 sd) | **KILL** | The trigger cannot fire: after the A21 reads, P(posterior mean <= -1 sd) is .000-.006 (MODEL, sec 9.2). At gold74 coupling the two OAI rulers disagree by more than the effect (raw +.00003 vs R1oa3 -.0003 to -.0005, CLAIMED). |
| 9 | Synovitis, Contusion, ACL directions | **KILL** (unchanged) | See sec 5. |
| 10 | pw25 on 10-21 (implied sec 7) | **KILL** | Its +1.9e-4 and P(.960) .64 are measured against eoa20, and the pw20 / pw30 reads already capture them (public-noise corr .99 with both). After A21 it becomes best-id with P .18-.28. It adds +0.07 to +0.15e-4 private and +.02 to +.04 to P(any .960). |
| 11 | eoa25 / pw35 / eoa35 / pw40 | **KILL** | After A21: eoa25 -0.02 to +0.04e-4, pw35 -0.13 to +0.02e-4. eoa35 and pw40 were already negative before the read. |
| 12 | R6e20 as B (bpick, one slot) | **KEEP** | It is a B hedge, not an LB push. Cell 18 is byte-equal to the scored eoa20 and only cell 14 differs (verified, sha 2a5ef93a). The OA columns match c4 at Spearman .989-.9996, and the gate PASSed. It falls below R6 only if E-OA fails privately (P about .02-.06). Still needs Eesh's go, Amendment 22 and the 17/17 commit gate. |
| 13 | E1 pow2 on E-OA (e1build contingency) | **KILL** | On the gold58 float32 path it is -.00010 (P .45). Its only support is the report-OOF ruler (CLAIMED). The tie-break side effect is bounded at about 1e-4 per label, of either sign. |

### 9.1 The census "above-chance MCL signal" fails a correct null (MEASURED `skeptic/skeptic_g58.log`)

- **Empty cells.** Two of the five "beyond 2 sd" pairs have an empty co-occurrence cell on gold58:
  - MCL-Bak: 2x2 table (37, 12, 9, 0). MCL-MOA: (34, 15, 9, 0). No MCL positive has Baker's or Medial OA.
  - Because no bootstrap resample can create a co-occurrence, the bootstrap tet stays near -.45 (sd .11).
  - That is why the census z for these two pairs (4.1 and 2.9) is inflated.
- **Parametric null.** Latent normal with c4's normal-score correlation for all 12 labels, gold58 positive counts fixed,
  n 58, 4,000 sims. This is the null "the experts couple exactly as c4 does".
  - Null sd of the tet: MCL-Bak .25, MCL-MOA .25, MCL-MM .22.
  - Two-sided p: MCL-MM .057, MOA-LOA .070, LOA-PFOA .091, MCL-Bak .131, Cont-Frac .135, LM-Syn .136, Bak-Frac .154,
    MCL-Cont .19, MOA-PFOA .19, MCL-MOA .21. Everything else is higher.
  - 0 of 66 pairs fall below .0455; the joint-null mean is 1.27.
  - The "5 of 66, 4 of them MCL" signal disappears. The MCL rows are no more surprising than the rest.
- **Calibration of the test.** The LB-confirmed OA trio pairs rank 2nd, 3rd and 9th (p .07-.19).
  - So gold58 cannot certify even a real excess.
  - MCL-MM is where the OA trio stood before OAI and the LB, but it has neither of those rulers, and it fails specificity
    (next bullet).
- **Specificity** (MEASURED from `census_dirs.csv`). Gap and d at a +.1 for each of MCL's 11 helpers:

  | helper | gold58 gap | d at a +.1 |
  |---|---|---|
  | MM | +.59 | +.0091 |
  | Syn | +.30 | +.0091 |
  | Cont | +.39 | +.0068 |
  | Frac | +.19 | +.0057 |
  | Eff | +.38 | +.0045 |
  | LOA | -.06 | +.0045 |
  | MOA | +.39 | +.0045 |
  | ACL | +.28 | +.0023 |
  | LM | -.14 | .0000 |
  | PFOA | +.18 | -.0023 |
  | Bak | +.56 | -.0034 |

  - Spearman(gap, d) is .34.
  - The second-largest excess (Bak) LOSES, and a helper with no excess (LOA) gains. This is the specificity failure the
    census used to demote Frac <- MOA.
  - MCL is not unique either. The number of helpers that help at +.1 is 9 of 11 for Fracture, 8 for MCL and 7 for Baker's.
    GUESS: rare acute labels near AUC .94-.98 gain from subtracting almost any degenerative score, because their few
    out-of-order negatives are degenerative knees.
- **Size.** The single-label d is a handful of pair flips:
  - MCL <- MM +.0091 = 4.0 of 441 pairs (9 x 49). Frac <- MOA = 8.5 of 720. LOA <- Eff = 6.0 of 517.
  - 3 MCL positives carry MCL <- MM (+2, +1, +1). Dropping them and re-ranking still gives +.0087 to +.0068, because the
    re-rank creates 2 new flips.
  - So the value is a near-tie artefact of the percentile grid, not a stable estimate.
- **Forking path.** MCL <- mean11 is +.0023 with `pct(pct(f_t) - a * mean_o pct(f_o))` and +.0091 if the mean is re-ranked
  first.

### 9.2 implied.md arithmetic (counts MEASURED by re-running `work/lbpos.py`; conversions MODEL; `skeptic/skeptic_lb.log`, `skeptic/skeptic_abc.log`)

- **Counts confirmed.**
  - c4: r217, with 23 teams above it inside .958 (04:17).
  - .959: 66 teams at both 05:52 and 05:53. eoa10 r167 (36 above, 29 below); eoa20 r135 (4 above, 61 below).
  - .960: r87-130 (44). .958: r197-1537 (1,341).
  - Between 05:52 and 05:53 only our row and one .954 team changed.
- **Ordering within a display band is by unrounded score, not by submission time.** Our two newest submissions sat mid-band
  (29 older .959 teams were below eoa10). So the in-band position method is valid.
- **Weakest input: the density just below the .959 floor.** It sets c4's distance below .959.
  - implied uses k' = k + U(0, 1.5). One exponential carrying all 1,341 .958 teams would need k' about 4.3.
  - Varying k' from flat to 4.3 on the ALL ruler: 1 - u_c4 moves from .30 to .19, G10 from 7.0 to 5.9e-4 and G20 from 12.1
    to 11.1e-4. The gap to .960 stays 0.8e-4.
  - implied's pooled 6.1 [4.4, 7.8], 11.3 [9.3, 12.7] and 1.0 [0.5, 2.5] cover all four cases. **The arithmetic holds**,
    including G20 = x20 + (1 - u_c4) = .90 + .24 band.
- **Correction: the G20 / G10 ratio.**
  - implied's "1.69 [1.34, 1.97]" is the median over the 70 % of LB draws where the OAI curve inversion is valid (the `ok`
    mask in `gapfit_pub.py`). Over the whole pooled posterior the ratio is 1.84 [1.40, 2.48].
  - No downstream number uses the ratio (the ABC uses G10 and G20).
  - But "the ratio alone would say gap ~.15-.18" should read "about .25, or outside the curve family for 30 % of draws".
    Either way the ratio is uninformative, because the noise is slope-like.
- **"Pessimists 2.3 sigma low" depends on the noise scale.**

  | read-noise assumption | R5 revision P | ladder P |
  |---|---|---|
  | x1.0, corr .94 (implied) | .023 | .026 |
  | x1.25 | .089 | .096 |
  | x1.5 | .19 | .20 |
  | corr .85 | .051 | .084 |
  | corr .98 | .001 | .000 |

  - The noise was measured on OAI / report proxies, not on expert labels with c4.
  - E-OA > 0 is robust. The strength of the rejection (1.3 to 2.3 sigma) is model-dependent. With eoa20 alone it is 2.3
    sigma at x1.0 and 1.5 at x1.5.
- **Regime identification is weak** (MODEL, 3 ABC variants).
  - The posterior sd of the regime shift is .63-.75, against a prior sd of .86-1.01. "Public labels at about gold74
    coupling" is mostly the prior (the flat prior gives -0.54).
  - After the A21 reads, the preposterior sd of the posterior mean is only .16-.24. So P(shift <= -1 sd) is .000-.006 and
    the implied sec 6 trigger will not fire.
- **Cross-check.** The private eoa20 - c4 estimate is +9.7e-4 (implied ABC) vs +8.3e-4 (bpick, different prior and noise
  route). Two independent MODEL routes agree within 1.4e-4.

### 9.3 E-OA shapes after the A21 reads (MODEL, `skeptic/skeptic_abc.log`; ABC on box aggregates already on the laptop)

| ABC variant | oracle headroom over R1-of-A21 (13-shape grid) | P(pw25 becomes best-id) | E[private gain of adding pw25] | P(any .960): A21 -> + pw25 |
|---|---|---|---|---|
| P_FT, N(0,1), logN (primary) | +1.40e-4 | .28 | +0.15e-4 | .70 -> .74 |
| P_AREA, N(0,1), logN | +1.05e-4 | .19 | +0.08e-4 | .67 -> .70 |
| P_FT, flat, log-uniform | +1.37e-4 | .18 | +0.07e-4 | .79 -> .81 |

- PW25's public noise correlates .988 / .991 with PW20 / PW30. Once both are read, a pw25 read is mostly a re-read.
- Even an oracle choice over the whole grid beats R1-of-A21 by only +1.0 to +1.4e-4 privately. **The A21 reads close the
  E-OA shape question to within about 1e-4.**

### 9.4 What survives

- **Only R6e20, as B.** A is decided by R1 on the A21 reads.
- **No probe on the census / implied / e1 lists is worth a slot on value.** Slots are not the constraint (10-21 has 5).
  Each probe costs Eesh's go, a PREREG amendment and a gate, for about +1e-5 private at best.
- **R1 option value.** A zero-mean probe followed by R1 is worth +6e-5 under the census mixture prior. Under the
  selection-CV prior alone it is -0.7 to -1.7e-5 (MODEL: mu -1.1 to -1.7e-4, tau 1.6e-4, read sd 2.5e-4). It does not
  favour any listed candidate.

### 9.5 Files (`push/skeptic/`)

| file | what it is |
|---|---|
| `skeptic_g58.py` -> `skeptic_g58.log` | 2x2 tables, parametric joint null for the 66 gaps, pair-flip arithmetic, MCL <- MM influence |
| `skeptic_lb.py` -> `skeptic_lb.log` | LB conversion under 4 below-floor densities; prior_check under noise and corr sensitivity |
| `skeptic_abc.py` -> `skeptic_abc.log` | pw25 / eoa25 / pw35 after A21; regime preposterior; unconditional G20 / G10 ratio |
