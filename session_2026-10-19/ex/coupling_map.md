# coupling_map.md: label-coupling map, expert vs model vs report (10-19/20 night, Claude). RESEARCH + MEASUREMENT ONLY.

Nothing was built, pushed, committed or submitted. No dataset, no GPU job, no scheduled task, PREREG, HANDOFF, PROGRESS or a20
file was touched. Box use: three read-only CPU python runs piped over ssh stdin (smoke, full map, follow-up). They wrote nothing
on the box and printed aggregates only (n, prevalence, phi, tetrachoric, rho, AUC deltas, P, ESS). No OAI id, row or per-knee
value left the box. Only aggregate gold74 2x2 counts were sent TO the box (embedded in the piped script).
Tags: MEASURED = computed tonight (file named); CLAIMED = older note says so (file named); MODEL = arithmetic on stated
assumptions; GUESS = judgement. Abbreviations: MM / LM = medial / lateral meniscus, MOA / LOA / PFOA = OA compartments,
Eff, Syn, Bak = Effusion, Synovitis, Baker's. "Macro" = 12-label macro AUC (single-label delta / 12).

## 0. Answer (top pairs and groups)

1. **The OA trio is the only group whose excess coupling is large, consistent across OAI, gold58 and gold74, and removable.**
   E-OA already covers it (MEASURED, sec 2, ranks 1, 2 and 4):

   | pair | gap OAI | gap gold58 (P>0) | gap gold74 |
   |---|---|---|---|
   | LOA-PFOA | +.76 | +.36 (.99) | +.42 |
   | MOA-LOA | +.62 | +.33 (.99) | +.26 |
   | MOA-PFOA | +.57 | +.26 (.97) | +.21 |

   - Gap = model Spearman minus expert tetrachoric.
   - The models reproduce the REPORT coupling almost exactly (R7 OOF train rho .73 / .83 / .73 vs report tetrachoric
     .77 / .83 / .80). This is the report-inheritance signature.
2. **New pair: MM-LOA (medial meniscus tear vs lateral OA), rank 3.**
   - Expert coupling is about 0 in all three expert sources: OAI tet .00, gold58 .02, gold74 .05 +- .21.
   - Reports couple the two at .42, R7 at .41 (train), c4 at .31, the OAI-free models at .23.
   - Gaps: +.23 OAI, +.30 gold58 (P .93), +.26 gold74.
   - **Transform LOA <- MM** (out_LOA = pct(f_LOA) - .1 pct(f_MM)):
     - Passes criterion 1 at a .1 (OAI 3 models): LOA AUC +.0027 / +.0019 / +.0023, P .98 / .93 / .97; a -.1 control
       -.0058 / -.0050 / -.0056.
     - Passes criterion 2 under pair OR-matching to gold74 (+.0018 / +.0010 / +.0014) and gold74 cell matching
       (+.0017 / +.0009 / +.0015).
     - Turns negative if hidden MM-LOA coupling is 1 sd above gold74 (-.0030 / -.0038 / -.0031).
     - gold58 c4 agrees (+.0077 LOA AUC). It loses on report labels (-.0013), as the mechanism predicts.
     - Size alone: about +.0002 macro.
   - **On top of E-OA .1 the increment is fragile:**

     | scheme | increment, macro (3 models) | P |
     |---|---|---|
     | raw | +.00016 / +.00010 / +.00006 | .94 / .84 / .74 |
     | 5-label coupling-only reweight (R2m) | +.00018 / +.00016 / +.00007 | |
     | skeptic OA-cell reweight (R1oa3) | -.00022 / -.00029 / -.00034 | |

     - It fails the P >= .90 rule for s1045.
     - gold58 c4 increment: +.00129 (P .95).
   - **Verdict: INCONCLUSIVE, expected about +.0001 macro (MODEL). Not worth a slot on its own (GUESS).**
3. **Second-tier consistent-excess pairs carry no supported transform** (MEASURED, sec 3): LM-Syn, MOA-Syn, LM-PFOA, MM-LM,
   Eff-Syn, ACL-LM.
   - On OAI, decorrelation either loses in both directions or dies under reweighting.
   - OAI cannot test any Synovitis pair for c4:
     - OAI "Synovitis" is Hoffa synovitis.
     - The OAI-free models' Synovitis output is essentially their Effusion output (rho .99 on OAI). This is inherited from
       the frac75 training labels, where Eff-Syn Spearman is .95.
     - c4's Eff-Syn rho on gold58 is .74.
4. **Four OAI passes are OAI-specific; they reverse or vanish on RSNA. Discard them** (MEASURED, sec 3):
   - PFOA <- ACL: biggest OAI pass, +.0092 / +.0057 / +.0080 and +.0115 / +.0079 / +.0102 after reweighting. But c4 already
     over-ANTI-couples ACL-PFOA on gold58 (-.60 vs expert -.41).
   - PFOA <- Eff: c4 shows no PF-Eff excess (rho .06).
   - PFOA <- Syn: the OAI Syn score is the Eff score.
   - Syn <- Eff: +.0004, which is the killed E2. gold58 c4 -.0084.
5. **Deficit side** (c4 couples LESS than experts on RSNA, consistently on gold58 and gold74; the reports are deficient too):
   - The pairs: Baker's with OA and menisci (MOA-Bak -.22 / -.23, LOA-Bak -.26 / -.16, LM-Bak -.24 / -.19), LOA-Eff
     (-.29 / -.31) and LM-MOA (-.17 / -.13).
   - OAI cannot confirm any of them (OAI expert MOA-Bak tet .01, Baker's n 653).
   - gold58 c4 Baker's <- OA borrowing at a -.1 loses (-.0082 x3).
   - Same-compartment meniscus borrow (MOA += .1 MM, LOA += .1 LM) on top of E-OA .1:
     - raw: +.00061 / +.00059 / +.00029 macro (P 1.0)
     - s1045 is negative under every reweighting
     - gold58 c4: -.00007
     - **INCONCLUSIVE, fails criterion 2.**
6. **Side finding on E-OA itself** (MEASURED, sec 4, matters for eoa20 vs eoa10).

   | model | OA rho | a .1, gold74-coupled (R1oa3) | a .2, gold74-coupled (R1oa3) | P>0 at a .2 |
   |---|---|---|---|---|
   | 5-fold mean D5 (skeptic's model) | about .76 | +.00041 | +.00043 | |
   | single-fold s1044 | .71 | +.00003 | -.00042 | .01 |
   | single-fold s1045 | .69 | +.00008 | -.00032 | .06 |

   - The 5-fold mean reproduces the skeptic's figures.
   - The single-fold models gain about 0 at a .1 and lose at a .2.
   - At +1 sd coupling (R2m_hi), a .2 loses for all 3 models (-.00018 / -.00105 / -.00087), while a .1 is +.00012 /
     -.00024 / -.00015.
   - So the E-OA gain at RSNA coupling needs a high-rho ensemble. c4 has OA rho about .82 on gold58, above D5 (MODEL: the
     gain scales with rho - q).
   - This strengthens the skeptic's preference for a .1 over a .2.

## 1. Method (MEASURED unless tagged)

**Expert labels on OAI** (box, `~/oai_build`; B1 is the primary, rubric-like binarisation, B2 the alternative):

| label | B1 | B2 |
|---|---|---|
| ACL | T123 >= .5 (partial + complete) | complete |
| MM / LM | tear (T123 = 1) | incl. intrasubstance .2 |
| OA trio | P_FT: any compartment cartilage subregion with full-thickness loss, MOAKS m_cm_* (as `oai_eoa_check.py`) | T123 composite >= .5 (includes osteophytes and JSN) |
| Effusion | MOAKS effusion-synovitis grade >= 2 | grade >= 1 |
| Synovitis | Hoffa grade >= 1 | grade >= 2 |
| Baker's | 1 | same |

- Coverage: ACL 1,654; menisci and OA 2,399; Eff 1,660; Syn 1,659; Bak 835 (625-653 in pairs).

**Coupling measures:**
- Expert coupling: phi, tetrachoric (latent, +.5 cell smoothing) and Spearman on graded T123.
- Model coupling: Spearman of predictions on the same knees or studies.
- The primary gap is model Spearman minus expert tetrachoric. Both sit on a continuous-latent scale. Phi is shown for
  reference; it is attenuated by dichotomisation.
- Caveat (MODEL): a noisy but unbiased model's coupling is attenuated below the latent label coupling. So positive gaps are
  conservative evidence of excess, while negative gaps are partly attenuation.

**Model sets:**
- OAI: the 3 OAI-free models, teacher_D5_mean_v00 (s1043 frac75, 5-fold mean), f0_s1044_ttr_mix50 and f0_s1045_ttr288frac75.
- gold58: c4 final, rebuilt as `trick/regime_work/m14_gate.py` (parity: macro .94102 = m14_gate.log), R6, S6fall
  (n42gpu_gold58) and Raptor (.75 r384 + .25 r224).
- R7 OOF: on gold58, and on train (4,333 studies after removing gold74).

**Labels on the RSNA side:**
- Expert: gold74 (n 65-74 per pair, bootstrap sd of tet) and gold58.
- Report: report_labels_v2, YES/NO with UNK dropped pairwise, plus a UNK=0 variant.
- Note: `_data/train.csv` labels exist only for the 58 gold studies (MEASURED), so report_labels_v2 is the report regime.

**Directional pair transform:** out_t = pct(f_t) - a * pct(f_o), a in {-.1, .1, .2}.
- The 3 OAI models were tested per direction (72 directions), with a paired bootstrap of 400.
- Coupling-matched reweighting (the skeptic method, by the 2-label cell):

  | scheme | what it matches |
  |---|---|
  | R1 | gold74 4-cell joint |
  | R2 | OAI marginals + gold74 log odds ratio (coupling only) |
  | R2hi | R2 with gold74 log OR + 1 sd |

**Group compositions:** on OAI (n 2,399) with 5 schemes:

| scheme | what it does |
|---|---|
| raw | OAI as is |
| R1oa3 | skeptic 3-label OA cells matched to gold74 |
| R2m | IPF on the 5-label cell (MOA, LOA, PF, MM, LM); every pairwise 2x2 set to OAI marginals + gold74 log OR |
| R2m_hi | R2m with every log OR + 1 sd |
| R2m_lo | R2m with every log OR - 1 sd |

Plus gold58 c4 and R7 (consistency only), and R7 vs report labels (should lose).

**Files** (all in `session_2026-10-19/ex/`):

| step | script | output |
|---|---|---|
| local coupling | `local_coupling.py` | `.log`, `local_coupling.json` |
| box coupling | `box_coupling_template.py` + `make_box_script.py` -> `box_coupling_run.py` | `box_coupling.log` (aggregates + JSON) |
| merge and rank | `merge_map.py` | `merge_map.out` (full 36-pair table + all 72 directional rows), `coupling_rows.json` |
| gold58 gap bootstrap | `local_gapboot.py` | `.log` |
| box group compositions | `box_followup_template.py` + `make_followup.py` -> `box_followup_run.py` | `box_followup.log` |
| local group compositions | `local_compose.py` | `.log` |
| table rendering | `table_md.py` | `table_md.out` |

## 2. Coupling map (36 pairs, sorted by consistent gap)

**How to read the table:**
- Consistent gap = the smallest of (gap OAI, gap gold58, gap gold74) when all three are > 0, or the largest when all are < 0.
  "mixed" means the signs disagree.
- Gap OAI = OAI-free models (mean of 3) minus OAI expert tetrachoric, on the same knees.
- Gap gold58 = c4 minus the gold58 expert tetrachoric, on the same 58 studies. The paired bootstrap sd and P>0 are in
  brackets.
- Gap gold74 = c4 (gold58) minus the gold74 expert tetrachoric.

All MEASURED.

| # | pair | OAI expert tet (phi) | OAI models rho | gap OAI | gold58 tet | gold74 tet +- sd | c4 rho g58 | gap g58 (sd, P>0) | gap g74 | R7 rho g58 / train | report tet (UNK=0) | consistent gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | LOA-PFOA | -0.01 (-0.01) | +0.75 | +0.76 | +0.45 | +0.39 +- 0.20 | +0.81 | +0.36 (0.19, 0.99) | +0.42 | +0.76 / +0.73 | +0.77 (+0.64) | +0.36 |
| 2 | MOA-LOA | +0.12 (+0.06) | +0.74 | +0.62 | +0.51 | +0.59 +- 0.16 | +0.85 | +0.33 (0.20, 0.99) | +0.26 | +0.79 / +0.83 | +0.83 (+0.67) | +0.26 |
| 3 | MM-LOA | +0.00 (+0.00) | +0.23 | +0.23 | +0.02 | +0.05 +- 0.21 | +0.31 | +0.30 (0.21, 0.93) | +0.26 | +0.25 / +0.41 | +0.42 (+0.25) | +0.23 |
| 4 | MOA-PFOA | +0.13 (+0.07) | +0.70 | +0.57 | +0.56 | +0.61 +- 0.13 | +0.82 | +0.26 (0.15, 0.97) | +0.21 | +0.78 / +0.73 | +0.80 (+0.65) | +0.21 |
| 5 | MOA-Syn | +0.37 (+0.22) | +0.62 | +0.25 | +0.38 | +0.30 +- 0.17 | +0.52 | +0.14 (0.17, 0.80) | +0.22 | +0.24 / +0.44 | +0.70 (+0.44) | +0.14 |
| 6 | LM-Syn | +0.23 (+0.12) | +0.36 | +0.13 | +0.25 | +0.27 +- 0.17 | +0.54 | +0.29 (0.16, 0.97) | +0.27 | +0.37 / +0.40 | +0.45 (+0.37) | +0.13 |
| 7 | LM-PFOA | -0.05 (-0.03) | +0.32 | +0.37 | +0.19 | +0.15 +- 0.18 | +0.32 | +0.13 (0.18, 0.74) | +0.17 | +0.37 / +0.30 | +0.31 (+0.18) | +0.13 |
| 8 | ACL-LM | +0.33 (+0.15) | +0.46 | +0.13 | -0.06 | -0.06 +- 0.17 | +0.12 | +0.18 (0.18, 0.86) | +0.19 | -0.01 / +0.23 | +0.42 (+0.40) | +0.13 |
| 9 | MM-LM | -0.01 (-0.01) | +0.12 | +0.13 | +0.18 | +0.18 +- 0.17 | +0.34 | +0.15 (0.15, 0.88) | +0.16 | +0.37 / +0.43 | +0.45 (+0.40) | +0.13 |
| 10 | Eff-Syn | +0.47 (+0.25) | +0.99 | +0.52 | +0.59 | +0.62 +- 0.12 | +0.74 | +0.16 (0.13, 0.88) | +0.12 | +0.68 / +0.83 | +0.82 (+0.64) | +0.12 |
| 11 | PFOA-Bak | +0.10 (+0.06) | +0.20 | +0.10 | +0.39 | +0.37 +- 0.18 | +0.44 | +0.06 (0.19, 0.66) | +0.07 | +0.36 / +0.30 | +0.23 (+0.30) | +0.06 |
| 12 | MM-Syn | +0.19 (+0.12) | +0.30 | +0.11 | +0.30 | +0.25 +- 0.18 | +0.36 | +0.05 (0.17, 0.59) | +0.10 | +0.24 / +0.37 | +0.49 (+0.30) | +0.05 |
| 13 | ACL-Bak | -0.07 (-0.03) | +0.15 | +0.21 | -0.13 | -0.22 +- 0.19 | -0.08 | +0.05 (0.21, 0.60) | +0.14 | -0.26 / -0.06 | +0.14 (+0.07) | +0.05 |
| 14 | ACL-Syn | +0.37 (+0.15) | +0.46 | +0.10 | -0.13 | -0.07 +- 0.18 | -0.02 | +0.10 (0.19, 0.71) | +0.05 | +0.14 / +0.13 | +0.33 (+0.20) | +0.05 |
| 15 | LM-Eff | +0.30 (+0.16) | +0.34 | +0.05 | +0.23 | +0.23 +- 0.19 | +0.33 | +0.10 (0.15, 0.70) | +0.10 | +0.28 / +0.30 | +0.34 (+0.31) | +0.05 |
| 16 | PFOA-Syn | +0.26 (+0.16) | +0.52 | +0.26 | +0.45 | +0.29 +- 0.17 | +0.47 | +0.02 (0.16, 0.57) | +0.18 | +0.25 / +0.33 | +0.64 (+0.37) | +0.02 |
| 17 | ACL-MM | +0.52 (+0.24) | +0.38 | -0.14 | +0.24 | +0.10 +- 0.19 | +0.21 | -0.03 (0.15, 0.41) | +0.11 | +0.12 / +0.13 | +0.35 (+0.32) | mixed |
| 18 | ACL-MOA | +0.53 (+0.27) | +0.50 | -0.02 | -0.42 | -0.38 +- 0.17 | -0.37 | +0.05 (0.17, 0.67) | +0.01 | -0.41 / -0.07 | +0.28 (+0.14) | mixed |
| 19 | ACL-LOA | +0.43 (+0.21) | +0.52 | +0.09 | -0.08 | +0.04 +- 0.21 | -0.30 | -0.22 (0.22, 0.15) | -0.34 | -0.27 / -0.02 | +0.32 (+0.16) | mixed |
| 20 | ACL-PFOA | -0.10 (-0.04) | +0.17 | +0.27 | -0.41 | -0.46 +- 0.14 | -0.60 | -0.19 (0.18, 0.19) | -0.15 | -0.50 / -0.20 | +0.10 (-0.02) | mixed |
| 21 | MM-MOA | +0.58 (+0.36) | +0.64 | +0.05 | +0.63 | +0.67 +- 0.12 | +0.54 | -0.09 (0.14, 0.27) | -0.13 | +0.46 / +0.58 | +0.66 (+0.55) | mixed |
| 22 | MM-PFOA | -0.11 (-0.07) | +0.18 | +0.29 | +0.07 | +0.33 +- 0.16 | +0.23 | +0.17 (0.16, 0.86) | -0.09 | +0.28 / +0.32 | +0.38 (+0.25) | mixed |
| 23 | MM-Bak | +0.07 (+0.04) | +0.12 | +0.05 | +0.50 | +0.48 +- 0.15 | +0.41 | -0.09 (0.16, 0.32) | -0.07 | +0.26 / +0.22 | +0.31 (+0.26) | mixed |
| 24 | LM-MOA | -0.12 (-0.06) | +0.34 | +0.47 | +0.61 | +0.56 +- 0.16 | +0.43 | -0.17 (0.16, 0.10) | -0.13 | +0.42 / +0.43 | +0.43 (+0.31) | mixed |
| 25 | LM-LOA | +0.57 (+0.35) | +0.74 | +0.17 | +0.65 | +0.61 +- 0.14 | +0.56 | -0.09 (0.18, 0.23) | -0.05 | +0.59 / +0.55 | +0.60 (+0.47) | mixed |
| 26 | LM-Bak | +0.12 (+0.06) | +0.13 | +0.01 | +0.46 | +0.40 +- 0.18 | +0.22 | -0.24 (0.17, 0.09) | -0.19 | +0.07 / +0.18 | +0.23 (+0.19) | mixed |
| 27 | MOA-Eff | +0.50 (+0.30) | +0.59 | +0.09 | +0.12 | +0.27 +- 0.18 | +0.11 | -0.01 (0.19, 0.45) | -0.17 | +0.11 / +0.38 | +0.45 (+0.36) | mixed |
| 28 | MOA-Bak | +0.01 (+0.01) | +0.20 | +0.18 | +0.70 | +0.71 +- 0.14 | +0.48 | -0.22 (0.15, 0.08) | -0.23 | +0.36 / +0.27 | +0.32 (+0.36) | mixed |
| 29 | LOA-Eff | +0.31 (+0.17) | +0.54 | +0.23 | +0.37 | +0.39 +- 0.18 | +0.08 | -0.29 (0.20, 0.09) | -0.31 | +0.14 / +0.33 | +0.39 (+0.28) | mixed |
| 30 | LOA-Syn | +0.20 (+0.11) | +0.57 | +0.37 | +0.56 | +0.48 +- 0.19 | +0.45 | -0.12 (0.17, 0.23) | -0.03 | +0.20 / +0.39 | +0.61 (+0.37) | mixed |
| 31 | LOA-Bak | -0.14 (-0.07) | +0.19 | +0.33 | +0.63 | +0.52 +- 0.18 | +0.36 | -0.26 (0.17, 0.09) | -0.16 | +0.23 / +0.22 | +0.18 (+0.25) | mixed |
| 32 | PFOA-Eff | +0.16 (+0.09) | +0.49 | +0.33 | +0.15 | +0.09 +- 0.18 | +0.06 | -0.09 (0.17, 0.31) | -0.03 | +0.11 / +0.26 | +0.34 (+0.27) | mixed |
| 33 | Eff-Bak | -0.19 (-0.10) | +0.14 | +0.33 | +0.10 | +0.28 +- 0.19 | +0.21 | +0.11 (0.21, 0.72) | -0.07 | +0.22 / +0.20 | +0.49 (+0.30) | mixed |
| 34 | Syn-Bak | +0.12 (+0.07) | +0.16 | +0.04 | +0.34 | +0.44 +- 0.17 | +0.32 | -0.02 (0.18, 0.45) | -0.12 | +0.09 / +0.14 | +0.48 (+0.27) | mixed |
| 35 | ACL-Eff | +0.47 (+0.24) | +0.46 | -0.01 | +0.38 | +0.29 +- 0.17 | +0.22 | -0.16 (0.14, 0.15) | -0.07 | +0.17 / +0.10 | +0.43 (+0.35) | -0.01 |
| 36 | MM-Eff | +0.30 (+0.17) | +0.28 | -0.02 | +0.35 | +0.36 +- 0.18 | +0.18 | -0.17 (0.14, 0.13) | -0.17 | +0.17 / +0.33 | +0.35 (+0.32) | -0.02 |

### Reading the map

1. **Report-inheritance signature.** Report tet, R7 train rho and c4 rho are all high, and the expert tet is low in every
   expert source. This holds cleanly only for:
   - LOA-PFOA
   - MOA-LOA
   - MOA-PFOA
   - MM-LOA
   - (weaker) MM-LM and LM-PFOA
2. **The OAI population differs from RSNA for ACL pairs.**
   - In OAI, ACL tears are degenerative: expert ACL-OA tet is .43 to .53, against -.38 / +.04 / -.46 on gold74 for MOA / LOA /
     PFOA.
   - So every ACL pair is mixed or OAI-specific. CLAIMED caveat in the task, now MEASURED.
3. **Model coupling depends on the population, not only on the model.**
   - R7 ACL-PFOA is -.20 on train but -.50 on gold58 (gold58 is ACL-enriched, 24 of 58).
   - So gold58 gaps must compare against gold58 labels. That is why gap g58 uses the gold58 tet.
4. **Synovitis pairs cannot be tested for c4 on OAI.**
   - The OAI Synovitis label is Hoffa synovitis.
   - The OAI-free models' Syn output is their Eff output (rho .99). The cause is that the frac75 training labels have Eff-Syn
     Spearman .95 (MEASURED, box `~/novel/frac75/labels_frac75_logit.csv`, RSNA train, n 4,407).
   - c4's members are not built that way: c4 Eff-Syn is .74 on gold58.

## 3. Directional transforms on OAI (criterion 1 and 2), gold58 c4 (criterion 3) and report regime

Values are single-label AUC deltas for the 3 OAI models (D5mean / s1044 / s1045); the P column is the minimum over the 3 models.

**Pair columns** (criterion 2, reweighted to gold74 coupling at a .1):
- R2 = OR-matched, R1 = cell-matched, R2hi = gold74 log OR + 1 sd.

**Sign conventions:**
- For a decorrelation direction, criterion 1 needs a .1 > 0 with P >= .90 for all 3 models and an a -.1 control < 0.
- For a borrow direction, the roles of a -.1 and a .1 swap.

All values MEASURED (`box_coupling.log`, `merge_map.out`, `local_coupling.json`).

| direction | a -.1 | a .1 (P min) | a .2 (P min) | R2 a .1 | R1 a .1 | R2hi a .1 | g58 c4 a .1 / .2 | R7 vs report a .1 / .2 | read |
|---|---|---|---|---|---|---|---|---|---|
| LOA <- PFOA | -.0079/-.0086/-.0074 | +.0073/+.0068/+.0053 (1.00) | +.0130/+.0122/+.0085 (1.00) | +.0031/+.0021/+.0007 | +.0029/+.0020/+.0005 | +.0012/+.0001/-.0014 | +.0077 / +.0010 | -.0024 / -.0074 | E-OA member; passes 1+2 at a .1 |
| PFOA <- LOA | -.0056/-.0038/-.0066 | +.0053/+.0030/+.0061 (1.00) | +.0097/+.0043/+.0110 (1.00) | +.0026/+.0001/+.0030 | +.0026/-.0000/+.0028 | +.0015/-.0012/+.0015 | +.0097 / +.0212 | -.0031 / -.0090 | E-OA member; 2 marginal |
| MOA <- LOA | -.0064/-.0068/-.0055 | +.0051/+.0052/+.0039 (1.00) | +.0083/+.0084/+.0054 (1.00) | +.0018/+.0019/+.0003 | +.0018/+.0019/+.0005 | +.0005/+.0006/-.0010 | -.0016 / -.0016 | -.0011 / -.0043 | E-OA member; passes 1+2 at a .1 |
| LOA <- MOA | -.0065/-.0055/-.0057 | +.0062/+.0048/+.0045 (1.00) | +.0112/+.0084/+.0068 (1.00) | +.0001/-.0014/-.0017 | -.0003/-.0018/-.0021 | -.0022/-.0038/-.0042 | +.0029 / +.0058 | -.0010 / -.0037 | fails 2 |
| PFOA <- MOA | -.0070/-.0057/-.0073 | +.0068/+.0046/+.0067 (1.00) | +.0124/+.0074/+.0121 (1.00) | +.0019/-.0007/+.0016 | +.0013/-.0018/+.0008 | +.0005/-.0022/+.0002 | +.0039 / +.0129 | -.0033 / -.0094 | fails 2 (s1044) |
| MOA <- PFOA | -.0070/-.0058/-.0055 | +.0053/+.0034/+.0034 (1.00) | +.0080/+.0035/+.0037 (0.97) | +.0005/-.0017/-.0020 | +.0004/-.0017/-.0019 | -.0008/-.0031/-.0034 | +.0031 / +.0016 | -.0028 / -.0086 | fails 2 |
| **LOA <- MM** | -.0058/-.0050/-.0056 | **+.0027/+.0019/+.0023 (0.93)** | +.0023/+.0008/+.0012 (0.59) | **+.0018/+.0010/+.0014** | **+.0017/+.0009/+.0015** | -.0030/-.0038/-.0031 | **+.0077 / +.0135** | -.0013 / -.0059 | **passes 1+2+3 at a .1 only; dies at +1 sd coupling** |
| MM <- LOA | -.0023/-.0012/-.0015 | -.0008/-.0020/-.0020 (0.01) | -.0049/-.0074/-.0075 | | | | +.0048 / -.0030 | -.0023 / -.0080 | reverse direction loses |
| PFOA <- LM | -.0058/-.0037/-.0066 | +.0035/+.0013/+.0042 (0.91) | +.0041/-.0002/+.0058 (0.48) | +.0008/-.0014/+.0014 | +.0019/-.0011/+.0017 | -.0014/-.0036/-.0009 | +.0051 / +.0064 | -.0021 / -.0076 | fails 2 (s1044) |
| MOA <- LM | -.0073/-.0065/-.0064 | +.0036/+.0033/+.0029 (1.00) | +.0036/+.0036/+.0021 (0.84) | -.0055/-.0056/-.0058 | -.0066/-.0071/-.0069 | -.0080/-.0081/-.0082 | -.0093 / -.0233 | -.0013 / -.0059 | OAI-only (gold74 LM-MOA tet .56) |
| PFOA <- MM | -.0078/-.0059/-.0080 | +.0053/+.0031/+.0055 (1.00) | +.0078/+.0031/+.0084 (0.95) | -.0029/-.0053/-.0029 | -.0026/-.0051/-.0028 | -.0060/-.0085/-.0060 | +.0000 / -.0026 | -.0029 / -.0092 | OAI-only (gold74 MM-PF tet .33) |
| PFOA <- ACL | -.0114/-.0087/-.0104 | +.0092/+.0057/+.0080 (1.00) | +.0159/+.0086/+.0137 (1.00) | +.0115/+.0079/+.0102 | +.0210/+.0136/+.0201 | +.0102/+.0067/+.0090 | +.0026 / -.0064 | -.0022 / -.0069 | OAI-only: OAI models +.17 vs c4 -.60 on gold58 |
| PFOA <- Eff | -.0083/-.0056/-.0066 | +.0070/+.0034/+.0050 (1.00) | +.0115/+.0041/+.0074 (0.95) | +.0081/+.0045/+.0060 | +.0091/+.0049/+.0073 | +.0056/+.0021/+.0036 | -.0026 / -.0142 | -.0038 / -.0111 | OAI-only: c4 PF-Eff rho .06 |
| PFOA <- Syn | -.0085/-.0058/-.0065 | +.0073/+.0037/+.0050 (0.99) | +.0125/+.0047/+.0076 (0.98) | +.0070/+.0034/+.0046 | +.0064/+.0030/+.0042 | +.0058/+.0020/+.0034 | +.0039 / -.0097 | -.0020 / -.0077 | OAI Syn score = Eff score; RSNA gap g58 +.02 |
| Syn <- Eff | -.0004/-.0003/-.0004 | +.0004/+.0005/+.0005 (0.98) | +.0009/+.0009/+.0013 (0.98) | +.0004/+.0004/+.0005 | +.0011/+.0006/+.0007 | +.0003/+.0004/+.0004 | -.0084 / -.0197 | -.0024 / -.0100 | trivial size (rho .99); = killed E2 |
| LM <- Syn | -.0027/-.0012/-.0017 | -.0020/-.0039/-.0034 (0.00) | -.0105/-.0133/-.0127 | | | | -.0043 / -.0186 | -.0035 / -.0106 | excess not removable |
| Syn <- LM | +.0029/+.0029/+.0017 | -.0052/-.0050/-.0041 (0.00) | -.0129/-.0123/-.0109 | | | | +.0030 / -.0006 | -.0079 / -.0191 | OAI Hoffa borrow; RSNA opposite |
| MM <- LM ; LM <- MM | -.0021 ; -.0035 (D5) | -.0012/-.0013/-.0022 ; -.0006/-.0008/-.0025 | | -.0026 ; -.0023 (D5) | | | a .1: +.0000 ; +.0068 | a .1: +.0005 ; -.0043 | excess not removable |
| MOA <- MM (borrow) | +.0030/+.0029/+.0010 (P .92 min) | -.0057/-.0054/-.0029 | | | | | a -.1: -.0008 | a -.1: +.0006 | borrow passes 1 on OAI; sec 4 for crit 2 |
| LOA <- LM (borrow) | +.0067/+.0065/+.0047 (P 1.0) | -.0091/-.0091/-.0069 | | | | | a -.1: +.0039 | a -.1: -.0006 | borrow passes 1 on OAI; sec 4 for crit 2 |

**Other directions** (full list in `merge_map.out`):
- Every Syn <- X direction passes criterion 1 only as a borrow (a -.1) on OAI: +.0017 to +.0062. The causes are the OAI
  Hoffa-synovitis definition and the OAI Syn = Eff model output. On RSNA, c4 over-couples Syn with OA and LM, which is the
  opposite sign. Discarded.
- ACL <- MM and ACL <- MOA borrows likewise pass only on OAI (degenerative ACL tears). Discarded.

**Multiplicity (MODEL):**
- 72 directions x 2 a values were screened.
- LOA <- MM's minimum P of .93 is close to the .90 bar, so treat it as a weak pass.

## 4. Group compositions on OAI (n 2,399): increments on top of E-OA a .1

- Values are OA-trio sums / 12, i.e. macro.
- "inc" = increment over E-OA .1. P = bootstrap P>0, 300 resamples.

MEASURED (`box_followup.log`, `local_compose.log`).

| transform | raw (D5 / s1044 / s1045) | R1oa3 | R2m | R2m_hi | R2m_lo | gold58 c4 | gold58 R7 | R7 vs report |
|---|---|---|---|---|---|---|---|---|
| E-OA .1 | +.00155/+.00123/+.00132 | +.00041/+.00003/+.00008 | +.00035/+.00002/+.00008 | +.00012/-.00024/-.00015 | +.00063/+.00034/+.00042 | +.00142 | +.00078 | -.00060 |
| E-OA .15 | +.00224/+.00174/+.00185 | +.00048/-.00012/-.00005 | +.00039/-.00015/-.00003 | +.00002/-.00056/-.00040 | +.00084/+.00035/+.00050 | +.00160 | +.00151 | -.00108 |
| E-OA .2 | +.00286/+.00213/+.00230 | +.00043/-.00042/-.00032 | +.00034/-.00046/-.00032 | -.00018/-.00105/-.00087 | +.00097/+.00025/+.00044 | +.00239 | +.00231 | -.00170 |
| inc +LOA<-MM .1 | +.00016/+.00010/+.00006 (P .94/.84/.74) | -.00022/-.00029/-.00034 | +.00018/+.00016/+.00007 (P .98/.96/.78) | +.00006/+.00006/-.00005 | +.00039/+.00036/+.00030 | +.00129 (P .95) | +.00097 (P .94) | -.00020 |
| inc +BOR .1 (MOA+=.1 MM, LOA+=.1 LM) | +.00061/+.00059/+.00029 (P 1.0) | +.00008/+.00015/-.00017 | +.00018/+.00022/-.00005 | +.00016/+.00019/-.00006 | +.00015/+.00018/-.00010 | -.00007 (P .43) | +.00026 (P .63) | -.00013 |
| inc +LOA<-MM +BOR | +.00074/+.00068/+.00034 | -.00014/-.00014/-.00049 | +.00032/+.00035/+.00001 | +.00020/+.00023/-.00013 | +.00049/+.00049/+.00017 | +.00042 (P .67) | +.00074 | -.00026 |

**Weighted label phi for each scheme** (MEASURED):

| scheme | MOA-LOA | MOA-PF | LOA-PF | MM-LOA | ESS |
|---|---|---|---|---|---|
| raw | .07 | .07 | -.01 | .00 | 2,399 |
| R1oa3 | .36 | .39 | .23 | .09 | 1,794 |
| R2m | .36 | .35 | .22 | .04 | 1,644 |
| R2m_hi | .49 | .44 | .33 | .16 | 1,060 |

**Reads:**
- R1oa3 does not control MM-LOA: it drifts to .09 there, against .00 to .05 in all expert sources. That explains why the
  LOA <- MM increment is negative under R1oa3 and positive under R2m. R2m is the cleaner test for this pair (MODEL).
- **E-OA's transfer gain depends on the model's rho** (MEASURED, single model sets on the same knees):
  - Only the 5-fold mean D5 keeps a positive a .2 gain at gold74 coupling.
  - Both single-fold models go negative at a .2 (P>0 .01 to .06) and sit at about 0 or slightly negative at a .15
    (P>0 .14 to .40).
  - Reading for the in-flight entries (MODEL, GUESS on c4's position): eoa10 has the safer shape. eoa20 is positive only if
    c4's higher OA rho (.82 on gold58 vs D5 .76) keeps it in the D5 regime. The skeptic's "P(>0) about .70 for a .2" may be
    optimistic.

## 5. Bottom line for decisions (GUESS unless tagged)

- **No new pair or group passes all three criteria robustly.**
  - The only structural excess coupling that transfers is the OA trio. It is already taken by E-OA, at a .1 more safely
    than at a .2.
  - LOA <- MM is the single best add-on. It meets criteria 1 and 3 and criterion 2 at the gold74 point estimate, but is
    worth about +.0001 macro on top of E-OA (MODEL) and flips at +1 sd coupling.
  - The meniscus borrow is weaker: s1045 fails criterion 2.
- **If a future entry is built on E-OA:** "E-OA .1 + LOA <- MM .1" is the only composition with positive OAI-R2m, gold58
  and R7-gold58 reads. Its expected gain is inside LB noise, so it is not a reason to spend a slot (GUESS).
- **Do not chase:** PFOA <- ACL / Eff / Syn, Syn <- anything, ACL borrows. Their OAI support comes from OAI's degenerative
  cohort and from the OAI-free models' Syn = Eff output, and it reverses or vanishes on RSNA (MEASURED, sec 2-3).
