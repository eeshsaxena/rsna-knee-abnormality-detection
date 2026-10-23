# mechanics.md: E-OA ensemble mechanics (E1 interaction, placement, a vs member count). RESEARCH + MEASUREMENT ONLY.

Nothing built, pushed, committed or submitted. No dataset, no scheduled task, no GPU job. PREREG, HANDOFF, PROGRESS and
session_2026-10-19/a20 untouched (read only). Box runs are read-only CPU python piped over ssh stdin, print aggregates only,
write nothing on the box. Tags: MEASURED (computed in this session, file named), CLAIMED (older file, named), MODEL (arithmetic
on stated assumptions), GUESS (judgement).

## 0. Pre-registration (written 2026-10-20 ~01:00 IST, before any number of this run was seen)

**Question.** Does any change to the ensemble mechanics of E-OA beat the in-flight shape (c4 + E-OA on the FINAL output,
exponent 1, a .1 = eoa10 and a .2 = eoa20) on an independent ruler by more than its paired sd?

**Independent ruler (primary).** OAI, n ~2,399 knees, label proxy P_FT (MOAKS full-thickness cartilage per compartment,
as oai_eoa_check.py / box_curve.py). Weighting R1oa3 (3-label OA cell matched to gold74) and R3p (pairwise gold74 log OR,
OAI marginals); R3p_hi (+1 sd log OR) as stress; raw as information. 400 weighted resamples per config and scheme, paired
across columns.

**Stand-in members (OAI-free, report-trained).** 7 single models: s1043 folds f0..f4, s1044 (f0 mix50), s1045 (f0 288).
- ROT1: R6 := prob-mean(f0, f1, f2); r384 := s1044; r224 := s1045; S6fall := prob-mean(f3, f4)
- ROT2: R6 := s1044; r384 := prob-mean(f0, f1, f2); r224 := prob-mean(f3, f4); S6fall := s1045
- ROT3: R6 := s1045; r384 := prob-mean(f0, f1, f2); r224 := prob-mean(f3, f4); S6fall := s1044
- FLAT3: equal rank-mean of D5 mean file, s1044, s1045 (the task's "3 OAI-free models as a 3-member blend")
The nested configs copy c4 exactly: rr = .75 pct(r384) + .25 pct(r224); V = w_t rr + (1 - w_t) pct(R6) with c4's table
(Medial OA .5, Lateral OA .2, PF OA .5, Synovitis .8, Baker's .5, Lateral Meniscus .4, else .5); final = .8 pct(V) + .2 pct(S6).

**Lens 1, E1 x E-OA.** E1 = kernel-natural pow2 (THINK_1019 sec 8.4): square pct of r384, r224, R6 before they mix, and
square rb = pct(V) and rs = pct(S6) before the .8/.2 mix; flat: mean of pct^2. Columns e in {1, 2} x a in {0, .05, .1,
.15, .2, .25, .3} with E-OA on the final. Read: E1 effect on the OA trio, E-OA gain at e 2 vs e 1 (interaction), and the
stack increment (e 2, a) - (e 1, a) at a .1 / .2. E1 on the 6 non-OA OAI labels (ACL, MM, LM, Effusion, Synovitis,
Baker's; B1 binarisation as box_coupling) under raw OAI with a plain bootstrap. Report-OOF ruler (local): R7 OOF + s1044 +
s1045 OOF, readers v4 / v2 / sol56, c4-shaped 3-member nest (V = w pct(B) + (1 - w) pct(R7); final = .8 pct(V) + .2
pct(C), and B / C swapped), all 12 labels.

**Lens 2, placement.** FINAL (in flight) vs V224 (E-OA on pct(V) before the S6fall mix) vs MEMBER (E-OA on each member's
pct before any mix; flat: before the mean), each at a in {.05, ..., .3}, e 1. Same rulers. gold58 c4 replay (real
members) as a code check only.

**Lens 3, a* vs member count.** Every subset of the 7 single models of size k in {1, 2, 3, 4, 5, 7} (120 flat rank-mean
blends). Per blend under R1oa3 and R3p: model rho, q, gap = rho - q (weighted normal scores), weighted-AUC E-OA curve on
a in {0, .025, ..., .4}, a* = argmax. Summaries by k, same-recipe (folds only) vs mixed, regression of a* on gap and k.
Bootstrap (100) of the per-k mean of d(.2) - d(.1).

**Decision rule.** A mechanics change M is BUILD-READY only if, vs the in-flight shape at the same nominal a (FINAL, e 1,
a .1 and separately a .2), on OAI P_FT its paired increment is > 1 paired boot sd under BOTH R1oa3 and R3p in ALL 4 configs,
it is not worse than the in-flight shape under R3p_hi, gold58 c4 does not clearly oppose it (P>0 >= .10), and for E1-
containing M the 6 non-OA OAI labels' increment is >= 0 in all 4 configs. Otherwise: no build; report as a map. A different a
(lens 3) is never a build here (a .1 / .2 are in flight); lens 3 only informs which eoa read to prefer.

**Expectations stated in advance (GUESS).** Placement differences are tiny (E-OA is near-linear in rank space); V224-only
acts like a weaker FINAL (S6fall's OA columns untouched); E1 x E-OA interaction near 0; a* rises with k for same-recipe
folds.

## 1. Verdict (plain language)

1. **No build-ready recommendation.** No mechanics change beats the in-flight shape (FINAL E-OA, exponent 1, a .1 / .2) on
   the independent OAI ruler by more than its sd in every stand-in config. The in-flight eoa10 / eoa20 stay as they are.
2. **E1 x E-OA (lens 1): additive, interaction ~0.** pct^2 before averaging does not change what E-OA gains: |interaction|
   <= .000018 macro (sd ~.000015) in all 32 OAI cells, and -.00004 to +.00001 on the report OOF ruler (MEASURED). So an
   E1 + E-OA stack is worth exactly E1's own effect, which OAI cannot test (stand-in members are too alike, sec 2.3). E1 stays
   where THINK_1019 left it: a report-ruler-only, near-zero-mean probe. Not a build.
3. **Placement (lens 2): a reparametrisation of a, not a new mechanism.** MEMBER (per member before blending) = FINAL within
   +-.00004 at every a in .05-.2. V224 (before the S6fall mix) = FINAL at ~0.8 a, because S6fall's 20 % share stays
   undecorrelated. The best-of-curve envelopes agree within .00002 in all 4 configs (MEASURED). V224 at a .2 beats FINAL at a
   .2 (5 of 6 nested cells) only because it is effectively a ~.16, i.e. the eoa15 hedge that EX_1019 C1 already rated
   INCONCLUSIVE.
4. **a* vs member count (lens 3): yes, but only through the coupling gap.** Blending raises the model's OA coupling more than
   q, so a* rises from ~.10 (single model) to ~.13-.15 (3-7 member blends) at gold74-matched coupling; member count adds
   nothing once the gap is known (MEASURED, 120 blends). The rise saturates by k ~3. For c4 (gold58 gap ~.10, above every
   OAI-free blend) the gap map puts a* near .2 and gives P(eoa20 beats eoa10 on hidden) ~.6 (MODEL). That is a mild prior
   for the read, not a reason to build; R1 (API best-id) still decides.

## 2. Lens 1: E1 soft-OR x E-OA (MEASURED: box_mech_L12.log, summ_mech.out, local_mech.log, local_e1_diversity.log)

### 2.1 OAI expert ruler (OA trio, P_FT; 4 stand-in configs, 400 paired resamples)

| config | blend gap e1 / e2 (R1oa3) | E1 alone, OA trio macro (R1oa3 / R3p) | interaction a .1 (R1oa3 / R3p) | interaction a .2 | stack (E1+E-OA) - (E-OA), a .1 (R1oa3 / R3p) | stack, a .2 |
|---|---|---|---|---|---|---|
| ROT1 | .075 / .074 | +.00002 / +.00002 | +.000001 / +.000000 | -.000004 / -.000004 | +.00002 / +.00002 (sd .00002) | +.00001 / +.00002 |
| ROT2 | .069 / .067 | +.00001 / +.00000 | -.000002 / -.000003 | -.000014 / -.000012 | +.00001 / +.00000 (sd .00003) | -.00000 / -.00001 |
| ROT3 | .063 / .063 | +.00000 / -.00000 | -.000001 / +.000002 | -.000013 / -.000008 | +.00000 / +.00000 | -.00001 / -.00001 |
| FLAT3 | .068 / .067 | +.00001 / +.00001 | -.000005 / -.000003 | -.000007 / -.000007 | +.00001 / +.00001 | +.00000 / +.00000 |

- Interaction sd is ~.000013-.000017 in every cell; none is distinguishable from 0. Under R3p_hi and raw it is the same.
- Decision rule: stack beats E-OA by > 1 sd in 1 of 8 cells at a .1 and 0 of 8 at a .2. **FAIL.**
- E1 on the 6 non-OA OAI labels (raw OAI, plain bootstrap): -.00001 (sd .00005), -.00002 (.00007), -.00000 (.00007),
  +.00004 (.00008) macro for ROT1 / ROT2 / ROT3 / FLAT3. Per label every value is within +-.0005 AUC with sd up to .0009
  (Baker's, n 653).
- E1 does not move the blend's OA coupling (gap e2 - e1 = -.000 to -.002 on OAI; gold58 c4 OA Spearman .828 vs .827 with
  E1). That is why the interaction is ~0: E-OA acts on the coupling, E1 does not change it.

### 2.2 Report-OOF ruler (R7 OOF + s1044 + s1045 OOF, n 3,467, 12 labels; 300 paired resamples)

NEST_B = s1044 in the Raptor slot and s1045 in the S6fall slot; NEST_C swaps them; FLAT3 = equal mean.

| reader / shape | E1 all labels | E-OA gain a .1 (e1 / e2) | E-OA gain a .2 (e1 / e2) | interaction a .1 / a .2 | stack a .1 / a .2 |
|---|---|---|---|---|---|
| v4 NEST_B | +.00033 (P .99) | +.00003 / +.00002 | -.00028 / -.00030 | -.00001 / -.00002 | +.00032 / +.00031 |
| v4 NEST_C | +.00027 (P .96) | +.00001 / +.00001 | -.00029 / -.00030 | -.00000 / -.00002 | +.00027 / +.00025 |
| v4 FLAT3 | +.00028 (P .98) | +.00006 / +.00007 | -.00018 / -.00019 | +.00001 / -.00001 | +.00029 / +.00027 |
| v2 NEST_B | +.00046 (P .99) | -.00002 / -.00003 | -.00035 / -.00037 | -.00001 / -.00002 | +.00045 / +.00044 |
| v2 NEST_C | +.00043 (P 1.0) | -.00004 / -.00005 | -.00036 / -.00040 | -.00001 / -.00004 (P .03) | +.00042 / +.00039 |
| v2 FLAT3 | +.00025 (P .95) | +.00003 / +.00004 | -.00021 / -.00023 | +.00001 / -.00002 | +.00025 / +.00023 |
| sol56 NEST_B | +.00030 (P .97) | +.00006 / +.00005 | -.00021 / -.00023 | -.00001 / -.00002 | +.00029 / +.00027 |
| sol56 NEST_C | +.00024 (P .94) | +.00004 / +.00004 | -.00021 / -.00023 | -.00000 / -.00002 | +.00024 / +.00022 |
| sol56 FLAT3 | +.00026 (P .96) | +.00009 / +.00010 | -.00011 / -.00012 | +.00001 / -.00002 | +.00026 / +.00024 |

- E-OA mechanism control holds in direction: on report labels a .2 loses in all 9 cells (-.00011 to -.00040); a .1 is ~0.
  (The single-model R7 figure, -.00060 at a .1 in EX_1019, is more negative; three-member blends carry more excess
  coupling, as lens 3 shows.)
- Interaction is -.00004 to +.00001 (sd .00001-.00002): E1 very slightly weakens E-OA on report labels, a wash.
- The stack's report-ruler gain is entirely E1's main effect (+.00024 to +.00046), the same report-only evidence that
  THINK_1019 sec 8.2 already DOWNGRADED.

### 2.3 Why OAI cannot test E1's main effect (diversity control, MEASURED)

| member set | mean per-label Spearman between members | E1 (flat pow2 vs exp1) |
|---|---|---|
| R7 + s1044 + s1045 (cross-family), report OOF | R7-s44 .787, R7-s45 .786 | +.00028 / +.00025 / +.00026 (v4 / v2 / sol56) |
| R7 + s1044, report OOF | .787 | +.00039 / +.00060 / +.00035 |
| s1044 + s1045 (same S6 family), report OOF | .972 | +.00004 / -.00004 / +.00006 |
| OAI stand-ins (box_diversity.log, OAI knees) | fold-fold .924; f0-s1044 .939; s1044-s1045 .915; D5-s1044 .950 | OAI OA trio +.00000 to +.00003 |
| c4 real members on gold58 (local_c4_diversity.log, n 58) | R6-S6 .879, R6-r384 .872, S6-r384 .880, r384-r224 .944 (R7-s43 on the same 58: .859) | gold58 +.00009 (P .55) |

- E1 needs cross-family member disagreement. The OAI-free stand-ins are one family (S6-recipe seeds and folds), so the OAI
  null for E1 is expected and says nothing about c4, whose members are cross-family like the report ruler's.
- So the only independent-ruler content in lens 1 is the interaction (~0). E1's own value is still the THINK prior
  N(+.5 to 1e-4, 3e-4) (CLAIMED THINK_1019 sec 8.2), neither raised nor lowered by this run.

### 2.4 gold58 code check (real c4 members; NOT a ruler)

- Parity: c4 .94102 and E1 kernel-natural .94111, both exact (CLAIMED targets THINK_1019 sec 8.4; MEASURED here).
- FINAL E-OA a .1 / .2: +.00142 / +.00239, the float64 numbers of the a20 gate (CLAIMED ex_r6eoa_gate.log kernel path
  +.00142 / +.00247).
- E1 + E-OA: a .1 +.00118 (interaction -.00032, sd .00043, P .12; stack vs eoa10 all labels -.00023, P .39); a .2 +.00248
  (interaction .00000; stack vs eoa20 +.00009, P .55). Nothing to read at n 58.

## 3. Lens 2: placement (MEASURED: box_mech_L12.log, summ_mech.out, local_mech.log)

### 3.1 OAI, increment over FINAL at the same nominal a (macro; R1oa3 / R3p; paired sd .00002-.00005)

| config | MEMBER a .1 | MEMBER a .2 | V224 a .1 | V224 a .2 | envelope max FINAL / V224 / MEMBER (R1oa3) | best a FINAL / V224 / MEMBER |
|---|---|---|---|---|---|---|
| ROT1 | +.00001 / +.00001 | +.00003 / +.00003 | -.00001 / -.00001 | +.00004 / +.00006 | .00034 / .00036 / .00035 | .15 / .15 / .15 |
| ROT2 | +.00001 / +.00000 | +.00001 / +.00001 | -.00002 / -.00002 | +.00008 / +.00010 | .00025 / .00025 / .00026 | .10 / .15 / .15 |
| ROT3 | -.00000 / -.00000 | +.00004 / +.00004 | -.00004 / -.00003 | +.00014 / +.00017 | .00024 / .00026 / .00024 | .10 / .15 / .10 |
| FLAT3 | +.00001 / +.00001 | +.00003 / +.00002 | n/a | n/a | .00026 / n/a / .00027 | .10 / n/a / .10 |

- Decision rule at a .1: MEMBER 0 of 8, V224 0 of 6 cells > sd. **FAIL.**
- At a .2: MEMBER 5 of 8, V224 5 of 6 (ROT1 R1oa3 +.00004 = its sd). **FAIL** by the rule, and the reason is not a
  mechanism: these stand-ins peak at a .10-.15, so anything weaker than a .2 wins there. Raw-OAI curves show the strength
  mapping directly (ROT1 raw): V224 .10 +.00123 vs FINAL .05 / .10 +.00076 / +.00150, so V224 .1 = FINAL ~.08; V224 .20
  +.00230 vs FINAL .15 / .20 +.00216 / +.00274, so V224 .2 = FINAL ~.16.
- Envelope differences (best a of each placement): +.00000 to +.00002. Under R3p_hi every placement loses at every a >= .1;
  the weaker ones lose less, again the a effect.
- Report OOF: same picture. V224 / MEMBER beat FINAL at a .2 by +.00007 to +.00016 on report labels (less E-OA, less loss);
  MEMBER is +.00004 to +.00006 at a .1 in the nested shapes (slightly weaker than FINAL with cross-family members).
- gold58 (real members, code check): V224 vs FINAL +.00008 (a .1) / +.00010 (a .2), P .54; MEMBER -.00035 (a .1, P .31) /
  +.00120 (a .2, P .89). n 58 noise; gold58 is not a selection ruler.

### 3.2 Reading

Where the transform sits only rescales its strength. If a different strength were wanted, the clean way is a different a on
the FINAL output (already in flight as .1 and .2), not a moved insertion point. A20's "no retry, no a .3, no stacking" holds.

## 4. Lens 3: a* vs number of blended members (MEASURED: box_mech_L3.log)

120 flat rank-mean blends of 7 single OAI-free models; exact weighted AUC for the curves; a* by parabolic refinement.

| k | blends | R1oa3 gap | R1oa3 a* | R1oa3 d.2 - d.1 | R3p gap | R3p a* | R3p d.2 - d.1 |
|---|---|---|---|---|---|---|---|
| 1 | 7 | .063 | .109 | -.00020 | .064 | .096 | -.00025 |
| 2 | 21 | .070 | .130 | -.00010 | .071 | .120 | -.00015 |
| 3 | 35 | .072 | .135 | -.00007 | .073 | .124 | -.00013 |
| 4 | 35 | .074 | .140 | -.00005 | .075 | .127 | -.00011 |
| 5 | 21 | .074 | .143 | -.00005 | .075 | .129 | -.00010 |
| 7 | 1 | .075 | .146 | -.00001 | .076 | .133 | -.00008 |
| 5 folds only (R6-like: one recipe) | 1 | .078 | .155 | +.00004 | .079 | .146 | -.00002 |

- Paired bootstrap: the k 7 minus k 1 shift in d.2 - d.1 is +.00019 (sd .00003) under R1oa3 and +.00017 (sd .00003) under
  R3p. Blending does move the optimum up, and the move is real.
- It is all coupling: OLS a* = -.102 + 3.34 gap - .0009 k (R1oa3), -.096 + 3.06 gap - .0010 k (R3p); corr(gap, a*) .80 / .79.
  Member count adds nothing once the gap is known. R1oa3, mixed blends: rho .732 (k 1) to .779 (k 7), q .678 to .703;
  folds only: rho .777 to .790 (k 5), q .711 to .712. rho climbs, q barely moves, as skeptic_eoa.md sec 2 argued.
- d.2 - d.1 = -.00123 + .0160 gap (R1oa3), zero at gap .077; R3p zero at .082. Under R3p_hi (+1 sd coupling) both a lose and
  a* falls to .01-.04 at every k.
- Same-recipe folds couple more than mixed recipes at equal k (gap .067-.078 vs .054-.075), so a 5-fold single-recipe
  ensemble like R6 sits at the top of the OAI-free range.
- The c4-shaped nested stand-ins (sec 2.1) have gaps .063-.075 and best a .10-.15, the same as flat blends of the same
  members: the nesting itself adds no coupling.

**Transfer to c4 (MODEL).**
- c4's gold58 gap is .10 (rho .816, q .714; CLAIMED skeptic_eoa.md sec 2), above every OAI-free blend here (max .079). Its
  members' gold58 gaps are R6 .09, S6fall .13, r384 / r224 .05 (CLAIMED), so the c4 blend sits ~.02 above its OA-weighted
  member mean (.4 x .09 + .4 x .05 + .2 x .13 = .082, my arithmetic). That ~.01-.02 blending lift matches lens 3.
- With the skeptic's hidden gap g ~ N(.096, .069): a* point ~.19-.22 from the OLS (an extrapolation just past the measured
  gap range), sd ~.2; E[d.2 - d.1] ~ +.0002 to +.0003 macro, sd ~.001; P(eoa20 > eoa10) = P(g > .077-.082) ~ .58-.61.
- Same conclusion as EX_1019 sec 4 (P .52-.76), with a narrower range: a mild lean to eoa20, essentially a coin flip, and
  a .2 is the one that turns negative if hidden coupling is ~1 sd above gold58 (R3p_hi). Follow the API best-id.
- Side note for EX_1019 sec 3.2 (R6 + E-OA contingency, not recommended there): R6 is a single-recipe 5-fold ensemble; its
  OAI analogue (5 folds) has a* .15-.16, and R6's gold58 gap .09 points to ~.18-.2. Nothing changes; it remains Eesh's call.

## 5. Decision against the pre-registered rule

| candidate M | a .1: cells > sd (OAI R1oa3 + R3p) | a .2: cells > sd | R3p_hi vs in-flight | gold58 | non-OA (E1) | result |
|---|---|---|---|---|---|---|
| E1 + E-OA (stack) | 1 / 8 | 0 / 8 | -.00002 to +.00002 | P .39 (a .1), .55 (a .2) | -.00002 to +.00004 | FAIL |
| MEMBER placement | 0 / 8 | 5 / 8 | not worse | P .31 / .89 | n/a | FAIL (a effect at .2) |
| V224 placement | 0 / 6 | 5 / 6 | not worse | P .54 / .54 | n/a | FAIL (a effect at .2) |
| different a by member count | not a build by rule | | | | | MAP ONLY |

Nothing is build-ready, so no kernel copy was built in more/. The EX_1019 sec 3.4 slot plan is unchanged.

## 6. Files (all in session_2026-10-20/more/; everything else read only)

- box_mech.py (L12 / L3 jobs; read-only, aggregates only, piped over ssh stdin), box_mech_L12.log, box_mech_L12_smoke.log,
  box_mech_L3.log; box_diversity.py + .log (member Spearman on OAI, aggregates only).
- local_mech.py + .log (report-OOF ruler for lenses 1 / 2, gold58 c4 parity and placements); local_e1_diversity.py + .log;
  local_c4_diversity.py + .log; summ_mech.py + summ_mech.out (summary of the L12 JSON).
- Box use: 4 CPU-only python runs of at most ~3.5 min each; GPU idle (nvidia-smi 0 %, 5 MiB) before the first run; nothing
  written on the box; no OAI id, row or per-knee value left it.
