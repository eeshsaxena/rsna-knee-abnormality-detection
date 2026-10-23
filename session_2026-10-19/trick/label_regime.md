# Label regime: expert (gold58) vs report-derived labels, and the transforms it suggests (10-19 ~21:45 IST)

RESEARCH + CPU MEASUREMENT ONLY. Nothing built, pushed, committed, submitted or created. No GPU, training or box job ran; the box
was not touched. No PREREG / HANDOFF / PROGRESS / scheduled-task edits. Nothing public was pulled.
Scripts and logs: `session_2026-10-19/trick/regime_work/` (m1 ... m15, r6_perfold). All read existing local files only
(gold74_eval_set.csv, R6_gold58_serve.csv, the r6g58_box dumps, gold58 r384 / r224 / S6fall / s43 serves, R7OOF.csv, our S6 seed OOF,
the local LLM label tables, the c4 notebook and the R6 bundle json files).
Tags: MEASURED (computed here, script named), CLAIMED (a note or third party says so), MODEL (arithmetic on stated assumptions),
GUESS (judgement).

## 0. Verdict

**PROMISING, one candidate: "OA decorrelation" (E-OA).** Final c4 output, for each OA label
`out_t = pct( pct(f_t) - a * (pct(f_o1) + pct(f_o2)) / 2 )`, o1 / o2 = the other two OA labels, other 9 columns untouched,
0 runtime, one block in cell 18. Pre-register a = .2 (a = .1 is the conservative twin; run one, not both).

Why it is different from the killed cross-label ideas (E2, pairscan, co-occurrence stacking):
1. Mechanism stated by the HOST, not by us: the rubric scores each compartment separately at a strict high-grade >= 1 cm
   threshold (733343), and the host confirmed reports call lateral PATELLAR FACET chondrosis "lateral compartment" (733491).
   Report-trained members therefore inherit a report-only coupling between the three OA labels.
2. The coupling gap is measured on the SAME 58 studies: reports co-call Lateral OA with PF OA far more than the experts
   (phi .59-.73 vs .28; paired bootstrap gap +.31, P .991), and Medial OA with Lateral OA (+.30, P .962). MEASURED m6, m8.
3. Same studies, same c4 predictions, two label sets: OA decor gains +.0014 (a .1) / +.0024 (a .2) macro under the expert labels
   and ~0 / -.0002 to -.0005 under 4 report readers. The gain is a label-regime effect, not sample composition. MEASURED m15.
4. Report-regime OOF (non-gold train, n 2,638-4,333): it LOSES in all 9 reader x member cells (-.0001 to -.0016). The sign
   flips between regimes exactly as the mechanism predicts. MEASURED m10.
5. Every gold58 prediction set gains (7 of 7 at a .1 and .2: c4, R6, r384, r224, S6fall, R7 OOF, s43). MEASURED m10.
6. Specific: beats all 36 "minus a random pair of non-OA labels" nulls at a .1 and .2; within-group decor of menisci,
   ligaments, fluid and bone does not show it. MEASURED m11.
7. Survives nested selection: fixed decor outer-half +.00063 (a .1, P>0 .79, P<0 .02); OA gap-rule with sign and helper chosen
   on the inner half +.00045 to +.00048; FULLY nested (group, sign and a all chosen on the inner half from 5 groups x 2 signs x 2
   a) +.00031, picking OA-decor in 1,268 of 2,000 inner halves. MEASURED m9, m13.
8. Robust inside gold58: c4 bootstrap a .1 +.00062 (sd .00040, P>0 .94), a .2 +.00175 (sd .00103, P .97); split-half both halves
   > 0 in 57.6 %, opposite signs 3.7 %; no single study's removal takes it below +.0010. MEASURED m10.

Honest caveats:
- The ONLY expert-regime ruler is gold58 (58 studies; OA positives 15 / 11 / 21). gold58 screens with P .85-.93 have read
  negative on the LB before (S6fall .3, PF OA .1). Those were member re-weights; this is a regime transform whose sign is
  predicted to flip, but the magnitude still rests on 58 studies.
- Partly post hoc: the brief named "OA labels with each other"; I first tested positive-weight OA borrowing (m5, negative), saw
  the coupling gaps (m8), then tested decorrelation. The fully nested m13 (+.00031) is the number that charges for that.
- If the hidden labels behaved like report labels the change would cost about -.0004 (a .1) / -.0013 (a .2) macro
  (MEASURED m12, report-regime mean). The host says the test set used the same image-review process as the 58 sample labels
  (CLAIMED 733491), so that branch has low prior (GUESS P ~.1).

Expected value (MODEL): bootstrap +.00175 (a .2) shrunk 55-90 % for the 58-study ruler and the post-hoc steps gives
+2e-4 to +8e-4 private; with P(report-like) ~.1 at -.0013, E ~ +3e-4 to +6e-4. P(true private gain > 0) ~.65-.72 (MODEL / GUESS).
For a .1: E ~ +1e-4 to +3e-4, P ~.65-.70.

Read power (MEASURED m12, paired subsample sd of this exact change): public (390) sd .00021 at a .1, .00042 at a .2; private
(910) .00012 / .00027. So the read is informative both ways: expert-regime truth near +.0008 to +.0018 vs report-regime truth
-.0013 at a .2 are ~5-7 sd apart. A public delta <= -.0008 would falsify the OA-regime premise.

## 1. Facts that correct the brief

- **train.csv carries labels for exactly 58 studies, and they are the gold58 studies (bit-identical to gold74's
  original_gold58 rows); the other 4,349 have a report and blank labels.** MEASURED (pandas on _data/train.csv). So there are
  no official report-derived train labels: every member learned third-party LLM report labels (v4_blend, sol56, JEV, Gemini,
  heliosli's opusv11b states), and gold58 IS the host's "sample labels" from image review (CLAIMED 733491).
- **"Experts call fluid positive more than reports" is false for Effusion and true for Synovitis.** Effusion: expert .60 vs
  report .86-.88 on the same 58 studies; report specificity vs expert only .30 (rubric = moderate or large, reports call small
  effusions). Synovitis: expert .47 vs report .28-.29, report sensitivity .48. MEASURED m1.
- **gold74's 16 extra rows are our own report reads under the gold convention (notes like "'mild synovitis' = 0 per gold-58
  convention"), not expert image labels.** MEASURED (gold74_eval_set.csv `note` column). Only gold58 is expert regime.

## 2. Regime table (MEASURED m1_regime.log; 58 gold studies; report reader binarized at >= .5)

| label | expert + (prev) | report prev (v2 / v4 / sol56 / JEV) | report sens / spec vs expert (v4) | reader AUC vs expert (v4 / JEV*) | c4 gold58 AUC | direction |
|---|---|---|---|---|---|---|
| ACL | 24 (.41) | .55 / .50 / .50 / .55 | .96 / .82 | .987 / .999 | .989 | expert stricter (>50 % fibres) |
| MCL | 9 (.16) | .31 / .28 / .28 / .31 | 1.00 / .86 | .968 / .981 | .968 | expert stricter (high-grade tear) |
| Medial Meniscus | 26 (.45) | .55 / .48 / .50 / .48 | .88 / .84 | .948 / .947 | .977 | ~same |
| Lateral Meniscus | 23 (.40) | .52 / .38 / .38 / .38 | .78 / .89 | .879 / .872 | .928 | ~same |
| Medial OA | 15 (.26) | .38 / .36 / .36 / .36 | .93 / .84 | .932 / .930 | .989 | expert stricter |
| Lateral OA | 11 (.19) | .28 / .29 / .29 / .29 | .82 / .83 | .833 / .855 | .843 | expert stricter + compartment-specific |
| PF OA | 21 (.36) | .41 / .43 / .43 / .43 | .81 / .78 | .902 / .902 | .901 | expert stricter |
| Effusion | 35 (.60) | .86 / .88 / .88 / .88 | 1.00 / .30 | .877 / .877 | .981 | expert MUCH stricter (moderate+) |
| Synovitis | 27 (.47) | .28 / .29 / .29 / .29 | .48 / .87 | .790 / .785 | .840 | expert broader |
| Baker's | 12 (.21) | .28 / .28 / .28 / .28 | .92 / .89 | .944 / .920 | .981 | expert stricter (moderate+) |
| Contusion | 19 (.33) | .62 / .48 / .48 / .47 | .84 / .69 | .860 / .870 | .958 | expert stricter (impact edema only) |
| Fracture | 18 (.31) | .33 / .36 / .36 / .33 | .78 / .82 | .793 / .904 | .936 | ~same prevalence, poor agreement |

*JEV HYBRID weights were fit on 67 gold studies (CLAIMED JEV README via session_2026-10-05/jump2/verify_labels.md l.50), so its
gold58 AUC is in-sample. Macro reader AUC vs expert: v2 .864, v4 .893, full .878, sol56 .835, JEV .904*, blendjev_sol .904*;
members: R6 .937, r384 .923, r224 .915, S6fall .931, c4 .941. Spearman(per-label reader gap, c4 gap) .53. MEASURED m1.
Most c4 gold58 errors on Lateral OA are studies whose REPORT says Lateral OA (v4 .91-.97) and the experts say no, 3 of the top 4
with expert PF OA = 1 (MEASURED m4_errors.log): the members faithfully learned the report label.

## 3. Transforms tested (all nested or pre-registered as stated)

| transform | gold58 (expert) | nested / robustness | report-regime control | verdict |
|---|---|---|---|---|
| **OA decorrelation (each OA label minus a x mean pct of the other two)** | c4 +.00142 (a .1), +.00239 (a .2); 7 / 7 members > 0 | boot P .94 / .97; fully nested +.00031; fixed +.00063 (P .79); beats 36 / 36 nulls; LOO min +.0010 | 9 / 9 negative (-.0001 to -.0016); same-study report labels ~0 / negative | **PROMISING** (sec 0) |
| Lateral OA <- -PF OA only (host facet mechanism) | c4 +.0077 on the label (+.00064 macro) at a .1; 6 / 7 sets > 0 (r384 -) | boot P .68; split-half sign agreement .19 (few pairs); rank 7 of 132 negative-pair nulls | 9 / 9 negative | WEAK alone; subsumed by OA decor |
| 13-pair radiologist family (Syn<-Eff, Eff<-Syn, Eff<-Baker, Baker<-Eff, Cont<-ACL, Cont<-Frac, Frac<-Cont, ACL<-Cont, MOA<-MM, LOA<-LM, LOA<- -PF, MOA<-PF, PF<-MOA), a .05-.2 | best in-sample Syn<-Eff .2 (+.0084 label) | nested per-target pick -.00050 (P .32); shared a -.00053; fixed a .1 -.00073 | n/a | DEAD (MEASURED m5) |
| label-level coupling-gap theory on all 132 pairs (inner half picks sign + helper from report vs expert phi, no prediction peeking) | in-sample +.0012 to +.0021 | nested -.00011 to -.00097 for every reader / thr / a | n/a | DEAD except inside the OA trio (MEASURED m7) |
| other within-group decor (menisci, ligaments, fluid, bone) | +.0006 / +.0003 / -.0005 / ~0 at a .1 | not specific | mostly negative | DEAD (MEASURED m11) |
| E2 Synovitis <- Effusion .1 | fixed outer-half +.00037 (P .72) but E2 was picked on the full gold58 earlier | cannot be nested any more | -.0003 / -.0014 (THINK) | stays KILLED |
| Rubric-mapped STATE score (heliosli state heads: e.g. Effusion = P(moderate) + P(large) + .5 P(small-to-moderate / unspecified)), weights written from the rubric before any number | R7 alone: rubric -.0094 (P .06), .5 / .5 mix +.0011 (P .62); STRICT-label mix +.0047 per label (Effusion +.0155, MCL +.0147, PF OA +.0116; Contusion, Fracture, Synovitis negative) | inside c4 via R7 proxy: +.00039 (boot -.00028, P .45); delta transferred onto R6 +.00087 (boot +.00063, sd .0022, P .63), split-half sign agreement .40 | rubric score loses on report labels (Effusion -.033 to -.056, Baker's -.036 to -.068): regime fingerprint on Effusion and MCL | WEAK; buildable without a worker patch (sec 5) but the evidence is noise-level (MEASURED m2, m3, m3b) |

Note on heliosli's own state stacker (MEASURED session_2026-10-15/r6v1/raw/state_stacker.json): it was fitted and judged on
REPORT-regime rulers ("explicit" / "corrected" opusv11b targets) and adopted only Baker's (coefficients load on moderate / large /
ruptured, i.e. already rubric-like; R6 Baker's gold58 .969). Its non-adoption of the other classes says nothing about the expert
regime.

## 4. S6fall, JEV labels, and an R6 fine-tune tonight

- **How S6fall's labels were made** (CLAIMED, nartaa README + RAPTOR_PORT_PLAN.md l.96 + verify_labels.md l.50 / l.75):
  heliosli S6 architecture from ImageNet-22k, 4,263 competition studies, gold58 excluded from gradients, on "JEV-blend" soft
  labels = mean(blend, JEV HYBRID), blend = mean(nartaa's private Gemini 3 Flash soft labels, stevenleehans v4_blend)
  (Gemini alone where v4 = .25 "not addressed"). JEV HYBRID (riadmohamed42) = 6-LLM ensemble whose member weights were fit by
  "AUC edge over 0.5 on 67 gold studies", with a plain-statement convention that matches gold58 (small effusion = 0, suspected = 0).
  So "gold-calibrated" means per-label LLM-ensemble weights tuned on the gold studies, NOT a prevalence or regime transform:
  JEV still calls Effusion positive on .88 of gold58 (expert .60) and Synovitis on .29 (expert .47). MEASURED m1.
  nartaa also ranked SWA epochs on gold58 (CLAIMED RESEARCH_1017 sec (b)), so S6fall's gold58 .931 is selection-inflated.
- **R6 single folds on gold58: .926 / .934 / .934 / .926 (mean .930); 4-fold mean z .938; served R6 .937.** MEASURED
  regime_work/r6_perfold.log (run_b dump, all 58). The public B-leg R6 fine-tunes (R6 warm start, 2 epochs on gold-calibrated
  soft-v2, gold58 inside the pixel cache) quote single-fold gold .911 / .927 (CLAIMED models.md row 7): no better than R6's own
  folds even with gold partly in-sample.
- Our JEV-style S6 retrains s43 / s44 read .921 / .915 vs nartaa .931 (MEASURED earlier, HANDOFF).
- **Recommendation: do not train tonight.** Realistic cost: 4 folds x ~1-1.5 h V100 for a 2-epoch warm start (GUESS from the KC
  all-data 3.4 h, PROGRESS_16) + ~2 h build / gate + ~6.5 h commit + ~6 h score: a read on 10-21 night at best. The only gate
  would be gold58, which is in-sample for any gold-calibrated label. Evidence that WOULD justify it (not obtainable by tonight):
  a fold-matched gold58 gain >= +.005 for an R6 fine-tune whose labels were calibrated without gold58 (e.g. on gold74's 16 extra
  rows only), plus a non-negative report-OOF read. If anything is trained after the competition, the regime finding says: train
  compartment-decoupled OA targets and moderate+ Effusion / Baker's targets, not another LLM blend.

## 5. Build and read spec for E-OA (only on Eesh's go + a PREREG amendment; nothing here was built)

- Base: c4 kernel (session_2026-10-18/a18/kernel_v224r6a25c4j, build_a18 lineage). One change, cell 18 `_s6f_blend`, after the
  S6fall blend loop and the blanked-column reset (after l.86 `out[c] = np.float32(0.5)`), before the l.87 validity check:
  `U = {c: pct(out[c].values) for c in OA}; for c in OA: o = [k for k in OA if k != c]; out[c] = pct(U[c] - A_OA * (U[o[0]] + U[o[1]]) / 2).astype(np.float32)`
  with OA = ['Medial OA', 'Lateral OA', 'PF OA'], A_OA = .2, and `rec['a_oa'] = A_OA`. pct keeps the column in (0, 1]. The canary path
  (S6fall failed) stays untouched, so a degraded run still shows the ACL canary.
- Gates (offline, CPU): A_OA = 0 gives a submission.csv bit-identical to c4 on the visible studies; the gold58 replay of
  regime_work/m14_gate.py gives macro .94341 at a .2 (.94244 at a .1; c4 .94102) with the 9 non-OA columns rank-identical.
  MEASURED m14_gate.log.
- Read by R1 vs c4. One change, never stacked with E1 or anything else; if both E-OA and E1 are wanted, E-OA goes first (stronger
  evidence) and THINK's "no second probe in parallel" rule still applies.
- If E-OA moves the best-id: A = E-OA, B stays R6 57248084 (PREREG A15). Optional and only on a clear positive read: an "R6 + OA
  decor" B variant (R6 alone gains +.0017 on gold58 at a .2, MEASURED m10) as one 10-21 entry.
- Timeline (MODEL from runtime.md latencies): build + gate tonight ~1.5 h, commit overnight (~6.5 h), submit after 05:30 IST
  10-20, read ~12:00-14:00 IST.

## 6. GPU / box measurements that would settle open points (specified, NOT run)

1. **R6 own state read-out on gold58** (for the state-head transform, sec 3): box V100, re-run
   `session_2026-10-18/next/g58/r6g58_box/scripts/r6g58_run.py` exactly as run_b, adding `--dump-logits <out>/dump.json --dump-state`
   to the worker args (compact_infer 8401812c supports it, l.595; no patch). ~15-20 min on a free V100 (run_b: 58 studies,
   852 s with an OOM-shared GPU). Output: `state_plain` per fold (use plain only; mirror needs the M/L block permutation). Then
   CPU: rubric mix on STRICT labels inside c4. Only worth a slot if it reads >= +.002 on gold58 with a report-OOF sign flip.
   Low priority.
2. **Population expert OA coupling** (would firm up E-OA's premise): box CPU only, ~5 min, on the OAI MOAKS tables that already
   live there (no data leaves the box; only 3 numbers come back): phi between per-compartment high-grade cartilage status
   (medial TF, lateral TF, PF) over the ~2,399 baseline knees, to compare with expert gold58 (.28 Lat-PF) and the report phi
   (.43-.52). Different population (OA cohort), so it is a sanity check, not a gate. Needs Eesh's OK as a box job.

## 7. Files

- regime_work/m1_regime.py/.log/.csv: regime table. m2_states.py/.log, m3_c4proxy, m3b_slot: state-head transform.
- m4_errors.log: c4's worst gold58 pairs per weak label with report text. m5_nested: 13-pair family nested.
- m6_latoa: Lateral OA <- -PF OA (label coupling, members, report-OOF control, 132-pair null).
- m7_gap_nested: coupling-gap theory nested. m8_phi_boot: same-study phi gaps. m9_oa_trio: OA gap rule nested.
- m10_oa_decor: members, bootstrap, split-half, LOO, report-OOF control. m11_null: 36 random-pair nulls + other groups.
- m12_lbnoise: public / private read sd. m13_full_nested: group + sign + a selected on the inner half.
- m14_gate: kernel-natural gate values. m15_samestudy: same studies, expert vs report labels. r6_perfold: R6 per-fold gold58.
