# Stacking / blending lens (2026-10-19 ~17:00-18:30 IST, research only)

Read-only. Nothing built, pushed, committed or submitted; no GPU, no box job. CPU numpy/sklearn only (py -3.12), each run < 5 min.
Scripts and logs: `session_2026-10-19/think/stack2/` (new, listed in sec 10). Reused partial work from the interrupted run:
`think/stack_ops.py/.json`, `think/cross_label.py/.json`, `think/tie_loss.py`, `think/work/xlabel_*.py/.json`, `think/ours_work/`.
Tags: MEASURED (our script output, file named), CLAIMED (someone else says so, source named), MODEL (our arithmetic on stated
assumptions), GUESS.

## 0. Bottom line
1. **No stacking or operator change is strong enough to bank a slot on.** The best one moves the expected public score by about +2e-4
   (MODEL), below the ~3.4e-4 gap to a displayed .959 (density.md), with P(positive) ~.6-.65 (MODEL).
2. **Only one operator family is positive on BOTH rulers: "soft-OR" rank combiners** (average of pct^2, or of -log(1 - pct)).
   - OOF ruler (R7 + our S6 seeds, 3,467-4,333 train studies, report labels): +.0003 to +.0011 macro, P>0 .95-1.0, 8-11 of 12 labels up,
     same sign under all 3 report readers and in a c4-shaped nested form (MEASURED, `stack2/oof_ops_*.json`, `nested_oof.log`).
   - gold58, c4 nested form: pow2 +.00048 (P>0 .64), invgeo +.00089 (P .73) (MEASURED, `stack2/g58_nested.log`). But split-half sign
     agreement is a coin flip (.44-.49), per-label signs agree with the OOF ruler on only 5 of 8-9 labels, and the gold58 gain sits mostly
     in Fracture (where the OOF ruler is slightly negative and the LB punished "more Raptor on Fracture", fr100).
3. **flat_logit's gold58 +.00088 (P .60) is not an operator effect.** R6's logit SD is 5-8x the other members' (MEASURED,
   `stack2/logit_scale.py`), so logit averaging silently hands R6 58-90 % of every label. A RANK blend at those effective weights reproduces
   its per-label deltas (corr .90, MEASURED `stack2/logit_equiv.log`). It is an R6-heavy re-weighting, the class the LB already rejected
   (pf10, R6 alone below c4). On the OOF ruler logit averaging loses -.0034 to -.0068 (MEASURED). **CLOSED.**
4. **The cross-label stacker's CV gain (+.0096 macro) was fitted reader noise, not knee biology.** Fit on v2 verdicts, scored on the same
   held-out rows against two other report readers (v4_blend, gpt56sol), the gain is -.0010 / -.0011 (MEASURED, `stack2/xlabel2.log`).
   Applied to R7's OWN gold58 rows (no model transfer) it is -.0059 (MEASURED). **CLOSED.**
5. **Co-occurrence priors: the label coupling is regime-specific, so the two rulers cannot agree by construction.**
   - Synovitis <- Effusion is +.0048 (a .1) / +.0084 (a .2) on gold58 for c4 and positive for 5 of 6 members (MEASURED, reused
     `work/xlabel_g58.json`, `xlabel_g58b.py`). On the OOF ruler it is negative (-.0003 / -.0014) (MEASURED, `work/xlabel_oof.py`).
   - Among all 132 target <- helper pairs on gold58 it ranks only 10th. The null spread is sd .0046 around a mean of -.0023, and
     gold58 pair deltas do not correlate with OOF pair deltas (r .05) (MEASURED, `stack2/pairscan.log`).
   - The mechanism is plausible: expert Synovitis prevalence is 47 % vs 12 % in reports, and expert P(syn | no effusion) is .22 vs .012
     in reports (MEASURED). A forum audit found 12 of 27 expert-positive Synovitis studies had no explicit Synovitis statement in the
     report (CLAIMED, forum 740610, `think/forum_raw/flat/flat_740610.txt`).
   - Weak lottery ticket only. A global "abnormality" prior is negative on both rulers (MEASURED, `stack2/severity.log`). **CLOSED.**
6. **Both rulers agree on what to avoid** (MEASURED): geometric rank mean, median / min / max, trimmed mean, probability averaging,
   equal-weight crowds and probit-rank are all <= 0 on both. z-scored logit averaging disagrees (OOF +.0005 to +.0016, gold58 -.0011 to
   -.0013), so it is not robust.
7. **The stacking lens is limited by labels, not compute.** There are 58 expert-labelled studies, and the 4,333-study OOF ruler uses a
   different label regime. Spare hidden runtime (~2.5 h) and GPU hours do not change that (sec 9).

## 1. The two rulers and why they can disagree
| ruler | studies | labels | members available | strength | weakness |
|---|---|---|---|---|---|
| gold58 | 58 | Kaggle annotation (same regime as test, CLAIMED hengck23 745861 / FORUM_DIGEST_10-12) | all c4 members (r384, r224, R6 serve, S6fall) | right regime, right members | tiny: bootstrap sd of a macro operator delta .0013-.0036; misread PF OA .1 at P .925 (HANDOFF 10-19) |
| OOF ruler (new) | 4,333 (2 members) / 3,467 (3 members) | report regime: v4_blend >= .5, gpt56sol >= .5, v2 YES/NO (UNK dropped) | R7 OOF (heliosli compact family, R6's sibling) + our S6-recipe seeds s1044 (5 folds), s1045 (4 folds) | 75x larger, tight CIs (sd 1-6e-4) | no Raptor family; labels from reports; R7 / S6 trained on report-derived labels |

- Gold studies are excluded from all S6 seed OOF files (0 of 58 present; MEASURED, `stack2/load.py` check), so the OOF ruler is clean of gold.
- v4_blend and gpt56sol behave like one reader (operator deltas match to ~1e-4; MEASURED); v2 is the semi-independent second reader.
- **Regime gap (MEASURED, `work/xlabel_gap.py`, gold58 vs v4 >= .5 on non-gold train):**

| label | gold58 prevalence | report prevalence |
|---|---|---|
| Synovitis | .47 | .12 |
| Fracture | .31 | .07 |
| Lateral Meniscus | .40 | .15 |
| ACL | .41 | .20 |
| Contusion | .33 | .17 |
| Medial OA | .26 | .37 |
| PF OA | .36 | .46 |
| (rest) | within .06 | |

- Label co-occurrence also differs, sometimes in sign. Example: MCL ~ Medial Meniscus phi is +.155 in reports and -.195 in gold58. Lateral
  Meniscus ~ Medial OA is +.109 vs +.407; Synovitis ~ Effusion is +.277 vs +.403 (MEASURED, `stack2/xlabel2.log`).
- Consequence: anything that exploits label coupling (cross-label stackers, borrowing priors) is regime-specific, so the OOF ruler cannot
  validate it. Single-label operators (how to combine members' scores for one label) are less regime-bound; that is where agreement is
  meaningful.

## 2. Operator table: where the two rulers agree
macro delta vs the rank-average base; P = paired bootstrap P(delta > 0).
- gold58 base = c4 as shipped (nested rank); "flat" rows from `think/stack_ops.json`, "nested" rows from `stack2/g58_nested.log`.
- OOF base = rank average at the same weights; R7 / s44 / s45 at .5 / .25 / .25, n 3,467 (`stack2/oof_ops_{v4,v2,sol56}.json`).

| operator | gold58 c4 | OOF v4 | OOF v2 | OOF sol56 | verdict |
|---|---|---|---|---|---|
| flat rank (vs nested c4) | -.00052 (P .35) | base | base | base | c4's nesting is fine |
| logit mean | +.00088 (P .60) | -.00361 (P 0) | -.00681 (P 0) | -.00355 (P 0) | implicit re-weighting (sec 3); CLOSED |
| probability mean | -.00382 (P .12) | -.00467 | -.00142 | -.00473 | both negative; CLOSED |
| z-scored logit mean | flat -.00112 (P .29); nested -.00130 (P .29) | +.00070 (P 1) | +.00154 (P 1) | +.00063 (P 1) | disagree; not robust |
| probit of rank (nscore) | flat -.00099; nested -.00006 | -.00006 | -.00051 | -.00004 | ~0 / negative; CLOSED |
| geometric rank mean (soft AND) | -.00220 (P .09) | -.00118 (P 0) | -.00191 (P 0) | -.00111 (P 0) | both negative; CLOSED |
| **pow2 rank mean (soft OR)** | flat -.00015; **nested +.00048 (P .64)** | **+.00039 (P 1)** | **+.00064 (P 1)** | **+.00037 (P 1)** | **same sign on both; small** |
| pow3 / pow4 | nested pow3 +.00132 (P .78); flat pow4 +.00001 | pow4 +.00011 (P .62) | +.00031 (P .79) | +.00006 | too sharp on OOF; noisy |
| **-log(1 - rank) mean (invgeo, soft OR)** | flat -.00016; **nested +.00089 (P .73)** | **+.00051 (P 1)** | **+.00059 (P .99)** | **+.00049 (P 1)** | **same sign on both; small** |
| median (3-4 members) | -.00566 (P .03) | eq3 -.00636 | -.00473 | -.00628 | both strongly negative; CLOSED |
| min / max | n/a | -.00522 / -.00202 | -.00799 / -.00368 | -.00499 / -.00211 | CLOSED |
| trimmed weighted mean | -.00164 (P .30) | n/a | n/a | n/a | CLOSED (with median) |
| equal weights (crowd) | eq4 -.00365 (P .09); eq3 families -.00043 | n/a (R7-heavy beats eq on OOF too) | | | c4 weights carry information; CLOSED |
| add LB-negative member as crowd | +R7 .15 -.00042; +S43 .15 -.00115; +R7 .25 -.00217 | | | | matches the LB negatives; CLOSED |
| Raptor slot as prob / logit mean | +.00016 / +.00012 (P .58 / .57) | | | | within-family, ~0; CLOSED |

The OOF result holds in other configurations too (MEASURED, same json files):
- 2 members at .5/.5 (n 4,333): pow2 +.00044 to +.00062 and invgeo +.00061 to +.00066 (P 1).
- At .8/.2: pow2 +.00054 to +.00120 (P 1).
- In the c4-shaped nested form (s44 + s45 first, then R7; `nested_oof.log`): pow2 +.00038 to +.00108 and invgeo +.00046 to +.00079, with
  8-11 of 12 labels up.

## 3. flat_logit (+.00088 on gold58): real or noise? Neither; it is a disguised weight change
- Logit SD per label on gold58 (MEASURED, `stack2/logit_scale.py`): r384 .7-1.9, r224 .7-2.0, S6fall .7-2.3, **R6 2.9-11.0**.
  - The heliosli compact family emits huge logits; 3-7 R6 cells per label sit at |logit| >= 15.
  - On the OOF ruler R7's logit SD is 3.4-10.6 vs s44's 1.1-2.5 (MEASURED).
- Effective weights under logit averaging = c4 flat weight x logit SD. **R6 gets .58-.90 per label** (Synovitis: c4 gives R6 .16, logit
  averaging gives .69; Lateral OA .64 -> .90).
- A rank blend at those effective weights reproduces flat_logit's per-label deltas with corr .90. Examples: Synovitis +.029 vs +.017,
  Lateral OA +.016 vs +.017, PF OA +.013 vs +.015, Fracture -.017 vs -.031, Effusion -.017 vs -.011 (MEASURED, `stack2/logit_equiv.log`).
  The residual +.0007 macro is far inside the bootstrap sd of .0036.
- The re-weighting it implies (R6 up on PF OA, Lateral OA and Synovitis) is the same direction as pf10 / c2 / c3max, which read negative
  on 10-19 (MEASURED LB, HANDOFF 17:00). **CLOSED.** z-scoring the logits first removes the re-weighting and the gold58 gain with it
  (-.0011).

## 4. Soft-OR combiners (pow2 / invgeo): the only cross-ruler survivor
- **What it does:** each member's percentile u is pushed through a convex map before the weighted average (u^2, or -log(1 - u)). A study
  that ONE member ranks near the top is pulled up more than under a plain rank mean. Disagreement at the top resolves toward the
  confident member.
  - Soft AND (geometric mean) is the mirror image and loses on both rulers.
  - Hard OR (max) and very sharp maps (pow4) lose or turn noisy on the OOF ruler, so the effect is a mild-convexity optimum, not "more
    is better" (MEASURED).
- **OOF ruler (report regime):** consistent across both readers, member counts 2 and 3, weights .5/.5, .8/.2 and .5/.25/.25, and flat
  vs nested. Bootstrap sd 1-3e-4, so it is a real property of R7 + S6-family blends against report labels (MEASURED).
  - Gains sit in ACL, Lateral Meniscus and Contusion, the low-prevalence labels in the report regime. Baker's is negative (MEASURED,
    `nested_oof.log`).
- **gold58 (expert regime, c4 members):** nested pow2 +.00048 and invgeo +.00089, but:
  - split-half sign agreement .49 / .44 (no stable signal at n 58);
  - per-label sign agreement with the OOF ruler only 5 of 8 (pow2) and 5 of 9 (invgeo);
  - Fracture alone carries +.0083 (pow2) / +.0069 (invgeo); with Fracture held at c4, pow2 is +.00028 and invgeo -.00018;
  - Baker's is negative on both rulers (-.0045 gold58) (MEASURED, `g58_nested.log`, `nested_oof.log`).
- Spearman of nested pow2 / invgeo vs c4 on gold58: .9968 / .9971, so this is a small perturbation, about the size of a table tweak
  (MEASURED).
- **Public / private noise, MEASURED on the OOF ruler** (disjoint 390 / 910 subsamples, percentiles recomputed inside each subset as the
  kernel does; `stack2/subsample_sd.log`):

| change | public-size sd | private-size sd | P(delta > 0) public / private (at the report-regime effect) |
|---|---|---|---|
| pow2 (+.00039 full ruler) | .00041 | .00025 | .76 / .93 |
| invgeo (+.00051) | .00048 | .00032 | .81 / .95 |
| z-logit (+.00070) | .00064 | .00036 | .83 / .95 |
| weight tweak R7 .5 -> .45 (-.00011) | .00030 | .00017 | .37 / .29 |

  The weight-tweak sd (3.0e-4) sits a bit above lb_private.md's MODEL 1.5-2.5e-4, so public reads of small changes are noisier than
  assumed there.
- **Transfer caveat:** the OOF ruler has no Raptor CoAtNet, and both of its members were trained on report-derived labels. The soft-OR
  gain could depend on "a confident report-trained member is usually right about report labels", which may weaken in the expert regime.
- **MODEL expectation:** effect prior mu ~ N(+2e-4, 3e-4) (OOF magnitude shrunk for regime transfer). Then (`stack2/need.log`):
  - P(public > 0) .65, P(private > 0) .70;
  - P(private > 0 | public best-id move) .79;
  - E[private delta | adopted under R1] +3e-4;
  - P(it displays .959) ~.3.

## 5. Cross-label stacking: why CV vs LLM labels gained but gold58 did not
Reused `think/cross_label.json`: a 12-logit logistic stacker per label (C .1) fitted on R7 OOF vs v2 verdicts.
- CV gain +.0096 macro: ACL +.035, Contusion +.050, Lateral Meniscus +.016, MCL +.012.
- Applied to R6 on gold58: mostly losses (Contusion -.053, MCL -.018).

New decomposition (MEASURED, `stack2/xlabel2.log`):

| test | macro gain | reading |
|---|---|---|
| (b) same held-out rows, scored vs v2 (the reader it was fit on) | **+.0096** | |
| (b) same rows, scored vs v4_blend | **-.0010** | gain does not survive a change of reader |
| (b) same rows, scored vs gpt56sol | **-.0011** | same |
| (a) R7-fit stacker on R7's OWN gold58 rows (no R7 -> R6 shift) | **-.0059** | not a model-transfer problem |

Per label:
- Contusion is +.050 under v2 but -.022 under v4 and -.068 on gold58. v2's own-score AUC on those rows is .846 vs .895 under v4, so v2's
  Contusion verdicts are noisier and the stacker learned to predict that noise from Fracture / PF OA / Medial OA logits.
- ACL is the only gain that survives other readers (+.015). gold58 has no headroom there (R7 .980, R6 .996).
- Synovitis is the only big gold58 gain (+.059 on R7's rows), driven by the Effusion (+.31) and Fracture (+.28) coefficients. That is the
  regime effect of sec 6, not a stacker virtue.

**Answer:** correlated label noise inside one LLM reader (v2's verdict errors co-vary with other findings) plus a regime mismatch.
"LLM-label leakage" in the strict sense is not needed to explain it. **CLOSED.**

## 6. Hierarchical / co-occurrence priors
- **Synovitis <- Effusion** (s' = pct(syn) + a * pct(eff)), MEASURED:
  - gold58 c4: a .1 +.0048 (P .85), a .2 +.0084 (P .81), i.e. macro +.0004 / +.0007 if real.
  - Positive for R6, R7, r384, S6fall and c4 at a .1-.2 (r224 ~0). Half-sample P(d > 0) .72-.75 (reused `work/xlabel_g58*.py`).
  - OOF ruler (all three readers): -.0003 (a .1), -.0014 (a .2). In the report regime Synovitis is narrow and effusion adds noise.
  - The 132-pair null on gold58 (every target <- helper at a .1): mean -.0023, sd .0046, 27 % positive. Syn <- Eff sits 10th, below
    implausible pairs (Lateral OA <- Effusion +.0106, Lateral Meniscus <- Synovitis +.0068). Corr of gold58 vs OOF pair deltas .05
    (MEASURED, `stack2/pairscan.log`).
  - Mechanism: expert P(syn | eff) .63 vs P(syn | no eff) .22, against .197 / .012 in reports (MEASURED). The forum audit found 12 of 27
    expert-positive Synovitis studies with no explicit statement (CLAIMED, flat_740610). A host answer on whether effusion surrogates
    count was never posted (CLAIMED, thread 737155 has only the question).
  - **Verdict:** mechanism-backed but evidenced only by 58 studies, weaker than the PF OA .1 screen that failed. MODEL prior
    mu ~ N(+1e-4, 3e-4): P(private > 0 | public move) .73. Not recommended over anything with LB support.
- **Global abnormality prior** (each label borrows the leave-one-out mean of the other 11 percentiles), MEASURED (`stack2/severity.log`):
  - gold58 a .05 / .1 / .2: -.00014 / -.00059 / -.00532;
  - OOF v4: -.0012 / -.0030 / -.0081; OOF v2: +.0002 / -.0002 / -.0021.
  - **CLOSED.**
- **Cross-label borrowing as a selection tool** (pick the best pairs on gold58): the pair null and the zero cross-ruler correlation say
  any pick is noise-dominated, the same lesson as nested_sel (free per-label picks lose out of sample, -.0007; HANDOFF). **CLOSED.**

## 7. What a +X macro move would need (MODEL, `stack2/need.log`)
- Display .959: +3.4e-4 public (p10-p90 2.2-4.6e-4; density.md). Display .960: +1.34e-3 (out of reach for any operator change here).
- +3.4e-4 macro = +.0041 on ONE label's AUC. At the public size (390) that is roughly:

| label | pair flips (gold58-like prevalence) | pair flips (report-like prevalence) |
|---|---|---|
| Synovitis | ~155 | ~66 |
| Fracture | ~133 | ~40 |
| PF OA | ~144 | ~155 |
| ACL | ~151 | ~100 |

  Or +3.4e-4 on every label at once, which is what an operator change would have to do.
- What a true effect mu buys (noise sds measured in sec 4):

| mu | P(best-id move) | P(shows .959) | P(private > 0) |
|---|---|---|---|
| 1e-4 | .60 | .28 | .65 |
| 2e-4 | .69 | .37 | .79 |
| 3e-4 | .77 | .46 | .89 |
| 4e-4 | .84 | .56 | .95 |

- A zero-mean change adopted under R1 still has P(private > 0 | public move) .65 and E[private delta | adopted] +1.4e-4 under this noise
  model (MODEL). So R1 selection itself carries a small edge; it shrinks with the number of probes tried (selection), and gold58 cannot
  pre-screen at this size.
- Near our score, ~1.2e-5 of public score per team inside the .958 band (density.md, MODEL). So +2e-4 is ~15-20 public ranks; it does
  not change the private S6fall-cliff picture (lb_private.md).

## 8. Robustness for the PRIVATE ~910 studies
- Public and private are disjoint random splits of the same test set (CLAIMED leaderboardPercentage 30). Every real effect therefore
  transfers, and the private sd is ~.6x the public one (MEASURED ratio .61-.67 in `subsample_sd.log`). The risk is never "public noise
  vs private", it is "is the effect real in the expert regime at all".
- Ranking by robustness of the evidence:
  1. Keep rank averaging over logit / probability / geometric / median / trimmed / equal-weight: two rulers agree, so not changing is
     robust.
  2. Soft-OR (pow2 / invgeo): agrees in sign on both rulers but proven only in the report regime. Best private bet among changes, still
     only ~.7 (MODEL).
  3. Synovitis <- Effusion: regime-specific and supported by 58 studies only (~.6).
  4. Anything selected on gold58 per label or per pair: no better than a coin (CLOSED).
- Neither pick needs a stacking change for the private board. A = c4 and B = R6 already differ where it counts:
  B carries no per-label Raptor table, which is the weights hedge that matters (lb_private.md); an operator change would move A and leave
  B alone.

## 9. Spare runtime, GPU and teammate hours, seen from the stacking lens
- Operator changes cost ~0 runtime (a few rank transforms over ~1,300 x 12 numbers), so the ~2.5 h margin is irrelevant to them.
- The binding limit is expert-labelled data: 58 gold studies plus 16 report-verified extras (`gold74_eval_set.csv`, sources
  manual_verified / _r2, written from report text, MEASURED). No GPU produces more Kaggle-annotated labels.
- The one compute-for-stacking idea would be a Raptor-family OOF on train, so the OOF ruler covers c4's real structure. It needs a
  5-fold CoAtNet r384 retrain (GUESS: days of V100) and would still use report labels. **Not feasible before the 10-20 12:00 freeze;
  not worth it.**
- Running c4 members over the 16 extra gold74 studies would add n = 16 with report-derived labels: value ~0 (MODEL).

## 10. Files (all new ones under `session_2026-10-19/think/stack2/`)
| file | what |
|---|---|
| load.py | shared loaders: R7 OOF, S6 seed OOF (s1043 / s1044 / s1045 globbed from session_2026-10-0x/kaggle_frac75*), gold58, report labels (v2 / v4 / sol56) |
| oof_ops.py, oof_ops_{v4,v2,sol56}.json, oof_ops_v2.log, oof_ops_sol56.log | operator screen on the OOF ruler, 5 configurations, paired bootstrap |
| logit_scale.py | logit SDs per member and label (gold58 and OOF) and implied effective weights |
| logit_equiv.py / .log | flat_logit vs a rank blend at the effective logit weights (corr .90) |
| g58_nested.py / .log | c4 nested structure with pow2 / pow3 / invgeo / nscore / zlogit combiners on gold58 (bootstrap + split-half) |
| nested_oof.py / .log | c4-shaped nested soft-OR on the OOF ruler, per-label sign agreement with gold58, Spearman vs c4 |
| subsample_sd.py / .log | MEASURED paired sd of operator changes at public (390) and private (910) sizes |
| xlabel2.py / .log / .csv | cross-label stacker decomposition: other readers, R7's own gold58 rows, co-occurrence phi by regime |
| pairscan.py / .log / .csv | all 132 cross-label borrowings on gold58 and on the OOF ruler |
| severity.py / .log | global abnormality prior on both rulers |
| need.py / .log | what a true effect mu buys on public / private, pair-flip arithmetic |

Reused from the interrupted run: `think/stack_ops.json` (gold58 operator table), `think/cross_label.json` (CV vs gold58 stacker),
`think/tie_loss.py` (z-rank instead of saturated p: -.0005, CLOSED), `think/work/xlabel_g58.json` + `xlabel_g58b.py` (Synovitis <-
Effusion per member, half-sample), `think/work/xlabel_gap.py`, `think/work/xlabel_oof.py`, `think/ours_work/` (per-member gold58 AUCs,
nested_member: free per-label picks lose out of sample).

## 11. If Eesh wants ONE stacking probe anyway (not a recommendation to submit; needs his go + a PREREG amendment)
- Pre-specify exactly one: **c4 with the nested combiner switched from pct to pct^2 at every stage** (pow2: the milder map, lower
  measured public sd than invgeo, and its gold58 gain is less Fracture-dependent). One-line change in the stage-3 / final blend
  functions of the c4 builder (`session_2026-10-18/a18/build_a18.py` lineage); no new models, no runtime cost.
- Read by R1 vs c4. A best-id move would be consistent with the soft-OR effect; a non-move says nothing (P(move | real 2e-4) only .69).
- It competes for the same 10-20 slots as anything with LB-backed evidence, and on expected value it should lose that competition
  (MODEL).
