# THINK_1019: merged think-only research (10-19, written ~17:50 IST; critique pass ~18:30 IST, sec 8). RESEARCH ONLY.

Nothing was built, pushed, committed, submitted or created. No GPU, training or box job ran. The only box access was read-only
copying of our own prediction CSVs. No PREREG, HANDOFF, PROGRESS or scheduled-task edits. No public code was executed.
This file merges six lens files in `session_2026-10-19/think/`: runtime.md, models.md, stacking.md, forum.md, ours.md,
picks.md. It also reuses the interrupted run's files (tie_loss.py, stack_ops.json, cross_label.json, work/xlabel_*,
ours_work/, forum_raw/, pulled/watch/).
Tags: MEASURED = computed or read by us (file named); CLAIMED = a third party or an older note says so (source named);
MODEL = our arithmetic on stated assumptions; GUESS = judgement. Paths are relative to `session_2026-10-19/think/` unless
they start with `session_` or name a top-level file.

## 1. Verdict for Eesh (plain language; rewritten after the three critiques, details in sec 8)
1. **Honest answer: nothing we found is likely to move the score by a display step, and nothing beats stopping by a clear margin.** Stopping costs about 0 on private (MODEL, sec 8).
2. **The most valuable move is already in hand: select A = API best-id (c4 57343435 today) and B = R6 57248084 by hand.** P(silver) ~.63 for the pair vs ~.54 for c4 alone; P(bronze or better) ~.82 vs ~.75. These are MODEL numbers on GUESS field inputs, so read them as about +-.08 (picks.md, picks_work/matrix.txt).
3. **If you want to spend ONE slot on 10-20, spend it on E1 only:** soft-OR, i.e. square each member's percentile before the weighted average, in the kernel-natural form, all 12 labels, 0 extra runtime. Treat it as a near-zero-mean, label-agnostic probe: expected about +0.5 to +1e-4, P(true gain > 0) ~.57-.60, P(best-id move) ~.40-.55 (MODEL). Its only support is the report-label OOF ruler; gold58 says nothing either way (+.00009, MEASURED skeptic/e1_stages.log).
4. **Killed by the critique:**
   - E2 Synovitis<-Effusion: its gold58 signal is what picking the best of 9 pairs gives by chance, and co-occurrence stacking already lost on OOF and gold74 (MEASURED stack2/pairscan.log, NOVEL_PLAN_10-07.md).
   - E3 R6 shifted-crop TTA (4 folds): no positive precedent for non-mirror views here, P(S6F degraded) up to ~.29, a read on 10-21 at best (MEASURED rt_work/addon_mc.txt). The 2-fold version is parked, not recommended.
5. **Spare runtime (~2.9 h median) is not a lever.** No filler has enough expected value to be worth a slot, and leaving the time unused costs 0 (MODEL).
6. **Other model types and training: none.** Our own members are 1 of 7 on the LB (HFR7-RR was the one win, MEASURED rt_work/snap_counts.txt). Every recipe we can train scores below the c4 members on gold58 (MEASURED ours.md).
7. **GPU:** not needed. The only use would be a T4 timing probe if the parked 2-fold E3 were revived.
8. **.959:** c4 is about 1-3e-4 short, so even a zero-effect change shows .959 with P ~.3-.4 (MODEL). That display is a lottery, not evidence, and the private reasoning does not change with it. .960 (+1.1 to 1.3e-3) is out of reach.
9. **Your choice:**
   - (a) **STOP (default):** keep the picks, watch the forum, select A and B by hand before 10-22 23:59 UTC.
   - (b) **One E1 slot on 10-20:** needs your go, a PREREG A20 amendment, a gate PASS and someone (a live session, or a tick task you approve) to submit after 05:30 IST. Nothing else.

## 2. Ideas ranked by expected PRIVATE gain x P(positive) x feasibility
**Note (critique pass):** sections 2 to 7 are the pre-critique draft, kept for the record with the misread numbers fixed
inline as [CORR]. Where they disagree with sec 8, sec 8 wins (final verdicts, ranking and the E1 build spec).

Index = P(best-id move) x E[private gain | adopted under R1] x P(ready and clean in the window).
- The first two factors come from the stack2/need.log noise model (MODEL). Feasibility is a GUESS.
- The +1e-4 private ~ +3.9 pts pair P(silver) conversion is from picks.md / picks_work/log_value_shift_base.txt (MODEL).
- Calibration warning (MODEL, picks.md sec 7, slots_mc.py): if probe means are calibrated to our own record (0 of 9 around c4, 4 of 19 since 10-16), a neutral new direction is worth only ~+0.6 pt P(silver) per probe.
- The need.log reference is a zero-mean change. Under R1 it still shows +0.7e-4, because the true effect has spread (index .50 x +1.4e-4). With several probes the winner's curse eats part of that.
- Read the indices below as "all within noise of each other and of a zero-mean probe". Realistic value: ~+1 to +3 pts P(silver) per probe (GUESS).

| # | idea | runtime | E[public] | P(best-id move) | index (MODEL/GUESS) | feasibility in window | verdict |
|---|---|---|---|---|---|---|---|
| E1 | soft-OR combiner: pct -> pct^2 at every nested stage of c4 | 0 | +1 to +2e-4 (MODEL) | .58-.65 | +1.1 to +1.8e-4 | high: few-line build, ~6 h scoring | best of a weak set |
| E2 | Synovitis <- Effusion nudge, a = .1, on the final c4 output | 0 | +1e-4 (MODEL) | ~.58 | ~+1.1e-4 | high | coin flip with a mechanism |
| E3 | R6 +1 shifted-crop TTA view per fold, S6F start guard 5.75 -> 6.6 h | +95-115 min | +0.5 to +3e-4 (GUESS) | ~.45-.55 | ~+0.5 to +0.7e-4 | medium: patch + 2 GPU measurements + guard change | only use of spare runtime with a mechanism |
| - | everything else (sec 2.4, sec 5) | | ~0 or negative | | <= 0 | | do not spend slots |

### 2.1 E1: soft-OR rank combiner (pow2)
- **Mechanism.** Each member's percentile u goes through a convex map (u^2) before the weighted average. A study that ONE
  member ranks near the top gets pulled up, so top-end disagreements resolve toward the confident member. Two related
  maps lose on both rulers (MEASURED, stack2/oof_ops_*.json, stack_ops.json): soft-AND (geometric mean) and hard OR
  (max, pow4).
- **Evidence.**
  - OOF ruler (R7 OOF + our S6 seeds s1044 / s1045, 3,467-4,333 non-gold train studies, 3 report readers): +.00037 to
    +.0012 macro, P>0 .95-1.0, 8-11 of 12 labels up, in flat and c4-shaped nested forms (MEASURED,
    stack2/oof_ops_{v4,v2,sol56}.json, stack2/nested_oof.log).
  - gold58 with c4 nested: +.00048 (P .64). Split-half sign agreement is .49. Fracture carries +.0083; with Fracture held
    at c4 the gain is +.00028 (MEASURED, stack2/g58_nested.log). [CORR: misread by one row; the held line prints BEFORE its
    operator, so pow2 with Fracture held is -.00021. This screen also re-ranked the Raptor mix, which c4 does not do; the
    kernel-natural pow2 reads +.00009 (MEASURED skeptic/e1_stages.log). See sec 8.1.]
  - Caveat found while merging (MEASURED, stack2/g58_nested.log, "OOF-ruler family on gold58" block): the SAME R7 + s43
    family scored on gold58 reads pow2 +.00035 (P .63) at .5/.5 but -.00082 (P .20) at .8/.2. So the report-regime gain
    does not carry cleanly to expert labels. Prior shrunk from N(2e-4, 3e-4) to ~N(1e-4, 3e-4) (MODEL).
  - Spearman vs c4 .9968. Measured noise: public-size paired sd .00041, private .00025 (MEASURED,
    stack2/subsample_sd.log).
- **Cost.**
  - 0 runtime.
  - The c4 notebook defines a percentile-rank helper in 4 cells (4, 11, 14, 18). MEASURED by grep of
    session_2026-10-18/a18/kernel_v224r6a25c4j/rsna_sub_v224r6a25c4j.ipynb. So this is a few-line edit, not one line.
  - R6's internal Effusion / Baker's stackers stay untouched.
  - Build from the build_a18 lineage, then gate, commit and score (~6 h latency, MEASURED runtime.md sec 1).
- **Risk.** Proven only in the report regime, with no Raptor member. On gold58 it is Fracture-led, the direction fr100
  punished. A non-move is uninformative: P(move | real +2e-4) is only .69 (MODEL, need.log).
- **LB test design (one change).**
  - E1 = c4 with u -> u^2 at exactly the three stages screened in stack2/g58_nested.py: Raptor r384/r224 mix -> stage 3
    per-label Raptor vs R6 -> final .8/.2 with S6fall.
  - Gate: an offline replay on the gold58 serve predictions must give macro .94150 (= c4 .94102 + .00048, MEASURED
    g58_nested.log). The same code with exponent 1 must be bit-identical to c4. [CORR: these two conditions conflict,
    because .94150 includes an extra Raptor re-rank. Superseded by the sec 8.4 spec: kernel-natural, gold58 .94111.]
  - Read: R1 vs c4. A best-id move is positive; a non-move is negative or unread. Do NOT also run invgeo
    (-log(1 - u), the sharper sibling): pre-specify one.

### 2.2 E2: Synovitis <- Effusion co-occurrence nudge
- **Mechanism.**
  - Hidden-test labels come from the same expert image review as gold58 (CLAIMED, host Po-Hao Chen, discussion 733491).
  - Expert Synovitis goes with effusion: gold58 P(syn | eff) .63 vs .22 without effusion, phi .40. Report labels give .197
    vs .012. Prevalence is .47 expert vs .12 report (MEASURED, work_forum/g58_labelrel.json, work/xlabel_gap.py).
  - Every member learned the narrow report Synovitis, so the well-learned Effusion score carries extra expert-Synovitis
    signal.
- **Evidence.**
  - gold58 per-label: +.0048 at a .1 (P .85), +.0084 at a .2 (P .81). Positive for 5 of 6 members; half-sample P(d>0)
    .72 (MEASURED, work/xlabel_g58.json, work/xlabel_g58b.py).
  - Against: the OOF ruler reads -.0003 / -.0014, but that is the wrong regime by construction (MEASURED,
    work/xlabel_oof.py).
  - Against: it ranks only 10th of 132 target<-helper pairs on gold58 (null mean -.0023, sd .0046), and gold58 vs OOF pair
    deltas correlate .05 (MEASURED, stack2/pairscan.log).
  - Forum audit: 12 of 27 expert-positive Synovitis studies had no explicit report statement (CLAIMED, discussion 740610).
    Host never answered the rubric question (CLAIMED, 737155).
  - Its gold58 evidence is weaker than the PF OA .1 screen (P .925) that read negative on 10-19 (MEASURED LB,
    HANDOFF 10-19 16:15).
- **Cost.** 0 runtime, one line on the final Synovitis column, about 6 h to a read.
- **Risk.** Synovitis is the noisiest label (per-label public sd .032; MODEL, picks_work/picks_cov.json). It carries about
  1/4 of the private macro variance, so the read is mostly noise. Winner's curse applies.
- **LB test design.**
  - E2 = c4 with Syn_final = rank-pct(pct(Syn_c4) + .1 x pct(Eff_c4)), a fixed at .1, matching the screened c4 row in
    work/xlabel_g58.json. Nothing else changes.
  - Read: R1 vs c4. Never stack with E1.

### 2.3 E3: R6 shifted-crop TTA (the only spare-runtime idea with a mechanism)
- **Mechanism.** R6 trained on random 352 crops of the 384 render but serves only the centre crop. MEASURED (runtime.md
  sec 4): compact_train `augment`, trainer md5 8622009e == R6 manifest; compact_infer l.755
  `off = (size - crop) // 2`. Averaging the same strong model over an in-distribution view reduces variance, and the
  change is label-agnostic, so a real gain should transfer to private (MODEL).
- **Evidence.**
  - For: flip TTA on the main CNN leg moved the best-id once, V224-exact r486 -> V224-FLIP r267 (MEASURED snapshots).
    Raptor mirror +.001 alone (CLAIMED, nartaa 746792).
  - Against: shift views are far more correlated than a mirror (MODEL). The r224 mirror was a non-move (MEASURED FLRM
    57200578). No read of shift TTA exists on this comp.
- **Cost (runtime).**
  - Hidden +95-115 min (MODEL from the commit's 4.6-4.9 ms per window-view-fold).
  - The S6F start guard G_START_H must go from 5.75 to 6.6, keeping DEADLINE_H 8.2 and the kill. P(S6F degraded) is .07
    at G 6.6 vs .47 at 5.75 (MODEL, rt_work/addon_mc.txt).
  - End median ~7.7 h; P(> 9 h) ~0 with the existing kills (MODEL).
  - 2-fold variant: +48-58 min, guard unchanged, P(degraded) ~.05, about half the gain (MODEL/GUESS).
- **Cost (build).**
  - Worker patch (`--shift-tta`, decode shared, logits averaged within fold) + parity test + gate: ~4-6 h.
  - Two GPU measurements first, each needing Eesh's go:
    - box V100 OOF screen of R6 views on ~400 non-gold box train studies, each scored by its held-out fold
      (fold = sha1(StudyInstanceUID)[:8] % 4, MEASURED compact_train l.2014), 30-50 min. It catches a bug or a large
      negative only (macro SE ~.002-.003, MODEL).
    - Kaggle T4 timing probe of the patched worker on 30-60 TRAIN studies, 20-40 min. This turns the +95-115 min MODEL
      into MEASURED.
- **Risk.**
  - A patched third-party worker.
  - A guard-degraded run reads as a big visible negative (not ambiguous, but the slot is lost).
  - ~7.7 h scoring pushes the read late.
  - Its index is no better than a zero-mean probe's R1 edge; its only case is label-agnostic private transfer.
- **LB test design.**
  - E3 = the CURRENT best at build time (c4, or E1 / E2 if one moved the best-id) + one pre-registered shift offset per
    R6 fold. Choose one offset before the build, e.g. (+16, +16) px of the 384 render. Plus S6F G 6.6. Nothing else.
  - Build on the current best so that R1 stays a valid one-change read.
  - The canary must fire on an S6F skip, so a degraded run is visible.

### 2.4 Considered and NOT recommended for a slot
| idea | numbers | why not |
|---|---|---|
| invgeo soft-OR (-log(1 - u)) | gold58 nested +.00089 (P .73) but -.00018 with Fracture held [CORR: +.00031 with Fracture held, same off-by-one]; OOF +.00035 to +.00081 (MEASURED stack2) | a sharper twin of E1 with higher public sd (.00048); run one or the other, never both |
| Dread maxspan CoAtNet leg after S6F, w .05-.10 | gold58 +.00036 at .10 (P .63), within-class rho vs c4 .79 (MEASURED models_work/div_g58.json, noisecorr_g58.json); solo .928 (CLAIMED) | inside the gold58 noise floor (S6fall .3 read +.0012 there and was LB-negative); new stage port; Dread quotes ~9 h for its 94-window pipeline (CLAIMED 737696); P ~.35 (GUESS) |
| Mycelis 4-fold ConvNeXt OOF check (neronrx, CC0, new 10-18) | 4 OOF CSVs, ~15 min CPU (MEASURED file list) | info only; needs Eesh's OK to download untrusted files; prior ~.1 that it is strong and different (GUESS); any build would miss the window |
| Raptor r384 multi-FOV TTA (320 crop + full 384) | gold58 inside c4 -.00032 (P .41) at .5, -.00045 at 1/3 (MEASURED work_forum/fov_tta_g58.json) | no signal; +~25 min |
| dreaddevelopment finespacing in the Raptor slot | our 10-04 v9 finespacing swap was a visible no-op (CLAIMED HANDOFF); solo .932 (CLAIMED) | dilutes the best Raptor; P ~.35 |
| hengck23 MaViT-288 leg | .935 solo (MEASURED page) | training data unstated, author named KneeCoT (host-banned): rule risk for a final pick |
| c4noS6 as pick B | MODEL .59 / .82 vs c4+R6 .63 / .82 (picks_work/matrix.txt) | only if Eesh believes S6fall is as likely to hurt as help; the public evidence leans pro-S6 (S6 .15 and .3 both lost to .2, MEASURED) |
| KC1 / KC-all .05-.10, KC on soft-tissue labels, KC replacing r224 | gold58 +.0003 to +.0009, split-half .40-.61 (MEASURED ours_work/ours_more_g58.log, ours_small_sh.log) | our members are 0 of 6 on the LB; KC is weaker than full-frame on ACL by .016 on 867 OOF studies (MEASURED ours_work/kc_vs_full_oof.log) |

## 3. Day plan (only on Eesh's explicit go + a PREREG amendment per entry)
Clock facts:
- The UTC day resets at 05:30 IST. All 5 slots of the 10-19 UTC day are used (MEASURED PROGRESS_19).
- 10-20 and 10-21 have 5 slots each. On 10-22 the last readable submit is ~10:15 IST (brief).
- c4-class scoring latency ~6 h, 5h40-6h27 (MEASURED runtime.md); E3 ~7.7 h (MODEL).
- Precedent: the A19 kernels were built and committed overnight, submitted 09:56, read ~16:00 (MEASURED PROGRESS_19), so
  one-change kernels built tonight can read on 10-20.

**Option (a), STOP:** no builds. Watch the forum and select picks (sec 4).

**Option (b), at most 3 probes:**
| when (IST) | action | GPU | gate / needs |
|---|---|---|---|
| 10-19 evening | Eesh picks E1 and/or E2 (and says yes or no to E3). Main session writes the PREREG A20 amendment, builds E1 / E2 from the build_a18 lineage (c4 + one change each) and runs the offline gold58 replay + exponent-1 parity. Commit overnight with `machine_shape` NvidiaTeslaT4, push with `--timeout`, <= 2 GPU sessions per account (CLAIMED SwapIY 750318 / 744230) | Kaggle commits | Eesh's go; gate PASS |
| 10-19 night (only if E3) | write the R6 `--shift-tta` worker patch + CPU parity test; run the box V100 OOF screen of R6 views (30-50 min) | box V100 | Eesh's go for the GPU job |
| 10-20 05:31+ | submit E1, E2 (slots 1-2 of the 10-20 UTC day) | | gates PASS, Eesh's go |
| 10-20 morning (only if E3) | T4 timing probe of the patched worker on 30-60 train studies. Run it on the friend's account if Eesh's account already has 2 GPU sessions busy | T4 20-40 min | Eesh's go |
| 10-20 ~12:00-14:00 | read E1, E2 by R1 vs c4 | | |
| 10-20 afternoon (only if E3) | if the timing probe says <= ~2 h and the box screen shows no bug: build E3 on the then-best, S6F G 6.6, gate, commit, submit as slot 3 by ~16:00-18:00 | | A20 addendum, Eesh's go |
| 10-21 ~00:00-03:00 | read E3 by R1 | | |
| 10-21 UTC day | default: nothing new. One allowed exception: if E3 degraded for a clear mechanical reason, its 2-fold variant (guard unchanged) as one entry | | Eesh's go |
| 10-22 05:30-10:15 | default unused (no readable follow-up after this) | | |
| 10-22 before 18:00 | A15 last read: A = API best-id, B = R6 | | |
| 10-22 before 23:59 UTC (10-23 05:29 IST) | Eesh selects A and B by hand and screenshots the checkmarks | | |

GPU summary:
- The box V100 is needed only for the E3 view screen.
- The friend's ~30 GPU-h are needed only for the E3 timing probe; a full 1,300-train-study end-to-end timing run (~8 h T4)
  is optional.
- Eesh's quota covers the commits.
- No training anywhere (sec 5).

## 4. Final-pick recommendation
- **A = API best-id at the last read before 10-22 18:00 IST.** Today that is c4 57343435. **B = R6 57248084** (PREREG A15).
- **Why R6 as B** (MODEL, picks.md sec 3-4, picks_work/matrix.txt):
  - The pair is .63 / .82 (silver / bronze+) vs .54 / .75 for c4 alone.
  - R6 is the best B, or within ~2 pts of it, in all 22 scenarios. [CORR: false. c4+S6F beats c4+R6 in 21 of 22 rows by
    +1 to +5 pts, and HW / fr100 each beat R6 by 3 pts in one row (MEASURED picks_work/matrix.txt). Keeping R6 over S6F is
    a GUESS-level judgement, sec 8.1 item 6.]
  - The c4 vs R6 private margins correlate only .54, so the hedge works through variance.
- **Select both by hand.** If nothing is selected, Kaggle auto-picks the top 2 public (CLAIMED rule). That likely pairs c4
  with a near-twin (pf10 / fr100 / a25): -4 to -8 pts silver (MODEL).
- **If a probe moves the best-id, A switches to it once and B stays R6.** Never use c4 as B then: it would be a near-twin of
  the new A (MODEL, picks.md sec 3).
- **Alternatives:**
  - HW 57262115 is an acceptable B (-1 pt silver, +1 pt bronze+), but only if Eesh trusts heliosli's per-finding table.
  - S6F 57201874 (.957) as B is not recommended: its edge is fragile.
- **Private context (MODEL, picks.md):**
  - Silver needs a private margin over the ~750-team copy block of ~5e-4. c4's posterior margin is 5.6 +- 6.9e-4.
  - P(c4 private < copy block) ~.21: the S6fall cliff, shared by every existing pick.
  - Watch the forum before selecting. Re-open the hedge only on a host ruling against the OAI-trained weights or the R6 /
    fastMRI provenance (forum.md sec 9).
  - The hidden test probably includes sites unseen in train (GUESS P ~.6-.7). Evidence (MEASURED, forum.md sec 3): the
    Acknowledgements page lists Mongolia / Italy / Australia sites, and train.csv has 0 Italian and 0 Mongolian reports.
    This favours label-agnostic changes and the less-tuned B.

## 5. What NOT to do, and ideas closed
Do not:
- submit anything without Eesh's explicit go + a PREREG amendment + a gate PASS;
- stack two public winners into one A (the 10-19 stacks lost a full display step, MEASURED);
- swap B away from R6 after a probe win;
- run more single-label weight probes (PF OA .1 step posterior P<0 .91; Fracture / Lateral OA noise-dominated; MODEL
  picks.md);
- pick weights, members or label pairs from gold58 (PF OA .1 miscall at P .925, MEASURED; nested free per-label picks
  -.0007, MEASURED ours_work/nested_member.py);
- gate a member on a single-model base (inflation 3-10x, MEASURED ours_work/wrongbase.py);
- add any unguarded stage longer than ~0.5 h (P(> 9 h) 1.5 % at 0.75 h, 10 % at 2.5 h, MODEL rt_work/addon_mc.txt);
- put inline compute before S6F without raising its 5.75 h guard;
- train anything (freeze 10-20 12:00; every recipe we can run is .877-.924 on gold58 vs .923-.937 for the c4 members,
  MEASURED ours.md);
- chase the .959 display for its own sake: worth ~90-110 public ranks, but only its shrunk true part on private (MODEL
  picks.md). P(shows .959) is .21 even for a zero-effect change (MODEL need.log), so it is mostly a lottery. [CORR: need.log
  used a25's +3.4e-4 gap; c4 is ~1-3e-4 short, so ~.3-.4 (MODEL).]

Closed (MEASURED unless tagged):
- **Stacking / operators**
  - logit averaging: gold58 +.00088 is an implicit R6 re-weighting to .58-.90 per label (corr .90 with that rank blend);
    OOF -.0034 to -.0068 (stack2/logit_scale.py, logit_equiv.log);
  - probability mean, geometric mean, median, min, max, trimmed mean, equal weights, probit-rank: <= 0 on both rulers;
  - z-scored logits: the rulers disagree;
  - cross-label logistic stacker: CV +.0096 is v2-reader noise (-.0010 / -.0011 vs the other readers, -.0059 on R7's own
    gold58 rows, stack2/xlabel2.log);
  - global abnormality prior (stack2/severity.log);
  - z-rank instead of saturated p: -.0005 (tie_loss.py);
  - adding LB-negative members as crowd legs (stack_ops.json).
- **Runtime fillers**
  - R6 vertical flip and multi-scale: OOD;
  - R6 +2 views and R6 views as a separate post-S6F stage: do not fit;
  - S6fall flip: trained without mirror, no worker flag;
  - r224 mirror: non-move;
  - extra Raptor arms (HFR7-RS .955 < .956);
  - R7 in any form: R6+R7 and the R7 late leg were non-moves;
  - heliosli S6 compact-v1: CLAIMED .957 vs .958;
  - window density: every centre is already used;
  - S6F removal.
- **Public models**
  - goodpjw 2.5D reader, B-leg R6 fine-tunes, jessky cnx50, other Dread arms, r9all, the mattiaangeli and RadImageNet
    families, pilkwang DINOv2 (.891), unscored new releases, foundation backbones (need training), isotriage OAI arm,
    private sources (404), copies (models.md sec 3).
- **Our assets**
  - KC1, KC-all, K14 DT, fleet D, T_a, T_b, our Raptor seeds, s43 / s44 and soups, at any weight, on any label subset,
    or as replacements (ours.md).
  - Training a new seed or architecture, a distilled student, or an OAI variant.
- **Forum**
  - metadata / site priors: gold58 -.0152, and riskier with unseen sites;
  - co-occurrence stacking: -.001;
  - multi-scale TTA on our models: -.0005;
  - external-data hedge pick: no clean entry above ~.94.
  - Host or staff posts since 10-14 19:56 UTC: none, checked to 11:38 UTC 10-19 (forum.md sec 2).

## 6. Corrections and caveats found while merging
- **s43 / s44 were never on the LB** (ours.md sec 2, citing HANDOFF 10-17 sec 3). The brief and runtime.md list them as
  LB-negative. They were closed on gold58 (.921 / .915). The 0-of-6 LB count for our members does not include them.
- **c4's own hidden latency is ~6h03-6h06** (MEASURED runtime.md, api_chain.log). The brief's "<= 6h33" is HW's loose bound.
  Unused runtime: median ~2.9 h, p05 ~2.5 h, >= 0.78 h guaranteed by the S6F kill (MODEL rt_work/addon_mc.txt).
- **The "one-line" stacking change touches 4 rank helpers in the c4 notebook** (MEASURED grep). The gate must replay the
  exact nested stages of stack2/g58_nested.py.
- **E1's evidence weakens on the same member family under expert labels** (MEASURED g58_nested.log, sec 2.1). The stacking
  lens headline did not carry this.
- **The display truncates:** `scoreTruncationNumDecimals 3` (MEASURED work_forum/pages/GetCompetition.json), so c4's .958
  means [.958, .959).
- **Two different value scales give different answers for the same probe.** The need.log Bayesian prior gives ~+4-7 pts
  P(silver) per good probe; calibrating to our probe record (slots_mc.py) gives ~+0.6 pt. The truth is probably in between
  (GUESS). That gap is why section 1 says "coin flip".

## 7. Source files (all MEASURED outputs unless noted)
- Lens files: runtime.md, models.md, stacking.md, forum.md, ours.md, picks.md.
- Runtime: rt_work/snap_counts.py/.txt, rt_work/addon_mc.py/.txt.
- Stacking:
  - stack2/oof_ops_{v4,v2,sol56}.json, nested_oof.log, g58_nested.py/.log;
  - stack2/logit_scale.py, logit_equiv.log, xlabel2.log, pairscan.log, severity.log, subsample_sd.log, need.log;
  - reused from the interrupted run: stack_ops.json, cross_label.json, tie_loss.py, work/xlabel_g58.json,
    work/xlabel_g58b.py, work/xlabel_oof.py, work/xlabel_gap.py.
- Models: models_work/div_g58.json, noisecorr_g58.json, calib_g58.py, kvm_top_1019.json; pulled2/ (metadata only,
  untrusted).
- Ours: ours_work/ours_g58.json, ours_more_g58.log, ours_small_sh.log, ours_oof.log, kc_vs_full_oof.log, nested_member.py,
  wrongbase.py.
- Forum: forum_raw/, forum_raw2/, work_forum/hi_rank_posts.txt, work_forum/pages/, g58_labelrel.json, fov_tta_g58.json.
- Picks: picks_work/matrix.txt, picks_cov.json, post_*.json, field_*.json, log_value_shift_base.txt, slots_mc.json;
  work/api_lb_1019T_think.json (LB snapshot 10-19 11:00 UTC).
- Kernel checked read-only: session_2026-10-18/a18/kernel_v224r6a25c4j/rsna_sub_v224r6a25c4j.ipynb.

## 8. Critique and final ranking (10-19 ~18:30 IST; three skeptic passes: evidence, LB history, stats)
Inputs: the three critique verdicts, their new CPU checks (skeptic/e1_stages.py/.log and skeptic/oof_norr.py/.log in this
dir; two scratchpad replays by the evidence and history passes that reproduce c4 gold58 .94102), and my re-reads of
stack2/g58_nested.py l.51-53 + .log, rt_work/addon_mc.txt, picks_work/matrix.txt, stack2/pairscan.log,
rt_work/snap_counts.txt, forum_raw/flat/flat_737696.txt, NOVEL_PLAN_10-07.md l.63/279 and c4 notebook cells 14 / 18
(read-only). Nothing was built, submitted or run on a GPU or the box.

### 8.1 Errors in the draft (each confirmed by re-reading the named file)
1. **Fracture-held numbers misread by one row.** g58_nested.py prints "(Fracture held)" BEFORE each operator row.
   Correct: pow2 -.00021 (draft +.00028), pow3 +.00028, invgeo +.00031 (draft -.00018), nscore -.00018 (MEASURED
   stack2/g58_nested.log, skeptic/e1_stages.log). E1's whole gold58 gain is Fracture (+.0083, about 6 pair flips among 18
   positives). "pow2 is less Fracture-dependent than invgeo" (stacking.md sec 11) is backwards.
2. **The gold58 E1 screen carried a second change.** Shipped c4 does not re-rank the Raptor mix at stage 3 (MEASURED cell
   14 l.71-75: rr = (1 - A224) * rr + A224 * pct_on(r224); b = wr * rr + (1 - wr) * ro). g58_nested.py's pow2 path does.
   Re-rank alone reads +.00094 (P .86); the kernel-natural pow2 (no re-rank) reads +.00009 (sd .00084, P .54), -.00014
   with Fracture held (MEASURED skeptic/e1_stages.log). The old gate ".94150 AND exponent 1 bit-identical" cannot pass.
3. **OOF upper end inflated.** The +.0012 top came from the .8/.2 shape, where prob-mean (+.0078) and pow4 (+.0032) also
   win, i.e. re-weighting toward the under-weighted member (CLAIMED evidence pass, stack2/oof_ops_v2.log). Same-structure
   pow2 vs exponent 1: +.00036 to +.00064 at R7 .5 and +.00041 to +.00112 at R7 .7, P .98-1.0, all 3 readers, with or
   without re-rank (MEASURED skeptic/oof_norr.log). The paired sd .00041 / .00025 was measured on R7 / s44 OOF members, so
   using it for c4 is MODEL, not MEASURED.
4. **Stale .959 gap.** +3.4e-4 is a25's gap (density.md, written 10-18 before c4). c4 is ~1-3e-4 short (picks.md sec 1;
   brief .9587-.9589). P(shows .959 | zero change) ~.3-.4, not .21 (MODEL).
5. **Our LB record is 1 of 7, not 0 of 6.** HFR7-RR 57146261 (HR7-RR + our K14 fleet at .2) was the API best-id from
   10-14 05:27 (MEASURED rt_work/snap_counts.txt; HANDOFF 10-14 13:05 block). Compute-adding entries are at least 3 of ~12.
   "Actively harmful (p ~1/64)" is overstated; "do not train" still holds on the gold58 member gap.
6. **Pick robustness overclaimed.** c4+S6F beats c4+R6 in 21 of 22 matrix rows (+1 to +5 pts silver); HW (tabReal .62 vs
   .59) and fr100 (skepS6 .49 vs .46) each beat R6 by 3 pts in one row (MEASURED picks_work/matrix.txt). Dropping S6F as
   B rests on a GUESS: its variance is inflated by the R7-OOF stand-in, R6 is stronger on gold58 (.937 vs .919), and a .957
   entry over a .958 one is hard to defend.
7. **The "S6fall cliff" P ~.21 is mostly private sampling noise** on c4 minus the copy block (posterior true-margin sd
   4.0e-4 of 6.9e-4 total, picks.md sec 3), not specific to S6fall. No existing B removes it; picks unchanged.
8. **E3 runtime optimistic.** From runtime.md's own inputs one extra view on 4 folds is +100-124 min (93-105 if GPU-bound,
   MODEL). P(S6F degraded) at G 6.6 is .067 at X 1.7 h but .287 at X 2.0 h, and G 6.25 vs 6.6 give the same numbers
   because DEADLINE_H 8.2 binds (MEASURED rt_work/addon_mc.txt). Fold rule is compact_train l.2016, not 2014.
9. **Dread "~9 h" quote is stale.** The same thread later says "under 2 hours", then 29 minutes (MEASURED
   forum_raw/flat/flat_737696.txt l.71-73). Its KILL stands for the other reasons.
10. **Every gold58 screen needs the PF OA-miss discount.** Raptor "was selected on the 58 gold" (CLAIMED, c4 notebook cell
    14 l.5 comment) and carries .8 of Synovitis, .65 of Fracture: gold58 is extra-optimistic on exactly the labels that
    drove E1 and E2.
11. **E3 timeline:** with ~6.5 h build+commit, a realistic submit is 19:00-22:00 IST 10-20 and the read 03:00-06:00 IST
    10-21 (MODEL, history pass), not 16:00-18:00 / 00:00-03:00.
12. **Missing from the day plan:** who submits at 05:31 IST. It needs a live session or a new tick task plus a GO file,
    and both need Eesh's go (no scheduled task was touched here).

### 8.2 Final verdict per idea
| idea | draft verdict | FINAL | reason |
|---|---|---|---|
| E1 soft-OR pow2 | best of a weak set | **DOWNGRADE** (still the only probe worth a slot) | evidence is the OOF ruler only: report labels, no Raptor member, and that ruler rated our own members +.01-.02 before they lost on the LB (CLAIMED ours.md sec 0). gold58 is ~0 in kernel-natural form (+.00009, P .54); the same R7 / s43 family at c4's .8/.2 shape reads -.00082 (P .20) on gold58. Prior ~N(+0.5 to 1e-4, 3e-4) (MODEL). Framed honestly: a near-zero-mean, label-agnostic operator probe |
| E2 Synovitis <- Effusion, a .1 | coin flip with a mechanism | **KILL** (2 passes KILL, 1 borderline KILL) | best of 9 pairs from the 132-pair null (mean -.0023, sd .0046) is expected at ~+.0045 vs observed +.0048 (MODEL on stack2/pairscan.log); ~8.1 null pairs expected at or above its rank-10 value; Effusion also "helps" Lateral OA +.0106, PF OA, MCL, so it is not Synovitis-specific; split-half sign agreement .499; OOF ruler -.0003 / -.0014; label co-occurrence stacking already -.00096 OOF and -.0010 gold74 (MEASURED NOVEL_PLAN_10-07.md l.63/279); breaks this file's own sec 5 rule. The host mechanism quote is real (CLAIMED 733491, 740610) but does not rescue the evidence |
| E3 R6 shifted-crop TTA, 4 folds | only spare-runtime idea with a mechanism | **KILL** for this window | non-mirror spatial views are 0 of 3 here (Raptor multi-FOV -.00032 MEASURED fov_tta_g58.json; multi-scale TTA -.0005 MEASURED NOVEL_PLAN; nartaa smaller crops -.001 / -.002 CLAIMED 746792); shifted views correlate ~.98-.99 with the centre vs ~.95 for a mirror, so ~+0.5e-4 on the final score (MODEL); P(degraded) up to ~.29; 2 GPU jobs, a patched third-party worker (cell 7 sha pin 8401812c re-pin, parity must cover R6's stacker inputs), offset "(+16, +16)" ambiguous (lands on the 0-32 corner); read 10-21 at best |
| E3 2-fold variant | fallback | **DOWNGRADE, parked** | +48-58 min, guard unchanged, P(degraded) ~.05 (MODEL); about half of a ~+0.5e-4 effect; needs a T4 timing probe + Eesh's go; not recommended |
| invgeo soft-OR | twin of E1, not both | **KILL** as an entry | "one operator, pre-specified" stays. With corrected numbers pow2 vs invgeo is a coin flip; pow2 is kept for its lower read noise (.00041 vs .00048, MODEL) and milder map, NOT for Fracture |
| Raptor-mix re-rank alone (exp1rr) | new in critique | **KILL** | gold58-only +.00094 (P .86), found post hoc; weaker than PF OA .1 (P .925), which read negative; no OOF read |
| Dread maxspan leg | not recommended | **KILL** | inside the gold58 noise floor, checkpoints picked on gold58 (CLAIMED), needs a new stage port |
| Mycelis OOF check | not recommended | **KILL** for a slot | info only, needs Eesh's OK to download untrusted files |
| MaViT, finespacing, multi-FOV, KC variants, s43 / s44, any training | not recommended | **KILL** | repeats of measured negatives or no-ops; MaViT is a rule risk |
| c4noS6 as B | not recommended | **KILL** unless a host ruling lands against S6fall | |
| Option (a) STOP | option | **KEEP, default** | costs ~0 on private |
| Picks A = API best-id, B = R6, by hand | KEEP | **KEEP** | R6-as-B beats c4 alone by +5 to +12 pts silver in every matrix row (MEASURED matrix.txt); the S6F alternative is a judgement call (8.1 item 6) |

### 8.3 Surviving ranked list (re-ranked)
| rank | item | expected gain | P(positive) | cost / timing |
|---|---|---|---|---|
| 1 | Select A = API best-id (c4 57343435 today), B = R6 57248084, both by hand | +9 pts P(silver) vs c4 alone (.63 vs .54), bronze+ .82 vs .75 (MODEL, +-.08) | R6-B above c4-alone in 22 of 22 rows (MEASURED matrix.txt) | 0 runtime; before 10-22 23:59 UTC, screenshot the checkmarks |
| 2 | Option (a) STOP: no new entries | 0 | n/a | 0 |
| 3 | E1 soft-OR pow2, kernel-natural, one slot on 10-20 | +0.5 to 1e-4 true (MODEL), ~+0.5 to 1.5 pts P(silver) | P(true > 0) ~.57-.60 (MODEL); P(best-id move) ~.40-.55 (MODEL, discounted by our record of 4 moves in 19 reads and 0 in 9 near c4) | 0 runtime; build + gate tonight, submit 10-20 after 05:30 IST, read ~12:00-14:00 IST |
| 4 | E3 2-fold shift TTA (parked) | ~+0.25e-4 (MODEL) | ~.45-.50 (GUESS) | T4 probe + 4-6 h patch; read 10-21 03:00-06:00 at best; not recommended |

Ranks 1 and 2 are compatible: rank 1 happens in every plan. The real decision is 2 vs 3, and they are within noise of
each other (GUESS). E1 is a reasonable lottery ticket because it repeats no LB negative and is not tuned on public or
gold58 labels; it is not a reasoned gain.

### 8.4 E1 build and read spec (replaces sec 2.1 "LB test design"; only on Eesh's go + PREREG A20)
- One change, kernel-natural: in cell 14 square the per-arm percentiles before they mix (pct_on of r384 and of r224, and
  ro for R6), with NO extra re-rank of the Raptor mix; in cell 18 square rb and rs AFTER the failed-row fill (l.77).
  Leave cells 4 and 11 alone: squaring a single-arm pct is monotone and is re-ranked downstream.
- All 12 labels, no exclusions. R6's internal Effusion / Baker's stackers untouched.
- Gate: exponent 1 gives a submission.csv bit-identical to c4; exponent 2 replayed on the gold58 serve predictions gives
  macro .94111 (= c4 .94102 + .00009, MEASURED skeptic/e1_stages.log "norr").
- Read by R1 vs c4. A best-id move: A switches to E1 once, B stays R6. A non-move: closed, no retry with invgeo or pow3.
- Never stack with anything else. No second probe in parallel (the loser of two winners would be unread under R1).

### 8.5 Recommended next action for 10-20
- Ask Eesh one question tonight: STOP, or one E1 slot.
- Default if no answer: STOP. No builds, no GPU, no box jobs.
- If go: the main session writes PREREG A20, builds E1 to the 8.4 spec from the build_a18 lineage, runs both gates and
  commits overnight; it is submitted as slot 1 of the 10-20 UTC day (after 05:30 IST) by a live session or an approved
  tick, and read ~12:00-14:00 IST by R1. Nothing else is submitted on 10-20, 10-21 or 10-22.
- In every case: watch the forum for a host ruling on OAI weights or R6 / fastMRI provenance, then select A = API best-id
  and B = R6 by hand before 10-22 23:59 UTC (10-23 05:29 IST).
