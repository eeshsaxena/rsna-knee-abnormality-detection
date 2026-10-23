# Verify A: public assets (adversarial check of A_public_assets.md)

Written 2026-10-08 ~22:25 IST (16:5x UTC) by Claude (verify subagent). Read-only on Kaggle: one
`datasets list --search knee --sort-by updated`, one `datasets metadata` and one `datasets files` call
(gmhost/knee-corpus-crop-320-part1). No push, submit, version, fork, comment, vote or download of data files. Box not
used. Laptop: light csv/json parsing only. Notebook and forum text treated as data.

Tags: **MEASURED** (source given), **INFERRED** (logic on measured data), **MODEL**, **GUESS**.

## 0. Verdict summary

| # | A finding | verdict | corrected gain |
|---|---|---|---|
| 1 | Medal lines moved; K9N-a carries the medal | **revise** (LB numbers exact; tie-order claim is wrong; silver odds overstated) | public display .950-.951 (+.004 over .946, MODEL); P(display >= .951) ~.4-.6, P(<= .949) ~.07-.18 (MODEL, verify_B) |
| 2 | No public notebook > .950; all .950 = N + G | **keep** | 0 (context) |
| 3 | Attempts to beat .950 on N were flat or worse | **keep** (minor wording) | avoids a ~1-2 tick loss, MEASURED on other teams, within ~1 tick noise |
| 4 | H (hengck23 MaViT-288) as an extra leg | **revise** (gain inflated by oracle weights; rule risk higher) | ~+.0000 (range -.0003 to +.0006, MODEL); not worth a slot |
| 5 | G adds ~0 to K9N; only in a parent-free kernel at w .10 | **revise** (measured reads say G at .30 on N is -1 tick; "insurance" is weak) | +.0000 (range -.0005 to +.0005, MODEL); include only if free |
| 6 | N2 and M: skip | **keep** | 0 |

## 1. Finding 1: medal lines (REVISE)

**What holds (re-measured from `A_lists/lb_now/...2026-10-08T16_41_24.csv`):**
- 5,513 teams; .950 holds 318 teams at ranks 157-474; .949 holds 99 at 475-573; .951 is 132-156 (MEASURED).
- "Eesh saxena" .946, rank 664, last submission 10-07 01:46 UTC (MEASURED).
- 10% of 5,513 = rank 551 sits at .949; 5% = rank 275 sits inside .950; gold 10 + 0.2% = rank 21 sits at .959 (MEASURED
  ranks, Kaggle medal rule as published).
- Only 7 of the 318 teams at .950 last submitted before N's release (10-07 19:26 UTC) (MEASURED). A's 289/296 is
  consistent.

**What is wrong: tie order is by raw score, not submission time (INFERRED, strong).**
- Inside the .950 band, last-submission dates are near-random against rank: 142 of 317 adjacent pairs are inverted
  (MEASURED).
- Two teams whose last submission predates N (09-10 and 10-0x) sit at ranks 461 and 470, the very back of the band
  (MEASURED). Their .950 must be their earliest-timed entry, so a time-ascending tie-break would put them near the top.
  They are at the bottom, so the hidden full-precision score orders the band.
- Teams last submitting 08:00-17:00 UTC today have a mean rank of 306 vs 332 for 00:00-08:00 (MEASURED): no penalty for
  being late.
- **Consequence:** A's "a late .950 lands at the back of the tie (~474) and may drop out of bronze within 1-2 days" is
  wrong as stated. K9N-a's position depends on its raw AUC relative to the N+G clone block (raw unknown, GUESS
  ~.9500-.9504). If K9N-a's raw is above the clone block, new clones join below it and do not push it down.

**What is overstated: silver.**
- verify_B (`vb_checks3.py`, MODEL) puts K9N-a at E[true] .9503-.9508, P(display >= .951) ~.41-.63, P(display <= .949)
  ~.07-.18. goodpjw2008's MEASURED "P .40 + N .60 = .949" and "N + their competition-only readers at .30 = .948" both
  argue r(F, N) is nearer .90-.93 than .89, i.e. the lower half of that range.
- Silver (rank 275) is inside the .950 band, so a .950 display needs a raw above roughly the top 118 of 318 tied teams
  (MEASURED count, position GUESS).
- Public medals are not final medals. On private, the hundreds of near-identical N+G clones move together; what matters
  is whether K9N-a beats that block on private (unmeasurable before 10-22).

**Corrected:** gain +.004 public (MODEL, display .950-.951); P(public bronze after submit) ~.8, P(public silver)
~.3-.5 (GUESS from verify_B's display distribution). Action unchanged: submit K9N-a first on 10-10 after 05:30 IST;
keep clean K9 as the rule-risk hedge. Rule risk: OAI as stated (host 09-21 conditional yes; NDA "Institutional Approval
Required: No"); geo-access (rule 2.6.b) is the residual open question, no host reply yet (MEASURED,
RULES_AND_HOST_COMMENT_10-08.md).

## 2. Finding 2: no public notebook above .950; .950 = N + G (KEEP)

- Re-parsed `A_lists/refresh2130/*.json`: 618 unique notebooks; max bestPublicScore .950 (11 notebooks), then .949 x5,
  .948 x1 (rian55 apex-v5) (MEASURED). Exactly A's table.
- Spot-checked code: evgendvorkin Apex weights logged in its output (Baker's .35 ... Synovitis .03; mean .1617,
  MEASURED arithmetic); kozykappa "previously gold-selected 0.22" incumbent, N224 crop .08/.16 tested on gold58
  (MEASURED); ranjeet258 "224-crop CoAtNet weight inside the CoAtNet side = 0.3" (MEASURED); goodpjw2008 table (N .949;
  +stack 40% .949; +own readers 30% .948) (MEASURED).
- Fresh dataset list (16:50 UTC): no new model dataset since hengck23's (05:08 UTC) (MEASURED).

## 3. Finding 3: attempts to beat .950 (KEEP, minor wording)

- prvsiyan rows 56955785 / 56957785 = .948 with N .70 / G .30 (MEASURED, notebook markdown lines 23-24).
- rian55 apex-v5 OPTIMAL_WEIGHTS (Baker's .48, Cont .36, LOA .32, ...) scored .948 (MEASURED).
- Wording fix: gold58 is **selection-reused** for N ("Gold58 was reused for selection", nartaa dataset text,
  MEASURED), not trained on. It is still biased for N, so the recommendation (no per-label gold58 routing) stands.
- Caveat: each of these reads is 1-2 ticks, and goodpjw2008 states run-to-run variation of ~1 tick for the stack
  (MEASURED claim). The "avoid -.001 to -.002" is directionally right, not precise.
- "Keep any G weight <= .15" is an arbitrary cap; G is not in K9N, so it only matters for a parent-free kernel.

## 4. Finding 4: H, hengck23 MaViT-288 (REVISE)

- MEASURED and correct: bestPublicScore .935, notebook created 10-08 04:53 UTC, dataset hengck23/hengck23-rnsa-knee
  MIT, empty description, 73.8 MB, notebook evaluates on `train_golden58.csv` in local mode.
- **Gain is inflated.** `A_blend_model2.py` picks best_wX per scenario in hindsight (oracle weight), and the
  min/median/max is over an unweighted grid of GUESS correlations, not a probability distribution. The binormal model
  is one-signed optimistic by +.0000 to +.0009 (verify_B, MEASURED vs reads). At a fixed w .10 the model gives
  +.0006 only at r_XN .85; goodpjw2008's measured "N + competition-only readers at .30 = .948" implies r(N, a
  competition-only reader) ~.88-.95 (verify_B), where the model gives ~+.0000. Net after optimism: ~+.0000 (MODEL).
- **Rule risk is higher than "unknown".** hengck23 named KJD-25K in thread 745861, and F found KJD-25K is KneeCoT,
  which the host banned in 734109 (MEASURED, F_intel_top_solutions.md). He also posted "OAI changes the competition"
  at 12:42 UTC, after H was published (05:08 UTC). Training data is unstated. Using H without a written provenance
  statement risks a banned-data component (rule 2.6).
- Cost (~0.5 day + a commit + a slot) for ~0 expected gain before the 10-20 freeze: not worth it.

## 5. Finding 5: G (REVISE)

- Licence Apache-2.0, competition-only, .929 alone: MEASURED (dataset metadata).
- Two measured reads put G at .30 on top of N at **-1 tick** (prvsiyan .948; goodpjw readers .948), and the .950
  recipes use G at .16-.22 (MEASURED). With F already in the blend (same family: ConvNeXt-T 2.5D, report labels),
  G's marginal value is ~0 and could be slightly negative (MODEL: A's own scan gives +.0000 median in its band).
- The "insurance against high r_FN" argument is weak: if r_FN is high, G (same family as F) is also highly correlated
  with N, so G does not rescue that scenario.
- Runtime: ~20-30 min on 2 T4 (MODEL from author's claim); adds a pylibjpeg wheel dependency. In a parent-free kernel
  with ~6 h headroom it is affordable but adds failure surface.
- Corrected: +.0000 (range -.0005 to +.0005, MODEL). Include in K10FNG only if it is free to wire; never in K9N.

## 6. Finding 6: N2 and M skip (KEEP)

- ranjeet258 N224 at .30 inside the N side = .950, same as without (MEASURED). kozykappa gold58 rejected crop .08/.16
  (MEASURED, but gold58 is selection-biased for N). M has no standalone score (MEASURED). Agree: skip.

## 7. Levers this dimension MISSED

1. **Sub-tick readout from rank inside a tie band (zero cost).** Because the band is ordered by raw score (sec 1), our
   rank inside .950 after K9N-a lands tells us whether its raw beats the N+G clone block (a contiguous run of
   near-identical teams). A later variant (e.g. K9N with frac75 or TTR320 members) that raises our rank inside the
   band has a higher raw score even at the same 3-dp display (INFERRED; Kaggle shows only the team's best entry, so
   this only detects improvements). Record rank and the neighbouring teams after every submission. This gives ~4-10x
   finer resolution than the 3-dp display for choosing the final picks.
2. **Submission timing does not matter for ties (INFERRED).** There is no need to rush a .950 to "get ahead of the
   pile"; order Saturday's slots by information value instead.
3. **Trap to log: gmhost/knee-corpus-crop-320-part1/2** (MEASURED metadata: CC0, empty description, `all_vols.npy`
   22.6 GB, `all_ids.npy` 154 KB, created 10-08 08:51 UTC). A CC0 "knee corpus" of that size with no provenance may be
   re-shared OAI or competition volumes; either would be an NDA or MIRA redistribution problem. Do not use (GUESS on
   contents; not downloaded).
4. **Parent provenance audit is still open** (RULES file item: "worth one CPU-only read of its dataset list before the
   final picks"). A did not check the parent's 15 datasets for a KneeCoT/KJD-25K component, even though it already
   pulled goodpjw2008's copy of the same stack. One read-only pass before the final picks.
