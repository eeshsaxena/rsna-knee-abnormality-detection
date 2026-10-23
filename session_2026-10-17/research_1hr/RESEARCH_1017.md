# RESEARCH_1017: RSNA Knee, 1-hour research merge (10-17 ~22:15 IST)

Read-only merge of six lenses in this dir: forum.md, notebooks.md, lb_private.md, timing.md, levers.md, metric.md. Nothing was submitted, committed, pushed, versioned, posted or voted; no box, PREREG or scheduled-task change. Every claim is tagged MEASURED / CLAIMED / MODEL / GUESS with its lens file in brackets.

## 1. Verdict for Eesh

There is no new lever worth a slot. All six lenses (forum, public notebooks, leaderboard, timing, our own kernel, the metric) say the same thing. HW 57262115 already beats every public .958 notebook. No top team has shared anything we can copy. Every knob left is a public-LB coin flip worth under 0.0005 on private. Tonight's probes will not decide our medal. What decides it is whether HW stays above the ~600-team block of heliosli copies just below us in the .958 band. S6fall .2 is the only thing that separates us from that block, and both final picks already have it. The plan: let the three A16 probes read tonight (keep a poller awake), apply A15 as written, and spend no more slots unless the hosts rule on external data. On 10-22, read the forum for host posts before the 18:00 IST cutoff, then select both picks by hand: A = API best-id, B = R6 57248084. Two risks remain that no pick can hedge. One is falling off the copy-block cliff to no medal, roughly 10-30% (MODEL). The other is a late external-data ruling, at most 3% (GUESS), because both picks use nartaa's OAI weights and heliosli's anatomy.pt.

## 2. Top 10 leads, ranked by expected PRIVATE-LB value

**1. Keep S6fall .2 in both final picks; reject any pick without it.** Highest value.
- Evidence: MEASURED [lb_private.md, metric.md]: .958 band = r142-994 (853 teams); silver cut r307 and bronze r614 both sit inside it; a ~608-team copy block at r388-995 (long exact-tie runs, all best sids after the heliosli publish). Gold58: HW .9369 vs clone-like no-S6 .9336 (+.0033, sd .0016). MODEL: P(HW private > clone) .87-.90 with gold58, .62-.67 on the public bit alone; P(bronze or better) is about P(beat the block).
- Cost/risk: none (decision rule). IF anyone proposes a no-S6 entry or a public .958 copy as a pick THEN reject it.

**2. Eesh selects both picks by hand (A15 pair).**
- Evidence: MODEL [lb_private.md] mc.py base: P(silver) A .72-.74, B .62-.64, pair .76-.78; B adds +9 pts if the heliosli weights are public overfit, 0 if real, never a loss. CLAIMED rule 1.c [timing.md]: if nobody selects, Kaggle auto-picks the top 2 public, which would likely pair HW with another heliosli-weights .958 (GUESS) and drop the hedge.
- Cost/risk: none. IF the A15 read is done by 10-22 18:00 IST THEN Eesh selects A + B before 10-23 05:29 IST (10-22 23:59 UTC).

**3. Keep a poller awake tonight so each A16 completion gets snapshotted.**
- Evidence: MODEL [timing.md] read windows wr45 23:48-00:03 IST, s615 01:22-01:37, a25 02:58-03:13 (late flags 00:48 / 02:22 / 03:58). Completions land ~1.5 h apart and the best-id is cumulative, so a missed snapshot can hide an earlier read. CLAIMED [HANDOFF]: the laptop has slept through scheduled tasks before.
- Value: keeps all 3 reads (information, not score). Cost: one awake fetch_api_lb.py loop (team submissionCount 85 -> 88). IF wr45 is still PENDING at 00:50 IST THEN assume a scoring queue and use the 3 h buffer plan (section 3).

**4. Apply the A16 reads as written; no combos and no follow-ups on the same knobs.**
- Evidence: MODEL [metric.md]: paired public sd 3-7e-4 per probe; P(private > 0 | public positive) .63-.75; |E private| <= ~4e-4 either way, P(loss < -5e-4) <= .03. MEASURED gold58 priors are slightly negative: wr45 -.0008 (sd .0013), s615 -.0016 (sd .0011); a25 can't be checked (no r224 gold preds).
- IF a probe moves the best-id THEN A = that probe and B stays R6 57248084 (A15/A16). IF none moves THEN A stays HW. Either way: no new W_RAPTOR / W_S6F / A224 step, and never stack two winners.

**5. Read the forum for host and staff posts on 10-22 ~17:00 IST, before the last A15 read.**
- Evidence: MEASURED [forum.md]: no host or staff post since 10-14 19:56 UTC. The 10-17 questions on nartaa OAI weights (750257), anatomy.pt SKM-TEA (750258), R7 fastMRI (750259) and RadImageNet (750260) have no answers and sit at -19/-20. Host 744056 (10-14): publicly shared weights are eligible for awards if they are listed.
- Cost: ~2 min, fr.py + msgs.py (host ids in forum.md). IF a ruling lands on 750257/750258 THEN go to lead 6. IF it is only on 750259/750260 THEN do nothing, because neither is in our picks (MEASURED: the R6 manifest has no fm_aux).

**6. External-data contingency (nothing to do unless the hosts rule).**
- Evidence: MEASURED [forum.md]: both picks attach nartaa OAI CoAtNet weights plus heliosli anatomy.pt (SKM-TEA), and use no fastMRI or RadImageNet. The only clean public model is the dreaddevelopment Raptor baseline at ~.936-.938 (CLAIMED, 737696), below bronze. GUESS: <= 3% chance a ruling hits both picks (top-25 teams also use OAI, CLAIMED).
- IF a host or staff post says those weights are not medal-eligible THEN tell Eesh at once. A Raptor-only hedge would need a new kernel, gate, slot and PREREG amendment, about 6.5 h to read. ELSE keep the picks.

**7. Re-fetch the LB at every read and redo the cut math.**
- Evidence: MEASURED [lb_private.md]: HW slid r135 -> r171 in 25 h (22 teams reached .959+ and 14 passed us inside the band). N is still growing after entry closed (6,047 -> 6,144 in 40 h), so the brief's fixed-N premise is wrong. MODEL: N_final ~6,300, silver cut ~r315; HW public r250-350 by 10-22 if it stays our best; the private silver cost is only ~1-2 pts per 100 teams that pass.
- IF > 150 teams pass HW by 10-21 THEN re-run lens_lb_scripts/mc.py with that K. IF a probe moves the best by a full display step (.959) THEN treat it as real (P ~.9+, MODEL).

**8. heliosli V2 weights: the only public variant we haven't tried (optional, EV ~0).**
- Evidence: MEASURED code [notebooks.md, levers.md]: heliosli 0-9586 v6 = V2 table {Medial OA .5, Lateral OA .15, PF OA .65, Synovitis .85, Baker's .30} plus the goodpjw reader, no S6fall. Its team ranks r332, below our r171. The ".9586" title is CLAIMED (the API shows only 3 decimals). GUESS: ~0 +- 1e-4 public. It is tuned on the public LB, beyond the A11 exception.
- Cost: 1 slot, a 1-line cell 14 change, a PREREG amendment and Eesh's go. IF Eesh explicitly wants one more weights probe on 10-18 THEN run this on the A16 winner. ELSE skip (recommended).

**9. Rebuild hygiene: any kernel started from a public slug must keep our private R6 copy.**
- Evidence: MEASURED [notebooks.md]: heliosli r6-frozen current v2 is R7 under the R6 name (pinned v1 20611930 is R6). lavinwins gold-medal current v6 is a .943 pipeline. R6 v1 fingerprint: json sizes 4120/5010/6857/24698, manifest sha 6508ce69; R7 is 5211/5003/6858/24673.
- IF any build starts from a public slug THEN check the fingerprint and sha before the push. Value: avoids a silent R7 swap.

**10. Keep the disclosure list ready; we never post it ourselves.**
- Evidence: MEASURED [forum.md]: host 744056 asks award winners to list the shared weights and their source datasets; 750236 asks whether medal positions get checked (no answer). The list is in forum.md section 3.
- IF Kaggle or the hosts ask medal teams for disclosure THEN Eesh posts it himself. Value: compliance only.

## 3. Scoring and timing facts, and the slot plan

- MEASURED comp.json [timing.md]: usesSynchronousReruns True, maxGpuRuntimeMinutes 540, maxDailySubmissions 5, numScoredSubmissions 2, deadline 2026-10-22T23:59Z, leaderboardPercentage 30, scoreTruncationNumDecimals 3.
- Picks are not re-run: the private score is fixed when the entry is scored (MEASURED flag + CLAIMED FAQ, quoted from memory). So no runtime, canary or GPU-noise risk remains for HW / R6, and runtime-safety edits to them are pointless.
- The LB and CSV show 3 decimals, but the API best-id compares at full precision (MEASURED: HW displaced R6 at .958) [metric.md]. Ties go by submission time (CLAIMED 749570). One display step = .012 on a single label (MEASURED identity).
- Noise: public split sd ~.003 (CLAIMED nartaa 749359); paired delta sd for a one-line change is 3-7e-4 public, private ~.65x that (MODEL [metric.md, lb_private.md]). Fracture (~62 private positives) and Synovitis (~115) each carry ~25% of private macro variance (MODEL).
- LB 10-17 16:24 UTC, MEASURED [lb_private.md]: N 6,144; gold cut r22 .963; .962 from r30, .961 r47, .960 r65, .959 r94-141, .958 r142-995; silver r307, bronze r614; HW r171; B (R6) bracket r298-322; teams at .959+ grow ~22/day.
- Latency, MEASURED bounds [timing.md]: HW <= 6h33, R6 6h03-6h15, hw3 <= 6h03, S6F 6h36-6h40, R7 late 6h37-7h26. Weight-only HW-class probes take ~6.0-6.25 h (MODEL). No scoring queue seen 10-14 through the morning of 10-17; evening-UTC submits not measured yet.
- Runtime margin: HW >= 2h27 under 540 min (MEASURED). Guards: DEADLINE 8.3 h, S6F start G 5.75 h / deadline 8.2 h (MEASURED); the canary did not fire in HW's scored run. A new leg adding > ~1.5 h before S6F would trip the S6F guard (MODEL).
- Commit queue on 10-17, MEASURED: s615 queued 2h25, a25 ~2h58; commits run 9-11 min. Push each kernel >= 4 h before its submit.
- Status 16:31 UTC, MEASURED: wr45 57310758, s615 57313454 and a25 57316395 PENDING; the 10-17 UTC day is 5/5 used [HANDOFF].
- Submissions reset daily at 05:30 IST (MEASURED). When the weekly GPU quota resets is unclear (HANDOFF ~05:50 IST vs timing.md GUESS Sat 10-24); each commit costs ~10 min, so it does not bind.
- Brief corrections, MEASURED [forum.md, levers.md]: the forced flip applies to the R6 family only (cell 8); CoAtNet 384 runs nartaa's anatomical mirror (cell 4); r224 runs plain.

Slot plan (nothing auto-submits; every entry needs Eesh's explicit go plus a PREREG amendment written before the commit):

| Day (IST) | Slots | Default | Optional, only with go + amendment |
|---|---|---|---|
| 10-17 night | 0 left | Read wr45 ~23:50, s615 ~01:25, a25 ~03:00; poller awake | none |
| 10-18 (opens 05:30) | 5 | 0 submits; apply A15 after the a25 read; update HANDOFF | at most 1: heliosli V2 on the A16 winner (lead 8), kernel pushed >= 4 h before |
| 10-19, 10-20 | 5 + 5 | 0; one forum + LB check per day | hold for the external-data hedge (lead 6) only |
| 10-21 | 5 | 0; push any 10-22 candidate by 22:00 IST | last evening submit 23:00 IST (it reads ~05:15 IST 10-22) |
| 10-22 | 5 | 0; ~17:00 host-filtered forum read; A15 LB read before 18:00 IST; Eesh selects both picks before 10-23 05:29 IST | submit window 05:30-10:15 IST (08:45 if a queue showed); nothing after |

## 4. Final-pick recommendation

Keep A15 exactly as written. **A = API best-id at the last read before 10-22 18:00 IST** (today HW 57262115; it changes only if an A16 probe moved the best-id). **B = R6 57248084** (uniform weights + S6fall .2).
- Both carry S6fall .2, the only measured separator from the copy block (MEASURED gold58 +.0033 over the clone; MODEL P .87-.90).
- A is the stronger silver bet: gold58 HW minus uniform = +.0024 (sd .0023), mostly from Lateral OA +.0387 (MEASURED [metric.md]); public r171 vs B at r298-322 (MEASURED).
- B hedges the one open question: heliosli's weights may be tuned on the public LB (CLAIMED 749570; heliosli gave no method, 748181). The two private margins correlate ~.83; the pair's P(silver) is .76-.78 vs .72-.74 for A alone, and .56 for the pair under a skeptic S6 prior (MODEL mc.py).
- An external-data-clean pick can't medal (~.937, CLAIMED), so a third-axis hedge would cost a medal-level pick for nothing.
- Never pick a public .958 copy, a no-S6 entry, or two heliosli-weights entries. If an A16 probe becomes A, B is still R6.

## 5. What we should NOT do

- Do not build "S6fall OFF" (HW minus cells 16-18), even if s615 reads positive. levers.md proposed it with conditions, but metric.md and lb_private.md show it would be a heliosli clone that lands in the copy block (gold58 -.0033 vs HW, MEASURED). If it is ever built, never use W_S6F=0.0, which fires the ACL canary (MEASURED [levers.md]).
- Do not rebase on or copy any public .958 notebook. Every author's team ranks below HW: Heliosli r332, goodpjw r402, kiyoc r887 (MEASURED [notebooks.md]).
- Do not add the goodpjw 2.5D reader (~0 gain, new env plus runtime risk), r224 mirror (FLRM read negative), R7 at .4 (non-move), S6fall flip TTA (no code path; trained without flips) or the S6fall side model (train/test mismatch risk) (MEASURED/CLAIMED [levers.md]).
- Do not tune any weight on gold58, or keep stepping W_RAPTOR / W_S6F / A224: HW sits near a local optimum, and goodpjw's gold-tuned weights lost .001-.003 (CLAIMED [notebooks.md]). Do not combine two probe winners.
- Do not reopen closed levers: KC family, our S6 seeds, R7-teacher student, K14 fleet, r224 mirror, R7 late leg, S6fall .3/.4, R6+R7 average.
- Do not make runtime-safety edits to the picks (there is no rerun), and do not leave the picks to auto-selection.
- Do not submit after ~10:15 IST on 10-22 (it would be unread under A15), and submit nothing without Eesh's go plus an amendment.
- Do not post, vote or comment on the 10-17 external-data threads, and do not post the disclosure list ourselves.
- Do not use laptop compute or the box for any of this; the A16 reads need only API polling.

Sources: forum.md, notebooks.md, lb_private.md, timing.md, levers.md, metric.md (this dir); raw data in forum_raw/, pulled/ (untrusted, never executed), lens_lb_scripts/, metric_work/, api_lb_1017T_lens.json; HANDOFF.md top block 10-17 15:06; PREREG_10-15.md A15/A16.

## Critique (adversarial review, 10-17 ~22:50 IST)

Read-only. For this review I reopened the cited sources: api_lb_1017T_lens.json (16:24 UTC, 6,144 rows), metric_work/{g58.py, g58_out.json, scale_out.json}, forum_raw/{host_posts.txt, all.txt, new/msgs_7502*.json}, session_2026-10-15/r6v1/raw/manifest.json, the HW kernel (cells 7, 14-18 and kernel-metadata.json), PREREG_10-15 A15/A16, PROGRESS_14/16, and the pulled public notebooks. I read those notebooks as JSON with python -I and never executed them. Nothing was submitted, committed, posted, voted or run on the box.

**Net:** the plan holds: keep A15, spend no new slots, select A and B by hand. Three findings change wording or numbers but not the plan, and one adds a zero-cost action.
1. The "608-team copy block" is a staircase of >= 10 separate exact-tie clusters, not one block. So the "cliff to no medal" is softer than 10-30%.
2. Gold58's +.0033 for S6fall is about 3x larger than the public margin allows (< .001). Treat it as evidence for the sign only. P(HW > clone on private) ~.80-.93, base ~.85.
3. S6fall is not the only thing separating us from the copies. The J decoder is a second, tail-only one.
4. New action: pre-select A + B in the Kaggle UI now, verify the selection, and re-select only if A16 changes A.

### C1. Top 5 leads checked against their sources

**Lead 1 (S6fall in both picks): stands. Fix four overclaims.**
- Cut numbers check out. MEASURED: the snapshot's own `medal` field gives GOLD r1-22, SILVER r23-307, BRONZE r308-614. The .958 band is r142-995 (854 teams). The 16:21 snapshot has r142-994 (853), and the draft mixes the two.
- Ranks inside a display band follow the unrounded score, not submission time. MEASURED: of the 824 band teams below HW, 206 have an earlier best sid than HW. Of the 29 above HW, 20 have a later one. So r171 is a real ordering.
- OVERCLAIM, "~608-team block of heliosli copies just below us". MEASURED:
  - r388-995 contains 136 increasing-sid runs. 390 teams sit in runs of >= 5; a random shuffle of the same sids gives about 22.
  - Those runs form >= 10 separate clusters spread over r523-943: r649-689 (n 41), r787-814 (28), r817-844 (28), r562-588 (27), r708-733 (26), r593-613 (21), r523-542 (20). Another ~220 teams have scattered distinct scores.
  - Each cluster is a different unrounded score, so a different recipe or version (heliosli v1, goodpjw v25-27, heliosli V2, R7-under-R6-name copies, ...).
  - The block also starts 217 ranks below HW, not "just below".
  - 99.8% of C's best sids are > 57240000 (about 10-15 18:00 UTC by interpolation), so "after the heliosli publish" holds. The claim that every W team's sid is after 57251394 is only 90.7%.
- OVERCLAIM, "gold58 +.0033 MEASURED". The numbers reproduce (scale_out.json: .00328, sd .00159), but three caveats are missing:
  - (a) It is a proxy blend: R7 OOF stands in for R6, and the Raptor slot is crop320 only, with no r224.
  - (b) It is not independent of S6fall. nartaa says the JEV HYBRID / blendjev targets were tuned on gold58 by their author, and the CoAtNet SWA epochs were ranked on gold58 (CLAIMED, forum_raw/all.txt, 749359/750180). Gold58 was kept out of gradient training (pool 4,349 = 4,407 - 58, MEASURED PROGRESS_16 PX_BUILD), so this is target and selection leakage, not train-set leakage.
  - (c) The magnitude does not fit the public LB. HW and the clones both show .958, so the public HW-minus-clone margin lies in (0, .001) (MEASURED). A true +.0033 would sit 3-4 paired public sd above that.
  - MODEL: combine gold58 (sd widened by 1e-3 for the proxy mismatch) with the public margin D of 3-8e-4 (paired sd 6-8e-4). Result: posterior mean +6e-4 to +1.2e-3, P(true > 0) .84-.96, P(private > 0) .80-.93, base ~.85. Same conclusion, smaller edge.
- OVERCLAIM, "S6fall is the only thing that separates us". lb_private itself says A - C = S6fall .2 + the J decoder.
  - MEASURED code: heliosli blend-mine-in v1 and lavinwins gold-medal v5 contain no pylibjpeg or gdcm. goodpjw .958 and heliosli 0-9586 install pylibjpeg only into the GP reader's own `--target` env (goodpjw cell 18), not into the R6 / Raptor stages.
  - Our 10-14 test (PROGRESS_14 20:20, MEASURED there): on JPEG Lossless studies the pinned image zero-fills Raptor and drops R7-family series.
  - MODEL: one garbage study costs about .9/n macro (~.002 on public). D < .001 therefore implies about 0-1 such studies in public, and the expected private count is about 0-2.
  - So J is tail insurance that both picks carry, and it is worth the most in the "S6fall fails on private" world.

**Lead 2 (Eesh selects the A15 pair by hand): stands. Add pre-selection.**
- The mc.py figures match the lb_private table: base A .74 / B .64 / pair .78; drift row .72 / .62 / .76; overfit row pair .83 (+9 pts). "Never a loss" is right, since the team score is the max of the two picks. The auto-select rule is CLAIMED (rules 1.c) and correct in substance.
- MISSING, zero cost: **pre-select now.** Kaggle lets a team change its two selections until the deadline (CLAIMED, standard Kaggle behaviour).
  - Select HW 57262115 + R6 57248084 tonight, and re-select only if an A16 probe moves the best-id.
  - This removes the auto-select risk and the overnight risk of the 10-23 05:29 IST deadline.
  - Precedent: in the AI Agent Security comp the official private score was 0 because the wrong notebook was selected (CLAIMED, memory index).
- Give Eesh a selection card, because every candidate shows .958 and the display cannot tell them apart:
  - A = 57262115, `rsna-sub-v224r6hws6fj` v1, submitted 10-16 08:43:59 UTC.
  - B = 57248084, `rsna-sub-v224r6flips6fj` v1, submitted 10-16 00:07 UTC.
  - After selecting, screenshot both checkmarks on the submissions page.

**Lead 3 (awake poller): arithmetic correct, but misranked.**
- The windows are right: 17:48:55 / 19:22:20 / 20:58:50 IST, plus 6.0-6.25 h, give 23:48-00:04 / 01:22-01:37 / 02:58-03:13. The +7 h late flags are 00:48 / 02:22 / 03:58.
- Its private-LB value is zero. A15 takes the API best-id at the last read before 10-22 18:00 IST, and the best-id is cumulative, so pick A comes out the same with or without the intermediate snapshots.
- A missed snapshot loses only R1 bookkeeping: which probe moved the best-id, and the s615 read. The only use of that read, the "S6 off" build, is rejected in section 5.
- Its real job is diagnostic: wr45 still PENDING at 00:50 IST means a scoring queue, which moves the 10-22 submit window to 08:45. Rank it below leads 5-7.
- "submissionCount 85 -> 88" is the detection signal, not a cost.

**Lead 4 (apply A16 as written): decisions stand. Two overclaims.**
- The numbers match g58_out.json: wr45 -.00078 (sd .00133), s615 -.00163 (sd .00106).
- OVERCLAIM, "HW sits near a local optimum" (also in section 5). The gold58 curves are jagged inside the noise:
  - W_S6F .1 = .9355 beats .15 = .9353, and .4 = .9373 is the curve's maximum, above .2 = .9369.
  - W_RAPTOR .6 = .9370 beats .5 = .9369.
  - Every .05 step is <= 1.5 bootstrap sd. The honest reading is a flat curve with no detectable optimum.
- Bias left out: the wr45 gold58 prior leans against wr45 by construction. The Raptor checkpoint was selected on gold58, so its gold58 skill is optimistic. R7 also stands in for R6, while wr45's rationale is that R6 is stronger than R7. metric.md says this; the draft drops it. Treat the wr45 prior as ~0, not negative.
- A probe showing .959 is not "unlikely". MODEL, with prior N(0, 5e-4), public noise 5e-4 and HW's offset inside .958 uniform: P(.959) is about .26 per probe (.13 under the gold58-negative prior), and P(private > 0 | .959) is about .79-.81.
  - So lead 7's "full display step -> treat as real, P ~.9+" overclaims. A .959 is a stronger read than a plain best-id move, but it is still only about 4 in 5.

**Lead 5 (forum check on 10-22): verified.**
- MEASURED from host_posts.txt: the last host or staff post is 10-14 19:56 UTC (Dustin, 748403).
  - There is also a 10-13 21:10 host post (Po-Hao Chen, 736301), but it is an AI-usage survey, not a ruling. forum.md leaves it out, which is harmless; I did not follow its link.
- Thread state: 750257/58/59/60 sit at -19 / -20 / -20 / -20 with 0 replies; 750236 is at -3 with 0 replies.
- HW's kernel-metadata dataset_sources (MEASURED): R6 private copy, nartaa publication-swa (OAI), heliosli orthofoundation (anatomy.pt), nartaa s6fall, glorysing codecs. No R7 or fastMRI dataset.
- The R6 manifest (MEASURED) has no fm_aux / fastmri / oai string, init convnext_small in22k_ft_in1k, gold_in_training 0, pool_studies 4,344.
- Loose end: the manifest's `source_run` is `cnx_r6fm`, and the "fm" suffix is unexplained. GUESS harmless, since run_config has no aux key. Recheck it only if a fastMRI ruling ever lands.
- Add a second forum check right before Eesh selects (10-22 ~18:00-19:00 IST). A ruling posted after 17:00 would otherwise be missed.

### C2. Runtime math
- Correct:
  - HW margin >= 2h27; latency is an upper bound on runtime.
  - The last readable HW-class submit on 10-22 is 10:15 IST (18:00 - 6.25 h - 1.5 h), or 08:45 with a 3 h buffer.
  - A 10-21 23:00 IST submit reads ~05:15 IST 10-22.
  - The deadline 10-22 23:59 UTC = 10-23 05:29 IST.
- Guards verified in the kernel:
  - Cell 7: DEADLINE_HOURS 8.3.
  - Cell 17: G_START_H 5.75, BUDGET_H 3.3, DEADLINE_H 8.2, STOP_MARGIN_S 240, MIN_RUN_S 1200.
  - Cell 18: W_S6F .2, COV_MIN .98, canary column ACL.
  - Cell 15: the r224 canary (Fracture).
- Correction (MODEL): under a uniform slowdown s, the S6F hard deadline binds first, not the start gate. A 6.0-6.25 h run times s must finish by 8.2 h - 240 s, so s < 1.30-1.36; the start gate allows s < 1.35-1.44. COV_MIN .98 also means a slow S6F that covers < 98% of studies fires the canary.
  - For a leg added before S6F, the start gate does bind first, at +1.5-1.75 h, as the draft says.
  - None of this affects the already-scored picks.
- Tag fix: "the canary did not fire in HW's scored run" is MODEL (inferred from the best-id move), not MEASURED. The inference is sound.

### C3. Private-LB reasoning
- Correct: medal cuts (from the API medal field), ranking by unrounded score, N growth, team score = max of the two picks, no re-run.
- Weak point: mc.py treats C as one shared private draw, but the measured structure is >= 10 clusters at different public levels (r523-943) plus ~220 distinct scores.
  - If S6fall's private effect is ~0, HW becomes roughly its own clone cluster and lands among the clusters, not below all 608. The public order among those public-tuned variants is mostly noise.
  - GUESS: HW at clone level lands ~r450-850, below the bronze cut r614 maybe half the time. That gives P(no medal) ≈ P(S6fall fails) x .5-.7 ≈ 5-15%, not 10-30%. The J tail lowers it further.
  - Silver risk is about unchanged, because silver is decided in W and upper C.
- Missing cheap check (read-only, ~15 min): find which cluster is the exact heliosli v1 copy by matching blend-mine-in fork authors to teamName inside the cluster runs. Its public rank is where HW lands if S6fall = 0, and it turns the GUESS above into a MEASURED anchor. Re-run mc.py with C split into clusters only if Eesh wants the number (light CPU).
- The lenses disagree on noise. lb_private's sign table uses sd 1.5-2.5e-4 for a 1-line weight change (P .90 at d 3e-4); metric.md uses 3-7e-4 (P .63-.75). Use the wider metric.md numbers for tonight's reads.
- Lead 7 drops lb_private's upper case: if 80% of the passers' gain is real, A loses ~4 silver pts per 100 passers, not 1-2.

### C4. Missing angles, by value
1. Pre-select A + B now, screenshot the selection, and re-select only if A changes (C1 lead 2). Zero cost. It removes the two failure modes that cost us the AI Agent Security private: auto-selection and the overnight deadline.
2. A selection card with sids, slugs and submit times, because every candidate shows .958.
3. Locate the heliosli v1 exact-copy cluster (C3).
4. Treat the J decoder as a second separator; A15 already makes "J form" mandatory in both picks, so keep it.
5. Read cut positions from the API `medal` field at every read instead of the 5% / 10% arithmetic.
6. Teams that hide their real score with 1 - p submissions can sit below .958 on public and above us on private (CLAIMED colum2131, forum.md sec 5). The ~136-rank cushion to the silver cut absorbs a few dozen of them. The draft never mentions this.
7. A second forum check right before selection on 10-22.

### C5. Minor fixes
- Use one snapshot throughout (16:24: .958 = r142-995, 854 teams; C = r388-995, 608).
- Section 3 "brief corrections": the brief's "R7-family flip" means the R7-family worker's flip applied to R6, which the draft itself confirms. It is not a correction.
- Lead 6's "about 6.5 h to read" needs a new kernel commit first; with 10-17's commit queue (up to 3 h), plan ~9.5 h from a ruling to a read.
