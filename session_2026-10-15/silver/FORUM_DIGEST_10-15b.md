# Forum + public-code digest 2026-10-15b (~15:55 IST = 10:25 UTC). No em dashes.

Merges two read-only lenses run 10:00-10:20 UTC 10-15 (forum lens, notebooks lens) on top of FORUM_DIGEST_10-15.md
(01:35 IST) and SILVER_PLAN.md. Tags: MEASURED (API / page / file data we read) / CLAIMED (a third party said it) /
MODEL / GUESS (ours). Nothing was submitted, posted, voted, forked, merged, uploaded or created. Pulled public code was
read as data only (python -I), never run or imported.
Time note: the notebooks lens wrote "~10:30" and "~10:50 UTC" for its probes; file mtimes say 10:15-10:20 UTC. This digest
uses file times.

## 0. Position and clock (MEASURED, silver/pulled2/forum_1015T1030/api_lb_now.json, fetched 10:19 UTC, 6,034 teams)
- Team best = V224-FLIP-AO + S6fall .2 (57201874), public .957, **rank 160**. Medal cuts on the public LB: gold r22,
  **silver r301**, bronze r603. Margin over the public silver cut: 141 ranks.
- Bands: .960 r48-66 | .959 r67-83 | .958 r84-127 (44, was 36 at 19:46 UTC 10-14) | **.957 r128-304 (177, was 115)** |
  .956 r305-1240 (936). The silver cut sits at the bottom of the .957 band.
- Neighbours: the bees knees (pand999) r128 .957 (69 subs); BliverRigPaDoge r193; Adel Zebiche r220 (76 subs);
  heliosli r224 (52 subs). heliosli's own R6 recipes rank BELOW our S6F.
- Moving to .958 needs about +3e-4 unrounded (GUESS, from our band position ~33 of 177).
- Pending reads: 57218523 FLIP + S (~16:30-17:30 IST), 57222884 composite S6F + S (~20:30-21:30 IST).
  All 5 slots of the 10-15 UTC day are used. Next slot: 10-16 UTC day slot 1 = 00:05 UTC (05:35 IST).
- Deadlines (MEASURED, comp metadata): kernelsPublishingDisabledDeadline = teamMergerExplicitDeadline =
  prohibitNewEntrantsExplicitDeadline = **10-15 23:59 UTC (10-16 05:29 IST)**. Final 10-22 23:59 UTC.
  numScoredSubmissions 2, maxGpuRuntimeMinutes 540, usesSynchronousReruns true, requiresIdentityVerification true,
  leaderboardPercentage 30 (public ~390 studies, private ~910). No host text yet on private split, reruns, final selection.

## 1. Top 10 actionable leads, ranked by expected private-LB value for us

### 1. R6 bundle fe055406 IS obtainable: build S6F-R6 (one change: R7 -> R6, flip forced)
- Evidence: MEASURED (authenticated API, read-only, in memory only): dataset heliosli/rsna-knee-r6-frozen-fe055406
  VERSION 1 (version id 20611930) still lists 9 files dated 10-12 02:42 (E_lrprior_s0_f0..f3.pt 200,379,391 B each,
  side_cls.pt, manifest.json 4,120 B, fluid_stacker, state_stacker, state_spec). v1 manifest sha256 = 6508ce6921eda986...
  = the R6 manifest heliosli's notebook expects; v2 (current, id 20731557) = c4bb0f73... = R7. Manifest: cnx_r6fm,
  convnext_small 384/crop 352, 4 folds, same .pt sizes and trainer as R7, so runtime equals R7 (MODEL); no flip_tta key.
  This REVERSES SILVER_PLAN lever 9 ("CLOSED, not obtainable"); the earlier 404 was most likely wrong version syntax (GUESS).
- Why it helps: heliosli/0-9576-latest-public-dataset-score-in-one-click (kernel 138670458, v1 only, pins 20611930) =
  R6 + forced flip + the same Raptor half; public .957 (MEASURED display), ".9576" is CLAIMED from the title only.
  Our V224-FLIP (R7 + flip, same Raptor half) shows .956, so R6 beats R7-flip as a BASE (MEASURED display). But heliosli's
  team best (max over his R6 recipes) is r224, below our S6F r160 (MEASURED), so the gain ON TOP of S6F is
  +2e-4 to +6e-4 public, P(>0) ~.6 (GUESS, selection-biased: he published his best version).
- Feasibility: MEASURED cell diff: our s6f/kernel_v224flips6f and heliosli 0-9576 share byte-identical embedded sources
  (compact_infer 8401812c, compact_train 04df7e2e, anchor_bank_v2 e564807a, probe_dual a6155eb0, probe_infer a6cc3144,
  px_cache b13e38a7, side_common bbcc5d02) and an identical Raptor-384 cell. Proof: s6f/celldiff_* style diff of
  s6f/kernel_v224flips6f/rsna_sub_v224flips6f.ipynb vs silver/pulled2/notebooks_1015T1010/heliosli__0-9576-.../.
  heliosli's COMPLETE T4 run proves forced flip works on R6 with this worker.
- Licence: dataset licence field "Other (specified in description)", description names none. Host ruling 744056: public
  weights are eligible; list them and their source data in any write-up (GUESS: a private copy for our own use is fine).
- IF Eesh OKs the ~823 MB download, THEN (exact steps):
  1. `kaggle datasets download heliosli/rsna-knee-r6-frozen-fe055406/1 -p C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-15/r6v1/raw --unzip`
     (fresh empty dir; kaggle CLI 2.x takes owner/slug/VERSION; GUESS on syntax, check the file count = 9).
  2. sha256 every file vs the v1 manifest: f0 59cc88a1, f1 3d9dd8d6, f2 8d32fa2c, f3 ea147aac, side_cls ab623e2b,
     state_spec 3914242b, fluid_stacker 9fbd6f65, state_stacker b05147bb, manifest 6508ce69. Any mismatch = STOP.
  3. Main creates a PRIVATE dataset, e.g. eeshsaxena/rsna-knee-r6-fe055406-v1 (same 9 file names, flat).
     Never build from heliosli's slug through the API: a push attaches the LATEST version (= R7), and swap_builder.py
     fails on drift by design (correct).
  4. Copy the build source -> kernel_v224r6flips6f. In cell 6 change exactly 2 lines:
     `BUNDLE_ID = '8c7b666e27dc0e8f'` -> `'fe055406fad6a12f'` and `C_SLUG = 'rsna-knee-compact-r7cnxfsss'` -> `'<our slug>'`.
     In kernel-metadata.json swap `heliosli/rsna-knee-compact-r7cnxfsss` -> `eeshsaxena/<our slug>`. Keep
     `FLIP_TTA_FORCED_ON = True` (cell 7) and `machine_shape: NvidiaTeslaT4`. Per the SILVER_PLAN ladder rule (AO + J from
     10-16), prefer building it from the J twin `rsna-sub-v224flips6fj` (jpeg/overlay/r2, J_CHECK 16/16) so the entry is
     pick-eligible in J form at once; J is score-neutral on uncompressed (MEASURED), so it is still one score change.
  5. Gates: (a) bundle_id assert passes; (b) the R6-flip + Raptor stage on the visible studies equals heliosli 0-9576
     submission.csv to <= 1e-6 (`kaggle kernels output heliosli/0-9576-latest-public-dataset-score-in-one-click`, small,
     needs OK; first confirm it is still at version 1); (c) S6fall rows equal the S6F v1 commit rows; (d) runtime review
     equals S6F (median ~6.3-6.6 h, p99 <= 8.0 h); (e) amendment written before submit.
  6. Do the private dataset + private kernel push BEFORE 23:59 UTC tonight as a hedge (GUESS: the lock only blocks
     PUBLIC kernels, but this costs nothing). Submit in a 10-16 slot: slot 1 (05:35 IST) if KC1 is not ready, else slot 2.
- IF composite 57222884 becomes the team best first (read ~20:30-21:30 IST), THEN put R6 on the composite kernel
  (s6f2/kernel_v224flips6fs20) instead, still as one change, after a runtime review (+0.29 h student, MEASURED microbench).
- IF heliosli deletes dataset v1 before step 1, THEN the lever closes again; do not try mirrors (all public "r6-frozen"
  copies are R7, see 3.4).

### 2. Final-selection hygiene (protects the whole result)
- Evidence: numScoredSubmissions = 2, requiresIdentityVerification = true (MEASURED). hoshinoyumemi CLAIMS: when no finals
  are selected, the two auto-selection markers on the Submissions page = your two highest UNROUNDED public scores; exact
  ties go to the earlier submission. Matches Eesh's 11:20 IST screenshot (FLRM and FLIP both had bars). Memory: the AI
  Agent Security comp scored private 0 because the wrong notebook was selected.
- IF two or more of our entries tie at .957 (S6F, composite, R6, J twins), THEN Eesh screenshots the Submissions page and
  we use the 2 markers as a free order read next to the API best-id (api_frozen.py). The API best-id wins on conflict.
- THEN by 10-22 21:30 UTC (03:00 IST 10-23) Eesh EXPLICITLY selects B and B' per SILVER_PLAN sec 4 / HANDOFF item 9
  (both scored, J form, flip AO, canary-free) and sends a screenshot of the 2 ticked rows; Claude re-reads it.
- IF Eesh's Kaggle account shows an identity-verification banner, THEN he completes it himself well before 10-22.

### 3. Lock-night watch: react only to a public .958+ or a public S6fall blend
- Evidence: MEASURED, 9 teams jumped overnight from <= .956 to .958-.960 (zzc .956 -> .960; z7777 .955 -> .960 one sub
  ~00:08 UTC; 没落者 .955 -> .958 ~00:06 UTC; kreininmv, Hyunwoong Kim, ikevayansky, 石澤凰雅, Ben Pepper, eierapfelxx
  -> .958). The highest public notebook is still .957 (3 notebooks, watch 10:03 UTC). Cause unknown; GUESS: own R6 + R7 +
  S6fall + flip combinations or private sharing (z7777 / 没落者 two minutes apart suggests a common source).
  S6fall: nartaa/rsna-knee-s6fall-weights scriptCount 1 (only the .950 solo), 645 downloads (MEASURED); no public blend
  uses it, so it is our differentiator at r160.
- THEN keep the watch loop (pid 18316, ~1.5 h) running through 05:30 IST 10-16.
- IF watch.py raises SCORE_APPEARED / KEY_AUTHOR_NEW_NOTEBOOK with public >= .958 and all inputs public, THEN build its
  one-change swap the same UTC day (10-16 slots 2-5).
- IF a public S6fall + R6/R7 blend >= .957 appears, THEN expect a copy wave into .957/.958 within ~1 day; move the R6
  swap (lead 1) and KC1 up; re-check scriptCount once after the lock.
- IF nothing public >= .958 by 00:30 UTC 10-16, THEN assume no copyable .958 exists; drop the watch to every 4-6 h
  (new VERSIONS of existing public notebooks stay possible after the lock, CLAIMED) and keep own-work levers.
- Do NOT run watch.py's RELEASE_CANDIDATE "next: swap_builder.py heliosli/..." suggestions; they fail on drift (correct).

### 4. Team merge / friend's GPU: last chance tonight 23:59 UTC (05:29 IST 10-16). Eesh's call only
- Candidates (MEASURED ranks): pand999 "the bees knees" r128 .957, 69 subs, top of our band; draft already at
  session_2026-10-15/teamup/DRAFT_pand999.md. Adel Zebiche r220 .957, 76 subs, topic 749426 (21:19 UTC 10-14) asks for a
  teammate with a strong single CNN; CLAIMS DINOv2 ViT-S fold-0 .952 public / .934 gold58, 224 px 2D slices, own labels,
  no external data, best blend .957. Only reply: "Is this legit?". Numbers unverified.
- Value: a merge is the only route to a truly different model family before 10-22 (GUESS +0 to +2e-4 private per added
  leg). Submission totals 76 + 69 = 145 or 76 + 76 = 152, below the usual cap (GUESS ~5/day x days).
- The friend's 9 h 40 m Kaggle GPU can only be used if the friend is a TEAMMATE (no private sharing outside a team), and
  entry + merge both lock at 23:59 UTC tonight (MEASURED). After that the friend route is closed for good.
- IF Eesh says yes, THEN he sends the message and accepts the merge himself before 23:59 UTC (Claude never sends
  messages). ELSE ignore; the 10-08 no-merge rule stands.

### 5. Kaggle GPU quota + stuck submissions: protect the final commits
- Evidence: 748403, staff (Dustin) cleared a submission stuck > 24 h only ~3 days after the report (MEASURED, 10-11 20:11
  -> 10-14 19:56 UTC). 730709 (noway, 22:08 UTC 10-14) CLAIMS 0 GPU minutes blocked saving the notebook needed to submit.
- THEN keep >= ~2 h of Eesh's Kaggle GPU quota for the final J-form commits (GUESS: weekly reset Saturday 00:00 UTC).
  Last risky candidate in by 10-21 06:00 UTC (11:30 IST).
- IF a final candidate sits in "running" > 12 h, THEN resubmit the same kernel version once (Eesh) and keep the
  other pick unchanged; do not wait on staff.

### 6. P100 trap on CLI pushes (CLAIMED risk, MEASURED clean on our side)
- Evidence: hoshinoyumemi CLAIMS a CLI push without machine_shape NvidiaTeslaT4 can land on a P100, where this torch has
  no sm_60 kernels even though cuda.is_available() is True. MEASURED now: all 59 GPU kernel-metadata.json files we built in
  session_2026-10-14/15 (excluding pulled/) set "NvidiaTeslaT4"; the 3 without it are CPU check kernels with
  enable_gpu false (jpeg/f20j/kernel_jj, jpeg/kernel_jc, kcrop/kernel_kc_side).
- THEN every new builder (R6 swap, KC stage, make_j, swap_builder) asserts machine_shape == NvidiaTeslaT4 before push.
- IF a commit log shows "no kernel image is available for execution" or sm_60, THEN it landed on a P100: re-push.

### 7. Label source for anything we still train (hoshinoyumemi, CLAIMED)
- Claims (silver/pulled2/hoshinoyumemi_knee-0-939-to-0-956-what-moved-the-score/, CPU write-up, 06:43 UTC): on gold58 the
  multilingual report lexicon beats the LLM label table that overrides it in the public Raptor training code for
  Contusion (.856 vs .689), Effusion (.777 vs .652), ACL (.953 vs .868), MCL (.964 vs .906), paired-bootstrap CI > 0; when
  a report is silent on synovitis the lexicon fills Synovitis from Effusion in 3,841 of 4,407 reports (87%); every visible
  public step came from a newly trained model, not re-weighting.
- IF any KC1 / S6-seed member still to launch uses LLM-table targets for those 4 findings, THEN (CPU, cheap) score our own
  label files per finding on gold58 vs the lexicon; switch targets only through a new PREREG amendment, never mid-run.
- THEN treat Synovitis targets from report-silent studies as weak (do not trust them as a gate signal).

### 8. R6 + R7 average as the base (only after lead 1 reads)
- Evidence: forum lens favours averaging R6 and R7; notebooks lens MODEL: both bundles cost ~+1.3 h, which with S6fall
  pushes p99 past the 8.0 h review limit.
- IF S6F-R6 moves the team best AND a runtime review of (R6 + R7, flip on both, + S6fall, + J) gives p99 <= 8.0 h, THEN one
  ladder slot 10-17..19. ELSE drop it.

### 9. rjings local-offset router (public, unscored)
- rjings/rsna-knee-target-local-router-20261015 v2 (ran 03:59 UTC, all inputs public): five extra R384 Raptor attention
  bags at the centre and fixed +/-8 mm offsets, averaged, routed only to Lateral OA, PF OA and Synovitis with a fixed 35%
  residual; falls back to the parent on failure. MEASURED: unscored at 10:03 UTC. The only new compartment-targeted idea.
- IF it scores >= .957 on its plain R7 + R384 base (a +1 display step), THEN prototype centre +/-8 mm bags for those 3
  findings on our r384 leg, only if the runtime review allows ~5 extra r384 passes (GUESS +0.5 to 1 h). ELSE ignore.

### 10. Rules: OAI, geo, missing planes (no change, watch only)
- MEASURED: no host reply since 13:20 UTC 10-14 on 743416 (LG22 geo, Driz Lo), 749145 (OAI + NIH geo), 749250
  (enforcement), 744519 (all three planes in every hidden study?, open since 10-06), 748925 (labels from a non-public
  model), 739905 (licence). Standing verdict: OAI-trained public weights and our OAI members are eligible (GUESS, high).
- THEN keep the per-study missing-plane fallbacks in every final candidate.
- IF a host post restricts OAI, THEN pick the OAI-light candidate as one of the 2 finals (it hits the nartaa SWA copy pile
  equally). Read the forum once after the lock, then once a day to 10-22, host/staff posts only.

## 2. Everything else (brief)
- Forum near silent (MEASURED): 10-14 20:30 -> 10-15 10:05 UTC = 12 messages in 4 topics (748181 x9 complaints, 749426 x2,
  730709 x1, 748403 staff reply 19:56 UTC). 0 edits across 348 bodies vs the 01:25 IST snapshot. No technical posts, no
  rulings, no host/staff after 19:56 UTC. The official Discord (linked in 730709) needs a login: unread.
- Top-team merges (MEASURED via teamIds / names): madmax0404 (r15 .964) vanished (GUESS merged); zosamu -> "Honda Gen-AI
  EXPs" (+22 subs); sakaku -> "Team Kangaroo"; "Nischay and Sho, Aillis CEO" .964 -> .966; "RB & Matheus & Tucker & Tony"
  .966 -> .967, now r1. Qile (r159) and Pickle Rick (r387) ids vanished. Effect on us: small; no action.
- heliosli renames (MEASURED): kernel 138669425 (old 0-9573, 154 votes, 229 forks) is now "This one is old version 0.957
  New one you can use"; its scored v1 = R6 no flip (.957), current v2 = R7 (.956 class), i.e. bait for anyone forking the
  current version. The 229 forks of the R6 v1 explain most of the .957 band growth (GUESS). Old slugs 404 on kaggle pull
  (renamed, not deleted). Dataset description now lists both manifest hashes (v1 6508ce69..., v2 c4bb0f73...).
- Every public "r6-frozen" copy is really R7 (MEASURED): crystalwong01/rsna-knee-r6-frozen-fe055406 mounts 20731557 (v2);
  crystalwong01 blend v3 .956; isotriage/rsna-knee-h957 v3/v4 R7 .956; lavinwins/rsna-gold-medal v4 and mayankraaz
  dual-stream v4 pin 20731557. mizeroluckygall/knee-mri-raptor-384-224-r7-blend and raihanjs/rsna-knee-fork-0957-heliosli
  now 404. Only a UI Copy & Edit of the v1-pinned 0-9576 keeps R6 (GUESS on UI behaviour).
- beraterolelk/0-957-sota-rsna-knee-multi-scale-foundation (kernel 138828840, v1 09:39 UTC, UNSCORED, 499 s): R7 (no flip)
  + r384 / r224 mirrors + bliverrigpdoge R9all with 140/156 mm two-scale TTA + 4x12 routing matrix + 0.65 z-logit / 0.35
  rank + hand-set co-occurrence smoothing (Effusion<->Synovitis +.04, ACL->Contusion +.03, Meniscus->OA +.025). Mounts an
  unrelated-named dataset (beraterolelk/istanbul-real-estate-2024-anonymized) used as a carrier / self-promo link. Its
  ablation is internally inconsistent (CLAIMS R7 + CoAtNet-384 = .957). IF it scores >= .958, THEN read cell 16. ELSE ignore.
- Cross-finding post-processing lost .020 macro AUC over 500 gold58 half-splits (hoshinoyumemi, CLAIMED). Spend nothing
  on co-occurrence smoothing.
- R9all stays closed: yuvrajlamba01/rsna-knee-research-optimized-0-957 (R7 + bliverrigpdoge R9all + nartaa SWA) = .956
  (v1, MEASURED); v3 unscored.
- Other watch alerts (10:03 UTC), none adds: eeshanvaghjiani2004/knee-oct15-224m .956 (R7); basith007/model-ensembling
  (private inputs); sameerk2004 sil-d5d-goldroute (10-09, old stack); karttikjangid05 LLM-labels comparison (analysis);
  tom99763/rsna-knee-sb and -sbb (R6 cnxfsss 209a32c8 + private sources, unscored, not copyable); rjings/rsna-versia-8
  v1 .956, v2 .939; sushanthtiruvaipati perfinding-dino5-prior .952; goodpjw2008 coatnet-2.5d v17 .956, v20 re-run.
  Unscored to re-check once: beraterolelk, oscarcarreoserpa, rjings router, varianceofx h957-reblend.
- 748181 (57 comments): Ravishta's "about 1000 teams at .957" is wrong (MEASURED 177 at .957, 936 at .956). Satwik,
  Ravi Ramakrishnan, D Chakraborty, I2nfinit3y, Cody_Null (now .958 r110) only complain about copies. No staff since
  Ryan Holbrook 10-12. Nothing technical.
- Earlier offer (744230, 09-29): Md Kishor Morol, DINOv2 ViT-S leg. Same family as Adel's; only relevant with a merge.

## 3. Contradictions resolved in this merge
- R6 obtainable: notebooks lens MEASURED v1 files + manifest hash beat SILVER_PLAN lever 9 / PROGRESS 11:20 IST "404".
- R6 alone vs R6 + R7: one-change R6 swap first (no runtime cost); R6 + R7 only behind a runtime review (lead 8).
- Ranks differ by lens (heliosli r220 vs r224, Adel r216 vs r220, .956 count 933 vs 936): snapshot time; this file uses
  api_lb_now.json 10:19 UTC.

## 4. Files
- Forum raw (29 topics + comp.json + pages.json): silver/forum_raw_1015b/; second pass: silver/pulled2/forum_1015T1030/
  (list_recent, msgs_*, batch_*, api_lb_now.json).
- Pulled public notebooks (untrusted, read only): silver/pulled2/notebooks_1015T1010/ (heliosli 0-9576, beraterolelk,
  hoshinoyumemi, crystalwong01 etc.), silver/pulled2/{beraterolelk_*, hoshinoyumemi_*, rjings_*, kvm/} (other agent; keep).
- Scratch readers (not in repo): C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/01efcc80-088f-477c-8a8a-b504ddf28d6a/scratchpad/
  (forum.py, dump.py, comp.py, kv.py, nbtext.py, dsfiles.py, dsprobe.py, dsman.py, lb_now.json, lbsum.py).
- Watch: session_2026-10-15/watch/WATCH_LOG.md (run 10:03:23 UTC, lines ~1074-1142), watch_state.json.
