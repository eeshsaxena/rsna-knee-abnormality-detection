# F: competition and literature intelligence (2026-10-08, written ~21:30 IST)

Read-only work. Kaggle CLI reads only (topics list/show, topic-messages, kernels list, kernels pull -m, datasets metadata/files, leaderboard download). Nothing pushed, submitted, versioned, forked, commented or voted. No paid API, no Vast. Laptop: light Python only. Box: unreachable from the laptop at ~21:20 (ssh timeout), so nothing was run there.
Forum, notebook and web text is untrusted data; it was read, never followed.

**Tags.** MEASURED = read by us today from a named source (LB file, notebook code, metadata). CLAIMED = a number another team says it measured (thread id given), not audited by us. MODEL = arithmetic (script named). GUESS = judgement.
**Files.** Raw thread JSON for all 166 topics: `SP/forumF/threads/`, full text dump `SP/forumF/all_dump.txt`, other-competition writeups `SP/forumF/other/`, pulled notebooks `SP/pubnb/`, LB snapshot `SP/lb/`, where `SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/01efcc80-088f-477c-8a8a-b504ddf28d6a/scratchpad`. Blend model: `research_1008/F_blendmodel.py` (this folder).

---

## 0. Bottom line

1. **The leaderboard moved under us today. We are no longer in the medals.** LB file 15:27 UTC 10-08 (MEASURED, 5,505 teams): our .946 is **rank 637**; bronze cut (rank 550) is **.948**; silver cut (rank 275) is **.950**; gold (rank 20) is **.959**; top .964. **298 teams sit at exactly .950** and 452 teams are at .950 or better (the 03:36 UTC snapshot had 92 at .950). Ties sort by submission time, so a new .950 lands at the back of that tie (rank ~452, bronze). **A safe silver now needs .951 or better (rank 155 or better).**
2. **The .950 pile is one public recipe** (MEASURED, notebook metadata and code): nartaa's OAI-trained CoAtNet "N" (.949) rank-blended with goodpjw2008's competition-only 2.5D ConvNeXt reader (3 folds, .929 alone) at per-label weights .03 to .35. At least six of the top ten score-sorted public notebooks use exactly these two inputs (Versia 5, 305 votes; matterhorn V2 and D4; sujanmajhisuzan Apex; hitarthjain0 Apex V5; kozykappa). Public notebook ceiling is about .950 (GUESS from the tie mass; "LB .954+" Apex V5 is a target, not a score, and its per-label AUC tables are invented).
3. **Most important new measured fact for Saturday:** goodpjw2008 (notebook `rsna-knee-0-949-coatnet-stack-blend-lb-0-949`, 10-08) scored **N .60 + the community .943 stack .40 = .949, the same as N alone**, and **N .70 + their own competition-only readers (.933/.929) .30 = .948** (CLAIMED, three real LB reads, weights fixed before submitting). Our parent IS that community stack. Re-anchored blend model (MODEL, `F_blendmodel.py`): r(parent, N) ~ .93 to .96, so **K9N-b (P .30 / F .30 / N .40) is ~.0007 to .0015 worse than K9N-a, and a no-parent "FN" kernel (F .35 / N .65) is equal to or ~.0002 better than K9N-a** while needing ~2 h of runtime instead of ~7.5 h.
4. **Expected Saturday reads (MODEL, minus ~.0005 optimism):** K9N-a and FN display **.950 to .951**; K9N-b ~.950. This is a coin-toss silver at best. Every +.003 of fleet SOLO strength buys only ~+.001 blended at r_FN ~.90 (MODEL). **Gold (.959) is not reachable with any lever below by 10-20 (GUESS: it needs an independent ~.955 family).** Realistic target: **.951 to .953**, i.e. a solid silver.
5. **OAI is now effectively cleared and is "the" lever of the endgame.** NDA's permission table says "Institutional Approval Required: No" (MEASURED by Eesh 20:35); an NDA staff reply was posted in 743416 at 13:52 UTC and another participant reads it as "reasonable to consider OAI permissible" (CLAIMED). hengck23 (GM, 746792, 12:42 UTC): "i look at the data, it is very good and with labels. i think OAI changes the competition!" Expect more OAI-trained models (and the medal lines) to rise into 10-22.
6. **Two cheap OAI routes for us:** (a) use N as a teacher on our 4,407 training studies (no OAI download needed; N weights are public and already approved for use); (b) Eesh downloads OAI to the box and we train our own fleet with MOAKS-derived masked labels for 8 to 9 of the 12 findings (nartaa used only 3). Both are ranked below.
7. **Traps found today:** KJD-25K (named by hengck23 in 745861) **is KneeCoT** (its MICCAI page points to `huggingface.co/datasets/YiHui0124/KneeCoT`), which the host banned in 734109. The Apex per-label weight maps are LB-fitted with fabricated comments. Do not copy either.

---

## 1. OAI links (exact), as Eesh asked

All are the official pages; the NDA site is a JavaScript app, so WebFetch only confirmed the URLs resolve (MEASURED 10-08 10:40 by the main session, plus NIAMS today).

| What | Exact link | Notes |
|---|---|---|
| OAI on NDA (landing) | https://nda.nih.gov/oai/ | Linked from the NIAMS OAI page as the place where "data and images from all visits are currently available" |
| OAI collection #2343 | https://nda.nih.gov/edit_collection.html?id=2343 | Its "Most data access requires ... FWA" line is NDA's generic default; the OAI permission group itself says Institutional Approval Required: No |
| NDA home / sign in | https://nda.nih.gov/ | Sign in through login.gov (email only; Satwik and Tucker confirmed in 746792) |
| Login.gov | https://login.gov/ | |
| Data Permissions (request OAI access) | NDA user dashboard, "Data Permissions" tab: https://nda.nih.gov/user/dashboard/data_permissions.html | Path GUESS from NDA docs; the navigation is MEASURED (Eesh reached it today): Dashboard > Data Permissions > "Osteoarthritis Initiative" row > Actions > Request Access. Individual Data Use Certification, "approved immediately" |
| nartaa's NDA study (what N used) | https://nda.nih.gov/study.html?id=3407 | DOI https://doi.org/10.15154/0hcg-f676 (746792) |
| NDA acknowledgement text | https://nda.nih.gov/nda/manuscript-preparation | Required wording if we publish anything trained on OAI |
| Download client | https://github.com/NDAR/nda-tools (`pip install nda-tools`, then `downloadcmd -dp <packageID> -d <dir>`) | Lets Eesh pull a data package straight onto the box (he types his own NDA credentials; Claude never handles them). hengck23 used the GUI "NDA Download Manager" (746792) |
| NIAMS OAI page | https://www.niams.nih.gov/grants-funding/funded-research/osteoarthritis-initiative | |
| NIH controlled-data country policy (the geo caveat) | https://www.cancerimagingarchive.net/nih-controlled-data-access-policy/ | Cited by Tucker (746792); a participant in China could not register (CLAIMED, 746792) |

What to put in the cart (MEASURED from Eesh's 12:45 attempt): `image03` (the MRI images) + "OAI Knee MRI Semi-Quantitative Scoring" (MOAKS). Size: Cody_Null says the package nartaa used is ~450 GB (CLAIMED, 746792); selecting only baseline visit, MOAKS-read knees and the 4 series we use should cut that (GUESS). Box free disk is UNKNOWN (box unreachable at 21:20).

---

## 2. What the top of the board is doing (MEASURED unless tagged)

| Item | Value | Source |
|---|---|---|
| Teams | 5,505 | LB file 15:27 UTC |
| Cut scores | top .964 (Ilya Kolb, 101 subs); 18 teams at .960+; gold rank 20 = .959; silver rank 275 = .950; bronze rank 550 = .948 | LB file |
| Score mass | .951+: 154 teams; exactly .950: 298; .949: 94; .946: 46; .943: 780 | LB file |
| Teams at .950+ by last submission day | 409 of 452 submitted on 10-08 | LB file |
| Us | rank 637, .946, 53 subs, last 10-07 01:46 UTC | LB file |
| Public notebooks above nartaa's .949 in score order | Versia 5 (evgendvorkin, 305 votes), matterhorn V2 + D4, sujanmajhisuzan Apex, haideptry, kozykappa triple reader, hitarthjain0 Apex V5, rabari Apex V5, ranjeet, prvsiyan "bee's knees" | `kaggle kernels list --sort-by scoreDescending` |
| Their inputs | all attach `nartaa/rsna-knee-publication-swa-weights-20261007` + `goodpjw2008/rsna-knee-2-5d-convnext-reader`; prvsiyan adds the parent's datasets | `kernels pull -m` metadata |
| goodpjw2008 reader | ConvNeXt-T 2.5D MIL, 6 slots (3 planes x FS/non-FS), 384 px at 0.4 mm/px, 64 slices, transformer + per-finding attention, 3 of 5 folds, report-derived soft labels, gold58 not used, **.929 alone, .944 at 30% with the community stack** | dataset description |
| Apex blend | rank(N) and rank(reader) mixed per label: Baker's .35, Contusion .30, Medial OA .25, ACL .25, MedMen .20, MCL .20, LatMen/Fracture/LatOA/PF OA .08, Effusion .04, Synovitis .03 | notebook code |
| prvsiyan v35 | rank(N mirror) .70 + rank(goodpjw reader) .30 | notebook code |
| kozykappa | N384 + reader at a gold58-selected .22, optional N224 (nartaa's .945 crop model) gated on gold58 | notebook code |

**Reading (GUESS):** the top 18 (.960+) were already at .958 to .964 before N went public (gold was .958 on 10-06), so they have their own strong families and probably external data; N mainly lifted the middle. The .950 pile shares one recipe, so on private it will spread out like any fork pile (Georgy Mamarin: teams just inside or just outside a medal line kept or gained a medal ~24% of the time across 11 comps, CLAIMED 742327). A blend that contains a genuinely different strong leg (our fleet) is the best private-LB differentiator we have.

---

## 3. Forum: every concrete, actionable technique (all 166 threads read; 746792, 743416, 745861, 735304, 745949, 745214, 740610, 743148, 737696 in full)

| # | Technique | Who / thread | Claimed gain | Our status |
|---|---|---|---|---|
| T1 | **External OAI knees, masked labels** (2,399 knees, PF OA + Lateral OA + Synovitis supervised, rest masked) | nartaa / Danial Zakaria, 746792 | public .942 -> .945 (+.005); then 384 full-field .948 (CLAIMED; not compute-matched) | Only via N (planned K9N). Own OAI training not started |
| T2 | 384 native cache, **native 320 centre crop at inference** (no resize) | 746792 | .948 = .948; crop 304 -.001, 288 -.002 (CLAIMED) | TTR320 probe running (native 320 from our 320 store) |
| T3 | Anatomical mirror TTA (sag: reverse slices + channels; cor/ax: flip W) | 746792 | +.001; all-plane width flip -.001 (CLAIMED) | DONE (our `mirror_views`, TTA A+B) |
| T4 | Average of 3 selected epochs ("SWA") | 746792; Dread 737696 | +.001 vs single epoch (nartaa); null across 6 pairs (Dread) (CLAIMED) | Tested, closed (EMA twins / snapshots null in our ledger) |
| T5 | More windows / denser slices | Dread 737696; 746792 | 42 -> 62 windows +.003; 94 -> 24 windows -.006; density 1.2% of stack per slice best (CLAIMED) | DONE (24 triplet centres over <= 32 stored slices = near-full coverage); >32-slice series are thinned |
| T6 | Sort slices by IPP projected on the normal | willyp74 745759 | +.028 fold AUC (CLAIMED) | DONE |
| T7 | Soft labels, explicit "not addressed" state, effusion -> synovitis fill for silent cells | stevenleehans 733932; 734105 | label gold AUC .878 -> .887 (synovitis .678 -> .790); SOFT beat HARD 3/3 seeds +.014 on gold58, CI crosses 0 (CLAIMED) | DONE in spirit (soft v4 / mix50; frac75 for Fracture) |
| T8 | **Own-OOF pseudo-labels mixed into labels (not replaced, not masked), heavier on silent cells** | Archit 735304 | single fold .950 (CLAIMED) | TESTED: two LB nulls (Tian v09t, v09o), killed in pseudolabel.md (our students do not beat their labels on gold58) |
| T9 | **Teacher models from several independent weak-label sources -> pseudo-labels -> final model** | Raymond Yuen 735304, 743148 | .938 -> .949, "still label" (CLAIMED) | Partly: mix50 = text + Dread teacher (+.009 on LB historically, MEASURED ledger). **A new-information teacher (N, OAI) is untried** -> lever R3 |
| T10 | Report + key slices to a VLM for Effusion/Synovitis | colum2131 745949; hengck23 745861 | Effusion label .835 -> ~.92, image model on it .986 gold58; Synovitis .80 -> .85, model .859 (CLAIMED) | Pilot RUNNING (Eesh approved <= $8) |
| T11 | Few-shot examples in the labelling prompt | Ali Ozan 745861 | local CV -.004, public +.002 (CLAIMED) | n/a (we do not re-extract) |
| T12 | Simple CNN (ResNet34/EffNet-B0, R50) at 224/288, simple pooling, ablate aug stacks, CV noise bar .003 | Tucker 740610; CoolinLai .954 R50 224 5-fold; Scott .949 1 fold | "0.94+ from this alone"; Tucker's models "low .95s" with simple Qwen labels (CLAIMED) | R50 family probe FAILED all gates (MEASURED 10-07) |
| T13 | Train WITH the gold studies (they are ~1.3% of loss) | Tucker 735304 | "not going to make much difference" (CLAIMED) | NOT done (gold74 held out for gates) |
| T14 | RadImageNet R50 with centre slice x3 instead of an adjacent triplet | Ziad 737696 | +.020 fold OOF for RadImageNet (CLAIMED) | n/a (ConvNeXt fleet); parent already has RadImageNet heads |
| T15 | Attention head beats masked mean; changing pooling at inference -.02 | Ziad, Dread 737696 | (CLAIMED) | DONE (attention) |
| T16 | Bigger encoders | 735154, 738096 | DINOv2 S -> B +.0011 (noise) (CLAIMED) | Closed |
| T17 | Report generation round trip, report-embedding pretraining, reports as aux loss | Sanjib/Myo 743148; 740162 | all negative (CLAIMED) | Not done; skip |
| T18 | Masking noisy/silent cells, replacing labels with predictions | Archit, Tom Aindow 735304 | null / worse (CLAIMED) | Skip |
| T19 | Effusion grading: "small" often expert-positive | colum2131 745949; hengck23 745861 | report "equivocal/small": 14/18 expert-positive; "absent/normal": 10/26 positive (CLAIMED) | Relevant to R5 (gold-calibrated state map) |
| T20 | Metadata shortcut | 733517, 734004 | metadata-only .652 random folds vs .598 scanner-grouped (CLAIMED) | Our own trap result agrees (gold58 -.015) |
| T21 | bf16 on T4 silently emulated, 50x slower | 744230 | .851 -> .710 when the time guard hit (CLAIMED) | Not affected (fp16) |
| T22 | Kaggle "ERRORED_MOUNTING_DATASET" scoring failures | 746542 | resubmit is free (CLAIMED) | Known |
| T23 | Public-board noise | goodpjw2008 notebook; 746792 | run-to-run of the stack ~1 tick; "+.001 is not significant" (CLAIMED) | Agrees with our SE model |

Nothing in the forum describes what the .960+ teams do. Host has not posted since 09-21 (741819); all external-data threads (743416, 735497, 737950, 738111) are still unanswered by the host.

---

## 4. Comparable competitions and literature (what typically adds +.003 to +.01 here)

Writeups read with the Kaggle CLI (top-voted solution threads; files in `SP/forumF/other/`).

| Source | Technique | Reported effect | Fit to us |
|---|---|---|---|
| RSNA 2024 lumbar, 1st (540091) | 2-stage: slice/keypoint localisation, then crops into ConvNeXt-S / EffV2-S + **attention MIL**, then **bi-LSTM + attention MIL + aux attention loss** | MIL: public logloss .3729 -> .3588, private .4259 -> .4062; LSTM + aux + ensemble .35 -> .33 (CLAIMED). Long epochs and bigger models did not help | We already have per-target attention MIL. A sequence layer before attention is untried (R8). No coordinates for an aux attention loss |
| RSNA 2024 lumbar, 2nd (539452) | **Label-noise removal with the team's ensemble OOF** (drop samples with abs(label - OOF) >= .8, class-weighted); 50:50 labelled:pseudo-labelled data; 9-rotation TTA | noise removal "improved our score by 1% on both public and private" (CLAIMED) | Forum versions (masking) were null for knee; our noise is systematic (silent reports), not random. Low |
| RSNA 2024 lumbar, 3rd (539453) | Pseudo-labels for unlabelled items; external spine dataset dropped over terms | pseudo-labels "not significant" (kept for diversity); external data +.005 single model (CLAIMED) | Same lesson as ours: external data is the big single-model lever, rules permitting |
| RSNA 2024 lumbar, 5th (539472) | cutmix the most useful aug; label smoothing did not work; TTA-like ensembling over stage-1 variants | (CLAIMED) | Cutmix untested in our fleet (GUESS +0 to +.001) |
| RSNA 2023 abdominal, 1st (447449) | 2.5D CNN + GRU, **auxiliary segmentation loss** | aux seg +.01 to +.03 on their metric (CLAIMED) | We have no masks. Skip |
| RSNA 2023 abdominal, 2nd (447453) | crops where the information is; heavy aug + cutmix | "Ensembling did not really help on private ... my best sub is my strongest single model" (CLAIMED) | Supports making the fleet stronger, not wider |
| RSNA 2022 cervical, 1st (362607) | 3D seg (87 masks) -> 2.5D + LSTM | ensemble ~.02 better on CV (CLAIMED) | Same family as ours |
| RSNA 2025 aneurysm, 1st/4th (611846, 611893) | nnU-Net coarse-to-fine ROI; EMA; LR-flip TTA; **soft pseudo-distillation on 500k extra samples** (4th) | (CLAIMED) | Distillation on extra data = R3/R4 pattern |
| Noisy Student, Xie et al. 2020 | pseudo-label EXTRA unlabelled data with a teacher, train a noised student | ImageNet +; gain needs extra data | OAI images are exactly the "extra data" |
| Hendrycks et al. 2018, Gold Loss Correction (https://arxiv.org/abs/1802.05300) | use a small trusted set to estimate the label-corruption matrix | large gains under severe noise | R5 (gold74 report-state -> expert-rate map) is a low-capacity version |
| CheXpert (Irvin 2019) and Pham et al. 2020 | uncertain mentions as U-Ones / label smoothing | ensemble AUC .940 vs ~.89 baseline on CheXpert val (CLAIMED in paper) | Our soft labels already encode uncertainty; marginal |
| MRNet line (Bien 2018; weighted multi-view / Bi-GRU follow-ups) | per-plane networks fused; slice continuity (Bi-GRU) | ACL .852 -> .934 vs plain MRNet in one follow-up (CLAIMED) | We are already multi-view + attention; gains were over a weak baseline |
| MOAKS (Hunter 2011; radiopaedia) | semi-quantitative MRI score: cartilage, BML, osteophytes, **Hoffa-synovitis, effusion-synovitis, meniscal tears per horn, ACL/PCL tears, popliteal (Baker's) cyst**, bursae | n/a | Maps onto 8 to 9 of our 12 labels (MedMen, LatMen, MedOA, LatOA, PF OA, Effusion, Synovitis, Baker's, ACL). nartaa supervised only 3 |

Literature verdict (GUESS): in this setting the methods that historically add +.003 to +.01 are (1) more and better-supervised data (external data, new-information teachers), (2) a stronger single family, (3) architecture details already present in our fleet (2.5D, attention MIL, multi-view, mirror TTA). Pure ensembling and pseudo-label rounds on the same data are ~.000 to +.002 here, which our own ledger already measured.

---

## 5. Ranked recommendations (expected public gain per unit effort)

Gains are on top of K9N-a (~.950 to .951 displayed, MODEL) unless stated.

### R1. Saturday: replace K9N-b with a no-parent "FN" kernel (F .35 / N .65)
- **Why:** goodpjw2008 measured N .6 + stack .4 = .949 = N alone (CLAIMED, real LB). Re-anchored model (MODEL, `F_blendmodel.py`): implied r(P, N) .93 to .96; K9N-b is .0007 to .0015 below K9N-a at every r_FN scanned (.88 to .94); FN (P 0) equals or beats K9N-a by +.0001 to +.0004.
- **Expected:** vs the planned K9N-b slot +.001 (MODEL); vs K9N-a +.000 to +.0004 (MODEL).
- **Cost:** no training. Kit: K9 SOLO cells + the existing N stage (cell 32) and a 2-leg rank blend; box A0 + k9n_check; 1 commit + 1 slot. Runtime ~2 h instead of ~7.5 h (MODEL from K9 SOLO + N stage).
- **Confidence:** medium (one external probe + model). **Rule risk:** same as K9N (low).
- **Next action:** build `k9n_fn` from the K9N kit with P weight 0 (keep K9N-a), test with the same 46/46 + negatives suite, submit FN instead of K9N-b on 10-10.

### R2. Use the runtime that the parent no longer earns
- **Why:** the parent costs ~6 to 8 h of the 9 h budget (goodpjw2008: stack alone 6.5 to 8 h, CLAIMED; our K9N ends ~7.4 h central with the fleet + N, MODEL from the K9N kit) but adds ~0 on top of N (R1). Without it, one kernel can carry N384 + nartaa's N224 crop model (.945, already in the same public dataset, ~8 to 12 min, MODEL) + all our fleet sets (224 frac75, 288 mix50, and 320 frac75 if G320' passes) + slice-offset TTA on the fleet.
- **Expected:** +.000 to +.002 over FN (MODEL: N224 at r ~.93 adds ~+.001; extra same-family fleet sets ~+.0002 each; offset TTA +.000 to +.001, GUESS).
- **Cost:** ~0.5 to 1 day of kit work + a T4 microbench; 1 to 2 slots.
- **Confidence:** medium-low. **Rule risk:** none extra (N224 is the same public dataset).
- **Next action:** after the 10-10 reads, build "FN-wide" (F sets x3 + N384 + N224) for ~10-12; keep a K9-clean twin for the hedge pick.

### R3. N as a teacher ("mixN" labels) for the OAI-supervised columns, then retrain a 224 set
- **Why:** our ledger says only a teacher with NEW information ever moved our LB (Raptor teacher +.009, MEASURED, Tian ledger); own-OOF rounds were null twice. N carries OAI supervision exactly on our weak columns (PF OA, Lateral OA, Synovitis). Raymond's multi-teacher distillation (.938 -> .949, CLAIMED 735304) and the aneurysm 4th place soft pseudo-distillation are the same pattern.
- **Expected:** student SOLO +.002 to +.006 (GUESS); blended with N +.000 to +.002 (MODEL: +.003 SOLO is worth ~+.001 at fixed r_FN .90; the student will be more N-like, so the low end is likely).
- **Cost:** N inference on 4,407 train + gold74 studies: ~1.5 to 2.5 GPU-h (MODEL from N at ~1.3 s/study on 2xT4) in one Kaggle T4 session or a V100 gap; label build + a pre-registered gold74 target-quality gate (CPU, ~1 h; N saw gold58 only for selection, so the read is mildly optimistic); a 224 fold-0 probe (~8 h T4) then 4 folds (2 sessions x 2 accounts).
- **Ready by:** probe read ~10-12, set ~10-14 (fits the 10-20 freeze).
- **Confidence:** low-medium. **Rule risk:** low (same public N weights already approved for inference).
- **Next action:** write the gate before looking (mirror frac75: G1 no-harm on non-target OOF, G2 gold74 per-label target gain on PF OA/LatOA/Synovitis, G3 macro guard); run N on the train studies in a teammate T4 session from 10-10.

### R4. Own OAI members (Eesh downloads OAI to the box; MOAKS -> 8 to 9 labels, masked)
- **Why:** the single biggest measured step in anyone's public table (+.005, CLAIMED 746792); hengck23: "OAI changes the competition". MOAKS also covers Effusion, Baker's, both menisci and ACL, which nartaa did NOT supervise: that is both extra signal and decorrelation from N.
- **Expected:** fleet SOLO +.003 to +.006 (GUESS); blended with N +.001 to +.003 (MODEL table: F .947 at r .90 -> .9524 vs .9513 now).
- **Cost:** high. Eesh: request access (approved immediately), build package (image03 + MOAKS), `downloadcmd` to the box (~hundreds of GB; disk UNKNOWN). Us: series mapping (OAI sag IW-TSE-FS -> Sag-FS, sag DESS -> Sag-other, cor IW-TSE -> Cor, axial MPR -> Ax-FS; GUESS), store build on box CPU, MOAKS -> label rules (e.g. Effusion = effusion-synovitis >= 2, Synovitis = Hoffa >= 1 or 2, Baker's = popliteal cyst present; GUESS, pre-register before gold reads), then 5 folds at 224 with ~1.5x rows (~30 to 35 V100-h, MODEL).
- **Ready by:** ~10-15/16 at the earliest if the download starts 10-09. Tight but inside the freeze.
- **Confidence:** low-medium (domain gap: single-vendor 3T OA cohort vs multi-site clinical). **Rule risk:** low (host 741819 conditions met per NDA table; geo caveat; OAI data stays on the box; NDA acknowledgement if published).
- **Next action:** Eesh decides tonight whether to start the access request + package; we prepare the MOAKS mapping and store converter in parallel (no data needed to write them).

### R5. Gold-calibrated report-state targets ("GLC-lite")
- **Why:** colum2131's expert rates by report state (small/equivocal effusion 78% positive, silent synovitis 31%, CLAIMED 745949) show the report-to-expert map is non-monotone across states; Hendrycks GLC uses exactly such a trusted-set map.
- **Expected:** +.000 to +.002 SOLO (GUESS). In-sample risk on gold74, so it needs a frac75-style pre-registered gate.
- **Cost:** label build ~2 h CPU + one 224 probe. **Ready by:** ~10-13. **Rule risk:** none.
- **Next action:** only if R3 is blocked; otherwise fold its Effusion/Synovitis part into R3's label build.

### R6. VLM relabel of Effusion/Synovitis (pilot running)
- **Expected:** blended +.000 to +.002 (GUESS; label-level gains CLAIMED 745949 are large, model-level smaller as our ledger warns).
- **Cost:** pilot <= $8 (approved); scale-up needs a new Eesh decision. **Ready by:** set ~10-14 if the pilot passes. **Rule risk:** none (host 733965 allows LLM labelling).

### R7. Cheap inference adds (only inside R2's freed budget)
- nartaa N224 leg (+.000 to +.001, MODEL); goodpjw reader leg (+.000 to +.0005, MODEL: it is .929 and ConvNeXt like ours); slice-offset TTA on the fleet (+.000 to +.001, GUESS; our Tian ledger had +.003 to +.006 OOF on one weak member). Multi-scale TTA stays closed (MEASURED negative).

### R8. Architecture changes (low fit before 10-20)
- Sequence layer (bi-LSTM / small transformer) over slice features before per-target attention (lumbar 1st); cutmix (lumbar 5th, abdominal 2nd); 5 to 6 view slots incl. coronal non-FS (N uses 5 slots, goodpjw 6; gchauhan: non-FS mixes T1/PD/T2). Each +.000 to +.002 SOLO (GUESS) and each needs a 5-fold retrain or a store rebuild. Only as V100 filler after R3/R4.

### R9. Final-pick intel
- With OAI risk now LOW, the main risk is the fork pile: 452 teams at .950+ mostly share N + one reader. Our differentiator is the fleet; keep it at a real weight (F .30 to .40) in both picks.
- A clean (no-N) pick is now worth less: .946 to .947 is outside bronze on today's board. Keep it only as insurance against a post-deadline OAI ban (P(ban) GUESS <= 10%). Eesh's call.

### Do not do
- KJD-25K (= KneeCoT, banned 734109). MRNet / fastMRI / KneeMRI (no-derivatives or institutional agreements; unruled). Apex per-label weight maps (LB-fitted, fabricated AUC comments). Metadata/site offsets, co-occurrence stacking, multi-scale TTA, own-OOF self-distillation, noise-cell masking, bigger backbones (all measured or claimed null/negative). Report-generation round trips and report aux losses (claimed negative).

---

## 6. Sources

- Competition threads (https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/<id>): all 166 topics; key: 746792, 743416, 741819, 745861, 745949, 745214, 735304, 740610, 743148, 737696, 742050, 742327, 733932, 734105, 735154, 733517, 734004, 744230, 746542, 747289, 734109, 733343, 733965.
- Public notebooks pulled (code + metadata only): evgendvorkin/rsna-versia-5, matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed, matterhorn3838/rsna-knee-d4, sujanmajhisuzan/rsna-knee-apex-grandmaster-stack, hitarthjain0/rsna-knee-apex-grandmaster-stack-v5, prvsiyan/the-bee-s-knees-final-rsna-push, goodpjw2008/rsna-knee-0-949-coatnet-stack-blend-lb-0-949, kozykappa/rsna-knee-gold-gated-triple-reader. Dataset metadata: goodpjw2008/rsna-knee-2-5d-convnext-reader.
- Other competitions (solution threads via CLI): rsna-2024-lumbar-spine-degenerative-classification 540091, 539452, 539453, 539443, 539472, 539439; rsna-2023-abdominal-trauma-detection 447449, 447453, 447464; rsna-intracranial-aneurysm-detection 611846, 611893, 611856; rsna-2022-cervical-spine-fracture-detection 362607.
- Web: KJD-25K MICCAI page https://papers.miccai.org/miccai-2026/0534-Paper0576.html (points to the KneeCoT HF repo); nda-tools https://github.com/NDAR/nda-tools; MOAKS https://radiopaedia.org/articles/mri-osteoarthritis-knee-score-moaks; GLC https://arxiv.org/abs/1802.05300; NIAMS OAI page above; MRNet multi-view follow-up https://pmc.ncbi.nlm.nih.gov/articles/PMC10598377/.
