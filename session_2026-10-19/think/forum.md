# Forum + top-notebooks lens, 10-19 (read 10:52-11:40 UTC = 16:22-17:10 IST). READ-ONLY. No em dashes.

Scope: every concrete hint from .959+ teams and top public notebooks (model families, resolution, TTA, checkpoint
averaging, external-data rulings, stacking, private-split remarks, runtime use), the competition data / evaluation /
acknowledgement pages, host posts since 10-14, and ideas we have NOT tried, with source links.
Nothing posted, voted, submitted, forked, pushed, committed or created. No GPU, no box job. No PREREG / HANDOFF /
PROGRESS / scheduled-task edit. No public code executed.
Tags: MEASURED (we fetched / computed it, file named) / CLAIMED (a third party or host said it) / MODEL (our arithmetic)
/ GUESS. "rank N" = the poster's live `competitionRanking` from the forum API at 10:52 UTC (not their rank when posting).
Live LB 11:00 UTC (MEASURED, work/api_lb_1019T_think.json): .959+ = ranks 1-183, so "rank <= 183" means a .959+ team.

Base URL for threads: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/<id>

## 0. Bottom line
1. **No host or Kaggle-staff post since 10-14 19:56 UTC** (Dustin, 748403). MEASURED: no thread moved between the
   10:52 UTC fetch and an 11:38 UTC re-list; last post anywhere 750529 at 09:03 UTC; the full host/staff list is in
   sec 2. Open and unanswered: missing planes in test (744519), site-held-out vs stratified split (734681), how
   Synovitis was graded (737155), the 10-17 external-data wave (750236, 750257-750260).
2. **No public notebook above .958** (MEASURED, watch run 10:56 UTC, 527 notebooks listed, 102 read: bands .958 x5,
   .957 x7). Publishing has been locked since 10-15 23:59 UTC; nothing new to copy. All .958 notebooks are the same
   R6 + Raptor (384 mirror + 224) + heliosli-table family that c4 already contains (session_2026-10-17 notebooks.md).
3. **Every .959+ poster credits own labels + own k-fold CNNs, not fusion tricks.** Families named: ConvNeXt-T 2.5D,
   EfficientNetV2-S, ResNet-50/34, EffNet-B0, CoAtNet 2.5D, DINOv2 ViT-S 2.5D, Qwen3.5-2B LoRA VLM (sec 4). None
   disclosed per-finding weights, rank-vs-prob fusion, stacking or extra TTA. Nothing there is buildable and readable
   before 10-21 without new training (MODEL).
4. **Two page facts we had not used (MEASURED 11:2x UTC, work_forum/pages/):** (a) `scoreTruncationNumDecimals: 3`, so
   the display TRUNCATES: c4's ".958" means [.958, .959). (b) The Acknowledgements page lists 19 data contributors plus
   **three "additional contributing sites": Intermed Hospital (Mongolia), Liverpool Hospital (Australia), Salus
   Vigevano (Italy)**. Our train.csv has **0 Italian and 0 Mongolian reports** (MEASURED, _data/_reports_lang.csv: en
   1736, es 682, tr 546, hr 406, el 321, de 262, bg 220, nl 153, fr 81; no `it`, no `mn`; regex check for Italian-only
   words and Mongolian letters = 0). GUESS (P ~.6-.7): the hidden test holds studies from sites never seen in train.
   Host (733491, 08-07): test labels came from the same image-review process as gold58, so the hidden labels follow the
   EXPERT regime (gold58-like label coupling), not the report regime.
5. **The spare ~2.5 h of hidden runtime has no forum-backed use.** The one cheap TTA candidate the forum suggests
   (Raptor r384 multi-FOV: 320 crop + full 384) screened on gold58 inside c4 at **-.0003 macro, P(>0) .41** (MEASURED,
   work_forum/fov_tta_g58.json). New-family training (ViT, ConvNeXt-T, VLM) cannot be built, committed and read
   before 10-21, and our own retrained seeds of public recipes have read negative (S6 s43/s44).
6. **The forum adds weight to keeping B as the less public-tuned pick**: staff opinion (public ensembles drop on
   private, 748181), the 742327 shake-up study (teams just inside a medal line on public medalled 24 % of the time,
   CLAIMED), and the likely unseen-site test (point 4). Picks stay Eesh's call; nothing here argues to change A.

## 1. Coverage (MEASURED)
- Reused `forum_raw/` (34 threads, l_hot/new/recent/top lists, fetched 10:52 UTC by the interrupted run).
- Fresh RECENT 3 pages + NEW 2 pages at ~11:20 UTC (`forum_raw2/l_recent.json`, `l_new.json`) and RECENT p1 again at
  11:38 UTC (`l_recent_b.json`): **0 threads moved since the 10:52 fetch**. The only threads with posts after the
  10-19 05:00 UTC digest (session_2026-10-19/forum/FORUM_10-19.md) are 750529 (07:10 Civitasmass rank 190 "haven't used
  any external data"; 07:59 and 09:03 chatter), 744230 (04:46) and 750318 (04:45), both by SwapIY (rank 199 = our team;
  GUESS from the rank match and our swapiy_* packs), operational tips only (sec 7).
- Completeness pass: 19 listed threads had never been fetched by any lens; fetched the 11 with possible content
  (`forum_raw2/msgs`, flattened in `forum_raw2/flat`): 743148, 738096, 736635, 734004, 733491, 737163, 747865, 746542,
  747660, 735121, 738156. Useful bits: host on test-label process (733491), PatientSex in test headers (734004), two
  corrupt train DICOMs (737163), DINOv2 small = base at convergence (738096). The rest are memes, team-up or rule
  threads already summarised in the 10-17 lens.
- Top-poster extraction: `work_forum/hi_rank_posts.txt` (201 posts by rank <= 200 or host/staff across the 34 threads).
- Competition pages: `work_forum/pages/ListPages.json`, `GetCompetition.json`, `rules.txt` (anonymous PageService).

## 2. Host and staff posts (MEASURED from msgs authorType HOST / ADMIN, plus the 10-17 and 10-19 digests)
| date (UTC) | thread | who | content |
|---|---|---|---|
| 08-06 | 733343 | Po-Hao Chen (HOST) | overview + label rubric (sec 3) |
| 08-07 | 733491 | Po-Hao Chen | gold labels from image review, 2 readers + adjudicator; "The same process was used for the testing set"; reports may contradict labels |
| 08-09 | 733965 | Po-Hao Chen | commercial LLMs on report text allowed |
| 08-25 | 734109 | Po-Hao Chen | KneeCoT NOT allowed |
| 08-26 | 737163 | Jason Sho (RSNA curator) | will review 2 corrupt train DICOMs (no follow-up found) |
| 08-27 / 09-21 | 733965 / 741819 | Po-Hao Chen | click-through registration generally fine; OAI OK if generally accessible and free |
| 10-12 | 748181 | Ryan Holbrook (ADMIN) | no anti-blend policy; low-effort public ensembles "tend to drop on the private leaderboard" (opinion) |
| 10-13 | 736301 / 749038 | Po-Hao Chen | AI-use survey; "can neither confirm nor deny the presence of gorillas" |
| 10-14 13:18 / 13:20 | 743416 / 744056 | Chris Carr (RSNA) | no per-dataset rulings; registration + licence "clearly" qualifies; public weights eligible for awards, disclose them |
| 10-14 19:56 | 748403 | Dustin (Kaggle) | cleared a > 24 h stuck submission |
**Since 10-14 19:56 UTC: none** (MEASURED to 11:38 UTC 10-19).

## 3. Competition pages: scoring and data details that matter (MEASURED unless tagged)
- Metric: unweighted mean of the 12 per-label ROC AUCs (Evaluation page). Kaggle metric id 499 "Roc Auc Score".
- `scoreTruncationNumDecimals: 3`: display truncates (HANDOFF l.481 already said "TRUNCATED"; now from the API field).
  With c4 unrounded at ~.9587-.9589 (MODEL, brief), a true +.0001 to +.0003 would display .959. The R1 best-id read is
  full precision anyway.
- `leaderboardPercentage 30`, test "about 1300 studies" (~390 public / ~910 private), `numScoredSubmissions 2`,
  `usesSynchronousReruns True` (private comes from the same run as public), `maxGpuRuntimeMinutes 540`, 6,209 teams.
- Data page: "the prevalence of abnormalities is not guaranteed to be the same across the training, public
  leaderboard, and final evaluation datasets". Series "typically 20-45 slices (median 30), with a long tail out to a few
  hundred". Transfer syntaxes: uncompressed, JPEG Lossless, JPEG 2000, Implicit VR; DICOMs stripped to 86 tags. No
  report at test time.
- Missing series: 744519 (Satwik, rank 96: does every test study have all planes?) unanswered since 09-30. Our scored
  entries ran clean, so this only matters for any NEW member's fallback path.
- Label rubric (733343, host): borderline ("on the fence") graded NEGATIVE; ACL / MCL = high-grade tears only; OA =
  >= ~1 cm area of > 50 % cartilage loss per compartment; Effusion = moderate or large; Synovitis = "inflammation and
  thickening of the synovial lining"; Fracture = acute cortical break. Single knee per exam.
- Gold58 vs report coupling (MEASURED now, work_forum/g58_labelrel.json): expert P(Synovitis | Effusion) .63 vs
  P(Synovitis | no Effusion) .22 (phi .40, n 58); report labels (v4_blend, n 4,333): .197 vs .012. Expert Synovitis
  prevalence .47 vs report .12; Fracture .31 vs .07; Lateral Meniscus .40 vs .15 (work/xlabel_gap.py output).
- Acknowledgements page (`overview/acknowledgements`): 19 contributors + 3 additional sites (point 0.4). Language map
  of train (MEASURED) covers the 19: es (Argentina, Chile), tr (3 Turkish sites), hr (Zagreb, Sarajevo), el (Crete),
  de (Wurzburg), bg (Sofia), nl (AZ Delta), fr (Marrakech / McGill), en (US, Canada, Malta, Thailand, Taiwan). Italian
  and Mongolian are absent. Liverpool (English) cannot be checked by language.
- Rules 2.6: external data must be "publicly available and equally accessible ... at no cost" or meet the
  Reasonableness standard (geo restrictions named explicitly); hand-labelling of test data forbidden.

## 4. What .959+ teams and top notebooks say, by topic (CLAIMED unless tagged; thread id in brackets)
**Model families and single-model scores**
- azamat1ch (rank 2): single architectures, 5-fold, public: ConvNeXt-T .959, EfficientNetV2-S .956, CoAtNet .952
  [735304, 10-09]; no external data; the secret is "labels", "AI, not by hand", "don't rely on reports alone" [748295].
- Rank-1 team = Tucker Arrants + Tony Li + RB (all show rank 1, MEASURED field): ResNet / EffNet at 224-288, "no
  attention mechanisms, just simple pooling", 5-fold, TTA, ~25 min total runtime; report-label CV, noise threshold
  ~.003; single-prompt Qwen extraction, label tweaks never moved LB [735304, 740610, 745214].
- colum2131 (rank 22): ConvNeXt-T 2.5D .953, DINOv2 ViT-S 2.5D .951, two-family ensemble .955; same preprocessing and
  aggregation head, image pseudo-labels blended with LLM labels at a fixed ratio [735304].
- Just Need Luck (rank 49): ResNet-50 5-fold 224 .954. Tanuki_boosting (rank 51): single model 5-fold .955. Musa Tur
  Farazi (rank 160): single model single fold .956. Archit Konde (rank 109): 2.5D CoAtNet 224 single fold .950.
  Raymond Yuen (rank 51): CoAtNet 288 .949, all from labels. NguyenThanhNhan (rank 74, GM): Qwen3.5-2B with LoRA on
  the LM + vision encoder, 384 px, single fold .950, ~3.5 h. Adel Zebiche (rank 176): DINOv2 ViT-S 224 single fold .952.
  datnt114 (rank 60): best .961 with 5-fold OOF .937, gold58 .938, "trust cv local" [748011].
- Komil Parmar (rank 58): DINOv2 small = base at convergence at 224 [738096].
**Resolution / FOV**: 224-288 is the "sweet spot" (Tucker, tennogh rank 13, Raymond: 384 -> 288 "barely had any
  effect"); FOV 140-160 mm better than 100 mm (Parag rank 47, Cody_Null rank 189) [735304].
**TTA**: tennogh .942 -> .943 with TTA at 288 [735304]; Tucker includes TTA; nartaa: anatomical mirror +.001, all-plane
  width flip -.001 on an older checkpoint, serve crop 320 -> 304 -.001, -> 288 -.002, 94 -> 24 windows -.006 [746792];
  dreaddevelopment: window count matters (42 -> 62 windows +.003) [737696].
**Checkpoint averaging / seeds**: 5-fold is the standard top recipe. SWA vs single epoch: nartaa top-3-epoch SWA +.001
  [746792]; dreaddevelopment "across six SWA/non-SWA pairs it averages out to nothing" [737696]. Fresh retrains of the
  published CoAtNet scored .944-.946 vs the published .948-.949 (nartaa table), i.e. the published checkpoints are
  favourable draws (MODEL: selection on gold58 + public).
**Labels (the lever every top team names)**: image-model pseudo-labels mixed with LLM labels (colum2131 fixed ratio;
  Raymond 50/50 teacher; Archit: OOF predictions mixed in, heavier on the model, soft labels better than 0/1, out-of-
  fold only); label-agreement filtering (Prateek Grover rank 260, +CV +LB); "I don't know" option and hard negatives
  (hengck23); Synovitis is mostly unreported in text (starkhushi: 16 % of reports mention it) [735304, 745861, 737566].
**Stacking / fusion**: nobody at .959+ disclosed any. heliosli (public) uses per-finding rank blends + refitted
  Effusion / Baker state stackers (R6/R7 bundles). hengck23: only ~10-parameter fits on gold58 are safe [745861].
**External data**: azamat1ch, diet1236364 (rank 20), YYama (rank 12), Cà Rốt (rank 61), Civitasmass (rank 190) say no
  external data; minhtu.mt.mt (rank 33) downloaded OAI [746792]. Host 10-14: public weights eligible, disclose.
**Validation / private split**: colum2131: LLM-label OOF vs public r .87, gold58 vs public r .66 [748011]; nartaa:
  public split SD ~.003 [749359]; 1 - p submissions hide strong teams (colum2131); Mamarin study: teams just inside a
  medal line on public medalled 24 % of the time (range 2-51 %) [742327, via 10-15 digest]; PC Jimmmy: past RSNA
  shake-ups modest [734681]; Amanie El-Hajj (rank 113): ranks 100-1000 share one display score [748181].
**Runtime use**: nobody reports using spare runtime for extra legs; strong entries run 5-30 min (Tucker 25 min,
  Scott Willis 5 min); dreaddevelopment's 94-window CoAtNet blend ran ~9 h.
**Top public notebooks** (MEASURED, watch 10:56 UTC): goodpjw .958 (R6 + Raptor 384/224 mirror + heliosli V1 + GP
  2.5D reader 1/10 + R7 .4), heliosli blend-mine-in .958 (= HW minus S6fall), heliosli 0-9586 .958 (V2 table + reader),
  lavinwins copies. No .959. Their authors' own teams rank below us (10-17 lens).

## 5. Ideas we have NOT tried (ranked by expected value for a 10-21 read; all need Eesh's go + PREREG amendment)
| # | idea | source | evidence | expected macro gain | P(positive) | cost | risk |
|---|---|---|---|---|---|---|---|
| 1 | Effusion -> Synovitis nudge in the final blend: Syn = pct(Syn) + a pct(Eff), a .1-.2 | 737566 (starkhushi), 737155 (unanswered rubric question), host rubric 733343 | gold58 per-label +.0048 (a .1, P .85) / +.0084 (a .2, P .81) (MEASURED, work/xlabel_g58.json); report-regime R7 OOF -.0003 / -.0014 (MEASURED now, wrong regime per host 733491); expert coupling .63 vs .22 | MODEL +.0001 to +.0004 (per-label /12, shrunk for the gold58 screen's PF OA miss) | ~.5 | 1-line c4 change, 0 runtime, kernel ~6.5 h commit + ~6.5 h score | Synovitis = ~25 % of private macro variance; gold58 screens misled before (PF OA .1, P .925, read negative) |
| 2 | Keep pick B the less public-tuned entry (R6 uniform) given a likely unseen-site test | Acknowledgements page + train language census (MEASURED), 748181 staff, 742327 | 3 contributing sites with no train reports; public-tuned per-label tables fit the public site mix | insurance, not EV on public | n/a | 0 | none; it is the current plan |
| 3 | dreaddevelopment competition-data-only Raptor checkpoint (finespacing .932 solo) at .1-.15 inside the Raptor slot | 737696, datasets dreaddevelopment/raptor-knee-finespacing, raptor-knee-widedense | different slice geometry decorrelates (varshitha's 224 add, our a25 positive) but much weaker; our 10-04 v9 finespacing swap was a visible no-op (HANDOFF) | 0 to +.0002 (GUESS) | ~.35 | +20-25 min hidden, new kernel cell | dilutes the best Raptor; one more public-read knob |
| 4 | Second CoAtNet r384 seed with nartaa's public trainer (box has OAI) to fill the spare runtime | 750180 / 749359 (training_code/ public, Apache-2.0), 10-13 digest lever 4 | nartaa's own fresh retrains were .002-.005 below the published checkpoint; our S6 s43/s44 retrains read negative; gold58 plus_S43 .15 -.0012 (MEASURED, stack_ops.json) | 0 to +.0003 (MODEL) | ~.3 | port ~0.5 day + ~12-20 h V100 + ~25 min hidden; misses the 10-20 push, so blind 10-22 only | GPU time on a likely-negative; OAI provenance same as today |
| 5 | Raptor r384 multi-FOV TTA (320 crop + full 384, each mirrored) | 746792 (crop table: 320 crop = full 384 at display) | gold58 inside c4: -.0003 (P .41) at .5, -.00045 at 1/3; standalone K24 +.0003 (MEASURED now, fov_tta_g58.json; full384 is nartaa's gold58 selection view, so optimistic) | ~0 | ~.4 | +~25 min hidden | none big; just no signal |
| 6 | New family (DINOv2 ViT-S 2.5D, ConvNeXt-T 2.5D, EffNetV2-S, Qwen3.5-2B LoRA) as a small-weight member | 735304 (colum2131, azamat1ch, Adel, NguyenThanhNhan) | top teams' singles are .950-.959 with their own labels; our K14 ConvNeXt-T fleet and frozen DINOv2 probe (.7186) were negative | ~0 for a single fold at small weight (MODEL) | ~.2 | 1-2 days V100 + build; not readable by 10-21 | burns the box for a likely-negative |
Not an idea for this window but worth a post-mortem: image pseudo-labels mixed into report labels (every top team), label
agreement filtering, hard negatives, 3DReasonKnee boxes (huggingface.co/datasets/rajpurkarlab/3DReasonKnee).

## 6. Closed or already covered (do not redo)
- goodpjw 2.5D reader 1/10, heliosli V2 table, R7 .4, 224 mirror, S6fall flip, side model: closed in the 10-17 levers /
  notebooks lenses (session_2026-10-17/research_1hr/levers.md, notebooks.md).
- Metadata priors (PatientSex is in test headers, ACL M 54 % vs F 32 %, 734004): our 10-07 guard, site / metadata
  offsets gold58 -.0152 (NOVEL_PLAN_10-07.md l.277). Keep closed (unseen sites make it worse).
- Multi-scale TTA on our own models (-.0005, NOVEL_PLAN_10-07), co-occurrence stacking (-.001), cross-label logistic
  stacker on R6 (no gain on gold58, cross_label.json), blend operators (stack_ops.json), tie handling (tie_loss.py).
- Per-label free picks on gold58 (nested split-half -.0007, brief); hengck23's "10-parameter" advice is the same thing.
- Seeds / retrains of public recipes (S6 s43 / s44), K14, KC, R7-teacher student, R6 + R7: LB-negative (brief).
- External-data hedge pick: still no clean entry above ~.94 (10-18b digest); no host ruling.

## 7. Operational hints from the forum (CLAIMED, relevant if a new kernel is pushed 10-20)
- SwapIY (our teammate, 744230 / 750318, 10-19 04:45): set `"machine_shape": "NvidiaTeslaT4"` in kernel-metadata.json
  (otherwise a P100 may be assigned); at most 2 GPU batch sessions per account (third commit waits QUEUED); a CLI push
  without `--timeout` reserves 12 h of quota and is refused when less is left (`kaggle quota`, CLI 2.2.4+); check each
  member's fallback / empty count in the commit log, not only that submission.csv exists; `PYTHONUTF8=1` on Windows.
- OzanM (GM) 750318: cancel and restart a commit that sits queued > 1 h (our 1 h rule from 10-18b).
- 744230: bf16 is emulated on T4 / P100; use `torch.cuda.is_bf16_supported(including_emulation=False)` (Roman Vuskov).

## 8. Standing context (MEASURED, LB 11:00 UTC)
- .959+ = 183 teams (r1-183); .958 band r184-1401 (1,218 teams); c4 = r199 (16th in band). Silver cut ~r310, bronze
  ~r621 (5 % / 10 % of 6,209), both inside the .958 band.
- .959+ grew 141 (10-17 16:21 UTC, metric.md) -> 183 (+42 in ~43 h, ~23/day). MODEL: ~265 teams at .959+ by the
  deadline, so c4 public rank ~280, still above the silver cut on public.

## 9. If / then
- IF a host or staff post appears on external data, missing planes or the private split before 10-22, THEN tell Eesh
  at once (re-open the hedge only for a ruling against OAI-trained weights or SKM-TEA anatomy.pt).
- IF Eesh wants one more zero-runtime entry on 10-20, THEN idea 1 (Synovitis <- Effusion a .1) is the only forum-
  sourced candidate with a mechanism; expect a coin flip; do not stack it with anything.
- IF anyone proposes filling the spare 2.5 h, THEN point to sec 5 rows 3-5: no forum-backed member or TTA clears ~0.
- IF a watch run shows a public notebook >= .959, THEN do the bait / pin check and compare its author's team rank first
  (10-17 notebooks lens rule).

## 10. Files (all new, this session)
- forum_raw2/: l_recent.json, l_new.json, l_recent_b.json, msgs/ (11 threads), flat/ (flattened).
- work_forum/: hi_rank_posts.txt, pages/ (ListPages.json, GetCompetition.json, rules.txt), pages.py, flatten2.py,
  g58_labelrel.py/.json, fov_tta_g58.py/.json.
- Reused: forum_raw/ (34 threads), pulled/watch/WATCH_LOG.md, work/api_lb_1019T_think.json, work/xlabel_g58.json,
  stack_ops.json, cross_label.json; ran work/xlabel_oof.py and work/xlabel_gap.py (CPU, < 1 min, printed only).
