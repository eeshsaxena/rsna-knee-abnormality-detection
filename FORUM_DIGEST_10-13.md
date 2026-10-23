# Forum + public-notebook digest 2026-10-13 ~16:00 IST (Claude, read-only sweep)

Sources: kaggle CLI (kernels list by score pages 1-3 + by date pages 1-2, kernels pull -m for 25 notebooks, datasets list/files),
Kaggle web API read in the browser (anonymous POSTs to ListKernels / GetLeaderboard: bestPublicScore per notebook, team rank +
submissionId per team), discussion pages. Nothing submitted, voted, commented or downloaded from the competition.

## 0. Leaderboard snapshot (10:26 UTC, 5,903 teams) and a correction

- Bands: .966 r1; .960 r33-49; .958 r60-80; **.957 r81-123 (43)**; **.956 r124-513 (390)**; .955 r514-586; .954 starts ~r587.
  Cuts by rule: gold r21, silver r295, bronze r590.
- **Us: .954, rank 590, team best = sub 57106365 (HF-R20).** HR7 (57137313) and HFR7-R20 (57137635) still PENDING at 10:20 UTC.
- **Correction:** the "us .954 r822" in PREREG_10-13 / PROGRESS_13-10 is wrong. The 08:39 UTC CSV
  (`scratchpad/lb13/...08_39_49.csv`) has us at **r535**, i.e. 5th of the 597-team .954 block (r531-1127). Now 4th of the block.

### Key mechanism (MEASURED from LB structure): rank inside a displayed-score block follows the UNROUNDED score
- If ties were ordered by submission time, submissionIds inside a block would be increasing. They are not: the .956 block has
  146 inversions in 389 steps (147 increasing runs, longest 27), the .954 block 243 inversions. So Kaggle orders by the full score,
  and only exact ties fall back to time.
- Corroborated by 748310 (user: a modified-weights sub "clearly ranks higher" than the unmodified copy at the same display) and by
  varshithakolupuri's notebook (R7 copy r233 vs R7 + 224 checkpoint r156 inside the .956 block).
- Caveat: copies of the same notebook are not exact ties either (the block is not time-sorted), so hidden-rerun GPU
  nondeterminism adds small noise to the unrounded score. Size unknown.
- **What it tells us already:** HF-R20 (H + our 20-member K14 fleet at .2) sits 4th of ~574 in the .954 block, above nearly all
  H copies. That is the first evidence that our fleet adds a real sub-.001 gain on top of a public blend (rank-based, not a score).
- **How to use it:** the GetLeaderboard API shows each team's best submissionId and rank. A new submission shows up only if it
  beats our current best, so every submission becomes a one-sided "is it better than the best so far" test at sub-.001 resolution.
  Order submissions so that each is expected to beat the previous one.

## (a) Public notebooks >= .955 (bestPublicScore from the API; inputs from kernel-metadata.json)

No public notebook scores >= .957. The top is .956, all R7 or heliosli family.

| Notebook | Public | All inputs public? | Difference vs R7 (heliosli/0-957-latest-dataset-score-in-one-click) |
|---|---|---|---|
| heliosli/0-957-latest-dataset-score-in-one-click (R7, 10-11 20:01) | .956 | yes (r7cnxfsss, nartaa SWA, orthofoundation) | reference: Raptor CoAtNet r384 + R7 ConvNeXt-small 4 folds, uniform .5/.5 per-class rank blend, refitted Effusion / Baker state stackers, probabilities written %.9g |
| heliosli/rsna-knee-blend-gold-old-dataset | .956 (V1) | V1 no (private S6); V2 yes (r6cnxfsss V2) | V1's .956 used private components; V2 = S6 bundle 209a32c8b50a6b1c, scores .954 per 2 commenters. Not a lever |
| mercurydust/rsna-v3 (10-13) | .956 | yes | code identical to R7 (empty diff) |
| raihanjs/rsna-knee-fork-0957-heliosli (10-13) | .956 | yes | identical to R7 |
| jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble (10-13) | .956 | yes (internet ON in metadata) | R7 + rewritten markdown + a multi-slug find_mount; same models and weights |
| z1358028281/0-957-..., piyush474/...-r7-ensemble (10-13) | not scored | yes | copies of R7 / Jenil |
| bliverrigpdoge/velci-kcd25-20 | .955 | no (one blank / private source) | compact-v1 S6 family, below R7 |
| **varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956 (10-13 08:39)** | not scored as a notebook; author team now r163 .956 (sub 57131495) | yes (same 3 datasets as R7) | **R7 unchanged + nartaa's second checkpoint `..._c08_r224_swa.pt` (0.945 alone, 224 px from a 0.8 centre crop) run with nartaa's 0.945 script as Stage 2b, mixed into the Raptor slot: rank_raptor = .65 x r384 + .35 x r224 (A224 = .35 fixed). Skips itself if < 30 min left (keeps 40 min margin, 2.5 h cap).** Author's claim: R7 copy r233 -> with 224 r156 inside the .956 block; same on the S6 .954 recipe r437 -> r320 |
| goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954 v11 (10-13 10:07) | .954 (older version); v11 being scored | yes | S6 full-precision base + the 224 checkpoint at .35 + their 2.5D MIL reader at 1/10 weight. **Their own note: on the full-precision .956 base the reader at full weight cost about .002** |
| sushanthtiruvaipati/rsna-knee-heliosli-r7cnxfsss-0-5-blend | not scored (team r178 .956) | yes | R7 bundle, preflight cell, same blend |
| hitarthjain0 "...0957" / "target-gated-silver-sota" | .954 / none | yes | compact-v1 S6 + goodpjw reader, Raptor .7 + per-target weights. Below R7 |
| ayodeji m01-m05 / l01-l05 (10-13) | none | yes | Raptor r384 + dreaddevelopment maxspan, per-label "dynamic" weights; their j-series of the same family scored .937-.949. Not a lever |

Other facts:
- **heliosli has no R8:** datasets = compact-v1, r6cnxfsss, r7cnxfsss (10-11 19:59), orthofoundation. No new RSNA notebook versions since
  10-11 20:01 (his 10-13 notebook "try merging it..." belongs to ARC-AGI-3). He says he will "continue putting in the work... and
  sharing" (748181). **Kaggle staff confirmed a one-week notebook publishing lock** before the deadline (~10-15 23:59 UTC), so
  an R8 could only appear in the next ~2.5 days.
- R7 fold files are 200,379,391 bytes vs S6 200,379,129: a retrained model (crop 352, fastMRI auxiliary training), not S6 weights.
- **Precision matters for rank blends:** goodpjw2008 explains part of .954 -> .956: S6 written at %.6f made many tiny probabilities
  exact ties (Baker's: 43 distinct values on gold58), which a rank blend loses. R7's compact_infer / probe_dual now write %.9g
  ("v3 %.6f rounded ~3 % of cells to 0 / 1"). Our HFR7 F leg writes via pandas to_csv (full precision): OK. Any new leg we add
  must also write full precision.
- nartaa dataset holds exactly two checkpoints (r384 used everywhere, c08_r224 unused by R7). Danial / dreaddevelopment have not
  released newer weights (dreaddevelopment: withholding his 0.945 weights because of forking, 748181).

## (b) Actionable discussion facts (threads updated since 10-12 07:40 IST)

- **748276 OAI eligibility (dragoshh1984): still 0 comments, no host reply.** Related 733965 (host thread on commercial LLMs, 5 h ago,
  Mahatva Goell): a new question asking whether the public OAI-trained weight sets (nartaa SWA, heliosli orthofoundation) are allowed
  for prize-eligible final submissions; unanswered. 748480 (nishchal jain) asks the same for OAI and fastMRI (application-based):
  no host reply. Note R7 itself has "fastMRI auxiliary training", so R7 carries fastMRI exposure as well as OAI (via Raptor).
  The only host statement remains 741819 (OAI ok if no institutional sign-off and free).
- **748181 (Cody_Null, 32 comments), Kaggle staff reply:** low-effort blends "tend to drop on the private leaderboard"; one-week
  publishing lock confirmed; report verbatim copies. heliosli: R7 trained on 8 x RTX PRO 6000; will keep sharing.
- **748295 "Stuck at .954":** rank 5 (azamat1ch) and rank 6 (diet1236364) say they use no external data; azamat1ch's one hint:
  "Short answer is labels". Danial Zakaria (nartaa) confirms the earlier .956 heliosli notebook used hidden weights; public copies
  got .954.
- **746792 (Danial Zakaria, 3 h ago): full training recipe of the .949 Raptor posted.** Targets = mean(own Gemini 3 Flash labels,
  v4_blend), then mean with JEV HYBRID where present; OAI = 2,399 baseline knees, MOAKS rules for PF OA / Lateral OA (full-thickness
  loss >= 1 in the listed subregions) / Synovitis (MSYIC >= 1), hard 0/1, other cells masked; backbone
  coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k, native 384 full FOV, 12 random 3-slice windows per step, +-10% intensity jitter only,
  batch 1 x accum 16, AdamW wd .0588, LR 4.01e-5 / head 1.95e-3, OneCycle 15% warm-up, 16 epochs, pos_weight (1-p)/p clipped
  1-10, bf16; selection = uniform average of the 3 best epochs by gold58 (24 windows). 16 epochs ~22 h on an 8 GB laptop GPU.
  The only non-public input is their Gemini half of the targets.
- **745861 hengck23 "Tricks" (new comments):** Prateek Grover: training only on studies where several LLM labellers agree with a
  strong signal (> .9 or < .1) dropped ~25% of samples and improved CV and public LB. hengck23: let the label prompt say
  "I don't know" and drop those; random / other-body-part images as sure negatives to suppress false positives; an LLM GT that
  scores .89 on gold58 gives ~.92 gold58 val and ~.930 public.
- 748111 "Single sub 0.960" (Optimo): rumour that the .960 entry is basically a single model; no detail.
- 748011, 747613, 743416: nothing new beyond the 10-12 digest (743416: a user claims NDA lists OAI as open to all countries; not
  a host statement).

## (c) Ranked levers for the next ~2 days

| # | Lever | Expected gain | Cost | Notes |
|---|---|---|---|---|
| 1 | **Read HR7 / HFR7-R20 by in-block rank** (GetLeaderboard API: our team's shown submissionId + rank) | 0 (it is the measurement) | free | If 57137635 (HFR7) becomes our team best at .956 and sits near the top of the .956 block (r124-~200), our fleet adds over R7 on public (MEASURED, rank-based). If HR7 is shown, HFR7 < R7 copy. Also gives where the R7 copy cluster sits |
| 2 | **HFR7-RR = R7 + Raptor 224 (A224 .35 in the Raptor slot) + our F at .2** | GUESS +.0002 to +.0006 over HFR7 (third-party rank move 233 -> 156 in the .956 band for the 224 alone; our F evidence from HF-R20 top-of-.954) | 1 build (agent 3's HR7-RR + the F cell from build_hfr7.py), +25-40 min hidden runtime, 1 slot | Submit after the HR7 / HFR7 read so it is a clean one-sided test vs the current best. Check the total hidden wall (R7 ~3.3 h + F + 224) stays well under 9 h; keep the 224 time guard |
| 3 | HFR7 variants on the F leg: w .3, + D2 / D3 seeds, + T123 s1044 when done | MODEL +.0001 to +.0003 each | builds exist; 1 slot each | Only if lever 1 shows HFR7 > HR7. One change per submission, ordered so each should beat the last |
| 4 | Raptor seed 2 with Danial's now-public recipe (box, OAI MOAKS rules known; targets v4_blend + JEV HYBRID, Gemini half replaced by our labels) | MODEL +.0003 to +.001 as a second member of the strongest family | ~0.5 day port + ~12-20 h V100 (GUESS from 22 h on 8 GB) + ~25 min hidden | Too tight for a visible move in 2 days; real candidate for the 10-22 private pick. Competes with PLAN_960's R7-recipe job for the GPU; Raptor recipe is the simpler port (single model, no state targets) |
| 5 | Per-class weights on the R7 blend (e.g. F only on labels where OOF says it helps) | GUESS 0 +- .001 | 1 build, 1 slot | No public evidence on the R7 base; overfit risk. Only under a prereg |
| 6 | goodpjw 2.5D MIL reader on the R7 base | MEASURED (third party) about -.002 at full weight; 1/10 weight being scored in goodpjw v11 | n/a | Skip unless v11 shows a gain |
| 7 | Per-class PROBE_CLASS submissions | none | slots | A probe scores ~.54 and never becomes the team best, so it is invisible in rank; display resolution on AUC_c is only ~.012. Skip |

Final-pick notes: staff expects low-effort blends to drop on private; top-5/6 claim no external data; the OAI ruling is still open and
R7 adds fastMRI exposure. Keep one pick whose differentiation is our own models (HFR7 family) and decide the second when 748276 /
733965 get a host answer.
