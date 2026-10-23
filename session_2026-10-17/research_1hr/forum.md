# Forum lens 2026-10-17 (~22:00-22:30 IST = 16:23-16:35 UTC). No em dashes.

Scope: every competition discussion topic and comment new or updated since **10-16 07:00 UTC**, plus a re-scan of the
top ~40 most-voted / key threads of the whole competition for anything we never acted on. READ-ONLY: anonymous
forum API through our own session_2026-10-16/forum/fr.py + msgs.py (no cookies of Eesh). Nothing was posted, voted,
submitted, forked, created or versioned. Raw dumps: research_1hr/forum_raw/ (l5_*.json topic lists, new/ = 14 updated
threads, top/ = 39 top / key threads incl. nartaa dataset discussion 750180, new2/ = last re-check 16:31 UTC;
flattened text: new_since.txt, all.txt, host_posts.txt, hi_rank.txt).
Tags: MEASURED (forum text we fetched, or our own files) / CLAIMED (a third party said it) / MODEL / GUESS (ours).
"rank N" next to a name = that author's current competition rank as returned by the forum API (MEASURED field
`competitionRanking`; it is a live rank, not their rank when they posted).

## 0. Bottom line (5 points)
1. **No host or Kaggle-staff post anywhere since 10-14 19:56 UTC** (Dustin, stuck-submission fix). MEASURED: the full
   host/staff post list (section 1) was rebuilt from 53 threads. No announcement exists on private split, reruns, final
   selection, or external-data enforcement. GUESS: P(any host answer on external data before 10-22 23:59 UTC) ~15%.
2. **New 10-17: a pressure wave on external data** (750236 + four one-dataset posts 750257-750260 by one account),
   naming exactly the components in our picks: nartaa OAI CoAtNet weights, heliosli `anatomy.pt` (SKM-TEA), heliosli R7
   (fastMRI), RadImageNet. The community downvoted the four posts to **-19 / -20** within 3 h (MEASURED), and none has
   a reply. Host's 10-14 rulings stand: public weights "are eligible for awards" (744056) and registration-only data
   "clearly" qualifies (743416).
3. **Our exposure, MEASURED from our files:** both final picks (A = HW 57262115, B = R6 57248084) contain OAI (nartaa
   CoAtNet 384 + 224, `*_oai_d96_*` files) and SKM-TEA (`anatomy.pt` places the R6 and S6fall crops). **No fastMRI**:
   the R6 v1 manifest has no `fm_aux` key, init = ImageNet convnext_small; fastMRI enters only through R7, which is in
   no pick (R7 late 57291336, hw3r7). No RadImageNet / DINOv2 dataset attached. So the 750259 (R7) and 750260
   (RadImageNet) questions cannot hurt our picks; 750257 (OAI weights) and 750258 (anatomy.pt) would hit both picks
   equally. A "clean" hedge pick does not exist in our ladder (every leg depends on anatomy.pt or OAI).
4. **Nothing new from .959+ teams that is buildable in 5 days.** The few top-ranked posters who spoke say "labels"
   (image-side pseudo-labels, AI-made, "don't rely on reports alone") and two of them say "no external data". No one
   at .959+ has published TTA, rank-vs-prob, per-finding weighting or post-processing details.
5. **No new runtime / stuck / swap issue that touches us.** nartaa updated `rsna-knee-s6fall-weights` on 10-17 (added
   `training_code/`); our A16 probes committed after that and their S6fall output equals HW v1 exactly (MEASURED,
   gate files), so the update was not a weights swap.

## 1. Host and staff rulings, complete list (MEASURED, host_posts.txt)
| date (UTC) | topic | who | ruling / content |
|---|---|---|---|
| 08-09 | 733965 | Po-Hao Chen (host) | commercial LLM APIs on report text allowed; not "private sharing" |
| 08-25 | 734109 | Po-Hao Chen | **KneeCoT NOT allowed** (institutional agreement) |
| 08-27 | 733965 | Po-Hao Chen | non-commercial licences alone do not exclude a dataset; click-through registration "would generally satisfy"; IRB / institution approvals may not; teams own the licence risk |
| 09-21 | 741819 | Po-Hao Chen | OAI does not violate the rules "if it is generally accessible to all participants without an escalated (i.e. institutional) review ... and is free" |
| 10-12 | 748181 | Ryan Holbrook (Kaggle) | no anti-blend policy; one-week notebook publishing lock; "low-effort ensembles ... tend to drop on the private leaderboard" (opinion) |
| 10-14 13:18 | 743416 | Chris Carr (RSNA) | will not designate specific datasets; registration + licence acknowledgement "clearly" qualifies; institutional agreements "clearly do not"; disclose datasets |
| 10-14 13:20 | 744056 | Chris Carr | **submissions incorporating publicly shared weights are eligible for awards**; list the weights and their source datasets with the submission |
| 10-14 19:56 | 748403 | Dustin (Kaggle) | cleared a submission stuck > 24 h (reported 10-11 20:11) |
Since then: silence. Open, unanswered: OAI geo-block (749145, 748276, 743416 Richard_61 10-16 06:11), withdrawn public
weights (744056 zhao_hapi 10-10), downstream OAI checkpoints (733965 10-13, 747455 10-13), and the 10-17 wave below.
hengck23 (10-16 09:02, 743416, CLAIMED): "The host already said they won't say yes or no on specific data ... move on."

## 2. Everything new or updated since 10-16 07:00 UTC (14 threads, 30 messages; MEASURED new_since.txt)
| topic | last post (UTC) | who (rank) | gist | matters to us? |
|---|---|---|---|---|
| 750236 | 10-17 11:59 | Michał Choiński (1442), votes -3 | asks 6 yes/no: is external-data compliance checked for medal positions or only prize teams; are non-compliant entries removed; are public checkpoints with OAI / fastMRI / private pretraining allowed for teams that never touched the data; fork responsibility; OAI vs NIH country block; pin before 10-22. Argues rational teams hedge one pick with and one without external data | yes: frames the hedge question (section 3) |
| 750257 | 10-17 13:39 | Karan Bhatti (2849), **-19** | nartaa OAI CoAtNet weights "in 90% public notebooks at 0.95+" (CLAIMED); can a final sub with them win a prize or medal? | yes: in both our picks |
| 750258 | 10-17 13:40 | Karan Bhatti, **-20** | heliosli orthofoundation `anatomy.pt` = "unchanged SKM-TEA/KneeXNet auxiliary weights", "no new open-source license is granted" (quotes the dataset page, CLAIMED); in ~70% of 0.95+ notebooks; allowed? licence vs CC BY-NC winner licence? | yes: in both our picks |
| 750259 | 10-17 13:41 | Karan Bhatti, **-20** | R7 uses fastMRI aux (`fm_aux`); fastMRI needs an affiliation + intended-use request; allowed? | no: R7 not in our picks |
| 750260 | 10-17 13:42 | Karan Bhatti, **-20** | RadImageNet ResNet-50 in ~60% of 0.90+ notebooks; images "by request"; licence MIT vs CC BY-NC-SA | no: not used |
| 749359 | 10-17 06:04 | nartaa (245), -2 | **S6 + CoAtNet training code now public** (Apache-2.0) in `rsna-knee-s6fall-weights/training_code/`; CoAtNet SWA epochs were ranked on Gold-58 | info only (see 750180 below) |
| 750180 (dataset discussion) | 10-17 06:03 | nartaa | S6 final weights = epoch 16, no SWA/EMA, **nothing selected on Gold-58 for S6**; targets = blendjev only (no prior fill, no per-cell weights); 86 studies missing from the pool were deleted locally by nartaa (not a host change); CoAtNet: 2,399 OAI knees supervise only PF OA, Lateral OA, Synovitis; **CoAtNet SWA = top-3 epochs ranked on Gold-58** | small: our CoAtNet half carries mild Gold-58 selection; S6fall does not (CLAIMED) |
| 749858 | 10-16 16:41 | Noir.s (96) | "Are external datasets allowed? I'm still confused" (edteoh's 10-16 05:50 table, already read: R6 fe055406 has "no fm_aux entry", R7 = fastMRI, nartaa = OAI, Raptor dreaddevelopment = none, anatomy.pt = SKM-TEA) | confirms section 3; R6 point now MEASURED by us |
| 748181 | 10-17 16:27 | heliosli / D Chakraborty / Y. Alouini | flame war on public blends and AI-written posts; heliosli: the .958 "came entirely from my work on the modeling side", turned comments off on "Blend mine in" (AK, CLAIMED); no method detail | no |
| 748295 | 10-16 14:49 | heliosli / AK / D Chakraborty | same argument; heliosli says he spent > $10k of credits (CLAIMED) | no |
| 743416 | 10-16 09:02 | hengck23 (880) | "move on", host will not rule on specific data | confirms silence |
| 735304 | 10-16 10:17 | colum2131 (38) 10-16 04:28; N. Karcher | colum2131: pseudo-labels "generated from the images, with the image model taking the same input as at submission time", blended with LLM labels "at a fixed ratio"; Karcher: checkpoint selection only fair on Gold-58 "but that is very noisy" | no new lever |
| 747658 | 10-16 09:16 | Kirderf | VLM label benchmark chatter, Kaggle quota limits | no |
| 743698 | 10-16 20:44 | Arun (764) | scratch .92 + public fusion = .93; asks Deotte how to tell if a public model overfits (no answer) | no |
| 749798 | 10-16 13:58 | Thin Sweet | hidden-rerun "Notebook Threw Exception" despite a 1,300-study T4 test; Deotte points to 749623 | no (our picks already scored) |
Re-check 16:31 UTC: only 748181 moved (two Yassine Alouini comments, no content). No new topic after 13:42 UTC.

## 3. External-data exposure of our final picks, and the hedge question
MEASURED (session_2026-10-16/hw/kernel_v224r6hws6fj/kernel-metadata.json + notebook cells 4, 7, 10, 11, 14, 18;
session_2026-10-15/r6v1/raw/manifest.json; session_2026-10-17/probes/gate_*.txt and out_*/ logs):
| component in HW 57262115 (and R6 57248084) | dataset | external data in training |
|---|---|---|
| R6 compact ConvNeXt-S, bundle fe055406 (our private copy of heliosli v1) | eeshsaxena/rsna-knee-r6-fe055406-v1 | manifest: no `fm_aux`, no fastMRI / OAI key, init convnext_small in22k_ft_in1k, `gold_in_training` 0. Crops placed by anatomy.pt |
| `anatomy.pt` (crop anchors for R6 and S6fall; sha asserted in cell 7) | heliosli/rsna-orthofoundation-fold0-epoch7-20261004 | SKM-TEA (CLAIMED by heliosli / nartaa / edteoh) |
| CoAtNet 384 `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` + 224 `..._oai_d96_c08_r224_swa.pt`, rank .65 / .35 (the "Raptor half") | nartaa/rsna-knee-publication-swa-weights-20261007 | OAI, 2,399 knees, 3 labels (CLAIMED nartaa); SWA epochs ranked on Gold-58 |
| S6fall ConvNeXt `E_prior_s42_fall.pt` (sha 0c7297d58f06... in our run log = nartaa's announced 0c7297d5...) | nartaa/rsna-knee-s6fall-weights | none directly; anatomy.pt anchors; CoAtNet fallback for unreadable studies |
| JPEG codecs | glorysing/rsna-knee-offline-codecs-r09 | software only |
- MODEL: a pick with no external data would have to drop R6, S6fall and the CoAtNet half, i.e. everything we have.
  The only clean public model is dreaddevelopment's Raptor baseline, "5 fold raptor sub .936-.938" (737696, CLAIMED),
  far below the bronze line (~.956 public). So hedging a pick buys nothing in every world except the one where Kaggle
  strips medals from almost all .95+ teams.
- GUESS on that world: P(host rules against public OAI-trained or SKM-TEA weights for medals before final ranks) <= 3%.
  Reasons: the 10-14 award-eligibility ruling for public weights; no host post in 3 days; community downvotes; Tucker
  Arrants (rank 1, 09-26 and 09-30, CLAIMED) and Eugene Khvedchenya (GM, 09-30, CLAIMED): only prize-eligible teams
  get audited; DeepFake (2020) precedent removed only PRIZE teams (CLAIMED, 743416). Top-25 teams also use OAI
  (minhtu.mt.mt rank 23, 10-08: downloaded OAI "extra 24M data", CLAIMED), so a ban would not be selective against us.
- Medal-eligibility needs no write-up. 744056 asks award winners to list weights + source datasets; for us that would
  be: nartaa publication-swa (OAI, NDA Study 3407), nartaa s6fall, heliosli orthofoundation anatomy.pt (SKM-TEA),
  heliosli R6 fe055406 v1 (competition data), dreaddevelopment Raptor trainer lineage.

## 4. What .959-.967 teams say worked (all-time re-scan of posts by rank <= ~140; MEASURED hi_rank.txt)
- **azamat1ch (rank 2)**, 10-09: "All single architectures, 5-fold (public LB): ConvNeXt-T 0.959, EfficientNetV2-S
  0.956, CoAtNet 0.952" (CLAIMED). 10-12: "haven't used any external datasets"; the secret: "Short answer is labels";
  "AI, not by hand. And don't rely on reports alone" (i.e. image-side relabelling, GUESS).
- **diet1236364 (rank 13)**, 10-12: "don't use external data" (CLAIMED, half sarcastic). 08-22: 5-fold single 0.929
  @224, LLM labels only.
- **YYama (rank 19)**: no external data (10-07); team, will share after the end.
- **datnt114 (rank 44)**, 10-10: best public .961 with 5-fold OOF .937 and Gold-58 .938; "i trust cv local" (CLAIMED).
- **colum2131 (rank 38)**: ConvNeXt-T 2.5D .953 + DINOv2 ViT-S .951 = .955 (10-09); image pseudo-labels blended with
  LLM labels at a fixed ratio (10-16); across 15 subs, LLM-label OOF AUC vs public r = .87, Gold-58 vs public r = .66
  (10-10); notes the 1 - p trick lets strong teams hide their public score (CLAIMED).
- **Tucker Arrants (rank 1)**: 224/288 px, simple CNN (ResNet / EffNet), CV against report labels only, noise
  threshold ~.003, image "student" beats the label "teacher"; better label extraction never moved his LB (08-31, 09-10).
- **tennogh (rank 14)**: .942 raw, .943 with TTA @288 (09-03); OOF pseudo-labels correlate with LB.
- **Raymond Yuen (rank 39)**: multi-source weak labels -> teacher models -> pseudo-labels -> final model; cheap LLMs fine.
- **Prateek Grover (rank 179)**: train only on cells where several LLM labellers agree and p > .9 or < .1 (drops ~25%
  of samples); "improved my results on CV and public LB" (CLAIMED).
- **hengck23 (GM, rank 880)**: random / other-body-part images as hard negatives suppress FP (10-12); 58-shot training
  only with ~10 parameters on a frozen prior (10-11).
- **nartaa (rank 245)**: the CoAtNet in our picks trained with "No flips, no affine transforms" (746792, 10-13); public
  split "SD of roughly 0.003" (749359, CLAIMED).
- Nobody at .959+ disclosed TTA, rank vs prob averaging, per-finding fusion weights or post-processing. MODEL: the top is
  driven by own labels + own 5-fold CNNs, not by fusion tricks; none of it is reachable in 5 days without training.

## 5. Leaderboard and selection mechanics claims
- Ties: "Ties are resolved by submission time, not by score" and "about 135 teams per thousandth" between .941 and .956
  (Shenzai25, 749570, 10-15, CLAIMED; consistent with our band counts in PROGRESS_17: .958 band ~750 teams).
- Hidden strength: 1 - p submissions hide a team's real public score (colum2131, CLAIMED). MODEL: some teams below .958
  publicly may sit above us privately; our margin to the silver cut (r171 vs ~r306) is ~135 ranks, which absorbs a few
  dozen such teams.
- Ryan Holbrook (staff): low-effort public ensembles "tend to drop on the private leaderboard" (opinion, 10-12). Our A
  pick uses heliosli's public-tuned per-finding weights; B (R6 uniform) is the hedge on exactly this, as in A15.
- Efficiency LB uses the top-two public submissions unless selected (Chikuwabu, 736678, 08-22, CLAIMED). Irrelevant to
  medals.
- No host statement on the private split (stratified or random), on rerun determinism, or on what happens to a selected
  submission that later errors. MEASURED absence.

## 6. Runtime failures, stuck submissions
- 749623 / 749798 (10-15, 10-16): hidden rerun "Notebook Threw Exception" while the visible run passes; Deotte's advice:
  commit on 1,300 random train studies, guard NaN and malformed DICOM (CLAIMED). Not relevant to already-scored picks.
- 748403: a submission stuck > 24 h was cleared by staff ~3 days after the report (MEASURED dates). IF any further entry
  sits "running" > 12 h, THEN do not wait on staff (lesson stands from the 10-15b digest).

## 7. Bait and dataset swaps
- heliosli/rsna-knee-r6-frozen-fe055406 current version holds R7, not R6 (edteoh 10-16, CLAIMED; we already MEASURED
  this on 10-15 and use our own copy of v1, manifest sha matches).
- nartaa/rsna-knee-s6fall-weights updated 10-17 (~06:00 UTC): `training_code/` added, README corrected. MEASURED no
  effect: A16 probe wr45 (committed after the update) gate "S6fall B == HW v1 maxdiff 0.00e+00" and model sha
  0c7297d58f06... in the run log.
- heliosli/rsna-knee-compact-v1 (S6) weights withdrawn (744056 zhao_hapi, 10-10, CLAIMED); not used by us.
- heliosli turned comments off on the .958 "Blend mine in" notebook (AK, 10-16 07:13, CLAIMED). Its per-finding weights
  are the ones in our A pick; no further source for how they were chosen.
- MODEL: our submissions are already scored on the full hidden set at submission time, so any later dataset change
  cannot alter the private score of a scored entry.

## 8. Top-thread re-scan: things raised that we never acted on (and why they stay closed)
| item | source | status |
|---|---|---|
| 3DReasonKnee OAI boxes / segmentation | 745861 hengck23 10-15 | closed (HANDOFF sec 3: needs a new leg, no runtime room) |
| image pseudo-labels blended with LLM labels | 735304 colum2131, 748295 azamat1ch | needs retraining; R7-teacher student already measured closed |
| label-agreement filtering (drop low-agreement cells) | 745861 Prateek Grover | needs retraining; no slot to validate |
| hard negatives from other body parts | 745861 hengck23 | needs retraining |
| flip TTA vs a CoAtNet trained without flips | 746792 nartaa | already covered: our forced flip is on the R6 family only (cell 8, FLIP_TTA_FORCED_ON); the CoAtNet 384 runs nartaa's own published anatomical MIRROR_TTA (cell 4, as in his 0.949 post title); the r224 mirror was measured and closed (task brief) |
| LLM-label OOF as validator (r .87 with LB) | 748011 colum2131 | not usable for fusion-weight choices (our labels are the public blendjev family, same as targets) |
| team merge (Adel Zebiche DINOv2 .952, pand999) | 749426, 735304 | closed: merger deadline 10-15 23:59 UTC passed (MEASURED comp metadata) |

## 9. Leads (ranked by expected private-LB value for us)
1. **Keep A15 picks; do not burn a pick on an external-data hedge.** Evidence: section 3 (MEASURED exposure, CLAIMED
   rulings). Value: protects ~0 to +1 medal step vs a clean pick that cannot medal. Cost/risk: tail risk <= 3% (GUESS)
   that both picks are hit together. IF a host or staff post before 10-22 23:59 UTC says public OAI-trained weights or
   SKM-TEA `anatomy.pt` are not allowed for medals, THEN tell Eesh at once and re-open the hedge (only candidate: a
   dreaddevelopment-Raptor-only entry, ~.937 public CLAIMED, needs a slot and a new kernel); ELSE nothing.
2. **One more forum read before the last A15 read (10-22 ~17:30 IST).** Value: catches a late host ruling or a
   selection / rerun announcement. Cost: 2 minutes (fr.py list + msgs.py on new topic ids; host filter =
   javaduke95, cdcarr, michelleriopel, shosys, ryanholbrook, macruzbar, herbison). IF a ruling on 750257 / 750258
   appears, THEN lead 1; IF a ruling on 750259 (R7 / fastMRI) appears, THEN no action (no R7 in picks).
3. **Prepare (do not post) a resource list for our picks** (section 3 table) in case the hosts or Kaggle ask medal teams
   for disclosure. Value: compliance insurance only. Cost: none now. IF asked, THEN Eesh posts it himself.
4. **Treat heliosli per-finding weights as public-tuned (B pick stays the hedge).** Evidence: 749570's analysis of
   public per-label fusion knobs (different notebook, same mechanism); staff opinion on blend optimism. Value: already
   priced into A15. IF A16 probes move the best-id, THEN A follows R1 as written; B stays the uniform-weights entry.

## Not done
- Did not read Discord or Rednote (host: Discord unmonitored). Did not open notebooks (other lens). Did not re-read
  forum threads older than 10-08 beyond the top / key list in forum_raw/top.
