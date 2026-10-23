# Forum digest 2026-10-15 ~01:35 IST (Claude, read-only forum lens of workflow "silver-levers")

Window: everything posted or edited after the 10-14 digest (FORUM_DIGEST_10-14.md, read up to ~10:35 UTC 10-14), plus a re-read
of every thread active since 10-12 so nothing was missed. Last activity seen: 10-14 19:56 UTC (fetch 20:03 UTC = 01:33 IST).
Tags: MEASURED / CLAIMED (a third party said it) / MODEL / GUESS. Nothing was posted, voted, forked, submitted or uploaded.

Sources and tools
- Kaggle's own discussion endpoints, called anonymously (no cookies): GetTopicListByForumId (forum 12044884, sorted by
  recent / new / hot), GetForumMessagesInTopic + BatchGetForumMessages (full bodies, including the collapsed comments that
  GetForumTopicById hides). Raw JSON: `session_2026-10-15/silver/forum_raw/` (msgs_*.json, batch_*.json, list_*.json).
  Reader scripts (scratchpad, not in the repo): fr.py, msgs.py, show2.py, grepf.py.
- The `rank=` shown for a poster is their CURRENT team rank (Kaggle's competitionRanking field; checked against the LB:
  madmax0404 15, Driz Lo 38), mapped to scores with `research960/lbprobe/api_lb_1015T0135.json` (19:46 UTC).
- kgweb.py (anonymous page model + version table) for nartaa/rsna-knee-s6fall-solo and heliosli/0-9573-...; v1 of the heliosli
  notebook pulled read-only to `silver/pulled/forumlens_heliosli_9573_v1/` (read as data, never run).

LB reference (19:46 UTC, 6,008 teams): .967 r1 | .966 r2-4 | .965 r5-7 | .964 r8-15 | .963 r16-23 | .962 r24-31 | .961 r32-42 |
.960 r43-59 | .959 r60-77 | .958 r78-113 | **.957 r114-228 (115 teams)** | **.956 r229-1113 (885)**. Us: V224-FLIP r294.

## 0. The five facts that matter for a comfortable silver

1. **HOST RULINGS ARRIVED (10-14 13:18 and 13:20 UTC, Chris Carr, Director of Informatics, RSNA). MEASURED.**
   - 743416 (external datasets): no per-dataset list, but "Datasets that require only registration and acknowledgement of
     licensing terms would clearly meet these criteria. Datasets that require an institutional research agreement or limit
     access to users with institutional affiliations clearly do not." Plus the reasonableness rule and a request to disclose
     every dataset used.
   - 744056 (public weights trained on competition data): "submissions incorporating publicly shared weights are eligible for
     awards"; list the shared weights and their source datasets in the submitted resources.
   - Combined with the 09-21 host condition (741819: OAI is fine if there is no institutional review/sign-off and no fee) and our
     MEASURED NDA permissions page (10-08: OAI "Institutional Approval Required: No", approved immediately), **OAI now reads as
     allowed** (GUESS, high confidence). The only open point is the geo block: Driz Lo (rk38, .961) asked 13:26 UTC whether an
     LG22 Login.gov block from some regions still counts as "equally accessible". No answer after 6.5 h. 749145 (Mobassir) and
     748276 still have no host reply.
   - Consequence: the nartaa SWA weights (OAI-trained), the R7 bundle and our OAI fleet members are eligible under the stated
     rules. The OAI-light fallback pick is no longer needed as a hedge (lever 3).
2. **heliosli's own .957 is public, and it is R6, not R7. MEASURED.** Notebook `heliosli/0-9573-latest-public-dataset-score-in-one-click`
   v1 (run 10-14 12:13 UTC, public .957, title "0.9573") is the V224 structure: Raptor r384 + r224 (.65 / .35) as the Raptor half
   .5, plus "ours" .5 = **R6 bundle fe055406fad6a12f** (4 folds + its own fluid / state stackers, no flip). It came from dataset
   `heliosli/rsna-knee-r6-frozen-fe055406` **version 1** (version id 20611930). The cell diff against v2 shows only the bundle
   changed (BUNDLE_ID, header text). At 18:12-18:13 UTC heliosli put the **R7** bundle (8c7b666e) into dataset version 2 under the
   same file names and re-ran the notebook as v2. The current version is therefore plain V224 (= .956 class). heliosli's team
   best moved .956 -> **.957** with submission 57183099 (team rank 203 -> 171, last submission 13:02 UTC). This is the cleanest
   public evidence of a .956 -> .957 step: R6 fe055406 beats R7 in the same blend on public (one display step; selection bias
   applies, see lever 1).
3. **The .957 band is filling from copies. MEASURED counts, GUESS on cause.** Teams entering .957+ per LB window (UTC): 8
   (05:27-07:08), 9 (07:08-10:24), 9 (10:24-15:58), **15 (15:58-19:46)**. In the last window **7 of the 15 jumped from BELOW .956**
   (.948 / .952 / .954 x3 / .955 x3 -> .957). A jump past the whole .956 copy pile in one step is the signature of forking a public
   .957; heliosli's v1 (R6) is the only all-public .957 notebook in that window (watch run 19:10 UTC). Whoever pins dataset v1
   gets R6; whoever forks the current v2 gets R7 (.956). Expect the .957 band to keep growing.
4. **New public model: nartaa S6fall, .950 single, OAI-free (749359, Danial Zakaria, 10-14 18:52 UTC). MEASURED / CLAIMED.**
   heliosli's S6 ConvNeXt-small (per-finding spatial attention + window-attention MIL, full field of view + 100 mm knee-centred
   local crops placed with the SKM-TEA anatomy V-Net anchors) retrained from ImageNet-22k on nartaa's JEV-blend soft labels, all
   4,263 studies, gold58 excluded, 16 epochs. Public .950 alone (fold-0 version .948; their CoAtNet .949). Weights
   `nartaa/rsna-knee-s6fall-weights` (one 186 MB checkpoint `E_prior_s42_fall.pt`, CC BY-NC 4.0, "No OAI data was used").
   Notebook `nartaa/rsna-knee-s6fall-solo` v2: all inputs public (s6fall weights + heliosli orthofoundation + nartaa SWA as a
   per-study fallback), commit 158 s on T4. It is the first public model that shares the R7 architecture family but uses the
   Raptor label source, so it is a natural third leg for the R7 + Raptor blend (lever 2). The copy crowd has until the
   publishing lock to put it into public blends.
5. **Deadlines tonight.** The notebook PUBLISHING lock starts at the one-week mark (Ryan Holbrook, staff, 748181; Raagav Anand
   10-13 11:19 UTC: "in effect in 3 days"), i.e. **10-15 23:59 UTC = 10-16 05:29 IST** (GUESS on the exact minute). New
   VERSIONS of existing public notebooks and new dataset versions are not locked (CLAIMED, consistent with heliosli's 18:13 UTC
   swap). The **team-merger and entry deadline is also 10-15 23:59 UTC** (MEASURED, comp timeline). Expect a final burst of new
   public blends (S6fall, R6 v1, flip, mirror) during 10-15 UTC.

## 1. What moved teams from .956 to .957-.967 (evidence, by strength)

| what | who / where | tag |
|---|---|---|
| R6 fe055406 instead of R7 in the V224 blend: .956 -> .957 | heliosli team (57183099) and his public 0-9573 v1 | MEASURED |
| Copy jumps into .957 from below .956 (7 of 15 entrants 15:58-19:46 UTC) | LB snapshots | MEASURED counts, GUESS cause |
| R7 + Raptor + a private knee-centred-crop ConvNeXt at .2 | bliverrigpdoge public .957 (private input, known 10-14) | MEASURED |
| Labels, made by AI, "don't rely on reports alone"; no external data at all | azamat1ch (rk7, .965), 748295 10-12/10-13 | CLAIMED |
| "don't use external data", "follow the rules" | diet1236364 (rk13, .964), 748295 10-12 | CLAIMED |
| ".95+ single fold single model", "don't even try external dataset" | Ca Rot Production (rk49, .960), 747613 | CLAIMED |
| OAI "made very little difference in my pipeline" | Cody_Null (rk102, .958), 749145 10-14 03:57 UTC | CLAIMED |
| .960 single sub is "basically a single model" (Rednote post, no link) | 748111 | CLAIMED, unverified |
| Knee-centred 100 mm crops from anatomy anchors + full FOV | nartaa S6fall (.950 single), bliverrigpdoge (.957 blend) | MEASURED (public scores) |

Reading (GUESS): the .96x teams did not get there with external data or tricks; they have image-informed AI labels (teacher /
pseudo-label loops on the images, not report-only LLM labels, which cap at ~.89 gold58 per the Kirderf benchmark) and their own
model families. For our silver goal the forum offers no copyable .96 trick. What is copyable is (a) the R6 bundle step, (b) the
new S6fall model as an extra leg, (c) knee-centred crops (already in S6fall), and (d) our own image-informed labels (the R7-teacher
student, already in flight).

## 2. Thread notes since the 10-14 digest

### Rulings and rules
- **743416** (39 msgs): Chris Carr ruling above (13:18 UTC). Driz Lo (rk38) asks about LG22 geo blocks (13:26 UTC), unanswered.
- **744056** (7 msgs): Chris Carr, public weights eligible for awards; disclose weights + source datasets (13:20 UTC). Earlier
  open sub-questions (withdrawn heliosli compact-v1 weights, fine-tunes of public checkpoints, 2.8 training-code needs for
  third-party weights) were not answered individually.
- **749145** (OAI + NIH geo restriction, Mobassir): 6 yes/no questions, including whether a ban would cover OAI-derived
  pseudo-labels and fine-tunes, and LB-wide removal vs prize-only checks. Replies: madmax0404 (rk15) "I don't think they will
  [answer]", Satwik (rk78), Cody_Null (rk102), Wildata ("assume legal until confirmed otherwise"). No host reply; the 13:18 ruling
  in 743416 answers its core indirectly.
- **749250** (new, Steve421471 rk186 .957, 14:07 UTC): asks if medals are reviewed for rule violations (OAI in public notebooks).
  Nikita Churkin: hosts will not answer; prize-zone removals happen rarely. No host reply.
- **739905** (winner licence CC BY-NC 4.0 vs OSI): RealEclypso ping 17:37 UTC, unanswered. Not a silver issue.
- **748925** (labels from a non-public model, Marco Colussi): still 0 replies. Our R7-teacher labels come from a public model on
  public datasets; the 744056 ruling (public weights eligible, disclose them) covers that case well (GUESS).

### Leaderboard mechanics, private-LB expectations
- **748181** (public notebooks debate, 47 msgs): new 10-14 posts from Dread Development (rk1508, Raptor's original author),
  Cyrus, k256.dev ("the public notebook scoring 0.954 might surprisingly not be overfitting"; cites BirdCLEF+ 2026 where public
  forks held up) and Cody_Null (rk102): "forks win medals more often than you would think", "local validation is close to
  useless... the LB [is] the largest sample we have that actually matches the scoring". Staff (Ryan Holbrook, 10-12): low-effort
  public ensembles "usually suffer from optimism bias" on private. Reading (GUESS): nobody has evidence either way; do not plan on
  the copy pile collapsing on private. Our private edge must come from real gains, not from others' shake-down.
- **742327** (older, re-read): Georgy Mamarin's study of 11 medal comps (3,000-6,750 teams): teams just inside the bronze line on
  public medalled 24 % of the time (range 2-51 %). A rank of ~r294 vs a cut of r300 is not a safe private silver (MODEL view
  belongs to the LB lens).
- **748011**: varshitha's SD ~.003 public noise at ~390 studies; in-band order follows the unrounded score. Her helper notebook
  "real scores of every public notebook" (pulled to silver/pulled/varshitha_realscores/, read only) lists public notebooks by
  bestPublicScore via ListKernels; it does NOT give unrounded scores (its text also says ties break by submission time, which
  contradicts her own post; our 10-13 measurement agrees with the unrounded-score order).

### Models, labels, tricks
- **749359** (new): S6fall write-up (fact 4). Danial notes their S6 fold-0 beat their CoAtNet locally (OOF .907 vs .890, gold58
  .931 vs .919) but lost by .001 on public: "Read the 0.001-0.002 differences as noise".
- **747658** (Kirderf benchmark): vision-LLM pass on gold58 mosaics; most API models failed to finish 58/58; local Qwen 27B w4a16
  on 10 studies .606 macro AUC. Prompt posted (MSK-radiologist montage prompt, specificity over sensitivity). No usable scores.
  Not a lever in 7 days.
- **745861** (hengck23 tricks): only links since 10-13 (SGMRI-VQA, orthopred CAD). Earlier: few-parameter fitting on gold58;
  "I don't know" option in label prompts; random / other-body-part images as hard negatives. Nothing new to act on.
- **748295**: no new technical content after azamat1ch's 10-13 "AI, not by hand. And don't rely on reports alone."
- **748111** (.960 single sub): Raagav asks for the Rednote link (10-14 06:19 UTC); none posted.
- **744230** (bf16 PSA): bf16 autocast on T4 is emulated (~50x slower) and silently starves a time-guarded kernel. CHECKED for us:
  heliosli's compact_infer uses `--amp auto` -> fp16 when compute capability < 8 (line 624), so R7 runs fp16 on T4; the
  bf16 autocast lines in our V224 kernel sit in heliosli's TRAINING paths (state OOF, augment loop) inside the bundle cell, and
  the Raptor cells have no bf16. No action. Also: Md Kishor Morol (rk187, .957) offered an independent DINOv2 ViT-S arm for a
  merge (09-29); Ayiyeke Olori (10-14 17:28 UTC) says they are in merge talks with a .963 team.

### Runtime / infrastructure
- **748403**: a submission stuck "running" > 24 h (Volodymyr, rk170); Kaggle staff (Dustin) acted 10-14 19:56 UTC.
  **748575**: T4 x2 queue waits of ~1 h (10-12). Lever 6: do not leave final candidates to 10-22.
- Compressed DICOMs / JPEG: **no forum post on transfer syntaxes or decoders since 10-13** (grep over all fetched threads). The
  DEC insurance plan stands on the data-page text alone.
- Private split / planes: 744519 (are all planes present in hidden test?) still unanswered since 10-06. No host statement about
  the private split anywhere.

### Not actionable
749038 / 736301 (AI-use survey; Po-Hao Chen), 748858 / 748536 / 745091 (team-up posts), 748535, 748310, 733965 (no change since
10-13 05:11 UTC), 747455 (no host reply), 748480 (no change).

## 3. OAI ruling status (one paragraph for the final-pick decision)
Host statements, in order: 08-27 click-through registration "generally satisfies accessibility" (733965); 09-21 OAI OK without
institutional review/sign-off and free (741819); **10-14 registration + licence acknowledgement "clearly" meets the criteria,
institutional agreements/affiliation limits "clearly do not"** (743416), and public shared weights are award-eligible with
disclosure (744056). NDA lists OAI as "Institutional Approval Required: No" (our MEASURED 10-08). Open: the 2.6.b geo point
(LG22 from China), asked again by a rk38 team; no answer, and experienced posters expect none before 10-22. Verdict (GUESS):
OAI-trained public weights and our OAI-trained members are eligible; the residual risk is small and, if it ever materialises,
would hit the whole R7 + Raptor copy pile equally (every public .956/.957 notebook carries the OAI-trained nartaa SWA weights).
Disclose OAI, the nartaa / heliosli weights and their source datasets in any winner write-up.

## 4. Levers (forum-derived, ranked for the silver goal)

1. **R6 bundle step (fe055406), ideally R6 + R7 both in the "ours" slot.** Evidence: heliosli 0-9573 v1 = R6 + Raptor .957
   public vs the same notebook with R7 = .956 class (MEASURED display), heliosli team .956 -> .957 (MEASURED). Expected:
   +.0005-.0015 public unrounded for a swap (GUESS; selection-biased: heliosli published the version that scored best for him);
   averaging R6 and R7 halves that bias and adds a second 4-fold ConvNeXt family (GUESS +.0003-.001 over the better of the two).
   Cost: a build that pins `heliosli/rsna-knee-r6-frozen-fe055406` VERSION 1 (version id 20611930); R6 + R7 both costs about
   +1.3 h plain (MODEL, from the MEASURED 4-fold flip delta 1.29 h), so R6 plain + R7 flip fits ~5.9 h before any fleet.
   Risks: dataset v1 may not be attachable any more (heliosli overwrote the files in v2; whether v1 stays downloadable must be
   checked, the ingredients lens has manifest_r6frozen_v1); the R6 manifest has no flip_tta key (flip on R6 must be forced and
   is untested); many teams will pin v1 too, so a plain swap only keeps pace.
2. **S6fall as a third leg.** One OAI-free ConvNeXt-S checkpoint on the Raptor label source; .950 alone (MEASURED). Expected
   +.0003-.001 unrounded on the V224-FLIP base at w .15-.25 inside the "ours" slot or as a separate rank leg (GUESS). Cost: one
   extra compact_infer pass, ~+0.4-0.8 h hidden (MODEL: ~1/4 of R7's 4-fold pass plus its own decode / anatomy step unless it
   shares R7's cache); check compact_infer version compatibility (Danial ran heliosli's orthofoundation compact_infer, R7 ships
   its own). Risk: the copy crowd adds it before the 10-15 23:59 UTC lock; correlation with R7 (same architecture family).
3. **Drop the OAI-light hedge from the 2 final picks.** After the 10-14 rulings both picks can be the best OAI-inclusive
   candidates (for example the best differentiated blend + a robust R6/R7/F average) instead of reserving one for an OAI-free
   fallback. Expected: indirect (a better second pick on private). Cost: none. Risk: a late geo ruling (low, GUESS < 5 %), which
   would hit the whole copy pile equally.
4. **Image-informed labels (R7-teacher student) stay the main own-work lever.** The only consistent message from the top-15
   teams (rk7, rk13; rk49, rk102 on external data) is labels from models looking at images, no external data. Our FF-S / FFSR
   entries are exactly that. Expected small on this base (+.0001-.0003 per the runtime review MODEL) but it is the one ingredient
   the copy pile cannot fork. Keep it; make sure the S pack has its own public-score read before the final picks.
5. **Watch the lock-day burst and re-versions.** 10-15 UTC is the last day for new public notebooks; heliosli has re-versioned a
   public notebook and dataset within 6 h before. Run watch.py every ~1 h on 10-15 UTC and once a day after (versions are not
   locked). React in the same UTC day: each public step up moves ~50-100 teams past us (MEASURED .957 band growth 95 -> 115 in 9 h).
6. **Do not leave final candidates to 10-22.** Stuck submissions (> 24 h) and ~1 h queues are reported; V224-FLIP-class runs
   take ~4.6 h submit-to-score and FF-S ~6.2 h (MODEL). Last risky candidate in by 10-21 morning UTC.

Closed by this sweep: bf16-on-T4 (our inference already fp16), team merge (deadline tonight 10-15 23:59 UTC; Eesh ruled merges
out on 10-08), per-dataset external-data probing (host declined a list), vision-LLM relabels (no working benchmark).
