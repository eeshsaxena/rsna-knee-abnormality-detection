# Forum + public-notebook digest 2026-10-14 ~16:05 IST (Claude, read-only sweep)

Sources: built-in browser (discussion list sorted by recent comments and by votes; thread bodies and every comment read through
Kaggle's own page endpoint GetForumTopicById, anonymous), the public-release watcher `session_2026-10-15/watch/watch.py` run 4
(15:55 IST: 381 notebooks listed, 113 re-read, public score from the notebook page model, inputs of the scored version;
log `session_2026-10-15/watch/watch_run4.out`), `kaggle kernels pull -m` copies in `session_2026-10-15/watch/pulled/` (cell
diffs vs R7 `heliosli/0-957-latest-dataset-score-in-one-click` and vs V224 `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956`),
`kaggle datasets list/files`, and the API LB snapshot `session_2026-10-14/research960/lbprobe/api_lb_1014T1555.json` (10:24 UTC).
Window: threads with comments since 10-13 08:00 UTC (overlaps the 10-13 digest by ~2.5 h). Nothing was submitted, voted,
commented, forked or downloaded from the competition. Tags: MEASURED / CLAIMED (a third party said it) / GUESS.

## 0. Leaderboard (API, 10:24 UTC, 5,967 teams)

| band | teams | ranks | vs 10-13 10:26 UTC |
|---|---|---|---|
| .966 | 4 | r1-4 | was 1 team |
| .961-.965 | 37 | r5-41 | |
| .960 | 17 | r42-58 | was r33-49 |
| .959 / .958 | 17 / 35 | r59-75 / r76-110 | .958 was r60-80 |
| **.957** | **95** | **r111-205** | **was 43 (r81-123)** |
| .956 | 789 | r206-994 | was 390 |
| .955 / .954 | 51 / 321 | r995-1045 / r1046-1366 | |

- Cuts by rule: gold r21, silver r298, bronze r596. **Us: V224-exact (57171955) r486, .956** (281st of 789 in the band).
- MEASURED: between 05:27 and 10:24 UTC today 24 teams moved up into .957+ (13 of them .956 -> .957, Choruru .956 -> .960,
  bunnywasnothere .953 -> .958). Only one of the 24 (renta.k) has public notebooks, and those are old (.937-.941, Sept).
  So the .957 band growth is private work on top of the public bases, not a public release.

## 1. Public notebooks and datasets

**No public notebook scores above .956. No RELEASE_CANDIDATE.** 25 notebooks sit at .956, all R7 or V224 family on the same
three public inputs (R7 bundle `heliosli/rsna-knee-compact-r7cnxfsss`, `nartaa/rsna-knee-publication-swa-weights-20261007`,
`heliosli/rsna-orthofoundation-fold0-epoch7-20261004`), one or two with goodpjw2008's reader dataset.

| Notebook (run, UTC) | Public | Inputs public? | Difference vs R7 / V224 (cell diff) |
|---|---|---|---|
| **jingshuifu/rsna-knee-a224-w30** v1 (10-14 06:57) | **.955** | yes | V224 with R7 .7 / Raptor slot .3 |
| **jingshuifu/rsna-knee-w30** (10-14 08:29) | **.955** | yes | R7 with .7 / .3 |
| jingshuifu/rsna-knee-a224-blend v1 (10-14 06:40) | .956 | yes | V224 verbatim (0 diff lines) |
| jingshuifu/rsna-knee-flip-tta (10-13 16:34) | .956 | yes | R7 + flip TTA (known 10-14) |
| jingshuifu/rsna-knee-a224-flip v1 (10-14 06:57) | not scored yet | yes | V224 + flip (11 diff lines) = the same one change as our V224-FLIP (57174021, read ~18:00) |
| jingshuifu/rsna-knee-a224-topview / -multiview | .956 (v1, run 10-13 10:42, 7 min commit) | yes (scored run) | current code adds dreaddevelopment's older Raptor arms v5 / v8 / v10 (w .7/.2/.1) to the Raptor slot from raptor-knee-maxspan / native384 / native384dense; the scored v1 run did not mount those datasets, so the .956 is not a read of that change. Old arms are the .937-.943 generation: GUESS no gain |
| jingshuifu/rsna-knee-raptor-topview, -raptor-multiview, -a224-w50 | .954 | yes | compact-v1 S6 base variants, below R7 |
| **mkhamzanov/rsna-b1-ctrl** (10-13 01:19) | .956 | yes | R7 + JPEG decoder wheels (pylibjpeg, libjpeg, openjpeg from goodpjw's reader dataset) put on PYTHONPATH for every stage, plus a transfer-syntax census of the test. Its header CLAIMS the pinned image (37c64f7) has no pylibjpeg / gdcm, so Raptor, c224 and the reader silently drop JPEG Lossless / JPEG 2000 slices. See lever 6 |
| picond/rsna-knee-r7-pinned-exact50-control (10-14 03:47) | .956 | run also mounts a private source | R7 rebuilt as a 99-line driver, BLEND swa .5 / r7 .5. Control only |
| evgendvorkin/rsna-versia-7 v22 (406 votes), le4ves, degnonguidi v10, sujanmajhisuzan, medvax, lavinwins, tahmidtawsif, yuvrajlamba01, kupedata | .956 | yes | R7 re-wraps (0-19 diff lines, or re-ordered cells with the same 3 inputs). medvax says its target .958 "has not been achieved" |
| karttikjangid05, nursrijan, mayankraaz | .956 | yes | V224 verbatim (0 diff lines) |
| goodpjw2008/...-mil-lb-0-954 (v15 current, best v10) | .954 | yes | reader on compact-v1, unchanged verdict |
| rudrpratapsingh/four-public-models-one-rank-blend v2 | not scored | yes | V224 + compact-v1 S6 |

heliosli / nartaa: **no new dataset, no new dataset version, no new RSNA notebook version** (r7cnxfsss v1 10-11 19:59 UTC,
nartaa SWA v1 10-07, heliosli RSNA notebook last run 10-11 20:01). A byte-identical mirror appeared
(varianceofx/rsna-knee-compact-r7cnxfsss, 823,095,598 bytes, 10-14 03:30 UTC).

New public label / data assets (none image-informed, none a lever):
- gperdrizet/rsna-knee-abnormality-pseudolabeled-training-data (536 downloads): Qwen2.5-7B + Phi4 report labels, majority vote,
  gated AND rule, binary, ~.83 accuracy on gold (their table).
- hukumjailer/rsna-knee-consensus-labels-clean (69 downloads): "5-source consensus" soft labels from reports, no description;
  used by hukumjailer/rsna-knee-consensus-raptor-training (unscored) to retrain Raptor.
- siamrahman29/rsna-knee-w4-labels, genesis96839/rsna-knee-train-ab-labels (today, 0 downloads); lonnieqin Raptor tensor caches
  S0-S3 (~20 GB, a training-speed asset); prvsiyan/rsna-knee-synthetic-dicom-fixtures (synthetic DICOM fixtures + golden arrays).
- Kirderf's Kaggle benchmark (747658): report-only label extraction caps at gold58 macro AUC .86-.89 for every frontier model
  (Opus 5.5 / GPT-5.5 / Sonnet 5.5 .89, Gemini 3.8 Flash .88). Vision-from-image-mosaic benchmark started 10-13, no scores yet;
  a local qwen 27b w4a16 vision run on 10 gold studies gave .606.

Staff notebook ryanholbrook/rsna-knee-abnormalities-efficiency-lb (75 versions, updated 10-13 21:27 UTC) maintains an
efficiency leaderboard; not our target.

## 2. Discussion (threads with activity since 10-13 08:00 UTC)

### OAI eligibility (748276, 733965, 748480, 747455, new 749145): still NO host answer
- **749145 (Mobassir, new, 10-14 ~03:00 UTC, 9 votes):** formal request for a pinned ruling. Lays out NIH NOT-OD-25-083
  (countries of concern incl. China, Russia, Venezuela, which are competition-eligible), China users failing Login.gov (LG22),
  one NDA help-desk reply saying OAI is open to all. Six yes/no questions: OAI allowed; public OAI-trained checkpoints allowed for
  teams that never touched OAI; removal from the whole LB or only prize verification; medal loss by tier; **whether a ban
  would also cover "labels or pseudo-labels produced by OAI-trained models, and models fine-tuned from OAI-trained weights"**;
  pin before 10-22. Replies: madmax0404 (rk27) "they never gave us an answer. I don't think they will now"; Satwik (rk74) notes
  hosts answered survey posts but none of the external-data questions; Wildata: assume legal until told otherwise.
- 747455: Arihant Jain (10-13 16:09 UTC) pinged @macruzbar and @javaduke95 for a ruling on the nartaa SWA weights (medal and
  prize eligibility) before the notebook lock. No reply. 748276: still 0 comments. 748480: one user reply only (10-12).
  733965: nothing after Mahatva Goell's 10-13 05:11 UTC question on downstream use of public OAI-trained weights (sec 2.5 / 2.8).
- Host activity since 10-13: only Po-Hao Chen, on the AI-use survey (749038, 10-13 21:20 UTC) and a joke reply in 736301.
  MEASURED: zero external-data answers in 8 days of open questions.

### Teacher-student and labels from other models
- **748925 (Marco Colussi, 10-13 15:53 UTC, 0 replies):** asks if soft labels for the training studies made by a model whose
  weights and data are NOT public (run offline, not in the final notebook) are allowed under 2.6, whether publishing the label
  file as a public dataset satisfies "publicly available", and whether winner verification needs the labeler itself.
  Unanswered. Our R7-OOF teacher is the easier case: the teacher is a public Apache-2.0 notebook on public datasets, so it is an
  ordinary public external model; verification would ship the label table + the rsna-r7-oof notebook (GUESS on how hosts read
  it). If OAI is ruled out with derivatives (749145 Q5), the T_b table is OAI-free by PREREG; the R7 ConvNeXt teacher carries
  fastMRI auxiliary training (748480 asks about fastMRI too).
- **748295 azamat1ch (rk7, 10-13 15:24 UTC)**, asked whether their labels were made by AI or by hand: **"AI, not by hand. And
  don't rely on reports alone."** Read with the Kirderf ceiling above (report-only labelers stop at ~.89 gold58) and hengck23's
  745861 note (an .89 LLM ground truth gives ~.930 public): the top teams' edge is image-informed AI labels. That is the same
  direction as the R7-teacher student (R7 OOF = image-informed soft targets). CLAIMED, no numbers; nobody posted evidence on
  distilling a public model specifically.
- 747658 hengck23: points to Hulu-Med-14B (medical MLLM that reads 3D nii volumes) and SGMRI-VQA; his local qwen3.8-27b on report
  text reaches AUC .87. Not a lever inside 8 days.

### Leaderboard mechanics and the lock
- **748181 Raagav Anand (10-13 11:19 UTC):** the one-week lock blocks publishing NEW notebooks only; new VERSIONS of already-public
  notebooks are not locked (he saw fresh versions of public notebooks 2.5 h before the end of Kaggriculture). CLAIMED, consistent
  with Kaggle behaviour. Dataset versions are not locked either. So a heliosli / nartaa / varshitha / jingshuifu update can still
  land up to 10-22; ForcewithMe asked staff to extend the lock to a month (no reply).
- 748181 heliosli (10-13 11:56 UTC, -5 votes), defending public sharing: tells a rival that domain background helped him "reach
  0.962" and that "even if I open-sourced a solution scoring 0.961, that would not erase your advantage". Hypothetical wording, no
  artifact; worth keeping the watcher armed for it.
- 748011 varshitha (rk374, 10-13 11:25 UTC): simulated single-submission public SD ~.003 at .955 on ~390 studies; inside a
  displayed block, rank follows the unrounded score (her 437 -> 320 and 233 -> 156). Matches our 10-13 finding.
- 748111: the .960 single-sub rumour (Rednote, "basically a single model"); no link posted.

### Nothing actionable
749196 (Rishavendra Sharma, -5 votes: LLM essay on the gold58-to-LB gap, no data), 745861 (orthopred CAD link), 746792 (OAI
download issues from China; varshitha's credit note), 736301 / 749038 (survey), 748575 / 745984 (queue and exception help).
"Hottest by votes" list: nothing new outside the threads above (top threads updated since 10-13 = 748181, 745861, 746792, 748111).

## 3. The 5 most actionable facts

1. **No public notebook or dataset beats .956 and heliosli / nartaa have not moved since 10-11 (watch run 4, MEASURED), but the
   lock only stops new notebooks: new versions of existing public notebooks and new dataset versions can still land until 10-22
   (748181, Raagav Anand, CLAIMED).** Keep `watch.py --loop-hours` running through 10-22 (it re-reads key authors and run-time
   changes and diffs dataset versions); heliosli talks about a hypothetical .961 release (748181).
2. **Free third-party read: R7 .7 / Raptor slot .3 = .955 on both the R7 and the V224 base (jingshuifu w30 and a224-w30,
   MEASURED public scores), one display step below .5 / .5.** Do not shift blend weight toward R7; keep .5 / .5 (or test toward
   the Raptor slot). a224-flip (= our V224-FLIP change) is still unscored; our own 57174021 read is the same test.
3. **OAI: still no host answer anywhere (748276 0 replies, 747455 / 733965 pings ignored); new formal request 749145 asks whether
   a ban would also cover pseudo-labels from OAI-trained models and fine-tunes, and whether removal is LB-wide or prize-only.
   Experienced voices (madmax0404 rk27, Satwik rk74) expect no ruling before 10-22.** Plan the 2 final picks under "no ruling",
   keep the T_b table strictly OAI-free (PREREG), and treat an OAI-light fallback (R7 ConvNeXt + OAI-free fleet + student) as
   the only hedge if a late ban comes (GUESS).
4. **Top-team hint on labels: azamat1ch (rk7, 748295): "AI, not by hand. And don't rely on reports alone."** With report-only
   LLM labels capped at ~.89 gold58 (Kirderf benchmark, 747658) and hengck23's .89 gold58 -> ~.930 public (745861), the gap to
   the .96x teams is image-informed AI labels, which is the R7-teacher direction. On whether a teacher's labels are allowed:
   748925 (non-public labeler) is unanswered; ours uses a public Apache-2.0 model on public datasets, the lower-risk case.
5. **The .957 band more than doubled (43 -> 95 teams) in a day, all from private work (24 teams rose into .957+ between 05:27 and
   10:24 UTC; none has a recent public notebook, MEASURED). Silver (r298) now sits ~90 ranks into the .956 band; we are r486.**
   A single own-model differentiator that moves the unrounded score is what separates the copy pile; V224 copies alone will
   keep sinking as later copies tie above them only by unrounded noise.

Extra (low cost, unverified): mkhamzanov/rsna-b1-ctrl CLAIMS the pinned image cannot decode JPEG Lossless / JPEG 2000 and that
Raptor / c224 drop such slices silently; the data page says the hidden test mixes uncompressed, JPEG Lossless and JPEG 2000,
while every train file sampled (our EDA, 2 public EDAs) is uncompressed. Our `_fastread.py` falls back to pydicom for any
non-LE syntax. A 10-minute check of whether our V224-F stages decode a compressed fixture (prvsiyan/rsna-knee-synthetic-dicom-fixtures)
in the pinned image would tell us if private has a silent-drop risk that the public score cannot show.
