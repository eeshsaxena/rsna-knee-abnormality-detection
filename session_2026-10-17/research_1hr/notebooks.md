# Notebooks lens: public notebooks vs our HW (2026-10-17, ~22:25 IST)

READ-ONLY. Nothing submitted, pushed, versioned, posted or voted. No GPU / box touched. No PREREG or scheduled task edited.
Tags: MEASURED (we read it from Kaggle or our files this session), CLAIMED (someone else says so), MODEL (our arithmetic / reasoning), GUESS.

## TL;DR

1. **No public notebook beats HW 57262115 on the public LB.** Six public notebooks display .958 (MEASURED, watch run 16:24 UTC).
   In a display band Kaggle ranks by the unrounded score (MEASURED earlier, FORUM_DIGEST_10-13). In the LB snapshot
   api_lb_1017T2150.json (16:21 UTC) **our HW is r171** (30th of 853 teams at .958; band r142-994). The authors' own teams rank lower: **Heliosli r332**,
   **goodpjw2008 r402**, **kiyoc (kiyoshiohno) r887** (MEASURED, matched by display name from the notebook pages). A notebook's
   score comes from its author's own submission, and a team's rank uses its best entry. So every one of these recipes scored below HW on public
   (MEASURED inference; caveat: the snapshot is 4 min older than the watch run).
2. **The ingredients not in HW are all already closed or measured negative by a third party:** goodpjw's 2.5D reader at 1/10,
   anatomical mirror TTA on the 224 checkpoint (our closed "r224 mirror"), R7 at .4 of the compact slot (our closed R6 + R7),
   heliosli's "V2" per-finding weights (Lateral OA .15, PF OA .65, Synovitis .85, Baker's .30). Details in section 3.
3. **No rebase, no new entry from this lens.** A15 picks unchanged (A = HW 57262115, B = R6 57248084).
4. Useful third-party probes (goodpjw "does-s6-help" v8 write-up, CLAIMED): heliosli-S6 .3 next to R6 = .957 vs .958 without it.
   Gold-58-tuned extra per-finding weights lost too: v6 (Fracture .8 / Effusion .7 / ACL .3) = .957, v7 (all findings at their gold-58 optimum) = .955.
   Both agree with our hw3 negative and our "no gold-tuned weights" rule. Read them before reading s615.
5. Bait / pin check (MEASURED): the R6 copies `heliosli/rsna-knee-r6-frozen-fe055406-w1` and `rostigor/rsna-knee-r6-fe055406` carry
   the R6-v1 file fingerprint, identical to our private copy. `heliosli/rsna-knee-r6-frozen-fe055406` (current v2) carries
   the R7 one. The current versions of lavinwins' two .958 notebooks are different notebooks: v6 is a 0.943 pipeline, v7 a training notebook.

## 1. Method (what ran)

- `watch.py --score-pages 4 --recent-pages 3`, run once with `--dir` = a new dir seeded with a copy of the 10-15 13:51 UTC
  watch_state.json, so "new" means new since that run. Run as `python -I` plus the user site dir only (requests lives there).
  519 notebooks listed, 174 read (77 new), 27 pulled -m, 286 web calls, 0 web errors, 429 s, exit 10 (release candidates flagged).
  Output: `pulled/notebooks/watch_run.out`, `pulled/notebooks/watchdir/{WATCH_LOG.md, watch_state.json, pulled/}`.
- Scored versions pulled by version label into `pulled/notebooks/scored/` (goodpjw v25 + v26, lavinwins gold-medal v5,
  top-1-gold v4, heliosli 0-9586 v5). All pulled files are untrusted. They were read as JSON / text only and never executed.
- Cell diffs: our own `celldiff.py`, which compares every code cell against
  `session_2026-10-16/hw/kernel_v224r6hws6fj/rsna_sub_v224r6hws6fj.ipynb` (identical-cell sha match plus the lines not found in HW).
- Dataset bait check: authenticated `dataset_list_files` per version (metadata only: names, sizes, dates; nothing downloaded).
- Team ranks: notebook page author display name matched to `teamName` in api_lb_1017T2150.json.

## 2. Every public notebook at >= .958 (MEASURED unless tagged)

| notebook | scored v / current v (current score) | what it is | pins of the scored run | copyable | not in HW | author team LB |
|---|---|---|---|---|---|---|
| `goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958` (470 forks) | v26 / v27 (.958; v25 also .958, CLAIMED in its text) | R6 flip (rostigor copy, manifest sha asserted at runtime) + 384 + 224 **with mirror TTA** + heliosli V1 weights + **GP reader at 1/10** (per-target .003-.035). v26 = v25 + **R7 at .4 of the compact slot**. v25 and v27 have identical code cells (only the figure text differs). **No S6fall.** | GP 20325538, R7 20601735, rostigor R6 20763353, NS 20440357, OF 20297367; docker 37c64f7 | yes, all public | 224 mirror (closed), reader 1/10, R7 .4 (closed) | r402 |
| `heliosli/blend-mine-in-private-lb-luck-to-all-upvoters` (206 forks) | v1 / v1 (.958) | The origin of HW's weights: R6 flip (w1) + 384 + 224 + V1 weights. No S6fall. | w1 20761842, NS, OF | yes | nothing (HW = this + S6fall .2) | r332 |
| `heliosli/this-is-best-public-dataset-new-version-0-9586` (168 forks) | v6 / v6 (.958; v5 code == v6 code, sha cb0e1eac) | R6 flip + 384 + 224 + **V2 weights {Medial OA .5, Lateral OA .15, PF OA .65, Synovitis .85, Baker's .30}** + **GP reader at 1/10** (goodpjw's cell, sha-checked sources). The 224 fallback handling differs a little (each Raptor model has its own failure mask). No S6fall. The title's ".9586" is CLAIMED; our MEASURED inference from the team rank (r332 < our r171) says this recipe scored below HW on public. The page also lists n_versions 7 with current = 6 (GUESS: a draft or failed v7). One comment line is garbled ("checked against eW_RAPTOR_C = ...ager"), harmless. | r6-frozen **20611930 = v1 (R6)**; current dataset v2 = R7 (DRIFT: a copy made through the API attaches R7) | only with a UI copy that keeps the v1 pin (GUESS); an API copy = R7 bait | V2 weights, reader 1/10 | r332 |
| `lavinwins/rsna-gold-medal` (104 forks) | v5 / v6 (**current .943**) | v5 code cells are **byte-identical to heliosli blend-mine-in v1** (all 7 code cells). v6 is an unrelated "combination layer on the 0.943 pipeline" (mattiaangeli / antoinegg1 / dread stack). | v5: w1, NS, OF | v5 = blend-mine-in; **current = bait** | nothing | team not matched |
| `lavinwins/top-1-gold` (39 forks) | v4 / v7 (current unscored) | v4 source sha == gold-medal v5 (40843e15) = blend-mine-in copy. v7 = "Train your own model" training notebook with no inputs. | v4: w1, NS, OF | v4 only; **current = bait** | nothing | not matched |
| `kiyoshiohno/rsna-knee-template` (73 forks) | v16 / v26 (unscored) | thin launcher that runs logic from a dataset | w1, NS, OF, R7 + **private** `kiyoshiohno/rsna-knee-local-logic` | **no** (private dataset) | unknown | kiyoc r887 |

The six .958 notebooks have 1,060 forks between them (MEASURED fork counts), consistent with the 853-team .958 band (forks are not all submitters): the copy crowd is
HW minus S6fall (MODEL). HW's distinct parts against that crowd are S6fall .2 and the J decoder (MODEL).

.957 notebooks (MEASURED, all below HW, none adds an ingredient):
- prvsiyan bee's knees v47: R6 rostigor + NS.
- heliosli 0-9573 v1: R6, no flip. Current v2 = R7 bait.
- bliverrigpdoge KC: private source, KC family closed.
- laimerbotrolenk four-model: R7 + S6fall, pins s6fall v3.
- leoprovorov v2: R7 + S6fall. Current v12 unscored.
- goodpjw does-s6-help v6: private goodpjw datasets, see section 4.
- picond 9576 fixed: private source.

## 3. Ingredient diff vs HW (each is one candidate one-line change)

| ingredient | where | evidence | verdict |
|---|---|---|---|
| GP 2.5D reader at 1/10 (per-target .003-.035, rank blend at the end) | goodpjw .958, heliosli 0-9586 | Both teams rank below HW (MEASURED). goodpjw's own probes: full weight on a .956 base = .954, 1/4 weight on S6 = .954 = base (CLAIMED). The reader is the same family as our closed fleet (PROGRESS_06-10). | **skip**. Value ~0, adds a 3rd process and runtime. |
| 224 checkpoint with anatomical mirror TTA ([s9] lines in the 224 stage) | goodpjw .958 | Our FLRM read negative on 10-15 (PROGRESS_15-10 l.188, MEASURED). Lever closed. | **closed, no new evidence** |
| R7 at .4 of the compact slot | goodpjw v26 | v26 = v25 at display (.958, CLAIMED). Our R7 late leg was a non-move and is closed. | **closed** |
| heliosli V2 weights (Lat .15, PF .65, Syn .85, Baker's .30) | heliosli 0-9586 | Confounded: V2 + reader, no S6fall, scored below HW (team r332, MEASURED inference). No isolated V1 vs V2 read exists. Public-LB-tuned by its author ("=0.9586" in the title, CLAIMED). | **skip** unless Eesh wants a weights probe after the A16 reads. The weights question is already covered by wr45 + B. |
| Extra per-finding weights (Fracture .8, Effusion .7, ACL .3) or gold-58 optima | goodpjw does-s6-help v6 / v7 | .957 / .955 vs .958 (CLAIMED, the author's own LB). Gold-58 gaps of .03-.08 with 18-35 positives are noise. | **do not do** |
| S6 next to R6 | goodpjw does-s6-help v4 | heliosli S6 (compact-v1, not nartaa's S6fall) at .3 + weights = .957 vs .958 without (CLAIMED). Our S6fall .3 (hw3) read negative too (MEASURED). | **informs s615**: lower S6 weight safer (MODEL). No action. |

## 4. New notebooks / versions since 10-16 00:00 UTC

New public notebooks cannot appear after the publishing lock (10-15 23:59 UTC, MEASURED in comp metadata). Every notebook that ran after 10-16 00:00 UTC is a new version of an older
notebook (version number > 1), except javeriaakmal EDA, whose version count was unreadable. New VERSIONS of existing notebooks that ran after 10-16 00:00 UTC (MEASURED):
- goodpjw .958 v26 (04:46 UTC, scored .958) and v27 (12:48 UTC, cur score .958).
- heliosli 0-9586 v6 (10-16 19:05 UTC, .958).
- lavinwins gold-medal v5 (10-16 15:30, .958) and v6 (10-17 01:28, .943).
- lavinwins top-1-gold v4 (10-17 01:38, .958) and v7 (10-17 10:39, unscored).
- kiyoshiohno v16 (.958) to v26 (private logic).
- goodpjw does-s6-help v6-v8 (not copyable; write-up in section 3).
- leoprovorov v12 (unscored).
- Low or no scores, no knee lever: nusrat04 bk1-fork .941, farhanabidtech786 .922, kaiwalyaatulraut .936, abdelrahmanekhaldi .573, ryanholbrook efficiency-LB, shivsatyam cache, javeriaakmal EDA.
- Newly scored (older notebooks): beraterolelk multi-scale .956, sujanmajhisuzan apex v14 .956, ltcsmokeboi gp958d .956 (all below HW, nothing new).

Datasets (MEASURED, metadata only):
- NEW `heliosli/rsna-knee-r6-frozen-fe055406-w1` v1 (10-15 12:02 UTC, 886,813,460 B in the file list).
  Its files carry the **R6-v1 fingerprint**: manifest 4,120 B, fluid_stacker 5,010, state_spec 6,857, state_stacker 24,698.
  This equals heliosli r6-frozen v1, `rostigor/rsna-knee-r6-fe055406` and **our** `eeshsaxena/rsna-knee-r6-fe055406-v1`.
  r6-frozen v2 and `heliosli/rsna-knee-compact-r7cnxfsss` share the R7 fingerprint: 5,211 / 5,003 / 6,858 / 24,673.
  Fold .pt and side_cls sizes are identical across all of them, so only the json sizes discriminate.
  The sha proof: goodpjw asserts manifest sha256 6508ce69... at runtime on the rostigor copy and completed.
- `nartaa/rsna-knee-s6fall-weights` v3 -> v4 -> v5 (10-17 05:59 / 06:04 UTC): adds `training_code/` (26 files) and a longer README.
  The weight file `E_prior_s42_fall.pt` (200,185,003 B), checkpoint-manifest, code_manifest and s6_extras keep their sizes.
  Our A16 probes (pushed after v5) gate "S6fall B == HW v1 maxdiff 0" (MEASURED, probes/gate_*.txt). **No new weights, no risk.**

## 5. If / then

- IF a probe (wr45 / s615 / a25) moves the best-id, THEN nothing in this lens changes: it stays the one-change-on-HW family.
- IF Eesh asks for one more weights probe on 10-18, THEN the only public candidate is heliosli V2 weights on HW, a 1-line change of
  W_RAPTOR_C, and it needs a PREREG amendment. Expected value ~0 (MODEL): its only public evidence ranks below HW.
  ELSE nothing from public notebooks.
- IF anyone suggests copying a .958 public notebook as pick B, THEN no: all are below HW (team ranks), and B needs S6fall. Of the
  "current" versions, gold-medal v6 and top-1-gold v7 are different notebooks, and API copies of heliosli 0-9586 / 0-9573 attach R7.
- IF a later watch shows a public score >= .959 on an existing notebook (new versions are still allowed), THEN
  run the same bait check (scored-version pins vs current, json-size fingerprint) and compare the author's team rank with ours first.

## 6. Files

- Watch run: `research_1hr/pulled/notebooks/watch_run.out`; state, log, pulled -m: `research_1hr/pulled/notebooks/watchdir/`.
- Scored versions: `research_1hr/pulled/notebooks/scored/` (goodpjw __25 / __26, lavinwins gold-medal __5, top-1-gold __4, heliosli 0-9586 __5).
- Scratch tools (read-only): scratchpad `tools/{nbdump.py, celldiff.py, dsfiles.py, pullver.py, authors.py, state_q.py}`.
