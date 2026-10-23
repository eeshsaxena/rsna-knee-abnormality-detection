# Runtime headroom lens (think workflow, 10-19 ~17:30 IST). RESEARCH ONLY.

Nothing built, pushed, committed, submitted or started. No GPU / box job. No PREREG / HANDOFF / PROGRESS / scheduled-task edit.
Local CPU only: two small numpy scripts in `think/rt_work/` (snapshot reader + Monte Carlo), < 10 s each.
Tags: MEASURED = read from our logs, receipts, API snapshots or code in this session; CLAIMED = someone else says so
(forum, Kaggle, our older notes); MODEL = arithmetic on stated inputs; GUESS = judgement.

## 0. Bottom line
1. **The c4-class hidden kernel takes about 5h40 to 6h27 (12 runs), and c4 itself about 6h03-6h06.** MEASURED submit-to-COMPLETE
   upper bounds from the API read chain and LB snapshots (sec 1). Latency = queue + kernel + scoring, so runtime is at most that.
2. **Unused vs the 9 h limit: about 2h55 at the median, at least 2h33 in the slowest observed run, and at least 0h47 by design**
   (S6F is SIGKILLed at 8.2 h and R6 has an 8.3 h deadline, so c4 cannot pass about 8.22 h even in a very slow session; MEASURED code,
   cells 3/5/11 of `kernel_v224r6a25c4j`). MC with a 5 % x1.3 slow-session tail (GUESS, as in the 10-15 sims): unused median 2.94 h,
   p05 2.47 h (`rt_work/addon_mc.txt`, MODEL).
3. **Margin we must keep (MODEL):** any NEW compute must be guarded (start guard + process-group kill at <= 8.3-8.4 h + exact c4
   fallback), which leaves >= 0.6 h before 9 h for the kill, final blend, output write and Kaggle overhead. Unguarded additions are safe
   only up to ~0.25-0.5 h (P(> 9 h) 0 / 1e-4); at 0.75 h it is ~1.5 %, at 1.0 h ~3 %, at 2.5 h ~10 % (all driven by the slow tail).
   Inline compute before S6F must also not push S6F past its 5.75 h start guard (sec 3).
4. **What could fill it, and the honest verdict:** the only filler with an in-distribution mechanism is **R6 shifted-crop TTA**
   (R6 was trained on random 352 crops of 384 and serves the centre crop; MEASURED trainer + worker code). It costs **+95-115 min**
   per extra view across 4 folds (MODEL from the commit's GPU s/window), which collides with the S6F 5.75 h start guard
   (P(S6F degraded) .47 at G 5.75; .07 if G is raised to 6.25-6.6). Everything that fits cleanly in the window (Raptor extra views,
   our Raptor seeds, KC, students, GP reader) is **LB-negative or ~0** on this comp.
5. **More compute has not meant more score here:** of ~11 compute-adding entries we scored, 2 moved the best-id (R7 flip TTA, S6F);
   the rest (K14 fleet x2, r224 mirror, student S, S6F + S, KC1, COMBO R6+KC1, R7 late leg, HFR7-RS 2 Raptor seeds) did not (MEASURED,
   PROGRESS_14..17, timing.md). Positives were (a) anatomical flip on the main CNN leg and (b) a strong EXTERNAL family at modest weight.
6. **Default recommendation (GUESS): do not spend 10-20/10-21 slots on runtime fillers.** If Eesh wants one label-agnostic
   compute probe (private-friendly: it is not tuned on public labels), the least-bad is R6 +1 shifted-crop view, built so the first
   ~4 h are byte-identical to c4 and S6F's guard is raised, after a real T4 timing probe and a box OOF screen (sec 7-8). Expected
   +0.5 to +3e-4 public (GUESS), P(best-id move) ~.45-.55.

## 1. MEASURED hidden latency of the c4 compute class (R6 flip + Raptor r384 + r224 + S6fall)
All rows below have the same compute; they differ only in weights (one line in cells 8 / 12). Submit times: tick / submit logs and
`lastSubmissionDate`; completion bounds: `session_2026-10-14/api_chain.log` (polls every 180 s, snapshot 90 s after detection,
`api_chain.sh`) and API LB snapshots (`session_2026-10-14/research960/lbprobe/api_lb_*.json`, team `submissionCount` counts COMPLETED
entries only). Extracted with `rt_work/snap_counts.py` -> `rt_work/snap_counts.txt`.

| entry | ref | submit UTC | latency bounds (h:mm) | how |
|---|---|---|---|---|
| s615 (S6 .15) | 57313454 | 10-17 13:52:20 | <= 5:42 (about 5:38-5:42 if the chain ran continuously) | chain detect 19:34 UTC |
| a25 (A224 .25) | 57316395 | 10-17 15:28:50 | <= 5:43 (about 5:39-5:43) | chain detect 21:11 UTC |
| c3max | 57382515 | 10-19 04:44:05 | <= 5:57 | snapshot 10:41:29 UTC, count 97 |
| c4 (best) | 57343435 | 10-18 06:44:32 | <= 6:06 (about 6:03-6:06) | chain detect 12:50:54 UTC |
| fr65 | 57343462 | 10-18 06:45:15 | 6:05-6:13 | not done at c4's detection, done at 12:58:47 |
| R6 | 57248084 | 10-16 00:07:02 | 6:03-6:15 | timing.md |
| c2max / fr100 / c2 | 57382007/11/15 | 10-19 04:26:10-20 | <= 6:15 | snapshot 10:41:29 UTC |
| wr45 (W_RAPTOR .45) | 57310758 | 10-17 12:18:55 | <= 6:23 (about 6:20-6:23) | chain detect 18:42 UTC |
| pf10 | 57382016 | 10-19 04:26:22 | 6:15-6:27 | pending at 10:41:29, count 98 at 10:53:20 UTC |
| HW | 57262115 | 10-16 08:43:56 | <= 6:33 (loose) | timing.md |
| S6W, hw3, a15, c4a15, pf50 | | | loose (<= 6:09 / <= 6:03 / <= 7:46-8:01) | snapshots too far apart |

- Range of the tight bounds: **5h40 to 6h27**, median ~6h05 (MEASURED). Max / median ~1.07. Two entries submitted 4 s apart (pf10
  vs c2) differ by >= 0-12 min; c3max (submitted 18 min after pf10) finished >= 18 min before it, so session-to-session spread is
  about +-6 % (MEASURED). No slow tail (x1.3) has been seen in these 12 runs; older evidence of spread: HF20 / HF30 4 s apart finished
  0.17-0.51 h apart, FN11 x1.14 (CLAIMED, RUNTIME_FFS_1015.md).
- Hidden study count: ~1,300 (our model), 1,322 (CLAIMED, forum 744230), 1,284 (CLAIMED, nartaa prereg in sim_s6f_runtime.py header).
  public 30 % (`leaderboardPercentage 30`, MEASURED comp.json via RESEARCH_1017.md). ~661 studies per T4.

## 2. Per-stage hidden runtime of c4 (MODEL, anchored on MEASURED commit receipts and latency deltas)
Commit numbers: `session_2026-10-18/a18/out_v224r6a25c4j_v1/` (log, submission_receipt.json, worker0/1_log.txt, s6f_receipt.json);
3 test studies, so fixed costs dominate there. Hidden logs are never returned, so the split below is MODEL.

| stage (order) | commit, 3 studies (MEASURED) | hidden, ~1,300 studies (MODEL) | anchor |
|---|---|---|---|
| DEC cell + Raptor process start | DEC 3.6 s; process start ~28 s | ~0.02 h | log t=12-40 s |
| Raptor r384, plain + anatomical mirror, 2 GPUs | stage 55 s (GPU 22 s incl. warmup) | **0.30-0.42 h** | PRE 0.42 h incl. wheels (sim calibration, CLAIMED sim_s6f_runtime.py); nartaa "26.1 min incl. queue" (CLAIMED 746792) |
| R6 setup (wheels, assets, compile check) | ASSETS_READY 173 s; compile check 209 s | ~0.08-0.1 h | worker receipt `check_seconds 208.8` |
| **R6 leg**: 4 ConvNeXt-S folds x (plain + flip), 1 worker per T4 | per study GPU 18.8-20.9 s (502-574 windows), decode 35-65 s per study per loader, 2 loaders | **3.1-3.5 h** (~18-19 s / study / T4, GPU-bound, decode about balanced) | R6 latency 6.03-6.15 minus S6F 2.0 minus Raptor / r224 / setup; commit 4.6-4.9 ms per window-view-fold |
| r224 (plain, 2 GPUs) | stage 36 s | 0.10-0.25 h | sim "r224 + stage 3 0.15-0.30 h" |
| stage 3 blend + canary | < 1 s | ~0 | |
| S6F setup (DEC, env, inputs sha, B probe) | ~40 s | 0.03-0.08 h | s6f_receipt stages |
| **S6F**: nartaa S6fall, 1 model, crop 288, no flip | B 64 s; GPU 10.6 s / study, decode 25.1 s / study / worker | **1.9-2.1 h** (~10-11 s / study / T4) | S6F 57201874 (6.6 h) minus V224-FLIP (4.62 h), MEASURED latencies |
| final .8 / .2 blend | < 1 s | ~0 | |
| **total** | 677 s | **~5.7-6.4 h** | matches sec 1 |

- Every stage already uses both T4s (Raptor 2 devices, R6 2 workers, S6F 2 parts; MEASURED logs). There is no idle second GPU to use
  in parallel; the slack is purely at the END of the run.
- S6F is decode-bound (25 s decode per study per worker with 2 loaders = ~12.5 s vs 10.6 s GPU), R6 roughly balanced (MODEL).

## 3. Unused time and the margin (MODEL, `rt_work/addon_mc.py` -> `addon_mc.txt`)
Inputs: kernel end drawn from the sec 1 bounds minus 0-0.1 h queue / scoring (GUESS); S6F stage U(1.9, 2.1) h; slow-session tail
x1.3 with p .05 (GUESS, same as the 10-15 sims; NOT seen in our 12 runs); add-on noise N(1, .07).

c4 as built: end median 6.06 h, p95 6.53, p99 8.07, max 8.22; P(S6F skipped or cut) .005; P(> 9 h) 0.
**Unused to 9 h: median 2.94 h, p05 2.47 h, guaranteed >= 0.78 h.**

(A) GUARDED stage AFTER S6F (start only if E + 1.15 C <= 8.3 h; kill at 8.3 h; all-or-nothing; fallback = c4's file):
| add-on C at s=1 | P(used) | P(skipped at start) | P(killed -> c4) | P(> 9 h) |
|---|---|---|---|---|
| 0.25 h | .982 | .014 | .004 | 0 |
| 0.5 h | .965 | .032 | .004 | 0 |
| 1.0 h | .950 | .050 | 0 | 0 |
| 1.5 h | .950 | .050 | 0 | 0 |
| 1.75 h | .798 | .201 | .001 | 0 |
| 2.0 h | .394 | .605 | .001 | 0 |
-> A guarded post-S6F stage of up to ~1.5 h runs in every non-tail session. A skip silently returns c4 (no canary): the LB read then
   equals c4 and is confounded with "neutral" (P ~.05).

(A0) UNGUARDED stage after S6F: P(kernel > 9 h) = 0 at 0.25 h, 1e-4 at 0.5 h, .015 at 0.75 h, .032 at 1.0 h, .049 at 1.5-2.0 h,
.10 at 2.5 h. Without the tail (only the observed spread) up to ~2.4 h would fit; the guard is what makes the tail harmless.

(B) INLINE compute X BEFORE S6F (e.g. an extra R6 view inside the R6 worker); "degraded" = S6F skipped (start > G) or cut at 8.2 h
(the canary fires; the run still completes, display visibly lower):
| X | G 5.75 (as built) | G 6.25 | G 6.6 |
|---|---|---|---|
| 0.25 h | .022 | .022 | .022 |
| 0.5 h | .037 | .037 | .037 |
| 0.85-1.0 h | .050 | .050 | .050 |
| 1.7 h | **.474** | .068 | .067 |
| 2.0 h | .795 | .294 | .287 |
P(> 9 h) = 0 in every row (existing kills). End median for X 1.7 h with G 6.6: 7.72 h.

## 4. What could fill it (2xT4, ~1,300 studies; minutes are MODEL unless tagged)
| filler | extra hidden min | fits? | evidence on THIS comp | expected gain | verdict |
|---|---|---|---|---|---|
| **R6 +1 shifted-crop view** (e.g. offset (0,0) / (32,32) or +-16 px of the 384 render), 4 folds, inside the worker (decode shared: worker cuts `off = (size - crop) // 2`, MEASURED compact_infer l.755) | **+95-115** | only if S6F G is raised to >= 6.25 h (else P(degraded) .47) | in-distribution: trainer random-crops 352 of 384 (MEASURED compact_train `augment`, trainer md5 8622009e == manifest); flip TTA on the main leg was positive once (V224 -> V224-FLIP moved the best-id, r486 -> r267 MEASURED snapshots); shift views are far more correlated than a mirror (MODEL) | +0.5 to +3e-4 final (GUESS) | the only plausible one; needs a worker patch + guard change (sec 8) |
| R6 +1 view on 2 of 4 folds | +48-58 | yes at G 5.75 (P(degraded) ~.05, tail only) | as above, half the variance reduction | ~half of the above | cheaper fallback variant |
| R6 +2 views (shift + its flip) | +190-230 | no (with S6F) | as above | | NO |
| R6 multi-scale (size 416 / crop 384) | +115-130 plus a second render path | no | OOD: `--aug crop` has no scale jitter (MEASURED trainer arg help) | <= 0 (MODEL) | NO |
| R6 second flip axis (vertical / AP) | +95-115 | as the shift view | OOD: never trained (only laterality mirror exists) | < 0 (MODEL) | NO |
| R6 extra view as a separate post-S6F stage | >= 200 (decode-bound ~18 s/study) | no | | | NO |
| Raptor r384 +1 view pair (full-field 384 + mirror) inside the Raptor process | +5-12 (GPU only; ~11 TFLOP / study for 2 views at 320, MODEL from RAPTOR_PORT_PLAN 29.8 GMAC/window) | yes, but delays S6F start (P(degraded) .005 -> ~.02) unless placed after S6F | no positive evidence; nartaa: crops 304 / 288 -.001 / -.002, width flip -.001, 384 full = 320 crop alone (CLAIMED 746792) | 0 to +1e-4 (GUESS) | not worth a slot |
| r224 mirror | +5-10 | yes | FLRM 57200578 non-move (MEASURED) | <= 0 | closed |
| extra Raptor arms (our RSNA-only seeds s43-s46) | +20-45 per arm as a separate stage (HFR7-RS vs HFR7-RR + fleet: +0.7-1.5 h for 2 arms, MEASURED latency deltas) | yes (guarded) | HFR7-RS .955 vs HFR7-RR .956 (MEASURED) | < 0 | closed |
| R7 (heliosli second bundle) | plain +140-160, flip +230 | no | R6 + R7 average negative; R7 late leg negative (MEASURED); goodpjw R7 .4 = same display (CLAIMED) | <= 0 | closed |
| heliosli S6 compact-v1 (4-fold eager) | ~160 (HV latency 3.18-3.35 h minus Raptor) | no | S6 .3 next to R6 = .957 vs .958 (CLAIMED goodpjw) | < 0 | closed |
| our S6 seeds s43 / s44 | ~120 each | no (next to S6fall) | LB negative (MEASURED) | < 0 | closed |
| K14 fleet 20 | +65-110 | guarded only | FF20 negative (MEASURED) | < 0 | closed |
| KC1 5 members / students S | +15-30 / +16-18 | yes | negative (MEASURED) | < 0 | closed |
| goodpjw 2.5D reader at 1/10 (3 folds) | 20-40 (GUESS) | yes (guarded) | ~0 at display, its teams rank below ours (CLAIMED / MEASURED, notebooks.md) | ~0 | no |
| S6fall flip TTA | +70-140 | no | worker has no flag; model never saw mirrors (MEASURED levers.md) | <= 0 | closed |
| S6fall side model | ~0 | yes | untested; train / test mismatch risk | <= +2e-4 (GUESS) | no |

## 5. Evidence ledger: does more TTA / more members help HERE?
POSITIVE (best-id moved or claimed gain):
- R7 anatomical flip TTA: V224-exact 57171955 (r486) -> V224-FLIP 57174021 (r267), same .956 display, +1.29 h (MEASURED).
- Raptor anatomical mirror: +.001 alone (.948 -> .949) (CLAIMED nartaa 746792). Already in c4.
- R6 flip forced in every R6 entry; heliosli R6 no flip .957 vs R6 flip recipes .958 (CLAIMED, confounded).
- S6F (an external family, nartaa S6fall at .2): .956 -> .957 (MEASURED 57201874). r224 (external) at .25: a25 positive (MEASURED).
NEGATIVE / NEUTRAL:
- r224 mirror (FLRM) non-move (MEASURED). nartaa: older all-plane width flip .940 -> .939, smaller crops -.001 / -.002 (CLAIMED).
- Window density: 94 -> 24 windows -.006 (CLAIMED nartaa), but R6 / Raptor / S6fall already use every centre (eval_stride 1, 94
  windows; MEASURED levers.md); Dread: more windows "stops paying ... then goes slightly negative" (CLAIMED 737696).
- Our extra members: K14 fleet, KC family, R7-teacher student, S6 seeds, R6 + R7, R7 late leg, HFR7-RS seeds (all MEASURED non-moves).
- forum 744230: 50 passes per study at bf16 on T4 hit an 8 h guard and scored .71 (CLAIMED): heavy TTA is mostly a runtime hazard.
Reading (MODEL): label-agnostic averaging of the SAME strong model over in-distribution views has paid; weaker members of our own
never have. c4 already uses every in-distribution mirror. What is left is low-diversity views (shifted crops) on R6.

## 6. Overrun risk, and why more compute is not automatically more score
- A run over 9 h is killed and the entry ERRORs: no score, the daily slot is gone, it cannot be a pick (CLAIMED Kaggle code-comp
  rules; maxGpuRuntimeMinutes 540 MEASURED comp.json). Hidden logs are not returned, so a slow or partly-skipped run is invisible except
  through the score (forum 744230 / SwapIY 10-19, CLAIMED).
- Picks are not re-run (usesSynchronousReruns True, MEASURED): c4 / R6 carry no further runtime risk. The risk applies only to a NEW
  entry, at its single scoring run.
- c4's design bounds itself at ~8.22 h. Any change must keep that property: guard + kill + exact fallback, all-or-nothing (rank
  blends need the same members on every study), and must not move S6F's start toward 5.75 h without raising G.
- Score: 2 of ~11 compute additions moved the best-id (sec 0.5). The expected value of a filler is set by its evidence, not by the
  minutes it uses. Gains we could plausibly get (+0.5-3e-4) are at or below the public read resolution (paired sd 3-5e-4, density.md),
  so even a real gain reads as a near coin flip (MODEL).
- Private board: TTA changes are not fitted to public labels, so a positive public read should transfer better than a weight probe
  (MODEL). The private delta sd is ~.65x the public one (910 vs 390 studies, MODEL).

## 7. How a GPU could help (proposals only; each is a GPU job and needs Eesh's go)
1. **Box V100 OOF screen of R6 views (~30-50 min GPU).** R6's fold rule is `sha1(StudyInstanceUID)[:8] % 4` over the pool
   (MEASURED compact_train l.2014; trainer md5 8622009e == R6 manifest `trainer_md5`). So every train study on the box (~400 non-gold
   studies) has a known held-out R6 fold model. Score views {centre + flip (current), + shifted crops, + their flips} with the held-out
   fold model, against the LLM report labels used in tie_loss.py / cross_label.py. Resolution: macro SE ~.002-.003 (MODEL), so it
   catches a bug or a large negative, not a +.0005 gain. The R6 bundle and worker are already on the box (~/r6g58, MEASURED HANDOFF).
2. **Kaggle T4 timing probe (~20-40 min of T4 quota).** A private commit that runs the patched R6 worker on 30-60 TRAIN studies
   (train DICOMs are mounted) and logs s / study with and without the extra view. Turns the +95-115 min MODEL into MEASURED. Commits see
   only 3 test studies, so this is the only way to measure hidden-scale cost (Deotte's 1,300-train-study advice, CLAIMED 749623).
3. Friend's 30 GPU-h: could run (2) at full 1,300-study scale (~8 h of T4) to measure the whole kernel end-to-end; too slow to gate a
   10-20 submit, possible for 10-21. Training anything new: no (freeze 10-20 12:00, and every new member of ours read negative).

## 8. If Eesh wants ONE compute probe: least-bad design (NOT built)
- Change vs c4: R6 worker gets `--shift-tta` (one extra crop offset per fold, logits averaged within fold, then across folds as now);
  S6F `G_START_H` 5.75 -> 6.6 (keep DEADLINE_H 8.2, BUDGET_H 3.3 and the kill). Raptor, r224, tables, S6F weight unchanged.
- Runtime (MODEL): end median ~7.7 h, p99 ~8.2 h; P(degraded) ~.07 (tail sessions); P(> 9 h) ~0.
  2-fold variant: +48-58 min, G unchanged, P(degraded) ~.05.
- Gain (GUESS): +0.5 to +3e-4 public, P(true > 0) ~.6-.65, P(best-id move) ~.45-.55; private transfer better than a weight probe.
- Cost / timeline: worker patch + parity test (box CPU) + gate update + commit (queue 0-3 h) ~4-6 h; push 10-20 morning,
  submit after the 05:30 IST UTC-day reset, read ~6.5-8 h later (10-20 evening). Follow-up on 10-21 if positive. Needs Eesh's go +
  a PREREG amendment + a runtime review; do (7.2) first so the minutes are MEASURED.
- Risks: patched third-party worker (compile graphs unchanged since the crop size is the same, MODEL); a guard-degraded run reads as
  a large negative (visible, not ambiguous); a slot spent on a ~coin-flip read.

## 9. Files (this lens)
- `think/rt_work/snap_counts.py` + `snap_counts.txt`: our team's completed-count / best-id per LB snapshot (latency bounds).
- `think/rt_work/addon_mc.py` + `addon_mc.txt`: c4 runtime and add-on Monte Carlo (sec 3).
- Read: timing.md, levers.md, notebooks.md (session_2026-10-17/research_1hr), sim_s6f_runtime.py / .txt (session_2026-10-15/s6f),
  RUNTIME_FFS_1015.md / RUNTIME_V224FFS.md (session_2026-10-15/v224f), RAPTOR_RENTED_RESULTS.md / RAPTOR_PORT_PLAN.md
  (session_2026-10-13), c4 kernel + commit outputs (session_2026-10-18/a18), api_chain.log + api_chain.sh (session_2026-10-14),
  forum threads 744230, 746792, 737696, 735304 (think/forum_raw/flat), R6 manifest (session_2026-10-15/r6v1/raw/manifest.json).
