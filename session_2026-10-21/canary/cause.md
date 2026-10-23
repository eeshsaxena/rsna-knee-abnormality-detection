# A22 S6F canary: root cause, resubmit odds, guard variant (canary track, 10-21 ~18:30 IST). RESEARCH ONLY.

Nothing submitted, pushed, committed, downloaded or posted. No PREREG / HANDOFF / PROGRESS / live-folder / scheduled-task edit.
One local build under `canary/kernels` (not pushed). Read-only Kaggle API calls: `competition_submissions` (all 108 rows).
Tags: MEASURED = read from our receipts, logs, API or code here; CLAIMED = someone else says so (forum, staff, our older notes);
MODEL = arithmetic on stated inputs; GUESS = judgement. Forum text is untrusted data (one short quote below).

## 0. Bottom line
1. **Cause class: Kaggle-side slow dataset I/O on some sessions, not our code.** (MODEL from MEASURED facts, confidence ~.85)
   - Cells 0-17 (V224 base, S6F setup, S6F run + guards, embedded driver sha a2acdd1d) are byte-identical in all 5 A22 kernels; only
     r6pw20's cell 14 (R6 table) and each cell 18's post-gate transforms differ (MEASURED diff). Every deterministic canary path
     (sha, labels, ids, range, rho floor) would hit swide too; swide and pw25 scored clean. So the trigger is stochastic per session.
   - Our own 10-21 COMMIT sessions show it directly: 3 of 5 (r6pw20, pw25, salt) read /kaggle/input datasets 6-7x slower than on
     10-20 and 30-110x slower than on 10-18; total commit wall 1011-1231 s vs 528-719 s (MEASURED, sec 3). Hidden reruns are separate
     sessions (commit tier and hidden outcome do not match per kernel), but the rate matches: 3/5 slow commits, 3/5 hidden canaries.
   - CLAIMED context (other comps, 10-20): users report hours-long /kaggle/input reads that also hit scoring; Kaggle staff say they are
     "trying a new caching system recently to ease the pain" (Dustin, name-that-llm 750924, 10-20 13:47 UTC) for datasets, and that
     a demand queue remains. Our commit read regime changed between 10-19 04:43 and 18:24 UTC (MEASURED), consistent with that rollout.
   - Most likely path inside the hidden run (MODEL): the extra I/O time pushed S6F past its guard (start > 5.75 h, SKIPPED_GUARD) or
     past its 8.2 h absolute deadline (compact_infer stops 24 min before it, coverage < .98, or SIGKILL), so cell 18 wrote the base
     with ACL = 0.5. The base itself stayed intact (all three read exactly .916 = one blanked column; r224 ran, else Fracture
     would be blanked too, ~.88). A worker crash under I/O stress is the residual alternative (GUESS P ~.1).
2. **History: no earlier canary-like score.** 27/27 S6F-bearing hidden runs from 10-15 to 10-20 scored .956-.959; the only
   .91x-.92x scores are September / 10-04 weak-model entries (sec 4). The 10-21 batch is a regime change, not a known base rate.
3. **P(canary) for one 10-22 r6pw20 entry (MODEL/GUESS):** plain resubmit **~.50** (persists .57, worse .80, clears .05; weights
   .5 / .25 / .25). Guard variant **~.44** (it rescues ~13 % of would-be canaries, sec 6). Two independent copies of the variant:
   P(at least one clean) **~.75** (vs ~.56 for one, ~.68 for two plain).
4. **Guard variant built and checked locally** (not pushed): `kernels/kernel_v224r6flips6fpw20g85j`, sha bb995de9...; one line of
   cell 17 changes: G_START_H 5.75 -> 6.25, BUDGET_H 3.3 -> 4.5, DEADLINE_H 8.2 -> 8.5. `check_guard_variant.py` PASS 26/26.
   It changes nothing in a run that would have been clean (same predictions, same end time); it only gives a slow run up to 0.3 h
   more for S6F. Worst-case end ~8.55 h after NB_T0 (+ <= ~0.33 h bounded DEC cell before NB_T0) < 9 h.
5. **Recommendation:** if Eesh wants the r6pw20 hedge as B, submit the **guard variant**, ideally **2 copies** in the 10-22 slots
   (05:31 IST and ~06:30 IST), B = the first COMPLETE copy with >= .958, else R6 57248084. One copy only: still the variant (it
   weakly dominates). If the variant cannot be pushed + committed + gated by ~04:30 IST, plain-resubmit r6pw20 v1 (already gated).
   "None" is defensible (B = R6 is a solid hedge; r6pw20 adds +0.7 to +2.6 silver pts per earlier MODELs), but a slot costs nothing else.

## 1. Every path that writes the S6F canary (MEASURED code, r6pw20 = swide = all A22 kernels, cells 16-18)
Cell 18 `_s6f_blend` catches any exception after loading the base and writes base + ACL 0.5 (`mode V224FLIP_CANARY`):
| # | trigger (cell / function) | stochastic? |
|---|---|---|
| 1 | setup failed: frozen copy, driver sha, R7 dir on PYTHONPATH (cell 16) -> run status SETUP_FAILED | no |
| 2 | no float NB_T0 (cell 17) -> NO_NB_T0 | no |
| 3 | **start guard: kernel age > G_START_H 5.75 h -> SKIPPED_GUARD** | **timing** |
| 4 | less than MIN_RUN_S 1200 s before stop_at -> SKIPPED_NO_TIME | timing |
| 5 | Popen error -> LAUNCH_ERROR | rare |
| 6 | **driver SIGKILLed at hard = min(start + BUDGET_H 3.3 h, NB_T0 + DEADLINE_H 8.2 h)** (`killed`) | **timing** |
| 7 | driver receipt ok != True: env probe (no CUDA; 300 s timeouts), inputs (dataset sha / manifest / DONE marker), B stack probe (900 s timeout), any exception | mostly no; timeouts under extreme I/O |
| 8 | B_probs.csv sha != driver receipt | no |
| 9 | B_probs lacks a label / ids differ / probs outside [0, 1] | no |
| 10 | **coverage < .98**: compact_infer takes no new study after stop_at - 20 min (stop_at = hard - 240 s); a crashed or killed part (no receipt) counts as timed out, so **no retry**, and run_group kills the sibling part when one exits rc != 0 | **timing / crash** |
| 11 | rho floor: N scored >= 100 and rank corr(base, S6fall) mean < .60 or min < .40 | no (same inputs every run) |
| 12 | blend output non-finite / out of [0, 1] / wrong order | no |
Note: the base itself is guarded separately: Raptor stage subprocess timeout 2 h (cell 5, a timeout = notebook exception = ERROR),
R6 controller deadline 8.3 h with max fallback 1.0 (cell 7/9), r224 skips if < 40 min would remain before 9 h (cell 12, then the
V224 Fracture canary fires). NB_T0 is set at cell 5, after the DEC cell (PRE 600 s + PIP 300 s + TEST 300 s bounds).

## 2. Expected hidden-run timeline (c4 / R6 class; MODEL anchored on MEASURED latency 5h40-6h27, think/runtime.md)
| stage | normal age at end (h) | guard |
|---|---|---|
| DEC cell (before NB_T0) | 15-20 s in commits | bounded ~20 min |
| Raptor r384 | 0.30-0.42 | 2 h subprocess timeout (ERROR) |
| R6 setup + 4 folds x flip | ~3.8-4.0 | 8.3 h deadline |
| r224 + stage 3 + V224 canary | ~4.0-4.2 | skip if age > 8.33 h |
| S6F setup -> **S6F start** | **~4.05-4.25** (1.5-1.7 h before G 5.75) | G 5.75 |
| S6F B (decode-bound, ~10-11 s / study / T4) | ~6.0-6.3 | compact stop ~start + 2.9 h; kill start + 3.3 / 8.2 |
| blend + E-OA | ~6.05 total | |
Canary thresholds (MODEL, S6F needs ~1.95 h for 98 % coverage):
| slowdown shape | A22 constants | variant (6.25 / 4.5 / 8.5) |
|---|---|---|
| extra delay D before S6F | D > ~1.6 h (guard skip; deadline cut at 1.7 h) | D > ~2.0 h |
| uniform factor f on all stages | f > ~1.28 (total > ~7.8 h; the 8.2 h deadline binds first) | f > ~1.33 |
| S6F-only factor g | g > ~1.49 (the 3.3 h budget binds) | g > ~2.03 |
The 3 canaried runs kept an intact base, so their pre-S6F delay was at most ~4.2 h (else R6's 8.3 h deadline or r224's skip would
show in the score) (MODEL).

## 3. MEASURED: dataset-read slowness in our own commit sessions (io_markers.py by the forum track, run read-only; work/io_markers_all.out, work/commit_receipts.out)
Seconds of the log gap before each dataset-read marker (weights sha256 / R7-R6 wheel install / ASSETS_READY / arm load), commit wall:
| regime | commits | weights_sha | wheel_proc | assets | arm_load | wall s | S6F inputs s |
|---|---|---|---|---|---|---|---|
| T0 (10-15 .. 10-19 04:43 UTC) | ~35 | 0.7-2.4 | 4-30 | 7-87 | 1.7-2.3 | 528-839 (most 550-610) | 2.3-9.7 |
| T1 (10-19 18:24 .. 10-21 03:51 UTC) | eoa10/20/30, pw20, pw30, spl, swide | 10.3-22.5 | 34-58 | 9-24 | 6.9-13.5 | 604-719 | 2.7-4.6 |
| **T2 (10-21 03:07-04:10 UTC)** | **r6pw20, pw25, salt** | **95-145** | **204-325** | **69-83** | **56-83** | **1011-1231** | **17.5-20.5** |
- Per-study work is NOT slower in T2 commits: R6 decode 80 s / 2 studies (normal 76-89), S6F decode 24-26 s / study / worker,
  S6F B stage 58-64 s (MEASURED worker logs, receipts). In commits the extra ~0.16 h is all first reads of dataset files.
- Hidden outcome by regime: T0 ~22/22 clean, T1 5/5 clean (10-20 submits), 10-21 2/5 clean (MEASURED API). Commit tier and hidden
  outcome per kernel disagree twice (spl: T1 commit, canary; pw25: T2 commit, clean), so sessions draw independently.
- Open point (no hidden logs exist): a commit-sized +0.16 h cannot fire the canary; a hidden T2 session must have lost >= ~1.6 h,
  so either hidden sessions hit a worse tier, or the hidden test DICOMs (~1,300 studies, read by 4 stages) were also slow (GUESS).

## 4. Submission history scan (MEASURED, work/subs_all.json, 108 entries)
- .91x-.92x scores: 56158961 / 56158963 .913 (09-11 RadImageNet K18), 56162694 .922, 56172048 .924, 56177087 .923, 56198031 .926
  (09-11..13), 56830643 .923 (10-04 K3 TTR SOLO). All are genuine weak models of that era (descriptions), none had a canary.
- 7 "exceeded the allowed compute time" errors, 09-07 .. 10-03, all our own unguarded heavy pipelines, none since guards.
- Since the V224 canary (10-14) and S6F canary (10-15): 27 S6F-bearing hidden runs scored .956-.959 before 10-21 (0 canaries);
  10-21: 57463719 / 57464296 / 57464911 = .916 (3 canaries), 57463742 .959, 57464326 .960.

## 5. P(canary) for a 10-22 resubmit (MODEL / GUESS)
Per-session rate by scenario (Beta(1,1) posteriors): persists like 10-21 morning 3/5 -> .57 (90 % ~.25-.85); clears to T1/T0
(0/27) -> ~.03-.1, use .05; worse (final-day demand across comps, staff report a queue) -> ~.8. Weights .5 / .25 / .25 (GUESS: 10-21
was worse than 10-20 despite staff saying 10-20 fixes helped; 10-22 is RSNA's final day).
| entry plan | P(canary or ERROR) for B | P(B = r6pw20-class clean) |
|---|---|---|
| plain resubmit r6pw20 v1, 1 copy | ~.50 | ~.50 |
| guard variant, 1 copy | ~.44 | ~.56 |
| 2 plain copies | ~.32 both fail | ~.68 |
| variant + plain | ~.28 | ~.72 |
| **2 variant copies** | **~.25** | **~.75** |
Scenarios enter as products within a scenario (both copies share its regime), so 2 copies help least in "worse" (.48 both fail).
Latency: a slow clean run can take up to ~8.5 h plus queue, so submit at 05:31 IST: reads ~11:30-15:30 IST, before the 18:00 cutoff.
Every other 10-22 entry (MOACF(W), public twins) carries the same ~.5 canary risk; the same 1-line guard change would apply to them.

## 6. Guard tweak evaluation (MODEL)
- Rescue fraction of would-be canaries: delay shape (D ~ U(0, 4.2) in slow sessions) .15; uniform shape (f ~ U(1, 2.07)) .06;
  S6F-only shape (g ~ U(1, 3)) .36; mixture .45 / .45 / .10 (GUESS: S6F's own dataset reads are tiny, +15 s in T2 commits) -> **~.13**.
- Cannot do much better: DEADLINE_H is capped by the 9 h limit. Variant worst case: kill at NB_T0 + 8.5 h, p.wait <= 120 s, blend
  seconds, plus the pre-NB_T0 DEC cell (15-20 s typical, ~20 min bound) and unknown container overhead: <= ~8.9 h. 8.6 h would leave
  <= ~0.05 h in the worst case: rejected. G above ~6.2 buys nothing (an S6F starting later cannot reach 98 % by 8.1 h); 6.25 kept.
- No downside in clean runs: a run that would finish S6F under the A22 constants finishes identically (the deadlines are never
  reached; the driver, cell 16 and cell 18 are byte-identical). Only runs that would canary anyway change, to clean / canary /
  ERROR; canary and ERROR are equally unusable for B. P(new ERROR) GUESS < .005.
- Rejected (not guard-only): lower COV_MIN (a partial S6F blend is a different, untested entry), copy datasets to /tmp (code change in
  every stage), reorder S6F before r224.

## 7. The variant (built, checked, NOT pushed)
- `canary/build_guard_variant.py` -> `canary/kernels/kernel_v224r6flips6fpw20g85j/` (notebook sha256 bb995de972b6d627...,
  receipt `canary/kernels/build_receipt_v224r6flips6fpw20g85j.json`). Byte substitution of one unique substring of the A22 r6pw20
  notebook (source sha e4d92c0a verified). kernel-metadata: only id / title / code_file (eeshsaxena/rsna-sub-v224r6flips6fpw20g85j).
- `canary/check_guard_variant.py` (log `check_guard_variant.log`): PASS 26/26: one raw line changed; only cell 17 differs; AST
  identical except exactly (5.75 -> 6.25, 3.3 -> 4.5, 8.2 -> 8.5); cells 16 / 18 byte-identical; metadata datasets / docker / GPU
  unchanged; cell 17 itself executed on non-launching paths: hard deadline = min(start + 4.5, 8.5) at ages 4.0-6.2, skip at 6.3
  (A22: start + 3.3 / 8.2, skip at 5.8). Cell 17's header comment still says ">= 0.75 h after the hard deadline" (now 0.5 h);
  left unchanged on purpose so the diff is constants only.
- To use it (main session, needs Eesh's GO + PREREG Amendment 23): A23 registry entry with the new sha, push + Save & Run All on GPU
  by ~23:00 IST 10-21 (T2 commits took ~20 min; ~0.35 GPU-h of 25.6 h), run the r6pw20 gate key on its output (the gate checks
  outputs only, no guard constants; expect PROBE_CHECK PASS 17/17), A23 tick submits at 05:31 IST. B rule: first COMPLETE copy >= .958.

## 8. Files (this track, all under session_2026-10-21/canary)
- `cause.md` (this), `build_guard_variant.py`, `check_guard_variant.py` + `.log`, `kernels/` (variant + build receipt).
- `work/subs_dump.py` -> `work/subs_all.json` (read-only API dump), `work/commit_receipts.py` -> `.out` (S6F receipts of 46 real
  commits), `work/stage_marks.py` -> `.out`, `work/io_markers_all.out` (from the forum track's `io_markers.py`), and
  `work/s6f_driver_r6pw20.py` (the embedded driver decoded, sha a2acdd1d, read only).
- Read: A22 kernels (val/kernels, stack/kernels, stack/skeptic/kernels), a22 commit outputs, think/runtime.md, research_1hr/timing.md,
  tick_a22.log, DAY_1021.md, PROGRESS_21 / HANDOFF top, compact_infer (older heliosli copy), forum track's other_comps.out /
  staff_other.out (untrusted).
