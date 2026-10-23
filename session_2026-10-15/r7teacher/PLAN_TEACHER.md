# PLAN_TEACHER: go / no-go for the R7-teacher student line and for per-label probing

Written 10-14 ~14:30 IST by the R7-TEACHER plan writer (workflow wf_e9583d42-2e3). Inputs: job 1 (oof/RESULT.md), job 2
(labels/RESULT.md, PREREG_TEACHER.md), job 3 (plprobe/RESULT.md), the leakage-validity review and the feasibility-gain review,
RESEARCH_960.md, PREREG_10-14.md, PROGRESS_14-10-2026.md, the live box queue and the live Kaggle kernels list.
No em dashes. Every number is tagged MEASURED / MODEL / GUESS / CLAIMED (CLAIMED = a third party said it).
Nothing in this plan has been launched. Nothing was submitted, no dataset was created.

State at writing (MEASURED): team best HFR7-RR 57146261, public .956, API rank 489 (08:00 UTC). 10-14 UTC slots 5/5 used;
V224-exact (57171955) reads ~16:30 IST, V224-FLIP (57174021) ~18:00 IST. Next slots 05:30 IST 10-15. Box: L13 T123 s1044 f1 at
ep 12/21 (14:06 IST, ~16 min/epoch), so f1 ends ~16:45, f2 ~16:45-22:20, f3 ~22:20-04:00, f4 ~04:00-09:40 (MODEL from the
MEASURED 5.6-5.9 h/fold). Kaggle: about 14 GPU commits on Eesh's account in the 10-14 UTC day and about 30 since 10-10 (MEASURED
count; ~5-6 GPU-h, MODEL at ~10-12 min each).

---

## 0. Decisions

| Line | Decision | One-line reason |
|---|---|---|
| **R7-teacher student (R7-OOF run -> T_b student -> V224-F-S entry)** | **GO-WITH-FIXES, only on Eesh's explicit OK for the Kaggle quota.** Framed as a pick-B diversity bet, NOT a .960 lever. | Leakage check passes, the fold map is proven, everything downstream is built and dry-run. But the central blend effect is about 0 +/- 3e-4 (MODEL), the label builder has a real bug, and the kernel weighting is wrong as built. |
| **Per-label LB probing (job 3 builder + schedule)** | **NO-GO.** Builder stays on the shelf. | Its private value has an unknown sign (it hinges on an unmeasurable tau). In the null world a 4-probe Phase 1 gives at least one false win with P about .68 (MODEL). It censors the student and F/flip reads. It breaks RESEARCH_960 sec 5 and PREREG_10-14 Amendment 3. |

What this plan says about .960, plainly: P(public .960 by 10-22) stays about .01 (GUESS, unchanged from RESEARCH_960).
.960 needs +.0034 to +.0039 on top of .956 (MODEL). The student line's top-end case is about +.0029 (MODEL: student solo .954
at weight .45), and its central case is about +.0001 to +.0003. It is still the one remaining lever in the direction the top
teams use (own family on image-informed pseudo labels), it costs no pick-A risk, and its main cost is Eesh's GPU quota. That
is why it is GO-WITH-FIXES and not NO-GO. The other path to .960 is a usable public release from a .962-class team (keep the
release watch in session_2026-10-15/watch armed); it is outside this plan.

This plan revises RESEARCH_960 D8 ("R7-distilled student: drop", because the full R7-OOF did not exist and the student would land
10-16/17 at best). The OOF notebook now exists and is proven, and the timeline below lands the student entry 10-15 evening.
A student entry needs **PREREG_10-14 Amendment 4** (text in sec 7) before it is submitted.

---

## 1. Must-fix list (all before the student can enter the LB)

Fixes FX1-FX5 change label/gate code and must be in **PREREG_TEACHER Amendment A1**, sha-frozen and copied to the box
**before any full R7-OOF output is downloaded**. The 44 commit/proof rows already seen were the format check and exposed the
bug; A1 is a method fix, not a value-tuned choice.

| # | Fix | Source | Where |
|---|---|---|---|
| FX1 | **Clip bug.** `build_teacher_labels.py` maps `p|` after `clip(1e-4)`, which ties 56.5% of R7 cells (ACL 80%, Fracture 75%; MEASURED) and turns M(R7) into a near-binary label. Use the raw `z|` fold logits, UNCLIPPED, as the logit in the per-label affine logit match onto v4 (the z-score already sets the temperature). Keep the ECDF branch as is. | review 1 | builder line 151 (`lr = logit(np.clip(r, ...))`) and the job-1 layout rename at line 99 |
| FX2 | **Effusion and Baker's from `z|`, not the stacked `p|`.** The stackers (4 and 7 coefficients) were fit on all 4,344 OOF rows including the scored study. FX1 covers this if all 12 labels read `z|`. | review 1 | same |
| FX3 | **L0 per-fold leak gate on the full table** (free, decisive). Macro AUC of R7-OOF `z|` vs confident v4 cells (v4 <= .1 or >= .9) per hash fold, non-gold pool rows. PASS if max - min < **.020**. Review 1 proposed .015; with fold SE about .005 (review 1) the range of 4 null folds exceeds .015 with P about .2 (MODEL, range of 4 normals), while a mis-mapped fold shows about +.035 (MEASURED in-fold .968 vs OOF .933, n 40). FAIL branch (pre-registered): drop the R7 component for the single outlier fold's rows (T_b renormalises to v4 + fleet there; missing cap raised to 27% for this branch only); 2 or more outlier folds -> STOP the line. | review 1 | new step before LABEL_GATE in `build_teacher_labels.py`; `on_r7oof.sh` prints it |
| FX4 | **Checkpoint selection on the student's own table, not v4.** v4-raw-best penalises exactly the R7 uptake we pay for (Spearman v4 ~ R7-OOF only .608, MEASURED n 40). Set `--eval_labels` to the student's own table (T_b or T_a) in the 5 queue lines; the trainer's raw-best checkpoint and OOF then follow AUC vs T on held-out val rows (rewards generalising the teacher, cannot reward copying). G0 must check `sel_labels == table name`. Re-run the CPU argv dry run (31 studies, 1 epoch). | review 1 | `make_sheets.sh` -> `queue_r7t{b,a}.txt`; `student_gate.py` line 105 |
| FX5 | **Diversity ceiling G3' plus reporting.** Report `d = rho(S, R7-OOF) - rho_ref` (rho_ref = mean(rho(D, R7), rho(D2, R7)), as in G2) and `rho(S, fleet OOF)`. Skip the LB entry if `d > .5 (1 - rho_ref)` (the student closes more than half of the fleet's gap to R7: it is more R7 than fleet, and the blend model says the gain is gone at that correlation unless solo >= .950). Threshold is a GUESS. Review 2 proposed a flat `d > .02`; that contradicts G2's own floor (`d >= .10 (1 - rho_ref)` = about .025 at rho_ref .75), so it would fail every student that passes G2. Also record the second-order chain leak in A1 (R6 state fusion, plr7 cell weights, all-row stackers, holdout_fold None: up to about 1e-3 on train-row rulers, 0 on the hidden test; MODEL/GUESS). | both reviews | `student_gate.py`; PREREG_TEACHER A1 |
| FX6 | **Realistic SIM dry run.** The DT stand-in never hit the clip, so the dry run could not catch FX1. Rebuild the SIM teacher as DT logits rescaled per label to the per-label `|z|` quantiles of the 44 real rows, and re-run the full chain (build, L0, label gate, argv, gate). | review 1 | `sim_r7_oof.py` |
| FX7 | **Kernel F weighting.** The V224-F fleet cell 19 takes an equal raw mean over members (`mean = P.astype(np.float64).mean(0)`, MEASURED), so 20 K14 + 5 S gives S .2 of F (.04 of the blend), not the pre-registered .5 of F (.1 of the blend). Build V224-F-S with an explicit two-group probability mean `.5 mean(K14) + .5 mean(S)`; update the "exactly 20 members" asserts (V224-F receipt, `h_check --members`) and add an independent recompute of the two-group mean to the gate. Fresh commit test. | review 2 | `session_2026-10-15/v224f/build_v224f.py` (+ `v224f_check.py`) |
| FX8 | **Runtime.** 5 more members add about +0.44 h central / +0.58 h slow to the fleet (MODEL from the MEASURED microbench). V224-F-S runs about 5.25 h central, about 8.0 h slow-end x1.3, against the 8.3 h fleet wall and 9 h hard limit. Lower the fleet start budget from 5.5 h to **5.0 h** (`MIN_START_MIN = (8.3 - 5.0) x 60 = 198`). Do NOT put S on the time-aware flip base (V224-FF) unless its 4.9 h flip budget is cut by about 0.5 h. Separate runtime review before the submit. | review 2 | `build_v224f.py` K2_CFG |
| FX9 | **Honest framing to Eesh** (sec 6 message). Gate passes (label gate, G0-G4, gold58) are not evidence of value: gold58 compresses a .005 public gap to .0017 (SE .014, MEASURED/MODEL). Only an API best-id move decides. | both reviews | message |

---

## 2. Timeline 10-14 to 10-20 (IST; the UTC day resets at 05:30 IST)

Assumes Eesh's OK arrives about 15:30 10-14. Slip rules at the end.

### 10-14 Wed
- **14:30-16:30 (no approval needed):** FX1-FX6 + PREREG_TEACHER Amendment A1 (write, sha-freeze, copy to `~/r7t_labels/`,
  verify identical). Realistic SIM dry run on box CPU (nice 19, cores 34-39, about 30-45 min MEASURED-rate). FX7/FX8 builder
  work on the laptop starts in parallel (no GPU).
- **On OK (target 15:30):** `kaggle kernels list --mine --sort-by dateRun` must show 0 running.
  - **Default = parallel:** push `oof/kernel_r7oof_full_s0of2` (slug rsna-r7-oof) AND `oof/kernel_r7oof_full_s1of2_parallel`
    (slug rsna-r7-oof-b), if no other commit is planned before ~19:30. Both 10-15 lineup kernels (V224-F20, V224-FF20/25/30,
    V224-RM) are already commit-tested (MEASURED kernels list), so tonight's main need for a slot is a possible canary-fix
    commit after the 16:30 / 18:00 reads, which can wait (the next slots open 05:30). Ends about 19:00-19:45 (MODEL: 3.1-3.9 h
    plus boot each).
  - **Fallback = sequential in one slot** (review 2's default) if a commit must stay possible: shard 0 ends ~19:30; download
    `kaggle kernels output eeshsaxena/rsna-r7-oof -p out_full_s0` BEFORE the next push (same slug keeps only the latest
    version), verify, build shard 1 with `--skip-from out_full_s0/r7_oof.csv`, push; ends ~23:30.
- **16:30 / 18:00:** V224-exact and V224-FLIP reads (main session, unaffected).
- **~19:45 (parallel case):** download both, `verify_commit.py` on each, `merge_sessions.py r7_oof_ALL.csv out_full_s0 out_full_s1`,
  coverage check 4,407, copy to the box, `bash ~/r7t_labels/on_r7oof.sh <merged csv>`: L0 per-fold gate, REAL build, LABEL_GATE,
  `labels_gate.sh`. Expected PASS ~20:15.
- **~20:15-22:00:** install the 5 T_b lines with `qctl top` (fold order 4 3 2 1 0, as the labels RESULT gives) ONLY after PASS.
  They start when L13 f2 ends (~22:20); L13 f3/f4 move behind them. No preemption needed in the parallel case.
  Sequential case: PASS ~00:30 10-15 while L13 f3 runs (started ~22:20); stop L13 f3 (MEASURED null on top of the public blends,
  RESEARCH_960 sec 5) through the main session's usual queue procedure and re-append it after the student lines.
- **T_b student on the box:** 5 folds x 3.52 h = 17.6 h (MEASURED rate): ~22:20 10-14 -> ~16:00 10-15.
  Then the 5 T_a lines (queued right after T_b as the box filler that replaces the L13 null; 17.6 h): ~16:00 10-15 -> ~09:40 10-16.

### 10-15 Thu
- **05:30 onward:** the 10-15 lineup (V224-F20 / V224-FF / V224-RM) per the 10-15 workflow; reads through the day decide the base.
- **Daytime (laptop, no GPU):** finish V224-F-S (FX7/FX8) on the V224-F builder; prove it offline with 5 stand-in S members (five
  K14 checkpoints relabelled) so the two-group mean, asserts and canaries are tested before the real checkpoints exist. Pre-stage
  the dataset folder and the OAI-id scan.
- **~16:00-16:45:** `student_gate.py --student 'f{F}_s1047_ttr_r7tb' ... --infer` (box CPU, about 45 min MEASURED rate):
  G0-G4 plus G3'. FAIL -> no entry.
- **~16:45-20:30 (main session):** OAI-id scan; new private dataset with the 5 T_b checkpoints; build V224-F-S on whichever base
  the 10-15 reads made team best (V224-F20 preferred; V224-FF only with FX8's flip-budget cut); commit test (~12 min); gate incl.
  the two-group recompute; runtime review.
- **~20:30-21:30: submit V224-F-S** (10-15 UTC day slot), PREREG_10-14 Amendment 4 written first. One change vs the base: the F
  slot becomes .5 K14 + .5 S, w_f unchanged. Never alongside another entry on the same base. Read ~02:30-04:00 10-16.

### 10-16 Fri
- **Read V224-F-S (API best-id).**
  - Moved (positive): build REPLACE (F = S alone, w_f unchanged; per PREREG_TEACHER sec 7), commit, submit 10-16. S enters the
    pick-B candidate list as its F component.
  - Not moved: the T_b line stops for the LB (no split or w_f probing).
- **T_a** lands ~09:40; gate it. It gets at most one entry (10-16 or 10-17, same F-split form), only if it passes G0-G3'.
  It is the less K14-correlated table (no fleet share), so it is the one second iteration worth a slot after a negative T_b.
- Box after T_a: the main session's filler of choice (L13 f3/f4 re-appended).

### 10-17 Sat / 10-18 Sun
- Reads of REPLACE and/or T_a. Lock pick-B composition (max 3 components: F(+S), flip, r224 mirror; RESEARCH_960 sec 6).
- Runtime audit of the final B kernel (expected wall < 7 h with every guard; MODEL budget 9 h).
- **Latest OK that still works:** box route about 10-18 06:00 (R7-OOF ~7 h + student 17.6 h + kernel ~3 h before the freeze);
  rental route about 10-19 18:00 (needs Vast spend OK and Eesh's 2FA renewal).

### 10-19 Mon
- Last new-component entry day. Final-pick rules (RESEARCH_960 Amendment N): pick A never contains S; pick B contains S only if
  its entry moved the API best-id.

### 10-20 Tue 12:00 IST: training freeze
- No student training after the freeze. Any final candidate without its own scored run gets built and submitted by 10-21.

---

## 3. Resources Eesh must approve

| Resource | Amount | Tag | Notes |
|---|---|---|---|
| **Kaggle GPU quota: full R7-OOF run** | 2 sessions on T4x2, about 3.1-3.9 h each, **6.5-9 GPU-h** total incl. boot; plus at most one resume session (<= 1.5 h) if a 8.5 h deadline guard trips | MODEL (from 8 + 36 MEASURED studies: 17 s CPU and 2.55 s T4 per study) | Uses both concurrent slots for about 4 h (parallel) or one slot for about 8 h (sequential). Not a submission. |
| **Kaggle GPU quota: extra commits** | 2-4 commits x ~12 min = **0.5-0.8 h** (V224-F-S, REPLACE, T_a entry) | MODEL from MEASURED 438-636 s commits | Beyond the 2-3 h commit allowance of 12:16, which today's ~14 commits have largely used (MEASURED count, MODEL hours). |
| Weekly quota picture | used ~5-6 h this week + this plan ~7-10 h + other planned commits ~2-3 h = **~14-19 h** vs a weekly quota of about 30 h | MODEL / GUESS (quota size and reset day) | Eesh can see the remaining hours on his Kaggle profile. |
| **Box queue change** | insert 5 T_b lines (17.6 h) ahead of L13 f3/f4, then 5 T_a lines (17.6 h); L13 f3/f4 delayed about 35 h; in the sequential case stop L13 f3 early | MEASURED rates | L13 is a MEASURED null for the LB. The box stays busy (never-idle rule kept). Student data is RSNA-only. |
| **Vast** | **$0 by default.** Contingency only if the OK slips past ~10-18 06:00: one 4x4090 rental, $7.5-9.3 (MODEL), cap $15, keep >= $2 of the $17.23 | MODEL / MEASURED credit | Needs Eesh's spend OK and a fresh Vast 2FA session. |
| Kaggle dataset (main session) | 1 new private dataset (5 T_b checkpoints), 1 more if T_a enters | - | OAI-id scan first. |
| Submission slots | 1 (V224-F-S), +1 REPLACE after a positive read, +1 T_a: **1-3 of the ~40** left 10-15 to 10-22 | MEASURED slot count | Reads are sequential on the current best base. |

---

## 4. Expected gains (honest)

| Item | Public effect | Probabilities | Private / picks |
|---|---|---|---|
| R7-OOF run completes and the label gate passes | n/a | P about .85-.9 (GUESS: commits v1/v2 clean, deadline guard + resume exist) | enabler only |
| **T_b student, half of the F slot (V224-F-S)** | central -0.2 to +2.4e-4 (MODEL, review 1); expected +1 to +3e-4 (MODEL x GUESS, review 2); pessimistic -2 to -5e-4 (MODEL) | P(best-id moves) about .5 (GUESS); P(>= +.001) about .04 (GUESS from MODEL); P(display .957 from this alone) about .05-.10 (GUESS) | pick B only after a positive read; E[private] about +0 to +2e-4 (MODEL x GUESS) |
| REPLACE (F = S alone) after a positive read | needs student solo about .950 to beat F20 (MODEL); student solo prior .944-.950, centre about .947 (GUESS) | P(positive) about .3 conditional on a positive T_b read (GUESS) | pick B candidate |
| Top-end case (solo .954, weight .45) | about +.0029 (MODEL) | P < .01 (GUESS) | still short of .960's +.0034 |
| T_a second iteration | similar to T_b, slightly less correlated with K14 | P(best-id moves) about .4 (GUESS) | pick B candidate |
| **Per-label probing Phase 1 (NO-GO)** | public reads overstate private by +6 to +15e-4 after 24-35 probes (MODEL) | under tau 0: P(>= 1 false win in 4 probes) about .68 (MODEL); P(display .960 via probing) about .01-.05 and such a display carries +9 to +13e-4 of overfit (MODEL) | mean private +0.7e-4 (hedged +1.9e-4), sign unknown (MODEL x GUESS prior) |
| **Overall by 10-22** | P(.957) about .20 -> about .22 with the student line; P(.958) about .04; **P(.960) about .01** | GUESS | |

If Eesh wants the public number itself, regardless of the final private rank: per-label probing is the only path with a
non-trivial P(display .960), about .01-.05 (MODEL), at 4-6 GPU-h of commits and with read interference. That display would be
mostly overfit and would not improve the final standing. This plan does not recommend it; if he chooses it anyway, it runs only
after the student and F/flip reads, never feeds pick A, and accepts a probe only on a large frozen-team jump (plprobe/RESULT.md sec 3).

---

## 5. Stop rules

- R7-OOF coverage < 99% after one resume session -> STOP (the builder's missing cap is 1%).
- L0: one outlier fold -> pre-registered drop of that fold's R7 component; 2 or more -> STOP.
- LABEL_GATE: neither T_b nor T_a passes -> STOP; keep the merged R7-OOF table as a diagnostic only.
- Student gate G0-G3' FAIL -> no LB entry for that table.
- Runtime review not SUBMIT -> no entry.
- T_b read negative and T_a read negative (or T_a gate FAIL) -> the line ends. No split probing, no w_f retune.
- After 10-20 12:00 IST: no training. Pick A never contains S.

---

## 6. Message to Eesh (ready to paste)

> The R7-teacher bet is ready to run. We proved which of heliosli's 4 R7 fold models never saw each train study, built a
> private notebook that scores every train study with that held-out model (tested on 44 studies), and prepared a student
> model on those labels for the box. Honest value: expected about +0.0001 to +0.0003 public; about a 4% chance of +0.001;
> .960 stays about 1%. It is a lottery ticket for pick B, not a sure step to .960. It does not touch pick A.
> I need your OK for: (1) about 6.5-9 GPU-hours of your Kaggle quota for the scoring run (2 sessions, about 4 h if run side by
> side tonight), plus about 0.5-0.8 h more for 2-4 commit tests; (2) moving the student (about 18 h) ahead of the L13 seed on
> the box. No Vast spend unless your OK comes after Saturday morning. Per-label LB probing: I recommend NOT doing it (it mostly
> overfits the public board). Reply "OK R7" to start, or "no" and nothing changes.

---

## 7. Pre-registration text to add (main session, before the entries)

**PREREG_TEACHER Amendment A1** (before any full R7-OOF output is downloaded): FX1-FX6 exactly as in sec 1, with the new
`build_teacher_labels.py`, `student_gate.py`, `sim_r7_oof.py`, `make_sheets.sh` md5s, and the new sheet md5s. Tables, mixing
weights, seeds (1047 / 1048), recipe and the G0-G4 bars are otherwise unchanged.

**PREREG_10-14 Amendment 4** (before the V224-F-S submit):
> Amendment 4. R7-teacher student entry. V224-F-S = the current team-best V224-F base with the F slot = .5 mean(K14) + .5 mean(S)
> (S = 5 T_b student checkpoints, PREREG_TEACHER + A1), w_f unchanged, fleet start budget 5.0 h. One change vs its base. A win
> counts only as an API best-id move. If it wins: one REPLACE entry (F = S alone). T_a gets at most one entry, same form, only if
> it passes G0-G3'. S never enters pick A; it enters pick B only through a winning entry, as part of the F component (max 3
> components). No per-label public tuning (unchanged); no tuning of the .5 split or of w_f on the student line.

---

## 8. First 3 actions

1. **Now, no approval needed:** apply FX1-FX6 and write PREREG_TEACHER Amendment A1 (sha-frozen, copied to `~/r7t_labels/`,
   laptop and box identical) BEFORE any full R7-OOF byte exists; run the realistic-|z| SIM dry run end to end on the box CPU
   (nice 19, cores 34-39; GPU queue untouched).
2. **Now:** send Eesh the sec 6 message. On "OK R7": check `kaggle kernels list --mine --sort-by dateRun` (0 running), then push
   `session_2026-10-15/r7teacher/oof/kernel_r7oof_full_s0of2` and, if no commit is needed before ~19:30,
   `kernel_r7oof_full_s1of2_parallel` (slug rsna-r7-oof-b) in parallel; otherwise shard 0 only and the sequential path.
3. **While it runs:** build V224-F-S (FX7 two-group F mean + assert updates + recompute gate; FX8 start budget 5.0 h) on
   `session_2026-10-15/v224f/build_v224f.py` and prove it offline with stand-in S members. When the R7-OOF lands: download,
   `verify_commit.py`, `merge_sessions.py`, copy to the box, `on_r7oof.sh` (L0 + label gate), and only on PASS install the
   5 T_b lines with `qctl top` so they run when L13 f2 ends (~22:20).

Key paths: `oof/RESULT.md` (run commands), `labels/RESULT.md` + `labels/PREREG_TEACHER.md` (tables, gates, kits),
`plprobe/RESULT.md` (shelved probing builder; amendment text in sec 3), `session_2026-10-15/v224f/RESULT.md` (base kernel for
V224-F-S), box `~/r7t_labels/` (builder, sheets, gate).
