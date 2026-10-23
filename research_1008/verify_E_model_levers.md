# Verification of dimension E (model-side levers), 2026-10-08 ~22:15 IST (v2)

This replaces the 21:19 version of this file. The verdicts are mostly the same. The main change is new box data: the
TTR320 probe has reached **epoch 10**, and that overturns the report's "below the bar, P(PASS) 5-10%" read.

Read-only throughout: no Kaggle write (one `kaggle datasets list -s "convnext small"` read), no paid call, no Vast,
light Python on the laptop only.
- **Box:** answered once at 22:04 IST (read-only `grep`/`tail` of the probe log and the gate script), then timed out
  again at ~22:08. Nothing was written on the box.

Tags: **MEASURED** (source given), **MODEL** (computed from measured inputs), **GUESS** (judgement), **CLAIMED** (forum).

Scratchpad: `VS2 = C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/01efcc80-088f-477c-8a8a-b504ddf28d6a/scratchpad/verify_E2/`
- `traj10.py`, `traj10b.py`: 320 forecast re-done from epoch 10 against the report's own 20 finished runs (`e_levers/all_logs.txt`).
- Earlier checks reused from `verify_E/`:
  - `vt.out`: K9N transfer and 4th leg, continuous optimiser;
  - `vt2.out`: transfer at K9N-a/b weights;
  - `timing.out`: wall time per epoch.

---

## 0. Bottom line

1. **The headline holds and is slightly too generous.** At fixed K9N weights, a fleet SOLO gain moves the blend by
   ~.28 x, which is just the fleet's weight (MODEL `vt.out`, `vt2.out`; it reproduces `transfer.out` to the 4th
   decimal).
2. **Systematic error: set swaps are counted as whole-fleet changes.** Every lever the report prices (K10a, 24-slot,
   ConvNeXt-S, 320) replaces ONE of the two 5-member sets. The fleet SOLO therefore moves by ~half the set-level
   change, and the blend by ~.14 x the set-level change (MODEL).
   - K10a: **+.0000 to +.0003, central ~+.00015** (MODEL).
   - 24-slot set: **-.0001 to +.0004**.
   - ConvNeXt-S set: **+.0000 to +.0004**.
   - Dimension total: **~-.0001 to +.0007, central ~+.0003** (MODEL, before the ~.0005 model optimism), vs the
     report's +.0002 to +.0012.
3. **NEW, refutes finding 2's premise: the 320 probe is no longer tracking below the bar.**
   - Raw val vs v4 by epoch: ep7 .8753, ep8 .8798, ep9 .8823, **ep10 .8833**. gold58 at ep10 is .9109, gold74 .9025
     [MEASURED `~/fleet_runs/f0_s1045_ttr320frac75.log`, read 22:04 IST].
   - The probe gained **+.0141** from ep6 to ep10. The 20 finished runs gained +.0101 (SD .0034) over the same span
     [MEASURED, `traj10.py`].
   - Re-forecast from ep10 [MODEL `traj10.py` / `traj10b.py`]:

     | reference runs | predicted best | P(best >= .8909) | predicted last-3 | P(last-3 >= .8900) |
     |---|---|---|---|---|
     | all 20 runs | .8895 (SD .0021) | .25 | .8891 | .33 |
     | the 10 runs with 21-22 epochs | .8895 (SD .0015) | .17 | .8891 | .24 |
     | 288 f0 at a matched schedule fraction (ep11 .8852, then +.0057) | ~.889 | | | |

   - **P(G320' PASS) is ~20-35% (MODEL)**, not 5-10% (report) or 3-7% (my 21:19 read). gold58 >= .905 at the best
     epoch is very likely to hold (.9109 at ep10).
   - The decision "no Vast for 320" still stands, because the value even on PASS is tiny (sec 2, F2). But the PASS
     branch now needs a pre-registered K10 layout (M6).
4. **Factual fix on 320 timing.**
   - The probe was relaunched at 18:08:53 and was in ep11 at it 2800/3466 by 22:04, which is ~19.9 min/epoch wall
     [MEASURED]. The raw-line "17.7 min" leaves out the EMA eval.
   - So 21 epochs = **~7.0 h/fold** and the probe closes ~01:10-01:30 10-09.
   - On PASS, box-only folds 1-4 take **~28 h**, landing ~10-10 05:30-06:00, not 25-26 h (MODEL).
5. **Decisions that stand:**
   - let G320' decide;
   - Q4 is the FAIL default;
   - close EMA, SWA, last-epoch, multi-scale/3rd TTA and reseeds;
   - keep two sets of <= 288 px next to P + N;
   - skip own CoAtNet;
   - ConvNeXt-S is deferred.

| # | finding | verdict | corrected expected public gain |
|---|---|---|---|
| 1 | K10a = Q4 288 frac75 replaces set D | **revise** | +.0000 to +.0003 blended, central ~+.00015 (MODEL); P(this branch) ~65-80% |
| 2 | No Vast on 5x320 | **revise** (premise refuted by ep10; decision stands; Eesh's call) | ~+.00003 expected (MODEL) |
| 3 | Structural ceiling, transfer ~0.3 | **keep** (+ half-fleet corollary) | <= ~+.0004 per set-swap lever (MODEL) |
| 4 | All-24-slot 288 probe | **revise** (gain, gate, ordering) | -.0001 to +.0004 blended (MODEL on GUESS) |
| 5 | Close EMA / SWA / last-epoch / multi-scale / 3rd TTA / reseeds | **keep** | 0 (protective) |
| 6 | Budget rule: two sets <= 288 next to P + N | **keep** (with caveats + PASS-branch conflict) | protective |
| 7 | ConvNeXt-S/B: defer | **revise** | +.0000 to +.0004 minus margin cost (MODEL on GUESS) |
| 8 | Own CoAtNet: skip | **keep** | +.0000 (MODEL, r GUESS) |

---

## 1. Cross-cutting checks

### 1.1 Transfer rule (finding 3): reproduced
- In `vt.out`, F +.001 gives K9N-a +.00028 and F +.002 gives +.00056, at r_FN .90 / .92 / .94 (MODEL). The same
  +.00028 holds at K9N-b weights and with N at .946-.949 (`vt2.out`).
- Measured analog: K5 SOLO .943 / main .946 and K7i SOLO .944 / main .946 [MEASURED `B_lb/submissions_1008.csv`].
  This is weak evidence (display rounded to .001), but it is consistent with the model.
- Precision on the 4th-leg band (MODEL `vt.out`): +.0015 or more needs SOLO >= .940 at r .85, or >= .944 at r .88.
  - .940 / r .88: only +.0006 to +.0011.
  - .935 / r .85: +.0006 to +.0010.
  - Strength matters more than the report's wording says.

### 1.2 Half-fleet factor (findings 1, 2, 4, 7)
- K9 fleet = set C (5x288 mix50 s1044) + set D (5x224 frac75 s1043), averaged with equal weight [MEASURED K9N_RUNBOOK
  sec 0].
- Replacing one set moves the fleet signal by ~half the set-level change, so blended ~= .28 x .5 = **.14 x the
  set-level SOLO change** (MODEL).
- The report's per-member evidence is valid: 288 minus 224 set A, +.0073 / +.0026 / +.0028 / +.0059 / +.0028, mean
  +.0043 v4 OOF [MEASURED, re-ran `parse.py`]. But its blended figure (.28 x .002 = .0006) treats the swap as a
  whole-fleet change.
- **Caveat the report also missed:** E (288 frac75 s1045) shares its resolution with C. The C-E pair is likely more
  correlated than C-D (288 vs 224 Spearman .959; same-recipe seed pairs .963-.969 [MEASURED NOVEL_PLAN/oof_report]).
  Some of the gain is lost to lower diversity inside the fleet. The effect is small (GUESS), but it argues for the
  gold read in M10.

### 1.3 320 timing [MEASURED log + `timing.out`]
- Raw line 17.5-18.0 min, EMA line 19.8-20.3 min (the EMA line is the wall time); TRAIN-ONLY 15.6 min at ep10.
- 18:08:53 to 22:04 = 235 min for ~11.8 epochs, which is 19.9 min/epoch.
- Cross-check: by the same reading the 288 log gives 6.6 h at 24 epochs, which matches HANDOFF's "~6.7 h each".

### 1.4 Probe forecast (finding 2): superseded by ep10
- The report's ep6-based forecast (.8855) is already about -.004 off: the ep10 value is .8833 and the probe still has
  11 epochs of cosine decay left.
- Why the ep6 method under-predicted: high-resolution and longer-schedule runs make larger late gains (288 runs
  ep10 -> best +.0065 to +.0102), and a single ep6 point carries the probe's own noise.
- Matched epochs vs 288 f0 (24 ep): ep10 .8833 vs .8821, gold74 .9025 vs .9039, gold58 .9109 vs .9128 [MEASURED].
  - So 320 is level with 288 at the same epoch.
  - At the same schedule fraction it is ~.002 behind, because 288 had 24 epochs.

---

## 2. Finding-by-finding

### F1. K10a (Q4 288 frac75 s1045 replaces set D): REVISE
- **Holds:**
  - Q4 is 5 x 288 frac75 s1045 at 21 epochs, ~5.8 h/fold (16.5 min wall/epoch), ~29 h [MEASURED queue copy; MODEL].
  - Runtime 7.56 / 8.10 h, margin 1.04 / 0.50 h [MODEL `runtime_e.out`, which reproduces K9N's 7.40 / 7.95 h].
  - The Fracture lever is kept, not re-counted; no double count with K9.
- **Refuted:**
  - The gain is about half: **+.0000 to +.0003 blended, central ~+.00015** (MODEL).
  - Expert-label evidence is ~0 anyway: T0-A0 gold74 -.0003 (SE .0046) and C-A gold74 +.0007 [MEASURED NOVEL_PLAN
    1.2, PROGRESS_08 10:05].
- **Branch probability changes:**
  - With P(G320' PASS) ~20-35%, Q4 runs right after the probe only ~65-80% of the time.
  - On PASS, Q3b (320 folds 1-4, ~28 h) runs first, so set E lands ~10-11 midday at the earliest (MODEL).
  - "Ready ~10-10 07:00" is the FAIL-branch best case. A power cut (one already, -6.5 h, MEASURED PROGRESS 18:10) or a
    label-side `qctl top` pushes it later.
- **"Same recipe as K9" is not exact:**
  - Q4 runs 21 epochs at 288, while set C ran 24.
  - "21 = 24 epochs" is verified only at 224 [MEASURED verify_recipe 2.6].
  - So the per-fold bars must not assume set C's 24-epoch level.
- **Corrected action:**
  - keep Q4 as the FAIL default;
  - build K10a with 21-epoch bars;
  - once E lands, choose among C+D (K9), C+E and D+E by a CPU gold74 read (M10);
  - keep the report's pre-registered K9-SOLO rule.

### F2. No Vast on 5x320: REVISE (premise refuted, decision stands, Eesh decides)
- **Refuted premise:** "tracking below the G320' bar, P(PASS) 5-10%". At ep10 the probe is level with 288 at matched
  epochs and gaining faster than the reference runs. P(PASS) is **~20-35%** (MODEL, sec 0.3).
- **The value on PASS is still tiny:**
  - G320' PASS means 320 frac75 reaches roughly the 288 f0 level (the .8909 bar IS 288 f0's best).
  - Against the FAIL alternative (Q4 = 288 frac75), the 320 set adds only 288 -> 320, GUESS +.000 to +.001
    set-level, x .14 = **<= +.00014 blended**.
  - Expected value: .2-.35 x ~+.0001 = **~+.00003** (MODEL).
  - The serving cost is the worst of all K10 options: C+320 slow margin 0.38 h [MODEL `runtime_e.out`].
- **Timing:**
  - Box-only folds 1-4 are ~28 h and land ~10-10 05:30-06:00.
  - Vast in parallel could land the set ~10-09 evening, i.e. one possible extra K10N slot on Saturday.
  - That slot is worth ~+.0001, and building plus testing a new kit on Friday night carries its own error risk (GUESS).
- **Eesh's call, not ours:** at 20:50 Eesh said "use the ~$9 Vast credit", with "Vast trains 320 in parallel if PASS"
  as the plan [MEASURED PROGRESS 20:50]. "No Vast" goes to him as a recommendation with these numbers. It must not be
  applied silently. On FAIL there is no conflict.
- **Data note:** a rented GPU under our own account is normally fine under rule 2.4 (data only to participants)
  [MEASURED RULES item 9]. Moving the 11 GB store is friction, not a rule breach (GUESS).

### F3. Structural ceiling: KEEP
- Reproduced exactly and robust (sec 1.1).
- Add the corollary: a one-set swap transfers at ~.14 of its set-level change, so every own-model lever is
  <= ~+.0004 blended (MODEL).
- Use the precision-fixed 4th-leg band from sec 1.1.

### F4. All-24-slot 288 probe: REVISE
- **Mechanism and cost check out:**
  - training uses 16 of the 24 slots that serving uses [MEASURED verify_recipe 2.7];
  - 8.1 h/fold and 10.0 GB at bs 1 [MODEL recipe.md 1.3];
  - the F1 24-slot run was confounded [MEASURED recipe.md].
- **Gain is half-counted:** blended **-.0001 to +.0004** (MODEL on the GUESS SOLO range).
- **The gate is weak:**
  - Single-run seed noise is SD ~.0013-.0015 [MEASURED verify_recipe sec 3], so a difference of two runs has SD
    ~.002. A +.002 bar passes a null effect ~15-20% of the time (MODEL).
  - The comparator (288 f0 mix50 s1044, 24 ep) differs in labels, seed and schedule.
  - gold58 >= .905 never binds (0/108 late epochs below it) [MEASURED verify_recipe 2.13].
- **Better design (M2):** a seed-matched pair with Q4 f0, decided before Q4 folds 1-4.

### F5. Close EMA / SWA / last-epoch / multi-scale / 3rd TTA / reseeds: KEEP
- **EMA:** last EMA minus best raw is v4 -.0014 (SD .0007, n 20) and gold74 -.0001 (SD .0020) [MEASURED
  `lastbest.out`]. The ep10 probe line shows the same pattern: EMA .8681 vs raw .8833.
- **Last-epoch vs val-best:**
  - set C: +.0009 / +.0008 / .0000 / -.0012 / -.0007 gold74;
  - set D: +.0006 / -.0006 / -.0001 / .0000 / +.0007;
  - both ~0.
- **Epoch averaging:** 288 last-3 .8900 = ep23 .8900, so the selection-free gain is 0 [MEASURED parse].
- **Multi-scale:** -.0005 to -.0007, from the box `ms_report` (not re-read; it agrees with NOVEL_PLAN's closed list).
- **nartaa's SWA +.001 (CLAIMED):** another schedule; it does not transfer.

### F6. Budget rule: KEEP, with caveats
- The numbers reproduce [MODEL `runtime_e.out` on the MEASURED microbench].
- **Caveats:**
  1. The 288 microbench is n = 4.
  2. The slow case holds the parent at 6.0 h, a single-session upper bound (an older model said 7.7-8.05 h).
  3. "Reject if the slow end is > 8.1 h" sits exactly at K10a's 8.10 h. Restate it as a parent allowance before N is
     cut: >= 6.5 h. K10a gives 6.50 h, K9N 6.65 h.
- **PASS-branch conflict (new):**
  - Under the report's own rule, C+320 (8.22 h slow, 0.38 h margin) is rejected.
  - The rule-compliant D+320 (8.08 h) drops the 288 mix50 set C instead.
  - So on PASS, K10 is either a parent-free kernel or a layout with only a small gain. Pre-register this before
    ~01:30 (M6).

### F7. ConvNeXt-S/B: REVISE (deferral stands)
- **Confirmed:**
  - none of our datasets holds S/B weights (21:00 read);
  - box cache absence is the 10-07 read [MEASURED verify_recipe 2.8].
- **Gain halved** to **+.0000 to +.0004** blended, minus the 0.15 h slow margin (MODEL on GUESS).
- **Missed fact:** public, competition-trained ConvNeXt-S KNEE weights exist [MEASURED `kaggle datasets list -s
  "convnext small"`, 22:10 IST]:
  - `kirilllabzin/knee-convnext-small-v17-source` (187 MB, 34 downloads);
  - `ngochienphan/rsna-convnext-small-384-ep20-raw` (184 MB);
  - also `dreaddevelopment/raptor-cnn336` (185 MB, 82 downloads). This is likely the Raptor CNN already behind our
    parent and teacher, so it is not decorrelated (GUESS).
  - These belong to the public-model dimension: no training cost. Score, licence and correlation are unknown, and any
    download needs Eesh's OK.

### F8. Own CoAtNet: KEEP (skip)
- +.0000 at r .93-.95 vs P and N (MODEL `vt.out`).
- At r .88-.90, a .940-.946 CoAtNet would add +.0006 to +.0020 (MODEL). But high r is the likely case: mix50 carries a
  Raptor teacher, the parent holds 3 Raptor CoAtNets, and N is one (GUESS).
- Add 1-2 days of trainer work, the fp16 BatchNorm risk [MEASURED HANDOFF 10-06] and serving time that does not fit.

---

## 3. Missed by dimension E

- **M1. Half-fleet accounting** (sec 1.2). Apply it to every set-swap estimate in the orchestrator's ranking.
- **M2. Paired slot-count test inside Q4 ($0, saves ~40 V100 h).**
  - Order: Q4 f0 (16 slots), then a 24-slot f0 with the same s1045, frac75 labels and 21 epochs, then Q4 f1-f4 with
    the winning setting.
  - Pre-register: PASS if last-3 raw >= Q4 f0 last-3 + .002 AND gold74 is not lower by more than .003.
  - Cost: K10a lands ~8 h later (GUESS: minor against the 10-20 freeze).
- **M3. CPU or 1-GPU-hour diagnostic first:** re-serve one existing 288 member's OOF fold with random 16-slot subsets
  vs all 24. If 24 already beats the subset mean clearly, the train/serve mismatch is not hurting, and the prior for
  the 24-slot probe drops (GUESS).
- **M4. Levers aimed at N, the .60 leg.** They transfer ~.6 per unit of SOLO, vs ~.14 for a set swap (MODEL).
  - Candidates: N multi-crop TTA (CLAIMED .948 for both nartaa crops) or the public .945 r224 as N2.
  - Each costs +25-40 min of serving, so they fit only parent-free.
  - Our own multi-scale result is a negative prior [MEASURED].
  - GUESS +.000 to +.0006. This is for the blend dimension.
- **M5. Public competition-trained ConvNeXt-S knee weights** (F7) are the only "decorrelated 4th leg" candidates found.
  Hand them to the public-model dimension: notebooks, licence and standalone score first, download only with Eesh's OK.
- **M6. Pre-register the G320' PASS branch before ~01:30 10-09.**
  - P(PASS) is now ~20-35%.
  - Decide in advance:
    - (a) which set the 320 set replaces: D+320 fits the budget, C+320 does not;
    - (b) whether Q4 still runs after Q3b;
    - (c) the Vast question goes to Eesh with the ~+.00003 EV number.
  - Without this, a PASS at night leaves the queue running 28 h of 320 folds toward a kit the budget rule rejects.
- **M7. T4 quota (~85 h from 10-10):** within this dimension only 224 jobs fit, and reseeds are worth ~+.0001
  (MODEL). Send T4 hours to label-side retrains (fp, VLM).
- **M8. Parent-free kernel** is the precondition for every serving-bound model lever (a third set, C+320, ConvNeXt-S,
  N TTA/N2). Rank it above them in the blend dimension.
- **M9. Per-label F/N weighting** to keep the frac75 Fracture gain (+.0118 gold74, MEASURED). In a .30-weight fleet that
  gain is diluted ~3x. Per-label weights from gold are in-sample, so this needs a pre-registered gate. Blend dimension,
  GUESS +.0000 to +.0005.
- **M10. K10 set choice by measurement, not assumption:** once E lands, run a CPU gold74 / OOF read of C+D, C+E and D+E,
  including the C-E correlation (sec 1.2 caveat).
- **M11. Full-data refit** (parked, GUESS ~0 to +.001 SOLO, cannot be gated) is <= ~+.00015 blended with the half-fleet
  factor (MODEL). Stays parked.

---

## 4. Sources
- **Box (22:04 IST, read-only):**
  - `grep "^ep .. raw"` and `tail` of `~/fleet_runs/f0_s1045_ttr320frac75.log`;
  - `~/queue/gates/g320_gate.py` (rule: best >= .8909 OR last-3 >= .8900, AND gold58 at best >= .905; it writes once,
    never overwrites).
- **Local:**
  - `research_1008/E_model_levers.md`;
  - `e_levers/{all_logs.txt, parse.py, traj.py, lastbest.out, transfer*.out, runtime_e.*}`;
  - old SP `queue/queue.txt`, `novel/{recipe.md, verify_recipe.md}`, `k9n/K9N_RUNBOOK.md`;
  - PROGRESS_08-10-2026.md (to the 21:20 entry), HANDOFF.md, NOVEL_PLAN_10-07.md, RESEARCH_TO_TOP_10-06.md,
    FORUM_DIGEST_10-08.md, PUBLIC949_ASSESSMENT.md, RULES_AND_HOST_COMMENT_10-08.md.
- **Kaggle (read-only CLI):** `datasets list -s "convnext small"`.
- **New:** `VS2/{traj10.py, traj10b.py}`; `verify_E/{vt, vt2, timing}.{py,out}`.
