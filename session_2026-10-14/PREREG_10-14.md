# PREREG 10-14 (written 02:00 IST 10-14, BEFORE the HFR7-RS read and before any 10-14 submission). No em dashes.

## Reading method (all entries display .956)
- Frozen-team method (session_2026-10-13/frozen_read.py OLD NEW): count teams whose score and LastSubmissionDate did not change between
  two LB snapshots and that rank above us. Fewer = the new entry beat our previous best. Take an LB snapshot right after EACH entry
  scores, so a move can be attributed to one entry.

## 10-14 UTC-day lineup (one change per entry vs HFR7-RS, the current candidate best)
1. **HFR7-RS-F30** (`eeshsaxena/rsna-sub-hfr7rs30` v1): fleet weight w_f .30 instead of .20. Gates RS_CHECK 38/38 + HCHECK 16/16.
2. **HFR7-RS-O45** (`eeshsaxena/rsna-sub-hfr7rso45` v1): Raptor_mix r384 .35 / r224 .20 / ours .45 (was .45 / .25 / .30); w_f .20.
   Gates RS_CHECK 38/38 + HCHECK 16/16.
   Order: submit 1, then 2 at least 45 min later so their scoring times separate; snapshot after each.
3. Reserved (only after the S42 OAI Raptor seed gate PASSES, ~11:00, and its own kernel prereg): swap S42 in for one rented seed.
4-5. Reserved for follow-ups on whichever of 1 / 2 moves the frozen count (e.g. w_f .40, or ours .60), each pre-registered in an
   amendment before submitting.

## Decision rules
- If F30 beats our best: next probe w_f .40 on the same base. If not: keep .20.
- If O45 beats our best: next probe ours .60 (r384 .25 / r224 .15). If not: keep .30.
- Final picks (10-22): A = copy floor (HR7 or HR7-RR), B = best differentiated entry by frozen-team evidence.

## READ 07:28 + AMENDMENT 1 (07:35, before any further 10-14 submission)
- **HFR7-RS = 0.955** (below HFR7-RR .956). Adding our 2 RSNA-only Raptor seeds at .30 of the Raptor slot HURT on public, despite the
  gold58 gate PASS (gold58 is a weak proxy: forum r .66 with LB). Likely cause: no OAI rows, so weaker on the OAI-supervised labels.
- **O45 CANCELLED** (more weight on the same seeds; the timed submit task was stopped before 07:55; not submitted).
- F30 (57162815, submitted 07:08) sits on the RS base, so it can only test w_f on a base that is already worse; it is kept as is.
- Next entries go back to the **HFR7-RR base** (our best): (a) HFR7-RR-F30 = w_f .30 on the RR base (one change vs our best);
  (b) the S42 OAI-trained Raptor seed only if its box gate passes AND at a small Raptor-slot weight (pre-register the weight before the
  kernel build; the RSNA-only seeds are retired from the LB line).

## AMENDMENT 2 (08:15, before any HFR7-RR-F30 / F40 read)
- Slot 2: **HFR7-RR-F30** (`eeshsaxena/rsna-sub-hfr7rr30` v1; RR_CHECK 31/31 + HCHECK 16/16), submitted now.
- Slot 3: **HFR7-RR-F40** (`eeshsaxena/rsna-sub-hfr7rr40` v1; RR_CHECK 31/31 + HCHECK 16/16), submitted >= 45 min later so the two
  score at different times; LB snapshot taken after each scores; frozen-team read vs the snapshot before it.
- Rule: if F30 beats our best (frozen count drops) and F40 beats F30, the fleet weight optimum is >= .40 (next probe .50); if F30 beats
  but F40 does not, keep .30; if neither beats, keep .20 (HFR7-RR) as pick B base.
- Slots 4-5: reserved for the S42 OAI seed (after its box gate) and one follow-up, each pre-registered first.

## AMENDMENT 3 (12:50, before any V224 read; text per RESEARCH_960 sec 4)
- Slot 4 = **V224-exact** (`eeshsaxena/rsna-sub-v224` v1): varshitha's public V224 recipe, her 13 cells verbatim in her order + a
  canary cell (r224 not used -> Fracture 0.5 + CANARY_FIRED). Gates: V224_CHECK PASS 21/21; its 3-row submission.csv is IDENTICAL
  to her own published commit (0 of 36 cells differ). Forced-fallback control commit (v224ctl): canary fired, Fracture = 0.5 exactly.
  Wheel diagnostic: r224 raw stock vs R7 wheels max |diff| 2.4e-4 = the same size as same-version run-to-run noise (A vs B), so
  package order does not explain the HFR7-RR gap.
- Slot 5 = **V224-FLIP** (`eeshsaxena/rsna-sub-v224flip` v1): one change, R7 flip TTA forced. V224_CHECK PASS 23/23 (flip mode;
  MCL plain pass only; 10 of 11 other R7 columns move). R7 per-study GPU time x2.08 on the commit. Submitted >= 45 min after slot 4
  only if a runtime review keeps the hidden wall clearly under her 9 h guard (else held).
- Expected: V224-exact lands in the V224 pile (r290-414) and becomes our team best; V224-FLIP >= V224-exact. A win counts only as a
  best-id move (API reader session_2026-10-14/research960/lbprobe/api_frozen.py). Pick A becomes V224-exact once it scores clean.
- w_f rule from today's pending reads (best-id after all three score): best stays HFR7-RR -> .20; RR-F30 -> .25; RR-F40 -> .30.

## READS 12:46-13:30 (API best-id reader)
- RR-F30 (57163771) = .956, team best unchanged (HFR7-RR 57146261). F30-RS (57162815) = .955. **RR-F40 (57164843) = .955.**
- Per the w_f rule: best stays HFR7-RR -> **w_f = .20** for every fleet build from now on (.30 does not beat .20; .40 drops to .955).
- V224-FLIP SUBMITTED 13:38, ref 57174021 (all 5 slots of the 10-14 UTC day used).

## AMENDMENT 4 (03:35 IST 10-15). Written before any V224-FF-S submit and before any 10-15 read. R7-teacher student entry
- **Entry:** V224-FF-S = `eeshsaxena/rsna-sub-v224ffs20` v1, notebook sha 26dae5c944022553 (BUILD_V224FS.json, builder pre_ao a54afad6).
  - Base: V224-FF20 (V224-FLIP + K14 fleet at w_f .20, time-aware flip).
  - F slot = .5 mean(K14, 20) + .5 mean(S, 5).
  - S = the 5 T_b student checkpoints (seed 1047; PREREG_TEACHER + A1 + A2), dataset eeshsaxena/rsna-knee-cnxt-fleet-r7tb, manifest bbf964bc.
  - Student gate ~/r7t_labels/gate/student_gate_20261015_0304.txt (sha 4d5ba176) VERDICT PASS.
  - Commit gate V224FFS_CHECK PASS 14/14 mode clean.
  - w_f .20 unchanged.
- **Runtime guard** (not read as a change): fleet start budget 5.5 -> 5.1 h (MIN_START_MIN 192); S costs +0.29 h central (0.31 s/study/GPU per member, MEASURED commit).
  - The time-aware flip budget stays 4.9 h, so the flip policy and cells 0-15 are byte-identical to V224-FF20.
  - This supersedes PLAN_TEACHER FX8 (see RUNTIME_FFS_1015 and fix F1; PREREG_10-15 Amendment 2).
- **Shared caveat**, recorded before the read:
  - The time-aware flip is a per-session lottery that FF-S shares with FF20 (flip >= 90 % of studies in ~74 % of sessions for both, MODEL), and the flip share is not observable after a hidden run.
  - So a non-moving read is a weak negative for S, not proof that S is negative.
  - The extra S stage can skip the fleet in a slow session (Synovitis canary, P ~.001-.002 MODEL), which shows on the LB as a canary entry, not as a read.
- **When it is submitted:** only under PREREG_10-15 Amendment 2 branch a (FF20 becomes the team best), read with the Amendment 3 R1 censoring rule.
  - One change vs its base. A win counts only as an API best-id move.
- **Follow-ups:**
  - If it wins: one REPLACE-form entry (F = S alone) on the then-best base, per PREREG_10-15 Amendment 3.
  - T_a gets at most one entry, same form, after the T_b read, only if it passes G0-G3' + G4.
- **Constraints:** S never enters pick A by itself; it enters a final pick only through a winning entry, as part of the F component. No per-label public tuning; no tuning of the .5 split or of w_f on the student line.
