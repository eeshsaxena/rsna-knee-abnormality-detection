# SILVER PLAN 10-15 .. 10-22 (written 01:55 IST 10-15, before any 10-15 read). No em dashes.
Goal (Eesh 01:30): a COMFORTABLE silver on the PRIVATE LB. Merges the 4 lenses of wf_ad00d12d-ae8 (forum, notebooks, own
assets + runtime, LB analytics). Tags: MEASURED / MODEL / GUESS. PREREG_10-15 Amendments 1-2 stay in force unchanged.

## 0. Position and what "comfortable" needs
- V224-FLIP (57174021): public .956, r296 / 6,009 at 20:05 UTC 10-14; 230 teams >= .957; .956 band 882 teams (MEASURED,
  lbprobe/api_lb_1015T0140.json). Drift at our spot ~4 ranks/h (MEASURED, frozen anchors). Losing public silver within
  ~1 h is expected and is NOT a decision signal.
- Deadline cut ~r305-310 (entrants lock tonight). It sits inside the .957 band at unrounded ~.9573-.9579, i.e. +4 to
  +10e-4 over FLIP (~.9566-.9569) (MODEL, silver_lb/sim_private.py).
- P(silver, private) by true offset over FLIP: +3e-4 .18, +4e-4 .47, +5e-4 .79, +6e-4 .94 (MODEL). Comfortable = ~+6e-4.
- Our scheduled edges (F, S, RM, J on FLIP) reach ~+2e-4 expected, P(silver) ~.21 (MODEL). So we need ONE new real
  ingredient, PLUS stacking of every read-positive edge, PLUS J in both picks. Honest odds now ~.2; with S6fall real ~.5-.6.

## 1. Levers, deduped and ranked (EV = private ranks x P(real) x feasibility)
| # | lever | evidence | gain | P(>0) | cost | verdict |
|---|---|---|---|---|---|---|
| 1 | S6F: nartaa S6fall leg, rank .8 base + .2 S6fall | .950 solo, all inputs public (MEASURED); no OAI; R7-family arch on Raptor-side labels; fold-0 OOF .907 vs CoAtNet .890 (CLAIMED 749359); no public blend uses it yet (MEASURED 20:05 UTC) | +3 to +10e-4 public (GUESS) | ~.6 GUESS | 1 build; +0.5 to 2.1 h hidden (MODEL; lenses disagree, runtime review decides) | BUILD NOW, top |
| 2 | Lock-day watch + same-day rebase | new public notebooks end 10-15 23:59 UTC = 05:29 IST 10-16 (MEASURED API); copy clumps K 300-650 | protects +100-250 ranks (MODEL) | release .3-.6 GUESS | watcher runs; 1 slot per rebase | RUN |
| 3 | J (DEC cell) + AO (flip always on) in every final candidate | J byte-identical on uncompressed (MEASURED); time-aware flip loses the flip silently in ~18 % of FF sessions (MODEL) | J +.03-.06 P(silver) (MODEL) | event .04-.11 GUESS | builder change + U-control | MANDATORY for picks |
| 4 | Student S (T_b, R7 teacher), Amendment 2 | gate pending; +0.29 h (MEASURED microbench) | +1 to +3e-4 (MODEL) | ~.55 GUESS | in flight | KEEP |
| 5 | K14 fleet F .20 on FLIP (FF20, 05:36) | F .20 on RR base moved +1 frozen team only (MEASURED) | +0.5e-4 sd 1.5e-4 (GUESS) | ~.6 | +1.8 h (MODEL) | KEEP; rivals S6fall for the 1.8 h |
| 6 | B + B' final picks, not copy floor + B | sim, 3 scenarios + sensitivity | +.02-.09 P(silver) (MODEL) | n/a | 0 | ADOPT (sec 4) |
| 7 | RM r224 mirror (FLRM, 06:25) | gate PASS 10/10 (MEASURED) | 0 to +1e-4 (MODEL) | ~.6 | +0.1 h | KEEP; fits anywhere |
| 8 | R7 crop-shift 3rd pass, time-aware | in-distribution: trained on random 352-of-384 crops (MEASURED code) | 0 to +2e-4 (GUESS) | ~.55 | +1.8-2.0 h, half-day build | CONTINGENCY only |
| 9 | R6 fe055406 swap or R6+R7 | heliosli v1 .957 used it (MEASURED); dataset v1 = HTTP 404; public mirrors hold 209a32c8; mizeroluckygall copy: 9/9 file sizes equal R7, download errors (MEASURED 01:50 IST) | ~+10e-4 if obtainable (selection-biased) | obtainable <.1 GUESS | 0 h | CLOSED unless heliosli re-publishes |
- Closed, no slots: S6 209a32c8 as 2nd ConvNeXt (dominated by S6fall; rjings .956), R9all leg (.956 MEASURED, no jump),
  GP reader (same family as F), quad-b arms (.955), box D2 / D3 / L13 / D-OAI / PL (measured nulls; PL f2 corrupt), extra
  TTA axes (R7 2nd axis out of distribution, r384 already mirror + K94, fleet 3rd pass), w_f tuning, per-label probing, seeds.
- Policy: OAI-light hedge DROPPED (host rulings 10-14 13:18 / 13:20 UTC, MEASURED); both picks may be OAI-inclusive;
  disclose weights + sources in any write-up. Geo question (LG22) still open, GUESS < 5 % risk, hits every public .956 equally.
- Team merge: Eesh ruled it out 10-08; merger deadline 05:29 IST 10-16. Only if he reopens it.

## 2. Build queue (agents build; main does every commit, dataset and submit)
- B1 S6F builder, start now, gated by ~17:00 IST 10-15.
  - Port ONLY the B path of nartaa/rsna-knee-s6fall-solo (prepare_inputs, probe_b_stack, run_b). Never its
    remove_submission(), its 8 h hang guard or its B-failure hard error.
  - Pin nartaa/rsna-knee-s6fall-weights v3, assert E_prior_s42_fall.pt sha 0c7297d5. Unpack s6_extras (compact_infer
    34f433fe, MONAI 0.6.0) on its own path and run it in a subprocess, so R7's wheels and code stay untouched.
  - Order: r384, r224 (+RM if base FLRM), R7 with flip FORCED, then S6fall behind a start guard (G from the runtime review)
    with a VISIBLE canary on skip. Blend per label: .8 rank(base) + .2 rank(S6fall); w is a prior, never fitted.
  - Two kernels (base FLIP, base FLRM). Each: 1 GPU commit, runtime review (p99 <= 8.0 h), gate (base rows identical with
    S6fall removed; blend recompute <= 1e-6), Amendment 3 written before submit.
- B2 AO + J overlay: relax the FX8 assert (build_v224fs / build_v224ff) and v224fs_check MIN_START so flip_budget can be 99;
  DEC cell (glorysing/rsna-knee-offline-codecs-r09) in every new build, U-control sha check per base. Build FLIP-J tonight.
- B3 watch.py: hourly until 05:30 IST 10-16, then every 4-6 h. Add the scored-version code diff (renderedOutputUrl) so bait
  releases (scored pin != current pin) get flagged; flag heliosli dataset bumps (fe055406 return) and S6fall public blends.
- B4 student line per RUNBOOK_STUDENT_1015 (gate after the rental fetch ~03:05); FF-S and FFSR20 both gated before reads.
- B5 contingency, only on trigger (neither FF20 nor S6F has become team best by the ~02:00 IST 10-16 read): R7 crop-shift
  pass, time-aware (drop if projected end > 6.0 h), patched after her sha asserts; sim, commit, gate.
- B6 final-candidate builds (10-19): B and B' as AO + J kernels from read-positive edges only; runtime review each.

## 3. Slot plan
UTC day D = 05:30 IST D to 05:29 IST D+1, 5 slots. Entries >= 45 min apart. Each = ONE change vs the team best at submit
time, read by API best-id (api_frozen.py). Reads land ~4.6 h (FLIP class) to ~6.5 h (FF / S6F class) after submit (MODEL).
| day | slot | IST | entry | condition / note |
|---|---|---|---|---|
| 10-15 | 1 | 05:36 | V224-FF20 (scheduled, time-aware flip) | read ~11:30-12:30 |
| 10-15 | 2 | 06:25 | V224-FLIP-RM (scheduled) | read ~11:00 |
| 10-15 | 3 | ~12:30-15:00 | student per Amendment 2 (a FF-S, b FFSR20, c HOLD, d STOP, e none) | on c or e: S6F takes this slot if gated |
| 10-15 | 4 | ~17:00-21:00 | S6F on the best of {FLIP, FLRM} (FLIP if FLRM unread) | only if B1 gated; else 10-16 slot 1 |
| 10-15 | 5 | until 04:30 10-16 | release rebase (flag -> swap_builder -> commit -> swap_check -> amendment) | else ~04:30: FLIP-J (or FLRM-J if best) = scored J floor |
| 10-16 | 1 | 05:35 | S6F if not in yet; else ladder step | module M fixed at the S6F read |
| 10-16 | 2-3 | after reads | ladder steps | |
| 10-16 | 4-5 | any | release / version reactions | else unused |
| 10-17..19 | <= 3/day | | ladder steps; FF-AO-J if M = K14; B5 entry if M = none | 2 slots/day held for reactions |
| 10-20 | 1-3 | weights frozen 12:00 IST | B and B' as AO + J, if not already scored in that form | |
| 10-21 | 1-2 | by 11:30 IST (06:00 UTC) | last risky candidate | after: only re-submits of a stuck / canaried candidate |
| 10-22 | - | | no new recipes; select 2 picks by 21:30 UTC (03:00 IST 10-23) | |
- Module M: S6fall if S6F becomes team best; K14 if FF20 is best and S6F does not beat it; none if neither moves the best.
  If both are positive (FF20 best, then S6F beats it), keep them apart: B carries S6fall, B' carries K14. Combine them in one
  kernel only if a runtime review gives p99 <= 8.0 h for the pair (MODEL today 7.6-8.5 h: no).
- Ladder rule (10-16 on): next entry = team best + the highest-priority edge not yet in it: M, then S, then RM. Every ladder
  entry is AO + J. An edge that fails to move the best is dropped; one retry only for K14 as FF-AO-J (FF20's time-aware
  flip confounds its read). A best-id move from a J-only change is rerun noise, not a read.
- Student on the FLRM base (branch c) needs its own amendment first; it then enters as a ladder step.
- Spare slots stay unused. No identical resubmits, no weight variants, no controls.

## 4. Final-pick rule (pre-registered now)
- Eligible: scored; J (DEC cell); flip AO or forced; no canary signature; every edge in it moved the team best at its own
  entry (flip and J count as MEASURED-mechanism edges).
- Pick 1 (B) = the eligible entry that is the API team best at 10-22 20:00 UTC.
- Pick 2 (B') = the most recent former team best among eligible entries that differs from B by at least one edge
  (preferred: B minus its weakest-read edge, or the other module).
- Pure copy floor (FLIP-J or V224-J) as pick 2 ONLY if the final LB shows the floor clump wholly inside the cut
  (teams above it + clump size <= ~260) or every edge of ours read negative.
- No weight tuning on public, no fishing for in-band position with identical recipes.

## 5. Guards
- Runtime: hidden 9 h limit, R7 wall 8.3 h. Nothing ships with runtime-review p99 > 8.0 h. Every heavy module sits behind
  a start guard with a VISIBLE canary on skip, so a silently degraded run can never become a pick.
- Every runtime MODEL rests on one FLIP latency sample (4.57-4.65 h): re-calibrate from each new submit-to-score time.
- Bait releases: build only from the scored version's pins (heliosli 0.9573 v1 -> v2 dataset swap is the precedent).
- Watchers die with the session: re-arm after any restart; check submits.log before any hand submit.
- Public gains regress on private (lambda .45-.75, MODEL): read-gated edges only, small fixed weights.
- Stuck runs (748403) and ~1 h T4x2 queues (748575, CLAIMED): all candidates in by 10-21 11:30 IST.
