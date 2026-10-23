# Verification of the "box-schedule" result (box_schedule.md, sched_binormal.*)

Written 2026-10-04 ~15:35 IST by the adversarial verifier. Read-only on Kaggle and on the box (only `ls`, `cat`,
`grep`, `sed`, `ps`, `nvidia-smi`, `find -newermt` on the box; nothing launched there). Laptop: re-ran the tiny
closed-form script only. Tags: MEASURED (I read or computed it, path given), MODEL, GUESS.

## Bottom line

- The central timing finding holds. Sharing the V100 buys no throughput: aggregate 0.93-1.01x of one job alone
  (MEASURED, rounding-limited), so "about 0.96x" is inside the noise but "no gain" is solid.
- TTR's real cycle is 29.0 min/epoch and it ends ~22:45 with gated running (MEASURED). The ~19:10 finish if gated
  stops is MODEL (the findings list tags it MEASURED; that tag is wrong).
- Every log number I checked matches the logs exactly. The binormal script reproduces byte-for-byte.
- No hard-rule violation found: nothing under `~/bigjump_build` is newer than 14:55, the probe wrote only to
  `~/k2_test/analysis_wf/sched/tiny`, and PID 267280 really is the gated run.
- Weak spots:
  1. "Halves the time to every gate read" is an overclaim.
  2. The S1-keyed label switch to v4 has no direct support.
  3. The .943-.947 expectation skips the model's own .002-.004 haircut.
  4. The "slow session" reading of v9crop25 is plausible but unproven.

## 1. Timing claims (re-derived from the logs)

| Claim | My check | Verdict |
|---|---|---|
| F1 alone 1.08-1.09 s/it | `~/fleet_runs/f0_s42_v4.log` ep 9 (0-based). Its 600-800: (800 x 1.96 - 600 x 2.25)/200 = 1.09, but printed 2-decimal averages give a range of 1.055-1.125. Its 1600-1733: TRAIN-ONLY 54.1 min vs 1.94 x 1600 gives 1.07 (range 0.98-1.14). Timeline: ep 9 t0 = 10:12:41 (EMA ckpt) - 58.1 min = ~09:14:35, so it 600 is ~09:37 and it 800 ~09:40:40. mix50 ended 09:37:45 (`f0_s42/` EMA ckpt). Parent `kaggle/` dir created 09:40:19, `localLB_parent_preds.csv` 10:04:18. `~/k2_test/runs` jobs ran 04:18 and 11:15-11:29, outside both windows. The first window may still hold ~40 s of mix50's final eval, so the true alone rate may be nearer ~1.0. | CONFIRMED (MEASURED, +/-5%) |
| Shared F1 2.22-2.25 s/it | Steady 2.22-2.26 in all F1 logs; v4 ep 8 reached 2.29-2.34; gated 2.20-2.23 next to TTR | CONFIRMED (the range is slightly wider: 2.20-2.34) |
| Aggregate 0.96x | (2/2.24)/(1/1.09) = 0.97; with the rounding ranges, 0.93-1.01x. Eval corroborates: one eval tag takes 2.0 min alone (v4 ep 9) vs 4.4-4.6 min shared, a 2.2-2.3x slowdown | CONFIRMED: no gain; the 4% loss is within noise |
| TTR alone ~0.42 s/it; sharing with TTR ~1.0x | No alone window exists in `f0_s1042_ttr.log`: both jobs started 11:09:38/43, and ep 0 is a warm-up. The 1.0x is assumed, not measured (it follows from F1's 49% share). Independent check: TTR compute per iteration ~ (16/24) x (224/320)^2 = 0.33 of F1, so if GPU-bound it runs ~0.36 s/it alone (MODEL). CPU load avg 3.9 on 40 cores, so not CPU-bound. | UNCERTAIN (plausible, 0.36-0.55) |
| TTR 29.0 min/epoch incl. evals | EMA-line epoch times 30.1, 28.6, 28.7, 28.7, 28.8, 28.9, 28.9, 29.0. 8 epochs from 11:09:38 to 15:01:30 (EMA ckpt) = 29.0 min mean | CONFIRMED (MEASURED) |
| Ends ~22:45 if gated runs | 15:01:30 + 16 x 29.0 min = 22:45 | CONFIRMED (MEASURED-rate projection) |
| Ends ~19:10 if gated stops ~15:30 | TTR's 9th epoch closes ~15:30:30, then 15 x ~14.7 min. At 0.55 s/it it is ~20:05 | MODEL, not MEASURED (mis-tagged in the findings list) |
| Gated 73 min/epoch, ends ~23:05, ~4.1 alone-h left | EMA lines 73.9 / 72.8 / 73.2. At 22:45 it has done ~9.5 of 10 epochs, the rest alone. Remaining at 15:10 is ~6.6 epochs x 35.3 = ~3.9 h | CONFIRMED (approx.) |
| Ep 3 (1-based) gated = v4 on val .8160/.8160; gold58 .8243 vs .8115; mix50 .8567 | `f0_s42_gated.log`, `f0_s42_v4.log`, `f0_s42.log` ep 2 (0-based) raw lines | CONFIRMED. box_schedule.md verdict line 2 says "epoch 2": an indexing slip |
| CPU probe ~10 s/study | `~/k2_test/analysis_wf/sched/tiny/tta_summary.json`: 2 val + 2 gold58, 0.67 min including model load; dir 15:02:08 -> files 15:02:49 | CONFIRMED |

## 2. Strength and correlation claims

- mix50 vs v4 on v4 labels: CONFIRMED (MEASURED).
  - Raw best .8811 (ep 8, 1-based) vs .8692 (ep 8); closing JSON .8811/.8692; EMA .8770/.8650.
  - Both raw peaks are at epoch 8, but epochs 8-10 differ by <= .0023, so the fixed-epoch choice is fine.
  - Caveat: this measures OOF agreement with LLM labels, not hidden-test labels.
- rho_raptorproxy, mix50 .889 vs v4 .840: CONFIRMED (MEASURED, `~/fleet_runs/gate_history.log` "== 10:13" block, raw rows).
- `fleet_train.py` supports the all-data plan: CONFIRMED (MEASURED, code).
  - `--stop_epoch` exists (line 73) and keeps the `--epochs` cosine.
  - All-data mode logs gold58 for raw and EMA every epoch (lines 377-389) when `--gold` is given.
  - The default `--epochs` is 12 (line 72).

## 3. Model claims

- `sched_binormal.py` re-run on the laptop gives output identical to `sched_binormal.out` (MEASURED diff).
  - The formulas are the standard equal-variance binormal blend and k-member average.
  - Every table in box_schedule.md sec 3.3, 3.5, 5.1 and 5.3 matches the output.
- Inversion sensitivity the doc omits (MODEL, my run):
  - A +/-.0005 rounding of the parent's .943 moves the implied rho_eff by about +/-.01 at W .30. For example,
    S1 .920 with B30 .942 gives .860 / .869 / .878 at AP .9425 / .943 / .9435.
  - That is small next to the +/-.025 from rounding of the read itself.
- Crop-leg inversion "~.9 or more": CONFIRMED (MODEL). With S .924, crop15 .942 and crop25 .941 invert to
  rho_eff .89-.97 across the rounding of the reads and of the parent.
- Serving numbers: CONFIRMED (MEASURED, `SP/k2/build/out_k2v1/fleet_receipt.json`).
  - Bench 295.61 / 304.05 ms per member.
  - decode_s_per_study_per_thread 1.48.
  - WALL_H 8.6.
- Capacity table: the arithmetic is CONFIRMED (MODEL).
  - The "6.0 h MEASURED turnaround" row is an upper bound on P from one session.
  - ENDGAME_V3 line 35 infers P = 7.7-8.05 h from timeouts. The realistic rows are therefore ~8-14 F1-equivalents,
    and at 8.05 h with slow decode the margin is about 0.
- Box slot arithmetic: CONFIRMED (156 / 232 / 350 alone-hours from 10-04 19:45). It assumes 100% uptime and
  flawless chaining (optimistic). The md verdict says "~41 by 10-15", while its table says ~38 by 10-14 12:00.

## 4. Kaggle facts

- Read with `kaggle competitions submissions` (read-only, ~15:25 IST).
  - 56816035 (K2 solo) and 56816039 (K2 blend) are PENDING.
  - 56799619 v9crop25 is COMPLETE with an empty publicScore (MEASURED).
- The CLI does not expose an error description, so "timeout" is an inference.
  - Support for it: HANDOFF.md line 36 says v9crop25 was still running at 03:40 IST, after its batch-mates had scored.
  - It is not new evidence today: it was submitted 10-03 19:47 IST.
  - The two halves scored separately (v9 .943, crop25 .941). That fits a parent runtime near the wall, or a bug
    in the combination.

## 5. Problems found (refutations or corrections)

1. **"Running jobs one at a time halves the time to every gate read": overclaim.**
   - Two jobs sharing both finish at 2T. Run back to back, they finish at T and 2T.
   - So every other read comes earlier and the rest come at the same time: a mean latency cut of ~25% (MODEL), not 50%.
   - For tonight specifically it is right that TTR's G-T2 moves ~3.6 h earlier, because gated is dropped, not deferred.
2. **Stopping gated is lower-stakes than presented.**
   - The doc's own conclusion is that box capacity is not binding (25+ slots vs an ~8-member serving cap). So the
     only gain is decision latency.
   - It discards ~2.0 alone-hours of completed gated work (3.4 epochs x 35.3 min, MODEL).
   - The G-L answer it would give is instrument-biased: val vs v4 favours v4-like targets, and gold58 favours
     teacher-mixed targets because the teacher was gold58-picked. So the answer would be weak anyway.
   - I agree the stop is reasonable. It is not a big lever either way.
3. **The S1-keyed label switch has no direct support.**
   - K2 solo measures the mix50 member only. A low S1 does not show that v4 is better: mix50 beats v4 on the
     v4-labelled OOF (+.012, MEASURED), an instrument tilted toward v4.
   - v4's only edge is decorrelation, and the doc's own model says that is worth ~0 at S <= .925.
   - Any switch should be keyed to a direct v4 measurement: a v4 solo in K3, or the gold58 rho/OOS-blend CPU job.
   - Also, the submitter's own pre-registration for K2 solo is ".905-.915" (Kaggle description of 56816035). The
     doc's central .918 is more optimistic (GUESS vs GUESS).
4. **The expected public value skips the doc's own haircut.**
   - The doc says the binormal model over-predicts lift by .002-.004. Yet sec 9 adds the raw model lifts
     (+.001 to +.004 PASS).
   - With the haircut, PASS is ~+.000 to +.002, so public is ~.943-.945 (MODEL on GUESS inputs).
   - P(>= .945) is more like 10-20% than 30-40% (GUESS), given that every member add so far read <= the parent
     (crop .942/.941/.940, K1 .942, cnx .930-.935, MEASURED).
5. **Tagging.** The ~19:10 TTR finish is listed as MEASURED in findings item 3. It is MODEL. The md body tags it
   correctly.

## 6. Rule and deadline checks

- The agent stayed read-only on Kaggle. It did not kill anything or touch `~/bigjump_build` (MEASURED, no file newer
  than 14:55). It wrote only under `~/k2_test/analysis_wf/sched/`.
- The stop command is for Eesh to run. It checks `ps -o args=` for `labels_gated75_logit` on PID 267280, which
  matches (MEASURED, `ps`).
  - It appends one line to `~/fleet_runs/gate_history.log`, which is fine as Eesh's own action.
  - After the stop, check `nvidia-smi`: the persistent DataLoader workers should exit with the parent.
- The optional CPU gold58 job is allowed by the hard rules: CPU-only, nice 19, output under `analysis_wf`. It was
  blocked only by the session permission system. If it runs:
  - set `OMP_NUM_THREADS` (e.g. 8) so it does not starve the training loaders;
  - set `PYTHONDONTWRITEBYTECODE=1` so importing from `~/bigjump_build/code` writes no `__pycache__` there.
- Deadline mapping is consistent: merge 10-15 23:59 UTC = 10-16 05:29 IST; final 10-22 with picks 10-21; ~6 h read latency.
