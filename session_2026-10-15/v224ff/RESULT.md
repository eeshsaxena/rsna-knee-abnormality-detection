# V224-F-FLIP (V224-F + time-aware R7 flip TTA), built and commit-tested 10-14 by a Claude subagent. NOT submitted.

No em dashes. Tags: MEASURED / MODEL / GUESS as in RESEARCH_960.

## Result in one block
| kernel | slug / version | gate | notebook sha256 |
|---|---|---|---|
| **V224-FF20** (submit candidate, w_f .20) | `eeshsaxena/rsna-sub-v224ff20` v1 | **V224FF_CHECK PASS 14/14** (nested: v224f_check 26/27 with only its exact-mode sub-gate failing by design, replaced by v224_check --flip 21/21; h_check hf 16/16) | b86f8b9f12c1d573... |
| control, forced time pressure (never submit) | `eeshsaxena/rsna-sub-v224ffctl` v1 | **V224FF_CHECK PASS 18/18** (control mode; same nested gates) | 56de857fe8a10dee... |
| negative tests | 11 cases (cross-mode, wrong w_f, wrong budget, 5 tampered outputs) | **NEG PASS** (`gate_negative_tests.txt`) | |
| driver offline tests | 10 scenarios, stub worker + her pinned probe_dual.py | **TA_TEST PASS 75/75** (`test_ta_driver.txt`) | |

- **Clean commit (3 studies): flip active on every study.** Both R7 workers kept flip (decision none, 3 flipped / 0 plain). The R7
  output `ours_sub.csv` equals the V224-FLIP commit's R7 output exactly (max diff 0) and `submission_V224.csv` is byte-identical to
  the V224-FLIP commit's submission. Final submission == (1 - .2) pct(V224 recipe recomputed from raptor_raw / raptor224_sub /
  ours_sub) + .2 pct(fleet member mean recomputed from the worker npz): max diff 2.5e-8 (float32). MEASURED.
- **Control (budget forced to notebook age + 1 min via env `V224FF_FLIP_BUDGET_H`): the flip disabled itself cleanly.** Worker 0
  (2 studies) printed `FLIP_TTA_DISABLED_AT_STUDY 1` (lower-bound rule, decided at 0.107 h vs budget 0.035 h), its flip child was
  killed (rc -9), no torn line, the plain child (pinned compact_infer.py without --flip-tta) scored study 2 (rc 0); worker 1 (1 study)
  had nothing left to switch. R7 PASS 3/3, 0 fallback. Row by row: the 2 flipped studies equal the V224-FLIP commit's R7 rows and the
  plain study equals the V224-exact commit's R7 row (max diff 0 both), and the plain row is not the flip row (logit diff 1.19). The
  fleet ran clean (20 members) and the final blend recompute matches (2.5e-8). MEASURED.
- **Cost of the driver:** clean R7 stage 340 s (V224-FLIP 370 s, V224-exact 350 s on their commits): no measurable overhead. A switch
  costs one restart of the pinned worker: 145-150 s on the commit (model load + the compile check; the second compile was ~100 s
  faster thanks to the inductor cache). MEASURED.
- GPU quota used: 2 commits, 585 s and 736 s notebook time (~10 and ~13 min with setup). Never more than 2 of our sessions at
  once (the push loop retries only while Kaggle answers "Maximum batch GPU session count of 2 reached"). Nothing submitted.

## 1. What was built (`build_v224ff.py`, one flag per variant)
`python session_2026-10-15/v224ff/build_v224ff.py --wf 0.20` writes `kernel_v224ff20/` (slug `eeshsaxena/rsna-sub-v224ff{WF*100}`);
`--control` writes `kernel_v224ffctl/`. The builder RUNS the V224-F builder (`session_2026-10-15/v224f/build_v224f.py`, sha
197a0364...) with the same `--wf / --start-budget-h / --wall-h` into `base/` and asserts the result is byte-identical to V224-F's own
commit-tested kernel (`v224f/kernel_v224f20`, sha 8f8b38e4..., V224F_CHECK PASS 28/28). All 22 V224-F cells are kept byte-identical and
in order (proof: dropping the inserted cells re-serialises to the V224-F file byte for byte). Every original sha assert is untouched
(her r384 / r224 / R7 CODE_SHA + SOURCE_SHA incl. compact_infer 8401812c / R7 bundle 8c7b666e; V224-F's fleet asserts).
Inserted (full text in `celldiff_v224ff20.txt`, `celldiff_v224ffctl.txt`):

| cell | what |
|---|---|
| 7 FLIP-TA (between her R7 assets cell 6 and R7 run cell 7) | V224-FLIP's asserts (compact_infer sha 8401812c with `--flip-tta` + FLIP_TTA_READY; manifest flip_tta off; no `--flip-tta` in WORKER_ARGS; timm backbone; also WORKER == compact_infer.py and the pinned file on disk re-hashed). Writes `compact_infer_ta.py` (this dir, sha 9cbfe37f...) next to her pinned sources and asserts its sha; `manifest['flip_tta'] = True` in memory; `WORKER_ARGS += ['--flip-tta']` (V224-FLIP's flag) + the driver args (`--ta-budget-end NB_T0 + budget`, `--ta-nb-t0`, `--ta-min-studies 10`, `--ta-switch-min-s 600`, `--ta-safety 1.05`); `WORKER = 'compact_infer_ta.py'`; `FLIP_TTA_FORCED_ON = True` (read by the V224 canary receipt). Prints `FLIP_TTA_FORCED` (driver, args, budget, budget source, elapsed). Budget = env `V224FF_FLIP_BUDGET_H` if set, else 4.9 h. |
| 9 REPORT (after her R7 run cell 7) | reads `pred/worker*/receipt.json` -> `time_aware_flip`, prints one `FLIP_TTA_DISABLED_AT_STUDY k {...}` line per worker that switched and `FLIP_TA_REPORT {flipped, plain, decisions}`, writes `/kaggle/working/flip_ta_receipt.json`. Never raises. |
| 7 CTL (control only, before FLIP-TA) | `os.environ['V224FF_FLIP_BUDGET_H'] = notebook age + 1 min`; prints `CONTROL_FLIP_BUDGET_H`. |

Her cell 7 is unchanged: probe_dual.py (pinned) launches `compact_infer_ta.py` per GPU through her `WORKER` variable.

### The driver (`compact_infer_ta.py`, stdlib only, not one of her pinned files)
- Phase F: runs the PINNED compact_infer.py (sha re-checked) with the exact V224-FLIP worker command (the `--ta-*` args are stripped),
  in its own process group. predictions.csv is flushed per study, so the driver sees each finished study (complete lines only).
- Rule (per worker, after every study k): rate = (t_k - t_1) / DICOM files of studies 2..k (study 1 carries the ~4 min compile check);
  projected end = t_k + 1.05 x rate x files left (k >= 10), or t_k (k < 10, a lower bound). Projected end > budget (and, for k >= 10,
  remaining flip work > 600 s, so a restart is worth it) -> **flip disabled for the remaining studies**: the child's process group is
  killed (SIGKILL), its complete rows kept (a torn last line is dropped and that study re-scored), `FLIP_TTA_DISABLED_AT_STUDY k` printed,
  and the pinned compact_infer.py WITHOUT `--flip-tta` (= the V224-exact worker) scores the rest into `<wdir>/ta_plain` (same R7
  deadline). The driver then writes `<wdir>/predictions.csv` (flip rows + plain rows, the worker's own bytes) and `receipt.json` (merged
  counts, `flip_tta` block for the flipped part, `time_aware_flip` block).
- Budget already passed when a worker starts -> no flip child at all (`FLIP_TTA_DISABLED_AT_STUDY 0`). Phase-F child exits non-zero on
  its own -> `FLIP_TTA_CHILD_FAILED`, the unscored studies go to the plain worker (her notebook would have given them Raptor ranks).
  Any driver setup error -> `FLIP_TA_DRIVER_ERROR` and the plain V224-FLIP worker command runs (fail-safe = V224-FLIP).
- No switch (the expected case): the child wrote predictions.csv / receipt.json exactly as V224-FLIP's worker; the driver only adds the
  `time_aware_flip` block. Every path is logged in the worker log, in the notebook log (REPORT cell) and in `flip_ta_receipt.json`.
- **Why 4.9 h, not 5.5 h:** V224-F's fleet starts only if the kernel is <= 5.5 h old (its own guard; else HF_FALLBACK + the Synovitis
  canary). R7 ends before r224 (~0.3 h hidden, MODEL; x1.3 slow ~0.4 h), stage 3 and the fleet pre-cells, so the flip budget is
  start budget - 0.6 h (builder flag `--flip-budget-h`). With the projection-with-flip rule, R7 ends <= the budget in the smooth case.

## 2. Gate output (`v224ff_check.py`; files `gate_v224ff20_v1.txt`, `gate_v224ffctl_v1.txt`, `gate_negative_tests.txt`)
```
python session_2026-10-15/v224ff/v224ff_check.py session_2026-10-15/v224ff/out_v224ff20_v1 --wf 0.20 \
   --ref-flip session_2026-10-14/v224/out_v224flip_v1 --ref-exact session_2026-10-14/v224/out_v224_v1 \
   --ref-fleet session_2026-10-13/hr7rr/out_hfr7rr_v1                      (add --control for the control output)
```
Clean (excerpt):
```
  OK   v224f_check.py: every check passes except its exact-mode V224 sub-gate (replaced ...) {'fails': [...exact...], 'oks': 26}
  OK   V224 part (log up to V224_DONE, submission_V224.csv) passes v224_check.py --flip --ref V224-exact      (21/21)
  OK   FLIP_TTA_FORCED: driver compact_infer_ta.py (sha), --flip-tta + driver args, budget 4.9 (built in)
  OK   markers in cell order (ASSETS_READY < FLIP_TTA_FORCED < OURS_STAGE_COMPLETE < FLIP_TA_REPORT < R224_STAGE < V224_DONE)
  OK   flip_ta_receipt.json: 2 R7 workers ... {'budget_h': 4.9, 'flipped': 3, 'plain': 0, 'decisions': ['none'], 'all_flipped': True}
  OK   clean: R7 ours_sub == the V224-FLIP commit R7 (0.001) max diff 0
  OK   clean: submission_V224.csv == the V224-FLIP commit submission (1e-06) max diff 0, byte-identical True
  OK   END-TO-END: submission.csv == (1 - w) pct(V224 recipe) + w pct(fleet npz member mean), 20 members max diff 2.54e-08
V224FF_CHECK PASS 14/14 mode clean
```
Control (excerpt):
```
  OK   control: exactly one worker switched at study 1 by the lower-bound rule, killed (-9), plain child rc 0
       {'decided_at_study': 1, 'flipped': 1, 'plain': 1, 'studies': 2, 'kill_rc': -9, 'plain_rc': 0, 'torn_last_line_dropped': False}
  OK   control: FLIP_TTA_DISABLED_AT_STUDY 1 in the notebook log and in the switching worker log, FLIP_TA_PLAIN_START there
  OK   control: flipped rows == the V224-FLIP commit R7 rows (0.001) {'276191': 0.0, '591402': 0.0}
  OK   control: plain rows == the V224-exact commit R7 rows (0.001) {'991776': 0.0}
  OK   control: the plain row is NOT the flip row (some R7 column differs by > 1e-3 in logit) {'991776': 1.1888}
  OK   END-TO-END: submission.csv == ... max diff 2.54e-08
V224FF_CHECK PASS 18/18 mode control
```
Negative tests (all fail as they must): control output with the clean gate, clean output with the control gate, wrong w_f .25, wrong
budget 5.5, submission cell +.01, submission_V224 cell +.01, R7 ours_sub cell +.01, a worker marked switched, control with the plain R7
row replaced by the flip row. Offline driver tests (`test_ta_driver.py`, stub worker speaking compact_infer's file protocol, on
Windows with taskkill in place of killpg): no switch; lower-bound switch at study 1; measured-rate switch at study 4; restart not
worth it -> no switch; prestart (study 0); child crash after 5 studies; torn last line; missing driver args -> fail-safe; R7 deadline
inside phase F; end to end through her pinned probe_dual.py. 75/75.

## 3. Runtime (hidden, MODEL; `sim_ta.py` -> `sim_ta.txt`)
- Inputs: V224-exact hidden 3.0 h central (2.4-3.6, 10-14 runtime review), flip +1.0 / +1.4 / +2.0 h on R7, V224-F fleet 1.80 h
  central / 2.07 h slow (V224-F wall model), session speed x0.9-x1.3, 640 studies per T4, restart 250 s.
- **Central (x1.0, +1.4 h): flip on 100 % of studies, R7 ends ~4.1 h, fleet starts ~4.5 h, notebook ends ~6.3 h.**
- Over the whole grid the time-aware kernel never skips the fleet (latest fleet end 7.0 h, x1.15 +1.0 h; fleet wall 8.3 h, hard 9 h).
  An always-on flip (V224-FLIP + fleet) skips the fleet (Synovitis canary, about -.03 display) in 3 of 12 cells (x1.15 +2.0, x1.3
  +1.4, x1.3 +2.0).
- Price of the rule: in borderline-slow cells it drops flip although an always-on flip would still have fit (x1.0 +2.0 h: 12-32 %
  flipped; x1.15 +1.4 h: 15-29 %), because it projects the end WITH flip (the task's rule). An alternative "latest switch" rule (keep
  flip while the remainder without flip still fits, using a 0.75 plain/flip prior) keeps 100 % there, but can overshoot the budget by
  ~0.1 h if R7 is decode-bound (ratio ~1); not used (appendix of `sim_ta.txt`). Raising `--flip-budget-h` to 5.0 is the other knob.

## 4. Submit (main session only; the workflow never submits)
```
python session_2026-10-14/submit_at.py eeshsaxena/rsna-sub-v224ff20 1 "V224-F-FLIP w_f .20: V224-F + R7 flip TTA, time-aware (budget 4.9 h)"
```
Other w_f (one flag, same gates): `python session_2026-10-15/v224ff/build_v224ff.py --wf 0.25` (or 0.30) -> `kernel_v224ff25/`
(already built, NOT commit-tested: sha 3ba4e99f..., v224ff30 2215fea0...; they differ from v224ff20 only in V224-F's two w_f / tag
lines) -> `python session_2026-10-15/v224ff/push_wait_fetch.py session_2026-10-15/v224ff/kernel_v224ff25 eeshsaxena/rsna-sub-v224ff25
session_2026-10-15/v224ff/out_v224ff25_v1` -> `v224ff_check.py out_v224ff25_v1 --wf 0.25 --ref-flip ... --ref-exact ... --ref-fleet ...`
-> `submit_at.py eeshsaxena/rsna-sub-v224ff25 1 "..."`. If V224-F's builder changes after its reviews, rebuild: the builder asserts
the base equals V224-F's kernel dir, and the gate must be re-run on a fresh commit.

## 5. Files (this dir)
`compact_infer_ta.py` (driver), `build_v224ff.py`, `v224ff_check.py`, `test_ta_driver.py` (+ `test_ta_driver.txt`), `neg_tests.py`
(+ `gate_negative_tests.txt`), `sim_ta.py` (+ `sim_ta.txt`), `push_wait_fetch.py`, `kernel_v224ff20/`, `kernel_v224ffctl/`,
`kernel_v224ff25/`, `kernel_v224ff30/`, `base/` (V224-F rebuilt by its own builder), `celldiff_*.txt`, `out_v224ff20_v1/`,
`out_v224ffctl_v1/` (+ logs), `gate_v224ff20_v1.txt`, `gate_v224ffctl_v1.txt`.

## 6. Concerns
1. The hidden run's flip decision is not observable from the leaderboard (logs are not returned), so a V224-FF read mixes "flip
   helps" with "how much of the run was flipped"; in a slow session it can be ~2 % flipped (= V224-F plus noise).
2. In borderline sessions the rule mixes flipped and plain R7 studies inside one ranking (flip-averaged logits are slightly less
   extreme); MODEL effect small, but it is not the pure V224-FLIP mechanism.
3. The measured-rate branch (k >= 10) cannot run on a 3-study commit; it is tested offline only (stub tests C, D, G). The commit
   control exercised the lower-bound branch, the POSIX kill, the plain restart and the merge.
4. The 4.9 h budget rests on a MODEL r224 hidden time (~0.3 h); if r224 is much slower, the fleet guard (5.5 h) could still be missed
   in a slow session (V224-F's own risk as well).
5. The gate depends on the exact name of v224f_check's exact-mode check (string match); if V224-F's gate is edited, re-run and
   re-check. `hf_receipt.json` keeps V224-F's tag (`v224f20`) because its HF cell is verbatim.
