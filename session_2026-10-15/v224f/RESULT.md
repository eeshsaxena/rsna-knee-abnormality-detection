# V224-F (V224-exact + our K14 R fleet, HF cell after stage 3), built and commit-tested 10-14 by a Claude subagent. NOT submitted.

No em dashes. Tags: MEASURED / MODEL / GUESS as in RESEARCH_960.

## Result in one block
| kernel | slug / version | gate | notebook sha256 |
|---|---|---|---|
| **V224-F20** (submit candidate, w_f .20) | `eeshsaxena/rsna-sub-v224f20` v1 | **V224F_CHECK PASS 28/28** (includes v224_check exact 17/17 on the V224 part + h_check hf 16/16) | 8f8b38e4e97c70b7... |
| control (never submit) | `eeshsaxena/rsna-sub-v224fctl` v1 | **V224F_CHECK PASS 21/21** (control mode; v224_check 17/17 + h_check hv) | a883e1046113... |

- The V224 part of V224-F20 is the V224-exact output **byte for byte** (`submission_V224.csv` == `out_v224_v1/submission.csv`,
  which itself equals varshitha's own commit, 0/36 cells). MEASURED.
- The fleet solo (20 members, K14 R) is **byte-identical** to the HFR7-RR commit's fleet solo on the same 3 studies, although
  here it runs after R7's `pip install` of timm 1.0.25 / pydicom 3.0.1 (HFR7-RR ran it on the stock timm 1.0.29 / pydicom 3.0.2).
  DICOM re-encode parity (shard 0, 4 train studies) frac_diff 0, maxdiff 0; decode probe 80/80. MEASURED.
- Final submission == (1 - .2) pct(V224) + .2 pct(F), recomputed independently: max diff 3.3e-8 (float32). MEASURED.
- Control: the fleet's own time guard refused to start ("only 490.2 min before the wall (< 600.0 min): stage not started"),
  HF_FALLBACK + CANARY_FIRED, Synovitis = 0.5 in every row, the other 11 columns string-identical to the V224 output. MEASURED.
- Hidden wall (MODEL from MEASURED commit microbench): **central 4.8 h** (V224 3.0 + fleet 1.8), range 4.1-5.7 h,
  slow session x1.3 on the slow end 7.4 h; fleet start budget 5.5 h, fleet wall 8.3 h (sec 4).
- GPU quota used: 2 commits (586 s and 464 s notebook time, about 10-11 min each with setup). Both COMPLETE, nothing else pushed.

## 1. What was built (`build_v224f.py`, one flag per variant)
`python session_2026-10-15/v224f/build_v224f.py --wf 0.20` writes `kernel_v224f20/` (slug `eeshsaxena/rsna-sub-v224f{WF*100}`).
`--control` writes `kernel_v224fctl/`. Builds for `--wf 0.25` / `--wf 0.30` were made in the scratchpad: they differ from v224f20
ONLY in 2 lines (cell 14 `_V224F` tag/w_f, cell 21 `W_F` / `HF_TAG`); sha256 v224f25 0a1bfec643ab..., v224f30 615f42b8c4d0...

Inputs, each sha-asserted by the builder: V224-exact notebook (4259ec97...), varshitha's notebook copy (6406e923...), HFR7-RR
notebook (ca3da2fb..., the scored team-best entry whose fleet cells we reuse). Every original sha assert inside the cells is
untouched (her r384 / r224 / R7 asserts; the K12-MERGE manifest + full sha256 of the 5 D-OAI checkpoints; the fleet worker's
model_sha / contract / parity checks).

Cells 0-13 = V224-exact, **byte-identical** (builder proof: dropping cells 14-21 re-serialises to the V224-exact file byte for
byte; dropping the canary as well gives her file). Appended (full text in `celldiff_v224f20.txt`, `celldiff_v224fctl.txt`):

| cell | what | origin |
|---|---|---|
| 14 SAVE | submission.csv (V224 output incl. the r224 canary) -> `submission_V224.csv`, sha256; prints `V224F_SAVED` (elapsed, budget decision) | new |
| 15 ENV-SNAP | environment snapshot | HFR7-RR cell 7, verbatim |
| 16 FN-IDS | competition root, ids, env defaults; writes an id placeholder submission.csv and sets T0 = now | HFR7-RR cell 8, verbatim |
| 17 K12-MERGE | links k11w (20 base) + k14w (5 D-OAI) into /tmp/k12_fleet, manifests + sha256 verified | HFR7-RR cell 9, verbatim |
| 18 CLOCK | `T0 = min(T0, NB_T0)`: the fleet wall is min(/proc start, T0) + WALL_H; without this a missing /proc clock would anchor the wall at FN-IDS (hours late) | new |
| 19 FLEET | K2 fleet stage, K14 R 20 members, TTA AB | HFR7-RR cell 10; only line 1 (K2_CFG literal): `WALL_H` 8.6 -> 8.3, `MIN_START_MIN` 20 -> 168 (control 600) |
| 20 ENV-RESTORE | environment restore, prints HF_ENV_RESTORED | HFR7-RR cell 11, verbatim |
| 21 HF V224F | the HFR7-RR HF blend on H = submission_V224.csv; canaries; receipts | new (HF semantics kept, see below) |

**Placement and time budget.** The fleet runs after R7, r224, stage 3 and the canary, so it cannot delay R7 (8.3 h deadline) or
the r224 guard (skip after 7.83 h). Its budget is the fleet cell's own tested P3 guard, set through the K2_CFG literal:
the stage starts only if at least MIN_START_MIN = (8.3 - 5.5) x 60 = 168 min remain to the 8.3 h wall, i.e. only if the kernel is
<= 5.5 h old; workers start no new chunk after 8.3 h and are killed 300 s later (8.38 h); a stage that is not clean writes no
fleet solo (fail-loud set). The SAVE cell logs the same decision (`fleet_may_start`). The 5.5 h start budget is chosen so that a
fleet that starts can finish before the wall even at the slow-session fleet time (2.7 h, sec 4).

**HF cell semantics** (same as HFR7-RR): per label, average-tie percentile rank, (1 - W_F) pct(H) + W_F pct(F), float32, not
re-ranked; F = `submission_fleetsolo.csv` (raw-prob member mean); `submission_H.csv` = the H used; `hf_receipt.json` with the same
keys (mode HF / H_only). Differences, all on the safe side: H is read from `submission_V224.csv` (sha checked against the save;
her in-memory `sub` if the file is unusable, logged `V224F_H_FROM_MEMORY`) instead of submission.csv (FN-IDS overwrites
that); the fleet must be `_k2_receipt['clean'] is True` with exactly 20 members (HFR7-RR only required the solo file).

**Canaries (every fallback visible).**
- Fleet not used for ANY reason (time guard, worker failure / parity / coverage gate, wall, missing or inconsistent files) ->
  `HF_FALLBACK` + `CANARY_FIRED fleet not used`, submission.csv = V224 output with **Synovitis = 0.5** for every study.
- r224 not used (V224 canary, unchanged): Fracture = 0.5. If the fleet then blends, Fracture is set back to 0.5 after the blend,
  so that canary stays visible (offline sim). Both can fire together (Fracture and Synovitis 0.5).
- R7 per-study fallbacks: her BLEND_FALLBACK log line, as in V224-exact (not canaried; exists in her runs too).
- Receipts: `canary_receipt.json` (V224 part, byte-identical cell; it says use_f false because that cell is V224-exact's),
  `hf_receipt.json`, `v224f_receipt.json` (use_f true, mode, fired, blanked, save record).

## 2. Gates (all outputs in this dir)
`v224f_check.py OUT --wf W [--control] --ref-v224 <V224-exact commit out> --ref-fleet <HFR7-RR commit out>`:
(1) runs `session_2026-10-14/v224/v224_check.py` (exact mode) on a view of OUT: log cut after V224_DONE, submission.csv :=
submission_V224.csv; (2) runs `session_2026-10-12/hf/h_check.py` (`--kind hf --wf W --members 20`, control `--kind hv`);
(3) own checks: V224-F markers in cell order, knobs in V224F_SAVED, CLOCK, files, receipts, fleet receipt (clean, n_keep 20/20,
no hard stop, coverage 1, parity, TTA AB, K2_CFG wall/start guard), fleet solo == member mean recomputed from the worker npz
(1e-6), final == independent rank blend (own tie-aware ranker, 1e-6), submission_V224 == V224-exact commit, fleet solo == HFR7-RR
fleet solo (1e-3); control: Synovitis exactly 0.5, the other 11 columns string-identical.

| run | result | file |
|---|---|---|
| v224f20 v1 (clean) | **V224F_CHECK PASS 28/28** (v224_check 17/17, HCHECK 16/16) | `gate_v224f20_v1.txt` |
| v224fctl v1 (control) | **V224F_CHECK PASS 21/21** | `gate_v224fctl_v1.txt` |
| negative tests on mutated copies of the real outputs | **11/11 as expected**: control output under the clean gate FAIL, clean output under the control gate FAIL, wrong w_f FAIL, submission cell +.01 FAIL, submission_V224 cell +.01 FAIL, fleet solo +.001 FAIL, control Synovitis not blank FAIL, control ACL +1e-6 FAIL, hard_stopped receipt FAIL; both unmutated PASS | `gate_negative_tests.txt`, `tests/neg_tests.py` |
| offline sim of cells 14/18/21 before any push (mocked fleet) | **7/7**: clean, budget refusal, r224 canary + clean fleet (Fracture kept 0.5), r224 canary + fleet fail (both blank), tampered save (H from memory), fleet bad ids, fleet 19 members | `tests/sim_cells_out.txt`, `tests/sim_cells.py` |

One gate fix during the run (documented): the first clean gate run flagged "R224_" in the V224_DONE line itself
(`"R224_OK": true`); the after-V224_DONE scan now starts on the next line and looks for the specific stage markers
(R224_STAGE / SKIPPED / TIMEOUT / UNUSABLE, NOTEBOOK_COMPLETE). Re-run: 28/28.

Commit timings (MEASURED, kernel log): V224 part done at 435.6 s (V224-exact commit 441.8 s); SAVE 435.7 s; fleet 444.0 -> 585.9 s
(2.37 min stage, 140 s workers); HF_COMPLETE 586.1 s. Fleet microbench under the R7 wheels: 9.63 / 9.86 s per study per T4 (all 20
members, both TTA passes) vs 8.52 / 8.33 in the HFR7-RR commit (+13-18 %; timm 1.0.25 or session variance, not separable from one
commit). Outputs are identical, only speed differs.

## 3. Diffs vs the bases
- vs V224-exact (`eeshsaxena/rsna-sub-v224` v1, submitted 57171955): cells 0-13 identical; 8 cells appended; metadata = V224-exact
  + datasets `eeshsaxena/rsna-knee-cnxt-fleet-k11w`, `eeshsaxena/rsna-knee-cnxt-fleet-k14w` (same as HFR7-RR; same docker pin,
  NvidiaTeslaT4, internet off, private).
- vs HFR7-RR (`eeshsaxena/rsna-sub-hfr7rr`): base H = V224 (her cell order, no re-rank, her 9 h guard) instead of HR7-RR; fleet
  moved from before R7 to after stage 3; fleet K2_CFG WALL_H 8.3 / MIN_START_MIN 168; HF cell reads the saved V224 output and adds
  the Synovitis canary; w_f .20 as HFR7-RR.
- vs control: control differs from v224f20 in 3 places only (cell 14 knobs, cell 19 MIN_START_MIN 600, cell 21 tag).

## 4. Hidden runtime (`wall_model.py`, output `wall_model_v224f20_v1.txt`)
Fleet hidden = fixed part (commit stage minus its 3-study loop, ~1.0 min) + studies per T4 x the slower GPU's s/study (MEASURED
9.86 s). MODEL:

| | 1,200 studies | 1,300 studies |
|---|---|---|
| fleet central | 1.66 h | 1.80 h |
| fleet slow (x1.15, FN11 slow/central) | 1.91 h | 2.07 h |
| fleet slow session (x1.3 more) | 2.48 h | 2.69 h |

- **V224-F hidden wall: central ~4.8 h** (V224-exact 3.0 + fleet 1.8); range **4.1-5.7 h**; slow session x1.3 on the slow end
  **~7.4 h**, still under the 8.3 h fleet wall and the 9 h limit. Cross-check: HFR7-R20 (same fleet, before R7) scored 4.2-5.5 h
  after submit incl. queue with an H of ~3.3 h, i.e. fleet <= ~2.2 h (MEASURED latency, MODEL split).
- Guards: R7 and r224 are unaffected (the fleet runs after them). The fleet starts only if the kernel is <= 5.5 h old (V224 part
  1.5x its 3.6 h slow end); it is cut at 8.3 h (+300 s kill) -> HF_FALLBACK (Synovitis canary), final file by ~8.4 h.
  P(fleet skipped or cut by time) ~1-2 % (GUESS).
- Not covered here: V224-F + FLIP (the v224ff builder's job): flip adds ~1.4 h (10-14 review), i.e. ~6.2 h central.

## 5. Exact submit command (main session decides; NOT run by me)
```
python session_2026-10-14/submit_at.py eeshsaxena/rsna-sub-v224f20 1 "V224-F20: V224-exact + K14 R fleet (20) HF w_f .20 after stage 3; fleet budget 5.5 h / wall 8.3 h; canaries Fracture=r224, Synovitis=fleet; V224F_CHECK 28/28"
```
Read: a Synovitis or Fracture column at 0.5 would show as a ~.02-.04 lower display; such an entry is not a valid pick.

If the 10-14 reads set w_f .25 or .30 (one-flag rebuild, same gates):
```
python session_2026-10-15/v224f/build_v224f.py --wf 0.25
python session_2026-10-15/v224f/running.py 6          # RUNNING_COUNT must be <= 1 before a push (2 GPU sessions max)
kaggle kernels push -p session_2026-10-15/v224f/kernel_v224f25
python session_2026-10-15/v224f/wait_fetch.py eeshsaxena/rsna-sub-v224f25 session_2026-10-15/v224f/out_v224f25_v1
python session_2026-10-15/v224f/v224f_check.py session_2026-10-15/v224f/out_v224f25_v1 --wf 0.25 --ref-v224 session_2026-10-14/v224/out_v224_v1 --ref-fleet session_2026-10-13/hr7rr/out_hfr7rr_v1
python session_2026-10-14/submit_at.py eeshsaxena/rsna-sub-v224f25 1 "V224-F25: ... V224F_CHECK 28/28"
```
(.30: replace 25 by 30.) Each new slug needs its own ~11 min commit before it can be submitted.

## 6. Risks / notes
- The fleet now runs under her global R7 wheels (timm 1.0.25 / pydicom 3.0.1). The commit shows identical fleet outputs and a clean
  re-encode parity; a hidden DICOM that only pydicom 3.0.2 decodes would be a failed study (median-filled if < 2 %, else the
  stage is not clean -> visible Synovitis canary). R7 decodes every hidden study with the same pydicom 3.0.1 in all V224 runs.
- FN-IDS writes an id placeholder to submission.csv between SAVE and the HF cell. Only a notebook that completes is scored, and
  the HF cell (the last cell) always rewrites submission.csv, so the placeholder can never be the scored file.
- The fleet weight was MEASURED positive on the HFR7 base (HFR7-RR beat HR7-RR); on the V224 base it is unmeasured: V224-F is the
  pre-registered re-check of that component (RESEARCH_960 sec 4, 10-15 slot 1).
- OAI exposure unchanged vs HFR7-RR (Raptor checkpoints + the DT fleet members).
- The canary cell's own `V224_DONE` line says `"use_f": false` because that cell is byte-identical to V224-exact's; the V224-F
  state is in `v224f_receipt.json` / `V224F_DONE`.

## Files (this dir)
`build_v224f.py` (builder), `v224f_check.py` (gate), `wall_model.py` (runtime model), `running.py` (concurrent-commit count),
`wait_fetch.py` (poll + fetch), `kernel_v224f20/`, `kernel_v224fctl/`, `celldiff_v224f20.txt`, `celldiff_v224fctl.txt`,
`out_v224f20_v1/`, `out_v224fctl_v1/` (+ `.log`), `gate_v224f20_v1.txt`, `gate_v224fctl_v1.txt`, `gate_negative_tests.txt`,
`wall_model_v224f20_v1.txt`, `push_v224f20.log`, `push_v224fctl.log`, `tests/` (sim + negative tests).

## Fixes 10-15 (00:25-00:45 IST, Claude fix agent; F1 / F2 / F3 of the 10-15 review, all CONFIRMED there)
Laptop scripts only. No frozen file was edited (student_gate.py still 8c018790; none of the files below are in the PREREG_TEACHER
A1 / A2 md5 tables). Nothing was pushed, submitted or created on Kaggle. The box was used on CPU only (cores 34-39, nice 19, ionice
idle) in the scratch dir `~/r7t_labels/rb_dry/fix1015/`; the box queue and GPU were not touched. Pre-change copies sit next to each
file as `*.pre_fix_1015`, and the old stand-in / mock dirs are kept as `tests/*.pre_fix_1015/`.

| file | md5 before | md5 after |
|---|---|---|
| build_v224fs.py | fda1e28f | a54afad651ff2bc30e1dd7ce6e55df48 |
| make_s_pack.py | 2eaf33e6 | 5b2df773586e23fb3c5b5e441e240c1c (box scratch copy `~/r7t_labels/rb_dry/fix1015/make_s_pack.py` == 5b2df773, MEASURED) |
| v224fs_check.py | 057d896f | b5f5d4e17befc60fc826a935384e7b0e |
| v224ffs_check.py | defeda80 | d91afcc3bdca947aec477a61dc2dce9a |
| tests/mock_fs_output.py | 40f04b10 | bc3603e8edd33acf3e4892cb930a4eef |
| tests/test_fix1015.py / tests/gate_neg_fix1015.py | new | 65d0f4d1de056c8bd5a38afcd47d8cde / 978317d7571505277d135e4744237551 |

Box mirror: none of these scripts is mirrored on the box (MEASURED 00:24: `~/r7t_labels/` has no make_s_pack / build_v224fs /
check copies). make_s_pack.py reaches the box in RUNBOOK step 4, whose md5 line now says 5b2df773.

**F1 (build_v224fs.py: the FF base kept the V224 part only with a 4.4 h flip budget).** Before: `--base ff` patched cell 7
`FLIP_TA_BUDGET_H` 4.9 -> 4.4 (the default, enforced by `flip <= start - 0.6`), so FF-S flipped far fewer R7 studies than its
FF20 base, and nothing on the LB shows it. MODEL from RUNTIME_FFS_1015: flip share ~36 % vs ~77 % for FF20. After:
- `--flip-budget-h` defaults to 4.9 (FF20's own value). At 4.9 the builder does not touch cell 7, so cells 0-15 (everything
  before SAVE) are byte-identical to `v224ff/kernel_v224ff20`. Any other value still patches cell 7, as before.
- The assert is relaxed to `1.0 < flip < start`.
- `--start-budget-h` defaults to 5.1 for ff (MIN_START_MIN 192) and to 5.0 for f. The runtime review's recommended 4.9 / 5.1:
  MC median 6.61 h, p99 7.67 h, flip >= 90 % in 74 % of sessions (FF20 73 %), P(fleet skipped) ~.001, P(hard stop) ~.001,
  slow corner 8.22 h; all MODEL. 5.0 stays allowed for ff (verdict MC P(skip) .004 at 4.9 / 5.0, MODEL). f is fixed at 5.0.
- The builder assert `MIN_START == 198` became `in (198, 192)`.
- v224fs_check.py: `--start-budget-h` now defaults to the build json, and the "build: MIN_START_MIN" check accepts (f, 198),
  (ff, 198) and (ff, 192). It used to hard-code 198, so v224ffs_check (which nests it without that flag) could not gate a 5.1
  build. The check count is unchanged.
- v224ffs_check.py: `--flip-budget-h` default 4.4 -> 4.9. It must still equal the build json, else the gate asserts.
- tests/mock_fs_output.py takes the flip / start budgets from the build json instead of hard-coded 4.4 / 5.0 / 198.

**F2 (make_s_pack.py: any REAL PASS gate file unlocked the T_b pack).** Before: the gate check was three substrings, so a T_a
PASS file packed the T_b checkpoints. After:
- The gate's first line (written by the frozen student_gate.py: `student_gate <ts> | student <tpl> | table <path> | teacher
  <path> | REAL`) must name `student == --tpl` and a table whose basename == basename(--table).
- Exactly one `R7T_STUDENT_GATE` line is allowed, and it must be the all-pass `R7T_STUDENT_GATE REAL A1 | G0 P G1 P G2 P G3 P
  G3' P G4 P | VERDICT PASS`.
- The old substring assert is kept.
- `S_PACK.json` now records `gate_sha256` and `gate_header`. The `S_PACK OK` line prints the gate sha256[:16].
- Not done (on purpose): an mtime check (a re-fetch would refuse a valid pack) and per-checkpoint shas from the gate (that would
  need the frozen student_gate.py).

**F3 (`--base` defaulted to f; stale NEXT text).** After:
- build_v224fs.py `--base` is required, so a build without it exits 2 and builds nothing.
- make_s_pack.py's NEXT text is the FLIP-base plan: `build_v224fs.py --wf 0.20 --base ff`, then `v224ffs_check.py` with
  `--ref-flip/--ref-exact/--ref-fleet`.
- The optional "v224fs_check refuses base f" guard was not added. v224ffs_check nests v224fs_check on ff builds, the f base is
  only reachable by typing `--base f`, and v224ffs_check already refuses a base-f build json.

Tests and dry runs (all MEASURED, 10-15):
- `python tests/test_fix1015.py`: **PASS 19/19** on the fixed scripts and **FAIL 9/19 on the `*.pre_fix_1015` copies**. All 10
  fixed cases fail before the fix: flip 4.9 / cell 7 / cells 0-15 == FF20 / start 5.1 / `--start-budget-h 5.0` with flip 4.9 /
  no-`--base` refused / NEXT text / T_a gate / T_b-student-on-T_a-table gate / two-verdict-line gate. The 9 regression cases
  pass both ways (explicit 4.4 still builds, flip >= start / start 5.5 / f at 5.1 refused, right T_b gate, FAIL and SIM gates
  refused, T_a gate with `--tpl` T_a). Logs: `tests/test_fix1015_post.txt`, `tests/test_fix1015_pre.txt`.
- D6 stand-in builds re-run (`--base ff|f --standin [--replace]`):
  - The f stand-in notebooks are byte-identical to the pre-fix ones (the f path is unchanged).
  - The ff stand-ins differ only in cells 7 (4.9, now == FF20), 16 (SAVE knobs 5.1 / 192) and 22 (K2 MIN_START_MIN 192).
  - New FF-S stand-in sha256 190d0c70 (was 037f67a7); REPLACE 4f4c0cf7.
  - S-MERGE (cell 20) and HF-S (cell 24) are byte-identical, so box D9 (`sim_ffs_cells` 15/15) still applies and was not
    re-run.
- D7 `python tests/mock_fs_output.py ff`: **V224FFS_CHECK PASS 14/14**, 7/7 negatives FAIL, MOCK_FS_GATE PASS 8/8
  (`tests/mock_fs_output_ff_fix1015.txt`). Nested lines: `build: MIN_START_MIN 192.0` OK; `V224FS_CHECK FAIL 30/31` and
  `V224_CHECK FAIL 14/17 exact` are by design (unchanged). f regression `mock_fs_output.py f`: V224FS_CHECK PASS 32/32, 7/7
  negatives (`tests/mock_fs_output_f_fix1015.txt`).
- `python tests/gate_neg_fix1015.py`: **PASS 5/5** (`tests/gate_neg_fix1015.txt`):
  - the clean mock passes;
  - a log FLIP_TTA_FORCED budget of 4.4 FAILs 13/14;
  - SAVED knobs 5.0 / 198 against a 5.1 / 192 build FAIL;
  - `--flip-budget-h 4.4` against the 4.9 build is refused;
  - the pre-fix gate (default 4.4) refuses the 4.9 build.
- Box D2 re-run for F2 (`~/r7t_labels/rb_dry/fix1015/d2_fix1015.sh` + `.log`; new mock from `mk_mock_pack.py`, 1.3 GB, deleted
  after; mock gate texts in the scratch dir, never in `~/r7t_labels/gate/`):
  - Positive: 5 x `f<F> OK` and `S_PACK OK`, manifest sha256 3f81cd0f (same as D2 at 00:00); S_PACK.json gate_sha256 ==
    sha256 of the gate file.
  - **12/12 negatives refused:** T_a gate, T_b student on the T_a table, T_b G3' FAIL, a FAIL file with an appended all-pass
    line, SIM, upload exists, seed 1043, sel_labels v4, `oai_9123`, extra top-level key, two raw-best files, fp32.
  - The **pre-fix copy packed the T_b mock under the T_a gate** (`S_PACK OK`), which reproduces the bug.
- `build_v224fs.py --wf 0.20 --base ff --s-manifest <mock manifest>` (the main-session command, inside test_fix1015) builds
  `kernel_v224ffs20`, 25 cells, flip 4.9, start 5.1, MIN_START_MIN 192.

Docs updated: `r7teacher/RUNBOOK_STUDENT_1015.md`:
- step 4 md5 + F2/F3 notes;
- step 7 expected MIN_START_MIN 192 + F1 defaults;
- step 9 "flip budget 4.9";
- step 10 criteria;
- **step 11 Amendment 4 draft**: start budget 5.5 -> 5.1, flip budget unchanged at 4.9 (supersedes PLAN_TEACHER FX8's 0.5 h cut),
  plus the shared flip-lottery caveat ("a non-moving read is a weak negative, not proof S is negative");
- step 12 submit message;
- App. B D11 row;
- App. C F3 / F4 marked fixed.

PLAN_TEACHER.md and PREREG files were NOT edited. Main must adopt or reject the change in Amendment 4: the FX8 row still says
"cut by about 0.5 h". To build the old FX8 variant, pass `--flip-budget-h 4.4 --start-budget-h 5.0` to the build and
`--flip-budget-h 4.4` to v224ffs_check.

## FFSR gate test 10-15 (00:45-01:05 IST, Claude subagent)
Laptop CPU only (light: builder + mocks + gate subprocesses, a few minutes in total). No gate was edited. Nothing was pushed,
submitted or created on Kaggle. The box, the box queue and the rental kit were not touched.

Why: PREREG_10-15 Amendment 2 may submit V224-FFSR20 instead of FF-S. FFSR is `build_v224fs.py --base ff --replace`: F = S alone,
s_share 1.0, MEMBERS = the 5 S. `v224ffs_check.py` had only been mock-tested on the default FF-S build, so it had to be proven on
a replace build too.

**Verdict: the gate is correct for replace mode.**
- It PASSes a correct FFSR mock and FAILs all 18 negatives, each on the intended check.
- No false pass and no false fail were found. So `v224ffs_check.py` and `v224fs_check.py` are unchanged, and no `*.pre_ffsr_1015`
  copy of them exists.
- One coverage gap remains (see below). It is by design, not a defect.

**1. Stand-in.**
- `tests/standin_v224ffsr20` was built at 00:33 by the current builder (a54afad6). I backed it up and rebuilt it with
  `python build_v224fs.py --wf 0.20 --base ff --replace --standin` (`--standin` always writes `tests/standin_<tag>`).
- All 5 files came out byte-identical, so the stand-in was not stale.
- Notebook sha256 4f4c0cf7f9a05cb8... (the "Fixes 10-15" value): 25 cells, members 5 (K14 0 + S 5), s_share 1.0,
  MIN_START_MIN 192, flip 4.9.

**2. `tests/mock_fs_output.py` mode `ffsr`.** The pre-change copy is `tests/mock_fs_output.py.pre_ffsr_1015`. The f / ff code paths
are unchanged. The mock output `tests/mock_out_v224ffsr20/` is built from `v224ff/out_v224ff20_v1`:
- worker npz: 5 rows = the commit's 5 DT rows under the S names, in the build's MEMBERS order (P, PA, PB);
- fleet receipt: models / names / n_keep 5, K2_CFG MEMBERS = S;
- log: FLIP budget 4.9; V224F_SAVED with members 5, groups K14 0 / S 5, s_share 1.0; the s-merge line says 30 checkpoints
  (S-MERGE still links K12 25 + S 5); the fleet RESULT and worker member lines name the 5 S;
- the HF-S cell of the FFSR stand-in notebook is EXECUTED on it (V224FS_F2G, HF_BLEND_OK, HF_COMPLETE, V224F_DONE). Here
  fleet2g == fleet solo byte for byte, as it should be for REPLACE.

Gate call: `v224ffs_check.py OUT --wf 0.2 --build tests/standin_v224ffsr20/BUILD_V224FS.json --ref-flip
session_2026-10-14/v224/out_v224flip_v1 --ref-exact session_2026-10-14/v224/out_v224_v1 --ref-fleet v224ff/out_v224ff20_v1`.

Commands:
```
python tests/mock_fs_output.py f      > tests/mock_fs_output_f_ffsr1015.txt
python tests/mock_fs_output.py ff     > tests/mock_fs_output_ff_ffsr1015.txt
python tests/mock_fs_output.py ffsr   > tests/mock_fs_output_ffsr_1015.txt    # after ff: the cross negatives read tests/mock_out_v224ffs20/
python tests/ffsr_gap_ffs_compare_1015.py > tests/ffsr_gap_ffs_compare_1015.txt
```

**Results (MEASURED):**

| mode | clean mock | negatives | MOCK_FS_GATE |
|---|---|---|---|
| f (regression) | V224FS_CHECK PASS 32/32 | 7/7 FAIL | PASS 8/8 |
| ff (regression) | V224FFS_CHECK PASS 14/14 | 7/7 FAIL | PASS 8/8 |
| ffsr (new) | V224FFS_CHECK PASS 14/14 | 18/18 FAIL | PASS 19/19 |

Regression proof for f and ff: each mode was run once with the pre-change copy and once with the new file.
- stdout is identical once the random temp view-dir names are normalised;
- every file in `tests/mock_out_v224fs20/` (29) and `tests/mock_out_v224ffs20/` (30) has the same md5.

ffsr clean run: the nested V224FS_CHECK is 29/30. Its single FAIL is the by-design exact-mode V224 sub-gate. FF-S has 31 checks
there, because the `--ref-fleet` K14 check is skipped when k14 is empty. Key OK lines:
- "fleet receipt: clean, 5 members";
- "submission_fleet2g.csv == (1 - 1.00) mean(K14 0) + 1.00 mean(S 5)": 1.1e-16;
- "two-group F differs from the equal member mean (REPLACE: equal)": 0;
- final blend: 3.3e-8;
- own END-TO-END, 5 members: 2.5e-8;
- v224_check --flip: 21/21.

Negatives. Each one FAILs. The checks that caught it are in brackets: n = nested v224fs_check, o = own v224ffs_check.
1. submission_fleet2g.csv +1e-3 [n fleet2g recompute, n f2g sha]
2. submission.csv +1e-3 [n in [0,1], n sha, n final recompute, o END-TO-END]
3. an S row x0.5 in the npz [n fleet solo, n fleet2g, n final, o END-TO-END]
4. fleet2g missing [n output files, n recompute]
5. no s-merge line [n markers]
6. V224F_SAVED members 25 [n knobs]
7. V224F_SAVED groups K14 20 [n knobs]
8. s_share .5 in SAVED + HF lines + hf / v224f receipts [n knobs, n hf_receipt, n v224f_receipt]
9. hf_receipt s_share .5 only [n hf_receipt]
10. K14 present: the fleet ran K14 20 + S 5, with npz / solo / receipt / RESULT line consistent [n markers, n fleet receipt
    models, n K2_CFG members, n npz names, o END-TO-END member count]
11. F = the K14 20-member mean, with fleet2g + submission + shas consistent [n fleet2g .177, n final .067, o END-TO-END]
12. F = .5 K14 + .5 S (the FF-S F), with shas consistent [n fleet2g .089]. On these 3 studies the per-label ranks equal the
    S-only ranks, so submission.csv is unchanged and only the fleet2g recompute catches it.
13. Synovitis canary, fleet not started: receipt not clean; no npz / solo / fleet2g / H; HF-S re-executed -> HF_FALLBACK +
    CANARY_FIRED [n markers, n foreign markers, n files, n fleet receipt, n hf mode H_only / blanked Synovitis,
    n v224f_receipt, n blanked column, o END-TO-END]
14. Synovitis canary, fleet clean on disk but HF-S saw 25 members ("fleet ran 25 members, expected 5") [same receipts and markers,
    n final .5, o END-TO-END .5]
15. weight not applied: submission = pct(V224) (w_f 0), receipt sha updated [n final .1, o END-TO-END .1]
16. wrong weight: w_f .30, receipt sha updated [n final .05, o END-TO-END .05]
17. cross: the FFSR mock under the FF-S build json [n markers, knobs, hf_receipt, fleet receipt, npz names, o END-TO-END]
18. cross: the FF-S mock under the FFSR build json [the same checks, the other way round]

Cosmetic note, not changed: when "fleet solo == npz mean" raises (e.g. the npz names differ), the next block FAILs with a
NameError ('mean' / 'Pm' not defined) instead of a clean message. It still fails closed.

**Coverage gap (by design, NOT fixed; main decides).**
- v224fs_check anchors the fleet numerics to a reference only through `--ref-fleet` on the K14 rows
  (`if A.ref_fleet and B['k14']`). A replace kernel has no K14 rows, so nothing in the gate compares its fleet output to a
  reference.
- Probe: the S predictions of worker 0's 2 studies are swapped (a study-id misalignment), and the kernel's own HF-S cell re-makes
  fleet solo / fleet2g / submission / receipts / log. The FFSR gate gives **V224FFS_CHECK PASS 14/14** while submission.csv moves
  by up to .133. This is the `GAP` line of `mock_fs_output.py ffsr`, not counted in 19/19.
- The same probe on the FF-S mock (`tests/ffsr_gap_ffs_compare_1015.py`): all 25 rows swapped FAILs 13/14 ("K14 group mean == the
  V224-F20 commit fleet solo" max diff .822); S rows only PASSes 14/14.
- So FF-S catches a worker-level fault through its 20 K14 rows and FFSR cannot. An S-only fault passes in both modes.
- What still covers FFSR: S-MERGE's in-kernel full sha256 of every S file, the fleet worker's model_sha / contract checks and the
  fleet's DICOM parity check.
- Proposed check (not implemented): an optional `--ref-s OUT_FFS_COMMIT` in v224fs_check, requiring the FFSR S group mean == the
  mean of the S rows in the V224-FF-S commit npz within 1e-3 (same 5 checkpoints, same worker, same 3 studies). It is usable only
  if the FF-S kernel was committed first, which Amendment 2's "after a positive V224-F-S read" implies.

| file | md5 |
|---|---|
| build_v224fs.py | a54afad651ff2bc30e1dd7ce6e55df48 (unchanged) |
| v224ffs_check.py | d91afcc3bdca947aec477a61dc2dce9a (unchanged) |
| v224fs_check.py | b5f5d4e17befc60fc826a935384e7b0e (unchanged) |
| tests/mock_fs_output.py | bc3603e8edd33acf3e4892cb930a4eef -> 87d511df47b2a4aa13bce9f7e7f09a66 |
| tests/mock_fs_output.py.pre_ffsr_1015 | bc3603e8edd33acf3e4892cb930a4eef |
| tests/standin_v224ffsr20/rsna_sub_v224ffsr20.ipynb | 353fac5421519bb523b9eff81bda4b29 (sha256 4f4c0cf7...) |
| tests/standin_v224ffsr20/BUILD_V224FS.json | cac8616a495357ff5081c50580a5aa10 |
| tests/mock_fs_output_ffsr_1015.txt | 567a90a4cdba22d50916012348e41e7b |
| tests/mock_fs_output_f_ffsr1015.txt / tests/mock_fs_output_ff_ffsr1015.txt | 290bb49e197b6eece18ed8f7bb4ffed8 / 9129542337dae4f5425c8dfb1c0831da |
| tests/ffsr_gap_ffs_compare_1015.py / .txt | 08a23824b9909ef4fc446e96f579f5a3 / 942a337e7768605a45be530eacda6065 |
| tests/mock_out_v224ffsr20/ submission.csv / submission_fleet2g.csv (== fleetsolo) | 970888601e9f589a4c6d670395b6c435 / 8591acdcc0100b48484574bb3baeedd1 |

## --ref-s 10-15 (01:05-01:15 IST, Claude subagent; main's request after the FFSR gate test above)
Laptop CPU only. Nothing was pushed, submitted or created on Kaggle. The box, the box queue and the rental kit were not touched.

**What changed** (pre-change copies are kept next to each file):
- `v224ffs_check.py` gets an optional `--ref-s <V224-FF-S commit output dir>`, for REPLACE builds only (k14 empty in the build json).
  - New check (5): "REPLACE: S group mean == the V224-FF-S commit S group mean (0.001)". The per-study mean of the 5 S rows in
    this output's worker npz must match the same mean from the FF-S commit's npz, max abs diff <= 1e-3 in npz column order.
  - The check fails cleanly (one FAIL line, with the reason) when: `--ref-s` is this output itself; the ref npz lacks the S members
    or has no K14 members (so it is not an FF-S output); this output's npz members are not exactly S; the study ids differ; or an
    npz is missing.
  - `--ref-s` with a non-REPLACE build is refused (AssertionError, rc 1, no gate line).
  - REPLACE without `--ref-s` prints `  WARN REPLACE build without --ref-s: the fleet numerics have NO reference anchor ...`. The
    check count is then unchanged (14).
- `v224fs_check.py`: when the worker npz fail the "fleet solo == member mean" block (no npz, or member names != the build), the
  two-group block now FAILs at once with `AssertionError('not run: the worker npz failed the member-mean check above (missing npz
  or member names != the build)')`. It used to fail with `NameError('mean' / 'Pm')` or a stray `ValueError(list.index)`. This is
  2 lines: `Pm = nm = mean = None` before block 1 and an assert at the top of block 2. It still fails closed, and no outcome
  changes.

Commands:
```
python tests/mock_fs_output.py f    > tests/mock_fs_output_f_refs1015.txt
python tests/mock_fs_output.py ff   > tests/mock_fs_output_ff_refs1015.txt
python tests/mock_fs_output.py ffsr > tests/mock_fs_output_ffsr_refs1015.txt
python tests/test_refs_1015.py      > tests/test_refs_1015.txt      # needs the ff + ffsr mock outputs
```

**Tests: `tests/test_refs_1015.py` gives TEST_REFS_1015 PASS 11/11 (MEASURED).** Each case runs on the fixed gate (post) and on
the `.pre_refs_1015` copy (pre).

| case | post | pre |
|---|---|---|
| FFSR mock, no --ref-s | PASS 14/14 + WARN | PASS 14/14 |
| FFSR mock, --ref-s FF-S mock | **PASS 15/15** (diff 0) | rc 2 (unknown flag) |
| study-id misalignment (the GAP probe), no --ref-s | PASS 14/14 + WARN (gap without the flag) | PASS 14/14 |
| study-id misalignment, --ref-s FF-S mock | **FAIL 14/15**, S anchor max diff .778 | rc 2 |
| --ref-s = the FFSR output itself | FAIL 14/15 (refused) | rc 2 |
| --ref-s = FF20 commit (no S members) | FAIL 14/15 (clean message) | rc 2 |
| --ref-s = FF-S mock with an S row x0.5 | FAIL 14/15 (max diff .098) | rc 2 |
| FF-S build + --ref-s | refused, rc 1 | rc 2 |
| FF-S mock, FF-S build, no --ref-s | PASS 14/14, no WARN | PASS 14/14 |
| v224fs_check, FFSR mock without npz | FAIL 23/26, clean "not run" message | FAIL 23/26, NameError 'Pm' |
| v224fs_check, FF-S mock under the FFSR build | FAIL 17/26, clean message | FAIL 19/29, NameError 'mean' |

**Regression without --ref-s (MEASURED):** the counts are unchanged.

| mode | clean mock | MOCK_FS_GATE |
|---|---|---|
| f | V224FS_CHECK PASS 32/32 | PASS 8/8 |
| ff | V224FFS_CHECK PASS 14/14 | PASS 8/8 |
| ffsr | V224FFS_CHECK PASS 14/14 | PASS 19/19 (GAP probe still PASS, as expected without --ref-s) |

- f and ff stdout is identical to the runs before this fix, once the temp-dir names are normalised.
- ffsr differs in two ways only:
  - the WARN line in the clean gate output;
  - 4 negatives (K14 present, canary not started, both cross runs): NameError / ValueError became the clean "not run" FAIL.
    In 2 of them the nested OK count drops (23 -> 20, 19 -> 17) because the checks that used to run on mismatched npz data are
    now skipped. All 4 still end V224FFS_CHECK FAIL 12/14.
- NameError appears 6 times in the old ffsr log and 0 times in the new one.

**Gate call for tonight's FFSR20 commit** (FF-S committed first, same S dataset version; compare `s_manifest_sha256` in the two
BUILD_V224FS.json files):
```
python session_2026-10-15/v224f/v224ffs_check.py <OUT_FFSR20> --wf 0.20 --build <kernel_v224ffsr20>/BUILD_V224FS.json \
  --ref-flip session_2026-10-14/v224/out_v224flip_v1 --ref-exact session_2026-10-14/v224/out_v224_v1 \
  --ref-fleet session_2026-10-15/v224ff/out_v224ff20_v1 --ref-s <OUT_FFS20 commit dir>
```
Expected: V224FFS_CHECK PASS 15/15. The S rows of the two commits come from the same 5 checkpoints, worker and 3 studies. The
cross-commit K14 rows were byte-identical before, so a diff near 0 is expected (MODEL).

| file | md5 |
|---|---|
| v224ffs_check.py | d91afcc3bdca947aec477a61dc2dce9a -> 2228b880c8f65d5017990a8b3aac67b3 |
| v224ffs_check.py.pre_refs_1015 | d91afcc3bdca947aec477a61dc2dce9a |
| v224fs_check.py | b5f5d4e17befc60fc826a935384e7b0e -> 9a4ad8bf047cd28094b9d411dc73fff5 |
| v224fs_check.py.pre_refs_1015 | b5f5d4e17befc60fc826a935384e7b0e |
| build_v224fs.py / tests/mock_fs_output.py | a54afad651ff2bc30e1dd7ce6e55df48 / 87d511df47b2a4aa13bce9f7e7f09a66 (both unchanged) |
| tests/test_refs_1015.py / .txt | c688d87efea1d939e4181838b218c986 / ce655d55d8c9f10cb47680c18e525170 |
| tests/mock_fs_output_f_refs1015.txt | 95427383d59a11cb92c602514edb2567 |
| tests/mock_fs_output_ff_refs1015.txt | 08b1341b07a77221e13e0cf31cb94cbb |
| tests/mock_fs_output_ffsr_refs1015.txt | d0defd9b95aee4d349676ab6eb548af4 |

## AO 10-15 (02:15-03:05 IST, Claude subagent): always-on-flip student builder + gate + runtime review (PREREG_10-15 Amendment 3 R2 / R3)
Laptop CPU only (light: builder subprocesses, mocks, gate subprocesses, one ~50 s numpy sim). Nothing pushed, submitted or created
on Kaggle. The box, its queue, the rental kit and scheduled tasks were not touched. No PREREG / frozen file edited. Pre-change copies:
`build_v224fs.py.pre_ao_1015` (a54afad6), `sim_ffs_runtime.py.pre_ao_1015` (3b67a420). v224fs_check.py, v224ffs_check.py,
tests/mock_fs_output.py and tests/test_refs_1015.py are unchanged (md5 9a4ad8bf / 2228b880 / 87d511df / c688d87e).

### Verdict
- **Builder ready** for branch b (`--base flip`, V224-FLIP-AO + S, slug suggestion `eeshsaxena/rsna-sub-v224flips20`) and branch c
  (`--base flrm`, V224-FLRM-AO + S, `eeshsaxena/rsna-sub-v224flrms20`). Blocked only on the real S dataset (its manifest
  `v224f/s_upload/rsna-knee-cnxt-fleet-r7tb/r7tb_manifest.json`, RUNBOOK_STUDENT step 5); then one build command per kernel, one GPU
  commit each (main), and the gate below. Stand-ins: `tests/standin_v224flips20` (nb sha256 2c2b8e63..., 24 cells) and
  `tests/standin_v224flrms20` (f3b6b0f9..., 26 cells). MEASURED.
- **Gate** `v224aos_check.py`: clean mocks PASS 28/28 for both bases (29/29 with `--ref-s`); every negative FAILs (flips 17/17,
  flrms 18/18, each on the intended check); HF-AO cell tests 6/6 per base. MEASURED (offline mocks built from real commit outputs).
- **Runtime** (MODEL, `sim_ffs_runtime.py` section D, `sim_ffs_runtime_ao.txt`): start guard **G = 6.5 h (MIN_START_MIN 108)**.
  SUPERSEDED by "AO fix 10-15" below: G = 7.0 h (MIN_START_MIN 78) is now the builder default and the recommendation.
  FLIP-AO + S kernel end median 4.85 h, **p99 6.62 h**; FLRM-AO + S median 4.98 h, **p99 6.74 h** (session model B); heavy-tail
  stress model p99 7.31 / 7.46 h. P(fleet skipped -> Synovitis canary) .004-.005 (FLIP) / .006-.010 (FLRM); P(hard stop) 0;
  P(> 9 h) 0. Latest clean end (start at G, S fleet x1.15 x1.45) 7.02 h; with a 2.5x S cost 7.72 h. Submit-to-score median ~5.0 h
  (FLIP-AO) / ~5.1 h (FLRM-AO) vs 4.66 h modelled for V224-FLIP itself.
- **f / ff byte-identical** (MEASURED): the 4 stand-ins (20 files) rebuild to their pre-change sha256; the `--s-manifest` f, ff,
  ff --replace and ff 4.4 / 5.0 builds of the new builder == the pre_ao_1015 builder's, every file byte for byte.

### 1. Builder (`build_v224fs.py --base flip | flrm`)
- **Base**: the commit-tested, Kaggle-pulled notebook, sha-pinned: V224-FLIP = `session_2026-10-14/v224/kernel_v224flip` (73a9d562,
  == `v224rm/pulled_v224flip`, the scored team best 57174021); V224-FLRM = `v224rm/kernel_v224flrm` (f14f6acc, == `pulled_v224flrm`,
  built by `build_v224rm.py --wf 0 --flip --slug rsna-sub-v224flrm`; that builder is NOT re-run because it rewrites review files in
  v224rm/). Every base cell is kept byte-identical and in order. Proof 1: the notebook minus the appended cells re-serialises to the
  base FILE byte for byte. Builder asserts: her 13 cells verbatim; the V224-FLIP cell right after her R7 assets cell (flip FORCED:
  pinned compact_infer.py `--flip-tta`, `FLIP_TTA_FORCED_ON`); the V224 CANARY after her stage 3; (flrm) RM after her r224 writefile
  and RM_GUARD last; no time-aware driver (`compact_infer_ta`, `FLIP_TA_BUDGET_H`) and no fleet cell in the base.
- **Appended 9 cells** (indices 15-23 flip, 17-25 flrm): the V224-F fleet cells built by `build_v224f.py --start-budget-h 6.5`
  (SAVE, ENV-SNAP, FN-IDS, K12-MERGE, CLOCK, FLEET, ENV-RESTORE, HF) with the S changes: SAVE-AO (header names the base; knob dict =
  the f / ff one + `ao_base`), S-MERGE (the f / ff text), K2_CFG line (MEMBERS / TTA = the 5 S, MIN_START_MIN 108, WALL_H 8.3), HF-AO.
  Proof 2: with SAVE / K2 line / HF reverted and S-MERGE dropped, the 9 cells == the V224-F (6.5 h) fleet cells byte for byte.
- **REPLACE semantics implied** (A3 R2): F = mean of the 5 S, s_share 1.0, w_f .20 (asserted), no K14 members (k11w + k14w stay
  attached: K12-MERGE links them and S-MERGE adds S to that dir, as in FFSR20).
- **HF-AO = the f / ff HF-S with ONE functional edit** (proven: undoing the 2 text edits gives the f / ff text). Fracture is set back
  to 0.5 after the blend when any earlier canary fired on Fracture, read from `_canary` (V224 CANARY) AND `_gcr` (RM_GUARD's receipt
  dict, flrm only). Why: RM_GUARD writes Fracture 0.5 into submission.csv (= H) but not into `_canary`, so the f / ff HF-S would let
  the S blend un-blank it (cell test: Fracture .60 / .73 / .67 instead of 0.5). hf_receipt gains `fired_before_hf`. On the clean
  path HF-AO output == HF-S output byte for byte (cell test). The builder asserts that no appended cell names `_gcr` / `_canary` and
  that only RM_GUARD binds `_gcr`.
- **Guards / canaries**: start guard G (default 6.5; 6.0 / 7.0 allowed, all reviewed in section D; anything else refused); not
  started / not clean / any HF problem -> HF_FALLBACK, **Synovitis 0.5** (V224-F semantics, visible on the LB); fleet wall 8.3 h +
  300 s kill. Refused (tested): `--flip-budget-h` with an AO base, FF start budgets, guard 7.5, w_f .25, no manifest.
- usage (since "AO fix 10-15": add `--start-budget-h 7.0`, also the default): `python session_2026-10-15/v224f/build_v224fs.py --wf 0.20 --base flip --s-manifest session_2026-10-15/v224f/s_upload/rsna-knee-cnxt-fleet-r7tb/r7tb_manifest.json`
  -> `v224f/kernel_v224flips20/` (notebook + kernel-metadata.json: the base's 3 datasets + k11w + k14w + r7tb, docker 2757e0c7,
  T4 x2, private, internet off). Same with `--base flrm` -> `kernel_v224flrms20/`.

### 2. Gate (`v224aos_check.py`)
`python session_2026-10-15/v224f/v224aos_check.py OUT --wf 0.20 --build <kernel dir>/BUILD_V224FS.json --ref-exact session_2026-10-14/v224/out_v224_v1 --ref-flip session_2026-10-14/v224/out_v224flip_v1 [--ref-flrm session_2026-10-15/v224rm/out_v224flrm_v1 (flrm, required)] [--ref-s OUT_WITH_SAME_S]`
-> `V224AOS_CHECK PASS 28/28` expected (29/29 with --ref-s). Checks:
- (0) build + notebook: AO REPLACE knobs, guard in the reviewed set, MIN_START_MIN == (8.3 - G) x 60; notebook sha == build; base cells
  re-serialise to the pinned base file (== Kaggle pull); 9 appended cells in order; no driver; K2 MEMBERS == S; S_CFG manifest == the
  build (sorted-json sha256) and every S sha256[:16] == sha256_full.
- (1) nested `v224fs_check.py`: all OK except exactly its exact-mode V224 sub-gate and its f / ff-only "build: MIN_START_MIN" line.
- (2) base part on a view (log cut at V224_DONE / V224RM_DONE, submission_V224.csv as submission.csv): flip -> `v224_check.py --flip
  --ref V224-exact` (21/21 on the mock); flrm -> `v224flrm_check.py` (10/10 on the mock).
- (3) **flip on for every study**: FLIP_TTA_FORCED payload is the V224-FLIP form (worker_args only, `--flip-tta` once, no `--ta-*`);
  no driver marker / file anywhere (log, worker logs, flip_ta_receipt.json); R7 controller and both workers = pinned compact_infer.py
  (8401812c) with the flip_tta block (M/L perm, MCL plain pass only) and no time_aware_flip block; the workers' per_study rows
  partition the test ids; every worker log FLIP_TTA_READY + WORKER_DONE studies == scored; R7 ours_sub == the V224-FLIP commit R7
  (1e-3); submission_V224.csv == the base commit submission (1e-6).
- (4) S members and shas: S-MERGE "RESULT OK ... + S: 30 checkpoints (... r7tb, sha256 verified"; fleet receipt names /
  members_filter / K2 MEMBERS == the 5 S, models 5, n_keep [5, 5], TTA AB; manifest_sha16 of each S == the build's S manifest; S
  microbench <= 400 ms / member / study; optional --ref-s S group mean anchor (1e-3).
- (5) END-TO-END recompute: V224 recipe from the stage csvs == submission_V224.csv; (1 - .2) pct(H) + .2 pct(mean of the S npz rows)
  == submission.csv.
- (6) no CANARY_FIRED / HF_FALLBACK / RM_GUARD_FIRED; canary receipt fired [] (flrm: rm_guard mirror_ok); hf_receipt mode HF, blanked
  [], fired_before_hf [] (proves HF-AO ran); no constant column; fleet started at age <= G with >= MIN_START_MIN left.
- Coverage note: without `--ref-s` the S numerics are only self-consistent (the FFSR20 gap; a WARN line says so). Use the first AO
  commit (or an FF-S / FFSR commit on the same S dataset version) as `--ref-s` for the second.

### 3. Tests (all MEASURED, laptop)
| test | result | log |
|---|---|---|
| `tests/mock_ao_output.py flips` (base = V224-FLIP commit, fleet = FF20 commit reduced to the 5 DT rows as S, HF-AO executed) | clean PASS 28/28, + --ref-s 29/29, 17/17 negatives FAIL, CELL 6/6 -> MOCK_AO_GATE PASS 19/19 | `tests/mock_ao_output_flips_1015.txt` |
| `tests/mock_ao_output.py flrms` (base = V224-FLRM commit) | clean 28/28, + --ref-s 29/29, 18/18 negatives FAIL, CELL 6/6 -> PASS 20/20 | `tests/mock_ao_output_flrms_1015.txt` |
| `tests/test_ao_1015.py` (builder: refusals, real --s-manifest builds, independent proofs, f / ff regression) | TEST_AO_1015 PASS 41/41 | `tests/test_ao_1015.txt` |
| `tests/mock_fs_output.py f / ff / ffsr` (regression) | V224FS_CHECK 32/32 + 8/8; V224FFS_CHECK 14/14 + 8/8; 14/14 + 19/19 (GAP probe still PASS, known) | `tests/mock_fs_output_{f,ff,ffsr}_ao1015.txt` |
| `tests/test_refs_1015.py` / `tests/test_fix1015.py` (regression) | 11/11 / 19/19 | `tests/test_refs_ao1015.txt`, `tests/test_fix1015_ao1015.txt` |
- Regression stdout is identical to the refs1015 / fix1015 runs except the mocks' wall-clock `hf_total_s` / `total_seconds`
  (600.8-601.0 s, HF cell timing) and, in test_fix1015, the argparse usage text that now lists the new `--base` choices. The 29 / 30 / 30
  regenerated mock files match the pre-change ones after masking `total_seconds`.
- Negatives (each must FAIL): time-aware driver ran; flip off on worker 1; R7 = the plain V224-exact rows; submission_V224 +1e-3;
  submission +1e-3; S row x0.5; K14 present; fleet not started by the 6.5 h guard (HF_FALLBACK); w_f .30; w_f 0; one S
  manifest_sha16 wrong; guard 5.1 h; a Fracture canary (flrm RM_GUARD / flip V224 CANARY) kept through the blend; notebook base cell
  +1 byte; cross base build json; the FFSR (time-aware) mock under the AO build; --ref-s with a different S row; (flrm) no --ref-flrm.
- HF-AO cell tests: clean path == HF-S byte for byte; receipts differ only by tag / sha / time + fired_before_hf; an RM_GUARD-fired
  Fracture stays 0.5 (HF-S un-blanks it); a V224 CANARY Fracture stays 0.5 in both; not clean + a Fracture canary -> Synovitis and
  Fracture 0.5, every other column == H.

### 4. Runtime review (`python sim_ffs_runtime.py --only-ao --ao-reps 4000` -> `sim_ffs_runtime_ao.txt`; sections A-C unchanged: the old and new scripts at --reps 30 print byte-identical A-C, 347 lines)
Model (MODEL on MEASURED inputs): V224-FLIP always-on flip calibrated to its MEASURED 4.57-4.65 h submit-to-score; RM +0.06-0.16 h
(MODEL, v224rm/FLIPRM_RESULT.md); pre-fleet cells 0.01 h; S-only fleet = 1.5 min fixed + 650 studies x 1.62 s x mf x s (MEASURED
microbench, 5 x TTR224 <= 325 ms; GPU-bound, decode 4 threads ~0.4 s / study); her base guards (R7 deadline 8.3 h, r224 skip 7.83 h)
included. Session model B = section B's (GUESS); STRESS = sd .12 with 9 % x1.3 and 3 % x1.5 sessions (GUESS).

| config (G = start guard) | median | p90 | p99 | P(skip) | P(hard) | submit-to-score median |
|---|---|---|---|---|---|---|
| V224-FLIP as scored (no fleet), B | 4.50 | 5.10 | 6.20 | - | - | 4.66 |
| **FLIP-AO + S, G 6.5**, B / STRESS | 4.85 / 4.93 | 5.51 / 6.20 | **6.62 / 7.31** | .0043 / .038 | 0 / 0 | 5.00 |
| FLIP-AO + S, G 6.0 / 7.0, B | 4.86 / 4.88 | 5.51 / 5.52 | 6.36 / 6.80 | .017 / .0003 | 0 | 5.01 / 5.03 |
| V224-FLRM (no fleet), B | 4.62 | 5.24 | 6.48 | - | - | 4.78 |
| **FLRM-AO + S, G 6.5**, B / STRESS | 4.98 / 5.08 | 5.63 / 6.33 | **6.74 / 7.46** | .0055 / .049 | 0 / 0 | 5.13 |
| FLRM-AO + S, G 6.0 / 7.0, B | 4.97 / 4.98 | 5.60 / 5.64 | 6.41 / 6.84 | .021 / .0015 | 0 | 5.12 / 5.13 |

- Sensitivity at G 6.5 (B): S cost 2.77 / 4.0 s -> p99 6.97 / 7.16 (FLIP), 6.98 / 7.22 (FLRM), P(hard) 0; 1,500 hidden studies ->
  p99 7.12 / 7.10, P(skip) .03-.035; FLIP calibration 4.57 / 4.65 h -> p99 6.55 / 6.65.
- Deterministic (D4): a skip needs a session > 1.44x (FLIP) / 1.40x (FLRM) slower than the measured V224-FLIP run; starting exactly
  at 6.5 h the S fleet ends by 6.87 / 6.97 / 7.02 h at x1.15 x {1, 1.3, 1.45}, and by 7.72 h even at 2.5x the S cost. G 7.0 would
  cut P(skip) to ~.0003-.0016, but its 2.5x-S corner reaches the 8.3 h wall (8.28-8.31 h). Hard bound for any G: fleet wall 8.3 h +
  300 s kill + HF -> kernel <= ~8.39 h, ending in the visible Synovitis canary.
- **Recommendation: G = 6.5 h** (the builder default): every reviewed corner stays <= 8.0 h; p99 6.6-6.8 h (B), <= 7.5 h (STRESS).
  SUPERSEDED by "AO fix 10-15" F1: a wall stop and a skip are the same canary, so 7.0 dominates 6.5; default and recommendation = 7.0.
- The calibration rests on ONE V224-FLIP latency sample (4.57-4.65 h); re-calibrate from the FF20 / FLRM submit-to-score latencies
  when they land (SILVER_PLAN sec 5).

### 5. Open points for main
- Needs the real S dataset + `r7tb_manifest.json` (RUNBOOK_STUDENT steps 4-6) and the student gate PASS; then per kernel: build,
  GPU commit (~10-13 min), fetch, `v224aos_check.py`. Commit the branch actually chosen first (b after a FLIP-stays-best read, c after
  an FLRM-best read); use its output as `--ref-s` for the other if both get committed.
- No DEC (J) cell: the AO-S kernels are V224-FLIP / FLRM + S only (one change, A3 R2 / R3). SILVER_PLAN B2 wants J in every final
  candidate; a J overlay on these kernels is a separate step (jpeg/f20j DEC cell + its U-control).
- The amendment text should name the guard (6.5 h, MIN_START_MIN 108) and the HF-AO Fracture rule before the submit.
- `v224fs_check.py`'s "build: MIN_START_MIN" line still knows only the f / ff budgets; v224aos_check tolerates exactly that line (and
  the exact-mode sub-gate) and replaces both with its own checks, so v224fs_check stays unchanged.

| file | md5 |
|---|---|
| build_v224fs.py | a54afad651ff2bc30e1dd7ce6e55df48 -> 4223cc4c87ccb4db629f5cbab38cebd3 (pre copy `build_v224fs.py.pre_ao_1015`) |
| sim_ffs_runtime.py | 3b67a4204517a9e9096f8d935b431805 -> 8b02804f7fd742025155dbd02642848e (pre copy `sim_ffs_runtime.py.pre_ao_1015`) |
| v224aos_check.py (new) | 31f9f52cd6f5ba432a80018e253b1e4e |
| tests/mock_ao_output.py / tests/test_ao_1015.py (new) | 2e95df7a2e0f3facf783d4e98719f086 / 41182d0ced7455ee1e4d8234e6dd9dfb |
| sim_ffs_runtime_ao.txt | 3323feb4b610cfa048a3de3c71688fe2 |
| tests/standin_v224flips20/rsna_sub_v224flips20.ipynb | sha256 2c2b8e636ab50d13e0e14bed0a9a104f631d611485ec7379562449213f5264f8 |
| tests/standin_v224flrms20/rsna_sub_v224flrms20.ipynb | sha256 f3b6b0f9f5126d1f3533171d83bad20a90c56de5aee88a7d3b3474340d0ecfd7 |

## AO fix 10-15 (Claude fix agent; findings F1 / F2 of the AO review, both verified REAL and fixed)
Laptop CPU only (builder subprocesses, mocks, gate subprocesses, a ~30 s numpy sim, torch CPU for 5 tiny synthetic checkpoints).
Nothing pushed, submitted or created on Kaggle. The box, its queue, the rental kit and scheduled tasks were not touched. No PREREG /
frozen file edited (student_gate.py, PREREG_*.md untouched; v224aos_check.py, v224fs_check.py, v224ffs_check.py, sim_ffs_runtime.py,
tests/mock_fs_output.py, tests/test_refs_1015.py unchanged, md5 below). Pre-change copies `*.pre_aofix_1015`: build_v224fs.py,
make_s_pack.py, RESULT.md, tests/mock_ao_output.py, tests/test_ao_1015.py, r7teacher/RUNBOOK_STUDENT_1015.md, and the dirs
tests/standin_v224flips20, tests/standin_v224flrms20, tests/mock_out_v224flips20, tests/mock_out_v224flrms20.

### F1 (medium): the AO start guard G 6.5 is dominated by G 7.0. REAL, FIXED
- **Verified** (MODEL; `guard_paired_1015.py` -> `guard_paired_1015.txt`; the section D model of sim_ffs_runtime.py loaded unchanged;
  20,000 sessions per row, each replayed under G 6.0 / 6.5 / 7.0 from the same RNG state). G only decides whether the fleet starts:
  the base part is identical across guards in every session. A higher G can therefore only turn a skip into a clean fleet or into a
  wall stop, and a skip and a wall stop end in the same HF_FALLBACK Synovitis canary (fleet wall 8.3 h + 300 s kill, kernel <= 8.39 h,
  the backstop V224-F / FF20 already ship with). Result in all 8 rows: **0 sessions canary at 7.0 but not at 6.5**, P(hard stop) 0,
  P(end > 8.0 h) identical across guards, P(end > 9 h) 0.

| base, session model, hidden studies | P(canary) G 6.0 / 6.5 / **7.0** | kernel end p99 G 6.5 / **7.0** |
|---|---|---|
| FLIP-AO + S, B, 1,300 | .0180 / .0047 / **.0008** | 6.64 / **6.71** |
| FLIP-AO + S, B, 1,500 | .0635 / .0296 / **.0121** | 7.08 / **7.35** |
| FLIP-AO + S, STRESS, 1,300 | .0691 / .0369 / **.0182** | 7.36 / **7.47** |
| FLIP-AO + S, STRESS, 1,500 | .1746 / .0939 / **.0546** | 8.10 / **8.10** (base-driven, same for every G) |
| FLRM-AO + S, B, 1,300 | .0232 / .0067 / **.0014** | 6.72 / **6.83** |
| FLRM-AO + S, B, 1,500 | .0858 / .0385 / **.0183** | 7.27 / **7.43** |
| FLRM-AO + S, STRESS, 1,300 | .0820 / .0457 / **.0233** | 7.54 / **7.54** |
| FLRM-AO + S, STRESS, 1,500 | .2080 / .1146 / **.0674** | 8.22 / **8.22** (base-driven) |

- Why the old reason fails: "7.0's 2.5x-S corner reaches the 8.3 h wall" (D4) compares a wall-stop canary at 7.0 with a session
  that 6.5 has already skipped, which is also a canary. The "AO 10-15" headline P(skip) .004-.005 / .006-.010 quoted session model B
  only; at 6.5 STRESS gives .037 / .046 and 1,500 studies .030 / .038 (MODEL). Concrete case (D1 grid): a session 1.45x slower than
  the measured V224-FLIP run reaches the fleet at ~6.6 h; 6.5 skips it (Synovitis canary: the slot and the S read are lost), 7.0 ends
  clean by ~7.1 h. The fleet cell's own P6 rule (never cut a fleet that could still finish before the wall) points the same way.
- **Fix**: `build_v224fs.py` AO_GUARD_DEFAULT 6.5 -> **7.0** (MIN_START_MIN 78) plus its help / usage / header text; nothing else in
  the builder changed (diff vs `.pre_aofix_1015`: 4 hunks, 1 code line). AO_GUARDS stays (6.0, 6.5, 7.0) in builder and gate (6.0 is
  dominated too: kept only because it was reviewed; do not use it). v224aos_check.py unchanged (it already accepts 7.0 and derives
  MIN_START_MIN from the build json). The f / ff code paths are untouched (test_ao_1015 block D: stand-ins and --s-manifest builds
  byte-identical, below).
- AO stand-ins rebuilt at the new default: `tests/standin_v224flips20` nb sha256 c2ea01a670b91b54... (was 2c2b8e63...),
  `tests/standin_v224flrms20` a77f66d6e469164d... (was f3b6b0f9...). Each notebook differs from its 6.5 version in exactly 2 lines
  (SAVE-AO knob line 6.5 / 108 -> 7.0 / 78; K2_CFG MIN_START_MIN 108 -> 78); BUILD json: start_budget_h / min_start_min / ao_guard_h,
  notebook sha, fleet_cells_from / fleet_cells_sha256 (V224-F fleet cells at 7.0).
- `tests/mock_ao_output.py` reads the guard from the stand-in BUILD json (it hard-coded 6.5 / 108: one assert, the not-started
  negative's numbers, the guard-5.1 negative's substitution, one label). `tests/test_ao_1015.py`: B expects the default 7.0 / 78;
  C now builds 6.5 (108) and 6.0 (138) (C used to build 7.0).
- Amendment text (main): name G **7.0 h (MIN_START_MIN 78)** and quote P(canary) .0008 / .0014 (B), .018 / .023 (STRESS), .012 /
  .018 (B, 1,500 studies) and p99 6.71 / 6.83 h (B), 7.47 / 7.54 h (STRESS) for FLIP-AO + S / FLRM-AO + S (MODEL, one FLIP latency
  sample).

### F2 (medium): the operator path still built the FF line. REAL, FIXED
- **Verified**: make_s_pack.py's printed NEXT block (pre lines 118-125) said build `--base ff` and gate `v224ffs_check.py`;
  RUNBOOK_STUDENT_1015 Step 7 built `--base ff` and said "Only `--wf 0.20 --base ff` is pre-registered" and "Do NOT build `--replace`
  now"; Step 13 built `--base ff --replace`; 0 occurrences of `--base flip`, `--base flrm` or `v224aos` in either file. Under
  PREREG_10-15 A3 R2 / R3 branches b (FLIP stays best) and c (FLRM best) need the AO kernels; following the old text would build
  kernel_v224ffs20 / ffsr20, which v224ffs_check.py PASSes (right gate for that kernel), i.e. the superseded two-change form.
- **Fix**: make_s_pack.py NEXT text = the A2 + A3 branch table, one command per line: a FF-S (`--base ff`, v224ffs_check, REPLACE
  only after a positive FF-S read); b FLIP-AO + S (`--base flip --start-budget-h 7.0`, `v224aos_check.py ... --ref-exact --ref-flip
  [--ref-s]`, "never --base ff --replace here"); c FLRM-AO + S (`--base flrm --start-budget-h 7.0`, `... --ref-flrm
  session_2026-10-15/v224rm/out_v224flrm_v1`, required); d FF20 canary -> no FF-S / FFSR; e gate FAIL -> no entry. The pack logic
  is unchanged (only the header comment and the final print). md5 5b2df773 -> f31f4702.
- RUNBOOK_STUDENT_1015.md: Step 4 md5 line (f31f4702) + note; Step 7 = branch table + b / c code block, the two pitfall lines
  rewritten (only `--wf 0.20` is pre-registered; `--base ff --replace` only for branch a after a positive FF-S read; b / c imply
  REPLACE); b / c notes in Steps 8 (kernel dir / slug / out dir), 9 (v224aos_check, PASS 28/28), 10 (runtime numbers above), 11 (A3
  entry text items), 12 (slugs; the pre-A3 base-question bullets marked superseded) and 13 (no REPLACE step; A3 R1 read rule).

### Tests (all MEASURED, laptop)
| test | result | log |
|---|---|---|
| `tests/test_aofix_1015.py` (new; F1 default builds, make_s_pack run end to end on a synthetic 5-fold T_b student, its b / c build lines run verbatim on its own manifest, its b / c gate lines run verbatim on the AO mocks, RUNBOOK text) | fixed files: **TEST_AOFIX_1015 PASS 15/15** ([FIX] 10/10, [INV] 5/5) | `tests/test_aofix_1015_post.txt` |
| same, `pre` mode on the `.pre_aofix_1015` copies | **[FIX] 0/10 pass** (every fixed case FAILs before the fix), [INV] 5/5 | `tests/test_aofix_1015_pre.txt` |
| `tests/mock_ao_output.py flips` (G 7.0 stand-in) | clean 28/28, --ref-s 29/29, 17/17 negatives FAIL, CELL 6/6 -> **MOCK_AO_GATE PASS 19/19** | `tests/mock_ao_output_flips_aofix1015.txt` |
| `tests/mock_ao_output.py flrms` (G 7.0 stand-in) | clean 28/28, --ref-s 29/29, 18/18 negatives FAIL, CELL 6/6 -> **PASS 20/20** | `tests/mock_ao_output_flrms_aofix1015.txt` |
| `tests/test_ao_1015.py` (updated expectations) | **TEST_AO_1015 PASS 41/41** (incl. block D: f / ff stand-ins == pre-change sha, --s-manifest f / ff builds == pre_ao_1015 builder byte for byte) | `tests/test_ao_1015_aofix1015.txt` |
| regression `tests/mock_fs_output.py f / ff / ffsr` | 32/32 + 8/8; 14/14 + 8/8; 14/14 + 19/19 (GAP probe PASS as known) | `tests/mock_fs_output_{f,ff,ffsr}_aofix1015.txt` |
| regression `tests/test_refs_1015.py` / `tests/test_fix1015.py` | 11/11 / 19/19 | `tests/test_refs_aofix1015.txt`, `tests/test_fix1015_aofix1015.txt` |
| `guard_paired_1015.py --n 20000` (MODEL) | table above | `guard_paired_1015.txt` |
- Every gate line of the new AO mock logs == the 6.5 logs after masking the guard values, except the 2 expected ones (the
  reviewed-guard tuple print and the notebook sha). The regression logs (mock_fs f / ff / ffsr, test_refs, test_fix1015) == the
  ao1015 logs after masking temp-dir names and wall-clock seconds (0 differing lines).

### Final state
- **ready_for_commit: false**, for the same reason as before and not because of F1 / F2: the S dataset and
  `v224f/s_upload/rsna-knee-cnxt-fleet-r7tb/r7tb_manifest.json` do not exist yet and the student gate PASS is still needed. The
  stand-ins stay DO_NOT_PUSH.
- Build (main, branch b): `python session_2026-10-15/v224f/build_v224fs.py --wf 0.20 --base flip --start-budget-h 7.0 --s-manifest
  session_2026-10-15/v224f/s_upload/rsna-knee-cnxt-fleet-r7tb/r7tb_manifest.json` -> `kernel_v224flips20` (branch c: `--base flrm`
  -> `kernel_v224flrms20`). Expected `BUILT ... MIN_START_MIN 78.0 standin False`.
- Gate (after the GPU commit): `python session_2026-10-15/v224f/v224aos_check.py <OUT_FLIPS20> --wf 0.20 --build
  session_2026-10-15/v224f/kernel_v224flips20/BUILD_V224FS.json --ref-exact session_2026-10-14/v224/out_v224_v1 --ref-flip
  session_2026-10-14/v224/out_v224flip_v1 [--ref-s <an output with the same 5 S>]` -> `V224AOS_CHECK PASS 28/28` (29/29 with
  --ref-s). FLRM: `kernel_v224flrms20/BUILD_V224FS.json` and `--ref-flrm session_2026-10-15/v224rm/out_v224flrm_v1` (required).
- Unchanged open points: no DEC (J) cell in the AO-S kernels (SILVER_PLAN B2, a separate overlay); the runtime model rests on one
  V224-FLIP latency sample (re-calibrate from the FF20 / FLRM submit-to-score latencies).

| file | md5 |
|---|---|
| build_v224fs.py | 4223cc4c87ccb4db629f5cbab38cebd3 -> 241333f6e5e2027a0d3a01a2c7e5c1ef (pre copy `.pre_aofix_1015`) |
| make_s_pack.py | 5b2df773586e23fb3c5b5e441e240c1c -> f31f4702e11a9192c01d6df8b48cbfcc |
| r7teacher/RUNBOOK_STUDENT_1015.md | 567cf1b5c226f0aaf337a2f82ea75b52 -> cc142f216e1f67473037c8bc89b7a5e1 |
| tests/mock_ao_output.py | 2e95df7a2e0f3facf783d4e98719f086 -> 1fbc39047b1988dd9c096104304f109b |
| tests/test_ao_1015.py | 41182d0ced7455ee1e4d8234e6dd9dfb -> 84d11642fc1c40c250018b0fcc508bae |
| tests/test_aofix_1015.py (new) / guard_paired_1015.py (new) | 34f783500526dabc6da14474b2932393 / 671f55faddc525be5a03e0d83ceb00dc |
| unchanged: v224aos_check.py / v224fs_check.py / v224ffs_check.py / sim_ffs_runtime.py | 31f9f52c / 9a4ad8bf / 2228b880 / 8b02804f |
| tests/standin_v224flips20/rsna_sub_v224flips20.ipynb | sha256 c2ea01a670b91b54c5b7f1b2be8bcf5e957d45a9f4e04ff887db302783b3f6f1 |
| tests/standin_v224flrms20/rsna_sub_v224flrms20.ipynb | sha256 a77f66d6e469164d0fcf722105fd53b706ed15975039236bb49a2b2d2f29d78b |
