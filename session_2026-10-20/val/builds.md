# builds.md: Track C, A22 candidate builds and offline validation (10-20 ~19:10 IST, Claude)

**Scope.** Local builds and offline checks only.
- Nothing was pushed, committed or submitted. No dataset was created, nothing was downloaded, no GPU or box job ran.
- No scheduled task was created or edited. `tick_a22.py` is written but NOT registered, and it was only run with `--selftest`
  (offline, no Kaggle import).
- PREREG, HANDOFF, PROGRESS, `a20/`, `a21/` and `push/` were only read.
- No OAI data was touched. Every number below comes from the gold58 serve files or the committed visible-test outputs on disk.
- My files are `build_val.*`, `gate_val_offline.*`, `selftest_commit_gate.*`, `run_gate.py`, `slate_a22.json`, `tick_a22.*`,
  `a22_selftest.log`, `kernels/`, `sim/` and this file. The other files in `val/` belong to other tracks.

**Tags.** MEASURED = computed here (log named). CLAIMED = from an older note (source named). MODEL = arithmetic. GUESS = judgement.

## 1. Result

| key | kernel dir (val/kernels unless noted) | notebook sha | base | E-OA | gold58 vs base (INFO) | offline gate | commit gate on sim |
|---|---|---|---|---|---|---|---|
| pw35 | kernel_v224r6c4pw35j (== push/shape, same bytes) | e65c0f2b | c4 b47c29c3 | pw .35 | +.00256 | PASS | PASS 17/17 |
| pw40 | kernel_v224r6c4pw40j | 353f0742 | c4 | pw .40 | +.00190 | PASS | PASS 17/17 |
| pw45 | kernel_v224r6c4pw45j | d18ab4af | c4 | pw .45 | +.00157 | PASS | PASS 17/17 |
| r6pw30 | kernel_v224r6flips6fpw30j | 27ce4be7 | R6 dfee2ba3 | pw .30 | +.00191 vs R6 | PASS | PASS 17/17 |
| r6e20 (existing) | push/kernel_v224r6flips6feoa20j | 2a5ef93a | R6 | sym .20 | +.00218 vs R6 | PASS (push gate + here) | PASS 17/17 (A20 gate) |
| controls | kernel_v224r6c4pw00j / kernel_v224r6flips6fpw00j | 2cbca90f / 09e307d2 | c4 / R6 | 0 | 0 | byte-identical | not for push |
| parity | kernel_v224r6c4pw30j | 5a903187 | c4 | pw .30 | +.00223 | == scored pw30 | not for push |

- **VAL_OFFLINE_GATE PASS** (MEASURED `gate_val_offline.log`, 26 of 26 checks).
- **COMMIT_GATE_SELFTEST PASS**: 15 of 15 rows, with 7 expected PASS and 8 expected FAIL (MEASURED `selftest_commit_gate.log`).
- **A22_TICK_SELFTEST PASS 22/22** (MEASURED `a22_selftest.log`).
- The gold58 column is info only, because gold58 is not a ruler.

## 2. Builds (MEASURED `build_val.log`, receipts `kernels/build_receipt_<tag>.json`)

`build_val.py {c4|r6} A_OA` is one builder for both bases.
- **The block.** It inserts the A21 pair-weighted 14-line block, copied verbatim from `a21/build_a21.py` (mode 'pw'). The anchor is
  the same: cell 18 `_s6f_blend`, S6F path, after the blanked-column fill.
- **Asserts.** The base sha is asserted. The only changed cell is 18, and the lines around the insert are unchanged.

**c4 base.**
- pw00 and pw30 rebuild byte-identical to the a21 kernels: notebook AND kernel-metadata.json, PARITY asserted.
- pw35 rebuilds byte-identical to `push/shape/kernel_v224r6c4pw35j`. So "reuse push/shape" and "push val/kernels" are the same
  bytes.
- pw40 and pw45 are new. Compared with c4, they differ only in cell 18, plus metadata id / title / code_file.

**R6 base (`v224r6flips6fpw%02dj`).**
- This is push/build_r6eoa.py with the pw block in place of the A20 sym block.
- XREF asserted: R6pw30 differs from the SCORED pw30 notebook only in cell 14, exactly as R6 differs from c4 (cells [14]). Its cell 18
  equals pw30's byte for byte.
- So both code paths have already run on the hidden set: R6 57248084 (cell 14) and pw30 57425970 (cell 18). MEASURED: build log.
  "Already ran" is CLAIMED from LB reads.
- Kernel id is `eeshsaxena/rsna-sub-v224r6flips6fpw30j`. Datasets, docker image and T4 are the same as R6.

## 3. Offline gate (MEASURED `gate_val_offline.log`)

The gate runs each notebook's own cell-18 code on two kinds of input.
- **gold58 (n 58).** The stage 3 is rebuilt as c4 (C4 table, A224 .25) or as R6 (uniform W .5, A224 .35). S6fall = n42gpu.
- **Committed Kaggle inputs.** These are each run's `submission_v224flip.csv` + `s6f/B/B_probs.csv` + its own setup / run dicts from
  `s6f_receipt.json`, for c4 57343435, scored pw30 57425970 and R6 57248084 (3 public rows).
- Kaggle writes LF and pandas on Windows writes CRLF, so replay bytes are compared after CRLF -> LF.

| gate | check | result |
|---|---|---|
| G1 / G2 | strength-0 controls, gold58 output bytes == c4 / == R6 | OK / OK |
| G6 | each base's own cell 18, replayed, == its committed submission.csv (sha 4ad325dd c4, e1324ecc R6, both == receipt) | OK |
| G6 | strength-0 controls on the committed inputs == the c4 / R6 committed submission.csv | OK / OK |
| G3a | val pw30 notebook sha == scored pw30 notebook (5a903187), metadata identical | OK |
| G3b | val pw30 gold58 bytes == scored pw30 notebook gold58 bytes | OK |
| G3c | val pw30 on the SCORED pw30 run's inputs == its downloaded submission.csv (sha 5b06fff9 == its receipt) | OK |
| G3d | replay receipt == scored run's s6f receipt on 12 keys (a_oa .3, oa_mode pw, rho_mean .7805, coverage 1.0, ...) | OK |
| G3e | c4 and pw30 committed inputs are the same files (stage-3 dcaabfb7, B 7975394d); pw30 on c4 inputs == scored output | OK |
| G4a | val pw35 == push/shape pw35 (notebook, metadata, gold58 bytes) | OK |
| G5 | gold58 OA columns == independent scipy recompute from the control output, max diff 2.8e-8; non-OA identical; receipt a_oa / oa_mode / S6F (pw30, pw35, pw40, pw45, R6pw30) | OK x5 |
| G7 | same on the committed inputs, max diff 1.3e-8 (pw30, pw35, pw40, pw45, R6pw30) | OK x5 |
| G8 | gold58 deltas == the given pw curve (.3 +.00223, .35 +.00256, .4 +.00190, .45 +.00157), within 1e-5 | OK x4 |

**gold58 (INFO ONLY, MEASURED)**

| entry | gold58 macro | vs base | Medial / Lateral / PF OA AUC | paired boot vs ref (B 2000) |
|---|---|---|---|---|
| c4 | .94113 | 0 | .9891 / .8433 / .9009 | |
| pw30 (scored best-id) | .94336 | +.00223 | .9876 / .8511 / .9215 | |
| pw35 | .94369 | +.00256 | .9845 / .8530 / .9266 | vs pw30 +.00034 sd .00089, P>0 .64 |
| pw40 | .94303 | +.00190 | .9829 / .8414 / .9318 | vs pw30 -.00038 sd .00136, P>0 .38 |
| pw45 | .94270 | +.00157 | .9829 / .8375 / .9318 | vs pw30 -.00063 sd .00178, P>0 .36 |
| R6 | .93960 | 0 | .9891 / .8240 / .8977 | |
| R6e20 | .94178 | +.00218 | .9876 / .8356 / .9138 | |
| R6pw30 | .94151 | +.00191 | .9876 / .8259 / .9202 | vs R6 +.00197 sd .00211, P>0 .83; vs R6e20 -.00025 sd .00107, P>0 .39 |

- From pw40 on, Lateral OA falls below c4: .8414 at pw40 vs .8433 for c4. PF OA keeps rising up to .40 and is flat at .45.
- None of these gold58 differences is resolved, since every sd is >= the gap. They are info only, not a pick signal.

## 4. Finding: the 3-row public output cannot tell the doses apart (MEASURED)

- On the 3 visible test rows, **pw30, pw35, pw40 and pw45 give identical OA outputs**. Only c4 differs from them.
- So a commit gate's math check (d) passes for any dose. Only check (d2) catches a wrong dose: the receipt's a_oa / oa_mode plus
  EOA_APPLIED in the log.
- The negative rows below confirm this. A pw40 gate on a pw35 output fails 16/17 on (d2) only, and so does a pw35 gate on pw45.
- The hidden-set dose identity therefore rests on three things:
  1. the notebook sha check in the tick (G3 guard: registry sha == the dir that is pushed);
  2. the receipt;
  3. this file's gold58 recompute.

**Commit-gate self-test matrix (MEASURED `selftest_commit_gate.log`).**
- A "sim" is the control's REAL committed dir with the new cell 18 replayed on it, plus an EOA_APPLIED log line. It is marked
  SIMULATED in its receipt.

| row | expect | got |
|---|---|---|
| pw35 / pw40 / pw45 gate on its sim | PASS | PASS 17/17 x3 |
| r6pw30 gate on R6pw30 sim | PASS | PASS 17/17 |
| r6e20 gate (A20 `probe_check_eoa.py`) on R6e20 sim | PASS | PASS 17/17 |
| a21 pw30 gate on the REAL scored pw30 dir, and on the pw30-on-c4 sim | PASS | PASS 17/17 x2 |
| pw35 gate on the REAL c4 dir; r6pw30 gate on the REAL R6 dir (no E-OA) | FAIL | 15/17 x2, (d) maxdiff .067 / .133 |
| dose swaps (pw40 gate on pw35 sim, pw35 gate on pw45 sim) | FAIL | 16/17, (d2) only |
| base swaps (c4 params on R6pw30, R6 params on pw35) | FAIL | 15/17, (a) table + (c) |
| shape swap (r6pw30 with --mode sym) | FAIL | 16/17, (d2) |
| a21 gate --mode sym on R6e20 | FAIL | 16/17, (d2): the A20 block writes no oa_mode, so r6e20 MUST use the A20 gate |

- The R6-param gate (a224 .35, table {MOA .5, LOA .5, PF .5}) passes (a), (b) and (c) on a real R6 committed dir. So the R6 entry is
  not blocked by the base checks (MEASURED).

## 5. Commit-gate command lines (run on the fetched committed output dir `<OUT>`; each must print `PROBE_CHECK PASS 17/17`)

The shell-safe form uses `run_gate.py`, which reads the exact argv from `slate_a22.json` so there is no table quoting:
```
py -3.12 -I session_2026-10-20/val/run_gate.py pw35   <OUT>
py -3.12 -I session_2026-10-20/val/run_gate.py pw40   <OUT>
py -3.12 -I session_2026-10-20/val/run_gate.py pw45   <OUT>
py -3.12 -I session_2026-10-20/val/run_gate.py r6pw30 <OUT>
py -3.12 -I session_2026-10-20/val/run_gate.py r6e20  <OUT>
```
The raw argv are below. C4T = `{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}`
and R6T = `{"Medial OA": 0.5, "Lateral OA": 0.5, "PF OA": 0.5}`.
- pw35: `a21/probe_check_eoa21.py <OUT> --a224 0.25 --table C4T --aoa 0.35 --mode pw`
- pw40: `a21/probe_check_eoa21.py <OUT> --a224 0.25 --table C4T --aoa 0.4 --mode pw`
- pw45: `a21/probe_check_eoa21.py <OUT> --a224 0.25 --table C4T --aoa 0.45 --mode pw`
- r6pw30: `a21/probe_check_eoa21.py <OUT> --a224 0.35 --table R6T --aoa 0.3 --mode pw`
- r6e20: `a20/probe_check_eoa.py <OUT> --a224 0.35 --table R6T --aoa 0.2` (matches CLAIMED bpick.md sec 5)

## 6. Tick a22 (`tick_a22.py` + `tick_a22.cmd`; WRITTEN, NOT REGISTERED)

The pattern is `a21/tick_a21.py`. One idempotent tick does: push, then wait for COMPLETE, then fetch (502-safe), then gate, then submit
once. State lives in `val/a22/`. It needs all of the following guards:
1. **GO file** `session_2026-10-20/val/A22_GO.txt`. Its first line is `GO <key> [<key> ...] [MAX=n]`, with keys from
   `slate_a22.json` (pw35 pw40 pw45 r6pw30 r6e20). The key order is the push / submit priority, and free text may follow the keys.
   - An unknown key invalidates the whole GO, so a typo cannot drop or swap an entry.
   - A lower-case `go` is invalid.
2. **PREREG.** `session_2026-10-15/PREREG_10-15.md` must contain `AMENDMENT 22`. It does not today, so a live tick would do nothing.
3. **Registry sha.** The kernel dir's notebook + metadata sha256 must equal the registry.
4. **Push.** It pushes only when fewer than 2 GPU commits are running (`running.py`). The pushed version is parsed from the push log and
   is the version it later submits; a21 hard-coded v1.
5. **Gate.** It runs the registry's exact gate argv. It submits only on `PROBE_CHECK PASS 17/17`, once per entry, deduped by the
   `A22-XXX:` description prefix on Kaggle or by the slug in submits.log.
6. **Window.** Submits happen from **10-21 05:31 IST** to 10-22 05:20 IST, i.e. the 10-21 UTC day. Pushes may happen as soon as
   guards 1-3 hold, so the commits are COMPLETE by 05:31.
   - Daily cap: MAX (default 5) minus that UTC day's submissions, taking the larger of the Kaggle `date` count and the submits.log
     count.

**Selftest (`--selftest`).** It is offline and imports nothing from kaggle. 22 of 22 checks pass:
- the registry shas;
- GO parsing, including typo and lower-case rejection;
- the window edges at 05:30:59 and 05:31:00;
- the UTC-day mapping;
- the cap count on the real submits.log: 10-20 UTC day = 5, 10-21 = 0;
- the API date parser;
- the PREREG and GO guards (both currently absent);
- every entry's gate argv on its sim, all PASS 17/17.

**To arm (main + Eesh; NOT done):**
1. Write PREREG Amendment 22, naming the slate and the B rule.
2. Write `val/A22_GO.txt`, for example `GO r6pw30 pw35 MAX=2 Eesh 10-21 hh:mm "..."`.
3. Register `val/tick_a22.cmd` every 10 min, plus a one-shot at 05:31:15 IST 10-21, for example
   `schtasks /Create /TN RSNA_A22_tick /SC MINUTE /MO 10 /TR "C:\Users\eeshs\Downloads\rsna-knee\session_2026-10-20\val\tick_a22.cmd"`.
   Delete the task after the slate.

**Timing.**
- Commits take ~20 min and 2 can run at once (CLAIMED HANDOFF 10-20 00:55). So 2 entries need ~20-25 min and 4 need ~45 min.
- For a 05:31 submit, GO + amendment are needed by ~04:30 IST (MODEL).
- Scoring takes ~5.5-6 h (CLAIMED bpick.md sec 6), so a 05:31 submit reads ~11:00-11:30 IST 10-21.

## 7. Context for the pick (not decided here)
- Older notes rated pw35 / pw40 negative after A21 (CLAIMED census.md row 11, lbstrat 6.2, PUSH_1020 sec 4). Those were written before
  the 18:50 read made pw30 the best-id. HANDOFF 18:55 lists pw35 / pw40 as "trend" options.
- On the gold58 curve, pw35 sits at a local bump: +.00034 over pw30, P>0 .64. pw40 and pw45 are below pw30. None of it is resolved
  (MEASURED, info only).
- R6pw30 vs R6e20 as B: gold58 -.00025 (P>0 .39, info). Its cell 18 is the code of the current best-id (pw30), while R6e20's is the
  code of eoa20. Choosing between them is a pick-model question (bpick / Track B), not a build question.
- Both are built and gated, and both are in the tick registry.
