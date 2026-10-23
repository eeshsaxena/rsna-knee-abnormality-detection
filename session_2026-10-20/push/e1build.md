# e1build.md: Track 4, E1 soft-OR pow2 stacked on E-OA, built and gated LOCALLY (10-20 ~12:30-15:00 IST, Claude)

**Scope.** This was a local build only.
- Nothing was pushed, committed or submitted, and no dataset was created. Nothing was downloaded.
- No scheduled task, PREREG, HANDOFF or PROGRESS file was touched. session_2026-10-19/a20 and session_2026-10-20/a21 were
  only read.
- No GPU and no box job was used. No OAI data was touched: every number here comes from gold58 serve files, the committed
  visible-test outputs, or a simulation.
- Everything is under `session_2026-10-20/push/e1/`.

**Tags.**
- MEASURED: computed in this session, with the log named.
- CLAIMED: from an older note, with the source named.
- MODEL: arithmetic or simulation on stated inputs.
- GUESS: judgement.

## 1. Result

1. **The build is done and the gate passes.** The candidate is `push/e1/kernel_v224r6c4eoa20e1j` (Kaggle id
   `eeshsaxena/rsna-sub-v224r6c4eoa20e1j`, notebook sha `fd232289...`). It is c4 (`b47c29c3`, asserted) plus E1 in cells
   14 and 18 plus the exact 9-line A20 E-OA block (sym, a .2).
   - It was built two independent ways and both produced the same sha `fd232289` (MEASURED `build_e1eoa.log`,
     `alt/build_e1_on.log`):
     - c4 + E-OA block + E1.
     - The shipped eoa20 notebook (`d40f43a4`) + E1.
2. **The offline gate passes 22 of 22 checks** (MEASURED `gate_e1_offline.log`). The gate executes the notebooks' own cells
   14 / 15 / 18 on the gold58 serve files.
   - Control E1 1 + a 0 is byte-identical to c4.
   - Control E1 1 + a .2 is byte-identical to the scored eoa20 path.
   - E1 alone reads .94111, matching the target .94111.
   - The candidate reads .94350, matching the target .94350.
   - An independent numpy recompute agrees to 3e-8.
3. **The draft commit gate `probe_check_e1eoa.py` is written and self-tested.** It has 18 checks, and all 17 rows of its
   self-test matrix match their pre-stated expectation (MEASURED `selftest/selftest_gate.log`).
4. **New, info only: E1's gold58 "+.00009" was a float64-screen artifact.** In the shipped float32 code path:
   - E1 alone vs c4 is -.00002.
   - E1 + eoa20 vs eoa20 is -.00010 (sd .00097, P>0 .45).

   (MEASURED `info_f32_ties.log`, `gate_e1_offline.log`.) gold58 is not a ruler, so this does not kill E1. It does remove
   the last weak positive that gold58 offered. The MORE_1020 rating (WEAK, contingency only, not recommended) stands. If
   anything, it shades slightly lower (GUESS).
5. **Contingency builds for every possible A21 outcome are also built and gated, 9 of 9 checks** (MEASURED
   `alt/gate_e1_alt.log`). If the ~17:00-19:00 IST read moves the API best-id to eoa30, pw20 or pw30, the matching E1 kernel
   is ready (sec 5).

## 2. The edit (MEASURED `build_receipt_v224r6c4eoa20e1j.json`)

**Cell 14 (stage 3).** Two lines are inserted and two are replaced:

| change | line |
|---|---|
| insert after `W_RAPTOR_C` | `E1 = 2.0   # [E1 soft-OR pow2, THINK_1019 8.4] ...` |
| replace | `rr, ro = pct_on(raw[c].values, ~rap_fail) ** E1, pct_on(ours[c].values, ~our_fail) ** E1` |
| replace | `rr = (1.0 - A224) * rr + A224 * pct_on(r224[c].values, ~rap_fail) ** E1` |
| insert before the receipt is written | `blend['e1'] = E1` |

- There is no re-rank of the Raptor mix.
- The fills are untouched: NaN ** 2 stays NaN on failed rows.
- The both-failed 0.5 fill is now on the squared scale. It is dormant (CLAIMED MORE_1020 sec 3.1).

**Cell 18 (`_s6f_blend`).** Two lines are inserted, and the E-OA block follows:

| change | line |
|---|---|
| insert after `rec = {...}` | `E1 = 2.0; rec['e1'] = E1` |
| insert right after `rs[~ok] = rb[~ok]` | `rb, rs = rb ** E1, rs ** E1` |

- The rho sanity floor is computed before the squaring and is rank-invariant.
- After the blank fill comes the 9-line A20 E-OA block, verbatim, at the same anchor.

**Builder asserts** (`build_e1eoa.py E1 A_OA`):
- The base sha is c4.
- The changed cells are exactly [14, 18], and every other cell is byte-equal to c4.
- Cell 14 with the E1 lines reverted equals c4's cell 14.
- Cell 18 with the 2 E1 lines removed equals the A20 builder's cell 18 and the shipped a20 cell 18. That is sha `d40f43a4`
  for a .2 and `b9cab541` for a 0.
- A receipt records the src and out sha and every inserted or replaced line.

## 3. Offline gate (gold58 replay of the notebook cells; MEASURED `gate_e1_offline.log`, 22/22 OK)

**Harness.**
- Cell 14 runs on these inputs:
  - raptor_raw = raptor_sub = r384 kernel serve.
  - raptor224 = r224 kernel serve.
  - ours = worker predictions = R6 serve.
- Cell 15 is the V224 canary, which stays clean.
- Cell 16's freeze is emulated (byte copy + sha).
- Cell 18 runs with S6fall = n42gpu.
- The harness was also proven against Kaggle itself (`selftest/selftest_replay.log`, T0 / T4). Replaying the shipped
  eoa20 and pw20 notebooks on their own commit's visible-test members reproduces both commits' `submission_v224flip.csv`
  and `submission.csv` byte-for-byte, after LF normalisation; maxdiff is 0.

| check | result |
|---|---|
| H1 c4 replay stage 3 == the A20/A21 gates' numpy V224 construction | byte-identical |
| H2 / H3 c4 / eoa20 kernel-path macro == gate_a20/a21 logs | .94113 / .94360 |
| H4 clean path on all 6 replays (canary none fired, S6F mode, frozen base) | OK |
| **G0a** E1 1 + a 0 vs c4 | **byte-identical** (stage 3 + final) |
| **G0b** E1 1 + a .2 vs the scored a20 eoa20 | **byte-identical** (stage 3 + final) |
| G0c control receipts (e1 1; a_oa only in the a .2 control) | OK |
| G1a E1 alone, target .94111 +- 1e-5 (THINK 8.4 / local_mech l.2) | .94111 |
| G1b candidate, target .94350 +- 1e-5 (local_mech l.10 (2, FINAL, .2)) | .94350 |
| G1c independent numpy recompute (E1 2 a 0, E1 2 a .2, c4, eoa20) | maxdiff <= 3.0e-8 at stage 3 and at the final |
| G1d candidate stage 3 and non-OA columns == E1 alone | identical |
| G1e candidate OA trio == E-OA sym .2 applied to the E1-alone output | maxdiff 2.8e-8 |
| G1f receipts: blend e1 2, s6f e1 2, a_oa .2, mode S6F, fired [] | OK |
| G1g E1 is live (rank order differs from eoa20) | 12 of 12 columns |

**gold58 numbers.** These are information only, in the kernel path, and never a selection ruler:

| entry | macro | vs c4 |
|---|---|---|
| c4 | .94113 | 0 |
| eoa20 (scored) | .94360 | +.00247 |
| c4 + E1 | .94111 | -.00002 |
| eoa20 + E1 (candidate) | .94350 | +.00237 |

- E1 per label vs c4:
  - Gains: ACL +.0012, Lateral OA +.0019, PF OA +.0013, Effusion +.0012, Baker's +.0009, Fracture +.0021.
  - Losses: MCL -.0023, Medial Meniscus -.0012, Lateral Meniscus -.0019, Synovitis -.0036.
  - Unchanged: Medial OA 0, Contusion 0.
- Bootstrap, 2000 resamples, seed 7:
  - candidate - eoa20: -.00010 (sd .00097, P>0 .454).
  - E1 - c4: -.00004 (sd .00083, P .484).

## 4. New finding: rank ties, float32, and what E1 really does on gold58

**MEASURED (`info_f32_ties.log`).** c4's stage 3 and final are linear mixes of rational percentiles, so different studies
often land on exactly the same value. At n 58 the tied rows are:

| path | stage 3 | final |
|---|---|---|
| c4, float64 | 28 | 33 |
| c4, float32 kernel (the cast merges near-ties) | 39 | 54 |
| E1 (squares) | 1 | 3 |

That gives:

| comparison | float64 screens (m14, e1_stages, local_mech) | shipped float32 path |
|---|---|---|
| c4 | .94102 | .94113 |
| eoa20 | .94341 | .94360 |
| E1 - c4 | +.00009 | -.00002 |
| (E1 + eoa20) - eoa20 | +.00009 | -.00010 |

- The c4 / eoa20 gaps come from Lateral Meniscus +.0006 and Fracture +.0007, plus Lateral OA +.0010 for eoa20.
- The E1 rows are identical in both paths, because the squares remove the ties. So the whole sign flip sits in the
  baseline.

**MODEL (`info_hidden_ties.log`: Gaussian-copula members at the gold58 Spearman range, prevalence 10 %).** c4's float32
path keeps about this many tied rows per column:

| n | tied rows per column | max AUC move if every tied pos-neg pair were broken correctly |
|---|---|---|
| 390 (public) | ~68 | 1.8e-4 per label |
| 910 (private) | ~160 | 8e-5 per label |

E1 leaves about 2 tied rows. So part of E1's mechanism is a deterministic tie-break toward "one member confident". On
hidden that side effect is bounded by about 1e-4 per label. A realistic size is a fraction of that, of either sign (MODEL).

**Takeaway.** The tie artifact is a small-n gold58 effect. It does not change the hidden-score prior in MORE_1020 (+.00003
to +.0001, P about .55; MODEL / GUESS). It does mean no gold58 replay supports E1. Its only positive evidence is still the
report-label OOF ruler: +.00022 to +.00045 when stacked (CLAIMED `local_mech.log`).

## 5. Contingency builds for the A21 read (MEASURED `alt/gate_e1_alt.log`, 9/9 OK)

`build_e1_on.py E1 BASE_TAG` starts from the SCORED notebook itself. It asserts:
- The notebook sha matches its A21 build receipt.
- The receipt's src is c4.
- The base differs from c4 only in cell 18.

It then inserts the same E1 lines. Per base, the gate checks:
- K0: the base replay macro matches the a21 gate log.
- K1: E1 1 is byte-identical to the scored base.
- K2: stage 3 and the non-OA columns equal E1 alone, the OA trio equals that base's E-OA formula applied to E1 alone
  (maxdiff 2.8e-8), and the receipts are right.

| API best-id after the A21 read | kernel to push (if Eesh says go) | notebook sha | commit-gate args | gold58 base -> +E1 (info) |
|---|---|---|---|---|
| eoa20 stays | `push/e1/kernel_v224r6c4eoa20e1j` | fd232289 | `--aoa 0.2 --mode sym --e1 2` | .94360 -> .94350 (-.00010) |
| eoa30 | `push/e1/alt/kernel_v224r6c4eoa30e1j` | df652aef | `--aoa 0.3 --mode sym --e1 2` | .94313 -> .94342 (+.00029) |
| pw20 | `push/e1/alt/kernel_v224r6c4pw20e1j` | 402f34af | `--aoa 0.2 --mode pw --e1 2` | .94378 -> .94349 (-.00029) |
| pw30 | `push/e1/alt/kernel_v224r6c4pw30e1j` | 4a0eed92 | `--aoa 0.3 --mode pw --e1 2` | .94336 -> .94347 (+.00011) |

- Every commit-gate call also needs `--a224 0.25 --table '<c4 table>'`, as tick_a20 passes them.
- The gold58 E1 increment averages about 0 over the 4 bases (MEASURED). That is noise, as expected.
- **Gate controls only, never push.** Each of these has a `DO_NOT_PUSH.txt`:
  - `kernel_v224r6c4eoa00e1x1j` and `kernel_v224r6c4eoa20e1x1j`.
  - `alt/*e1x1j`.
- `kernel_v224r6c4eoa00e1j` (c4 + E1, sha 7a23644b) is THINK 8.4's exact entry. It applies only if E-OA were closed, which
  it is not (eoa20 moved the best-id).

## 6. Commit gate draft: `push/e1/probe_check_e1eoa.py`

It is `a20/probe_check_eoa.py` plus the following changes:
- `--e1`, default 1.
- `--mode {sym,pw}`, the same pair weights as `a21/probe_check_eoa21.py`.
- **(c)** `stage3()` raises each `pct_on` (r384, r224, ours) to the power e1 before the mix. The HW v1 reproduction stays
  at e1 1.
- **(d)** Per label: rb = pct(float32 V224), rs = pct(S6fall) over the scored rows, then the fill `rs[~ok] = rb[~ok]`, then
  `rb, rs ** e1`, then the .8 / .2 mix, then a float32 cast, then E-OA (sym or pw; skipped at a 0).
- **(a2)** is new: blend_receipt e1 == e1 and the S6F blend e1 == e1. A missing key is accepted only at e1 1, so the gate
  still PASSes pre-E1 commits.
- **(d2)** checks a_oa, oa_mode (missing is allowed only for sym, since a20-lineage kernels carry none), and that
  EOA_APPLIED is present if and only if a > 0.
- The default `--ref` was repointed for the new folder depth.
- **18 checks.** The A20 / A21 ticks submit on the string `PASS 17/17`. An E1 tick must key on `PASS 18/18`.

**Self-test** (MEASURED `selftest/selftest_gate.log`: 17 of 17 rows as expected).
- Real commits under e1 1 PASS 18/18: c4 (a 0), eoa20, eoa30, pw20 (pw), pw30 (pw).
- The same commits checked as e1 2, with the wrong mode, or with the wrong a all FAIL. That is 15/18, 17/18 or 16/18
  depending on the case.
- SYNTHETIC E1 fixtures (`selftest/SYNTH_*`) PASS 18/18 at e1 2, and FAIL as e1 1 or as the wrong mode. Each fixture
  holds the real commit's members, log and receipts, with stage 3 and the final replayed from the E1 notebook. Each is
  marked by `SYNTHETIC.txt`.
- The OLD a20 gate FAILs the E1 fixture at 15/17. So an E1 commit cannot pass the old gate by accident.
- **Caveat (MEASURED).** On the 3-row visible test, (d) cannot tell sym from pw: the ranks are the same, and a pw20 commit
  checked as sym fails only through the receipt's oa_mode. E1 vs no-E1 is caught by both (c) and (d) (maxdiff .22).

## 7. What a push would still need (none of this was done)

1. **Eesh's explicit go.** MORE_1020 rates E1 WEAK and not recommended, and sec 4 here shades it lower. The default is
   STOP.
2. **A PREREG amendment.** It must override A20's "never stacked" and THINK_1019 8.4's "never stack with anything else".
3. **Base = the API best-id after the A21 read** (R1). Pick the row in sec 5. Never put E1 on a base that has another
   pending entry (R1 censoring).
4. **Timing (MODEL).**
   - 10-20 has 0 slots left.
   - The earliest submit is 10-21 05:30 IST, after a ~20 min commit, which needs a live session or an approved tick plus
     a GO file.
   - The read lands ~11:30-12:00 IST 10-21, well before the last readable submit (10-22 ~10:15 IST).
   - A best-id move means A switches once and B stays R6 57248084. A non-move closes E1 (no pow3, no invgeo).

## 8. Files (push/e1/)

**Build.**
- `build_e1eoa.py` + `build_e1eoa.log`.
- `build_receipt_*.json` (4).
- `kernel_v224r6c4eoa20e1j` (candidate), `kernel_v224r6c4eoa00e1j`, and 2 `*e1x1j` controls.

**Gate.** `gate_e1_offline.py` + `.log`.

**Contingency.**
- `build_e1_on.py`.
- `alt/build_e1_on.log`, `alt/build_receipt_*.json`, `alt/kernel_*` (3 candidates + 3 controls).
- `gate_e1_alt.py` + `alt/gate_e1_alt.log`.
- The alt copies of the eoa20 builds were deleted after confirming the identical sha.

**Commit gate.** `probe_check_e1eoa.py`, `selftest_gate.py`, `selftest_replay.py`, `selftest/` (logs, per-case outputs,
2 SYNTH fixtures).

**Info.** `info_f32_ties.py` + `.log`, `info_hidden_ties.py` + `.log`.
