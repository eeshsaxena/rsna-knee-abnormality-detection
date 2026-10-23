# J overlay (make_j.py) + FLIP-J / S6F-J / FLIPS-J. Built 10-15, REVIEWED + REBUILT (r2) 10-15 by Claude subagents. NOT pushed, NOT committed, NOT submitted.

No em dashes. Tags: MEASURED (files / outputs produced or read here), MODEL (inference), GUESS. Inputs: ../RESULT.md (JPEG Lossless
drop is real), ../f20j/RESULT.md (DEC cell + Kaggle CPU proof on the same pin), ../../s6f/RESULT.md issue 8 (S6F needs J in the base).
Budget used: no Kaggle call of any kind (no push, no commit, no dataset, no API read); laptop CPU only (light). Nothing outside
`jpeg/overlay/` was written; no existing kernel dir, READY file, box job or scheduled task touched. Every edited script has its
pre-change copy in `prechange_rev/` (and `prechange_rev/tests/`). Final offline run on the r2 kernels: see `r2/offline_j_all.txt`.

**Current kernels = `r2/kernel_TAG` (review rebuild). The r1 dirs `kernel_TAG` next to this file (c6e95b5a / 80246296 / 3e350f3a)
are SUPERSEDED: do not push them (gpu_commit_j.sh pushes r2 only, and the r2 gate FAILS an r1 output: no preflight, no envwatch).**

## Review (10-15): verdicts and fixes
| id | claim | verdict | fix (minimal) | fail before / pass after (all MEASURED, laptop) |
|---|---|---|---|---|
| J1 | fail-closed covered only the install; a native decoder fault on a hidden file hangs Raptor's fork Pool (TimeoutExpired, no submission) | **REAL**: pool worker killed by os._exit hangs `imap_unordered(chunksize=2)` (`POOL_HANG ... got [0, 1, 4, 5, 6, 7] of 8`); Raptor cell has no catch around `subprocess.run(..., timeout=2*3600)` | DEC cell PREFLIGHT on the test data before anything is switched on (sec. 2) | harness on the r1 cell: N9 `pool: HANG`, N9-N11/N13 + P preflight FAIL (17/24); r2: 24/24 (FLIP-J) |
| J2 | no in-run evidence that any stage process had /tmp/_dec; S6F receipts not gated; self-test only on pydicom 3.0.2 | **REAL** (j_check read neither S6F receipt; harness hops use their own argv) | step 1: j_check gates the S6F setup + driver receipts; step 2: DEC cell /proc watcher thread -> `dec_envwatch.json`, gated by `j_envwatch.py` (sec. 3, 6) | pre-change gate passes 11 / 15 / 11 of the new negatives that must FAIL; new gate 38/38, 43/43, 40/40; harness P / N12 |
| J3 | the make_j audit passes -I / -E / -S launches, sys.path reassignment, transformed PYTHONPATH; never sees run-time-loaded code | **REAL** (the 5 synthetic forms pass the r1 audit; the 3 built kernels have none of them: r2 audit passes all 7 lineup bases) | stricter audit (isolation options, sys.path removals in every unit, structural PYTHONPATH forms + exact-text PP_ALLOW); receipt lists UNAUDITED run-time code (sec. 1) | pre-change make_j BUILDS all 9 new refusal cases (TEST_MAKE_J 18/30); new 30/30 |
| J6 | info: inheritance, shadowing, S6F isolation, identity on uncompressed data verified | **confirmed** (re-checked: every r2 launch inherits / extends PYTHONPATH, r2 cells 0-15 of S6F-J and FLIPS-J == FLIP-J's except the DEC tag, offline suite re-run) | none | |
pydicom 3.0.1 (the stages after R7's pip) with these wheels is MEASURED decoding JPEG Lossless exactly on this pin in rsna-jpeg-check
v2 (rows RW / R7W of ../RESULT.md); the preflight runs on the image's 3.0.2 and feeds the same frame bytes to the same C decoders (MODEL).

## Verdict (r2)
**J_OFFLINE PASS (MEASURED, `r2/offline_j_all.txt`, re-runs everything on the current files).**
| check (all MEASURED, laptop) | FLIP-J | S6F-J | FLIPS-J |
|---|---|---|---|
| cell-diff proof `j_celldiff_proof.py` | J_CELLDIFF PASS 19/19 | PASS 19/19 | PASS 19/19 |
| local CPU harness `j_local_harness.py` | J_HARNESS PASS 24/24 | PASS 27/27 | PASS 25/25 |
| gate negatives `tests/neg_j_check.py` (real base commit outputs) | NEG_J_CHECK PASS 38/38 | PASS 43/43 | PASS 40/40 |
| builder tests `tests/test_make_j.py` (all variants) | TEST_MAKE_J PASS 30/30 | | |
Fail-before evidence (same new tests, pre-change code): `harness_v224flipj_PRECHANGE_CELL.txt` (r1 cell: J_HARNESS FAIL 17/24),
`tests/neg_j_check_TAG_PRECHANGE_GATE.txt` (pre-change gate: 27/38, 28/43, 29/40), `tests/test_make_j_PRECHANGE_MAKE.txt` (pre-change
make_j placed in a temp tree so its relative paths resolve: 18/30; the 3 D failures are expected, it rebuilds the r1 shas).

Not proven here (needs the GPU commit, prepared): the DEC cell on the Kaggle image itself (its install / self-test predecessor was
MEASURED there: rsna-v224f20j-check v1 DEC_OK 5.6 s; the preflight scan and the /proc watcher have NOT run on Linux yet), and
end-to-end output identity on the 3 public studies.

## The J kernels (r2, current)
| | FLIP-J | S6F-J | FLIPS-J |
|---|---|---|---|
| dir (new) | `overlay/r2/kernel_v224flipj/` | `overlay/r2/kernel_v224flips6fj/` | `overlay/r2/kernel_v224flips20j/` |
| slug | eeshsaxena/rsna-sub-v224flipj | eeshsaxena/rsna-sub-v224flips6fj | eeshsaxena/rsna-sub-v224flips20j |
| base dir (read only) | session_2026-10-14/v224/kernel_v224flip | session_2026-10-15/s6f/kernel_v224flips6f | session_2026-10-15/v224f/kernel_v224flips20 |
| base notebook sha | 73a9d5620d0a637e (== the pulled v1 copy) | 9c0e2558d51e4adb (pushed v1) | d2066b8c9c5e3a77 (pushed v1) |
| base commit output (gate ref) | session_2026-10-14/v224/out_v224flip_v1 | session_2026-10-15/s6f/out_v224flips6f_v1 | session_2026-10-15/v224f/out_v224flips20_v1 |
| J notebook sha256 (r2) | 7be2aa2cbb1ba46fb215275f2bad15d1a04de0a57119692999ef0ac47902b9e9 | 1868634ea68ef131035a01a347f4b1c4471f0a3d7a58b8a22f9efec3a309b02a | a582d23dff0b0a8c4fc8a47cdd9b6eb439322a219861ff458067f0a9116f848d |
| cells (base) / DEC index | 16 (15) / 2 | 19 (18) / 2 | 25 (24) / 2 |
| DEC cell sha (only the tag differs; template e2b15926) | c9b6250a58d81eac | 8c877bb53d14e1d6 | 6645c423df0b4fcf |
| metadata change | id / title / code_file + glorysing codec dataset | id / title / code_file (codec dataset already in the base) | id / title / code_file + codec dataset |
| gate command (from C:/Users/eeshs/Downloads/rsna-knee) | `py -3.12 -I session_2026-10-15/jpeg/overlay/j_check.py session_2026-10-15/jpeg/overlay/r2/out_v224flipj_v1 --base-out session_2026-10-14/v224/out_v224flip_v1` | `... j_check.py .../r2/out_v224flips6fj_v1 --base-out session_2026-10-15/s6f/out_v224flips6f_v1` | `... j_check.py .../r2/out_v224flips20j_v1 --base-out session_2026-10-15/v224f/out_v224flips20_v1` |
| push + wait + gate (PREPARED, NOT RUN) | `bash session_2026-10-15/jpeg/overlay/gpu_commit_j.sh v224flipj` | `... gpu_commit_j.sh v224flips6fj` | `... gpu_commit_j.sh v224flips20j` |
S6F-J and FLIPS-J start with exactly FLIP-J's 16 cells (DEC tag aside, MEASURED on r2). Private, T4 x2, internet off, docker pin
2757e0c7, all copied from the base and asserted by the builder and the proof.

## 1. `make_j.py IN_KERNEL_DIR OUT_KERNEL_DIR --slug OWNER/rsna-sub-TAG [--expect-base-sha SHA] [--dry-run]`
- Reads the base kernel dir, reproduces the exact serialisation of the notebook (compact JSON) and kernel-metadata.json (indent 1,
  CRLF) or refuses; refuses a pulled dir (`id_no`), a non-empty OUT, a slug equal to the base, a base that already has a J cell.
- **Where the cell goes.** No separate clock cell in the V224 family (`NB_T0 = time.time()` is inside the Raptor r384 launch cell),
  so the DEC cell is the first code cell (index 2, before the "## Stage 1" heading, the Kaggle-proven F20-J place): its seconds run
  before NB_T0 (sec. 8 issue 3).
- **Audit (refuses on any doubt), r2 rules (review fix J3):** AST scan of every cell, %%writefile script and embedded source
  (R7 SOURCES, Raptor fastread, S6F driver + PROBE / LAUNCHER / B_PROBE / ENV2): (a) every process launch passes no env= or an
  env built from os.environ whose PYTHONPATH is, structurally, the inherited value, `X + os.pathsep + <inherited>` (either side)
  or `os.pathsep.join([..., <inherited>, ...])`, also through plain names; anything that transforms the inherited value
  (.replace / .split / comprehension / filter / slicing) is refused unless an exact-text PP_ALLOW entry covers it (one: S6F's
  `stock_pp` = inherited entries minus empty ones, with its 4 anchors); (b) no Python launch anywhere passes -I / -E / -S (argv
  lists incl. combined options like -uI, list + list, shell strings, f-strings, `!python` magics) or isolated=True; (c) no unit of
  any kind reassigns, slice-assigns or deletes from sys.path or calls sys.path.remove / pop / clear (insert / append / += allowed);
  (d) no cell sets / pops os.environ PYTHONPATH; a generic env restore must restore a snapshot taken after the DEC cell (FLIPS
  ENV-SNAP / RESTORE: OK); (e) the 4 exact-text ALLOW entries of r1 (S6F_POP_NAMED, S6F_POP_EMPTY, S6F_RUN_ENV,
  S6F_DRIVER_SPECS) unchanged. Dry builds of V224-FLIP-RM, FLRM-AO-S, FF-S and F20 pass the r2 audit (MEASURED, test_make_j A1-A4).
- **UNAUDITED list (receipt `runtime_code`, J3):** what the static audit cannot read: R7 workers' argv (built at run time from
  `a.worker`), the R7 controller's `str(CODE)` code dir on PYTHONPATH (mainline from cloud_code.zip), FLIPS `fleet_worker.py`
  (copied from a dataset) + the `fleet_dir` code dir, the S6F B workers' argv. For these the in-run envwatch (sec. 3) is the check.
- Writes OUT/{kernel-metadata.json, rsna_sub_TAG.ipynb} only, plus `j_build_receipt_TAG.json` and `celldiff_TAG.txt` next to OUT.

## 2. The DEC cell (`dec_cell_j_template.py`, template sha e2b15926)
Install / self-test unchanged from r1 (Kaggle-proven F20-J logic): the 3 cp313 wheels, sha256 == PyPI, from `/kaggle/input/{,*/,
datasets/*/}rsna-knee-offline-codecs-r09`; `pip --no-index --no-deps --target /tmp/_dec`; exactly the 8 expected entries; a fresh
subprocess decodes 3 embedded DICOMs exactly with the decoders from /tmp/_dec. New in r2:
1. **PREFLIGHT (J1), after the self-test and BEFORE any change:** finds the test root exactly as the Raptor scripts do
   (`/kaggle/input/competitions/rsna-knee-abnormality-detection`, `/kaggle/input/rsna-knee-abnormality-detection`, then a walk for
   test.csv); a throwaway subprocess reads the transfer syntax (raw file-meta read, no pydicom, 32 threads) of the first and last
   file (by name) of every directory under test_series (or test_images); every file of a directory whose probe is not
   uncompressed / RLE / deflate (or unreadable / no file meta) is decoded with `pydicom.dcmread(p).pixel_array` by up to 4
   subprocesses with PYTHONPATH = /tmp/_dec + old. A decode process that dies (signal, os._exit, OOM kill) or any timeout within
   PRE_TIMEOUT_S = 600 s (scan + decode, MODEL cap) = J_UNAVAILABLE, nothing changed. A Python exception is counted, not fatal:
   pydicom then tries its other plugins (MEASURED in the pydicom 3.0.2 source: `_decode_frame` loops over all available
   plugins) and the stages catch the rest exactly as for the base's "no plugin" RuntimeError.
2. **envwatch (J2), only on DEC_OK:** a daemon thread (`dec_envwatch`) reads /proc every 1 s, prints nothing, and keeps
   `/kaggle/working/dec_envwatch.json` (atomic rewrite on change): for every Python process seen after DEC (argv[0] python*):
   pid, ppid, script (first non-option argument, or `-c` / `-m X`), interpreter options -I / -E / -S, the PYTHONPATH of its
   /proc environ (= what it was exec'd with), /tmp/_dec in it, and whether it descends from the notebook process (ppid chain);
   counters per script; first 400 records, first 50 bad ones. Forks of the notebook process before their exec (argv == the
   kernel's) are skipped (their /proc environ is the kernel's pre-DEC block). A start failure leaves DEC_OK and says so (gate FAIL).
3. DEC_OK / dec_receipt.json gain `preflight` (test root, series dir, groups / files / flagged / decoded ok / err, probe syntaxes,
   seconds) and `envwatch`. Everything else (fail closed, never raises, versions, in-process sys.path insert only while pydicom
   is not imported) as in r1.

## 3. How each kernel launches each DICOM-decoding stage (MEASURED: build audit + harness; in-run: envwatch on the GPU commit)
| stage | launch (base cell) | env | sees /tmp/_dec via | envwatch script |
|---|---|---|---|---|
| Raptor r384 / r224 | cell 4 / 11: `subprocess.run([sys.executable, '-u', '/tmp/raptor*_run.py'])`; inside, `mp.get_context('fork').Pool` | inherited | PYTHONPATH (fork children: same environ) | raptor_run.py, raptor224_run.py |
| R7 controller | cell 8: `env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), ...)` | extended | PYTHONPATH (CODE first) | probe_dual.py |
| R7 workers | R7 SOURCES probe_dual.py: `env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gi), ...)`; DataLoader children | inherited | PYTHONPATH | compact_infer.py (+ `-c` spawn children) |
| fleet workers (FLIPS) | cell 21: `env = {**os.environ, ..., 'PYTHONPATH': fleet_dir + os.pathsep + os.environ.get('PYTHONPATH', '')}` | extended | PYTHONPATH; ENV-SNAP (16) after DEC | fleet_worker.py |
| S6fall driver + B workers (S6F) | SETUP cell 15 env (PP_ALLOW S6F_PP_STOCK) -> RUN cell 16 Popen(env=_S6F_ENV) -> b_env() | extended | /tmp/s6f_dec:/tmp/_dec:<image> | s6f_driver.py, k1_b_infer.py |
| in-process (notebook) | none (audit) | | sys.path insert (inert here) | |

## 4. Cell-diff proof (`j_celldiff_proof.py BASE_DIR J_DIR`, independent of make_j.py)
19 checks per kernel (unchanged): only metadata + notebook in the J dir; one extra cell; all base cells identical and in order;
deleting it re-serialises to the base byte for byte; its text == the template re-rendered by the proof itself (fixtures sha
725bd7d8, PyPI shas from the rsna-jpeg-check driver); first code cell; metadata equal except id / title / code_file / codec
dataset; private / internet off / same pin; build receipt agrees. Outputs: `r2/celldiff_proof_TAG.txt`.

## 5. Local CPU harness (`j_local_harness.py J_DIR`; `tests/j_sim.py`, `tests/stub_wheels.py`)
The DEC cell VERBATIM from each built notebook except its path line, its PRE_TIMEOUT_S / WATCH_EVERY_S / PROC line (fake /proc
dir, 0.2 s polls) and, where a working decoder is needed, WHEELS / TOP (stand-in wheels, pure-Python JPEG Lossless decoder; variants
crash / hang / raise misbehave only on a 37-row TRAP image). Every case has a fake competition test root (2 studies, 4 series dirs, 7
files: uncompressed, JPEG Lossless .70 / .57 and the TRAP). Simulated processes write their real environ / argv / ppid into the fake
/proc for the watcher. Checks (FLIP-J 24, S6F-J 27, FLIPS-J 25): BASE loses JPEG Lossless in every chain; P: DEC_OK,
versions, env, receipt, every chain decodes 4/4 from TARGET, in-process 4/4, **preflight scanned 4 dirs / 7 files, flagged the 5
files of the 3 compressed dirs, 5 decoded by live processes; watcher thread alive, every simulated stage process recorded with TARGET
on its PYTHONPATH, `j_envwatch.verdict` PASS**; Q (S6F): S6F own install fails, J holds; N1-N6 fail closed (no dataset, tampered, not
exact, pip failure, shadow numpy, self-test timeout); N7 pydicom pre-imported; N8 PYTHONPATH unset; **N9 decoder dies on the TRAP
(os._exit(139)): fail closed, and a Raptor-like spawn Pool(2).imap_unordered over the 7 test files COMPLETES (r1 cell: HANG); N10
decoder hangs on the TRAP: preflight timeout (8 s cap here), fail closed; N11 decoder raises on the TRAP: DEC_OK, counted (4 ok / 1
err), chains decode; N12 one stage launched with -E (a J3 form): that chain loses JPEG Lossless and the watcher verdict FAILS naming
-E; N13 no test root: fail closed.** Limits: stand-in decoder (the C libjpeg is MEASURED on Kaggle), Windows py3.12 + spawn, fake
/proc (Linux /proc formats are standard; the 1 s polling miss-rate is not simulated: stage processes live >= 20 s, MEASURED in the
base logs: R7 workers ~360 s, S6F B parts ~60 s, r224 23 s).

## 6. Gate for the GPU commit: `j_check.py OUT --base-out BASE_OUT [--expect-script S ...] [--envwatch-waive=S ...]`
(A) logs, J log named rsna-sub-TAG.log; (B) one DEC_OK before the first stage marker, no DEC_FALLBACK / J_UNAVAILABLE; versions,
self-test, receipt (PyPI wheels, 8 entries, decoders from /tmp/_dec), PYTHONPATH after == /tmp/_dec + ':' + before; **preflight ok
in DEC_OK + receipt, series dir scanned, every flagged file decoded by a live process**; (E, new) **envwatch started; dec_envwatch.json
judged by `j_envwatch.verdict`: watcher ok, every expected stage script of `j_envwatch.EXPECT[tag]` (sec. 3 last column) seen as a
descendant with /tmp/_dec on PYTHONPATH and no -I / -E / -S, and NO descendant Python process without it (records consistent with
counters); S6F kernels: setup receipt and driver env_isolation PYTHONPATH == /tmp/s6f_dec + ':' + the DEC PYTHONPATH, /tmp/_dec on
the driver sys.path, decoders from /tmp/s6f_dec**; (C) submission.csv == base output (1e-6; bytes reported), every base CSV, same
log-marker path, no more Traceback / FALLBACK / ... than the base; (D) info. A false FAIL from a benign descendant process (e.g. an
unforeseen helper without /tmp/_dec) can be re-gated OFFLINE after a look with `--envwatch-waive=SCRIPT` (printed); no new commit.
Negatives (`tests/neg_j_check.py TAG BASE_OUT [--gate PATH]`, clean mock = REAL base commit output + DEC_OK + the REAL Kaggle DEC
receipt with the new fields MOCKED + a MOCK dec_envwatch.json in the writer's schema): the r1 negatives plus J1 (no / empty
preflight), J2 (no envwatch file, watcher not ok, a stage script never seen / without /tmp/_dec / with -I, an extra descendant
without /tmp/_dec, the waiver, inconsistent counters, 4 S6F receipt mutations).

## 7. GPU commits (PREPARED, NOT RUN; main does GPU commits)
`bash session_2026-10-15/jpeg/overlay/gpu_commit_j.sh TAG` from the project root: asserts the r2 notebook sha == build receipt and
that it carries the review-fixed DEC cell, the metadata (slug, private, internet off, no id_no), re-runs `run_offline_j.py TAG`,
refuses if RUNNING_COUNT > 1, pushes `r2/kernel_TAG`, waits <= 45 min, fetches to `r2/out_TAG_v1`, runs the gate. Cost (MODEL):
the base commit time + ~6 s DEC + ~2 s preflight scan (15 series, 30 header reads, nothing flagged on the all-uncompressed public
test). A PASS = same public score as the base, J insurance on the private split (never let J block a pick).

## 8. Open issues / risks
1. The r2 DEC cell has not run on the Kaggle image (preflight scan + /proc watcher only on the laptop with a fake /proc); the GPU
   commit + gate is the proof. If DEC fails there: J_UNAVAILABLE = the base, but no J pick (gate FAIL).
2. GPU noise (unchanged): submission.csv within 1e-6 (MEASURED max diff 0 between two FLIP runs).
3. DEC runs before NB_T0. Public: ~6 s + ~2 s (MODEL). Hidden (~1300 studies): scan ~5-30 s (MODEL, 32 threads, 2 reads per
   series dir) + decoding of the compressed dirs on 4 processes (MEASURED rates: JPEG Lossless ~3.5 s, J2K ~7.3 s per 135-slice
   study per core) capped at 600 s; worst case ~20 min (300 + 300 + 600 s caps, MODEL), inside her 40 min r224 margin; hidden V224
   walls ~3 h vs the 9 h limit (GUESS from ../RESULT.md).
4. Preflight residuals (J1): a directory whose first and last files are uncompressed but a middle file is compressed is not
   preflighted (GUESS negligible: syntax is per export); nondeterministic native faults; >600 s of compressed decoding (MODEL ~330
   J2K studies) -> J_UNAVAILABLE by design (a J2K-heavy hidden set loses J's JPEG Lossless benefit, never worse than the base).
5. With the decoders, pydicom prefers pylibjpeg-openjpeg over Pillow for JPEG 2000 (and libjpeg over Pillow for JPEG baseline):
   identical on the tested lossless + one lossy fixture (MEASURED); a lossy stream decoding differently is a GUESS-level residual.
   The preflight now exercises the new openjpeg on every J2K test dir (faults / hangs caught), not its output values.
6. Third-party dataset glorysing/rsna-knee-offline-codecs-r09: sha-pinned to PyPI (changed file = J_UNAVAILABLE); a deleted dataset
   might block a rerun (GUESS); pylibjpeg-libjpeg is GPLv3.
7. FF-S, FLRM, FLRMS20 J variants not written (dry builds pass the r2 audit, MEASURED): one make_j.py call each into r2/, then add
   their expected stage scripts to `j_envwatch.EXPECT` (or pass `--expect-script`), and gate against their own base commit output.
8. UNAUDITED run-time code (sec. 1) is checked only in-run by envwatch, and only for the processes the public commit starts.

## Files (session_2026-10-15/jpeg/overlay/)
Scripts: `make_j.py`, `dec_cell_j_template.py`, `j_celldiff_proof.py`, `j_local_harness.py`, `j_check.py`, `j_envwatch.py` (new),
`run_offline_j.py`, `gpu_commit_j.sh`, `tests/` (`j_sim.py`, `stub_wheels.py`, `neg_j_check.py`, `test_make_j.py`). Pre-change copies:
`prechange_rev/`. Current kernels + outputs: `r2/` (`kernel_TAG/`, `j_build_receipt_TAG.json`, `celldiff_TAG.txt`, `build_j_r2.log`,
`celldiff_proof_TAG.txt`, `harness_TAG.txt`, `neg_j_check_TAG.txt`, `test_make_j.txt`, `offline_j_all.txt`). Fail-before outputs:
`harness_v224flipj_PRECHANGE_CELL.txt`, `tests/neg_j_check_TAG_PRECHANGE_GATE.txt`, `tests/test_make_j_PRECHANGE_MAKE.txt`.
Superseded r1 (kept untouched, do not push): `kernel_TAG/`, `j_build_receipt_TAG.json`, `celldiff*_TAG.txt`, `harness_TAG.txt`,
`build_j.log`, `offline_j_all.txt`, `tests/neg_j_check_TAG.txt`, `tests/test_make_j.txt`. Self-test fixtures: `../f20j/selftest_fixtures.json`.
