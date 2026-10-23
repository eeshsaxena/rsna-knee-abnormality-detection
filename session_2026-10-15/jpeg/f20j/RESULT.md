# V224-F20-J (V224-F20 + JPEG Lossless decoders): built + CPU-proven 10-14 by a Claude subagent. NOT pushed to GPU, NOT submitted.

No em dashes. Tags: MEASURED (kernel output), MODEL (inference), GUESS. Inputs: ../RESULT.md (the drop is real), FORUM_DIGEST_10-14.md.
Budget used: 1 private CPU commit (`eeshsaxena/rsna-v224f20j-check` v1, 19:21 -> 19:51 IST, 1,776 s, COMPLETE); total CPU commits
3 of 4 (2 by the jpeg-check run). No GPU, no dataset created, nothing submitted.

## Verdict
**V224F20J_CPU PASS 94/94 (MEASURED).** The inserted DEC cell, run verbatim from the built notebook on the same docker pin, makes
every V224-F20 stage decode JPEG Lossless (.70, .57, mixed) exactly, and every stage output on uncompressed data is byte-identical
with and without it (9 real studies, the uncompressed fixture, 48 synthetic edge-case files). The GPU commit (end-to-end identity on
the 3 public studies) is prepared, not run.

| stage (its environment inside V224-F20) | OFF / ON | 9 real uncompressed studies ON == OFF | U fixture | IMPL, JLL70, JLL57, J2KR, RLE, MIX ON == U | J2K lossy ON == OFF | OFF loses JPEG Lossless |
|---|---|---|---|---|---|---|
| Raptor r384 (stock image) | S / SJ | 9/9 | OK | 6/6 | OK | yes |
| Raptor r224 (after R7 pip) | R / RJ | 9/9 | OK | 6/6 | OK | yes |
| R7 worker (CODE + PYTHONPATH) | R7 / R7J | 9/9 | OK | 6/6 | OK | yes |
| our fleet in V224-F20 (after R7 pip) | R / RJ | 9/9 | OK | 6/6 | OK | yes |
| fleet in the stock image (HFR7-RR) | S / SJ | 9/9 | OK | 6/6 | OK | yes |
| plain pydicom, stock / after R7 pip | S/SJ, R/RJ | 9/9, 9/9 | OK | 6/6 | OK | yes |

(`out_jj_v1/v224f20j_cpu_table.md`, `out_jj_v1/v224f20j_cpu_result.json` with all 94 checks, kernel log.) Real studies = the 3 public
test studies + 6 random train studies with all 3 planes (1,682 original files copied unchanged). Output shas match the jpeg-check
v2 run (Raptor 37f0bb06, fleet b5527c93 here with the contrast class in the hash, R7 7d577755). Per-study decode, OFF -> ON:
Raptor JPEG Lossless 0.17 s (zero-filled) -> 3.4 s (exact); R7 0 series (study fallback) -> 510 windows; fleet 0 views -> 4 views.

## 1. What was built (`build_v224f20j.py`)
`kernel_v224f20j/rsna_sub_v224f20j.ipynb` sha256 **2a480ce894ee9f2f5c2f10da57343c052ff1fd1f60039ccd57cb99953790dcb5**, slug
`eeshsaxena/rsna-sub-v224f20j` (private, T4 GPU, internet off, docker pin 2757e0c7 = V224-F20's). 23 cells = V224-F20's 22 cells
**byte-identical and in order** + ONE code cell at index 2 (before "## Stage 1"). Builder proof: deleting it re-serialises to the
V224-F20 notebook (8f8b38e4...) byte for byte. Metadata = V224-F20's + dataset `glorysing/rsna-knee-offline-codecs-r09`. Every sha
assert of V224-F20 is untouched. Full cell text: `celldiff_v224f20j.txt`; receipt: `build_receipt_v224f20j.json`.

**DEC cell** (`dec_cell_template.py` + `selftest_fixtures.json`, cell sha cf9f09c0...), wrapped in a function (leaves only
`_DEC_RECEIPT` in the notebook namespace; never raises):
1. finds the 3 cp313 wheels under `/kaggle/input/{,*/,datasets/*/}rsna-knee-offline-codecs-r09` and accepts a file only if its
   sha256 == the official PyPI wheel (pylibjpeg 2.1.0, pylibjpeg-libjpeg 2.4.0, pylibjpeg-openjpeg 2.6.0);
2. copies exactly those 3 files to `/tmp/_dec_wheels`, `pip install -q --no-index --no-deps --target /tmp/_dec --find-links` them
   (versions pinned), asserts `/tmp/_dec` holds exactly the 8 expected entries (nothing like numpy / pydicom can shadow the image);
3. self-test in a fresh subprocess with PYTHONPATH = /tmp/_dec + old: versions, modules imported from /tmp/_dec, pydicom lists
   pylibjpeg for .57 / .70, and 3 embedded tiny DICOMs decode exactly (JPEG Lossless .70 16-bit signed, .57 12-bit unsigned, an
   uncompressed control; made by `make_selftest.py` with our `jll.py` encoder);
4. only then `os.environ['PYTHONPATH'] = '/tmp/_dec:' + old` (the cell's only effect on later cells). Every decoding process
   inherits it (builder asserts this from the notebook text with an AST scan of every process launch): Raptor r384 / r224 =
   `subprocess.run` with no env (+ a fork pool), R7 controller env = CODE + os.environ PYTHONPATH and probe_dual's workers =
   `dict(os.environ, ...)`, fleet workers = fleet_dir + os.environ PYTHONPATH; ENV-SNAP / ENV-RESTORE keep it. The notebook
   process's own sys.path is untouched.
5. Any failure -> `DEC_FALLBACK {...}`, PYTHONPATH untouched (= exactly V224-F20), `dec_receipt.json` written. Success ->
   `DEC_OK {...}`. Measured cost 5.6 s, before NB_T0 (her 9 h guard keeps 40 min margin; the fleet wall uses min(/proc start, T0)).

## 2. CPU proof (`build_jj.py`, `kernel_src/driver_jj.py`, `kernel_src/probe_jj.py` -> `kernel_jj/`, sha 309933b2...)
Everything decode-relevant is taken from the built J notebook and sha-asserted: the DEC cell text (exec'd in the driver exactly as
the notebook runs it), the `_fastread` module (equal in r384 / r224 cells), R7's SOURCES / CODE_SHA (+ anatomy.pt / side model sha);
fleet_common.py from the k11w dataset. Environments re-created with the notebook's own formulas (asserted by the builder). R7's
pip install was run with the cell-6 command **under the J environment** (as in the J notebook): rc 0, "Successfully installed
pydicom-3.0.1 timm-1.0.25" (the v224_check marker) unchanged. Fixtures: as jpeg-check v2 (one real public study x 8 syntaxes,
135/135 exact for every lossless syntax). DEC negatives in subprocesses (MEASURED): tampered libjpeg wheel, no dataset, self-test
digest wrong -> DEC_FALLBACK + PYTHONPATH unchanged + rc 0; positive control in a fresh process -> DEC_OK. In this kernel the
codec dataset mounted at `/kaggle/input/datasets/glorysing/...` (v2 had it at `/kaggle/input/rsna-knee-offline-codecs-r09`):
the cell's search covers both.

## 3. Local tests (before the push)
`tests/sim_dec_local.py` 6/6 (fallback paths on win32: no dataset, tampered, partial, pip failure; self-test program logic),
`tests/smoke_probe_local.py` PASS (probe wiring on mocks), `tests/sim_verdict.py` 14/14 (verdict logic: clean PASS, 13 single
defects FAIL the right check).

## 4. GPU commit test (PREPARED, NOT RUN) and gate
`bash session_2026-10-15/jpeg/f20j/gpu_commit_v224f20j.sh` (from the project root, after 19:30 IST): asserts the notebook sha +
CPU PASS, refuses if RUNNING_COUNT > 1, `kaggle kernels push -p session_2026-10-15/jpeg/f20j/kernel_v224f20j`, waits / fetches into
`out_v224f20j_v1`, then the gate:
```
python session_2026-10-15/jpeg/f20j/v224f20j_check.py session_2026-10-15/jpeg/f20j/out_v224f20j_v1
```
Gate = (1) the unchanged V224-F20 gate `v224f_check.py --wf 0.20` with the V224-exact and HFR7-RR refs (28 checks; it also scans
the DEC part for Traceback / RuntimeError); (2) exactly one DEC_OK before RAPTOR_STAGE_DONE, no DEC_FALLBACK, receipt (3
PyPI-identical wheels, 8 target entries, decoders from /tmp/_dec, 3/3 self-test, PYTHONPATH form); (3) submission.csv /
submission_V224 / submission_H (1e-6) and fleet solo / raptor_sub / raptor_raw / raptor224_sub / ours_sub (1e-3) vs the V224-F20 v1
commit, byte identity reported (expected 8/8, MODEL: no compressed public file). Prints `V224F20J_CHECK PASS n/m (byte-identical k/8)`.
Gate negatives (`tests/neg_gate.py --receipt out_jj_v1/dec_receipt.json`, real V224-F20 v1 output + the real CPU DEC receipt):
**13/13** (clean simulated output PASS 15/15 incl. the full V224-F20 gate; DEC_FALLBACK, DEC_OK after stage 1, extra package,
submission +.01, raptor_raw +.01, two DEC_OK, PYTHONPATH not extended, self-test not exact, 2 of 3 wheels, no receipt: all FAIL).
Cost: one ~11 min T4x2 commit (V224-F20 v1: 586 s notebook time) + 5.6 s for the DEC cell.

## 5. PREREG amendment text
`PREREG_AMENDMENT_J.txt` (Amendment 2, not appended; the main session decides). Core: F20-J is not a lineup entry (F20 was dropped by
Amendment 1); its GPU commit is the end-to-end proof; the DEC cell may be added to final-pick candidates, each with its own commit
equal to its base; a J entry must score exactly its base's public value and is read as private insurance only.

## 6. Same fix on the current lineup kernels (dry builds in the scratchpad, not written here, not pushed)
`build_v224f20j.py --base <nb> --base-sha <sha> --tag <tag>` passes every builder assert on **V224-FF20** (b86f8b9f... -> v224ff20j
743d188f...) and, with `--allow-no-fleet`, on **V224-FLIP-RM** (f14f6acc... -> v224flrmj 9b42a54d...). Both have the same `_fastread`,
R7 SOURCES and CODE_SHA as F20 (so this CPU proof covers their readers); FF20's time-aware flip driver launches the pinned
compact_infer.py with an inherited env; FLRM's rm_verify launch uses dict(os.environ, CUDA_VISIBLE_DEVICES=''), PYTHONPATH kept.
Each would still need its own GPU commit + gate (v224f20j_check.py is F20-specific: refs and --wf).

## 7. Risks / notes
- Public LB effect exactly 0 (MODEL from the MEASURED census: 557/557 public files uncompressed). Value = private insurance only,
  <= ~2 % of hidden studies (MODEL, fleet coverage gate).
- Third-party dataset: sha-pinned to PyPI, so a changed file is refused (fallback = V224-F20). If the dataset were deleted, Kaggle
  might refuse to start a rerun at all (GUESS); owning a copy would remove that, but this task does not create datasets.
- A fallback in the hidden rerun is invisible on the LB (log only), by design: it degrades to V224-F20, never below it.
- With the wheels, pydicom's J2K path also changes (Raptor J2K lossless 12.6 s -> 7.2 s per 135-slice study, so pylibjpeg-openjpeg
  is used): outputs identical for J2K lossless AND the 16-bit lossy fixture (MEASURED); a different lossy J2K stream decoding
  differently between the two OpenJPEG builds is a GUESS-level residual. JPEG Lossless decode costs ~3.4 s / study on 4 CPU cores
  (Raptor) vs 0.8 s uncompressed: runtime-neutral at <= 2 % of studies.
- pylibjpeg-libjpeg is GPLv3 (same decoder as the public b1-ctrl notebook).

## Files (this dir)
`build_v224f20j.py`, `dec_cell_template.py`, `make_selftest.py`, `selftest_fixtures.json` (725bd7d8...), `kernel_v224f20j/`,
`build_receipt_v224f20j.json`, `celldiff_v224f20j.txt`, `build_jj.py`, `kernel_src/{driver_jj,probe_jj}.py`, `kernel_jj/`,
`push_jj_v1.log`, `wait_jj.sh`, `wait_jj_v1.log`, `out_jj_v1/` (result json, table, dec_receipt.json, log), `v224f20j_check.py`,
`gpu_commit_v224f20j.sh`, `PREREG_AMENDMENT_J.txt`, `tests/` (sim_dec_local, smoke_probe_local, sim_verdict, neg_gate + outputs).
