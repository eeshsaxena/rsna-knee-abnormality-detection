# J overlay (make_j.py) + FLIP-J / S6F-J / FLIPS-J: built + offline-proven 10-15 by a Claude subagent. NOT pushed, NOT committed, NOT submitted.

No em dashes. Tags: MEASURED (files / outputs produced or read here), MODEL (inference), GUESS. Inputs: ../RESULT.md (JPEG Lossless
drop is real), ../f20j/RESULT.md (DEC cell + Kaggle CPU proof on the same pin), ../../s6f/RESULT.md issue 8 (S6F needs J in the base).
Budget used: no Kaggle call of any kind (no push, no commit, no dataset, no API read); laptop CPU only (light: ~1 min per variant).
Nothing outside `jpeg/overlay/` was written (tests use system temp dirs, removed after each run); no existing kernel dir, READY file,
box job or scheduled task touched; no existing script edited (so no pre-change copies were needed). Final offline run 05:31 IST 10-15.

## Verdict
**J_OFFLINE PASS 10/10 (MEASURED, `offline_j_all.txt`).** Three J kernels exist, each = its base notebook + ONE inserted DEC cell + the
codec dataset in the metadata; every other byte identical.
| check (all MEASURED, laptop) | FLIP-J | S6F-J | FLIPS-J |
|---|---|---|---|
| cell-diff proof `j_celldiff_proof.py` | J_CELLDIFF PASS 19/19 | PASS 19/19 | PASS 19/19 |
| local CPU harness `j_local_harness.py` | J_HARNESS PASS 17/17 | PASS 20/20 | PASS 18/18 |
| gate negatives `tests/neg_j_check.py` (real base commit outputs) | NEG_J_CHECK PASS 26/26 | PASS 27/27 | PASS 28/28 |
| builder tests `tests/test_make_j.py` (all variants) | TEST_MAKE_J PASS 21/21 | | |

Not proven here (needs the GPU commit, prepared): the generalised DEC cell on the Kaggle image itself (its predecessor, same install /
self-test logic, was MEASURED there: rsna-v224f20j-check v1 DEC_OK, 5.6 s), and end-to-end output identity on the 3 public studies.

## The J kernels
| | FLIP-J | S6F-J | FLIPS-J |
|---|---|---|---|
| dir (new) | `overlay/kernel_v224flipj/` | `overlay/kernel_v224flips6fj/` | `overlay/kernel_v224flips20j/` |
| slug | eeshsaxena/rsna-sub-v224flipj | eeshsaxena/rsna-sub-v224flips6fj | eeshsaxena/rsna-sub-v224flips20j |
| base dir (read only) | session_2026-10-14/v224/kernel_v224flip | session_2026-10-15/s6f/kernel_v224flips6f | session_2026-10-15/v224f/kernel_v224flips20 |
| base notebook sha | 73a9d5620d0a637e (== the pulled v1 copy, MEASURED) | 9c0e2558d51e4adb (pushed v1) | d2066b8c9c5e3a77 (pushed v1) |
| base commit output (gate ref) | session_2026-10-14/v224/out_v224flip_v1 | session_2026-10-15/s6f/out_v224flips6f_v1 | session_2026-10-15/v224f/out_v224flips20_v1 |
| J notebook sha256 | c6e95b5aab8de379f66bfe747faf5371e313f032fcdb084732079f76f0228a0d | 802462966859e147ec0d778fcfdd8f9f4b91408e3e02b3307ad63eec49919fbe | 3e350f3ad308d1478f4105b66d537f9e32e5f0cbb10f835b4e9016501e241132 |
| cells (base) / DEC index | 16 (15) / 2 | 19 (18) / 2 | 25 (24) / 2 |
| DEC cell sha (only the tag differs) | e41eaed77e662dde | 0c92c60b04fbd509 | 75787650cae1eb17 |
| metadata change | id / title / code_file + glorysing codec dataset | id / title / code_file (codec dataset already in the base) | id / title / code_file + codec dataset |
S6F-J and FLIPS-J start with exactly FLIP-J's 16 cells (DEC tag aside, MEASURED), i.e. S6F-J IS "the 3 S6F cells on a FLIP-J base"
(s6f/RESULT.md issue 8) and FLIPS-J is "the AO student cells on a FLIP-J base". Private, T4 x2 (machine_shape NvidiaTeslaT4), internet
off, docker pin 2757e0c7, all copied from the base and asserted by the builder and the proof.

## 1. `make_j.py IN_KERNEL_DIR OUT_KERNEL_DIR --slug OWNER/rsna-sub-TAG [--expect-base-sha SHA] [--dry-run]`
- Reads the base kernel dir, detects the exact serialisation of the notebook (compact JSON for the V224 family) and of
  kernel-metadata.json (indent 1, CRLF) and refuses if it cannot reproduce the bytes; refuses a pulled dir (`id_no` in the metadata: a
  push would target the BASE kernel), a non-empty OUT, a slug equal to the base, a base that already has a J cell.
- **Where the cell goes (decision).** The task said "first code cell after the notebook-start clock cell". The V224 family has no
  separate clock cell: `NB_T0 = time.time()` is set inside the Raptor r384 launch cell (cell 4), so "after the clock" and "before any
  decoding stage" cannot both hold without editing that cell. Rule implemented: if the first code cell is a pure clock cell (sets a T0,
  no process / pydicom), insert right after it; otherwise insert before the first code cell and its "## Stage" heading (index 2 here,
  the same place as the Kaggle-proven F20-J). Consequence: the DEC seconds run before NB_T0 (MEASURED 5.6 s for the F20-J cell; the new
  cell's pip and self-test timeouts are 300 s each, so <= ~10 min worst case, MODEL), inside her 40 min r224 margin; the fleet wall and
  the S6F start guard count from /proc start / kernel age anyway.
- **Audit (refuses on any doubt).** AST scan of every notebook cell, every %%writefile script and every embedded source (string
  literals that parse as Python and *_B64 constants: R7 SOURCES x7, Raptor fastread, the S6F driver and its PROBE / LAUNCHER / B_PROBE /
  ENV2): every process launch (subprocess.*, os.system / exec / spawn, multiprocessing) must pass no env= or an env built from
  os.environ whose PYTHONPATH is untouched or extended with the inherited value (scope-aware: the launch's own function + module level);
  no cell may set / pop os.environ PYTHONPATH (setdefault is allowed: it cannot replace an existing value); a generic environment
  restore must restore a snapshot taken after the DEC cell (FLIPS: ENV-SNAP cell 16, restore cell 22: OK). Four forms a static scan
  cannot prove are allowed only by an exact-text ALLOW entry whose anchors are present (all S6F): `S6F_POP_NAMED` (pops 3 named vars),
  `S6F_POP_EMPTY` (drops PYTHONPATH only when the inherited one is empty, impossible after DEC_OK), `S6F_RUN_ENV` (the RUN cell launches
  the driver with the SETUP cell's env), `S6F_DRIVER_SPECS` (run_group's spec env = b_env() = dict(os.environ, ...)).
- Writes OUT/{kernel-metadata.json, rsna_sub_TAG.ipynb} only, plus `j_build_receipt_TAG.json` and `celldiff_TAG.txt` next to OUT.
  Dry builds of V224-FLIP-RM, V224-FLIP-RM-AO-S, V224-FF-S and V224-F20 also pass the audit (MEASURED, test_make_j A1-A4).

## 2. The DEC cell (`dec_cell_j_template.py`, generalised from ../f20j/dec_cell_template.py)
Same install / self-test logic as the Kaggle-proven F20-J cell: the 3 cp313 wheels found under `/kaggle/input/{,*/,datasets/*/}
rsna-knee-offline-codecs-r09` and accepted only if sha256 == the PyPI file; `pip install --no-index --no-deps --target /tmp/_dec`;
/tmp/_dec must hold exactly the 8 expected entries; a fresh subprocess with PYTHONPATH = /tmp/_dec + old decodes 3 embedded DICOMs
exactly (JPEG Lossless .70 16-bit signed, .57 12-bit unsigned, uncompressed control) with the decoders imported from /tmp/_dec and
pylibjpeg listed by pydicom for .57 / .70. Changes vs F20-J:
1. on success it also puts /tmp/_dec first on the notebook process's sys.path (in-process decoding), ONLY if pydicom is not imported
   yet (pydicom fixes its plugin table at import); env + sys.path change together with rollback;
2. failure is "fail closed, visibly": nothing is changed (= the base notebook), `DEC_FALLBACK {...}` and `J_UNAVAILABLE {...}` are
   printed, dec_receipt.json says why; nothing raises (outer try as well). Invisible on the LB by design (the base is never worse);
3. versions recorded: python, pydicom, numpy, pylibjpeg / libjpeg / openjpeg, pydicom's own dir (must not be /tmp/_dec), in DEC_OK
   and /kaggle/working/dec_receipt.json; pip / self-test timeouts 300 s each (were 900 / 600, all before NB_T0).

## 3. How each kernel launches each DICOM-decoding stage (MEASURED: build audit + harness)
| stage | launch (base cell) | env | sees /tmp/_dec via |
|---|---|---|---|
| Raptor r384 / r224 | cell 4 / 11: `subprocess.run([sys.executable, '-u', '/tmp/raptor*_run.py'])`; inside, `mp.get_context('fork').Pool` | inherited | PYTHONPATH (fork children: sys.path) |
| R7 controller | cell 8: `env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), ...)` | extended | PYTHONPATH (CODE first) |
| R7 workers | R7 SOURCES probe_dual.py: `env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gi), ...)`; DataLoader children | inherited | PYTHONPATH |
| fleet workers (FLIPS) | cell 21: `env = {**os.environ, ..., 'PYTHONPATH': fleet_dir + os.pathsep + os.environ.get('PYTHONPATH', '')}`; threads | extended | PYTHONPATH; ENV-SNAP (16) after DEC, ENV-RESTORE (22) keeps it |
| S6fall driver + B workers (S6F) | SETUP cell 15: env = dict(os.environ), PYTHONPATH = /tmp/s6f_dec + the inherited entries; RUN cell 16 Popen(env=_S6F_ENV); driver: b_env() = dict(os.environ, ...) | extended | PYTHONPATH = /tmp/s6f_dec:/tmp/_dec:<image> (its own identical wheels first; ours if its install fails) |
| in-process (notebook) | none: no notebook-process code imports pydicom (audit) | | sys.path insert (inert here) |
Not decoding: R7's global pip (pydicom 3.0.1 + timm 1.0.25; MEASURED unchanged under the J env in rsna-v224f20j-check), S6F's own
pip --target (pip sets ignore_installed for --target, so /tmp/_dec on PYTHONPATH does not stop it: MEASURED in the harness, S6F_DEC_OK).

## 4. Cell-diff proof (`j_celldiff_proof.py BASE_DIR J_DIR`, independent of make_j.py)
19 checks per kernel: the J dir holds only metadata + notebook; notebook-level fields equal; exactly one extra cell; all base cells
identical and in order; deleting the inserted cell re-serialises to the base file byte for byte; the inserted cell is a bare code cell
whose text == the template re-rendered by the proof itself (fixtures sha 725bd7d8, PyPI shas read from the rsna-jpeg-check driver,
the 8 target entries from its v2 result); first code cell; no DEC name collision; metadata equal except id / title / code_file /
codec dataset; private / internet off / same pin and accelerator; build receipt agrees. Outputs: `celldiff_proof_TAG.txt`.

## 5. Local CPU harness (`j_local_harness.py J_DIR`; `tests/j_sim.py`, `tests/stub_wheels.py`)
Takes the DEC cell VERBATIM from each built notebook and changes only its path line (and, where a working decoder is needed, the WHEELS
/ TOP constants -> local stand-in wheels with the same distribution / import names and versions and a pure-Python JPEG Lossless decoder;
the real cp313 manylinux wheels cannot be installed on win32 and nothing was downloaded). A simulated notebook process runs the cell,
then launches every stage chain of THAT notebook with the env expression extracted from the notebook text (evaluated in the launching
process, hop by hop: R7 controller -> probe_dual worker -> DataLoader child; fleet worker; S6F SETUP cell executed for real -> driver ->
b_env worker -> child; Raptor -> pool child); each chain ends in `pydicom.dcmread(path).pixel_array` (the primitive all 4 stage readers
call) on 4 files: the 3 self-test DICOMs + a NEW 48x64 16-bit JPEG Lossless .70 made here with ../jll.py.
- BASE (no DEC cell): every chain loses JPEG Lossless ("no plugin" RuntimeError), the uncompressed control decodes.
- P (DEC_OK): one DEC_OK, no DEC_FALLBACK / J_UNAVAILABLE / Traceback; versions + 3/3 self-test; PYTHONPATH == TARGET + old;
  sys.path[0] == TARGET; only `_DEC_RECEIPT` left in the namespace; every hop has TARGET on its import path and every chain decodes 4/4
  exactly with libjpeg from TARGET (S6fall: from its own /tmp/s6f_dec); in-process decode 4/4.
- Q (S6F only): S6F's own decoder install fails but J holds: S6fall chain still decodes 4/4 via /tmp/_dec.
- N1-N6 fail closed: no dataset, stand-ins under the real names (sha mismatch = tampered), self-test not exact, pip failure, a shadow
  `numpy/` in the target, self-test timeout: each prints DEC_FALLBACK + J_UNAVAILABLE, raises nothing, changes neither PYTHONPATH nor
  sys.path, and every chain behaves exactly as BASE (same PYTHONPATH strings in every hop).
- N7 pydicom imported before the cell: DEC_OK, subprocesses decode, sys.path left alone (noted). N8 no PYTHONPATH before: PYTHONPATH
  == TARGET exactly (no empty entry).
Limits (MODEL / GUESS): stand-in decoder, not the C libjpeg (that one is MEASURED on Kaggle in rsna-jpeg-check v2 / rsna-v224f20j-check
v1: every stage byte-identical on uncompressed data, JPEG Lossless exact); Windows Python 3.12 + spawn instead of Linux 3.13 + fork
(spawn is the stricter case); no cv2 / CUDA locally, so the stage readers themselves are not executed (their decode primitive is).

## 6. Gate for the GPU commit: `j_check.py OUT --base-out BASE_OUT`
(A) J and base logs present, J log named rsna-sub-TAG.log; (B) exactly one DEC_OK, before the first stage marker; no DEC_FALLBACK and no
J_UNAVAILABLE anywhere; DEC_OK: tag, pip rc 0, python 3.13, pydicom 3.0.2, decoders 2.1.0 / 2.4.0 / 2.6.0, pylibjpeg for .57 / .70, 3/3
exact; dec_receipt.json: 3 PyPI-identical wheels, the 8 target entries, decoders from /tmp/_dec, pydicom not, PYTHONPATH after ==
/tmp/_dec + ':' + before; (C) submission.csv == the base commit output (ids / columns / order equal, max |diff| <= 1e-6; byte identity
reported, `--strict-bytes` requires it); every other base CSV present and equal (rank blends 1e-6, raw stage outputs incl.
submission_fleetsolo / fleet2g 1e-3: GPU noise MEASURED 4.3e-5 on raptor_raw between two FLIP runs; S6F part files compared by id);
same multiset of log markers (minus DEC_OK / PROGRESS) = same stage / fallback / canary path; no more Traceback / RuntimeError /
FALLBACK / SKIPPED / TIMEOUT / CANARY_FIRED mentions than the base; (D) info: versions, DEC seconds, timings, in-process insert
(`--require-inproc` makes it a check). Prints `J_CHECK PASS n/m tag (submission byte-identical ...)`.
Negatives (`tests/neg_j_check.py TAG BASE_OUT`: clean mock = the REAL base commit output + DEC_OK + the REAL Kaggle DEC receipt of
rsna-v224f20j-check v1 with the new fields mocked): clean PASS (also --strict-bytes, --require-inproc); GPU-noise-level raw edits PASS;
FAIL on DEC_FALLBACK, J_UNAVAILABLE, DEC_OK after stage 1, two DEC_OK, submission +1e-5 or reordered, a rank blend +1e-5, raptor_raw
+.01, a missing stage csv, wrong versions / tag / python, self-test not exact, PYTHONPATH not extended, extra target entry, decoders
not from /tmp/_dec, 2 of 3 wheels, no receipt, log named after the base, a changed stage marker, an extra Traceback.

## 7. GPU commits (PREPARED, NOT RUN; main does GPU commits)
`bash session_2026-10-15/jpeg/overlay/gpu_commit_j.sh TAG` (TAG = v224flipj | v224flips6fj | v224flips20j), from the project root:
asserts the notebook sha == build receipt and the metadata (slug, private, internet off, no id_no), re-runs `run_offline_j.py TAG`
(exact verdict lines only), refuses if RUNNING_COUNT > 1, pushes, waits <= 45 min, fetches to `overlay/out_TAG_v1`, then
`py -3.12 -I session_2026-10-15/jpeg/overlay/j_check.py session_2026-10-15/jpeg/overlay/out_TAG_v1 --base-out <base commit output>`.
Cost (MODEL): the base commit time + ~6 s (base logs end at 481 s FLIP, 591 s S6F, 527 s FLIPS). A PASS means: same public score as
the base (expected exactly equal; the public test is 557/557 uncompressed, MEASURED), J insurance on the private split
(PREREG Amendment 2 rules 2-3: a J entry must score its base's public value; never let J block a pick).

## 8. Open issues / risks
1. The generalised cell itself has not run on the Kaggle image yet (only its predecessor did): the GPU commit + j_check is the proof.
   If DEC fails there, J_UNAVAILABLE = the base (never worse), but the J kernel is then not a J pick (gate FAIL).
2. Run-to-run GPU noise: submission.csv must match within 1e-6 (rank blends on 3 studies; MEASURED max diff 0 between two FLIP runs).
   A FAIL that comes only from stage-level noise flipping a rank would need a main-session look (stage diffs are printed).
3. DEC runs before NB_T0 (no separate clock cell to put it after without editing the r384 cell): worst case ~10 min uncounted by the
   NB_T0 guards (MODEL), typical ~6 s (MEASURED for the predecessor).
4. Third-party dataset glorysing/rsna-knee-offline-codecs-r09: sha-pinned to PyPI; a changed file = J_UNAVAILABLE = base; a deleted
   dataset might stop a rerun from starting (GUESS; S6F already depends on it). pylibjpeg-libjpeg is GPLv3.
5. With the decoders, pydicom prefers pylibjpeg-openjpeg over Pillow for JPEG 2000 (MEASURED identical on lossless + one 16-bit lossy
   fixture; a different lossy stream decoding differently between the two OpenJPEG builds is a GUESS-level residual).
6. FF-S, FLRM, FLRMS20 J variants were not written (dry builds pass the audit): `make_j.py <base dir> overlay/kernel_<tag>j --slug
   eeshsaxena/rsna-sub-<tag>j`; the harness and the gate are generic (FLRM: RM verify chain included), the gate needs that base's commit
   output.

## Files (session_2026-10-15/jpeg/overlay/)
`make_j.py`, `dec_cell_j_template.py`, `j_celldiff_proof.py`, `j_local_harness.py`, `j_check.py`, `run_offline_j.py`,
`gpu_commit_j.sh`, `kernel_v224flipj/`, `kernel_v224flips6fj/`, `kernel_v224flips20j/`, `j_build_receipt_TAG.json`, `celldiff_TAG.txt`,
`celldiff_proof_TAG.txt`, `harness_TAG.txt`, `build_j.log`, `offline_j_all.txt`, `tests/` (`j_sim.py`, `stub_wheels.py`,
`neg_j_check.py` + `neg_j_check_TAG.txt`, `test_make_j.py` + `test_make_j.txt`). Self-test fixtures: `../f20j/selftest_fixtures.json`.
