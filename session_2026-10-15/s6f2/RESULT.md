# s6f2: FLRM-AO + S6fall and FLIP-AO + S6fall + S. Built + offline-gated 10-15 by a Claude subagent; review fix F1 (gate only) applied 10-15 (sec 5, 9). NOT pushed, NOT committed, NOT submitted.

No em dashes. Tags: MEASURED (files / outputs read or run here), MODEL (sim / inference from measured numbers), GUESS, CLAIMED (author text).
Rules followed: PREREG_10-15 Amendments 3-4; each component weight is the pre-registered one (S6fall .2 rank; S as the F slot, w_f .20,
s_share 1.0), never fitted; visible canaries (ACL = 0.5 for an S6fall skip / cut, Synovitis = 0.5 for an S skip, Fracture = 0.5 for
the base's r224 / RM_GUARD canary). Nothing under s6f/, v224f/, v224rm/, jpeg/ or session_2026-10-14/ was written (builder refuses
those dirs; test_builder B5: sha of the 13 s6f files it reads unchanged). Only Kaggle call: one read-only `kaggle kernels pull
eeshsaxena/rsna-sub-v224flips20` into `s6f2/pulled_v224flips20/` (== the local flips20 notebook, sha d2066b8c). Laptop CPU only.

## Result in one block
| item | (1) FLRM-AO + S6fall | (2) FLIP-AO + S6fall + S |
|---|---|---|
| kernel dir | `s6f2/kernel_v224flrms6f/` (rsna_sub_v224flrms6f.ipynb sha **66445ea85a406bbe...**, 20 cells) | `s6f2/kernel_v224flips6fs20/` (rsna_sub_v224flips6fs20.ipynb sha **5f836de171e2b0b2...**, 27 cells) |
| slug | `eeshsaxena/rsna-sub-v224flrms6f` (private, T4 x2, internet off, docker 2757e0c7) | `eeshsaxena/rsna-sub-v224flips6fs20` (same) |
| base (pushed kernel, every cell byte-identical, APPEND only) | V224-FLIP-RM = `rsna-sub-v224flrm` v1 (v224rm/kernel_v224flrm, f14f6acc, 17 cells, == its Kaggle pull; commit gate V224FLRM_CHECK 10/10) | flips20 = `rsna-sub-v224flips20` v1 (v224f/kernel_v224flips20, d2066b8c, 24 cells = V224-FLIP + the 9 S cells, == its Kaggle pull; commit gate V224AOS_CHECK 29/29) |
| appended | the 3 S6F cells of the pushed s6f/kernel_v224flips6f (SETUP / RUN / BLEND, driver a2acdd1d) with 3 documented splices + G (sec 2) | same |
| datasets | base 3 + nartaa/rsna-knee-s6fall-weights + glorysing/rsna-knee-offline-codecs-r09 | base 6 (incl. eeshsaxena/rsna-knee-cnxt-fleet-r7tb) + the same 2 = 8 |
| blend | final = .8 pct(V224-FLIP-RM output) + .2 pct(S6fall) | X = flips20 HF output = .8 pct(V224-FLIP) + .2 pct(mean of the 5 S) (unchanged cells); final = .8 pct(X) + .2 pct(S6fall) |
| stage order | r384 -> R7 flip forced -> RM + r224 -> stage 3 -> V224 canary -> RM_GUARD -> S6F | r384 -> R7 flip forced -> r224 -> stage 3 -> V224 canary -> **S (guard 7.0 h, wall 8.3 h)** -> HF -> **S6F** |
| S6F policy | start guard **G 6.0 h**, budget 3.3 h, hard deadline min(start + 3.3, NB_T0 + 8.2 h), SIGKILL of every S6F process group | **G 6.25 h**, same budget / deadline / kill |
| canaries kept through the S6F blend | ACL (S6F); Fracture from the V224 canary OR RM_GUARD | ACL (S6F); Synovitis from the S / HF fallback; Fracture from the V224 canary |
| runtime (MODEL, session model B / STRESS, 20,000 paired sessions) | end median 6.43 / 6.46 h, **p99 7.81 / 7.81 h**; P(S6F used) **.945 / .854**; P(ACL canary) .055 / .146; P(> 8.3 h) 0 / .0012 (base-driven, == G 5.75) | end median 6.68 / 6.71 h, **p99 7.81 / 7.82 h**; **P(both used) .915 / .810**; P(ACL canary) .085 / .190; P(Synovitis canary) .0007 / .019; P(> 8.3 h) 0 / .0008 (base-driven, == flips20 alone) |
| offline gates (MEASURED, after review fix F1) | cell-diff **S6F2_CELLDIFF PASS 16/16**; mock commit gate **V224S6F2_CHECK PASS 36/36** (nested V224FLRM_CHECK 10/10); MOCK_S6F2 23/23 | cell-diff **PASS 16/16**; mock gate **V224S6F2_CHECK PASS 35/35** (nested V224AOS_CHECK 29/29); MOCK_S6F2 24/24 |
| all tests | **OFFLINE_TESTS_S6F2 PASS 8/8** (`run_offline_tests_s6f2.py`, ~6 min CPU): both proofs, TEST_BUILDER 14/14, NEG_CELLDIFF 18/18, TEST_CELLS 41/41, both mocks, TEST_GATE_REG 6/6 | same run |
| GPU commit (main; never submits) | `bash session_2026-10-15/s6f2/gpu_commit_v224flrms6f.sh` (~10 min T4x2) | `bash session_2026-10-15/s6f2/gpu_commit_v224flips6fs20.sh` (~11 min) |
| gate after the commit | `py -3.12 -I session_2026-10-15/s6f2/v224s6f2_check.py session_2026-10-15/s6f2/out_v224flrms6f_v1 --build session_2026-10-15/s6f2/build_receipt_v224flrms6f.json` -> `V224S6F2_CHECK PASS 36/36 base flrm mode S6F` expected | same with `out_v224flips6fs20_v1` / `build_receipt_v224flips6fs20.json` -> `PASS 35/35 base flips20 mode S6F` expected |

## 1. Builder `build_v224s6f2.py --base flip | flrm | flips20` (s6f/build_v224s6f.py generalised; s6f/ itself untouched)
- Reads the S6F part UNCHANGED from s6f/ (driver source + nartaa v2 verbatim statements, the 3 cell templates, kill-control template,
  DEC wheels / fixtures). Per base: pinned notebook sha, == its Kaggle pull, content asserts on the V224-FLIP cells (by index per base:
  NB_T0, R7 pip, flip forced, r224 run, stage 3, V224 canary), plus flrm: RM after her r224 writefile and RM_GUARD last, flrm minus
  cells 11 / 16 == V224-FLIP; flips20: cells 0-14 == V224-FLIP, the 9 S cells in order (SAVE-AO .. ENV-RESTORE .. HF-AO), MIN_START_MIN
  78, W_F .2, S_SHARE 1.0, BUILD_V224FS.json sha == the base. Every base must END with its done marker (V224_DONE / V224RM_DONE /
  V224F_DONE); no base cell binds an S6F name or sets os.environ PYTHONPATH; only the known cells bind _canary / _gcr / _v224f.
- Submit tags are tied to their base and the reviewed policy (flip 5.75, flrm 6.0, flips20 6.25; budget 3.3, deadline 8.2, w .2,
  coverage .98, ACL): anything else needs another tag. Refuses an --out inside a pushed-kernel dir, the old cut-control flags, a kill
  control on a submit tag (test_builder B3, 8 refusals).
- **FLIP path byte-identical (MEASURED, test_builder B1):** `--base flip` rebuilds s6f/kernel_v224flips6f exactly: notebook
  **9c0e2558d51e4adb...**, kernel-metadata, build_receipt_v224flips6f.json, celldiff_v224flips6f.txt, s6f_driver_built (5 of 5 files).
  flrm / flips20 rebuild deterministically (B2), and their receipts record the s6f template shas (== the current files).
- Commands: `py -3.12 -I session_2026-10-15/s6f2/build_v224s6f2.py --base flrm` and `--base flips20` (defaults = the submit builds;
  logs `build_v224flrms6f.log`, `build_v224flips6fs20.log`).

## 2. The 3 appended cells vs the pushed S6F cells (proof: `celldiff_s6f2_proof.py --kernel flrm|flips20`, independent of the builder)
Undoing exactly these gives the s6f cells 15-17 byte for byte (and each must be present once; the proof fails if one is missing):
1. a 2-line header on each cell naming the base ("V224-FLIP" in the S6F text means the output of the cells above; the frozen file
   keeps the name submission_v224flip.csv and the canary mode keeps the name V224FLIP_CANARY, so gate / tests / receipts stay one form);
2. tag literal (v224flrms6f / v224flips6fs20) in the SETUP and BLEND receipts; RUN: G 6.0 / 6.25 instead of 5.75;
3. SETUP: `rec['base_fired']` = every earlier canary of the base, from `_canary` (V224 CANARY), `_gcr` (RM_GUARD's receipt dict,
   flrm) and `_v224f` (HF receipt dict, flips20: 'fleet->Synovitis' on an S fallback); BLEND: `blank` = every label a base canary
   fired on + any constant base column (instead of "Fracture if _canary fired r224->Fracture"); blanked columns stay 0.5 after the blend.
**Why the splice is required (MEASURED, test_cells R2 / S2 CONTROL):** the unchanged S6F cells on a base where RM_GUARD fired
(Fracture 0.5 written into submission.csv but NOT into `_canary`) or where the S stage fell back (Synovitis 0.5) re-rank the constant
column and blend S6fall into it: Fracture / Synovitis came out [.667, .733, .600], i.e. the base canary silently disappears. The same
reason made the flips20 builder add its HF-AO rule. With the splice both stay 0.5 (R2, S2, S3), and an S6F skip on top keeps both
canaries (R6: ACL + Fracture, S6: ACL + Synovitis; every other column == the base text).
Proof results (MEASURED): base sha == pushed + == Kaggle pull; flrm minus RM / RM_GUARD == V224-FLIP / flips20 cells 0-14 == V224-FLIP;
alignment `equal base[0:N] -> new[0:N]`, `insert new[N:N+3]` (N 17 / 24); new minus the 3 cells re-serialises to the base FILE byte for
byte; notebook metadata equal; driver a2acdd1d == the pushed S6F kernel's == receipt; kernel-metadata == base's + the 2 datasets:
**S6F2_CELLDIFF PASS 16/16** for both (`celldiff_proof_flrm.txt`, `celldiff_proof_flips20.txt`). `tests/neg_celldiff_s6f2.py`
**18/18**: clean copies PASS; FAIL on a base cell +1 byte, S6F cells before the base end, codec dataset dropped, driver changed,
FLIP blank rule restored, G 5.75 restored, an extra statement, a receipt of another build. (The first version of the proof passed the
two "splice absent" cases: undo was replace-if-present. Fixed: each splice must be present exactly once; pre-fix copy
`celldiff_s6f2_proof.py.pre_negfix`.)

## 3. (2) FLIP-AO + S6fall + S: order, blend, canaries
- **Order = S first, S6F second** (the composite is the pushed flips20 notebook + the 3 S6F cells appended, so its first 24 cells are
  byte-identical to `rsna-sub-v224flips20` v1 and the S read is exactly flips20's). Decided by runtime (MODEL, paired sim, model B):
  S first gives P(both used) **.915** (S used .9993, S6F used .915); S6F first gives **.810** (S6F used .949, but S is starved: used
  .840, the S guard 7.0 h is hit after a 1.2-2.4 h S6fall stage). STRESS .810 vs .703; N 1,500 .607 vs .389; c 13-16 s .644 vs .350.
  The cheap stage (S, ~0.3 h) first costs S6fall only ~0.3 h of start time; the expensive stage first starves S. Also S6F-first would
  need a splice in the HF cell (the HF rank blend would erase the ACL canary), S-first needs none in the S cells.
- **Blend (fixed weights, documented order):** H = V224-FLIP output (SAVE: submission_V224.csv). Step 1 (flips20 HF-AO, unchanged):
  X = per label (1 - .20) pct(H) + .20 pct(F), F = mean of the 5 S raw probs (s_share 1.0), float32, not re-ranked; Fracture back to
  0.5 if the V224 canary fired; on any S problem X = H with Synovitis 0.5 (HF_FALLBACK). Step 2 (S6F, appended): final = per label .8
  pct(X) + .2 pct(S6fall), float32, not re-ranked; an unscored study takes its X rank; every column a base canary blanked stays 0.5;
  on any S6F problem final = X with ACL 0.5. Approximate rank shares H .64, S .16, S6fall .20 (MODEL: pct of a blend is not linear).
  Removing S6F (deleting cells 24-26) gives flips20 byte for byte; removing S does not give the S6F entry (not required).
- **Guards:** S stage unchanged (starts only at kernel age <= 7.0 h, MIN_START_MIN 78, wall 8.3 h + 300 s kill). S6F start guard
  6.25 h, hard deadline min(start + 3.3 h, NB_T0 + 8.2 h) with the process-group SIGKILL, compact_infer takes no new study in the
  last 24 min before it. S6F cannot push the kernel past ~8.24 h. The only path past 8.3 h is the inherited base tail (R7 deadline 8.3 h / S wall
  8.3 h + 300 s, kernel <= ~8.39-8.40 h incl. the S6F skip, < 9 h): P(> 8.3 h) 0 (B) / .0008 (STRESS), identical to flips20 alone
  (MODEL). So "never exceeds 8.3 h" holds for everything S6F adds; the residual tail is the pushed base's own.

## 4. Runtime (`sim_s6f2_runtime.py` -> `sim_s6f2_runtime.txt`; MODEL on MEASURED inputs; 20,000 paired sessions per model)
- Model pieces copied from the reviewed sims: base V224-FLIP (always-on flip) calibrated to its MEASURED 4.57-4.65 h submit-to-score,
  her R7 deadline 8.3 h / r224 skip 7.83 h, RM +U(.06, .16) h (v224f/sim_ffs_runtime.py D); S stage 1.5 min + 650 x 1.62 s x mf x s
  (MEASURED microbench, 5 TTR224; ~0.3 h central), guard 7.0, wall 8.3 + 300 s; S6fall c ~ U(6.5, 13) s / study / T4, setup .03-.08 h,
  2 parts, no new study after hard - 24 min, kill at hard (s6f/sim_s6f_runtime.py). New: common random numbers across configs (each
  session draws one set and every config is evaluated on it), so G differences are not sampling noise; paired monotonicity check:
  0 sessions used at a lower G but a canary at a higher G (both kernels, both models).
- **G choice.** Rule: G = the FLIP-S6F guard 5.75 h shifted by the extra pre-S6F stage cost at a slow (x1.3) session, rounded up to
  the grid (RM ~0.16 h -> 6.0; S ~0.33 h -> 6.25), checked in the paired sim to sit at the no-guard plateau of P(S6F used) without
  moving p99:

| config | P(S6F used) B | STRESS | N 1,500 (B) | end median / p99 (B) |
|---|---|---|---|---|
| FLIP + S6F G 5.75 (pushed s6f, reference) | .9522 | .8642 | | 6.31 / 7.81 |
| FLRM + S6F G 5.75 / **6.0** / 6.25 / none | .9413 / **.9446** / .9449 / .9449 | .8454 / **.8535** / .8545 / .8545 | .716 / **.731** / .734 / | 6.41 / **6.43** / 6.44 / 6.45, p99 7.81 all |
| S then S6F, G 5.75 / 6.0 / 6.1 / **6.25** / none | .9062 / .9133 / .9143 / **.9146** / .9146 | .7914 / .8063 / .8088 / **.8100** / .8100 | .551 / .598 / / **.607** / | 6.63 / 6.66 / 6.67 / **6.68** / 6.71, p99 7.81 all |
| S6F then S (alternative), G 5.75 | .9492 (S used .8404, both .8095) | .8669 (S .7783, both .7026) | .769 (S .486, both .389) | 6.68 / 7.83 |

- (1) FLRM-AO + S6fall at G 6.0, model B: end median 6.43 h, p90 7.34, **p99 7.81**, max 7.97; P(S6F used) .9446, P(skip) .0245,
  P(cut) .0309 -> **P(ACL canary) .055**; P(> 8.0 h) 0. STRESS: used .854, skip .083, cut .064 -> P(ACL) .146; p99 7.81, P(> 8.3 h)
  .0012 (base-driven: the same at G 5.75), Fracture canary (base r224 skip) .0017. Sensitivity (B): c 6.5-9 s used .969; c 10-13 .916;
  c 13-16 .735; N 1,500 .731.
- (2) FLIP-AO + S6fall + S at G 6.25, model B: end median 6.68 h, p90 7.60, **p99 7.81**, max 8.02; **P(both used) .915**; P(ACL
  canary) .085 (skip .029, cut .057); **P(Synovitis canary) .0007**; P(Fracture canary) 0; P(> 8.0 h) .0001, P(> 8.3 h) 0. STRESS:
  both .810, ACL .190, Synovitis .019, Fracture .002, p99 7.82, max 8.37, P(> 8.3 h) .0008 (== flips20 alone). Sensitivity (B):
  c 6.5-9 both .957; c 10-13 .868; c 13-16 .644; N 1,500 .607 (S .986).
- Submit-to-score = kernel end + queue (0-0.3 h): median ~6.6 h (1) / ~6.8 h (2) (MODEL; V224-FLIP MEASURED 4.6 h). P(S6F used) is
  lower than the pushed FLIP-S6F entry (.952): the S6F stage starts ~0.1 h (RM) / ~0.35 h (S) later against the same 8.2 h deadline.
- Recalibration: the first hidden latency of the pushed S6F entry (submit-to-score - 4.6 h = its S6F wall) re-scales C_LO / C_HI here
  as for s6f (RECAL rule unchanged in the gate).

## 5. Gate `v224s6f2_check.py OUT --build build_receipt_<tag>.json` (s6f gate generalised; base from the receipt)
- New: build receipt base kind + base sha; the runtime-reviewed submit policy for that base (G 5.75 / 6.0 / 6.25, budget 3.3, deadline
  8.2, margins, w .2, coverage .98, ACL); (0, flrm / flips20) the kernel notebook sha == receipt and minus its 3 cells it re-serialises
  to the pinned base FILE; (A) the frozen output + the log up to the base's last marker pass the base's own UNCHANGED commit gate on a
  view without the S6F files: flrm -> `v224rm/v224flrm_check.py` (10/10 on its commit), flips20 -> `v224f/v224aos_check.py --wf .20
  --build v224f/kernel_v224flips20/BUILD_V224FS.json --ref-s v224f/out_v224flips20_v1` (29/29: V224-FLIP part, flip on every study,
  S members / shas, END-TO-END (1 - .2) pct(H) + .2 pct(S), canaries, S guard); stage csvs == the base v1 commit's (R7 / r384 1e-3, r224
  ranks 1e-6, flrm also r224 raw 1e-3); freeze after the last marker and no S6F marker before it; (C) the notebook's base-canary
  blank rule re-derived from the receipts + frozen base, and in clean mode base_fired [] / v224_canary fired [] / base_blanked [].
- Unchanged from the s6f gate: (B) parity vs nartaa k1/B/B_probs.csv max |dp| <= 1e-3, (C) blend recompute 1e-6, (D) coverage, (E)
  receipts / env isolation / versions / DEC / B probe / timings / worker receipts / RECAL info, (F) canary, (G) markers (base last
  marker first), (H) format; `--expect-canary [--control skip|kill]` for control builds.
- Flip mode == the s6f gate (MEASURED, tests/test_gate_regression_s6f2.py 6/6): on the REAL S6F v1 commit, the s6f mock and the s6f
  canary mock the old gate gives PASS 31/31 / 31/31 / 18/18 and this gate PASS 32/32 / 32/32 / 18/18 (one extra check: the reviewed
  policy); every cross-base receipt FAILs.
- **Review fix F1 (10-15, gate only; notebooks, builder, receipts and proofs unchanged; pre-change copy `v224s6f2_check.py.pre_f1`).**
  Gap (MEASURED): the pre-F1 gate's (E) only required PYTHONPATH to START with /tmp/s6f_dec and forbade R7 / working dirs, so an S-stage
  dir on S6fall's import path passed: the reviewer's mutation (`/tmp/s6f_dec:/tmp/k12_fleet` in the setup receipts and the driver
  env, fleet dir on the driver sys.path) gave PASS 34/34 (reproduced here on a scratch copy). A fleet dir there could shadow modules
  of nartaa's code tree that are imported by name (common.py, native_data.py, ...); only the 3-study parity check could catch that.
  Fix: (E) forbids `/tmp/k12_fleet*` and `/tmp/k2_members*` (the K12 / S-MERGE fleet dir and K2's mkdtemp member dirs) on every
  import path incl. the setup receipt's paths (all bases; no flip verdict changes); and for flrm / flips20 a new check: the kernel
  PYTHONPATH the setup saw == `['/kaggle/lib/kagglegym', '/kaggle/lib', '/kaggle/input/competitions/rsna-knee-abnormality-detection']`
  (the S6F v1 commit's, MEASURED in s6f/out_v224flips6f_v1/s6f_setup_receipt.json; docker pin 2757e0c7 is the same in both s6f2
  kernels and the S6F kernel) and S6fall's PYTHONPATH == `/tmp/s6f_dec:` + exactly that, equal in all 4 receipt copies (setup file,
  s6f_receipt.setup, driver receipt env_isolation, s6f_receipt.run.driver). Not applied in flip mode (the s6f mock has the laptop
  form, and flip mode must stay == the s6f gate). So the commit itself now closes open issue 1(b): a leak of the S stage (or of RM)
  into os.environ PYTHONPATH or into the subprocess sys.path FAILs the gate. Gate counts: flrm 35 -> **36**, flips20 34 -> **35**.
  If the commit FAILs only this check with a benign image-path change, it is a gate false-FAIL to review, not a kernel fault (GUESS
  unlikely: the image is pinned by digest).
  Fail-before / pass-after (MEASURED): the mocks now carry the real Kaggle import path (setup / driver receipts copied in form from
  the S6F v1 commit) and 3 new negatives, each consistent across every receipt copy: N18 /tmp/k12_fleet leaked into the kernel
  PYTHONPATH (the K2 per-worker form), N19 a /tmp/k2_members_* dir on the subprocess sys.path only (PYTHONPATH clean), N20 the S
  dataset dir /kaggle/input/rsna-knee-cnxt-fleet-r7tb appended to the kernel PYTHONPATH (no forbidden prefix: only the exact rule
  catches it). Pre-F1 gate: MOCK_S6F2 **FAIL 20/23** (flrm) / **21/24** (flips20), exactly N18-N20 failing (the gate PASSed them
  35/35 / 34/34; `tests/mock_s6f2_output_<kind>_failbefore_f1.txt`). Fixed gate: **23/23** / **24/24**; N18 fails both env checks,
  N19 only the forbidden-dir check, N20 only the exact-path check; positives 36/36 / 35/35, controls 20/20 / 19/19 unchanged.

## 6. Tests (all MEASURED, laptop CPU, `py -3.12 -I`; `run_offline_tests_s6f2.py` -> **OFFLINE_TESTS_S6F2 PASS 8/8**, `offline_tests_s6f2.txt`)
| test | result | what |
|---|---|---|
| `celldiff_s6f2_proof.py --kernel flrm` / `flips20` | 16/16, 16/16 | sec 2 |
| `tests/test_builder_s6f2.py` | 14/14 | B1 FLIP byte-identical (5 files), B2 deterministic + template shas, B3 8 refusals, B4 flrm skip / flips20 kill control builds (RUN == the s6f kill-control RUN + header + G), B5 s6f/ unchanged |
| `tests/neg_celldiff_s6f2.py` | 18/18 | sec 2 |
| `tests/test_cells_s6f2.py` | 41/41 | the appended cells executed as written (paths redirected, real driver port on nartaa's real part outputs, base = the REAL base commit submission): R1-R7 FLRM (clean == recompute 2.5e-8; RM_GUARD / V224 canary Fracture kept; age 5.9 h runs, 6.1 h skips; skip + RM_GUARD both canaries; hung driver killed ~15 s), S1-S8 composite (clean 2.4e-8; HF fallback Synovitis kept; + Fracture; age 6.2 runs, 6.3 skips; skip + HF fallback both canaries; constant column kept; driver failure), 2 CONTROLs with the unchanged S6F cells (canary lost) |
| `tests/mock_s6f2_output.py flrm` | 23/23 (F1: was 20/20; 20/23 on the pre-F1 gate) | mock commit = the REAL FLRM v1 output + the executed cells (mock-only: Kaggle DEC_OK form; F1: the S6F v1 commit's Kaggle import path in every receipt copy): PASS 36/36; 17 negatives each FAIL on the intended check (parity +2e-3, submission +1e-5, canary fired, guard skip, frozen base changed, R7 dir on PYTHONPATH, freeze before V224RM_DONE, compact_infer 8401812c, run G 5.75, weight .3, DEC fallback, F1 N18-N20 S-stage / S dataset dir on S6fall's import path, receipt G 5.75, RM_GUARD fired kept / un-blanked, r224 raw +2e-3); permuted B_probs PASS; skip-control mock (age 6.1) PASS 20/20 with --expect-canary and FAILs the normal gate |
| `tests/mock_s6f2_output.py flips20` | 24/24 (F1: was 21/21; 21/24 pre-F1) | the same on the REAL flips20 v1 output: PASS 35/35; base negatives: HF fallback kept / un-blanked, S6F blended on H instead of the HF output (S dropped), S npz halved; N18-N20; control mock (age 6.35) PASS 19/19 |
| `tests/test_gate_regression_s6f2.py` | 6/6 | sec 5 |
Step 0 of both commit scripts executed alone (MEASURED; truncated copies that stop before step 1, so no running check, no push):
kernel sha OK 66445ea8 / 5f836de1, S6F2_CELLDIFF PASS, OFFLINE_TESTS_S6F2 PASS 8/8 each, rc 0 (`step0_dryrun_both.log`,
`offline_tests_before_push_v224flrms6f.txt`, `offline_tests_before_push_v224flips6fs20.txt`; these two files are from BEFORE fix F1
and step 0 rewrites them at commit time). After F1 (MEASURED): `run_offline_tests_s6f2.py` -> **OFFLINE_TESTS_S6F2 PASS 8/8**
(6 min 22 s; `offline_tests_s6f2.txt`), and step 0 up to and incl. the cell-diff proof (scratch copies, no suite, no push) rc 0 for
both (kernel sha OK 66445ea8 / 5f836de1). Bash note for whoever edits these files:
this shell layer strips one level of backslashes inside heredocs; edit with a file editor, not a heredoc patch.

## 7. Commit + gate commands (main runs them; they never submit)
- `bash session_2026-10-15/s6f2/gpu_commit_v224flrms6f.sh` and `bash session_2026-10-15/s6f2/gpu_commit_v224flips6fs20.sh` (pattern of
  s6f/gpu_commit_v224flips6f.sh rfix form): step 0 asserts notebook sha == receipt, metadata id / private / internet off, base kind and
  G, then pipes the cell-diff proof and `run_offline_tests_s6f2.py` (~6 min) into the exact `verdict_ok`; step 1 RUNNING_COUNT <= 1;
  step 2 `kaggle kernels push -p session_2026-10-15/s6f2/kernel_<tag>` ("successfully pushed" required); step 3 `wait_fetch.py` into
  `s6f2/out_<tag>_v1`; step 4 the gate (expected PASS 36/36 flrm, 35/35 flips20; 35 / 34 before review fix F1). One commit each, ~10-11 min of T4x2 (MODEL: base
  commit 432 s / 508 s + S6F ~2 min as measured on the S6F v1 commit). The two scripts both re-run the same offline suite; run them one
  after the other (the suite rewrites shared test outputs).
- Submit (NOT run here; main decides, after the commit gate PASS and an amendment): `python session_2026-10-14/submit_at.py
  eeshsaxena/rsna-sub-v224flrms6f 1 "..."` / `... rsna-sub-v224flips6fs20 1 "..."`.

## 8. Open issues / concerns
1. Not commit-tested yet (no GPU here): everything above is offline on real base outputs + nartaa's real part outputs. The commit is the
   first run of S6F after RM_GUARD / after the S stage on Kaggle. Specific things only the commit shows: (a) flips20's fleet cell
   imports torch in the notebook process (`torch.cuda.device_count()`); GPU memory left for the S6fall workers is GUESS fine (the
   S6F commit after V224 used 2 T4 without issue); downgraded by review F4: K2 calls only `torch.cuda.is_available()` /
   `device_count()` and `empty_cache()` under `if n_dev` (MEASURED in the cell text; that these create no CUDA context in the
   notebook process is GUESS from torch's lazy init), and the workers are separate processes that are killed + waited in K2's
   `finally`; (b) ENV-RESTORE must leave os.environ as before the S stage: since review fix F1 the commit gate CHECKS this for
   S6fall's import path (exact image PYTHONPATH in every receipt copy, no /tmp/k12_fleet or /tmp/k2_members dir; sec 5), so the
   commit closes it; the thread / CUDNN variables FN-IDS sets are still not recorded by any S6F receipt (their restore is covered
   only by the nested v224fs_check's HF_ENV_RESTORED marker check); code reading (MEASURED cell text) found no leak path: K2
   puts FLEET_* and the fleet PYTHONPATH only into its per-worker env dict, ENV-RESTORE deletes / restores every key against the
   c16 snapshot; (c) the flips20 base log contains 'V224_DONE' before 'V224F_DONE' and the cut / marker logic keys on V224F_DONE
   (tested on the real flips20 log in the mock); (d) disk: the S stage's cells keep no cache (K12 / S merge dirs and the per-stage
   member dir are symlinks, the member dir is removed in K2's `finally`; /tmp/k12_fleet stays, symlinks only; MEASURED in the cell
   text); the fleet worker uses a ThreadPoolExecutor(4), no subprocess / pip / temp dir (MEASURED in the local copy
   jpeg/src/fleet/fleet_worker.py, sha 21effe04; that the k11w dataset copy is identical is GUESS); S6fall's px_cache then needs the
   same /tmp room as in the pushed FLIP-S6F entry. The POSIX process-group kill in RUN has still never run on Linux (inherited,
   s6f/RESULT.md item 3); if it fails, RUN still continues after `p.wait(timeout=120)`, so the kernel ends.
2. Both entries are READs, not pick-eligible under SILVER_PLAN sec 4 (no J / DEC cell in the base stages; gate info line
   `PICK_ELIGIBLE_J no`), exactly as the pushed S6F entry. (2) is two changes vs V224-FLIP (S + S6fall), i.e. the Amendment 3 R5
   "same-cycle composite / stacking read" form; (1) is one change vs V224-FLIP-RM (R4 B1 "two kernels", base FLRM). Each needs its
   amendment text before a submit (main). Note (MEASURED, read only): another workflow has since committed J overlays in
   `jpeg/overlay/r2/` for v224flipj, v224flips20j and v224flips6fj (J_READY files, J_CHECK PASS). A pick-eligible form of these two
   entries would be the same overlay applied to `s6f2/kernel_v224flrms6f` / `kernel_v224flips6fs20` (+ its own commit and J_CHECK);
   not done here (out of scope; it would also need the s6f2 gate's base last marker / view logic to accept the J base).
3. P(visible canary) is higher than for the pushed FLIP-S6F entry: ACL .055 (1) / .085 (2) in model B, .15 / .19 STRESS; with 1,500
   hidden studies (2) uses S6fall only .61 (MODEL). All from one V224-FLIP latency sample and a wide S6fall cost range (c 6.5-13 s
   GUESS-bounded). The S6F hidden latency of the pushed entry (when it scores) is the calibration to re-run the sim with.
4. Environment: as in s6f, S6fall sees timm 1.0.25 / pydicom 3.0.1 (R7's global pip), not nartaa's 1.0.29 / 3.0.2; parity was max |dp|
   0 on the S6F v1 commit (MEASURED there); gate (B) re-checks on each commit.
5. Expected gain GUESS: S6fall +3 to +10e-4 public (SILVER_PLAN lever 1, as s6f); S unknown (student). In (2) the two are confounded by
   design (composite read); a best-id move by (2) does not say which component moved it.
6. The S6F text in the appended cells still says "V224-FLIP" (and the frozen file / canary mode keep their names): one form for gate,
   tests and receipts; the header line on each cell names the actual base.
7. Inherited S6F fallback: if the frozen copy cannot be read at blend time, the BLEND cell writes its canary from the in-memory `sub`
   (her stage-3 output). On these bases that is V224-FLIP(-RM) without RM_GUARD / without S, but always with ACL 0.5, so the entry is
   visibly broken either way (MODEL: reachable only on a disk failure after the setup's sha-checked copy).
8. Other canary probabilities (MODEL / GUESS): (1) Fracture via RM_GUARD (r224 used without a valid mirror receipt) < 1 % (GUESS,
   v224rm), via the base r224 skip 0 (B) / .0017 (STRESS); (2) Fracture via the base r224 skip 0 / .0022. The sim does not model
   RM_GUARD or S6fall worker crashes (both end in their visible canaries).

## 9. Review 10-15 (findings F1, F4): verdicts
- **F1 (medium): CONFIRMED, FIXED in the gate only** (sec 5). Reproduced on a scratch copy of the flips20 mock: pre-F1 PASS 34/34 with
  /tmp/k12_fleet on S6fall's PYTHONPATH and driver sys.path; post-F1 FAIL 33/35 (both env checks). Notebooks 66445ea8 / 5f836de1,
  builder, build receipts, driver a2acdd1d and the cell-diff proofs are unchanged (MEASURED: notebook shas == receipts; proofs
  16/16). Edited: `v224s6f2_check.py` (pre: `.pre_f1`), `tests/mock_s6f2_output.py` (pre: `.pre_f1`), this file (pre: `RESULT.md.pre_f1`).
- **F4 (info): verified, no action.** (a) base cells unchanged, appended cells output-free, metadata == base + 2 datasets: already
  MEASURED by both cell-diff proofs (16/16, re-run). (b) blend order / weights / base_fired rule: MEASURED by test_cells 41/41 and the
  mocks (re-run). (c) gate baselines: mock PASS 36/36 / 35/35 after F1 (re-run). (d) the G plateau with fast S6fall (c 6.5-9 s):
  the reviewer's MODEL run (FLRM G 6.0 .9715 vs no guard .9724; composite G 6.25 == no guard .9630 B / .8782 STRESS), not re-run
  here; consistent with sec 4's c 6.5-9 sensitivity (.969 / .957). (e) base tails past 8.3 h are the base's own (sec 3, 4). The
  S-stage code facts (K2 torch calls, `finally` kill + wait + member-dir removal, ENV-RESTORE, fleet worker threads only, RUN's
  `p.wait(timeout=120)` after the kill) were re-read here in the cell text / local worker copy (MEASURED as text) and folded into
  open issue 1. Optional (reviewer, agreed): wait for the pushed S6F entry 57201874's hidden latency before choosing which of the
  two GPU commits to spend first; P(ACL canary) for (2) is .085 B / .19 STRESS / .39 at N 1,500 (MODEL).

## Files (session_2026-10-15/s6f2/)
`build_v224s6f2.py`, `build_v224flrms6f.log`, `build_v224flips6fs20.log`, `kernel_v224flrms6f/`, `kernel_v224flips6fs20/`,
`build_receipt_v224flrms6f.json`, `build_receipt_v224flips6fs20.json`, `celldiff_v224flrms6f.txt`, `celldiff_v224flips6fs20.txt`
(full appended text + driver), `s6f_driver_built_v224flrms6f.py` / `..._v224flips6fs20.py` (== a2acdd1d), `pulled_v224flips20/`,
`celldiff_s6f2_proof.py` (+ `.pre_negfix`), `celldiff_proof_flrm.txt`, `celldiff_proof_flips20.txt`, `v224s6f2_check.py` (+ `.pre_f1`),
`RESULT.md.pre_f1`, `tests/mock_s6f2_output.py.pre_f1`, `tests/mock_s6f2_output_<kind>_failbefore_f1.txt` (F1 fail-before runs),
`sim_s6f2_runtime.py` + `sim_s6f2_runtime.txt`, `run_offline_tests_s6f2.py`, `offline_tests_s6f2.txt`, `gpu_commit_v224flrms6f.sh`,
`gpu_commit_v224flips6fs20.sh`, `offline_tests_before_push_<tag>.txt`, `step0_dryrun_both.log`, `tests/` (s6f2_harness.py, test_cells_s6f2.py, test_builder_s6f2.py,
neg_celldiff_s6f2.py, mock_s6f2_output.py, test_gate_regression_s6f2.py, their .txt outputs, mock_out_v224flrms6f(_canary)/,
mock_out_v224flips6fs20(_canary)/, mock_s6f2_gate_positive_<kind>.txt, mock_s6f2_gate_control_<kind>.txt).
