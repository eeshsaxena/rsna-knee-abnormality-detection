# V224-FLIP-AO + S6fall (eeshsaxena/rsna-sub-v224flips6f): built + offline-gated 10-15 by a Claude subagent. NOT pushed, NOT committed, NOT submitted.

**Update ~04:00 IST 10-15 (review round "rfix", findings S6F-R1..R4, sec 7).** MEASURED from the logs: main ran
`gpu_commit_v224flips6f.sh` at 03:30 IST; v1 pushed, commit COMPLETE 03:41, gate **V224S6F_CHECK PASS 31/31 mode S6F**
(`gate_v224flips6f_v1.txt`; the fixed gate gives the same 31 check lines + the new info lines: `gate_v224flips6f_v1_rfix.txt`).
Parity max |dp| 0 vs nartaa. Not submitted. The fixes change no byte of the pushed kernel (default build re-derived byte-identical,
notebook sha 9c0e2558...). Two consequences for main: this entry is a **READ only, not pick-eligible** (no J in the base stages,
gate line `PICK_ELIGIBLE_J no`); the runtime recalibration rule reads **RECAL_TRIGGER no** on the real commit (sec 5).

No em dashes. Tags: MEASURED (files / outputs read here), MODEL (sim / inference from measured numbers), GUESS, CLAIMED (author text).
PREREG_10-15 Amendment 3 R4 (gates i-vi) + SILVER_PLAN sec 2 B1. Base = V224-FLIP (flip FORCED = always on; "AO" needs no change).

## Result in one block
| item | value |
|---|---|
| kernel dir | `session_2026-10-15/s6f/kernel_v224flips6f/` (rsna_sub_v224flips6f.ipynb sha256 **9c0e2558d51e4adb...**, 18 cells; kernel-metadata.json) |
| slug | `eeshsaxena/rsna-sub-v224flips6f` (private, NvidiaTeslaT4 x2, internet off, docker pin 2757e0c7 = V224-FLIP's) |
| composition | the submitted V224-FLIP notebook (sha 73a9d562, 15 cells byte-identical, same order r384 -> R7 flip forced -> r224 -> stage 3 -> V224 canary) + 3 APPENDED cells (S6F SETUP / RUN / BLEND) |
| blend | per label `.8 pct(V224-FLIP) + .2 pct(S6fall)`, float32, not re-ranked (HF convention); S6fall ranked over its scored studies, an unscored study takes its V224-FLIP rank; weight fixed, never fitted |
| time policy | start guard **G = 5.75 h** kernel age; hard deadline **min(start + 3.3 h, NB_T0 + 8.2 h)** with SIGKILL of every S6F process group; compact_infer takes no new study after hard - 24 min |
| canary | any skip / failure / coverage < 98 % / sanity-floor fail -> V224-FLIP output with **ACL = 0.5 on every row** (visible on the LB); receipt + `S6F_SKIPPED` / `S6F_CANARY_FIRED` in the log |
| offline gates (MEASURED here) | cell-diff proof **S6F_CELLDIFF PASS 17/17**; driver port on nartaa's real part outputs **TEST_DRIVER PASS 15/15**; appended cells executed **TEST_CELLS PASS 35/35**; gate on a mock commit output + 16 negatives + 2 positives + controls **MOCK_S6F PASS 21/21**; proof negatives **NEG_CELLDIFF PASS 6/6**; rfix: **TEST_CONTROLS PASS 18/18**, **TEST_GATE_RECAL PASS 10/10**, **TEST_STEP0 PASS 7/7**; all re-run by `run_offline_tests.py` -> **OFFLINE_TESTS PASS 7/7** |
| runtime (MODEL) | kernel end median 6.30 h, p90 7.18, **p99 7.81**, max 7.81 h (no new S6 study after the deadline - 24 min); P(S6F used) .96, P(skip canary) .02, P(cut canary) .02; submit-to-score median ~6.45 h |
| GPU commit | DONE by main 03:30-03:45 IST (v1, PASS 31/31). Script: `bash session_2026-10-15/s6f/gpu_commit_v224flips6f.sh` (asserts, re-run of every offline test, push, wait, fetch, gate); a re-run pushes a new version (rename the _v1 paths first). |
| gate | `py -3.12 -I session_2026-10-15/s6f/v224s6f_check.py session_2026-10-15/s6f/out_v224flips6f_v1` -> `V224S6F_CHECK PASS 31/31 mode S6F` expected |

## 1. Source: nartaa/rsna-knee-s6fall-solo (all MEASURED unless tagged)
- Pulled read-only (`pull_s6fall.py` via watch/kgweb.py: python API with `version_label`, the CLI `/N` form returns 403): `pulled_v1/` (sha 3d4b4501),
  `pulled_v2/` = `pulled_cur/` (sha **6e87350d**), version table + view model in `pull_info.json`.
  - **v1** (run 2026-10-12 07:12 UTC, 159 s commit, T4, pin 2757e0c7) = the **scored 0.950** (best_version 1). Its run pinned two then-private
    datasets (nartaa/rsna-knee-s6f0-infer, nartaa/rsna-knee-oai-original384-s42-pair).
  - **v2** (run 2026-10-14 19:13 UTC, 158 s, same pin) = all-public re-run (s6fall-weights + heliosli orthofoundation + nartaa SWA), not
    submitted (cur_version_score null). v1 vs v2 code diff: only DS_SLUG / HELIOS_SLUG / EXTRAS_FILE and prepare_inputs' code-tree
    assembly (v1: one private payload; v2: s6_extras.bin + heliosli's dataset, every file checked against the scored sha256, CLAIMED
    in its header). Model, B_GEOMETRY, worker args, launcher, collect / blend: identical text.
- Commit output (`kaggle kernels output`, = v2's) in `ref_output/`: k1/B/B_probs.csv on the 3 visible studies is the parity reference;
  its k1_log self sha == 6e87350d (the pulled source). Env of that run: python 3.13.15, torch 2.11.0+cu128, **timm 1.0.29, pydicom 3.0.2
  (stock image)**, no pylibjpeg / gdcm. B stage 64 s for 3 studies over 2 T4 (parts [2, 1]).
- **B path** (the only part ported): `prepare_inputs` (mounts, model sha, anatomy sha, run_config / DONE checks, code tree = s6_extras.bin
  + heliosli files, each sha-checked; SOURCE_SHA / CODE_SHA; launcher), `probe_b_stack` (CPU: launcher, compact_infer import, vendored
  **MONAI 0.6.0** V-Net + anatomy.pt forward, compact_train.Net strict load, 50,018,284 params), `run_b` (split_parts by series count,
  one `compact_infer` worker per GPU via `b_cmd` / `b_env`, `collect_part` checks, z = mean logit, P = sigmoid(z), B_probs.csv).
- Environment pins: compact_infer **34f433fe72b7...**, compact_train 14767c7a, px_cache b13e38a7, probe_infer a6cc3144, knee_anchor_bank_v2
  e564807a (heliosli's public S6 family; extracted for reading from a public heliosli notebook into `src_extract/`), cloud_code CODE_SHA
  (8 files, identical to R7's), anatomy.pt 5904dd16 (= the R7 manifest's anchor sha), code_manifest.json **b66c574b5c0c13cc...** (pins
  all 293 members; from nartaa's commit receipt), checkpoint `E_prior_s42_fall.pt` sha
  **0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12** (== checkpoint-manifest.json == DONE.json == the kernel
  constant; asserted by prepare_inputs on Kaggle and by the builder). Dataset files listing (10-14 19:08 UTC, before the v2 run) in this session.
- Outputs / label order: `B_probs.csv` = StudyInstanceUID, the 12 probabilities in LAB order (ACL, MCL, Medial Meniscus, Lateral Meniscus,
  Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture = sample_submission order), then 12 `z|<label>` logits,
  `%.17g`. Our gate and blend read labels BY NAME anyway.
- Never included (builder + proof assert, code tokens): `remove_submission`, the hang guard (`_watchdog`, `arm_guard`, faulthandler
  backstop, GUARD_*), component A (`run_a`, A_B64), nartaa's `run_b` with its HARD ERRORS, `b_fail_cap`, blend / submission writer, parity mode.

## 2. What was built (`build_v224s6f.py`; `py -3.12 -I session_2026-10-15/s6f/build_v224s6f.py`)
- Base asserted: sha 73a9d562, re-serialises byte for byte, == the Kaggle pull of rsna-sub-v224flip (v224rm/pulled_v224flip), and the
  cells it relies on by content (NB_T0 in cell 4, R7's global pip in cell 6, FLIP forced in 7, r224 in 11, stage-3 `sub` in 13, V224
  canary `_canary` + V224_DONE last in 14). No base cell sets PYTHONPATH; no name collision with ours.
- **Driver** `/tmp/s6f/s6f_driver.py` (embedded base64 in cell 15, sha **a2acdd1d917fb36d...**, 49 KB; readable copy
  `s6f_driver_built_v224flips6f.py`; full text also in `celldiff_v224flips6f.txt`) = `s6f_driver_src.py` with 50 `# @@VERBATIM NAME`
  markers replaced by nartaa v2's statements byte for byte (ast statement lines). Ours, marked [s6f]: paths (OUT /kaggle/working/s6f,
  TMP /tmp/s6f/work), `CODE_MANIFEST_SHA256` pin, `env_isolation()` (fresh interpreter: sys.path, PYTHONPATH, module versions + dirs;
  fails if an R7 dir /tmp/probe.. or /kaggle/working is importable), `run_b_s6f` (= run_b with: compact_infer `--deadline` = stop_at -
  1200 s; a deadline / failed / killed part is never an error; studies without logits are unscored NaN rows; a killed part's flushed
  predictions.csv rows count; the one retry only if 10 min remain; no fail cap; microbench in the receipt), `main` (env -> inputs + manifest
  pin -> B probe -> B; exit 0 with `{"ok": true}` whenever B_probs.csv was written, else exit 3). It never names submission.csv.
- **Cell 15 SETUP** (never raises): (1) byte copy submission.csv -> `submission_v224flip.csv` + sha (the frozen base); (2) writes + sha-checks
  the driver; (3) S6F env = this kernel's `os.environ` (its site-packages), PYTHONHOME / STARTUP dropped, ipykernel's inline MPLBACKEND ->
  Agg, the image's own PYTHONPATH kept (an R7 dir there would fail the setup); (4) DEC: the reviewed V224-J logic (same PyPI shas, same
  8-entry target check, same self-test program and fixtures) into ITS OWN dir `/tmp/s6f_dec`; only on self-test success `PYTHONPATH =
  /tmp/s6f_dec + image PYTHONPATH` in the S6F env (os.environ of the notebook untouched; DEC failure = S6F runs without decoders, gate fails
  it on the commit). Writes `s6f_setup_receipt.json`; prints `S6F_DEC_OK` / `S6F_SETUP_OK`.
- **Cell 16 RUN** (never raises): skip unless setup ok and kernel age <= G; else one `python -u /tmp/s6f/s6f_driver.py --stop-at (hard -
  240 s) --nb-t0` in its own session (`start_new_session`), env = the S6F env, cwd /tmp/s6f/work; polls every 5 s; at the hard deadline
  kills every S6F process group (driver's, each worker's incl. DataLoader children: pgid of every /proc process whose cmdline contains
  `/tmp/s6f/`, never the notebook's own group), and sweeps once more after a normal exit. Prints `S6F_START` / `S6F_SKIPPED ...` /
  `S6F_KILLED` / `S6F_STAGE_DONE {status, rc, coverage, microbench}`.
- **Cell 17 BLEND** (never raises): uses S6fall only if run clean, driver ok, B_probs.csv sha == driver receipt, 12 labels by name, every
  base id once, scored rows finite in [0, 1], **coverage >= .98**, and (N scored >= 100) rank corr(base, S6fall) mean >= .60 and min >= .40
  over the non-blanked labels (nartaa's sanity floor). A column the V224 canary blanked (Fracture) stays 0.5. Else the S6F canary. Writes
  `s6f_receipt.json` (setup, run incl. the driver receipt subset, blend), prints `S6F_BLEND_OK` or `S6F_CANARY_FIRED`, then `S6F_DONE`.
- Metadata = V224-FLIP's + `nartaa/rsna-knee-s6fall-weights` + `glorysing/rsna-knee-offline-codecs-r09` (5 datasets). Build receipt
  `build_receipt_v224flips6f.json` (policy, shas, verbatim list), `build_v224flips6f.log`.

### Canary design (decided here)
A byte-identical base on skip would be invisible after the hidden run (receipts and logs of a hidden rerun are never returned): the LB
would show V224-FLIP's score under the S6F name, i.e. a false "S6F did not move the best" (R1 would drop the edge) and a possible pick
carrying an edge it never had. So every non-use writes the base with **ACL = 0.5 for all rows** (one constant column, about -.03 to -.04
macro AUC on the display, MODEL), plus the receipt / log markers for the commit. ACL is used by no other kernel's canary (Fracture =
r224 / RM, Synovitis = fleet), so canaries stay separable and additive in any later composite.

### Environment note (deviation from R4 (ii) wording; main decides)
"Stock site-packages" cannot exist for an APPENDED stage: V224's cell 6 runs `pip install --no-index --no-deps` of heliosli's wheels
globally (MEASURED, V224-FLIP log: "Uninstalling timm-1.0.29 ... pydicom-3.0.2 ... Successfully installed pydicom-3.0.1 timm-1.0.25"). So
S6fall runs on the image's site-packages **as left by R7: timm 1.0.25, pydicom 3.0.1**, everything else stock, + DEC; never an R7 code dir
(env_isolation proves it in the receipt). nartaa's parity reference ran timm 1.0.29 / pydicom 3.0.2. Mitigating facts: 1.0.25 / 3.0.1 is
heliosli's own pin for this code family (his public 0.954 S6 notebook runs compact_infer 34f433fe after the same pip install; MEASURED,
code read), the state dict loads strictly (B probe), and gate (i) (max |dp| <= 1e-3 on the 3 studies) is the arbiter. Restoring stock
versions would need an insert before cell 6 (not append-only) or a new wheel dataset.

## 3. Cell-diff proof vs V224-FLIP (`celldiff_v224flips6f_proof.py` -> `celldiff_proof_v224flips6f.txt`: S6F_CELLDIFF PASS 17/17)
Independent of the builder: base sha 73a9d562 == the Kaggle pull; alignment = `equal base[0:15] -> new[0:15]`, `insert new[15:18]` (append
only); cells 0-14 equal as whole dicts; new minus 15-17 re-serialises to the base FILE byte for byte; notebook metadata equal; base stage
order unchanged; every base sha / flip / canary needle in the same cells; driver sha == cell constant == build receipt; nartaa v2 sha ==
pulled == his commit's self sha; 57 driver statements identical to nartaa v2 (incl. prepare_inputs, probe_b_stack, LAUNCHER, B_PROBE,
collect_part, b_cmd, b_env, split_parts, run_group, extract_payload, B_MODEL, SOURCE_SHA, CODE_SHA, B_GEOMETRY, B_AMP, LAB), our text only
for KROOT / OUT / TMP / MODE comment / _RECEIPT / main, ours-only: configure, ENV2, env_isolation, collect_part_csv, run_b_s6f, pins;
nothing excluded is defined; metadata == V224-FLIP except id / title / code_file / the 2 added datasets. `tests/neg_celldiff.py` 6/6: FAIL
on a changed base cell, S6F cells inserted before the V224 canary, codec dataset dropped, driver bytes changed, driver with remove_submission.

## 4. Gate `v224s6f_check.py` (31 checks on the clean path; `--expect-canary` for control builds)
(A) frozen base: sha == receipts; the frozen output + the log up to V224_DONE pass the **unchanged** `v224_check.py --flip --ref
out_v224_v1` (V224-FLIP-form recompute 1e-6 + all its marker / receipt checks); stage csvs == the V224-FLIP v1 commit (R7 / r384 1e-3,
r224 1e-6); freeze after V224_DONE; no Traceback / RuntimeError anywhere. (E) setup ok + DEC ok + PYTHONPATH /tmp/s6f_dec first; policy ==
build and hard deadline <= NB_T0 + 8.2 h; RAN, rc 0, not killed, start age <= G; driver ok; code manifest + model sha; versions (python,
torch, cuda, timm, pydicom, numpy, pandas, cv2, scipy); no R7 / working dir on any import path; torch / timm / pydicom / numpy / pandas from
the image dist-packages, decoders from /tmp/s6f_dec (info line prints the timm / pydicom seen vs nartaa's); B probe MONAI 0.6.0 vendored,
spatial, 50,018,284 params; timings + microbench; worker receipts PASS / fp16 / E_prior_s42_fall.pt / 5 source shas incl. 34f433fe /
anchor sha / timed_out False. (B) **parity** B_probs vs nartaa k1/B/B_probs.csv, labels by name, **max |dp| <= 1e-3**; sha chain.
(C) **blend recompute** (independent average-tie ranks) <= 1e-6; w .2. (D) coverage 1.0 driver == blend. (F) no canary / failure marker,
no constant column. (G) marker order V224_DONE < S6F_DEC_OK < S6F_SETUP_OK < S6F_START < S6F_DRIVER_DONE < S6F_STAGE_DONE < S6F_BLEND_OK <
S6F_DONE. (H) format + sha.
rfix additions: `--expect-canary` now also checks the control's PATH, from the build receipt `policy.control` (or `--control
skip|kill`): skip = run SKIPPED_GUARD, no S6F_START; kill = run RAN + killed, >= 3 process groups killed incl. the stub's and its 2
child sessions, `S6F_KILL_CHECK` alive == [], kill within 3 min of the hard deadline; a receipt without a control kind FAILs. The
hard-deadline check now requires a number (it passed vacuously on None). Info lines (never a failure): `RECAL ...` +
`RECAL_TRIGGER yes|no` (sec 5) and `PICK_ELIGIBLE_J yes|no` (sec 7, R3). The 31 normal-mode checks are unchanged.
Tests (`tests/`, all offline, laptop CPU, `py -3.12 -I`; `run_offline_tests.py` re-runs all 7 and accepts only exact verdict lines):
- `test_driver_units.py` 15/15: the built driver (== the notebook bytes) runs its own main on nartaa's REAL part outputs (only GPU /
  dataset steps stubbed): split_parts == nartaa's part ids; worker argv == nartaa's argv_part0 except interpreter path and deadline; our
  B_probs.csv vs nartaa's: header, ids, order and all 12 z| columns byte-identical, P within 1.1e-16 (numpy exp build); killed part ->
  csv rows (|dp| 4.8e-7); timed-out part -> coverage 2/3, no error; wrong compact_infer sha -> those studies unscored; wrong manifest ->
  exit 3.
- `test_cells_local.py` 35/35: the 3 cells as written (paths redirected): clean == independent recompute (2.5e-8); permuted columns
  identical; start guard skip, hung driver killed at the hard deadline (~15 s), driver failure, missing label, sha mismatch, 97 %
  coverage, sanity floor, setup failure -> canary with the other 11 columns' csv text == V224-FLIP; 98 % coverage used with unscored rows
  == their base rank; Fracture canary kept; os.environ / sys.path never changed.
- `mock_s6f_output.py` 21/21: mock commit output = the real V224-FLIP v1 output + the cells executed with the driver port on nartaa's parts
  (mock-only: DEC fields set to the Kaggle DEC_OK form, env-isolation stand-in) -> **PASS 31/31**; N1-N16 each FAIL on the intended
  check (prob +2e-3, meniscus names swapped, submission +1e-5, coverage 2/3, canary fired, start-guard skip, frozen base changed, no timm
  version, R7 dir on PYTHONPATH, driver not ok, freeze before V224_DONE, manifest pin, compact_infer 8401812c, G 6.0, weight .3, DEC
  fallback); permuted B_probs PASS; control mock (age 6.0 h -> skip -> canary) PASS with `--expect-canary --control skip` (18/18)
  and FAILs the normal gate.
- `test_controls.py` 18/18 (rfix, R1): builder refuses the old cut flags; `--kill-control` builds (cells 0-14 / SETUP / BLEND == the
  submit kernel but the tag, RUN = submit RUN + only the splices); default build byte-identical to the project files; kill-control
  cells executed (stub + 2 child sessions + 2 grandchildren started, killed at the hard deadline, canary); canary gate per control:
  kill mock PASS, old-cut mock FAIL (also with --control kill), skip mock PASS / FAIL with the wrong kind, survivor / 1 group / no
  deadline number FAIL. Against the pre-change builder + gate: FAIL 3/11 (the old gate passed the old-cut mock 16/16).
- `test_gate_recal.py` 10/10 (rfix, R2 + R3): RECAL x1.00 on the mock, trigger on warm GPU x1.6 / warm decode x1.6 / B stage x2.1,
  not on cold-study GPU x3; real commit trigger no; PICK_ELIGIBLE_J no / yes with a base DEC_OK / no with DEC_FALLBACK; every case
  still PASS 31/31. Against the pre-change gate: FAIL 0/10.
- `test_commit_step0.py` 7/7 (rfix, R4): the helper rejects a FAIL verdict after " PASS " lines, a stale PASS txt, PASS 20/21, PASS
  with rc 1; step 0 pipes the helper and the proof into `verdict_ok` (bash-executed: accepts PASS 7/7 incl. CRLF, rejects the rest).
  Against the pre-change script: FAIL 1/4.

## 5. Runtime (`sim_s6f_runtime.py` -> `sim_s6f_runtime.txt`; MODEL on MEASURED inputs)
- Base = V224-FLIP always-on flip, calibrated to its MEASURED 4.57-4.65 h submit-to-score; session model as RUNTIME_FFS_1015 (s =
  exp(N(0,.08)) x 1.3 w.p. .05, queue 0-.3 h, r224 .15-.30 h). S6F starts right after the V224 canary (~4.5 h at s 1).
- S6fall cost per study per T4 c ~ U(6.5, 13) s (MODEL): low end = the warm GPU time of nartaa's commit (7.3 s for 502 windows = 14.6 ms /
  window, 1 model, crop 288); high end = the compact decode path's hidden floor (R7 plain / HV-S6 hidden 12.7-15 s per study, same decode
  code). Setup 0.03-0.08 h (nartaa commit: env 19 s + inputs 8 s + B probe 26 s + model 4 s; + DEC ~6 s).
- Grid: G 5.0 / 5.25 / 5.5 / 5.75 / 6.0 / 6.25 / 6.5, budgets 2.6 / 3.0 / 3.3 / none, deadline 8.2. **Chosen: G 5.75 h, budget 3.3 h,
  deadline NB_T0 + 8.2 h**: median 6.30, p90 7.18, **p99 7.81 h** (cap: no new S6 study after hard - 24 min), P(used) .961, P(skip) .022,
  P(cut < 98 %) .017, P(end > 8.0 because of S6F) 0, P(> 9 h) 0. G 5.5 gives P(used) .945; no guard .958 but more 7.8 h runs.
- Sensitivity (G 5.75 / 3.3): c 6.5-9 -> used .97; c 10-13 -> .94; c 13-16 -> .75 (cuts); N 1,500 -> .77; slow-session share 15 % -> .88.
  Bound: whatever S6fall costs, the kernel cannot pass ~8.22 h because of S6F (9 h margin >= .75 h).
- **The GPU commit must measure** (all in the receipts; the gate prints them): S6F setup seconds (DEC, freeze, driver write); driver
  stage seconds env / inputs / B_probe / B; per part seconds and per-study [windows, decode, render, gpu]; microbench s/study/GPU,
  gpu_s/study, decode_s/study/worker, windows/study; versions seen (timm / pydicom); total log end vs V224-FLIP's 481 s.
- **Recalibration rule (rfix S6F-R2, replaces "warm GPU > 1.5x 7.3 s"):** read the gate's `RECAL` line, which compares like for like
  with nartaa's commit (same studies, same parts; warm = not the first study of its part; loader = decode + render): rebuild only on
  `RECAL_TRIGGER yes` (warm GPU > 1.5x, or warm loader > 1.5x, or B stage > 2.0x his 64.1 s); then scale the sim's C_LO / C_HI by the
  printed factor, re-run the sim and rebuild with `--g-start-h`. The driver's microbench totals (gpu_s_per_study, s_per_study_per_gpu_max,
  decode_s_per_study_per_worker) include worker spawn, model load and CUDA warm-up of 3 cold studies: they are never the sim's c (the
  gate prints nartaa's same-field totals next to them: 10.43 / 50.5 / 26.9). **Real commit (MEASURED, gate_v224flips6f_v1_rfix.txt):
  warm GPU 7.58 vs 7.32 s (x1.04), warm loader 23.99 vs 26.82 s (x0.89), B stage 60.1 vs 64.1 s (x0.94) -> RECAL_TRIGGER no**, G 5.75
  stands. (The old rule would have compared its gpu_s_per_study 10.6 with 7.3 s = 1.45x, a near-false trigger.) The real calibration is
  still the first hidden latency: (S6F submit-to-score) - 4.6 h = the S6F hidden wall.
- Note (GUESS): 14.6 ms per window for ONE model at crop 288 is ~1.75x HV-S6's per-fold cost at 352; a likely cause is compact_infer's
  fp32 re-run when a bf16-trained model gives a non-finite fp16 pass. It is inside the c range; the commit microbench shows it again.

## 6. Open issues / concerns
1. Site-packages: S6fall sees timm 1.0.25 / pydicom 3.0.1 (R7's global pip), not stock 1.0.29 / 3.0.2 as in nartaa's reference (sec 2).
   Gate (i) decides; if parity fails, the options need a main decision (amend R4 (ii), or a stock-wheel dataset, or an insert before cell 6).
2. Runtime rests on one V224-FLIP latency sample and a wide S6fall cost range; P(visible canary) ~4 % MODEL (2 % skip, 2 % cut).
3. The POSIX kill path (os.killpg by /proc pgid) is code-reviewed and mirrors nartaa's run_group, but was only exercised here through the
   non-POSIX fallback (Windows laptop; no WSL / docker here). Optional control kernels (never submit; one GPU commit each; gate with
   `py -3.12 -I v224s6f_check.py OUT --expect-canary --build build_receipt_<tag>.json`, the control kind comes from the receipt):
   `--tag v224flips6fskip --g-start-h 0` (start guard skip -> canary; control skip) and **`--tag v224flips6fkill --kill-control
   --budget-h 0.01 --min-run-s -1000`** (control kill: the RUN cell starts a stub tree under /tmp/s6f, i.e. the stub in its own
   session, 2 children each in their own session, each with a grandchild in the child's group whose cmdline does not name /tmp/s6f,
   like a spawned DataLoader worker; at hard = start + 36 s the cell's /proc process-group kill must take >= 3 groups, and 2 s later
   S6F_KILL_CHECK must find no live stub process; canary). [rfix S6F-R1] The previously documented "cut" control
   (`--budget-h 0.03 --min-run-s 0`) never started the driver (stop_at = hard - 240 s < start -> SKIPPED_NO_TIME) and the old
   `--expect-canary` gate passed it 16/16, so it was a second skip control; the builder now refuses it.
4. v1 (the scored version) cannot be re-run (private datasets then); parity is against v2's public commit, whose code tree / model are
   sha-pinned to the scored run (CLAIMED by nartaa; the model sha and code manifest sha are MEASURED identical in his receipts).
5. Kaggle attaches the dataset's latest version at push; content is pinned by sha (model, code_manifest -> all 293 files); a re-upload by
   nartaa would make the commit fail visibly, never silently.
6. Expected gain GUESS +3 to +10e-4 public (SILVER_PLAN lever 1), P(>0) ~.6; a public S6fall blend by others may arrive before the lock.
7. Only the FLIP base was built; the FLRM-base variant (B1 "two kernels") needs a base switch in the builder (asserts are FLIP-specific).
8. [rfix S6F-R3] No J in the base stages (cells 0-14 = V224-FLIP): r384 / R7 / r224 still have no JPEG Lossless decoder and no
   U-control; only the S6fall subprocess decodes (/tmp/s6f_dec). Valid as a read under Amendment 3 R4 (which does not mention J), but
   NOT pick-eligible under SILVER_PLAN sec 4 (J required) and against sec 2 B2 ("DEC cell in every new build"). Record this entry as
   a READ. A pick-eligible S6fall entry needs the same 3 cells on a FLIP-J base, which does not exist yet (B2 "build FLIP-J tonight":
   no kernel_v224flipj on disk at 04:00). Builder work then (not done here, the base is missing): base path / sha / pulled copy as
   parameters, base cells located by content instead of fixed indexes 4 / 6 / 7 / 11 / 13 / 14 and 15 base cells, the assert "no
   base cell sets PYTHONPATH" relaxed for exactly the DEC cell's `os.environ['PYTHONPATH'] = after`, the same in the cell-diff proof;
   S6F may then reuse the base's /tmp/_dec instead of its own install (optional: /tmp/s6f_dec first on the S6F PYTHONPATH still
   passes every gate check). Budget one more build + commit + gate + slot for it (not in the current slot plan).

## 7. Review round rfix (10-15 ~04:00 IST): findings S6F-R1..R4 (each verified here before fixing)
Pre-change copies: `build_v224s6f.py.pre_rfix_1015`, `v224s6f_check.py.pre_rfix_1015`, `gpu_commit_v224flips6f.sh.pre_rfix_1015`,
`tests/mock_s6f_output.py.pre_rfix_1015`, `RESULT.md.pre_rfix_1015`. No kernel byte changed (default build re-derived byte-identical:
notebook 9c0e2558, metadata, receipt, driver, celldiff; test_controls K3). The fixes were deployed after main's commit script had
exited (it ran 03:30-03:45 with the old step 0 and the old gate; both edits would otherwise have hit a running bash script / gate).
| id | verdict | evidence (MEASURED here) | fix | test (fails before, passes after) |
|---|---|---|---|---|
| R1 kill path never exercised by the "cut" control | **REAL** | old cut cells (budget .03, min-run 0) -> `SKIPPED_NO_TIME` before Popen, killed None; old gate `--expect-canary` on that output: **PASS 16/16** | builder refuses any build whose RUN cell cannot launch (budget - margin < min-run); `--kill-control` stub-tree control (`cell_run_killctl_tpl.py`, splices only in control builds); receipt `policy.control`; gate checks the control's path, deadline must be a number | `tests/test_controls.py`: before FAIL 3/11, after PASS 18/18 |
| R2 recal rule not evaluable / apples vs oranges | **REAL** | identical pipeline (mock) reads gpu_s_per_study 10.43 = 1.43x 7.3 s; real commit 10.6 = 1.45x; the gate labelled s_per_study_per_gpu_max (50.5 / 52.1) "feed C_LO / C_HI"; loader slowdown triggered nothing | gate prints nartaa's same-field totals next to the microbench, relabelled "NOT the sim c"; new `RECAL` line (warm GPU, warm loader, B stage, same uid) + `RECAL_TRIGGER`; sec 5 rule rewritten | `tests/test_gate_recal.py` G1-G6: before FAIL, after PASS |
| R3 no J in the base stages | **REAL** (plan compliance) | cells 0-14 == V224-FLIP (no DEC cell; real log has no DEC_OK before V224_DONE); SILVER_PLAN sec 4 needs J | record as READ only (header, issue 8); gate prints `PICK_ELIGIBLE_J no`; FLIP-J + S6F builder deferred (base missing) | `tests/test_gate_recal.py` G7-G8 + G2 (real: no): before FAIL, after PASS |
| R4 step 0 trusted old txt files via `grep " PASS "` | **REAL** | `tests/mock_s6f_output.txt` has 6 non-verdict " PASS " lines; a copy with `MOCK_S6F FAIL 20/21` still matched | `run_offline_tests.py` re-runs all 7 tests and accepts only `<NAME> PASS n/n` + rc 0; step 0 pipes it and the proof into `verdict_ok` (exact, CR-stripped) | `tests/test_commit_step0.py`: before FAIL 1/4, after PASS 7/7 |
Final runs (MEASURED, laptop CPU): step 0 of the commit script executed alone (no push): kernel sha OK, S6F_CELLDIFF PASS 17/17,
`OFFLINE_TESTS PASS 7/7` (TEST_DRIVER 15/15, TEST_CELLS 35/35, NEG_CELLDIFF 6/6, MOCK_S6F 21/21, TEST_CONTROLS 18/18,
TEST_GATE_RECAL 10/10, TEST_STEP0 7/7; `offline_tests_before_push.txt`); fixed gate on the real v1 output PASS 31/31.

## Files (session_2026-10-15/s6f/)
`pull_s6fall.py`, `pull_info.json`, `pulled_v1/`, `pulled_v2/`, `pulled_cur/`, `ref_output/` (nartaa public commit output), `ds_meta/`,
`extract_sources.py` + `src_extract/` (heliosli S6 sources, read only), `s6f_driver_src.py`, `cell_setup_tpl.py`, `cell_run_tpl.py`,
`cell_blend_tpl.py`, `build_v224s6f.py`, `build_v224flips6f.log`, `build_receipt_v224flips6f.json`, `celldiff_v224flips6f.txt`,
`s6f_driver_built_v224flips6f.py`, `kernel_v224flips6f/`, `celldiff_v224flips6f_proof.py` + `celldiff_proof_v224flips6f.txt`,
`v224s6f_check.py`, `sim_s6f_runtime.py` + `sim_s6f_runtime.txt`, `gpu_commit_v224flips6f.sh`, `tests/` (s6f_harness.py,
test_driver_units.py/.txt, test_cells_local.py/.txt, mock_s6f_output.py/.txt, mock_s6f_gate_positive.txt, mock_s6f_gate_control.txt,
mock_out_v224flips6f/, mock_out_v224flips6f_canary/, neg_celldiff.py/.txt). rfix: `cell_run_killctl_tpl.py`, `run_offline_tests.py`,
`offline_tests_before_push.txt`, `gate_v224flips6f_v1_rfix.txt`, `tests/test_controls.py/.txt`, `tests/test_gate_recal.py/.txt`,
`tests/test_commit_step0.py/.txt`, `*.pre_rfix_1015`.
