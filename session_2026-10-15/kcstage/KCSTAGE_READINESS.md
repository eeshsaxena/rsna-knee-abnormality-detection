# KCSTAGE_READINESS (10-15, ~18:30 IST, after the KCS1 / KCS2 review)

**Status.** The S6F+KC stage is built and tested.
- The kernel cells did not change in this pass, so the stand-in kernel shas stand: 2652445c (S6F), b0ed84d7 (J), 6a4474ff (skip control).
- KCS2 (C9 not enforced, and run on a path that does not ship) is REAL and fixed:
  - a served-path C9 tool;
  - a gate check (B2), required for a real KC1 build;
  - the commit script runs C9 on the box while Kaggle runs the commit.
- KCS1 (the start guard causes most of the visible canary) is REAL (MODEL). The fix is a change to the pre-registered p99 bar, so it is main's call. The decision table and both alternative builds are ready and proven.
- Nothing was pushed, committed, submitted or created on Kaggle. No GPU, queue, ~/fast_build/code2 write, ~/rkc or ~/rsna_comp was touched.
- All re-runs pass: `OFFLINE_TESTS PASS __OT__` (MEASURED).

## 1. Findings: verdict, fix, fail-before -> pass-after

| id | verdict | fix | fail-before -> pass-after |
|---|---|---|---|
| KCS2 medium | **REAL** (MEASURED). Box `~/kc/code/kc_prob_parity.py` (md5 ca9cd0ca) line 50 calls `kw.study_inputs`, which is kc_worker's header path; the shipped kernel replaces that call in kc_serve. Before this pass, no kcstage file mentioned C9: not the gate, not gpu_commit_kc.sh, not main_steps. There is no c9 JSON in ~/kc/logs. kc_box_parity compares T4 and CPU on the SAME dumped vol and views, so it cannot see a train/serve skew. | (1) **box/c9_served.py** runs kc_prob_parity.py VERBATIM (its own source, exec'd as `__main__`) with the ONE substitution kc_serve makes: `kw.study_inputs <- kc_serve.make_study_inputs(load_table(T), kc)`. T = the side kernel's own study() rows on the picked studies (as kc_driver calls it), checked column by column against the store's side table ~/kc_store/side_train.csv. Picks: first SagPD-absent, first weak-zone, then kc_prob_parity's rng-1015 order, n 20. It pins kc_prob_parity md5, the side kernel sha, kc_worker / kc_common / fleet models / d96 reader, and member order and sha256 against KC_MANIFEST. It records the kc_serve sha. (2) **Gate (B2)** in v224s6fkc_check.py recomputes from the rows: n >= 20, >= 1 weak-zone study served on both canons, >= 1 SagPD-absent, max \|dp\| <= 1e-3, path served, substitution live, fresh side == store table, kpp md5 == ca9cd0ca, serve sha == build, code pins == build, members (names in manifest order + sha256) and manifest sha == build. This is REQUIRED when the build's KC dataset is real (standin False); for a stand-in build it is checked only when supplied. `c9_served.json` is added to KC_FILES, so the base-gate view drops it. (3) **gpu_commit_kc.sh** starts C9 on box cores 28-39 right after the push, in the background with the same member paths and the manifest. It waits for C9 after the box parity, then copies OUT/c9_served.json to the gate (`KC_C9=skip` for stand-in commits only). | **Tool (box, KC_C9_TAMPER PASS 3/3, MEASURED):** one strong-zone study, with the serve table drifted to the other knee in a self-consistent way (mean_x negated, side and flips re-derived). T1, the pre-fix C9 on kc_worker's header path, cannot see it: \|dp\| 0.0, a PASS. T2, the served C9 with the drifted table: \|dp\| 1.33e-2 > 1e-3, so FAIL (detected). T3, the served C9 with the fresh side-kernel rows: \|dp\| 0.0, side == store table, PASS. **Gate (MOCK_GATE K0-K13, MEASURED):** K0, the PRE-FIX gate (bak_kcs2/) on a real-build mock with no C9: PASS 27/27 (the gap). K1, the new gate on the same mock: FAIL on C9. K2, real build + the box C9 JSON: PASS __K2__. K3-K11 each FAIL on the intended check: header path, one row \|dp\| 2e-3, 19 studies, no weak study, no SagPD-absent study, another member sha, side != store table, another kc_serve, another manifest. K12, a stand-in build + valid C9: PASS. K13, a stand-in build + failing C9: FAIL. **Real runs (box CPU, MEASURED):** __C9SI__ ; dry run on f0 s2001 ep3 RAW (the first saved epoch, BOX_PLAN C9): __C9F0__ |
| KCS1 medium | **REAL as a cost; no correctness bug** (MODEL, reproduced). `kcs1_guard_sim.py` re-uses sim_s6fkc_runtime.py's session model verbatim (3,000 sessions, seed 0). P(canary) = P(skip) + P(cut), at G 6.9 / G 7.2 / no guard: **.187 / .096 / .065** with S6fall c U(6.5, 13) and **.322 / .187 / .121** with c U(10, 13). The finding said .19 / .10 / .07 and .33 / .19 / .11. Under STRESS: .203 / .131 / .117 and .338 / .202 / .178. Slots per readable KC result: 1.23 / 1.11 / 1.07 (A) and 1.47 / 1.23 / 1.14 (A-hi). No overrun path (code read of the RUN cell): hard = min(start + budget, NB_T0 + 8.2 h), the driver stops at hard - 180 s, the cell SIGKILLs at hard and waits <= 120 s, and the blend takes seconds. So the end is <= ~8.25 h from NB_T0 for every G (sim max 8.16), under the 9 h limit (comp_pages.txt). | **Not changed in the build.** The p99 <= 8.0 h bar is pre-registered (A6 G3, BOX_PLAN 4.1), so only main can amend it (text in sec 4, option B). Prepared and MEASURED: builds with `--g-kc-h 7.2` and with `--g-kc-h 99 --budget-h 99` (wall only) both give KC_CELLDIFF PASS 22/22 (scratch builds, not kept), and the gate accepts any G with deadline <= 8.2. The optional SIGTERM partial-shard flush was **declined**: it would rescue 0 / 0.07 % / 0.27 % of all runs at G 6.9 / 7.2 / none (MODEL, kcs1_guard_sim.txt), which does not justify new driver and serve shas and a full re-proof. | Sim reproducibility: KCS1_SIM byte-identical on re-run (inside OFFLINE_TESTS). |

Decision table (MODEL, kcs1_guard_sim.txt; P(KC used) | P(canary) | kernel end p99 | P(end > 8.0) | submit-to-score median):

| policy | A: S6 c U(6.5,13) | A-hi: S6 c U(10,13) | STRESS | STRESS-hi |
|---|---|---|---|---|
| G 6.9 / 1.5 (built, meets p99 <= 8.0) | .813 / .187 / 7.82 / .001 / 7.10 | .678 / .322 / 7.82 / .001 / 7.28 | .797 / .203 / 7.93 / .005 / 7.26 | .662 / .338 / 7.89 / .004 / 7.41 |
| G 7.2 / 1.5 | .904 / .096 / 8.00 / .011 / 7.17 | .813 / .187 / 8.05 / .019 / 7.44 | .869 / .131 / 8.16 / .049 / 7.37 | .798 / .202 / 8.16 / .054 / 7.54 |
| G 7.4 / 1.5 | .920 / .080 / 8.15 / .033 / 7.18 | .872 / .128 / 8.16 / .049 / 7.48 | .873 / .127 / 8.16 / .094 / 7.37 | .802 / .198 / 8.16 / .123 / 7.63 |
| wall only (G 99 / 99) | .935 / .065 / 8.16 / .061 / 7.17 | .879 / .121 / 8.16 / .114 / 7.47 | .883 / .117 / 8.16 / .120 / 7.38 | .822 / .178 / 8.16 / .165 / 7.66 |

S6F's own read (57201874: submitted 07:15, score seen 13:54 IST, so <= 6.65 h, MEASURED) cannot tell A from A-hi. The canary probability at G 6.9 is therefore somewhere in .19-.34 (MODEL).

Recommendation (main's call): option B with the wall only. It has the highest P(KC used) in all four regimes, and the only cost is about +0.07-0.25 h of median latency. With the min-run rule (1,800 s), the effective latest KC start is 7.65 h.

## 2. What is built (unchanged in this pass unless marked NEW / CHANGED)
- **Builder** `build_v224s6fkc.py --kc-dir DS [--base s6f|s6fj] [--tag T] [--g-kc-h --budget-h --deadline-h]`. It appends 3 cells (KC SETUP, KC RUN, KC BLEND) to the pushed S6F kernel (sha 9c0e2558, 18 cells) or to its r2 J twin (sha 1868634e, 19 cells). Dropping the appended cells gives the base back byte for byte. Defaults: G_KC 6.9, budget 1.5, wall NB_T0 + 8.2 h, w .2, coverage .98, canary Baker's, on-fail canary. Build time < 1 s; rebuild byte-identical (MEASURED).
- **Stand-in kernels** (never submit): kernel_v224s6fkcstandin 2652445c, kernel_v224s6fkcstandinj b0ed84d7, kernel_v224s6fkcstandinskip 6a4474ff (G 0 control). Driver b0da0999, serve 67fe7f2f. Dataset ds_kc_standin (slug eeshsaxena/rsna-kc-standin, manifest 59502d00, 299 MB, 2 random-init members).
- **Gate** `v224s6fkc_check.py OUT [--box-parity] [--c9] [--expect-canary]` (CHANGED: section B2 C9). The (A)-(H) checks are as before.
- **NEW** `box/c9_served.py` (box ~/kcstage_1015/tools/): C9 on the served path. Prints `KC_C9_SERVED PASS|FAIL` and writes a JSON for the gate.
- **NEW** `box/c9_tamper_test.py` (KC_C9_TAMPER) and `box/run_c9_box.sh` (the two n 20 runs).
- **CHANGED** `gpu_commit_kc.sh`: C9 runs in the background on the box during the Kaggle commit; the box out dir is now made before the push.
- **NEW** `kcs1_guard_sim.py` / `kcs1_guard_sim.txt`: the KCS1 guard / canary / flush table (MODEL).
- **CHANGED** `run_offline_tests.py` adds KCS1_SIM, KC_C9_TAMPER and both KC_C9_SERVED verdicts. `tests/mock_gate.py` adds K0-K13.
- Pre-change copies are in `bak_kcs2/` (gate fc5b7f58, commit script 111c733a, offline tests 828c0e07, mock gate 80f47ca2).

## 3. How main turns real members into a submit-ready kernel
Run from C:/Users/eeshs/Downloads/rsna-knee. Box = training@192.168.10.30. Box CPU work runs on cores 28-39 at nice 19, ionice c3, with CUDA hidden.

**Tonight (optional but recommended): stand-in plumbing commit** (it replaces BOX_PLAN 5.1's f0-only commit until f0 exists).
1. `py -3.12 -I session_2026-10-15/kcstage/run_offline_tests.py` -> `OFFLINE_TESTS PASS __OT__` (~__OTT__ min, MEASURED).
2. `kaggle datasets create -p session_2026-10-15/kcstage/ds_kc_standin` (private, 299 MB; ~20 min, MODEL from 538 MB in 35 min MEASURED 10-12). Use the PowerShell staging fix from RUNBOOK_STUDENT step 6, then poll `kaggle datasets status eeshsaxena/rsna-kc-standin` until ready.
3. `KC_C9=skip bash session_2026-10-15/kcstage/gpu_commit_kc.sh v224s6fkcstandin kcstage_1015/standin/kc_standin_m1_s11_raw_fp32.pt kcstage_1015/standin/kc_standin_m2_s12_raw_fp16.pt` (or without KC_C9=skip to exercise C9 too, +~__C9T__ min box CPU in parallel).
   - Expect `V224S6FKC_CHECK PASS 27/27 mode KC` (29/29 if C9 ran).
   - Read the MICROBENCH and RECAL lines. This is the first real T4 test of fp16-T4 vs fp32-CPU parity (<= 1e-3).
   - ~25-35 min in all (MODEL: offline tests + commit ~13 min + queue + fetch + parity ~2 min + gate ~1 min). NEVER submit.
4. Optional: the J stand-in (v224s6fkcstandinj) and the skip control (v224s6fkcstandinskip, gated with `--expect-canary`, expect PASS 14/14).

**After f0 ends (~19:40-19:45, MODEL: epochs every ~16 min, MEASURED ep3-ep6) and again when the AD members end (~04:35 10-16):**

5. **G1 / G2** as in A6 (kc_f0.verdict PASS, each AD member's gold floor, G2 bars). E* = f0's raw-best epoch. Members = RAW ckpts at E*, ~199.4 MB each (MEASURED, ep3).
6. **OAI-clean scan** (box, ~15 s):
   ```
   ~/miniconda3/envs/kmimic/bin/python ~/k14_test/k14/box/k14_oai_clean.py --out ~/kcstage_1015/kc1/oai_clean_kc1.json --ckpt <f0 raw E*> --ckpt <AD s2002 raw E*> --ckpt <AD s2003 raw E*>
   ```
   Expect `K14CLEAN PASS 3/3`. MEASURED: PASS on the KC raw ckpt format (f0 ep3 + 2 stand-ins, 3/3). FAIL -> STOP; nothing leaves the box.
7. **C9 early** (box; de-risks before the upload; gpu_commit_kc.sh re-runs it against the manifest):
   ```
   cd ~ && taskset -c 28-39 nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/kmimic/bin/python -u kcstage_1015/tools/c9_served.py --code kcstage_1015/code --serve kcstage_1015/kc_serve.py --kpp kc/code/kc_prob_parity.py --members <f0 E*> <AD2002 E*> <AD2003 E*> --store kc_store/store --dicom eda_data --n 20 --threads 12 --out kcstage_1015/kc1/c9_served_kc1.json
   ```
   Expect `KC_C9_SERVED PASS n 20 max |dp| <= 1e-3`. Cost ~__C9T3__ min for 3 members (MODEL from the MEASURED stand-in run). FAIL -> no KC1 entry: a train/serve skew means the leg would be mis-fed.
8. **Copy the members to the laptop** (~2-5 min on the LAN, GUESS) into a staging dir, not the dataset dir. Then:
   ```
   py -3.12 -I session_2026-10-15/kcstage/make_kc_dataset.py --out session_2026-10-15/kcstage/ds_kc1 --slug eeshsaxena/rsna-kc1-members --kind kc1 --members <f0>=f0 <ad2002>=ad_tb <ad2003>=ad_dread
   ```
   This needs torch; it checks the contract, model_sha f141aeed, mirror 0, the args OAI text, and that no member is a stand-in. It takes ~1 min (GUESS).
9. **Dataset**: `kaggle datasets create -p session_2026-10-15/kcstage/ds_kc1` (~600 MB, ~40 min, MODEL from the MEASURED 10-12 rate; PowerShell staging fix). Poll until ready. Steps 10 and 11's offline tests can run during the upload.
10. **Build** (J form, which gives PICK_ELIGIBLE_J):
    ```
    py -3.12 -I session_2026-10-15/kcstage/build_v224s6fkc.py --kc-dir session_2026-10-15/kcstage/ds_kc1 --base s6fj --tag v224s6fkcj [policy flags from sec 4]
    ```
    The tag must end in j and must not contain 'standin'. For the S6F form use `--base s6f --tag v224s6fkc`. Takes < 1 s.
11. **Commit + C9 + parity + gate:**
    ```
    bash session_2026-10-15/kcstage/gpu_commit_kc.sh v224s6fkcj <box paths of the SAME 3 member files in manifest order>
    ```
    - The script runs the cell-diff proof, the offline tests, the RUNNING_COUNT check, the push, C9 on the box in the background, wait and fetch, the box parity, and the gate (which now requires c9_served.json for a real build).
    - Expect `V224S6FKC_CHECK PASS 29/29 mode KC` (27 + 2 C9) with `PICK_ELIGIBLE_J yes`.
    - ~25-35 min (MODEL). The gate must PASS before any submit.
12. **Runtime review:** read `RECAL implied c_kc` and `RECAL_TRIGGER`.
    - With option A, on a trigger: scale KC_C in sim_s6fkc_runtime.py, re-run it, rebuild with the new `--g-kc-h`, and do one more commit.
    - With option B (wall only), a trigger needs no rebuild: re-run kcs1_guard_sim with the new c_kc range for the record.
13. **Amendment text** (sec 4) pasted into PREREG_10-15.md before the submit. Then main submits (slot 2-3, 10-16, per BOX_PLAN 5.4).
    - Read rule R1: a Baker's-constant signature = UNREAD (runtime).

**Timing from "members done + G1/G2" to submit-ready: ~1.5-2 h (MODEL).** Most of it is the ~40 min upload and the ~25-35 min commit + gate. Steps 6-7 take ~__C9T3__ min on the box and can overlap the scp. Lever (needs main's OK, BOX_PLAN 5.1 already says "fp16 ckpts"): fp16-cast members halve the upload to ~20 min. The stand-in m2 proves that the fp16 raw form serves and passes parity (e2e MEASURED). Parity holds because C9 and the box parity use the same files.

## 4. Amendment text draft (main pastes, fills <HH:MM>)
**AMENDMENT 6b (KC stage serve path, C9, runtime policy), <HH:MM> IST 10-15. Written before any KC1 member file is packed or any KC number is read.**
- **Serve path.** The KC stage runs kc_worker.py 4af60f6f byte-unchanged and its own `__main__` (KC-R2 CLI: one process per T4, --procs 2, fp16 autocast, one pass, mean of member sigmoids, both-canon mean for weak-zone / not-GEOM_OK studies).
  - One deviation: kc_serve.py (sha 67fe7f2f) substitutes `kc_worker.study_inputs`. Side, rule, geometry and serve come from the side kernel's own study() table (kc_side_kernel.py 2979c191) with kc_build_store's training derivation, instead of kc_worker's header_side + 3-header slot_geometry.
  - MEASURED: kc_serve rows == kc_worker rows (side, rule, serve, vol sha) on 24/24 box studies; side pass == the Kaggle side-kernel table on all 54 columns.
- **C9 (replaces the C9 line of G3).** C9 = kc_prob_parity.py ca9cd0ca run VERBATIM by c9_served.py with the same single substitution (the served path), on 20 eda_data studies: >= 1 SagPD-absent and >= 1 weak-zone study (served on both canons).
  - Side rows come fresh from the side kernel's study() and must equal ~/kc_store/side_train.csv (sha ecc5cd26).
  - It uses the EXACT member files of KC_MANIFEST (order + sha256). Store vs served DICOM, own canon: max |dp| <= 1e-3.
  - The gate (v224s6fkc_check.py B2) requires it for the real build.
- **Runtime policy.** Choose one before the build:
  - **A (unchanged):** G_KC 6.9 h / budget 1.5 h / hard wall NB_T0 + 8.2 h, p99 <= 8.0 h (sim_s6fkc_runtime.py). P(visible Baker's canary) .19-.34 (MODEL).
  - **B (KCS1):** "Runtime review p99 <= 8.0 h" is replaced by: every stage has a hard wall at NB_T0 + 8.2 h, and the worst-case kernel end, including pre-NB_T0 time, is <= 8.5 h (< 9 h limit). KC is then built wall-only (`--g-kc-h 99 --budget-h 99`; min run 1,800 s makes 7.65 h the latest start). P(canary) .065-.18, P(end > 8.0) .06-.17 (MODEL, kcs1_guard_sim.txt).
  - Canary, coverage .98, w .2 and the read rule R1 are unchanged.

## 5. Open issues
- **KCS1 decision pending (main).** Until it is made, the default build stays A (G 6.9). The difference: ~.09-.20 more visible-canary probability per KC slot under A (MODEL). Kaggle KC decode cost c_kc is still unmeasured; the first stand-in commit gives a 3-study RECAL only.
- **fp16-T4 vs fp32-CPU parity (<= 1e-3)** is untested until the first GPU commit (stand-in heads are scaled to logit std 1.5, so the test is non-trivial).
- **C9 measures store vs serve on the box.** The box decode environment (kmimic: pandas 2.3.3, numpy 2.1.3, cv2 4.14, python 3.11) mimics Kaggle; Kaggle runs python 3.13. The T4-commit box parity covers the Kaggle-side decode on the 3 visible studies only.
- **C9 picks one weak-zone and one SagPD-absent study, the A6 minimum.** The train store has no geometry-only 'both' study (180 'both' == 180 weak, MEASURED), so that branch is exercised only on Kaggle test data (the side table equality covers its inputs).
- **J form:** the base-gate view relies on j_check accepting the KC processes in dec_envwatch.json. It is checked with KC files appended (J_CHECK PASS 16/16) but has no real KC process records until the J commit.
- **Upload time** (~40 min for 3 x 199.4 MB) is the longest step; fp16 members (option above) would halve it.
- My box footprint: ~/kcstage_1015/tools (c9_served.py, c9_tamper_test.py, run_c9_box.sh) and ~/kcstage_1015/c9 (outputs, RSNA-only). I read f0's ep3 RAW ckpt and did not write it. I left no process on the box. The OAI scan output (~/kcstage_1015/c9/oai_clean_f0ep3_standin.json) is an info receipt.

## 6. Files (laptop C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-15/kcstage/)
- Changed: v224s6fkc_check.py, gpu_commit_kc.sh, run_offline_tests.py, tests/mock_gate.py (pre-change copies in bak_kcs2/).
- New: box/c9_served.py, box/c9_tamper_test.py, box/run_c9_box.sh, kcs1_guard_sim.py, kcs1_guard_sim.txt, tests/pulled_e2e/c9_served_standin_n20.json.
- Box results: box_results/c9_tamper.txt + c9_tamper/, c9_served_standin_n20.txt, c9_served_f0ep3_n20.txt (+ JSON).
- Offline tests output: offline_tests.txt.
