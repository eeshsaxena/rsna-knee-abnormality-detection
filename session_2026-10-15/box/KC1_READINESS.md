# KC1_READINESS (10-15, written ~14:55 IST after the review fixes KC-R1..R7)

State in one line: KC_READY is 4 of 8 conditions short (SIDE, STORE, C2B, C7 ok; C8, C4, M2, M4 wait on main's install / paste steps below). Every code fix is done and tested, the full store is built, and the C4 smoke is re-run on the new code. No queue edit, no shim install, no PREREG file and no GPU was touched by this pass (queue.txt md5 f4af419a unchanged).

## 1. Done (MEASURED)

### Side kernel, store, C2b, refs
- [x] **Side kernel pushed, fetched, validated, installed** (main, 12:50). ~/kc_store/side_train.csv sha256 ecc5cd26 = the validated Kaggle output.
  - kc_side_validate PASS on I1-I8: 4,407 rows; 0 slot-pick mismatches over 4,407 x 5; presence = d96 mask; U 19/4,407.
  - GEO_ASSERT: all slot series have the expected geometry.
  - Kaggle run: 685.8 s, python 3.13.15.
- [x] **C6 full KC store**: built 14:05-14:12 by this pass on cores 28-39 (12 procs), to retire KC-R7 early.
  - KC_BUILD PASS: 4,407/4,407 ok, 0 errors, 7.6 min wall, 12.93 GB.
  - U 19, weak 180, geom_ok 4,407, SagPD present 4,288.
  - test458 subset 458/458 array-equal. ~/kc_store/store_summary.json sha256 527da479.
- [x] **C2b** visual canon check: PASS by main (~/kc/contact/C2B_OK).
- [x] **C7 refs** frozen (~/kc/gate/refs.json, kc_oof_kit 62218d1c).
  - The refs reproduce exactly through the G2 code path: a stand-in leg = R7 gives leg .9158, gain +.0000, rho .8170.

### Findings verified, then fixed or refuted
Each fix has a fail-before / pass-after test. Pre-change copies are in ~/kc/bak/kcfix_1015/ (box) and kc/code/bak_kcfix_1015/ (laptop). Test dirs are under ~/kc/kcfix/.

| id | verdict | fix | fail-before -> pass-after (MEASURED) |
|---|---|---|---|
| KC-R1 HIGH | REAL: with the dread table the trainer's gold74 covered 16/74, and G2 raised KeyError | make_dread_labels.py 11e6d997 appends the 58 gold58 ids with NaN labels. It asserts they are fold -1 and keeps 74/74 gold in the table. | Trainer gold_ids line (exec'd verbatim): 16/74 -> 74/74. `kc_oof_kit --g2`: rc 1 KeyError -> rc 0. New table: 4,407 rows, md5 41261ed7, 0 NaN rows in the folds5 pool. AD smoke s8 (fold -1, full store, new table): rc 0, train 4,333 / held 74, gold58 ids now in the gold eval. |
| KC-R2 MED | REAL: no --procs (argparse error), serial decode, no model_sha check, autocast off for 'cuda:N' | kc_worker.py 4af60f6f adds `--procs` (a fork pool forked before torch/CUDA, with a bounded ordered window), `--shard K/N` (one process per T4), `--decode_only` and the model_sha assert. Per-study errors give NaN + kc_err (coverage reported). The manifest gains a KC_DATASET section (fleet_model.py 897f83b2, fleet_model_ttr.py bcdf7d15). | Old `--procs 4`: rc 2 -> new rc 0. Tampered ckpt model_sha: old accepted -> new AssertionError. Real ckpt: model_sha f141aeed matches. New (procs 2) vs old (serial) probabilities: max abs dp 0.0. Serial == procs 4 == shard 0/2 + 1/2, byte-identical over 24 studies. Decode on 4 cores (28-31), warm: serial 5.9 s/study -> procs 4 1.55 s/study. |
| KC-R3 MED | REAL (the confound is GUESS; the canon itself is MEASURED correct) | kc_g1.py a6793625: the ablation FAILs iff med_margin < 0 AND lat_margin < 0 (or a margin is non-finite); per-pair signs are info only. A6 now records the patient-space canon evidence. | Synthetic 867-study case with a correct canon and a flat lateral pair (med +.151, lat -.005): old FAIL -> new PASS. Clear FAIL (-.152 / -.165): FAIL under both. Clean PASS: PASS under both. Non-finite: FAIL. Canon check confirmed: SagFS 457/457, SagPD 444/444, CorFS 457/457, Ax 457/457, 0 wrong. |
| KC-R4 LOW | REAL: cuDNN/cuBLAS OOM wrote FAIL while the runner retried | make_fleet_train_kc.py 6bb3a17b now uses the runner's is_oom regex, giving fleet_train_kc.py f1d266cf. The box regeneration is byte-identical. | gc0 + CUDNN_STATUS_NOT_ENOUGH_MEMORY and gc0 + CUBLAS_STATUS_ALLOC_FAILED: old FAIL written -> new no verdict. torch OOM gc0: no verdict. gc1 cudnn and gc0 non-OOM: FAIL (correct). The runner regex agrees on every log. |
| KC-R5 LOW | REAL (text gap) | A6: every KC member is served with RAW weights at E*; EMA is never used. | text only |
| KC-R6 LOW | REAL: M2 accepted any 64-hex | kc_ready.py 5f8ac74f now requires sha256(~/kc/PREREG_10-15.md) == A6_SHA256.txt. That file must hold the A6 title, no '<HH:MM>', and the md5[:8] of the installed fleet_train_kc / kc_common / kc_g1 and 8d12515f. | Fake HOME: stale text (placeholder, old md5s): old M2 ok -> new FAIL. sha mismatch: old ok -> new FAIL. Correct text and sha: ok under both. |
| KC-R7 LOW | REAL (code stricter than A6) | kc_build_store.py cdc7a244: up to 1 % data-stage errors (npy/shape/mask) are allowed and listed; logic errors stay zero-tolerance. A missing subset file counts as a diff, not a crash. The full build ran now: 0 errors. | 120-study fixture with 1 corrupt npy: old FAIL -> new PASS (listed). 2 corrupt: FAIL. 1 corrupt + 1 table-flip drift: FAIL (logic). |
| KC-R8 | ACCEPTED (informational) | none | Spot-checked: queue md5 f4af419a unchanged; s8 train 4,333 / held 74; f0 split 3,466 / 867 per the review |

### Smoke and parity re-run on the new code (cores 28-39, GPU hidden)
- [x] **C4 smoke s7** (the C4 argv, fleet_train_kc.py f1d266cf, kc_g1.py a6793625): rc 0, 239 s, MAXRSS 4.3 GB.
  - Closing line {"raw": 0.5416666666666666, "ema": 0.5416666666666666}, identical to s6.
  - SMOKE_TEST_ONLY verdict written. ablation_rule is the new rule; the smoke would FAIL only on the gold74 floor.
  - New ~/kc/smoke/C4_RECEIPT.txt (md5 38263b81) names f1d266cf / 45255ae0 / a6793625. The old receipt is kept as C4_RECEIPT.txt.prev_5a22d1eb.
- [x] **AD dread smoke s8** (fold -1, full store, new dread table): rc 0, 147 s.
- [ ] **C3 serve parity, re-run on the FULL store**: RUNNING, not in KC_READY (started 14:43, `kc_parity.py --dicom ~/eda_data --store ~/kc_store/store --procs 12`, PID 1101073). It is I/O-bound on cold DICOM reads (12 workers, ~8 % CPU each), so expect it at ~15:00-15:10. Result: ~/kc/logs/parity458_kcfix.json. kc_worker.study_inputs is unchanged by KC-R2, so the expectation is the 13:07 result: side/flip/presence 458/458, crop max 2 LSB, mean <= .047 (meets the restated A6 bar; the script's own PASS uses the plan's max 1). If it is still running when the regression starts, it is safe to leave it (I/O only); main may kill PID 1101073 if wanted.
- [ ] **C9 probability parity test** (s7 smoke ckpt, full store, n 4): NOT re-run. I stopped my runner before C9 so the CPU stays free for the regression; it is deferred, and C9 must be re-run on the real f0 ckpt anyway. The decode/serve equivalence of the new kc_worker is MEASURED instead: max abs dp 0.0 vs the old worker on 2 studies, byte-identical decode rows on 24. Command for later: `taskset -c 28-39 nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/rsna/bin/python ~/kc/code/kc_prob_parity.py --ckpts <f0 raw E* ckpt> --store ~/kc_store/store --dicom ~/eda_data --n 20 --device cpu --threads 12 --out ~/kc/logs/c9_prob_parity.json`
- [x] **Docs:**
  - AMENDMENT_6_KC1_REVISED_1015.md: R1/R3/R4/R5/R6/R7 text, canon evidence, store MEASURED, serve CLI and decode cost, new code freeze. '<HH:MM>' is still open for main.
  - PREREG_TEACHER_A3_SHIM_1015.md: (c) now names the frozen KC md5s. The shim is unchanged at 8d12515f.
  - KC_QCTL_1015.txt: sections 1-5 unchanged (no queue line changes); section 6 revised.
  - MANIFEST_1015.txt regenerated (box == laptop).

## 2. Main: do next, in this order (box time; T_a f3 ends ~15:45)
1. **Review the fixes** (~5 min). Diffs:
   `cd ~/kc/code && for f in make_dread_labels.py kc_g1.py fleet_train_kc.py kc_ready.py kc_build_store.py kc_worker.py make_fleet_train_kc.py; do diff --strip-trailing-cr ~/kc/bak/kcfix_1015/$f $f; done`
   - Three originals were CRLF; the edited files are LF, hence --strip-trailing-cr.
   - The decisions still open from the build pass (they go into the A6 paste): KC on all 12 labels (drafted), and the restated C3 crop bar (max 2 LSB, mean < .05).
2. **File PREREG_TEACHER A3** as a SEPARATE file with its own .sha256 (box/PREREG_TEACHER_A3_SHIM_1015.md; fill <HH:MM>).
   - Never append it to PREREG_TEACHER.md or touch PREREG_TEACHER.sha256 (labels_gate.sh compares 9e987880).
   - ~2 min.
3. **Install + regression, finished by 15:20** (~9 min MODEL from the MEASURED (b) 294 s + (c) ~165-240 s; start by 15:08):
   `bash ~/kc/code/shim/install_c8.sh && bash ~/kc/code/shim/install_shim.sh && bash ~/kc/code/shim/kc_shim_regress.sh post; tail -n 1 ~/kc/regress/REGRESS.txt`
   - Require `REGRESS PASS trainer_md5=8d12515f237d204ad63ca93bd74b36b9`.
   - install_c8 must report the copied md5s f1d266cf (fleet_train_kc.py), 45255ae0 (kc_common.py) and a6793625 (kc_g1.py).
4. **C5 dread labels** (seconds):
   `taskset -c 28-39 nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/rsna/bin/python ~/kc/code/make_dread_labels.py --src ~/jump2/labels_in/r2_llm_soft.csv --ref ~/r7t_labels/real/labels_r7t_b.csv --folds ~/rsna_train/folds5.csv --out ~/kc/labels/labels_dread_fleet.csv`
   - Expect rows 4407, gold74_nan_rows_added 58, gold74_in_table 74, md5 41261ed7f62c46ffc6e3243ee7f22f0d.
5. **Queue edits, done by 15:25**: KC_QCTL_1015.txt sections 1-3 exactly as written (L13 HOLD, then the 6 tops in the listed order), then section 3's check.
   - Expected: 0 BAD. The dread lines are ok once C5 has run.
   - Then arm the watcher (section 4): `setsid nohup bash ~/kc/code/kc_gate_watch.sh > ~/kc/logs/gate_watch.log 2>&1 < /dev/null &`
   - ~3 min.
6. **Paste Amendment 6** (box/AMENDMENT_6_KC1_REVISED_1015.md) into PREREG_10-15.md on the laptop.
   - Fill <HH:MM>. Keep the md5s f1d266cf / 45255ae0 / a6793625 / 8d12515f in the text: KC_READY M2 checks for them.
   - Then from the laptop:
     `scp <PREREG_10-15.md> training@192.168.10.30:kc/PREREG_10-15.md`
     `ssh -o BatchMode=yes training@192.168.10.30 'sha256sum ~/kc/PREREG_10-15.md | cut -c1-64 > ~/kc/A6_SHA256.txt'`
   - ~3 min.
7. **KC_READY**: `bash ~/kc/kc_ready.sh --status` (expect all 8 ok), then `bash ~/kc/kc_ready.sh`. Expect `PASS kc_ready store=4407/4407 ...`. Seconds.
   - Do NOT re-run C6: a second run marks every npz 'exists', overwrites store_summary.json with n_ok 0, and STORE then fails.
8. **After launch:**
   - f0 starts when T_a f3 ends (~15:45), within the 120 s gate poll.
   - Once f0 is #RUN, arm kc_g1_early.sh (section 4). Its output arrives ~1.2 h in, plus ~50 min of CPU.
   - Write ~/kc/gate/s_read.verdict after the S read (~16:30-17:30).

## 3. Expected times (box)
| step | time | tag |
|---|---|---|
| A3 filing | ~2 min | GUESS |
| install_c8 + install_shim + regression (a)(b)(c) | ~9 min | MODEL from MEASURED parts |
| C5 | < 10 s | MEASURED (test build) |
| qctl edits + check + watcher | ~3 min | GUESS |
| A6 paste + scp + sha | ~3 min | GUESS |
| KC_READY | < 5 s | MEASURED (--status) |
| f0 launch | T_a f3 end ~15:45 + <= 120 s | MODEL |
| early G1 info | ~1.2 h into f0 + ~50 min CPU | MEASURED rate 7.6 s/study |

## 4. Open issues (not blocking KC_READY)
- **Serve runtime (KC-R2).** Decode is the bottleneck: 1.55 s/study at 4 procs on 4 box cores (MEASURED). For ~1,300 hidden studies that is ~0.55 h at box speed, or ~0.8-1.1 h if Kaggle vCPUs are 1.5-2x slower (MODEL). The room table assumed 0.25-0.35 h, so redo the runtime review before choosing the entry form.
  - Lever (GUESS): if the base kernel already builds the d96 volume per hidden study, the KC stage could take that volume. It would then add only header_side + slot_geometry (~0.02 s/study, MEASURED by the review) instead of a second decode.
- **C9 on the real f0 checkpoint** is still to do (bar 1e-3). KC-F2 serving (Amendment 8) is still TTR-only in kc_worker.
- **KC-R4 residual.** No watcher-side FALLBACK guard was added. A mismatch can now arise only if the log tail holds an OOM string from a non-final error. LOW.
- **Disk cleanup** after review (not needed for the run): ~/kc/kcfix (test fixtures, ~1 GB), ~/kc/smoke/s7 and s8 (~0.8 GB of ckpts), ~/kc/shimtest.
- **Rule note.** All my CPU work ran under taskset 28-39 nice 19 ionice c3 with CUDA hidden. Nothing named fleet_train.py was run or written by command line. The placeholder trainer file in the R6 fake HOME was created from Python. No PREREG file, ~/queue or r7t_labels/final was touched. The receipts copied to the laptop (kc/box_receipts/kcfix/) are RSNA-only.
