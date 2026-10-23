# K2 FIX_LOG (review of 10-04 ~03:56, applied 04:00-04:45 IST)

Each review issue was verified against the code and, where it touches the trainer, against the box (read-only:
`~/fleet/fleet_train.py`, `~/fleet_runs/*`). Pre-fix copies: `k2/_review/*_pre_fix*`. Patch scripts: `k2/_fix/`.
Nothing was pushed, created, versioned or submitted; on the box only `~/k2_test/` was written.

| # | Issue (review) | Verdict | Fix |
|---|---|---|---|
| F1 | d1_pick `final` = 10 logged epochs, while the last EMA eval + save run ~4.6 min after `ep 9 raw` | **REAL.** `fleet_train.py` prints `ep 9 ema ...` BEFORE `save()` (line 248 vs 256), `save()` removes the old best before `torch.save` (190-193), and `print(json.dumps(best))` (258) is the only post-save marker. Both runs are `--epochs 10`, no `--stop_epoch`. | `box/d1_pick.py`: final = closing `{"raw": ..}` json line present AND 10 epochs; a provisional call never prints `SHIP` and exits 3 (also on the "both fail G1" branch); an unreadable best file / OOF while not done -> exit 3 `PROVISIONAL`, after done -> crash (exit 1). RUNBOOK sec 3/4 gate = exit 0 + `SHIP`. |
| F2 | RUNBOOK submit used `-v 1`; a re-push after `fleet dataset not attached` makes the good commit V >= 2 | **REAL.** | RUNBOOK sec 4: write down V at push, output dir `out_k2v<V>`, no push between check and submit; sec 5: `-v <V>` + the warning. |
| F3 | d1_pick matched the best file's epoch from the truncated cv in its name | **REAL** (box 04:07: v4 raw file `cv8402`, checkpoint cv .84025). | Epoch and cv read from the checkpoint (`ck['epoch']`, `ck['cv']`, torch CPU in kmimic); the log row of that exact epoch is used, and asserted to match the checkpoint cv to the log's 4 decimals. Tag comparison uses the exact cvs. |
| F4 | build_k2 "one tag per run" keyed on (folder, `f0_s42`): refuses two label variants in one folder, misses raw+ema in two folders | **REAL.** | Run key = checkpoint args (fold, seed, train_labels, backbone, model_mod); raw and ema of one run share the args dict. RUNBOOK gap 7: second variant goes to its own folder (same file names). Smoke D negative (same file twice) still refused. |
| F5 | Combined kernel keeps the K1 cell's sandbox `submission*.csv` (parent copies or P+C under K2's own names) | **REAL.** | `k2_crop_post_cell.py` C6: after the CF writer has read the sandbox, every `_k2_crop/submission*.csv` is renamed `k1out*.csv` (`sandbox_renamed` in the receipt); `k2_check.py` reads `_k2_crop/k1out_fleetsolo.csv` and checks no `submission*` is left in the sandbox. Not applied tree-wide: the parent itself writes `submission_checkpoint_rank_mean.csv` (top level + `g96_input/`, `d4_input/`, box smoke runs), so a whole-tree check would fail every real commit; RUNBOOK sec 0/5 names it as never-submit. |
| F6 | Fleet deadline `min(t_cell + 150 min, wall)` contradicts P6 / RUNBOOK ("runs to the wall") | **REAL** (moot at P ~6 h, but it could only throw away a fleet that would still finish). | Deadline = kernel start + 8.6 h whenever a clock exists (code now matches P6 and the RUNBOOK). New smoke check: `left_min == (8.6 h - age) x 60` on fleet_normal / stub_ok / commit3 (commit3 at age 0.2 h would show the old 150 min cap). |
| F11 | (truncated in the issue list, "If no kernel clock is found ... past 9 h ... fails the WHOLE submission") | **REAL but practically unreachable**: needs no `/proc/self/stat` AND no parent `T0` (the sha-pinned parent sets `T0 = time.time()` in cell 5 and needs it itself). | No-clock backstop shortened 150 -> 40 min (`NOCLOCK_MIN`, replaces `BUDGET_MIN`): parent at its 8.0 h cap + 40 min + 5 min grace < 9 h. New scenario `fleet_noclock` (no T0, no /proc): stage clean, `kernel_age_h` None, `left_min` 40. |
| F7 | RUNBOOK sec 6 understates the crop stage, misreads the 0.68 s/study basis, no slow / 8.0 h cases | **REAL** (numbers re-derived: 0.43 x 1,322 = 9.5 min; 0.68 s/study is from 4 processes with cv2 at 1 thread, `round3/fleet/fleet_prep_kernel.py:46,85`; P 7.7-8.05 h is still compute.md's model). | Sec 6 rewritten: setup / decode / GPU / totals, crop ~10 min (~20 slow), a P table (6.0 / 7.7 / 8.0-8.05 h) with the outcome of each case; the 10 min crop floor kept (nothing runs after it). GPU-memory figures marked as the reviewer's reading, not re-verified. |
| F8 | No per-study decode timeout: one stuck decode blocks a worker until the wall kill | **REAL** (low probability, costs a whole slot and delays the read ~2.5 h). | `fleet_worker.py` K2e: `f.result(timeout=max(5, min(FLEET_STUDY_TIMEOUT_S=300, DEADLINE + 60 - now)))`; a timeout skips that study (NaN, NV -1, `n_fail`, `stats.decode_timeout`), the 2 % coverage gate decides. Smoke B: a 30 s hang on one study with timeout 1 s -> skipped, other 23 rows identical (max abs 0.0). |
| F9 | Worker never calls `cv2.setNumThreads(1)`: 8 decode threads x cv2 pool on 4 vCPU | **REAL** (speed only). | `fleet_worker.py` K2d: `cv2.setNumThreads(1)` guarded by try. Output identity cannot be shown on the laptop (PIL shim); guards = box A0 with Kaggle's cv2 4.14 (OOF parity, decode probe) and the commit's PARITY re-encode (exit 4). |
| F10 | RUNBOOK requires `[fleet g]` lines that come from Popen children (fd 1 forwarding unproven on Kaggle) | **REAL** (documentation; could block a good commit). | RUNBOOK sec 4: `[k2-fleet]` lines required, `[fleet g]` facts may come from `fleet_receipt.json` instead; new `k2_check.py --kaggle_commit` gates on the receipt: 2 GPUs, decode probe match >= 1 / mismatch 0 per worker, shard-0 parity ran with frac_gt8 <= .01 and no mismatch, one error-free microbench per GPU, n_fail = n_blank = 0. |

Also adopted from the review's harness copies: stub modes `blank` (2/24 failed studies) and `views2` (2 views/study)
with scenarios `fleet_blank` / `fleet_views2`: the coverage and views gates fail the stage and no fail-loud file is
written.

## Verification after the fixes

- Laptop smoke 04:09-04:15: `K2 SMOKE PASS` in 5.2 min (`tests/smoke_transcript.txt`): worker parity K2 vs fleet2
  P max abs 0.0; K2e timeout test PASS; 22/22 scenarios (new: fleet_blank, fleet_views2, fleet_noclock; new checks:
  wall-only deadline on fleet_normal / stub_ok / commit3, C6 rename on every combined run with a crop dataset); real
  interim checkpoint build PASS; one-tag-per-run refusal PASS. `k2_check.py --kaggle_commit`: PASS 27/27 on the
  3-study commit run, FAIL (microbench empty) on the 24-study run.
- Box (`~/k2_test` only, CPU, nice 19, next to the running trainers):
  - `d1_pick.py` 04:07 on the live runs: exit 3 PROVISIONAL, exact checkpoint cvs (mix50 raw .87425, v4 raw .84025).
    On a symlinked copy (`~/k2_test/d1fake`, removed after): no json line -> exit 3; json lines + `--final_epochs 4` ->
    exit 0 + SHIP; json lines + 10 epochs required -> exit 3; best file missing after the json line -> exit 1.
  - A0 04:15 with the patched worker + rebuilt combined notebook: `BOXSTAGE PASS 36/36`, OOF max abs 1.34e-4,
    `submission_fleetsolo.csv` byte-identical to the 03:22 pre-fix run, `_k2_crop/` = k1out*.csv only
    (`box/box_cpu_transcript_fix.txt`).
