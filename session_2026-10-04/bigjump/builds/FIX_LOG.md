# FIX_LOG: review of builds B1-B5, 2026-10-04 06:10-06:50 IST

Scope: the 8 review findings handed to this pass (the 8th arrived truncated; its patch `review_tmp/tta_bench.patch` was
read in full) plus the seed finding that the critical finding refers to ("the seed and dir changes are from the seed
issue below"; its patch is `review_tmp/fleet_train.patch`, R-5). Each one was checked against the code before
anything was changed. Pre-fix copies: `BLD/code_pre_fix/`, `BLD/tests_pre_fix/`, `BLD/box_pre_fix/`,
`BLD/BUILD_pre_fix.md`. Edit scripts: `BLD/review_tmp/mk_fix_*.py`.

Hard rules kept: nothing submitted, pushed, versioned or posted. No running process touched, no GPU used. Box writes
only under `~/bigjump_build/`. The box scheduler (`~/auto_after_fleet.sh`) was NOT read or touched: the ssh allow-list
for this pass is copy / md5 / tiny CPU tests, so finding 6 is handled by self-checking commands for Eesh (below).

| # | Finding (severity) | Verdict | Change |
|---|---|---|---|
| 1 | tta_bench +-0.5 offset hits the round-half-to-even tie on 24-slice stacks (medium) | REAL (measured) | FIX-1 `HALF = 0.5 - 1e-3` for P / N / B; test (6) |
| 2 | G-T1 must not gate on the EMA line (medium) | REAL (arithmetic) | BUILD.md G-T1 = max(raw, EMA), in practice raw |
| 3 + 7 | ttr_preflight has no GPU guard, default runs the ~7 GB f1 case (medium / high) | REAL | FIX-6: default `--cases ttr`, refuse below need + headroom, cap at free - headroom |
| 4 | TTR state_dict strict-loads into a plain KneeFleetNet (low) | REAL | TTR-6 persistent buffer `ttr_size_t`; worker + tta_bench assert it; model_sha now `f141aeed136dfc01` |
| 5 | pack manifest `sources` ships absolute local paths (low) | REAL | FIX-5: original name, run folder, sha256[:16], model_mod, model_sha only |
| 6 | the armed `~/auto_after_fleet.sh` collides with the TTR launch (critical) | CONSISTENT with HANDOFF, not re-read on the box | BUILD.md 3.0 (stop it first, parent gold58 by hand) + guarded launch |
| 8 | tta_bench: no GPU guard; full run ~45-75 min; `--passes A,B` gives no A+B and no gate (high) | REAL (code + new test) | R-1..R-4 applied (+ PARTIAL flag); tests (7), (8) |
| 9 | TTR seed 42 makes key / save prefix / OOF names equal family 1's f0_s42 (from R-5) | REAL (defence in depth) | BJ-5: `--model ttr` needs seed 1000-1999; launch uses seed 1042, dir `f0_s1042_ttr` |

## Details

**1. HALF offset (tta_bench.py; B6 must copy it).** `fleet_common.sample_triplets` computes
`(linspace(0, n-1, 24) + off).round()`, and numpy rounds half to even. For n = 24 the grid is integers, so an exact
+0.5 gives centres `[0,2,2,4,4,...,22,22,23]` and -0.5 gives `[0,0,2,2,...,22,22]`: every odd slice dropped, every even
one doubled (laptop, reproduced). Training never makes this view: F1 jitter U(-.5,.5) never shifts an n=24 stack and
heavy U(-1,1) shifts by whole slices. The reviewer's box sample (n=24 in 5.5 % of views, 11.7 % of studies) was not
re-measured here.
Fix: `HALF = 0.5 - 1e-3`. A sweep of every stack length 1-32 and both signs confirmed it changes nothing except:
(a) n = 24, where P / N become offset 0 (identity), not the corrupted grid; (b) a case the review missed: on the N
pass for EVEN n, the last slot is n-1 instead of the tie's n-2. The review said "endpoints round identically". That
holds for P but not for N on even n. Both values are valid views, and HALF removes the tie, so the fix stands.
`sample_triplets` is unchanged because it is contract code. New test t_b5 (6) asserts all of this. B6 (worker pass B,
10-05) must use the same `HALF`. BUILD.md sec 4 says so.

**2. G-T1 and the EMA lag.** The trainer's EMA is `AveragedModel(..., get_ema_multi_avg_fn(A.ema))`, updated once
per optimizer step. Its first update copies the step-1 weights (pretrained backbone + random head). Train 3,466
studies at bs 2 with drop_last gives 1,733 batches; with accum 4 that is 433 steps per epoch. After epoch 3 (1,299
updates), the step-1 copy still weighs 0.9995^1298 = 0.52, and 0.27 after epoch 6. Family 1 ran at the default
`--ema 0.999` (0.999^1298 = 0.27 at epoch 3), which is why its early EMA lines looked usable and TTR's will not.
BUILD.md G-T1 now gates on max(raw, EMA), in practice the raw line. It also notes that the early `--save_epochs` EMA
snapshots lag. G-T2's max(raw, EMA) is unaffected.

**3 + 7. ttr_preflight GPU guard.** Verified: the default was `--cases f1,ttr` (a full family-1 step, ~7 GB, run
first). There was no free-memory check (it only printed) and no cap. I did not take the review_tmp patch as is: its
fixed 5 GB cap would make a TTR case that peaks between 5 and ~7 GB print "OOM" even though it would fit next to v4
(8.9 GB free). That would be a false "TTR does not fit". Applied instead (FIX-6):
- default `--cases ttr`;
- `NEED = {f1: 7.5, ttr: 5.0}` GB (f1 measured ~7.1 GB per job in nvidia-smi; ttr is an ESTIMATE);
- refuse before any model build unless free >= max(NEED) + `--headroom_gb 1.5` (ttr 6.5 GB, f1 9.0 GB);
- `set_per_process_memory_fraction((free - headroom) / total)`, so the preflight OOMs itself (caught, printed) and the
  running jobs keep >= 1.5 GB.

The CUDA context exists before `mem_get_info`, so it is already excluded from `free`. CPU plumbing smoke ran on the 4
real parity studies (below). The GPU guard lines themselves were NOT executed (no GPU in this pass). BUILD.md: the
preflight is optional, TTR-only, and runs only after the parent gold58 has exited.

**4. TTR-6 buffer.** Verified pre-fix (t_b1 asserted "state_dict keys identical"). Added
`register_buffer('ttr_size_t', int64, persistent=True)`. Before committing to it I checked the trainer's EMA:
`AveragedModel` defaults to `use_buffers=False`, so the int64 buffer is never averaged. Averaging it would have been a
real hazard: the non-float EMA path truncates `224*.9995 + 224*.0005`. t_b1 (e) now runs 5 real EMA updates and
asserts int64 224 in raw and EMA, a strict-load refusal into KneeFleetNet (unexpected `ttr_size_t`), and a TTR->TTR
strict load. The worker (W-TTR) and `tta_bench.load_member` assert `ttr_size_t == args ttr_size == model.ttr_size`.
**TTR model_sha `dd2dd29b6718600e` -> `f141aeed136dfc01`.** No TTR checkpoint existed yet, so nothing is orphaned.
Family 1 `fa35880fca963652` and family 2 `ba3e1f85e38128bf` are unchanged (t_b1, box_check).

**5. Manifest provenance.** Verified (`os.path.abspath(src)`). Now `{original, run_dir (folder basename), sha256[:16],
model_mod, model_sha}`. The worker still reads only `checkpoints`. t_b4 (3) and t_b2 (pack) assert the new keys,
sha256 == the source bytes, and no ':' or '/' in the map.

**6. Scheduler collision (critical).** I did not read the box script, because of the ssh allow-list. The finding is
consistent with HANDOFF:
- "my `~/auto_after_fleet.sh` runs gold58 first, min-free 8 GB";
- TTR to launch "after the gold58 parent run (replacing the family-2 auto-launch if ready)";
- the fleet2 staging notes ("EV2S preflight then family-2 fold-0").

The mechanism holds:
- `run_parent.py --min-free-gb 8` exits if TTR takes memory first;
- an EV2S launch next to v4 + TTR is ~16-18 GB of demand on 16 GB;
- f0_s42_v4 is the G-T2 reference and a K2 member.

BUILD.md now has a 3.0 step (Eesh, BEFORE mix50 ends). The stop command kills PID 241241 only if `ps` shows its
command line contains `auto_after_fleet`; otherwise it kills nothing and lists matching processes. It never uses
`pkill -f` and never edits the running script. Then the parent gold58 runs by hand (the scheduler's own command,
quoted from the review), and TTR launches only after the parent exits. The launch command is guarded and refuses on:
more than 1 GPU process, < 8000 MiB free, the scheduler still alive, or an existing output dir.
Deviations from the review patch:
- `cd ~/fleet3_code` -> `cd ~/bigjump_build/code` (no `fleet3_code` exists; the md5-checked code is in bigjump_build);
- the PID identity check;
- a branch for "the scheduler already moved on".

**8. tta_bench guard / cost / A,B.**
- Verified: there was no GPU guard.
- 925 studies x 5 passes x 2 members = 9,250 forwards, ~45-75 min as a third job.
- `--passes A,B` produced only 'none', because `full = set(passes) == set(PASSES)`.

Applied review_tmp/tta_bench.patch:
- R-1 `cv2.setNumThreads(1)`;
- R-2 variants from the passes that ran;
- R-3 `--min_free_gb 4` / `--max_gpu_gb 3`;
- R-4 `--max_minutes` with val/gold58 interleaved.

Added: a `partial` field and PARTIAL tag on the gate line, and the HALF and TTR-6 changes.

New t_b5 tests:
- (7) `--passes A,B` gives none / A+B / A+B lesion-max per member, raw+EMA A+B and 2 gate lines, with A+B values and
  gate dicts identical to the 5-pass run;
- (8) a cut before the first study does not crash, and the summary and gate lines say PARTIAL.

**9. TTR seeds (BJ-5).** Verified: with seed 42 the TTR run's member key, its `save()` prefix and its OOF names
(`oof_f0_s42_{raw,ema}.csv`) equal family-1 f0_s42's. Packing was already fail-loud (duplicate key refused), but
`f0_s42*` globs in other tools and a reused `--out` were not covered. Applied the R-5 assert (F2-T3 precedent: family 2
owns seeds >= 100). `--model ttr` needs `--seed` 1000-1999: fold-0 1042, all-data 1001-1005.

The seed only drives shuffle order and aug draws. It has no pairing value against f0_s42_v4 because the model and aug
differ, so the G-T2 comparison is unaffected. Tests updated to seed 1042, plus a guard test that `--model ttr --seed 42`
is refused. t_b2's pack now re-keys only the TTR fold-0 member (`f0_s1042ttr`); the all-data TTR member keeps its own
unique key `f-1_s1042`.

## Re-run results (all PASS)

Laptop (`BLD/tests/run_all.sh`, CPU, 64 px synthetic, 06:24-06:36): t_b1_model, t_b2_default_parity,
t_b2_heavy_equiv, t_b2_ttr_train, t_b4_pack_tag, t_b5_tta_bench and t_gt2_saveep all exit with rc 0.
- Default path still bit-identical to `~/fleet` and `~/fleet2_code`.
- New checks pass: TTR-6 (e), the BJ-5 guard, FIX-5 keys, and t_b5 (6)/(7)/(8).

Transcript `BLD/tests/transcript.txt`; the pre-fix one is `BLD/tests_pre_fix/transcript.txt`. The ttr_preflight CPU
smoke (real parity npz, convnext_atto) is OK: the default case is ttr only and an unknown case is refused.

Box CPU (`bash ~/bigjump_build/box/run_box_tests.sh 3`, 06:36-06:46): BOXCHECK, BOXPARITY and BOXGP0 PASS. Transcript
`BLD/tests/out/box_run2.txt`.
- All 16 staged files have md5 == laptop.
- TTR sha `f141aeed136dfc01` in the box env.
- G-P0 on real studies with pretrained ConvNeXt-T and seed 1042: trainer vs worker path max |diff| 0.0 at chunk 32
  and 48.
- Real tta_bench smoke: pass A vs GPU OOF 2.1e-4.

## Not changed / left for Eesh
- PIDs, the scheduler file and the parent command were not verified on the box. BUILD.md 3.0 (a) prints them, and
  3.0 (b) refuses to kill a PID whose command line is not `auto_after_fleet`.
- GPU guard lines (tta_bench R-3, ttr_preflight FIX-6) were not executed; the CPU paths were.
- B6 (worker pass B) must use HALF.
