# S6SEED READINESS (10-15 ~17:55 IST): second S6fall-family model, seed 43, on the box V100

Verdict: READY for main's S5 now; READY to launch once main has (1) PX_BUILD.verdict PASS, (2) PX_PARITY.verdict, (3) Amendment 7
pasted (final text in section 4) and gates/A7.verdict written. All 6 review findings were REAL; all are fixed in the kit with
fail-before / pass-after tests (MEASURED, box CPU, cores 28-39, GPU never touched). S6R-1 needs a decision from main (soup option,
section 4 point 5). No em dashes. Tags: MEASURED / MODEL / GUESS.

## 1. Done (MEASURED)

### 1.1 Findings: verification and fixes
| id | verdict | evidence (MEASURED) | fix + test (before -> after) |
|---|---|---|---|
| S6R-1 2-model S6F cost | REAL | compact_infer 34f433fe: `shared_trunk` only for the OFL backbone (l.332); for convnext_small `for m in models: run_model(...)`, a full forward per checkpoint in sequence; only the decode is shared | A7 runtime rule rewritten (sec 4). sim/sim_s6f_2model.py (reuses sim_s6f_runtime.py unchanged, 3,000 reps/row, MODEL): today 1 model used .947 / cut .023; A7 as written (c x 1.25) 2-model .863 / cut .106; GPU floor c2 U(14.6, 16): G2 5.75 2-model .629 / cut .342, G2 5.0 .629 / .247, **G2 4.5 2-model .492, nartaa-only .449, cut .028**, G2 4.25 .251 / .703 / .017; serial c1 + 7.3: G2 4.5 .220 / cut .278. Soup option made concrete: S6_HEAD_SEED hook (P7), s6_soup.py, s6_head_init_check.py, s6_member_gate.py --soup |
| S6R-2 partial cache | REAL, plus one more bug | main_s5 v1 never checked the build rc and always finalized; px_build v2 returned rc 0 when a worker died (test_px_build_exit: "all workers exited: [1]", rc 0); launcher took S6_EXPECT_TRAIN from the same cache | px_build_s6.py v3 (sha 4ee76cf3): exit 4 on a dead worker, BUILD_END line; one real study rebuilt byte-identical (px + meta cmp) to the v1 smoke1 cache. main_s5 v2 (a40f48ad): rc checked, no finalize on failure, one retry pass for per-study errors, completeness + check-targets, gates/PX_BUILD.verdict + PX_ERRORS.txt. Launcher: A7.verdict pins `pool N`; check-targets pool == N, gold == 58, N >= 4,328 (4,349 x (1 - .005)); S6_EXPECT_TRAIN = N from A7. Tests: test_main_s5 v1 0/7 -> v2 7/7; test_px_build_exit v2 FAIL -> v3 PASS; launcher C13-C15 |
| S6R-3 PAUSE race | REAL | v1 had no LAUNCH-line guard, removed PAUSE by existence; --pause-after fires at the END of the job ahead | Trigger = the job ahead's `LAUNCH ... -> .../fleet_runs/<name> (` line in queue_runner.log, still running (no DONE/FAILED after it, a compute app present), >= 3,000 s after its launch (clears the runner's early-OOM fallback window); n0 = log line count BEFORE PAUSE; any `] LAUNCH ` after n0 -> remove our PAUSE, FAIL; owner-tagged PAUSE removed only if content + mtime unchanged; a job that already ended -> FAIL, no PAUSE. Tests C1, C8, C10, C11, C12 |
| S6R-4 no time limits | REAL | no `timeout` on the smoke, no DataLoader timeout in 14767c7a (nartaa: loader_timeout 600), unbounded waits | ct_s6v100 v2 P6: DataLoader timeout 600 s (N4). Launcher: smoke under `timeout -k 60 2700`; free-GPU wait cap 10 h; trigger wait cap 36 h; watcher after launch: no log output for 30 min -> gates/STALL_s6s43 + IDLE_ALARM line (never kills); crash -> re-PAUSE within ~10 s, free checks, resume with --resume (max 1, never after ABORT / SIGTERM / STOP file); RUN verdict + alarm on failure. Tests C16-C20, C22 (real-mode command lines with a stub python) |
| S6R-5 scale decay | REAL | v1 test: an inf-loss step halved the scale 65,536 -> 32,768 and counted as skipped (2 skips for 1 inf loss + 1 inf grad) | ct_s6v100 v2 P5: finite check before backward; a non-finite loss skips backward / step and leaves scale + growth tracker alone, sched.step() kept; counted once; ABORT if scale < 256 (S6_MIN_SCALE), WARN < 1,024; loss_scale_min in S6_SMOKE / epoch line / ckpt / DONE.json. Smoke gate: final scale >= 1,024, min >= 256. test_s6v100_v2 v1 4/10 -> v2 10/10; launcher C21 |
| S6R-6 not seed-only | REAL | build_labels_raptor.py header (sol56 replaces nartaa's Gemini read); nartaa run_config pool 4,263 / feature 4,321; verify_labels.md (JEV weights fit on 67 gold studies) | recipe_s6s43.json v2 (479d57f5): `differs_from_scored_S6fall` (seed, labels, pool, steps, precision, rendering, trainer), gold58 caveat, v2 env + smoke gate; A7 text amended (sec 4). test_recipe_consistency v1 1/7 -> v2 7/7 (also proves launcher argv/env == recipe) |

### 1.2 Regression and other tests (all box CPU, MEASURED)
- Original suite (T1-T6) on ct_s6v100 v2: **S6V100_TESTS PASS 20/20** (reader 2,860 window sets byte-equal; fp32 loss/grads bitwise equal; one step bitwise equal scaler off/on; all-data run, strict load into the original Net, gold logits reproduced 4.9e-6, resume == uninterrupted; 20 original lines replaced).
- test_s6v100_v2 (N1-N6) v2 10/10; head-seed hook: init == seed-42 init on all 362 tensors, first batch bytes == seed-43 run.
- Launcher suite (final 2424e8c5): **LAUNCH_TESTS PASS 28/28** (v1: 9/25). Includes A7 head_seed cross-check (C23/C24).
- test_soup 6/6 (strict load into the original Net); s6_head_init_check: T4 model MATCH seed 43 (median cos .9967) and NO_MATCH seed 42 (.0077); s6_member_gate v2 selftest PASS.
- px_build v3 one-study real build: 29.9 s/study at 2 threads (cores 28-39, 1 study), bytes equal to v1.

### 1.3 Installed (box ~/s6seed, atomic mv; pre-change copies kept; laptop mirror session_2026-10-15/s6seed/, s6seed_px/, box_results/v2/)
ct_s6v100.py e5345b38 (RECEIPT_ct_s6v100.sha256 updated; v1 = ct_s6v100.v1_7f10854c.py), launch_when_free.sh 2424e8c5 (v1 a5db4932),
recipe_s6s43.json 479d57f5 (v1 e655f3aa), main_s5_unzip_px.sh a40f48ad (v1 28e30014), px_build_s6.py 4ee76cf3 (v2 8d94eff2),
s6_member_gate.py f06faae6 (v1 3deb1ef0), new s6_soup.py 143b7fae, s6_head_init_check.py cf068e0e, sim/sim_s6f_2model.py 7fb652b8,
tests/test_*.py|sh. Receipt: ~/s6seed/RECEIPT_v2_review.txt. Diffs: tests/out_s6v100_v2/ct_s6v100_v2.diff (vs 14767c7a),
ct_s6v100_v1_to_v2.diff.

### 1.4 State now (MEASURED)
- Competition zip complete: ~/rsna_comp/rsna-knee-abnormality-detection.zip, 265,018,939,620 bytes, mtime 16:38; no downloader running. (Only names and stat read; nothing else in ~/rsna_comp.)
- GPU: KC1 f0 s2001 running since 16:05:17 (1 compute app). No S5 process yet; ~/s6seed/px320 absent; ~/s6seed/gates absent.

## 2. Main actions, in order
1. **S5 now** (cores 12-27; single archive, so PARTS = the zip):
   `PARTS="$HOME/rsna_comp/rsna-knee-abnormality-detection.zip" DATA=$HOME/rsna_comp/data setsid nohup bash ~/s6seed/main_s5_unzip_px.sh > ~/s6seed/logs/main_s5.log 2>&1 < /dev/null &`
   (PROCS=8 THREADS=2 = the exact parity config, about +2 h MODEL.) Ends with `~/s6seed/gates/PX_BUILD.verdict` = `PASS ... pool N gold 58 errors E missing 0`. On FAIL: read the verdict (rc 3 MEM_CAP, rc 4 dead worker, floor / gold / missing), fix, re-run the same command (resumes). Watch the KC epoch time; if it rises > 10 %, stop and re-run with fewer PROCS.
2. **nartaa's E_prior_s42_fall.pt** (sha 0c7297d5, public dataset nartaa/rsna-knee-s6fall-weights), only with Eesh's OK for the download, into `~/s6seed/nartaa/`. Then:
   - real-model parity: `cd ~/s6seed/parity && N=2 setsid nohup bash run_parity_nartaa.sh > run_parity_nartaa.log 2>&1 < /dev/null &` (15-25 min MODEL, cores 28-39);
   - head-init fingerprint (about 2 min): `taskset -c 28-39 nice -n 19 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/miniconda3/envs/rsna/bin/python ~/s6seed/s6_head_init_check.py --ckpt ~/s6seed/nartaa/E_prior_s42_fall.pt --seed 42 --ct ~/s6seed/src/compact_train.py --init ~/s6seed/pretrained/convnext_small_in22k_ft_in1k.safetensors --out ~/s6seed/gates/HEAD_INIT_nartaa.verdict`.
3. **PX_PARITY.verdict**: `echo "PASS <time> window bytes 58/58, surrogate logits max |dlogit| 0 (atto) / 9.5e-7 (small)[, real nartaa <result>]" > ~/s6seed/gates/PX_PARITY.verdict`. The window-byte part is model-independent (MEASURED); the real-checkpoint run is a strengthening, main decides whether to wait for it.
4. **Soup decision** (before A7): HEAD_INIT MATCH -> head seed 42 (soup possible); NO_MATCH -> 43; check not possible before launch -> 42 (GUESS: weakly dominant; small diversity cost if nartaa's init does match).
5. **Paste Amendment 7** (sec 4, fill N, S = 16 x floor(N/2), head seed), then
   `echo "PASS A7 pasted <YYYY-mm-dd HH:MM> pool <N> steps <S> head_seed <42|43>" > ~/s6seed/gates/A7.verdict`.
6. **Arm the launcher** (any time after 5; it holds nothing until the job ahead has run 50 min):
   `setsid nohup env S6_HEAD_SEED_RUN=42 bash ~/s6seed/launch_when_free.sh --px ~/s6seed/px320 --after-job fA_s2002_kc256_r7tb --after-job fA_s2002_kc256_dread2 > ~/s6seed/logs/launch_when_free.log 2>&1 < /dev/null &`
   (drop `env S6_HEAD_SEED_RUN=42` for head seed 43; it must equal A7's head_seed or pre-flight FAILs). If the KC f0 gate fails and both s2002 lines SKIP, re-arm with `--after-job <the job that does run>` or `--now`. Stop everything: `touch ~/s6seed/gates/STOP_s6s43`, then kill the launcher and run pids (runs/s6s43.pid).
7. **Watch**: logs/launch_when_free.log; gates/LAUNCH_s6s43.verdict, SMOKE_s6s43.verdict, RUN_s6s43.verdict, STALL_s6s43, PX_ERRORS_s6s43.txt; IDLE_ALARM lines tagged s6s43. First-epoch check: s/step and loader-wait share in the `S6V100 step` lines vs the smoke; `loss scale min` in the epoch line.
8. **After training**: gold58 for nartaa and ours through compact_infer 34f433fe; `python ~/s6seed/s6_member_gate.py --train-csv <train.csv> --nartaa <preds> --ours <preds> --out ~/s6seed/gates/A7_GOLD58.verdict`; soup path: `python ~/s6seed/s6_soup.py <nartaa.pt> <ours.pt> --out <dir>/E_prior_s4243_fsoup.pt`, score it, `s6_member_gate.py --soup ...`. Then serve parity on the 3 visible studies, OAI-string scan, the 2-model (or soup) GPU commit MEASURING warm GPU s/study, re-sim with sim_s6f_2model.py, entry per sec 4.

## 3. Timings (IST)
| step | when | duration |
|---|---|---|
| S5 unzip | arm ~18:00 10-15 | 1.5-2.5 h (MODEL) |
| S5 px build 16 x 1 + finalize + check | ~20:00 -> ~01:00-05:30 10-16 | 5-9 h (MODEL; 68 CPU-s/study at 1 thread MEASURED by the px agent; 29.9 s/study at 2 threads uncontended, 1 study) |
| GPU queue ahead | KC f0 16:05 -> ~20:30; AD s2003 -> ~01:50; AD s2002 -> ~07:10 10-16 | critique rates (MODEL) |
| PAUSE | s2002 launch + 50 min (or at arming if later) | held through s2002 |
| smoke | ~07:15 | 6-10 min expected, hard cap 45 min |
| S6 run | ~07:25 10-16 -> ~19:40 (17:10-23:10) | 12.2 h at pool 4,349 (9.8-15.7; smoke cap 1.8 s/step = 17.4 h) MODEL |
| gates + kernel | 10-16 evening / night | gold58 ~1 h CPU or minutes on GPU (MODEL) |
| plan stop rule | not launched by 10-17 12:00 -> stop | |

## 4. Amendment 7, final text (paste with N, S and the head seed filled in)
AMENDMENT 7 (S6 member s43). Written <date time IST>, before any S6 GPU use.
1. Member: `E_prior_s43_fall.pt` trained by ~/s6seed/ct_s6v100.py v2 (sha256 e5345b38...) = compact_train 14767c7a patched for fp16 autocast + GradScaler (a non-finite loss skips the step without touching the loss scale; ABORT at loss scale < 256, 100 consecutive bad steps or > 2 % bad steps in an epoch), DataLoader timeout 600 s, all-data, contiguous px read. Recipe ~/s6seed/recipe_s6s43.json v2 (sha 479d57f5): convnext_small from timm init 2457488b, 16 epochs, batch 2, 64 windows, crop 288, lr 1e-4 / head 1e-3, wd .05, grad ckpt, spatial attention. RSNA only.
2. Differences vs the scored S6fall (nartaa E_prior_s42_fall, public .950), all intended: (a) seed 43 for the data, sampler and augment streams; head-init seed <43 | 42 (soup option)>; (b) labels blendjev_sol: lixin's GPT-5.6 sol56 read replaces nartaa's non-public Gemini 3 Flash read; (c) pool N = <N> (every non-gold study the px build rendered; nartaa 4,263 after an unpublished pool_ok filter); 16 epochs = <S> = 16 x floor(N/2) optimizer steps (34,784 at 4,349; nartaa 34,096); (d) fp16 on the V100 (nartaa bf16); (e) pre-rendered px cache through compact_infer 34f433fe (PX_PARITY PASS). A gate miss cannot be attributed to one of (a)-(e).
3. Before the GPU (launch_when_free.sh v2 enforces): PX_BUILD.verdict PASS (gold 58, pool N >= 4,328, 0 missing); PX_PARITY PASS; check-targets pool == N and gold == 58; 300-step smoke within 2,700 s: rc 0, 0 non-finite losses, median s/step <= 1.8, peak reserved < 15.5 GB, scaler skips <= 25, final loss scale >= 1,024, minimum loss scale >= 256. At most one automatic resume after a crash, none after an ABORT.
4. Gate (RELATIVE bars only: gold58 is in-sample for both members' labels, JEV weights were fit on gold studies): g1 our gold58 >= nartaa gold58 - .015; g2 rank-mean(nartaa, ours) gold58 >= nartaa - .003; serve-path parity on the 3 visible studies; OAI-string scan.
5. Entry (replaces "re-sim with c x 1.25"): compact_infer 34f433fe runs each convnext_small checkpoint as a full forward in sequence, so 2 checkpoints cost c2 = max(decode, 2 x 7.3 s) per study per T4, about U(14.6, 16) (MODEL). (i) Soup path, only with head seed 42 and HEAD_INIT MATCH: soup(nartaa, ours) replaces nartaa's file if soup gold58 >= nartaa gold58 - .003 (s6_member_gate.py --soup); 1-model cost, today's G 5.75. (ii) Otherwise 2 models only if the S6F stage starts at kernel age <= G2 = 4.5 h, else nartaa alone (folder chosen at S6F start), SKIP after G 5.75 as today. MODEL at G2 4.5: 2 models in .49 of sessions, nartaa alone .45, cut-to-canary .028 (today .023). The GPU commit must MEASURE warm GPU s/study; if > 16 s, re-run sim_s6f_2model.py and lower G2 before submitting.
6. One change vs a team best that contains S6F; read under R1. No entry if: not launched by 10-17 12:00, RUN verdict FAIL, or any gate FAIL.

## 5. Open issues
- nartaa's checkpoint is on neither the laptop nor the box. It blocks the real-model parity, the head-init check, nartaa's gold58 bar and the soup; it does not block S5 or training. Download needs Eesh's OK.
- fp16: guards now catch scale decay and forward overflows, but there is no fast fallback if the run ABORTs (fp32 about 4-8x slower, GUESS). Each epoch's checkpoint allows a resume; lowering S6_MIN_SCALE on resume is main's call.
- Soup quality and nartaa's head-init order (ae18d2e1 not available) are GUESS until s6_head_init_check and the gold58 read.
- 2-model runtime is a MODEL on one MEASURED commit (7.3 s/study/T4); P(2 models used) is only about .49 at G2 4.5.
- Loader I/O with 8 workers on the HDD is a MODEL (single-reader 0.45 s/study MEASURED); sda2 (446 GB SSD, unmounted, root needed) stays Eesh's call.
- The crash-resume path is tested in dry run (C19) and on CPU (T4 resume == uninterrupted), never on the V100.
- Pool floor 4,328 is my reading of nartaa's max_fail_frac .005 (S5_POOL_FLOOR / S6_POOL_FLOOR override it, but A7 must say so). pool_sha256 378f782f stays unmatched (hash recipe unknown).
- If both s2002 lines SKIP, the launcher waits up to 36 h without holding PAUSE; main re-arms.
- Hygiene: I started and stopped only my own test processes, no pkill, no kill of others; process lists via pgrep on my own patterns. Earlier agents' notes stand (one anchored pkill of the px agent's own driver at 14:57; partial curl command lines once seen in ps, not stored).
