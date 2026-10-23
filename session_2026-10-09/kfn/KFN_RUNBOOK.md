# KFN RUNBOOK: parent-free FN (K9 fleet + N, F .40 / N .60) and N-only, STRICT, for Saturday 10-10

Written 2026-10-08 22:00-24:00 IST by Claude (subagent), per `Downloads/rsna-knee/RESEARCH_TO_TOP_10-08.md` sec 2 / 2.1.
Nothing was created, versioned, pushed, forked or submitted on Kaggle (no Kaggle call at all). The K9 dataset version
(`eeshsaxena/rsna-knee-cnxt-fleet`, 10 K9 members, manifest sha256 `43038ec6...aadf1`) is the one both kernels read; it
is NOT changed. Box work: CPU only (CUDA hidden, nice 19), writes only `~/kfn_test`. No em dashes. Tags: MEASURED,
MODEL, GUESS.

`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`,
`PY` = `C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe`, `B` = `training@192.168.10.30`.
Read rules for the Saturday reads: `Downloads/rsna-knee/PREREG_10-10.md` (R1-R5, written before any read; sha256 sec 6).

## 0. What the two kernels are

| kernel | slug | cells | submission.csv | diagnostics (never scored) |
|---|---|---|---|---|
| FN | `eeshsaxena/rsna-sub-k9fn` | md, FN-IDS, K9 fleet stage, N stage (MODE FN) | rank_pct(.40 r(F) + .60 r(N)) (P/F/N 0/.40/.60) | `submission_pfn_003070.csv`, `submission_pfn_005050.csv`, `submission_n949.csv`, the fleet files (`submission_fleetsolo.csv`, w40/w50/w60 = rank(F) here) |
| N-only | `eeshsaxena/rsna-sub-nonly` | md, FN-IDS, N stage (MODE NONLY) | rank_pct(r(N)) (P/F/N 0/0/1) | `submission_n949.csv`, `submission_pfn_0000100.csv` (== submission.csv) |

- **No parent at all.** The 31 parent cells (repro_0946) are gone; the FN-IDS cell (`cells/fn_ids_cell.py`) only sets the
  kernel clock T0 and the env defaults the parent used to leave for child processes (OMP/OPENBLAS/MKL 4 threads,
  CUDNN_CONV_WSCAP_DBG, HF offline, RSNA_COMP_ROOT), finds the competition root exactly as the fleet worker does, reads
  test.csv ids (unique, non-empty; sample_submission columns == the 12 targets and same ids), and writes
  `_fn_ids.csv` = the ids x 12 targets all 0.5 (ID-ONLY placeholder), copies it to submission.csv (the fleet cell takes
  its "parent" from there) and `_fn_ids.json`. Any failure: none of the three files exists, every later stage refuses.
- **Fleet stage (FN)** = the K9-main cell 31 BODY byte for byte (K5 cell `e37f0604997cd40f`, strict, W .50, the 10 K9
  members, TTA AB, cell metadata kept); its K2_CFG literal = the K9-main literal + `RHO_MIN_N: 1e9` only (the parent~fleet
  Spearman gate cannot work against a constant placeholder; it is replaced by the F~N gate in the N stage). Because the
  placeholder ranks are all tied, the fleet's w40/w50/w60 files equal rank(F): harmless diagnostics.
- **N stage** = `cells/fn_n949_cell.py` = the K9N cell-32 body (`bd87efb8fa89bbac`) + 10 FN-P patches (`make_fn_cell.py`;
  the test proves FN cell minus the patches == the K9N cell byte for byte). Same child (`8d567a63...`, public cell 4
  verbatim + 11 K9N-P patches), same weights pin (`7e5315da...44ef`), same gates, same 8.6 h wall / 20 min skip / kill.
  FN-P changes: MODE FN / NONLY; every triple has P weight 0 (refused otherwise); NONLY triples are exactly [0, 0, 1];
  the placeholder must exist with all 0.5 and, in FN, `_k2_parent.csv` must be its exact bytes (proves no real parent);
  NONLY needs no fleet stage and refuses if one exists; Spearman gate reference = F (FN) or N's own verbatim csv (NONLY,
  bar .90) once n >= 50; status / log name the MODE.
- **Strict fail-loud** (both): the N stage deletes submission.csv first and writes it only as the last step of a clean
  stage; FN's fleet stage is strict (deletes on any failure) and the N stage refuses to start unless the fleet status is
  clean + strict + fleetsolo bytes unchanged. Any failure anywhere = no submission.csv = the slot errors.
- **Metadata**: FN = competition + `eeshsaxena/rsna-knee-cnxt-fleet` + `nartaa/rsna-knee-publication-swa-weights-20261007`;
  N-only = competition + the N weights dataset. No kernel_sources, no model_sources (the parent's 14 datasets, 2 kernel
  sources and the DINOv2 model are dropped; build + test assert no parent source name appears as a path or slug in the
  notebook text or the decoded child, and the fleet worker code reads only the competition paths). Private, T4, GPU, no
  internet, no docker pin (as K9 / K9N).
- **Attribution**: the md cell and the child header credit nartaa (code + weights), dreaddevelopment's Raptor
  CoAtNet/MIL recipe and timm, and name the weights dataset and its model-use terms.

### 0.1 Weights (MODEL, `tests/fn_weights.py`, `.out`)

F .40 / N .60 as the research specifies (sec 2 row 2: "P weight 0, F .40 / N .60"); kept, not changed. Check with the
K9N binormal model (2-leg, F .942-.946, N_true .9475 / .949, r_FN .86-.94, 30 scenarios): optimal W_N .53-.82 (median
.62); W_N .60 has the smallest max regret of .50 / .60 / .70 (.0005 vs .0011 / .0007), central (F .944, N .949, r .92)
.9504 vs the optimum .9505 at .67. The difference is below one display tick.

### 0.2 N-only deviation from the research text (justified)

The research describes #5 as "the FN kernel at weights 0/0/1 (fleet runs, weight 0)". The brief asked for "just the N
stage", which is what is built: FN-IDS + N stage, no fleet. The scored file is identical by construction (a weight-0 leg
cannot move a per-label rank blend; fn_check asserts every FN file == the same blend with the P leg removed, and N-only
submission.csv == avg_rank_pct(N) exactly), while dropping the fleet removes ~1 h of runtime and the fleet's failure modes
from the N calibration read.

## 1. Runtime and quota (MODEL; `tests/runtime_kfn.py`, `.log`)

| case | fleet stage | N stage | FN end | FN margin to 8.6 h | N-only end | N-only margin |
|---|---|---|---|---|---|---|
| central | 58.2 min (MEASURED microbench inputs, as K9) | 26 min | **1.45 h** | 7.15 h | **0.48 h** | 8.12 h |
| slow | 76.1 min | 41 min | **2.00 h** | 6.60 h | **0.73 h** | 7.87 h |

- Includes ~3 min boot + FN-IDS (GUESS). K9N-a for comparison: 7.40 / 7.95 h at a 6.0 h parent. The skip / kill guards
  can only fire if a session runs > 4x slower than the slow case.
- Commits (3 public studies): MEASURED fleet stage 1.2-1.7 min on the K7i 10-member commits; N on 3 studies ~2 min
  (GUESS) -> FN commit ~7 min kernel time, N-only ~5 min. Quota: ~0.1-0.2 h GPU each (the research's ~0.25 h per
  parent-free commit is the safe figure).
- Box CPU timings (sec 5) are CPU fp32 and say nothing about T4 speed.

## 2. Operator steps

### 2.1 Before pushing (laptop, ~2 min; repeat only if anything in SP/k9/build or SP/kfn changed)

```bash
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
cd "$SP/kfn"
$PY -B n949/make_child.py --check        # "n949_child.py OK (== make()), sha256 8d567a63..."
$PY -B make_fn_cell.py --check           # "fn_n949_cell.py OK (== make()), sha256 2237d8ce..."
$PY -B build_kfn.py --out "$SP/kfn/build_recheck" --compare "$SP/kfn/build"
#   gate: "COMPARE OK: both kernels byte-identical to .../kfn/build" and "KFN BUILD OK FINAL"
$PY -B -W ignore tests/t_kfn_build.py | tail -n 1     # "TKFNBUILD PASS 82/82"
```
- The build refuses a non-FINAL K9 build and asserts the fleet manifest sha256 == the one the K9N kit used (the live
  K9 version). If `SP/k9/build` was rebuilt, COMPARE fails: rebuild, re-run the tests and the box A0 (sec 5) first.

### 2.2 Push (PowerShell; after the 05:30 IST 10-10 quota reset)

Commit order does not matter (no dataset version is involved). Submit order follows PREREG_10-10.md.
```powershell
$KFN = "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\kfn\build"
Test-Path "$KFN\DO_NOT_PUSH_TEST.txt"                       # must be False
kaggle datasets files eeshsaxena/rsna-knee-cnxt-fleet       # 21 files incl. f{0..4}_s1043f75_raw_ttr224-... and f{0..4}_s1044t288_raw_ttr288-... (the K9 version; never version it while these run)
kaggle quota
kaggle kernels push -p "$KFN\kernel\rsna-sub-k9fn"
kaggle kernels push -p "$KFN\kernel\rsna-sub-nonly"
kaggle kernels status eeshsaxena/rsna-sub-k9fn              # until complete; write the version V down (same for -nonly)
kaggle kernels output eeshsaxena/rsna-sub-k9fn  -p "$KFN\out_fn_v<V>" --page-size 200
kaggle kernels output eeshsaxena/rsna-sub-nonly -p "$KFN\out_no_v<V>" --page-size 200
```
- An instant ERROR with an empty log = Kaggle refused the session (low quota): push again later. Never add `--timeout`
  to a submission kernel.
- These are new slugs: the first push creates each kernel (private).

### 2.3 Gates (Git Bash; each must end "FNCHECK PASS")

```bash
BD="$SP/k9/build"; KFN="$SP/kfn/build"
MEM=$($PY -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['members']))" "$BD/K5_BUILD.json" | tr -d '[:cntrl:]')
MAN="$BD/upload/rsna-knee-cnxt-fleet/fleet_manifest.json"
$PY "$SP/kfn/fn_check.py" --work "$KFN/out_fn_v<V>" --fn_build "$KFN/FN_BUILD.json" --kernel fn --expect_n clean \
    --expect_fleet clean --kaggle_commit --expect_members "$MEM" --expect_manifest "$MAN" --expect_tta "$BD/K5_BUILD.json" \
    --expect_n_ranks log949
$PY "$SP/kfn/fn_check.py" --work "$KFN/out_no_v<V>" --fn_build "$KFN/FN_BUILD.json" --kernel nonly --expect_n clean \
    --kaggle_commit --expect_n_ranks log949
```
- **Part 0** (both): `_fn_ids.csv` all 0.5 == its json record; no parent output in the dir (the parent never ran); FN:
  `_k2_parent.csv` == the placeholder bytes; N-only: no fleet file at all.
- **Part A** (FN): k2_check UNCHANGED on the fleet stage's own file (strict, W .50, the 10 members, manifest sha16 ==
  the local K9 build, TTA AB modes / forwards / HALF, K5c worker npz served numbers, commit receipt: 2 GPUs, decode
  probe, parity re-encode, microbench, n_fail 0) + the fleet K2_CFG == FN_BUILD k2_cfg.
- **Part B** (both): N config (W3, variants, TAG, MODE, RHO_REF, child / source / weights sha); receipt (weights sha,
  child sha, rc 0, placeholder sha, Spearman reference leg); npz (ids == placeholder ids, finite, [0, 1], label order,
  recipe K94 / 320 / 384 / 96 / study_lut, weights sha on 2 GPUs, 0 fallback / 0 empty, P == (plain + mirror) / 2
  exactly, mirror exercised); N's own csv == its formula; **N's 3 public-study ranks == the author's public commit log
  exactly** (`n949/log949.txt`, the N parity gate); every `submission_pfn_*.csv` and submission.csv == an independent
  numpy blend AND == the same blend with the P leg removed; N-only submission.csv == avg_rank_pct(N); file sha256s
  unchanged; FN fleet files untouched by the N stage.
- In the commit log also read: `[fn-ids ...] RESULT OK 3 test ids`, `[k2-fleet ...] RESULT OK fleet stage clean: 10
  member(s)` (FN), two `[n949-child] [weights] sha256 7e5315da...` lines, `[k9n ...] RESULT OK N stage clean: ...
  rho(fleet~N)` (FN) / `rho(N-own-csv~N)` (N-only), and the kernel age at the end (~0.1-0.2 h).
- If only the log949 check fails: fn_check prints the flipped label and the probability gap; < 2e-3 = fp16 / cudnn noise
  between images (accept by hand only then), else STOP.
- **What the gates cannot see**: the hidden rerun itself (its ~1,300-study fallback count, gated in-kernel at <= 2 %;
  T4 runtime, MODEL sec 1; VRAM / RAM there), and private-LB behaviour.

### 2.4 Submit (UTC day = 05:30 IST to 05:29 IST next day; only after FNCHECK PASS)

```powershell
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k9fn -v <V> -f submission.csv -m "FN parent-free strict P/F/N 0/.40/.60: K9 10-member fleet + public .949 CoAtNet (nartaa weights)"
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-nonly -v <V> -f submission.csv -m "N-only strict: public .949 CoAtNet (nartaa weights) in our code path, rank(N)"
```
- Saturday list and order (PREREG_10-10.md sec 1): #1 K9N-a (`SP/k9n/K9N_RUNBOOK.md`), #2 FN, #3 K9 SOLO, #4 K9 main
  (`SP/k9/K9_RUNBOOK.md`), #5 N-only. Fallback if FN / N-only fail their gates: slot 2 = K9-KF, slot 5 empty until fixed.
- Rule context (unchanged from K9N): both kernels attach the OAI-trained N weights dataset; keep a clean (no N) final
  pick unless the host OKs OAI-trained weights.

## 3. Reads

Pre-registered in `Downloads/rsna-knee/PREREG_10-10.md` (R1 FN vs K9N-a, R2 N-only <= .947 = N-path audit, R3 K9 SOLO,
R4 timestamps, R5 LB / band rank). FN expected = K9N-a's display (-.0002 to +.0005, MODEL); N-only .948-.949 (GUESS).
An ERROR on FN / N-only = a strict gate fired on the hidden rerun (or a Kaggle-side error); the commit log cannot show
which, so audit before resubmitting the same version.

## 4. Tests (summary; details sec 5) : ALL PASS

| test | where | result |
|---|---|---|
| TKFNBUILD `tests/t_kfn_build.py` | laptop | **PASS 82/82** |
| A0fn_chain (full FN: FN-IDS + real 10-member K9 fleet + N, K9 A0 pair) | box CPU | **KFNBOX PASS 95/95** |
| A0fn_stub (FN, stubbed strict fleet, 4 studies) | box CPU | **PASS 55/55** |
| A0no_n4 (N-only, 4 studies) | box CPU | **PASS 42/42** |
| A0no_gold12 (N-only, 12 gold58 studies, n >= 11 gate path) + kfn_gold_sanity | box CPU | **PASS 42/42**, KFNGOLD PASS (macro .9369) |
| 13 negatives (9 FN, 4 N-only) | box CPU | **13/13 PASS** (no submission.csv in every case) |
| runtime model `tests/runtime_kfn.py` | laptop | FN 1.45 / 2.00 h, N-only 0.48 / 0.73 h (central / slow) |

## 5. Evidence (2026-10-08 22:57 to 23:36 IST, MEASURED; copies in `tests/box/`, box `~/kfn_test/runs/*_2257*`)

| test | result |
|---|---|
| **TKFNBUILD** (laptop, no model) | **PASS 82/82**: T1 FN cell == make_fn_cell.make(); FN cell minus the 10 FN-P patches (each found once) == the K9N cell byte for byte; FN-P4 only re-indents the S2 block. T2 FN = md / FN-IDS / fleet (K9-main cell 31 body byte for byte, metadata kept, K2_CFG == K9-main + RHO_MIN_N only) / N (MODE FN, 0/.40/.60, child == the K9N child 8d567a63); N-only = md / FN-IDS / N (MODE NONLY, 0/0/1); metadata exact (no kernel / model source, private, T4, GPU, no internet, no docker); no parent source name as a path / slug in notebook text or decoded child (17 parent sources checked); every /kaggle/input path is competition / fleet / N weights (or the child's unreachable verbatim fallbacks); fleet worker reads only competition paths; nbformat v4 valid. T3 sim build refused without --test, --test marked DO_NOT_PUSH, fn_check fails it. T4 --compare byte-identical rebuild + one-byte catch. T5 rank arithmetic cell == fn_check == pandas (random + ties), fail-loud names. T6 FN-IDS: clean (placeholder in test.csv order, RSNA_COMP_ROOT, T0) + 5 failures (no test_series, duplicate ids, empty test.csv, sample_submission column order, sample_submission ids) leave none of the 3 files (a stale submission.csv removed too), no exception escapes. T7 fake-child gate logic: FN and N-only clean paths pass fn_check; 25 negatives fail loud with submission.csv ABSENT (P weight != 0, RHO_REF wrong, MODE unknown, placeholder missing, real parent bytes, fleet not clean, NONLY triple != [0,0,1], NONLY with fleet files, NONLY placeholder missing, and 8 child corruptions x 2 kernels); the FN cell in K9N mode still passes k9n_check. T8 log949 parity PASS on the author ranks, FAIL naming the label + gap on one swap (both kernels). T9 n = 60: FN F = N + noise passes, F with reversed columns FAILS on fleet~N Spearman; N-only N~own-csv Spearman == 1 |
| **A0fn_chain** (box CPU, 2 fold-0 studies, real K9 fleet dataset byte-identical to the live upload: "DS IDENTICAL (21 files)", real N weights) | **KFNBOX PASS 95/95**: FN-IDS found the competition root itself (RSNA_COMP_ROOT unset) and wrote the placeholder; fleet stage clean (10 x TTR members, all AB, decode probe 40/40, 33.9 min CPU); N stage clean (4.0 min); part A = every K9-main gate via k2_check unchanged (strict, W .50, members, manifest sha16 == local build, TTA AB modes / forwards / HALF, K5c served numbers) + K2_CFG == FN_BUILD; part B all N checks; submission.csv == numpy blend == the blend with the P leg removed == a scipy-rank F/N .40/.60 reference (max diff 0); mirror unit test; no child left. **Cross-kit:** submission_fleetsolo.csv == the K9N A0 chain's (same 2 studies, run WITH the parent stand-in) max diff **0.0**: removing the parent does not change F; N probs == K9N's to 6e-8 (CPU fp32 ulp) |
| **A0fn_stub** (4 studies) / **A0no_n4** (same 4) | **PASS 55/55 / 42/42**; N probs of the two kernels identical (max diff 0.0): same N path; N-only rho(N-own-csv~N) 1.0 |
| **A0no_gold12** (12 gold58 studies) + `box/kfn_gold_sanity.py` | **PASS 42/42**; n >= 11 gate path (2 % fallback cap) exercised; label-order sanity: N macro AUC **.9369** over 12 labels (== K9N's .937 on the same set), the N-only submission.csv gives the same .9369 (rank-invariant), columns-permuted control .5847 (bar .80; NOT a CV read: the author reused gold58 for selection) |
| **Negatives** FN (stub fleet, 2 studies) | missing_weights, bad_sha, fleet_fail, time_skip (10 min left < 20: SKIPPED), wall_kill (child killed at the wall, rc -9), child_crash (rc 1), child_sha (refused before launch, and fn_check flags the config), ids_fail (no test_series: FN-IDS fails, N refuses "placeholder unreadable"): **8/8 PASS** |
| **Negatives** FN real chain | fleet_missing (fleet dataset not attached: the real fleet cell refuses strict before any worker, N not started): **PASS 32/32** |
| **Negatives** N-only | missing_weights, bad_sha, child_crash, ids_fail: **4/4 PASS** |

In every negative: submission.csv ABSENT, no N fail-loud file, the why names the cause, no child process left.

Risk notes:
- FN-IDS asserts sample_submission ids == test.csv ids. This is not a new hidden-rerun assumption: the scored parent
  (parent cell 21) raises on the same mismatch, and it scored on the hidden set (K7i etc.).
- What the box A0 cannot show: T4 fp16 numerics (the commit's log949 rank parity covers N), T4 runtime, 2-GPU sharding
  (the commit gate requires 2 GPUs + 2 weights sha entries), the 3 public test studies.

## 6. Files and hashes (`SP/kfn/`; `SP/k9` and `SP/k9n` untouched)

| path | role |
|---|---|
| `build_kfn.py` | builds both kernels from the K9 FINAL build (`--k9_build`, default `SP/k9/build`), asserts the K9N child / cell / manifest, `--compare`, `--test` (DO_NOT_PUSH) |
| `make_fn_cell.py`, `cells/fn_n949_cell.py` | the N cell = K9N cell + 10 FN-P patches (each anchor once); `--check` |
| `cells/fn_ids_cell.py` | FN-IDS cell |
| `cells/k2_fleet_cell.py`, `cells/k9n_n949_cell.py`, `n949/*`, `k2_check.py`, `k9n_check.py` | byte copies of the k9n kit (inputs; `k9n_check` is imported by fn_check) |
| `fn_check.py` | commit / A0 gate for both kernels |
| `tests/t_kfn_build.py` (+ `.log`), `tests/runtime_kfn.py` (+ `.log`), `tests/fn_weights.py` (+ `.out`) | laptop tests, runtime model, weight check |
| `box/kfn_box_stage.py`, `box/kfn_box_launch.sh`, `box/kfn_gold_sanity.py`, `kfn_stage_box.sh` | box CPU A0 under `~/kfn_test` |
| `build/` | **the kernels to push** (`kernel/rsna-sub-k9fn`, `kernel/rsna-sub-nonly`), `FN_BUILD.json` |
| `logs/`, `tests/box/` | build log, copies of the box run logs and reports |

Hashes (sha256): `rsna_sub_k9fn.ipynb` 4889578e9d63b3a3..., `rsna_sub_nonly.ipynb` 668c11b83d99aaaa...,
`FN_BUILD.json` 3dbbfa2ff76d0c32..., `cells/fn_n949_cell.py` 2237d8cef51bddde..., `cells/fn_ids_cell.py` e1c398d18f155112...,
fleet manifest 43038ec629a8f0d3...aadf1 (== K9 / K9N), K9-main nb sha16 50d6ef5014ef751d, child 8d567a63220fe8c6...,
weights 7e5315dad125b99f.... **PREREG_10-10.md (with the 10-09 01:20 G320' FAIL status note, before any read):
d195b34da92c9a1e3397b3f17a8439cebee73bc9c37bca1e9816f9c7560b030a.**

Disk: box `~/kfn_test` ~1.5 GB (fleet dataset copy 1.1 GB, weights 279 MB, runs); removable once FN / N-only are pushed
and scored.

## 7. Later: how FN takes the K10 288 frac75 s1045 set (NOT built; plan only)

The 320 gate FAILED (coordinator, 10-09), so K10 = set E = 288 frac75 s1045 (Q4), landing ~Friday 10-09 afternoon. FN is
the vehicle (research L2 / R1); the Saturday FN kernel itself stays exactly as built and tested here.
1. **Do not version `eeshsaxena/rsna-knee-cnxt-fleet` until every Saturday commit AND submission that reads it (K9N-a,
   FN, K9 SOLO, K9 main, N-only) has scored** (whether a pending rerun reads its committed version or the latest one is
   not documented).
2. Collect + gate set E as K9 did sets C / D (`SP/k9/k9_collect.sh` copy: picker closing line, cv rescore, model_sha,
   receipt / bars vs the same fold of set C, min epochs per the research: last-3 bars at the 21-epoch level).
3. Choose the composition by the pre-registered CPU gold58 / OOF read of C+D / C+E / D+E (incl. the C-E correlation)
   BEFORE any public read (research L2, verify_E M10). Parent-free, all three sets fit: + 5 x TTR288 A+B members adds
   ~26 min central / ~32 min slow per T4 (MEASURED microbench 466 / 532 ms per member per study), so a 15-member
   FN-wide ends ~1.9 h central / ~2.6 h slow, far inside the 8.6 h wall.
4. Build: a new K9-style FINAL build with the new members + manifest (new dataset version), then a `build_kfn.py` copy
   with: the K9 FINAL-build asserts widened to the new set map (C/D/E), the "manifest == the K9N kit's" assert replaced
   by "== the new build's own manifest", a NEW slug (e.g. `rsna-sub-k10fn`, so the scored FN version is never
   overwritten), and F/N weights refit from the Saturday reads (r_FN from N-only, K9 SOLO, FN, K9N-a; research sec 3.5).
   The FN-IDS and N cells are reused byte for byte (re-run `tests/t_kfn_build.py` T1 / T5-T9 and a box A0 chain on the
   new dataset before any push).
