# K9N RUNBOOK: K9 (parent + 10-member fleet) + N (the public .949 CoAtNet by nartaa), 3-leg rank blend, STRICT

Written 2026-10-08 ~10:00-11:30 IST by Claude (subagent). OPTIONAL kit: Eesh decides whether to push or submit it.
Nothing was created, versioned, pushed, forked or submitted on Kaggle. The only Kaggle action was a READ: the weights file
was downloaded once to the box (`kaggle datasets download ... -f raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`,
279 MB) for the CPU tests. No em dashes in this file. Tags: **MEASURED**, **MODEL**, **GUESS**.

`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`,
`PY` = `C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe`, `B` = `training@192.168.10.30`
(`ssh -o BatchMode=yes`, append `< /dev/null`). Box tests: CPU only (CUDA hidden, nice 19), writes only `~/k9n_test`.
Rule context: `Downloads/rsna-knee/PUBLIC949_ASSESSMENT.md` (incl. its CORRECTION: the host's 09-21 answer on OAI is a
conditional yes; residual doubt = "generally accessible to all" / geo access). Any K9N submission attaches the OAI-trained
weights dataset, so it is identifiable from its metadata. Keep at least one clean (K9 / K9-KF) final pick.

## 0. What K9N is

| kernel | slug | cells | submission.csv | also written (diagnostics, never scored) |
|---|---|---|---|---|
| K9N-a | `eeshsaxena/rsna-sub-k9n-a` | parent 0-30 + K9 fleet cell 31 (strict W .50) + **N cell 32** | rank_pct(**.10** r(P) + **.30** r(F) + **.60** r(N)) | `submission_pfn_303040.csv`, `submission_pfn_403030.csv`, `submission_n949.csv`, the K9 fleet files (w50 = K9-main bytes, w40, w60, fleetsolo) |
| K9N-b | `eeshsaxena/rsna-sub-k9n-b` | same, cell 32 literal differs | rank_pct(**.30** r(P) + **.30** r(F) + **.40** r(N)) | `submission_pfn_103060.csv`, `submission_pfn_403030.csv`, ... |

- **Cells 0-31 are the K9-main notebook byte for byte** (asserted by `build_k9n.py`; K9-main nb sha16 `50d6ef5014ef751d`).
  K9N reads the **same** `eeshsaxena/rsna-knee-cnxt-fleet` version as K9 (manifest sha256 `43038ec6...aadf1`); it ships no
  dataset of its own. Metadata = K9-main + `nartaa/rsna-knee-publication-swa-weights-20261007`.
- **N stage (cell 32)** runs only after the fleet stage finished CLEAN in strict mode (workers gone, GPUs free):
  - first action: delete `submission.csv` (the P+F bytes stay in `submission_w50.csv`); it is rewritten only as the
    LAST step of a clean N stage. Any failure = **no submission.csv** (the slot errors, never silently scores P+F or P).
  - weights resolved at the public notebook's own paths, sha256 `7e5315da...44ef` asserted in the cell and again per GPU
    by the child.
  - child process = the public cell 4 **verbatim** (sha256 `bcd864cb...`, == the author's stated source sha and
    `checkpoint-manifest.json` source_sha256) + 11 marked K9N-P patches (`n949/make_child.py`: private `_fastread.py`
    dir, exact weights path, output never `submission.csv`, receipt npz with both views / fallback / empty ids / weights
    sha per GPU). Reader, 96 slices / 5 slots, 384 cache, K94, native 320 crop, plain + anatomical mirror, fp16 autocast,
    one model per GPU, fork prep pool: unchanged. Child sha256 `8d567a63...34ca`, embedded base64 in the cell.
  - gates: ids == parent ids, finite, [0, 1], P == (plain + mirror)/2 exactly, mirror != plain on every non-empty study,
    fallback + empty <= 2 % (0 on the 3-study commit), N's own csv == its rank formula, label order, recipe constants,
    Spearman(P, N) mean >= .30 once n >= 50.
  - blend: average-tie percentile rank per label; F = `submission_fleetsolo.csv` (raw-prob mean of the 10 members),
    P = `_k2_parent.csv`, N = the child's raw probability (fallback / empty rows median-filled, rank-neutral).
- **Attribution**: the child header and the cell header credit nartaa (code + weights), dreaddevelopment's Raptor
  CoAtNet/MIL recipe and timm, and name the weights dataset and its model-use terms.

### 0.1 Weights (MODEL/GUESS, `k9n_weights.py`, `.out`; `k9n_weights_stress.py`, `.out`)

Same binormal macro model as `SP/jump2/lbmath.py` / `SP/p949/p949_model.py`. Inputs P .943, F .944 (K7i SOLO, K9 SOLO not
read yet), P+F W .50 = .946, N .949; r_PF .923-.967 (point .945, MODEL); r_PN .899-.966 (point .925, GUESS from the 92-team
.950 cluster); r_FN scanned .86-.94; N_true .949 or .9475 (winner's curse: .949 is the best of the author's >= 6 public
reads). 88 PSD scenarios.

| P / F / N | C92 (central, r_FN .92) | N_true .9475 | min over 88 | max regret | read as displayed LB (minus ~.0005 optimism) |
|---|---|---|---|---|---|
| **K9N-a .10 / .30 / .60** | .9505 | .9497 | .9485 | **.0010** (minimax on the .01 grid: .08/.29/.63, .0009) | **~.950** (range .948-.952) |
| **K9N-b .30 / .30 / .40** (the brief's K9N-a) | .9499 | .9493 | .9477 | .0026 | **~.949-.950** |
| .40 / .30 / .30 (the brief's K9N-b, DROPPED) | .9492 | .9488 | .9471 | .0036 | ~.949 |
| K9-main P+F .50 / .50 / 0 | .9460 | .9460 | | | .946 (measured family) |

- **Deviation from the brief, justified:** per-scenario optimal N weight is .51-.73 (median .59) in every one of the 88
  scenarios, so N .40 and N .30 are both below the optimum everywhere. .40/.30/.30 is dominated by .30/.30/.40 at every
  N_true >= .944 in the stress table (it never wins), so it ships only as a diagnostic file. K9N-a takes the
  near-minimax-regret triple; K9N-b keeps the brief's .30/.30/.40 as the hedge that wins only if N's true strength is
  ~.944 or lower (public .949 not carrying to private). Both triples are written by both kernels.
- LB context (assessment, MEASURED counts 10-08): .950 = rank ~124-215 (silver), .949 = 216-300 (silver cut .949 at
  rank 272), our .946 = 373-418. The 92-team .950 cluster is most likely P+N.
- **If K9 SOLO reads away from .944** (expected .943-.946): re-run `k9n_weights.py` with F = that read before pushing; if
  the best triple moves, rebuild with `build_k9n.py --w3_a P,F,N` (only the cell 32 literal changes; re-run sec 2.2 and
  the box stub test, ~10 min).

### 0.2 Runtime (MODEL; `tests/runtime_k9n.py`, `.log`)

| case | parent | K9 fleet stage | N stage | K9N end | margin to the 8.6 h wall | margin to 9 h |
|---|---|---|---|---|---|---|
| central | 6.0 h | 58.2 min (MEASURED microbench inputs) | 26 min (25 + 1 start) | **7.40 h** | **1.20 h** (K9 alone 1.63) | 1.60 h |
| slow | 6.0 h | 76.1 min | 41 min (40 + 1) | **7.95 h** | **0.65 h** (K9 alone 1.33) | 1.05 h |

- N inputs (MODEL): <= the author's 26.1 min submit-to-score turnaround on the same 2 x T4 rerun (incl. queue); decode
  <= ~7 min (his 7.2 min efficiency run uses the same reader); slow = x1.6.
- **Guard (cell 32):** the stage is SKIPPED (LOUD, strict: no submission.csv) when < 20 min remain to the wall after the
  fleet; once started, the child and its fork prep pool are KILLED as a process group at kernel start + 8.6 h - 2 min
  (strict: no submission.csv). N therefore never runs past 8.6 h; the 9 h limit keeps >= 0.4 h.
- **Deviation from the brief, justified:** a 45 min start floor was suggested; I used 20 min + kill at the wall. Starting
  with 20-45 min left can still finish (central N ~26 min) and costs nothing when it cannot (the kill comes at the same
  8.6 h wall, the outcome is the same "no submission" a skip gives), so the kill is the real guard and the floor only skips
  hopeless cases. The cell logs its central / slow estimate (scaled by the fleet's own speed) and a WARNING when the time
  left is below the slow estimate.
- The N stage is cut only if the parent runs past ~6.62 h in a slow session (~7.16 h central). K9 alone tolerates 7.33 h.
  Parent 6.0 h = the MEASURED end-to-end LB turnaround of one session (an upper bound for that session; the older
  ENDGAME compute model said 7.7-8.05 h). A parent near 7 h in a slow session is the scenario where K9 scores and
  K9N errors (killed at the wall): that is the price of the N leg, and the reason to keep a clean K9 pick.

## 1. Quota and slots (read first)

- Each K9N commit runs the whole parent on the 3 public studies: ~0.5 h GPU each (K9_RUNBOOK sec 1 MODEL); N adds ~1-2 min.
- 10-08 budget (K9_RUNBOOK sec 1): 2.52 h until 10-10 05:30 IST; K9 main + SOLO ~1.0 h. K9N-a + K9N-b ~1.0 h more =
  ~2.0 h of 2.52 (~0.5 h spare). If K9N is wanted today, push **K9N-a only** unless `kaggle quota` shows >= 1.2 h left
  after the K9 commits; push K9N-b after the 10-10 reset.
- K9N needs no dataset version, so it never conflicts with K9 on the dataset slug; but **never version
  `eeshsaxena/rsna-knee-cnxt-fleet` while a K9N commit or submission is running or pending** (same rule as K9).
- 2 submission slots if both are submitted.

## 2. Operator steps

### 2.1 Preconditions

1. K9 FINAL build exists in `SP/k9/build` (main session, `build_k9.py`), its dataset version is **live** and is the K9
   one:
   ```powershell
   kaggle datasets status eeshsaxena/rsna-knee-cnxt-fleet      # ready
   kaggle datasets files  eeshsaxena/rsna-knee-cnxt-fleet      # 21 files incl. f{0..4}_s1043f75_raw_ttr224-... and f{0..4}_s1044t288_raw_ttr288-...
   kaggle datasets files  nartaa/rsna-knee-publication-swa-weights-20261007   # raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt 292840787 B present
   ```
2. Eesh's decision on the rule risk (assessment sec 5-6 + CORRECTION) is recorded.

### 2.2 Re-check that the kit to push equals the tested kit (laptop, ~40 s; DONE once at 11:05, repeat if K9 is rebuilt)

`SP/k9n/build` was built at ~11:05 from the main session's `SP/k9/build` (10:54) and is byte-identical to the kit the box
A0 ran (`SP/k9n/build_k9base_tested`, built from the `k9base` copy); the k9base upload folder == `SP/k9/build/upload`
(21 files, sha256 lists equal). Re-run only if `SP/k9/build` changes:
```bash
cd "$SP/k9n"
$PY -B n949/make_child.py --check                                   # "n949_child.py OK (== make()), sha256 8d567a63..."
$PY -B build_k9n.py --k9_build "$SP/k9/build" --out "$SP/k9n/build_recheck" --compare "$SP/k9n/build_k9base_tested"
#   gate: "COMPARE OK: both kernels byte-identical to .../k9n/build_k9base_tested" and "K9N BUILD OK FINAL"
#   (compares K9-main nb sha, fleet manifest sha256, members, TTA, child sha, weights sha and both kernel files)
#   If COMPARE fails, K9 was rebuilt from different members: rebuild SP/k9n/build from it, re-run
#   tests/t_k9n_build.py and the box stub A0 (sec 4) before any push.
$PY -B -W ignore tests/t_k9n_build.py | tail -n 1                  # "TK9NBUILD PASS 39/39"
```

### 2.3 Push (PowerShell; after K9 main + SOLO are pushed, the K9 dataset version is live)

```powershell
$K9N = "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k9n\build"
Test-Path "$K9N\DO_NOT_PUSH_TEST.txt"                       # must be False
kaggle quota
kaggle kernels push -p "$K9N\kernel\rsna-sub-k9n-a"
kaggle kernels push -p "$K9N\kernel\rsna-sub-k9n-b"        # only if quota allows (sec 1)
kaggle kernels status eeshsaxena/rsna-sub-k9n-a             # until complete; write the version V down
kaggle kernels output eeshsaxena/rsna-sub-k9n-a -p "$K9N\out_a_v<V>" --page-size 200
kaggle kernels output eeshsaxena/rsna-sub-k9n-b -p "$K9N\out_b_v<V>" --page-size 200
```
- An instant ERROR with an empty log = Kaggle refused the session (low quota): push again after the 10-10 05:30 reset.

### 2.4 Gates (Git Bash; each must end "K9NCHECK PASS")

```bash
BD="$SP/k9/build"; K9N="$SP/k9n/build"
MEM=$($PY -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['members']))" "$BD/K5_BUILD.json" | tr -d '[:cntrl:]')
MAN="$BD/upload/rsna-knee-cnxt-fleet/fleet_manifest.json"
for t in a b; do
  $PY "$SP/k9n/k9n_check.py" --work "$K9N/out_${t}_v<V>" --k9n_build "$K9N/K9N_BUILD.json" --tag $t \
      --expect_n clean --expect_fleet clean --kaggle_commit --expect_members "$MEM" --expect_manifest "$MAN" \
      --expect_tta "$BD/K5_BUILD.json" --expect_n_ranks log949
done
```
- Part A = every K9-main gate (k2_check unchanged, strict, W .50, members, manifest bytes, TTA AB, commit receipt: 2 GPUs,
  decode probe, parity re-encode, microbench, n_fail 0) on the fleet part, with submission.csv = the fleet's own
  `submission_w50.csv` (sha256 == the fleet status record).
- Part B = N: config (W3, variants, tag, child / source / weights sha), receipt, npz (ids, finite, [0, 1],
  P == (plain + mirror)/2, mirror exercised, weights sha on 2 GPUs, recipe, 0 fallback / 0 empty), N's csv formula,
  **N's 3 public-study ranks == the author's public commit log exactly** (`n949/log949.txt`: the only author-side parity
  available), every `submission_pfn_*.csv` and `submission.csv` == an independent numpy 3-leg rank blend, file sha256
  unchanged, fleet files untouched.
- Also read in the commit log: `[k9n ...] RESULT OK N stage clean`, two `[n949-child] [weights] sha256 7e5315da...` lines,
  `[k9n ...] N GPU s/study {...}` (first study includes cudnn warm-up) and the `gpu_free_mib` / host RAM line (the
  parent process keeps its own VRAM; a clean 3-study commit with 0 fallback is the evidence that N fits beside it,
  per-study memory does not grow with n).
- **What the gates cannot see:** the hidden rerun itself: its ~1,300-study fallback count (gated in-kernel at <= 2 %),
  N's T4 runtime there (MODEL sec 0.2; guarded by skip + kill), VRAM / RAM there, and private-LB behaviour of N.

### 2.5 Submit (UTC day = 05:30 IST to 05:29 IST next day)

```powershell
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k9n-a -v <V> -f submission.csv -m "K9N-a strict P/F/N .10/.30/.60: K9 parent + 10-member fleet + public .949 CoAtNet (nartaa weights)"
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k9n-b -v <V> -f submission.csv -m "K9N-b strict P/F/N .30/.30/.40: K9 parent + 10-member fleet + public .949 CoAtNet (nartaa weights)"
```

## 3. Reads (pre-registered, MODEL/GUESS)

| read | expected | rule |
|---|---|---|
| K9N-a | ~.950 (.948-.952) | >= K9 main + .002 = N leg working as modelled; **<= K9 main = audit the N leg** (k9n_check output, `n949_receipt.json`, `rho_mean`) before any further use |
| K9N-b | ~.949-.950 | b > a by >= .001 would say r_FN is high or N_true lower than modelled: refit `k9n_weights.py` with the two reads |
| ERROR (no score) | | strict fail-loud fired on the hidden rerun (wall, fallback > 2 %, gate): the K9N commit log cannot show it; resubmitting the same version will usually fail again |

Final-pick logic stays Eesh's (assessment sec 6, option B): one clean pick (K9 / K9-KF) + at most one K9N pick, and drop
K9N if the host rules against OAI-derived models before 10-22. Re-read threads 741819 / 743416 / 746792 before picking.

## 4. Evidence (2026-10-08, MEASURED; logs `SP/k9n/tests/`, `SP/k9n/logs/`, box `~/k9n_test/runs/`)

| test | where | result |
|---|---|---|
| **TK9NBUILD** (`tests/t_k9n_build.py`, laptop, no model) | laptop | **PASS 39/39**: child == public cell 4 + 11 patches (only the 2 replaced public lines removed); both kernels = K9-main cells 0-31 + cell 32 (W3, variants, child sha, body), metadata = K9-main + weights dataset, private / T4 / no internet / no docker pin; a SIMULATION K9 build refused without `--test`, `--test` marks DO_NOT_PUSH and k9n_check fails it; `--compare` byte-identical rebuild and one-byte-change catch; rank arithmetic cell == numpy reference == pandas (random + tie-heavy); 9 gate-logic cases with a fake child (clean passes k9n_check; P != (plain+mirror)/2, mirror == plain, ids, label order, fallback on a 3-study run, csv formula, child weights sha, rc 1: each fails loud with the right why, submission.csv ABSENT); `--expect_n_ranks log949` PASS on the author's exact ranks and FAIL naming the label + gap on one swap |
| **COMPARE** vs the main session's `SP/k9/build` (10:54) | laptop | **OK**: K9N built from `SP/k9/build` == the tested kit byte for byte; k9base upload == `SP/k9/build/upload` (21 files) (`logs/compare_k9build.log`, `logs/build_k9n_final.log`) |
| **A0n_stub_a / A0n_stub_b** (N on 4 fold-0 studies, stubbed strict fleet) | box CPU | **K9NBOX PASS 46/46 each** (`K9NCHECK PASS 41/41`): weights sha256 (cell + child), child launched once with the pinned bytes, 4 ids == parent ids, shapes (4, 12), finite, [0, 1], label order, recipe, P == (plain+mirror)/2 exactly, mirror != plain on every study (max per study .028-.041), N csv == formula, every pfn file and submission.csv == independent numpy 3-leg blend (max diff 0) and == a scipy-rank reference, fleet files untouched, no child process left, mirror unit test (47 sagittal windows reverse channels, 47 coronal/axial flip width, mirror o mirror == id, eval_windows == gpu_windows bit-identical); N stage 9.1-9.3 min on CPU fp32 (~133 s/study, 8 threads, loaded box; T4 is MODEL sec 0.2) |
| **7 negatives** (stubbed fleet, 2 studies) | box CPU | **K9NBOX PASS 15/15 each**: `missing_weights` (why: not found), `bad_sha` (why: sha256 mismatch), `fleet_fail` (N not started), `time_skip` (10 min left < 20: RUNTIME GUARD skip), `wall_kill` (child started, model loaded, prep done, killed as a process group at the wall after 1.2 min, rc -9, no process left), `child_crash` (injected raise: rc 1), `child_sha` (tampered CHILD_SHA256: refused before launch, and k9n_check flags the config); in every case submission.csv ABSENT and no N file written |
| **A0n_chain_a** (the full K9N-a cells 31 + 32 on the K9 A0 pair, real 10-member K9 fleet dataset byte-identical to `SP/k9/build/upload`, real N weights) | box CPU | **K9NBOX PASS 85/85, K9NCHECK PASS 80/80**: fleet stage clean (10 x KneeFleetNetTTR, all AB, decode probe 40/40, 42.8 min CPU) then N stage clean (4.95 min, 2 studies, kernel age 7.80 h at the end of a 7.0 h stand-in start); part A = every K9-main gate unchanged (strict, W .50, members, manifest sha16 == local build, TTA AB modes / forwards / HALF, K5c npz identities, w40/w50/w60 == independent P+F blends, no KF parent) on the fleet's own P+F file (`submission_w50.csv`, sha256 == fleet record); part B = all N checks; the hand-off proven: the fleet's submission.csv deleted first and recorded, fleet files untouched, submission.csv == independent numpy and scipy 3-leg references |
| **A0n_gold12_a** (N on 12 gold58 studies, stubbed fleet; exercises the n >= 11 gate path) + `box/k9n_gold_sanity.py` | box CPU | **K9NBOX PASS 46/46**; label-order / pipeline sanity: N macro AUC **.937** over 12 labels vs **.585** with the columns permuted (bar .80; NOT a CV read: the author reused gold58 for selection) (`runs/gold_sanity_a.txt`). A twin run on 12 more gold studies was stopped by me at 11:06 to leave CPU to the main session's K9 A0 |

Logs: `SP/k9n/tests/t_k9n_build.log`, `SP/k9n/tests/box/` (copies of the box `~/k9n_test/runs/*_1041.log`, `*_1042.log`,
`*_1052.log` + their `report.json` / `k9n_check.json`), `SP/k9n/logs/`. The first round of box runs (10:22-10:33, kit
bytes before two small cell-32 hardenings: explicit length check, no-clock skip) also passed (7/7 negatives; stub
K9NCHECK 41/41; its harness mirror unit test hit a harness path bug, fixed); all rows above are the FINAL bytes
(nb sha256 a `14d291ff...`, b `df0d2843...`).

What the box A0 cannot show: T4 fp16 numerics (the commit's log949 rank parity covers it), T4 runtime, 2-GPU sharding
(the child's own verbatim consumer-per-GPU code; the commit gate requires 2 weights sha entries = model on both GPUs), the
real parent (stand-in ranks), and the 3 public test studies (no test DICOMs on the box or laptop; the commit runs them).


## 5. Files (`SP/k9n/`, a copy of the `SP/k9` tools; `SP/k9` untouched)

| path | role |
|---|---|
| `build_k9n.py` | builds K9N-a/b from a K9 FINAL build (`--k9_build`), `--compare` against a tested build, `--test` for non-final inputs (DO_NOT_PUSH) |
| `cells/k9n_n949_cell.py` | cell 32 body (strict N stage, guard, gates, blend) |
| `n949/make_child.py`, `n949/n949_child.py` | the child = public cell 4 verbatim + 11 K9N-P patches (each anchor matched exactly once); pinned shas |
| `n949/rsna-knee-0949-anatomical-mirror.ipynb`, `n949/kernel-metadata.json`, `n949/log949.txt` | the pulled public notebook (source of the child) and its public commit log (3-study rank reference) |
| `k9n_check.py` | commit / A0 gate: part A = k2_check unchanged on a shadow dir, part B = N |
| `k9n_weights.py`, `k9n_weights_stress.py` (+ `.out`) | LB model for the triples |
| `tests/t_k9n_build.py`, `tests/runtime_k9n.py` (+ `.log`) | laptop tests (no model), runtime table |
| `box/k9n_box_stage.py`, `box/k9n_box_launch.sh`, `k9n_stage_box.sh`, `box/k9n_gold_sanity.py` | box CPU A0 (fake hidden layout under `~/k9n_test`), gold label-order sanity |
| `k9base/` | a FINAL K9 build made by the k9n copy of `build_k9.py` from `SP/k9/members.json` (09:52) + `SP/k9/ckpt` (identical inputs to the main session's K9 build; used for the box A0 dataset and as the K9N base). Never push or version from it: K9's own `SP/k9/build` is the one that goes to Kaggle |
| `build/` | **the K9N kernels to push** (`kernel/rsna-sub-k9n-{a,b}`), `K9N_BUILD.json`; built from `SP/k9/build` (10:54) |
| `build_k9base_tested/` | the byte-identical kit the box A0 ran (built from `k9base`); reference for `--compare` |
| `build_k2.py`, `build_k9.py`, `make_k2_kernel.py`, `make_kf_cell.py`, `k2_check.py`, `cells/`, `code/`, `box/k5_*.py` | byte copies of `SP/k9` (needed for imports) |

Disk: `k9base/` ~2.2 GB on the laptop (removable after K9N is pushed or dropped); box `~/k9n_test` 2.4 GB (ckpt snapshot
1.1 GB, fleet dataset copy 1.1 GB, weights 279 MB); all removable once K9N is pushed or dropped.
