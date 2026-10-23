# K3 RUNBOOK: two fleet members (M1 family-1 + M2 TTR), two kernels, one dataset version

Written 2026-10-04 ~22:00-23:00 IST by Claude. Nothing was created, versioned, pushed or submitted on Kaggle. The
operator (Eesh) runs every `kaggle` command below. No em dashes in this file.

`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`,
`PY` = `C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe`, `B` = `training@192.168.10.30`
(`ssh -o BatchMode=yes`), `PYK` = `~/miniconda3/envs/kmimic/bin/python` (box, Kaggle-matching libs).

## 0. What K3 is

| Kernel | Slug | MAIN_MODE | submission.csv | MEMBERS (K2_CFG literal) |
|---|---|---|---|---|
| K3-main | `eeshsaxena/rsna-sub-k3-fleet` | **strict** (new) | rank_pct(.75 r(P) + .25 r(F)), F = raw-prob mean(M1, M2) | [M1, M2] |
| K3-ttrsolo | `eeshsaxena/rsna-sub-k3-ttrsolo` | solo | F = M2 alone (raw probs, parent row order) | [M2] |

- M1 = `f0_s42_raw_cnxt320x24_cv8810_best.pt` (the K2 member: family 1, ConvNeXt-T 320, mix50 labels, ep 8, val cv .8811,
  model_sha `fa35880fca963652`, sha256 `dc7e1495...`).
- M2 = the TTR v4 member from `~/fleet_runs/f0_s1042_ttr` (model_mod `fleet_model_ttr`, ttr_size 224, model_sha
  `f141aeed136dfc01`, pure steven_v4 labels). **Built and tested tonight with the then-best
  `f0_s1042_ema_ttr224-320x24_cv8763_best.pt` (ema, ep 21 of 24, val cv .876335, sha256 `e62c9e12...`)**; the final
  file is swapped in after the run ends (section 2).
- Both kernels read ONE new version of `eeshsaxena/rsna-knee-cnxt-fleet` (13 files, 225.3 MB): M1, M2,
  `fleet_common.py`, `fleet_model.py`, `fleet_model_f2.py`, **`fleet_model_ttr.py`**, `fleet_worker.py` (= the box
  bigjump worker, `d5edacbe...`), `fleet_manifest.json` ([M1, M2] + `sources`), 4 `parity_<sid>.npz`,
  `dataset-metadata.json`. `kaggle datasets version` REPLACES the whole file set: the new version is exactly this folder.
- Both kernels: parent cells 0-30 = repro_0946 unchanged (sha-pinned), cell 31 = `cells/k2_fleet_cell.py` (K3 patch)
  + its `K2_CFG` line. `dataset_sources`, `kernel_sources`, `competition_sources`, `model_sources`, machine, GPU,
  internet: identical to the proven K2 kernel (checked by `build_k3.py`).
- Only `submission.csv` is scored (Kaggle rejects other names, measured 10-04). The other files are diagnostics.

**Strict (K3a)**: same blend as `blend`, but on ANY failure (worker error, missing dataset, wrong member set, wall, gate,
write error, exception outside the guarded path) `submission.csv` is DELETED, never restored to the parent bytes. A
failed hidden rerun then errors; our LB best is unaffected and the read is unambiguous (no silent 0.943).

**Member filter (K3b)**: `K2_CFG['MEMBERS']` = the exact checkpoint file names. The cell checks them against the
attached manifest BEFORE any worker starts (a name not in the manifest = fail loud: wrong dataset version attached),
gives the workers a private FLEET_DIR (`/tmp/k2_members_*`, links to every dataset file except unselected
checkpoints + a filtered manifest; removed after the stage), requires the workers to report exactly those names, and
the full-member-count gate counts the filtered members. Both K3 kernels name their members explicitly, so a commit
that mounts the OLD dataset version (pushed before the new version is `ready`) fails loud instead of running M1 alone.

## 1. Files (`SP/k3/`, a copy of `SP/k2/`; `SP/k2` and the box dirs were not edited)

| Path | Role |
|---|---|
| `build_k3.py` | ONE command: upload folder + both kernels (wraps `build_k2.py`; M2 = the one `f0_s1042_*_best.pt` in `k3/ckpt`, refuses 0 or 2+) + post checks (cell 31 text, `K2_CFG`, metadata == K2) |
| `build_k2.py` | K3-aware builder: TTR members (model_sha vs the SHIPPED `fleet_model_ttr.py` + `fleet_model.py`, `ttr_size_t` == args), `--main_mode strict`, `--members`, `--kernel_only` (one more kernel against an existing upload, re-verified byte for byte), duplicate fold/seed and two-tags-of-one-run refused; old flags unchanged |
| `make_k2_kernel.py` | + `--main_mode strict`, `--members` (written into `K2_CFG` only when given) |
| `cells/k2_fleet_cell.py` | + K3a strict, K3b member filter, K3c manifest sha256 in log + receipt (header documents all three); blend / solo outputs unchanged when unset (one solo exception, sec 9.4) |
| `k2_check.py` | strict-aware (twin `submission_w<W>.csv` when clean, `submission.csv` ABSENT when not), member filter (names + count), `--expect_members`, `--expect_mode`, `--expect_manifest` (attached bytes == local build) |
| `code/` | `fleet_worker.py` + `fleet_model_ttr.py` + `pack_fleet_ds.py` = byte copies of box `~/bigjump_build/code` (sha256 `d5edacbe`, `25b41631`, `791f4df9`); `fleet_common.py`, `fleet_model.py`, `fleet_model_f2.py` unchanged (`c112dac9`, `2a3b176d`, `bec2c230`) |
| `box/k3_pick_ttr.py` | box: final M2 pick (closing json line, tag rule, cv == line, OOF mtime == ckpt mtime, TTR sha) + `--copy` |
| `box/k3_box_launch.sh` | box: A0 (both kernels, 2 studies) and gold58 CPU stage tests, detached, nice 19 |
| `box/k2_box_stage.py` | + `--oof_member NAME=CSV` (per-member OOF parity from the worker npz) + fleet-solo vs mean of member OOFs |
| `analysis/gold58_k3.py` | gold58 macro AUC per member / mean, Spearman vs `raptor_coat_side_rank.csv`, OOS-half lift (info) |
| `tests/smoke_k2.py`, `tests/run_scenario.py`, `tests/smoke_transcript.txt` | laptop CPU smoke (K2 + K3 scenarios) |
| `ckpt/` | M1 + the current M2; `_real/store/` the 4 parity npz (same 4 decode-probe studies for M1 and M2) |
| `build/` | the built upload folder + 2 kernels + `BUILD_SUMMARY.json` (current M2 = `..._cv8763_best.pt`, epoch 21, sha256 `e62c9e12...`; notebooks sha256 `baf4494d...` main, `5d416801...` ttrsolo after the 22:45 review fix, sec 9.9; the pre-fix pair `be5c0209...` / `d2232758...` is kept in `_review_prev_build/`) |

## 2. Final M2 swap (after the TTR run ends, ~22:55 IST)

The trainer prints its closing `{"raw": <best raw cv>, "ema": <best ema cv>}` line only after the LAST save. Until then
the best files can still be replaced (save() deletes the old best before writing the new one).

**Tag rule** (K2 RUNBOOK sec 3.1, one tag per run): ship the tag whose best checkpoint has the higher val cv (exact,
read from the checkpoint; must equal the closing line); a tie goes to EMA. At 22:00 the provisional pick was EMA
(ema .876335 at epoch 21 vs raw .875900 at epoch 15; epochs 1-based as in `build_k2.py`, the log's `ep N` lines are
0-based, so log `ep 20 ema` = epoch 21). The last epoch is log `ep 23`; the closing line follows its `ep 23 ema` line.

**NAME COLLISION (seen tonight):** the file name truncates cv to 4 digits, so a later epoch that improves cv in the
5th digit is saved under the SAME name. At 21:55 log `ep 21 ema` (.8764, epoch 22) replaced the shipped epoch-21
`..._cv8763_best.pt` (sha256 `e62c9e12...`) with a different file of the same name, and at 22:24 log `ep 22 ema`
(epoch 23) did it again. Never decide by the name: compare
sha256 (and the epoch printed by the pick).

```bash
# Git Bash on the laptop
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
B=training@192.168.10.30; PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
# 1. pick + snapshot on the box (exit 0 = final + copied; 3 = not final yet, re-run in 5 min; 1 = a check failed: STOP)
ssh -o BatchMode=yes $B 'tail -n 2 ~/fleet_runs/f0_s1042_ttr.log; CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/kmimic/bin/python -W ignore ~/k3_test/k3/box/k3_pick_ttr.py --copy; echo EXIT $?'
#    -> "copied ... (sha256 <16 hex>)" and "SHIP f0_s1042_<tag>_ttr224-320x24_cv<NNNN>_best.pt epoch <E> cv <..> tag <tag>"  (= M2)
# 2. same bytes as the current build? (built tonight: e62c9e12b96f03db = epoch 21). Equal -> skip steps 3-4 and the
#    re-test below and go straight to section 3 (dataset version): SP/k3/build is already final.
ssh -o BatchMode=yes $B 'sha256sum ~/k3_test/ckpt/<M2>'; sha256sum "$SP"/k3/ckpt/f0_s1042_*_best.pt
# 3. different -> move the old M2 out of k3/ckpt (build_k3 refuses two TTR files), fetch the new one, check the bytes
mkdir -p "$SP/k3/ckpt_old" && mv "$SP"/k3/ckpt/f0_s1042_*_best.pt "$SP/k3/ckpt_old/"
scp -o BatchMode=yes $B:k3_test/ckpt/<M2> "$SP/k3/ckpt/"
ssh -o BatchMode=yes $B 'sha256sum ~/k3_test/ckpt/<M2>'; sha256sum "$SP/k3/ckpt/<M2>"           # must match
# 4. rebuild (about 1 min, 450 MB of copies)
$PY -W ignore "$SP/k3/build_k3.py" --overwrite
#    gate: last line "K3 BUILD OK: upload ... (225.3 MB, 13 files) | M1 f0_s42_raw_cnxt320x24_cv8810_best.pt | M2 <M2> | kernels ..."
#          the "member" line of M2: fleet_model_ttr, sha f141aeed136dfc01, the SHIP epoch and cv;
#          build/upload/rsna-knee-cnxt-fleet/fleet_manifest.json "sources" sha256 of M2 == the first 16 hex of step 1
```

**Re-test the new M2 on the box (recommended, ~5 min, CPU, does not touch the GPU):**

```bash
U="$SP/k3/build/upload/rsna-knee-cnxt-fleet"
ssh -o BatchMode=yes $B 'rm -rf ~/k3_test/ds/rsna-knee-cnxt-fleet ~/k3_test/kernel && mkdir -p ~/k3_test/ds/rsna-knee-cnxt-fleet ~/k3_test/kernel'
(cd "$U" && scp -q -o BatchMode=yes fleet_*.py fleet_manifest.json dataset-metadata.json parity_*.npz $B:k3_test/ds/rsna-knee-cnxt-fleet/)
scp -q -o BatchMode=yes "$SP"/k3/build/kernel/*/*.ipynb $B:k3_test/kernel/
ssh -o BatchMode=yes $B 'cp -p ~/fleet_runs/f0_s42/f0_s42_raw_cnxt320x24_cv8810_best.pt ~/k3_test/ckpt/<M2> ~/k3_test/ds/rsna-knee-cnxt-fleet/ && cd ~/k3_test/ds/rsna-knee-cnxt-fleet && sha256sum * | LC_ALL=C sort -k2' > /tmp/k3_box_sha.txt
(cd "$U" && sha256sum * | sed 's/ \*/  /' | LC_ALL=C sort -k2) > /tmp/k3_lap_sha.txt; diff /tmp/k3_box_sha.txt /tmp/k3_lap_sha.txt && echo "DS IDENTICAL"
ssh -o BatchMode=yes $B 'bash ~/k3_test/k3/box/k3_box_launch.sh a0_main; bash ~/k3_test/k3/box/k3_box_launch.sh a0_ttrsolo'
# ~4 min later:
ssh -o BatchMode=yes $B 'cd ~/k3_test/runs && for f in $(ls -t A0_main_*.log | head -1) $(ls -t A0_ttrsolo_*.log | head -1); do echo == $f; grep -E "^FAIL|BOXSTAGE" $f | cut -c1-300; done'
```

Pass = both `BOXSTAGE PASS n/n` (K3-main 26/26, K3-ttrsolo 25/25 tonight), member OOF lines `< 2e-2` (tonight 5.3e-4 /
3.7e-4). The launcher compares M2 with `~/k3_test/oof/oof_f0_s1042_<tag>_cv<NNNN>.csv`, the OOF `k3_pick_ttr.py
--copy` took together with the checkpoint (same mtime).

## 3. Dataset version (operator)

```powershell
# PowerShell. Windows kaggle CLI staging-dir bug: pre-create %TEMP%\.kaggle\uploads\<path with ':' -> '_'> first.
$UP = "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k3\build\upload\rsna-knee-cnxt-fleet"
New-Item -ItemType Directory -Force -Path (Join-Path "$env:TEMP\.kaggle\uploads" ($UP -replace ':', '_')) | Out-Null
(Get-ChildItem $UP).Count                                  # 13
kaggle datasets version -p $UP -m "K3: M1 f0_s42 raw cv8810 + M2 TTR f0_s1042 <tag> cv<NNNN> + bigjump worker + fleet_model_ttr.py"
```

- If the CLI still prints `No such file or directory` for a path under `.kaggle\uploads`, create exactly that folder
  and re-run. 225 MB: a few minutes. An expired token fails silently: check the file list below.
- Wait for ready, then check the file set of the NEW version (13 files, both `.pt`, `fleet_model_ttr.py`):

```powershell
kaggle datasets status eeshsaxena/rsna-knee-cnxt-fleet          # repeat until: ready
kaggle datasets files eeshsaxena/rsna-knee-cnxt-fleet           # 13 files incl. f0_s1042_..._best.pt and fleet_model_ttr.py
```

Do not push a kernel before `ready` and the 13-file list: a commit that mounts the previous version (M1 only, K2
worker) logs `MEMBERS [...] not in the fleet manifest` and is NOT CLEAN (strict and solo: no `submission.csv`).

## 4. Push both kernels (operator)

```powershell
kaggle kernels push -p "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k3\build\kernel\rsna-sub-k3-fleet"
kaggle kernels push -p "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k3\build\kernel\rsna-sub-k3-ttrsolo"
```

Write down `V` for each from `Kernel version V successfully pushed` (new slugs: V = 1 unless re-pushed). Commit run:
parent on the 3 public test studies (~4 min) + fleet stage (~1-3 min) + queue: ~20-40 min.

```powershell
kaggle kernels status eeshsaxena/rsna-sub-k3-fleet              # until complete
kaggle kernels status eeshsaxena/rsna-sub-k3-ttrsolo
kaggle kernels output eeshsaxena/rsna-sub-k3-fleet   -p "$env:TEMP\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k3\build\out_main_v<V>"
kaggle kernels output eeshsaxena/rsna-sub-k3-ttrsolo -p "$env:TEMP\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k3\build\out_ttrsolo_v<V>"
```

## 5. Gate each commit (k2_check, per version)

```bash
# Git Bash, SP / PY as in section 2
M1=f0_s42_raw_cnxt320x24_cv8810_best.pt; M2=<M2>; MAN="$SP/k3/build/upload/rsna-knee-cnxt-fleet/fleet_manifest.json"
$PY "$SP/k3/k2_check.py" --work "$SP/k3/build/out_main_v<V>"    --expect_fleet clean --kaggle_commit --expect_mode strict --expect_members $M1,$M2 --expect_manifest "$MAN"
$PY "$SP/k3/k2_check.py" --work "$SP/k3/build/out_ttrsolo_v<V>" --expect_fleet clean --kaggle_commit --expect_mode solo   --expect_members $M2     --expect_manifest "$MAN"
```

Both must end `K2CHECK PASS`. What the checks cover: the attached manifest's names AND per-member sha256[:16] == the
local build (proves the commit used these bytes, not an older same-name file), `submission.csv` == its twin (main:
`submission_w25.csv`; solo: `submission_fleetsolo.csv`), each wXX == an independent recomputation, member names and
count == the filter, 2 GPUs,
decode probe match >= 1 and mismatch 0 per worker, shard-0 PARITY re-encode frac_gt8 <= .01 and no mismatch, one
microbench per GPU, n_fail = n_blank = 0, schema / row order / finite / [0, 1].

### What the commit log must show (`out_*_v<V>/rsna-sub-k3-*.log`)

K3-main (`rsna-sub-k3-fleet`):

```
[k2-fleet ..] STAGE fleet start; fail-loud files ['submission_fleetsolo.csv', 'submission_w25.csv', 'submission_w20.csv', 'submission_w40.csv']; STRICT: submission.csv = P+F at W 0.25 (DELETED on any failure); members ['f0_s42_raw_cnxt320x24_cv8810_best.pt', '<M2>']
[k2-fleet ..] manifest sha256[:16] {'f0_s42_raw_cnxt320x24_cv8810_best.pt': 'dc7e1495404e4e7a', '<M2>': '<first 16 hex of M2 sha256>'}
[k2-fleet ..] member filter: running ['f0_s42_raw_cnxt320x24_cv8810_best.pt', '<M2>'] of the manifest ['f0_s42_raw_cnxt320x24_cv8810_best.pt', '<M2>']
[fleet 0] members: f0_s42_raw_cnxt320x24_cv8810_best.pt[KneeFleetNet], <M2>[KneeFleetNetTTR]          (same for [fleet 1])
[fleet 0] PARITY re-encode vs stored train bytes: {... 'frac_gt8': <= 0.01, ... 'mismatch': []}
[fleet 0] decode probe vs ckpt (train-side decode): match 8 mismatch 0                                (same for [fleet 1])
[fleet g] MICROBENCH 5 forwards/member on cuda: ms/member/study {M1: .., M2: ..}                        (one per GPU)
[k2-fleet ..] RESULT OK fleet stage clean: 2 member(s) ['f0_s42_raw_cnxt320x24_cv8810_best.pt', '<M2>'], 3 studies, 0 median-filled, ...
[k2-fleet] submission.csv / submission_fleetsolo.csv / submission_w25.csv / submission_w20.csv / submission_w40.csv   rows=3 nan=0   (none ABSENT)
```

K3-ttrsolo (`rsna-sub-k3-ttrsolo`): `STAGE fleet start; ... SOLO: submission.csv = the fleet alone (DELETED on any
failure); members ['<M2>']`, the same `manifest sha256[:16]` line, `member filter: running ['<M2>'] of the manifest
[M1, M2]`, `[fleet g] members: <M2>[KneeFleetNetTTR]`, `decode probe ... match 4 mismatch 0`, `RESULT OK fleet stage
clean: 1 member(s) ['<M2>']`, no ABSENT file.

The `[fleet g]` lines come from the worker processes; if a Kaggle log does not show them, `k2_check.py
--kaggle_commit` reads the same facts from `fleet_receipt.json` (also echoed in the `[k2-fleet] receipt` line).
With 3 studies the wXX files print `(byte-identical to parent)`: expected (3-row ranks coincide).

Stop signs: `RESULT FLEET STAGE NOT CLEAN (STRICT|SOLO: submission.csv DELETED ...)` or any ABSENT `submission.csv` = do
not submit that version. Reasons on that line: `MEMBERS [...] not in the fleet manifest` = the old dataset version
was mounted (re-push after `ready` = version V+1), `fleet dataset not attached`, `worker rc [1, ..]` (traceback above
it: contract / model_sha / timm / ttr_size), `rc [4, ..]` = PARITY FAIL (Kaggle decode differs: stop).

## 6. Submit (operator; 2 slots; the UTC day ends 05:30 IST)

```powershell
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k3-fleet   -v <V> -f submission.csv -m "K3 strict P+F W.25: M1 f0_s42 raw cv8810 + M2 TTR f0_s1042 <tag> ep<E> cv<NNNN>"
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k3-ttrsolo -v <V> -f submission.csv -m "K3 TTR solo f0_s1042 <tag> ep<E> cv<NNNN>"
```

`V` = the version whose output passed K2CHECK (never a version whose log shows NOT CLEAN). Reads ~6 h later.

## 7. Hidden-rerun timing (2 x T4, 9 h)

Same wall logic as K2 (8.6 h wall, stage starts with >= 20 min left). Fleet stage with 2 members stays decode-bound:
decode ~15-20 min (0.68 s/study measured), GPU per T4 ~661 studies x (M1 ~0.30 s + M2 at 224 px ~0.15 s ESTIMATE) =
~5 min; stage ~17-22 min central, ~32 min slow, so K3 ends by ~6.4-6.6 h at the measured ~6 h parent. A stage that
does not finish = no `submission.csv` (strict / solo) = an error, never a silent parent score. GPU memory ~2.5-4 GB per
T4 (two ConvNeXt-T, ESTIMATE).

## 8. Evidence (10-04 evening)

All with M2 = the epoch-21 EMA file (sha256 `e62c9e12...`, val cv .876335). Transcripts in `SP/k3/box/transcripts/`
and `SP/k3/tests/smoke_transcript.txt`.

- **Worker / sources.** `k3/code/fleet_worker.py` and `fleet_model_ttr.py` are byte copies of the box
  `~/bigjump_build/code` files; the worker diff vs the K2 kit worker is ONLY the W-TTR lines (4 hunks: header comment,
  `_model_sha` TTR branch, `model_mod` whitelist, `KneeFleetNetTTR` build + `ttr_size_t` assert). The worker's own
  `_model_sha` on `k3/code`: family 1 `fa35880fca963652`, TTR `f141aeed136dfc01` (family 2 `ba3e1f85e38128bf`).
- **Laptop CPU smoke** (`tests/smoke_k2.py`, final run 22:17-22:26: `K3 SMOKE PASS in 8.3 min`, 38/38 scenarios): worker parity K3 (bigjump) vs fleet2 AND vs the
  K2 kit worker on family 1: P max|diff| 0.0; a tiny real TTR member (KneeFleetNetTTR, model_sha f141aeed136dfc01) served
  next to family 1 (family-1 rows unchanged, 0.0); all 24 K2 scenarios (22 + solo ok / fail) still pass; 14 new K3 scenarios pass (strict ok /
  worker fail / write fail on submission.csv / low time / no dataset / exception outside the guarded path, solo and
  blend outer handler, strict with the real TTR pair, no-filter TTR pair, TTR solo via the filter, stub filter ok /
  mismatch / unknown name); cross-run: the filter and the extra family change no prediction (max|diff| 0.0); real
  `build_k2.py` on M1 + M2 (strict kernel + `--kernel_only` TTR-solo kernel) and 5 refusals (same fold/seed from another
  run, two tags of one run, unknown `--members`, `--kernel_only` unknown name, `--kernel_only` existing slug).
- **Box A0 (CPU, Kaggle cv2 4.14, real DICOMs, 2 fold-0 val studies, final notebooks):** K3-main `BOXSTAGE PASS 26/26`,
  K3-ttrsolo `BOXSTAGE PASS 25/25`. Train/serve parity: M1 vs `~/fleet_runs/f0_s42/oof_f0_s42_raw.csv` max|diff| 5.3e-4;
  M2 vs its OOF (`oof_f0_s1042_ema.csv` of the same save, == `epochs/oof_f0_s1042_ema_ep21.csv`) 3.7e-4; fleet solo vs the
  mean of the two OOFs 3.7e-4 (all < 2e-2). OOF mtime == checkpoint mtime to the second for both members (M1
  07:03:30, M2 21:26:29; `mtimes_2147.txt`). The workers saw the filtered FLEET_DIR (K3-ttrsolo loaded only
  KneeFleetNetTTR); attached manifest sha256 == the local build.
- **Box old-dataset test:** both K3 notebooks against the K2 dataset (M1 only, K2 worker) = what a push before `ready`
  mounts: `RESULT FLEET STAGE NOT CLEAN (...: submission.csv DELETED ...): MEMBERS [M2] not in the fleet manifest`,
  `submission.csv ABSENT`, no worker started (`BOXSTAGE PASS 7/7` each).
- **Box gold58 (K3-main notebook on the 58 gold studies, CPU 8 threads, real parent preds):** `BOXSTAGE PASS 22/22`,
  stage 16.0 min on CPU, RESULT OK 2 members, rho(parent, fleet) .907. Reads (`analysis/gold58_k3.py`,
  `analysis/k3_members_boot.py`):

| Candidate | gold58 macro AUC | Spearman vs raptor_coat_side | lift into raptor_coat_side W .15 / .25 / .35 / .50 (info) |
|---|---|---|---|
| M1 f0_s42 raw (K2 member) | **.9144** (K2 measured .914) | .921 | -.0001 / -.0008 / -.0026 / -.0074 |
| M2 TTR f0_s1042 ema ep21 | **.8792** (trainer log ep 20 ema .8790) | .837 | -.0022 / -.0046 / -.0098 / -.0184 |
| mean(M1, M2) raw-prob (= K3 fleet solo) | **.9081** | .903 | -.0013 / -.0027 / -.0043 / -.0091 |
| mean(M1, M2) rank | .9047 | .902 | -.0023 / -.0025 / -.0066 / -.0097 |

raptor_coat_side itself = .9353 on gold58. rho(M1, M2) = .874 (gold58), .923 (val 867).
Paired bootstrap (1000 x, 90 % interval): gold58 M2 - M1 = -.035 [-.049, -.022]; mean - M1 = -.006 [-.015, +.001], P>0
.07. On the 867 fold-0 val studies vs steven_v4 (both runs' selection target; reproduces the trainers' cv exactly):
M1 .8811, M2 .8763, mean .8840; mean - M1 = +.003 [+.001, +.005], P>0 1.00.
Per label, M2 is weakest on Synovitis (.631 vs M1 .792) and Effusion (.878 vs .976) on gold58.
**gold58 is in-sample for the parent's DINO / RadImageNet arms, so the full P+F blend is not judged here.** Reading:
on gold labels M2 is clearly weaker than M1 and pulls the mean below M1 (the val-set gain of the mean is against the
v4 report labels M2 was trained on); every lift into the parent's out-of-sample half is negative, M1's own (-.0008 at
.25) being the one that read flat on the LB (K2 blend W .30 = .943).

## 9. Gaps / caveats

1. **CUDA / 2 x T4 untested for TTR until the commit.** fp16 TTR on a T4, two-GPU sharding of the 2-member fleet,
   the microbench. The CPU paths ran on real data; `k2_check.py --kaggle_commit` gates the commit.
2. **The final TTR file is not built yet.** Everything above used the epoch-21 EMA file. At 21:55 epoch 22 (.8764)
   replaced it under the SAME name; the run ends ~22:55. Section 2 swaps by sha256, never by name; `k3_pick_ttr.py`
   picked and copied correctly in a dry run on the real files (exit 0, sha256 of the copy == source).
3. **The box gold58 run used an earlier text of cell 31** (before the K3c manifest line and the solo/strict log-text
   edits); the clean path and the worker are unchanged, and the final notebooks were re-run on A0 (26/26, 25/25).
4. **Solo semantics changed in ONE place on purpose:** the outer last-defence handler (an exception escaping the guarded
   path) now DELETES `submission.csv` in solo mode as in strict (K2 restored the parent there, i.e. a silent 0.943 on a
   solo read). Blend is unchanged; the solo and strict log lines now say DELETED instead of `parent bytes`. Test:
   `solo_outer`, `strict_outer`, `blend_outer`.
5. **Parity re-encode is skipped on the box** (only 1 of the 4 parity studies has DICOMs there); it runs on Kaggle
   (shard 0, exit 4 on mismatch) and is a `--kaggle_commit` gate.
6. **Member filter = symlinks in a tempfile dir** on Kaggle (hard link / copy fallback only where symlinks are not
   allowed, e.g. Windows test hosts). Proven on Linux (box A0) and Windows (smoke).
7. **Timing for 2 members** is an ESTIMATE (decode-bound; TTR is cheaper per study than M1). A stage that misses the
   wall errors in both K3 kernels (no silent parent score).
8. `kaggle datasets version` REPLACES the file set: the new version has exactly the 13 files of the upload folder.
   Kernels that were already committed (K2) keep their pinned version; new pushes mount the latest.
9. **Review fix (22:45, cell 31 only, outer handler):** `_k2_entry`'s last-defence handler used to log
   (`traceback.print_exc()`, `_k2_log`) BEFORE deleting `submission.csv` in solo / strict, inside one try whose
   `except: pass` swallowed everything; a raising log (probe: MemoryError from print_exc) left the parent bytes in
   `submission.csv` = a silent 0.943 from a strict slot. Now the delete runs first (`dm = _k2_delete_main()`), then
   the logging; blend is unchanged (`dm` unused). Upload folder unchanged (same 13 files, same bytes); both notebooks
   rebuilt by `build_k3.py --overwrite`; the clean path is untouched, so the box A0 evidence still holds for it; the
   laptop smoke was re-run in full after the fix.

## 10. Wait for the last TTR epochs before using the 2 slots?

Facts, not a decision (the operator decides):
- The run ends ~22:55 IST (log `ep 23 ema` + closing line); the UTC day ends 05:30 IST. Swap + rebuild ~5 min, box A0
  ~5 min, dataset version + `ready` ~10-20 min, two commits in parallel ~20-40 min, checks ~5 min: submissions by
  ~00:30 IST after waiting, ~23:30 without. Either way 5+ h of margin; LB reads ~6 h after each submission.
- What waiting buys: the EMA has plateaued (val .8759 / .8762 / .8763 / .8764 / .8764 at epochs 19-23; gold58 .880 /
  .880 / .879 / .879 / .878), so the last epoch should move M2 by about +-.001 val; that is below anything a single LB read can
  resolve. Waiting costs nothing but ~1 h, and ships the finished member instead of a mid-run snapshot.
- What the gold58 read says about the slots themselves: M2 is weaker than M1 on gold labels (.879 vs .914) and the
  2-member mean sits below M1 (.908); every blend into the parent's out-of-sample half is negative. Expect K3-main to
  read at or slightly below K2's .943 and the TTR solo below K2 solo's .932. The TTR-solo slot is the informative one
  (first LB read of the TTR recipe the big-jump plan would scale); K3-main mostly re-measures K2 with a weaker leg.
