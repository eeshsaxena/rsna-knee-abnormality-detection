# K12 RESULT: GATE FINAL 2026-10-10 19:36 = NO K12 (R and A both FAIL G2 gold58; FN11 stays). See PREREG_K12.md RESULTS.

Written 2026-10-09 by Claude (subagent). No em dashes. Nothing was created, versioned, pushed or submitted on Kaggle.
`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`,
`K12` = `SP/k12` (kit; a no-binary copy is in `session_2026-10-09/K12/kit/`).

**STATUS (01:40 IST 10-10): PREREG written (sha256 31884631...); stub (simulated D-OAI f1-f4) builds R + A pass the laptop
tests (TK12BUILD 23/23 each); box A0 on the stub R build ALL PASS. Real D-OAI f0 + f1 collected (OAI-clean PASS, both
accepted: f0 .887764, f1 .892041); f2 training (~07:25), f3 (~13:45), f4 (~20:05, MODEL). Gate after f4.**

## 1. Gate

(filled in after the decision read; see PREREG_K12.md RESULTS)

## 2. Runtime (hidden rerun; rule slow end <= 5.85 h)

| kernel | composition | central | slow | unused of 9 h |
|---|---|---|---|---|
| K12 R (REPLACE) | 5 x 224 + 15 x 288 + N (== FN11) | 2.31 h | 3.08 h | 66 % PASS |
| K12 A (ADD) | 10 x 224 + 15 x 288 + N | 2.58 h | 3.47 h | 61 % PASS |

Model: `k11/tests/runtime_k11.py` (MEASURED T4 microbench; FN hidden run MEASURED <= 1.2 h). The K12-MERGE cell adds
seconds (symlinks + sha256 of 5 x 107 MB).

## 3. Design

- Dataset `eeshsaxena/rsna-knee-cnxt-fleet-k12w` (NEW) = ONLY the 5 D-OAI raw best checkpoints, re-keyed
  `f{F}_s1043f75oai_raw_ttr224-320x24_cv<NNNN>_best.pt`, + `k12w_manifest.json` (sha256 full + [:16], model_mod /
  model_sha, tta AB) + dataset-metadata.json. No code, no OAI data / images / labels / ids. 538 MB.
- Every D-OAI checkpoint passes K12-OAI-CLEAN on the box BEFORE it leaves (`box/k12_oai_clean.py`: top-level keys
  model / args / cv / epoch only, tensors-only state dict, no `oai_<digit>` string anywhere, no table / id list / big array,
  decode_probe = RSNA UIDs; its OOF ids == set D's OOF ids), and again on the laptop at build time.
- Kernels `rsna-sub-k12fn` (F .40 / N .60) and `rsna-sub-k12fn-n70` attach `-k11w` (the LIVE FN11 dataset, unchanged) +
  `-k12w` + the N weights. A new K12-MERGE cell links both datasets into `/tmp/k12_fleet` (symlinks), writes the merged
  manifest embedded in the notebook, and fails loud (no merged dir -> the strict fleet stage refuses -> no submission.csv)
  if the k11w manifest or the k12w manifest differ from the build or any D-OAI sha256 differs.

## 4. Tests so far

| test | result |
|---|---|
| `box/t_k12_clean.py` (K12-OAI-CLEAN) | TK12CLEAN PASS 8/8: real D-OAI f0 + OOF pass; planted OAI id string, pandas table, extra key, OAI decode_probe key, id list, oai OOF row, OOF id-set mismatch each FAIL |
| `tests/t_k12_build.py` stub R / stub A | TK12BUILD PASS 23/23 each |
| box A0 stub R (`~/k12_test/a0sim`, 23:28-01:36) | **k12_chain KFNBOX PASS 100/100, k12n70_chain KFNBOX PASS 100/100** (K12-MERGE RESULT OK in 1.8 s: 20 base + 5 D-OAI; RESULT OK fleet stage clean: 20 members; N clean, 0 fallback; fnw_check weights 4/4; fn_check 91/92 = the intended TEST-build miss); fleet_missing KFNBOX 35/35; N70 missing_weights 21/21; **K12NEGX PASS** for k12w not attached, tampered D-OAI f1 (sha256), stale base (k10w under the k11w slug) |
| real interim collect 01:05 | D-OAI f0 + f1 picked, K12CLEAN 2/2, both ACCEPTED (bars .878777 / .885967) |

## 5. Commands for the main session (filled with the decided variant V = R or A after the gate; Eesh runs every kaggle command)

Prerequisite: the FN11 dataset `eeshsaxena/rsna-knee-cnxt-fleet-k11w` exists and is `ready` (FN11_RESULT.md sec 5); K12
reads it unchanged. Never version it while an FN11 or K12 commit / submission that reads it is pending.

Dataset CREATE (new slug, ~538 MB, only the 5 D-OAI checkpoints + manifest):
```powershell
$K12 = "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k12"
Get-ChildItem "$K12\build_w12", "$K12\build_fn12_<V>", "$K12\build_fn12_<V>_n70" -Filter "DO_NOT_*"   # must list nothing
$UP = "$K12\build_w12\upload\rsna-knee-cnxt-fleet-k12w"
(Get-ChildItem $UP).Count                                   # 7: 5 x f{F}_s1043f75oai_raw_ttr224-...pt + k12w_manifest.json + dataset-metadata.json
Get-Content "$UP\dataset-metadata.json"                     # id eeshsaxena/rsna-knee-cnxt-fleet-k12w
New-Item -ItemType Directory -Force -Path (Join-Path "$env:TEMP\.kaggle\uploads" ($UP -replace ':', '_')) | Out-Null
cmd /c "kaggle datasets create -p `"$UP`" > `"$K12\logs\ds_create_k12w.log`" 2>&1"     # private by default
kaggle datasets status eeshsaxena/rsna-knee-cnxt-fleet-k12w   # until: ready
kaggle datasets files  eeshsaxena/rsna-knee-cnxt-fleet-k12w   # the 5 D-OAI files + k12w_manifest.json
```
Push / output:
```powershell
kaggle kernels push -p "$K12\build_fn12_<V>\kernel\rsna-sub-k12fn"
kaggle kernels push -p "$K12\build_fn12_<V>_n70\kernel\rsna-sub-k12fn-n70"
kaggle kernels status eeshsaxena/rsna-sub-k12fn          # until complete; note version v
kaggle kernels output eeshsaxena/rsna-sub-k12fn     -p "$K12\build_fn12_<V>\out_v<v>"     --page-size 200
kaggle kernels output eeshsaxena/rsna-sub-k12fn-n70 -p "$K12\build_fn12_<V>_n70\out_v<v>" --page-size 200
```
Commit gate (Git Bash; each must end PASS):
```bash
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
K12=$SP/k12; BF=$K12/build_fn12_<V>; BF7=$K12/build_fn12_<V>_n70
MEM=$($PY -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['members']))" "$BF/K12_BUILD.json" | tr -d '[:cntrl:]')
MAN=$K12/build_w12/merged_manifest.json
$PY $K12/fnw_check.py --work $BF/out_v<v> --fn_build $BF/FN_BUILD.json --kernel fn --expect_w3 0.40,0.60 --expect_n clean --expect_fleet clean --kaggle_commit --expect_members "$MEM" --expect_manifest "$MAN" --expect_tta "$BF/K12_BUILD.json" --expect_n_ranks log949     # FNWCHECK PASS
$PY $K12/fn12_check.py --work $BF/out_v<v> --fn_build $BF/FN_BUILD.json --kernel fn --expect_n clean --expect_fleet clean --kaggle_commit --expect_members "$MEM" --expect_manifest "$MAN" --expect_tta "$BF/K12_BUILD.json" --expect_n_ranks log949                        # FNCHECK PASS
$PY $K12/fnw_check.py --work $BF7/out_v<v> --fn_build $BF7/FN_BUILD.json --kernel fn --expect_w3 0.30,0.70 --expect_n clean --expect_fleet clean --kaggle_commit --expect_members "$MEM" --expect_manifest "$MAN" --expect_tta "$BF/K12_BUILD.json" --expect_n_ranks log949  # FNWCHECK PASS
grep -h "k12-merge.*RESULT OK" $BF/out_v<v>/*.log      # the merge line: 20 base members + 5 D-OAI members, sha256 verified
```
Submit (only after the gate PASS):
```powershell
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k12fn -v <v> -f submission.csv -m "K12 parent-free P/F/N 0/.40/.60: fleet with the OAI-trained 224 set D-OAI (<V>) + public .949 CoAtNet (nartaa weights)"
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k12fn-n70 -v <v> -f submission.csv -m "K12-N70 parent-free P/F/N 0/.30/.70: fleet with D-OAI (<V>) + public .949 CoAtNet (nartaa weights)"
```
- K12 is OAI-trained in the fleet AND in N: N pick only, never the clean pick.
