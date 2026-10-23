# FN11 RESULT: E2 JOINS; FN11 + FN11-N70 BUILT AND A0-TESTED (FINAL, 2026-10-09 22:35 IST)

Written 2026-10-09 by Claude (subagent). No em dashes. Nothing was created, versioned, pushed or submitted on Kaggle.
`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`,
`K11` = `SP/k11` (the kit, sibling of `SP/k10`; a no-binary copy is in `session_2026-10-09/FN11/kit/`).

**STATUS FINAL (22:35 IST): gate PASS (5-fold decision read, E2 JOINS); FINAL builds done; laptop TK11BUILD 27/27;
box A0 on the FINAL 20-member build ALL PASS (FN11 + FN11-N70 chains KFNBOX 99/99 with FNCHECK 90/90 and weights 4/4;
3 negatives PASS). Ready for dataset create + push (sec 5).**

## 1. Gate (PREREG_FN11.md, sha256 5d19d441... as written 16:50, before any E2 fold was final)

| read | when | G1 dOOF (bar >= 0) | G2 d58 (bar >= -.0005) | info d74 | verdict |
|---|---|---|---|---|---|
| E2-4 PROVISIONAL (f0-f3, 19 members) | 18:40 | +.00046 PASS (B0 .89037, B1 .89083) | -.0001 PASS (SE .0010) | +.0003 | E2 JOINS (info) |
| **DECISION, E2 5/5 (20 members)** | 20:41 | **+.00049 PASS** (B0 .89121, B1 .89171, 4,333 rows) | **-.0001 PASS** (B0 .9220, B1 .9219, SE .0013) | -.0002 (SE .0009) | **E2 JOINS** |

- B0 = C + D + E (== FN10-3, 15 members), B1 = + E2 (TTR288 frac75 s1046, Modal L40S). Fleet = equal member mean (as the
  K5 worker); G1 = fold-matched OOF macro AUC vs llm_labels_v4_blend (non-gold rows; selection-biased equally for every set);
  G2 = gold58 pass A CPU fp32 predictions. Gate script `k11/box/k11_gate.py` sha256 8bfa6d70... (recorded in
  `GATE_SCRIPTS_pre_E2.sha256` before E2 landed). Logs: `logs/k11_gate_20261009_{1756,1840,2041}.txt`.
- Per-fold acceptance (bar E2_F >= E_F - .005, all picker checks + K11-MREC Modal receipt): all 5 ACCEPTED: f0 .889364,
  f1 .891364, f2 .893418, f3 .880323, f4 .894199 (E2 - E: +.0000 / -.0008 / +.0017 / +.0015 / +.0028).
- Info: E2 alone gold58 .9199 vs E .9163 (+.0035, SE .0039); Spearman E2~E .966 (gold74) / .965 (OOF); Fracture d58 -.0028.
  Honest size: the fleet change is invisible on gold58 (-.0001, SE .0013) and +.0005 on (biased) OOF; E2 is a second seed of
  the weakest-on-gold set E, so the expected LB effect of FN11 vs FN10-3 is ~0 to +.0003 (F is only .40 of a rank blend).
- No gate on the F .40 / N .60 blend: N is in-sample on train and gold rows (trained on all RSNA training studies, selected
  on gold58; only 12 gold N predictions exist), so the fleet-only G1 + G2 decide (PREREG sec 3).
- Disclosure: before writing the prereg I saw status.sh epoch lines of E2 f0-f3 at epochs 9-11 (and later 13-14) of 21; no
  final checkpoint, OOF or gold prediction existed then.

## 2. Hidden-rerun runtime (PREREG sec 5 rule: slow end <= 0.65 x 9 h = 5.85 h)

`K11/tests/runtime_k11.py` (= runtime_k10.py inputs: MEASURED T4 microbench of the K5 / K7i commits, TTR224 300 / 426 ms,
TTR288 466 / 532 ms per member per study, 661 studies per T4; N 25 / 40 min):

| kernel | central | slow | unused of 9 h (slow) |
|---|---|---|---|
| FN (K9, scored 10-09) | 1.45 h | 2.00 h | 78 % (MEASURED: submit 11:22 -> COMPLETE by 12:34, <= 1.2 h) |
| FN10-3 (15 members) | 1.88 h | 2.54 h | 72 % |
| **FN11 (20 members)** | **2.31 h** | **3.08 h** | **66 % -> PASS** (5 more TTR288 members = +32 min slow) |
| FN11-D2 (25 members, optional) | 2.58 h | 3.47 h | 61 % -> PASS |

Cross-check: the FN commit MICROBENCH (swapiy v1) had TTR288 474-488 ms/member/study per T4, inside the model. No member
drop needed. Memory: 20 x 107 MB weights; the worker serves members in turn.

## 3. What was built (all under K11; FINAL = no DO_NOT_* marker)

| item | path | key facts |
|---|---|---|
| dataset folder (NEW slug) | `K11/build_w11/upload/rsna-knee-cnxt-fleet-k11w` | id `eeshsaxena/rsna-knee-cnxt-fleet-k11w`, 31 files, 2,162 MB: 5 code + 20 members + fleet_manifest.json (892631263bddeb0c...) + dataset-metadata.json + 4 parity npz. The 24 non-manifest k10w files (code, C/D/E members, npz) are byte-identical to the live k10w dataset; manifest = k10w manifest + 5 E2 entries. Full sha256 list: `FN11/k11w_upload_sha256.txt` (== the box A0 copy, "DS IDENTICAL") |
| FN11 kernel | `K11/build_fn11/kernel/rsna-sub-k11fn` | notebook sha256 b19a4320c340fc11..., datasets [k11w, nartaa N weights], TAG FN11, P/F/N 0/.40/.60 |
| FN11-N70 kernel | `K11/build_fn11_n70/kernel/rsna-sub-k11fn-n70` | notebook a2f03c4a1884709e..., 0/.30/.70; wdiff vs FN11: only md weights + K9N_CFG W3 / W3_VARIANTS + id/title/code_file |
| byte diff vs the TESTED FN10-3 twin | `K11/build_fn11/NB_DIFF_vs_FN10-3.txt`, `K11/build_fn11_n70/NB_DIFF_vs_FN10-3-N70.txt` | k11_nbdiff OK: 14 text lines: md title + fleet bullet; K2_CFG MEMBERS (+5 E2, the 15 FN10-3 members first, in order) and TTA (+5 AB); fleet cell dataset slug k10w -> k11w (3 strings, both reverse to the K5 cell); K9N_CFG TAG FN10-3 -> FN11; metadata id / title / code_file / dataset_sources (fleet slug only). FN-IDS cell, N cell body, embedded child (8d567a63...), weights sha, every other K2_CFG / K9N_CFG key: identical |
| diff vs the A0-tested 19-member TEST notebook | `K11/build_fn11/NB_DIFF_vs_A0_i19.txt` | only E2 f4 added (MEMBERS / TTA + the md member count) |

Kit (generated by `make_k11_tools.py` with counted substitutions from the tested k10 files; `--check` passes):
`build_k11.py` (sets E2 / D2, kits k11w / k11w2), `build_k11fn.py` (kits fn11 / fn11d2), `fn11_check.py`, `fnw_check.py`
(K11 slugs), `box/k5_pick.py` (+ K11-MREC Modal receipts), `cells/k11w*_fleet_cell.py`. Hand-written: `k11_collect.sh`,
`box/k11_merge.py`, `box/k11_gate.py`, `box/k11_gate_bg.sh`, `k11_nbdiff.py`, `k11_stage_box.sh`, `box/k11_box_launch.sh`,
`tests/t_k11_build.py`, `tests/runtime_k11.py`, `box/t_k11_pick.py`. D2 is parametric (`--kit k11w2` / `--kit fn11d2`,
NEW slugs -k11w2 / rsna-sub-k11fn-d2); its gate = `k11_gate.py --with_d2` (PREREG sec 6). k11_nbdiff assumes the twin's
members lead, so a D2 build needs its own diff rule (D2 sorts between D and C).

## 4. Tests

| test | result |
|---|---|
| `box/t_k11_pick.py 0` (K11-MREC on the real f0) | **TK11PICK PASS 9/9**: real receipt picks; no receipt / verify != OK / wrong closing line / file not listed / args not OK / wrong run / epoch_oof 31 each FAIL naming the cause |
| `tests/t_k11_build.py` on the FINAL builds | **TK11BUILD PASS 27/27** (tools --check, k11w cell == K5 cell, upload layout + k10w bytes, FN11 / N70 records, provenance == FN10-3, k11_nbdiff both twins, --compare rebuild byte-identical, 6 refusals, gate slugs, runtime rule) |
| same on the 19-member TEST builds | TK11BUILD PASS 27/27 |
| laptop simulation (E2 = relabelled E weights, before E2 landed) | builds OK; caught + fixed one k11_nbdiff title-offset bug |
| box A0, 19-member TEST build (`~/k11_test/a0i19`, CPUs 34-39, 18:41-20:29) | **fn11_chain KFNBOX PASS 100/100**, **fn11n70_chain KFNBOX PASS 100/100** (real 19-member fleet + N on 3 gold studies: RESULT OK fleet stage clean: 19 members, decode probe 76/76, N clean 0 fallback, fnw_check W1-W4 4/4: submission.csv == an independent pandas rank blend at the pinned weights, the 3 studies tell .40/.60 from .30/.70); fn_check 91/92 = the intended "TEST build: never push" miss. Negatives: fleet_missing KFNBOX PASS 35/35, N70 missing_weights PASS 21/21, **stale dataset (k10w bytes under the k11w slug) K11NEGX PASS**: "MEMBERS not in the manifest", no submission.csv |
| commit-gate negative (fn11_check on the TEST chain output) | with the 15 FN10-3 members as --expect_members: member-filter + worker checks FAIL (gate catches a stale member list) |
| **box A0, FINAL 20-member build** (`~/k11_test/a0`, CPUs 34-39, 20:43-22:33) | **fn11_chain KFNBOX PASS 99/99** (FNCHECK 90/90, FNWCHECK weights 4/4: RESULT OK fleet stage clean: 20 members; N clean, 0 fallback; submission.csv == the independent pandas rank blend at .40/.60; the 3 studies tell it from .30/.70), **fn11n70_chain KFNBOX PASS 99/99** (same at .30/.70); fleet_missing KFNBOX PASS 34/34; N70 missing_weights PASS 21/21; stale k10w dataset under the k11w slug K11NEGX PASS. Logs: box `~/k11_test/a0/runs/*_2043.log` |

## 5. Commands for the main session (Eesh runs every kaggle command)

Dataset CREATE (new slug; ~2.2 GB upload; never `version` an existing dataset, never touch eeshsaxena/rsna-knee-cnxt-fleet):
```powershell
$K11 = "C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad\k11"
Get-ChildItem "$K11\build_w11", "$K11\build_fn11", "$K11\build_fn11_n70" -Filter "DO_NOT_*"     # must list nothing
$UP = "$K11\build_w11\upload\rsna-knee-cnxt-fleet-k11w"
(Get-ChildItem $UP).Count                                                                     # 31
Get-Content "$UP\dataset-metadata.json"                                                       # id eeshsaxena/rsna-knee-cnxt-fleet-k11w
New-Item -ItemType Directory -Force -Path (Join-Path "$env:TEMP\.kaggle\uploads" ($UP -replace ':', '_')) | Out-Null   # Windows CLI staging bug
cmd /c "kaggle datasets create -p `"$UP`" > `"$K11\logs\ds_create_k11w.log`" 2>&1"           # private by default
Get-Content "$K11\logs\ds_create_k11w.log"
kaggle datasets status eeshsaxena/rsna-knee-cnxt-fleet-k11w                                    # until: ready
kaggle datasets files  eeshsaxena/rsna-knee-cnxt-fleet-k11w                                    # 31 files incl. f{0..4}_s1046t288f75_raw_ttr288-...
```
Push (any time after the dataset is ready; never --timeout on a submission kernel; quota ~0.15-0.3 h per commit):
```powershell
kaggle kernels push -p "$K11\build_fn11\kernel\rsna-sub-k11fn"
kaggle kernels push -p "$K11\build_fn11_n70\kernel\rsna-sub-k11fn-n70"
kaggle kernels status eeshsaxena/rsna-sub-k11fn                                               # until complete; note version V
kaggle kernels output eeshsaxena/rsna-sub-k11fn     -p "$K11\build_fn11\out_v<V>"     --page-size 200
kaggle kernels output eeshsaxena/rsna-sub-k11fn-n70 -p "$K11\build_fn11_n70\out_v<V>" --page-size 200
```
Commit gate (Git Bash; each must end PASS):
```bash
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
K11=$SP/k11; BW=$K11/build_w11
MEMW=$($PY -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['members']))" "$BW/K5_BUILD.json" | tr -d '[:cntrl:]')
MANW=$BW/upload/rsna-knee-cnxt-fleet-k11w/fleet_manifest.json
$PY $K11/fnw_check.py --work $K11/build_fn11/out_v<V> --fn_build $K11/build_fn11/FN_BUILD.json --kernel fn --expect_w3 0.40,0.60 --expect_n clean --expect_fleet clean --kaggle_commit --expect_members "$MEMW" --expect_manifest "$MANW" --expect_tta "$BW/K5_BUILD.json" --expect_n_ranks log949      # FNWCHECK PASS
$PY $K11/fn11_check.py --work $K11/build_fn11/out_v<V> --fn_build $K11/build_fn11/FN_BUILD.json --kernel fn --expect_n clean --expect_fleet clean --kaggle_commit --expect_members "$MEMW" --expect_manifest "$MANW" --expect_tta "$BW/K5_BUILD.json" --expect_n_ranks log949                         # FNCHECK PASS
$PY $K11/fnw_check.py --work $K11/build_fn11_n70/out_v<V> --fn_build $K11/build_fn11_n70/FN_BUILD.json --kernel fn --expect_w3 0.30,0.70 --expect_n clean --expect_fleet clean --kaggle_commit --expect_members "$MEMW" --expect_manifest "$MANW" --expect_tta "$BW/K5_BUILD.json" --expect_n_ranks log949  # FNWCHECK PASS
```
- In the commit log: `RESULT OK fleet stage clean: 20 member(s)` with 5 x `f{F}_s1046t288f75_raw_ttr288-...`, TTA AB,
  MICROBENCH ~466-532 ms per 288 member and ~283-426 per 224 member, two `[weights] sha256 7e5315da...` lines,
  `RESULT OK N stage clean`. If only the log949 rank check fails by < 2e-3: fp16 noise (accept by hand, KFN_RUNBOOK 2.3).

Submit (only after the gate PASS):
```powershell
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k11fn -v <V> -f submission.csv -m "FN11 parent-free P/F/N 0/.40/.60: C + D + E + E2 (20 members: 288 mix50 s1044, 224 frac75 s1043, 288 frac75 s1045 + s1046) + public .949 CoAtNet (nartaa weights)"
kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-k11fn-n70 -v <V> -f submission.csv -m "FN11-N70 parent-free P/F/N 0/.30/.70: C + D + E + E2 (20 members) + public .949 CoAtNet (nartaa weights)"
```
- Read rule suggestion (not pre-registered by me; the main session decides): FN11 vs FN10-3 is a pure fleet-composition read
  (same N, same weights); FN11-N70 vs FN10-3-N70 likewise. Expect the same 3-dp display.

## 6. Files

- `FN11/PREREG_FN11.md` (+ RESULTS appended), `PREREG_FN11.as_written.md` (prefix, sha256 5d19d441...), `PREREG_FN11.sha256`,
  `GATE_SCRIPTS_pre_E2.sha256`, `k11w_upload_sha256.txt`, `logs/k11_gate_*.txt`, `kit/` (scripts, no binaries).
- Laptop logs: `K11/logs/` (collects, builds, t_k11_build, t_k11_pick, stages). Box: `~/k11_test/` (members.json, ckpt =
  E2 picks, oof, gold/preds + sidecars, a0i19 + a0 A0 areas; hard links only, nothing outside ~/k11_test written).
- A0 note: the strict fleet cell leaves its k2_members_* symlink dir in TMPDIR when the stage fails (negatives) and the
  N child leaves 2 small temp dirs; harmless on Kaggle (ephemeral /tmp), same as the tested FN10-3 cell.
- Disk: laptop C: ~20 GB free; `K11/build_w11` holds the 2.2 GB upload folder (its members/ staging copy was deleted after
  the build; `build_w11_i19/upload` deleted, TEST only).
