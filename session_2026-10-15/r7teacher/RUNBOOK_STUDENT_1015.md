# RUNBOOK_STUDENT_1015: T_b student, from "5/5 folds present" to a submission-ready V224-FF-S kernel

Written 10-15 00:15 IST by the runbook agent (workflow job "runbook + dry runs"). No em dashes. Numbers tagged MEASURED / MODEL /
GUESS. Nothing in this file was submitted, pushed or created on Kaggle; the box queue and the box GPU were not touched; the rental
kit (`rental_student/`, owner a69fd02) was only read.

Sources read: PREREG_TEACHER.md (frozen text + A1 + A2, sha256 9e987880), PLAN_TEACHER.md sec 1, 2, 5, 7, PREREG_10-15.md,
PREREG_10-14.md (Amendments 1-3), r7teacher/RUN_LOG.md, FIXES_A1.md, rental_student/RUN_LOG.md + RUNBOOK.md, student_gate.py,
make_s_pack.py, build_v224fs.py, v224ffs_check.py, v224fs_check.py, wait_fetch.py, v224f/v224ff RESULT.md, the A1 dry-run logs.

**ROLES.** `[MAIN ONLY]` = only the main session may do it (Kaggle dataset create, kernel push / commit, submit, PREREG
amendments, any box queue change). Everything else is box CPU (nice 19) or laptop CPU and may be run by any agent.

**Golden rules for this chain.**
- The student S is exactly the 5 runs `~/fleet_runs/f{0..4}_s1047_ttr_r7tb` (f0-f3 from the rental, f4 from the box; same argv,
  PREREG A2). No 4-fold student, no ema checkpoints, no `epochs/` checkpoints, no `_rentpartial_` dirs.
- Raw-best checkpoint per fold = `f{F}_s1047_raw_ttr224-320x24_cv####_best.pt` (selected on the T_b table, A1 FX4). The ema file
  in the same dir is ignored by both the gate and the pack (MEASURED on f4, App. C F1).
- Never pass `--standin` or `--pred_dir` to the REAL gate (App. C F2). Never run `make_s_pack.py` without a REAL `VERDICT PASS` file.
- OAI: student checkpoints are RSNA-only (T_b = v4 + OAI-free fleet OOF + R7 OOF); they may leave the box only after the two scans
  of Steps 4 and 4b. Gate output text stays on the box except the summary lines. Step numbers below are the run order; Appendix A-D = T_a, dry-run record, findings, timeline.

Paths. Laptop root `/c/Users/eeshs/Downloads/rsna-knee` (Git Bash; `R=` below). Box `ssh -o BatchMode=yes training@192.168.10.30`,
`PY=~/miniconda3/envs/rsna/bin/python`, CPU prefix `CPU="taskset -c 34-39 nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES="`.

---

## Step 0. Before T0 (now -> ~02:50 IST), monitoring only

| # | where | command | expected |
|---|---|---|---|
| 0.1 | box | `tail -n 2 ~/fleet_runs/f4_s1047_ttr_r7tb.log` | f4 started 22:03:39, 13 epochs done by 00:13 (10.0 min/epoch MEASURED) -> last line `{"raw": 0.9..., "ema": 0.9...}` at **~01:35** (MODEL) |
| 0.2 | box | `bash ~/queue/queue_runner.sh --status` (read only) | after f4 closes the runner sees L118-L121 (T_b f0-f3 box fallback) with existing out dirs (the rental placeholders) -> `#REFUSED`, then **T_a f0 launches** (~01:37). The box GPU stays busy; nothing to do. Do NOT edit ~/queue. |
| 0.3 | box | `cd ~/r7t_labels && $CPU $PY -W ignore ~/r7t_labels/rb_dry/rb_fold_check.py --folds 4` (or the laptop copy `session_2026-10-15/r7teacher/rb_fold_check.py` scp'd to the box) | after 0.1: `RB_FOLD_CHECK PASS 10/10`, folds [4] |
| 0.4 (optional fast path) | box | only after 0.3 PASS: `cd ~/r7t_labels && $CPU NTHR=6 $PY -W ignore ~/novel/gold_infer5.py --passes A --out ~/r7t_labels/gold_preds ~/fleet_runs/f4_s1047_ttr_r7tb/f4_s1047_raw_ttr224-320x24_cv*_best.pt` | ~7.5 min (450 s/ckpt MEASURED); writes `~/r7t_labels/gold_preds/f4_s1047_ttr_r7tb__raw__A.csv`. This is the exact command the gate's `--infer` runs (same script, same `--out`, same name rule), so the gate later reuses it. Never run it on an OPEN run (the cache is keyed by run name, not by sha). |
| 0.5 | box | `tail -n 5 ~/rstud/fetch/fetch.log; cat ~/rstud/fetch/resolved_f* 2>/dev/null; ls ~/rstud/FINAL_FETCH_OK ~/fleet_runs/rstud_f*_s1047_ttr_r7tb_receipt.json` | rental lanes (2 x A100, 4 folds paired) end **~02:30** (smoke 10.0 min/epoch MEASURED -> 3.89 h/fold MODEL, started 22:36); fetch loop polls every 5 min; `f<F> FETCHED + VERIFIED + PLACED` x4, `resolved_f0..3 = ok`, `~/rstud/FINAL_FETCH_OK` **~02:45-02:55** (MODEL). Guard hard deadline 04:48 (partial pulls from 04:08). The rental agent owns the instance; read only here. |

**T0 = both: `~/rstud/FINAL_FETCH_OK` exists with resolved_f0..f3 = ok, and f4 has its closing line.**

### Branch 0-A: a rental fold did not arrive (resolved = failed / verify_failed / skipped, `bad_*` aside, or a `_rentpartial_` dir)
- The fetch loop releases that fold's EMPTY placeholder. `tb_fallback.sh` (box, pid in ~/rstud/tb_fallback.pid, 120 s loop) turns
  `BOX_TB_FALLBACK.verdict` to PASS once the rental ended and re-queues the missing fold with `qctl top` (it owns that; agents do not
  touch the queue). The box then trains it AFTER the running job (T_a f0, ~01:37-05:10 MODEL): one missing fold -> fold ready ~08:45,
  gate ~09:35, submission-ready ~11:30 (MODEL); two missing folds -> +3.5 h each.
- `[MAIN ONLY]` option to save ~3.5 h: stop T_a f0 through the queue runbook and let the T_b fold start now. Agents never kill.
- A `_rentpartial_` dir and anything under `~/rstud/fetch/bad_*` are NEVER student members.
- Before the gate, delete nothing; just re-run step 1 when the box fold closes. Freeze: no student that cannot have 5/5 folds before
  10-20 12:00 IST (PREREG sec 11).

### Branch 0-B: box f4 fails (FAILED in ~/fleet_runs/queue_runner.log)
- The runner re-runs an early CUDA OOM once with `--grad_ckpt 1` (out dir moved aside as `_fastfail_`). That re-run differs from the
  frozen argv token `--grad_ckpt 0` (numerically equivalent, G0 does not check it): `[MAIN ONLY]` decide whether it counts; record it
  in Amendment 4. Any other failure: the fold must be retrained under the same name (old dir moved aside, never deleted) and its
  cached gold prediction (if 0.4 ran) moved aside too.

---

## Step 1. Fold preflight (box, ~1 min)
```
cd ~/r7t_labels && $CPU OMP_NUM_THREADS=4 $PY -W ignore ~/r7t_labels/rb_dry/rb_fold_check.py
```
Expected: `RB_FOLD_CHECK PASS 50/50 | tpl f{F}_s1047_ttr_r7tb | folds [0, 1, 2, 3, 4]` (10 checks x 5 folds). If 0.4 ran,
f4's "no cached gold74 prediction" line FAILs by design: then expect `RB_FOLD_CHECK FAIL 49/50` with exactly that one FAIL line.
Checks per fold: no cached gold pred for the run name, closing line, exactly one raw-best ckpt, staged-name regex, OOF ids == folds5
val ids minus gold74, ckpt args (fold, seed 1047, 21 ep, ttr 224, fleet_model_ttr, model_sha f141aeed136dfc01, fp16,
train_labels == sel_labels == labels_r7t_b.csv), recipe keys == the k11w D member, decode_probe, OAI-clean shape; INFO line for the
rental receipt `~/fleet_runs/rstud_<run>_receipt.json` (folds 0-3).
Pass: PASS (allowing only the f4 cache line from 0.4). FAIL on any other line: STOP, do not start the 40 min inference; fix the cause
(Branch 0-A / 0-B) or report.

## Step 2. Gate preflight, no inference (box, ~10 s)
```
cd ~/r7t_labels && $CPU NTHR=6 $PY -W ignore student_gate.py --student 'f{F}_s1047_ttr_r7tb' \
   --table real/labels_r7t_b.csv --receipt real/teacher_labels_receipt.json --teacher real/R7OOF.csv
```
Expected (MEASURED on a mock with these exact files, App. B D4): header `... | REAL`, `G0 integrity: PASS`,
`MISSING gold predictions for 5 checkpoint(s): re-run with --infer` (4 if 0.4 ran), exit code 3. The D / D2 reference gold
predictions are cached (k9gold + r7t_labels/gold_preds, MEASURED), so only the student members are missing.
G0 FAIL -> STOP (the reason is printed: missing ckpt / OOF, ids, args, table md5 vs receipt, PREREG md5 self-check).

## Step 3. Student gate with inference (box, ~35-45 min; or ~20 min with the fast path)
Default (frozen path):
```
cd ~/r7t_labels && $CPU NTHR=6 $PY -W ignore student_gate.py --student 'f{F}_s1047_ttr_r7tb' \
   --table real/labels_r7t_b.csv --receipt real/teacher_labels_receipt.json --teacher real/R7OOF.csv --infer \
   2>&1 | tee ~/r7t_labels/gate/student_gate_REAL_r7tb_console.log
```
Fast path (optional, same outputs): run the 4 rental folds' gold inference as two parallel processes on disjoint cores, then the
default command (it finds all 5 cached and only computes the legs):
```
cd ~/r7t_labels && for P in "28-33 0 1" "34-39 2 3"; do set -- $P; \
  taskset -c $1 nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES= NTHR=6 $PY -W ignore ~/novel/gold_infer5.py --passes A --out ~/r7t_labels/gold_preds \
  ~/fleet_runs/f$2_s1047_ttr_r7tb/f$2_s1047_raw_*_best.pt ~/fleet_runs/f$3_s1047_ttr_r7tb/f$3_s1047_raw_*_best.pt > ~/r7t_labels/gate/infer_f$2f$3.log 2>&1 & done; wait
```
Time: inference 450 s per checkpoint (MEASURED, PREREG sec 10): 5 x 7.5 = 37.5 min sequential, ~15 min with the fast path (and f4
from 0.4); gate legs 5.5 min (MEASURED 10-15 00:02-00:07 with the real teacher and cached preds).

Expected console (one line per leg) and the frozen bars. Reference numbers are MEASURED with the REAL teacher file in a stand-in run
(App. B D5); they do not depend on S:

| leg | printed line starts | PASS iff (frozen) | reference values (REAL teacher, MEASURED) |
|---|---|---|---|
| G0 | `G0 integrity:` | PASS | as step 2 |
| G1 | `G1 gold58: S ... D 0.9214 \| D2 0.9192` | AUC58(S) - .9203 >= -.005, i.e. AUC58(S) >= .9153 | E2 stand-in .9199 |
| G2 | `G2 uptake: rho(S,R7) ...` (ruler `z|`, 4,333 rows) | rho(S,R7) - .7847 >= .0215 (rho(S,R7) >= .8062) AND OOF AUC(S vs v4) >= .85 | rho(D,R7) .7848, rho(D2,R7) .7846; OOF AUC vs v4 D .8860 |
| G3 | `G3 diversity:` | rho(S,R7) <= rho(D,D2) = .9625 | |
| G3' | `G3' diversity ceiling:` | d = rho(S,R7) - .7847 <= .1077 (rho(S,R7) <= .8924) | |
| G4 | `G4 blend (gold58): base ... 0.9304 \| +F .2 0.9333 \| +split(F,S) .2 ...` | split - .9333 >= -.002 | E2 stand-in d +.0003 |
| verdict | `R7T_STUDENT_GATE REAL A1 \| G0 P G1 P G2 P G3 P G3' P G4 P \| VERDICT PASS` | all six P | |

Output file: `~/r7t_labels/gate/student_gate_<YYYYmmdd_HHMM>.txt` (no `SIM_` prefix; the gate writes it for PASS and FAIL).
Copy only the console summary lines to the laptop (`r7teacher/labels/real_run/`), not the json tail.

### Branch 3 (PREREG sec 6, A1, sec 11; no exceptions, no re-weighting, no bar change)
- **VERDICT PASS** -> step 4.
- **Any of G0, G1, G2, G3, G4 FAIL** -> STOP: no T_b entry. The T_b line ends for the LB. Record the verdict in RUN_LOG +
  PREREG_10-15 READ notes. T_a continues on the box (Appendix A).
- **G3' FAIL alone** (the student is more R7 than fleet: closes > half the gap) -> VERDICT FAIL by the A1 definition ->
  **skip the LB entry** (same as a FAIL; make_s_pack refuses it). T_a may still enter later under its own gate.
- **exit 3 after `--infer`** -> gold_infer5 failed or wrote elsewhere: read `~/r7t_labels/gate/student_gate_REAL_r7tb_console.log`;
  never add `--pred_dir` (the gate only searches its hard-coded GOLD_CACHES, so a non-default --pred_dir always ends in exit 3).
- A stale cached prediction (rb_fold_check line FAIL) must be moved aside BEFORE step 3, else the gate silently reuses it.

## Step 4. Stage the S weights pack (box, ~1 min)
`make_s_pack.py` is NOT on the box yet (MEASURED). Copy it, check md5, run it on the PASS file:
```
# laptop
scp -o BatchMode=yes $R/session_2026-10-15/v224f/make_s_pack.py training@192.168.10.30:r7t_labels/make_s_pack.py
# box
cd ~/r7t_labels && md5sum make_s_pack.py          # f31f4702e11a9192c01d6df8b48cbfcc (fix aofix 10-15 F2; was 5b2df773 after fix 10-15 F2/F3, 2eaf33e6 before)
$CPU OMP_NUM_THREADS=4 $PY -W ignore make_s_pack.py --gate ~/r7t_labels/gate/student_gate_<stamp>.txt
```
Expected (MEASURED on the mock, App. B D2): five lines `f<F> OK f<F>_s1047_raw_ttr224-320x24_cv####_best.pt -> f<F>_s1047r7tb_raw_...
cv 0.#### ep <E> sha <16 hex>`, then `S_PACK OK /home/training/r7t_labels/s_pack/upload/rsna-knee-cnxt-fleet-r7tb | manifest sha256 ...`,
`~/r7t_labels/s_pack/S_PACK.json`. Folder = 5 hard-linked .pt (~108 MB each, ~565 MB) + `r7tb_manifest.json` + `dataset-metadata.json`
(`eeshsaxena/rsna-knee-cnxt-fleet-r7tb`, CC0, private when created by the CLI).
Fix 10-15 (F2): the gate file must be the T_b one. Its first line must name `student f{F}_s1047_ttr_r7tb` and a table whose
basename is `labels_r7t_b.csv`, and its only `R7T_STUDENT_GATE` line must be the all-pass REAL A1 line. A T_a gate file, a FAIL
file, or a file with two verdict lines is refused. `S_PACK.json` records `gate_sha256` and `gate_header`, and the `S_PACK OK` line
prints the gate sha256[:16]. (MEASURED on the box mock 10-15 00:36: positive OK and 12/12 negatives refused,
`~/r7t_labels/rb_dry/fix1015/d2_fix1015.log`.)
Fix 10-15 (F3): the printed NEXT text is now the FLIP-base plan (`--wf 0.20 --base ff`, `v224ffs_check.py`). Steps 6-10 below
remain the reference.
Fix aofix 10-15 (F2, v224f/RESULT.md "AO fix 10-15"): the printed NEXT text is now the PREREG_10-15 A2 + A3 branch table of Step 7
(a FF-S `--base ff` + `v224ffs_check.py`; b FLIP-AO + S `--base flip --start-budget-h 7.0` + `v224aos_check.py`; c FLRM-AO + S
`--base flrm --start-budget-h 7.0` + `v224aos_check.py --ref-flrm`).
Any AssertionError -> STOP (the refusals were all exercised in App. B D2: SIM gate, upload exists, seed, sel_labels, amp, OAI string,
extra top-level key, two raw-best files).

### Step 4b. Independent OAI-clean scan (box, ~15 s; the dataset rule, as for k14w)
```
cd ~/r7t_labels && U=s_pack/upload/rsna-knee-cnxt-fleet-r7tb && A=""; for F in 0 1 2 3 4; do c=$(ls $U/f${F}_s1047r7tb_raw_*_best.pt); \
  A="$A --ckpt $c=$HOME/fleet_runs/f${F}_s1047_ttr_r7tb/oof_f${F}_s1047_raw.csv=$HOME/fleet_runs/f${F}_s1043_ttr_frac75/oof_f${F}_s1043_raw.csv"; done; \
  $CPU ~/miniconda3/envs/kmimic/bin/python ~/k14_test/k14/box/k14_oai_clean.py --out s_pack/oai_clean_r7tb.json $A | tail -n 2; \
  grep -il oai $U/r7tb_manifest.json $U/dataset-metadata.json || echo "no oai text in manifest/metadata"
```
Expected: `K14CLEAN PASS 5/5 -> s_pack/oai_clean_r7tb.json` and `no oai text in manifest/metadata` (MEASURED on the mock).
FAIL -> STOP, nothing leaves the box.

## Step 5. Copy the pack to the laptop (~2-5 min GUESS on the LAN)
```
mkdir -p $R/session_2026-10-15/v224f/s_upload
scp -r -o BatchMode=yes training@192.168.10.30:r7t_labels/s_pack/upload/rsna-knee-cnxt-fleet-r7tb $R/session_2026-10-15/v224f/s_upload/
cd $R/session_2026-10-15/v224f/s_upload/rsna-knee-cnxt-fleet-r7tb && ls && python -c "import json,hashlib; m=json.load(open('r7tb_manifest.json')); \
  print('SHA_OK', all(hashlib.sha256(open(n,'rb').read()).hexdigest()==m['sources'][n]['sha256_full'] for n in m['checkpoints']), len(m['checkpoints']))"
```
Expected: 7 files (5 .pt + r7tb_manifest.json + dataset-metadata.json), `SHA_OK True 5`.

## Step 6. `[MAIN ONLY]` Create the private S dataset (~35 min MEASURED for the 538 MB k14w on 10-12; 2.2 GB took ~2 h on 10-09)
PowerShell (Windows CLI staging bug fix, HANDOFF sec 3):
```
$UP = "C:\Users\eeshs\Downloads\rsna-knee\session_2026-10-15\v224f\s_upload\rsna-knee-cnxt-fleet-r7tb"
New-Item -ItemType Directory -Force -Path (Join-Path "$env:TEMP\.kaggle\uploads" ($UP -replace ':', '_')) | Out-Null
cmd /c "kaggle datasets create -p $UP > C:\Users\eeshs\Downloads\rsna-knee\session_2026-10-15\v224f\s_upload\create_r7tb.log 2>&1"
kaggle datasets status eeshsaxena/rsna-knee-cnxt-fleet-r7tb      # poll until 'ready' (403 = not created yet)
kaggle datasets files eeshsaxena/rsna-knee-cnxt-fleet-r7tb       # must list exactly the 7 local files
```
Pass: `ready` and the file list == the local folder. Private (no `--public`). Never `datasets version` it later; a T_a pack gets
its own slug.

## Step 7. Build the student kernel, by branch (laptop, ~5 s)
**Branch table** (fix aofix 10-15 F2; PREREG_10-15 Amendments 2 + 3, each read by API best-id under A3 R1). Pick ONE row:

| branch | 10-15 read | entry | build flags | kernel dir / slug | gate (Step 9) |
|---|---|---|---|---|---|
| a | V224-FF20 became the team best | V224-FF-S (PREREG_10-14 A4) | `--base ff` (block below) | `kernel_v224ffs20` / `eeshsaxena/rsna-sub-v224ffs20` | `v224ffs_check.py` (Step 9 block) |
| b | V224-FLIP stays the team best (A3 R2; supersedes A2 b FFSR20) | FLIP-AO + S | `--base flip --start-budget-h 7.0` | `kernel_v224flips20` / `eeshsaxena/rsna-sub-v224flips20` | `v224aos_check.py` `--ref-exact --ref-flip` |
| c | V224-FLIP-RM became the team best (A3 R3) | FLRM-AO + S | `--base flrm --start-budget-h 7.0` | `kernel_v224flrms20` / `eeshsaxena/rsna-sub-v224flrms20` | `v224aos_check.py` `--ref-exact --ref-flip --ref-flrm` (required) |
| d | FF20 shows a canary | no FF-S / FFSR (A2 d) | - | - | - |
| e | student gate FAIL or the G3' skip | no student entry | - | - | - |

Branches b / c (AO, v224f/RESULT.md "AO 10-15" + "AO fix 10-15": flip FORCED as in V224-FLIP, F = S alone, REPLACE implied, w_f .20,
start guard G 7.0 h = MIN_START_MIN 78, wall 8.3 h):
```
cd $R && python session_2026-10-15/v224f/build_v224fs.py --wf 0.20 --base flip --start-budget-h 7.0 \
   --s-manifest session_2026-10-15/v224f/s_upload/rsna-knee-cnxt-fleet-r7tb/r7tb_manifest.json            # c: --base flrm
# after the Step 8 commit (out_v224flips20_v1; c: out_v224flrms20_v1):
cd $R && python session_2026-10-15/v224f/v224aos_check.py session_2026-10-15/v224f/out_v224flips20_v1 --wf 0.20 \
   --build session_2026-10-15/v224f/kernel_v224flips20/BUILD_V224FS.json \
   --ref-exact session_2026-10-14/v224/out_v224_v1 --ref-flip session_2026-10-14/v224/out_v224flip_v1 [--ref-s <an output with the same 5 S>]
# c: out_v224flrms20_v1, --build session_2026-10-15/v224f/kernel_v224flrms20/BUILD_V224FS.json, plus
#    --ref-flrm session_2026-10-15/v224rm/out_v224flrm_v1 (required)
```
Expected (MEASURED on the mock manifest, tests/test_aofix_1015.py): `BUILT v224flips20 ... cells 24 members 5 (K14 0 + S 5)
s_share 1.0 MIN_START_MIN 78.0 standin False` (c: `BUILT v224flrms20 ... cells 26`); after the commit `V224AOS_CHECK PASS 28/28`
(29/29 with `--ref-s`). Never `--base ff --replace` for branch b: that is the superseded FFSR20 form (time-aware flip + S = two
changes vs V224-FLIP). Branch a: the block below.
```
cd $R && python session_2026-10-15/v224f/build_v224fs.py --wf 0.20 --base ff \
   --s-manifest session_2026-10-15/v224f/s_upload/rsna-knee-cnxt-fleet-r7tb/r7tb_manifest.json
```
Expected (MEASURED with the mock manifest, App. B D8; re-measured after fix 10-15): `base: BUILT v224ff20 ... sha256 b86f8b9f12c1d573... cells 24 inserted [7, 9]`
then `BUILT v224ffs20 .../v224f/kernel_v224ffs20/rsna_sub_v224ffs20.ipynb sha256 <new> cells 25 members 25 (K14 20 + S 5) s_share 0.5
MIN_START_MIN 192.0 standin False datasets [... 'eeshsaxena/rsna-knee-cnxt-fleet-k11w', 'eeshsaxena/rsna-knee-cnxt-fleet-k14w',
'eeshsaxena/rsna-knee-cnxt-fleet-r7tb']`. kernel-metadata.json: id `eeshsaxena/rsna-sub-v224ffs20`, private, GPU T4, internet off.
Fix 10-15 (F1): defaults for `--base ff` are now flip budget **4.9 h** (V224-FF20's own value, cell 7 NOT patched, so cells 0-15
are byte-identical to the commit-tested V224-FF20 kernel) and fleet start budget **5.1 h** (MIN_START_MIN 192). BUILD_V224FS.json:
`flip_budget_h 4.9`, `patched_base_cells [16, 21]`, `start_budget_h 5.1`. `--start-budget-h 5.0` (MIN_START_MIN 198) is still
accepted; the old 4.4 h cut needs an explicit `--flip-budget-h 4.4`.
Sanity: vs the stand-in build `tests/standin_v224ffs20` only cells 20 (S_CFG literal), 22 (K2_CFG members) and 24 (G_S) differ, and
only by the S names / sha (checked on the mock build with `cmp_builds.py`-style diff, App. B D8).
Pitfalls (MEASURED): omitting `--base` is now refused (fix 10-15 F3; it used to build the NON-flip V224-F-S silently); `--wf 0.25 --base ff`
builds on the on-disk `kernel_v224ff25` that was never commit-tested. Only `--wf 0.20` is pre-registered; the base is the branch
table's (a `--base ff`, b `--base flip`, c `--base flrm`). Branch a: do NOT build `--base ff --replace` now (REPLACE only after a
positive FF-S read, Step 13). Branches b / c: REPLACE is implied by the AO base (no `--replace` flag).

## Step 8. `[MAIN ONLY]` Commit test (~10-13 min MEASURED for FF20: 585 s notebook + setup)
Only when step 6 says `ready` (else S-MERGE cannot find the dataset -> Synovitis canary -> the gate fails).
```
cd $R && python session_2026-10-15/v224f/running.py 6            # RUNNING_COUNT must be <= 1
python session_2026-10-15/v224ff/push_wait_fetch.py session_2026-10-15/v224f/kernel_v224ffs20 eeshsaxena/rsna-sub-v224ffs20 \
   session_2026-10-15/v224f/out_v224ffs20_v1
```
Pass: "successfully pushed" (only that counts), then COMPLETE and the output fetched (fallback fetch: `python
session_2026-10-15/v224f/wait_fetch.py eeshsaxena/rsna-sub-v224ffs20 session_2026-10-15/v224f/out_v224ffs20_v1 40`).
Branches b / c (fix aofix 10-15 F2): the same two commands with `kernel_v224flips20` / `eeshsaxena/rsna-sub-v224flips20` /
`out_v224flips20_v1` (c: `kernel_v224flrms20` / `eeshsaxena/rsna-sub-v224flrms20` / `out_v224flrms20_v1`).

## Step 9. Commit gate (laptop, ~40 s)
```
cd $R && python session_2026-10-15/v224f/v224ffs_check.py session_2026-10-15/v224f/out_v224ffs20_v1 --wf 0.20 \
   --build session_2026-10-15/v224f/kernel_v224ffs20/BUILD_V224FS.json \
   --ref-flip session_2026-10-14/v224/out_v224flip_v1 --ref-exact session_2026-10-14/v224/out_v224_v1 \
   --ref-fleet session_2026-10-15/v224ff/out_v224ff20_v1 | tee session_2026-10-15/v224f/gate_v224ffs20_v1.txt
```
Expected last line `V224FFS_CHECK PASS 14/14 mode clean` (MEASURED on the mock FF-S output, App. B D7). The nested lines
`V224_CHECK FAIL 14/17 mode exact` and `V224FS_CHECK FAIL 30/31` are BY DESIGN (the exact-mode V224 sub-gate cannot pass with flip;
it is replaced by the `--flip` sub-gate, which must be OK). Also look for `[s-merge ...] RESULT OK merged fleet dir + S: 30
checkpoints`, `RESULT OK fleet stage clean: 25 member`, `V224FS_F2G`, `HF_BLEND_OK`, flip budget 4.9 in FLIP_TTA_FORCED (fix 10-15;
the gate's `--flip-budget-h` default is now 4.9 and must equal the build json), K2 MIN_START_MIN 192.0.
FAIL -> no entry; read the first FAIL line.
Branches b / c: run the `v224aos_check.py` command of the Step 7 table instead; pass = last line `V224AOS_CHECK PASS 28/28`
(29/29 with `--ref-s`). Its nested `V224FS_CHECK` exact-mode sub-gate and f / ff MIN_START_MIN line FAIL by design and are replaced.

## Step 10. Runtime review (laptop, ~15 min; may overlap step 8)
```
cd $R && python session_2026-10-15/v224f/wall_model.py session_2026-10-15/v224f/out_v224ffs20_v1 session_2026-10-15/v224ff/out_v224ff20_v1
```
(its last line still says "fleet start budget 5.5 h" and "V224 3.0 h": text of the V224-F model; for FF-S use 5.0 h and add the
flip.) Read: fleet hidden central / slow / slow-session for 25 members (MODEL from the commit microbench). Pass criteria
(PLAN_TEACHER FX8, FIXES_A1 sec 2b; updated by fix 10-15 F1 and v224f/RUNTIME_FFS_1015.md): V224 part with flip ~4.4 h central
(3.0 + 1.4; FLIP scored ~4.6 h submit-to-score incl. queue on 10-14, MEASURED); flip budget 4.9 h (== V224-FF20) < fleet start
budget 5.1 h (asserted by the builder); worst case = fleet starting at 5.1 h + slow-session fleet ends 8.22 h vs the 8.3 h fleet
wall (MODEL, RUNTIME_FFS_1015 sec 4); MC median 6.61 h, p99 7.67 h, P(fleet skipped) ~.001, P(hard stop) ~.001 (MODEL).
Not SUBMIT -> no entry (PLAN sec 5).
Write the verdict into `session_2026-10-15/v224f/RUNTIME_V224FFS.md` (or the main session's runtime-review file).
Branches b / c: reviewed already (MODEL): `sim_ffs_runtime.py` section D (`sim_ffs_runtime_ao.txt`) and the paired guard comparison
`guard_paired_1015.py` (`guard_paired_1015.txt`). At G 7.0: kernel end p99 6.71 h (FLIP-AO + S) / 6.83 h (FLRM-AO + S) under
session model B, 7.47 / 7.54 h under STRESS; P(canary) .0008 / .0014 (B), .018 / .023 (STRESS), .012 / .018 (B, 1,500 studies);
P(hard stop) 0, P(> 9 h) 0. A higher guard never adds a canary (paired: 0 sessions), so 7.0 is used (fix aofix 10-15 F1).

## Step 11. `[MAIN ONLY]` Amendment 4 (append to `session_2026-10-14/PREREG_10-14.md`, one cross-reference line in
`session_2026-10-15/PREREG_10-15.md`), BEFORE the submit. Text = PLAN_TEACHER sec 7, adapted to the FLIP base (draft):
> Amendment 4 (<time> IST 10-15, before the V224-FF-S submit). R7-teacher student entry. V224-FF-S (`eeshsaxena/rsna-sub-v224ffs20`
> v1, notebook sha <from BUILD_V224FS.json>) = V224-FF20 (V224-FLIP + K14 fleet at w_f .20, time-aware flip) with the F slot
> = .5 mean(K14, 20) + .5 mean(S, 5); S = the 5 T_b student checkpoints (seed 1047; PREREG_TEACHER + A1 + A2; student gate
> `<gate file>` VERDICT PASS; V224FFS_CHECK PASS 14/14). w_f .20 unchanged. Runtime guard (not read as a change): fleet start
> budget 5.5 -> 5.1 h (MIN_START_MIN 192; S costs +0.29 h central, MEASURED microbench / MODEL). The time-aware flip budget stays
> 4.9 h, so the flip policy and the V224 part (cells 0-15) are byte-identical to V224-FF20 (this supersedes PLAN_TEACHER FX8's
> "cut the flip budget by ~0.5 h": RUNTIME_FFS_1015 + fix F1 10-15 showed the cut would silently drop the flip on most studies,
> MODEL flip share ~36 % vs ~77 % for FF20). Shared caveat, recorded before the read: the time-aware flip is a per-session lottery
> that FF-S shares with FF20 (flip >= 90 % of studies in ~74 % of sessions for both, MODEL), and the flip share is not observable
> after a hidden run; a non-moving read is therefore a weak negative for S, not proof that S is negative. The extra S stage can
> also skip the fleet in a slow session (Synovitis canary, P ~.001-.002 MODEL), which shows on the LB as a canary entry, not a
> read. One change vs its base. A win counts only as an API best-id move. If it
> wins: one REPLACE entry (F = S alone, `rsna-sub-v224ffsr20`). T_a gets at most one entry, same form, after the T_b read, only if it
> passes G0-G3' + G4. S never enters pick A; it enters pick B only through a winning entry, as part of the F component (max 3
> components). No per-label public tuning; no tuning of the .5 split or of w_f on the student line.

Branches b / c (fix aofix 10-15 F2): the entry text is an A3 R2 / R3 line in `session_2026-10-15/PREREG_10-15.md`, not A4. It names:
the base (V224-FLIP or V224-FLIP-RM, byte-identical), F = the 5 S alone at w_f .20, the start guard **7.0 h (MIN_START_MIN 78)**
with the visible Synovitis canary on a skip or a wall stop, the HF-AO Fracture rule (a Fracture canary from the V224 CANARY or
RM_GUARD stays 0.5 through the S blend), V224AOS_CHECK PASS, and the P(canary) figures of Step 10 for B, STRESS and 1,500 studies.

## Step 12. `[MAIN ONLY]` Submit
```
cd $R && python session_2026-10-14/submit_at.py eeshsaxena/rsna-sub-v224ffs20 1 "V224-FF-S: V224-FF20 with F = .5 K14 + .5 S (R7-teacher T_b student, 5 folds); w_f .20; fleet start 5.1 h, flip 4.9 h (= FF20); V224FFS_CHECK 14/14; PREREG_10-14 A4"
```
Conditions (PREREG_10-15 + PLAN_TEACHER sec 2/5): steps 3, 9, 10 PASS; Amendment 4 written; a 10-15 UTC-day slot (opens 05:30 IST;
05:36 FF20 and 06:25 FLIP-RM are armed; entries >= 45 min apart; PREREG_10-15 puts the student in slot 4-5).
Branches b / c (fix aofix 10-15 F2): submit `eeshsaxena/rsna-sub-v224flips20` (b) or `eeshsaxena/rsna-sub-v224flrms20` (c) the same
way, after the A3 entry text (Step 11 note) and `V224AOS_CHECK PASS`. The base-question bullets below are pre-A3 text: PREREG_10-15
A3 R2 / R3 replace "FF20 not best -> HOLD" with branch b and "FLIP-RM best -> NOT built" with branch c.
Base question (decision for main, not pre-registered beyond "the best base"):
- FF20 (05:36) reads as the new team best (expected read ~11:30-12:30 IST, MODEL: FLIP latency 4.6 h + fleet ~1.8 h) -> FF-S is the
  clean one-change entry on the best base. It shares that base with the midday composite (FF20-RM): PLAN sec 2 says never alongside
  another entry on the same base, so submit one, read it, then the other (main chooses the order).
- FF20 does NOT become best (the fleet does not add on the FLIP base) -> the Amendment 4 form (F slot of the best base) does not exist
  on V224-FLIP; FF-S would then test fleet + S on FLIP at once. Not pre-registered: default HOLD, main decides.
- FF20 shows a canary (Synovitis / Fracture 0.5) -> the FF base is broken for hidden runs: STOP, no FF-S.
- If a FF20-RM composite becomes the team best first, the student belongs on that base: NOT built (build_v224fs supports only
  `--base f|ff`); a new builder + commit + gate would be needed.

## Step 13. After the read (`[MAIN ONLY]`)
- API best-id moved to FF-S -> build REPLACE (`build_v224fs.py --wf 0.20 --base ff --replace --s-manifest ...`, tag v224ffsr20;
  MEASURED to build on the mock manifest), commit, `v224ffs_check.py` on its output with `--build kernel_v224ffsr20/BUILD_V224FS.json`,
  submit as its own entry. S enters the pick-B candidate list as part of F.
- Not moved -> the T_b line stops for the LB (no .5-split or w_f probing).
- Branches b / c (fix aofix 10-15 F2): no REPLACE step (F is already S alone). Read by API best-id under A3 R1 (a non-move while
  the base stopped being the team best is UNREAD, not negative).

---

## Appendix A. T_a handling (box queue, after T_b f4)
- Queue (MEASURED 23:55): T_b f4 (running) -> T_b f0-f3 box-fallback lines (REFUSED while the rental placeholders exist) -> **T_a
  f0-f4** (seed 1048, gate `labels_gate_labels_r7t_a.verdict` = PASS) -> L13 f3/f4. T_a f0 starts ~01:37; 5 x 3.52 h (MEASURED rate)
  -> 5/5 ~19:15 10-15 (MODEL), later by 3.5 h per T_b fold the box has to retrain (Branch 0-A).
- Gate T_a the same way: `rb_fold_check.py --tpl 'f{F}_s1048_ttr_r7ta' --table-name labels_r7t_a.csv`, then
  `student_gate.py --student 'f{F}_s1048_ttr_r7ta' --table real/labels_r7t_a.csv --receipt real/teacher_labels_receipt.json
  --teacher real/R7OOF.csv [--infer]`. Its bars are the same (rho_ref is the D family's, so G2 floor / G3' ceiling are identical).
- LB: only after the T_b read, as its own entry, same form, only if its own G0-G3' + G4 pass (PREREG sec 7, A2).
- **Tooling gap (not needed tonight):** `make_s_pack.py` hard-codes seed 1047 (`want`, glob, staged name `s1047r7tb`, S_RE) and
  `build_v224fs.py` hard-codes `S_RE = f[0-4]_s1047r7tb_...` and the kernel tag `v224ffs20` (a T_a build would collide with the T_b
  kernel slug). A T_a entry needs a parametrised copy (seed / tag / slug `rsna-knee-cnxt-fleet-r7ta` / manifest name / kernel tag e.g.
  `v224ffsa20`) with its own mock tests (the App. B D2 / D8 harness) before use. Neither file is PREREG-frozen.

---

## Appendix B. Dry runs re-run 10-14 23:51 -> 10-15 00:14 IST against the CURRENT layout (box CPU cores 34-39, nice 19, ionice idle; laptop CPU)

Box scratch `~/r7t_labels/rb_dry/` (logs kept; the 1.3 GB of mock checkpoints and the mock gate files were deleted after use;
README.txt says so). Laptop scratch = the session scratchpad `rb/`. All results MEASURED; mocks describe plumbing only.

| # | where | command (exact) | result |
|---|---|---|---|
| D0 | both | `md5sum` / `sha256sum` of the frozen files | laptop == box: student_gate.py 8c018790, build_teacher_labels.py a7f8823d, labels_gate.sh 6f15f414; box dry_run_student.sh 50e1d57d, finalize_labels.sh ff4d1ec9, install_r7ta_top.sh a921eabb, fleet_train.py 3e62434a; PREREG sha256 9e987880 both; real T_b table md5 d206bbcf (real/ == final/, sha256 7c51bb3a == LABELS_READY); receipt table_b md5 == file; R7OOF.csv 49fb633a; laptop build_v224fs.py fda1e28f, v224ffs_check.py defeda80, v224fs_check.py 057d896f, make_s_pack.py 2eaf33e6 (== FIXES_A1; pre-fix values, current ones in D11); FF20 kernel b86f8b9f; V224 notebook 4259ec97. No cached s1047 / s1048 gold preds in any GOLD_CACHES dir. |
| D1 | box | `cd ~/r7t_labels/rb_dry && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 taskset -c 34-39 nice -n 19 ionice -c3 ~/miniconda3/envs/rsna/bin/python -W ignore check_f4.py` | 14/14 OK on the RUNNING f4 (App. C F1) |
| D2 | box | `mk_mock_pack.py ~/r7t_labels/rb_dry/mock` (D set s1043 raw-best ckpts with args seed 1047, epochs 21, train/sel_labels labels_r7t_b.csv) then `make_s_pack.py --gate mockgate/MOCK_gate_pass.txt --runs mock/runs --out mock/s_pack` + 8 negatives | first try refused the mock (`f0 args {'epochs': 17}`: D ran 17 epochs; mock fixed, a correct refusal). Positive: 5 x `f<F> OK`, `S_PACK OK`, 6 s. Negatives all refused: upload exists, SIM gate file, seed 1043, sel_labels v4, `oai_9123` in args, extra top-level key, two raw-best files, amp fp32 |
| D3 | box | `~/miniconda3/envs/kmimic/bin/python ~/k14_test/k14/box/k14_oai_clean.py --out oai_clean_mock.json --ckpt <pack ckpt>=<oof>=<D oof> x5` | `K14CLEAN PASS 5/5`, 11 s; no oai text in manifest / metadata |
| D4 | box | `cd ~/r7t_labels && student_gate.py --student 'f{F}_s1047_ttr_r7tb' --table real/labels_r7t_b.csv --receipt real/teacher_labels_receipt.json --teacher real/R7OOF.csv --runs rb_dry/mock/runs --out rb_dry/gate_out` | header `REAL`, `G0 integrity: PASS` (REAL branch: table md5 vs receipt, PREREG md5 self-check, ckpt args, OOF ids, closing line), `MISSING gold predictions for 5 checkpoint(s)`, rc 3, nothing written. First exercise of the REAL G0 path against the real receipt. |
| D5 | box | `student_gate.py --student 'f{F}_s1046_ttr288frac75' --table real/labels_r7t_b.csv --receipt real/teacher_labels_receipt.json --teacher real/R7OOF.csv --standin --out rb_dry/gate_out` | SIMULATION run with the REAL teacher layout (z| ruler 4,333 rows, p| on gold58): G0 P G1 P **G2 F** G3 P G3' P G4 P, VERDICT FAIL (not a real verdict; E2 never learned the teacher, as designed). 5.5 min. Gives the reference values in step 3. |
| D6 | laptop | `python build_v224fs.py --wf 0.20 --base ff --standin` and `... --replace` | base FF20 rebuilt byte-identical (b86f8b9f); stand-ins byte-identical to the 18:26 files (037f67a7 / 24009f3f; all 5 files per dir `cmp` same) |
| D7 | laptop | `python tests/mock_fs_output.py ff` | `V224FFS_CHECK PASS 14/14 mode clean`; 7/7 negatives FAIL (fleet2g +1e-3, submission +1e-3, FX7 equal-mean bug, S npz row, SAVED members 20, no s-merge line, fleet2g missing); `MOCK_FS_GATE PASS 8/8`; 38 s. Regenerated tests/mock_out_v224ffs20 differs from the old copy only in timing fields (hf_receipt / v224f_receipt / log). |
| D8 | laptop | `python build_v224fs.py --wf 0.20 --base ff --s-manifest <mock r7tb_manifest.json from D2> --out <scratch>` (+ `--replace`, + negatives) | `BUILT v224ffs20` (43f3ebd0, 25 cells, 25 members) and `BUILT v224ffsr20` (e0b843f9, 5 members); vs the stand-in only cells 20 / 22 / 24 differ and only by S names / sha (cell 20: its first line); metadata id rsna-sub-v224ffs20, private, GPU, no internet, r7tb dataset attached. Negatives: `--s-share 0.4` refused, `--start-budget-h 5.5` refused; pitfalls: no `--base` builds the non-flip kernel, `--wf 0.25 --base ff` builds on the untested ff25. Scratch kernel-metadata renamed (unpushable). |
| D9 | box | `sim_ffs_cells.py ~/r7t_labels/rb_dry/ffs_sim` = tests/sim_fs_cells.py with ONLY the FF-S file names and cell indices +2 (copy; the original handles the f base only) on the FF-S stand-in notebooks + the FF20 commit npz / submission_V224 | `SIM_FS_CELLS PASS 15/15` (HF-S 9 scenarios incl. REPLACE, failed study, permuted targets, canaries; S-MERGE 6 incl. missing / tampered / extra / collision) |
| D10 | box | `rb_fold_check.py` (new, Step 1) on the mock / live f4 / live all | mock `PASS 50/50`; live f4 `--allow-open` `PASS 9/9`; live all now `FAIL 13/22` (f0-f3 empty rental placeholders, f4 open), as expected before T0 |
| D11 (fix 10-15 00:30-00:45) | both | fixes F1 / F2 / F3 (v224f/RESULT.md "Fixes 10-15"); new md5 laptop: build_v224fs.py a54afad6, make_s_pack.py 5b2df773 (box scratch copy `~/r7t_labels/rb_dry/fix1015/make_s_pack.py` == 5b2df773), v224fs_check.py b5f5d4e1, v224ffs_check.py d91afcc3, tests/mock_fs_output.py bc3603e8; the D0 values above are pre-fix (copies `*.pre_fix_1015`). `tests/test_fix1015.py` PASS 19/19 (pre-fix copies FAIL 9/19); D6 re-run: f stand-in notebooks byte-identical, ff stand-ins differ from the 00:07 ones only in cells 7 / 16 / 22 (flip 4.9, start 5.1 / 192), new sha256 190d0c70 / 4f4c0cf7; D7 re-run `V224FFS_CHECK PASS 14/14` + 7/7 negatives (f mock 32/32 + 7/7); `tests/gate_neg_fix1015.py` 5/5; D2 re-run on the box (new mock, deleted after): positive `S_PACK OK` (manifest 3f81cd0f, same as D2), 12/12 negatives refused (T_a gate, T_b-student-on-T_a-table, FAIL, two verdict lines, SIM, upload exists + the 6 old ckpt negatives); the pre-fix copy accepted the T_a gate. D9 not re-run: the S-MERGE and HF-S cells are byte-identical to the stand-in D9 used. |
| not re-run | box | `dry_run_student.sh` (trainer argv, 31 studies, ~12 min CPU) | superseded: the live f4 (and the rental smoke) run the frozen argv; f4's ckpt args match (D1). Re-running would rewrite ~/r7t_labels/dry/. `on_r7oof.sh` / `finalize_labels.sh`: REAL build done, re-runs refused by design. |

## Appendix C. Findings (report only; frozen files untouched)

- **F1 f4 naming / args (MEASURED 23:59 on the running run, ep 11 per-epoch ckpt + live files):** top-level files
  `f4_s1047_raw_ttr224-320x24_cv9741_best.pt`, `f4_s1047_ema_..._best.pt`, `oof_f4_s1047_raw.csv`, `oof_f4_s1047_ema.csv`, `epochs/`;
  log `~/fleet_runs/f4_s1047_ttr_r7tb.log`. The gate glob `f4_s1047_raw_*_best.pt` matches exactly 1 (ema excluded; `epochs/` is not
  globbed); staged name `f4_s1047r7tb_raw_ttr224-320x24_cv9741_best.pt` matches the make_s_pack / build_v224fs regex; OOF 866 rows ==
  folds5 fold-4 val minus gold74; args fold 4, seed 1047, epochs 21, ttr_size 224, fleet_model_ttr, model_sha f141aeed136dfc01, fp16,
  grad_ckpt False, train_labels == sel_labels == labels_r7t_b.csv; recipe keys == k11w D member; decode_probe 4 RSNA keys in the
  parity set; top-level keys {model, args, cv, epoch}; no `oai_<digit>`. The trainer's closing line format of a finished run matches
  the gate regex. Raw vs ema: both are saved; gate and pack use raw only (PREREG sec 5, A1 FX4). The trainer replaces the best file by
  delete-then-write (fleet_train.py `save`), so never read a top-level best file of an OPEN run (use `epochs/` for peeks).
- **F2 student_gate.py (frozen):** `--pred_dir` other than the default cannot work (the gate only looks in GOLD_CACHES) and the cache
  key is the run name, not the checkpoint sha: a stale `f{F}_s1047_ttr_r7tb__raw__A.csv` would be reused silently (gold_infer5 also
  skips existing outputs). Mitigation = rb_fold_check's stale-cache line before step 3.
- **F3 make_s_pack.py:** not on the box (copy in step 4); its printed NEXT block is the pre-FLIP plan (no `--base ff`, v224fs_check).
  FIXED 10-15 (D11): NEXT text = the FLIP-base plan; the gate file is now bound to student / table / all-pass verdict.
- **F4 build_v224fs.py:** default `--base f`; `--base ff` accepts any w_f whose kernel dir exists on disk (ff25 / ff30 were built but
  never commit-tested). Use exactly `--wf 0.20 --base ff`. FIXED 10-15 (D11): `--base` is required. The w_f point stands (use 0.20).
- **F5 tests/sim_fs_cells.py** covers the f-base notebook only (hard-coded file names + cell indices); the FF-S cells were proven with a
  +2 copy (D9). The cells are identical modulo tag, so this is a test-harness gap, not a kernel issue.
- **F6 T_a tooling** hard-coded to s1047 / r7tb (App. A).

## Appendix D. Timeline (IST 10-15)

| time | event | tag |
|---|---|---|
| ~01:35 | box f4 closes (13 epochs by 00:13, 10.0 min/epoch) | MODEL from MEASURED |
| ~01:37 | box T_a f0 starts (fallback lines REFUSED) | MODEL |
| 01:40-01:50 | 0.3 preflight f4 + optional 0.4 f4 gold inference (7.5 min) | MEASURED rate |
| ~02:30 | rental lanes end (3.89 h/fold from the smoke) | MODEL |
| **02:50 = T0** | FINAL_FETCH_OK (4 folds x ~3.7 GB incl. epochs/ at ~16-20 MB/s) | MODEL (task assumption) |
| 02:50-02:55 | step 1 rb_fold_check + step 2 gate preflight | MEASURED speeds |
| 02:55-03:15 | step 3 (fast path: 4 ckpts in 2 parallel procs ~15 min + legs 5.5 min); default path ends ~03:35 | MEASURED rates |
| 03:15-03:25 | step 4 make_s_pack (6 s) + 4b scan (15 s) + step 5 scp + sha (2-5 min GUESS) | |
| 03:25-04:00 | step 6 dataset create -> ready (~35 min for 538 MB on 10-12) | MEASURED once; GUESS tonight |
| 04:00-04:15 | step 7 build (5 s) + step 8 commit (10-13 min; longer if 2 GPU sessions are busy) | MEASURED |
| 04:15-04:30 | step 9 gate (1 min) + step 10 runtime review (overlaps 8) + step 11 amendment text | |
| **~04:30 (range 04:15-06:30)** | **submission-ready V224-FF-S kernel** | MODEL |
| 05:30 | 10-15 UTC slots open; 05:36 FF20, 06:25 FLIP-RM armed | MEASURED (armed) |
| ~11:30-12:30 | FF20 read (base question, step 12) | MODEL |
| after the FF20 read | FF-S submit (slot per main); read ~+6.6-7 h | MODEL |

Slack: default (sequential) inference +20 min; a slow upload (10-09 rate) +60-90 min; one missing rental fold -> ready ~11:30
(Branch 0-A, or ~3.5 h earlier if main stops T_a f0). The submit itself is gated by the FF20 read, not by this chain, so a
04:30-06:30 finish costs nothing.

## Preflight verdict

Written 10-15 00:45 IST by the go/no-go synthesis agent. Plain text, read-only checks only.

```
PREFLIGHT VERDICT (10-15 00:45 IST): GO. Two conditions: T0 (5/5 folds) must arrive, and main must adopt the 4.9/5.1 runtime policy in Amendment 4.
Ready: YES. Every dry run passes on the current layout and on the post-fix scripts (MEASURED). Nothing was pushed, submitted or created; the queue, the box GPU and the rental kit were not touched.
Blockers: none now. Still waiting on: box f4 (MEASURED at 00:41: ep 15 of 21; closes ~01:35, MODEL) and FINAL_FETCH_OK (absent at 00:41; expected ~02:45-02:55, MODEL). The box gate dir holds only SIM_ files, and no s1047 gold cache exists yet (MEASURED).
Decisions for main:
(1) Amendment 4 must adopt flip 4.9 h / fleet start 5.1 h (MIN_START_MIN 192). This overrides PLAN_TEACHER FX8. To revert, pass --flip-budget-h 4.4 --start-budget-h 5.0 on the build and --flip-budget-h 4.4 on the gate.
(2) Submit timing vs the FF20 read and the FF20-RM composite. Both share the same base, and PLAN says never alongside.
(3) If FF20 is not team best: HOLD (the default).
Timeline, IST 10-15 (MODEL unless tagged):
- 02:50 T0 (FINAL_FETCH_OK and f4 closed); 02:50-02:55 steps 1-2.
- 02:55-03:15 step 3 gate, fast path (the default path ends ~03:35). 03:15-03:25 pack, OAI scan, scp (scp time GUESS).
- 03:25-04:00 [MAIN] dataset create (35 min MEASURED once; at 10-09 rates add 60-90 min). 04:00-04:15 build plus [MAIN] commit test.
- 04:15-04:30 v224ffs_check, runtime review, Amendment 4. Submission-ready ~04:30 (range 04:15-06:30).
- 05:30 slots open; FF20 read ~11:30-12:30; FF-S submit after that, read ~6.6-7 h later.
- One missing rental fold moves ready to ~11:30 (about 3.5 h sooner if main stops T_a f0).
Runtime verdict: FITS 9 h.
- As fixed (flip 4.9 / start 5.1, MODEL MC): median 6.61 h, p99 7.67 h, slow corner 8.22 h vs the 8.3 h wall. P(fleet skipped) ~.001, P(hard stop) ~.001, P(>9 h) 0.
- S costs +0.29 h central (MEASURED microbench, 1.44-1.62 s/study/T4).
- Flip coverage now matches FF20: flip on >= 90 % of studies in ~74 % of sessions (FF20 ~73 %; 28 % as originally built), MODEL.
Confirmed defects, all FIXED on the laptop. No frozen file was edited (student_gate.py is still 8c018790).
- F1 (medium): FF-S cut the flip budget from 4.9 to 4.4 h, silently losing most of the flip and confounding the S read. Fixed in build_v224fs a54afad6: flip 4.9, cells 0-15 = FF20, start 5.1. Checks updated: v224fs_check b5f5d4e1, v224ffs_check d91afcc3.
- F2 (real, low reachability): make_s_pack accepted any REAL PASS file, so a T_a PASS could pack T_b. Fixed in make_s_pack 5b2df773: binds student, table and the all-pass verdict line, and records gate_sha256.
- F3 (low): --base defaulted to f, and the NEXT text was the non-flip plan. Fixed: --base is now required, and the NEXT text is the FLIP plan.
- Evidence (MEASURED): test_fix1015 19/19 (the pre-fix copies fail 9/19); FF-S mock gate 14/14 plus 7/7 negatives; f regression 32/32; gate_neg 5/5; box pack dry run S_PACK OK with 12/12 negatives refused. md5s re-checked on the laptop and the box at 00:41.
Unverified or low:
- F4 (frozen student_gate): gold cache keyed by run name. Never run gold inference on an OPEN run, and move a stale cache aside before step 3.
- sim_fs_cells and dry_run_student.sh were not re-run (the S-MERGE and HF-S cells are byte-identical).
- F is all-or-nothing: an S failure drops K14 too (canary). Runtime rests on one FLIP latency sample; check in the commit bench that each S member is <= 0.40 s/study/T4.
- Runbook step 10 still says "for FF-S use 5.0 h"; read 5.1 h (doc nit, not fixed). T_a tooling is hard-coded to s1047/r7tb.
First 3 commands at T0 (box; $CPU and $PY as in the runbook header):
1. cd ~/r7t_labels && $CPU OMP_NUM_THREADS=4 $PY -W ignore ~/r7t_labels/rb_dry/rb_fold_check.py    -> PASS 50/50 (49/50 if 0.4 ran)
2. cd ~/r7t_labels && $CPU NTHR=6 $PY -W ignore student_gate.py --student 'f{F}_s1047_ttr_r7tb' --table real/labels_r7t_b.csv --receipt real/teacher_labels_receipt.json --teacher real/R7OOF.csv    -> REAL, G0 PASS, rc 3
3. Command 2 plus --infer 2>&1 | tee ~/r7t_labels/gate/student_gate_REAL_r7tb_console.log (fast path: step 3)    -> VERDICT PASS, else STOP per Branch 3
```
