# RUNBOOK (Eesh): TTR224 frac75 seed 1044 folds 1-4 on two teammates' Kaggle GPU (kaggle_frac75c)

Written 2026-10-09 by Claude. Nothing here was done on Kaggle: no dataset was created or versioned, nothing was pushed,
shared or submitted. Every Kaggle step below is yours. No em dashes.

## 0. What and why

Two notebooks, ONE per teammate (each needs one GPU session and >= 12 h of free quota; weekly quotas reset 10-10
05:30 IST):

| pack | teammate | slug | GPU0 | GPU1 | outputs |
|---|---|---|---|---|---|
| `friend_s1044_A.zip` | swapiy | `rsna-train-ttr-frac75-s1044-f12` | fold 1, seed 1044 | fold 2, seed 1044 | `f1_s1044_ttr_frac75`, `f2_s1044_ttr_frac75` |
| `friend_s1044_B.zip` | kmrsuman | `rsna-train-ttr-frac75-s1044-f34` | fold 3, seed 1044 | fold 4, seed 1044 | `f3_s1044_ttr_frac75`, `f4_s1044_ttr_frac75` |

- **Goal.** With `~/fleet_runs/f0_s1044_ttr_frac75` (from `rsna-train-ttr-frac75-f4x`, kmrsuman, 10-08: E 22, OOF raw
  .8866 / ema .8846) these complete a 5-fold **TTR224 frac75 s1044** set: the "next widening for 10-11+" of PROGRESS
  10-09 08:40 (a 4th set for FN-wide, MODEL ~+.0002).
- **Recipe.** Identical to the frac75 runs that completed on Kaggle 10-07/08 (`rsna-train-ttr-frac75-f23` E 21,
  `-f4x` E 22): the box TTR224 recipe with ONE change, `--labels labels_frac75_logit.csv` (dataset
  `eeshsaxena/rsna-knee-frac75-labels`, sha256 `5036425c...` baked in), `--seed 1044` for both trainers.
- **Only change vs the kaggle_frac75b kit:** the `NB_PLAN` line in `nb_src/c01_lib.py` (written per notebook by
  `build.py`) and comment lines. `build.py` asserts it: code cells 2-8 are byte-identical to the proven notebook
  (sha256 `d6175d4a...`), cell 1 differs only in comments and `NB_PLAN`.
- **Paired reads (report-only):** folds 1-3 have the same-platform, same-seed mix50 runs
  `~/fleet_runs/f{1,2,3}_s1044_ttr_mix50` (Kaggle T4 x2, E 21, workers 3); only the labels differ. Fold 4 has none
  (`f4_s1044_ttr_mix50` does not exist on the box).
- **Box queue:** no line of `~/queue/queue.txt` trains a `*_s1044_ttr_frac75` name (checked read-only 10-09 08:31 IST),
  so there is nothing to refuse or restore. The placeholders (sec 3) only reserve the names.

### Budget (the full proven budget, unchanged)

Training is planned to END by **10.3 h**; hard wall **11.4 h**. Measured on Kaggle T4 x2: setup + store + checks +
probe end at 0.135-0.144 h; 0.78-0.82 s/it; finalize < 1 min. Latest receipts: `-f23` E 21 at 0.820 s/it, session
10.23 h; `-f4x` E 22 at 0.780 s/it, session 10.45 h (2.8 % over its estimate); `rsna-train-ttr-f34` (slowest ever)
+5.6 %, session 10.74 h.

| per notebook | expected | bound |
|---|---|---|
| E (epochs) | **21** at 0.79-0.83 s/it; **22** at <= ~0.785 s/it; 20 at 0.83-0.87; WARNING below 16 | min(24, floor((10.3 h - elapsed - 2 min) / (1733 x s/it x 1.21))) |
| `--save_epochs` | (E-2)-E (last 3 epochs: ckpt + OOF) | |
| session ends | ~T0 + 10.0-10.7 h | hard wall T0 + 11.4 h, then quiet-wait <= 2 min + kill grace <= 1 min + finalize: < 11.5 h |
| GPU quota | ~10.0-10.8 h per notebook | 12 h (the push `--timeout 43200`), so each teammate needs **>= 12 h free** |
| outputs | ~1.8 GB per notebook | |

- **Push with `kaggle kernels push -p <dir> --timeout 43200`** (12 h, the Kaggle maximum). Without `--timeout` Kaggle
  refuses the session when the account has < ~12 h free (EMPTY log, ERROR within ~2 min).
- The wall path ends ~31 min inside the 12 h cap (the dry test checks it against the tighter 11.5 h).

```
KT  = C:\Users\eeshs\Downloads\rsna-knee\session_2026-10-09\kaggle_frac75c
  nb_src\ build.py                    shared notebook source + builder (A and B differ only in the NB_PLAN line)
  kernel\<slug>\                      kernel folders, id eeshsaxena/... (only if you ever run them yourself)
  kernel_teammate\<slug>\             the same notebooks, id YOUR_KAGGLE_USERNAME/... (in the teammate packs)
  pack\*.tmpl.md  pack_friend.py      teammate README / TODO templates + pack builder (one pack per notebook)
  verify_outputs.py                   checks downloaded outputs against run_receipt.json
  dry\                                box dry tests (box copies: ~/kt_frac75c/dry_src/, logs ~/kt_frac75c/*.log)
Packs: Downloads\rsna-knee\friend_s1044_A.zip (swapiy), friend_s1044_B.zip (kmrsuman), each: README_TEAMMATE.md,
  TODO.md, RUNBOOK_EESH.md (this file), the notebook, cli\<slug>\{kernel-metadata.json, notebook}
Inputs (all LIVE, unchanged): datasets eeshsaxena/rsna-knee-fleet-train + eeshsaxena/rsna-knee-frac75-labels,
  kernel outputs eeshsaxena/rsna-fleet-prep-s0 + -s1
B = training@192.168.10.30   (ssh -o BatchMode=yes $B '...' < /dev/null)
```

## 1. Before anything is shared

1. **Team + rules.** Both teammates are on team "Eesh saxena" (they ran the 10-07/08 frac75 runs). Re-check nothing
   changed.
2. **Quota.** Each teammate: `kaggle quota` shows **>= 12 h** free (after the 10-10 05:30 IST reset).
3. **Box: no `*_s1044_ttr_frac75` dir for folds 1-4 yet:**
   ```bash
   ssh -o BatchMode=yes $B 'ls -ld ~/fleet_runs/f[1234]_s1044_ttr_frac75* 2>&1; grep -c "s1044_ttr_frac75" ~/queue/queue.txt' < /dev/null
   ```
   Expected: one `No such file or directory` line (the glob matches nothing; `f0_s1044_ttr_frac75` is not in it) and
   `0`. If a dir exists, find out what wrote it before going on.

## 2. Share the inputs and hand over the packs (web UI)

1. Check that both teammates can still see, **Can view**: datasets `eeshsaxena/rsna-knee-fleet-train` and
   `eeshsaxena/rsna-knee-frac75-labels`; notebooks `eeshsaxena/rsna-fleet-prep-s0` and `eeshsaxena/rsna-fleet-prep-s1`
   (shared for the 10-07 runs; add them again if one is missing).
2. Send `Downloads\rsna-knee\friend_s1044_A.zip` to swapiy and `friend_s1044_B.zip` to kmrsuman.
3. Say **GO** to each when you want them started (the README tells them to wait for it). They are independent: one
   can start before the other.

## 3. When a notebook shows `running` (= its T0)

1. Write down T0. It ends by ~T0 + 10.0-10.7 h, never after T0 + 11.5 h (+ output saving).
2. **Make that notebook's EMPTY placeholders right away** (A: folds 1 2; B: folds 3 4). They reserve the names: no box
   job or collector may write or read them until sec 6 has filled them.
   ```bash
   FOLDS="1 2"   # A (swapiy). For B (kmrsuman): FOLDS="3 4"
   ssh -o BatchMode=yes $B "for F in $FOLDS; do d=~/fleet_runs/f\${F}_s1044_ttr_frac75; if [ -e \$d ] || [ -e \$d.log ]; then echo \"EXISTS \$d (stop: sec 1 step 3)\"; else mkdir \$d && echo \"placeholder \$d\"; fi; done" < /dev/null
   ```
3. Do not collect the s1044 frac75 set while a placeholder is still empty.

## 4. Monitor

```powershell
kaggle kernels status swapiy/rsna-train-ttr-frac75-s1044-f12
kaggle kernels status kmrsuman/rsna-train-ttr-frac75-s1044-f34
```
(the owner part is the teammate's Kaggle username; use what `kaggle kernels list --user <name>` shows if it differs)

| status | meaning / action |
|---|---|
| `error` at once, **EMPTY log** | Kaggle refused the session (quota < 12 h, or no free GPU slot). The teammate must not retry blindly: ask for a `kaggle quota` screenshot. Retry only with >= 12 h free. Placeholders: sec 7 |
| `error` within ~30 min | an input check failed: the last `[kt ...] ABORT` line says why (usually an input not shared). Re-run only with >= 12 h free |
| `error` later | unexpected: fetch the log (sec 5) and ask Claude |
| `complete` | sec 5 |

## 5. Fetch and verify (PowerShell, per notebook)

```powershell
$KT = "C:\Users\eeshs\Downloads\rsna-knee\session_2026-10-09\kaggle_frac75c"
foreach ($X in @(@("swapiy", "f12"), @("kmrsuman", "f34"))) {
  $U = $X[0]; $S = $X[1]; $OUT = "$KT\out_s1044_$S"
  kaggle kernels output "$U/rsna-train-ttr-frac75-s1044-$S" -p $OUT --page-size 200 --file-pattern '^(?!_[a-z0-9]+_env/)(?!.*epochs/.*\.pt$)'
  python "$KT\verify_outputs.py" $OUT
}
```

- The pattern drops the 12 per-epoch checkpoints per notebook (~1.35 GB) and any vendored `_xxx_env/` dir. It keeps the
  4 best checkpoints, the best and per-epoch OOF csvs, the logs, `run_receipt.json` and `probe/`.
- Expected: `status DONE`, `plan E=21` (or 22) with `save_epochs=19-21` (or 20-22), `0 mismatch, 12 absent`, exit 0.
- Exit 2 = PARTIAL / ABORTED: read the warnings. A fold is usable only if its `.log` ends with `{"raw": ..., "ema": ...}`.
- Exit 1 = a file differs or a ckpt is UNUSABLE: fetch again before anything else.

## 6. Box: copy into the placeholders and verify (Git Bash)

```bash
KT=/c/Users/eeshs/Downloads/rsna-knee/session_2026-10-09/kaggle_frac75c; B=training@192.168.10.30
for X in "f12 f1_s1044_ttr_frac75 f2_s1044_ttr_frac75" "f34 f3_s1044_ttr_frac75 f4_s1044_ttr_frac75"; do
  set -- $X; S=$1; shift; OUT=$KT/out_s1044_$S
  for R in "$@"; do
    ssh -o BatchMode=yes $B "test -d ~/fleet_runs/$R && [ -z \"\$(ls -A ~/fleet_runs/$R)\" ] && [ ! -e ~/fleet_runs/$R.log ]" < /dev/null \
      || { echo "STOP: ~/fleet_runs/$R is not an empty placeholder"; break 2; }
    scp -r "$OUT/$R/"* $B:fleet_runs/$R/ && scp "$OUT/$R.log" $B:fleet_runs/$R.log
  done
  scp "$OUT/run_receipt.json" $B:fleet_runs/kaggle_frac75_s1044_${S}_run_receipt.json
  ssh -o BatchMode=yes $B "~/miniconda3/envs/rsna/bin/python ~/kt_frac75c/verify_outputs.py ~/fleet_runs --receipt ~/fleet_runs/kaggle_frac75_s1044_${S}_run_receipt.json | tail -n 2" < /dev/null
done
```

- Run it for one notebook only by dropping the other string from the `for X in` list.
- Result: `~/fleet_runs/f{1,2,3,4}_s1044_ttr_frac75/` (+ `.log`) and `kaggle_frac75_s1044_{f12,f34}_run_receipt.json`.
  Box `verify_outputs.py`: `0 mismatch, 12 absent`, exit 0 (the 12 absent are the `epochs/*.pt` left on Kaggle).
- With `f0_s1044_ttr_frac75` (E 22) the set is complete: TTR224 frac75 s1044, folds 0-4, E 21-22 (all above the
  `K5_MIN_EPOCHS=14` floor used for set D).

## 7. If a run fails or is refused

- Remove ONLY that notebook's EMPTY placeholders (`rmdir` refuses non-empty dirs):
  ```bash
  FOLDS="1 2"   # A; B: FOLDS="3 4"
  ssh -o BatchMode=yes $B "for F in $FOLDS; do rmdir ~/fleet_runs/f\${F}_s1044_ttr_frac75 2>/dev/null && echo removed f\$F; done" < /dev/null
  ```
- No queue line to restore (none exists for these names).
- A retry needs >= 12 h free on that account; otherwise ask Claude for a one-fold rebuild or another account.

## 8. Never

- Version `eeshsaxena/rsna-knee-fleet-train` or `eeshsaxena/rsna-knee-frac75-labels`.
- Kill a box job, or write into a placeholder that is not empty.
- Pick a run whose log lacks the closing line.

## 9. Verification done

- 2026-10-11 13:45 IST: `python build.py` re-run PASS: code dataset copy == kt_manifest.json (14 files); labels csv sha256 5036425cd0d64f27 == LIVE; code cells 2-8 byte-identical to the proven kaggle_frac75b notebook (sha256 d6175d4a..., completed on Kaggle 10-07/08 as rsna-train-ttr-frac75-f23 E 21 and -f4x E 22); cell 1 differs only in comments + NB_PLAN. Built: A f12 sha256 2f9092bb29a103ce, B f34 0f48ca548f29d604.
- The separate box dry tests (dry/kt_dry*.py) were NOT run for this kit (it was parked on 10-09 08:45). The risk accepted: only NB_PLAN (fold numbers, seed 1044) differs from a notebook that already completed twice on Kaggle.
- 10-11: Eesh decided to use kmrsuman's 30 h for BOTH notebooks (A f12 and B f34 on kmrsuman's account; two concurrent T4 x2 sessions are allowed, 2 x 12 h <= 30 h). Pack: Downloads/rsna-knee/kmrsuman_d2_pack.zip. When the folds land, remove the box queue's D2 filler lines (f1-f4 s1044) with qctl, or the placeholders will make the runner refuse them.
