# RUNBOOK (Eesh side): TTR224 frac75 seed 1045 folds 0-3 on swapiy's Kaggle GPU (kaggle_frac75d)

Written 2026-10-11 by Claude. Kit = a copy of `session_2026-10-09/kaggle_frac75c` with ONE change: the plan line
(folds 0-3, seed 1045). `build.py` asserts code cells 2-8 are byte-identical to the proven notebook that completed on
Kaggle 10-07/08 (sha256 d6175d4a...) and that the labels csv matches the LIVE dataset. No em dashes.

## 0. What and why
- Two notebooks for swapiy's account (2 GPU sessions may run at once): A `rsna-train-ttr-frac75-s1045-f01` (folds 0 + 1),
  B `rsna-train-ttr-frac75-s1045-f23` (folds 2 + 3). Each ~10.0-10.7 h, hard wall 11.4 h, push with `--timeout 43200`.
- Goal: a third-seed TTR224 frac75 set (D3) for the final fleet. Fold 4 s1045 follows later (next quota week or the box).
- RSNA-only data (OAI can never go to Kaggle). Expected value is small (MODEL +.0001 to +.0002 as an extra same-family
  set); it costs only teammate quota. Any use in a submission needs its own pre-registered fleet gate first.

## 1. Before GO
1. swapiy is on team "Eesh saxena" and can view: datasets `eeshsaxena/rsna-knee-fleet-train`, `eeshsaxena/rsna-knee-frac75-labels`; notebooks `eeshsaxena/rsna-fleet-prep-s0`, `eeshsaxena/rsna-fleet-prep-s1`.
2. swapiy's `kaggle quota` shows >= 24 h free for both notebooks (>= 12 h for one).
3. Box: no `~/fleet_runs/f[0-3]_s1045_ttr_frac75` dir exists and no queue line writes one:
   `ssh -o BatchMode=yes training@192.168.10.30 'ls -ld ~/fleet_runs/f[0-3]_s1045_ttr_frac75 2>&1; grep -c s1045_ttr_frac75 ~/queue/queue.txt'`
   (expected: "No such file" and 0; note `f*_s1045_ttr288frac75` is set E and is a different name).

## 2. At T0 (both show RUNNING)
Make the EMPTY placeholders so nothing else writes these names:
`ssh -o BatchMode=yes training@192.168.10.30 'for F in 0 1 2 3; do d=~/fleet_runs/f${F}_s1045_ttr_frac75; if [ -e $d ] || [ -e $d.log ]; then echo EXISTS $d; else mkdir $d && echo placeholder $d; fi; done'`

## 3. Monitor
`kaggle kernels status swapiy/rsna-train-ttr-frac75-s1045-f01` and `-f23` (owner = swapiy's Kaggle username).
ERROR at once with an EMPTY log = quota refusal (do not retry blindly). ERROR in the first ~30 min = an input not shared (log line with ABORT).

## 4. Fetch + verify (PowerShell, per notebook S in f01, f23)
`kaggle kernels output swapiy/rsna-train-ttr-frac75-s1045-<S> -p <KT>\out_s1045_<S> --page-size 200 --file-pattern '^(?!_[a-z0-9]+_env/)(?!.*epochs/.*\.pt$)'`
then `python <KT>\verify_outputs.py <KT>\out_s1045_<S>` (expect status DONE, E 21 or 22, 0 mismatch, exit 0).

## 5. Box copy (Git Bash), same as kaggle_frac75c RUNBOOK sec 6 with s1045 names
For each run R in f0/f1 (S=f01) and f2/f3 (S=f23): check the placeholder is EMPTY, `scp -r out/R/* box:fleet_runs/R/`,
`scp out/R.log box:fleet_runs/R.log`, receipt to `~/fleet_runs/kaggle_frac75_s1045_<S>_run_receipt.json`, then box
`~/kt_frac75c/verify_outputs.py ~/fleet_runs --receipt <receipt>`.

## 6. If a run fails: `rmdir` only that notebook's EMPTY placeholders. Never version the input datasets.

## 9. Verification done
- 2026-10-11: `python build.py` PASS (code cells == proven notebook; labels sha == LIVE). Box dry tests not run (same
  accepted risk as kaggle_frac75c: only the plan line differs from a notebook that completed twice on Kaggle).
