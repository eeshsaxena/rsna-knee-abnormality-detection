# S6 entry runbook (10-16): S6 seed s43 -> soup or 2-model entry (PREREG_10-15 Amendment 7)
Written 10-15 ~23:55 IST by Claude. Run from C:/Users/eeshs/Downloads/rsna-knee unless a line says box. No em dashes.

## What is ready (MEASURED tonight)
- `s6entry/patch_s6_model.py`: swaps ONLY the S6fall model of ANY built kernel that has the S6F setup cell (S6F, S6F-J, R6, KC1).
  The driver reads the model trio (`.pt`, `.run_config.json`, `.DONE.json`) from a 2nd mount = our private dataset; nartaa's dataset
  still supplies the code tree. Asserts: the setup cell differs only in DRIVER_SHA / DRIVER_B64; every other cell byte-identical.
- `s6entry/s6m_check.py OUT --tag T --base-out BASE_OUT [--control]`: the gate (swap took effect, stages before S6F == base, B probs,
  S6F blend and KC blend recomputed, format). Negative controls FAIL exactly the 2 swap checks.
- `s6entry/gpu_commit_s6m.sh TAG BASE_OUT [--control]`: running check, push, wait + fetch, gate. NEVER submits.
- **CONTROL PASS:** `rsna-sub-v224s6mctlj` v1 (nartaa s42 via `eeshsaxena/rsna-s6m-control`) = S6F-J output exactly (S6M_CHECK 20/20).
- Box `~/s6seed/parity/run_test3_parity.sh MODEL REF_B_PROBS NAME`: A7 item 4 serve parity on the 3 visible studies (CPU fp32 vs
  Kaggle GPU fp16, bar |dp| <= 2e-3). Control run on nartaa s42: ~/s6seed/parity/test3/ctl_s42.out.
- Box `~/s6seed/parity/run_gold58_serve.sh MODEL NAME`: gold58 serve-path predictions (s6_member_gate input). nartaa's runs tonight
  -> ~/s6seed/gold58_serve/nartaa_s42/out/predictions.csv (DONE.txt says rows of 58).

## AUTOMATED (10-16 10:39): box `~/s6seed/after_s6s43.sh` (pid in logs/after_s6s43.log) runs steps 1-4 below by itself the moment
RUN_s6s43.verdict appears: gold58 GPU for nartaa / ours / soup -> A7_GOLD58 + A7_SOUP verdicts -> nartaa GPU-vs-CPU cross-check ->
OAI scan (gates/oai_clean_s6.json) -> s44 launcher (PREREG A7b; gates/A7b.verdict PASS) iff a gate PASSES. Result: gates/AFTER_s6s43.done.
Laptop step 5 is now one command (run from s6entry, MSYS_NO_PATHCONV=1 or Git Bash mangles /home paths):
`MSYS_NO_PATHCONV=1 py -3.12 -I make_s6_ds.py --box-model /home/training/s6seed/soup/E_prior_s4243_fsoup.pt --slug eeshsaxena/rsna-s6-soup-s4243 --note "uniform soup nartaa s42 + s43"`
(prints the exact patch_s6_model.py arguments). Tested on nartaa s42 (sha matches the box, run_config == the control dataset).

## After the S6 run (box gates/RUN_s6s43.verdict PASS; expected ~15:00-19:00 10-16)
Gold58 serve preds go on the box GPU once it is idle (s6 run ended, nothing else running; the script refuses if any compute
process is on the GPU). All three models through the SAME path (GPU fp16, 2 workers = the Kaggle serve), minutes each:
`~/s6seed/parity/run_gold58_serve_gpu.sh MODEL NAME` -> ~/s6seed/gold58_serve_gpu/NAME/out/predictions.csv + DONE.txt.
(The CPU twin run_gold58_serve.sh runs ~8 min/study under tonight's load; nartaa's CPU run is an overnight cross-check only.)
1. `bash ~/s6seed/parity/run_gold58_serve_gpu.sh ~/s6seed/nartaa/E_prior_s42_fall.pt nartaa_s42`
   `bash ~/s6seed/parity/run_gold58_serve_gpu.sh <run out>/E_prior_s43_fall.pt ours_s43` (path from logs/launch_when_free.log)
2. Soup (A7 sec 5 (i), HEAD_INIT MATCH holds): `~/miniconda3/envs/kmimic/bin/python ~/s6seed/s6_soup.py ~/s6seed/nartaa/E_prior_s42_fall.pt <run out>/E_prior_s43_fall.pt --out ~/s6seed/soup/E_prior_s4243_fsoup.pt`
   then `bash ~/s6seed/parity/run_gold58_serve_gpu.sh ~/s6seed/soup/E_prior_s4243_fsoup.pt soup_s4243`.
3. Gates (box CPU, seconds; train.csv = ~/kmimic/kaggle/input/rsna-knee-abnormality-detection/train.csv; G = ~/s6seed/gold58_serve_gpu):
   - `python ~/s6seed/s6_member_gate.py --train-csv <train.csv> --nartaa $G/nartaa_s42/out/predictions.csv --ours $G/ours_s43/out/predictions.csv --out ~/s6seed/gates/A7_GOLD58.verdict`
   - `python ~/s6seed/s6_member_gate.py --soup --train-csv <train.csv> --nartaa $G/nartaa_s42/out/predictions.csv --ours $G/soup_s4243/out/predictions.csv --out ~/s6seed/gates/A7_SOUP.verdict`
   - Entry form: soup PASS -> SOUP entry (1-model cost). Soup FAIL but g1 + g2 PASS -> 2-model entry needs a driver that runs 2
     checkpoints (NOT built: compact_infer averages every E_prior_s*_f*.pt in --models, so the 2-model form = 2 files in our mount and
     the driver's "models folder holds exactly this one file" symlink step extended; plus the A7 runtime rule G2 4.5 h). Any FAIL -> no entry.
4. OAI-string scan (box): `~/miniconda3/envs/kmimic/bin/python ~/k14_test/k14/box/k14_oai_clean.py --out ~/s6seed/gates/oai_clean_s6.json --ckpt <entry .pt>`
   (expect PASS; FAIL -> STOP).
5. Dataset (laptop): copy the entry `.pt` + a run_config (nartaa's geometry keys; for the soup copy nartaa's run_config) + a DONE json with
   `model` = file name, `model_sha256` = its sha, `epochs` 16 (`train_studies` only if B_MODEL expect_train is set) into
   `s6entry/ds_<slug>/` + dataset-metadata.json (private, licence other), then the PowerShell staging fix:
   `$UP='...\s6entry\ds_<slug>'; New-Item -ItemType Directory -Force (Join-Path "$env:TEMP\.kaggle\uploads" ($UP -replace ':', '_')) | Out-Null; cmd /c "kaggle datasets create -p `"$UP`""`
   Poll `kaggle datasets files eeshsaxena/<slug>` until the 3 files list.
6. Build on the CURRENT team best kernel dir (by the R1 reads: S6F-J = jpeg/overlay/r2/kernel_v224flips6fj; R6 = r6entry/kernel_v224r6flips6fj;
   KC1 = kcstage/kernel_v224s6fkcj):
   `py -3.12 -I session_2026-10-15/s6entry/patch_s6_model.py --base <best kernel dir> --tag <tag ending in j> --ds eeshsaxena/<slug> --model-file E_prior_s4243_fsoup.pt --model-sha <sha> --run-config <rc name> --done <done name> --expect-epochs 16 --expect-train none`
7. Commit + gate: `bash session_2026-10-15/s6entry/gpu_commit_s6m.sh <tag> <the best's commit out dir>` -> `S6M_CHECK PASS n/n`.
8. Serve parity (box): copy `s6entry/out_<tag>_v1/s6f/B/B_probs.csv` to the box, then
   `bash ~/s6seed/parity/run_test3_parity.sh <entry .pt> <copied B_probs> entry_<tag>` -> `S6_TEST3_PARITY PASS`.
9. PREREG amendment (entry form, tag, gates, shas) BEFORE the submit; submit via the python API (kernel version from the push log).
