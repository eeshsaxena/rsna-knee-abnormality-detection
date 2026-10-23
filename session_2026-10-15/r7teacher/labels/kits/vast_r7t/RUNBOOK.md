> **LOCKED 10-14 18:10 IST: do NOT launch this kit for T_b.** T_b rents via `session_2026-10-15/rental_student/launch.sh` (one rental at a time, $15 total cap); `vast_launch.sh` refuses while `LOCKED_USE_RENTAL_STUDENT` exists. The NEXT text printed by `on_r7oof.sh` (frozen, not edited) still names this kit: ignore that line.

# vast_r7t: rented RSNA-only run of the R7-teacher student (PREREG_TEACHER sec 5). NOT LAUNCHED.

Kit = port of `session_2026-10-09/vastS1044` (the TTR224 fold kit: box push with OAI-path refusal + md5 parity, parity.py,
deadline-aware lanes, laptop watcher, on-instance guard) plus the 10-13 Raptor lesson: a BOX-side fetch loop
(`remote/box_fetch_loop.sh`) so a dead laptop / Claude session cannot lose the folds. No em dashes.

## What it trains
- `remote/kit.env` + `remote/argv_template.txt`: T_b student, seed 1047, folds 0-4, out `f{F}_s1047_ttr_r7tb`, labels
  `~/r7t_labels/real/labels_r7t_b.csv`. `box_ops.sh argv-check` proves the argv equals the box sheet line
  (`~/r7t_labels/sheets/queue_r7tb.txt`) token for token (MEASURED 10-14: 5/5 IDENTICAL, 38 tokens).
- T_a instead: edit kit.env (RUNTAG ttr_r7ta, SEED 1048, TABLE labels_r7t_a.csv, SHEET queue_r7ta.txt, QGATE ..._a.verdict)
  and argv_template.txt (labels_r7t_a.csv, 1048, f@F@_s1048_ttr_r7ta). One table per rental (keep separate state.env).

## RSNA-only guarantees
- The box pushes ONLY: `~/fast_build/code2`, the gated label table, folds5 / gold74 / v4 csvs, the HF ConvNeXt-T cache,
  `~/fleet_store` (4,407 RSNA npz, 11.66 GB MEASURED). `box_push.sh` refuses any source resolving to `*oai*` /
  `union_store`, symlinks, non-npz store files, a SIM_ table, a table that mentions "oai", a table without 4,408 lines, or
  a table whose `labels_gate` verdict is not PASS. The label table is OAI-free by construction (fleet component = sets
  C, D, E, E2, D2; DT and L13 excluded).
- The launch itself refuses unless `bops verdict` (= `~/r7t_labels/real/labels_gate_labels_r7t_b.verdict`) starts with PASS.

## Preconditions (in order)
1. R7-OOF file exists and is copied to the box as `~/r7t_labels/real/R7OOF.csv` (job 1 + Eesh's OK for the Kaggle run).
2. REAL build (box CPU, seconds):
   `cd ~/r7t_labels && ~/miniconda3/envs/rsna/bin/python build_teacher_labels.py --teacher real/R7OOF.csv --teacher_tag R7OOF
    --fleet_oof fleet/oof_fleetOF_mean.csv --fleet_gold fleet/gold_fleetOF_mean.csv --out_dir ~/r7t_labels/real
    --prereg ~/r7t_labels/PREREG_TEACHER.md --prereg_sha ~/r7t_labels/PREREG_TEACHER.sha256`
   then `bash ~/r7t_labels/labels_gate.sh labels_r7t_b.csv` (prints PASS / FAIL; the kit needs PASS).
3. Eesh: approve the spend; Vast 2FA session (`vastai tfa send-email`, then `vastai tfa login -t email -s <secret> -c <code>`).
   Credit was $17.23 on 10-14 (MEASURED); kit cap $15 (`CAP_USD`), floor $2.

## Launch (Git Bash, this folder; lab network for the box)
1. `bash vast_watch.sh` (Bash run_in_background: 2FA keepalive, becomes the money watcher at launch)
2. `bash vast_launch.sh --search` (read only; ranks offers for 5 folds; prefer a verified 4 x RTX 4090, DC first)
3. `bash vast_launch.sh --dry-run <offer>` then `bash vast_launch.sh <offer>` (rents; guard armed with the hard deadline)
4. `bash vast_setup.sh` (S1 facts, S2 pins = box versions, S3 kit + plan, S4 box push + md5 parity, S5 parity.py box vs
   instance, S6 round-1 placeholders on the box)
5. `bash vast_train.sh` (one lane per GPU, folds round-robin; also starts the BOX fetch loop)
Finish: the box loop pulls + verifies each fold into `~/fleet_runs/f{F}_s1047_ttr_r7tb` and writes FINAL_FETCH_OK; the
on-instance guard destroys 15 min later; the laptop watcher also fetches / destroys (backup). Manual kill switch:
`bash vast_destroy.sh --yes`.

## Estimates (MODEL; replace with the first measured epoch)
| Plan | Rounds | Wall | Cost |
|---|---|---|---|
| 4 x 4090, 5 folds (T_b) | 2 | 4.1 h mid / 4.9 h worst | $7.5-7.7 mid / $9.1-9.3 worst (at $1.83-1.90/h, 10-13 offers MEASURED) |
| 2 x 4090, 5 folds | 3 | 5.7 / 7.0 h | $5.5-9.4 mid (Taiwan $0.94/h to Maryland $1.53/h, 10-13 offers) |
| 4 x 4090, 10 folds (T_b + T_a, two kit copies) | 3 | 5.7 / 7.0 h | $10.6 / $12.9 |
Fold time = 3.52 h V100 (MEASURED) / 2.1 (4090 speed ratio, MODEL; 3090 1.47-1.57 x MEASURED 10-09) = 1.68 h; worst x 1.25.
Setup 0.45 h (store push 11.66 GB box -> instance; pip), fetch 0.25 h. Offers JSON from 10-13 in `logs/offers_4090_*.json`
(priced with 200 GB storage, so slightly high for this 80 GB plan).

## Validation done 10-14 (no money spent, no Vast API call)
- `bash -n` on every script; `py_compile` on offers_rank.py / parity.py.
- Box helpers synced to `~/vastR7T` and run: `paths` (only the REAL table missing, as expected), `argv-check` 5/5
  IDENTICAL, `ph-status` (no placeholders), `verdict` MISSING (expected), `keyinfo` (dedicated key present).
- parity.py on the box with the SIM table: fold counts 3466/867 x3, 3467/866 x2, model_sha f141aeed136dfc01, store 4,407.
- ckpt-args reference check: set D f3, D3 f1 and L13 f0 compared with the reference (set D f2) differ only in the skipped
  keys (fold / epochs / save_epochs / grad_ckpt / seed / train_labels): 0 unexpected diffs.
- offers_rank.py on the saved 10-13 offers: table above.
