# TODO: frac75 Kaggle run (`rsna-train-ttr-frac75-f01`)

Three parts, in order:
- **A** = Eesh before the handoff.
- **B** = the teammate (details in `README_TEAMMATE.md`).
- **C** = Eesh after the run is COMPLETE (details in `RUNBOOK_EESH.md`, the section numbers below).

## A. Eesh, before the handoff

1. [ ] Confirm the teammate is ALREADY on competition team **"Eesh saxena"** (Team tab). Share nothing before this.
2. [ ] Confirm the box has not started its own frac75 fold-0 probe (RUNBOOK sec 1 step 3). If it has, do not hand out
       this notebook; ask Claude for a folds 1 + 2 build.
3. [ ] Create the labels dataset `eeshsaxena/rsna-knee-frac75-labels` from `kaggle_frac75\ds_labels\` with the Windows
       staging-dir fix (RUNBOOK sec 2). Do NOT version `rsna-knee-fleet-train`.
4. [ ] Wait until `kaggle datasets status eeshsaxena/rsna-knee-frac75-labels` says **ready**. `kaggle datasets files`
       must list exactly 1 file, `labels_frac75_logit.csv`.
5. [ ] Share the inputs with the teammate (Can view). That is 2 datasets + 2 prep notebooks:
   - [ ] dataset `eeshsaxena/rsna-knee-fleet-train`
   - [ ] dataset `eeshsaxena/rsna-knee-frac75-labels`
   - [ ] notebook `eeshsaxena/rsna-fleet-prep-s0`
   - [ ] notebook `eeshsaxena/rsna-fleet-prep-s1`
6. [ ] Send `friend_frac75_pack.zip`. (Box dry tests all PASS, confirmed 2026-10-07 16:02: 74/74, 8/8, 19/19, 22/22,
       10/10; RUNBOOK sec 11.)

## B. Teammate

1. [ ] You are on team "Eesh saxena" and have accepted the competition rules.
2. [ ] Your weekly GPU quota shows **>= 10 h** available, and no other GPU notebook is running on your account.
3. [ ] Upload `rsna_train_ttr_frac75_f01.ipynb` (File -> Import Notebook). With the CLI instead, push
       `cli/rsna-train-ttr-frac75-f01` after putting your username into `kernel-metadata.json`.
4. [ ] Settings: **Accelerator GPU T4 x2**, **Internet ON**, notebook **Private**.
5. [ ] Add all 4 inputs: the 2 datasets and the OUTPUT of the 2 prep notebooks (README sec 2.4).
6. [ ] Keep the name/slug exactly **`rsna-train-ttr-frac75-f01`**.
7. [ ] **Save Version -> Save & Run All (Commit)**. Keep the laptop awake until it shows **Running**.
8. [ ] Share the notebook with **`eeshsaxena`**, **Can view**.
9. [ ] Do NOT stop it. It ends itself within ~9.3 h.
10. [ ] ERROR at once with an **EMPTY log** = Kaggle refused the session: report it to Eesh with a quota screenshot.
        Do NOT retry blindly.
11. [ ] ERROR within ~30 min: send Eesh the last `ABORT` line.
12. [ ] COMPLETE: tell Eesh.

## C. Eesh, after COMPLETE

1. [ ] As soon as the run shows **Running**: write down T0, and make the 2 EMPTY placeholders on the box
       (RUNBOOK sec 4) so the V100 queue does not also train fold 0 / fold 1.
2. [ ] Fetch with `--page-size 200 --file-pattern '^(?!_[a-z0-9]+_env/)(?!.*epochs/.*\.pt$)'` (RUNBOOK sec 6).
3. [ ] `verify_outputs.py <out dir>`: expect `status DONE`, E 17 (or 18), `0 mismatch, 12 absent`, exit 0.
4. [ ] Copy to the box into the empty placeholders, plus the receipt as `~/fleet_runs/kaggle_frac75_f01_run_receipt.json`.
       Then run `verify_outputs.py` on the box (RUNBOOK sec 7).
5. [ ] Score the gate: `~/kt_frac75/box/score_frac75_kaggle.py` (CPU, ~15 min, RUNBOOK sec 7a). Read the line
       `KAGGLE ADDENDUM READ`.
6. [ ] Decide folds 2-4:
   - [ ] **PASS:** write the queue verdict with `F75_SCORER=... frac75_gate.sh` (sec 7b). The V100 then trains folds
         2-4 (~12 h), then K9.
   - [ ] **FAIL:** write the verdict the same way. Folds 2-4 are skipped; close target re-mixing.
   - [ ] **INCONCLUSIVE-NARROW:** no verdict from Kaggle. Choose: accept FAIL, or the 24-epoch box tie-break (sec 7c).
7. [ ] If the run failed or was refused: remove the empty placeholders and, if needed, restore Q1 / Q2 f1 (RUNBOOK sec 8).
