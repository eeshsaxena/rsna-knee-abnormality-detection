# TODO: frac75 seed 1044 folds 1 + 2 (`rsna-train-ttr-frac75-s1044-f12`, teammate kmrsuman)

Three parts, in order:
- **A** = Eesh before the handoff.
- **B** = the teammate (details in `README_TEAMMATE.md`).
- **C** = Eesh once it runs and after it is COMPLETE (details in `RUNBOOK_EESH.md`, the section numbers below).

## A. Eesh, before the handoff

1. [ ] Confirm the teammate is still on competition team **"Eesh saxena"** (Team tab) and has accepted the rules.
2. [ ] Confirm the box has no `~/fleet_runs/f[12]_s1044_ttr_frac75` dirs yet (RUNBOOK sec 1 step 3).
3. [ ] Confirm the 4 inputs are still shared with the teammate, **Can view** (they were for the 10-07 frac75 runs; all
       LIVE, nothing to create or version):
   - [ ] dataset `eeshsaxena/rsna-knee-fleet-train`
   - [ ] dataset `eeshsaxena/rsna-knee-frac75-labels`
   - [ ] notebook `eeshsaxena/rsna-fleet-prep-s0`
   - [ ] notebook `eeshsaxena/rsna-fleet-prep-s1`
4. [ ] Send `kmrsuman_s1044_A.zip`. (Box dry tests: RUNBOOK sec 9.)
5. [ ] Say **GO**.

## B. Teammate

1. [ ] You are on team "Eesh saxena" and have accepted the competition rules.
2. [ ] `kaggle quota` shows **>= 12 h** of GPU time free (weekly reset: Saturday 05:30 IST).
3. [ ] Wait for Eesh's **GO**.
4. [ ] Put your username into `cli/rsna-train-ttr-frac75-s1044-f12/kernel-metadata.json`, then push with GPU **T4 x2**:
       `kaggle kernels push -p cli/rsna-train-ttr-frac75-s1044-f12 --timeout 43200`
       (or the web UI, README sec 2b, with all 4 inputs added).
5. [ ] Keep the slug exactly **`rsna-train-ttr-frac75-s1044-f12`**.
6. [ ] Share the notebook with **`eeshsaxena`**, **Can view**.
7. [ ] Do NOT stop it. It ends itself within ~11.4 h.
8. [ ] ERROR at once with an **EMPTY log** = Kaggle refused the session: report it to Eesh with a `kaggle quota`
       screenshot. Do NOT retry.
9. [ ] ERROR within ~30 min: send Eesh the last `ABORT` line.
10. [ ] COMPLETE: tell Eesh.

## C. Eesh, once Running and after COMPLETE

1. [ ] As soon as it shows **Running**: write down T0 and make the EMPTY box placeholders `~/fleet_runs/f1_s1044_ttr_frac75` and `~/fleet_runs/f2_s1044_ttr_frac75`
       (RUNBOOK sec 3).
2. [ ] If it fails or is refused: `rmdir` its empty placeholders (RUNBOOK sec 7).
3. [ ] After COMPLETE, fetch with `--page-size 200 --file-pattern '^(?!_[a-z0-9]+_env/)(?!.*epochs/.*\.pt$)'`
       (RUNBOOK sec 5).
4. [ ] `verify_outputs.py <out dir>`: expect `status DONE`, E 21 (or 22), `0 mismatch, 12 absent`, exit 0.
5. [ ] Copy into the placeholders and the receipt as `~/fleet_runs/kaggle_frac75_s1044_f12_run_receipt.json`; run `verify_outputs.py` on the box
       (RUNBOOK sec 6).
