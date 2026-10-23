# TODO: frac75 folds 2-4 + fold 0 seed 1044 (`rsna-train-ttr-frac75-f23` + `rsna-train-ttr-frac75-f4x`)

Three parts, in order:
- **A** = Eesh before the handoff.
- **B** = the teammate (details in `README_TEAMMATE.md`).
- **C** = Eesh once they run and after they are COMPLETE (details in `RUNBOOK_EESH.md`, the section numbers below).

## A. Eesh, before the handoff

1. [ ] Confirm the teammate is ALREADY on competition team **"Eesh saxena"** (Team tab) and has accepted the rules.
       Share nothing before this.
2. [ ] Confirm the box has not started Q2 f2-f4 and has no `f[234]_s1043_ttr_frac75` / `f0_s1044_ttr_frac75` dirs
       (RUNBOOK sec 1 step 4), and the frac75 verdict is not already FAIL.
3. [ ] Share the inputs with the teammate (Can view). That is 2 datasets + 2 prep notebooks (all LIVE, nothing to create):
   - [ ] dataset `eeshsaxena/rsna-knee-fleet-train`
   - [ ] dataset `eeshsaxena/rsna-knee-frac75-labels`
   - [ ] notebook `eeshsaxena/rsna-fleet-prep-s0`
   - [ ] notebook `eeshsaxena/rsna-fleet-prep-s1`
4. [ ] Send `friend2_frac75b_pack.zip`. (Box dry tests: RUNBOOK sec 9.)
5. [ ] Say **GO**.

## B. Teammate

1. [ ] You are on team "Eesh saxena" and have accepted the competition rules.
2. [ ] `kaggle quota` shows **>= 24 h** available, and no other GPU notebook is running on your account.
3. [ ] Wait for Eesh's **GO**.
4. [ ] Push BOTH notebooks at the same time, each GPU T4 x2, Internet ON, Private:
       `kaggle kernels push -p cli/rsna-train-ttr-frac75-f23 --timeout 43200` and
       `kaggle kernels push -p cli/rsna-train-ttr-frac75-f4x --timeout 43200` (your username in both
       `kernel-metadata.json` first). Or the web UI (README sec 2b) with all 4 inputs added.
5. [ ] Keep the slugs exactly **`rsna-train-ttr-frac75-f23`** and **`rsna-train-ttr-frac75-f4x`**.
6. [ ] Share both notebooks with **`eeshsaxena`**, **Can view**.
7. [ ] Do NOT stop them. Each ends itself within ~11.4 h.
8. [ ] ERROR at once with an **EMPTY log** = Kaggle refused the session: report it to Eesh with a `kaggle quota`
       screenshot. Do NOT retry blindly.
9. [ ] ERROR within ~30 min: send Eesh the last `ABORT` line.
10. [ ] COMPLETE: tell Eesh.

## C. Eesh, once Running and after COMPLETE

1. [ ] As soon as both show **Running**: write down both T0s and make the EMPTY box placeholders
       `~/fleet_runs/f{2,3,4}_s1043_ttr_frac75` (RUNBOOK sec 3) so the box queue skips Q2 f2-f4. Do it before any PASS
       verdict is written.
2. [ ] If a run fails or is refused: remove that notebook's empty placeholders and, if the runner already refused the
       line, restore it from `~/kt_frac75b/box/queue_lines_f234.txt` (RUNBOOK sec 7).
3. [ ] After COMPLETE, fetch each notebook with `--page-size 200 --file-pattern '^(?!_[a-z0-9]+_env/)(?!.*epochs/.*\.pt$)'`
       (RUNBOOK sec 5).
4. [ ] `verify_outputs.py <out dir>` for each: expect `status DONE`, E 21 (or 22), `0 mismatch, 12 absent`, exit 0.
5. [ ] Copy into the placeholders (+ `~/fleet_runs/f0_s1044_ttr_frac75`) and the receipts as
       `~/fleet_runs/kaggle_frac75_{f23,f4x}_run_receipt.json`; run `verify_outputs.py` on the box (RUNBOOK sec 6).
6. [ ] Use them only if the frac75 gate PASSED (K9 frac75 members f2-f4; f0 s1044 = extra seed). On FAIL they stay as
       a record.
