# RSNA Knee: GPU help from a teammate (3 Kaggle training runs, ~28 GPU hours)

Thanks for lending your GPU quota! This pack trains a **second seed (1044)** of our best model, the "TTR mix50"
ConvNeXt fleet, on all 5 folds. It runs as 3 Kaggle notebooks, about 28 of your 30 weekly GPU hours. Our current
best (0.946) is a 5-fold set of this exact recipe with seed 1043; a second seed makes the final ensemble stronger.

**Rule check:** you must be on our competition team ("Eesh saxena") before you use any of this. Competition rules
forbid sharing code or data outside a team.

---

## 0. Before you start (Eesh does steps 0.1-0.2, you do 0.3)

1. **Eesh shares the 3 private inputs with your Kaggle account.** On each page: Settings / Sharing -> add your
   username (view access is enough):
   - dataset `eeshsaxena/rsna-knee-fleet-train` (~120 MB: code, labels, ConvNeXt weights, timm wheel);
   - notebook `eeshsaxena/rsna-fleet-prep-s0` (its output = half of the preprocessed image store);
   - notebook `eeshsaxena/rsna-fleet-prep-s1` (the other half).
2. Eesh checks that nothing else is running on Kaggle under his account. Not required, but it avoids confusion.
3. **You:** open the competition page once and make sure you have accepted the rules and are on the team.
   Then check your quota on kaggle.com -> your profile -> Settings: it must show **>= 29 h** GPU available.

## 1. The 3 runs

| notebook file | trains | planned run time | hard stop | can run at the same time as |
|---|---|---|---|---|
| `rsna_train_ttr_s1044_f01.ipynb` | folds 0 + 1 (one per T4) | ~10.4 h | 11.4 h | f23 |
| `rsna_train_ttr_s1044_f23.ipynb` | folds 2 + 3 (one per T4) | ~10.4 h | 11.4 h | f01 |
| `rsna_train_ttr_s1044_f4.ipynb`  | fold 4 | ~7.1 h | 7.8 h | anything |

- Kaggle allows **2 GPU sessions at once** per account.
- So: start **f01 and f23 together** (done in ~10.5 h), then **f4** (done ~7 h later).
- Total ~28 GPU hours, about 18 h of wall time.
- Each notebook sizes its epoch count by itself, to finish inside its time budget: about 22 epochs for f01 and
  f23, about 14-15 for f4.

## 2. How to run one notebook (web UI, easiest)

Repeat for each of the 3 files:

1. kaggle.com -> **Create -> New Notebook**.
2. **File -> Import Notebook** -> upload the `.ipynb` file.
3. Right panel, **Session options**:
   - **Accelerator = GPU T4 x2** (must be T4 x2, it trains 2 folds in parallel);
   - **Internet = ON** (only a fallback; the weights are bundled);
   - Persistence: no change needed.
4. Right panel, **+ Add Input**:
   - Datasets: `eeshsaxena/rsna-knee-fleet-train`
   - Notebooks: `eeshsaxena/rsna-fleet-prep-s0` and `eeshsaxena/rsna-fleet-prep-s1`. Add their **output**, so the
     tar files are visible.

   (If they do not show up in search, the sharing from step 0.1 has not reached your account yet.)
5. Rename the notebook to the file name (e.g. `rsna-train-ttr-s1044-f01`). Keep it **Private**.
6. **Save Version -> Save & Run All (Commit)**. Close the tab: it runs in the background. Do NOT use the
   interactive session for training.

**CLI alternative** (if you use the Kaggle API):
- Folder `cli/` has one subfolder per run, with the notebook and a `kernel-metadata.json`.
- Replace `YOUR_KAGGLE_USERNAME` in each json, then run:
  ```
  kaggle kernels push -p cli/rsna-train-ttr-s1044-f01
  kaggle kernels push -p cli/rsna-train-ttr-s1044-f23
  # later, when one finishes:
  kaggle kernels push -p cli/rsna-train-ttr-s1044-f4
  ```
- Check progress with `kaggle kernels status YOUR_KAGGLE_USERNAME/rsna-train-ttr-s1044-f01`.

## 3. What happens inside, and what can go wrong

The notebook checks everything **before** it trains, so a problem costs only minutes, not hours:
1. inputs present + every file's sha256;
2. environment (2 GPUs, disk);
3. it unpacks the image store (~12 GB) and verifies it byte-for-byte;
4. parity checks against our training server (pixel / tensor / model hashes);
5. a 25-minute speed probe of both trainers;
6. training with a hard time wall, then a `run_receipt.json` listing every output file with its sha256.

| you see | meaning / what to do |
|---|---|
| status **ERROR within the first ~40 min** | an input check failed: open the log (Output / Logs) and look for the last `ABORT` line. Usually an input is missing (step 2.4) or not yet shared. Fix it and run again: only minutes were used |
| status ERROR with an **empty log** right at the start | Kaggle refused the session (it happened to us once). Wait a few minutes and commit again |
| a `WARNING ... E < 16` line (f4 only) | expected for f4 (shorter budget). Fine |
| status COMPLETE | done: go to section 4 |

Please do **not** stop a run early unless it has errored. A stopped run loses everything not yet written.

## 4. When a run is COMPLETE: get the outputs to Eesh

Either option works:
- **(a) easiest:** on the notebook page, Settings / Sharing -> add **`eeshsaxena`** as a collaborator (view). Eesh then
  downloads the outputs himself with the Kaggle CLI.
- **(b)** download from the notebook's **Output** tab and send the files (Drive etc.):
  - the folders `fF_s1044_ttr_mix50/` (two `*_best.pt` checkpoints ~108 MB each + two `oof_*.csv` per fold; the
    `epochs/` subfolder is NOT needed);
  - `fF_s1044_ttr_mix50.log`;
  - `run_receipt.json` (needed: it carries the sha256 of every file).

Then tell Eesh which runs finished. That's it, thank you!

---

### For Eesh (after the runs finish)
- Fetch the outputs. Skip `epochs/*.pt` and anything under a vendored env dir:
  ```
  kaggle kernels output FRIEND/rsna-train-ttr-s1044-fXX -p <dir> --page-size 200 --file-pattern '^(?!.*epochs/.*\.pt$)'
  ```
- Verify with `kaggle_train/verify_outputs.py <dir>`.
- Copy to the box as `~/fleet_runs/fF_s1044_ttr_mix50/` + `.log`, plus each receipt (one receipt per run; see
  `kaggle_train/RUNBOOK.md` sec 5-6).
- Collect with the k5/k6 picker using `--seed 1044` and the matching `--kaggle_receipt`. Then a K-build of the
  10-member mix50 set (seeds 1043 + 1044), blended at W .50.
- Expected value (MODEL): a 2nd seed adds about +.0005-.001 to the fleet solo, a small but positive gain.
- Build source of these notebooks: `SP/kaggle_train_s1044_{f01,f23,f4}` (copies of the f34 kit; only FOLDS / SEED
  / time budget / slug differ). The dataset manifests were rebuilt byte-identical, so no dataset re-upload is
  needed.
