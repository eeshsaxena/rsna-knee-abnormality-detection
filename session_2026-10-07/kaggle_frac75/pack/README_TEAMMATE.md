# RSNA Knee: one Kaggle training run on your GPU quota (~9 h)

Thanks for lending your GPU quota! This pack trains one experiment of our model, called **"frac75"**:
- the same ConvNeXt model and recipe as our best submission;
- with a re-mixed training target for the Fracture label.

It is **1 Kaggle notebook** that trains 2 folds at once (one per T4 GPU). It uses about **9 h** of your weekly GPU
quota and stops itself before 9.3 h. Its fold-0 result decides whether we train the remaining 3 folds on our own
server.

**Rule check:** you must already be on our competition team ("Eesh saxena") before anything is shared with you.
Competition rules forbid sharing code or data outside a team. If you are not on the team yet, stop here and tell Eesh.

`TODO.md` is the step-by-step checklist (your part is section B). `RUNBOOK_EESH.md` is Eesh's own runbook: you can
ignore it.

---

## 0. Before you start

1. **Eesh shares 4 private inputs with your Kaggle account** (view access):
   - dataset `eeshsaxena/rsna-knee-fleet-train` (~120 MB: code, labels, ConvNeXt weights, timm wheel);
   - dataset `eeshsaxena/rsna-knee-frac75-labels` (new, 1.3 MB: the frac75 training labels);
   - notebook `eeshsaxena/rsna-fleet-prep-s0` (its output = half of the preprocessed image store);
   - notebook `eeshsaxena/rsna-fleet-prep-s1` (the other half).
2. **You:** open the competition page once and check that you have accepted the rules and are on the team.
3. **You:** check your GPU quota on kaggle.com (your profile, Settings / quota). It must show **at least 10 h**
   available. If it shows less, do not start: tell Eesh. A run that hits the quota limit is killed and its outputs are
   lost.
4. **No other GPU notebook may be running on your account** while this one runs.

## 1. The run

| notebook file | trains | planned run time | hard stop | quota needed |
|---|---|---|---|---|
| `rsna_train_ttr_frac75_f01.ipynb` | folds 0 + 1 (one per T4) | ~8.1-8.9 h | 9.2 h (ends < 9.3 h) | >= 10 h available |

- It sizes its own epoch count to its time budget (about 17 epochs).
- Kaggle adds a few minutes at the end to save the outputs (~1.8 GB).

## 2. How to run it (web UI, easiest)

1. kaggle.com, **Create -> New Notebook**.
2. **File -> Import Notebook**, then upload `rsna_train_ttr_frac75_f01.ipynb`.
3. Right panel, **Session options**:
   - **Accelerator = GPU T4 x2** (it must be T4 x2: it trains 2 folds in parallel);
   - **Internet = ON** (only a fallback; the weights are bundled);
   - Persistence: no change needed.
4. Right panel, **+ Add Input**, all four:
   - Datasets: `eeshsaxena/rsna-knee-fleet-train` and `eeshsaxena/rsna-knee-frac75-labels`.
   - Notebooks: `eeshsaxena/rsna-fleet-prep-s0` and `eeshsaxena/rsna-fleet-prep-s1`. Add their **output**, so the tar
     files are visible.

   If an input does not show up in search, the sharing from step 0.1 has not reached your account yet.
5. Rename the notebook to exactly **`rsna-train-ttr-frac75-f01`**. Keep this name: Eesh fetches the outputs by it. Keep
   it **Private**.
6. **Save Version -> Save & Run All (Commit)**.
   - Wait until the version shows **Running**, and keep your laptop awake until then. After that it runs on Kaggle's
     servers, and you can close the tab or the laptop.
   - Do NOT train in the interactive editor session: it would burn quota twice.
7. **Share it with Eesh right away:** on the notebook page, Settings / Sharing -> add **`eeshsaxena`** with
   **Can view**. Then Eesh can follow the status and download the outputs directly.

**CLI alternative** (if you use the Kaggle API):
- Folder `cli/rsna-train-ttr-frac75-f01/` has the notebook and a `kernel-metadata.json`.
- Replace `YOUR_KAGGLE_USERNAME` in the json (keep the slug `rsna-train-ttr-frac75-f01`), then run:
  ```
  kaggle kernels push -p cli/rsna-train-ttr-frac75-f01
  kaggle kernels status YOUR_KAGGLE_USERNAME/rsna-train-ttr-frac75-f01
  ```
- Then share it with `eeshsaxena` (Can view) in the web UI, as in step 7.

## 3. What happens inside, and what can go wrong

The notebook checks everything **before** it trains, so a problem costs minutes, not hours:
1. inputs present + every file's sha256 (including the new labels file);
2. environment (2 GPUs, disk);
3. it unpacks the image store (~12 GB) and verifies it byte for byte;
4. parity checks against our training server (pixel / tensor / model hashes);
5. a short speed probe of both trainers (~6 min);
6. training with a hard time wall, then a `run_receipt.json` listing every output file with its sha256.

| you see | meaning / what to do |
|---|---|
| status **ERROR at once with an EMPTY log** | Kaggle refused the session (usually quota, or another GPU session running). **Do not retry blindly.** Send Eesh a screenshot of your quota bar and wait for Eesh's answer |
| status ERROR within the first ~30 min | an input check failed: open the log (Output / Logs) and send Eesh the last line containing `ABORT`. Usually an input is missing (step 2.4) or not shared yet. Re-run only after the fix, and only if your quota still shows >= 9.5 h |
| a `WARNING ... E = ... < 16` line | the GPUs were slower than expected. Let it run and mention it to Eesh |
| status COMPLETE | done: section 4 |

**Please do not stop the run**, not even when it looks slow: it stops itself before the time limit and keeps
everything it has trained. A run stopped by hand loses its outputs.

## 4. When it is COMPLETE

- Tell Eesh. If you shared the notebook with `eeshsaxena` (step 2.7), that is all: Eesh downloads the outputs directly.
- If sharing did not work, download from the notebook's **Output** tab and send them to Eesh (Drive etc.):
  - the folders `f0_s1043_ttr_frac75/` and `f1_s1043_ttr_frac75/`: two `*_best.pt` checkpoints (~113 MB each) + the
    `oof_*.csv` files per fold, plus the `epochs/oof_*.csv` files (the `epochs/*.pt` files are NOT needed);
  - `f0_s1043_ttr_frac75.log` and `f1_s1043_ttr_frac75.log`;
  - `run_receipt.json` (needed: it carries the sha256 of every file).

That's it, thank you!
