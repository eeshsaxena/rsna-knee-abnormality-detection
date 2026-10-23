# RSNA Knee: one Kaggle training run on your GPU quota (~10-11 h of 30 h)

Thanks for lending your GPU quota again! This pack trains **folds 0 + 1** of our **"frac75"** model with a third
random seed (seed 1045):
- the same ConvNeXt model, recipe and notebook code as the frac75 runs that completed on Kaggle on 10-07/08;
- only the folds and the seed are different.

It is **1 Kaggle notebook**, `rsna-train-ttr-frac75-s1045-f01`. It trains 2 folds at once (one per T4 GPU), uses about **10-11 h** of your
weekly GPU quota and stops itself before 11.5 h.

**Rule check:** you must be on our competition team ("Eesh saxena") and have accepted the competition rules. Competition
rules forbid sharing code or data outside a team. If anything changed there, stop here and tell Eesh.

**Please START ONLY WHEN EESH SAYS GO.**

`TODO.md` is the step-by-step checklist (your part is section B). `RUNBOOK_EESH.md` is Eesh's own runbook: you can
ignore it.

---

## 0. Before you start

1. **Inputs:** the same 4 private inputs as our earlier frac75 runs (view access, shared by Eesh):
   - dataset `eeshsaxena/rsna-knee-fleet-train` (~120 MB: code, labels, ConvNeXt weights, timm wheel);
   - dataset `eeshsaxena/rsna-knee-frac75-labels` (1.3 MB: the frac75 training labels);
   - notebook `eeshsaxena/rsna-fleet-prep-s0` (its output = half of the preprocessed image store);
   - notebook `eeshsaxena/rsna-fleet-prep-s1` (the other half).

   If one of them does not show up for your account, tell Eesh before you push.
2. **Check your GPU quota:** run `kaggle quota` (or on kaggle.com: your profile, Settings / quota). It must show **at
   least 12 h** of GPU time free. The weekly quota resets every Saturday at 05:30 IST. If it shows less than 12 h, do not start:
   tell Eesh. A run that hits the quota limit is killed and its outputs are lost.
3. This run needs **one** GPU session (T4 x2). Do not start other GPU work while it runs unless your quota still has
   room for this run's full 12 h on top of it.

## 1. The run

| notebook file | slug (keep exactly) | trains | planned run time | hard stop |
|---|---|---|---|---|
| `rsna_train_ttr_frac75_s1045_f01.ipynb` | `rsna-train-ttr-frac75-s1045-f01` | folds 0 + 1 (seed 1045) | ~10.0-10.7 h | ends < 11.5 h |

- It sizes its own epoch count to its time budget (about 21 epochs).
- Kaggle adds a few minutes at the end to save the outputs (~1.8 GB).

## 2. How to run it: CLI (preferred: it sets the 12 h time cap)

Folder `cli/rsna-train-ttr-frac75-s1045-f01/` has the notebook and a `kernel-metadata.json` (GPU **T4 x2**, Internet ON, Private, all 4 inputs
already listed).

1. In `cli/rsna-train-ttr-frac75-s1045-f01/kernel-metadata.json`, replace `YOUR_KAGGLE_USERNAME` with your Kaggle username. Change nothing
   else (**keep the slug** `rsna-train-ttr-frac75-s1045-f01`).
2. When Eesh says GO, push it:
   ```
   kaggle kernels push -p cli/rsna-train-ttr-frac75-s1045-f01 --timeout 43200
   kaggle kernels status YOUR_KAGGLE_USERNAME/rsna-train-ttr-frac75-s1045-f01
   ```
   - `--timeout 43200` (12 h, the Kaggle maximum) is required: Kaggle refuses a session without it when less than
     ~12 h of quota is free.
   - If the CLI says `--timeout` is unknown, update it first: `pip install -U kaggle`.
3. **Share it with Eesh right away:** on the notebook page, Settings / Sharing -> add **`eeshsaxena`** with
   **Can view**. Then Eesh can follow the status and download the outputs directly.

## 2b. Web UI alternative (only if you cannot use the CLI)

1. kaggle.com, **Create -> New Notebook**.
2. **File -> Import Notebook**, then upload `rsna_train_ttr_frac75_s1045_f01.ipynb`.
3. Right panel, **Session options**: **Accelerator = GPU T4 x2** (it must be T4 x2), **Internet = ON**.
4. Right panel, **+ Add Input**, all four: datasets `eeshsaxena/rsna-knee-fleet-train` and
   `eeshsaxena/rsna-knee-frac75-labels`; notebooks `eeshsaxena/rsna-fleet-prep-s0` and `eeshsaxena/rsna-fleet-prep-s1`
   (add their **output**, so the tar files are visible).
5. Rename the notebook to exactly `rsna-train-ttr-frac75-s1045-f01`. Keep it **Private**.
6. **Save Version -> Save & Run All (Commit)**. Wait until the version shows **Running** (keep your laptop awake until
   then); after that it runs on Kaggle's servers. Do NOT train in the interactive editor session: it would burn quota
   twice.
7. Share it with **`eeshsaxena`**, **Can view** (Settings / Sharing).

## 3. What happens inside, and what can go wrong

The notebook checks everything **before** it trains, so a problem costs minutes, not hours:
1. inputs present + every file's sha256 (including the labels file);
2. environment (2 GPUs, disk);
3. it unpacks the image store (~12 GB) and verifies it byte for byte;
4. parity checks against our training server (pixel / tensor / model hashes);
5. a short speed probe of both trainers (~6 min);
6. training with a hard time wall, then a `run_receipt.json` listing every output file with its sha256.

| you see | meaning / what to do |
|---|---|
| status **ERROR at once with an EMPTY log** | Kaggle refused the session (usually quota, or another GPU session running). **Do not retry.** Report it to Eesh with a screenshot of `kaggle quota` (or the quota bar) and wait for Eesh's answer |
| status ERROR within the first ~30 min | an input check failed: open the log and send Eesh the last line containing `ABORT`. Usually an input is missing or not shared. Re-run only after Eesh says so, and only if your quota still shows >= 12 h free |
| a `WARNING ... E = ... < 16` line | the GPUs were slower than expected. Let it run and mention it to Eesh |
| status COMPLETE | done: section 4 |

**Please do not stop the run**, not even when it looks slow: it stops itself before the time limit and keeps everything
it has trained. A run stopped by hand loses its outputs.

## 4. When it is COMPLETE

- Tell Eesh. If you shared the notebook with `eeshsaxena`, that is all: Eesh downloads the outputs directly.
- If sharing did not work, download from the notebook's **Output** tab and send them to Eesh (Drive etc.):
  - the two run folders `f0_s1045_ttr_frac75/` + `f1_s1045_ttr_frac75/`: the `*_best.pt` checkpoints (~113 MB each), the `oof_*.csv` files and the
    `epochs/oof_*.csv` files (the `epochs/*.pt` files are NOT needed);
  - the two `.log` files next to them;
  - `run_receipt.json` (needed: it carries the sha256 of every file).

That's it, thank you!
