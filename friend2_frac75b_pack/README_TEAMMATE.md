# RSNA Knee: two Kaggle training runs on your GPU quota (~21 h of 30 h)

Thanks for lending your GPU quota! This pack trains one experiment of our model, called **"frac75"**:
- the same ConvNeXt model and recipe as our best submission;
- with a re-mixed training target for the Fracture label.

It is **2 Kaggle notebooks** that run **at the same time** (Kaggle allows 2 GPU sessions per account). Each one trains
2 folds at once (one per T4 GPU), uses about **10-11 h** of your weekly GPU quota and stops itself before 11.5 h.
Together they use about **21 h (at most 23 h)** of your 30 h.

**Rule check:** you must already be on our competition team ("Eesh saxena") and have accepted the competition rules
before anything is shared with you. Competition rules forbid sharing code or data outside a team. If you are not on the
team yet, stop here and tell Eesh.

**Please START ONLY WHEN EESH SAYS GO.**

`TODO.md` is the step-by-step checklist (your part is section B). `RUNBOOK_EESH.md` is Eesh's own runbook: you can
ignore it.

---

## 0. Before you start

1. **Eesh shares 4 private inputs with your Kaggle account** (view access):
   - dataset `eeshsaxena/rsna-knee-fleet-train` (~120 MB: code, labels, ConvNeXt weights, timm wheel);
   - dataset `eeshsaxena/rsna-knee-frac75-labels` (1.3 MB: the frac75 training labels);
   - notebook `eeshsaxena/rsna-fleet-prep-s0` (its output = half of the preprocessed image store);
   - notebook `eeshsaxena/rsna-fleet-prep-s1` (the other half).
2. **You:** open the competition page once and check that you have accepted the rules and are on the team.
3. **You:** check your GPU quota. With the Kaggle CLI: `kaggle quota` (or on kaggle.com: your profile, Settings /
   quota). It must show **at least 24 h** available. If it shows less, do not start: tell Eesh. A run that hits the
   quota limit is killed and its outputs are lost.
4. **No other GPU notebook may be running on your account** (these two need both GPU slots).

## 1. The two runs

| notebook file | slug (keep exactly) | trains | planned run time | hard stop |
|---|---|---|---|---|
| `rsna_train_ttr_frac75_f23.ipynb` | `rsna-train-ttr-frac75-f23` | folds 2 + 3 | ~10.0-10.7 h | ends < 11.5 h |
| `rsna_train_ttr_frac75_f4x.ipynb` | `rsna-train-ttr-frac75-f4x` | fold 4 + fold 0 (2nd seed) | ~10.0-10.7 h | ends < 11.5 h |

- Each sizes its own epoch count to its time budget (about 21 epochs).
- Kaggle adds a few minutes at the end to save the outputs (~1.8 GB per notebook).

## 2. How to run them: CLI (preferred: it sets the 12 h time cap)

Folders `cli/rsna-train-ttr-frac75-f23/` and `cli/rsna-train-ttr-frac75-f4x/` each have the notebook and a
`kernel-metadata.json` (GPU T4 x2, Internet ON, Private, all 4 inputs already listed).

1. In BOTH `kernel-metadata.json` files, replace `YOUR_KAGGLE_USERNAME` with your Kaggle username. Change nothing else
   (keep the slugs).
2. When Eesh says GO, push both, one right after the other:
   ```
   kaggle kernels push -p cli/rsna-train-ttr-frac75-f23 --timeout 43200
   kaggle kernels push -p cli/rsna-train-ttr-frac75-f4x --timeout 43200
   kaggle kernels status YOUR_KAGGLE_USERNAME/rsna-train-ttr-frac75-f23
   kaggle kernels status YOUR_KAGGLE_USERNAME/rsna-train-ttr-frac75-f4x
   ```
   - `--timeout 43200` (12 h, the Kaggle maximum) is required: Kaggle refuses a session without it when less than ~12 h of quota is free.
   - If the CLI says `--timeout` is unknown, update it first: `pip install -U kaggle`.
3. **Share both with Eesh right away:** on each notebook page, Settings / Sharing -> add **`eeshsaxena`** with
   **Can view**. Then Eesh can follow the status and download the outputs directly.

## 2b. Web UI alternative (if you do not use the CLI)

Do this twice, once per notebook file, and start both at about the same time:

1. kaggle.com, **Create -> New Notebook**.
2. **File -> Import Notebook**, then upload the `.ipynb` file.
3. Right panel, **Session options**: **Accelerator = GPU T4 x2** (it must be T4 x2), **Internet = ON**.
4. Right panel, **+ Add Input**, all four: datasets `eeshsaxena/rsna-knee-fleet-train` and
   `eeshsaxena/rsna-knee-frac75-labels`; notebooks `eeshsaxena/rsna-fleet-prep-s0` and `eeshsaxena/rsna-fleet-prep-s1`
   (add their **output**, so the tar files are visible). If an input does not show up in search, the sharing from step
   0.1 has not reached your account yet.
5. Rename the notebook to exactly the slug in the table above (`rsna-train-ttr-frac75-f23` or
   `rsna-train-ttr-frac75-f4x`). Keep it **Private**.
6. **Save Version -> Save & Run All (Commit)**. Wait until the version shows **Running** (keep your laptop awake until
   then); after that it runs on Kaggle's servers. Do NOT train in the interactive editor session: it would burn quota
   twice.
7. Share it with **`eeshsaxena`**, **Can view** (Settings / Sharing).

## 3. What happens inside, and what can go wrong

Each notebook checks everything **before** it trains, so a problem costs minutes, not hours:
1. inputs present + every file's sha256 (including the labels file);
2. environment (2 GPUs, disk);
3. it unpacks the image store (~12 GB) and verifies it byte for byte;
4. parity checks against our training server (pixel / tensor / model hashes);
5. a short speed probe of both trainers (~6 min);
6. training with a hard time wall, then a `run_receipt.json` listing every output file with its sha256.

| you see | meaning / what to do |
|---|---|
| status **ERROR at once with an EMPTY log** | Kaggle refused the session (usually quota, or another GPU session running). **Do not retry blindly.** Send Eesh a screenshot of `kaggle quota` (or the quota bar) and wait for Eesh's answer |
| status ERROR within the first ~30 min | an input check failed: open the log and send Eesh the last line containing `ABORT`. Usually an input is missing or not shared yet. Re-run only after the fix, and only if your quota still shows >= 12 h per notebook |
| a `WARNING ... E = ... < 16` line | the GPUs were slower than expected. Let it run and mention it to Eesh |
| status COMPLETE | done: section 4 |

**Please do not stop either run**, not even when it looks slow: each stops itself before the time limit and keeps
everything it has trained. A run stopped by hand loses its outputs.

## 4. When they are COMPLETE

- Tell Eesh. If you shared both notebooks with `eeshsaxena`, that is all: Eesh downloads the outputs directly.
- If sharing did not work, download from each notebook's **Output** tab and send them to Eesh (Drive etc.):
  - the two run folders (`f2_s1043_ttr_frac75/` + `f3_s1043_ttr_frac75/`, or `f4_s1043_ttr_frac75/` +
    `f0_s1044_ttr_frac75/`): the `*_best.pt` checkpoints (~113 MB each), the `oof_*.csv` files and the
    `epochs/oof_*.csv` files (the `epochs/*.pt` files are NOT needed);
  - the two `.log` files next to them;
  - `run_receipt.json` (needed: it carries the sha256 of every file).

That's it, thank you!
