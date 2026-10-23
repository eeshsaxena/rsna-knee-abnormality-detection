# For swapiy: two Kaggle training runs on your GPU quota

Thanks for helping! This pack has **two** notebooks. Together they train folds 0, 1, 2 and 3 of our "frac75" knee
model with a new random seed (1045). Each extra set of models makes our final blend a bit stronger and more stable.

| folder | notebook (keep the slug exactly) | trains | GPU time |
|---|---|---|---|
| `A_folds_0_1/` | `rsna-train-ttr-frac75-s1045-f01` | folds 0 + 1 | ~10.5-11 h |
| `B_folds_2_3/` | `rsna-train-ttr-frac75-s1045-f23` | folds 2 + 3 | ~10.5-11 h |

## Quick plan

1. Check `kaggle quota`. You need **at least 24 h free** to run both (12 h reserved per run). If you have less than 24 h
   but at least 12 h, run only `A_folds_0_1` and tell Eesh.
2. **Start only when Eesh says GO.**
3. Both can run **at the same time** (Kaggle allows 2 GPU sessions per account). Push A, wait until it shows Running,
   then push B. Do not start any other GPU notebook while they run.
4. For each folder: put your Kaggle username into `cli/<slug>/kernel-metadata.json`, then
   `kaggle kernels push -p cli/<slug> --timeout 43200` (each folder's README has the exact commands and a web UI
   alternative).
5. Right after each push: share the notebook with **`eeshsaxena`** (Settings / Sharing, **Can view**).
6. **Do not stop them**, even if they look slow. Each one stops itself before 11.5 h and keeps everything it trained.
7. When both show COMPLETE, tell Eesh. Nothing to download: Eesh fetches the outputs.

Each folder also has its own `README_TEAMMATE.md` (details, what can go wrong) and `TODO.md` (checklist).
`RUNBOOK_EESH.md` is Eesh's side: you can ignore it.

Rule check: you must still be on team "Eesh saxena". Competition rules forbid sharing this code or data with anyone
outside the team.
