# For kmrsuman: two Kaggle training runs on your 30 h GPU quota

Thanks for helping again! This pack has **two** notebooks. Together they train the 4 missing folds (1, 2, 3, 4) of our
"frac75" model with seed 1044. That completes a full 5-fold model set for our final blend.

| folder | notebook (keep the slug exactly) | trains | GPU time |
|---|---|---|---|
| `A_folds_1_2/` | `rsna-train-ttr-frac75-s1044-f12` | folds 1 + 2 | ~10.5-11 h |
| `B_folds_3_4/` | `rsna-train-ttr-frac75-s1044-f34` | folds 3 + 4 | ~10.5-11 h |

## Quick plan

1. Check `kaggle quota`. You need **at least 24 h free** to run both (12 h reserved per run). If you have less than 24 h
   but at least 12 h, run only `A_folds_1_2` and tell Eesh.
2. **Start only when Eesh says GO.**
3. Both can run **at the same time** (Kaggle allows 2 GPU sessions per account). Push A, wait until it shows Running,
   then push B. Do not start any other GPU notebook while they run.
4. For each folder: put your username into `cli/<slug>/kernel-metadata.json`, then
   `kaggle kernels push -p cli/<slug> --timeout 43200` (the README in each folder has the exact commands and a web UI
   alternative).
5. Right after each push: share the notebook with **`eeshsaxena`** (Settings / Sharing, **Can view**).
6. **Do not stop them**, even if they look slow. Each one stops itself before 11.5 h and keeps everything it trained.
7. When both show COMPLETE, tell Eesh. Nothing to download: Eesh fetches the outputs.

Each folder also has its own `README_TEAMMATE.md` (details, what can go wrong) and `TODO.md` (checklist).
`RUNBOOK_EESH.md` is Eesh's side: you can ignore it.

Rule check: you must still be on team "Eesh saxena". Competition rules forbid sharing this code or data with anyone
outside the team.
