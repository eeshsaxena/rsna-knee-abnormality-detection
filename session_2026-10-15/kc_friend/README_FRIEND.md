# For swapiy: run one Kaggle notebook (about 9 h of your GPU, no work after starting it)

Thank you! This trains two extra models for our team (RSNA Knee). Everything it needs is in two private datasets that
Eesh shares with your Kaggle account. It uses only the competition's own data.

## Steps (about 5 minutes of your time)
1. Eesh shares two datasets with you (you get an email / they appear under Datasets -> Shared with you):
   - `eeshsaxena/rsna-knee-kc-store-v1` (13 GB)
   - `eeshsaxena/rsna-knee-kc-bundle-v1` (360 MB)
2. On Kaggle: Code -> New Notebook -> File -> Import Notebook -> upload `rsna_train_kc_friend.ipynb` (Eesh sends it).
3. Right panel:
   - Add Input -> Datasets -> "Shared with you" -> add BOTH datasets above.
   - Session options -> Accelerator: **GPU T4 x2**.
   - Internet: **On** (it installs a few pinned Python packages).
4. Do not edit any cell. Click **Save Version -> Save & Run All (Commit)** -> Save.
5. Close the browser if you like; it runs by itself and stops itself after at most 9 hours.
6. When it shows "Complete" (Version tab), open the notebook's Settings -> Sharing and add **eeshsaxena** (Can view),
   or just tell Eesh the notebook's URL. We then fetch the output ourselves.

## Check (optional)
- In the log you should see `RSNA_ONLY_OK`, then two `LAUNCH` lines, and at the end `FRIEND_KC_DONE`.
- If it errors in the first 10 minutes, send Eesh a screenshot of the error; nothing else is needed.

## Option B (if your Claude / the kaggle CLI pushes notebooks for you, as for the s1045 runs)
- Unzip this pack; the folder `kernel_friend/` already has `kernel-metadata.json` with id `swapiy/rsna-train-kc-friend`,
  GPU on, internet on, both datasets attached.
- First check both datasets are visible to you: `kaggle datasets files eeshsaxena/rsna-knee-kc-store-v1` and
  `kaggle datasets files eeshsaxena/rsna-knee-kc-bundle-v1` (403 = not shared yet: ask Eesh).
- Push: `kaggle kernels push -p kernel_friend --timeout 43200`, then `kaggle kernels status swapiy/rsna-train-kc-friend`.
- After it shows COMPLETE: share it with eeshsaxena (Can view) or tell Eesh; nothing else.
- Do not run anything else on your GPU while it runs (it uses both T4s, about 9 h).
