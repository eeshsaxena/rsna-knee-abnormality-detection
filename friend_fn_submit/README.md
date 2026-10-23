# One-notebook submission for team "Eesh saxena" (needs ~15 min of your GPU quota)

This submits our strongest new entry, FN (our 10-model fleet + the public 0.949 model), from your account. It counts for the team.

## Before you start (Eesh)
- Eesh shares the dataset `eeshsaxena/rsna-knee-cnxt-fleet` with your Kaggle account (Can view).
- You must already be on team "Eesh saxena".

## Steps (Kaggle CLI)
1. In `rsna-sub-k9fn/kernel-metadata.json`, replace `YOUR_KAGGLE_USERNAME` with your username. Change nothing else.
2. Check your quota: `kaggle quota`. You need at least ~0.5 h of GPU left.
3. Push it. Your quota is under 12 h, so use a time cap of a bit less than your remaining quota. Example for 2 h left:
   ```
   kaggle kernels push -p rsna-sub-k9fn --timeout 6600
   ```
4. Share it with `eeshsaxena` (Can view) in the web UI: notebook page, then Settings / Sharing.
5. Wait until `kaggle kernels status YOUR_KAGGLE_USERNAME/rsna-sub-k9fn` says COMPLETE, then tell Eesh. Claude checks the output (fn_check).
6. ONLY after Eesh says "PASS, submit":
   ```
   kaggle competitions submit rsna-knee-abnormality-detection -k YOUR_KAGGLE_USERNAME/rsna-sub-k9fn -v 1 -f submission.csv -m "FN: K9 fleet + N (F .40 / N .60), parent-free"
   ```
   Use the version number shown by the push if it is not 1.

If the push shows an instant ERROR with an empty log, Kaggle refused the session. Report it; do not retry blindly.
