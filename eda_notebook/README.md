# RSNA Knee EDA notebook

Public-ready exploratory data analysis for the RSNA Knee Abnormality Detection competition.

- Notebook: `rsna-knee-eda.ipynb`
- Kaggle metadata: `kernel-metadata.json` (id `eeshsaxena/rsna-knee-eda-labels-and-mri-views`)
- Kaggle URL (currently PRIVATE): https://www.kaggle.com/code/eeshsaxena/rsna-knee-eda-labels-and-mri-views
- Kaggle title: "RSNA Knee EDA: Labels and MRI Views". It was shortened from "RSNA Knee: Labels, Co-occurrence and MRI Views EDA" because Kaggle wants the title to slugify to the id. The notebook's own heading keeps the longer name.

## Runtime

Version 4 on Kaggle, CPU, no internet, no GPU: about 101 s for all cells (about 2 minutes for the whole session). Folder listing takes about 17 s and header reading about 15 s, both with a small thread pool. It also runs in about 40 s on the box against the 458-study subset.

## What is in it

1. **Files**: top-level files and folders, row counts, study → series → slice layout, CSV vs folder consistency checks, slice-file counts for all 24,371 series (819,078 files).
2. **Reports**: preview of `train.csv`, report length, rough language mix (stop words + script check), de-identification placeholders.
3. **Labels** (the 58 labelled studies): prevalence with 95% Wilson intervals, findings per study, co-occurrence counts, phi correlation heatmap, strongest pairs, most common combinations.
4. **Series table**: series per study, plane x sequence type, coverage per study, Fluid_Sensitive vs Fat_Suppression, slices per series.
5. **DICOM headers** (seeded 800-study sample, 4,429 series): vendors, field strength, transfer syntax, image size, pixel spacing, field of view, slice thickness and gap, TE vs TR by flag, CSV plane vs ImageOrientationPatient, tilt, 2D vs 3D, SeriesDescription quality, laterality, sex, PatientID, misc tags.
6. **Images**: decoder check, file order vs anatomical order, 4-study x 3-plane montage, 12-slice strip, one sagittal FS slice per vendor, raw intensity ranges, positive vs negative examples for Effusion, ACL and Medial OA (middle slice only).
7. **Test notes**: example test files, format checks, test montage, a neutral list of data pitfalls.
8. **Summary**.

Every block is wrapped so a bad file prints a "skipped" line instead of stopping the run. All sampling is seeded. A PNG copy of each figure goes to `figures/` (the Output tab on Kaggle).

Content rules followed: official competition data only, no external data, no models, no submission file, no modelling advice, nothing about our labels, models, blends or scores. No em dashes.

## How to publish (Kaggle UI)

1. Open https://www.kaggle.com/code/eeshsaxena/rsna-knee-eda-labels-and-mri-views while logged in as eeshsaxena.
2. Check that the version shown is **Version 4** and that the output looks right (it should end with "total runtime: ...").
3. Click **Share** (top right of the notebook page).
4. Change the visibility from **Private** to **Public** and click **Save**.
5. Optional: add tags such as "exploratory data analysis", "data visualization", "image", "healthcare".

Notes:
- Versions 1 to 3 were test runs. Version 1 has placeholder text in the observation cells, and versions 2 and 3 have an extra debug cell that prints tables as plain text. None of them contains anything sensitive, but they will be visible in the version history once the notebook is public.
- If you push again from the CLI after publishing, set `"is_private": false` in `kernel-metadata.json` first so the push does not try to make it private again.
- To re-run it outside Kaggle, set `RSNA_KNEE_DIR` to a folder with the competition files. On the box: `~/eda_data/rsna-knee-abnormality-detection` (read-only symlinks to the existing copy) with the venv `~/eda_venv`.

## Suggested description

A walkthrough of the RSNA Knee data using only the official files: how studies, series and slices fit together, how the twelve labels behave (prevalence with confidence intervals, co-occurrence, phi correlation and common combinations), and what the DICOM headers say about scanners, resolution and sequences. It ends with image montages across planes and scanners, a slice strip, intensity ranges, labelled positive and negative examples, and a short list of data pitfalls. Runs on CPU in about two minutes.
