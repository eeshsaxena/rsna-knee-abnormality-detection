# Kaggle dataset/kernel evidence (Kaggle CLI 2.2.4 read-only, 2026-10-04; metadata + file listings only, nothing downloaded except dataset-metadata.json)

## Leaderboard top (CLI + LB page, 2026-10-04)
1 rhoskeri 0.963 (/rhoskeri) | 2 Pa3BaJluHa 0.962 (/hettpugymajlhuk) | 3 joinT fusion 0.961 (+2, /tomokihirose) | 4 Wassim Dobbi 0.961 | 5 YOKKOISHO 0.960 (+2, /tennogh) | 6 TomKroumov 0.959 (/tom1115) | 7 dalab 0.959 (+4, /junpeimorioka) | 8 NMarkS 0.959 | 9 knee-deep 0.959 (+4, /yosukeyama = YYama, author of 743416) | 10 Scott Willis 0.959 | 11 Zhengru Li | 12 takoi & charmq 0.959 | ... 21 Mazzutti 0.958 (/profmazzutti)
- None of the visible top-12 members has a public notebook or dataset for this competition (kaggle kernels list --user / datasets list --user). Only tennogh has "My knee MRI" (his own personal scan, CC0, 2026-08-19).

## External knee MRI brought onto Kaggle (public datasets)
- ibrahimshaikhdev/knee-mri-917 "Knee MRI 917": 4,146,632,539 B, created 2026-10-03 10:59 UTC, license field CC0-1.0, no description. Files: metadata.csv + volumetric_data/volumetric_data/*.pck. 917 volumes + .pck + metadata.csv = KneeMRI (Stajduhar, Rijeka) layout -> INFERRED re-upload of KneeMRI (source license CC BY-NC-ND 4.0 per 743416; zenodo 14789903). CC0 relabel is not the source license.
- techtech69/mrnet-knee-mri-stanford-aimi "MRNet Knee MRI Stanford AIMI": 7.57 GB, 2026-08-27, 2,321 downloads, license field CC0-1.0. Files train-abnormal/acl/meniscus.csv + train/*.npy = MRNet v1.0 layout. Re-upload of MRNet (source = Stanford AIMI Research Use Agreement).
- older MRNet re-uploads: cjinny/mrnet-v1 (2019, 2,411 dl), henryhaefliger/kneescans, abdurrahimsheikh/mrnet-dataset, flap1812/mrnet-v1-0 (2025-09), sharif123diu/mrnet-dataset (2025-11).
- sohaibanwaar1203/kneemridataset (2019, 5,329 dl): KneeMRI 917 volumes, license "unknown".
- bharath111l/fastmri-plus-official-knee-annotations: 2026-08-24, 76 dl; files LICENSE, README.md, knee.csv, knee_file_list.csv (fastMRI+ annotation CSV only, no images).
- fastMRI image re-uploads exist (moshiurrahman00/nyu-fastmri-knee-singlecoil-val 16.5 GB 2025-09; arafatshovon/fastmri-knee-multicoil 126 GB 2024-10) but predate comp and are recon-oriented.
- tsuyu122/rsna-knee-mri-balanced-labels-public: 2026-09-06, 208 dl, CC0. "Harmonized 8,631 Knee MRI Studies & 12 Labels across RSNA, MRNet, OAI & fastMRI": RSNA 4,407 + MRNet 1,248 + OAI 3D-DESS 2,976 (+ fastMRI/OAI-ZIB auxiliary). Labels only (labels.parquet). Cites a "Kaggle Competition Rule 4.b (AI Provenance & Open Weights)" that does not exist in this comp's rules (rule 2.4.b is Data Security) -> treat as unreliable.
- OAI re-uploads on Kaggle are mostly knee X-ray (jeftaadriel, mahfuzahmed17, chauvvan) or segmentations (kgaooo/oai-tissue-segmentations 2023).
- SKM-TEA: no Kaggle dataset found (datasets list -s skm-tea -> none).

## Pretrained weights derived from external knee MRI
- quannguyen0501/dinov2-s-mrnet: file MST_MRNet.ckpt (275.6 MB), 2026-09-29, license unknown. INFERRED: Medical Slice Transformer (DINOv2-S) checkpoint trained on MRNet.
- juejiezeng/rsna-knee-mrnet50-inference (competition notebook, 2026-09-15, 0 votes) - title suggests MRNet-style model.
- kirilllabzin/knee-radiodino-s16-v21-source (RadioDINO weights, CC BY 4.0 label), prithvirajrp/orthofoundation-rsna-knee-raw-embeddings (OrthoFoundation embeddings, apache-2.0 label), fabotelli/pillar-0-knee-fine-tune-loader-demo, lorenabangovigo/rsna-knee-orthodiffusion-specialist: medical foundation models whose pretraining corpora may include external MRI; not verified.

## Which notebooks attach external knee datasets (kaggle kernels list --dataset X, 2026-10-04)
- sohaibanwaar1203/kneemridataset (KneeMRI 917, Rijeka) is listed as input for ~25 RSNA-knee notebooks: sakhawathossen/rsna-knee-enhanced-ensemble (94 votes, 2026-09-14), sakhawathossen notebook-01..05 (2026-08-11), amanatar/rsna-knee-super-ensemble(-0920) (14-15 votes), romanrozen/rsna-knee-data-structure-eda-baseline (141 votes, 2026-08-10), shashwat1729 nishantk infer forks, nishantkharga, ispromashka, yamadan96 forks, sonameyden/rsna-knee-solution (2026-10-02), michaelcardoza, beamhonor0911, xuanleekaggle, tazwarul/ligament-tear, kurtdynguyen.
  * Verified on page: sakhawathossen/rsna-knee-enhanced-ensemble /input lists "kneeMRIdataset" among Data Sources, BUT the rendered notebook (__results__.html) contains no "kneemri", ".pck", "metadata.csv" or "MRNet" strings; "external" there means other public notebook ensembles ("external public family"). => attached, not used in code (MEASURED for current version).
  * romanrozen current version /input lists only competition data + DINOv2 models (no kneeMRIdataset); rendered code has no KneeMRI reference. CLI --dataset match likely reflects an older version (INFERRED). Forks inherit attachments.
- cjinny/mrnet-v1 (MRNet re-upload) is used by tazwarul/ligament-tear (2026-09-15) and older non-competition MRNet notebooks.
- techtech69 MRNet re-upload, ibrahimshaikhdev knee-mri-917, bharath111l fastMRI+, quannguyen0501 MST_MRNet, tsuyu122 labels: no public notebooks list them as input ("Not found").
