# Transductive / test-time trick check (10-19, ~21:30 IST). RESEARCH + MEASUREMENT ONLY.

Nothing was built, pushed, committed, submitted or created. No GPU, training or box job ran. Box access was read-only:
`ls` of the data dirs and ONE pydicom header read (no pixels, no writes) of 3 DICOM files (2 train, 1 visible test).
No PREREG, HANDOFF, PROGRESS or scheduled-task edits. No public code was pulled or executed.
Tags: MEASURED = computed by me in this session (script named); CLAIMED = a third party or an older note says so
(source named); MODEL = arithmetic on stated assumptions; GUESS = judgement.
Scripts and outputs: `session_2026-10-19/trick/work/` (my files: dup_reports.py, near_dup.py, pairs_en.py, fp.py, sim.py,
sim2.py, sim3.py, dups.py, exactdup_all.py, knn_trans.py, report_nn.csv, en_template_fields.csv, best_01.npy, best_11.npy,
csls_score.npy, dup4394.png). Other files in that dir belong to parallel agents. I deleted only my own regenerable caches
(fp32.pkl 888 MB, avgsim.npy; `fp.py` rebuilds them in 40 s) because C: is 96 % full.

## 1. Verdict (plain language)
**DEAD. No transductive or test-time trick here can plausibly add +2e-4 on private by 10-21.**
- (a) Duplicates / same patient: the train set holds exactly ONE pair of studies with identical images, out of 9.7 M
  study pairs. Report duplicates are templates for different patients. The de-identified headers carry nothing that
  links two studies of one patient. Expected duplicates between train and the private test: ~0.4 studies, each worth
  ~1e-5 or less. Not a lever.
- (a') Using similar-looking neighbours (kNN in image space), either their predictions or their train labels: negative
  on the expert-label ruler (gold58), positive only on the report-label ruler, the same signature as the site prior
  that was already closed as LB-negative.
- (b) Test-time BN adaptation: only the Raptor legs have BatchNorm; no evidence, unmeasurable at the needed precision
  in the window. Pseudo-label self-training: needs a training loop inside the hidden run, and our distilled students
  were already LB-negative.
- (c) Spare runtime: nothing new with evidence (agrees with THINK_1019 sec 1 item 5).
- Legal: yes, in principle (rule 3.4.b bans only HAND labelling of test; automated use of train labels and unlabeled
  test images in-kernel is allowed, CLAIMED RULES_AND_HOST_COMMENT_10-08.md item 8). Feasible by 10-21: (a) technically
  yes (fingerprints are cheap) but worthless; (b) no.

## 2. (a) Near-duplicate and same-patient studies
### 2.1 Report text (MEASURED dup_reports.py, near_dup.py, pairs_en.py)
- 4,407 train studies, 4,407 unique StudyInstanceUIDs. Labels: only the 58 gold rows are filled in train.csv.
- Exact duplicate reports: 177 rows in 46 groups. Unicode-normalized (digits, [DATE], punctuation stripped): 224 rows in
  52 groups; only 23 of those rows are >= 500 chars. Top groups are short "normal knee" or one-finding templates
  (Turkish normal template x37, Spanish Hoffa template x21, Spanish "sin anomalias" x15, etc.).
  (An earlier ASCII normalization printed 433 rows / 76 groups; 109 of those were Cyrillic / Greek reports emptied by
  the regex, an artifact. The Unicode count is the right one.)
- TF-IDF char 4-5-gram nearest neighbour, long reports (>= 500 chars, n 3,517): nn cosine >= .9 for 199, >= .95 for 70,
  >= .98 for 35. Reading the top pairs: same radiologist templates with different findings (e.g. Turkish SOL DIZ
  grade II meniscopathy twins, US "Exam Type: MRI KNEE LEFT/RIGHT" template).
- None of the 1,212 report-duplicate pairs is an image duplicate (max image score 1.98 vs 3.92 for the one true image
  duplicate; max sagittal-FS slice corr .9675 < .99) (MEASURED, inline check on csls_score.npy / best_01.npy).
- US template site (351 studies, "Exam Type:" reports): 178 RIGHT / 173 LEFT; Comparison = "No relevant prior studies"
  in 260; a prior knee MRI is cited in only 6-8 (1.7-2.3 %); identical non-generic indications with opposite sides
  (likely bilateral same patient) = 2 pairs, 4 studies (1.1 %). Host: "single knee per exam" (CLAIMED forum.md, 733343).
- Reports mentioning any comparison / prior word (crude multilingual regex): 1,548 of 4,407, mostly "Comparison: No
  relevant prior studies", so this is not a repeat-exam rate.

### 2.2 Images (MEASURED fp.py, sim*.py, dups.py, exactdup_all.py)
Source: our own preprocessed train crops `_crop_shards` (111 shards, 4,407 studies, 8 slices per series, 150 mm centre
crop, 384 px; `kernels/preprocess_crop_out`). Note: its 8 slices are drawn from glob-sorted random-UID filenames, so
they are random slices within the middle 76 % of each series, not matched positions.
- Fingerprint: every slice block-averaged to 32x32, z-normalized; slice-vs-slice correlation across all 4,407 studies
  for sag FS, cor FS, ax FS, sag non-FS, cor non-FS, ax non-FS (~194k slices).
- **Identical slices (corr >= .999, content sd > 10) across different studies: 9 slice pairs, ALL from ONE study pair
  (si 4394 / 4404).** Every other study pair has max slice corr <= .9877 (sag FS) / <= .98 (cor FS).
- Sensitivity: two uploads of the same series (re-anonymized filenames, so different random 8-slice draws) share at
  least one identical slice with P ~.99 per series of ~23 usable slices (MODEL, hypergeometric), and there are 5-6
  series per study, so a missed exact duplicate is very unlikely.
- That pair: 4394 report "SOL DIZ" (left knee), 4404 report "SAG DIZ" (right knee), Turkish site, Siemens Aera 1.5 T;
  the DICOM header says Laterality = R in BOTH, and the montage (dup4394.png) is one knee. So it is one exam filed twice
  with a left and a right report: a label-noise row, not exploitable information. Its PatientIDs differ
  (265fbf2e-013 vs 640798e5-163) and so do the FrameOfReferenceUIDs (MEASURED, read-only header read on the box).
- Same-knee repeat exams (non-identical images): a hub-corrected multi-series score (sim3.py, z per series type, CSLS
  correction, mean over >= 3 shared series types) has a smooth tail: dup pair 3.92, next 2.61, then 2.54, 2.42, ... with
  no separated cluster; 0 studies have a best partner >= 4 besides the dup pair. The top 200 image pairs are 97 %
  same-language (random pairs 22 %) with report similarity median .43: protocol / scanner twins with DIFFERENT findings
  (e.g. 1100 / 3844: ACL tear vs diffuse degeneration). So the similarity mostly captures site and protocol, and I found
  no evidence of repeat exams (a weak 32 px fingerprint cannot fully exclude non-identical repeats; GUESS that they are
  rare, given 1.7-2.3 % prior-MRI mentions at the one site where it is checkable).
- gold58 (the closest proxy for how the hidden test was drawn): 0 of 58 have an image duplicate in train (max slice
  corr .978; best-partner score max 2.27).

### 2.3 Linking studies by DICOM metadata (MEASURED, header dump of 3 files on the box, read-only)
Tags present: ImageType, SOP/Study/Series/FrameOfReference UIDs (re-generated under 1.2.826.0.1.3680043.8.498), Modality,
Manufacturer, ModelName, SeriesDescription, PatientID (anonymized), PatientSex, BodyPartExamined, sequence parameters,
SoftwareVersions, coil names, PatientPosition, InstanceNumber, ImagePosition/Orientation, Laterality, SliceLocation,
pixel format, window. ABSENT: StudyDate, StudyTime, AcquisitionTime, AccessionNumber, InstitutionName, StationName,
DeviceSerialNumber, PatientAge, PatientBirthDate. The duplicate pair proves the anonymizer salts PatientID and every UID
per study. So bilateral or repeat studies of one patient cannot be linked by metadata, and the EDA's "each PatientID in
one study" (CLAIMED eda_notebook/README.md) is uninformative by construction.

### 2.4 What a duplicate would be worth (MODEL)
- Pair rate per study pair p = 1 / C(4407, 2) = 1.03e-7. Train x hidden test pairs 4,407 x ~1,300 = 5.7 M -> E ~0.59
  duplicates; private share .7 -> E ~0.41. Within the hidden test C(1300, 2) = 0.84 M -> E ~0.09 (and identical images
  get identical predictions anyway).
- Value of one: perfectly re-ranking one study moves a label's AUC by about (1 - AUC) / n_same_class
  ~ .06 / 300 (positive) or .06 / 600 (negative), ~1.3e-4 macro at most. But every c4 member was trained on that train
  study (Raptor alldata, 3 of 4 R6 folds, S6fall), so its prediction is already in-sample; the report label is the wrong
  regime for experts; and the one real duplicate carries a wrong-side report. Recoverable share ~.3 (GUESS).
- Expected gain ~0.41 x 1.3e-4 x .3 ~ 1.6e-5 (MODEL), at the cost of a new dataset (train fingerprints) and a new
  stage. DEAD.

### 2.5 kNN "transductive smoothing" and kNN label transfer (MEASURED knn_trans.py)
Neighbours = top-k studies by the image score above (a test-time version would use test-test neighbours for the
prediction variant, and test-train neighbours plus shipped train labels for the label variant). Base = R7OOF
(heliosli-family OOF on train) percentile per label; truth = report labels v2 (> .5) on 4,349 non-gold studies
("OOF", report regime) and gold58 (expert regime). Combined score = pct(own) + a x pct(neighbour feature).
| variant | k | a = .05 dOOF / dG58 | a = .10 | a = .20 |
|---|---|---|---|---|
| neighbours' predictions (pure test-time) | 5 | -.00048 / -.00085 | -.00178 / -.00454 | -.00633 / -.01468 |
| neighbours' predictions | 20 | -.00059 / +.00063 | -.00190 / -.00031 | -.00637 / -.00829 |
| neighbours' train labels | 5 | +.00142 / -.00267 | +.00198 / -.00843 | +.00103 / -.02053 |
| neighbours' train labels | 20 | +.00163 / +.00123 | +.00235 / +.00024 | +.00168 / -.00978 |
| reference: same-language mean prediction (site prior) | all | +.00074 / -.00407 | +.00059 / -.00929 | -.00199 / -.02292 |
Base: OOF .86275, gold58 .91929. Solo AUC of the neighbour features is .56-.65 (site-level information only).
Reading: the prediction variant is negative on both rulers. The label variant is the site prior again: up on report
labels (reporting habits per site), down on expert labels except one cell (+.0012 at k 20, a .05) that sits inside the
gold58 noise (SE ~.003-.005; THINK_1019 notes the PF OA .1 miscall at P .925). Same pattern as the closed metadata /
site-offset lever (gold -.015, CLAIMED THINK_1019 sec 5). DEAD.

## 3. (b) Test-time adaptation and pseudo-label self-training
- BatchNorm inventory (MEASURED, timm 1.0.29 local instantiate): Raptor `coatnet_rmlp_2_rw_384` has 25 BN layers (stem +
  the two MBConv stages) and 33 LayerNorm-type; ConvNeXt-S (R6, and S6fall's compact_infer family, GUESS same) has 0 BN.
  So AdaBN can touch only the Raptor r384 / r224 legs (stage-3 share .2-.8 per label, then x .8).
- Raptor serves one study per forward (`xwins.unsqueeze(0)`, K_EVAL windows; MEASURED c4 cell 4), so "batch statistics"
  at inference would be per-study instance statistics: a large, untested behaviour change. Proper AdaBN needs a second
  calibration pass over all test windows (+0.30-0.42 h for r384, +0.1-0.25 h for r224, from runtime.md sec 2 MODEL).
- Evidence: none on this comp. A clean measurement does not exist: Raptor was trained on ALL train including gold58, so
  train-based AdaBN stats equal its running stats in expectation (effect ~0 by construction), and per-site AdaBN can be
  scored only on gold58 (SE ~.004 >> 2e-4). Also intensities are already percentile-normalized per slice, which removes
  most of the scanner shift that AdaBN usually fixes (GUESS).
- Pseudo-label self-training: a training loop in the hidden run on ~1,300 studies x 2 T4 inside ~2.5 h spare, with no way
  to validate before 10-21 (a kernel needs ~6.5 h to commit and ~6.5 h to score, CLAIMED brief). Our distilled students
  were LB-negative as members (CLAIMED brief / THINK_1019 sec 5), and a test-time student sees 3x fewer images.
- If someone insists on settling AdaBN anyway (NOT recommended, GPU, needs Eesh's go): box V100, ~40 min. Load the r384
  SWA checkpoint; inputs = the c4 Raptor preprocessing of gold58 + 400 non-gold train studies; arm A = eval mode;
  arm B = BN layers in train mode with momentum None, one no-grad pass over all 458 studies' windows, then eval. Report
  Spearman(A, B) per label and gold58 macro A vs B. Kill if Spearman < .98 (too big a change for a blind probe) or if
  gold58 drops; even a pass does not establish +2e-4.

## 4. (c) The ~2.5 h spare runtime
Nothing new with evidence. The only evidenced-in-mechanism use remains THINK_1019's parked E3 (R6 shift TTA), which is
outside this lens. Fingerprinting the hidden test for duplicates would cost ~10 min, but there is nothing to gain (sec 2.4).

## 5. Side findings (not levers)
- Train label noise: si 4394 / 4404 (StudyInstanceUIDs ...99725523560254732094432244556623376711 and
  ...99926624968240681772735102303531613209) are one right-knee exam filed twice, one with a LEFT-knee report.
  Irrelevant now (training is frozen) but worth a line in any write-up.
- Report-text clusters are a strong site / radiologist proxy (top image pairs 97 % same language). Anything that learns
  from report labels picks up per-site reporting habits, which is why every site-flavoured lever reads + on OOF and
  - on gold58.

## 6. Numbers to quote
- Exact image duplicates in train: 1 pair / 4,407 studies (9 identical slices) (MEASURED exactdup_all.py).
- gold58 studies with a train duplicate: 0 / 58 (MEASURED).
- E[train-private duplicates] ~0.41; E[gain] ~1.6e-5 (MODEL).
- kNN smoothing: -.0005 to -.0064 OOF, -.0009 to -.0147 gold58 (MEASURED knn_trans.py).
- AdaBN reach: 25 BN layers in Raptor, 0 in ConvNeXt (MEASURED timm).
