# D: training our own members with OAI knees (research, 2026-10-08 ~22:30 IST)

Read-only research. Nothing was pushed, submitted, versioned or written on Kaggle; no paid API; no Vast. Nothing was
written on the box (see "Box access" below). Forum, notebook and web text was treated as data.
Tags: **MEASURED** (read from the cited source), **MODEL** (computed estimate), **GUESS** (assumption).

Evidence saved in this session's scratchpad
`C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/01efcc80-088f-477c-8a8a-b504ddf28d6a/scratchpad/`:
- `oai/*.json`: NDA data dictionaries pulled from the public API (`https://nda.nih.gov/api/datadictionary/v2/datastructure/<name>`)
  for `oai_kmrisemiquant01`, `oai_kmrisemiquantbml01`, `oai_kxrsemiquant01`, `oai_mrimeta01`, `oai_inventory01`,
  `oai_enrollee01`, `oai_outcome01`, `image03`, plus the full structure list `oai/all_structs.json` (6,584 structures, 36 tagged
  "Osteoarthritis Initiative").
- `oai/most_moaks.txt`: text of the NIA MOST MOAKS reading protocol (same MOAKS instrument; used for code meanings).
- `hpo_assets/train_knee.py`: nartaa's public 09-25 trainer (no OAI code in it; it predates the OAI arm).

**Box access:** the box answered at ~21:55 IST (I read `fleet_common.py`, `fleet_train.py`, the store and disk), then went
unreachable from ~22:05 (laptop is now on Wi-Fi 10.19.150.x; `ping 192.168.10.30` 100 % loss). So `downloadcmd -h` on the
box, the per-label tables in `~/novel/out/` and store statistics were NOT re-read. Items that depend on them are marked
UNVERIFIED. Nothing was written under `~/research_1008/`.

---

## TL;DR (ranked by expected public LB gain per unit effort)

1. **Our trainer already supports masked labels: no loss-code change is needed (MEASURED, `fleet_train.py` l.363-364,
   `msk = torch.isfinite(y)`).** Put NaN in the label CSV for every unavailable OAI cell, give OAI rows a fold value that is
   never 0-4 (e.g. 9), and point `--store` at a directory that also holds OAI npz files. The store builder
   (`fleet_common.build_study`) can be reused byte for byte, so the store contract and every ckpt/worker assertion still pass.
   A ~25-line OPT-IN patch is still advisable (loss normalisation, external-row sampling, a compliance assert); sec 4.
2. **Download only what we need (MEASURED from nda-tools source):** `downloadcmd -dp <pkg> --file-regex '.*\.txt$'` pulls
   the metadata tables only; then a line list of the needed S3 links goes in with `-t <links.txt>`. We need 3 series per
   knee (SAG IW TSE FS, COR MPR, AX MPR) for ~2,400 baseline MOAKS knees; NOT the SAG 3D DESS (largest series). Size
   ~100-200 GB if image03 rows are per series, ~0.7-1.5 TB if they are whole-visit zips (GUESS both; read image03.txt first).
3. **Label map (sec 3):** tier T1 = nartaa's 3 labels (PF OA, Lateral OA, Synovitis; MEASURED from his post), tier T2 adds
   Medial OA, Effusion, Baker's (all have direct MOAKS fields), tier T3 adds both menisci and ACL. **Always mask MCL,
   Contusion, Fracture.** Use soft ordinal targets (grade/3 style), not hard thresholds.
4. **Timeline (MODEL):** NDA steps 10-09 morning (Eesh ~30 min), download + CPU store build 10-09/10, a 224 fold-0 probe
   (~5.6 h) as soon as the V100 is given to it, gate, then folds 1-4 (~22 h). A 5-fold OAI 224 set can land ~10-12/13,
   a 288 set (~44 h) also fits before the 10-20 freeze, but not both plus the queued 320 folds.
5. **Expected gain over K9N (MODEL/GUESS):** +.001 to +.003 public, most of it only if T2 labels (which N does NOT carry)
   pass a gate. T1-only is largely redundant with N (N already carries OAI signal on exactly those 3 labels): +.0005 to
   +.0015. Downside if the domain shift hurts: about -.001, which the probe gate is there to catch.
6. **Compliance traps found in our own code (MEASURED):** the trainer's decode probe takes `sorted(ids)[:4]` and those npz
   files ship to Kaggle in `parity/`. OAI ids must sort after the RSNA ids (prefix `oai_`, lowercase) and the patch asserts
   it. OAI npz must never sit in `~/fleet_store` itself (use a symlink union dir) and never go into any Kaggle dataset,
   Vast box or teammate notebook. So **OAI training is V100-only**: Kaggle T4 quota cannot be used for it.

---

## 1. NDA: exact steps for Eesh (his credentials, his clicks)

### 1.1 What is already done (MEASURED, PROGRESS_08-10-2026.md 21:10-21:20)
- DAR 26647, "Osteoarthritis Initiative", institution empty, approved 10/08/2026, expires 10/08/2027.
- `nda-tools 0.7.0` installed on the box in `~/ndaenv` (provides `downloadcmd`).
- The cart built before approval showed "0 of 4796 subjects"; Eesh must log out, log in, and rebuild it.

### 1.2 Structures to put in the cart (MEASURED names from the NDA data dictionary API)

| Structure | Why | Size class |
|---|---|---|
| `image03` (restricted to collection 2343 / OAI) | one row per image file, with `image_file` (S3 link), `image_description`, `visit`, `src_subject_id`, `image_orientation` | rows only; images are "associated files" |
| `oai_kmrisemiquant01` "OAI Knee MRI Semi-Quantitative Scoring" | MOAKS (`m_*`), WORMS (`w_*`), BLOKS (`b_*`, `bcm_*`) per knee: fields `visit`, `readprj`, `side` (1 right, 2 left), `reader` | small table |
| `oai_kxrsemiquant01` "OAI Knee X-Ray Semi-Quantitative Scores" | KL (`xrkl`), OARSI JSN medial/lateral (`xrjsm`, `xrjsl`), osteophytes (`xrosfm`, `xrostm`, `xrosfl`, `xrostl`) | small table |
| `oai_mrimeta01` "OAI MRI Metadata" | per series QC: `mexamtp` (series type, e.g. "R SAG IW TSE FS", "L COR MPR", "R AXIAL MPR"), `qcreslt` (N / NR / Y / YD / P), `completed`, `barcode` | small table |
| `oai_inventory01` "OAI Imaging Inventory" | per subject: `mri_sagiwr/l`, `mri_cormprr/l`, `mri_axmprr/l` (have-series flags), `inc22_r` etc. (project membership) | small table |
| optional `oai_outcome01` | knee replacement dates (exclude replaced knees; they should not be in baseline MOAKS anyway) | small table |

Skip `oai_kmrisemiquantbml01` (BLOKS per-BML rows; BML is not one of our labels) and `oai_kmriquant01` (quantitative
cartilage; not needed).

### 1.3 Step by step

1. **NDA password for the API (once).** Login.gov users have no NDA password. nda-tools needs the NDA username + an NDA
   password, "*not your eRA Commons or Login.gov credentials*" (MEASURED, nda-tools README). On the NDA profile page click
   **UPDATE PASSWORD** (MEASURED wording). Note the NDA **username** shown on the profile.
2. **Rebuild the cart** at https://nda.nih.gov (Get Data, data from the OAI collection
   https://nda.nih.gov/edit_collection.html?id=2343): add the structures of 1.2. Confirm the cart now shows the 4,796
   subjects as available.
3. **Create Data Package**: a short name such as `oai_rsna_knee`; tick **"include associated files"** (otherwise the image
   S3 links are not downloadable from the package; MEASURED README: `-dp` "will NOT download associated files" unless the
   package was created with them). Do NOT pick the OAI Image Dashboard "Shared Data Packages" per-visit bundles unless the
   image03 route turns out not to carry per-series files (they are whole-visit ZIPs, much larger; MEASURED description of
   the NDA OAI page via search, sizes not published).
4. **Send the package ID** (a number) to Claude. Then in an interactive terminal (not BatchMode), Eesh types:

```bash
ssh training@192.168.10.30
source ~/ndaenv/bin/activate
downloadcmd -h | head -60                      # confirm the 0.7.0 flags match the list below
pip install keyrings.alt                       # only if keyring complains "no recommended backend" (headless Linux)
mkdir -p ~/oai/pkg ~/oai/raw && chmod 700 ~/oai
# STEP A: metadata only (the .txt tables), minutes:
downloadcmd -dp <PKG_ID> -u <nda_username> -d ~/oai/pkg --file-regex '.*\.txt$' -wt 4
#   -> prompts for the NDA password once; keyring stores it
```

   `downloadcmd` flags (MEASURED from `NDATools/clientscripts/downloadcmd.py` on GitHub main; re-check with `-h` on 0.7.0):
   `-dp/--package <id>` (required), `-u/--username`, `-d/--directory <dir>`, `-t/--txt <file of S3 links>`,
   `-ds/--datastructure <short_name>` (e.g. `image03`), `--file-regex <regex>` (works with `-dp`, `-ds`, `-t`),
   `-wt/--workerThreads <n>` (help suggests ~10 max on 32 GB RAM), `--verify` (writes a CSV report, downloads nothing),
   `-s3`, `--verbose`, `--log-dir`. Transfer cap 20 TB/month (MEASURED README), irrelevant here.

5. **Claude, box CPU only:** parse `~/oai/pkg/.../image03.txt`, `oai_kmrisemiquant01.txt`, `oai_mrimeta01.txt`; select
   baseline MOAKS knees and their 3 series (sec 2); write `~/oai/links_needed.txt` (one S3 URL per line) and
   `~/oai/links_test5.txt` (5 knees).
6. **Timed test, then full download** (Eesh's keyring entry lets this run non-interactively; needs Eesh's OK to launch):

```bash
source ~/ndaenv/bin/activate
downloadcmd -dp <PKG_ID> -u <nda_username> -t ~/oai/links_test5.txt -d ~/oai/raw -wt 8      # measure MB/knee and MB/s
nohup downloadcmd -dp <PKG_ID> -u <nda_username> -t ~/oai/links_needed.txt -d ~/oai/raw -wt 10 \
      > ~/oai/dl.log 2>&1 < /dev/null &
downloadcmd -dp <PKG_ID> -u <nda_username> -t ~/oai/links_needed.txt -d ~/oai/raw --verify  # after it ends
```

   (Use `nohup ... &` from a plain shell, not the forbidden `cd X && setsid ... &` form.)
7. **After the store is built and checked:** remove the stored password
   (`python -c "import keyring; keyring.delete_password('nda-tools','<nda_username>')"`); keyrings.alt keeps it in a
   plaintext file under `~/.local/share/python_keyring/`. Keep `~/oai` at mode 700.

### 1.4 Size and time (GUESS until step 6 measures them)

Per knee, 16-bit DICOM (protocol figures, MEASURED from the OAI protocol summary in search results and memory; sizes GUESS):

| Series | Slices x matrix (approx.) | Raw MB/knee (GUESS) |
|---|---|---|
| SAG IW TSE FS | 37 x 444x448, 3 mm, FOV 160 mm | ~15 |
| COR MPR (of DESS WE) | ~80-100 x 384x384, 1.5 mm | ~25 |
| AX MPR (of DESS WE) | ~80-100 x 384x384, 1.5 mm | ~25 |
| (skipped) SAG 3D DESS WE | 160 x 384x384, 0.7 mm | ~47 |
| (skipped) COR IW TSE, SAG T2 MAP, COR T1 FLASH | | ~10 / ~55 / ~47 |

- **Per-series image03 rows:** ~65 MB/knee x 2,400 knees = **~150 GB** (GUESS), ~1-3 h at 15-40 MB/s (GUESS).
- **Whole-visit ZIP rows:** ~300-600 MB per subject-visit, ~1,500-2,400 subjects = **~0.5-1.4 TB** (GUESS), ~4-20 h.
- Box disk: 7.0 TB free on `/` (MEASURED `df -h ~` 21:55).
- Download speed from NDA S3 (us-east-1) to the lab is unknown: the 5-knee test in step 6 is the measurement.

---

## 2. Which series go in which slot, and how many knees

### 2.1 Our slots (MEASURED, `~/fast_build/code2/fleet_common.py`)
`VIEWS = [('Axial', 1), ('Sagittal', 0), ('Coronal', 1), ('Sagittal', 1)]` = v0 Axial fluid-sensitive, v1 Sagittal non-FS,
v2 Coronal FS, v3 Sagittal FS. C1 rule: per (plane, flag) the candidates are sorted by SeriesInstanceUID, the first
decodable one wins; an unknown flag leaves the view ABSENT, and absent views are a trained state (zeros, `present=False`,
contrast class 3). In the competition CSV `Fluid_Sensitive == Fat_Suppression` on every row (MEASURED: 14,010 / 10,361 for
both columns in `_data/train_series.csv`), i.e. our "FS" flag means fat suppressed.

Competition coverage (MEASURED, 4,407 train studies): AxFS 4,407, SagNonFS 4,266, CorFS 4,248, SagFS 4,150
(CorNonFS 3,406 and AxNonFS 857 have no slot).

### 2.2 OAI baseline knee protocol and mapping (series names MEASURED from `oai_mrimeta01.mexamtp`; protocol from the OAI protocol summary)

| Our slot | OAI series | Fit | Notes |
|---|---|---|---|
| v3 Sagittal FS | **SAG IW TSE FS** (TE 30 / TR 3200, 3 mm) | direct (PD-like FS) | `contrast_class` = 1 (PD), same as most competition FS PD |
| v2 Coronal FS | **COR MPR** (reformat of SAG 3D DESS WE, water excitation = fat suppressed) | good by the competition's own flag rule | contrast class 0 (TR 16 ms < 800), a mismatch vs competition PD/T2 FS |
| v0 Axial FS | **AX MPR** (reformat of DESS WE) | good by flag | contrast class 0, as above |
| v1 Sagittal non-FS | none; leave **ABSENT** | the protocol has no sagittal non-FS TSE | do NOT put DESS here (it is FS) or SAG T2 MAP (multi-echo, right knee only: a linspace over its files would mix echoes) |
| not used | COR IW TSE (non-FS), SAG 3D DESS WE (FS, but v3 already has the better match), COR T1 FLASH, T2 MAP | | not downloaded |

- Contrast class for the MPRs: leave the honest value (0) that `contrast_class` computes. The contract hashes the pixel
  path, and `contrast_class` is inside `pixel_sha`, so it must not be edited. `v{j}_ctr` is stored separately and could be
  overridden to 1 in the npz, but GUESS: the honest "different contrast" token lets the model absorb the DESS domain
  shift instead of confusing it with PD FS. Not worth a separate probe.
- Crop: `CROP_MM 140` on a 140-160 mm FOV keeps essentially the whole OAI image (fine).
- OAI DICOM files have **no `.dcm` extension** (GUESS from the legacy OAI layout; check after the test download).
  `load_series` globs `*.dcm`, so the prep script builds a symlink tree `~/oai/dcm/<knee_id>/<series_uid>/<NNNN>.dcm`.
- Both knees were imaged at baseline with these 3 series (inventory flags `mri_sagiwr/l`, `mri_cormprr/l`, `mri_axmprr/l`).

### 2.3 Which knees
- **Baseline (V00) knees with MOAKS: ~2,399** (MEASURED: nartaa used "the same 2,399 baseline OAI knees"; "the unsuccessful
  12-month arm is excluded", so do not add follow-up visits).
- Filter: `oai_kmrisemiquant01.visit` = baseline, MOAKS fields non-empty, `side` in {1, 2}; one row per (subject, side)
  (if a knee was read in several `readprj` projects, keep the row with the most non-missing MOAKS fields, ties to the
  higher project number); join to `oai_mrimeta01` and drop series with `qcreslt` = N; require all 3 series present.
  Expect ~2,200-2,400 knees (GUESS).
- Phase 2 (NOT before the freeze): the X-ray KL/JSN table covers ~9,000 baseline knees, which would 4x the OA-label rows
  but also the download and GPU time.
- Knee id: `oai_<src_subject_id>_<R|L>_v00` (lowercase prefix; see sec 4.4 for why).

---

## 3. Label map: OAI variables to our 12 labels

All codes below must be re-checked against the value counts in the downloaded `oai_kmrisemiquant01.txt` before the map
is frozen. The NDA dictionary gives field names but no code notes (MEASURED); code meanings come from the MOAKS protocol
(MEASURED, `oai/most_moaks.txt`).

MOAKS coding facts used (MEASURED, MOAKS reading protocol):
- Cartilage `m_cm_*` = "size.fullthickness" decimal: 0.0 normal; 1.0 / 1.1; 2.0 / 2.1 / 2.2; 3.0-3.3. The integer part is the
  area grade (1: <10 %, 2: 10-75 %, 3: >75 %); the decimal part is the full-thickness grade.
- Meniscus morphology: 0 normal, 1 signal not a tear, 2 radial, 3 horizontal, 4 vertical, 5 complex, 6 partial maceration,
  7 progressive partial maceration (follow-up only), 8 complete maceration; "scores of 0 or 1 are considered to not be
  tears". The OAI NDA table also has separate `m_*_sa` (signal abnormality) and `m_*_morph` fields, so check whether 1
  appears in `_morph` at all.
- Effusion-synovitis `m_eff_wk` 0-3 (0 physiological, 1 small, 2 medium, 3 large); Hoffa-synovitis `m_syic` 0-3.
- Osteophytes `m_os_*` 0-3 (OAI MOAKS has them; the MOST project did not score them).

| Our label | OAI source (field names MEASURED) | Proposed soft target (GUESS thresholds) | Tier |
|---|---|---|---|
| **PF OA** | cartilage `m_cm_mp`, `m_cm_lp`, `m_cm_amf`, `m_cm_alf`; osteophytes `m_os_pi`, `m_os_pl`, `m_os_pm`, `m_os_ps`, `m_os_fla`, `m_os_fma` | `max(area grade over PF regions, max PF osteophyte)` / 3, set to 1 if any PF full-thickness >= .1 | T1 (nartaa) |
| **Lateral OA** | `m_cm_clf`, `m_cm_plf`, `m_cm_alt`, `m_cm_clt`, `m_cm_plt`; `m_os_flc`, `m_os_flp`, `m_os_tl`; X-ray `xrjsl`, `xrosfl`, `xrostl` | as PF over the lateral TF regions; average with `xrjsl`/3 when the X-ray read exists | T1 (nartaa) |
| **Synovitis** | `m_syic` (Hoffa), `m_eff_wk` (effusion-synovitis) | `m_syic`/3; optionally `max(m_syic, m_eff_wk - 1)`/3 | T1 (nartaa) |
| **Medial OA** | `m_cm_cmf`, `m_cm_pmf`, `m_cm_amt`, `m_cm_cmt`, `m_cm_pmt`; `m_os_fmc`, `m_os_fmp`, `m_os_tm`; X-ray `xrjsm`, `xrosfm`, `xrostm` | as Lateral OA, medial regions | T2 |
| **Effusion** | `m_eff_wk` | 0 -> 0, 1 -> .3, 2 -> .8, 3 -> 1 | T2 |
| **Baker's** | `m_popcys` (WORMS `w_popcys` as fallback) | present -> 1 (check if graded 0-3; then grade >= 1 -> 1) | T2 |
| **Medial Meniscus** | `m_mmah_morph`, `m_mmbody_morph`, `m_mmph_morph`, `m_mmpr_tear`, `m_mm*_sa`, extrusion `b_mmm_extrusion` | any tear code 2-8 or root tear -> 1; signal only -> .2; else 0 | T3 |
| **Lateral Meniscus** | `m_lm*_morph`, `m_lmpr_tear`, `m_lm*_sa` | as medial | T3 |
| **ACL** | `m_acl_tear`, `m_acl_repair` (WORMS `w_acl_tear`) | complete tear or repair -> 1 (check the code values first) | T3 |
| **MCL** | only `w_mcl_tear` in WORMS-read knees (few) | **MASK** | never |
| **Contusion** | MOAKS BMLs are degenerative, not traumatic | **MASK** | never |
| **Fracture** | none | **MASK** | never |

Why tiers:
- **T1** is what nartaa did (MEASURED: "PF OA, Lateral OA and Synovitis received external supervision; unavailable finding
  cells were masked rather than treated as negatives. Positive class weights remained based on RSNA labels."). It is the
  only set with LB evidence (+.005 on his 224 single model; MEASURED table, not a controlled comparison per his own note).
- **T2** labels have direct, image-read MOAKS fields and are NOT supervised by N. This is where our own OAI members can add
  signal N does not already carry (sec 5).
- **T3** labels have the biggest definition gap (degenerative meniscal tears are very common in an OA cohort; report
  language may call them differently; ACL tears are rare in OAI), so they come last and only through a gate.
- Why nartaa left Medial OA, Effusion and the others out is NOT stated (GUESS: tried and did not help, or only weak
  labels were targeted). Treat T2 as a hypothesis to gate, not a known gain.

Soft ordinal targets: the metric is per-label rank AUC and the trainer uses BCE with soft targets already (mix50 is soft),
so grade/3 targets teach the ordering without choosing a positive threshold. Also compare OAI positive rates (grade >= 2)
with the mix50 rates per label (UNVERIFIED: mix50 rates not re-read, box down).

---

## 4. Trainer and store changes (proposal only; nothing applied)

### 4.1 What already works with zero code change (MEASURED, `fleet_train.py` md5 3e62434a per argv_frac75.sh)
- **Loss masking:** l.363-364 `msk = torch.isfinite(y).float(); yy = torch.nan_to_num(y); loss = (BCE * msk).sum() / msk.sum().clamp(min=1)`.
- **Eval masking:** `macro_auc` masks NaN labels per target (T9).
- **Fold handling:** `held = folds < 0 | --exclude`; training ids are `folds[s] != A.fold`. An OAI row with fold 9 is
  trained in every fold and never validated or gold-scored. `--eval_labels` is only asserted for `va_ids`.
- **Absent views:** both the default and heavy paths (`fleet_aug.heavy_study`) zero-fill a missing view with
  `present=False`, contrast 3.
- **Store contract:** `build_study` / `study_npz_dict` from the unchanged `fleet_common.py` give the same `STORE_CONTRACT`
  (`pixel_sha 6f49c74fc48f3345`, MEASURED in the frac75 log), so `check_store_contract` passes and the worker needs nothing.

So the minimum is: (a) `labels_frac75_oai.csv` = the frac75 CSV + OAI rows (NaN where masked); (b) `folds5_oai.csv` =
folds5 + OAI rows with fold 9; (c) `~/fleet_store_oai/` = symlinks to every `~/fleet_store/*.npz` plus the OAI npz files;
(d) run the frac75 argv with those three paths swapped.

### 4.2 Why a small patch is still advisable
1. **Loss weight per cell.** With `bs 2` (TTR default) and the finite-cell mean, a batch of two OAI T1 rows has 6 finite
   cells, so each cell weighs 1/6 against 1/24 for two RSNA rows: OAI cells get up to **4x** the per-cell weight
   (MEASURED arithmetic from the code). nartaa's normalisation is unknown. Option `--mask_norm cells` divides by all cells
   instead. For batches with no NaN it is identical to the current loss.
2. **Epoch time.** About 3,470 RSNA training rows per fold (MODEL: 4,333 non-held x 0.8) plus ~2,400 OAI = x1.69 training
   time. `--ext_frac 0.5` draws a fresh half of the OAI rows every epoch (x1.35).
3. **Compliance assert.** `probe_ids = sorted(ids)[:4]` and those studies' npz ship in the Kaggle dataset `parity/`
   folder (MEASURED, `decode_sha` docstring). With the `oai_` prefix they sort after `1.2.826...` ids, but the patch
   asserts it so a renamed id can never ship OAI pixels.

### 4.3 Patch (OPT-IN; with the defaults the trainer is bit-for-bit unchanged)

```diff
@@ argparse block (after --grad_ckpt)
+ap.add_argument('--mask_norm', default='finite', choices=('finite', 'cells'))  # OAI-1 'cells': BCE sum / all cells
+ap.add_argument('--ext_prefix', default='oai_')                                 # OAI-2 external-row id prefix
+ap.add_argument('--ext_frac', type=float, default=1.0)                          # OAI-3 share of external rows per epoch
@@ after: gold58 = ([s for s in gold_ids ...
+EXT = [s for s in tr_ids if s.startswith(A.ext_prefix)]                                          # OAI-2
+assert not [s for s in va_ids + gold_ids if s.startswith(A.ext_prefix)], 'external rows must never be validated/gold-scored'
+assert not [s for s in sorted(ids)[:8] if s.startswith(A.ext_prefix)], 'decode-probe/parity npz ship to Kaggle: RSNA only'
+if EXT:
+    print(f'OAI opt-in: {len(EXT)} external train rows | mask_norm {A.mask_norm} | ext_frac {A.ext_frac}', flush=True)
@@ before def dl(...)
+class ExtSampler(torch.utils.data.Sampler):                     # OAI-3: fresh external subset each epoch (main process)
+    def __init__(self, ids, frac):
+        self.core = [i for i, s in enumerate(ids) if not s.startswith(A.ext_prefix)]
+        self.ext = [i for i, s in enumerate(ids) if s.startswith(A.ext_prefix)]
+        self.k = int(round(frac * len(self.ext)))
+    def __len__(self):
+        return len(self.core) + self.k
+    def __iter__(self):
+        pick = list(np.random.choice(self.ext, self.k, replace=False)) if self.k else []
+        return iter(np.random.permutation(self.core + pick).tolist())
@@ def dl(ids, tr):
-    return torch.utils.data.DataLoader(DS(ids, tr), batch_size=A.bs, shuffle=tr, drop_last=tr,
+    smp = ExtSampler(ids, A.ext_frac) if (tr and EXT and A.ext_frac < 1) else None               # OAI-3
+    return torch.utils.data.DataLoader(DS(ids, tr), batch_size=A.bs, shuffle=(tr and smp is None), sampler=smp, drop_last=tr,
@@ after: if not A.grad_ckpt: ... (args block)
+if EXT or A.mask_norm != 'finite':                                                                # OAI-1..3
+    args.update(mask_norm=A.mask_norm, ext_prefix=A.ext_prefix, ext_frac=A.ext_frac, n_ext=len(EXT))
@@ training loop
-        loss = (F.binary_cross_entropy_with_logits(logit.float(), yy, reduction='none') * msk).sum() / msk.sum().clamp(min=1)
+        _den = msk.sum().clamp(min=1) if A.mask_norm == 'finite' else float(msk.numel())          # OAI-1
+        loss = (F.binary_cross_entropy_with_logits(logit.float(), yy, reduction='none') * msk).sum() / _den
```

Notes:
- `steps = A.epochs * len(tr_dl) // A.accum` picks up the sampler length automatically, so the cosine schedule follows.
- `np.random` in the main process is seeded by `--seed`, so the sampler is reproducible. Opt-in runs change the RNG stream,
  which is expected.
- Parity test before any GPU use (CPU, tiny): run with no external rows and the default flags and check that the OOF bytes
  match the current trainer (the `t_b2_default_parity.py` pattern); run with 20 OAI rows, `--device cpu --time_epochs 1`,
  and check the asserts fire on a mis-prefixed id.
- If mix50/frac75 labels already contain NaN cells (UNVERIFIED), `--mask_norm cells` also slightly reweights RSNA rows.
  Check `labels.isna().sum()` first.
- Pack / K-builder: the member key needs a new `train_labels` tag (e.g. `s1046f75oai`), the same kind of picker change
  that set D needed in the K9 kit. The worker is unchanged (same model, contract and TTA).

### 4.4 Store build (prep script outline, box CPU, `nice -n 19`)

```python
import fleet_common as fc            # the SAME file, unmodified (contract and pixel_sha unchanged)
# per knee kid = 'oai_<subject>_<R|L>_v00', with ~/oai/dcm/<kid>/<uid>/<NNNN>.dcm symlinks to the raw files
rows = [dict(SeriesInstanceUID=uid_ax,  Anatomical_Plane='Axial',    Fluid_Sensitive=1, Fat_Suppression=1),
        dict(SeriesInstanceUID=uid_cor, Anatomical_Plane='Coronal',  Fluid_Sensitive=1, Fat_Suppression=1),
        dict(SeriesInstanceUID=uid_sag, Anatomical_Plane='Sagittal', Fluid_Sensitive=1, Fat_Suppression=1)]
st = fc.build_study(pd.DataFrame(rows), IMG_ROOT, kid, return_bytes=True, stats=stats, errs=errs)
assert all(fc._plane_from_header(f'{IMG_ROOT}/{kid}/{r["SeriesInstanceUID"]}') == r['Anatomical_Plane'] for r in rows)
np.savez(f'{OUT}/{kid}.npz', **fc.study_npz_dict(st))     # OUT = ~/oai/store (mode 700), NEVER ~/fleet_store
```

- Size: ~11 GB for 4,407 RSNA studies (MEASURED `du -sh ~/fleet_store`), so ~2.5 MB per npz, ~6 GB for 2,400 knees (MODEL).
- Time: each knee decodes at most 3 x 32 slices; ~1-2 s/knee/core (GUESS), so ~10-30 min on 8-16 processes.
- QC: per-view presence counts, a montage of 20 random knees (v0/v2/v3 centre slices, saved on the box only), and a
  check that AX and COR MPR come out at plausible sizes after the 140 mm crop.
- Union dir for training: `~/fleet_store_oai/` = symlinks to `~/fleet_store/*.npz` + `~/oai/store/*.npz`. `~/fleet_store`
  stays untouched, so no K-build or Kaggle pack can ever pick up an OAI file.

---

## 5. Timeline and expected gain

### 5.1 GPU cost (MODEL from MEASURED epoch times in NOVEL_PLAN_10-07 sec 3: 224 gc0 10.5 min/epoch, 288 16.0-16.8 min/epoch)

Assume ~80 % of an epoch is training (scales with rows) and ~20 % eval (fixed) (GUESS split).

| Set | min/epoch | Epochs | h/fold | 5 folds |
|---|---|---|---|---|
| 224 frac75 today | 10.5 | 21 | 3.7 | 18 h |
| 224 + OAI all rows (x1.69 train) | ~16.3 | 21 | ~5.7 | **~29 h** |
| 224 + OAI `--ext_frac .5` (x1.35) | ~13.4 | 21 | ~4.7 | **~24 h** |
| 288 + OAI all rows | ~25.4 | 21 | ~8.9 | ~44 h |
| 288 + OAI `--ext_frac .5` | ~21.0 | 21 | ~7.4 | ~37 h |

Kaggle T4 and Vast are not usable for OAI (data must stay on the box), so this is V100-only.

### 5.2 Calendar (MODEL; the V100 order is Eesh's call)

| When (IST) | What | Who |
|---|---|---|
| 10-09 morning | NDA password, rebuild cart, package ID, step A (metadata) | Eesh (~30 min) |
| 10-09 | selection + links, 5-knee timed test, prep script, label map frozen from value counts, trainer patch + CPU parity tests | Claude (CPU) |
| 10-09 to 10-10 | full download (1-20 h, GUESS) and store build (<1 h CPU) | box CPU + network |
| first free V100 slot | **probe:** 224, fold 0, seed 1043, T1+T2 labels, `--mask_norm cells --ext_frac .5`, paired with `f0_s1043_ttr_frac75` (same seed, only labels/store/folds differ) | ~4.7 h |
| probe + 1 h | gate (5.3); per-label keep/mask decision for folds 1-4 by the pre-registered rule | Claude |
| +19 h | folds 1-4 | V100 |
| then | K11N = parent + [5x288 mix50 + 5x224 frac75+OAI] + N, 3 weights; build/A0 ~3 h; submit | Claude + Eesh |

- If the V100 goes to OAI right after the running 320 work (Q3b ends ~10-10 22:00 if G320' passed), the probe reads
  ~10-11 04:00, folds 1-4 end ~10-12 00:00 and K11N can be submitted 10-12. A 288 OAI set (~37-44 h) would then still end
  ~10-13/14, before the 10-20 freeze. **Both OAI sets plus all of Q4 do not fit**; OAI likely beats a 288 frac75 reseed
  (+.0004-.0006 SOLO MEASURED analog), but Eesh decides.
- If the probe preempts Q3b (320 folds), the OAI 224 set is ready ~10-11 evening. Only worth it if G320' failed or the
  320 trajectory stays behind 288.

### 5.3 Pre-registered probe gate (proposal; numbers GUESS, write them down before the probe starts)
- **G1 no harm:** OOF on the masked labels (MCL, Contusion, Fracture plus any label not trained from OAI) vs v4 >= paired
  frac75 f0 - .002.
- **G2 mechanism:** gold74 mean AUC over the OAI-supervised labels >= paired + .005, AND the OOF-vs-v4 mean over the same
  labels >= paired - .001.
- **G3 guard:** gold74 macro >= paired - .001.
- **Per-label rule for folds 1-4:** keep a T2/T3 label only if both its gold74 delta >= 0 and its OOF-vs-v4 delta >= -.003;
  otherwise mask it in folds 1-4 (fold 0 stays as trained). Gold74 per-label SE is large (~.02-.03, GUESS), so this is a
  weak screen; the OOF leg (870 studies per fold) carries most of the weight.
- Caveat (MEASURED from nartaa's post): "epoch 14 improved gold AUC while losing on the public board", and his fresh
  native-320 OAI retrain scored .946 vs .949 despite "encouraging gold diagnostics". Gold58/74 is a weak selector.

### 5.4 Expected gain (MODEL / GUESS)

- **Anchor (MEASURED):** nartaa +.005 public (.940 -> .945) on a 224 single model from T1 OAI supervision, not compute-matched
  (updates 4,336 -> 6,736).
- **Our fleet SOLO** (5x224 frac75 + OAI vs 5x224 frac75): GUESS +.002 to +.004 for T1, +.001 to +.002 more if T2 passes.
  Fold-ensembling and our stronger base (.944 SOLO vs his .940) likely shrink the gain.
- **On top of K9N (P/F/N .10/.30/.60):** N already carries OAI signal on exactly the T1 labels, and our OAI members will
  correlate more with N on those labels, so the blend keeps only a fraction of the SOLO gain there: **+.0005 to +.0015**
  (MODEL: about w_F 0.3-0.5 of the per-label F gain, on 3 of 12 labels). T2 labels are not in N, so more of their gain
  survives: **+.0005 to +.0015** more if they pass the gate. **Total +.001 to +.003** (GUESS), with maybe a 25 % chance of
  ~0 or slightly negative (domain shift: 3T Siemens Trio OA cohort, DESS-WE MPRs in FS slots, an absent sagittal non-FS
  view).
- **Rank context (MEASURED counts from PUBLIC949_ASSESSMENT):** .950 = rank 124-215, .951 = 110-123. A K11N at .951-.952
  would be safely silver. Gold (.959) stays out of reach through this lever.
- **Effort:** Eesh ~30 min; Claude ~1 day of CPU/scripting; V100 ~24-29 h (224 set). Lower expected gain per GPU-hour than
  K9N itself (0 GPU), but higher than any remaining reseed on the queue.

### 5.5 Cheaper variant (not recommended first)
Fine-tune each existing frac75 224 fold checkpoint for ~5 epochs on RSNA + OAI at low LR: ~1/4 of the GPU (~7 h for 5
folds, MODEL). It needs an `--init_ckpt` path for `--model ttr` (today only `--f2` has one), so it is a second patch, and
members stay close to set D (less new signal). Use it only if the V100 calendar gets tight after the probe passes.

---

## 6. Compliance checklist (OAI DUC + our rules)
- OAI raw data, the symlink tree and OAI npz stay in `~/oai/` (mode 700) on the box. Never in `~/fleet_store`, never in
  any Kaggle dataset or kernel, never on Vast, never shared with swapiy or kmrsuman (they have no DAR). Model weights
  trained on OAI are what ships (the same pattern as nartaa's public weights; the host declined to judge OAI's own terms,
  MEASURED 741819).
- The decode-probe / parity assert (sec 4.3) prevents OAI pixels from riding along in the Kaggle dataset `parity/` folder.
- Use only `image03` rows from the OAI collection, baseline visit, the 3 needed series.
- Keep the NDA acknowledgement text and DOI (nartaa cites "Dataset identifier(s): 10.15154/0hcg-f676") for any write-up.
  Prize-stage NDA duties (an NDA Study, as nartaa did with #3407) only matter at top-10.
- Delete the keyring password after the download.

## 7. Open items that need the box or the package
1. `downloadcmd -h` on 0.7.0 (flags above are from GitHub main).
2. image03.txt: per-series or per-visit files, the `image_description` strings, the visit coding and file naming.
3. MOAKS value counts (ACL, popliteal cyst and `_morph` codes) before the label map is frozen.
4. mix50 per-label positive rates and NaN counts (`~/novel/out/*.txt`, label CSVs) to compare against the OAI rates.
5. Download throughput from the 5-knee test.
