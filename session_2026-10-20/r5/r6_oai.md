# r6_oai.md: R6 and S6fall served on OAI V00 knees, scored against expert MOAKS reads (10-20, r5 task A, Claude)

Research and measurement only. Nothing was submitted, pushed, committed or uploaded, and no dataset was created. No scheduled
task was touched, and no PREREG, HANDOFF, PROGRESS or session_2026-10-19/a20 file was touched. No em dashes.

**OAI governance:**
- Everything that reads OAI ran on the box (training@192.168.10.30) and printed aggregates only.
- New box files live under ~/r5_oai only (mode 700).
- One slip is listed in sec 9.

**Box GPU:**
- Inference only, one job at a time.
- Every step waited for an empty `nvidia-smi` (2 checks 60 s apart). At first the GPU was busy with a sibling task's job (~/r5_oai/taskC). I waited for it, and I killed no process I did not start.

**Tags:**
- MEASURED: measured here, log named.
- CLAIMED: an older note says so, with the source.
- MODEL: arithmetic on stated inputs.
- GUESS: judgement.

**Units:**
- "OA-mean dAUC" is the change in the mean AUC of the 3 OA labels.
- Macro-12 equivalent = OA-mean dAUC x 3 / 12.

## 1. Verdict

1. **The task's premise was wrong. The OAI V00 raw DICOMs are no longer on the box** (MEASURED):
   - `~/oai_raw/image03/00m` holds 0 tar.gz files. `~/oai_build/raw_deletions.csv` lists the V00 knees.
   - `build_oai_store.py --delete_raw 1` pruned them on 10-09.
   - The only V00 image form left is the 320 px JPEG fleet store `~/oai_store/<kid>.npz`: <= 32 slices per series, 3 series, no geometry.
   - So R6 could not be served on real OAI DICOMs. I rebuilt DICOM series from the store with fabricated geometry (sec 2), the same idea as the 10-13 Raptor approximation (`~/raptor2/approx_parity.py`).
   - The exact R6 serve code then ran on the rebuilt DICOMs unchanged: bundle fe055406, 4 folds, flip TTA, cell 7 / 8 shas asserted.
2. **The rebuild path is faithful enough on RSNA gold58** (MEASURED, `box_r6_final.log` part A, `box_g58_bias_r6.log`):
   - Macro-12 AUC: exact .9366, approx .9409. Mean per-label Spearman between exact and approx outputs is .933.
   - But the rebuild **lowers R6's OA coupling**:
     - rho falls by .041 (90 % [-.080, .000]).
     - The coupling gap (rho - q) falls by .050 (90 % [-.096, +.004]).
     - The gold58 E-OA delta falls by .0014 at a .1 and .0021 at a .2 (sd .0019 / .0039).
   - Read every OAI E-OA number for R6 below as biased low by about that much.
3. **R6 reads OAI well, and the medial / lateral geometry is right** (MEASURED, n 1,704 OAI knees):

   | Label | AUC |
   |---|---|
   | ACL | .945 |
   | Medial Meniscus | .944 |
   | Lateral Meniscus | .953 |
   | Effusion >= 2 | .958 |
   | Baker's | .917 |
   | OA P_FT M / L / PF | .891 / .811 / .859 |

   - In the swap check, the medial OA label scores .891 from the medial output and .654 from the lateral output.
   - The worker's side check looked at 1,736 studies and flipped 1.
   - R6 beats the OAI-free fleet reference D5 on 8 of 9 RSNA labels.
4. **E-OA on R6 at RSNA expert coupling is zero to negative** (MEASURED, n 1,704; skeptic method `box_reweight.py` / `box_ladder_transfer.py`):
   - Unweighted OAI (label phi .04): a .1 +.0040, a .2 +.0064 OA-mean, both P 1.00.
   - Reweighted to the gold74 8-cell OA joint (R1, label phi .32):

     | Proxy | a .05 | a .1 | a .15 | a .2 |
     |---|---|---|---|---|
     | P_FT | -.0006 | **-.0017** | -.0034 | **-.0057** |
     | P_AREA | -.0004 | -.0014 | -.0030 | -.0050 |

     All have P(>0) <= .08 (sd .0005 at a .1, .0011 at a .2).
   - R6's R1 gap (rho - q) is +.006. With q about equal to rho, decorrelation has nothing left to gain (skeptic_eoa theory: E-OA helps only if q < rho).
   - Adding back the rebuild bias from item 2 (MODEL):
     - a .1 is about -.0003 OA-mean (sd ~.002), or -.0001 macro.
     - a .2 is about -.0036 OA-mean (sd ~.004), or -.0009 macro.
   - The same reweighting gives D5 +.0017 / +.0021 at a .1 / .2 on these knees. The skeptic got +.0016 / +.0017 on all 2,399, so the method reproduces.
5. **A two-family OAI-free blend that includes R6 behaves like the skeptic's blends** (MEASURED):
   - R6 + D5, 50/50 rank, under R1: a .1 +.0011 (P .99), a .2 +.0004 (P .65), a .3 -.0031.
   - The skeptic's blend3 gave +.0010 / +.0003 (CLAIMED `skeptic_ladder/box_ladder_transfer_blend.log`).
6. **S6fall (the c4 B_MODEL) and the c4-style .8 R6 / .2 S6fall rank blend behave like R6** (MEASURED, n 1,704 common knees; sec 5):
   - S6fall's OAI AUCs are about R6's: ACL .965, Medial OA P_FT .899, Lateral OA P_FT .786.
   - On gold58 the rebuild costs S6fall a bit more AUC (-.007 macro) but leaves its OA coupling about unchanged (gap -.011).
   - Under R1, P_FT:

     | Model | a .1 | a .2 |
     |---|---|---|
     | S6fall | -.0027 | -.0079 |
     | .8 / .2 blend | **-.0016** | **-.0055** |

     All have P(>0) <= .04. P_AREA: blend -.0009 / -.0042.
   - **Caveat (sec 5.4):** on RSNA gold58 (exact serve, n 58) both models GAIN from E-OA (+.0035 to +.0067 OA-mean), and their OA outputs are far more coupled there (rho .78-.85) than on OAI (.53-.55). OAI is a pessimistic ruler for these two models and gold58 an optimistic, noisy one.
7. **What this means for the in-flight entries** (MODEL / GUESS; detail in sec 6):
   - This is the first OAI-free measurement on real c4 members: R6 + S6fall, about .4 of c4. Raptor (~.6) is OAI-trained and cannot be measured. Measured directly, those two members lose at RSNA coupling on OAI. A two-family OAI-free proxy (R6 + D5) still gains a little at a .1.
   - This does not overturn the skeptic ladder, but it shades it down, eoa20 most:

     | Entry | Macro | P(>0) | Was |
     |---|---|---|---|
     | eoa10 | about +.0001 (range -.0001 to +.0004) | about .6 | +.0003, P .75 |
     | eoa20 | about -.0001 (range -.0009 to +.0003) | about .4 | +.0001, P .55-.6 |

   - P(eoa20 beats eoa10 on hidden): about .35-.4 (GUESS; the mechanics lens had .58-.61).
   - The picks rule does not change: A = API best-id under R1, B = R6 57248084. If neither eoa entry moves the best-id, E-OA is closed, as already planned.
   - A public-only eoa20 win is worth less than it looks. Its public paired read sd is ~.0004, and its hidden expectation here is below eoa10's. That is a tie-break input for Eesh, not a rule change.

## 2. Recipe: OAI store -> DICOM -> R6 / S6fall serve

### 2.1 What exists on the box (MEASURED)
- **OAI images:** `~/oai_store/<kid>.npz` for all 2,399 MOAKS-labelled V00 knees. The format is fleet_common r3:
  - views v0 = Axial FS (AX_MPR), v2 = Coronal FS (COR_MPR), v3 = Sagittal FS (SAG_IW_TSE);
  - <= 32 slices (linspace subsample), a central 140 mm crop resized to 320 px, a 0.5 / 99.5 % window, uint8, JPEG q92.
- **What the store does not keep:**
  - COR_IW_TSE (Coronal non-FS), because fleet_common has no such view;
  - any geometry: IPP, IOP, spacing.
- **Raw tarballs:** the only ones left are 10 follow-up tarballs (24m / 48m, a few right knees). `proto_stats.py` read their headers (aggregates only) to get the protocol constants:

  | Series | Matrix | Spacing |
  |---|---|---|
  | SAG_IW_TSE | 444 x 448 | 0.357 mm, slices 3.0 mm |
  | COR_IW_TSE | 384 | 0.365 mm, 3.0 mm |
  | AX_MPR | 384 | 0.365 mm, 1.5 mm, 70-75 slices |
  | COR_MPR | 384 | 0.365 mm, 1.5-1.64 mm, 70-75 slices |

  - No Laterality / ImageLaterality tags.
  - Right knees sit at IPP x ~ -60 to -70 mm.
  - The IOPs are cardinal, except that 1 of the 2 COR_MPR series has row direction -x (mirrored).
- **R6 input contract** (`~/r6g58/r6g58_run.py`, compact_infer 8401812c, `training/native_cache.prepare`):
  - layout `<data>/<split>.csv` (StudyInstanceUID), `<split>_series.csv` (StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive, Fat_Suppression, Anatomical_Plane) and `<split>_series/<study>/<series>/*.dcm`;
  - headers: IPP / IOP / PixelSpacing / InstanceNumber / Photometric;
  - side: a single laterality tag, else the sagittal IPP x median, then the bundle's side classifier;
  - views v4: SKM-TEA joint anchors on the sagittal FS series, projected into the other series through world coordinates.

### 2.2 `box/build_synth.py` (box CPU; 2,399 OAI knees in 828 s with 16 processes; gold58 in 25 s)

**Per stored view, the DICOM series is written as:**
- **Pixels:** the uint8 x 16 as uint16 MONOCHROME2. This is linear; the worker re-quantizes by min / max.
- **In-plane spacing:** kr x ps / 320 with kr = min(rows, max(32, round(140 / ps))). This is exact for the store crop and gives 0.4375 mm for every OAI series.
- **Slice positions:** stored slice k sits at native index linspace(0, n_native - 1, 32).round(), where n_native = n_kept from `oai_series_meta.csv`. The step is SAG 3.0, AX / COR MPR 1.5 mm.
- **IOP:** cardinal (Sag 0,1,0 / 0,0,-1; Cor 1,0,0 / 0,0,-1; Ax 1,0,0 / 0,1,0).
  - **Mirror repair:** the R6 bundle's own side classifier (`side_cls.pt`, `side_common.side_inputs`) scores each Cor / Ax series on its canonical slices for the known side. A score < 0 means the series is written with row direction -x, so that canonicalization un-mirrors it.
  - Result: 3 of 2,399 AX_MPR and 9 of 2,399 COR_MPR series were flipped. The OAI score distribution was negative .25 %, |score| < 2 in 1.25 %, median 5.1.
  - On gold58, with the real IOP, the same classifier was positive for 98.2 % of 114 series and none was ambiguous, so it is a reliable detector on store-degraded images.
- **Centre:** one centre per study at (x0, 0, 0). OAI uses x0 = -65 mm for right knees and +65 mm for left. No laterality tag, as in the raw OAI headers, so the side comes from the sagittal IPP x and then the worker's side check (1 of 1,736 flipped).
- **Validation copy:** gold58 (RSNA, from `~/fleet_store`) was rebuilt the same way with views 0 / 2 / 3 only (the Sagittal non-FS view was dropped to match OAI). For RSNA:
  - real native spacing, n and step;
  - the real IOP rounded to cardinal axes;
  - the real sagittal x;
  - the real single laterality tag.

**Known losses of the rebuild (all OAI knees):**
- COR_IW_TSE is absent.
- Slices are subsampled (SAG 37 -> 32; MPR ~71 -> 32, so ~3.2 mm spacing).
- A JPEG / window round trip.
- Cross-series planning offsets are lost. All series share one centre (10-20 mm offsets in the raw sample), so the anchors are projected with that error. SKM-TEA still found the joint in 1,729 of 1,760 studies; the rest used the slab prior.

**Run-time CPU check:** `box/parse_check.py`. The mainline native prepare and geometry bank accept the rebuilt series (3 series per study, 0 errors).

### 2.3 Serve
- **R6:** `box/r5_run.py` is `~/r6g58/r6g58_run.py` with data / split / ids / out / deadline as arguments, plus `--keep-order`.
  - Unchanged: the cell 7 / 8 sha asserts, the bundle fe055406 sha asserts, the WORKER_ARGS exec of the kernel's own cell 7 + cell 8 (flip TTA forced), and the worker command and env.
  - Loader setting: 1 loader worker + expandable segments, as run B.
  - Worker sources are re-written from cell 7 into `~/r5_oai/r6src` (sha-checked).
- **One worker run, gold58 approx first:** 58 gold58 approx studies, then the OAI knees in a stratified order (`box/select_oai.py`). Strata = OA P_FT 8-cell x any meniscus tear x effusion >= 2. The order is proportional, so every prefix keeps the OAI joint distribution. Deadline 2.75 h.
- **S6fall:** `box/r5_s6.sh` runs the c4 B_MODEL, nartaa E_prior_s42_fall.pt (sha 0c7297d5), through `~/s6seed/parity/ci_gpu_launcher.py` (heliosli compact_infer 34f433fe), with the worker args of `run_gold58_serve_gpu.sh`. Same id list, deadline 1 h.
- **Chain:** `box/r5_chain.sh`.

## 3. Rebuild validation on RSNA gold58 (MEASURED; `box_r6_final.log` part A, `box_g58_bias_r6.log`)

| R6, gold58 (n 58) | exact (real DICOM, run_b) | approx (store rebuild) |
|---|---|---|
| macro-12 AUC vs gold | .9366 | .9409 |
| OA-trio mean AUC | .9239 | .9369 |
| mean per-label Spearman exact vs approx | | .933 (min Synovitis .829, max Contusion .987) |
| OA model rho (normal scores) | .778 | .737 |
| q (cross-predictiveness) / gap rho - q | .686 / .092 | .696 / .041 |
| gold58 E-OA OA-mean dAUC a .1 / a .2 | +.0035 / +.0067 | +.0021 / +.0045 |

**Paired bootstrap, approx - exact:**

| Quantity | approx - exact | 90 % interval |
|---|---|---|
| rho | -.041 | [-.080, .000] |
| gap | -.050 | [-.096, +.004] |
| E-OA a .1 | -.0014 | [-.0036, +.0027] |
| E-OA a .2 | -.0021 | [-.0077, +.0050] |

**Read:**
- The rebuild keeps R6's per-label signal. The AUC differences are within gold58 noise, and the largest, Fracture +.04 and PF OA +.03, are noise at n 58.
- But it removes part of R6's shared OA component. That biases the E-OA deltas against E-OA. The bias is measured only at n 58, so its size is uncertain (sd ~.002 OA-mean at a .1).
- The exact R6 gold58 gap .092 matches the CLAIMED .09 (mechanics.md).

## 4. R6 on OAI (MEASURED; `box_r6_final.log` part B; n 1,704 OAI knees, the stratified prefix)

### 4.1 Per label (AUC vs MOAKS-derived binary targets)

| label (target) | n | prevalence | R6 | D5ref (OAI-free fleet) |
|---|---|---|---|---|
| ACL (tear code >= 1) | 1,174 | .079 | .945 | .938 |
| Medial Meniscus (tear / maceration / root) | 1,704 | .407 | .944 | .934 |
| Lateral Meniscus | 1,704 | .209 | .953 | .933 |
| Effusion (grade >= 2) | 1,178 | .198 | .958 | .957 |
| Effusion (grade >= 1) | 1,178 | .576 | .877 | .882 |
| Baker's (popliteal cyst) | 459 | .392 | .917 | .898 |
| Synovitis (Hoffa >= 2) | 1,177 | .110 | .800 | .804 |
| Medial OA P_FT / P_AREA | 1,704 | .216 / .470 | .891 / .845 | .879 / .830 |
| Lateral OA P_FT / P_AREA | 1,704 | .195 / .368 | .811 / .813 | .779 / .770 |
| PF OA P_FT / P_AREA | 1,704 | .433 / .748 | .859 / .872 | .813 / .845 |

**How the targets were built:**
- The OA trio uses the `oai_eoa_check.py` proxies:
  - P_FT: any full-thickness loss in the compartment;
  - P_AREA: any subregion with area grade >= 2;
  - every compartment must have >= 1 read region.
- The other labels use `oai_labels_T123.csv` >= .5, built by `build_oai_labels.py`.

**Swap check, P_FT:**

| Target | From the matching output | From the opposite output |
|---|---|---|
| Medial OA | .891 | .654 |
| Lateral OA | .811 | .631 |
| Medial Meniscus | .944 | .536 |
| Lateral Meniscus | .953 | .535 |

The geometry is not mirrored.

### 4.2 OA coupling and E-OA, unweighted (OAI coupling: label phi ML .059, MPF .071, LPF -.008)

| model | model rho ML / MPF / LPF (mean) | P_FT dAUC a -.1 / .05 / .1 / .15 / .2 / .3 | P_AREA a .1 / .2 |
|---|---|---|---|
| R6 | .616 / .555 / .466 (.546) | -.0053 / +.0022 / +.0040 / +.0055 / +.0064 / +.0067 | +.0033 / +.0053 |
| D5ref | .743 / .773 / .747 (.754) | -.0070 / +.0034 / +.0065 / +.0095 / +.0122 / +.0164 | +.0059 / +.0112 |
| R6 + D5 (.5 / .5 rank) | .695 / .689 / .632 (.672) | -.0073 / +.0034 / +.0065 / +.0093 / +.0117 / +.0149 | +.0058 / +.0108 |

- Bootstrap: B 2000; all positive a have P(>0) 1.00, and a -.1 has P 0.00.
- **R6 PF OA loses at every positive a:** per-label -.0020 at a .1, -.0057 at a .2 (P_FT), against about +.007 for Medial and Lateral.
- R6's OA outputs are much less coupled on OAI than the fleet model's: .546 against .754. On gold58 the exact R6 has .778, so OAI also lowers R6's coupling beyond the rebuild's -.04. GUESS: the DESS MPR coronal / axial inputs and the missing coronal non-FS series make R6 lean more on compartment-specific evidence.

### 4.3 Skeptic coupling-matched reweighting (R1 = gold74 8-cell OA joint; tilt = coupling only, OAI marginals kept; B 300)

OA-mean dAUC (sd, P>0) per cell:

| model | proxy | regime | ESS | phi | rho | q | gap | a .05 | a .1 | a .15 | a .2 | a .3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R6 | P_FT | R1 | 1260 | .32 | .597 | .592 | +.006 | -.0006 (P .01) | **-.0017** (sd .0005, P .00) | -.0034 | **-.0057** (sd .0011, P .00) | -.0120 |
| R6 | P_FT | tilt 1.0 | 1452 | .28 | .590 | .565 | +.025 | -.0004 (P .09) | -.0011 (P .02) | -.0025 | -.0045 (P .00) | -.0102 |
| R6 | P_FT | tilt 1.5 | 1200 | .41 | .610 | .655 | -.045 | -.0015 | -.0036 | -.0063 | -.0097 | -.0184 |
| R6 | P_AREA | R1 | 836 | .32 | .628 | .621 | +.007 | -.0004 (P .08) | -.0014 (P .01) | -.0030 | -.0050 (P .00) | -.0115 |
| R6 | P_AREA | tilt 1.0 | 1479 | .34 | .593 | .622 | -.029 | -.0006 | -.0018 | -.0035 | -.0058 | -.0126 |
| D5ref | P_FT | R1 | 1260 | .32 | .780 | .700 | +.080 | +.0010 | +.0017 (P 1.00) | +.0021 | +.0021 (P .99) | +.0005 |
| D5ref | P_FT | tilt 1.5 | 1200 | .41 | .790 | .754 | +.035 | +.0004 | +.0002 (P .67) | -.0004 | -.0014 (P .05) | -.0054 |
| R6 + D5 | P_FT | R1 | 1260 | .32 | .712 | .646 | +.065 | +.0008 | +.0011 (P .99) | +.0010 | +.0004 (P .65) | -.0031 |
| R6 + D5 | P_FT | tilt 1.0 | 1452 | .28 | .708 | .621 | +.086 | +.0011 | +.0016 (P 1.00) | +.0018 | +.0014 (P .90) | -.0014 |
| R6 + D5 | P_FT | tilt 1.5 | 1200 | .41 | .723 | .705 | +.017 | +.0000 | -.0006 (P .08) | -.0017 | -.0035 | -.0092 |
| R6 + D5 | P_AREA | R1 | 836 | .32 | .726 | .681 | +.045 | +.0010 | +.0015 (P 1.00) | +.0017 | +.0013 (P .93) | -.0015 |

- The D5ref R1 rows reproduce the skeptic's D5 (+.0016 / +.0017 at a .1 / .2 on 2,399 knees, CLAIMED `skeptic_eoa_work/box_reweight.log`).
- So the method and labels are unchanged, and the R6 numbers are not an artifact of the analysis code.
- **Bias-adjusted R6 at R1 (MODEL):** add the gold58 rebuild bias (+.0014 at a .1, +.0021 at a .2; sd .0019 / .0039).

  | a | OA-mean | Macro | P(>0) |
  |---|---|---|---|
  | .1 | about -.0003 (sd ~.002) | about -.0001 | ~.45 |
  | .2 | about -.0036 (sd ~.004) | about -.0009 | ~.2 |

- In gap terms, the exact-path R6 R1 gap is about +.006 + .050 = +.056. That sits between s1045 (+.044) and s1044 (+.068) in the skeptic ladder, where a .1 was +.0002 to +.0007 and a .2 was -.0007 to -.0015 OA-mean (CLAIMED `skeptic_ladder/box_ladder_transfer.log`, FT rows).
- Both routes say the same: for R6 alone at RSNA coupling, a .1 is about 0 and a .2 is negative.

## 5. S6fall and the c4-style .8 R6 / .2 S6fall rank blend

**Run** (MEASURED, `logs/run_s6.log` on the box):
- S6fall scored all 2,457 studies (58 gold58 approx + 2,399 OAI) in 3,120 s: 1.27 s / study, loader-bound, gpu_s 1,513.
- 0 fallbacks. Anchors: skm_tea_joint 2,416, slab_prior 41.

### 5.1 Rebuild validation for S6fall on gold58 (MEASURED; `box_partA_s6.log`, `box_g58_bias_s6.log`)

| S6fall, gold58 (n 58) | exact (`~/s6seed/gold58_serve_gpu/nartaa_s42`) | approx (store rebuild) |
|---|---|---|
| macro-12 AUC vs gold | .9314 | .9242 |
| OA-trio mean AUC | .9115 | .9045 |
| mean per-label Spearman exact vs approx | | .950 |
| OA model rho / q / gap | .845 / .714 / .131 | .870 / .750 / .120 |
| gold58 E-OA OA-mean dAUC a .1 / a .2 | +.0041 / +.0056 | +.0015 / +.0059 |

**Paired bootstrap, approx - exact:**

| Quantity | approx - exact | 90 % interval |
|---|---|---|
| rho | +.025 | [-.006, +.060] |
| gap | -.011 | [-.058, +.028] |
| E-OA a .1 | -.0026 | [-.0031, +.0028] |
| E-OA a .2 | +.0003 | [-.0068, +.0061] |

- S6fall loses a little more per-label signal than R6: Effusion -.031, Lateral OA -.025, Lateral Meniscus -.020.
- Its OA coupling barely moves.

### 5.2 OAI per label (n 1,704 knees common to all models; `box_final_s6.log`)

| label | R6 | S6fall | .8 R6 + .2 S6fall | D5ref |
|---|---|---|---|---|
| ACL | .945 | .965 | .957 | .938 |
| Medial Meniscus | .944 | .945 | .946 | .934 |
| Lateral Meniscus | .953 | .949 | .954 | .933 |
| Effusion (>= 2) | .958 | .961 | .961 | .957 |
| Baker's | .917 | .897 | .919 | .898 |
| Synovitis (>= 2) | .800 | .791 | .801 | .804 |
| Medial OA P_FT | .891 | .899 | .896 | .879 |
| Lateral OA P_FT | .811 | .786 | .811 | .779 |
| PF OA P_FT | .859 | .854 | .862 | .813 |

- **S6fall swap check:** MedOA from the medial output .899, from the lateral output .651. MedMen .945 vs .563.

### 5.3 E-OA, unweighted and reweighted (P_FT unless noted; B 2000 unweighted, B 300 reweighted)

| model | OAI model rho | R0 a .1 / a .2 | R1 gap | **R1 a .05 / a .1 / a .15 / a .2** | tilt 1.0 a .1 / a .2 | R1 P_AREA a .1 / a .2 |
|---|---|---|---|---|---|---|
| R6 | .546 | +.0040 / +.0064 | +.006 | -.0006 / **-.0017** / -.0034 / **-.0057** | -.0011 / -.0045 | -.0014 / -.0050 |
| S6fall | .531 | +.0029 / +.0036 | -.003 | -.0010 / **-.0027** / -.0049 / **-.0079** | -.0020 / -.0066 | -.0011 / -.0048 |
| S6fall, all 2,399 knees | .542 | +.0028 / +.0035 | +.007 | -.0009 / -.0024 / -.0045 / -.0073 | -.0021 / -.0066 | -.0013 / -.0053 |
| **.8 R6 + .2 S6fall** | .551 | +.0041 / +.0065 | +.013 | -.0005 / **-.0016** / -.0033 / **-.0055** | -.0010 / -.0042 | -.0009 / -.0042 |
| R6 + D5 (.5 / .5, sensitivity) | .672 | +.0064 / +.0116 | +.065 | +.0008 / +.0011 / +.0010 / +.0004 | +.0016 / +.0014 | +.0015 / +.0013 |
| D5ref (fleet, OAI-free) | .754 | +.0064 / +.0121 | +.080 | +.0010 / +.0017 / +.0021 / +.0021 | +.0022 / +.0030 | +.0019 / +.0026 |

**Reading the R1 cells:**
- Sd at R1 is about .0005 at a .1 and .0011 at a .2.
- Every R1 cell of R6, S6fall and the .8 / .2 blend at a >= .1 has P(>0) <= .04.
- The blend at a .05 has P .05.

### 5.4 The gold58 / OAI disagreement (read this before using sec 4-5)

- **On RSNA gold58 (exact serve, RSNA expert labels, n 58),** both OAI-free c4 members gain from E-OA:
  - R6: +.0035 / +.0067 at a .1 / .2.
  - S6fall: +.0041 / +.0056.
  - Sd is about .002 / .004.
  - Their gaps are .092 / .131.
- **On OAI under R1,** the same models have gaps of about 0 and lose.
- **R6:** the bias-adjusted OAI gap (~.056) is within gold58 noise of .092 (gap sd ~.03-.04).
- **S6fall:** gold58 .131 against OAI ~.01 is a larger gap than noise explains.
- **Both models' OA outputs are much less coupled on OAI:** rho .53-.55, against .78-.85 on gold58. The fleet model D5, trained on the same 320 px store format, keeps .75 on OAI.
- GUESS: the OAI protocol (DESS water-excitation reformats as the coronal / axial input, no coronal non-FS) and the rebuild weaken the shared OA component that the competition-pipeline models build from RSNA PD-FS TSE studies.
- So **OAI is a pessimistic ruler for R6 / S6fall's E-OA, and gold58 is an optimistic, very noisy one.** The truth for the hidden set probably lies between. For the blend at a .1, that is roughly -.0016 (OAI R1) to +.0035 (gold58) OA-mean.

## 6. What it means for eoa10 / eoa20 (MODEL / GUESS)

**c4's make-up and gaps:**
- c4 = .8 rank(V224-FLIP) + .2 rank(S6fall). V224-FLIP mixes Raptor (r384 / r224, OAI-trained) with R6; the slug says r6a25, so R6 is about .25 of it (GUESS from the kernel name). Weights: Raptor ~.6, R6 ~.2, S6fall .2.
- c4's gold58 gap is .10 (CLAIMED mechanics.md), close to the exact gold58 gaps of R6 (.092) and S6fall (.131).

**What OAI says, at RSNA expert coupling (R1, P_FT, OA-mean):**
- **The two OAI-free members:**
  - Raw: blend -.0016 / -.0055 at a .1 / .2.
  - With R6's rebuild bias added back (+.0014 / +.0021): about -.0002 / -.0034, i.e. -.00005 / -.0009 macro.
- **Two-family OAI-free proxy (R6 + D5):** +.0011 / +.0004, i.e. +.0003 / +.0001 macro. This matches the skeptic's blend3 (+.0010 / +.0003).
- **The domain caveat (5.4)** says the R6 / S6fall OAI numbers lean pessimistic. gold58 says these models gain, but n 58 cannot size it (sd ~.002-.004 OA-mean).

**Combined** (MODEL: weight the member evidence ~.4 and the two-family proxy ~.6, then shade up a little for the domain caveat):

| Entry | Macro | P(>0) (GUESS) |
|---|---|---|
| eoa10 | about +.0001 (range -.0001 to +.0004) | about .6 |
| eoa20 | about -.0001 (range -.0009 to +.0003) | about .4 |

**eoa20 vs eoa10:** under R1, a .2 < a .1 for R6 (-.0040), S6fall (-.0052), the .8 / .2 blend (-.0039) and R6 + D5 (-.0007). Only the fleet model D5 has a .2 slightly above a .1 (+.0004). P(eoa20 > eoa10 on hidden) is about .35-.4 (GUESS).

**Not a reason to act:**
- No new build, no new slot. The reads stay under rule R1.
- If eoa20 alone moves the API best-id, the move is within public noise (sd ~.0004), and this note puts its hidden expectation below eoa10's. That is a tie-break input for Eesh.

**Cheapest way to shrink the main uncertainty** (not done; the GPU budget was used):
- Serve R6 exact and rebuilt on ~200 more RSNA train studies. That tightens the rebuild bias (now n 58).
- Note that R6 is in-sample on train, so only the exact-vs-approx difference is usable, not the AUCs.

## 7. Timings (MEASURED)

**CPU:**
- Protocol header scan (10 tarballs): ~1 min.
- Rebuild: gold58 25 s; OAI 2,399 knees 828 s (side scores ~3 min, then 16 processes). Disk: 45 GB OAI + 0.9 GB gold58, under ~/r5_oai.
- Analysis, all CPU: a few minutes per run.

**GPU wait:** 02:15-02:41 behind the sibling task's job.

**R6 (V100, fp16, 4 folds x flip TTA, compile):**
- Worker 9,908 s (2.75 h deadline), 02:41-05:26.
- First study (compile) 256 s.
- gold58 approx ~5.0 s / study.
- OAI 5.47 s / study steady (GPU-bound: gpu_s 9,575 of 9,888).
- Result: 1,762 scored (58 + 1,704 OAI), 695 OAI not reached, 0 failures. Anchors: skm_tea_joint 1,729, slab_prior 31.
- The exact gold58 serve (run_b) took ~9.8 s / study, because RSNA studies have more slices.

**S6fall (V100, fp16, 1 model, no TTA):**
- 3,120 s, 05:27-06:19.
- 1.27 s / study, loader-bound (decode 3,490 s across 2 loader workers, gpu_s 1,513).
- All 2,457 studies scored.

**Total GPU used:** about 3.6 h (R6 2.75 h + S6fall 0.87 h).

## 8. Domain caveat (read before using any number here)

- **OAI is not RSNA:**
  - OAI: one vendor (Siemens Trio 3T), one fixed research protocol, 4 clinical sites, mostly older adults at OA risk.
  - RSNA: multi-site and multi-vendor clinical MRI.
  - Not "single-site", but far more homogeneous than RSNA.
- **Series mismatch with the RSNA protocols the models were trained on:**
  - SAG IW TSE FS is close to the RSNA Sag PD FS (good match).
  - Coronal and axial come from 3D DESS water-excitation reformats (gradient echo, T2*/T1 mixed contrast, bright cartilage, different BML / meniscus conspicuity). They are labelled FS + fluid-sensitive (1 / 1) the way `build_oai_store.py` mapped them, but they are not PD FS TSE.
  - There is no sagittal non-FS series, and the coronal non-FS series (COR_IW_TSE) exists in OAI but not in the store.
- **On top of that, the store rebuild (sec 2.2)** lowers R6's OA coupling (sec 3).
- **Labels:** MOAKS semi-quantitative reads mapped to RSNA labels by our own rules (`build_oai_labels.py`, `oai_eoa_check.py`), not the RSNA rubric. Expert image reads, but a different reader team, definitions and population. The OA joint distribution is reweighted to gold74 only in sec 4.3.
- **R6 and S6fall are OAI-free** (trained on competition data only), so these are genuinely out-of-sample. Raptor (c4's largest member) is OAI-trained and cannot be measured this way.

## 9. Governance log

- **One slip (10-20 ~01:50):** while learning the file formats I ran `head -3` / `head -5` over ssh on `converted_manifest.csv`, `oai_series_meta.csv`, `oai_labels_T123.csv` and `oai_label_sources.csv`. That printed 2-4 OAI rows each (knee ids, tarball paths, series UIDs, per-knee grades) into this session's transcript.
  - Nothing was written to any local file.
  - Every later command printed headers, counts or aggregates only.
  - The local logs in this folder hold aggregates only.
- **GPU:** two of my processes (a waiting R6 controller and a chain whose R6 step had failed an assert) were stopped by me before they reached the GPU. The failed assert came from a `__pycache__` my own CPU check had written into `~/r5_oai/r6src`.
  - I deleted that `__pycache__`.
  - I also deleted a `side_common` `__pycache__` my rebuild had written into `~/r6g58/src` at 01:56, so that directory is back to its prior state.
  - No other process was killed.
- **Files outside ~/r5_oai:** nothing new was written outside ~/r5_oai. Reads only: ~/oai_store, ~/oai_build, ~/fleet_store, ~/r6g58, ~/r7port, ~/s6seed, ~/kmimic, ~/rsna_train.

## 10. Files

**This folder (`session_2026-10-20/r5/`):**
- `r6_oai.md`: this note.
- **Box scripts (copies), `box/`:**
  - `proto_stats.py`
  - `select_oai.py`
  - `build_synth.py`
  - `merge_split.py`
  - `parse_check.py`
  - `r5_run.py` (derived from `r6g58_run_ref.py`, the original)
  - `r5_s6.sh`
  - `r5_chain.sh`
  - `r5_analyze.py`
  - `g58_bias.py`
- **Aggregate logs:**
  - `box_partA_r6.log`
  - `box_g58_bias_r6.log`
  - `box_partB_smoke300.log` (n 316)
  - `box_partB_mid1000.log` (n 1,006; its a .05 / .15 labels print as "0.1", fixed later)
  - `box_r6_final.log` (n 1,704)
  - `box_partA_s6.log`
  - `box_g58_bias_s6.log`
  - `box_final_s6.log` (n 1,704, all models)
  - `box_s6_all2399.log` (S6fall alone, n 2,399; the model is labelled "S6fall-all")

**Box (`~/r5_oai`, mode 700; OAI-derived, never copied off):**
- `data_r5/` (rebuilt OAI + gold58 approx DICOMs)
- `run_r6/worker0/predictions.csv`
- `s6_r5/out/predictions.csv`
- `ids_*.json`
- `logs/`

## 11. Skeptic (10-20 ~06:30-06:55 IST, Claude; independent re-check on the box, aggregates only)

**Verdict:** the measurement is sound; two pieces of the inference are wrong and one caveat is under-weighted. Fixing them leaves the bottom line about where sec 6 put it: eoa10 slightly positive, eoa20 about zero and more likely below eoa10. The picks rule does not change.

Method: a new script, `box/skeptic_r6oai.py` (box copy `~/r5_oai/skeptic/`), that does NOT import `r5_analyze.py`. It parses the MOAKS codes as strings, uses sklearn AUCs, and computes R1 as an exact importance-weighted estimate (weighted mid-rank pct + weighted Mann-Whitney AUC, no resampling). It adds gold74 target uncertainty and the real c4 OA structure. CPU only (nice 5), B 600. Log: `box_skeptic_B600.log`.

### 11.1 What checks out (MEASURED)

**Input contract:**
- Series per study:
  - OAI: 2,398 studies have 3 series (Ax / Cor / Sag, all written FS 1 / fluid 1); 1 has no Sag.
  - gold58 approx: 54 studies x 3 series, 4 x 2.
  - Real gold58: 4-9 series per study, including 72 Sag non-FS, 54 Cor non-FS and 18 Ax non-FS. Both rebuilds drop these, so the gold58 validation does cover the missing-series loss.
- Sequences (from `oai_series_meta.csv` joined to the rebuilt series list):
  - OAI Cor / Ax are DESS MPR (TE 4.71 ms, TR 16.3 ms).
  - Sag is IW TSE FS (TE 30 ms, TR 3,200 ms).
  - The gold58 validation does NOT cover the DESS-for-Cor/Ax-FS swap.
- Planes and mirror repair:
  - `build_meta.csv` has status ok for all 7,196 OAI series, and the plane check against `oai_series_meta` passed.
  - The mirror repair flipped 3 Ax and 9 Cor series. Every right knee is at x0 -65 mm, every left at +65 mm.
- Side and flip:
  - The R6 receipt `side_check` checked 1,736 studies and flipped 1 (threshold 2.0).
  - Flip TTA is the per-fold .5 (z + P z_mirror) in logit space (coronal width flip with views 2 <-> 3, sagittal slice reversal).
  - **The S6fall worker has no side check** (`side_check` None), so its side came only from the fabricated sagittal IPP x (+-65 mm). Its swap check (MedOA .899 vs .651, MedMen .945 vs .563) shows that rule was right.
- Fallbacks and the prefix:
  - R6 scored 1,762 studies (58 + 1,704). The 695 timed-out studies are absent from `predictions.csv`.
  - The scored OAI rows have 0 NaN and 0 duplicated OA-trio rows, so no prior-filled rows got in.
  - The 1,704 are exactly the first 1,704 of the stratified order, and their P_FT 8-cell joint matches all 2,399 eligible knees within .001 per cell.
  - Receipt anchors: skm_tea_joint 1,731, slab_prior 31. Sec 2.2 / 7 say 1,729; that is cosmetic.
  - S6fall: 2,457 scored, 0 fallbacks, anchors 2,416 / 41.

**Labels:**
- The string parse of the MOAKS "area.ft" codes equals the float parse: 0 mismatches for both proxies on 2,399 knees.
- Code set: 0.0, 1.0, 1.1, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2, 3.3, NA. FT is never above area.
- Other T123 encodings (value sets):
  - ACL is {0, .5, 1} from m_acl_tear 0 / 1 / 2. The builder reads 1 as "partial" (its own GUESS), so ACL >= .5 counts partial tears.
  - Effusion is {0, .3, .8, 1}, not grade / 3 as the `r5_analyze.py` header says. The thresholds are still right: >= .5 means grade >= 2, and >= .3 means grade >= 1.
  - Synovitis is grade / 3. Menisci are {0, .2, 1}, so >= .5 counts only 1.

**Numbers:**
- Per-label AUCs reproduce to 4 decimals with sklearn.
- The exact weighted R1 reproduces the resampled R1. P_FT at a .1 / .2:

  | Model | a .1 | a .2 | gap |
  |---|---|---|---|
  | R6 | -.0018 | -.0058 | +.005 |
  | S6fall | -.0027 | -.0079 | -.004 |
  | .8 / .2 blend | -.0016 | -.0056 | |
  | D5 | +.0017 | +.0021 | +.080 |
  | R6 + D5 | +.0011 | +.0003 | |

  rho, q and gap agree within .001.

### 11.2 What is wrong or overstated

1. **c4's make-up (sec 6) is wrong** (MEASURED from the kernel source, `session_2026-10-18/a18/kernel_v224r6a25c4j`):
   - "a25" is `A224 = 0.25`, the 224-crop model's share inside the Raptor slot. It is not R6's weight.
   - Cell 14 blends Raptor with R6 using per-label Raptor weights `W_RAPTOR_C`: Medial OA .5, Lateral OA .2, PF OA .5. Cell 18 then takes .8 V224-FLIP + .2 S6fall.
   - OA-trio weights (Medial / Lateral / PF):

     | Member | Weight |
     |---|---|
     | Raptor | .40 / .16 / .40 |
     | R6 | .40 / .64 / .40 |
     | S6fall | .20 |

   - So the OAI-free members carry .60 / .84 / .60 of c4's OA output (mean .68), not ~.4. The ".4 member / .6 R6+D5" weighting in sec 6 has no basis.
2. **The c4-shaped proxy replaces that weighting** (MEASURED, n 1,704):
   - "c4D5" puts D5 in the Raptor slot with the exact kernel weights and nested pct.
   - Under R1:

     | Proxy | a .05 | a .1 | a .15 | a .2 |
     |---|---|---|---|---|
     | P_FT | +.0005 | +.0003 | -.0002 | -.0014 |
     | P_AREA | +.0007 | +.0010 | +.0007 | .0000 |

   - R0 (unweighted) a .1: +.0060. R1 gap +.053, which sits between R6 + D5 (+.065) and the .8 / .2 blend (+.012).
3. **The "P(>0) <= .04" claims count knee sampling only.**
   - The R1 target is the gold74 joint (n 71). Drawn from Dirichlet(counts + .5), its phi is .32 (sd .073, 90 % [.20, .44]).
   - Adding that uncertainty raises the sd at a .1 three- to four-fold, from about .0004 to .0012-.0016.
   - P(>0), knee + target:

     | Model | a .1 | a .2 |
     |---|---|---|
     | R6 | .14 | .04 |
     | S6fall | .03 | .01 |
     | .8 / .2 blend | .15 | .03 |
     | R6 + D5 | .79 | .52 |
     | D5 | .92 | .78 |
     | **c4D5** | **.58** | **.34** |

   - The a .2 losses of the members survive. The members' a .1 losses are not settled.
4. **"At RSNA expert coupling" means label coupling only, and R6 / S6fall's own OA coupling collapses on OAI.**
   - R1 moves the labels to phi .32 but leaves the model coupling in the OAI regime:

     | Model | R1 rho on OAI | rho on gold58 (exact serve) |
     |---|---|---|
     | R6 | .597 | .778 |
     | S6fall | .581 | .845 |

   - The drop is specific to these two models (MEASURED):

     | Model | RSNA rho | OAI rho | Drop | Part explained by the rebuild |
     |---|---|---|---|---|
     | D5 | .826 (5-fold OOF on RSNA train, n 4,333; folds .821-.835) | .754 | -.07 | not rebuilt |
     | R6 | .778 | .546 | -.23 | -.04 |
     | S6fall | .845 | .531 | -.31 | +.025 |

   - D5 scored the same three OAI series through the fleet loader from the raw tarballs (8.7 s per study, `teacher/run_shard0.log`), so it also saw the DESS views, but it did not go through the DICOM rebuild.
   - mechanics.md finds E-OA "is all coupling". So the OAI R6 / S6fall cells measure these members in a coupling regime they do not have on RSNA.
   - Sec 5.4 flags this, but sec 6 still uses the member numbers with only the rebuild correction. That under-corrects, in the pessimistic direction. The direction is clear; the size is unknown (GUESS).
   - What causes it cannot be split here: it may be the DESS protocol, the OAI population, or the OAI-specific fabricated geometry (+-65 mm centres, lost 10-20 mm cross-series offsets that move the anchored local crops). The raw V00 DICOMs that could split it are gone.
5. **The S6fall gold58 rebuild bias (sec 5.1) is unmeasured, not small:**
   - Its a .1 point (-.0026) lies at about the 6th percentile of its own 90 % interval [-.0031, +.0028].
   - The q bootstrap sd of .46 comes from near-zero denominators in resamples.
   - At n 58 the decorrelated scores sit on a coarse grid, so float rounding decides ties. My implementation gives S6fall exact gold58 a .1 = +.0037 where the log has +.0041; R6 matches exactly (+.0035).
   - So gold58 E-OA numbers carry about +-.0004 of tie noise on top of their sampling sd.

### 11.3 What survives

- R6 and S6fall read OAI well, medial and lateral are the right way round, and every R1 number reproduces with independent code.
- **Direction:** on OAI, every OAI-free combination that contains R6 or S6fall prefers a .1 to a .2. P(a .2 > a .1), knee + target:
  - c4D5 .16;
  - R6 + D5 .33;
  - R6, S6fall and the blend .00-.01;
  - only D5 alone is above half, at .57.
- On gold58, R6 and S6fall point the other way (a .2 > a .1), but n is only 58.

### 11.4 Re-estimate for the in-flight entries (MODEL / GUESS; replaces the sec 6 arithmetic, not the picks rule)

**Arithmetic, starting from c4D5 raw (R1, P_FT):**

| Step | eoa10 | eoa20 |
|---|---|---|
| c4D5 raw | +.0003 OA-mean = +.0001 macro | -.0014 OA-mean = -.0003 macro |
| + about half of R6's gold58 rebuild bias (+.0007 / +.0010 OA-mean) | about +.00025 macro | about -.0001 macro |

- Half, because R6 + S6fall are .68 of the weight and blending dilutes a coupling shift. S6fall's own bias is unknown.
- Then shade up a little, without a number, for the R6 / S6fall coupling collapse (11.2.4).

**Re-estimate:**

| Entry | Macro | P(>0) |
|---|---|---|
| eoa10 | about +.0002 (range -.0003 to +.0006) | about .6-.65 |
| eoa20 | about -.0001 to 0 (range -.0012 to +.0008) | about .4-.45 |

- P(eoa20 > eoa10 on hidden) is about .3-.45. OAI c4D5 gives .16; the mechanics gap map on c4's gold58 gap (.10, CLAIMED) gives .58-.61.
- Against sec 6: eoa10 is a touch higher (+.0002 against +.0001) and eoa20 is the same. The note reached about the right answer through a wrong c4 decomposition.
- **Picks rule unchanged:** A = API best-id under R1, B = R6 57248084. An eoa20-only best-id move is within public noise (sd ~.0004), and its hidden expectation is slightly below eoa10's. That is a tie-break input for Eesh, not a rule change.

### 11.5 Governance (skeptic)

- **My slip:** while reading `run_r6/worker0/receipt.json` I printed its `fallback` field, cut to 600 characters. It is a dict keyed by OAI knee id, so about 12 OAI knee ids went into this session's transcript, each with only the text "not reached before the deadline" (no grades, no images).
  - Nothing was written to a local file.
  - Every later box command went through an id-redacting `sed` on the box, and `box_skeptic_B600.log` has 0 OAI ids.
  - I also printed the first 20 keys of `dump.json`. These are RSNA gold58 study UIDs (competition data, not OAI).
- **New box files:** only `~/r5_oai/skeptic/` (mode 700), holding `skeptic_r6oai.py` and `sk_B600.raw` (aggregates only).
- **GPU:** not used. CPU only, nice 5. No process killed.
- No submit, push, commit, dataset, download, scheduled-task edit or protected-file edit.

**Files (this folder):**
- `box/skeptic_r6oai.py`
- `box_skeptic_B600.log`

