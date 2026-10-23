# OAI build summary (box ~/oai_build), 2026-10-08 23:10 to 10-09 01:15 IST

CPU only (nice 19, CUDA hidden, 4 conversion workers + main = 5 processes; the load stayed 4-17, well below 30). Nothing
touched the GPU, ~/queue (one read-only `queue_runner.sh --check` on a separate file), ~/fleet_store, existing
~/fleet_runs runs or running processes. No OAI pixels left the box. The orientation check printed only aggregate
32x32 mean images and correlations. The PNG montages are in ~/oai_build/check/ and were never copied off the box.
**No training launched, queue not edited.**

## Counts
- **2,680 / 2,680** V00 knees converted to `~/oai_store/oai_<subject>_<R|L>.npz` (mode 700). That is 10,771 series,
  **0 failures**, 7.13 GiB. Views: v0 Axial FS + v2 Coronal FS + v3 Sagittal FS in 2,679 knees. 1 knee has no SAG_IW_TSE,
  so it has v0+v2 only. v1 (Sagittal non-FS) is absent for every OAI knee, as the research planned.
- Labels: 2,399 knees have a MOAKS read and so a T1 row (`oai_labels.csv`). 2,581 have T2 / T123 rows; the extra 182
  have only a WORMS popliteal-cyst read. 99 knees have no usable read and are dropped.
- Union: `~/union_store` = 4,407 RSNA + 2,399 OAI symlinks. Fold-0 trainer view: train 5,865 (3,466 RSNA + 2,399 OAI),
  val 867 (unchanged).
- Rate: about 88 knees/min unconstrained (about 2.7 s per knee per process). It ran download-bound at 37 knees/min and
  finished at 00:32, together with the download.
- **Disk:** raw tarballs pruned per Eesh's rule: 81.35 GiB in 10,771 files deleted after each knee's npz was written,
  re-loaded, decoded and sha256-recorded. `~/oai_raw` is now 18 MB of empty dirs. `/` used went from 1,685,090 MB
  (23:21, raw was 15.7 GB then) to 1,678,424 MB now. The total OAI footprint is 7.2 GiB instead of about 88.5 GiB.
  `converted_manifest.csv` holds kid, series, image03 paths and npz sha256, and `raw_deletions.csv` lists every deleted
  file, so any knee can be fetched again with `downloadcmd -t`. Caveat: rerunning the original `get_oai.sh` regex
  download would fetch everything again, because the files are gone.

## Series / DICOM findings
- Header-verified mapping, with **0 mismatches** over 10,771 series:
  - SAG_IW_TSE: Sagittal, ScanOptions SAT2|FS, so FS=1. Goes to v3 with contrast class 1 (TE 30 / TR 3200).
  - COR_IW_TSE: Coronal, no FS, so FS=0. CorNonFS has no slot, so it is unused (downloaded for nothing).
  - AX_MPR: Axial, DESS ScanOptions WE (water excitation, counted as fat-sat). Goes to v0 with contrast class 0
    (TE 4.7 / TR 16.3).
  - COR_MPR: Coronal, WE. Goes to v2 with contrast class 0.
  - All of these are Siemens 2.89 T, MONOCHROME2, single-frame, files without an extension.
  - Table in `~/oai_build/oai_series_meta.csv`: train_series.csv columns plus base, slot, header fields and drop counts.
- **Trap found and fixed:** each AX_MPR / COR_MPR tarball holds 1 stray reference image in another plane, for example
  75 axial images plus 1 sagittal. `fleet_common.order_slices` takes the normal from the first file and
  `_plane_from_header` reads the middle file, so this image would have mis-ordered or polluted the stack. The builder
  drops files that are off-plane or have a different shape before calling `build_study`. 5,365 files were dropped, 1 per
  MPR series (4 series had none). `fleet_common.py` itself is unmodified: md5 658ab810, pixel_sha 6f49c74fc48f3345
  asserted, jpeg_fp 34819a62 matches the trainer.
- Knees with 2 series of the same type (about 50) follow the C1 rule: lowest SeriesInstanceUID first, the second one
  unused.
- Format matches the RSNA store: uncompressed npz with the same keys, uint8 320x320 JPEG stacks of 32 slices per view,
  int64 ctr/offs. npz size is about 2.7 MB versus about 2.5 MB for RSNA.
- Orientation, OAI vs 200 RSNA studies, correlation of the mean central slice under 8 flips/rotations:
  - Identity wins for Sagittal (.88 vs flipLR -.22) and Coronal (.88; flipUD .74; transpose .02).
  - Axial: identity .89 ties with flipLR .89, as expected with mixed L/R knees. flipUD .67.
  - So the in-plane orientation matches RSNA.
- Domain notes:
  - The axial slice-axis intensity profile does not match RSNA (r -.40). The DESS MPR covers about 115-125 mm of the
    knee at 1.5 mm, a different extent and contrast. The direction is the same physical convention (same `order_slices`).
  - v0/v2 carry contrast token 0 (DESS), not PD-FS.

## Label map (rules and per-label prevalence in ~/oai_build/oai_labels_report.txt; code in build_oai_labels.py)
- Reads are merged per field across the V00 reads, ordered by number of fields read and then readprj. X-ray JSN/KL come
  from V00 readprj 15 first (2,671 knees have JSN).
- MOAKS cartilage: area grade = integer part, full-thickness grade = decimal.
- OA compartment target = max(area grade, osteophyte grade)/3, set to 1 on any full-thickness loss. For the TF
  compartments it is averaged with JSN/3.
- Synovitis = Hoffa / 3. Effusion maps 0/1/2/3 to 0/.3/.8/1.
- Baker's = popcys (MOAKS, else WORMS).
- Menisci: tear codes 2-8 or a root tear give 1, signal only gives .2.
- ACL maps 0/1/2 to 0/.5/1 (code meaning GUESSED from prevalence); a repaired ACL is NaN.
- MCL, Contusion and Fracture are always NaN.

| label | tier | OAI n | OAI mean | OAI >=.5 | RSNA mean | RSNA >.5 |
|---|---|---|---|---|---|---|
| PF OA | T1 | 2399 | .693 | .769 | .468 | .453 |
| Lateral OA | T1 | 2399 | .242 | .247 | .301 | .258 |
| Synovitis | T1 | 1659 | .232 | .107 | .349 | .273 |
| Medial OA | T2 | 2399 | .350 | .386 | .378 | .356 |
| Effusion | T2 | 1660 | .280 | .197 | .526 | .562 |
| Baker's | T2 | 835 | .371 | .371 | .319 | .250 |
| Medial Meniscus | T3 | 2399 | .489 | .411 | .457 | .437 |
| Lateral Meniscus | T3 | 2399 | .240 | .206 | .226 | .179 |
| ACL | T3 | 1654 | .062 | .089 | .216 | .166 |

Label problems:
- **PF OA is skewed:** 77 % of OAI knees are >= .5, against 45 % in RSNA. MOAKS PF cartilage damage is near-universal in
  this OA cohort, and 44 % of knees get the full-thickness 1.0.
- Synovitis is covered for only 1,659 knees.
- The ACL code meaning is unverified (T3 only).
- `oai_labels.csv` (the probe file) is **T1 only**, following RESEARCH_TO_TOP L6 and verify_D (one question per probe).
  T2 and T123 files are ready for a later gated step.

## Probe (pre-registered in ~/oai_build/PROBE_PREREG.md, md5 b57c10a1, written 01:15 before any training)
- **Schedule deviation from the brief:** the control `f0_s1043_ttr_frac75` was really trained **17 epochs, save 15-17**
  on Kaggle T4x2 (its ckpt args and the Kaggle receipt). It was NOT trained with the 24-epoch argv in argv_frac75.sh:
  queue Q1 with that argv was REFUSED because the dir existed. So the paired probe uses 17 epochs and `--grad_ckpt 1`
  (the default). The scorer refuses any run whose ckpt args differ from the control beyond `train_labels`, so a 24-epoch
  probe would be unscorable as a pair.
- Mask norm: the default `finite` (trainer unmodified, md5 3e62434a).
- Expected cost: about 5.0 h V100 (MODEL).
- CPU dry run of the exact argv on 24 studies (16 RSNA + 8 OAI) passed:
  - OAI npz pass the contract.
  - NaN cells are masked.
  - The ckpt decode_probe is RSNA-only.
  - The ckpt args differ from the control only in train_labels plus the dry overrides (amp, epochs).
- Assertions in make_union.py, all PASSED:
  - OAI ids are `oai_` and sort after every RSNA id.
  - `sorted(ids)[:8]` are RSNA.
  - The RSNA label and fold rows are byte-verbatim and bit-identical through the trainer's read path.
  - OAI rows are fold 9 only: no OAI id in folds 0-4, gold74 or v4.
  - MCL, Contusion and Fracture are NaN on every OAI row.
  - No OAI file in ~/fleet_store.
  - Every OAI npz sha256 equals the manifest.
- Registered inputs:
  - labels `~/oai_build/union/labels_frac75oai_T1.csv`, md5 bd4ca845e00b6cb9e5b7718cc9926223
  - folds `~/oai_build/union/folds5_oai.csv`, md5 84b7926a891c053d3f60be5348f64ba8

Argv (from ~/fast_build/code2):
```
fleet_train.py --model ttr --aug heavy --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs 17 --seed 1043 --save_epochs 15-17 --workers 10 --store ~/union_store --labels ~/oai_build/union/labels_frac75oai_T1.csv --folds ~/oai_build/union/folds5_oai.csv --exclude ~/rsna_train/gold74_eval_set.csv --eval_labels ~/rsna_train/llm_labels_v4_blend.csv --gold ~/rsna_train/gold74_eval_set.csv --fold 0 --out ~/fleet_runs/f0_s1043_ttr_frac75oai --grad_ckpt 1
```
**qctl line to insert (validated by `queue_runner.sh --check`: ok, 0 BAD; NOT inserted):**
```
bash ~/queue/qctl.sh top "$(cat ~/oai_build/qline_oai_probe.txt)"
```
The line text is in ~/oai_build/qline_oai_probe.txt:
`L6 OAI T1 probe f0 (TTR224 s1043 frac75+OAI-T1, 17 ep, gc1, paired with f0_s1043_ttr_frac75) | ~/fleet_runs/f0_s1043_ttr_frac75oai | - | <argv above>`

Gate (bars from L6, measured against the control's numbers, paired with f0_s1043_ttr_frac75, best raw ckpt):

| gate | rule | bar |
|---|---|---|
| G1 | OOF-vs-v4 mean over the 9 non-OAI labels (control .8896) | >= .8876 |
| G2 | OOF-vs-v4 T1 mean (control .8663) AND gold58 T1 mean (control .8192) | >= .8653 AND >= .8242 |
| G3 | gold58 12-label macro (control .9163) | >= .9113 |

- PASS: folds 1-4 (21 ep, gc0), N pick only.
- FAIL: stop L6.
- Score with `CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/rsna/bin/python ~/oai_build/score_oai_probe.py`. The
  scorer passed a selftest on the control itself, reproducing score_frac75's per-label table exactly.

## Files (box)
- ~/oai_build/:
  - Scripts: build_oai_store.py, build_oai_labels.py, make_union.py, score_oai_probe.py, argv_oai_probe.sh,
    dry_run_oai.sh, orient_check.py
  - Records: converted_manifest.csv, raw_deletions.csv, oai_series_meta.csv
  - Labels: oai_labels{,_T12,_T123}.csv, oai_label_sources.csv, oai_labels_report.txt
  - Probe: union/ (labels, folds, receipt), PROBE_PREREG.md, qline_oai_probe.txt
  - Other: check/ (box-only PNGs), dry/
- ~/oai_store (2,680 npz), ~/union_store (6,806 symlinks).

## Blockers / open
- The V100 is needed after Q4 (about 07:15 IST): insert the qctl line above. Do not rerun make_union.py before launch
  unless its md5s come out identical (it is deterministic).
- Eesh decision: whether he accepts 17 epochs as the pair, as recommended. A 24-epoch run would need a new 24-epoch
  control.
- Compliance follow-ups (Eesh):
  - Delete the NDA keyring password (research D 1.3 step 7).
  - OAI-trained weights stay in the N pick only.
