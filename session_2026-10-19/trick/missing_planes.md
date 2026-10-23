# Missing planes / missing fluid-sensitive series: does a per-study member fallback gain? (10-19, Claude subagent)

RESEARCH + MEASUREMENT ONLY. Nothing built, pushed, committed, submitted; no dataset; no GPU / box job; no ssh used.
Local CPU only (py -3.12, each script < 1 min). No em dashes. Tags: MEASURED (computed here, file named), CLAIMED (from a
code comment or prior note), MODEL (inference from measured numbers), GUESS.
Work files: `session_2026-10-19/trick/mp_work/` (scripts m1..m4, csv outputs, c4 cells dump `c4_cells.py`, decoded R6
sources `r6src/`, decoded S6F driver `s6f_driver_decoded.py`, decoded fastread `fastread_decoded.py`).

## Verdict: WEAK (in-distribution part DEAD; residual value is insurance whose EV hinges on an unmeasured hidden property)

- Train is CURATED: 0 of 4,407 studies lacks a plane, 0 lacks a fluid-sensitive series, min 3 series (MEASURED). With
  that inclusion rule almost certainly applied to test too (all 3 visible test studies are complete, MEASURED), the
  "member silently degrades on a study with a missing plane" scenario has a low prior (GUESS P 5-10 %, only via the 3
  test-only sites, and only if the host's filter differed for them).
- The in-distribution per-pattern effects are tiny or noise: (a) the only per-study member asymmetry that exists today,
  R6's Effusion fluid-stacker "miss" path on no-FS-sagittal studies (German protocol, 5.8 % of train, 1 of 3 visible
  test studies), is worth +0.000001 to +0.000047 macro through a c4-shaped blend on train (4 report-label sets) and 0 on
  gold58; (b) slot-model vs set-model relative AUC per series pattern flips sign between label sets (+-0.008), so no
  pattern-gated reweighting. MEASURED.
- A per-study fallback guard is an identity on complete studies, so its value is P(hidden incomplete) x conditional gain
  (+0.0003 to +0.005 macro at 0.5-5 % affected studies, MEASURED sim). EV about +1e-4 (MODEL/GUESS), below the +2e-4 bar.
- Recommended next step if Eesh wants to settle it: a 1-bit CPU-only probe submission (sec 6) that reads only the hidden
  test_series.csv; ~1 slot, < 1 h turnaround, no GPU. If it says "all complete", the hypothesis is dead for good.

## 1. Series composition (MEASURED, mp_work/m1_series.py on _data/train_series.csv, test_series.csv, _reports_lang.csv)

Train: 4,407 studies, 24,371 series. Plane values: only Sagittal 9,864 / Coronal 8,609 / Axial 5,898; no NaN anywhere;
Fluid_Sensitive == Fat_Suppression in 24,371 / 24,371 rows (the "fluid" flag IS the fat-sat flag).

| property (share of train studies) | value |
|---|---|
| lacks Sagittal / Coronal / Axial / any plane | 0 / 0 / 0 / 0 |
| no fluid-sensitive series at all | 0 |
| no FS sagittal | 5.83 % (257) |
| no FS coronal | 3.61 % (159) |
| no FS axial | 0 |
| no non-FS sagittal | 3.20 % (141) |
| <= 2 series / <= 3 series | 0 / 0.02 % (1 study) |
| n series 4 / 5 / 6 / 7+ | 15.3 % / 52.2 % / 15.8 % / 16.6 % |
| >= 2 FS sagittal (Raptor slot-0 alternative) | 11.6 % |
| >= 3 sagittal | 21.8 % |
| any Raptor slot has an alternative candidate | 44.9 % |

By report language (site proxy, langs with >= 40 studies):
- no FS sagittal is a German-site signature: de 93.5 % (245 / 262), el 1.9 %, en 0.3 %, es 0.1 %, others 0.
- no FS coronal: en 8.9 % (154 / 1,735), nl 0.7 %, el 0.6 %, tr 0.4 %.
- no non-FS sagittal: el 29.9 %, tr 5.3 %, others <= 0.9 %.
- >= 6 series: hr 74 %, bg 46 %, en 36 %, tr 32 %, de 21 %, es 22 %, el 8 %, nl 4 %, fr 1 %.
Top patterns (FS/non-FS per plane): S1/1 C1/1 A1/0 39.6 %, S1/1 C1/0 A1/0 10.8 %, S1/1 C1/2 A1/0 6.6 %, S2/1 C1/0 A1/0
5.2 %, S0/2 C1/0 A1/1 4.6 % (German).
Visible test (3): `...1004703...` = S0/2 C1/0 A1/1 (German pattern, no FS sagittal), `...1006286...` = S2/1 C1/0 A1/0,
`...1006751...` = S1/1 C1/1 A1/0. All complete.
Gold74: 3 no-FS-sag (2 in gold58), 3 no-FS-cor (2), 1 no-nFS-sag (1), 12 dup-FS-sag (8).

Both CSVs are sorted lexicographically by (Study, Series) and study-contiguous (MEASURED), so Raptor's "first candidate"
pick is the same rule at train-corpus build and at test (pick = random w.r.t. sequence type, but consistent).

## 2. Code audit: what each c4 member does with a missing / odd series

All members choose series ONLY from the host CSV flags (Anatomical_Plane, Fluid_Sensitive, Fat_Suppression); DICOM tags
are used for slice order (IPP on IOP normal), spacing and side; no SeriesDescription anywhere. MEASURED (code read).

Raptor r384 and r224 (cells 4 / 11; the actual reader is `_fastread.build_study_v2`, same picker, MEASURED by decoding
`_FASTREAD_B64`):
- 5 slots, 96 slices: (Sag, FS 1, 26), (Sag, FS 0, 22), (Cor, FS 1, 18), (Cor, FS 0, 12), (Axial, any, 18).
- `_pick_series_for_slot`: first unused series of the plane with the preferred flag; else first unused of the plane;
  else the slot stays ZERO. Missing series dir / no files: slot zero. A slice that fails to decode: zero slice.
- Zero-slot shares in train (what the model was trained on): slot 1 2.7 %, slot 3 25.5 %, slots 0 / 2 / 4 NEVER (0 %).
  So a hidden study with no sagittal, no coronal or no axial would be an input Raptor never saw. MEASURED.
- Non-preferred fill in train: slot 0 gets a non-FS sagittal in 5.8 % (German), slot 2 a non-FS coronal in 3.6 %.
- Silent path: `_pre_one` catches a prep exception and returns an ALL-ZERO volume; the GPU consumer still produces a
  (constant) prediction, so the row is NOT 0.5 and stage 3's `rap_fail` (all-0.5 rows only) does not see it. Only GPU
  exceptions leave 0.5; > 3 % of those stops the notebook (cell 14).
R6 (cell 7-9: probe_dual -> compact_infer v4c -> px_cache.StudySlices -> mainline native_cache.prepare):
- Uses EVERY series of the study (after exact-duplicate removal), each with views v4 (full + locals), meta per window
  carries plane / fluid / fat flags. Non-cardinal plane rows are skipped; NaN flags become -1 (SLOTS has (plane, -1) keys,
  no crash). A series with no decodable slice is dropped; a study with no series raises -> worker fallback -> column-mean
  row -> stage 3 detects it from predictions.csv and uses Raptor's rank (cell 14).
- Fluid stacker (Effusion ONLY; adopted "by user override", its pre-registered rule said DROP, MEASURED from
  fluid_stacker.json): needs a FS sagittal series; otherwise the miss path z_out = b0 + b_miss + b_z * z (no volume).
  Deployed R6 coefficients b0 -3.772, b_z .330, b_lv .640, b_miss 2.298; 255 / 4,344 pool studies missing.
- State stacker: adopted for Baker's only; it has no volume / miss features.
S6fall (cells 16-18, driver decoded): the same compact_infer / px_cache path (320 / crop 288, views v4), no stacker.
Unscored studies take the V224 rank in the S6 slot (cell 18 `rs[~ok] = rb[~ok]`); coverage gate >= 98 %.
Hidden-run bounds implied by c4 scoring cleanly (.958, no canary drop): Raptor GPU failures <= 3 %, S6fall coverage
>= 98 %, R6 unscored <= 50 %. Nothing bounds Raptor's silent zero-volume / zero-slot rows. MODEL.

## 3. R7 OOF (R6 family, 4,406 train studies) by series pattern (MEASURED, mp_work/m3_oof.py, m3e_pair.py)

Per-group AUC vs report labels (v2 YES vs NO) is confounded by site (report style, prevalence): no_fs_sag .863 macro vs
complete .888, but within German: all-de .864 vs de & no-FS-sag .862; Effusion no_fs_sag .883 vs complete .880 (no image
-model degradation). no_fs_cor .835 vs en-all .897 (en & no-FS-cor .834; OA labels .77 vs .91): one English site with
weaker OA agreement, could be report style or protocol (not separable).

Paired slot-model vs set-model on the SAME studies: TTR s1044 5-fold OOF (our Raptor-recipe slot model, CLAIMED from the
frac75 runbooks) vs R7 OOF, 4,333 studies, mean over 12 labels:

| group (n) | labels | TTR | R7 | 0.5/0.5 blend | R7 - TTR |
|---|---|---|---|---|---|
| complete (3,783) | jev_bin | .9035 | .9064 | .9153 | +.0029 |
| no FS sag (254) | jev_bin | .8965 | .8955 | .9075 | -.0010 |
| no FS cor (156) | jev_bin | .8789 | .8854 | .8927 | +.0065 |
| dup FS sag (499) | jev_bin | .9053 | .9098 | .9175 | +.0045 |
| complete | v4blend | .8858 | .8862 | .8980 | +.0004 |
| no FS sag | v4blend | .8797 | .8828 | .8931 | +.0031 |
| no FS cor | v4blend | .8455 | .8373 | .8562 | -.0082 |
| dup FS sag | v4blend | .8964 | .8882 | .9046 | -.0083 |

The sign flips with the label set and the blend beats both singles by >= +.007 in every group: no pattern where one
member collapses, so dropping or down-weighting a member per pattern would lose. MEASURED.

Effusion miss path (mp_work/m3c_eff.py, m3d_eff_labels.py, m3f_eff_blend.py):
- Relative to imputing the volume at its median, the miss path adds +0.77 logit (R7 stacker) / +1.15 (R6 deployed
  stacker) to no-FS-sag studies; their mean Effusion pct rank goes .620 (model) -> .676 (stacker). Report Effusion YES
  rate: no-FS-sag .708 vs others .599.
- Replacing the miss path by E[log volume | z] imputation, R7 member alone: +.0019 / +.0013 / +.0007 / +.0013 Effusion AUC
  (v2 yes/no, v4blend, jev_bin, gpt56sol; 90 % CIs [+.0009,+.0029], [+.0003,+.0021], [.0000,+.0014], [+.0004,+.0021]).
- Through a c4-shaped blend (.4 TTR + .4 R7 + .2 R7 pre-stacker as S6 proxy): +.00056 / +.00033 / +.00002 / +.00031
  Effusion = +0.000047 / +0.000027 / +0.000001 / +0.000026 macro. CIs straddle 0 for 3 of 4.
- Gold58 (2 affected studies, expert Effusion 1 and 0): R6 alone .9453 -> .9528, c4-shaped blend (r384 .75 / r224 .25,
  R6, R6 pre-stacker as S6 proxy) .9789 -> .9789 (no change).

## 4. Value of a per-study fallback vs share X of degraded hidden studies (MEASURED sim, mp_work/m4_sim.py)

Members TTR (slot) + R7 (set), 0.5/0.5 pct blend, jev_hybrid_binary labels, clean macro .9147. TTR degraded on a random
X of studies; fallback = those rows take R7's rank. 6 reps each.

| X | degradation | loss vs clean | fallback gain |
|---|---|---|---|
| 0.5 % | uninformative / half / constant | .00048 / .00018 / .00040 | +.00041 / +.00017 / +.00035 |
| 1 % | " | .00126 / .00037 / .00066 | +.00119 / +.00028 / +.00055 |
| 2 % | " | .00224 / .00061 / .00161 | +.00193 / +.00047 / +.00141 |
| 5 % | " | .00588 / .00153 / .00386 | +.00544 / +.00102 / +.00332 |
| 10 % | " | .01226 / .00343 / .00858 | +.01138 / +.00239 / +.00753 |

The guard recovers ~70-95 % of the loss. EV = P(hidden has such studies) x gain. With train 0 / 4,407 and visible 0 / 3
incomplete, GUESS P 5-10 %, conditional X 1-5 %, severity half-to-full (zero slots 0 / 2 / 4 never seen in training):
EV about +0.5e-4 to +2e-4 (MODEL/GUESS). For the Effusion miss path the hidden share is likely ~6 % (German protocol;
visible test 1 / 3) but the measured gain is <= +0.5e-4 macro.

## 5. The buildable trick (only if the probe in sec 6 says "incomplete studies exist")

T1 Raptor guard (r384 and r224 processes, identical edit): after prep, if slot 0, 2 or 4 has zero filled slices, or the
prep raised (today an all-zero volume), write 0.5 for that study (do not run the GPU), so stage 3 gives it R6's rank. The
3 % stop in cell 14 must then count only GPU exceptions (keep a separate counter for intentional guards). Identity on
every complete study; runtime ~0. T2 optional: Effusion miss path recomputed in stage 3 from the worker's
fluid_stacker.csv (median-volume imputation); worth <= +0.5e-4, skip. GPU measurement to size T1 before any submission:
on the box V100, gold74 DICOMs, Raptor r384 + r224 + R6 run 4 times with the series CSV ablated (drop all Sagittal / all
Coronal / all Axial rows / all FS rows), macro AUC per member and c4 blend with vs without T1. Raptor ~74 x 4 x 2 s
~10 min; R6 ~74 x 4 x 20 s ~1.6 h (MODEL). Not needed unless the probe fires.

## 6. Cheap hidden-data probe (recommendation only, not done)

CPU-only kernel, competition data attached, no model: read test.csv + test_series.csv (+ os.listdir of each series dir).
Hidden rerun outcome encodes the answer: all studies have Sag + Cor + Ax, >= 1 FS series, only standard plane values,
non-empty series dirs -> write all-0.5 submission (LB exactly .500); some study lacks a plane -> raise (status
"Notebook Threw Exception"); planes complete but some lack an FS series or odd values -> write a csv with a missing column
(scoring error). Public commit (3 complete studies) writes the .5 file. Cost: 1 slot, minutes of runtime, < 1 h to read.
