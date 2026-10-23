# Unseen-site lens (10-19, ~21:30 IST). RESEARCH + MEASUREMENT ONLY.

Nothing was built, pushed, committed, submitted or created. No GPU, training or box job ran. Box access was read-only:
`ls`/`head` of `~/novel/site` and `~/novel/verify_site` logs, and `scp` of four of OUR OWN competition-data feature
files (no OAI data) into `trick/box/`. No public code or data was pulled (`trick/pulled/` is empty). No PREREG, HANDOFF,
PROGRESS or scheduled-task edits. All scripts are in `trick/work/` (CPU, each < 1 min).

Tags: MEASURED = computed or read by us here (file named); CLAIMED = a note or third party says so (source named);
MODEL = our arithmetic on stated assumptions; GUESS = judgement.

## 1. Verdict: DEAD as a pre-deadline trick

1. **Test-time site normalisation is closed.**
   - All three c4 families already normalise each series by its own percentiles. None uses raw intensities or a
     per-volume z-score. So a scanner's gain or offset is already removed (MEASURED, code read, sec 2).
   - The only label-free offset fix that does not assume equal prevalence is to align each member's per-cluster mean to
     the members' consensus. For a linear rank blend that is exactly a no-op (MODEL, and MEASURED d = 0.00000, sec 6).
   - Aligning one member's site offsets to another's loses on both report readers and both cluster definitions:
     -.0002 to -.0031, 7 of 8 cells with P(>0) <= .11 (MEASURED, `work/m4_offset_align.py`).
   - The within-site rank normalisation trap (gold58 -.015) was already closed on 10-07 (CLAIMED NOVEL_PLAN_10-07.md sec 6;
     logs re-read on the box).
2. **"Weight the more robust member higher" cannot be supported by evidence.**
   - We have no out-of-fold predictions for the Raptor family at all. R7 (R6 family) and our S6 seeds (S6fall family)
     use random folds, so every train site is seen.
   - The within-train stand-ins show no robustness difference that survives noise:
     - Acquisition-atypical train studies are not harder for either family: atypical minus typical AUC is -.002 to
       +.009 (MEASURED sec 5).
     - The R7-vs-S6 difference in that drop is +.0068 (sd .0081) on reader v2 and +.0004 to +.0016 on reader v4 (MEASURED).
     - On gold58 split by site proxy, every member's gap is within about 1 sd of c4's (n 28 / 30; MEASURED sec 7).
   - The sign of any reweighting is therefore unknown, so its expected value is 0 (MODEL).
3. **If the public/private split is random, the public LB already prices unseen sites.**
   - The ~390 public studies would hold unseen-site studies in the same share as private.
   - c4's per-label weights were chosen on that LB, so a reweighting has no information edge over the LB (MODEL).
   - The trick would only exist if the split were by site (thread 734681, unanswered). Even then the direction would be
     a GUESS.
4. **The visible test looks like seen sites.** Each of the 3 visible test studies has a series signature that exactly
   matches train studies (MEASURED, sec 3):
   - one signature appears in 203 train studies, ALL from the German site;
   - one appears in 228 studies (en);
   - one appears in 1,747 studies (7 sites).

   Weak evidence only, n = 3: P(3 of 3 seen | unseen share .3) = .34 (MODEL).
5. **Gain estimate: 0 (MODEL).**
   - Upper bound if a robustness gap were known: a share s = .2 of unseen-site studies, a family gap of .01 AUC on them,
     and a routing shift of .2 weight gives about +4e-4 to +7e-4 macro (MODEL).
   - The sign is unknown, so the expected value is 0.
   - No validation set exists, so the effect cannot be resolved before 10-22: public LB noise for a change like this is
     about 5e-4 or more (CLAIMED stack2/subsample_sd.log).
6. **Pick implication (minor).**
   - These measurements give no reason to believe the less-tuned pick B (R6 uniform) is more site-robust than c4.
     On gold58, c4 and the uniform-table c4 are equal on English and non-English studies: .9476 / .9446 vs
     .9454 / .9450 (MEASURED).
   - B's hedge value is against public-tuning noise, not against site shift. The current picks plan is unchanged.

## 2. How each c4 member reads a study (MEASURED, code read of the c4 notebook + embedded sources)

| member | intensity | geometry | series choice | source |
|---|---|---|---|---|
| Raptor r384 / r224 (CoAtNet, nartaa SWA) | per-series 2-98 % percentile window over the picked slices, linear clip to [0, 1], uint8, ImageNet mean/std | fixed 140 mm centre crop from PixelSpacing (fallback .5 mm), 96 slices in 5 fixed slots (SagFS 26, Sag 22, CorFS 18, Cor 12, Ax 18), slices spread over 2-98 % of each series | per slot: plane + Fluid_Sensitive flag; falls back to any series of that plane; a missing plane = zero slots | `c4_cells.py` l.870-960 (scratchpad extract of cell 4) |
| R6 (heliosli compact ConvNeXt-S, 4 folds) | mainline `intensity_mapping.series_percentile`: per-series 1-99 % window (pixels subsampled ::4), linear clip | knee-centred views `v4` (anatomy anchors from the SKM-TEA V-Net `anatomy.pt`), 384 / crop 352, 3-slice windows at 3.5 mm physical spacing, side (laterality) classifier, flip TTA | trainer's own window rows | `session_2026-10-15/jpeg/src/r7code/intensity_mapping.py` (sha 19be2d8c = c4 CODE_SHA), `r6v1/upload/.../manifest.json` |
| S6fall (nartaa B path) | same mainline code as R6 | same `v4` anchors, 320 / crop 288, 3.5 mm | same | decoded S6F driver (`B_GEOMETRY`) |

Consequences:
- All three are invariant to a series' global gain and offset. A test-time intensity normalisation has nothing left to fix.
- R6 and S6fall share one reader: the decoder, intensity code and anchor model. Their failure modes under an acquisition
  shift are correlated, not independent (MEASURED code; correlation of errors is a GUESS).
- Raptor is the structurally different reader (fixed centre crop, slot packing).
- Failed rows are already handled well. A study one member fails on takes the other member's rank (stage 3, S6F blend).
  More than 3 % Raptor failures stops the run, so the scored c4 had at most 3 % Raptor failures over the whole hidden set
  (MEASURED code + c4 scored).
- JPEG Lossless is covered by the J decoder cell. JPEG 2000 / RLE / implicit VR decode in the stock image
  (CLAIMED `session_2026-10-15/jpeg/RESULT.md`).

## 3. Site proxies in train (MEASURED, `work/m1_sites.py`, box `~/novel/site/out/*` copies in `trick/box/`)

- **Report language, 9 groups:** en 1,736, es 682, tr 546, hr 406, el 321, de 262, bg 220, nl 153, fr 81. There is no
  `it` and no `mn`. English covers at least 5 contributors (CLAIMED forum.md).
- **Scanners,** from the 458-study DICOM header sample (InstitutionName / StationName are blank):
  - Vendors: Siemens, GE, Philips, Canon/Toshiba, Fujifilm.
  - Field strength: 1.2 T (2 studies), 1.5 T, 3 T. No low-field (< 1 T) scanner in the sample.
  - Several sites use a single vendor: bg Philips 3 T; de Siemens 1.5 T; el / fr / nl Siemens.
  - Pixel spacing .09-1.17 mm; slice thickness .6-5 mm; 3D acquisitions 71 of 2,382 series.
  - FOV 81-300 mm (median 150-180 by site); 6.5 % of series below 140 mm, 0.4 % below 120 mm.

  Train therefore spans a wide acquisition envelope already. An unseen 1.5 / 3 T site would very likely fall inside it.
  A low-field site (for example an Esaote open scanner) would not; that is a GUESS with no evidence either way.
- **Protocol signatures** (series counts per plane x fluid flag):
  - The German site lacks sagittal fluid-sensitive series in 93.5 % of studies; that signature is de-only.
  - en lacks coronal FS in 9 %; fr has no coronal non-FS; axial non-FS appears mainly in en (32 %).
  - 557 of 4,407 train studies miss at least one of SagFS / Sag / CorFS / AxFS. None misses all fluid-sensitive
    series, and none misses sagittal entirely.
- **Visible test** (test_series.csv, 3 studies):
  - signature 0-2-1-0-1-1 (no SagFS) matches 203 train studies, all de;
  - 2-1-1-0-1-0 matches 228 (226 en, 2 fr);
  - 1-1-1-1-1-0 matches 1,747 (7 sites).
  - Fluid_Sensitive equals Fat_Suppression in every row, as in train.
- Prior 10-07 work (CLAIMED box logs): site can be inferred from observable features with 95.3 % accuracy
  (`verify_site/v4_obs.log`). Per-site offsets lose on gold: lam +1 gives -.0152 and lam -.25 gives +.0008, sd .0024
  (`site/s5_gold_grid.log`, `verify_site/v2_gold.log`).

## 4. Per-site behaviour of the members we hold OOF for (MEASURED, `work/m2_member_site.py`, `work/m2_v2.csv`, `m2_v4.csv`)

Members: R7 OOF (R6 family, 4 random folds; the fold x language crosstab is balanced) and our S6-recipe seeds s1044 and
s1045 (S6fall family). The common set is 3,467 non-gold studies.

| ruler | member | pooled | within-site | pooled - within | mean site offset SD among negatives (pct) |
|---|---|---|---|---|---|
| v2 | R7 | .8852 | .8904 | -.0051 | .0646 |
| v2 | S6 s44 | .8958 | .8966 | -.0008 | .0618 |
| v4 | R7 | .8853 | .8874 | -.0021 | .0658 |
| v4 | S6 s44 | .8844 | .8869 | -.0025 | .0664 |

- Per-site macro AUC is driven by the report reader, not by the member. For R7, es reads .904 on v2 but .875 on v4,
  and fr reads .863 on v2 but .897 on v4. Site-level AUC on report labels measures label noise per site.
- No size effect: small sites (fr 69, nl 116) are not consistently worse than large ones for either family.
- Site offsets among negatives are about .06 pct-rank SD for both families. Neither is more "site-invariant".

## 5. Acquisition-atypical train studies as an unseen-site stand-in (MEASURED, `work/m3_atypical.py`, `work/atypicality.csv`)

Atypicality is the mean distance to the 20 nearest train studies in store features (slice counts, JPEG bytes/slice,
middle-slice intensity stats per view) plus series composition. Each row compares the top 10 % / 20 % against the rest.
"Gap" is the atypical-minus-typical AUC drop for R7 minus the same drop for S6; positive means R7 is more robust.

| ruler | stratum | R7 atyp-typ | S6 atyp-typ | R7 - S6 (boot sd) | P(R7 more robust) |
|---|---|---|---|---|---|
| v2 | top 10 % | +.0041 | -.0027 | +.0068 (.0081) | .87 |
| v2 | top 20 % | +.0094 | +.0028 | +.0066 (.0060) | .89 |
| v4 | top 10 % | +.0032 | +.0016 | +.0016 (.0060) | .62 |
| v4 | top 20 % | -.0016 | -.0020 | +.0004 (.0045) | .54 |

Atypical acquisitions are not harder in train. The family difference is under 1.2 sd and shrinks to ~0 on the second
reader. There is no robustness signal to build on.

## 6. Test-time offset normalisation (MEASURED, `work/m4_offset_align.py`; base = .5 pct(R7) + .5 pct(S6 mean of 2 seeds))

| ruler | clusters | consensus alignment | R7 offsets -> S6's | S6 offsets -> R7's |
|---|---|---|---|---|
| v2 | language | +.00000 | -.00019 (P>0 .33) | -.00249 (P .00) |
| v2 | observable KMeans-12 | +.00000 | -.00049 (P .11) | -.00305 (P .00) |
| v4 | language | -.00000 | -.00272 (P .00) | -.00059 (P .01) |
| v4 | observable KMeans-12 | -.00000 | -.00223 (P .00) | -.00195 (P .00) |

- **Why consensus alignment is a no-op (MODEL):** blend = sum_m w_m r_m. Shifting r_m by (c - o_m) on a cluster, with
  c = sum_m w_m o_m, moves the blend by sum_m w_m (c - o_m) = 0.
- Any non-trivial offset change, whether one-sided or median-of-members, discards cluster-level signal that the members
  agree on. On OOF it loses in both directions.
- With 3 families a median could differ from the mean, but we cannot measure that: Raptor has no OOF, and gold58 is too
  small to support per-cluster offsets.

## 7. c4 members on gold58 by site proxy (MEASURED, `work/m5_gold_sites.py`; n 28 en / 30 non-en; noise-level)

| member | en | non-en | non-en - en | typical half | atypical half |
|---|---|---|---|---|---|
| Raptor (.75 r384 + .25 r224) | .9328 | .9243 | -.0085 | .9195 | .9244 |
| R6 | .9465 | .9368 | -.0096 | .9272 | .9453 |
| S6fall | .9327 | .9395 | +.0068 | .9068 | .9486 |
| c4 | .9476 | .9446 | -.0030 | .9286 | .9481 |
| c4, uniform .5 table | .9454 | .9450 | -.0004 | .9299 | .9473 |

Each member's (non-en - en) gap minus c4's gap, with bootstrap sd (1,000 resamples):
- Raptor -.0047 (.0114);
- R6 -.0069 (.0109);
- S6fall +.0098 (.0155).

All are within 1 sd. If anything S6fall leans robust, but this is noise. The two LB reads around S6fall (.15 and .3) were
both negative, which argues against moving its weight on this basis (CLAIMED task brief).

## 8. GPU measurement that would settle member robustness (spec only; NOT recommended before 10-22)

- **Goal:** rank Raptor / R6 / S6fall by prediction stability under simulated acquisition shift. This is the only way to
  get a direction for a per-study routing.
- **Inputs:** the 458 train studies whose DICOMs are on the box, minus gold74 (~400). Each member runs its own reader
  in serve mode, reusing the gold58 kernel-serve scripts.
- **Perturbations,** applied to the pixel arrays before each reader:
  - P1: Rician noise, sigma = 3 % of the series p99 (low-field SNR);
  - P2: in-plane down-then-up sampling x0.6;
  - P3: slice-thickness doubling (average adjacent pairs, halve the count, double the spacing);
  - P4: gamma .75 and 1.33 after the percentile window (contrast / TE shift);
  - P5: 120 mm FOV crop with PixelSpacing kept.
- **Metrics,** per member and label:
  - Spearman of clean vs perturbed predictions;
  - mean |delta pct|;
  - delta macro AUC vs v2 labels. The delta only, since Raptor and S6fall are in-sample on train.
- **Time (GUESS, V100):** Raptor r384 + mirror ~1 h for 6 conditions; r224 ~20 min; R6 4 folds + flip ~40 min;
  S6fall ~40 min. About 2.5-3 h in total.
- **Decision rule:** act only if one family loses at least .01 more AUC than the others on at least 4 of 6
  perturbations. Even then:
  - the routing needs an unseen-site detector built from the 86 stripped tags;
  - it has no labelled validation;
  - it would land on the LB 10-21 at the earliest, inside public noise.

  So it cannot meet the prereg bar before the deadline.

## 9. Files

- `work/m1_sites.py` (site proxies, scanners, protocols, visible test), `work/series_comp_local.csv`
- `work/m2_member_site.py`, `work/m2_v2.csv`, `work/m2_v4.csv` (per-site OOF behaviour, R7 vs S6 seeds)
- `work/m3_atypical.py`, `work/atypicality.csv` (atypical-acquisition stand-in)
- `work/m4_offset_align.py` (label-free offset alignment)
- `work/m5_gold_sites.py` (gold58 by site proxy)
- `box/{store_feats,series_comp,lang,dicom_series_tags}.csv`: read-only copies of our own 10-07 feature files
  (`~/novel/site/out/`), competition studies only
- Code reads: c4 notebook `session_2026-10-18/a18/kernel_v224r6a25c4j/rsna_sub_v224r6a25c4j.ipynb` (cells 4, 7, 14, 16,
  18); R6 embedded sources and the decoded S6F driver were extracted to the session scratchpad only
