# Laterality (left / right knee) audit, 2026-10-19

Question: is the left / right knee decision wrong for some studies, so that a member swaps the medial and lateral
compartments (Medial / Lateral Meniscus, Medial / Lateral OA, MCL)? If so, is there a test-time fix worth a slot?

Tags: MEASURED = computed here or read from code (file named); CLAIMED = a note or a third party says so; MODEL = derived
from measured numbers with a stated assumption; GUESS = no evidence either way.
Research and CPU measurement only. No kernel, no submit, no box job. Scripts and outputs: `trick/lat_work/`.

## Verdict: WEAK (do not spend a slot on it)

- The two members that carry most of the M/L weight are safe in-distribution.
  - Raptor r384 / r224 does not use laterality at all. Its compartment identity comes from anatomy. MEASURED (code).
  - R6 identifies compartments by image position, so a wrong side would swap its M/L outputs. MEASURED, rho .99.
    But it already uses tag, then sagittal x, then an image side classifier.
  - R7 OOF (same family and side logic) shows no extra swaps in any laterality-suspect group, language or vendor.
    MEASURED, 4,406 train studies.
- S6fall is the one member with a real gap. It uses the tag, then the sagittal-x sign, and has no image check.
  - It is wrong on about 1-2 % of studies, mostly untagged knees within 20 mm of x = 0. MEASURED (reports) + MODEL.
  - S6fall has weight .2, so one swapped study costs only -5e-5 macro (gold58 scale). MEASURED.
  - Expected gain from fixing it: +3e-5 to +6e-5 in-distribution. MODEL.
  - Even if the hidden unseen sites put 5 % of studies in S6fall's wrong-side zone, the gain is only +1.5e-4. MODEL.
- Nothing measured here reaches +2e-4. Laterality is not the missing trick.

## 1. How each member decides the knee side (MEASURED, code)

| Member (c4 cell) | Side source | Missing / inconsistent tag | Canonical render | M/L identity | TTA |
|---|---|---|---|---|---|
| Raptor r384 (cell 4) and r224 (cell 11) | none. IPP / IOP are read only to sort slices (`pos = IPP . (row x col)`); InstanceNumber fallback | irrelevant | none: pixels as stored, mm centre crop | anatomy. Attention-MIL head with no positional input (`RaptorClassifier`: LayerNorm, attention, softmax over windows), trained on raw L and R knees | r384: anatomical mirror (sagittal = reverse the 3 slice channels; cor / ax = width flip), labels kept. r224: plain |
| R6 bundle fe055406 (cells 7-9, compact_infer 8401812c + mainline `dicom_geometry` 12aa501f) | `decide_study_side`: (1) Laterality / ImageLaterality (L/LEFT/R/RIGHT; a file whose two tags conflict raises GeometryError; two different sides across series = `conflicting_laterality_tags`, side None); (2) else median sagittal IPP x sign (>= 0 = L); (3) else `unknown_no_flip` | image side check (`side_cls.pt`, threshold 2.0, position 20 mm). It runs on every study that lacks one explicit tag, and on tagged studies whose \|x\| >= 20 mm says the other side. It re-renders with the other side when the mean logit < -2 | R: coronal / axial width flip; sagittal slice positions negated | position (view ids 2/3 = medial / lateral coronal locals, rel_pos) | flip TTA forced on: mirror pass, then permute Med<->Lat Meniscus / OA, logit mean; MCL plain only |
| S6fall (cells 16-18, s6f_driver b_cmd, compact_infer 34f433fe) | same `decide_study_side` | **no side check**: `--side-model` is not passed (b_cmd), although the worker supports it | same as R6 | position (same Net family; trained with `aug: crop`, no mirror, `px_override: null` = no side fix in training) | none |

Nobody uses BodyPartExamined or SeriesDescription for the side (grep, MEASURED).

## 2. DICOM side in train (MEASURED, `session_2026-10-15/kcrop/out_kc_side_v1/kc_side_train.csv`, all 4,407 studies)

- Laterality tag: present on 2,189 studies (49.7 %), absent on 2,218.
  - Tagged by vendor: Siemens Healthineers 1,051 / 1,053; GE MEDICAL SYSTEMS 0 / 868; TOSHIBA 0 / 181; CANON 0 / 45.
- Tag vs sagittal-x sign: they disagree on 48 / 2,189 (2.2 %), mostly where \|x\| < 20 mm.
- Knees near the scanner centre:
  - \|x\| < 20 mm: 121 studies (2.7 %), of which 62 are untagged (1.4 %).
  - The untagged near-centre studies are mostly TOSHIBA (46 / 181 have \|x\| < 20) plus a few GE and Philips.
  - Near-centre tagged studies are GEHC, Hitachi and FUJIFILM, and there the tag agrees with the report.
- Slice geometry is regular. For all 4,407 train studies and all 3 visible test studies, every slot has the expected IOP sign:
  - sagittal normal x < 0;
  - coronal and axial row x > 0.
- The 3 visible test studies: one tagged L (x = +61), two untagged (x = -81, -112). None is near the centre.

## 3. Reports vs DICOM (MEASURED, `lat_work/report_side.py` -> `report_side.csv / .json`)

- Method: knee-context side words in 9 languages (left/right, links/rechts, izquierda/derecha, sol/sağ, lijevo/desno,
  ляво/дясно, αριστερό/δεξί, gauche/droit) within one token of a knee word.
- 1,661 / 4,407 reports name exactly one side.
- Agreement of the report side with:
  - x sign: 96.3 % (61 disagree);
  - KC side (x, then tag near 0): 98.3 %;
  - DICOM tag: 98.4 % (15 / 933 disagree).
- Disagreement with the x sign by \|x\| bin:

| \|x\| | 0-10 | 10-20 | 20-30 | 30-50 | 50-80 | > 80 mm |
|---|---|---|---|---|---|---|
| report != x sign | 26 / 61 | 11 / 40 | 3 / 42 | 4 / 127 | 7 / 548 | 9 / 842 |

  - Under 20 mm the x sign is close to a coin flip (37 %); above 30 mm it is wrong about 1 % of the time.
- The large-\|x\| disagreements are partly report artefacts.
  - bg reports carry a "normal soft tissues of the left knee" template line even in right-knee reports, so I excluded bg.
  - Some en reports name the other side.
- 48 studies have a tag that disagrees with x. The report sides with the tag in 25 (all near the centre: GEHC, Hitachi, FUJIFILM),
  sides with x in 6 (all \|x\| > 59 mm, Siemens / Philips) and is silent in 17.
  - This matches heliosli's comment that 18 train studies had the tag contradicted by both position and images (CLAIMED,
    compact_infer docstring).
- The rule S6fall uses (tag, else x sign) disagrees with the report on 36 / 1,519 non-bg report-sided studies (2.4 %).
  - 17 of these are untagged with \|x\| < 20 mm (S6fall almost surely wrong there).
  - About half of the rest are report noise.
  - Estimated S6fall wrong-side rate in train: f_S6 about 1-2 % of studies (MODEL from MEASURED).

## 4. Is there a swap signal in model outputs?

### R6 is position-based (MEASURED, gold58 R6 worker dump `r6g58_box/run_b/worker0/dump.json`)

The mirrored pass before the permutation (`z_mirror_raw`, fold mean) compared with the plain pass:

| column of the mirrored pass | Spearman with the same plain column | Spearman with the other compartment's plain column | AUC vs own gold | AUC vs other-compartment gold |
|---|---|---|---|---|
| Medial Meniscus | .344 | .989 | .668 | .912 |
| Lateral Meniscus | .320 | .991 | .595 | .962 |
| Medial OA | .829 | .988 | .864 | .855 |
| Lateral OA | .798 | .989 | .779 | .992 |

- A mirrored knee makes R6 report the other compartment, so a wrong side swaps R6's M/L outputs. The flip TTA cannot catch this,
  because both passes are swapped.
- The permutation inside R6's flip TTA is correct.

### R7 OOF on train vs LLM report verdicts (MEASURED, `lat_work/swap_oof.py`, `lang_cross.py`)

R7 shares R6's side logic, and its training cache had heliosli's side fix (`px_override .../px384_v4_sidefix`).

| group | studies | discordant M/L pairs | model orders the pair correctly | studies with swap LLR > 2 |
|---|---|---|---|---|
| all | 4,406 | 2,104 | .908 | 2.5 % |
| clean (tagged, tag = x, \|x\| > 50) | 1,906 | 991 | .896 | 3.2 % |
| untagged, \|x\| < 20 | 62 | 14 | .857 | 3.2 % |
| untagged, \|x\| < 20, report != x | 17 | 5 | 1.000 | 5.9 % (1) |
| tag != x | 48 | 24 | .917 | 4.2 % |
| report != (tag, else x) | 36 | 13 | .846 | 5.6 % (2) |

- No suspect group shows a swap excess beyond noise. The 17 studies where the x sign is wrong by the report are ordered correctly
  (5 / 5 pairs), which is consistent with the side classifier / side fix doing its job.
- The top swap-LLR studies are mostly in the clean group: label or model errors, not laterality.
- By language, meniscus pairs are ordered correctly .86-.97 everywhere. By vendor the rate is .92-1.00 (vendors with n >= 40).
  - OA pairs are lower in hr (.68, n 37) and bg (.65, n 17), but hr / bg menisci are .91 / .97.
  - A laterality fault would hit both pairs equally, so this is OA label noise, not side.

### gold58 (MEASURED, `lat_work/g58_swap.py`, `r6s6.py`, `whichflip.py`, `g58_cross.py`)

- gold58 has 26 tagged and 32 untagged studies.
  - Untagged with \|x\| < 20: 2 studies, and in both the report agrees with the x sign.
  - The 22 report-sided gold58 studies all agree with S6fall's rule.
- R6's side check ran on 32 and flipped 1 (receipt). No gold58 study shows opposite M/L asymmetry between any two members
  (R6, S6fall, R7oof, r384). The flipped study is evidently near-symmetric, so the flip changes nothing visible.
- Every member's Lateral OA column scores about as well against the Medial OA gold as against its own:
  - R6 .864 vs .865;
  - r384 .902 vs .776.
  - This happens in Raptor (anatomy-based) as much as in R6, so it is OA co-occurrence, not laterality. Co-occurrence stacking is
    already closed (-.001 LB).

## 5. What a swap costs, and the value of fixing it

The c4 recipe was rebuilt on gold58 (r384 / r224 .75 / .25, c4 Raptor shares, .8 / .2 S6fall; macro .9410). For each study I
swapped the M/L columns and took the mean change in macro (MEASURED, `g58_swap.json`):

| swapped in | mean d macro per swapped study (58-study scale) | worst |
|---|---|---|
| S6fall only | -5e-5 | -7e-4 |
| R6 only | -3.5e-4 | -3.4e-3 |
| R6 and S6fall (Raptor is immune) | -5.8e-4 | -4.7e-3 |

On a hidden set of N studies, a wrong-side fraction f costs about f x 58 x (per-study figure) (MODEL). A Monte Carlo with 3 of
58 swapped in S6fall gives -2.1e-4 against -1.6e-4 from the linear formula.

| f | S6fall only | R6 only | both |
|---|---|---|---|
| 1 % | -3e-5 | -2.0e-4 | -3.3e-4 |
| 2 % | -6e-5 | -4.1e-4 | -6.7e-4 |
| 5 % | -1.5e-4 | -1.0e-3 | -1.7e-3 |

### Buildable fixes and their expected value

1. **Pass R6's side classifier to S6fall** (`--side-model <R6 bundle>/side_cls.pt --side-threshold 2.0 --side-position-mm 20
   --side-max-x 0` in s6f_driver b_cmd). compact_infer 34f433fe supports these flags (MEASURED,
   `session_2026-10-15/s6f/src_extract/compact_infer.py.34f433fe`, side check l.19-79, argparse l.265-266), and `side_cls.pt` is already attached.
   - Gain: f_S6 is about 1-2 % in-distribution, so +3e-5 to +6e-5 (MODEL).
   - If the unseen Italian / Mongolian sites put about 5 % of hidden studies in the untagged near-centre zone, the gain is
     about +1.5e-4. GUESS: nothing is known about their scanners. An Esaote-style dedicated extremity scanner would centre the
     knee at x = 0.
   - Costs:
     - S6F parity changes;
     - extra decode for re-rendered studies;
     - a side-net pass on about half the studies;
     - a new gate;
     - a full slot (about 13 h to a score).
   - P(positive on the private LB) is about .55-.6, and the size is far below the resubmit noise. Not recommended. The
     10-17 notes (`research_1hr/levers.md` sec 6) reached the same conclusion.
2. **Lower R6's side threshold, or let Raptor (anatomy-based) referee R6 on side-uncertain studies.** R7 OOF shows no residual
   swaps in-distribution, so this would only add false flips. The benefit exists only in the unseen-site scenario, which cannot be
   measured. Not recommended.
3. **Swap S6fall's M/L columns wherever R6's side check flipped a study.** This is the same benefit as fix 1, but the R6 worker
   (sha-pinned) would have to export its per-study side notes. Not recommended.

## 6. GPU measurements that would settle the rest (specified, not run)

- **G1, Raptor anatomy check (about 10 min on the V100).**
  - Run cell 4's r384 path on gold58 (box `~/kmimic` mirror of the kernel) twice: plain only (`mirror=False`) and mirror only.
  - Save 12 probabilities per pass. Report the Spearman of the mirror-only Medial / Lateral columns against the same and the
    crossed plain columns, plus per-label AUC for plain, mirror and mean.
  - Expectation: same > cross. That would confirm Raptor is immune to side errors, and that its label-keeping mirror TTA does
    not blur M/L.
- **G2, S6fall with the side model (about 25 min).**
  - Run the S6F B path on gold58 plus the H458 mimic studies with and without the four `--side-*` flags.
  - Count the studies whose side changes, and the d macro on gold58 and on the H458 LLM labels.
  - Expectation: about 1-2 % of studies change, and |d macro| < 1e-4.

## Files

- `lat_work/report_side.py` -> `report_side.csv`, `report_side.json`: report side vs DICOM (tag, x, KC side).
- `lat_work/swap_oof.py` -> `swap_oof.csv`, `swap_oof.json`: R7 OOF swap evidence by suspect group.
- `lat_work/lang_cross.py` -> `lang_cross.csv`: own / cross AUC and discordant-pair ordering by language, vendor and near-centre.
- `lat_work/g58_swap.py` -> `g58_swap.json`: R6 position test, gold58 suspects, swap costs.
- `lat_work/r6s6.py`, `whichflip.py`, `g58_cross.py`: gold58 member-vs-member asymmetry and own / cross AUC.
