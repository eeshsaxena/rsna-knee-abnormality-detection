# FIGS.md: writeup figures (RSNA Knee, 10-23)

Five PNGs, 1600 px wide, 200 dpi, white background, Segoe UI 10 to 15 pt, colour-blind-safe palette (dataviz reference palette:
blue #2a78d6, orange #eb6834, aqua #1baf7a, yellow #eda100, magenta #e87ba4, accent violet #6250d6). The palette was checked with
the dataviz validator on #ffffff: all checks PASS (worst adjacent CVD dE 9.1, normal-vision dE 19.6); aqua, yellow and magenta sit
below 3:1 contrast, so every figure carries direct value labels or marker-shape encoding as well as colour.

Regenerate (CPU, under a minute, writes only inside writeup/figs):
```
cd C:/Users/eeshs/Downloads/rsna-knee/writeup/figs/src
py -3.12 dose_gold58.py     # -> ../data/dose_gold58.json (aggregates only; self-check must print DOSE_SELF_CHECK PASS)
py -3.12 make_figs.py       # -> ../pipeline.png, coupling.png, oai_validation.png, lb_progress.png, dose_curve.png
```

Governance: OAI numbers are aggregate AUC deltas, bootstrap sd, P values and correlations copied from laptop logs; no OAI row, id or
per-knee value was read or shown. No competition image, report text or study-level label appears in any figure.
`dose_gold58.py` reads the gold58 labels locally only to compute macro AUC; it writes aggregates only and deletes its scratch dir.
Paths below are relative to `C:/Users/eeshs/Downloads/rsna-knee/`.

---

## 1. pipeline.png

**Caption.** Final pipeline. Three public model families (Raptor CoAtNet r384 + r224, the R6 compact ConvNeXt-S bundle, S6fall)
are rank-blended per label into the c4 output, then E-OA decorrelates the three OA columns. Pick A (swide) adds two small
meniscus terms; pick B (r6pw20) applies pair-weighted E-OA to the R6 entry. Canaries make any failed stage visible on the
leaderboard instead of letting it pass silently.

| item in the figure | source |
|---|---|
| JPEG-lossless decoder cell, zero-fill / drop behaviour without it | c4 notebook `session_2026-10-18/a18/kernel_v224r6a25c4j/rsna_sub_v224r6a25c4j.ipynb` cell 2 header comment |
| Raptor r384 anatomical mirror TTA, nartaa SWA weights | same notebook, cells 0-1 (credits) and cell 4 (`_mirror_tri`, MIRROR_TTA) |
| rank mix .75 r384 + .25 r224 | cell 14 `A224 = 0.25` (share of the Raptor slot given to r224) |
| R6 4 folds, flip TTA with M / L swapped back | cell 8 (V224-FLIP comment: mirrored windows, Medial / Lateral Meniscus and OA swapped, logit average) |
| R6 Effusion and Baker's stackers | cell 7 (`fluid_stacker`, `state_stacker`); `session_2026-10-19/trick/label_regime.md` sec 3 note (state stacker adopted for Baker's) |
| per-label Raptor share table (Medial OA .5, Lateral OA .2, PF OA .5, Synovitis .8, Baker's .5, Lateral Meniscus .4, Fracture .65, rest .5) | cell 14 `W_RAPTOR, W_OURS = 0.5, 0.5` and `W_RAPTOR_C` |
| final = .8 pct(blend) + .2 pct(S6fall), average-tie pct | cell 18 header comment and `W_S6F = 0.2` |
| canary r224 -> Fracture = 0.5 | cell 15 (`CANARY_FIRED r224 not used ... Fracture set to 0.5`) |
| canary S6fall -> ACL = 0.5 | cell 18 `CANARY_COL = 'ACL'` (fires on coverage < .98 or low rank corr with V224) |
| E-OA formula, sym w = 1, pw w(LOA,PF) 1.416, w(MOA,LOA) .876, w(MOA,PF) .708 | `session_2026-10-20/a21/gate_a21_offline.py` l.46-52 (`Wp`, `expect`); HANDOFF.md 10-20 11:35 block |
| swide = pw a .25 + Lateral OA <- Medial Meniscus .05 + MCL <- Medial Meniscus .1 | `session_2026-10-20/stack/build.md` l.55-57 and l.63; HANDOFF.md FINAL HANDOFF block (l.11) |
| r6pw20 = R6 57248084 base + pw a .2; R6 = uniform weights + S6fall .2; r224 share .35 | HANDOFF.md l.12 and l.319; `session_2026-10-20/val/kernels/kernel_v224r6flips6fpw20j` (`A224 = 0.35`) |
| about 0.916 public for a canary run; 4 late entries hit it | `session_2026-10-23_scores.json`: 57463719, 57464296, 57464911 (.916 / .901-.902), 57522767 (.916 / .901); HANDOFF.md l.23 |

## 2. coupling.png

**Caption.** Pairwise coupling of the three OA labels. Report-trained model scores (c4 on gold58) couple the compartments at
.81 to .85, almost exactly like the LLM-read report labels (.77 to .83). Expert-regime labels couple them far less: .39 to .61
on the RSNA expert sets and about 0 on 2,399 OAI knees read by MOAKS radiologists. This excess coupling is what E-OA removes.

All values from `session_2026-10-19/ex/coupling_map.md` sec 2 table (MEASURED there), rows #2 MOA-LOA (l.171), #4 MOA-PFOA (l.173),
#1 LOA-PFOA (l.170):

| bar | MOA-LOA | MOA-PF | LOA-PF | column in coupling_map.md |
|---|---|---|---|---|
| c4 model scores, Spearman on gold58 | .85 | .82 | .81 | "c4 rho g58" |
| report labels (report_labels_v2, YES / NO, UNK dropped pairwise), tetrachoric | .83 | .80 | .77 | "report tet" (UNK=0 variant in brackets there: .67 / .65 / .64) |
| gold58 expert labels, tetrachoric | .51 | .56 | .45 | "gold58 tet" |
| gold74, tetrachoric, error bar = bootstrap sd | .59 +- .16 | .61 +- .13 | .39 +- .20 | "gold74 tet +- sd" |
| OAI MOAKS expert reads, tetrachoric | .12 | .13 | -.01 | "OAI expert tet (phi)" |

Notes:
- Scale: model coupling is Spearman of scores, label coupling is tetrachoric (latent) correlation, so both sit on a
  continuous-latent scale (coupling_map.md sec 1, "Coupling measures"). Phi values are lower because dichotomisation attenuates
  them; for reference, OAI label phi is .061 / .074 / -.006 (`session_2026-10-19/trick/oai_check/g1_box_run.log` l.1) and on gold58
  reports co-call Lateral OA with PF OA at phi .59-.73 vs experts .28 (`label_regime.md` sec 0 item 2). The brief's phi ranges
  (experts .24-.40, OAI ~.05, reports .53-.61) are in HANDOFF.md 10-19 23:30 block; the figure does not mix phi with Spearman.
- gold74 is NOT a pure expert set: its 16 extra rows are our own report reads under the gold convention (`label_regime.md` sec 1,
  third bullet). The legend says so.
- "The rubric scores each compartment separately": `label_regime.md` sec 0 item 1 (host comment 733343).

## 3. oai_validation.png

**Caption.** Independent validation before any submission. On 2,399 OAI knees with expert MOAKS cartilage reads, three OAI-free,
report-trained models all gain OA-mean AUC from E-OA (+.005 to +.006 at a .1, +.009 to +.012 at a .2; P(>0) = 1.000), and the
opposite transform (a = -0.1, positive borrowing) loses for all three. The decision rule was written before the run.

Primary proxy `[mean P_FT]` (any cartilage subregion with full-thickness loss), OA-mean AUC delta, error bars = +-1.96 x the logged
paired-bootstrap sd:

| model (figure name) | run name | a -0.1 (sd) | a +0.1 (sd) | a +0.2 (sd) | source |
|---|---|---|---|---|---|
| Model 1: 5-fold mean (s1043) | teacher_D5_mean_v00 (f0-4 s1043_ttr_frac75) | -.00659 (.00038) | +.00624 (.00042) | +.01154 (.00089) | `session_2026-10-19/trick/oai_check/g1_box_run.log` l.2-4 |
| Model 2: single fold (s1044) | f0_s1044_ttr_mix50 | -.00581 (.00040) | +.00498 (.00044) | +.00859 (.00092) | `session_2026-10-19/trick/oai_check/g1_replication_box.log` l.3-5 |
| Model 3: single fold, 288 px (s1045) | f0_s1045_ttr288frac75 | -.00610 (.00042) | +.00531 (.00045) | +.00929 (.00094) | `g1_replication_box.log` l.19-21 |

- n 2,399 and P>0 1.000 at a .1 / .2 for every model: same log lines; VERDICT lines (`g1_box_run.log` l.33-35,
  `g1_replication_box.log` l.14-16, l.30-32): SUPPORT, 3/3 proxies, control negative.
- a = 0 is 0 by definition (untransformed model).
- Pre-registered rule: `session_2026-10-19/trick/TRICK_1019.md` sec 4 (G1) and `session_2026-10-19/a20/A20_VALIDATION_PREREG.md`
  ("Strong" rule, RESULT: PUSH BOTH). OAI-free status of the models: CLAIMED by their headers (TRICK_1019.md sec 4 and sec 7 B).
  288 px for s1045: A20_VALIDATION_PREREG.md l.7.
- Caveat for the text (not in the figure): OAI label coupling (phi ~.05) is far below RSNA expert coupling, so the raw OAI gain
  overstates the RSNA gain; the skeptic's coupling-matched transfer estimate was about +.0005 (a .1) / +.0007 (a .2) macro
  (A20_VALIDATION_PREREG.md RESULT, V2); the coupling-matched ruler understated the public effect about 1.6-3x at a .1 / .2
  and turned negative at a .3, while gold58 and raw OAI overstated it about 2.4x and 3x (`session_2026-10-20/night/recal.md` sec 0).

## 4. lb_progress.png

**Caption.** Public (top) and private (bottom) scores of the entries on the path to the final pick, in submission order, as
Kaggle displays them (3 decimals). Model and blend changes moved the public score from .957 to .958; E-OA (OA decorrelation) gave
the next step on both boards (+.001 public and private) and held on the private set. Final: private .944, rank 246 / 6,425,
silver.

All scores from `session_2026-10-23_scores.json` (Kaggle API display values, MEASURED 10-23):

| x label | submission id | public | private | what changed (source) |
|---|---|---|---|---|
| S6F | 57201874 | .957 | .942 | V224-FLIP (Raptor + R7) + S6fall .2 (scores json description) |
| R6 | 57248084 | .958 | .943 | R7 -> R6 swap (HANDOFF.md l.324) |
| HW | 57262115 | .958 | .943 | + heliosli per-finding weight table (HANDOFF.md l.318) |
| a25 | 57316395 | .958 | .943 | r224 share .35 -> .25 (HANDOFF.md l.265) |
| c4 | 57343435 | .958 | .943 | table PF OA .5, Baker's .5, Lateral Meniscus .4, Fracture .65 (HANDOFF.md l.258) |
| eoa10 | 57413380 | .959 | .944 | c4 + E-OA sym a .1 (HANDOFF.md l.139) |
| eoa20 | 57413368 | .959 | .944 | c4 + E-OA sym a .2 (HANDOFF.md l.133) |
| pw30 | 57425970 | .959 | .944 | c4 + E-OA pw a .3 (HANDOFF.md l.119, l.134) |
| swide | 57464326 | .960 | .944 | pw a .25 + 2 meniscus terms (HANDOFF.md l.11) |

- Subtitle numbers: rank 246 / 6,425, silver cut rank 321: HANDOFF.md l.2. Pick B r6pw20 copy 3 = 57523232, .959 / .944:
  scores json.
- "E-OA: +0.001" = eoa10 minus c4 display values on each board (.959 - .958, .944 - .943). Display values are truncated, so the
  true step is only bounded, not measured.
- c4 changed four table entries at once and swide stacks three terms, so not every entry is a single knob; each entry was
  pre-registered and gated (PREREG_10-15.md amendments).

## 5. dose_curve.png

**Caption.** gold58 dose curve (58 hand-labelled studies, info only): macro AUC change vs c4 as the E-OA strength a grows, for
the symmetric and pair-weighted shapes. Both rise to about +.0025 near a .2 to .35 and turn negative past a .5. Filled markers
are doses that were submitted. With 58 studies the paired bootstrap sd is about .001 at a .2, so this curve was a code gate,
not a ruler.

Numbers: `writeup/figs/data/dose_gold58.json`, made by `writeup/figs/src/dose_gold58.py` (runs the c4 notebook's own cell 18 on the
gold58 serve predictions in `session_2026-10-18/next/g58/`, the code path of `session_2026-10-20/a21/gate_a21_offline.py`, then
the E-OA recompute that the gates matched to 2.8e-8). c4 base macro .94113.

| a | sym delta | pw delta | previously logged (file) |
|---|---|---|---|
| .05 | +.00027 | +.00027 | new |
| .10 | +.00142 | +.00142 | sym: `session_2026-10-19/a20/gate_a20_offline.log` l.6, l.10 (.94255, +.00142) |
| .15 | +.00160 | +.00185 | new on this code path (m14 path also +.00160 sym: `session_2026-10-19/ex/local_curve.log`) |
| .20 | +.00247 | +.00265 | `session_2026-10-20/a21/gate_a21_offline.log` (a20_eoa20 +.00247, pw20 +.00265) |
| .25 | +.00210 | +.00255 | `session_2026-10-20/push/shape/gate_shape_offline.log` l.31, l.34 |
| .30 | +.00200 | +.00223 | gate_a21_offline.log (sym30 +.00201, pw30 +.00223; sym differs by rounding only) |
| .35 | +.00211 | +.00256 | pw: gate_shape_offline.log l.36, `session_2026-10-20/val/gate_val_offline.log` (pw35) |
| .40 | +.00209 | +.00190 | pw: gate_val_offline.log (pw40 +.00190) |
| .45 | +.00146 | +.00157 | pw: gate_val_offline.log (pw45 +.00157) |
| .50 | +.00028 | +.00016 | pw: HANDOFF.md l.120 (CLAIMED, matches) |
| .55 | -.00089 | -.00120 | new |
| .60 | -.00283 | -.00256 | pw: HANDOFF.md l.120 (CLAIMED, matches) |

- The script exits non-zero unless the base (.94113) and all 10 previously logged points reproduce to 1e-5 (run output:
  DOSE_SELF_CHECK PASS).
- sym .40 / .50 / .60 (+.00209 / +.00028 / -.00283) were not in any laptop log before this run; they match the brief's
  +.0021 / +.0003 / -.0028.
- The m14 rebuild (`label_regime.md`, base .94102) gives sym .2 = +.00239 instead of +.00247: a slightly different c4 rebuild.
  The figure uses the kernel code path (base .94113) throughout.
- Submitted doses (filled markers): sym .1 / .2 / .3 = eoa10 57413380 / eoa20 57413368 / eoa30 57425223; pw .2 / .25 / .3 = pw20
  57425269 / pw25 57463742 / pw30 57425970 (all .959 / .944 in `session_2026-10-23_scores.json`). swide uses pw .25.
- Bootstrap sd about .001 at a .2: `label_regime.md` sec 0 item 8 (c4 bootstrap a .2 sd .00103).

## Files

- `writeup/figs/pipeline.png`, `coupling.png`, `oai_validation.png`, `lb_progress.png`, `dose_curve.png`
- `writeup/figs/data/dose_gold58.json` (aggregates only)
- `writeup/figs/src/make_figs.py`, `writeup/figs/src/dose_gold58.py`
