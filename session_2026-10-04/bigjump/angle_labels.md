# Big jump, angle = training targets (2026-10-04 ~05:00 IST)

Research only: nothing submitted, pushed, versioned, posted or run on the box. Laptop CPU only (pandas on 58-4407 rows).
Tags: **MEASURED** (our runs / ledger / LB, or computed today on gold58), **CLAIMED** (forum self-report),
**MODEL** (closed-form binormal), **GUESS**. `SP` = scratchpad, `BJ` = `SP/bigjump`, `POSTS` = `SP/intel/live/top_team_posts.txt`,
`TIAN` = `SP/knee/tian/experiments.md`, `LEDGER` = `SP/round2/measured_levers.csv`.
Scripts and outputs written today: `BJ/labels_angle_check.py|.out`, `BJ/labels_yes_decomp.py|.out`, `BJ/labels_ownfill.py|.out`, `BJ/labels_combo.py|.out`.

## 0. Bottom line (honest)

- Targets alone cannot deliver +0.007..+0.015 private over the 0.943 parent. Using the effective-rho calibration
  (gold58 rho + ~0.10, `BJ/calib_rho.py`: v09r .835 -> ~.935, crop .81 -> ~.91, which reproduces Tian's 0.000/-0.001 fork reads),
  the best target choice is worth about **+0.001..+0.002 more blend lift** than v4 or mix50 at the same base fleet (MODEL, section 3).
- The band needs an own fleet at **solo >= ~0.945 with gold-rho <= ~0.80** (MODEL: 0.945 @ .80 -> +0.0056; 0.950 @ .80 -> +0.0086;
  0.951 @ .83 -> +0.0084 effective). Targets supply part of the solo (+0.006..+0.011 GUESS) without spending as much rho as mix50 does.
- Best target route = **DSD, "decorrelated silent-cell distillation"**: own (non-parent) out-of-fold image predictions fill the report-silent
  and report-NO cells; the Raptor teacher touches only report-YES cells, lightly. On gold58 at target level it **dominates mix50 on both axes**
  (AUC .9307 vs .9262, within-class rho to the parent's Raptor arm .685 vs .741; MEASURED today with an optimistic own-teacher analog).

## 1. What the label noise actually is (MEASURED today, gold58, `labels_angle_check.out` [1])

| report verdict (pilkwang) | cells | P(gold = 1) | v4 within-stratum AUC | Raptor within-stratum AUC |
|---|---|---|---|---|
| YES | 311 | **0.653** (MCL .50, Contusion .44, Lat OA .56) | 0.811 | 0.872 |
| UNK (silent) | 177 | 0.136 (Synovitis .34, PF OA .21, Fracture .15) | **0.50** (v4 is constant there) | 0.93 |
| NO | 196 | 0.041 | 0.681 | 0.785 |

- Verdict-level errors: 108 report-YES false positives vs 32 positives hidden in UNK/NO cells (77% / 23%).
- Verdict order YES > UNK > NO holds for every label except two tiny inversions (Lat OA UNK 0/22 vs NO 2/19; Medial OA 0/14 vs 1/21), so
  **per-label calibration toward gold is AUC-invariant** (and forward noise correction with a class-conditional T is too, at the optimum). Dead.
- Teacher-mix gain decomposed (`labels_yes_decomp.out`): Raptor 0.5 inside YES only +0.017, inside UNK only +0.017, inside NO +0.001
  (total mix50 +0.034). Oracle bounds: perfect ranking inside YES +0.044, inside UNK +0.028.
- **Site / language debiasing is dead** (MEASURED, [2]): removing per-language logit offsets (fit on the 4349 non-gold studies) lowers
  target AUC: v4 -0.0076 (lambda .5, P>0 .02) / -0.019 (lambda 1, P>0 .00); Raptor arm -0.0003 / -0.009. The parent's language offsets
  correlate .83-.99 with v4's on 8 of 12 labels (Synovitis .35 is the outlier): per-site prevalence differences are mostly real case mix.

## 2. Target frontier on gold58: AUC vs within-class Spearman to the parent's Raptor arm (MEASURED today)

| target table | gold58 AUC | wc-rho Raptor | note |
|---|---|---|---|
| steven_v4 (LLM only) | 0.8927 | 0.443 | current v4 runs |
| own fill UNK a .75 | 0.9051 | 0.476 | +0.0126 vs v4, SD .0057, P>0 .995 |
| own fill UNK+NO a .75 | 0.9130 | 0.551 | "DSD-own" |
| **own UNK+NO a .75 + Raptor YES a .25** | **0.9307** | **0.685** | **"DSD-combo" (primary)** |
| mix50_logit (Raptor 0.5 everywhere) | 0.9262 | 0.741 | current mix runs |
| mixSA_logit | 0.9300 | 0.766 | |
| own UNK+NO a .75 + Raptor YES a .5 | 0.9345 | 0.776 | |
| widedense teacher alone | 0.9127 | 0.874 | |

Own teacher here = rank-mean of our 3 v4-label crop R50 members (gold58 .8884, i.e. weaker than v4 labels .8927). CAVEAT: those
checkpoints were best-of-30 selected on gold58, so the own-fill rows are optimistic; section 5 replaces them with honest OOF.
Per label the own UNK fill gains sit in Fracture (.793 -> .861), PF OA (+.024), Baker's (+.016), Synovitis (+.015).

Why this reconciles the ledger: Tian's OOF self-distillation nulls (v09t 0.917 vs 0.918, TIAN:80; v09o 0.927 = v09r, TIAN:103) mixed a
WEAKER teacher flat over ALL cells, diluting YES/NO cells where the LLM carries signal; NTejas (-0.0064 gold, LEDGER:63) REPLACED 86% of
Synovitis cells. Archit's claim (POSTS:289-295, CLAIMED) is exactly mix-not-replace on silent cells, OOF, heavier than 50/50. On UNK
cells the LLM's within-stratum AUC is 0.50, so any OOF image model beats it there: his precondition holds by construction.

## 3. Expected solo and blend (MODEL, `BJ/labels_combo.py` + the one-liner in this session; best-W lift over .943)

Student transfer assumptions (GUESS from measured ratios): solo LB gain ~0.28 x target gain (Tian: target +0.032 -> LB +0.009, TIAN:83/93);
student rho shift ~0.21 x target-rho shift (v4 -> mix50: target .443 -> .741, fleet fold-0 OOF rho_v8 .811 -> .873, `SP/k2/RUNBOOK.md`:104).

| fleet variant (base v4 solo 0.935) | solo | gold-rho | lift @ gold-rho | lift @ effective rho (+.10) |
|---|---|---|---|---|
| v4 | 0.935 | .780 | +0.0068 | +0.0024 |
| mix50 | 0.944 | .842 | +0.0077 | +0.0031 |
| DSD-own | 0.941 | .802 | +0.0082 | +0.0036 |
| **DSD-combo** | **0.946** | **.830** | **+0.0092** | **+0.0048** |

Base 0.930: v4 +0.0011, mix50 +0.0011, DSD-combo +0.0023 (effective). Base 0.940: v4 +0.0041, mix50 +0.0065, DSD-combo +0.0084.
So DSD-combo reaches the +0.007..+0.015 band only if the v4-equivalent fleet ensemble is already ~0.94 solo (seeds, all-data, model work
from the other angles) AND transfer holds. Label-only share: +0.0012..+0.0043 over v4.

## 4. Recipe

1. **Own OOF teacher table `own_oof_v1.csv` (4407 x 12)**: crop R50 by-site folds on v4 labels, existing pipeline (`box_finetune_aug.py`,
   K=8, 150 mm crop shards already on the box). Folds 1/2/4 checkpoints exist (cropaug_f1 Turkish, cropaug_f2 Spanish, crop_f4 English):
   inference only on each fold's held-out studies. Train folds 0 and 3 (remaining ~1,490 hr/el/de/bg/nl/fr studies), 30 epochs, ship the
   `_last` epoch (never best-on-gold). gold58 rows = mean of the 5 fold models (gold never trained on). V100: 2 x ~2.8 h + ~0.5 h inference.
2. **Targets `labels_dsd_v1.csv`** (0 GPU; same logit mean/std match as `build_mixSA.py ... logit`): per label, verdict from
   `report_labels_v2.csv` (missing -> UNK). YES: 0.75 logit(v4) + 0.25 matched Raptor proxy (`teacher_v2_out/raptor_proxy_v5v8.csv`).
   UNK and NO: 0.25 logit(v4) + 0.75 matched own OOF. Pre-registered constants, no per-label tuning.
3. **Gate T (0 GPU)**: on gold58 with the REAL table: target AUC >= 0.926 (mix50) AND wc-rho to Raptor <= 0.70. Fail -> D1 as planned.
4. **Student**: fleet ConvNeXt-T, exact f0_s42 contract (4 views x 24 triplets, 320 px, 140 mm, fp16 AMP, EMA .999, mirror aug, masked soft BCE),
   **fold 0 first** (`--fold 0 --labels labels_dsd_v1.csv --eval_labels llm_labels_v4_blend.csv`, seed 42, 10 epochs) so its OOF rho_v8 and
   gold58 compare 1:1 with the two running fold-0 runs. If it passes gate S (section 5), all later all-data seeds (s43, s143, ...; ENDGAME fv3)
   use DSD instead of the D1 winner. V100: ~7.5 h alone / ~12 h with the second lane busy; all-data seeds replace planned ones (0 extra).
5. **Kaggle T4**: 0 for training. One K-commit per shipped member set (parent + fleet, per-file fail-loud, multi-file outputs), ~0.5-0.7 T4-h
   per commit on the 3-study sample; scoring reruns cost slots, not quota. Plan 2 commits (fold-0 DSD probe, final DSD fleet) = ~1.5 T4-h.
6. **Runtime in the 9 h kernel**: identical to any fleet member (same backbone, input contract, member count); the crop teacher never ships.

Timeline: f0_s42 (mix50) ends ~09:30, f0_s42_v4 ~10:45 IST (RUNBOOK) -> crop folds 0 and 3 in the two lanes -> teacher table ~16:00-17:00 -> DSD fold-0 epoch 5 ~23:00-01:00 (gate S) -> epoch 10 ~10-05 morning ->
K-commit + submit 10-05 (read ~6 h later) -> all-data DSD seeds 10-05..10-08 -> fleet read 10-09/10 -> before the 10-12 decision.

## 5. Falsification within 24 h (pre-registered)

- **F0 (box, ~20 min, no training)**: dump per-study gold58 predictions of the two fold-0 fleet checkpoints and the crop `_last` files, rerun
  `labels_ownfill.py` / `labels_combo.py` with them as the own teacher. KILL if own UNK+NO a .75 gains < +0.008 over v4, or DSD-combo is not
  >= mix50 AUC at lower wc-rho.
- **Gate T** (above), ~17:00 IST with the real OOF table.
- **Gate S** (DSD fold-0 at epochs 4-5, same epochs as the running runs; d1_pick-style mean of raw+EMA): KILL if OOF rho_v8 >= 0.86
  (mix50 is .873, v4 .811 at the provisional read) OR gold58 < the v4 run's gold58 at the same epochs. Do not judge on val AUC vs v4:
  it is biased against any target that changes silent cells (Tucker POSTS:194-195; Tian trap 39).

## 6. Risks

- Own analog is gold-selected; by-site OOF is cross-site, so the real teacher is weaker (F0 / gate T catch it).
- Target -> student transfer is noisy (our crop: mixSA target +0.035 gave gold58 -0.0095, HANDOFF 21:50; fleet mix50 vs v4: gold58 +0.045
  at epoch 4, HANDOFF D1 provisional).
- V100 contention: the teacher folds and the DSD fold-0 run displace ENDGAME items (EV2S fold-0 or one all-data seed). Total ~14-18 V100-h,
  of which ~7.5-12 h replace a planned fleet run. crop fold ids (crop_f4 = English) must be checked in the checkpoint metadata first.
- T4 per commit (0.5-0.7 h) is a GUESS from ENDGAME's KT estimate.
- One solo LB read has SD ~.0047 and one blend read ~.0012 (ENDGAME_V3 sec 0.5): a +0.001..+0.002 label effect is not LB-readable;
  the decision rests on target-level + fold-0 rho. Effective rho may be even closer to 1 than +0.10.
- n = 58 everywhere; per-label gains rest on 4-14 positives in silent cells.

## 7. Runner-ups

1. **Online ELR / soft bootstrapping inside the fleet** (target <- 0.7 label + 0.3 EMA-model prediction from epoch 3): attacks memorisation of the
   35% report-YES false positives (gold curves peak then drift: Tian 16 ep -0.012 OOF, LEDGER:46). 0 extra GPU; own-family so no rho import.
   Evidence thin: Nicolai "lift on gold+CV, 0 on LB" (CLAIMED, LEDGER:69). GUESS solo +0.000..+0.003.
2. **Severity-aware re-read of report-YES cells** (mild/small/partial/sprain/possible -> downweight): text-only, decorrelated from every image
   model; within-YES v4 AUC .811 vs perfect +0.044 target. Needs an LLM pass (Qwen-7B on the V100 ~2 h) or multilingual regex. Tucker:
   better extraction did not move LB (CLAIMED, POSTS:183/192). GUESS +0.000..+0.004 solo.
3. **mixSA / mix50 as D1 already plans**: the safe default if F0 or gate T fails (+0.009 solo LB, Tian x2, TIAN:83/88; rho cost measured).

Dead for this angle (measured): per-label calibration and constant-T noise correction (AUC-invariant), site debiasing (-0.008/-0.019),
flat OOF self-distillation (Tian -0.001 / 0.000), replacement (-0.0197 target, -0.0064 NTejas), masking/rounding (LEDGER:67/72), multi-round
OOF (v09o null), per-label alpha tuning on gold58 (overfit; Shiven per-target -0.0008, LEDGER:91).
