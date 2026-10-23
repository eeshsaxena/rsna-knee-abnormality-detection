# Angle: contrarian / unusual levers (2026-10-04 ~05:00 IST)

Research only. Nothing was submitted, pushed, versioned, posted or run on the box. All compute below was laptop CPU
(seconds to minutes) on files already in the scratchpad. Scripts and outputs are in this folder.

Tags: **MEASURED** = our ledger, LB, or a number computed here today. **CLAIMED** = forum/paper statement.
**MODEL** = equal-variance binormal blend model (same family as the brief's closed-form numbers). **GUESS** = my estimate.

`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`

---

## 0. Bottom line (honest)

1. **No lever in this angle credibly delivers +0.007 on its own.** P(any contrarian lever adds >= +0.007 private by
   10-12) < 5% (GUESS). Four of the eight listed levers fail cheap gold58 tests I ran today (Section 2); three more
   are dead or blocked by the ledger.
2. **The binormal model says the jump is set by own-member solo strength (S0), not by labels or post-hoc tricks.**
   Across every scenario I ran, the label choice (pure v4 / gated / mix50) moves the blend lift by <= 0.002, while
   moving S0 from 0.92 to 0.93 moves it by +0.002..+0.004 (Section 3). +0.007..+0.015 needs an own ensemble at
   >= ~0.935-0.945 solo with rho <= ~0.82. That is a member-count / diversity / merge problem.
3. **Best evidence-to-cost survivor: "report-gated teacher fill"** (privileged information: the report decides WHERE
   the image teacher speaks). Zero new code, label file already built: `SP/bigjump/labels_gated75_logit.csv`.
   Use it as the target for ONE extra fleet seed (s43), not as a separate programme. Expected +0.0005..+0.0035 private
   over a pure-v4 member of the same S0 (MODEL), roughly a tie with mix50. It is a D1 hedge, not a jump.
4. **Useful by-product for D1 (MODEL):** mix50 beats pure v4 in the blend only if its LB solo is >= ~0.008-0.010
   higher (rho .857 vs .80). If the fleetsolo canaries show a smaller gap, pick pure v4 (or gated).

---

## 1. The survivor: report-gated teacher fill

### Mechanism
- 27% of label cells are report-silent (pilkwang `__verdict == UNK`): Synovitis 84%, Fracture 56%, Baker's 46%,
  Lateral OA 33%, Medial OA 26%, Contusion 21%, PF OA 18% (MEASURED, train-wide, `gated_perlabel_and_blend.py`).
  There the LLM target is a near-constant fill. Tian measured that inside silent cells the LLM ranks gold positives
  at the 0.17 (Fracture) / 0.48 (Baker's) / 0.75 (Synovitis) percentile vs Raptor 0.74 / 0.70 / 0.82
  (MEASURED, `SP/knee/tian/proposals.md` P-62).
- Flat teacher mixing (mix50) fixes those cells AND drags every report-addressed cell toward the parent family,
  which is what raises student rho (fleet D1 epoch 4: mix50 rho vs Raptor proxy .857 vs pure v4 .796, MEASURED, HANDOFF).
- Gated fill: target = steven_v4 on report-addressed cells (teacher weight 0), 0.25 v4 + 0.75 teacher on silent cells.
  Teacher = `SP/teacher_v2_out/raptor_proxy_v5v8.csv`, logit mean/std matched onto v4 per label.

### Evidence
| Quantity | v4 (pure LLM) | gated .75/0 | mix50 (flat .5) | Tag / source |
|---|---|---|---|---|
| gold58 target macro AUC | 0.8927 | 0.9109-0.9111 | 0.9314 | MEASURED today, `build_gated.py`, `gated_perlabel_and_blend.py` |
| paired bootstrap vs v4 | | +0.0189 (SD .0061, P>0 1.00) | +0.0404 (SD .0087) | MEASURED, `silentfill_target_gold58.py` |
| gold58 within-class Spearman vs teacher | 0.443 | 0.503-0.513 | 0.850-0.853 | MEASURED |
| train-wide (n=4333) per-label Spearman vs Raptor proxy | 0.652 | 0.724 | 0.867 (`labels_mix50_logit.csv`) | MEASURED |
| same vs real v8 arm | 0.625 | 0.690 | 0.834 | MEASURED |

- Per-label gated gains (gold58, target level): Fracture +0.114, Synovitis +0.038, Baker's +0.024, PF OA +0.018,
  Lat Meniscus +0.011, Contusion +0.008, Lateral OA +0.007, rest ~0 (MEASURED). These are the parent-proxy's weak
  labels on gold58 (Synovitis .830, Lat OA .832, PF OA .834, Lat Men .870; MEASURED `perlabel_blend_gold58.py`).
- Own-family fill as a control: the v4 crop ensemble on silent cells only gives +0.0141 at rho .465 (MEASURED,
  optimistic: crop checkpoints were selected on gold58).
- CLAIMED support: Archit (0.954) filled silent cells with OOF predictions, "moved right away" (POSTS:294-295);
  Tom (0.949): masking silent negatives did nothing, "the model needs more positives" (POSTS:655).
- AGAINST (MEASURED): target gains shrink ~3x into students (Tian +0.032 target -> +0.009 LB, LEDGER:52); our mixSA
  crop members gained nothing on gold58 and read worse blended (mixsacrop25 0.940 vs crop25 0.941); NTejas silent-cell
  replacement -0.0064 gold58 (within floor); the teacher's epochs were chosen on gold58, which flatters every
  teacher-leaning row by an unknown amount; at our flat-vs-silence control the "silence" effect alone was +0.0009.

### Expected numbers (MODEL + GUESS)
- Student solo: +0.005 (transfer 0.28 of +0.018) to +0.009 (transfer 0.5) over pure v4. Student rho: ~.816 (linear
  map of target rho .652/.724/.867 onto measured student rho .796/?/.857). GUESS.
- Blend lift over 0.943 (`gated_blend_perlabel_model.py`, per-label binormal, flat and gold-like parent spreads):

| S0 (pure-v4 solo) | transfer | pure v4 | gated | mix50 |
|---|---|---|---|---|
| 0.92 | 0.28 | +0.0020..0.0036 | +0.0025..0.0044 | +0.0023..0.0040 |
| 0.92 | 0.50 | +0.0020..0.0036 | +0.0038..0.0060 | +0.0053..0.0070 |
| 0.93 | 0.28 | +0.0043..0.0062 | +0.0054..0.0076 | +0.0057..0.0076 |
| 0.93 | 0.50 | +0.0043..0.0062 | +0.0073..0.0098 | +0.0104..0.0120 |

  Read: gated beats pure v4 by +0.0005..+0.0035 and ties or trails mix50. It is the right pick only if D1 shows mix50's
  realised solo gain is small (crop evidence) while its rho cost is real.

### Exact recipe
- Labels: `SP/bigjump/labels_gated75_logit.csv` (4407 x 12, built today by `SP/bigjump/build_gated.py`; args
  `[teacher] [out] 0.75 0.0`). Missing pilkwang rows count as silent (same convention as `build_mixSA.py`).
- Data/model/hparams: identical to `f0_s42` (fleet store 4407 npz, 4 views x 24 triplets, 320 px, 140 mm, ConvNeXt-T
  contract, AdamW bb 1e-4 / head 1e-3, wd .05, EMA .999, fp16, 10 epochs, `folds5.csv` fold 0, gold74 excluded,
  `--eval_labels` = steven_v4). Seed 43, so it doubles as a planned seed member.
- V100: fold-0 ~6-7 h alone (35-45 min/epoch, ESTIMATE in WIN_PLAYBOOK) or ~11-12 h shared; epoch-5 read at 3.5-6 h.
  If kept, the all-data run replaces a planned F1 seed (net ~0 extra V100 h).
- Kaggle T4: 0 extra. Same fleet cell, one member file swapped; in-kernel cost unchanged (~3-4.5 min/member,
  ESTIMATE `endgame3/compute.md`). Fits the >= ~2.5 h window after the <= ~6 h parent.

### Risks
- Transfer ~0 (as for the mixSA crop members): then gated = pure v4 plus a little rho; loss <= 0.0005 (MODEL).
- Raptor proxy is the parent's own arm family; on Synovitis (84% silent) the column becomes mostly teacher-driven,
  so per-label rho there rises more than the average.
- Fold-0 solo differences of ~0.005 are below the 0.008 OOF floor (Tian), so solo is NOT resolvable offline.

### First falsifier (< 24 h)
At epoch 5 of the s43 gated run vs `f0_s42` (mix50) and `f0_s42_v4` at epoch 5, on the 867 fold-0 val studies:
- KILL if mean per-label Spearman(OOF, Raptor proxy) >= mix50-run minus 0.02 (no decorrelation advantage), or if
  val AUC vs v4 < v4-run minus 0.005 (gating broke the addressed cells).
- KEEP if that Spearman <= v4-run + 0.03 and gold58 (direction only, floor ~0.05) is >= the v4 run.
- The real test is a fleetsolo LB canary later; offline only rho is resolvable.

---

## 2. Measured today: contrarian levers that fail (CPU, gold58 referee, no gold used for fitting)

| Lever | Test | Result | Verdict |
|---|---|---|---|
| Site / metadata debiasing (report language = site proxy) | per-site logit offset of v4 vs 3 image teachers, fitted on train only, removed on gold58 | 0.8927 -> 0.8862..0.8870 (d -0.006, SD .004, P>0 .06-.11); shuffled-site control -0.0076 | DEAD: cross-site gaps are mostly real prevalence, not reporting bias. Zhukov meta-only 0.598 (CLAIMED, memory L89) |
| Cross-label consistency on the parent | Raptor proxy + 0.3 x sibling rank (Syno<-Effusion/Baker, LatOA<-MedOA/PFOA/LatMen, PFOA<-MedOA/LatOA, Cont<-ACL/Frac, Frac<-Cont, LatMen<-LatOA/ACL) | per-label -0.016..+0.016, no consistent sign | DEAD (3rd null: box_xlabel_rerank +0.0002; 0.936 notebook "clinical residual" +0.001 stated) |
| Report severity (privileged text) | regex mild/hedge words within 60 chars of the finding term, 9 languages | hedged YES less gold-positive for Effusion .52 vs .88, Synovitis .67 vs .88, Baker's .57 vs .78, PF OA .62 vs .75; MORE for Contusion, Med OA, Fracture; a -0.25 shift loses AUC on 6/8 labels; parent proxy already ranks hedged Effusion/Synovitis lower (pct .44 vs .69, .55 vs .79) | WEAK/NEGATIVE as built; an LLM severity pass is untested but would raise rho |
| Per-label "headroom mask" for own members (W only on parent-weak labels) | gold58 crop analog; per-label binormal | gold58 +0.0020 -> +0.0037 (gold-selected, optimistic); MODEL +0.0008..+0.0017 over flat if parent spread is gold-like, -0.0025..-0.006 if flat; Tian P-53: gold per-label weakness was gold-specific on public | NOT A LEVER |

Ledger kills (no new test): laterality normalisation (conflicting MEASURED: Tian +0.0147 OOF for it, aastik
canonicalised -0.006 LB vs mirror aug); TTA / AdaBN (ConvNeXt has no BN; slice-offset TTA +0.0016 on a blend, Tian;
geometric TTA rejected); stacking (parent in-sample on all train, only 58 clean studies); hard-negative mining
(the hard negatives are report-YES false positives, e.g. Contusion YES 49% gold-positive: upweighting learns noise);
per-target heads (wash, memory L81).

---

## 3. Runner-ups (brief)

1. **Per-label public probe of the parent (Tian P-53 method).** Add `submission_probe.csv` = parent with 6 labels at
   0.5 to an already-planned commit: 0 T4, 1 slot, group mean resolved to ~+-0.001. MEASURED precedent (Tian #28).
   It only tells where public headroom is; required before spending V100 on any specialist.
2. **Fracture / Synovitis specialist on gated targets** (the two labels with the largest gated target gains), only if
   the probe shows the group <= 0.92 on public. MODEL +0.001..+0.003; cost ~6-8 V100 h + one inference stage.
3. **Archit's exact recipe with OWN-family OOF** (silent cells filled from 5-fold fleet OOF): target-level analog with
   own crop members +0.014 at rho .465 (most decorrelated option measured), but needs 5-fold OOF (~30+ V100 h), so
   not before ~10-08; only if the merge or a late slot needs one more decorrelated member.

---

## 4. Files written (all in `SP/bigjump/`)
- `site_bias_check.py`, `site_debias_gold58.py`: site / metadata test.
- `xlabel_gold58.py`: cross-label test on the parent proxy.
- `severity_gold58.py`: report-severity test.
- `perlabel_blend_gold58.py`, `masked_blend_model.py`: per-label headroom and masking model.
- `silentfill_target_gold58.py`, `gated_perlabel_and_blend.py`, `gated_blend_perlabel_model.py`: gated-fill evidence and blend model.
- `build_gated.py` -> `labels_gated75_logit.csv`: the recipe's label table (4407 x 12).
