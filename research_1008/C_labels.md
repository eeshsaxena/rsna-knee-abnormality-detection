# Dimension C: label levers (research_1008, written 2026-10-08 ~21:00-22:30 IST)

Read-only research. No Kaggle write, no API call, no GPU, no Vast. Box use: `ls`, `grep`, `md5sum`, `cat` and an `scp`
read of 22 small gold-prediction CSVs; nothing written on the box. All analysis ran on the laptop (light Python, about
2 min CPU in total).

Tags: **MEASURED** (script + output file named), **MODEL** (arithmetic on measured inputs), **GUESS** (judgement),
**CLAIMED** (forum).

Scripts and outputs are in `C:/Users/eeshs/Downloads/rsna-knee/research_1008/C/`:

| file | what |
|---|---|
| `target_menu.py` -> `target_menu.out`, `target_menu.json` | 14 candidate targets per label vs expert gold58, paired bootstrap, Efron optimism, Synovitis x Effusion |
| `transfer.py` -> `transfer.out`, `transfer.json` | measured target -> student transfer from 7 student pairs |
| `epoch_opt.py` -> `epoch_opt.out` | gold58 epoch-selection optimism from 12 of our own training logs |
| `build_k11.py` -> `build_k11.out`, `labels_k11_frac75_pf25_logit.csv` | the proposed K11 label file |
| `box_preds/` | read-only copies of the box gold74 prediction CSVs used by `transfer.py` |

**Ops note for the main session:** from ~20:50 to ~22:00 IST, ssh from the laptop to the box timed out about half the
time (port 22 unreachable). When it answered, `uptime` was 8 h 28 min, so the box had NOT rebooted. Laptop-side
watchers that poll the box can miss events during these gaps.

---

## 0. Bottom line

1. **Label column surgery cannot move the public display.** The measured transfer is about **1.0 point of per-label
   target AUC -> +.005 SOLO LB -> +.0023 in a W .5 blend -> +.0015 in the K9N 3-leg blend** (MODEL on MEASURED inputs,
   sec 1). Even the oracle "best recipe per label" menu (+.0164 apparent macro target gain, sec 3) is worth at most
   ~+.001 SOLO before the selection correction, and ~0 after it.
2. **Picking per-label mixes on gold58 is mostly winner's curse.** For the per-label teacher weight on a 5-point grid,
   the apparent macro target gain is +.0086 and the Efron optimism is .0106, so the corrected gain is **-.0020**
   (MEASURED, `target_menu.out`). Only Fracture (+.035 corrected) and Contusion (+.018 corrected) survive one at a time.
   PF OA's text advantage (+.0135) is fully explained by optimism (corrected -.0004).
3. **The label lever that ever moved our LB was wholesale, not per-column.** The v4 -> mix50 swap (all 12 columns,
   hard -> soft, image teacher filling report-silent cells) gave +.016 SOLO LB (MEASURED). In those swaps the
   per-label student change does not follow the per-label target change: Spearman(dT, dS) -.50 to +.06 across labels,
   slope ~0 once a common intercept of +.011 to +.019 is fitted (sec 1.2). frac75, the only clean single-column test,
   behaved like the local model (5-fold set macro gold58 -.0008, SE .0028), not the global one (+.0053 predicted).
4. **frac75 (done) is real but tiny on the LB.** At set level, a +.0458 Fracture target gain became a +.0083 student
   Fracture gain (ratio .18). In K9 (5 of 10 members frac75) the MODEL value is about +.0001 SOLO and +.00005 blended.
   No submission can isolate it.
5. **Recommendation for dimension C:** spend almost nothing more here.
   - Read the VLM pilot with its pre-registered gate (sunk cost).
   - Do NOT pay for a VLM scale-up unless S >= +.03; even then the EV is +.0001 to +.0005 blended.
   - Use the proposed **K11** (frac75 + PF OA .75 text / .25 teacher, file built) only as a free label twist on a 224
     reseed that the team would train anyway.
   - Points toward the top must come from student strength and decorrelation (N/OAI, 320, other dimensions). The one
     label-side input there is the OAI label mapping (sec 8).

---

## 1. Measured transfer chain: target gold -> student gold -> LB

### 1.1 frac75, the one clean single-column experiment (MEASURED)

Target and student deltas are on gold58 (n 58, masked); the target deltas are from `target_menu.out` and the student
deltas from `transfer.out`.

| quantity | Fracture | 12-label macro |
|---|---|---|
| target frac75 - mix50 | **+.0458** (boot SE .0287, P>0 .96) | +.0038 |
| student, single fold: Kaggle E17 probe - J1 | +.0236 (SE .0315) -> ratio **.52** (SE .69) | +.0031 (SE .0053) |
| student, 5-fold set: D - A | +.0083 (SE .0154) -> ratio **.18** (SE .34) | **-.0008** (SE .0028) |
| fleet, 10 members: K9 - K8 (5 swapped) | +.0056 | +.0002 (SE .0021) |
| noise floor: SD of dS on the 11 unchanged labels | .0170 (single fold), .0104 (5-fold set) | |
| noise floor: seed-only null, set B4 s1044 - A4 s1043 | per-label SD .0120 | **-.0051** (SE .0035) |

- The single-fold probe looks better than the 5-fold set: probe Fracture .919 vs the D-set mean .912 on gold58. That
  points to probe luck (noise), not a schedule effect. **The set-level ratio is the one to use.**
- The seed-only null shifts the macro gold58 by .005. **A 4-5 member set comparison cannot resolve macro effects below
  ~.007 on gold58.**

### 1.2 All label-swap pairs on gold58 (MEASURED, `transfer.out`)

| pair (same fold 0 unless noted) | target macro dT | student macro dS (SE) | WLS per-label slope through 0 | with intercept: a / b | Spearman(dT, dS) over labels |
|---|---|---|---|---|---|
| TTR mix50 (J1) - TTR v4 (s1042) | +.0335 | **+.0398** (.0102) | +.13 (SE .10) | +.011 / -.07 | +.06 |
| cnxt s42 mix50 - s42 v4 (labels only) | +.0335 | **+.0533** (.0124) | +.27 (SE .16) | +.015 / +.04 | -.11 |
| cnxt s42 gated75 - s42 v4 | +.0181 | +.0235 (.0096) | -.08 | +.019 / -.27 | -.50 |
| cnxt s42 mix50 - s42 gated75 | +.0153 | +.0298 (.0075) | +.19 | +.017 / -.03 | -.15 |
| frac75 probe - J1 (Fracture only) | +.0038 | +.0031 (.0053) | ratio .52 | | |
| set D - set A (Fracture only) | +.0038 | -.0008 (.0028) | ratio .18 | | |

- **Pooled per-label WLS slope over all 37 changed label-pairs: +.144 (SE .068).**
- **Two regimes:**
  - *Wholesale swaps* (all columns, hard -> soft, new image information in report-silent cells) move the student
    macro by **1.2-1.9x** the target macro. The gain is a common intercept, not a per-label effect: Synovitis dT +.033
    became dS +.18, and Contusion dT +.046 became dS -.01.
  - *Column surgery* (frac75) moves the student by **~.15-.2x** the per-label target change, and leaves the macro
    within noise.
- This matches the earlier pseudolabel track (`SP/jump2/pseudolabel.md` sec 3.1): within a recipe, target macro gains
  transferred at 1.0-1.6x, but own-OOF self-distillation transferred to **zero** on the LB, twice.

### 1.3 Student gold58 macro -> public LB SOLO, kappa (MEASURED anchors)

| change | student gold58 macro | public LB SOLO | kappa |
|---|---|---|---|
| TTR fold 0: v4 labels -> mix50 labels (K3 vs K4i SOLO) | +.0398 | .923 -> .939 = **+.016** | **.40** |
| J1 alone -> 5-fold set A + TTA (K4i SOLO vs K5 SOLO) | +.0089 (pass A) + ~.0015 (TTA, J1 tta_bench) | .939 -> .943 = +.004 | **.38** |
| Tian v09a -> v09r/v09u (0.5 LLM + 0.5 Raptor teacher) | +.0171 / +.0073 | .918 -> .927 = +.009 | ~.74 |
| Tian v09t (own 5-fold OOF mixed into the target) | +.0087 | .918 -> .917 = -.001 | ~0 |

- **kappa ~ .4** for new-information changes; ~0 for self-distillation.
- gold58 differences overstate LB differences by ~2.5x. That fits a noisy 58-study sample and the selection on it.

### 1.4 SOLO -> blend pass-through (MODEL, binormal)

- The blend d' derivative with respect to the fleet leg is W / sqrt((1-W)^2 + W^2 + 2W(1-W)r).
  - W .5, r .86-.94: .50-.52. Times the AUC-density ratio ~.9, that gives **~.45** for K9-type blends.
  - K9N-a (P .10 / F .30 / N .60, pairwise r ~.9): **~.31**.
- **Conversion used below:** per-label target dT
  - -> student .15 dT (sec 1.2)
  - -> student macro .0125 dT
  - -> SOLO LB **.005 dT** (kappa .4)
  - -> K9 blend **.0023 dT**
  - -> K9N blend **.0015 dT**.
- Under the wholesale model the factor would be ~10x larger, but frac75 rejects that model for column changes at ~2 SE
  (sec 0.3).

---

## 2. In-sample selection bias on gold58: size of the shrinkage

### 2.1 The three sources

| source | size | evidence |
|---|---|---|
| A. Choosing the per-label mix on the same gold58 (winner's curse over labels x recipes) | **large**: macro optimism .0106 on a 5-point grid, more on the full 14-recipe menu | MEASURED, Efron bootstrap B 2000, `target_menu.out` |
| B. The Dread teacher was epoch-selected on gold58 (AUDIT 10-04 l.100) | **small**: our own runs picking the epoch on gold58 instead of val gain **+.0022 macro** on average (SD .0016, max .0052, 12 runs); per label the average is the same, larger on volatile labels | MEASURED analog, `epoch_opt.out` |
| C. Raptor proxy and parent arms (checkpoints "top-3 by gold58", possibly trained on gold58) | **unknown, likely large** | HANDOFF l.838; cannot be measured |

- **Also: gold16 is NOT expert gold.** The 16 extra gold74 studies were labelled by reading reports (HANDOFF l.786;
  `manual_verified*`). They favour text-direction moves, e.g. PF OA text .924 on gold74 vs .902 on gold58.
- So text-direction decisions must be judged on **gold58 only**. Teacher-direction moves look worse on gold16 for the
  same reason, e.g. Contusion w 1.0 gold16 .679 vs .946.

### 2.2 Efron optimism per label (MEASURED; best w from {0, .25, .5, .75, 1}; w = teacher weight)

| label | best w | apparent gain vs mix50 | optimism | corrected | shrink |
|---|---|---|---|---|---|
| Fracture | 1.0 | +.0486 | .0136 | **+.0350** | .72 |
| Contusion | 1.0 | +.0283 | .0107 | **+.0177** | .62 |
| PF OA | 0 (text) | +.0135 | .0139 | -.0004 | ~0 |
| Medial OA | .75 | +.0062 | .0073 | -.0011 | <0 |
| Lateral OA | .75 | +.0058 | .0231 | -.0173 | <0 |
| Effusion | .75 | +.0012 | .0058 | -.0046 | <0 |
| Synovitis | .5 | 0 | .0170 | -.0170 | n/a |
| ACL, MCL, MedMen, LatMen, Baker's | .25-.5 | 0 | .003-.010 | <0 | n/a |
| **macro** | | **+.0086** | **.0106** | **-.0020** | **<0** |

- Source B lowers the teacher-direction rows (Fracture, Contusion) by roughly another .002-.005 (MODEL).
- **Rule:** per-label targets fitted on gold58 have negative expected out-of-sample value as a package. A
  label-specific change needs a prior reason plus a student-level test, as frac75 had.

---

## 3. Ranked per-label target fixes

Gains are on gold58, measured vs mix50. "Student" and "LB" use the sec 1.4 conversion (MODEL). For K9N, divide the
SOLO column by ~3.3.

| rank | label | best defensible recipe (vs mix50) | apparent target gain (boot SE, P>0) | shrunk target gain | expected student per-label gain | SOLO LB (full fleet) | verdict |
|---|---|---|---|---|---|---|---|
| 1 | **Fracture** | frac75, w .75 (**DONE**, in K9/K10) | +.0458 (.0287, .96) | ~+.03 | +.007 (measured set: +.0083) | +.0002 | keep; nothing more to do |
| 1b | Fracture, next step | w 1.0 (+.0028 over frac75), gated s100/a50 (+.011 over frac75), Raptor mixes (in-sample) | +.003 to +.011 over frac75 | ~0 | <+.002 | <+.00005 | **skip** |
| 2 | **Contusion** | w .75 | +.0148 (.0176, .80); w 1.0 +.0283, corrected +.0177 | ~+.007 | +.001 | +.00004 | **skip**: the student already beats every target (gold58 .958 vs target .906) and it raises Raptor/N-likeness (.866 -> .901) |
| 3 | **PF OA** | w .25 (.75 text / .25 teacher) | +.0077 (.0105, .76) | ~0 to +.004 | +.0003 to +.0006 | ~+.00002 | **K11 only**, as a free twist; it also lowers Raptor-likeness .916 -> .882 |
| 3b | PF OA alternative | gated: teacher .5 on report-silent cells, pure v4 on addressed cells | +.0199 (.019, .85) | ~0 to +.006 | | | **skip**: 82% of cells become coarse hard v4 values (0.25/0.975), which loses the softness that drives the wholesale gain; Spearman vs mix50 .919 (big shift) |
| 4 | **Lateral OA** | w .75 | +.0058 (.0126, .66); 11 positives | -.017 | ~0 | 0 | **skip**: weak for every source (.83-.86), including the parent side (.846) |
| 5 | **Effusion** | none | at the ceiling: target .984, student .995 | 0 | 0 | 0 | **skip** |
| 6 | **Synovitis** | none survives: best is v4 .5 + Raptor .5 at +.018 (.026, .76), but Raptor is in-sample and corrected it is -.017; gated75 +.013 | | ~0 or < 0 | | 0 | **skip** re-mixing. Effusion as a Synovitis proxy HURTS: Eff target alone -.081, rank .5/.5 -.031, rank .75/.25 -.010 (gold58 crosstab Eff x Syn: 22 both, 13 Eff only, 5 Syn only, 18 neither). Also the student trails its own Syn target (-.03), so it is student-limited too. |
| 7 | MCL / LatMen / MedOA / MedMen / ACL / Baker's | small Raptor or gating moves | +.002 to +.014 apparent | <0 corrected | | 0 | skip |
| 8 | **VLM relabel** (pilot running, sec 4) | .5 logit mix50 + .5 logit img on 6 labels | S unknown (prereg bar +.010) | posterior ~.64 x S | .15 x 6S (summed over labels) | **+.03 S**: S .01 -> +.0003; S .03 -> +.0009 | read the pilot; scale-up NOT recommended unless S >= .03 |

**One-line read:** every per-label fix beyond frac75 is worth <= +.0001 SOLO LB by the measured chain. The only
label-side item with more than ~+.0005 potential is a new-information, wholesale label source (sec 8), not a re-mix.

---

## 4. The VLM relabel pilot (kit `~/gold_prep/relabel`, prereg_v2; Eesh approved <= $8)

- **What I could see:** the old scratchpad `gold_prep/relabel/results/` holds only `sample_request_{img,text}.json`
  (20:36-20:43 IST). There is no fetched batch output yet. I called no API.
- **Its gate is clean.** The prompt, model, image set and eval set were fixed before any call, and the VLM never saw
  gold labels. S is therefore unbiased.
- The GO decision (S >= +.010 at SE ~.0075) adds a mild winner's curse. With a prior centred ~+.005 (SD .01, GUESS),
  the posterior factor is ~.64: an observed S of .015 implies about .011 true.
- **Value if scaled (MODEL):** the change is column-level (6 labels, ~1,806 uncertain studies), so the local model
  applies.

  | S | SOLO LB | K9 blend | K9N blend |
  |---|---|---|---|
  | +.01 | +.0003 | +.00014 | +.0001 |
  | +.03 | +.0009 | +.0004 | +.0003 |

  If the wholesale model applied (it did not for frac75), it would be ~10x those numbers.
- **Cost of a scale-up:** ~$15-51 at Opus batch for the uncertain subset (kit estimate; needs a NEW Eesh approval
  beyond the $8 pilot cap), 1-2 days, ~27 T4 quota hours for 5 folds at 224, and a kit change.
- **Recommendation:**
  1. Run `pilot_eval.py --prereg prereg_v2.json` when the batches land, and log the verdict.
  2. Scale up only if **S >= +.03 AND the secondary (img - text over 5 non-Effusion labels) >= +.02** (sign of real
     image reading). This is my advice, not a change to the prereg.
  3. Otherwise record NO-GO and close the relabel line.

---

## 5. K11: one concrete label set (built, not used anywhere yet)

**Recipe (exact):** K11 = frac75 with **only** the PF OA column replaced by

    PF OA = 0.75 x steven_v4 + 0.25 x logit_match(Dread teacher, steven_v4)

- This is the same prob-space mix and logit-match method as mix50/frac75 (`build_frac75.py`).
- The logit-matched teacher half is recovered exactly as `tm = 2 x mix50 - v4`. The frac75 Fracture column equals
  .25 v4 + .75 tm to < 1e-9, which proves the recovery.
- Every other column is byte-identical to frac75:

  | column | recipe |
  |---|---|
  | Fracture | .25 v4 + .75 tm |
  | PF OA | .75 v4 + .25 tm |
  | the other 10 columns | .5 v4 + .5 tm (mix50) |

**File (MEASURED, `build_k11.out`):**
- `research_1008/C/labels_k11_frac75_pf25_logit.csv`, md5 **e7371c1ae1068dc7c6b0d991762a1316**, 4,407 x 12, CRLF.
- 11 columns byte-identical to frac75 (ad7582b8).

| | mix50 | frac75 | K11 |
|---|---|---|---|
| gold58 target macro | .9262 | .9300 | **.9307** |
| gold58 PF OA target | .8880 | .8880 | **.8958** |
| PF OA Spearman vs Raptor proxy (train pool) | .916 | .916 | **.882** |

- Training pool (4,333 studies): PF OA mean target .467 -> .474; share >= .5 .452 -> .456; Spearman vs mix50 .992. It
  is a small shift that stays continuous (4,333 distinct values).

**Why this one and not the alternatives:**
- It is the only change that is (a) not worse on gold58, (b) less like the parent/N family on the changed column, and
  (c) keeps soft targets.
- **K11c** (+ Contusion w .75) is rejected: N-likeness, and the student is already above the target.
- **K11g** (gated PF OA) is rejected: hard targets.
- **K11R** (Raptor co-teacher on all columns) is rejected:
  - N is a Raptor-recipe CoAtNet, so this makes the fleet more N-like exactly where the K9N blend needs it different;
  - its gold58 is in-sample.

**Expected value (MODEL):**
- Label effect: ~+.0001 SOLO LB (sec 3, row 3), unresolvable.
- Its real role: if the team trains a second 224 set anyway, train it on K11 instead of plain frac75.
  - Same cost and same reseed value (+.0004-.0006 SOLO, ~+.0002 blended, MEASURED analog for a second seed).
  - Plus a free label option and slightly lower within-fleet correlation on PF OA.
- **It is not worth T4 hours on its own.**

---

## 6. Pre-registered gate for K11 (write it before launch; bars must not move after)

**Design:** paired with the existing frac75 Kaggle runs (same seed, same platform, same epoch count, same folds). The
only difference is the PF OA column. Score with `~/novel/gold_infer5.py --passes A` on the box CPU, masked convention,
**gold58 only** (gold16 is report-read and favours text-direction moves).

| gate | measured on | bar | rationale |
|---|---|---|---|
| **G1 no harm** | per fold, raw-best OOF vs v4, mean over the 11 non-PF-OA labels | >= the paired frac75 run's same quantity - .003 | fold-0 seed noise .0028 (MEASURED) |
| **G2 sign** | 5-member gold58 PF OA AUC, K11 set - D set | >= -.010 | null SD of a 5-member per-label delta .0104 (MEASURED) |
| **G3 guard** | 5-member gold58 12-label macro, K11 set - D set | >= -.005 | seed-only null macro -.0051 (SE .0035) (MEASURED) |
| report only | fold-0 second-seed replicate: K11 f0 s1044 - frac75 f0 s1044 (kmrsuman f4x); PF OA OOF vs v4 (biased toward K11 by construction) | | |

- **Power (MODEL):** expected PF OA student effect +.001 to +.004 vs a noise SD of ~.010, so P(dS > 0) ~.55-.65.
  **A PASS shows no harm, not a gain.**
- **Use rule:**
  - PASS -> eligible only as an ADDED 5-member set in the final kit, and only if the modelled slow-case runtime margin
    stays >= 0.75 h. A 224 set adds ~20 min hidden runtime (MODEL: 5 x ~0.35 s/study per T4, ~1,300 studies on
    2 T4s).
  - Never REPLACE set D with K11 on gold evidence.
  - FAIL -> drop K11 and close per-label target work.

---

## 7. Cheapest compute path (224 TTR on Kaggle T4, 2 folds per session)

**Three sessions, in parallel, from 10-10 05:30 IST (quota reset).** Each teammate pushes on their own account
(rule 3.5a). Each session is ~8.1-8.9 h + ~0.3 h save, about **27 quota hours in total**.

| session | trains | paired with |
|---|---|---|
| S1 | K11 f0 + f1, s1043, **E 17** | swapiy `rsna-train-ttr-frac75-f01` (E 17) |
| S2 | K11 f2 + f3, s1043, **E 21** | kmrsuman frac75 f23 (E 21) |
| S3 | K11 f4 s1043 + K11 f0 s1044, **E 22** | kmrsuman frac75 f4x (E 22) |

- **Pack:** clone `friend_frac75_pack` / `friend2_frac75b_pack`.
  - New labels dataset `eeshsaxena/rsna-knee-k11-labels` (~1.3 MB, the file above, sha256 asserted in cell 1).
  - New slugs `rsna-train-ttr-k11-{f01,f23,f4x}`.
  - Cap the self-sized epoch count at the paired E (a run that gets fewer epochs is reported with a schedule note, as
    frac75 had).
  - The same `--labels` swap is the only code-path change.
- **Keep >= 4 h on Eesh's account** for the K9 / K9N / final-kit commits. Do not start K11 on his account before those
  commits are done.
- **Gate:** box CPU, gold_infer5 pass A for 6 checkpoints (~10 min each at 8 threads, ~1 h), nice 19, no GPU. Then
  score G1-G3.
- **ETA:** sessions end ~14:00-16:00 IST 10-10, gate ~17:00 10-10, kit (if PASS and runtime allows) 10-11.
- **Priority:** below any T4 use with a higher modelled value (for example OAI-trained 224 members, if that dimension
  goes ahead). K11 only replaces a plain frac75 reseed.

---

## 8. Cross-reference: the one label lever with real potential is OAI's image-read expert labels

- The hidden test labels and gold58 look image-read and blind to the report (hengck23, CLAIMED). OAI's semi-quantitative
  MRI readings are image-read by experts. That makes them a *new-information, wholesale* label source, the regime where
  transfer was 1.2-1.9x (sec 1.2).
- The public N model gained +.003 from 2,399 OAI knees with masked labels (.942 -> .945, CLAIMED in 746792).
- If the team trains its own OAI members (Eesh downloads; data stays on the box), here is a label mapping to verify
  against the downloaded data dictionary (all **GUESS**, field names not seen):

  | OAI SQ reading | RSNA label | note |
  |---|---|---|
  | effusion-synovitis grade | Effusion | experts' "moderate or large" = positive (host 733343) |
  | Hoffa-synovitis | Synovitis | gold58: 22 of 27 Syn positives are also Eff positive |
  | bone marrow lesion size | Contusion | soft or masked: degenerative BMLs are not contusions |
  | cartilage + osteophytes per compartment | Medial OA / Lateral OA / PF OA | |
  | meniscal tear / morphology per horn | Medial / Lateral Meniscus | |
  | ACL integrity | ACL | |
  | popliteal cyst | Baker's | |
  | MCL, Fracture | mask | |

  Use masked BCE on unmapped cells and keep RSNA studies on mix50/frac75.
- **Caveat (GUESS):** OAI-trained members correlate with N, which is OAI-trained too. Value vs N in the K9N blend is
  uncertain. This belongs to whichever dimension owns OAI; it is listed here only for the label mapping.

---

## 9. Do not do (adds to "Settled")

| item | why | evidence |
|---|---|---|
| Per-label target mixes chosen on gold58 as a package | corrected macro gain -.0020 | MEASURED, Efron, `target_menu.out` |
| Effusion as a Synovitis proxy in the target | -.010 to -.081 on gold58 | MEASURED |
| Raptor co-teacher targets | in-sample gold58; raises N/parent-likeness | MEASURED rhoRap .94-.96 vs .87 |
| Judging text-direction label moves on gold74 | gold16 is report-read | HANDOFF l.786 |
| Expecting the LB to resolve frac75 or K11 | the modelled effect is below +.0002 | MODEL |
| Treating the frac75 single-fold probe ratio (.52) as the transfer rate | the 5-fold set says .18 | MEASURED |

---

## 10. Inputs (md5, all MEASURED)

| file | md5 | location |
|---|---|---|
| gold74_eval_set.csv | a0ff70f7 | identical on the box `~/rsna_train` |
| labels_mix50_logit.csv | 0ec64d32 | identical on the box |
| llm_labels_v4_blend.csv | 44874d8d | identical on the box |
| labels_frac75_logit.csv | ad7582b8 | |
| labels_gated75_logit.csv | a813b986 | identical on the box |
| labels_mixSA_logit.csv | 137d5e32 | |
| raptor_proxy_v5v8.csv | d76f2535 | |
| report_labels_v2.csv | fa9f175f | |

- Student gold predictions: the box `~/jump2/gold74_preds`, `~/novel/gold74_preds`, `~/novel/k9gold/preds` and
  `~/novel/frac75/gold74_preds`, copied to `research_1008/C/box_preds/`.
- Box tables read: `~/novel/out/{teacher_labels, label_files_gold, perlabel_update, perlabel_oof, gold_perlabel,
  member_gold_spread, loss_share, target_likeness, frac75_label}.txt`, `~/novel/frac75/{PREREG.md,
  out/gate_f0_s1043_ttr_frac75.txt}`, `~/novel/k9gold/k9_gold_read_20261008_1005.txt`.
