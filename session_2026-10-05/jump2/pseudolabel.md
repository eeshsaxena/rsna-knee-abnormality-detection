# Track: own-model pseudo-labels / self-distillation (jump2, 2026-10-05 ~09:40 IST)

Tags: **MEASURED** (source given) = an LB read, a log, or a number computed today from on-disk data. **CLAIMED** = forum
self-report. **LIT** = published paper. **MODEL** = arithmetic on measured inputs. **GUESS** = judgement.
`SP` = scratchpad, `J2` = `SP/jump2`. All box work was CPU-only (`CUDA_VISIBLE_DEVICES=`, nice 19, 2 threads) under `~/jump2/`.
Nothing was submitted, pushed or posted. No process was touched.

Scripts and outputs (laptop copies in `J2/pl_out/`):
- `gold_infer.py` (box `~/jump2/`): fold-0 checkpoints scored on the 74 gold studies, log `gold_infer.log`.
- `pl_sim.py` (section A) and `pl_sim_a2.py`: target and student parent-likeness on the 867 fold-0 val studies.
- `pl_sim_b2.py`: transfer table, 40 candidate targets on gold58, paired bootstrap.
- `pl_sim_b3.py`: own-OOF targets vs the plain v4 to mix50 line.
- `pl_sim_b4.py`: gate noise widths.
- `pl_sim_b5.py`: re-scores the 10-04 "DSD" plan against its own F0 kill rule.
- `build_pl_tm5.py`: the ready round-2 target builder plus gate T, dry-run tested.

---

## 0. Bottom line

1. **Own-OOF self-distillation does not give us a much stronger, less parent-like fleet.** The data split cleanly in two:
   - **v4-trained students are too weak to teach.** As a teacher they sit **.010 to .019 gold58 below** the plain "add a
     little Dread" line at equal parent-likeness. They are no better than the flat 0.5 self-distillation table (Tian
     `v09t`) that already read **LB -0.001** (MEASURED).
   - **mix50-trained students only re-teach mix50.** Their OOF is as strong as mix50 and as parent-like:
     - target gold58 .9267 to .9293 vs mix50 .9262 (+.001 to +.004, P .56 to .69);
     - Spearman to the Raptor proxy .869 to .874 vs .864.
2. **The precondition the forum winners rely on fails for us** (MEASURED, gold58).
   - Every one of our 5 students scores *below* the target it was trained on: **-.012 to -.032**.
     - v4: F1 .861, TTR .878, label .893.
     - mix50: F1 .914, TTR .913, label .926.
   - Archit, Scott and Raymond report OOF *above* their labels by ~.03 to .04 (CLAIMED).
   - Our v4 students beat the v4 label only in report-silent cells (within-cell AUC .70 vs .57) and report-NO cells
     (.66 vs .62). They lose in report-YES cells (.74 to .76 vs .765).
3. **Our own LB history already has two OOF-self-distillation nulls** (MEASURED, Tian ledger):
   - `v09t`: .917 vs .918;
   - `v09o` (own cross-fit OOF + Raptor): .927 vs .927.
   - Nicolai's 50/50 bootstrap is a third (CLAIMED).
   - Only a teacher carrying new information moved the LB: Raptor, +.009.
4. **The 10-04 "DSD" plan is killed by its own pre-registered F0 test** (MEASURED today, using the real fold-0 fleet as
   the own teacher):
   - DSD-own passes clause 1: +.0103 over v4, bar +.008.
   - DSD-combo fails clause 2: .9195 < mix50 .9262.
5. **Round 2 design (sec 4)** is built and tested, but it is **optional filler, not a lever.**
   - `labels_pl_tm5_v1` = steven_v4 mixed with the 5-fold TTR mix50 OOF at 0.75 silent / 0.5 NO / 0.5 YES.
   - It runs as one TTR fold-0 probe, 4.8 V100-h, after the chain ends.
   - Expected effect = another TTR mix50 member: true blend lift change **-.0002 to +.0004** (MODEL).
   - Run it only if the GPU would otherwise idle. It must never displace a labels-track student.
6. **Do not run TTR v4 folds 1-4 as a "clean self-teacher"** (blendmath sec 8, lever 2). Today's data say that teacher
   is below the frontier. Those folds can only be justified as low-r fleet members, which blendmath already prices at
   ~+.0002.
7. **Realistic gain from this track: public display +0.000 (range -.001 to +.002); P(>= .945) < .05** (MODEL/GUESS).
   - The pseudo-label idea becomes worth a round only *after* another track produces a student that beats its own
     labels on gold58: a better recipe, or better report labels.

---

## 1. Literature and forum: what makes pseudo-labels work, and the preconditions

**LIT (mechanisms):**
- Soft bootstrapping (Reed et al. 2015) sets target = b*label + (1-b)*prediction.
  - It helps when the model's predictions are more accurate than the noisy labels on the cells being mixed.
- Born-again networks (Furlanello et al. 2018) and self-distillation theory:
  - Mobahi et al. 2020: self-distillation amplifies regularisation, a few rounds help, more rounds underfit.
  - Allen-Zhu and Li 2020: self-distillation acts as implicit ensembling plus distillation.
  - So the gain is bounded by roughly what an ensemble of the same models would add.
- Students track their teachers closely (Stanton et al. 2021), so distillation also *raises* correlation with the teacher.
- Noisy Student ([Xie et al. 2020](https://arxiv.org/pdf/1911.04252)):
  - The gain comes from pseudo-labelling **extra unlabeled data** with an un-noised teacher and training an equal-or-larger
    **noised** student.
  - We have no extra data: the test set is hidden and external data is off by default (EXTERNAL_DATA_VERDICT).
  - So we are in the same-data relabelling regime, where gains are regularisation-sized.
- Medical noisy-label survey ([Karimi et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7484266/)): the same
  teacher/student and uncertainty-weighting toolbox; no free lunch without a better teacher.

**Forum (CLAIMED, thread 735304 unless noted; local copy `J2/verify/t735304_flat.txt`):**

| who (rank) | claim | the mechanism under it |
|---|---|---|
| Archit Konde (53) | 2.5D CoAtNet@224 single fold **.950**; OOF on gold58 **.930** > labels; mixing OOF into silent cells helped "right away", more than 50/50 helped more; masking did nothing; rounding hurt; "in-fold ones just parrot" | **His trigger: OOF already beat the labels on the 58.** Ours does not (sec 3.1) |
| Raymond Yuen (48) | .938 -> .940 -> **.949**, "still label"; **multiple weak-label sources (public, API, local) -> teacher models -> pseudo-labels -> final model**; "different teachers help different targets"; best label ~.89 gold, model ~.925 gold, LB .947 (thread 742327) | New information = **several independent label sources**, distilled through image models, not one model re-teaching itself |
| tennogh (5) | .942 raw / .943 TTA @288; "OOF pseudo-labels have been pretty well correlated with LB"; own labels not better than public | Ambiguous; no gain size given |
| Scott Willis (10) | Gemma-4 local labels ~.89 gold; single-fold single model .949, ~.930 gold | Student beats labels by ~.04 |
| Chris Deotte (122) | with pseudo-labels "every single model can match our best ensemble CV" | = the ensembling view of self-distillation |
| **Nicolai Karcher** (1429) | 50/50 soft bootstrap: clear lift on gold and CV, **0 on LB** | Same as our `v09t` |
| Tom Aindow (120) | masking or replacing with own strongest predictions: no effect | |
| Tucker Arrants (46) | student beats the extractor; repaired labels cannot be validated against; simple Qwen labels, models "low .95" | The gap to the top is the student, not labels |
| SpeedSci (126, thread 745214) | LLM-vote labels, then OOF/teacher refinement; one label version .945 gold58 -> **.910 LB** | gold58 overfits label work |

Reading:
- The claimed wins share one feature: the image student is already better than its labels where the labels are blind.
  Raymond's also comes from several label sources. That is new information.
- The nulls (Nicolai, Tian `v09t`/`v09o`) are same-source, same-family self-distillation.

## 2. Our own prior LB evidence (MEASURED, `SP/knee/tian/experiments.md` and `SP/round2/measured_levers.csv`)

| run | training target | student gold58 | public LB | vs control |
|---|---|---|---|---|
| `v09a` (CoAtNet-1 c02, all-data) | LLM blend | .8922 | .918 | control |
| `v09t` | 0.5 LLM + 0.5 own 5-fold OOF (quantile-matched) | .9009 (+.0087) | **.917** | **-.001** |
| `v09r` / `v09u` | 0.5 LLM + 0.5 Raptor (new info) | .9093 / .8995 | **.927 / .927** | **+.009** |
| `v09o` / `v09o2` | .25 LLM + .375 Raptor + .375 own cross-fit OOF | .9121 / .9104 | **.927 / .927** (pair .928) | **null** vs `v09r` (pair .930) |
| NTejas (readme) | OOF self-repair of silent cells (86% of Synovitis rewritten) | -.0064 gold58 | - | negative |

Gold58 gains of +.006 to +.009 from own-OOF targets transferred to **zero** LB, twice. Tian's verdict (traps 39) was:
OOF-vs-label metrics reward agreement with a smoothed copy of the labels, so judge only by gold58 direction and solo LB.

## 3. Today's data (MEASURED unless tagged)

Setup:
- gold74 are fold -1 in `folds5.csv`, so none falls in fold-0 val. Every run `--exclude`s them.
- The fold-0 checkpoints were therefore scored on the 74 gold studies (`gold_infer.py`, CPU fp32). This gives
  OOF-status predictions for gold58, the same status as the 867 val rows.
- These reproduce the trainers' logged gold58: F1 mix50 .9144, TTR mix50 .9132.
- Parent-likeness is measured two ways against the parent's Raptor proxy (`~/fleet/teacher_v2_raptorproxy_shard0.csv`):
  - `rho` = mean per-label Spearman on the 867;
  - `ws` = within-(label, report verdict) Spearman on the 867, which removes the report signal both sides share.

### 3.1 Transfer: what each target turned into (`pl_sim_b2.out` sec 1)

| student | trained on | target gold58 | student gold58 | student minus target | target rho / ws | student rho / ws | OOF AUC vs v4 |
|---|---|---|---|---|---|---|---|
| F1 v4 | v4 | .8927 | .8611 | **-.032** | .645 / .336 | .840 / .753 | .8692 |
| TTR v4 | v4 | .8927 | .8780 | **-.015** | .645 / .336 | .850 / .771 | .8764 |
| F1 gated75 | gated75 | .9109 | .8846 | **-.026** | .718 / .512 | .872 / .798 | .8734 |
| F1 mix50 | mix50 | .9262 | .9144 | **-.012** | .864 / .838 | .889 / .826 | .8811 |
| TTR mix50 | mix50 | .9262 | .9132 | **-.013** | .864 / .838 | .896 / .836 | .8836 |

- Raptor proxy itself: gold58 .9246. Two-member student ensembles:
  - v4 pair .8796;
  - mix pair .9191;
  - all four .9040 (`pl_sim_b3.out`).
- **Gains move with the teacher.** Target gold58 gains transfer to the student at ~1.0 to 1.6x within a recipe
  (v4 -> mix50: F1 +.053, TTR +.035; v4 -> gated: F1 +.024).
- **Parent-likeness barely follows the target.** Target ws -.33 (gated vs mix50) became student ws only -.028.
  - The student's parent-likeness is mostly set by being an image model that learned Dread's information, not by the
    exact mixing formula.

gold58 within-verdict AUC (cells pooled over labels, label-wise ranks; `pl_sim_b2.out` last block):

| cells | n | P(gold=1) | v4 label | mix50 label | TTR v4 | F1 v4 | Ev4 (2 v4 students) | TTR mix50 | F1 mix50 |
|---|---|---|---|---|---|---|---|---|---|
| report YES | 311 | .653 | **.765** | .804 | .759 | .743 | .766 | .805 | .811 |
| report silent (UNK) | 189 | .153 | .574 | .752 | .702 | .713 | .707 | **.830** | **.836** |
| report NO | 196 | .041 | .624 | .745 | .664 | .662 | .661 | .688 | .673 |

- Our v4 students add information only in silent and NO cells.
- The mix50 students beat even the mix50 label in silent cells (.83 vs .75). That is real image information, but it is
  Dread-derived.

### 3.2 Candidate targets (`pl_sim_b2.out` sec 2; 40 rows; `own` = fold-0 student logits, logit-matched to v4)

| target | gold58 | vs v4 (P>0) | vs v09t analog (P>0) | vs mix50 (P>0) | rho_rap | ws_rap |
|---|---|---|---|---|---|---|
| v4 | .8927 | - | - | -.033 (0) | .645 | .336 |
| gated75 (file) | .9109 | +.018 (1.0) | | -.015 (.04) | .718 | .512 |
| mix50 (file) | .9262 | +.033 (1.0) | +.017 (.99) | - | .864 | .838 |
| **Ev4 ALL .50 = `v09t` analog** | .9087 | +.016 (.96) | 0 | -.017 (.01) | .844 | .767 |
| Ev4 UNK .75 | .8993 | +.007 (.84) | -.010 (.12) | -.027 | .691 | .450 |
| Ev4 UNK+NO .75 (DSD-own) | .9029 | +.010 (.91) | -.006 (.19) | -.023 | .833 | .721 |
| Ev4 UNK .9 / NO .5 / YES .25 | .9099 | +.018 (.98) | +.001 (.65) | -.016 | .851 | .768 |
| Ev4 uncertainty-weighted (a 1.5, cap .9) | .9078 | +.015 (.98) | -.001 (.44) | -.018 | .783 | .664 |
| Eall (4 students) UNK .75 / NO .5 / YES .5 | .9210 | +.028 (1.0) | +.012 (1.0) | -.005 (.27) | .868 | .808 |
| **TTRmix UNK .75 / NO .5 / YES .5** | .9267 | +.034 (1.0) | +.018 (1.0) | **+.001 (.56)** | .869 | .813 |
| Emix (2 mix students) UNK .75 / NO .5 / YES .5 | .9293 | +.037 (1.0) | +.021 (1.0) | +.004 (.69) | .874 | .823 |
| mix50 with Ev4 on silent cells | .9198 | +.027 | +.011 | -.006 (.10) | .865 | .827 |

- **Per label, the v4-student teacher hurts where it is weak.** Ev4 fills (v4 -> Ev4 UNK .9/NO .5/YES .25):
  - Synovitis .790 -> .69 to .71;
  - MCL .968 -> .93 to .94.
- It helps elsewhere:
  - Baker's .944 -> .97 to .98;
  - Contusion .860 -> .92 to .94;
  - Fracture .793 -> .88 to .89.
- The v4 students' own per-label gold58 shows why: MCL .75 to .90 and Synovitis .62 to .63.
- Per-label teacher choice would have to be tuned on 9 to 12 positives per label (overfit; not done).

### 3.3 The real test: do own-OOF targets beat "just add a little Dread"? (`pl_sim_b3.out`)

Frontier = (1-b) v4 + b mix50 (Dread share b/2):

| b (Dread share) | gold58 | ws_rap | rho_rap |
|---|---|---|---|
| 0 (0) | .8927 | .336 | .645 |
| .25 (.125) | .9143 | .718 | .797 |
| .5 (.25) | .9204 | .777 | .825 |
| .75 (.375) | .9226 | .813 | .847 |
| 1 (.5) | .9262 | .838 | .864 |

Own targets vs the frontier at equal ws_rap:

| own teacher | gold58 above (+) / below (-) the line |
|---|---|
| **Ev4 (v4 students), any weighting** | **-.010 to -.019** |
| Eall (v4 + mix students) | -.001 to -.004 |
| TTRmix (single mix student) | **+.004** (SD .0074, P .71) |
| Emix (2 mix students) | +.002 to +.007 (UNK .75/NO .5/YES .5: +.0068, SD .0072, P .83) |

- At target level only the mix-student teacher clears the line, by a margin under 1 SD.
- It is not less parent-like than mix50: rho .869 to .874 vs .864; ws .813 to .823 vs .838.
- From the transfer ratio in 3.1 (target ws -.33 -> student ws -.03), a target ws gap of .015 to .025 maps to
  roughly **-.002 student ws** (MODEL), which is invisible.

### 3.4 The 10-04 DSD plan, re-scored with the real teacher (`pl_sim_b5.out`)

The 10-04 version (`bigjump/angle_labels.md`) used gold-selected crop R50s as the own teacher. With the real fold-0
fleet:

| target | gold58 | ws | vs line |
|---|---|---|---|
| DSD-own (own UNK+NO .75), TTRv4 / Ev4 | .9018 / .9029 | .709 / .721 | -.012 |
| **DSD-combo (+ Raptor YES .25)**, TTRv4 / Ev4 | .9176 / **.9195** | .781 / .793 | -.003 / -.002 |
| DSD-combo (+ Raptor YES .5), Ev4 | .9226 | .811 | +.000 |

F0 kill rule (pre-registered 10-04):
- Clause 1, own UNK+NO >= +.008 over v4: **pass** (+.009 / +.010).
- Clause 2, DSD-combo >= mix50 AUC at lower rho: **FAIL** (.9195 < .9262).
- **DSD is dead.** It is also below the Dread-share line.

### 3.5 Noise floors (`pl_sim_b4.out`) and a failed simulator

- Paired-bootstrap SD of a **student-vs-student gold58 delta = .007 to .012**. Examples:
  - F1 mix50 - TTR mix50 = +.0012, SD .0068;
  - F1 v4 - TTR v4 = -.017, SD .0105.
- A round-2 student's expected +.000 to +.006 is **not resolvable on gold58**.
- A single solo LB read has SD ~.003 to .005 (blendmath), so it is not resolvable on one LB read either.
- Projection simulator (ridge from the 4 students' logits to the target, to predict the student): **failed validation**.
  It predicts the v4 student at gold58 .906 / rho .889, vs real .861 to .878 / .840 to .850. Not used.

## 4. Round 2, designed concretely (optional; default = do not displace anything)

### 4.1 Target `labels_pl_tm5_v1` (pre-registered constants, no per-label tuning; builder `J2/build_pl_tm5.py`, also at `~/jump2/`)

```
S   = TTR mix50 seed 1043, raw best ckpt per fold: ~/fleet_runs/f{k}_s1043_ttr_mix50/oof_f{k}_s1043_raw.csv, k = 0..4
      logit -> z-score WITHIN each fold -> pool -> z-score -> map onto steven_v4 logit mean/std (non-gold rows)
w   = 0.75 report-UNK, 0.50 report-NO, 0.50 report-YES   (report_labels_v2 __verdict, missing = UNK)
y   = (1 - w) * steven_v4 + w * S'                      gold74 rows = steven_v4 (excluded from training anyway)
```

- Effective Dread share is ~.30, second-hand through the TTR student, vs .50 in mix50.
- Fold-0 analog (MEASURED): gold58 .9265 to .9267, rho .869, ws .813.
- Dry run on box (`build_pl_tm5_dry.out`): code path OK, gate T PASS on fold 0. Fold 1's OOF is partial while it trains;
  judge only finished folds.
- **Gate T** (CPU, ~1 min after the last fold, needs gold preds for folds 1-4):
  - target gold58 >= mix50 - .005 (= .9212);
  - AND ws_rap (all non-gold rows) <= mix50's.
  - Gold preds come from
    `NTHR=4 CUDA_VISIBLE_DEVICES= nice -n 19 python gold_infer.py ~/fleet_runs/f{1..4}_s1043_ttr_mix50/f*_s1043_raw_ttr224-*_best.pt`
    (~5 min each, MEASURED TTR rate 280-304 s at 12 threads).
- Then `python build_pl_tm5.py` writes `~/jump2/labels_pl_tm5_v1.csv`.

### 4.2 Student (recipe = J1 exactly, only labels and seed change)

```
cd ~/bigjump_build/code && ~/miniconda3/envs/rsna/bin/python fleet_train.py --model ttr --aug heavy --train_slices 16 \
  --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs 24 --seed 1044 --save_epochs 6-24 --workers 10 \
  --store ~/fleet_store --labels ~/jump2/labels_pl_tm5_v1.csv --folds ~/rsna_train/folds5.csv \
  --exclude ~/rsna_train/gold74_eval_set.csv --eval_labels ~/rsna_train/llm_labels_v4_blend.csv \
  --gold ~/rsna_train/gold74_eval_set.csv --fold 0 --out ~/fleet_runs/f0_s1044_ttr_pltm5
```

**Why TTR:**
- The teacher is TTR.
- 4.8 h per fold alone vs ~6 h for F1 (MEASURED J1 11.2 to 12.3 min/epoch alone).
- It is cheaper at serve time.
- The K3 kit already serves it.

**Why fold-0 first, not all-data:**
- Only a fold run gives OOF parent-likeness on the 867, which is the one thing measurable precisely here.
- An all-data seed (~6 h, MODEL = 4.8 h x 1.25) would only give gold58, which cannot resolve the expected effect.

**Leak caveat:** the teacher values on folds 1-4 come from models that trained on fold 0. So the probe's val AUC vs v4
is inflated (stacking leak plus traps 39). **Never gate on it.**
- gold58 is clean: gold74 is excluded from every teacher and every student.
- Parent-likeness on the 867 is only mildly affected.

**GPU cost:**

| item | V100 hours (alone) |
|---|---|
| fold-0 probe | 4.8 |
| full 5-fold | 24 |
| 2 all-data seeds instead | ~12 |
| CPU side (gold preds + build) | ~25 min |

### 4.3 Gate S (read at the end of the 24 epochs; best raw by v4-val selection, as J1; pre-registered)

| outcome | rule | then |
|---|---|---|
| **KILL** | gold58 < .903 (TTR mix50 f0 .9132 - .010) OR OOF rho_rap > .900 OR ws_rap > .840 | stop; the checkpoint is not shipped |
| **HOLD** | none of the KILL conditions, and not PROMOTE | the run = one more TTR mix50-type member (seed diversity only). Use it in the final fleet if serve budget allows; **train no more R2 folds** (blendmath: more mix50-type members have ~zero value) |
| **PROMOTE** | gold58 >= .913 AND (rho_rap <= .886 OR ws_rap <= .826), i.e. >= .01 less parent-like than TTR mix50 f0 at no loss | train R2 folds 1-4 (24 h); read with a parent + R2 fleet **W .50** blend (most information on r, per blendmath sec 7); never a solo slot for a single fold |

Also log, without gating: r_eff vs the Raptor proxy (oofx-style blend fit); PROMOTE expects <= .906 vs TTR mix50 .916.
GUESS outcome odds: HOLD ~.6, KILL ~.25, PROMOTE ~.15.

### 4.4 The rejected variant: an own-v4 self-teacher (DSD-own)

What it needs:
- TTR v4 folds 1-4 (~19-20 V100-h) to get a full v4 OOF table;
- then students (24 h).

What it would buy:
- Target gold58 +.010 over v4 (P .91), but **.012 below** the Dread-share line.
- It is at the `v09t` analog that read -.001.
- MODEL student: ~.925 to .928 solo at r_LB ~.885 to .89, i.e. on the same iso-lift band as every fleet blendmath
  modelled (+.0003 to +.0008).

Verdict: ~44 GPU-h for nothing measurable. Rejected.

## 5. Realistic LB gain and timeline

| route | ready | true blend lift vs parent .943 (MODEL) | public display (GUESS) | V100-h |
|---|---|---|---|---|
| 5-fold TTR mix50 (running; reference) | 10-06 ~03:45 | +.0004 (blendmath A1) | .943 to .944 | (running) |
| R2-TM probe, HOLD case (expected) | 10-06 ~09:00 | swaps into A1 at ~+.0000 to +.0001 | .943 to .944 | 4.8 |
| R2-TM 5-fold, PROMOTE case (P ~.15) | 10-07 ~09:00 | +.0004 to +.0008 | .944 (P .945+ ~.05) | 24 |
| DSD-own (rejected) | 10-09/10 | +.0003 to +.0008 | .943 to .944 | ~44 |

ETA note (MEASURED):
- Fold 1 epochs ran 12.9 / 11.5 / 11.5 / **17.2** min, and epoch 4 is at ~0.53 s/it.
- Epoch 3 coincided with CPU-heavy jobs on the box, including `gold_infer.py` (pid 314726, nice 19, ~12 threads).
- That job is now scoring EMA twins this track does not need. It ends on its own ~09:55-10:00.
- The chain ETA slips ~10-20 min (to ~03:45 10-06). Future CPU inference on the box: `NTHR=4`.

**Track total: public +0.000 central (-.001 to +.002), P(display >= .945) < .05, well before 10-15 / 10-22.**
- This is below blendmath's C6/C7 rows (P(.945+) .06 to .10). Their inputs ("own-OOF members .927 +- .004 at
  r = TTRv4 + .015") were optimistic: today's data put the v4-teacher student at the `v09t` analog.

## 6. What would change the verdict (and the cross-track pointers)

1. **A student that beats its own labels on gold58 is the trigger** (Archit's check), measured as student gold58 minus
   target gold58 > 0 on a fold-0 OOF.
   - Ours is -.012 to -.032 on all 5. The forum's best claim +.03 to +.04.
   - When a recipe-track or labels-track student flips that sign, rerun `pl_sim_b2/b3` with it as the own teacher
     (~5 min CPU). A teacher above the frontier by more than 2 SD (~.015) re-opens this track.
   - The weakest spots are where our students lose to the v4 text labels (gold58, MEASURED):

     | label | v4 label | our v4 students |
     |---|---|---|
     | MCL | .968 | .75 to .90 |
     | ACL | .987 | .90 to .95 |
     | Synovitis | .790 | .62 to .63 |

2. **Multi-source teachers (Raymond):** the one pseudo-label variant with a plausible new-information channel. Students
   trained on *different* LLM label tables, then their OOFs pooled.
   - It costs one 5-fold run per source (~24 h each).
   - Pooling the label tables directly at text level is ~free and belongs to the labels track. Try that first.
3. **gold58 overfits label work** (SpeedSci .945 -> .910 LB; Tian `v09t`/`v09o`). Any pseudo-label target must be built
   from pre-registered constants and judged on parent-likeness plus a blend read, never on gold58 alone.
4. **Dread share is concave at target level** (frontier table 3.3): a quarter of the Dread already buys ~.6 of mix50's
   gold58 gain. Student-level parent-likeness, however, barely follows target ws (3.1). Tell the labels track not to
   count on a low-Dread mix to drop r.
