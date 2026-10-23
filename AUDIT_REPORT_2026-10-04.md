# RSNA Knee: independent audit report (2026-10-04, 19:00-21:15 IST, v2)

Scope: a read-only audit of the 10-04 handoff (`HANDOFF.md`, `NEXT_SESSION.md`, `session_2026-10-04/`), the live
Kaggle state, the box (V100), the out-of-fold (OOF) predictions of every fold-0 run, the serving code, and the forum.
Nothing was submitted, pushed, posted, killed or edited, either on the box or in this folder. This file is the only
thing added to the folder. v2 adds the training status, a third audit (OOF correlation with the parent, ensembling,
K3 readiness, the model-hash check) and the full session log.

Tags: **MEASURED** = read from a log, file, API or page (source given). **MODEL** = binormal blend model (the same
one as `session_2026-10-04/jump_wf/lb_proxy_model.md`). **GUESS** = judgement.

---

## 0. Bottom line

1. **K2 BLEND scored 0.943 (MEASURED). That read is AMBIGUOUS.** On ANY failed gate during the hidden rerun, the K2 cell
   restores the parent's bytes into `submission.csv`, and the parent scores 0.943. The Kaggle commit ran the fleet on
   only 3 studies, so the coverage and views gates (active only at n >= 11) have never run on Kaggle. Bayes:
   **P(BLEND was the parent fallback | read .943) is about .35-.65.** The pending **SOLO read settles it** (SOLO
   deletes `submission.csv` on failure: ERROR = fallback, a score = the fleet ran). It was still PENDING at 21:00 IST.
2. **If the fleet ran, its LB correlation with the parent is about .86 (q10-90 .79-.93), not the handoff's .95**
   (MODEL). The best global weight is then .15-.25, not .30-.45.
3. **Public evidence, independent of K2: aastik's LLM-label ConvNeXt fleet implies r_LB about .77-.80** (MODEL on
   MEASURED public reads .945 at W .5 and .6). At r about .80 an own ensemble of **.9255 gives +.003 (displays .946)
   and .938 gives +.007 (.950).**
4. **NEW (third audit, MEASURED on 867 fold-0 val studies):** parent-likeness (mean per-label Spearman vs the
   parent's own Raptor arms) by run:

   | run | Spearman vs Raptor |
   |---|---|
   | mix50 F1 | **.889** |
   | gated F1 | .872 |
   | v4 TTR | .856 |
   | v4 F1 | **.840** |

   - So **gated labels do NOT decorrelate much** (they sit closer to mix50 than to v4). **v4 + TTR is the best
     strength/correlation trade** we have.
   - Our members correlate .92-.94 with each other, so **ensembling them barely helps**: 4-member OOF .8812 vs
     best single .8811.
   - Lower r and more member diversity have to come from a **different recipe or family**, not more seeds.
5. **Training is healthy (21:02 IST).**
   - TTR (v4) is at ep20 of 24, best val .8762 (G-T2 +.0070 vs ref so far; PASS needs +.008); it ends ~22:55.
   - Gated F1 ep7 reads val .8734 (above the v4 reference's best .8692) and gold58 .8846; it ends ~23:15.
   - `~/auto_j1.sh` then launches J1 (TTR mix50 fold-0) for certain (KILL is no longer possible); J1 ends ~04:30-07:00
     on 10-05.
6. **K3 readiness:**
   - The K2 kit (worker, `build_k2.py`) refuses TTR members by design.
   - The K3 builder (B7) **is not built yet** (BUILD.md sec 5).
   - The model-hash check PASSES: the TTR checkpoints carry `f141aeed136dfc01` and gated/F1 carry
     `fa35880fca963652`, both equal to the worker's recipe on the box sources. The Kaggle dataset's `fleet_model.py`
     is byte-identical.
7. **Forum:** the top teams' likely hidden lever is **external knee MRI data**. The host rule is "OK only if free AND
   no institutional review or sign-off"; KneeCoT is banned. It is rule-gray and Eesh's call only (section 9).
8. **Winning outright is not realistic** (top .963, gold .958). The realistic range is **.946-.948 from our own fleet,
   .949-.951 via a merge.** On the private LB, any true lift over .943 jumps the ~950-team tie.

---

## 1. Training status (MEASURED, box 21:02 IST)

GPU: 80 C, SM 1027 MHz (of 1290), 100% util, throttle reason 0x20 (SW thermal slowdown), 10.3 GB used. Load average
~3.8 of 40 cores. Two jobs share the GPU (0.96x of one job alone, so the serial rule stands).

### 1.1 TTR fold-0 (pid 267158, `~/fleet_runs/f0_s1042_ttr`)

v4 labels, ConvNeXt-T at 224 px, heavy aug, 16-of-24 slices, 24 epochs, seed 1042. 0.83 s/it shared, ~29 min per
epoch including eval.

| ep | raw val | raw gold58 | ema val | ema gold58 |
|---|---|---|---|---|
| 13 | .8757 | .8838 | .8688 | .8775 |
| 14 | **.8759** | .8734 | .8713 | .8793 |
| 15 | .8740 | .8792 | .8731 | .8786 |
| 16 | .8741 | .8763 | n/a | n/a |
| 18 | .8754 | .8760 | .8759 | .8804 |
| 19 | .8749 | .8746 | **.8762** | .8795 |

- At 21:02 it was on ep20, iteration 200 of 1733. **ETA: last epoch (23) ends ~22:55 IST.**
- Saved bests: `f0_s1042_ema_ttr224-320x24_cv8762_best.pt` (ep19) and `..._raw_..._cv8759_best.pt` (ep14).
- **G-T2 so far: TTR best .8762 vs ref (f0_s42_v4) best .8692 = +.0070.** PASS needs +.008 (.8772), plus no label
  below ref - .02, plus gold58 >= ref. It is at MIDDLE now; PASS is possible if the EMA keeps rising over the last
  4 epochs.
- Plateau note: raw val has been flat at .874-.876 since ep13, while EMA is still creeping up. The cosine tail
  should help EMA most.

### 1.2 Gated F1 fold-0 (pid 267280, `~/fleet_runs/f0_s42_gated`)

`labels_gated75_logit`, family-1 ConvNeXt-T 320 px, **10 epochs**, seed 42. 2.23 s/it shared, ~69-74 min per epoch.

| ep | raw val (vs v4) | raw gold58 | val vs own gated labels |
|---|---|---|---|
| 3 | .8425 | .8560 | .8666 |
| 4 | .8604 | .8644 | .8851 |
| 5 | .8620 | .8768 | .8868 |
| 6 | .8679 | .8802 | .8927 |
| 7 | **.8734** | **.8846** | .8973 |

- **ETA:** ep8 ends ~22:15, then ep9 runs shared until TTR exits (~22:55) and alone after that. Ends ~23:15 IST.
- Already above the v4 F1 reference's best val (.8692) and gold58 (.861). Behind mix50 F1 (.8811 / .914), but mix50's
  gold58 is inflated (its teacher was epoch-selected on gold58).

### 1.3 The armed launcher and J1

- `~/auto_j1.sh` (pid 286637) was read in full; its log shows only "armed" (18:18). Sequence:
  1. wait for both pids;
  2. `gt2_read.py` -> `~/fleet_runs/gt2_final.txt`;
  3. guard: no python compute app and >= 12 GB free (the labmate's 313 MB gnome-remote-desktop is not python, so it
     does not block);
  4. launch J1.
- **J1 = TTR fold-0 on mix50_logit, seed 1043, 24 epochs**, out `~/fleet_runs/f0_s1043_ttr_mix50` (PASS or MIDDLE
  both take this branch; KILL is impossible now). Starts ~23:15-23:20 IST.
- J1 alone runs at an expected 0.36-0.55 s/it (MODEL), so about 13-19 min per epoch with eval. **ETA ~04:30-07:00 IST
  10-05.**

### 1.4 Finished reference runs (MEASURED logs)

| run | labels | best val (ep) | gold58 at best | epoch time |
|---|---|---|---|---|
| f0_s42 (F1) | mix50_logit | .8811 (ep7) | .9143 | ~69 min shared |
| f0_s42_v4 (F1) | steven_v4 | .8692 (ep7) | .8610 | ~69 min shared |

---

## 2. Live Kaggle state (MEASURED)

### 2.1 Submissions (`kaggle competitions submissions`, 19:22, 19:54 and 21:00 IST)

| ref | what | status |
|---|---|---|
| 56816039 | K2 BLEND: parent + fleet f0 mix50, global rank blend W .30 | **COMPLETE, 0.943** |
| 56816035 | K2 SOLO: fleet f0 mix50 alone | **PENDING** (~15 h after submit) |
| 56808853 | K1 cropfleet W .25 | 0.942 |
| 56800423 / 56799622 / 56799613 / 56799626 | mixsacrop25 / crop15 / crop25 / v9 | .940 / .942 / .941 / .943 |
| 56768512 / 56768236 / 56767629 | cnx W.40 fast / cnx fullparent W.50 / cnx solo | .935 / .930 / .894 |

### 2.2 Leaderboard (downloaded 13:53 UTC, 5,112 teams)

- Us: **rank 414, 0.943**, 41 submissions (team "Eesh saxena", eeshsaxena + swapiy).
- The .943 tie covers **ranks 357-1312** (956 teams) and is broken by submission time.
- Teams at or above each score: .958: 20 (gold line, top 10 + 0.2% N) · .950: 106 · .947: 182 · .946: 229 ·
  .945: 291 · .944: 356. Silver (top 5%, rank ~256) is about **.945-.946**.
- Score buckets: .963 x1, .962 x1, .961 x2, .960 x1, .959 x6, .958 x9, .957 x7, .956 x9, .955 x13, .954 x7, .953 x12,
  .952 x10, .951 x16, .950 x12, .949 x17, .948 x26, .947 x33, .946 x47, .945 x62, .944 x65, .943 x956, .942 x186.
- Top 10: rhoskeri .963 · Pa3BaJluHa .962 · joinT fusion .961 · Wassim Dobbi .961 · YOKKOISHO .960 · TomKroumov
  .959 · dalab .959 · NMarkS .959 · knee-deep .959 · Scott Willis .959.
- Merge targets:
  - **Myo Min Htet (caoyupeng, myominhtet) .949, rank 122, 213 subs**;
  - ringbearer (mehmetbarankrat) .948, rank 134, 101 subs;
  - Prateek Grover .945, rank 249, 47 subs.
- Other known authors: OzanM. .951, Theo Viel .951, Bend the Knee .950, RB & Tony .950, pjmathematician .947, tbongc
  .947, aastik .946, Dread .945, T.Kawahara .944.
- Merge cap check: Myo 213 + our 41 = 254, which is under the stricter 5 x days-running cap (~330+). Two-way only, as
  `merge_outreach.md` says.

---

## 3. Finding A: the K2 BLEND read cannot be taken at face value

### 3.1 The code path (`session_2026-10-04/k2/cells/k2_fleet_cell.py`)

- Lines 286-296: on any `_K2Fail` or exception in blend mode, the fail-loud files are deleted and **`submission.csv`
  is restored to the parent's bytes**. That is by design (P2), so that a broken fleet never costs the parent.
- Lines 332-338: only in `MAIN_MODE == 'solo'` is `submission.csv` deleted on failure, so the submission ERRORs.
- Gates that can fail on the hidden set (lines 201-237): full member count, hard stop at the wall, study ids, views
  per study < 3.0, coverage <= .98 (more than 2% of studies without a fleet prediction), a constant column, and the
  parent~fleet Spearman mean < .30. Coverage and views are active **only when n >= 11** (`GATE_MIN_N`).

### 3.2 How much of this was exercised (MEASURED receipts)

| run | studies | coverage | views/study | median-filled | decode s/study/thread |
|---|---|---|---|---|---|
| Kaggle commit `build/out_k2v1` | **3** | 1.0 | 3.67 | 0 | 1.48 |
| box A0 fleet / solo / fix | 2 | 1.0 | 4.0 | 0 | 1.37-1.44 |
| box `C_gold58` | **58** | 1.0 | 3.91 | 0 | 1.63 |
| box `stage_cpu2` | 2 | 1.0 | 4.0 | 0 | 2.20 |

Plus: all 4,407 train studies prepared with status OK. So the gates passed on every set we have, but **they were
never exercised on Kaggle's hidden data or its DICOM reader at scale.** The wall is not a plausible cause: the
parent's hidden runtime is <= ~6 h, the stage takes minutes, and the wall is 8.6 h. The plausible causes are
hidden-set DICOM oddities (more than 2% of studies undecodable) or a series-mix difference (views/study < 3).
GUESS prior: 5-15%.

### 3.3 What the read means (MODEL, `k2_audit.py` in appendix B)

Under "the fleet ran", a displayed .943 at W .30 has about 10% probability. Under "fallback" it has probability 1. So:

| assumption | P(fallback given read .943) | r_LB if it ran: mean (q10-90) | P(r <= .90 if it ran) |
|---|---|---|---|
| base: prior r U(.60,.99), solo N(.922,.008), model error .001, prior fail .10 | **.64** | .857 (.790-.927) | .79 |
| handoff's prior r U(.88,.98) | .58 | .914 (.885-.952) | .36 |
| prior fail .25 | .84 | .857 | .78 |
| model error .002 | .65 | .854 (.770-.938) | .75 |
| solo prior N(.915,.010) | .64 | .819 (.742-.899) | .90 |
| solo prior N(.930,.008) | .66 | .899 (.833-.963) | .49 |

r implied by the read, by solo (fleet ran, parent and read both in [.9425,.9435]): solo .905: .74-.78 · .910:
.77-.81 · .915: .80-.84 · **.920: .82-.87** · .925: .85-.90 · .930: .89-.93 · .935: .92-.96.

If it ran (MODEL): the expected true lift by W is +.0005 at W .10-.20, .0000 at W .30, -.0011 at W .40 and
-.0027 at W .50. **Best W about .15 (+.0006).**

### 3.4 Action

- **Read SOLO first.** ERROR -> BLEND was the parent and carries no information about r. Score -> the fleet ran, and
  the table above applies (read r off by the solo).
- Before the next blend submission (K3), make the gates fire on Kaggle: a commit over ~50-100 **train** studies
  through the exact serve path, with the gate stats printed in the log.

---

## 4. Finding B: public evidence that a pure-LLM-label fleet is low-r

### 4.1 aastik inversion (MODEL; public scores MEASURED, `jump_wf/public_intel.md`)

aastik s75-w50 / s75-w60 = d4 parent (same 14 datasets as ours) + his ConvNeXt-T 320x24 fleet (5 folds x 2 seeds,
LLM labels) at one global W; both read **.945**. His script's raw mean solo is .927 (`intel/aastik_ours_script.py:1061`).

| assumed fleet solo | r consistent with W .5 = .945 | with W .6 = .945 |
|---|---|---|
| .920 | .730-.759 | .678-.705 |
| .925 | .771-.800 | .728-.756 |
| **.928** | **.796-.826** | **.759-.788** |
| .932 | .831-.862 | .803-.832 |

At solo ~.927-.928 the two reads agree at **r about .79 within the model's ~.001 error**. The earlier "no r fits
both" verdict came from treating the reads as exact. This datum does not depend on our K2 gates.

### 4.2 What an own fleet must reach (MODEL, `lb_proxy_model.md` table 3.4, parent .943)

| r_eff | +.003 (displays ~.946) | +.005 (~.948) | +.007 (~.950) |
|---|---|---|---|
| .75 | .9160 (W .26) | .9250 (.34) | .9315 (.40) |
| **.80** | **.9255 (.29)** | **.9325 (.38)** | **.9380 (.44)** |
| .85 | .9330 (.34) | .9390 (.43) | .9435 (.51) |
| .90 | .9395 (.41) | .9440 (.53) | .9475 (.62) |

### 4.3 Scenario table (MODEL, `k2_update.py`, posterior "fleet ran"; lift at best global W)

| fleet | ensemble solo | r_e | W* | lift |
|---|---|---|---|---|
| current single member | .922 | .859 | .15 | +.0006 |
| 5 same-recipe (r_mm .95) | .926 | .877 | .18 | +.0007 |
| 5 mixed recipes/labels (r_mm .91) | .930 | .892 | .20 | +.0008 |
| 8 mixed, members +.005 | .935 | .895 | .32 | +.0019 |
| 8 mixed, +.005, lower-r labels (r -.04) | .935 | .854 | .37 | +.0037 |
| 8 mixed, +.010, r -.04 | .940 | .854 | .45 | +.0055 |
| 10 mixed, +.010, r -.06, r_mm .90 | .941 | .838 | .47 | +.0067 |

K3 (5 fold-0 members: F1 mix50, F1 v4, TTR v4, TTR mix50, F1 gated; r_mm .915) is worth about **+.001 at W ~.20 for
any SOLO read .910-.930** (`k3_scen.py`). It is a private-LB bet more than a public move. Section 5 measures
r_mm directly and finds it HIGHER (.92-.94), which makes this estimate optimistic.

---

## 5. Finding C (third audit): OOF strength and parent-likeness of every fold-0 run (MEASURED)

Method (`oof_audit.py`, appendix B):
- Copied each run's best-epoch OOF CSVs (867 fold-0 val studies; gold74 is excluded from training and val) and the
  label tables from the box to the laptop scratchpad, read-only, and ran CPU only.
- Parent proxy = `raptor_proxy_v5v8.csv`: the parent's own Raptor arms (maxspan-v5 .75 + native384-v8 .25) run over
  all train by teacher-v2. These arms were trained on all train, so they are **in-sample** here. Their Spearman with
  our members includes shared label fitting; use it for the **ordering** between runs, not as an LB r.
- Second proxy: Dread widedense (`teacher_train_probs.csv`).
- TTR and gated are still training: their OOF is best-so-far (TTR ema ep19, gated raw ep7).

### 5.1 Strength and parent-likeness

| run [tag] | AUC vs v4 | vs mix50 | vs gated | Spearman vs Raptor arms | vs widedense |
|---|---|---|---|---|---|
| mix50 F1 [raw] | **.8811** | .9313 | .9043 | **.889** | .864 |
| gated F1 [raw] | .8734 | .9198 | .8973 | .872 | .840 |
| v4 TTR [ema] | .8762 | .9178 | .8943 | .856 | .832 |
| v4 F1 [raw] | .8692 | .9072 | .8863 | **.840** | .810 |
| (Raptor arms, in-sample) | .9078 | | | 1 | |

Each label table favours the members trained on it, so read the three AUC columns together. Ordering on every column:
mix50 > gated ~ TTR v4 > v4 F1.

### 5.2 Member-to-member correlation (mean per-label Spearman)

|  | mix50 F1 | v4 F1 | gated F1 | v4 TTR |
|---|---|---|---|---|
| mix50 F1 | 1 | .931 | .944 | .926 |
| v4 F1 | .931 | 1 | .939 | .926 |
| gated F1 | .944 | .939 | 1 | .919 |
| v4 TTR | .926 | .926 | .919 | 1 |

### 5.3 Rank-mean ensembles

| ensemble | AUC vs v4 | vs gated | Spearman vs Raptor | vs widedense |
|---|---|---|---|---|
| mix50 F1 + v4 TTR | **.8830** | .9041 | .888 | .862 |
| v4 TTR + gated F1 | .8795 | .9009 | .881 | .852 |
| mix50 F1 + v4 TTR + gated F1 | .8824 | .9046 | .892 | .864 |
| v4 F1 + v4 TTR + gated F1 | .8787 | .8989 | .876 | .846 |
| all 4 | .8812 | .9024 | .886 | .857 |

### 5.4 Per-label AUC vs v4 (best tags)

| label | mix50 F1 | v4 F1 | gated F1 | v4 TTR | ens 4 |
|---|---|---|---|---|---|
| ACL | .8668 | .8347 | .8220 | **.8695** | .8584 |
| MCL | **.8077** | .7815 | .7889 | .8048 | .8028 |
| Medial Meniscus | .9205 | .9228 | .9223 | .9204 | **.9253** |
| Lateral Meniscus | **.9081** | .8858 | .8937 | .9006 | .9051 |
| Medial OA | .9056 | .8962 | **.9070** | .8996 | .9060 |
| Lateral OA | **.8694** | .8464 | .8573 | .8390 | .8587 |
| PF OA | **.8766** | .8558 | .8634 | .8631 | .8709 |
| Effusion | .8485 | **.8650** | .8495 | .8589 | .8631 |
| Synovitis | .8569 | .8699 | .8590 | **.8730** | .8710 |
| Baker's | .9166 | .9123 | **.9292** | .9227 | .9269 |
| Contusion | .8781 | .8645 | **.8825** | .8638 | .8785 |
| Fracture | **.9180** | .8958 | .9064 | .8990 | .9077 |

### 5.5 What it means

1. **Labels move parent-likeness:** mix50 is +.049 Spearman over v4 at the same recipe (.889 vs .840). That
   confirms the teacher-mix penalty.
2. **Gated labels are not a decorrelation lever:** +.032 over v4 (.872), i.e. two thirds of the way to mix50, for
   +.004 of v4-AUC. The handoff's hope that gated buys strength "at a quarter of the rho cost" holds at the target
   level (`angle_topteams.md` sec 3: target rho .717 vs .867) but shrinks in the trained model. GUESS: a stronger
   student re-learns the silent cells from the image either way.
3. **TTR v4 is the best trade so far:** +.007 AUC over v4 F1 for +.016 Spearman. Strength costs some r (a stronger
   model is more parent-like), but less than labels do.
4. **Our members are too similar to ensemble well:** r_mm .92-.94 (Spearman) even across recipe and labels. The
   4-ensemble equals the best single model. Note: OOF AUC is measured vs noisy LLM labels, which compresses ensemble
   gains, so the true gain is somewhat larger, but the direction is clear. **More of the same family will not raise
   solo strength enough**; the K3 estimate in 4.3 (r_mm .915) is optimistic.
5. **Strategic consequence:** the cheapest real gain is a **genuinely different second family** (different backbone
   and input, on v4 labels), which is what lowers both r_mm and r. Candidates already staged on the box:
   - EfficientNetV2-S / ResNet-50 family 2 (`~/fleet2_code`; EV2S needs `f2_preflight.py` PASS first);
   - a small-CNN 5-fold (EffNet-B0 / ResNet-34 at 256);
   - a sequence head (bi-GRU over slice tokens).

   GUESS: one of these at solo ~.92 with r_mm ~.85 to our ConvNeXt members is worth more than 4 more ConvNeXt seeds.
6. Per label: v4 TTR wins ACL and Synovitis, gated wins Baker's, Contusion and Medial OA, and mix50 wins the OA and
   meniscus side. Per-label routing was measured as noise before, so this does NOT justify per-label weights; it
   does show the members are not redundant everywhere.

---

## 6. Finding D: K3 readiness (MEASURED code and checkpoint metadata)

- **Model hash:** the worker computes `model_sha = sha256(getsource(fleet_model_ttr) + "\n" + getsource(fleet_model))[:16]`
  (sources line-joined). Recomputed on the box sources, read-only: TTR **f141aeed136dfc01** and base
  **fa35880fca963652**. Checkpoints:
  - `f0_s1042_{ema,raw}_ttr224-320x24_*_best.pt` have model_mod `fleet_model_ttr`, sha `f141aeed136dfc01`,
    ttr_size 224, fold 0, seed 1042 (ema at ep19, raw at ep14);
  - `f0_s42_{ema,raw}_cnxt320x24_*_best.pt` (gated) have sha `fa35880fca963652` (ema at ep6, raw at ep7).

  **All match.** The shipped Kaggle dataset `rsna-knee-cnxt-fleet/fleet_model.py` has the same md5 as the box source
  (897f83b2...). (A first naive hash of the raw file bytes gave a different value; that came from the hashing method,
  not a mismatch.)
- **Serving support:**
  - The bigjump worker (`~/bigjump_build/code/fleet_worker.py`) loads `fleet_model_ttr` (W-TTR, ttr_size buffer
    checked).
  - The K2 kit worker and `build_k2.py` refuse TTR by design.
  - The shipped Kaggle dataset contains only `fleet_model.py`, `fleet_model_f2.py` and the K2 worker; **no
    `fleet_model_ttr.py`.**
- **Missing for K3:**
  - B7 (the K3 builder) and B6 (worker pass B + family-interleaved valve) are listed as "10-05 items" in
    `bigjump/builds/BUILD.md` sec 5;
  - a new dataset version with `fleet_model_ttr.py` + the TTR/gated checkpoints, via `pack_fleet_ds.py --tag`;
  - the train-study gate commit (section 3.4).
- **Duplicate guard:** the worker asserts no duplicate fold/seed members ("raw AND ema?"). K3 must ship ONE tag per
  run (raw and EMA are near-duplicates anyway: Spearman .99 for mix50).
- **Timing budget (MODEL from receipts):** decode ~1.5 s/study/thread, F1 member ~3-4.5 min, TTR member ~0.5 F1-eq.
  A 5-member K3 is ~15-25 min on top of a parent at <= ~6 h. That fits the 8.6 h wall.

---

## 7. Audit of the handoff's other claims

| claim (handoff / NEXT_SESSION) | verdict |
|---|---|
| Crop leg dead (.940-.942 at every tested form) | **Agree.** Four reads, all at or below the parent. |
| gold58 cannot judge the parent (DINO .96-.996, Rad .97/.98 in-sample) | **Agree.** Those levels are not achievable out-of-sample. Only Raptor/CoAt arms are a fair gold58 reference. |
| Fleet r_LB prior ~.95 (k ~.4 transfer from crop anchors) | **Too pessimistic** for LLM-label fleets: aastik implies ~.79, and K2 (if it ran) implies ~.86 for mix50. |
| "P(.950 from own fleet) <= 5%" | **Revised up a little:** GUESS 10-20% by 10-22 if a second family on v4 labels joins TTR v4. Still well below the merge. |
| K2 W .30 as the default blend weight | **Lower it.** W* is .15-.25 at r .85-.90; ~.30-.45 only for a stronger, lower-r ensemble. |
| Serial rule (one job at a time) | **Agree** (MEASURED 0.96x when shared). |
| TTR beats F1 | **Agree** at equal labels (val +.0070, gold58 +.015-.02, OOF on every label table). |
| Gated labels as the decorrelating option | **Disagree (new):** trained-model Spearman vs the parent's Raptor arms is .872 vs .840 for v4. Gated is a strength option, not a decorrelation option. |
| J1 = TTR on mix50 | **OK as an A/B and a member.** After J1, production should be TTR v4 plus a second family on v4 labels, not more mix50. |
| Merge = only route >= .945 | **Partly revised.** Merge is the only route to >= .949. An own fleet at r ~.80 can plausibly reach .946-.948. |
| inner_injection dropped | **Agree** (~0, and no better than a smaller global W). |
| Fast path (no grad ckpt / compile / fused AdamW / less eval) | **Agree, higher value now.** |
| Model-hash parity for TTR | **Verified** (section 6). |

---

## 8. Session log (this conversation, 10-04 evening)

1. **Folder context (~19:15):**
   - Listed Downloads; the most recent work was rsna-knee (18:24), then gemma-4-developer-agent (14:47).
   - Read `NEXT_SESSION.md`, `HANDOFF.md` and the Gemma handoff.
   - Fixed a stale memory index line (Gemma v13 had scored 0.10, not "pending").
2. **First RSNA analysis (~19:20-19:45):**
   - K2 BLEND found COMPLETE at 0.943; SOLO pending.
   - Box: TTR at ep16, gated at ep6, `auto_j1.sh` armed and correct.
   - Live LB pulled: rank 414.
   - Read `public_intel.md`, `angle_topteams.md`, `lb_proxy_model.md` and `merge_outreach.md`.
   - Binormal posterior: r ~.86 if the fleet ran, W* .15. The K3 scenario gives ~+.001 at W .20.
   - First reported conclusion: "fleet route alive, lower W, post merge, K3 at W .20, production lean v4/gated".
3. **First audit (~19:50-20:30):**
   - Found the fallback ambiguity (section 3); P(fallback | read) .35-.65.
   - Box receipts confirmed the gates only ran on 3 studies on Kaggle.
   - aastik inversion (section 4).
   - Forum read (section 9).
   - Wrote report v1.
4. **Second audit (~21:00-21:15, this v2):**
   - Training status (section 1).
   - OOF parent-likeness and ensembling audit (section 5).
   - K3 readiness and model-hash check (section 6).
   - Corrected "lean to v4/gated" to "lean to TTR v4 plus a second family".
5. **Memory** (`project-rsna-knee-kaggle.md`) holds the K2 read and the audit findings. Scratch scripts are in the
   session scratchpad (temporary), so the code is copied into appendix B.

---

## 9. Forum check (MEASURED, read signed out on 10-04 ~19:40 IST)

- **External knee MRI data** (thread 743416, 59-62 votes, last comment 21 h ago, by YYama, 7th):
  - The author says relevant external knee MRI could let "a fairly simple approach … achieve a dramatic
    improvement".
  - Datasets named: MRNet, SKM-TEA, fastMRI / fastMRI+, OAI, KneeMRI, KneeXNet-2.5D. There is no host answer
    there.
  - Mazzutti (20th, .958) argued in thread 733652 that rule 2.6(b) permits them.
  - Tucker (41st-42nd) and others say nearly all fail: MRNet's agreement bans derivative works, KneeMRI is CC
    BY-NC-ND, and OAI via NDA needs institutional sign-off.
- **Host ruling (thread 741819, OAI):** allowed "if it is generally accessible to all participants without an
  escalated (i.e. institutional) review and sign-off or access to a legal team, and is free of charge". KneeCoT is
  banned (thread 734109).
- **Read:** this is a plausible part of the gap between 0.943-0.95 and the top 20. fastMRI+ (pathology boxes for
  meniscus, ACL, edema, effusion on fastMRI knee) is the only candidate that might pass, and only if its data-sharing
  agreement is a personal click-through. **Rule-gray. Eesh must read the terms himself and decide; Claude should
  never download or use any of it on its own.** Cost if allowed: about a week, which is tight for 10-22.
- **LLM labels:** the host's pinned post (733965, "Use of Commercially Hosted LLMs") covers this. It is already noted
  as allowed in memory.
- **Prize vs medal:** the eligibility threads (744056, 745666, 735121, 745283) are still open. Public weights in the
  kernel are medal-OK and prize-questionable. Not decision-relevant for us (medal target).
- The "submission stuck" thread (745441) is a newcomer; 9 h score latency is normal.
- Already known and not a lever for us: "slice order is random in the public baseline" (745759); our parent and fleet
  sort by IPP.

---

## 10. Recommended plan (decision points marked EESH)

1. **Read SOLO** (ref 56816035) before acting on K2.
   - ERROR: BLEND was the parent.
   - Score: read r off table 3.3 by the solo value.
2. **10-05 morning, build K3 properly:**
   - B7 builder;
   - new dataset version with `fleet_model_ttr.py` + TTR v4 ep19-EMA + gated ep(best) + mix50 F1 (+ J1 when it ends),
     ONE tag per run;
   - train-study gate commit (>= 50 studies, gate stats logged);
   - then submit K3 at **W ~.20**.

   Expected +.000-.001 (MODEL, optimistic given r_mm .92-.94). Value: the private-LB tie and a measured r.
3. **Production after J1 (~05:00-07:00 10-05):**
   - (a) TTR v4 all-data seeds at a fixed epoch (~ep19-22, EMA);
   - (b) **a second family on v4 labels** (EV2S after `f2_preflight.py` PASS, or R50 / small CNN) for lower r_mm and r;
   - (c) finish the fast path first (1.3-1.6x members/day, GUESS).

   Gate each new family on OOF Spearman vs the Raptor proxy (target <= .85) and r_mm vs TTR v4 (target <= .90),
   using `oof_audit.py` logic.
4. **EESH: post the merge requests 10-05/06** (`merge_outreach.md`; refresh it with: our own ConvNeXt-T fleet, the
   TTR recipe at fold-0 val .876, our V100). Myo (.949) first. You post; Claude never posts.
5. **EESH: external-data decision** (section 9). Default = no.
6. **Final two picks:** one safe (parent, or the best fleet blend with a confirmed clean stage) and one fleet blend at
   model W*. Never pure repro_0946. Only scored candidates.

Honest odds (GUESS): bronze high (public .943 sits in the bronze zone, and any true lift jumps the private tie);
silver about 30-45% via own fleet + merge; gold < 3%; outright win < 0.5%.

---

## Appendix A: sources

- Kaggle CLI: `competitions submissions` (19:22, 19:54, 21:00 IST), `competitions leaderboard --download` (13:53 UTC).
- Box (read-only ssh):
  - `~/fleet_runs/{f0_s42,f0_s42_v4,f0_s1042_ttr,f0_s42_gated}.log` and their `oof_*.csv` / `*_best.pt` metadata;
  - `~/auto_j1.sh`, `~/fleet_runs/auto_j1.log`;
  - `ps` of pids 267158 / 267280 / 286637 / 75493, `nvidia-smi`;
  - `~/k2_test/runs/*/working/fleet_receipt.json`, `~/k2_test/analysis_wf/proxy_oof867.*`;
  - `~/bigjump_build/code/{gt2_read.py, fleet_worker.py, fleet_model_ttr.py, fleet_model.py}`;
  - `~/k2_test/ds/rsna-knee-cnxt-fleet/`;
  - `~/rsna_train/{llm_labels_v4_blend, labels_mix50_logit, labels_gated75_logit, teacher_train_probs, folds5, gold74_eval_set}.csv`.
- Local:
  - `session_2026-10-04/k2/cells/k2_fleet_cell.py`, `k2/build*/out_k2v1/fleet_receipt.json`, `k2/box/BOX_TEST.md`;
  - `bigjump/builds/BUILD.md`, `bigjump/angle_topteams.md`;
  - `jump_wf/lb_proxy_model.md`, `jump_wf/public_intel.md`;
  - `merge_outreach.md`, `HANDOFF.md`, `NEXT_SESSION.md`;
  - the old session scratchpad `teacher_v2_out/raptor_proxy_v5v8.csv`.
- Forum (signed out, read-only): threads 743416, 733652, 741819, 745441, plus the listing sorted by recent comments.

## Appendix B: model and audit scripts (the scratchpad is temporary, so the code is kept here)

Shared model: AUC = Phi(d / sqrt 2). Blend at weight W: d_b = (1-W) d1 + W d2,
v = (1-W)^2 + W^2 + 2 W (1-W) r, AUC_b = Phi(d_b / sqrt(2 v)). Ensemble of k members with member-to-member r_mm:
v_k = (1 + (k-1) r_mm) / k, solo_e = Phi(d / sqrt(2 v_k)), r_e = r_m / sqrt(v_k).

```python
# k2_audit.py: posterior on "fallback" and on r given BLEND W.30 displayed .943
import numpy as np
from scipy.stats import norm
d = lambda a: np.sqrt(2) * norm.ppf(a)
def blend(P, S, r, W):
    v = (1-W)**2 + W**2 + 2*W*(1-W)*r
    return norm.cdf(((1-W)*d(P) + W*d(S)) / np.sqrt(2*v))
rng = np.random.default_rng(1); N = 1_000_000
def run(solo_mu=.922, solo_sd=.008, rlo=.6, rhi=.99, model_err=.001, pfail=.10):
    S = rng.normal(solo_mu, solo_sd, N); r = rng.uniform(rlo, rhi, N); P = rng.uniform(.9425, .9435, N)
    fail = rng.random(N) < pfail
    b = np.where(fail, P, blend(P, S, r, .30) + rng.normal(0, model_err, N))
    k = (b >= .9425) & (b < .9435)
    return fail[k].mean(), r[k & ~fail]          # P(fallback | read), r samples if it ran
```

```python
# aastik inversion: r consistent with W .5 / .6 reads of .945 on a .943 parent
rs = np.linspace(.5, 1, 501)
for S in [.920, .925, .928, .932]:
    for W in (.5, .6):
        ok = [r for r in rs if any(.9445 <= blend(P, S, r, W) < .9455 for P in np.linspace(.9425, .9435, 11))]
```

```python
# k3_scen.py: 5 fold-0 members, r_mm .915, member solo = SOLO - .002, member r = K2-implied r - .02
for S1, rk in [(.910, .79), (.915, .815), (.920, .845), (.925, .875), (.930, .905)]:
    Sm, rm, k, rmm = S1 - .002, rk - .02, 5, .915
    vk = (1 + (k-1)*rmm) / k; Se = norm.cdf(d(Sm) / np.sqrt(2*vk)); re = rm / np.sqrt(vk)
    # best W over 0..0.7 -> W* ~.19-.25, lift ~+.0009..+.0013
```

```python
# oof_audit.py: fold-0 OOF strength vs three label tables, Spearman vs the parent's Raptor arms, ensembles
# inputs (copied read-only from the box): <run>_{raw,ema}.csv = ~/fleet_runs/<run>/oof_*.csv;
# llm_labels_v4_blend.csv, labels_mix50_logit.csv, labels_gated75_logit.csv, teacher_train_probs.csv,
# raptor_proxy_v5v8.csv (teacher_v2_out)
import pandas as pd, numpy as np
from scipy.stats import rankdata, spearmanr
T = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',
     "Baker's",'Contusion','Fracture']
rd = lambda f: pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
def auc(y, s):
    ok = np.isfinite(y) & np.isfinite(s); y = y[ok] >= .5; s = s[ok]
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1*(n1+1)/2) / (n1*n0)
macro = lambda P, lab: np.nanmean([auc(lab[t].values, P[t].values) for t in T])
sp = lambda A, B: np.nanmean([spearmanr(A[t].values, B[t].values, nan_policy='omit')[0] for t in T])
# best tag per run = max macro AUC vs v4 over {raw, ema}; ensembles = mean of per-label pct ranks
```

```python
# model hash exactly as the worker computes it (fleet_worker.py _model_sha)
import hashlib
j = lambda f: "\n".join(open(f).read().splitlines())
ttr_sha  = hashlib.sha256((j("fleet_model_ttr.py") + "\n" + j("fleet_model.py")).encode()).hexdigest()[:16]  # f141aeed136dfc01
base_sha = hashlib.sha256(j("fleet_model.py").encode()).hexdigest()[:16]                                     # fa35880fca963652
```
