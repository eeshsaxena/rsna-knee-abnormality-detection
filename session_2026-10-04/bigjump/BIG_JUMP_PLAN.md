# BIG JUMP PLAN: RSNA Knee, 10-04 to 10-12

Written 2026-10-04 ~05:00 IST. Research only: nothing was submitted, pushed, versioned, posted or run on the box.
One laptop CPU script was run for this file: `bigjump/plan_numbers.py` -> `bigjump/plan_numbers.out` (closed-form
blend numbers only).

This plan merges the five angle write-ups in `bigjump/` (topteams, models3d, labels, ensemble, contrarian), the two
judge rankings, HANDOFF (10-04 top), ENDGAME_V3, `k2/RUNBOOK.md` and `fleet2/code`. Where it departs from the judges'
combination, section 1.3 says why.

Tags: **MEASURED** = our LB reads, logs, code facts, our CPU checks, or Tian's audited ledger. **CLAIMED** = forum or
notebook self-reports. **MODEL** = closed-form binormal arithmetic (`calib2.py`, `plan_numbers.py`).
**GUESS** = judgement.

`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`.
`POSTS` = `SP/intel/live/top_team_posts.txt`, `TIAN` = `SP/knee/tian/experiments.md`, `LEDGER` =
`SP/round2/measured_levers.csv`. Times are IST unless marked UTC. A slot day runs 05:30 IST to 05:29 IST next day
(5 team slots, shared with SwaplY).

---

## 0. Verdict in ten lines

1. **No single lever gives +0.007..+0.015.** Every public-stack tweak reads 0.940-0.943 (MEASURED). The only route
   is our own ensemble, blended into the untouched 0.943 parent, and its value is set by two numbers: own solo S
   and effective correlation with the parent rho_eff (MODEL).
2. **Displayed 0.950 needs** S >= 0.937 at rho_eff .80, >= 0.9415 at .84, >= 0.9455 at .88
   (`plan_numbers.out`, MODEL). The calibration rho_eff = gold58 rho + .04..+.10 comes from four measured reads
   (`calib2.py`; ensemble angle section 3).
   [verify: only Tian gives a clean offset (+.10, and that member is teacher-mixed). The two crop reads were served
   at K=24 against a K=8 training contract, so their implied rho swings from .81 to .93 with the unknown served
   solo (`calib2.py` output). aastik has no gold58 rho at all. So +.10 is as well supported as +.07; treat the
   central +.07 as a GUESS until G-C (K1, K=8, ~11:30) gives the first clean point.]
3. **Route = "Strong-fleet stack":** (a) a 224 px long-schedule heavy-augmentation ConvNeXt-T recipe (TTR) to raise
   per-member strength; (b) report-gated silent-cell labels to buy strength without parent correlation; (c) about
   12-14 members from 3 recipes or families; (d) a serving stage that uses the measured >= 2.3 h window after the
   parent, with 2-pass TTA and one global weight W* ~ .40-.47. [verify: the window is 0.55-2.6 h (P 6.0-8.05 h);
   section 0 item 7.]
4. **Both gates read tonight on the V100 alone, at 0 extra Kaggle GPU:**
   - G-T2: does TTR beat family-1 by >= +0.008 OOF?
   - G-L: do gated labels add gold58 strength at <= +.03 rho?
5. **Expected public (MODEL, calibrated, rho_eff .84 central):**

   | Branch (probability) | Own solo | Public |
   |---|---|---|
   | TTR KILL (~60%) | 0.937 | ~0.947-0.948 |
   | TTR PASS central (~40%) | 0.940 | ~0.949 |
   | PASS upper tail | 0.944 | ~0.950-0.951 |

   Private is about 0.001 lower (GUESS; the model over-predicted Tian's member by ~0.002).
   [verify: the member solos behind this table (F1 .929, EV2S .926, TTR .934) are GUESS inputs about +.006..+.008
   above the nearest MEASURED analog: aastik's 6 ConvNeXt-T 320x24 members read .927 TOGETHER, which the same model
   inverts to ~.920 per member (`review_numbers.out` part 1). On analog inputs (F1 .922, EV2S .917, TTR .928): KILL
   own .930 -> .9455 at rho .84 (.9472 at .80); PASS own .934 -> .9467 (.9485 at .80). Also, a model
   over-prediction lowers public and private alike, so it is a haircut on both, not a public-to-private gap.
   P(PASS) .40 is generous: reviewer ~.30 (six factors changed at once, incl. 224 px; Tian MEASURED over-training at
   16 ep).]
6. **P(displayed >= 0.950 by 10-12) ~20% (GUESS).** P(>= 0.947) ~60% (GUESS). The team merge is still the larger
   lever. [verify: reviewer ~10% (range 5-15%) for >= 0.950 and ~40-45% for >= 0.947; section 12. The 10-04 night
   K2 fleetsolo read (S1) moves this more than anything else; section 12 has the S1 table.]
7. **Stop now:** crop-leg training and variants, W curves, flat teacher-mixed labels for new members, parent edits
   and speedups, and ENDGAME_V3's 5-member cap and W clip. The cap and clip rest on a parent runtime that is now
   measured shorter. Full list in section 9.
   [verify: "measured shorter" is one session (v9: submit 19:33, scored by ~01:50). The same evidence base also has
   MEASURED slow outcomes: 3 of ~10 member-add kernels gave no score (56526314, 56690233, 56768269), 2 K=24 crop
   members timed out (`endgame3/compute.md` s2), and v9crop25 was still running at the 03:40 check, >= 7.9 h after
   its 19:47 submit (HANDOFF line 32). Treat P as 6.0-8.05 h until v9crop25's finish time (Eesh, submissions page)
   or K2's per-file reads resolve it. Lifting the cap is fine only for valve-droppable extras.]
8. **Kaggle T4:** 4 commits (K2-K5) plus 1 spare, ~2.0-2.5 T4-h in total. **V100:** ~80 alone-hours of the ~138
   available to 10-10; the rest is reserve.
9. **Slots:** 5 on 10-04, 3 on 10-05, 2 on 10-07, 2-3 on 10-10, 1-2 on 10-11. About 15-18 of 45 by 10-12, no W
   curves.
10. **Merge decision point is 10-11 evening.** If K5 displays < 0.950, the plan recommends posting anyway with the
    best number. Agreement must be done by 10-13, and the merge closes 10-15 23:59 UTC. This is Eesh's call.

---

## 1. The route

### 1.1 Mechanism

- **Blend arithmetic.** Lift over 0.943 is roughly +0.4 to +0.5 for every +1 of own solo near W .4 (MODEL). Cutting
  rho_eff by .04 is worth about +0.0015..+0.002 (MODEL, `calib2.py` table).
- **Why strength comes first.** Every measured own-member read so far behaved as if rho were high (.84-.93), so
  strength, not more decorrelation, is the binding variable.
  - The exception is aastik's LLM-label ConvNeXt fleet: implied rho .79-.85 (MEASURED read, MODEL inversion).
  - Our fleet is the same design on the same kind of labels. A ConvNeXt on v4 labels is also unlike every parent arm
    (DINOv2, DINOv3, RadImageNet R50, CoAtNet). So this is the one member class with a measured low rho_eff.
  - [verify: the crop reads behind ".84-.93" were served at K=24 vs a K=8 training contract, so they cannot separate
    "high rho" from "weaker served member". And the analog is used one-sidedly: the plan takes aastik's rho but not
    his solo. The same analog measures the solo too: 6 such members = .927 LB together (~.920 each, MODEL).]
- **Why the TTR recipe.** Family-1 peaks at epoch 5-6 of a 10-epoch cosine (ensemble angle, fold-0 curve), which is
  the signature of fitting label noise early.
  - The 0.94-class claims share long schedules under heavy augmentation:
    - Myo: best epoch 28 of 50, "heavy aug + regularisation helps a lot" (POSTS:700-723, CLAIMED);
    - Will: 25 -> 50 epochs +0.004 (LEDGER, CLAIMED).
  - The same schedule without augmentation failed: 16 vs 8 epochs at light aug -0.012 OOF (TIAN:67, MEASURED).
  - 224 px makes it affordable: x0.49 compute, and 16-of-24 training slots give x0.67, so about x0.33 per epoch
    (MODEL). The price is ~-0.002 (224 vs 320, LEDGER:29, MEASURED).
    [verify: LEDGER:29 is one LB read that Tian himself scored INCONCLUSIVE (TIAN:89: +.002 vs a .005 floor, solo
    read SD ~.0047). The 224 cost is unmeasured in practice: anywhere from 0 to ~-.005, and on a CoAtNet, not a
    ConvNeXt. The x0.33 speed is MODEL; eval at 24 slots and fixed overheads do not shrink by x0.33 (G-T0 catches it).]
- **Why gated labels.** On report-silent cells (27% of cells) steven_v4 is close to a constant: within-stratum AUC
  0.50 (labels angle, MEASURED).
  - Filling only those cells with the image teacher keeps v4 wherever the report speaks.
  - That buys about half of the flat mix's target gain at about a third of its correlation cost (section 2,
    MEASURED at target level).
- **Why many members and TTA.**
  - 1 -> 5 folds gave +0.009 for a weak member (LEDGER:36). 2 seeds gave +0.003 and a third +0.001 (TIAN, MEASURED).
  - Cross-family members at within-family rho .86-.90 add +0.002..+0.003 (MODEL).
  - [verify: LEDGER:36 is a .877 member; for ConvNeXt-T, XuKong CLAIMED 5-fold vs 1-fold -0.001 (LEDGER:45), and
    Tian MEASURED 5 x 80% cross-fit = 1 all-data model (TIAN:100). The cross-family MODEL claim has a MEASURED
    miss: Tian's .921 third family at OOF rho .86 added 0 to the .932 c03 pair (TIAN:105), while the same model
    says .9348 (+.0028, `calib2.py`); reproducing .932 needs cross rho >= ~.93 (`review_numbers` run). So add
    the same +.07..+.10 offset to within- and cross-family rho, not only to the parent rho. Then a family that is
    more than ~.005 weaker than the lead adds ~0 (Myo's +.007 from 3-4 families came from near-equal ones, CLAIMED
    POSTS:700).]
  - Slice-offset TTA gave +0.0016 in a blend (LEDGER:15, MEASURED).
  - The measured parent bound (<= 6.3 h) leaves room to serve all of this (section 7).

### 1.2 What is new vs ENDGAME_V3

| Item | ENDGAME_V3 | This plan | Why |
|---|---|---|---|
| Lane 1 at 09:05 | crop v4 all-data R50 s0 | TTR-A fold-0 | Crop leg read 0.941-0.942 (MEASURED). K1 K=8 reads decide whether crop gets any more V100 |
| Lane 2 at 10:45 | F1 all-data s43 | F1 fold-0 on gated labels (paired, seed 42) | One-change label test at student level; directly usable member |
| Member cap | 5 fleet-only, 3 combined | ~12-14 members, 2 passes, valve | Parent <= 6.3 h incl. queue (MEASURED, HANDOFF line 35) |
| W | .30, clip .20-.35 | W* from solo read + calibrated rho, ~.38-.47 | `calib2.py` table: W* rises with own strength |
| Labels for new seeds | D1 winner (v4 vs mix50) | G-L winner among v4 / gated; mix50 only with a >= .008-.010 solo edge | Contrarian decision rule (MODEL) |

### 1.3 Departure from the judges' combination

The judges put TTR arm B (gated labels) on lane 2. This plan runs the gated labels on the **family-1 recipe**
instead, paired with f0_s42_v4 (same seed 42, same folds, same epochs; only the targets differ). Reasons:

- Each lane changes one factor:
  - lane 1 = recipe only (TTR-A on v4 vs f0_s42_v4);
  - lane 2 = labels only (gated vs v4, with mix50 as a third point already running).
  - This answers the judge critique that TTR bundles too much: labels are at least separated out.
- Under KILL (~60%), a TTR-B fold-0 would be wasted, while an F1-gated fold-0 is a shippable member and tells us the
  labels for every F1 seed.
- Under PASS, TTR all-data seeds take the G-L winner labels. This assumes the label effect carries across recipes
  (GUESS, acceptable at a +/-0.002 effect size).

Keep `bigjump/labels_gated75_logit.csv` as built:
- Raptor proxy, logit-matched, 0.75 on report-UNK cells, v4 elsewhere.
- Target gold58 .9109 at train-wide rho .724 vs the proxy and .690 vs the real v8 arm (MEASURED).
- The widedense variant (.9083 at .703) has the same gain per unit of rho. It is not worth a second file.
  [verify: the arithmetic says otherwise. Raptor gated75 = +.0182 gold58 for +.071 rho (.26 per unit); widedense
  .75 = +.0156 for +.050 (.31 per unit, ~20% better). The Raptor proxy IS the parent's v5/v8 arms, which is why
  angle_topteams s3 recommends widedense. Both teachers' gold58 gains are inflated (Dread picked checkpoints on
  these 58). The stakes are small (+/-.001), but the choice is not neutral: prefer widedense if the file is rebuilt.]

---

## 2. Evidence base (the numbers this plan stands on)

| Fact | Value | Tag, source |
|---|---|---|
| Public-stack tweaks | v9 swap .943; crop W .15/.25 .942/.941; teacher-mixed crop .940 | MEASURED, HANDOFF 10-04 |
| Parent hidden runtime | <= 6.3 h incl. queue (submit 19:33, scored by ~01:50) | MEASURED, HANDOFF line 35 [verify: HANDOFF line 36. One session. Conflicts with MEASURED slow outcomes: 3 no-score member-add kernels, 2 K=24 crop members timed out, v9crop25 still running >= 7.9 h after submit (HANDOFF line 32). Working range 6.0-8.05 h] |
| Fleet fold-0 at ep4/5 | mix50: gold58 .9002 (ep5), AUCvsV4 .8611, rho_proxy .857. v4: gold58 .8359, AUCvsV4 .8401, rho_proxy .796 (ep4) | MEASURED, HANDOFF D1 provisional |
| Target-level label frontier, gold58 / rho vs Raptor proxy | v4 .8927/.653; gated75 .9109/.724; mix50 .9262/.867 | MEASURED, `silent_fill_check.out`, contrarian |
| Target-to-student shrink | Tian +.032 target -> +.009 LB; our crop mixSA target +.035 -> gold58 -.0095 | MEASURED |
| rho_eff vs gold58 rho | Tian .935 vs .835; our crop .84-.93 vs .78-.81; aastik fleet .79-.85 | MEASURED reads, MODEL inversion (`calib2.py`) |
| Seed averaging | .927 + .927 (rho .952) -> .930, model .9294 | MEASURED, TIAN #26 |
| Slice-offset TTA | +0.003..0.006 single, +0.0016 in a blend | MEASURED, LEDGER:15 |
| 224 vs 320 px | -0.002 | MEASURED, LEDGER:29 [verify: one read; Tian's own verdict INCONCLUSIVE (TIAN:89)] |
| Cross-family add, measured vs model | .921 family at OOF rho .86 added 0 to the .932 pair; model said +.0028 | MEASURED TIAN:105; MODEL `calib2.py` [added by review] |
| Closest analog of our fleet, solo | aastik 6 x ConvNeXt-T 320x24 = .927 LB raw mean (~.920 per member, MODEL) | MEASURED `intel/aastik_ours_script.py:1060-1064` [added by review] |
| Single fold with heavy aug + long schedule | Myo ResNet single fold .935 on LLM labels with gold58 .9083; 5-fold .940 | CLAIMED POSTS:714-723 [added by review: the best support for TTR; note the labels are ~+.016 gold58 above steven_v4] |
| Long schedule, light aug | 16 vs 8 ep -0.012 OOF | MEASURED, TIAN:67 |
| Long schedule, heavy aug | Myo best ep 28/50; Will 25->50 ep +0.004 | CLAIMED, POSTS:700-723, LEDGER |
| Single models 0.94-0.95 | all 5-fold / all-data ensembles on own labels + TTA | CLAIMED, topteams section 1 |
| Best auditable single models | Tian c03 .932, Dread .932, our crop R50 .924, aastik 6x ConvNeXt .927 | MEASURED |
| Dead today (target level) | site debias, cross-label rerank, severity regex, per-label calibration, per-label masks | MEASURED, labels + contrarian |

---

## 3. Exact recipes

### 3.1 TTR-A fold-0 (lane 1, the recipe test)

- **Data.** `~/fleet_store` (4,407 npz, 4 views, <= 32 slices, 140 mm, 320 px), `SP/folds5.csv` fold 0
  (train 3,466 / val 867), gold74 excluded.
- **Model.** New module `fleet_model_ttr.py` = `KneeFleetNet` with one change in `encode()`:
  - After the uint8 -> float step and before normalisation: `F.interpolate(x, size=(224,224), mode='bilinear',
    antialias=True)`.
  - The same file is used by the trainer and the worker. It has its own `model_sha`, so family-1 checkpoints still
    load.
  - Backbone `convnext_tiny.fb_in22k_ft_in1k`, drop_path 0.2, head dropout 0.3, grad checkpointing on.
- **Labels.** Targets = `llm_labels_v4_blend.csv` (steven_v4). Eval = same file (`--eval_labels`).
- **Augmentation** (patch `_aug_view` behind `--aug heavy`, one draw per view):
  - affine: rot +-15 deg, scale 0.85-1.15, shift +-8%;
  - intensity: gamma 0.75-1.35, gain 0.85-1.15, bias +-15;
  - noise: Gaussian sigma 0-6 (p .3);
  - CoarseDropout: 1-3 holes of 8-16% side, same holes on every slice of the view (p .3);
  - slices and views: slice jitter +-1.0, view dropout 0.2, per-study mirror 0.5 (unchanged).
  - `--train_slices 16` (16 of 24 slots; infer on all 24). Move the slot subset before the affine, so the CPU augments
    only 16 triplets.
- **Optimiser.**
  - AdamW, backbone 1e-4, head 1e-3, wd 0.05, warmup 1 epoch, cosine over **24 epochs**.
  - EMA 0.9995, fp16 + GradScaler, bs 2 x accum 4, clip 1.0, seed 42.
  - OOF on the 867 every epoch (raw and EMA).
- **New flag `--save_epochs a-b`.** Saves raw and EMA from epoch a to b. All-data seeds then pick the stop epoch
  after the fact instead of committing to `--stop_epoch` at launch.
- **Time (MODEL).**
  - 12-15 min/epoch alone (family-1 measured 65-80 min shared, ~36-44 alone, x0.33).
  - About 22-28 min shared with an F1 job, so 24 epochs is ~9-11 h wall.
  - The first epoch is the time gate (G-T0).

### 3.2 F1-gated fold-0 (lane 2, the label test)

- The exact f0_s42_v4 contract (`~/fleet/fleet_train.py`, family 1, 320 px, 10 epochs, seed 42, fold 0,
  `--eval_labels llm_labels_v4_blend.csv`, gold74 excluded). Only change: `--labels labels_gated75_logit.csv`.
- Output dir `f0_s42g`. Its pack tag must be overridden (`f0_s42g`), the same re-key the two existing f0_s42 runs
  need (`pack_fleet_ds.py --tag`, small patch).
- **Time.** ~11-13 h wall shared (MEASURED rate).

### 3.3 Production seeds (from ~23:00 10-04, by branch)

- **PASS** (G-T2 pass):
  - TTR all-data seeds `--fold -1 --epochs 24 --save_epochs (peakA-3)-24`, seeds 1, 2, 3, 4, 5.
  - Labels alternate between the G-L winner and v4: s1 v4; s2 G-L winner; s3 v4; s4 G-L winner; s5 v4.
  - Time ~6-7.5 h alone each (x1.25 studies).
- **KILL** (G-T2 kill):
  - F1 all-data seeds s43-s47, `--fold -1 --epochs 10 --stop_epoch <F1 peak>`, G-L winner labels, ~5 h alone each.
- **MIDDLE:** TTR-A fold-0 ships as one extra member; seeds follow KILL.
- **Both branches:**
  - Family 2 EV2S (`tf_efficientnetv2_s.in21k_ft_in1k`, frozen BN, lr 2e-4, `fleet2/code`). `f2_preflight.py` must
    PASS before launch.
  - Fold-0 s142 first, then all-data s143-s145 only if G1c passes. Labels = the G-L winner. ~6-6.5 h alone each.

### 3.4 Serving stage (K3 onward; the parent is untouched)

- **Members.** Every shipped checkpoint, re-keyed:
  - F1 f0 mix50, F1 f0 v4, F1 f0 gated, TTR-A f0;
  - all-data TTR (or F1) seeds;
  - EV2S f0 and seeds.
  - The F1 f0 v4 run ships only if its gold58 at the shipped epoch is within .02 of mix50's.
    [verify: gold58 is biased against v4 here. mix50's teacher (widedense) was checkpoint-picked on these 58, and
    the epoch-4 gap was already .045 (.8808 vs .8359, HANDOFF line 33). The rule would drop the lowest-rho member
    on a biased instrument. Use the K-read solos instead, or default-include v4 (ENDGAME: "ties go to v4").]
- **Decode.** One shared decode for all families (fleet2 worker already does this, code fact).
- **Two passes per member:**
  - pass A = offset 0, no mirror;
  - pass B = `mirror_views` (the exact training op) + slice offset +0.5;
  - only if G-TTA passes.
- **Valve.** Projected after 64 studies against the 8.6 h process-age wall:
  1. drop pass B for all members;
  2. drop members in a family-interleaved order (TTR, F1, EV2S, TTR, ...);
  3. skip the crop stage.
  - Per-file fail-loud, as in K2.
  - [verify: this conflicts with the valve. ENDGAME s0.4 fail-loud writes a file only with the FULL member count,
    so a valve-trimmed main would not be written, and on a slow session (P 7.7-8.05 h) the read errors instead of
    scoring. Decide per file: `submission.csv` (main) is valve-tolerant, written with the surviving members, which
    are logged; diagnostic solos stay fail-loud. At P 8.0 h, pass A of ~8-12 members still fits
    (7 + max(18, GPU) <= 36 min with the estimated g). So the valve costs mostly TTA (~+.0007 blend), not members.
    The g values are ESTIMATES until K2's microbench.]
- **Combine:**
  - probability mean inside a recipe family (TTR / F1 / EV2S);
  - rank mean across families, weighted by fold-0 OOF strength (default .5/.3/.2 for TTR/F1/EV2S under PASS,
    .65/.35 F1/EV2S under KILL);
  - outer `rank_pct((1-W)*rank(P) + W*rank(F))`.
- **W\* rule.** Pre-registered, from the latest fleet-solo read S and rho_eff = gold58 rho vs the real parent (box
  mimic) + .07:

  | S | W* |
  |---|---|
  | <= .930 | .30 |
  | .935 | .38 |
  | .938 | .42 |
  | .941 | .46 |
  | >= .944 | .50 |

  W error costs <= .0005 across .30-.50 (`plan_numbers.out`).
- **Crop leg C.** W_C .10 inside a CF file only if K1 reads w15 >= .943 AND cropsolo >= .918. Otherwise no crop.
- **Files per commit:** `submission.csv` (main, W*), `submission_fleetsolo.csv`, plus at most one diagnostic solo
  (`ttrsolo`). Read main + fleetsolo by default.

---

## 4. Pre-registered gates

| Gate | When | Input | Pass | Fail / action |
|---|---|---|---|---|
| G-C calibration | K1 reads ~11:30 10-04 | `calib2.implied(.943, cropsolo, [(.25,d25),(.15,d15)])` | rho_eff <= .86: W* table stands | >= .90: own blends cap near +0.002 for S <= .938. Keep the plan (strength still pays), tell Eesh the merge is the only 0.950 route, and suggest posting by 10-06 |
| G-P0 TTR preflight | before the 09:40 launch | CPU forward on the 4 parity npz (`k2/_real/store`) via trainer path and worker path; augmented-view PNG dump of 1 study | logits max abs diff == 0; finite; `model_sha` distinct; mirror reverses sagittal depth | fix and re-run; launch slips at most 1 h |
| G-T0 time | TTR epoch 1 | epoch wall | <= 20 min alone-equivalent (<= 32 shared) | restart with `--epochs 18` (same aug) |
| G-T1 bug guard | TTR epoch 3 | OOF AUCvsV4 (867) | >= .80 | kill, debug |
| **G-T2 recipe** | ~20:00-22:00 10-04 | best-epoch max(raw, EMA) OOF AUCvsV4 on the 867, TTR-A vs f0_s42_v4 | **PASS: >= f0_s42_v4 best + .008**, no label below -.02, gold58 >= f0_s42_v4's (direction only) | **KILL: <= f0_s42_v4 best** -> KILL branch. Between: MIDDLE |
| **G-L labels** | ~22:00-00:30 10-04/05 | F1-gated vs f0_s42_v4 vs f0_s42 mix50, epochs 4-10: gold58 (mean raw/EMA of the best 3 epochs), rho vs the real v8 arm on the 867 AND on gold58 (box-mimic parent) | gated wins if gold58 >= v4's AND rho <= v4's + .03 on both sets AND rho >= .02 below mix50's | else v4. mix50 for new seeds only if its LB solo (K-reads) beats v4's by >= .008 |
| G-TTA | ~12:00 10-04 (3rd job, ~45 min) | `tta_bench.py`: f0_s42 mix50 checkpoints (raw, EMA) through the worker path on 867 + gold58: none / mirror / offset +-.5 / A+B / per-label max (Fracture, Contusion, menisci, ACL) / raw+EMA; fp16-half vs autocast parity | A+B >= +.002 OOF AUCvsV4 (paired bootstrap P > .9) and gold58 not lower | serve pass A only; spend the window on members |
| G2 / G3 (K2) | reads ~21:00-23:30 10-04 | fleetsolo S1; main d at W .30 | S1 >= .922: plan stands | S1 < .915 AND main d <= 0: Lane F in doubt; keep only the strongest recipe |
| G1c (EV2S) | EV2S f0 done | `g1c_corr.py` vs the F1 / TTR fold-0 OOF; closed-form lift of the F + EV2S mean vs F alone | within-class rho <= .86 AND higher lift | .86-.90: one seed; > .90: none, seeds go to TTR / F1 |
| G-K3 | K3 reads ~10-05 night | ttrsolo vs K2 fleetsolo; K3 main d | TTR solo >= F1 solo + .003 confirms PASS weighting | TTR solo < F1 solo: TTR family weight .3, no new TTR seeds |
| G-K5 final | K5 reads 10-10 night | main d, fleetsolo | best displayed -> standing Pick 1 | Pick 2 on a different leg (ENDGAME rule) |

Review notes on the gates [verify]:
- **G-L is tilted toward gated.** "gold58 >= v4's" is a direction test on an instrument the gated targets were
  built to please (the Raptor teacher was checkpoint-picked on gold58). Require gold58 >= v4 + .01, or (better)
  compare OOF vs v4 on the report-ADDRESSED cells only, where the gated and v4 targets are identical
  (angle_topteams s4). Keep the two rho tests as they are; they are the real guard.
- **G-T2 compares 48 candidates (24 epochs x raw/EMA) against 20.** Max-of-more selection on the same 867 adds
  roughly +.001..+.002 (GUESS). The +.008 bar absorbs it, but the MIDDLE band is partly selection noise.
- **G-TTA "only if SM < 85%"** will not hold while TTR and F1-gated both train. Expect it only when a lane frees
  (~19:00-00:30). K3 (10-05 11:00) can still use it.
- **G2/G3 thresholds** (S1 >= .922 "plan stands") are too lax for the 0.950 goal. S1 is the single best predictor
  of the 10-12 outcome; see the S1 table in section 12.
- **G1c (EV2S)** should also require solo parity: EV2S fold-0 OOF within .005 of the lead family. A weaker family
  adds ~0 (TIAN:105, MEASURED).

---

## 5. Day-by-day schedule, 10-04 to 10-12

Lane times assume f0_s42 mix50 ends ~09:05 and f0_s42_v4 ~10:45 (HANDOFF). Shared jobs run at ~1.8x their alone
time.

### 10-04 (Sun): slots 5

**Kaggle**
- 05:35 (Eesh, already approved): K1 `submission.csv`, `submission_fleetsolo.csv`, `submission_w15.csv`.
- ~13:30 (Eesh): push K2 as built (`k2/RUNBOOK.md`; one member = the `d1_pick.py` pick).
  [verify: HANDOFF line 31 records a later DECISION to ship BOTH fold-0 members averaged
  (`--ckpt <mix> --ckpt <v4>=f0_s42v4`), while RUNBOOK s7 says one member by rule. Settle this before 13:30. Both
  members make S1 a 2-member read, which shifts the S1 table in section 12 by about +.003.]
- 15:00-17:30 (Eesh): submit K2 main + fleetsolo. The deadline for a same-day read is 17:30.

**V100**
- Lane 1:
  - to 09:05 mix50 f0;
  - 09:05-09:40 box-mimic parent on gold58 (`~/auto_after_fleet.sh`, ~25 min) + K2 A1 GPU stage test (~8 min);
  - ~09:45 TTR-A fold-0 (G-P0 first), ends ~19:00-21:00.
- Lane 2:
  - to 10:45 f0_s42_v4;
  - 10:45 `d1_pick.py`, pack K2;
  - ~10:55 F1-gated fold-0, ends ~22:00-00:30.
- Third job: ~11:00-11:45 G-TTA bench. TTR at 224 leaves >= 8 GB free (MODEL). Run it only if SM < 85%, per the
  ENDGAME rule.

**Decisions and owners**
- Before 05:35, unqueue the crop all-data R50 s0 job from lane 1.
- 05:00-09:30 (Claude drafts, Eesh reviews): `fleet_model_ttr.py`, trainer flags (`--aug heavy`, `--drop_path`,
  `--dropout`, `--save_epochs`, subset-before-aug), `pack_fleet_ds.py --tag`, worker TTA pass B + family valve,
  `tta_bench.py`, laptop CPU smoke and parity.
  [verify: this is ~4.5 h for a new model module, 5 trainer flags, an aug rewrite, a pack re-key and a bench, in the
  same window as the K1 submit. A 2-4 h slip is likely (GUESS). Queue an F1 all-data v4 seed (`--stop_epoch` 6)
  behind f0_s42 mix50 as lane 1's default so the GPU never idles. G-T2 then reads ~10-05 00:00-03:00, and the
  PASS seeds start then, not at 21:00-23:00.]
- Upload `labels_gated75_logit.csv` to the box (md5 check).
- 11:30 G-C. Evening: G2/G3, G-T2, G-L.
- Rewrite HANDOFF.

### 10-05 (Mon): slots 3 (+0-2 by rule)

**Kaggle**
- ~11:00 push K3 = parent + {F1 mix50 f0, F1 v4 f0, F1 gated f0, TTR-A f0, TTR all-data s1 if done}, TTA per G-TTA,
  W* from G2 + G-C.
- Submit main + fleetsolo + ttrsolo by 17:00.
- Use a 4th slot only for a K2 rule follow-up: K2 main d <= -1 with S1 >= .915 -> K2 `w20`.

**V100, PASS branch**
- Lane 1: ~21:00 10-04 TTR s1 (v4) to ~09:00.
- Lane 2: ~00:30 TTR s2 (G-L labels) to ~13:00.
- Then lane 1 ~09:30: `f2_preflight.py` + EV2S f0 s142 to ~22:00.

**V100, KILL branch**
- Lane 1 F1 s43; lane 2 F1 s44; EV2S f0 after s43.

**Decisions:** K3 reads land ~19:00-02:00 -> G-K3.

### 10-06 (Tue): slots 0-2 (only to re-read a broken K3)

**V100, PASS**
- Lane 2 ~13:00 TTR s3 (v4) to ~01:00.
- Lane 1 ~22:00 F1 all-data s43 (G-L labels; recipe diversity) to ~10-07 08:00.
- G1c when EV2S f0 ends (~22:00).

**V100, KILL**
- F1 s45, EV2S f0 / G1c.
- V3D gate (models3d angle: Swin3D-T fold-0 to epoch 6) in the lane that would otherwise run a 4th F1 seed. With TTR
  dead, decorrelation is worth more.

### 10-07 (Wed): slots 2

**Kaggle:** ~10:00 push K4 (~8-9 members). Submit main + fleetsolo by 17:00.

**V100, PASS:** lane 2 ~01:00 TTR s4 (G-L labels) to ~13:00; lane 1 ~08:00 EV2S s143 (if G1c) else TTR s5.

**Decisions**
- Eesh sends merge partners the K3 numbers (ENDGAME already had this date).
- If K3 main >= .947, the K5 target of 0.950 is live.

### 10-08 (Thu): slots 0-1

**V100:**
- EV2S s144, TTR s5 (v4) or F1 s44.
- An idle lane may run the CoAtNet-0 TTR arm gate (runner-up 1).

**Decisions:** K4 reads -> W* for K5 (S table, section 3.4).

### 10-09 (Fri): slots 0-1

**V100**
- Last starts: a member must start by ~12:00 to finish by 10-10 05:00 (shared).
- EV2S s145 or a TTR or F1 seed of the stronger family per G-K3.

**Build:** Claude builds K5 (final members, W*, valve order) and runs the box A1 test as soon as any lane gap
allows.

### 10-10 (Sat): slots 2-3

**Kaggle**
- Quota refresh 05:30 IST.
- ~09:00 push K5 = final structure (12-14 members, 2-3 families, TTA, W*), plus the CF file only if the crop rule
  holds.
- Submit main + fleetsolo (+ cf) by 13:00; reads ~19:00-23:00.

**V100:** post-10-12 work starts (runner-up 2, own-OOF round 2), only if K4 S >= .935.

### 10-11 (Sun): slots 1-2

**Kaggle:** a K5 W variant only if K5 S moved >= .004 from K4's, or K5b with members that finished late.

**Merge decision point (evening).**
- Displayed >= .950: Eesh posts the merge requests.
- Below that: post anyway with the best read (recommended). The partner needs 10-12 and 10-13 to agree.

### 10-12 (Mon): slots 0-2

**Decisions**
- Update the standing pair: best P+F (K5 main) as Pick 1, a different leg as Pick 2.
- Freeze the recipe. Seeds and members may still change until the 10-19 code freeze.

**Slot totals to 10-12:** ~15-18 used of 45. **Kaggle T4:** K2 0.4 + K3 0.5 + K4 0.5 + K5 0.5 + spare 0.5 =
~2.4 T4-h (MODEL from K1's commit cost), ~1.4 of it before the 10-10 refresh.

---

## 6. Expected numbers

All public. Private ~ public - 0.001 (GUESS). Member solos are GUESS inputs; blends are MODEL with
rho_eff = gold58 rho + .07 (central .84, range .80-.88). Source: `plan_numbers.out`.

| Read | Own solo S | Public (rho_eff .84; range .80-.88) |
|---|---|---|
| K2: F1 f0 single, W .30 | .924-.934 (central .929) | .944-.946 |
| K3: 4-5 fold-0 members, 2 recipes, TTA | .932-.937 | .945-.948 |
| K4: ~9 members | .935-.940 | .946-.949 |
| K5, KILL: F1 x6 + EV2S x3 + TTA | .937 | .948 (.946-.950) |
| K5, PASS central: TTR x5 (.934 each) + F1 x4 + EV2S x3 + TTA | .940 | .949 (.947-.951) |
| K5, PASS upper: TTR members .938 each, cross-rho .88 | .944 | .951 (.949-.9525) |
| [review] K2 on analog inputs: F1 f0 single ~.918-.926 | ~.920 | ~.943-.944 |
| [review] K5 KILL, analog: F1 .922 x6 + EV2S .917 x3 + TTA | .930 | .9455 (.944-.947) |
| [review] K5 PASS central, analog: TTR .928 x5 + F1 .922 x4 + EV2S .917 x3 + TTA | .934 | .9467 (.945-.9485) |
| [review] K5 PASS upper, analog: TTR .932 x5, cross .88 | .937 | .948 (.946-.950) |

[verify: the review rows come from `bigjump/review_numbers.py` -> `.out` (same formula as `plan_numbers.py`). With
the model's measured over-prediction of cross-family adds (TIAN:105), the EV2S contribution in every row is ~0.]

- **Haircut.** The model over-predicted our crop and Tian's adds by 0.003-0.004 at gold58 rho. Using rho_eff absorbs
  most of that, but treat the central row as optimistic by ~0.001.
- **Probabilities (GUESS):**
  - P(PASS) ~.40;
  - P(displayed >= .950 by 10-12) ~.20, from .40 x .40 + .60 x .08;
  - P(>= .947) ~.60;
  - P(K2 main >= .944) ~.55.
- **Noise.** One read cannot separate +0.002 from +0.004 (blend delta SD .0012, solo SD .0047; MODEL, ENDGAME
  section 4).
- **Ranking.** Compare structures by posterior, not by the raw top read.

---

## 7. Cost and runtime fit

**V100 alone-hours to 10-10 05:00 (~138 available, MODEL)**

| Branch | Jobs | Alone-hours |
|---|---|---|
| PASS | TTR-A f0 ~5.5, F1-gated f0 ~6.7, parent gold58 + tests ~1.5, TTA bench ~0.75, TTR all-data x5 ~33, EV2S f0 + 3 ~25, F1 all-data x1-2 ~7-10 | **~80-83** |
| KILL | the same gates + F1 x5 ~25 + EV2S ~25 + V3D gate and seeds ~12 | **~76** |

Reserve ~55 h: runner-ups or more seeds.

**New spend versus the plan already in place:** ~12 alone-h for the two fold-0 tests. Every seed replaces an
ENDGAME seed.

**Hidden 9 h kernel.** Parent <= 6.3 h (MEASURED bound). Stage = 7 min + max(18 decode, GPU) + 10 crop (optional).
[verify: one-session bound; slow outcomes are also MEASURED (section 0 item 7). Plan for P up to 8.05 h: the full
2-pass load fits only to P ~7.0 h; pass A alone to ~7.6 h; at P 8.0 h ~8-12 pass-A members (estimated g). That is
fine IF the main file is valve-tolerant (section 3.4 note). Logistics, not GPU: K5's ~12-14 checkpoints are ~0.7-1.6
GB of dataset upload from the laptop (the box token is expired); budget the relay time before the 10-10 push.]
Per-pass g, MODEL from the timm bench (K2's microbench replaces it):

| Family | g per pass |
|---|---|
| F1 ConvNeXt-T 320 | 3-4.5 min |
| TTR 224 | 1.5-2.2 min |
| EV2S | 2.7-4.1 min |

| K5 load | GPU time | Stage | Ends at | Fits for P up to |
|---|---|---|---|---|
| TTR x6 + F1 x4 + EV2S x3, 2 passes | 58-87 min | 75-104 min | 7.6-8.0 h | ~7.0 h |
| Valve level 1 (pass A only) | 29-44 min | ~55-60 min | | ~7.6 h |

---

## 8. Risks

1. **rho_eff >= .90** (G-C fails).
   - Own blends cap near +0.002 whatever we build.
   - Mitigations: v4 or gated labels only; the W* table; and saying early that the merge is the only route to 0.950.
2. **TTR does not beat family-1** (~60%, GUESS). The KILL branch keeps the ENDGAME seeds plus EV2S and V3D. The cost
   is ~5.5 alone-h.
3. **224 px and heavy aug blur fine labels** (MCL, Lateral Meniscus). G-T2's per-label floor (-.02) catches it.
   Back off CoarseDropout first.
4. **Gated-label transfer fails** (crop precedent: target +.035 -> gold58 -.0095, MEASURED). G-L defaults to v4.
   Loss <= ~0.0005 (contrarian, MODEL).
5. **New parity surfaces** (TTR module, TTA pass B, re-keyed tags). G-P0, the box A0/A1 tests and `k2_check.py` are
   required before every commit. Fail-loud files mean a broken stage errors instead of reading 0.943.
6. **Runtime.** A slow session adds ~40 min (MODEL, PROFILE). The valve absorbs it down to pass A plus ~8 members at
   P 7.4 h.
7. **Gold58 is n=58** and gold-picked teachers inflate target-level gains. Every label decision uses rho + direction,
   never gold58 magnitude alone.
8. **Winner's curse.** ~15-18 reads by 10-12. Only main + fleetsolo per structure; posterior picks.
9. **[added by review] Member solos are overestimated.** This is the dominant risk. Every K3-K5 number assumes
   members ~.006-.008 above the aastik analog. K2's S1 (10-04 night) is the first measurement. If S1 <= .922, the
   0.950 goal by 10-12 is < 5% on this route (section 12), and the merge should not wait for 10-11.
10. **[added by review] Valve vs fail-loud** (section 3.4 note). Unresolved, a slow session errors the main file.

---

## 9. Stop doing (effective now)

1. Public-stack tweaks: v9 swaps, new crop W variants, teacher-mixed crop members (all 0.940-0.943, MEASURED).
2. Crop training: the queued all-data R50 s0/s1 come off lane 1. Crop seeds return only if K1 w15 >= .944.
3. New members on flat teacher-mixed labels (mix50 / mixSA). The existing f0_s42 mix50 run ships as one member only.
4. W curves, paired subset reads, per-label weights, per-label masking. The whole W question is worth ~+0.0004
   (MODEL, ENDGAME); masking is -0.006..+0.0017 (MODEL).
5. Dead target levers:
   - cross-label rerank;
   - site/language debiasing;
   - severity regex;
   - per-label calibration;
   - forward correction;
   - DSD's separate cross-site teacher table;
   - flat self-distillation.
   (MEASURED nulls today and in the ledgers.)
6. Parent edits: speedups p1-p3 (only if g >= 6 min/pass), dropping Repair-v1/Global96 (valve level 3 only),
   replacing parent arms.
7. R50 family 2, family-1 folds 1-4, native-3D / slices-as-channels / MRNet experts.
8. The KT / local-LB Kaggle GPU kernel and the held Hisern ext commit.
9. ENDGAME_V3's 5-member cap, "3 in combined" and the W clip at .20-.35. All three rest on a 7.7-8.05 h parent that
   the measured <= 6.3 h bound has replaced.
   [verify: "replaced" overstates one session against 3 timeouts and a >= 7.9 h latency. Dropping the W clip is
   fine (W error costs <= .0005). Lift the member cap only through a valve-tolerant main file.]

---

## 10. Runner-ups

1. **CoAtNet-0 TTR arm** (`coatnet_0_rw_224`, fixed 224 input, so the TTR resize fits).
   - CLAIMED +0.005..+0.006 solo over ConvNeXt in two teams' 5-fold reads (ringbearer .943 vs .937; kalyang .935 vs
     .930), but it is the parent's architecture family.
   - Gate: fold-0 OOF >= TTR-A + .005 AND rho vs v8 <= TTR-A's + .03.
   - ~6 alone-h. PASS branch only, idle lane from 10-08. Needs the timm weights on the box (wheel / weight route as
     for EV2S).
2. **Own-OOF round 2** (Archit's recipe).
   - Finish TTR folds 1-4, fill silent cells with our own 5-fold OOF at .75, retrain all-data.
   - Target-level analog: +.0126..+.014 at rho +.03 (labels angle, MEASURED with gold-picked teachers, so
     optimistic).
   - ~30 alone-h. Starts 10-10, so it serves the 10-17 final, not 10-12.
3. **V3D (Swin3D-T, video-pretrained)**, only as the KILL-branch decorrelation lane or a PASS idle lane. Use the
   models3d pre-registered gate:
   - kill if AUCvsV4 < .815 or rho to the Raptor proxy >= .79;
   - P+F+V must beat P+F by >= +.002 on gold58 (box-mimic parent).
   - [verify: a +.002 blend delta on n=58 is well inside gold58 noise. Use it as direction only, plus the rho test.]
4. **6-view store** (adds Cor T1, present in 3,985/4,407 studies; the parent uses it, MEASURED from
   `train_series.csv`). Needs a Kaggle CPU re-prep and +25-50% decode. GUESS 0..+0.002. Not before 10-12.

---

## 11. Build list for today (owner: Claude drafts, Eesh reviews and launches)

| # | Item | Needed by | Test |
|---|---|---|---|
| B1 | `fleet_model_ttr.py` (encode resize; `model_sha` registry entry in trainer and worker) | 09:30 | G-P0 parity, max abs 0 |
| B2 | `fleet_train.py`: `--model ttr`, `--aug heavy`, `--drop_path`, `--dropout`, `--save_epochs`, subset-before-aug (applies to TTR only; family-1 path byte-identical) | 09:30 | diff review; 1-epoch smoke on 8 studies, CPU |
| B3 | `labels_gated75_logit.csv` to `~/` on the box | 10:45 | md5 |
| B4 | `pack_fleet_ds.py --tag` (re-key f0_s42 x3) | 10:45 | pack two runs, tags differ, `k2_check.py` |
| B5 | `tta_bench.py` (worker path, variants of G-TTA) | 11:00 | runs on 10 studies first |
| B6 | worker: pass B (mirror + offset .5), family-interleaved valve, TTR module load | 10-05 08:00 | `smoke_k2.py` scenarios + new TTA/valve scenarios; box A0/A1 |
| B7 | K3 builder (multi-member, multi-family, W* literal, files main/fleetsolo/ttrsolo) | 10-05 09:00 | `k2_check.py --kaggle_commit` on the commit output |

---

## Sources

Local:
- `SP/bigjump/angle_{topteams,models3d,labels,ensemble,contrarian}.md`, `silent_fill_check.out`, `calib2.py`,
  `capacity.py`, `plan_numbers.py/.out`, `labels_gated75_logit.csv`, `build_gated.py`.
- `C:/Users/eeshs/Downloads/rsna-knee/HANDOFF.md` (lines 26-63).
- `SP/endgame3/ENDGAME_V3.md` (sections 0-4), `SP/k2/RUNBOOK.md`, `SP/fleet2/code/fleet_train.py` (args, `_aug_view`,
  T6 subset), `SP/fleet2/code/fleet_worker.py`.
- `SP/round2/measured_levers.csv` (lines cited), TIAN, POSTS.

Web (via the angle files):
- AnyMC3D benchmark and Ke et al. MIDL 2023 (models3d angle).
- RSNA 2023 abdominal 1st place (github.com/Nischaydnk/RSNA-2023-1st-place-solution).
- RSNA 2024 lumbar 2nd place (github.com/brendanartley/RSNA-2024-Competition).

---

## 12. Adversarial review verdict (2026-10-04, research only)

The reviewer ran no box, Kaggle or GPU jobs. Two laptop CPU scripts ran in seconds:
`bigjump/review_numbers.py` -> `.out` and `bigjump/review_s1_table.py` -> `.out`. Both use the same closed form
as `plan_numbers.py`. Inline edits are marked [verify]; review rows are marked [review] or [added by review].

### 12.1 What holds

- **Route choice is right.** No public-stack tweak has read above 0.943 (MEASURED). The only lever with a measured
  lift is an independent LLM-label fleet rank-blended into the untouched parent: aastik +0.002 (MEASURED).
- **Arithmetic checks out.** The `plan_numbers.out` thresholds reproduce exactly (needs .937 / .9415 / .9455 at rho
  .80 / .84 / .88).
- **Kaggle GPU stays within submission commits.** The gold58 parent comes from the box mimic. The TTA bench, the
  fold-0 tests and every gate run on the V100. Weights come through the CPU wheel kernel. K2-K5 is ~2.4 T4-h.
- **V100 hours fit.** PASS needs ~80 alone-h of ~138. If TTR runs at x0.5 instead of x0.33, add ~+25 h; it still
  finishes before 10-10. The binding limit is wall-clock to G-T2 plus build slip, not hours.

### 12.2 What is overclaimed or misread

1. **Member solos (the big one).** F1 .929, EV2S .926 and TTR .934 are GUESS inputs about +.006..+.008 above the
   only MEASURED analog: aastik's 6 ConvNeXt-T 320x24 read .927 together, about .920 each. The plan borrows the
   analog's low rho but not its solo. angle_topteams itself used ~.920 for F1 fold-0.
   - On analog inputs: KILL -> .9455, PASS -> .9467, PASS-upper -> .948 (rho .84).
   - Reaching 0.950 needs PASS-upper and rho_eff <= .80 together.
2. **rho_eff offset +.07.** The only clean offset is Tian's +.10. The crop reads are confounded (served at K=24,
   trained at K=8).
3. **Cross-family adds.** The model over-predicted Tian's MEASURED family add by >= .0023. The EV2S family (~25 V100
   alone-h) is worth ~0 unless it matches the lead family's solo within ~.005.
4. **Parent runtime "<= 6.3 h, replaces ENDGAME".** One session against 3 no-score kernels, 2 timeouts and a
   >= 7.9 h latency (v9crop25). The serving plan survives only if the main file is valve-tolerant. As written,
   per-file fail-loud plus the valve errors the main file on a slow session.
5. **Gold58 used where it is biased.** (a) G-L's "gold58 >= v4" favours gated (gold-picked Raptor teacher).
   (b) The "v4 ships only within .02 of mix50 on gold58" rule would drop the lowest-rho member. (c) The V3D +.002
   threshold on n=58 is noise.
6. **Smaller points.**
   - 224 px cost: one INCONCLUSIVE read.
   - 1 -> 5 folds +.009: a .877 member; XuKong's ConvNeXt-T read -.001.
   - The widedense teacher has ~20% better gain per unit rho, not "the same".
   - "Private = public - .001 because the model over-predicted" is a non sequitur.
   - G-TTA cannot run at SM < 85% while two jobs train.
   - K2 one-member vs both-members conflicts with HANDOFF line 31.

### 12.3 Probability and the pivot

**P(displayed >= 0.950 public by 10-12) = ~10%** (range 5-15%, GUESS built on MODEL).
- **Method.** Own solo was mixed 60/40 between the analog and the plan's centre. P(PASS) was set to .30. rho_eff
  was drawn 50/50 from U(.80,.88) and U(.84,.92). Read noise had SD .0012. A x0.85 haircut covers build slip,
  parity risk and the valve.
- **Other thresholds:** P(>= 0.947) ~40%; P(>= 0.945) ~70%; P(K2 main >= .944) ~45%.

**The pivot is K2's fleetsolo S1, read 10-04 night.** It is the first measurement of the input the whole plan rests
on (`review_s1_table.out`; S1 = one F1 fold-0 member; if K2 ships both members, subtract ~.003 before using the table):

| S1 read | P(>= .950 by 10-12) | P(>= .947) | Action |
|---|---|---|---|
| <= .918 | ~1% | ~10-15% | Route cannot reach .950 by 10-12. Tell Eesh to post merge requests 10-05 with the best read. Keep the fleet for the private board |
| .922 | ~5% | ~30% | Same: post merge requests by 10-06 |
| .925 | ~11% | ~45% | Plan as amended; merge decision at the K3 read (10-05 night), not 10-11 |
| .928 | ~21% | ~60% | Plan as amended |
| >= .931 | 35-50% | 75-85% | Plan as written; K5 target .950 is live |

### 12.4 Reviewer's route (amendments, not a new plan)

1. **Keep the strong-fleet route.** Pre-register S1 as the merge trigger (table above). Waiting to 10-11 with
   P ~10% spends the merge window. Partners need 10-12/13 to agree, and the merge closes 10-15 23:59 UTC.
2. **Make `submission.csv` valve-tolerant;** keep the solos fail-loud. Plan for P up to 8.05 h. At P 8.0 h, pass A of
   ~8-12 members still fits, so the valve costs TTA, not members.
3. **Gate EV2S on solo parity** (fold-0 OOF within .005 of the lead family) on top of G1c. Otherwise its ~25 alone-h
   go to seeds of the stronger recipe: +.003 for the 2nd seed and +.001 for the 3rd (MEASURED, Tian).
4. **Fix the gold58-biased rules.**
   - G-L: gold58 >= v4 + .01, or OOF on report-addressed cells, with the rho tests kept.
   - v4 member: default-include it.
   - Rebuild the gated file on widedense if it is rebuilt at all.
5. **Queue an F1 all-data v4 seed as lane 1's default.** The build will slip; G-T2 then reads ~10-05 00:00-03:00.
6. **First falsifiers within 24 h, all at 0 extra Kaggle GPU:**
   - G-C at 11:30 (K1 K=8 reads -> a clean rho_eff point);
   - S1 tonight (solo level);
   - G-T2 by ~03:00 10-05 (TTR vs F1, +.008 OOF bar).
   Any two of {rho_eff >= .88, S1 <= .922, G-T2 KILL} put 0.950-by-10-12 below 5%.

**Runner-ups (brief):**
- (a) **Merge-first.** Post on 10-05/06 with the K2/K3 reads. A two-way merge with a .947-.949 own-model team is
  the only route whose ceiling clearly exceeds .950 (GUESS; rules allow two-way only).
- (b) **Online soft-bootstrapping (ELR) as a free extra arm** in a TTR or F1 seed. It attacks the epoch-5 peak
  without importing parent rho (angle_labels runner-up 1; GUESS +0..+.003).
- (c) **Sequence head (bi-GRU over the 24 slice tokens)** as a fold-0 C arm only if a lane idles (GUESS 0..+.004).
