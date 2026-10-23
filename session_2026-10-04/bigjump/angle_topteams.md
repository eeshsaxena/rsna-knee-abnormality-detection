# Big jump, angle: what top-team single models have that our fleet lacks (2026-10-04 ~05:00 IST)

Research only. Nothing submitted, pushed, versioned, posted or run on the box. One CPU check was run on the laptop
(target tables only, no model inference): `bigjump/silent_fill_check.py` -> `bigjump/silent_fill_check.out`.
Forum extracts used: `bigjump/forum_filtered.txt`, `bigjump/topusers.txt` (built from `intel/comments_all.jsonl`).

Tags: **MEASURED** = LB read, ledger number, our run, or a number computed today. **CLAIMED** = forum self-report.
**MODEL** = binormal blend formula (`SP/fleet_g1b.py:best_lift`, parent 0.943) or arithmetic from measured rates.
**GUESS** = my judgement.
`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`.
`POSTS` = `SP/intel/live/top_team_posts.txt`, `TIAN` = `SP/knee/tian/experiments.md`, `LEDGER` = `SP/round2/measured_levers.csv`.

---

## 0. Bottom line

1. **Nobody whose numbers we can audit has a single model above 0.932.** Tian c03 CoAtNet-1 0.932 (TIAN:98), Dread
   finespacing SWA 0.932 (forum 09-05), Tian ResNet-34 0.921 (TIAN:104), our crop R50 0.924, aastik's 6-model
   ConvNeXt-T 320x24 fleet 0.927 (`intel/aastik_ours_script.py:1061-1062`). All MEASURED.
   The 0.94-0.95 "single models" are all CLAIMED, and almost every one is (a) a 5-fold or all-data ensemble of one
   recipe, (b) trained on the team's OWN labels, (c) with TTA, and several with (d) long, heavily augmented schedules.
2. **The levers that separate 0.927-class from 0.94-class, in order of evidence:** fold/seed ensembling + TTA
   (+0.005..+0.010), image-teacher targets (+0.009 MEASURED, but it collapses decorrelation), long schedule with heavy
   augmentation (+0.002..+0.006, CLAIMED/GUESS), then small stuff. Resolution, backbone size, slice count (beyond
   what triplets already cover) and pooling are NOT the gap (measured nulls).
3. **New today (MEASURED, target level):** filling ONLY the report-silent cells with the image teacher, and leaving
   every addressed cell on steven_v4, keeps about half of the teacher's gold58 gain at about a quarter of its
   correlation cost with the parent's Raptor arms (Section 3). This is Archit's "fill the silent cells" (POSTS:294)
   in a form that does not turn our member into a copy of the parent.
4. **Proposal (Section 4): "TTR fleet" = the top-team recipe on our existing fleet store and code**: ConvNeXt-T at
   224 px (on-GPU resize of the 320 store), 16-of-24 triplet bags in training, 24 epochs with heavy augmentation,
   EMA, mirror TTA, two label variants (pure steven_v4, and silent-only teacher fill), shipped as fold-0 + 3 all-data
   seeds. It is ~3x cheaper per epoch than family 1, which is what pays for the 2.4x longer schedule and the seeds.
   Expected solo 0.935 (range 0.927-0.942, GUESS), rho vs parent 0.77-0.80, blend **+0.004..+0.009, centre +0.006**
   (MODEL). ~31 V100 alone-hours, ~1 T4-hour of commits, fits the 9 h kernel.
5. **Honest ceiling:** the closest existing measurement of our fleet design is aastik's 10-model ConvNeXt-T 320x24
   fleet: +0.002 on 0.943 at W .5-.6 (s74-c 0.943 -> s75 0.945, MEASURED public). To reach +0.006 we must beat his
   members by ~+0.008 solo. P(displayed >= 0.950 from this route alone) ~35-45% (GUESS). The merge stays the
   bigger lever.

---

## 1. What the 0.94-0.95 single models actually are (evidence)

| Who (team score) | Claim | Backbone / px | Ensembled? | Labels | Training | Tag, source |
|---|---|---|---|---|---|---|
| Archit Konde (0.954) | single fold 0.950; OOF 0.930 on gold58 | 2.5D CoAtNet, 224, "a few neighbouring slices" | single fold | own extraction + OOF predictions filled into SILENT cells, "heavier than 50/50" | n/a | CLAIMED, POSTS:279-295 |
| Raymond (0.955) | full-train 0.938 -> 0.940 -> 0.949, "still label"; gold 0.925 -> LB 0.947 | CoAtNet 288 | full train | own Qwen-32B labels, pseudo-labelling + blending label sources | n/a | CLAIMED, POSTS:254-261 |
| Tucker Arrants (0.955) | 0.934 -> 0.943 "some tweaks" at 224; later "most models low 0.95" | "standard for these competitions, relatively small"; advises ResNet34/EffNet-B0, simple pooling | validates on full OOF; TTA in the 25 min kernel | crude Qwen labels; better labels gave 0 | ablate standard + MRI-specific aug stacks | CLAIMED, POSTS:168-245 |
| tennogh (0.959) | 0.942 raw, 0.943 with TTA | 288 | n/a | OOF pseudo-labels | n/a | CLAIMED, POSTS:58 |
| Myo team (0.949) | 5-fold EffNet 0.940, 5-fold ResNet 0.940, single fold 0.935; DINOv2 5-fold 0.931 | simple ResNet / EffNet, 24 slices | 5-fold | 3-LLM vote, gold58 0.908 | heavy aug + regularisation "helps a lot", 50 epochs patience 8, best at 28; drops no-info slices | CLAIMED, POSTS:700, 714, 723 |
| cody11null | "a number of teams ... EffNet-B0 to 0.94+ with 5-fold + TTA with 0.88 AUC extraction labels" | EffNet-B0 | 5-fold + TTA | 0.88 labels | n/a | CLAIMED, topusers.txt:73 |
| XuKong Ji | ConvNeXt-T one fold 0.939 (last epoch), 5-fold 0.938 | ImageNet ConvNeXt-T | 1 vs 5 | n/a | n/a | CLAIMED, LEDGER:45 |
| ringbearer (0.948) | 5-fold CoAtNet 0.943, CoAtNet variant 0.940, ConvNeXt 0.937, ResNet 0.933 | mixed | 5-fold | n/a | n/a | CLAIMED, POSTS:726 |
| kalyang17 (0.942) | CoAtNet@384 0.935, ConvNeXt-T 0.930, R50 RadImageNet 2.5D 288 5-fold 0.924 | mixed | 5-fold | own LLM | n/a | CLAIMED, forum 10-02 |
| Tom Aindow (0.949) | (no single number) 392 px, 150 mm, random 32-slice bag per study | n/a | n/a | n/a | n/a | CLAIMED, POSTS:647 |

What the audited ledgers say (MEASURED):
- Tian, one recipe, solo LB: LLM-only all-data CoAtNet-1 0.918 -> Raptor 0.5 teacher 0.927 (+0.009, two seeds) ->
  c03 input (24 triplet slots, 150 mm) 0.932 -> 2-seed pair 0.932 (TIAN:78-99). ResNet-34 0.921 (TIAN:104).
  5 x 80% cross-fit 0.928 = one all-data model 0.927 (TIAN:100). 16 vs 8 epochs at light aug -0.012 OOF (TIAN:67).
- aastik: same 6 ConvNeXt-T 320x24 models 0.927 raw mean (`aastik_ours_script.py:1061`); his 10-model 5-fold x 2
  fleet (fold CVs 0.896-0.915 vs his labels, script line 79) added +0.002 at W .5/.6 (WIN_PLAYBOOK lever 2).
- Ours: crop R50 single fold 0.924; cnx M448 fold-0 0.894; fleet fold-0 not yet read (G2 pending).

**Gold58 does not rank families (MEASURED/CLAIMED mix):** CNNs read ~0.03 lower on gold58 than hybrids for the same
LB. Myo EffNet/ResNet gold 0.89 -> LB 0.940, Myo CoAtNet gold 0.915 -> LB 0.926 (POSTS:706-709, CLAIMED); our crop
R50 gold 0.873 -> 0.924 and Tian CoAtNet c03 gold 0.920 -> 0.932 (MEASURED). Dread picked Raptor checkpoints on gold58
(forum 08-27), so every Raptor-distilled number is gold-inflated. => the fleet's gold58 0.900 (fold-0, epoch 5) is
compatible with any solo from ~0.92 to ~0.95. The G2 solo read is the only instrument.

---

## 2. Gap table: our ConvNeXt fleet vs the top-team pattern, with likely gains

Fleet today (code read in `SP/round3/fleet/`): ConvNeXt-T in22k, 4 views (Ax-FS, Sag-nonFS, Cor-FS, Sag-FS), <=32
stored slices, 24 triplet centres per view, 320 px, 140 mm, 0.5/99.5 window, per-target slice + view attention,
10 epochs cosine, EMA 0.999, drop_path 0.1, aug = rot +-10, scale 0.9-1.1, shift 5%, gain/bias 10%, slice jitter
+-0.5, view dropout 0.1, mirror 0.5; fold 0 of 5 (80% data); targets mix50_logit or pure steven_v4; no TTA.

| # | Gap | Fleet | Top-team pattern | Likely solo gain | Basis |
|---|---|---|---|---|---|
| 1 | Folds / seeds | one fold-0 model | 5-fold or all-data seeds | **+0.005..+0.009** (fold-0 -> 3-4 all-data members) | MEASURED: 1 -> 5 folds +0.009 weak member (LEDGER:36); 2 seeds +0.003 (TIAN:90); cross-fit = all-data (TIAN:100). CLAIMED: Myo 0.935 -> 0.940 |
| 2 | Targets | v4 or flat mix50 | own labels + image/OOF teacher | flat teacher +0.009 (MEASURED, TIAN:83) but student rho .742 -> .835; silent-only fill **+0.003..+0.005** (GUESS from Section 3 at Tian's ~0.28 target-to-LB ratio) | Section 3 |
| 3 | Schedule x augmentation | 10 ep, light-moderate | 25-50 ep, heavy aug + regularisation | **+0.002..+0.006** | CLAIMED Myo (best epoch 28/50), Will 25 -> 50 ep +0.004 (LEDGER:47); MEASURED Tian light aug +0.0039 OOF (sub-floor), long schedule WITHOUT aug -0.012 OOF |
| 4 | TTA | none | mirror and/or slice offset | **+0.001..+0.003** | MEASURED Tian offset TTA +0.003..0.006 single member OOF, +0.0016 in a blend (LEDGER:15); CLAIMED tennogh +0.001 |
| 5 | Sequence context across slices | none (triplet channels + attention pooling) | RSNA standard: CNN -> GRU/LSTM -> pooling (2022 cervical, 2023 abdominal 1st, 2024 lumbar 2nd) | 0..+0.004 | GUESS; Tucker gets 0.94+ with simple pooling; Tian attn vs concat tie (LEDGER:100) |
| 6 | Resolution | 320 | 224-288 (Archit, Tucker, Raymond) | 224 costs ~-0.002 | MEASURED 224 -> 320 +0.002 LB (LEDGER:29); it buys 2x speed for gaps 1 and 3 |
| 7 | Backbone | ConvNeXt-T | CoAtNet / small CNN / ConvNeXt all at 0.93-0.94 in claims | ~0 to +0.006 for CoAtNet, at a rho cost | CLAIMED ringbearer cnx 0.937 vs CoAtNet 0.943, kalyang 0.930 vs 0.935, XuKong cnx 0.939; MEASURED Tian CoAtNet-1 vs DINOv2-S +0.009 |
| 8 | Slice coverage | 24 centres x (c-1,c,c+1) over <=32 slices | 24-32 slices | ~0 | 72 slice reads over a median 30-32 slice series already cover every slice; Dread's coverage gains were at 44-64 slices of a 96 corpus (LEDGER:2-8) |
| 9 | Views | 4 | Tian c03 6 slots (adds T1 slots), Kishor 6 series | 0..+0.002 | GUESS; needs a re-prep, not worth it before 10-12 |
| 10 | Capacity / SWA / pooling swap / label smoothing | n/a | n/a | ~0 | MEASURED nulls: S -> B +0.0011 (LEDGER:32), SWA +0.0006 (LEDGER:48), pooling swap -0.02 (LEDGER:105); soft targets already |

Read: gaps 1 + 3 + 4 are model-side and decorrelation-neutral; gap 2 is the only big measured lever and it trades
against rho; gaps 6-10 are not the 0.93 -> 0.95 difference. Family 1 at 320 px / 10 epochs is too slow
(65-80 min/epoch shared, MEASURED) to buy gaps 1 and 3 together before 10-12.

---

## 3. New measurement: silent-only teacher fill (MEASURED, target level, CPU)

`bigjump/silent_fill_check.py`. Gold = the 58 `original_gold58` rows. rho = mean per-label Spearman of the TARGET
table vs the parent's own Raptor arms (`SP/teacher_v2_out/raptor_proxy_v5v8.csv`) over the 4,349 non-gold train
studies. Teacher logit-matched onto steven_v4 per label (the build_mixSA "logit" map). 1000 paired bootstraps.

| target table | gold58 | vs v4 (SD) | rho vs parent Raptor | cells changed |
|---|---|---|---|---|
| steven_v4 | 0.8927 | 0 | **0.653** | 0 |
| widedense flat 0.5 (= mix50_logit) | 0.9262 | +0.0333 (.0095) | 0.867 | 100% |
| widedense mixSA (.75 silent / .5) | 0.9300 | +0.0371 (.0104) | 0.878 | 100% |
| **widedense silent-only 1.0 / 0** | **0.9099** | **+0.0171 (.0075)** | **0.717** | 27% |
| widedense silent-only 0.75 / 0 | 0.9083 | +0.0156 (.0057) | 0.703 | 27% |
| Raptor-proxy flat 0.5 | 0.9314 | +0.0387 (.0089) | 0.931 | 100% |
| Raptor-proxy silent-only 1.0 / 0 | 0.9127 | +0.0200 (.0068) | 0.747 | 27% |

Read: silent-only keeps ~51% of the flat mix's gold58 gain for ~30% of its rho increase (+0.064 vs +0.214).
Mapping target rho to student rho with Tian's two anchors (target .653 -> student .742; target .931 -> student
.835, linear, GUESS) gives student rho ~.763 for widedense silent-only vs ~.814 for flat mix50. Binormal at an
LLM-only base of 0.930 (MODEL): LLM-only 0.930 @ .76 -> +0.0061; silent-only 0.935 @ .766 -> **+0.0075**; flat
0.939 @ .814 -> +0.0068. Caveats: the teacher was epoch-selected on these 58 by Dread, so every teacher row is
inflated by an unknown amount; the 1/3 student shrink is one pair (Tian). Use the widedense teacher, not the Raptor
proxy (the proxy IS the parent's v5/v8 arms; +0.03 more rho for +0.003 gold).

---

## 4. The proposal: TTR fleet (top-team recipe on the existing fleet store)

**Mechanism.** Spend compute where the 0.94-class teams spend it (more epochs under heavier augmentation, several
members, TTA) instead of on pixels, and take the one label lever that does not raise parent correlation much.

**Exact recipe**
- Data: existing `~/fleet_store` (4,407 npz, 4 views, <=32 slices, 140 mm, 320 px JPEG q92). No re-prep.
- Model: new module `fleet_model_ttr.py` (own model_sha, F2-style, so family-1 checkpoints stay loadable):
  KneeFleetNet unchanged except `encode()` resizes on GPU 320 -> 224 (`F.interpolate(..., antialias=True)` after
  the uint8 -> float step, identical in trainer and worker); ConvNeXt-T `fb_in22k_ft_in1k`; drop_path 0.2, head
  dropout 0.3. LayerNorm backbone, so no BatchNorm hazards.
- Training bag: `--train_slices 16` (16 of 24 slots per view, position ids kept; infer on all 24). Code exists (T6).
- Augmentation (replace `_aug_view` constants, one draw per view): rot +-15 deg, scale 0.85-1.15, shift +-8%;
  gamma 0.75-1.35; gain 0.85-1.15, bias +-15; Gaussian noise sigma 0-6 (p 0.3); 1-3 CoarseDropout holes of 8-16%
  side, same holes on all slices of the view (p 0.3); slice jitter +-1.0; view dropout 0.2; mirror 0.5 (as is).
- Optimiser: AdamW, backbone 1e-4, head 1e-3, wd 0.05, warmup 1 epoch, cosine over **24 epochs**, EMA 0.9995,
  fp16 + GradScaler, bs 2 x accum 4, clip 1.0, per-epoch OOF.
- Labels: **arm A** = pure steven_v4. **arm B** = silent-only fill (v4 on addressed cells; logit-matched widedense
  teacher at weight 1.0 on pilkwang-UNK cells; build with `build_mixSA.py`'s logic, weights 1.0/0.0).
- Selection: arm A by fold-0 OOF vs steven_v4 (EMA and raw). Arm B by OOF vs v4 on ADDRESSED cells only (where A and
  B targets are identical), never by full OOF (biased against B, Tian trap 39).
- Production: all-data `--fold -1 --stop_epoch <A's peak epoch>`: seeds A-s1, A-s2, B-s1. Fleet = raw-probability
  mean within a label arm, rank-mean across arms; mirror TTA in the worker (+1 pass). Optional: add the family-1
  320 px members at equal vote.
- Blend: one global W* from the binormal formula (solo from the fleet-solo read, rho from fold-0 OOF vs the real
  v8 arm); expect W* ~0.40.

**Expected numbers (GUESS unless tagged)**
| stage | solo | rho vs parent | blend on 0.943 (MODEL) |
|---|---|---|---|
| family-1 fold-0 (comparator) | ~0.920 | ~0.80 | ~+0.002 (aastik analog, MEASURED +0.002) |
| TTR fold-0, arm A | ~0.925 (0.915-0.932) | ~0.77 | +0.003..+0.005 |
| TTR production (fold-0 A + B, A-s1, A-s2, B-s1, mirror TTA) | **~0.935 (0.927-0.942)** | 0.77-0.80 | **+0.004..+0.009, centre +0.006** |
| + family-1 members as a second vote | +0.001 own-ensemble | | +0.0005..+0.001 |
Private: same expectation; the binormal over-predicted Tian's Raptor-distilled member by ~0.002 and matched
aastik's fleet, so apply -0.001 to the centre for safety. Displayed 0.950 needs solo >= ~0.937 at rho .78.

**Compute**
- V100: family-1 measured 65-80 min/epoch with two jobs sharing (alone ~36-44, MODEL). TTR = x0.49 (224/320 px)
  x0.67 (16/24 bags) = x0.33 -> **12-15 min/epoch alone** (MODEL). Fold-0 A + B concurrently: 2 x 24 ep = ~11
  alone-hours, ~10 h wall. Three all-data seeds (x1.25 studies): ~20 alone-hours. **Total ~31 V100 alone-hours**,
  done ~10-06 evening if lanes start 10-04 09:05 / 10:45. Displaces crop all-data s0/s1 (crop leg read 0.941-0.942,
  i.e. dead) and family-1 seeds s44/s45. If the loader becomes CPU-bound, do the affine after the 224 resize.
- Kaggle T4: 2 commits (fold-0 pair solo + blend file; production fleet) ~0.6-1.0 T4-h. Slots: 10-05 B: 2 (TTR
  fold-0 solo, P + TTR at W*); 10-07: 2 (production solo, P + production); 10-10/11: 1-2 confirmations.
- 9 h kernel: family-1 member 3-4.5 min on 2xT4 (MODEL); TTR member x0.49, x2 for mirror = 3-4.4 min; 5 members
  15-22 min + shared decode 15-20 min = ~40 min. Parent <= ~6 h (MEASURED: full hidden rerun incl. queue in ~6 h).
  Total <= ~6.8 h vs the 8.6 h wall. Keep per-file fail-loud.

**Risks**
1. 224 px gives back ~0.002 (MEASURED LEDGER:29); the longer schedule must out-earn it.
2. Long schedule over-trains (Tian -0.012 at 16 ep, light aug); heavy aug is the guard, per-epoch OOF picks the stop.
3. Heavy affine/dropout may blur fine structures (menisci, MCL): watch per-label OOF; MCL/LatMen falling > 0.01 =
   back off CoarseDropout first.
4. Arm B: gold58 gain is inflated by Dread's gold selection; rho +~0.02. LB cannot resolve A vs B (solo SD ~.0047),
   so ship both arms in the fleet rather than choose.
5. New model module = new train/serve parity surface (on-GPU resize must be byte-identical code in the worker;
   ship the module with the weights; assert model_sha).
6. The design's measured analog (aastik) tops at +0.002; if TTR fold-0 does not clearly beat family-1 fold-0, the
   route is dead (falsifier below).

**First falsifying experiment (<= 24 h, V100 only, no Kaggle GPU)**
Run TTR fold-0 arm A from ~09:05 IST 10-04 (lane 1, after f0_s42 mix50 ends) on the same folds5 fold 0, with
`--eval_labels llm_labels_v4_blend.csv`, and compare its best-epoch OOF vs steven_v4 (867 studies) with the
family-1 pure-v4 run `f0_s42_v4` (ends ~10:45). Pre-registered: **PASS if TTR >= f0_s42_v4 best + 0.008**
(Tian's one-seed floor, LEDGER:9) with no label below -0.02; **KILL if TTR <= f0_s42_v4 best** (then spend the
V100 on family-1 all-data seeds and EV2S as ENDGAME_V3 plans); in between = keep only as an extra fleet vote.
Read lands ~21:00-23:00 IST 10-04. Arm B (lane 2, from ~10:45) must be within 0.008 of A on addressed-cell OOF.
Gold58 >= 0.87 is a bug guard only.

---

## 5. Runner-ups

1. **Archit round 2 (decorrelated self-teacher).** Finish TTR folds 1-4 (~22 alone-hours), fill silent cells with
   our own 5-fold OOF (weight 0.75-1.0, addressed cells stay v4), retrain all-data. rho stays at the LLM level.
   Archit CLAIMED it "moved right away" (POSTS:294); Tian MEASURED null for flat all-cell self-distillation with an
   OOF weaker than its target (LEDGER:54-55); RSNA 2024 lumbar 2nd place used 0.5 label + 0.5 pseudo-label loss.
   GUESS +0..+0.005. Too slow for 10-12 unless folds start 10-05.
2. **Sequence context head.** One bi-GRU (or 2-layer transformer, dim 256) per view over the 24 slice tokens
   before the per-target attention: the standard RSNA head (2023 abdominal 1st place: 32 triplets + GRU; 2024
   lumbar 2nd: encoder -> LSTM -> attention). GUESS 0..+0.004; needs all 24 slots in training (x1.5 cost) and a
   new parity surface. Add as a fold-0 C arm only if lane time is free.
3. **Small-CNN 5-fold family** (EffNet-B0 or ResNet-34 at 256, frozen BN via `fleet2/code/fleet_model_f2.py`,
   30 epochs, uniform LR 3e-4): the most decorrelated option (CNN on LLM labels, rho maybe .74-.77) and the most
   CLAIMED route to 0.940 (Myo, cody, Tucker), but our audited small-CNN reads are 0.921-0.924. ~15-25 alone-hours.

---

## Sources

Local: POSTS (lines cited), `bigjump/forum_filtered.txt`, `bigjump/topusers.txt`, TIAN (lines cited), LEDGER
(lines cited), `SP/round2/WIN_PLAYBOOK.md` (lever 2, Lane F), `SP/endgame3/ENDGAME_V3.md` (sections 0-2),
`SP/fleet2/FAMILY2.md`, `SP/round3/fleet/fleet_train.py`, `fleet_model.py`, `fleet_common.py`,
`SP/intel/aastik_ours_script.py` (lines 79, 1061-1064), `C:/Users/eeshs/Downloads/rsna-knee/HANDOFF.md` (top).
Web: RSNA 2023 abdominal 1st place (https://github.com/Nischaydnk/RSNA-2023-1st-place-solution): 32 adjacent-slice
triplets at 384, CoaT/EffNetV2-S + GRU, aux segmentation +0.01..0.03; RSNA 2024 lumbar 2nd place
(https://github.com/brendanartley/RSNA-2024-Competition): encoder -> LSTM -> attention pooling, heavy aug, manifold
mixup, 0.5 label + 0.5 pseudo-label loss.
