# Big jump, angle: ensemble engineering under the T4 runtime (2026-10-04 ~04:30 IST)

Research only. Nothing was submitted, pushed, versioned, posted or run on the box. Laptop CPU only for three
small closed-form scripts in this folder (`calib_rho.py`, `calib2.py`, `capacity.py`).

Tags: **MEASURED** = our LB reads, ledgers, logs, code facts, or Tian's ledger (another team's own measured runs).
**CLAIMED** = forum or notebook self-reports. **MODEL** = closed-form binormal arithmetic on stated inputs.
**GUESS** = my judgement.

`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`.

---

## 0. Bottom line

1. **The runtime premise of ENDGAME_V3 is stale.** Its 5-member cap, "3 in a combined kernel", W* clipped to
   .20-.35 and the 33-56 min window all assume a parent of 7.7-8.05 h. HANDOFF 10-04 has a MEASURED upper bound:
   rsna-sub-v9 (parent + a zero-cost swap) was submitted 19:33 IST and scored by ~01:50 IST, so the parent
   including queue ran in **<= 6.3 h**. The efficiency LB puts the 0.943 cluster at a median of 6.0 h (PROFILE.md).
   The window to an 8.6 h wall is therefore **>= 138 min**, room for **~20-26 ConvNeXt-T member-passes** at
   g = 4.5-6 min per pass (section 2). The binding resource is now **own-member quality and count**, not T4 time.
2. **Route (this angle):** leave the parent untouched. Append ONE stage that serves **~11 own members from 2
   families with 2-pass TTA** (F1 ConvNeXt-T x7: 6 all-data seeds + the fold-0 run; F2 EfficientNetV2-S x4 all-data),
   with one shared decode and a valve that drops TTA first, then members (family-interleaved prefix).
   Combine with a **probability mean inside each family**, a **rank mean across families**, and **one global outer
   rank weight W_F ~ .40** (more than today's .25-.30), plus the crop leg at W_C .10 only if today's K1 reads support it.
3. **Honest size of the jump. Calibration is the crux.** Every measured "own member + public stack" read sits far
   below the binormal model run at gold58 rho (section 3). The implied effective test rho is **~.08-.10 above the
   gold58 rho** for Tian's distilled member and for our crop members. Under that calibration this route gives
   **+0.002 to +0.006 public, central +0.0035 (0.946-0.947)**. It reaches **>= 0.950 only if the own ensemble reads
   >= ~0.940-0.942 solo at an effective rho <= .84-.86**, which needs the recipe angles. This route is the multiplier
   that turns any own-solo gain into LB: about +0.4 blend per +1 own solo near W .4.
4. **Do not drop or replace parent parts.** There is room without it. Global96 + Repair-v1 (128 min) are MEASURED at
   +0.001 together (Mattia, Speedy Raptors 0.942 to 0.943). Replacing parent arms cannot be validated: the parent arms
   are in-sample on train, gold58 is n=58, and the LB blend SD is .0012 per read. Dropping Repair-v1 (56 min) is
   valve level 3 only.
5. **First falsifier, today:** K1's three reads (crop K=8 solo, W .25, w15) give the first clean calibration point
   for a v4-label own member. If the implied effective rho is >= .90, then no own ensemble <= .938 can lift the parent
   by more than +0.002, and this route is packaging only. Second falsifier: an offline TTA test on the 867 fold-0 val
   studies (section 7).

---

## 1. Inputs (with tags)

| Item | Value | Tag / source |
|---|---|---|
| Parent hidden runtime incl. queue | <= 6.3 h (submit 19:33 -> scored by ~01:50 IST) | MEASURED, HANDOFF 10-04 "LB READ LATENCY" |
| 0.943-cluster implied runtimes | median 6.0 h, IQR 5.8-6.4, p90 7.0 | MEASURED-derived, `SP/parent_speed/PROFILE.md` (efficiency LB) |
| Slow-session penalty | Raptor 4.6 -> 6.5 s/study, ~+40 min | MODEL, PROFILE.md stage table |
| Fleet decode (4 views x <=32 slices, 140 mm, 320 px) | 0.68 s/study aggregate on 4 vCPU = ~15 min at 1,322 | MEASURED, `SP/fleet_prep_out/rsna-fleet-prep-s0.log` |
| ConvNeXt-T 320 x 96 img per member on 2xT4 | 3-4.5 min (timm 3090 bench scaled); Tian's T4 CoAtNet-1@224 GPU share ~0.36-0.4 s/study implies ~4.5-6 min for ours | ESTIMATE / MODEL from MEASURED Tian rates (98 s vs 136 s per 100 studies, experiments.md #25) |
| EV2S / R50 vs ConvNeXt-T inference | 0.91x / 0.88x; channels_last gives EV2S +22%, R50 +14%, ConvNeXt +1% | timm RTX3090 bench, `SP/fleet2/benchmark-infer-amp-*` |
| Crop leg K=8, 3 R50 members | 0.43 s/study on 3 studies, so ~10 min at 1,322 | MEASURED K1 commit log (HANDOFF) |
| fp16 half weights vs autocast on T4 | 2.08x | CLAIMED (/d/743374, Prateek) |
| fp32 vs fp16 inference AUC | -0.00013 | MEASURED (memory, cnx gold74) |
| Prob mean vs rank mean inside a fleet | LB 0.927 vs 0.924; OOF +0.0006/+0.0009 | CLAIMED-with-numbers, `SP/intel/aastik_ours_script.py:1060-1064` |
| Outer operator choice | 6 operators all 0.937 | CLAIMED (/d/740439) |
| 2-seed pair / third member / 5x80% folds vs all-data | +0.003 / +0.001 / 0.928 vs 0.927 (one all-data) | MEASURED (Tian experiments.md #26, #29, #33) |
| Cross-family 0.921 R34 (rho .86) into 0.932 pair | 0.000 | MEASURED (Tian #38) |
| Own 4-family blends (0.940/0.940/0.931/0.928) | 0.947 (+0.007 over best) | CLAIMED (Myo/SpeedSci /d/744511) |
| Slice-offset TTA (-1/0/+1) | +0.003-0.006 single, +0.0016 in a blend | MEASURED (Tian OOF n=882, measured_levers.csv) |
| Aastik fleet serving | offsets [-0.5, 0, 0.5] + mirror TTA, valve drops TTA then members | code fact, aastik_ours_script.py:109-110, 1020-1050 |
| Own fleet fold-0 (mix50) | gold58 .9002 at ep5, val .8742; ep4 rho vs Raptor proxy .857 (v4 run .796, gold58 .836) | MEASURED, HANDOFF D1 provisional |
| Crop R50 member | gold58 .873, LB solo .924 | MEASURED |
| Parent outer blend | per-label rank: (1-w) DINO/A5/Rad chain + w CoAt hybrid, w .60 (ACL .75, MedMen .80, LatMen 1.00, LatOA .75, Fracture .75) | code fact, `SP/parent_speed/parent_cells.py:4721-4741` |

---

## 2. How much fits after the parent

Stage time = 7 min (spawn, load, write) + max(18 min decode, n x g) + 10 min crop. Wall = 8.6 h process age.
From `capacity.py` (n = member-passes that fit):

| P (h) | window (min) | g 3.0 | g 3.75 | g 4.5 | (g 6.0, by hand) |
|---|---|---|---|---|---|
| 6.00 | 156 | 46 | 37 | 30 | 23 |
| **6.30 (MEASURED bound)** | **138** | 40 | 32 | **26** | **20** |
| 6.65 | 117 | 33 | 26 | 22 | 16 |
| 7.00 (slow session) | 96 | 26 | 21 | 17 | 13 |
| 7.70 (old premise) | 54 | 12 | 9 | 8 | 6 |

**Nominal plan = 11 members x 2 passes = 22 passes**, with F2 at 0.91x. At g = 4.5 that is ~94 min of GPU and a
~111 min stage, which fits for P <= 6.75 h. With the TTA valve fired it is ~65 min and fits for P <= 7.5 h.
Dropping to 5 members fits for P <= 8.0 h. So at the measured P this is comfortable when g <= 4.5 and tight when
g = 6. The K2 commit's microbench gives the real g.

**Speedups for OUR stage (the parent is not touched):**

| Lever | Gain | Tag | Risk / rule |
|---|---|---|---|
| One decode feeding both families (fleet2 worker already serves F1 + F2 on the same tensor) | saves a second 15-min decode | code fact (`fleet2/code/fleet_worker.py:104-112`) | none |
| `torch.backends.cudnn.benchmark = True` inside the worker processes only (fixed 48-image chunks) | ~5-15% | GUESS | Our processes only; the parent's D4 assertion lives in its own child process |
| channels_last for F2 (already applied when trained with it) | EV2S +22% | timm bench | none |
| fp16 half weights instead of autocast | 2.08x | CLAIMED | Only after an offline parity test: max abs dlogit and OOF AUC delta < 1e-4 on the 867. Time both modes in the K2 microbench. EV2S has an fp16 overflow history, so F1 first. |
| TTA as the first valve level (drop TTA, then members by a family-interleaved prefix) | keeps the stage inside the wall | design (Aastik's valve order) | The worker's current valve only drops members (`fleet_worker.py` n_keep), so this needs a patch |
| Budget from the 8.6 h wall instead of `left = 9.0 - age - 0.35; assert left > 1800` | avoids throwing away a feasible fleet | code fact, `fleet_blend_cell.py:48-49` | already listed in ENDGAME build item 1 |
| Fewer slices at test | rejected | MEASURED analog: train/serve K mismatch (crop K=24 served vs K=8 trained) read 0.941-0.942 | spend the slack on TTA instead |
| torch.compile / TorchScript | rejected | GUESS: 1-3 min compile per shape, no retry on a hidden run, ~10-20% gain at best | not worth the risk |
| Side-car pre-decode during the CoAt children | ~0 once n x g > 18 min | MODEL (decode is hidden behind GPU-bound passes) | skip |

**Parent speedups (`SP/parent_speed/SPEEDUPS.md`).** Do not ship them by default. The ~26 min they save is not
needed at P <= 6.3 h, and p1's GPU-side identity is still unproven. The exception is a microbench showing g >= 6
min: then adopt p1 alone (15 lines, ~13 min), and only after box parity tier 1 passes.

**Dropping or replacing parent parts: no.**
- Global96 + Repair-v1 cost ~128 min and were worth +0.001 (MEASURED, public).
- The DINO/A5/Rad chain still carries 20-40% weight on 11 of 12 labels in the parent's per-label routing (0 on
  Lateral Meniscus; code fact).
- Tian measured flattening that routing at -0.002, and we measured -0.002 too.
- Any replacement is unvalidatable offline.
- Valve level 3, used only if the hidden outcomes show our stage being cut: skip Repair-v1 (the parent's own
  first-cut child, 56 min).

---

## 3. Blend math, calibrated to what was actually read

The binormal model run at gold58 rho over-predicts every measured "own member into a public stack" read
(`calib_rho.py`, `calib2.py`; reads are treated as +-0.0005 display rounding):

| Measured read | Model at gold58 rho | Implied effective rho | gold58 rho |
|---|---|---|---|
| Tian v09r 0.927 (Raptor-distilled) into the 0.942 fork: W .10 -> .942, W .20 -> .941 | +0.0022 at W .2 | **.935** (range .91-.97) | .835 |
| Our crop (K=24 served) into 0.943: W .15 -> .942, W .25 -> .941 | +0.0016 at W .25 | **.84-.93** (.835 if served solo .905, .911 if .920) | .775-.81 |
| Our mixSA crop: W .25 -> .940 | | .88-.94 | .83-.85 |
| Our cnx 0.894 into 0.943: W .5 -> .930 | | .82-.85 | n/a |
| Aastik ConvNeXt fleet into s74 0.943: W .5/.6 -> .945 | | .76-.89 (.79 if solo .928, .85 if .934) | n/a |
| Tian c03 pair 0.932 + R34 0.921 (rho .86): flat 1/3 -> .932 | .9348 | above .86 | .86 |
| Seeds: 0.927 + 0.927 (rho .952) -> .930 | .9294 | consistent | .952 |

**Reading:** seed averaging behaves as modelled. Cross-family and own-vs-parent blends behave as if rho were
**~.08-.10 higher than measured on gold58**, except Aastik's LLM-label fleet, which sits nearer .80-.85. For the
parent blend I use effective rho = gold58 rho vs the real parent + .07 (central), range +.04 to +.10.

**Lift over the 0.943 parent at the best W (`calib2.py`):**

| own ensemble solo | eff rho .80 | .84 | .88 | .91 |
|---|---|---|---|---|
| 0.933 | W.38 +0.0052 | W.35 +0.0035 | W.29 +0.0018 | W.22 +0.0008 |
| 0.936 | W.42 +0.0063 | W.39 +0.0044 | W.35 +0.0027 | W.30 +0.0015 |
| 0.938 | W.44 +0.0070 | W.42 +0.0052 | W.39 +0.0034 | W.36 +0.0021 |
| 0.940 | W.46 +0.0078 | W.45 +0.0060 | W.44 +0.0041 | W.41 +0.0028 |
| 0.942 | W.49 +0.0087 | W.48 +0.0068 | W.48 +0.0050 | W.47 +0.0036 |
| 0.945 | W.53 +0.0101 | W.53 +0.0083 | W.54 +0.0065 | W.56 +0.0051 |

Consequences:
- W rises with own strength. The current .25-.30 is right for one ~.92-.93 member and too low for an 11-member,
  2-family fleet at ~.937.
- Pre-register one W per structure. The whole W question is worth ~+0.0004 (MODEL, ENDGAME_V3).
- **Operators:**
  - inside a family: probability mean (MEASURED-claimed +0.003 LB over rank);
  - across F1/F2: rank mean (calibration-safe; operator choice measured flat);
  - outer: rank-pct, because the parent outputs ranks.
- No per-label weights: Shiven -0.0008 OOF, and a tie per /d/740439. Aastik's within-fleet per-target weights
  (+0.0034 OOF split-half, CLAIMED) were left at None in his own final, so they are not used.

---

## 4. Expected numbers for the route (public; private central equal, GUESS)

| Step | Own solo | Basis |
|---|---|---|
| F1 single fold-0 (gold58 .900) | 0.929 (0.924-0.934) | GUESS from the gold58 -> LB pairs (.873 -> .924 ours; .9065 -> .930, .9207 -> .932 Tian) |
| F1 x7 (6 all-data seeds + f0), seed rho .95 | +0.004 -> 0.933 | MODEL, matches Tian's MEASURED +0.003 pair / +0.001 third |
| + F2 EV2S x4 (solo within .004 of F1, effective cross-rho .90) | +0.002-0.003 -> 0.935-0.936 | MODEL (`calib2.py`: F2 at .86 +0.005, .90 +0.003, .93 +0.001) |
| + 2-pass TTA (mirror + offset .5) | +0.001-0.002 -> 0.936-0.938 | Tian MEASURED +0.0016 in a blend for offsets; mirror GUESS |
| **Parent + own at W_F ~.40 (eff rho .84-.88)** | **+0.003 to +0.005 -> 0.946-0.948** | MODEL, calibrated |
| + crop leg C at W_C .10 | +0.000-0.001 | MODEL A3 (ENDGAME slots_tables) shrunk by the calibration |
| Stretch: own 0.942 (needs a recipe gain from other angles), eff rho .84 | +0.007 -> 0.950 | MODEL |

So **+0.007 is not reachable on today's member quality.** It needs an own ensemble around 0.940-0.942 or better.
This route does give three things:
- the capacity to carry such an ensemble (the old 5-member cap would waste it);
- the right W;
- +0.003-0.005 on its own, which per the shake-up model (WIN_PLAYBOOK s6) is worth P(silver) ~0.7 at +0.003
  private.

---

## 5. Exact recipe

**Members (train on the box V100, `fleet2/code`):**
- **F1:** `convnext_tiny.fb_in22k_ft_in1k`, KneeFleetNet, 4 views x 24 triplets, 320 px, 140 mm.
  - Store = the 4,407-npz fleet store; gold74 excluded.
  - Labels = the D1 winner.
  - `--fold -1 --epochs 10 --stop_epoch P` with P = the fold-0 peak (5-6).
  - AdamW 1e-4 / 1e-3, EMA .999, fp16, mirror p.5, jitter +-.5.
  - Seeds s43-s48.
  - Also ship BOTH fold-0 runs (mix50 and v4) as members, re-keyed (they collide on `f0_s42`). They cost no V100.
  - Serve the v4 one only if its ep10 gold58 is within .02 of mix50's.
- **F2:** `tf_efficientnetv2_s.in21k_ft_in1k`, frozen BN, lr 2e-4, same store, labels and schedule.
  - Gated by `f2_preflight.py` PASS, G1 and G1c.
  - fold-0 s142, then all-data s143-s146.
  - If F2's fold-0 OOF AUC vs steven_v4 (867) is more than .01 below F1's, drop F2 and run F1 s49/s50 instead.
- **C:** the crop K=8 members already packed (cropaug_f1, cropaug_f2, crop_f1), or the all-data v4 R50 s0/s1 once
  scored.

**Serving cell (port of `fleet_blend_cell.py` + worker patch):**
- one worker per T4, one decode per study for all members;
- pass A = offset 0, no mirror; pass B = `mirror_views` (the exact training op) + offset +0.5;
- valve projected after 64 studies: drop pass B for all members first, then members by a prefix interleaved
  F1/F2/F1...;
- deadline from the 8.6 h wall; per-file fail-loud.

**Combine:**
- F = rank-mean(prob-mean(F1 members), prob-mean(F2 members)) with family weights .6/.4. Use .75/.25 if F2's fold-0
  OOF trails F1's by .005-.01.
- final = rank_pct((1 - W_F - W_C) rank(P) + W_F rank(F) + W_C rank(C)).
- **W_F by the F solo read:** <= .930 -> .30; .935 -> .38; .938 -> .42; >= .941 -> .48. W_C = .10 if K1 shows
  w15 >= .943 and an implied rho <= .88, else 0.

**Files per commit:**
- main (W_F as above), `fleetsolo`, `notta` (pass A only), `wlo` (W_F - .10);
- read only main + fleetsolo by default (slot rule: no W curves).

**Commit ladder:**

| Commit | Members | Reads |
|---|---|---|
| K2 (10-04/05) | both f0 runs, TTA on | G2 solo, G3 blend, microbench g |
| K3 (~10-06) | F1 x3 + F2 f0 | |
| K4 (~10-08) | ~8 members | |
| K5 (10-10/11) | ~11 members | final, read on 10-12 |

---

## 6. Cost

**V100, alone-hours, 2 lanes at ~1.0x aggregate:**
- F1 6 all-data seeds x ~5 h (50 min/epoch x 6) = 30 h;
- F2 fold-0 ~5 h + 4 x ~4.5 h = 23 h;
- offline TTA/serving/parity tests ~3 h;
- box-mimic parent on gold58 ~1 h (rho vs the REAL parent, both out of sample);
- **total ~57 h, done ~10-07/08 if no lane idles.**
- ~90 alone-hours remain to 10-11 for recipe upgrades from other angles. The serving cell is recipe-agnostic within
  the `fleet_common` contract.

**Kaggle T4:** commits only, K2-K5 at ~0.4-0.5 T4-h each, **~1.8 T4-h total**. No training and no local-LB kernel.

**Hidden 9 h kernel:** P <= 6.3 h, plus our stage of ~111 min at g 4.5 with TTA (~65 min without), ends at ~8.2 h.
The valve keeps it under the 8.6 h wall for P up to ~7.5 h.

---

## 7. Risks and the 24 h falsifiers

**Risks:**
- **Effective rho >= .90** (the teacher-correlation trap: Tian's and our crop members sit there). The lift is then
  <= +0.002 whatever we serve. Mitigations:
  - v4-leaning labels;
  - gate on gold58 rho vs the box-mimic REAL parent <= .82;
  - prefer the decorrelated family.
- **g >= 6 min/pass:** TTA falls to the valve. That is a small loss, not an error.
- **Silent valve or skip:** the fleet reads 0.943. The fleetsolo canary catches it.
- **Mirror TTA op mismatch:** the offline test catches it.
- **One-read noise:** blend delta SD .0012 (MODEL), so +0.002 vs +0.004 needs the posterior rule, not the raw read.
- **P bound is a single read:** a slow session adds ~40 min, which the valve absorbs.

**Falsifier 1 (today, no compute): K1 reads at ~11:30 IST.**
- K1 serves crop K=8 v4-label members, with gold58 rho .775-.806 vs the v8 arm.
- Solve `calib2.implied(.943, solo, [(.25, d25), (.15, d15)])`.
- **Kill rule:** implied rho >= .90 means the parent-blend route is capped at ~+0.002 for any own ensemble <= .938.
  Move V100 priority to own-solo strength (other angles) and keep this angle as packaging only.
- **On track:** implied rho <= .86. Then W_F .40-.45 is right.

**Falsifier 2 (by ~12:00 IST, ~45 min of V100 beside training): offline serving test.**
- Take each finished fold-0 F1 checkpoint (raw + EMA) through the worker's exact serving path on the 867 fold-0
  val studies + gold58.
- Variants:
  - no TTA;
  - mirror;
  - offset +-.5;
  - pass A + pass B;
  - per-target max over passes for Fracture / Contusion / menisci / ACL;
  - raw+EMA averaged;
  - the two fold-0 label runs averaged;
  - fp16-half vs autocast parity.
- **Pass:** pass A + B >= +0.002 OOF AUC vs steven_v4 (paired bootstrap P > .9) and not negative on gold58.
- **Fail:** ship no TTA and spend the window on members only.
- The same run gives the mix50 x v4 member correlation (whether serving both is worth it) and the half-precision
  parity.

**Falsifier 3 (K2, ~10-05):**
- fleetsolo < .925 with the main read <= .943: W_F .30 and only all-data seeds of the stronger label variant.
- fleetsolo >= .930: proceed to K3/K4.

---

## 8. Runner-ups

1. **Per-target pooling over TTA passes** (max for focal labels, mean for OA/Effusion/Synovitis) plus raw+EMA twin
   members.
   - Free at inference.
   - CLAIMED +0.003 (public frontier 0.891 -> 0.894 per Aastik's notes); Aastik also found the public per-label
     dict wrong on 2 labels.
   - Decided by falsifier 2.
2. **Parent-lite host (skip Repair-v1, 56 min; then Global96, 72 min) for a heavier own ensemble.**
   - Example: ConvNeXt-T at 384 or ConvNeXt-S members at ~2x per-pass cost.
   - Only if own >= .940 and the pass count needed exceeds ~25.
   - Costs ~0.001 public (MEASURED, Mattia).
3. **Three-way with the crop leg** (C at .10 inside the same stage, ~10 min).
   - +0.0007-0.0015 on top of F (MODEL A3), less after calibration.
   - Keep only if K1's w15 >= .943.
4. **5-fold F1 instead of seeds.** Rejected for compute: 5 x 80% models ~= one all-data model (Tian MEASURED
   0.928 vs 0.927), i.e. ~2.5x the V100 hours per unit of ensemble gain. Keep only fold-0 as the OOF/gating run.
