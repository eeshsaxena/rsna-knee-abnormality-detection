# Dimension E: model-side levers within our compute (2026-10-08 ~21:30 IST)

Read-only research. Nothing on Kaggle was written; no paid calls; no Vast rental. Box: read-only ssh plus
`mkdir ~/research_1008` (empty); nothing on the GPU, no process touched. Light Python on the laptop only.
Tags: **MEASURED** (source given), **MODEL** (computed from measured inputs), **GUESS** (judgement), **CLAIMED** (forum).

**Connectivity note.** The box answered until ~20:40 IST and then timed out (the laptop moved to the 10.19.x network, the same
pattern as 10-07 10:20). So the TTR320 probe is read up to **epoch 6 of 21** (log at 20:39 IST). The ConvNeXt-S/B cache
listing is the 10-07 MEASURED one, not re-run today. Box files I could not re-read today (`~/gold_prep/gates/out/*`) are
cited through HANDOFF/NOVEL_PLAN.

Scripts and outputs (laptop scratchpad `SP2 = C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/01efcc80-088f-477c-8a8a-b504ddf28d6a/scratchpad/e_levers/`):
- `all_logs.txt`: per-epoch raw/ema lines of every `~/fleet_runs/*.log`, pulled 20:40 IST.
- `parse.py`: per-run closing table.
- `traj.py`: 320 probe vs 20 finished TTR runs.
- `lastbest.py` / `.out`: last epoch vs val-best checkpoint, on gold.
- `transfer.py`, `transfer2.py` / `.out`: how a fleet change moves the K9N blend.
- `runtime_e.py` / `.out`: hidden-rerun runtime per candidate kit.

---

## 0. Bottom line

1. **No model-side lever within our compute is worth more than about +.0006 public, blended (MODEL).** It is about +.0001
   to +.0003 after the model's known optimism of ~.0005.
   - The reason is structural. In the K9N blend (P .10 / F .30 / N .60), a fleet SOLO gain transfers at only ~0.3. Fleet
     SOLO +.001 gives blended +.0003; +.003 gives +.0008 to +.0011. [MODEL `transfer.out`, same binormal as
     k9n_weights.py]
   - The LB resolves about .001 to .0015 between correlated blends. So no lever in this dimension can be confirmed on the
     public LB.
2. **The 320 probe is tracking BELOW the G320' bar. P(PASS) is about 5-10% (MODEL).**
   - Epoch 6 raw val vs v4 is .8692 [MEASURED `~/fleet_runs/f0_s1045_ttr320frac75.log`]. 20 finished TTR runs gained
     +.0163 (SD .0031) from epoch 6 to their best; fold-0 runs gained +.0144 (SD .0012) [MEASURED `traj.py`].
   - That predicts a best of **~.8855 (+-.003)**, vs the .8909 bar and the .8900 last-3 bar.
   - On gold, 320 is level with the others (epochs 4-6 mean gold58 .8890 vs .8872-.8931) [MEASURED]. The gate is
     pre-registered on OOF, so it decides.
3. **"K10 with 5x320" is not worth the Vast spend.** Expected value is about P .05-.10 x +.0003 blended = ~+.00002.
   - It also cuts the K10N slow-case margin to the 8.6 h wall to **0.38 h**, vs 0.65 h for K9N [MODEL `runtime_e.out`].
   - V100 hours are not the binding constraint. ~270 V100 h remain to the 10-20 freeze, and the planned jobs need ~60.
   - Keep the ~$9 as the power-cut reserve.
4. **Closed by measurement, never ship:**
   - EMA instead of raw: v4 OOF -.0012 in 11/11 runs; gold ~0.
   - SWA or epoch averaging of saved checkpoints: selection-free = 0.
   - Last-epoch instead of val-best checkpoint: gold74 +.0005, noise; ~0 for K9's members.
   - Multi-scale TTA: -.0005 to -.0007.
   - More same-recipe seeds: ~+.0002 blended.
   - All MEASURED, sec 3.
5. **Default K10 = K9 with the 224 frac75 set replaced by the queued Q4 set** (TTR288 frac75 s1045, 5 folds, ~29 h
   V100, runs automatically after a G320' FAIL).
   - Expected +.0000 to +.0006 blended (MODEL). It keeps the Fracture label lever at 288.
   - It still fits with N: margin 1.04 h central / 0.50 h slow (MODEL).
   - 0 extra effort, already queued.
6. **The only model-side shape that could move the blend by more than ~+.0015** is a strong AND decorrelated 4th leg:
   SOLO >= .940 at r <= .85-.88 vs P, F and N gives +.0015 to +.0021 (MODEL).
   - Nothing we have trained comes close. R50 failed (OOF .8710, rho vs J1 .914); EV2S was weak (.8635) and diverged.
   - Our own CoAtNet, ConvNeXt-S or an OAI-trained fleet would all be P-like or N-like: +.0000 to +.0006 (MODEL).
7. **Best use of idle V100 hours inside this dimension (cheap, zero serving cost):** one all-24-slot training probe at 288
   (8.1 h, flag only). It is the one untested recipe knob with a mechanism: train uses 16 of 24 slots, serving uses 24.
   GUESS -.001 to +.003 SOLO, so -.0003 to +.0009 blended. Label-side retrains (VLM relabel, "fp") pre-empt it whenever
   they are ready.

---

## 1. The transfer rule: why model-side gains are small once N is in (MODEL)

Same macro binormal as `p949_model.py` / `k9n_weights.py`. Inputs:
- P .943, F .944, N .949, P+F at .50 = .946 (MEASURED reads);
- r_PF .945 (MODEL), r_PN .925 (GUESS calibration);
- r_FN scanned .90 / .92 / .94 (GUESS).

The model is optimistic by about +.0005, so subtract that. Source: `transfer.out`, `transfer2.out`.

| change to the fleet leg F | K9N-a (.10/.30/.60) | re-optimised weights | for reference: P+F at .50 |
|---|---|---|---|
| SOLO +.001, same correlations | +.0003 | +.0002 to +.0003 | +.0005 |
| SOLO +.002 | +.0006 | +.0004 to +.0007 | +.0010 |
| SOLO +.003 | +.0008 to +.0009 | +.0007 to +.0011 | +.0015 |
| same SOLO, r vs P and N -.02 | +.0007 | +.0006 to +.0008 | |
| SOLO +.003 but r_FN +.01 (e.g. own OAI-trained fleet) | | +.0006 | |
| SOLO +.004 but r_FN +.02 (e.g. N distilled into the fleet) | | +.0006 | |

| new 4th leg (SOLO, r vs F/P/N) | gain over best 3-leg |
|---|---|
| .940, r .85 / .85 / .85 (hypothetical strong decorrelated family) | **+.0016 to +.0019** |
| .944, r .88 / .88 / .88 | +.0015 to +.0021 |
| .935, r .88 | +.0002 |
| .925, r .84 (EV2S-like) | +.0001 |
| own CoAtNet .940 (r F .90, P .93, N .93) | **+.0000** |
| own CoAtNet .942 (r P .94, N .95) | +.0000 |
| own OAI CoAtNet .946 (r N .95) | +.0004 |
| nartaa's public .945 r224 model (r vs N .95-.97) | +.0000 to +.0002 |

Reading:
- With N at .60 weight, the fleet is a .25-.40 leg. Every "stronger fleet" lever is divided by ~3.
- A new leg must be both strong and decorrelated to matter; strength alone or decorrelation alone does almost nothing.
- **Measured analog that agrees:**
  - K7i (10 members) vs K5 (5 members): SOLO .943 -> .944 and main .946 -> .946 [MEASURED reads,
    PROGRESS_07-10-2026.md l.191-192].
  - K9 vs K8 on gold74: +.0010 (SE .0017) [MEASURED, PROGRESS_08 10:05].

---

## 2. Lever-by-lever (measured evidence, expected SOLO and blended gain, cost, fit)

Blended = K9N world (P/F/N) unless stated. V100 times are at 21 epochs [MODEL `recipe.md` 1.3, re-confirmed by the live
logs]:
- 224 gc0: 3.6 h/fold;
- 288 gc0: 5.8 h/fold (measured 14.5-14.8 min per raw-line epoch);
- 320 bs1 x accum 8: **6.2-6.5 h/fold**. Measured 17.6-17.9 min per epoch incl. eval in the live probe, faster than the
  7.3 h MODEL.

T4 (Kaggle) 224: ~23 min/epoch, 2 folds per 11 h session on 2 GPUs. T4 288 is ~38 min/epoch and 320 ~47-57 min/epoch,
so neither is viable at full length.

### E1. Resolution: 224 vs 288 vs 320

**224 -> 288 (MEASURED).**
- Best raw vs v4, 288 s1044 minus 224 set A s1043, per fold: +.0073, +.0026, +.0028, +.0059, +.0028. Mean **+.0043 OOF**
  [MEASURED, `parse.py` closing lines].
  - Selection-free: fold 0 +.0068, fold 1 +.0025 [MEASURED `~/novel/verify_ensemble/out/v_ens.txt` V3].
  - Seed-matched: about +.0045 on fold 0 [MEASURED NOVEL_PLAN 1.2].
- On expert labels the lead is about 0:
  - T0 - A0 gold74 -.0003 (SE .0046);
  - 288 f0 - mean(224 seeds) gold74 +.0014 (SE .0038);
  - K9 gold read: C (288) - A gold74 +.0007.
  - [MEASURED v_ens.txt V4, verify_recipe 2.12, PROGRESS_08 10:05]
- The gain is broad: 12/12 labels positive selection-free, largest on Effusion, Baker's and Contusion [MEASURED
  verify_recipe 2.5].
- 288 is as parent-like as J1 (.895), with Spearman vs J1 .959 [MEASURED HANDOFF 10-07 00:15]. So it adds strength,
  not diversity.
- **Expected:** SOLO +.000 to +.002 for a 5x288 set vs 5x224; blended +.000 to +.0006 (MODEL).
- The LB answer (K9 SOLO vs K7i SOLO .944) arrives 10-10. Pre-registered: K9 SOLO >= .942 is sane, <= .941 means audit.

**288 -> 320.**
- The acquisition argument (MEASURED `v2_acq.out`): the median acquired resolution is .50 x .63 mm, and 288 (.486 mm)
  already matches it. 320 adds real detail for only 15-39% of series (frequency direction) and 2-7% (phase direction).
- Prior: +.0005 OOF (range -.001 to +.002) [GUESS, verify_recipe].
- CLAIMED for: nartaa's native 320 crop beat 288 (.948 vs .946) on a CoAtNet built for 384. CLAIMED against: Raymond
  "384 -> 288 barely had any effect"; Tian 224 -> 320 +.002 LB, +.0001 gold, "inconclusive".
- **Live probe** (`f0_s1045_ttr320frac75`, frac75 labels, seed 1045, 21 ep, bs1 x accum8, gc0) [MEASURED log, 20:39 IST]:

  | epoch | 320 probe v4 | 224 s1044 mix50 (21 ep) | 224 s1044 frac75 (22 ep) | 288 s1044 mix50 (24 ep) | J1 224 (24 ep) |
  |---|---|---|---|---|---|
  | 4 | .8493 | .8605 | .8571 | .8601 | .8605 |
  | 5 | .8655 | .8687 | .8698 | .8713 | .8647 |
  | 6 | .8692 | .8729 | .8708 | .8754 | .8692 |
  | gold58 at ep 6 | .9024 | .9022 | .9060 | .9023 | .8853 |
  | final best (v4) | ? | .8864 | .8866 | .8909 | .8836 |

  - The crashed first 320 attempt was at the same level at epoch 3 (.8442 vs .8434), so this is not a bad seed
    [MEASURED].
  - **Forecast (MODEL `traj.py`):** the epoch 6 -> best gain over 20 finished TTR runs is +.0163 (SD .0031, range +.0120
    to +.0227); over the 5 fold-0 runs it is +.0144 (SD .0012).
  - That gives a predicted best of **.8855 +- .0031** and a last-3 of .8851.
  - **P(best >= .8909) ~4%, P(last-3 >= .8900) ~6%; P(G320' PASS) ~5-10%,** allowing for 288-type late gains (288 f1
    went from -.005 behind at epoch 6 to +.0024 ahead).
  - The auto gate (`~/queue/gates/g320_gate.py --wait`) decides at ~01:00-01:30 10-09.
- **Expected value of a 5x320 set even if PASS:** SOLO +.000 to +.001 over 5x288, blended +.0000 to +.0003 (MODEL).
  Serving +6 min central / +7.5 min slow vs 5x288 (MODEL `runtime_e.out`).
- **Verdict:** let the pre-registered gate decide; do nothing by hand. On FAIL (expected), Q4 (288 frac75 s1045) starts
  automatically. On PASS, run 320 folds 1-4 on the box only (~25-26 h at the measured 6.2-6.5 h/fold), with no Vast
  (sec 4).

**384 / new store:** stays closed. It needs a re-prep plus a second hidden-set decode (+15-32 min), and the acquisition
data says no information gain [MEASURED v2_acq; NOVEL_PLAN parked list].

### E2. Seeds vs new sets

- Second seed of the same recipe, per fold (v4 OOF): +.0022 over the mean of the two seeds, +.0025 over seed A (SE
  .0004), Spearman A~B .963-.969 [MEASURED `~/novel/ens/out/oof_report.txt` S2].
- At set level on gold, the gain is small:
  - k-curve 5 -> 10 members: gold74 +.0006, gold58 +.0007;
  - A+B sets - A set: gold74 -.0004 (SE .0014) [MEASURED `gold_report.txt`, `v_ens.txt` V4].
- On the LB: K5 -> K7i, SOLO .943 -> .944, main .946 -> .946 [MEASURED].
- **Expected:** a 3rd 5-member same-recipe set gives SOLO +.0002-.0004 and blended +.0001 (MODEL). Filler only.
- **A new set with a label change is worth more than a reseed:**
  - frac75 passed its gate; D - A gold74 +.0003 macro, but Fracture +.0118 [MEASURED K9 gold read].
  - So Q4 (288 + frac75, seed 1045) is a reseed that also carries the label lever. That makes it the right default use
    of the V100 after the probe.
- K10 options under the inference cap [MODEL `runtime_e.out`]:

  | kit (fleet part) | stage central / slow | K10N end central / slow | margin to 8.6 h |
  |---|---|---|---|
  | K9 = 5x224 frac75 + 5x288 mix50 (built) | 58 / 76 min | 7.40 / 7.95 h | 1.20 / 0.65 h |
  | **K10a = 5x288 frac75 (Q4) + 5x288 mix50** | 67 / 85 min | 7.56 / 8.10 h | **1.04 / 0.50 h** |
  | 5x288 + 5x320 | 73 / 92 min | 7.66 / 8.22 h | 0.94 / **0.38 h** |
  | 5x224 + 5x320 | 64 / 84 min | 7.50 / 8.08 h | 1.10 / 0.52 h |
  | 15 members (5x224 + 10x288) | 84 / 108 min | 7.83 / 8.49 h | 0.77 / **0.11 h** (too tight) |

  - The K9 row reproduces K9N_RUNBOOK sec 0.2 exactly (7.40 / 7.95 h).
  - Inputs: MEASURED T4 microbench medians (224 300 ms, 288 466 ms per member per study, A+B) and parent 6.0 h. 320 is
    scaled by pixel count (GUESS).
  - A timeout fails loud: no submission.csv, so the submission errors. Public and private come from the same rerun, so the
    cost is a slot and ~0.5 h quota, not a final pick.

### E3. Epoch averaging / SWA of saved checkpoints (CLOSED, MEASURED)

- Prediction averaging of the last 4/6/8/10 raw epochs vs the val-selected checkpoint, s1043 folds: -.0002 to -.0004 v4
  OOF. Adding EMA epochs: -.0005 to -.0007. 288 f0: -.0004 to -.0011 [MEASURED `~/novel/out/epoch_ens.txt`].
- Selection-free (vs the last epoch instead of the selected best) it is exactly 0: 288 last-3 .8900 = ep24 .8900; J1
  last-3 .8832 = ep24 .8831 [MEASURED verify_recipe 2.6, oof_report S7].
- Why: the cosine schedule runs to zero, so epochs 21-23 agree to the 4th decimal (e.g. A0 .8832 / .8832 / .8831). A
  weight-space SWA of near-identical late weights is about the last checkpoint.
- nartaa's "SWA 3 epochs +.001 public" (CLAIMED) is another recipe (CoAtNet, other schedule) and does not transfer.
- **Expected 0. Cost 0. Do not build.**

### E4. EMA vs raw (CLOSED, MEASURED)

- v4 OOF: EMA ends below raw in 11/11 runs (mean -.0012, range -.0023 to -.0003) [MEASURED `v_ens.txt` V1].
- Gold, last EMA minus best raw over 20 runs: gold74 -.0001 (SD .0020), gold58 +.0004 (SD .0022) [MEASURED
  `lastbest.out`]. That is noise.
- The two short Kaggle E17 runs (swapiy frac75 f0/f1) have EMA lagging by -.003 to -.005, because 0.9995 has a half-life
  of ~0.4 epoch and the schedule is short [MEASURED].
- **We serve raw everywhere already; keep it.** Raw+EMA twins as extra members: -.0002 to -.0006 [MEASURED oof_report S6].

### E5. Checkpoint choice: last epoch vs val-best (tested here, CLOSED)

- Over 20 TTR runs, last minus val-best raw [MEASURED `lastbest.out`]:
  - gold74 +.0005 (SD .0015);
  - gold58 +.0004;
  - v4 -.0004 (selection optimism).
- The positive mean comes from two runs that peaked early at epoch 14 (s1044 f0 mix50 +.0035, s1044 f0 frac75 +.0050).
  Neither is in K9.
- For K9's 288 set C the per-fold differences are +.0009 / +.0008 / .0000 / -.0012 / -.0007, mean about 0.
- Also, the Kaggle runs were fetched without their epoch .pt files. **Expected 0 for K9/K10; no action.**

### E6. TTA variants

- **Served now:** A + B (anatomical mirror + half-slice offset).
  - A+B vs A: val n=120 J1 +.0052 (SE .0036), T288 +.0012 (SE .0017); gold74 T288 +.0030 (SE .0015, P>0 .98) [MEASURED
    `~/novel/ens/out/ms_report.txt`]. Keep.
  - Our mirror already equals nartaa's anatomical mirror (FORUM_DIGEST 10-08 sec 1).
- **Multi-scale passes on top (256/288/320):** -.0005 to -.0007 v4, -.0007 gold74 [MEASURED ms_report]. Closed.
- **All-plane width flip:** nartaa .940 -> .939 (CLAIMED). Skip.
- **A third pass** (e.g. a +-1 slice offset or an intensity jitter), not measured.
  - Passes are r .985-.990 with each other [MEASURED ms_report], so the expected gain is SOLO +.0002-.0005 and blended
    ~+.0001 (MODEL/GUESS).
  - Cost +50% fleet GPU: K9N slow margin 0.65 -> 0.18 h [MODEL `runtime_e.out`].
- **Verdict: no.**

### E7. Slice count (16 train / 24 serve)

- Store: <= 32 slices per series; the 24 serving triplets cover every stored slice [MEASURED verify_recipe 2.7].
  Training draws 16 of 24 slots (slot dropout), which leaves a 16-vs-24 attention mismatch at serving.
- No clean measurement exists. F1 (24 slots) was confounded with light aug and 10 epochs.
  - CLAIMED for more slices: Sanjib 24 -> 32 cached slices "worth more than any label change"; nartaa "fewer windows
    94 -> 24: .940 -> .934" (both at the store/serving side, which we already do densely).
  - CLAIMED against: Berat "slice-count sweeps within noise".
- **Option E7a: `--train_slices 0` (all 24) at 288.**
  - Cost x1.4 per step: 8.1 h fold-0 probe and ~40 h for 5 folds (MODEL `recipe.md` 1.3). Memory 10.0 GB at bs 1 fits.
  - **Zero serving change** (same contract and worker).
  - Expected SOLO -.001 to +.003 (GUESS); blended -.0003 to +.0009 (MODEL from sec 1).
  - Gate already written: last-3 raw >= the same resolution's f0 + .002 (it costs x1.4, so it must clearly win)
    [`recipe.md` 5.2].
  - **Best idle-hour model lever:** it is informative, free at serving and costs one queue line.
- **E7b (32 centres) and E7c (48 source slices):** new tensor contract or re-prep, x1.33 serving. Skip.

### E8. Bigger backbone (ConvNeXt-S/B)

- **Not available offline anywhere we control.**
  - Box: no `convnext_small/base` in `~/.cache/huggingface/hub`, `~/.cache/torch/hub/checkpoints` or `find ~ -maxdepth 6`
    [MEASURED 10-07, verify_recipe 2.8; not re-run today because the box dropped].
  - Kaggle: none of our 21 datasets holds them [MEASURED today, `kaggle datasets list --user eeshsaxena`].
  - Public ImageNet copies exist on Kaggle, e.g. `nguyenncds/convnext-small-fb-in22k-ft-in1k-384` (373 MB) [MEASURED
    `kaggle datasets list -s "convnext small"`]. Public pretrained weights are allowed by the rules (RULES 10-08 item 5).
  - Getting them onto the box needs **Eesh's OK for a download** (session rule). Serving needs no ImageNet weights,
    because our checkpoints carry everything.
- **Cost:**
  - S at 288 = x1.66 compute: 9.7 h/fold, ~48 h for 5 folds (MODEL);
  - serving x1.66 on that set;
  - 5x288 + 5xS288 with N: slow margin **0.15 h** (MODEL `runtime_e.out`). So with P + N in the kernel it only fits as
    a replacement of the *whole* fleet with fewer members, or in a parent-free kernel.
- **Evidence:**
  - CPU fwd+bwd S/T x1.66, B/T x3.05 [MEASURED `actmem.out`].
  - CLAIMED, mostly against: Archit "bigger models weren't helping", Raymond "difference minimal", Berat "within noise".
- **Expected:** SOLO +.000 to +.003 (GUESS), r unchanged (same family, same labels). Blended +.000 to +.0009 (MODEL),
  minus the margin cost.
- **Verdict:** below E7a on gain per hour and per serving minute. Ask Eesh only if E7a and the label levers are flat by
  ~10-14.

### E9. CoAtNet own-recipe (public Raptor recipe)

- **What exists:**
  - `coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k` is cached on the box [MEASURED 10-07 listing].
  - The Raptor inference code is public (dreaddevelopment `knee-mri-twelve-findings-from-a-single-model`: 64 slices in
    5 fixed slots, 336-384 px, 42 eval windows, coatnet_rmlp_2 384, .924 single) [MEASURED, local copy
    `SP/dread_single/`].
  - nartaa's training script is in the public dataset `nartaa/rsna-knee-hpo-assets`; downloading it needs Eesh's OK
    [MEASURED PUBLIC949 sec 3].
  - nartaa's pre-OAI stages were .939 / .940 / .942, reported as table rows only (CLAIMED).
- **Engineering risk:**
  - CoAtNet MBConv stages have BatchNorm, and our frozen-BN F2 path failed twice;
  - EV2S diverged in fp16 at lr 2e-4, and the V100 has no bf16 [MEASURED HANDOFF 10-06].
  - So this means 1-2 days of new trainer work.
- **Cost:** GUESS 6-10 h per all-data model on the V100, plus serving ~25 min (an N-sized stage of its own), which does
  not fit next to K9N.
- **Value:** the parent already holds 3 Raptor CoAtNets plus a CoAt family, and N is a Raptor-recipe CoAtNet. An own
  CoAtNet at .940-.942 with r ~.93-.95 vs P and N adds **+.0000** (MODEL `transfer2.out`).
- **Verdict: skip.** Highest effort for zero modelled gain.

### E10. Inference-time budget (MODEL on MEASURED inputs)

- **Measured:**
  - K9 box A0 fleet stage ~42 min on the loaded box CPU [MEASURED PROGRESS_08 12:23];
  - T4 microbench 224 300 / 426 ms, 288 466 / 532 ms per member per study (median / max, A+B) [MEASURED k8
    `runtime_k8.log`];
  - parent 6.0 h end-to-end (MEASURED as one session's upper bound; an older model said 7.7-8.05 h).
- **K9N:** ends 7.40 h central / 7.95 h slow; margin 1.20 / 0.65 h. The N stage is cut if the parent runs past 6.62 h in a
  slow session [MODEL K9N_RUNBOOK 0.2, reproduced by `runtime_e.out`].
- **The budget rule this implies:** with P + N in the kernel, the fleet holds at most two 5-member sets of 288-or-smaller
  members.
  - Anything that adds ~25+ min (a third set, a third TTA pass, ConvNeXt-S, a 320 set next to 288 in slow sessions)
    pushes the slow margin to <= 0.4 h.
  - Each lever in this dimension is worth <= +.0006 blended, so **no model-side lever justifies eating the N margin.**
- **The serving cost is the parent, not the fleet.**
  - P runs ~6.0 h of the 7.4 h for a .10 weight in K9N-a. The modelled best weights put P at 0-.10, and an F+N kernel
    loses ~.0000-.0004 vs P+F+N [MODEL PUBLIC949 sec 4 table, `transfer.out`].
  - That is a decision for the blend dimension, not this one. It matters here only because a parent-free kernel would
    remove every serving limit above.
  - For model levers that changes little: their blended value stays <= +.0006.

### E11. Out of scope but compared, so the ranking is honest

- **N as a teacher** (fleet labels mixed with N's train-set predictions; N runs on all 4,407 train studies in ~2 h of Kaggle
  T4).
  - Measured analog: v4 -> mix50 (Dread teacher) lifted fleet SOLO .923 -> .939 [MEASURED reads, HANDOFF].
  - In the K9N world, F +.004 with r_FN +.02 is only **+.0006** (MODEL).
  - It also makes the "clean" fleet OAI-derived, which removes the clean final-pick hedge. Label dimension.
- **Own OAI-trained fleet:** F +.003 at r_FN +.01 gives +.0006 (MODEL). 2-4 days of work, and Eesh must download OAI
  himself. Data dimension.
- **nartaa's public .945 r224 OAI model as a 2nd N member:** +.0000 to +.0002 (MODEL; r vs N ~.95-.97). Not worth the
  +8-12 min.

---

## 3. Settled by this read (add to "Settled"; do not re-run)

| item | verdict | evidence |
|---|---|---|
| EMA checkpoint instead of raw | no (-.0012 v4, ~0 gold) | MEASURED `v_ens.txt` V1, `lastbest.out` |
| SWA / epoch-snapshot averaging | 0 selection-free | MEASURED `epoch_ens.txt`, oof_report S7, verify_recipe 2.6 |
| Last-epoch vs val-best checkpoint | 0 for K9 members (+.0005 gold74 pooled, noise) | MEASURED `lastbest.out` |
| Multi-scale / third TTA pass | -.0005 to -.0007 / ~+.0001 for +50% serving | MEASURED ms_report; MODEL |
| Own CoAtNet (Raptor recipe) | +.0000 blended (P- and N-like) | MODEL `transfer2.out` |
| Public .945 r224 as a 2nd N | +.0000 to +.0002 | MODEL |
| 5x320 via Vast | ~+.00002 expected; margin 0.38 h slow | MODEL (sec 4) |

---

## 4. Judgement: "K10 with 5x320" vs alternatives for the Vast spend

| option | P(happens) | blended gain if it happens | expected | cost | K10N slow margin |
|---|---|---|---|---|---|
| K10-320 (5x288 + 5x320), Vast trains f1-f4 in parallel | G320' PASS ~5-10% (MODEL) | +.0000 to +.0003 (MODEL) | **~+.00002** | ~$8.5 Vast (all of it) + transfer of the 11 GB store + data-handling friction; lands ~1 day earlier than the box alone | 0.38 h |
| K10-320 on the box only (if PASS) | ~5-10% | same | ~+.00002 | ~25 V100 h (box measured 6.2-6.5 h/fold), $0 | 0.38 h |
| **K10a = 5x288 frac75 (Q4) + 5x288 mix50** | ~90% (runs on FAIL) | +.0000 to +.0006 (MODEL) | **~+.0002 to +.0003** | 29 V100 h, already queued, $0 | 0.50 h |
| E7a all-24-slot 288 probe, then folds if it passes +.002 | probe 100%, pass GUESS ~25% | -.0003 to +.0009 | ~+.0001 to +.0002 | 8.1 h probe + ~40 h folds, $0 | no change |
| Keep $9 as the power-cut reserve (one lost fold rerun) | power cut recurred once in 8 days (MEASURED 10-08) | protects one queued set landing before 10-20 | n/a | $0 unless used | n/a |

**Verdict:**
- Do **not** spend Vast on 320.
- Weighting the probe by its probability, the expected public value of 5x320 is about +.00002. It also costs the most
  serving time of any K10 option.
- Speed is not the constraint: V100 spare hours to 10-20 are ~270 vs ~60 h of planned jobs (MODEL), and the Saturday
  jump comes from N, not from 320.
- Keep the $9 as the emergency reserve for a fold lost to another lab power cut. That is the one scenario where wall-clock
  speed matters.

---

## 5. Ranked findings (expected public LB gain per unit effort)

| rank | lever | expected blended gain | cost / effort | ready by | action |
|---|---|---|---|---|---|
| 1 | Let G320' decide; no Vast on 320; Q4 (288 frac75 s1045) auto-runs on FAIL -> **K10a** | +.0000 to +.0006 (central ~+.0003) MODEL | 0 effort (queued); 29 V100 h | folds ~10-10 07:00 -> K10a kit ~10-10 evening | build K10a = K9 builder with set D swapped for set E (288 frac75); same tests and A0 |
| 2 | Stop the closed levers from re-entering (EMA, SWA/epoch avg, last-epoch, multi-scale / 3rd TTA, own CoAtNet, 2nd public N) | protects against -.0005 to -.0007 and wasted serving minutes | 0 | now | add sec 3 to HANDOFF "Settled" |
| 3 | **E7a all-24-slot probe at 288** (flag `--train_slices 0`, fold 0, 21 ep, frac75 labels) | -.0003 to +.0009 (GUESS SOLO -.001..+.003) | 8.1 V100 h; 1 queue line; 0 serving | ~10-10 15:00 if queued after Q4 | queue after Q4, behind any label-side retrain |
| 4 | Keep the fleet at 2 sets <= 288 while N is in the kernel (budget rule) | protects the 0.5-0.65 h slow margin | 0 | now | reject any kit whose slow end exceeds ~8.1 h |
| 5 | ConvNeXt-S 288 probe | +.000 to +.0009 minus margin cost | Eesh OK for a ~200-370 MB public download; 9.7 h/fold; x1.66 serving | probe ~10-11 if OK'd | only if E7a and the label levers are flat by ~10-14 |
| 6 | 5x320 set | ~+.00002 expected | ~25 V100 h (box) | only if G320' PASS | box only, never Vast |
| 7 | Same-recipe reseeds | ~+.0001 | 18-29 V100 h | any | filler only |
| 8 | Own CoAtNet recipe | +.0000 | 1-2 days of trainer work, fp16 BN risk, ~25 min serving | n/a | skip |

**Honest total for dimension E:** ranks 1 + 3 together give about **+.0002 to +.0012 blended** (MODEL, before the ~.0005
optimism). That is below LB resolution, and nowhere near the .959 gold line. Bigger moves have to come from the N weight
or final picks, labels (VLM / "fp"), or another strong decorrelated public model, all of which are other dimensions.

---

## 6. Pre-registered decision rules for the next 48 h (model side)

1. **~01:30 10-09:** the auto gate writes `g320.verdict`. FAIL -> Q4 runs, nothing to do. PASS -> 320 f1-f4 run on the
   box (no Vast). In both cases record the 320 best / last-3 / gold58 against the forecast (.8855 +- .0031) as a check of
   the trajectory model.
2. **10-10, K9 SOLO read:** >= .942 = sane. If K9 SOLO >= K7i SOLO (.944), the 288 set transfers, so K10a (2 x 288) is the
   default K10. If it is <= .943, K10 keeps a 224 set (K9 layout) and Q4's set swaps in only if its gold read is
   >= set D - .002.
3. **K10a build gates:** same as K9 (collect bars per fold, gold read tie rule, box A0 40/40, k2_check).
   - Also add a slow-case runtime row: the slow end must be <= 8.1 h.
4. **E7a probe:** queue after Q4. PASS = last-3 raw >= that fold's 288 f0 last-3 + .002 AND gold58 >= .905. FAIL closes
   slot-count work for good.

---

## 7. Sources

- **Box (read 20:00-20:40 IST):**
  - `~/fleet_runs/*.log` (pulled to `SP2/all_logs.txt`);
  - `~/novel/out/epoch_ens.txt`;
  - `~/novel/ens/out/{oof_report,ms_report,gold_report,traj_report,wstar}.txt`;
  - `~/novel/verify_ensemble/out/v_ens.txt`, `~/novel/verify_ensemble/v_model_out.txt`.
- **Local:**
  - old SP `novel/{recipe.md, verify_recipe.md}`;
  - `session_2026-10-08/k9/K9_RUNBOOK.md`, `k9n/K9N_RUNBOOK.md`, `k8/tests/runtime_k8.log`, `p949/p949_model.py`;
  - PROGRESS_07/08, HANDOFF, NOVEL_PLAN_10-07, RESEARCH_TO_TOP_10-06, FORUM_DIGEST_10-08, PUBLIC949_ASSESSMENT,
    RULES_AND_HOST_COMMENT_10-08.
- **Kaggle (read-only CLI):** `datasets list --user eeshsaxena`, `datasets list -s "convnext small"`.
- **New this session:** `SP2/{parse,traj,lastbest,transfer,transfer2,runtime_e}.py` and their `.out` files.
