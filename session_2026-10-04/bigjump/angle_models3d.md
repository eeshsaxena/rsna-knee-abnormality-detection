# Big jump, angle: architectures beyond 2.5D ConvNeXt (3D, video, MIL, MRNet, slice-sequence)

Written 2026-10-04 ~05:00 IST. Research only: nothing submitted, pushed, versioned, posted, or run on the box.
Tags: **MEASURED** (our LB/ledger/runs, or code fact), **CLAIMED** (forum or paper), **MODEL** (arithmetic or
simulation, assumptions stated), **GUESS** (judgement).
`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`.
Scenario table: `SP/bigjump/models3d_binormal.py` -> `models3d_binormal.out`.

---

## 0. Bottom line

**Pick: "V3D", a Kinetics-400 Video Swin-T (torchvision `swin3d_t`) run per view as a 3D volume
(32 slices x 224 px), with per-finding attention MIL over its 3D tokens and the fleet's per-target view
attention, trained on the existing fleet store with pure steven_v4 labels.**

Why this one:
- It is the only design in this angle that is structurally absent from the parent: the parent is all 2D/2.5D
  slice encoders plus slot attention. That covers DINOv2 x20, DINOv3 A5, RadImageNet R50 heads, CoAtNet Raptor
  arms, and CoAt children (MEASURED, `SP/kernels/repro_0946` cells 5/20/26/28). It is also absent from family 1/2.
  V3D mixes slices inside the network (3D shifted-window attention), and its pretraining corpus (Kinetics video)
  is used by no one in the stack. It is the best a-priori decorrelation candidate.
- It is cheap. Forward cost is about 4 x 44 = 176 GMACs per study, against about 873 for a ConvNeXt fleet member
  (9.09 GMACs x 96 images). The V100 cost is about 1/5 of a fleet run and the T4 cost about 1/5 of a fleet member.
  It reuses the fleet's decoded stacks at inference, so it adds no decode (MODEL).
- The strongest literature evidence for "3D beats 2D" is specifically for video-pretrained Swin-T (CLAIMED, Ke et
  al.). The literature evidence against 3D is specifically against native or scratch 3D and medical-3D
  pretraining (CLAIMED, AnyMC3D). That is the line V3D sits on.

**Honest size of the jump. This angle alone does not deliver +0.007.**
- Central case (MODEL with the measured haircut): +0.001..+0.003 on top of P+F.
- Upper tail, if V3D reaches >= 0.935 solo at rho <= 0.75: about +0.005. Probability ~10-15% (GUESS).
- P+F+V reads >= 0.950 public only if the fleet reaches ~0.935 AND V3D >= 0.925 at rho <= 0.72. Probability
  ~15% (GUESS).
- It is still the best decorrelation per V100-hour in this angle: ~12-14 V100-h alone, ~0 extra T4 quota, and
  falsifiable within 24 h.

---

## 1. Evidence

### 1a. What the architecture options look like (sources + tags)

| Option | Evidence | Tag | Verdict |
|---|---|---|---|
| Native 3D CNN, scratch or medical-3D pretraining (3D ResNet/DenseNet/ConvNeXt, MedicalNet, VoCo) | AnyMC3D, 10-task 3D benchmark: 2D-FM slice encoder + attention pooling 0.894 mean AUROC. 3D ResNet 0.809, 3D DenseNet 0.833, 3D ConvNeXt 0.683, MedicalNet 0.783, VoCo 0.793 ([arXiv 2512.12887](https://arxiv.org/abs/2512.12887)) | CLAIMED | Dead: 6-21 points below 2D |
| Video-pretrained 3D (Kinetics) | Kinetics pretraining raised 3D models' mean AUC on RSNA-PE by +0.146 [0.098, 0.241]. Best 3D (Swin-T) beat best 2D by 0.118 on PE and 0.052 on LIDC ([Ke et al., MIDL 2023, PMLR 227:758-774](https://proceedings.mlr.press/v227/ke24a/ke24a.pdf)). Caveat in the same paper: its 2D baselines are weak, and all top-10 RSNA-PE Kaggle subs were 2D | CLAIMED | Only 3D variant with a real case |
| Kinetics init on knee MRI, few-shot | Kinetics/general 3D init *decreased* few-shot knee performance; knee-domain init helped ([MedNet-FS, Sci Rep 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13458554/)) | CLAIMED | Different regime (few-shot) but a warning |
| 3D vs 2D on MRNet | 3D CNN 0.937 vs 2D 0.923 abnormal ([Frontiers 2025](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1590962/full)); architecture unspecified | CLAIMED, weak | Low weight |
| Anisotropy | ~0.3 mm in-plane vs ~3.5 mm between slices "favours 2.5D" (forum /d/741310) | CLAIMED | Why V3D keeps all 16 temporal tokens (no depth downsampling in Swin3D) |
| Early depth fusion (slices as channels) | Tian `v07s` 16-slices-as-channels DINOv2-S: fold-0 OOF **0.7366**, blend **-0.015** despite rho 0.61 (`SP/knee/tian/experiments.md` scoreboard) | MEASURED | Dead. Naive depth fusion collapses strength |
| Slice-sequence heads (GRU/LSTM/transformer over slice features) | AnyMC3D appendix: query attention pooling beat LSTM and transformer fusion. RSNA 2022/2023 winners used 2.5D + LSTM/GRU, but on 96-slice CT ([search summary](https://github.com/TheoViel/kaggle_rsna_abdominal_trauma)). Here: attention vs concat head -0.0024 OOF (tie, Tian). /d/738495 asked, no answers | CLAIMED + MEASURED | Head-only, ~0 decorrelation. Skip |
| Head diversity on one backbone | +0.019 LB at the 0.877 level (rho 0.773, Tian #5). A 0.921 third family at within-class rho 0.86 added **0.000** to the 0.932 c03 pair (Tian #38) | MEASURED | Only pays in a weak regime |
| Backbone swap on a shared 2.5D input | ResNet-34 vs CoAtNet within-class rho **0.860** vs seed twins **0.916** (Tian P-59) | MEASURED | Family 2's ceiling for decorrelation |
| ConvNeXt-T MIL, own pipeline (= our fleet's design) | gold58 0.919, rank corr 0.925 to Raptor arms on gold, "expect >= 0.925" LB (Pand, /d/745091) | CLAIMED | Fleet-like members reach ~0.925-0.935 |
| MRNet per-plane experts + LR fusion | MRNet-style multi-plane ResNet on 13k patients: primary-abnormality AUC 0.898 internal, 0.81-0.85 external ([PMC12508577](https://pmc.ncbi.nlm.nih.gov/articles/PMC12508577/)) | CLAIMED | Less information than joint views. Skip |
| 2D foundation (Pillar-0) | ~0.10 below task-specific 2.5D CNNs (forum /d/738036) | CLAIMED | Kill list already |

### 1b. Our own measured anchors

- Fleet fold-0, mix50, epoch 5: gold58 .9002, val .8742 (HANDOFF). MEASURED.
- D1 epoch 4, raw OOF on the fold-0 867:
  - pure v4: AUCvsV4 **.8401**, rho_raptorproxy **.796**, gold58 .8359.
  - mix50: .8611 / .857, gold58 .8808.
  - So teacher mixing costs about .06 of rho (HANDOFF). MEASURED.
- Crop R50: 0.924 solo. In the stack, crop15 0.942 and crop25 0.941. Tian's 0.927 member: 0.942 at beta .10,
  0.941 at beta .20. MEASURED.
- Fleet decode 0.68 s/study on 4 vCPU. A ConvNeXt member costs 3-4.5 min per ~1,322 studies on 2xT4 (ENDGAME_V3).
  MEASURED / MODEL.
- Series usage, computed today from `train_series.csv`:
  - 5.53 series per study.
  - The fleet's 4 slots use 70% of series.
  - Cor non-FS (T1) is in 3,985/4,407 studies and the fleet ignores it, while the parent uses it (A5/DINO/Raptor
    slots).
  - Side note for the input/fleet angle: a 6-slot store is a strength lever for the fleet, not a decorrelation
    lever. MEASURED.

### 1c. Calibration of the closed-form blend model (MODEL vs MEASURED)

The binormal model over-predicts own-member adds by about **0.003-0.004**:
- crop .924 @ rho .80: predicts 0.9458; measured 0.941-0.942.
- Tian .927 @ .835: predicts 0.9451; measured 0.942 / 0.941.

Reasons:
- rho against the full parent is higher than rho against one arm.
- Shared label noise.
- Non-optimal W.

Every blend number below is quoted raw and with this haircut.

---

## 2. Expected numbers (V3D)

| Quantity | Value | Tag |
|---|---|---|
| fold-0 val AUCvsV4 (1 seed, best epoch) | 0.82-0.845 (family-1 pure v4 at epoch 4: .8401) | GUESS |
| gold58 (reported only) | 0.86-0.90 | GUESS |
| LB solo, 1 all-data seed | 0.905-0.928, central 0.915 | GUESS |
| LB solo, 3-seed mean | central 0.918 (+0.003; Tian 2-seed +0.003 / +0.0015 MEASURED analog) | GUESS + MEASURED analog |
| rho vs parent (within-class) | 0.70-0.78 (fleet ~0.80-0.85; v07s 0.61 at a far weaker solo) | GUESS |
| rho vs fleet | 0.75-0.85 | GUESS |

Blend scenarios (`models3d_binormal.out`, parent 0.943):

| Scenario | Raw model | With the 0.003-0.004 haircut |
|---|---|---|
| P+V, V .918 @ .72 | ~0.948 | ~0.944-0.945 |
| P+V, V .925 @ .72 | 0.9503 | ~0.946-0.947 |
| P+F+V, F .935 @ .80, V .925 @ .72, rFV .75 | 0.9515 vs P+F 0.9489 (+0.0026) | ~0.947-0.948 |
| P+F+V, F .930 @ .82, V .925 @ .72, rFV .75 | 0.9501 vs 0.9465 (+0.0037) | ~0.946-0.947 |
| Upper tail: F .935 @ .80, V .935 @ .75, rFV .75 | 0.9530 (+0.0041) | ~0.949-0.950 |

- Weight W* for V is about 0.20-0.35 in the 3-way blend.
- Floor case: V .915 @ .78, rFV .85. Add <= +0.001 raw, so 0 after the haircut. Drop it.

---

## 3. Exact recipe

**Data**
- The existing fleet store, unchanged: 4 views (AxFS, Sag non-FS, CorFS, SagFS), <= 32 slices per view,
  320 px, 140 mm crop, q92 JPEG, 0.5/99.5 window, `fleet_common.py` contract.
- Splits: same `folds5` fold-0 (867 val). gold58 is never trained on and never used for selection.
- Per view:
  - Resample the stored n slices to T=32 by linspace-nearest. Training adds a slice-centre jitter
    off ~ U(-0.5, 0.5), as in the fleet.
  - Grayscale to 3 channels.
  - GPU resize 320 to 224 (bilinear, antialias).
  - Normalise with `Swin3D_T_Weights.KINETICS400_V1.transforms()` mean/std.
  - An absent view is zeros plus present=0.
- Layout: (B*V, 3, 32, 224, 224).

**Model**
- Backbone: `torchvision.models.video.swin3d_t(weights=KINETICS400_V1, stochastic_depth_prob=0.1)`, 28.2M params.
  Weights are 121.5 MB and must be cached on the box and in the members dataset.
  - Use `patch_embed -> features -> norm` and drop the K400 head.
  - Tokens: (B*V, 16, 7, 7, 768). Patch (2,4,4) keeps 16 depth tokens; Swin3D never downsamples depth.
- Head: copy fleet `KneeFleetNet` from `proj` onward.
  - Linear 768->256, LayerNorm, GELU.
  - Plus learned temporal (V,16) and spatial (49) position embeddings, plus the view embedding.
  - 12 per-target queries attend over the 784 tokens per view.
  - Per-target view attention with presence mask and `vbias`.
  - Per-target linear heads.
  - Sagittal laterality lives on the depth axis, so the temporal position embedding must be per view.

**Training**
- Targets: soft BCE on **pure `llm_labels_v4_blend.csv`**. Not mix50: teacher mixing measured +.06 rho to Raptor.
- Optimiser: AdamW, wd 0.05.
  - Backbone lr 5e-5 with per-stage decay 0.85. Head lr 1e-3.
  - 1-epoch warmup, then cosine over **12 epochs**.
  - EMA 0.999, grad clip 1.0.
- Precision: fp16 autocast + GradScaler. Swin3D is LayerNorm-only, so there is no EV2S-style BN overflow.
- Memory: `torch.utils.checkpoint` per Swin stage. Batch 1 study (4 clips) x accum 8.
- Augmentation, the same per clip across slices:
  - `mirror_views` p .5 (sagittal = reverse the depth axis; ax/cor = flip W).
  - Random-resized-crop scale .85-1.
  - Rotation ±8°.
  - Gamma .8-1.25.
- Epoch selection: fold-0 val AUCvsV4 curve only. gold58 is logged, never selected on.
- Output: an OOF csv in the fleet format, so `fleet_g1b.py` and `fleet2/code/g1c_corr.py` run unchanged.
- Seeds:
  - `v3d_f0_s700`: the fold-0 gate.
  - Then all-data `s701`, `s702`, `s703` at the chosen epoch count with `--select last`.
- Never select on gold58 (caoyupeng trap: gold .945 -> LB .910).

**Inference**
- Add a V3D stage to `fleet_worker.py`. It consumes the already-decoded <= 32-slice stacks (no second decode).
- Weights: `.half()`, channels-first 5D.
- Fail-loud fp16 parity probe on 3 studies: fp16 vs fp32 max |dp| < 1e-2. Never use bf16 (Kishor: 0.851 -> 0.710).
- Own members: raw-probability mean. Into the parent: global rank blend at one W.
- Outputs: `submission_pfv.csv` (P+F+V at W*), `submission_vsolo.csv`, plus the existing files.

---

## 4. Cost and runtime fit

**V100 (MODEL)**
- Basis: fwd + bwd + checkpoint recompute is about 1.4 TFLOP/study, against about 7 for a fleet member at 320.
  The fleet takes ~40 min/epoch alone, so V3D should take ~8-12 min/epoch alone and ~2x that while sharing.
- Preflight: 0.3 h.
- fold-0, 12 epochs: 2-3 h alone (4-6 h shared).
- 3 all-data seeds: 3 x 2.5-3.5 h.
- **Total ~11-14 V100-h alone (~20-25 shared).** That fits easily before 10-12, beside family 1/2.

**Kaggle T4**
- About 1/5 of a ConvNeXt member: <= 1-2 min per seed per ~1,322 studies on 2xT4, with no added decode.
  3 seeds <= 6 min. MODEL; the K-commit microbench replaces it.
- The parent's hidden runtime is <= ~6 h (MEASURED read latency), so the window to the 8.6 h wall is >= ~2.5 h.
  V3D fits even beside 5 fleet members.
- Quota: 0 extra commits. It piggybacks on the K2/K4 commits, adding ~2-3 min each plus the parity probe.

---

## 5. Risks

1. **Strength collapse.** 3D on anisotropic MRI with a video prior may underfit: native 3D is -6..-21 points
   (AnyMC3D), and early depth fusion collapsed (v07s). This is the main risk, and the 24 h gate targets it.
2. **Domain gap of the Kinetics prior.** Grayscale input, 3-4 mm slice steps vs 40 ms frames. MedNet-FS saw
   Kinetics init hurt in few-shot knee MRI.
3. **Inflated rho on fold-0.** Parent arms trained on the fold-0 studies, so rho on them is inflated. Use the gap
   vs family-1 on the same 867 only. Make W decisions on gold58 with the box-mimic real-parent predictions
   (direction only; gold58 SD is large).
4. **GPU contention.** Memory and time while sharing the V100. Run the preflight before launch and keep peak
   <= 7 GB with checkpointing.
5. **Library versions.** torchvision >= 0.15 for `swin3d_t` (box env and Kaggle image). Verify in the preflight.
6. **Positional/laterality bug.** Sagittal mirror must reverse depth, matching the fleet's `mirror_views`.
   A unit test on one study: mirrored logits differ, and labels are unchanged.

---

## 6. First falsification experiment (<= 24 h, starts when the V100 frees ~10:45 IST)

1. **Preflight (20 min).**
   - One study fwd/bwd in fp16: finite loss, s/step, peak memory.
   - Mirror unit test.
   - 320->224 resize parity.
2. **Train `v3d_f0_s700`** (12-epoch schedule), and read at epoch 6 (~1.5-3 h alone).
3. **Run gates on fold-0 867.**
   - `fleet_g1b.py`: AUCvsV4 plus rho vs the Raptor proxy, maxspan_v5, and native384_v8.
   - `g1c_corr.py`: within-class Spearman vs family-1 pure-v4 OOF at the same epoch.
4. **Pre-registered reads.**
   - **K-bug:** epoch-2 AUCvsV4 < 0.78. Stop and debug.
   - **KILL-strength:** best AUCvsV4 through epoch 6 < **0.815** (family-1 v4 .8401 - .025). Even rho .65 cannot
     pay at that solo after the haircut.
   - **KILL-corr:** rho_raptorproxy >= **0.79** (no better than family-1 v4's .796), or within-class vs family 1
     >= **0.88**. That makes it just another 2D-like member.
   - **PASS:** AUCvsV4 >= **0.830** AND rho_raptorproxy <= **0.75** AND within-class vs F1 <= **0.85**.
   - On PASS:
     - Launch `s701` all-data immediately.
     - Run the box-mimic gold58 check: P+F+V best-W lift vs P+F >= +0.002 (direction only).
     - Put V3D files into the next K commit.
5. **LB falsification (first read, ~10-06).**
   - `submission_vsolo.csv` <= 0.910: DROP.
   - `submission_pfv.csv` <= the P+F read: DROP.

---

## 7. Runner-ups (brief)

**R1. Pseudo-3D 2D backbone (residual temporal shift, TSM; 1/8 channels shifted along the 24-triplet axis
inside every block).**
- Uses family 2's EV2S/R50 trainer and keeps ImageNet weights.
- Strength ≈ 2.5D ± 0.003. rho vs F1 ~0.88-0.90. GUESS, from TSM's +2-3% on appearance-dominated Kinetics and
  AlignShift's DeepLesion gains, CLAIMED ([AlignShift](https://arxiv.org/abs/2005.01969)).
- Cost = a family-2 fold-0 (~7 h alone).
- Use only if V3D fails its gate, as the family-2 variant.

**R2. Second 3D family: MViTv2-S (K400 80.8%, 64 GFLOPs) or X3D-M.**
- Only if V3D passes and its seeds saturate (seed twins rho > 0.92).
- About 1.5x V3D's cost.

**R3. Slice-sequence head (bi-GRU / 2-layer transformer over the 24 slice tokens) on the fleet.**
- Near-zero cost if trained jointly.
- But attention pooling >= LSTM/transformer (AnyMC3D), and the pooling choice was a tie (-0.0024, Tian).
- Expect 0. Skip unless free.

**Do not spend on:**
- Native or medical-3D-pretrained 3D CNNs (AnyMC3D).
- Slices-as-channels (v07s).
- MRNet per-plane experts.
- Pillar-0 / foundation 3D.

---

## 8. Sources

- AnyMC3D, "Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification": https://arxiv.org/abs/2512.12887 (HTML: https://arxiv.org/html/2512.12887)
- Ke et al., "Video Pretraining Advances 3D Deep Learning on Chest CT Tasks", MIDL 2023: https://proceedings.mlr.press/v227/ke24a/ke24a.pdf
- MedNet-FS (knee, few-shot, Kinetics init): https://pmc.ncbi.nlm.nih.gov/articles/PMC13458554/
- 3D vs 2D on MRNet (Frontiers 2025): https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1590962/full
- Multi-task nine-abnormality knee system (MRNet-style multi-plane ResNet): https://pmc.ncbi.nlm.nih.gov/articles/PMC12508577/
- AlignShift (TSM-style shifts on anisotropic CT): https://arxiv.org/abs/2005.01969
- RSNA 2023 ATD 2nd place repo (2.5D + RNN pattern): https://github.com/TheoViel/kaggle_rsna_abdominal_trauma
- torchvision video weights (Swin3D_T K400 77.7%, 28.2M, 43.88 GFLOPS): https://docs.pytorch.org/vision/main/models.html
- Forum (CLAIMED), from `SP/intel/knee_topics.json` / `knee_comments.json`: /d/741310 (anisotropy, 3-14 series), /d/738495 (GRU question, no answers), /d/745091 (Pand ConvNeXt-T MIL), /d/738036 (Pillar-0).
- Ledgers (MEASURED): `SP/knee/tian/experiments.md` (v07s, P-59, #5, #38, P-40), `C:/Users/eeshs/Downloads/rsna-knee/HANDOFF.md` (D1, fleet epoch 5, LB reads), `SP/endgame3/ENDGAME_V3.md` (decode, member cost), `SP/kernels/repro_0946/repro0946.ipynb` (parent architecture).
