I have full grounding. Here is the design.

---

# FASTEST KAGGLE-ONLY PATH TO A REAL RSNA SUBMISSION (>0.5)

## The one decision that makes this fit the quota: freeze the backbone, precompute embeddings, train only the head

The binding constraint is NOT the 9h inference cap (huge headroom, see below) — it is the ~30h/wk **train** quota combined with the fact that decoding 570 GB of DICOM inside a training loop is death, and Kaggle notebook output is capped at ~20 GB so you cannot cache full-resolution pixel shards for all 4407 studies.

Solution: run the expensive pass (DICOM decode → resize → CNN/ViT forward) exactly **once** through a **frozen** backbone in a preprocessing notebook, dump per-slice embeddings (a ~0.5 GB file), then train a tiny attention-MIL pooling head on those embeddings in minutes. This is precisely the AnyMC3D result (frozen 2D FM + query-attention pooling + ~1-3M trainable params beats 3D CNNs and 2.5D-BiLSTM at ~10k-volume shoulder-MRI scale, the near-identical regime). Frozen + **mean** pooling leaves ~0.11 AUC on the table; frozen + **learned attention** pooling recovers most of it without ever backpropping into the backbone. LoRA fine-tuning claws back the last bit but needs the backbone in the loop — that is the Week-3 upgrade, not the fastest path.

Consequence: after the one-time embed pass, an epoch over all 4407 studies is **seconds**, so you can run 100 epochs × 5 folds × 2 seeds × several head variants in well under 30 min of GPU. Iteration becomes essentially free within quota.

## Three-notebook architecture

```
Kaggle Dataset A: frozen backbone weights (DINOv2 ViT-S/14, uploaded once, offline)
Kaggle Dataset B: pseudo_labels (rule-v0 now / Qwen when ready) + gold-58 + train_series.csv

NB1  embed_train.ipynb   [T4, run 1-3x total, ~2-4h each]
     comp DICOM -> decode+sample+normalize -> frozen backbone -> emb_train.npy  (~0.5 GB)
     -> published as Kaggle Dataset C

NB2  train_head.ipynb    [T4 or even CPU, minutes]
     load Dataset C + B -> attention-MIL + 12 heads -> select on gold-58
     -> head_weights.pt published as Kaggle Dataset D

NB3  infer.ipynb  (THE SUBMISSION) [T4, internet OFF, <1h of the 9h budget]
     mount comp test -> SAME decode+embed (Dataset A backbone) -> load head (Dataset D)
     -> submission.csv
```

Only NB3 is the scored notebook. NB1/NB2 are dev notebooks whose outputs become datasets NB3 reads offline.

## NB1 — decode, sampling, normalization, backbone

**Backbone (primary):** `DINOv2 ViT-S/14` (dinov2_vits14, 21M params, 384-dim embedding), frozen, fp16, at **224 px** (14-patch → 16×16 tokens; use CLS token). It is small, its frozen features are the strongest general option for radiology structure, and it loads offline from a Kaggle Dataset. Add **one** diversity backbone later for the ensemble: `EfficientNetV2-S` (timm, ImageNet, 256 px) — a second embed pass, another ~2h of quota. Do DINOv2-S alone for the first submission.

**Plane routing is free:** `test_series.csv` ships `Anatomical_Plane` and `Fluid_Sensitive` per series at test time (confirmed in `_data/test_series.csv`). No plane classifier needed. `Fat_Suppression` is byte-identical to `Fluid_Sensitive` — drop it.

**Slice sampling (per series):**
1. Read headers only (`pydicom.dcmread(..., stop_before_pixels=True)`) for every slice — fast (~1000s/s).
2. Sort along through-plane axis: project `ImagePositionPatient` onto the slice normal (cross product of the two `ImageOrientationPatient` vectors); fall back to `InstanceNumber`.
3. Drop outer 12% each end (empty/edge slices). From the central ~76%, uniformly sample K slices.
4. Budget **16 slices per plane** pooled across that plane's series (≈48 slices/study). Cap the ~2% long tail (≥10 series) by sampling, not looping.
5. **2.5D as RGB:** for each sampled center index i, decode the triplet (i−1, i, i+1) from the same series and stack as 3 channels — real adjacent-slice context into a 2D backbone at zero extra model cost. Edges replicate.

**Normalization:** per-series percentile clip [0.5, 99.5] → scale [0,1] → ImageNet mean/std (DINOv2 expects it), applied per channel. (Optional, cheap, defer to W3: laterality flip left→right using `ImageLaterality`/`Laterality` when present; skip when unknown, do not corrupt.)

**Output:** for each study, an array `[n_slices, 384]` plus parallel `plane_id` and `fluid_bit` vectors and a study→rows index. Whole train set ≈ 212k slices × 384 × 2 B ≈ **0.16 GB**. Trivial.

## NB2 — labels, fusion, loss, CV

**Labels to train on:** soft targets, blended, NOT hard Qwen. Rules are high-precision/low-recall (0.65), Qwen is high-recall/low-precision (0.73) and regressed on PF OA. Build the target as a per-label blend:
- Start NOW on `pseudo_labels_rulev0.csv` (exists) to validate the whole pipeline end-to-end before Qwen's 4407 run lands.
- Swap in `pseudo_labels_qwen.csv` when ready; blend `soft = 0.5*rule + 0.5*qwen` (agreement→confident, disagreement→~0.5 soft), and keep rules as a **high-precision override** specifically on PF OA where Qwen regressed. The image pipeline is label-source-agnostic, so this is a one-line swap.
- Gold-58: **never train on it.** Use only for model selection / early-stop / threshold-free eval, with bootstrap 95% CIs (MCL n=9, LatOA n=11, Baker's n=12 have wide CIs — quote them, don't trust the 3rd decimal).

**Fusion (primary — simplest, handles variable series count natively):** a single **gated-attention MIL** (Ilse et al.) over ALL sampled slice tokens of a study. Add a learned `plane` embedding (3) + `fluid` bit embedding to each token so attention can weight by plane/sequence. Masked attention over the variable-length set → one study embedding → 12 independent sigmoid heads. ~1-2M params. (Ablation / W3: per-plane attention pool → concat 3 plane vectors → heads. Analysis 3 shows all 3 planes present in 100% of train studies, but do NOT hard-code that for test — masked set-attention tolerates a missing plane for free.)

**Loss:** soft-target **BCE with mild focal (γ=1.5)**. Because the metric is threshold-free macro-AUC (ranking), aggressive `pos_weight` mostly hurts calibration without helping ranking — use mild weighting derived from the **pseudo-label prevalence over 4407** (much lower than the cherry-picked gold-58 rates), capped at ~3. Priority label-quality effort goes to the rare/noisy trio **MCL, Lateral OA, Fracture** (each 1/12 of macro-AUC, fewest positives, noisiest pseudo-labels).

**Embedding-space augmentation (no re-embed needed):** slice-dropout (randomly drop 20-40% of a study's tokens each step), study-level mixup/CutMix on the pooled embedding (CutMix p≈1.0 was RSNA-2024 5th's single biggest win), light Gaussian token noise. Label smoothing OFF (negative result in prior RSNA). Geometric TTA is optional and would need extra embed views — skip for the first submission.

**CV:** StratifiedGroupKFold grouped by **detected report language** (from `_reports_lang.csv`, the strongest site proxy; en 39% / es 15% / tr 12% / hr 9% / el 7% / de 6% / bg 5% / nl 3% / fr 2%) for honest OOF on pseudo-labels, plus hold one language out as an OOD probe. Language is a train-only CV device, never a model feature (absent at test). Final selection anchor is gold-58 macro-AUC.

**Optimizer:** AdamW, lr 1e-3 (head only), cosine, wd 1e-2, ~50-100 epochs, early-stop on gold-58 macro-AUC. Ensemble the 5 folds × 2 seeds by mean of sigmoids.

## NB3 — the offline inference notebook (structure + timing)

```python
# settings: accelerator = NvidiaTeslaT4 (NOT P100 - sm_60 breaks torch), internet OFF
# 1. glob comp test recursively (mount = /kaggle/input/competitions/<slug>/, extra 'competitions/' prefix)
# 2. read test_series.csv for plane + fluid per series
# 3. per study: read headers -> sort -> sample 16/plane -> decode triplets -> normalize
# 4. frozen DINOv2-S (from Dataset A) -> per-slice embeddings   [batch 256, fp16]
# 5. gated-attention MIL + heads (fold/seed ensemble from Dataset D) -> 12 probs
# 6. write submission.csv in sample_submission column order (labels incl. "Baker's" apostrophe, "Medial Meniscus" spaces)
```

**Timing (test ≈ 1300 studies):** ~1300 × 48 = 62k slice decodes / 4 vCPU ≈ 200-500 s; embed 62k / ~2000 img·s⁻¹ ≈ 30 s; head inference instant. **Total well under 1 hour** even with a 2-backbone ensemble and 3× rotation TTA. The 9h cap is a non-issue for this design — which also positions you well for the efficiency track later.

**Offline gotchas to verify in a throwaway run:** (a) DICOM decoders present in the offline image for the test's transfer syntax — Kaggle usually ships `gdcm` + `pylibjpeg`, but if any series is JPEG-compressed and no decoder exists you must bundle `pylibjpeg`/`pylibjpeg-libjpeg` wheels as a dataset; (b) `torch`/`timm` load DINOv2 from local weights, not `torch.hub` (no internet); (c) each plane branch must tolerate zero series (masked/zero-fill) since the hidden test may violate the "all 3 planes always present" train invariant.

## Quota math (the whole point)

| Item | GPU cost | Notes |
|---|---|---|
| NB1 embed pass, full train, 1 backbone | ~2-4 h | Re-run only if backbone/res/sampling changes |
| NB2 head training (100 ep × 5 folds × 2 seeds) | <0.5 h | Epoch = seconds on 0.16 GB embeddings |
| Second backbone (EffV2-S) embed pass | ~2-4 h | Optional, W3 ensemble |
| NB3 dry-run + real submission | <2 h | Well inside the 9 h cap |
| **Weekly total for the first submission** | **~5-7 h of 30** | Leaves room for 2-3 more backbones |

## Expected AUC and reasoning

| | gold-58 macro AUC | public LB | reasoning |
|---|---|---|---|
| Constant 0.5 | 0.500 | 0.500 | validated |
| Rule-v0 labels (already) | 0.653 | — | text only |
| Qwen labels (teacher) | 0.731 | — | text only, the training ceiling |
| **This image model, first submission** | **0.68 - 0.74** | **0.70 - 0.78** | see below |
| Frontier / published best | ~0.90+ | 0.906 (DINOv2-S) / 0.93+ | needs LoRA + ROI crops + ensemble |

Reasoning: the image head is trained on ~0.73-quality labels, so it inherits that ceiling on the noisy labels, but on gold it can partly exceed the teacher on **anatomical** labels (ACL, meniscus, effusion, Baker's → ~0.75-0.85) because it sees the pathology directly rather than the report's phrasing, while **subtle/rare** labels (Lateral OA, MCL, PF OA, Synovitis → ~0.55-0.68) drag the macro down because frozen general features are weak for cartilage-grade and small-ligament detail and the pseudo-labels there are noisiest. Net gold-58 ~0.70; public LB likely a touch higher and lands squarely in the published baseline band (0.66-0.81). This clears 0.5 decisively and is a legitimate first real submission — not yet the 0.90 frontier, which is the W3-W4 job (LoRA fine-tune the backbone, ROI-crop meniscus/ACL, add EffV2-S/ConvNeXt-T ensemble, confident-learning label cleaning).

## Build order this week

- **Day 1-2:** NB1 on ~100 studies → confirm decode + sampling + normalization visually and that embeddings look sane. Verify offline DICOM decoders in a no-internet dry run. Upload DINOv2-S weights as Dataset A.
- **Day 2-3:** NB1 full 4407 → embeddings Dataset C. NB2 head on **rule-v0** labels (available now) → gold-58 OOF. Expect ~0.62-0.68 (rule ceiling limits it).
- **Day 3-4:** NB3 end-to-end on the public placeholder test → confirm valid `submission.csv` and column order → **first real submission** (should beat 0.5 clearly).
- **Day 4-5:** swap in Qwen pseudo-labels when the 4407 run lands → retrain head (minutes) → resubmit. Expect the jump toward 0.70-0.74. Add EffV2-S embed pass if time.

## The alternative you asked about (pixel 2.5D from scratch) — why it is the W3 upgrade, not the fastest path

An ELNet-style / EfficientNetV2-S 2.5D CNN trained on the actual pixels can generalize better to the multi-site test (learns MRI-specific features), but it needs pixels in the training loop. That forces cached npy shards, and the 20 GB output cap means: 4407 × 48 slices × 160 px × fp16 ≈ 16 GB fits only at 160 px / 16 slices-per-plane with no augmentation headroom, and each training epoch now costs real GPU minutes → far fewer folds/seeds per quota. It is a better ceiling but slower to a first number. Do it in W3 as a LoRA fine-tune of the same DINOv2-S (unfreeze with LoRA r=8, keep the same attention head) rather than from scratch — reuses everything above and is the cleanest path from 0.73 toward 0.85+.

**Files referenced (all absolute):** `C:/Users/eeshs/Downloads/rsna-knee/HANDOFF.md`, `C:/Users/eeshs/Downloads/rsna-knee/strategy/WINNING_STRATEGY.md`, `C:/Users/eeshs/Downloads/rsna-knee/_data/test_series.csv` (confirms plane+fluid metadata ships at test time), `C:/Users/eeshs/Downloads/rsna-knee/_data/sample_submission.csv` (exact 12-column output order, note `Baker's` apostrophe and spaced label names), `C:/Users/eeshs/Downloads/rsna-knee/_data/pseudo_labels_rulev0.csv` (start-now labels), `C:/Users/eeshs/Downloads/rsna-knee/_data/_reports_lang.csv` (language groups for CV). Note: `pseudo_labels_qwen.csv` is not yet on disk — build NB1/NB2/NB3 on rule-v0 first and hot-swap Qwen when its 4407 run completes.