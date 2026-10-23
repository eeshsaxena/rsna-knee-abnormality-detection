# RSNA Knee Abnormality Detection (2026) — Technical Research Brief

**Scope:** multi-label abnormality classification on multi-plane/multi-sequence knee MRI exams; weak labels from multilingual radiology reports; gold-label evaluation; ~5k train exams / ~570 GB; efficiency prize track.
**Compiled:** from live web research (arXiv, GitHub, Kaggle/RSNA pages). Uncertain claims marked **[UNCERTAIN]**.

---

## 0. TL;DR — Ranked highest-leverage techniques for THIS competition

1. **2.5D slice-stack + per-slice 2D encoder + learned slice aggregation (attention pooling / BiLSTM), late-fused across planes & sequences.** Consensus winning recipe across MRNet literature, RSNA 2023/2024 winners, and the 2025 AnyMC3D benchmark — decisively beats from-scratch 3D CNNs at ~5k-exam scale. (Sections 1, 2, 4)
2. **Self-supervised foundation backbone (DINOv2/DINOv3 ViT-B/L, or ConvNeXt/EfficientNetV2 as strong CNN fallback), fine-tuned — not frozen-linear-probed.** Adaptation method matters more than backbone choice (delta AUC up to 0.11 from adaptation alone). Domain-continued SSL pretraining on the 570 GB of unlabeled knee slices is the single biggest differentiator available. (Section 4)
3. **Treat label noise as a first-class problem:** LLM-extracted soft labels + confident-learning sample filtering + two-stage training (train for AUC first, then fine-tune head with metric-weighted loss). Directly from RSNA 2024 2nd/5th place playbooks. (Sections 2, 3)
4. **Coarse-to-fine / localize-then-classify.** RSNA 2024 (1st, 2nd, 5th) and RSNA 2025 (1st) winners localize anatomy first, then classify crops. For knee: meniscus/ACL ROI cropping via a small heatmap/segmentation model likely worth several points, especially for cartilage/meniscus labels. (Section 2)
5. **Ensemble diversity + TTA:** multi-backbone x multi-fold x rotation TTA; per-series models combined by logistic regression or attention fusion (MRNet-style). (Sections 1, 2)
6. **Efficiency track:** single 2.5D student (EfficientNetV2-S / ConvNeXt-T / distilled ViT-S) distilled from the big ensemble, FP16/INT8 via ONNX -> TensorRT. RSNA 2025 1st place itself shipped TensorRT FP16 engines. (Section 5)

---

## 1. MRNet & knee MRI classification literature

**MRNet (Bien et al., PLOS Medicine 2018)** — the reference point. 1,370 Stanford knee MRI exams; per-plane AlexNet over all slices -> global max-pool over slices -> FC; one model per task (abnormal/ACL/meniscus) x plane (axial/coronal/sagittal) = 9 models; plane outputs combined by logistic regression. AUC: abnormal 0.937, ACL 0.965, meniscus 0.847. External validation (Croatia, T1): 0.824 zero-transfer -> 0.911 after retraining — **sequence/scanner shift is real; expect multi-site multilingual RSNA data to punish single-protocol overfitting.** Class imbalance: inverse-frequency loss weighting; augmentation = rotation +/-25 deg, shift +/-25 px, hflip.
Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/

**Follow-up SOTA on MRNet:**
- **ELNet (Tsai et al., 2020)** — lightweight (~0.2M params) from-scratch CNN with multi-slice normalization + BlurPool, single imaging stack; ACL 0.960 / abnormal 0.941 / meniscus 0.904 — beats MRNet with ~1/1000 the parameters. Best plane differs per task (coronal for meniscus, axial for ACL). **Key lesson for the efficiency track: small from-scratch 2.5D CNNs are very strong on knee MRI.**
  Source: https://arxiv.org/html/2005.02706v3
- **Azcona et al. 2020** — ResNet-18/50/152 transfer learning beats AlexNet MRNet; per-plane-per-task models beat multi-plane-joint and multi-task models (negative transfer observed); interpolation to fixed slice count works nearly as well and is much faster.
  Source: https://arxiv.org/pdf/2010.01947 ; https://github.com/dazcona/mrnet
- **MFF9.8Net (2021)** — multi-view ResNet18 with log-sum-exp slice pooling: ACL 0.977 on unseen test. Pooling function choice (max vs LSE vs attention) matters.
  Source: https://arxiv.org/pdf/2108.08136
- **SSL ViT for ACL (2022)** — self-supervised pretraining on knee slices + transformer encoder + oversampling minority class + 5-seed ensembling. SSL pretraining on unlabeled knee MRI helps at small scale.
  Source: https://arxiv.org/pdf/2208.13923
- Backbone comparison (MDPI 2022): VGG16 > AlexNet > ResNet > EfficientNet per-plane on MRNet; sagittal best for abnormality, axial/coronal contribute for ACL/meniscus — **no single plane suffices; fuse all three.**
  Source: https://www.mdpi.com/2504-2289/6/3/69
- KneeXNet (2025) claims 0.985 AUC with 3D multi-scale fusion — **[UNCERTAIN]** (single-institution, likely validation-protocol differences).

**What works best at ~5k-exam scale (synthesis):** per-slice 2D encoder (pretrained) -> slice aggregation (attention/LSE pooling or BiLSTM) -> per-plane models -> plane fusion (logreg or attention). 3D CNNs from scratch underperform at this scale; pure 2D single-slice loses context. 2.5D (3-9 adjacent slices as channels) is the sweet spot used by every recent RSNA winner.

---

## 2. Winning solutions: RSNA 2024 Lumbar Spine & RSNA 2025 Aneurysm

### RSNA 2024 Lumbar Spine Degenerative Classification (multi-sequence MRI, 3-class severity x 25 labels — closest analog)

**1st place (Avengers / NANACHI):** localize-then-classify. 3D localization of level-wise coordinates -> multi-view 2.5D crops (sagittal + axial stacks) -> 2D backbones (**ConvNeXt-S, EfficientNetV2-S**) -> **BiLSTM + attention-based MIL** fusing slices and views -> auxiliary attention losses -> ensemble across backbones and folds.
Source (via AnyMC3D Appendix E): https://arxiv.org/html/2512.12887v3 ; writeup: https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution

**2nd place (IanPan-Kevin-Yuji-Bartley; brendanartley part):** sagittal-only 2-stage. Stage 1: 2D backbone + attention head predicts disc coordinates from 8 middle frames (99.1% within 5 px). Stage 2: middle 24 frames -> encoder -> LSTM -> attention pooling. Tricks: **pseudo-labeling (0.5*label + 0.5*pseudolabel loss)**, **confident learning** (OOF ensemble; drop samples where prediction strongly disagrees with given label), manifold mixup on embeddings, sequence-order flip, heavy aug, **9-rotation TTA**. Trained on 1x A100 40GB, ~24 h.
Source: https://github.com/brendanartley/RSNA-2024-Competition

**5th place (siwooyong):** Stage 1 heatmap keypoint detection with **Gaussian-expanded labels** (fixes class imbalance, speeds convergence); 2D U-Net + sequence model > 3D U-Net; EfficientNet-B5 best for stage 1. Stage 2: 2.5D crops (z +/-2, xy +/-32 -> 5x128x128), CNN+RNN with **level-wise sequence modeling** (model co-occurring labels at same level — big gain). **Two-step training: (1) full training with plain loss maximizing AUC; (2) freeze backbone, train head with metric-weighted loss.** **CutMix p=1.0 was the single most important augmentation.** Ensembles: backbone diversity (convnext_small/tiny, caformer_s18, pvt_v2_b3) + TTA-like stage1 x stage2 cross-product ensemble. Did NOT work: label smoothing, bigger stage-2 backbones, mask-as-channel.
Source: https://github.com/siwooyong/RSNA-2024-Lumbar-Spine-Degenerative-Classification

**7th place (hengck23 + lhwcv):** one-stage PVTv2-B4 + 3D decoder head; two-stage sibling models; trained on 2x A6000 / RTX 4090.
Sources: https://github.com/hengck23/solution-rsna-2024-lumbar-spine ; https://github.com/lhwcv/solution-rsna-2024-lumbar-spine

**9th place (adamnarai):** DeepLabV3+ Gaussian-heatmap keypoints -> level-wise ROI crops -> 2.5D + GRU head, ResNet18/Swin-Tiny/ConvNeXt-Nano ensemble.
Source: https://github.com/adamnarai/kaggle-rsna-2024

**Class imbalance (severity):** weighted CE matching metric weights (1/2/4 for normal/moderate/severe) applied only in step 2; Gaussian label expansion; oversampling.

### RSNA 2025 Intracranial Aneurysm Detection (3D multi-modal series classification)

**1st place (tomoon33):** coarse-to-fine "find vessels first, then find aneurysms on vessels." nnU-Net ResEnc M: (1) low-res (1.0 mm) vessel-grouping model (3 classes) + DBSCAN denoise -> 140 mm^3 ROI crop; (2-3) high-res (0.4-0.8 mm) vessel segmentation with auxiliary task reconstructing a sphere around each aneurysm point (forces local feature learning); (4) ROI classifier for 13 locations + presence. Inference converted to **ONNX -> TensorRT FP16 engines** (directly relevant to efficiency track). Single RTX 4090 24 GB for development.
Sources: https://github.com/uchiyama33/rsna2025_1st_place ; summary: https://blog.csdn.net/csdn_xmj/article/details/155648215

**7th place (MIC-DKFZ):** pure nnU-Net ResEnc pipeline, 200x160x160 mm ROI crop, 3d_fullres bs32, 4x A100 40 GB, 4.5 days — brute-force 3D works but is compute-hungry; Educational Merit Award.
Source: https://github.com/MIC-DKFZ/kaggle-rsna-intracranial-aneurysm-detection-2025-solution

**8th place (KonradHabel):** 2-stage feature reuse: stage-1 slice classifier trained on positive spots + **hard-negative mining from OOF false positives**; stage-2 transformer over extracted per-slice features for whole-sequence classification. Cheap and effective — good pattern for weak slice-level supervision.
Source: https://github.com/KonradHabel/rsna

**9th place (tom99763):** YOLO-11m / YOLO+EfficientNetV2-S 2.5D detector + 3D CenterNet with vessel-seg auxiliary branch (seg masks as prior).
Source: https://github.com/tom99763/9th-place-solution-RSNA-IAD

**Class imbalance (aneurysm):** focal loss (mixed ~0.3), class weight ~3.0, hard-negative spot sampling, 10-fold CV.
Source: https://github.com/NeoGreenCode/AneurysmDetection

**Cross-competition constants:** 2.5D > 3D except with nnU-Net-scale compute; attention/LSTM slice fusion; Gaussian/soft auxiliary targets; two-stage localize -> classify; confident-learning label cleaning; heavy geometric aug + CutMix; backbone-diverse fold ensembles; rotation TTA; A100-class local training, Kaggle GPU only for inference notebooks.

---

## 3. Weak-label learning from multilingual radiology reports

**Label extraction:**
- Open-source LLMs beat the CheXpert rule-based labeler by wide margins (95% vs 51% sensitivity on fracture; Llama-3-8B/Phi-3/Zephyr on 227k MIMIC reports, zero-shot, temp=0, JSON + rule-based post-processing). **Image classifiers trained on noisy labels are remarkably tolerant to label noise — but LLM labels still beat rule-based labels downstream.**
  Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC12920854/
- Llama-2-70B (8-bit, local) structures English AND German reports at ~human accuracy (MCC 0.75 EN / 0.66 DE) with English prompts — **a strong multilingual LLM with an English prompt template handles non-English reports directly; no translate-then-extract needed.** Constrained generation guarantees valid structured output.
  Source: https://link.springer.com/article/10.1007/s00330-024-11074-y
- Spinal MRI report labeling with Zephyr-7B / Llama-3-8B (summarize-then-label per condition) matched/exceeded GPT-4 F1; classifiers trained on these labels matched classifiers trained on expert image labels (MICCAI 2024) — **directly transferable recipe: per-label "summarize with condition in mind -> binary label", class name + definition only, no fine-tuning.**
  Source: https://papers.miccai.org/miccai-2024/paper/1510_paper.pdf
- 2025 Kaggle meta-trend: Qwen2.5/Qwen3 dominated all text-processing competitions; weight quantization at inference standard. **Practical pick: Qwen2.5-14B/32B-Instruct (or Qwen3) locally with vLLM for multilingual report -> soft label extraction; validate on a few hundred manually read reports first.**
  Source: https://mlcontests.com/state-of-machine-learning-competitions-2025/
- Multilingual CheXbert finding: multilingual BERT gave no real gain over English BERT on Spanish reports; fine-tune on auto-labels then human labels works best — argues for LLM extraction + small human-verified calibration set.
  Source: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/reports/final_summaries/summary075.html

**Training with noisy report labels vs small gold set:**
- **Confident learning / OOF filtering** (RSNA 2024 2nd place): train ensemble, drop or down-weight samples where OOF prediction strongly disagrees with the report label.
- **Two-step training** (RSNA 2024 5th place): plain-loss pretraining for AUC -> frozen-backbone head fine-tune with metric-weighted loss. Prevents overfitting to weighted/noisy labels.
- **Pseudo-labeling** with 0.5/0.5 loss mix (RSNA 2024 2nd place).
- Standard noisy-label toolbox with medical-imaging evidence: DivideMix, ELR (early-learning regularization), FixMatch-style consistency, soft labels; label smoothing found NOT helpful in RSNA 2024 5th place ablation.
  Source: https://arxiv.org/html/2607.16293v1 (CheXpert noisy-label benchmark)
- If a small gold set exists for training (not just eval): fine-tune on gold last, or use gold for early stopping/model selection only — never train from scratch on gold alone.

---

## 4. Foundation models for medical imaging (2025-2026)

- **AnyMC3D (Siemens, Dec 2025)** — the most decision-relevant paper. 12-task 3D classification benchmark incl. **multi-view shoulder MRI at ~11-12k volumes** (near-identical regime to this competition). Findings: (1) frozen 2D FM + LoRA (rank 8) + **query-based attention pooling over slices** + per-view LoRA/queries with late attention fusion beats all 3D CNNs, 3D FMs (MedicalNet, VoCo), and RSNA-Kaggle-style 2.5D-BiLSTM, with ~1.3M trainable params; (2) **DINOv3 ~ MedImageInsight when properly adapted** — general FMs match medical FMs; frozen-embedding average pooling (the common approach) leaves delta AUC ~0.11 on the table; (3) 2D-based methods beat 3D architectures for 3D classification.
  Source: https://arxiv.org/html/2512.12887v3
- **OrthoFoundation (Jan 2026)** — DINOv3 backbone continued-pretrained (SSL contrastive) on **1.2M unlabeled knee X-ray+MRI**; SOTA on 14 downstream tasks, first place on MRI structural injury detection, 50% label efficiency, cross-anatomy transfer. Proof that **knee-domain SSL pretraining on DINOv3 is the current SOTA recipe.** Weight availability/Kaggle-legality **[UNCERTAIN]** — but you can replicate the approach on the competition's own 570 GB.
  Source: https://www.arxiv.org/abs/2601.18250
- Transferability study selecting DINOv3-L over DINOv2/MAE/MoCov3/SimMIM for knee continued pretraining (PED/SFDA/LogME/PARC estimators). Source: https://www.arxiv.org/pdf/2601.18250
- **RAD-DINO** (DINOv2-style SSL on 3.5M CXR) beats CLIP-style and larger models on classification/segmentation transfer; DINOv2-family scales better than CLIP for structure-continuity features. CXR-only, but supports image-only SSL > text-supervised for radiology backbones.
  Sources: https://arxiv.org/pdf/2401.10815 ; https://arxiv.org/html/2509.12818v1
- DINOv2 on radiology benchmarks: strong linear-probe/PEFT classification; PEFT ~ full fine-tune with <1% params. Caveat: on data far from natural images (brain MRI glioma grading) ImageNet CNNs sometimes beat frozen DINOv2 — **fine-tune, don't just linear-probe.**
  Sources: https://arxiv.org/html/2312.02366v3 ; https://arxiv.org/pdf/2402.07595
- "KNEEL" is a 2019 knee **radiograph landmark** hourglass model, not an MRI foundation model — no public knee-MRI pretrained weights of note besides MRNet checkpoints (and possibly OrthoFoundation **[UNCERTAIN]**).
- **Kaggle practicality:** DINOv2/DINOv3 ViT-B/L weights are on HuggingFace/timm and loadable offline as Kaggle datasets (standard practice; verify the competition's external-data rules). MedImageInsight/BiomedCLIP usable but text encoders wasted; MedGemma underperformed in AnyMC3D. Recommended: **DINOv3-B/L or DINOv2-L primary; ConvNeXt-S / EfficientNetV2-S CNN diversity for the ensemble.**

---

## 5. Efficiency prize track

- Winning pattern: **distill the big ensemble into one small 2.5D student** (EfficientNetV2-S / ConvNeXt-T / ELNet-style custom CNN / ViT-S distilled), then **FP16 or INT8 via ONNX -> TensorRT**. RSNA 2025 1st place shipped TensorRT FP16 engines; AIMO-2 winner used TensorRT-LLM FP8 (1.5x over BF16).
  Sources: https://github.com/uchiyama33/rsna2025_1st_place ; https://mlcontests.com/state-of-machine-learning-competitions-2025/
- Real INT8 PTQ for 3D medical models via ONNX Q/DQ nodes + TensorRT calibration — no retraining, big size/latency cuts.
  Source: https://arxiv.org/pdf/2501.17343
- Reported medical-imaging deployment numbers (A100, batch 16): ONNX Runtime-CUDA 8.2 ms -> TensorRT-INT8 3.1 ms, VRAM 3240 -> 1890 MiB. **[UNCERTAIN — single blog source, but consistent with typical 2-3x speedups]**
  Source: https://blog.csdn.net/PixelShoal/article/details/160560514
- ELNet evidence (~0.2M params, competitive AUC on MRNet) suggests an aggressively small from-scratch student is viable for knee MRI specifically — model size is not the binding constraint; slice count x resolution at inference is. Cut slices intelligently (ROI crop + middle-slice selection) before shrinking the backbone.
- Practical checklist: fixed slice budget (e.g., 16-24 slices/series), FP16 everywhere, channels_last + torch.compile during dev, ONNX export with dynamic batch, TensorRT INT8 with calibration on ~500 representative slices, measure accuracy drop per step (expect <0.5% AUC for FP16, ~1% for INT8 **[UNCERTAIN, task-dependent]**).

---

## 6. Recommended architecture & infra for this competition

**Pipeline (accuracy track):**
1. **Preprocessing:** DICOM -> per-series volumes; sort slices by InstanceNumber/ImagePositionPatient; normalize intensity per-series (percentile clip 0.5-99.5, z-score or MRNet-style histogram normalization); cache as fp16 .npy (570 GB -> manageable at 256-384 px in-plane).
2. **Series/plane handling:** group series by plane (sagittal/coronal/axial) and sequence type (T1/T2/PD/fs) from DICOM headers + a small series-classifier CNN where headers are messy. Train per-plane branches; fuse by attention or logreg (MRNet-style) — per-plane-then-fuse beat joint multi-plane training in MRNet follow-ups.
3. **Stage A (SSL, optional but highest-ceiling):** DINOv2/DINOv3-style self-distillation on all unlabeled knee slices. Skip if compute-bound; start from public DINOv3-B/L.
4. **Stage B (localization, optional):** small heatmap/UNet model for meniscus/ACL ROI (report-derived slice hints or gold subset); crop 2.5D stacks around ROIs. Highest value for cartilage/meniscus labels.
5. **Stage C (classifier):** 2.5D stacks (3-5 adjacent slices as channels, or full-sequence slice sampling of 16-32) -> DINOv3-B (LoRA r=8 or full FT) + ConvNeXt-S/EfficientNetV2-S -> per-slice embeddings -> **query-based attention pooling (MIL)** or BiLSTM -> multi-label heads (BCE per label). Auxiliary soft-label heads from LLM extraction confidence.
6. **Label pipeline:** Qwen2.5-32B/Qwen3 (vLLM, temp=0, constrained JSON) on original-language reports -> per-label probabilities; calibrate on ~200-500 manually checked reports; confident-learning filtering with OOF; two-step training (AUC phase -> metric-weighted head phase).
7. **Ensemble/TTA:** 5-fold x 2-3 backbones x 2 seeds; hflip + 90-deg rotation TTA; per-label blend weights tuned on OOF.

**Hyperparameters (starting points, evidence-backed):**
- Optimizer AdamW, lr 1e-4 (CNN) / 1e-5-3e-5 (ViT full FT) / 1e-3 (LoRA+head), cosine schedule, wd 1e-4-1e-2, 10-30 epochs with early stop on OOF metric.
- Batch: 8-32 exams (slice-batched), grad accumulation to effective 32-64; mixed precision (bf16/AMP) mandatory.
- Resolution: 256-384 px in-plane for CNNs; 518 px for DINOv2/3 if budget allows (positional-interp penalty at 224).
- Augmentation: rotation +/-25 deg, shift, hflip (careful with left/right labels!), brightness/contrast/gamma, **CutMix p=1.0** (RSNA 2024 5th place's biggest single win), manifold mixup optional; elastic aug for CNN branches.
- Loss: BCE with class weights matched to metric; focal (gamma~2) for rare labels; NO label smoothing (negative result in RSNA 2024).
- CV: 5-fold stratified by patient/exam, grouped by site/language if identifiable — **trust OOF, not public LB** (RSNA comps are notorious for LB shakeups; 2024 lumbar public 0.43 vs private 0.49 reordering).

**Infra for Kaggle limits:**
- Standard Kaggle GPU quota: T4x2 or P100 (16 GB) ~30 h/week; L4x4 on some comps. **[UNCERTAIN for 2026 comp — check comp page]** Plan: train locally/on cloud (A100/4090-class, 24-48 GB — both RSNA winners used exactly this), use Kaggle notebooks only for inference within the runtime cap: pre-export ONNX/TensorRT, precompute series -> npy, stream inference.
- 570 GB: convert once to fp16 npy/WebDataset shards; do NOT decode DICOM in the training loop.
- Efficiency track target: single student model, <=24 slices/series, FP16 TensorRT, measure ms/exam on the comp's inference hardware.

---

## 7. Open uncertainties

- Exact label set, metric (likely weighted multi-label log-loss or AUC), and whether report text ships in the comp data or labels are pre-extracted — changes Section 3 substantially.
- OrthoFoundation weight availability and license; DINOv3 license acceptance on Kaggle.
- Whether gold labels exist for a train subset (affects fine-tune-on-gold strategy).
- KneeXNet 0.985 AUC claim and the 2-3x TensorRT speedup figure are single-source; treat as directional only.
