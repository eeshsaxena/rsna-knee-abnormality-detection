# RSNA 2026 Knee Abnormality Detection — Winning Strategy

**Competition:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
**Today:** 2026-09-01 · **Entry/team-merge deadline:** 2026-10-15 · **Final deadline:** 2026-10-22
**Time budget:** ~7.5 weeks total, ~6.5 weeks of effective work. That is enough for a medal run if we start the data pipeline this week.

---

## 1. What we are actually solving (verified facts)

| Item | Fact | Source |
|---|---|---|
| Metric | **Macro ROC AUC over 12 binary labels** (plain mean of 12 per-label AUCs; rare labels count as much as common ones) | konradcinkusz README §2; Kaggle discussions 733343/733934 |
| Labels | `ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture` — multi-label, study-level, sigmoid | same |
| Train data | 4,407 studies / 24,371 series / ~819k DICOMs / ~570 GB; every study has sagittal+coronal+axial; sequence metadata CSV gives `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane` | JunhaoLiXD + konradcinkusz READMEs |
| Supervision | **Only 58 of 4,407 studies have expert gold labels.** The rest have only free-text radiology reports (multilingual, ~9–12 languages). Report-derived labels agree with gold only ~82% | homeshwarnelakurthi audit (community) |
| Test | ~1,300 studies, **images only — no report text at test time**. Text is a training-time weak-supervision signal only | konradcinkusz §5/§8 |
| Format | Code competition: notebook submission, `submission.csv` (12 probabilities per StudyInstanceUID), ≤9 h GPU, **internet OFF**, use **T4 (not P100)** | konradcinkusz §6 |
| Rules | External data + public pretrained models allowed; **no sending competition data to hosted LLM APIs** (local/offline LLM only); winners release code+weights under CC-BY-NC 4.0 | konradcinkusz §7 |
| Prizes | $77k total: main LB top-10 ($5k–$9k each) + **efficiency track top-3 ($7k/$6k/$5k)** scored by AUC-vs-walltime index | konradcinkusz §13 [formula unverified] |
| LB state | Public LB top ≈ **0.93+** by late Aug; DINOv2-small best public model score 0.906; baseline notebooks 0.66–0.81 | Kaggle Models tab via konradcinkusz §9 |

**The three things that decide this competition:**

1. **Label quality.** 58 gold labels vs 4,349 report-only studies. Whoever builds the best local-LLM label extractor + noise-robust training wins the metric. This is the single biggest lever.
2. **Honest validation.** Random K-fold inflates AUC by ~0.05 via scanner/site memorization. Grouped CV (by patient/site) + the 58-gold OOF is the only score to trust. Public LB is partially report-label-noise — do not overfit it.
3. **Multimodal-to-unimodal distillation.** Test has no reports, so use text at train time (as labels, as a teacher, or as a co-training signal) but ship an image-only model.

---

## 2. Architecture plan (evidence-backed)

**Core model — 2.5D, per-plane, attention-pooled:**

```
DICOM series → preprocess (resize 384–518px, percentile norm, fp16 .npy shards)
  → per-slice encoder: DINOv3/DINOv2 ViT-B (fine-tuned, LoRA r=8–16 for large)
  → slice aggregation: attention MIL pooling (or BiLSTM) over 16–24 sampled slices
  → per-plane/per-sequence heads → late fusion (learned or logistic stacking)
  → 12 sigmoid outputs
```

Why: consensus winner at ~5k-exam scale — MRNet follow-ups, RSNA 2024 lumbar 1st/2nd/5th place, RSNA 2025 aneurysm 1st place, and the AnyMC3D benchmark (frozen DINOv3 + LoRA + attention pooling beat all 3D approaches with ~1.3M trainable params). DINOv2-large scored 0.5 (random) on the public Models tab — **large models overfit the tiny gold set; adaptation method matters more than backbone size.**

**Key techniques, ranked by leverage:**

1. **LLM label extraction (offline):** Qwen2.5/3-14B–32B + vLLM, temp=0, constrained JSON output, English prompt works on multilingual reports (proven for German; skip translate-then-extract). Output 12 labels × {neg/pos/uncertain} + severity. Open LLMs beat CheXpert-style rules 95% vs 51% sensitivity. pilkwang already published an "RSNA Knee LLM-read report labels" Kaggle dataset — use it as a starting point, then beat it with a bigger model + adjudication prompts.
2. **Noise-robust training:** soft/hierarchical labels → confident-learning OOF filtering → two-step training (plain BCE for AUC, then frozen-backbone head with metric-weighted loss). From RSNA 2024 2nd/5th place playbooks.
3. **Laterality normalization** (flip left knees to right) — cheap, proven (+ in pilkwang baseline).
4. **CutMix p=1.0** — biggest single win in RSNA 2024 5th place; label smoothing did NOT work there.
5. **Per-plane-per-task heads** rather than one joint model (negative transfer is real — Azcona et al.).
6. **9-rotation TTA + backbone-diverse fold ensembles** (DINOv3-B + ConvNeXt-S + EfficientNetV2-S).
7. **Continue-pretrain DINOv3 on the 570 GB of unlabeled knee DICOMs** (Orthofoundation proved knee-domain SSL is SOTA on 14 tasks) — highest-ceiling move if GPU budget allows; else skip.
8. **Efficiency track:** distill the ensemble → single small 2.5D student (ELNet-style tiny models work on knee MRI) → ONNX → TensorRT FP16/INT8. RSNA 2025 1st place shipped exactly this. Cut slice budget before shrinking the backbone.

---

## 3. Week-by-week plan (Sep 1 → Oct 22)

| Week | Dates | Goals | Exit criterion |
|---|---|---|---|
| **W0 — Setup** | Sep 1–5 | Join comp, accept rules; download 570 GB (start NOW — it takes days); Kaggle API + local env; DICOM→npy preprocessing pipeline; verify a few studies visually | Pipeline converts 100 studies without error |
| **W1 — Labels + baseline** | Sep 6–12 | Run local LLM label extraction on all 4,349 reports; grouped 5-fold CV splitter (group by site via DICOM metadata + patient); reproduce pilkwang-style DINOv2-small + attention-pool baseline | Baseline OOF ≈ 0.80+ on report labels, gold-58 OOF tracked separately |
| **W2 — First submission** | Sep 13–19 | Offline inference notebook (T4, <9 h, no internet); first real submission; gap analysis OOF vs LB | 1 submission on the board; LB-OOF delta documented |
| **W3 — Scale up** | Sep 20–26 | DINOv3-B + LoRA; per-plane models; laterality norm; CutMix; hard-negative mining from OOF FPs | Gold-58 OOF + grouped-CV OOF both beat W1 by ≥0.02 |
| **W4 — Noise surgery** | Sep 27–Oct 3 | Confident learning pass; retrain on cleaned labels; pseudo-label high-confidence test predictions (0.5/0.5 mix); ensemble v1 | Ensemble OOF > best single by ≥0.01 |
| **W5 — Efficiency + ensemble v2** | Oct 4–10 | Distill → small student → ONNX/TensorRT FP16; measure wall-time; decide main-track vs dual-track submission selection | Efficiency submission beats sample benchmark with margin |
| **W6 — Final** | Oct 11–22 | **Oct 15: entry deadline — make sure rules accepted + team set.** Final 2-submission selection: pick by grouped-CV + gold-58 OOF, NOT public LB. Freeze code, docs, prepare winner obligations (open-source repo, method video) | 2 final selections submitted by Oct 20 (2-day buffer) |

---

## 4. Compute & logistics decisions (need your input)

1. **GPU:** Kaggle's ~30 h/week T4×2 quota is NOT enough to train competitively — it's for inference notebooks only. Options: (a) local GPU if you have one (24 GB+ VRAM ideal), (b) rented A100/4090 cloud (~$1–2/h; budget ~$100–300 for the run), (c) Colab Pro. Which do you have?
2. **Storage:** ~600 GB free disk for raw data + ~100–200 GB for preprocessed shards. SSD strongly preferred.
3. **Time:** expect this to need ~2–4 h/day of your attention on weekends-heavy schedule; I'll do the heavy lifting.

## 5. Risk register

| Risk | Mitigation |
|---|---|
| Label noise caps AUC | LLM ensemble extraction (2 models + adjudication), confident learning, weight gold-58 in final fine-tune |
| Public LB ≠ private (distribution notice) | Select finals by grouped OOF; keep one "safe" and one "aggressive" final pick |
| 9-h inference limit with 570 GB-class test | Preprocess to fp16 shards in-notebook is too slow — precompute embeddings? No: test data is hidden. Must decode+infer within 9 h: lean pipeline (pydicom + parallel decode, 16–24 slices/series, FP16, batch by plane) |
| Overfitting 58 gold labels | Use them only for validation/early-stop, never (or only in last-layer) training |
| P100 kernel trap | Always select T4 accelerator in notebook settings |

---

*Research sources: [full prior-art brief](../research/rsna-knee-2026-research-brief.md); competition facts via konradcinkusz/rsna-knee-abnormalities-efficiency-lb, JunhaoLiXD/RSNA_Knee_Abnormality_Detection, homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection (GitHub), RSNA press release 2026-08-05.*
