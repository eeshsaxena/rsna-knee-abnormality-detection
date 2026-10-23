# RSNA Knee — winning with 4× V100 (compute plan)

*Assumption: "V100 with 4 workers" = 4× V100 GPUs (16 or 32 GB each) you can run in parallel.
If it's 1 V100 with 4 dataloader workers, the parallelism section changes — tell me.*

## The one structural fact
RSNA is a **kernels-only** competition: the only thing you submit is a Kaggle notebook that
runs offline on a T4 in ≤9 h. You therefore CANNOT train at scale on Kaggle (30 h/wk T4 quota).
The winning shape is:

```
TRAIN off-Kaggle on your 4× V100  →  upload trained weights as a private Kaggle Dataset
   →  ship a thin Kaggle inference notebook that loads them and writes submission.csv
```

Your V100s replace the "rent a 4090/A100 for ~$120–330" line the max-accuracy design assumed —
same role, but **free and 4× parallel**. That flips the recommended path from the Kaggle-only
freeze-backbone MVP to the full medal pipeline.

## What a V100 can/can't do (gotchas)
- V100 = Volta, sm_70, ~112 fp16-TFLOPS tensor cores. Plenty for DINOv3-B / ConvNeXt-S /
  EfficientNetV2-S 2.5D training at 320–384 px.
- **Use fp16 AMP, NOT bf16** — Volta has no bf16. (Ampere+ only.)
- **No FlashAttention-2** on sm_70 — use xformers memory-efficient attention or vanilla; fine
  for ViT-B at 384px with batch accumulation.
- 16 GB/card fits DINOv3-B LoRA + attention-MIL at 384px, batch ~8–16 with AMP + grad-accum.
  32 GB cards let you skip most accumulation.

## How to use the 4 GPUs (this is the leverage)
The training grid is the bottleneck: **5 folds × 3 backbones × 2 seeds = 30 runs, ~6 h each
≈ 180 GPU-h.** Don't DDP-shard small models — run **independent jobs, one per GPU**
(`CUDA_VISIBLE_DEVICES=0/1/2/3`), an embarrassingly-parallel grid:

| GPU | Job |
|---|---|
| 0 | DINOv3-B/16 + LoRA r16 + attention-MIL (primary), all folds/seeds |
| 1 | ConvNeXt-S full-FT + BiLSTM (diversity member) |
| 2 | EfficientNetV2-S full-FT + attention-MIL (diversity + efficiency-track seed) |
| 3 | Label upgrade (Qwen-14B/32B vLLM) → then confident-learning OOF passes → then SSL stretch |

30 runs ÷ 4 ≈ **~2 days wall-clock** for the whole ensemble grid, vs ~8 days on one card.
Reserve GPU3 for the non-training compute (the LLM labeler + noise-surgery inference + optional
SSL), because that work is what actually moves the score (see "the governor" below).

## THE GOVERNOR — compute is necessary, not sufficient
Every design agent converged on the same hard fact: **the score ceiling is label quality, not
the backbone.** Only 58 studies are gold-labeled; the image model trains on ~4,349 report-derived
labels currently worth macro-AUC 0.65 (rules) → 0.73 (Qwen-3B) vs gold. A public 0.906 model
exists because its extraction is better, and DINOv2-**large** scored 0.5 (overfit the tiny gold
set). So budget effort ~50% labels / ~35% noise-robust training + honest CV / ~15% backbone.

Your 4th V100 pays off here specifically:
- **Qwen-14B/32B via vLLM** for the labels (a V100-32GB runs 14B fp16 or 32B-AWQ). The measured
  ensemble finding: soft `(rule+Qwen)/2` fusion already lifts labels 0.731 → ~0.78; a bigger LLM
  on the **low-resource 34%** (Turkish/Croatian/Greek/Bulgarian, where 3B is weakest and gold
  can't validate) pushes toward the ~0.82–0.85 text ceiling.
- **SSL stretch becomes realistic (free):** a light DINOv2-style continued-pretrain on a
  200–300k-slice knee subset (~50–100 GPU-h) is ~1 day on GPU3. The max-accuracy design called
  this "skip unless budget" — your free V100s turn it into a real edge.

## The full pipeline (medal path, now feasible at $0)
1. **Labels** (highest leverage): finish Qwen-3B on Kaggle T4 (free, in progress) → soft
   rule×Qwen fusion → PF-OA rule override → **upgrade with Qwen-14B/32B on GPU3** on the
   disagreement + low-resource subset → per-language QA set (hand-check 15–20 studies each for
   tr/hr/el/bg/fr, the only way to catch non-English failure).
2. **Preprocess** 570 GB → ~64 GB fp16 shards (256–384px, ~20 slices/series, laterality-flipped
   left→right). Do this once (Kaggle CPU free, or on the V100 box), never decode DICOM in the
   train loop.
3. **Train** the 3-backbone × 5-fold × 2-seed grid across GPU0–2 (fp16 AMP, CutMix p=1.0,
   plane-drop p=0.15, NO label smoothing, NO naive hflip — it swaps medial/lateral).
4. **Noise surgery** on GPU3: confident-learning OOF down-weighting, two-step training (BCE →
   frozen-backbone head FT with gold-58), test pseudo-labeling.
5. **Ensemble**: per-label logistic stacking on OOF + 9-rotation TTA.
6. **CV discipline**: StratifiedGroupKFold grouped by **detected report language** (the only site
   proxy — no patient/site/date key exists) + one held-out-language OOD fold. Select finals by
   grouped-OOF + gold-58, NEVER public LB (RSNA shakes up hard).
7. **Ship**: weights → private Kaggle Dataset → inference notebook (decode-bound, ~4–5 h of the
   9 h budget for the full ensemble; trim folds/TTA for margin).

## Honest expected outcome
- Well-executed with your V100s: **private macro-AUC ~0.88–0.91** (image denoising clears the
  0.73 label AUC; ~0.90 is where good labels land, per the public 0.906 DINOv2-S).
- **Medal (0.93+)** requires the label upgrade to actually land on the non-English 34% — that's
  the swing factor, not GPU count. Your 4th V100 is aimed exactly at it.
- Biggest risk: French (n=81, zero gold) + Croatian/Bulgarian mislabels, and private-test
  site-shift that grouped-language CV only partially exposes.

## Data logistics
You need ~64 GB of shards near the V100s (not the full 570 GB). Preprocess on Kaggle free CPU
→ download shards to the box, or pull the raw data to the box and preprocess there if it has the
disk + bandwidth. Your RTX 4050 6 GB is dev-only (dataloader debug, OOF blend fitting) — it
trains nothing.

## Immediate next 5 build steps
1. Save Qwen's per-study gold-58 predictions (~3 lines, ~6 min re-run) → makes the exact
   rule×Qwen ensemble AUC measurable (currently a 0.78 estimate).
2. `qwen_extract_v2` prompt (evidence-span citation + 3-state soft + PF-OA vocab + few-shot incl.
   Spanish) → `qwen_preds_full.csv` for all 4,407.
3. `ensemble_labels.py` → soft `pseudo_labels_v1.csv` + agreement `sample_weights.csv`.
4. Preprocess → 64 GB shard dataset.
5. Stand up the V100 box; train DINOv3-B LoRA **fold 0 as a vertical-slice smoke test** — gate:
   gold-58 OOF must beat 0.80 before scaling to the full 30-run grid.
