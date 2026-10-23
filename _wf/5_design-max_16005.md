# EFFICIENCY TRACK DESIGN — RSNA Knee (distilled 2.5D student, T4, ≤9h)

## 0. The one insight that reframes the whole track

On T4 for this task, **walltime is DICOM-decode I/O-bound, not GPU-compute-bound.** A single small 2.5D student finishes ~1,300 test studies in ~7–20 min, versus ~4–4.5 h for the accuracy ensemble — a ~20–35× reduction that comes almost entirely from *(a) shipping one student instead of the ensemble* and *(b) decoding fewer slices*, NOT from INT8 kernels.

Consequence, which drives every choice below:
- The decode floor (slices decoded ÷ effective slices/sec) sets the wall clock. GPU forward hides under it if the student is small + FP16.
- FP16/INT8/TensorRT mostly buy **headroom and memory**, not wall-time — they only cut wall-time once forward-time rivals decode-time (big student, high res, or heavy TTA). So spend the freed compute on *accuracy* (more slices, ROI two-pass, light TTA, a slightly bigger student), never shrink the backbone below the decode floor.
- **Measure real slices/sec on the Kaggle T4 first** (the single load-bearing number). Everything downstream is parametric in it. RSNA MR DICOMs are usually JPEG2000/JPEG-lossless; `pylibjpeg-openjpeg` does ~15–40 slices/s/thread, `gdcm` faster; 4 vCPUs → assume ~150 slices/s aggregate (conservative) to ~350 (uncompressed/RLE/gdcm-fast).

The AUC-vs-walltime index formula is **unverified** (research brief §5, strategy §1 flagged `[formula unverified]`). Design for the general shape (reward AUC, penalize walltime relative to the 9 h cap) and pick the operating point empirically once the formula and real decode rate are known.

---

## 1. What must exist FIRST (hard prerequisites)

Efficiency work is a *downstream* of the accuracy track (strategy W5, Oct 4–10). You cannot start distillation until:

1. **A trained teacher** — the accuracy-track ensemble (5-fold × {DINOv3-B+LoRA, ConvNeXt-S/EffV2-S} × rotation TTA), the thing that scores ~0.88–0.90. This is the non-negotiable blocker. (Strategy has it landing W3–W4.)
2. **Cached teacher soft predictions on all 4,407 train studies** — the 12 soft probabilities (and, ideally, the teacher's per-slice MIL attention weights) as the distillation targets. Also the teacher **OOF** predictions for honest student eval.
3. **The shared DICOM→tensor preprocessing pipeline** (sort by `ImagePositionPatient`/`InstanceNumber`, percentile-clip 0.5–99.5, per-series norm, laterality flip left→right).
4. **Eval harness: gold-58 + grouped-CV OOF** (group by detected report language per Analysis 1/2) to measure per-label AUC drop at *each* compression step. Rare labels (MCL n=9, Lateral OA n=11, Baker's n=12, Fracture n=18) are the ones that break under aggressive slicing/INT8 — track them individually, not just macro.

**Can be built in parallel, before the teacher, to de-risk (do these now):**
- A **decode-benchmark notebook** on Kaggle T4 against test-like DICOMs → the real slices/sec, header-read rate, and compression syntax. This validates every walltime number below.
- Confirm **offline availability** of `tensorrt` / `onnxruntime-gpu` in the Kaggle image (the handoff already burned on `bitsandbytes`/`autoawq` being absent). If absent, ship wheels as a dataset, or fall back to torch-FP16 (see §6).
- The pipelined inference-notebook skeleton (§7), tested end-to-end against the constant-0.5 plumbing that already validated the submission path.

---

## 2. The student (single model, one forward graph, one engine)

Efficiency track needs ONE model that fuses planes internally — not an ensemble. Shape it as a smaller mirror of the teacher so distillation aligns:

```
2.5D sample = 3 adjacent slices as channels  [B, 3, H, W]
  → per-slice 2D encoder  (shared trunk; plane passed as learned embedding / plane-specific BN)
       backbone: ConvNeXt-Nano (~15M) or EfficientNetV2-B0 (~7M)   ← default
       (ELNet-style ~0.2–2M is viable per lit, but distillation from ImageNet-pretrained is easier with the above)
  → attention-MIL pooling over the K sampled slices  → per-(plane,fluid) embedding
  → attention fusion across buckets                  → study embedding
  → 12 sigmoid heads
```

Why not go maximally tiny: decode-bound means even ConvNeXt-Small keeps up under TRT-FP16 on T4. Pick the **most accurate student whose forward stays ≤ decode**, not the smallest. ConvNeXt-Nano/EffV2-B0 is the sweet spot (accurate, small engine, low memory, fast load).

**Split for export (important):** the expensive **2D trunk** goes to the TRT/ONNX engine (dynamic batch); the **cheap set-aggregation** (MIL pooling + fusion + heads = a few matmuls) stays in torch/numpy. This sidesteps TRT's variable-length/dynamic-shape pain entirely and keeps the engine trivial.

---

## 3. Distillation recipe (offline, on rented A100/4090 — NOT on Kaggle)

Distill **into the efficient operating point** — train the student with the *same* slice budget, ROI crop, and resolution it will use at inference, so there's zero train/inference mismatch.

- **Targets, in priority:**
  1. **KD on the 12 logits** from the teacher ensemble on all 4,407 (MSE on logits, or BCE on soft probs, temperature T≈2–4). This is "distill the ensemble" — the student learns the ensemble's smoothed function and beats a student trained on the noisy report labels directly.
  2. **Auxiliary hard-ish labels**: the report-derived pseudo-labels (rule∩Qwen ensemble) as a secondary BCE, low weight.
  3. **Attention transfer** (optional, high-synergy): match the teacher's per-slice MIL attention. This teaches the student *which slices carry the finding* → lets it use fewer slices without dropping the diagnostic ones. Directly enables the slice-budget cut.
  4. **Gold-58**: last-step calibration + early-stop / model selection only. Never train from scratch on it.
- Loss: `L = α·KD + β·BCE_pseudo + γ·attn_transfer`, tune α≈0.7. Class weights from the **pseudo-label distribution**, not gold-58 (Analysis 1: gold is pathology-enriched, ~2× the true rare-class rate).
- Aug (dev only): rotation ±25°, hflip *after* laterality-norm, brightness/contrast, **CutMix p=1.0** (the single biggest win in the RSNA-2024 5th-place ablation). No label smoothing (negative result there).
- Output: one student checkpoint per config you want to A/B (K=8/16/24, 224/288/384). Cheap to produce — same training, different slice/res knobs.

---

## 4. Slice budget + series selection + ROI crop (the real walltime levers)

**Header-only selection (no extra decode):** read every series header (`stop_before_pixels=True`, ~2,000/s), sort each series by slice position, then choose which K slices and which series to actually decode pixels for. Only selected slices get decoded.

**Series (bucket) selection R** — from Analysis 3's 6 canonical buckets (plane × fluid-sensitive):
- Default R=4: **Sagittal-FS0** (ACL/MCL/meniscus/OA anatomy) + **Sagittal-FS1** + **Coronal-FS1** + **Axial-FS1** (fluid: effusion/synovitis/Baker's/marrow-edema). Fall back within plane if a bucket is absent; every study is guaranteed all 3 planes, so a per-plane branch never starves. Design each branch to tolerate a zero/missing bucket (learned zero-fill + mask) — the hidden test may not honor train invariants.
- This is ~4 of the ~5.5 series/study → ~27% fewer decodes than "all series," at negligible AUC cost.

**Slice budget K per series** (evenly sampled across the sorted stack, weighted toward the joint-center):
- K=24 → max-accuracy-within-efficiency
- **K=16 → balanced (recommended)** — below this, localized findings (meniscus tear, fracture) start to drop in recall; the attention-transfer distillation is what lets you hold 16.
- K=8 → fast floor; risks rare-label recall — ablate on gold-58 before trusting.
Each selected slice → 2.5D 3-channel group (i-1, i, i+1).

**ROI crop (localize-then-classify, proven +several points on meniscus/ACL):**
- Cheap heuristic first: downsample decoded slice to 128px, intensity-threshold the knee (bright, central) vs air (dark), bbox + margin, crop, resize to model res. Pure numpy, ~zero cost, robust. Doubles as a sanity check on slice selection.
- Only upgrade to a distilled 1-channel UNet-nano localizer if the ablation shows meniscus/ACL AUC gains justify the extra model + load. For the efficiency track, prefer the heuristic — no second engine.
- ROI also lets you drop model res to 288 (from teacher's 518) with KD closing the gap.

---

## 5. Export: ONNX → TensorRT FP16/INT8

Export only the **2D trunk** (dynamic batch, fixed spatial dims — TRT likes fixed/profile-bounded shapes):

1. `torch.onnx.export(trunk, dummy[1,3,288,288], opset=17, dynamic_axes={'x':{0:'batch'}})`. Verify with `onnxruntime` CPU that outputs match torch to 1e-3.
2. **FP16 engine** (near-lossless): `trtexec --onnx=trunk.onnx --fp16 --minShapes=x:1x3x288x288 --optShapes=x:128x3x288x288 --maxShapes=x:256x3x288x288 --saveEngine=trunk_fp16.engine` (or the Python `Builder` API).
3. **INT8 engine** (post-training quant, no retrain): entropy calibration on ~500–1,000 *preprocessed* representative slices spanning all planes/fluids/languages. Measure per-label AUC drop on gold-58 + OOF. If INT8 costs a rare label (MCL/LatOA/Fracture) >0.01, keep FP16 for the trunk (or run those labels' heads off the FP16 path). Expect ~0 for FP16, ~0.003–0.01 for INT8 (task-dependent, concentrated in rare labels).

**Critical Kaggle gotcha — engine portability:** TRT engines are locked to GPU arch (T4=sm_75) + TRT version + CUDA. Do **not** ship a prebuilt `.engine` from your rented box — it breaks if Kaggle's TRT differs. Instead **ship the ONNX as a dataset and build the FP16/INT8 engine in-notebook at startup** (one-time ~2–8 min for a small model, immune to version drift). Cache the built engine to `/kaggle/working` in case of a warm rerun.

**If TensorRT/onnxruntime-gpu is absent offline** (verify first — the `bitsandbytes` precedent): ship the wheels as a dataset, OR fall back to **torch-FP16 + `channels_last` + CUDA graphs**. Because the pipeline is decode-bound, this fallback loses ~nothing in walltime and carries zero export risk. Honest recommendation: **torch-FP16 is the robust default; reserve TRT-INT8 for when profiling proves forward is a real fraction of wall** (big student / heavy TTA).

---

## 6. Inference-notebook shape (pipelined producer/consumer)

Keep the GPU fed while CPUs decode. Minimize startup (index measures whole-notebook walltime — import only the inference stack, lazy-load, trim).

```python
# --- startup (trim this; it competes with a 7–12 min run) ---
import onnx, tensorrt as trt, numpy as np, pydicom, torch
engine = build_or_load_engine("trunk.onnx")          # in-notebook, cache to /kaggle/working
head  = torch.load("mil_fusion_heads.pt").cuda().half()  # tiny
studies = recursive_glob("/kaggle/input/competitions/<slug>/test_series/*")

# --- Stage 1: header pass (fast, metadata only) ---
plan = {}   # study -> {bucket -> [selected dicom paths]}  via headers + bucket/slice selection
for s in studies:
    hdrs = read_headers(s, stop_before_pixels=True)   # ~2000/s
    plan[s] = select_buckets_and_slices(hdrs, R=4, K=16)

# --- Stage 2: pipelined decode (CPU procs) -> preprocess -> GPU (main) ---
from torch.utils.data import DataLoader
loader = DataLoader(SliceDataset(plan),               # __getitem__ = decode+ROI-crop+norm one 2.5D sample
                    batch_size=128, num_workers=4,    # = vCPUs; decode is the bottleneck
                    pin_memory=True, prefetch_factor=3)
embs = {}                                             # (study,bucket) -> list of slice embeddings
for batch, keys in loader:                            # GPU idles waiting on decode -> that's expected
    z = trt_infer(engine, batch.half().cuda())        # 2D trunk only
    scatter(embs, keys, z)

# --- Stage 3: cheap aggregation (torch, negligible) ---
rows = []
for study in studies:
    plane_emb = {b: attn_pool(embs[study,b]) for b in embs_buckets(study)}   # MIL over K slices
    study_emb = attn_fuse(plane_emb)                                          # across buckets
    rows.append(sigmoid(head(study_emb)))             # 12 probs
write_submission(rows)                                # StudyInstanceUID + 12 cols
```

Notes: `num_workers = vCPU count`; decode is the wall, so GPU util will be low — correct, not a bug. `channels_last`, batch 128 @288px fits T4 16GB with ~10GB spare. Handle missing plane/bucket with zero-fill+mask. Never hard-code "all 3 planes present."

---

## 7. Expected walltime-vs-AUC tradeoff (1,300 studies, T4, 4 vCPU)

Parametric in effective decode rate D (measure it!). Walltime ≈ `overhead + max(N_decode/D, N_forward/F)`, decode-dominated since D≪F. Table uses D=175 slices/s (conservative-compressed), F≈900/s (TRT-FP16 @288). AUC deltas are estimates vs a ~0.89 teacher; the eval harness produces the real curve.

| Config | R × K × res | TTA | precision | slices decoded | est. walltime | est. macro AUC |
|---|---|---|---|---|---|---|
| **Teacher (ref, NOT efficiency)** | 5.5 × 32 × 518 | 3-rot | FP16 ens×10 | ~229k | **~4.0–4.5 h** | ~0.90 |
| **B · Efficient-Max** | 5 × 24 × 288 | 2× hflip | TRT-FP16 | ~156k | **~19–20 min** | ~0.885 |
| **C · Efficient-Balanced (recommend)** | 4 × 16 × 288 | none | TRT/torch-FP16 | ~83k | **~11–12 min** | ~0.875 |
| **D · Efficient-Fast (floor)** | 3 × 8 × 224 | none | TRT-INT8 | ~31k | **~7 min** | ~0.855 |

Read-off:
- **Headline efficiency move = single distilled student vs the ensemble: ~22–35× less walltime for ~0.02 AUC.** Everything else is second-order.
- **C is the default submission**: ~0.025 below teacher for ~22× less wall. B vs C vs D depends on the index formula — if walltime is heavily discounted (already a tiny fraction of 9 h), don't sacrifice AUC chasing minutes; if walltime is strongly rewarded, D can win despite lower AUC.
- FP16 vs INT8 barely moves the wall column (decode-bound) — INT8's only job is to let a *bigger/more-accurate* student or more TTA still hide under decode. Don't pay its rare-label AUC risk unless profiling shows forward ≈ decode.
- If D turns out low (slow J2K, ~60/s), every row scales up ~3× and the slice budget K becomes the dominant knob — cut K/R before touching the backbone.

---

## 8. Build order (checklist)

**Now (pre-teacher, de-risk):**
1. Decode-benchmark notebook on Kaggle T4 → real D, header rate, compression syntax.
2. Verify `tensorrt` + `onnxruntime-gpu` offline; else stage wheels or commit to torch-FP16.
3. Pipelined inference skeleton (§7) tested against the 0.5-baseline plumbing.

**W5 (teacher exists):**
4. Cache teacher soft preds + MIL attention on all 4,407; cache OOF.
5. Distill ConvNext-Nano/EffV2-B0 student at the C operating point (R4·K16·288, ROI, KD+attn-transfer), gold-58 early-stop. Also train B and D variants (same recipe, different knobs).
6. Export trunk → ONNX; build FP16 + INT8 engines in-notebook; calibrate INT8 on ~500 slices.
7. Ablation on gold-58 + grouped-CV OOF: per-label AUC at each {K, R, res, precision} → the real tradeoff curve.
8. Pick the config maximizing the actual index (once formula known); default C, keep B/D as alternates.
9. Trim startup; end-to-end timed rerun; confirm submission validity.

---

**Relevant files (read-only, referenced):** `C:/Users/eeshs/Downloads/rsna-knee/strategy/WINNING_STRATEGY.md` (§2 techniques, §8 efficiency line, W5 timing), `C:/Users/eeshs/Downloads/rsna-knee/research/rsna-knee-2026-research-brief.md` (§5 efficiency-track prior art: RSNA-2025 1st = TRT-FP16; ELNet tiny-CNN evidence; INT8 PTQ; slice-count-before-backbone), `C:/Users/eeshs/Downloads/rsna-knee/HANDOFF.md` (offline-image gotchas: T4-not-P100, missing quant libs — same class of risk as TensorRT availability).

**No teacher = no efficiency submission.** The single biggest efficiency win (student vs ensemble, ~22–35×) and the biggest walltime lever (slice budget under a decode-bound wall) both dwarf the TensorRT-INT8 export the track's name implies — build in that priority order.