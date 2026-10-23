# R7 port plan: feasibility, inputs, time, cost (written 10-13 ~16:45 IST, no training done)

Goal: train extra members of heliosli's R7 recipe (ConvNeXt-small, bundle `heliosli/rsna-knee-compact-r7cnxfsss`,
bundle_id 8c7b666e27dc0e8f) on rented GPUs (RSNA data only) or on the box (RSNA + OAI).

## 0. Bottom line

- **Code: complete and working.** The exact R7 trainer, its pixel-cache builder and every decode / anchor / side-check
  module are public. I extracted them, verified every hash against the R7 manifest, and ran the whole chain on the box
  CPU: raw DICOM -> px384_v4 rows (5 studies, deterministic) -> `compact_train.py` with the full R7 flag set
  (check-targets OK, 22 optimizer steps, OOF + gold + state read-out + fold model written). The GPU was never touched.
- **Inputs: the decisive parts of R7 cannot be copied.** R7's strength comes from its targets: Opus-written report
  states plus 7 refinement rounds (`probe_targets_opusv11b_rr7_plr7_cdr.csv`, `cell_weights_plr7_v7.csv`,
  `state_targets_opusv11b_rr7.csv`). None of these files is public, and neither are their builders. The fastMRI+
  auxiliary task (`fm384_v1`) needs NYU fastMRI raw data (data use agreement, about 1 TB), which is not possible in 2 days.
  We can only train "R7 architecture + our labels" (state head off), or add our own LLM-made state targets, which is
  untested and risky.
- **Time / money:** 1 fold = 9.2 M window passes at 352 px. Estimated 5.8 h on an H100, 11.6 h on an A100 80 GB,
  16 h on an RTX 4090 and 29 h on the box V100 (MODEL, +-40 %). Getting to an LB read from the moment we rent takes
  about 16 h on 4 x H100 (~$95-110) or about 26 h on 4 x 4090 (~$40).
- **Recommendation: NO-GO for the 2-day .960 push.** Expected visible public gain on top of R7 is 0 to +.001 (GUESS).
  P(visible gain within 2 days) is about 10 % (GUESS). That is lower than PLAN_960's 20-30 %, because the targets
  (not the architecture) are what we cannot reproduce. Optional slower GO only for a final-pick-B candidate
  (sec. 9). Revisit only if HFR7-R20 reads .957, which would mean our labels add on top of R7.

## 1. What exists (all verified)

| Item | Where | Check |
|---|---|---|
| R7 trainer `compact_train.py` v9.8 (230,308 B, 3,072 lines) | R7 notebook cell 4, `SOURCES` dict (`heliosli/0-957-latest-dataset-score-in-one-click`) | md5 8622009e = manifest `trainer_md5`; sha256 04df7e2e = manifest `src_sha256` |
| `px_cache.py` (pixel cache, views v4), `compact_infer.py` (inference + sidefix-cache builder), `knee_anchor_bank_v2.py`, `side_common.py`, `probe_dual.py`, `probe_infer.py` | same cell | all 7 sha256 = `SOURCE_SHA` / manifest `src_sha256` |
| Mainline decode / geometry / render package (`native_cache`, `native_data`, `input_provenance`, `intensity_mapping`, `coronal_views`, `exact_duplicates`, `dicom_geometry`, `common`, `geometry_bank`, `anatomy`, `anatomy_runtime`, ...) | `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` `cloud_code/training/` (185 code files) + `vendor_runtime/monai060` (194 files); 5 listing pages, 837 entries | all 379 downloaded (one by one, Kaggle 429 back-off); the vendored MONAI renders byte-identical rows to PyPI monai 0.6.0 |
| `anatomy.pt` (SKM-TEA V-Net, knee anchors, 112 MB) | orthofoundation dataset | sha256 5904dd16 = manifest `anchor_weights_sha256` |
| `side_cls.pt` (image side check, 85 MB), `state_spec.json`, `fluid_stacker.json`, `state_stacker.json` | R7 bundle | sha256 = manifest |
| ConvNeXt-S init `convnext_small.fb_in22k_ft_in1k` | HF hub, reachable from the box (15.6 MB/s) | box `~/r7port/pretrained/`, sha256 2457488b; loads with missing 0 / unexpected 2 (= the ImageNet classifier) |

Not the R7 trainer: `cloud_code/training/four_gpu/train.py` and `engine.py` / `data.py` / `labels.py` belong to
heliosli's OrthoFoundation mainline (torchrun, `KneeModel`). R7 is `compact_train.py` ("compact fine-tuned image model on the pixel cache").
`semantic_review_contract.py` / `semantic_review_compact.py` are the mainline LLM report-review prompt + schema, which are
useful as a prompt template, but the R7 state targets came from a different, unpublished builder (`make_state_targets.py`, `make_state_spec_r5.py`).

**Copies:** scratchpad `r7port/{src,ofl,manifest,cells}`; box `~/r7port/{src,code,weights,pretrained,bundle_r7,scripts}`;
our own scripts + test results in `session_2026-10-13/r7port/{scripts,results}`.

### Preprocessing chain (one pass reproduces px384_v4 + px384_v4_sidefix)
`compact_infer.CStudies` is both R7's inference path and heliosli's own "training-cache override builder" (it built
`px384_v4_sidefix`). The chain is: raw DICOM -> lean native render (`training.native_cache.prepare` per study) -> geometry bank ->
image side check (`side_cls.pt`, threshold 2.0, position 20 mm) -> SKM-TEA joint / patella anchors (`anatomy.pt`) ->
`px_cache.StudySlices.build` with views v4 at 384 px. v4 views per series: sagittal / axial = full field + 3 x 100 mm locals (joint, PF,
posterior), coronal = full field + medial / lateral locals; thin 3-D stacks are averaged to 3.5 mm slabs.
Our `px_build_all.py` runs it in parallel and merges the parts into the trainer's `--px` format (index.json / px.u8.npy /
meta.f32.npy). Running studies with the side check in the same pass makes `--px-override` unnecessary.

## 2. Box tests (CPU only: `nice -n 19 taskset -c 0-7`, `CUDA_VISIBLE_DEVICES=""`; the V100 queue was untouched)

1. **Pixel cache, 5 random train studies** (`scripts/px_build_test.py`, results `r7port/results/px_build_test.json`): 5 / 5 OK.
   Per study (1 process, 2 threads): native + geometry + side check 5.9 s, SKM-TEA anchors on CPU **22.9 s**, render 1.5 s,
   total 30.2 s. 580 rows (424-724) and **81.6 MiB of px per study**, 121 MB of DICOM per study. The side check ran on 4 of 5
   (no explicit laterality tag); 0 flips. The anchors used the `skm_tea_joint` method on 5 / 5.
   The parallel builder + merge (3 processes) is byte-identical to the single process on 5 / 5 studies, with both MONAI copies.
2. **Model memory / FLOPs** (`scripts/mem_flop_probe.py`, `results/mem_probe.log`): encoder forward = **42.9 GFLOP per
   3 x 352 x 352 window**. Saved activations in bf16 = **222 MB per window without checkpointing, 43 MB with timm
   `--grad-ckpt`**. One R7 step = 2 studies x 176 windows = 352 windows -> **78 GB of activations**, so the recipe as run
   (`grad_ckpt false`) fits only 96 GB+ cards (heliosli used RTX PRO 6000s; `calibrate_pro6000.py`).
3. **Trainer, full R7 flag set minus fm-aux** (`scripts/run_pipe.sh`, `results/pipe_run1.log`): 30 synthetic studies (the 5 real
   ones replicated), SYNTHETIC random state targets on the real R7 `state_spec.json`, our soft labels as the lr targets.
   `--check-targets OK`; 2 epochs x 11 steps (train_windows 16, CPU device patch only); rare-boost, pair-rank (8
   families), extent, mirror, syn-decomp, aug v2, draw-seed all active. Eval wrote `partial_oof`, `partial_oof_state`,
   `partial_preds(_state)` and the fold model `E_lrprior_s0_f0.pt` = 200,378,978 B (the R7 fold files are 200,379,391 B:
   same network).
4. **R7-lite (state head off) and `--grad-ckpt`** (`scripts/run_lite.sh`): forward / backward OK on our real targets. Loss and
   gradient mass are identical with and without checkpointing. `--check-targets` passes on our full 4,407-row target file.

## 3. Every required input and whether we can produce it

| Input (R7 run_config) | Format | Can we produce it? |
|---|---|---|
| `px384_v4` + `px384_v4_sidefix` | `px.u8.npy` uint8 [N, 384, 384] (one row per series x view x slice), `meta.f32.npy` [N, 12], `index.json` (`done` {uid: [start, n, n_series]}, `blocks`) | **Yes**, `px_build_all.py`. Size: 4,407 x 81.6 MiB = **~350 GiB (~2.6 M rows)**. CPU: 30 s / study on the box with CPU anchors, about 9 s / study with GPU anchors (Kaggle T4 log: anchors 1.4-1.9 s). **Blocker on the box: only 458 of 4,407 train studies (57 GB) have DICOMs there**; the full train set is about 550-570 GB |
| `--targets` (`probe_targets_...csv`) | `StudyInstanceUID, is_gold, review_holdout, fold` + per class `L\|prior`, `L\|lr_prior` (likelihood ratio > 0, used by the arm), `L\|strict`; gold rows: every cell blank (asserted); a non-gold row with any blank prior is dropped ("blanked"); `fold` only matters as -3 (review holdout) | **Yes, from our labels**: `scripts/make_targets.py` (done on the box: `~/r7port/targets_ours/`, 4,349 pool + 58 gold found from train.csv). prior = clip(llm_labels_v4_blend, .02, .98); lr_prior = odds(prior) / odds(pool base). **Caveat:** our blend codes "not mentioned" as 0.25 and has high base rates (ACL .23 vs R7 spec .14, Effusion .58 vs .36), so the lr arm will learn those calibrations. The soft-BCE arm `E_prior` (`--arms E_prior`) is a safer first choice |
| `--extra` (`cell_weights_...csv`) | `StudyInstanceUID, L\|w` (>= 0) | Yes: 1.0, or from the FINAL2 agreement tier (done: very_high 1, high .85, medium .65, low .45) |
| Official train.csv | Only the 58 gold rows have labels | It is used **only** to mark `is_gold` (never trained, by design: `official_labels_read_by_trainer: false`) |
| `--state-targets` | `L\|state` (index into the spec's category list), `L\|sw` (certainty weight: definite 1 / probable .7 / possible .4), `L\|<binary>` for sign_osteophytes / sign_subchondral_marrow / sign_bone_on_bone (3 OA classes) and acute (MCL, Fracture, Contusion, ACL) | **No ready source.** Our labels have no graded states. Options: (a) switch the head off (drop `--state-targets --state-spec --state-weight --rare-boost --rare-boost-names --pair-rank --pair-frac --extent* --save-state-oof`; verified to run) = **R7-lite**; (b) a new LLM extraction into the exact R7 categories (Qwen2.5-32B-AWQ with vLLM on one rental GPU, ~1 h, or Claude in-session subagents), reusing R7's `state_spec.json` verbatim (frozen prior / base). (b) adds 3-4 h of work and its quality can only be checked on gold58 |
| `--state-spec` | R7 `state_spec.json` (12 classes, 5-8 categories each, prior per category, base, binaries) | Yes, public in the bundle |
| `--fm-aux fm384_v1` (the main R6 -> R7 change) | fastMRI knee volumes rendered at 384 + fastMRI+ lesion boxes, >= 1,000 volumes | **No** (NYU DUA + ~1 TB of raw data). Without the flag the trainer is v9.7 bit for bit (the R6-era trainer) |
| Folds | Built in: `sha1(uid)[:8] % K` over the pool (K = 4 in R7); the CSV fold column is ignored except -3 | Keep the hash folds (same assignment as R7, so OOF lines up with R7's split). `folds5` reuse would need a 3-line patch (`pool_fold` from the CSV); not worth it |
| `--init` | `convnext_small_in22k_ft_in1k.safetensors` | Yes (HF, on the box). R7's own file hash is not recorded (`init_weights_sha256: null`); timm's fb_in22k_ft_in1k is the standard file |
| Stacker fitters (`fit_fluid_stacker.py`, `fit_state_stacker.py`), `merge_partial_v2.py` | not public | Skip the stackers for our group; merging the per-fold `partial_*` outputs takes ~20 lines (each file has its fold's rows) |

## 4. Training time per fold (MODEL)

R7 schedule: 16 epochs, batch 2 x 176 windows (crop 352, 3 slices). Pool about 4,349 -> about 3,262 training studies per fold
-> 1,631 steps per epoch -> **26,096 steps = 9.2 M window passes per fold**. FLOPs: forward + backward = 3 x 42.9 = 129 GFLOP per window,
**1.18 EFLOP per fold**, or 1.58 EFLOP with checkpoint recompute. End-of-fold OOF / gold eval (eval_stride 1, ~0.6 M windows forward) adds about 5 %.

Throughput anchor: our own V100 ConvNeXt trainers measured **~11.7 TFLOP/s achieved** (TTR ConvNeXt-T, 0.29 s/it at 224 and
0.49 s/it at 288, 128 images per step). Ratios for ConvNeXt AMP training vs V100 (GUESS from public benchmarks): 4090 1.8x,
A100 2.5x, H100 SXM 5x, H200 5.5x; `--compile` x1.3 (heliosli measured 1.4x).

| GPU | Memory fit for 352 windows / step | Per fold (one fold per GPU) | 4 folds on 4 GPUs |
|---|---|---|---|
| V100 16 GB (box) | needs checkpointing **and** a bf16 -> fp16 + GradScaler patch (no bf16 hardware) | ~29 h (20-40) | ~5 days sequential on the single box GPU, which is also busy: **out** |
| RTX 4090 24 GB | timm `--grad-ckpt` est. peak ~25-27 GB: **OOM risk**. Needs a chunked-checkpoint patch (encoder in 64-window checkpointed chunks, math-identical, peak ~17 GB) or train_windows ~140 | ~16 h (11-22) | ~16 h wall |
| A100 80 GB | `--grad-ckpt` (~25 GB) fits; no-ckpt (78 GB + state) does not reliably fit | ~11.6 h (8-16) | ~12 h wall |
| H100 80 GB SXM | `--grad-ckpt` fits | ~5.8 h (4-8) | ~6 h wall |
| H200 141 GB / RTX PRO 6000 96 GB | no checkpointing needed (as R7 ran) | H200 ~3.9 h; PRO 6000 ~7 h | same wall |

Data loading per step: 352 windows x 3 rows x 147 KB = 155 MB from the 350 GiB memmap. That needs local NVMe (or RAM) and
6 loader workers per GPU, so plan for >= 32 vCPU on a 4-GPU machine.

The 20-iteration CPU probe validates the pipeline only. CPU speed (48 s per 32-window step on 8 cores) says nothing about
GPU speed. A first-epoch timing on the rental (smoke step 5 below) replaces this whole table within 15 minutes.

## 5. Cost and wall time from rental to an LB read (one machine, 4 GPUs, one fold each)

Prep on the rental: env + code 0.5 h; `kaggle competitions download` of ~560 GB, 1-2 h (GUESS, 100-200 MB/s) + unzip 0.5 h;
px cache 4,407 x ~9 s / 14-16 processes with GPU anchors ~0.8 h + merge 0.2 h; targets + check + smoke 0.5 h -> **~3.5-4 h**.
After training: OOF merge + gold58 check 0.5 h, bundle upload (4 x 200 MB) + notebook + submit 1 h, hidden scoring ~4-6 h.
Disk: 560 GB zip + 560 GB extracted (delete the zip) + 375 GB px -> **2 TB NVMe**.

| Machine (indicative Vast / RunPod rates, GUESS) | Rental hours | Cost | Rent -> LB read |
|---|---|---|---|
| 4 x RTX 4090 (~$0.40 / GPU-h) | ~21 h | **~$40** (incl. disk) | ~26 h: no retry inside 2 days |
| 4 x A100 80 GB (~$1.2) | ~16.5 h | ~$80 | ~22 h |
| 4 x H100 SXM (~$2.3) | ~10.5 h | **~$95-110** | ~16 h: one shot + one fix |
| 4 x H200 (~$3.2) | ~8.5 h | ~$110 | ~14 h |

PLAN_960's ~$40-60 budget only covers the 4090 route, which cannot finish with a retry inside the 2-day window.
Not a China-located host (10-09 pip / wheel problems).

## 6. Exact steps to a first smoke run (rental; the box has already passed steps 2, 4 and 5 on CPU)

1. Rent: 4 GPUs (H100 80 GB preferred), >= 32 vCPU, >= 128 GB RAM, 2 TB NVMe, CUDA 12.x image, PyTorch >= 2.5
   (2.6+ preferred: `torch.compiler.set_stance`). `pip install timm==1.0.25 pydicom==3.0.1 opencv-python-headless scipy pandas safetensors kaggle`
   (the versions pinned by the orthofoundation dataset wheels).
2. Code: tar `~/r7port/{src,code,scripts,bundle_r7}` on the box (about 15 MB, RSNA-only code, no OAI) -> scp to the rental.
   Weights: `anatomy.pt`, `side_cls.pt` (Kaggle), ConvNeXt-S (HF). Eesh places his own `kaggle.json` on the rental himself.
3. Data: `kaggle competitions download -c rsna-knee-abnormality-detection`, unzip to `/data/rsna` (layout
   `train.csv`, `train_series.csv`, `train_series/<study>/<series>/*.dcm`, as on the box).
4. Mini cache: `python scripts/px_build_all.py --data /data/rsna --code code --src src --anchor w/anatomy.pt --side w/side_cls.pt
   --out /nvme/px_smoke --procs 8 --gpus 0,1,2,3 --uids smoke200.txt` then `--merge`. Check: 0 errors, rows / study ~580, and the
   5 box test studies byte-identical to `~/r7port/test_px` (GPU vs CPU anchors may move a few rows: compare, then accept the GPU version).
5. Targets: `scripts/make_targets.py --labels llm_labels_v4_blend.csv --train train.csv --tiers FINAL2_labels_with_confidence.csv`
   (RSNA-only CSVs from `~/rsna_train`). Then `compact_train.py ... --check-targets`.
   Smoke = fold 0 on the 200-study cache with the real flags (below), `--epochs 2 --ckpt-every 1`. Read s / step, peak memory
   (`nvidia-smi`) and that the loss moves. **Decide the GPU count / checkpoint mode from this number.**
6. Full cache (`--procs 16 --gpus 0,1,2,3`, all 4,407) -> merge -> one process per GPU:
   `CUDA_VISIBLE_DEVICES=k python -u src/compact_train.py --only-folds k --out runs/f{k} <flags>`.

R7-lite flags (R7 run_config minus fm-aux and minus everything that needs state targets):
```
--px /nvme/px384_v4 --targets targets.csv --extra cell_weights.csv --arms E_lrprior=lr:lr_prior:w  (or E_prior)
--backbone convnext_small --init convnext_small_in22k_ft_in1k.safetensors --seeds 1 --seed-base 1 --folds 4 --epochs 16
--batch 2 --train-windows 176 --crop 352 --slices 3 --slice-jitter 0.25 --ema 0.999 --include-review-holdout --syn-decomp
--phys-mm 3.5 --lr 1e-4 --freeze-blocks 16 --llrd 1.0 --head-lr 1e-3 --wd 0.05 --workers 6 --eval-workers 3 --compile
--resume --ckpt-every 1 --eval-stride 1 --train-sampling cover --cover-stride 3 --spatial-attn --mirror 0.5 --aug v2
--aug-grid-fp32 --draw-seed --grad-ckpt (cards < 96 GB) --i-have-user-authorisation-to-finetune
```
For R7-full, add: `--state-targets ours.csv --state-spec state_spec.json --state-weight 0.5 --rare-boost --rare-boost-names
--pair-rank 0.5 --pair-frac 0.33 --extent --extent-weight 0.2 --save-state-oof`.

7. After the folds: merge the `partial_*` OOF; gold58 AUC of the 4-fold mean. Run the R7 bundle itself on the 58 gold
   DICOMs with `compact_infer.py` (about 10 min on 1 GPU) to get a paired R7-vs-ours gold58 comparison and the 2-way blend
   gold58. Pre-register the gate before looking.
8. Inference: **do not** put our folds into the R7 bundle as `--models2`. `compact_infer.py` refuses `--state-stacker` with a
   second group, and the fluid stacker would need a refit, which would change R7's own outputs. Instead, add a second compact
   stage (our own bundle and manifest, no stackers) to the R7 notebook and blend the 3 CSVs by fixed per-class ranks in stage 3.
   Hidden runtime: R7 notebook ~3.3+ h (GUESS from HV's 3.3 h; HR7's own wall is not read yet) plus ~1.5-2 h for the
   second decode, which is under the 9 h limit.
   Use a versioned Kaggle dataset (the "non-versioned" refusal from 10-12).

## 7. OAI rows on the box (task 4)

- The R7 config has no OAI. **In code, OAI rows are possible:** render OAI DICOMs through the same `CStudies` path (this needs a
  competition-style `train_series.csv` with plane / fluid / fat-sat and a `train_series/<study>/<series>/` symlink tree), then
  build masked targets. Masked cells get `lr = 1` (neutral) and `w = 0`. The prior must be filled (the pool base), otherwise the
  study is dropped as "blanked". State cells stay blank (the state loss masks them) and the extent targets are NaN (masked).
  OAI rows fall into the hash folds, so their OOF rows must be filtered before any metric.
- **In time, impossible:** OAI cannot leave the box, so training must run on the busy V100 at ~29 h per RSNA-only fold
  (+~2x steps with ~9k OAI visits), after ~19 h of CPU rendering (9k visits x 30 s / 4 processes). That is several
  days of GPU time per fold, it does not fit before 10-22, and OAI eligibility is still disputed (thread 748276).
  The +.005 that Raptor's author reported for OAI masked rows does not transfer to this route in the time we have.

## 8. Risks (largest first)

1. **Target gap.** R7 = Opus graded report states + 7 refinement rounds + fastMRI+ aux; ours = llm_labels_v4_blend soft labels
   (0.25 = "unknown", high base rates). Most likely result: an R7-architecture model at about our fleet's label quality
   (gold58 ~.92), which by the 10-12 / 10-13 evidence adds < .001 on top of an R7 blend.
2. **R7-lite is not R7.** Without states it loses the state-evidence logit, pair ranking, rare boost and extent. Public
   history: S6 (`r6cnxfsss` manifest: tag cnx_s6, built 10-09, targets `opusv11b_rr3_plv21_cdr`, a state spec in the bundle,
   no fm-aux; its run_config is only partly recorded) scored .954 in the Raptor blend, and R7 (rr7 targets + cell weights +
   pair rank + mirror + extent + fm-aux) scored .956. So heliosli's own target / recipe iterations were worth about +.002
   at the blend level, and our labels are further from rr7 than his rr3 was.
3. **Throughput uncertainty (+-40 %).** The anchor is our V100 ConvNeXt-T measurement; the smoke run settles it.
   torch.compile + `--grad-ckpt` + dynamic shapes may recompile; fallback is eager (-25 %).
4. **4090 memory:** needs the chunked-checkpoint patch (not written yet; ~10 lines; must be checked for identical loss / gradients on CPU first).
5. **Data movement:** 560 GB download speed from Kaggle to the host is unknown; the box itself gets ~15.6 MB/s (10 h for 560 GB).
6. **No clean ruler:** R7's OOF is not public; gold58 (SE ~.014) is the only paired comparison.
7. **Inference integration:** the second compact stage doubles decode time; it fits the 9 h limit but adds a new failure mode.
8. A new LLM state extraction (R7-full) brings its own risks: category drift vs the R7 spec priors, and multilingual
   reports (tr / hr / el / bg).

## 9. Go / no-go for the 2-day window

- **NO-GO** for the .960 push (10-13 -> 10-15). The machinery is ready, but the inputs that make R7 strong are not
  reproducible in 2 days. Best case is about 16 h to an LB read for ~$100, with ~10 % (GUESS) odds of a visible public gain.
- **Revisit trigger:** HFR7-R20 (R7 + our fleet at .2) reads **.957**. That would show our labels add on top of R7, and an
  R7-architecture member on our labels becomes the natural next member.
- **Optional slow GO (final pick B, not the 2-day push):** if Eesh approves ~$40-60, run R7-lite on 4 x 4090 (after a
  15-minute smoke run picks the checkpoint mode). LB read ~26 h after renting, ready for the 10-22 final-pick decision.
  Pre-register the gate (gold58 paired vs R7, OOF) and the blend weights first.
- Box: nothing to run. Its GPU stays on the OAI queue. `~/r7port` (2.2 GB: code, weights, 5-study test cache, our targets)
  is ready to tar for a rental.
