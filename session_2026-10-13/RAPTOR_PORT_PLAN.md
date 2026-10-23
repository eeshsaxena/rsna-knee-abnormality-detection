# RAPTOR PORT PLAN: a second Raptor (nartaa .949 CoAtNet MIL) seed trained by us (written 10-13 ~16:30 IST)

## >>> EXECUTION STATUS (main: GO for path (a), no spend; updated 10-13 ~18:15 IST) <<<
- Kit: laptop `session_2026-10-13/raptor/` (box scripts in `box/`, Kaggle prep kit in `prep/`), box `~/raptor2/`, data
  `~/raptor_store/` (mode 700; OAI parts never leave the box), fetch dir `~/raptor_in/`.
- **PREREG** `raptor/PREREG_RAPTOR_SEED.md` sha256 bc2cef53... (recorded in `PREREG_RAPTOR_SEED.sha256`, box copy checked by the window script).
- **Corpus:** Kaggle CPU kernels `eeshsaxena/rsna-raptor-d96-prep-s{0,1,2}` (private, GPU off, internet off) COMPLETE 17:58
  (pushed 17:31; 3 x 1,469 studies, 11 tar parts each, manifests OK; Kaggle numpy 2.1.3 / cv2 4.14.0 / pydicom 3.0.2).
  Box-mirror reference build with the same kernel code: 458/458 OK in 4.2 min. Fetch to the box via signed URLs (~44 MB/s;
  pass 1 lost 10 parts to HTTP/2 stream errors, pass 2 with `--http1.1 --retry-all-errors` got them): 33/33 parts sha OK.
  **CORPUS_OK 18:39:** 4,407/4,407 volumes unpacked to `~/raptor_store/rsna` (59 GB, `rsna_index.csv`), each = its Kaggle
  receipt sha256. Cross-check: box rebuild in the numpy-2.1.3 env (`kmimic`) is byte-identical to the Kaggle build 458/458;
  the numpy-1.26 env build differs by exactly 1 uint8 level on ~0.5 % of pixels (NumPy 2 float32 promotion, NEP 50), so the
  Kaggle build = the serving environment. Full-index CPU dry run: 6,748 rows (4,349 RSNA + 2,399 OAI), OK.
- **OAI d96 (approx):** 2,399/2,399 knees, 62 valid slices each (slots 1 + 3 empty), 153 s, 32 GB, box only.
- **Targets:** `~/raptor2/labels_raptor_rsna_blendjev_sol.csv` md5 f1fa9d51 (gold58 .9042).
- **Trainer** `train_raptor.py` (md5 13c78df0 at arming): CPU dry runs PASS (train, eval, 3-best SWA, swa-only rebuild,
  strict reload at serve img_size 320, channels_last on/off).
- **GPU window:** `~/raptor2/window.sh` (pid 550546, armed 17:48) waits for L13 f0 (f0_s1044_ttr_frac75oaiT123v00, ETA ~23:30)
  to log DONE, touches `~/queue/PAUSE` inside the runner's 120 s free-check gap, runs the smoke (<= 30 min), and if SMOKE_GO +
  CORPUS_OK + PREREG sha + FULL_APPROVED launches the full run (16 ep if the projection <= 26 h, else 12) in the same window;
  it ALWAYS removes `~/queue/PAUSE` when it exits. Logs: `~/raptor_store/smoke/window.log`, `verdict.txt`, `run_s42/train.log`.
- **Path (b) RUNNING (Vast, RSNA-only, main + Eesh approved $25):** instance 55762630, 4x RTX 4090 Taiwan $1.898/h, seeds 43-46 in
  parallel since 20:57, measured 0.108-0.112 s/step (bf16), ~9.5 min/epoch -> ETA ~23:30; spend so far $1.73 (21:26), projected
  ~$6.2. Rental smoke passed first. SV gold58 predictions are scored on the rental (`sv_watch.sh`) and pulled with each seed by the
  box fetch loop. Details: `raptor/VAST_RAPTOR.md`. Gate: `raptor/PREREG_RAPTOR_RENTED.md` (sha 9117ee60, written before any
  seed read), script `box/gate_raptor_rented.py` (plumbing-tested).
- **10-14 00:25 path (b) DONE: RAPTOR_RENTED_GATE PASS** (`raptor/RAPTOR_RENTED_RESULTS.md`): seeds 43-46 SWA gold58 .904-.912,
  SV .911-.914; Rr(4) SV .9179 vs N .9229; rank-mean(N, Rr) .9240 (+.0011); Spearman(Rr, N) .949; with F .9268 = N-side.
  Instance self-destroyed 00:37; spend **$8.21**. Kernel seeds k = 2 (s46 + s45, PREREG_RAPTOR_KERNELS Amendment 1);
  dataset folder `raptor/ds_raptor_seeds/` (0.59 GB) handed to main 00:33; kernels `raptor/kaggle/kernel_hr7rs/`,
  `kernel_hfr7rs/` built (sim PASS), push after main creates the dataset.
- **10-14 01:08 HR7-RS v2 commit: RS_CHECK PASS 34/34** (main pushed; v1 fell back because stage 1c dumped the seeds' prob-mean but
  wrote a per-arm rank-mean as its rank file; v2 ranks the dumped mean). HFR7-RS rebuilt with the same fix (sha f374cc48), main
  pushes / gates / submits. Weight-soup test (s46 + s45 averaged weights) = gold58 .9000 < each seed: soup fails, so **no second
  Vast batch** (RAPTOR_RENTED_RESULTS.md). Vast credit left $17.23.
- **Box seed S42 after the run:** `~/raptor2/window3.sh` (armed 01:07) waits for window2.sh to exit, re-takes the queue gap for
  ~3 min of V100 (serve_check S + N), runs `gate_raptor.py` (PREREG_RAPTOR_SEED) -> `~/raptor2/GATE_RESULT.txt`, then removes
  ~/queue/PAUSE. If PASS: S is OAI-trained -> separate private dataset after an OAI-id scan of the checkpoint dict, OAI pick only,
  and a new kernel prereg (e.g. ours = S42 + s46) before any build.
- **10-13 23:15 path (a) box full run RUNNING** (`~/raptor_store/run_s42`, fp16 + grad_ckpt, 0.37 s/step, ETA ~10:45 10-14; queue
  PAUSED by window2.sh until it ends). Gate after it: `box/gate_raptor.py` (PREREG_RAPTOR_SEED).
- **Kaggle integration prepared:** `raptor/kaggle/build_hr7rs.py` builds HR7-RS (on HR7-RR v2)
  and HFR7-RS (on HFR7-RR v1) with stage 1c = our seeds (r384 serve path, 60 min cap, RAPTOR_SEEDS_FALLBACK = parent) and
  Raptor_mix r384 .45 / r224 .25 / ours .30 (`raptor/PREREG_RAPTOR_KERNELS.md`, sha 5fb2554c). Stage-3 sim `sim_stage3_rs.py`
  PASS (A recompute, B/C fallback byte-identical to the parent, D r224 failed). Commit gate `rs_check.py`. Dataset folder
  builder `box/pack_seeds.py` (OAI-string scan of the checkpoint dicts, sha manifest, seeds.json).

Scope: feasibility + port plan only. No GPU used, ~/queue not touched, CPU work on the box only under
`nice -n 19 taskset -c 8-15` with CUDA hidden (own PIDs only, listed in sec 11). No OAI byte left the box.
Does not duplicate the R7 trainer study (`R7_PORT_PLAN.md`, other agent).

Tags: MEASURED (we ran / read it), CLAIMED (nartaa's post), MODEL (arithmetic on measured data), GUESS.

---

## 0. Bottom line

| Path | Verdict | Why |
|---|---|---|
| (a) Box V100 WITH OAI (exact nartaa recipe, our labels) | **CONDITIONAL GO** | Every input is reproducible and the risky parts are already VERIFIED on box CPU: OAI label rules reproduced EXACTLY (2,396 / 2,397 / 1,659 labelled knees on 2,399 = nartaa's counts); our d96 corpus build + eval geometry re-score N at gold58 **.919111 vs its stored .918952** (sec 2.2); OAI images can come from `~/oai_store` (N loses nothing on store-derived inputs, sec 2.3), so no NDA re-download is needed. Cost: ~1 day prep (mostly unattended) + **~15-26 h V100** (central ~20 h, MODEL) + ~0.5 day to kernel. Fits before the 10-20 12:00 freeze with ~3 days slack if the GPU is given to it by ~10-14 evening. Needs: GPU priority over the L13 / allv filler (main session / Eesh), 3 Kaggle CPU prep kernels on Eesh's account (RSNA DICOMs are not on the box). |
| (b) Rented GPU WITHOUT OAI (RSNA rows only) | **GO only as an OAI-free hedge, if Eesh funds ~$10-25** | 3.5-6.5 h per seed on one A100 / 4090 (MODEL), seeds in parallel. nartaa's own OAI-free rows scored .940-.942 (CLAIMED), so it is a weaker member than (a); its real value is that it is OAI-clean (useful if thread 748276 rules OAI out, which would also taint N and every R7 copy). |

Expected value (GUESS): a second Raptor-family member at Raptor-slot weight .5 next to N: +.0003 to +.001 on private; P(visible
public +.001 on top of HR7 / HFR7) ~20%. Too slow for the next 2 days of public reads; it is a candidate for the 10-22 pick B.

---

## 1. The recipe (746792 full read, README + checkpoint-manifest, N's checkpoint metadata, our HR7 Raptor source)

Sources: thread 746792 main post + nartaa's 10-13 reply (all 17 comment threads read with nested replies expanded, MEASURED); dataset
`nartaa/rsna-knee-publication-swa-weights-20261007` README.md + checkpoint-manifest.json (downloaded, MEASURED); N's `.pt` dict
keys (box CPU, MEASURED); `session_2026-10-13/hr7/kernel/rsna_sub_hr7.ipynb` cell 2 = `/tmp/raptor_run.py` (verbatim copy on
the box: `~/raptor2/raptor_ref.py`, md5 e53ed4f4).

**Code availability:** training code is NOT public (`finetune_raptor.py`, `cache/build_corpus96.py` are only named in source
comments). Public: the two static inference notebooks (`nartaa/rsna-knee-0949-anatomical-mirror`, `nartaa/rsna-knee-0945-efficient-224crop`),
the weights dataset, and the 10-13 training description. Not public and never will be: their Gemini 3 Flash labels and the
OAI label rows (NDA). The model class (`RaptorClassifier`, "verbatim from finetune_raptor.py") and the eval windowing
("verbatim from finetune_raptor.py StudyWindows (train=False)") ARE in the scoring source, so train/serve parity is reachable.

| Item | Raptor .949 (N) | Source |
|---|---|---|
| Backbone | timm `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, `num_classes=0, global_pool='avg', in_chans=3`, ImageNet init (`ck['src']='timm-pretrained'`), 73.17 M params incl. head | ck dict + source (MEASURED) |
| Head | `RaptorClassifier`: LayerNorm(768) -> per-finding attention MLP (768-256 tanh, dropout .2, 256-12), softmax over windows, per-finding pooled features, per-finding linear (clsW, clsb) | source (MEASURED) |
| Corpus ("d96") | 96 slices/study, 5 slots in order: Sagittal fluid 26, Sagittal non-fluid 22, Coronal fluid 18, Coronal non-fluid 12, Axial any 18 (pick by `test_series.csv` Anatomical_Plane + Fluid_Sensitive; fallback to any unused series of that plane; missing slot = zeros + mask 0) | source (MEASURED) |
| Slice picks | per series, `linspace(int(.02 n), int(.98 n)-1, k)`, ordered by position along the slice normal | source |
| Intensity | per series, one 2 / 98 percentile window over the PICKED slices (modality LUT, MONOCHROME1 inverted), clip to [0,1] | source |
| Geometry | 140 mm centre crop by PixelSpacing, `cv2.resize(..., (384,384), INTER_AREA)`, uint8 | source |
| Window | 3 neighbouring slices (c-1, c, c+1) as RGB; centres = `_eval_centers` over the valid-mask range (windows may straddle slot borders) | source |
| Train input | **native 384, full field (no crop)**, 12 random windows / study / step | 746792 reply (CLAIMED) |
| Augmentation | only the window sampling + **+-10 % intensity jitter per study**; no flips, no affine | CLAIMED |
| Batch | 1 study, grad accumulation 16; 6,748 rows -> floor(6748/16) = 421 updates/epoch, 6,736 in 16 epochs (= the "4,336 -> 6,736" in the main post) | CLAIMED + MODEL |
| Optimiser | AdamW wd .0588; backbone LR 4.01e-5, head LR 1.95e-3; OneCycle, 15 % warm-up, 16 epochs; grad-norm clip 3.0; bf16 autocast; no EMA; seed 42 | CLAIMED |
| Loss | BCE-with-logits, pos_weight = (1-p)/p, p = RSNA prevalence over the 4,349 RSNA studies only, p clipped .03-.7, weight clipped 1-10 (Effusion = 1.0); weighted cell losses SUMMED and divided by 12 x studies in batch, masked cells included (rows with masked cells weigh less) | CLAIMED |
| Targets RSNA | blend = mean(own Gemini 3 Flash soft labels, stevenleehans v4_blend), but v4 == 0.25 ("not addressed", ~19 % of cells) -> Gemini alone; blendjev = mean(blend, riadmohamed42 JEV HYBRID) where JEV exists | CLAIMED |
| Targets OAI | 2,399 V00 knees with a MOAKS read (kMRI_SQ_MOAKS_BICL* + FNIH), hard 0/1: PF OA = full-thickness loss >= 1 in PM/PL/FMA/FLA; Lateral OA = same over FLC/FLP/TLA/TLC/TLP; Synovitis = MSYIC >= 1; 1 if any valid subregion hits, 0 only if all needed subregions valid and none hits, else masked; other 9 findings masked. Counts 2,396 / 2,397 / 1,659 | CLAIMED, **reproduced exactly on the box** (sec 3) |
| Split | all-data: 4,349 RSNA + 2,399 OAI = 6,748 rows, no held-out fold; gold58 excluded from gradients, used for per-epoch selection | CLAIMED |
| Selection ("SWA") | 3 best epochs by gold58 macro AUC with 24 evenly spaced windows; uniform weight average incl. BN buffers (not recomputed). N = epochs [14, 11, 12] (0-based), stored gold_auc .918952 | CLAIMED + ck (MEASURED) |
| Serving | 94 windows (all centres of the 96-slice volume), **centre crop [32:352] = 320 px, no resize**, model built with `img_size=320`; fp16 autocast; anatomical mirror TTA (sagittal: reverse the 3 slice channels; coronal/axial: flip width; decided by the centre slice's slot; applied on uint8 before normalisation); mean of the 2 sigmoid vectors; per-finding percentile rank | source (MEASURED) |
| Compute they quote | 16 epochs ~22 h on an 8 GB laptop GPU; hidden turnaround of the mirrored entry 26.1 min incl. queue | CLAIMED |
| Ablations | OAI rows +.005; 384 full field +.003; 320 serve crop 0 (304: -.001, 288: -.002); mirror +.001; 94 -> 24 windows -.006 (older model); single epoch 14 vs SWA -.001; fresh run with BN replay -.004; fresh native-320-crop training -.003 | CLAIMED (public LB, not isolated) |

Implication of the last two rows: **seed-to-seed spread of this recipe on public is ~.002-.004** (two "fresh" retrains scored
.944 / .946 vs .948 / .949). A second seed from us will most likely land .943-.948 alone; its value is as a 2nd member, not alone.

---

## 2. What exists on the box vs what Raptor needs

### 2.1 Inventory (MEASURED 10-13)

| Need | On the box | Gap |
|---|---|---|
| RSNA DICOMs, 4,407 train studies | **only 458** (`~/kmimic/kaggle/input/...`, fold 0 + gold58; HANDOFF 10-05 K4 note) | build the d96 corpus on **Kaggle CPU** (precedent `eeshsaxena/rsna-fleet-prep-s0/s1`: 2,204 studies in 24.9 min on 4 CPUs, MEASURED log) and fetch the tar parts to the box |
| d96 / 384 corpus | none. `~/rsna_shards_384*` are an older 44.8-slice layout; public dreaddevelopment corpus = 44 x 336 (old recipe); `shikhageekachad` native384 = 170 mm / P1-P99. None matches | new build (sec 4 step 1) |
| Corpus size | reference build 1.4 s/study on 1 box process (MEASURED, 8 studies); 14.2 MB raw, **8.3 MB as PNG-3**, 10.8 MB savez_compressed | RSNA 62.5 GB raw / **36.6 GB PNG** -> 3 Kaggle shards (~12 GB each, under the ~19.5 GB output cap) |
| OAI images | raw tarballs **deleted** 10-09 (81 GB pruned). `~/oai_store` = fleet format: 320 px, JPEG q92, <= 32 slices/series, window .5/99.5, 3 views (AX_MPR, COR_MPR, SAG_IW_TSE); COR_IW_TSE was downloaded but never stored | exact path: re-download V00 (10,771 files, 81 GB; 10-08 run was download-bound at 37 knees/min = ~75 min) with `downloadcmd -t` over `converted_manifest.csv` paths, Eesh types the NDA password; or approximate from the store (sec 2.3) |
| OAI -> Raptor slots | slot 0 Sag FS = SAG_IW_TSE; slot 1 Sag non-FS = **none** (no OAI series; empty, as for nartaa unless he used DESS); slot 2 Cor FS = COR_MPR (DESS WE); slot 3 Cor non-FS = COR_IW_TSE; slot 4 Axial = AX_MPR (from `oai_series_meta.csv` plane / FS, 0 mismatches) | the stray cross-plane image in every MPR tarball must be dropped first (reuse `build_oai_store.extract_series`) |
| OAI labels | `~/oai_build/oai_label_sources.csv` (merged V00 MOAKS reads, 2,680 knees) | **done**: `~/oai_build/oai_labels_raptorT1.csv` (mode 600), counts = nartaa's exactly (sec 3) |
| RSNA label sources | `llm_labels_v4_blend.csv`, `labels_llm_gpt56sol.csv` (lixin GPT-5.6, binary), `report_labels_v2.csv` (pilkwang), `~/jump2/labels_in/jev_hybrid_scores.csv` (4,407 rows), `labels_frac75_logit.csv` | Gemini half is not public: substitute (sec 3) |
| gold58 | `~/rsna_train/gold74_eval_set.csv` source == original_gold58 (58 rows); DICOMs in the 458 mirror | none |
| Pretrained weights | HF hub reachable from the box (HTTP 200 on the model API); only the `_224` CoAtNet is cached | one `timm.create_model(..., pretrained=True)` download (~295 MB) |
| Software | env `rsna`: torch 2.5.1+cu121, timm 1.0.29; env `kmimic` (Kaggle mirror) has the same timm | none; A0 load test in `kmimic` |
| GPU | V100-FHHL 16 GB, **no bf16 tensor cores** | fp16 autocast + GradScaler + NaN guard instead of bf16 (risk R1) |

### 2.2 Parity of OUR corpus build with nartaa's (box CPU, MEASURED, `~/raptor2/gold_parity.py`, 32 min)
N re-scored from OUR d96 build (verbatim reference reader) of the 58 gold DICOMs on the box, fp32 CPU:

| Geometry | gold58 macro | vs N's stored gold_auc .918952 |
|---|---|---|
| **full field 384, 24 evenly spaced windows** | **.919111** | **+.00016; 11 of 12 labels within .001, Synovitis .836 vs .839** |
| centre crop 320, 24 windows | .918530 | -.00042 |

So (1) our corpus build reproduces nartaa's training corpus (for these studies), (2) his per-epoch gold selection used the
training geometry (full field 384, 24 windows), (3) the model class / state dict / eval windowing are right. The
port's preprocessing and eval are therefore VERIFIED before any GPU time; only the training loop is new code.

### 2.3 OAI from the store vs exact re-download (box CPU, MEASURED, `~/raptor2/approx_parity.py`, 37 min)
Proxy test on the 58 gold RSNA studies (no OAI involved): N scored, 24 windows, full field 384, on
(exactB) the exact build with slot 1 (Sagittal non-FS) zeroed = what an exact OAI re-download gives, vs (approxB) the
store-derived approximation = what OAI rows built from `~/oai_store` give (320 px JPEG q92 store -> same 2-98 % slice
span on the stored stack -> 2/98 percentile re-window on the uint8 -> linear upsample to 384; slots 1 AND 3 empty).

| Input | gold58 macro | PF OA | Lateral OA | Synovitis |
|---|---|---|---|---|
| full exact (all 5 slots) | .919111 | .840 | .778 | .836 |
| exactB (slot 1 empty) | .918375 | .848 | .772 | .833 |
| **approxB (store-derived, slots 1 + 3 empty)** | **.920458** | .874 | .752 | .838 |

Prediction agreement exactB vs approxB: per-label Spearman .909-.990 (median ~.98), mean |prob diff| .040; mean pixel
|diff| 23.7 uint8 levels on shared slices (slice positions differ when a series had > 32 slices). **N does not lose AUC on
store-derived inputs** (difference within the gold58 noise, SE ~.01). Verdict: OAI rows can be built TODAY from
`~/oai_store` (no re-download, no Eesh step). The exact re-download remains an optional upgrade (adds the COR_IW_TSE slot and
removes the JPEG / upsampling texture shift, which a model could in principle use to recognise OAI rows).

---

## 3. Labels (box CPU, MEASURED)

**OAI:** `~/raptor2/build_oai_labels_raptor.py` applies nartaa's rules to our merged V00 MOAKS reads:
PF OA 2,396 labelled (pos .434), Lateral OA 2,397 (pos .195), Synovitis 1,659 (pos .580), 2,399 knees with >= 1 label.
**Identical to nartaa's published counts**, so our OAI rows = his OAI rows (up to image matching; he used 3DReasonKnee, we use
the image03 subject + side + V00 selection).

**RSNA (the Gemini half is the one missing input).** gold58 macro AUC of candidate target tables (`~/raptor2/label_gold58.py`):

| Target table | gold58 macro | Note |
|---|---|---|
| v4_blend | .8927 | v4 == 0.25 in 19.1 % of cells (= nartaa's "about 19 %") |
| sol56 (lixin GPT-5.6, binary) | .8352 | closest public analogue of "our own frontier-LLM read" |
| JEV HYBRID | .9037 | tuned on gold58 by its author (in-sample) |
| **blendjev_sol** = mean(mean(sol56, v4) [v4 == .25 -> sol56], JEV) | **.9042** | the recipe's structure with sol56 in place of Gemini |
| blendjev_pilk | .8938 | pilkwang already enters via JEV |
| v4jev | .8972 | |
| frac75 (ours: v4 + image teacher) | .9300 | image-teacher labels; the teacher carries Raptor-family signal, so a Raptor trained on it would correlate with N (diversity loss) |

Proposal (pre-register before launch): **blendjev_sol** (fidelity to the LB-verified recipe; text-only targets keep the
diversity on the image side). frac75 is the documented alternative if the main session prefers target strength over
fidelity. Gold58 is in-sample for JEV (as for nartaa), so it is a diagnostic, not a gate on the label choice.

---

## 4. Port: exact steps (all code new under box `~/raptor2/` and a laptop kit `session_2026-10-13/raptor/`)

1. **RSNA d96 corpus on Kaggle CPU** (no GPU quota): kernel script = the prep-s0 pattern (tar parts ~1 GB, receipt, manifest with
   versions) with the reader replaced by the VERBATIM `_make_reader` + `_pick_series_for_slot` + `build_study` from
   raptor_ref.py (d96, IMG 384, CROP_MM 140). Store per study ONE lossless PNG mosaic (96*384 x 384, compression 3) + mask, and
   in the manifest the sha256 of the raw `vol.tobytes()`. 3 copies `eeshsaxena/rsna-raptor-d96-s{0,1,2}` (SHARD literal),
   4,407 / 3 studies each, ~12 GB output each, ~15-25 min each (MODEL from prep-s0 timing + PNG encode). Gold58 inside.
2. **Fetch to box** with the `~/fleet_in/fetch_all.sh` pattern (signed URLs listed on the laptop, curl on the box), unpack to
   `~/raptor_store/rsna/<sid>.npy` (uint8 96x384x384, raw, mmap-able; 62.5 GB) + `masks.npy`; verify every decoded vol
   against the manifest sha256. **Cross-check:** rebuild the 458 mirror studies on the box with raptor_ref.py and require
   byte equality with the Kaggle build (catches cv2 / numpy drift; Kaggle build wins because the hidden rerun uses Kaggle's image).
3. **OAI d96 corpus** (box only, mode 700, `~/raptor_store/oai/`, the 2,399 labelled V00 knees):
   - DEFAULT (no Eesh step): `approx_vol()` of `~/raptor2/approx_parity.py` applied to `~/oai_store/oai_<id>.npz` (V00 files
     only, store view -> slot map v3 -> slot 0, v2 -> slot 2, v0 -> slot 4; slots 1 and 3 empty); ~15 min on 8 cores. Justified by sec 2.3.
   - optional exact upgrade: re-download the 10,771 V00 tarballs (`downloadcmd -t <list>` from `converted_manifest.csv`; Eesh
     types the NDA password; ~75-90 min), then per knee `extract_series` (stray-image drop, unchanged) -> series table ->
     raptor_ref `build_study` -> `.npy`; prune raw after sha-verified reload (same rule as 10-09).
4. **Label files:** `labels_raptor_rsna.csv` (4,407 x 12, blendjev_sol) and the union table = RSNA 4,349 (gold58 removed by id) +
   `oai_labels_raptorT1.csv` (2,399; 9 NaN columns). pos_weight from the 4,349 RSNA rows: mean target, clip .03-.7, (1-p)/p clip 1-10.
5. **Trainer `train_raptor.py`** (one file, ~250 lines):
   - model = `raptor_ref.RaptorClassifier(timm.create_model(ARCH, pretrained=True, num_classes=0, in_chans=3, img_size=384, global_pool='avg'))`
     (do NOT call raptor_ref.build_backbone: it hard-codes img_size=320 for serving);
   - dataset: per row, np.load(mmap) vol + mask; 12 window centres drawn without replacement from the `_eval_centers`
     candidate set (lo+1 .. hi-1 of the valid mask; with replacement if fewer than 12); triplet (c-1, c, c+1) / 255;
     x *= U(.9, 1.1) per study, clip [0, 1]; ImageNet normalise; channels_last; shuffle all 6,748 rows each epoch;
   - masked BCE (pos_weight above, NaN targets weight 0), sum / (12 x B); accum 16 (drop the last partial group, = 421 updates / epoch);
     AdamW (2 groups); OneCycleLR(max_lr=[4.01e-5, 1.95e-3], pct_start .15, total_steps 16 x 421); clip 3.0 after unscale;
   - precision: fp16 autocast + GradScaler on V100 (bf16 on Ampere / Hopper rentals); skip-step + counter on non-finite loss;
     abort if > 1 % of steps non-finite;
   - every epoch: gold58 with 24 evenly spaced windows at full field 384 (`_eval_centers(mask, 96, 24)`, eval mode, fp16),
     macro + per-label AUC; save `ep{E}.pt` (state_dict fp32, ~293 MB);
   - end: top-3 epochs by gold58, uniform average of every float tensor incl. BN running stats (num_batches_tracked from the
     best epoch), save in N's exact dict format `{'model','gold_auc','aucs','src':'timm-pretrained','res':384,'arch','lab','epoch','swa_over'}`;
   - `--smoke N`: stop after N optimizer updates, print s/step, peak memory, non-finite count, then run the full end path on
     whatever epochs exist (proves SWA + save + reload).
6. **Serve parity test (box, A0):** load the SWA with the HR7 Raptor stage code (model at img_size 320, 94 windows, crop
   [32:352], mirror), only the hash constant changed; run on gold58 on the box (kmimic env) and require (i) strict load,
   (ii) no fallback, (iii) gold58 macro >= the 24-window training-time number - .01.

### 4.1 Smoke run on the box (exact sequence)
1. CPU dry run (now possible, no GPU): `CUDA_VISIBLE_DEVICES= nice -n 19 taskset -c 8-15 python train_raptor.py --device cpu
   --rows 4 --windows 2 --smoke 2 --pretrained 0` on 2 RSNA + 2 OAI rows of the 458 mirror / store-approx (catches shape,
   mask, loss, SWA, save / load bugs).
2. GPU smoke (needs a queue boundary; main session puts ONE line at the top with qctl, ~30-40 min):
   `train_raptor.py --smoke 60` (= 960 studies, ~10-15 min at 0.6 s/step) on the real union. Read: s/step (decides 16 vs 12
   epochs, sec 5), peak memory (< 14 GB), non-finite steps (0 expected), gold58 eval runtime (~1-2 min).
3. A0 serve test (sec 4 step 6) on the smoke SWA, then the full run as one queue line (16 epochs).

---

## 5. Time estimates

**FLOPs (MEASURED with torch FlopCounterMode on box CPU, conv + matmul only; timm's table says 47.7):** CoAtNet-2 rmlp
= 42.8 GMACs / window at 384 (29.8 at 320) -> 12 windows = 514 GMACs forward per study-step. Our TTR224 step = bs 2 x
3.3-3.9 views x 16 triplets = 106-125 ConvNeXt-T images x 4.45 GMACs = 470-556 GMACs: **the same forward compute per step**,
measured 0.37-0.39 s/it on this V100 (MEASURED, current allv log). The box's earlier CoAtNet-224 MIL run
(`~/rsna_train/pipe_coatnet_g1.log`, 8 windows / step, no accumulation) ran 0.25 s/step for ~120 GMACs, i.e. ~2-3x worse
FLOP efficiency than ConvNeXt (attention without flash kernels on sm70, per-step AdamW), and it **went NaN in fp16** once the
backbone was unfrozen (MEASURED). So the V100 estimate for the Raptor step is **0.45-0.85 s/step, central ~0.65** (MODEL);
nartaa's 8 GB laptop GPU did 0.73 s/step (22 h / 16 / 6,748, CLAIMED), consistent with our V100 being about a 3090 / 2 (10-09 Vast: one 3090 = 1.95x the V100 on ConvNeXt-288, MEASURED).

| Run | Steps | V100 (box) | Rented (bf16) |
|---|---|---|---|
| With OAI, 16 ep, 6,748 rows | 107,968 | **15-26 h** (central ~20 h) + 16 gold evals (~0.5 h) | n/a (OAI never leaves the box) |
| With OAI, 12 ep fallback (if smoke > 0.8 s/step) | 80,976 | 11-19 h | n/a |
| RSNA only, 16 ep, 4,349 rows | 69,584 | 9-16 h | A100 80 GB ~0.19-0.26 s/step -> **3.7-5 h**; 4090 ~0.22-0.33 -> 4.3-6.4 h; H100 ~2-2.5 h (GUESS ratios 2.5-4x V100) |
| Corpus prep | 4,407 RSNA studies | n/a | 3 Kaggle CPU kernels ~15-25 min each + fetch 36.6 GB (1-3 h, GUESS) + unpack / verify ~30 min |
| OAI build | 2,680 knees | exact: ~1.5 h download + ~15 min build; approx: ~15 min | n/a |
| Hidden scoring cost of a 2nd Raptor arm | per study 94 x 2 windows at 320 on T4 | +10-20 min on the ~3.3 h R7 wall (GUESS; the Raptor stage is ~25 min total incl. prep, and prep is shared) | same |

Rented cost (b): ~$2-3 per seed on a 4090, ~$4-7 on an A100 (Vast list prices ~$0.35-0.45 / 1.0-1.4 per h), plus ~1 h setup and
corpus push per instance. Vast credit today $0.44, Modal ~$1.3: needs Eesh to top up; the R7-recipe job (PLAN_960) can share
the same instance (both need the RSNA DICOM-derived caches, different layouts).

**Calendar for (a), if Eesh says GO tonight (MODEL):** 10-13 night prep kernels + trainer + CPU dry run; 10-14 morning fetch /
unpack / OAI build; 10-14 midday GPU smoke at a queue boundary; full run 10-14 afternoon -> 10-15 late evening (central);
10-16 SWA + serve parity + weights dataset (293 MB, OAI-clean scan: weights + manifest only) + kernel + box A0; first
submission 10-16 / 10-17. Slack to the 10-20 12:00 freeze: ~3 days (room for a 2nd box seed or nothing).

---

## 6. Kernel integration (after a seed exists)
- Template = HR7-RR (`session_2026-10-13/hr7rr/`, `build_hr7rr.py`, `HR7RR_RESULT.md`): a stage-1b writefile + runner cell
  (own process, time cap, `RR_FALLBACK` on any failure) and a stage-3 `Raptor_mix` of per-label percentile ranks. Our seed
  is simpler than r224: it uses the r384 scoring source UNCHANGED except the weight filename + the two sha256 asserts
  (same arch, 320 crop, 94 windows, mirror, fp16), written to `/tmp/raptor_ours_run.py` -> `raptor_ours_raw.csv`.
  Stage 1c must run before the R7 assets cell (that cell pip-installs timm 1.0.25 over the image's 1.0.29).
- Raptor_mix weights fixed in a prereg BEFORE any read, e.g. r384 .5 / ours .5 (or r384 / r224 / ours .40 / .25 / .35).
- Fallback rules as in HR7-RR (missing / non-finite / > 3 % failed / Spearman vs own sub < .99 -> drop our arm).
- Weights go to a NEW private dataset (OAI-trained weights only, scanned like k12w / k14w); OAI-trained kernels go to the
  OAI pick only (HANDOFF sec 5). The (b) seed is OAI-free and could also sit in an OAI-free pick.

## 7. Gates (to pre-register before launch, suggested)
- G0 smoke: s/step <= 0.85 (else 12 epochs), peak memory <= 14.5 GB, 0 non-finite steps in the smoke.
- G1 seed quality: SWA gold58 (24 windows, full field) >= .905 (N stored .9190; nartaa's two failed fresh runs still scored
  .944-.946 public, so .905 is a "the port is not broken" floor, not a target).
- G2 blend: gold58 of rank-mean(N, ours) at the serving geometry (320, 94 windows, mirror; box GPU A0 after training) >= N alone
  - .002 (both scored by the same code on the same 58).
- G3 serve parity: kernel output on the 3 public test studies + box A0 equal (max abs < 1e-4) to the box recompute.

## 8. Risks
| # | Risk | Mitigation |
|---|---|---|
| R1 | fp16 overflow on V100 (our CoAtNet-224 MIL run went NaN after unfreezing) | GradScaler, clip 3.0 after unscale, skip non-finite steps, abort at > 1 %; smoke must show 0; fallback: fp32 for the transformer stages (slower) or rent |
| R2 | Throughput (0.45-0.85 s/step is a model, not a measurement) | the smoke measures it in 15 min; 12-epoch fallback pre-registered |
| R3 | Labels: Gemini half replaced by sol56 (binary, weaker read) | blendjev_sol still has gold58 .904 (v4 .893); diversity from N is a plus |
| R4 | Unknown trainer details: window sampler (uniform over candidates vs valid-only), jitter form (multiplicative vs additive), OneCycle anneal shape, BN behaviour with 12 correlated windows | choices fixed in sec 4 step 5 and logged; seed spread on public is already ~.002-.004 per nartaa |
| R5 | OAI approximation (store-derived): blur / JPEG / window shift, Cor non-FS slot empty; a texture cue could let the model tell OAI rows apart | sec 2.3: N itself is insensitive (gold58 .9205 vs .9184 exact); exact re-download path ready if Eesh wants it |
| R6 | Corpus parity (Kaggle cv2 / numpy vs box) | build on Kaggle (= the hidden-run image) and byte-compare 458 studies against a box build |
| R7 | GPU contention: L13 seed-1044 OAI folds (~30 h) and the allv folds sit in the queue; the R7 port wants GPU too | main session decides priority; the rented path (b) avoids the box entirely |
| R8 | OAI ruling (748276 unanswered) | (a) is in the same risk class as N / HR7; (b) is OAI-clean |
| R9 | gold58 in-sample (selection + JEV tuned on it) | same as nartaa; G2 checks relative value only |
| R10 | Hidden runtime | +10-20 min on a ~3.3 h wall; keep the time guard pattern of HR7-RR |
| R11 | Disk: 62.5 + 38 GB raw npy | 6.9 TB free (MEASURED) |

## 9. Decisions needed from Eesh / main session
1. GO for (a)? If yes: (i) allow the 3 Kaggle CPU prep kernels on his account; (ii) give the GPU to the Raptor seed (smoke ~40 min,
   then ~20 h) ahead of the L13 / allv filler; (iii) OAI images from `~/oai_store` (default, sec 2.3) unless he wants the
   exact re-download (one NDA session, he types the password).
2. GO for (b)? Budget ~$10-25 (2-3 OAI-free seeds in parallel), provider Vast or RunPod, can share the R7-recipe instance.
3. Target table: blendjev_sol (default) or frac75.

## 10. What NOT to do
- Do not fine-tune from N's weights to save time: cheap but nearly a copy of N (no diversity), and it inherits N's OAI exposure.
- Do not train at 320 crop (nartaa: fresh native-320 training lost .003) or with flips (all-plane flip lost .001).
- Do not mix our 320-px fleet store into the Raptor corpus for RSNA rows (different window, JPEG, 4-view layout).

## 11. Evidence log (box CPU, 10-13; own processes only)
- `~/raptor2/label_gold58.py` (seconds): table in sec 3.
- `~/raptor2/prep_bench.py`: 8 mirror studies, 1.4 s/study build, 14.2 / 8.3 / 10.8 MB raw / PNG-3 / npz-compressed.
- `~/raptor2/build_oai_labels_raptor.py`: OAI counts in sec 3; output `~/oai_build/oai_labels_raptorT1.csv` (box only, mode 600).
- `~/raptor2/ck_inspect.py`: N dict keys, arch, res 384, epochs [14, 11, 12], gold_auc .918952, 50 BN tensors.
- `~/raptor2/flops.py`: CoAtNet-2 rmlp 42.84 GMACs @384, 29.75 @320; ConvNeXt-T 4.45 @224 (FlopCounterMode, conv + matmul).
- `~/raptor2/gold_parity.py` (PID 466778, 16:13-16:46, 32 min): sec 2.2 (N gold58 full384_k24 .919111 vs stored .918952).
- `~/raptor2/approx_parity.py` (PID 482426, 16:46-17:23, 37 min): sec 2.3 (exactB .918375, approxB .920458).
- All own processes ended (checked 17:24); nothing left running. Training queue throughput unchanged during the runs
  (TTR224 allv 0.37-0.39 s/it).
- Laptop copies of every script + the two logs: `session_2026-10-13/raptor/box/`; nartaa README + manifest:
  `session_2026-10-13/raptor/`. Box outputs: `~/raptor2/` (scripts, gold58 prediction .npy of N, RSNA only) and
  `~/oai_build/oai_labels_raptorT1.csv` (OAI, box only).
