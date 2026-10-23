# Lens: other model types (10-19 ~17:30 IST). RESEARCH ONLY.

Nothing was built, pushed, committed, submitted or created. No GPU or box job ran. Box access was read-only `find` / `ls` /
`scp` of four of our own prediction files FROM the box (Dread-arm train-set predictions and our crop / fleet gold
predictions, competition studies only, no OAI data). No public code or data was downloaded: dataset METADATA only
(`kaggle datasets files` / `datasets metadata` text, into `think/pulled2/`) and anonymous page-model reads.
Tags: MEASURED (we read or computed it this session), CLAIMED (a third party says so), MODEL (our arithmetic on
stated assumptions), GUESS (judgement). Source after each claim.

## TL;DR

1. **No public model type is worth a 10-20 build.** Every public model outside c4's three families
   (nartaa CoAtNet MIL r384 / r224, heliosli compact ConvNeXt-S R6, nartaa S6fall = heliosli S6 arch on JEV labels) is
   either weak as a single model (public solo .891-.935) or a compact-family sibling we already read negative on the LB
   (R7, heliosli S6 compact-v1, our S6 seeds). MODEL expected gain of the best candidate at w .05-.10: about 0 to +.0003,
   below one display tick. P(LB positive) ~.35 (GUESS).
2. **The most diverse decent family we can measure is Dread Development's competition-only Raptor CoAtNet**
   (maxspan, public solo .928, CLAIMED dataset page). On gold58 its within-class rank correlation with the c4 output is
   .79 vs .87-.91 for c4's own legs (MEASURED, `models_work/noisecorr_g58.json`), and c4 + maxspan at .10 / .20 reads
   +.0004 / +.0011 (P .63 / .71). That is inside the gold58 noise floor: S6fall .3 reads +.0012 (P .85) on the same
   harness and was LB-negative (MEASURED, `models_work/calib_g58.py` + LB history).
3. **LB history says weak-but-diverse members hurt here, and only strong members helped (MEASURED, our reads).**
   Positives: S6fall .2 (solo .950), A224 .25 (solo .945). Negatives: our RadImageNet crop R50 leg at .15 / .25
   (.942 / .941 vs .943 base, 10-04), K14 ConvNeXt-T fleet, KC1 (gold74 +.0068, LB negative), R7, s43 / s44,
   R7-teacher student. Third party: P (.943 stack incl. Dread arms) at .40 on N = .949 = N alone (CLAIMED goodpjw).
   The binormal blend model predicts +.0004 to +.003 for these (MODEL, below), so it is uncalibrated optimism here.
   **Working rule: a new leg needs public solo >= ~.945 AND a different label / data source.** No public candidate
   outside c4 meets it.
4. **New since 10-17 (MEASURED, metadata):** B-leg R6 fine-tunes (3 accounts, R6 warm start on gold-calibrated soft
   labels, gold58 in the pixel cache), neronrx "Mycelis 4-Fold ConvNeXt 384px" (CC0, 971 downloads in ~1 day, OOF CSVs
   inside, no score anywhere), cyruspinto engines 8-11 (ConvNeXt / Swin-B 384, 9-10 downloads; author team .958 r604),
   dkim54 v11 (CC BY-NC, 10 x 100 MB folds, no score), riadmohamed42 DINO 5-fold (team .958 r652), copies of S6fall
   (yeshawang/model412) and R6 v1 (ayodeji). None has a public score; none is a reason to build.
5. **LB top notebooks: nothing new at >= .958** (MEASURED: anonymous page reads 10-19 ~11:35 UTC; watch run 10:56 UTC).
   Max public notebook score .958, the same 5 notebooks as 10-17, no new scored version since 10-17.
6. **Authors of public weights rank below us, with two exceptions** (MEASURED, LB API 10-19 11:00 UTC, display-band
   order = unrounded order): BliverRigP?Doge .959 r179 (their KC models are private) and Heliosli .958 r192. Below us
   (r199): renta.k r396, goodpjw2008 r461, CYRUS PINTO r604, RIAD MOHAMED r652, Dread Development r856, zentropy r886,
   Kirderf r992, hengck23 r1267, Pilkwang Kim r1330. So nobody is getting a better public score out of the other model
   types we could copy.
7. **Runtime constraint for any new leg (MODEL from timing.md):** the S6F start guard is 5.75 h and c4's base ends about
   4.0-4.25 h in. A new stage placed BEFORE S6F that takes > ~1.5 h can make S6F skip (canary fires, column lost). A new
   leg must run AFTER S6F (ends ~6.0-6.25 h) with its own skip guard and a c4 fallback, in the ~2 h window to ~8.2 h.

## 1. What c4 already holds (the diversity baseline)

| c4 leg | family | train data / labels | public solo | gold58 AUC | within-class rho vs c4 output |
|---|---|---|---|---|---|
| r384 | nartaa CoAtNet-2 MIL, 384, anatomical mirror TTA | OAI 2,399 knees + competition, report labels | .949 (CLAIMED page) | .9228 | .914 |
| r224 | same, 224 crop, one view | same | .945 (CLAIMED) | .9150 | .878 |
| R6 | heliosli compact ConvNeXt-S, 4 folds, flip TTA forced | competition, ImageNet init, report-derived targets | n/a | .9366 | .913 |
| S6fall | heliosli S6 arch retrained by nartaa, 1 all-data model | competition 4,263 studies, JEV-blend soft labels, gold58 excluded | .950 (CLAIMED nartaa page) | .9314 | .869 |

All MEASURED from `ours_work/ours_g58.json` and `models_work/noisecorr_g58.json` unless tagged. Pairwise per-label
Spearman (MEASURED, ours_g58.json): R6-r384 .872, R6-S6fall .879, r384-S6fall .880, r384-r224 .944, R6-R7oof .940.
So R6, Raptor and S6fall are already three roughly equally separated families; a 4th family has to beat that.

## 2. Gold58 diversity screen of every model we hold predictions for (MEASURED)

Script `models_work/div_g58.py` (numpy, CPU, 30 s). c4 recipe re-implemented exactly as in ours_g58.py; candidate X
added at the end, `(1-w) pct(c4) + w pct(X)`; paired bootstrap 2,000 resamples. gold58 is NOT for picking weights
(A19 lesson); this only ranks FAMILIES by diversity.

| member | source | gold58 AUC | rho c4 (overall) | within-class rho | d @ .10 (P>0) | d @ .20 (P>0) |
|---|---|---|---|---|---|---|
| Dread maxspan v5 (CoAtNet 384, competition-only) | box `~/jump2/labels_in/T_maxspan_v5.csv` (10-04 teacher run) | .9221 | .890 | .792 | +.00036 (.63) | +.00114 (.71) |
| Dread native384 v8 | `T_native384_v8.csv` | .9165 | .876 | .777 | -.00020 (.41) | -.00101 (.32) |
| Dread finespacing v9 | `T_finespacing_v9.csv` | .9137 | .878 | .779 | +.00063 (.69) | -.00066 (.39) |
| Dread parent proxy (v5 .75 + v8 .25) | `T_raptor_proxy.csv` | .9246 | .894 | .801 | +.00073 (.76) | +.00073 (.63) |
| our OAI-free ConvNeXt fleet mean (25 members) | box `~/r7t_labels/fleet/gold_fleetOF_mean.csv` | .9217 | .924 | .843 | -.00019 (.39) | -.00131 (.21) |
| our RadImageNet R50 crop f1 / cropaug f1 / mixSA | box `proxy_crop_gold74_preds.npz` | .873 / .890 / .881 | .81-.83 | .69-.71 | -.0002 to -.0006 | -.0028 to -.0050 |
| our S6 seed s43 | g58/s43_gold58.csv | .9210 | .930 | .863 | -.00054 (.21) | -.00087 (.26) |
| heliosli R7 OOF | g58/R7OOF.csv | .9193 | .917 | .830 | -.00044 (.31) | -.00075 (.33) |

Calibration on the same harness (MEASURED, `models_work/calib_g58.py`): S6fall .2 vs none +.0005 (LB positive);
S6fall .3 vs .2 +.0012, P .85 (LB NEGATIVE, hw3); A224 .25 vs none +.0004 (LB positive); A224 .15 vs .25 0.0000
(LB negative). Earlier: KC1 gold74 +.0068 / gold58 +.0072 over R7, LB negative (PROGRESS_16). **Noise floor ~ +-.001;
the Dread-arm deltas sit inside it.** Caveat: Dread picked checkpoints on gold58 (BIG_JUMP_PLAN 10-04, CLAIMED in our
notes), so their gold58 AUC is somewhat inflated.

Binormal MODEL (`models_work/noisecorr_g58.py`, c4 public .9588, macro treated as one label, score averaging):
adding a .928 model at w .10 gives +.0004 at rho .8, +.0018 at rho .7; a .935 model +.0008 / +.0022; a .91 model
-.0007 / +.0007. The same model predicts positive gains for the crop leg, K14 and KC1, all of which read negative.
**Treat these as upper bounds; the LB-calibrated expectation for any public sub-.94 leg is ~0 (MODEL + MEASURED history).**

## 3. Inventory: every public model / weights source relevant to this comp

Solo = public LB of the model alone. "Author team" = MEASURED LB API 10-19 11:00 UTC (`work/api_lb_1019T_think.json`).
Notebook counts = public notebooks attaching it (MEASURED, `pulled/watch/watch_state.json`, 576 notebooks known).

### 3a. Already in c4 (for completeness)
| source | content | licence | note |
|---|---|---|---|
| nartaa/rsna-knee-publication-swa-weights-20261007 | CoAtNet r384 + r224 SWA | other (research incl. this comp) | 217 notebooks; OAI-trained |
| R6 bundle fe055406 (our copy eeshsaxena/rsna-knee-r6-fe055406-v1; public twins heliosli ...-w1, rostigor, ayodejiibrahimlateef 10-18) | compact ConvNeXt-S x4 + Effusion / Baker's stackers | other | v1 fingerprint = ours (notebooks.md 10-17) |
| nartaa/rsna-knee-s6fall-weights v5 | S6 ConvNeXt-S all-data, E_prior_s42_fall.pt | other | trained WITHOUT mirror aug: run_config `aug crop`, no `mirror` key (MEASURED `session_2026-10-15/silver/pulled/s6fall_meta`); R7 had mirror .5 |
| heliosli/rsna-orthofoundation-fold0-epoch7-20261004 | code slug: SKM-TEA anatomy V-Net + compact_infer code | n/a | not a prediction leg |

### 3b. Candidates, ranked by expected blend value on c4 (best first)

| # | source | architecture / input | train data / labels | solo public | different from c4? | runtime (hidden, ~1,300 studies) | copyable now | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | dreaddevelopment/raptor-knee-maxspan (1,904 dl) | CoAtNet-rmlp-2 384 + per-finding attention MIL, 64 slices over 2-98 % span, 140 mm crop, 336 px, 62 windows (CLAIMED page) | competition only, all 4,349 studies, report labels (CLAIMED page) | .928 (CLAIMED page) | yes: no OAI, different corpus; within-class rho vs c4 .79 (MEASURED) | GUESS 0.5-1.5 h incl. its own decode / resample (Raptor stage ~25 min incl. prep, RAPTOR_PORT_PLAN GUESS; Dread's "~9 h for the full test set" for a 94-window pipeline is a red flag, CLAIMED 737696) | yes, CC0, public inference notebook knee-mri-twelve-findings-from-a-single-model | best of a weak set: EV 0 to +.0003, P ~.35. Not recommended for a slot |
| 2 | hengck23/hengck23-rnsa-knee (1,360 dl) | MaViT (linear-attention ViT) 288, 6 views x 8 slices, 1 ckpt 79.6 MB (MEASURED file list) | **unstated** (no description; hengck23 named KJD-25K = KneeCoT, host-banned, in 745861: RESEARCH_TO_TOP_10-08) | .935 (MEASURED page, v4) | yes: architecture and input layout; not measured on our side | ~33 min on 1 T4 (title "1.5 s/study", CLAIMED) | MIT, but provenance = rule risk | EV 0 to +.0004 (MODEL upper), P ~.35; **do not put in a final pick without provenance**; author team .958 r1267 |
| 3 | renta0426/rsna-knee-public0033-meniscus-bag-v1 (2,586 dl) | DINOv2-Base, last 6 blocks + LN tuned, 6 series slots, 336 px, 10 overlapping 2.5D windows (CLAIMED page) | competition 4,349 studies, report weak labels, gold58 excluded (CLAIMED page) | no standalone read; .945 inside a parent stack (MEASURED watch) | yes (ViT, different slots) | GUESS 20-40 min | Apache-2.0 | EV ~0; author team renta.k .958 r396 |
| 4 | neronrx/mycelis-rsna-knee-4fold-hive (NEW 10-18, 971 dl) | "4-Fold ConvNeXt 384px": 4 x 209.6 MB fold .pt (ConvNeXt-S size class, same as R6 folds), knee.py 8 KB, log + **oof_fold0-3.csv** (MEASURED file list) | unknown | none anywhere (not in any of 527 listed notebooks) | unknown; ConvNeXt 384 suggests R6-like | GUESS 0.7-1.5 h (4 folds at 384) | CC0 | unknown prior ~0; the OOF CSVs (4 x ~181 KB) would let us measure OOF AUC and rho vs R7 OOF offline, **needs Eesh's OK to download** |
| 5 | goodpjw2008/rsna-knee-2-5d-convnext-reader (9,196 dl) | ConvNeXt-T 2.5D, 6 slots, study transformer, 3 of 5 folds (CLAIMED page) | competition only, 4 public label tables, gold58 not used (CLAIMED) | .929 (CLAIMED A_public_assets) | same family as our K14 / HF fleets (fleet within-class rho .84, LB negative) | ~20-50 min | Apache-2.0 | closed: at 1/10 in goodpjw .958 and heliosli 0-9586, both authors below us (MEASURED notebooks.md) |
| 6 | other Dread arms: widedense (.924), arms-x (.917), arms = ConvNeXtV2-B 336 + CoAtNet 384 + EffNetV2-L 480 rank-mean (.915 vs .914 CoAtNet alone), native384 / native384dense / finespacing / widefov / fullspan / cnn336 | CoAtNet family + 2 other backbones | competition only | .914-.928 (CLAIMED pages) | redundant with #1 (native384 / finespacing rho .78 vs c4, gold58 deltas negative or noise) | each like #1 | CC0 | subsumed by #1; Dread's own page says 3 backbones bought ~.001 |
| 7 | heliosli S6 compact-v1 (bundle 209a32c8), heliosli R7, jessky/rsna-knee-local-cv-cnx50 (S6 d50, 5 folds, Apache-2.0, gold58 excluded), B-leg R6 fine-tunes (kritikabenjwal s0 f01, sarthakgupta1801 s0 f23, xhaitanya s1 f01; NEW 10-18) | compact ConvNeXt-S | competition; B-leg = R6 warm start, 2 epochs on gold-calibrated soft-v2, pixel cache includes all 58 gold studies (CLAIMED page) | S6 era .954 with Raptor; R7 + Raptor .956; B-leg single-fold gold .911 / .927 (CLAIMED) | NO: R6 family (R6-R7oof rho .94) | +0.5-1.3 h (R7 late leg MEASURED +~1 h) | yes | closed: R6+R7 avg, R7 late leg negative (MEASURED); S6 .3 next to R6 .957 vs .958 (CLAIMED goodpjw); B-leg gold numbers in-sample-ish |
| 8 | bliverrigpdoge/rsna-knee-leg-r9all (1,146 dl) | Raptor CoAtNet student, 293 MB | competition (GUESS) | none solo; yuvrajlamba01 research-optimized = R7 + Raptor + r9all .956 = base of that time (MEASURED page) | Raptor family | ~Raptor arm | CC0 | closed (no gain measured by a third party) |
| 9 | mattiaangeli family: knee-mri-fold-weights (206 nb), coat-resgated-ep10-top3, coatnet-d4-depthzone-swa3-b2, coatnet-global96-top3, radimagenet-foldsv1-heads, cnx-m448-f0 | CoAtNet / ConvNeXt / RadImageNet heads | competition only (CLAIMED 750356) | M448 .894 solo; parent stack .941-.943 | partly; heads in-sample on gold58 (MEASURED 10-04) | 1-7 h for the full parent stack | CC0 / CC BY-NC-SA / other | closed: M448 hurt at 40-50 % (MEASURED 10-04); P stack .40 on N = N (CLAIMED) |
| 10 | antoinegg1 e9 RadImageNet heads v15 / e11 diverse heads v20; marwanmath ResNet-50 RadImageNet; prvsiyan v52 heads | RadImageNet R50 heads on frozen features | competition; e9 training unstated (CLAIMED 750357) | .949-.952 only inside N stacks | RadImageNet family = our crop R50 (within-class rho .69-.71, LB negative at .15 / .25) | small | CC BY-NC-SA | closed |
| 11 | pilkwang/rsna-knee-weights (20 ckpts DINOv2-S, 4 seeds x 5 folds) | DINOv2-S slice windows | competition only (CLAIMED 750355) | .891 (MEASURED page) | yes, but far too weak | moderate | CC0 | closed; author team .958 r1330 |
| 12 | NEW 10-16..18, no score anywhere: cyruspinto engine8 (ConvNeXt 24-ch multi-plane 111 MB), engine9 (ConvNeXt 384, 185 MB), engine10 (Swin-B-384, 323 MB), engine11 (ConvNeXt-B 384, 352 MB), all CC0, 9-10 dl; dkim54/rsna-knee-models-v11a..i (10 x 100 MB, CC BY-NC 4.0, 491 dl, own code + timm 1.0.30); riadmohamed42/rsna-knee-dino-5fold-weights (5 x 91.5 MB, CC0); abhiiiish 3m (ConvNeXt-T + 2 ResNet34, licence unknown); arumpuri weigh-knee-v4 / v5-tiny | various | unstated | none | unknown | unknown | metadata only (MEASURED) | no evidence; authors' own teams below us where matched (CYRUS PINTO .958 r604, "Cyrus" .955 r1928, RIAD MOHAMED .958 r652) |
| 13 | tonylica rsna2026-models / dinov3 repro assets; prvsiyan orthodiffusion-official-knee-mri-foundation; royceplayz MedicalNet R50 (3D); Kaggle Models DINOv2 S/B/L, DINOv3-L, MedSigLIP-900M-448, BiomedCLIP, RadioDINO | foundation backbones / 3D R50 | n/a | notebooks .932-.943 (MEASURED watch) | backbones, not knee models | n/a | mixed | would need training: impossible before the 10-20 freeze |
| 14 | isotriage/rsna-jev-oai-arm (13 dl, licence other) | unstated, OAI | OAI | quad-b2 .954 (with R6 / R7 / Dread arms; below the .956 base of the day, MEASURED) | ? | ? | rule exposure | closed |
| 15 | nartaa/rsna-knee-oai-original384-s42-pair, nartaa/rsna-knee-s6f0-infer, kiyoshiohno local-logic, goodpjw rsna-knee-core / assets, bliverrigpdoge KC | various | various | .957-.959 entries | ? | ? | **not public** (MEASURED watch: not_public / 404) | not copyable |

Copies (no new information, MEASURED by byte size): yeshawang/model412 = nartaa S6fall (E_prior_s42_fall.pt 200,185,003 B,
10-18); ayodejiibrahimlateef r6-frozen-v1-public-20261018 = R6 v1 (823,098,827 B); rostigor / heliosli w1 / goodjane /
varianceofx / mizeroluckygall = R6 or R7 copies.

## 4. What changed since 10-17 (MEASURED)

- Datasets (kaggle datasets list -s knee / "rsna knee" / "knee mri" / coatnet, sorted by updated; `pulled2/ds_raw.csv`):
  the new model releases are in row 12 above plus B-leg (row 7) and Mycelis (row 4). Others are caches, embeddings,
  label tables (omerzalman "99 % accuracy" soft labels, kritikabenjwal soft-v2, abdullahmathar argon4, lingyu07 v53,
  siamrahman29 W5, srinath2495 dual-teacher consensus) or code (haungjiawei train-v2 code+labels 10-19, dkim54 blend-code).
  Labels are useless to us after the training freeze.
- Notebooks run since 10-17 (`pulled2/kl_dateRun_p1.csv`): basith007 model-ensembling (unscored, CoAtNet features),
  sergiofigal04 knee-detection (MedSigLIP + DINOv2-S, unscored), govind123445 knee-12, thegoku v9, shardulsalodkar
  v2-v4, notfu4y deterministic-report .695, toghrulaliyev aze (Qwen2.5-7B + DINOv2-L, unscored), ryanholbrook
  efficiency LB, goodpjw does-s6-help v8 (unscored), lavinwins / heliosli re-runs (already in notebooks.md). New
  notebooks are locked since 10-15 23:59 UTC; nothing new scores >= .956.
- Forum (FORUM_10-19.md + think/forum_raw): Mattia Angeli says his six releases are competition-data only (750356);
  Pilkwang says the same (750355); M. Ehtasham (rank 294) "zoomed in method", no detail (750529). No model release
  from any team at >= .959.

## 5. LB top notebooks (>= .958), MEASURED 10-19 ~11:35 UTC (`models_work/kvm_top_1019.json`)

goodpjw2008 coatnet-r6-2-5d-mil v27 .958; heliosli blend-mine-in v1 .958; heliosli 0-9586 best v7 / current v6 .958;
lavinwins gold-medal best v5 .958 (current v6 .943); lavinwins top-1-gold best v4 (current v7 unscored);
kiyoshiohno template now 404. **No public notebook above .958 and no new ingredient** (all dissected in
`session_2026-10-17/research_1hr/notebooks.md`). Band counts on the LB: .960+ = 121 teams, .959 = 62 (r122-183),
.958 = 1,218 (r184-1401); we are r199 (MEASURED api_lb_1019T_think.json).

## 6. Using the unused ~2.5 h of hidden runtime with another model: how it would have to look

- Placement: AFTER the S6F stage, not before (MODEL: base ends ~4.0-4.25 h, S6F guard 5.75 h, S6F ~2.0 h; a 1-1.5 h
  stage in front pushes S6F to the guard). Own start guard (e.g. start only if kernel age <= 6.5 h), hard deadline
  ~8.2 h, process-group kill, and fall back to the c4 submission.csv byte copy if skipped. A skipped leg is invisible
  on the LB unless a canary is wired, so a canary is needed to read it.
- Blend: one global end weight (.05 or .10), never per-label (nested split-half: free per-label picks lose -.0007,
  MEASURED 10-18).
- Cost: new kernel build + commit (~6.5 h incl. queue risk) + score (~6.5 h) = push 10-20, read 10-21; one slot.
- Same-model alternative seen while reading the kernel (outside this lens, for the runtime lens): S6fall runs WITHOUT
  flip TTA (MEASURED: the decoded S6F driver has no `flip` token; R6 has it forced). But S6fall was trained without
  mirror augmentation (MEASURED run_config), which makes it the r224-mirror case (trained without flips, read
  negative 10-15), not the R7-flip case (mirror .5 in training, +214 ranks). EV ~0 (MODEL), P ~.45 (GUESS). Also
  unverified: whether the OF compact_infer (sha 34f433fe) has `--flip-tta`.

## 7. How a GPU could help this lens (and why it barely does)

- MaViT ships `train_golden58.csv` and a gold58 eval path; Dread's notebook runs any arm; a T4 could score both on
  gold58 in minutes. But gold58 cannot resolve +-.001 (section 2 calibration, KC1 +.0068 failed to transfer), so the
  measurement would not change the decision. Value ~0.
- The only cheap measurement that could change a decision is the Mycelis OOF (CPU only, 4 small CSVs): if its OOF
  macro vs LLM labels were >= R7 OOF (.95x-class) AND its OOF rank correlation with R7 OOF were < ~.85, it would be the
  one public leg that passes the "strong + different" rule. Needs Eesh's OK to download (untrusted public files, read
  as data only). Prior that it passes: ~.1 (GUESS: 4-fold ConvNeXt 384 is the R6 recipe class).

## 8. If / then

- IF Eesh wants one model-type experiment anyway, THEN the order is: (a) Mycelis OOF check on CPU (needs download OK);
  (b) only if it passes, a Mycelis or Dread-maxspan leg at w .05 placed after S6F, built 10-20, read 10-21 by R1 vs c4;
  never MaViT in a final pick before hengck23's training data is known. ELSE spend the slots on other lenses.
- IF any public notebook shows >= .959 (watch), THEN run the 10-17 bait / pin check before reading anything into it.
- Final picks are unaffected by this lens: A = c4 57343435, B = R6 57248084.

## 9. Files (all under session_2026-10-19/think/)

- `models_work/div_g58.py` + `div_g58.json` (gold58 diversity screen, 14 members);
  `models_work/noisecorr_g58.py` + `noisecorr_g58.json` (within-class rho + binormal MODEL);
  `models_work/calib_g58.py` (LB-known moves on the same harness); `models_work/kvmtop.py` + `kvm_top_1019.json`;
  `models_work/dsbasics.py`, `ds_basics_new.json`.
- Copied from the box (ours, competition studies only): `models_work/T_{maxspan_v5,native384_v8,finespacing_v9,raptor_proxy}.csv`,
  `gold_fleetOF_mean.csv`, `proxy_crop_gold74_preds.npz`.
- Kaggle listings + metadata (untrusted, text only): `pulled2/kl_*.csv`, `pulled2/ds_raw.csv`, `pulled2/ds_files.txt`,
  `pulled2/meta_*/dataset-metadata.json`.
