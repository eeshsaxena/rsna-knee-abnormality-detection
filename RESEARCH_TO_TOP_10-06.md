# RSNA Knee: research plan from .946 toward the top (2026-10-06 ~22:30 IST)

Read-only research. Forum and data pages read signed out in my own built-in-browser tab; one read-only
`kaggle kernels list` call; one web search. Nothing posted, voted, accepted, downloaded or submitted.
Builds on (does not repeat) `session_2026-10-05/jump2/topsolutions.md`, `JUMP_PLAN_10-05.md` and
`PROGRESS_06-10-2026.md`. New since those files is marked **NEW**.

Tags: **MEASURED** = our own LB/log/file, or a page fact read today. **CLAIMED** = a forum self-report.
**GUESS** = my judgement. Forum ranks are the "Nth in this competition" badge on 10-06.
Thread links: `https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/<id>`.

---

## 0. Bottom line

1. **The gap to the top is a single-family gap, not a blend gap.** At least four teams claim a single family
   (5-fold) at .949-.955: Tanuki_boosting .955 5-fold single model (735304, posted ~3 h ago, **NEW**),
   CoolinLai ResNet-50 224 5-fold .954 (735304), Tucker Arrants "most of my models in the low 0.95 range"
   with a simple Qwen extraction (745214, **NEW** wording), Scott Willis small ResNet .949. Our best family
   (5-fold ConvNeXt-T TTR + TTA) reads .943 SOLO [MEASURED]. Blending a .943 family into a .943 parent gave
   .946; the same arithmetic says a .950 family at r ~.90 would give ~.951-.953.
2. **Label quality is probably not what separates us.** Tucker (16th): labels from "a simple extraction
   process with Qwen, no frontier models", and he is "doubtful that improving labels matters much"
   (745214, CLAIMED). Cody_Null: teams take EfficientNet-B0 to .94+ with ~.88 gold-AUC labels (743148,
   CLAIMED). Sanjib measured that swapping LLMs is worth ~0 on gold58 and that 22 of 33 extraction misses
   are reports that never name the finding (743148, CLAIMED with numbers). Our own text re-extraction work
   agrees (closed, MEASURED). Exception: findings the report is silent on (Effusion, Synovitis, Fracture,
   Contusion), where only image evidence helps; see lever 2.
3. **The levers that remain, ranked by expected public gain per V100 hour** (detail in sec 1):

   | # | lever | expected public gain | cost | confidence |
   |---|---|---|---|---|
   | 1 | Merge with an own-model team (Eesh posts) | +.002 to +.007 | 0 GPU, Eesh's time, by 10-15 | medium |
   | 2 | Image-aware relabel of the "silent" labels (report + key slices to a VLM), then retrain our fleet on it | +.001 to +.004 | API ~$15-80 (or ~10 V100 h local, weaker), 1-2 days code, ~20-25 GPU h retrain | low-medium |
   | 3 | New simple-pooling CNN family (ResNet-50 / EffNet-B0 at 224) on mix50 | +.001 to +.004 | ~4 h probe + ~16-20 h for 4 folds | low-medium |
   | 4 | Redirect the teammate's 28 GPU h away from a same-recipe seed 2 | turns ~+.0005 into lever 2 or 3's gain | 0 extra | high (that the seed is low value) |
   | 5 | TTR at 288 px as a replacement set (running) | +.0005 to +.002 | ~30-35 GPU h for 4 more folds | medium |
   | 6 | Final-pick hedge against shake-up | protects ~.002-.004 of private | 0 | medium |

   Realistic target in 16 days: **.948-.951 public** with a merge, **.947-.949** without (GUESS). Gold (~.958-.959
   today, rank ~20) needs a .95+ single family of our own or a partner who has one.

---

## 1. Ranked levers

### Lever 1: merge with a team whose models are their own (Eesh's action; deadline 10-15 23:59 UTC)

- **Why:** a merged team displays its best member at once, and two independently built families decorrelate
  far more than any second seed. Chris Deotte and Dread Development both state that models that disagree make
  better ensembles (742926, CLAIMED).
- **New leads posted in the last 4 days (all CLAIMED, read 10-06):**

  | who | score / assets | why a fit | thread |
  |---|---|---|---|
  | Christoffer Thimsen (+1) | .940; own structured labels for all 4,407 studies, attention-MIL CoAtNet/ConvNeXt, a per-series family, fp16 kernel; **compute-bound** | we have a V100; their models are not public-notebook forks | 746060 (posted ~19 h ago) |
  | Nikita Ageev (truenikita) | own CoAtNet-2 384 at .937 with no public competition weights, own Claude labels (Haiku + Sonnet on ~1,300 ambiguous), better-calibrated labels, RTX 4090 | most "independent" single model on offer; he only has 4 submissions | 743374 |
  | KalyanG17 | .942; own CoAtNet@384 .935, ConvNeXt-T .930, RadImageNet R50 .924, two with 5-fold OOF, RTX 5070 | own labels; wants "built differently" partner | 745091 |
  | Pand | .940 own models with full OOF, RTX 4060 | OOF allows offline weight setting | 745091 |
  | Tanuki_boosting (58th) | .955 single model 5-fold | wants partners with .950+ single models; we do not qualify today | 735304 |

- **Expected gain:** our .946 plus a .937-.942 family at r ~.80-.88 with our blend: +.002 to +.007 (GUESS from the
  binormal table in `jump2/topsolutions.md` sec 4, which over-predicted once; discount toward the low end).
- **Cost/risk:** 0 GPU. Combined submission count must fit the merge cap (see `JUMP_PLAN_10-05.md` sec 5).
  Never share files or predictions before the merge (rules). Risk: no reply, or partner's models turn out to
  be public-weight forks.
- **First step and gate:** Eesh messages Christoffer and Nikita first (both independent and compute-poor, so
  our V100 is an asset). Gate: partner states a solo number for a model with no public competition weights
  >= .935; merge agreed by 10-13.

### Lever 2: image-aware relabel of the labels reports are silent on, then retrain

**Evidence (all CLAIMED, numbers scored on the 58 expert studies):**
- colum2131 (99th), 745949, **NEW**: Effusion label AUC .835 (reports only) -> ~.91-.92 (report + MRI to an LLM);
  Synovitis .80 -> ~.84-.86. An image model trained on those pseudo-labels: Effusion .986, Synovitis .859. He
  trains on soft probabilities and ranks "explicit > equivocal > silent > absent".
- Same post: experts call fluid positive far more often than reports: 10 of 26 "effusion absent/normal" reports
  and 14 of 18 "equivocal/small" reports were expert-positive; 11 of 36 report-silent synovitis cases were
  expert-positive.
- gchauhan EDA (741310): fracture appears in 7% of reports but 31% of expert labels; reports are silent on about
  half of expert-positive synovitis.
- Will (740610): report-target AUC vs experts: Effusion .696, Synovitis .676, Fracture .814, Contusion .821, PF OA
  .819 (his extractor); his image model already beats the report target on Effusion (.901) and Contusion (.915).
- Nikita (743374): typical text-label errors are degenerative edema read as contusion, "small" effusion
  under-called, impaction read as fracture; about 6 of 39 disputed gold cells contradict the report text
  (image-only findings).
- Counter-evidence: Sanjib (743148): "No reader recovers text that was never written, and neither will a VLM"
  (his argument is about text readers; it does not cover a VLM that looks at pixels). Ali Ozan (745861):
  Claude labels -> .942 single model; MedGemma probing "didn't help much". OmerZalman: 800 studies relabeled,
  "only around 80-90% correct".

**Which labels benefit:** Effusion (large, high-contrast fluid; strongest evidence), Baker's (fluid, location
fixed), Synovitis (weaker gain; synovium vs fluid is hard without contrast), Fracture and Contusion (report
silence is common, but VLM reading of marrow edema vs fracture line is hard, GUESS: small gain). Leave ACL, MCL,
menisci and the three OA labels on text labels: reports name them well (Sanjib: misses are on silent reports).

**How to choose key slices (we already have the machinery):**
- Option A, model-guided (recommended): our TTR models have per-target slice attention and per-target view
  attention. For each study and each target label, take the top-2 attended slices in the top-2 attended views
  from one fold model (OOF for that study). That is the "ROI" hengck23 suggests (745861), at zero extra cost.
- Option B, fixed anatomy rule (fallback): Effusion = sagittal fluid-sensitive at 45/50/55% of the stack plus
  axial FS at the patella level; Baker's = axial FS lower third (posterior); Synovitis = axial FS (suprapatellar
  recess, Hoffa fat pad) plus sagittal FS mid; Contusion/Fracture = coronal FS 40/50/60% plus sagittal FS
  30/50/70%. Window each slice at 1-99 percentile, 448-512 px, label each image with plane and sequence.
- Keep 8-12 images per study. Ask for a probability per label with the 4-level evidence scale, plus the
  report text, plus the official label definitions from the host overview (733343: "on the fence" is graded
  negative; effusion means moderate or large).

**Model options and cost:**

| option | quality (GUESS) | throughput | cost |
|---|---|---|---|
| Claude Sonnet 5.5 via Message Batches (50% off) | best of these for medical image reading among API models we can use; colum's ~.92 Effusion was a frontier-class LLM | 4,400 studies in hours | ~10-12 images x ~350 image tokens (512 px; ~w*h/750) + ~1.5k text = ~6k input tokens/study -> ~26M input + ~4-6M output tokens. At $2/$10 per MTok halved by batch: **~$45-60** for all studies |
| Claude Haiku 4.5 batch | weaker | same | **~$20-30** |
| Active labelling (hengck23): only ~1,500 studies where the report is silent/equivocal on Effusion/Synovitis/Fracture/Contusion, or where v4 and our OOF disagree | most of the gain | 1/3 | **~$15-20** (Sonnet batch) |
| Local open VLM on the V100 | low on fine findings; possibly OK for Effusion/Baker's | ~6-10 s/study -> ~8-12 h for 4,400 (GUESS) | 0 $, but ~10 GPU h taken from training |

Prices: Claude API pricing table cached 2026-09-25 (Sonnet 5.5 $2/$10, Haiku 4.5 $1/$5, Opus 5.5 $4/$20 per MTok;
batch 50% off). Local VLM notes: the V100 is compute capability 7.0 with no bf16; Qwen2.5-VL-7B in fp16 (~16.6 GB)
does not fit 16 GB; vLLM on V100 falls back to the legacy GPTQ kernel, reported buggy for 4-bit
([vLLM forum](https://discuss.vllm.ai/t/speeding-up-vllm-inference-for-qwen2-5-vl/615),
[community V100 fork](https://github.com/humanjesse/vllm-v100/blob/main/README.md),
[compatibility matrix](https://www.speediyo.com/ai-infra/vllm-gpu-compute-capability-matrix)). Feasible local
choices: a 3-4B VLM (Qwen2.5-VL-3B, Qwen3-VL-4B) in fp16 with plain transformers, or the 7B in 8-bit
bitsandbytes. Both are GUESS-level for accuracy on knee MRI; Ali Ozan found MedGemma unhelpful.

**Rules:** host post 733965 (MEASURED read): submitting Competition Data "including report text" to an external
LLM/API is not prohibited private sharing; the service must be reasonably accessible and of "minimal cost". Host
reply: LLM API use to generate labels is allowed. Images are Competition Data like the reports, so the same
ruling covers them (GUESS, consistent with the wording). Tens of dollars fits "minimal cost" (Raymond spent
< $5 on all his labels, 735304). Our house rule so far was "no API keys": **this lever needs Eesh to decide
to get one.**

**Expected gain:** Effusion +.02 to +.04 and Synovitis/Fracture/Contusion +.005 to +.02 each at the LB label
level gives macro +.002 to +.006 solo for the retrained fleet (macro = mean of 12). New labels also make our
fleet less parent-like on exactly those columns. Blended: **+.001 to +.004** (GUESS, low-medium confidence).

**First experiment and gate (1 day, ~$3):**
1. Pick the 74 gold studies (58 public gold + our 16 extra held-out gold if present in gold74) and build the key
   slice pack (option A, using each study's OOF fold model).
2. Run Sonnet 5.5 (non-batch, ~74 calls, ~$1-3) with and without images, same prompt.
3. Gate: images must lift gold AUC on Effusion by >= +.04 AND not drop the macro over the 4 targeted labels.
   Pre-register the prompt before looking at gold errors (our KX lesson: prompts written after reading gold
   errors are not clean evidence). Then run the batch on the ~1,500 uncertain studies, mix the new soft
   probabilities 50/50 with mix50 on the 4 labels only, and train 1 TTR fold. Gate 2: fold-0 OOF vs the new
   labels is not the test; use gold58 per-label on the 4 labels (>= +.02 mean) and a SOLO LB read >= .939.

### Lever 3: a simple-pooling CNN family (ResNet-50 or EfficientNet-B0 at 224) on mix50 labels

- **Evidence (CLAIMED):** CoolinLai R50 224 5-fold .954 ("just resnet 50", 735304); Tucker: start with
  ResNet34/EffNet-B0 at 224/288, "No attention mechanisms, just use simple pooling", 0.94+ from that alone,
  ImageNet init, ablate MRI augmentation stacks, CV noise bar ~.003 (740610); Scott small ResNet .949; SpeedSci
  R50 5-fold .940 and EffNet-B4 .940 (735304); Myo Min Htet simple ResNet, 24 slices, heavy aug, 5-fold ~.94
  (743148). Ziad's contrary data point: attention head > masked mean by +.014 for RadImageNet R50 (737696).
- **Our measured context:** EfficientNetV2-S on v4 was too weak (OOF .8635) and diverged under the heavy TTR
  recipe in fp16 at lr 2e-4. That is a different backbone and the v4 label set, which is ~.016 weaker on LB
  for every family we tried. The prior F2 plan already named R50.a1 as the fallback but never ran it.
- **Why it might decorrelate:** different backbone family, different pooling (mean+max over slices per series,
  concatenate series features, MLP head) from both our attention ConvNeXt and the parent's CoAtNet/DINO stack.
  Our members agree at .92-.94 among themselves [MEASURED]; a pooling + backbone change is the cheapest way to
  move that number without changing labels.
- **Expected gain:** solo .936-.946 at 5 folds (GUESS); blended into the current .946 at r ~.88: **+.001 to +.004**.
- **Cost:** R50 at 224 has about ConvNeXt-T FLOPs, so ~4-5 V100 h per fold (fast trainer); 5 folds ~20-25 h.
  BatchNorm: use fp32 BN or `channels_last` + AMP with lr 1e-4 to 2e-4, AdamW, cosine, EMA, the TTR aug stack.
  EffNet-B0 is ~2x cheaper but historically fragile in fp16 (our EV2S divergence); start with R50.
- **First experiment and gate:** fold 0, seed 1043, mix50, TTR data pipeline unchanged, swap the head for
  mean+max pooling. Gate (pre-register): OOF vs mix50 >= TTR224 fold 0 minus .005, AND mean per-label Spearman vs
  the Raptor proxy <= .87 (TTR224 is .896), AND gold58 >= .905. Pass -> 4 more folds on the teammate's quota
  (lever 4) while the V100 continues lever 2 or 5.

### Lever 4: re-point the teammate's ~28 GPU h

- Currently packed: TTR mix50 seed 1044 on all 5 folds (`friend_gpu_pack`). Our own ledger says more seeds of the
  same recipe add ~+.0005 to +.001 (members correlate .92-.94; `PROGRESS_06-10-2026.md` sec 2 item 5).
- **Recommendation:** if the teammate has not started, give them the R50 simple-pool folds (lever 3) or, once
  lever 2's gate passes, the relabeled TTR folds. Same notebooks, a different config flag. Cost: a config change
  and a re-share (~1 h of Eesh's time). Gain: converts ~+.0005 into lever 2/3's +.001 to +.004.

### Lever 5: TTR at 288 px (running on the V100)

- MEASURED: fold 0 val .8878 at ep 13 vs .8836 for the 224 run. CLAIMED context: Raymond "384 -> 288 barely had any
  effect"; tennogh "not sure resolution is such a big driver"; Archit .950 at 224. Our ledger: 224 -> 320 +.002.
- **Expected:** as a *replacement* for the 224 set (same family, higher res): solo +.001 to +.003, blended
  +.0005 to +.002. As an *addition* next to the 224 set: ~+.0005 (same family, high r).
- **Cost:** ~1.65x the 224 compute, ~7 h/fold on the V100 -> 4 more folds ~28-35 h, i.e. most of the V100 until
  10-08/09.
- **Gate (already queued, keep):** final fold-0 OOF >= .8876. Add a second gate before folds 2-4: one SOLO LB read
  of the 288 fold 0 >= .940 (the 224 fold-0 solo was .939). If it reads .939, stop and give the V100 to lever 2/3.

### Lever 6: final picks and shake-up control

- MEASURED (leaderboard page): the public LB is ~30% of ~1,300 test studies, about 390 studies; private is ~910.
- GUESS arithmetic: with ~390 studies and 10-35% prevalence, one label's AUC has SE ~.01-.02; the 12-label macro
  ~.004-.006. Between two highly correlated blends (r ~.95) the SE of the *difference* is ~.001-.0015. So any
  public difference under ~.002 between our candidates is noise, and the .945-.950 band is a coin toss on private.
- CLAIMED: Georgy Mamarin's study of 11 comps: teams just inside or just outside the medal line on public finished
  with a medal 24% of the time either way (742327). PC Jimmmy: piles of forks of one public notebook shake hard.
  starkhushi: the 0.941 public fork recipe carries a per-label weight map tuned on the public split (Lateral
  Meniscus = 1.00); keep a flat-weight version as a final pick (742050). Mattia Angeli: minute-scale blend tweaks
  are public-LB overfitting (742050).
- **Plan:** pick 1 = best public blend with W in .4-.6 (flat over that range by our model). Pick 2 = the blend with
  the most own-model weight and no per-label public-tuned maps, if the parent kernel exposes a flat-weight switch
  (check whether our parent build includes the per-label map; if it does, build a flat version once). No
  per-label weights from public reads. Do not spend slots fishing W.

### Smaller items (cheap, low gain)

- **More TTA on the fleet** (slice offset +-1, small scale jitter): +.0003 to +.001, 0 training GPU, ~+10 min
  inference on 2xT4. Only if the 9 h budget has room after any new family.
- **Series coverage:** Sanjib: going from 24 to 32 cached slices per series (69% -> 84% of all slices) was worth
  more than any label change (743148, CLAIMED). We keep <= 32 slices and sample 24 triplet centres
  (`fleet_common.py`, MEASURED); series with > 32 slices (p90 39, p99 160 per Sanjib; median 30 per the data
  page) are thinned. Low priority; a store rebuild is not worth it this late.
- **Slice order:** willyp74 measured +.028 fold AUC from sorting by ImagePositionPatient instead of filename
  (745759). We already sort by IPP along the slice normal with an InstanceNumber fallback (`fleet_common.py`
  lines 74-76, MEASURED). gchauhan: InstanceNumber runs against position in over a third of series, so the
  fallback path is only safe when IPP is missing; no action unless IPP-missing series are common in test.

---

## 2. Dead ends confirmed or added by the forum (do not spend GPU here)

- Swapping or re-prompting text LLMs for labels (Sanjib: qwen2.5:14b / qwen3:14b / qwen3:30b within .010 on gold58;
  Tucker; Yann: label sets .86-.895 on gold gave very similar LB). Matches our closed lever.
- Report-embedding pretraining (Sanjib: BGE-M3 probe .842 vs .884 for extracted labels on gold58).
- Generating reports from images and re-extracting (Myo Min Htet: negative; Sanjib: lossy).
- Tuning to gold58 (SpeedSci: a label set at gold .945 scored only .910 LB; Tucker: 58 is too small).
- Blend-weight fishing on public (Mattia, starkhushi: the 200-team .941 pile).
- External datasets: host said click-through, research-only datasets are not excluded "on that basis alone"
  (733965), but KneeCoT is banned and the big list (MRNet, SKM-TEA, fastMRI+, OAI, KneeMRI) is still unanswered
  after 11 days (743416); Ali Ozan notes MRNet's agreement forbids derivative works. Tucker and YYama expect
  external data to move scores a lot and expect only top-10 audits. Keep our "assume not used" default; Eesh's
  call.

---

## 3. Hidden test distribution (what we know)

- ~1,300 test studies; public LB ~30%, private ~70% (MEASURED: data and leaderboard pages).
- "Diverse international mix of imaging sites ... scanners, protocols, and populations"; reports multilingual;
  no report at test time (MEASURED: data page). Prevalence "not guaranteed" equal across train/public/private.
  AUC is prevalence-invariant, so this mainly adds noise, not bias.
- Missing planes: Satwik (87th) asked whether every test study has all planes; the host has not answered (744519).
  Train studies have 3-14 series and the most common combination covers only 40% of studies (gchauhan, 741310).
  Our worker uses a presence mask for missing views [MEASURED in earlier audits]; keep the fallback for a study
  with zero usable views (predict the train prior) so one odd study cannot zero the run.
- Corrupted DICOMs exist in train (738708); the host never said the test is clean (740094). Keep the guarded
  loader.
- Effect on approach: nothing suggests site-specific tricks; heavy intensity augmentation and per-series
  percentile normalisation (which TTR already does) are the right defaults for an unseen-site mix (GUESS).

---

## 4. Suggested schedule (GPU = the V100 unless stated)

| dates | V100 | teammate (~28 h) | API / CPU | submissions |
|---|---|---|---|---|
| 10-06 night to 10-07 | finish TTR288 fold 0; gate | hold until lever 2/3 gate (or start R50 fold 0) | Eesh decides on an API key; build the key-slice pack for gold74 from OOF attention | none |
| 10-07 | R50 simple-pool fold 0 probe (~4-5 h), or TTR288 fold 1 if its gate passed | R50 folds 1-2 if the probe passes | Sonnet gold74 image vs text test (~$3); if pass, batch ~1,500 uncertain studies (~$15-20) | TTR288 f0 SOLO (1 slot) |
| 10-08 to 10-10 | relabeled-TTR fold 0, then folds 1-4 (or TTR288 folds) | R50 folds 3-4 | merge talks (Eesh) | R50 set SOLO + blend |
| 10-11 to 10-13 | whichever set passed, finish 5 folds | spare | - | relabeled set SOLO + 3-family blend |
| 10-13 to 10-15 | merge integration if merged | - | - | merged blends |
| 10-16 to 10-22 | second seed of the best new family only | - | - | 2 final picks per lever 6 |

---

## 5. Sources (read 10-06, signed out)

Threads: 745861 (hengck23 tricks; Ali Ozan .942 Claude labels; Chris Deotte on which images to show), 745949
(colum2131 multimodal relabel table), 746124 (Kaggle-only reproduction .936), 746060 (Thimsen merge), 745759
(slice order), 745823 (5-fold strategy; hengck23 on validation by label type), 745214 (Tucker: simple Qwen labels,
low .95s), 745091 (KalyanG17, Pand, Adel), 743374 (fast inference; Nikita Ageev label error analysis), 743148
(Sanjib measurements; Raymond on resolution; Myo Min Htet recipe), 743416 (external-data question, unanswered),
733965 (host: commercial LLMs allowed), 734109 (KneeCoT banned), 742327 (shake-up), 742926 (decorrelation),
741310 (gchauhan data traps), 740610 (Will ablations; Tucker recipe), 742050 (0.941 fork pile), 740094 (metric
edge cases), 744519 (missing planes question), 745576, 745730, 745820, 735304 (best single model; Tanuki .955).
Pages: competition Data tab, Leaderboard tab (30% public note; top 0.964, ~0.959 at rank 15-19, 0.956 at 49),
host overview 733343 (label definitions). `kaggle kernels list --sort-by scoreDescending` (public notebooks are
still parent forks at .941-.946; new: medvax "Independent 2.5D Reader Trial" .944, goodpjw2008 "2.5D ConvNeXt MIL"
.944, bliverrigpdoge "Raptor student blend" .944). Pricing: Claude API skill model table (cached 2026-09-25).
