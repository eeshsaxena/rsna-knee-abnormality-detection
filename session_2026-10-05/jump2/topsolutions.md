# RSNA Knee: what the strongest single models and top teams do that we do not (2026-10-05 ~10:30 IST)

Track: forum research for the leaderboard jump. Read-only: forum pages read signed out in the built-in browser,
no posts, votes or submissions. One small CPU calculation was run on the laptop (`jump2/blend_scen.py`, blend
arithmetic only). The box was only read (`~/fleet_runs` logs), nothing written there.

Tags: **MEASURED** = our own LB read, our run log, code we read, or a number computed today.
**CLAIMED** = a forum self-report (the poster's LB or gold58 number; nobody can audit it).
**MODEL** = equal-variance binormal blend arithmetic (`jump2/blend_scen.py`, same formula as `SP/fleet_g1b.py:best_lift`).
**GUESS** = my judgement. Ranks are the "Nth in this competition" badge shown next to each post on 10-05 (MEASURED from the page).
`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`.
Prior work this extends: `SP/bigjump/angle_topteams.md` (10-04 05:00). Everything below marked NEW was not in that file.

---

## 0. Bottom line

1. **The top single models are better students, not better-labelled.** Their honest gold58 scores sit at
   .925-.930 (Archit .930 OOF, Scott .930, Raymond ~.925, Renmq ~.93; CLAIMED) with LB .940-.950. Our best student
   reads .914 on gold58 (F1 mix50, held-out, MEASURED) with LB .932, and TTR v4 reads .879 with LB .923 (MEASURED).
   The gold58 gap (~.011-.016) matches the LB gap (~.015-.018). Their label sets score about .89 on gold58
   (Scott, Raymond, Tucker, CLAIMED), which is no better than our steven_v4 (.8927, MEASURED). So label source is
   not what separates us. Caveat: gold58 is a weak ruler across families (SpeedSci's CNNs read ~.89 gold with LB .940).
2. **Things they all do and we do not ship yet:** (a) all-data or 5-fold training instead of one 80% fold,
   (b) TTA (our fleet worker has none, MEASURED code read), (c) a pseudo-label round built from their OWN
   out-of-fold predictions, kept soft and blended into the report labels (Archit, Raymond, tennogh, SpeedSci,
   Chris Deotte).
3. **NEW, important nuance on (c):** Archit says the self-teacher only worked because the predictions were truly
   out-of-fold; in-fold predictions just echo the labels. Our mix50 teacher is Dread's CoAtNet, which Dread trained
   as ONE run over the full labelled set (forum, 08-27). Its train-set predictions are therefore in-sample: mix50 is
   closer to an average of two label sets than to Archit's OOF pseudo-labels. The real Archit round has never been
   run by us. The 5-fold TTR mix50 chain now on the box produces the first full own-OOF (all 4,407 studies).
4. **Counter-evidence (NEW):** Nicolai Karcher (1429th) ran a 50/50 label/prediction blend and saw gold and CV rise
   with zero LB change. Tucker (46th, low .95s) and tennogh (5th) both say better labels did not move their LB.
   Our own ledger has a null for flat self-distillation with a weak OOF (Tian). So the pseudo-label round is a
   real but uncertain lever: GUESS +0 to +.008 solo, centre +.003.
5. **Backbone pattern:** the highest single-model claims are CoAtNet 2.5D at 224-288 (Archit .950, Raymond .949,
   Renmq .940), ResNet-50 at 224 5-fold (CoolinLai 16th, .954), small ResNet/EffNet (Scott 10th .949, Tucker
   low .95s) and one Qwen3.5-2B VLM fine-tune at 384 (.950 single fold, NEW). ConvNeXt claims cluster lower
   (.930-.939: xukongji .939, ringbearer .937, kalyang .930). Our ConvNeXt-T sits in that cluster.
6. **Blend arithmetic (MODEL):** K2 (solo .932, W .30, displayed .943) implies rho_eff about .90-.92 for the mix50
   member. At that correlation, solo .940 lifts the parent by only ~+.002-.003 and solo .944 by ~+.004-.005. A
   decorrelated member at rho .84 needs only solo .936 for +.004. Decorrelation is worth as much as ~.008 of solo
   strength, so the family choice of the next member matters as much as its recipe.
7. **Strategy note from the forum (CLAIMED/GUESS):** several own-model teams at .947-.949 are openly looking for
   merges (SpeedSci .949 says labels are what they need most; ringbearer .948). Posters also expect the public
   .941-.946 blend pile to shake down on private. The merge stays the biggest lever. Eesh posts any request himself.

---

## 1. Recipe claims, one row per poster (all LB and gold numbers CLAIMED unless marked)

| Who (rank 10-05) | Date (approx) | LB claim | Labels: source / extraction | Pseudo-label loop | Backbone | Input | Training | Folds / seeds | TTA | Per-label tricks / notes | Thread |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CoolinLai (16th) | 09-28, 10-02 | .954 | n/a | n/a | ResNet-50 | 224 px | n/a | 5-fold | n/a | NEW | 735304 |
| Scott Willis (10th) | 09-05..09-16 | .938 single fold -> .947 5-fold -> .949 single fold single model | own, local Gemma 4, several passes, ~.89 on gold58 | none stated; spent most effort working around low-quality labels | smaller ResNet / EfficientNet | n/a | n/a | best is 1 fold | n/a | model gold58 ~.930; 5-min scoring; top efficiency LB | 735304 |
| Archit Konde (53rd) | 09-17..09-21 | .950 single fold | own extraction; checked calibration on gold58 (0.3 labels positive ~30%) | YES: OOF predictions MIXED into labels (not replaced, not masked); 50/50 jumped, heavier toward the model added a bit more; mostly helps report-silent cells; must be OOF | CoAtNet, 2.5D, a few neighbouring slices | 224 px | n/a | single fold | n/a | rounding to 0/1 hurt, keep soft; OOF gold58 .930 (Synovitis .797, Lat OA .816, PF OA .874); bigger models did not help | 735304 |
| Raymond Yuen / laymond (48th) | 09-16..09-23 | .938 -> .940 -> .949 (he credits all of it to label work) | multiple weak sources (public, API, local Qwen2.5-32B); label picked by LB ~.89 gold | YES: weak labels -> several teacher models -> teacher pseudo-labels -> final model; different teachers help different targets | CoAtNet | 288 px | full train (all data), eval on gold only | 1 model, all data | n/a | model gold ~.925; cheap and expensive LLMs give about the same labels | 735304, 742327 |
| tennogh (5th) | ~09-03 | .942 raw, .943 TTA | own labels, not measurably better than public ones | YES: OOF pseudo-labels, track LB well | n/a | 288 px | many small incremental improvements | n/a | yes (+.001) | resolution not a big driver | 735304 |
| Tucker Arrants (46th) | 08-16..10-03 | .934 -> .943 (5-fold, 224); later most models in the low .95s | own, simple single-prompt Qwen; gold ~.89; corrected thousands of extraction failures with no LB change | none stated | ResNet / EffNet (advises ResNet34 / EffNet-B0), simple pooling, no attention | 224 / 288 sweet spot | ablate generic + MRI aug stacks from past comps; CV noise bar .003 | 5-fold | yes, inside a 25-min kernel | trains WITH gold58 (1.3% of loss); validates on extracted labels only | 735304, 740610, 745214 |
| NguyenThanhNhan (67th) | ~09-21 | .950 single fold | n/a | n/a | **Qwen3.5-2B VLM**: LoRA on the language model + vision encoder fine-tuned jointly (NEW) | 384 px | ~3.5 h (unclear: train or run) | 1 fold | n/a | rented GPU; Kaggle GPU not enough | 735304 |
| Yann Majewski (85th) | ~09-01 | .936 single fold | own labels + a public set combined: +.015 LB | n/a | small ResNet | 224 px | encoder pre-training, longer training, more regularisation | 1 fold | n/a | CV-LB link broke when many things changed at once | 735304 |
| SpeedSci / caoyupeng (123rd-126th) | 09-29..10-03 | DINOv2 5-fold .942, EffNet-B4 .940, R50 .940; own-only ensemble .949 | GPT + Claude + Gemini vote, then refined with OOF/teacher models | YES (OOF/teacher refinement) | DINOv2, EffNet-B4, ResNet-50, CoAtNet | n/a | n/a | 5-fold each | n/a | label change alone moved DINOv2 .931 -> .942 (gold +.01); a label set with gold .945 scored only .910 (overfit to the 58) | 735304, 744511, 745214 |
| ringbearer (140th) | ~10-01 | blend .948 | n/a | n/a | CoAtNet 5-fold .943, CoAtNet variant .940, ConvNeXt .937, ResNet .933, RadImageNet model | n/a | n/a | 5-fold | n/a | own models only | 744511 |
| Renmq (1856th) | 10-05 | .940 | n/a | n/a | CoAtNet | n/a | all data except gold58 | 1 model; 4 ways of 5-fold all failed to beat it | n/a | gold58 ~.93 | 745823 |
| xukongji (228th) | 09-09 | one fold .939 (last epoch), 5-fold .938 (best-val epoch) | n/a | n/a | ConvNeXt-T, ImageNet | n/a | n/a | 1 vs 5 | n/a | closest analog to our member | 740375 |
| kalyang17 | 10-02 | .942 with public | own LLM labels | n/a | CoAtNet@384 .935, ConvNeXt-T .930, RadImageNet R50 2.5D 288 5-fold .924 | n/a | n/a | 5-fold | n/a | "never gave the public stack anything" | 745091, 744511 |
| Less (156th) | ~09-14 | .937 single fold | n/a | n/a | CoAtNet | 384 px | n/a | 1 fold | n/a | fusion (blending) did not help him | 735304 |
| myominhtet | 09-25 | .935 single fold, ~.94 5-fold | LLM labels, gold .9083 | n/a | simple ResNet | 24 slices, drops low-quality / no-info slices | heavy aug + regularisation; 50 ep, best at 28 | 5-fold | n/a | | 743148 |
| Dread Development (273rd; Raptor author, part of our parent) | 08-27..09-23 | Raptor single model .932-.938; .944 single in 13 min (09-19) | own labels (published; Chris Deotte uses them) | none: one CoAtNet-384 run over the FULL labelled set, small gold holdout picks the checkpoint | CoAtNet-384 | 80 slices per study over 2-98% of each series, 140 mm, 336 px; 5 slots (22 sag-fluid, 18 sag, 15 cor-fluid, 10 cor, 15 axial) | 40 models in 2 weeks | 1 run, no folds | slice windows: 42 -> 62 windows +.003, full coverage best | slice density ~1.2% of stack per slice is best; SWA null; attention pooling matters (-.02 if swapped at inference) | 737696, 742050 |
| Ziad Ahmed | 09-16..09-23 | n/a | n/a | n/a | RadImageNet R50 | **centre slice repeated x3 beat an adjacent triplet by +.020** (fold-0 OOF) for RadImageNet | n/a | n/a | n/a | attention head > masked mean (+.014); coverage saturates near 31 windows | 737696 |
| Cody_Null (2515th) | 10-01 | .931 single | own LLM tricks, labels ~.91 gold | n/a | n/a | FOV 100 -> 140 mm helped a lot; 160 no better | n/a | n/a | n/a | val .90, gold .92-.93 | 735304 |
| Parag (14th) | ~09-11 | n/a | n/a | n/a | n/a | first said 140-160 mm FOV; later: different models prefer different FOVs | n/a | n/a | n/a | | 735304 |
| willyp74 | 10-04 | n/a | n/a | n/a | n/a | sort slices by ImagePositionPatient, not filename: +.028 fold AUC (seed noise ~.001) | n/a | n/a | n/a | NEW; we already sort by IPP (MEASURED `fleet_common.order_slices`) | 745759 |
| Chris Deotte (122nd) | ~09-26 | n/a | uses Dread's labels | pseudo-labels from the best ensemble let any single model match the ensemble CV | n/a | n/a | n/a | n/a | n/a | models that disagree make better ensembles (Dread, agreed by Chris, 742926) | 735304, 742926 |
| hengck23 (GM) | 10-05 | n/a | NEW tip: show a VLM the report AND the image, ask it to label the ROI | n/a | n/a | n/a | n/a | n/a | n/a | needs an external API: NOT usable by us (no API keys rule) | 745861 |

Smaller data points (CLAIMED): MengYe 5-fold CoAtNet .932 with public labels; kmn CoAtNet single .930;
diet1236364 5-fold 224 .929 LLM labels; k256 5-fold 336 .926 LLM labels; Tom Aindow 392 px / 150 mm / random
bag of 32 slices, later .94+ at 224 with a small simple model; Berat (EffNet-B0) centre-9 adjacent slices beat
9 equal-spaced by +.018 while EMA (+.003), mixup (+.002), 60 epochs (+.001) stayed in noise; starkhushi
RadImageNet lifts Synovitis to .78 vs .62-.72 for ImageNet encoders.

---

## 2. Their recipe vs ours

Ours (MEASURED from `SP/bigjump/builds/code/*` and the run logs): ConvNeXt-T in22k; 4 views (Ax-FS, Sag-nonFS,
Cor-FS, Sag-FS); slices ordered by IPP; <=32 stored slices per series; 24 triplet centres (c-1, c, c+1) per view;
140 mm crop; 320 px store; per-target slice attention + per-target view attention with a presence mask; soft BCE.
F1 = 320 px, 10 epochs, light aug. TTR = 224 inside, 16 of 24 slots in training, heavy aug, 24 epochs, EMA.
LB members: one fold-0 model (80% of data), no TTA. gold74 is held out of training, so our gold58 numbers are
out-of-sample.

| Dimension | Top single-model pattern | Ours | Gap? | Evidence |
|---|---|---|---|---|
| Label source | own local/API LLM, ~.89 on gold58 | steven_v4 (.8927) and mix50_logit (.9262; gold-inflated by Dread's checkpoint pick) | **No** | CLAIMED .89 for Scott/Raymond/Tucker vs MEASURED ours |
| Pseudo-label loop | own OOF mixed into soft labels, heavier on silent cells, retrain (Archit, Raymond, tennogh, SpeedSci) | external in-sample teacher (Dread) mixed 50/50; no own-OOF round | **Yes** | CLAIMED; Dread's training protocol (737696) |
| Data used per member | all data, or 5 folds | one 80% fold on LB; 5-fold TTR mix50 running | **Yes, closing** | MEASURED runs; Tian cross-fit = all-data (ledger) |
| TTA | mirror / standard TTA (tennogh +.001, Tucker in-kernel) | none in the fleet worker | **Yes** | MEASURED code read (`fleet_worker.py`, K2/K3 cells: no TTA) |
| Backbone | CoAtNet 224-288, ResNet-50 224, small ResNet/EffNet, one VLM | ConvNeXt-T | **Partly** | ConvNeXt claims .930-.939 vs CoAtNet/R50 .940-.954 (CLAIMED) |
| Head | simple pooling (Tucker) or learned attention (Dread, Ziad) | per-target slice + view attention | Unclear | conflicting claims |
| Resolution | 224-288 | 320 (F1) / 224 (TTR) | No | MEASURED 224 -> 320 +.002 for us (ledger) |
| FOV / crop | 140-160 mm | 140 mm | No | Cody, Parag CLAIMED |
| Slice order | IPP along the slice normal | IPP | No | MEASURED code |
| Slice coverage | neighbouring triplets; dense coverage (Dread 80 slices) | 24 centres over <=32 slices = every slice | No (small risk for >32-slice series) | MEASURED code |
| Series slots | 5-6 (Dread 5, Tian 6) | 4 (no coronal non-FS, no axial non-FS) | Small | GUESS 0..+.002 |
| Schedule / aug | long + heavy aug + regularisation | F1 short/light; TTR long/heavy | F1 yes, TTR no | TTR OOF beat F1 on v4 by +.007 (MEASURED .8764 vs .8692) |
| External data | none in evidence | none | No | `EXTERNAL_DATA_VERDICT.md` |

---

## 3. Deltas ranked by plausible LB gain for OUR fleet (solo today: F1 mix50 fold-0 .932)

Blend column = best-W lift over the .943 parent (MODEL, `jump2/blend_scen.py`), at rho_eff .90-.92 for
mix50-trained members (implied by K2) and .84 for a decorrelated family. The binormal over-predicted by ~.002 on
one past member (Tian) and matched another (aastik), so read the blend column as an upper-middle estimate.

| Rank | Delta | Solo gain (from .932) | Blend gain on .943 | Cost | Ready | Confidence |
|---|---|---|---|---|---|---|
| 1 | **Ship the 5-fold TTR mix50 set (+ F1 fold-0) as one fold-averaged member, with mirror TTA** | +.004..+.008 (GUESS; anchors: 1 -> 5 folds +.009 on a weak member, MEASURED ledger; TTA +.001..+.003) | +.001..+.003 at rho .90-.92 | 0 extra V100 (already running); TTA doubles member inference (+~4-9 min on 2xT4, MODEL); 1-2 slots | chain ends ~05:00-05:30 IST 10-06 (fold 1 epoch 0 took 12.9 min, MEASURED -> ~5.2 h/fold); LB read ~10-06 evening | medium (solo), low-medium (blend) |
| 2 | **Decorrelated second family on v4 labels: ResNet-50 or EffNet at 224-288, simple pooling, all-data or 5-fold** (CoolinLai R50 224 .954, Scott .949, Tucker low .95s; `~/fleet2_code` F2 exists) | GUESS .926-.940 (our audited CNNs read .921-.924) | +.003..+.006 at rho .80-.86 even at solo .932-.936 | ~15-25 V100 alone-hours | train 10-06 05:30 -> 10-07/08; LB 10-08 | low-medium |
| 3 | **Archit round: own-OOF pseudo-labels** from the 5-fold TTR mix50 OOF. Targets = steven_v4 on report-addressed cells mixed ~0.5 with OOF; silent cells 0.75-1.0 OOF; keep soft; never use in-fold predictions; retrain all-data (TTR, 1-2 seeds; or the family of rank 2) | +0..+.008, centre +.003 (GUESS; Archit/Raymond/SpeedSci +.010-.011 CLAIMED vs Nicolai/Tucker/Tian nulls) | +.000..+.003 if trained on the mix50 OOF (rho stays ~.90); more if it feeds the rank-2 family | label build ~1 h CPU; ~6-7 V100 h per all-data TTR seed (MODEL from 12.9 min/epoch x 1.25) | labels 10-06 06:00; model 10-06 evening; LB 10-07 | low-medium |
| 4 | TTA alone (if rank 1 is delayed): mirror + slice-offset +-1 on the existing fold-0 member | +.001..+.003 (MEASURED Tian offset TTA +.003..+.006 OOF single member; CLAIMED tennogh +.001) | +.0005..+.001 | worker change + parity test, inference x2-x3 | 10-06 | medium |
| 5 | CoAtNet 2.5D at 224-288 instead of ConvNeXt (Archit, Raymond, Renmq, ringbearer) | +.003..+.008 GUESS | +.001..+.002 (CoAtNet = parent's Raptor family, rho up) | ~20-30 V100 h | 10-09+ | low |
| 6 | 5-6 series slots (add coronal non-FS / axial non-FS) | 0..+.002 GUESS | ~0..+.001 | store rebuild + retrain | after 10-10 | low |
| 7 | VLM moonshot: Qwen3.5-2B, LoRA + vision encoder, 384 px (one claim .950) | unknown; GUESS P(>= .940 in time) ~10% | +.008..+.012 IF it reached .945 at rho ~.80 (MODEL) | very high: 16 GB V100, new code, 2xT4 inference inside the parent's ~6 h | not before 10-12 | very low |

Not worth doing (forum + our ledger agree): re-extracting labels with a local LLM (CPU-only on the box; Tucker,
tennogh, Cody say no LB gain; our steven_v4 is already ~.89 gold), higher resolution, bigger backbone (DINOv2 S->B
+.0011), SWA (Dread null), swapping pooling at inference (-.02), external data (verdict: don't), slice order and
FOV (already right). hengck23's VLM ROI labelling needs an external API, which our rules forbid.

---

## 4. Blend arithmetic (MODEL, `jump2/blend_scen.py`)

K2 calibration: solo .932 at W .30 displayed .943 => rho_eff ~.90-.92 (blend .9438 at rho .90, .9430 at .92).

| solo \ rho | .80 | .84 | .88 | .90 | .92 |
|---|---|---|---|---|---|
| .932 | +.0049 | +.0032 | +.0016 | +.0009 | +.0003 |
| .936 | +.0063 | +.0044 | +.0027 | +.0019 | +.0011 |
| .940 | +.0078 | +.0060 | +.0041 | +.0032 | +.0023 |
| .944 | +.0096 | +.0078 | +.0060 | +.0050 | +.0041 |
| .948 | +.0116 | +.0099 | +.0081 | +.0073 | +.0065 |

Reading: to show .946 (silver line) from the parent alone we need ~+.003 true lift: a mix50-family member at
solo ~.940-.944, or a decorrelated member (rho ~.84) at solo ~.932-.936. Gold (~.958) is not reachable by our own
members on this arithmetic (needs solo ~.95+ at rho ~.80); it needs a merge.

---

## 5. What did NOT work for others (saves us slots)

- Masking noisy or silent label cells: no gain (Archit, Tom Aindow). Replacing labels with predictions: worse
  than mixing (Archit).
- 50/50 soft bootstrapping: gold and CV up, LB flat (Nicolai Karcher).
- Better extraction prompts / correcting thousands of extraction errors: no LB change (Tucker); fancy extraction
  overfits the 58 (Tucker, tennogh, SpeedSci gold .945 -> LB .910).
- Bigger encoders: DINOv2 S -> B +.0011 (stevenleehans); bigger models did not help (Archit).
- EMA, mixup, 60 epochs on a weak pipeline: all within noise (Berat).
- 5-fold vs one all-data model: no gain (Renmq, xukongji, our Tian ledger).
- Generating report text from images then re-extracting: loses information (aidiagnosis).
- Blend-weight and blend-operator tuning at a fixed member set: invisible at 3 decimals (zhincez, 13 subs at .937).

---

## 6. Sources (all read 10-05, signed out)

Forum threads (https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/<id>):
735304 (best single model; all reply trees expanded), 745214 (how do you optimize labels), 740610 (plateau .903;
Tucker's recipe), 737696 (Raptor weights; Dread geometry, Ziad), 744511 (SpeedSci teammates), 742926
(Chris Deotte list, decorrelation), 742327 (shake-up; Raymond), 740375 (one fold vs five), 745823 (5-fold
strategy), 738339 (.74 OOF tier), 735826 (bottleneck), 745759 (slice order), 745861 (hengck23), 745576, 736170.
Local: `SP/intel/knee_topics.json` (147 topics to 10-03), `SP/intel/knee_comments.json`,
`SP/intel/live/top_team_posts.txt`, `SP/bigjump/angle_topteams.md`, `SP/bigjump/builds/code/fleet_common.py`,
`fleet_worker.py`, `fleet_aug.py`, `SP/k3/cells/k2_fleet_cell.py`, `Downloads/rsna-knee/NEXT_SESSION.md`,
box `~/fleet_runs/auto_folds.log`, `f1_s1043_ttr_mix50.log`, `~/jump2/label_eval.txt` (sibling agent, gold58 of label sets).
