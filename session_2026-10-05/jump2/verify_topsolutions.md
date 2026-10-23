# Verify: "topsolutions" track (adversarial check, 2026-10-05 ~09:45 IST)

Target: `jump2/topsolutions.md` + `jump2/blend_scen.py` + the track's summary/findings/recommendations.
Tags: **MEASURED** (re-run or read by me today, source given) | **CLAIMED** (forum self-report) | **MODEL** (binormal
arithmetic) | **GUESS** (judgement). `SP` = the session scratchpad. Box work: CPU only (`CUDA_VISIBLE_DEVICES=`,
`nice -n 19`), written only under `~/jump2/vts/`. Kaggle: signed-out reads only, nothing posted/voted/submitted.

## 0. Verdict

The forum transcription is mostly faithful (I re-read every 735304 reply plus 11 other threads from the saved JSON,
and re-fetched 745214 + 744511 signed out: byte-identical). The arithmetic in `blend_scen.py` reproduces exactly.
What does not survive:

1. **The headline inference ("better image models, not better-labelled"; "the gold58 gap matches the LB gap") is
   refuted by our own members.** F1 mix50 minus TTR v4 is +.0365 on gold58 (paired bootstrap SE .0087, MEASURED) but
   only +.009 on LB, a slope of ~0.25, not 1. A single member's gold58 macro has bootstrap SE .014-.017, so "their
   ~.927 vs our .914" is under 1 SE. SpeedSci's own models (gold .917/.89/.915/.945 vs LB .942/.940/.926/.910) show
   gold58 does not even rank models. Our own 10-04 note already said "gold58 does not rank families". And three of
   the posters the track cites (Raymond, SpeedSci, Yann) attribute their LB gains to label/target work.
2. **The "NEW" point that mix50's teacher is in-sample is not new and its implication is contradicted by our ledger.**
   HANDOFF 10-03 already measured "teacher echo is mild" (.912 in-sample vs .886 held-out) and records Tian:
   a strong in-sample Raptor teacher gave +.009 LB while OOF self-distillation gave -.001.
3. **Rank 3 (the "Archit round" from our own OOF) is dominated at the label level, MEASURED today.** Built on gold58
   with J1's held-out predictions, the track's recipe scores .9254-.9263 on gold58: the same as mix50_logit (.9262)
   and below the already-built mixSA_logit (.9300; paired diff -.0036 to -.0045, SE ~.0075). Expected gain ~0.
4. **The decorrelated-family blend gain (+.003..+.006 at rho .80-.86) has no anchor.** K3 (parent + mean(F1 mix50,
   TTR v4) at W .25 = .943) forces TTR v4, our least parent-like member, to rho_eff >= ~.86 (MODEL). The same
   Spearman-to-rho reasoning predicted +.004..+.009 for the TTR fleet on 10-04 (rho .77-.80) and K3 read zero.
5. **Misattributions/misreads:** "Kaggle GPU not enough" was said by Ace (853rd), not NguyenThanhNhan; "rented GPU"
   comes from SpeedSci's remark about a 9B model. SpeedSci's ".945 gold -> .910 LB" is an EffNet MODEL's gold58 score
   on one label version, not a label set's score. tennogh's "OOF pseudo-labels ... correlated with LB" may describe
   validation, not training. SpeedSci's ".947-.949 own-model" ensemble includes a public single-model CoAtNet (.928).
6. **The merge recommendation needs three corrections:** a merge with a .947-.949 team does not reach gold (.958) even
   on the track's own GUESS (.943 + .012 = .955); Kaggle forbids private sharing of code/data outside a team, so the
   mix50 labels, model outputs or V100 runs can be described in a pitch but not handed over before the merge
   completes; and the merged team's total submissions must be under the cap (HANDOFF: two-way merges only).
7. **Timing:** "1.5-2 h later than 03:30" is too pessimistic when the GPU is free: J1 (fold 0) took 5.0 h wall alone
   (MEASURED), so folds 1-4 end ~04:35 IST 10-06. But right now a GPU benchmark from another track
   (`~/fast_build/speed_ab.py`, timeout 50 min from 09:33) and a CPU job (`~/jump2/gold_infer.py`, ~11.5 cores)
   are slowing fold 1 from 0.36 to ~0.5 s/it (MEASURED). Realistic end: 04:35-05:15 IST.

---

## 1. Claim-by-claim

| # | Track claim | Status | What I found | Tag |
|---|---|---|---|---|
| 1 | Top single models ~.925-.930 on gold58 with LB .940-.950; ours .914/.932 and .879/.923; "gold58 gap ~.011-.016 matches the LB gap" | **Numbers confirmed, inference refuted** | Archit .930 OOF (09-18), Scott ~.930 (09-05), Raymond ~.925 (742327), Renmq ~.93 (745823): all in the JSON. Ours reproduced: F1 mix50 .9144, TTR v4 .8780 (box `~/jump2/vts/g58_check.py`). Bootstrap SE per member .0144-.0195; F1mix50 - TTRv4 = +.0365 (SE .0087) on gold58 vs +.009 LB, slope ~0.25. SpeedSci's 4 model points are non-monotone. Scott never says his gold58 rows were held out (Tucker trains WITH them). Our mix50 members are gold-inflated (Dread picked checkpoints on the gold holdout). | MEASURED / CLAIMED |
| 2 | Label source is not the gap: their labels ~.89 on gold58 = our steven_v4 .8927 | **Uncertain** | .89 quotes confirmed (Scott, Raymond "picked by LB", Tucker "maybe .89, a bit lower"). steven_v4 .8927 reproduced, but its bootstrap 95% CI is [.857, .924]: gold58 cannot tell .87 from .91 labels. Raymond credits all of .938 -> .949 to labels (incl. pseudo-labels), SpeedSci .931 -> .942 from a label change alone, Yann +.015 from label work. So raw extraction may be at parity, but their training targets are not. | MEASURED / CLAIMED |
| 3 | Archit's recipe: mix OOF predictions into soft labels, heavier on silent cells; replace/mask/round worse; must be OOF | **Confirmed (as one poster's claim)** | 735304 ids 3525701, 3526342: 50/50 jumped, heavier added a bit; replacing worse than mixing; rounding hurt; "in fold ones just parrot the labels back". Masking null also from Tom Aindow. n = 1 poster; his mix weights were steered by gold58 inspection (calibration check). | CLAIMED |
| 4 | NEW: mix50 teacher in-sample, so mix50 ~ average of two label sets; the real round was never run | **Partly refuted** | Dread 08-27 quote confirmed (one CoAtNet-384 run over the full labelled set, small labelled holdout picks the checkpoint). Not new: HANDOFF line 187 (10-03) measured the echo as mild and records Tian's in-sample teacher +.009 LB vs OOF self-distillation -.001. See section 2.2 for today's label-level test. | MEASURED (ledger) / CLAIMED |
| 5 | Evidence against label/pseudo-label work moving LB (Nicolai, Tucker, tennogh, SpeedSci ".945 label set -> .910") | **Partly refuted** | Nicolai (1429th) 50/50 null: confirmed, but his OOF status and base LB are unknown. Tucker: confirmed. tennogh: "own labels not measurably better than public": confirmed. SpeedSci: misread; it is "a high Gold of 0.945 on EffNet" for one label version, i.e. a model's gold58 score (745214, re-fetched). | CLAIMED |
| 6 | Our fleet has no TTA; top teams use TTA | **Confirmed, with an omission** | `fleet_worker.py` and `k2_fleet_cell.py` have no TTA path (grep). But `tta_bench.py` (gate G-TTA: passes A/M/P/N/B, gate bar +.002 on val with P > .9 and gold58 >= 0) was built on 10-04 and only smoke-run on 3 studies (`~/bigjump_build/out/gp0/tb_real.log`); `~/bigjump_build/tta_*` does not exist. Models train with mirror p .5 (and TTR with slice jitter +-1), so mirror TTA gains should be small. tennogh +.001, Tucker TTA in 25-min kernel: confirmed. | MEASURED / CLAIMED |
| 7 | Best families: CoAtNet 224-288, ResNet-50 224 5-fold, small ResNet/EffNet; ConvNeXt lower .930-.939 | **Confirmed as claims, weak** | All quotes present. Within-team gaps are ~.005 (ringbearer cnx .937 vs CoAtNet .943; kalyang .930 vs .935), xukongji's ConvNeXt-T .939 sits at the CoAtNet range, and SpeedSci's CoAtNet (.926) is below their ResNet (.940). CoolinLai's ".954" is not labelled LB or single-model in the post. | CLAIMED |
| 8 | Qwen3.5-2B VLM .950 single fold, ~3.5 h; "he says the Kaggle GPU was not enough"; table: "rented GPU" | **Partly refuted** | .950, 2B, LoRA on the LM + joint vision-encoder tuning: confirmed (ids 3526410, 3527348). "Kaggle GPU is not sufficient" is Ace (853rd, id 3530733), replying to someone else. "Renting" is SpeedSci about a 9B. Whether 3.5 h is training or inference is unstated. | CLAIMED |
| 9 | Our input already matches what works: IPP sort, 140 mm, 2.5D triplets, full coverage | **Mostly confirmed** | `fleet_common.order_slices` sorts by IPP along the slice normal (code). Run contract: crop 140 mm, store 320, 24 triplets from <=32 source slices per view (fold-1 log header). willyp74's +.028 was on a .77-.80 baseline. Parag retracted 140-160 mm ("different models do well with different FOVs"). Series with >32 slices are subsampled, so "full coverage" holds only up to 32. | MEASURED / CLAIMED |
| 10 | K2 implies rho_eff .90-.92; solo .940 adds +.002-.003; a rho .84 family needs only solo .936 for +.004 | **Arithmetic confirmed, premise uncertain** | `blend_scen.py` reproduces to 4 decimals. Exact-value K2 range is .907-.932 (rho .90 gives .9438, which would display .944); with display rounding of all three numbers it is .895-.944 (`jump2/vts/blend_check.py`). K2's member has a best-W lift of only +.0006 to +.0010. K3 forces TTR v4 to rho_eff ~.86-.925 (MODEL), so nothing we have measured sits at .84. The parent already contains RadImageNet heads, so ResNet-50 is not a new family to it. | MODEL |
| 11 | All-data ~ 5-fold, so most of the 5-fold gain over fold-0 is data + averaging | **Confirmed as claims** | Renmq: 4 fold schemes never beat all-data .940 (no claim they were worse). xukongji: "One fold (No valid setup)" .939 vs 5-fold .938. Tian cross-fit .928 vs all-data .927 is Tian's ledger, not ours. | CLAIMED / ledger |
| 12 | 5-fold TTR set ends ~05:00-05:30 IST 10-06 (1.5-2 h late) | **Partly refuted** | J1 fold 0: launched 23:12, log last written 04:13 = 5.0 h alone (MEASURED). Fold-1 epochs 1-2: 12.6-12.7 min each including evals. `~/auto_folds.sh` polls every 120 s, so the gaps are small. Projection: ~04:35 if uncontended, ~1 h after 03:30. Contention now: GPU benchmark `fast_build/speed_ab.py` (35 open nvidia fds, timeout 3000 s from 09:33) + CPU `gold_infer.py` (1152% CPU, nice 19); fold-1 epoch 4 is at 0.48-0.56 s/it vs 0.36. Realistic end 04:35-05:15. | MEASURED / MODEL |
| 13 | Own-model teams at .947-.949 want merges; SpeedSci wants labels; posters expect the public blend pile to drop | **Partly refuted** | SpeedSci's .947 ensemble includes "a public single-model CoAtNet at 0.928" (744511, re-fetched); that is plausibly a public Raptor checkpoint, i.e. our parent's family (GUESS). ringbearer never says "own models only". "We need most is a good label": confirmed. Shake-up expectations come from posters ranked 3186th and 1430th, while Georgy Mamarin's base-rate study says the public bronze line barely predicts private medals (24% either side). | CLAIMED |

---

## 2. Re-run computations

### 2.1 gold58 (box CPU, `~/jump2/vts/g58_check.py`, 58 rows, all 12 labels have 9-35 positives)

| Set | gold58 macro | bootstrap SE | 95% CI |
|---|---|---|---|
| steven_v4 labels | .8927 | .0169 | .857-.924 |
| mix50_logit labels | .9262 | .0127 | .900-.950 |
| gated75 labels | .9109 | .0148 | .879-.937 |
| F1 mix50 (f0_s42 raw) | .9144 | .0144 | .884-.939 |
| F1 v4 (f0_s42_v4 raw) | .8611 | .0195 | .819-.897 |
| F1 gated (raw) | .8846 | .0161 | .851-.913 |
| TTR v4 (f0_s1042 ema) | .8780 | .0165 | .842-.907 |
| TTR mix50 J1 (raw) | .9132 | .0147 | .882-.938 |

Paired: F1 mix50 - TTR v4 = +.0365 (SE .0087); F1 mix50 - steven_v4 labels = +.0217 (SE .0142). All MEASURED.

### 2.2 The Archit round at label level (box CPU, `~/jump2/vts/archit_g58.py`)

Self-teacher = J1 TTR mix50 fold-0 raw. gold74 is held out of its training, so its gold58 predictions are
out-of-sample, the same status as an OOF prediction. Mapping = logit mean/std match onto steven_v4 (as
`build_mixSA.py logit`). Silent = `report_labels_v2` verdict UNK (27.2% of gold58 cells).

| Label set | gold58 | minus mixSA_logit (SE, P>0) |
|---|---|---|
| steven_v4 | .8927 | -.0374 (.0103, .00) |
| own prediction alone | .9132 | -.0166 (.0099, .04) |
| own 50/50 flat (Nicolai/Archit start) | .9239 | -.0060 (.0073, .20) |
| own 0.5 addressed / 0.75 silent (track rank 3, low) | .9263 | -.0036 (.0072, .32) |
| own 0.5 addressed / 1.0 silent (track rank 3, high) | .9254 | -.0045 (.0079, .28) |
| own 0.75 flat | .9252 | -.0046 (.0079, .28) |
| mix50_logit (Dread teacher, what we train on now) | .9262 | -.0038 (.0027, .07) |
| mixSA_logit (Dread teacher, silence-aware, already built) | .9300 | 0 |

Reading (MEASURED, caveat: gold58 favours Dread-teacher sets because Dread picked checkpoints on the gold holdout):
the own-OOF round produces targets equal to mix50 on gold58, so as a label change it is worth ~0. A 5-fold OOF does
not make the teacher stronger on train cells (each train study still gets one fold's prediction).

### 2.3 Blend arithmetic (laptop, `jump2/vts/blend_check.py`, MODEL)

- K2 rho_eff: .907-.932 at exact values, .895-.944 allowing display rounding of parent, solo and blend.
- Max lift of the K2 member at its best W: +.0006 to +.0010.
- K3, 3-variable binormal with rho_FP .91-.92 and member-member r .92-.94: TTR v4 rho_TP must be in ~.86-.925 for the
  W .25 blend to display .943; the implied mean-member solo is ~.931.
- A rho .84 member at solo .923-.930 would have shown +.0012 to +.0026 at W .25 on K3's terms; K3 showed 0.
- One external calibration point the track left out: Charles Savas (735304, 09-04) says an external arm with
  correlation .83 against a DINOv2-heavy .943 stack was worth +.004 public across four submissions (CLAIMED).

---

## 3. Corrected recommendations (ranked)

1. **Keep: ship the 5-fold TTR mix50 set (+ F1 fold-0) as one averaged member.** Solo +.003..+.006 over .932
   (GUESS; the top of the track's +.008 needs TTA to pay too). Blend +.000..+.002 (MODEL: averaging raises rho toward
   the K2 range .91-.93). Before adding TTA to the worker, run the already-built G-TTA gate (`tta_bench.py --passes
   A,B`) on J1 and fold-1 checkpoints when the GPU is free; serve A+B only on PASS. Inference: 6 members x 2 passes is
   ~+8 min per T4 (MODEL from the K3 runbook's 0.30 / 0.15 s/study estimates and 0.68 s/study decode). Ready: chain
   end ~04:35-05:15 IST 10-06.
2. **Keep, lower the expectation: a second family**, preferably EfficientNet (absent from the parent) over ResNet-50
   (the parent has RadImageNet R50 heads), on steven_v4. Blend +.000..+.003 (GUESS; nothing measured sits at rho
   .84). There is one V100 (`/dev/nvidia0`), so it serializes with anything else after the chain: ~15-25 h, LB 10-08.
3. **Demote: the own-OOF "Archit round".** Section 2.2 shows its targets tie mix50 on gold58. If a label change is
   wanted, mixSA_logit (built, .9300, the same silence-aware weights with the stronger teacher) dominates it, at the
   known cost of higher parent-likeness (HANDOFF 10-03: teacher mixing raised Raptor rho .79 -> .83-.84 for crop
   members). Expected LB gain of the own-OOF round ~0 (+0..+.002, GUESS). Never pick mix weights on gold58.
4. **Keep only behind the gate: TTA on the existing fold-0 member.** Mirror is already a training augmentation
   (p .5), so expect +0..+.002 solo, ~+.0005 blend. Spend a slot only if G-TTA passes.
5. **Keep as the biggest lever, but correct the pitch: merge outreach by Eesh.** It is a silver play, not gold
   (.943 + .005..+.012 = .948-.955, GUESS). Describe, do not transfer: Kaggle bans private sharing of code or data
   outside a team, so labels, predictions and V100 runs move only after the merge is accepted. Check the combined
   submission count against the cap at the request date first (HANDOFF: two-way only). SpeedSci already blends a
   public CoAtNet and can add the public parent themselves, so what is uniquely ours is compute plus fleet members.
6. **Keep the skip list.** hengck23's tip would send competition images to an external API, which our rules forbid
   (and it is likely a competition-data-sharing problem too). The Qwen3.5-2B VLM stays a long shot: one V100, already
   booked through 10-08 by items 1-2.

---

## 4. Rules check

- The track's own work: no box writes found under `~/fleet_runs`, `~/rsna_train`, `~/fleet`, `~/bigjump_build` by
  this check (not exhaustive). The forum JSON cookie jars in `jump2/cj.txt` and `jump2/verify/cj2.txt` are anonymous
  (`"anon":true`), so the reads were signed out. No posts or votes found.
- Not this track, but seen while timing: `~/fast_build/speed_ab.py` (PID 339781, nice 5, no CUDA_VISIBLE_DEVICES, 35
  /dev/nvidia fds) is running a GPU A/B on the V100 alongside fold 1. If it came from a subagent, it breaks the
  "never use the GPU" rule. I did not touch it.
- My own actions: CPU-only scripts under `~/jump2/vts/`, nice 19. One read-only `grep -rln G-TTA ~` that I launched
  without nice ran ~5 min at ~10% CPU and exited on its own; nothing was killed or signalled. Two signed-out forum
  re-fetches; the temporary anonymous cookie jar was deleted.

## 5. Files

- This note: `SP/jump2/verify_topsolutions.md`
- Laptop: `SP/jump2/vts/blend_check.py`, `SP/jump2/vts/t735304_flat_mine.txt`, `SP/jump2/vts/re_745214.json`,
  `SP/jump2/vts/re_744511.json`
- Box: `~/jump2/vts/g58_check.py`, `~/jump2/vts/archit_g58.py`
