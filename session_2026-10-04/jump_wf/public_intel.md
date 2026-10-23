# Public intel, RSNA Knee (2026-10-04, ~15:00 to 16:00 IST)

Scope: read-only scan of the public leaderboard, public notebooks, public datasets and public Kaggle Models, looking for anything not already in our parent (0.943) that could lift us quickly.

Tags: MEASURED = read or computed by me from a file or page (source given), MODEL = derived from a stated model, GUESS = judgement.

## Bottom line

1. **No public asset forks above 0.943.** MEASURED: only 5 public notebooks score above 0.943 (0.947, 0.946, 0.945, 0.945, 0.945). All 5 attach at least one **private dataset** (their `dataset_sources` come back as `""` from `kaggle kernels pull -m`, and the Kaggle page for the 0.947 one shows `[Private Dataset]`). If you fork one, the private member is missing and you fall back to the public stack. matterhorn3838/rsna-knee-version-3 is a copy of pj d4-blend without the private assets, and it scores **0.943** (MEASURED, score listing).
2. The four d4-family notebooks above 0.943 (pj d4-blend .946, pj d4-lite .945, aastik s75-w50/w60 .945) all use **the same recipe as our K2**: the public d4 stack (byte-identical to our parent's 14 public datasets), plus a **private independent fleet**, combined by a **single global rank blend at W .45 to .60**. So the public evidence says the lever is our own fleet, not anything we can download. K2 (pending, ~17:50 IST) is the right experiment.
3. New public weight datasets with **no public score** exist. The most interesting ones come from teams above us: tbongc ("I eat dogs", LB .947) published 2 Raptor CoAtNet full-data retrains and a 4-fold EffNet-B0 MIL. kawaharataishi (LB .944) published a CoAtNet-384 distilled SWA checkpoint. None of these is used in any public notebook, so nothing measures their LB value. Every one needs integration work plus an LB slot.
4. Leaderboard: 5,082 teams. The 0.943 tie now holds **944** teams (was 683 on 10-01). **31 teams** moved above 0.943 in the last 24 h (18 of them straight from 0.943) and 78 in about 3 days. Silver line (rank 254) = **0.945**, gold line (rank 20) = **0.958**.

## 1. Leaderboard shape

Source: `jump_wf/lb/rsna-knee-abnormality-detection-publicleaderboard-2026-10-04T09_27_20.csv` (MEASURED, downloaded 14:57 IST via `kaggle competitions leaderboard -d`). Script: `jump_wf/lb_stats.py`, `jump_wf/lb_move.py`.

| snapshot (UTC) | teams | >=.950 | .945-.949 | .944 | .943 | .942 | >.943 | top | gold (rank 20) | silver (top 5%) | bronze (top 10%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10-01 06:52 | 4702 | 86 | 138 | 55 | 683 | 182 | 279 | .961 | .956 | .944 | .943 |
| 10-02 03:24 | 4816 | 92 | 145 | 58 | 781 | 177 | 295 | .961 | .957 | .944 | .943 |
| 10-03 06:09 | 4957 | 101 | 157 | 58 | 880 | 173 | 316 | .961 | .957 | .945 | .943 |
| 10-03 09:25 | 4967 | 102 | 162 | 57 | 883 | 180 | 321 | .961 | .957 | .945 | .943 |
| **10-04 09:27** | **5082** | **106** | **182** | **63** | **944** | **186** | **351** | **.963** | **.958** | **.945** | **.943** |

All MEASURED. The older snapshots are from earlier sessions (`scratchpad/` root, `lb_extract/`, `intel/lb/`, `intel/lb2/`). Medal lines use the standard Kaggle rule for N >= 1000: gold = top 10 + 0.2% N, silver = top 5%, bronze = top 10% (MODEL).

- Score buckets on 10-04: >=0.950: 106 · 0.945-0.949: 182 · 0.944: 63 · 0.943: 944 · 0.942: 186 · 0.941: 279 · 0.940: 163 (MEASURED).
- 0.943 tie spans ranks 352 to 1295. 0.944 spans 289 to 351, and 0.945 spans 226 to 288 (MEASURED).
- **Us:** team "Eesh saxena" (eeshsaxena, swapiy) is at **rank 419, 0.943**, 40 submissions, last 10-04 00:00 UTC (MEASURED). The tie is broken by earlier submission time. Rank 419 sits inside the public bronze zone (rank 508 is the cutoff), but that zone shrinks as more teams leave the tie, and the private LB decides.
- Top 20 (MEASURED): .963 rhoskeri, .962, .961 joinT fusion, .961 Wassim Dobbi, then .959 x5 (TomKroumov, NMarkS, knee-deep, YOKKOISHO, Scott Willis), then .958 x11 (takoi & charmq, AndreyKvachov, Parag & Ian, Attention Is All You Knee, dalab, YC & NB, daimaru, Volkopat, Q, Brandon Low, Mazzutti).
- **Movers** (TeamId join, 10-03 09:25 to 10-04 09:27): 31 teams newly above 0.943: 13 to .944, 5 to .945, 5 to .946, 3 to .947, 3 to .948, 1 to .949, 1 to .950. 18 of the 31 came from exactly 0.943 (MEASURED). Over about 3.1 days (from 10-01 06:52): 78 movers, 19 to .944, 17 to .945, 15 to .946, 9 to .947 (MEASURED). The steady stream of +.001 to +.003 lifts out of the tie matches the "own fleet at W .45 to .6" recipe (GUESS based on the public examples below).
- Known authors on the LB (MEASURED): analyticaobscura (OzanM.) .951 r87 · jiweiliu (team Theo Viel) .951 r83 · mattiaangeli .950 r103 · tonylica .950 r104 · pjmathematician .947 r153 · **tbongc .947 r176** · aastikrajan15 .946 r222 · dreaddevelopment .945 r262 · **kawaharataishi .944 r304** · fabotelli .944 r307. romantamrazov, yamadan96, sushanth, prvsiyan, evgendvorkin, andreluizpedroso and others are all at .943.

## 2. Public notebooks

Listing: `kaggle kernels list --competition ... --sort-by {dateRun,scoreDescending,voteCount} --page-size 100 --page 1..3` gives 300 unique per sort (MEASURED, `jump_wf/kl_*.csv`). 111 public notebooks were run since 2026-09-28 (MEASURED). The CLI does not show scores. I read scores from the Kaggle web page sorted by "Public Score" and from individual notebook pages (built-in browser, signed out, read-only).

**Every public notebook scoring above 0.943 (complete list, MEASURED from the score-sorted code page):**

| notebook | public | updated | private assets? | what it is |
|---|---|---|---|---|
| analyticaobscura/rsna-knee-v4 "RSNA Knee V4" | **0.947** | 10-02 | **yes**: its only dataset is `[Private Dataset]` | Standalone 1-cell runner ("Lab v4"): one private dataset of its own members + `infer_core.py`/`knee_v4_prep.py`, weighted mean, parity gate. Uses **none** of the d4 stack. Runtime 1m10s on the public test. Team OzanM. is at .951. Comment thread confirms nobody can reproduce it. |
| pjmathematician/rsna-knee-d4-blend | 0.946 | 09-28 | yes (3 blank sources: `rsna-knee-eff6-assets`, `rsna-knee-ens14-assets` + 1) | Our parent's d4 stack (30 of 33 cells byte-identical to our parent) + "OUR LEG" fleet (arms S_dist, S_coat, S_cnxl) at **one global rank weight `RSNA_OURS_W=0.45`**. Searching for `ens14-assets`/`eff6-assets` finds no public dataset. |
| pjmathematician/rsna-knee-d4-lite | 0.945 | 09-26 | yes (3 blank) | Same structure as d4-blend. |
| aastikrajan15/knee-s75-w50 | 0.945 | 09-27 | yes (1 blank) | d4 stack + base64 `ours_stage.py` (ConvNeXt-T fleet) at `OUR_WEIGHT=0.5`. Also sets a flat `_coatnet_weight=0.70` for every label and adds Fracture to `_RAD_EXCLUDE`. |
| aastikrajan15/knee-s75-w60 | 0.945 | 09-27 | yes (1 blank) | Same notebook at W .6 (byte-identical size to w50). |

**Recent or high-vote public notebooks at or below 0.943 (scores MEASURED from notebook pages):**

| notebook | public | new vs our parent |
|---|---|---|
| romantamrazov/rsna-knee-dinosaur-v5 (127 votes, 10-02) | 0.943 (best V23) | No new weights. Re-uses the d4 graph and adds about 300 lines of internal re-weighting: ACL/MCL only = 0.927 calibrated transformer + 0.055 fold-soft DINO + 0.018 weighted DINO; CoAt family = 0.75 prob-mean + 0.25 rank-mean; Rad excludes Fracture. Attaches dinov2/base (model source). Zero-training fork, **no lift** (0.943). |
| dantian77/rsna-knee-dinosaur-v5-audit | no score | Byte-identical size to DINOsaur V5. |
| romantamrazov/rsna-knee-dinosaur-v5-train (10-04, internet on) | n/a (trainer) | ConvNeXt-T trainer: 150 mm, 224 px, 3 adapt epochs, 4 views, plus label-aware heads fit on **Gold58 5-fold OOF**. Its current public output (`kaggle kernels files`) is an **old Aug-31 DINOv3 slothead v16 run**, so no ConvNeXt weights are public yet. Watch item only. |
| andreluizpedroso/rsna-knee-tony-rank-ensemble | 0.942 (best V29) | Uses dreaddevelopment/raptor-knee-finespacing (v9) + tonylica/rsna-knee-bend-dinov3-0917-repro-assets. We already measured v9 swapped in: 0.943 no-op. |
| anthonytherrien/rsna-knee-dino-radimagenet-dual-coatnet | 0.934 | raptor-knee-widedense (v4, our single .924) + tonylica dinov3 repro assets. |
| prvsiyan/the-bee-s-knees-final-rsna-push (83 votes) | 0.941 | Older stack without global96/d4 CoAt. |
| sushanthtiruvaipati/rsna-knee-d4-public0946-no-correction | 0.943 (best V1) | Latest version adds an "MCL swap" cell that swaps the top-2 MCL predictions. Harmful noise on the hidden set. Attaches mattiaangeli/rsna-knee-cnx-m448-f0-public. |
| kinjaza/rsna-knee-d4-cnx-r1-five-percent | no score | d4 + ConvNeXt M448 f0 (public) at 5% rank weight. We MEASURED this family hurting at 40-50% (0.930 / 0.935). At 5% it is a near no-op (MODEL). |
| sushanthtiruvaipati/rsna-knee-d4-mcl-latmen-blend-v1 | no score | Same cnx-m448 family. |
| ryokucha/rsna-knee-d4-blend-0946-ours05 | public 0.942, best 0.943 | pj d4-blend copy with the private legs missing (2 blank sources). |
| matterhorn3838/rsna-knee-version-3 | 0.943 | pj d4-blend copy without private assets: falls back to 0.943. |
| adonis122/rsna-knee-parent0957-routing | 0.939 | Older stack plus routing. The name is not a 0.957 asset. |
| adonis122/rsna-knee-0943-four-reader | 0.943 | Line-identical to yamadan96 d4-public0946 (0 lines differ). |
| udayr77/rsna-knee-abnormality-detection | 0.943 (best V5) | 28 sources incl. 3 private + Dread corpora. Rewritten pipeline, no lift. |
| nswitzer/creaky-0-94-ensemble | no score (team .885) | d4 + 1 private dataset ("creaky" ConvNeXt). |
| arey34/rsna-knee-v38-win94-submit | no score | Older v38 stack. |

Diff method (MEASURED): `jump_wf/celldiff.py`, `jump_wf/linediff.py`. Pulled code is in `jump_wf/pub/<owner>_<slug>/` (`code.py` + `diff_vs_*.txt`). Our parent's cells are byte-identical to all 31 cells of yamadan96/rsna-knee-d4-public0946. Our parent attaches exactly the same 14 public datasets + 2 kernel sources + dinov2/small as every d4-family notebook, plus `eeshsaxena/rsna-knee-cnxt-fleet`.

Discussion (MEASURED, topic 745759, 2 h old): "Slice order in the public baseline is basically random (+0.028 AUC)". It applies to forks of the plain baseline's `read_slot(sorted(filename))`. Our parent's `read_slot` already uses `rec['ordered']` (IPP/InstanceNumber, `ORDER_TAGS`), and our fleet sorts by IPP. **Not a lever for us.**

## 3. Public datasets and models since 2026-09-25 that our parent does not attach

Search: `kaggle datasets list -s {"rsna knee","knee mri","knee","raptor knee","coatnet knee","dinov2 knee","convnext knee"} --sort-by updated`, pages 1-3. That gives 155 unique datasets, 53 updated since 09-25 (MEASURED, `jump_wf/ds_since0925.csv`). For each weights dataset I ran `kaggle kernels list --dataset` (all returned "Not found", so **no public notebook uses them and there is no public score**), `kaggle datasets files`, and looked up the owner's LB.

| dataset | contents (MEASURED) | owner LB | public evidence | adoption |
|---|---|---|---|---|
| **tbongc/rsna-knee-raptor-v8geo-v7w-f-1-s42-imagenet** (09-28) | `final_raptor_v8geo_v7w_f-1_s42_imagenet.pt`, 292.88 MB | .947, rank 176 | none | Same size class as Dread's `raptor_ft_coatnet_v8_full_swa.pt`/`v10_full.pt` (292.83 MB), so likely the same CoAtNet and a near drop-in for the parent's native384 (v8) arm. Needs a check of the checkpoint key format + 1 GPU commit + 1 LB slot. |
| **tbongc/rsna-knee-raptor-v10geo-v7w-f-1-s42-imagenet** (09-28) | `final_raptor_v10geo_..._imagenet.pt`, 292.88 MB | .947 | none | Same, for the native384dense (v10) arm. |
| **tbongc/rsna-knee-effb0mil-sl16-tv7w** (09-28, 53 downloads) | 4 folds `best_effb0mil_sl16_tv7w_f{0..3}.pt` (16.6 MB each) + `effb0mil_lib.py` + **per-fold OOF npy + ids csv** + result json (fold-0 `best` 0.9357 at ep 6; label set "tv7w" unknown, so not comparable to gold) | .947 | none | A different family (EffNet-B0 MIL, 16 slices). Its OOF files (about 0.3 MB total) allow a **cheap offline check** before any GPU work: Spearman vs Dread teacher on 4407, and AUC on any gold58 studies in its OOF. Integration = their lib + preprocessing in a new cell: medium work. |
| kawaharataishi/rsna-knee-coatnet-distilled-checkpoint (09-29) | `coatnet_384_full_cohort_swa.pt`, `coatnet_384_stage1_best.pt` (292.83 MB each) | .944, rank 304 | none | Raptor-size CoAtNet. Likely drop-in to a Raptor arm runner, but geometry and labels are undocumented. |
| royalacecat/raptor-knee-coatnet384-full-bs14 (09-30) | `raptor_ft_coat384_full_bs14.pt` | .943 | none | Raptor retrain. Low prior. |
| soyaoki/rsna-knee-my-models (09-28, 3.7 GB) | Synovitis-spatial Raptor 5-fold, PF OA slice-position gates, 4 x 89 MB fold models | .943 | none | Specialist heads. Owner gets no lift from them. Low. |
| apoorvadixit/rsna-knee-weights-v4cx / -finalcx | ConvNeXt-S in12k "latL" 5-fold (199 MB each) | .938 | none | Weak owner. Low. |
| kirilllabzin/knee-convnext-small-v17-source, radiodino-s16-v21, medsiglip448-v19 | "original source" backbone packages | .943 (2 subs) | none | Backbones, not knee-trained members. Skip. |
| fabotelli/pillar0-knee-finetune (Kaggle Model) | Pillar-0 BreastMRI full fine-tune | .944 | **author-reported gold58 macro .797** | Too weak, would drag. Skip. |
| gperdrizet/rsna-knee-abnormality-pseudolabeled-training-data | labels csv | n/a | used only in EDA notebooks | Training input only. |
| mattiaangeli/rsna-knee-cnx-m448-f0-public (older) | ConvNeXt M448 fold 0 | .950 | **we MEASURED it hurting**: 0.930 (full parent @50%), 0.935 (fast base @40%) | Dead. |
| dreaddevelopment/raptor-knee-finespacing v9 (older) | Raptor v9 | .945 | **we MEASURED 0.943** (v9 swapped into the v5-reverse arm). tony-rank-ensemble 0.942 | No-op. |
| dreaddevelopment/raptor-knee-widedense v4 (older) | Raptor v4 | .945 | single CoAtNet LB .924 (ours), dual-coatnet notebook .934 | Correlated Raptor family. Low. |

## 4. Ranked list of public assets not in our parent

| # | asset | evidence for public score | fork or integration | value now |
|---|---|---|---|---|
| 1 | tbongc effb0mil_sl16_tv7w (4 folds + lib + OOF) | No public score. Owner team .947 (MEASURED). Fold-0 internal `best` .9357 on unknown labels (MEASURED, Kaggle file viewer) | Integration (new inference cell from `effb0mil_lib.py`; preprocessing unknown). Can be **pre-screened for free on its OOF** | Highest expected value of the public assets, because it is a different architecture family with OOF to gate on (GUESS: if OOF Spearman vs teacher <= .85 and gold58 >= .90, worth one slot at W .15-.25) |
| 2 | tbongc raptor v8geo / v10geo full-data retrains | No public score. Owner .947 | Low integration: drop the checkpoint into the existing native384 / native384dense arm and prob-mean with Dread's v8/v10 | Small (GUESS +0 to +.002 public). The Raptor family is already in the parent, and our v9 swap was a measured no-op |
| 3 | kawaharataishi CoAtNet-384 full-cohort SWA | No public score. Owner .944 | Low to medium (geometry undocumented) | Small, same reasoning as #2 (GUESS) |
| 4 | DINOsaur V5 internal re-weight | 0.943 (MEASURED) | Zero-training fork | None (same public score as parent) |
| 5 | cnx-m448-f0, raptor v9, raptor v4, Pillar-0, convnext v4cx, soyaoki specialists, royalacecat retrain | Measured dead/no-op or weak owners | n/a | None |
| n/a | analyticaobscura V4 (.947), pj d4-blend (.946), d4-lite (.945), aastik s75 (.945) | MEASURED public scores | **Not adoptable**: private datasets | They prove that d4 + an independent fleet at W .45-.6 is worth +.002 to +.003 public (MEASURED from their scores vs the 0.943 base) |

## What this means for the >= 0.95 goal

- Nothing public gets us there. The best public notebook is .947, and its weights are private. The only public route above .943 is the same one we are already running: an **independent, decorrelated fleet blended at one global W**.
- The public evidence suggests our fleet is undersized. pj's fleet is 3 arms (S_dist, S_coat, S_cnxl) at W .45, and aastik's is ConvNeXt-T 5 folds x 2 seeds at W .5-.6. Ours is a **single fold-0 model**. Expect K2's blend lift to be capped below theirs until more folds or seeds exist (MODEL: averaging k weakly-correlated members raises member AUC, and both public cases used >= 3 to 10 models).
- Merge math: 0.945 is silver today (rank 254). Getting to .950 or more needs either a merge with a team that has its own strong private fleet, or several folds of ours.

## Recommended next steps (no Kaggle writes by me)

1. Read K2 (solo + blend) at ~17:50 IST before anything else. It is the only lever with direct public precedent.
2. **Cheap pre-screen of tbongc effb0mil.** Download only the 4 `oof_*.npy`, 4 `oof_*_ids.csv` and 4 `result_*.json` files (about 0.5 MB total, `kaggle datasets download tbongc/rsna-knee-effb0mil-sl16-tv7w -f <file>`). Then compute Spearman vs `SP/teacher_v2_out` and vs the fleet, plus AUC on gold58 if those studies are in its OOF. I did **not** download these files: downloading third-party dataset files needs Eesh's explicit OK.
3. Only if #2 passes: build an inference cell for effb0mil from `effb0mil_lib.py` (needs the 66.8 MB dataset attached) and spend one slot on parent + effb0mil at W .2.
4. Keep the tbongc Raptor retrains as a low-priority 1-slot test (add as extra checkpoints to the v8/v10 arms). Expected value is about 0.

## Files

- `jump_wf/lb/rsna-knee-abnormality-detection-publicleaderboard-2026-10-04T09_27_20.csv` (leaderboard)
- `jump_wf/lb_stats.py`, `jump_wf/lb_move.py` (shape + movers)
- `jump_wf/kl_{dateRun,scoreDescending,voteCount}_p{1,2,3}.csv` (kernel listings)
- `jump_wf/ds_all.csv`, `jump_wf/ds_since0925.csv` (dataset search)
- `jump_wf/pub/<owner>_<slug>/` (26 pulled notebooks, metadata, `code.py`, `diff_vs_*.txt`)
- `jump_wf/nb2py.py`, `jump_wf/celldiff.py`, `jump_wf/linediff.py` (diff tools)
