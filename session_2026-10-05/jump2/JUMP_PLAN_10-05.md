# JUMP PLAN 10-05 (RSNA Knee): how to get off the .943 tie

Written 2026-10-05 ~10:20 IST by Claude for Eesh. Inputs: the 4 research tracks and their adversarial verifications
(`jump2/topsolutions.md`, `labels.md`, `blendmath.md`, `pseudolabel.md` and `verify_*.md`), `SP/k4/K4_RUNBOOK.md`,
`SP/fast_build/SWITCH.md`, `HANDOFF.md` (09:45 entry), `merge_outreach.md`, and a read-only box check at 10:02.
Only claims that survived verification are used; refuted ones are listed in sec 6 so nobody reuses them.
Nothing was submitted, pushed, posted, killed or written outside the scratchpad.

Tags: **[M: source]** measured (LB read, log, file, computation today). **[MODEL]** arithmetic on measured inputs
(equal-variance binormal unless said; `jump2/plan_calc.py` for the new tables). Caution: this model over-predicted
once already (the 10-04 forecast of +.004 to +.009 for TTR v4 read 0 in K3). **[GUESS]** judgement.
**[CLAIMED]** forum self-report. SP = scratchpad, LB = public leaderboard at 3 decimals, times IST.

---

## 0. State at 10:05 IST

- **LB** [M LB]: parent .943. K2 SOLO (F1 mix50 fold 0) .932, K2 BLEND W .30 .943. K3 TTR v4 SOLO .923, K3 STRICT .943.
  Every blend so far reads .943.
- **Box** [M f1 log, nvidia-smi 10:02]:
  - Fold 1 TTR mix50 started epoch 6 of 24 at 10:02. Epoch 5 took 17.0 min because the GPU was shared.
  - The GPU now holds only the fold trainer (3.0 GB) and gnome-remote-desktop. The morning side-jobs (tta_bench
    09:04-09:27, speed_ab 09:33-~10:00) are gone.
  - Fold 1 ETA ~13:50 [MODEL: 18 epochs x 12.6 min alone].
- **TTA gate** [M `~/jump2/tta/run1.log`]:
  - TTR mix50 J1 with A+B (mirror + half-slice offset): +.0020 val (P 1.00), gold58 +.0015. PASS.
  - F1 mix50: +.0014 val, gold58 -.0014. FAIL, so no TTA for F1.
  - mean(J1, F1 mix50): val .8861 (+.0025 over J1 alone), gold58 .9188. With A+B on both: val .8875.
  - The submission worker does not serve TTA yet: pass B ("B6") is not built [M HANDOFF 09:45].
- **K4-i1** (J1 only) is built and box-tested: A0 BOXSTAGE PASS 25/25 on all 3 kernels, k2_check 28/28
  [M K4_RUNBOOK sec 6]. It is ready for Eesh's Kaggle steps now.
- **Fast trainer** (`--grad_ckpt 0`): weights and OOF are bitwise identical, step time x0.78, ~4.1 h per fold vs
  5.0 h [M SWITCH.md; fold time MODEL]. Not switched.
- **Kaggle folds 3+4** (`rsna-train-ttr-f34`): built and dry-run at 10:00 [M SP/kaggle_train]. Live status is
  UNCONFIRMED here.
- **Round 2** (`~/auto_round2.sh`): built, not started [M ps].
- **Slots**: 5/day, shared with SwaplY, reset 05:30 IST. The docs show none used for UTC day 10-05 [GUESS from HANDOFF].

## 1. Bottom line

**The single highest-EV path is a two-way merge with a team already at >= .946 that built its own models.** Eesh
pitches it with measured facts only. In parallel, our 5-fold fleet holds the floor: K4 today, K5 on 10-06. It also
produces the one number a partner will ask for, the fleet's SOLO score.

Our own models are not a route to .945 on their own:
- Every lever we can still pull adds at most about +.001 of true lift [MODEL].
- The label levers that remain do not transfer to the LB:
  - our ledger: Tian v09t -.001 [M ledger];
  - every one of our students scores below its own training target on gold58 [M pseudolabel];
  - no text reader we hold beats steven_v4 on gold58 [M labels].

**Why a merge:**
- On Kaggle a merged team shows its best member's score at once (merge_outreach.md). So a merge with a .947-.949 team
  is an immediate jump.
- Blending our parent with the partner's models:

  | partner fleet | r .85 | r .90 |
  |---|---|---|
  | .947 | .952 | .950 |
  | .949 | .953 | .951 |

  [MODEL plan_calc.py sec B; discount it, because the same model over-predicted K3.]
- **Gold (~.958) is not a realistic target by any route.** With a .949 partner it needs r <= .74 between their models
  and our parent [MODEL sec C]. Teams that all train on similar LLM labels are not that independent [GUESS].

**Honest probabilities** [GUESS, anchored on MODEL where noted]:

| outcome | own models only, by 10-15 | own only, by 10-22 | with merge outreach, by 10-15 | with merge outreach, by 10-22 |
|---|---|---|---|---|
| display >= .944 | .45-.55 | .50-.60 | .55-.65 | .60-.70 |
| display >= .945 | .05-.12 | .07-.15 | .25-.35 | .30-.40 |
| display >= .948 | <= .02 | <= .03 | .10-.18 | .12-.20 |
| gold (~.958) | ~0 | ~0 | < .02 | < .02 |

How these were built:
- **Own-only rows:**
  - K4 at W .20: P(.944) .33-.39, P(>= .945) .02-.03 [MODEL fleetsim_k4];
  - plus TTA and M1 strength in K5 [GUESS];
  - minus the verifier's downside, a member of ~.929 rather than .9325 [MODEL].
- **Merge rows:**
  - P(at least one two-way merge with a team at >= .945 agreed by 10-13) is ~.2-.3 [GUESS; merge_outreach put it at
    .15-.25 per target];
  - >= .948 needs a partner already at >= .948, or a post-merge blend submitted before the deadline.

**What the verifications settled** (do not re-open without new evidence):
- **Report re-extraction is closed** [M labels]:
  - no text reader we hold beats steven_v4 (.8927 on gold58);
  - the best second reader adds +.0042 (P .83).
- **An own-OOF pseudo-label round is filler** [M pseudolabel]:
  - students score -.012 to -.032 on gold58 below the targets they were trained on;
  - the round's target ties mix50 on gold58 (.9267 vs .9262);
  - two of our ledger runs read null on the LB.
- **"Top teams have better image models, not better labels" is refuted as stated** [M verify_topsolutions]:
  - the gold58 gaps are under 1 SE;
  - the LB moves ~.25x as much as gold58.
- **Training on pure steven_v4 is the only measured decorrelation lever:**
  - parent-likeness (OOF Spearman vs the parent's Raptor arm, fold-0 val) .840-.850, vs .889-.896 for mix50 members
    [M oofx];
  - but those members are ~.009 weaker on the LB (.923 vs .932) [M K2/K3].

## 2. Ranked actions

**At a glance:**

| # | action | expected public gain | ready by | what kills it |
|---|---|---|---|---|
| 1 | Merge outreach (Eesh posts) | +.002 to +.006 to the shown score; merged blend .950-.953 | post 10-05/06, agree by 10-13 | rules or cap block it, or no reply |
| 2 | K5: 5-fold TTR mix50 + TTA + M1 | true +.0004 to +.001; P(>= .945) .05-.12 | submit 10-06 | B6 parity fails; K5 SOLO <= .932 |
| 3 | K4-i1 today | E[display] .9433, P(.944) .33 | submit ~11:00, read ~17:00 | K2CHECK fails; W .20 reads <= .942 |
| 4 | Pure-v4 decorrelation set (F2, or TTR v4 fallback) | F2 blend +.000 to +.003; fallback +.000 to +.0005 | probe ~01:00 10-06; set ~10-07; LB 10-08 | preflight or probe gate fails; SOLO < .925 |
| 5 | Second seed of the best-reading set | +.000 to +.001 | 10-10 to 10-12 | its own set never read >= .944 |
| 6 | Round-2 own-OOF probe (filler) | -.0002 to +.0004 true | idle GPU after 10-06 | its own GATE.md bars |
| 7 | KX local-LLM closure pilot | ~0 | any fold boundary | no explicit OK from Eesh |

Costs are in each action's detail below.

**1. Merge outreach (Eesh posts; see sec 5)**
- **Gain:**
  - shown score +.002 to +.006 at once if the partner is at .945-.949 [M their LB minus .943];
  - merged blend .950-.953 [MODEL, r .85-.90].
- **Cost:** 0 GPU; slots are shared after the merge; Eesh ~1-2 h plus follow-ups.
- **Ready by:** post 10-05/06; follow up with the K5 SOLO on 10-06/07; agree by 10-13. The merge deadline is 10-15
  23:59 UTC.
- **Kill gate:**
  - a partner asks for files or predictions before the merge: decline (rules line 144);
  - combined submissions exceed 5 x days running at the request date: that target is out;
  - no reply by 10-10: go to the next target.

**2. K5 = 5-fold TTR mix50 with TTA A+B, plus M1 (F1 mix50, no TTA), STRICT**
- **Gain:**
  - true lift +.0004 to +.001; E[display] .9435-.944; P(.944) .40-.50; P(>= .945) .05-.12 [MODEL at fleet solo
    .936-.940, r ~.92; how much TTA and M1 add is GUESS];
  - the SOLO read is the number for the merge pitch.
- **Cost:**
  - no extra GPU (the chain is running);
  - ~2-4 h of CPU work for the B6 build and parity test;
  - 2-3 slots; Eesh ~1.5 h.
- **Ready by:** submit 10-06 ~08:00-14:00. The 5-fold set lands ~22:00 10-05 via Kaggle, or ~02:10-04:50 10-06 on the
  box.
- **Kill gate:**
  - B6 parity fails: ship K4-final + M1 without TTA;
  - K5 SOLO <= .932: the fleet is no stronger than M1 alone, so keep W <= .15.

**3. K4-i1 today: SOLO (J1) + main W .20 (+ optional W .50)**
- **Gain:**
  - E[display] .9433, P(.944) .33, P(>= .945) .02 [MODEL K4 runbook];
  - the SOLO read pins the TTR mix50 member's strength (sd .003 -> .0004).
- **Cost:** 0 GPU; 2-3 slots; Eesh ~1 h.
- **Ready by:** submit ~11:00; reads ~17:00, in time to choose the GPU job after fold 2.
- **Kill gate:**
  - any K2CHECK FAIL: do not submit;
  - W .20 reads <= .942: mix50 TTR members hurt at W .20, so K5 goes to W .10.

**4. Pure-v4 decorrelation set**
- **What:** F2 (EfficientNetV2-S with frozen BN; ResNet50.a1 if EV2S fails the fp16 preflight), fold-0 probe, then
  folds 1-4. If the probe gate fails, fall back to TTR v4 folds 1-4 (known recipe, gc0).
- **Gain:**
  - F2: blend +.000 to +.003 [GUESS; nothing measured: TTR v4 alone at W .25 read 0 in K3];
  - TTR v4 fallback: +.000 to +.0005 [MODEL blendmath].
- **Cost:**
  - preflight a few minutes; probe ~5-6 h [GUESS: F2 has never been timed];
  - 16-24 h for 4 more folds;
  - 2 slots; Eesh ~1 h.
- **Ready by:** probe read ~01:00 10-06 if the box is free at ~18:50; the set ~10-07 evening; LB 10-08.
- **Kill gate:**
  - the F2 preflight fails;
  - probe gate F2-G (sec 3) fails;
  - the set's SOLO is < .925: never blend it.

**5. Second seed of whichever set read best** (aastik pattern: 5 folds x 2 seeds)
- **Gain:** +.000 to +.001 [MODEL: members of the same family move along the same iso-lift curve].
- **Cost:** ~20 h GPU; 1-2 slots.
- **Ready by:** 10-10 to 10-12.
- **Kill gate:** only if that set read >= .944 in a blend. Never add mix50 seeds (blendmath: ~zero value).

**6. Round-2 own-OOF probe** (`~/auto_round2.sh`, a50 or a50s75)
- **Gain:** -.0002 to +.0004 true, i.e. one more HOLD-type member [MODEL pseudolabel].
- **Cost:** 5 h GPU; 0-1 slots.
- **Ready by:** only in idle GPU time after 10-06.
- **Kill gate:**
  - its own GATE.md bars (a)-(c); predicted P(pass) ~.2 [GUESS, GATE.md];
  - it gives up by design at 10-07 12:00.

**7. KX local-LLM closure pilot** (Qwen2.5-7B, gold74 only)
- **Gain:** ~0 [GUESS]. Its only value is closing the lever with our own data.
- **Cost:** 0.25-0.5 GPU h at a fold boundary under PAUSE_CHAIN; 0 slots; ~30 min review.
- **Ready by:** any fold boundary.
- **Kill gate:**
  - it needs Eesh's explicit OK;
  - its result is not clean evidence: the prompt was written after reading 18 gold58 error reports [M verify_labels].

**Cheap CPU item at the fold-1 boundary (~13:50):**
- Run `gold_infer.py` on the fold-1 checkpoint (`NTHR=4`, nice 19, `CUDA_VISIBLE_DEVICES=`), then `g74pairs.py`.
- Convert with r_mm ~ .924 + .663 (sp - .905) and read it as +-.015 [M calibration; MODEL map].
- This replaces the guessed fold-to-fold r_mm. It does not change K4 or K5.

**Skip list** (saves GPU hours and 5-10 slots):
- re-extracting all 4,407 reports with a local LLM;
- tuning any weight, rule or prompt on gold58 (n 58, per-member SE .014-.017 [M]);
- the DSD label plan (fails its own F0 kill test [M pl_sim_b5]);
- TTR v4 folds as a *self-teacher*;
- more mix50 seeds beyond the 5 folds;
- higher resolution, a bigger backbone, SWA, pooling swaps;
- external data (default no; Eesh's call);
- API-based labelling (no API keys);
- the Qwen 2B vision-model fine-tune (P < .10 [GUESS], GPU booked);
- W above .30 without a measured low r;
- 9-member mixes, unless a SOLO shows they beat the 5-fold set.

## 3. Box GPU queue to 10-15

**Rules for the queue:**
- **Keep the TTR mix50 chain.** Do not pause it for a labels pilot (that lever is closed). Do not reorder it for
  round 2 (round 2 needs all 5 folds anyway, and it is filler).
- **No GPU co-tenants during a fold.**
  - This morning's side-jobs cost fold 1 ~16+ min [M verify_pseudolabel 2.1].
  - `auto_folds.sh` waits for any GPU python at a fold boundary, so a side-job there stalls the chain
    [M verify_blendmath].
- **New TTR launches use `~/fast_build/code2/fleet_train.py --grad_ckpt 0`** (bitwise-identical math, x0.78 step time
  [M SWITCH]). F2 runs use gc1 unless the preflight's memory check shows headroom: the gc0 trainer peaks at ~9.1 GB
  [M SWITCH; F2 is GUESS].
- **Operator only (Eesh):** placeholder dirs, the gc0 switch, launches and any PAUSE_CHAIN use. Agents may not write
  `~/fleet_runs`, use the box GPU or signal processes.

| window [MODEL] | GPU job | decision / gate |
|---|---|---|
| now to ~13:50 10-05 | fold 1 TTR mix50 (old chain, gc1) | keep; nothing else on the GPU |
| ~13:50 boundary | none (CPU only: gold_infer fold 1, g74pairs) | - |
| decide by ~13:30 | Are Kaggle folds 3+4 live? | **YES** (running, first epochs on pace for the recipe or a documented shorter schedule inside the 11.4 h watchdog): before fold 2 ends, Eesh creates `~/fleet_runs/f{3,4}_s1043_ttr_mix50` placeholders. The old chain then runs fold 2 on gc1 and exits; no process is signalled. **NO**: the box does folds 2-4; Eesh does the gc0 switch per SWITCH.md steps 0-4 at the fold-1/2 boundary (saves ~2.7 h) |
| ~13:50 to ~18:50 (gc1) | fold 2 | - |
| ~18:50 | F2 preflight (`~/fleet2_code/f2_preflight.py`, a few minutes; EV2S has a known fp16 overflow risk) | PASS: EV2S. FAIL: R50.a1 with ImageNet init (the parent already has RadImageNet R50 heads) |
| ~19:00 to ~01:00 10-06 | F2 fold-0 probe on pure steven_v4 (`--f2 1`, F1 recipe, gold74 excluded) | gate F2-G below |
| 10-06 to ~10-07 evening | F2-G PASS: F2 folds 1-4. FAIL: TTR v4 folds 1-4 (gc0, ~16.4 h) | read the set's SOLO and blend on 10-08 |
| 10-07 evening to 10-10 | if the v4 set's blend read >= .944: its second seed. Otherwise the round-2 probe as filler, or idle | merge outcome first |
| 10-10 to 10-15 | if merged: the partner's recipes, folds or seeds (largest marginal value). Otherwise a second seed, or idle | - |
| fallback (no Kaggle folds) | folds 2-4 on gc0 end ~18:00, ~22:05 and ~02:10 10-06; F2 probe ~02:15 to ~08:00 10-06 | same gates, ~7 h later |

**Gate F2-G** (pre-registered; compared with TTR v4 fold 0 on the same labels; scored with oofx-style CPU scripts on
the 867 fold-0 val studies). PASS needs all four:
- (a) OOF AUC vs steven_v4 >= .8714, i.e. TTR v4 .8764 - .005 [M].
- (b) gold58 >= .868, i.e. .8780 - .010, about one seed SD [M Tian twins .0098].
- (c) Spearman vs the Raptor proxy <= .855, i.e. TTR v4 .850 + .005 [M].
- (d) Spearman vs the TTR v4 OOF <= .90. Our two own v4 models agree at .917 [M verify_labels].

**Kaggle folds 3-4 gate:**
- Each fold's OOF AUC vs v4 must be >= .8786 (J1 .8836 - .005). Otherwise ship a 3-fold set.
- Their outputs must land in the box run format before `k4_pick.py` or `auto_round2.sh` can use them. Copying them is
  Eesh's step.

**PAUSE_CHAIN** is used only for:
- the gc0 switch (SWITCH.md step 0);
- an optional KX pilot at a fold boundary, if Eesh OKs it.

## 4. Submission plan

### 10-05 (UTC day ends 05:30 IST 10-06)

| slot | kernel | what it measures | pre-registered read |
|---|---|---|---|
| 1 | K4-i1 SOLO (`rsna-sub-k4-solo` v1, J1 only) | single-member LB strength of TTR mix50 | **<= .929**: weaker than F1 (.932); K5 keeps M1, W .15. **.930-.934**: as modelled (posterior .929/.933/.937 [MODEL]). **>= .935**: TTR is our strongest recipe; K5 W .25, and TTR is the v4 fallback recipe |
| 2 | K4-i1 main W .20 STRICT (`rsna-sub-k4-fleet` v1) | best expected score today | **.944**: off the tie, true lift ~+.0009. **.943**: ~+.0001 (a re-measure of K2/K3). **<= .942**: mix50 TTR hurts at W .20; K5 at W .10 |
| 3 (optional) | K4-i1 W .50 (`rsna-sub-k4-w50` v1) | J1's correlation r with the parent | look it up in the table below with the SOLO read |
| 4-5 | leave for SwaplY | - | - |

Optional: if the 5-fold set and B6 are final before ~03:00 IST, slots 4-5 can carry K5 SOLO + main inside this UTC
day. Only if Eesh is up for it.

W .50 lookup [MODEL plan_calc sec D]. Each cell is the expected display; W* is the best weight for that member:

| J1 SOLO | r .88 | r .91 | r .94 |
|---|---|---|---|
| .931 | .943 (W* .25) | .941-.942 (W* .16) | .940 (W* .00) |
| .933 | .944 (W* .29) | .942 (W* .22) | .941 (W* .07) |
| .935 | .945 (W* .33) | .943 (W* .27) | .942 (W* .15) |
| .937 | .946 (W* .37) | .944 (W* .33) | .943 (W* .24) |

Use the nearest r to set K5's W cap: r ~.88 allows up to .30; r ~.91 keeps .20-.25; r ~.94 means <= .15.

### 10-06

**Build K5** when the folds are final: `k4_auto.sh`, then `build_k4.py --with_m1`, plus TTA B6 if all its gates pass.

B6 gates (pre-registered):
1. Worker TTA output equals the tta_bench A+B output on 16 val studies (max |prob diff| <= 1e-3).
2. Box A0 BOXSTAGE PASS with TTA on.
3. Kaggle commit K2CHECK PASS, with the fleet stage <= 60 min. MODEL: ~20 min GPU per T4 for 5 TTR members x 2
   passes + M1 (K4 runbook timing x2 for TTR), on a 8.6 h wall with the parent at ~6 h.

Any B6 failure means K4-final + M1 without TTA.

| slot | kernel | pre-registered read |
|---|---|---|
| 1 | K5 SOLO | **<= .932**: the 6-member fleet is no stronger than M1; never use W > .15. **.934-.937**: as modelled. **>= .938**: strong; next main at W .30 if r <= .92 |
| 2 | K5 main at the W set by the 10-05 reads (default .20) | .944 is the expected good case; .945 has P ~.05-.12 [MODEL/GUESS] |
| 3 (optional) | K5 at W .30 if the 10-05 reads gave r <= .91; otherwise K5 at W .50 to measure r | as in the lookup above |
| 4-5 | SwaplY / spare | - |

**W rule** after the K5 SOLO read s (r ~.92 from K2 [MODEL]):
- s < .936: W .20;
- .936 <= s < .938: W .25;
- s >= .938 and the reads put r <= .92: W .30.

The difference between these Ws is <= .0005 true [MODEL sec E], so never fish for W with slots.

### 10-07
- One slot for the F2 (or TTR v4) fold-0 probe SOLO, only if F2-G passed. It measures the new family's LB strength,
  the same way the K3 TTR solo did.
- Resubmit K5 only if the 10-06 reads moved the W rule.
- Leave the other slots: display rounding hides gains under .0005.

### 10-08
- v4 set SOLO.
- Parent + mean(mix50 set, v4 set, M1) at W .20-.30.
- Per-set weights would need a kit change plus the K2CHECK gates. Start with one mean.

### Final picks (10-22)
- Pick 1: the best-reading blend with W <= .30.
- Pick 2: a lower-W hedge, because the public line barely predicts private medals (24% either side
  [CLAIMED Mamarin, as verified]).
- If merged: the merged blend plus one of the above.

**Eesh time:**
- today: ~1 h for K4-i1, ~1 h for merge posts, ~10 min for the 13:30 decision;
- 10-06: ~1.5 h for K5;
- 10-07 and 10-08: ~1 h each.

## 5. What to tell merge partners (facts only; Eesh posts; nothing private before the merge)

**Facts we can state publicly:**
- Team "Eesh saxena", public .943 [M].
- **Submission count:** check the team page; it was 34 on 10-03, plus K2 x2 and K3 x2, plus today's K4 [M].
  - The cap is 5 x days running at the request date, about 340 on 10-05 [MODEL: ~330 on 10-03 + 5/day].
  - A two-way merge fits; a three-way merge does not.
- **Our models:** ConvNeXt-T, 2.5D slice triplets, 4 MRI views x 24 triplets, 140 mm crop, slices sorted by
  ImagePositionPatient, slice/view attention pooling, per-target heads.
- **Single-fold solo LB:** F1 mix50 .932, TTR v4 .923 [M]. TTR mix50 J1 solo comes today; the 5-fold + TTA fleet
  SOLO on 10-06/07.
- **Out-of-sample gold58** (the 74 gold studies are excluded from all our training): F1 mix50 .914, TTR mix50 .913,
  mean of the two .919 [M].
- **Labels:** our mix50 target (0.5 public LLM labels + 0.5 logit-matched public CoAtNet teacher) scores .926 on gold58,
  vs .893 for the public steven_v4 LLM labels [M].
  - Say plainly that the teacher's checkpoint was chosen on those same 58 studies, so .926 is optimistic.
- **Compute:** a dedicated V100 16 GB (one TTR fold takes 4.1-5 h) plus Kaggle GPU quota.
- **Infra:** a 9 h-safe 2xT4 build of the public .943 parent, plus a fleet stage of ~25-40 min. It runs in strict
  mode (an error instead of a silent fallback) and every commit is gate-checked.
- **If asked:** our members are parent-correlated. Parent + member reads .943 at W .20-.30 [M K2/K3]. Their value to a
  partner is as an independent own-model leg, and that correlation can only be measured after a merge.

**Do NOT say:**
- "Independent GPT labels on all 4,349 studies." The GPT-5.6 table we hold scores .849 on gold58, below v4 [M]. Drop
  it from the `merge_outreach.md` drafts.
- ".927 on the 58." Use .926 with the optimism caveat.
- "+0.009 solo LB for another team." That was a solo gain; inside a .942 stack it read .941-.942 [M ledger].
- Any promise of gold, or any binormal-model number presented as if it were measured.
- That SpeedSci's .947 is own-models only. It includes a public CoAtNet at .928 [CLAIMED 744511, verified].

**Targets** (Eesh checks the live LB first):

| priority | team | score | thread | why them |
|---|---|---|---|---|
| 1 | caoyupeng / SpeedSci team (Myo Min Htet) | .947-.949 [CLAIMED] | 744511 | said "we need most is a good label" [CLAIMED] |
| 2 | mehmetbarankrat / ringbearer | .948 [CLAIMED] | 744511 | - |
| 3 | prateekgroverin | .945 | 743374 | asked for partners strong on labels or with a single model above .940 [CLAIMED] |
| secondary | aastik, pjmathematician | .945-.946 (parent + private fleet) | - | our parent adds nothing to them; our fleet adds a second private leg, +.001-.002 at most [GUESS] |

**Mechanics:**
- No files, predictions or weights before the merge (rules line 144). Compare only public scores and approach text.
- Blend checks happen after merging.
- Timeline: post 10-05/06; follow up with the K5 SOLO on 10-06/07; agree by 10-13; the deadline is 10-15 23:59 UTC.

## 6. Flags and do-not-use list

- **Rule breach by another workflow:** GPU side-jobs on the box (tta_bench, speed_ab, ~09:04 to ~10:00) broke the
  no-GPU rule for agents and cost fold 1 ~16+ min [M]. They ended by 10:02. Any further box GPU work should be Eesh's,
  at a fold boundary.
- **`auto_round2.sh` waits for f3/f4 closing logs on the box.** If folds 3-4 come from Kaggle and are not copied in,
  it never launches. That is acceptable, since round 2 is filler.
- **`build_pl_tm5.py` accepts a mid-training OOF file.** Do not use it. If it is ever needed, port the r2 builder's
  closing-line guard first.
- **mix50, gated75 and mixSA gold58 scores are optimistic:** the teacher was epoch-selected on gold58 [M AUDIT 10-04].
- **The Raptor proxy is in-sample on the 867 fold-0 val studies.** Use its parent-likeness numbers for ordering only.
- **Refuted claims, do not reuse:**
  - "the gold58 gap matches the LB gap";
  - "mix50 in-sample is a new finding";
  - "SpeedSci label set .945 -> .910" (it was an EffNet model's score);
  - "Kaggle GPU not enough" attributed to NguyenThanhNhan (it was Ace);
  - "a merge reaches gold";
  - "own teacher .9340 beats mixSA" (best of 18 variants; +.0040, P .85);
  - "the 7B LLM lost only to binarization";
  - "labels cannot buy decorrelation";
  - the KX gate as pre-registered;
  - KX at 1-1.5 GPU-h (it is 1.3-2.2);
  - chain ETA ~03:30 (it was ~04:45-05:15 for 4 box folds on gc1);
  - "Stanton 2021: students track teachers" (the paper says the opposite);
  - "Archit's precondition fails" (true overall; in report-silent cells our v4 students beat v4, .70 vs .57).

**Sources:** `SP/jump2/{topsolutions,labels,blendmath,pseudolabel}.md`, `SP/jump2/verify_*.md`, `SP/jump2/plan_calc.py`
(new MODEL tables B-E), `SP/k4/K4_RUNBOOK.md` and `analysis/fleetsim_k4.out`, `SP/fast_build/SWITCH.md`,
`~/jump2/tta/run1.log`, `~/jump2/r2/GATE.md`, `~/fleet_runs/f1_s1043_ttr_mix50.log`,
`Downloads/rsna-knee/{HANDOFF,NEXT_SESSION,merge_outreach}.md`.
