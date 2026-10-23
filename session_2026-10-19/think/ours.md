# ours.md: lens "our own unused assets" (10-19, written ~17:30 IST)

Research only. Nothing built, pushed, committed, submitted or uploaded; no GPU job, no box write (read-only ssh `ls` / `cat` /
`tar cf -` of small prediction CSVs into `think/ours_work/boxpreds/`; these are predictions on RSNA studies, no OAI data).
CPU numpy only, each script < 3 min. Tags: MEASURED (computed or read here, file named), CLAIMED (someone's text), MODEL
(inference from measured numbers), GUESS.

Scripts and outputs (all in `session_2026-10-19/think/ours_work/`):
- `ours_g58.py` / `ours_g58.json`, `r7mix_g58.py` / `.json`, `nested_member.py`, `wrongbase.py` (from the interrupted run; the last
  two re-run here, output quoted below).
- NEW `ours_more_g58.py` -> `ours_more_g58.json` / `.log`: every member with gold58 preds, LB calibration, small weights,
  structural subsets, replacements, split-half.
- NEW `ours_small_sh.py` -> `ours_small_sh.log`: split-half for the .05 additions.
- NEW `ours_oof.py` -> `ours_oof.json` / `.log`: large-n OOF (report labels) per label, and OOF vs gold58 agreement.
- NEW `kc_vs_full_oof.py` -> `kc_vs_full_oof.log`: knee-centred vs full-frame on the same 867 fold-0 studies.

## 0. Verdict

- **None of our own models is worth a slot or a share of the spare runtime, at any weight, on any label subset, or as a
  replacement for r224 / S6fall.**
  - Every self-trained member we put on the LB read negative: 0 of 6, and two of the six dropped a full display step (table in
    section 2).
  - If those members were truly neutral, each non-move would be a coin flip, and 0 of 6 would have probability ~1/64. So they
    are actively harmful at .2-.3 (MODEL), and a .05-.1 share is expected to be slightly negative too, not zero.
  - Gold58 shows no member addition that survives a split-half check (best sign agreement .61, most .40-.58).
  - The large-n OOF does not support the "structurally strong labels" story: the knee-centred crop is not better than a
    full-frame model on ACL or menisci (section 4).
- **No model we could train before 10-20 night is plausibly additive.**
  - Feasible on paper: one all-data member by ~23:00 tonight, or 5 folds by ~12:40 10-20 (past our own 12:00 freeze); either
    reads 10-20 evening or 10-21 early.
  - Expected value ~0 or below:
    - every recipe we can run (ConvNeXt, Swin-T, ResNet50-RadImageNet, KC crop, our Raptor port, OAI-pretrained) has already
      landed at gold58 .877-.924, against the c4 members' .923-.937;
    - every one that reached the LB read negative.
- **Mechanism (MODEL, evidence in section 7): label source plus a recipe gap, not architecture.**
  - Our members beat R7 by .010-.020 macro AUC on report-label OOF, yet tie or lose on radiologist gold58.
  - They are tuned to the LLM report-label definition; the public members are tuned to labels closer to the hidden ones.
  - A new architecture trained on the same labels does not fix that.
- **Recommendation:** spend no slot and no runtime on our assets. Keep picks A = c4 57343435, B = R6 57248084. Leave the box idle
  or give it to another lens's measurement job.

## 1. Inventory: our unused assets, gold58 and OOF

Gold58 = 58 radiologist-labelled studies, never in training. c4 itself scores .9410 (MEASURED `ours_more_g58.log`).
Correlation = mean per-label Spearman.

| asset | what | gold58 macro | Spearman with c4 | Spearman with R6 | source |
|---|---|---|---|---|---|
| r384 (in c4) | nartaa CoAtNet 384 | .9228 | .956 | .872 | MEASURED |
| r224 (in c4) | nartaa CoAtNet 224 | .9150 | .935 | .856 | MEASURED |
| R6 (in c4) | heliosli ConvNeXt-S, 4 folds, flip | .9366 | .959 | 1 | MEASURED |
| S6fall (in c4) | nartaa compact | .9314 | .936 | .879 | MEASURED |
| KC1 | ours, knee-centred ConvNeXt-S, s2001 f0 + s2003 all-data | .9237 | .909 | .869 | MEASURED |
| KC-all | 6 KC members | .9248 | .907 | .866 | MEASURED |
| K14 DT | ours, ConvNeXt fleet, OAI T123 pretrain, 5 folds | .9235 | .918 | .870 | MEASURED |
| fleet D | ours, ConvNeXt fleet, RSNA only, 5 folds | .9192 | .920 | .872 | MEASURED |
| T_b | ours, R7-teacher student, 5 folds | .9191 | .934 | .898 | MEASURED |
| s43 / s44 | ours, S6 recipe seeds | .9210 / .9149 | .930 / .920 | .883 / .868 | MEASURED |
| soup3 | weight soup of 3 S6 seeds | .8871 | .894 | .838 | MEASURED |
| R7 OOF | heliosli R7 (R6 sibling) | .9193 | .917 | .940 | MEASURED |

Not on gold58 here:
- T_a (s1048): OOF only.
- Our RSNA-only Raptor seeds s43-s46: serve gold58 .911-.914 (MEASURED `session_2026-10-13/raptor/RAPTOR_RENTED_RESULTS.md`).
- Our OAI Raptor S42: .9069, gate FAIL "port-broken" (MEASURED PROGRESS_14 10:45).
- Swin-T fold 0: gold58 .9182 at ep16, 25.1 min/epoch (MEASURED box `fleet_runs/f0_s1043_ttr_frac75oaiT123swinT.log`).
- ResNet50-RadImageNet fold 0: gold58 .8767 at ep7, killed (MEASURED box log).

Per label, c4 is weakest on Lateral OA (.843, 11 positives), Synovitis (.840, 27) and PF OA (.901, 21) (MEASURED).
- Lateral OA: KC1 (.874) and R7 (.878) beat c4. Only 11 positives, so one swapped pair moves AUC by ~.009 (MODEL).
- Synovitis and PF OA: every one of our members is below c4 (Synovitis .751-.811, PF OA .835-.878).

## 2. What the LB already rules out (our own members, rule R1)

| our member | entry | change | public read | source |
|---|---|---|---|---|
| K14 R fleet | HF-R20 57106365 | .2 on H | .954 = base, non-move | MEASURED PROGRESS_12 l.100 |
| K14 F | V224-FF20 57199082 | .2 on FLIP | .956, non-move = negative | MEASURED PREREG_10-15 READ 2 |
| our RSNA-only Raptor seeds (2) | HFR7-RS 57154866 | .30 of the Raptor slot | .955 < base .956 (display step DOWN) | MEASURED PROGRESS_14 l.12-13 |
| T_b student (5 folds) | 57222884 | .2 on S6F | .957, non-move = negative | MEASURED PREREG_10-15 l.283 |
| KC1 | combo 57250689 | .2 rank at the end, on the R6 entry | .957 < R6 .958 (display step DOWN) | MEASURED PROGRESS_16 l.105 |
| R7 (R6 sibling) | R7 late 57291336 | prob-mean .5 with R6, on HW | .958, non-move (practically negative) | MEASURED PREREG A14 |

- Correction to the brief: s43 / s44 were never on the LB. They were closed on gold58 (.921 / .915 vs nartaa .931; soups .905 /
  .887) with "no entry path" (MEASURED HANDOFF 10-17 sec 3). My numbers reproduce them: .9210 / .9149, soup3 .8871.
- Public checkpoints we added, by contrast, read positive 3 of 3: r224 (V224 lift), S6fall (S6F .957), R6 swap (.958) (MEASURED
  HANDOFF / PREREG).
- Base-rate reading (MODEL):
  - Under "our member = 0 effect", each non-move is a coin flip (paired public sd 3-7e-4, metric.md).
  - So 0 of 6 has p ~ 1/64, and two of the six fell a full display step.
  - Our members are negative at w .2-.3. A best guess for the size is -3e-4 to -1e-3 at .2. Near 0 the effect scales about
    linearly in w, which gives roughly -1e-4 to -2.5e-4 at .05.

## 3. Small weights (.05 / .10) on c4, gold58 (MEASURED `ours_more_g58.log`, `ours_small_sh.log`)

Paired bootstrap (2,000 resamples) of the c4 plus member gain; split-half = sign agreement between two disjoint 29-study halves,
300 splits.

| config (appended at the end, all labels) | gold58 gain | sd | P>0 | split-half sign agree | held-out gain if half A says + |
|---|---|---|---|---|---|
| R7 OOF .05 | +.00039 | .00040 | .86 | .40 | +.00000 |
| KC1 .05 | +.00032 | .00049 | .76 | .40 | +.00005 |
| KC1 + DT + T_b .05 | +.00027 | .00030 | .82 | .44 | +.00010 |
| K14 DT .05 | +.00017 | .00034 | .71 | .40 | +.00012 |
| KC1 .10 | +.00033 | .00089 | .65 | .61 | +.00041 |
| KC-all .10 | +.00047 | .00099 | .70 | .57 | +.00045 |
| fleet D .05 / .10 | -.00019 / -.00033 | | .29 / .34 | | |
| T_b .05 / .10 | -.00035 / -.00086 | | .22 / .14 | | |
| s44 .05 / .10 | -.00045 / -.00091 | | .10 / .14 | | |
| soup3 .05 / .10 | -.00084 / -.00190 | | .02 / .04 | | |

- All the gains are inside one sd, picked from ~50 configs, and none has a stable sign out of sample (sign agreement .40-.61).
- Held-out gains are 0 to +4e-4: below the public paired sd (3-7e-4) and far below the .959 step (+3.4e-4 needed from the
  unrounded MODEL ~.9587).
- A .05 member moves the final ranks by only ~.13-.17 % (MEASURED `ours_g58.json` rank_move).
- On ~910 private studies such an entry is a coin flip on the public board and ~0 +- 1e-4 on private (MODEL: gold58 sd scaled by
  sqrt(58/910)).
- Calibration: gold58 already mis-called PF OA .1 at P .925 (brief), so these P values carry little weight.
- The nested per-label pick test agrees (MEASURED re-run of `nested_member.py`). Labels picked on half A and scored on half B
  give: s43 .1 end -.00003; s43 .2 end -.00031; R7 .2 ours-slot +.00011; R7 .1 end +.00003; R7 .2 end -.00004 (all with P>0
  <= .45).

## 4. "Only on labels where they are structurally strong": not supported

The hypothesis:
- the knee-centred KC crop (sagittal and coronal soft-tissue view) should help ACL, MCL and the menisci;
- OAI-pretrained K14 should help the OA grades.

Gold58 (MEASURED `ours_more_g58.log`):

| config | gain | sd | P>0 | split-half agree |
|---|---|---|---|---|
| KC1 .20 on {ACL, MCL, MM, LM} | +.00089 | .00107 | .79 | .41 |
| KC1 .05 / .10 on the same labels | +.00021 / +.00031 | | .70 / .70 | |
| KC-all .20 on the same labels | +.00064 | .00094 | .76 | .42 |
| K14 DT .10 / .20 on {Medial, Lateral, PF OA} | -.00027 / -.00096 | | .16 / .22 | |

The OA hypothesis is simply negative. The soft-tissue numbers are positive but have the worst sign stability in the table.

Large n (OOF, LLM report labels, non-gold train studies; MEASURED `ours_oof.log`, `kc_vs_full_oof.log`).
- Same 867 fold-0 studies, AUC of KC vs the full-frame student T_b:

  | label | KC | T_b | KC - T_b (sd) |
  |---|---|---|---|
  | ACL | .848 | .864 | -.016 (.008) |
  | MCL | .856 | .857 | -.001 (.008) |
  | Medial Meniscus | .895 | .900 | -.005 (.004) |
  | Lateral Meniscus | .911 | .905 | +.005 (.008) |

  - The knee-centred crop is not structurally stronger on soft tissue. It is weaker on 9 of 12 labels.
- Added at .2 on a 2-model base (R7 + T_b), KC gives +.0006 to +.0034 per label, about the same as K14 DT (-.0023 to +.0049).
  It carries no soft-tissue-specific signal.
- Per-label gains on OOF vs on gold58, same member, same R7 base (Spearman over 12 labels):

  | member | Spearman |
  |---|---|
  | KC | -.13 |
  | T_b | +.13 |
  | K14 DT | +.38 |
  | fleet D | +.40 |

  The per-label pattern does not replicate across label sources, so a per-label member weight cannot be chosen from either one.

## 5. Replacing a component (MEASURED `ours_more_g58.log`)

| replacement on c4 | gold58 gain | P>0 | note |
|---|---|---|---|
| KC-all replaces r224 (A .25) | +.00051 | .62 | split-half .58, held-out +.00086; picked as best of ~50 |
| KC1 replaces r224 | +.00044 | .59 | split-half .47 |
| K14 DT / fleet D / T_b / s44 replace r224 | -.00034 / -.00037 / -.00129 / -.00135 | .19-.42 | |
| r224 dropped (A 0) | -.00038 | .38 | |
| S6 slot = soup3 | -.00516 | .02 | |
| S6 slot = prob-mean(S6fall, s43, s44) | -.00079 | .24 | |
| ours slot (R6) .1 rank of KC1 / K14 DT / DT + T_b | +.00023 / +.00049 / +.00023 | .57-.70 | split-half (K14 DT) .40 |
| ours slot .1 rank of KC-all / fleet D / T_b / s44 | -.00029 to -.00058 | .21-.36 | |

- The LB says r224 is a positive component:
  - the V224 lift;
  - a25 beat .35;
  - a15 lost to a25 (brief history).
- So swapping r224 for our member removes a LB-positive leg and adds a LB-negative one: expected negative (MODEL). The gold58 +.0005
  is noise.
- The S6 slot already holds the best S6 checkpoint. Seeds and soups only dilute it.

## 6. Is gold58 biased for member additions? (LB calibration, MEASURED `ours_more_g58.log`)

| member change (on the base that was on the LB) | gold58 | LB |
|---|---|---|
| KC1 .2 at the end on the R6 entry (uniform, A224 .35, S6fall .2) | -.00002 (P .50, sd .0015) | display step DOWN |
| R7 prob-mean .5 on HW (R7 OOF as proxy for the public no-flip R7) | -.00082 (P .35) | non-move |
| S6fall .2 added on R6-uniform (reference) | +.00095 (P .72) | positive (S6F entry) |

- On a realistic base, gold58 does not over-rate our members. It reads them as roughly 0, and the LB says somewhat negative.
- The KC G2 gate's "+.0068 / +.0072" was measured on an R7-alone base, which inflates a member's value by 3-10x. MEASURED
  `wrongbase.py` re-run, gain of .2 member on base:

  | member | on R7 alone | on R6 alone | on c4 |
  |---|---|---|---|
  | s43 | +.0062 | +.0019 | -.0009 |
  | r384 | +.0073 | +.0032 | -.0020 |
  | S6fall | +.0068 | +.0020 | +.0006 |

  Lesson: never gate a member on a single-model base.
- Per label, the KC1-on-R6 gold58 split was +.0106 Lateral OA, -.0084 PF OA, -.0042 Fracture, -.0030 Synovitis. Those are exactly
  the labels where c4 is fragile and our members are weak.

## 7. Why our members fail (MODEL, with the evidence)

1. **Label source.**
   - On LLM report-label OOF (n 4,333), our members beat R7 by .010-.020 macro AUC: T_b .906, K14 DT .898, fleet D .895 vs R7
     .885 (MEASURED `ours_oof.log`). Their biggest edges are on ACL (+.03-.04) and Contusion (+.05-.06).
   - On radiologist gold58 they only tie R7 (.919-.924 vs .919) and lose to R6 (.937).
   - Our training targets are report-derived (frac75 / llm_labels_v4_blend; T_b moved ~.3 of the target onto R7's estimate).
   - They learn the report definition of a finding. The hidden labels behave like gold58 (MODEL).
   - The public members are tuned to gold58-like labels: nartaa says the JEV / blendjev targets and the SWA epoch choice were
     tuned on gold58 (CLAIMED, research_1hr critique C1). R6 / R7 use heliosli's private relabels.
2. **Recipe gap.**
   - Our ports of the strong public recipes lose quality:
     - our RSNA-only CoAtNet seeds serve .911-.914 vs nartaa's .923;
     - our OAI CoAtNet .907 ("port-broken");
     - our S6-recipe seeds .915-.921 vs nartaa's .931 (MEASURED RAPTOR_RENTED_RESULTS, PROGRESS_14, HANDOFF).
   - Part of that gap may be nartaa's gold58-based selection inflating the public checkpoints' gold58 numbers (MODEL). The LB
     still agrees with the direction.
3. **Architecture is not the bottleneck.**
   - Swin-T on our labels: .918 single fold, the same as ConvNeXt at .916-.925 per fold.
   - ResNet50-RadImageNet: .877 (MEASURED box logs).
   - CLAIMED in RESEARCH_960 (10-14): top-50 teams use their own 5-fold families on image-informed pseudo labels, with no
     external data. That is a labels project, not a backbone swap, and it is not buildable in 2.5 days.

## 8. Training before 10-20 night: options, timelines, honest feasibility

Facts:
- Box V100 idle (MEASURED nvidia-smi 0 %, 5 MiB, 17:05).
- Timings:
  - ConvNeXt fleet ~3.5 h per fold (T_a, MODEL in PROGRESS_14);
  - KC all-data member ~3.4 h (s2007 23:31 -> 02:54, PROGRESS_16);
  - Swin-T ~6.7 h per fold (25.1 min/epoch x 16, box log);
  - S6-recipe seed ~12 h (PROGRESS_16);
  - Raptor CoAtNet seed ~4 h on a 4090 (rental), GUESS 6-8 h on the V100.
- Friend's ~30 Kaggle GPU-h (T4 / P100), GUESS 2-3x slower than the V100 per fold. A 12 h session cap and about 2 concurrent GPU
  sessions are CLAIMED (Kaggle docs from memory).
- The RSNA shards would have to be uploaded as a dataset first (GUESS hours).

| option | ready (if started ~19:00 10-19 on Eesh's go) | LB read | plausibly additive? |
|---|---|---|---|
| 1 all-data ConvNeXt / KC member (new seed or label mix) | trained ~22:30-23:00; gold58 sanity ~23:30; kernel = c4 + existing fleet / KC stage appended after S6F, pushed 10-20 ~03:00-06:00 | 10-20 ~13:00-18:00 | No. Same recipe family, 0 of 6 on the LB; single model weaker than KC-all |
| 5-fold ConvNeXt on the V100 | 10-20 ~12:40 (after our own 12:00 freeze); push ~16:00-18:00 | 10-21 ~01:00-05:00 | No, same reason |
| Swin-T (different family, same labels) 5 folds | ~33 h on the V100: not before 10-21 | too late to read and follow up | No: .918 at fold 0 = ConvNeXt |
| Our Raptor port (CoAtNet) seed | ~6-8 h, i.e. 10-20 morning | 10-20 evening at best | No: our port loses .009-.016 and read negative at .30 |
| OAI-pretrained variant (rulings 10-14 allow public external data) | K14 already IS this (OAI T123); our OAI Raptor broke | n/a | No: K14 read 0 twice |
| Distil a student from c4 (in-sample teacher on train) | ~3.5 h per fold | 10-21 | No: the teacher is in-sample on train, so the student copies c4 (no diversity) |
| Friend's 30 GPU-h | 1-2 folds by 10-20 noon after a shard upload | 10-21 | Same recipes as above, so no |

- **Honest answer.** One model can be trained, gated, built and LB-read before 10-22 10:15 IST, with ~1 day of slack. But:
  - P(best-id move) is ~.3, mostly the noise coin, discounted by the 0-of-6 prior;
  - P(true private gain > 2e-4) is GUESS <= .05;
  - it costs ~17 GPU-h, 1-2 slots, a PREREG amendment, Eesh's go, a new appended leg plus gate, and crosses our own 10-20 12:00
    freeze if 5 folds.
- **Recommendation:** do not train.
- The one training direction with a real mechanism is a gold-aligned label table (image-informed relabels).
  - Building it needs labels we do not have: heliosli's relabels are private, and tuning a table on gold58 spends our only
    independent check.
  - It is a multi-day project. That puts it out of reach for this comp.

## 9. Using the unused ~2.5 h of hidden runtime with our assets

What is ready and gated (MEASURED PREREG_10-15):
- the KC stage, which ran in the combo, ~0.8-1 h on hidden (MODEL from read times: R6 6h03-6h15 vs combo ~7 h);
- the fleet / S stage (V224AOS gate);
- the R7 late leg (A14, ~1.3 h);
- K14 fleet stage code.

Each can be appended after S6F with its own guard, which is how A14 was built. Each fits c4's measured <= 6h33 latency within the
9 h limit.

Each of those legs read negative at .2 (section 2). At .05 they are expected slightly negative and are unreadable at three
decimals (sections 2-3). Using runtime for them adds hidden-run risk for no expected gain. **From this lens, leave the runtime
unused**, or give it to a public-checkpoint leg if another lens finds one. Public additions are 3 of 3 positive; ours are 0 of 6.

## 10. How a GPU could help (this lens)

- Training: no (section 8).
- Measurement:
  - Serving c4's legs on the 16 extra gold74 studies would grow the screen 58 -> 74 (about a 10 % narrower sd).
  - Gold58 is already ruled unusable for weight tuning (PF OA miscall), so this buys little.
  - Box GPU ~20-30 min (GUESS). Only worth it if another lens needs a slightly larger sanity set.
- Nothing else in our assets needs a GPU: all gold58 and OOF predictions already exist and were copied read-only.

## 11. Private-board note

- The picks are already scored once; no rerun (MEASURED comp.json flag, research_1hr timing.md).
- Our members cannot hedge the private board:
  - they are weaker and correlated with the report-label axis;
  - the one open private risk (heliosli's weights being public-tuned) is already hedged by B = R6 uniform.
- Adding any of our members to a final pick would move private by ~0 to slightly negative (MODEL).

## 12. Closed by this lens (reopen only on new evidence)

- Our members at any weight, at the end, in the ours slot, or on label subsets: KC1, KC-all, K14 DT, fleet D, T_b, T_a, s43,
  s44, soups, R7.
- Replacing r224 or S6fall with any of our members.
- Training any new model before the deadline. Reopen only if a public gold-aligned label table appears, and even then it is not
  buildable in time.
- Gating members on a single-model base: inflation of 3-10x (section 6).
