# Submission-slot plan, 10-04 to 10-22 (lens: SLOTS)

Written 2026-10-03 23:30 IST (18:00 UTC). Research only: nothing was submitted, pushed, versioned or posted, and
the box was not touched. Tags: **MEASURED** (LB, gold or code fact), **MODEL** (simulation, assumptions stated),
**GUESS**.

`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`.
Helpers in this folder: `slots_sim.py` (closed-form tables + W-curve slot-value simulation; output
`slots_tables.txt`, `slots_sim_out.txt`) and `slots_resolution.py` (how precise one read is;
`slots_resolution_out.txt`).

Short names used below:

| Name | Meaning |
|---|---|
| P | the 0.943 parent (repro_0946) |
| C8 | crop leg in the new crop-fleet cell: K=8, v4-label members cropaug_f1 + cropaug_f2 + crop_f1 (kernel built, `SP/cropfleet/kernel/rsna-sub-cropfleet`) |
| C24 | old cell 31 crop leg (cropaug_f1, K=24): the 10-03 probes |
| F0 | ConvNeXt-T fleet, fold-0 checkpoint of the winning label variant |
| Fv2 / Fv3 / Fv4 / Fv5 | fleet dataset versions from `SP/fleet2/SCHEDULE.md` section 5 (v4 = full 10 members) |
| H1 / H2 | Hisernberg ext anchor (C1) / 6-view Raptor (C2), `SP/hisern_kernel/` |
| W* | model-best fleet weight from the closed-form table (A1 below), fed by the solo read and the offline rho |
| Wave A / Wave B | first / second submission batch of a UTC day (see section 1) |

---

## 0. Bottom line

1. **95 slots remain** (19 UTC days x 5, shared with SwaplY). The plan uses about 65-80. Unused slots are fine.
   A slot spent on a low-information read is not free when it tempts the final pick toward a lucky public score
   (section 2.4).
2. **One public read is coarse (MODEL, `slots_resolution_out.txt`).** The parent + fleet delta has public SD
   0.0012, plus 0.0003 display rounding, plus the unknown position of the parent inside its 0.943 bucket. The
   public luck does not carry to private: corr 0.10 at fixed true effect. A solo AUC read has SD 0.0047. A paired
   difference between two fleet subsets from one commit (adding a second family) has a true value of +0.0001 and
   SD 0.00055, under the 0.001 display step.
   - So the LB can answer three things: is it broken, how strong is the member, and is the blend level about
     +0.003 or more vs about 0.
   - It cannot rank W values, label variants or family subsets. Those go offline: G1b rho, local LB, gold58.
3. **W curves are not worth slots (MODEL, `slots_sim_out.txt`, 700 worlds).** Prior: fleet solo U(.915, .940),
   rho U(.72, .88). Expected private delta of the chosen W:

   | How W is chosen | Slots | Expected private delta |
   |---|---|---|
   | Fixed W .30 | 0 | +0.00323 |
   | Solo read + offline rho, then closed form | 2 | +0.00347 |
   | Argmax of 4 displayed W reads | 4 | +0.00342 |
   | Fit to 4 W reads + solo | 5 | +0.00355 |
   | Oracle | - | +0.00365 |

   The whole W question is worth +0.0004. The 2-slot policy gets most of it.
   - **Rule: per new configuration, read the blend at W\* plus the solo, and nothing else.**
   - Member strength is the real lever. At rho .80, a solo .935 fleet is worth about +0.005 and a .925 fleet
     about +0.003 (table A1).
4. **Make every commit carry every next-step variant, and make failure loud.** Two gaps:
   - The fleet cell writes one file only (`SP/round3/fleet/fleet_blend_cell.py`: a single `RSNA_FLEET_W`), and
     no fleet-solo kernel exists (FLEET_REVIEW.md:312).
   - The crop and fleet cells both fall back silently to a byte copy of the parent. On the hidden rerun a 0.943
     read then cannot be told apart from "ran, no effect", and the receipts cannot be read.

   The build items are in section 6. The biggest is ONE "combo" kernel (P + C8 cell + multi-output fleet cell)
   that writes the P+C8, P+F, P+C8+F and solo files from one run. It is recommitted once per fleet dataset version.
5. **Untested assumption the multi-file plan rests on.** `SP/round2/rules/Code_Requirements.md:8` says
   "Submission file must be named `submission.csv`". The CLI does pass `file_name` for code submissions
   (`kaggle_api_extended.py:1862-1899`). But whether the scorer accepts `submission_w45.csv` on the hidden rerun
   has never been tried by us.
   - **Slot 10-04 A2 tests it, with the most useful second file anyway (C8 solo).**
   - Every commit keeps the pre-registered main candidate in `submission.csv`, so a "no" costs one slot, not a
     plan.
6. **Stop probing at end of 10-17 UTC.** Freeze code 10-19 00:00 UTC. Final-candidate reads 10-19 to 10-21 A.
   Select provisionally in the UI from 10-08 on, and lock 10-21 12:00 UTC. Use 10-21 B and 10-22 only for retries
   of errored candidates.

---

## 1. Mechanics and constraints

- **Slots.** 5 per UTC day, team-wide (Eesh + SwaplY). Day boundary 00:00 UTC = 05:30 IST.
  - Remaining: 10-04 .. 10-22 = 95.
  - All 5 slots of 10-03 are used: crop25, v9crop25, crop15, v9 and mixsacrop25, all on the OLD K=24 cell.
    Their reads are due about 22:00-01:00 UTC tonight.
- **Turnaround** about 8-10 h per read (GUESS: "rerun queue up to ~8 h", 09-24; the parent alone runs about
  7-8 h). Measure it on the 10-03 batch.
- **Two waves per day turn 1 decision/day into 2:**
  - Wave A: 00:05-03:30 UTC (05:35-09:00 IST), 3 slots.
  - Wave B: 08:30-15:00 UTC (14:00-20:30 IST), 2 slots, decided with wave A's reads in hand (or the previous
    night's).
  - If a read is late, go ahead with that rule's default branch. Never hold a slot past 16:00 UTC waiting.
- **Commits.**
  - Each new notebook version needs one commit: about 0.3-0.5 h of T4 quota for the 3-study run. Reruns for
    scoring do not use our quota.
  - Eesh asked to keep Kaggle GPU use low. Quota is about 20 h until about 10-10 (teacher-v2 is still running),
    and SwaplY's quota is the backup.
  - This plan needs about 10-14 commits in total (4-7 h). If the filename test fails (section 0, item 5), about
    20-25.
- **Final picks** must be already-scored submissions. In a code competition the private score comes from the
  same rerun as the public read, so a successful read is a successful pick. There is no later rerun risk.
  - Kaggle auto-picks the top-2 public if nothing is selected, and that is the max-of-noise choice. So keep a
    provisional selection set from 10-08 on.
- **Merge.** Two-way only, Eesh posts. Team cap = 5 x days running at request date (about 330 on 10-03, +5/day).
  Ours is about 39 now. Myo's team was 210 on 10-03. Two-way stays legal through 10-15 even if both teams use all
  their slots (margin about 80 - 5 x days). After a merge the TEAM has 5 per day in total (section 5).
- **SwaplY.** Ask him to take slots only from that day's wave-B tail item. Keep one wave-B slot flexible until
  12:00 UTC.
- **Kaggle API is rate-limited right now.** Poll reads once per wave (`kaggle competitions submissions`), not in
  a loop.

---

## 2. What one read is worth (numbers behind the rules)

### 2.1 Closed-form blend table (MODEL, `slots_tables.txt` A1)

Binormal two-reader model, the same formula as `SP/fleet_g1b.py:best_lift`. Entries are best W and lift over the
0.943 parent. rho is the within-class correlation with the parent.

| solo \ rho | .74 | .78 | .82 | .86 |
|---|---|---|---|---|
| .915 | W.27 +.0032 | W.21 +.0018 | W.14 +.0006 | W.03 +.0000 |
| .920 | W.30 +.0043 | W.26 +.0027 | W.21 +.0013 | W.11 +.0003 |
| .925 | W.35 +.0055 | W.31 +.0038 | W.27 +.0022 | W.19 +.0009 |
| .930 | W.39 +.0070 | W.36 +.0052 | W.33 +.0035 | W.27 +.0019 |
| .935 | W.43 +.0087 | W.42 +.0068 | W.39 +.0050 | W.36 +.0032 |
| .940 | W.48 +.0106 | W.47 +.0088 | W.46 +.0069 | W.45 +.0051 |

Three caveats:
- The table is optimistic against our ledger: crop members around 0.924 have read +0.000 at low W
  (deploy_final, and Tian's 0.927 at beta 0.1-0.2).
- The measured Spearman values (.78-.85) are overall, not within-class. So treat the table as an ORDERING.
- The first reads calibrate its level, which is what the "model misfit" rules in section 4 are for.

### 2.2 Three-way blend (MODEL, `slots_tables.txt` A3)

P + C8 + F beats the better two-way blend only when the crop leg is nearly as strong as the fleet, or weakly
correlated with it:

| Case | Three-way | Fleet alone |
|---|---|---|
| C .925 / F .925, rho_CF .80 | +.0037 | +.0022 |
| C .925 / F .935, rho_CF .75 | +.0075 | +.0068 |
| C .925 / F .935, rho_CF .85 | wC -> 0, same as fleet alone | +.0068 |

So the crop leg stays only while the fleet is weak or the offline rho_CF is 0.80 or lower. Expect it to fade out
as the fleet grows, so read the three-way at Fv2 and again at Fv4 only.

### 2.3 Resolution (MODEL, `slots_resolution_out.txt`, fleet .930 / rho .80, W .35, 400 test draws)

| Read | SD public | SD private | SD(pub - priv) |
|---|---|---|---|
| Solo AUC | .0047 | .0030 | .0055 |
| P+F delta | .0012 | .0008 | .0014 |
| Paired subset diff | .00055 | .00034 | .0006 |

What this means:
- A single blend read separates a true +0.003 from 0 at about 2 SD.
- A solo read separates .915 from .930.
- Nothing finer is readable.

### 2.4 Winner's curse

Public luck is about 90% independent of private luck. Choosing the top public read out of N different
configurations adds roughly SD(pub - priv) x E[max of N normals] = 0.0014 x 1.16 (N=5) or x 1.54 (N=10) of pure
noise to the chosen candidate's public score, with nothing added on private (about +0.0016 for N=5, about
+0.0022 for N=10). That is as large as the effects being chosen between. So:
- Final picks are chosen by a posterior (section 7), never by the raw max.
- Each extra W variant read on the same structure adds less temptation, because the variants share their public
  noise. Different structures add the most.

---

## 3. Day-by-day plan

Training inputs below come from `SP/fleet2/SCHEDULE.md`. Its times are ESTIMATES at 40 min/epoch alone; the 10-03
evening log says 65-80 min/epoch while two jobs share the V100, which that schedule already prices in as about
1.8x. Re-anchor every row to the G0 line.

"K#" = a commit. Each commit's `submission.csv` is its pre-registered main file; the other files are listed in
section 6.

| UTC day | Inputs expected (IST) | Wave A (3) | Wave B (2) | What the reads decide |
|---|---|---|---|---|
| **10-04** | 10-03 reads about 03:30-06:30 IST. K1 = crop-fleet commit tonight (dataset manifest version first). F1 label decision at epoch 4 (about 05:00-06:30). F0 winner checkpoint about 08:00-14:00. EV2S f0 starts. | A1 K1 `submission.csv` (P+C8 @.25). A2 K1 `submission_fleetsolo.csv` (C8 solo; FILENAME TEST + canary). A3 K1 `submission_w45.csv` (rho probe), or `w15` if C24 crop25 and crop15 both read 0.942 or lower | B1 K2 combo with Fv1=F0: `submission.csv` = P+F0 @W\*. B2 K2 `fleetsolo_F.csv` (G2 + canary). Default if K2 is not committed by 12:00 UTC: H1 + H2 if that commit was made, else K1 `w15`/`w35` (whichever side A1/A3 point to) | Crop leg in or out (R1). Multi-file works? G2/G3 for the fleet (R2, R3) gate about 70 V100-h (box rules in section 4) |
| **10-05** | K1/K2 reads. EV2S f0 done about 01:00 IST. F1 all-data s43 done about 04:00 IST. R50 f0 running. | A1 K2 `combo_c15_f.csv` (P+C8@.15+F0@W\*-.05). A2 K2 `w_hi.csv` (P+F0 @W\*+.15, fleet rho check), only if A1 of 10-04 read 0.943 or higher; else a spare filler. A3 filler (H2 if committed) | B1 K3 combo with Fv2 (F0 + s43 + EV2S f0): `submission.csv` P+Fv2 @W\*. B2 K3 `fleetsolo_F.csv` | Three-way kept? (R4) Fv2 step (R5). EV2S value decided OFFLINE (G1c), not on the LB |
| **10-06** | R50 f0 done about 13:00 IST 10-05. EV2S s143 about 19:00. R50 s243 about 03:00 10-06. F1 s44 about 10:00. | A1 K3 `combo_c15_f.csv` (only if R4 kept the three-way). A2 + A3: H1 + H2 as a pair (one commit, if quota allows), else K3 `w_hi.csv` + spare | B1 K4 combo with Fv3: `submission.csv`. B2 K4 `fleetsolo_F.csv` | Fv3 step. Hisern adoption (R6). Merge partners get the Fv2/Fv3 numbers (merge plan wants them by about 10-07) |
| **10-07** | EV2S s144 about 01:00 IST. R50 s244 running. F1 s45 starts. | A1 K4 three-way (if kept). A2 the ladder item with the largest model gain not yet read (section 7.3). A3 spare | B1-B2 held for Fv4 if all 10 members are done (about 16:00 IST), else ladder | |
| **10-08** | Fv4 = full fleet (about 10 members). Local LB on the box may be live (kmimic), and moves W choices offline. | A1 K5 (Fv4) `submission.csv` @W\*. A2 K5 `fleetsolo_F.csv`. A3 K5 `combo_c15_f.csv` | B1 K5 `w_lo.csv` or `w_hi.csv` (only if the model and the local LB disagree on W by 0.1 or more). B2 K5L (light: the 6-member prefix) only if the K5 main ERRORED | Main structure fixed (P+Fv4, with C8 or without). Provisional selection from now on. |
| **10-09 .. 10-12** | Slack seeds of the family with the best offline G1c/solo (SCHEDULE 10-08 to 10-14). Fv5 about 10-11. Merge replies. | 2-3 per day: ladder items (section 7.3). Fv5 main + solo on its day. | 0-2 per day: rest of the ladder. Otherwise leave empty. | Each extra-seed version is adopted by R5. No W reads unless the solo moved by 0.004 or more. |
| **10-13 .. 10-15** | Merge mechanics (closes 10-15 23:59 UTC). | Merged: section 5 schedule (3 per day for combined kernels). Not merged: 2 per day, ladder or last seeds. | | |
| **10-16 .. 10-18** | Last training starts 10-17 18:00 IST. Last dataset version 10-19. | 2-3 per day: final-structure reads (merged combos, or Fv6 if seeds keep paying) | 1-2 per day | **Probing ends 10-17 23:59 UTC.** From here on, a read can change the picks but not what gets built. |
| **10-19** | **Code freeze 00:00 UTC.** KF = final combo commit (last fleet version, final W\*, crop decision, base decision). | A1 KF `submission.csv`. A2 KF `fleetsolo_F.csv` (canary for the final member set). A3 KF pick-2 alternative (section 7.2) | B1-B2: KF ladder items not yet scored | |
| **10-20** | | A: KF ladder (W\* +/- 0.1, crop on/off, subset without the weakest family) | B: rest of the ladder, or re-submit any ERRORED final candidate once | |
| **10-21** | | A: anything still unscored that could be a pick. **12:00 UTC: final selection in the UI** (rule in section 7). | B: retries of errored candidates only | |
| **10-22** | | Retries only (each must be scored before about 20:00 UTC). Change the selection only by the 7.1 rule. | none | Deadline 23:59 UTC |

Expected use: 10-04 to 10-08 about 25; 10-09 to 10-12 about 10; 10-13 to 10-18 about 18; 10-19 to 10-22 about
15. **Total about 68 of 95.**

---

## 4. Pre-registered reads

Notation:
- d = displayed score minus 0.943, in 0.001 steps.
- "Canary" = a solo file that reads EXACTLY like the parent (0.943). It means the cell restored the parent on the
  hidden rerun, so every read from that commit's runs is void. Section 6 item 2 turns this into a loud failure.

### R0. 10-03 batch (C24 cell), already registered in WIN_PLAYBOOK section 3

- Feed crop15/crop25 into `policy_sim.posterior_means`.
- v9:
  - 0.944 or higher: rebuild the combo kernel on the v9 base from K3 on.
  - 0.943: do not bundle it (no evidence, adds an unmeasured change).
  - 0.942 or lower: never bundle.
- mixsacrop25 vs crop25: d(mixsa) at least 1 above d(crop25) would contradict the 21:50 rho result. Otherwise
  confirmed: v4-label crop members only.
- Any C24 read of 0.944 or higher is already a scored pick-2 candidate.

### R1. Crop leg C8 (K1)

| Read | Outcome | Action |
|---|---|---|
| C8 solo | canary (0.943) | K1 void: debug the restore reason before any other crop read |
| C8 solo | below 0.915 | Crop leg out of the main structure |
| C8 solo | 0.915-0.930 | Expected |
| C8 solo | above 0.930 | Raise the crop-weight prior one row in A1 |
| C8 W.25 | d +1 or more | Crop leg stays (two-way and three-way) |
| C8 W.25 | d 0 | Kept only through R4 |
| C8 W.25 | d -1 or less | Crop leg out of the main structure; pick-2 only through the posterior |
| C8 w45 | d +1 or more | rho about .80 or lower: the crop weight in combos may go to .20 |
| C8 w45 | d -2 or less | rho .86 or higher: crop weight at most .10, or out |

### R2. Fleet solo (G2, every fleet version)

- Canary: void.
- F0 below 0.915: family 1 stops after the s43 already running; the box lanes go to family 2 (EV2S, R50) and
  extra crop all-data seeds.
- 0.915-0.921: all-data seeds only, re-judge at Fv2.
- 0.922 or higher: the full SCHEDULE.

From Fv2 on, a version's solo must not fall by 0.006 or more against the previous version (beyond 1 SD); if it
does, look for a broken member.

### R3. Fleet blend at W\* (G3)

| Read | Action |
|---|---|
| d +1 or more | Final-structure candidate |
| d 0, solo 0.925 or higher | Keep building: seeds raise the solo, and the model expects a lift to come |
| d 0, solo below 0.925 | As above, and switch the next seeds to pure steven_v4 targets if they are on mix50 (decorrelation) |
| d -1 or less, solo fine | rho is higher than modelled: read once at W\*-0.15 (`w_lo.csv`, same commit), then pick W from that |
| d -2 or less | Look for a broken member: compare the solo and the commit log parity lines |

### R4. Three-way (P + C8 + F)

Keep the crop leg in the main structure only if BOTH hold:
- three-way d is at least the matching two-way P+F d;
- offline rho(C8, F) on the 867 fold-0 val studies is 0.85 or lower.

Re-check once at Fv4, then freeze.

### R5. Fleet version step (v1 -> v2 -> ...)

- Adopt the new version unless its main read is 2 or more below the previous version's, or R2 flags its solo.
- One-step drops of 1 are noise (SD 0.0012): ignore them.
- W\* moves only if the solo moved by 0.004 or more.

### R6. Hisernberg base

- Adopt H2 as the base for the combo kernel only if H2 reads 0.945 or higher, OR H2 is 0.944 or higher AND 2 or
  more above H1.
- H1 below 0.943 means the anchor (repro minus Repair-v1, outer .65) is weaker: drop the family.
- It is a filler: never displace a fleet read for it.

### R7. Errors

- A rerun ERROR or timeout on a candidate that could be picked: re-submit the same file once (infrastructure
  noise).
- A second error: structural. Use the light variant (6-member prefix) for that version.

### Box allocation the reads trigger (throughput)

- R2 below 0.915: the V100 moves to family 2 plus crop all-data seeds.
- R4 says the crop leg is out: stop all crop training on the box.
- Solo 0.935 or higher at Fv2: every free lane goes to more all-data seeds of the best family. The model gain per
  +0.005 of solo is about +0.0015-0.002 (A1).

---

## 5. If a merge happens (decided by 10-15)

- A merged team still has 5 per day. The partner's scored submissions become team submissions with no slots
  spent.
- Days M to M+2: 3 per day go to combined kernels (partner stack + our fleet, and partner + P only if the
  runtime fits).
  - Our parent takes about 7.7-8.3 h. P plus a full partner stack will not fit in 9 h.
  - So the combined kernel = partner stack + our fleet (+ C8), weights from the closed form with the partner's
    solo in place of 0.943.
- Pre-register:
  - The combined kernel's main file is pick-1 eligible if d is at least 0 against the partner's best public.
  - Pick 2 = our best non-merged structure (section 7). That is the "most different d4 share" rule from
    WIN_PLAYBOOK section 6.
- Merge requests near the deadline: do not let a pending request eat the 10-15 slots. Keep the wave-A reads on
  our own structure until the merge is confirmed.

---

## 6. Build items this slot plan needs (in priority order)

1. **Multi-output combo cell** (append to the crop-fleet notebook, after the crop-fleet cell).
   - It reuses the crop-fleet cell's atomic multi-file writer (`SP/cropfleet/crop_fleet_cell.py:607-644`) and the
     fleet worker unchanged.
   - Fleet raw-prob mean is computed once. Files:
     - `submission.csv` = the pre-registered main;
     - `fleetsolo_F.csv`;
     - `w_lo.csv` / `w_hi.csv` (W\* -/+ 0.15);
     - `combo_c15_f.csv` (P + C8 .15 + F at W\* - .05);
     - `crop_only_c25.csv` (P + C8 .25: an in-run copy of the K1 main, for pick-2);
     - `sub_<family>.csv` (main structure minus one family).
   - Ranks over the full test set, as in both cells.
2. **Loud fallback.** On any failure of the fleet or crop leg, write the parent with ONE label's ranks reversed
   (Fracture), not a byte copy.
   - That file reads about 0.87 instead of 0.943, so a silent restore can no longer pass as "no effect".
   - It costs nothing for the final picks: a failed run was never pickable, because an exact parent copy ties
     the bottom of the 0.943 block.
   - Keep the parent-copy behaviour only for `submission.csv` of KF (belt and braces at the end).
3. **Filename test (10-04 A2).**
   - If the submit is rejected or the scorer errors, switch to "one variant per version": edit W_MAIN and re-push,
     about 0.4 T4 h each.
   - Solo reads then come from a parent-free light notebook (fleet or crop cell only). It also turns around faster
     (about 1 h runtime instead of 8).
   - The plan then needs about 2 commits per version instead of 1. Still inside quota, using SwaplY's.
4. **Pack key collision.** f0_s42 mix50 and f0_s42_v4 share the `f0_s42` key (SCHEDULE section 2). Only the
   winning variant ships. Re-key (for example seed 46) if the loser should ever join the fleet.
5. **Runtime.** The parent takes about 7.7-8.3 h; C8 at K=8 about 5-10 min; a 10-member fleet about 35-60 min
   (ESTIMATE). The fleet valve keeps a prefix (family-interleaved manifest order, SCHEDULE section 5), so a short
   run degrades rather than failing.
   - The hidden-run member count is invisible.
   - The Fv4 solo read is the only check that the full set runs at scale: a solo well under the Fv3 solo means
     the valve cut members.
   - K5L (6-member prefix) is pre-built but committed only if R7 fires.

---

## 7. Final picks, stopping, and the leftover slots

### 7.1 Selection rule (pre-registered)

1. **Eligible:** scored, not a loud fallback, and not an exact copy of P or of a public notebook output.
2. **Posterior** for each eligible candidate: post = m + 0.5 x (d - m), where
   - m = the model delta (A1/A3 at the candidate's weights, using the latest solo reads and offline rho);
   - d = displayed delta;
   - 0.5 = prior SD about read SD, about 0.0013.

   Variants of one structure share the structure's average d (shared public noise) and differ only through m.
3. **Pick 1** = max posterior.
4. **Pick 2** = max posterior among candidates that differ from pick 1 in member set or base (not only in W). If
   that is more than 0.0015 below pick 1, take pick 1's structure at the other side of W instead (W\* -0.1 if
   pick 1 used W\*).
   - Merged: pick 1 = the merged main, pick 2 = our best unmerged structure.
5. **Never:** pure repro_0946 (WIN_PLAYBOOK section 6), a structure scored only once with d of -1 or less, an
   unscored file.
6. **UI:** provisional selection from 10-08, updated at each wave; final at 10-21 12:00 UTC.
   - 10-22 retries may replace a pick only when the retry is the errored version of the pre-registered pick.

### 7.2 When to stop probing and lock

| When | What stops |
|---|---|
| End of 10-17 UTC | Structure questions (new families, crop on/off, base swap, label variant). Matches the last training start. |
| 10-19 00:00 UTC | Code freeze: KF is the last new kernel code, carrying the last fleet version. 10-19 B and 10-20 read KF files only. |
| 10-21 12:00 UTC | Lock, after the 10-21 A reads. |

### 7.3 Leftover slots: the near-final ladder

From 10-07, spare slots go down this list. Read only items not yet scored for the current structure, and at most
one item per W step of 0.1.

1. The current main at a fleet version not yet read (always first).
2. Pick-2 candidate: the best structure that differs in members or base (crop-only C8 blend, H2 + fleet if R6
   adopted it, merged combo).
3. Main structure minus its weakest family (`sub_<family>.csv`): a decorrelated pick-2 if one family is worse on
   private.
4. W\* -0.1 (cheap insurance against a model that is too optimistic, as the crop ledger suggests).
5. W\* +0.1 only if the local LB or the solo read says the fleet is stronger than modelled.
6. Retries of errored candidates.

Never: identical files (zero information, same output); W steps under 0.1; paired subset reads meant to rank
families (unreadable, 2.3); per-label maps (measured noise).

Slots still left after the ladder stay unused. That is cheaper than adding candidates that only feed the winner's
curse.

---

## 8. Day-1 checklist (10-04)

1. Overnight: dataset `rsna-knee-cropaug-members` version with the C8 manifest ready.
   - Then commit K1 (Eesh's OK, about 0.4 T4 h).
   - Check the log for `RESULT OK fleet blended: 3 member(s)` (cropfleet README section 2).
2. 05:35 IST: read the 10-03 batch (one poll), apply R0, submit A1-A3.
   - Note whether the A2 `-f submission_fleetsolo.csv` submit call is accepted.
3. By 05:00-06:30 IST: F1 label decision at epoch 4 (val AUC vs steven_v4 and G1b rho vs the real v8 arm).
   Then, by about 15:00 IST:
   - pack the F0 winner;
   - build K2 (combo cell, items 6.1-6.2);
   - commit;
   - check the log.
4. 14:30-17:30 IST: submit B1-B2, or the default fillers. Write all 5 reads and the R-rule verdicts into
   HANDOFF.md the same night.
