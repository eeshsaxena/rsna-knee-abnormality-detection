# ENDGAME V3: RSNA Knee plan, 10-04 to 10-22

Written 2026-10-03 ~23:50 IST. Research only: nothing was submitted, pushed, versioned, posted or run on the box.
This file merges the three endgame3 lenses (`slots.md`, `compute.md`, `risk.md`) with HANDOFF (10-03 evening),
WIN_PLAYBOOK sections 3/4/6/7, TOP_TEAM_RECIPES (bottom line + skeptic verdict), `policy_sim.py` /
`crop_table.json`, localLB, cropfleet, hisern_kernel and fleet2. Where the lenses disagree, section 10 records
the call and the reason.

Tags: **MEASURED** = LB read, log line, file or code fact. **MODEL** = simulation, or arithmetic from measured
rates and benchmarks (the lenses' "ESTIMATE" is filed here). **GUESS** = judgement.

`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`.
Times are IST unless marked UTC. A UTC day (5 team slots, shared with SwaplY) runs 05:30 IST to 05:29 IST next day.

Short names:

| Name | Meaning |
|---|---|
| P | the 0.943 parent (repro_0946) |
| C | crop leg served by the K=8 crop-fleet cell. Cv1 = legacy v4-label members cropaug_f1 + cropaug_f2 + crop_f1 (in K1). Cv2 = all-data v4 R50 s0 + s1 + crop_f1 |
| F | ConvNeXt-T fleet (4 views x 24 triplets, 320 px). fv1 = fold-0 D1 winner. fv2 = s43 + f0_s42 + EV2S f0_s142. fv3 = 5 all-data members (s43, s143, s44, s144, s45; F1 seeds only if family 2 fails). fv4 = optional expansion |
| CF | P + C + F blended in one hidden rerun |
| W* | closed-form best weight (`slots_tables.txt` A1, same formula as `SP/fleet_g1b.py:best_lift`) from solo + rho |
| K1..K5, KT, KM, KF, K7 | Kaggle commits (section 2 and 5). K1 = `rsna-sub-cropfleet`, already built |
| d | displayed public score minus 0.943, in 0.001 steps |

---

## 0. The plan in ten lines

1. **Final picks come from two independent legs (fleet, crop) and the blend of both.** MODEL (`risk_eval`,
   central prior, bronze, base field): crop leg alone 0.24-0.28; P+F at W .30 0.58; CF 0.71; two W of one
   member 0.28; pick rule "top, then other leg" 0.75; status quo 0.23. Use these for ranking options, not as odds.
2. **The binding resource is T4 time after the parent inside the hidden rerun, not the V100.** Parent time
   P = 7.7-8.05 h (MODEL, inferred from MEASURED outcomes: one K=24 member fits, two time out; the parent's own
   8 h guard). Fleet decode alone ~15-20 min (MODEL from the MEASURED 0.68 s/study). So: at most 5 ConvNeXt
   members fleet-only, 3 inside a combined kernel, until a measurement says more fits.
   [verify] The stage is decode-bound (compute.md: ~3 + max(15-20, N x 3-4.5) min), so N=3 instead of 5 saves
   only ~0-7.5 min. Whether a combined kernel fits depends mainly on P (plus any estimate margin): <= ~7.95 h
   for fleet + crop, <= ~8.1 h for a 5-member fleet alone (section 4, G-T). Above that, cutting N does not rescue
   the crop stage; only separate kernels or a lighter host (K7) do.
3. **Measure the window without new Kaggle GPU [verify]:** Eesh's 10-03 21:30 instruction put the Kaggle-side
   local LB ON HOLD (avoid Kaggle GPU) and chose the box mimic for parent-on-H. So: parent gold58 predictions
   come from the box mimic (compute.md), T4 member cost from K2's commit microbench, and the window itself from
   the hidden reruns: with per-file fail-loud, a scored file proves its stages fit and a missing-file error
   proves they did not, at the real ~1,322-study scale, for one slot and no quota. KT (gold58-only parent
   commit, ~0.5 T4-h plus the localLB README's required smoke run ~0.2 T4-h, or an explicit waiver of it) is
   optional, only on Eesh's explicit OK, and only once expansion past 5 members is on the table (after G2/G3).
   Gate G-T then sets N.
4. **Every commit from K2 on: multi-file output + per-file fail-loud** (a file whose stages did not all finish
   with the full member count is simply not written; never a parent copy), one 8.6 h process-age wall, start a
   stage if at least 20 min are left, ConvNeXt stage before crop stage.
5. **One public read answers only three questions** (MODEL, `slots_resolution_out.txt`): is it broken; how
   strong is the member (solo SD .0047); is the blend about +0.003 or about 0 (blend SD .0012). W, labels and
   families are chosen offline. Per structure: one blend read at W* plus one solo read. No W curves (the whole
   W question is worth +0.0004, MODEL), no paired subset reads (SD .00055, under the display step).
6. **Labels (D1):** provisional at epoch 4, final at v4 epoch 10 (~10:45 on 10-04). Solo input = gold58 mean
   over the last 3 epochs (raw and EMA), decorrelation input = rho vs the parent's real v8 arm. Ties go to pure
   steven_v4. Every later fleet run (EV2S too) uses the winner; crop runs always use v4.
7. **The V100 never idles.** Label-independent jobs are queued behind every running job. The 5-member core is
   done ~10-06 14:00; after that ~270 alone-hours remain to 10-17 (MODEL, 2 lanes at ~1.0x aggregate). They go
   to all-data seeds, crop seeds, one seed-soup test and one gated solo-strength probe; G-T and the soup test
   decide whether extra seeds ship as members, as soups, or not at all.
8. **Picks:** a standing pair is selected tonight (deploy_final + deploy_B, only to beat auto-select);
   candidates are ranked by a posterior, never by the raw top public read; Pick 2 sits on a different leg;
   never pure repro. Target: both picks already scored by 10-14. Final selection 10-21 12:00 UTC, latest
   18:00 UTC.
9. **The merge is still the biggest single lever** (two-way only, Eesh posts, closes 10-15 23:59 UTC; send the
   fleet numbers ~10-07).
10. **Honest odds (GUESS, unchanged from WIN_PLAYBOOK s7):** bronze ~45-55%, silver ~15-20% (mostly via the
    merge), gold ~1%, win under 0.2%.

---

## 1. Inputs as of 10-03 23:00 IST

| Item | Value | Tag |
|---|---|---|
| Public LB, 10-03 06:09 UTC | 0.943 tie = 880 teams (ranks 317-1196); 0.944 = 58 teams; silver line 0.945 (rank 247); gold 0.957; top 0.961 | MEASURED |
| Our best | 0.943 (repro 56750056, deploy_final 56753683), rank ~800 | MEASURED |
| 10-03 slots (all 5 used, old K=24 cell) | crop25, v9crop25, crop15, v9, mixsacrop25; reads due ~03:30-06:30 IST | MEASURED; expected public crop15 .9437, crop25 .9435, v9 .9430 = MODEL |
| Crop members, rho vs the real v8 arm on gold58 | v4-label .775-.806; teacher-mixed .830-.846; v4 ensemble .813 | MEASURED |
| Teacher mix on crop members, gold58 | cropaug_f1 v4 .8904 vs mixSA same split .8809, all-data mixSA .8802 | MEASURED |
| Best-W lift (binormal, parent .943) | .918 @ rho .742: +.0037; .927 @ .835: +.0021; .935 @ .80: +.006; .940 @ .78: +.009; best W ~.25-.30 for ~.92 members | MODEL |
| gold58 proxy (v8 arm + widedense as parent stand-in) + v4 crop ensemble | peak W .25 +.0034 (.15 +.0024, .35 +.0019, .45 0, .60 -.006) | MEASURED on 58 studies (SD .01-.02), proxy parent. [verify] Not evidence of a lift: the localLB self-test shows a pure-noise member reaching +.0029 (P .78) at its gold58 argmax W. Only the shape (falls past W .35) is usable. The same checkpoints also move ~.004-.007 on gold58 with slice sampling alone (cropaug_f1 .8860 vs .8904; mixSA all-data .8874 vs .8802, HANDOFF) |
| Fleet fold-0, two label variants | 65-80 min/epoch while sharing, 10 epochs each | MEASURED |
| Member-add kernels with no score | 3 of ~10 (56526314, 56690233, 56768269), all heavier variants | MEASURED |
| Fleet decode rate | 2,204 studies in 24.9 min on 4 vCPU = 0.68 s/study | MEASURED |
| Per ConvNeXt member on 2xT4, ~1,322 studies | 3-4.5 min | MODEL (timm 3090 bench scaled) |
| Kaggle GPU quota | ~17-20 h to the 10-10 00:00 UTC refresh. [verify] teacher-v2 finished 23:26 (HANDOFF), so there is no remainder left to subtract; re-read the quota page once before K1 | MEASURED (quota page; re-check) |
| Read turnaround | ~8-10 h typical; up to ~8 h extra queue seen | GUESS; queue MEASURED 09-24 |

---

## 2. Calendar 10-04 to 10-22

Wave A = 05:35-09:00 IST (00:05-03:30 UTC). Wave B = 14:00-21:30 IST (08:30-16:00 UTC), decided with whatever
reads have landed; never hold a slot past 16:00 UTC. V100 times assume f0_s42_v4 started ~22:45 on 10-03; if it
auto-launched ~00:30, add 1.75 h to every Lane 2 time before 10-05 (compute.md A1). Shared jobs run at ~1.8x
their alone time.

| UTC day | Slots (planned) | V100 lane 1 | V100 lane 2 | Kaggle GPU (T4-h) | Gates and decisions |
|---|---|---|---|---|---|
| **10-04** | **5.** A: K1 `submission.csv` (P+Cv1 W .25); K1 `submission_fleetsolo.csv` (filename test, crop solo, canary for the fail-safe K1); K1 `submission_w15.csv`. B: K2 `submission.csv` (P+fv1 at W*, default .30); K2 fleetsolo (G2) | f0_s42 mix50 to ep10 ~09:05; crop v4 all-data R50 s0 09:05-15:05 (queued tonight); G-1 preflight; EV2S f0 s142 (D1 labels) ~15:15 to 10-05 03:35 | f0_s42_v4 to ep10 ~10:45; F1 all-data s43 (D1 labels, `--stop_epoch P`) 11:00 to 10-05 03:00 | K1 (overnight) 0.3-0.5; K2 ~13:30 0.3-0.5. Day ~0.6-1.0. [verify] KT not booked (Eesh's avoid-Kaggle-GPU instruction; box mimic supplies parent gold58) | R0 at the 05:35 poll; D1 provisional 01:55 / 03:35, final ~10:45; G1, G1b; G-1; R1 as K1 reads land (~14:00-22:00). K2 ships only with the ported cell (section 5), else it moves to 10-05 A |
| **10-05** | **3-5.** A (0-2, by rule only): K1 `w20` if w15/w25 split; K2 `wlo` if K2 main d <= -1 with solo >= .915. B (3): K3 main (P+fv2 at W*), K3 `cf`, K3 `c` (P+Cv2). [verify] If K2 slipped to 10-05 A (2 slots), A holds only K2 main + fleetsolo and K1 `w20` waits for 10-06 A, so the day stays at 5. Send K3 `cf` / `c` only after K2 has at least one scored file (proves the ConvNeXt stage fits at N=1); K3 main is protected by per-file fail-loud either way | EV2S f0 to 03:35; F1 s44 03:35-19:35; F1 s45 19:35 to 10-06 11:35 | s43 to 03:00; crop v4 all-data s1 03:00-09:00; EV2S all-data s143 09:15 to 10-06 00:20 if G1 + G1c pass (else F1 s46) | K3 combined ~11:30, 0.3-0.5 | G2, G3 (K2 reads ~00:00-08:00); G1 / G1c for EV2S (~04:00); **G-T** (K2 microbench + K2 per-file outcomes; KT only if OKed [verify]) sets N for K3/K4; pack-time gold58 for s43, EV2S f0, crop s0/s1 |
| **10-06** | **3-4.** A (<=1): one K3 follow-up by rule. B (3): K4 main (P+fv3 at W*), K4 fleetsolo (solo of the final fleet), K4 `cf`. [verify] If G-T or K3's `cf` outcome says the sum does not fit, K4 is fleet-only (main, fleetsolo, `wlo`) and CF stays at K3's read: a 3-member CF file written inside a 5-member run saves no time, because the crop stage starts after all 5 members | s45 to 11:35; then J2 soup partner run (fold-0, same init as f0_s42, new data seed) | EV2S s144 00:20-13:50 (else F1 s46/s47); then J3 crop seed s2 (~3.5 h alone) | K4 ~15:30, 0.3-0.5. [verify] At the 80 min/epoch end of the measured range s144 ends ~15:30 and K4 slips to 10-07 A; keep 10-07 A free for it. All-data seeds stop at `--stop_epoch P`, so a P < 10 shortens every seed row | R4 (does C stay in CF?), R5 (fv2 step). Core fleet complete ~14:00 |
| 10-07 | 2-3: ladder (K4 at W*-.10; best P+C file not yet read). Update the standing pair | J2 run; then all-data seeds of the passing family | crop seeds, then all-data seeds | 0 (K7 0.3-0.5 only if a stage keeps failing) | Eesh sends merge partners the fv1/fv3 numbers |
| 10-08 | 2: ladder. Provisional selection maintained from here on | J2 soup eval on fold-0 OOF (decides how extra seeds ship); seeds | seeds; J7 decision | 0 | Main structures fixed (P+F, P+C, CF) |
| 10-09 | 1-2 or none | seeds | seeds or J7 | 0 | |
| 10-10 | 1-2 | seeds | seeds | quota refresh 00:00 UTC; K5 fv4 0.3-0.5, only if expansion is justified (G-T allows N > 5, or J2 passed) | |
| 10-11 | 2: K5 main + K5 solo (if K5 exists) | seeds | seeds | 0 | R5 for fv4 |
| 10-12 | 1-2 ladder | seeds | seeds | 0 | |
| 10-13 | 2-3 (merged: 3 to combined kernels) | seeds / J4 partner jobs | seeds / J4 | KM 0-1.0 | Last day to agree a merge |
| 10-14 | 2-3 | as above | as above | KM 0-0.5 | **Target: both picks already scored** |
| 10-15 | 2-3 | as above | as above | KM 0-0.5 | Merge closes 23:59 UTC |
| 10-16 | 1-3 | last NEW recipe start 09:00 IST | seeds | 0-0.5 | |
| 10-17 | 1-3 | last same-recipe seed start 22:00 IST (shared) | same | 0 | **Probing ends 23:59 UTC** |
| 10-18 | 0-2 confirmations | a seed started alone must start by 12:00 IST | | 0 | |
| 10-19 | 0-2 | checkpoints off the box 10:00; pack + laptop verify 12:00; dataset versions ready 15:00 | | KF <= 3 commits + 1 spare by 23:00 IST (1.2-2.0), ONLY if the final structure differs from an already scored kernel | **Code freeze 00:00 UTC** (after it, only seed or data swaps into frozen code) |
| 10-20 | A (<=3): KF candidates. B (<=2): retries or ladder. No new candidate after 12:00 UTC | free | free | 0 (spare re-commit only after an error) | |
| 10-21 | A: retries and unscored pick candidates only | | | 0 | **Final selection 12:00 UTC, latest 18:00 UTC (23:30 IST)** |
| 10-22 | retries of errored pick candidates only (must be scored by ~20:00 UTC) | | | 0 | Deadline 23:59 UTC |

**Totals.** Slots: ~40 planned of 95 (range 30-55). Unused slots are fine: extra candidates mainly feed the
winner's curse (MODEL: picking the top public of 5-10 different structures adds +0.0016 to +0.0022 of public
noise, about the size of the effects being compared). [verify] That is the slots lens's figure for N independent
candidates; risk.md simulates +0.0002 to +0.0004 for 8 candidates whose public errors are correlated (mostly W
variants). Truth lies between; the 0.5 posterior shrink (section 6) is the remedy in both cases. V100: core ~57
alone-hours to 10-06 14:00, then ~270 alone-hours of reserve to 10-17 (MODEL). Kaggle GPU: ~1.2-2.0 T4-h
before the 10-10 refresh for K1-K4 (+0.3-0.5 for K7 only if a stage keeps failing, +~0.7 for KT only on
Eesh's explicit OK) of ~17-20, 0.4-4 h after, ~2-5.5 h in total (MODEL) [verify].

**Branch S (the filename test fails, i.e. Kaggle only scores `submission.csv`).** Known by 10-04 Wave B from the
K1 fleetsolo submission (a CLI rejection is known at once; a scorer error a few hours later; if `submission.csv`
of K1 scores and fleetsolo errors, the name is the likely cause). Then every read is its own commit: edit the
main W or the main file, re-push (~0.4 T4-h each); solo reads come from a parent-free light notebook (~1 h
runtime, faster turnaround). Commits rise from ~8-10 to ~15-20 (MODEL), still inside quota, with SwaplY's quota as
reserve. Per-file fail-loud collapses to "no `submission.csv` means an error", the same principle. The slot
calendar keeps its counts.
[verify] Under Eesh's avoid-Kaggle-GPU instruction, Branch S should not simply double the commits: cap it at ~1
commit per day, drop the crop solo read and all W variants first (the whole W question is worth +0.0004, section
0 item 5), and keep only the main read per structure plus the fleet G2 solo. SwaplY's quota is used only with his
own agreement. The parent-free light notebook does not exist yet: build item 9 stages it. If K2 fleetsolo is
submitted before Branch S is known, its loss is one slot.

---

## 3. Tonight and 10-04 hour by hour

**Tonight (10-03 23:00 to 10-04 05:30), all Eesh:**
1. UI: select the standing pair deploy_final 56753683 + deploy_B 56754330 (different base and member mode).
   Both are ~0 medal value; the pair only stops auto-selection from taking a pure repro on a 0.943 tie. Tell
   SwaplY that this plan uses all 5 slots of 10-04.
   [verify] "Tell" should be "agree with": the 5 slots are shared. If SwaplY needs a 10-04 slot, K1 `w15` moves to
   10-05 A first. (The 0.23 status-quo figure in section 0 comes from policy_sim's prior, deploy_final true delta
   ~N(+.0003, .0005); the risk lens treats deploy_final as ~0. Same conclusion: neither pick carries medal value.)
2. Box: `scp` `SP/teacher_v2_out/teacher_v2_native384_v8_shard0.csv` and `SP/fleet_g1b.py`. Note the real start
   time of f0_s42_v4. Queue on lane 1 "after f0_s42 exits: crop v4 all-data R50 s0" (`box_finetune_aug_mix50.py
   --labels llm_labels_v4_blend.csv --strat_labels llm_labels_v4_blend.csv --fold -1 --seed 0`, `_last.pt` rule
   of LANE_A_RUNBOOK). Copy `SP/fleet2/code` to `~/fleet2_code` and cache the EV2S weights (SCHEDULE 4a, no GPU).
   [verify] HANDOFF (23:45) says `~/fleet2_code` is already staged with the fixed code and the EV2S/R50 weights
   cached; only re-copy if `SP/fleet2/code` changed after that, and confirm with the SCHEDULE 4a sha check.
3. Members dataset version with the Cv1 manifest "ready", then push K1 (`SP/cropfleet/kernel/rsna-sub-cropfleet`).
   Submit nothing from it until the log shows `RESULT OK fleet blended: 3 member(s)` and all six files.
   [verify] `cropfleet/README.md` sections 1 and 3 are stale (mixSA member set, W .35 main, w60 file). The built
   kernel is the v4 set (cropaug_f1, cropaug_f2, crop_f1) with `W_MAIN` .25 and files w15/w20/w35/w45/fleetsolo
   (checked in the notebook: `'W_MAIN': 0.25`, `'K': 8`). Follow this plan, not README section 3, for which files
   to send. The expected log line names those three members.
4. Merge replies, if not posted yet (`Downloads/rsna-knee/merge_outreach.md`).
5. Optional at 01:55 / 03:35: `fleet_g1b.py` on each run's epoch-4 OOF. Kill one fold-0 run early only if it is
   BOTH >= 0.005 lower on gold58 and >= 0.03 higher in rho vs the v8 arm; otherwise both finish (the loser can
   never ship anyway: both carry the key `f0_s42`).

**R0, the 10-03 batch (apply at the 05:35 poll):**
- Any ERROR among the old-cell runs: a runtime alarm (P near the top of its range). No further old-cell variants
  in any case.
- crop25 / crop15: read through `risk.md` tables 6.1-6.2 (e.g. crop25 .944 means P(private > +.001) ~.40,
  .945+ ~.81, .943 ~.12; MODEL). Any old-cell read >= .944 is already a scored Pick-2 candidate. If both read
  <= .942, do not drop the crop leg: K=24 served frames the members never trained on (K=8 reads pending).
- v9 is the control for the parent's unknown place inside the 0.943 bucket: credit a crop .944 fully only if
  v9 reads lower (MODEL: P(> +.001) .56 if v9 = .942, .38 if .943, .35 if .944). Rebuild later kernels on the v9
  base only if v9 >= .944 AND v9crop25 >= crop25; otherwise never bundle v9.
- mixsacrop25 at least 0.001 above crop25 would contradict the teacher-correlation result (re-open mixSA crop
  members); anything else confirms: v4-label crop members only.
  [verify] Overstated: per risk.md S1 only mixsacrop25 >= 0.002 BELOW crop25 confirms; a tie (within 0.001) is no
  information (blend-delta SD .0012 plus display rounding). The v4-only call rests on the gold58 rho result
  (.775-.806 vs .830-.846) and stands either way; just do not log a tie as LB confirmation.

**10-04 day:**

| IST | Action | Who |
|---|---|---|
| 05:35 | One poll of submissions; R0; submit Wave A (K1: main, fleetsolo, w15). If the CLI refuses `-f submission_fleetsolo.csv`, do not send `w15` (same problem); keep that slot for Wave B | Eesh |
| 05:35-09:00 [verify] | Fallback missing above. If the K1 commit log is not verified (no `RESULT OK ... 3 member(s)` or a file missing) by 09:00, Wave A stays empty: K1 main + fleetsolo move to Wave B, K2 moves to 10-05 A, `w15` to 10-05 A. Never fill with old-cell variants. Wave A does not depend on R0, so a late 10-03 batch never blocks it; Eesh may submit on waking any time before 09:00 IST | Eesh |
| 07:00-12:00 | Port the fleet cell (section 5 item 1); add the gold58 subset filter to `localLB/build_kernel.py` (item 4) | Claude drafts, Eesh reviews |
| ~10:00 | Push KT (gold58-only parent commit) if Eesh OKs the ~0.5 T4-h. [verify] Not booked by default (section 0 item 3); item 4's filter is optional work | Eesh |
| ~10:45 | D1 final (section 4). Pack fv1 with the fleet2 pack tool, dataset `rsna-knee-cnxt-fleet` v1 | Eesh |
| 11:00 | s43 on lane 2 with the D1 labels and `--stop_epoch P` | Eesh |
| ~13:30 | Push K2 (ported cell only). Check `RESULT OK`, the microbench lines and that every file exists | Eesh |
| ~15:00 | Submit Wave B: K2 main + K2 fleetsolo. If K2 is not verified by 21:30: K2 moves to 10-05 A; B gets K1 `w20` only if w15/w25 split, else stays empty | Eesh |
| 15:05 | crop s0 done: pack-time gold58 check. Then `f2_preflight.py`; EV2S f0 with the D1 labels only on PASS | Eesh |
| evening | KT timings (if any) + K2 microbench into G-T (draft). K1 reads into R1. Rewrite HANDOFF | Claude / Eesh |

---

## 4. Pre-registered gates

What one read can resolve (MODEL, `slots_resolution_out.txt`, fleet .930 / rho .80, 390 public studies):
solo SD .0047 public; P+F delta SD .0012 (+ display rounding + the bucket position); paired subset difference
SD .00055; corr(public luck, private luck) at fixed true effect ~0.1.

| Gate | When | Input | Pass | Fail action |
|---|---|---|---|---|
| G-1 | before the first EV2S launch (~10-04 15:10) | `f2_preflight.py` (fp16 scan, 6 real steps) | PASS | EV2S dropped; R50 family 2 stays cut; F1 seeds take its slots |
| D1 | ep4 provisional; v4 ep10 final | gold58 (mean raw/EMA, last 3 epochs) as solo; per-label Spearman vs the real v8 arm on the 867 fold-0 val studies as rho | higher closed-form best-W lift wins; lifts within .0005, or both rho > .85: v4 | loser is analysis only. Do not use `fleet_g1b.py`'s AUC-vs-v4 as the solo (it favours the v4 run). [verify] The .0005 tie band is far inside gold58 noise: a .005 gold58 edge (under 1 paired SD; slice sampling alone moves ~.004-.007) already shifts the closed-form lift by ~.001. Rule: mix50 wins only if its lift beats v4's by > .001 AND its gold58 edge exceeds 1.5x the paired-bootstrap SD of the difference AND its rho is not higher on gold58 too; else v4 (the measured decorrelation prior). Epoch averaging cuts checkpoint noise, not the 58-study sampling noise. The 867-study rho is against an in-sample v8 arm (`*_full` checkpoints saw those studies), so require the same rho ordering on gold58 before it can override the v4 default |
| G1 | fold-0 ep10 | val AUC vs steven_v4; gold58 as a bug guard | >= .87 and gold58 >= .87 | both variants < .87: no s43, EV2S f0 is the family switch; if EV2S also < .87, close Lane F |
| G1b | same | rho vs v8 arm | <= .85 | later runs on v4 labels |
| G1c | EV2S f0 done (~10-05 04:00) | `g1c_corr.py` vs F1 fold-0 OOF on the same target | <= .86: s143 + s144 | .86-.90: one seed only; > .90: none, F1 seeds instead. [verify] G1c says only "different", not "better", and fleet2 SCHEDULE risk 2 says do not ship family 2 on G1c alone. Since the paired LB read is unreadable (SD .00055), add the offline analog: the F1+EV2S fold-0 OOF mean must give a higher closed-form lift (OOF AUC vs steven_v4 on the 867, rho vs the v8 arm) than F1 alone; else EV2S seeds go to F1 |
| G-T | 10-05 morning | P_hat from KT phase timings scaled to ~1,322 studies (fixed part from the 3-study commit, 242 s MEASURED); g = K2 microbench ms/member/study | N_max = largest N with 20 min decode + N x g + 5 min (+ 10 min if crop runs after it) <= 8.6 h - (P_hat + 0.25 h) | no KT: caps stay 5 fleet-only, 3 combined. P_hat <= 7.4 h: expansion to <= 8 members allowed (K5). P_hat >= 7.9 h: 3 fleet-only, 2 + crop combined (GUESS thresholds). [verify] The formula adds decode and GPU time serially, which contradicts compute.md's pipelined stage (~3 + max(D, N x g)) and the plan's own 5-member cap: as written it gives N_max 3-5 fleet-only (1-2 combined) at P_hat 7.65, 1-2 at 7.8 and 0 at 7.9, so a central KT result would have cut the fleet to 1-3 members. Corrected: N_max = largest N with 3 + max(20, N x g) + 5 (+ 10 if crop follows) <= 8.6 h - (P_hat + 0.25 h), in minutes. At g 3-4.5: P_hat 7.4 -> 10-16 fleet-only / 8-13 combined; 7.65 -> 7-11 / 5-8; 7.8 -> 5-8 / combined does not fit; 7.9 -> nothing fits at D 20 (lighter host K7, or D 15 if the microbench shows it). So the P_hat >= 7.9 row is wrong in kind: below the decode floor cutting N saves nothing; drop the crop stage first. Without KT, take P = 8.05 h (the top of the inferred range, no extra 0.25 h margin since it is already a bound): 5-8 fleet-only and no combined kernel at the top of the range, so K3/K4 `cf` files are a bet on P <= ~7.95 h and their outcomes measure the truth on the hidden set |
| G2 | K2 / K4 fleetsolo | LB solo | >= .922: full plan | .915-.921: all-data seeds only, re-judge at fv2. < .915: kill s43/s44; keep EV2S only if it passed G1/G1c, else close Lane F. [verify] G2 and G3 land together from K2. A weak solo with a good blend means low rho (risk.md 6.4), so kill on < .915 only if K2 main also reads d <= 0; a d >= +1 main with a solo < .915 keeps s43 and gets one confirming read at fv2 |
| G3 | K2 main | P+F at W* (default .30) | d >= +1: final-structure candidate | d 0: keep building if solo >= .925 (if solo < .925 and runs are on mix50, switch the next seeds to v4). d <= -1 with a fine solo: one `wlo` read. d <= -2: hunt a broken member. Two versions in a row <= .942 at their best W: stop F1 seeds |
| R1 | K1 reads | crop solo, w25, w15 | solo .915-.930 expected; solo > .930: crop W one row up | solo < .915: crop leg out of main structures (Pick-2 only via posterior). Crop W from `risk.md` 6.2/6.4: .15-.20 by default; .25 only if solo >= .924 AND w25 >= .945; never w35/w45. [verify] The out rule contradicts risk.md 6.4 ("a weak solo is not a reason to drop a leg that blends well"): out only if solo < .915 AND both w15 and w25 read <= .943; a w15 or w25 >= .944 keeps the leg whatever the solo. K1's `W_MAIN` .25 was set from the gold58 proxy peak (cell comment), which is noise-level vs .15 (section 1), so w15 is the default W* read, not a side variant |
| R4 | K3 / K4 | CF d vs P+F d; offline rho(C, F) on the 867 fold-0 studies | keep C in CF if CF d >= P+F d AND rho(C, F) <= .85 | C stays only as its own leg (P+C) for Pick 2. [verify] Cv2's all-data crop members trained on all 867 fold-0 studies, so rho(C, F) there is in-sample for C and biased upward. Use gold58 (both legs out of sample; noisy, so only a coarse check) or the v4 fold members' OOF where they exist, and let the LB comparison CF d vs P+F d carry the decision |
| R5 | each fleet version | main d vs the previous version | adopt unless 2+ below, or solo dropped >= .006 (hunt a broken member) | W* moves only if the solo moved >= .004 |
| R6 | K3 onward | crop W inside CF | crop .15 + fleet W_F with total <= .40 (risk sim used .15 / .25) | |
| R7-err | any rerun error | | re-submit the same file once | second error: structural, cut N (light prefix) |
| Pack | every member | gold58 >= ~.86-.87, `verify` / parity lines | | `--disable` or do not ship |

**Box reallocation the reads trigger:** G2 < .915 moves the V100 to family 2 + crop seeds; R4 "crop out" stops
crop training; solo >= .935 at fv2 or fv3 sends every free lane to all-data seeds of the best family (MODEL: +.005
solo is worth about +.0015-.002 of blend).

**Reserve list for the V100 from 10-06 14:00** (priority order; run only what the reads justify). [verify] Jobs renamed R1-R7 -> J1-J7: the R labels clashed with gates R1-R7 above (e.g. calendar "R4" meant both "does C stay in CF" and "merge-partner jobs"):

| # | Job | Alone time | Trigger | Note |
|---|---|---|---|---|
| J1 | F1 s45 + EV2S s144 replace the fold-0 members | in the core | G3 not failed | already in the calendar |
| J2 | Seed-soup test: fold-0 partner of f0_s42 with the SAME init seed and a new data seed; soup vs ensemble on fold-0 OOF | ~7 h | the window cannot hold N > 5 | Soups of models whose new layers (projection, attention, heads) had different random inits usually fail (GUESS from the model-soup literature), so the trainer needs an init seed separate from the data seed (section 5 item 6). Pass = soup within .002 OOF of the ensemble |
| J3 | crop v4 all-data seeds (R50 s2, cnxt224-v4) | ~3.5 h each | crop v2 read >= crop v1 | cheap to serve (K=8 crop stage ~5-10 min) |
| J7 | Solo-strength probe: folds 1-4 of the fleet, then mix its OOF into report-silent cells of steven_v4 | ~4 x 8 h | only if the fleet fold-0 beats its labels on gold58 by >= .01 (steven_v4 = .8927) AND the mixed table beats steven_v4 at target level on gold58 (paired bootstrap P > .9) | Archit's exact recipe, never tested in our ledger (TOP_TEAM skeptic). Own-model teacher, so it should not raise rho vs the parent (GUESS). Decide 10-08; last start of the student 10-16 09:00 |
| J4 | merge-partner folds or seeds | as needed | merge agreed | any partner leg must fit the same T4 window |
| J5 | F1 / EV2S seeds beyond 5 members | ~9 h each | G-T says N >= 7 fits, or J2 passed | otherwise the valve or the cap drops them |
| J6 | CoAtNet retry | ~7 h | only if written into the playbook with its own kill line before G2 reads | the most parent-like member we could build |
| cut | R50 family 2 | | | T4 cap, 1.69x memory, same arch as the parent's Rad heads and the crop leg |

A third concurrent job only for a crop R50 job, and only when `nvidia-smi dmon -s um -c 60` shows SM below ~85%
with two jobs running and more than 6 GB free.

---

## 5. Build items (laptop; Claude can draft, Eesh reviews and pushes)

1. **Fleet cell port, before K2 (critical path for 10-04 B).** In `fleet_blend_cell.py` (fleet2 copy):
   - multi-file writer from `SP/cropfleet/crop_fleet_cell.py:607-644` (tmp + `os.replace`, `submission.csv` last);
     files `submission.csv` (P+F at W*), `submission_wlo.csv` (W* - .10), `submission_whi.csv` (W* + .15),
     `submission_fleetsolo.csv`;
   - per-file fail-loud: a file is written only if every stage it depends on finished with the full member
     count; no parent copies anywhere (an absent file makes exactly the submissions that need it error, GUESS
     that Kaggle reports a missing selected file as an error; the commit log still shows everything);
   - margins: one 8.6 h process-age wall, start if >= 20 min are left (as coded it needs > 30 min with a 0.35 h
     margin and throws away a feasible fleet at P 8.05-8.15 h, MEASURED in code at lines 50-51);
     [verify] the lines are 49-50 of `SP/fleet2/code/fleet_blend_cell.py` (`left = (9.0 - _proc_age_h() - 0.35)
     * 3600`, `assert left > 1800`); the fleet2 copy is byte-identical to `round3/fleet` here (`diff -q` clean);
   - W is a literal in the cell (pushed kernels get no env vars; `make_parent_fleet_kernel.py` writes
     `RSNA_FLEET_W` as a literal, default 0.5): the port must hard-code W* per file, never rely on the 0.5
     default [verify];
   - commit-only microbench (<= 3 test studies): 5 forwards per member per GPU, log ms/member/study; also log
     the decode s/study on the 3 studies (D is the other G-T input) [verify];
   - rerun the round3 dry suite S1-S5 and the fleet2 t5 serving test.
2. **Combined kernel builder, before K3 (10-05 ~11:00).** P, then the ConvNeXt stage, then the crop stage; each
   stage owns its files; `submission_cf.csv` needs both; files: main P+F, `cf`, `c` (P+C), `fleetsolo`,
   `cropsolo`, `wlo`. A crop overrun can then never void the P+F file.
3. **Crop cell fail-loud patch, before K3.** K1 ships as built (fail-safe, parent restore), because it is built,
   verified and urgent; its fleetsolo read is the canary for that commit.
4. **KT:** gold58-only subset filter in `SP/localLB/build_kernel.py` (~5 lines; 58 > 48 studies, so the CoAt
   readers run one at a time like the hidden rerun). The first cell already raises on a competition rerun.
   [verify] Optional, off the critical path (section 0 item 3). If ever pushed, the README's smoke run is listed
   as required; a 58-study run is small enough to be its own smoke only if Eesh accepts that explicitly.
5. **Re-key rule:** if both fold-0 label variants ever ship, re-seed one (both use key `f0_s42`).
6. **J2 trainer option:** init seed separate from data seed (model_sha untouched; trainer only).
7. **Pick calculator:** post = m + 0.5 x (d - m), converted to P(private > +.001) with the `risk.md` read
   tables, so each wave updates the standing pair mechanically.
8. HANDOFF rewritten after every wave.
9. **[verify] Branch S light notebooks (staged, not pushed):** parent-free fleet-only and crop-only solo
   notebooks (the fleet cell or crop cell over the test set, `submission.csv` = solo). Branch S relies on them
   and none exists (fleet2 SCHEDULE section 5 also notes "fleet-only kernel NOT drafted"). One commit each, only
   if Branch S fires.
10. **[verify] Cv2 fallback:** K3's `c` file (P+Cv2) runs after the ConvNeXt stage, so a high P voids it. If K3 `c`
    errors, the next crop read is K1 rebuilt with the Cv2 manifest (crop-only kernel, one commit), not another
    combined kernel.

---

## 6. Final-pick policy (pre-registered)

**Eligible** (all must hold):
- E1: scored (not pending, not errored). One hidden rerun scores both splits (rules.md 199(a)), so a scored read is
  a scored private result.
- E2: not a fallback. Automatic for per-file fail-loud kernels. For K1 (fail-safe): its w25/w15 reads count
  fully only if K1's fleetsolo read is not 0.943; otherwise apply a haircut. Old deploy cell: only exceptions fall
  back. [verify] Each submitted file is its own hidden rerun, so a scored fleetsolo proves the crop stage fit in
  THAT rerun only. It is strong evidence for the sibling files (same code, ~5-10 min stage in a 36-55 min window),
  not proof: a K1 W file reading exactly 0.943 keeps a small P(copy) (risk.md F3: <= 5%) in its haircut.
- E3: not pure repro_0946, not a public notebook's output, never ranked on gold58.

**Rank:** posterior per candidate, post = m + 0.5 x (d - m), where m = model delta at the candidate's weights
(A1/A3 tables with the latest solo reads and offline rho) and d = displayed delta; W variants of one structure
share the structure's average d. Convert to P(private > +.001) with `risk.md` section 6. Tie-break: lower W, then
the leg with lower rho vs the real v8 arm.

**Pick 1** = top-ranked eligible candidate.
**Pick 2** = top-ranked eligible candidate whose leg set differs from Pick 1's (crop vs fleet; CF vs F or C is
allowed; another W of Pick 1 is not; MODEL private-delta correlation: W variants .97-1.00, crop vs fleet ~0, CF vs
F .82, CF vs C .45). Within 0.05 of each other, prefer the candidate sharing fewer legs with Pick 1. If no other
leg is eligible, take Pick 1's structure at the other side of W only if its posterior is within 0.05; else the
best other-leg candidate even at a lower posterior.

**If merged:** Pick 1 = the merged team's best public blend; Pick 2 = our best unmerged structure.

**Standing pair:** selected tonight, updated within 24 h of each read, changed only if the new pair's model bronze
improves by >= 0.03 (avoids churn on noise). Both teammates agree on every change. Final selection 10-21 12:00 UTC
(latest 18:00 UTC). On 10-22 a retry may replace a pick only if it is the errored version of the pre-registered
pick.

**Never:** pure repro; two W of one leg as the pair; an unscored or pending file; the raw top public read; a
structure read once at d <= -1; anything chosen on gold58.

---

## 7. Failure modes that matter most

| # | Failure | Visible? | Likelihood | Fix in this plan |
|---|---|---|---|---|
| 1 | Fleet stage silently cut or restored after a slow parent | no (reads 0.943) | GUESS 20-70% for a full 10-member fleet as coded | cap N, G-T measurement, retuned margins, per-file fail-loud |
| 2 | Hidden-rerun timeout | yes (error) | MEASURED 3 of ~10 member-add kernels; GUESS 15-35% per old K=24 member run | K=8 crop cell, ConvNeXt stage first, no old-cell variants |
| 3 | Display bucket: every read shares the parent's place inside 0.943 | partly | MODEL: a true +.0002 reads .944 20% of the time | v9 control; posterior shrink; never the raw max |
| 4 | Teacher-correlation trap | via G1b | MEASURED rho +.04-.05 with teacher-mixed labels | v4 labels unless D1 says otherwise; G1b |
| 5 | Old crop reads understate the member (K=24 served, K=8 trained) | no | systematic | do not kill the crop leg before the K=8 reads |
| 6 | Filename not accepted | yes | unknown (rules say `submission.csv`) | Branch S |
| 7 | Selection mishap (auto-select, teammate overwrite, selecting a pending file) | yes if checked | low, total cost | standing pair tonight; both teammates confirm |
| 8 | Late-queue failure near the deadline | yes | rises after 10-19 | no new candidate after 10-20 12:00 UTC; picks scored by 10-14 |
| 9 | GPU quota exhaustion | yes | low under this plan | one commit, many files; keep 3 h in reserve to 10-20 |
| 10 | Push order: kernel pushed before the dataset version is ready | yes (commit log) | low | read the log line before every submit |
| 11 [verify] | V100 unreachable (it has been before: playbook "when the box is back") | yes | GUESS low per day, but one outage before 10-06 hits the whole core | No Kaggle-side training fallback (T4 ~2 h/epoch, measured infeasible). Ship what is already packed (fv1/fv2) plus the crop leg; copy each checkpoint to the laptop at pack time and out of Temp (HANDOFF note), not only on 10-19 |
| 12 [verify] | New per-file cell has a systematic bug (K2 port, K3 builder, both new code within ~24 h) | yes (missing files in the commit log, or every file erroring) | GUESS low-moderate | the 3-study commit log check catches deterministic bugs; a bug that only shows at scale costs the K2 and K3 slots (~5), which slots can afford, but then fall back to K1-style fail-safe cells rather than a third new design |

---

## 8. What Eesh must do himself

- Every Kaggle push (K1, KT, K2, K3, K4, K5, KM, KF, K7), every dataset version (crop members manifest,
  `rsna-knee-cnxt-fleet` v1-v4), and every submission, including the 10-04 filename test.
- Approve the Kaggle GPU spend: ~2-3 T4-h this week; KT (~0.5 h) is the only optional item and needs his explicit
  OK. Without KT the fleet caps stay at 5 / 3 and the box mimic supplies the parent gold58 predictions.
  [verify] Default is now no KT: ~1.2-2.0 T4-h this week (K1-K4), matching his 10-03 21:30 instruction. The
  box mimic still needs a CPU-only Kaggle kernel to pack the gold58 DICOMs (no GPU quota) plus the T4-shim build
  on the box (boxmimic/INVENTORY.md section 5); that is his box work, not Kaggle GPU.
- Select and update the final pair in the UI, and agree each change with SwaplY; coordinate the shared 5 daily
  slots with him (this plan uses all 5 on 10-04 to 10-06). [verify] The calendar uses 5 on 10-04, up to 5 on
  10-05 and up to 4 on 10-06; SwaplY's quota is a reserve only with his agreement.
- Box work: queue jobs (the `auto_fold0.sh` wait-for-PID pattern so lanes never idle overnight), `scp` files,
  run `fleet_g1b.py`, `g1c_corr.py`, `f2_preflight.py`, kill or launch runs at the gates. Only his own processes:
  the harness blocks killing other users' jobs.
- Read every commit log for the `RESULT OK` lines before submitting.
- Merge: post the replies, agree by 10-13, run the team-leader mechanics by 10-15 23:59 UTC; share public scores
  and approach only before a merge (rules line 144).

---

## 9. Merge (unchanged in substance)

Two-way only (rules 143(ii) check the team cap at the request date, ~330 on 10-03 plus 5 per day; a three-way
fails). Drafts in `Downloads/rsna-knee/merge_outreach.md`, which lead with the label work the Myo team says it is
stuck on. Send the fv1 G2/G3 numbers and, by ~10-07, the fv3 numbers. After a merge the team still has 5 slots
per day: days M to M+2 give 3 per day to combined kernels (partner stack + our fleet, + C if it fits; P plus a full
partner stack will not fit in 9 h). Keep wave-A reads on our own structures until the merge is confirmed.

---

## 10. Lens conflicts and how they were resolved

| # | Question | Lenses | Call | Why |
|---|---|---|---|---|
| 1 | Third K1 file on 10-04 | slots: w45 (rho probe); risk: w15 + w25 bracket | **w15** | w35/w45 are never the posterior-best cell (risk 6.4); solo read + offline rho already pins W (slots 2.1) |
| 2 | Style of fail-loud | slots: write the parent with Fracture reversed (reads ~.87); risk: restore then raise | **per-file absent output** | Raise kills every file of a combined rerun; reversing a label deliberately encodes run state into the LB score, which risk.md lists under "Do not". An absent file errors only the submissions that need it |
| 3 | Fleet size | slots / SCHEDULE: 10 members; compute / risk: 4-5 | **5 fleet-only, 3 combined; expand only through G-T** | The T4 window after the parent, not the V100, binds |
| 4 | When to stack crop + fleet | slots: from K2; risk: only after each fits alone; compute: ~10-10 | **first combined kernel K3 on 10-05**, ConvNeXt stage first | K1 and K2 reads plus KT/microbench cover "fits alone"; per-file fail-loud protects the P+F file. [verify] Without KT, "fits alone" for the fleet = at least one scored K2 file; gate K3's `cf` / `c` submissions on that (section 2, 10-05) |
| 5 | Family-2 value read | compute: F1-only paired subset read | **offline only** (G1c, fold-0 OOF) | paired subset SD .00055 is under the display step |
| 6 | Canary reads per runtime shape | compute: a fleetsolo per shape | **only where the solo value is needed** (K1, K2, K4) | with per-file fail-loud a scored file already proves its stage ran |
| 7 | Parent-on-gold58 | compute: box mimic first, Kaggle C6 as fallback | ~~**KT on Kaggle first** (needs OK)~~ [verify] **box mimic first; KT only on explicit OK, after G2/G3, if expansion past 5 is on the table** | The original call overrode Eesh's 10-03 21:30 instruction (avoid Kaggle GPU; the Kaggle-side local LB is ON HOLD in favour of the box mimic). The T4 timing it buys mainly decides N > 5; per-file outcomes on K2-K4 measure "fits" at the real scale for a slot each, and the mimic costs ~0.5-1 h of V100 next to one job |
| 8 | Hisernberg kernel | slots: filler; risk: do not spend on it | **held; only as K7 lighter host** if a stage keeps failing | never scored, ancestors errored on the hidden set |
| 9 | Lock time | slots 12:00 UTC; compute 18:00 UTC | **12:00 UTC target, 18:00 UTC latest** | |
| 10 | v9 base | slots: v9 >= .944 means rebuild on v9; risk: v9 is the bucket control | **adopt only with paired agreement** (v9 >= .944 and v9crop25 >= crop25) | |
| 11 | First fleet W | playbook: W .50; risk / compute: .30 | **W* clipped to .20-.35, default .30** | F30 beats F50 in every prior (MODEL) |
| 12 | Label decision timing | slots: epoch 4 | **compute's rule: provisional ep4, final v4 ep10** | the ep10 information is worth more than an early lane; the loser cannot ship |
| 13 | EV2S labels | SCHEDULE 4c hard-codes mix50_logit | **D1 winner** | |
| 14 | Two-lane throughput | compute 1.1x; fleet2 review ~1.0x | **plan at 1.0x** | GPU-bound jobs time-slice. [verify] The calendar's shared durations (1.8x alone, i.e. 1.11x aggregate) come from the MEASURED 65-80 min shared epoch, so they stand; 1.0x only changes the alone-hour bookkeeping. At the 80 min end every shared row is ~10% later (section 2, 10-06 note) |
| 15 | Seed soups | compute: GUESS that soups hold | **same init seed required** | different random head inits usually do not average (GUESS) |
| 16 [verify] | Winner's-curse size | slots: +0.0016-0.0022 for top-of-5-10 structures; risk: +0.0002-0.0004 for top-of-8 mixed candidates | **both bounds kept; the 0.5 posterior shrink handles either** | slots assumes independent public errors (upper bound, right for different legs); risk simulates correlated ones (mostly W variants). Not listed before |

---

## 11. Open uncertainties (what would change this plan)

1. **Parent hidden-run time.** 7.7-8.05 h (MODEL from outcomes) vs ~7.0 h (localLB README figure, origin unclear).
   KT settles it; the answer moves the fleet between 3 and ~8 members.
   [verify] Without KT (default now), the K2/K3/K4 per-file outcomes settle the part that matters (does a 5-member
   fleet fit; does fleet + crop fit) on the hidden set itself. Under the corrected G-T formula (no extra margin) a
   fleet-only kernel holds 5-8 members at P 8.05 h and 10+ at 7.65 h, not 3; fleet + crop fits only below ~7.95 h.
   The 7.7-8.05 h inference still rests on the 2-member
   old-cell timeouts being runtime failures; if they were not, P ~7.0 h and the window is ~1.5 h.
2. **Filename acceptance.** Branch S if not.
3. **Fleet strength.** No LB read yet. The whole medal case for the fleet rests on solo >= ~.925 at rho <= ~.82
   (MODEL A1). G2 on 10-05 is the first real number.
4. **Member cost on T4.** 3-4.5 min per ConvNeXt member is a scaled benchmark; the K2 microbench replaces it.
5. **One rerun scores both splits.** Supported by rules 199(a); per-file fail-loud is safe only under it.
6. **Merge acceptance.** GUESS 15-25%.

---

## 12. Adversarial review verdict (10-04 ~00:15 IST, research only)

Every edit above is marked [verify]. Nothing was submitted, pushed, versioned, posted or run on the box.

**Verdict: GO for tonight and 10-04 Wave A as written (K1 main + fleetsolo + w15, standing pair, box queueing),
with four corrections that change decisions:**

1. **Constraint violation fixed.** The plan booked KT (a Kaggle-GPU parent commit) on the critical path and
   reversed compute.md's box-mimic-first call, against Eesh's 10-03 21:30 instruction. Default is now no KT;
   parent gold58 comes from the box mimic; "does it fit" is measured by per-file outcomes on the hidden set
   (a slot each, no quota). Kaggle GPU this week drops to ~1.2-2.0 T4-h.
2. **G-T formula was wrong.** It added decode and member time serially plus a margin, so a central P_hat would
   have cut the fleet to 1-3 members, and its ">= 7.9 h: cut N" row cannot help in the decode-bound regime.
   Replaced by the pipelined form; the real consequence is that fleet + crop in one kernel fits only for
   P <= ~7.95 h, while a 5-member fleet alone fits across the whole 7.7-8.05 h range.
3. **gold58 over-reliance trimmed.** D1's .0005 tie band sat inside gold58 noise (a pure-noise member reaches
   +.0029 at its gold58 argmax; slice sampling alone moves ~.005), K1's W .25 main was a gold58 argmax, and two
   rho checks (D1 on 867, R4 on 867) use in-sample predictions (v8 `*_full` arm; all-data crop members). v4 stays
   the default unless mix50 wins clearly on gold58 AND on rho; R4 uses gold58 rho and the LB.
4. **Gate logic vs risk.md 6.4.** R1 and G2 dropped a leg on a weak solo even when its blend read well; now a
   leg leaves only when solo AND blend reads are both weak.

**Also fixed:** 10-05 could book 6 slots if K2 slipped (now capped at 5); a 3-member CF file inside a 5-member run
saved no time; missing fallbacks (K1 not verified by 09:00, Branch S light notebooks, Cv2 read if K3 `c` errors,
box outage, a bug in the new per-file cells); reserve jobs renamed J1-J7 (they clashed with gates R1-R7);
quota line (teacher-v2 already done); "Tell SwaplY" -> agree with him; mixsacrop25 tie is not "confirmation";
stale cropfleet README section 3 flagged; fleet cell line numbers 49-50; W must be a literal per file.

**Numbers checked and left as written:** LB snapshot, bronze/silver model figures (risk.md), the closed-form lift
table (slots_tables A1), decode 0.68 s/study, 3-4.5 min per member, read-resolution SDs, V100 calendar arithmetic
(shared durations from the MEASURED 65-80 min epoch), slot totals (~28-51, ~40 planned).

**Still open, unchanged:** the parent's hidden-run time (7.7-8.05 h inferred vs ~7.0 h quoted), filename
acceptance (Branch S), and fleet strength (first real number at G2). The plan's medal case rests on G2/G3;
everything before 10-05 is plumbing that is now cheap to get wrong (one slot per mistake), which is the right
place to be.
