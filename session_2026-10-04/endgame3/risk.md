# Endgame lens: RISK + FINAL PICKS (2026-10-03, 23:30 IST)

Research only: nothing submitted, pushed, versioned, posted, no ssh, no Kaggle API calls. Laptop CPU only.

Tags: **MEASURED** (LB read, ledger number or code read), **MODEL** (simulation below, assumptions stated),
**GUESS** (my estimate). `SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`.

Files (all in `SP/endgame3/`): `risk_sim.py` (joint simulation), `risk_eval.py` (policies, outputs `eval_opt/cen/pes.txt`),
`risk_calib.py` (anchor check), `risk_extra.py` (selection optimism), `risk_p0.py` (display-bucket confound).
Re-run: `python risk_sim.py cen 700 <seed>` (x6 seeds), then `python risk_eval.py cen`.

---

## 0. Bottom line

1. **Medal value is zero below about +0.0005 private, for a copy and for a loser alike** (MODEL, lens_rules Pcurves:
   exact copy P(bronze) 0.000; +0.0005 0.26; +0.0010 0.66; +0.0020 0.98 in the base field; creep+overfit needs about
   +0.002). So the "safe floor" is not a low-variance pick. A pure repro and a near-zero tweak (deploy_final, mean
   crop weight 0.033) are both about 0 medal probability. **Safe = the best VERIFIED posterior, not the lowest
   variance.** Never select pure repro_0946 (confirmed: it adds 0.00 bronze in every prior and every scenario).
2. **Pick 2 must carry a different leg than Pick 1.** W variants of one member are the same bet: private-delta
   correlation 0.97-1.00 (MODEL). Two W of the crop member: bronze 0.28 (central prior) vs 0.24 for one; crop + fleet
   legs: 0.68; best pair with a combined leg: 0.74-0.76.
3. **Many W variants do not overfit much, they waste slots.** Picking the max public of 5 W of one member has
   optimism +0.00025 (a quarter of a display step) and is slightly *better* than a fixed W (MODEL). Top-1 of 8 mixed
   candidates: optimism +0.00016 to +0.00037. The real public trap is the display bucket (Section 6.3).
4. **Default W is too high for ~0.92 members unless rho is low.** In all three priors the a-priori best crop W is
   0.15-0.20 (central E priv: W.15 +0.9e-4, W.25 -5.4e-4, W.45 -33.9e-4). Given a read, 0.25 is best only when the
   solo read is >= 0.924 AND the W.25 read is >= 0.945. W .35/.45 are never the posterior-best cell. For the
   ConvNeXt fleet, **G3 at W 0.30, not 0.50** (F30 beats F50 in every prior: central +13 vs -11 e-4).
5. **Biggest silent risk: the fleet stage after the 0.943 parent.** `fleet_blend_cell.py` needs >= 30 min left
   (process age < 8.15 h when the parent finishes), then gives the fleet (8.65 h - parent end - 10 min). The parent's
   hidden-run time is not measured: ~7.0 h (localLB README, 19 s/study) gives a ~90 min window and the fleet fits;
   7.7-8.3 h (cropfleet README s4, inferred from which heavier variants timed out) gives 0-47 min vs a 35-60 min
   fleet-stage estimate. GUESS 20-70% of hidden reruns restore the parent silently for a full 10-member 320 px fleet,
   and that read (0.943) looks exactly like "the fleet adds nothing". Measure and fix before G3 (Section 4, F2).
6. **Make every new probe fail-loud on the hidden rerun.** A parent restore that reads 0.943 is indistinguishable
   from a null blend and is an exact copy privately. Restore-then-raise in rerun mode turns it into a visible error
   (one lost slot, recoverable), and every scored probe is then proof the blend ran on all ~1300 studies.
7. **The 10-04 reads mostly decide which world we are in.** P(crop25 reads >= 0.944): 0.47 optimistic prior,
   0.31 central, 0.04 pessimistic. A 0.944+ read is a likelihood ratio of ~8-12 against the pessimistic world.
8. **Lock a standing pair now and update it after every read** (manual selection on My Submissions, both
   teammates agree). Auto-selection takes the best public, which on a 0.943 tie can be the pure repro.

---

## 1. Model (what the numbers above rest on)

Joint binormal two-reader simulation (`risk_sim.py`, extends `round2/lens_rules/noise_calib.py`): 1300 hidden
studies, 390 public / 910 private, per-label prevalence from gpt56, parent macro AUC 0.943. Two own legs share the
study draw with the parent: crop member C (macro AUC m1, within-class noise correlation r1 with the parent) and an
independent fleet F (m2, r2), with C-F noise correlation r12. Candidates are rank blends:
`C15..C45` = parent + C at W, `F30/F50` = parent + F, `CF` = 0.60 parent + 0.15 C + 0.25 F, solo reads.
Displayed public = round(0.943 + p0 + dpub, 3), p0 ~ U(-0.0005, 0.0005) is the parent's unknown place in its bucket.
Private delta -> P(bronze/silver) through `round2/lens_rules/Pcurves.npy` (field model; base and creep+overfit).

**Calibration against measured anchors** (`risk_calib.py`, public delta of a rank blend, MODEL vs MEASURED):

| Anchor (MEASURED) | Model fits at effective rho |
|---|---|
| cnx 0.894 at W .4 / .5 -> -0.008 / -0.013 | 0.80-0.85 |
| Tian 0.927 own member at W .1 / .2 -> 0.000 / -0.001 | ~0.90 (W.15 -2.1e-4, W.20 -4.7e-4 at 0.91) |
| aastik ~0.927 private fleet at W .45-.6 -> +0.002 | ~0.80 (W.45 +22e-4 at 0.80) |

So "effective rho" (which absorbs shared label-noise errors, not just Spearman) runs from ~0.80 for an independent
fleet to ~0.90 for an own member that leans on the parent's family. Three prior sets:

| Prior | crop m1 / r1 | fleet m2 / r2 | r12 |
|---|---|---|---|
| opt | .912-.930 / .78-.90 | .915-.935 / .72-.86 | .45-.75 |
| cen (used below) | .912-.930 / .80-.93 | .915-.935 / .76-.90 | .50-.80 |
| pes | .910-.926 / .85-.94 | .912-.928 / .82-.92 | .60-.85 |

Limits: the model has no parent public-overfit term per candidate (the field curves carry it), treats labels as
independent across the 12 columns, and puts all non-sampling uncertainty into the priors. Use it for relative
decisions (which pick, which W, which order), not for absolute medal odds.

---

## 2. Final 2-pick policy

**Eligibility (all must hold):**
- E1. Already scored (no pending, no error). A scored run is the private run too: rules.md 199(a) says the hidden
  test holds both splits, so one rerun produces both scores. Never select on a commit log or gold58.
- E2. Verified non-fallback: from a fail-loud cell, OR from the old deploy cell (only exceptions fall back there,
  timeouts show as errors), OR its read differs from the parent's display. A fail-safe-cell read equal to 0.943 is
  eligible only with a haircut (treat P(copy) = the cell's fallback guess, Section 4).
- E3. Not a public notebook's output and not pure repro_0946.

**Ranking:** posterior P(private delta > +0.001) from the read tables (Section 6), not raw display. Tie-break:
lower W, then the leg with lower measured rho vs the parent's real v8 arm (G1b numbers).

**Pick 1** = top-ranked eligible candidate.
**Pick 2** = top-ranked eligible candidate whose leg set differs from Pick 1's (crop vs fleet; CF vs F or C is
allowed; a W variant of Pick 1 is not). If no such candidate exists, Pick 2 = the next W of Pick 1's leg only if its
posterior is within 0.05 of Pick 1's, else the best candidate of any other leg even at a lower posterior (it is
the only pick with independent upside).

**If merged (by 10-15):** Pick 1 = the merged team's best public blend; Pick 2 = our best leg blended into their
stack, judged by post-merge reads. Same eligibility rules.

**Portfolio numbers** (MODEL, central prior, bronze base / creep+overfit, silver base):

| Policy | Bronze | Silver |
|---|---|---|
| Status quo deploy_final + repro (playbook s6, policy_sim.py priors) | 0.23 / 0.02 | 0.00 |
| Repro + crop W .25 | 0.24 / 0.11 | 0.07 |
| Two W of the crop member (top-2 display) | 0.28 / 0.11 | 0.07 |
| Crop W .25 + fleet W .30, pre-registered | 0.68 / 0.46 | 0.37 |
| Best-display crop W + best-display fleet W | 0.68 / 0.46 | 0.36 |
| CF + best fleet W | 0.74 / 0.57 | 0.48 |
| Rule above (top, then other leg) | 0.75 / 0.56 | 0.47 |
| Top-2 display overall (all 8 candidates) | 0.76 / 0.58 | 0.49 |

Same ordering in the optimistic prior (same-member 0.45, crop+fleet 0.88, top-2 overall 0.94) and the pessimistic
one (0.03, 0.13, 0.15). "Top-2 overall" edges the rule only because CF and F are both strong in-model; the rule
keeps the two picks on different failure modes (a hidden-set problem in the fleet hits CF and F together), so use
the rule unless the top-2 are already on different legs.

Private-delta correlations (MODEL, central): W variants of one leg 0.97-1.00; crop vs fleet ~0.00; CF vs F 0.82;
CF vs crop 0.45.

---

## 3. Single candidates (no selection), MODEL

Bronze base, central prior (opt / pes in brackets):

| Candidate | E priv (1e-4) | P(>0) | P(>+.001) | Bronze |
|---|---|---|---|---|
| R pure repro | 0 | 0 | 0 | 0.00 (0.00 / 0.00) |
| C15 | +0.9 | 0.52 | 0.24 | 0.28 (0.45 / 0.03) |
| C20 | -1.6 | 0.46 | 0.24 | 0.27 (0.44 / 0.03) |
| C25 | -5.4 | 0.39 | 0.21 | 0.24 (0.40 / 0.02) |
| C35 | -17.0 | 0.25 | 0.14 | 0.16 (0.28 / 0.00) |
| C45 | -33.9 | 0.14 | 0.07 | 0.08 (0.16 / 0.00) |
| F30 | +13.2 | 0.70 | 0.56 | 0.58 (0.80 / 0.11) |
| F50 | -11.3 | 0.39 | 0.29 | 0.31 (0.51 / 0.01) |
| CF (0.15 C + 0.25 F) | +21.2 | 0.83 | 0.69 | 0.71 (0.93 / 0.08) |

Reading: the crop leg alone is a ~25% bronze ticket at best under the central prior; the medal mass sits in a second,
independent leg, and most of it in a blend that carries both legs at a modest total weight.

---

## 4. Failure modes

| # | Failure | Where | Visible? | P (GUESS unless tagged) | Cost | Mitigation |
|---|---|---|---|---|---|---|
| F1 | Hidden-rerun timeout, no score | old K=24 deploy cell (13 min glob + ~29 min loop per member in a ~36-55 min window, cropfleet README s4); any heavier leg | Yes (error) | 15-35% per 1-member K=24 run; MEASURED history: 3 of ~10 member-add kernels gave no score (56526314, 56690233, 56768269), all heavier variants | One slot | K=8 crop-fleet cell; never resubmit old-cell variants; the 4 old-cell 10-04 reads measure this rate |
| F2 | **Fleet stage silently cut or restored** | `fleet_blend_cell.py:50-53,86` (needs left > 30 min, `left = 9.0 - age - 0.35 h`, deadline `left - 600`, > half the models kept) | No (reads 0.943 or a weaker prefix) | 20-70% for a full 10-member 320 px fleet behind the 0.943 parent (parent time unmeasured: 7.0 h vs 7.7-8.3 h estimates) | Lane wrongly killed, or a disguised copy picked | Measure first, on T4 and legitimately: (a) a BENCH mode in the fleet commit that times decode + members on ~100 train studies (like the crop-fleet cell's `_cf_bench_gold`), (b) the `[phase] elapsed_seconds` lines of the localLB full run (458 studies, 2xT4) projected to 1322 studies; then size the fleet (<= 4-5 members, R5 "pixels before slices": 256 px or 16 triplets) so it fits in ~25 min after an 8.0 h parent; fail-loud in rerun mode; manifest order already puts the best members first |
| F3 | Silent parent restore in other fail-safe cells | crop-fleet (deadline, >2% decode failures, rho gate <0.30 on 50+ studies, OOM at scale), Hisern C2-C5 anchor copies, old deploy cell exceptions | No | crop-fleet K=8 <= 5%; Hisern variants unknown (Hisernberg v2-v5 errored on the hidden set, MEASURED hisern_ext.md:13) | As F2 | Fail-loud (restore then raise when `KAGGLE_IS_COMPETITION_RERUN` is set); commit-log check catches deterministic failures only |
| F4 | Partial fleet differs across reruns | time valve keeps a prefix; each submitted file is its own rerun | No | Tied to F2 | W variants of one version read different member sets | Paired reads only within one rerun-safe config; fail-loud below the full member count for the read that decides the fleet |
| F5 | Display-bucket confound | every read shares the parent's unknown p0 | Partly | MODEL: a true +0.0002 public delta reads 0.944 20% of the time, +0.0005 50% | A cluster of 0.944s with no real gain | v9 read as a p0 control (Section 6.3); require a crop read above the v9 read |
| F6 | Teacher-correlation trap | mixSA/teacher-mixed members (rho vs real v8 arm .830-.846 vs .775-.806, MEASURED) | Via G1b | Systematic | Blend lift halves (closed-form +0.0021 vs +0.0037) | v4-label legs; G1b <= 0.85; mixsacrop25 vs crop25 is the paired read |
| F7 | Train/serve mismatch | all 10-03 crop probes served K=24, trained K=8 | No | Systematic | Old crop reads understate the member | Do not kill the crop leg on K=24 reads alone while the K=8 probe is pending |
| F8 | Parent run-to-run variance | parent's own 8 h guard skips CoAt children, neutral fills | No | Low (repro scored 0.943 on several runs, MEASURED) | Noise in paired reads | Paired reads count only at >= 0.002 differences |
| F9 | Selection mishap | auto-select = best public (can pick pure repro on a tie); teammate overwrites; selecting a pending submission | Yes if checked | Low but total | Whole outcome | Standing pair selected now, updated within 24 h of each read, both teammates confirm; lock 10-21 |
| F10 | Late-queue failure | rerun queue up to ~8 h (MEASURED, 09-24), congestion near the deadline | Yes | Rises after 10-19 | Candidate unscored at lock | Last final-candidate submission 10-20 12:00 UTC; nothing new needed on 10-22 |
| F11 | Push-order and dataset-version mistakes | kernel pushed before the members dataset version is ready attaches the old version | Yes (commit log `RESULT PARENT RESTORED`) | Low | One commit (0.3-0.5 T4 h) | Read the commit log line before every submit |
| F12 | GPU quota exhaustion | 27.7 h left until the 10-10 00:00 UTC refresh minus teacher-v2 (MEASURED teacher_v2 README) | Yes | Medium this week | No new commits | One commit -> many files; keep 3 h in reserve for an emergency rebuild until 10-20 |
| F13 | Gold58 ranking | n=58, SD .01-.02; parent selection-biased on it | n/a | Systematic | Wrong pick | Gold58 is a bug guard only; never rank finals on it |
| F14 | Merge rules | two-way only (cap at request date), no private sharing before merging | n/a | n/a | Merge refused or a rules breach | merge_outreach.md as drafted; scores and approach only |

**Why fail-loud dominates for anything submitted from now on.** Fallback contamination at selection time is mild
in the model (per-version fallback f = 0.10: bronze 0.68 -> 0.62; f = 0.25: -> 0.55, central, identical for
fail-safe and fail-loud because a 0.943 read is rarely top). The real cost is upstream: a fail-safe fallback reads
like a null result, so the lane is dropped and the loss above is never recovered; a fail-loud error is fixed and
resubmitted, which recovers most of it. Assumption: Kaggle does not rerun selected submissions after the deadline
(one hidden test set, both splits scored in the same run). If that assumption were wrong, fail-loud would be
dangerous for final picks, so keep the restore-then-raise switch keyed to rerun mode only and leave everything
already scored untouched.

---

## 5. Decision rules by stage (each keeps a standing pair)

**S0, tonight.** Scored and eligible today: deploy_final 56753683 (0.943), deploy_A 56753685 / deploy_B 56754330
(0.942, older jiweiliu base), 56754549 (0.942, undescribed), repro 56750056 (excluded, E3). Standing pair until the reads land: deploy_final + deploy_B 56754330
(different base and member mode, so different failure modes). Both are ~0 medal value; this pair only exists to beat
auto-selection.

**S1, 10-04 reads (crop25, v9crop25, crop15, v9, mixsacrop25).**
- Any error among the 4 old-cell crop runs: F1 confirmed, no further old-cell variants.
- Classify crop25 / crop15 with Section 6.1-6.2; adjust with v9 (6.3). mixsacrop25 below crop25 by >= 0.002
  confirms F6; within 0.001 is no information.
- New standing pair: best-posterior crop read + the best candidate on any other leg (today only deploy_final or
  v9; v9 is a parent tweak, so it is a different leg).
- If both crop reads are <= 0.942: crop leg priority drops, but keep the K=8 probe (F7).

**S2, crop-fleet K=8 probe (one commit, 6 files).** Check the commit log for `RESULT OK fleet blended: 3 member(s)`.
Submit the bracket **w15 + w25** first (both are plausible finals; a final must be scored, so do not leave the
likely best W unsubmitted). Then w20 only if the bracket is split (one 0.944, one 0.943). fleetsolo is a
diagnostic: spend a slot on it only when two W reads tie (it moves the best W by at most one step, Section 6.4).
Never w35/w45 unless w25 >= 0.945 and solo >= 0.924.

**S3, ConvNeXt fleet G2/G3.** Before the first push: fix F2 (member count and pixel cost so the stage fits in
~25 min after an 8.0 h parent; fail-loud in rerun mode). G3 at **W 0.30**; F50 only if F30 >= 0.945. G2 (fleet solo)
needs its own light kernel, so it is optional; the paired reads already carry most of the information.

**S4, combined leg CF.** Build only when both legs have a verified read >= 0.943 and the runtime of the sum fits.
In-model it is the best single candidate (central bronze 0.71). Weights: crop 0.15 + fleet 0.25 as simulated, or
the posterior-best W of each leg scaled so the total stays <= 0.40.

**S5, 10-21 lock.** Apply Section 2. Last submission that may become a final: 10-20 12:00 UTC. 10-22: nothing.

**Standing-pair update rule (all stages).** After each read: Pick 1 := argmax posterior; Pick 2 := best on another
leg. Change the selection only if the new pair's model bronze (base) improves by >= 0.03, to avoid churn on noise.

---

## 6. Read tables (MODEL, central prior; opt / pes where it matters)

### 6.1 What one crop25 read means (old K=24 cell, member .912-.930)

| crop25 display | P(priv > 0) | P(priv > +.001) | Bronze base / c+o | Implied rho |
|---|---|---|---|---|
| 0.941 | 0.02 | 0.01 | 0.01 / 0.00 | 0.89 |
| 0.942 | 0.15 | 0.02 | 0.05 / 0.01 | 0.87 |
| 0.943 | 0.45 | 0.12 | 0.19 / 0.04 | 0.86 |
| 0.944 | 0.78 | 0.40 | 0.46 / 0.15 | 0.84 |
| 0.945+ | 0.98 | 0.81 | 0.82 / 0.48 | 0.83 |

Opt prior: 0.944 -> 0.52 bronze, 0.945+ -> 0.86. Pes prior: 0.944 -> 0.18, 0.945+ -> 0.20 (but the pes prior gives
a 0.944+ read only 4% probability, so such a read mostly refutes it).
P(read) by prior, as a world test: >= 0.944 opt 0.47, cen 0.31, pes 0.04; <= 0.942 opt 0.27, cen 0.34, pes 0.41.

### 6.2 Two reads of one leg (crop15, crop25) -> which W to keep

| crop15 / crop25 | E priv 1e-4: W.15 / .20 / .25 / .35 | P(>+.001) W.15 / .25 | Keep |
|---|---|---|---|
| .942 / .942 | -3.9 / -7.2 / -11.5 / -22.9 | 0.00 / 0.00 | neither (leg is dead at K=24) |
| .943 / .942 | -1.2 / -4.3 / -8.7 / -21.5 | 0.04 / 0.03 | neither |
| .943 / .943 | +3.1 / +1.5 / -1.4 / -11.2 | 0.11 / 0.09 | W.15, weak |
| .944 / .943 | +5.4 / +4.1 / +1.2 / -9.2 | 0.22 / 0.16 | W.15 |
| .944 / .944 | +9.2 / +9.1 / +7.5 / -0.3 | 0.44 / 0.40 | W.15 (or .20) |
| .944 / .945 | +13.3 / +14.5 / +14.3 / +9.2 | 0.71 / 0.65 | W.20-.25 |

### 6.3 The display bucket and the v9 control

A candidate with a true public delta d reads >= 0.944 with probability P(p0 + d >= 0.0005): d = +0.0002 -> 0.20,
+0.0005 -> 0.50, +0.0010 -> 1.00 (no sampling noise). Every read shares p0, so a batch of 0.944s can be one lucky
bucket. v9 swaps a measured no-op arm, so its read is mostly p0 (caveat: v9 solo 0.932 vs v5 0.928, so it may carry
a small real gain). crop25 = 0.944 means P(priv > +.001) = 0.56 if v9 reads 0.942, 0.38 if 0.943, 0.35 if 0.944
(central; opt 0.56 / 0.43 / 0.44). Rule: credit a crop 0.944 fully only if v9 reads lower.

### 6.4 Solo read + W.25 read -> best W (the fleetsolo file's value)

Across all (solo bucket, W.25 read) cells with n >= 25 (central): best W is 0.15 in 8 of 11 cells, 0.20 in 2,
0.25 in 1 (solo >= 0.924 and W.25 >= 0.945). W .35 and .45 are never best. A low solo with a decent blend read
implies low rho (good), so a weak solo is not a reason to drop a leg that blends well.

### 6.5 Fleet read at W 0.50 vs 0.30

| F50 display | Bronze (F50) | E priv F30 (1e-4) |
|---|---|---|
| 0.942 | 0.15 | +13.5 |
| 0.943 | 0.29 | +19.5 |
| 0.944 | 0.46 | +24.3 |
| 0.945+ | 0.82 | +38.7 |

Even when F50 reads 0.942-0.943, F30 is expected positive: a flat or slightly negative W .5 read must not kill the
fleet (central prior; pes: F30 still +4.9 to +15.0 while F50 is -6 to -21). Hence G3 at W .30 first.

---

## 7. Do not

- Select pure repro, two W of one leg, an unscored or pending candidate, or anything ranked on gold58.
- Read a 0.943 from a fail-safe cell as "no gain" without checking that the cell could not have fallen back.
- Submit w35/w45 (or F50) before the lower-W read of the same leg is in.
- Stack the fleet and crop legs in one kernel before each fits alone with margin (cnx-crop-fast timed out, ref
  56768269, MEASURED).
- Spend Kaggle GPU on Hisern C2 as a final candidate: never scored, built only from public checkpoints (a small,
  low-variance delta, so low medal value even if it reads 0.943), and its ancestor versions errored on the hidden set.
- Use submission scores as a channel for anything about the hidden run. Runtime is measured on train studies
  (F2 mitigation), not through the leaderboard.
