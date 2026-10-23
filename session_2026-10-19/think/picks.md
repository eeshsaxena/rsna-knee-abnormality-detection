# Final-pick lens for the PRIVATE board (10-19 ~18:30 IST, research only)

Nothing submitted, pushed, built or selected. No GPU, no box job. CPU numpy/scipy only (each script < 1 min).
Tags: MEASURED (read from an API snapshot, the Kaggle CLI or our files), CLAIMED (someone else says so), MODEL (our
arithmetic on stated assumptions), GUESS (an assumption with no data behind it).
Scripts and outputs: `session_2026-10-19/think/picks_work/` (file list at the end).

## 0. Bottom line
1. **Keep A = c4 57343435 and B = R6 57248084.** B adds about **+9 pts P(silver)** and **+8 pts P(bronze or better)** over
   c4 alone (MODEL base: pair .63 vs .54 silver, .82 vs .75 bronze+). R6 is the best or within ~2 pts of the best B in all
   22 scenarios I ran (`picks_work/matrix.txt`).
   - HW 57262115 is an equal alternative: -1 pt silver, +1 pt bronze+. It is better only if heliosli's per-finding table is
     truly good on private.
   - a25 is worse by about 4 pts. The near-identical variants are worse by 6-8 pts (pf10, fr65, pf50, c4a15), so do NOT pair
     c4 with them.
   - A must stay c4: HW + R6 without c4 drops to .53.
2. **The hedge works through variance, not through a better mean.** On private, c4 vs R6 is closer to a coin flip than the
   public chain suggests (MODEL):
   - true c4 - R6 = +2.9e-4 +- 3.9e-4, P(c4 truly better) .77;
   - P(R6 beats c4 on private) .34;
   - correlation of their private margins only .54.
   - The whole public climb R6 -> HW -> a25 -> c4 (~7.7e-4, MODEL ruler) is less than 1 public sd of that paired difference
     (8.9e-4, MODEL bootstrap).
3. **Biggest private risk = the S6fall cliff, which no existing pair hedges.** Every candidate carries S6fall .2.
   - P(c4 private < heliosli copy block) ~ **.21** (MODEL base; .14 to .34 across priors).
   - If that happens, the ~750 copy teams all pass us and we land at ~r1,000+ (no medal).
   - The pair cuts this to ~.14. A purpose-built "c4 without S6fall" B would only help if Eesh doubts S6fall, and it costs
     ~4 pts silver in the base case (sec 5).
4. **Noise scale (MODEL, gold58 placement bootstrap scaled to the report-rate prevalence):**
   - one entry's absolute macro AUC: public sd .0046, private sd .0030, public-to-private gap sd ~.0055 (mostly shared
     by all teams);
   - paired difference between our variants: 2-10e-4 public, x0.65 on private;
   - vs a different-lineage model: ~1.5-2e-3 private;
   - Fracture, Synovitis and Lateral OA carry most of it (per-label public sd .022 / .032 / .026).
5. **Silver needs our private margin over the copy block >= ~5e-4 (MODEL, range 4.4-6.2e-4 across field scenarios).**
   - c4's posterior private margin is 5.6e-4 +- 6.9e-4, so silver is close to a coin flip.
   - Each real +1e-4 on private is worth ~+4 pts P(silver) for the pair (+6 for c4 alone).
   - A **+3e-4 public probe win is worth only ~+0.1 to +0.9e-4 true** (shrinkage .05-.3), i.e. +0.5 to +4 pts. A real
     +3e-4 would be worth +11.5 pts.
6. **Slots:** under A15 (one switch of A, B fixed at R6), spending them on probes is roughly neutral to slightly positive
   for private:
   - more weight tweaks: +0 to +1 pt;
   - neutral new directions: +1 to +4 pts;
   - a genuinely new model leg: +4 to +11 pts.
   - Chasing public hurts private only if we stack public winners, re-probe exhausted knobs, or swap B. The public display
     race (.959) is worth nothing on its own.
7. **Action (Eesh): select c4 + R6 by hand before 10-22 23:59 UTC.**
   - If nothing is selected, Kaggle auto-picks the top 2 public (CLAIMED rule). That would pair c4 with a near-twin
     (fr100 / pf10 / a25), MODEL -4 to -8 pts silver.

## 1. What is measured now (10-19 11:00 UTC snapshot `think/work/api_lb_1019T_think.json`; scripts in picks_work)
- **Board** (MEASURED, `bands_now.py`):
  - N 6,209; c4 r199 / .958 (15 teams above us inside .958);
  - .959+ = r1-183 (.959 62, .960 40, .961 18, .962 22, .963 15, >= .964 26), .958 = r184-1401 (1,218 teams);
  - silver r310, bronze r620, gold r22 (.964).
- **Zones** (MEASURED, `wzone_ts.py`, `runs_list.py`): exact-tie runs (copy clusters, increasing submissionId) dominate
  r659-1401.
  - The W zone between us and the first big tie run grew 368 -> 459 teams in 20 h (+109 per day). It was 216 on 10-17.
  - Teams at >= .959: 141 (10-17 16:21) -> 168 (10-18 15:05) -> 183 now, i.e. +18 to +24 per day.
  - N +36 per day. Our rank r178 -> r199 in 20 h.
- **Frozen-team ruler** (MEASURED, `ruler.py`; teams whose best submissionId is identical in the R6 / HW / a25 / c4 /
  now snapshots since 10-16 08:45 UTC):
  - 11 frozen teams sit between R6 and HW;
  - 0 between HW and a25;
  - 5 between a25 and c4;
  - **0 above c4 inside .958**;
  - 10 in .959, 12 at >= .960.
  - c4 is above every frozen .958 team, consistent with unrounded ~.9587-.9589 and ~1e-4 short of .959 (MODEL).
- **Brackets today for our non-best entries** (MEASURED, `bracket.py`): a25 r215-232, HW r219-253, R6 r225-525.
- **Our entries** (MEASURED, `kaggle competitions submissions`, 98 rows, all COMPLETE):
  - .958: c4, a25, HW, R6, S6W, hw3, R7late, wr45, s615, fr65, a15, c4a15, pf50, fr100, pf10;
  - .957: c2max, c2, c3max, KC1, COMBO, S6F 57201874;
  - private scores empty, as expected.
- **Deadline projection** (MODEL, linear rates):
  - >= .959 ~ 258 teams, in-band above c4 ~ 40, so c4's public rank ~ r295-300;
  - public silver cut ~ r316 (N ~ 6,336);
  - so on public, c4 most likely stays just inside the silver line (by 15-25 places). Deadline acceleration could push it
    out. Public rank does not decide medals.

## 2. Public-to-private noise (MODEL, `picks_cov.py` -> `picks_cov.json`)
Method:
- The REAL gold58 serve predictions (R6 leg, Raptor r384 / r224 kernel serve, S6fall n42gpu) are blended exactly as in the
  c4 kernel. R7 OOF stands in for R7 in S6F / FLIP.
- Per label, I resample n_pos / n_neg studies from gold58's positives / negatives at public (390) and private (910)
  counts, under the report-rate prevalence (Fracture 27 public / 62 private positives) and the gold58 prevalence.
- Only the noise structure is used. Gold58 means are NOT used (the screen is discredited, sec 3).

| quantity | public 390 (report / gold58 prev.) | private 910 (report / gold58) |
|---|---|---|
| absolute macro AUC of c4, sd | .0046 / .0039 | .0030 / .0026 |
| public-to-private gap of one entry, sd | .0055 / .0047 (largely common to all teams) | |
| c4 - copy block (S6fall + tuning) | 8.7e-4 / 6.5e-4 | 5.6e-4 / 4.2e-4 |
| HW - copy (= S6fall add) | 7.7e-4 / 5.3e-4 | 5.0e-4 / 3.4e-4 |
| c4 - R6 | 8.9e-4 / 7.0e-4 | 5.8e-4 / 4.5e-4 |
| c4 - HW / c4 - a25 | 7.2e-4 / 5.2e-4 | 4.6e-4 / 3.4e-4 |
| fr100 - c4 (Fracture .65 -> 1.0) | 9.7e-4 / 5.3e-4 | 6.3e-4 / 3.4e-4 |
| pf10 - c4 (PF OA .5 -> .1) | 3.2e-4 / 3.3e-4 | 2.1e-4 / 2.2e-4 |
| c2max - c4 | 1.0e-3 / 6.1e-4 | 6.6e-4 / 4.0e-4 |
| c4 - S6F (R7 bundle, component swap) | 1.6e-3 | 1.0e-3 |
| c4 - single leg (R6 / Raptor / S6 alone): other-lineage proxy | 2.3-3.1e-3 | 1.5-2.0e-3 |

- Per-label sd of c4 (report regime, public / private):
  - Synovitis .032 / .021, Lateral OA .026 / .017, Fracture .022 / .015, PF OA .018 / .012, Lateral Meniscus .017 / .012;
  - every other label <= .009.
  - So about 3/4 of the macro noise is in 4 labels (matches metric.md, Hanley).
- **Cross-check vs our own reads (MEASURED):**
  - c2max (Fracture 1.0 + PF OA .1) fell at least ~9e-4 (.958 -> .957). That is ~0.9 public sd for that pair, so it fits
    "small true effect + noise". It does NOT fit the gold58 point of +3e-3.
  - Each step of the public climb (HW - R6 3.9e-4, a25 - HW 1.5e-4, c4 - a25 2.3e-4; MODEL ruler) is 0.3-0.5 sd.
  - Our probe record since 10-16 is 4 best-id moves in ~19 single-change reads (MEASURED, PREREG / submits.log). That fits
    reads dominated by noise.
- Binomial intuition (MODEL): with 27 public Fracture positives, one study flipping rank against ~360 negatives moves
  Fracture AUC by up to ~.003 and macro by ~2.5e-4. A single-label weight probe therefore reads mostly noise.

## 3. Posterior of our entries on private (MODEL, `picks_post.py`)
- **Knob model.** Each entry's true macro margin over the copy block is a sum of knobs:
  - s6 = S6fall .2 add;
  - th = heliosli table vs uniform;
  - a = A224 .35 -> .25;
  - the four c4 table moves;
  - Fracture, PF OA and Lateral OA steps;
  - S6 .3 / .15, W_RAPTOR .45;
  - r = R7 -> R6 swap.
- **Priors (GUESS):**
  - s6 ~ N(3e-4, 3e-4): an independent comparable model at .2 rank weight, prior positive;
  - th ~ N(0, 3e-4);
  - the other weight knobs N(0, 1.5-3e-4).
- **Conditioning.**
  - Joint Gaussian with the public-noise covariance above.
  - Exact conditioning on the MODEL magnitudes (HW - R6 3.9 +- 1.3e-4, a25 - HW 1.5 +- .5, c4 - a25 2.3 +- .8,
    R6 - copy 1 +- 1).
  - Then importance weights on every MEASURED inequality:
    - c4 > fr65, a15, c4a15, pf50, fr100, pf10, c2, c2max, c3max;
    - HW > hw3, s615, wr45, S6W;
    - R6 > copy;
    - R6 > S6F > FLIP;
    - fr100 / pf10 display .958; c2 / c2max / c3max / S6F display .957; FLIP displays .956;
    - c4 is below .959.
  - Then a fresh private draw (independent of the public noise, since the study sets are disjoint).
  - ESS ~2,500 of 30M (base).
- **Posterior (base), true margin over the copy block x1e-4 (mean +- sd):**
  - c4 5.6 +- 4.0, a25 3.9 +- 3.2, HW 3.7 +- 2.9, R6 2.8 +- 3.9, fr100 4.9 +- 4.9, pf10 2.6 +- 4.7, S6F 1.5 +- 6.2;
  - s6 3.7 +- 2.9 (public reads lift it only a little);
  - th 1.0 +- 2.8 (public says almost nothing about heliosli's table);
  - PF OA .1 step -3.0 +- 2.4 (P < 0 .91), Fracture 1.0 step -0.7 +- 2.8.
- **Private margin over the copy block:** c4 5.6 +- 6.9 (P > 0 .79), HW 3.7 +- 5.7 (.74), R6 2.7 +- 7.5 (.64),
  a25 3.9 +- 5.5 (.76).

| B (with A = c4), base | P(B > A on private) | mean B - A | sd B - A | E[max] - E[A] | corr | P(max > copy) |
|---|---|---|---|---|---|---|
| R6 57248084 | .34 | -2.9e-4 | 6.9e-4 | +1.55e-4 | .54 | .86 |
| HW 57262115 | .36 | -1.9e-4 | 5.4e-4 | +1.33e-4 | .65 | .87 |
| a25 57316395 | .34 | -1.7e-4 | 4.2e-4 | +0.95e-4 | .80 | .86 |
| fr100 57382011 | .46 | -0.7e-4 | 7.0e-4 | +2.41e-4 | .78 | .83 |
| pf10 57382016 | .17 | -3.0e-4 | 3.1e-4 | +0.28e-4 | .91 | .80 |
| S6F 57201874 (.957) | .36 | -4.2e-4 | 1.2e-3 | +2.98e-4 | .42 | .86 |
| c4 alone | | | | | | .79 |

## 4. Private rank model (MODEL / GUESS, `field_mc.py`)
**Field at the deadline, relative to the copy block's private score:**
- **Copy block:** ~750 teams at 0.
- **>= .959 teams** (~258): public margin from today's band counts plus projected arrivals.
  - About 60 % of .959, 40 % of .960, 25 % of .961 and 10 % above are "derivatives like us". These shrink like our own
    entries.
  - The rest keep their public margin minus 1.5e-4, with private sd 1.2e-3 and a common other-lineage shock of 2e-4.
- **In-band above us** (~40) **and the W zone** (~710):
  - 15 % "spread" teams up to c4's level, shrunk with prior N(2e-4, 4e-4) and public sd 6e-4;
  - 85 % near-copy tweakers within 3e-4 of the copy block, shrunk hard.
- **Below .958** (~710 teams in .957-.954): private sd 1e-3.
- **Max-of-2 bonus** of +.7e-4 for non-copy teams.
- **Medal cuts:** silver r316, bronze r633 (N ~6,336).

The checks behind these numbers (MEASURED today):
- the W-zone spread fraction matches the HW / a25 brackets;
- the public silver line sits ~2.5e-4 above the copy block.

**Result:** silver needs a private margin over the copy block of x* ~ 5.0e-4. Bronze needs ~1e-4: essentially "beat the
copy block" (`value_shift.py`).
x* moves 4.4-6.2e-4 across field scenarios: fieldBad 4.4, T1deriv / N6400 4.8, base 5.0, T1keep / fieldGood 5.6,
belowLoose 5.8, Wbig 6.2.

| (base our-side x base field) | silver | bronze+ | median rank | P(r <= 200) | P(> bronze) |
|---|---|---|---|---|---|
| c4 alone | .54 | .75 | r298 | .24 | .25 |
| **c4 + R6** | **.63** | **.82** | r262 | .30 | .18 |
| c4 + HW | .62 | .83 | r269 | | |
| c4 + a25 | .59 | .81 | r278 | | |
| c4 + fr100 | .62 | .79 | r256 | | |
| c4 + pf10 (near-twin) | .55 | .76 | r291 | | |
| c4 + S6W 57264835 | .62 | .81 | r262 | | |
| c4 + S6F 57201874 | .66 | .82 | r244 | | |
| c4 + c4noS6 (not built) | .59 | .82 | r280 | | |
| HW + R6 (no c4) | .53 | .77 | r304 | | |

Rank quantiles for c4 + R6 (p5 / 10 / 25 / 50 / 75 / 90 / 95): r130 / 146 / 185 / 262 / 430 / 1,703 / 1,846.
This is **bimodal**: the upper mode is the W-zone shake-up, the lower tail is the S6fall cliff. Gold ~0 everywhere (r22
needs ~+4e-3).

Robustness (silver / bronze+, `matrix.txt`). c4 + R6 across scenarios:
- our-side scenarios (field base):
  - skeptic S6 N(0, 3e-4): .46 / .70;
  - strong S6 N(5e-4, 2e-4): .72 / .89;
  - table real: .59 / .79; table overfit: .66 / .84;
  - knobs x2: .68 / .85; knobs x0.5: .58 / .79;
  - noise x1.5: .63 / .78; noise x0.7: .66 / .88;
  - gold58 prevalence: .66 / .87;
  - "R6 truly better than R7": .63 / .83.
- field scenarios (our side base):
  - T1 all keep margins: .60; T1 mostly derivatives: .64;
  - W 850 with 25 % spread: .56;
  - field additions as real as ours: .60; field probe-inflated: .66;
  - below-band noise 1.3e-3: .58; N 6,400: .64.
- Overall the pair's silver probability is ~.46-.72 (central ~.63) and bronze+ ~.70-.89 (central ~.82).
- In every row, B = R6 beats c4 alone by +5 to +12 pts silver.

**Why not the alternatives:**
- **HW:** ties R6 within ~1-3 pts. It wins only when the heliosli table is real (tabReal: .62 vs .59) and loses when the
  table is public-overfit (.62 vs .66). The public says nothing either way (th posterior ~ prior).
  - R6 differs from c4 on two tuned axes (table + A224). HW differs on one. R6 is also the A15 rule and was read-positive
    in its day.
- **S6F (.957, R7 bundle):** highest silver in the base case (+2.5 pts), because the component swap adds variance. This is
  fragile:
  - it falls below R6 when R6 is truly better than R7 (gold58 R6 .937 vs R7 OOF .919, MEASURED but OOF vs ensemble);
  - its noise is likely overstated, because R7 OOF is a single-fold model standing in for a served 4-fold + flip R7;
  - choosing a .957 entry over a .958 one is also hard to defend.
  - Not recommended, noted only as a sensitivity.
- **Near-twins (pf10, fr65, pf50, c4a15, a25):** private correlation .8-.9 with c4, so almost no hedge value.
- **"Both near-identical best entries":** the worst option.

## 5. The S6fall cliff and the only hedge that would address it
- Every existing candidate carries S6fall .2, so a bad private S6 leg hits A and B together. The pair still lifts
  P(beat copy) from .79 to .86, because R6 / HW diverge from c4 elsewhere.
- **Hypothetical c4noS6** (c4 table + A224 .25, no S6fall; one-line build, also saves runtime). As B (MODEL):
  - base case: .59 / .82, i.e. -4 pts silver vs R6;
  - skeptic S6 prior: .46 / .73, i.e. +3.4 pts bronze+ vs R6.
- **Verdict:** only worth a slot + new kernel if Eesh believes S6fall is as likely to hurt as help on private. The public
  evidence leans the other way:
  - S6F > FLIP best-id move (MEASURED);
  - S6 .15 and .3 both lost to .2 (MEASURED);
  - posterior s6 3.7 +- 2.9e-4.
- Rules hedge (GUESS-level, no action): R6 sits in both picks. The host ruled registration-only data OK (CLAIMED 10-14).
  The open fastMRI question (743416) would also hit the whole copy block. Watch the forum before selecting; nothing to do
  now.

## 6. What +3e-4 on the public LB is worth on private
- **A one-line probe that reads +3e-4 public** (shows .959, since c4 is ~1e-4 short):
  - E[true gain] = lambda x 3e-4, with lambda = tau^2 / (tau^2 + sd_pub^2);
  - tau 2e-4 and public sd 5e-4: +0.4e-4;
  - tau 3e-4 and sd 3e-4: +1.5e-4;
  - rare-label probe (sd 8e-4): +0.2e-4;
  - minus a winner's-curse when it is the best of 5 (`value_shift.json`, MODEL).
  - That is **+0.5 to +4 pts P(silver)** for the pair (+1e-4 private = +3.9 pts pair, +5.7 pts c4 alone).
- **A genuine +3e-4** (a real model gain): +11.5 pts pair silver, +7 pts bronze+.
- **Public rank only** (density.md MODEL): displaying .959 at the deadline is worth ~90-110 public ranks and ~0 private
  ranks beyond its true part.

## 7. Spend the remaining 15 slots? (`slots_mc.py`, MODEL)
Rule simulated: A = public argmax at the last read, B fixed at R6. Probe win rate calibrated to our record (0/9 around c4,
4/19 since 10-16).

| probe type | 5 probes: P(A switches) | E[true gain if switched] | d pair silver | d bronze+ |
|---|---|---|---|---|
| weight tweaks near the tuned optimum (true mean -1e-4) | .67 | +0.13e-4 | +0.8 pts | -0.1 |
| neutral new directions (mean 0) | .76 | +1.1e-4 | +3.1 | +1.5 |
| rare-label tweak (public sd 9e-4) | .76 | +0.6e-4 | +2.7 | +0.2 |
| a genuinely new model leg (mean +2e-4) | .88 | +3.8e-4 | +11.1 | +5.8 |

- **Chasing public does NOT hurt private if:**
  - A switches at most once per read to a single-change winner;
  - B stays R6;
  - we stop re-probing knobs the board has already answered. The PF OA .1 step is negative (posterior P .91), and
    Fracture / Lateral OA are noise-dominated.
- **It does hurt if:**
  - we stack two public winners into one A (c2 / c2max style); the 10-19 stacks lost a full display step;
  - B is swapped to a probe winner, which removes the hedge;
  - a new kernel's runtime risk is accepted for a weight tweak.
- **Recommendation:**
  - spend slots only on changes whose true mean is plausibly > 0: a new independent leg using the unused ~2.5 h, or more
    TTA. See the models / runtime lenses.
  - Do not spend them on more single-label weight probes. If no such change is ready by the 10-20 push, leaving the slots
    unused costs ~0 on private.

## 8. Recommendations
1. Final picks: **A = API best-id at the last read before 10-22 18:00 IST (today c4 57343435), B = R6 57248084.** This
   keeps A15. HW 57262115 is an acceptable B only if Eesh trusts heliosli's table.
2. **Eesh selects both by hand** (screenshot the checkmarks), and re-checks after any best-id move. Auto-selection would
   likely pair two near-twins (MODEL -4 to -8 pts silver).
3. If A switches to a probe winner X, keep B = R6. Do not use c4 as B: X and c4 would be near-twins.
4. No new kernel for picks alone. The only pick-motivated build (c4noS6) loses in the base case.
5. Use slots only for positive-mean levers. Each real +1e-4 private is worth ~+4 pts silver. That, not the .959 display,
   is the private lever.

## 9. Caveats and what would change this
- The noise model uses gold58 placements (58 studies, hand labels that disagree with the hidden labels on PF OA /
  Fracture). The noise levels fit the observed public deltas (sec 2), but per-label structure could be off.
- Field parameters are GUESSES: T1 composition, W spread fraction, private noise of other lineages, the max-of-2 bonus.
  They move P(silver) by +-.08. They barely change the ranking of B options.
- Not modelled:
  - non-additivity of knobs;
  - teams that probed per label on public with the 1-p trick (CLAIMED possible, forum 748011);
  - copy-block sub-clusters (R6 copies vs R7 copies, CLAIMED in PROGRESS_16), which would soften the cliff into steps.
- Would change the pick:
  - a host / forum ruling on R6 / fastMRI;
  - evidence that heliosli's table is real (then HW over R6);
  - strong evidence against S6fall (then build c4noS6 as B);
  - a new best-id whose change is a new model leg. Then A = it, and B stays R6.

## Reused partial results from the interrupted run (verified by reading the JSONs, MEASURED on gold58 / R7 OOF)
- `stack_ops.json`: no blend operator beats c4 significantly (flat_logit +.00088 P .60; others negative). So there is no
  new A candidate from existing predictions.
- `cross_label.json`: CV gains on R7 OOF do not transfer to R6 on gold58 (ACL .996 -> .995, MCL .964 -> .946).
- `tie_loss.py`: closed.
- The 10-19 morning stress test (`research/stress_work/final_picks_stress.py`) ranked c2max / c3max as A on gold58 means.
  The LB reads refuted that, and this lens replaces the gold58 means with LB-conditioned priors.

## Files (`session_2026-10-19/think/picks_work/`)
- `bands_now.py` / `.json`, `runs_list.py`, `wzone_ts.py`, `ruler.py`, `bracket.py`: snapshot structure (MEASURED).
- `picks_cov.py` -> `picks_cov.json`, `macro_*.npy`: noise covariance (MODEL).
- `picks_post.py` -> `post_<scenario>_<regime>.json / .npz`, `log_post_*.txt`: posterior + private draws.
  Scenarios: base, skepS6, strongS6, tabReal, tabOver, bigKnobs, smallKnobs, noise1.5, noise0.7, r6better; regimes report
  and g58.
- `field_mc.py` -> `field_<our>_<field>.json`, `log_field_*.txt`, `matrix.txt`: private-rank Monte Carlo.
- `value_shift.py` -> `value_shift_<field>.json`, `log_value_shift_base.txt`: the S(x) step, thresholds, value of shifts.
- `slots_mc.py` -> `slots_mc.json`: slot-spending simulation.
