# LB + private-split lens (2026-10-17 ~22:00 IST)

Read-only. New snapshot: `research_1hr/api_lb_1017T_lens.json` (fetched 16:24 UTC = 21:54 IST). Scripts: `research_1hr/lens_lb_scripts/`
(traj.py bands/trajectory, ties.py tie clusters, zones.py zone profile, bc.py B bracket, mc.py private-rank Monte Carlo).
Tags: MEASURED (from API snapshots), CLAIMED (forum/notes), MODEL (simulation/derivation), GUESS (assumption).

## 0. Bottom line
1. **Keep the A15 pair: A = API best-id at the last read before 10-22 18:00 IST (today HW 57262115), B = R6 57248084.** The second pick is
   not wasted: MODEL corr of the two private margins ~.8, and B adds ~+4 pts P(silver) in the base case, +9 if heliosli's weights are public
   overfit, 0 if they are real, never negative (team score = max of the two).
2. **The private outcome is bimodal.** About 600 teams (r388-995) hold near-identical heliosli-copy predictions (MEASURED tie runs). A beats
   that block only through the S6fall .2 leg. If S6fall holds on private, A lands ~r150-350 (silver likely); if not, A drops below the whole
   block to ~r850-1000 (no medal). MODEL: P(A beats the block) ~.86 (base), ~.67 (skeptic prior on S6fall).
3. **P(silver), MODEL base: A alone ~.72-.74, B alone ~.62-.64, pair ~.76-.78; P(bronze or better) ~.86-.89.** Range across scenarios: pair .53-.85.
   No pair can hedge the S6fall cliff (both picks carry S6fall; a no-S6 entry would just be a copy of the block).
4. **The denominator is NOT fixed (MEASURED):** N grew 6,047 (10-16 00:07 UTC, right after entry closed) to 6,144 now, ~2-3 teams/h, because
   pre-registered teams still make first submissions. Projected N 6,250-6,400 at 10-22, silver cut r312-320, inside the .958 band (MODEL).
5. **B's public rank today is r298-322 (MEASURED bracket)**, right on the public silver line; A is r171. If Eesh does not select, Kaggle
   auto-selects the top-2 public (CLAIMED Kaggle rule), which would pair HW with another heliosli-weights entry and drop the weights hedge:
   **select both picks by hand.**

## 1. Leaderboard state and drift (MEASURED, api_lb snapshots, ranks by unrounded score)
| UTC | N | us | .959 band | .958 band | .957 band | silver cut r (score) |
|---|---|---|---|---|---|---|
| 10-14 05:27 | 5,956 | r449 .956 | r56 +16 | r72 +33 | r105 +85 | r297 (.956) |
| 10-15 05:44 | 6,013 | r352 .956 | r67 +15 | r82 +41 | r123 +146 | r300 (.956) |
| 10-15 08:24 | 6,032 | r158 .957 (S6F) | r67 +16 | r83 +44 | r127 +167 | r301 (.956) |
| 10-16 06:22 | 6,065 | r147 .958 (R6) | r72 +34 | r106 +99 | r205 +393 | r303 (.957) |
| 10-16 15:17 | 6,094 | r135 .958 (HW) | r78 +42 | r120 +395 | r515 +329 | r304 (.958) |
| 10-17 16:24 | 6,144 | r171 .958 (HW) | r94 +48 | r142 +854 | r996 +223 | r307 (.958) |

- Gold cut r22 = .963 since 10-14 (MEASURED). Teams at >= .959: 81 (10-15) -> 119 (10-16 15:17) -> 141 now, ~+22/day (MEASURED).
- Our drift since the HW move: r135 -> r171 in 25 h (+36: 22 teams climbed to >= .959, 14 passed us inside the band); last 14 h +22, no
  slowdown yet (MEASURED). Projection if HW stays best: public r250-350 at 10-22 23:59 UTC (MODEL: 1.4/h now, halving every 2 days gives
  +~80; constant rate gives +~180). The public silver line is at risk too, but the private LB decides.
- Silver cut stays in the .958 band to 10-22 (MODEL: moving it to .959 needs ~175 more teams at >= .959; ~+115 at the current pace).

## 2. Structure of the .958 band (MEASURED via tie runs + frozen teams)
Kaggle breaks exact ties by earlier submission, so an exact-tie cluster shows up as a run of consecutive ranks with increasing submissionId
(ties.py). Profile (zones.py, current snapshot):

| zone | ranks | n | median subs | reading |
|---|---|---|---|---|
| >= .960 | r1-93 | 93 | 107-168 | heavy iterators |
| .959 | r94-141 | 48 | 106 | iterators |
| V (in band, above us) | r142-170 | 29 | 78 | distinct scores |
| **A = HW** | r171 | | | |
| W (distinct scores) | r172-387 | 216 | 41-50 | random order, all best sids after 57251394 = heliosli-derived variants |
| **B = R6 (if it were our best)** | r298-322 | | | bracket below |
| C (tie runs) | r388-995 | 608 | 24-26 | long increasing-sid runs (up to 41), 100 % of best sids after the heliosli publish (10-15 ~17:20 UTC) = copy block |

- **B bracket (bc.py):** teams above R6 at its last visible read (10-16 08:45 UTC) can only have risen; the lowest of them is now r297. Frozen
  teams (same best id) that were below R6 then are now at r323-964. So B sits between r297 and r323 today, above the copy block (MEASURED).
- So on public: A - C spans 217 W-zone ranks, A - B ~139, B - C ~78. Under additivity (MODEL) the heliosli weights' public gain on the S6
  base is ~64 % of the S6fall gain on the weights base. A - C is exactly the S6fall .2 leg (+J decoder): heliosli's notebook = V224 + R6 + flip
  + per-finding weights, no S6fall (CLAIMED, PROGRESS_16 line 110, pulled notebook). Any public overfit in the weights is SHARED by A and C and
  cancels in that comparison.
- Caveat (CLAIMED, PROGRESS_16 line 91): the public "r6-frozen" dataset later held R7 under the R6 name, so C may be two sub-blocks (R6 copies,
  R7 copies) with slightly different private scores; that softens the cliff into two steps of ~300 teams each.

## 3. Noise of the public-private gap for small macro-AUC differences
Paired difference d between two blends; public n ~390, private ~910, disjoint (CLAIMED leaderboardPercentage 30).
- sd_priv(d) = sqrt(390/910) sd_pub(d) = .65 sd_pub. Gap (d_priv - d_pub) sd = sqrt(1 + .43) sd_pub = 1.20 sd_pub (MODEL, binomial-style scaling).
- sd_pub by change class: 1-line weight tweak 1.5-2.5e-4 (MODEL, lbprobe/sim3_07x: .2->.15 1.3-1.5e-4, .2->.3 2.3-2.5e-4); add/remove a .2 leg
  4.7-5.2e-4 (MODEL, sim3 .2->0); component swap ~1.5e-3 (gold58 paired SD ~.004 MEASURED on 58 studies, scaled by sqrt(58/390), MODEL).
  Hidden-rerun noise ~4-5e-6 (MODEL, sim_noise) is negligible. Single-model public SD ~.003 (CLAIMED 748011), consistent with Hanley-McNeil
  for 12 labels at n 390 (MODEL).
- P(private sign = public sign | observed d_pub), flat prior (MODEL; subtract selection bias first if d was a max over K tries):

| d_pub | weight tweak (sd 2e-4) | .2 leg (sd 5e-4) | component swap (sd 1.5e-3) |
|---|---|---|---|
| 1e-4 | .66 | .57 | .52 |
| 3e-4 | .90 | .69 | .57 |
| 1e-3 | ~1.0 | .95 | .71 |

- Read: a 1-3e-4 public edge from a one-line weight probe is close to a coin flip on private; the pending A16 probes (wr45, s615, a25) are
  private-neutral in expectation whichever way they read. Switching A to a probe winner under A15 costs nothing measurable: s615 keeps
  ~94 % of the S6 margin over C under a concave gain curve peaking near .2 (MODEL).

## 4. Private-rank model (mc.py, 6,000 sims per row; all MODEL)
Margins are relative to the copy block C (private shared draw). A's margin = posterior of the true S6fall effect (prior N(4e-4, 4e-4) GUESS;
public obs D - .5e-4 selection, sd 4e-4) + private noise .65 x 4e-4. W/V teams: uniform public margins over (0, D) and above A, own noise
sd 1.2 x 4e-4, iterator bias .5e-4. Bands >= .959 and .957/.956: noise 1.0-1.4e-3, bias 1e-4, plus a common other-lineage shift N(1e-4, 3e-4)
(GUESS). B = A - (weights true effect, N(.25 x .64 D, 2e-4) GUESS, public-probe tuned) - private noise. Silver = top 5 % of N_final 6,250-6,300.
D = public A - C margin, unknown in absolute terms (GUESS 3-8e-4, base 5e-4; display rounding hides it).

| scenario | P(A>C) | silver A | silver B | silver pair | bronze+ pair | median r pair |
|---|---|---|---|---|---|---|
| base D 5e-4 | .86 | .74 | .64 | .78 | .89 | 227 |
| D 3e-4 | .80 | .76 | .69 | .81 | .84 | 211 |
| D 8e-4 | .93 | .67 | .55 | .72 | .95 | 251 |
| noise x1.5 | .79 | .70 | .62 | .73 | .82 | 240 |
| noise x0.6 | .94 | .81 | .67 | .85 | .96 | 207 |
| skeptic S6 prior N(0, 3e-4) | .67 | .50 | .42 | .56 | .73 | 295 |
| weights real (no shrink) | .87 | .75 | .45 | .75 | .87 | 243 |
| weights overfit | .86 | .74 | .77 | .83 | .92 | 205 |
| lineage penalty 3e-4 | .86 | .68 | .59 | .73 | .89 | 243 |
| + 120 W teams pass A by 10-22 (half real) | .87 | .72 | .62 | .76 | .89 | 238 |
| + drift 120 and skeptic S6 | .67 | .46 | .39 | .53 | .73 | 307 |

- bronze+ ~= P(beat the copy block) in every row: the block decides medal vs no medal; W-zone reshuffling decides silver vs bronze given that.
- Gold: ~0 in all rows (r22 needs ~+5e-3).
- Correlation: MODEL corr of A and B private margins ~.83 (var A margin 1.5e-7 vs var of the weights difference 6.7e-8); rank corr .64-.77.
  Expected private gain of max(A, B) over A alone ~+.7e-4 (~30 W-zone ranks). A literal duplicate would add 0; R6 is far from a duplicate.

## 5. Final-pick recommendation
- **A = A15 rule (HW 57262115 unless a pending probe moves the API best-id; then that entry).** It is the best public entry, and its edge over
  the 600-team copy block is the S6fall leg only, so the shared weights overfit cancels against the block.
- **B = R6 57248084 (uniform weights + S6fall .2, J form, read-positive on 10-16).** It hedges the one ingredient most likely to be public-probe
  tuned (heliosli's per-finding weights, CLAIMED "very likely public-probe tuned" in PROGRESS_16) and has a measured public bracket above the
  copy block.
- Rejected B alternatives: hw3 57291203 (read negative, same weights side); R7 late 57291336 (same weights side, non-move, similar max-value
  but hedges a less likely overfit; its extra hour of runtime no longer matters once scored, CLAIMED code-comp mechanics: one hidden run scores
  public and private); S6W 57264835 (uniform + S6 .3, two hedges at once but no public bracket, non-best entries are invisible to the API;
  GUESS-level only); any non-S6 entry (it would sit with the copy block).
- Action items for Eesh (no submits implied): select both picks by hand before 10-22 23:59 UTC; re-run fetch_api_lb.py at the last read before
  10-22 18:00 IST to fix A.

## 6. What would change this
- If the forum or a host post shows the copy block is not the heliosli notebook (or carries S6fall), the cliff analysis flips: re-run ties.py
  and bc.py on a fresh snapshot.
- If another 100+ teams pass HW on public by 10-21 (drift above MODEL), A's silver odds fall ~1-2 pts per 100 if half of their gain is real, ~4 pts if 80 % is real (mc.py K, tf).
- Untested input: the true absolute size of D. The tie-run positions only give ranks; nothing public gives unrounded scores (CLAIMED 748011).
