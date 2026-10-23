# LB density math: where a25 sits inside .958, and what .959 / .960 would take (2026-10-18 ~12:50 IST)

Read-only. One fresh API snapshot fetched (`session_2026-10-14/research960/lbprobe/api_lb_1018T0645.json`, 06:45 UTC = 12:15 IST).
Nothing submitted, pushed or versioned; no GPU used. Scripts: `session_2026-10-18/next/density_work/` (listed at the end).
Tags: MEASURED (API snapshots), CLAIMED (someone else says so), MODEL (our arithmetic on stated assumptions), GUESS.
Units: "band position" u = (score - band floor) / .001, so u = 0 is the floor of the .958 band and u = 1 is where .959 starts.

## 0. Bottom line
1. **a25 57316395 sits at u ~ .66 of the .958 band (p10-p90 .54-.78), MODEL.** The ruler is the set of frozen "natural" teams:
   pre-heliosli iterators whose best entry dates from before 10-14 17:13 UTC, so their scores are fixed. Unrounded score: **.95866
   (.95854-.95878) if Kaggle truncates the display, .95816 (.95804-.95828) if it rounds.** Which one Kaggle uses is unknown (sec 4).
2. **Public gain needed (the same under both conventions):**
   - to display .959: **+3.4e-4** (p10-p90 2.2-4.6e-4);
   - to display .960: **+1.34e-3** (1.22-1.46e-3). That is out of reach for a one-line probe (sec 6).
3. **Rank each gives (MODEL, the counts are MEASURED):**
   - Today: the .959 floor is r160 (just over the line ~r160-163; +5e-4 gives ~r148) and the .960 floor is r103.
   - At the deadline (10-22 23:59 UTC), projected: the .959 floor ~r277 and the .960 floor ~r171.
   - Frozen a25 drifts to ~r380. The public silver cut is ~r318 (N_final ~6,375).
   - So **holding the public silver line at the deadline needs ~+2e-4, and displaying .959 there gives ~r270-290** (the private LB still decides medals).
4. **Boundary drift (MEASURED fits, last 24-72 h):**
   - teams at >= .959 grow +24 to +26 per day (the first .958 rank moves r160 -> ~r271 at 10-22 06:00 UTC, ~r277 at the deadline; r270-310 if it speeds up);
   - teams at >= .960 grow +14 to +16 per day;
   - teams above a25 inside the band grow +15.5 per day;
   - so our frozen rank grows ~+40 per day.
5. **Read key for tonight's 5 (MODEL):**
   - Near our score, each team a probe passes inside the band is worth ~1.2e-5 of public score (~1.75e-5 per team inside .959).
   - A probe that **shows .959** has a public delta above the gap (>= ~2.2-4.6e-4).
   - A best-id move that stays at .958 means delta < gap.
   - With an expected delta of +2e-4 and paired noise sd 3-5e-4: P(best-id move) .66-.75, P(shows .959) .33-.40.
6. **Title anchors are not measurements.**
   - Heliosli's ".9586" cannot be a true public score. Under rounding it would display .959. Under truncation it contradicts the
     MEASURED order: heliosli's 10-17 best (57282622, the 0-9586 v6 run) sits below R6, and R6 sits at u ~ .12.
   - So the titles carry no information about the convention or our score.
7. **Side findings:**
   - **HW minus the copy block ~ 4e-4 (2.4-5.6e-4), and a25 adds ~1.5e-4** (MODEL). This supports lb_private's base D = 5e-4.
   - **Heliosli's team has a new best, 57323261 (submitted 10-17 19:24 UTC), now at r166.** That is ~2.7e-4 above a25 (MODEL, in-band
     position 7). New versions of existing public notebooks are still allowed (notebooks.md), so if he republishes, a copy wave could
     land ABOVE us. Watch his notebooks.

## 1. Band counts (MEASURED, api_lb snapshots; `traj2.py`, `rates.py`)
| UTC | N | our entry | rank | position in .958 | >= .960 | .959 | .958 | .957 |
|---|---|---|---|---|---|---|---|---|
| 10-16 06:22 | 6,065 | R6 57248084 | r147 | 42 / 99 | 71 | 34 | 99 | 393 |
| 10-16 15:17 | 6,094 | HW 57262115 | r135 | 16 / 395 | 77 | 42 | 395 | 329 |
| 10-17 16:21 | 6,144 | HW | r171 | 30 / 853 | 93 | 48 | 853 | 222 |
| 10-17 19:35 | 6,150 | HW | r178 | 34 / 890 | 95 | 49 | 890 | 228 |
| 10-17 21:13 | 6,153 | a25 57316395 | r170 | 24 / 908 | 97 | 49 | 908 | 227 |
| 10-18 06:45 | 6,164 | a25 | r189 | 30 / 1,002 | 102 | 57 | 1,002 | 210 |
- Bands above .960 now: .960 34, .961 17, .962 18, .963 11, >= .964 22. Gold cut r22 (= 10 + 0.2 % of N).
- a25 passed **exactly 10 frozen teams** at its read: 33 teams were above HW at 19:35, 23 above a25 at 21:13, and no team in that
  slice changed between the two snapshots (MEASURED).
- The copy zone (exact-tie runs, increasing submissionId) now spans ~r465-r1149. It is many separate exact-score clusters, not one
  score (MEASURED, `ties.py`).

## 2. The ruler: frozen natural teams (MEASURED order, MODEL density)
- The reference set is the teams in the .958 band whose best entry predates 10-14 17:13 UTC. That is before any public notebook
  above .956 existed (FORUM_DIGEST_10-13), so they are independent iterators. Each has a fixed score.
- Kept: the teams frozen through the R6 and HW reads, minus one exact-tie member (57125447, it ties with copies).
- 11 teams remain. Their order against our three entries (MEASURED, `natlist.py`):
  `[57164725] < R6 < [57146242, 57159810, 57076561, 57121645, 57152227] < HW < [57153608, 57159649, 57154329, 57173807, 57131112]`.
  Four of the five above HW are still above a25. 57131112 sits directly above us (r188 vs r189).
- Density model (MODEL): natural scores inside the band follow `f(x) ~ exp(-k x)`, thinning upward. The measured natural tail
  (.960 4 : .959 8 : .958 11 at the HW read) gives k ~ .3-.7. Prior k ~ U(.2, 1.0).
- Likelihood: multinomial on those counts, with uniform priors on u_R6 < u_HW.
- a25 from the 10-team jump: `1 - u_a25 = (23/33)(1 - u_HW)`, with +-10 % for non-uniform density in the top slice.

| entry | u p50 (p10-p90) | truncation | rounding |
|---|---|---|---|
| R6 57248084 | .10 (.03-.23) | .95810 | .95760 |
| HW 57262115 | .51 (.34-.69) | .95851 | .95801 |
| a25 57316395 | **.66 (.54-.78)** | **.95866** | **.95816** |
| copy-block top | <= R6 (57164725 sits inside the copy zone) | ~.9581 | ~.9576 |

Sensitivity of u_a25 (`posterior.py`):
| variant | u_a25 p50 | gap to .959 p50 |
|---|---|---|
| base | .66 | 3.4e-4 |
| flat density | .70 | 3.0e-4 |
| steep density (k .8-1.4) | .63 | 3.7e-4 |
| tie member counted | .68 | 3.2e-4 |
| 57164725 treated as copy-level | .64 | 3.6e-4 |

- Derived (MODEL):
  - HW - R6 (heliosli weights on the S6 base) = 3.9e-4 (2.4-5.6e-4);
  - a25 - HW = 1.5e-4 (~1.0-2.0e-4);
  - HW - copy block >= HW - R6, i.e. ~4e-4 or a bit more.

## 3. Cross-checks (all MODEL)
- **All-teams smooth tail.** This continues the .959 band density down into the top of .958 (`smooth.py`).
  - It gives a25 u = .64 (k = ln(57/34)), range .59-.71. That matches.
  - The same estimator is unstable over time: for frozen HW it drifted .77 (10-16 15:17) -> .54 (10-17 19:35), because the strip above
    us filled faster than the .959 band did (+15.5 per day vs +~10 per day net).
  - The natural-team ruler gives HW .51, which matches the late value. So the early smooth estimates were too high: the strip above us
    had just been emptied.
- **Landing model.** Teams that improved from below our entry during the two frozen periods (38 h) landed: 32 in the strip above us,
  23 in .959, 7 at >= .960 (MEASURED, `flows.py`).
  - Assume an exponential landing density: λ = ln(1 + 23/7) = 1.46 per band, and strip width = ln(1 + 32/30) / λ = .50 (CI ~.3-.8).
  - The ruler predicts a time-weighted width of ~.44 (HW .49 for 28 h, a25 .34 for 9.5 h). That is consistent.
- **Crossing rates.**
  - Teams in the strip above us that improved: 8 of 9 crossed into .959.
  - Teams in .959 that improved: 16 of 30 crossed into .960+, which gives an exponential jump scale of ~.7 band.
  - With a strip width of .44, the model expects ~6.7 of 9 crossing; P(>= 8) ~ .3. Consistent, and it leans a little toward a
    narrower gap.
- **Flow into .959 (MEASURED, 38 h, ~per day):**
  - from the strip above us 5, from lower in .958 7.5, from below .958 7 (~20 per day in total);
  - .959 -> .960+ 10 per day;
  - .958 or lower straight to >= .960, 4.4 per day.

## 4. Rounding vs truncation (unresolved)
- Our own submission list, the leaderboard API and the notebook scores are all 3-decimal strings (MEASURED). No public surface gives
  more digits.
- Heliosli's 4-digit titles: "0-9576" (R6 + flip + Raptor, uniform) displays .957, and "0-9586" displays .958.
  - Both titles fit truncation only if they were true scores.
  - ".9586" cannot be true under either convention. Under truncation it would put heliosli's 10-17 best above R6 (u >= .6), but the
    measured order puts it below R6 (u ~ .1; 10 of the 11 natural teams sit above R6). P ~ .002 under the density model.
  - So the titles are estimates, not evidence.
- GUESS: rounding is slightly more likely (standard formatting rounds). The gains and ranks above do not depend on this. Only the
  absolute 5th digit does.

## 5. Gain over a25 -> public rank (MODEL; median [p10-p90 over the position posterior])
Counts are projected linearly at the MEASURED rates: >= .960 +14.5 per day, >= .959 +25 per day, above-us strip +15.5 per day.
Inside .959 the density thins upward (k .52, from .959 : .960 = 57 : 34).

| extra public gain | today 10-18 06:45 | 10-22 06:00 UTC (last read) | deadline 10-22 23:59 | P(shows >= .959) | P(shows .960) |
|---|---|---|---|---|---|
| 0 (a25 frozen) | r189 | r370 | r380 | 0 | 0 |
| +1e-4 | r180 | r341 | r350 | 0 | 0 |
| +2e-4 | r172 | r311 | r319 | .07 | 0 |
| +3e-4 | r163 | r282 | r288 | .35 | 0 |
| +4e-4 | r155 | r263 | r269 | .74 | 0 |
| +5e-4 | r148 | r251 | r256 | .96 | 0 |
| +7.5e-4 | r133 | r222 | r227 | 1 | 0 |
| +1.0e-3 | r119 | r197 | r201 | 1 | 0 |
| +1.25e-3 | r107 | r175 | r178 | 1 | .18 |
| +1.5e-3 | r96 | r155 | r158 | 1 | .96 |
| +2.0e-3 | r78 | r124 | r127 | 1 | 1 |
- Public silver cut at the deadline is ~r318 (N grows +42-50 per day -> N_final ~6,360-6,400, MODEL), which sits ~+2e-4 above a25.
  Today the cut is r308 and we are safely inside it.
- Displaying .959 is worth ~30 ranks today but ~90-110 ranks at the deadline, because the strip under .959 keeps filling. This is
  public only: lb_private.md (private-rank model) still governs the picks.

## 6. What a one-line probe can and cannot reach (read key for tonight's c4, fr65, a15, c4a15, pf50)
- P(best-id move) and P(shows .959) by expected public delta mu and paired public noise sd. The noise sd comes from metric.md: a
  1-line weight tweak is 3-7e-4. MODEL.

| mu | sd 3e-4: move / .959 | sd 5e-4: move / .959 |
|---|---|---|
| -2e-4 | .25 / .04 | .34 / .14 |
| 0 | .50 / .14 | .50 / .25 |
| +1e-4 | .63 / .23 | .58 / .32 |
| +2e-4 | .75 / .33 | .66 / .40 |
| +3e-4 | .84 / .46 | .73 / .47 |
| +5e-4 | .95 / .70 | .84 / .63 |
- GUESS mapping from the A17 gold58 proxy screen, shrinking public mu to .1-.25 x the gold58 delta (in-sample selection, R7 proxy,
  no r224):
  - c4 mu ~2.5-6e-4 -> P(.959) ~.35-.65;
  - fr65 ~1-3e-4 -> ~.25-.45;
  - pf50 / a15 ~0-1.5e-4 -> ~.15-.3.
  - The five reads are correlated, so P(at least one shows .959) is ~.5-.7 (GUESS).
- .960 needs ~+1.34e-3. That is beyond 2 sd for any single weight probe, and it is not a target for this week.
- How to convert a read:
  - in-band teams passed x ~1.2e-5 = delta (today's strip: 29 teams over ~3.4e-4);
  - a jump inside .959 is worth ~1.5e-5 per team near its floor, ~2.1e-5 near its top;
  - teams that move during the 6 h read window blur this by +-2-3 teams.

## 7. Caveats
- The natural set is small (11 teams). The 80 % interval on the gap (2.2-4.6e-4) is mostly that sampling noise.
- The method assumes that staying frozen does not depend on position inside the band.
- The method assumes a smooth natural tail inside .958. That held on 10-14: no public notebook above .956 existed then.
- The projections are linear. Deadline-week acceleration would push every projected rank up (worse), by ~+20-40.
- The rank table uses uniform density in the strip above us. If the strip fills unevenly (for example, a heliosli republish that
  lands just above us), the +1-3e-4 rows shift.

## Files (session_2026-10-18/next/density_work/, all read-only on the snapshots)
- `traj2.py`: band table per snapshot.
- `rates.py`: growth rates.
- `sidtime.py`: maps submissionId to UTC time, using the lower envelope of team lastSubmissionDate. It matches our known submit times.
- `natural.py` and `natlist.py`: the natural ruler.
- `posterior.py`: the u posterior with sensitivity.
- `ranks.py`: gain -> rank and the probe key.
- `smooth.py`: all-teams tail cross-check.
- `flows.py`: zone transitions.
- `who2.py`: authors' team ranks.
- `vzone.py`: neighbours.
