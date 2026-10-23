# decision.md: final picks after the eoa10 / eoa20 read (10-20 ~02:15 IST, Claude, r5 Task B)

RESEARCH ONLY, CPU only. Nothing was submitted, pushed, committed or built. No dataset was created. No scheduled task, PREREG,
HANDOFF, PROGRESS or session_2026-10-19/a20 file was touched (a20 was only read). No box use: the OAI inputs are the skeptic-ladder
aggregate logs that were already local, so no OAI row, id or per-knee value was read. One read-only LB API fetch
(`fetch_api_lb.py` -> `r5/api_lb_1020T_r5.json`). One local CPU run on competition train OOF (`noise_corr.py`, < 1 min).

**Tags.** MEASURED (computed tonight or read from a log, file named), CLAIMED (an older note says so), MODEL (arithmetic on stated
inputs), GUESS (judgement).
**Units.** d10 / d20 = hidden macro-AUC gain of eoa10 / eoa20 over c4. "Margin" = private macro margin over the heliosli copy block
(the picks.md ruler). "pts" = percentage points of P(silver). Team private score = max of the two picks.

## 0. Answers

### 0.1 Decision table (read once BOTH entries have a public score; values MODEL, base prior / model-averaged "mix" prior)

| read (R1 vs c4 57343435) | P before the read (base / mix) | A | B | P(silver) | P(bronze+) | P(r<=200) | P(r<=100) | E[max margin] |
|---|---|---|---|---|---|---|---|---|
| **eoa20 is the API best-id** | .36 / .38 | **eoa20** | **R6 57248084** | .864 / .851 | .943 / .936 | .68 / .66 | .13 / .13 | +15.1 / +14.9e-4 |
| **eoa10 is the API best-id** | .37 / .32 | **eoa10** | **R6** | .753 / .732 | .895 / .883 | .46 / .43 | .017 / .014 | +9.8 / +9.4e-4 |
| **no move** (both below c4) | .27 / .31 | **c4** | **R6** | .631 / .631 | .826 / .826 | .30 / .30 | .004 / .004 | +7.2 / +7.2e-4 |
| only eoa10 scored (eoa20 failed), moved | | eoa10 | R6 | .797 / .782 | .917 / .909 | .53 / .51 | .031 / .029 | +11.2 / +10.8e-4 |
| only eoa10 scored (eoa20 failed), no move | | c4 | R6 | .631 / .632 | .826 | .30 | .004 | +7.2e-4 |
| only eoa20 scored (eoa10 failed), moved | | eoa20 | R6 | .819 / .812 | .919 / .914 | .60 / .59 | .096 / .103 | +13.3 / +13.4e-4 |
| only eoa20 scored (eoa10 failed), no move | | c4 | R6 | .632 / .632 | .826 | .30 | .004 | +7.2e-4 |
| one entry still pending at ~13:00 | | wait for it | | | | | | |
| both failed / nothing scored | | c4 | R6 | .631 | .825 | .30 | .004 | +7.2e-4 |

- Gold (r22) is .000 in every cell: it needs a margin of about +4e-3 (MODEL, field curves). "Gold-adjacent" is reported as
  r<=200 (needs about +1e-3) and r<=100 (needs about +3e-3).
- **The A15 rule (A = API best-id, B = R6) is the best non-S6F policy in every main outcome.** It holds in 17 / 22 / 21 of 23
  sensitivity rows for the eoa20 / eoa10 / no-move outcomes (sec 3); the exceptions are near-ties or rows whose prior the read
  itself rejects.
- Pre-read value of the policy (MODEL, `policies.log`): P(silver) .759 (base) / .746 (mix) / .748 (ladder prior) vs .631 for the
  status quo c4 + R6. So the two in-flight entries are worth about **+11 to +13 pts expected P(silver)**. The size rests on the
  sharp silver step of the field model (sec 7). Following the read beats ignoring it ("always eoa10 + R6": .736 / .718 / .734) by
  +1.4 to +2.8 pts.

### 0.2 "Should eoa10 be A (or B) even if neither entry moves the best-id?" **No, to both.**
- After a no-move the posterior on eoa10 turns negative: d10 -2.0e-4 (P>0 .25, base), -1.8e-4 (P>0 .24, mix) (MODEL).
  - Why: this change has an unusually precise public read. The paired public noise of eoa10 - c4 is sd 2.1e-4 (MEASURED
    `noise_corr.log`, matches m12_lbnoise.log's .00021), less than half the prior sd (4.5e-4 for MORE_1020's P .75). A non-move is
    therefore strong evidence, unlike the usual weight probes where public noise swamps the prior.
- **eoa10 as A after a no-move:** eoa10 + R6 vs c4 + R6 is -5.8 pts (base), -5.3 (mix), -3.2 (ladder prior), and -0.4 to -7.9
  across all 23 sensitivity rows (MODEL).
- **eoa10 as B after a no-move (c4 + eoa10):** -6.5 pts (base), -2.6 to -10.6 across rows. It is a near-twin of c4 with a
  negative-mean change, so it hedges nothing.
- **Break-even (sec 4):** eoa10 as A after a no-move pays only if your pre-read P(d10 > 0) is at least about .86 at the measured
  read noise, or about .80 if the hidden read noise is 1.5x. MORE_1020's skeptic ladder says P about .75. The explicit ladder
  mixture says .80. skeptic_eoa.md's earlier number was .80-.82 (CLAIMED). All sit on the c4 side.
- Partial case: if only eoa20 scored and did not move (eoa10 failed), d10's posterior is about 0 (-0.3e-4, P .49). eoa10 + R6
  then ties c4 + R6 (.631 vs .632). The point is moot, because a failed entry cannot be selected.

### 0.3 "Should B change from R6?" **No.**
- **HW 57262115:** -0.1 to -3.6 pts as B in every row, with three exceptions: "heliosli table real" (tabReal, +1.0 to +3.4), a
  tie under strongS6 after a no-move, and +0.4 in one neg-prior row. Same as picks.md: HW is better only if heliosli's table is
  truly good on private.
- **a25:** -2 to -4 pts.
- **c4 as B when A has moved to eoaX:** -1.4 pts (eoa20 read) and -3.9 pts (eoa10 read) in base. Policy level: -1.9 (base) /
  -1.9 (mix) / -2.4 (ladder) pts.
- **The other eoa entry as B:** worse after an eoa10 read (-4.3 base). A near-tie after an eoa20 read (-0.2 base, -0.8 mix,
  -0.8 to +1.0 across rows). R6 is the only candidate that hedges both the E-OA risk and the c4 tuning risk.
- **S6F 57201874** shows +0.8 / +1.6 / +2.4 pts as B after an eoa20 / eoa10 / no-move read (base), and up to +5.2 under the
  skeptic-S6 scenario, as in picks.md. That edge is an artifact (MODEL):
  - It comes entirely from S6F's modelled extra private noise, which uses a single-fold R7 OOF stand-in.
  - Halving S6F's idiosyncratic noise cuts the edge to +0.2 to +0.5 pts. Removing it gives +0.0 to +0.2.
  - If R6 is truly better than R7 (r6better), S6F loses 0.4 to 1.4 pts.
  - It is also a .957 entry.
  - So it is not a reason to move B (agrees with picks.md sec 4).
- **Side note, not in the candidate set:** a new "R6 + E-OA a .1" as B, built only after a POSITIVE read, models +1.6 to +2.4 pts
  in the eoa20 / eoa10 outcomes and -1.8 to -1.9 pts after a no-move (MODEL, `dec_extra_*.json`).
  - Assumption: R6's gold58 coupling gap .09 is close to c4's .10 (CLAIMED skeptic_eoa.md sec 2), so it gets 0.9 x d10.
  - It would need Eesh's go, a PREREG amendment (A20 already lists "B = R6 + E-OA is a separate later decision"), a build, gates
    and a push. B would then carry the same OA change as A.
  - Not recommended without his explicit go. Never after a no-move.

### 0.4 If eoa20 is the best-id, keep A = eoa20 (do not override R1 with eoa10)
- eoa20 + R6 vs eoa10 + R6: +2.3 pts (base), +2.9 (mix), +1.6 (ladder), +5.3 (orig prior), +0.5 at 1.5x read noise (MODEL).
- **It is a near-tie or reverses only in these cases:**
  - a tight prior (prior sd of d10 <= 2e-4: -1.2 pts);
  - 1.5x noise with the ladder prior (-0.3);
  - the pessimistic "neg" prior (-8 to -10 pts). Under that prior an eoa20 best-id has P only .04-.08, so the read rejects it: in
    the mix its posterior weight falls from .10 to .01 after an eoa20 move (MEASURED in `dec_prior_mix.json`, MODEL inputs).
- **By move size** (sec 5): under the mix prior eoa20 stays ahead at every public margin. An eoa20 win is almost always large: the
  margin is > 3e-4 in 90 % of cases, and < 1.1e-4 in only 2 % (MODEL).
- **If the polls show eoa10 moving first and eoa20 then moving above it:** both are positive. A = eoa20, B = R6.

## 1. Model (MODEL unless tagged; scripts in `r5/`)

### 1.1 Our side
- Source: the think/picks posterior `post_<our>_report.npz` (`picks_post.py`, 200k draws). It is conditioned on every public read
  up to 10-19 (CLAIMED picks.md sec 3).
- Private margins over the copy block for c4, R6u (= R6 57248084), HW, a25 and S6F. Reused unchanged.

### 1.2 E-OA
- eoaX private = c4 private + dX + private paired noise.
- **Priors on (d10, d20)** (`decision_mc.py prior=`; moments from the draws):

| prior | d10 mean, sd, P>0 | d20 mean, sd, P>0 | P(d20 > d10) | source |
|---|---|---|---|---|
| **more (base)** | +3.0e-4, 4.5e-4, .75 | +1.0e-4, 9.0e-4, .54 | .34 | MORE_1020 sec 1: +.0003 P .75 / +.0001 P .55-.60. Corr .95 (ladder .99, loosened) |
| ladder | +2.8e-4, 3.4e-4, .80 | +1.1e-4, 7.4e-4, .58 | .35 | Explicit mixture, see the steps below |
| orig | +5.0e-4, 5.0e-4, .84 | +7.0e-4, 1.1e-3, .74 | .62 | skeptic_eoa.md 10-19 (pre-ladder, c4 gap map) |
| null | 0, 2e-4, .50 | 0, 4e-4, .50 | .50 | E-OA does nothing |
| neg | -1.5e-4, 2.5e-4, .27 | -8.4e-4, 5e-4, .05 | .01 | Ladder tilt1.5 cell means (hidden coupling 1 sd above gold58) |
| **mix** | +2.5e-4, 4.4e-4, .70 | +0.9e-4, 9.1e-4, .53 | .37 | Weights .40 / .25 / .15 / .10 / .10 (GUESS). The read reweights them |

- **How the ladder prior is built:**
  1. Start from the 60 MEASURED ladder cells: R1g74, tilt1.0 and tilt1.5, each with 4 OAI-free models x 5 MOAKS definitions
     (`skeptic_ladder/*.log`, parsed by `prior_ladder.py`).
  2. Weight the couplings .375 / .375 / .25, from gold58 phi .32 sd .09 (CLAIMED skeptic_eoa.md).
  3. Convert to macro: OA-mean / 4.
  4. Add a c4 transfer shift of k_a x dgap, with dgap ~ N(.03, .04). k = .0039 (a .1) and .0079 (a .2) per unit of coupling gap,
     fitted across the cells (MEASURED fit).
  5. Add a shared structural spread of 1.5e-4 / 3e-4 (GUESS).
- **Raw ladder** (no c4 shift; MEASURED `prior_ladder.log`):
  - d10 +1.7e-4 sd 2.5e-4, P .76;
  - d20 -1.3e-4 sd 5.6e-4, P .41;
  - cell corr(a .1, a .2) .99.
  - R1g74 cells alone: +.00026 / +.00011. That is MORE's headline, reproduced.
- **Note on MORE's "eoa20 vs eoa10 is a coin flip (P about .5)":** its own point estimates (+.0003 vs +.0001) imply P(d20 > d10)
  about .34-.37 under any of these priors. This changes only the outcome odds, not any pick.

### 1.3 Read
- **Public paired noise** (MEASURED `noise_corr.log`; 600 subsamples of 390 / 910 from 4,333 train studies; R7 OOF, report labels;
  both a values on the same subsample):

  | set size | sd a .1 | sd a .2 | corr |
  |---|---|---|---|
  | 390 (public) | .00020 | .00044 | .962 |
  | 910 (private) | .00013 | .00028 | .983 |

  - With the finite-population correction: public 2.1e-4 / 4.6e-4, private 1.46e-4 / 3.15e-4.
  - Public and private noise are independent (disjoint studies).
- **R1 outcomes:**
  - eoa20 best-id iff pub20 > 0 and pub20 > pub10;
  - eoa10 best-id iff pub10 > 0 and pub10 >= pub20;
  - no move iff both <= 0.
- **Magnitude:** observed delta = teams passed x ~1.2e-5 (CLAIMED density.md). Observation error sd sqrt((3e-5)^2 + (0.3 delta)^2)
  (GUESS).

### 1.4 Value
- `field_curves.py` reuses the 10-19 field model verbatim (`field_mc.py` head, 20k field draws). It gives P(rank <= k | margin x)
  for k = 22 / 50 / 100 / 200 / silver r316 / bronze r633.
- Silver threshold x* = 5.0e-4 (base). The other field scenarios give 4.4e-4 (fieldBad), 5.6e-4 (fieldGood) and 6.2e-4 (Wbig)
  (MODEL, `field_curves.log`). These match picks.md.

## 2. Pair tables (P(silver), MODEL; base prior / mix prior; `dec_base.json`, `dec_prior_mix.json`)

| pair | eoa20 best-id | eoa10 best-id | no move | prior only (no read) |
|---|---|---|---|---|
| c4 + R6 | .630 / .631 | .632 / .630 | **.631 / .631** | .631 / .631 |
| c4 + HW | .621 / .623 | .623 / .621 | .622 / .622 | .622 / .622 |
| c4 + a25 | .593 / .596 | .596 / .594 | .595 / .595 | .595 / .595 |
| c4 + S6F | .654 / .655 | .655 / .653 | .655 / .656 | .655 / .655 |
| c4 + eoa10 | .819 / .796 | .713 / .689 | .566 / .563 | .711 / .691 |
| c4 + eoa20 | .851 / .837 | .655 / .645 | .554 / .557 | .698 / .690 |
| eoa10 + R6 | .841 / .822 | **.753 / .732** | .573 / .578 | .736 / .718 |
| eoa10 + HW | .836 / .816 | .745 / .725 | .570 / .573 | .730 / .713 |
| eoa10 + eoa20 | .862 / .843 | .710 / .685 | .438 / .449 | .691 / .672 |
| eoa20 + R6 | **.864 / .851** | .656 / .644 | .467 / .480 | .679 / .672 |
| eoa20 + HW | .861 / .847 | .654 / .642 | .478 / .490 | .680 / .673 |
| eoa20 + S6F | .873 / .860 | .674 / .661 | .485 / .499 | .694 / .686 |
| HW + R6 (no c4 family) | .529 / .529 | .531 / .529 | .529 / .531 | .530 / .530 |
| c4 alone | .538 / .540 | .540 / .538 | .540 / .539 | .539 / .539 |

**Posterior per outcome** (mix, MODEL):

| outcome | d10 (P>0) | d20 (P>0) | posterior weights of more / ladder / orig / null / neg |
|---|---|---|---|
| eoa20 best-id | +5.8e-4 (.95) | +8.6e-4 (.91) | .38 / .25 / .24 / .12 / .01 |
| eoa10 best-id | +2.8e-4 (.86) | -0.1e-4 (.49) | .47 / .31 / .10 / .03 / .09 |
| no move | -1.8e-4 (.24) | -7.7e-4 (.09) | .35 / .19 / .09 / .15 / .22 |

## 3. Sensitivity (`summary.log`: 23 rows; silver delta vs the A15 pair, in pts)
Rows:
- priors: orig, ladder, null, neg, mix;
- read noise x0.7 / x1.5, noise corr .85;
- our-side scenarios: skepS6, strongS6, r6better, tabReal, tabOver, noise1.5, bigKnobs, smallKnobs;
- fields: fieldBad, Wbig, fieldGood;
- S6F noise deflation x0.5 / x0;
- ladder + noise 1.5; neg + noise 1.5.

| outcome | A15 pair is the best non-S6F pair | exceptions |
|---|---|---|
| eoa20 best-id | 17 of 23 | eoa10 + eoa20 +0.1 to +1.0 (noise1.5, ncorr .85, tabReal, ladder+noise1.5). eoa10 + R6 +8 to +10 under the neg prior, where this outcome has P .04-.08 |
| eoa10 best-id | 22 of 23 | eoa10 + HW +1.7 (tabReal) |
| no move | 21 of 23 (+1 tie) | c4 + HW +3.4 (tabReal); tie under strongS6 |

- eoa10 as A after a no-move: -0.4 (ladder + noise1.5) to -7.9 (neg) in every row.
- S6F as B (with the A15 A): +0.2 to +5.2 in raw rows, +0.2 to +0.5 with half the stand-in noise, -0.4 to -1.4 under r6better
  (sec 0.3).

## 4. The crux: how tight is your E-OA prior? (`sweep_width.log`; d10 ~ N(+3e-4, s), d20 ~ N(+1e-4, 2.2 s), corr .95; MODEL)

| prior sd s of d10 | prior P(d10 > 0) | P(no move) | d10 after a no-move (P>0) | no move: eoa10 + R6 minus c4 + R6 | eoa20 best-id: eoa20 + R6 minus eoa10 + R6 |
|---|---|---|---|---|---|
| 1.0e-4 | .999 | .10 | +2.2e-4 (.99) | +8.5 pts | -5.0 |
| 1.5e-4 | .977 | .12 | +1.6e-4 (.89) | +5.9 | -3.0 |
| 2.0e-4 | .933 | .15 | +0.9e-4 (.71) | +3.3 | -1.2 |
| 2.5e-4 | .885 | .18 | +0.2e-4 (.55) | +1.0 | +0.2 |
| 3.0e-4 | .841 | .21 | -0.4e-4 (.44) | -1.0 | +1.4 |
| **4.5e-4 (MORE's P .75)** | .748 | .27 | -2.0e-4 (.24) | **-5.9** | **+3.3** |
| 4.5e-4, read noise x1.5 | .748 | .29 | -1.4e-4 (.35) | -3.4 | +1.5 |
| 3.0e-4, read noise x1.5 | .841 | .24 | +0.4e-4 (.56) | +1.7 | -0.8 |

- Break-even is about P(d10 > 0) = .86 at the measured noise and about .80 at 1.5x noise.
- **If you trust E-OA at P >= .9 before the read, the read barely matters and eoa10 should be A whatever happens.** None of our
  notes supports that: the independent rulers give .75-.82 (CLAIMED MORE_1020, skeptic_eoa.md; MEASURED ladder mixture .80).
- At the documented prior, follow R1.
- The no-move probability itself is informative. At P .75 a no-move has P about .27-.31. At P .93 it would have only .15.

## 5. Read key by move size (mix prior; observed public delta = teams passed x 1.2e-5; P(silver), MODEL)
- c4 sits about 1.1e-4 under .959 (CLAIMED picks.md). 18 teams are inside .958 above it now (MEASURED, sec 6).
- So passing all 18 and displaying .959 means a public delta of about >= 1.1e-4.

| outcome, observed delta | share of outcome | d10 / d20 posterior | best-id + R6 | other eoa + R6 | c4 + R6 | best-id + c4 |
|---|---|---|---|---|---|---|
| eoa10, 0-0.5e-4 (<= ~4 teams) | .09 | +0.9 / -3.0e-4 | **.665** | .562 | .632 | .615 |
| eoa10, 0.5-1.1e-4 | .14 | +1.3 / -2.4e-4 | **.679** | .580 | .631 | .629 |
| eoa10, 1.1-3e-4 (shows .959) | .37 | +2.3 / -0.7e-4 | **.718** | .627 | .631 | .672 |
| eoa10, > 3e-4 | .37 | +4.4 / +2.3e-4 | **.788** | .712 | .629 | .754 |
| eoa20, 0-0.5e-4 | .01 | +0.1 / +0.4e-4 | **.652** | .636 | .626 | .641 |
| eoa20, 0.5-1.1e-4 | .01 | +0.6 / +0.9e-4 | **.669** | .659 | .637 | .648 |
| eoa20, 1.1-3e-4 | .07 | +1.8 / +2.3e-4 | **.713** | .699 | .632 | .696 |
| eoa20, > 3e-4 | .90 | +6.2 / +9.4e-4 | **.867** | .836 | .631 | .853 |

- Even a hair-line eoa10 move is worth taking: +3.3 pts over c4 + R6.
- Under tight priors (s <= 2.5e-4), eoa10 + R6 leads eoa20 + R6 by up to 2.5 pts after a 1.1-3e-4 eoa20 move (MODEL
  `dec_tight_*.json`). That is the only place where the "which eoa" choice is close.

## 6. Latest LB (MEASURED, `r5/api_lb_1020T_r5.json`, 10-19 20:17 UTC = 10-20 01:47 IST)
- **Board:**
  - N 6,220. c4 57343435 is at r208 (.958).
  - **.959+ = r1-189** (189 teams): .959 67, .960 40, .961 19, .962 17, .963 19, >= .964 27.
  - **.958 = r190-1476** (1,287 teams). 18 teams are in-band above c4.
  - Cuts today: silver r311, bronze r622, gold r22.
- **Since the 10-19 11:00 UTC snapshot (9.3 h):** N +11, >= .959 +6, in-band above c4 +3, c4 r199 -> r208.
- **Copy clusters:** long exact-tie runs (>= 20 members) start at about r694 (`runs_top.py`). The short 4-7 runs from r225 are
  consistent with chance ordering.
- **Field model:** the deadline projection (>= .959 about 258, N about 6,336, silver r316) is still consistent with these rates, so
  the field was not refit (MODEL).
- **Public rank only:** an eoa move that shows .959 is worth about 90-110 public ranks at the deadline (CLAIMED density.md). It does
  not change the picks.

## 7. Caveats
- **Absolute P(silver) values rest on the field model's sharp step** at x* about 5e-4. The step comes from 750 copy teams plus a
  tight near-copy W zone (GUESS parameters, picks.md sec 4). Pair differences are more stable than levels. The "+11 to +13 pts from
  E-OA" is the least robust number here.
- **Read noise was measured on the wrong regime:** report labels and the R7 OOF single model, not expert labels with c4. Lower
  expert prevalence for Lateral OA could raise it. The x1.5 rows cover that, and every pick is unchanged.
- **Prior tails are generous upward.** The eoa20-read posterior (d20 about +8.6e-4) exceeds every RSNA-coupling ladder cell (max
  +8.5e-4, MEASURED), so the eoa20-row P(silver) of .85-.86 is probably a few pts optimistic. The pick (A = eoa20) also holds at
  small margins (sec 5).
- **Independence assumption:** the E-OA private paired noise is assumed independent of the c4-vs-R6 private noise.
- **R1 is assumed exact:** the best-id is the maximum unrounded public score.
- **Not modelled:** host or forum rulings (fastMRI question 743416), or an S6fall cliff beyond picks_post's knob prior. These hit
  every candidate alike.

## 8. What Eesh does (by hand)
1. Wait until BOTH eoa entries show a public score in the submissions list (expected ~12:00-14:30 IST 10-20). Read the API best-id
   (R1). If one is still pending at 13:00, wait for it: the pick depends on the final state.
2. Pick from the sec 0.1 table:
   - eoa20 best-id: eoa20 + R6;
   - eoa10 best-id: eoa10 + R6;
   - no move: c4 + R6.
   Select by hand before 10-22 23:59 UTC and screenshot the checkmarks. If nothing is selected, Kaggle auto-picks the top 2 public
   entries, which would be near-twins.
3. Nothing needs to be built or submitted for the picks. The R6 + E-OA B variant (sec 0.3) is only an option after a positive read,
   and only with his explicit go plus a PREREG amendment.

## 9. Files (`session_2026-10-20/r5/`, Task B only)
- **Inputs and rulers:**
  - `api_lb_1020T_r5.json`, `bands_now.json`, `runs_top.py`: LB snapshot (MEASURED).
  - `noise_corr.py` / `.log`: joint read noise (MEASURED).
  - `prior_ladder.py` / `.log`, `ladder_cells.json`: parsed ladder cells (MEASURED aggregates).
  - `field_curves.py` / `.log`, `curves_<field>.npz`: rank-threshold curves (MODEL).
- **Model:**
  - `decision_mc.py`: read -> pick model. Args: our=, field=, prior=, noise=, ncorr=, s6fdefl=, mag=1, extra=1.
  - Outputs: `dec_<tag>.json`, `dec_base.log`, `grid_runs.log`.
- **Summaries:**
  - `summarize.py` -> `summary.log` (23-row matrix);
  - `policies.py` -> `policies.log` (pre-read policy values);
  - `sweep_width.py` -> `sweep_width.log` (prior-width sweep);
  - `dec_tight_*.json` (tight priors with move-size bins).
