# shakeup.md: public-to-private shake-up for our band, and what the public says about a site split (10-20 ~23:30 IST, private track)

RESEARCH ONLY. Nothing was submitted, pushed, committed, versioned, downloaded, posted or voted; no dataset, no GPU job, no
scheduled task, no box contact, no OAI data. PREREG, HANDOFF, PROGRESS and the a20 / a21 / val / night folders were only read.
One anonymous public-LB read (`work/api_lb_1020T_shake.json`, 17:10 UTC = 22:40 IST, via
`session_2026-10-14/research960/lbprobe/fetch_api_lb.py`). Everything else is local CPU on competition data (train reports, gold74
labels, LB snapshots) and on box aggregates already in older notes. Forum text used here is the local copy of 742327 (fetched 10-15)
and 734681 (fetched 10-19); no fresh forum fetch.

Tags: MEASURED (computed now, file named), CLAIMED (older note or forum poster, source named), MODEL (arithmetic / simulation on
stated assumptions), GUESS (judgement). Units: macro AUC, "e-4" = 1e-4, one display band = 10e-4. "Margin" = private score over
the heliosli copy block (public .95802), the picks.md / bpick.md / picks2.md ruler. "Family" = every entry built on the public
R6 + Raptor + heliosli-table notebooks: the ~600 exact-copy teams, the tweakers around them, and all of our entries.
A = pw30 57425970 (API best-id). Pair = pw30 + R6pw20 (the proposed B hedge; modelled with picks2.md's B - A spread).

## 0. Answer

1. **Noise for our band (MODEL, sec 1).** One entry's absolute macro AUC has sd **38-50e-4 public / 25-33e-4 private** (Hanley,
   gold58 vs report prevalence), but almost all of that is common to every team and cannot move ranks. What moves ranks:
   - vs a family twin (copy block, c4): paired public 6.5-8.7e-4, private 4.2-5.6e-4 (CLAIMED picks_cov);
   - **vs an independent-lineage team: paired public ~19-32e-4, private ~13-21e-4** (MEASURED bootstrap on gold58 placement: c4 vs
     a single leg 18-31e-4, leg vs leg 29-43e-4; MODEL Hanley at estimate corr .8: 24-32e-4). Base used below: 22e-4 public, 14e-4
     private.
   - So an independent team's standing relative to us can move **~2.6 display bands** (sd) between public and private.
   - Rare labels carry it: under report prevalence Fracture (27 public positives, per-label sd .032), Synovitis, Lateral Meniscus
     and MCL hold 67 % of the macro variance; under gold58 prevalence MCL, Fracture, Lateral OA and Baker's hold 48 %.
2. **Field now (MEASURED, sec 2).** pw30 is **r141 / 6,276** at 17:10 UTC (3 teams above it inside .959; .960+ = r1-137;
   .959 = r138-211; .958 = r212-1611, 1,400 teams of which 598 sit in exact-tie copy runs). Density: 4.4 teams per 1e-4 in .960,
   7.4 in .959, ~58 in the .958 zone above the copies. **Of the 66 teams at >= .962, 56 were already >= .958 on 10-14 05:27**,
   before any public .958 notebook existed: the top is independent lineages. Of the .959-.961 teams, 108 of 145 are post-wave
   risers (family plus their own additions).
3. **Private rank, base model (MODEL, sec 4; `work/shake_mc_all.log`):**

   | | P(r<=22 gold) | P(r<=100) | P(r<=150) | P(r<=211) | **P(silver, r<=316)** | P(bronze+, r<=633) | median | p10-p90 |
   |---|---|---|---|---|---|---|---|---|
   | pw30 alone | .000 | .17 | .33 | .49 | **.70** | .92 | r216 | r80-548 |
   | pw30 + R6pw20 | .001 | .21 | .37 | .54 | **.74** | .94 | r196 | r72-477 |
   | GUESS-weighted mix of 6 scenarios, alone / pair | .001 | .16 / .20 | .32 / .36 | .48 / .52 | **.69 / .73** | .92 / .93 | | |

   - Range over 17 scenarios: silver .59-.82 alone, .64-.86 pair; P(r<=150) .21-.45 alone. Joint worst / best corners:
     .43 / .91 alone.
   - Expected private rank (capped at 2,000): r302 alone, r273 pair. The distribution is wide and right-skewed: the upper mode is
     "top lineages regress", the tail is "the whole family drops vs the independents".
   - Fixed-margin curve: P(silver) is .41 / .63 / .80 / .92 at a private margin of +5 / +8 / +13 / +20e-4; bronze needs ~+2e-4
     (50 %), i.e. "beat the copy block". Below the copy block we land ~r1,580.
4. **This is more pessimistic than picks2.md** (A alone silver .81, r<=150 .50; pair .877 / .564, CLAIMED) for two reasons:
   - **a common family shock**: the family's public sampling luck vs the independent lineages (one draw, shared by every family
     team) re-appears with fresh private noise. With it switched off, silver goes back to .82 (alone) / .86 (pair). It costs
     ~12 pts of silver and **B does not hedge it at all** (R6pw20 is the same family).
   - **more passers from below**: with lineage noise of 22e-4, lineage teams now at .958 pass us 14-24 % of the time each, and
     the 447 lineage teams at <= .957 add another ~15-25 passers.
   - In exchange, the top regresses harder than field_mc assumed: teams at .959-.962 shrink by 7-12e-4 (EB), so P(r<=100) is higher
     (.17 vs field_mc-like .05).
   - Net: the B hedge is worth **+3.8 pts silver / +4.7 pts P(r<=150)** here, vs +6.6 / +6.2 in picks2. Still positive in all 17
     scenarios (+2.7 to +5.1 silver).
5. **History (CLAIMED, sec 5).** The 742327 study (Mamarin, 11 medal comps with 3,000-6,750 teams): teams just inside the bronze line
   on public medalled 24 % of the time (range 2-51 %), the same rate as teams just outside. Our model gives 39-43 % inside vs 31-34 %
   outside at the bronze line, and 19-23 % vs 18-20 % at the silver line. Same regime ("the public line barely matters"), so the
   model is not over-confident. PC Jimmmy (734681) recalls RSNA comps where the public #1 seldom won but did not drop far; the model
   puts the public #1 at private r2-13 (p10-p90).
6. **Site split (sec 6). The public dose response cannot tell.**
   - Between-site spread of OA-compartment coupling is real (MEASURED, train report labels by language: Q 30-35 on 8 df,
     tau .40-.51 log OR = .70-.85 gold74-se).
   - The public's implied coupling sits at -0.17 [-1.0, +0.6] gold74-se from gold74 (CLAIMED implied2.md), i.e. inside the spread
     of a single site and inside the posterior width. Likelihood ratio random : site split is ~1.1-1.3 (MODEL). Not informative.
   - The E-OA 2-5x underestimate by the offline rulers is a LABEL-REGIME fact (expert image review vs reports / OAI MOAKS), not a
     site fact: the public behaves like gold74-style expert labels from the train sites, not like an external cohort.
   - What a site split WOULD change (MODEL): an extra public-vs-private coupling shift of sd .35 (22 sites) to .69 (9 language
     groups) gold74-se on top of the .52 random-split sampling sd. Per implied2 sec 8.4 that swings pw20 - pw30 by about +-1.6 to
     +-3.3e-4 per .5 sd, in an unknown direction. Under a site split the dose hedge (A at .3, B at .2) and pw25 as the minimax dose
     gain value; the rank model loses ~4 pts silver (site20: .66 alone, .70 pair) and top-150 retention drops .74 -> .65.
   - GUESS P(site split) ~.2-.3: Kaggle's default is a random split, leaderboardPercentage is 30 (CLAIMED PROGRESS_15), and the
     host never answered 734681 (MEASURED unanswered since 08-12).
7. **Decision-relevant reading (GUESS on MODEL):** nothing here argues against the proposed slate. R6pw20 as B stays positive;
   pw25 as an A probe is worth ~+1 pt silver in expectation (its R1 value +0.4e-4, CLAIMED implied2 sec 0.6, x 2-3 pts per 1e-4).
   The largest single risk (the common family shock, sd ~19e-4 relative to the independents) has no hedge among our entries:
   every entry of ours at >= .957 contains the family. Our own early independent models are far below .957, so a non-family B
   would cost more than it hedges.

## 1. Noise (MODEL on MEASURED inputs; `work/hanley_noise.py` -> `hanley_noise.log`, plus picks_cov outputs re-read)

### 1.1 Analytic (Hanley-McNeil), per-label AUC profile ~.96 (GUESS: ACL .99, Lateral Meniscus / Fracture .94, Synovitis .935, rest .955-.97)

| prevalence regime | public 390 abs sd | private 910 abs sd | top-4 labels (share of macro variance) |
|---|---|---|---|
| gold58 (expert-like; MEASURED rates) | 37.9e-4 | 24.8e-4 | MCL .018 (60 pos), Fracture .015, Lateral OA .015, Baker's .015 (48 %) |
| train-report rates (CLAIMED picks_cov) | 50.2e-4 | 32.9e-4 | Fracture .032 (27 pos), Synovitis .025, Lateral Meniscus .022, MCL .018 (67 %) |

Paired sd = abs sd x sqrt(2 (1 - r)) for estimate correlation r:

| r | public (g58 / report) | private (g58 / report) | what it describes |
|---|---|---|---|
| .95 | 12 / 16 | 8 / 10 | near-twin variants |
| .9 | 17 / 22 | 11 / 15 | heavy-overlap blends |
| .8 | 24 / 32 | 16 / 21 | two strong different-lineage ensembles (base) |
| .7 | 29 / 39 | 19 / 26 | two single models of different lineage |

### 1.2 Bootstrap on gold58 placement (MEASURED, re-read of `session_2026-10-19/think/picks_work/macro_*.npy`, B 4,000)

| pair | public report / g58 | private report / g58 | AUC-estimate corr |
|---|---|---|---|
| c4 - copy block | 8.7 / 6.5 | 5.6 / 4.2 | |
| c4 - R6 (uniform) | 8.9 / 7.0 | 5.7 / 4.5 | |
| c4 - single leg (R6 / Raptor / S6) | 23-31 / 18-21 | 15-20 / 12-14 | .84-.92 |
| leg vs leg (R6 / Raptor / S6, genuinely different lineages) | 41-43 / 29-32 | 27-29 / 19-21 | .64-.79 |

- gold58's AUC level (~.93) is below the public's (~.96), which inflates these by ~1.2x (Hanley, 1 - A ratio). Ensembles correlate
  more than single legs. Hence **independent team vs us: ~19-32e-4 public**; base 22e-4 (sigP), per-lineage sd sI = 15.6e-4.
- Private = .655 x public (sqrt(390/910)); the two samples are disjoint, so public and private noises are independent.
- Cross-check: a forum poster simulated a single-submission public sd ~.003 at .955 (CLAIMED varshitha 748011 / 749359): same
  order as our absolute 38-50e-4.

### 1.3 What it means
- The public-to-private change of a single entry (~45-60e-4) is mostly common to all teams: the whole board shifts.
- For rank, only relative noise counts. vs the copy block: private 4-6e-4, so our +13e-4 expected margin is a ~2-3 sd lead.
- vs independent lineages: the relative change has sd sqrt(22^2 + 14^2) ~ 26e-4 before shrinkage. That is why the .959-.962 teams
  are not "safely above" or "safely below" us on private.

## 2. Field structure (MEASURED; `work/lb_struct.py` -> `lb_struct_now.json`, `work/lineage_xtab.py`)

### 2.1 Bands at 17:10 UTC 10-20 (6,276 teams; ties = exact-tie runs of >= 4, increasing submissionId)

| display | teams | ranks | in tie runs | median subs (non-tie) |
|---|---|---|---|---|
| >= .962 | 66 | r1-66 | 0 | 103-341 |
| .961 | 27 | r67-93 | 4 | 90 |
| .960 | 44 | r94-137 | 4 | 86 |
| **.959** | **74** | **r138-211 (pw30 r141)** | 16 | 96 |
| .958 | 1,400 | r212-1611 | 598 (404 in runs >= 8, all at r786-1589) | 39 |
| .957 / .956 | 180 / 272 | r1612-2063 | 30 / 56 | 28 / 14 |

- Teams at >= .959: 159 (10-18 06:45) -> 183 (10-19 10:41) -> 193 (10-20 04:17) -> 208 (13:20) -> **211** (17:10), ~22 per day over
  the last 30 h. Our public rank at the deadline would be ~r165-185 if nothing of ours moves (MODEL, .960+ inflow 10-15 per day).
- Medal cuts now: silver r313, bronze r627. At the deadline (N ~6,335): gold r22, silver r316, bronze r633.

### 2.2 Who is where: current band vs band on 10-14 05:27 (before any public .958 notebook)

| now | was >= .958 | was .957 | was .956 | was < .956 | new team | total |
|---|---|---|---|---|---|---|
| >= .962 | 56 | 1 | 2 | 6 | 1 | 66 |
| .961 | 11 | 3 | 5 | 7 | 1 | 27 |
| .960 | 10 | 8 | 11 | 12 | 3 | 44 |
| .959 | 16 | 14 | 17 | 24 | 3 | 74 |
| .958 | 5 | 53 | 544 | 652 | 146 | 1,400 |

- The top is old independent lineages (56 of 66 at >= .962 were strong before the copy wave); the .959-.961 bands are mostly post-wave risers (family + own additions); .958 is the family.
- The W zone (.958 above the first long copy run, r212-785, 574 teams) holds 49 pre-wave .957 / .958 teams; most jumped exactly with
  the copy wave, so I treat half of them as family tweakers (GUESS, scenario lowIndHi / lowIndLo).

## 3. Model (`work/shake_mc.py`; MODEL with GUESS parameters)

- Every team on the board down to .930 (3,819 teams) gets an unrounded public margin from its band and its rank inside the band
  (pw30 at .95991, u .91 of .959, CLAIMED implied2; c4 at u .78 of .958 with ~30 teams above it, GUESS from 23 at 04:17 UTC).
- Classes (from sec 2): copy (exact-tie runs, 598 at .958), lineage teams (w = non-family share; pre-wave strong w U(.6, 1),
  risers w U(.3, .7) or U(.6, 1) with P .3, 15 % of the W zone and 30 % of the .950-.957 non-tie teams), family tweakers (w < .15).
- public p = t + w (l_i - l_fam) + e + b; private q = t + w (l'_i - l'_fam) + w (g_i - g_fam) + e' + drift + bonus.
  - l = lineage sampling noise (sd sI = sigP / sqrt 2), l' = .655 l, e = family-internal tweak noise (3e-4), b = own-selection
    inflation (beta 4e-4 x U(.5, 1.5) for >= 50 subs), drift = Exp(1e-4) real improvement by the deadline, bonus = .7e-4
    max-of-two picks, g = site term (0 unless site split).
  - t | p by empirical Bayes: an NPMLE prior fitted to the ~800 lineage teams (EM on a 1e-4 grid); tweakers get a parametric
    prior (field_mc style).
  - **l_fam - l'_fam is ONE draw per simulation, shared by every team in proportion to w**: the family's public luck vs the field
    re-drawn on private. Its sd is 1.2 sI ~ 19e-4.
- Copy runs: one private offset per run, N(0, 1e-4). 40 W-zone tweakers become .959-.961 risers by the deadline (inflow 12-25 per
  day).
- Ours: A margin ~ N(13, 7.5)e-4 (between picks2's 15.2 and lbstrat's knob-skeptic ~11); B = A + N(-4, 8.5) (fits picks2's
  P(B > A) .32 and E[max] gain +1.8e-4).
- 12 field replicates x 2,500 private draws per scenario.

EB shrinkage at sigP 22 (MODEL, `work/diag.py`): public -> posterior mean margin, e-4.

| band | lineage teams | mean public | mean posterior | shrink |
|---|---|---|---|---|
| .965-.969 | 21 | 74-114 | 61-92 | -5 to -22 |
| .962-.964 | 45 | 45-65 | 36-55 | -5 to -10 |
| .961 / .960 / .959 | 27 / 44 / 73 | 35 / 25 / 15 | 22 / 16 / 8 | -12 / -9 / -7 |
| .958 / .957 / .956 | 84 / 47 / 64 | 4.5 / -4.4 / -14.9 | 3.0 / 0.2 / -8.1 | -1.5 / +4.6 / +6.8 |

Who passes us at a private margin of +13e-4 (expected count, common shock averaged): >= .962 63, .960-.961 46, .959 28,
.958 lineage 20, <= .957 25, plus ~half of the 40 projected risers.

## 4. Results (`work/shake_mc_all.log`, `shake_mc_all.json`; A = pw30 alone / pair = pw30 + R6pw20)

| scenario | what changes | silver A / pair | P(r<=150) A / pair | P(r<=100) A / pair | median A / pair |
|---|---|---|---|---|---|
| **base** | sigP 22, no family luck, random split | **.705 / .743** | **.327 / .374** | .167 / .206 | r216 / r196 |
| sigP15 | lineage noise low | .763 / .802 | .287 / .344 | .098 / .138 | r208 / r191 |
| sigP30 | lineage noise high | .644 / .682 | .356 / .396 | .235 / .271 | r228 / r204 |
| sigP10 (field_mc-like noise) | | .806 / .841 | .246 / .314 | .051 / .090 | r205 / r189 |
| famLucky | family public luck +.5 sI (+7.8e-4): copied because lucky | .592 / .639 | .215 / .259 | .097 / .126 | r273 / r251 |
| famUnlucky | -.5 sI | .788 / .821 | .454 / .504 | .267 / .314 | r166 / r149 |
| famEB | family treated as a typical lineage at .958 (EB: -4.9e-4) | .750 / .785 | .390 / .438 | .214 / .258 | r190 / r171 |
| site10 / site20 | site term sd 10 / 20e-4 per lineage | .689 / .657 (A), .730 / .696 (pair) | .313 / .296 (A) | .165 / .162 (A) | r222 / r236 (A) |
| usSkeptic / usOptim | our margin N(10, 8) / N(16, 7) | .589 / .800 (A) | .254 / .410 (A) | | r265 / r180 (A) |
| beta0 / beta8 | field own-selection 0 / 8e-4 | .646 / .756 (A) | .257 / .399 (A) | | r249 / r186 (A) |
| noCommon | no common family shock | .821 / .856 | .213 / .295 | .036 / .079 | r205 / r190 |
| newHi | 60 risers + drift 2e-4 | .647 / .690 | .274 / .320 | | r245 / r223 |
| lowIndHi / lowIndLo | more / fewer independents below .958 | .689 / .725 (A) | .335 / .323 (A) | | r214 / r220 (A) |
| worst | sigP 30 + famLucky + site10 + usSkeptic | .426 / .471 | .178 / .210 | .102 / .123 | r365 / r334 |
| best | sigP 15 + famUnlucky + usOptim | .911 / .929 | .504 / .561 | .224 / .284 | r150 / r138 |

- GUESS weights for the headline mix: base .4, famLucky .15, famEB .15, sigP15 .1, sigP30 .1, site20 .1.
- Gold (r22) is ~0 everywhere: it needs a private margin of roughly +55 to +70e-4 over the copy block (GUESS from the posterior
  means of the top 22 lineage teams).
- Lineage noise has opposite effects on the two ends: more noise makes the top regress more (P(r<=100) up) but lets more teams
  from below pass (silver down). The family-luck term moves both ends together and is the largest single swing (+-10 pts).
- The B hedge adds +2.7 to +5.1 pts silver and +3.7 to +8.2 pts P(r<=150) across the 17 scenarios (MODEL): smaller than
  picks2's because the dominant risk is family-common.

## 5. History and calibration (CLAIMED forum + MODEL, `work/calib.py` -> `calib.log`)

- 742327 (Mamarin, 09-28, CLAIMED): in 11 medal comps since 2022 with 3,000-6,750 teams and an ordinary split, teams at public
  ranks 75-100 % of the bronze cut medalled 24 % of the time (range 2-51 %, under half in 10 of 11), and teams just outside it also
  24 %. PC Jimmmy (742327 / 734681, CLAIMED): one comp where most pre-final medals came from one shared notebook that "failed" let
  him jump ~1,000 places; across 6 past competitions he looked at, the public #1's shake-up ranged from none to ~1,000 places, and in
  RSNA comps the public #1 seldom won but did not fall far. Staff (748181, CLAIMED opinion): low-effort public ensembles tend to drop
  on private.
- The model in the same units (one field replicate, 400 private draws):

  | scenario | just inside bronze cut keeps bronze | just outside gets it | just inside silver keeps it | just outside gets it | public top-150 still top-150 | public #1 private rank p10/50/90 |
  |---|---|---|---|---|---|---|
  | base | .40 | .32 | .21 | .19 | .74 | 2 / 7 / 13 |
  | sigP10 | .43 | .34 | .23 | .20 | .89 | 1 / 2 / 4 |
  | sigP30 | .39 | .31 | .20 | .19 | .61 | 2 / 7 / 14 |
  | famLucky | .41 | .33 | .19 | .18 | .74 | 3 / 7 / 13 |
  | site20 | .39 | .32 | .20 | .19 | .65 | 2 / 9 / 21 |

- The line churn comes from the dense .958 family zone (hundreds of teams within ~1e-3), not from lineage noise, so it barely depends
  on sigP. It matches the historical "line barely matters" regime, slightly milder than the 24 % / 24 % average. The model is not
  over-confident at the lines; the open question is only the lineage-noise level at the top, which history cannot pin down.
- RSNA-specific history beyond these posts is not checked here (no download of Meta Kaggle). GUESS from memory: RSNA comps with
  a stage-2 or new-site test had the larger shake-ups.

## 6. Is the public / private split by site? (MEASURED / MODEL; `work/site_coupling.py` -> `site_coupling.log`)

### 6.1 Between-site spread of the quantity E-OA depends on (train report labels v4_blend >= .5, language = site proxy)

| pair | pooled log OR | Q (8 df) | tau (log OR) | tau in gold74-se |
|---|---|---|---|---|
| Medial x Lateral OA | 2.38 | 32.0 | .48 | .77 |
| Medial x PF OA | 2.01 | 29.9 | .40 | .70 |
| Lateral x PF OA | 2.02 | 35.4 | .51 | .85 |

- OA prevalence also varies by site: sd .17-.18 across language groups (Medial OA .27-.72; hr and fr are OA-heavy).
- gold74 expert (n 71 with OA labels): log OR 1.84 / 1.79 / 1.15, se .56-.62. That se is the unit of implied2's regime axis.
- Report-label tau mixes real population differences with reporting style; one central expert review should vary less. So this is
  an upper-bound proxy.

### 6.2 What the public response can and cannot say
- The public's implied coupling regime is -0.17 [-1.0, +0.6] gold74-se (CLAIMED implied2.md sec 2.2): the public labels look like
  gold74-style expert labels from the train sites.
- A public made of ~4-7 whole sites would sit ~.35-.7 sd from the train-site pool; a random 30 % sits ~.3 sd away (sampling).
  Against a posterior sd of ~.62, both predict "near zero". The likelihood ratio random : site is ~1.0-1.3 (MODEL). **No
  information about the split.**
- The 2-5x offline underestimate of E-OA is explained by the label regime: hidden labels are expert image review (host 733491,
  CLAIMED), coupled like gold74 and unlike OAI MOAKS or train reports. It says the public is not an external-cohort-like sample.
  It does not say the private is like the public.
- Other probes: our per-label weight reads (Fracture 1.0, PF OA .1, Lateral OA .1) behaved like random-split noise (lbstrat.md
  class record, CLAIMED). Under a site split, public per-label prevalence would wander by ~.17 between site mixes, but no read
  looked anomalous. Weak evidence for a random split, not a test.
- Never probed: site / metadata offsets (closed as a trap on 10-07, NOVEL_PLAN). Those would have been the direct test.
- Context: 3 acknowledged sites have no train reports (Italy, Mongolia, Liverpool; CLAIMED session_2026-10-19/think/forum.md).
  So the TEST likely holds unseen sites (GUESS P ~.6-.7). That is a statement about test vs train, compatible with a random
  public / private split inside the test. The poster in 734681 speaks of 16 sites (CLAIMED).

### 6.3 What a site split would change (MODEL)
- **E-OA dose:** extra public-vs-private coupling shift sd .35 (22 equal sites) to .69 (9 language groups) gold74-se, on top of
  the .52 random-split sampling sd (total .63-.87 vs .52). implied2 sec 8.4 (CLAIMED): d = +.5 gives pw20 +0.8 to +3.3e-4 over
  pw30, d = -.5 gives pw30 +1.6 to +3.2 over pw20. The direction is unknown, so the expected dose is unchanged but the value of
  holding two doses (A .3 / B .2) and of pw25 as the minimax dose goes up.
- **Rank:** site10 / site20 cost ~1.6 / ~4.8 pts silver (A) and lower top-150 retention from .74 to .65 (site20). The family is
  shared by A and B, so B does not hedge this either.
- **Selection:** under a site split, public-tuned per-label tables (heliosli's, our c4 stack) fit the public site mix; that is
  the famLucky direction (-11 pts silver). It is another reason to keep B off the c4 knob layer (R6 base), as already planned.

## 7. Caveats
- **Lineage noise (sigP) is the key unknown.** It rests on single-leg bootstraps at gold58 placement and Hanley at an assumed
  correlation; no measurement exists for two strong independent ensembles on the hidden labels. Range used 10-30e-4.
- **Class assignment is judgement** (who is a lineage team, w). The scenarios lowIndHi / lowIndLo move silver by only +-2 pts.
- **Family luck has no data**: its prior mean is the selection story (famLucky), the EB story (famEB, opposite sign) or neither
  (base). It is the largest swing and cannot be learned before the deadline.
- **Our margin** N(13, 7.5) mixes picks2's posterior and lbstrat's knob skepticism (GUESS centre). Each 1e-4 of real private
  margin is worth ~2-3 pts silver near +13e-4 (fixed-margin curve, `work/curve_base.log`).
- **Projected arrivals and drift** are linear extrapolations; a deadline surge (newHi) costs ~6 pts silver.
- The model ranks teams by one score per team; Kaggle uses each team's best of 2 selected entries (bonus +.7e-4 for the field,
  max(A, B) for us). Teams that auto-select two near-twins get less; not modelled.
- Gaussian posterior approximation of the NPMLE posterior per team; the top bands have few teams, so the shrinkage there is noisy
  (e.g. .969 -22e-4 from a single team).

## 8. Files (all under `session_2026-10-20/private/work/`)
- `api_lb_1020T_shake.json`: public LB, 17:10 UTC 10-20 (anonymous read).
- `lb_struct.py` -> `lb_struct_now.json`: bands, tie runs, submission counts.
- `lineage_xtab.py`: current band x 10-14 pre-wave band.
- `hanley_noise.py` -> `hanley_noise.log`: analytic noise.
- `shake_mc.py` -> `shake_mc_all.log`, `shake_mc_all.json`: the rank Monte Carlo, 17 scenarios.
- `diag.py`: EB shrinkage by band and passers by group. `curve.py` -> `curve_base.log`, `curve_base.json`: P(rank) at a fixed margin.
- `calib.py` -> `calib.log`: line-churn calibration vs the 742327 study.
- `site_coupling.py` -> `site_coupling.log`: between-site OA coupling heterogeneity and the site-split shift.
