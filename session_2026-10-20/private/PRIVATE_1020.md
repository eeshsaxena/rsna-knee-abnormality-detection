# PRIVATE_1020.md: private-first synthesis for the final picks and the 10-21 slate (10-20 ~23:50 IST, Claude, PRIVATE synthesis)

RESEARCH ONLY. Nothing was submitted, pushed, committed, versioned, downloaded, posted or voted. No dataset, no GPU job, no scheduled-task
registration or change. PREREG, HANDOFF, PROGRESS and the a20 / a21 / val / night folders were only read. **OAI:** this synthesis touched
no OAI data and did not contact the box; every OAI-derived number below is an aggregate already on disk, quoted from a sibling note.
**Tags:** MEASURED (computed or read now, source named), CLAIMED (sibling or older note, or a forum poster, named), MODEL (arithmetic /
simulation on stated assumptions, file named), GUESS (judgement). **Units:** e-4 = 1e-4 macro AUC. "Margin" = private macro over the
heliosli copy block (picks / picks2 ruler). "pts" = percentage points. "Step curve" = the 10-19 field model used by picks2 / dose.md;
"smooth curve" = shakeup.md's fixed-margin curve (lineage noise + common family shock). Silver / bronze = P(private medal).
**Sources (this folder):** `forum_lb.md` (forum + LB track), `dose.md` sec 0-4 (dose track) and sec 5 (Skeptic), `shakeup.md`
(shake-up track), `fresh.md` (fresh-ideas track); older: `val/VAL_1020.md`, `night/NIGHT_1020.md`, `val/implied2.md`, `val/picks2.md`.

## 1. Verdict (plain)

1. **Slate, 05:31 IST 10-21 (00:01 UTC): r6pw20 (priority 1) + pw25 (priority 2). Drop lomm05.** Slots 3-5 empty; 10-22 slots only for an ERROR retry.
2. **A, private-first: pw20 57425269, fixed now** by a written override of A15 / R1 in Amendment 22, before any 10-21 score is seen.
   It is best or tied-best on silver and bronze in every model cell run tonight: +2.0 silver / +2.9 bronze vs pw30 fixed, +1.0 / +1.9 vs R1 (smooth curve; MODEL dose.md sec 5).
   Its cost is -1.2 to -1.4 pts P(r<=150) and about -2 pts P(r<=100), which carry no medal. The edge is small and rests on one model family.
3. **Acceptable alternative: keep A15 / R1** (A = API best-id of pw30 / pw25). It is within about 1 silver pt of pw20 and keeps the pre-registration. It needs pw25 entered.
   **Not acceptable: pw30 as A with no pw25 entered** (worst A on silver in 6/6 Skeptic cells).
4. **B = R6pw20** if COMPLETE and it displays >= .958; on ERROR one resubmit 10-22; otherwise **R6 57248084** (mandatory if A = pw20).
5. **Expectation (MODEL):** P(silver) about .73-.77 on the smooth curve (.85-.89 on the step curve), P(bronze+) about .92, gold about 0.
   Median private rank about r200, p10-p90 about r70-r480. The largest risk, the family's shared luck vs independent teams (+-10 silver pts), has no hedge among our entries.
6. **Forum / LB:** no host news, rules unchanged. We are public r141, .959, inside the silver zone. Nothing blocks a submit (MEASURED forum_lb.md).
7. **Eesh: GO + A rule by about 04:30 IST 10-21** (sec 6). If he gives no GO, nothing is armed and nothing submits.

## 2. Forum and LB news (read 17:06-17:10 UTC = 22:36-22:40 IST; MEASURED forum_lb.md unless tagged)

- **Host / staff:** no post since the 10-19 22:46 UTC external-data non-ruling (747455). Every thread with a post after 04:00 UTC was fetched (9 + 2 new); no host author.
- **Rules / metadata:** unchanged: deadline 10-22 23:59 UTC, 2 scored picks, 5 per day, public = 30 %, 3-decimal display. Teams 6,249 -> 6,275; submissions +2,757 in about 12.8 h.
- **Watch list:**
  - Unchanged items:
    - site split 734681: still unanswered, last post 08-14;
    - missing planes 744519;
    - Synovitis rubric 737155;
    - OA / label regime 745949.
  - External data: still GRAY, Eesh's call.
  - There is no thread on final selection or rerun.
- **New threads:**
  - 751021 (r2506) asks what private will look like given few expert labels. It has 0 replies.
  - 751027 (r50, CLAIMED) reports Kaggle dataset uploads failing for about 2 h.
    - Google Cloud status shows one closed europe-west2 incident of about 7.9 h on 10-20. The Kaggle impact is unknown.
    - Our 10-21 kernels attach the same 5 existing datasets and need no upload (MEASURED kernel metadata).
    - FYI: check the push rc at the 05:30 IST run and allow one retry.
- **Private board:** no information. The API returns the public board only, and privateScore is blank on all our subs.
- **Our row (17:08 UTC):**
  - **r141 / 6,276, .959, best sid 57425970 (pw30).** We were r140 at 13:20 UTC.
  - We are 4th of 74 in the .959 band (r138-211). The .960+ band is r1-137.
  - The silver line is r313 and displays .958, so we are 172 places inside it.
- **Since 13:20 UTC:**
  - One team passed us (OsakanaFish .959 -> .961).
  - Three new .959 entrants are all below us.
  - 18 teams crossed into .959+ since 04:17 UTC, none fell out.
- **Pace (MODEL forum_lb.md, shakeup.md):**
  - .960+ inflow is 6-24 per day, so +29 to +55 teams above us by the deadline. pw30 alone would end near public r165-200. Public rank does not decide anything.
  - At the deadline (N about 6,335) the medal cuts are gold r22, silver r316, bronze r633.
- **Scoring lag:** about 5.5-6.5 h (MODEL). 05:31 IST submits should read about 11:00-12:00 IST 10-21.
  - A15 reads R1 at the last read before 10-22 18:00 IST, so the last readable submit is about 10:15 IST 10-22 (CLAIMED VAL_1020).
- **Automation state (MEASURED this pass):**
  - A22_GO.txt does not exist and PREREG has no Amendment 22 yet.
  - RSNA_A20_tick and RSNA_A21_tick are still enabled. Their slates are fully submitted (tick logs) and both scripts stop at 05:20 IST 10-21 (STOP in code), so they are no-ops and cannot submit.
  - No RSNA_A22_tick is registered.

## 3. Dose and pair table under scenarios (MODEL dose.md + Skeptic sec 5; one OAI-proxy curve family fitted to 5 public reads)

Scenarios: S0 = private labels coupled like public (weight .6); C+ / C- = private OA-trio coupling .5 sd more / less (.1 each); SITE =
public / private split by site with unseen sites in private (.2); C++ = 1.0 sd more (stress, weight 0). Weights are GUESS.

### 3.1 A alone: expected gap vs pw30 (e-4) and P(silver) alone (step curve)

| scenario | pw30 margin | pw20 gap | pw25 gap | eoa20 gap | P(silver): pw20 / pw25 / R1{pw30,pw25} / pw30 |
|---|---|---|---|---|---|
| S0 (.6) | +15.6 (sd 11.6, p10 +0.9) | -0.05 | +0.50 | -0.92 | .879 / .864 / .847 / .823 |
| C+ (.1) | +9.1 | **+2.43** | +1.81 | +1.62 | .769 / .723 / .690 / .641 |
| C- (.1) | +21.3 | -2.19 | -0.61 | -3.07 | .920 / .916 / .910 / .902 |
| SITE (.2) | +13.5 (sd 18.9) | +0.94 | +1.06 | +0.09 | .770 / .737 / .717 / .688 |
| C++ (stress) | +3.4 | +4.59 | +2.95 | +3.82 | .630 / .550 / .504 / .439 |
| MIX | +15.1 | +0.18 | +0.63 | -0.68 | .850 / .830 / .811 / .786 |

- Expected private is the same for every A (+-0.6e-4 in S0). The choice is about tails: pw30's p10 margin (+0.9) sits below the silver step (about +5e-4); pw20's (+4.0) sits at it.
- The response is lopsided: more coupling costs pw30 6.6e-4, less coupling gains it 5.7e-4. Only C- favours pw30, and only on P(r<=150) (.745 vs .719 as pairs); pw20 still wins silver there.
- **pw30's extra spread is not an OAI artefact (Skeptic, MODEL on MEASURED fresh.md):** the RSNA train-ruler bootstrap scales to a public noise sd of 8.4 (box 8.40), and 3.5 for pw30 - pw20 (box 3.48).

### 3.2 Pairs with B = R6pw20 by scenario (step curve): silver / P(r<=150); MIX adds E[rank]

| pair | S0 | C+ | C- | SITE | C++ | MIX |
|---|---|---|---|---|---|---|
| **pw20 + R6pw20** | **.914** / .579 | **.825** / **.399** | **.939** / .719 | **.813** / .516 | .704 / .253 | **.887** / .562 / **242** |
| pw25 + R6pw20 (pw25 fixed) | .905 / **.592** | .798 / .393 | .937 / .744 | .792 / **.522** | .655 / .235 | .875 / **.573** / 257 |
| R1{pw30,pw25} + R6pw20 (VAL plan) | .896 / .587 | .781 / .382 | .934 / **.745** | .782 / .518 | .632 / .223 | .866 / .568 / 268 |
| pw30 + R6pw20 | .885 / .577 | .757 / .365 | .930 / .745 | .768 / .511 | .600 / .207 | .853 / .559 / 283 |
| R6 + pw20 (scored entries only) | .896 / .528 | .810 / .358 | .934 / .676 | .825 / .485 | .709 / .230 | .877 / .517 / 244 |
| R6 + pw30 (A15 as written today) | .859 / .529 | .731 / .318 | .925 / .714 | .780 / .481 | .602 / .178 | .837 / .517 / 287 |

### 3.3 Same pairs on the smooth curve (Skeptic, MIX): silver / bronze / P(r<=150) / P(r<=100)

| A (B = R6pw20 unless named) | smooth | step (silver / r150 / E[rank]) |
|---|---|---|
| **pw20 57425269** | **.768** / **.929** / .434 / .262 | .887 / .562 / 242 |
| pw25 fixed | .765 / .918 / .448 / .280 | .875 / .573 / 257 |
| R1{pw30, pw25} | .758 / .910 / .448 / .283 | .866 / .568 / 268 |
| R1{pw30, pw25, lomm05} (lomm05 mean +.41) | .763 / .911 / **.459** / **.295** | .867 / .580 / 267 |
| pw30 fixed | .748 / .900 / .446 / .286 | .853 / .559 / 283 |
| pw20 + R6pw30 | .763 / .919 / .441 / .273 | .876 / .565 / 256 |
| pw20 + R6 | .751 / .927 / .409 / .241 | .877 / .517 / 244 |
| pw30 + R6 (A15 as written) | .728 / .896 / .423 / .268 | .837 / .517 / 287 |

**Reads (MODEL unless tagged):**
- **The size of the pw20 edge:** dose.md's headline +3.4 silver pts for pw20 over pw30 is about half that on the smooth curve: +2.0 silver, +2.9 bronze vs pw30, and +1.0 / +1.9 vs R1.
  - dose.md's "40/40 cells" all sit on one step-like curve family, so they are not 40 independent confirmations (Skeptic).
- **Break-even:** pw20's edge over R1 is gone if its true mean is about 0.45e-4 lower than the fit (smooth curve).
  - That shift is inside implied2's prior range for pw20 - pw30 (-1.26 to +0.94, CLAIMED).
  - Yet pw20 still wins silver and bronze in the flatU prior, the corner most favourable to high doses (.740 / .912 vs R1 .732 / .892).
- **The tail risk of a high dose is robust; its mean is not.** pw30's extra variance is 63 % population spread and 37 % private noise.
- **E[rank] is a cliff metric.** It is driven by P(margin < 0), which is .044 vs .069 for the pw20 vs pw30 pairs. Quote silver and bronze, not E[rank].
- **Independent private lenses point the same way:**
  - **Train FE site meta-analysis** (MEASURED fresh.md): pw30 - pw20 is -0.1 / -0.9e-4 and pw30 - pw25 is -0.5 / -1.1e-4 (v2 / v4 readers). Between-site pairs prefer pw20-pw25.
  - **gold58** (info only, CLAIMED VAL_1020): vs c4, pw20 +.00265, pw25 +.00255, pw30 +.00223.
  - None of these is strong on its own.
- **B = R6pw20 is the best B for every A** on silver and bronze, under both curves.
  - Its value vs B = R6 is +1.7 to +2.0 silver and +2.3 to +2.5 r150 (smooth).
  - R6pw30 as B for A = pw20 is -0.5 silver and -1.0 bronze vs R6pw20.
- **If no slate runs at all:** the pick rule alone matters. {pw20, R6} beats A15's {pw30, R6} by +4.0 silver on the step curve and +2.3 on the smooth curve.

## 4. Shake-up expectation (MODEL shakeup.md: empirical-Bayes field, 17 scenarios; one LB read 17:10 UTC)

- **Noise that moves ranks:**
  - vs a family twin (the copy block, c4): private sd 4-6e-4, so our expected +13-16e-4 margin is a 2-3 sd lead over the copies.
  - vs an independent-lineage team: private sd about 13-21e-4. Relative standing can move about 2.6 display bands between public and private.
  - Rare labels (Fracture, Synovitis, Lateral Meniscus, MCL) carry about 2/3 of the macro variance under report prevalence.
- **Field shape (MEASURED):**
  - 56 of the 66 teams at >= .962 were already >= .958 before any public .958 notebook existed: the top is independent lineages.
  - The .959-.961 bands are mostly post-wave risers.
  - The .958 band (1,400 teams) is the family: 598 sit in exact-tie copy runs.
- **Private rank for pw30 + R6pw20** (shakeup base; pw30 alone in brackets):
  - P(silver) **.74** (.70), range .64-.86 over scenarios;
  - P(bronze+) .94 (.92);
  - P(r<=150) .37 (.33), P(r<=100) .21 (.17);
  - median r196 (r216), p10-p90 r72-477;
  - gold about 0: it needs about +55-70e-4 over the copies.
  - The GUESS-weighted 6-scenario mix gives silver .73 for the pair and .69 alone.
- **The pw20 pair on the same field:** the Skeptic's smooth-curve rows (sec 3.3) give silver .768 vs .748 for the pw30 pair. Same regime, about +2 pts.
- **Why this is below picks2** (.81 alone / .877 pair, CLAIMED):
  - **One shared family shock:** the family's public luck vs the independents is one draw, with sd about 19e-4. It costs about 12 silver pts, and B cannot hedge it.
  - **Passers from below:** lineage teams now at .957-.958 can pass us on private.
  - **In exchange,** the .959-.962 teams shrink 7-12e-4 on private, so P(r<=100) is higher.
- **Value of B (R6pw20):** +3.8 silver pts (+2.7 to +5.1 over scenarios), against +6.6 in picks2.
- **Calibration:** the model reproduces the CLAIMED 742327 history (11 comps) that the public medal line barely predicts private medals.
  - At the bronze line it gives 40 % inside vs 32 % outside; at the silver line 21 % vs 19 %.
  - It puts the public #1 at private r2-13.
- **Site split: the public cannot tell (likelihood ratio random : site about 1.1-1.3).** GUESS P(site split) .2-.3.
  - If it is a site split: silver drops about 2-5 pts, and holding two doses (A low, B .2) plus pw25 as the minimax dose gains value.
  - The 2-5x offline underestimate of E-OA is a label-regime fact (expert image review), not a site fact.
- **The weakest inputs:**
  - The lineage-noise level (10-30e-4).
  - The family's public luck, which has no data and is the largest single swing (+-10 silver pts).

## 5. Fresh ideas and Skeptic outcome

**Fresh (fresh.md, local CPU, RSNA train OOF raked to the gold74 expert OA joint; report language as the site proxy, 9 groups):**

| # | idea | result | verdict |
|---|---|---|---|
| F1 | Site stability of the shipped E-OA dose response (and lomm05) | MEASURED: 60 Cochran Q tests, min p .125. tau point 0 for every dose difference; 90 % upper 1.7-2.3e-4 (pw30 - pw25), 5.3-6.1e-4 (pw30 - pw20). Pooled parity with adj_c12 exact | stable; no lever |
| F2 | Within-site vs between-site E-OA gain | MEASURED: pw30 +17.6 within vs +8.4 between (v2); +10.6 / +5.1 (v4). Not a site-offset artefact; between-site pairs prefer pw20-pw25 | robust; mildly favours dose <= .25 |
| F3 | Header / OOD-gated E-OA (skip on out-of-envelope scans) | not measurable (no < 1.2 T train study); OAI analog flips sign with coupling | NO |
| F4 | Hidden-composition probe kernel | not run; no decision depends on it; LB-probing grey zone | NO |
| F5 | Silent member-degradation guard | needs GPU anchor dumps | NO |
| F6 | Rare-label stability of pending entries | pending entries touch only the OA trio | nothing to do |

- The one private-specific risk left is out-of-envelope sites (Italy, Mongolia, Liverpool have no train reports, CLAIMED). Its expected cost to pw30 vs pw25 is about 0.1e-4 (MODEL), and even the sign is uncertain.

**Skeptic (dose.md sec 5; local CPU, 7 runs):**
1. The re-derivation is exact to the printed digit (MEASURED skep_dose_base.log).
2. A = pw20: the direction holds, but the size is about half and most of the edge over R1 disappears.
   - pw20 is never worse on silver or bronze in any cell: +0.2 to +2.1 silver over the R1 plans.
   - Its edge over R1 is <= about 1 pt on the smooth curve, which is inside the prior uncertainty.
3. B = R6pw20 is confirmed. If A = pw20, the fallback must be R6 57248084.
4. Slate:
   - r6pw20 is the most valuable 10-21 slot under every policy.
   - pw25 is worth +1.0 to +1.3 silver, but only under R1.
   - lomm05 is neutral to +0.5 silver under R1, but it raises the share of draws where A ends at a .3 dose from 44 % to 64-69 %. It has no role if A = pw20, and its own skeptic's OAI-only mean (-0.9) would make it negative. Keep it at priority 3 or drop it.
5. fresh.md yields no new entry. F1 held coupling fixed within each site, so it says nothing about the C+ / C- axis.
6. Scale: every A / B difference is <= 2 pts on the smooth curve, against a +-10 pt family shock nobody can hedge.

**Skeptic's own call (GUESS on MODEL):** keep A15 / R1 by default, with the slate r6pw20 + pw25 and B = R6pw20. The pw20 override is legitimate if only medals matter, but it must be written into Amendment 22 before the 10-21 reads. Never pw30 fixed without pw25.

**How this synthesis resolves it (GUESS on MODEL):**
- **Where the Skeptic and I agree:** the slate, B, the fallback, and "never pw30 fixed".
- **Where I lean one step further, to pw20:** the medal is the objective, and gold is about 0.
  - Every lens with private relevance leans to a dose <= .25: coupling asymmetry, the train site meta-analysis, gold58 (info) and the tail structure.
  - R1 picks between pw30 and pw25 on a public paired-read sd (about 1.4e-4, CLAIMED VAL_1020) that is larger than their expected true difference (about 0.5e-4, MODEL dose.md). That makes it close to a coin flip (P(pw25 chosen) .56) at the dose that carries the tail.
- **The honest size is about +1 silver / +2 bronze pts.** That is small enough that keeping R1 is a defensible second choice.
- **This updates VAL_1020's "do not override A to pw20".** That line rested on pw20 - pw30 changing sign across priors, and that is still true of the mean. Tonight's private tracks show the decision runs on the tail, and pw20 wins the tail in every prior tried.

## 6. What Eesh must approve, and by when

Times are IST, with UTC in brackets. Nothing below is done or armed (MEASURED: no A22_GO.txt, no Amendment 22, no A22 tick).

| # | item | by when | notes |
|---|---|---|---|
| 1 | **The GO.** Main writes `session_2026-10-20/val/A22_GO.txt` on his word: `GO r6pw20 pw25 MAX=2 Eesh 10-21 hh:mm "<his words>"` | **~04:30 IST 10-21 (23:00 UTC 10-20)**; tonight is better | Key order = push / submit priority. Minimum fallback: `GO r6pw20 MAX=1`. Do not name lomm05 |
| 2 | **The A rule (his decision).** P: A = pw20 57425269 fixed (overrides A15 / R1). R: keep A15 / R1 (API best-id of pw30 / pw25) | Same time as #1. At the latest before the first 10-21 read (~11:00 IST). Never after seeing a 10-21 score | P is the private-first recommendation; R is acceptable. If he says nothing, R stands (pre-registered) |
| 3 | **PREREG Amendment 22**, appended by main to `session_2026-10-15/PREREG_10-15.md`. The literal `AMENDMENT 22` is required by the tick guard | With #1 | Use the VAL_1020 sec 5 draft (r6pw20, pw25, gates). Replace its "Picks" line by P or R (below). Add "lomm05 not entered" |
| 4 | **Two GPU commits** (r6pw20, pw25): about 20 min each, run in parallel, Kaggle GPU quota | Right after #1-#3 | No dataset upload needed. If 751027-style Kaggle trouble shows, allow one push retry |
| 5 | **Register `RSNA_A22_tick`** (every 10 min + one-shot 05:31:15 IST; command in VAL_1020 sec 5 item 4) | With #1 | Delete it after the slate. The A20 / A21 ticks are idle no-ops that stop at 05:20 IST (MEASURED) |
| 6 | **Two submits** in the 10-21 UTC day, by the tick at 05:31 IST | auto | Reads about 11:00-12:00 IST. A late GO still works until 05:29 IST 10-22; only the reads slip |
| 7 | **ERROR retry policy for 10-22.** Needs a fresh GO; the tick window closes at 05:20 IST 10-22, so a retry is a manual submit or a moved window | Only if r6pw20 ERRORs | Last readable submit ~10:15 IST 10-22 (A15 reads before 18:00 IST) |
| 8 | **Final hand selection** of A and B in the Kaggle UI | **Before 05:29 IST 10-23 (23:59 UTC 10-22)**; suggest 10-22 ~18:30-20:00 IST, after the last read | Auto-selection would pick two c4 near-twins. Do it by hand |

**Picks line for Amendment 22, option P (private-first):**
> - Picks, fixed before any 10-21 score: A = pw20 57425269. This overrides A15 / R1 as a private-first dose choice; evidence in session_2026-10-20/private/PRIVATE_1020.md and dose.md sec 0 + 5; Eesh: "<quote>". B = r6pw20 if COMPLETE and it displays >= .958. If it ERRORs, one resubmit on 10-22; otherwise B = R6 57248084. pw25 has no pick role. r6pw20 is never A. lomm05 is not entered. No dose step above pw30.

**Picks line for Amendment 22, option R (keep R1):**
> - Picks, fixed before any 10-21 score: A = API best-id among pw30 57425970 and pw25 (A15 / R1, last read before 10-22 18:00 IST). B = r6pw20 if COMPLETE and it displays >= .958. If it ERRORs, one resubmit on 10-22; otherwise B = [pw20 57425269 (model) or R6 57248084 (keeps A15's other-side clause)]. r6pw20 is never A. lomm05 is not entered. No dose step above pw30, whatever pw25 reads.

**If Eesh approves nothing:** nothing submits. The pre-registered A15 pair, {pw30, R6}, stays in force; it is the weakest pair in sec 3 (.837 step / .728 smooth silver). A pick-rule amendment alone (A = pw20, B = R6, no new slot) still recovers +4.0 step / +2.3 smooth silver pts of that gap (MODEL sec 3); adding r6pw20 as B is worth another +1.0 / +1.7.

## 7. Caveats

- **One model family.**
  - All dose numbers come from one OAI-proxy curve family plus 5 public reads (about 1.5 independent numbers).
  - The field curves are two models (the step curve and shakeup's smooth curve).
  - R6pw20's transfer is modelled (k from gold58, n 58).
  - lomm05 is modelled crudely.
- **Our ranks are all MODEL.** Our private margin is not observable before the deadline, and the private board API returns nothing.
- **The site analysis uses report language as the site proxy.** It cannot see out-of-envelope sites. The SITE scenario's parameters are GUESS.
