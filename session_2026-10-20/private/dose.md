# dose.md: private-optimal A dose and the final pair under private label-coupling scenarios (10-20 ~22:15-23:10 IST, Claude, PRIVATE track)

RESEARCH ONLY. Nothing was submitted, pushed, committed, versioned or downloaded. No dataset, no GPU job, no scheduled task, no forum
post or vote. PREREG, HANDOFF, PROGRESS, a20 / a21 / val / night were only read. **OAI governance:** this track touched no OAI data and
did not use the box. The only box-derived inputs are aggregates already on disk (`val/work/box/perlabel_{PFT,PAREA}.log`: per-label
mean curves per regime and the pooled n-390 noise covariance, written by the implied2 track). Everything new is local CPU arithmetic
under `private/dose_work/`. The other files in `private/` (fetch_lb_full.py, lb/, raw/, ...) belong to a sibling track and were not touched.
**Tags:** MEASURED (computed, file named), CLAIMED (older note, file named), MODEL (arithmetic on stated assumptions), GUESS (judgement).
**Units:** "margin" = private macro AUC over the heliosli copy block x1e-4 (the picks / picks2 ruler). "priv" = .95802 + margin, i.e. the
private score on the public scale with no common public-to-private shift (picks2 convention). Pair metrics: P(silver) / P(private rank
<= 150) / E[private rank]. "pts" = percentage points. Intervals [p10, p90]. gold58 is info only.

## 0. Answer

1. **For the private board, A should be the conservative dose, not the public max: A = pw20 57425269 (already scored, .959),
   B = R6pw20 (if it is submitted and completes; otherwise R6 57248084).** (MODEL, primary, scenario mix)
   - {pw20, R6pw20}: **P(silver) .887, P(r<=150) .562, E[rank] 242.**
   - {pw30, R6pw20}: .853 / .559 / 283. So pw20 buys +3.4 silver pts and -42 expected places, at +0.3 r150 pts (a tie).
   - Robustness: over 10 model variants x 4 scenarios, pw20 + R6pw20 beats pw30 + R6pw20 on silver in **40/40** cells (+0.7 to
     +7.0 pts) and on E[rank] in **40/40** (-5 to -93). On r150 it is a coin flip: 21/40 (-3.7 to +3.9). In the scenario mix it
     wins silver and E[rank] in 10/10 variants; r150 is -1.1 to +0.7.
   - eoa20 is not the right conservative choice: pw20 beats eoa20 on all three metrics in every variant (pw20 > eoa20 in population
     at P .88-.98, CLAIMED oaigrid.md; here pw20 +0.87e-4 over eoa20 in S0).
2. **Expected private macro is the same for every A; only the tails differ.** S0 (private coupled like public), margin and priv:

   | A | E margin (sd) [p10, p90] | priv | vs pw30 (P > pw30) | P(silver) alone | P(r<=150) alone | E[rank] (median) |
   |---|---|---|---|---|---|---|
   | eoa20 57413368 | +14.72 (9.0) [+3.2, +26.3] | .95949 | -0.92 (.40) | .861 | .483 | 265 (153) |
   | **pw20 57425269** | **+15.59 (9.1) [+4.0, +27.3]** | **.95958** | -0.05 (.49) | **.879** | .520 | **246 (147)** |
   | pw25 (unscored) | +16.14 (10.2) [+3.1, +29.2] | .95963 | +0.50 (.60) | .864 | **.539** | 266 (143) |
   | pw30 57425970 | +15.64 (11.6) [+0.9, +30.5] | .95958 | 0 | .823 | .518 | 321 (146) |
   | R1{pw30, pw25} | +15.98 (10.8) [+2.2, +29.8] | .95962 | +0.34 | .847 | .532 | 289 (144) |

   pw30 carries the widest E-OA spread: population gain over c4 +10.0 [+0.6, +19.5] vs pw20 +10.0 [+3.9, +16.3] (MODEL). Its p10
   margin (+0.9) sits below the silver step (~5e-4, CLAIMED 10-19 field model), pw20's (+4.0) sits near it. That is the whole effect.
3. **Coupling scenarios move the expected private scores, and only one of them favours pw30 (MODEL, primary; e-4 vs pw30 = population + private noise):**

   | scenario | eoa20 | pw20 | pw25 | pw30 margin (priv) | best pair by silver / r150 / E[rank] |
   |---|---|---|---|---|---|
   | S0: same coupling | -0.92 | -0.05 | +0.50 | +15.6 (.95958) | pw20+R6pw20 / pw25+R6pw20 / pw20+R6pw20 |
   | C+: private .5 sd MORE coupled | +1.62 | **+2.43** | +1.81 | +9.1 (.95892) | pw20+R6pw20 on all three |
   | C-: private .5 sd LESS coupled | -3.07 | -2.19 | -0.61 | +21.3 (.96015) | pw20+R6pw20 / R1+R6pw20 = pw30+R6pw20 / HW+pw25 |
   | C++: 1.0 sd more (stress) | +3.82 | +4.59 | +2.95 | +3.4 (.95836) | eoa10+R6pw20 / pw20+R6pw20 / eoa10+R6pw20 |
   | SITE: held-out unseen sites | +0.09 | +0.94 | +1.06 | +13.5 (.95937), sd 18.9 | R6+pw20 / pw25+R6pw20 / HW+pw20 |

   - Only C- favours the high dose, and only on r150 (pw30 + R6pw20 .745 vs pw20 + R6pw20 .719); pw20 still wins silver there (+1.0).
   - Asymmetry: C+ costs pw30 -6.6e-4 of margin, C- gains it +5.7e-4, while pw20 moves -4.1 / +3.6. The curve falls faster past its
     optimum (CLAIMED implied2 sec 8.3-8.4), so any uncertainty about the private regime pushes the private-optimal dose down.
4. **Best pairs (scenario mix S0 .60 / C+ .10 / C- .10 / SITE .20, GUESS weights; primary):**

   | pair | P(silver) | P(r<=150) | E[rank] | E[max] e-4 | note |
   |---|---|---|---|---|---|
   | **pw20 + R6pw20** | **.887** | .562 | **242** | +16.7 | best on silver; E[rank] tied with HW + pw20 (240) |
   | pw25 + R6pw20 (pw25 fixed as A) | .875 | **.573** | 257 | +17.2 | best on r150; needs the pw25 slot to complete; beats pw30 + R6pw20 on all three in 10/10 variants |
   | eoa20 + R6pw20 | .878 | .539 | 252 | +16.1 | dominated by pw20 + R6pw20 |
   | R1{pw30, pw25} + R6pw20 (VAL_1020 plan) | .866 | .568 | 268 | +17.2 | the R1-compliant plan; +1.2 / +0.9 / -15 over pw30 + R6pw20 (10/10 variants) |
   | pw30 + R6pw20 | .853 | .559 | 283 | +17.2 | the plan if pw25 reads below pw30 |
   | R6 + pw20 (no new slot) | .877 | .517 | 244 | +15.9 | best pair from scored entries only |
   | HW + pw20 (no new slot) | .875 | .510 | 240 | +15.8 | |
   | eoa10 + pw30 | .858 | .529 | 262 | | the best B for A = pw30 on silver / E[rank] (a low-dose partner) |
   | R6 + pw30 (A15 as written) | .837 | .517 | 287 | +16.5 | today's pre-registered pair |

5. **What pw30 as A buys, honestly:** upside, not floor. In S0, {pw30, R6pw20} has E[max] +0.45e-4 over {pw20, R6pw20}, P(r<=130)
   .459 vs .440 and median rank 136 vs 138. It pays with silver -2.9 pts, bronze -2.1 pts and E[rank] +31.
6. **Rule status.** A15 / R1 (A = API best-id) is pre-registered; picking pw20 over pw30 is a model-driven override, which R1 exists
   to prevent. picks2.md sec 8 flagged the same signal (+2.3 to +4.0 silver, info only). This file adds that the signal survives
   every coupling scenario and grows when the private labels are more coupled or site-shifted. Eesh's call. If he keeps R1:
   - enter pw25 (R1 picks it with P .56): +1.2 silver, +0.9 r150, -15 E[rank], positive in 10/10 variants;
   - keep B = R6pw20.
   - If A becomes pw20, the VAL_1020 B fallback ("pw20 if R6pw20 shows .957") collides with A, so the fallback has to be R6 57248084.
   - pw25 has value only under R1 or as a fixed-A alternative; r6pw20 remains the most valuable 10-21 slot under every plan.

## 1. Method (MODEL; `dose_work/dose_mc.py`)
- **Public fit = implied2's abc3 primary, re-run with the per-draw parameters kept.**
  - D5 per-label curves from `val/work/box/perlabel_PFT.log` (box aggregates), regime grid raw, R3p -2 to +2.
  - c4 population curve = s x D5 curve at the regime whose D5 gap equals c4's gap (rc4 - q(sh)).
  - Likelihood O1 (eoa10 / eoa20 kernel on `lb_post2.npz`), O2 (frozen A21 ruler), O3 (R1 argmax), O4 (.959 floors). 600k sims.
  - Primary ESS 2,773.
  - Check (MEASURED `dose_base.log` vs CLAIMED implied2 sec 3): pop vs pw30 eoa20 -0.92 (-0.91), pw20 -0.05 (-0.06), pw25 +0.51
    (+0.50), pw35 -1.85 (-1.83); optimum share pw20 / pw25 / pw30 / pw35 .34 / .28 / .23 / .14 (.32 / .30 / .23 / .12);
    public pw30 - eoa20 +1.02 (+1.0).
- **Private.**
  - Population at the private regime sh + d, re-mapped through the same gap rule; amplitude s unchanged. d is in sd of gold74
    log OR, the same scale as the Skeptic's sec 8.4.
  - Private noise = per-label box covariance x 390/910, one draw per sim, shared across scenarios (paired).
- **SITE scenario (GUESS):** public / private split by site with unseen sites in private.
  - Private d ~ N(0, .75), rc4 + N(0, .03), s x logN(0, .30), E-OA private noise x1.25, all independent of the public fit.
  - Our-side c4 / R6 / HW private draws are left unchanged. The forum point that public-tuned tables may lose on unseen sites is
    not modelled; it would favour an R6-based B and does not touch the A dose.
- **Pick layer** (picks2_mc.py conventions, CLAIMED):
  - our side = 10-19 think/picks posterior (`post_base_report.npz`: c4, R6u, HW), paired at random with the E-OA draws;
  - R6pw20 = R6 + k Pop_pw20 + eps + noise (corr .8), k ~ N(.86, .15);
  - display guard: B = R6pw20 if it shows >= .958, else pw20 (P(.957) .006);
  - field = `val/picks2_work/curves_base.npz` (10-19 field model; erank capped at 2000).
  - N = 400k per scenario.
- **Mix weights (GUESS):**
  - S0 .60. The host says test labels come from one expert image-review process (CLAIMED session_2026-10-19/think/forum.md), and a
    random split moves coupling only ~.2-.25 sd, which is already inside the private noise (CLAIMED implied2 sec 8.4).
  - C+ .10 and C- .10.
  - SITE .20 = P(unseen sites in hidden) ~.6-.7 (CLAIMED think/forum.md) x P(site-held-out split) ~.3 (GUESS; question 734681 is
    unanswered).
  - Per-scenario rows are given so the mix can be reweighted.

## 2. Robustness (MEASURED `dose_work/summ_dose.out`, sec 5-8; 10 variants)
Variants: base (P_FT, N(0,1), logN s); PAREA; flatU (flat prior + log-uniform s, the corner most favourable to higher doses);
PAREAflatU; fieldBad; fieldGood; drift1 (field gains 1e-4 by the deadline, about the .959+ inflow); kpess (R6 transfer k N(.6, .3));
seed29 (MC check); siteWide (SITE d sd 1.0, s sd .5, rc sd .05).

| variant | pw30+R6pw20 (mix) | pw20+R6pw20 | pw25+R6pw20 | R1+R6pw20 | eoa20+R6pw20 | R6+pw20 | R6+pw30 |
|---|---|---|---|---|---|---|---|
| base | .853/.559/283 | **.887**/.562/**242** | .875/**.573**/257 | .866/.568/268 | .878/.539/252 | .877/.517/244 | .837/.517/287 |
| PAREA | .872/.574/258 | **.902**/.574/**222** | .891/**.585**/235 | .882/.581/246 | .898/.563/226 | .888/.527/232 | .851/.530/271 |
| flatU | .831/.528/311 | **.862**/.517/**269** | .851/**.533**/285 | .841/.532/297 | .854/.497/279 | .855/.473/267 | .822/.492/305 |
| PAREAflatU | .847/.524/286 | **.877**/.513/**249** | .866/**.528**/263 | .855/.527/276 | .873/.504/253 | .863/.467/258 | .830/.484/294 |
| fieldBad | .863/.581/277 | **.897**/.588/**235** | .884/**.597**/250 | .876/.591/261 | .888/.565/245 | .889/.544/237 | .850/.538/280 |
| fieldGood | .841/.541/290 | **.875**/.540/**248** | .863/**.553**/263 | .854/.549/274 | .865/.517/258 | .863/.495/250 | .822/.499/294 |
| drift1 | .832/.525/310 | **.866**/.521/**265** | .854/**.536**/281 | .845/.532/293 | .856/.497/277 | .852/.476/270 | .811/.483/319 |
| kpess | .845/.544/291 | **.880**/.544/248 | .868/**.558**/263 | .858/.553/275 | .870/.519/259 | .877/.517/**244** | .837/.517/287 |
| seed29 | .853/.559/283 | **.887**/.564/**241** | .875/**.574**/257 | .865/.569/268 | .878/.541/252 | .878/.519/244 | .836/.517/288 |
| siteWide | .842/.553/301 | **.876**/.556/258 | .863/**.566**/275 | .854/.562/286 | .867/.533/268 | .870/.513/**252** | .831/.512/295 |

- Versus pw30 + R6pw20, over 40 variant x scenario cells (S0, C+, C-, SITE):
  - pw20: silver 40/40, r150 21/40, E[rank] 40/40.
  - pw25 fixed: 40/40, 31/40, 40/40.
  - R1{pw30, pw25}: 40/40, 35/40, 40/40.
  - eoa20: 40/40, 9/40, 40/40.
- pw20's r150 deficit is large only in C- (-2.2 to -3.7). In S0 / SITE it is -1.4 to +0.8, and in the mix -1.1 (flat-prior
  variants) to +0.7.
- Seed check: mix numbers move <= .002 and 1 rank.
- Family cross-check (Track A c4D5 OAI grid, CLAIMED implied2 sec 8.4): population pw20 - pw30 at d +.5 is +0.8 to +3.3, at d -.5 is
  -1.6 to -3.8. Here: +1.7 to +2.4 and -1.1 to -2.8. Same sign and size.

## 3. Caveats
- **One model family.** The dose-variance structure (pw30's wider population and private spread) comes from the D5 OAI proxy curves
  and the box n-390 covariance, fitted to five public reads (about 1.5 independent numbers). If the proxy overstates pw30's dose
  tail, the silver edge shrinks. Expected-score differences between the A candidates are within +-0.6e-4 in S0, far below private
  noise; the decision is about tails.
- **Field model** is the 10-19 one (silver step at ~5e-4 margin, r150 at ~15e-4). The silver edge depends on the step sitting near
  pw30's p10. fieldBad / fieldGood / drift1 keep the ranking. E[rank] means are tail-driven (rank cap 2000); medians are 135-150 for
  every top pair.
- **R6pw20** transfer and display are modelled (k from gold58 n 58, CLAIMED picks2); kpess keeps the ranking. Without R6pw20, B =
  R6 or HW, and pw20 is still the better A (R6 + pw20 .877 / .517 / 244 vs R6 + pw30 .837 / .517 / 287).
- **SITE** parameters are guesses. siteWide changes nothing in the ranking.
- **Not modelled:** lomm05 (night slate, optional), host rulings, teams copying E-OA, and an S6fall cliff beyond the knob prior.
- gold58 (info only, CLAIMED VAL_1020 / ext_a22): vs c4, pw20 +.00265, pw25 +.00255, pw30 +.00223. Same direction, no weight given.

## 4. Files (`session_2026-10-20/private/dose_work/`)
- `dose_mc.py`: abc3 re-fit with per-draw sh / s / rc4, coupling-shift and SITE private scenarios, pick layer. Args: prox, psh, ps, M,
  field, fshift, k_mu, k_sd, seed, site_d / site_s / site_rc / site_n, tag.
- `dose_<tag>.log` / `dose_<tag>.json` for base, PAREA, flatU, PAREAflatU, fieldBad, fieldGood, drift1, kpess, seed29, siteWide.
- `run_sens.sh` (variant runner), `summ_dose.py` -> `summ_dose.out` (mix, pair tables, win counts).
- Read only: `val/work/abc3.py`, `val/work/box/perlabel_*.log`, `val/work/lb_post2.npz`, `val/picks2_work/picks2_mc.py`,
  `val/picks2_work/curves_*.npz`, `session_2026-10-19/think/picks_work/post_base_report.npz`, implied2.md, oaigrid.md, picks2.md,
  VAL_1020.md, NIGHT_1020.md, think/forum.md, FORUM_10-20.md, PREREG A15 lines.

## 5. Skeptic (10-20 ~23:15-23:50 IST, Claude, SKEPTIC track)

RESEARCH ONLY. Local CPU only (7 runs of ~2 min). No OAI data, no box, no GPU job; nothing submitted, pushed, committed, posted,
downloaded or registered. PREREG / HANDOFF / PROGRESS / a20 / a21 / val / night were only read. New files only in
`private/skeptic_work/`. Same units and tags as above; "step curve" = the 10-19 field curve dose.md used (`val/picks2_work/curves_base.npz`),
"smooth curve" = shakeup.md's fixed-margin curve (`private/work/curve_base.json`: lineage noise + common family shock). Pair rows are
mix (S0 .6 / C+ .1 / C- .1 / SITE .2) unless marked; B = R6pw20 unless named. Smooth-curve cells: silver / r150 / r100 / bronze.

### 5.0 Verdict
1. **Re-derivation: exact.** Re-running `dose_mc.fit()` plus main()'s draw construction with the same seeds reproduces `dose_base.log`
   to the printed digit (S0 pw20 +15.59 sd 9.10 p10 +3.97, silver .879; mix {pw20, R6pw20} .887 / .562 / 242, {pw30, R6pw20}
   .853 / .559 / 283) (MEASURED `skep_dose_base.log`). The C+ / C- asymmetry (-6.6 / +5.7 vs -4.1 / +3.6) and pop p10 (+0.6 vs +3.9)
   also check.
2. **A = pw20: the direction holds, the size is about half, and most of the edge over R1 disappears.** The +3.4 silver pts headline
   rides on the step curve (P(silver) 0 at +3, .57 at +5, .95 at +6, 1.00 at +8e-4; P(r<=150) .03 at +13, .78 at +16). On the same
   margin draws, the smooth curve (silver .41 / .63 / .80 / .92 at +5 / +8 / +13 / +20) gives (MODEL, `skep_dose_base.log` MIX):
   - pw20 vs pw30 as A: **+2.0 silver, +2.9 bronze, -1.2 r150, -2.4 r100**;
   - pw20 vs R1{pw30, pw25} (A15 kept, pw25 entered): **+1.0 silver, +1.9 bronze, -1.4 r150, -2.1 r100**;
   - pw20 vs pw25 fixed: +0.3 silver, +1.1 bronze, -1.4 r150.
   - The "40/40 cells" of sec 2 all sit on step-like curves (base / fieldBad / fieldGood are one field_mc family), so they are one
     curve family, not 40 confirmations.
3. **Break-even (MODEL):** pw20's smooth-curve silver edge over R1 is gone if pw20's true mean is ~0.45e-4 below what the fit says
   (over pw30 fixed: ~0.85e-4); on the step curve ~1.0 / ~1.7e-4. implied2 sec 8.1 puts population pw20 - pw30 at -1.26 to +0.94
   across priors (CLAIMED), so the edge over R1 is inside the prior uncertainty.
4. **What IS robust: the variance input, and "do not keep pw30 as a fixed A".** pw30's extra margin variance (133.6 vs 82.8 (e-4)^2,
   S0) splits 63 % epistemic population spread (pop sd 7.47 vs 4.86, corr .984) and 37 % private sampling noise (5.50 vs 3.43). Both
   come from pw30's larger paired noise vs c4: box public sd 8.40 vs 5.24 at n 390. Independent cross-check (MODEL on CLAIMED fresh.md
   sec 3.1): the RSNA 3-mean train-ruler bootstrap gives pw30 gain sd 3.6 at n 2,139 -> x sqrt(2139/390) = **8.4** at n 390, and
   pw30 - pw20 sd 1.5 -> **3.5** vs box 3.48. So the dose-variance structure is not an OAI artefact; only the field-curve shape is soft.
   pw30 fixed as A is the worst E-OA A on silver in 6/6 (variant x curve) cells I ran, and on smooth-curve bronze in 3/3.
5. **E[rank] is a cliff metric.** The -42 places are the copy-block cliff: pair P(margin < 0) .044 vs .069 (mix), x ~1,500 places
   below the copy block ~ -38. Quote silver / bronze, not E[rank].
6. **B = R6pw20: confirmed.** Best B for every A on silver and bronze under both curves. Value vs B = R6 on the smooth curve: +1.7
   silver / +2.5 r150 (A pw20), +2.0 / +2.3 (A pw30). R6pw30 (built, 27ce4be7) as B for A = pw20 is -0.5 silver, -1.0 bronze,
   +0.7 r150 vs R6pw20: no change. If A = pw20, the fallback must be R6 57248084 (agree with sec 0.6).
7. **Slate.** r6pw20 is the most valuable 10-21 entry under every policy (agree). pw25 matters only under R1: +1.0 silver (smooth) /
   +1.3 (step) vs pw30-as-A. lomm05 (my model: pw30 + N(+0.41, 1.42) e-4, read sd 1.01, CLAIMED NIGHT): under R1 it is neutral to
   slightly positive (+0.5 / +0.3 silver smooth at mean +0.41 / 0; +0.1 / 0.0 step; +1.1 / +0.7 r150), but it raises the share of
   draws where A ends at a .3 dose from 44 % to 64-69 % (P(A = pw30 / pw25 / lomm05) .18 / .31 / .51), and it has no role if A is
   overridden to pw20. Keep it at priority 3 or drop it; never above pw25.
8. **fresh.md: no fresh idea survives as an entry (agree).** F1 / F2 mildly support a dose <= .25 (train FE pw30 - pw25 -0.5 / -1.1e-4,
   between-site pairs prefer pw20-pw25), but F1 held label coupling at gold74 inside each site, so it says nothing about the d axis
   that drives sec 0.3. F3-F6 stay NO (F4 is LB-probing grey zone).
9. **Scale check (CLAIMED shakeup.md):** the common family shock moves silver by about +-10 pts and nothing we pick hedges it. Every
   A / B difference here is <= 2 pts on the smooth curve.

**My call (GUESS on MODEL).** Default: keep A15 / R1 and make the slate r6pw20 + pw25 (lomm05 optional), B = R6pw20. That captures
about half of the dose-tail gain without breaking the pre-registration. The pw20 override is a legitimate medal-only choice (never
worse on silver or bronze in any cell I ran, +0.2 to +2.1 silver over the R1 plans), but its edge over R1 is <= ~1 pt on the
smooth curve and below the prior uncertainty; if Eesh wants it, it must be written into Amendment 22 BEFORE the 10-21 reads, with B
fallback R6. Not acceptable: pw30 as a fixed A with no pw25 entered.

### 5.1 Pairs on both curves (MODEL, base fit, mix; `skep_dose_base.log`, `skep_dose_base_l0.log`)

| A (B = R6pw20 unless named) | step curve: silver / r150 / E[rank] | smooth curve: silver / r150 / r100 / bronze |
|---|---|---|
| pw20 57425269 | .887 / .562 / 242 | **.768** / .434 / .262 / **.929** |
| pw25 fixed (if COMPLETE, whatever it reads) | .875 / .573 / 257 | .765 / .448 / .280 / .918 |
| R1{pw30, pw25} (VAL_1020 plan) | .866 / .568 / 268 | .758 / .448 / .283 / .910 |
| R1{pw30, pw25, lomm05}, lomm05 mean +.41 / 0 | .867 / .580 / 267 ; .866 / .576 / 268 | .763 / **.459** / **.295** / .911 ; .761 / .455 / .291 / .910 |
| pw30 57425970 fixed | .853 / .559 / 283 | .748 / .446 / .286 / .900 |
| pw20 + R6pw30 | .876 / .565 / 256 | .763 / .441 / .273 / .919 |
| pw20 + R6 (scored only) | .877 / .517 / 244 | .751 / .409 / .241 / .927 |
| pw30 + R6 (A15 as written) | .837 / .517 / 287 | .728 / .423 / .268 / .896 |

- **S0 only, smooth:** pw20 .789 vs pw25 .788, R1 .782, pw30 .772 (silver).
- **Variants, smooth, silver / bronze (MODEL `skep_dose_{flatU,PAREA}.log`):** flatU pw20 .740 / .912, pw25 .738 / .901, R1 .732 / .892,
  pw30 .725 / .884; PAREA .781 / .941, .778 / .931, .772 / .924, .764 / .916. r150 goes the other way in both (pw20 -1.4 to -1.9 vs
  R1 / pw25 / pw30).
- **Curve-width sweep** (probit silver step at +6e-4 with the copy-block cliff at 0; pw20 pair minus pw30 pair / minus R1 pair, silver
  pts): tau .5 +3.4 / +2.1, tau 4 +3.0 / +1.8, tau 6 +2.5 / +1.4, tau 10 +1.7 / +0.8. The edge shrinks monotonically as the field gets
  noisier; it never changes sign at the model's own means.
- **Break-even shift of pw20's mean** (R6pw20 shifted by .86 x the same): -0.5e-4 -> .757 (smooth) / .878 (step); -1.0 -> .745 /
  .867; -1.5 -> .733 / .857.

### 5.2 Caveats of this pass
- The smooth curve is shakeup.md's base field (sigP 22, family shock on); its own range (sigP 10-30) would move the width. The tau
  sweep brackets it.
- lomm05 is modelled crudely (Gaussian delta on pw30, independent of the coupling scenarios). Its NIGHT skeptic's OAI-only mean
  (-0.9) is not run; there it would be negative.
- Same single OAI-proxy fit and 10-19 our-side posterior as sec 1; only the field curve and the extra entries are new.

### 5.3 Files (`session_2026-10-20/private/skeptic_work/`)
- `skep_dose.py`: imports `dose_work/dose_mc.py`, re-runs fit + draws with the same seeds, adds lomm05 / R1{pw30, pw25, lomm05} /
  R6pw30 / break-even rows, scores every entry and pair on the step curve, the smooth curve and a probit tau sweep.
- `skep_dose_{base,base_l0,flatU,PAREA}.{log,json}` (base_l0 = lomm05 mean 0; flatU / PAREA logs predate the R6pw30 and break-even rows).
