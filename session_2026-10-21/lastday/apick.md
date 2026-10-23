# apick.md: is swide the right A for the private board? (10-21 ~19:20-20:45 IST, Claude, LASTDAY track A)

RESEARCH + LOCAL BUILDS ONLY. Nothing was submitted, pushed, committed, versioned, downloaded or scheduled. No dataset, no GPU job, no
box use at all (the box was not needed), no OAI rows, ids or per-knee values touched, no Kaggle or LB call. PREREG, HANDOFF, PROGRESS
and every older session folder were only read. All new files are in `session_2026-10-21/lastday/`. Local CPU only, 2 threads, about 25
light MC runs of ~35 s each. gold58 is info only.

**Tags.** MEASURED = read or computed now, file named. CLAIMED = older note, named. MODEL = arithmetic on stated inputs. GUESS = judgement.
**Units.** E = private margin, 1e-4 macro over the heliosli copy block (picks2 / dose / readtable / bpick ruler). Pub-scale private =
.95802 + E x 1e-4. "pts" = percentage points. Silver = P(private rank <= 315) on two field curves: step (10-19 field model) and smooth
(shakeup.md fixed-margin curve incl. family shock). P(r<=150) on both. Bronze on the smooth curve. "Pair" = max(A, B) private.

## 0. Answer

1. **Keep A = swide 57464326.** It is the A22 answer under R1 and under H2 (a stack won), and the model does not justify overriding it.
   (MODEL, 26 variants + a 15-world prior grid averaged 6 ways; sec 2-4)
2. **pw25, pw30 and eoa20 are never the right A.**
   - pw30: worse than swide on silver by 1.2 to 4.2 pts in every prior weighting (it carries the widest dose tail). DEAD.
   - eoa20: worse on r150 (-2.8 to -8.6 pts), E (-0.9 to -2.5e-4) and smooth silver (-0.4 to -3.6) in every weighting; step silver
     -1.6 to +0.7. DEAD.
   - pw25 = swide minus the stack. It beats swide only if the stack (lomm05 + MCLmm) is net harmful privately, and in exactly those
     worlds pw20 beats pw25 on silver. Its best case (EB-heavy weighting) is +0.4 to +0.8 silver, +0.3 to +1.2 r150. DEAD as A.
3. **The only real alternative is pw20 57425269 (the A22 private-first clause), and it is a split, not a win:**
   - vs swide it gains step silver (+0.8 to +2.0 pts) and bronze (+0.8 to +1.6) whenever at least ~20 % of the prior sits on
     "stack harmful" worlds; it ties step silver under the A22 priors (-0.2 / +0.1).
   - it loses smooth silver under every weighting except skeptic / EB-heavy (-0.2 to -1.8), and loses r150 (-0.5 to -5.4 pts) and E
     (-0.4 to -1.7e-4) under EVERY weighting.
   - Averaged over the two silver curves at my central weighting ("levers", sec 3): pw20 +0.15 to +0.25 silver (a tie), swide +2.4 to
     +3.1 r150 and +0.9 to +1.1e-4 E. Ties on the medal, losses on everything else: keep swide.
4. **Decision rule (fix it now, apply mechanically at selection):**
   - **A = swide** unless BOTH hold: (a) the goal is P(silver) only (rank inside silver does not matter), and (b) Eesh believes the swide
     extras are more likely to hurt than help privately (private stack E <= 0; >= ~40 % weight on the MCLmm-EB / lomm05-OAI-negative
     worlds). Then **A = pw20** (not pw25): +1.3 to +2.0 step / +0.3 to +1.0 smooth silver, at -0.5 to -1.9 r150 and -0.4 to -0.75e-4 E.
   - Under (b) with a rank goal instead, pw25 edges swide by <= 1.2 r150 pts; not worth a rule.
   - The B rule is unchanged and independent of the A choice: B = earliest clean r6pw20 copy >= .958 if it exists, else R6 57248084.
     The A ranking is the same at P(clean) 0 and 1, and the pair metrics are linear in P(clean) (sec 3.2).
5. **Robustness** (26 variants, B = R6; MEASURED from `apick_*.json`). swide is best or within 1 pt of the best A on smooth silver in
   22/26, on r150 in 21/26 (step) / 22/26 (smooth), and within 0.5e-4 of the best E in 23/26. On step silver it is within 1 pt in 16/26;
   it is behind pw20 by > 1 pt only in 10 skeptic-component variants (+1.3 to +5.7, the max in the "both components negative and
   private independent of public" corner). With B = clean r6pw20 the counts are 16 / 22 / 22 / 23 / 23. Max regret across the 6 prior
   weightings (sec 3): swide 2.0 step / 0.9 smooth silver, 1.2 r150, 0.24e-4 E; pw20 0.2 / 1.8 silver, 5.4 r150, 1.7e-4 E.
6. **How much of swide > pw25 is real?** (MODEL, sec 1)
   - Base posterior: public swide - pw25 = +2.03e-4; population +1.10 (54 %: lomm05 +0.44, interaction +0.16, MCLmm +0.50), read noise
     +0.94 (46 %, mostly MCLmm read noise +0.78). Private stack +1.08, P(< 0) .33 (prior +0.24, P(< 0) .46).
   - The read is weak evidence about the stack: across 15 stack priors its likelihood ratio spans 0.46-1.27 (max Bayes factor 2.8,
     base vs MCLmm-EB only 1.3). So the prior decides. Prior-weighted private stack E after the read: A22 priors +1.08, optimist +0.85,
     levers +0.41, flat +0.35, skeptic 0.00, EB-heavy -0.33.
   - The read moved every world TOWARD swide. An override to pw20 now can only come from the prior, which existed before the read (and
     LEVERS_1021's MCLmm-EB finding at 11:30 IST also came before the 17:33 read). Choosing it now would be the fork R1 exists to prevent,
     in the direction the evidence does not support.

## 1. The read and its decomposition (MEASURED read, MODEL decomposition; `apick_mc.py`, `apick_base.log`, `evid.log`)

- Read (MEASURED, PREREG A22 READ note; `canary/lb_bands.out`): swide .960 r146 (.960 band r100-150, 4 teams below it), pw25 .959;
  spl / salt / r6pw20 = .916 canary (unobserved for the model). Silver ends r315.
- Conditioning = `canary/bpick_work/bpick_mc.py` sec 1 verbatim (swide > pw25, pw30 public; swide .960; pw25 .959; older entries < .960;
  uniform band ruler t^4 (1 - t)^46). World = `day/skeptic_wc.py` sec 1-2b verbatim. Base run reproduces `bpick_base.log` to the digit
  (A = swide, B = R6: +17.33, .876 / .763, .561 / .446).

| base posterior (e-4) | prior | posterior | share of the public gap |
|---|---|---|---|
| lomm05 population muL | +0.18 | +0.44 | 22 % |
| interaction I | +0.16 | +0.16 | 8 % |
| MCLmm population muM | -0.09 | +0.50 | 24 % |
| lomm05 read noise | 0 | +0.15 | 8 % |
| MCLmm read noise | 0 | +0.78 | 39 % |
| **public swide - pw25** | +0.24 | **+2.03** | 100 % |

- P(muM > 0) .46 -> .59; P(muL + I + muM > 0) .55 -> .73. Public pw25 - pw30 posterior -0.11 (no information on their order).
- Private (scenario mix S0 .6 / C+ .1 / C- .1 / SITE .2): stack +0.24 -> +1.08, P(stack < 0) .46 -> .33; pw25 - pw20 dose +0.44
  (unchanged by the read); P(swide < pw20) .32.
- Population share of the gap in other worlds (`apick_summ.out` sec 1): lskep 38 %, mzero 34 %, nf15 38 %, pf5 30 %, skepall 5 %,
  pure 0 %, MCLmm-neg 22 %, MCLmm-EB -16 % (the population part is negative; noise explains more than the whole gap).
- Read likelihood per stack prior (`evid.log`, relative to base): lomm05 stack x MCLmm {stack 1.21, skep2 1.00, neg 1.02, zero 1.17,
  EB .78}; lomm05 skeptic {1.27, .95, .96, 1.11, .68}; lomm05 OAI-only {1.09, .79, .71, .85, .46}.
- gold58 (info only, CLAIMED `val/ext_a22.log` + LEVERS_1021 table): pw25 .94368, spl .94417, swide .94492. gold58 nominated both
  components, so it is not independent evidence.

## 2. Variant grid (MODEL; `apick_summ.out`; 26 variants: 14 as-built, 12 skeptic-component)

As-built: base, base_norank, lskep, mstack, rhoS0, nf15, pf5, flatU, PAREA, kpess, fieldBad, fieldGood, c4over, c4real.
Skeptic-component (S): mzero, worst, rhoA0, skepall, skepallnf2, pure, base_mneg (MCLmm N(-.5, 1)), base_meb (MCLmm N(-1.5, 1), the
levers/regime.md EB reading), worst_meb, worstA0_meb (+ rho 0), flatU_meb, PAREA_meb.

### 2.1 B = R6 57248084: alt A minus swide (silver step / smooth, r150 step / smooth in pts; E e-4)

| variant | swide pair | pw25 | pw20 | pw30 | eoa20 |
|---|---|---|---|---|---|
| base | .876/.763, .561/.446, +17.33 | -1.3/-1.9, -3.6/-2.6, -1.00 | -0.2/-1.8, -5.4/-4.4, -1.71 | -4.2/-4.0, -5.2/-2.9, -1.08 | -1.6/-3.6, -8.6/-6.4, -2.46 |
| flatU | .859/.743, .528/.423, +16.37 | -1.7/-2.1, -3.8/-2.7, -1.04 | -0.8/-2.4, -6.3/-4.7, -1.78 | -3.9/-3.6, -4.4/-2.4, -0.92 | -2.0/-3.9, -9.0/-6.3, -2.40 |
| PAREA | .886/.773, .572/.451, +17.51 | -1.3/-1.9, -3.8/-2.7, -1.03 | -0.1/-1.8, -5.6/-4.4, -1.71 | -3.9/-3.8, -5.2/-2.8, -1.01 | -0.7/-2.6, -7.1/-5.3, -2.03 |
| lskep | .869/.757, .552/.440, +17.07 | -0.9/-1.3, -2.5/-1.8, -0.72 | +0.2/-1.2, -4.3/-3.6, -1.43 | -3.8/-3.4, -4.2/-2.2, -0.80 | -1.2/-3.0, -7.5/-5.7, -2.18 |
| nf15 | .865/.754, .551/.440, +17.09 | -0.6/-1.2, -2.5/-2.0, -0.78 | +0.6/-1.0, -4.3/-3.8, -1.48 | -3.5/-3.3, -4.1/-2.3, -0.86 | -0.8/-2.8, -7.5/-5.8, -2.22 |
| mzero (S) | .870/.757, .549/.437, +16.97 | -0.8/-1.1, -2.2/-1.6, -0.60 | +0.4/-1.0, -3.9/-3.4, -1.31 | -3.6/-3.3, -3.9/-1.9, -0.69 | -1.1/-2.8, -7.2/-5.4, -2.07 |
| rhoA0 (S) | .861/.746, .534/.428, +16.63 | +0.2/-0.2, -0.9/-0.8, -0.30 | +1.3/-0.1, -2.7/-2.6, -1.01 | -2.7/-2.3, -2.5/-1.1, -0.38 | -0.1/-1.9, -5.9/-4.6, -1.76 |
| pure (S) | .862/.745, .527/.423, +16.43 | +0.2/+0.1, +0.1/-0.0, -0.01 | +1.3/+0.2, -1.7/-1.9, -0.73 | -2.6/-2.0, -1.7/-0.4, -0.10 | -0.1/-1.6, -5.0/-3.9, -1.49 |
| base_meb (S) | .854/.736, .514/.414, +16.10 | +0.7/+0.8, +1.1/+0.6, +0.22 | +1.9/+0.9, -0.7/-1.2, -0.48 | -2.1/-1.4, -0.6/+0.1, +0.11 | +0.5/-0.9, -3.9/-3.2, -1.23 |
| worst_meb (S) | .832/.711, .473/.386, +15.02 | +2.8/+3.2, +5.2/+3.3, +1.26 | +4.0/+3.4, +3.5/+1.5, +0.58 | -0.0/+1.0, +3.4/+2.8, +1.13 | +2.6/+1.6, +0.1/-0.5, -0.18 |
| worstA0_meb (S) | .815/.692, .444/.367, +14.28 | +4.6/+5.2, +8.1/+5.2, +2.00 | +5.7/+5.3, +6.3/+3.4, +1.32 | +1.7/+3.0, +6.2/+4.7, +1.87 | +4.3/+3.5, +3.0/+1.4, +0.56 |

(all 26 rows and B = r6pw20 / mix rows in `apick_summ.out`.)

### 2.2 Counts over 26 variants, B = R6 (rows where swide >= alt; as-built / skeptic)

| alt | silver step | silver smooth | r150 step | r150 smooth | bronze | E |
|---|---|---|---|---|---|---|
| pw25 | 16 (14/14, 2/12) | 17 (14, 3) | 19 (14, 5) | 20 (14, 6) | 16 (14, 2) | 20 (14, 6) |
| pw20 | **9 (9/14, 0/12)** | 17 (14, 3) | 22 (14, 8) | 23 (14, 9) | 4 (4, 0) | 23 (14, 9) |
| pw30 | 24 (14, 10) | 23 (14, 9) | 21 (14, 7) | 20 (14, 6) | 25 (14, 11) | 20 (14, 6) |
| eoa20 | 18 (14, 4) | 23 (14, 9) | 23 (14, 9) | 24 (14, 10) | 16 (14, 2) | 24 (14, 10) |

- Mean (alt - swide) over the 26: pw20 +0.9 step silver / -0.4 smooth / -3.1 r150 step / -1.1e-4 E; pw25 -0.2 / -0.5 / -1.3 / -0.4;
  pw30 -3.0 / -2.6 / -2.9 / -0.5; eoa20 -0.4 / -2.1 / -6.2 / -1.8. B = r6pw20 (clean) gives the same picture (pw20 +0.8 / -0.3 / -2.5).
- Dose-family and field variants (flatU, PAREA, fieldBad / Good, kpess, c4over / real) never change the ranking: all A candidates are
  c4-lineage, so they move together.

## 3. Averaging the stack priors (MODEL; `evid.py`, `bma.py` -> `bma.out`, `bma_abs.py` -> `bma_abs.out`)

15 worlds = lomm05 prior {stack (A22), skeptic N(0, .9), OAI-only N(-.9, .9)} x MCLmm prior {stack (builder), skep2 (A22), neg
N(-.5, 1), zero, EB N(-1.5, 1)}, base dose world, rho 1 (7 new grid runs: `apick_wSK*.json`, `apick_wOAI*.json`). Posterior world weight =
prior weight (GUESS) x read likelihood. Weightings: A22 priors (stack, skep2 only); optimist (.6/.3/.1 x .3/.4/.1/.1/.1); flat; **levers**
(.4/.4/.2 x .1/.35/.2/.15/.2: my central GUESS from LEVERS_1021, where lomm05 central E +0.06 and MCLmm is EB-negative on every ruler);
skeptic (.2/.4/.4 x .05/.2/.25/.2/.3); EB-heavy (.1/.4/.5 x 0/.1/.3/.2/.4).

### 3.1 Pairs with B = R6 57248084 (silver step / smooth | r150 step / smooth | bronze | E (pub-scale))

| weighting (post. stack E) | swide | pw25 | pw20 | pw30 | eoa20 |
|---|---|---|---|---|---|
| A22 priors (+1.08) | **.876 / .763** \| **.561 / .446** \| .924 \| **+17.33** (.95975) | .862 / .744 \| .525 / .421 \| .915 \| +16.33 | .874 / .745 \| .507 / .403 \| .925 \| +15.62 | .834 / .723 \| .509 / .417 \| .894 \| +16.25 | .860 / .727 \| .475 / .382 \| .916 \| +14.87 |
| optimist (+0.85) | .872 / **.760** \| **.555 / .442** \| .921 \| **+17.15** | .862 / .745 \| .527 / .421 \| .915 \| +16.36 | **.874** / .746 \| .509 / .403 \| .925 \| +15.64 | .834 / .724 \| .510 / .418 \| .894 \| +16.28 | .860 / .728 \| .476 / .383 \| .916 \| +14.89 |
| flat (+0.35) | .864 / **.750** \| **.538 / .430** \| .916 \| **+16.69** | .862 / .745 \| .526 / .421 \| .915 \| +16.34 | **.873** / .746 \| .508 / .403 \| .925 \| +15.63 | .834 / .723 \| .509 / .417 \| .894 \| +16.25 | .859 / .728 \| .476 / .383 \| .916 \| +14.88 |
| **levers (+0.41)** | .865 / **.751** \| **.540 / .431** \| .917 \| **+16.74** (.95969) | .861 / .744 \| .526 / .421 \| .915 \| +16.34 (.95965) | **.873** / .746 \| .508 / .403 \| .925 \| +15.63 (.95958) | .833 / .723 \| .509 / .417 \| .893 \| +16.25 (.95964) | .859 / .728 \| .476 / .383 \| .916 \| +14.87 (.95951) |
| skeptic (0.00) | .858 / .742 \| **.526** / .422 \| .912 \| **+16.37** | .861 / .744 \| .526 / .420 \| .915 \| +16.33 | **.873 / .745** \| .508 / .403 \| .925 \| +15.63 | .833 / .723 \| .509 / .416 \| .893 \| +16.23 | .859 / .728 \| .476 / .382 \| .916 \| +14.87 |
| EB-heavy (-0.33) | .853 / .736 \| .514 / .414 \| .909 \| +16.08 | .861 / .744 \| **.526 / .420** \| .915 \| **+16.32** | **.873 / .745** \| .508 / .403 \| .925 \| +15.63 | .833 / .722 \| .508 / .416 \| .893 \| +16.22 | .859 / .727 \| .476 / .382 \| .916 \| +14.87 |

### 3.2 Pairs with B = a clean r6pw20 resubmit (same columns)

| weighting | swide | pw25 | pw20 | pw30 | eoa20 |
|---|---|---|---|---|---|
| A22 priors | .882 / **.776** \| **.594 / .464** \| .923 \| **+17.86** (.95981) | .871 / .759 \| .563 / .441 \| .915 \| +16.95 | **.883** / .763 \| .552 / .427 \| .926 \| +16.44 | .849 / .743 \| .550 / .439 \| .898 \| +16.91 | .874 / .751 \| .529 / .413 \| .920 \| +15.88 |
| levers | .875 / **.767** \| **.578 / .452** \| .918 \| **+17.36** (.95976) | .871 / .760 \| .565 / .441 \| .915 \| +16.95 | **.884** / .764 \| .553 / .428 \| .927 \| +16.45 (.95967) | .849 / .743 \| .551 / .439 \| .898 \| +16.91 | .874 / .751 \| .530 / .413 \| .920 \| +15.88 |
| skeptic | .871 / .760 \| **.567 / .444** \| .915 \| **+17.06** | .871 / .760 \| .564 / .441 \| .915 \| +16.95 | **.884 / .764** \| .553 / .428 \| .927 \| +16.45 | .849 / .743 \| .551 / .439 \| .898 \| +16.90 | .874 / .751 \| .530 / .413 \| .920 \| +15.88 |
| EB-heavy | .867 / .755 \| .558 / .438 \| .912 \| +16.82 | .871 / .760 \| **.564 / .441** \| .915 \| **+16.95** | **.884 / .764** \| .553 / .428 \| .927 \| +16.45 | .849 / .743 \| .550 / .439 \| .898 \| +16.90 | .874 / .751 \| .530 / .413 \| .920 \| +15.88 |

- P(clean) only scales the B effect: at P(clean) .55 every cell sits 55 % of the way from 3.1 to 3.2 (`bma.out` mix0.55 rows). The
  A ranking is the same at both ends, so the A decision does not depend on the resubmit.
- P(alt private > swide private), levers: pw25 .44, pw20 .39, pw30 .38, eoa20 .30 (A22 priors .33 / .32 / .31 / .24; EB-heavy .56 / .48 /
  .47 / .38). MEASURED from `apick_*.json`.
- A alone (no B), levers: swide .829 / .723 step / smooth silver, .529 r150, +15.90; pw20 .845 / .724, .498, +15.05; pw25 .825 / .716,
  .515, +15.49. The B hedge (R6) covers the E-OA dose tail, which is why pw20's silver edge is small inside a pair.

## 4. Break-even: how much of swide's posterior stack must be real (MODEL; `apick_sweep.py`, `apick_sweep_*.log`)

Shift swide's private value down by x (x = the part of the +1.08 posterior stack that is not real), base dose world:

| swide still >= | silver step | silver smooth | r150 step | r150 smooth | E |
|---|---|---|---|---|---|
| pw25 (B = R6 / r6pw20) | x .83 / .98 | .97 / 1.11 | 1.09 / 1.21 | 1.14 / 1.26 | 1.14 / 1.26 |
| pw20 | x .12 / .00 | .93 / .87 | 1.64 / 1.66 | 1.95 / 2.04 | 1.97 / 2.00 |
| pw30 | x 2.44 / 2.80 | 2.02 / 2.23 | 1.59 / 1.72 | 1.30 / 1.35 | 1.24 / 1.31 |
| eoa20 | x .98 / .72 | 1.81 / 1.71 | 2.61 / 2.59 | 2.88 / 2.94 | 2.87 / 2.86 |

- In words: swide beats pw25 as long as the true private stack effect is >= roughly zero (-0.2 to +0.25e-4). It beats pw20 on smooth
  silver if the stack is >= ~+0.15 to +0.2, on r150 and E if it is >= ~-0.6 to -0.9. Step silver vs pw20 is a tie even at the full
  posterior: the step curve rewards pw20's narrower dose tail, so the stack must beat its posterior mean to win there.
- Similar (break-evens within about +-0.4) in flatU, PAREA, fieldBad and kpess (`apick_sweep_{flatU,PAREA,fieldBad,kpess}.log`).

## 5. Reading the result

- **The A question reduces to one number: the private value of the swide extras (lomm05 + MCLmm).** E-OA dose is shared (pw25 dose for
  swide and pw25), the our-side base is shared (c4 for all five), and B covers the dose tail.
- **The evidence on that number is weak in both directions.** For: gold58 nomination (MCLmm 9-11 MCL+ knees, P .015), lomm05 positive in
  every population-consistent regime and on OAI raw (CLAIMED levers/regime.md), and this read (Bayes factor ~1.0-1.6). Against: MCLmm
  EB-negative on every report ruler (-16 to -48 label), lomm05 R7-negative under EB and under a gold-like case mix (CLAIMED
  LEVERS_1021 l.36, regime.md sec 5). My central weighting (levers) puts the private stack at +0.41e-4, P(stack world negative) .20.
- **At that weighting the medal question is a tie** (pw20 +0.8 step / -0.5 smooth silver) **and swide wins the rest** (+3.1 / +2.8 r150,
  +1.1e-4 E). The ties are far below the family shock (about +-10 silver pts, CLAIMED shakeup.md) and below the model's own resolution.
- **Process:** R1 and H2 both name swide. The A22 PF clause allows pw20 "before selection", but the case for it is prior-only, the
  prior was on the table before the read, and the read itself leans the other way. If Eesh wants the medal-only override, it must be
  written down before he opens the selection page, with the stance in item 4 of sec 0 stated.

## 6. Caveats

- One dose family (dose_mc, OAI-proxy D5 curves fitted to the public reads, aggregates only, CLAIMED dose.md). Its pw20-vs-pw25 dose tail
  is what gives pw20 its step-silver edge; dose.md's skeptic found the step curve inflates it (the smooth curve halves it).
- Stack read-noise sds (lomm05 1.00 / .62, MCLmm 2.40 / 1.50 public / private) and the -.2 lomm05-pw25 read correlation are inherited
  (CLAIMED stack/skeptic/skep_value_mc.py); nf15 / nf2 rows stress them and change nothing.
- World weights are GUESS; every weighting is shown so they can be swapped. The MCLmm EB prior N(-1.5, 1) is my macro translation of
  regime.md's label-scale EB numbers with a haircut (GUESS).
- Field curves date from 10-19 / 10-20 (CLAIMED). Our public position (r146, 169 places inside silver, MEASURED lb_bands.out) is not an
  input; the margin model is.
- rho (public-to-private carry-over of the stack effect) is 1 in the grid; rhoS0 / rhoA0 rows (sec 2) show the cost: rhoA0 makes step
  silver favour pw20 by +1.3 but keeps swide ahead on smooth silver, r150 and E.
- Not modelled: host rulings, teams copying E-OA, an S6fall cliff, ERROR tails. Picks are fixed scores, so swide's own rerun risk is nil
  (CLAIMED HANDOFF 17:40 RISK NOTE).

## 7. Files (`session_2026-10-21/lastday/`)

- `apick_mc.py`: world (skeptic_wc sec 1-2b exec'd) + bpick read conditioning + A x B pair metrics, decomposition, pairwise P, MCLmm
  override knob `malt=neg|eb`. Outputs `apick_<tag>.log/.json` (33 runs). Runners `run_apick.sh` (`run_apick.log`), `run_grid.sh`
  (`run_grid.log`).
- `apick_summ.py` -> `apick_summ.out` (26-variant tables, counts, best-A per row).
- `evid.py` -> `evid.log/.json` (read likelihood per stack prior). `bma.py` -> `bma.out`; `bma_abs.py` -> `bma_abs.out`.
- `apick_sweep.py` -> `apick_sweep_<tag>.log/.json` (break-even), runner `run_sweep.sh` (`run_sweep.log`).
