# picks2.md: B for the final pair now that A = pw30 (10-20 ~19:00-20:00 IST, val Track D)

RESEARCH + OFFLINE VALIDATION ONLY. Nothing was submitted, pushed, committed, versioned or downloaded; no dataset, no GPU job, no
scheduled task. PREREG, HANDOFF, PROGRESS, a20, a21 and push were only read. **No box and no OAI data in this track**: the E-OA
posterior reuses box aggregates already on disk (push/work/box_abc2_*.log). Local CPU only. Everything I wrote is under
`val/picks2_work/` plus this file. Two LOCAL builds (not pushed) under `val/picks2_work/build/` (sec 6).

**Tags.** MEASURED (computed here or read from a named file), CLAIMED (older / sibling note says so), MODEL (arithmetic on stated
inputs), GUESS (judgement). gold58 is info only.
**Units.** "Margin" = private macro AUC over the heliosli copy block, x1e-4 (the picks.md / bpick.md ruler). "pts" = percentage
points. "Pub-scale" = .95802 + margin (private score on the public scale, no common public-to-private shift).
P(r<=150) = P(private rank <= 150) under the 10-19 field model (silver r316, step at margin ~5e-4; r150 at ~15e-4).
**Names.** pw30 = 57425970 (A, API best-id). pw20 = 57425269 (scored, .959). eoa20 = 57413368. c4 = 57343435. R6 = 57248084.
R6e20 = R6 + sym .2 block (push/, built + gated). R6pw30 = R6 + pw .3 block (asked as "unbuilt"; now built here, byte-identical to
Track C's val/kernels build). R6pw20 = R6 + pw .2 block (NOT in the asked list; built + gated here because it scores best).

## 0. Answer

1. **Among the five asked B's, B = R6e20.** It is the best or within 0.4 pts of the best on every metric in 33 sensitivity rows
   (asked-only mean regret: silver .23 pts, r150 .23 pts, E .05e-4). The others:
   - R6pw30: -1.8 pts silver vs R6e20 in 33/33 rows (it carries the same a .3 dose risk as A); r150 ties (+0.07, 20/33). Do not use.
   - eoa20: -0.4 silver (8/33 above), -1.8 r150 (4/33 above) vs R6e20. R6: -1.9 silver, -3.9 r150 (1/33 and 0/33 above). c4: -2.5
     silver, -4.3 r150. **The A15 B (R6) is now dominated.**
2. **Best overall (my recommendation): enter R6pw20 in ONE 10-21 slot, B = R6pw20, fallback pw20 (not R6).**
   - R6pw20 beats R6e20 in 33/33 rows on all three metrics, by a little: +0.5 silver, +0.8 r150, +0.19e-4 E (mean). R6e20 is an
     acceptable substitute if Eesh prefers the push/ build and its drafted amendment.
   - Rule, fixed before its score: B = R6pw20 if COMPLETE and it displays >= .958; otherwise B = pw20 57425269.
3. **Base numbers, A = pw30 (MODEL, `picks2_work/picks2_base.log`):**

   | B | E[max] e-4 | pub-scale | P(silver) | P(bronze+) | P(r<=200) | P(r<=167) | **P(r<=150)** | P(r<=130) | P(B>A) | slot |
   |---|---|---|---|---|---|---|---|---|---|---|
   | (A alone) | 15.21 | .95954 | .811 | .890 | .653 | .567 | .502 | .397 | | |
   | R6 (A15 B) | 16.03 | .95962 | .851 | .930 | .680 | .584 | .514 | .404 | .14 | none |
   | R6pw30 | 16.76 | .95970 | .853 | .920 | .707 | .621 | .556 | .446 | .30 | 1 |
   | **R6e20 (best asked)** | **16.80** | .95970 | **.871** | .937 | .717 | .625 | **.554** | .438 | .29 | 1 |
   | eoa20 | 16.41 | .95966 | .866 | .937 | .701 | .605 | .533 | .417 | .42 | none |
   | c4 | 15.93 | .95961 | .846 | .927 | .673 | .579 | .510 | .401 | .15 | none |
   | pw20 (free) | 16.76 | .95970 | .879 | .945 | .719 | .621 | .548 | .428 | .51 | none |
   | **R6pw20 (rec.)** | **17.02** | **.95972** | **.877** | .941 | .726 | .634 | **.564** | .446 | .32 | 1 |

   - Gold is 0 everywhere. The expected private score of the pair is ~.9597 pub-scale for every sensible B; the differences are
     in the tails.
4. **Does the entry deserve a 10-21 slot? Yes, one, at 05:31 IST, but know what it buys.**
   - Versus the A15 B (R6): +2.4 silver (32/33 rows), +4.7 r150 (33/33), +0.9e-4 E (33/33). Clear.
   - Versus the free option B = pw20 (already scored, no slot): **silver -0.2 (mean; 11/33 rows positive, range -2.7 to +3.2),
     r150 +1.3 (28/33), E +0.23e-4 (28/33)**. If A later moves to pw25: silver +1.1 (27/30), r150 +2.8 (30/30).
   - So the slot buys upside (rank <= ~150) and the c4-base hedge, not silver. Slots are not scarce (5 on 10-21, 5 on 10-22) and
     the c4 shape probes on Track C's slate are worth ~nothing as A (sec 4.3), so the slot is cheap.
   - **If no slot is spent (or R6pw20 fails), B = pw20, not R6**: pw20 beats R6 in 33/33 rows (+2.7 silver, +3.3 r150, +0.7e-4 E).
5. **A probes for 10-21 (MODEL, sec 1.3 / 4.3):** pw25 is the only one with positive value (P(new best-id) .55 [.45, .63],
   P(.960) .22 [.14, .39]; pair with R6pw20 +1.3 silver, +1.1 r150). pw35 / pw40 have P(new best-id) .11 / .03 and a slightly
   negative private effect through R1 (-0.1e-4 each): skip them. A sibling refit agrees (CLAIMED val/implied2.md: pw35 P(> pw30)
   .12, private -1.9e-4; pw40 .03, -5.7e-4; pw25 .56, +0.6e-4).
6. **INFO only, not a recommendation (sec 8):** the R1 rule makes the noisiest high-dose entry A. The pair {pw20, R6pw20} beats
   {pw30, R6pw20} on silver by +2.3 to +4.0 pts in 9/9 rows, at -0.2 to -1.0e-4 E (8 of 9 rows) and r150 within +-2 pts. That is
   Eesh's call on the A15 rule.

## 1. New evidence

### 1.1 A21 board (MEASURED, `picks2_work/a21_frozen.py` -> `a21_frozen.log`; snapshots lbprobe/api_lb_1020_eoaread2.json 05:53 UTC
and api_lb_1020_a21read.json 13:20 UTC)
- pw30 is our best at r140, .959. Bands at 13:20: .960 r92-136 (45 teams), .959 r137-208 (72), so 3 teams above pw30 inside .959.
- 5,957 teams kept the same best submission between the two boards, and their relative order is identical (sanity check).
- **Frozen ruler:** the 4 .959 teams that sat above eoa20 at 05:53 (r131-134) are all frozen. pw30 is below the top 2 (now r137,
  r139) and above the other 2 (now r141, r143). The third team above pw30 moved in from .958 (t1 r204) and says nothing.
- So pub(pw30) - pub(eoa20) lies between the 2nd and 3rd of 4 frozen scores in (eoa20, .960). pw30 did not reach .960
  (pre-read forecast P(.960) was .57, CLAIMED push/implied.md).

### 1.2 ABC update (MODEL, `picks2_work/abc_a21.py` -> `abc_a21.log`, `abc_a21.json`; `abc_a21_allruler.log`)
- Prior = the push/work ABC (OAI-free D5 curve family reweighted to RSNA coupling, public n 390, kernel on the eoa10 / eoa20
  in-band reads; box aggregates only). Added likelihood: t = (pub pw30 - pub eoa20) / gap960 with 4 uniform frozen teams in the
  gap -> 6 t^2 (1 - t)^2 on (0, 1); pw30 argmax of {eoa20, eoa30, pw20, pw30}; eoa30 and pw20 at .959 (CLAIMED from the brief).
- Read: **pub pw30 - eoa20 = +1.1e-4 [+0.3, +2.2]** (pooled ruler, median +0.9) / **+0.6e-4 [+0.2, +1.0]** (ALL-teams ruler).
  Sibling refit: +0.5 [0.2, 1.2] LB-only, +1.0 [0.3, 1.9] joint (CLAIMED val/implied2.md). Consistent.
- Amplitude s .90 -> .84; every population gain vs c4 drops ~0.8-0.9e-4.

| variant (8 + ALL ruler) | pub pw30 - eoa20 | pop vs c4: eoa20 / pw20 / pw30 | pop pw20 - pw30 (P priv pw20 > pw30) | pw25 P(new) / P(.960) / R1 gain | pw35 | pw40 |
|---|---|---|---|---|---|---|
| **PFT N01 logN (primary)** | +1.08 | +8.9 / +9.8 / +9.6 | +0.14 (.53) | .55 / .22 / +0.43 | .11 / .03 / -0.12 | .03 / .01 / -0.08 |
| PFT N01 logU | +1.21 | +8.0 / +8.8 / +8.8 | +0.02 (.50) | .49 / .19 / +0.32 | .19 / .06 / -0.16 | .08 / .03 / -0.18 |
| PFT flat logN | +1.11 | +9.9 / +10.8 / +11.4 | -0.55 (.45) | .53 / .20 / +0.21 | .13 / .05 / -0.08 | .04 / .01 / -0.04 |
| PFT flat logU | +1.25 | +8.9 / +9.7 / +10.4 | -0.78 (.41) | .47 / .16 / +0.06 | .21 / .07 / -0.11 | .08 / .03 / -0.14 |
| PAREA N01 logN | +1.17 | +10.2 / +10.6 / +11.0 | -0.44 (.45) | .49 / .15 / +0.20 | .06 / .01 / -0.07 | .01 / .00 / -0.02 |
| PAREA flat logU | +1.33 | +9.7 / +10.1 / +11.0 | -0.89 (.39) | .45 / .14 / +0.07 | .13 / .04 / -0.06 | .06 / .02 / -0.11 |
| noise x1.25 | +1.12 | +8.8 / +9.7 / +9.3 | +0.31 (.54) | .52 / .22 / +0.37 | .13 / .06 / -0.19 | .04 / .03 / -0.15 |
| rho_c4 .78 | +1.10 | +8.3 / +9.2 / +8.7 | +0.53 (.57) | .55 / .20 / +0.52 | .11 / .03 / -0.15 | .02 / .01 / -0.05 |
| ALL-teams ruler | +0.58 | +9.7 / +10.7 / +10.7 | -0.06 (.48) | .63 / .39 / +0.41 | .09 / .05 / -0.12 | .02 / .01 / -0.07 |

(e-4; "R1 gain" = E[population gain over pw30 if R1 picks the probe], i.e. what the probe adds to A privately.)
- **What matters for B:** pw30's population gain is uncertain (sd 7.4e-4 vs c4, p10 ~0) and its private E-OA noise is larger
  (sd 5.7e-4 at a .3 vs 3.5e-4 at a .2: box C x sqrt(390/910)). The read shrank the mean a little; it did not shrink this spread.
  So the low tail of A is a dose tail, and the best B covers it with a lower dose.

### 1.3 Transfer to R6 (MEASURED gold58, n 58, info; `picks2_work/r6transfer_pw.log`)

| E-OA on | c4 delta | R6 delta | ratio | boot corr | P(R6 delta > 0) |
|---|---|---|---|---|---|
| sym .2 | +.00247 | +.00218 | .88 | .74 | .96 |
| pw .2 | +.00265 | +.00215 | .81 | .80 | .96 |
| pw .3 | +.00223 | +.00191 | .86 | .86 | .85 |
| pw .35 / .4 | +.00256 / +.00190 | +.00141 / +.00136 | .55 / .72 | .88 | .69 / .66 |

- Post-hoc transforms reproduce the in-cell A20 block to 2.8e-8. On gold58 R6pw30 - R6e20 = -.00027 (sd .00104) and
  R6pw20 - R6e20 = -.00003: no gold58 preference between the three R6 versions.
- Model input: one common k ~ N(.86, .15) for all three R6 versions (+ N(0, .05) each); skeptic rows N(.6, .3), N(.4, .4).

## 2. Model (`picks2_work/picks2_mc.py`; MODEL unless tagged)
- **Our side:** the 10-19 think/picks posterior (200k draws; c4 private margin +5.6e-4 sd 6.9, R6 +2.7 sd 7.5, corr .54; CLAIMED),
  unchanged. Paired at random with the E-OA draws (independence, MODEL).
- **E-OA:** 200k resampled ABC posterior draws (sec 1.2). Entry X = c4 + Pop_X + private noise, noise = MVN(0, C x 390/910) jointly
  over all shapes (C = box n-390 covariance at the gold74 regime).
- **R6 versions:** R6X = R6 + k Pop_X + eps (1e-4) + private noise with corr .8 to c4's (unshared part from an independent draw).
  Public display of R6X from R6's place at u .10 of .958 (CLAIMED density.md): P(.957 / .958 / .959) = .005 / .50 / .50 (R6e20),
  .006 / .48 / .51 (R6pw20), .029 / .41 / .56 (R6pw30).
- **A:** pw30. Also R1 winners if 10-21 adds pw25 / pw35 / pw40 (argmax of public draws).
- **Field:** the 10-19 field model, unchanged, re-run with r140 / r150 thresholds (`picks2_work/field_curves150.py`).
- **Alternative E-OA family** ('mix' rows): push/bpick_mc.py priors and read model, plus the same A21 likelihood. No pw25-40 there.

## 3. Results by A outcome (base; P(silver) / P(r<=150) / E e-4)

| A | P(A) | A alone | R6 | R6pw30 | R6e20 | eoa20 | c4 | pw20 | R6pw20 |
|---|---|---|---|---|---|---|---|---|---|
| pw30 (now) | | .811 / .502 / 15.21 | .851 / .514 / 16.03 | .853 / .556 / 16.76 | .871 / .554 / 16.80 | .866 / .533 / 16.41 | .846 / .510 / 15.93 | .879 / .548 / 16.76 | **.877 / .564 / 17.02** |
| R1{pw30, pw25} | pw25 wins .55 | .838 / .519 / 15.64 | .869 / .529 / 16.25 | .871 / .570 / 17.06 | .885 / .566 / 16.96 | .869 / .535 / 16.30 | .861 / .524 / 16.10 | .879 / .546 / 16.56 | **.890 / .575 / 17.16** |
| R1{pw30, pw35} | pw35 wins .11 | .805 / .498 / 15.09 | .847 / .511 / 15.97 | .850 / .553 / 16.70 | .868 / .552 / 16.75 | .865 / .532 / 16.41 | .843 / .507 / 15.88 | .879 / .548 / 16.77 | **.875 / .561 / 16.98** |
| R1{pw30, pw35, pw40} | .10 / .01 | .804 / .498 / 15.06 | .847 / .510 / 15.96 | .849 / .552 / 16.69 | .868 / .551 / 16.74 | .865 / .532 / 16.41 | .843 / .507 / 15.87 | .879 / .548 / 16.78 | **.874 / .561 / 16.97** |

- The B ranking does not change if A moves to pw35 / pw40: R6pw20 (or R6e20) stays best on r150 and E; pw20 stays best on silver
  by <= 0.5. If A moves to pw25, R6pw20 is best on all three.

## 4. Sensitivity (33 rows; `picks2_work/run_sens2.sh` -> `sens2.log`, `summ_picks2.py` -> `summ_picks2.out`)
Rows: 8 ABC variants + ALL-teams ruler; 'mix' E-OA family (and mix + k_pess, mix + r6better); shape spread 1e-4 / 2e-4 (GUESS
structural noise between shapes); k N(.6, .3), N(.4, .4), N(1, .1); rho .6 / .9; private E-OA noise x1.5 / x0.7; our side
skepS6 / strongS6 / tabReal / tabOver / r6better / noise1.5 / bigKnobs / smallKnobs; field fieldBad / fieldGood / Wbig; k_pess +
noise x1.5.

### 4.1 Regret vs the row's best (A = pw30; pts, E in e-4; mean / max over 33 rows)

| B | silver | r150 | E | rows best (silver / r150 / E) |
|---|---|---|---|---|
| R6 | 2.96 / 4.46 | 4.88 / 6.49 | 1.00 / 1.55 | 0 / 0 / 0 |
| R6pw30 | 2.83 / 4.17 | 0.93 / 2.32 | 0.31 / 0.72 | 0 / 2 / 1 |
| R6e20 | 1.04 / 2.95 | 1.00 / 2.01 | 0.26 / 0.58 | 0 / 0 / 0 |
| eoa20 | 1.43 / 4.44 | 2.85 / 6.32 | 0.60 / 1.77 | 0 / 0 / 0 |
| c4 | 3.53 / 6.19 | 5.30 / 8.13 | 1.12 / 2.25 | 0 / 0 / 0 |
| pw20 | 0.30 / 3.24 | 1.56 / 5.09 | 0.30 / 1.42 | 8 / 1 / 2 |
| R6pw20 | 0.54 / 2.74 | **0.23 / 1.96** | **0.07 / 0.51** | 5 / 24 / 19 |
| R6pw20, but B = pw20 unless R6pw20 shows .959 | **0.13 / 1.40** | 0.41 / 2.06 | 0.06 / 0.57 | 19 / 5 / 10 |
| R6e20, same .959 rule | 0.37 / 1.76 | 0.87 / 2.68 | 0.17 / 0.75 | 1 / 0 / 0 |

- Asked five only (silver / r150 / E mean regret): R6 2.14 / 4.11 / .79; R6pw30 2.01 / .16 / .10; **R6e20 .23 / .23 / .05**;
  eoa20 .62 / 2.08 / .39; c4 2.71 / 4.53 / .91.
- With A = R1{pw30, pw25} (30 rows): R6pw20 regret .09 / .06 / .02 (best in 26 / 24 / 23 rows); R6e20 .53 / .83 / .21;
  pw20 1.19 / 2.82 / .62.
- **Where pw20 beats R6pw20 on silver** (by 1.4-2.7 pts): k skeptic / pessimistic (N(.6, .3), N(.4, .4), with or without noise
  x1.5), mix + k_pess, our side tabReal. That is, when the E-OA gain does not carry over to R6, or c4's per-finding table is genuinely better than R6's
  uniform W. **Where R6pw20 wins big**: our side noise1.5 (+3.2 silver), tabOver (+1.3).
- **The .959-conditioned rule** (B = R6pw20 only if it displays .959, else pw20) trades +0.4 silver for -0.2 r150 when A stays
  pw30, and loses -0.35 silver / -0.9 r150 if A moves to pw25. Too small to justify a two-branch rule; I keep the simple rule.

### 4.2 Robustness counts (A = pw30, 33 rows)
- R6pw20 > R6e20: 33/33 on silver (+0.50, range +0.00 to +0.76), r150 (+0.77), E (+0.19). The edge comes from the ABC family's
  structural pw20 > sym20 population gain (+0.8e-4); in the mix family it is ~0 (+0.06 silver, +0.12 r150). gold58 has no
  preference (sec 1.3). It is never negative.
- R6e20 > R6pw30 on silver: 33/33 (+1.8, range +1.0 to +2.9). On r150 they tie (-0.07, 13/33).
- R6e20 > R6 on r150 / E: 33/33. On silver 32/33 (one row -0.46: k_pess + noise x1.5).

### 4.3 What a 10-21 A probe adds (base, MODEL)
- pw25 (built: push/shape/kernel_v224r6c4pw25j, CLAIMED push/shape gate log): A alone +2.7 silver, +1.7 r150. With B = R6pw20:
  +1.3 silver, +1.1 r150, +0.14e-4 E.
- pw35: with B = R6pw20, -0.2 silver, -0.3 r150, -0.04e-4 E. pw35 + pw40: -0.3 / -0.3 / -0.05. Negative because when they win
  publicly it is mostly public noise (R1 winner's curse), and their population mean is lower.
- So slot priority on 10-21: R6pw20 (B) first, pw25 (A) second; pw35 / pw40 / pw45 (Track C slate) are not worth slots under this
  model. gold58 (info) prefers pw35 over pw30 on c4 (+.00256 vs +.00223, sd ~.0018); I do not weight that.

## 5. Why R6pw30 loses and pw20 is strong
- A = pw30 fails (silver) mostly in the E-OA dose tail: low coupling gap or ceiling effects make a .3 overshoot, and its private
  noise is the largest. R6pw30 shares that tail (same shape, same dose); it only hedges the c4-vs-R6 base, so it adds upside
  (r150) but little silver.
- pw20 and R6pw20 sit at a lower dose (the posterior-mean population optimum is pw25 in 8 of 9 variants and pw30 by .06e-4 in
  PAREA flat logU, MODEL; sibling: PW20 .32, PW25 .30, PW30 .23 of the argmax mass, CLAIMED val/implied2.md), with less private
  noise. R6pw20 adds the base hedge on top.

## 6. Local builds and offline gate (MEASURED; NOT pushed)
- `picks2_work/build/build_r6pw.py` = build_a21.py's pw block (verbatim, 14 lines) on the R6 kernel at the same cell-18 anchor.
  - **kernel_v224r6flips6fpw20j** (R6pw20): sha e4d92c0a. Kernel id eeshsaxena/rsna-sub-v224r6flips6fpw20j.
  - kernel_v224r6flips6fpw30j (R6pw30): sha 27ce4be7, notebook AND kernel-metadata byte-identical to Track C's val/kernels build.
  - kernel_v224r6flips6fpw00j: a .0 control, sha 09e307d2 (same as Track C's control).
  - XREF: each differs from the scored / built c4 pw kernel at the same a (a21/kernel_v224r6c4pw20j = scored pw20 57425269;
    pw30j = scored pw30) only in cell 14, exactly as R6 differs from c4; cell 18 is byte-identical. So both code paths have run on
    the hidden set: runtime risk is about nil.
- **`picks2_work/build/gate_r6pw_offline.log`: R6PW_OFFLINE_GATE PASS**
  - G-a: the R6pw00 output is byte-identical to R6's.
  - G-b: gold58 delta vs R6 +.00215 (R6pw20) / +.00191 (R6pw30), equal to the post-hoc targets; a_oa, oa_mode pw, mode S6F.
  - G-c: non-OA columns identical to R6. G-d: OA columns equal an independent pw transform of R6's output.
- **Commit gate after a push (NOT run; needs the committed output):**
  `python session_2026-10-20/a21/probe_check_eoa21.py <out_dir> --a224 0.35 --table '{"Medial OA": 0.5, "Lateral OA": 0.5, "PF OA": 0.5}' --aoa 0.2 --mode pw`
  (Track C's r6pw30 slate entry uses the same args with --aoa 0.3 and passed its simulated-output selftest, CLAIMED
  val/builds.md). It must PASS 17/17 before the submit. R6e20, if used instead, needs the A20 gate (a20/probe_check_eoa.py,
  --aoa 0.2): the a21 gate fails it on oa_mode (CLAIMED val/builds.md).

## 7. Slot plan and draft amendment (for Eesh; nothing done)
- **Timing.** 10-21 opens 05:30 IST (00:00 UTC) with 5 slots. Push R6pw20 tonight on Eesh's go (~20 min commit), run the commit
  gate, submit 05:31 IST. Scores ~11:00-11:30 IST 10-21. A 10-22 05:31 submit still completes before selection but has no retry.
- **Draft PREREG Amendment (text only; NOT written into PREREG):**
  - "B-hedge entry R6pw20 = rsna-sub-v224r6flips6fpw20j = R6 57248084 + the A21 pw block at a .2. Its cell 18 is byte-identical to
    scored pw20 57425269; only cell 14 differs, as R6 vs c4.
  - Offline gate PASS (val/picks2_work/build/gate_r6pw_offline.log). Commit gate probe_check_eoa21.py (R6 params, --aoa .2
    --mode pw) must PASS 17/17. One submit.
  - Selection rule, fixed before its score: A = API best-id (A15 / R1). B = R6pw20 if COMPLETE and displays >= .958; otherwise
    B = pw20 57425269. This replaces A15's B = R6 57248084.
  - R6pw20 cannot become A: R6 sits ~.9581 publicly and the expected gain is ~+8.4e-4, so R6pw20 lands ~.9589, ~1e-3 below
    pw30 (.9599) (MODEL, implied2.md unrounded scores)."
- If Eesh prefers R6e20 (push/ build and bpick.md draft Amendment 22): same rule with R6e20, and fallback pw20 instead of R6.
- **Eesh selects A and B by hand** before 10-22 23:59 UTC. Auto-selection would pick two c4 E-OA near-twins.

## 8. INFO: the A rule (not a recommendation; A15 / R1 is pre-registered)
`picks2_work/run_info.sh` -> `info.log` (A = pw20 or eoa20 instead of the best-id, MODEL):

| row | {pw30, R6pw20} silver / r150 / E | {pw20, R6pw20} | {pw30, R6e20} | {pw20, R6e20} | {pw30, pw20} |
|---|---|---|---|---|---|
| base | .877 / .564 / 17.02 | **.910** / .569 / 16.71 | .871 / .554 / 16.80 | .906 / .561 / 16.51 | .879 / .548 / 16.76 |
| mix family | .860 / .509 / 15.36 | .898 / .527 / 15.66 | .859 / .508 / 15.33 | .898 / .526 / 15.63 | .863 / .494 / 15.14 |
| ALL ruler | .898 / .603 / 18.05 | .927 / .610 / 17.60 | .892 / .593 / 17.82 | .924 / .601 / 17.39 | .900 / .588 / 17.77 |
| PAREA | .908 / .611 / 18.17 | .932 / .608 / 17.51 | .907 / .607 / 18.08 | .931 / .604 / 17.41 | .907 / .593 / 17.82 |
| PFT flat logU | .882 / .587 / 17.63 | .905 / .567 / 16.65 | .877 / .580 / 17.46 | .902 / .560 / 16.48 | .877 / .565 / 17.22 |
| k_pess | .857 / .535 / 16.36 | .897 / .540 / 16.09 | .854 / .531 / 16.27 | .895 / .537 / 16.01 | .879 / .548 / 16.76 |
| our skepS6 | .812 / .458 / 14.26 | .846 / .446 / 13.94 | .804 / .449 / 14.03 | .841 / .436 / 13.73 | .807 / .439 / 13.92 |

- Swapping pw30 for pw20 in the pair gains +2.3 to +4.0 pts silver in 9/9 rows (shape2 and noise0.7 rows are in info.log), costs
  0.2-1.0e-4 E in 8/9 (mix +0.3), and moves r150 by -2.0 to +1.8. pw30 alone has the lowest P(silver) of the E-OA entries (.811
  vs pw20 .873, eoa20 .854) despite a near-equal mean (15.2 vs 15.4 / 14.5): its dose tail and private noise are the widest.
- This rests on the ABC curve family's dose-variance structure (OAI proxy) and on the field model's sharp silver step. R1 exists
  to avoid exactly this kind of model-driven override. Flagged for Eesh; I did not change the A in any recommendation.

## 9. Caveats
- **The E-OA family is an OAI proxy** (D5 curve, box noise at n 390, gold74-regime covariance). The R6pw20 > R6e20 edge and the
  dose-tail story both come from it. The mix family (bpick priors) keeps the ranking R6pw20 >= R6e20 > R6pw30 on silver, with
  pw20 / eoa20 relatively stronger.
- **Frozen-ruler conversion** assumes the 4 frozen teams are uniform in (eoa20, .960). The pooled-ruler posterior pushes the eoa20
  gap to ~2.0e-4 (from 1.2) to fit a higher prior increment; the ALL-teams ruler keeps it at 1.1e-4. Both rows are in the grid and
  agree on the B ranking.
- **Transfer k** comes from gold58 (n 58) plus structure; there is no hidden-label check until R6pw20 scores. Its display gives a
  coarse one: .959 means k x gain >~ 9e-4.
- **Absolute levels** rest on the 10-19 field model (silver x* 5e-4, r150 at ~15e-4), not refit since 10-19. Pair differences are
  more robust than levels. Expected private differences between the top B's are <= 0.3e-4, far below any private noise; the
  choice is about tails.
- Our-side and E-OA draws are paired independently. Not modelled: the S6fall cliff beyond the knob prior, host rulings, teams
  copying E-OA.

## 10. Files (`session_2026-10-20/val/picks2_work/`)
- LB: `a21_frozen.py` / `.log`.
- E-OA posterior: `abc_a21.py` -> `abc_a21.log`, `abc_a21.json`, `abc_a21_draws_<variant>.npz`; `abc_a21_allruler.py` / `.log`.
- Transfer: `r6transfer_pw.py` / `.log`.
- Field: `field_curves150.py` -> `curves_{base,fieldBad,fieldGood,Wbig}.npz`, `field_curves150.log`.
- Pick model: `picks2_mc.py` (args: abc=, family=abc|mix, our=, field=, k_mu=, k_sd=, rho=, noise=, shape_sd=, info_a=1, tag=);
  `picks2_base.log`, `picks2_<tag>.json`; `run_sens2.sh` -> `sens2.log`; `summ_picks2.py` -> `summ_picks2.out` (`summ_picks2_v1.out`
  = first grid without the policy columns, same numbers); `run_info.sh` -> `info.log`.
- Build (local only): `build/build_r6pw.py`, `build/kernel_v224r6flips6fpw{00,20,30}j/`, `build/build_receipt_*.json`,
  `build/build_r6pw.log`, `build/gate_r6pw_offline.py` / `.log`.
