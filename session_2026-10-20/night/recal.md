# recal.md: offline-to-public calibration from the E-OA family, applied to the killed candidates (10-20 night, ~20:30-21:50 IST, Claude, TRACK 1). RESEARCH ONLY.

Nothing was submitted, pushed, committed, versioned or downloaded. No dataset, no GPU job, no scheduled task. PREREG, HANDOFF,
PROGRESS, session_2026-10-19/a20, session_2026-10-20/a21 and session_2026-10-20/val were only read. Everything I wrote is under
`session_2026-10-20/night/` with a `recal_` prefix plus this file (the `syn_*`, `box_syn*`, `lmfx*` files there belong to the
sibling tracks).
**OAI governance.** One read-only CPU script (`recal_box_gapmatch.py`) was piped over ssh stdin into `python -I -`: a B 4 smoke,
then the B 150 run. It printed aggregates only (per model x regime means of rho_ns, q, gap and AUC deltas over weighted
resamples). Nothing was written on the box. No OAI id, row or per-knee value left the box (grep for id patterns in both logs: 0).
The only data sent to the box were five gold74 2x2 count tables (local aggregates). Every other OAI number is an aggregate that was
already on the laptop (named).
**Tags.** MEASURED = computed tonight, log named (all in `night/`). CLAIMED = older note, named. MODEL = arithmetic on stated
inputs. GUESS = judgement. Units: e-4 = 1e-4 macro AUC. "label" = single-label AUC delta (macro = label / 12). Intervals [p10, p90].
**Caveat up front.** The calibration comes from ONE family (five E-OA entries whose public noise is .94-.98 correlated), so it rests
on about 1.5 independent numbers. Whether it transfers to another transform is an assumption, tested per candidate below, not
a given.

## 0. Bottom line

1. **The underestimate is ruler-specific, not a constant factor** (MEASURED `recal_rulers.log`, public CLAIMED val/implied2.md):

   | ruler (offline prediction of the E-OA gain vs c4) | public / offline | shape across the 5 entries |
   |---|---|---|
   | gold58 c4 (kernel path) | .42 / .42 / .48 / .39 / .51: OVERstates ~2.4x | fits (one factor). But gold58 boot sd is 50-90 % of each value, so the factor is .42 [~.25, ~.85] |
   | raw OAI (D5, P_FT) | .38 / .36 / .25 / .35 / .28: overstates ~3x | too steep at a .3 (predicts eoa30 > eoa20 by +3.4, public says -0.7) |
   | OAI coupling-matched at the PROXY's own gap (R3p+0 / R1oa3 / skeptic ladder) | 1.6 / 3.2 / neg / 2.5 / neg (D5); all neg (s1044) | WRONG SHAPE: optimum at a <= .1, a .3 negative. This is the "2-5x" |
   | OAI gap-matched (D5 P_FT read where its gap = c4's .128, regime -0.83 sd) | .80 / .90 / .90 / .83 / .88 | fits all five with one factor (.86) |
   | report OOF (R7 vs report v2) | wrong sign by design (-4.9 / -14.7 / -30.7 / -15.5 / -32.8) | mechanism control, not a ruler |

   Order of the five: eoa10, eoa20, eoa30, pw20, pw30 (public 5.9, 10.3, 9.6, 10.4, 11.3).
2. **Cause: the rulers read the effect at too small a coupling gap, and mostly on the MODEL side.**
   - The reads identify an effective c4 gap rho - q of .128 [.081, .176] (CLAIMED implied2). The OAI proxies at gold74 label
     coupling have gaps of only .075 (D5) and .060 (s1044) under P_FT, and .047 / .028 under P_AREA (MEASURED).
   - Split of the D5 shortfall (.075 -> .128):
     - ~60 % is c4 co-firing the OA trio more (rho .816 on gold58, label-free, vs D5 .784 at gold74 coupling).
     - ~40 % is q falling from .710 to ~.69: hidden labels slightly less coupled than gold74. That is -0.17 sd [-1.0, +0.6]
       (CLAIMED implied2), which cannot be told apart from 0.
     - The public draw also sits ~12 % above the population (CLAIMED implied2 sec 4).
   - So:
     - **(a) a constant factor: NO.**
     - **(b) label-specific (OA only): not identifiable** from macro reads (the free-Medial variant has posterior = prior), and
       not the right frame. The excess is pair-specific through c4's own co-firing.
     - **(c) a regime shift: only in the effective gap, mostly model side.** The reads pin the gap, not how it splits.
3. **How it transfers (the honest rule):**
   - A generic "multiply the killed numbers by 2-5x" is undefined: R2m and R1oa3 disagree in sign for C12 / C13, so the
     multiplier would revive or bury them depending on the ruler.
   - The transferable piece is the gap-matched read: put the OAI curve at the gap c4 has for THAT pair, using c4's label-free rho
     and an expert-label q.
   - The label-side shift is too small and too uncertain to apply generically.
   - gold58-only evidence gets κ ≈ .42 times an Empirical-Bayes-shrunk delta. gold58 directional gaps outside the OA trio have
     split-half reliability .14 (MCL targets .01), so most of each value is noise (MEASURED `recal_reliab.log`).
4. **Result: no killed candidate reaches PROBE.**
   - Three are MAYBE (lottery-grade, below read noise), all on the OA / meniscus side, where the E-OA lesson bites:
     - **LM <- MM at a .05** on pw30: +0.4 [-0.4, +1.2], P(move) .63.
     - **LOA <- MM at a .05** on pw30: +0.3 [-0.5, +1.1], P(move) .60.
     - Their **stack**: +0.6 [-0.6, +1.8], P(move) .64, private +0.6, R1 value +0.57.
     - **LOA += LM** (a -.1, a borrow, a different mechanism): +0.3 [-1.0, +1.6].
   - These are the same size as pw25 (CLAIMED VAL_1020: private +0.56, R1 +0.37).
   - Everything else is NO, including MCL <- MM, every fluid / ACL / acute-label / global transform, convex and SUB_HIGH.
5. **The one real surprise.**
   - The menisci kill (EX C6) was the most exposed to the E-OA lesson. c4 co-fires MM-LM at .31 on RSNA images. OAI proxies give
     .15-.18 at gold74 coupling (MEASURED, box), and c4's own members R6 / S6fall on OAI images give .18-.23 (CLAIMED
     lmfx_work/o1_lm_oai.log). That model-side excess is about 5x the trio's.
   - But the gap-matched box read (MEASURED) shows that single-partner decorrelation in rank space needs a gap of ~.13-.15 to
     break even at a .1 (~.10 at a .05). c4's plausible LM <- MM gap is ~.08 (gold58) to ~.19 (Route O).
   - So it survives only re-dosed to a .05, and only as a lottery.
6. **Recommendation (MODEL / GUESS):** nothing in the killed list beats leaving the 3 free 10-21 slots empty by more than read
   noise.
   - If Eesh wants one non-dose ticket instead of (not as well as) pw25, the least-bad is the stack (LM <- MM .05 + LOA <- MM .05
     on pw30). It carries the same "fishing" caveats as pw25.
   - It would need his GO, a PREREG amendment, a build in the build_a21 pattern and a gate. None of that was done.

## 1. Calibration table (E-OA family; e-4 vs c4 57343435)

Public = CLAIMED val/implied2.md joint refit (truncating display; eoa30 / pw20 are censored and model-assisted). LB-only pooled
draws (MEASURED from val/work/lb_post2.npz, `recal_transfer.log`): G10 6.5 [4.8, 8.3], G20 11.6 [9.8, 13.2], pw30 - eoa20 +0.45
[0.16, 1.16]. Offline MEASURED in `recal_rulers.log` unless tagged.

| ruler | eoa10 | eoa20 | eoa30 | pw20 | pw30 | note |
|---|---|---|---|---|---|---|
| **public (joint)** | **5.9 [5.0, 6.9]** | **10.3 [9.2, 11.3]** | **9.6 [8.1, 11.1]** | **10.4** | **11.3 [10.4, 11.9]** | population ~12 % lower (common-mode draw) |
| gold58 c4 | 14.2 (sd 7.5) | 24.7 (12.7) | 20.1 (17.5) | 26.5 (15.9) | 22.3 (19.9) | boot corr .7-.94 across shapes |
| R7 OOF x gold58 | 7.8 | 23.1 | 15.4 | 26.3 | 20.4 | single model, gap .07 (CLAIMED skeptic_eoa) |
| R7 OOF x gold74 | 2.8 | 6.6 | 1.8 | 10.6 | 1.8 | n 74; erratic |
| report OOF R7 v2 / 3-model mean v2 | -4.9 / +0.3 | -14.7 / -2.1 | -30.7 / -9.3 | -15.5 / -2.2 | -32.8 / -9.7 | must lose (it does) |
| OAI raw D5 / s1044 (P_FT) | 15.6 / 12.5 | 28.9 / 21.5 | 38.5 / 26.4 | 29.3 / 22.1 | 39.7 / 27.3 | OAI coupling (label phi .04) |
| OAI gold74-coupled D5 / s1044 (P_FT, R3p+0) | 3.6 / -0.2 | 3.2 / -5.1 | -2.5 / -15.5 | 4.2 / -3.8 | -0.5 / -14.0 | proxy gap .075 / .060 |
| same, P_AREA | 4.8 / 2.5 | 6.9 / 1.2 | 4.6 / -5.5 | 7.3 / 1.7 | 4.9 / -5.4 | gap .047 / .028 |
| EX R1oa3 (CLAIMED EX_1019; 3 models) | 4.1 / 0.3 / 0.8 | 4.3 / -4.2 / -3.2 | | 5.4 / -2.8 / -1.7 | | |
| skeptic MODEL (CLAIMED a20/skeptic_eoa) | 5 | 7 | | | | gap .096 |
| skeptic ladder (CLAIMED MORE_1020) / R5 revision (CLAIMED implied.md) | 3 / 2 | 1 / -0.5 | | | | |
| **OAI gap-matched, D5 P_FT at gap .128 (regime -0.83)** | **7.4** | **11.5** | **10.7** | **12.5** | **12.9** | x .86 = public |
| OAI gap-matched at gold58's gap .102 (regime -0.44) | 5.5 | 7.5 | 4.3 | 8.5 | 6.4 | too steep at a .3 |
| OAI gap-matched, P_AREA at .128 (regime -1.21) | 10.6 | 19.3 | 24.9 | 19.8 | 25.3 | factor .39-.56 |
| Gaussian coupling theory on c4's gold58 couplings, fitted to eoa10 / eoa20 (s .54 [.33, .78], extra curvature m .04) | 6.4 | 11.7 | 14.0 [10.7, 17.3] | 11.3 | 13.3 [9.8, 16.7] | too flat at a .3 (MEASURED `recal_transfer.log`) |

## 2. Which hypothesis survives (MODEL on the table)

- **Constant factor.** It holds only for rulers that already use c4's own gap: gold58 c4 (.42) and gap-matched OAI (.86).
  - It fails for rulers built on another model's gap.
  - Raw OAI and the gold74-coupled proxies miss the SHAPE, not just the level:
    - The proxies predict eoa10 >= eoa20 and pw30 < 0. The public has eoa20 / eoa10 = 1.75 and pw30 ~ eoa20.
    - With noise sd at n 390 of S10 2.1e-4 and S20 4.3e-4 (corr .94; CLAIMED implied.md x s .82), the D5 R3p+0 shape is
      rejected at >2.5 sd.
- **Label-specific.** Five macro reads cannot split the gain by label. The free-Medial-amplitude variant returns its prior
  (CLAIMED implied2 sec 2.2).
  - What IS label-specific is c4's co-firing excess over the ruler's model. For the trio it is +.03 in rho (ensemble averaging;
    CLAIMED skeptic_eoa sec 2: members' gaps .03-.13, c4 .10).
  - Sec 3 shows this varies pair by pair (MM-LM +.13-.16, LOA-MM ~0).
- **Regime shift.**
  - Using the proxies' own rho, the reads need hidden labels 0.83 sd (P_FT) / 1.21 sd (P_AREA) less coupled than gold74
    (MEASURED regime where the D5 gap = .128).
  - With c4's measured rho the label side is only -0.17 sd [-1.0, +0.6] (CLAIMED).
  - The reads cannot tell these apart. c4's label-free gold58 rho (.816, representative: R7 rho train = gold58, CLAIMED
    skeptic_eoa) favours the model-side reading.
- **Theory check.** Gaussian coupling theory on c4's own gold58 couplings fits eoa10 / eoa20 with amplitude .54 and ~0 extra
  curvature. It then overpredicts eoa30 (14.0 vs 9.6) and pw30 (13.3 vs 11.3): real rank-space curvature is stronger than theory
  at a .3.
  - For single-partner, high-AUC targets it is much worse. On LM <- MM the box curve is 3-5x below the trio-anchored theory at
    the same gap (MEASURED, sec 3.3).
  - So the theory map (`recal_transfer.log` "NO shrink" column) is an upper bound only. It is not used for the verdicts.

## 3. Transfer machinery (MEASURED unless tagged)

### 3.1 How much of a gold58 gap is real outside the trio (`recal_reliab.log`, c4 on the eoa20 base, 126 directions)
- EB on the directional gap rho - q: observed sd .151 against rms se .146-.159, so tau ML = 0 with either se.
- Split-half reliability across directions, Spearman-Brown to n 58:
  - all .137;
  - OA target with a non-OA partner .130;
  - acute targets (MCL, Fracture, Contusion) .014;
  - MCL target .010.
- So a gold58 gap such as MCL <- MM +.42 (z 3.0) shrinks to ~+.03-.08.
- This agrees with the census's held-out test (CLAIMED census.md sec 3) and the skeptic's parametric null (sec 9.1).
- The trio is the one validated exception: gold58 .117 (this parametrisation) vs public-implied .128.

### 3.2 Route O: c4's label-free rho minus an expert-label q from OAI at gold74 coupling (`recal_routeO.log` + box `recal_box_gapmatch.log`)
- **Trio check.** c4 .816 - D5 q .710 = .106 vs public .128: it works for the trio.
- **Measured on the box.** Proxy q moves .47 per unit of label tet (LM <- MM: .03 -> .12 from the OAI to the gold74 log OR); the
  trio gives .43.

| direction | c4 rho_ns (gold58) | proxy q at gold74 coupling (box, 3 models) | Route O gap | gold58 direct gap (robust se) | validity |
|---|---|---|---|---|---|
| LM <- MM | .312 | .123 / .113 / .119 | **+.194** | +.084 (.128) | OK (target / partner head quality similar on both sides) |
| MM <- LM | .312 | .168 / .175 / .169 | +.141 | -.031 (.102) | OK |
| LOA <- MM (pw30 base) | .206 | .115 / .069 / .104 | +.110 | +.224 (.234) | OK |
| LOA <- LM (pw30 base) | .624 | 1.04 / 1.01 / 1.00 | -.39 | -.218 (.150) | q inflated: the proxy LOA head is weak on OAI (AUC .78) |
| MOA <- MM (pw30 base) | .589 | .75 / .71 / .75 | -.15 | -.135 (.105) | MOA at ceiling on c4 (.988) |
| Eff / Syn / Bak, ACL pairs | | | | | invalid: OAI Syn = Hoffa, proxy Syn = Eff; OAI ACL is degenerative |

- **Population, not model.** c4's MM-LM co-firing is RSNA-specific. c4's own members on OAI images give .175 (R6) and .234
  (S6fall) (CLAIMED sibling lmfx_work/o1_lm_oai.log).
- So Route O assumes the extra RSNA co-firing carries no label signal, as the trio's did. If it does carry signal, q rises with it
  and the gap shrinks toward the gold58 value.

### 3.3 Gap-matched box read (`recal_box_gapmatch.log`; per-label AUC deltas, proxy D5 / s1044 / s1045, OR-matched regimes -2..+1 sd)
- **LM <- MM:**
  - Break-even gap ~.13-.15 at a .1 and ~.10 at a .05.
  - At gaps .15-.18: +.0010-.0012 per label (a .05 or a .1). At gap .10: a .1 -.0008, a .05 +.0002. At gaps .045-.065: a .1 -.0028 to -.0030.
- **MM <- LM:** the proxy gap never exceeds .10; a .1 <= +.0001.
- **LOA <- MM:** break-even ~.08 at a .1. Gap .146: a .1 +.0017, a .05 +.0012. Gap .071: a .1 -.0006, a .05 +.0001.
- **LOA <- LM:** decorrelation loses in every regime (gap -.15 to -.33). The borrow (a -.1) gains +.0021-.0083, rising with the
  gap's magnitude.
- **MOA <- MM:** decorrelation loses (gap -.02 to -.17). The borrow gains up to +.0049 on proxies whose MOA AUC is ~.88. c4's MOA
  sits at .988, and gold58 on pw30 shows 0 at a -.1.
- **LOA <- PFOA control (a trio member):** the curve rises smoothly with the gap, as for E-OA.

### 3.4 The OA-target candidates on the actual pw30 base (`recal_oa_targets.log`; gold58 c4, 2,000 boots, 1,000 split-halves)

| change on pw30 | gold58 label d at a: -.10 / -.05 / +.05 / +.10 | P>0 at the best a | net pair flips | split-half same sign | R7 x gold74 | report OOF R7 / 3-mean | read noise pub / priv (e-4) |
|---|---|---|---|---|---|---|---|
| LOA <- MM | -.0077 / -.0039 / **+.0058** / +.0048 | .78 (a .05) | +3 of 517 | .32 | -.0019 (a .05), -.0063 (a .1) | -.0017 / -.0003 (a .05) | 1.93 / 1.29 at a .1 (~1.0 / .65 at a .05, MODEL) |
| LOA += LM | **+.0039** / .0000 / -.0019 / .0000 | .62 (a -.1) | +2 of 517 | .16 | +.0050 (a -.1) | -.0001 / -.0024 (a -.1) | 1.73 / 1.14 (a -.1); .90 / .56 (a -.05) |
| MOA <- MM | .0000 / .0000 / -.0008 / .0000 | | 0 | | | | MOA at ceiling |

## 4. Every killed candidate, recalibrated (macro e-4 vs pw30; MODEL `recal_table.py` -> `recal_table.log`)

- **Column definitions:**
  - mu = calibrated population effect, which is the expected public and the expected private gain.
  - P(move) = P(public delta vs pw30 > 0), the R1 best-id move, with that candidate's own MEASURED paired read noise.
  - R1 = E[delta x 1(move)], the private value of probing then following R1. It carries ~+0.1 to +0.4 of option value for any
    wide zero-mean change, so it does not discriminate.
- **Verdict rule, fixed before computing:** PROBE if mu >= +1.0 and P(mu > 0) >= .80; MAYBE if mu > 0 and P(mu > 0) >= .60;
  else NO.
- **"Same kind?"** asks whether the mechanism is report-vs-expert coupling, the only kind this calibration speaks to.

| candidate (source) | same kind? | offline as recorded (CLAIMED unless noted) | calibration route used | mu [p10, p90] | P(mu>0) | P(move) | private (P>0) | R1 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| C6 menisci MM <-> LM a .1 (EX) | yes | OAI raw -1.5 / -1.7 / -4.0 (e-4 macro); R1 -5.0 / -5.2 / -7.8; g58 +5.7 (P .75, 1-2 rank swaps); report -3.3 | gap-matched box at Route O / gold58 gaps (LM side ~0, MM side -.0013 at a .1) + kappa route | -0.7 [-2.0, +0.6] | .24 | .39 | -0.7 (.35) | -0.12 | NO |
| C6' **LM <- MM only, re-dosed a .05** | yes | g58 +.0019 at a .05, +.0075 at a .1 (census; P .83); OAI R2 -.003..-.005 (proxy gap) | gap-matched box at gap .08-.19 -> +.0002..+.0012 label, x .86 | **+0.4 [-0.4, +1.2]** | .75 | .63 | +0.4 (.67) | +0.37 | **MAYBE** |
| C7 fluid FL3 a .1 (EX) | yes | OAI R2 -.0008..-.0010; g58 -.00047; report -.0009 | Route O Eff/Bak gaps <= 0; Syn untestable; kappa | -0.5 [-1.8, +0.8] | .31 | .44 | -0.5 | -0.10 | NO |
| C8 fluid ES2 a .1 (EX) | yes | g58 -.0008 (Syn -.0084); OAI uninformative | kappa x EB | -0.5 [-1.5, +0.5] | .27 | .41 | -0.5 | -0.09 | NO |
| C9 OA-meniscus 4 dirs (EX) | yes | OAI increment -.0010 / -.0013 / -.0006 macro; all reweighted <= 0 | c4 gaps (-.13..-.22) ~ proxies' -> EX R2 x .86 | -0.8 [-1.3, -0.3] | .02 | .41 | -0.8 | -0.31 | NO |
| C10 ACL <- LM / MM / Eff (EX) | partly | OAI ACL -.0037..-.0114 (degenerative ACL); g58 4 / 5 < 0 | kappa x EB; ACL at .989 | -0.5 [-1.1, +0.1] | .16 | .40 | -0.5 | -0.16 | NO |
| C11 global G12 (EX) | yes | OAI R9 -.0006..-.0011; g58 on E-OA +.00028; report -.0018 | sign hinges on a GENERIC label-side q shift (-.02 gives theory +4.6, 0 gives +1.1, both theory upper bounds); OAI says < 0 | 0.0 [-2.6, +2.6] | .50 | .50 | 0.0 | +0.36 | NO |
| C12 LOA <- MM a .1 on pw30 (EX / census / implied sec 6) | yes | on E-OA .2: raw +.00008 / +.00003 / -.00004, R1oa3 -.0003..-.0005; R2m_lo +.00027 | gap-matched box at c4 gap .06-.12 | +0.1 [-0.9, +1.1] | .55 | .52 | +0.1 | +0.17 | NO |
| C12' **LOA <- MM, re-dosed a .05** on pw30 | yes | g58 pw30 base +.0058 (P .78, 3 flips) | same, a .05 | **+0.3 [-0.5, +1.1]** | .69 | .60 | +0.3 (.63) | +0.30 | **MAYBE** |
| **stack C6' + C12'** (two lines on pw30) | yes | as above | sum (independent errors, GUESS) | **+0.6 [-0.6, +1.8]** | .75 | .64 | +0.6 (.68) | +0.57 | **MAYBE** |
| C13 BOR MOA+=MM, LOA+=LM a -.1 (EX) | no (deficit / attenuation) | raw +.00061 / +.00059 / +.00029; R2m +.00018 / +.00022 / -.00005; g58 -.00007 | LOA half as C14; MOA half ~0 (ceiling) | +0.2 [-1.3, +1.7] | .57 | .53 | +0.2 | +0.32 | NO |
| C14 LOA += LM a -.1 on pw30 (EX) | no (reports and experts couple LOA-LM equally, .60 vs .61) | raw LOA +.0054 / +.0054 / +.0034; R2m +.0026 / +.0030 / +.0011; g58 +.00032 macro | gap-matched box at c4 gap ~-.1 +- .12 (gold58 -.22 EB-shrunk; Route O -.39 inflated) | +0.3 [-1.0, +1.6] | .62 | .56 | +0.3 (.58) | +0.37 | MAYBE (outside scope) |
| MCL <- MM a .1 (census, more / expert2) | yes | g58 +.0091 label (P .79; 4 flips of 441); no OAI ruler; skeptic KILL (specificity, p .057) | kappa .42 x EB-shrunk (MCL reliability .01) | -0.2 [-1.5, +1.1] | .42 | .47 | -0.2 | +0.05 | NO |
| Frac <- MOA a .1 (census) | yes | g58 +.0118 (P .89); placebo helpers gain as much; extra16 -.025 | kappa x EB | -0.3 [-1.6, +1.0] | .38 | .47 | -0.3 | -0.03 | NO |
| MCL <- mean11 a .1 (census) | yes | g58 +.0023 (+.0091 re-ranked, a forking path) | kappa x EB | 0.0 [-0.8, +0.8] | .50 | .50 | 0.0 | +0.09 | NO |
| LOA <- Eff borrow a -.1 (census) | no (the deficit is a gold58 artefact: c4 rho .08 vs R7 train .33) | g58 +.0116 (P .90) | kappa x EB; hidden c4 rho ~.3 removes the room | -0.3 [-1.6, +1.0] | .38 | .44 | -0.3 | +0.06 | NO |
| MCL <- Syn / Cont / Frac / MOA (census) | yes | g58 +.0045..+.0091 | kappa x EB | -0.3 [-1.6, +1.0] | .38 | .46 | -0.3 | 0.00 | NO |
| LM <- MM a .1 / MM <- LOA a .1 (census KILL) | yes | OAI-killed | gap-matched box | -0.1 / -1.0 | .45 / .02 | .48 / .26 | -0.1 / -1.0 | | NO |
| MCL + Frac minus mean(DEG) a .1 (MORE / expert2) | yes | g58 +.00035 (P .68); random-helper null rank 175 / 252 | kappa x EB | -0.2 [-1.5, +1.1] | .42 | .48 | -0.2 | +0.01 | NO |
| Contusion <-> Fracture a .1 (MORE / expert2) | yes | q ~ rho (gap +.04 / -.05); g58 +.00001 | theory, no room | -0.1 [-1.1, +0.9] | .45 | .49 | -0.1 | +0.03 | NO |
| E2 Syn += Eff a -.1 / a -.2 (THINK) | partly: a label-definition gap (expert Syn broader), not a coupling excess (c4 Eff-Syn rho .74 already exceeds expert tet .62) | g58 +.0048 / +.0084 label; best of 9 / 132 (null max ~+.0045) | kappa x EB (no OAI ruler) | -0.1 / -0.5 | .46 / .37 | .48 / .43 | -0.1 / -0.5 | +0.17 / +0.07 | NO on this route. Sibling Track 2 (CLAIMED synovitis.md) rates E2s (a .2) MAYBE, +0.4, via a site-language prior that this calibration does not speak to |
| Convex E-OA twin p2 (LEVERS) | yes (E-OA shape) | selection-corrected winner held-out R1: +0.29 / -0.46; RSNA INFO p2 - lin < 0 | gap-matched regime R3p_lo (D5 gap .147 ~ c4): winner +0.36 / +0.36, fixed p2 +1.12 / +1.71 (p2 picked on all knees), x .82; gold58 c4 p2 - lin -1.4..-7.7 x kappa | +0.2 [-1.1, +1.5] | .58 | .53 | +0.2 | +0.27 | NO (the calibration moves it up ~+0.2, not enough) |
| SUB_HIGH (LEVERS) | yes (E-OA shape) | wins 8-37 of 80 half-searches; held out ~0 | same as convex (R3p_lo) | +0.2 [-1.1, +1.5] | .58 | .53 | +0.2 | +0.27 | NO |
| Raptor PF down / S6fall OA up / Raptor OA up (LEVERS) | no (member-weight knobs) | F1 Raptor OA up wrong sign 7 / 7; F4 S6fall at R3p_lo: -.0004..+.0006 OA-mean | not applicable; at R3p_lo S6fall OA up is ~0 to +0.5 | about 0 | | | | | NO (unchanged) |
| T-STATE, T-VOL (LEVERS) | no | rubric-text OOF, OAI effusion | not applicable | T-STATE ~+0.1, T-VOL < 0 (CLAIMED) | | | | | NO (unchanged) |

Notes on the table:
- **How mu was set** (MODEL): mu is a precision-weighted judgement over the routes shown. Gap-matched box reads are scaled by the
  OAI -> public amplitude .82-.86. gold58 values are multiplied by kappa .42 after EB shrinkage with the reliabilities of sec 3.1.
  The full arithmetic and the read-noise sds are in `recal_table.py`.
  - Where a route is missing (MCL, Fracture, Contusion, Synovitis: no valid OAI label), mu rests on gold58 alone. It is therefore
    ~0 by construction of the EB shrinkage, not by a measurement.
- **Display (MODEL):** P(.960 display) for the MAYBE rows is ~.35-.45. That only reflects their larger read noise around a gap of
  0.75 [0.3, 1.8]e-4 (CLAIMED implied2). It is lottery mass, not evidence.
- **Multiplicity (GUESS):** I looked at ~25 candidates under 3-4 routes. The three MAYBEs are the top of that search, so shade them
  ~30 % (stack ~+0.4).

## 5. What would change this

- **A hidden-label read of any one meniscus change.** That is the only way to learn whether c4's extra RSNA MM-LM co-firing
  carries label signal (gap shrinks to gold58's .08) or not (Route O's .19).
- **A box run of LOA <- MM ON TOP OF a gap-matched E-OA pw block,** instead of standalone. The standalone curve is used here;
  E-OA lowers rho(LOA, MM) from .31 to .21 on c4. Not done: 3-label plus pair reweighting is a bigger script.
- **The generic label-side shift.** If the private labels are >= 0.5 sd less coupled than gold74 everywhere (implied2 sec 8.4 puts
  that out of reach under a random split), C11 global would move toward +1 to +3. It stays NO here.
- **This calibration is one family.**
  - The trio is the only place where c4's gap, an independent expert ruler (OAI) and the public read all exist together.
  - Every transfer above assumes that a gap-matched OAI curve reads c4's hidden response for other pairs as it did for the trio.
  - That held for the trio's shape (five reads, one factor). It is untested elsewhere.

## 6. Files (`session_2026-10-20/night/`)

- **`recal_rulers.py` -> `recal_rulers.log` / `.json`** (local + OAI aggregates on disk). E-OA offline predictions on every ruler,
  gold58 bootstrap and correlation, OAI curves at the raw, gold74 and gap-matched regimes.
- **`recal_transfer.py` -> `recal_transfer.log` / `.json`, `recal_eb_gaps.csv`.**
  - The Gaussian coupling-theory map fitted to the public (s, m).
  - EB on 126 gold58 directional gaps.
  - Per-candidate theory columns (upper bounds) and read noise (R7 OOF + sol56, 390 / 910).
- **`recal_reliab.py` -> `recal_reliab.log`, `recal_reliab_dirs.csv`.** Robust-se EB and split-half reliability of gold58 gaps.
- **`recal_routeO.py` -> `recal_routeO.log` / `.json`.** c4 rho vs proxy rho / q per direction (box JSON already on disk).
- **`recal_box_gapmatch.py` -> `recal_box_gapmatch_smoke.log`, `recal_box_gapmatch.log`.** Box CPU, aggregates only.
- **`recal_oa_targets.py` -> `recal_oa_targets.log`.** LOA <- MM, LOA += LM, MOA <- MM on the pw30 base: gold58, R7 x gold74,
  report OOF, read noise.
- **`recal_table.py` -> `recal_table.log`.** Decision arithmetic for sec 4.
