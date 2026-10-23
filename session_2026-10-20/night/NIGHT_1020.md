# NIGHT_1020.md: night synthesis, killed candidates re-scored after the E-OA lesson (10-20 ~20:30-22:15 IST, Claude)

RESEARCH + LOCAL BUILDS ONLY. Nothing was submitted, pushed, committed, versioned or downloaded. No dataset, no GPU job, no scheduled
task. PREREG, HANDOFF, PROGRESS, session_2026-10-19/a20, session_2026-10-20/a21 and session_2026-10-20/val were only read (the a21 pw30
committed dir was copied, not changed). Everything written is under `session_2026-10-20/night/`.
**OAI governance.** This synthesis touched no OAI data: `adj_c12.py` uses RSNA train OOF + gold74 counts, `night_slate_mc.py` is
arithmetic, the build / gate / sim use RSNA kernel outputs. The sibling tracks used the box only through ssh-stdin scripts printing
aggregates (CLAIMED recal.md, synovitis.md, lmfx.md, skeptic logs); their box logs were read here as aggregates only.
**Tags.** MEASURED = computed tonight, log named (all in `night/`). CLAIMED = a sibling or older note (named). MODEL = arithmetic on
stated inputs. GUESS = judgement. Units: e-4 = 1e-4 macro AUC vs pw30 57425970 unless stated; "label" = one-label AUC delta (macro =
label / 12). Intervals are [p10, p90]. "R1" = A is the API best-id (max public); "increment" = expected private gain of adding a probe
to {r6pw20, pw25} under R1.
**Sources.** Track 1 `recal.md`, Track 2 `synovitis.md`, Track 3 `lmfx.md`; skeptics `skc6/`, `skep_c12p*`, `skep_box_c12p*`,
`skep_stack*`, `skc14.*`, `skp_syn*`; this pass `adj_c12.log`, `night_slate_mc.log`, `c12_gate.log`, `c12_sim.log`. The skeptic
verdict texts for the stack, C14 and E2s were not in my brief; their verdicts below are derived from their logs and say so.

## 1. Verdict

1. **Slot 3 on 10-21: one optional lottery probe, lomm05 (C12': pw30 + Lateral OA <- Medial Meniscus at a .05).** It is priority 3,
   after r6pw20 and pw25, and it needs Eesh's GO. **Slots 4 and 5 stay empty.**
2. Nothing reached PROBE grade (recal rule: mu >= +1.0e-4 and P(mu > 0) >= .80). The skeptics killed C6', C12', the stack, C14 and E2s.
3. lomm05 is kept as a ticket, not a lever: E +0.4e-4 [-1.4, +2.2] against 1.0e-4 public read noise, P(beats pw30 on public) about .60 (MODEL).
4. Why it alone: its slate increment stays positive with its own skeptic's inputs (+0.21 to +0.54e-4; -0.07 only in an OAI-only world),
   and the night's one new independent ruler (RSNA train OOF raked to expert couplings) reads it up in 5 of 5 models, MM-specific (MEASURED).
5. E2s, the stack, C6' and C14 are KILL (increments -0.03, no better than C12' alone, -0.1, about 0). Every other re-scored candidate stays NO.
6. lomm05 is built and offline-gated (C12_GATE PASS). Its commit gate is written and self-tested (PASS 9/9, including PASS 18/18 on the REAL scored pw30 output). Nothing is armed.
7. Slates: minimum = r6pw20 alone; standard = + pw25 (VAL_1020); full = + lomm05. If only one A probe runs, keep pw25, the validated one.

## 2. Every proposed probe

"Calibrated public" = expected public (and private) effect vs pw30 in e-4. Offline numbers are as recorded by their source (CLAIMED);
rows 1-5 also carry tonight's checks.

### 2.1 Night proposals that reached a skeptic

| # | probe (source) | one change | offline | calibrated public | skeptic | NIGHT verdict |
|---|---|---|---|---|---|---|
| 1 | **C12' LOA <- MM a .05 on pw30** (recal) | `LOA = pct(pct(LOA) - .05 pct(MM))` after E-OA | gold58 pw30 +.0058 label (rank 1 of 132 at a .05, placebo P .083 / .25; MEASURED by skeptic). R7 x gold74 .0000. OAI c4 members at the public trio regime, gold74 LOA-MM coupling: R6 -.0013, S6fall -.0010, c4D5 -.0013 label (skep_box_c12p_mem). **RSNA train OOF raked to the gold74 expert (MOA, LOA, PF, MM) joint: +0.84 / +1.61 / +1.23 / +1.56 / +2.11e-4 for R7 / s1044 / s1045 / S6mean / 3mean, all P >= .94 (MEASURED adj_c12.log)** | recal +0.3 [-0.5, +1.1]. Skeptic +0.2 [-1.3, +1.7] (P > 0 .56). **NIGHT +0.41 [-1.43, +2.20], P(> 0) .62, P(move) .60** (MODEL night_slate_mc.log) | **KILL** (skep_c12p_post: P(move) .55, private +0.2; the gold58 gain is selection noise; OAI members negative) | **KEEP, lottery, priority 3.** Sec 3 explains why the KILL is reversed only this far |
| 2 | C6' LM <- MM a .05 on pw30 (recal) | `LM = pct(pct(LM) - .05 pct(MM))` | gold58 +.0019 label. OAI R2 -.003..-.005. RSNA raked (MM, LM) 2x2: R7 -0.11, S6 seeds +0.31 / +0.38, 3mean +0.63e-4 (MEASURED adj_c12). For this pair the c4-like model is R7: c4 .312 vs R7 .340 on gold58, while the S6 seeds and 3mean co-fire at .53-.54 on train vs R7 .42 | recal +0.4 [-0.4, +1.2]. Skeptic -0.1 [-0.9, +0.7], P(move) .47. c4-like ruler -0.1 | **KILL** (skc6: Route O gap is an artefact of the OAI bridge, which fails for this pair, RSNA q .28 vs OAI .12; RSNA ruler at gold74 coupling -.0003 label) | **KILL** |
| 3 | stack C6' + C12' (recal) | both lines | as rows 1-2 | recal +0.6 [-0.6, +1.8]. With skeptic inputs mu +0.19, increment +0.52 vs C12' alone +0.54 (MODEL) | **KILL** (derived from skep_stack1-6: C6' half ~0 under the c4-like ruler, label-side mechanism holds for both pairs) | **KILL**: the C6' half adds noise and a second change for no gain |
| 4 | C14 LOA += LM borrow a -.1 on pw30 (recal) | `LOA = pct(pct(LOA) + .1 pct(LM))` | OAI raw LOA +.0054 / +.0054 / +.0034 (weak OAI LOA heads). gold58 c4 +.0039 = net +2 pair flips of 517; LOO flips sign in 2 of 58. R7 pw30 gold58 -.0029 (MEASURED skc14) | recal +0.3 [-1.0, +1.6]. Skeptic about 0 to -0.3, R1 +0.03..+0.17 | **KILL** (derived from skc14.log: no report-vs-expert coupling excess, LOA-LM tet report .601 vs expert .608; 3-model report ruler -.0025 / -.0033 label, P > 0 .00) | **KILL** (it also competes with C12' for the LOA column) |
| 5 | E2s Synovitis += .2 Effusion on pw30 (synovitis.md) | `Syn = pct(pct(Syn) + .2 pct(Eff))` | gold58 c4 +.0084 label (P .82; top 3 studies 78 %). OOF site-language prior Spearman .62-.88 on raw Syn means. OAI cannot define the target | proposer +0.4 (-0.3 to +1.1), P(move) .54. Skeptic base -0.26, P(> 0) .39 | **KILL** (derived from skp_syn*: after residualising Syn on Eff the language gradient falls to Spearman .23-.45, and with English set aside the low-site gap is about -.02 (R7 -.075 raw); it survives train case-mix control, but on gold58 the site offset drops from -.124 (p .026) to -.088 (p .086) once expert ACL is controlled; +.0078 of the +.0084 comes from 10 ACL+ positives; slate increment -0.03) | **KILL**. Its build stays unarmed (`syn_kernel_v224r6c4pw30syn20j`). Adding it next to C12' lowers the slate (+0.75 vs +0.87, MODEL) |

### 2.2 Re-scored killed candidates that were not escalated (proposer already NO; recal.md sec 4, CLAIMED)

| probe (source) | offline as recorded | calibrated public [p10, p90] | P(move) | verdict |
|---|---|---|---|---|
| C6 menisci MM <-> LM a .1 (EX) | OAI raw -1.5 / -1.7 / -4.0e-4; R1 -5.0..-7.8; g58 +5.7; report -3.3 | -0.7 [-2.0, +0.6] | .39 | NO |
| C7 fluid FL3 a .1 (EX) | OAI R2 -.0008..-.0010; g58 -.00047; report -.0009 | -0.5 [-1.8, +0.8] | .44 | NO |
| C8 fluid ES2 Eff <-> Syn a .1 (EX) | g58 -.0008 (Syn -.0084) | -0.5 [-1.5, +0.5] | .41 | NO |
| C9 OA-meniscus, 4 directions (EX) | OAI increment -.0010 / -.0013 / -.0006 | -0.8 [-1.3, -0.3] | .41 | NO |
| C10 ACL <- LM / MM / Eff (EX) | OAI ACL -.0037..-.0114 (degenerative ACL); g58 4 / 5 < 0 | -0.5 [-1.1, +0.1] | .40 | NO |
| C11 global G12 a .1 (EX) | OAI R9 -.0006..-.0011; g58 +.00028; report -.0018 | 0.0 [-2.6, +2.6] | .50 | NO |
| C12 LOA <- MM a .1 (EX, census) | g58 pw30 +.0048; OAI members -.0034 label at k 0; RSNA raked +1.1..+3.4e-4; read noise 2.0 public | +0.1 [-0.9, +1.1] | .52 | NO (C12' carries the line at the lower-variance dose; the dose was fixed before tonight's ruler) |
| C13 BOR MOA += MM, LOA += LM (EX) | raw +.00061 / +.00059 / +.00029; g58 -.00007 | +0.2 [-1.3, +1.7] | .53 | NO |
| MCL <- MM a .1 (census, expert2) | g58 +.0091 (4 flips of 441); reliability of MCL gaps .01 | -0.2 [-1.5, +1.1] | .47 | NO |
| Frac <- MOA a .1 (census) | g58 +.0118; placebo helpers gain as much; extra16 -.025 | -0.3 [-1.6, +1.0] | .47 | NO |
| MCL <- mean11, MCL <- Syn / Cont / Frac / MOA (census) | g58 +.0023..+.0091 (forking path) | -0.3 to 0.0 | .46-.50 | NO |
| LOA <- Eff borrow a -.1 (census) | g58 +.0116 (deficit is a gold58 artefact) | -0.3 [-1.6, +1.0] | .44 | NO |
| LM <- MM a .1, MM <- LOA a .1 (census) | OAI-killed; gap-matched break-even ~.13-.15 | -0.1 / -1.0 | .48 / .26 | NO |
| MCL + Frac minus mean(DEG) (MORE, expert2) | g58 +.00035; random-helper rank 175 / 252 | -0.2 [-1.5, +1.1] | .48 | NO |
| Contusion <-> Fracture (MORE, expert2) | q ~ rho; g58 +.00001 | -0.1 [-1.1, +0.9] | .49 | NO |
| E2 Syn += Eff a -.1 (THINK spec) | g58 +.0036 (P .76) | -0.1 (coupling route) | .48 | NO (substitute dose of E2s; dominated) |
| Convex E-OA p2 (LEVERS) | selection-corrected held-out +0.29 / -0.46 | +0.2 [-1.1, +1.5] | .53 | NO |
| SUB_HIGH (LEVERS) | held out ~0 | +0.2 [-1.1, +1.5] | .53 | NO |
| Raptor / S6fall OA weights, T-STATE, T-VOL (LEVERS) | F1 wrong sign 7 / 7; T-VOL < 0 | about 0 | n/a | NO |
| LM ACL-context shrink k .4 (lmfx) | g58 +.0062; OAI R6 / S6fall / c4-shaped -.0091 / -.0054 / -.0068 (P .00); report -.0046 / -.0104 | -1.5 (sd 4) | about .35 | NO |
| Frac <- mean(OA trio) a .1 (lmfx) | g58 +.0042; placebo set +.0076; report -.0017 / -.0023 | -0.5 (sd 3) | about .45 | NO |
| Synovitis extras (synovitis.md sec 9): E2 x (1 - ACL), Syn borrows Frac / Cont, Hoffa route, R6 Syn weight, OR / AND, Syn <- MOA, site-centring | gold58 only or post hoc; site-centring not implementable | about 0 or below | n/a | NO |

## 3. C12': why the skeptic's KILL becomes a lottery KEEP, and no further

**What the C12' skeptic got right (CLAIMED skep_c12p.log, skep_c12p_post.log, skep_box_c12p*.log):**
- The gold58 +.0058 is selection noise: rank 1 of 132 directions, not monotone in a, placebo P .083.
- On OAI with c4's own members at the public trio regime and gold74 LOA-MM coupling, the line loses -.0010 to -.0013 label.
- The calibrated value is small even in its favourable scenarios.

**What it under-weighted:**
- Those OAI reads sit at the members' own OAI co-firing. Their pw30-base gap rho - q is -.04 to +.03 (rho .17-.21).
- c4 co-fires LOA with MM more on RSNA images: .206 on gold58 (CLAIMED skep_c12p_rho.log), about .40 on a train-like mix (MODEL:
  c4 - R7 on gold58 +.077, CLAIMED skep_stack4.log, ≈ 3mean - R7 on train +.072, MEASURED adj_c12.log). Reading a proxy at its own
  small gap is the reading that missed E-OA by 2-5x (recal.md sec 0).
- The skeptic's own box law says the line pays at a .05 once the gap exceeds .063 (gain_label = .0157 x gap - .00098, CLAIMED
  skep_c12p_post.log).

**New ruler, population-matched (MEASURED `adj_c12.log`; non-gold train n 3,467, report v2 labels raked by IPF to gold74's EXPERT
16-cell (MOA, LOA, PF, MM) joint, +.5 smoothing; same rows for every model):**

| model | trio co-firing MOA~LOA | LOA~MM on pw30 | E-OA trio-raked, public / ruler (eoa10 .. pw30) | C12' a .05 raked (P > 0) | raw report coupling | MM rank among 11 partners | raking effect MM vs placebo mean (max) | C6' a .05 raked |
|---|---|---|---|---|---|---|---|---|
| c4 (gold58, reference) | .845 | .206 | public 5.9 / 10.3 / 9.6 / 10.4 / 11.3 | | | | | |
| R7 | .824 | .328 | ruler wrong sign from eoa20 on (pw30 -10.3) | +0.84 (.94) | -.0017 label | 1 | +.0028 vs +.0011 (+.0028) | -0.11 |
| s1044 | .842 | .387 | 1.70 / 2.12 / 3.25 / 1.75 / 2.99 | +1.61 (1.00) | -.0009 | 2 | +.0028 vs +.0011 (+.0023) | +0.31 |
| s1045 | .835 | .376 | 1.85 / 2.66 / 14 / 2.25 / 9.5 | +1.23 (.99) | -.0012 | 2 | +.0027 vs +.0009 (+.0021) | +0.38 |
| S6mean | .841 | .385 | 1.69 / 1.98 / 3.03 / 1.68 / 2.85 | +1.56 (1.00) | -.0010 | 2 | +.0028 vs +.0010 (+.0021) | +0.42 |
| **3mean (R7 + s1044 + s1045)** | **.854** | **.400** | **1.04 / 1.12 / 1.04 / 1.01 / 1.03** | **+2.11 (1.00)** | -.0003 | 1 | +.0029 vs +.0012 (+.0024) | +0.63 |

Reads:
- **The ruler fits E-OA only when the model co-fires the trio like c4.** The 3-mean matches the five public reads at 1.01-1.12. The S6
  seeds (co-firing .835-.842) understate them 2-3x (s1045 up to 14x at a .3), and R7 (.824) gets the sign wrong. That is the recal finding again, seen from
  the RSNA side: the E-OA effect is driven by model co-firing and is steep in it.
- **The 3-mean is also the c4-like model for LOA~MM** (above). Its C12' read is +2.11e-4. Every model reads C12' positive.
- **The MM effect is specific.** The raking effect, raked minus raw, is largest for MM in every model. Raw report coupling is negative
  in every model, as it must be.
- **The absolute level is partly generic.** Under the 3-mean raking, LOA <- Contusion / ACL / MCL also gain (+.0014 to +.0020 label),
  so part of the +.0025 is a property of that ruler's LOA head.
- **The ruler cannot test H_prop**, the case where c4's extra RSNA co-firing carries real LOA signal. Within report cells it inherits
  report-label errors.
- **It also assumes a train-like hidden mix.** gold58 is acute-enriched (ACL .50 vs .21 by report), and R7 co-fires LOA~MM .2 lower
  there. GUESS: the hidden set is closer to train, because gold58 c4 overstated E-OA 2.4x while the train-mix 3-mean did not. Not shown.

**Scenario mix (MODEL `night_slate_mc.log`).** I used the skeptic's three scenarios (CLAIMED skep_c12p_post.log):
- train-like, no signal: +1.54, sd 1.23;
- train-like, q proportional to rho: -0.48, sd 1.23;
- gold-like: -0.26, sd 1.65.

The new ruler moves weight toward the first. NIGHT weights are .45 / .30 / .25, with a .85 haircut for the ~25-candidate search
(the skeptic used .35 / .35 / .30).

| C12' prior (e-4) | mu | P(mu > 0) | alone: P(move), R1 value | added to {r6pw20, pw25}: increment | P(C12' becomes A) | P(A truly harmful) vs pw25 alone |
|---|---|---|---|---|---|---|
| NIGHT .45 / .30 / .25 x .85 | +0.41 [-1.43, +2.20] | .62 | .60, +0.70 | **+0.50** | .44 | .13 vs .13 |
| skeptic mix x 1.0 / x .7 | +0.29 / +0.21 | .57 | .56 / .55, +0.73 / +0.46 | +0.54 / +0.31 | .42 / .39 | .13 / .15 vs .13 |
| pessimistic .20 / .45 / .35 x .7 | 0.00 | .49 | .50, +0.33 | +0.21 | .34 | .16 vs .13 |
| OAI-only N(-0.9, 0.6) | -0.90 | .07 | .22, -0.11 | **-0.07** | .13 | .22 vs .13 |
| RSNA-raked-only N(+2.1, 0.8) | +2.10 | 1.00 | .95, +2.04 | +1.58 | .79 | .03 vs .13 |

Inputs: pw25 true N(+0.3, 0.8), paired public sd 1.4 (CLAIMED VAL_1020 / implied2). C12' paired public sd 1.01 (MEASURED skep_c12p.log).
With read-noise correlation .3 between the two probes, every increment moves by at most +0.02 (MODEL, same log).

- **Why KEEP:** this is the admission rule VAL_1020 used for pw25 ("never negative in any variant"). C12' clears it except in the
  OAI-only world (-0.07). It does not raise the chance of a harmful A in its central rows.
- **Why only a lottery:** mu is below read noise and below the PROBE bar. P(.960 display) is about .36-.43 (MODEL, gap .75e-4 from
  implied2). That is the same noise-driven lottery mass as for any wide change. A .960 would not prove anything.
- **What would flip it back to KILL:**
  - evidence that the hidden set is gold58-like in composition;
  - an expert-labelled RSNA read of LOA~MM coupling showing that the extra co-firing carries signal;
  - Eesh choosing the minimum slate.

None of these is available before 05:31 IST.

## 4. KEEP build: lomm05 (all MEASURED, local, unarmed)

**Spec.**
- Base: scored pw30 57425970, notebook `a21/kernel_v224r6c4pw30j` (sha 5a903187...).
- One change in cell 18 (`_s6f_blend`, S6F path), right after the A21 E-OA block. This is the same insertion point and the same
  6-line pattern as the E2s build:
  `out['Lateral OA'] = pct(pct(LOA_final) - 0.05 * pct(MM_final))`
- It runs only if neither column is blanked. The receipt records `a_loamm`, and the log prints `LOAMM_APPLIED`.
- Every other column and every other cell is byte-identical to pw30.
- The dose a .05 comes from recal (the box break-even), before tonight's ruler. It was not re-picked.
- Never stack it with pw25 or anything else in one entry.

**Build** (`c12_build.py`):

| dir | notebook sha256 | kernel-metadata sha256 | role |
|---|---|---|---|
| `night/c12_kernel_v224r6c4pw30lomm05j/` | c0dd7ec8ce3bad8fd3b0ebd99c7b25c4aefb8efcc48cc425089f851d64d679e5 | 170f47adff91823c607f21698128dde4367fa59ffe2bb33bf204dc44d397c480 | the probe, id `eeshsaxena/rsna-sub-v224r6c4pw30lomm05j` |
| `night/c12_kernel_v224r6c4pw30lomm00j/` | fd1c2d100157d2049df4a22fec9d195e239c8b0626caefc3a6305c9512eb334b | | no-op control |

- Only cell 18 differs from pw30, by +6 lines. The metadata differs from pw30's only in id / title / code_file (asserted).
- Receipts: `c12_build_receipt_*.json` (armed: false).

**Offline gate** (`c12_gate.py` -> `c12_gate.log`: **C12_GATE PASS**). It execs the exact cell-18 code of c4, pw30, lomm00 and lomm05
on the gold58 serve predictions.
- G1: lomm00 is byte-identical to pw30 (sha cab96d91...) and has no `a_loamm`.
- G2: in lomm05, 11 of 11 other columns are identical to pw30. LOA equals `pct(pct(LOA) - .05 pct(MM))` of pw30's own output (max
  diff 4.1e-8). The receipt shows a_loamm .05, a_oa .3, pw.
- G3: lomm05 equals a from-scratch recompute (c4 output -> pw a .3 on the trio -> float32 -> the C12' line) on all 12 columns (4.1e-8).
- G4 (info): gold58 LOA AUC .8511 -> .8569 (+.0058). Macro .94336 -> .94384.

**Commit gate** (new check needed, so I wrote it): `night/c12_probe_check.py`.
- It is a copy of `a21/probe_check_eoa21.py` (a21 untouched) plus `--aloamm`.
- (d) extended: final == .8 / .2 blend, then E-OA, then float32, then the C12' line.
- New (d4): the receipt has `a_loamm` == the dose and the log has `LOAMM_APPLIED`. With `--aloamm 0` it requires neither.
- Total: 18 checks.

`c12_sim.py` -> `c12_sim.log`. The REAL committed pw30 dir plus lomm05's cell 18 replayed on its own committed inputs. The pw30 replay
equals its real submission.csv. **C12_SIM_GATE_SELFTEST PASS:**

| gate run | expected | got |
|---|---|---|
| C12' gate (aloamm .05) on the lomm05 sim | PASS | PASS 18/18 |
| C12' gate aloamm 0 on the REAL scored pw30 dir (self-test on the scored output) | PASS | PASS 18/18 |
| a21 gate on the REAL scored pw30 dir (reference) | PASS | PASS 17/17 |
| C12' gate (aloamm .05) on the REAL pw30 dir | FAIL | FAIL 17/18, (d4) |
| C12' gate aloamm 0 on the lomm05 sim | FAIL | FAIL 17/18, (d4) |
| wrong dose (aloamm .1) on the lomm05 sim | FAIL | FAIL 17/18, (d4) |
| wrong E-OA dose (aoa .25) on the lomm05 sim | FAIL | FAIL 17/18, (d2) |
| tampered LOA (2 values swapped) on the lomm05 sim | FAIL | FAIL 17/18, (d) |
| INFO: old a21 gate on the lomm05 sim | PASS | PASS 17/17 |

- **Caveat:** the 3 committed public rows cannot show the change. LOA ranks [.33, 1, .67] stay put because MM runs the other way
  ([1, .67, .33]). So the a21 gate would pass a lomm05 commit unchanged. Dose identity rests on the tick's registry sha guard plus
  (d4), as it does for the OA doses and E2s. The (d) arithmetic is proven on 58 rows (G2 / G3) and by the tamper row.
- **Gate command** (on the fetched committed dir `<OUT>`; it must print `PROBE_CHECK PASS 18/18`):
  ```
  py -3.12 -I session_2026-10-20/night/c12_probe_check.py <OUT> --a224 0.25 --table '{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker'"'"'s": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}' --aoa 0.3 --mode pw --aloamm 0.05
  ```
- **Registry entry for main to add to `val/slate_a22.json`** (I cannot write val/):
  `"lomm05": {"tag": "v224r6c4pw30lomm05j", "slug": "eeshsaxena/rsna-sub-v224r6c4pw30lomm05j", "kernel_dir": "session_2026-10-20/night/c12_kernel_v224r6c4pw30lomm05j", "nb_sha256": "c0dd7ec8ce3bad8fd3b0ebd99c7b25c4aefb8efcc48cc425089f851d64d679e5", "meta_sha256": "170f47adff91823c607f21698128dde4367fa59ffe2bb33bf204dc44d397c480", "base": "c4", "a_oa": 0.3, "oa_mode": "pw", "gate_script": "session_2026-10-20/night/c12_probe_check.py", "gate_args": [pw25's args with "--aoa", "0.3", then "--aloamm", "0.05"], "gate_pass": "PASS 18/18", "desc": "A22-LOMM05: pw30 57425970 + Lateral OA <- Medial Meniscus a .05 on the final output (one change)"}`
- **Before re-running the tick self-test**, copy `night/c12_sim/out_v224r6c4pw30lomm05j_sim` to `val/sim/`. The self-test looks for
  `val/sim/out_<tag>_sim`. Then `py -3.12 -I session_2026-10-20/val/tick_a22.py --selftest` should read PASS 28/28: 26 plus a
  registry check and a gate check for lomm05.
- **Draft PREREG text** (for main, inside Amendment 22, because the tick guard looks for the literal `AMENDMENT 22`; NOT written):
  > - lomm05 = rsna-sub-v224r6c4pw30lomm05j (nb sha c0dd7ec8...): scored pw30 57425970 + `Lateral OA = pct(pct(LOA) - 0.05 pct(MM))` on the final output after E-OA (cell 18, +6 lines). One change. A lottery A probe: E +0.4e-4 [-1.4, +2.2], below the 1.0e-4 public read noise. Evidence: session_2026-10-20/night/NIGHT_1020.md (recal C12', skeptic KILL, NIGHT adjudication adj_c12.log). gold58 is info only.
  > - Gates: night/c12_gate.log C12_GATE PASS; commit gate night/c12_probe_check.py (pw30 params + --aloamm 0.05) must PASS 18/18.
  > - Read by R1 together with pw25: A = API best-id among pw30 / pw25 / lomm05. lomm05 is never B. No re-dose, no stacking with pw25 or any other winner, no 10-22 follow-up on this line.

## 5. Recommended 10-21 00:00 UTC slate (submit 05:31 IST; reads about 11:00-11:30 IST)

| prio | key | one change | role | expected (MODEL) | status |
|---|---|---|---|---|---|
| 1 | **r6pw20** `rsna-sub-v224r6flips6fpw20j` (sha e4d92c0a) | R6 57248084 + A21 pw block a .2 | B hedge; never A | P(silver) .851 -> .877 vs B = R6 (CLAIMED VAL_1020) | built, gated, in registry |
| 2 | **pw25** `rsna-sub-v224r6c4pw25j` (sha 64ed7eba) | pw30 with A_OA .3 -> .25 | A probe under R1 | private +0.33e-4 alone under R1 (CLAIMED) | built, gated, in registry |
| 3 | **lomm05** `rsna-sub-v224r6c4pw30lomm05j` (sha c0dd7ec8) | pw30 + LOA <- MM a .05 | A probe under R1, lottery | slate +0.83 vs +0.33 with pw25 alone (increment +0.50; +0.21..+0.54 on skeptic inputs; -0.07 OAI-only) | built, gated, gate self-tested; **not** in registry (main adds it) |
| 4-5 | empty | | | E2s / stack / C6' / C14 / pw35+ all have increments <= 0 or no better than lomm05 | |

- **GO line examples** (main writes `val/A22_GO.txt` on Eesh's word; key order = push / submit priority):
  - full: `GO r6pw20 pw25 lomm05 MAX=3 Eesh 10-21 hh:mm "<his words>"`
  - standard: `GO r6pw20 pw25 MAX=2 ...`
  - minimum: `GO r6pw20 MAX=1 ...`
- **Picks** (fixed before any score):
  - A = API best-id (A15 / R1) among pw30, pw25 and lomm05.
  - B = r6pw20 if COMPLETE and it displays >= .958. Otherwise follow the VAL_1020 fallback.
  - Eesh selects A and B by hand before 10-22 23:59 UTC.
- **Timing:**
  - Commits take about 20 min, 2 at a time. Push r6pw20 + pw25 first and lomm05 when a slot frees.
  - The GO, amendment and registry entry are due by about 04:30 IST at the latest (CLAIMED VAL_1020).
- **Do not submit:**
  - E2s (`syn_kernel_v224r6c4pw30syn20j`): skeptic KILL, slate increment -0.03.
  - The stack: two changes, no gain over lomm05.
  - C6', C14, C12 at a .1, the lmfx probes.
  - Any VAL_1020 "do not submit" entry (pw35 / pw40 / pw45 / eoa40 / LP / R6pw30).

## 6. Caveats

- **One family, again.** Every calibration rests on the E-OA family (five public reads, about 1.5 independent numbers).
  - The 3-mean's 1.0 ratio is one model out of five. It was found after the public reads were known (6 rulers looked at).
  - Its C12' read transfers only if c4's hidden LOA~MM co-firing is train-like and carries no extra label signal.
- **Option value.** Most of lomm05's slate value is R1 option value on a wide prior. It is honest only if that spread is real
  (MODEL scenario sds 1.2-1.65e-4).
- **Interaction with pw25.** Both probes touch Lateral OA. R1 will pick at most one, and the best combination (pw25 + lomm05) is
  deliberately not submitted.

## 7. Files (this pass, `session_2026-10-20/night/`)

- **Adjudication:** `adj_c12.py` -> `adj_c12.log` (RSNA raked ruler per model, E-OA calibration, placebos, C6' read).
- **Slate model:** `night_slate_mc.py` -> `night_slate_mc.log`.
- **Build (unarmed):**
  - `c12_build.py`, `c12_kernel_v224r6c4pw30lomm05j/`, `c12_kernel_v224r6c4pw30lomm00j/`,
    `c12_build_receipt_v224r6c4pw30lomm05j.json`, `c12_build_receipt_v224r6c4pw30lomm00j.json`.
  - Gates: `c12_gate.py` -> `c12_gate.log`; `c12_probe_check.py`; `c12_sim.py` -> `c12_sim.log`,
    `c12_sim/out_v224r6c4pw30lomm05j_sim/`, `c12_sim/out_v224r6c4pw30lomm05j_tamper/`.
- **This note:** `NIGHT_1020.md`.
- **Read only:** `recal.md` + `recal_*`, `synovitis.md` + `syn_*` / `box_syn*`, `lmfx.md` + `lmfx_work/`, `skc6/`, `skep_c12p*`,
  `skep_box_c12p*`, `skep_stack*`, `skc14.*`, `skp_syn*`, `val/VAL_1020.md`, `val/slate_a22.json`, `val/tick_a22.py`,
  `a21/probe_check_eoa21.py`, `a21/gate_a21_offline.py`, `a21/kernel_v224r6c4pw30j/`, `a21/out_v224r6c4pw30j_v1/`, HANDOFF top block.
