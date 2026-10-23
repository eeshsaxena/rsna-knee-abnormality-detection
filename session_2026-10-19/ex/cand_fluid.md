# cand_fluid.md: fluid-group decorrelation (Effusion, Synovitis, Baker's), 10-20 ~00:10-00:40 IST, Claude. RESEARCH + MEASUREMENT ONLY.

Nothing was built, pushed, committed or submitted. No dataset, GPU job, scheduled task, PREREG, HANDOFF, PROGRESS or a20 file
was touched. Box use: two read-only CPU python runs piped over ssh stdin (smoke BOOT 20, full BOOT 400 / RBOOT 300). They wrote
nothing on the box and printed aggregates only (n, prevalence, phi, rho, q, AUC deltas, P, ESS). No OAI id, row or per-knee value
left the box. Only gold74 aggregate cell counts went TO the box (embedded in the script by `make_fluid.py`).
Tags: MEASURED = computed tonight (file named); CLAIMED = older note says so (file named); MODEL = arithmetic on stated
assumptions; GUESS = judgement. Macro = label-sum delta / 12. E / S / B = Effusion / Synovitis / Baker's.

## 0. Verdict: KILL (both versions)

| version | crit 1 OAI (3 models, a .1) | crit 2 gold74-coupled OAI | crit 3 gold58 c4 (a .1) | report OOF | verdict |
|---|---|---|---|---|---|
| FL3: each of E, S, B minus a x mean pct of the other two | FAIL: -.00025 / -.00023 / -.00020 (P .06-.10) on B1; <= +.00007 (P <= .67) on any binarisation | FAIL: -.0004 to -.0015 (R1/R2/R2hi, every binarisation and model, P <= .04) | against: -.00047 (P>0 .30), leave-one-out 0 of 58 > 0; all 4 gold58 serves < 0 | -.00090 (P .00) | **KILL** |
| ES2: E <- S and S <- E only | nominal pass: +.00003 to +.00011 (P .92-1.00), control < 0; **uninformative** (OAI-free models' S output = E output, rho .99) | nominal pass: +.00001 to +.00011 (B3 R2hi s1045 -.00000) | **clearly against**: -.00080 (P>0 .12), leave-one-out 0 of 58 > 0; R6 -.00101, S6fall -.00090, Raptor -.00029 | -.00050 (P .00) | **KILL** |

- Best a: none. Both versions lose more at a .2 than at a .1 on every RSNA read except Raptor ES2 (a .2 +.00007, a .1 -.00029) (MEASURED, sec 3).
- Expected hidden macro at a .1 (MODEL, sec 4):
  - FL3: about -.0006 (range -.0003 to -.0010).
  - ES2: about -.0005 (range -.0001 to -.0010).
- P(hidden gain > 0) (GUESS): FL3 about .05, ES2 about .10.
- Why (MEASURED + MODEL): a small-a decorrelation t <- o gains only when the model couples t and o more than the cross-signal
  is worth, i.e. rho(f_t, f_o) > q = corr(f_o, y_t) / corr(f_t, y_t).
  - For the OA trio on OAI, q is far below rho, which is why E-OA works.
  - For the fluid labels on RSNA experts, q >= rho (sec 3.3): expert Synovitis is predicted by the Effusion score almost as
    well as by the Synovitis score (q .81 for c4, .94 for R7).
  - The Baker's pairs are on the deficit side (c4 couples Baker's to fluid LESS than gold74 experts: gaps -.07 E-B, -.12 S-B; gold58 mixed +.11 / -.02; CLAIMED coupling_map.md rows 33-34), so
    subtracting fluid from Baker's, or Baker's from fluid, moves the wrong way.

## 1. Method

**Transforms** (pct over the evaluation population; outer pct omitted because it does not change a label's AUC):

| name | definition | a values |
|---|---|---|
| FL3(a) | out_t = pct(f_t) - a x mean(pct(f_o)) over the other two fluid labels | -.1 (control), .1, .2 |
| ES2(a) | out_E = pct(f_E) - a x pct(f_S); out_S = pct(f_S) - a x pct(f_E); Baker's untouched | -.1 (control), .1, .2 |

**OAI** (box, `box_fluid_template.py` + `make_fluid.py` -> `box_fluid_run.py`, log `box_fluid.log`, tables `fluid_table.out`):
- Models: the 3 OAI-free models (D5 5-fold mean s1043, single-fold s1044 mix50, single-fold s1045 288frac75).
- pct over all 2,581 OAI knees with predictions (deployment-like). Each label is scored on its labelled knees: E 1,660, S 1,659,
  B 835. Paired bootstrap (400) over knees with at least one fluid label.
- Binarisations (Baker's = 1 in all):

  | name | Effusion (MOAKS effusion-synovitis) | Synovitis (Hoffa) | OAI prevalence E / S / B | note |
  |---|---|---|---|---|
  | B1 | grade >= 2 | grade >= 1 | .197 / .580 / .371 | the coupling map's primary |
  | B2 | grade >= 1 | grade >= 2 | .570 / .107 / .371 | |
  | B3 | grade >= 1 | grade >= 1 | .570 / .580 / .371 | closest to gold74 prevalence .649 / .493 / .269 |

**Criterion 2: coupling-matched reweighting** (skeptic method: importance-resample knees by label cell, keeping predictions
given labels as observed; bootstrap 300):
- FL3 on the 628 complete-case knees (all 3 fluid labels):

  | scheme | what it matches |
  |---|---|
  | raw_cc | complete-case knees as is |
  | R1 | gold74 8-cell E/S/B joint (67 complete, +.5 smoothed) |
  | R2 | IPF: OAI marginals + gold74 pairwise log OR (coupling only) |
  | R2hi / R2lo | R2 with gold74 log OR +1 sd / -1 sd |

- ES2 (and the E, S part of FL3) on the 1,659 E+S knees: raw, R1 (gold74 4-cell), R2, R2hi, R2lo.
- gold74 pairwise counts [c00, c01, c10, c11]: E-S [20, 5, 17, 31], E-B [20, 4, 29, 14], S-B [29, 5, 20, 13]. 3-label cells
  E/S/B: 000 16, 001 3, 010 4, 011 1, 100 13, 101 2, 110 16, 111 12.

**RSNA side** (local, `local_fluid.py` -> `local_fluid.log`):
- gold58 c4 final: m14_gate rebuild, parity macro .94102 (MEASURED). Paired bootstrap 4,000 and leave-one-out.
- R6, S6fall and Raptor on gold58 (point reads).
- R7 OOF on gold58 and gold74.
- R7 OOF on 4,333 train studies (minus gold74) vs report_labels_v2: YES/NO with UNK dropped per label, and UNK = 0.

## 2. OAI results (MEASURED, `box_fluid.log`, `fluid_table.out`)

**Coupling on OAI:**
- Model rho(ns) across the 3 models: E-S .991-.992, E-B .27-.29, S-B .28-.31.
- Expert phi:

  | binarisation | E-S | E-B | S-B |
  |---|---|---|---|
  | B1 | +.25 | -.10 | +.07 |
  | B2 | +.21 | -.02 | -.10 |
  | B3 | +.23 | -.02 | +.07 |

- gold74 expert phi for comparison: E-S +.42, E-B +.17, S-B +.28 (tetrachoric .62 / .28 / .44, CLAIMED coupling_map.md).
- So OAI experts couple the fluid labels LESS than RSNA experts. That favours decorrelation on raw OAI, which makes the raw
  OAI read a lenient test.

### 2.1 Criterion 1 (raw OAI, macro, P = bootstrap P>0; D5 / s1044 / s1045)

| transform | B1 | B2 | B3 |
|---|---|---|---|
| FL3 a -.1 | -.00013 / -.00012 / -.00026 | -.00031 / -.00038 / -.00046 | -.00023 / -.00021 / -.00027 |
| FL3 a .1 | -.00025 (.06) / -.00023 (.07) / -.00020 (.10) | -.00004 (.39) / +.00007 (.67) / +.00005 (.61) | -.00008 (.23) / -.00003 (.40) / -.00009 (.25) |
| FL3 a .2 | -.00101 / -.00101 / -.00089 (P .00) | -.00046 / -.00030 / -.00035 | -.00053 / -.00048 / -.00056 |
| ES2 a -.1 | -.00002 / -.00002 / -.00004 | -.00009 / -.00008 / -.00006 | -.00008 / -.00006 / -.00004 |
| ES2 a .1 | +.00003 (.94) / +.00003 (.92) / +.00005 (.99) | +.00011 / +.00009 / +.00006 (1.00) | +.00010 / +.00008 / +.00004 (.98-1.00) |
| ES2 a .2 | +.00006 / +.00006 / +.00009 (.90-.98) | +.00023 / +.00019 / +.00012 | +.00020 / +.00017 / +.00008 |

**FL3 per label** (D5, B1, a .1): E -.0004, S -.0023, B -.0002.
- The Synovitis term (S minus mean of E and B) is the loser on every binarisation.
- Baker's (B minus mean of E and S) is about 0 on all labelled knees.
- The only positives sit on the 628 complete-case knees under raw_cc (sec 2.2), where the OAI-specific negative E-B coupling
  helps.

**FL3 fails criterion 1:** delta <= 0 in 7 of 9 model x binarisation cells, and the 2 positive cells have P .61 / .67. Its
control (a -.1) is also negative on raw OAI, so there is no clean decorrelation signal at all.

**ES2 formally passes criterion 1 but cannot test the mechanism:**
- The OAI-free models' Synovitis output is their Effusion output (rho .99; their frac75 training labels correlate the two at
  .95, CLAIMED coupling_map.md sec 2).
- So pct(S) - a pct(E) is a near-identity rank map: the size is <= +.00011 macro.
- Theory on OAI (D5, B1; MEASURED q):

  | direction | q | rho | rho - q |
  |---|---|---|---|
  | E <- S | .994 | .992 | -.00 |
  | S <- E | .971 | .992 | +.02 |

  This matches the tiny sizes.
- c4's E-S rho is .74 on gold58, so these models cannot stand in for c4 on this pair.

### 2.2 Criterion 2 (reweighted to gold74 fluid coupling)

**FL3 on complete-case knees, macro at a .1** (P>0 in brackets; D5 / s1044 / s1045):

| scheme | B1 | B2 | B3 |
|---|---|---|---|
| raw_cc | +.00015 / -.00000 / -.00004 | +.00051 / +.00051 / +.00058 (.97-.98) | +.00023 / +.00019 / +.00012 |
| R1 (gold74 joint) | -.00114 / -.00104 / -.00101 (.00) | -.00084 / -.00082 / -.00038 (<= .04) | -.00042 / -.00043 / -.00063 (<= .02) |
| R2 (gold74 coupling) | -.00077 / -.00089 / -.00104 (.00) | -.00076 / -.00080 / -.00060 (<= .01) | -.00047 / -.00049 / -.00069 (<= .01) |
| R2hi | -.00120 / -.00132 / -.00149 | -.00122 / -.00128 / -.00107 | -.00084 / -.00087 / -.00114 |
| R2lo | -.00031 / -.00043 / -.00054 | -.00028 / -.00028 / -.00015 | -.00004 / -.00005 / -.00021 |

- At a .2, every gold74-coupled cell is -.0013 to -.0036.
- The control a -.1 turns POSITIVE under R1/R2/R2hi (for example B1 R2 +.00041 / +.00048 / +.00042, P .98-.99). So at RSNA
  expert coupling the fluid group wants borrowing, not decorrelation.
- The raw_cc positives come from OAI's negative E-B expert coupling (phi -.10 on B1), which RSNA experts do not share (gold74
  E-B phi +.17).

**ES2 on the E+S knees, macro at a .1:**

| scheme | B1 | B2 | B3 |
|---|---|---|---|
| R1 | +.00010 / +.00006 / +.00006 | +.00008 / +.00007 / +.00005 | +.00008 / +.00005 / +.00002 |
| R2 | +.00002 / +.00003 / +.00004 | +.00011 / +.00009 / +.00006 | +.00006 / +.00004 / +.00001 |
| R2hi | +.00002 / +.00002 / +.00004 | +.00011 / +.00008 / +.00005 | +.00004 / +.00002 / -.00000 |

- It survives reweighting only because the transform is a near-identity on these models (sec 2.1). This says nothing about
  c4.
- The E,S part of FL3 at a .1 on the same knees is <= 0 everywhere (for example B1 R2 -.00023 / -.00034 / -.00023). The Baker's
  term in the E and S outputs is what hurts.

## 3. RSNA side (MEASURED, `local_fluid.log`)

### 3.1 gold58 c4 final

Consistency only; gold58 is NOT a selection ruler. Base fluid AUC E / S / B: .9814 / .8399 / .9810.

| transform | macro (P>0, sd) | per label E / S / B |
|---|---|---|
| FL3 a -.1 | -.00033 (.17, .00043) | -.0012 / .0000 / -.0027 |
| FL3 a .1 | -.00047 (.30, .00091) | +.0012 / -.0060 / -.0009 |
| FL3 a .2 | -.00092 (.31, .00176) | +.0012 / -.0096 / -.0027 |
| ES2 a -.1 | +.00009 (.56, .00051) | -.0037 / +.0048 / 0 |
| ES2 a .1 | -.00080 (.12, .00073) | -.0012 / -.0084 / 0 |
| ES2 a .2 | -.00185 (.06, .00130) | -.0025 / -.0197 / 0 |

- Leave-one-out at a .1:
  - FL3: range -.00082 to -.00002, 0 of 58 > 0.
  - ES2: range -.00104 to -.00062, 0 of 58 > 0.
  - The sign does not hang on one study.
- Other gold58 serves (point reads, a .1, macro):

  | serve | FL3 a .1 | ES2 a .1 |
  |---|---|---|
  | R6 | -.00065 | -.00101 |
  | S6fall | -.00030 | -.00090 |
  | Raptor | -.00047 | -.00029 |

  All 4 serves agree in sign.
- Consistency check with the coupling map: the ES2 per-label c4 values (E -.0012, S -.0084 at a .1) equal the map's directional
  rows "Eff <- Syn" and "Syn <- Eff" (CLAIMED `merge_map.out` l.112-113). So the code agrees.

### 3.2 R7 OOF (report-trained) on expert labels and on report labels

| set | FL3 a .1 | FL3 a .2 | ES2 a .1 | ES2 a .2 | ES2 a -.1 |
|---|---|---|---|---|---|
| gold58 experts | -.00130 (P .12) | -.00256 | -.00120 (P .08) | -.00251 | +.00074 (P .80) |
| gold74 experts | -.00130 (P .06) | -.00260 | -.00083 (P .13) | -.00180 | +.00070 (P .87) |
| train vs report v2, UNK dropped (E 3,901 / S 665 / B 2,346) | -.00090 (P .00) | -.00247 | -.00050 (P .00) | -.00127 | +.00027 (P .99) |
| train vs report v2, UNK = 0 | -.00062 (P .00) | -.00183 | -.00031 (P .00) | -.00095 | +.00009 (P .96) |

- Report-label OOF LOSES, as the mechanism requires.
- But expert labels lose by as much or more (gold74 -.0013 vs report -.0009 for FL3). There is no regime contrast, which is
  the signature E-OA had and this candidate lacks.

### 3.3 Theory (q = corr(ns f_o, y_t) / corr(ns f_t, y_t); decorrelation t <- o gains only if rho > q)

| direction | c4 gold58 q / rho / rho-q | R7 gold74 q / rho / rho-q | R7 vs report q / rho / rho-q |
|---|---|---|---|
| E <- S | .76 / .76 / .00 | .83 / .72 / -.11 | .90 / .76 / -.14 |
| S <- E | .81 / .76 / -.05 | .94 / .72 / -.22 | .81 / .66 / -.15 |
| E <- B | .12 / .20 / +.08 | .38 / .26 / -.12 | .29 / .23 / -.07 |
| S <- B | .30 / .31 / +.02 | .30 / .13 / -.18 | .10 / .03 / -.07 |
| B <- E | .19 / .20 / +.01 | .28 / .25 / -.03 | .39 / .32 / -.07 |
| B <- S | .41 / .31 / -.10 | .35 / .14 / -.20 | .41 / .26 / -.15 |

- Nowhere is rho - q meaningfully positive for the fluid pairs on RSNA expert labels. The best case is c4 E <- B at +.08, on 58
  studies.
- Contrast with E-OA (CLAIMED skeptic_eoa.md / box_reweight.log): OA q sits well below OA rho on OAI.
- The coupling-map gap for E-S (c4 rho .74 vs expert tet .59-.62, gap +.12 to +.16, P .88; CLAIMED coupling_map.md row 10) is
  real but not removable. The model's effusion score carries genuine expert-synovitis signal (RSNA "synovitis" on non-contrast
  MRI is read largely through effusion-synovitis; GUESS on the rubric).

## 4. Expected hidden effect (MODEL)

**FL3 a .1:**
- OAI at gold74 coupling: -.0005 to -.0010 for R2, by binarisation; B3 (prevalence-matched) -.0005 to -.0007.
- gold58 c4: -.0005.
- R7 gold74: -.0013.
- Central about -.0006 macro, range -.0003 to -.0010. P(>0) about .05 (GUESS).

**ES2 a .1:**
- OAI cannot measure it for c4.
- RSNA reads: c4 -.0008, the other serves -.0003 to -.0010, R7 gold74 -.0008. Theory: rho - q <= 0 for c4.
- Central about -.0005, range -.0001 to -.0010. P(>0) about .10 (GUESS).

**The reverse (borrow) side:**
- The a -.1 control is positive under gold74-coupled OAI (FL3) and on R7 reads (ES2 +.0007 on gold74).
- But it is about 0 for c4 (ES2 +.00009, P .56; FL3 -.00033, P .17). R7 also gains from it on REPORT labels (+.00027, P .99),
  so it is not a regime lever.
- This is the killed E2 direction (Syn += .1 Eff; THINK_1019.md sec 8.2). Nothing here revives it (GUESS).

## 5. Files (all in `session_2026-10-19/ex/`)

| step | script | output |
|---|---|---|
| fill gold74 fluid aggregates into the box script | `make_fluid.py` (from `box_fluid_template.py`) | `box_fluid_run.py` |
| box OAI run (aggregates only, JSON at the end) | `box_fluid_run.py` | `box_fluid.log` (`box_fluid_smoke.log` = smoke) |
| compact tables | `fluid_table.py` | `fluid_table.out` |
| gold58 c4 / serves / R7 / report labels | `local_fluid.py` | `local_fluid.log` |
