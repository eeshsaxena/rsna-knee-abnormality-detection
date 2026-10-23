# cand_oa_meniscus.md: same-compartment OA <-> meniscus decorrelation on top of E-OA (10-20 ~00:30 IST, Claude)

RESEARCH + MEASUREMENT ONLY. Nothing was built, pushed, committed or submitted. No dataset, GPU job, scheduled task, PREREG,
HANDOFF, PROGRESS or a20 file was touched. The box was used for one smoke run (20 boots) and one full run (400 boots). Both were
read-only CPU python piped over ssh stdin, wrote nothing on the box and printed aggregates only (n, prevalence, phi, rho, q,
AUC deltas, P, ESS). No OAI id, row or per-knee value left the box. Only RSNA aggregate counts were sent to the box: gold74
pairwise 2x2, gold74 3-label OA cells and the gold74 5-label joint over 70 complete studies.

Tags: MEASURED = computed tonight, file named. CLAIMED = an older note says so. MODEL = arithmetic on stated assumptions.
GUESS = judgement. "Macro" means the 12-label macro AUC, i.e. the sum of single-label deltas / 12.
Models are listed in this order: D5 = teacher_D5_mean_v00 (5-fold mean), s1044 = f0_s1044_ttr_mix50, s1045 =
f0_s1045_ttr288frac75.

## 0. Verdict: KILL

The decorrelation candidate loses everywhere it can be measured.

- **OAI, criterion 1 fails.** On top of both E-OA .1 and E-OA .2, the candidate loses for all 3 models at a .1 and a .2
  (P>0 = .00). The a -.1 "control", which is a borrow, is the side that gains.
- **Criterion 2 fails.** No coupling-matched reweighting rescues it. Across 3 models, 10 scheme/label sets, 2 E-OA levels and
  a .1 / .2 (120 cells), the best cell is -.000002 (P .50). The Epartner variant's best cell is +.00005 (P .67).
- **RSNA side agrees.** The theory diagnostic (decorrelation helps only if q < rho) says decorrelation hurts in all 4 directions
  on gold58 c4, R7 gold58 / gold74 and R7 vs report labels.
  - gold58 c4 reads +.00037 at a .1 (P .61). That number is one meniscus label's noise.
  - gold58 c4 reads -.00225 at a .2.
- **Mechanism.** It is the opposite of E-OA's (MEASURED, coupling map):
  - Experts couple same-compartment OA and meniscus tear strongly. Gold74 tetrachoric: MM-MOA .67 +- .12, LM-LOA .61 +- .14.
  - Reports couple them the same (.66 / .60).
  - c4 couples them LESS than experts (rho .54 / .56 on gold58). So there is no report-inherited excess to remove.

## 1. Candidate and method

**Transform.** Let E = E-OA(e) on the OA trio, e in {.1, .2}, with E_t = pct(f_t) - e * mean(pct(f_o)). Then:
- MOA' = pct(E_MOA) - a pct(f_MM)
- LOA' = pct(E_LOA) - a pct(f_LM)
- MM' = pct(f_MM) - a pct(f_MOA)
- LM' = pct(f_LM) - a pct(f_LOA)

Values of a:
- a in {.1, .2} is the candidate.
- a -.1 is the negative control. a -.2 was also run.
- Sensitivity variant "Epartner": the meniscus side uses the E-OA'd partner pct(E_MOA) instead.

**Increment.** The increment is measured over E-OA(e) alone, summed over the 4 touched labels, and expressed as macro (/12).
- The partners are raw percentiles, so the columns are independent. BOTH therefore equals OAside + MENside exactly within each
  bootstrap draw.
- Splits reported: OA side (MOA, LOA), meniscus side (MM, LM), medial pair, lateral pair and per label.

**OAI labels** (box `~/oai_build`):
- B1 (primary), n 2,399: OA = P_FT, the MOAKS full-thickness proxy (as `oai_eoa_check.py`); menisci = tear (T123 = 1).
  Prevalence MOA .216, LOA .195, PF .434, MM .411, LM .206.
- B2 (sensitivity): OA = T123 composite >= .5; menisci >= .2 (includes intrasubstance signal).

**Coupling-matched reweighting** (skeptic method extended; knees are importance-resampled by label cell). The weighted phi
columns are MEASURED.

| scheme | what it matches | ESS | phi MOA-MM | phi LOA-LM |
|---|---|---|---|---|
| raw | OAI as is | 2,399 | .361 | .355 |
| R1oa3 | skeptic: 3-label OA cell to gold74 | 1,794 | .357 | .280 |
| R1m5 | 5-label cell to the gold74 joint (70 studies, +.5 smoothing, noisy) | 745 | .346 | .305 |
| R2m | IPF: OAI marginals + gold74 log OR on all 10 pairs | 1,644 | .409 | .370 |
| R2m_hi / R2m_lo | R2m with every log OR +1 / -1 sd | 1,060 / 2,019 | .487 / .304 | .513 / .240 |
| R2m_schi / R2m_sclo | R2m with only MOA-MM and LOA-LM shifted +1 / -1 sd (the candidate's key unknown) | 1,491 / 1,684 | .484 / .307 | .496 / .231 |

- Paired bootstrap: 400 draws per model and scheme (box). Locally, 2,000 draws on gold58 / gold74 and 300 on report labels.
- **Parity checks** (MEASURED):
  - E-OA-alone values reproduce the coupling map. R1oa3 E-OA .1 is +.00042 / +.00004 / +.00009 (map: +.00041 / +.00003 /
    +.00008). R2m_hi E-OA .2 is -.00019 / -.00103 / -.00087 (map: -.00018 / -.00105 / -.00087).
  - The gold58 c4 rebuild gives 12-label macro .94102, matching m14_gate.log.

## 2. Criterion 1: OAI, 3 models (MEASURED, `cand_oam_box.log`, `cand_oam_summary.out`)

Values are increments over E-OA(e), macro, D5 / s1044 / s1045. P = bootstrap P>0, shown as the range over the 3 models.

| labels | e | a .1 (candidate) | a .2 (candidate) | a -.1 (control) |
|---|---|---|---|---|
| B1 raw | .1 | -.00099 / -.00126 / -.00062 (P .00) | -.00314 / -.00378 / -.00221 (P .00) | +.00027 / +.00046 / +.00000 (P .50-1.00) |
| B1 raw | .2 | -.00075 / -.00105 / -.00042 (P .00) | -.00258 / -.00327 / -.00172 (P .00) | +.00007 / +.00030 / -.00014 (P .10-.98) |
| B2 raw | .1 | -.00120 / -.00136 / -.00086 (P .00) | -.00341 / -.00381 / -.00263 (P .00) | +.00052 / +.00064 / +.00025 (P .97-1.00) |

**Where the loss comes from** (B1 raw, e .1, a .1; per-label AUC, not macro):

| label | change |
|---|---|
| MOA | -.0040 / -.0038 / -.0015 |
| LOA | -.0075 / -.0076 / -.0052 |
| MM | +.0002 / -.0026 / -.0003 |
| LM | -.0005 / -.0012 / -.0004 |

- OA side: -.00096 / -.00094 / -.00056.
- Meniscus side: -.00003 / -.00032 / -.00005.
- Medial pair: -.00032 / -.00053 / -.00015 (P <= .04).
- Lateral pair: -.00067 / -.00073 / -.00046 (P .00).
- So neither compartment alone survives.
- The only positive piece: the B2 meniscus side (MM' = MM - .1 MOA, with "meniscus" including intrasubstance signal) gains
  +.00046 / +.00030 / +.00037 raw (P 1.00). That gain vanishes under R2m (-.00002 / -.00018 / -.00010).

## 3. Criterion 2: coupling-matched reweighting (MEASURED)

BOTH increment over E-OA .1, macro, D5 / s1044 / s1045:

| scheme | a .1 | a .2 | a -.1 (control) |
|---|---|---|---|
| R1oa3 | -.00039 / -.00082 / -.00012 (P .00-.17) | -.00189 / -.00281 / -.00113 | -.00029 / +.00006 / -.00046 |
| R1m5 | -.00082 / -.00111 / -.00065 | -.00274 / -.00352 / -.00223 | +.00001 / +.00036 / -.00004 |
| R2m | -.00098 / -.00132 / -.00072 | -.00311 / -.00388 / -.00239 | +.00023 / +.00052 / +.00006 |
| R2m_hi | -.00129 / -.00171 / -.00108 | -.00383 / -.00472 / -.00316 | +.00049 / +.00082 / +.00034 |
| R2m_lo | -.00059 / -.00087 / -.00030 (P .00-.01) | -.00222 / -.00287 / -.00148 | -.00010 / +.00014 / -.00026 |
| R2m_schi | -.00124 / -.00164 / -.00098 | -.00375 / -.00459 / -.00300 | +.00046 / +.00075 / +.00027 |
| R2m_sclo | -.00063 / -.00093 / -.00037 | -.00228 / -.00296 / -.00161 | -.00008 / +.00020 / -.00022 |
| B2 R2m | -.00105 / -.00127 / -.00077 | -.00311 / -.00362 / -.00241 | +.00040 / +.00055 / +.00017 |

- On top of E-OA .2 the pattern is the same and slightly less negative. For example, R2m a .1 is -.00082 / -.00120 / -.00057.
  The best cell is R1oa3 s1045, -.000002 (P .50).
- Even the friendliest scenario, R2m_sclo with less same-compartment expert coupling, stays negative for all 3 models.
- The "Epartner" variant has the same sign in every a > 0 cell, with differences <= .0002. Its best cell is +.00005
  (P .67), at R1oa3 s1045, e .2, a .1.

## 4. Criterion 3 and the regime checks (MEASURED, `cand_oam_local.log`, local CPU)

BOTH increment over E-OA(e), macro:

| set | e | a .1 | a .2 | a -.1 |
|---|---|---|---|---|
| gold58 c4 final (n 58) | .1 | +.00037 (P .61) | -.00225 (P .16) | -.00017 (P .44) |
| gold58 c4 final | .2 | +.00029 (P .61) | -.00179 (P .21) | -.00044 (P .32) |
| gold58 R7 OOF | .1 | -.00109 (P .16) | -.00318 (P .07) | +.00006 (P .53) |
| gold74 R7 OOF (n 74) | .1 | -.00158 (P .07) | -.00467 (P .02) | +.00016 (P .53) |
| train R7 OOF vs report v2 (n 2,574, all 5 YES/NO) | .1 | -.00097 (P .00) | -.00315 (P .00) | +.00010 (P .75) |
| train R7 vs report v2 (UNK dropped per label, n 4,333) | .1 | -.00097 (P .00) | -.00303 (P .00) | +.00015 (P .94) |

- **gold58 c4 +.00037 at a .1** is entirely MM +.0072, about 1 to 2 rank pairs on a .977 base. The other three labels are
  MOA -.0016, LOA .0000 and LM -.0012. It flips to -.00225 at a .2. This is consistency noise, not support. gold58 is not a
  selection ruler.
- **The report-label OOF loses, but that is not the mechanism signature here.**
  - Reports do not over-couple these pairs: report tet .66 / .60 vs gold74 .67 / .61 (CLAIMED, coupling map).
  - R7 under-couples relative to its own report labels (rho .59 / .59). So decorrelation loses on every label regime, report
    and expert alike.
  - The report check is therefore non-discriminating for this candidate.

## 5. Why it fails (MODEL, with MEASURED inputs, `cand_oam_box.log` diag lines, `cand_oam_theory_local.log`)

Decorrelating t <- o helps only if q = corr(ns f_o, y_t) / corr(ns f_t, y_t) < rho = corr(ns f_t, ns f_o). This is the same
criterion the skeptic used for E-OA. In every set below, q > rho or q is about equal to rho:

| set | MOA <- MM | LOA <- LM | MM <- MOA | LM <- LOA |
|---|---|---|---|---|
| OAI B1, 3 models (rho / q) | .57-.67 / .68-.73 | .73-.76 / .99-1.03 | .57-.67 / .65-.72 | .73-.76 / .70-.76 |
| gold58 c4, rho - q (P>0) | -.18 (.05) | -.26 (.07) | -.00 (.47) | -.04 (.33) |
| gold74 R7, rho - q (P>0) | -.28 (.00) | -.26 (.03) | -.08 (.18) | -.10 (.17) |
| train R7 vs report, rho - q (P>0) | -.08 (.00) | -.04 (.03) | -.10 (.00) | -.14 (.00) |

- The model's meniscus head carries as much information about the same-compartment OA label as the OA head does. On OAI
  LOA <- LM, q is about 1. Subtracting it therefore removes signal, not report-inherited noise.
- E-OA works because OA-trio q < rho. Experts barely couple the three compartments, while the models couple them at .7-.8.
- Here, experts couple the same-compartment pairs at .6-.7 and c4 couples them at .54-.56. The sign is reversed.

## 6. Side findings (not the candidate; for the orchestrator)

### 6a. Lead (post hoc): lateral borrow LOA += b pct(f_LM)

This is the reverse sign of the candidate, on the lateral side only. The numbers below are for a -.1 on top of E-OA .1
(MEASURED). The LOA single-label AUC delta:

| scheme | LOA delta |
|---|---|
| raw | +.0054 / +.0054 / +.0034 |
| R2m | +.0026 / +.0030 / +.0011 |
| R1oa3 | +.0019 / +.0024 / +.0003 |
| R1m5 | +.0022 / +.0030 / +.0017 |
| R2m_hi | +.0024 / +.0030 / +.0009 |
| R2m_sclo | +.0016 / +.0017 / +.0002 |
| B2 R2m | +.0042 / +.0048 / +.0026 |

- RSNA-side checks:
  - gold58 c4: LOA +.0039.
  - gold74 R7: +.0038.
  - gold58 R7: .0000.
  - R7 vs report: -.0009.
- Theory: LOA <- LM has q > rho on every RSNA set (sec 5). Borrowing helps only if q > rho, so this direction is predicted to
  help. The measured R7-vs-report value (-.0009) does not follow the theory there.
- Size: about +.0001 to +.0002 macro at gold74 coupling (MODEL), with s1045 near 0.
- The bidirectional lateral pair (adding LM += .1 pct LOA) is +.00030 / +.00036 / +.00022 under R2m but -.00005 / +.00006 /
  -.00014 under R1oa3. R1oa3 lets LOA-LM phi drift to .28, below gold74.
- Caveats:
  - Selected post hoc: 2 compartments x 2 sides x 4 a x 2 e.
  - It overlaps the coupling map's INCONCLUSIVE "BOR" result, where the medial half was what failed for s1045.
- GUESS: inside LB noise and not worth a slot on its own. If anything is ever stacked on E-OA, test LOA <- LM borrow at
  a -.1 alongside the map's LOA <- MM .1 (both act on LOA).

### 6b. E-OA's own transfer also depends on same-compartment expert coupling

These are E-OA-alone numbers (MEASURED). Only the MOA-MM and LOA-LM log ORs were moved, by +-1 sd:

| scheme | E-OA .1 | E-OA .2 |
|---|---|---|
| R2m | +.00035 / +.00002 / +.00008 | +.00034 / -.00046 / -.00033 |
| R2m_schi | +.00053 / +.00019 / +.00026 | +.00068 / -.00013 / +.00002 |
| R2m_sclo | +.00014 / -.00018 / -.00013 | -.00005 / -.00086 / -.00073 |

- The OA-trio phi is essentially unchanged across these schemes: MOA-LOA .35-.36, MOA-PF .35, LOA-PF .22.
- The between-scheme differences, about .0002, are roughly 1.5 to 2 bootstrap sd. Schemes are not paired, so treat this as
  MODEL-grade.
- Reading: less same-compartment coupling on the hidden set hurts E-OA, and hurts a .2 more than a .1. This is one more reason,
  besides the coupling map's sec 4, that eoa10 is the safer shape (GUESS).

## 7. Files (all in `session_2026-10-19/ex/`)

| file | what it is |
|---|---|
| `cand_oam_box_template.py` + `make_cand_oam.py` -> `cand_oam_box_run.py` | box script; fills in RSNA gold74 aggregates; piped via `ssh ... python -I -` |
| `cand_oam_box.log` | aggregates + JSON |
| `cand_oam_summary.py` -> `cand_oam_summary.out` | all tables: BOTH, OAside, MENside, medial, lateral, Epartner, per label, B1 and B2 |
| `cand_oam_local.py` -> `cand_oam_local.log` | gold58 c4, R7 gold58 / gold74, R7 vs report v2 |
| `cand_oam_theory_local.py` -> `cand_oam_theory_local.log` | rho vs q on the RSNA sets |
