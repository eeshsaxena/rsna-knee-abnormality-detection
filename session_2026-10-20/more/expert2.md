# expert2.md: expert regime II, beyond coupling (10-20 ~01:15-03:00 IST, Claude). RESEARCH + MEASUREMENT ONLY.

Nothing was built, pushed, committed or submitted. No dataset, no scheduled task, no PREREG / HANDOFF / PROGRESS edit, and the
session_2026-10-19/a20 folder was not touched (other 10-19 folders were only read). No GPU was used (V100 idle at 0 %, 5 MiB when
checked). Box use: four read-only CPU python runs piped over ssh stdin (`box_inspect.py`, `box_sev.py`, `box_mdis.py`,
`box_eoa_ladder.py`). They wrote nothing on the box and printed aggregates only (column names, grade histograms, n,
prevalence, AUC, AUC deltas, sd, P, tetrachoric, Spearman). No OAI id, row or per-knee value left the box.
Local work: three CPU scripts on existing local files (`local_sev.py`, `local_acute.py`, `local_bone.py`), each < 1 min.
Tags: MEASURED = computed tonight, file named; CLAIMED = an older note or the host says so, file named; MODEL = arithmetic on
stated assumptions; GUESS = judgement. Macro = 12-label macro AUC (single-label delta / 12). gold58 is used only as a code and
consistency check, never to select.

## 1. Verdict (plain language)

1. **No new fix passes.** I tested four expert-vs-report mechanisms that go beyond OA coupling. Three are dead on the
   rulers we have. The fourth (MCL decorrelation) has a mechanism but no independent ruler. None is worth a slot.
2. **The main lens (severity thresholds x own-label member disagreement) is KILLED on an independent expert set.** On 2,399 OAI
   knees with MOAKS grades and OAI-free members, AND-like combiners (penalise member disagreement) do not gain at the
   rubric-like cut-off for OA, menisci or effusion. They also do not gain more as the cut-off gets stricter. The rubric-directed
   version is worth about -.0001 to -.0002 macro (MEASURED `box_sev.log`).
3. **Synovitis and Fracture, two of the five weakest labels, have no independent expert ruler at all.** OAI has no
   fracture or contusion grades. Its "Synovitis" is Hoffa synovitis, and the OAI-free models' Synovitis output is their
   Effusion output (rho .99, CLAIMED ex/coupling_map.md). Every idea for these labels rests on gold58 alone, and gold58
   mis-called PF OA .1.
4. **A new coupling family turned up for the acute-trauma labels (MCL, Fracture).** It has the largest expert-vs-model gap of
   all 66 pairs (MCL-MM +.59 on gold58, +.56 on gold74), and the report-OOF control loses as the mechanism predicts. But it
   fails the specificity null: the rubric-chosen degenerative helper set ranks 175 of 252 random helper sets on gold58. Its
   split-half sign agreement is .38, and no OAI ruler exists for MCL or Fracture. Verdict: DEAD for a slot (sec 3.2).
5. **The in-flight E-OA entries get one more robustness check, not a new idea.** On raw OAI the E-OA gain is positive at all
   5 MOAKS cartilage definitions, for all 3 OAI-free models, with P 1.00. The run reproduces the skeptic's P_FT row exactly
   (`box_eoa_ladder.log`). The size question (transfer to RSNA coupling) is unchanged.
6. **Side finding for E1 (soft-OR pow2, THINK sec 8.4).** On the OAI expert grades E1 is about 0 macro (+.00000 for 3 families,
   -.00004 for 7 members). So the only independent expert ruler does not show E1's report-OOF gain (+.0004 to +.0011). That
   is one more reason not to spend a slot on E1.
7. **Recommendation: no new build.** Leave the 3 + 5 + 5 remaining slots unused, as EX_1019 sec 3.4 says. Picks follow A20:
   A = API best-id after the eoa reads, B = R6 57248084.

## 2. Label by label: what the expert does differently, and whether a post-hoc rank fix can use it

AUC is unchanged by any monotone transform of one label's score. So a post-hoc rank fix needs information the label's own
score does not hold: (i) other labels (coupling), (ii) member disagreement (operator), (iii) R6 / R7 state heads (severity
sub-scores), or (iv) metadata (closed). The rubric is from forum 733343 and the "same process for test" ruling from 733491
(CLAIMED, host). Regime numbers are MEASURED in trick/regime_work/m1_regime.log (expert vs report prevalence on the same 58).

| label | c4 g58 | expert vs report (m1) | what the expert does differently (rubric, paraphrased) | lever it suggests | independent ruler | result tonight |
|---|---|---|---|---|---|---|
| Synovitis | .840 | expert BROADER (.47 vs .28-.29) | inflamed / thickened synovium seen on images; the host never said whether MOAKS-style effusion-synovitis counts (737155 unanswered) | Syn <- Eff (E2), OR-combiner for a broad label | none: OAI Syn = Hoffa, OAI-free Syn = Eff (rho .99) | E2 stays KILLED; gold58 Part B goes the WRONG way for a broad label (AND +.0038, OR -.0014): DEAD |
| Lateral OA | .843 | expert stricter + compartment-specific | high-grade loss >50 % depth over ~1 cm, in that compartment only; facet chondrosis is not "lateral compartment" | E-OA (in flight); severity AND | OAI MOAKS cartilage | AND at FT: -.0001 / -.0004 (P .03 for 7 members); trend reversed (AND best at the lenient level): DEAD. E-OA robust across definitions |
| PF OA | .901 | expert stricter | same high-grade rule, patella + trochlea | E-OA; severity AND | OAI MOAKS | AND at FT: -.0005 / -.0003: DEAD |
| Lateral Meniscus | .928 | ~same (.40 vs .38-.52) | definite surface contact on >= 2 images, or a morphologic (truncated / macerated) meniscus; degeneration not reaching the surface is negative | severity AND (signal vs tear vs complex) | OAI MOAKS morphology | AND at tear: -.0004 / -.0003 (P .05-.10), flat across levels: DEAD |
| Fracture | .936 | ~same prevalence, poor agreement (sens .78, spec .82) | ACUTE cortical break or fracture line only | decorrelate from degenerative labels (insufficiency / old fractures in reports); state-head "old_or_healed" removal | none (OAI has no fracture grade) | acute <- DEG fails the null (sec 3.2); state rubric map was negative on Fracture (CLAIMED m2): DEAD |
| Contusion | .956 | expert stricter (impact edema only) | edema from impact WITHOUT a discrete fracture line; degenerative edema negative | Cont <-> Frac decorrelation | none | q ~ rho, gold58 ~0: DEAD (sec 3.3). c4 already anti-couples Contusion with OA as the experts do |
| MCL | .968 | expert stricter (.16 vs .28-.31) | high-grade ACUTE tear only; low-grade sprains and chronic / remote changes are negative | decorrelate from degenerative labels | none (OAI has no MCL grade) | largest gap of all 66 pairs, but non-specific on gold58: WEAK / DEAD (sec 3.2) |
| ACL | .989 | expert stricter (>50 % fibres) | high-grade partial or complete | severity AND | OAI m_acl_tear | AND +.0005 (3 families, P .96) at "complete", i.e. +.00004 macro: too small |
| Effusion | .981 | expert MUCH stricter (moderate+) | moderate or large only | severity AND | OAI m_eff_wk | AND at grade 2+: -.0001: DEAD. R6's state stacker already encodes severity |
| Baker's | .981 | expert stricter (moderate+) | moderate or large, typical location | severity AND | OAI popcys (n 600) | +.0002 / -.0001, P ~.5: noise |
| Medial Meniscus | .977 | ~same | as LM | severity AND | OAI MOAKS | AND at tear -.0003 / -.0001; it gains only at the complex / maceration level, which is beyond the rubric: DEAD |
| Medial OA | .989 | expert stricter | as LOA | severity AND | OAI MOAKS | AND at FT -.0006 / -.0008 (P .00-.04): DEAD |

## 3. Measurements

### 3.1 Severity threshold x own-label member disagreement (the brief's main example). KILL.
- **Mechanism tested.** The rubric grades "on the fence" findings negative and uses strict cut-offs. If expert positives are
  the cases every member agrees on, an AND-like combiner should help at strict cut-offs, and more so the stricter the cut-off.
  An OR-like combiner (the E1 family) should help at lenient, report-like cut-offs.
- **Ruler (independent).** OAI V00, MOAKS grades read on the box (`box_sev.py`). Members are OAI-free: the 3 families
  (teacher D5 mean, s1044, s1045) and 7 members (the 5 s1043 folds + s1044 + s1045). Ladders run lenient to strict:
  - ACL: partial+, complete.
  - Menisci: signal+, tear, complex / maceration.
  - OA: any area, area 2+, FT (rubric-like), area 2 + FT, FT 2+.
  - Effusion: grade 1+, 2+ (rubric), 3.
  - Hoffa synovitis: 1+, 2+.
  - Baker's: popcys, WORMS 1+, 2+.
- **Operators.**
  - AND family: geometric rank mean, mean of sqrt(u), mean - .5 sd.
  - OR family: pow2, invgeo, mean + .5 sd.
  - Also min and max.
- **Pre-registered reading** (in the script header, before any number).
  - SUPPORT: in >= 3 of 5 strict families, AND > 0 with P >= .90 at the rubric level AND above its lenient-level delta.
  - KILL: AND <= 0 or P < .5 in >= 3 families.
- **Power check** (MEASURED `box_mdis.log`, `local_sev.log`). The mean per-study member spread on OAI is .040-.068 (3 families)
  and .052-.104 (7 members), vs .050-.088 for c4's families on gold58. So OAI has c4-like member disagreement, and the null is
  informative.

Rubric-level single-label AUC deltas, 3 families / 7 members (MEASURED `box_sev.log`; P for the geo member of the family):

| label (rubric level) | n, prev | base AUC | AND family | OR family (pow2 P) | min | read |
|---|---|---|---|---|---|---|
| ACL (complete) | 1,660, .035 | .970 | +.0005 / +.0003 (geo P .96 / .79) | -.0011 / -.0006 | +.0014 / -.0016 | only pass; +.00004 macro |
| Medial Meniscus (tear) | 2,397, .41 | .936 | -.0003 / -.0001 | -.0000 / -.0001 | -.0016 / -.0015 | KILL |
| Lateral Meniscus (tear) | 2,395, .21 | .932 | -.0004 / -.0003 (P .05 / .10) | +.0001 / +.0001 | -.0028 / -.0028 | KILL |
| Medial OA (FT) | 2,398, .22 | .878 | -.0006 / -.0008 (P .04 / .00) | +.0005 / +.0007 (P .94 / 1.00) | -.0041 / -.0065 | KILL |
| Lateral OA (FT) | 2,397, .20 | .781 | -.0001 / -.0004 (P .64 / .03) | +.0000 / +.0003 | -.0012 / -.0054 | KILL |
| PF OA (FT) | 2,396, .43 | .813 | -.0005 / -.0003 (P .04 / .10) | +.0003 / +.0003 | -.0033 / -.0033 | KILL |
| Effusion (grade 2+) | 1,660, .20 | .957 | -.0001 / -.0001 | +.0000 / +.0001 | -.0010 / -.0022 | KILL |
| Baker's (popcys) | 600, .42 | .919 | +.0002 / -.0001 (P .61 / .43) | -.0006 / -.0005 | -.0010 / -.0063 | noise |
| Synovitis (Hoffa 1+, broad) | 1,659, .58 | .715 | -.0002 / +.0008 | +.0000 / -.0008 | -.0013 / +.0027 | opposite of the broad-label prediction for 7 members |

- **Trend with strictness** (MEASURED, `box_sev.log` TREND block). It is not monotone:
  - Medial OA: AND -.0007 / -.0010 / -.0006 / -.0005 / -.0002 (3 families). It rises toward 0 but never crosses it.
  - Lateral OA: +.0006 / +.0003 / -.0001 / -.0003 / -.0004. This is reversed: AND helps only at the most lenient level.
  - Menisci: AND turns positive only at complex / maceration (+.0005 / +.0006), past the rubric's "definite tear".
  - Effusion: flat.
- **Pre-registered read: KILL** (OA, menisci and Effusion are all <= 0 at the rubric level; 1 of 5 families passes).
- **Rubric-directed composite** (AND on the 8 strict labels, OR on Synovitis), OAI macro-equivalent: -.00011 (3 families) and
  -.00022 (7 members) (MODEL arithmetic on MEASURED deltas).
- **Report-OOF negative control** (R7 + s1044 + s1045, n 3,467; MEASURED `local_sev.log` part A):
  - AND loses on v2 (-.00143 macro) and on v4 (-.00090). OR gains +.00018 / +.00027.
  - This direction is what the report regime predicts. But the expert ruler does not reverse it at the rubric cut-offs, so
    there is no regime flip to exploit. (Contrast E-OA: report loses, expert gains.)
- **gold58 c4** (consistency only; 3 families with c4's effective weights, flat; MEASURED part B):
  - macro dAND -.00022, dOR +.00026.
  - STRICT-label mean dAND -.0002, dOR -.0002.
  - geo was -.0017 vs the flat rank on gold58 (CLAIMED think/stack_ops.json).
- **Verdict: DEAD.** Expected hidden macro about -.0001 (MODEL), P(>0) about .25 (GUESS).

### 3.2 Acute-trauma labels (MCL, Fracture) decorrelated from degenerative labels. DEAD (WEAK for MCL alone).
- **Where it came from.** The 10-19 coupling map covered only the 9 OAI labels. Part C of `local_sev.log` adds the 30 pairs
  that involve MCL, Contusion and Fracture (gap = c4 Spearman minus expert tetrachoric; MEASURED):

  | pair | gold58 tet | gold74 tet | c4 rho g58 | gap g58 / g74 | R7 train rho | report tet | c4 decor a .1 (label AUC) | R7 vs report a .1 |
  |---|---|---|---|---|---|---|---|---|
  | MCL - Medial Meniscus | -.35 | -.31 | +.25 | **+.59 / +.56** | +.33 | +.31 | +.009 | -.008 |
  | MCL - Baker's | -.47 | -.30 | +.09 | +.56 / +.39 | +.04 | +.16 | -.003 | -.007 |
  | MCL - Contusion | +.21 | +.30 | +.60 | +.40 / +.30 | +.50 | +.43 | +.007 | -.002 |
  | MCL - Effusion | -.09 | -.03 | +.29 | +.38 / +.33 | +.32 | +.33 | +.005 | -.006 |
  | MCL - Medial OA | -.54 | -.39 | -.16 | +.38 / +.23 | +.17 | +.25 | +.002 | -.009 |
  | Fracture - Baker's | -.27 | -.32 | +.13 | +.40 / +.46 | +.15 | +.04 | +.004 | -.002 |
  | Fracture - Contusion | +.49 | +.40 | +.83 | +.34 / +.44 | +.65 | +.83 | +.003 | -.002 |
  | Fracture - Medial OA | -.39 | -.40 | -.22 | +.16 / +.17 | +.18 | +.17 | +.011 | -.003 |
  | Contusion - OA trio | -.57 / -.45 / -.47 | -.49 / -.39 / -.44 | -.51 / -.45 / -.50 | about 0 | ~0 | +.20 to +.33 | about 0 | -.013 to -.014 |

- **Mechanism.** The rubric keeps only ACUTE MCL tears and ACUTE fractures; it grades chronic / remote MCL change and low-grade
  sprains negative. Reports also name chronic and degenerative variants, so the report-trained heads couple MCL and Fracture
  positively with degenerative findings. Experts couple them negatively. c4 already anti-couples Contusion with OA as the
  experts do, so only MCL and Fracture carry the excess.
- **Pre-specified transform** (from the rubric, AFTER seeing the table, so it is post hoc).
  - out_t = pct(pct(f_t) - a * mean pct(DEG)) for t in {MCL, Fracture}.
  - DEG = {Medial Meniscus, Medial OA, Lateral OA, PF OA, Baker's}.
- **Results** (MEASURED `local_acute.log`):
  - gold58 c4: a .1 +.00035 macro (MCL +.0000, Fracture +.0042), boot P .68, split-half same sign .38. a .2 +.00081 (P .74,
    split-half .49).
  - **Specificity null fails.** Over all 252 five-label helper sets drawn from the 10 non-acute labels, DEG ranks 175 of 252.
    The null mean is +.00048, which is ABOVE DEG. So on gold58, subtracting almost any other labels from MCL / Fracture
    "helps". That is generic noise or a generic effect, not the degenerative mechanism.
  - Theory on gold58: rho(f_t, DEG) -.03 / -.07 vs q -.31 / -.32, P(q < rho) .975 / .914. But q is estimated on the same 58
    (9 MCL positives), so it is not independent.
  - Report-OOF control: it loses for R7, s1044, s1045 and their mean, on v2 and v4: -.0003 to -.0006 macro at a .1 and -.0009
    to -.0015 at a .2. This is the predicted regime fingerprint, but every decorrelation of coupled report labels loses there,
    so it does not discriminate.
- **Independent ruler: none.**
  - OAI has no MCL, contusion or fracture grade (T123 MCL / Contusion / Fracture all NaN; MEASURED `box_inspect.log`).
  - gold74's extra 16 are our own report reads under the gold convention, not expert reads (CLAIMED trick/label_regime.md
    sec 1), and they hold too few MCL positives.
  - There is no LB history.
- **MCL <- Medial Meniscus alone.** It is the single largest gap, with +.009 MCL AUC (+.0008 macro) on gold58. It was picked from
  30 pairs and rests on 9 positives. That is WEAK and post hoc. Expected hidden about +.0001 +- .0004 (GUESS).
- **Verdict.**
  - Family: DEAD. Expected hidden about 0 +- .0004 (GUESS), P(>0) about .5.
  - MCL <- MM alone: WEAK. Not eligible under the "independent ruler" rule.

### 3.3 Contusion <-> Fracture (rubric: contusion "without a discrete fracture line"). DEAD.
- **Coupling gap.** It is large: c4 rho .83 vs expert tet .49 / .40, report tet .83.
- **Theory says there is no room** (MEASURED `local_bone.log`). Contusion <- Fracture: rho .771, q .732 (gap .04). Fracture <-
  Contusion: rho .771, q .823 (q > rho).
  - The Fracture score predicts expert Contusion almost as well as the Contusion score does: 10 of 19 expert contusions have a
    fracture on gold58.
- **gold58 c4, symmetric decor.**
  - a .1: Cont -.0027, Frac +.0028, macro +.00001 (P .50).
  - a .2: +.00022 (P .61).
- **Report OOF.** It loses at both a (Cont -.0072 / -.0183 on v2).
- This matches the 10-19 "bone" within-group null (CLAIMED trick/regime_work/m11).
- **Expected hidden.** About 0 (MODEL).

### 3.4 E-OA robustness across MOAKS cartilage definitions (in-flight entries; NOT a new entry)
MEASURED `box_eoa_ladder.log`: raw OAI OA-mean AUC delta, 2,393 knees with every OA subregion read.

| definition | prev M / L / PF | expert tet ML / MPF / LPF | D5 a .1 / .2 | s1044 a .1 / .2 | s1045 a .1 / .2 |
|---|---|---|---|---|---|
| any area | .61 / .52 / .85 | +.10 / +.27 / +.25 | +.0051 / +.0097 | +.0038 / +.0064 | +.0042 / +.0072 |
| area 2+ | .47 / .38 / .74 | +.06 / +.20 / +.26 | +.0061 / +.0117 | +.0050 / +.0090 | +.0052 / +.0093 |
| FT (skeptic P_FT) | .22 / .20 / .43 | +.12 / +.13 / -.01 | +.0062 / +.0115 | +.0049 / +.0085 | +.0053 / +.0092 |
| area 2 + FT (closest to "moderate area, high grade") | .19 / .15 / .40 | +.04 / +.13 / +.00 | +.0066 / +.0120 | +.0053 / +.0090 | +.0056 / +.0094 |
| FT 2+ | .14 / .12 / .31 | +.10 / +.14 / +.05 | +.0055 / +.0095 | +.0045 / +.0069 | +.0043 / +.0063 |

- Every cell is P 1.00.
- The FT row reproduces the skeptic's R0 row exactly (+.0062 / +.0115; CLAIMED a20/skeptic_eoa.md sec 1). That is a parity
  check of this script.
- The sign does not depend on the cartilage definition.
- No MOAKS definition reaches RSNA's trio coupling (OAI tet <= .27 vs gold74 .41-.61). So the 5-7x transfer shrink in
  skeptic_eoa.md stays the operative size estimate: a .1 about +.0005, a .2 about +.0006 to +.0007 (CLAIMED). No change to
  the eoa read plan.

### 3.5 E1 (soft-OR pow2) on the independent expert ruler
- **Rubric-level single-label pow2 deltas** (MEASURED `box_sev.log`), 3 families / 7 members:

  | label | 3 families | 7 members |
  |---|---|---|
  | ACL | -.0002 | -.0002 |
  | MM | .0000 | .0000 |
  | LM | +.0002 | .0000 |
  | MOA | +.0002 | +.0004 |
  | LOA | -.0001 | +.0001 |
  | PF | +.0001 | +.0004 |
  | Eff | .0000 | -.0000 |
  | Syn (Hoffa 1+) | -.0000 | -.0008 |
  | Bak | -.0002 | -.0004 |

- **Macro-equivalent:** +.00000 / -.00004.
- **Comparison:** the report-OOF ruler says +.0004 to +.0011 (CLAIMED THINK sec 8.1), and gold58 kernel-natural says +.00009
  (CLAIMED).
- **Caveat:** these OAI members are one recipe family. E1's OOF gain was measured across families (R7 vs the S6 seeds), so
  this is moderate evidence, not a kill.
- **Effect on E1's case:** it adds an independent expert-regime null to the gold58 null. E1's prior moves toward 0 (MODEL), and
  "no E1 slot" holds.

## 4. Idea list (also returned as structured output)

| # | idea | ruler | measured | expected hidden macro | P(>0) | cost | verdict |
|---|---|---|---|---|---|---|---|
| 1 | Rubric-directed member combiner: AND (penalise own-label member disagreement) on strict labels, OR on Synovitis | OAI MOAKS ladders, OAI-free members (independent); report OOF control; gold58 check | OAI rubric level AND <= 0 for OA / menisci / Effusion (-.0001 to -.0008 per label), trend not monotone; composite -.00011 / -.00022; report OOF AND -.0009 / -.0014; gold58 dAND -.0002 | about -.0001 (MODEL) | about .25 (GUESS) | 0 runtime; cells 14 + 18 edit + gate ~1.5 h; ~13 h to a read | DEAD |
| 2 | E1 soft-OR re-checked on OAI expert grades (not new, ruler result) | OAI MOAKS (independent) | +.00000 / -.00004 macro-equivalent at rubric levels | 0 to +.00005 (MODEL) | about .5 | 0 runtime; one slot | DEAD as a slot use |
| 3 | Acute labels (MCL, Fracture) minus a x mean(MM, MOA, LOA, PF, Baker's) | gold58 (post hoc); report OOF control; no OAI labels | g58 +.00035 (a .1, P .68, split-half .38), +.00081 (a .2); random-helper null rank 175 / 252 (null mean +.00048); report -.0003 to -.0015 | about 0 +- .0004 (GUESS) | about .5 | 0 runtime; few lines in cell 18 | DEAD |
| 4 | MCL <- Medial Meniscus alone (largest gap of 66 pairs: +.59 / +.56) | gold58 only (9 positives; picked from 30 pairs) | MCL AUC +.009 (+.0008 macro) at a .1; R7 vs report -.008 | about +.0001 +- .0004 (GUESS) | about .55 (GUESS) | 0 runtime; one line | WEAK |
| 5 | Contusion <-> Fracture decorrelation (rubric: contusion without a fracture line) | gold58 theory + check; report OOF | q ~ rho (.73 vs .77; .82 vs .77); g58 +.00001 (a .1), +.00022 (a .2); report loses | about 0 (MODEL) | about .45 | 0 runtime | DEAD |
| 6 | Rubric-mapped R6 state score for the weak non-OAI labels (Synovitis fat-pad / thickening, Fracture minus old / healed, Contusion minus degenerative) | gold58 only (R7 proxy); no OAI, R6 cannot read the OAI store | CLAIMED m2: Contusion, Fracture, Synovitis negative under the rubric map; family +.0011 mix (P .62) | <= 0 for these labels (MODEL) | about .4 (GUESS) | box GPU dump 15-20 min + worker flag + build; read 10-21 at best | DEAD |
| 7 | Synovitis as a broad label: borrow Effusion (E2) or OR-combine members | none independent (OAI Syn = Eff); gold58 | E2 killed earlier (CLAIMED THINK 8.2); gold58 Part B Syn OR -.0014, AND +.0038 (the wrong direction for a broad label) | <= 0 (MODEL) | about .4 | 0 runtime | DEAD |
| 8 | E-OA robustness across 5 MOAKS cartilage definitions (in-flight eoa10 / eoa20; NOT a new entry) | OAI MOAKS (independent) | raw OA-mean dAUC +.0038 to +.0066 (a .1), +.0063 to +.0120 (a .2), P 1.00 in 15 / 15 cells; FT row = skeptic parity | +.0005 (a .1) / +.0006 to +.0007 (a .2), CLAIMED skeptic transfer | .80 / .70 (CLAIMED) | 0, already submitted at 05:31 IST | PROMISING (already in flight, no action) |

## 5. Why nothing else is left in this lens (MODEL)
- **Severity information inside one label's score is invisible to AUC.** To use it you need a second view of the same label.
  The only second views are the members, and the OAI ladder shows the member operator is not the channel.
- **Coupling.** OA is the only group with a large model-over-expert excess AND an independent ruler (OAI) AND a regime flip
  (report OOF loses). MCL / Fracture show the excess but have no ruler, and the gold58 effect is not specific.
- **Synovitis, the weakest label, is a definition problem**, not a ranking one. Experts call it about 1.6x more often than
  reports, the host never clarified the criterion (737155), and every member learned the narrow report label. A post-hoc rank
  fix would need a better Synovitis feature than any member has. The only candidate (Effusion) failed the 132-pair null. The
  real fix is retraining on expert-like targets, which is out of the window (CLAIMED trick/label_regime.md sec 4).
- **What would change this.** A host answer on 737155 (Synovitis criterion), or any expert-graded external set with MCL /
  fracture labels. Neither exists before 10-22.

## 6. Slot plan (unchanged from EX_1019 sec 3.4 / sec 4)
- No new builds. The eoa20 / eoa10 read (~12:00-14:30 IST 10-20) by R1 vs c4 57343435 decides A.
- B = R6 57248084 in every case. Eesh selects both by hand before 10-22 23:59 UTC.
- If Eesh still wants one more 10-21 entry, nothing in this file qualifies. The least-bad lottery ticket remains E1, and
  tonight's OAI null makes it weaker, not stronger.

## 7. Files (all in session_2026-10-20/more/)
- **Box scripts and logs** (aggregates only):
  - `box_inspect.py` / `.log`: column names and grade histograms.
  - `box_sev.py` / `.log`: severity ladders x operators (BOOT 300). `box_sev_smoke.log` is the BOOT 20 smoke run.
  - `box_mdis.py` / `.log`: member spread.
  - `box_eoa_ladder.py` / `.log`: E-OA across cartilage definitions.
- **Local scripts and logs:**
  - `local_sev.py` / `.log`: report-OOF control (part A), gold58 c4 operators (part B), MCL / Contusion / Fracture coupling (part C).
  - `local_acute.py` / `.log`: acute <- degenerative sizing, the null, theory and the report control.
  - `local_bone.py` / `.log`: Contusion <-> Fracture.
- **Read only:**
  - EX_1019.md, coupling_map.md, skeptic_eoa.md, THINK_1019.md, TRICK_1019.md, label_regime.md.
  - stack2/load.py and stack_ops.json, regime_work/m14_gate.py.
  - Forum 733343 (rubric), 733491 (host on image-review labels), 737155 (Synovitis question, unanswered), 740610.
- **Box command form** used for every run:
  `ssh training@192.168.10.30 '~/miniconda3/envs/rsna/bin/python -I -' < SCRIPT.py` (BOOT set as an env var for box_sev).
