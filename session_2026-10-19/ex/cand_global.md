# cand_global.md: global factor removal ("general abnormality" factor), 10-20 night, Claude. RESEARCH + MEASUREMENT ONLY.

Nothing was built, pushed, committed or submitted. No dataset, GPU job, scheduled task, PREREG, HANDOFF, PROGRESS or a20 file
was touched. Box use: two read-only CPU python runs piped over ssh stdin (a 6-boot smoke run, then the full run). They wrote
nothing on the box and printed aggregates only (n, prevalence, AUC deltas, P, ESS, phi, rho). No OAI id, row or per-knee value
left the box. Only the aggregate gold74 2x2 counts (36 pairs, from local_coupling.json) went TO the box, embedded in the script.

Tags: MEASURED = computed tonight (file named); CLAIMED = older note; MODEL = arithmetic on stated assumptions; GUESS = judgement.
"Macro" = 12-label macro AUC equivalent (sum of per-label AUC deltas / 12). MM / LM = menisci, MOA / LOA / PF = OA compartments,
Eff / Syn / Bak = Effusion / Synovitis / Baker's. "Core" = the 7 covered labels without ACL (rare, degenerative on OAI) and
Synovitis (OAI = Hoffa synovitis; the OAI-free models' Syn output tracks their Eff output).

## 0. Verdict: KILL

The global factor removal loses on OAI for all 3 OAI-free models at every tested a, in both shapes:

- **Standalone** (every covered label minus a times the mean pct of the others). It loses outright and is always worse than E-OA
  alone.
- **On top of E-OA .1** (the non-OA labels only). It loses even on raw OAI, where expert coupling is lower than RSNA's, which is
  the regime that favours decorrelation.

It loses more when OAI is reweighted to the gold74 expert coupling of all 36 pairs. c4 has no excess coupling outside the OA trio
to remove in the first place: mean gap over the 33 non-trio pairs is -.011, with 15 pairs up and 18 down.

E-OA, the group-wise transform, stays the only member of this family worth having. Best a for the global part: none (0).

## 1. Criteria

| criterion | result |
|---|---|
| (1) OAI support, 3 models, a .1 / .2, P >= .90, a -.1 control < 0 | **FAIL.** Every delta is < 0 with P>0 of .00, both standalone and on top of E-OA. |
| (2) still gains after coupling-matched reweighting to gold74 | **FAIL.** More negative under R9, R1oa3 and R9hi. About 0 only under R9lo (gold74 coupling 1 sd lower on all 36 pairs at once). |
| (3) gold58 c4 not clearly against | Passes (noise): G9 .1 +.0008 (P .79); on top of E-OA .1 +.0003 (P .64). |
| report-label OOF should lose | It loses (-.0021 standalone, -.0018 on top of E-OA). This is uninformative here, because R7 OOF also loses on the EXPERT labels (gold58 -.0012, gold74 -.0016). |

## 2. Transforms (same code on the box and locally)

U = pct of the 12 prediction columns. The deployed form adds an outer pct, which leaves AUC unchanged.

| name | what it changes |
|---|---|
| G9 a | each of the 9 OAI-covered labels: U_t - a * mean(U of the other 8 covered labels) |
| G12 a | as G9, but the mean runs over the other 11 prediction columns |
| G12all a | the deployed 12-label form; also moves MCL, Contusion and Fracture (local only) |
| E1+N9 a | OA trio = E-OA .1; each non-OA covered label (ACL, MM, LM, Eff, Syn, Bak) gets U_t - a * mean(U of the other 8) |
| E1+N12 a / E1+N12all a | as E1+N9 with the 11-column mean |

Tested at a in {-.1 (control), .1, .2}.

Box script: `box_global_template.py`, filled by `make_global.py` into `box_global_run.py`.

**OAI labels.** B1 binarisation as in box_coupling_template.py: ACL >= .5; menisci = tear; OA = P_FT MOAKS full-thickness
proxy; Eff grade >= 2; Hoffa Syn >= 1; Bak = 1.
- 2,581 knees.
- Coverage: ACL 1,654, MM / LM / OA 2,399, Eff 1,660, Syn 1,659, Bak 835 (MEASURED, box_global.log).
- B2 (alternative thresholds) was run on raw OAI only.

**Reweighting.**
- R9: raking over all 36 pairwise 2x2 tables, using the knees where both labels exist. It keeps the OAI marginals and sets the
  gold74 log OR (+.5 smoothed). It converged: |dlogOR| mean .000, max .001.
- R9hi / R9lo: every gold74 log OR +/- 1 sd.
- R1oa3: the skeptic's OA-cell match.

Weighted mean label phi over the 36 pairs (MEASURED):

| scheme | mean phi, 36 pairs | ESS |
|---|---|---|
| raw | .106 | 2,581 |
| R9 | .169 | 1,037 |
| R9hi | .262 | 687 |
| R9lo | .072 | 1,107 |

For reference, gold74 itself has a mean phi of .181.

## 3. OAI results (MEASURED, box_global.log, summ_global.out; 3 models = D5 5-fold mean / s1044 / s1045)

### Standalone (macro delta vs base, P>0)

| transform | raw | R9 (gold74 coupling) | R9lo |
|---|---|---|---|
| G9 +.1 | -.00047 / -.00096 / -.00096 (P .00) | -.00058 / -.00156 / -.00113 | +.00072 / -.00009 / +.00048 |
| G9 +.2 | -.00220 / -.00306 / -.00299 | -.00299 / -.00445 / -.00341 | -.00002 / -.00117 / -.00009 |
| G9 -.1 (control) | -.00054 / -.00004 / -.00001 | +.00006 / +.00064 / +.00050 | -.00110 / -.00063 / -.00093 |
| G12 +.1 | -.00054 / -.00099 / -.00093 | -.00065 / -.00155 / -.00119 | +.00057 / -.00016 / +.00038 |
| EOA .1 (reference) | +.00155 / +.00124 / +.00133 | +.00035 / -.00006 / +.00001 | +.00068 / +.00025 / +.00027 |

- G9 +.1 minus EOA .1, raw: -.0020 / -.0022 / -.0023 (P .00).
- The same at R9: -.0009 / -.0015 / -.0011 (P .00).
- The global factor dilutes the OA decorrelation (2/8 of a instead of a) and adds losses on the other labels.

### On top of E-OA .1 (macro increment vs EOA .1, P_inc; core increment)

| transform | raw | R9 | R9hi | R9lo |
|---|---|---|---|---|
| E1+N9 +.1 | -.00125 / -.00132 / -.00164 (P .00); core -.0003 / -.0004 / -.0004 (P <= .01) | -.00060 / -.00108 / -.00103; core -.0008 / -.0010 / -.0009 | -.0012 / -.0019 / -.0021 | +.00022 / -.00013 / +.00016 (P .84 / .30 / .79); core -.0002 / -.0004 / -.0003 |
| E1+N9 +.2 | -.0036 / -.0036 / -.0042 | -.0029 / -.0032 / -.0029 | -.0044 / -.0052 / -.0050 | -.0009 / -.0011 / -.0005 |
| E1+N12 +.1 | -.0013 / -.0013 / -.0016 | -.0007 / -.0011 / -.0012 | -.0013 / -.0019 / -.0022 | +.0000 / -.0002 / -.0000 |

- **Core increment is negative under every coupling scheme for all 3 models** with the rubric-like B1 labels. The labels that
  move in core are MM, LM, Eff and Bak.
- Only B2 makes core positive: +.0004 / +.0003 / +.0002. That comes entirely from the MM >= .2 threshold (any intrasubstance
  signal, prevalence .80), which is not a rubric-like tear label. B2 macro is still -.0004 / -.0005 / -.0011.

### Per label, E1+N9 +.1, raw (ACL, MM, LM, MOA, LOA, PF, Eff, Syn, Bak)

D5: [-.0070, -.0001, -.0006, +.0055, +.0070, +.0061, -.0010, -.0043, -.0020]

- Syn and Bak lose under every scheme.
- ACL flips sign with the scheme: -.007 raw, +.006 R9. OAI ACL tears are degenerative, so this is OAI-specific.

### Why it fails although the OAI models over-couple

The OAI-free models' mean Spearman over the 33 non-trio pairs is .42 / .40 / .38. OAI expert phi on those pairs is .11. So on OAI
the apparent excess is large, and removing it still loses.

The shared component outside the OA trio is informative for each label. It is real shared disease or severity, not inherited
report coupling. This is the failure case already flagged in the oai_eoa_check.py header: a low label phi plus a high model rho
is not enough (CLAIMED there, MEASURED here).

## 4. RSNA side (MEASURED, local_global.log / local_global.json)

**Coupling** (36 covered pairs, from local_coupling.json):

| source | OA trio | other 33 |
|---|---|---|
| gold74 tetrachoric | .533 | .259 |
| c4 rho on gold58 | .828 | .249 |
| R7 OOF train rho | .762 | .278 |
| report v2 tetrachoric | .798 | .408 |

- c4's gap on the other 33 pairs: -.011 on average, 15 positive and 18 negative.
- So outside the OA trio c4 is NOT over-coupled relative to experts. A general factor has nothing to remove.

**gold58 c4 final** (m14_gate rebuild, parity macro .94102; consistency only, not a ruler):

| transform | macro (P) |
|---|---|
| EOA .1 | +.00142 (P .99) |
| G9 +.1 | +.00079 (P .79); vs EOA .1: -.00062 |
| G12all +.1 | +.00117 (P .83) |
| E1+N9 +.1, increment over EOA .1 | +.00028 (P .64) |
| E1+N12all +.1, increment over EOA .1 | +.00037 (P .60) |
| E1+N9 +.2, increment over EOA .1 | -.00090 |

- Per label, E1+N9 +.1: ACL -.0012, LM -.0012, Eff +.0012, Syn +.0024, Bak +.0009. These are single-pair steps on 58 studies,
  i.e. noise.
- Not clearly against, and not for either.

**R7 OOF on expert labels** (report-trained, so it is the closest analogue to c4's training regime):

| set | G9 +.1 | E1+N9 +.1 increment |
|---|---|---|
| gold58 | -.00117 (P .10) | -.0007 (P .20) |
| gold74 (n 74) | -.00161 (P .03) | -.0011 (P .07) |

**R7 OOF on train vs report_labels_v2** (per label, UNK dropped, n 4,333):

| transform | macro (P) |
|---|---|
| G9 +.1 | -.00212 (P .00) |
| E1+N9 +.1 increment | -.00179 (P .00) |
| G9 -.1 (borrow the factor) | +.00125 (P 1.00) |

The report-label check loses in the predicted direction. It cannot separate "report coupling" from "real shared signal" here,
because the expert-label R7 checks lose too.

## 5. Side note: the reverse (borrowing the general factor for non-OA labels, E1+N9 -.1)

This is not a candidate.

| check | increment |
|---|---|
| OAI raw | +.00034 / +.00047 / +.00078 (P >= .99), but core -.0002 / -.0001 / -.0002 (ACL and Syn carry it, both OAI-specific) |
| R9 | +.00025 / +.00042 / +.00050, core +.0003 / +.0004 / +.0004 |
| R9hi | +.0009 / +.0011 / +.0013 |
| R9lo | -.0005 / -.0004 / -.0005 |
| gold58 c4 | -.00054 (P .22) |
| report labels | +.0011 |

Its sign follows the assumed hidden coupling, and gold58 c4 is against it. This matches the coupling map's deficit-side pairs
(Bak with OA and menisci, LOA-Eff), which OAI cannot confirm. INCONCLUSIVE leaning KILL (GUESS). Not worth a slot.

## 6. Expected hidden effect

- **E1+N9 .1 on top of eoa10** (MODEL): R9 gives -.0006 to -.0011 macro across the 3 OAI-free models. c4 has less non-trio
  excess than those models (gap -.01 vs RSNA experts), so the expected value is about -.0008 macro (range -.0016 to +.0002).
- **Standalone G9 .1 instead of E-OA:** -.0009 to -.0015 vs EOA .1 at R9.
- P(gain > 0): about .08 (GUESS).

## Files (all in session_2026-10-19/ex/)

- box_global_template.py, make_global.py, box_global_run.py, box_global.log (box aggregates + JSON block)
- summ_global.py and summ_global.out (tables above)
- local_global.py, local_global.log, local_global.json (gold58 c4, R7 OOF gold58 / gold74 / report labels, coupling summary)
