# LEVERS_1020.md: new levers after E-OA, the skeptic results, and the 10-21 / 10-22 slot plan (10-20 ~16:30 IST, Claude, levers synthesis)

RESEARCH + MEASUREMENT ONLY. Nothing was submitted, pushed, committed, versioned or downloaded. No dataset was created, and no
scheduled task, PREREG, HANDOFF, PROGRESS, a20 or a21 file was edited. The only local builds are the space lens's convex kernels
under `levers/`; this synthesis built nothing. This synthesis did not touch the box or any OAI data: every OAI number below is an
aggregate already sitting in a lens or skeptic log on the laptop. The A21 reads (eoa30 57425223, pw20 57425269, pw30 57425970,
submitted 11:41-12:01 IST) had NOT landed at writing time, so the plan is conditional on them.

Inputs:
- Three lenses: `rubric.md` (host rubric vs report language), `space.md` (output-transform space after E-OA), `lbstrat.md`
  (LB slot strategy).
- Three skeptics:
  - facet: `skeptic_facet_rubric*.log`, `box_facet_skeptic.log`;
  - convex: `skeptic_convex/*.log`;
  - R6e20 / lbstrat: `skeptic_r6e20_lbstrat.log`.
- One arithmetic script made here: `synth_convex_r1.py` -> `synth_convex_r1.log`.

Tags: MEASURED (computed in the named log), CLAIMED (older note, named), MODEL (arithmetic on stated inputs), GUESS (judgement).
Units: macro AUC. "e-4" = 1e-4 macro, and macro = OA-mean x 3/12. gold58 / gold74 numbers are INFO only, never a selection ruler.

## 1. Verdict (plain)

1. **No new lever is worth a slot.** Nothing found today is in the E-OA class: a mechanism, plus an independent expert ruler, plus an effect large against public noise. The best surviving idea is worth about +0.2e-4 private (MODEL), roughly 1/50 of E-OA's ~+10e-4.
2. **Convex E-OA** (subtract a x mean(U_o^2) instead of U_o) looked PROMISING in the lens but is **WEAK after its skeptic**. With selection corrected on OAI, the search winner beats linear by only -0.5 to +0.4e-4 (P .36-.75). The gold74 "tricompartmental over-call" mechanism is a sample quirk relative to the training labels (P .04), and RSNA INFO leans negative. E about +0.2e-4 (range -0.2 to +0.5), P(>0) ~.5-.55. It is built and gated, but not recommended.
3. **Facet confusion as support for pw over sym: REFUTED.** The lateral facet lift is PF severity: it falls to .49-.55 once the PF output is held fixed. On OAI the residual lift is medial. On raw OAI, pw minus sym is ~0 under P_FT and -1 to -2.3e-4 under P_AREA (5 / 5 models).
4. **Rubric lens: all six output-only rubric levers are DEAD** on independent rulers (fluid volume, state read-out, ACL / MCL subclasses, meniscal signal, Synovitis). The label contamination is severity thresholds that the members already rank correctly.
5. **Member weights after E-OA:** Raptor OA weight up is DEAD (wrong sign in 7 / 7 OAI configs, and the controls flip). Raptor PF down, S6fall OA up and SUB_HIGH are consistent with the mechanism but are knob-class: WEAK, with an R1 value of about 0 or below under our LB record.
6. **The plan is unchanged in substance.**
   - 10-21 05:31 IST: one entry, **R6e20 as B**. It is the only material lever left: +1.6 to +2.1 pts P(silver) (CLAIMED bpick.md).
   - Optional **pw25** only if no A21 entry displays .960. This is display fishing.
   - **10-22: nothing**, except the R6e20 no-retry fallback.
7. **Skip Amendment 21b** (the pw-over-sym overrides). Its premise is the gold74-reweighted OAI curve, which the raw-OAI skeptic run contradicts under P_AREA, so its value is about 0. Use plain R1. E1 stays DEAD.

## 2. All proposals, ranked

Ranked by expected private value x confidence. Only three proposals got a dedicated adversarial skeptic pass (rows 1, 2, 15). The
other DEAD rows were killed by their own lens on the independent ruler. "E" is the expected private macro change vs A (MODEL unless
tagged).

| # | proposal | lens | independent ruler | measurement (MEASURED unless tagged; log) | skeptic | E | P(>0) | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | B = R6e20 (R6 57248084 + the scored eoa20 block) | lbstrat / bpick | LB: E-OA scored on the hidden set (eoa20 +11e-4 public); R6 public-proven | +1.6 to +2.1 pts P(silver) vs B = R6; best B in 27 / 29 (26 / 29) sensitivity rows (CLAIMED bpick.md) | `skeptic_r6e20_lbstrat.log`: the K-ledger sign test is P .090-.145, not .033, once the exact-sum stacks c2 / c2max / c3max are not double counted. P(eoa20 private < c4) is .055 / .116 / .285 under wide / null / OAI-shaded priors. The .285 case is bpick's "neg" row (R6e20 - R6 = 0.0 / -0.6 pts), a prior the public read leaves at weight .003 (CLAIMED bpick) | pick lever (not an A change) | P(R6e20 > R6 private) ~.92 (CLAIMED) | **KEEP: slot 1, 10-21** |
| 2 | Convex E-OA twin of the A21 winner (p 2) | space | OAI MOAKS (1,704 knees), OAI-free members | 65 / 65 cells beat linear at their own best a (+.0004 OA-mean); held-out-a P .78-.90 (`box_F3.log`) | **Selection-corrected** (whole 13-family search on one half, scored on the other, 40 x 2 splits): winner held-out -0.46 / +0.29 / +0.36 / +0.36e-4, P .36-.75. Fixed p2: +0.33 to +1.71e-4 (P .70-.94), but p2 was picked on all knees (`box_selcorr_*.log`). **Mechanism:** gold74 report C +.53 vs train-report C +.08, P(C >= .53) .039; train-report minus expert at k 2 is +.10 (sd .12) (`local_mech.log`). **RSNA INFO** at matched decorrelation, p2 - lin: R7 x gold74 -.00016 to -.003, c4 x gold58 -.00014 to -.00077; extra16 identical (`local_matched.log`, `local_r7g74.log`) | -0.2 to +0.5e-4 (central ~+0.2); R1 value -0.03 to +0.34e-4 (`synth_convex_r1.log`) | ~.5-.55 (GUESS) | **WEAK** (was PROMISING). Built, not recommended |
| 3 | pw25 fill-in, only if no A21 entry shows .960 | lbstrat / PUSH | none new (E-OA axis, knob regime) | in-branch private +0.07 to +0.57e-4, P(.960) +.07 to +.27; +0.00 to +0.13e-4 if .960 is already shown (MODEL `lbstrat_fill.log`) | raw-OAI T3 (row 4) mildly lowers the pw prior | +0.02 to +0.05e-4 overall | .37-.84 by branch (CLAIMED slate_cond) | **FISHING, optional (display)** |
| 4 | Amendment 21b (eoa20 best -> A = pw20; eoa30 best -> A = pw30) | lbstrat | gold74-reweighted OAI D5 curve family | O1 +0.5 to +1.0e-4, O2 -0.3 to +1.5e-4 in the outcomes it touches; +0.1 to +0.3e-4 averaged (MODEL `lbstrat_override.log`) | `box_facet_skeptic.log` T3, raw OAI, pw - sym at a .2: P_FT -0.25 to +0.55e-4 (P .37-.79); P_AREA -1.0 to -2.3e-4 (5 / 5 models, P .00-.11). The two OAI weightings disagree, so the pw prior is about a coin | ~0 | ~.5 (GUESS) | **SKIP** (plain R1) |
| 5 | Raptor PF / OA weight down after E-OA | space | OAI (interaction only; the level effect is not measurable there) | E-OA x low-gap-weight interaction negative in 7 / 7 configs (`box_F1.log`) | none. Knob-class EB prior: R1 value -0.45 to +0.06e-4 (MODEL `lbstrat_classes.log`) | +0.5 to +2.5e-4 only IF c4's table was optimal before E-OA | ~.55-.6 | WEAK (knob) |
| 6 | S6fall OA-only weight up after E-OA | space | OAI S6-slot interaction | interaction +.0012 to +.0024 OA-mean (P 1.00); level R1 +.0006-.0009, R3p_lo ~0 (`box_F4.log`) | none. LB hw3 (S6 .3) read negative before E-OA (CLAIMED) | 0 to +1e-4 | ~.55 | WEAK (knob) |
| 7 | SUB_HIGH: subtract only the R6 / S6fall part's OA pcts | space | OAI | +.00026 to +.00052 OA-mean vs linear (P .61-.81) (`box_F2.log`) | wins 8-37 of 80 half-searches, but the search winner holds out at ~0 (`box_selcorr_*.log`) | ~+0.5e-4 | ~.55 | WEAK; needs a new cell-18 build |
| 8 | T-STATE, Baker's and MCL only | rubric | report text re-thresholded at the rubric (R7 OOF) | Baker's +.0067 (P .84, 68 pos), MCL +.0011 (P .80, 22 pos) (`local_states.log`) | none | ~+0.1e-4 | ~.55 | WEAK. R6 states are not dumped in c4, so a 10-22 read at best: no |
| 9 | T-VOL Effusion: + b pct(R6 effusion volume) | rubric | OAI MOAKS effusion at the rubric cut (grade >= 2) | c4D5 +.0005 / -.0000 (P .49) / -.0009 at b .1 / .2 / .3. Helps only at the report-like cut (grade >= 1). Rubric-text OOF mean3 -.0021 (P .12) (`box_rubric.log`, `local_fluid.log`) | none | 0 to -0.1e-4 | ~.4 | DEAD |
| 10 | Demote ACL mucoid / low-grade and MCL chronic / grade 1 | rubric | rubric-text OOF; OAI ACL code 1 | perfect-demotion bounds +.0018 to +.0090 per label (<= +8e-4 macro); no transform realizes them (`local_subclass.log`) | none | ~0 | ~.5 | DEAD |
| 11 | T-STATE overall (rubric-mapped heliosli states, mix .5) | rubric | rubric-text OOF | 6 / 8 labels lose (Effusion -.0023, Fracture -.0042) (`local_states.log`) | none | ~0 | ~.45 | DEAD |
| 12 | Meniscal signal-only demotion | rubric | OAI meniscal T123 | signal-only vs normal AUC .69-.79, bound +.045 / +.025, but no second view survives (`box_rubric.log` B) | none | 0 | n/a | DEAD |
| 13 | Synovitis broadening (experts broader than reports) | rubric | none independent | census only (`census_lang.log`) | none | unknown | ~.4 | DEAD (no ruler) |
| 14 | T-VOL Baker's | rubric | OAI popcys / WORMS size; rubric-text OOF | negative on every ruler (OAI c4D5 -.0078 at b .2) | none | < 0 | ~.2 | DEAD |
| 15 | Facet confusion as support for pw over sym | rubric | report-regime OOF (mechanism only) | Lateral OA AUC .745-.764, facet-only vs other PF (`local_facet.log`) | **REFUTED**: PF-residual AUC .49-.55; lateral-vs-medial facet contrast .47-.58. On OAI the residual facet lift is medial (.57-.64), not lateral (.47-.52). Raw-OAI pw - sym is ~0 or negative (`skeptic_facet_rubric*.log`, `box_facet_skeptic.log`) | n/a | n/a | DEAD (no pw tie-break) |
| 16 | Raptor OA weight up after E-OA | space | OAI role-matched nests + reversed controls | I(.2) -.0009 to -.0035 OA-mean, 7 / 7, P(>0) <= .03; controls +.0034 / +.0037 (`box_F1.log`) | not needed | < 0 | ~.25 | DEAD |
| 17 | Regime-aware per-member E-OA | space | OAI | -.00012 to -.00059 OA-mean vs linear (`box_F2.log`) | | < 0 | ~.3 | DEAD |
| 18 | Probit-space E-OA | space | OAI | -.00006 to -.00026 | | < 0 | ~.3 | DEAD |
| 19 | Agreement-gated E-OA | space | OAI | agree -.0003 to -.0011; disagree -.0006 to +.00015 | | < 0 | ~.3 | DEAD |
| 20 | Concave (sqrt) subtraction | space | OAI | -.0003 to -.0005 | | < 0 | ~.2 | DEAD |
| 21 | SUB_LOW: subtract Raptor's own OA outputs | space | OAI | -.0006 to -.0007 (P .01-.24) | | < 0 | ~.2 | DEAD |
| 22 | E1 soft-OR on the winner | PUSH / lbstrat | report-label OOF only (weak ruler) | gold58 float32 -.0001 (P .45, CLAIMED e1build.md). The weak-ruler class's LB record is 1 / 10 | lbstrat EB: R1 value -0.46 to +0.09e-4 | <= 0 | ~.5 | DEAD for a slot |

### 2.1 Why convex fell from PROMISING to WEAK (MEASURED unless tagged)

- **Selection.** The space lens held out only the choice of a. The skeptic re-ran the whole search (13 families x 11 a values) on
  one stratified half and scored the winner on the other half (`skeptic_convex/box_selcorr_*.log`):
  - The winner is unstable: p3, SUB_HIGH, hinge and probit all win often, and p2 is rank 1 in only 0-33 % of half-searches.
  - In-sample edge +0.85 to +2.1e-4 shrinks to -0.46 to +0.36e-4 held-out.
  - "Fixed p2" holds +0.33 to +1.71e-4, but p2 was not chosen before the data. The honest number lies between the two.
- **Mechanism.** The models learned from train report labels, not from the gold74 reports.
  - The train reports are additive in logit across k (P(Y | k) .161 / .41 / .736; C +.077).
  - The gold74 reports' strong convexity (C +.53) has P .039 under 70-knee draws of the train labels.
  - The excess of train reports over experts does sit at k 2 (+.10 vs +.007 / +.01), but at about 0.8 sd (`local_mech.log`).
  - So the direction survives, but the size does not.
- **RSNA INFO.** At matched decorrelation strength, p2 is below linear on R7 x gold74 and c4 x gold58 at every a. The 16 extra
  gold74 studies (the only fresh labels) cannot tell the two apart (`local_matched.log`, `local_r7g74.log`). INFO, not a veto.
- **R1 value of one twin ticket** (MODEL `synth_convex_r1.log`; public twin sd 2.2e-4, MEASURED `box_F5*.log`):

  | prior on the population gain | R1 value | P(wrong sign given it moves) |
  |---|---|---|
  | space lens, N(+1.0, .5) | +0.71e-4 | .02 |
  | fixed-p2 held-out shaded, N(+.5, .5) | +0.34e-4 | .12 |
  | selection-corrected, N(+.15, .5) | +0.12e-4 | .32 |
  | RSNA INFO shaded, N(-.2, .6) | -0.03e-4 | .55 |

  At the skeptic-informed priors, that is at most ~0.1-0.2 pts P(silver) (GUESS slope 0.5 pt per 1e-4, from bpick).

## 3. Build-ready specs for the survivors

C4T = `{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}`
(as in `a21/tick_a21.py`). Every push or submit needs Eesh's go plus its PREREG amendment. None of that is done.

### 3.1 R6e20: slot 1 on 10-21, the B hedge (KEEP)

| item | value |
|---|---|
| kernel | `session_2026-10-20/push/kernel_v224r6flips6feoa20j`, notebook sha 2a5ef93a, src R6 dfee2ba3. Kaggle id eeshsaxena/rsna-sub-v224r6flips6feoa20j |
| what it is | R6 57248084 + the verbatim A20 E-OA block (sym, a .2). The xref shows it differs from the scored eoa20 only in cell 14 (exactly as R6 differs from c4), and cell 18 is byte-identical (MEASURED build receipt) |
| offline gate | R6EOA_OFFLINE_GATE PASS (MEASURED `push/gate_r6eoa_offline.log`) |
| commit gate | `python session_2026-10-19/a20/probe_check_eoa.py <OUT> --a224 0.35 --table '{"Medial OA": 0.5, "Lateral OA": 0.5, "PF OA": 0.5}' --aoa 0.2` must print PASS 17/17 |
| B rule (fix before its score; draft Amendment 22 in bpick.md sec 6) | B = R6e20 if it is COMPLETE and displays >= .958, else B = R6 57248084. R6e20 is never A |
| caveat (skeptic) | R6e20 hedges the c4 knob layer, not E-OA itself. If E-OA failed privately (P .02-.06 per bpick; .285 only under an OAI-shaded prior the public read rejects), eoa10 or R6 would be the better B by <= 0.8 pts |

### 3.2 pw25: optional display fill-in, conditional (FISHING)

- Kernel: `session_2026-10-20/push/shape/kernel_v224r6c4pw25j`, sha 64ed7eba, src c4 b47c29c3.
  - SHAPE_OFFLINE_GATE PASS (MEASURED `push/shape/gate_shape_offline.log`).
  - Kaggle id eeshsaxena/rsna-sub-v224r6c4pw25j.
- Commit gate: `python session_2026-10-20/a21/probe_check_eoa21.py <OUT> --a224 0.25 --table '<C4T>' --aoa 0.25 --mode pw` must
  print PASS 17/17.
- Use only if no A21 entry displays .960 and Eesh wants a display ticket. It is the one fishing ticket. Never pair it with eoa25,
  pw35 or a convex twin.

### 3.3 Convex twin: WEAK; only on Eesh's explicit request, and INSTEAD of pw25 (never both, never with E1)

| A21 winner | twin | sha | commit gate (`levers/probe_check_convex.py <OUT> --a224 0.25 --table '<C4T>' ...`) |
|---|---|---|---|
| pw20 | `levers/kernel_v224r6c4pw20p2j` (exponent only) | 5e0cf2ca | `--aoa 0.2 --mode pw --p 2` |
| pw30 | `levers/kernel_v224r6c4pw25p2j` (exponent AND a .3 -> .25: two changes) | 9750dfbd | `--aoa 0.25 --mode pw --p 2` |
| eoa20 | `levers/kernel_v224r6c4eoa20p2j` (exponent only) | 2d51de83 | `--aoa 0.2 --mode sym --p 2` |
| eoa30 | not built (`python levers/build_convex.py sym 0.25 2`, then gate) | | `--aoa 0.25 --mode sym --p 2` |

- The gate must print **PASS 18/18**. Check (d3), the receipt p_oa, is what separates p2 from linear on the visible studies.
- CONVEX_OFFLINE_GATE PASS: the p1 controls are byte-identical to the scored outputs, and the OA columns match an independent
  recompute at 2.8e-8 (MEASURED `gate_convex_offline.log`).
- Never push the p1 controls `kernel_v224r6c4pw30p1j` (b865439d) or `kernel_v224r6c4eoa20p1j` (b1f1a5cc).
- It needs a PREREG amendment that overrides Amendment 21's shape closure.

### 3.4 Nothing else is build-ready, by design

SUB_HIGH, Raptor PF down and S6fall OA up are unbuilt and knob-class, with no slot. T-STATE would need an R6 `--dump-state` worker
change, so it is a no.

## 4. Adaptive slot plan for 10-21 / 10-22 (from lbstrat, amended by the skeptics)

### 4.1 Timeline (IST)

| when | action | who / needs |
|---|---|---|
| 10-20 ~17:00-18:30 | A21 reads land. (1) Snapshot (`session_2026-10-14/research960/lbprobe/fetch_api_lb.py`). (2) Note the best-id and every A21 display. (3) A = R1 best-id (plain R1, no 21b). (4) Run `push/work/abc_update.py` for the refreshed P(.960) of pw25 | main session, read-only |
| 10-20 evening | Eesh's go on R6e20 + Amendment 22, then push R6e20 (commit ~20 min) and run the commit gate (PASS 17/17). The one optional ticket of sec 4.2 follows the same steps, if approved | Eesh |
| 10-21 05:31 | Submit R6e20, plus the optional ticket if approved, through a Windows scheduled tick (overnight automation only that way) | tick + GO file |
| 10-21 ~11:00-11:30 | Reads arrive. Apply the B rule (3.1). A = R1 best-id: an optional ticket that moves the best-id switches A once, with no follow-up | main session |
| 10-22 05:31 | Only if R6e20 errored or was not submitted: the R6e20 fallback (no retry). No A probe | Eesh's go |
| 10-22 ~10:15 | Last readable submit. Do not use | |
| before 10-22 23:59 UTC (10-23 05:29 IST) | Eesh selects A and B by hand. Never auto-selection: it would pair two E-OA twins | Eesh |

### 4.2 10-21 slate by A21 outcome (outcome P and values MODEL, `lbstrat_fill.log`; P(any A21 entry shows .960) ~.70, CLAIMED implied.md)

| A21 outcome (P) | A (plain R1) | slot 1, 05:31 | optional slot 2 (FISHING, needs its own go + amendment) | slots 3-5 |
|---|---|---|---|---|
| best-id displays .960 (.67-.78) | best-id | R6e20 | none (pw25 is worth +0.00 to +0.13e-4 and no display) | empty |
| pw20 or pw30 best, .959 (.11-.16) | best-id | R6e20 | pw25: P(moves) .33-.53, private +0.07 to +0.43e-4, P(.960) +.14 to +.27 | empty |
| eoa30 best, .959 (.03-.07) | eoa30 | R6e20 | pw25: private +0.11 to +0.57e-4, P(.960) +.07 to +.10 | empty |
| eoa20 stays (.07-.12) | eoa20 | R6e20 | none (Amendment 21 closes the shape search) | empty |

### 4.3 Mechanism-backed vs fishing

- **Mechanism-backed or public-proven** (the only classes the stopping rule S1 allows):
  - R6e20 only. Its E-OA block is scored on the hidden set (eoa20 57413368) and its base is scored (R6 57248084).
  - It is a pick hedge, not a public probe.
  - **No mechanism-backed A probe exists after today.** Our LB record for that class is 3 / 3 (CLAIMED lbstrat.md), but nothing
    is left in it.
- **Fishing** (everything else):
  - What is left is the knob regime on the E-OA axis: corr(public delta, private delta) .1-.45, and sign agreement .56-.70
    (MODEL, lbstrat sec 3).
  - Our knob record is 2 / 11, with sign-test P .09-.15 after the skeptic correction.
  - The candidates: pw25 (display), the convex twin, the 21b overrides, eoa25 / pw35 / pw40, E1, and any table or weight move.
  - **At most ONE fishing ticket in total**: pw25 by default, only in its branch.

### 4.4 Stopping rules (lbstrat 6.5, kept)

- **S1.** A probes only from the mechanism + expert-ruler class or the public-proven class. None is available, so STOP.
- **S2.**
  - At most one E-OA fill-in, pw25 only.
  - A non-move closes the shape search.
  - A move switches A once.
  - Never map the curve: the reads are .97 correlated.
- **S3.** No E1, no knob or table move, no gold58-picked anything, no own-member leg.
- **S4.** A non-move counts as negative unless a parallel entry moved first (R1 censoring).
- **S5.** Nothing whose read cannot land before 10-22 ~16:00 IST.
- **S6.** A and B are selected by hand.

### 4.5 What would reopen this

- A host answer on the Synovitis criterion (forum 737155), or any expert-graded set with synovitis / MCL / fracture grades:
  row 13 only.
- An A21 pattern where pw and sym split by a full display step: that would re-weigh row 4, but only for a rule fixed before the
  10-21 reads.
- None of these is expected before 10-22.

## 5. Corrections to today's lens notes (for whoever writes HANDOFF)

- `rubric.md`: the following no longer hold; the facet skeptic refuted them. The facet AUC is PF severity, and the residual
  side-specific lift on OAI is medial.
  - sec 0 item 5, sec 3.4 and proposal 7: "facet confusion supports pw";
  - "use it as a pw tie-break".
- `space.md`:
  - convex is WEAK, not PROMISING;
  - E private is about +0.2e-4 (range -0.2 to +0.5), not +0.5 to +1.5e-4;
  - the "n 70 report over-call" mechanism is a gold74 sample quirk relative to the training labels (P .039). Only its direction
    survives.
- `lbstrat.md`:
  - The K-ledger sign test is P .090-.145, not .033. The direction is unchanged.
  - 21b is SKIP, not "write before the reads": the raw-OAI pw - sym run contradicts the gold74-reweighted family under P_AREA.
  - The R6e20 hedge caveat of sec 3.1 applies.
- `push/PUSH_1020.md` sec 2.2: "E1 is the better optional ticket" is superseded (lbstrat EB, then this note). E1 is DEAD for a slot.
  If one optional ticket is used, it is pw25 in its branch.

## 6. Files (session_2026-10-20/levers/)

- This note: `LEVERS_1020.md`.
- Made here: `synth_convex_r1.py` -> `synth_convex_r1.log`. This is arithmetic only, with no data.
- Lens notes: `rubric.md`, `space.md`, `lbstrat.md`. Their scripts and logs are listed in each note's Files section.
- Skeptics:
  - `skeptic_facet_rubric.py` / `.log`, `skeptic_facet_rubric2.py` / `.log`, `box_facet_skeptic.py` / `.log` (facet);
  - `skeptic_convex/box_selcorr.py` + `box_selcorr_{R1,R3p_lo}_{P_FT,P_AREA}.log`, `local_mech.py` / `.log`, `local_matched.py` /
    `.log`, `local_r7g74.py` / `.log` (convex);
  - `skeptic_r6e20_lbstrat.py` / `.log` (R6e20 / lbstrat).
- Builds (space lens; none pushed):
  - candidates: `kernel_v224r6c4{pw20p2,pw25p2,eoa20p2}j/`;
  - controls, never push: `kernel_v224r6c4{pw30p1,eoa20p1}j/`;
  - `build_convex.py`, `build_receipt_*.json`, `gate_convex_offline.py` / `.log`, `probe_check_convex.py`.

## 7. Governance

- This synthesis made no box contact and read no OAI file. It only read lens and skeptic outputs already on the laptop.
- **Slip, reported by the rubric lens (10-20 ~15:42 IST).**
  - A structure probe (`box_struct.py`, since deleted) printed about 30 OAI knee ids as dict keys into that sub-session's
    transcript.
  - It printed no grades, labels or per-knee values, and nothing was written to a file.
  - Every later box command went through an id-redacting `sed`.
  - Eesh should know about this. It is the second id exposure.
- Check made here (MEASURED): no file in `levers/` (including `skeptic_convex/`) matches `oai_[0-9]+_[LR]`, `oai_[0-9]{4,}` or a
  bare 7-digit `9xxxxxx` id (0 matches each).
- Box GPU was not used by any lens; at ~15:40 IST it was busy with another job (85-96 %).
