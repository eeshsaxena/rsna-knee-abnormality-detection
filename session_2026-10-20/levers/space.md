# space.md: output-only transform space after E-OA (10-20 ~15:45-16:10 IST, Claude, levers track). RESEARCH + MEASUREMENT ONLY.

Nothing was submitted, pushed, committed, versioned or downloaded. No dataset, no scheduled-task edit, no GPU job. PREREG,
HANDOFF, PROGRESS, a20 and a21 were only read (a21 output folders were read by the gate self-test, nothing written there).
Local builds only, all under `session_2026-10-20/levers/` (sec 6). Sibling tracks are writing other files into this same folder
(rubric / lbstrat / census_lang / local_facet ...); they are not mine and I did not touch them.

**OAI governance.** Every OAI number below comes from `box_space.py` piped over ssh stdin into
`~/miniconda3/envs/rsna/bin/python -I -` (CPU, nice 5, 4 threads), which prints aggregates only (AUC deltas, sd, P, rho, q,
gap, ESS, n, prevalence, argmax). No OAI id, row or per-knee value was printed, written or copied off the box, no OAI CSV was
head / cat'ed, nothing was written on the box. The box GPU was busy with someone else's job (96 %); I did not use it.

Tags: MEASURED (computed here, log named), CLAIMED (older note, named), MODEL (arithmetic on stated inputs), GUESS (judgement).
Units: "OA-mean" = mean AUC of the 3 OA labels; macro = OA-mean x 3/12; "e-4" = 1e-4 macro.

## 1. Verdict (plain)

1. **Raptor should NOT get more OA weight after E-OA. If anything it should get less, mainly in PF OA.** On the independent OAI
   expert ruler, E-OA and the weight of the lower-coupling-gap member are substitutes: after decorrelation the optimal weight
   moves toward the member with the HIGHER gap. That holds in 7 of 7 role-matched configs (P(>0) <= .03 each) and flips sign in
   both role-reversed controls (P 1.00) (MEASURED `box_F1.log`). On RSNA, Raptor is the lower-gap member (gold58 gap .052 vs
   .121 for the R6 + S6fall part, P .96, INFO; label-free: Raptor's PF output couples less, P .90) (MEASURED `local_space.log`,
   `local_pairs.log`). So "Raptor OA up" is DEAD (wrong sign). "Raptor PF / OA down" is mechanism-right but its size rests on an
   assumed pre-E-OA optimum: WEAK, not a slot.
2. **One new family survives the independent ruler: CONVEX subtraction** (subtract a x mean of U_o ** 2 instead of U_o).
   - OAI: beats linear E-OA at its own best a in 65 of 65 cells where E-OA helps at all (12 configs x 4 coupling schemes x 2
     cartilage proxies), mean +.0004 OA-mean (+1e-4 macro); held-out a selection P .78-.90 in 5 of 5 blends (MEASURED `box_F3.log`).
   - Mechanism (MEASURED, n 70): on the SAME gold74 knees, report labels over-call a compartment when BOTH others are diseased
     (P(Y_t = 1 | 2 others positive) report .88 vs expert .64; 111 cell 17 vs 7) and under-call when one is (.27 vs .40). The
     report-trained excess is convex in other-compartment severity; convex subtraction matches it. Report-OOF control: convex
     removes more report-regime coupling than linear (-.00034, P>0 .04), as the mechanism predicts.
   - Against it: gold58 c4 replay (INFO ONLY, n 58) has convex below linear by .0005-.0008 macro (P>0 .15-.26), a handful of
     pair swaps. At gold74-level coupling (R1) the same-a twin is ~0 on OAI; the gain shows at public-like gap (R3p_lo).
   - E private +0.5 to +1.5e-4 macro (MODEL), P(>0) ~.6 (GUESS). **PROMISING as an optional upside ticket only**: a better
     ticket than E1 (which has no independent ruler), not a change to the default slate (R6e20).
3. Every other family is DEAD or WEAK on the same ruler (sec 3): regime-aware per-member E-OA, probit space, agreement-gated
   E-OA (both directions), concave subtraction, subtracting Raptor's own OA outputs. Subtracting only the R6 / S6fall part's
   OA outputs ("SUB_HIGH") and S6fall-OA-weight-up are mechanism-consistent but small and WEAK.
4. Builds: convex twins `v224r6c4pw20p2j` / `pw25p2j` / `eoa20p2j`, CONVEX_OFFLINE_GATE PASS; commit gate
   `probe_check_convex.py` self-tested (sec 5). Nothing pushed. Any use needs Eesh's go + a PREREG amendment (Amendment 21
   closed the shape search; this is a new functional form, but it is still an E-OA shape change).

## 2. The Raptor question in detail

### 2.1 Mechanism on OAI (MEASURED `box_F1.log`, B 200, R1 = gold74 8-cell joint, knee Poisson x target Dirichlet)

Role mapping: on OAI under gold74 coupling, R6 / S6fall are the LOW-gap members (gap +.005 / -.004; their OA coupling
collapses on OAI, CLAIMED r6_oai.md 11.2) and the fleet D5 / s1044 / s1045 the HIGH-gap ones (+.080 / +.060 / +.048).
"LOW in the Raptor slot" is therefore the role-matched proxy for c4; "HIGH in the Raptor slot" is the role-reversed control.

Interaction I(a) = [A(more LOW weight, a) - A(less LOW weight, a)] - same at a 0, OA-mean:

| config (Raptor slot / R6 slot / S6 slot) | weight step | I(.2) | I(.3) | P(I > 0) |
|---|---|---|---|---|
| pair R6 / D5 | wl .25 -> .55 | -.00229 | -.00402 | .00 |
| pair S6 / D5 | same | -.00295 | -.00499 | .00 |
| pair R6 / s1044 | same | -.00106 | -.00194 | .01 |
| pair R6 / s1045 | same | -.00086 | -.00183 | .03 |
| c4 nest R6 / D5 / s1044 | Raptor-table multiplier m .5 -> 1.5 | -.00293 | -.00504 | .00 |
| c4 nest S6 / D5 / s1045 | same | -.00354 | -.00578 | .00 |
| c4 nest R6 / s1044 / s1045 | same | -.00171 | -.00286 | .01 |
| **control:** pair D5 / R6 (reversed) | wl .25 -> .55 | **+.00343** | **+.00527** | 1.00 |
| **control:** c4D5 nest D5 / R6 / S6 (reversed) | m .5 -> 1.5 | **+.00372** | **+.00554** | 1.00 |

- The size tracks the gap difference between the members (about -.1 OA-mean per unit weight per unit gap difference at a .2;
  R6/D5 diff .075 -> -.0076 per unit weight, R6/s1045 diff .043 -> -.0029). Same sign under R3p, R3p_lo, R3p_hi and P_AREA.
- Per label it sits in PF OA (nest R6 / D5 / s1044, I(.2) Medial +.0004, Lateral +.0002, PF -.0094).
- Same mechanism for the S6 slot (`box_F4.log`): with the highest-gap member in the S6 slot, raising its weight .1 -> .3 gains
  after E-OA, I(.2) +.0012 to +.0014, I(.3) +.0020 to +.0024 (P 1.00, 2 configs); reversed control I ~0 / -.0003 (P .25-.42).

### 2.2 Which member is low-gap on RSNA (gold58, MEASURED `local_space.log`, `local_pairs.log`; labels used only as INFO)

| member (gold58, n 58) | rho (label-free) | q | gap | OA AUC M / L / PF (info) |
|---|---|---|---|---|
| Raptor .75 r384 + .25 r224 | .797 | .745 | +.052 | .981 / .780 / .862 |
| R6 exact | .778 | .686 | +.091 | .991 / .865 / .916 |
| S6fall (n42gpu) | .845 | .714 | +.131 | .981 / .851 / .902 |
| R6 + S6fall part (.67 / .33) | .807 | .686 | +.121 | .988 / .864 / .914 |
| c4 final | .813 | .711 | +.102 | .989 / .843 / .901 |

- Raptor minus the R6 + S6fall part: gap -.077 (sd .099, P<0 .96); label-free rho -.012 (P<0 .64). Raptor is low-gap through a
  higher q (less compartment-specific output), not a lower rho.
- Label-free per pair, Raptor minus R6 + S6fall: M-L +.041 (P>0 .86), M-PF -.051 (P>0 .10), L-PF -.021 (P>0 .34). Raptor's PF
  output is the less coupled one, matching the OAI PF concentration. S6fall is more coupled than R6 on every pair (P .92-.99).
- Raptor was trained with OAI MOAKS OA rows (CLAIMED, nartaa d96 card), which fits it being the expert-regime OA member.

### 2.3 Answer and size (MODEL / GUESS)

- **More Raptor OA weight after E-OA: DEAD.** Wrong sign on the independent ruler, with the control flipping as predicted.
- **Less Raptor weight, mainly PF OA** (e.g. c4 table PF .5 -> .35-.4, or OA multiplier .75):
  - OAI cannot measure the level effect: its Raptor stand-in (R6) is the strongest OAI member, so OAI's own optimum is m ~1.5
    before and ~1.0-1.25 after E-OA (`box_F3.log` mgrid). Only the interaction transfers.
  - If c4's Raptor table was at the public optimum before E-OA (pf10 PF .5 -> .1 read below c4, 10-19, CLAIMED PREREG; HW .6 and
    c4 .5 were both in positive packages), the interaction moves the PF optimum down by ~.1-.2 share and is worth about +0.5 to
    +2.5e-4 macro (MODEL: OAI PF interaction -.0275 PF-AUC per unit share at a .3, curvature from the pf10 read).
  - P(>0) ~.55-.6 (GUESS): the sign is robust, the size rests on a gold58-derived gap difference (sd .1) and an assumed optimum.
  - It is a public-tuned-table move, the family whose gold58 calls failed on 10-19. **WEAK, not a slot.** Additive with convex
    within +-.0003 OA-mean (`box_F4.log`).

## 3. Family table (expected hidden macro is vs the A21 winner shape)

| # | family | status before today | ruler / measurement (MEASURED unless tagged) | control | E private | P(>0) | verdict |
|---|---|---|---|---|---|---|---|
| 1 | **Convex subtraction** U_o ** p (p 1.5-3, hinge) | new | OAI 65 / 65 cells, mean +.0004 OA-mean; held-out +.00023 to +.00038 (P .78-.90) in 5 blends; boot +.00033 to +.00040 (P .70-.88) (`box_F2.log`, `box_F3.log`, `f3_summary.out`) | report OOF: convex removes more report coupling (-.00034, P>0 .04); gold74 report vs expert at k 2: .88 vs .64; gold58 INFO -.0005 to -.0008 (P .15-.26) | +0.5 to +1.5e-4 | ~.6 | **PROMISING (optional ticket)** |
| 2 | Raptor OA weight UP after E-OA | new (asked) | interaction negative in 7 / 7 role-matched configs (sec 2.1) | reversed controls flip (P 1.00) | < 0 | ~.25 | **DEAD** |
| 3 | Raptor PF / OA weight DOWN after E-OA | new | same interaction; level effect not measurable on OAI | LB pf10 (pre-E-OA) negative | +0.5 to +2.5e-4 if c4's table was optimal pre-E-OA | ~.55-.6 | WEAK |
| 4 | S6fall OA weight up after E-OA | S6fall weight CLOSED (global) | S6-slot interaction +.0012 to +.0024 (P 1.00); level at a .3: R1 +.0006-.0009 OA-mean, R3p_lo ~0 (`box_F4.log`) | reversed ~0; LB hw3 (.3 global) negative pre-E-OA | 0 to +1e-4 | ~.55 | WEAK |
| 5 | SUB_HIGH: subtract only the R6 / S6fall part's OA pcts | new | +.00026 / +.00052 / +.00027 OA-mean vs FINAL (P .61-.81) | reversed -.00002; SUB_LOW (subtract Raptor's) -.0006 to -.0007 | ~+0.5e-4 | ~.55 | WEAK |
| 6 | Regime-aware per-member E-OA (decorrelate only high-gap members) | MEMBER placement CLOSED (equal a) | -.00059 / -.00030 / -.00012 vs FINAL (P .14-.41) | reversed -.00003 | < 0 | ~.3 | DEAD |
| 7 | Probit (normal-score) space E-OA | new | -.00026 / -.00006 / -.00021 vs FINAL | reversed -.00005 | < 0 | ~.3 | DEAD |
| 8 | Agreement-gated E-OA (stronger when members agree / disagree) | new | agree -.0003 to -.0011 (P .04-.35); disagree -.0006 to +.00015 | | < 0 | ~.3 | DEAD |
| 9 | Concave subtraction (sqrt) | new | -.0003 to -.0005 vs FINAL | mechanism-consistent mirror of #1 | < 0 | ~.2 | DEAD |
| 10 | Cross-member: subtract Raptor's own OA outputs | new | SUB_LOW -.0006 to -.0007 (P .01-.24) | | < 0 | ~.2 | DEAD |
| 11 | Rank vs score mixing (E1 pow2, prob / logit / geo / median) | CLOSED | no new evidence; E-OA x E1 interaction ~0 (CLAIMED mechanics.md) | | | | CLOSED |
| 12 | Per-label member weights on non-OA labels | CLOSED (A19) | E-OA leaves the 9 other columns byte-identical, so there is no interaction and no new evidence | | | | CLOSED |
| 13 | Own-label member-disagreement terms | CLOSED (expert2 3.1 KILL) | not re-tested | | | | CLOSED |

Notes on the table:
- Rows 5-10 are paired bootstraps (B 200, R1 knee x target) of each family at its own best a vs FINAL E-OA at its best a, in
  configs pair R6 / D5, pair S6 / D5, c4 nest R6 / D5 / s1044 and the reversed nest (`box_F2.log`).
- OAI was pessimistic about E-OA's own size (it said ~0 at a .2; public gave +11.3e-4, CLAIMED implied.md). Relative
  comparisons between shapes are what is used here. R3p_lo (-1 sd label coupling) gives blend gaps .12-.14, the public-implied
  c4 gap (.127, CLAIMED implied.md), and is reported next to R1 throughout.

## 4. Convex subtraction: measurements

### 4.1 OAI (MEASURED `box_F3.log`, `f3_summary.out`)

env(p2) - env(lin), OA-mean, each at its own best a on the 21-point grid 0-.5:

| proxy / scheme | mean | > 0 | note |
|---|---|---|---|
| P_FT R1 | +.00027 | 9 / 12 | the 3 zeros: R6, S6, R6 + S6 alone, where no E-OA helps at R1 |
| P_FT R3p | +.00023 | 9 / 12 | |
| P_FT R3p_lo (public-like gap) | +.00044 | 12 / 12 | |
| P_FT R3p_hi | +.00002 | 4 / 12 | E-OA itself fails here; convex never loses |
| P_AREA R1 / R3p / R3p_lo / R3p_hi | +.00012 / +.00044 / +.00056 / +.00017 | 9, 11, 12, 9 / 12 | |

- Exponents: p1.5 +.00026, p2 +.00040, p3 +.00045, hinge .3 +.00025, hinge .5 +.00034 (65 cells where linear E-OA gains > .0002).
  p2 is the middle choice; p3 adds little and is riskier at the top end.
- convex's best a is .025-.05 below linear's (R3p_lo: linear .20-.225, p2 .175-.20).
- Held-out a selection (20 stratified split-halves x 2, R1 P_FT), p2 minus linear: N1 +.00023 (P .80), N3 +.00038 (.90), fleet
  N5 +.00025 (.80), FLAT3 +.00026 (.78), single D5 +.00050 (.85); single R6 -.00010 (R6 alone gains nothing from any E-OA at R1).
- At the in-flight strengths with the A21 pair weights, R3p_lo: pw-p2 at a .2 beats pw-linear at its best a (.25) by +.0003 to
  +.0009 OA-mean in 4 / 4 blends. At a .3 the two are equal (convex is past its optimum there).

### 4.2 Public-size read noise and same-a twins (MODEL on MEASURED `box_F5.log`, `box_F5_R3plo.log`; 400 draws of 390 knees)

| twin (macro) | R1 population | R3p_lo population | paired sd |
|---|---|---|---|
| pw20p2 - pw20 | -0.6 to +0.9e-4 | **+1.1 to +2.2e-4** (P .68-.86) | 2.1-2.4e-4 |
| eoa20p2 - eoa20 | -0.6 to +0.4e-4 | +0.7 to +1.6e-4 | 2.1-2.3e-4 |
| pw25p2 - pw25 | -1.9 to 0e-4 | +0.2 to +1.5e-4 | 2.7-3.0e-4 |
| pw25p2 - pw30 | +3.0 to +4.2e-4 | +2.2 to +3.0e-4 | 2.5-2.8e-4 |

- A public read of a convex twin is about a 0.5-1 sd signal: it can move the best-id, but it cannot settle the question.
- The gain appears where E-OA itself is large (public-like gap); at gold74-level gap it is ~0. The public reads say c4 sits at
  the high-gap end (+11.3e-4 at a .2), which is why E private is put at +0.5 to +1.5e-4 rather than ~0.

### 4.3 RSNA side (MEASURED `local_space.log`, `local_space2.log`)

- Matched gold74 (70 knees, same studies): P(Y_t = 1 | k other compartments positive, same labeller):

  | k | expert | report |
  |---|---|---|
  | 0 | .15 (18 / 117) | .19 (19 / 100) |
  | 1 | .40 (24 / 60) | .27 (14 / 52) |
  | 2 | .64 (21 / 33) | .88 (51 / 58) |

  Report false-positive rate by k (expert-negative compartments): .11 / .31 / .58; false-negative .22 / .17 / .00. The
  report regime's excess is concentrated where both other compartments are diseased ("tricompartmental" over-call).
- Report labels on 4,333 train studies (minus gold74): P(Y_t | k report-positive others) .16 / .41 / .74.
- Report-OOF ruler (R7 OOF x report labels, mechanism control): every E-OA form loses on report labels (linear -.0044 at a .2,
  p2 -.0047, sqrt -.0030); p2 - linear at a .2 = -.00034 (sd .0002, P>0 .04). Convex removes more of exactly the coupling report
  labels contain, as the mechanism says.
- gold58 c4 replay, INFO ONLY (not a ruler): p2 vs linear at a .2 -.00079 macro (P>0 .15), p1.5 -.00058 (.21), pw p2 vs pw .2
  -.00051 (.26). Per label: Medial -.003, PF -.005, Lateral 0 at a .2 sym (2-4 pair swaps). gold58 also overstated linear E-OA
  (+.0025 vs public +.0011), so I count it as a weak contrary sign, not a veto.

## 5. If Eesh wants one upside ticket (NOT on the default slate; needs his go + a PREREG amendment)

PUSH_1020's default stays: one entry, R6e20, 05:31 IST 10-21. Convex replaces E1 as the better optional ticket (E1 has only the
report-OOF ruler; convex has the independent OAI ruler, a held-out test and a mechanism). Never stack it with E1 or with a
Raptor-table move. One ticket only.

| A21 outcome | convex twin | change vs the winner | kernel (sha) | commit gate args |
|---|---|---|---|---|
| pw20 best | `levers/kernel_v224r6c4pw20p2j` | exponent only | 5e0cf2ca | `--aoa 0.2 --mode pw --p 2` |
| pw30 best | `levers/kernel_v224r6c4pw25p2j` | exponent + a .3 -> .25 (convex's optimum is lower; at a .3 the twin is ~equal) | 9750dfbd | `--aoa 0.25 --mode pw --p 2` |
| eoa20 stays | `levers/kernel_v224r6c4eoa20p2j` | exponent only | 2d51de83 | `--aoa 0.2 --mode sym --p 2` |
| eoa30 best | not built: `python levers/build_convex.py sym 0.25 2` and add it to the gate | exponent + a | | `--aoa 0.25 --mode sym --p 2` |

- **Offline gate** `gate_convex_offline.py` -> `gate_convex_offline.log`: **CONVEX_OFFLINE_GATE PASS**.
  - The P 1 controls are byte-identical to the scored A20 eoa20 output and the A21 pw30 output.
  - Each p2 kernel's OA columns equal an independent recompute (maxdiff 2.8e-8); non-OA columns identical to c4; receipts
    src = c4 b47c29c3, p_oa 2. gold58 macro (INFO): pw20p2 .94328, pw25p2 .94291, eoa20p2 .94286 (pw20 .94378, eoa20 .94360).
- **Commit gate** `probe_check_convex.py <OUT> --a224 0.25 --table '<C4T>' --aoa A --mode M --p 2` must print **PASS 18/18**
  (a21 gate + check (d3) receipt p_oa). Self-test: the A21 pw20 output with `--p 1` PASS 18/18 (the original a21 gate gives
  17/17); with `--p 2` FAIL 17/18 at (d3). On the 3 visible studies the p2 and linear recomputes can coincide, so (d3) (the
  receipt) is the check that tells them apart. A tick must key on "PASS 18/18".
- **Do NOT push** `kernel_v224r6c4pw30p1j` (b865439d) or `kernel_v224r6c4eoa20p1j` (b1f1a5cc): they are P 1 controls whose
  output equals already-scored entries.
- Value (MODEL): E private +0.5 to +1.5e-4 vs the winner, P(>0) ~.6 (GUESS). Public twin read sd ~2.2e-4 (sec 4.2), so R1
  switches A on noise often; P(the switch has the wrong sign) is roughly .35-.4 (GUESS by analogy with e1_option.log).
- One A21 read helps calibrate this: pw vs sym is itself an OAI-derived shape refinement. If pw20 / pw30 read above eoa20 /
  eoa30, the OAI shape ruler has a public hit and P(convex > linear) moves to ~.65-.7; if sym reads above pw, to ~.5 (GUESS).

## 6. Files (all mine; session_2026-10-20/levers/)

- `space.md`: this note.
- **Box (aggregates only), `box_space.py`**: JOB F1 (Raptor-slot weight x E-OA), F2 (10 transform families vs FINAL), F3 (convex
  robustness, held-out, in-flight strengths, Raptor m grid), F4 (S6-slot weight x E-OA, additivity), F5 (390-knee twin noise;
  3rd arg = scheme). Logs: `box_F1.log`, `box_F1_smoke.log`, `box_F2.log`, `box_F2_smoke.log`, `box_F3.log`, `box_F3_smoke.log`,
  `box_F4.log`, `box_F5.log` (R1), `box_F5_R3plo.log`; summary `f3_summary.out`.
- **Local (competition data only)**: `local_space.py` -> `local_space.log` (gold58 member coupling, report-OOF control, gold74
  report vs expert, gold58 INFO replay); `local_space2.py` -> `local_space2.log` (gold58 INFO paired bootstraps, matched gold74
  table); `local_pairs.log` (label-free per-pair coupling bootstrap).
- **Builds**: `build_convex.py` (build_a21.py + P_OA), `build_receipt_v224r6c4{pw20p2,pw25p2,eoa20p2,pw30p1,eoa20p1}j.json`,
  `kernel_v224r6c4{pw20p2,pw25p2,eoa20p2}j/` (candidates), `kernel_v224r6c4{pw30p1,eoa20p1}j/` (controls, never push),
  `gate_convex_offline.py` / `.log`, `probe_check_convex.py`.

## 7. Caveats

- OAI members are stand-ins: R6 / S6fall served from a rebuilt 320 px store with DESS coronal / axial inputs, and their OA
  coupling collapses on OAI (CLAIMED r6_oai.md 11.2). The role mapping in sec 2 uses that collapse; the fleet-only configs
  (N5, FLAT3, single fleet models) carry the convex result without it.
- Raptor cannot be measured on OAI (OAI-trained). Its RSNA role rests on gold58 (n 58): the label-free rho comparison is clean
  but noisy; the gap comparison uses gold58 labels (INFO).
- The convex family was found by searching 9 families on OAI (F2) and then tested on new configs, proxies, schemes and
  held-out halves (F3). The search itself is a selection step; the held-out numbers and the report-structure mechanism are the
  protection against it, not a proof.
- Nothing here changes the picks rule: A = API best-id under R1, B = R6e20 if Eesh approves it, else R6 57248084.
