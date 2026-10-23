# sink.md: kitchen-sink and conservative stacks on each plausible A22 winner (10-21 ~09:40 to ~10:15 IST, Claude, SINK track)

RESEARCH + LOCAL BUILDS ONLY. Nothing was submitted, pushed, committed, versioned, registered or downloaded. No dataset, no GPU job,
no scheduled-task edit. PREREG, HANDOFF, PROGRESS and the live folders (session_2026-10-20/val incl. a22/, stack/, a20, a21) were
only read. The real committed outputs of pw30 and of the four A22 commits (val/a22/out_*) were read and copied into sims, never
changed. Everything written by this track is under `session_2026-10-21/levers/`: `sink*`, `box_sink*`, `build_sink_kernels.py`,
`make_sink_gate.py` and `kernels/`. The `regime*`, `rg_lib.py` and `oai_claimed*` files in the same folder belong to a sibling
REGIME track; this track did not read them for its results or touch them.

**OAI governance.** Box use = one read-only CPU script piped over ssh stdin into `python -I -` (`box_sink.py`: a B 2 smoke, then
B 60), printing aggregates only (means / sd / P of macro AUC deltas over weighted full-size resamples). Nothing was written on the
box, no OAI CSV was printed, no id / row / per-knee value left the box. The only data sent were the gold74 2x2 counts already used
by stack/box_stack.py. Scan of `levers/` (all files, incl. sims): 0 hits for `oai_[0-9]+_[LR]` / `oai_[0-9]{4,}`. No GPU used.

**Tags.** MEASURED = computed in this track, log named. CLAIMED = older note, named. MODEL = arithmetic on stated inputs
(`sink_value_mc.py`). GUESS = judgement. Units: e-4 = 1e-4 macro AUC vs pw30 57425970 unless stated. Intervals [p10, p90].
gold58 (G58) numbers are INFO only, never a selection ruler.

## 0. Bottom line

1. **Only one new component survives the skeptic-aware screen: CVX** = the convex E-OA exponent p 2 on the subtracted
   other-compartment pct, at the winner's own dose (levers/build_convex.py spec). Every other near-miss is out: a skeptic or
   this round's rulers found it negative or zero (sec 1). MCLmm (skeptic E -0.10) is kept inside the A22 entries that already
   carry it, but it is not added to any new stack.
2. **New ruler: RSNA train OOF raked to the public-implied regime (RAK-PUB).** The OA-trio shift that best fits the five public
   E-OA reads is s* = +0.23 sd of the gold74 log OR. At s*, with no free factor, the c4-shaped nests give eoa10 / eoa20 / eoa30 /
   pw20 / pw30 = 5.2-6.1 / 9.2-9.9 / 8.4-10.0 / 10.9-11.8 / 10.9-12.6 vs public 5.9 / 10.3 / 9.6 / 10.4 / 11.3 (SSE 1.6-4.1,
   MEASURED `sink_local.log`). It is the best-calibrated ruler in the project so far.
   - On this ruler the public reads do NOT imply hidden labels much less coupled than gold74. The fitted shift is -0.2 to +0.6
     sd across the four target constructions (`sink_cvx_skep.log`). The "much less coupled" reading is specific to the OAI
     rulers, whose members co-fire less on OAI images (consistent with night/recal.md: the gap excess is mostly model-side).
   - RAK-PUB values of the A22 components (MEASURED, INFO for the 10-21 reads): pw25 +0.35 / +0.95 / +0.80 (three nests), lomm05 on
     pw25 +0.76..+1.19, MCLmm +3.15..+3.50 (MCL column is not OA-regime dependent; the gold74 MCL basis is 11 knees).
3. **CVX evidence (pw25-family base = pw25 / spl / salt / swide):**
   - RAK-PUB +1.64 (pw25) / +1.49 (spl), positive in 3 / 3 nests (row-bootstrap P .73-.95). Dirichlet target uncertainty (RAK-u):
     +1.04 / +2.71 / +1.83, P .68 / .91 / .79.
   - It does not ride on a gold74 3-way quirk: the gold trio all-three-positive count equals its pairwise max-entropy prediction
     (7 vs 7.0). With pairwise-only targets it holds at +1.20; with gold58-only targets at +0.78 / +1.09. Leave-one-gold-study-out
     74 / 74 > 0 (min +0.50).
   - At matched decorrelation (co-firing) it still gains (+1.0 / +2.4 / +1.8 over linear). The placebo direction (concave p .5)
     loses (-3.1 to -4.2), and the report-label control loses (-1.2 to -1.7), as an expert-regime transform must.
   - OAI public-regime cells split by proxy: P_AREA +2.68 (5 / 5 nests, the best-calibrated OAI cells), P_FT +0.50 (4 / 5).
     At gold74 coupling P_FT gives -1.58 (0 / 5) and P_AREA +3.46.
   - **gold58 says the opposite: -7.7 on pw25 (P .15), -12.6 on spl (P .04).** At the E-OA calibration x .42 that is -3.2 / -5.3
     public-equivalent. GUESS: part of this is in-sample. The pw pair weights were fitted on the gold74 coupling gaps (CLAIMED
     a21/build_a21.py), and G58 also penalises the concave placebo (-4.4).
   - Skeptic-aware prior used below: N(+0.5, 1.0) on the pw25 family, N(+0.2, 1.1) on pw30. The levers skeptic prior is
     N(+0.2, .8) / N(-0.1, .9).
4. **Interactions** (sec 2.4):
   - CVX x MCLmm is exactly additive (disjoint columns).
   - CVX x lomm05 is a mild overlap on Lateral OA: RAK-PUB -0.06 to -0.20, OAI pub -0.08 (P_AREA) / -0.26 (P_FT). G58 is the
     outlier: -4.8, CVX erases lomm05's gold58 gain.
   - CVX x the dose step .3 -> .25 is super-additive on RAK-PUB (+0.6 to +1.1). CVX is weak on the pw30 base: RAK-PUB +0.75,
     pairwise and gold58-only targets ~0, OAI P_FT pub -0.76.
   - CVX x SUBH is a small overlap (-0.3..0). No sign flip among survivors.
5. **Stacks (sec 3)**, all built from the winner's own notebook (scored pw30 or the A22 notebook exactly as registered) plus the CVX
   edit:
   - **Kitchen sink (KS) = winner + lomm05 (if absent) + CVX.**
   - **Conservative (CONS) = winner + CVX.**
   - For spl and swide the two coincide, because there is only one survivor left to add.
   - Offline gate: **SINK_KGATE PASS (165 / 165)**. It covers the full on/off factorial; all-off is byte-identical to scored pw30,
     and CVX-off is byte-identical (output and receipt) to the base A22 entry.
   - Commit gate: `kernels/sink_commit_gate.py` (27 checks). It prints PASS 27/27 on the REAL scored pw30 output and on all four
     REAL A22 commits. **SINK_KSIM_GATE_SELFTEST PASS (43 / 43 rows as expected).** REGISTRY_SINK_CHECK PASS 18/18.
6. **Value of a 10-22 entry (MODEL, sec 4; increment of E private of A under R1, e-4):**

   | CVX prior | CONS | KS | both | P(.960 shown), A22 alone 0.66 -> CONS |
   |---|---|---|---|---|
   | ruler N(+1.5, .9) | +1.00 | +0.75 | +1.01 | 0.84 |
   | **primary N(+0.5, 1.0)** | **+0.34** | +0.20 | +0.34 | 0.80 |
   | levers skeptic N(+0.2, .8) | +0.13 | +0.02 | +0.13 | 0.79 |
   | zero | 0.00 | -0.07 | 0.00 | 0.78 |
   | gold58-led N(-1.0, 1.0) | -0.31 | -0.33 | -0.31 | 0.74 |

   - Each row is flat (within +-0.01) across MCLmm priors (skep2 / stack / zero) and lomm05 priors (stack / skeptic).
   - **KS never beats CONS.** In the rows where KS differs (W = pw30 / pw25 / salt), it re-adds lomm05, whose head-to-head read
     already lost.
   - **Both** = CONS: KS adds nothing next to it.
   - P(A < pw30) moves .22 -> .24 (primary), .27 (zero) and .30 (gold58-led).
   - The P(.960) lift is mostly read noise: the zero prior still gets +.12, because CVX's public read sd is 2.6-3.4 (MEASURED).
7. **Recommendation (GUESS on MODEL):**
   - **Never the kitchen sink.**
   - At most ONE optional 10-22 lottery slot: CONS(W) for the 10-21 best-id W. It ranks after an r6pw20 ERROR retry.
   - Its expected value is +0.1 to +0.3e-4 under the skeptic-aware priors (about +0.05 to +0.17 silver pts at the GUESS slope
     .5 pt per 1e-4), negative only if gold58 is right.
   - That is below public read noise. A .960 it shows proves nothing.
   - If Eesh adopts readtable's H1 / H2 private-first addendum (A = pw20 when W = pw30 / pw25), CONS has a role only for
     W in {spl, swide, salt}, and the addendum must say how a cvx entry is treated.
   - Default stays as readtable.md: nothing on 10-22. This file only adds the single positive-expectation option, with its caveats.

## 1. Near-miss inventory under a skeptic-aware prior (step 1)

Sources: stack/inventory.md, stack/build.md sec 8, levers/LEVERS_1020.md, session_2026-10-19/ex (EX_1019 via NIGHT / recal),
more/MORE_1020.md, night/NIGHT_1020.md (all CLAIMED); this round's rulers (MEASURED: `sink_summ.out`, `box_sink_summ.out`,
`sink_cvx_skep.log`, `sink_cvx_matched.log`). Ruler cells give e-4 on the pw25 base unless stated: RAK-PUB = mean of 3 nests
[min, max]; OAI pub = mean of 5 nests, P_AREA / P_FT; REP = report-label control; G58 point (boot P).

| component | exact spec (cell 18 unless stated) | prior verdict (CLAIMED) | RAK-PUB | OAI pub P_AREA / P_FT | REP | G58 | skeptic-aware E | verdict |
|---|---|---|---|---|---|---|---|---|
| pw25 | `A_OA = 0.3 -> 0.25` | KEEP low value; implied2 private +0.56 | +0.70 vs pw30 [+0.35, +0.95] | +1.0 / +2.6 (vs pw30) | +5.2 | +3.2 | +0.4 (inventory MODEL) | in A22 (read pending) |
| lomm05 | `LOA = pct(pct(LOA) - .05 pct(MM))` after E-OA | lottery; skeptic +0.17..+0.28 | +1.02 [+0.76, +1.19] | +0.11 / +0.52 | -0.1 | +4.8 | +0.17 (stack) / 0 (skeptic) | in A22 (spl, swide); KS only |
| MCLmm | `MCL = pct(pct(MCL) - .1 pct(MM))` after E-OA | skep2 -0.10, P .46 | +3.2 (not OA-regime) | n/a (no MCL on OAI) | -3.2 | +7.6 | -0.10 (skep2) | stays in A22 swide / salt; **not added** (skeptic negative) |
| **CVX** | E-OA subtracts `_U[o] ** 2.0` (P_OA 2) instead of `_U[o]`, own term `_U[c]` kept; at the base dose | WEAK, ~+0.2 (LEVERS sec 2.1) | **+1.64 [+0.99, +2.18]** (spl +1.49; pw30 +0.75) | **+2.68 / +0.50** (pw30 +2.85 / -0.76) | -1.4 (loses: OK) | -7.7 (P .15) | **+0.5 [pw25 family] / +0.2 [pw30] (primary); +0.2 / -0.1 (levers skeptic)** | **SURVIVES: the only new component** |
| CVXd | p 2 at base dose - .05 (= levers "twin") | WEAK | +1.43 (on pw25) / +2.34 (on pw30, = pw25 + CVX) | +1.80 / +2.29 | +3.2 (reports like it: partly "less decorrelation") | -4.1 | below CVX on the pw25 family | OUT (an alternative to CVX, not additive) |
| SUBH | E-OA subtracts `pct(2/3 pct(R6) + 1/3 pct(S6fall))` of the other compartments | WEAK; < 0 next to pw25 | -0.14 [-1.32, +1.38] | +0.18 / +0.62 | +0.1 | +1.3 | ~0 | OUT |
| S6up | W_S6F .2 -> .3 on the OA trio | WEAK knob; LB hw3 negative | +2.0 [-3.1, +11.1] (role artefact) | -1.1 (P_FT) | +2.0 | +5.9 | ~0 | OUT |
| RPF | cell 14 `W_RAPTOR_C['PF OA'] .5 -> .4` | WEAK knob; R1 -0.45..+0.06 | -0.25 [-2.9, +2.3] | -1.2 (P_FT) | -0.2 | +3.2 | <= 0 | OUT |
| E1 | soft-OR, member pct ** 2 (cells 14 + 18) | DEAD (expert ruler negative) | +3.7 (carried by unraked report labels) | +0.2 / -0.6 | +3.3 | -2.7 | <= 0 | OUT |
| BOR (C13) | MOA += .1 MM, LOA += .1 LM | OUT (RAK negative) | -2.6 [-2.9, -2.4] | +5.7 / +1.4 (weak OAI heads) | -2.9 | +1.9 | < 0 | OUT |
| C6' | LM <- MM .05 | skeptic KILL | -0.44 (P .03-.06) | -1.25 / -1.20 (0 / 5) | -0.4 | +1.6 | < 0 | OUT (kill confirmed) |
| C14 | LOA += .1 LM | skeptic KILL | -0.71 [-0.82, -0.51] | +2.6 / +1.7 | -1.6 | +3.2 | < 0 (skeptic) | OUT |
| killed list: E2s, C6, fluid, ACL, global, LP shapes, doses >= .35, convex-search variants (p3, hinge, probit), T-STATE (needs an R6 state dump, not buildable in cell 18), T-VOL, MCL <- other partners / MCLmm re-dose | as their sources | KILL / forking | not re-measured | | | | <= 0 or not buildable | OUT |

## 2. Rulers and measurements (step 2)

### 2.1 Rulers (MEASURED)
- **G58**: gold58 c4 kernel path from member serve files. Parity exact: c4 .94113, pw30 .94336, pw25 .94368, pw30 + lomm05 .94384,
  spl .94417, swide .94492, salt .94444, pw25p2 .94291, pw20p2 .94328 (`sink_local.log`).
- **RAK** (stack/inventory.md construction), three c4-shaped nests on 3,467 non-gold train studies. OA columns are raked to the
  gold74 32-cell (MOA, LOA, PF, MM, LM) joint; MCL to the (MCL, MM) 2x2. Regimes:
  - g74 = gold74 couplings;
  - **PUB** = trio pair log ORs shifted to s* = +0.23 sd (level fit to the 5 public reads);
  - PUBM = PUB with the OA-meniscus pairs also shifted;
  - s-0.83 = the OAI "pub" shift, for contrast.
- **Regime scan** (`sink_local.log`, mean of nests, SSE vs public):

  | s | -0.83 | -0.4 | 0 | +0.4 | +0.8 |
  |---|---|---|---|---|---|
  | SSE | 846 | 278 | 36.5 | 21.2 | 185 |

  - s +0.23 gives SSE 1.6 / 3.5 / 4.1 per nest.
  - Shape: the ruler cannot hold both eoa30 < eoa20 and pw30 > pw20 at once. At s* both are near 0 (public -0.7 / +0.9).
- **Target robustness** (`sink_cvx_skep.log`), each target with its own fitted s*: T32 +0.2 (SSE 2.2), pairwise-only T2W -0.2
  (1.0), gold58-only T58 +0.6 (1.2), T58_2W 0.0 (6.2).
- **REP**: raw report labels, the mechanism control. Ladder -0.2 / -3.3 / -11.3 / -3.4 / -12.1: decorrelation loses by design.
- **OAI** (`box_sink.log`, `box_sink_summ.out`): R6 + S6fall served on OAI + fleet D5 / s1044 / s1045, five c4-shaped nests,
  P_FT / P_AREA, regimes g74 / pub (-0.83) / raw.
  - P_AREA pub ladder pw30 9.7-14.4 is the best-calibrated OAI cell.
  - P_FT pub over-penalises a .3 (pw30 1.0-9.1).

### 2.2 Singles on each winner base (mean of nests; full tables `sink_summ.out` sec "marginal", `box_sink_summ.out`)
| extra | base pw30: RAK-PUB / OAI P_AREA pub / OAI P_FT pub / G58 | base pw25 | base spl (= swide) | base salt (= pw25 for OA) |
|---|---|---|---|---|
| CVX | +0.75 / +2.85 / -0.76 / -6.6 | +1.64 / +2.68 / +0.50 / -7.7 | +1.49 / +2.60 / +0.24 / -12.6 | +1.64 / (= pw25) / -7.7 |
| CVXd | +2.34 / +3.68 / +3.11 / -4.5 | +1.43 / +1.80 / +2.29 / -4.1 | +1.56 / +1.90 / +2.30 / -5.7 | +1.43 |
| SUBH | -0.16 / +0.31 / +1.00 / +2.9 | -0.14 / +0.18 / +0.62 / +1.3 | -0.01 / +0.10 / +0.56 / +2.9 | -0.14 |

### 2.3 CVX skeptic (`sink_cvx_skep.log`, `sink_cvx_matched.log`)
- Gold trio 3-way: all-three-positive g74 7 vs pairwise max-ent 7.0; g58 5 vs 4.8. Train reports: P(LOA+ | MOA+, PF+) .74 vs
  P(LOA+ | MOA+) .67 (g74 .47 vs .43).
- Target variants, CVX on pw25 / spl / pw30: T32 +1.69 / +1.54 / +0.81; T2W +1.20 / +0.96 / +0.02; T58 +0.78 / +0.63 / -0.34;
  T58_2W +1.09 / +0.87 / -0.05.
- Exponent shape on pw25 (T32): p 1.5 +1.20, p 2 +1.69, p 3 +1.03. Concave p .5: -3.1 / -4.2 / -3.3 (RAK-PUB), -4.4 (G58).
- RAK-u (Dirichlet target x row bootstrap, B 150 per nest): CVX on pw25 +1.04 / +2.71 / +1.83 (P .68 / .91 / .79). On spl
  +0.83 / +2.61 / +1.71. On pw30 +0.19 / +2.34 / +0.90.
- LOO over the 74 gold studies (T32, N_A): full +1.03, min +0.50, max +1.74, 74 / 74 > 0. The extra16 report-read rows contribute
  +0.19.
- Matched decorrelation: the label-free trio co-firing of p 2 @ .25 equals linear @ .24-.25. p 2 minus matched linear:
  +0.99 / +2.37 / +1.76 (RAK-PUB), -5.80 (G58).
  - This contradicts the levers' RSNA INFO direction on the RSNA-OOF ruler. It agrees on gold58.
  - Not re-run: the levers' R7 x gold74 read (n 74, CLAIMED negative).
- Read noise (`sink_read_noise.log`, R7 OOF, 1,000 subsets): CVX public / private sd 2.59 / 1.49 (pw25 family), 3.37 / 1.93
  (pw30). Corr with the pw25 read -.38, with the lomm05 read +.14, cvx25 ~ cvx30 +.93.

### 2.4 Pairwise interactions among survivors (I = d(A+B) - d(A) - d(B), e-4)
| pair | columns | RAK-PUB (3 nests) | OAI pub P_AREA / P_FT | G58 | REP | class |
|---|---|---|---|---|---|---|
| CVX x lomm05 | both write LOA | -0.20 / -0.06 / -0.19 | -0.08 / -0.26 | **-4.84** | -0.17..-0.32 | overlap (mild on rulers, total on G58) |
| CVX x MCLmm | disjoint | exactly 0 | n/a | exactly 0 | 0 | exact add |
| CVX x dose (.3 -> .25) | OA trio | +0.99 / +0.58 / +1.11 | -0.17 / +1.26 | -1.2 | +2.0 | super-additive on RAK (CVX belongs on the pw25 family) |
| CVX x SUBH | OA trio | -0.29 (pw30) .. -0.01 (pw25) | -0.26..-0.31 / -0.12..+0.05 | -2.2..+4.6 | ~0 | small overlap (SUBH is out anyway) |
| lomm05 x MCLmm | disjoint | 0 | n/a | 0 | 0 | exact add (CLAIMED inventory) |
| pw25 x lomm05 | LOA | ~0 here (LOMM +0.76..+1.19 on pw25) | +0.24 (CLAIMED) | 0 | +0.2..+0.3 | mildly super-additive (CLAIMED) |

No cancellation among the survivors. The only overlap that could matter is CVX x lomm05, and only gold58 says it is large.

## 3. Stacks per winner (step 3): definitions, builds, gates

### 3.1 Definitions (KS = winner + every surviving component it lacks: lomm05, CVX; CONS = winner + the top survivor, CVX)
| A22 winner W | KS | CONS | note |
|---|---|---|---|
| none (pw30 stays) | `cvx_lomm30` = pw30 + lomm05 + CVX | `cvx_pw30` = pw30 + CVX | CVX weakest on this base (primary +0.2) |
| pw25 | `cvx_spl` = pw25 + lomm05 + CVX | `cvx_pw25` = pw25 + CVX | W = pw25 means spl (lomm05) already lost |
| spl | `cvx_spl` | `cvx_spl` | KS = CONS (MCLmm excluded) |
| salt | `cvx_swide` = salt + lomm05 + CVX | `cvx_salt` = salt + CVX | W = salt means swide (lomm05) already lost |
| swide | `cvx_swide` | `cvx_swide` | KS = CONS |

### 3.2 Builds (`build_sink_kernels.py` -> `kernels/build_sink_kernels.log`, receipts `kernels/build_receipt_<tag>.json`, `kernels/build_index.json`)
Edit (cell 18 only, inside the E-OA block, 4 lines added + 2 lines edited):
- `P_OA = 2.0` after `_OA = [...]`;
- `_S = _U if P_OA == 1 else {c: _U[c] ** P_OA for c in _OA}` after `_U = ...`;
- `_U[_o[k]] -> _S[_o[k]]` in both subtraction lines;
- after the EOA print: `if P_OA != 1: rec['p_oa'] = P_OA; print('CVX_APPLIED p_oa %g' % P_OA)`.

With P_OA 1 the arithmetic, receipt and stdout are the base's. The builder asserts:
- base nb sha (registry / receipts), metadata sha, and a byte-exact JSON round trip;
- only cell 18 differs from the base;
- the cell-18 diff is exactly the 6 expected lines;
- metadata differs only in id / title / code_file.

| build | tag / Kaggle id `eeshsaxena/rsna-sub-<tag>` | base (nb sha8) | nb sha256 | meta sha256 | role |
|---|---|---|---|---|---|
| cvx_pw30 | v224r6c4pw30cvx2j | pw30 scored 57425970 (5a903187) | f3203107c8b97aa9cc654bdf7924fb48bb6c9dbeb51c2aed08feb2c6853de3a2 | 9f1e8efd0fc6f53012bf8f98aae21f5164d8cf9e686078b8a82ab1edc4dab46c | CONS(pw30) |
| cvx_lomm30 | v224r6c4pw30lomm05cvx2j | night lomm05 (c0dd7ec8, never pushed) | 268938cef52c906a091be889e11700b5a23bd4c916bb1c3d188ed3680a26dfd1 | d770e064e884029f8993d8c5b24013cd7dc34d69eb24c54cd22d96743499035d | KS(pw30) |
| cvx_pw25 | v224r6c4pw25cvx2j | pw25 A22 57463742 (64ed7eba) | 015598db1998f135c27542778257e4618bb0b4b5942d3a51118a2b8365e980cc | 48e407034517fc3baf3bad2dc707edfb716d67800e07a17b12374736d3148c62 | CONS(pw25) |
| cvx_spl | v224r6c4pw25lomm05mcl00cvx2j | spl A22 57464296 (e641e88f) | 5b4bbd791aba590481532edb872603197a861a073d98b993078c2de9eb2029b9 | 1a0229c169156a366ce88a319e6d12a47ebd775704549115f9f885d3285e4a0a | KS(pw25) = KS / CONS(spl) |
| cvx_salt | v224r6c4pw25mcl10cvx2j | salt A22 (051232e4) | ebeb31f8bd47565116eb49cc4b3123d1d20e8bcb68e412743b73b9710eb5d436 | f8d164ba0df380736b448be2e1383d5a76d27b5bb5a706e99b0c16b4519697d9 | CONS(salt) |
| cvx_swide | v224r6c4pw25lomm05mcl10cvx2j | swide A22 57464326 (c9801a80) | b6915d970693b7fad46ce97718e69c74a6ffb8c6e2a70d0eed87aaba3414993b | 188a511c2752a365b0d5eb74aaddbe4775f709ae213c06d8ff0b73c9c48f059a | KS(salt) = KS / CONS(swide) |
| ctrl_pw30p1 (DO_NOT_PUSH) | v224r6c4pw30cvx1j | pw30 (5a903187) | a0a3467ca44b5765da3d355d93d6b86afe301c193e42b93e9c953c0201b5dde5 | 36bf0e137f1d805268d88e1b5e0d395af8f737cfc65eaa70d22eb6bb8c2ead1c | control |

Every stack kernel dir carries NOT_ARMED.txt.

### 3.3 Offline gate (`sink_kgate.py` -> `kernels/sink_kgate.log`: **SINK_KGATE PASS**, 165 OK, 0 FAIL)
Each notebook's own cells 14 + 15 + 18 are executed on gold58 (stack_kgate.py harness).
- H: all six bases replay to their known gold58 macros; the pw30 stage 3 equals the numpy construction; clean path.
- K: sha chain build -> receipt -> index -> registry; only cell 18 differs from the base; cell 14 is pw30's.
- F: full factorial over {dose, loamm, mclmm, cvx}, as present in each build (42 combinations):
  - F1: all off -> byte-identical to scored pw30 with the pw30 receipt.
  - **F1b: CVX off -> submission.csv AND blend receipt byte-identical to the base A22 entry's replay.**
  - F2: changed columns = union of the ON components.
  - F3: two independent numpy recomputes, maxdiff < 1e-6.
  - F4: receipt values and markers in order EOA < CVX < LOAMM < MCLMM < S6F_BLEND_OK.
- C:
  - the control is byte-identical to pw30;
  - **cvx_pw25 is byte-identical to the independent c4-lineage twin levers/kernel_v224r6c4pw25p2j (9750dfbd);**
  - cross-build column identities hold;
  - the CVX change is live.
- INFO gold58 vs base: cvx_pw30 -.00066, cvx_lomm30 -.00114, cvx_pw25 -.00077, cvx_spl -.00126, cvx_salt -.00077,
  cvx_swide -.00126.

### 3.4 Commit gate (`kernels/sink_commit_gate.py`, generated by `make_sink_gate.py` from stack/kernels/stack_commit_gate.py by asserted substitutions)
- Changes vs the parent:
  - `--poa`;
  - expect() subtracts U ** poa;
  - new (d8): receipt p_oa == poa, and every CVX_APPLIED copy carries poa (none of either at 1);
  - (d6) order EOA < CVX < LOAMM < MCLMM < S6F_BLEND_OK < S6F_DONE;
  - (d7) key set + p_oa.
- **27 checks; a tick must key on `PROBE_CHECK PASS 27/27`.**
- Self-test `kernels/sink_ksim.py` -> `kernels/sink_ksim.log` (**SINK_KSIM_GATE_SELFTEST PASS, 43 / 43, harness True**):
  - **Harness:** each base notebook's cell 18 replayed on its OWN REAL commit reproduces the real bytes, the receipt sha, the log
    lines and the blend receipt (pw30 + pw25 / spl / swide / salt A22 commits).
  - **POS PASS 27/27:**
    - the 7 sims, built on their base's REAL committed inputs;
    - the REAL pw30 commit and the 4 REAL A22 commits with their own flags at p 1;
    - a benign repeated-log sim.
  - **NEG FAIL:**
    - each REAL commit checked as its CVX stack (d8 / d6 / d7);
    - each sim at p 1 (d8 / d7);
    - wrong lomm / mcl / dose / p flags (d2 / d4 / d5 / d8);
    - the control as cvx_pw30;
    - tampers: OA row swap (d / d-oa / f), conflicting CVX marker (d8), CVX before EOA (d6), receipt p_oa 3 vs log 2 (d8),
      extra key (d7).
  - **XCHK:** the parent gate FAILS every cvx sim on (d7) (unknown p_oa key) and PASSES 26/26 on the control sim.
  - **Caveat (as for every stack):** all 5 real commits and all 7 sims have a byte-identical 3-row submission.csv (sha 5b06fff9).
    CVX identity therefore rests on the registry sha guard plus (d8) / (d7) / (d6) / (f), as for lomm05 and MCLmm.
- Gate commands (on the fetched commit dir `<OUT>`; C4T = the c4 table as in stack/inventory.md sec 1):
  ```
  py -3.12 -I session_2026-10-21/levers/kernels/sink_commit_gate.py <OUT> --a224 0.25 --table "$C4T" --mode pw --aoa A --aloamm L --amclmm M --poa 2
  cvx_pw30 A .3 L 0 M 0 | cvx_lomm30 A .3 L .05 M 0 | cvx_pw25 A .25 L 0 M 0 | cvx_spl A .25 L .05 M 0 | cvx_salt A .25 L 0 M .1 | cvx_swide A .25 L .05 M .1
  ```
- Registry: `kernels/registry_entries_sink.json` holds keys cvxpw30 / cvxlomm30 / cvxpw25 / cvxspl / cvxsalt / cvxswide, gate
  PASS 27/27, sims in `kernels/sim/`. `kernels/registry_sink.log`: REGISTRY_SINK_CHECK PASS 18/18 (no key / slug collision
  with val/slate_a22.json). Not merged anywhere.

## 4. Value (step 4; MODEL `sink_value_mc.py` -> `sink_value_mc.log`, `.json`)

Model: stack/skeptic/skep_value_mc.py `world()` verbatim, exec'd as final_slate_mc*.py do (pw25 = implied2 conditional on A21;
lomm05 scenario prior; MCLmm skep2 / stack / zero; I(pw25, lomm05); read noise; gap lognormal median .75, p90 1.8). Component
reads are recovered exactly from its entries (asserted).
- CVX is added with a separate RNG, with the priors in sec 0.6. I(CVX, lomm05) N(-0.2, .3) (gold58-led N(-0.8, .4)).
- Read sd and correlations as MEASURED in sec 2.3.
- W = argmax public over {pw30, pw25, spl, swide, salt}; then the policy's 10-22 entry for that W; R1 over all.

### 4.1 Primary (MCLmm skep2, lomm05 stack, CVX primary), by W
A22 alone: E priv A +1.57, P(.960) .66, P(A < pw30) .22.

| W | P(W) | incr CONS / KS | P(.960 shown) A22 -> CONS / KS | P(entry becomes A) CONS / KS | E gain when it becomes A, CONS / KS | stack vs W: E public / private (CONS) |
|---|---|---|---|---|---|---|
| pw30 | .15 | +0.23 / -0.02 | .00 -> .40 / .30 | .50 / .38 | +0.46 / -0.04 | +0.00 [-4.6, +4.6] / +0.20 [-2.6, +3.0] |
| pw25 | .12 | +0.42 / +0.06 | .38 -> .65 / .55 | .54 / .36 | +0.77 / +0.17 | +0.30 [-3.3, +3.8] / +0.51 [-1.8, +2.8] |
| spl | .26 | +0.32 (KS = CONS) | .73 -> .85 | .56 | +0.57 | +0.46 [-3.1, +4.0] / +0.30 [-2.0, +2.6] |
| swide | .28 | +0.33 (KS = CONS) | .93 -> .96 | .56 | +0.58 | +0.46 [-3.1, +4.1] / +0.31 [-2.0, +2.7] |
| salt | .19 | +0.42 / +0.07 | .83 -> .91 / .88 | .54 / .36 | +0.78 / +0.20 | +0.29 [-3.3, +3.8] / +0.50 [-1.8, +2.8] |

The P(W) column is the MODEL's (pre-read). The real reads replace it; the per-W rows are what to apply.

### 4.2 Robustness
- The grid (5 CVX priors x 5 MCLmm / lomm05 combinations) is in sec 0.6. It moves < .02 with the MCLmm / lomm05 priors.
- Per-W CONS / KS increments by CVX prior (MCLmm skep2, lomm05 stack):

  | CVX prior | pw30 | pw25 | spl | swide | salt |
  |---|---|---|---|---|---|
  | ruler | +0.62 / +0.26 | +1.14 / +0.50 | +1.02 | +1.03 | +1.14 / +0.51 |
  | primary | +0.23 / -0.02 | +0.42 / +0.06 | +0.32 | +0.33 | +0.42 / +0.07 |
  | lskep | +0.04 / -0.15 | +0.19 / -0.07 | +0.11 | +0.11 | +0.20 / -0.06 |
  | zero | -0.01 / -0.14 | 0.00 / -0.16 | 0.00 | 0.00 | +0.01 / -0.15 |
  | neg | -0.32 / -0.39 | -0.20 / -0.24 | -0.37 | -0.36 | -0.20 / -0.23 |

- Sensitivities (CONS, primary / lskep / zero): LB-only gap +0.34 / +0.13 / 0.00; CVX read error corr -.38 with the pw25 read
  +0.34 / +0.12 / 0.00; read sd x 1.3 +0.31 / +0.10 / 0.00; read sd x .7 +0.39 / +0.16 / 0.00.

### 4.3 Reading (GUESS on MODEL)
- The kitchen sink is dominated in every prior and every W. Re-adding a component whose head-to-head read already lost only
  re-selects on noise already seen.
- CONS is a small positive-expectation lottery whenever the CVX prior mean is >= ~0. Its value comes from the population mean
  (zero-prior value 0.00), not from selection.
- It raises P(A < pw30) by .02 (primary) to .08 (gold58-led): the tail grows slightly, because CVX's read noise (2.6-3.4) is
  large next to its expected effect.
- Against readtable.md sec 4, the best already-built recombination there was +0.06e-4 (lomm05@pw30 when pw30 stays). CONS is
  the first 10-22 option with clearly positive E under the skeptic-aware priors, but it stays below public read noise.

## 5. For main (nothing here was done)
1. Any 10-22 entry needs Eesh's GO + a PREREG Amendment 23 + a registry merge + a tick (or a manual submit after 05:30 IST
   10-22, last readable ~10:15 IST).
   - Priority: an r6pw20 ERROR retry first (readtable row 12); CONS(W) second; KS never.
2. Draft A23 text (NOT written):
   > SINK (session_2026-10-21/levers/sink.md). One optional 10-22 A probe: CONS(W) for W = the API best-id at the last 10-21
   > read: pw30 -> rsna-sub-v224r6c4pw30cvx2j (nb f3203107), pw25 -> rsna-sub-v224r6c4pw25cvx2j (nb 015598db), spl ->
   > rsna-sub-v224r6c4pw25lomm05mcl00cvx2j (nb 5b4bbd79), salt -> rsna-sub-v224r6c4pw25mcl10cvx2j (nb ebeb31f8), swide ->
   > rsna-sub-v224r6c4pw25lomm05mcl10cvx2j (nb b6915d97). Each = W's own notebook + the convex E-OA exponent p 2 (levers spec),
   > nothing else.
   > Offline gate levers/kernels/sink_kgate.log SINK_KGATE PASS; commit gate levers/kernels/sink_commit_gate.py must print
   > PROBE_CHECK PASS 27/27 with the entry's flags. Read by R1 (A = API best-id); never B. No KS, no CVX re-dose or other exponent,
   > no further follow-up. [If H1 / H2 is adopted: CONS only for W in {spl, swide, salt}; a cvx entry counts as a stack for H2.]
3. Before a tick self-test, copy `levers/kernels/sim/out_<tag>_sim` into the tick's sim dir and merge the chosen key from
   `kernels/registry_entries_sink.json`.
4. Push-to-gate time: ~19-20 min per commit (CLAIMED a21 / a22 tick logs). Push by ~05:00 IST for a 05:31 submit.

## 6. Caveats and what would change this
- **CVX's prior is the whole story.**
  - Two calibrated rulers say +1.5 to +2.7 (RAK-PUB, OAI P_AREA pub). P_FT pub says +0.2 to +0.5, gold58 -3 to -5
    public-equivalent, and the levers skeptic's selection-corrected OAI search ~0.
  - Treating gold58 as a world rather than as a noisy read moves the prior to about 0, and then CONS is worth nothing.
- p 2 was the levers' choice among 13 shape families on OAI, so the OAI cells here are selection-exposed. The RAK reads are
  this ruler's first look at p 2 (no tuning on RAK), but they come from the same pipeline family that already over-read E-OA
  levels before calibration.
- No ruler is validated on a shape change; only the E-OA dose ladder calibrates them. A public read of CONS (sd 2.6-3.4) cannot
  validate CVX either.
- If the 10-21 reads put pw25 or spl clearly above pw30, the dose-.25 world (where CVX is strongest on RAK) gains weight. If
  pw30 stays best-id, CVX's weakest base is the only one left, and readtable's H1 / H2 makes the entry moot.
- Not done: an independent adversarial skeptic agent on CVX (this file's sec 2.3 is mine); the levers' R7 x gold74 matched read
  was not re-run; the dose-model MC of readtable_mc.py was not extended with CVX (the world() of skep_value_mc.py was used, as
  the brief named).

## 7. Files (`session_2026-10-21/levers/`)
- **Measurement:**
  - `sink_local.py` -> `sink_local.log`, `sink_local.json`, `sink_local.stdout` (G58, RAK regime scan, RAK g74 / PUB / PUBM /
    s-0.83, REP; singles and pairs on 5 bases);
  - `sink_summ.py` -> `sink_summ.out`;
  - `sink_cvx_skep.py` -> `sink_cvx_skep.log` (3-way check, target variants, RAK-u, LOO);
  - `sink_cvx_matched.py` -> `sink_cvx_matched.log`;
  - `sink_read_noise.py` -> `sink_read_noise.log`;
  - `box_sink.py` -> `box_sink_smoke.log`, `box_sink.log` (OAI aggregates);
  - `box_sink_summ.py` -> `box_sink_summ.out`.
- **Builds:** `build_sink_kernels.py`; `kernels/kernel_<tag>/` x 7 (6 stacks NOT_ARMED, 1 control DO_NOT_PUSH),
  `kernels/build_receipt_<tag>.json`, `kernels/build_index.json`, `kernels/build_sink_kernels.log`.
- **Gates:**
  - `sink_kgate.py` -> `kernels/sink_kgate.log` (offline);
  - `make_sink_gate.py` -> `kernels/sink_commit_gate.py`;
  - `kernels/sink_ksim.py` -> `kernels/sink_ksim.log`, `kernels/sim/out_<tag>_sim` x 7, `kernels/sim/neg_*` x 6;
  - `kernels/registry_sink.py` -> `kernels/registry_entries_sink.json`, `kernels/registry_sink.log`.
- **MODEL:** `sink_value_mc.py` -> `sink_value_mc.log`, `sink_value_mc.json`.
