# inventory.md: stack inventory on top of pw30, pairwise interactions, and 3 stack definitions (10-20 ~22:30 to 23:10 IST, Claude, STACK round)

RESEARCH + LOCAL BUILDS ONLY. Nothing was submitted, pushed, committed, versioned or downloaded. No dataset, no GPU job, no scheduled
task. PREREG, HANDOFF, PROGRESS and the a20 / a21 / val / night / private folders were only read (val/slate_a22.json was read, not
written). Everything written is under `session_2026-10-20/stack/`.

**OAI governance.** Box use = 3 read-only CPU scripts piped over ssh stdin into `python -I -` (`box_probe_cols.py`: column presence +
counts; `box_stack.py`: a B 2 smoke, then B 60), printing aggregates only (means / sd / P over weighted full-size resamples, prevalences,
weighted label phi). Nothing was written on the box, no OAI CSV was printed, no id / row / per-knee value left the box. The only data sent
to the box were gold74 2x2 count tables (local aggregates). Check: no file in `stack/` matches `oai_[0-9]+_[LR]` or `oai_[0-9]{4,}`
(0 hits; the bare 7-digit-pattern hits are inside sha256 hex strings). Box GPU was busy with another job (97 %); not used.

**Tags.** MEASURED = computed in this round, log named (all in `stack/`). CLAIMED = older note, named. MODEL = arithmetic on stated
inputs (`stack_mc.py`). GUESS = judgement. Units: e-4 = 1e-4 macro AUC vs pw30 57425970 unless stated; "label" = one-label AUC delta
(macro = label / 12). Intervals [p10, p90]. gold58 numbers are INFO only, never a selection ruler.

Rulers (sec 3 has the calibration):
- **G58**: gold58 (n 58) on the c4 kernel path from member serve files (parity: c4 .94113, pw30 .94336, pw25 .94368, lomm05 .94384, E1 on
  pw30 .94347, all exact; MEASURED `stack_local.log`).
- **RAK**: RSNA train OOF (R7 / s1044 / s1045, n 3,467 non-gold) in three c4-shaped nests (each OOF model takes the Raptor slot once),
  report-v2 labels raked by IPF to the gold74 EXPERT joint: OA labels on (MOA, LOA, PF, MM, LM), MCL on (MCL, MM); other labels unraked.
  "RAK-u" adds gold74 target uncertainty (Dirichlet draws of the target table x row bootstrap).
- **REP**: the same nests on raw report labels. Mechanism control: a decorrelation that matches the expert regime MUST lose here.
- **OAI**: R6 + S6fall served on OAI + fleet D5 / s1044 / s1045, five c4-shaped nests, MOAKS P_FT / P_AREA, regimes g74 (all pairs at
  gold74 log OR), pub (trio pairs -0.83 sd, the public gap-matched regime), raw. OA trio + menisci only (no MCL on OAI).

## 0. Bottom line

1. **One new component survives and it is column-disjoint from everything else: MCLmm** (`MCL = pct(pct(MCL) - .1 pct(MM))` on the final
   output). It was a census candidate killed for "specificity" (CLAIMED expert2 / census, recal -0.2). This round brings new evidence:
   - Mechanism on the SAME knees: reports over-call MCL in expert-MM+ knees. Of 12 report-MCL+ knees that are expert-MM+, 9 are
     expert-MCL-negative; in expert-MM- knees 2 of 10 (hypergeometric P .015 gold74; .028 gold58 alone; MEASURED). Expert MCL~MM log OR
     -0.98 vs report +0.48 on the same 73 studies; train reports +0.91. Negative in both ACL strata.
   - RAK-u: +3.5 / +3.1 / +3.6e-4 (P .90 / .93 / .92) in the three nests; REP control -3.4 / -3.4 / -3.0 (loses, as it must).
   - Robust to the raking set (MCL, MM, + ACL / Contusion / MOA / Effusion): +2.4 to +5.3, P .79-.96; 4-way (+ACL + Contusion) +1.1 to
     +1.8, P .68-.72 (`stack_mcl.log`).
   - Mini-skeptic (`skep_mcl.log`): leave-one-gold-study-out 74 / 74 positive (min +2.7); gold-like mix +2.0 to +3.3; target table from
     the 16 extra gold studies alone (not the gold58 that nominated MM) +1.25 to +1.64; wrong-direction borrow -7; single members +1.5 to +3.7.
   - Placebo partners: the gain tracks the report-minus-expert coupling gap (corr .75-.78; MM rank 2 of 11 after Baker's); partners with
     no gap (ACL, LM, LOA, Contusion) give <= 0. So the "nonspecific" pattern the census skeptic saw is the mechanism, not noise.
   - Expert reads, not independent of the nomination: G58 c4 path +.0091 label (+7.6e-4); R7 OOF x gold74 +.0101 label (P .95).
   - MODEL value: +0.9 [-0.4, +2.3]e-4 population, P(>0) .79 under the stack prior; .71 pessimistic; .50 skeptic-null; .42 recal prior.
2. **lomm05 is weaker than NIGHT said, but stays E >= 0.** The c4-shaped RAK-u reads +0.8 / +1.1 / +1.1 (P .68-.72), about half of the
   3-mean ruler NIGHT used (+2.1), and OAI at the public regime averages +0.1 (P_FT pub +0.24, P_AREA pub -0.03; role-dependent). MODEL
   +0.17 [-1.5, +1.8], P(>0) .56 (NIGHT +0.41, P .62).
3. **pw25 stays E >= 0**: RAK-u ~0 (-0.5..0.0), OAI pub +0.4 to +3.5 in 10 / 10 cells, G58 +3.2. MODEL +0.4 [-0.6, +1.4], P .69 (VAL
   private .61).
4. **Everything else drops out of any stack:**
   - E1: negative on the only expert ruler for its OA / meniscus columns (OAI P_FT pub -0.07 to -1.53 in 5 / 5 nests; P_AREA pub ~0);
     its RAK gain comes from the unraked report labels (REP +1.3 to +4.3), i.e. the old report-OOF evidence again.
   - BOR (C13): RAK -2.4 in 3 / 3 nests (P .06-.09), BORm -1.9 (P .02-.03); OAI positive only via weak OAI MOA heads (P_AREA); its LOA
     half is the killed C14.
   - S6up, RPF, SUBH: level effects are member-role artefacts on both RAK and OAI (sign flips with which model sits in which slot), and all
     three are SUBSTITUTES for the E-OA dose: I(pw25, x) < 0 (OAI pub mean -0.09 / -0.22 / -0.26; G58 -5.9 / -9.7 / -1.6).
   - KILLED list unchanged (no new evidence): E2s, C6 / C6', C14, fluid, ACL, global, convex E-OA, LP shapes, doses >= .35.
5. **Interactions among the survivors are small or exact:** pw25 x lomm05 is mildly SUPER-additive (OAI +0.24 mean, > 0 in 25 / 25 cells;
   REP +0.21..+0.31; RAK-u +0.26 / -0.14 / +0.29; G58 0). MCLmm x anything is EXACTLY additive (disjoint written column; verified by
   G4 in `stack_gate.log`). No column conflict exists between MCLmm and the OA-side components; pw25 and lomm05 both write Lateral OA
   but add.
6. **Stacks (sec 1), all built, offline-gated (STACK_GATE PASS) and commit-gate self-tested (STACK_SIM_GATE_SELFTEST PASS, 12 / 12 rows with an expectation, +1 INFO):**
   - **S_safe = pw30 + MCLmm** (only component with P(>0) >= .6 under the stack prior that has no column conflict).
   - **S_wide = pw25 + lomm05 + MCLmm** (every E >= 0 survivor whose stack-marginal is >= 0).
   - **S_alt = pw25 + MCLmm** (S_safe moved onto the pw25 dose).
   - Slate MODEL: {S_alt, S_wide} beats the NIGHT A pair {pw25, lomm05} under all four MCLmm priors (+1.68 / +1.28 / +0.90 / +0.76 vs
     +0.70e-4 private of the R1 pick); adding S_wide to {pw25, lomm05} is never negative (+0.18 to +0.85). Nothing is armed.

## 1. Stack definitions (precise)

Common: base notebooks are the SCORED pw30 (`a21/kernel_v224r6c4pw30j`, nb sha 5a903187...) or the BUILT pw25
(`val/kernels/kernel_v224r6c4pw25j`, nb sha 64ed7eba...; = pw30 with `A_OA = 0.3` -> `0.25`). Each stack inserts lines in cell 18
(`_s6f_blend`, S6F path) right after the A21 E-OA block (anchor `print('EOA_APPLIED a_oa ...')`), in this order, and nothing else:

```
        A_LOAMM = 0.05   # [night C12'] ...                         (lomm05 block, byte-identical to night/c12 build; S_wide only)
        if A_LOAMM and {'Lateral OA', 'Medial Meniscus'} <= set(labels) and not {'Lateral OA', 'Medial Meniscus'} & set(blank):
            out['Lateral OA'] = pct(pct(out['Lateral OA'].values.astype(np.float64))
                                    - A_LOAMM * pct(out['Medial Meniscus'].values.astype(np.float64))).astype(np.float32)
            rec['a_loamm'] = A_LOAMM
            print('LOAMM_APPLIED a_loamm %g' % A_LOAMM, flush=True)
        A_MCLMM = 0.1   # [stack MCLmm] MCL decorrelated from Medial Meniscus on the final output (stack/inventory.md); 0 = no-op
        if A_MCLMM and {'MCL', 'Medial Meniscus'} <= set(labels) and not {'MCL', 'Medial Meniscus'} & set(blank):
            out['MCL'] = pct(pct(out['MCL'].values.astype(np.float64))
                             - A_MCLMM * pct(out['Medial Meniscus'].values.astype(np.float64))).astype(np.float32)
            rec['a_mclmm'] = A_MCLMM
            print('MCLMM_APPLIED a_mclmm %g' % A_MCLMM, flush=True)
```

| stack | definition (vs pw30) | kernel dir (stack/) / Kaggle id | nb sha256 | meta sha256 | columns changed vs pw30 | gold58 INFO macro vs pw30 | MODEL population [p10, p90] | P(mu>0) | MODEL private [p10, p90] | P(public > pw30) | P(.960 display) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **S_safe** | + MCLmm a .1 | `kernel_v224r6c4pw30mcl10j` / `eeshsaxena/rsna-sub-v224r6c4pw30mcl10j` | 962a2c9fc642c163616771d8056a9f11b4dcb22adf527ef16f8f9328b88621e7 | 820a89bef0f8c5b804e16f73501243eaedbb91f8861428d94f83c1c40926bfbb | MCL | +.00076 | +0.93 [-0.42, +2.33] | .79 | +0.93 [-1.57, +3.44] | .63 | .50 |
| **S_alt** | A_OA .25 + MCLmm a .1 | `kernel_v224r6c4pw25mcl10j` / `eeshsaxena/rsna-sub-v224r6c4pw25mcl10j` | 051232e424f10840412f9618e56081681e571a6a77815132bac83fa99af2f0d1 | 084cecc2f608ebe025bd47d5c741f858d1c8f5cf4be343798e908a9582f6f924 | OA trio, MCL | +.00108 | +1.33 [-0.37, +3.06] | .84 | +1.33 [-1.61, +4.28] | .66 | .55 |
| **S_wide** | A_OA .25 + lomm05 + MCLmm a .1 | `kernel_v224r6c4pw25lomm05mcl10j` / `eeshsaxena/rsna-sub-v224r6c4pw25lomm05mcl10j` | c9801a80bbc64ecbd1aba15e9470657fca4631ee7ac091f68b7742d1e17d7478 | 6a2745e7f68841c50f200bdc40e1259e95452237163885c67ed73473ec7f8d1e | OA trio, MCL | +.00156 | +1.66 [-1.00, +4.38] | .77 | +1.66 [-2.03, +5.36] | .67 | .58 |
| S_mid (built, not proposed by default) | lomm05 + MCLmm | `kernel_v224r6c4pw30lomm05mcl10j` | b92bd7846e7a74d3a6a2853f967b851f5632f660d9383462305da75ad4d118fa | b81b5a828840e809f081851adbe39d3ce8bb1115a58487a2297a763b66151179 | LOA, MCL | +.00124 | +1.11 [-1.34, +3.61] | .69 | +1.10 [-2.25, +4.47] | .63 | .52 |
| control, NEVER push | A_MCLMM 0 | `kernel_v224r6c4pw30mcl00j` (DO_NOT_PUSH.txt) | e0c11f1e... | 7711e393... | none (byte-identical to pw30) | 0 | | | | | |

MODEL rows: `stack_mc.log` (stack prior; sensitivity in sec 6). Metadata differs from the base's only in id / title / code_file (asserted).

**Offline gate** (`stack_gate.py` -> `stack_gate.log`: **STACK_GATE PASS**), exact cell-18 code exec'd on gold58:
- G1: the A_MCLMM 0 control is byte-identical to pw30 (sha cab96d91...), no a_mclmm.
- G2: each build changes exactly {MCL} or {Lateral OA, MCL} vs its base (pw30 / pw25); every other column byte-equal.
- G3: MCL == pct(pct(MCL) - .1 pct(MM)) of the base's own output (maxdiff 4.1e-8); LOA == the lomm05 line (4.1e-8).
- G4: S_mid's LOA == night lomm05's LOA; the MCL column is identical across all 4 builds (MCL does not depend on A_OA or LOAMM).
- G5: receipts a_oa / oa_mode of the base, a_loamm .05 only in lomm variants, a_mclmm .1.
- G6: each build == an independent from-scratch numpy pipeline (member serve files -> c4 -> pw E-OA -> lines), 12 columns, 2.9e-8.

**Commit gate** (new): `stack/stack_probe_check.py` = `night/c12_probe_check.py` + `--amclmm`, (d) extended with the MCL line, new (d5):
receipt a_mclmm == dose and MCLMM_APPLIED in the log (none of either at 0). 19 checks; a tick must key on **`PASS 19/19`**.
`stack_sim.py` -> `stack_sim.log` (**STACK_SIM_GATE_SELFTEST PASS**, 12 / 12 rows as expected + 1 INFO; REAL committed pw30 dir + each build's cell 18 replayed on it):
- PASS 19/19 on all four build sims and on the REAL scored pw30 dir with `--aloamm 0 --amclmm 0`.
- FAIL 18/19 as required: amclmm .1 on the real pw30 dir (d5); amclmm 0 or .2 on the S_safe sim (d5); aloamm 0 on S_mid (d4); pw25
  sims checked as a .3 and vice versa (d2); MCL tamper (d).
- **Caveat (same as lomm05):** the 3 visible test rows cannot show MCLmm (MCL ranks [.67, 1, .33] stay put; the old C12' gate PASSes
  the S_safe sim 18/18). Dose identity rests on the tick's registry sha guard plus (d5); the (d) arithmetic is proven on 58 rows (G3 / G6)
  and by the tamper row.

Gate commands (on the fetched committed dir `<OUT>`; must print `PROBE_CHECK PASS 19/19`):
```
C4T='{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker'"'"'s": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}'
S_safe: py -3.12 -I session_2026-10-20/stack/stack_probe_check.py <OUT> --a224 0.25 --table "$C4T" --aoa 0.3  --mode pw --aloamm 0    --amclmm 0.1
S_alt : py -3.12 -I session_2026-10-20/stack/stack_probe_check.py <OUT> --a224 0.25 --table "$C4T" --aoa 0.25 --mode pw --aloamm 0    --amclmm 0.1
S_wide: py -3.12 -I session_2026-10-20/stack/stack_probe_check.py <OUT> --a224 0.25 --table "$C4T" --aoa 0.25 --mode pw --aloamm 0.05 --amclmm 0.1
```
Registry entries for main to merge into `val/slate_a22.json` (keys s_safe / s_alt / s_wide / s_mid, exact shas and gate args):
`stack/registry_entries_stack.json`. Before the tick self-test, copy `stack/sim/out_<tag>_sim` to `val/sim/`.

**Draft PREREG text** (for main, inside Amendment 22; NOT written):
> - STACK (stack/inventory.md). MCLmm = `MCL = pct(pct(MCL) - 0.1 pct(MM))` on the final output after E-OA (cell 18, +6 lines). Dose a .1 is the
>   census spec, not re-picked. Entries: s_safe = rsna-sub-v224r6c4pw30mcl10j (nb 962a2c9f), s_alt = rsna-sub-v224r6c4pw25mcl10j (nb 051232e4),
>   s_wide = rsna-sub-v224r6c4pw25lomm05mcl10j (nb c9801a80). Gates: stack/stack_gate.log STACK_GATE PASS; commit gate stack/stack_probe_check.py
>   must PASS 19/19. Read by R1 with the other A probes (A = API best-id). Never B. No re-dose of MCLmm, no MCL <- other partners, no 10-22 follow-up.

## 2. Component inventory (every candidate named in the brief)

E columns: "prior" = the source's own number (CLAIMED); "stack" = this round's value (MODEL `stack_mc.py` for the three survivors,
otherwise the measured reads summarised). Public and private share the population mean; private intervals are wider by the read noise
(sec 6). Ruler cells give single deltas vs pw30 in e-4 (P > 0); nest order RAK: N_A R7|s1044|s1045, N_B s1044|R7|s1045, N_C s1045|s1044|R7;
OAI pub: N1 R6|D5|s1044, N3 R6|s1044|s1045, N2 S6|D5|s1045, N4 R6|s1045|D5, c4D5 D5|R6|S6.

| component | exact spec (cell, change) | built? | E prior (source) | E stack (this round) | G58 INFO | RAK-u (3 nests) | REP control | OAI pub P_FT / P_AREA (5 nests) | sign robust? | skeptic negative? | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **MCLmm** | cell 18 after E-OA: `MCL = pct(pct(MCL) - .1 pct(MM))`, float32 | yes (S_safe) | recal -0.2 [-1.5, +1.1], P .42; census / expert2 KILL (specificity p .057) (CLAIMED night/recal.md, more/expert2.md) | **+0.93 [-0.42, +2.33], P .79** (pessimistic +0.54 P .71; skeptic null 0) | +7.6 (.81); dose .05 / .1 / .2: +.0045 / +.0091 / +.0159 label | +3.5 (.90) / +3.1 (.93) / +3.6 (.92); dose-monotone to .2 | -3.4 / -3.4 / -3.0 (loses: OK) | not measurable (no MCL on OAI) | yes on every expert-anchored ruler (G58, RAK 6 raking sets x 3 nests, R7 x gold74, LOO 74 / 74) | YES before (census / expert2 KILL, recal NO); this round's mini-skeptic did not break it (sec 5) | **KEEP: in every stack** |
| **pw25** | cell 18 `A_OA = 0.3 -> 0.25` | yes (val/kernels, 64ed7eba) | public +0.1 [-0.9, +1.1], private +0.56 [-2.0, +3.0], P .61 (CLAIMED val/implied2.md) | **+0.40 [-0.63, +1.42], P .69** | +3.2 (.65) | -0.0 (.50) / -0.1 (.48) / -0.5 (.44) | +5.3 / +5.6 / +4.8 (report favours less decorrelation, by design) | +1.7 / +2.9 / +2.3 / +2.9 / +3.5 ; +0.4 / +1.2 / +0.4 / +1.5 / +1.4 | mostly: OAI 10 / 10 > 0, G58 > 0, RAK ~0 | implied2 sec 8: "beats pw30" is prior-driven (population -0.27..+0.90), KEEP low value; not negative | **KEEP: S_wide, S_alt** |
| **lomm05** | cell 18 after E-OA: `LOA = pct(pct(LOA) - .05 pct(MM))` | yes (night, c0dd7ec8) | NIGHT +0.41 [-1.43, +2.20], P .62; skeptic +0.2, P .56, KILL (CLAIMED night/NIGHT_1020.md) | **+0.17 [-1.48, +1.81], P .56** | +4.8 (.81) | +0.8 (.68) / +1.1 (.72) / +1.1 (.72) | -0.4 / -0.6 / -0.3 (loses: OK) | +0.42 / +0.68 / +0.24 / -0.12 / -0.02 ; +0.15 / +0.01 / +0.07 / -0.03 / -0.34 | no: OAI role-dependent, g74 P_AREA 5 / 5 < 0 | YES (skep_c12p_post KILL; NIGHT kept it as a lottery) | **KEEP in S_wide only** (P < .6) |
| E1 soft-OR | cells 14 + 18: every member pct ** 2 before mixing | yes on pw30 (push/e1/alt, 4a0eed92) | -0.3..+1 (CLAIMED HANDOFF 15:40); LEVERS: R1 value -0.46..+0.09, DEAD for a slot | **< 0 on the expert ruler** | +1.1 (.54) | +4.3 / +5.4 / +1.1, carried by the 9 unraked report labels | +4.3 / +4.0 / +1.3 | -1.18 / -1.03 / -0.07 / -0.20 / -1.53 ; -0.03 / +0.17 / +0.64 / +0.06 / -0.84 | no: report-label rulers +, expert ruler - | YES (LEVERS DEAD; e1build: the gold58 +.00009 was a float64 artefact) | **OUT** |
| BOR (C13) | cell 18 after E-OA: `MOA = pct(pct(MOA) + .1 pct(MM))`, `LOA = pct(pct(LOA) + .1 pct(LM))` | no | +0.2 [-1.3, +1.7], P .57 (CLAIMED recal.md) | **< 0 (RAK)** | +3.2 (.66); MOA half 0.0 | -2.4 (.06) / -2.3 (.09) / -2.5 (.07); MOA half -1.9 / -2.1 / -1.8 (P .02-.03) | -3.1 / -2.7 / -3.2 | +1.4 / +0.5 / +0.7 / +0.7 / +1.7 ; +5.7 / +5.8 / +4.7 / +4.6 / +5.6 | no: OAI + (weak OAI MOA heads, recal 3.3), RAK - | its LOA half = C14, KILLED (skc14) | **OUT** |
| S6up | cell 18 blend loop: W_S6F .2 -> .3 for the OA trio only | no | 0..+1 (CLAIMED LEVERS row 6, WEAK knob) | **~0, sign not identified** | +11.8 (.84) | -3.3 / -1.4 / +11.7 (whichever model sits in the S6 slot) | -2.0 / -0.1 / +8.8 | -3.75 / +0.73 / -0.84 / +0.61 / -2.27 ; -3.69 / +0.05 / -0.87 / -0.29 / -1.34 | no (member-quality artefact) | LEVERS: knob, R1 ~0 | **OUT** (dose substitute) |
| RPF | cell 14 `W_RAPTOR_C['PF OA'] .5 -> .4` | no | +0.5..+2.5 only if c4's table was optimal; R1 -0.45..+0.06 (CLAIMED LEVERS row 5) | **~0 or < 0** | +12.9 (.92) | -3.1 / +2.6 / +0.0 | -1.6 / +1.3 / -0.1 | -0.70 / -2.02 / -0.06 / -0.68 / -2.11 ; -0.59 / -1.49 / -0.50 / -0.90 / -1.12 (level not transferable, space.md 2.3) | no | LEVERS: WEAK knob | **OUT** |
| SUBH (SUB_HIGH) | cell 18 E-OA block: subtract `pct(2/3 pct(R6) + 1/3 pct(S6fall))` of the other compartments instead of the final's (weights are my spec; LEVERS never fixed them) | no | ~+0.5 (LEVERS row 7); recal +0.2 [-1.1, +1.5], P .58 | **~+0.2 alone, < 0 next to pw25** | +2.9 (.63) | +1.5 / -0.2 / -1.9 | +2.3 / -0.2 / -1.6 | +1.24 / +1.53 / +2.75 / +1.10 / -1.82 ; +0.92 / +1.69 / -0.25 / +0.47 / -0.19 | no: role-matched +, c4D5 - | convex skeptic: search winner holds out ~0 (box_selcorr) | **OUT** (dose substitute) |
| MCL <- MM | = MCLmm | | task prior ~+0.1 +- .4 | see MCLmm | | | | | | | |
| killed list: E2s, C6 / C6', C14, fluid, ACL, global, convex E-OA, LP shapes, doses >= .35 | as NIGHT_1020 sec 2 / VAL_1020 sec 4 | some | <= 0 or below noise (CLAIMED) | unchanged | | | | | | yes | **OUT** (no new evidence; BOR's RAK read is consistent with the C14 kill) |

## 3. Rulers and their calibration (MEASURED)

| ruler | E-OA ladder vs c4 (eoa10 / eoa20 / eoa30 / pw20 / pw30, e-4) vs public 5.9 / 10.3 / 9.6 / 10.4 / 11.3 | public / ruler at pw30 | use here |
|---|---|---|---|
| G58 c4 path | 14.2 / 24.7 / 20.1 / 26.5 / 22.3 | .51 (all five .39-.51) | INFO (x .42, CLAIMED recal) |
| RAK c4 nests, fixed target | 6.2-7.1 / 11.4-12.0 / 11.9-13.4 / 13.1-14.0 / 14.5-16.1 | .70-.78 | x .7 for OA-side components |
| RAK-u (with target uncertainty) | | median .63-.67 [p10 .35, p90 1.9-3.1] | as above, wide |
| RAK 16-cell (MOA, LOA, PF, MM) | 5.4-5.8 / 8.8-9.4 / 8.2-9.5 / 9.3-10.0 / 8.8-9.5 | 1.2-1.3 | sensitivity only |
| REP | -0.2 / -3.3 / -11.3 / -3.4 / -12.1 (N_A) | wrong sign by design | control |
| OAI P_AREA pub (5 nests) | 7.0-8.2 / 10.1-13.0 / 8.4-13.5 / 10.9-13.6 / 9.1-14.2 | .80-1.24 | best-calibrated OAI cells |
| OAI P_FT pub N1 | 6.6 / 9.7 / 7.2 / 10.7 / 8.9 | 1.27 (others 1.7-9) | N1 only |
| OAI g74 / raw | g74 wrong shape (pw30 < 0 in most nests); raw 2.5-3.5x too large | | interactions only |

Reading: the c4-shaped RAK nests get the E-OA shape right with a ~.7 factor; OAI fits only in the pub regime. Neither ruler is validated
on a non-OA label (no public read of any non-OA transform exists), which is the main caveat on MCLmm.

## 4. Pairwise interactions (I = d(A+B) - d(A) - d(B), e-4)

OAI = mean over the 10 pub cells (P_FT + P_AREA x 5 nests), with the count of positive cells; RAK = three nests, fixed target (RAK-u for
the survivors in brackets); REP = three nests; G58 = point (boot sd 4-9, INFO). Logs: `box_stack.log`, `stack_local.log`, `stack_rak2.log`.

| pair | column relation | OAI pub | RAK | REP | G58 | class |
|---|---|---|---|---|---|---|
| **pw25 x lomm05** | both write LOA | **+0.24 (10 / 10 > 0; 25 / 25 over all regimes)** | +0.28 / -0.11 / +0.33 [+0.26 / -0.14 / +0.29] | +0.27 / +0.31 / +0.21 (P .96-1.0) | 0.00 | **adds, mildly super-additive** |
| **pw25 x MCLmm** | disjoint (OA trio vs MCL) | n/a | exactly 0 | exactly 0 | exactly 0 | **exact add** |
| **lomm05 x MCLmm** | disjoint (LOA vs MCL; both READ MM, neither writes it) | n/a | exactly 0 | exactly 0 | exactly 0 | **exact add** (errors correlated through the shared gold74 basis: corr .31 in the MODEL) |
| pw25 x SUBH | same OA trio | -0.26 (0 / 10 > 0; 1 / 25 over all regimes) | -0.32 / -0.24 / +0.68 | -0.35 / -0.13 / +0.38 | -1.61 | overlap (substitutes) |
| pw25 x RPF | same OA trio | -0.22 (role-matched -0.14..-0.57; c4D5 +0.34..+0.39) | +0.01 / -0.10 / -0.18 | -0.19 / +0.03 / +0.07 | -9.66 (P .02) | overlap |
| pw25 x S6up | same OA trio | -0.09 | +0.04 / -0.85 / -0.17 | -0.07 / -0.41 / -0.28 | -5.90 (P .13) | overlap |
| pw25 x E1 | E1 writes all 12 | +0.19 | +0.21 / -0.29 / +0.58 | -0.02 / +0.15 / +0.06 | -3.75 | ~0 |
| pw25 x BOR | same MOA / LOA | +0.41 | +0.44 / -0.26 / +0.18 | +0.18 / 0.00 / +0.04 | -1.29 | adds (BOR itself < 0) |
| lomm05 x SUBH | LOA | -0.08 (role-matched -0.04..-0.26) | +0.31 / -0.19 / +0.01 | +0.17 / +0.10 / -0.08 | -3.22 | overlap, small |
| lomm05 x BOR | same LOA column | +0.03 | +0.37 / -0.12 / +0.43 | +0.31 / +0.31 / +0.37 | -3.22 | ~0 (column conflict, no cancel) |
| lomm05 x S6up / RPF / E1 | LOA vs trio / all | -0.03 / -0.01 / -0.03 | +0.05 / -0.07 / -0.11 ; +0.11 / -0.03 / +0.08 ; +0.23 / -0.25 / +0.29 | small | 0.00 / -1.61 / 0.00 | ~0 |
| S6up x RPF | member weights, OA | -0.22 | +0.01 / -0.10 / -0.21 | -0.32 / +0.22 / +0.11 | -11.80 (P .04) | overlap |
| S6up x SUBH | OA | -0.08 (role-dependent, -0.61..+0.53) | +0.29 / -1.31 / +2.08 | +0.27 / -0.88 / +1.64 | -4.30 | role-dependent, unstable |
| SUBH x RPF | OA | +0.05 | +0.23 / -0.04 / +0.06 | | -5.37 | ~0 |
| RPF x E1 | | +0.29 (role-dependent) | +0.50 / -0.14 / -0.04 | | +3.75 | ~0 |
| MCLmm x E1 | E1 writes MCL too | n/a | +0.08 / +0.08 / -0.04 | -0.03 / -0.03 / -0.13 | 0.00 | ~0 (E1 is out anyway) |

Reading:
- The only real interaction among survivors is pw25 x lomm05, and it helps. Mechanism (GUESS): at a .25 the LOA output keeps more of
  the MOA / PF signal, which co-fires with MM, so the C12' line has a slightly larger gap to remove.
- The OA-side knobs (S6up, RPF, SUBH) all move the same thing as the dose: how strongly the OA trio is decorrelated. Their interactions
  with pw25 are negative on OAI and G58. In a stack with pw25 each one's marginal is about E(x) + I(pw25, x) ~ +0.2 - 0.3 < 0.
- No measured interaction changes sign across rulers for the survivors.

## 5. MCLmm: why the earlier KILL is reversed (and how far)

What the census / expert2 skeptic got right (CLAIMED): MCL gains on gold58 from many partners (MCL <- Syn / Cont / Frac / MOA / mean11),
so MM looked nonspecific (p .057); gold58 directional gaps outside the trio have split-half reliability ~.01 for MCL targets (recal 3.1).

What this round adds (MEASURED):
- **Mechanism on the same 73 gold74 studies (report v2 vs expert):** expert MCL+ is a subset of report MCL+ (11 of 22). The 11 report-only
  MCL calls sit in expert-MM+ knees 9 : 2 (P .015; gold58 alone P .028; extra16 2 : 0). Expert MCL~MM log OR -0.98 vs report +0.48 on the
  same studies; train reports +0.91 (n 3,845); within ACL- / ACL+ strata gold74 -0.75 / -1.14 vs train +0.69 / +0.68. This is the
  "MCL chronic / grade-1" contamination the rubric lens found (rubric.md: perfect-demotion bound +.0035 chronic / +.0090 grade 1 per label)
  and called unrealisable; MCL <- MM realises part of it (RAK +.0038 label, below the grade-1 bound).
- **Specificity re-read:** under per-partner raking the MCL <- X gain tracks the report-minus-expert coupling gap (corr .75-.78); MM is
  rank 2 of 11 (Baker's first, then MM, PF OA, Fracture, MOA, Effusion); ACL, LM, LOA, Contusion (small gaps) give -.0042 to +.0001 label.
  The "many partners help" pattern is the mechanism (expert MCL is acute-type, decoupled from degenerative findings), not noise.
- **Ruler and skeptic checks:** RAK-u P .90-.93; 3-way rakings +2.4..+5.3; 4-way +1.1..+1.8 (P .68-.72); LOO over the 74 gold studies
  74 / 74 > 0 (min +2.7); gold-like population +2.0..+3.3; target from the 16 non-gold58 studies only +1.25..+1.64; single members
  +1.5..+3.7; wrong-direction borrow -7 (must lose, does); REP -3.0..-3.4 (must lose, does).

What still limits it (and sets the MODEL haircuts):
- All expert information is 11 MCL+ knees in gold74 (3 MM+). The Dirichlet covers their sampling noise, not the fact that gold58 also
  nominated MM (census). Haircut x .75 (GUESS).
- No ruler is calibrated on a non-OA label. The E-OA factor (.63-.78) is assumed to transfer.
- c4 co-fires MCL~MM at .186 on gold58 vs .46-.48 for the train nests (label-free, `stack_rak2.log`). If the hidden set is gold-like,
  the room is smaller; but gold58 itself (at .186) still reads +.0091 label.
- S2 risk: c4's extra co-firing could carry real MCL signal on the hidden labels (then ~0). Weighted .35 in the MODEL.
- No separate adversarial skeptic agent has reviewed MCLmm in this round (only `skep_mcl.py`, written by me). If time allows before the
  GO, a 20-minute skeptic on exactly this section is the best use of it.
- Not pursued (forking): MCL <- mean(MM, MOA, PF, Baker's), higher MCL doses (RAK is monotone to a .2). The a .1 census dose is kept.

## 6. Slate arithmetic under R1 (MODEL `stack_mc.py`, `stack_mc_sens.py`)

Inputs: pw25 mu N(+0.4, 0.8); lomm05 and MCLmm from a shared scenario draw (S1 train-like / S2 extra co-firing carries signal / S3
gold-like, weights .40 / .35 / .25; lomm05 S1 +1.10 (mean of the 3-mean x .73 and c4-nest x .7 reads), S2 -0.48, S3 -0.26, x .85;
MCLmm S1 +2.3, S2 0, S3 +1.3, x .75); I(pw25, lomm05) N(+0.16, .12); public paired read sd pw25 1.4 (CLAIMED), lomm05 1.01 (CLAIMED),
MCLmm 2.5 (MODEL: gold58 sd 7.1 x sqrt(58 / 390)); a component's read error is SHARED by every entry containing it; private sd x .65;
.960 gap lognormal median 0.75 [0.3, 1.8] (CLAIMED implied2). r6pw20 is B and never A, so it is left out.

E private of the R1 pick vs keeping pw30 (e-4), by MCLmm prior (stack / pessimistic / skeptic-null N(0,1) / recal N(-.2,1)):

| A-probe set (plus pw30 itself) | stack prior | pessimistic | skeptic null | recal prior | P(A worse than pw30), stack prior | P(.960 shown), stack prior |
|---|---|---|---|---|---|---|
| NIGHT {pw25, lomm05} | +0.70 | +0.70 | +0.71 | +0.71 | .19 | .58 |
| {pw25} | +0.39 | | | | .17 | .38 |
| {S_safe} | +0.74 | +0.41 | +0.15 | +0.06 | .17 | .50 |
| {S_alt} | +1.08 | +0.74 | +0.43 | +0.31 | .16 | .55 |
| {S_wide} | +1.54 | +1.13 | +0.79 | +0.67 | .14 | .58 |
| **{S_alt, S_wide}** | **+1.68** | **+1.28** | **+0.90** | **+0.76** | .14 | .63 |
| {S_safe, S_wide} | +1.59 | +1.19 | +0.81 | +0.67 | .16 | .65 |
| {pw25, lomm05, S_wide} | +1.55 | +1.20 | +0.97 | +0.89 | .19 | .72 |
| {pw25, lomm05, S_safe, S_wide} | +1.60 | | | | .20 | .78 |

- {S_alt, S_wide} is >= the NIGHT pair in all four priors and uses the same 2 slots. Adding S_wide to the NIGHT pair is >= 0 in all four
  (+0.18 to +0.85): the admission rule VAL / NIGHT used ("never negative in any variant") is met.
- These numbers are only as good as the MCLmm prior; the sensitivity rows are there so Eesh can pick his own. The .960 column is mostly
  read-noise lottery mass (as NIGHT said for lomm05) and proves nothing if it lands.
- A stack read cannot be decomposed after the fact (one public number). If S_wide moves and S_alt does not, the 10-22 picks are still
  "A = API best-id"; no follow-up re-dose either way (S2 / S5 of lbstrat).

## 7. What would change this

- A separate skeptic that finds a reason the gold74 MCL-MM decoupling does not hold for the hidden experts (for example a labelling
  instruction that counts chronic MCL), or that train-report MCL+ in MM+ knees are mostly acute sprains. Then MCLmm -> 0 and S_safe / S_alt
  lose their point; S_wide falls back to roughly pw25 + lomm05 (+0.73).
- Evidence that the hidden set is gold58-like in composition: weakens lomm05 (NIGHT) more than MCLmm (gold58 reads MCLmm +.0091).
- Eesh preferring the minimum slate (r6pw20 alone) or the NIGHT slate: then this file is information only.

## 8. Files (`session_2026-10-20/stack/`)

- Measurement: `stack_local.py` -> `stack_local.log` / `.json` (G58 + RAK + RAK16 + REP, all singles and 28 pairs); `stack_rak2.py` ->
  `stack_rak2.log` (RAK-u, stacks, MCL placebo partners, co-firing); `stack_mcl.py` -> `stack_mcl.log` (MCLmm raking sets, dose, R7 x
  gold74, G58 dose); `skep_mcl.py` -> `skep_mcl.log` (mini-skeptic); `box_probe_cols.py` -> `box_probe_cols.log`; `box_stack.py` ->
  `box_stack_smoke.log`, `box_stack.log` (OAI aggregates).
- MODEL: `stack_mc.py` -> `stack_mc.log`; `stack_mc_sens.py` -> `stack_mc_sens.log`.
- Builds (unarmed): `stack_build.py`, `kernel_v224r6c4{pw30mcl10,pw25mcl10,pw25lomm05mcl10,pw30lomm05mcl10,pw30mcl00}j/`,
  `build_receipt_*.json`, `registry_entries_stack.json`.
- Gates: `stack_gate.py` -> `stack_gate.log` (STACK_GATE PASS); `stack_probe_check.py` (commit gate, 19 checks); `stack_sim.py` ->
  `stack_sim.log` (STACK_SIM_GATE_SELFTEST PASS), `sim/out_*_sim/`, `sim/out_v224r6c4pw30mcl10j_tamper/`.
- Read only: night/NIGHT_1020.md, night/recal.md, night/adj_c12.py, night/c12_*, levers/LEVERS_1020.md, levers/space.md, levers/box_space.py,
  levers/rubric.md, val/VAL_1020.md, val/implied2.md, val/slate_a22.json, push/e1build.md, push/e1/gate_e1_offline.py, more/c4cells,
  a21/build_a21.py, a21 kernel and committed output, HANDOFF top block.
