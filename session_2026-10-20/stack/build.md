# build.md: stack kernels built from scored pw30, offline gate, commit gate, slate value (10-20 ~23:15 to 23:45 IST, Claude, STACK BUILD round)

RESEARCH + LOCAL BUILDS ONLY. Nothing was submitted, pushed, committed, versioned, registered or downloaded. No dataset, no GPU job, no
scheduled task. PREREG, HANDOFF, PROGRESS and the a20 / a21 / val / night / private folders were only read. The scored pw30 committed dir
`a21/out_v224r6c4pw30j_v1` was already on disk, so no `kaggle kernels output` fetch was needed; it was read and copied, never changed.
`val/tick_a22.py` was imported once with `-B` (no bytecode written) to test its GO parser. Everything new is under `stack/kernels/`
(scripts, logs, kernels, sims) plus this file.

**OAI governance.** This round touched no OAI data and did not use the box. Local inputs: gold58 serve files, RSNA train OOF (R7) with
report labels, the RSNA committed pw30 output, and aggregates already in `stack/` / `val/`. Scan: no file under `stack/kernels/` matches
`oai_[0-9]+_[LR]` or `oai_[0-9]{4,}`.

**Tags.** MEASURED = computed in this round, log named (all in `stack/kernels/`). CLAIMED = older note, named. MODEL = arithmetic on
stated inputs (`stack_value_mc.py`). GUESS = judgement. Units: e-4 = 1e-4 macro AUC vs pw30 57425970. Intervals are [p10, p90]. gold58
numbers are INFO only (code check, never a selection ruler).

## 0. Bottom line

1. **All four stacks are rebuilt from ONE base, the scored pw30 notebook** (`a21/kernel_v224r6c4pw30j`, nb sha `5a903187...c0c9fd`, pushed
   v1 10-20 11:42 IST, submitted 57425970 per `a21/tick_a21.log`). The builder asserts the base nb + metadata sha, a byte-exact JSON
   round trip, and an exact cell-18 diff. **Every build is byte-identical (nb + metadata) to the inventory build in `stack/`** (MEASURED
   `build_stack_kernels.log`). For S_alt / S_wide this is a second, independent route (pw30 + dose edit here vs the val pw25 notebook +
   blocks in the inventory). The dose edit alone reproduces `val/kernels` pw25 (64ed7eba) byte for byte.
2. **Offline gate `stack_kgate.py`: STACK_KGATE PASS, 79 / 79 checks** (MEASURED `stack_kgate.log`). It executes each notebook's own cells 14 +
   15 + 18 on gold58 and runs the full on/off factorial of every stack's components (18 combinations). With all components off, each of the
   4 stacks gives output byte-identical to scored pw30, and so does the all-off control kernel. Every changed column matches two
   independent numpy recomputes (max diff 2.9e-8), and every untouched column is byte-identical as CSV text.
3. **Commit gate `stack_commit_gate.py` (26 checks).** On the REAL scored pw30 committed dir it prints PROBE_CHECK PASS 26/26.
   - It also passes on all 4 stack sims, the control sim and a benign repeated-log sim.
   - It FAILS, as required, on all 15 negatives: wrong dose flags, swapped sims, a tampered column, a stale receipt, a conflicting log
     marker, marker order, an extra receipt key, and the inventory-style sim with two EOA markers.
   - Self-test: STACK_KSIM_GATE_SELFTEST PASS, 26 / 26 rows as expected (MEASURED `stack_ksim.log`). Harness check: replaying pw30's
     cell 18 on its real committed inputs reproduces the real `submission.csv` bytes, the receipt's `submission_sha256`, the real log
     lines and the whole blend receipt.
4. **Slate value (MODEL `stack_value_mc.py`, primary):**
   - S_wide is the best single extra entry next to r6pw20 + pw25. Its increment is +1.29e-4 under the stack prior, and it stays
     positive under every MCLmm prior (worst +0.46, recal prior).
   - S_alt (+0.79, worst +0.11) and S_mid (+0.89, worst +0.18) are positive but dominated by S_wide.
   - S_safe is the weakest (+0.52) and the only stack that goes negative (-0.07 under the recal prior).
   - Within the 5-slot cap, {pw25, lomm05, S_alt, S_wide} has the best worst case (+0.95; +1.72 under the stack prior; P(.960 shown)
     .71). {pw25, S_alt, S_wide} has the best stack-prior value (+1.77; worst +0.88). The top sets differ by <= 0.1e-4, which is
     inside the precision of the inputs.
5. **Caveats that bind.**
   - The 3 visible test rows CANNOT show any stack component. All four replayed commit outputs are byte-identical to the real pw30
     `submission.csv` (sha 5b06fff9; MEASURED). Dose identity therefore rests on the registry sha guard (tick G3) plus the receipt and
     log checks (d2 / d4 / d5 / d7 / f).
   - MCLmm (in every stack) has still not had an independent adversarial skeptic. Its prior dominates every number below.
   - The inventory's registry keys `s_safe` etc. break the tick's GO typo guard (sec 5). Use the underscore-free keys here.

## 1. Builds (`build_stack_kernels.py` -> `build_stack_kernels.log`, receipts `build_receipt_<tag>.json`, `build_index.json`)

The cell-18 edits are the inventory's spec (`stack/inventory.md` sec 1), applied to the scored pw30 notebook. Order inside `_s6f_blend` on
the S6F path:
- [DOSE] `A_OA = 0.3` -> `0.25` (pw25 stacks only). This is the only changed line in the E-OA block.
- [LOAMM] 6 lines after the `EOA_APPLIED` print: `LOA = pct(pct(LOA) - .05 pct(MM))`. The block is byte-identical to the night/c12
  lomm05 build (asserted).
- [MCLMM] 6 lines after that: `MCL = pct(pct(MCL) - .1 pct(MM))`.

| stack | components vs pw30 | tag / Kaggle id `eeshsaxena/rsna-sub-<tag>` | nb sha256 | meta sha256 | cell-18 diff | parity with `stack/` build |
|---|---|---|---|---|---|---|
| S_safe | MCLmm | v224r6c4pw30mcl10j | 962a2c9fc642c163616771d8056a9f11b4dcb22adf527ef16f8f9328b88621e7 | 820a89bef0f8c5b804e16f73501243eaedbb91f8861428d94f83c1c40926bfbb | +6 | identical |
| S_alt | pw25 + MCLmm | v224r6c4pw25mcl10j | 051232e424f10840412f9618e56081681e571a6a77815132bac83fa99af2f0d1 | 084cecc2f608ebe025bd47d5c741f858d1c8f5cf4be343798e908a9582f6f924 | 1 line edited, +6 | identical |
| S_wide | pw25 + lomm05 + MCLmm | v224r6c4pw25lomm05mcl10j | c9801a80bbc64ecbd1aba15e9470657fca4631ee7ac091f68b7742d1e17d7478 | 6a2745e7f68841c50f200bdc40e1259e95452237163885c67ed73473ec7f8d1e | 1 line edited, +12 | identical |
| S_mid | lomm05 + MCLmm | v224r6c4pw30lomm05mcl10j | b92bd7846e7a74d3a6a2853f967b851f5632f660d9383462305da75ad4d118fa | b81b5a828840e809f081851adbe39d3ce8bb1115a58487a2297a763b66151179 | +12 | identical |
| CTRL_off (DO_NOT_PUSH.txt) | both blocks at dose 0 | v224r6c4pw30lomm00mcl00j | 8f3601199a5f33bb5f380cad5863684ea931dfae934b3fd186110a095cc490b4 | 87fb8d5a407ab267c6b48246a9688d4bbf2b88cfa54d642329cf5cd8aa433b8c | +12 | n/a (new) |

Builder asserts (all held; MEASURED):
- Base nb sha == `a21/build_receipt_v224r6c4pw30j.json` out_sha256; base metadata sha ee1a7c44...; JSON re-dump == raw bytes (CRLF, like
  every earlier builder).
- Only cell 18 differs from the base, compared as whole-cell JSON (outputs and metadata included). Cell 14 is untouched, so stage 3 is
  pw30's.
- The cell-18 unified diff == exactly the expected `-`/`+` lines (stored in each receipt as `cell18_diff`).
- Metadata differs from the base only in id / title / code_file.
- Each receipt records base path, base nb / meta sha, the doses, components, out sha, the parity result, `builder_sha256` 0fcfc52a... and
  `armed: false`.

## 2. Offline gate (`stack_kgate.py` -> `stack_kgate.log`: STACK_KGATE PASS, 79 / 79)

Pattern of `push/e1/gate_e1_offline.py`. For each notebook the gate EXECUTES cell 14 (stage 3) and cell 15 (V224 canary), emulates cell
16's freeze, then executes cell 18's `_s6f_blend`. Inputs are the gold58 serve files: R384 / R224 raptor, R6 ours, and n42gpu as the
S6fall stand-in. A stage-3 replay is built in, but it is not load-bearing here because cell 14 is identical in every stack (asserted
K2). The stage-3 bytes equal pw30's in every run.

| block | what is checked | result |
|---|---|---|
| H harness | pw30 stage-3 replay byte-identical to the numpy members->c4 construction. gold58 macro pw30 .94336, pw25 .94368, lomm05 .94384 (== a21 / stack gate logs). Clean path for all three | OK |
| K1 / K2 x 5 | nb sha == receipt == build_index; base == scored pw30; only cell 18 differs; cell 14 identical | OK |
| F1 x 4 | each stack with ALL components switched off (exact single-literal substitution in its own cell 18: `A_OA 0.25 -> 0.3`, `A_LOAMM 0.05 -> 0.0`, `A_MCLMM 0.1 -> 0.0`): `submission.csv` byte-identical to scored pw30 (local sha cab96d914225), stage 3 identical, receipt a_oa .3 pw, no a_loamm / a_mclmm | OK |
| C1 | CTRL_off kernel as built: byte-identical to pw30, no LOAMM / MCLMM marker or receipt key | OK |
| F2 x 18 | every on/off combination: changed columns vs pw30 == union of the ON components' columns (dose: OA trio; lomm05: LOA; MCLmm: MCL); every other column byte-identical as CSV text | OK |
| F3 x 18 | all 12 columns == numpy recompute by route A (this run's stage 3 + S6fall B) and route B (member serve files, no notebook code), max diff 2.9e-8 | OK |
| F4 x 18 | receipt a_oa / oa_mode / a_loamm / a_mclmm, and the stdout markers `EOA_APPLIED a_oa x`, `LOAMM_APPLIED`, `MCLMM_APPLIED` in that order before `S6F_BLEND_OK` | OK |
| C2-C5 | MCL column byte-identical in all 4 stacks; S_wide == S_alt and S_mid == S_safe except LOA; S_alt == val pw25 and S_mid == night lomm05 except MCL; S_safe == pw30 except MCL; the MCL change is live | OK |

INFO gold58 macro (n 58, never a ruler): pw30 .94336. S_safe +.00076 (MCL +.0091 label); S_alt +.00108; S_wide +.00156 (LOA +.0097
label); S_mid +.00124. These equal `stack/stack_gate.log`, and the replay output shas equal that log's (aa825b00 / ccb28b69 / bddb6593 /
592d1188), so the cell-14 replay path and the inventory's numpy-V path agree byte for byte.

## 3. Commit gate (`stack_commit_gate.py`; self-test `stack_ksim.py` -> `stack_ksim.log`)

The gate extends `stack/stack_probe_check.py` (lineage: a21/probe_check_eoa21 -> night/c12_probe_check -> stack_probe_check). All 19 of
its checks are kept, with the same (a)-(e) arithmetic. Three of them are stricter:
- (d2), (d4) and (d5) parse the logged dose. Every copy of `EOA_APPLIED a_oa x mode m`, `LOAMM_APPLIED a_loamm x` and
  `MCLMM_APPLIED a_mclmm x` must carry the flag's value.
- Kaggle repeats some log lines (82 distinct lines appear 2-4 times in the real pw30 log, all from worker output; MEASURED). So an
  identical repeat passes and only a conflicting value fails.

Seven checks are added:
- **(d-oa) (d-loa) (d-mcl) (d-rest):** the (d) recompute split per component, so a failure names the component. (d-rest) = the 8
  columns no stack writes must equal the plain .8 / .2 blend.
- **(d6):** marker presence and first-occurrence order, `EOA < LOAMM < MCLMM < S6F_BLEND_OK < S6F_DONE`.
- **(d7):** the blend receipt's keys must equal the scored pw30 key set plus exactly the active stack keys. This catches any other
  transform that writes into the receipt.
- **(f):** the receipts are bound to these files. `submission_sha256` == sha(`submission.csv`), setup `base_sha256` ==
  sha(`submission_v224flip.csv`), and B `probs_sha256` == sha(`B_probs.csv`).

**A tick must key on `PROBE_CHECK PASS 26/26`.** INFO lines (not counted) state whether the visible rows can tell each component from its
absence.

Sims: each kernel's own cell 18 is replayed on the REAL pw30 committed inputs with `os.linesep = '\n'` (Kaggle writes LF). The replayed
stdout lines replace the real log's `EOA_APPLIED .. S6F_BLEND_OK` entries, so the markers sit where a real run prints them.

| row | expectation | got |
|---|---|---|
| HARNESS: pw30 replay on its real inputs | bytes / receipt sha / log lines / blend receipt == REAL | all True |
| POS: REAL scored pw30 dir (.3 / 0 / 0) | PASS 26/26 | PASS 26/26 (also `gate_real_pw30.txt`) |
| POS: S_safe / S_alt / S_wide / S_mid / CTRL_off sims with their own flags | PASS 26/26 | PASS 26/26 x 5 |
| POS: S_wide sim with every E-OA..S6F_DONE log line repeated | PASS 26/26 | PASS 26/26 |
| NEG: real pw30 as S_safe; S_safe sim at amclmm 0 / .2; S_mid at aloamm 0; S_alt as aoa .3; S_mid as aoa .25; S_wide as S_alt; CTRL as S_safe | FAIL | FAIL (d2 / d4 / d5 / d6 / d7 as applicable) |
| NEG tampers on sims: MCL rows swapped; Effusion swapped with stale receipt; 2nd MCLMM marker at .2; MCLMM before EOA; extra receipt key; log .2 vs receipt .1 | FAIL | FAIL ((d) + (d-mcl) + (f); (d) + (d-rest) + (f); (d5); (d6); (d7); (d5)) |
| NEG: inventory sim `stack/sim/out_v224r6c4pw25mcl10j_sim` (appended second EOA marker 0.25 after the real 0.3; LF-normalised file vs CRLF receipt sha) | FAIL | FAIL (d2, d6, f) |
| XCHK: inventory gate `stack_probe_check.py` on the 4 new sims | PASS 19/19 | PASS 19/19 x 4 |
| INFO: inventory gate on the log-value and extra-key tampers | n/a | PASS 19/19 (it misses both; the new gate catches them) |

**Visible-row caveat (MEASURED).**
- All 4 stack replays on the real committed inputs give a `submission.csv` byte-identical to the real pw30 one (sha 5b06fff9). On 3
  rows, neither the dose (.25 vs .3), LOAMM nor MCLMM changes a single rank.
- The commit therefore carries no output evidence of the stack. Identity rests on the tick's registry sha guard (G3: pushed notebook ==
  the gated build) plus (d2) / (d4) / (d5) / (d7) / (f) on the run's own receipt and log.
- This is the same caveat lomm05 had (CLAIMED NIGHT_1020), now made explicit by the INFO lines.

Registry check (`registry_kernels.py` -> `registry_kernels.log`: REGISTRY_KERNELS_CHECK PASS 16/16). It re-implements the tick's
`kernel_ok` and selftest gate step on `registry_entries_kernels.json`: nb + meta sha + slug; the gate argv on each sim, where the last
line of stdout + stderr contains `PASS 26/26` and rc is 0; the keys match `[a-z0-9]+`; and there is no collision with current slate keys.

## 4. Estimates (MODEL `stack_value_mc.py` -> `stack_value_mc.log`)

Logic = `night/night_slate_mc.py` (A = argmax public among pw30 = 0 and the A probes; value = the pick's private) + `stack/stack_mc.py`
component structure (an entry's delta = sum of its components; a component's read error is shared by every entry containing it).
r6pw20 is B and never A (VAL_1020 picks text), so it does not enter A.

Inputs:
- **pw25 = val/implied2.md abc3 primary** (CLAIMED), conditional on the A21 reads: public vs pw30 N(+.09, .80); population N(+.50,
  1.45); private noise 1.32. The corr .17 is fitted so that the slate {pw25} reproduces implied2's +0.37 (this model gives +0.37,
  P(move) .54, P(.960) .20 vs implied2 .22).
- **lomm05 / MCLmm priors** = the inventory's (CLAIMED `stack/stack_mc.py`), shared scenario draw. MCLmm prior variants: stack /
  pessimistic / skeptic-null N(0, 1) / recal N(-.2, 1).
- **I(pw25, lomm05):** N(+.16, .12).
- **Fresh read noise, MEASURED `read_noise.log`.** R7 OOF, 1,000 paired subsets, two report-label sets, random and gold-like MCL
  prevalence. Public / private sd: MCLmm 2.2-2.5 / 1.4-1.6 (confirms the inventory's MODEL 2.5); lomm05 0.99-1.05 / 0.61-0.65
  (reproduces the CLAIMED 1.01). The read-error correlation is -.2 for pw25 vs lomm05 (shared LOA) and ~0 for the other pairs.
- **Gap to the .960 display:** lognormal median 0.75, p90 1.8 (joint, CLAIMED implied2). LB-only median 0.5 is used as a sensitivity.

### 4.1 Per stack (primary; stack prior unless stated)

| stack | E public vs pw30 [p10, p90] | P(public > pw30) | P(.960 display) joint / LB-only | E private [p10, p90] | P(private > pw30) | E private by MCLmm prior (stack / pessimistic / skeptic-null / recal) |
|---|---|---|---|---|---|---|
| S_safe | +0.93 [-2.43, +4.30] | .64 | .50 / .54 | +0.93 [-1.42, +3.30] | .69 | +0.93 / +0.54 / 0.00 / -0.20 |
| S_alt | +1.02 [-2.50, +4.54] | .65 | .51 / .55 | +1.43 [-2.03, +4.88] | .70 | +1.43 / +1.04 / +0.49 / +0.29 |
| **S_wide** | **+1.35 [-2.91, +5.61]** | .66 | **.55 / .58** | **+1.76 [-2.32, +5.86]** | .71 | **+1.76 / +1.37 / +0.83 / +0.63** |
| S_mid | +1.10 [-3.03, +5.24] | .63 | .52 / .55 | +1.10 [-2.12, +4.35] | .66 | +1.10 / +0.72 / +0.17 / -0.03 |
| ref pw25 | +0.09 [-0.93, +1.12] | .54 | .20 / .28 | +0.49 [-2.02, +3.00] | .60 | (no MCLmm) |
| ref lomm05 | +0.17 [-1.91, +2.24] | .54 | .34 / .39 | +0.17 [-1.66, +1.99] | .55 | (no MCLmm) |

Public is lower than private for the pw25 stacks because implied2's public pw25 read is conditioned on the A21 reads, which came in
under expectation (CLAIMED implied2 sec 0.3). The .960 column is mostly read-noise lottery mass, because MCLmm's fresh public read sd is
2.4. A .960 that lands proves little (same caveat as NIGHT gave for lomm05).

### 4.2 Value as an extra slate entry under R1 (increment of E private of A, e-4)

| X added | to {pw25} (r6pw20 B): stack / pessimistic / skeptic-null / recal | worst | P(X becomes A), stack prior | P(A worse than pw30) | P(.960 shown) | to NIGHT {pw25, lomm05}: stack / pess / null / recal | worst |
|---|---|---|---|---|---|---|---|
| S_safe | +0.52 / +0.22 / +0.01 / -0.07 | -0.07 | .58 | .24 | .59 | +0.29 / +0.10 / -0.06 / -0.11 | -0.11 |
| S_alt | +0.79 / +0.47 / +0.20 / +0.11 | +0.11 | .60 | .23 | .57 | +0.54 / +0.32 / +0.12 / +0.04 | +0.04 |
| **S_wide** | **+1.29 / +0.88 / +0.57 / +0.46** | **+0.46** | .62 | .20 | .61 | **+0.90 / +0.54 / +0.29 / +0.20** | **+0.20** |
| S_mid | +0.89 / +0.51 / +0.27 / +0.18 | +0.18 | .59 | .22 | .61 | +0.53 / +0.21 / +0.02 / -0.04 | -0.04 |
| ref lomm05 (NIGHT optional) | +0.33 (all priors) | +0.33 | .46 | .21 | .46 | | |

Baseline {pw25}: E private of A +0.37, P(.960 shown) .20. Sensitivities give the same ranking within +-0.1:
- SENS A: pw25 as stack_mc, with fresh read 1.7.
- SENS B: stack_mc inputs exactly. Its per-entry rows reproduce `stack/stack_mc.log`: S_wide private +1.66 [-2.03, +5.36], P(public >
  pw30) .67, P(.960) .58; S_alt +1.33 [-1.61, +4.28], .66, .55.
- SENS C: LB-only gap. Only the P(.960) values move, up by about .03-.08.

### 4.3 Whole 5-slot sets (r6pw20 + pw25 + up to 3 more A probes; E private of A by prior; ranked by worst case)

| A-probe set | stack / pessimistic / skeptic-null / recal | worst | P(.960 shown) | P(A worse than pw30) |
|---|---|---|---|---|
| {pw25, lomm05, S_alt, S_wide} | +1.72 / +1.36 / +1.05 / +0.95 | +0.95 | .71 | .21 |
| {pw25, lomm05, S_wide} | +1.60 / +1.24 / +0.98 / +0.90 | +0.90 | .66 | .21 |
| {pw25, S_alt, S_wide} | +1.77 / +1.37 / +1.01 / +0.88 | +0.88 | .66 | .20 |
| {pw25, S_wide} | +1.65 / +1.25 / +0.94 / +0.83 | +0.83 | .61 | .20 |
| NIGHT {pw25, lomm05} | +0.70 (all priors) | +0.70 | .46 | .21 |
| {pw25} | +0.37 | +0.37 | .20 | .20 |

Reading (GUESS):
- **If one more slot is used, make it S_wide.** It is the only stack that adds value under every MCLmm prior. Even if MCLmm is null,
  S_wide still carries pw25 + lomm05 (+ their mild super-additivity) in one notebook.
- **If the full cap is used,** {r6pw20, pw25, lomm05, S_alt, S_wide} is the most robust (worst +0.95). Dropping lomm05
  ({r6pw20, pw25, S_alt, S_wide}) costs ~0.07 worst-case and frees a slot.
- S_safe and S_mid are dominated: S_safe = S_alt without the pw25 part, and S_mid = S_wide without it.
- S_alt and S_wide share MCLmm's read error, so they rise and fall together publicly. The pick between them is decided by the LOA
  difference, which is small (lomm05 read sd 1.0).

## 5. For main (nothing here was done; val/ and PREREG untouched)

1. **Registry.** Merge `stack/kernels/registry_entries_kernels.json` into `val/slate_a22.json`. Its keys are `ssafe`, `salt`, `swide`,
   `smid`; the gate is `session_2026-10-20/stack/kernels/stack_commit_gate.py`, `gate_pass` `PASS 26/26`, and the kernel dirs are in
   `stack/kernels/`.
   - Do NOT also merge `stack/registry_entries_stack.json`. It has the same slugs, which risks a double push, the weaker 19-check gate,
     and underscore keys.
   - **GO-parse hazard (MEASURED with `val/tick_a22.py` parse_go):** `'GO pw25 s_wid'` -> `['pw25']`, so the typo is silently dropped.
     `'GO pw25 swid'` -> invalid GO, which is the intended guard. The parser treats a token outside `[a-z0-9]+` as free text.
2. **Sims for `tick_a22.py --selftest`.** Copy `stack/kernels/sim/out_<tag>_sim` to `val/sim/`. Do not copy the inventory's
   `stack/sim/` sims of the same tags: they fail this gate by design on (d2) / (d6) / (f).
3. **PREREG Amendment 22, draft text (NOT written):**
   > STACK (stack/build.md). Kernels built from the scored pw30 notebook (nb 5a903187): swide = rsna-sub-v224r6c4pw25lomm05mcl10j (nb
   > c9801a80) = A_OA .25 + LOA <- MM a .05 + MCL <- MM a .1; [salt = rsna-sub-v224r6c4pw25mcl10j (nb 051232e4)]. Offline gate
   > stack/kernels/stack_kgate.log STACK_KGATE PASS (79/79, all-off byte-identical to pw30); commit gate stack/kernels/stack_commit_gate.py
   > must print PROBE_CHECK PASS 26/26. A probes read by R1 (A = API best-id); never B. No re-dose, no follow-up stack on 10-22.
4. **GO line**, if Eesh picks the robust full set: `GO r6pw20 pw25 lomm05 salt swide` (plus his own text). The key for lomm05 is
   whatever main registered for NIGHT. Pushes are capped at 2 GPU commits at a time (tick G4), and pw30's commit took ~19 min push ->
   gate (CLAIMED `a21/tick_a21.log`). So 5 commits need ~50-60 min before the 05:31 IST window; push by ~04:30 IST.

## 6. What is still open

- **The MCLmm skeptic (not done in this round or the inventory round).** Everything that separates the stacks from the NIGHT slate rides
  on MCLmm's prior. The recal and skeptic-null rows show the floor. S_wide stays >= +0.46 even at the recal prior, because its
  pw25 + lomm05 part does not depend on MCLmm.
- **No ruler is calibrated on a non-OA label** (CLAIMED `stack/inventory.md` sec 3). The MCLmm haircut x .75 is a GUESS.
- **implied2's pw25 is the only component with a public-conditioned model.** lomm05 and MCLmm use fresh read noise, which is the
  correct treatment because no read of either exists yet.

## 7. Files (`session_2026-10-20/stack/kernels/`)

- **Build:** `build_stack_kernels.py` -> `build_stack_kernels.log`, `build_index.json`, `build_receipt_<tag>.json` x 5,
  `kernel_v224r6c4{pw30mcl10, pw25mcl10, pw25lomm05mcl10, pw30lomm05mcl10, pw30lomm00mcl00}j/` (the last has DO_NOT_PUSH.txt).
- **Offline gate:** `stack_kgate.py` -> `stack_kgate.log` (STACK_KGATE PASS).
- **Commit gate:** `stack_commit_gate.py`; `gate_real_pw30.txt` (PASS 26/26 on the real scored pw30 dir); `stack_ksim.py` ->
  `stack_ksim.log` (STACK_KSIM_GATE_SELFTEST PASS); `sim/out_<tag>_sim/` x 5 and `sim/neg_*/` x 7.
- **Registry:** `registry_kernels.py` -> `registry_entries_kernels.json`, `registry_kernels.log` (PASS 16/16).
- **Estimates:** `read_noise.py` -> `read_noise.log` (MEASURED); `stack_value_mc.py` -> `stack_value_mc.log` (MODEL).


## 8. SKEPTIC: independent review of the stacks (10-21 ~00:00 to 00:50 IST, Claude, STACK SKEPTIC round)

RESEARCH + LOCAL BUILDS ONLY. Nothing was submitted, pushed, committed, versioned, registered or downloaded. No dataset, no GPU job, no
scheduled task, no ssh: the box and OAI data were not touched. PREREG, HANDOFF, PROGRESS and the a20 / a21 / val / night / private
folders were only read (HANDOFF line 1795 for the gold74 provenance). Local inputs: gold58 / gold74 eval files, R7 / s1044 / s1045 OOF,
report-v2 labels, the scored pw30 committed dir (copied, never changed). Everything new is under `stack/skeptic/` plus this section.
OAI scan of `stack/skeptic/`: 0 hits for `oai_[0-9]+_[LR]` / `oai_[0-9]{4,}`. Tags as in the header; skep2 = the skeptic MCLmm prior of 8.1.

### 8.0 Verdicts (MODEL `skeptic/skep_value_mc.py` -> `skep_value_mc.log`, PRIMARY inputs of sec 4 except the MCLmm prior)

| stack | verdict | E public vs pw30 [p10, p90] (sec 4.1 value) | P(.960 display) | E private [p10, p90] (sec 4.1) | P(private > pw30) | increment to {pw25}: skep2 (stack prior) | increment over 7 priors (min..max) |
|---|---|---|---|---|---|---|---|
| S_wide | **KEEP** (the MCLmm lottery carrier) | +0.33 [-4.48, +5.10] (+1.35) | .44 (.55) | +0.74 [-3.92, +5.35] (+1.76) | .58 (.71) | +0.89 (+1.30) | +0.32..+1.30 |
| S_alt | **KEEP-optional** (5th slot only, next to spl + swide) | -0.01 [-4.26, +4.30] (+1.02) | .39 (.51) | +0.40 [-3.79, +4.64] (+1.43) | .55 (.70) | +0.58 (+0.80) | 0.00..+0.80 |
| S_safe | **KILL** | -0.10 [-4.23, +4.07] (+0.93) | .37 (.50) | -0.09 [-3.41, +3.31] (+0.93) | .48 (.69) | +0.38 (+0.52) | -0.16..+0.52 |
| S_mid | **KILL** | +0.08 [-4.62, +4.74] (+1.10) | .41 (.52) | +0.08 [-3.82, +3.93] (+1.10) | .52 (.66) | +0.58 (+0.90) | +0.07..+0.90 |
| **S_pl (NEW, built here)** = pw25 + lomm05, MCLmm block at dose 0 | **KEEP** (the robust stack) | +0.43 [-1.90, +2.75] | .40 | +0.83 [-2.28, +3.94] | .63 | +0.63 (+0.63) | +0.63 flat |

The 7 priors: the builder's four (stack / pessimistic / skeptic-null / recal) plus skep2, MCLmm = 0 exactly, and N(-.5, 1). NIGHT lomm05 (on
pw30) is superseded once pw25 is in the slate: {pw25, lomm05} +0.70 vs {pw25, S_pl} +1.00, under every MCLmm prior.

Slates (E private of A vs pw30, e-4; r6pw20 = B, never A):

| A-probe set | skep2 | stack prior | worst over 7 priors | P(.960 shown) skep2 / MCLmm = 0 | P(A < pw30) |
|---|---|---|---|---|---|
| {pw25} (VAL) | +0.37 | +0.37 | +0.37 | .21 / .20 | .20 |
| {pw25, lomm05} (NIGHT) | +0.70 | +0.70 | +0.70 | .46 / .46 | .21 |
| {pw25, S_wide} (builder, one extra) | +1.26 | +1.67 | +0.69 | .52 / .51 | .22 |
| {pw25, lomm05, S_alt, S_wide} (builder, full cap) | +1.42 | +1.73 | +0.83 | .66 / .64 | .23 |
| **{pw25, S_pl}** (one extra, robust) | **+1.00** | +1.00 | **+1.00** | .47 / .46 | .21 |
| **{pw25, S_pl, S_wide}** (two extra) | **+1.48** | +1.68 | **+0.95** | .62 / .60 | .22 |
| {pw25, S_pl, S_alt, S_wide} (full cap) | +1.57 | +1.80 | +0.98 | .66 / .64 | .22 |

Reading (GUESS on MODEL): {r6pw20, pw25, spl, swide} beats the builder's full set under every one of the 7 priors with one slot less. With
a single extra slot, spl is the robust pick (+1.00 flat); swide is higher only if the MCLmm lottery is wanted (+0.26 under skep2, +0.67
under the stack prior, -0.05 to -0.31 under the null-side priors).

### 8.1 MCLmm: the independent skeptic (the open item of sec 6)

1. **All the expert evidence is 9 MCL+ knees in gold58, the sample that nominated MM.** HANDOFF line 1795 (CLAIMED): the 16 extra
   gold74 studies were labelled by reading reports. So (MEASURED `skeptic/skep_mcl_curve.log`):
   - the inventory's "target from the 16 extra only +1.25..+1.64" and "extra16 2 : 0" compare report-v2 with a human report reading,
     not with experts. extra16 holds 2 MCL+ knees (MCL x MM table [[5, 8], [1, 1]]);
   - "LOO 74 / 74 > 0" resamples the same coupling and cannot address the nomination;
   - the gold74 raking target mixes 58 expert and 16 report-read rows. For MCL~MM this makes RAK slightly conservative (log OR -0.95 vs
     gold58 -1.06), so it is not a source of optimism.
2. **The "specificity re-read" is circular.** RAK's gain for MCL <- X is by construction a function of the raked (MCL, X) coupling gap,
   so corr(gain, gap) .75-.78 restates the ruler. On real expert labels the gap did not predict the gain (Spearman .34 over MCL's 11
   helpers, CLAIMED push/census.md sec 9.1).
3. **Partner pooling neither rescues nor kills it** (MEASURED `skep_mcl_curve_gold58.log`). On gold58 the 11 MCL partner gaps are
   heterogeneous (REML tau .62-.64, P(tau < .2) <= .02), and MM's gap +1.95 is typical of the degenerative partners (EB posterior
   +1.89..+1.95). The selection risk sits at the MCL-target level (9 positives; MCL was the largest-excess target of 12, CLAIMED census.md),
   not at the partner level. Recal's MCL reliability .01 is a between-direction statistic, so it does not test a common MCL decoupling.
4. **The ruler and the population move the answer more than the data do.** MCLmm a .1 gain in ruler units (x F for public), 3 c4-shaped
   nests, as a function of the hidden coupling (MEASURED `skep_mcl_curve*.log`, `skep_mcl_relabel.log`):
   - RAK reweight, train population (the inventory's ruler): break-even expert log OR theta* -0.07; at gold58's -1.06 about +3.5.
   - RAK reweight, gold58-like population: train rows density-weighted to gold58 on R7 OOF logits (ESS 1,124-1,323). R7 label-free
     MCL~MM falls .336 -> .19-.21 (gold .12-.14), report log OR +0.90 -> +0.46..+0.48, nest co-firing .52 -> .23-.28 (c4 on gold58 .186).
     theta* -0.73..-0.88; at -1.06 about +0.6..+1.1.
   - RELABEL, the mechanism the inventory actually claims: experts drop report-MCL+ calls, more often in MM+ knees. Drop fraction on
     gold58 = .50 (9 of 18). Break-even drop log OR psi* +1.49 (train) / +0.91 (gold58-like). gold58 psi +2.20 [se 1.03] by expert MM,
     +1.53 [1.09] by report MM, which is what the train emulation conditions on.
   - So at the measured coupling the four corners give roughly +0.5..+3.3e-4 after F ~ .65, and in two corners the break-even lies
     within ~0.3 of the measured value. Any shrinkage of a 9-positive estimate erases most of the effect there.
5. **Skeptic prior skep2 (MODEL).** Posterior on the hidden coupling: measured .25 / mechanism prior .50 (theta N(+.2, .8), psi N(+1, 1),
   each x the gold58 data) / report-anchored skeptic .25 (theta N(+.9, .6), psi N(0, .7) x data). Then ruler 50 / 50 x population (tied to
   lomm05's scenario, P(gold-like) .43) x F U(.45, .85). Result: **MCLmm -0.10 [-2.72, +2.74], sd 2.24, P(>0) .46.** The inventory's stack
   prior was +0.93 [-0.43, +2.34], sd 1.05, P .79. Corner means (reweight-train / reweight-gold / relabel-train / relabel-gold): measured
   +2.29 / +0.96 / -0.14 / +1.56; mechanism +0.78 / -1.15 / -0.37 / +0.89; skeptic -0.73 / -3.32 / -1.35 / -1.62.
6. **Verdict on MCLmm (GUESS on MODEL): not a lever (E about 0), a wide lottery.** The reversal of the census / recal KILL rests on the
   nominating sample read through one ruler corner. MCLmm stays admissible only inside an entry that carries real expected value of its
   own (S_wide, optionally S_alt), never alone (S_safe). Its R1 value is option value from its variance (8.5), not expectation.

### 8.2 Hidden negatives in the other components

- **pw25:** no new negative. RAK-u about 0 (-0.5..0.0), implied2 population +0.50 (CLAIMED inventory sec 2, val/implied2.md). Every
  pw25 stack inherits its risk.
- **lomm05:** the same evidence shape as MCLmm: gold58 rank 1 of 132, placebo P .083 / .25, OAI c4 members at the public regime
  -.0010..-.0013 label, RAK raked to a gold74 target with report-read rows (CLAIMED night/NIGHT_1020.md sec 1). The inventory's +0.17 is
  kept; a skeptic N(0, .9) with I tied to 0 is the sensitivity: S_pl +0.28, S_wide +0.64 (skep2), lomm05 alone +0.11 over {pw25}.
- **I(pw25, lomm05) +0.16:** measured where lomm05's mechanism is live (OAI pub, RAK), but drawn independent of lomm05's scenario. Under
  the lomm05 skeptic it is tied to 0 (row above). This is the only soft double count found.

### 8.3 Double counting, and a claim to correct

- **Arithmetic: none.** MCLmm is column-disjoint (exact add); lomm05's single reads are on the pw30 base and I is added once on top.
  Re-running the builder's PRIMARY with its own priors reproduces sec 4.2 (S_wide +1.30 vs +1.29, S_alt +0.80 vs +0.79, S_mid +0.90
  vs +0.89, S_safe +0.52; {pw25} +0.37; NIGHT +0.70; MEASURED `skep_value_mc.log`).
- **Evidence: yes, once.** The stack prior's S1 (+2.3, RAK at the gold coupling) and S3 (gold58 +7.6 x .42, shrunk) both read the same
  9 gold58 positives. The mixture treats them as alternatives, not a sum, but the x .75 haircut is the only selection correction on
  evidence that is all nominating-sample.
- **Correction to 0.4 / 4.3:** "S_wide is the only stack that adds value under all four MCLmm priors" is not what the sec 4.2 table
  shows: S_alt (min +0.11) and S_mid (min +0.18) are positive under all four next to {pw25} too. S_wide is the best, not the only one.
  Under the null-side priors added here (MCLmm = 0, N(-.5, 1)) S_safe reaches -0.16 and S_alt 0.00.

### 8.4 Is P(.960) overstated? Moderately, and it is noise either way

- S_wide alone .55 -> .44 (skep2); {pw25, S_wide} .61 -> .52; the builder's full set .71 -> .66 (MODEL).
- With MCLmm exactly 0 the same slates still show .51 / .64. A .960 from an MCLmm stack says almost nothing about MCLmm.
- The computed brief says ".960 needs roughly +1-2e-4"; val/implied2.md line 19 says 0.75 median joint, 0.5 LB-only (CLAIMED). The builder
  used implied2, which is the source. If the gap were median 1.4 (p90 2.5), {pw25, S_wide} would show .41 (skep2) / .50 (stack prior).

### 8.5 Private cost of adopting a stack under R1 by public noise (MODEL, `skep_value_mc.log` sec 6, slate {pw25, X})

- MCLmm's public read sd 2.4 decides A about half the time. Under skep2, S_wide is A in 50 % of worlds:
  - 17 % with MCLmm < 0 (mean -1.04): expected cost -0.18e-4;
  - 33 % with MCLmm > 0: +0.65;
  - net E[MCLmm x 1(S_wide = A)] +0.47 (stack prior +0.76, skeptic-null +0.13, recal +0.03, N(-.5, 1) -0.10).
- Against its MCLmm-free twin S_pl, adopting S_wide is worth +0.67 (stack) / +0.26 (skep2) / -0.05 (null) / -0.16 (recal) / -0.17
  (MCLmm = 0: its noise only dilutes the lomm05 selection) / -0.31 (N(-.5, 1)).
- P(A worse than pw30) stays .20-.24 in every slate: the stacks do not fatten the tail, they spread A across entries.
- E[MCLmm | S_wide = A] is +0.94 while E[MCLmm] is -0.10 (skep2): the value is selection on a wide prior, so it holds only if public and
  private are a random split of one population.

### 8.6 Build: S_pl (MEASURED `skeptic/kernels/build_spl.log`: SPL_BUILD PASS)

| item | value |
|---|---|
| tag / Kaggle id | `v224r6c4pw25lomm05mcl00j` / `eeshsaxena/rsna-sub-v224r6c4pw25lomm05mcl00j` (registry key `spl`) |
| kernel dir | `stack/skeptic/kernels/kernel_v224r6c4pw25lomm05mcl00j/` (NOT_ARMED.txt) |
| nb sha256 | e641e88fcaa5c41f85a7c7e8302bdeb6df3bd0a55d30ae669fcde5c2c8d5446b |
| meta sha256 | 258fb790aab28123d6f1c4241c29da24677a5083fb457c474e6dc0428d606011 |
| how | S_wide nb (c9801a80) with ONE literal `A_MCLMM = 0.1   #` -> `0.0   #` in cell 18 (bytes edit, 1 occurrence): the exact substitution stack_kgate.py replayed as S_wide[dose+loamm] (F2 / F3 / F4 OK, `stack_kgate.log` lines 50-52). Metadata = S_wide's with the tag swapped (3 occurrences). |

Checks (all OK):
- **B:** source shas; vs S_wide only cell 18 differs, vs scored pw30 only cell 18 differs, cell 14 identical; cell 18 == S_wide's with
  the one substitution; metadata differs only in id / title / code_file.
- **O (gold58, notebook cells executed with stack_kgate's own replay):**
  - pw30 harness .94336;
  - S_pl bytes == S_wide replayed with A_MCLMM 0.0;
  - 12 columns == the numpy recompute final_from(V, .25, .05, 0), maxdiff 2.9e-8;
  - changed vs pw30 = OA trio, vs val pw25 = Lateral OA only;
  - receipt a_oa .25 pw, a_loamm .05, no a_mclmm; markers EOA .25 < LOAMM .05 < S6F_BLEND_OK.
  - INFO gold58 macro .94417 (+.00081 vs pw30; n 58, never a ruler).
- **S (commit-gate sim):** the pw30 harness replays its real committed bytes / receipt / log exactly. The S_pl sim's 3-row
  submission.csv is byte-identical to the real pw30 one (sha 5b06fff9): the visible rows cannot show S_pl either.
  - `stack/kernels/stack_commit_gate.py` with `--aoa 0.25 --aloamm 0.05 --amclmm 0.0`: **PROBE_CHECK PASS 26/26**.
  - FAIL as required: S_pl as S_wide (d5 / d6 / d7), as pw25 (d4 / d7), as night lomm05 (d2); the builder's S_wide sim with S_pl flags
    (d5 / d7); the real pw30 dir with S_pl flags (d2 / d4 / d6 / d7).
- **Registry** (`registry_entry_spl.json`, `registry_spl_check.log`: REGISTRY_SPL_CHECK PASS 3/3): nb + meta sha + slug; key `[a-z0-9]+`
  with no key or slug collision with val/slate_a22.json or stack/kernels entries; gate argv on the sim ends in PASS 26/26 with rc 0.

### 8.7 For main (nothing here was done)

1. **Registry.** If spl is adopted, merge `stack/skeptic/kernels/registry_entry_spl.json` (key `spl`) next to
   `stack/kernels/registry_entries_kernels.json` (`swide`, optional `salt`). Do not merge `ssafe` / `smid`. Before
   `tick_a22.py --selftest`, copy `stack/skeptic/kernels/sim/out_v224r6c4pw25lomm05mcl00j_sim` to `val/sim/`, plus the builder's
   swide / salt sims.
2. **GO lines** (examples): robust `GO r6pw20 pw25 spl swide`; full cap `GO r6pw20 pw25 spl salt swide`. The pw30-based lomm05 is then
   not pushed. 4 commits need about 40 min at 2 concurrent and 5 about 60 min (~19 min each, CLAIMED a21/tick_a21.log).
3. **PREREG.** The NIGHT draft for lomm05 says "no stacking with pw25 or any other winner" (CLAIMED NIGHT_1020.md sec 4). spl and swide
   both stack it, so Amendment 22 must say so. Draft text (NOT written):
   > STACK (stack/build.md sec 8). spl = rsna-sub-v224r6c4pw25lomm05mcl00j (nb e641e88f) = pw30 with A_OA .25 + Lateral OA <- MM a .05
   > (MCLmm block at dose 0); swide = rsna-sub-v224r6c4pw25lomm05mcl10j (nb c9801a80) = spl + MCL <- MM a .1; [salt =
   > rsna-sub-v224r6c4pw25mcl10j (nb 051232e4)]. This replaces the NIGHT lomm05 rule "no stacking with pw25": lomm05 is read only inside
   > spl / swide and the pw30-based lomm05 is not pushed. Offline: stack/kernels/stack_kgate.log STACK_KGATE PASS (the S_wide[dose+loamm]
   > row is spl) and stack/skeptic/kernels/build_spl.log SPL_BUILD PASS. Commit gate stack/kernels/stack_commit_gate.py must print
   > PROBE_CHECK PASS 26/26 with each entry's flags. A probes read by R1 (A = API best-id); never B. MCLmm is a lottery (skeptic E
   > -0.1e-4, P(>0) .46): no re-dose, no MCL <- other partner, no follow-up stack on 10-22.
4. Every slot still needs Eesh's GO.

### 8.8 Files (`session_2026-10-20/stack/skeptic/`)

- **MCLmm rulers:**
  - `skep_mcl_curve.py` -> `skep_mcl_curve.log` (gold74, as the inventory) and `skep_mcl_curve_gold58.log` (`gold58` arg; expert only).
    Contents: partner table, bootstrap covariance, REML / EB pooling, reweight curves over theta for the train and gold58-like
    populations, posterior maps.
  - `skep_mcl_relabel.py` -> `skep_mcl_relabel.log`: gold crosstabs (drop vs confirm by MM) and relabel curves over psi.
- **Value:** `skep_value_mc.py` -> `skep_value_mc.log` (skep2 prior, 7-prior robustness, gap sensitivity, 5-slot sets incl. S_pl, R1 cost
  decomposition).
- **Build:** `kernels/build_spl.py` -> `kernels/build_spl.log`, `kernels/kernel_v224r6c4pw25lomm05mcl00j/`,
  `kernels/build_receipt_v224r6c4pw25lomm05mcl00j.json`, `kernels/sim/out_v224r6c4pw25lomm05mcl00j_sim/`, `kernels/registry_entry_spl.json`,
  `kernels/registry_spl_check.log`.
