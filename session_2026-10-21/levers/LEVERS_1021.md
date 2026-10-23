# LEVERS_1021.md: 10-22 lever synthesis, keyed to the A22 outcome (10-21 ~11:30 IST, Claude, LEVERS synthesis)

RESEARCH + LOCAL BUILDS ONLY. Nothing was submitted, pushed, committed, versioned, registered, downloaded or turned into a dataset. No
GPU job, no scheduled-task edit. PREREG, HANDOFF, PROGRESS and the live folders (session_2026-10-20/val incl. a22/, stack/, a20, a21)
were only read; the REAL committed dirs (a21/out_*, val/a22/out_*) were copied into sims, never changed. This synthesis step did not use
the box. The SINK track used it once (read-only CPU script over ssh stdin, aggregates only; sink.md); the REGIME track used none (OAI
numbers CLAIMED from session_2026-10-19/ex/box_coupling.log aggregates). OAI id scan of `levers/` (`oai_[0-9]+_[LR]`, `oai_[0-9]{4,}`): 0 hits.
**Tags.** MEASURED = computed this round, file named. CLAIMED = older note, named. MODEL = arithmetic on stated inputs. GUESS = judgement.
**Units.** e-4 = 1e-4 macro AUC (label AUC x1e4 / 12). "Increment" = change in E private of A under the read rule, vs A22 alone.
gold58 (G58) numbers are INFO only. Sources: sink.md, regime.md (this folder), the skeptic logs `rakpub_skep.log`, `skep_moacf*.log`,
`skep_lomm05*.log`, and `lever_value_mc.log` (new, this step). Readtable / plan1022 / DAY_1021 in `session_2026-10-21/day/` (CLAIMED).

## 1. Plain verdict

1. No lever clearly earns a 10-22 slot. The default stays readtable.md sec 8: nothing on 10-22 except a one-off r6pw20 resubmit if it ERRORs.
2. The one optional slot with positive expectation is **MOACF(W)** = the A22 best-id W + Medial OA decorrelated from Contusion, then Fracture (a .05 each).
3. MOACF(W) value (MODEL `lever_value_mc.log`): +0.26e-4 private (+0.19 if only stack-class W, H2), +0.20 even if public and private effects are independent; P(>0) ~.66; ~+0.1 silver pts; far below read noise (paired public sd ~1.8).
4. By A22 outcome: spl -> moacfspl, swide -> moacfswide, salt -> moacfsalt. pw30 / pw25 under H2: no lever (A = pw20; moacfpw20 exists, needs its own clause). Under R1 as written: moacfpw30 / moacfpw25.
5. Runner-up CONS(W) = W + convex E-OA (CVX): +0.26 mostly via prior variance, +0.09 when public and private disagree; gold58 against it; weakest on pw30. Only if Eesh prefers an OA-trio lever, never next to MOACF.
6. Never: the kitchen sink, MOACF + CVX together (one kernel or two entries), re-added lomm05, any re-dose.
7. 10-22 order: 05:31 IST r6pw20 retry (if ERROR) and MOACF(W) (if GO). Public-only twins only AFTER the MOACF read (~11:00-12:30 IST); otherwise the 3-decimal API cannot tell W from MOACF(W).
8. Every lever entry needs Eesh's GO, an A23 amendment, a registry merge and its commit gate PASS. Draft A23 text in sec 3.4.

## 2. Ranked proposals with evidence and skeptic verdicts

### 2.1 Table (e-4; increment MODEL `lever_value_mc.log`, world() = stack/skeptic/skep_value_mc.py verbatim, R1 over A22 + entry)

| # | proposal | spec | evidence (tagged) | skeptic verdict | increment E private [stack-only] | P(>0) | built / gated | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | **MOACF(W)** | cell 18 after E-OA (and LOAMM / MCLMM): `MOA = pct(pct(MOA) - .05 pct(Cont))`, then `- .05 pct(Frac)`; marker MOACF_APPLIED, receipt a_moacf | MEASURED regime_sens / skep_moacf: positive in all 4 rulers, all 6 c4-shaped nests, report labels v2 / v4 / sol56, EB built from v2 or v4, both population hypotheses; EB +2.2..+3.9 macro on rulers, REP +0.4..+2.4. Not visible to OAI (no Cont / Frac grade) or gold58 (MOA at ceiling) | skep_moacf*: WEAK-positive. Not a v2-labeler artefact, but partner not specific (MOA <- ACL ties Frac), so it is a generic "MOA vs acute" explain-away fix of report-trained models; ceiling unknown; no expert ruler. Corrected prior E +0.35 [q05 -0.50, q95 +1.30] | **+0.26 [+0.19]**; rho0 +0.20 [+0.15]; pess +0.11; optim +0.51; flip -0.15 | .66 (skeptic) | yes: 6 bases, offline gate PASS 6/6, commit gate PASS 29/29, self-test PASS 51/51, registry PASS 18/18 | **WEAK, best available. Optional single slot** |
| 2 | CONS(W) = W + CVX | E-OA subtracts `_U[o] ** 2` (P_OA 2) at W's dose; marker CVX_APPLIED, receipt p_oa | MEASURED sink: RAK-PUB +1.64 (pw25) / +1.49 (spl) / +0.75 (pw30); OAI pub P_AREA +2.7, P_FT +0.2..+0.5; gold58 -7.7 / -12.6 (P .04-.15) | rakpub_skep: **KILL the RAK-PUB ruler** (units artifact, regime not pinned, 5 rungs fix one number); corrected CVX prior N(+0.3, 1.1) pw25 family, N(+0.1, 1.1) pw30. CVX@pw30 -0.24..+0.97 across routes | +0.26 [+0.19] (pw30 row +0.18); rho0 +0.09 [+0.07]; lskep +0.13 | ~.60 pw25 family, ~.53 pw30 (GUESS, vote) | yes: 6 stacks + control, SINK_KGATE 165/165, commit gate 27/27, self-test 43/43, registry 18/18 | WEAK. Runner-up only |
| 3 | MOACF(pw20) under H2 | pw20 57425269 + MOACF; needs "A = better public of pw20 and moacfpw20" | as #1 (dose-independent on rulers: v4 REP 3mean +19.1 / +20.5 / +22.9 label on pw30 / pw25 / no-E-OA bases, skep_moacf2 MEASURED). G58 INFO here -1 rank pair (MOA .9907 -> .9891) vs +1 on the other 5 bases | not separately skeptic-checked; same prior as #1 | ~+0.26 conditional on W in {pw30, pw25} (GUESS by symmetry; pw20 is outside world()) | .66 | yes, gate PASS 29/29 on its sim | OPTIONAL, not recommended (extra H2 clause) |
| 4 | BOTH = MOACF(W) and CONS(W) as two entries | two A-eligible entries | MODEL | its gain over MOACF alone is selection on prior variance; P(new = A and worse) .30 vs .21 | +0.41 [+0.30]; **rho0 +0.21 = MOACF alone**; flip +0.02 | | both built | DEAD (2 slots, more curse, no robust gain) |
| 5 | KS(W) = W + lomm05 + CVX | sink.md sec 3.1 | MODEL sink_value_mc: never beats CONS; re-adds a lomm05 that lost its head-to-head | sink: DEAD | <= CONS everywhere | .36-.38 | yes | DEAD |
| 6 | RAK-PUB as a calibrated ruler | RSNA OOF raked to trio shift s* +0.23 | MEASURED fits 5 rungs (SSE 1.6-4.1) | rakpub_skep **KILL**: s* +0.23 on T32 = -0.43 sd in the pairwise convention; Dirichlet s* sd .38; LORO RMSE .45-1.33 for every route; INFO values route-dependent (pw25-pw30 -0.73..+1.51, lomm05@pw25 +0.85..+2.53, MCLmm uncalibrated +3.27) | n/a | n/a | script only | INFO-only ruler |
| 7 | lomm05 / MCLmm added anew | as stack/inventory.md | MEASURED skep_lomm05: v4 / sol56 EB R7 -3.4 / -4.0 label, gold-like case mix v4 EB nest mean -2.2..-3.2, placebo MM rank 6 / 11; regime: MCLmm EB negative in every ruler (-16..-48 label), break-even f .38-.79 | lomm05 MC E +0.06 central [-1.51, +1.62] (stack +0.17); MCLmm skep2 -0.10 | ~0 or below | ~.5 | in A22 already | DEAD as additions (they ride inside spl / swide / salt) |
| 8 | other regime pairs | Bak <- Frac, Frac <- Bak, Bak <- ACL, ACL += OA, MM <- LM, LOA / PF <- Cont, LOA <- Frac, MM <- MCL, LM <- ACL, MCL <- Bak, Synovitis pairs | MEASURED regime sec 4: low n / heavy unknowns / placebos gain too; ACL borrows flip sign with population; MM <- LM negative on OAI (CLAIMED); H_gold-only pairs R7 negative under EB; Synovitis on 16 % of reports | regime: WEAK / DEAD | <= 0 skeptic-aware | | no | DEAD for 10-22 |
| 9 | older near-misses | SUBH, S6up, RPF, E1, BOR / C13, C6', C14, CVXd, killed list | MEASURED sink sec 1 on pw25 under RAK-PUB: SUBH -0.14, RPF -0.25, BOR -2.6, C6' -0.44 (OAI 0 / 5), C14 -0.7; E1 +3.7 only via unraked report labels; CVXd dominated by CVX | sink + earlier skeptics: OUT | <= 0 | <= .5 | no | DEAD |
| 10 | post-processing of .960+ teams | fold-level rank averaging, per-label member tuning, patient linkage, blend operator, site normalisation, pseudo-labels | regime sec 6: not measurable offline / already in / CLAIMED closed / untested | GUESS |effect| < 0.5e-4 | ~0 | | no | not for 10-22 |

### 2.2 MOACF: why it ranks first, and its weak points
- **Robust on every ruler we have** (MEASURED `skep_moacf.log`, `regime_sens.log`, `regime_boot.log`):
  - REP (raw report labels), MOACF on pw30, label x1e4, R7 / 3mean: v2 dropUNK +4.7 / +16.7, v4 +12.6 / +19.1, sol56 +13.5 / +20.8; nests ALL+.
  - EB built from v4 and scored on v4 (self-consistent): R7 +22.0, 3mean +30.2, nests +28.5..+32.0. From v2: +26.2..+42.2.
  - Base-independent: v4 REP 3mean +22.9 (no E-OA) / +20.5 (pw25) / +19.1 (pw30) (`skep_moacf2.log`).
  - gold58 target bootstrap P(>0) 1.00 (3mean) / .91-1.00 (worst ruler); partner placebo rank 1-2 / 11.
- **Mechanism** (MEASURED `skep_moacf2.log` sec C): the gain sits in MOA-negative knees with a high acute score, which move down more
  (-.040 mean pct, 3mean) than MOA-positive acute-high knees (-.024). Report-trained models read bone-marrow edema of acute injury as
  medial OA evidence; experts call Contusion only for impact edema (CLAIMED expert2.md).
- **c4 behaves like the weakest ruler** (MEASURED): on gold58, c4 MOA~Cont rho -.51 vs R7 OOF -.45 (s1044 / s1045 co-fire +.41 on train,
  R7 +.01). So R7 is the c4-relevant ruler, and R7 is positive in every row (REP v4 +12.6 label = +1.05e-4 macro).
- **Weak points:**
  - No expert ruler: OAI has no Contusion / Fracture grade; gold58 MOA sits at .988, one rank pair moves (+.00013 macro on 5 bases,
    -.00013 on pw20; the placebo MOA <- Eff moves the same, `moa_check.log`).
  - Partner specificity is weak (MEASURED placebo field): v4 EB 3mean Cont +19.1, ACL +14.5, Frac +14.4, Eff +10.0, MCL +9.5. It is a
    "MOA vs acute findings" family, picked from ~40 robust cells out of 132 directions x 4 doses.
  - Ceiling: rulers' MOA AUC .89-.90; c4's hidden MOA is unknown and probably higher, which shrinks the gain.
  - gold58 is acute-enriched (P .001, `popcheck.log`). If the hidden set is gold-like, the rulers say MOACF is larger (G74 +6..+8e-4
    on rulers); if train-like, +2..+4. Both are positive.
- **Value** (MODEL): skeptic mixture (w .65 transfers, lognormal median .55; w .25 no bias in c4, N(-.1, .05); w .10 sign flip, N(-.5, .3))
  gives E +0.35, P(>0) .66. In the A22 world this is +0.26e-4 for one entry (+0.20 rho0). The regime track's GUESS +0.8 [-1.0, +2.5]
  gives +0.82 (+0.51 rho0). Read sd 1.8 / 1.15 public / private (rulers, MEASURED `moacf_noise.log`); 1.5 / 1.0 gives +0.28.

### 2.3 CVX: why it drops to runner-up
- Its evidence came from two "calibrated" rulers (RAK-PUB, OAI P_AREA pub). The skeptic killed RAK-PUB as a calibrated ruler (`rakpub_skep.log`):
  - the +0.23 shift is -0.43 sd in the pairwise convention;
  - the 5 public rungs (themselves MODEL posterior means, not 3-decimal reads) fix about one number;
  - both rulers rake to the same gold74 trio table with the level fitted to public information (double counting).
- gold58 is strongly negative (-7.7 pw25, -12.6 spl; -3.2 / -5.3 public-equivalent) and the levers' R7 x gold74 read was negative (CLAIMED).
- p 2 was chosen among 13 shape families on OAI (selection-exposed).
- On pw30, CVX ranges -0.24..+0.97 across calibration routes (MEASURED).
- With the corrected prior, its value is mostly R1 selection on prior variance: +0.26 at rho 1, +0.09 at rho0 (MODEL).
- CVX x lomm05 overlap on Lateral OA: mild on rulers (-0.06..-0.26), total on gold58 (-4.8).

### 2.4 Value model (MODEL `lever_value_mc.log`; A22 alone: E private A +1.57, P(.960 shown) .66, P(A < pw30) .22)

| MOACF prior / CVX prior | MOACF(W) all W [stack-only] | CONS(W) | BOTH (two entries) | P(entry becomes A and is worse), MOACF / CVX / BOTH |
|---|---|---|---|---|
| skeptic mixture / rk N(+0.3, 1.1) | **+0.26 [+0.19]** | +0.26 [+0.19] | +0.41 [+0.30] | .21 / .21 / .30 |
| optim / rk | +0.51 [+0.38] | +0.26 | +0.59 | .18 / .21 / .27 |
| pess / rk | +0.11 [+0.08] | +0.26 | +0.31 | .23 / .21 / .32 |
| regime N(+0.8, 1.37) / rk | +0.82 [+0.60] | +0.26 | +0.85 | .14 / .21 / .23 |
| zero / zero | 0.00 | 0.00 | 0.00 | .25 / .25 / .37 |
| sign flip N(-0.5, .5) / lskep | -0.15 [-0.11] | +0.13 | +0.02 | .24 / .22 / .35 |
| skeptic / primary N(+0.5, 1.0) | +0.26 | +0.34 [+0.26] | +0.47 | |
| **rho0** (entry's private effect independent of public): skeptic / rk | **+0.20 [+0.15]** | +0.09 [+0.07] | +0.21 [+0.15] | |
| rho0: optim / primary; pess / lskep; regime / rk; flip / lskep | +0.42; +0.07; +0.51; -0.20 | +0.20; +0.03; +0.09; +0.03 | +0.43; +0.07; +0.43; -0.10 | |

- Flat (within .02) across lomm05 skeptic prior, MCLmm zero and the LB-only gap.
- P(.960 shown) .66 -> .80 with MOACF (zero prior still .77: read noise). P(A < pw30) .22 -> .24 (all W) / .21 (stack-only); .27 / .23 with
  a zero prior.
- Scale: +0.2e-4 is ~+0.1 silver pts at the GUESS slope .5 pt per 1e-4 (sink.md), against a family shock of about +-10 pts (CLAIMED shakeup.md).

### 2.5 What this round says about the A22 reads (INFO for 14:30-17:00 IST, no action)
- lomm05 (spl, swide): positive in every population-consistent regime (regime.md), but v4 / sol56-label EB is weak and R7-negative,
  placebo-unspecific, and negative under a gold-like case mix (skep_lomm05). Central E +0.06. A spl < pw25 read is unsurprising.
- MCLmm (swide, salt): EB-negative in every ruler. It rests only on its pair-specific gold58 evidence (11 MCL+ knees). salt < pw25 is
  what EB predicts.
- None of this touches MOACF (different column, different mechanism, no gold58 dependence). It also does not change the A rule (H2 default,
  readtable sec 8).
- KEY LESSON revision (MEASURED, both tracks): the 2-5x public underestimate of E-OA was mostly ruler co-firing, not a "much less
  coupled" expert label regime. RSNA-ruler label-side shift is -0.16..-0.43 sd across target constructions (pairwise convention, rakpub_skep) vs the OAI -0.83; the ensemble
  ruler fits all 5 rungs at lambda ~.99 of raw gold74 coupling (`regime_cal.log`).

## 3. Built candidates (all UNARMED; NOT_ARMED.txt in every dir)

### 3.1 MOACF set (gate `levers/kernels/moacf_commit_gate.py`, 29 checks, tick keys on `PROBE_CHECK PASS 29/29`)

| W (A22 best-id) | registry key | Kaggle slug `eeshsaxena/rsna-sub-<tag>` | base (nb sha8) | nb sha256 | meta sha256 (16) | flags --aoa / --aloamm / --amclmm | G58 INFO macro |
|---|---|---|---|---|---|---|---|
| **spl** | moacfspl | v224r6c4pw25lomm05mcl00moacf05j | spl A22 57464296 (e641e88f) | 70bf132eb8678ee162bc74e9eb5d779ee4c4dfa47a976e6c43b53e8efd75023b | 76d5ad5baf5f813b | .25 / .05 / 0 | .94417 -> .94430 |
| **swide** | moacfswide | v224r6c4pw25lomm05mcl10moacf05j | swide A22 57464326 (c9801a80) | e85ceb043cdfeb68b42107e0bb0621a2f37e371f2441b8bd845de78d54d3b7ef | f249c914848cd105 | .25 / .05 / .1 | .94492 -> .94505 |
| **salt** | moacfsalt | v224r6c4pw25mcl10moacf05j | salt A22 57464911 (051232e4) | 2993857e8fa00e182a2239b4c6751f9477887ccdcd7c85c5abf47e13fddbf283 | 05ad340756a162cf | .25 / 0 / .1 | .94444 -> .94457 |
| pw25 (R1 only) | moacfpw25 | v224r6c4pw25moacf05j | pw25 A22 57463742 (64ed7eba) | c6f48d1764f628d46d26ede9ce897cb45d4f2c5b53255518b16b3440134e0aac | f2a5d23984021b93 | .25 / 0 / 0 | .94368 -> .94381 |
| pw30 (R1 only) | moacfpw30 | v224r6c4pw30moacf05j | pw30 57425970 (5a903187) | 21449a05eb0ab3d2f8e799eb540b98626477cb2059e9dddef13f1c8cf0de2385 | 03fde194bc128fac | .3 / 0 / 0 | .94336 -> .94349 |
| pw20 (H2 option) | moacfpw20 | v224r6c4pw20moacf05j | pw20 57425269 (09ba09b9) | e69adf9b2101fad9dc107cd408bc042a5a2d1ff76e693a69f910b4a32a8969aa | 2ab1e630ba9797ba | .2 / 0 / 0 | .94378 -> .94365 |

Kernel dirs: `session_2026-10-21/levers/kernels/kernel_<tag>/`. Receipts: `kernels/build_receipt_<tag>.json`. Index:
`kernels/build_index_moacf_all.json`. Registry entries (not merged anywhere): `kernels/registry_entries_moacf.json`.

Gate command on the fetched commit dir `<OUT>` (C4T as in stack/inventory.md sec 1):
```
C4T='{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker'"'"'s": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}'
py -3.12 -I session_2026-10-21/levers/kernels/moacf_commit_gate.py <OUT> --a224 0.25 --table "$C4T" --aoa A --mode pw --aloamm L --amclmm M --poa 1 --amoacf 0.05
# A / L / M per row of the table above; must print PROBE_CHECK PASS 29/29
```

Checks behind "ready" (all MEASURED this round):
- **Builds:** pw30 / pw25 / spl / swide by the regime track (`kernels/build_moacf.log`). salt and pw20 by `build_moacf_salt.py` /
  `build_moacf_pw20.py`; these exec `build_moacf.py` with only BASES and the log / index names changed, under asserts:
  - base nb sha;
  - byte-exact JSON round trip;
  - only cell 18 differs;
  - the diff is exactly the 8-line block;
  - metadata differs only in id / title / code_file;
  - offline gold58 gate: the other 11 columns are identical and MOA equals the python reference.
- **Builder sha:** the current `build_moacf.py` (sha 019e1083) differs from the builder sha in the four regime receipts (2fe7b021). A scratch
  rebuild with the current file reproduces all four notebooks and metadata byte-identically, so the edit did not touch the block.
- **Commit gate:** `make_moacf_gate.py` -> `kernels/moacf_commit_gate.py` (on-disk sha ae5925ee). It is the sink gate plus:
  - `--amoacf`, with expect() applying MOACF after MCLMM;
  - (d-oa) split into PF OA and a new (d-moa) for Medial OA;
  - new (d9): receipt a_moacf == [a, a], and every MOACF_APPLIED copy carries a;
  - (d6) order ... MCLMM < MOACF < S6F_BLEND_OK;
  - (d7) key set + a_moacf.
- **Self-test** `kernels/moacf_ksim.py` -> `kernels/moacf_ksim.log`: **MOACF_KSIM_GATE_SELFTEST PASS (51 rows, harness True)**.
  - Harness: each base notebook's cell 18 replayed on its own REAL commit (pw20, pw30, pw25, spl, swide, salt) reproduces bytes, receipt
    sha, log lines and the blend receipt.
  - POS PASS 29/29: the 6 sims, the 6 REAL commits at a_moacf 0, a benign repeated log, and two sink CVX sims at p 2 (the MOACF gate is a
    superset of the sink gate).
  - NEG FAIL:
    - each REAL commit checked as MOACF;
    - each sim checked at a 0;
    - wrong lomm / mcl / dose / a / p flags;
    - tampers: MOA row swap, conflicting marker, marker before EOA, marker before MCLMM, receipt [.1, .05], extra key, marker removed.
  - XCHK: the parent sink gate FAILS every MOACF sim on (d7) and PASSES 27/27 on the REAL pw30 commit.
- **Registry:** `kernels/registry_moacf.py` -> `kernels/registry_moacf.log`: **REGISTRY_MOACF_CHECK PASS 18/18**.
  - Each kernel's sha, metadata and slug match its entry.
  - Each gate argv run on its sim gives PASS 29/29.
  - No key or slug collides with val/slate_a22.json or registry_entries_sink.json.
- **Caveat (as for every stack):** all REAL commits and sims share a byte-identical 3-row submission.csv (sha 5b06fff9). MOACF cannot be told
  apart from its absence on the 3 visible rows (gate INFO), so its identity rests on (d9) / (d6) / (d7) + the registry sha guard. The block
  skips silently if the r224 -> Fracture canary blanks Fracture; the gate's (a) canary check and (d9) would then FAIL the commit.

### 3.2 CVX set (SINK track; gate `levers/kernels/sink_commit_gate.py`, `PROBE_CHECK PASS 27/27`; sink.md sec 3)

| W | key | slug tag | nb sha256 | flags A / L / M, `--poa 2` |
|---|---|---|---|---|
| spl | cvxspl | v224r6c4pw25lomm05mcl00cvx2j | 5b4bbd791aba590481532edb872603197a861a073d98b993078c2de9eb2029b9 | .25 / .05 / 0 |
| swide | cvxswide | v224r6c4pw25lomm05mcl10cvx2j | b6915d970693b7fad46ce97718e69c74a6ffb8c6e2a70d0eed87aaba3414993b | .25 / .05 / .1 |
| salt | cvxsalt | v224r6c4pw25mcl10cvx2j | ebeb31f8bd47565116eb49cc4b3123d1d20e8bcb68e412743b73b9710eb5d436 | .25 / 0 / .1 |
| pw25 | cvxpw25 | v224r6c4pw25cvx2j | 015598db1998f135c27542778257e4618bb0b4b5942d3a51118a2b8365e980cc | .25 / 0 / 0 |
| pw30 | cvxpw30 | v224r6c4pw30cvx2j | f3203107c8b97aa9cc654bdf7924fb48bb6c9dbeb51c2aed08feb2c6853de3a2 | .3 / 0 / 0 |
| (KS, DEAD) | cvxlomm30 | v224r6c4pw30lomm05cvx2j | 268938cef52c906a091be889e11700b5a23bd4c916bb1c3d188ed3680a26dfd1 | .3 / .05 / 0 |
| control, NEVER push | | v224r6c4pw30cvx1j (DO_NOT_PUSH.txt) | a0a3467ca44b5765da3d355d93d6b86afe301c193e42b93e9c953c0201b5dde5 | |

```
py -3.12 -I session_2026-10-21/levers/kernels/sink_commit_gate.py <OUT> --a224 0.25 --table "$C4T" --mode pw --aoa A --aloamm L --amclmm M --poa 2
```
Gates (MEASURED, sink.md): SINK_KGATE PASS 165/165; SINK_KSIM_GATE_SELFTEST PASS 43/43; REGISTRY_SINK_CHECK PASS 18/18. Shas re-hashed this step: match.

### 3.3 Mechanics for a 05:31 IST 10-22 submit (CLAIMED plan1022 / DAY_1021 unless tagged)
- The RSNA_A22 tick closes 10-22 05:20 IST. Any 10-22 entry needs an A23 tick (main builds it after the GO) or a manual submit after 05:30 IST.
- Push-to-gate is ~19-20 min per commit, so push by ~05:00 IST. Last readable submit ~10:15 IST; a 05:31 submit reads ~11:00-12:30 IST.
- Before a tick self-test:
  - copy `levers/kernels/sim_moacf/out_<tag>_sim` into the tick's sim dir;
  - merge the chosen key from `kernels/registry_entries_moacf.json`;
  - the gate line is `PASS 29/29`.
- Read: A = the better public of W and MOACF(W), via the API best-id after the MOACF read. This works only while no other 10-22 entry has
  scored (the submission list shows 3 decimals; MEASURED readtable sec 8). So public-only twins (p30L / ssafe / smid) are submitted only
  after the MOACF read. They still score well before the 10-23 05:29 IST deadline (scoring lag 4.8-7.1 h). MOACF(W) is never B.
- Order at 05:31 IST: r6pw20 retry (only if ERROR) first, then MOACF(W).
- If an A22 entry is still PENDING at 05:30 IST, hold MOACF until it is COMPLETE, because W is not known yet.

### 3.4 Draft A23 text (NOT written into PREREG; main appends it only on Eesh's GO)
> LEVERS (session_2026-10-21/levers/LEVERS_1021.md). One optional 10-22 A-eligible entry, MOACF(W), where W = the API best-id after the
> A22 reads and W is a stack: spl -> rsna-sub-v224r6c4pw25lomm05mcl00moacf05j (nb 70bf132e), swide ->
> rsna-sub-v224r6c4pw25lomm05mcl10moacf05j (nb e85ceb04), salt -> rsna-sub-v224r6c4pw25mcl10moacf05j (nb 2993857e). [Only if the A rule
> stays R1: pw25 -> rsna-sub-v224r6c4pw25moacf05j (nb c6f48d17), pw30 -> rsna-sub-v224r6c4pw30moacf05j (nb 21449a05).] Each = W's own
> notebook + Medial OA <- Contusion .05 then <- Fracture .05 in cell 18, nothing else. Commit gate levers/kernels/moacf_commit_gate.py
> must print PROBE_CHECK PASS 29/29 with the entry's flags. A = the better public of W and MOACF(W), read from the API best-id before any
> other 10-22 entry scores; MOACF(W) is never B. No CVX, no re-dose, no second lever entry, no follow-up.
> [Optional H2 clause, not recommended: if H2 sets A = pw20, enter rsna-sub-v224r6c4pw20moacf05j (nb e69adf9b) and A = the better
> public of pw20 57425269 and that entry.]

## 4. What not to do
1. **No kitchen sink and no double lever.** MOACF and CVX both write Medial OA and their interaction is unmeasured. As two entries, the
   second only adds selection on prior variance (BOTH +0.21 vs MOACF +0.20 at rho0) at P(entry is A and worse) .30 vs .21.
2. **No re-adds or re-doses:**
   - no lomm05 where its head-to-head lost (KS);
   - no MCLmm added anew (EB-negative in every ruler);
   - no lomm05 .08, MOACF .1, CVX p 3, or E-OA dose above .30 (A22 cap).
   These re-select on noise already seen.
3. **Do not use RAK-PUB or OAI P_FT / P_AREA numbers to read the A22 entries or to pick W.** Their INFO values move with the calibration
   route by more than the effects (pw25-pw30 -0.73..+1.51). MCLmm has no regime calibration at all.
4. **Do not rake non-trio pairs to raw gold74 couplings.** gold58 is acute-enriched (P .001), and the non-trio same-knee D map has
   reliability .13. Raking imports case mix, not expert regime.
5. **Do not read the KEY LESSON as "experts are much less coupled, decorrelate harder".** On RSNA rulers the label-side shift is only
   -0.16..-0.43 sd; the gap was mostly ruler co-firing.
6. **Do not let any public-only entry score before the MOACF read,** and do not let Kaggle auto-select. Eesh selects A and B by hand with
   both ids written down, by 10-23 05:29 IST.
7. **Do not push without the four steps:** GO + A23 + registry merge + commit gate PASS on the fetched commit. Never push the CVX control
   (v224r6c4pw30cvx1j).
8. **Do not spend a slot on:** SUBH, S6up, RPF, E1, BOR / C13, C6', C14, CVXd; the Baker / Fracture pairs; ACL borrows; MM <- LM;
   Synovitis pairs; the H_gold-only pairs; fold-level rank averaging (not measurable offline, needs the pinned worker changed);
   patient linkage (untested; a train PatientID header check on the box comes first); pseudo-labelling.
9. **Do not treat any gold58 delta as selection evidence.** CVX -7.7 / -12.6 and MOACF +-1 rank pair are INFO only (n 58).

## 5. Files written by this synthesis step (all under `session_2026-10-21/levers/`)
- `LEVERS_1021.md` (this file).
- `lever_value_mc.py` -> `lever_value_mc.log`: MOACF / CVX / BOTH value in the A22 world, with rho0 and read-sd sensitivities.
- `build_moacf_salt.py`, `build_moacf_pw20.py`: write `kernels/kernel_v224r6c4pw25mcl10moacf05j/` and
  `kernels/kernel_v224r6c4pw20moacf05j/`, their build receipts, `kernels/build_moacf_{salt,pw20}.log`,
  `kernels/build_index_moacf_{salt,pw20}.json` and `kernels/build_index_moacf_all.json`.
- `make_moacf_gate.py` -> `kernels/moacf_commit_gate.py`.
- `kernels/moacf_ksim.py` -> `kernels/moacf_ksim.log`, `kernels/sim_moacf/` (6 sims, 8 negatives).
- `kernels/registry_moacf.py` -> `kernels/registry_entries_moacf.json`, `kernels/registry_moacf.log`.
- No other track's file was edited. The scratch rebuild check ran in the session scratchpad.
