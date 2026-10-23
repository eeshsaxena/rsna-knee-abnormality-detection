# regime.md: public-implied regime search over all 66 label pairs (10-21 ~09:00-11:00 IST, Claude, REGIME track). RESEARCH + LOCAL BUILDS ONLY.

Nothing was submitted, pushed, committed, versioned, registered or downloaded. No dataset, no GPU job, no scheduled-task edit. PREREG, HANDOFF,
PROGRESS and the live folders (session_2026-10-20/val incl. a22/, stack/, a20, a21) were only read. Everything I wrote is under
`session_2026-10-21/levers/` (files in sec 8; this folder is shared with the sibling "sink" track, whose files I did not touch, see sec 8 note).
**OAI governance.** No box use at all this round. Every OAI number below is CLAIMED from the aggregate JSON already on the laptop
(`session_2026-10-19/ex/box_coupling.log`, parsed by `oai_claimed.py`: per direction x model means / P only). No OAI id, row or per-knee value was read.
**Tags.** MEASURED = computed this round, log named. CLAIMED = older note, named. MODEL = arithmetic on stated inputs. GUESS = judgement.
**Units.** "label" = single-label AUC delta x1e4; macro e-4 = label / 12. Rulers: RSNA train OOF R7, s1044, s1045 and their rank mean
"3mean" (n 3,467 non-gold rows), always on the pw30 base (E-OA pw .3 applied to the ruler first, as in the kernel). gold58 is INFO only.

## 0. Bottom line

1. **The public reads do NOT say the expert labels are much less coupled than the gold knees show. They say our rulers co-fire less than c4.**
   Two readings of the OA trio both reproduce all five public E-OA reads with ONE parameter per ruler (MEASURED `regime_cal.log`, `regime_dlt.log`):
   - raw gold74 coupling scaled by lambda: lambda_m 3mean .99, s1044 .82, s1045 .77, R7 .57 (sse 1.0-4.9 on 5 reads);
   - population-matched (train report coupling + the SAME-KNEE expert-minus-report difference D on gold58, scaled by gamma): gamma_m 3mean .85,
     s1044 1.12, s1045 1.20, R7 1.51 (sse 1.0-7.8).
   The "if experts are as decoupled as the dose response suggests" regime (gold74 log OR shifted -0.83 / -1.21 sd = lambda .71 / .57) is what a
   SINGLE-model ruler needs; the ensemble ruler (closest to c4's co-firing) needs lambda ~1 / gamma ~.85. So the 2-5x underestimate was the
   ruler's co-firing and raking target, not a hidden "much less coupled" label regime (agrees with night/recal.md sec 0.2, CLAIMED).
2. **NEW: gold58 is not a random sample of the train image population** (MEASURED `popcheck.log`, label-free, R7 OOF predictions only):
   co-firing structure S 2.16 vs null median .88 (p99 1.82), P .001; mean predicted percentile shifted to acute findings (ACL .72, MCL .66,
   Cont .71, Frac .71, Eff .70 vs .50); acute-vs-chronic pairs anti-coupled (R7 MOA-Cont -.40, MOA-Frac -.37, LOA-Frac -.36 vs train).
   Consequence: raking train labels to the RAW gold74 joint (the stack-round RAK, night adj_c12, my G74 / I_FT / IPF) imports gold58's case
   mix, not only the expert label regime. The hidden case mix is unknown (train-like "H_train" or gold-like "H_gold"); the trio cannot tell them
   apart (both readings fit). I therefore score every pair under BOTH and only trust what survives both.
3. **Outside the trio the per-pair expert regime is ~87 % noise.** Split-half reliability of the same-knee D map across the 63 non-trio pairs is
   .13 (MEASURED `regime_dlt.log`). The generic part (mean D -.13, se .036) is label-source dependent: v2 -.13, v4 -.005 (MEASURED). So the
   population-matched "EB" regime used below = trio raw D + for every other pair mean D + .13 x (D - mean D).
4. **Pairs with a consistent positive sign across all rulers AND both population hypotheses (beyond lomm05 / MCLmm):** one family survives
   every check I could run: **MOACF = Medial OA decorrelated from Contusion and Fracture** (MOA <- Cont .05, then MOA <- Frac .05).
   - The stacked line is positive in all 4 plain rulers and all 6 c4-shaped nests under REP, EB, RAW and G74, with unknowns dropped or set
     to 0 (8 / 8 rows ALL+, 80 / 80 values; MEASURED `regime_sens.log`): EB +26..+47 label (+2.2..+3.9e-4 macro), G74 +74..+93 label,
     REP +5..+29 label. The single lines are all-positive under EB / RAW / G74; under REP with unknowns dropped R7 is ~0 to -7.
   - Target-uncertainty bootstrap (100 gold58 redraws of the whole D map) P(>0) 1.00 (3mean) / .91-1.00 (worst ruler); placebo rank of the
     partner among 11 for the MOA target 1-2 / 11, placebo mean ~+0.7 label (MEASURED `regime_boot.log`).
   - c4 co-fires these pairs like R7 on the SAME gold58 images (MOA-Cont -.45 vs -.41; MOA-Frac -.18 vs -.20; experts -.57 / -.39), so the
     weakest ruler (R7) is the c4-relevant one: EB +2.2e-4 (unknowns dropped) / +3.0e-4 (unknown = 0) macro.
   - **Honest:** the implied regime is NOT the only thing making it positive. It is positive even on raw report labels (REP +0.4..+2.4e-4); the
     expert regimes amplify it 1.5-3x. Mechanism (CLAIMED expert2.md): experts call Contusion only for impact edema, reports also code
     degenerative bone-marrow edema and insufficiency fractures, so report labels (and models trained on them) couple MOA with Cont / Frac.
   - **Why it was missed before:** OAI has no Contusion / Fracture grade; gold58 cannot see it (c4 MOA .988 on gold58: the line moves one rank
     pair, +.0016 label, all 4 bases); the stack-round RAK raked only (trio, MM, LM) / (MCL, MM). Only the train-OOF rulers can see it.
   - **MODEL / GUESS value:** ceiling discount (rulers' MOA AUC .88-.90 vs c4's hidden MOA unknown, likely higher: x .35-1) and a skeptic transfer
     haircut like the stack round's (RAK -> MODEL x ~.25-.3) give **+0.8e-4 macro [-1.0, +2.5], P(>0) ~.68** (GUESS), private = public mean.
     Public read noise (paired, ruler resamples) ~1.8e-4 at n 390, ~1.1-1.2e-4 at n 910 (MEASURED `moacf_noise.log`; c4 should be a bit tighter).
     Same order as pw25 (+0.4, CLAIMED VAL) and above lomm05 (+0.17, CLAIMED stack). Lottery-to-modest, not a sure thing.
   - **BUILT (unarmed)**: 4 kernels (pw30, pw25, spl, swide bases + MOACF), offline gate PASS 4 / 4 (sec 5).
5. **What this says about the A22 entries in flight** (MEASURED `regime_sens.log`, `breakeven.log`):
   - **lomm05 (spl, swide):** positive in every population-consistent regime, in all 4 rulers and all 6 nests (EB +0.6..+15 label dropped,
     +8..+19 unknown = 0), REP negative (mechanism-consistent), OAI + (CLAIMED). Supports spl. Weakest ruler is R7 (+0.6 / +8.1).
   - **MCLmm (swide, salt):** NEGATIVE under EB in every ruler and nest (-16..-48 label, unknowns dropped; -29..-46 unknown = 0) and positive
     only under RAW / G74. Break-even: at least 40 % (s1044 / s1045 / 3mean) to 79 % (R7) of MCL-MM's own same-knee D (-.61 v2, -.38 v4)
     must be real. So MCLmm rests entirely on the MCL-MM-specific evidence (stack hypergeometric P .015, rubric), not on the regime search.
     If salt reads below pw25, that is what EB predicts; it would not argue against MOACF (different column, different mechanism).
6. **Everything else is WEAK or DEAD** (sec 4): Bak <- Frac / Frac <- Bak and Bak <- ACL are all-regime positive but rest on 862-1,842 report
   rows with heavy unknowns and Baker's placebos gain ~+7 label too; ACL += OA borrows flip sign between H_train (+) and H_gold (-16..-20 label);
   MM <- LM is killed by OAI (raw, R2, R2lo all 3 / 3 negative, CLAIMED); LOA / PF <- Cont, LOA <- Frac, MM <- MCL, LM <- ACL, MCL <- Bak are
   H_gold-only (R7 negative under EB); every Synovitis pair rests on the 16 % of reports that state Synovitis (n 315-505) and gains on report
   labels too (and Syn transforms were killed before for label-definition reasons, CLAIMED NIGHT_1020).
7. **Post-processing used by .960+ teams that we have not tried** (sec 6): the only buildable one with a measurable edge is the one above (more
   cross-label regime lines). Fold-level rank averaging cannot be measured offline (our serve files are fold-aggregated) and needs the pinned
   worker changed; per-label member selection is already the c4 table and tuning it is noise (CLAIMED nested split-half -.0007); patient
   linkage across test studies is untested and needs a cheap train-side check first.

## 1. What the public E-OA reads imply for coupling (MEASURED `regime_cal.log`, `regime_dlt.log`)

Public (CLAIMED val/implied2.md joint refit, e-4 vs c4): eoa10 5.9, eoa20 10.3, eoa30 9.6, pw20 10.4, pw30 11.3.

| ruler | raw-gold74 reading: lambda_m (sse) | same-knee reading: gamma_m (sse) | REP (report labels) pw30 | IPF-g74 pw30 (adj_c12 parity) |
|---|---|---|---|---|
| R7 | .57 (4.9) | 1.51 (7.8) | -37.9 | -10.3 |
| s1044 | .82 (1.7) | 1.12 (2.1) | -18.0 | +3.8 |
| s1045 | .77 (1.3) | 1.20 (0.9) | -20.0 | +1.2 |
| 3mean | .99 (1.0) | .85 (1.0) | -13.6 | +11.0 (adj_c12 CLAIMED +11.0) |

- Trio-implied common shrink if the label side carried the whole gap (gold74 log OR -k se -> tet): k -0.17 (c4-rho reading) lambda .945;
  k -0.83 (P_FT) .713; k -1.21 (P_AREA) .568. Only the single-model rulers need lambda that low; the ensemble needs ~1.
- Same-knee trio D (v2): MOA-LOA -.36, MOA-PF -.16, LOA-PF -.43 (v4 -.31 / +.04 / -.36). Applied to the train report coupling at gamma ~1 it
  reproduces the public shape and level. The trio is the one place where the same-knee regime difference is both large and public-validated.
- Gaussian-copula raking at lambda 1 reproduces the IPF raking of adj_c12 (3mean 5.6 / 9.0 / 9.0 / 10.3 / 10.9 vs 5.7 / 9.2 / 9.3 / 10.3 / 11.0).

## 2. Why per-pair implied coupling cannot be read off the gold set (MEASURED)

- **Case mix** (`popcheck.log`): gold58 vs 1,000 random 58-row train subsets, R7 OOF predictions only: S 2.161 (null p99 1.819, P .001); mean
  percentile M .298 (null p99 .052, P .000). Biggest drifts: MOA-Cont -.40, MOA-Frac -.37, LOA-Frac -.36, LM-Cont -.35, MCL-MOA -.33.
- **Reliability** (`regime_dlt.log` sec 2): same-knee D (expert - report v2 tet on the gold58 knees) split-half r .067 -> Spearman-Brown .125
  (non-trio .127), the same order as night/recal.md's gold58 gap reliability .14 (CLAIMED).
- **Label source**: D(v2) vs D(v4) correlate .64 across pairs; generic mean -.129 (v2) vs -.005 (v4). The EB regime is internally consistent
  (v2 on train and on gold58) but the generic part should not be read as "experts couple less" in general.
- Per-pair table for all 66 pairs (report tet, gold74 tet, gold58 expert tet, report-v2 tet on the same knees, D v2 / v4, OAI B1 tet, c4 and R7
  co-firing, and every implied target: lambda_FT x gold74, lambda_AREA x gold74, DLT-EB, DLT-EBg, DLT-RAW): `regime_pairs.csv`.
  Largest same-knee D (experts couple less): MOA-Frac -.93 (v2 n 26; v4 -.44), MOA-Cont -.82 (v4 -.13), Bak-Frac -.65, MCL-MM -.61, MOA-Eff
  -.57, MCL-PF -.55, MCL-Bak -.53. Largest positive D: PF-Bak +.34, MCL-Frac +.34, ACL-Cont +.32.

## 3. The 132-direction sweep (MEASURED `regime_sweep.log`, `regime_dlt_sweep.log`, `regime_summary.log`)

Each direction t <- o at a +.05, +.10 (decorrelate) and -.05, -.10 (borrow), 4 rulers, regimes REP (unraked), COP0 (copula control, tracks REP),
IPF / G74 / I_FT / I_AR / RC / A_FT (raw-gold74 family), GEN / EB / RAW / EBg (population-matched family). Raking set = {t, o} plus the trio when
either is an OA label; Gaussian copula with the subset's report marginals.

| regime | all-4-rulers-positive cells, decorrelation / 264 | borrow / 264 |
|---|---|---|
| REP | 8 | 88 |
| G74 (raw gold74) | 71 | 47 |
| I_FT (lambda .71) | 68 | 27 |
| RC (ruler lambda_m) | 69 | 30 |
| EB (population-matched) | 35 | 49 |
| EBg (EB x gamma_m) | 45 | 42 |
| RAW (per-pair D unshrunk) | 58 | 80 |

Classification (`regime_summary.log`): BOTH hypotheses all-positive 41 cells (17 of them Synovitis-subset), H_train only 20, H_gold only 56.
The raw-gold74 regimes double the decorrelation hits, mostly acute-vs-chronic pairs, i.e. the case-mix import of sec 2.

## 4. Short list with every check (MEASURED `regime_boot.log`, `regime_sens.log`, `breakeven.log`; OAI CLAIMED `oai_claimed.log`)

| candidate | EB label 3mean / worst | boot P (3mean / worst) | G74 label | REP label | placebo rank (pl mean) | nests / unknown=0 | c4 vs R7 gold58 rho | OAI | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **MOA <- Cont .05** | +17.0 / +5.3 | 1.00 / .91 | +40.0 | +8.1 / -0.1 | 1 / 11 (+0.7) | all + / all + | -.45 / -.41 (exp -.57) | no label | part of MOACF |
| **MOA <- Frac .05** | +15.9 / +13.0 | 1.00 / 1.00 | +20.5 | +6.5 / +2.2 | 2 / 11 (+0.8) | all + / all + | -.18 / -.20 (exp -.39) | no label | part of MOACF |
| **MOACF (both, .05 + .05)** | +42.2 / +26.2 | (components above) | +89.1 / +78.4 | +16.7 / +4.7 | | nests +38..+45; unk0 R7 +35.7 | | | **PROMISING, built** |
| lomm05 LOA <- MM .05 (in flight) | +14.9 / +0.6 | .97 / .51 | +24.8 | -4.5 / -17.1 | 2 / 11 (+5.4) | all + / all + | +.21 / +.13 | + (raw, R2, R2lo) | supported (weak R7) |
| MCLmm MCL <- MM .1 (in flight) | -17.2 / -48.2 | | +38.8 | -41.4 / -76.1 | | EB all - | +.19 / +.08 (exp -.34) | no label | rests on its own pair D (break-even f .38-.79) |
| Frac <- Bak .05 | +12.3 / +12.3 | .99 / .99 | +24.9 | -0.8 / -0.8 | 1 / 11 (+2.4) | all + / all + | +.12 / +.08 (exp -.27) | no label | WEAK (n 862, Frac acute target, gold reliability ~.01 CLAIMED) |
| Bak <- Frac .05 | +12.5 / +6.2 | 1.00 / .99 | +17.6 | +7.2 / -0.5 | 2 / 11 (+7.0) | all + / all + | same | no label | WEAK (Baker's gains from most partners) |
| Bak <- ACL .05 | +11.2 / +4.2 | 1.00 / .96 | +19.3 | +6.9 / -0.0 | 3 / 11 (+7.1) | all + / all + | -.07 / -.26 | mixed | WEAK |
| ACL += PF / MOA / LOA (a -.05) | +17.6..+22.5 | .98-1.00 | -11.5..-17.2 | +25..+30 | 2-4 / 11 | G74 all - | | raw + (degenerative ACL) | WEAK: sign flips with the population; CLAIMED discarded 10-19 |
| MM <- LM .05 / .1 | +10.7 / +5.0 | 1.00 / .94 | +17.8 | +4.2 / -1.2 | 4 / 11 | | +.31 / +.34 | - in raw, R2, R2lo | DEAD (OAI) |
| LOA <- Cont, PF <- Cont, LOA <- Frac, MM <- MCL, LM <- ACL, MCL <- Bak | worst ruler < 0 | worst .01-.43 | +16..+81 | | | dropped-unknown R7 < 0 | | n/a | DEAD (H_gold only) |
| Synovitis pairs (Syn <- ACL, ACL / PF / MM / Cont += Syn, Syn += MOA) | large + | 1.00 | mixed | large + | 1-2 / 11 | | | OAI Syn = Hoffa (invalid) | DEAD for a slot: n 315-505 selected rows, gains on report labels |

Notes:
- "boot P": 100 gold58 bootstrap redraws of the full D map (so meanD, the trio D and the pair D all move), EB* targets, P(delta > 0).
- "worst" = min over R7 / s1044 / s1045 / 3mean. The c4-shaped nests (6 permutations of the three OOF models in the Raptor / R6 / S6fall
  slots with the c4 table and the .8 / .2 S6F blend) sit between s1044 and 3mean for every row checked.
- gold58 direct (INFO): c4 kernel path MOA .9876 -> .9891 on all four bases (one rank pair); R7 x gold74 +.0027 label for each MOA line but the
  placebo MOA <- Eff also +.0027: n too small to separate (MEASURED `moa_check.log`).

## 5. MOACF build (MEASURED `kernels/build_moacf.log`, receipts `kernels/build_receipt_*moacf05j.json`)

One block in cell 18 after E-OA (and after LOAMM / MCLMM when present), right before `v = out[labels]`:
`MOA = pct(pct(MOA) - .05 pct(Contusion))`, then `MOA = pct(pct(MOA) - .05 pct(Fracture))`, guarded by label presence / blanking, prints
`MOACF_APPLIED`, records `rec['a_moacf']`. Asserts: base nb sha, byte-exact JSON round-trip, only cell 18 differs, the diff is exactly the block,
metadata differs only in id / title / code_file. Offline gate on gold58 (cbase exec route, CPU): the other 11 columns identical, MOA equal to the
python reference, print present, finite 58 x 12.

| base | new tag | nb sha (16) | gate | gold58 INFO macro base -> new |
|---|---|---|---|---|
| pw30 57425970 | v224r6c4pw30moacf05j | 21449a05eb0ab3d2 | PASS | .94336 -> .94349 |
| pw25 (A22) | v224r6c4pw25moacf05j | c6f48d1764f628d4 | PASS | .94368 -> .94381 |
| spl (A22) | v224r6c4pw25lomm05mcl00moacf05j | 70bf132eb8678ee1 | PASS | .94417 -> .94430 |
| swide (A22) | v224r6c4pw25lomm05mcl10moacf05j | e85ceb043cdfeb68 | PASS | .94492 -> .94505 |

All four are UNARMED (NOT_ARMED.txt). Any 10-22 use needs Eesh's go, a PREREG amendment, a registry entry and a commit gate in the
stack_commit_gate.py pattern (not written here). Suggested use (GUESS): ONE entry = MOACF on whatever A22 says is the best base, read against
that base (paired read sd ~1.2-1.8e-4, so a single read is weak evidence either way; under R1 the expected private cost of trying it is small).

## 6. Post-processing plausibly used by .960+ teams, and our status

| idea | what we do now | status / buildable on our outputs? |
|---|---|---|
| fold-level rank averaging inside a member | R6: fold-mean LOGIT, flip TTA averaged in logit space (CLAIMED compact_infer v4c docstring); Raptor 384 / 224 single SWA checkpoints; members rank-blended per label | not measurable offline (gold58 serve files and OOF are fold-aggregated / one fold per row); needs the pinned worker changed. GUESS |effect| < 0.5e-4. WEAK, not for 10-22 |
| per-label member selection / weights | c4 per-label table (Raptor vs R6 weights), S6fall .2 | already in; nested split-half on gold58 says tuning it is noise (CLAIMED PROGRESS_18 -.00073); no OOF for Raptor. DEAD as a new lever |
| cross-label regime adapters (expert vs report structure) | E-OA (validated), lomm05 / MCLmm in flight | the live direction; MOACF (sec 4-5) is the new buildable one. A full 12 x 12 adapter fitted on gold would fit noise (reliability .13) |
| blending operator (logit / geometric / median / probit) | rank average | CLAIMED closed (session_2026-10-19/think/stacking.md: all <= 0 on both rulers) |
| test-time site / scanner normalisation | none | CLAIMED closed (session_2026-10-19/trick/unseen_sites.md) |
| hierarchy / soft-OR consistency (e.g. Synovitis needs Effusion) | none | CLAIMED tested: E1 soft-OR negative on the expert ruler, E2 killed (MORE_1020, NIGHT_1020) |
| more independent members at small weight | Raptor 384 + 224, R6, S6fall | runtime- and GPU-bound, not post-processing; K-series history (CLAIMED) says weak members add <= +.001 |
| patient / visit linkage across test studies (smooth chronic labels like OA within a patient) | none | untested GUESS, low prior; first check train DICOM PatientID duplication on the box (CPU, headers only) before any build |
| transductive pseudo-labelling on the hidden test | none | not feasible inside the hidden-rerun budget before 10-22 |

## 7. Caveats and what would change this

- **Population.** Every train-OOF ruler measures the train population. If the hidden test is gold-like (H_gold), the raw-gold74 numbers apply:
  MOACF is then larger (G74 +6.1..+7.8e-4 macro on the rulers), lomm05 and MCLmm also positive; ACL borrows would lose. If it is train-like
  (H_train), EB applies: MOACF +2.2..+3.9e-4 on rulers, MCLmm negative.
- **Ceiling.** c4's hidden MOA AUC is unknown; rulers sit at .88-.90. The E-OA calibration (rulers matched public level at lambda ~1) suggests
  the OOF ceiling gap did not inflate the trio, but MOA alone may be closer to ceiling (gold58 .988).
- **Multiplicity.** I looked at 132 directions x 4 doses x ~11 regimes. MOACF was picked from ~40 robust cells; the rulers' sampling noise is
  small (n 1,000-3,500, boot P ~1), so the risk is systematic transfer (OOF ruler -> c4 -> expert hidden labels), not noise. The GUESS value in
  sec 0.4 carries that haircut.
- **No expert ruler for Contusion / Fracture.** OAI has neither; gold58 is at MOA ceiling. The only direct expert evidence is one gold58 rank
  pair and an uninformative R7 x gold74 read. A public read of MOACF vs its base is the first real test.
- **Report unknowns.** Fracture is unknown in 56 % of v2 reports and Contusion in 21 %; both handlings (drop, = 0) give the same sign everywhere.

## 8. Files (all under `session_2026-10-21/levers/`)

- `rg_lib.py`: loaders, transforms, weighted AUC, exact and fast tetrachoric, Gaussian-copula targets (exact and MC).
- `regime_cal.py` -> `regime_cal.log/.json`: 66-pair couplings, trio-implied lambda, per-ruler public calibration.
- `regime.py` -> `regime_sweep.log/.json`: 132 x 4 sweep, raw-gold74 family (REP, IPF, G74, I_FT, I_AR, RC, A_FT).
- `regime_dlt.py` -> `regime_dlt.log/.json`: same-knee D map, reliability, gamma calibration.
- `regime_dlt_sweep.py` -> `regime_dlt_sweep.log/.json`: population-matched sweep (REP, COP0, GEN, EB, RAW, EBg).
- `summarize.py` -> `regime_summary.log`; `pairs_table.py` -> `regime_pairs.csv` (one row per pair, every implied coupling).
- `popcheck.py` -> `popcheck.log`; `regime_boot.py` -> `regime_boot.log/.json`; `breakeven.py` -> `breakeven.log`;
  `regime_sens.py` -> `regime_sens.log`; `moa_check.py` -> `moa_check.log`; `moacf_noise.py` -> `moacf_noise.log`;
  `oai_claimed.py` -> `oai_claimed.log/.json` (laptop-side parse of CLAIMED OAI aggregates).
- `build_moacf.py` -> `kernels/kernel_*moacf05j/`, `kernels/build_receipt_*moacf05j.json`, `kernels/build_index_moacf.json`, `kernels/build_moacf.log`.
- Note: my first build run wrote `kernels/build_index.json`, which belongs to the sibling sink track (written 09:53 by build_sink_kernels.py).
  I restored it from that track's seven build receipts (same keys and values as its builder writes) and moved my index to
  `build_index_moacf.json`. The sink track's later readers (registry_sink.py, sink_ksim.py) had already run before the overwrite.
