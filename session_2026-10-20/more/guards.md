# guards.md: zero-public-cost hidden-test guards for c4 (10-20 ~01:45 IST, Claude subagent)

RESEARCH + MEASUREMENT ONLY. Nothing was pushed, committed or submitted. No dataset was created. No scheduled task, PREREG,
HANDOFF, PROGRESS or a20 file was edited (a20 files were only read). No OAI data was used: gold58 and the 458 DICOM-header
studies are competition TRAIN studies.

Box use: one sequential chain on the idle V100, 20 min in total (box clock 01:04:44 to 01:24:52). nvidia-smi was empty before
every job. Everything was written under `~/guards1020` only, and nothing was killed.

Tags: MEASURED (computed here, file named), CLAIMED (a prior note or code comment says so), MODEL (arithmetic on stated inputs),
GUESS (judgement).

What was audited:
- the c4 notebook `session_2026-10-18/a18/kernel_v224r6a25c4j` (cells dumped to `more/c4cells/`);
- the decoded fastread, R6 sources and S6F driver in `session_2026-10-19/trick/mp_work/`;
- the R6 mainline modules `session_2026-10-15/jpeg/src/r7code/native_cache.py` (sha c53ce898) and `native_data.py` (f59afe20).
  Both match c4's CODE_SHA pins.

Work files are in `more/work/`: scripts g1 to g7 with their `.log` files, and box outputs in `more/work/box/`.

## 0. Verdict: no guard is worth building. Every idea is DEAD or WEAK.

| idea | lens | no-op on visible + train | P(fires on hidden) | measured value if it fires | EV (macro) | verdict |
|---|---|---|---|---|---|---|
| G-FLAG: canonicalise the series flags (both := min(Fluid, Fat), plus NaN / invalid / plane case) | a | yes, MEASURED (0 rows changed: 15 visible rows, 24,371 train rows) | systematic ~.01-.02, any row ~.03-.05 (GUESS) | **-.0012 +- .0013** on gold58 (the guard is WORSE than doing nothing, P .82) | ~0 (-3e-5 to +1e-5) | DEAD |
| Raptor-only "prefer (1,1) over (1,0)" | a | yes | same | -.0005 vs doing nothing on gold58; leaves 27 / 458 picks and all R6 inputs off | ~0 | DEAD |
| G-NAN: only NaN / invalid flags and odd plane spellings | a | yes | ~.01 (GUESS) | removes a real Raptor crash (all-zero volume, unflagged) | ~1e-6 | DEAD (negligible) |
| F1 / F2 rank-neutral fill of single-member rows (stage 3) | b | yes (structural) | R6 fallbacks ~0, Raptor GPU failures <= 3 % | +1e-5 at 1 % failed rows, +4e-5 at 5 % (MEASURED, OOF simulation) | ~1e-6 | DEAD (c4's fill is already near-optimal) |
| S6F unscored-row fill, both-failed rows, unmasked r224 failures | b | yes | ~0 | <= .2 x the line above | ~0 | DEAD |
| Raptor all-zero volume -> mark the study failed | b | yes | ~.02-.05 (GUESS) | ~1-2e-4 per affected study (MODEL) | ~5e-6 | DEAD |
| any S6F canary guard | c | n/a | ~.01-.02 per NEW run (MODEL; 0 drops in ~20 runs, CLAIMED) | a lost read only; final picks are already-scored runs | 0 | DEAD |
| (side finding, not a guard) feed R6 "definition" fluid flags | lever | NO (changes the visible German study) | always | c4 +.0007 +- .0009 on gold58; driven by 1-3 studies | unknown | WEAK |

**Bottom line.** The flag mismatch the data page warns about is real as a possibility: members do read the two flags
inconsistently, and Raptor degrades by about .003 on its own. But it is unlikely on hidden (P <= .02-.05), and when it is
simulated on gold58, c4 as a whole is not harmed (+.0012 +- .0013). R6's Effusion stacker gains about what Raptor loses.

The fills and canaries in c4 are already the right design. Nothing here is worth a cell, let alone a slot.

The one strictly-positive fix is the Raptor NaN-flag crash (section 1.5). It fires with P ~.01, so its EV is ~1e-6.

## 1. (a) Fluid_Sensitive vs Fat_Suppression

### 1.1 How each c4 member reads the two flags (MEASURED, code read)
| member | flag used | where | a (1,0) row (PD / T2 without fat sat; fluid-sensitive by the data-page definition) | a (0,1) row (T1 with fat sat) |
|---|---|---|---|---|
| Raptor r384 / r224 slot picker | Fluid_Sensitive only | `_fastread._pick_series_for_slot` (cells 4 / 11, base64): `int(r.get('Fluid_Sensitive', 0) or 0) == fluid` | becomes an FS-slot candidate (slot 0 Sag FS, slot 2 Cor FS); the first by SeriesUID order wins | becomes a non-FS slot candidate |
| R6 / S6fall slot embedding | **Fat_Suppression** | `native_cache.prepare` l.112-116: `slot = SLOTS[(plane, fs)]`, where fs = Fat_Suppression | non-FS slot (as in train) | FS slot |
| R6 / S6fall meta column 7 "fluid" | Fluid_Sensitive | native_cache l.117 -> px_cache l.179 / l.224 -> compact_train l.261 (meta6) | 1 (train never pairs a non-FS slot with fluid 1) | 0 |
| R6 Effusion fluid stacker | meta fluid | compact_infer l.123: FS sagittal = meta fluid == 1, then the lowest-median one | a new FS-sagittal candidate; a no-FS-sagittal study leaves the "miss" path | not a candidate |
| R6 / S6fall knee anchors | fluid_sensitive | knee_anchor_bank_v2 l.155: sagittals sorted FS first | may put a non-FS sagittal first | as in train |

How each member handles a NaN flag:
- **Raptor:** `int(nan or 0)` raises ValueError inside build_study. `_pre_one` (cell 4 l.463-469) catches it and returns an ALL-ZERO volume, and the study is NOT flagged as failed.
- **R6:** NaN becomes -1, which selects slot 7-9 (untrained rows of a 10-row embedding) and sets fluid to -1.

The members do not agree on which column means "fat sat". Raptor and R6's meta read Fluid_Sensitive, while R6's slot reads
Fat_Suppression. This corrects TRICK_1019 sec 7, which says every member keys on Fluid_Sensitive.

### 1.2 What the flags mean in train and in the visible test (MEASURED)

**Train** (`g1b_tags.py`, 458 train studies with DICOM headers, 2,529 series):
- Fat sat is read from ScanOptions FS / SFS / WE, or from a description containing FS / FATSAT / SPAIR / SPIR / STIR / TIRM / WE / F/S, or from IR. Weighting is read from TR / TE.
- 463 series are PD or T2 WITHOUT fat sat. **462 of them are flagged (0,0)** and 1 is (1,1).
- 5 series are T1 WITH fat sat, including `RT.SAG_T1W_TSE_FS+C`. **All 5 are (0,0).**
- The train flag equals "fat sat by DICOM" on 99.45 % of series. The rest is classifier noise (for example "SAG PD NON FATSAT").
- So the train rule is: **flag = 1 only for fluid-bright, fat-suppressed sequences** (PD-FS, T2-FS, STIR, DESS-WE), which is min(definition-Fluid, definition-Fat).
- The data page's own definition would give 467 mismatched rows on these 458 studies.

**Visible test** (`vis_hdr.py` / `vis_hdr.log`, DICOM headers of the 3 visible studies on the box):
- `pd_tse_tra_d`, `t2_tse_sag_d`, `pd_tse_sag_d` and GE `Sag PD FSE` (no FS option) are all **(0,0)**.
- Under the definition, 4 of the 15 visible rows would be (1,0). None is.
- **The visible test CSV follows the train rule, not the definition.**

**Hidden** (GUESS):
- P(the hidden CSV follows the definition systematically) <= .02. It comes from the same host pipeline as the visible CSV, and 0 of 24,371 train rows and 0 of 15 visible rows mismatch.
- P(a few sporadic odd rows) is about .03-.05.
- The data-page sentence reads as a generic caveat.

### 1.3 Exposure if the hidden CSV did follow the definition (MEASURED, `g2_exposure.py`, pickers copied from the code)
DEF flags are the shipped flags with PD/T2 non-FS set to (1,0) and T1-FS set to (0,1), read from DICOM headers (`g2_def_flags.csv`).

| set | studies with a changed row | Raptor slot 0 (Sag FS) changed | Raptor slot 2 (Cor FS) changed | any Raptor slot changed | R6 meta changed | fluid-stacker candidate added |
|---|---|---|---|---|---|---|
| 458 header studies | .653 | .273 | .090 | .358 | .653 | .552 |
| gold58 | .672 | .397 | .103 | .483 | .672 | .603 |

### 1.4 Harm on gold58 (MEASURED, box V100, `g5_harm.log`, `g5b_members.log`, `g5c_c4labels.log`)
Setup:
- **Raptor r384 and r224:** run with the BASE and with the DEF series CSV. The code is c4 cells 4 / 11, byte-exact except for paths, weights_only and a raw-probability dump. r384 ran on CUDA here.
- **R6:** run_b's exact command (c4 cells 7-9) on a shadow data dir that holds the DEF CSV, with symlinked DICOMs. Result: 58 / 58 scored, 0 fallbacks, compile active.
- **S6fall:** NOT rerun; it uses BASE in both arms.
- **Sanity checks:** r384 GPU BASE vs the CPU kernel serve has min per-label Spearman .9999. Studies without a DEF change move by <= 1.9e-4 (Raptor) and 0 (R6).

| arm (gold58, 39 / 58 studies affected) | macro | delta vs BASE | boot sd | P(delta < 0) |
|---|---|---|---|---|
| r384 alone, DEF | .9194 | -.0035 | .0029 | .89 |
| r224 alone, DEF | .9123 | -.0027 | .0026 | .87 |
| R6 alone, DEF | .9388 | +.0022 (Effusion +.027, LM -.001, Syn +.0006; nothing else moves) | .0014 | .00 |
| **c4, BASE (= what G-FLAG gives under DEF)** | **.94087** | | | |
| c4, DEF in every member (status quo under DEF) | .94204 | **+.00117** | .00130 | .18 |
| c4, DEF in Raptor only | .94059 | -.00028 | .00137 | .58 |
| c4, DEF in R6 only (= a Raptor-only guard under DEF) | .94154 | +.00067 | .00094 | .22 |

Reading:
- Raptor does lose about .003 on its own, mostly on ACL, MCL, LM, PF OA and Fracture.
- R6 gains on Effusion. One German no-FS-sagittal study with expert Effusion 0 leaves the stacker's miss path (+.0174 of R6's +.027); two other studies add +.0099 and +.0037.
- Through c4 the net is +.0012 +- .0013. **There is no harm for G-FLAG to remove: on this ruler the guard would cost .0012.**
- The final c4 ranks on the affected studies barely move (per-label Spearman mean .994, min .988).
- EV (MODEL): P(fires) .01-.05 x (-.0012 to +.0005) is about -3e-5 to +1e-5. **DEAD.**

### 1.5 G-FLAG function and offline gate (kept for the record; NOT recommended to build)
The function is `gflag_canon` in `more/work/g6_gate.py`:
- A row is canonical if both flags hold the same value in {0, 1} and the plane is exactly Sagittal, Coronal or Axial.
- If any row is not canonical, it writes a copy of test_series.csv with both flags := min(Fluid, Fat).
  - A missing or invalid flag takes the other flag's value.
  - If both are missing, it writes -1, which R6 already uses for NaN.
  - A plane is renamed when it matches one of the three names after stripping spaces and folding case.
- If no row needs a change, it writes nothing.

Gate results (MEASURED, `g6_gate.log`, `g7_prefer.log`):
- **Visible test_series.csv:** 0 rows changed, frame identical.
- **train_series.csv (24,371 rows):** 0 rows changed, frame identical.
- **DEF (458 studies, 467 rows changed):** with no guard, Raptor picks differ in 164 studies and R6 records in 299. After G-FLAG, 0 and 0.
- **Stress tests on train (200 random rows each):**
  - *Fluid NaN:* with no guard, Raptor RAISES in 145 of 196 studies, giving an all-zero volume that is not flagged. With G-FLAG there are 0 crashes and 0 differences.
  - *Fluid = 2:* 44 Raptor picks change; with G-FLAG, 0.
  - *Lower-case plane:* 172 Raptor and 194 R6 changes; with G-FLAG, 0.
  - *Both flags NaN:* G-FLAG removes the 152 crashes, but 44 picks still differ because the information is gone.
- **The narrower "prefer (1,1) over (1,0)" guard:** 27 of 458 Raptor picks are still off, and all R6 records are untouched.

How the guard would be wired in (spec only, not built; G-NAN would be the same wiring restricted to NaN / invalid / plane rows):
1. A new cell after cell 2 runs gflag_canon. If it fires, it builds `/tmp/gflag/rsna-knee-abnormality-detection/` with symlinks plus the canonical CSV and sets `GFLAG_ROOT`.
2. In the Raptor cells 4 / 11: `ROOT = os.environ.get('GFLAG_ROOT') or find_test_root()`.
3. In cell 7: `DATA = Path(GFLAG_ROOT)` when it is set.
4. In the S6F driver: the same override for `I.comp`. This means re-encoding the driver and updating DRIVER_SHA.

The R6 shadow-dir mechanism itself is proven by the box DEF run. With no fired rows, every member runs its original code path.

## 2. (b) Failed-row fills

### 2.1 Stage 3 single-member rows (cell 14 l.75-77): c4's fill is already the right one (MEASURED, `g4_fill.log`)
c4's fill (F0): a study that one member failed takes the other member's percentile as it is.

Alternatives that change nothing when no study fails:
- **F1:** quantile-map that percentile onto the blend's own distribution.
- **F2:** impute the missing member's percentile as E[r_missing | r_present] (20 bins, no labels used).
- **F3:** fill with 0.5.

Simulation setup:
- Members: TTR s1044 OOF (the Raptor-recipe slot model) and R7 OOF (the R6 family), 4,333 train studies.
- Blend: c4's per-label weights.
- Failures at random, 20 reps.
- Labels: 2 report label sets, used only to check the mechanism, not to select anything.

| failed share | F0 loss vs no failures | F1 - F0 | F2 - F0 | F3 - F0 |
|---|---|---|---|---|
| 0.5 % | 3-9e-5 | +0.1 to +0.7e-5 | -0.4 to +0.7e-5 | -.0010 to -.0012 |
| 1 % | 1.0-1.3e-4 | +1.0 to +1.1e-5 | +0.5 to +1.3e-5 | -.0020 to -.0023 |
| 2 % | 2.1-2.5e-4 | +0.8 to +1.9e-5 | -1.3 to +1.2e-5 | -.0041 to -.0048 |
| 5 % | 5.8-6.8e-4 | +2.9 to +4.6e-5 | +1.6 to +2.7e-5 | -.0107 to -.0121 |

- F1 wins in 57-90 % of reps, but only by about 1e-5 at realistic failure rates.
- Hidden failures are bounded: more than 3 % Raptor GPU failures stops the notebook, and R6 had 0 fallbacks in all 54 commits and in gold58 (CLAIMED, kernel_audit 3.4). So the EV is ~1e-6. Keep F0.
- A 0.5 fill (F3) would cost about .002 per 1 % of rows. That is the trap to avoid, and c4 avoids it.

### 2.2 Other fills (MODEL / code read)
- **S6F unscored rows** take the V224 rank in the S6 slot (cell 18 `rs[~ok] = rb[~ok]`). That is the same F0 shape at weight .2, on <= 2 % of rows (the coverage gate). Worth <= 1e-6.
- **Both members failed** gives 0.5 in stage 3. S6fall uses R6's reader, so it has almost surely failed on the same study too. ~0.
- **r224 GPU failures are not masked.** Cell 14 masks with r384's failures only, and r224's 0.5 rows enter as ordinal ranks (cell 11 writes ranks only). This needs r384 to succeed and r224, the lighter net, to fail on the same study. Dormant, ~0.
- **A Raptor all-zero volume** (prep raised, or every slot empty) is not flagged as failed (kernel_audit 4.1). The NaN path in 1.1 is one way to reach it. A one-line `mask.sum() == 0 -> leave 0.5` in cells 4 / 11 would route the study to R6's rank. EV ~5e-6 (MODEL: P ~.02-.05 that any hidden study hits it, times ~1-2e-4 per study).

## 3. (c) S6F canary paths (code read of cells 16-18 and the driver; rates are MODEL)
What fires the canary:
- a setup failure (decoder self-test, environment);
- the start guard (kernel age > 5.75 h);
- a driver error;
- a SIGKILL at the hard deadline, min(start + 3.3 h, 8.2 h);
- a B_probs sha / label / id mismatch;
- probabilities outside [0, 1];
- coverage below .98;
- the rho floor (n >= 100: mean < .60 or min < .40);
- a non-finite blend.

Why these are not a risk to the final score:
- **c4's own hidden run passed every check.** Its .958 shows no -.03 ACL drop. A rerun sees the same hidden data and code, so every deterministic check passes again.
- **Only stochastic triggers remain:** timing, a random worker crash or OOM, or Kaggle infrastructure.
  - On hidden, S6F starts at about 3.7-4.4 h (think/runtime.md), which leaves 1.3-2 h of margin.
  - About 20 S6F-class hidden runs have read .955-.958 with no canary-sized drop (CLAIMED, HANDOFF / PROGRESS reads).
  - P(fire) per new run is ~.01-.02.
- **Harm is a lost read only.** The entry would show about .92-.93, so that slot's read is lost. It cannot touch the final private score: the picks are already-scored runs, and a fired canary always shows on the public LB of the same run.
- **The E-OA lines in eoa20 / eoa10 add no canary path.** They sit inside the same try block (a20/build_a20.py), after the blank fill, and are only pct and arithmetic on finite float columns, so they cannot raise.
- **No guard proposed.** A silent S6F fallback would only make a failed read look like a real one.

## 4. Side finding (not a guard; outside this lens)
Feeding R6 definition-style fluid flags (PD/T2 non-FS -> fluid 1) changes R6's Effusion through two routes: the fluid stacker
(German no-FS-sagittal studies leave the miss path) and the meta input.

| gold58 result | value |
|---|---|
| R6 Effusion | +.027 |
| c4 Effusion | +.009 |
| c4 macro | +.0007 +- .0009 (P < 0: .22) |

Caveats:
- Three studies drive it, and one German study (expert 0) alone gives +.0174.
- It changes the visible German study, so it is a lever and not a no-op guard.
- It overlaps the closed miss-path / site-offset topic. missing_planes.md measured the imputation variant at <= +0.5e-4 macro through a c4-shaped blend on report OOF.
- There is no independent ruler, and R6 is in-sample on train. WEAK. Do not pursue without an OOF ruler.

## 5. Files
- `more/c4cells/cellNN.*`: the c4 cells as text.
- `more/work/g1_tags.py`, `g1b_tags.py` (+ `.log`, `g1b_classified.csv`): the train flag rule from DICOM headers.
- `more/work/vis_hdr.py` (+ `.log`): the visible-test headers.
- `more/work/g2_exposure.py` (+ `.log`, `g2_def_flags.csv`): the DEF flags and member exposure.
- `more/work/g3_setup.py`, `g3_r6run.py`, `chain.sh`: the box runs (`box/chain.log`).
- `more/work/g4_fill.py` (+ `.log`, `.csv`): the fill simulation.
- `more/work/g5_harm.py`, `g5b_members.py`, `g5c_c4labels.py` (+ `.log`): the gold58 harm analysis.
- `more/work/g6_gate.py` (+ `.log`, `g6_gate_lib.py`): the G-FLAG function and offline gate.
- `more/work/g7_prefer.py` (+ `.log`): the narrower guard.
- `more/work/box/`: Raptor BASE / DEF raw outputs, R6 DEF predictions and the fluid-stacker dump.
- On the box: `~/guards1020/` holds the run dirs, the DEF shadow data dir, the logs and a private inductor-cache copy. Nothing else was touched.
