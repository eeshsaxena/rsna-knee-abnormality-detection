# TRICK_1019: "my rank is slipping, we are still missing a trick" (10-19, written ~22:00-22:45 IST). RESEARCH ONLY.

Nothing was built, pushed, committed or submitted. No dataset was created. No GPU, training or box job ran. Box access was
read-only (ls, wc, head of headers, our own label-build script and its summary text; no OAI row was printed or copied).
No PREREG, HANDOFF, PROGRESS or scheduled-task edits. Nothing public was pulled (`pulled/` is empty).
Inputs: the 7 lens files in this dir (missing_planes, laterality, label_regime, unseen_sites, leaders, kernel_audit,
transductive), the 2 skeptic votes on label_regime, THINK_1019.md, PROGRESS_19-10-2026.md, the c4 notebook (cells 14, 18).
New work this pass (all CPU, local, < 1 min each): `oai_check/oai_eoa_check.py` (the box check in sec 4, dry-run on gold58
and on synthetic data only), `oai_check/fam_g58.py` + `.log` (E-OA on every gold58 prediction set we hold), an LB growth fit
on `work/lead_ts.parquet`, and a read of `_data/test.csv`.
Tags: MEASURED (computed or read now, file named), CLAIMED (a note or third party says so), MODEL (arithmetic on stated
assumptions), GUESS (judgement). Paths are relative to `session_2026-10-19/trick/` unless they start with `session_` or `_data`.

## 1. What trick are we missing? (plain answer)

1. No cheap trick is left to find and ship in time: 7 lenses + 2 skeptic passes found no c4 bug and no solid lever >= +2e-4.
2. The teams passing us share no public trick. 83 % of teams with their own model above the public frontier before the R6 wave
   are now .959+, vs 2.8 % of the public-copy crowd: their edge is own labels + own k-fold CNNs (MEASURED leaders.md).
3. That edge cannot be built and read before 10-22 (~5 h V100 + dataset + ~13 h commit/score; our trained members are 1 for 7).
4. The one idea with a mechanism is OA decorrelation (E-OA): reports tie the 3 OA compartments together, experts score each alone.
   gold58 says +.0014 to +.0024, but both skeptics show that is ~20 pair swaps sharing noise with today's 5 failed gold58 tweaks.
   Honest value now: 0 to +3e-4.
5. It can be settled tonight without a slot: 2 min of box CPU on 2,399 expert-read OAI knees (MOAKS) with our OAI-free
   report-trained model (sec 4, G1). SUPPORT: spend one 10-20 slot on E-OA. Otherwise it is dead.
6. The slip is public and structural (~24 teams/day cross .959): projected public rank ~270-300 at the deadline vs a silver line
   near 318 (MODEL, sec 2.3). Most of those teams will likely beat us on private too.
7. Our biggest lever is unchanged: select A = API best-id and B = R6 57248084 by hand (~+9 pts P(silver), MODEL picks.md).

## 2. Every lead, ranked

### 2.1 Ranking
Index = E[private gain] x feasibility. E already folds in P(real). Conversion (MODEL, picks.md l.37): +1e-4 of real private
gain is about +4 pts P(silver) for the A/B pair.

| # | lead (lens) | E[private gain] | P(real > 0) | feasibility by a 10-21 read | index | skeptic votes | verdict |
|---|---|---|---|---|---|---|---|
| 0 | Final picks by hand: A = API best-id, B = R6 (THINK, picks.md) | ~+9 pts P(silver), about +2e-4 equivalent (MODEL) | R6-B beats c4-alone in 22 of 22 matrix rows (MEASURED picks_work/matrix.txt) | 1.0 | highest | n/a | KEEP. Not new, but the largest lever |
| 1 | E-OA, OA decorrelation on the final c4 output, GATED by the OAI check G1 (label_regime) | if G1 = SUPPORT: ~+2 to +4e-4 (MODEL/GUESS); before G1: ~+1e-4 (pooled lead/skeptics, MODEL) | after SUPPORT ~.7; now ~.5-.6 | ~.75 (Eesh's go tonight, 1-1.5 h build, overnight commit) | ~1.5-3e-4 after SUPPORT; ~.75e-4 now; 0 after KILL | 2 of 2 REFUTED as a solid positive (corrected ~0 and 0 to +3e-4) | CONDITIONAL. Run G1 first |
| 2 | E1 soft-OR pow2, kernel-natural (THINK 8.4) | +0.5 to +1e-4 (MODEL) | .57-.60 | ~.9 | ~.5-.9e-4 | THINK critique: DOWNGRADE, still the only probe worth a slot | Lottery ticket. Not a gold58 screen |
| 3 | Missing-plane CPU probe, then Raptor guard T1 (missing_planes) | ~+1e-4 overall (+.0003 to +.005 only if 0.5-5 % of hidden studies are incomplete) | 5-10 % that the probe fires (GUESS) | ~.5 (probe < 1 h; the guard rebuild reads 10-21 night at best) | ~.5e-4 | none | Optional info probe. Low |
| 4 | Rubric-mapped R6 state-head score (label_regime sec 3) | ~+0.5e-4 (proxy +6.3e-4, sd .0022, P .63, split-half .40) | ~.5 | ~.3 (box GPU dump plus build) | ~.15e-4 | none | WEAK. Skip |
| 5 | S6fall side classifier flags (laterality) | +3 to +6e-5 | .55-.6 | ~.5 | ~.1e-4 | none | WEAK. Skip |
| 6 | Mycelis 4-fold leg after S6F (leaders) | ~0 | .1-.2 | ~.3 (download OK, port, T4 timing) | ~0 | none | DEAD for this window |
| 7 | Unseen-site routing, intensity norm, AdaBN (unseen_sites, transductive) | 0 (sign unknown) | ~.5 | low | 0 | none | DEAD |
| 8 | Train-test duplicate match, kNN smoothing (transductive) | +1.6e-5 / negative | .4 / <.2 | low | ~0 | none | DEAD |
| 9 | Raptor blank-study flag fix, fallback-row CDF map (kernel_audit) | ~0 (dormant) | n/a | needs a rebuild | ~0 | none | DEAD unless rebuilt for another reason |
| 10 | R6 Baker's from raw z instead of the report-fitted state stacker (kernel_audit) | -1.5e-4 on the gold58 blend (P>0 .23) | ~.45 | high | negative | none | DEAD |
| 11 | Train an independent strong model, the leaders' edge (leaders) | large for them (+.001 to +.005) but ours read .877-.924 on gold58 vs .923-.937 for c4 members | low for our recipes | ~0: ~4-6 h V100 + dataset + ~13 h latency gives a blind 10-22 entry | ~0 | none | DEAD for this competition |

### 2.2 Evidence per lead (short; full detail in each lens file)

**E-OA (label_regime.md; skeptics in the brief and `skeptic_regime/`).**
- For (MEASURED): gold58 c4 +.00142 (a .1) / +.00239 (a .2) macro; bootstrap P .94 / .97; report-regime OOF loses 9 of 9 cells,
  as the mechanism predicts; beats 36 of 36 non-OA random-pair nulls; fully nested +.00031 (sd .00187, P>0 .63).
- Against (MEASURED, skeptic 1): over all 220 label triples the cross-triple spread equals bootstrap noise, so empirical-Bayes
  shrink gives +.00008 (a .1) / -.00054 (a .2); OA is the best triple in only 0.7 % / 12 % of resamples; nested pick-best-of-220
  -.00007 / -.00066; the gain is ~20 pair swaps (Medial 0, Lateral +5.5, PF +14) and greedy removal of 5-6 studies zeroes it.
  The "4 report readers" are one label set on OA (sol56 = v4blend = JEV on all 174 gold58 OA cells).
- Against (MEASURED, skeptic 2): the host never confirmed the facet mechanism (a participant raised it; the host replied in general
  terms, think/forum_raw2/flat/flat_733491.txt). The facet direction (Lat<-PF) adds only +.00008 at a .2; most of the gain is in
  the PF OA column, the column pf10 moved and lost on the LB today. E-OA's gold58 noise correlates .37-.64 with pf10, lat10 and
  c3max; dropping the 8 studies that drive pf10 + lat10 removes ~60 % of E-OA's gain. Conditioning on the A19 LB reads gives an
  in-sample +.0001 to +.0008 before any selection charge.
- New this pass (MEASURED, `oai_check/fam_g58.log`): E-OA on all 12 gold58 prediction sets we hold. OA-mean AUC delta at a .1 / .2:
  c4 +.0057 / +.0096 (largest), R6 +.0035 / +.0067, S6fall +.0041 / +.0056, r384 +.0004 / +.0006, r224 +.0033 / +.0036,
  R7 OOF +.0031 / +.0092, TTR s1044 frac75 -.0003 / +.0021, TTR s1047 +.0005 / +.0044, KC s2001 +.0021 / +.0009,
  KC s2011 +.0028 / +.0058, KC dread +.0015 / -.0004, OAI-trained s1043 +.0040 / +.0024. So 11 of 12 are positive at each a, and
  Medial OA gains ~0 everywhere. All 12 are scored on the same 58 labels, so a consistent sign fits BOTH stories (a real
  report-coupling effect, or a handful of gold58 studies). Only an independent expert set can separate them: that is G1.
- New this pass (MEASURED on synthetic data, `oai_check/oai_eoa_check.py`): low label coupling (phi ~.14) plus high model coupling
  (rho ~.72) is NOT enough for decorrelation to help. If the models' shared OA component is real shared disease, decorrelation
  loses (-.0157 OA-mean at a .2 in the synthetic run). So the "phi .28 vs rho .81" gap on gold58 is not by itself evidence;
  the AUC delta on an independent expert set is.

**E1 (THINK_1019 sec 2.1, 8.1-8.4).** OOF (report labels, R7 + our S6 seeds) +.00036 to +.00112, P .98-1.0 (MEASURED
skeptic/oof_norr.log); gold58 kernel-natural +.00009 (P .54), -.00014 with Fracture held (MEASURED skeptic/e1_stages.log);
the same OOF ruler rated our own members positive before they lost on the LB (CLAIMED ours.md). It is the only candidate whose
ruler is not gold58, the ruler that went 0 for 5 on 10-19 (MEASURED PROGRESS_19: pf10, fr100 .958 < c4; c2max, c2, c3max .957).

**Missing planes (missing_planes.md).** Train 0 of 4,407 studies lacks a plane or an FS series; 3 of 3 visible test studies are
complete (MEASURED). R6's Effusion missing-volume path is worth at most +5e-5 macro (MEASURED). A per-study fallback recovers
70-95 % of a member's loss only if hidden studies are incomplete (MEASURED sim).

**Laterality (laterality.md).** Raptor is anatomy-based and immune; R6 is guarded by tag, then sagittal x, then its side
classifier, and R7 OOF shows no swap excess in any language or vendor (MEASURED). Only S6fall (weight .2, no side classifier)
is wrong on ~1-2 % of studies (MODEL), worth +3 to +6e-5.

**Unseen sites (unseen_sites.md).** All three families already normalise each series by its own percentiles; aligning member
score levels per site is 0 (consensus) or -.0002 to -.0031 (MEASURED). No family is more robust on atypical train studies or on
gold58 split by language (MEASURED). Raptor has no OOF.

**Leaders (leaders.md).** No jump spike after any public release; .959+ grows ~24/day (MEASURED fit below). Authors of every
recent public model are still at .958 or below. The predictor of .959+ is an independent model above the frontier before 10-12.

**Kernel audit (kernel_audit.md).** Alignment, label order, R6 flip TTA, float32 ties (~1e-5), port fidelity (S6fall vs author
1.8e-3), decoders and 68 logs are clean (MEASURED). Two dormant defects (Raptor blank-study path not flagged; fallback rows at
the extremes) matter only if a hidden study fails.

**Transductive (transductive.md).** 1 duplicate pair in 4,407 train studies (one exam filed twice); 0 for gold58; headers carry no
date or patient link; kNN smoothing loses on both rulers; AdaBN has no clean ruler (MEASURED).

### 2.3 Rank-slip arithmetic (MEASURED fit on `work/lead_ts.parquet`, 10-16 to 10-19 snapshots; projection MODEL)
- Teams grow 44.7 per day; .959+ teams grow 24.0 per day (185 at 10-19 15:45 UTC).
- At 10-22 23:59 UTC (+3.35 days): ~6,367 teams, so the silver line (top 5 %, CLAIMED Kaggle progression rule) is ~rank 318 and
  bronze ~636; .959+ ~265 (252-279 at 20-28 per day) plus ~17 teams inside the .958 band above c4 today: public rank ~270-300.
- This matches picks.md (silver cut ~r316 at N ~6,336; "c4 most likely stays just inside the silver line" on public). The
  private outcome is the real question, and picks.md puts P(silver) at ~.63 for the A/B pair (MODEL, +-.08).

### 2.4 Brief corrections found on the way
- "Experts call fluid positive more than reports" is false for Effusion (expert .60 vs report .86-.88 on the same 58 studies) and
  true for Synovitis (.47 vs .28-.29) (MEASURED label_regime m1).
- The host did not confirm the patellar-facet "lateral compartment" mechanism (MEASURED read of flat_733491.txt, skeptic 2).
- test.csv holds only StudyInstanceUID: no report text reaches the hidden run (MEASURED `_data/test.csv`), so no report-LLM
  trick is possible at test time.
- train.csv labels exist only for the 58 gold studies (MEASURED label_regime sec 1): every member learned third-party LLM labels.

## 3. Build specs for the surviving tricks (only on Eesh's go + a PREREG amendment; nothing here was built)

### 3.1 T1 = E-OA, only if G1 (sec 4) prints SUPPORT
- Base: the c4 kernel `session_2026-10-18/a18/kernel_v224r6a25c4j/rsna_sub_v224r6a25c4j.ipynb`, built from the build_a18 lineage
  the same way build_a19.py patched cell 14 (assert the base is c4, patch one block, write a receipt with src/out sha256).
- One change, cell 18 `_s6f_blend`, S6F path only. Insert between l.86 (`out[c] = np.float32(0.5)` for blanked columns) and l.87
  (`v = out[labels].values...` validity check):
  ```
  _OA = ['Medial OA', 'Lateral OA', 'PF OA']
  _U = {c: pct(out[c].values.astype(np.float64)) for c in _OA}
  for c in _OA:
      _o = [k for k in _OA if k != c]
      out[c] = pct(_U[c] - A_OA * (_U[_o[0]] + _U[_o[1]]) / 2.0).astype(np.float32)
  rec['a_oa'] = A_OA
  ```
  with `A_OA` = the SHIP_A value G1 prints (.1 or .2), defined once at the top of cell 18. `pct` is the average-tie helper
  defined inside `_s6f_blend` (cell 18 l.17), the same tie rule as the gold58 replay.
  The canary path (l.96 onward) is untouched, so a failed S6F run still shows the ACL canary. The other 9 columns are untouched.
- Offline gates (CPU, all must PASS before the commit is submitted):
  - G-a: the same builder with A_OA = 0 gives a notebook whose committed submission.csv is byte-identical to c4's commit output.
  - G-b: gold58 replay (`regime_work/m14_gate.py`, or `oai_check/oai_eoa_check.py --labels-binary` on the c4 gold58 output)
    gives macro .94244 at a .1 / .94341 at a .2 (c4 .94102), with the 9 non-OA columns rank-identical (MEASURED m14_gate.log,
    reproduced this pass).
  - G-c: the commit's 9 non-OA columns equal c4's commit output exactly; receipt shows `a_oa`, S6F_BLEND_OK, 0 fallbacks.
- PREREG-style one-change test (to copy into the A20 amendment):
  - Entry: c4 57343435 + E-OA at a = SHIP_A. Nothing else. Tag e.g. v224r6c4eoa10j / v224r6c4eoa20j.
  - Hypothesis: hidden labels are expert-regime; decoupling the OA trio raises Lateral OA and PF OA AUC.
  - Evidence used to pick it: G1 on OAI (independent of gold58 and the LB). gold58 is only a code gate.
  - Read: R1 (API best-id) vs c4. Move: A = E-OA, B stays R6 57248084. No move: E-OA closed; no retry at the other a; never stacked
    with E1 or anything else.
  - Read noise (MEASURED m12): public paired sd .00021 (a .1) / .00042 (a .2). P(best-id move) ~.75 if the true effect is +1.5e-4
    at a .1 or +3e-4 at a .2, ~.5 at 0 (MODEL). A negative read cannot separate gold58 noise from report-like test labels
    (skeptic 2), so it only closes the lead.
- Cost: 0 hidden runtime (rank arithmetic on 3 columns). Build + gates ~1-1.5 h; commit ~6.5 h; score ~6.5 h.

### 3.2 T2 = E1 soft-OR pow2 (unchanged from THINK_1019 sec 8.4)
- Cell 14: square the per-arm percentiles before they mix (`pct_on` of r384 raw and of r224 at l.71-73, and `ro`), no extra re-rank
  of the Raptor mix. Cell 18: square `rb` and `rs` after the failed-row fill (l.77). Cells 4 and 11 untouched. All 12 labels.
- Gates: exponent 1 byte-identical to c4; exponent 2 on the gold58 serves = .94111 (MEASURED skeptic/e1_stages.log "norr").
- Read R1 vs c4. A move: A = E1, B stays R6. No move: closed (no invgeo or pow3 retry).
- Use it only as a lottery ticket: its ruler is report-regime OOF, which has over-rated our members before.

### 3.3 T3 = missing-plane CPU probe (optional, information only)
- CPU kernel with competition data attached, no model: read test.csv + test_series.csv and list each series dir. All studies
  have Sag + Cor + Ax, >= 1 FS series, standard plane values and non-empty dirs: write all-0.5 (LB .500). Some study lacks a plane:
  raise. Planes complete but some study lacks an FS series or has odd values: write a csv missing one column (scoring error).
  Spec and the follow-up guard in missing_planes.md sec 5-6. 1 slot, < 1 h. Only if Eesh wants the bit; the guard it could unlock
  is worth ~1e-4 in expectation.

### 3.4 Day plan (IST)
| when | action | needs |
|---|---|---|
| 10-19 ~22:45-23:15 | Ask Eesh: (a) STOP, or (b) run G1 (2 min box CPU, aggregates only) and build E-OA if it says SUPPORT, and/or (c) E1 per THINK | Eesh's explicit go for the box job and any build |
| 10-19 ~23:15-00:45 | If (b) and SUPPORT: builder from build_a18 lineage, gates G-a/G-b, PREREG A20 amendment, push the commit with `--timeout` on T4. If (c): same for E1. Two commits may run in parallel (<= 2 GPU sessions per account, CLAIMED) | gates PASS |
| 10-20 ~06:30-07:30 | Commits land (~6.5 h). Submit after 05:30 IST (new UTC day) via a live session or an approved tick + GO file | Eesh's go; nobody submits otherwise |
| 10-20 ~13:00-14:30 | Read by R1. If two entries ran, adopt at most one (the higher public delta), never combine them | |
| 10-20 evening to 10-21 | Nothing new by default. B stays R6 whatever happens | |
| 10-22 before 18:00 | A15 last read: A = API best-id | |
| 10-22 before 23:59 UTC (10-23 05:29 IST) | Eesh selects A and B = R6 by hand and screenshots the checkmarks | |

If G1 is not run or does not print SUPPORT: E-OA is not built. The default is then THINK's: STOP, or E1 alone as one ticket.
Running E-OA and E1 side by side as two separate one-change entries is weakly better than one under R1 if both have a
non-negative mean (MODEL: adopting a -1e-4 loser by noise costs ~-3e-5 in expectation at public sd .00021), but THINK's rule
says one probe; that is Eesh's call.

## 4. Measurements worth running on the idle box tonight (specified, NOT run; each needs Eesh's OK as a box job)

Box state at 21:58 IST (MEASURED, read-only `nvidia-smi`): V100 idle, 5 MiB used.

**G1 (the only one that matters): OAI expert-regime check of E-OA. Box CPU, ~1-2 min, no GPU, no file written on the box.**
- Why: the only expert ruler we have is gold58 (58 studies). The box holds MOAKS image reads by musculoskeletal radiologists for
  2,399 OAI V00 knees (MEASURED `~/oai_build/oai_labels_report.txt`: "knees with any m_ field 2399"), and predictions on those
  knees from an OAI-FREE, report-trained model: `~/oai_build/teacher/teacher_D5_mean_v00.csv` (2,581 rows) plus its 5 fold files
  `f{0..4}_s1043_ttr_frac75__raw__A__shard{0..3}of4.csv`, made by `oai_teacher_infer.py` from "RSNA-only fleet fold checkpoints"
  (MEASURED header comment). Raptor is OAI-trained (models.md) and R6 / S6fall cannot read the npz store, so they are not used.
- Labels: built inside the script from `~/oai_build/oai_label_sources.csv` (m_cm_* = MOAKS "area.fullthickness" per cartilage
  subregion), compartments = build_oai_labels.py COMP cartilage regions, with NO osteophytes and NO X-ray JSN (they add coupling the
  rubric does not have). Proxies of the rubric ("moderate or large area, ~1 cm+, of > 50 % thickness loss", flat_733343.txt):
  P_FT (primary) = any subregion with full-thickness grade >= 1; P_AREA = any subregion with area grade >= 2; P_BOTH = both in one
  subregion. A knee enters only if every compartment has >= 1 read subregion.
- Command (Git Bash on the laptop; the script goes over stdin, so nothing is written on the box; only aggregates print):
  ```
  ssh training@192.168.10.30 '~/miniconda3/envs/rsna/bin/python -I - --sources ~/oai_build/oai_label_sources.csv --preds ~/oai_build/teacher/teacher_D5_mean_v00.csv --fold-glob "~/oai_build/teacher/f{k}_s1043_ttr_frac75__raw__A__shard*of4.csv"' < C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/trick/oai_check/oai_eoa_check.py
  ```
  Runtime: 50 s locally on a synthetic 2,400-knee set with 2 folds (MEASURED); ~1-2 min on the box with 5 folds (MODEL).
- Pre-registered reading (in the script header, written before any OAI number was seen): per a in {.1, .2}, SUPPORT = primary
  OA-mean AUC delta > 0 with paired-bootstrap P>0 >= .90, > 0 on >= 2 of 3 proxies, and the a = -.1 control (positive borrowing)
  < 0; KILL = primary delta <= 0 or P>0 < .5; else INCONCLUSIVE. SHIP_A = .2 only if SUPPORT at .2 and it beats .1; else .1 if
  SUPPORT at .1; else none. Fold lines are a replication read, not part of the rule.
- Checked this pass (MEASURED): on the c4 gold58 output with gold58 labels the script reproduces m14 exactly (base OA AUCs
  .9891 / .8433 / .9009; a .1 +.00142, a .2 +.00239 macro-equivalent) and prints SUPPORT, SHIP_A .2; on random synthetic labels it
  prints KILL. Power (MODEL from the synthetic run): paired sd of the OA-mean delta ~.0005 (a .1) / ~.001 (a .2) at n ~2,400, so
  an effect the size of the teacher recipe's gold58 effect (+.0021 at a .2, `oai_check/fam_g58.log`) is detectable.
- Limits, stated before the run: OAI is an OA cohort on one 3T protocol, and MOAKS is not the competition rubric. A KILL can mean
  "real shared disease in an OA cohort" (see the synthetic result), so KILL is weaker evidence against E-OA than SUPPORT is for it.
  But without SUPPORT, E-OA has no evidence outside gold58 and does not earn a slot.
- Governance: OAI rows never leave the box; only AUCs, deltas, P values, phi, rho, n and prevalences come back.

**G2 (optional, low): R6 state read-out on gold58.** Box V100 ~15-20 min: re-run `~/r6g58/r6g58_run.py` exactly as run_b with
`--dump-logits <out>/dump.json --dump-state` added to the worker args (compact_infer 8401812c supports it, l.595). Feeds only the
rubric-state transform (rank 4), whose ruler would again be gold58. Not recommended tonight.

**G3 (not the box): the missing-plane CPU probe (sec 3.3)** if Eesh wants that bit. 1 Kaggle slot, < 1 h.

Nothing else needs the V100 before the deadline. No GPU run can settle E1 (its open question is the expert regime with Raptor in
the mix, and only gold58 has that). The Raptor ablation, the S6F side-flag run, the R6 T4 OOM test and the acquisition-shift
stability run in the lens files are all specified there and none is worth tonight's slot.

## 5. Dead leads (closed; do not reopen without new evidence)
| lead | why dead | source |
|---|---|---|
| A shared public trick behind the .959 wave | no jump spike after any release; every recent public-model author still <= .958; jumpers' common factor is an own model | leaders.md sec 1-4 (MEASURED) |
| Training an own member or R6 fine-tune now | ~4-6 h V100 + dataset + ~13 h latency = blind or 10-21-night read; our recipes .877-.924 on gold58 vs .923-.937; B-leg R6 fine-tunes (public) no better than R6 folds and their authors stayed .958 | label_regime sec 4, leaders sec 4, THINK sec 5 |
| Report-LLM labels at test time | test.csv has only StudyInstanceUID | `_data/test.csv` (MEASURED) |
| Silent bug or leak in c4 | every audit clean or ~1e-5 | kernel_audit.md |
| R6 Baker's raw z (drop the state stacker) | -1.5e-4 on the gold58 blend, P>0 .23 | kernel_audit sec 5 |
| Laterality fixes | Raptor immune, R6 guarded, S6fall fix +3 to +6e-5 | laterality.md |
| Missing-plane guard without a probe | train and visible test complete; identity on complete studies | missing_planes.md |
| Site / scanner routing, intensity normalisation, member re-levelling per site | 0 or -.0002 to -.0031; no robustness signal | unseen_sites.md |
| Duplicates, kNN smoothing, kNN label transfer, AdaBN, pseudo-label self-training | 1 dup pair; kNN negative on both rulers; no ruler for AdaBN | transductive.md |
| E2 Syn<-Eff and the 13-pair / 132-pair cross-label families | nested -.0005 / -.0001 to -.0010 | label_regime m5, m7; THINK 8.2 |
| Per-label Raptor-share tweaks picked on gold58 (pf10, fr100, c2, c2max, c3max; lat10 screened, not submitted) | 5 of 5 read negative on 10-19 although their gold58 screens had P .905-.997 | PROGRESS_19 (MEASURED LB); research/calibration_r6_v1.log (CLAIMED via skeptic 2) |
| E3 shifted-crop TTA, multi-FOV / multi-scale TTA, invgeo, Raptor-mix re-rank, Dread / MaViT / finespacing legs, KC variants | see THINK 8.2 | THINK_1019 |
| Mycelis OOF leg | author .958 r1331 after 15 subs; prior ~.1 it passes a CPU gate | leaders.md sec 7 |
| E-OA on gold58 evidence alone | in-sample selection over ~220 triples; shares noise with the failed A19 reads | skeptic votes; sec 2.2 |

## 6. Files written this pass
- `oai_check/oai_eoa_check.py`: the G1 script (aggregates only; dry-run checked on gold58 parity and on synthetic data).
- `oai_check/fam_g58.py`, `oai_check/fam_g58.log`: E-OA on 12 gold58 prediction sets (run from the scratchpad, where its two
  input CSVs, the c4 gold58 final output and the gold58 labels, were written by a copy of the m14 rebuild).
- Scratch inputs (c4 gold58 output CSV, synthetic test files) stay in the session scratchpad.

## 7. Completeness critic (10-19 ~23:00 IST; CPU + read-only ssh ls / log head; nothing built or submitted). Ranking unchanged.
**A. Trick categories no lens examined**
1. Host-flagged series-flag shift (NEW, unexamined). The data page (`think/work_forum/pages/ListPages.json`, data-description)
   says Fluid_Sensitive and Fat_Suppression "are not necessarily equivalent for every case"; no lens cites it (grep). Train is
   24,371 / 24,371 equal and the 3 visible studies are equal (MEASURED missing_planes, unseen_sites). Every member keys on
   Fluid_Sensitive: Raptor's slot picker (`c4_cells.py`, `pref = ... Fluid_Sensitive == fluid`), R6 / S6fall model meta col
   "fluid" (compact_infer 8401812c l.517 meta6 uses px_cache col 7 = fluid, not col 8 fatsat) and R6's fluid stacker
   (l.123 `fluid != 1`) (MEASURED code read). So a hidden T2/PD non-FS series (1,0) is read as fat-sat by ALL members, no member
   is a clean fallback, and nothing guards it. Value unknown: P(mismatch rows exist) ~.3-.5, harm unmeasured (GUESS).
   Action: if T3 (sec 3.3) is run, add one bit at zero cost (any hidden row with Fluid_Sensitive != Fat_Suppression ->
   distinct failure mode). GPU spec only if it fires: box V100, gold58 + ~100 train studies with a non-FS sag/cor series; run
   c4 cell 4 (r384) and cells 7-9 (R6) on the stock series CSV vs a copy with Fluid_Sensitive=1 on every non-FS series;
   report per-member Spearman and gold58 macro delta; ~45-60 min. Guard only if a member drops > .005 or rho < .98.
2. Patient covariates (CLOSED, MEASURED `critic_work/{hdr,sex}_check.py` on `box/dicom_series_tags.csv`): PatientSex is in the
   86-tag allowlist (M 209 / F 198 of 458 header studies, all 58 gold). Logistic label ~ R7 OOF z + sex: sum z^2 9.2 on 12 df
   (report v2, n 359) and 10.9 on 12 df (gold58, n 48), i.e. null. PatientAge absent (0 / 458). Contrast (+C) series in 1 of
   458 studies, so no Synovitis contrast cue.
3. Both knees in the FOV (thread 735639): ~7 train studies (CLAIMED); Raptor's fixed 140 mm crop misses the knee, R6 / S6fall
   use anchors. ~2 hidden studies, <= ~1e-5 (MODEL). Negligible.
4. E-OA on B: if G1 = SUPPORT the regime argument covers R6 too (gold58 +.0017 at a .2, m10), yet sec 3.1 keeps B = raw R6;
   {c4+E-OA, R6} vs {c4+E-OA, R6+E-OA} is not modelled in picks.md. A defensible choice, not a result.
**B. Claims that are unverified or over-tagged**
- Sec 1 item 2 tags "their edge is own labels + own k-fold CNNs" MEASURED: the 83 % / 2.8 % counts are; the attribution is
  MODEL / GUESS (leaders.md sec 0 item 4).
- G1 validity: the teacher is f0-4 `s1043_ttr_frac75` ep15 (MEASURED box `run_shard0.log`, fold cv .88-.89, 2,581 V00 ids;
  OAI-free CLAIMED by its 10-10 header; the dir holds no oaiT123 file, so the glob is unambiguous). It is the Raptor-like slot
  recipe whose gold58 siblings show the SMALLEST E-OA effect (TTR s1044 -.0003 / +.0021, r384 +.0004 / +.0006 vs c4 +.0057 /
  +.0096; fam_g58.log). SUPPORT would be strong; KILL says little about c4's R6 / S6fall-driven effect. Effective n after the
  every-compartment filter and the id join is unknown, so the power numbers (sd .0005 / .001 at n ~2,400) are MODEL.
- label_regime.md sec 0 item 1 still says the host confirmed the facet mechanism (stale; sec 2.4 corrects it).
  "<= 2 GPU sessions per account" (sec 3.4) stays CLAIMED.
**C. Sources not read**
- 12 of 45 saved forum threads are cited nowhere in think/ or trick/ (733313, 735154, 735639, 735767, 735826, 743374, 743916,
  745759, 746124, 747613, 748111, 748310). Skimmed now: only 735639 (above) and 745759 (filename slice order; c4 sorts by IPP,
  MEASURED missing_planes sec 2) touch inference; the rest are backbone size, speed and chatter.
- File lists only, never contents: BliverRigPaDoge trainkit (a .959 team: judge_image.py, consensus / teacher CSVs),
  haungjiawei dup_groups.csv (could cross-check transductive's 1-pair count), Mycelis OOF. Each needs Eesh's download OK; by
  name none is an inference add-on (GUESS).
- Threads 744519 (all planes?) and 734681 (site split) were not re-polled after ~16:00 UTC 10-19. No public notebook code was
  re-read this pass (`pulled/` empty), so "no public trick" rests on LB dynamics plus earlier sessions' reads (CLAIMED).
