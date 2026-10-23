# critic.md: completeness critic over every round report (10-21 ~20:30 IST, Claude, FINAL critic track)

RESEARCH ONLY. Nothing was submitted, pushed, committed, versioned, downloaded, posted, scheduled or turned into a dataset. PREREG,
HANDOFF, PROGRESS and the live folders were only read. No box contact, no GPU, no OAI data. Files written: this note,
`hdr_cues.py` -> `hdr_cues.log` (CPU, local RSNA competition files only) and `subs_now.json` (one read-only Kaggle submissions list).
**Tags:** MEASURED = computed or read now, file named; CLAIMED = a round report says so (named); MODEL = arithmetic on stated
inputs; GUESS = judgement. Units: e-4 = 1e-4 macro AUC (private unless stated).
**Inputs read:** THINK_1019, TRICK_1019 (incl. sec 7 critic), EX_1019, MORE_1020, ROUND5, PUSH_1020, LEVERS_1020, VAL_1020,
NIGHT_1020, PRIVATE_1020, stack/build.md (incl. sec 8 skeptic), FORUM_10-20, DAY_1021, LEVERS_1021, CANARY_1021 (+ cause.md),
LASTDAY_1021, HANDOFF top blocks + 10-17 closed list + early key findings; spot reads of transductive.md, unseen_sites.md,
push/e1/info_hidden_ties.log, day/plan1022.md.

## 0. Verdict (plain)

1. **No gap clears both bars** (plausibly > 1e-4 private AND shippable by 10-22 09:00 IST). The plan does not change: r6pw20 v1
   resubmit copies at 05:31 / ~06:15 (optional ~07:00) IST, A = swide 57464326, B = earliest clean r6pw20 copy >= .958 by id,
   else R6 57248084, selected by hand.
2. 20 lever categories were examined across the rounds. 18 are closed with an independent ruler or an LB read. Two had
   superficial coverage: DICOM metadata (only sex / age / contrast / language) and final-output ties (a GUESS of "first-order 0").
3. **Best gap measured: DICOM protocol / sequence cues** (series descriptions, 3D / GRE / STIR, series count, field strength,
   repeated series). They carry real report-regime signal within site (MEASURED, perm P .000). But as a shippable nudge they are
   about 0 on report labels and negative on gold58 experts at every dose (-.0022 to -.0149 macro). That is the same signature as the
   site priors that already lost (CLAIMED gold58 -.0152). **DEAD.**
4. **Ties are bounded below the bar:** perfect tie-breaking is worth at most 8.0-8.3e-5 per label at n 910 on the c4 path
   (MEASURED info_hidden_ties.log). swide re-ranks 5 of 12 columns, so its ceiling is about 4.7e-5 macro and a realistic gain is 1-2e-5
   (MODEL). **DEAD.**
5. **"Notebook threw exception":** read-only API now (MEASURED `subs_now.json`): all 108 submissions are COMPLETE. None has carried an
   error_description since 10-03 (those 7 were runtime overruns). The three .916 entries are our own cell-18 handler: it catches the
   S6F failure and writes the ACL canary (CLAIMED cause.md sec 1). Their outputs are ~8.6-8.7 KB smaller than the clean ones
   (238,686-238,746 vs 247,369-247,389 bytes, MEASURED). That fits ACL = 0.5 on all ~1.2-1.3k hidden rows (MODEL), so private is
   blanked too and those three can never be picks. Nothing new errored. 0 slots remain in the 10-21 UTC day.

## 1. Every lever category examined, with verdict

| # | category | what was tried (examples) | where | verdict | depth |
|---|---|---|---|---|---|
| 1 | Model members | add / swap / drop: R7 leg, KC1 / KC-all, s43 / s44 seeds, Dread maxspan / widedense / finespacing, Mycelis, MaViT, CoAtNet-2, DINOv2, goodpjw, RadImageNet, OAI-trained heads | THINK 2.4 / 5 / 8.2, TRICK 2.1 #6 #11, MORE 20-23, HANDOFF 10-17 sec 3 | CLOSED. Own members are 1 of 7 on the LB, and every candidate scores below the c4 members on gold58 (CLAIMED) | deep |
| 2 | Member / per-label weights | W_RAPTOR, A224, S6fall .15 / .2 / .3, heliosli table, c4 table, fr / pf / c2 probes, Raptor OA up / PF down, S6fall OA up, SUB_HIGH | HANDOFF A16-A19, THINK 5, LEVERS_1020 #5-7 #16 | CLOSED: knob class is 2 of 11 on the LB, and gold58 mis-called 5 of 5 on 10-19 (CLAIMED) | deep (LB) |
| 3 | TTA and spare runtime | flip (in), R6 shifted crop (E3), multi-FOV, multi-scale, vertical flip, S6fall flip, r224 mirror, window density, the ~2.9 h spare | THINK 2.3 / 8.2, runtime.md | CLOSED. Non-mirror views are 0 of 3; spare runtime is not a lever. **Intensity / gamma TTA: never as TTA** (only a robustness spec, unseen_sites sec 8) | deep (spatial), gap (intensity) |
| 4 | Blend operators | rank mean (in), logit, prob, geo, median, min / max, trimmed, z, probit-rank, soft-OR E1 / invgeo, Raptor re-rank, AND / OR severity | THINK 5 / 8, MORE #2 #17, PUSH 4, LEVERS_1020 #22 | CLOSED (E1 = lottery, DEAD for a slot) | deep |
| 5 | Learned stacking | cross-label logistic stacker, co-occurrence stacking, R6 state stacker (Baker's raw z) | THINK 5, TRICK #10 | CLOSED (stacker gain was v2-reader noise; co-occurrence -.001) | medium |
| 6 | OA-trio decorrelation (E-OA) | sym / pw, a .1-.45, OAI MOAKS ruler, coupling-matched transfer | TRICK, EX, MORE, ROUND5, PUSH, VAL, A20-A22 | SHIPPED in A (swide = pw .25). Dose optimum ~.225-.30; nothing >= .35 (CLAIMED VAL) | deepest |
| 7 | E-OA shape / placement | asymmetric LP, convex CVX, concave, probit, agreement-gated, per-member, V224 placement, SUB_LOW | EX C2-C5, MORE #5-7, LEVERS_1020 #2 #17-21, LEVERS_1021 #2 | CLOSED (CVX runner-up, DEAD for 10-22) | deep |
| 8 | Other cross-label pairs | full 66-pair census; menisci, fluid, OA-meniscus, ACL, global factor, Syn <- Eff, Cont <-> Frac, MCL / Frac minus degenerative, C6', C14, BOR, LOA <- MM (lomm05), MCL <- MM, MOACF, regime pairs | EX, MORE, PUSH census, NIGHT, stack sec 8, LEVERS_1021 | Complete. lomm05 + MCLmm ride inside swide; MOACF optional, not recommended (+0.14e-4 after P(clean)); the rest KILL | deepest |
| 9 | Rubric / severity definitions | T-STATE, T-VOL, ACL / MCL subclass demotion, meniscal signal, Synovitis broadening, rubric-mapped R6 states | LEVERS_1020 #8-14, MORE #19 | DEAD on independent rulers | medium |
| 10 | Guards / data handling | missing planes, laterality, G-FLAG, G-NAN, Raptor zero volume, failed-row fills, slice order, both knees in FOV | TRICK lenses, MORE guards #8-16 | DEAD (<= 1e-6 to 1e-4, dormant) | deep |
| 11 | Hidden-run reliability | S6F canary paths, start guard variant g85j, resubmit copies, P(clean), timing | CANARY_1021, cause.md, bpick 3 / 7, LASTDAY | Resubmit r6pw20 v1 2-3 times; g85j SKIP | deep (today) |
| 12 | Transductive | duplicates, kNN smoothing / label transfer, AdaBN, pseudo-labels, patient linkage | TRICK transductive.md | DEAD. The anonymizer salts PatientID per study, so there is no linkage (CLAIMED transductive 2.3) | medium |
| 13 | Site / domain | unseen sites, per-site re-levelling, intensity norm, site priors, header / OOD-gated E-OA, SITE scenario | TRICK unseen_sites, PRIVATE fresh F1-F3 | DEAD as a lever; site risk is carried in the pick model | medium |
| 14 | Metadata covariates | PatientSex (null), PatientAge (absent), contrast keyword (1 / 458), language-level site priors (gold58 -.0152) | TRICK sec 7 B2, THINK 5 | Partial. **Protocol / sequence content never tested -> G1** | superficial |
| 15 | Training | own members, OAI warm start, decoupled OA head, OAI-trained OA head | ROUND5 C, LASTDAY #5-7 | NO-GO: the freeze has passed and there is no ruler | deep |
| 16 | Pick strategy | A rule (R1 / H2 / pw20 fixed), B choice (R6, R6e20, r6pw20, R6pw25, eoa10, pw20, S6F, HW, c4), auto-select hazard, family shock, step vs smooth curve | picks, decision, bpick x3, apick, PRIVATE, readtable, plan1022 | Most-examined category. A = swide; B by the earliest-clean rule | deepest |
| 17 | Public-display tactics | public-only twins, .960 fishing, re-doses | DAY, LASTDAY #8 / #15 | DEAD | medium |
| 18 | Forum / leaders / compliance | public tricks, leaders' timeline, OAI-weights ruling | FORUM_10-20, leaders.md, forum tracks | No shared trick. Compliance GRAY (Eesh's call) | deep |
| 19 | Ties / numeric precision | float32 ties, tie-break by a member | kernel_audit, PUSH 5 #7, e1/info_hidden_ties.log | Dismissed by a GUESS ("first-order 0") but with a MEASURED ceiling -> G2 | superficial |
| 20 | Kernel correctness | alignment, label order, flip TTA, decoders, logs | kernel_audit.md | Clean | deep |

## 2. Gaps (never examined, or only superficially) and minute-triage

Bar: plausibly > 1e-4 private AND shippable by 10-22 09:00 IST. A 10-22 entry needs build + push + GPU commit + gate + A23 + GO tonight,
has P(clean) ~.55 (CLAIMED CANARY), and cannot be A unless Eesh writes an exception.

| gap | > 1e-4 plausible? | shippable by 09:00? | verdict |
|---|---|---|---|
| G1 DICOM protocol / sequence cues as label evidence | Low a priori (site priors lost); **measured, sec 3** | Only as a new A-eligible cell-18 entry (same path as MOACF) | **DEAD** (measured) |
| G2 Tie-breaking of the final output | No. The ceiling at perfect tie-breaking is 8.0-8.3e-5 per label at n 910 (MEASURED info_hidden_ties.log). swide re-ranks the OA trio + LOA + MCL, so about 7/12 x 8e-5 ~ 4.7e-5 macro at most; ~1-2e-5 at a realistic in-tie AUC of .6-.7 (MODEL) | New entry, cannot be A | DEAD (bounded) |
| G3 Intensity / gamma TTA | Unknown sign, no ruler (spatial analogues 0 of 3) | No: GPU worker patch, and it adds hidden runtime on the day runtime is what canaries | DEAD |
| G4 Fold- / checkpoint-level aggregation inside R6 / Raptor | GUESS < 0.5e-4 (CLAIMED LEVERS_1021 2.1 #10), knob class | No: pinned worker change + GPU | DEAD |
| G5 Purpose-built max-of-two B (R6 + swide's cell-18 blocks) | No. R6pw25 as B was already KILLED vs r6pw20 (CLAIMED plan1022 #3), and r6pw20 vs R6 is only +0.52e-4 E (CLAIMED bpick) | Needs build + commit + gate tonight; P(clean) .55 | DEAD |
| G6 Hidden-composition side channels (submission total_bytes) | No decision depends on them | n/a (read-only) | INFO only (sec 0 item 5) |
| G7 Synovitis as an expert-broad label | No ruler (CLAIMED LEVERS_1020 #13, NIGHT 2.1 #5) | No | DEAD |

## 3. Measurement: G1 DICOM protocol / sequence cues (MEASURED `hdr_cues.py` -> `hdr_cues.log`, 3m45s CPU)

Data:
- 458 train studies with a header dump (`trick/box/dicom_series_tags.csv`);
- R7 OOF as the image-model stand-in;
- report_labels_v2 on the 400 non-gold studies;
- gold58 expert labels;
- report language as the site proxy.

ProtocolName and InstitutionalDepartmentName are empty everywhere. Features: an ACL / oblique series, a PATELLA series, 3D, GRE, STIR,
3 T, log series count, log slice count, and a repeated series description.

- **A. Within-site association** (logistic: y ~ R7 z + language FE + feature, 12 labels x 9 features):
  - sum z^2 191.3 vs a within-language permutation null mean of 96.7 (200 perms, P .000); max |z| 3.53 (P .015).
  - Largest cells: repeated series vs Medial Meniscus -3.53 and Medial OA +2.61; 3D vs Lateral OA +2.67 and PF OA +2.46; STIR vs
    Contusion -3.15.
  - So the cue is real in the report regime.
- **B. Shippable form:** final = pct(pct(R7) + b pct(header score)), with the header score from a 5-fold CV logistic without language.
  The R7 base report-OOF AUC is .864, far weaker than c4, which flatters side information.

  | b | report-OOF macro-12 | labels > 0 | gold58 expert macro-12 | gold58 labels > 0 |
  |---|---|---|---|---|
  | .05 | +.00069 | 8 / 12 | **-.00220** | 1 / 12 |
  | .10 | +.00026 | 8 / 12 | **-.00494** | 3 / 12 |
  | .20 | -.00329 | 6 / 12 | **-.01491** | 1 / 12 |

- **C. Placebo** (header rows shuffled, b .1): -.00289 (sd .00106). Observed +.00026, P .000. The signal is real on report labels,
  but the net gain is about 0 even on a weak base.
- **Read (GUESS on MEASURED):**
  - The cues encode protocol and patient mix, which the report regime shares. Expert labels reject them at every dose, as they
    rejected the site priors.
  - On c4's stronger base the report-side gain would shrink further.
  - P(> 1e-4 private) ~.05. Not worth a build. No entry, no amendment.

## 4. What this changes
- Nothing in LASTDAY_1021 sec 1 / 3. Keep the r6pw20 v1 copies, A = swide 57464326, the B rule, and selection by hand with ids
  written down.
- For whoever updates HANDOFF: metadata is now closed at the protocol level too (G1). The tie lever is closed by its ceiling (G2). The
  "exception" Eesh saw is the in-notebook S6F canary, not a Kaggle error status (sec 0 item 5).

## 5. Files (`session_2026-10-21/final/`)
- `critic.md` (this note).
- `hdr_cues.py`, `hdr_cues.log` (G1 measurement; aggregates only).
- `subs_now.json` (read-only submissions list, 108 rows, all COMPLETE).
