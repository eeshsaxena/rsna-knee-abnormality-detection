# ROUND5.md: R6 on OAI, picks per read, training go / no-go, actions for 10-20 (written 10-20 ~06:50 IST, Claude, r5 synthesis)

Research only. Round 5 did not submit, push, commit, build or download anything, and created no dataset. It did not edit any
scheduled task, PREREG, HANDOFF, PROGRESS or session_2026-10-19/a20 file. Three governance slips are disclosed in sec E.

**Tags:**
- MEASURED: measured, log named.
- CLAIMED: an older note says so, source named.
- MODEL: arithmetic on stated inputs.
- GUESS: judgement.

**Units:** "OA-mean" = change in the mean AUC of the 3 OA labels. Macro-12 = OA-mean x 3 / 12. "pts" = percentage points of
P(silver) from the 10-19 field model.

## Verdict (plain)

1. **The plan does not change.** Wait for the eoa10 / eoa20 reads, then set A = API best-id (rule R1) and B = R6 57248084. Both
   entries were PENDING at 06:44 IST (MEASURED). Eesh selects the two picks by hand.
2. **R6 and S6fall read OAI expert labels well** (MEASURED, n 1,704). But on OAI, E-OA helps c4's own members less than hoped.
   A proxy built with c4's real OA weights scores +.0003 / -.0014 OA-mean at a .1 / a .2.
3. **New hidden estimates** (MODEL / GUESS):
   - eoa10: about +.0002, P(>0) about .6-.65.
   - eoa20: about -.0001 to 0, P(>0) about .4-.45.
   - P(eoa20 beats eoa10) is about .3-.4.
   - Before round 5: +.0003 (P .75) and +.0001 (P .55-.6).
4. **The pick model, re-run with these priors, still backs the rule in every outcome** (MODEL):
   - After a no-move, eoa10 should be neither A nor B (7 to 8 pts worse).
   - If eoa20 is the best-id, keep it over eoa10 (3 to 4 pts better).
5. **Training is NO-GO.** A held-out OAI fold overstates OA gains 5-7x and passes models that hurt inside c4 (MEASURED). Let the
   12:00 IST freeze pass without starting anything.
6. **Drop the "R6 + E-OA as B" side option and any a .3 step.** OAI measures R6 as the member that does worst under E-OA.
7. **For the rest of 10-20:** no builds and no extra submits. Read once both entries have scored (about 11:15-14:30 IST), select
   the pair straight away, and re-check the selection on 10-22 at about 18:00 IST.

## Status at write time (MEASURED, 10-20 06:44-06:46 IST)

- **Submissions** (read-only `kaggle competitions submissions`): both PENDING.
  - eoa20 = 57413368, submitted 05:31:38 IST.
  - eoa10 = 57413380, submitted 05:31:42 IST.
  - `a20/tick_a20.log` agrees.
  - The tick submits at most once (lock plus a description check) and stops after 05:20 IST 10-21 (CLAIMED `tick_a20.py` header).
    Leave it alone.
- **Expected scoring:** c4-class runtime is 5h40-6h27, median 6h05 (CLAIMED `runtime.md`). With no queue that puts the scores at
  about 11:10-12:00 IST. A queue can add 2-4 h (CLAIMED RESEARCH_1017).
- **Box:** the V100 has 0 compute processes, 5 MiB used, 0 %, 80 C (its idle temperature). `~/r5_oai` holds 46 GB.

## A. R6 (and S6fall) on OAI: does E-OA help c4's real members?

### A.1 What was measured

**Why a rebuild was needed:** the V00 raw DICOM tarballs were pruned on 10-09 (`~/oai_raw/image03/00m` has 0 tar.gz; MEASURED).
So the DICOM series were rebuilt from the 320 px fleet store (`~/oai_store/*.npz`) with fabricated geometry. The unchanged c4
serve code then ran on them (r6_oai.md sec 2).

**Runs:**
- R6: bundle fe055406, 4 folds, flip TTA. It scored 1,704 OAI knees in 2.75 h of GPU. These are a stratified prefix that keeps
  the OAI OA 8-cell joint distribution; 695 knees were not reached before the deadline.
- S6fall: scored all 2,399 knees in 0.87 h.

**Rebuild check on RSNA gold58** (real DICOM vs rebuild, n 58; MEASURED `box_r6_final.log` part A, `box_g58_bias_*.log`):

| | R6 real | R6 rebuilt | S6fall real | S6fall rebuilt |
|---|---|---|---|---|
| macro-12 AUC | .9366 | .9409 | .9314 | .9242 |
| mean Spearman, real vs rebuilt outputs | | .933 | | .950 |
| OA model rho | .778 | .737 | .845 | .870 |
| coupling gap (rho - q) | .092 | .041 | .131 | .120 |
| gold58 E-OA OA-mean, a .1 / a .2 | +.0035 / +.0067 | +.0021 / +.0045 | +.0041 / +.0056 | +.0015 / +.0059 |

- **R6:** the rebuild keeps the per-label signal but cuts R6's OA coupling (gap -.050, 90 % [-.096, +.004]). That biases its E-OA
  result low by about .0014 at a .1 and .0021 at a .2 (sd .0019 / .0039).
- **S6fall:** its rebuild bias is effectively unmeasured. The n-58 bootstrap is unstable, and tie handling alone moves the gold58
  E-OA numbers by about +-.0004 (skeptic, r6_oai.md 11.2.5).

### A.2 Per-label AUC against expert MOAKS reads (MEASURED, n 1,704; `box_final_s6.log`)

| Label | R6 | S6fall | .8 R6 / .2 S6fall | D5 (OAI-free fleet reference) |
|---|---|---|---|---|
| ACL | .945 | .965 | .957 | .938 |
| Medial / Lateral Meniscus | .944 / .953 | .945 / .949 | .946 / .954 | .934 / .933 |
| Effusion >= 2 | .958 | .961 | .961 | .957 |
| Baker's | .917 | .897 | .919 | .898 |
| Synovitis (Hoffa >= 2) | .800 | .791 | .801 | .804 |
| OA P_FT Medial / Lateral / PF | .891 / .811 / .859 | .899 / .786 / .854 | .896 / .811 / .862 | .879 / .779 / .813 |

**Medial and lateral are the right way round:**
- Medial OA scores .891 from R6's medial output and .654 from its lateral one. For S6fall the figures are .899 / .651.
- R6's worker side check flipped 1 study of 1,736.
- S6fall has no side check, but its swap check passes.

**How the numbers were checked:** they reproduce exactly with sklearn, and the MOAKS string parse equals the float parse
(0 mismatches) (skeptic, MEASURED).

### A.3 E-OA on the real members (P_FT, OA-mean dAUC; MEASURED `box_final_s6.log`, `box_skeptic_B600.log`)

**How the table was built:**
- "Raw OAI" uses the unweighted OAI knees, where the label coupling is low (phi ~.05).
- "R1" reweights the knees to the gold74 8-cell OA joint (phi .32), which is the skeptic method.
- The P(>0) column adds the uncertainty of the gold74 target (phi 90 % [.20, .44]) on top of knee sampling.
- c4D5 uses c4's real OA weights with D5 in the Raptor slot, as an OAI-free stand-in for Raptor.

| Mix | Raw OAI a .1 / a .2 | R1 a .1 / a .2 | P(>0) a .1 / a .2 (knee + target) |
|---|---|---|---|
| R6 | +.0040 / +.0064 | -.0018 / -.0058 | .14 / .04 |
| S6fall | +.0029 / +.0036 | -.0027 / -.0079 | .03 / .01 |
| .8 R6 / .2 S6fall | +.0041 / +.0065 | -.0016 / -.0056 | .15 / .03 |
| **c4D5 (c4 OA weights)** | +.0060 at a .1 | **+.0003 / -.0014** (P_AREA +.0010 / .0000) | **.58 / .34** |
| R6 + D5, 50 / 50 | +.0064 / +.0116 | +.0011 / +.0003 | .79 / .52 |
| D5 alone (control) | +.0064 / +.0121 | +.0017 / +.0021 | .92 / .78 |

- **Control:** D5 reproduces the skeptic's earlier +.0016 / +.0017 (CLAIMED `skeptic_eoa_work/box_reweight.log`), so the method
  is unchanged.
- **a .1 vs a .2:** every OAI-free mix that contains R6 or S6fall prefers a .1. P(a .2 > a .1) is .16 for c4D5, .33 for
  R6 + D5 and .00-.01 for the members. Only D5 alone is above half (.57).

**c4's real OA make-up** (MEASURED from the c4 kernel, cells 14 and 18; this corrects r6_oai.md sec 6, which read "a25" as R6's
weight):

| Member | Medial OA | Lateral OA | PF OA |
|---|---|---|---|
| Raptor (OAI-trained, cannot be measured on OAI) | .40 | .16 | .40 |
| R6 | .40 | .64 | .40 |
| S6fall | .20 | .20 | .20 |

So the OAI-free members carry about .68 of c4's OA output.

### A.4 Caveats (read before using A.3)

1. **The model coupling collapses on OAI, and only for these two models** (MEASURED):

   | Model | Coupling on RSNA | Coupling on OAI |
   |---|---|---|
   | R6 | .778 (gold58) | .546 |
   | S6fall | .845 (gold58) | .531 |
   | D5 | .826 (RSNA OOF) | .754 |

   - The rebuild explains only about .04 of R6's drop.
   - E-OA works through coupling, so the OAI member cells measure R6 and S6fall in a regime they are not in on RSNA. That makes
     OAI pessimistic for them. How pessimistic is unknown.
   - On gold58 both members gain (+.0035 to +.0067), but n 58 cannot size the gain (sd ~.002-.004).
2. **The DESS swap is not covered by the gold58 check.**
   - OAI's coronal and axial inputs are DESS water-excitation reformats (TE 4.71 ms, TR 16.3 ms), not PD FS TSE. The gold58
     validation covers the dropped non-FS series, but not this swap.
   - The cause of the collapse cannot be split into protocol, population or fabricated geometry, because the raw V00 files are gone.
3. **OAI is a narrow domain:** one vendor (Siemens 3T), one research protocol, 4 sites. The labels are MOAKS reads mapped to RSNA
   labels by our own rules, not the RSNA rubric.
4. **Raptor (~.32 of c4's OA output on average) is OAI-trained and stays unmeasured.**

### A.5 Skeptic verdict and the estimate carried forward (MODEL / GUESS)

**Skeptic verdict:**
- The measurement is sound; it was re-derived with independent code on the box, aggregates only.
- Three things were overstated:
  - c4's make-up (A.3);
  - P(>0) values that counted knee sampling only;
  - the coupling caveat, which was under-weighted.
- The bottom line survives.

**Estimate carried forward** (r6_oai.md 11.4; start from c4D5 at R1, add about half of R6's rebuild bias, then shade up slightly
for the coupling collapse):

| Entry | Hidden macro | P(>0) | Before round 5 (MORE_1020) |
|---|---|---|---|
| eoa10 | about +.0002 (range -.0003 to +.0006) | about .6-.65 | +.0003, P .75 |
| eoa20 | about -.0001 to 0 (range -.0012 to +.0008) | about .4-.45 | +.0001, P .55-.6 |

- P(eoa20 beats eoa10 on hidden) is about .3-.4.
  - OAI c4D5 gives .16.
  - The gold58 gap map on c4's gap (.10, CLAIMED mechanics.md) gives .58-.61.
- Both entries are still worth reading: eoa10 keeps a positive expectation, and a public move is informative (sec B).

### A.6 Consequences beyond the picks

- **"R6 + E-OA a .1 as B" (decision.md sec 0.3 side option): drop it.**
  - It applies E-OA to R6 alone, the member OAI scores worst at R1: -.0018 OA-mean, P(>0) .14, or about -.0003 after the
    rebuild-bias correction.
  - decision.md valued it at 0.9 x d10 on the assumption that R6's coupling matches c4's, which the OAI collapse undermines.
  - It would also copy A's OA change into B and remove the hedge.
- **No a .3 strength step, even if eoa20 wins publicly.**
  - At R1 every c4-like mix already loses from a .1 to a .2: c4D5 goes from +.0003 to -.0014.
  - R6 + D5 at a .3 is -.0031 (MEASURED).
  - Building one would need Eesh's go, a PREREG amendment, a build and about 6 h of scoring.

## B. Pick decision table per read outcome

**Model:** `decision_mc.py` (MODEL), re-run tonight with two round-5 priors (`r5x_r5skep.log`, `r5x_r5sec6.log`,
`dec_r5skep.json`, `dec_r5sec6.json`). The priors are Normal pairs chosen to match the A.5 summaries:

| Prior | d10 | d20 | r | P(d10 > 0) | P(d20 > 0) | P(d20 > d10) |
|---|---|---|---|---|---|---|
| r5skep | N(+2e-4, 5e-4) | N(-0.5e-4, 8e-4) | .85 | .65 | .48 | .29 |
| r5sec6 | N(+1e-4, 4e-4) | N(-1e-4, 7e-4) | .90 | .60 | .44 | .30 |

**Read noise:** the public paired sd is .00021 at a .1 and .00046 at a .2, with corr .96 (MEASURED `noise_corr.log`).

**R1 outcome rule:**
- eoa20 is the best-id if pub20 > 0 and pub20 > pub10.
- eoa10 is the best-id if pub10 > 0 and pub10 >= pub20.
- Otherwise there is no move.

| Read (R1 vs c4 57343435) | P before read: r5skep / r5sec6 (decision.md base) | A | B | P(silver): r5skep / r5sec6 (decision.md base) |
|---|---|---|---|---|
| **eoa20 57413368 is the API best-id** | .29 / .31 (.36) | **eoa20** | **R6 57248084** | .819 / .785 (.864) |
| **eoa10 57413380 is the API best-id** | .38 / .30 (.37) | **eoa10** | **R6** | .780 / .735 (.753) |
| **no move** (c4 stays best-id) | .33 / .39 (.27) | **c4 57343435** | **R6** | .631 / .631 (.631) |
| only eoa10 scored (eoa20 ERROR), moved | | eoa10 | R6 | .789 / .751 |
| only eoa20 scored (eoa10 ERROR), moved | | eoa20 | R6 | .787 / .761 |
| one scored, no move, other ERROR | | c4 | R6 | .631 |
| one still PENDING at 13:00 IST | | wait for it | | |
| both ERROR | | c4 | R6 | .631 |

**Margins of the rule over the alternatives** (pts, r5skep / r5sec6):
- **eoa20 best-id:**
  - eoa10 + R6 is -3.6 / -2.9.
  - eoa20 + c4 is -1.6 / -1.8.
  - eoa10 + eoa20 is -1.5 / -2.0.
  - B = R6 is clearer here than in decision.md, where eoa10 + eoa20 was a near-tie.
- **eoa10 best-id:**
  - c4 + eoa10 is -3.4 / -4.2.
  - eoa20 + R6 is -12.6 / -8.9.
- **No move:**
  - eoa10 + R6 is -8.2 / -7.1.
  - c4 + eoa10 is -7.2 / -7.1.
  - After a no-move, d10's posterior is -2.9e-4 / -2.4e-4 (P>0 .18 / .19), so eoa10 is neither A nor B.
- **Move size:**
  - A hairline eoa20 move (public delta < .5e-4, about 2 % of eoa20 moves) is a tie between eoa20 and c4 as A (.623 vs
    .625). Follow R1 anyway.
  - A hairline eoa10 move is worth taking: +3.3 to +4.3 pts over c4 + R6.

**Value of the two entries** (pre-read, following the rule):
- P(silver) .742 / .710 against .631 for c4 + R6 alone, i.e. +8 to +11 pts (decision.md had +11 to +13).
- Following the read beats always using eoa10 + R6 by +3.7 / +3.8 pts.
- The absolute levels rest on the field model's sharp silver step. The differences are more robust than the levels.

**B stays R6:**
- S6F 57201874 as B shows +1.1 to +2.4 pts, but that is the known artifact of S6F's single-fold noise stand-in (decision.md sec
  0.3).
- HW, a25 and c4 as B are worse.
- P(gold) is about 0 in every cell.

## C. Training go / no-go: NO-GO for every option

Source: training.md.

**Why: no independent ruler for a trained model.** The K14 OAI-trained / OAI-free twin pair shows the problem:

| Ruler | OA-mean gain | Lateral OA gain |
|---|---|---|
| Held-out OAI (1,269 follow-up visits / 637 knees never trained on) | +.072 (strict cohort +.091), P 1.00 | +.154 |
| RSNA radiologists, model alone (gold58 / gold74) | +.0145 / +.0104 | +.0077 / +.0088 |
| Inside c4 (OA trio at .10 / .20, macro) | -.00027 / -.00096 (CLAIMED ours.md) | |
| Leaderboard (whole fleet) | 1 in-band move, 2 non-moves | |

- The held-out OAI and gold58 / gold74 rows are MEASURED (`box_taskC/ho_eval.log`, `work_c/g58_twin.log`).
- So a held-out OAI fold certifies "learned MOAKS on OAI". It does not certify "adds to c4 on hidden".
- Round 5 adds a second reason: even RSNA-only models shift in a model-specific way on OAI (A.4 item 1). So OAI cannot select a
  trained model for hidden either way.

**Other reasons:**
- gold58 / gold74 is too small: it mis-called 5 of 5 probes on 10-19 (CLAIMED PROGRESS_19).
- Our own-trained components on the strong base are 1 of 7 on the LB (CLAIMED training.md sec 4).

**Best-mechanism candidate, C1: R6 warm-start with OAI MOAKS, 4 folds, full R6 swap.**
- Timing: about 38 h nominal or about 47 h slipped. With a go at 10:00 IST, the read lands 10-21 about 23:50 or 10-22 about
  09:15. That is one blind shot, and it crosses the 10-20 12:00 IST freeze.
- Value (MODEL): -1e-4 to +1e-4 hidden macro, P(true > 0) .40-.45, EV about -1 to +0.5 pts.
- Every other candidate (OA-trio-only C1, a decoupled OA head, decoupled-target member, K14 routing, rescaled S6 seed, new OAI
  fleet) is dominated, measured negative, or breaks the S6F runtime guard (training.md sec 3).

**What would change this:** nothing that could still be read before 10-22 about 10:15 IST. If eoa20 and eoa10 both move with
eoa20 on top, that argues for more decorrelation. But A.6 argues against an a .3 step, and training stays dominated.

## D. Exact actions for 10-20 (IST)

| When | Who | Action |
|---|---|---|
| now to ~11:00 | nobody | **Nothing.** No builds, no submits; the 3 unused 10-20 slots stay unused. Leave RSNA_A20_tick alone (already submitted both, at-most-once). Box stays idle; no GPU job is worth queueing (A.6, C). |
| from ~11:15, every ~30 min | Claude or Eesh | Poll the status (read-only): `kaggle competitions submissions -c rsna-knee-abnormality-detection` (top 3 rows: 57413380 eoa10, 57413368 eoa20). |
| 12:00 | nobody | The training freeze passes. Nothing is launched (C). |
| when BOTH show COMPLETE (or one ERROR) | Claude | Read R1. Run `cd session_2026-10-14/research960/lbprobe && python fetch_api_lb.py api_lb_1020_read.json` to get our team's best submissionId. Map it: 57413368 = eoa20, 57413380 = eoa10, 57343435 = c4. R1 uses the unrounded API best-id: a displayed .958 on both does NOT mean no move. |
| 13:00 if one is still PENDING | both | Wait for it: the pick depends on the final state. If it errors, use the partial rows in sec B. If it is still pending at 14:30, keep waiting; the hard deadline is days away. |
| right after the read (target before ~15:00) | **Eesh, by hand** | On the Kaggle Submissions page tick "Use for Final Score" on exactly two entries, per sec B: eoa20 best-id → 57413368 + 57248084; eoa10 best-id → 57413380 + 57248084; no move → 57343435 + 57248084. Untick any earlier selection (for example HW 57262115). Screenshot the checkmarks. |
| after the pick | main session (not this task) | Write the read into the PREREG A20 result, HANDOFF and PROGRESS. If there was no move, mark E-OA closed and stop. |
| 10-21 | nobody | Nothing planned. Any new idea needs Eesh's go plus a PREREG amendment; round 5 found none worth it. |
| 10-22 ~18:00 | Eesh | Re-check that exactly the two checkmarks are set. Hard deadline 10-22 23:59 UTC = 10-23 05:29 IST. If nothing is selected, Kaggle auto-picks the top 2 public entries, which would be near-twins. |

**Optional, decision value ~0, only if Eesh wants the box busy:**
- Serve R6 real vs rebuilt on ~200 RSNA train studies to tighten the rebuild bias (now n 58).
- Cost: about 1 h of GPU plus a CPU rebuild.
- It cannot change any pick, because the sec B rule holds under every prior tried.

## E. Governance and box state

**Slips** (OAI rows reached session transcripts only; no local file holds OAI rows or ids, CLAIMED by each lens):
1. Task A, ~01:50: `head -3` / `head -5` over ssh on 4 OAI CSVs (`converted_manifest`, `oai_series_meta`, `oai_labels_T123`,
   `oai_label_sources`). This printed 2-4 rows each: knee ids, tarball paths, series UIDs and per-knee grades.
2. Task C: `head -2` on 2 OAI CSVs. This printed 1 OAI knee id, its 12 teacher probabilities and its 12 MOAKS label values.
3. Skeptic: printed the `fallback` field of `run_r6/worker0/receipt.json`, cut to 600 characters. That is about 12 OAI knee ids
   with the text "not reached before the deadline" (no grades). After that the skeptic piped all output through an id-redacting
   `sed`.

**Box** (MEASURED 06:45 IST unless tagged):
- New round-5 files live only under `~/r5_oai` (46 GB, mode 700): rebuilt DICOMs, predictions, id lists, `taskC/` and
  `skeptic/`. All OAI-derived, and they stay on the box.
- Task A deleted two `__pycache__` folders it had created by mistake, one in `~/r6g58/src`. That directory is back to its prior
  state (CLAIMED r6_oai.md sec 9).
- GPU idle: 0 compute processes.
- Round 5 killed only its own processes and never used pkill.

**This synthesis:**
- Read-only steps: one `kaggle competitions submissions` call and one ssh `nvidia-smi` / `du`.
- Compute: two local CPU runs of `decision_mc.py` (under a minute each, competition-free inputs).
- Only new local files were written.

## F. Files (`session_2026-10-20/r5/`)

**Notes:**
- `ROUND5.md`: this file.
- `r6_oai.md`: task A plus the skeptic in sec 11.
- `decision.md`: task B.
- `training.md`: task C.

**Box logs (aggregates only):**
- `box_r6_final.log`, `box_final_s6.log`, `box_s6_all2399.log`
- `box_partA_r6.log`, `box_partA_s6.log`
- `box_g58_bias_r6.log`, `box_g58_bias_s6.log`
- `box_partB_smoke300.log`, `box_partB_mid1000.log`
- `box_skeptic_B600.log`
- `box_taskC/ho_eval.log`

**Scripts:**
- `box/`: copies of the task A and skeptic box scripts.
- `box_taskC/`: copies of the task C box scripts.
- `decision_mc.py`
- `work_c/g58_twin.py`

**Pick model outputs:**
- `dec_*.json`, `summary.log`, `policies.log`, `sweep_width.log`, `noise_corr.log`
- New tonight: `r5x_r5skep.log`, `r5x_r5sec6.log`, `dec_r5skep.json`, `dec_r5sec6.json`.
