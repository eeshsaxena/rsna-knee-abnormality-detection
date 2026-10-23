# Verify: dimension C (label levers), adversarial pass (2026-10-08 ~22:45 IST)

Read-only. Laptop light Python only (system Python 3.12, < 10 s CPU). No Kaggle write, no API call, no GPU. The box
was unreachable (ssh port 22 timed out at ~22:40 IST), so box tables were re-derived locally from
`research_1008/C/box_preds/` instead.

Tags: **MEASURED** (file named), **MODEL**, **GUESS**, **CLAIMED** (forum).

## Checks re-run

| check | result | source |
|---|---|---|
| K11 file md5 | e7371c1ae1068dc7c6b0d991762a1316, matches | MEASURED, md5sum |
| frac75 file md5 | ad7582b817243f8f22e8bcfdd7e3f84a, matches | MEASURED, md5sum |
| transfer.out and target_menu.out numbers vs the claims | all quoted numbers match | MEASURED |
| Student vs target per-label gold58 (set A / set D / mix50 target) | Contusion .958 / .965 / .906; Synovitis .793 / .781 / .823; **PF OA .853 / .853 / .888**; Effusion .995 / .996 / .984; Lateral OA .834 / .818 / .843 | MEASURED, recomputed from box_preds |
| Wholesale pairs: pooled per-label slope with one intercept per pair | **-.090 (SE .100)**, intercepts +.011 / +.018 / +.016 / +.018 | MEASURED, recomputed from transfer.json |
| frac75 set Fracture ratio SE including seed noise (per-label seed SD .012) | ratio .18, SE **.43** (not .34) | MODEL on MEASURED |
| "frac75 rejects the wholesale model at ~2 SE" | 2.18 SE on bootstrap-only SE; **1.06 SE** once seed noise (macro SD ~.005, from the B4-A4 null) is added | MODEL on MEASURED |
| Efron package, decision-conditional (count 0 where the pick is mix50 or tied with it) | macro **+.0024** incl. Fracture; **-.0005** excluding Fracture (Fracture is already in frac75) | MODEL on MEASURED target_menu.json |
| Hidden-runtime room for an added 224 set | K9N slow margin already **0.65 h** (PROGRESS 10-08 l.229), below C's own 0.75 h gate; clean K9 ~1.07 h slow (0.65 + ~25 min N stage), so +20 min leaves ~0.74 h | MODEL on MEASURED |
| VLM pilot state | **BLOCKED at the first call, $0 spent** (PROGRESS 21:10: HTTP 401). Eesh must run `run_pilot.py` with his own key; dry-run cost ~$3.79 | MEASURED, PROGRESS_08-10-2026.md |

## Verdicts

### 1. Stop label column surgery: REVISE (conclusion holds, numbers and units wrong)
- **Unit error.** The chain ".005 dT" is per 1.0 AUC unit, not per "1.0 point". Read normally, +.01 of per-label
  target AUC gives **+.00005 SOLO**, not +.005. The internal numbers (Fracture +.0458 -> +.0002 SOLO) use the right unit;
  the headline sentence does not.
- **The .144 "local slope" is not a measurement of column transfer.** It is a through-origin fit dominated by wholesale
  pairs whose gain is a common intercept. With one intercept per pair the per-label slope is -.09 (SE .10) (MEASURED).
  The only column-level data are the frac75 ratios .18 (SE .43 with seed noise) and .52 (SE .69). True column transfer
  is unknown in roughly 0 to 1.
- **The "~2 SE rejection" of the wholesale model is ~1 SE** once seed variance is included (the seed-only null itself
  moved macro gold58 by -.0051).
- **The conclusion survives the upper bound.** At transfer ratio 1.0, +.01 target -> student macro +.00083 -> SOLO
  +.00033. The best remaining corrected item (Contusion +.0177) would be ~+.0006 SOLO, ~+.0002 in K9N (MODEL). Still
  not worth T4 hours. Corrected claim: every per-label fix beyond frac75 is worth **+.00005 to +.0006 SOLO, <= +.0003
  blended** (MODEL).
- kappa: the two anchors give .38-.40, but the Tian anchor gives ~.74; use .4-.7 (MEASURED anchors). Does not change
  the verdict.

### 2. Efron optimism: REVISE (sign of the package number is an artifact)
- The per-label "corrected" value for labels where the chosen w is mix50 itself (MCL, MedMen, Syn, Bak) or tied with it
  (ACL, LatMen at w .25, apparent 0) is reported as -optimism (e.g. Synovitis -.017). The actual decision there is "keep
  mix50", whose true gain is exactly 0. Efron's optimism here describes the selection procedure, not the chosen recipe.
- Decision-conditional package: **+.0024 macro target** including Fracture, **-.0005** excluding Fracture (MODEL on
  MEASURED). Since Fracture is already done (frac75), the practical conclusion ("no further per-label package") stands.
- The Fracture / Contusion survival and the PF OA ~0 corrected are correct. Epoch optimism +.0022 (SD .0016) verified
  in epoch_opt.out. gold16 report-read caveat stands.

### 3. VLM relabel pilot: REVISE (state is stale; value range too narrow)
- **The pilot is not running.** It was blocked at the first live call (401) with $0 spent. It is not "sunk"; nothing
  lands 10-09 unless Eesh runs `run_pilot.py` with his own key (~$3.79 dry-run estimate, cap $8).
- **The local model should not be applied by default.** A VLM relabel adds new image information in report-silent cells
  for ~1,806 studies, the same mechanism as the v4 -> mix50 teacher fill (wholesale regime), unlike frac75 (a re-weighting
  of two existing sources). And frac75 rejects the wholesale model at only ~1 SE.
- Corrected value if scaled (MODEL): S .01 -> **+.0003 to +.003 SOLO**, K9N blend **+.0001 to +.001**. At S .03 the
  wholesale end is implausibly large; cap at ~+.003 SOLO (GUESS).
- Feasibility: a scale-up needs a NEW Eesh spend approval (> $8), 1-2 days of labelling, then 5 x 224 folds (~27 T4 h)
  by ~10-14. Possible before the 10-20 freeze, but the added set hits the same runtime wall as K11 (see 4) unless it
  REPLACES set D. A relabelled set is a legitimate replacement for D (new labels, gate on paired student).
- Gate advice (S >= .03 AND img - text >= .02) kept, plus: a scale-up should replace a set, not be added.

### 4. K11 as a free twist on a 224 reseed: REVISE to PARK (runtime makes "added set" infeasible)
- File verified (md5 e7371c1a, 11 columns identical to frac75).
- **Runtime:** C's own use rule requires >= 0.75 h slow margin. K9N already sits at 0.65 h slow; the clean K9 would drop
  to ~0.74 h with an extra 224 set (MODEL). So K11 cannot be ADDED to either final kit, and C itself forbids replacing D
  on gold evidence. The T4 plan therefore produces a set with no slot.
- **PF OA is student-limited too:** set A and set D both read PF OA .853 vs the mix50 target .888 (MEASURED), the same
  pattern C uses to close Synovitis. That lowers the odds the +.0077 target bump transfers.
- The "reseed value +.0004-.0006 SOLO, MEASURED analog" has no cited measurement; K5 (2 seeds, A+B) did not move the 3rd
  decimal. Treat it as MODEL/GUESS.
- Corrected gain: **+0 to +.0001 blended** (MODEL). Action: keep the file, schedule no T4 sessions for it; reuse it only
  if a future kit swaps a set and needs new labels anyway.

### 5. Pre-registered K11 gate: KEEP (moot while K11 is parked)
- Bars and noise floors match transfer.out / epoch logs. Correctly framed as no-harm only. If K11 is ever run, copy it
  into a PREREG.md before the push.

### 6. Close Synovitis / Lateral OA / Effusion / Contusion / further Fracture: KEEP
- Re-derived locally: Contusion student .958 (A) / .965 (D) vs target .906; Synovitis student .793 / .781 vs target
  .823; Effusion at ceiling (.995 / .984). Effusion-as-Synovitis numbers match target_menu.out. Only nuance: the
  Synovitis "-.017 corrected" carries the Efron artifact from item 2; the true value of "keep mix50" is 0, which still
  means close.

### 7. OAI image-read labels: REVISE (gain inflated by double count with N; feasibility narrower)
- K9N already carries OAI signal through N (r with the parent ~.93-.96, F_intel). An own OAI-trained fleet correlates
  with N, so its blended increment over K9N is **~0 to +.0005** (GUESS), not the CLAIMED +.001-.003 SOLO analog.
- Compute: OAI data must stay on the box, so T4 sessions cannot be used; only the single V100, already queued with the
  320 folds (~7 h/fold). The OAI package was still not downloaded at ~22:00 (Eesh relay: "package still not read").
  Realistic earliest own-OAI members ~10-14 to 10-16 (MODEL), squeezing the 320 plan.
- Rule risk now LOWER than stated: DAR 26647 approved as an individual with no institution (PROGRESS 21:20, MEASURED).
- Mapping: BML -> Contusion should be **masked**, not soft (degenerative BMLs are not contusions; teaching that would
  move Contusion away from the expert concept where the student already beats its target) (GUESS). MCL / Fracture mask
  is fine. All field names still GUESS until the dictionary is read.

## Levers this dimension missed

1. **Few-shot text relabel (wholesale, cheap).** CLAIMED (FORUM_DIGEST sec 4): adding labelled examples to a report-only
   Claude labeler cost .004 local CV but gave +.002 public LB. This is a wholesale relabel (the regime with large
   transfer), text-only (cheaper than VLM, ~$2-6 GUESS), but needs Eesh's key and spend OK, plus a 5-fold retrain that
   must replace a set (runtime wall). Blended GUESS +.0002 to +.001. Caveat: examples must come from train reports only
   (no gold58 leakage into the prompt if gold58 stays the judge).
2. **Third "unknown" state / masked loss on report-silent cells** (Mingquan, CLAIMED) as an alternative to teacher
   fill. Partly covered by gated75 (worse), but a soft-masked loss (down-weight, not hard-zero) was not tested. Low
   priority: GUESS ~0.
3. **Label-side runtime accounting.** Every label lever that ends in a new 5-member set needs a slot; with K9N at 0.65 h
   slow margin, label work must be planned as a REPLACEMENT of set C or D (or of the parent's weakest stage), which
   changes its value (replace = gain minus the replaced set's contribution). C never priced that.
4. **PF OA student gap (.853 vs target .888) is a student-side lever,** e.g. a crop or plane emphasis for the patellofemoral
   compartment (axial plane). Belongs to the model dimension; flagged here because the label analysis exposed it.
5. **Distilling N or the K9N blend into own members** was not discussed. It would raise SOLO but increase N-likeness;
   likely net-negative in K9N (GUESS). Worth one line in Settled.
