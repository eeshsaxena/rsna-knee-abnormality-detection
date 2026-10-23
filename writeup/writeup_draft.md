# 246th Place (Silver): Report Labels Are Not Expert Labels

*A public-weights ensemble plus one validated post-processing step: decorrelating the three osteoarthritis compartments.*

Thanks to RSNA, the annotators and Kaggle for a hard, well-run competition, and to the authors whose public models carry most of our score (credits below). We finished **private .944, rank 246 / 6,425 (silver)**, public .960 (rank ~157 at close). This is not a top solution. We write it up because the one idea that moved us, and the way we checked it on an independent expert-labelled set before spending a submission, may help anyone who trains on report-derived labels and is scored on expert reads.

## Overview of the approach

![pipeline](figs/pipeline.png)

Every step works on percentile ranks, $\mathrm{pct}(\cdot) \in (0, 1]$, per label:

1. **Raptor slot:** .75 pct(CoAtNet r384) + .25 pct(CoAtNet r224).
2. **Per-label blend** of the Raptor slot and **R6** (a compact ConvNeXt-S bundle) with a per-label Raptor share.
3. **Final blend:** .8 pct(blend) + .2 pct(S6fall).
4. **E-OA:** decorrelate Medial OA, Lateral OA and PF OA (ours, the key idea).
5. Pick A only: two small cross-label tweaks of the same kind.

Steps 1 to 3 are public weights inside our own plumbing. Step 4 is worth about +.001 on both public and private (display). We trained many models of our own; none survived into the final blend (see "What did not work").

## Base ensemble (public building blocks)

| component | source | what we did |
|---|---|---|
| Raptor r384 | nartaa's OAI-trained CoAtNet SWA weights (rsna-knee-publication-swa-weights-20261007), anatomical mirror TTA | used as published |
| Raptor r224 | the second checkpoint in the same dataset (224 px, 0.8 centre crop), first blended by varshithakolupuri at .35 of the Raptor slot | share .35 to .25 (one read) |
| R6 | heliosli's compact ConvNeXt-S bundle (version 1 of rsna-knee-r6-frozen-fe055406), 4 folds, its own Effusion / Baker's stackers | horizontal flip TTA forced on (varshithakolupuri's "free flip" idea; on the older R7 bundle it lifted us ~214 places inside the .956 band) |
| per-label Raptor share | heliosli's per-finding table from "blend-mine-in-private-lb-luck-to-all-upvoters" | four entries changed (read together as entry c4); final: Medial OA .5, Lateral OA .2, PF OA .5, Synovitis .8, Baker's .5, Lateral Meniscus .4, Fracture .65, others .5 |
| S6fall | nartaa's retrain of heliosli's S6 architecture on blended LLM report labels | weight .2 (.15 and .3 did not move the best-id) |
| JPEG Lossless decoder cell | pylibjpeg cp313 wheels from glorysing/rsna-knee-offline-codecs-r09 | insurance, see Engineering |

Public path: S6F .957, R6 swap .958, then three best-id moves inside the .958 band (HW, a25, c4), then E-OA .959 and the swide stack .960.

## The key idea: report labels vs expert labels

Only 58 training studies carry expert labels. Every model in our blend learned from labels that LLMs extracted from the radiology reports. The hidden test is graded by expert image review (the host: two readers plus an adjudicator, the same process as the 58 sample labels), and the rubric grades osteoarthritis **separately in each compartment** with a strict threshold.

Reports are not written to that rubric: they tend to describe degenerative change at the level of the knee, so report-trained models learn to fire on the three compartments together. On gold58, most of our Lateral OA errors were knees where the report-derived label was positive and the experts said no. We measured the coupling (pairwise, over Medial-Lateral, Medial-PF and Lateral-PF):

![coupling](figs/coupling.png)

| source | OA-OA coupling |
|---|---|
| our models' predictions (rank correlation) | .69 to .85 |
| report-derived labels (phi) | .53 to .61 |
| expert labels, gold58 / gold74 (phi) | .24 to .40 |
| OAI MOAKS expert cartilage reads (phi) | ~.05 |

The models co-fire far more than the expert labels justify. The fix subtracts a little of the other compartments' rank from each OA column:

$$\mathrm{out}_t = \mathrm{pct}\Big(\mathrm{pct}(f_t) - \frac{a}{2}\sum_{o \ne t} w_{t,o}\,\mathrm{pct}(f_o)\Big), \quad t \in \{\text{Medial, Lateral, PF}\}$$

- **Symmetric (eoa):** all $w = 1$.
- **Pair-weighted (pw):** $w$ proportional to the report-minus-expert coupling gap on gold74, scaled to mean 1: w(Lateral, PF) 1.416, w(Medial, Lateral) .876, w(Medial, PF) .708.

It costs zero runtime: nine lines in the last cell, and the other nine columns are untouched.

**When should it help?** In a simple Gaussian model, let $\rho$ be the models' OA-OA score correlation and $q$ the cross-predictiveness of the other compartments' scores for label $k$, $q = \mathrm{corr}(f_{\text{other}}, Y_k) / \mathrm{corr}(f_k, Y_k)$. Decorrelation raises AUC only when $q < \rho$, i.e. when the model couples the trio more than the labels do. Ensembling averages away member noise, which raises $\rho$ but not $q$, so the full blend gains more than any member (on gold58 it had the largest gain of the 12 prediction sets we held, +.0096 OA-mean AUC at a .2).

**Mechanism check:** the same transform *loses* on report-labelled out-of-fold predictions (9 of 9 reader x member cells negative). The sign flips with the label regime, exactly as the mechanism predicts.

## Validating before submitting: an independent expert ruler

gold58 liked E-OA (+.0014 macro at a .1, +.0025 at a .2), but 58 studies is a weak ruler: two skeptic passes showed the gain rested on ~20 pair swaps, and five gold58-picked weight tweaks had just lost on the leaderboard the same day. We needed an expert ruler that had never been used for selection.

**OAI.** Our university machine holds MOAKS cartilage reads by musculoskeletal radiologists for 2,399 baseline knees, plus predictions on those knees from three of our own **OAI-free, report-trained** models (trained on RSNA only, for an earlier experiment). Before running anything we fixed the rule: SUPPORT needs an OA-mean AUC delta > 0 with paired-bootstrap P(>0) >= .90, a positive delta on at least 2 of 3 cartilage proxies (full-thickness, area, both), and a negative a = -.1 control. Only aggregate numbers left that machine.

![oai](figs/oai_validation.png)

| OAI-free model (OA-mean AUC delta, full-thickness proxy) | a .1 | a .2 | control a -.1 |
|---|---|---|---|
| teacher (5 folds) | +.0062 | +.0115 | -.0066 |
| replication 1 (other seed and label set) | +.0050 | +.0086 | -.0058 |
| replication 2 (288 px) | +.0053 | +.0093 | -.0061 |

All rows: P(>0) 1.000, 3 of 3 proxies positive, 5 of 5 folds positive (teacher).

**Transfer.** OAI expert labels are almost uncoupled (phi ~.05), RSNA expert labels are not (~.3), and real disease coupling eats into the gain. Importance-reweighting OAI knees to gold74's coupling shrank the gain 5 to 7 times, to an expected hidden macro gain of about +.0005 at a .1 (P(>0) ~.8) and +.0007 at a .2 (~.7). That cleared our pre-registered bar, so we submitted a .1 and a .2.

**What the rulers got wrong.** They got the sign right and the size wrong, in both directions. Raw OAI and gold58 *overstated* the public gain about 2.5 to 3 times. Our deliberately conservative coupling-matched transfer, the one we used to decide, *understated* it roughly 2 to 5 times and put the optimum at a <= .1, while the leaderboard favoured a ~.2 to .3. Only after five reads could we fit one consistent factor (reading the OAI curve at c4's own coupling gap). Our lesson: an independent ruler tells you the sign; the dose came from the leaderboard.

## Leaderboard results

![lb](figs/lb_progress.png)

Before this pipeline we went from public-stack reproductions (.940 to .943) through our own fleets (.945 to .951) and heliosli's public blends (.954 to .956). Kaggle truncates the public score to 3 decimals but ranks inside a band by the unrounded score, so most of our reads were "did the team best-id move", not display changes.

| entry | what it is | public | private |
|---|---|---|---|
| S6F | public V224 recipe (Raptor + R7) + R7 flip TTA + S6fall .2 | .957 | .942 |
| R6 | R7 bundle swapped for R6 | .958 | .943 |
| HW | heliosli per-label table | .958 | .943 |
| a25 | r224 share .35 to .25 | .958 | .943 |
| c4 | four table entries | .958 | .943 |
| fr100 | c4 + Fracture share 1.0 | .958 | .943 |
| pf10 | c4 + PF OA share .1 | .958 | .944 |
| eoa10 | c4 + E-OA sym a .1 | .959 | .944 |
| eoa20 | sym a .2 | .959 | .944 |
| eoa30 | sym a .3 | .959 | .944 |
| pw20 | pair-weighted a .2 | .959 | .944 |
| pw30 | pair-weighted a .3 | .959 | .944 |
| pw25 | pair-weighted a .25 | .959 | .944 |
| **swide (pick A)** | pw25 + Lateral OA <- Medial Meniscus .05 + MCL <- Medial Meniscus .1 | **.960** | **.944** |
| cons | swide with a convex E-OA variant | .959 | .944 |
| **r6pw20 copy 3 (pick B)** | R6 (uniform share) + pw a .2 | **.959** | **.944** |
| 4 canary-hit reruns | S6fall stage failed in the hidden rerun | .916 | .901 to .902 |

Every E-OA entry landed at .944 private, one display step above c4, R6 and the other bases (.943): the gain held on private. To be fair, the PF OA share tweaks that lost on public (pf10 and three combinations) also reached .944.

## Refinements: shape, strength and small stacked tweaks

![dose](figs/dose_curve.png)

- **Strength.** gold58 macro gain vs a, symmetric: .1 +.00142, .2 +.00247, .3 +.0020, .4 +.0021, .5 +.0003, .6 -.0028. Pair-weighted: .2 +.00265, .3 +.00223, .35 +.00256, .4 +.0019, .5 +.00016, .6 -.00256. Both rise, flatten, then fall hard past ~.5 (in the pw curve Lateral OA starts to drop past .35). On the leaderboard the order was c4 < eoa10 < eoa20 < pw30 (best-id moves). Our public-implied estimates put the optimum near a .225 to .30, so we stopped there rather than chase .35 and up.
- **Shape.** Pair weighting puts most of the correction on the Lateral-PF pair, where reports and experts disagree most. pw30 beat eoa30 and became the best-id; pw25 was kept as the lower-risk dose.
- **Stack (swide).** Notation: X <- Y at a means $X = \mathrm{pct}(\mathrm{pct}(X) - a\,\mathrm{pct}(Y))$. A census of all 66 label pairs found nothing robust under a correct null. After recalibrating our rulers on the five E-OA reads, two small pairs on the degenerative side came back as lottery tickets: Lateral OA <- Medial Meniscus (a .05) and MCL <- Medial Meniscus (a .1). Our skeptic put the MCL one at an expectation of about zero, so we only allowed it inside an entry that already carried pw25. swide was our only .960, but its edge over pw25 is inside read noise, and both are .944 private.

## What did not work

| idea | what we saw |
|---|---|
| Our own trained models: 2.5D ConvNeXt-T fleets, knee-centred crops, OAI-trained seeds, R7-teacher students, S6 retrains | Solo .923 to .944 public. They helped weaker bases (our .951 on 10-09 was our fleet .4 + public Raptor .6; small in-band gains on the .954 / .956 blends) but stopped adding on the flip-TTA base: fleet, knee-centred leg (.957 / .941 vs .957 / .942 without it) and students all failed to move the best-id; our S6 retrains read .921 / .915 on gold58 vs nartaa's .931 |
| RSNA-only Raptor seeds (rented GPUs) | .955 vs .956 base |
| More TTA: r224 anatomical mirror, multi-scale, multi-FOV, shifted crops | mirror read negative; the rest killed offline (runtime risk, no precedent) |
| gold58-picked per-label share tweaks (PF OA .1, Fracture .9 / 1.0, combinations) | 5 of 5 below c4 on public (.957 to .958). On private they scored .943 to .944, so PF OA .1 at least was not harmful there; nothing on our side could have told us |
| S6fall .15 / .3, Raptor share .45, r224 share .15 | no best-id move |
| Blend operators: probability, logit, geometric, median, soft-OR | no gain on gold58; soft-OR looked good only on report-label OOF, a ruler that had over-rated our own models before |
| Cross-label stackers, co-occurrence stacking, decorrelating other groups (menisci, fluid, ACL, a global factor) | negative under nested selection on gold58; killed on OAI expert reads; 66-pair census null under a correct null |
| Site / scanner offsets, AdaBN, kNN smoothing, duplicate matching | zero or negative; one duplicate pair in 4,407 training studies |
| Last-day pushes: convex E-OA (cons), Medial OA <- Contusion / Fracture (MOACF) | cons .959 / .944, no better than swide; MOACF's hidden rerun errored |

## Engineering: pre-registration, gates and canaries

- **Pre-registration.** Every entry was written into one pre-registration file before its commit and submit: the single change, the hypothesis, the evidence, the read rule, and what it would mean for the final picks. The picks rule itself was fixed days before the last reads.
- **One change per entry, read by API best-id.** The API returns the team's best submission. A move to the new entry is a positive read even when the display is unchanged; a non-move is negative only if the base was still the best while it was pending (otherwise "unread").
- **Builders and gates.** Each kernel was built from a scored notebook by an exact, asserted cell diff with a receipt; an all-off control had to be byte-identical to the base. An offline gate replayed the notebook's own cells on gold58 against two numpy recomputes, and a commit gate (17 to 26 checks on the committed output, log and receipt) had to PASS before any submit. The 3 visible test rows cannot show most changes, so identity rests on shas and receipts.
- **Canaries.** If a stage fails or times out, one column is set to a constant (ACL for S6fall, for example), which costs ~.04 and shows as .916. A broken run can never pass as "same score".
- **Flaky hidden reruns.** In the last three days, 4 of 10 entries came back at .916 (the S6fall stage failed in the hidden rerun although every commit ran clean; dataset reads were 5 to 10 times slower that week), 2 reruns ended in "unhandled error", and 2 manual submits failed on DNS errors. Without the canaries those could have looked like ordinary .959 entries. We resubmitted pick B unchanged three times; copy 3 scored clean.
- **JPEG Lossless.** The pinned image could not decode JPEG Lossless DICOMs: Raptor zero-filled them silently and R7 dropped the series. Our decoder cell installs the three pylibjpeg wheels only if each sha256 matches PyPI, and outputs on uncompressed studies stay byte-identical. Train and visible test are uncompressed, so this was insurance only.

## Final submissions

- **A = swide** (57464326): c4 + pair-weighted E-OA a .25 + the two meniscus tweaks. Public .960, private .944.
- **B = r6pw20, copy 3** (57523232): R6 with a uniform Raptor share + pair-weighted E-OA a .2. Public .959, private .944. It hedges A on both open questions, the per-label table and the dose.

Both landed at .944, rank 246 / 6,425 (silver cut: rank 321). We slipped from ~157 public to 246 private: the private .944 band is dense, so a small difference in the unrounded score moves many places.

## Code and sources

Our code (kernel builders, gates, OAI and gold58 validation scripts, pre-registration log) is in a private repository for now. Public work we built on:

- **nartaa:** rsna-knee-publication-swa-weights-20261007 (Raptor r384 and r224), rsna-knee-s6fall-solo and rsna-knee-s6fall-weights (S6fall), and the Raptor training recipe posted in the forum.
- **heliosli:** rsna-knee-r6-frozen-fe055406 (R6, version 1), rsna-knee-compact-r7cnxfsss (R7), the S6 architecture, 0-957-latest-dataset-score-in-one-click, and blend-mine-in-private-lb-luck-to-all-upvoters (per-finding table).
- **varshithakolupuri:** rsna-knee-one-unused-checkpoint-lifts-the-0-956 (V224 base, the r224 checkpoint idea) and rsna-0-956-one-free-flip-68-ranks-up (flip TTA).
- **glorysing:** rsna-knee-offline-codecs-r09 (cp313 pylibjpeg wheels).
- **Label sets** used by the public models above: nartaa's Gemini labels, stevenleehans' v4_blend and riadmohamed42's JEV HYBRID.

**External data.** Raptor's author trained it with OAI. The host's ruling allowed publicly and freely available external data, including registration-only sets such as OAI. We used OAI MOAKS reads only for offline validation on our own machine, and only aggregate numbers left it. Some of our earlier, discarded entries used fleets we trained with OAI images under the same ruling; the final picks contain only public weights.

## Acknowledgements

Thanks to RSNA and the radiologists who read and adjudicated the test set, to Kaggle for hosting, and to heliosli, nartaa, varshithakolupuri and glorysing, whose public work is most of our score. Thanks to the OAI investigators and the MOAKS readers for the data that let us test an idea before trusting it. And thanks to our teammates **swapiy** and **kmrsuman**, who trained folds of our own ConvNeXt fleet on their Kaggle GPU quota; our first .951 was submitted from swapiy's account.
