Thanks to RSNA, the annotators and Kaggle for a hard, well-run competition, and to the authors whose public models carry most of our score (credits below). We finished **private .944, rank 246 / 6,425 (silver)**, public .960 (rank ~157 at close). This is not a top solution. We write it up because the one idea that moved us, and the way we checked it on an independent expert-labelled set before spending a submission, may help anyone who trains on report-derived labels and is scored on expert reads.

## Overview of the approach

*Figure 2 (media gallery above): the final pipeline, the public-weights blend (c4) plus E-OA.*

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
| Raptor r384 | nartaa's OAI-trained CoAtNet SWA weights (rsna-knee-publication-swa-weights-20261007), retrained from dreaddevelopment's Raptor CoAtNet/MIL recipe, with anatomical mirror TTA | used as published |
| Raptor r224 | the second checkpoint in the same dataset (224 px, 0.8 centre crop), first blended by varshithakolupuri at .35 of the Raptor slot | share .35 to .25 (one read) |
| R6 | heliosli's compact ConvNeXt-S bundle (version 1 of rsna-knee-r6-frozen-fe055406), 4 folds, its own Effusion / Baker's stackers | horizontal flip TTA forced on (a switch in heliosli's inference code that the bundle manifest leaves off; we followed jingshuifu's forced-flip commit, heliosli's own 0-9576 notebook forces it for R6, and varshithakolupuri published the same idea; on the older R7 bundle it lifted us ~214 places inside the .956 band) |
| per-label Raptor share | heliosli's per-finding table from "blend-mine-in-private-lb-luck-to-all-upvoters" | four entries changed (read together as entry c4); final: Medial OA .5, Lateral OA .2, PF OA .5, Synovitis .8, Baker's .5, Lateral Meniscus .4, Fracture .65, others .5 |
| S6fall | nartaa's retrain of heliosli's S6 architecture on blended LLM report labels | weight .2 (.15 and .3 did not move the best-id) |
| JPEG Lossless decoder cell | pylibjpeg cp313 wheels from glorysing/rsna-knee-offline-codecs-r09 | insurance, see Engineering |

Public path: S6F .957, R6 swap .958, then three best-id moves inside the .958 band (HW, a25, c4), then E-OA .959 and the swide stack .960.

## The key idea: report labels vs expert labels

Only 58 training studies carry expert labels (we call them gold58; gold74 adds 16 studies we labelled ourselves from the reports under the same convention, so it is not purely expert). Every model in our blend learned from labels that LLMs extracted from the radiology reports. The hidden test is graded by expert image review (the host: two readers plus an adjudicator, the same process as the 58 sample labels), and the rubric grades osteoarthritis **separately in each compartment** with a strict threshold.

Reports are not written to that rubric: they tend to describe degenerative change at the level of the knee, so report-trained models learn to fire on the three compartments together. On gold58, most of our Lateral OA errors were knees where the report-derived label was positive and the experts said no. We measured the coupling (pairwise, over Medial-Lateral, Medial-PF and Lateral-PF):

*Figure 1 (media gallery above): OA-compartment coupling of our model scores vs report labels vs expert labels.*

| source (Medial-Lateral / Medial-PF / Lateral-PF) | latent correlation, as in the figure | phi |
|---|---|---|
| our final blend's scores on gold58 (Spearman) | .85 / .82 / .81 | n/a |
| report-derived labels (tetrachoric) | .83 / .80 / .77 | .60 / .58 / .53 |
| expert labels, gold58 (tetrachoric) | .51 / .56 / .45 | .32 / .37 / .28 |
| gold74 (gold58 + 16 of our gold-convention reads) | .59 / .61 / .39 | .38 / .40 / .24 |
| OAI MOAKS expert cartilage reads (tetrachoric) | .12 / .13 / -.01 | .06 / .07 / -.01 |

The models co-fire far more than the expert labels justify. The fix subtracts a little of the other compartments' rank from each OA column:

$$\mathrm{out}_t = \mathrm{pct}\Big(\mathrm{pct}(f_t) - \frac{a}{2}\sum_{o \ne t} w_{t,o}\,\mathrm{pct}(f_o)\Big), \quad t \in \{\text{Medial, Lateral, PF}\}$$

- **Symmetric (eoa):** all $w = 1$.
- **Pair-weighted (pw):** $w$ proportional to each pair's excess coupling (our blend's score correlation minus the gold74 label correlation: Lateral-PF .42, Medial-Lateral .26, Medial-PF .21), scaled to mean 1: w(Lateral, PF) 1.416, w(Medial, Lateral) .876, w(Medial, PF) .708.

It costs zero runtime: nine lines in the last cell, and the other nine columns are untouched.

**When should it help?** In a simple Gaussian model, let $\rho$ be the models' OA-OA score correlation and $q$ the cross-predictiveness of the other compartments' scores for label $k$, $q = \mathrm{corr}(f_{\text{other}}, Y_k) / \mathrm{corr}(f_k, Y_k)$. Decorrelation raises AUC only when $q < \rho$, i.e. when the model couples the trio more than the labels do. Ensembling averages away member noise, which raises $\rho$ but not $q$, so the full blend gains more than any member (on gold58 it had the largest gain of the 12 prediction sets we held, +.0096 OA-mean AUC at a .2).

**Mechanism check:** the same transform *loses* on report-labelled out-of-fold predictions (9 of 9 reader x member cells negative). The sign flips with the label regime, exactly as the mechanism predicts.

## Validating before submitting: an independent expert ruler

gold58 liked E-OA (+.0014 macro at a .1, +.0025 at a .2), but 58 studies is a weak ruler: two skeptic passes showed the gain rested on ~20 pair swaps, and five gold58-picked weight tweaks had just lost on the leaderboard the same day. We needed an expert ruler that had never been used for selection.

**OAI.** Our university machine holds MOAKS cartilage reads by musculoskeletal radiologists for 2,399 baseline knees, plus predictions on those knees from three of our own **OAI-free, report-trained** models (trained on RSNA only, for an earlier experiment). Before running anything we fixed the rule: SUPPORT needs an OA-mean AUC delta > 0 with paired-bootstrap P(>0) >= .90, a positive delta on at least 2 of 3 cartilage proxies (full-thickness, area, both), and a negative a = -.1 control. We worked only with aggregate numbers from that machine, and no OAI rows are stored anywhere else.

*Figure 3 (media gallery above): E-OA on 2,399 OAI knees read by expert radiologists, three OAI-free models.*

| OAI-free model (OA-mean AUC delta, full-thickness proxy) | a .1 | a .2 | control a -.1 |
|---|---|---|---|
| teacher (5 folds) | +.0062 | +.0115 | -.0066 |
| replication 1 (other seed and label set) | +.0050 | +.0086 | -.0058 |
| replication 2 (288 px) | +.0053 | +.0093 | -.0061 |

All rows: P(>0) 1.000, 3 of 3 proxies positive, 5 of 5 folds positive (teacher).

**Transfer.** OAI expert labels are almost uncoupled (phi ~.05), RSNA expert labels are not (~.3), and real disease coupling eats into the gain. Importance-reweighting OAI knees to gold74's coupling shrank the a .2 gain 5 to 7 times (to +.0004 to +.0006 macro); adding our blend's own, higher OA coupling, the expected hidden macro gain was about +.0005 at a .1 (P(>0) ~.8) and +.0007 at a .2 (~.7). That cleared our pre-registered bar, so we submitted a .1 and a .2.

**What the rulers got wrong.** They got the sign right at the doses we decided on and the size wrong, in both directions. gold58 and raw OAI *overstated* the public gain about 2.4 and 3 times. Our deliberately conservative coupling-matched transfer, the one we used to decide, *understated* it about 1.6 to 3 times at a .1 and .2, turned negative at a .3 and put the optimum at a <= .1, while the leaderboard favoured a ~.2 to .3. Only after five reads could we fit one consistent factor (reading the OAI curve at c4's own coupling gap). Our lesson: an independent ruler tells you the sign; the dose came from the leaderboard.

## Leaderboard results

*Figure 4 (media gallery above): public and private scores along our entry sequence.*

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
| **r6pw20 copy 3 (pick B)** | R6 (uniform share, r224 share .35) + pw a .2 | **.959** | **.944** |
| 4 canary-hit reruns | S6fall stage failed in the hidden rerun | .916 | .901 to .902 |

Every E-OA entry that ran clean landed at .944 private, one display step above c4, R6 and the other bases (.943): the gain held on private. To be fair, the PF OA share tweaks that lost on public (pf10 and three combinations) also reached .944.

## Refinements: shape, strength and small stacked tweaks

*Figure 5 (media gallery above): E-OA strength and shape on the 58 hand-labelled studies (info only).*

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
- **Builders and gates.** Each kernel was built from a scored notebook by an exact, asserted cell diff with a receipt; an all-off control had to be byte-identical to the base. An offline gate replayed the notebook's own cells on gold58 against two numpy recomputes, and a commit gate (16 to 31 checks on the committed output, log and receipt) had to PASS before any submit. The 3 visible test rows cannot show most changes, so identity rests on shas and receipts.
- **Canaries.** If a stage fails or times out, one column is set to a constant (ACL for S6fall, for example), which costs ~.04 and shows as .916. A broken run can never pass as "same score".
- **Flaky hidden reruns.** On the last two submission days, 4 of 10 entries came back at .916 (the S6fall stage failed in the hidden rerun although every commit ran clean; our commits read datasets 5 to 10 times slower on 10-21), 2 reruns ended in "unhandled error", and 2 manual submits failed on DNS errors. Without the canaries those could have looked like ordinary .959 entries. We resubmitted pick B unchanged three times; copy 3 scored clean.
- **JPEG Lossless.** The pinned image could not decode JPEG Lossless DICOMs: Raptor zero-filled them silently and R7 dropped the series. Our decoder cell installs the three pylibjpeg wheels only if each sha256 matches PyPI, and outputs on uncompressed studies stay byte-identical. Train and visible test are uncompressed, so this was insurance only.

## Final submissions

- **A = swide** (57464326): c4 + pair-weighted E-OA a .25 + the two meniscus tweaks. Public .960, private .944.
- **B = r6pw20, copy 3** (57523232): R6 with a uniform Raptor share and the original r224 share (.35) + pair-weighted E-OA a .2. Public .959, private .944. It hedges A on three choices: the per-label table, the r224 share and the dose.

Both landed at .944, rank 246 / 6,425 (silver cut: rank 321). We slipped from ~157 public to 246 private: the private .944 band is dense, so a small difference in the unrounded score moves many places.

## Code and sources

Final notebook (pick A, swide): https://www.kaggle.com/code/eeshsaxena/rsna-sub-v224r6c4pw25lomm05mcl10j . All our code (kernel builders, gates, OAI and gold58 validation scripts, the pre-registration log and daily notes): https://github.com/eeshsaxena/rsna-knee-abnormality-detection (data, weights and OAI-derived files are not included). Public work we built on:

- **dreaddevelopment:** the original Raptor CoAtNet/MIL inference recipe and baseline weights (raptor-knee-widedense), from which nartaa's checkpoints were retrained.
- **nartaa:** rsna-knee-publication-swa-weights-20261007 (Raptor r384 and r224), the inference notebooks rsna-knee-0949-anatomical-mirror and rsna-knee-0945-efficient-224crop, rsna-knee-s6fall-solo and rsna-knee-s6fall-weights (S6fall), and the training recipe in the forum post "Reaching 0.949 with one CoAtNet".
- **heliosli:** rsna-knee-r6-frozen-fe055406 (R6, version 1; the dataset's current version holds R7, so we attached a private copy of version 1), rsna-orthofoundation-fold0-epoch7-20261004 (the decode / geometry package and the anatomy.pt knee-anchor model that R6 and S6fall use), rsna-knee-compact-r7cnxfsss (R7), the S6 architecture, 0-957-latest-dataset-score-in-one-click and 0-9576-latest-public-dataset-score-in-one-click (the R7 and R6 recipes), and blend-mine-in-private-lb-luck-to-all-upvoters (per-finding table).
- **jingshuifu:** rsna-knee-flip-tta (flip TTA forced on an unchanged bundle).
- **varshithakolupuri:** rsna-knee-one-unused-checkpoint-lifts-the-0-956 (V224 base, the r224 checkpoint idea) and rsna-0-956-one-free-flip-68-ranks-up (the same flip idea, published independently).
- **glorysing:** rsna-knee-offline-codecs-r09 (cp313 pylibjpeg wheels).
- **Label sets** used by the public models above: nartaa's Gemini labels, stevenleehans' v4_blend (rsna-knee-llm-report-labels), pilkwang's rsna-knee-llm-labels, riadmohamed42's JEV HYBRID, and heliosli's own LLM-graded report states (R6, R7).

**External data.** The host allowed external data that is free and needs only registration and acceptance of a licence (it gave OAI a conditional yes in September and confirmed the rule on 10-14), and asked teams to list every external dataset and the source data of any public weights. Ours:

- **OAI (Osteoarthritis Initiative):** nartaa's Raptor checkpoints (r384 and r224) were trained with OAI images. We used OAI MOAKS reads ourselves only for the offline validation above, on our own machine, working only with aggregate numbers from it. Some earlier, discarded entries also used a ConvNeXt fleet we trained with OAI images.
- **SKM-TEA:** heliosli's anatomy.pt knee-anchor model (in rsna-orthofoundation-fold0-epoch7-20261004) places the R6 and S6fall crops, so both final picks depend on it.
- **fastMRI:** heliosli's R7 bundle used a fastMRI auxiliary task in training. R7 is in earlier entries (S6F and before), not in the final picks.
- **RadImageNet:** some of our earliest, discarded entries used a RadImageNet-pretrained ResNet-50 member.

The final picks contain only public weights (Raptor, R6, S6fall) plus our own post-processing.

## Acknowledgements

Thanks to RSNA and the radiologists who read and adjudicated the test set, to Kaggle for hosting, and to heliosli, nartaa, dreaddevelopment, jingshuifu, varshithakolupuri and glorysing, whose public work is most of our score. Thanks to the OAI investigators and the MOAKS readers for the data that let us test an idea before trusting it. And thanks to our teammates **swapiy**, **kmrsuman** and **akkigupta2005** (Akanksha Gupta); swapiy and kmrsuman trained folds of our own ConvNeXt fleet on their Kaggle GPU quota, and our first .951 was submitted from swapiy's account.

Data and/or research tools used in the preparation of this manuscript were obtained and analyzed from the controlled access datasets distributed from the Osteoarthritis Initiative (OAI), a data repository housed within the NIMH Data Archive (NDA). OAI is a collaborative informatics system created by the National Institute of Mental Health and the National Institute of Arthritis, Musculoskeletal and Skin Diseases (NIAMS) to provide a worldwide resource to quicken the pace of biomarker identification, scientific investigation and OA drug development. Dataset identifier(s): NDA collection 2343.
