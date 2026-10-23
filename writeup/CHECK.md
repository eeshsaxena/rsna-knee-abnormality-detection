# CHECK.md: fact and safety check of writeup_draft.md and figs/ (10-23)

Scope: every number in the draft text, tables and the five figures was checked against `session_2026-10-23_scores.json`,
HANDOFF.md and the session notes named below. Credits were checked against the c4 notebook's own credit cell
(`session_2026-10-18/a18/kernel_v224r6a25c4j`, cell 1), its `kernel-metadata.json` dataset list, HANDOFF.md and the
10-14 to 10-17 forum notes. Safety: grep of the draft, FIGS.md and both scripts for IPs, usernames, credentials, OAI id
patterns and non-ASCII characters; visual read of all five PNGs. Nothing was posted, uploaded or pushed.

**Verdict.** The scores, ranks, the OAI table, the dose curve and the leaderboard table are correct. Fix before posting:
(1) credits: dreaddevelopment (the original Raptor author) is missing, the flip-TTA idea is credited to the wrong person,
and heliosli's orthofoundation package, nartaa's two inference notebooks and pilkwang's labels are missing;
(2) external-data disclosure: SKM-TEA (anatomy.pt in R6 and S6fall) is missing, "Raptor's author" is wrong, and the
host asked for every external dataset to be listed;
(3) the OAI / NDA acknowledgement is missing;
(4) the coupling table does not match its figure and has one unsourced range (.69);
(5) the pw weights are described with the wrong gap;
(6) pick B has the r224 share wrong;
(7) four smaller number or wording errors (gate counts, "three days", ruler factors, "every E-OA entry");
(8) the images need uploading to Kaggle.
No em dashes anywhere (every file is pure ASCII). No OAI rows, ids or per-knee values. No report text, study-level
labels, credentials, box IP or box username in the draft or figures.

---

## A. Required fixes in writeup_draft.md (exact old -> new)

### A1. Raptor credit: dreaddevelopment is the original Raptor author (line 25)
Source: c4 notebook cell 1 ("Built on dreaddevelopment's Raptor CoAtNet/MIL inference recipe and public baseline
weights ... These are our retrained checkpoints"); session_2026-10-15/silver/FORUM_DIGEST_10-15.md l.103 ("Dread
Development ..., Raptor's original author").
OLD:
```
| Raptor r384 | nartaa's OAI-trained CoAtNet SWA weights (rsna-knee-publication-swa-weights-20261007), anatomical mirror TTA | used as published |
```
NEW:
```
| Raptor r384 | nartaa's OAI-trained CoAtNet SWA weights (rsna-knee-publication-swa-weights-20261007), retrained from dreaddevelopment's Raptor CoAtNet/MIL recipe, with anatomical mirror TTA | used as published |
```

### A2. Flip TTA credit (line 27)
Source: session_2026-10-14/RESEARCH_960.md l.117-135 (the flip switch is in heliosli's compact_infer and is off in the
R7 manifest; we followed jingshuifu's forced-flip commit, `jingshuifu/rsna-knee-flip-tta`); HANDOFF l.511 (heliosli's own
0-9576 notebook = R6 + forced flip, which our R6 entry was gated against); varshithakolupuri's flip notebook appears in
session_2026-10-15/watch/WATCH_LOG.md l.210 with run time 2026-10-14 10:52, after our plan was written.
OLD:
```
horizontal flip TTA forced on (varshithakolupuri's "free flip" idea; on the older R7 bundle it lifted us ~214 places inside the .956 band)
```
NEW:
```
horizontal flip TTA forced on (a switch in heliosli's inference code that the bundle manifest leaves off; we followed jingshuifu's forced-flip commit, heliosli's own 0-9576 notebook forces it for R6, and varshithakolupuri published the same idea; on the older R7 bundle it lifted us ~214 places inside the .956 band)
```
and in the sources list (line 154):
OLD:
```
and rsna-0-956-one-free-flip-68-ranks-up (flip TTA).
```
NEW:
```
and rsna-0-956-one-free-flip-68-ranks-up (the same flip idea, published independently).
```

### A3. Sources list: missing authors and items (lines 152-156)
Sources: c4 notebook cell 1 and kernel-metadata.json (datasets attached to every final kernel include
`heliosli/rsna-orthofoundation-fold0-epoch7-20261004`); session_2026-10-13/R7_PORT_PLAN.md l.32-33 (that dataset holds the
decode / geometry package and `anatomy.pt`, the SKM-TEA knee-anchor V-Net); forum topic 746792 (author nartaa, "Reaching 0.949
with one CoAtNet: labels, native crops, and anatomical mirroring"); label_regime.md sec 1 (heliosli's own Opus-graded report
states). The slugs below all appear in HANDOFF.md or the session notes.

OLD:
```
- **nartaa:** rsna-knee-publication-swa-weights-20261007 (Raptor r384 and r224), rsna-knee-s6fall-solo and rsna-knee-s6fall-weights (S6fall), and the Raptor training recipe posted in the forum.
- **heliosli:** rsna-knee-r6-frozen-fe055406 (R6, version 1), rsna-knee-compact-r7cnxfsss (R7), the S6 architecture, 0-957-latest-dataset-score-in-one-click, and blend-mine-in-private-lb-luck-to-all-upvoters (per-finding table).
```
NEW:
```
- **dreaddevelopment:** the original Raptor CoAtNet/MIL inference recipe and baseline weights (raptor-knee-widedense), from which nartaa's checkpoints were retrained.
- **nartaa:** rsna-knee-publication-swa-weights-20261007 (Raptor r384 and r224), the inference notebooks rsna-knee-0949-anatomical-mirror and rsna-knee-0945-efficient-224crop, rsna-knee-s6fall-solo and rsna-knee-s6fall-weights (S6fall), and the training recipe in the forum post "Reaching 0.949 with one CoAtNet".
- **heliosli:** rsna-knee-r6-frozen-fe055406 (R6, version 1; the dataset's current version holds R7, so we attached a private copy of version 1), rsna-orthofoundation-fold0-epoch7-20261004 (the decode / geometry package and the anatomy.pt knee-anchor model that R6 and S6fall use), rsna-knee-compact-r7cnxfsss (R7), the S6 architecture, 0-957-latest-dataset-score-in-one-click and 0-9576-latest-public-dataset-score-in-one-click (the R7 and R6 recipes), and blend-mine-in-private-lb-luck-to-all-upvoters (per-finding table).
- **jingshuifu:** rsna-knee-flip-tta (flip TTA forced on an unchanged bundle).
```
OLD:
```
- **Label sets** used by the public models above: nartaa's Gemini labels, stevenleehans' v4_blend and riadmohamed42's JEV HYBRID.
```
NEW:
```
- **Label sets** used by the public models above: nartaa's Gemini labels, stevenleehans' v4_blend (rsna-knee-llm-report-labels), pilkwang's rsna-knee-llm-labels, riadmohamed42's JEV HYBRID, and heliosli's own LLM-graded report states (R6, R7).
```

### A4. External-data disclosure (line 158)
Sources: forum notes session_2026-10-15/silver/FORUM_DIGEST_10-15.md sec 1 and sec 3 (host 09-21 conditional yes for OAI,
thread 741819: no institutional sign-off, no fee; host 10-14, thread 743416: free data that needs only registration and
accepting a licence qualifies, institutional agreements do not, and participants should disclose every dataset used; thread
744056: public weights are award-eligible if the weights and their source datasets are listed);
session_2026-10-17/research_1hr/forum.md l.18-26 (MEASURED: both picks carry OAI via nartaa's weights and SKM-TEA via
anatomy.pt; fastMRI only through R7, which is in no pick); session_2026-10-10/K14_RESULTS.md (our OAI-trained K14 fleet,
used in e.g. V224-FF20); HANDOFF l.1938 and l.2004 (early entries used a RadImageNet-pretrained ResNet-50 member).
"Raptor's author" is wrong: dreaddevelopment wrote Raptor; nartaa retrained it with OAI.
OLD:
```
**External data.** Raptor's author trained it with OAI. The host's ruling allowed publicly and freely available external data, including registration-only sets such as OAI. We used OAI MOAKS reads only for offline validation on our own machine, and only aggregate numbers left it. Some of our earlier, discarded entries used fleets we trained with OAI images under the same ruling; the final picks contain only public weights.
```
NEW:
```
**External data.** The host allowed external data that is free and needs only registration and acceptance of a licence (it gave OAI a conditional yes in September and confirmed the rule on 10-14), and asked teams to list every external dataset and the source data of any public weights. Ours:
- **OAI (Osteoarthritis Initiative):** nartaa's Raptor checkpoints (r384 and r224) were trained with OAI images. We used OAI MOAKS reads ourselves only for the offline validation above, on our own machine, working only with aggregate numbers from it. Some earlier, discarded entries also used a ConvNeXt fleet we trained with OAI images.
- **SKM-TEA:** heliosli's anatomy.pt knee-anchor model (in rsna-orthofoundation-fold0-epoch7-20261004) places the R6 and S6fall crops, so both final picks depend on it.
- **fastMRI:** heliosli's R7 bundle used a fastMRI auxiliary task in training. R7 is in earlier entries (S6F and before), not in the final picks.
- **RadImageNet:** some of our earliest, discarded entries used a RadImageNet-pretrained ResNet-50 member.
The final picks contain only public weights (Raptor, R6, S6fall) plus our own post-processing.
```
(Before posting, Eesh should confirm the RadImageNet line is wanted; it is accurate per HANDOFF l.1938 / l.2004, entries
56753683 and earlier.)

### A5. OAI / NDA acknowledgement is missing (Acknowledgements, line 162)
The writeup reports results computed on OAI data (the OAI table, the OAI bars in coupling.png, oai_validation.png).
The NDA data use terms require the standard OAI acknowledgement in any publication; nartaa's notebook (copied in our c4
cell 1) carries it. Add after the Acknowledgements paragraph:
NEW (append):
```
Data and/or research tools used in the preparation of this manuscript were obtained and analyzed from the controlled access datasets distributed from the Osteoarthritis Initiative (OAI), a data repository housed within the NIMH Data Archive (NDA). OAI is a collaborative informatics system created by the National Institute of Mental Health and the National Institute of Arthritis, Musculoskeletal and Skin Diseases (NIAMS) to provide a worldwide resource to quicken the pace of biomarker identification, scientific investigation and OA drug development. Dataset identifier(s): NDA collection 2343.
```
Eesh to confirm the identifier against nda.nih.gov/nda/manuscript-preparation (NDA may ask for an NDA Study DOI instead of
the collection id, and may ask that the publication be reported to NDA). Do NOT add his DAR number or NDA username.

### A6. "Only aggregate numbers left that machine" is not literally true (line 66)
HANDOFF l.147 (10-20 07:00 block) records a governance slip: a few OAI ids and grades were printed into session transcripts
by `head` on 10-19/20; no local file holds OAI rows. Use wording that is true and reveals nothing:
OLD:
```
Only aggregate numbers left that machine.
```
NEW:
```
We worked only with aggregate numbers from that machine, and no OAI rows are stored anywhere else.
```
(The second occurrence, in the external-data paragraph, is replaced by A4.)

### A7. Coupling table: mixes measures with the figure and has an unsourced range (lines 42-47)
The figure shows Spearman (scores) and tetrachoric (labels); the table shows phi, so a reader sees two different sets of
numbers for the same pairs (reports .77 to .83 in the figure vs .53 to .61 in the table; OAI -.01 to .13 vs ~.05). The model
range ".69 to .85" has no source: c4 on gold58 is .81 to .85 (coupling_map.md sec 2, rows 1/2/4), the members on gold58 run
.73 to .89 (ex/local_coupling.log), the OAI-free models on OAI .66 to .81 (g1 logs). Values below: coupling_map.md sec 2
(Spearman, tetrachoric, OAI phi), ex/local_coupling.log l.25/26/30 (report phi), a20/skeptic_eoa.md sec 1 (gold58 / gold74 phi).
OLD:
```
| source | OA-OA coupling |
|---|---|
| our models' predictions (rank correlation) | .69 to .85 |
| report-derived labels (phi) | .53 to .61 |
| expert labels, gold58 / gold74 (phi) | .24 to .40 |
| OAI MOAKS expert cartilage reads (phi) | ~.05 |
```
NEW:
```
| source (Medial-Lateral / Medial-PF / Lateral-PF) | latent correlation, as in the figure | phi |
|---|---|---|
| our final blend's scores on gold58 (Spearman) | .85 / .82 / .81 | n/a |
| report-derived labels (tetrachoric) | .83 / .80 / .77 | .60 / .58 / .53 |
| expert labels, gold58 (tetrachoric) | .51 / .56 / .45 | .32 / .37 / .28 |
| gold74 (gold58 + 16 of our gold-convention reads) | .59 / .61 / .39 | .38 / .40 / .24 |
| OAI MOAKS expert cartilage reads (tetrachoric) | .12 / .13 / -.01 | .06 / .07 / -.01 |
```

### A8. Define gold58 / gold74 at first use (line 36)
gold74 is not purely expert: its 16 extra rows are our own report reads under the gold convention (label_regime.md sec 1).
OLD:
```
Only 58 training studies carry expert labels.
```
NEW:
```
Only 58 training studies carry expert labels (we call them gold58; gold74 adds 16 studies we labelled ourselves from the reports under the same convention, so it is not purely expert).
```

### A9. Pair weights come from the MODEL-minus-expert gap, not report-minus-expert (line 54)
Source: session_2026-10-19/ex/cand_eoa_curve.md l.30-31 ("F3 pair-weighted by the gold74 excess gap (coupling map: LOA-PF .42,
MOA-LOA .26, MOA-PF .21; weight = gap / mean gap = 1.416 / .876 / .708)"); coupling_map.md defines gap gold74 = c4 Spearman
minus gold74 tetrachoric. Check: .42 / .2967 = 1.416, .26 / .2967 = .876, .21 / .2967 = .708.
OLD:
```
- **Pair-weighted (pw):** $w$ proportional to the report-minus-expert coupling gap on gold74, scaled to mean 1: w(Lateral, PF) 1.416, w(Medial, Lateral) .876, w(Medial, PF) .708.
```
NEW:
```
- **Pair-weighted (pw):** $w$ proportional to each pair's excess coupling (our blend's score correlation minus the gold74 label correlation: Lateral-PF .42, Medial-Lateral .26, Medial-PF .21), scaled to mean 1: w(Lateral, PF) 1.416, w(Medial, Lateral) .876, w(Medial, PF) .708.
```

### A10. Transfer sentence conflates two numbers (line 78)
skeptic_eoa.md sec 0 / sec 1: reweighting to gold74 coupling cut the a .2 gain 5 to 7 times, to +.0004 to +.0006 macro;
the +.0005 / +.0007 (P ~.8 / ~.7) figures are the sec 4 hidden-set model that also uses c4's own coupling (rho .816).
OLD:
```
Importance-reweighting OAI knees to gold74's coupling shrank the gain 5 to 7 times, to an expected hidden macro gain of about +.0005 at a .1 (P(>0) ~.8) and +.0007 at a .2 (~.7).
```
NEW:
```
Importance-reweighting OAI knees to gold74's coupling shrank the a .2 gain 5 to 7 times (to +.0004 to +.0006 macro); adding our blend's own, higher OA coupling, the expected hidden macro gain was about +.0005 at a .1 (P(>0) ~.8) and +.0007 at a .2 (~.7).
```

### A11. Ruler factors (line 80)
Source: session_2026-10-20/night/recal.md sec 0 table (public / offline over eoa10, eoa20, eoa30, pw20, pw30): gold58
.39 to .51 ("OVERstates ~2.4x"); raw OAI .25 to .38 ("~3x"); OAI coupling-matched 1.6 / 3.2 / negative / 2.5 / negative,
i.e. understated 1.6 to 3.2 times where positive and the wrong sign at a .3. "2 to 5 times" overstates the measured range,
and "got the sign right" is false for the coupling-matched ruler at a .3.
OLD:
```
They got the sign right and the size wrong, in both directions. Raw OAI and gold58 *overstated* the public gain about 2.5 to 3 times. Our deliberately conservative coupling-matched transfer, the one we used to decide, *understated* it roughly 2 to 5 times and put the optimum at a <= .1, while the leaderboard favoured a ~.2 to .3.
```
NEW:
```
They got the sign right at the doses we decided on and the size wrong, in both directions. gold58 and raw OAI *overstated* the public gain about 2.4 and 3 times. Our deliberately conservative coupling-matched transfer, the one we used to decide, *understated* it about 1.6 to 3 times at a .1 and .2, turned negative at a .3 and put the optimum at a <= .1, while the leaderboard favoured a ~.2 to .3.
```

### A12. "Every E-OA entry" includes the canary-hit ones (line 108)
r6pw20 copy 1 / 2, spl and salt are E-OA entries that scored .901 to .902 (scores json 57463719, 57464296, 57464911, 57522767).
OLD:
```
Every E-OA entry landed at .944 private,
```
NEW:
```
Every E-OA entry that ran clean landed at .944 private,
```

### A13. Pick B uses r224 share .35, not .25 (lines 105 and 144)
Source: session_2026-10-20/val/kernels/kernel_v224r6flips6fpw20j (A224 = 0.35, W_RAPTOR_C all .5, A_OA = 0.2); the same in
session_2026-10-15/r6entry/kernel_v224r6flips6fj; VAL_1020.md gate command `--a224 0.35`.
OLD:
```
| **r6pw20 copy 3 (pick B)** | R6 (uniform share) + pw a .2 | **.959** | **.944** |
```
NEW:
```
| **r6pw20 copy 3 (pick B)** | R6 (uniform share, r224 share .35) + pw a .2 | **.959** | **.944** |
```
OLD:
```
- **B = r6pw20, copy 3** (57523232): R6 with a uniform Raptor share + pair-weighted E-OA a .2. Public .959, private .944. It hedges A on both open questions, the per-label table and the dose.
```
NEW:
```
- **B = r6pw20, copy 3** (57523232): R6 with a uniform Raptor share and the original r224 share (.35) + pair-weighted E-OA a .2. Public .959, private .944. It hedges A on three choices: the per-label table, the r224 share and the dose.
```

### A14. Commit-gate check counts (line 136)
HANDOFF: S6F V224S6F_CHECK 31/31, R6 R6_CHECK 19/19, HW 17/17, A16 / A19 PROBE_CHECK 16/16, E-OA 17/17, stacks 26/26
(stack/build.md sec 0), CONS 27/27, MOACF 29/29.
OLD:
```
a commit gate (17 to 26 checks on the committed output, log and receipt)
```
NEW:
```
a commit gate (16 to 31 checks on the committed output, log and receipt)
```

### A15. "Last three days" and "that week" (line 138)
The 4 canaries and 2 errors are among the 10 entries of the 10-21 and 10-22 UTC days (A22: 5, final day: 5); counting 10-20
(A21, 3 clean entries) gives 13. CANARY_1021.md: "Our 10-21 commits read Datasets 5-10x slower".
OLD:
```
In the last three days, 4 of 10 entries came back at .916 (the S6fall stage failed in the hidden rerun although every commit ran clean; dataset reads were 5 to 10 times slower that week),
```
NEW:
```
On the last two submission days, 4 of 10 entries came back at .916 (the S6fall stage failed in the hidden rerun although every commit ran clean; our commits read datasets 5 to 10 times slower on 10-21),
```

### A16. Images will not render on Kaggle from relative paths (lines 9, 40, 68, 84, 112)
All five files exist locally (figs/pipeline.png, coupling.png, oai_validation.png, lb_progress.png, dose_curve.png), but a
Kaggle writeup cannot resolve `figs/...`. Upload the five PNGs in the writeup editor and replace each `(figs/<name>.png)` with
the uploaded image URL. Uploading is Eesh's step.

---

## B. Figure fixes (make_figs.py; regenerate with py -3.12, writes only inside writeup/figs)

### B1. coupling.png subtitle attributes the model's .81 to .85 to the reports (make_figs.py l.174-175)
OLD:
```
'Expert graders score each compartment on its own; the rubric does too. The model inherits the\n'
'report coupling (.81 to .85) that expert labels do not have (.39 to .61 on RSNA, \u2212.01 to .13 on OAI).'
```
NEW:
```
'The rubric scores each compartment on its own. Our blend couples them (.81 to .85) about as much as the\n'
'report labels do (.77 to .83); expert labels couple them far less (.39 to .61 on RSNA, \u2212.01 to .13 on OAI).'
```

### B2. pipeline.png pick B line (make_figs.py l.130)
OLD:
```
'Pick B, r6pw20: the R6 entry (uniform blend) + pw a .2',
```
NEW:
```
'Pick B, r6pw20: the R6 entry (uniform blend, r224 share .35) + pw a .2',
```

### B3. pipeline.png decoder note (make_figs.py l.73), recommended
Cell 2 of the c4 notebook: Raptor zero-fills a JPEG Lossless slice; R7, our fleet and S6fall drop the whole series.
OLD:
```
'without it the stock image zero-fills or drops JPEG-lossless slices'
```
NEW:
```
'without it Raptor zero-fills JPEG-lossless slices and the other stages drop the whole series'
```

---

## C. Recommended (accuracy or completeness, not errors)

- **Line 19, credit for the recipe.** "Steps 1 to 3 are public weights inside our own plumbing." -> "Steps 1 to 3 are public
  weights and public recipes (varshithakolupuri's V224 notebook on heliosli's and nartaa's work) with a few re-tuned shares;
  the canaries, the decoder cell and the gates are ours."
- **Line 38, unsourced characterization.** "Reports are not written to that rubric: they tend to describe degenerative change at
  the level of the knee, so report-trained models learn to fire on the three compartments together." -> "Report-derived labels
  do not follow that rubric: on the same 58 studies they call the three compartments together far more often than the experts
  do, so report-trained models learn to fire on the three together." (The facet mechanism was never confirmed by the host:
  TRICK_1019 sec 2.4 and sec 7 B.)
- **Line 58.** "more than the labels do" -> "more than the labels justify" (skeptic_eoa.md sec 1 wording; q is cross-predictiveness,
  not label coupling).
- **Line 86.** "our own fleets (.945 to .951)" -> "our own fleets (.943 to .951)" (K2 / K3 blends .943, scores json 56816039, 56830644).
- **Line 86, early public stacks are uncredited.** The ".940 to .943" reproductions were of mattiaangeli's
  bend-the-knee-to-the-dinosaurs stack, maverickss26's .942 notebook, yamadan96's d4-public0946 and a jiweiliu fast base
  (HANDOFF l.1932-1972, scores json 56516549, 56690562, 56750056). Add them to the sources list as "earlier, discarded entries";
  the maverickss26 and jiweiliu slugs are not in the notes, so check them on Kaggle first.
- **Line 124.** The r224 anatomical mirror we tried was also public (rjings, rsna-r224-anatomical-mirror-20261014, HANDOFF l.638);
  credit if kept.
- **Line 139.** "Raptor zero-filled them silently and R7 dropped the series." -> "Raptor zero-filled those slices silently, and R7
  and S6fall dropped the whole series." (c4 cell 2 header).
- **Line 146.** Add the band size: the private .944 band is ranks 212 to 311 (100 teams), and our public .960 band was ranks 108
  to 158 (lbprobe/api_lb_1023_private.json).
- **Line 134-135, "One change per entry".** c4 changed four table entries and swide stacks three terms; say "one pre-registered
  change per entry (a change can be a declared bundle, e.g. c4's four table entries)".
- **Line 134, picks rule.** The A rule (API best-id) was fixed 10-17 (A15); the B rule was amended on 10-21 and 10-22 (A22, A23),
  each time before the reads it used. "fixed days before the last reads" is true for A only.
- **Acknowledgements.** Add dreaddevelopment and jingshuifu to "heliosli, nartaa, varshithakolupuri and glorysing"; optionally
  timm (CoAtNet backbone) as in the c4 credit cell.
- **Captions.** The images have only one-word alt text; FIGS.md has a caption per figure. Add a one-line caption under each.
- **Kaggle markdown.** Tables and `$...$` / `$$...$$` are fine in Kaggle markdown, but lines 51 and 58 carry several `_` inside
  math (`\mathrm{out}_t`, `w_{t,o}`, `f_{\text{other}}`, `Y_k`). Check the preview; if italics appear, escape them as `\_`.
  The `# 246th Place ...` H1 repeats the writeup title field; it can go.

## D. Decisions for Eesh (not errors)

- **Team.** The final leaderboard lists four members on team "Eesh saxena": Akanksha Gupta (akkigupta2005), Eesh saxena,
  SwapIY (swapiy), SUMAN KUMAR (kmrsuman) (lbprobe/api_lb_1023_private.json). The draft thanks swapiy and kmrsuman only.
  Decide whether to name akkigupta2005. Kaggle display names only; no other teammate info is in the draft.
- **RadImageNet line in A4** (keep or drop).
- **NDA identifier in A5** (collection id or a Study DOI).
- **Posting** the writeup and uploading the images are his steps.

## E. Verified correct (no change)

- Result: private .944, rank 246 / 6,425, silver, silver cut rank 321, public .960 rank 157 (api_lb_1023_private.json; HANDOFF l.2).
- Every public / private pair in the leaderboard table, the 4 canary reruns (.916 / .901 to .902), pf10 and the three PF
  combinations at .944, fr100 .943, S6F .942 (scores json).
- Overview: .75 / .25 Raptor mix, .8 / .2 S6fall, the c4 table values (HANDOFF l.206; c4 cell 14, cell 18), nine lines in cell 18,
  +.001 on both boards, "four entries changed" (HW table vs c4 table, build_hw.py header).
- ~214 places from flip (HANDOFF l.655); r224 at .35 first by varshithakolupuri, 224 px 0.8 centre crop (c4 cell 0);
  S6fall .15 / .3 non-moves (HANDOFF l.310, l.329).
- 58 labelled train studies, two readers plus an adjudicator, compartment-wise rubric (label_regime.md sec 1; skeptic_eoa.md sec 3).
- Lateral OA error pattern on gold58 (label_regime.md sec 2 note); 9 of 9 report-OOF cells negative; c4 largest of 12 sets,
  +.0096 at a .2 (TRICK_1019 sec 2.2, fam_g58.log); q < rho condition (skeptic_eoa.md sec 1-2).
- E-OA formula (a/2 times the sum) matches TRICK_1019 sec 3.1 and dose_gold58.py; pw weight values.
- gold58 +.0014 / +.0025; ~20 pair swaps; five gold58 tweaks lost the same day (TRICK_1019 sec 1, sec 5).
- OAI: 2,399 baseline (V00) knees; pre-registered SUPPORT rule; all nine table values and P(>0) 1.000, 3 of 3 proxies, 5 of 5
  folds (g1_box_run.log, g1_replication_box.log, A20_VALIDATION_PREREG.md); phi ~.05 vs ~.3; +.0005 / +.0007 estimates.
- One consistent factor after five reads (recal.md: gap-matched read, factor .86); optimum a ~.225 to .30 (VAL_1020 sec 1).
- Dose curve values in the text and figure (data/dose_gold58.json; self-check against 10 logged points). pw .05 and .10 equal
  sym exactly; the code applies the weights correctly, so this is a tie in AUC, not a bug.
- 66-pair census; MCL skeptic "E about 0", allowed only inside swide (stack/build.md sec 8); swide only .960.
- What did not work: solo .923 to .944, .951 = fleet .4 + Raptor .6 from swapiy's account (HANDOFF l.989), KC .957 / .941,
  S6 seeds .921 / .915 vs .931, RSNA-only Raptor seeds .955, one duplicate pair in 4,407 (TRICK_1019 sec 2.2).
- Canary ~.04 to .916; ACL / Fracture canaries (c4 cells 15, 18); 2 rerun errors, 2 DNS failures, copy 3 clean (HANDOFF l.23, l.30).
- JPEG: sha-checked cp313 wheels from glorysing, byte-identical on uncompressed data, train and visible test uncompressed
  (c4 cell 2; session_2026-10-15/jpeg/RESULT.md l.22-24).
- "Kaggle truncates to 3 decimals": consistent with varshithakolupuri's note (c4 cell 0) and the density model (a25 .95866
  needs +3.4e-4 for .959). FIGS.md l.120 says "rounded"; change to "truncated" there (internal file).
- Teammates: swapiy and kmrsuman trained fleet folds on Kaggle (HANDOFF l.741, l.751-752, l.1118).
- Figures: lb_progress, oai_validation and dose_curve labels match the sources; the coupling bars match coupling_map.md;
  pipeline numbers match the c4 notebook (apart from B1 to B3).

## F. Safety scan

- Em dashes: none. writeup_draft.md, FIGS.md, make_figs.py and dose_gold58.py are pure ASCII; the PNGs show no em dash
  (the minus sign in the figures is U+2212).
- OAI: only aggregates (n, AUC deltas, sd, P, correlations). No ids, rows or per-knee values in the draft, FIGS.md or figures.
- Competition data: no report text, no study-level label, no image. gold58 appears only as aggregate AUCs and correlations.
- No credentials, box IP, box username, NDA account details or DAR number in the draft or figures.
- FIGS.md and both scripts contain local paths (C:/Users/eeshs/...). Fine as internal files; strip them if any of these files
  is ever published.
- Teammates: Kaggle names only.
