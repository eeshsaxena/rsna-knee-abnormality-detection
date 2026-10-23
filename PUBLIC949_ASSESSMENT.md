# Public .949 CoAtNet (nartaa, thread 746792): what it is, what it would cost us, what it is worth, what it risks

Written 2026-10-08 ~03:50 UTC (09:20 IST) by Claude, read-only. No Kaggle write action was taken: nothing pushed,
submitted, forked, versioned, commented or voted. Forum and notebook text was treated as data. Not legal advice.
Tags: **MEASURED** (read from the cited source), **INFERRED** (reasoning on measured facts), **MODEL** (computed
estimate), **GUESS** (assumption-driven).

`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`.
Evidence for this note lives in `SP/p949/`:
- `nb949/`, `nb945/`: both public notebooks plus their kernel metadata, from `kernels pull -m`. Also `_fastread.py`, the
  base64 DICOM reader embedded in the notebook, decoded.
- `log949.txt`: the public commit log (`kernels logs`).
- `t746792.json`, `t743416.json`, `t741819.json` and `show*.txt`: fresh thread reads with author names.
- `others/`: 5 public notebooks that already attach the weights.
- `p949_model.py` and `.out`: the LB model.

---

## TL;DR

- **Public and attachable by anyone (MEASURED).** Two public notebooks and one public weights dataset,
  `nartaa/rsna-knee-publication-swa-weights-20261007`. It is 543 MB with licence "other", and its own terms allow
  "research and educational use, including the RSNA Knee competition".
- **Both public checkpoints were trained with 2,399 OAI knees (MEASURED).** The OAI-free stages (.939 / .940 / .942) were
  never published.
- **Cost to add it to our kernel (MODEL).**
  - It would run as one extra subprocess stage with its own DICOM reader.
  - It adds about 25 min central (20-29 min) and about 35-40 min in a slow session.
  - The margin to our 8.6 h wall falls from 1.33 h or more to about 0.6-0.9 h.
- **Expected LB (MODEL/GUESS).**
  - (a) parent + fleet + N would display **.950** most likely, sometimes .951.
  - (b) fleet + N would display **.949-.950**.
  - Today .950 is rank about 124-215 (silver; the cut is .949 at rank 272), but it puts us inside a 92-team cluster
    that is still growing.
- **Rule risk is real and unresolved.**
  - The host has not posted anywhere since 09-21. The OAI test from 09-21 ("no escalated institutional review and
    sign-off") is still the only guidance.
  - New thread evidence points toward unequal access: a competitor in China reports a failed OAI download.
  - Because this is a code competition, any submission that attaches this dataset is trivially identifiable. That makes
    any later ban easy to enforce leaderboard-wide, not only at prize level.
- **Decision is Eesh's (sec 6).** Our standing rule (no external data) says no by default.

---

## 1. Slugs, weights, licence, visibility (MEASURED)

| item | value |
|---|---|
| Accuracy notebook | `nartaa/rsna-knee-0949-anatomical-mirror` (id 137529884): public, GPU, internet off, `machine_shape` NvidiaTeslaT4 (2 T4 in its log). Docker pinned `gcr.io/kaggle-private-byod/python@sha256:2757e0c7...`. Sources: the competition and the weights dataset only. Last run 2026-10-07 20:28 UTC, ~36 votes. Scored submission 56906959 = .949 |
| Efficiency notebook | `nartaa/rsna-knee-0945-efficient-224crop` (id 137529933): same layout, docker `sha256:37c64f7d...`, last run 20:57 UTC. Submission 56707773 = .945 |
| Weights dataset | `nartaa/rsna-knee-publication-swa-weights-20261007` (datasetId 12431568), created 2026-10-07 19:26 UTC, 543,312,838 B, licence `other`, 258 downloads / 81 views / 1 vote at read time |
| Files | `README.md` (2,983 B); `checkpoint-manifest.json` (645 B); `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` (292,840,787 B, sha256 `7e5315da...44ef`, the .949 model); `raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt` (292,842,755 B, sha256 `3394fdd7...15d6`, the .945 model) |
| Licence text (dataset description) | "provided for research and educational use, including the RSNA Knee competition". It grants no rights to OAI data, and NDA terms still govern OAI access. Third-party code keeps its own licences. No open-data licence is applied |
| Private originals (not attachable) | Scored sources `nartaa/rsna-knee-original384-s42-mirror320-infer` and `nartaa/rsna-knee-coat224c-oai-ac-infer`. Their private dependencies are `nartaa/rsna-knee-oai-original384-s42-pair` and `nartaa/rsna-knee-raptor-alldata-d96-blendjev-oai-c08r224` |
| Public uptake | The weights are already attached by `matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed` (a byte clone, 58 votes), `sujanmajhisuzan/rsna-knee-apex-grandmaster-stack` (N + goodpjw2008 ConvNeXt reader, per-label rank fusion), `haideptry/rsna-knee-v2-vs-apex-comparative-study` and `xianhan/rsna-knee-0949-anatomical-mirror-candidate` |

Any team can attach the dataset; it is public and the notebooks are public, `is_private: false` (MEASURED).

---

## 2. What it needs at inference, and fitting it into our kernel

### 2.1 Pipeline (MEASURED from the notebook source, cell 4, 602 lines)

- **Own DICOM preprocessing.** `_fastread.py` (variant `study_lut`), embedded as base64. It claims to be
  byte-identical to the Raptor `build_study`.
  - 5 fixed slots per study: Sagittal fluid 26, Sagittal other 22, Coronal fluid 18, Coronal other 12, Axial 18. That is
    96 slices.
  - Picks are a linspace over the 2-98 % span of each series. Each series gets a 2-98 % percentile window over its own
    picks.
  - Each slice gets a 140 mm centre crop, then an INTER_AREA resize to 384 px, stored as uint8.
- **Windows and views.** 94 three-slice windows. Each window takes a native 320 crop `[32:352, 32:352]` with no
  resize, then ImageNet normalisation.
  - It runs two views: plain, and the anatomical L/R mirror. Sagittal windows reverse their 3 channels; coronal and
    axial windows flip their width.
  - The two sigmoid outputs are averaged, then each finding is percentile-ranked.
- **Model.** `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` (timm, `pretrained=False`, `img_size=320`) with a per-finding
  MIL attention head. It runs in chunks of 16 windows under fp16 autocast.
  - The checkpoint sha256 is asserted twice. It requires CUDA, and `cudnn.benchmark` is True.
- **Parallelism.**
  - One model copy per GPU, each with a consumer thread.
  - A `fork` pool prepares the studies: `nproc = max(4, cpu-8)`, so 4 on Kaggle, each with 32 header threads and
    16 pixel threads.
  - Up to 48 prepared studies wait in a queue, about 14 MB each, so about 0.7 GB of RAM.
- **Dependencies.**
  - Python 3.11 or newer (`hashlib.file_digest`), torch with CUDA, timm, numpy, pandas, pydicom and cv2.
  - No wheels, no pip installs, no internet.
  - It recorded torch 2.11.0+cu128 and timm 1.0.29.
  - Our parent already builds the same arch with the image's timm for its Raptor arms, so no extra wheel is needed
    (INFERRED).

### 2.2 Runtime

| quantity | value | tag |
|---|---|---|
| Their submit-to-score turnaround, which includes queueing | 26.1 min (accuracy), 7.2 min (efficiency) | MEASURED (author's figures) |
| Upper bound on standalone hidden runtime | 26 min or less for accuracy. The 7.2 min efficiency run uses the same reader, so the full hidden-set decode takes about 7 min or less and is not the bottleneck | INFERRED |
| Public commit, 3 studies | load 3 s per GPU; prep 3 s; done 23 s | MEASURED `log949.txt` |
| GPU per study | CoAtNet-2 at 384 fp16 runs ~20 ms/window on a T4 (`SP/parent_speed/PROFILE.md`, 1.80-1.86 s per 94 windows). At 320 that is ~14 ms. 188 windows (94 x 2 views) gives ~2.6 s per GPU, or ~1.3 s/study on 2 T4 | MODEL |
| Stage time at N 1,150-1,322 | **~25 min central (20-29)**; slow session (x1.3-1.4) **35-40 min** | MODEL |
| Efficiency model instead (.945, one view at 224) | ~8-12 min, decode-bound | MODEL |

### 2.3 Running it as an extra stage in our kernel (INFERRED)

- **It cannot reuse our decode.**
  - The parent's Raptor stage uses capacity-aware dense-96 sampling, takes its percentile over the 64-slot "original"
    picks, and clears its pixel cache when it finishes.
  - N uses fixed 26/22/18/12/18 slots with its own percentile scope.
  - The volumes are therefore not byte-identical, so reusing the parent's decode would break parity.
- **Run it as a subprocess child**, the way the parent's CoAt children run.
  - Write cell 4 to `/kaggle/working`, with the output path changed from `submission.csv` to something like
    `_n949.csv`.
  - Reason: its `main()` forks 4 prep processes after CUDA initialisation. Forking our multi-GB kernel process is a RAM
    risk, while a fresh child keeps the fork small.
  - Start it only after the fleet stage has freed both GPUs. Estimated VRAM is 2-4 GB per T4 (GUESS); weights are
    0.3 GB per GPU.
- **Budget.**
  - K9 already has a margin of 1.33 h or more to the 8.6 h wall (K9_RUNBOOK sec 0).
  - After the N stage that margin is about 0.6-0.9 h central and about 0.4-0.7 h slow (MODEL).
  - The parent's own 8 h guard and the 9 h hard limit are unchanged.
  - The N stage needs its own start check, for example start only if 45 min or more remain, plus fail-loud handling
    that matches the strict mode.
- **Parity check available.**
  - The commit log prints N's per-label ranks for the 3 visible studies (`log949.txt`).
  - A stage that reproduces those ranks plus the sha256 is a cheap gate. It is weak, because it covers only 3 studies.
- **Quota.**
  - Each commit costs about 0.5 h of GPU (K9_RUNBOOK sec 1).
  - Today's 2.52 h is already planned for K9 main + SOLO, so an N variant would compete for that quota and for a
    submission slot.
- **Variant (b), fleet + N without the parent**, would take about 1.3 h fleet + 0.4 h N, roughly 2 h in total instead
    of about 7.5 h (MODEL).

---

## 3. Provenance (MEASURED unless tagged)

- **Accuracy (.949) and efficiency (.945) both include OAI.**
  - Both filenames contain `_oai_`.
  - The dataset description says "Both models used the same 2,399 baseline OAI knees for masked external supervision".
  - The post says OAI labels supervised PF OA, Lateral OA and Synovitis, with the other findings masked.
- **No public competition-only checkpoint from this author exists.**
  - The author's public datasets are only this weights pack and `nartaa/rsna-knee-hpo-assets`. The latter is LLM
    report labels, folds and training scripts (`train_knee.py`), with no weights.
  - The author has no Kaggle Models.
  - The .939 / .940 / .942 "Blend + JEV labels" stages are table rows only.
- The base recipe, `dreaddevelopment/raptor-knee-widedense` (.924, competition-only), is a different checkpoint. Our
  parent already carries three dreaddevelopment Raptor CoAtNets (maxspan-v5, native384dense-v10, native384-v8).
- **No host reply since 10-07.** The fresh `topics show` read at 03:38 UTC 10-08 confirms it:
  - **746792:** authors are Tucker Arrants, Danial Zakaria, Satwik, xun1993, Cody_Null, Cà Rốt Production, plus one
    deleted empty comment.
  - **743416:** the last message is Satwik's, 10-07 13:18. 22 messages, none from the host.
  - **741819:** the only host post is still 2026-09-21 15:34.
  - **Pinned 733343 and 733965:** no new host post.
  - **739905 (winner licence):** bumped 10-08 02:43 by a competitor and unanswered.
- **New in 746792 since the digest.**
  - Satwik: login.gov needs only an email. He shared a UMass email reply that, in his own words, "only confirms its
    usage from their end".
  - Tucker: some countries are barred by NIH controlled-access policy.
  - xun1993 (China): "previously tried to download the OAI data but failed".
  - The author describes OAI as "controlled access datasets distributed from ... NDA" and says he "checked OAI's rules"
    and expects a host answer. None has come.

---

## 4. LB estimates (MODEL/GUESS; `SP/p949/p949_model.py`, `.out`)

**Method.** The same macro binormal rank-blend model as `SP/jump2/lbmath.py`. On 11 OOF pairs it had an error of
+.0000 to +.0009, always optimistic, so subtract about .0005.

**Inputs (MEASURED reads):**
- P (parent) = .943.
- F (fleet solo) = .944.
- P+F at W .50 = .946.
- N = .949.

**Inferred correlations:**
- **r_PF = .923-.967**, point .945 (MODEL).
- **r_PN = .90-.97**, point .925 (GUESS). It is calibrated on the 10-08 cluster: 92 teams sit at .950, and 45 of them
  were at .943 on 10-04. 81 of the 92 last submitted after the post; only 4 of the 14 teams at .951 did. The model
  reads this as the best P+N blend displaying .950.
- **r_FN is not measured.** It is scanned from .86 to .94. N and our fleet are different architectures, but mix50
  carries a Raptor teacher, so .90-.94 is the plausible band (GUESS).

| r_FN | (a) P+F+N, best weights | (a) fixed .25/.25/.50 | (b) F+N, best W_N | (b) W_N .50 |
|---|---|---|---|---|
| .90 | .9513 (w 0/.36/.64) | .9508 | .9513 (.64) | .9509 |
| .92 | .9506 (w .11/.26/.63) | .9503 | .9505 (.67) | .9501 |
| .94 | .9501 (w .21/.13/.66) | .9499 | .9497 (.73) | .9492 |
| range over every PSD combo scanned | .9497-.9529 | .9487-.9525 | .9497-.9529 | |

**Reading:**
- **(a) displays .950 most likely and .951 sometimes.** .949 is possible only if r_FN is .94 or higher together with
  the model's optimism.
- **(b) displays .949-.950.** Once N is in, the parent adds little unless r_FN is high. That is why (b) nearly matches
  (a) at a quarter of the runtime, but it gives up the parent's diversity on private (GUESS).
- **Rank on the 10-08 LB (MEASURED counts).**
  - .950 means rank 124-215. A late submitter lands at the back of the tie.
  - .951 means rank 110-123.
  - .949 means rank 216-300, and the silver cut is rank 272.
  - Our .946 is rank 373-418 today. On 10-04 there were 180 teams above .946; now there are 372.
- **Crowding cuts both ways (GUESS).**
  - If more teams adopt N, the silver and bronze lines move up. Bronze is the top 10 %, about rank 544, and a clean
    .946-.947 could drift toward it.
  - If OAI is later ruled out and enforced, those teams fall back below a clean entry.
- **Caveat.** These are public reads, on roughly 30 % of the test set (~390 of ~1,300 studies, GUESS). The author
  himself says gaps of +.001 are not significant. How much of the OAI gain carries to private is unknown.

---

## 5. Risk summary for Eesh

1. **The rule is unresolved (MEASURED).**
   - Rules 2.6(b): external data and models are allowed unless the Host prohibits them, and they must be reasonably
     accessible to all. Geo restriction is an accessibility factor.
   - Host 09-21 (741819): OAI is fine only "if it is generally accessible to all participants without an escalated
     (i.e. institutional) review and sign-off ... and is free of charge".
   - There is no per-dataset ruling. 743416 (68 votes) has gone 12 days unanswered, and the host has made no forum post
     since 09-21.
   - The author's own text calls OAI "controlled access" via NDA.
   - A competitor in China reports a failed download (10-08 02:51 UTC). Under the host's own test, that is evidence
     against OAI.
   - **Second layer:** whether using a *public model trained on OAI* counts the same as using OAI was asked in 743416
     (Cemal Atas, 09-27) and is unanswered. The weights themselves are free and on Kaggle for everyone, which is the
     strongest argument for "allowed" (INFERRED).
2. **Precedent (MEASURED, quoted in 743416 by Eugene Khvedchenya, 10-03).**
   - In the Deepfake Detection Challenge the top two prize teams were removed for external data that was not allowed,
     "despite numerous clarification requests" that the host never answered.
   - Our verdict file adds:
     - RSNA 2023 Breast: the host ruled 3 days before the deadline.
     - RSNA 2025 Aneurysm: the host rejected sets that needed institution or PI details.
3. **Who gets audited (MEASURED opinion plus INFERRED).**
   - Forum consensus (Tucker Arrants, YYama in 743416) is that thorough audits hit prize and top teams.
   - The rules (3.4(d), 3.8(e)) still let the Sponsor disqualify any entrant and withhold medals.
   - **Why a public-weights blend is different from hidden training use (INFERRED):** the attached dataset slug sits in
     every submission kernel's metadata. If OAI-derived models are banned, finding every user is a database query, not
     an audit, so a medal-level sweep is cheap.
   - Whether a sweep would void single submissions or whole teams is not documented (UNCERTAIN).
4. **Timing (INFERRED).**
   - A ruling before 10-22 can be handled: keep N out of the final picks.
   - A ruling after 10-22, or a silent post-deadline review, is the scenario with real downside. The Deepfake removals
     were post-deadline.
5. **Licence (MEASURED).**
   - The weights' own terms permit competition use.
   - The prize-stage duties (2.8(a) public weights, 3.14(a) warranty) are irrelevant at our rank.

---

## 6. Options and recommendation (decision left to Eesh)

| option | what it means | upside (MODEL/GUESS) | downside |
|---|---|---|---|
| **A. Stay clean (default under our standing rule)** | K9 as planned. No N anywhere | none beyond K9 (.946-.948) | Crowding may push a clean .946-.947 toward the bronze line if adoption continues |
| **B. Hedge** | Build an (a) variant: parent + fleet + N stage. Spend 1 commit (~0.5 h GPU) and 1 slot to read it. Pick 1 clean + 1 with N at the end, and drop N if the host rules against OAI before 10-22 | likely .950 (silver today), sometimes .951 | Uses quota and a slot while K9 needs them. If the selected N pick stands and a post-deadline ban is enforced, the outcome depends on whether Kaggle voids submissions or teams (unknown) |
| **C. Commit to N** | Make N-including blends the main picks | same as B, plus both picks optimised | Full exposure to the rule risk in sec 5 |

**Recommendation.**
- Keep **A** as the working default: it is our standing rule, and the rule question is genuinely open.
- If Eesh wants exposure, **B** is the only form that keeps a clean fallback. Do it only after K9 main + SOLO are read,
  and only with the N stage written as a fail-loud subprocess (sec 2.3).
- C is not recommended while the host is silent and the access evidence in the thread is moving against OAI.
- Whatever he chooses, re-read 746792, 743416 and 741819 before the final-pick selection on 10-22. A host ruling would
  settle it.

## CORRECTION 2026-10-08 ~10:15 IST (main session, after re-reading thread 741819)
The host DID rule on OAI. Po-Hao "Howard" Chen (@javaduke95), 2026-09-21, thread 741819:
"Using OAI as an external dataset does not violate the RSNA Knee Abnormality Detection challenge rules if it is
generally accessible to all participants without an escalated (i.e. institutional) review and sign-off or access to a
legal team, and is free of charge. We do not own or control the OAI dataset and cannot comment on whether this challenge
complies OAI data-use terms from the dataset stewards' perspective."

So the ruling is a CONDITIONAL YES, not "no ruling".

Checking the conditions against the evidence in 746792 (10-07/08):
- **Free:** yes.
- **No institutional sign-off:** several participants (Danial, Satwik, Tucker after correcting himself) got access with an individual login.gov email signup, so this condition looks MET.
- **"Generally accessible to all":** one participant in China reports a failed request, and NIH bars some countries. This is the only remaining doubt (rule 2.6.b names geo restrictions).
- **OAI's own data-use terms:** the host explicitly declined to judge these. That risk sits with the user of the data or weights.

**Revised risk:** LOW-MODERATE, down from "unruled". Option B (build parent + fleet + N with a fail-loud N stage; one clean pick + one N pick) is now well supported. Option C (N in both picks) is defensible but keeps the geo-access tail risk. Eesh decides.

## DECISIVE EVIDENCE 2026-10-08 ~20:35 IST (Eesh, NDA "Data Permissions" page, logged in)
The NDA permission-group table lists:
"Osteoarthritis Initiative | The National Institute of Arthritis and Musculoskeletal and Skin Diseases Osteoarthritis
Initiative. Data Access Requests do not require Institutional sponsorship and will be approved immediately upon ... |
Institutional Approval Required: **No**"

Every other general NDA permission group on that page says "Yes". So OAI meets the host's 09-21 condition (741819): free, and no institutional review or sign-off. The NDA collection page's "Most data access requires ... FWA" is NDA's generic default; it does not apply to the OAI group.

The remaining caveat is geo access (one China report; NIH country rules). That is a "generally accessible to all" question only the host can settle, and Taeyang has asked (743416).

**Revised risk: LOW.** K9N is a legitimate final-pick candidate. Keep one clean pick only as an insurance hedge, or go N in both if the host confirms or stays silent and the LB supports it (Eesh decides ~10-20).

## CONFIRMED 2026-10-08 ~21:20 IST: Eesh's own OAI access approved as an INDIVIDUAL
NDA "Approved Data Access Requests (DARs)": ID 26647 | lead recipient Saxena, Eesh (You) | permission Osteoarthritis
Initiative | INSTITUTION: (empty) | approved 10/08/2026 | expires 10/08/2027 (screenshot from Eesh).

This is direct proof that OAI is obtainable free and without institutional review or sign-off. That meets the host's
09-21 condition (741819). K9N risk is LOW. The geo-access question stays open, but it is the host's call.
