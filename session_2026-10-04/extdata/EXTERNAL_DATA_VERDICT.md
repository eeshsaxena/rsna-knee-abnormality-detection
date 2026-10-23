# RSNA Knee: may we train on external knee MRI? Verdict as of 2026-10-04

For: Eesh (team Eesh saxena). Competition: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
Key dates (MEASURED, Overview): entry/merger 2026-10-15, final submission 2026-10-22 23:59 UTC, winners' requirements 2026-11-05.

How this was built: two research passes plus two independent verifier passes on 2026-10-04. Every source was read signed out and read-only, with nothing downloaded, no forms, no terms accepted and no posts. This file uses only claims a verifier CONFIRMED, plus claims marked UNCERTAIN where flagged. Claims a verifier refuted are left out or corrected.

Tags: MEASURED means read on the cited page. INFERRED means reasoning from measured facts. UNCERTAIN means the verifiers could not settle it.

The wording below is paraphrased. Verbatim quotes for every item are logged in `rules_forum.md`, `verify_rules.md`, `verify_licenses.md` and `verify_usage.md` in this folder. This is not legal advice.

---

## TL;DR

- **External data is allowed in principle.** The Host can prohibit any specific set. Only one set has been banned by name: **KneeCoT**.
- **The Host's general test** (2026-08-27):
  - a free set with simple registration or a click-through agreement generally passes;
  - institution-specific approval, a negotiated agreement, IRB approval or lengthy credentialing may fail;
  - a non-commercial-only licence is not a problem on its own, and prize money does not count as commercial use.
- **No per-dataset yes or no exists** for MRNet, SKM-TEA, fastMRI, OAI, KneeMRI or KneeXNet-2.5D.
  - The request for one (thread 743416, 62 votes) is unanswered.
  - The Host has said nothing since 2026-09-21 but is still active on the site.
- **No dataset has an explicit Host "yes".** The closest to safe on access is KneeMRI from its official Zenodo page: open, free, no registration. Its no-derivatives clause mainly bites when weights are published, which is a prize-stage duty.
- **There is no public evidence that any top-12 team uses external knee MRI.**
- **For a medal, gray data is still not risk-free.** The rules let the Sponsor disqualify any entrant, remove them from the leaderboard and withhold medals, not only prize winners.

---

## 1. The rule, in plain words

### 1.1 Official rules text (source type: official rules)

Source for all items: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules. All MEASURED unless noted.

| Section | Plain meaning |
|---|---|
| 2.6(b) | External data and models are "acceptable unless specifically prohibited by the Host". They must be reasonably accessible to all and of minimal cost. The Host applies a Reasonableness Standard (cost thresholds, accessibility), may assess a dataset and exclude it, and says buying a proprietary licence that costs more than a prize is not reasonable. The verifier notes 2.6(b) names geo restrictions as an accessibility factor. |
| 2.6(a) | External data must be public, equally accessible and free, OR pass 2.6(b). Using it does not reduce your Section 2.8 winner obligations. |
| 2.5(a) | The winning Submission and code are licensed CC-BY-NC 4.0. You need not open-source licence input data or pretrained models that carry an incompatible licence. INFERRED: the carve-out covers those inputs, not your own trained weights. The licence text contradicts itself: it also calls this a licence that does not limit commercial use. Thread 739905 (2026-09-07) asks about this and has no reply. |
| 2.5(b) | Winners give a reproducible method description, including training details. INFERRED: any external data use becomes visible at winner review. |
| 2.8(a) | Winners deliver training code, inference code and model weights. The weights go out as a public Kaggle dataset. Winners represent that they have an unrestricted right to grant the licence. |
| 3.14(a) | The winner warrants sole ownership of the Submission and no infringement of third-party rights, including confidentiality obligations. |
| 3.9(b) | On non-compliance the Sponsor may disqualify, or allow a one-week fix that can include resolving licence conflicts. |
| 3.4(d), 3.8(e) | The Sponsor may disqualify any entrant who breaks the rules. A disqualified participant may be removed from the leaderboard and denied points or medals. |
| (absence) | There is no external-data disclosure clause, thread or deadline. This was checked against the full rules text, the pinned topics and a forum search. |

**Overview page** (https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview, MEASURED):
- Code Requirements: freely and publicly available external data is allowed, including pretrained models.
- Winners are also asked for a video, a forum link to the open-sourced code and weights, and to share the final model publicly.

**Data page** (https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data, MEASURED): no external-data statement. The competition data licence reads Subject to Competition Rules.

### 1.2 Host and staff statements (source type: host or staff)

All Host statements are by Po-Hao "Howard" Chen (account javaduke95), badge **COMPETITION HOST**. Times are UTC. All MEASURED by both verifiers.

| Date (UTC) | Thread | What the Host ruled (paraphrase) | Status |
|---|---|---|---|
| 2026-08-09 09:52 | [733965](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965) (pinned Host topic on commercially hosted LLMs) | Hosted LLM services are allowed under 2.6(b). The Host reserves the right to judge accessibility or unfair advantage. Host replies on 2026-08-27 confirm that LLM-API labelling and LLM-derived labels and embeddings are fine. | Current |
| 2026-08-25 11:00 | [734109](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109) | **KneeCoT is not allowed.** Reason: it needs a formal data agreement with the dataset's institution, which makes access uneven. This could change if the Hugging Face access policy changes; as of 2026-10-04 it has not, and the page is still gated behind a hospital agreement. | Current |
| 2026-08-27 22:14 | [733965](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965) (reply to UGUR OZER, who named OAI, MRNet, fastMRI+ and SKM-TEA) | (1) Using these datasets counts as non-commercial; prize money alone does not make it commercial. (2) A non-commercial-only licence does not by itself exclude a dataset. (3) Simple registration or a click-through agreement generally satisfies accessibility. Institution-specific approvals, negotiated written agreements, IRB approval or lengthy credentialing may be a barrier, especially given the remaining timeline. (4) Teams stay responsible for complying with each dataset's licence, including any required release of code or models. RSNA and Kaggle will not interpret third-party agreements. | Current. This is the most general statement and nothing supersedes it. Accessibility was not ruled per dataset. |
| 2026-09-21 15:34 | [741819](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819) | **OAI** is within the rules only if it is free and accessible to all without an escalated institutional review and sign-off, and without needing a legal team. The Host cannot comment on OAI's own data-use terms. | Current and consistent with 08-27. It is a condition, not a yes. |

**Kaggle staff:**
- María Cruz (badge KAGGLE STAFF) posted only the getting-started and Discord thread ([730709](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/730709), 2026-07-30). It has no external-data content and says important questions belong on the forum.
- Ryan Holbrook (Kaggle staff) has no posts in this competition.

**Supersession and recency:**
- The Host profile (https://www.kaggle.com/javaduke95/discussion) shows 19 posts. The newest is the OAI reply of 2026-09-21. Nothing later changes the three rulings above. The profile also shows the Host was last seen within the past day.

**Competitor readings (opinions, not rulings):**
- Mazzutti (733652, 2026-08-20) argued that MRNet, fastMRI, OAI, SKM-TEA and KneeCoT all pass 2.6(b). This is partly superseded by the KneeCoT ban (08-25) and the OAI condition (09-21).
- Competitors read 741819 differently:
  - Tucker Arrants (2026-09-21) reads it as a no for OAI.
  - Amanie El-Hajj (733652, 2026-10-02) reads it as a yes.

**Unanswered threads.** None of these has a Host or staff reply as of 2026-10-04 (MEASURED):
- [743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416): per-dataset yes or no. YYama, 9th, posted 2026-09-26, 62 votes. The last comment is from 2026-10-03.
- [743420](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743420): does 2.5(a) exempt weights trained on no-redistribution data from the 2.8 public-weights duty? Richard_61, 2026-09-26, bumped 09-29. **This is the question that decides prize eligibility for no-derivatives data.**
- 737950, 738111, 738156 (OAIZIB-CM), 735497: general external-data questions.
- 735121 and 745283: RadImageNet weights licence.
- 739905: the contradictory winner licence.
- 744056: whether other participants' public checkpoints are prize-eligible.

---

## 2. Dataset by dataset

### 2.1 Facts

| Dataset | Access route | Cost | Institutional sign-off? | Licence terms that matter here |
|---|---|---|---|---|
| **fastMRI** (NYU knee images) | Online form at https://fastmri.med.nyu.edu/ with an e-signature. Required fields: full name, email, institution (MEASURED). Whether approval is automatic is UNCERTAIN: an undated third-party document says links arrive by email shortly, while a competitor's account of having to email NYU was refuted as to the form. | Free (MEASURED) | No signing official on the form. However, the Institution field is required, the page says the application must include institutional affiliation and proposed uses, and the agreement requires all approvals for the proposed use under institutional policies (MEASURED). **Borderline.** | Internal research or education only. No selling or monetizing. No transfer of the data or of data files derived from it, except to people you supervise and for academic publications. A clause against linking with other datasets, which sits in the re-identification sentence. Destroy all copies afterwards. (MEASURED) |
| **fastMRI+** | Labels on GitHub (https://github.com/microsoft/fastmri-plus), MIT licence (MEASURED). A CSV-only Kaggle copy also exists (bharath111l). | Free | Labels: no. Images: same as fastMRI. | Labels are MIT. The images still need the fastMRI agreement. The labels are useless without the images. |
| **MRNet** (Stanford AIMI) | Now on Redivis (https://stanford.redivis.com/datasets/4a2c-4cpkzrn2c). Sign in with a Stanford ID or Google account, join the AIMI organization, then apply for the Stanford research agreement for this dataset (MEASURED). Whether approval is automatic is UNCERTAIN. | Free for non-commercial research. A commercial licence is USD 70,000 per dataset per year (FY25) (MEASURED). By the Host's 08-27 ruling it is not needed. | None shown. Each user must register individually, so every teammate would register (INFERRED from the agreement text). | Research Use Agreement on the legacy AIMI page: personal non-commercial research only; no monetization; **no derivative works**; **no distribution or publication of copies** (MEASURED verbatim by the licence verifier). The new aimi.stanford.edu page renders no terms text when signed out. Whether the Redivis agreement text is identical is UNCERTAIN. |
| **SKM-TEA** (Stanford AIMI) | Same Redivis route and same two steps (https://stanford.redivis.com/datasets/5r8z-achpw6q4f) (MEASURED). There is no Kaggle copy. | Same as MRNet | Same as MRNet | Same Stanford research agreement: no derivative works, no redistribution (MEASURED). |
| **OAI** (NIMH Data Archive, NDA) | Login.gov, then an NDA account (staff review takes 1 to 3 business days, longer when shared data is involved), then request the OAI permission group and accept the Data Use Certification (MEASURED). Every NDA page carries a banner saying the archive is under review for possible changes (MEASURED). | Free | **UNDETERMINED.** NDA's Open Access groups need no institutional sponsor and are approved on submission. But the same NDA pages also say requesters need an institutional email, a Signing Official's e-signature and affiliation with an NIH-recognized institution, and must not be in a country of concern (MEASURED). NDA contradicts itself, and whether the OAI group is Open Access is not public (UNCERTAIN). | The Data Use Certification covers research and research development. It has no commercial-use ban and forbids sharing with people not listed on it (MEASURED). |
| **KneeMRI** (Rijeka, 917 sagittal PD-FS volumes, ACL labels) | Official copy on Zenodo (https://zenodo.org/records/14789903): open access, no registration, 3.7 GB, record created 2025-02-02 (MEASURED). The old university page now returns 403/404. | Free | No | **CC BY-NC-ND 4.0** (MEASURED). Per the Host, non-commercial is fine. Creative Commons guidance (https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/) says the no-derivatives term is triggered only when adaptations are publicly shared. Its conservative line is not to train on ND content (MEASURED by verifier). |
| **Knee MRI 917 on Kaggle** (ibrahimshaikhdev/knee-mri-917) | Kaggle download. Published 2026-10-03, labelled CC0, no description, no provenance, about 6.67 GB (MEASURED). | Free | No | All 918 files match another Kaggle re-upload of KneeMRI by name and byte size, and the label counts are 690/172/55 (MEASURED). It is a re-upload of KneeMRI (INFERRED, strong). The CC0 label cannot relicense CC BY-NC-ND material (INFERRED). |
| **KneeXNet-2.5D** (Pitt) | GitHub (https://github.com/pitthexai/Knee_MRI_Segmentation_2.5D): masks only for 222 OAI MRIs; the images must come from OAI (MEASURED). The .pth weights are 133-byte stubs, likely Git LFS pointers, so the weights may not actually be available (INFERRED). | Free | Masks: no. Images: same as OAI. | No LICENSE file. The README says free for research and educational use (MEASURED). |
| **OAIZIB-CM** (Hugging Face, YongchengYAO/OAIZIB-CM) | Ungated download, about 507 OAI knee MRIs plus masks, CC BY-NC 4.0 (UNCERTAIN: verifier read a fetch summary, not the full page). | Free | No for this route | It re-hosts OAI images, so whether that re-hosting complies with OAI's terms is questionable (INFERRED). Thread 738156 asks about it and has no Host answer. |
| **3DReasonKnee** (Hugging Face, rajpurkarlab) | Annotations CC BY 4.0. Images must come from OAI through NDA (UNCERTAIN, fetch summary). | Free | Same as OAI | Follows OAI. |
| **MOST** (NIA Biobank) | Request must come from a PI-level researcher (not students or post-docs), with IRB approval, NIA scientific review over several weeks, and an institutional signing official (MEASURED). | Free | **Yes** | Not available for commercial use (MEASURED). |
| **KneeCoT** (Hugging Face) | Gated; needs a formal ethical-use agreement with the affiliated hospital (MEASURED). | Free | **Yes** | CC-BY-NC-4.0. Banned by the Host. |
| **Other Kaggle re-uploads** | techtech69/mrnet-knee-mri-stanford-aimi (MRNet, labelled CC0, 9.51 GB); techtech69/rijeka-knee-mri-dataset (KneeMRI); sohaibanwaar1203/kneemridataset (KneeMRI, 2019, licence unknown) (MEASURED). | Free | No | A re-upload does not change the source terms. The MRNet agreement bars distribution, so the MRNet re-upload itself appears to breach it (INFERRED). |

### 2.2 Verdicts

ALLOWED means an explicit Host yes, or clearly inside the Host test with no conflicting terms. NOT ALLOWED means a Host ban, or the set clearly fails the Host test. GRAY means no ruling and a live problem. All verdicts are INFERRED from the measured facts above.

| Dataset | (a) Medal eligibility | (b) Prize eligibility | Why |
|---|---|---|---|
| fastMRI | GRAY | GRAY, leaning NOT ALLOWED | The form looks like the click-through the Host accepts, but the required institution, institutional approvals and research-only purpose push toward the barrier side. Approval mode is unverified. The ban on transferring derived data files sits badly with publishing weights under 2.8(a) and the unrestricted-right warranty. |
| fastMRI+ (labels only) | ALLOWED but useless alone | ALLOWED (labels) | MIT labels with no access barrier. Any use with the images inherits the fastMRI verdict. |
| MRNet | GRAY | GRAY, leaning NOT ALLOWED | Access is probably click-through, which the Host generally accepts, but the approval mode is unverified and there is no per-dataset ruling. The Host cleared the non-commercial point by name on 08-27. The no-derivatives and no-distribution clauses clash with the public-weights duty (2.8(a), 3.14(a)). Thread 743420 is unanswered. Using the Kaggle CC0 re-upload adds a provenance problem. |
| SKM-TEA | GRAY | GRAY, leaning NOT ALLOWED | Same as MRNet. It is also small: 155 qDESS scans. |
| OAI (NDA) | GRAY: NOT ALLOWED if any institutional step applies | GRAY | The Host's condition depends on facts that are only visible after signing in, and NDA's pages contradict each other. If an institutional email or Signing Official is required, OAI fails the 09-21 test. The country-of-concern limit is a further 2.6(b) accessibility factor. Its terms are otherwise the friendliest: no non-commercial ban and no no-derivatives clause. |
| KneeMRI (Zenodo) | GRAY, lowest risk of all | GRAY | It passes the Host's access test on its face: open, free, no registration. The Host says non-commercial is fine. There is still no explicit yes, and 2.6(b) lets the Host exclude a set after assessment. No-derivatives bites when weights are publicly shared, which is the 2.8(a) duty, so prize eligibility hinges on unanswered 743420. |
| Knee MRI 917 (Kaggle) | GRAY, worse than Zenodo | GRAY | Same data as KneeMRI, but with a false CC0 label and no provenance. There is no reason to prefer it over the official Zenodo copy. |
| KneeXNet-2.5D | GRAY | GRAY | Masks only, and the images come from OAI, so it inherits OAI's verdict. It has no licence file, and the weights may not be downloadable. |
| OAIZIB-CM | GRAY | GRAY | Ungated, but it re-hosts OAI images. Whether the Host judges it by the access path or by OAI's origin is unanswered (738156). |
| 3DReasonKnee | GRAY | GRAY | Its images come from OAI, so it follows OAI. |
| MOST | NOT ALLOWED | NOT ALLOWED | It needs a PI, IRB approval and an institutional signing official, which clearly fails the Host test. |
| KneeCoT | NOT ALLOWED | NOT ALLOWED | Banned explicitly by the Host on 2026-08-25. |

**Pretrained models relevant to the current pipeline** (2.6 covers models too):
- **RadImageNet ResNet-50:** its licence is reported as CC-BY-NC-SA-4.0 on a Kaggle mirror but MIT on Hugging Face. That discrepancy was not independently re-checked (UNCERTAIN). Threads 735121 and 745283 ask about it and have no Host answer (MEASURED).
- **Other participants' public competition checkpoints** (for example CoAtNet weights): prize eligibility is unanswered in 744056 (MEASURED).
- **OrthoFoundation, RadioDINO and the MST_MRNet checkpoint:** their training corpora are unverified. The MST checkpoint is probably MRNet-trained (INFERRED).
- Medal-level risk from public pretrained models looks low, because the Overview explicitly allows them (INFERRED). The prize stage is open.

---

## 3. Obligations if you use external data

1. **Disclosure:** there is none. No external-data disclosure thread, deadline or rule exists (MEASURED absence).
2. **If you finish in a prize place:**
   - training code, inference code and weights, with the weights as a public Kaggle dataset, plus a statement that you have an unrestricted right to license them (2.8(a));
   - a reproducible write-up including training details (2.5(b));
   - a video, a forum link to the code and weights, and the final model shared publicly (Overview);
   - all by 2026-11-05;
   - the 3.14(a) warranty applies, with a possible one-week fix for licence conflicts under 3.9(b). (MEASURED)
3. **Winner licence:** CC-BY-NC 4.0 for your Submission. The incompatible-licence carve-out covers the input data and models only (MEASURED, scope INFERRED). The licence text contradicts itself (MEASURED).
4. **The dataset's own terms apply even if you only chase a medal.** The Host puts this on the team (MEASURED, 733965).
   - KneeMRI: attribute the source, and do not publicly share adaptations (INFERRED from CC BY-NC-ND).
   - MRNet and SKM-TEA: each user registers individually, and no redistribution.
   - fastMRI: no sharing of the data or derived data files, and destroy copies when done.
   - OAI: no sharing with people not listed on the Data Use Certification. (MEASURED terms)
5. **Keep weights private:** weights trained on no-derivatives data, kept in a private Kaggle dataset for inference, are not public sharing. Making that dataset public would be (INFERRED).

---

## 4. What top teams appear to do (evidence only)

**Leaderboard on 2026-10-04 (MEASURED):**
- 1. rhoskeri 0.963
- 2. Pa3BaJluHa 0.962
- 3. joinT fusion 0.961
- 4. Wassim Dobbi 0.961
- 5. YOKKOISHO 0.960
- 6 to 12 all at 0.959

**No disclosed external knee MRI use by any top-12 team (MEASURED absence):**
- No visible top-12 member has a public notebook or dataset for this competition. The verifier also checked 12 hidden team members.
- rhoskeri, Pa3BaJluHa, Wassim Dobbi, TomKroumov and NMarkS have zero discussion posts.
- The Pa3BaJluHa account joined 5 days ago, so its silence means little.

**What top-LB posters credit** (thread 735304, MEASURED):
- tennogh (5th): pseudo-labels.
- Scott Willis (10th): local LLM label extraction, about 0.949 single fold.
- CoolinLai (14th): ResNet50 5-fold, 0.954.
- Parag (13th): field-of-view tips.
- After expanding every reply, the thread has zero mentions of MRNet, OAI, fastMRI or KneeMRI.

**The forum worry is not a disclosure:**
- YYama (9th) worries that external data could give a big jump. That is not a statement that anyone uses it.
- Richard_61's mention of teams using no-redistribution data is phrased conditionally (MEASURED, weak as evidence).

**Kaggle re-uploads exist (MEASURED):**
- The MRNet re-upload shows 2,321 downloads but only 30 views. About 2,185 of the downloads come in one spike around 2026-09-24, which looks like scripted pulls (INFERRED). So the count says little about real use.
- About 26 notebooks run in 2026 attach the 2019 KneeMRI upload. The ones checked never read it in their code.
- A notebook with MRNet in its title means the MRNet-style architecture trained on competition data, not MRNet data.
- tsuyu122's harmonized label table cites a Rule 4.b that does not exist in these rules. Treat it as unreliable.

**Conclusion (INFERRED):** the claim that top teams use external data is forum suspicion, not disclosure. Absence of evidence is not proof of non-use. Methods will only surface in post-deadline write-ups, which is also when audits happen.

**Precedent** (MEASURED):
- **RSNA 2025 Aneurysm (identical 2.6 text):**
  - About 3 weeks before the end, the Host rejected sets that needed institution or PI details on a request form, and approved fully open CC0 OpenNeuro sets.
  - 1st place used no external data.
  - 2nd place excluded the rejected set.
  - 6th place said the open data did not really help CV.
- **RSNA 2023 Breast:**
  - The Host ruled 3 days before the deadline that models built on non-public NYU data could not win.
  - The Host also rejected BSD2-licensed models on licence grounds.
  - 3rd place used VinDr (registration plus a signed agreement) and kept its place, as far as the record shows (INFERRED).
- **Deepfake Detection Challenge (2020):** the Host removed the top two prize places after a licensing review, for external data not allowed under the rules (https://www.kaggle.com/competitions/deepfake-detection-challenge/discussion/157925, 2020-06-12).

---

## 5. Recommendation for a medal (not prize) target, about 18 days left

All INFERRED from the evidence above.

1. **No external knee MRI set is clearly safe.** None has an explicit Host yes, and 2.6(b) lets the Host exclude a set after the fact.
2. **If you use one, use KneeMRI from the official Zenodo record only.**
   - It is the only set that passes the Host's access test with no ambiguity: open, free, no registration.
   - Its non-commercial clause is cleared by the Host.
   - Its no-derivatives clause matters mainly if you must publish weights, which is a prize-stage duty.
   - Do not use the Kaggle CC0 re-upload (knee-mri-917).
   - Limits: it is 917 sagittal PD-FS exams with ACL-only labels, so its value for this competition's labels is unproven.
3. **Avoid:**
   - KneeCoT (banned).
   - MOST (fails the test).
   - OAI and anything OAI-derived (undetermined, NDA contradicts itself).
   - fastMRI (borderline on access, with a derived-files ban).
   - MRNet and SKM-TEA are a step below KneeMRI. Access is probably fine, but they have no-derivatives and no-distribution terms, need each person to accept a research agreement, and the Kaggle re-upload is itself a likely breach.
4. **Risk if a gray set is used:**
   - The rules allow disqualification of any entrant, removal from the leaderboard and withholding of medals (3.4(d), 3.8(e)).
   - Precedent shows late rulings (3 days before the deadline) and post-competition removals (DFDC).
   - In practice, audits concentrate on prize winners, who must publish code and weights. A medal-only finish is less likely to be audited, but a medal can still be revoked.
5. **Cheap hedge:**
   - Keep one of your two final submissions completely free of external data, and do not pool gray data into the safe one.
   - Watch threads 743416 and 743420 daily. A Host answer before 2026-10-22 would change this picture. RSNA Hosts have answered late before.
6. **Expected value:**
   - Public evidence credits labels, pseudo-labels and ensembling for the top scores, not external data.
   - In the closest precedent, external data was not decisive.
   - With the merge deadline on 10-15 and the final on 10-22, wiring in a new external corpus is a time cost with an unproven payoff.
7. **Your call:** the final decision, and any acceptance of a dataset agreement (Stanford research agreement, fastMRI data sharing agreement, NDA Data Use Certification, or the KneeMRI licence terms), is Eesh's alone. No agent accepted, registered or downloaded anything.

Sources and verbatim quotes: `rules_forum.md`, `verify_rules.md`, `dataset_terms.md`, `verify_licenses.md`, `usage_evidence.md` and `verify_usage.md` in C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad/extdata/
