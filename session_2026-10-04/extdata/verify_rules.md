# Adversarial verification of the external-data "rules" research (RSNA Knee Abnormality Detection)

Verifier run: 2026-10-04. I opened every source myself in my own built-in-browser tab (tab-3), plus WebFetch/curl for non-Kaggle pages. Not signed in anywhere. Nothing downloaded except page HTML for text search and one public OAI how-to guide PDF that WebFetch cached (read for its text only). No forms, posts, votes or cookie accepts.
Times: Kaggle hover titles are IST (UTC+5:30); converted to UTC below.
Tags: MEASURED = I read it on the page today. INFERRED = my reading.

## Verdict in one paragraph

The research holds up. Every rules quote, every host quote, every date and every author role I re-checked matches the live pages. I found no host or staff statement that contradicts it, and no newer host post. The host account's profile still shows 19 posts, the newest from 2026-09-21. Corrections are small:
- Two nuances in the host's 08-27 reply were dropped.
- One Tucker Arrants paraphrase mixes two threads.
- One competitor claim about fastMRI access does not match the fastMRI page.
- The researcher's practical line "OAI via NDA likely fails" goes further than the evidence. The OAI access route documented today is an individual Login.gov flow, and nothing I could read says an institution must sign. I could not open the NDA pages themselves (sign-in wall), so OAI stays undetermined.

The researcher also left out some rules clauses that matter for risk:
- 3.4.d and 3.8.e let the Sponsor disqualify any entrant, not only prize winners, and remove medals.
- 3.14.a: the "sole and exclusive owner" warranty.
- 3.9.b: a one-week remediation that covers "resolution of license conflicts".

## 1. Official rules (https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules), MEASURED

| Claim | Status | Note |
|---|---|---|
| 2.6.b "acceptable unless specifically prohibited by the Host", "reasonably accessible to all", "minimal cost", Reasonableness Standard "cost thresholds and accessibility" | CONFIRMED | Exact text present. |
| 2.6.a: publicly available, equally accessible, no cost, OR Reasonableness; does not limit 2.8 | CONFIRMED | Wording: "for purposes of the competition at no cost". |
| 2.5.a carve-out for "input data or pretrained models with an incompatible license" | CONFIRMED | The carve-out exempts "that data and/or model(s)", meaning the incompatible inputs. On a plain reading it does not obviously cover the team's own final weights, which 2.8.a requires to be published (INFERRED). |
| 2.8.a weights "as a public kaggle dataset"; "unrestricted right to grant that license" | CONFIRMED | Both are present. The "unrestricted right" sentence is a separate bullet under 2.8.a. |
| Winner license contradiction (CC-BY-NC 4.0 called a license that "in no event limits commercial use") | CONFIRMED | Thread 739905 (owaowata, 2026-09-07 05:26 UTC): 0 comments, no host reply. |
| No external-data disclosure clause, thread or deadline | CONFIRMED | Rules text has 0 hits for "disclos". Pinned topics are only 733965, 733375, 733343 and 730709. The forum search "disclosure" returns one unrelated thread (a submission failure). |
| Rules 1.4 slug typo "abnormalities" | CONFIRMED | |

Clauses the research did not surface in its summary (all MEASURED, same URL):
- 3.4.d: Sponsor "reserves the right to disqualify any entrant who does not follow these Rules". This is not limited to prize winners.
- 3.8.e: a disqualified Participant may be removed from the leaderboard, and "Kaggle points or medals, may also not be awarded". INFERRED: this undercuts the practical line that a medal-only finish is safer. Enforcement below the top 10 is unlikely in practice, but the rules permit it.
- 3.9.b: on non-compliance the Sponsor may disqualify or require remediation within one week, "including ... the resolution of license conflicts". This is a possible cure path, at the Sponsor's discretion.
- 3.14.a: the winner warrants they are "the sole and exclusive owner and rights holder of the Submission" and can "grant all required licenses", with no infringement of "confidentiality obligations". INFERRED: this is where a dataset-agreement breach (ND, no-redistribution) would bite at the prize stage.
- 3.6.b: public sharing of Competition Code must not violate third-party IP. 3.6.c (OSI code licenses) covers code. Competitor Anirudh (745283) cites a "Rule 6.c" for external models, which misreads it; the researcher already noted this.

## 2. Overview and Data pages, MEASURED

- Overview (https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview): "Freely & publicly available external data is allowed, including pre-trained models". CONFIRMED.
- Winners' asks (i) video, (ii) "publish a link to your open sourced code and the weights on the competition forum", (iii) "Share final version of model as publicly available for open distribution and validation". Winners' Requirement Deadline is November 5, 2026. CONFIRMED.
- Data page (https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data): no external-data statement (0 hits for "external"). License: "Subject to Competition Rules". CONFIRMED.

## 3. Host / staff statements (author role verified on each page)

| Item | URL | Author / badge | Date (UTC) | Status |
|---|---|---|---|---|
| KneeCoT banned: "the KneeCoT dataset is not allowed in this competition." | .../discussion/734109 | Po-Hao "Howard" Chen, COMPETITION HOST | 2026-08-25 11:00:25 | CONFIRMED. Question by Mazzutti 2026-08-09 23:34:53. The host said the stance may change if the HF policy changes. HF page today (https://huggingface.co/datasets/YiHui0124/KneeCoT, via WebFetch): still gated, still requires "a formal ethical use agreement with the affiliated hospital". So the ban still applies. |
| Non-commercial licenses "would not be excluded by the competition rules solely" for that | .../discussion/733965 | same, COMPETITION HOST | 2026-08-27 22:14:33 | CONFIRMED. Nuance the research missed: the reply opens "use of these datasets would be considered non-commercial", answering UGUR OZER (2026-08-14 21:30 UTC), who named OAI, MRNet, fastMRI+ and SKM-TEA. So on the non-commercial point the host did address those four together. Accessibility was not ruled per dataset. The host frames this "From the competition's perspective" and says RSNA/Kaggle "cannot interpret or waive" third-party agreements. |
| Click-through access "would generally satisfy this requirement"; institution-specific approvals, negotiated agreements, IRB, lengthy credentialing "may present a meaningful accessibility barrier" | same | same | same | CONFIRMED. Missed qualifier: "particularly given the remaining competition timeline". The barrier test is partly time-based. |
| "Teams remain responsible" incl. "any required release of code or models" | same | same | same | CONFIRMED. INFERRED: this is the closest thing to a host answer on the prize-stage question. The host treats model release as a requirement teams must check against each dataset's terms, which leans against reading 2.5.a as excusing public weights. |
| OAI conditional: "without an escalated (i.e. institutional) review and sign-off or access to a legal team, and is free of charge" | .../discussion/741819 | same | 2026-09-21 15:34:26 | CONFIRMED. Question by John Doe 2026-09-17 17:03:14 (rank now 197th, was 193rd). |
| Pinned "Use of Commercially Hosted LLMs" topic; later "You can use LLM API, such as those from OpenAI..." and "Yes." on LLM-derived labels/embeddings | .../discussion/733965 | same | topic 2026-08-09 09:52:50; replies 2026-08-27 21:59:45 and 22:05:41 | CONFIRMED |
| Host profile: "Discussion (19)", newest "Comment on Is the Osteoarthritis Initiative (OAI) permitted..." 13d ago | https://www.kaggle.com/javaduke95/discussion | profile | read 2026-10-04 | CONFIRMED. Profile also says "last seen in the past day": the host is active but has not answered 743416, 743420 or 733652. |
| Pinned host topics 733343 (Overview) and 733375 (Welcome) | .../discussion/733343, .../733375 | COMPETITION HOST | 2026-08-06 | Checked: no external-data, pretrain or license content. |
| Kaggle staff María Cruz, getting-started/Discord thread | .../discussion/730709 | KAGGLE STAFF | 2026-07-30 03:29:19 | CONFIRMED. Says staff and hosts "will not monitor Discord". Her profile feed (https://www.kaggle.com/macruzbar/discussion) shows no other post in this competition in recent months. |
| Kaggle staff Ryan Holbrook (named in the citation) | https://www.kaggle.com/ryanholbrook/discussion?query=knee | profile search | read 2026-10-04 | Search "knee" returns no posts (the search works: "LoRA" returns hits). No staff statement from him. |

Supersession: no later host post contradicts H3 (KneeCoT), H6 (08-27 test) or H7 (OAI). CONFIRMED.

## 4. Competitor threads, MEASURED (all opinions, none host)

- 743416 YYama (9th), 2026-09-26 01:52:06, 62 votes. CONFIRMED. Minor: the thread header says "12 Comments", the list view says 13, and 13 comment items are visible. No host reply. Last comment Eugene Khvedchenya 2026-10-03 16:53:51.
- Ali Ozan Memetoglu (289th), 2026-09-30 19:08:19: MRNet agreement bars "create derivative works from the dataset". The quote is CONFIRMED as a competitor claim. I could not read the MRNet agreement myself: the AIMI page (https://aimi.stanford.edu/datasets/mrnet-knee-mris) shows "Terms & Conditions" and "License" tabs with no text in the DOM, and Redivis needs sign-in. So the underlying term is UNVERIFIED by me (the researcher's raw notes say they read it on the older AIMI Azure page).
- 733652 Mazzutti (20th), 2026-08-20 09:27:03: disjunction argument; lists "MRNet, fastMRI/fastMRI+, OAI, SKM-TEA and KneeCoT". CONFIRMED. KneeCoT was banned 5 days later, so his reading is partly superseded. Host reply in that thread (2026-08-09 09:57:21) only points to 733965. Amanie El-Hajj bump 2026-10-02 23:37:33 reads 741819 as a yes ("they said we can use external").
- 741819 Tucker Arrants (now 45th), 2026-09-21 20:56:27: "It is a no, as it requires institutional sign off." CONFIRMED. Correction: in 741819 he only speaks generally about "a no derivatives clause/license". The naming of "MRNet, KneeMRI" as ND comes from his 737950 comment, 2026-09-01 18:15:18.
- 737950 Ercan Gurvit, 2026-08-28 10:15:00: says OAI needs a DUC "from an institution holding an active Federal Wide Assurance". This is a competitor claim and UNVERIFIED. 9 comments, no host reply.
- 737950 Fernando Faria, 2026-09-06 03:34:04: fastMRI "requires emailing NYU and waiting on an approved data-sharing agreement". The quote is CONFIRMED, but its substance is PARTLY REFUTED by the fastMRI page (section 5).
- 743420 Richard_61 (40th), 2026-09-26 03:15:55, follow-up 2026-09-29 02:15:55. No host reply. CONFIRMED.
- 735121 DK2LONE, 2026-08-14 03:57:10; bump by quant_kiid 2026-10-03 03:03:35. No host reply. CONFIRMED.
- 735497 (2026-08-16 09:52:30) and 738111 (2026-08-29 16:33:29): 0 comments each. 738156 (2026-08-30 05:53:48): 1 deleted comment. CONFIRMED.
- 744056 Shalini S, 2026-09-28 13:16:14. 745283 Anirudh, 2026-10-02 22:22:49. 745666 Kuan-Chieh Huang, 2026-10-03 15:05:55. 740718 oshiko0450, 2026-09-10 21:40:23. None has a host reply. CONFIRMED.
- Forum searches "external", "MRNet", "license" and "disclosure" (2026-10-04): nothing newer than the researcher's list, apart from unrelated 736170 (tennogh sharing his own knee MRI, CC0).

## 5. Dataset-term checks (the "misread agreement" angle)

- KneeMRI (https://zenodo.org/records/14789903, WebFetch): "CC BY-NC-ND 4.0". Access type Open, no registration. 917 volumes, 3 ACL classes. Files: metadata.csv, volumetric_data.7z, .pck example. MEASURED. This upgrades the researcher's "a competitor lists" to a direct check.
  - KneeMRI clearly passes the host's access test.
  - Its risk is ND at publication. Creative Commons' own guide (https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/) says ND is "triggered only when works or adaptations of works are publicly shared". Its conservative practice line says ND content should "not be used as training data". MEASURED.
  - INFERRED: the risk concentrates at the winner's public-weights step (2.8.a plus Overview item iii).
- Kaggle dataset ibrahimshaikhdev/knee-mri-917 (https://www.kaggle.com/datasets/ibrahimshaikhdev/knee-mri-917): MEASURED.
  - "CC0: Public Domain"; "No description available".
  - Updated 2026-10-03 10:59:34 UTC; "Version 1 (6.67 GB)"; 918 files.
  - metadata.csv (42.41 kB): examId, seriesNo, aclDiagnosis (690/172/55 = 917), kneeLR, roi*.
  - Provenance SOURCES and CITATIONS are empty.
  - INFERRED, strong: this is a re-upload of KneeMRI (same count, same 3 classes, same metadata.csv + volumetric_data + .pck layout). The CC0 label cannot override CC BY-NC-ND. The 4.1 GB in the task brief is likely the compressed download size.
- fastMRI (https://fastmri.med.nyu.edu/, curl): MEASURED.
  - Access is an online form with "electronic signature" (Full Name, Email, Institution), not emailing NYU.
  - The page says the application "must include the investigator's institutional affiliation and the proposed uses".
  - Use limited to "internal research or educational purposes only"; "Not SELL OR OTHERWISE MONETIZE".
  - No distribution of "data files derived from the fastMRI Dataset", except to supervised individuals and for academic publications.
  - "must register as an individual user".
  - Also "obtaining all approvals for the proposed use ... institutional policies".
  - INFERRED: borderline on the host's access test, because the institution field is required and approval steps are unclear. Derived-file distribution conflicts with public weights.
- OAI (https://www.oaiknowledgebase.com/about/FAQs/ in browser; the OAI CORE guide PDF "Version 1.0, June 3, 2026", read page by page): MEASURED.
  - Steps: "Create an NDA account and complete the Data Use Certification (DUC)", "Login via Login.Gov", "Request Access" to the "Osteoarthritis Initiative" permission group.
  - Neither page mentions an institution, a Signing Official or an FWA.
  - Warning: WebFetch's summary of that PDF claimed "a Signing Official from your institution must sign". The PDF itself contains no such sentence, so that summary was a hallucination and I did not use it.
  - NDA pages (https://nda.nih.gov/oai/accessing_images) redirect to a sign-in wall, and nda.nih.gov/nda/access-data-info rendered no content. I could not confirm the permission-group type.
  - Verdict on the research's "OAI through NDA likely fails": UNCERTAIN / OVERSTATED. The research's own raw notes (notes_raw.md) say NDA Open Access groups need no institutional sponsorship.
- MRNet / SKM-TEA agreement text: not verifiable today without sign-in (see section 4).

## 6. Useful precedent (MEASURED)

- Deepfake Detection Challenge final results (https://www.kaggle.com/competitions/deepfake-detection-challenge/discussion/157925). Host account "Mozaic", COMPETITION HOST, 2020-06-12 14:47 UTC: the top two teams "used external data sources in their winning submissions that were not allowed". The prior first place was kept at 7th after "an extensive review of the context and licensing" of its external data.
- This confirms the precedent competitor Eugene Khvedchenya cites: post-deadline, license-based review of external data has removed prize places before.

## 7. Bottom line for Eesh (INFERRED)

- Allowed in principle: rules 2.6.b and the Overview.
- Banned by name: KneeCoT only (host, 2026-08-25). Still in force.
- The host's test is non-commercial OK, click-through OK, institutional sign-off or legal review not OK. No per-dataset yes/no exists, and the host has been silent since 2026-09-21 while still active on Kaggle.
- Click-through or open sets: KneeMRI (fully open), MRNet and SKM-TEA (individual click-through) probably pass the access test.
  - All carry derivative or redistribution limits that collide with the public-weights duty (2.8.a, Overview item iii) and the 3.14.a warranty.
  - The host has put that on teams.
  - Rules 3.4.d and 3.8.e let enforcement reach any entrant, not only prize winners.
- fastMRI: borderline on access, and conflicts on derived files.
- OAI: unresolved. It depends on whether the NDA OAI DUC is individual (the documented flow suggests it is) or institutional.
