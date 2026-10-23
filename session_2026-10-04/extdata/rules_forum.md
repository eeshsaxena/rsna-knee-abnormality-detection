# RSNA Knee Abnormality Detection: external data, official rules + forum (read 2026-10-04)

Method: Claude built-in browser, NOT signed in to Kaggle (all pages below render read-only without login). Nothing downloaded, no forms, no posts, no votes. Times converted to UTC from the page's hover timestamps. Tags: MEASURED = read on the page; INFERRED = my reading.
Competition slug: rsna-knee-abnormality-detection (note: rules 1.4 prints the website as ".../rsna-knee-abnormalities-detection", a typo; the live slug is "abnormality").

## 0. Bottom line (short)

1. The rules ALLOW external data and pretrained models in principle. Official wording: 2.6.b "acceptable unless specifically prohibited by the Host"; Overview > Code Requirements: "Freely & publicly available external data is allowed, including pre-trained models". MEASURED.
2. The ONLY dataset the host has explicitly banned is KneeCoT (host Po-Hao "Howard" Chen, COMPETITION HOST, 2026-08-25). MEASURED.
3. The host's general test (2026-08-27, pinned thread 733965): a non-commercial license is NOT by itself disqualifying; "straightforward registration or click-through" access generally passes; institution-specific approvals, negotiated written agreements, IRB, lengthy credentialing "may present a meaningful accessibility barrier". MEASURED.
4. OAI (2026-09-21): allowed ONLY "if it is generally accessible to all participants without an escalated (i.e. institutional) review and sign-off or access to a legal team, and is free of charge". This is conditional, not a yes. MEASURED. If NDA access to OAI really needs an institution-signed Data Use Certification (several competitors say so), OAI fails the host's own test. INFERRED.
5. No host yes/no exists for MRNet, SKM-TEA, fastMRI/fastMRI+, KneeMRI, KneeXNet-2.5D, OAIZIB-CM, or the RadImageNet weights. The 62-vote thread 743416 asking for a per-dataset ruling has NO host reply as of 2026-10-04. MEASURED.
6. The host explicitly put dataset-license compliance, INCLUDING the required public release of winning code/weights, on the teams ("Teams remain responsible..."). The winners' obligation conflict (2.5.a carve-out vs 2.8.a public weights) for no-derivatives / no-redistribution data is UNANSWERED (threads 743420, 737950, 735121, 739758, 741212, 744056). MEASURED.
7. There is NO required "External Data Disclosure" thread or disclosure deadline in the rules, and no such pinned thread exists. MEASURED. Winners must still deliver training code and a reproducible description (2.5.b, 2.8.a), so external data would have to be disclosed at the winner stage. INFERRED.
8. The host account has posted nothing since 2026-09-21 (profile javaduke95 lists 19 posts; newest is the OAI reply "13d ago"). MEASURED.

Practical reading (INFERRED, for Eesh): click-through, free, non-geo-restricted knee MRI sets (MRNet, SKM-TEA, KneeMRI) are probably inside the host's accessibility test, but they carry a separate, unresolved risk at the prize stage (no-derivatives / research-use terms vs. the requirement to publish weights). OAI via NDA likely fails the host's institutional-sign-off test. KneeCoT is banned. For a non-prize (medal) finish the practical risk is lower, since detailed review targets prize winners, but nothing official says that.

## 1. Official rules (https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)

Section numbering below follows the page: "1. Competition-specific terms", "2. Competition-specific rules", "3. General competition rules". All MEASURED unless marked.

- 1.6 WINNER LICENSE TYPE: "CC-BY-NC 4.0".
- 1.7 DATA ACCESS AND USE: "Commercial and Academic Research - MIRA license".
- 2.4.a Competition Data: may be used for any purpose incl. commercial, subject to http://rsna.org/mira-license. Sponsor may disqualify for use outside the Rules.
- 2.4.b Data Security: must not "transmit, duplicate, publish, redistribute" Competition Data to non-participants. (Host later clarified that sending report text to an LLM API is not by itself prohibited sharing, thread 733965.)
- 2.5.a WINNER LICENSE: winning Submission and source code licensed under CC-BY-NC 4.0, described in the same sentence as an OSI-approved license that "in no event limits commercial use" (internally contradictory; raised in thread 739905, unanswered).
  - Carve-out 1: generally commercially available software procurable without undue expense need not be licensed.
  - Carve-out 2 (key for external data): if "input data or pretrained models with an incompatible license" are used, you need not grant an open-source license for that data and/or model(s). Competitors call this "2.5.a.4".
- 2.5.b Winners may be required to give a detailed, reproducible description (architecture, preprocessing, loss, training details, hyper-parameters) plus "a link to a code repository" with complete instructions, and may join a recorded/panel call.
- 2.6 EXTERNAL DATA AND TOOLS
  - 2.6.a: External Data may be used. It must be "publicly available and equally accessible to use by all Participants ... at no cost", OR satisfy the Reasonableness criteria of 2.6.b. Using External Data "does not limit your other obligations", including 2.8 Winners Obligations.
  - 2.6.b: "The use of external data and models is acceptable unless specifically prohibited by the Host." Use must be "reasonably accessible to all" and of "minimal cost". Host applies a "Reasonableness Standard" based on "cost thresholds and accessibility". Example: small LLM subscription (Gemini Advanced) OK; buying a proprietary dataset costing more than a prize is not. (Text cites "Sec. 8.2", a template artifact; no 8.2 exists in this rule set.)
  - 2.6.c AMLT: allowed if you hold an appropriate license.
  - No competition-specific clause names any external dataset, LLM, or external-data disclosure thread.
- 2.8.a WINNER'S OBLIGATIONS: deliver final model code "including model weights", training code, inference code, environment description. "The model weights should be provided as a public kaggle dataset" (competitors call this "2.8.a.1"). Also grant the 2.5 license and "represent that you have the unrestricted right to grant that license".
- 3.4.b: no hand labeling / human prediction of validation or test records.
- 3.5.d / 3.6.a: no private sharing of code or data outside teams; 3.6.b public sharing must be on Kaggle forums/notebooks.
- 3.6.c Use of Open Source: unless otherwise stated in the specific rules, open source CODE used must be OSI-approved and not limit commercial use. (INFERRED: this clause is about code; a competitor in 745283 reads it as covering model weights. The 2.5.a carve-out arguably is the "otherwise stated" exception for data/models.)
- 3.9.b: non-compliant winners may be disqualified or given one week to remediate, including "the resolution of license conflicts".
- 3.14.a: you warrant the Submission does not infringe third-party rights, incl. "confidentiality obligations". (INFERRED: this is where a dataset-agreement breach would bite.)

## 2. Overview and Data pages

- Overview > Code Requirements (https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview): "Freely & publicly available external data is allowed, including pre-trained models". Internet disabled, 9 h CPU/GPU. MEASURED.
- Overview > Prizes: beyond standard obligations, host asks winners to (i) make a video, (ii) publish link to open-sourced code and weights on the forum, (iii) "Share final version of model as publicly available for open distribution and validation". MEASURED. INFERRED: (ii)/(iii) are what make no-derivatives / no-redistribution training data risky for prize winners.
- Timeline: Entry and Team Merger deadline 2026-10-15; Final Submission 2026-10-22; Winners' Requirement Deadline 2026-11-05 (training code, video, method description). MEASURED.
- Citation lists Kaggle staff Ryan Holbrook and María Cruz among authors. MEASURED.
- Data page (https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data): no statement about external data. License field: "Subject to Competition Rules". MEASURED.

## 3. Host / Kaggle-staff statements (complete list relevant to external data)

Host account: Po-Hao "Howard" Chen, profile /javaduke95, badge "COMPETITION HOST" on every post below (Radiologist, Cleveland Clinic; also Challenge Organizing Team lead on the Overview). His profile shows 19 discussion posts; I read all of them in the profile feed; only the items below concern external data/models.

| # | Date (UTC) | Thread | Badge | What it says (short quote) | Dataset |
|---|---|---|---|---|---|
| H1 | 2026-08-09 09:52 | 733965 (PINNED topic "Use of Commercially Hosted LLMs") | COMPETITION HOST | Hosted LLMs/APIs permitted if "reasonably accessible to all participants and of minimal cost"; host reserves right to judge accessibility/cost/unfair advantage | LLM services (not datasets) |
| H2 | 2026-08-09 09:57 and 09:59 | 733652 and 733873 | COMPETITION HOST | "This post was just added and hopefully adds clarity." (points to 733965; does NOT answer the dataset part of 733652) | none |
| H3 | 2026-08-25 11:00 | 734109 | COMPETITION HOST | "the KneeCoT dataset is not allowed in this competition." Reason: required data agreement with the institution makes the field uneven; may change if HF policy changes | KneeCoT (HF YiHui0124/KneeCoT) |
| H4 | 2026-08-27 21:59 | 733965 | COMPETITION HOST | "You can use LLM API, such as those from OpenAI" to label reports | LLM API |
| H5 | 2026-08-27 22:05 | 733965 | COMPETITION HOST | LLM-derived labels/embeddings OK at train and inference ("Yes.") | LLM outputs |
| H6 | 2026-08-27 22:14 | 733965 (reply to UGUR OZER re OAI/MRNet/fastMRI+/SKM-TEA research-only agreements) | COMPETITION HOST | Non-commercial licenses: "not be excluded by the competition rules solely" for that. Click-through/straightforward registration "would generally satisfy this requirement". Institution-specific approvals, negotiated agreements, IRB, lengthy credentialing "may present a meaningful accessibility barrier". "Teams remain responsible" for license compliance incl. any required release of code or models | Category: OAI, MRNet, fastMRI+, SKM-TEA named in the question, not individually ruled |
| H7 | 2026-09-21 15:34 | 741819 | COMPETITION HOST | OAI "does not violate" the rules IF accessible "without an escalated (i.e. institutional) review and sign-off or access to a legal team" and free; host "cannot comment" on OAI's own data-use terms | OAI |

Kaggle staff: María Cruz (/macruzbar), badge "KAGGLE STAFF", pinned 730709 "How to get started + Competition's Official Discord" (2026-07-30 03:29 UTC): no external-data content; says staff/hosts do not answer questions on Discord. Her profile shows no other post in this competition. MEASURED. No other Host/Staff account found in any external-data thread. MEASURED.

Supersession: no later host post contradicts an earlier one. H6 generalizes H3's reasoning (institutional agreement = access barrier); H7 applies the same test to OAI. Overview's "Freely & publicly available" line is consistent. The only "superseding" layer is that H3/H6/H7 narrow 2.6.b's "acceptable unless specifically prohibited" by adding the equal-accessibility test the host says it will apply. INFERRED.

## 4. Thread-by-thread (all https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/<id>)

- 733652, Fernando Faria (208th), 2026-08-07 18:25. Asks if click-through research-use sets (MRNet, fastMRI+, OAI, SKM-TEA) count as "equally accessible at no cost", and about LLM APIs. Host reply only points to 733965 (H2). Competitor opinions:
  - NNMax (08-08): public datasets usually allowed regardless of license unless rules say so; mention NC in write-up. (competitor opinion)
  - Mazzutti (20th), 2026-08-20 09:27: 2.6(a) is a disjunction; under 2.6.b these sets are "free, not geo-restricted" and clear the Reasonableness branch unless Host prohibits. Also asks whether NC/ND-trained weights satisfy 2.8's "unrestricted right to grant". (competitor opinion; NOTE he listed KneeCoT among them, and KneeCoT was later banned on 08-25, so his reading is partly superseded by H3/H6)
  - Amanie El-Hajj, 2026-10-02 23:37: "still no answer from the competition host". (competitor)
- 733965, host topic (pinned), 2026-08-09; 13 comments. Host replies H4, H5, H6. Competitors: Nicolai Karcher (08-09) and UGUR OZER (08-14) asked the dataset question that H6 answers.
- 734109, Mazzutti, 2026-08-09 23:34. KneeCoT, gated HF, CC BY-NC 4.0, requires formal ethical-use agreement with affiliated hospital. Host H3 bans it (2026-08-25). Mazzutti accepts (08-25 14:43).
- 735121, DK2LONE, 2026-08-14 03:57. RadImageNet ResNet-50 (CC-BY-NC-SA-4.0) vs winners' open-license obligation. Bump by quant_kiid 2026-10-03 03:03. NO host reply.
- 735497, Demir Poyraz Elcin, 2026-08-16 09:52. Registration-gated sets (MRNet, OAI) under 2.6(a)? 0 comments. NO host reply.
- 737950, Ercan Gurvit, 2026-08-28 10:15. Does KneeCoT ruling generalize to OAI (DUC from an institution with Federal Wide Assurance) and MRNet (Research Use Agreement)? 9 comments, NO host reply. Competitor views:
  - Tucker Arrants (43rd), 2026-09-01 18:15: NC OK per host, but "ND licenses are a problem"; says MRNet, KneeMRI carry ND. (competitor opinion)
  - Fernando Faria, 2026-09-06 03:34: access patterns per competitor: MRNet = click-through RUA on AIMI portal; fastMRI = "requires emailing NYU" and an approved data-sharing agreement; OAI = signed Data Use Certification; SKM-TEA = same as MRNet. (competitor claims, not verified here)
  - Doublej37 / junicus (09-01, 09-03): public checkpoints trained on private data (MRI-CORE) unanswered.
- 738111, pizzaboy, 2026-08-29 16:33. Free but signed research-use agreements with NC + no-derivative clauses (Stanford AIMI). 0 comments. NO host reply.
- 738156, "Yet, WiaLive !", 2026-08-30 05:53. Ungated OAI-derivative OAIZIB-CM on HF (CC BY-NC 4.0). 1 comment, deleted. NO host reply.
- 739758 (09-06), 739905 (09-07), 741212 (09-13): mixed upstream weight licenses; CC-BY-NC vs "no commercial limits" contradiction and Apache-2.0 fine-tuning; public checkpoint trained by others and training-code obligation. All 0 comments, NO host reply.
- 741819, "John Doe" (193rd), 2026-09-17 17:03. OAI yes/no. Host H7 (2026-09-21 15:34). Competitors: Optimo "It's difficult to be less clear"; Tucker Arrants (2026-09-21 20:56): "It is a no, as it requires institutional sign off." and claims virtually all useful external sets fail either accessibility or ND. (competitor opinion)
- 743416, YYama (9th), 2026-09-26 01:52, 62 votes. Per-dataset yes/no asked for: Stanford MRNet (RUA), Stanford SKM-TEA (registration + DUA), NYU fastMRI / fastMRI+ (DSA for images; annotations separate on GitHub microsoft/fastmri-plus), OAI (DUC + institutional), KneeMRI (CC BY-NC-ND 4.0, zenodo 14789903), KneeXNet-2.5D (OAI-derived, no explicit license, github pitthexai/Knee_MRI_Segmentation_2.5D). Also asks training-on-images vs fine-tuning released weights, and compatibility with the model-release requirement. 13 comments, NO host reply (latest comment Eugene Khvedchenya 2026-10-03 16:53). Competitor views:
  - Ali Ozan Memetoglu (289th), 2026-09-30/10-01: MRNet agreement bars modify/reverse engineer/"create derivative works". (competitor claim)
  - Eugene Khvedchenya (565th): hosts usually avoid dataset-clearance answers; "common sense": public registration ok, NC ok, academia-email-gated less ok. Cites DFDC (discussion 157925) where top-2 teams were removed for disallowed external data after the host never answered. (competitor opinion)
  - Tucker Arrants: only top-10 get audited, so unclear rules reshape non-prize ranks; external data "can substantially increase score". (competitor opinion)
  - Cemal Atas (23rd): asks about MIT-licensed public models trained on OAI-derived data. (unanswered)
- 743420, Richard_61 (40th), 2026-09-26 03:15 (+follow-up 09-29 02:15). Does 2.5.a.4 exempt weights trained on no-redistribution data (KneeMRI CC BY-NC-ND, MRNet RUA section 6, fastMRI DSA) from 2.8.a.1 public weights? If not, are such solutions prize-ineligible? NO host reply.
- 744056, Shalini S, 2026-09-28 13:16. Prize eligibility of public weights trained on competition data by other participants (e.g. CoAtNet weights mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2, CC0 per Lutfiya). NO host reply.
- 745283, Anirudh, 2026-10-02 22:22. RadImageNet ResNet-50: CC-BY-NC-SA-4.0 on Kaggle dataset marwanmath/resnet-50-radimagenet-marwan vs MIT on HF Lab-Rasool/RadImageNet. 0 comments.
- 745666, Kuan-Chieh Huang, ~2026-10-03. torchvision ConvNeXt-Tiny IMAGENET1K_V1 eligibility. 0 comments (from list view).
- 740718, oshiko0450, 2026-09-10. Hand-labeling unlabeled TRAIN studies: competitors (ParsaAshrafi, PC Jimmmy) say this is derived data, not External Data, and 3.4.b only covers val/test. NO host reply.

Searches run on the discussion list: "external", "external data", "fastMRI", "MRNet", "OAI", "SKM-TEA", "pretrained", "license", "disclosure", "KneeMRI". All hits are listed above; no host/staff post outside H1 to H7.

## 5. Kaggle dataset ibrahimshaikhdev/knee-mri-917 (https://www.kaggle.com/datasets/ibrahimshaikhdev/knee-mri-917)

- Owner Ibrahim Shaikh, "Updated a day ago" = 2026-10-03 10:59 UTC; Version 1 shown as 6.67 GB (Data Explorer; the 4.1 GB figure may be compressed size); folder volumetric_data (918 files) + metadata.csv (42.41 kB). "No description available". License field: "CC0: Public Domain". 3 downloads, 2 views at read time. MEASURED.
- metadata.csv columns: examId, seriesNo, aclDiagnosis (values 0/1/2), kneeLR, roiX, roiY, roiZ, roiHeight, roiWidth, roiDepth (11th column not shown); 917 rows (aclDiagnosis histogram 690 + 172 + 55). MEASURED.
- INFERRED: this schema and count (917 exams, ACL 0/1/2) match the KneeMRI dataset (Rijeka, Stajduhar et al.), which 743416 lists as CC BY-NC-ND 4.0 on zenodo. If so, the CC0 label is a re-license by an uploader without provenance. Using this copy does not cure the original ND/NC terms; it also adds a provenance risk. Not discussed in any forum thread found.

## 6. Dataset status summary (competition-rules angle only)

| Dataset | Host ruling | Host test applied (INFERRED) | Extra risk at prize stage (INFERRED) |
|---|---|---|---|
| KneeCoT | BANNED (H3, 2026-08-25) | n/a | n/a |
| OAI (NDA) | Conditional (H7, 2026-09-21) | Likely fails if DUC needs institutional sign-off (per competitors) | DUA terms; host won't interpret |
| MRNet (Stanford AIMI) | None | Click-through RUA: likely passes accessibility (H6) | RUA reportedly bars derivative works vs public weights (2.8.a) |
| SKM-TEA (Stanford AIMI) | None | Same pattern as MRNet per competitors | Same as MRNet |
| fastMRI images (NYU) | None | Email + approved DSA: borderline "negotiated written agreement" | DSA reportedly bars distributing derived files |
| fastMRI+ annotations | None | Public GitHub labels, but images need fastMRI | depends on fastMRI |
| KneeMRI (zenodo) | None | Open download, NC ok per H6 | CC BY-NC-ND vs releasing derived weights |
| KneeXNet-2.5D | None | Public GitHub, OAI-derived | no license file; OAI provenance |
| OAIZIB-CM (HF, ungated) | None (738156 unanswered) | Ungated passes accessibility on its face | OAI provenance; CC BY-NC |
| RadImageNet weights | None (735121, 745283 unanswered) | Public download | CC-BY-NC-SA vs MIT discrepancy |
