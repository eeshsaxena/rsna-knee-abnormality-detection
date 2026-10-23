# Forum thread notes (read 2026-10-04, browser, signed-out view; times converted from IST title attrs to UTC)

## 743416 "Please clarify which external knee MRI datasets are allowed"
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416
- Author YYama (shown "9th in this competition"), posted 2026-09-26 01:52 UTC, 62 votes, 12 comments. Tags @javaduke95.
- Lists: Stanford MRNet (research-use agreement), SKM-TEA (registration + DUA), NYU fastMRI/fastMRI+ (data-sharing agreement for images; annotations separately released), OAI (Data Use Certification + institutional reqs), KneeMRI (CC BY-NC-ND 4.0, zenodo 14789903), KneeXNet-2.5D (OAI-derived, github pitthexai, no explicit license).
- Summarises host positions: KneeCoT ruled not allowed (734109); non-commercial terms alone do not exclude a dataset (733965); registration vs substantial institutional approval may be treated differently.
- Links: 733652 (original question), 735497 (registration-gated), 737950 (institutional-agreement follow-up), 735121 (pretrained weights), 738111 (research-use agreements), 734109 (KneeCoT), 733965 (non-commercial).
- NO host/staff reply visible in this thread as of 2026-10-04 (all 12 comments are competitors).
- Competitor comments: Tucker Arrants (44th) 2026-09-26: only top-10 likely audited; external data "can substantially increase score". YYama: external data could let a simple approach get a dramatic improvement. Ali Ozan Memetoglu (289th) 2026-09-30/10-01: MRNet agreement forbids derivative works. Cemal Atas (23rd) 2026-09-27: asks about MIT-licensed pretrained models trained on OAI-derived data. Eugene Khvedchenya (565th) 2026-10-01: hosts often ignore clearance Qs; cites DFDC 157925 where top-2 removed for disallowed external data. Optimo (26th) 2026-10-01: has not tried external data yet.

## 741819 "Is the Osteoarthritis Initiative (OAI) permitted as external data?"
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819
- Author "John Doe" (193rd), posted 2026-09-17 17:03 UTC.
- HOST reply: Po-Hao "Howard" Chen, badge COMPETITION HOST, 2026-09-21 15:34 UTC: OAI use "does not violate" rules "if it is generally accessible to all participants without an escalated (i.e. institutional) review and sign-off or access to a legal team, and is free of charge." Also: host does not own/control OAI and cannot comment on compliance with OAI data-use terms.
- Competitors: Optimo (30th) "It's difficult to be less clear"; Tucker Arrants (42nd, 2026-09-21): reads it as a NO since OAI needs institutional sign-off; says virtually all useful external knee MRI sets fail either accessibility or no-derivatives.

## 734109 "Is the gated KneeCoT dataset permitted as external data?"
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109
- Author Mazzutti (20th), posted 2026-08-09 23:34 UTC. KneeCoT = HF gated YiHui0124/KneeCoT, CC BY-NC 4.0, needs formal ethical-use agreement with affiliated hospital.
- HOST reply Po-Hao "Howard" Chen (COMPETITION HOST) 2026-08-25 11:00 UTC: "decided to not allow the use of KneeCoT dataset due to the required data agreement with dataset institution" ... "the KneeCoT dataset is not allowed in this competition." Reason: uneven playing field for those who can/cannot complete formal agreement. May change if HF policy changes.
- Mazzutti accepted (2026-08-25).

## 733652 "Rules clarification: external knee-MRI datasets, and using an LLM API to derive labels"
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733652
- Author Fernando Faria (208th), posted 2026-08-07 18:25 UTC, 23 votes. Asks if MRNet, fastMRI+, OAI, SKM-TEA (free, click-through research agreement) count as "publicly available and equally accessible at no cost".
- HOST Po-Hao "Howard" Chen reply ~2026-08-09: "This post was just added and hopefully adds clarity." (links to a new post; k256.dev says it is 733965; this host reply addresses the LLM-API question, not external datasets).
- Mazzutti (20th), 2026-08-20 09:27 UTC: argues 2.6(a) is a disjunction; MRNet/fastMRI/OAI/SKM-TEA/KneeCoT free, not geo-restricted, click-through, so clear 2.6(b) unless Host specifically prohibits. Raises 2.8 "unrestricted right to grant" issue for NC/ND-trained weights. (KneeCoT was later banned by host on 08-25, so his argument was partly superseded.)
- NNMax (no rank shown) ~2026-08-08: public datasets generally allowed unless rules say otherwise (competitor opinion).
- Amanie El-Hajj (1313th) 2026-10-02: "still no answer from the competition host" on the category question.

## 733965 "Use of Commercially Hosted LLMs" (HOST post)
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965
- Author Po-Hao "Howard" Chen, COMPETITION HOST badge, posted 2026-08-09 09:52 UTC, 22 votes. Hosted LLM APIs for label extraction permitted under 2.6(b); host reserves right to judge accessibility/cost/unfair advantage.
- UGUR OZER (3206th) ~2026-08-14: asks about OAI/MRNet/fastMRI+/SKM-TEA (free, click-through, non-commercial).
- HOST reply Po-Hao "Howard" Chen, approx 2026-08-27 22:14 UTC (timestamp mapping from DOM order; between 21:59 and 22:14 UTC on 08-27):
  * prize money does not by itself make use commercial; NC-only datasets "would not be excluded by the competition rules solely because" of NC license.
  * "Datasets that can be accessed through a straightforward registration or click-through data-use agreement would generally satisfy this requirement." (accessibility)
  * Institution-specific approvals, negotiated agreements, IRB, lengthy credentialing "may present a meaningful accessibility barrier".
  * Teams remain responsible that use, including prize competition and required release of code/models, is permitted by the dataset license.
- This is the most general host statement on external datasets. Later host statements (08-25 KneeCoT ban; 09-21 OAI conditional) are consistent with it (institutional agreement = not OK). No later post found that supersedes it.

## 735497 "Request for official clarification: Rule 2.6(a) and registration-gated public datasets (MRNet, OAI, etc.)"
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735497
- Demir Poyraz Elcin (3063rd), 2026-08-16 09:52 UTC, 5 votes, 0 comments. No host answer.

## 737950 "External data behind an institutional data use agreement - does the KneeCoT ruling generalize?"
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737950
- Ercan Gurvit (2623rd), 2026-08-28 10:15 UTC. 9 comments, NO host reply.
- Tucker Arrants (43rd) ~2026-09-01: NC ok per host but ND is a problem; says MRNet, KneeMRI carry ND-type restriction.
- Cody_Null (2459th): has not tried external data; curious if top teams use it.
- Fernando Faria (208th) ~2026-09: access patterns: MRNet = click-through RUA on Stanford AIMI portal; fastMRI = email NYU, wait for approved data-sharing agreement; OAI = signed Data Use Certification; SKM-TEA = same as MRNet. (competitor description, not verified by me)
- SpeedSci (122nd): local training on external data unlikely to be detectable. Steve421471 (105th): winners must be replicable.
- Doublej37 / junicus (1147th): ask about MRI-CORE checkpoint (public, trained on private clinical data, Apache-2.0 in README only). No host answer.

## 735121 "Are CC-BY-NC pretrained weights compatible with the winners open-licence obligation?"
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735121
- DK2LONE (2982nd), 2026-08-14 03:57 UTC. RadImageNet ResNet-50 is CC-BY-NC-SA-4.0; widely used in public notebooks. quant_kiid 2026-10-03 bump. NO host answer.

## 738111 "External data rule - datasets that are free to access but carry research-use agreements"
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738111
- pizzaboy (97th), 2026-08-29 16:33 UTC, 0 comments, NO host answer. Stanford AIMI releases: free but RUA with NC + no-derivative clauses.

## 738156 "External data: are ungated, openly-licensed OAI-derivative datasets allowed?"
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738156
- "Yet, WiaLive !" (137th), 2026-08-30 05:53 UTC. Example OAIZIB-CM on Hugging Face (ungated, CC BY-NC 4.0, 507 DESS exams w/ tissue masks). Only comment deleted; NO host answer.

## 743420 "Rule 2.5.a.4 vs 2.8.a.1 - do weights trained on no-redistribution external data have to be published?"
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743420
- Richard_61 (40th), 2026-09-26 03:15 UTC; follow-up 2026-09-29. Claims: KneeMRI CC BY-NC-ND 4.0; MRNet RUA section 6 prohibits derivative works; fastMRI DSA prohibits distributing derived data files. Asks whether 2.5.a.4 exempts such weights from 2.8.a.1 public release. Mentions "every team using no-redistribution external data (KneeMRI, MRNet, fastMRI)" (implies such teams exist; INFERRED). NO host answer as of 2026-10-04.

## Discussion search "external" (2026-10-04) also lists: 733873 (LLM APIs, host answered), 744056 (prize eligibility w/ public weights trained on comp data), 745283 (RadImageNet weights for prize-eligible, 2026-10-02, no comments).

## 735304 "Best single-model score" (roy214, ~2026-08, 69 votes, 50+ comments) - what top-LB people say they do
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304
- tennogh (5th, team YOKKOISHO), ~1 month ago (~early Sep): "So far 0.942 raw, 0.943 with TTA at 288x288"; "OOF pseudo-labels have been pretty well correlated with LB"; "I have my own labels but they are not measurably better than the public ones." No mention of external data. Later: reports often lack the info; fixes overfit the 58 gold set.
- Scott Willis (10th): single model single fold 0.938, later 0.947 (5-fold) and 0.949 single fold; "I used Gemma 4 locally to extract labels from the reports"; smaller ResNet; ~5 min scoring; top of Efficiency LB. No external data mentioned.
- Tucker Arrants (43rd): 5-fold 0.943 @224; single-prompt LLM extraction; elsewhere (741819, 737950) says external knee MRI sets are effectively not allowed.
- CoolinLai (14th) ~2026-09-28: five-fold ResNet50 224px -> 0.954.
- Parag (13th, team Parag & Ian): FOV 140-160 mm tips.
- Yann Majewski (82nd) ~Sep: "combined my labels with some public dataset" (+0.015 LB) then "some pre training on the encoder" (data source unstated; ambiguous).
- Charles Savas (444th): "Our one external arm (correlation 0.83) was worth +0.004" (external = external model arm, not stated as external data).
- NguyenThanhNhan (66th): Qwen 3.5 2B fine-tune single fold 0.950. Raymond Yuen (45th): weak labels from public/API/local LLMs + teacher pseudo-labels, 0.949.
- Overall: every top-LB poster who explains their gains attributes them to labels/pseudo-labels/pipeline, none says they train on external knee MRI. (MEASURED absence in this thread; does not prove non-use.)
