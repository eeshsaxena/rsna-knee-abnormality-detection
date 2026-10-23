# RSNA Knee Abnormality Detection: external knee MRI data, who uses what, and public evidence

Compiled 2026-10-04 (UTC) for team "Eesh saxena". Read-only: Kaggle CLI 2.2.4 (list/metadata/file listings only, nothing downloaded except small dataset-metadata.json files) plus a signed-out browser tab. No forms, no sign-in, no cookie acceptance, no posts or votes.

Tags: MEASURED = I read it on the page or in CLI output. INFERRED = my interpretation.
Times are UTC, converted from the page's hover timestamps (shown in IST).

Supporting notes in the same folder: notes_rules_page.md, notes_threads.md, notes_datasets.md, cli_*.txt, meta/.

---

## 1. Bottom line

1. Rules allow external data in principle. Rule 2.6 and the Overview both say freely and publicly available external data is allowed. (MEASURED)
2. The host has ruled on only two datasets. KneeCoT is banned (2026-08-25). OAI is allowed only if no institutional sign-off is needed (2026-09-21). The general test (2026-08-27) is: straightforward registration or click-through = generally OK; institutional approval, negotiated agreement, IRB or lengthy credentialing = may not be OK. Non-commercial licenses alone do not exclude a dataset. Teams stay responsible for the dataset's own terms, including the required public release of weights. (MEASURED)
3. Nobody has answered the per-dataset question in 743416 (MRNet, SKM-TEA, fastMRI/fastMRI+, OAI, KneeMRI, KneeXNet-2.5D; posted 2026-09-26, 62 votes). The same goes for 735497, 737950, 738111, 738156, 743420, 735121 and 745283. No pinned "approved list" exists. (MEASURED)
4. Top teams: there is no public evidence that any top-12 team trains on external knee MRI. None of the visible top-12 accounts has a public notebook or dataset for this competition. rhoskeri (1st), Pa3BaJluHa (2nd), Wassim Dobbi (4th), TomKroumov (6th) and NMarkS (8th) have zero discussion posts. The top-LB people who do post (tennogh/YOKKOISHO 5th, Scott Willis 10th, CoolinLai 14th, Parag 13th) put their gains down to labels, pseudo-labels, small CNNs and FOV, never to external data. (MEASURED absence. It does not prove non-use. INFERRED: the "top teams use external data" belief comes from forum worry, not from any disclosure.)
5. External knee MRI is already on Kaggle as re-uploads that carry a CC0 license field the source licenses do not support:
   - MRNet: techtech69, 2,321 downloads.
   - KneeMRI/Rijeka twice: techtech69/rijeka-knee-mri-dataset, and ibrahimshaikhdev/knee-mri-917 (2026-10-03), whose file names and byte sizes are identical to the Rijeka upload.
   - A harmonized RSNA+MRNet+OAI label table: tsuyu122.
   - An MST checkpoint trained on MRNet: quannguyen0501.

   (MEASURED for the listings. The "license mismatch" is INFERRED from the source terms competitors quote.)
6. Public notebooks: many popular notebooks list "kneeMRIdataset" (sohaibanwaar1203, the 2019 KneeMRI upload) as an input. The ones I opened do not reference it in code. Notebooks with "MRNet" in the title or text turned out to mean the MRNet-style architecture, not MRNet data. (MEASURED for the notebooks checked.)
7. Precedent from the most recent RSNA competition with identical 2.6 text (RSNA 2025 Intracranial Aneurysm):
   - The host rejected a request-form dataset that needs institution/PI details, and a challenge-only dataset.
   - The host approved CC0 OpenNeuro datasets.
   - Prize winners used only open-download or host-approved data (2nd place: TopCoW Zenodo, explicitly excluding the rejected LargeIA; 6th place: OpenNeuro).
   - In RSNA 2023 Breast, the host ruled 3 days before the deadline that models built on a non-public dataset (NYU) were ineligible to win.

   (MEASURED)

---

## 2. Official rules text (source type: official_rules)

URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules (MEASURED, read 2026-10-04)
- 2.6(a): External Data must be "publicly available and equally accessible ... at no cost", or meet the Reasonableness criteria in 2.6(b).
- 2.6(b): "The use of external data and models is acceptable unless specifically prohibited by the Host". It must also be "reasonably accessible to all" and of "minimal cost". The Host applies a Reasonableness Standard (cost thresholds, accessibility).
- 2.5(a): if input data or pretrained models with an incompatible license are used, the winner need not open-source license that data/model.
- 2.8: winners deliver training code, inference code and weights; "The model weights should be provided as a public kaggle dataset".
- Data license for the competition data: MIRA. Winner license: CC-BY-NC 4.0.
- No list of approved or prohibited external datasets or pretrained models appears anywhere in the rules.

URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview (MEASURED)
- Code Requirements: "Freely & publicly available external data is allowed, including pre-trained models".
- Extra host asks of winners: a video, a forum link to open code and weights, and "Share final version of model as publicly available for open distribution and validation".
- Timeline: entry/merger deadline 2026-10-15, final submission 2026-10-22, winners' requirements 2026-11-05.

---

## 3. Host statements, in date order (source type: host_or_staff)

All of these are by Po-Hao "Howard" Chen, shown with the COMPETITION HOST badge.

| Date (UTC) | Thread | What the host said | Status |
|---|---|---|---|
| 2026-08-09 09:52 | 733965 "Use of Commercially Hosted LLMs" (host post) | Hosted LLM APIs for label extraction are permitted under 2.6(b). The host reserves the right to judge accessibility, cost or unfair advantage. | Still current |
| 2026-08-25 11:00 | 734109 (KneeCoT, asked by Mazzutti) | "the KneeCoT dataset is not allowed in this competition", because of the "required data agreement with dataset institution". May change if the HF policy changes. | Still current |
| ~2026-08-27 22:14 | 733965, reply to UGUR OZER's OAI/MRNet/fastMRI+/SKM-TEA question | NC licenses alone do not exclude a dataset. Prize money is not commercial use by itself. "straightforward registration or click-through data-use agreement would generally satisfy" accessibility. Institution-specific approvals, negotiated agreements, IRB or lengthy credentialing "may present a meaningful accessibility barrier". Teams stay responsible that their use, including required release of code/models, fits the dataset license. | Still current. The most general statement, and not superseded. |
| 2026-09-21 15:34 | 741819 (OAI) | OAI use "does not violate" the rules "if it is generally accessible to all participants without an escalated (i.e. institutional) review and sign-off", and free. The host cannot comment on OAI's own terms. | Consistent with 08-27. Competitors read it as a de facto "no" for OAI via NDA. |

As of 2026-10-04, none of these threads has a host or staff reply: 743416 (YYama, 2026-09-26), 735497 (2026-08-16), 737950 (2026-08-28), 738111 (2026-08-29), 738156 (OAIZIB-CM, 2026-08-30), 743420 (Richard_61, 2026-09-26, on weights release vs 2.5.a.4), 735121 (RadImageNet CC-BY-NC-SA, 2026-08-14) and 745283 (RadImageNet, 2026-10-02). (MEASURED)

---

## 4. How top teams use external data: evidence

### 4.1 Leaderboard identities (MEASURED, LB page 2026-10-04)

Ranks 1 to 12, score and visible member:
1. rhoskeri, 0.963 (/rhoskeri)
2. Pa3BaJluHa, 0.962 (/hettpugymajlhuk)
3. joinT fusion, 0.961 (/tomokihirose, +2)
4. Wassim Dobbi, 0.961
5. YOKKOISHO, 0.960 (/tennogh, +2)
6. TomKroumov, 0.959 (/tom1115)
7. dalab, 0.959 (/junpeimorioka, +4)
8. NMarkS, 0.959
9. knee-deep, 0.959 (/yosukeyama = YYama, +4)
10. Scott Willis, 0.959
11. Zhengru Li, 0.959
12. takoi & charmq, 0.959

Mazzutti is 21st at 0.958.

### 4.2 Public artefacts from top accounts (MEASURED: kaggle datasets/kernels list --user)

- No visible top-12 member has a public notebook or dataset for this competition.
- tennogh's only knee dataset is "My knee MRI": his personal scan, CC0, 2026-08-19, 15 votes. It is a single study, not a training corpus.
- The other accounts' public items are from older competitions.

### 4.3 What top-LB people said in public (MEASURED)

**Discussion activity**
- No discussion posts at all: rhoskeri, Pa3BaJluHa, Wassim Dobbi, TomKroumov, NMarkS (profile Discussion tab shows "No results found").

**Thread 735304 "Best single-model score"** (https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304):
- tennogh (5th), about a month ago: 0.942 raw and 0.943 TTA at 288x288; "OOF pseudo-labels have been pretty well correlated with LB". His own labels are "not measurably better than the public ones". He never mentions external data.
- Scott Willis (10th): runs Gemma 4 locally to extract labels, uses a smaller ResNet, got 0.949 single fold with about 5 min scoring, and has trained about 900 models. He never mentions external data.
- CoolinLai (14th), ~2026-09-28: five-fold ResNet50 at 224 = 0.954.
- Parag (13th): FOV tips only.

**Clarification threads**
- YYama (knee-deep, 9th) wrote the 743416 clarification request. He says external data "could allow even a fairly simple approach to achieve a dramatic improvement". That is a worry, not a disclosure.
- Mazzutti (21st) asked on 2026-08-09 to use KneeCoT "only for auxiliary training" and accepted the ban. He also argued (733652, 2026-08-20) that MRNet/fastMRI/OAI/SKM-TEA pass 2.6(b).
- Richard_61 (40th) on 2026-09-29 asks for an answer so that "every team using no-redistribution external data (KneeMRI, MRNet, fastMRI)" can pick final submissions. INFERRED: some teams near the top-50 are using such data, possibly his own.

**Lower-ranked hints**
- Yann Majewski (82nd): "combined my labels with some public dataset" (+0.015) and "some pre training on the encoder". The source is not stated.
- Charles Savas (444th): "Our one external arm" was worth +0.004. That is an external model arm.

**Sceptics**
- Tucker Arrants (42nd/43rd/44th across threads) says useful external knee sets fail either the accessibility test or the no-derivatives terms. Cody_Null and Optimo (26th to 30th) say they have not tried external data.

**INFERRED overall:** the public record supports "the forum suspects top teams use external data". It does not support "top teams have disclosed using it". Top-team methods will only be known from post-deadline write-ups. Winners must publish code and weights, which is when the host would audit.

---

## 5. Public Kaggle datasets that bring external knee MRI into this competition (MEASURED via CLI)

| Kaggle dataset | What it is | Created/updated | Downloads | License field | Notes |
|---|---|---|---|---|---|
| techtech69/mrnet-knee-mri-stanford-aimi | MRNet v1.0 re-upload (train-abnormal/acl/meniscus.csv + train/*.npy), 7.57 GB | 2026-08-27 | 2,321 | CC0-1.0 | Source is Stanford AIMI Research Use Agreement. Competitors (743416, 743420) say it bars derivative works (INFERRED: CC0 relabel is not valid). |
| techtech69/rijeka-knee-mri-dataset | KneeMRI (Rijeka) *.pck volumes, 4.15 GB | 2026-08-27 | 7 | CC0-1.0 | Same uploader, same day as MRNet. |
| ibrahimshaikhdev/knee-mri-917 | "Knee MRI 917", 4.15 GB, metadata.csv + volumetric_data/*.pck, no description | 2026-10-03 10:59 | 3 | CC0-1.0 | File names and byte sizes match techtech69/rijeka (e.g. 1001866-5.pck, 7,395,745 B), so this is the KneeMRI/Rijeka set (917 volumes). Source is CC BY-NC-ND 4.0 per 743416 (zenodo 14789903). |
| sohaibanwaar1203/kneemridataset | KneeMRI 917 volumes (2019) | 2019-01-15 | 5,329 | unknown | Attached to many RSNA-knee notebooks (see section 6). |
| cjinny/mrnet-v1, henryhaefliger/kneescans, abdurrahimsheikh/mrnet-dataset, flap1812/mrnet-v1-0, sharif123diu/mrnet-dataset | Older MRNet re-uploads | 2019 to 2025 | up to 2,411 | various | Older uploads, but still downloadable. |
| bharath111l/fastmri-plus-official-knee-annotations | fastMRI+ knee annotation CSVs only (no images) | 2026-08-24 | 76 | unknown (LICENSE file inside) | Images still need the NYU fastMRI DSA. |
| tsuyu122/rsna-knee-mri-balanced-labels-public | labels.parquet: 8,631 studies = RSNA 4,407 + MRNet 1,248 + OAI 3D-DESS 2,976 (+ fastMRI/OAI-ZIB auxiliary) | 2026-09-06 | 208 | CC0-1.0 | Labels only. It cites a "Kaggle Competition Rule 4.b (AI Provenance & Open Weights)" that does not exist in these rules, so treat it as unreliable. It also republishes RSNA labels. |
| quannguyen0501/dinov2-s-mrnet | MST_MRNet.ckpt, 275.6 MB | 2026-09-29 | 0 | unknown | INFERRED: Medical Slice Transformer (DINOv2-S) checkpoint trained on MRNet. Same author runs RSNA-knee preprocessing notebooks (2026-09-27 to 10-03). |
| tsuyu122/knee-b01-orthofoundation-l-public, prithvirajrp/orthofoundation-rsna-knee-raw-embeddings | OrthoFoundation-L checkpoint/embeddings | 2026-09-06 / 2026-10-03 | 18 / 25 | CC0 / apache-2.0 | MSK foundation model. Pretraining corpus not verified (may include external knee MRI). |
| kirilllabzin/knee-radiodino-s16-v21-source | RadioDINO weights + WEIGHT_LICENSE | 2026-10-03 | 16 | CC BY 4.0 | Radiology foundation model. Corpus not verified. |

- No SKM-TEA dataset on Kaggle (`datasets list -s skm-tea` returns none).
- OAI uploads on Kaggle are knee X-ray or segmentations (kgaooo/oai-tissue-segmentations, 2023), not MRI exams.

---

## 6. Public competition notebooks (MEASURED)

**kneeMRIdataset as input**
- `kaggle kernels list --dataset sohaibanwaar1203/kneemridataset` lists about 25 RSNA-knee notebooks, including:
  - sakhawathossen/rsna-knee-enhanced-ensemble (94 votes, 2026-09-14)
  - romanrozen/rsna-knee-data-structure-eda-baseline (141 votes)
  - amanatar super-ensembles
  - shashwat1729 inference forks
  - sonameyden/rsna-knee-solution (2026-10-02)
- Spot checks:
  - sakhawathossen: the /input page lists "kneeMRIdataset". The rendered notebook has no "kneemri", ".pck", "metadata.csv" or "MRNet" strings, so it is attached but not used. "External" in that notebook means other public notebook ensembles.
  - romanrozen: the current version's input lists only competition data and DINOv2 models. The CLI hit likely reflects an older version, and forks inherit inputs. (INFERRED)

**"MRNet" mentions that turned out to mean the architecture or the literature, not the data**
- juejiezeng/rsna-knee-mrnet50-inference: an "MRNetStyleClassifier" trained on competition PNGs.
- jirkaborovec/kneeabdet-12-label...: cites "the MRNet knee-MRI literature".

**Intent, not use**
- gengsr/rsna-knee-dicom-protocol-audit (2026-08-20) audits RSNA DICOM protocols to "Decide which external data is most similar to RSNA" (fastMRI/MRNet).

**External pretrained models seen in competition notebooks**
- fabotelli/pillar-0-knee-fine-tune-loader-demo: Pillar-0 BreastMRI to knee fine-tune, gold-58 macro-AUC 0.797, backbone ECL-2.0.
- RadImageNet ResNet-50 (CC BY-NC-SA) in many "DINOv2 + RadImageNet" public stacks. INFERRED: RadImageNet's corpus includes MSK MRI. The license question is unanswered (735121, 745283).

---

## 7. Precedent from earlier RSNA Kaggle competitions

**RSNA 2025 Intracranial Aneurysm** (same 2.6 text). Rules: https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/rules (MEASURED)
- Host JeffRudie (COMPETITION HOST) in https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/608179:
  - ~2025-09-22: "we have decided that these do not qualify". The LargeIA Zenodo request form needs institution/PI. ADAM's agreement limits use to that challenge.
  - ~2025-09-24: OpenNeuro CC0 aneurysm sets are fully open and "can be used for this challenge".
- Prize solutions:
  - 2nd place used TopCoW Zenodo data and states "The LargeIA dataset was excluded from training" (https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution).
  - 6th place lists "External data from OpenNeuro.org" (https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/6th-place-solution).

**RSNA 2024 Lumbar Spine** (rules 7.C): https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/rules (MEASURED)
- External Data must be "publicly available and equally accessible ... at no cost".
- Winners must publish code "with external datasets made public".
- 10th place (2024-10-09) lists "pretraining on external datasets" under things that did not work (https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539587).

**RSNA 2023 Abdominal Trauma**: same 7.C and "external datasets made public" clauses. Overview: "Freely & publicly available external data is allowed, including pre-trained models" (https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/overview). A commenter on the 1st-place write-up says it trained segmentation on the TotalSegmentator dataset (competitor statement, open CC BY data).

**RSNA 2023 Screening Mammography**
- Host Chris Carr (COMPETITION HOST), ~2023-02-24, 3 days before the end: public datasets are allowed, but entries using models based on the non-public NYU dataset "would not be eligible as contest winners" (https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/379133).
- 3rd place used CBIS-DDSM, CMMD, VinDr and Mini-DDSM (https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/391779). INFERRED, not verified here: VinDr-Mammo is on PhysioNet behind credentialing.

**Cross-competition warning** (competitor-cited): DFDC 157925, where the top-2 were removed for disallowed external data after the host never answered clarification requests.

**INFERRED pattern:**
- RSNA hosts let open-download data through and block request-form or institution-gated data, sometimes late.
- The real risk shows up at the prize audit, not on the leaderboard.

---

## 8. Practical read for Eesh (INFERRED, not legal advice)

- **KneeCoT:** NO (explicit host ban, 2026-08-25).
- **OAI via NDA** (Data Use Certification signed by the institution): likely NO under the 2026-09-21 condition. Ungated OAI derivatives (OAIZIB-CM, KneeXNet-2.5D, OAI-trained public weights) are unanswered.
- **MRNet and SKM-TEA** (Stanford AIMI click-through, per competitors): pass the host's accessibility test ("click-through ... would generally satisfy"). Competitors say the RUA bars derivative works, which conflicts with 2.8's public weight release. Host: teams carry that responsibility. Moderate leaderboard risk, high prize-eligibility risk.
- **KneeMRI/Rijeka** (Zenodo open download, CC BY-NC-ND 4.0): accessible, and NC is OK per the host. ND conflicts with releasing derived weights. Same profile as MRNet.
- **fastMRI images** (emailed DSA approval, per competitors): closer to a "negotiated agreement", so uncertain. fastMRI+ CSV annotations alone are useless without the images.
- **Using the Kaggle CC0 re-uploads** (techtech69, ibrahimshaikhdev) does not change the source terms. A CC0 label on a re-upload is not a license grant.
- **Evidence gap:** no top-12 team has disclosed external knee MRI use, so there is no public proof that it is what separates 0.959 from 0.963.

---

## 9. Open questions

1. Will the host answer 743416 (per-dataset yes/no) or 743420 (whether 2.5.a.4 exempts weights trained on no-derivatives data from 2.8 public release) before 2026-10-22?
2. Does the MRNet RUA actually prohibit derivative works or model release? This rests on competitor quotes (743420, 743416). Not verified on the Stanford AIMI page here.
3. Does fastMRI image access require an individually approved DSA? (Competitor claim in 737950.)
4. Which corpora were OrthoFoundation, RadioDINO and the MST_MRNet checkpoint trained on, and under which licenses?
5. Do the hidden top teams (rhoskeri, Pa3BaJluHa, Wassim Dobbi) use any external MRI? This can only be known after the write-ups.
6. Does the RadImageNet CC BY-NC-SA backbone, widely used in public stacks, satisfy winners' obligations? (735121, 745283 unanswered.)
