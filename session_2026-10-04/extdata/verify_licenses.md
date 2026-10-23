# Adversarial verification of the "licenses" research (RSNA Knee Abnormality Detection, external knee MRI data)

Checked 2026-10-04 (evening IST). Read-only: no dataset downloaded, no form filled, no sign-in, no terms accepted, no post or vote. Kaggle pages read signed-out in my own browser tab (tab-4). Raw captures are in ./verify/ (fastmri_v.txt, aimi_*_v.txt, nda_duc.txt, oai_guide.txt, most_guide.txt, k917.json, kx_*.json, kx_readme.md, fmplus_*).
Timestamps: Kaggle title attributes were shown in IST and are converted to UTC here.
Tags: MEASURED = I read it on the cited page today. INFERRED = my reasoning. Status per claim: CONFIRMED / REFUTED / PARTLY REFUTED / UNCERTAIN.

## A. Kaggle side

### A1. Official rules, https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules (MEASURED)
- 2.6(a) and 2.6(b): CONFIRMED as quoted ("acceptable unless specifically prohibited by the Host", "reasonably accessible to all", "minimal cost").
- MISSED by the research, same clause 2.6(a), last sentence: using External Data "does not limit your other obligations ... including ... Section 2.8 (Winners Obligations)".
- 2.5(a) carve-out: CONFIRMED wording, but scope matters: it only says you "do not need to grant an open source license ... for that data and/or model(s)". It does not exempt the trained weights from 2.8.
- MISSED, 2.8(a): weights "should be provided as a public kaggle dataset"; winner must "represent that you have the unrestricted right to grant that license".
- MISSED, 3.14(a): you warrant the Submission does not infringe third-party rights, listed to include "confidentiality obligations".
- MISSED, 3.9(b): sponsor may disqualify, or require remediation within one week, including "the resolution of license conflicts".
- 2.6(b) example: buying a proprietary dataset licence that "exceeds the cost of a prize" is not reasonable (relevant only if someone argued a USD 70,000 AIMI commercial licence were needed; host says it is not, see A3).
- No external-data disclosure thread or disclosure deadline exists in the rules text (MEASURED absence). 2.5(b) does require a reproducible method description incl. training details, so external data use will be visible at winner review (INFERRED).
- Overview page (MEASURED, https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview): "Freely & publicly available external data is allowed, including pre-trained models"; host additionally asks winners to "Share final version of model as publicly available for open distribution and validation". Final deadline shows as 2026-10-22 23:59 UTC; entry/merger 2026-10-15; winners' requirement deadline 2026-11-05.

### A2. Host statement 741819 (OAI), https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819
- CONFIRMED. Author Po-Hao "Howard" Chen, badge COMPETITION HOST, posted 2026-09-21 15:34 UTC. Text: OAI "does not violate" the rules "if it is generally accessible to all participants without an escalated (i.e. institutional) review and sign-off or access to a legal team, and is free of charge"; host "cannot comment" on compliance with OAI's own terms.
- Topic by "John Doe" (now 197th), 2026-09-17 17:03 UTC. Competitor Tucker Arrants (45th) reads it as "a no" because OAI "requires institutional sign off" (opinion, not host).

### A3. MISSED host statement 733965 (pinned), https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965
- Host reply by Po-Hao "Howard" Chen (COMPETITION HOST), 2026-08-27 22:14 UTC, answering a question that named OAI, MRNet, fastMRI+ and SKM-TEA (MEASURED):
  - prize money "does not, by itself, make their use a commercial sale or commercial exploitation";
  - NC-only datasets "would not be excluded by the competition rules solely because" of the NC licence;
  - "straightforward registration or click-through data-use agreement would generally satisfy" accessibility;
  - "institution-specific approvals, negotiated written agreements, IRB approval, lengthy credentialing" may be a barrier;
  - "Teams remain responsible" that use, "including ... any required release of code or models", is permitted by the dataset licence.
- Not superseded: 734109 (08-25) and 741819 (09-21) apply the same access test. This is the single most important host text for the licence question and the research summary does not cite it.

### A4. Host statement 734109 (KneeCoT), https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109
- Content CONFIRMED ("due to the required data agreement with dataset institution"; "not allowed in this competition"; may change if HF policy changes).
- Date REFUTED: host reply is 2026-08-25 11:00 UTC, not "early Sept". Topic by Mazzutti 2026-08-09 23:34 UTC.

### A5. Thread 743416, https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416
- CONFIRMED: YYama (9th), 62 votes, 12 comments, no COMPETITION HOST reply as of 2026-10-04. Posted 2026-09-26 01:52 UTC (research said about 09-25; minor). Latest comment Eugene Khvedchenya 2026-10-03 16:53 UTC citing DFDC 157925 (top-2 removed for disallowed external data).
- The thread itself links KneeMRI to https://zenodo.org/records/14789903 (see B5).

### A6. Other external-data threads re-checked today, no host reply in any
- 743420 (Richard_61, 2026-09-26): asks whether 2.5.a.4 exempts weights trained on no-redistribution data (KneeMRI, MRNet, fastMRI) from 2.8 public release. No host answer. This is the open question that decides prize eligibility for ND-type data. MISSED by the research.
- 737950, 738111, 738156 (OAIZIB-CM, ungated OAI-derived), 733652 (contains Mazzutti's 2.6(a)/(b) argument, competitor opinion; host reply there only points to 733965).
- Pinned host posts 733343 and 733375 contain nothing on external data.

## B. Dataset side

### B1. fastMRI, https://fastmri.med.nyu.edu/
- Form: CONFIRMED. HTML form has 3 required inputs (name, email, institution) after "By entering my electronic signature here". "without charge" CONFIRMED.
- DSA clauses: CONFIRMED ("internal research or educational purposes only", "Not SELL OR OTHERWISE MONETIZE", no transfer of "data files derived", "link the fastMRI Dataset with any other dataset" inside the re-identification sentence, "destroy all copies", MLCommons carve-out).
- Nuances the research did not quote: the transfer ban excepts "individuals under my direct supervision" and "uses in academic publications and presentations with citations"; user must obtain "all approvals for the proposed use" under "institutional policies"; page says the application "must include the investigator's institutional affiliation and the proposed uses" but the form has no proposed-use field.
- Auto-approval: UNCERTAIN. Only source is a third-party InnerEye doc (https://innereye-deeplearning.readthedocs.io/md/fastmri.html, quote "You will shortly receive an email with links" CONFIRMED, page undated). A competitor (Fernando Faria, 737950) claims instead it "requires emailing NYU and waiting on an approved data-sharing agreement". Neither verifiable without applying.
- AWS registry "License: MIT License" with "ControlledAccess" pointing to the NYU page: CONFIRMED.

### B2. fastMRI+, https://github.com/microsoft/fastmri-plus
- CONFIRMED: MIT (LICENSE + API), "The entire knee dataset ... have been labeled", images obtainable "after signing the data sharing agreement" for "academic purposes", last push 2021-09-07.

### B3. MRNet and SKM-TEA
- Redivis access panels: CONFIRMED for both (https://stanford.redivis.com/datasets/4a2c-4cpkzrn2c?access and https://stanford.redivis.com/datasets/5r8z-achpw6q4f?access): "0 / 2 complete", "AIMI membership" Join, "Stanford_Research_Agreement per dataset" Apply. Extra detail: "Data exports Unrestricted". MRNet "Last updated: 2026-04-23".
- AIMI org page https://stanford.redivis.com/AIMI: CONFIRMED "Sign in using your Stanford ID or Google account"; "review and accept the applicable terms". Approval mode still UNCERTAIN. Warning: a web-search hit saying Redivis staff review "typically within one business day" is from Stanford GSB Library's Redivis org (https://gsb-research-help.stanford.edu/library/faq/358602), not AIMI; do not reuse it.
- Research Use Agreements on the legacy AIMI pages (MRNet bface6fc..., SKM-TEA 4aaeafb9...): CONFIRMED verbatim, both: "personal, non-commercial research purposes only"; "Any commercial use, sale, or other monetization is prohibited."; "must not modify, reverse engineer, decompile, or create derivative works"; "YOU MAY NOT DISTRIBUTE, PUBLISH, OR REPRODUCE A COPY". Extra: other users "must register as an individual user" (each teammate needs own registration, INFERRED).
- AIMI pages link "Download here" to Redivis (MEASURED, aimi.stanford.edu/datasets/mrnet-knee-mris and /skm-tea-knee-mri).
- Commercial licence https://aimi.stanford.edu/datasets-commercial-use: CONFIRMED "$70,000 USD per dataset", "up to 4 weeks". Caveat: fee is stated "for agreements contracted in FY25 (through 8/31/25)" and the process is written for companies.

### B4. OAI / NDA
- OAI CORE FAQ https://www.oaiknowledgebase.com/about/FAQs/: CONFIRMED ("You MUST complete the Data Use Agreement before accessing OAI data"; Login.gov, My Account, Data Permissions, "Osteoarthritis Initiative", Request Access, Start Request). Guide PDF v1.0 June 3, 2026 CONFIRMED.
- NDA access page https://nda.nih.gov/nda/access-data-info: Open Access "only need an NDA account", "approved upon submission" CONFIRMED. PARTLY REFUTED as a clean picture: the same page's generic section says "An NDA account with an institutional email is required to request data access", "Each request requires an electronic signature from the Lead Recipient as well as an authorized Signing Official", and recipients "must not be located in a country of concern" (NOT-OD-25-083). NDA glossary (https://nda.nih.gov/nda/glossary) "Affiliated Institution": DARs "require that the requestor be affiliated with an institution recognized by NIH in the eRA Commons system", while defining Open Access groups as accessible to "users who are not affiliated". NDA's own pages are internally inconsistent.
- SOPs https://nda.nih.gov/nda/standard-operating-procedures: SOP-04B Rev 3, Effective 1/14/2025, "clicks to accept the Terms and Conditions", no Institutional Sponsorship: CONFIRMED. SOP-01 "1-3 business days" CONFIRMED, but it continues "Account requests that involve access to shared data ... will take longer" (omitted by the research).
- OAI group type: UNCERTAIN (agree with research). NDA collection page https://nda.nih.gov/edit_collection.html?id=2343 ("Osteoarthritis Initiative #2343", DOI 10.15154/64dx-3822) shows "Permission Group:" empty when signed out.
- https://nda.nih.gov/oai banner "under review for potential modification in compliance with Administration directives": CONFIRMED. https://nda.nih.gov/oai/query-download: "After registering and obtaining a password, you may browse, download" CONFIRMED; it adds use "for human subjects research for scientific and educational purposes" (not quoted by research).
- DUC PDF https://s3.amazonaws.com/nda.nih.gov/cms/prod/NDA-Data-Access-Request-DUC-FINAL.pdf: "Last updated: July 2026", purpose clause, no commercial-use ban: CONFIRMED. Caveat: "any resulting commercial product(s)" sits in a no-endorsement clause, it is not a grant of commercial rights. This is the institutional DUC (SO attestations); the Open Access click-through text is not public.
- re3data https://www.re3data.org/repository/r3d100011843: "restricted", "registration", last update 2024-06-19 CONFIRMED.

### B5. KneeMRI
- REFUTED: "official page now returns 403/404, so only mirrors remain". The riteh page does return 403 (http) and 301 to 404 (https) (MEASURED), but the curator moved the dataset to Zenodo: https://zenodo.org/records/14789903 "KneeMRI dataset", Data curator Štajduhar, "Dataset Open", record created February 2, 2025, files incl. metadata.csv 42.4 kB and volumetric_data.7z 3.7 GB, licence CC BY-NC-ND 4.0, HISTORY says it "was recently moved to Zenodo". No registration shown. So KneeMRI has an official, ungated, free source today.
- Licence CC BY-NC-ND 4.0 CONFIRMED on Zenodo and the 2023 Wayback copy.

### B6. Kaggle "Knee MRI 917"
- CONFIRMED via https://www.kaggle.com/api/v1/datasets/view/ibrahimshaikhdev/knee-mri-917 and the dataset page: licence "CC0: Public Domain", empty description, v1 2026-10-03T10:59Z, totalBytes 6,673,316,292, aclDiagnosis histogram 690 / 172 / 55, kneeLR 448 / 469, metadata.csv 42.41 kB (same size as Zenodo's metadata.csv). Re-upload of KneeMRI: high confidence (INFERRED). CC0 label cannot relicense CC BY-NC-ND material (INFERRED). With the official Zenodo copy available there is no reason to use this mirror.

### B7. KneeXNet-2.5D, https://github.com/pitthexai/Knee_MRI_Segmentation_2.5D
- CONFIRMED: licence null, README "publicly and freely available for any research and educational purposes", masks only (KneeMRI_Annotations.zip, 12,001,754 bytes), users "must individually download the corresponding MRI ... from the official OAI repository".
- PARTLY REFUTED detail: not "a 2026 repository": created 2024-11-14, last push 2026-06-27 (paper is 2026). The .pth files in /Models are 133 bytes each, which looks like Git LFS pointers, so weight availability is unverified (INFERRED).

### B8. MOST
- CONFIRMED (https://agingresearchbiobank.nia.nih.gov/studies/most/details and User's Guide PDF): knee MRIs to 168-month visit; "not available for commercial use"; requester must be at least "a tenure-track professor or senior researcher", not "post-docs or graduate students"; IRB; "Under Biobank Scientific Review", "several weeks"; agreement signed by "an institutional signing official (SO)". Fails the host test.

### B9. KneeCoT, https://huggingface.co/datasets/YiHui0124/KneeCoT
- CONFIRMED: CC-BY-NC-4.0, gated, formal ethical use agreement with the affiliated hospital. Banned by host 2026-08-25.

## C. Overall conclusions of the research, re-scored
- "Latest host statement is 741819 (09-21)": CONFIRMED as latest, but INCOMPLETE: the governing general statement is 733965 (08-27), not cited.
- "Every set except OAI has an NC, ND or monetization restriction that clashes with a prize-eligible finish": PARTLY REFUTED. The NC/monetization part is answered by the host in 733965 (prize money is not commercial; NC alone does not exclude). What still clashes is the ND / "no derivative works" / "no transfer of derived data files" language versus the 2.8 obligation to publish weights publicly and warrant an "unrestricted right". That is exactly the unanswered 743420 question.
- Access test: KneeMRI passes more clearly than stated (official ungated Zenodo copy). fastMRI and MRNet/SKM-TEA look like the "click-through" category the host called generally acceptable, approval mode unverified. OAI unresolved, with NDA pages contradicting themselves and a possible country-of-concern geo limit. MOST and KneeCoT fail.
- No host per-dataset yes/no exists as of 2026-10-04: CONFIRMED.

## D. Missed sources worth knowing
- OAIZIB-CM, https://huggingface.co/datasets/YongchengYAO/OAIZIB-CM (WebFetch summary): ungated, CC BY-NC 4.0, 507 OAI knee MRIs plus masks. Its use was asked in 738156, no host answer; it redistributes OAI images, so OAI DUC compliance of the re-host is questionable (INFERRED).
- 3DReasonKnee, https://huggingface.co/datasets/rajpurkarlab/3DReasonKnee (WebFetch summary): CC BY 4.0 annotations, images must come from OAI via NDA.
