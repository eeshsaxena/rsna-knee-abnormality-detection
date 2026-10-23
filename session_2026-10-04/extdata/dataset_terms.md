# External knee MRI datasets: current access routes and terms

Checked 2026-10-04 for the Kaggle competition "RSNA Knee Abnormality Detection" (team "Eesh saxena").
Read-only research: no form filled, no sign-in, no download of any dataset, no terms accepted.
Tags: MEASURED = read on the cited page today. INFERRED = my reasoning or an indirect source. UNVERIFIED = could not see it without signing in.

## 0. The yardstick (competition side, for context)

Three different layers decide "allowed":

1. Official rules text, Section 2.6 (MEASURED, https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules):
   - 2.6(a): External Data must be "publicly available and equally accessible to use by all Participants", "at no cost to the other Participants", or meet the 2.6(b) Reasonableness criteria.
   - 2.6(b): "acceptable unless specifically prohibited by the Host"; must be "reasonably accessible to all" and of "minimal cost".
   - 2.5 Winner License: winning Submission and code licensed under CC-BY-NC 4.0. Carve-out: input data or pretrained models "with an incompatible license" need not be open-sourced.
2. Host statements (both by Po-Hao "Howard" Chen, badge "COMPETITION HOST"):
   - Thread 734109 (host reply about a month ago, roughly early Sept 2026), KneeCoT banned "due to the required data agreement with dataset institution" (MEASURED, https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109).
   - Thread 741819 (host reply 13 days ago, about 2026-09-21), OAI: allowed "if it is generally accessible to all participants without an escalated (i.e. institutional) review and sign-off or access to a legal team, and is free of charge." The host also says they cannot comment on compliance with the data stewards' own terms (MEASURED, https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819). This is the latest host statement I found. It does not override the KneeCoT ban; it states the same test in general form.
3. The dataset owner's own terms. This is what this file is about. The host explicitly does not rule on it, so the team carries that risk.

The working test I apply below for column (d): can one individual Kaggle participant get the data free, by themselves, with no institutional signatory, no review committee and no legal team?

## 1. Summary table

| Dataset | (a) Access route today | (b) Cost | (c) Terms (key restrictions) | (d) Individual, free, no institutional sign-off? |
|---|---|---|---|---|
| fastMRI knee (raw + DICOM) | Web form with e-signature: name, email, institution; links emailed | Free | NYU fastMRI Dataset Sharing Agreement: internal research/education only, no monetizing, no transfer of data "or derived" files, no linking with other datasets, destroy when done | Yes, probably (form, no SO). Owner-terms risk is high for a prize competition |
| fastMRI+ annotations | Open GitHub repo | Free | MIT (labels only; images still under fastMRI DSA) | Labels yes; images same as fastMRI |
| MRNet | Redivis: sign in (Stanford ID or Google), join AIMI org, apply "Stanford_Research_Agreement" | Free non-commercial; commercial licence USD 70,000/yr | Stanford RUA: personal non-commercial research only, no monetization, NO derivative works, no redistribution | Gate is individual (Google account). Whether approval is automatic is UNVERIFIED. Owner terms conflict with prize + weight release |
| SKM-TEA | Same Redivis gate as MRNet | Same | Same Stanford RUA template | Same as MRNet |
| OAI (NDA) | Login.gov/RAS sign-in, NDA account (staff-reviewed), request OAI permission group, accept DUC | Free | NDA Data Use Certification: research/teaching; no ban on commercial products; no redistribution; acknowledgement; 1-year access | UNCERTAIN. Points to an Open Access click-through group, but not confirmable without login; repository "under review" banner |
| KneeMRI (Stajduhar, Rijeka) | Was direct download, no registration. Official page now 403/404 | Free | CC BY-NC-ND 4.0 | Yes historically (no gate). Official host now offline; only mirrors |
| KneeXNet-2.5D | Open GitHub repo (masks, code, weights) | Free | No LICENSE file; README "any research and educational purposes" | Masks/weights yes; images must come from OAI |
| MOST | NIA AgingResearchBiobank request | Free data; biospecimen fees | PI-level requester, IRB, NIA review, transfer agreement signed by institutional SO, no commercial use | No |
| Kaggle "Knee MRI 917" | Kaggle dataset, open | Free | Declares CC0, but it is a re-upload of KneeMRI (CC BY-NC-ND 4.0) | Yes to download, but the CC0 label is not valid; original licence still applies (INFERRED) |
| KneeCoT (already banned) | HF gated + hospital ethical-use agreement | Free | CC-BY-NC-4.0 + formal agreement | No (host ruled it out) |

## 2. Per-dataset detail

### 2.1 fastMRI (NYU Langone; knee raw k-space and DICOM)
Source page: https://fastmri.med.nyu.edu/ (MEASURED, full text pulled 2026-10-04).
- Content: >1,500 fully sampled knee raw scans (coronal PD with/without FS) and "DICOM images from 10,000 clinical knee MRIs" (cor PD, cor PD FS, ax PD FS, sag PD, sag T2 FS). No pathology labels ship with it (INFERRED from page; labels come from fastMRI+ for the raw set only).
- (a) Access: "acceptance of the Data Sharing Agreement" plus "an online application form". The HTML form has three required fields: Full Name, Email, Institution, then "By entering my electronic signature here" (MEASURED). The page also says the application must include "institutional affiliation and the proposed uses" (MEASURED), but there is no proposed-use field in the form (MEASURED). After registering "You will shortly receive an email with links" (MEASURED on https://innereye-deeplearning.readthedocs.io/md/fastmri.html), which suggests automatic approval (INFERRED). Data sit in a controlled-access S3 bucket arn:aws:s3:::fastmri-dataset (MEASURED, https://raw.githubusercontent.com/awslabs/open-data-registry/main/datasets/nyu-fastmri.yaml). Note: that AWS registry entry says "License: MIT License"; the authoritative data terms are the NYU agreement on fastmri.med.nyu.edu, the MIT label most likely refers to code (INFERRED).
- (b) Cost: download link "without charge" (MEASURED).
- (c) Agreement: "NYU Langone Health fastMRI Dataset Sharing Agreement" (MEASURED). Key clauses:
  - Licence for "internal research or educational purposes only" (MEASURED).
  - "Not SELL OR OTHERWISE MONETIZE any portion or all" (MEASURED).
  - Not distribute/transfer the dataset "or any subsequent variables or data files derived" from it (MEASURED). Whether trained weights are such "derived data files" is a legal question (INFERRED risk).
  - No attempt to "link the fastMRI Dataset with any other dataset" (MEASURED; it sits in the re-identification sentence, so its intent is privacy, but the words are broad).
  - Each user "must register as an individual user"; do not share the link (MEASURED).
  - "destroy all copies" when work is completed (MEASURED).
  - Citation: "NYU fastMRI" in abstracts, Knoll et al. 2020 + arXiv 1811.08839 in manuscripts (MEASURED).
  - The only named benchmark carve-out is MLCommons/MLPerf, declared "non-commercial use" (MEASURED). Nothing mentions Kaggle or prize competitions.
- (d) Individual, free, no SO: Yes, very likely. Institution is a free-text field, not a signatory step (INFERRED that an independent participant can complete it). Meets the host's 741819 access test on its face. Owner-terms risk: prize money vs "monetize", and CC-BY-NC release of weights vs the "derived ... data files" clause (INFERRED).

### 2.2 fastMRI+ (pathology annotations for fastMRI)
Source: https://github.com/microsoft/fastmri-plus (MEASURED), paper https://www.nature.com/articles/s41597-022-01255-z (via search snippet).
- Content: bounding-box annotations; README says "The entire knee dataset" (raw k-space knee set) was labeled (MEASURED). Paper abstract: 16,154 knee bounding boxes, 22 pathology categories (search snippet, not opened; treat as MEASURED-secondhand).
- (a) Access: open GitHub, no gate (MEASURED). (b) Free. (c) Repo licence MIT (MEASURED, LICENSE file + GitHub API). README: images obtained from fastMRI "after signing the data sharing agreement" for "academic purposes" (MEASURED). Last push 2021-09-07 (MEASURED).
- README limitation: labels "should not be considered clinical ground truth" (MEASURED).
- (d) Labels yes. Useless without the fastMRI images, which carry the fastMRI agreement (INFERRED). Labels are on coronal PD raw-reconstruction images only, not on the 10k DICOM set (INFERRED from README wording).

### 2.3 MRNet (Stanford ML Group / Stanford AIMI)
Sources: https://stanfordmlgroup.github.io/competitions/mrnet/ (MEASURED: "Find the dataset on the Stanford AIMI website"), https://aimi.stanford.edu/datasets/mrnet-knee-mris (MEASURED: "Download here" points to Redivis; cite DOI 10.71718/rcbp-8c35), https://stanford.redivis.com/datasets/4a2c-4cpkzrn2c (MEASURED in browser), legacy https://stanfordaimi.azurewebsites.net/datasets/bface6fc-7859-47d7-a1c8-022cd6b17419 (MEASURED, still up, file MRNet-v1.0.zip 6.09 GB, "Please login in order to export").
- Content: 1,370 exams, labels abnormal/ACL/meniscus from reports (MEASURED).
- (a) Access today (Redivis, last updated 2026-04-23): "Data access ... 0 / 2 complete": "AIMI membership" (Join) and "Stanford_Research_Agreement per dataset" (Apply) (MEASURED, ?access panel). AIMI org page: "Sign in using your Stanford ID or Google account, and join the Stanford AIMI organization" then "review and accept the applicable terms" (MEASURED, https://stanford.redivis.com/AIMI). Whether membership/agreement are auto-approved or admin-reviewed is UNVERIFIED (Redivis supports both: "automatically approved or will be marked as pending", per https://docs.redivis.com search snippet).
- (b) Cost: free for non-commercial research; "no charge for non-commercial research use" (MEASURED, https://aimi.stanford.edu/shared-datasets). Commercial licence "$70,000 USD per dataset" per year (FY25), committee review "up to 4 weeks" (MEASURED, https://aimi.stanford.edu/datasets-commercial-use).
- (c) Licence: "Stanford University Dataset Research Use Agreement" (MEASURED on Azure page; same text in the 2019 MRNet page commit https://github.com/stanfordmlgroup/stanfordmlgroup.github.io/commit/a07223dfda18ebae2a09a8c43e11e7365c3ccc90). Key clauses (all MEASURED):
  - use "for personal, non-commercial research purposes only"
  - "Any commercial use, sale, or other monetization is prohibited."
  - "You must not modify, reverse engineer, decompile, or create derivative works"
  - "YOU MAY NOT DISTRIBUTE, PUBLISH, OR REPRODUCE A COPY" without written permission
  - others must "register as an individual user"; California law; indemnify Stanford.
  - The Redivis "Stanford_Research_Agreement" text itself is only visible after sign-in (UNVERIFIED that it is identical; INFERRED it is the same template).
- Attribution: Redivis/AIMI asks to acknowledge Stanford AIMI with a fixed statement (MEASURED, https://stanford.redivis.com/AIMI).
- (d) Individual, free: the gate is an individual Google-account click-through, no SO named (MEASURED). Approval mode UNVERIFIED. Owner-terms conflict is the strongest of all sets: the "no derivative works" and "monetization" clauses vs training a prize model whose weights must be released (INFERRED). Kaggle mirrors (cjinny/mrnet-v1, 2019, licence "Unknown"; rinkusadhd24csa005/mrnet-knee-mris, labelled CC0) are redistributions the RUA forbids; they do not change the terms (MEASURED licence fields via https://www.kaggle.com/api/v1/datasets/view/...; conclusion INFERRED).

### 2.4 SKM-TEA (Stanford AIMI)
Sources: https://github.com/StanfordMIMI/skm-tea (MEASURED), https://stanford.redivis.com/datasets/5r8z-achpw6q4f (MEASURED), legacy https://stanfordaimi.azurewebsites.net/datasets/4aaeafb9-c6e6-4e3c-9188-3aaaf0e0a9e7 (MEASURED).
- Content: 155 qDESS knee scans, raw k-space + DICOM, 6-tissue segmentations, "bounding boxes for 16 pathologies" (MEASURED). Size: "900GB in a compressed format" (MEASURED, DATASET.md).
- (a) Same two Redivis requirements as MRNet: AIMI membership + "Stanford_Research_Agreement per dataset" (MEASURED). Approval mode UNVERIFIED.
- (b) Free non-commercial; same commercial licence regime (MEASURED).
- (c) "STANFORD UNIVERSITY SCHOOL OF MEDICINE SKM-TEA DATASET RESEARCH USE AGREEMENT": identical clauses to MRNet (personal non-commercial, no monetization, no derivative works, no redistribution) (MEASURED). Repo code is MIT; the data are not (MEASURED).
- (d) Same as MRNet.

### 2.5 OAI, Osteoarthritis Initiative (NIMH Data Archive / NDA)
Sources: https://nda.nih.gov/oai, https://nda.nih.gov/oai/query-download, https://nda.nih.gov/nda/access-data-info, https://nda.nih.gov/nda/standard-operating-procedures (all MEASURED in browser), https://www.oaiknowledgebase.com/about/FAQs/ (MEASURED), OAI CORE guide v1.0 dated June 3, 2026 https://www.oaiknowledgebase.com/siteassets/pdfs/step-by-step-guide-to-using-oai-data.pdf (MEASURED), https://www.re3data.org/repository/r3d100011843 (MEASURED), https://datacatalog.med.nyu.edu/dataset/10162 (MEASURED).
- Content: 4,796 subjects, "almost 26,626,000 images", bilateral knee MRI annually (MEASURED).
- Banner on every NDA page: "under review for potential modification in compliance with Administration directives" (MEASURED). Access could change before 10-22 (INFERRED).
- (a) Access route:
  - OAI page: "After registering and obtaining a password, you may browse, download" (MEASURED); need "an NDA user account" (MEASURED).
  - OAI CORE FAQ: "You MUST complete the Data Use Agreement before accessing OAI data"; steps: Login.gov, My Account, Data Permissions, "Osteoarthritis Initiative", Request Access, Start Request (MEASURED). Guide: "Create an NDA account and complete the Data Use Certification (DUC)" (MEASURED).
  - NDA has three permission-group types. Open Access: "you only need an NDA account" and requests "are approved upon submission" (MEASURED, access-data-info). SOP-04B (Rev 3, eff. 1/14/2025): user "clicks to accept the Terms and Conditions", no Institutional Sponsorship, valid one year (MEASURED). Broad Use / Controlled: eRA Commons PI role, Signing Official, FWA, NIH Data Access Committee review, requester must be faculty/senior researcher, not students (MEASURED).
  - NDA account creation itself is reviewed: "NDA staff review the account information provided", 1-3 business days (MEASURED, SOP-01). access-data-info also says "An NDA account with an institutional email is required to request data access" (MEASURED; unclear if this applies to Open Access groups).
  - Which type the OAI group is: NOT shown on any public page I could reach (UNVERIFIED). Evidence for Open Access: Login.gov path in the OAI CORE steps, re3data lists access restriction "registration" (not institutional membership), NYU catalogue says "Free to All" (all MEASURED). Evidence against: competitors in 741819 claim it "requires institutional sign off" (MEASURED as an opinion, Tucker Arrants, not host). My lean: OAI is an Open Access click-through group (INFERRED, medium confidence).
- (b) Cost: free (MEASURED, NYU catalogue "Free to All").
- (c) Terms: NDA Data Use Certification (form last updated July 2026; this is the institutional version, the Open Access click-through text was not visible without login). Purpose "scientific investigation, scholarship or teaching, or other forms of research and research development" (MEASURED, https://s3.amazonaws.com/nda.nih.gov/cms/prod/NDA-Data-Access-Request-DUC-FINAL.pdf). No commercial-use ban; it even refers to "any resulting commercial product(s)" (MEASURED). No distribution of data to people not on a DUC; do not publish derived data that could aid re-identification; acknowledge NDA/OAI; destroy data at close-out (MEASURED). OAI acknowledgement wording given on the OAI CORE FAQ (MEASURED).
- (d) UNCERTAIN. If OAI is Open Access, an individual can get it free after an NDA staff account check, with no SO: that would pass the host test. If it is Broad Use, it fails. Of all sets, OAI's own terms are the friendliest to a prize competition (no non-commercial or no-derivatives clause) (INFERRED from the DUC text).

### 2.6 KneeMRI (Stajduhar et al., Clinical Hospital Centre Rijeka)
Sources: official page http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/ returns HTTP 403 today, https variant 404 (MEASURED by curl, 2026-10-04). Last good archive copy: https://web.archive.org/web/20230510103617/http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/ (MEASURED). Wayback shows 301 redirects in 2024-2025 (MEASURED via CDX).
- Content: 917 sagittal PD-FS volumes, Siemens Avanto 1.5T, ACL label healthy / partial / complete rupture, plus ROI box (MEASURED, archived page).
- (a) Access: was direct download of vol01..vol10.7z (~3.2 GB), no form or account (MEASURED, archived page). The official source appears gone now (MEASURED status codes; "gone" is INFERRED).
- (b) Free.
- (c) Licence: "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)" (MEASURED, archived page). Attribution: acknowledge Clinical Hospital Centre Rijeka and cite Stajduhar et al., CMPB 140 (2017) 151-164 (MEASURED). CC BY-NC-ND permits copying and use for non-commercial purposes; it forbids sharing adapted material and commercial use (INFERRED from the standard licence). Whether training a model creates "Adapted Material", and whether a cash-prize entry is "NonCommercial", are open legal questions (INFERRED).
- (d) Yes historically (fully open, no gate). Today only mirrors exist, e.g. https://www.kaggle.com/datasets/sohaibanwaar1203/kneemridataset (2019, licence "Unknown") and the new one below (MEASURED via Kaggle API).

### 2.7 Kaggle "Knee MRI 917" (ibrahimshaikhdev/knee-mri-917)
Source: https://www.kaggle.com/datasets/ibrahimshaikhdev/knee-mri-917 and https://www.kaggle.com/api/v1/datasets/view/ibrahimshaikhdev/knee-mri-917 (MEASURED, read-only, nothing downloaded).
- Declared licence: "CC0: Public Domain" (MEASURED). Description: "No description available"; provenance sources and citations empty (MEASURED).
- Version 1 "Initial release" created 2026-10-03T10:59Z; 1 vote, 3 downloads at time of reading (MEASURED). totalBytes 6,673,316,292 (about 6.7 GB unpacked; the "4.1 GB" figure is presumably the zip) (MEASURED / INFERRED).
- Files: metadata.csv (42.41 kB) with columns examId, seriesNo, aclDiagnosis, kneeLR, roiX, roiY, roiZ, roiHeight, roiWidth, roiDepth, plus volumetric_data/volumetric_data/<examId>-<seriesNo>.pck (MEASURED; the public file listing returned at least 599 .pck files before pagination ended).
- Provenance: aclDiagnosis histogram shows 690 / 172 / 55, exactly the KneeMRI split healthy 690, partial 172, ruptured 55; kneeLR 448 / 469; same metadata columns and .pck format as the original archive (MEASURED numbers; identification as a KneeMRI re-upload INFERRED, high confidence).
- Licence conclusion: a re-uploader cannot relicense CC BY-NC-ND material as CC0, so the original CC BY-NC-ND 4.0 terms still govern the content (INFERRED). The upload itself also drops the required attribution (INFERRED).
- (d) Downloadable by anyone on Kaggle for free, so the access test is trivially met; the owner-terms issue is the same as KneeMRI.

### 2.8 KneeXNet-2.5D
Sources: https://github.com/pitthexai/Knee_MRI_Segmentation_2.5D (MEASURED), paper npj Health Systems 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC13354193/ (MEASURED via fetch summary).
- What it is: a 2.5D U-Net framework for knee cartilage and meniscus segmentation on sagittal OAI MRIs; 222 scans from 218 OAI participants (MEASURED via PMC summary). Repo has code, annotation guideline, masks, model weights, an app (MEASURED).
- (a) Open GitHub, no gate (MEASURED). Fully_annotated_dataset holds KneeMRI_Annotations.zip (about 12 MB, NIfTI masks) and a README: "users must individually download the corresponding MRI or X-ray data from the official OAI repository" (MEASURED). So images are NOT redistributed; only masks.
- (b) Free.
- (c) No LICENSE file; GitHub API licence = null (MEASURED). README: "publicly and freely available for any research and educational purposes" (MEASURED). Paper code statement adds "academic purposes" (MEASURED via PMC summary). Without a licence file, default copyright applies beyond what the README grants (INFERRED).
- (d) Masks and weights: yes, freely. Images: only through OAI (see 2.5). Labels are segmentations, not abnormality labels (MEASURED/INFERRED).

### 2.9 MOST, Multicenter Osteoarthritis Study (NIA AgingResearchBiobank)
Sources: https://agingresearchbiobank.nia.nih.gov/studies/most/details (MEASURED via fetch), User's Guide (July 2026 cover, 03/2026 footer) https://agingresearchbiobank.nia.nih.gov/static/docs/Users_Guide_07_17_2026.f6b33a3270b1.pdf (MEASURED).
- Content: knee radiographs and "knee MRIs from the MOST cohort through the 168-month follow-up" (MEASURED).
- (a) Requests must come from a PI/senior researcher who is a permanent employee "at a minimum, a tenure-track professor"; excludes "post-docs or graduate students" (MEASURED). IRB approval attached; NIA "Biobank Scientific Review", "could take several weeks"; transfer agreement for "the P.I. and an institutional signing official (SO) to sign" (MEASURED).
- (b) Data free; biospecimens carry processing fees (MEASURED).
- (c) "Materials from this study are not available for commercial use." (MEASURED).
- (d) No. Fails the host test clearly (institutional review + SO).

### 2.10 KneeCoT (already ruled out by host, for contrast)
Source: https://huggingface.co/datasets/YiHui0124/KneeCoT (MEASURED via fetch). CC-BY-NC-4.0, gated, requires "ethical use agreement with the affiliated hospital". Host banned it in 734109 (MEASURED).

## 3. What this means (INFERRED, my reading)

- Host access test (741819) vs each set:
  - Pass on access: fastMRI (+fastMRI+ labels), KneeMRI and its Kaggle mirrors, KneeXNet masks/weights.
  - Probably pass on access, approval mode not verifiable without sign-in: MRNet, SKM-TEA (individual Google-account click-through on Redivis).
  - Unclear: OAI (hinges on whether the NDA OAI group is Open Access; NDA staff vet the account).
  - Fail: MOST, KneeCoT.
- Owner-terms layer (the host refuses to rule on it, the team carries the risk):
  - Worst conflict: MRNet and SKM-TEA ("no derivative works", no monetization).
  - Next: fastMRI (internal research only, no monetizing, no transfer of derived data files, no linking with other datasets).
  - KneeMRI / Knee MRI 917: CC BY-NC-ND (non-commercial + no adapted material).
  - Friendliest: OAI (DUC has no non-commercial or no-derivatives clause).
- The rules' Winner License carve-out ("incompatible license" input data need not be open-sourced) covers the data, not the obligation to release the trained Submission (INFERRED). Every NC/ND/RUA set therefore creates a tension for a prize-eligible finish, even when access passes.
- No host has given a per-dataset yes/no for MRNet, SKM-TEA, fastMRI, KneeMRI or KneeXNet as of 2026-10-04. Thread 743416 (YYama, 62 votes, posted about 2026-09-25) asks exactly that; I saw no host reply among its 12 comments (MEASURED, https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416).

## 4. Open questions
- Is the NDA "Osteoarthritis Initiative" permission group Open Access (click-through, auto-approved) or Broad Use (SO + DAC)? Needs a logged-in view of the NDA Data Permissions dashboard, or an NDA Help Desk answer.
- Are the Redivis "AIMI membership" and "Stanford_Research_Agreement" requirements auto-approved or reviewed by AIMI staff? Needs a signed-in view; the agreement text on Redivis was not readable without sign-in.
- Does the fastMRI form accept an independent (non-institution) applicant and approve automatically? Page does not say.
- Is the KneeMRI official page permanently gone, or moved? Author's new homepage has no link.
- Would the host treat NC / ND / "no derivative works" owner terms as disqualifying for prize winners, given 2.5 and 2.8? Not answered.
