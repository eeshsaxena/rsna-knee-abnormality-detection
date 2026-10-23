# raw notes (2026-10-04)
## fastMRI  https://fastmri.med.nyu.edu/  (MEASURED via curl)
- form fields: Full Name, Email, Institution (required), "By entering my electronic signature here"
- "internal research or educational purposes only"; "without charge"; "Not SELL OR OTHERWISE MONETIZE"
- Not DISTRIBUTE...COPY OR OTHERWISE TRANSFER ... "any subsequent variables or data files derived from the fastMRI Dataset"
- "link the fastMRI Dataset with any other dataset" (no-attempt clause, in re-identification sentence)
- "must register as an individual user"; destroy all copies when done
- MLCommons/MLPerf footnote = non-commercial; no mention of Kaggle/competitions
- page text says application "must include the investigator's institutional affiliation and the proposed uses" (form has no 'proposed use' field)
## MRNet  (MEASURED in browser)
- Azure: https://stanfordaimi.azurewebsites.net/datasets/bface6fc-7859-47d7-a1c8-022cd6b17419 License "Stanford University Dataset Research Use Agreement" text: personal non-commercial; "Any commercial use, sale, or other monetization is prohibited"; no derivative works; no distribution
- Redivis now: https://stanford.redivis.com/datasets/4a2c-4cpkzrn2c ?access: Data access 0/2: "AIMI membership" Join + "Stanford_Research_Agreement per dataset" Apply; last updated 2026-04-23
- AIMI Redivis org https://stanford.redivis.com/AIMI : "Sign in using your Stanford ID or Google account, and join the Stanford AIMI organization."
## SKM-TEA (MEASURED)
- Azure https://stanfordaimi.azurewebsites.net/datasets/4aaeafb9-c6e6-4e3c-9188-3aaaf0e0a9e7 same RUA template; Redivis https://stanford.redivis.com/datasets/5r8z-achpw6q4f same 2 requirements
- repo code MIT (github StanfordMIMI/skm-tea) but data under RUA
## AIMI commercial page https://aimi.stanford.edu/datasets-commercial-use : orig license "restricted to non-commercial"; commercial license $70,000/yr/dataset (FY25), committee review up to 4 weeks
## fastMRI+ https://github.com/microsoft/fastmri-plus : MIT license on annotation CSVs (knee.csv covers raw knee set); README says fastMRI raw data obtained after "signing the data sharing agreement" for "academic purposes"; last push 2021-09-07
## OAI (MEASURED)
- https://nda.nih.gov/oai banner: "This repository is under review for potential modification in compliance with Administration directives."
- https://nda.nih.gov/oai/query-download : "After registering and obtaining a password, you may browse, download, and use the data for human subjects research for scientific and educational purposes." need NDA user account
- OAI CORE Knowledgebase FAQ https://www.oaiknowledgebase.com/about/FAQs/ : "You MUST complete the Data Use Agreement before accessing OAI data"; Login.gov -> My Account -> Data Permissions -> Request Access "Osteoarthritis Initiative" -> Start Request
- Step-by-step guide v1.0 June 3 2026 https://www.oaiknowledgebase.com/siteassets/pdfs/step-by-step-guide-to-using-oai-data.pdf : "Create an NDA account and complete the Data Use Certification (DUC)."
- NDA access-data-info https://nda.nih.gov/nda/access-data-info : 3 PG types; Open Access "only need an NDA account"; "Open Access requests are approved upon submission"; Broad/Controlled need eRA Commons PI + SO + FWA + NIH DAC review
- SOP-04B (Rev 3, eff 1/14/2025) https://nda.nih.gov/nda/standard-operating-procedures#sop4b : open access = "without the need for Institutional Sponsorship"; user "clicks to accept the Terms and Conditions"; valid 1 yr
- SOP-01 account creation: RAS login (eRA Commons, Login.gov, NIH smart card), "NDA staff review the account information provided", 1-3 business days
- re3data https://www.re3data.org/repository/r3d100011843 : access "restricted", restriction "registration" (updated 2024-06-19)
- NYU catalog https://datacatalog.med.nyu.edu/dataset/10162 : "Free to All"
- NOT directly measured: that OAI PG is Open Access type (INFERRED from Login.gov path + re3data registration + SOP-04B)
- NDA DUC (institutional, last updated July 2026) purpose "scientific investigation, scholarship or teaching, or other forms of research and research development"; mentions "any resulting commercial product(s)" (no commercial ban); no distribution to non-listed persons
## KneeMRI (Stajduhar) (MEASURED)
- official page now: http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/ -> HTTP 403 (curl 2026-10-04); https -> 404; author homepage moved to WordPress, no kneeMRI link found
- Wayback 2023-05-10 snapshot https://web.archive.org/web/20230510103617/http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/ : "licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)"; direct download links (vol01..vol10.7z, ~3.2GB), no form; 917 volumes; Siemens Avanto 1.5T PD FS sagittal; labels healthy/partial/ruptured ACL; acknowledge Clinical Hospital Centre Rijeka + cite Stajduhar 2017 CMPB
## Kaggle ibrahimshaikhdev/knee-mri-917 (MEASURED)
- page https://www.kaggle.com/datasets/ibrahimshaikhdev/knee-mri-917 : License "CC0: Public Domain"; "No description available"; provenance sources empty; version 1 "Initial release" 2026-10-03T10:59Z; 1 vote, 3 downloads
- API https://www.kaggle.com/api/v1/datasets/view/ibrahimshaikhdev/knee-mri-917 totalBytes 6,673,316,292
- files: metadata.csv (42.41 kB, cols examId seriesNo aclDiagnosis kneeLR roiX roiY roiZ roiHeight roiWidth roiDepth) + volumetric_data/volumetric_data/<examId>-<seriesNo>.pck
- aclDiagnosis histogram 690/172/55 == KneeMRI 917 (healthy/partial/ruptured); kneeLR 448/469 -> re-upload of KneeMRI; CC0 label conflicts with CC BY-NC-ND 4.0 (ND forbids redistribution of adapted material; CC licenses don't allow relicensing; NC)
## KneeXNet-2.5D (MEASURED)
- https://github.com/pitthexai/Knee_MRI_Segmentation_2.5D : no LICENSE file (GitHub API license null); README "publicly and freely available for any research and educational purposes"
- Fully_annotated_dataset/README.md: masks only (KneeMRI_Annotations.zip 12 MB, NIfTI), "users must individually download the corresponding MRI or X-ray data from the official OAI repository"
- paper npj Health Systems 2026 (PMC13354193): 222 sagittal knee MRIs from 218 OAI participants, cartilage+meniscus segmentation (not abnormality labels)
