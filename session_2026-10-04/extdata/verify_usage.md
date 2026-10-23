# Adversarial verification of the "usage" research (RSNA Knee external knee MRI data)

Verifier run 2026-10-04. I re-opened every cited source myself in my own built-in-browser tab (tab-5). I was signed out the whole time and declined nothing beyond what loads by default. I made no posts or votes and filled no forms. I used Kaggle CLI 2.2.4 only for list and file-listing calls; no dataset bytes were downloaded. I fetched two rendered public notebook pages (`__results__.html`) to search their code text.

Raw outputs are in `verify_usage/`:
- `cli_hidden_members.txt`
- `cli_lists.txt`
- `files_*.txt` and `*.norm` (file-name and size comparison)
- `cli_kernels_kneemri.txt`
- `sakh_results.html`, `juej_results.html`

Kaggle hover times are in IST. I converted them to UTC below.

Tags: MEASURED means I read it today on the page or in CLI output. INFERRED means it is my reading.

## Verdict

The research is accurate on every quote, date, author role and dataset fact I re-checked. I found no host or staff statement that contradicts it, and nothing newer. The host profile javaduke95 shows 19 posts, the newest being the 2026-09-21 OAI reply. The seven "unanswered" threads (735497, 737950, 738111, 738156, 743420, 735121, 745283) still have 0 host and 0 staff comments.

Corrections and additions:
1. **Partly refuted: "the hosts have ruled on only two datasets."** The 08-27 host reply answers UGUR OZER, who named OAI, MRNet, fastMRI+ and SKM-TEA. It opens with "use of these datasets would be considered non-commercial". So on the non-commercial question, the host did cover those four. Accessibility was not ruled per dataset.
2. **Refuted: the evidence file's INFERRED note that VinDr-Mammo sits behind PhysioNet credentialing.** PhysioNet says "Only registered users who sign the specified data use agreement can access the files." That is the Restricted tier: a registration plus a DUA, with no credentialing (https://physionet.org/content/vindr-mammo/1.0.0/, MEASURED). So the 2023 Breast 3rd-place prize team used a registration-plus-DUA dataset (MEASURED) and kept its place, as far as the visible record shows (INFERRED).
3. **Missed in the Breast precedent.** The same host reply (Chris Carr, COMPETITION HOST) also rejected the NYU-based models on license grounds. Their BSD2 license "fails to meet" the clause that winning models must have a license "that in no event limits commercial use". The Knee rules contain the same phrase in 2.5(a), but they add the incompatible-license carve-out that the 2023 Breast rules lack (MEASURED on both rules pages).
4. **Weak inference.** Richard_61's line "every team using no-redistribution external data" is phrased conditionally. It does not show that any team actually uses such data.
5. **Download count is misleading.** techtech69's MRNet re-upload shows 2,321 downloads but only 30 views ("77.36667 downloads per view"). The chart has one spike of about 2,185 on about 2026-09-24 (my reading of the SVG chart). INFERRED: these are scripted pulls, so the count says little about how many teams use MRNet.

## Claim-by-claim

| # | Claim (short) | Status | What I found |
|---|---|---|---|
| 1 | 2.6(b) "acceptable unless specifically prohibited by the Host"; 2.6(a) public/equal/no-cost or Reasonableness | CONFIRMED | Exact text at https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules. Missed text: the Host "will employ an assessment of whether or not the following criteria can exclude" a dataset. Also: buying a proprietary dataset costing more than a prize "would not be considered reasonable". |
| 2 | Overview: "Freely & publicly available external data is allowed, including pre-trained models"; no approved list | CONFIRMED | https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview. Host asks winners to "publish a link to your open sourced code and the weights" and to share the final model publicly. Deadlines: final 2026-10-22, winners' requirements 2026-11-05. There is no external-data disclosure deadline. |
| 3 | 2.8 weights "as a public kaggle dataset"; 2.5(a) carve-out | CONFIRMED | Both present. The carve-out covers "that data and/or model(s)", meaning the incompatible inputs. On a plain reading it does not cover the winner's own weights (INFERRED). 2.8 also requires representing "the unrestricted right to grant that license". The CC-BY-NC 4.0 winner license is self-contradictory: the rules call it a license that "in no event limits commercial use". |
| 4 | KneeCoT banned, 734109 | CONFIRMED | Po-Hao "Howard" Chen, COMPETITION HOST, 2026-08-25 11:00:25 UTC: "the KneeCoT dataset is not allowed in this competition". The question was posted 2026-08-09 23:34:53 UTC. The HF page today is still gated and still says users "must sign a formal ethical use agreement with the affiliated hospital", so the ban stands. |
| 5 | Host general test, 733965 | CONFIRMED | Same host, 2026-08-27 22:14:33 UTC (exact): "straightforward registration or click-through data-use agreement would generally satisfy". Nuances the research left out: (a) it answers UGUR OZER's four named datasets on the NC point; (b) the barrier test applies "particularly given the remaining competition timeline"; (c) teams are responsible for "any required release of code or models". Nothing later supersedes it. |
| 6 | OAI conditional, 741819 | CONFIRMED | Host, 2026-09-21 15:34:26 UTC, exact wording as quoted. Tucker Arrants' "It is a no" was posted 2026-09-21 20:56:27 UTC. He now shows as 45th, not 42nd. Amanie El-Hajj (733652, 10-02) reads the same reply as a yes, so competitors disagree. |
| 7 | YYama 743416, no host reply | CONFIRMED | Posted 2026-09-26 01:52:06 UTC, 62 votes. Header says "12 Comments" but 13 are visible, none from the host. The latest is Eugene Khvedchenya, 2026-10-03 16:53:51 UTC. |
| 8 | Mazzutti 733652 disjunction argument | CONFIRMED | 2026-08-20 09:27:03 UTC. He shows 20th in-thread and 21st on the LB. Partly superseded by the KneeCoT ban (08-25) and the OAI condition (09-21). |
| 9 | Richard_61 743420 | CONFIRMED (quote); inference WEAK | Posted 2026-09-26 03:15:55 UTC, follow-up 2026-09-29 02:15:55 UTC. Quote accurate. No host reply. See correction 4. |
| 10 | No visible top-12 member has public competition artefacts; five have zero posts | CONFIRMED and EXTENDED | Profile Discussion tabs for rhoskeri, hettpugymajlhuk, wassimdobbi, tom1115 and nmarks show "No results found". I also checked 12 team members the research did not list: wadakoki, nantohock, syamoji141chan, shin9915, shuyaakagawa, tomoyamiyazawa26, sotakurihara1, sugiuraakie, t88take, takashisomeya, shunyaiida, takeharuyoshikawa. None has a public RSNA-knee notebook or dataset. New context: Pa3BaJluHa (2nd, 31 entries) shows "Joined 5 days ago", so its silence means little. |
| 11 | Top posters credit labels and pseudo-labels, 735304 | CONFIRMED | tennogh (5th): "OOF pseudo-labels have been pretty well correlated with LB". Scott Willis (10th): "I used Gemma 4 locally to extract labels", 0.949 single fold. CoolinLai (14th): ResNet50 5-fold 0.954. Parag (13th): FOV tips. After expanding all replies, there are 0 mentions of MRNet, OAI, fastMRI or KneeMRI. |
| 12 | No public evidence top teams train on external knee MRI (INFERRED) | CONFIRMED as stated | Discussion searches for "MRNet" and "pretraining" return only rules-clarification threads. YYama's quote is a worry, not a disclosure. Absence does not prove non-use. |
| 13 | techtech69 MRNet re-upload, CC0, 2,321 downloads | CONFIRMED with caveat | Page shows "CC0: Public Domain", "Version 1 (9.51 GB)", 4,670 files, train and valid folders, "Code (0)". CLI: 7,566,126,617 B, 2026-08-27 13:42 UTC. See correction 5 on the download count. |
| 14 | ibrahimshaikhdev/knee-mri-917 is a KneeMRI (Rijeka) re-upload | CONFIRMED (strong) | All 918 files (917 .pck plus metadata.csv) match techtech69/rijeka-knee-mri-dataset by name and byte size. Rijeka has 3 extra files: READ.ME, example.py, example.pck. Those are exactly the extra files of the Zenodo archive (https://zenodo.org/records/14789903), which says "CC BY-NC-ND 4.0", is Open access, and was published 2017. Kaggle page: CC0, "No description available", updated 2026-10-03 10:59:34 UTC. |
| 15 | tsuyu122 harmonized labels, fake "Rule 4.b" | CONFIRMED | Subtitle and "Kaggle Competition Rule 4.b (AI Provenance & Open Weights)" are present. The rules have no such clause (2.4.b is Data Security). 4,407 + 1,248 + 2,976 = 8,631. |
| 16 | quannguyen0501 MST_MRNet.ckpt | CONFIRMED (listing); training data still INFERRED | License "Unknown", 275.6 MB, 0 downloads, no description. |
| 17 | sakhawathossen notebook attaches kneeMRIdataset but never uses it | CONFIRMED | The /input page lists "kneeMRIdataset". Rendered code has 0 hits for kneemri, .pck, metadata.csv or mrnet. "External" there means public notebook families. The notebook was copied from tonylica and also attaches mattiaangeli/knee-mri-fold-weights, whose training data is undocumented. CLI: 26 notebooks run in 2026 attach the 2019 KneeMRI upload, so "about 25" holds. |
| 18 | juejiezeng notebook = MRNet architecture, not MRNet data | CONFIRMED in substance | The docstring names "MRNetStyleClassifier" and says the model was trained on PNGs exported from DICOM by export_dicom_png.py. MRNet ships as .npy, not DICOM. The quote in the claim is a paraphrase, not verbatim. Weights come from a "[Private Dataset]". Public score 0.766, so it is irrelevant to top teams. |
| 19 | IAD 2025 precedent, 608179 | CONFIRMED | 2.6 text is identical on the IAD rules page. JeffRudie (COMPETITION HOST), 2025-09-22 16:50:58 UTC: "we have decided that these do not qualify". He added that his own request without institution or PI details "was rejected". On 2025-09-24 15:36:54 UTC he approved the OpenNeuro sets "without any restrictions". The IAD deadline was 2025-10-14, so the ruling came about 3 weeks before the end. |
| 20 | IAD 2nd and 6th used only open or approved data | CONFIRMED | 2nd: "The LargeIA dataset was excluded from training" (TopCoW Zenodo plus TopBrain; the team includes TopCoW organizers). 6th: OpenNeuro data is listed under things that "did not really help CV but were kept for robustness". The 1st-place IAD writeup mentions no external data at all. |
| 21 | Breast 2023 NYU ruling, 379133 | CONFIRMED; part missed | Chris Carr (COMPETITION HOST), 2023-02-24 20:16:34 UTC. The deadline was 2023-02-27 23:59 UTC. The question had been posted 2023-01-18, so the answer came 5 weeks late. See correction 3 for the license ruling the research missed. The 3rd-place post (391779) lists "CBIS-DDSM, CMMD, Vindr and Mini-DDSM". |
| 22 | Lumbar 2024 / ATD 2023 rules | CONFIRMED | Both rules pages contain "publish a link to your open sourced code on the competition forum with external datasets made public". ATD overview has the same "Freely & publicly available" line. INFERRED: the Knee rules do NOT carry this "external datasets made public" clause. The Knee duty is to publish code and weights. |

## Missed or worth adding (all MEASURED unless marked)

1. **Breast 2023 license-chain ruling (correction 3).** It is the only precedent where an RSNA host policed the license of a pretrained model at prize time. It bears directly on weights trained on NC or ND data. The Knee 2.5(a) carve-out softens this, but the 2.8 "unrestricted right" representation remains. Rules 3.14.a ("sole and exclusive owner", no breach of "confidentiality obligations") and 3.9.b (one-week remediation incl. "resolution of license conflicts") also remain. The earlier verify_rules.md also flags 3.4.d and 3.8.e (disqualification of any entrant, removal of medals).
2. **VinDr correction (correction 2).** It supports the host's 08-27 test: a registration plus a signed DUA has survived an RSNA prize audit before (INFERRED).
3. **External data was not decisive in the closest precedent.** In IAD 2025, 1st place used no external data, and 6th place said OpenNeuro "did not really help CV".
4. **The host has gone quiet since 09-21 but is still active.** The profile says "last seen in the past day". Precedent shows RSNA hosts answer late (Breast: 5 weeks after the question, 3 days before the end). A late ruling before 2026-10-22 is possible.
5. **Hidden members and the 5-day-old 2nd-place account** (claim 10).
6. **MRNet download spike** (correction 5).
7. **2.6(b) after-the-fact exclusion language** (claim 1). "Acceptable unless specifically prohibited" is paired with an explicit Host power to "exclude" a dataset on assessment, so the absence of a ban today is not a safe harbor.

## Sources opened today (all signed out)

- Rules: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules
- Overview: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview
- Leaderboard: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard
- Competition threads: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/{743416, 743420, 741819, 733965, 734109, 733652, 735304, 737950, 745283, 735121, 738111, 738156, 735497, 733343, 745820}
- Profiles: https://www.kaggle.com/{rhoskeri, hettpugymajlhuk, wassimdobbi, tom1115, nmarks, tennogh, javaduke95}/discussion
- Datasets: https://www.kaggle.com/datasets/{ibrahimshaikhdev/knee-mri-917, techtech69/mrnet-knee-mri-stanford-aimi, tsuyu122/rsna-knee-mri-balanced-labels-public, quannguyen0501/dinov2-s-mrnet}
- Notebooks: https://www.kaggle.com/code/sakhawathossen/rsna-knee-enhanced-ensemble/input and https://www.kaggle.com/code/juejiezeng/rsna-knee-mrnet50-inference
- IAD 2025: https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/608179, /rules, /writeups/{1st,2nd,6th}-place-solution
- Breast 2023: https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/379133, /discussion/391779, /rules
- Lumbar 2024 rules: https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/rules
- ATD 2023: https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/overview and /rules
- External: https://zenodo.org/records/14789903, https://huggingface.co/datasets/YiHui0124/KneeCoT, https://physionet.org/content/vindr-mammo/1.0.0/
- Could not verify: https://aimi.stanford.edu/datasets/mrnet-knee-mris. The "Terms & Conditions" and "License" tabs render no text when signed out, so MRNet's no-derivatives clause rests on competitor quotes only (UNCERTAIN).
