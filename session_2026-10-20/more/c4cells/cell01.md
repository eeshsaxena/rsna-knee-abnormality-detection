## Credits and required acknowledgement

Built on dreaddevelopment's [Raptor CoAtNet/MIL inference recipe and public baseline weights](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense). These are our retrained checkpoints, rather than those baseline weights. Weak-label sources include [stevenleehans](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels), [pilkwang](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels), and [riadmohamed42's JEV labels and verification](https://www.kaggle.com/code/riadmohamed42/jev-knee-labels-verification), combined with our report-derived labels. Source reports and study-level label tables are not included here. The CoAtNet backbone comes from timm. Competition data is provided under the [RSNA Knee competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules).

Data and/or research tools used in the preparation of this manuscript were obtained and analyzed from the controlled access datasets distributed from the Osteoarthritis Initiative (OAI), a data repository housed within the NIMH Data Archive (NDA). OAI is a collaborative informatics system created by the National Institute of Mental Health and the National Institute of Arthritis, Musculoskeletal and Skin Diseases (NIAMS) to provide a worldwide resource to quicken the pace of biomarker identification, scientific investigation and OA drug development. Dataset identifier(s): 10.15154/0hcg-f676.

[Shared NDA Study](https://nda.nih.gov/study.html?id=3407) 路 [DOI: 10.15154/0hcg-f676](https://doi.org/10.15154/0hcg-f676). Official acknowledgement source: [NDA manuscript preparation](https://nda.nih.gov/nda/manuscript-preparation).


**This notebook's part** is only the second-checkpoint stage and its weight. All models and the base
pipeline are the work of the authors credited above; please upvote their notebooks and datasets too:
[nartaa's 0.949](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror) and
[0.945](https://www.kaggle.com/code/nartaa/rsna-knee-0945-efficient-224crop) notebooks,
[heliosli's R7 notebook](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click),
and dreaddevelopment's Raptor recipe.

If this moved your rank, an upvote helps others find it. Questions and corrections welcome.