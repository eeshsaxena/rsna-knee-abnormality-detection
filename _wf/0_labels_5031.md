FINDINGS — train.csv gold-58 analysis (RSNA Knee)

**(1) POS/NEG COUNTS AMONG THE 58 GOLD** (all 12 labels filled; 0 studies with zero positives — the gold set is abnormality-enriched, not population-representative)

| Label | pos | neg | prev |
|---|---|---|---|
| Effusion | 35 | 23 | .603 |
| Synovitis | 27 | 31 | .466 |
| Medial Meniscus | 26 | 32 | .448 |
| ACL | 24 | 34 | .414 |
| Lateral Meniscus | 23 | 35 | .397 |
| PF OA | 21 | 37 | .362 |
| Contusion | 19 | 39 | .328 |
| Fracture | 18 | 40 | .310 |
| Medial OA | 15 | 43 | .259 |
| Baker's | 12 | 46 | .207 |
| Lateral OA | 11 | 47 | .190 |
| MCL | 9 | 49 | .155 |

Rarest gold positives: MCL (9), Lateral OA (11), Baker's (12). These three drive macro-AUC variance — a handful of mislabels swings their AUC.

**(2) CO-OCCURRENCE / CORRELATION** (phi/Pearson on binary; mean 4.14 positive labels per study, range 1–9)
- Strong positive clusters (labels that travel together): **Medial OA & Baker's r=+.48**; **degenerative axis** — Medial Meniscus↔Medial OA (+.42), Lateral Meniscus↔Lateral OA (+.42), Lateral OA↔Baker's (+.40), Medial OA↔PF OA (+.37); **inflammatory** Effusion↔Synovitis (+.40); **trauma axis** ACL↔Contusion (+.38), Contusion↔Fracture (+.33).
- Notable negatives: Medial OA↔Contusion (−.33), PF OA↔Contusion (−.30), ACL↔PF OA (−.27) — degenerative and acute-traumatic presentations are largely mutually exclusive.
- Implication for heads: labels are NOT independent; two latent groups dominate (chronic-degenerative vs acute-traumatic). A shared trunk + 12 sigmoid heads is right, but you should (a) exploit correlation — the OA/meniscus co-firing lets a weak head borrow signal from its partner, and (b) NOT calibrate thresholds independently. Consider grouping the 4 OA/meniscus labels and the Effusion/Synovitis pair when stacking or ensembling report-vs-image predictions.

**(3) CV GROUPING KEY — NONE EXISTS beyond StudyInstanceUID.** Inspected carefully:
- StudyInstanceUID: all 4407 share the identical anonymization org-root `1.2.826.0.1.3680043.8.498.*` (pydicom generate_uid root); the trailing 40-digit component is a random hash carrying **zero** patient/site/scanner/date signal. Every study is unique — no repeat-patient leakage risk, but also no natural group.
- No date, patient-id, site, or scanner column in train.csv or train_series.csv. train_series.csv adds only SeriesInstanceUID, Fluid_Sensitive, Fat_Suppression, Anatomical_Plane — all every study has the same {Axial, Coronal, Sagittal} plane set, so plane gives no grouping.
- **Recommended honest-CV proxies** (build these, no native key):
  1. **Report language = strongest institution proxy.** Detected split: ES 2163 / EN 1742 / NL 502 (gold-58: EN 28 / ES 26 / NL 4). Language almost certainly maps to source country/institution → this is the domain shift the private test may probe. Use **StratifiedGroupKFold grouped by detected language**, or explicitly hold out one language as an OOD fold to estimate generalization. WARNING: language is only in train reports, NOT at test time — so use it for CV honesty, never as a model feature.
  2. Acquisition-protocol fingerprint (n_series per study: 4→675, 5→2299, 6→698, 7→310, tail to 14) is a weaker secondary site proxy if you need finer groups within a language.
- For the gold-58 specifically: n=58 is too small for grouped CV; report **per-label AUC + macro AUC with the full gold set as a fixed eval set** and use bootstrap 95% CIs (rare labels MCL/LatOA/Baker's will have wide CIs — quote them). Do not trust a single macro number to 3 decimals on n=58.

**(4) CLASS IMBALANCE → LOSS WEIGHTING.** BCE pos_weight (neg/pos) on gold: MCL 5.44, Lateral OA 4.27, Baker's 3.83, Medial OA 2.87, Fracture 2.22, Contusion 2.05, PF OA 1.76, Lat Men 1.52, ACL 1.42, Med Men 1.23, Synovitis 1.15, Effusion 0.66 (majority-positive).
- **Critical caveat: gold-58 prevalence is NOT the deployment prevalence.** Rule-v0 over all 4407 gives far lower rates (ACL .10, Fracture .05, Lateral OA .03, MCL .05, Synovitis .09 vs gold .41/.31/.19/.16/.47). The gold set was cherry-picked for pathology. Whatever you use for pseudo-label training (Qwen over 4407) will be much more imbalanced — derive pos_weight from the **pseudo-label training distribution, not from gold-58**, or you will badly under-weight the true rare classes (Fracture, Lateral OA, MCL).
- Because the metric is macro-AUC (threshold-free, ranking-based), extreme class weighting matters less than for F1; AUC is fairly robust to prior shift. Prefer mild weighting or focal loss (γ≈1.5–2) over aggressive pos_weight, which mainly hurts calibration without helping ranking. Priority for label-quality effort = the rare/high-pos_weight trio MCL, Lateral OA, Fracture, since each is ~1/12 of macro-AUC yet has the fewest positives to learn from and the noisiest pseudo-labels.

Files used: C:/Users/eeshs/Downloads/rsna-knee/_data/train.csv, C:/Users/eeshs/Downloads/rsna-knee/_data/train_series.csv, C:/Users/eeshs/Downloads/rsna-knee/_data/pseudo_labels_rulev0.csv