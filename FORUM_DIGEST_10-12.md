# Forum + LB digest 2026-10-12 ~07:40 IST (Claude, read-only sweep)

## Leaderboard (downloaded 07:35; 5,797 teams)
- **Us: .951, rank 788** (was ~244 on 10-11). Public notebooks now score .954, so the .954 band holds **537 teams (ranks 159-695)**.
- Bands: .966 top; .964 = r4; .960 = r28; .958 = r54; .956 = r85; **.955 = r110-158**; **.954 = r159-695**; .953 = r696-731; .952 = r732-771; .951 = r772-864.
- Medal cuts (Kaggle rules, 1000+ teams): silver = top 5% (~r290), bronze = top 10% (~r580). Ties rank by submission time, so a NEW .954 lands at the END of that band (~r695): **outside bronze**. **.955 is now the floor for a medal (silver)**.

## Public notebooks (CLI, sort by score)
- Many ".954" blends posted 10-10 / 10-11: goodpjw2008 "CoAtNet + 2.5D ConvNeXt MIL | LB 0.954" (158 votes), Jenil "Dual SOTA Ensemble LB 0.954", heliosli "rsna-knee-blend-gold" (200 votes) and "Blend mine in, private LB luck to all upvoters" (137), Roman Tamrazov "DINOsaur the last?", Matterhorn "Velciraptor DINOsaur Speed", evgendvorkin "RSNA_Versia_7" (366 votes).
- heliosli blend inputs: datasets heliosli/rsna-knee-compact-r6cnxfsss, nartaa/rsna-knee-publication-swa-weights-20261007 (OAI-trained), heliosli/rsna-orthofoundation-fold0-epoch7-20261004. goodpjw2008 adds goodpjw2008/rsna-knee-2-5d-convnext-reader + heliosli/rsna-knee-compact-v1.

## Discussion (new since the 10-10 sweep)
- **748276 "Clarification on eligibility of OAI dataset"** (dragoshh1984, 16 h ago, 13 votes, NO host reply yet): OAI is on NDA, and since 2025-04-04 NIH blocks NDA access from China, Russia, Iran, etc.; China is not excluded by Rule 3.1; Rule 2.6.b lists geo restrictions as a reason external data is not "reasonably accessible to all" (the reason KneeCoT was ruled out). **Risk: a host ruling against OAI would hit our T123 / PROBE4 sets AND N (nartaa, OAI-trained) AND most public blends.** Watch this thread daily.
- 748295 "Stuck at .954": rank-5 user diet1236364 (2 h ago): "don't use external data ... follow the established rules". Hint that top teams do not use OAI and that rule-following matters.
- 748181 "public notebooks" (Cody_Null, 63 votes): late public .954 blends; people expect a .955-.956 blend soon.
- 747613 "The core to reach top score without any ensemble" (-7), 745861 hengck23 "Tricks" (35 comments, active) : not read in detail this sweep.

## Implications
1. Our .951 is now far below the public-notebook floor. The fastest lever is **our fleet F blended with the best public blend in place of N (.949)**; F .40 / N .60 was +.002 over N alone.
2. Every final pick should consider the OAI ruling: keep one pick whose OAI exposure is as low as possible (decision for Eesh once the host answers).

## Addendum 11:05 (deeper read)
- **748011 (colum2131)**: datnt114 (27th) "my best pb is 0.961 with oof (5 fold) 0.937 and gold58 0.938" -> strong teams have single-family OOF ~.937 (ours ~.891 vs v4 labels; their label set unknown) and gold58 ~.938 (our fleet .922). colum2131: the 1 - p trick lets teams hide their real score; yu4u / tattaka sit at .949 "suspiciously". Expect a shake-up and hidden strong teams.
- hengck23 (same thread): "submit classwise. you should see something" = per-class public-LB probes (the H notebook already has a PROBE_CLASS switch: other 11 columns at 0.5, LB = (11 x .5 + AUC_c) / 12).
- **743416 (YYama, 91 votes, external data)**: still NO host ruling on OAI / MRNet / fastMRI / SKM-TEA. Tucker Arrants (19th) posted an NDA reply on OAI accessibility; YYama: "reasonable to consider OAI permissible". Eesh's own comment there: individual NDA approval, no institution. DeepFake precedent: prize teams removed for unanswered external-data questions (risk sits mainly with prize-eligible top 10).
- 747613: rank-48 team: ".95+ single fold single model" without external data ("nah dont even try external dataset").
- **746792 (Danial = the N / Raptor .949 author)**: recipe table. OAI masked-label rows +.005 (.940 -> .945), 384 full-field +.003, native 320 crop 0, anatomical mirror +.001. Failures: fewer windows (94 -> 24) -.006, crop 304 / 288 -.001 / -.002, single epoch vs 3-epoch SWA -.001, fresh native-crop training -.003. Hidden turnaround for the mirrored .949 entry 26.1 min (incl. queue) -> Raptor stage ~25 min.
- hengck23 745861: gold58 contains label errors (example study: report "No knee effusion" but label Effusion 1); "kaggle annotation > external annotation > negative" for mixing label sources.
