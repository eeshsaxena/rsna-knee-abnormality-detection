# WATCH_LOG: public-release watch for rsna-knee-abnormality-detection (watch.py; RESEARCH_960 lever 3)

Read-only. One section per run, newest at the bottom. Scores = the public score the notebook page shows (Kaggle
page model, read anonymously; display is truncated to 3 dp). "inputs" = every data source of the SCORED version,
tested anonymously (public dataset = GetDatasetBasics 200). RELEASE_CANDIDATE = score >= .957 and all inputs public.
DRIFT = a dataset has a newer version than the one the scored run pinned (a CLI copy would attach the newer one).

## Run 2026-10-14T07:45:12Z (10-14 13:15 IST) BASELINE

- listed 386 notebooks (score pages + recency pages + key authors), known 386, read 386 (new 386), pulled -m 107, web calls 522 (errors 0), 732 s.
- max public score among public notebooks: **0.956**; score bands (count of notebooks): {'0.956': 22, '0.955': 1, '0.954': 29, '0.953': 1}; list sort inversions vs page scores: 0.
- RELEASE_CANDIDATE (score >= 0.957 and all inputs public): **none**
- browser fallback queue (page model unreadable): 0 (browser_todo.json)

Top 15 public notebooks by page score:

| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |
|---|---|---|---|---|---|---|
| 1 | `evgendvorkin/rsna-versia-7` | 0.956 | v22 / v22 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T14:02 |
| 2 | `heliosli/rsna-knee-blend-gold-old-dataset` | 0.956 | v1 / v2 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, heliosli/rsna-knee-compact-r6cnxfsss, nartaa/rsna-knee-publication-swa-weights-20261007 | heliosli/rsna-knee-compact-r6cnxfsss | 2026-10-11T14:16 |
| 3 | `heliosli/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-11T20:01 |
| 4 | `degnonguidi/rsna-knee-abnormality-detection` | 0.956 | v10 / v10 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T08:55 |
| 5 | `jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble` | 0.956 | v1 / v1 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T06:00 |
| 6 | `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` | 0.956 | v1 / v1 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T08:39 |
| 7 | `lavinwins/rsna-gold-medal` | 0.956 | v1 / v3 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T13:59 |
| 8 | `raihanjs/rsna-knee-fork-0957-heliosli` | 0.956 | v1 / v1 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T05:55 |
| 9 | `mkhamzanov/rsna-b1-ctrl` | 0.956 | v1 / v1 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, goodpjw2008/rsna-knee-2-5d-convnext-reader, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T01:19 |
| 10 | `medvax/rsna-knee-r7-source-trial` | 0.956 | v1 / v2 | all public: heliosli/rsna-knee-compact-r7cnxfsss, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-orthofoundation-fold0-epoch7-20261004 | - | 2026-10-13T15:18 |
| 11 | `yuvrajlamba01/rsna-knee-multiscale-orthofoundation-sota-0-956` | 0.956 | v2 / v2 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T10:56 |
| 12 | `mayankraaz/rsna-knee-abnormality-pipeline` | 0.956 | v5 / v5 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T22:47 |
| 13 | `sujanmajhisuzan/rsna-knee-apex-grandmaster-v13-r7-superstack` | 0.956 | v1 / v1 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T10:45 |
| 14 | `le4ves/rsna-knee-0-957-r7-blend` | 0.956 | v1 / v1 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T14:12 |
| 15 | `jashsurendra/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T02:03 |

## Run 2026-10-14T07:59:35Z (10-14 13:29 IST)

- listed 385 notebooks (score pages + recency pages + key authors), known 387, read 111 (new 1), pulled -m 1, web calls 168 (errors 0), 162 s.
- max public score among public notebooks: **0.956**; score bands (count of notebooks): {'0.956': 24, '0.955': 1, '0.954': 29, '0.953': 1}; list sort inversions vs page scores: 1.
- RELEASE_CANDIDATE (score >= 0.957 and all inputs public): **none**
- browser fallback queue (page model unreadable): 0 (browser_todo.json)

Alerts this run:

- WATCH SCORE_APPEARED `jingshuifu/rsna-knee-flip-tta` public score 0.956 (v1)

Top 15 public notebooks by page score:

| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |
|---|---|---|---|---|---|---|
| 1 | `evgendvorkin/rsna-versia-7` | 0.956 | v22 / v22 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T14:02 |
| 2 | `heliosli/rsna-knee-blend-gold-old-dataset` | 0.956 | v1 / v2 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, heliosli/rsna-knee-compact-r6cnxfsss, nartaa/rsna-knee-publication-swa-weights-20261007 | heliosli/rsna-knee-compact-r6cnxfsss | 2026-10-11T14:16 |
| 3 | `heliosli/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-11T20:01 |
| 4 | `jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble` | 0.956 | v1 / v1 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T06:00 |
| 5 | `degnonguidi/rsna-knee-abnormality-detection` | 0.956 | v10 / v10 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T08:55 |
| 6 | `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` | 0.956 | v1 / v1 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T08:39 |
| 7 | `lavinwins/rsna-gold-medal` | 0.956 | v1 / v3 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T13:59 |
| 8 | `raihanjs/rsna-knee-fork-0957-heliosli` | 0.956 | v1 / v1 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T05:55 |
| 9 | `mkhamzanov/rsna-b1-ctrl` | 0.956 | v1 / v1 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, goodpjw2008/rsna-knee-2-5d-convnext-reader, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T01:19 |
| 10 | `mayankraaz/rsna-knee-abnormality-pipeline` | 0.956 | v5 / v5 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T22:47 |
| 11 | `medvax/rsna-knee-r7-source-trial` | 0.956 | v1 / v2 | all public: heliosli/rsna-knee-compact-r7cnxfsss, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-orthofoundation-fold0-epoch7-20261004 | - | 2026-10-13T15:18 |
| 12 | `yuvrajlamba01/rsna-knee-multiscale-orthofoundation-sota-0-956` | 0.956 | v2 / v2 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T10:56 |
| 13 | `le4ves/rsna-knee-0-957-r7-blend` | 0.956 | v1 / v1 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T14:12 |
| 14 | `sujanmajhisuzan/rsna-knee-apex-grandmaster-v13-r7-superstack` | 0.956 | v1 / v1 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T10:45 |
| 15 | `jashsurendra/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | all public: heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-knee-compact-r7cnxfsss | - | 2026-10-13T02:03 |

## Run 2026-10-14T08:03:35Z (10-14 13:33 IST)

- listed 384 notebooks (score pages + recency pages + key authors), known 387, read 109 (new 0), pulled -m 0, web calls 164 (errors 0), 157 s.
- max public score among public notebooks: **0.956**; score bands (count of notebooks): {'0.956': 24, '0.955': 1, '0.954': 29, '0.953': 1}; list sort inversions vs page scores: 2.
- RELEASE_CANDIDATE (score >= 0.957 and all inputs public): **none**
- browser fallback queue (page model unreadable): 0 (browser_todo.json)

Top 15 public notebooks by page score (inputs: R7 = heliosli/rsna-knee-compact-r7cnxfsss, OF = heliosli/rsna-orthofoundation-fold0-epoch7-20261004, NS = nartaa/rsna-knee-publication-swa-weights-20261007, R6 = heliosli/rsna-knee-compact-r6cnxfsss, CV1 = heliosli/rsna-knee-compact-v1, GP = goodpjw2008/rsna-knee-2-5d-convnext-reader):

| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |
|---|---|---|---|---|---|---|
| 1 | `evgendvorkin/rsna-versia-7` | 0.956 | v22 / v22 | public NS+OF+R7 | - | 2026-10-13T14:02 |
| 2 | `heliosli/rsna-knee-blend-gold-old-dataset` | 0.956 | v1 / v2 | public NS+OF+R6 | R6 | 2026-10-11T14:16 |
| 3 | `heliosli/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-11T20:01 |
| 4 | `jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T06:00 |
| 5 | `degnonguidi/rsna-knee-abnormality-detection` | 0.956 | v10 / v10 | public NS+OF+R7 | - | 2026-10-13T08:55 |
| 6 | `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T08:39 |
| 7 | `lavinwins/rsna-gold-medal` | 0.956 | v1 / v3 | public NS+OF+R7 | - | 2026-10-13T13:59 |
| 8 | `raihanjs/rsna-knee-fork-0957-heliosli` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T05:55 |
| 9 | `mkhamzanov/rsna-b1-ctrl` | 0.956 | v1 / v1 | public GP+NS+OF+R7 | - | 2026-10-13T01:19 |
| 10 | `mayankraaz/rsna-knee-abnormality-pipeline` | 0.956 | v5 / v5 | public NS+OF+R7 | - | 2026-10-13T22:47 |
| 11 | `medvax/rsna-knee-r7-source-trial` | 0.956 | v1 / v2 | public NS+OF+R7 | - | 2026-10-13T15:18 |
| 12 | `yuvrajlamba01/rsna-knee-multiscale-orthofoundation-sota-0-956` | 0.956 | v2 / v2 | public NS+OF+R7 | - | 2026-10-13T10:56 |
| 13 | `le4ves/rsna-knee-0-957-r7-blend` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T14:12 |
| 14 | `picond/rsna-knee-r7-pinned-exact50-control` | 0.956 | v1 / v3 | public NS+OF+R7 | - | 2026-10-14T03:47 |
| 15 | `sujanmajhisuzan/rsna-knee-apex-grandmaster-v13-r7-superstack` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T10:45 |

Pending free reads (no public score yet; ran in the last 72 h; all inputs public, R7 bundle among them; a score
appearing raises SCORE_APPEARED):

- `jingshuifu/rsna-knee-w30` v3, ran 2026-10-14T07:34 UTC, inputs NS+OF+R7
- `rudrpratapsingh/rsna-knee-four-public-models-one-rank-blend` v2, ran 2026-10-14T07:12 UTC, inputs CV1+NS+OF+R7
- `jingshuifu/rsna-knee-a224-flip` v1, ran 2026-10-14T06:57 UTC, inputs NS+OF+R7
- `jingshuifu/rsna-knee-a224-w30` v1, ran 2026-10-14T06:57 UTC, inputs NS+OF+R7
- `jingshuifu/rsna-knee-a224-blend` v1, ran 2026-10-14T06:40 UTC, inputs NS+OF+R7
- `piyush474/rsna-knee-0-956-raptor-convnext-r7-ensemble` v1, ran 2026-10-13T09:02 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-heliosli-r7cnxfsss-0-5-blend` v4, ran 2026-10-13T06:42 UTC, inputs NS+OF+R7
- `z1358028281/0-957-latest-dataset-score-in-one-click` v2, ran 2026-10-13T02:28 UTC, inputs NS+OF+R7

Watched datasets (heliosli + nartaa lists, RESEARCH_960 baseline set; knee-related only, all in watch_state.json):

| dataset | version | last updated | bytes | via |
|---|---|---|---|---|
| `dreaddevelopment/raptor-knee-widedense` | v2 | 2026-08-23T04:12 | 543320725 | GetDatasetBasics (anonymous) |
| `heliosli/rsna-knee-compact-r6cnxfsss` | v2 | 2026-10-11T13:18 | 823081993 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-r7cnxfsss` | v1 | 2026-10-11T19:59 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-v1` | v13 | 2026-10-10T19:50 | 823082549 | datasets list --user heliosli |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | v1 | 2026-10-04T07:54 | 1343091054 | datasets list --user heliosli |
| `jessky/rsna-knee-local-cv-cnx50` | v1 | 2026-10-12T01:15 | 931795323 | GetDatasetBasics (anonymous) |
| `nartaa/rsna-knee-hpo-assets` | v9 | 2026-09-25T12:01 | 1079732 | datasets list --user nartaa |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | v1 | 2026-10-07T19:26 | 543312838 | datasets list --user nartaa |
| `nartaa/rsna24-helpers` | v8 | 2024-09-04T17:37 | 519217604 | datasets list --user nartaa |
| `nartaa/rsna24-npy1` | v1 | 2024-08-09T17:47 | 10975586069 | datasets list --user nartaa |
| `nartaa/rsna24-npy2` | v1 | 2024-08-07T18:36 | 11052914944 | datasets list --user nartaa |
| `nartaa/rsna24-npy3` | v1 | 2024-08-07T18:37 | 11004266683 | datasets list --user nartaa |
| `nartaa/rsna24-npy4` | v1 | 2024-08-07T18:37 | 3379991785 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models` | v16 | 2024-09-10T08:38 | 259425853 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models-testbed-000` | v1 | 2024-09-12T09:47 | 86077478 | datasets list --user nartaa |
| `nartaa/rsna24-train-images` | v2 | 2024-09-12T07:27 | 5877188280 | datasets list --user nartaa |

## Run 2026-10-14T10:25:36Z (10-14 15:55 IST)

- listed 381 notebooks (score pages + recency pages + key authors), known 392, read 113 (new 5), pulled -m 7, web calls 174 (errors 0), 199 s.
- max public score among public notebooks: **0.956**; score bands (count of notebooks): {'0.956': 25, '0.955': 3, '0.954': 29, '0.953': 1}; list sort inversions vs page scores: 4.
- RELEASE_CANDIDATE (score >= 0.957 and all inputs public): **none**
- browser fallback queue (page model unreadable): 0 (browser_todo.json)

Alerts this run:

- WATCH KEY_AUTHOR_NEW_NOTEBOOK `jingshuifu/rsna-knee-a224-topview` score 0.956, inputs {'competition:rsna-knee-abnormality-detection': 'ours', 'dataset:heliosli/rsna-orthofoundation-fold0-epoch7-20261004': 'public', 'dataset:nartaa/rsna-knee-publication-swa-weights-20261007': 'public', 'dataset:heliosli/rsna-knee-compact-r7cnxfsss': 'public'}
- WATCH KEY_AUTHOR_NEW_NOTEBOOK `jingshuifu/rsna-knee-a224-multiview` score 0.956, inputs {'competition:rsna-knee-abnormality-detection': 'ours', 'dataset:heliosli/rsna-knee-compact-r7cnxfsss': 'public', 'dataset:nartaa/rsna-knee-publication-swa-weights-20261007': 'public', 'dataset:heliosli/rsna-orthofoundation-fold0-epoch7-20261004': 'public'}
- WATCH SCORE_APPEARED `jingshuifu/rsna-knee-a224-blend` public score 0.956 (v1)
- WATCH SCORE_APPEARED `jingshuifu/rsna-knee-w30` public score 0.955 (v1)
- WATCH SCORE_APPEARED `jingshuifu/rsna-knee-a224-w30` public score 0.955 (v1)
- WATCH KEY_AUTHOR_NEW_NOTEBOOK `jingshuifu/rsna-knee-a224-w50` score 0.954, inputs {'competition:rsna-knee-abnormality-detection': 'ours', 'dataset:heliosli/rsna-orthofoundation-fold0-epoch7-20261004': 'public', 'dataset:goodpjw2008/rsna-knee-2-5d-convnext-reader': 'public', 'dataset:nartaa/rsna-knee-publication-swa-weights-20261007': 'public', 'dataset:heliosli/rsna-knee-compact-v1': 'public'}
- WATCH KEY_AUTHOR_NEW_NOTEBOOK `jingshuifu/rsna-knee-raptor-multiview` score 0.954, inputs {'competition:rsna-knee-abnormality-detection': 'ours', 'dataset:goodpjw2008/rsna-knee-2-5d-convnext-reader': 'public', 'dataset:heliosli/rsna-knee-compact-v1': 'public', 'dataset:nartaa/rsna-knee-publication-swa-weights-20261007': 'public', 'dataset:heliosli/rsna-orthofoundation-fold0-epoch7-20261004': 'public'}
- WATCH KEY_AUTHOR_NEW_NOTEBOOK `jingshuifu/rsna-knee-raptor-topview` score 0.954, inputs {'competition:rsna-knee-abnormality-detection': 'ours', 'dataset:heliosli/rsna-knee-compact-v1': 'public', 'dataset:nartaa/rsna-knee-publication-swa-weights-20261007': 'public', 'dataset:heliosli/rsna-orthofoundation-fold0-epoch7-20261004': 'public'}
- WATCH NEW_INPUT `basith007/model-ensembling` score None uses never-seen ['model:basith007/coatnet/pytorch/default/1', 'model:basith007/dino/pytorch/default/1']
- WATCH NEW_INPUT `philgear/3d-knee-anatomy-simulator-and-waiting-game` score None uses never-seen ['dataset:philgear/pocketgull-font']

Top 15 public notebooks by page score (inputs: R7 = heliosli/rsna-knee-compact-r7cnxfsss, OF = heliosli/rsna-orthofoundation-fold0-epoch7-20261004, NS = nartaa/rsna-knee-publication-swa-weights-20261007, R6 = heliosli/rsna-knee-compact-r6cnxfsss, CV1 = heliosli/rsna-knee-compact-v1, GP = goodpjw2008/rsna-knee-2-5d-convnext-reader):

| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |
|---|---|---|---|---|---|---|
| 1 | `evgendvorkin/rsna-versia-7` | 0.956 | v22 / v22 | public NS+OF+R7 | - | 2026-10-13T14:02 |
| 2 | `heliosli/rsna-knee-blend-gold-old-dataset` | 0.956 | v1 / v2 | public NS+OF+R6 | R6 | 2026-10-11T14:16 |
| 3 | `heliosli/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-11T20:01 |
| 4 | `degnonguidi/rsna-knee-abnormality-detection` | 0.956 | v10 / v10 | public NS+OF+R7 | - | 2026-10-13T08:55 |
| 5 | `jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T06:00 |
| 6 | `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T08:39 |
| 7 | `raihanjs/rsna-knee-fork-0957-heliosli` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T05:55 |
| 8 | `lavinwins/rsna-gold-medal` | 0.956 | v1 / v3 | public NS+OF+R7 | - | 2026-10-13T13:59 |
| 9 | `mkhamzanov/rsna-b1-ctrl` | 0.956 | v1 / v1 | public GP+NS+OF+R7 | - | 2026-10-13T01:19 |
| 10 | `mayankraaz/rsna-knee-abnormality-pipeline` | 0.956 | v5 / v5 | public NS+OF+R7 | - | 2026-10-13T22:47 |
| 11 | `medvax/rsna-knee-r7-source-trial` | 0.956 | v1 / v2 | public NS+OF+R7 | - | 2026-10-13T15:18 |
| 12 | `yuvrajlamba01/rsna-knee-multiscale-orthofoundation-sota-0-956` | 0.956 | v2 / v2 | public NS+OF+R7 | - | 2026-10-13T10:56 |
| 13 | `picond/rsna-knee-r7-pinned-exact50-control` | 0.956 | v1 / v3 | public NS+OF+R7 | - | 2026-10-14T03:47 |
| 14 | `jashsurendra/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T02:03 |
| 15 | `le4ves/rsna-knee-0-957-r7-blend` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T14:12 |

Pending free reads (no public score yet; ran in the last 72 h; all inputs public, R7 bundle among them; a score
appearing raises SCORE_APPEARED):

- `rudrpratapsingh/rsna-knee-four-public-models-one-rank-blend` v2, ran 2026-10-14T07:12 UTC, inputs CV1+NS+OF+R7
- `jingshuifu/rsna-knee-a224-flip` v1, ran 2026-10-14T06:57 UTC, inputs NS+OF+R7
- `piyush474/rsna-knee-0-956-raptor-convnext-r7-ensemble` v1, ran 2026-10-13T09:02 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-heliosli-r7cnxfsss-0-5-blend` v4, ran 2026-10-13T06:42 UTC, inputs NS+OF+R7
- `z1358028281/0-957-latest-dataset-score-in-one-click` v2, ran 2026-10-13T02:28 UTC, inputs NS+OF+R7

Watched datasets (heliosli + nartaa lists, RESEARCH_960 baseline set; knee-related only, all in watch_state.json):

| dataset | version | last updated | bytes | via |
|---|---|---|---|---|
| `dreaddevelopment/raptor-knee-widedense` | v2 | 2026-08-23T04:12 | 543320725 | GetDatasetBasics (anonymous) |
| `heliosli/rsna-knee-compact-r6cnxfsss` | v2 | 2026-10-11T13:18 | 823081993 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-r7cnxfsss` | v1 | 2026-10-11T19:59 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-v1` | v13 | 2026-10-10T19:50 | 823082549 | datasets list --user heliosli |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | v1 | 2026-10-04T07:54 | 1343091054 | datasets list --user heliosli |
| `jessky/rsna-knee-local-cv-cnx50` | v1 | 2026-10-12T01:15 | 931795323 | GetDatasetBasics (anonymous) |
| `nartaa/rsna-knee-hpo-assets` | v9 | 2026-09-25T12:01 | 1079732 | datasets list --user nartaa |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | v1 | 2026-10-07T19:26 | 543312838 | datasets list --user nartaa |
| `nartaa/rsna24-helpers` | v8 | 2024-09-04T17:37 | 519217604 | datasets list --user nartaa |
| `nartaa/rsna24-npy1` | v1 | 2024-08-09T17:47 | 10975586069 | datasets list --user nartaa |
| `nartaa/rsna24-npy2` | v1 | 2024-08-07T18:36 | 11052914944 | datasets list --user nartaa |
| `nartaa/rsna24-npy3` | v1 | 2024-08-07T18:37 | 11004266683 | datasets list --user nartaa |
| `nartaa/rsna24-npy4` | v1 | 2024-08-07T18:37 | 3379991785 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models` | v16 | 2024-09-10T08:38 | 259425853 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models-testbed-000` | v1 | 2024-09-12T09:47 | 86077478 | datasets list --user nartaa |
| `nartaa/rsna24-train-images` | v2 | 2024-09-12T07:27 | 5877188280 | datasets list --user nartaa |

Errors (5): pull jingshuifu/rsna-knee-0-957-fork-v1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernel; pull jingshuifu/rsna-knee-jenil956-fork: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernel; pull jingshuifu/rsna-knee-dinosour-fork: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernel; pull jingshuifu/rsna-knee-v11-fork: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernel; pull jingshuifu/rsna-knee-versia7-fork: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernel

## Run 2026-10-14T15:58:32Z (10-14 21:28 IST)

- listed 383 notebooks (score pages + recency pages + key authors), known 405, read 120 (new 13), pulled -m 15, web calls 200 (errors 0), 240 s.
- max public score among public notebooks: **0.957**; score bands (count of notebooks): {'0.957': 1, '0.956': 29, '0.955': 4, '0.954': 28}; list sort inversions vs page scores: 11.
- RELEASE_CANDIDATE (score >= 0.957 and all inputs public): **none**
- browser fallback queue (page model unreadable): 0 (browser_todo.json)

Alerts this run:

- WATCH HIGH_SCORE_NOT_COPYABLE `bliverrigpdoge/r7-raptor-knee-centred-convnext` score 0.957 v1 (current v1), inputs {'competition:rsna-knee-abnormality-detection': 'ours', 'dataset:heliosli/rsna-orthofoundation-fold0-epoch7-20261004': 'public', 'dataset:nartaa/rsna-knee-publication-swa-weights-20261007': 'public', 'dataset:?/private-data-source': 'unresolved_owner', 'dataset:heliosli/rsna-knee-compact-r7cnxfsss': 'public'}
- WATCH SCORE_UP `goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` 0.954 -> 0.956
- WATCH SCORE_APPEARED `jingshuifu/rsna-knee-a224-flip` public score 0.956 (v1)
- WATCH KEY_AUTHOR_NEW_NOTEBOOK `varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up` score 0.956, inputs {'competition:rsna-knee-abnormality-detection': 'ours', 'dataset:heliosli/rsna-orthofoundation-fold0-epoch7-20261004': 'public', 'dataset:nartaa/rsna-knee-publication-swa-weights-20261007': 'public', 'dataset:heliosli/rsna-knee-compact-r7cnxfsss': 'public'}
- WATCH NEW_INPUT `varianceofx/rsna-knee-quad-b1` score 0.955 uses never-seen ['dataset:varianceofx/rsna-knee-compact-r6cnxfsss', 'dataset:varianceofx/rsna-knee-compact-r7cnxfsss']
- WATCH NEW_INPUT `isotriage/rsna-knee-quad-b2` score None uses never-seen ['dataset:isotriage/rsna-jev-oai-arm']

Top 15 public notebooks by page score (inputs: R7 = heliosli/rsna-knee-compact-r7cnxfsss, OF = heliosli/rsna-orthofoundation-fold0-epoch7-20261004, NS = nartaa/rsna-knee-publication-swa-weights-20261007, R6 = heliosli/rsna-knee-compact-r6cnxfsss, CV1 = heliosli/rsna-knee-compact-v1, GP = goodpjw2008/rsna-knee-2-5d-convnext-reader):

| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |
|---|---|---|---|---|---|---|
| 1 | `bliverrigpdoge/r7-raptor-knee-centred-convnext` | 0.957 | v1 / v1 | NOT all public: ?/private-data-source+NS+OF+R7 | - | 2026-10-14T08:26 |
| 2 | `evgendvorkin/rsna-versia-7` | 0.956 | v22 / v22 | public NS+OF+R7 | - | 2026-10-13T14:02 |
| 3 | `heliosli/rsna-knee-blend-gold-old-dataset` | 0.956 | v1 / v2 | public NS+OF+R6 | R6 | 2026-10-11T14:16 |
| 4 | `heliosli/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-11T20:01 |
| 5 | `goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` | 0.956 | v11 / v18 | public GP+NS+OF+R7 | - | 2026-10-14T14:50 |
| 6 | `degnonguidi/rsna-knee-abnormality-detection` | 0.956 | v10 / v10 | public NS+OF+R7 | - | 2026-10-13T08:55 |
| 7 | `jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T06:00 |
| 8 | `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T08:39 |
| 9 | `mayankraaz/rsna-knee-abnormality-pipeline` | 0.956 | v5 / v7 | public NS+OF+R7 | - | 2026-10-14T11:40 |
| 10 | `lavinwins/rsna-gold-medal` | 0.956 | v1 / v3 | public NS+OF+R7 | - | 2026-10-13T13:59 |
| 11 | `raihanjs/rsna-knee-fork-0957-heliosli` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T05:55 |
| 12 | `mkhamzanov/rsna-b1-ctrl` | 0.956 | v1 / v1 | public GP+NS+OF+R7 | - | 2026-10-13T01:19 |
| 13 | `varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-14T10:52 |
| 14 | `medvax/rsna-knee-r7-source-trial` | 0.956 | v1 / v2 | public NS+OF+R7 | - | 2026-10-13T15:18 |
| 15 | `yuvrajlamba01/rsna-knee-multiscale-orthofoundation-sota-0-956` | 0.956 | v2 / v2 | public NS+OF+R7 | - | 2026-10-13T10:56 |

Pending free reads (no public score yet; ran in the last 72 h; all inputs public, R7 bundle among them; a score
appearing raises SCORE_APPEARED):

- `yuvrajlamba01/rsna-knee-research-optimized` v1, ran 2026-10-14T15:15 UTC, inputs NS+OF+R7+bliverrigpdoge/rsna-knee-leg-r9all
- `rjings/rsna-r224-anatomical-mirror-20261014` v1, ran 2026-10-14T13:05 UTC, inputs NS+OF+R7
- `rjings/rsna-knee-four-public-models-rank-blend-20261014` v1, ran 2026-10-14T12:27 UTC, inputs CV1+NS+OF+R7
- `rudrpratapsingh/rsna-knee-four-public-models-one-rank-blend` v2, ran 2026-10-14T07:12 UTC, inputs CV1+NS+OF+R7
- `piyush474/rsna-knee-0-956-raptor-convnext-r7-ensemble` v1, ran 2026-10-13T09:02 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-heliosli-r7cnxfsss-0-5-blend` v4, ran 2026-10-13T06:42 UTC, inputs NS+OF+R7
- `z1358028281/0-957-latest-dataset-score-in-one-click` v2, ran 2026-10-13T02:28 UTC, inputs NS+OF+R7

Watched datasets (heliosli + nartaa lists, RESEARCH_960 baseline set; knee-related only, all in watch_state.json):

| dataset | version | last updated | bytes | via |
|---|---|---|---|---|
| `dreaddevelopment/raptor-knee-widedense` | v2 | 2026-08-23T04:12 | 543320725 | GetDatasetBasics (anonymous) |
| `heliosli/rsna-knee-compact-r6cnxfsss` | v2 | 2026-10-11T13:18 | 823081993 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-r7cnxfsss` | v1 | 2026-10-11T19:59 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-v1` | v13 | 2026-10-10T19:50 | 823082549 | datasets list --user heliosli |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | v1 | 2026-10-04T07:54 | 1343091054 | datasets list --user heliosli |
| `jessky/rsna-knee-local-cv-cnx50` | v1 | 2026-10-12T01:15 | 931795323 | GetDatasetBasics (anonymous) |
| `nartaa/rsna-knee-hpo-assets` | v9 | 2026-09-25T12:01 | 1079732 | datasets list --user nartaa |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | v1 | 2026-10-07T19:26 | 543312838 | datasets list --user nartaa |
| `nartaa/rsna24-helpers` | v8 | 2024-09-04T17:37 | 519217604 | datasets list --user nartaa |
| `nartaa/rsna24-npy1` | v1 | 2024-08-09T17:47 | 10975586069 | datasets list --user nartaa |
| `nartaa/rsna24-npy2` | v1 | 2024-08-07T18:36 | 11052914944 | datasets list --user nartaa |
| `nartaa/rsna24-npy3` | v1 | 2024-08-07T18:37 | 11004266683 | datasets list --user nartaa |
| `nartaa/rsna24-npy4` | v1 | 2024-08-07T18:37 | 3379991785 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models` | v16 | 2024-09-10T08:38 | 259425853 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models-testbed-000` | v1 | 2024-09-12T09:47 | 86077478 | datasets list --user nartaa |
| `nartaa/rsna24-train-images` | v2 | 2024-09-12T07:27 | 5877188280 | datasets list --user nartaa |

## Run 2026-10-14T16:03:51Z (10-14 21:33 IST)

- listed 383 notebooks (score pages + recency pages + key authors), known 405, read 111 (new 0), pulled -m 0, web calls 178 (errors 0), 178 s.
- max public score among public notebooks: **0.957**; score bands (count of notebooks): {'0.957': 1, '0.956': 29, '0.955': 4, '0.954': 28}; list sort inversions vs page scores: 11.
- RELEASE_CANDIDATE (score >= 0.957 and all inputs public): **none**
- browser fallback queue (page model unreadable): 0 (browser_todo.json)

Alerts this run:

- WATCH SCORE_APPEARED `yatinpatil/rsa-iteration-2` public score 0.602 (v3)

Top 15 public notebooks by page score (inputs: R7 = heliosli/rsna-knee-compact-r7cnxfsss, OF = heliosli/rsna-orthofoundation-fold0-epoch7-20261004, NS = nartaa/rsna-knee-publication-swa-weights-20261007, R6 = heliosli/rsna-knee-compact-r6cnxfsss, CV1 = heliosli/rsna-knee-compact-v1, GP = goodpjw2008/rsna-knee-2-5d-convnext-reader):

| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |
|---|---|---|---|---|---|---|
| 1 | `bliverrigpdoge/r7-raptor-knee-centred-convnext` | 0.957 | v1 / v1 | NOT all public: ?/private-data-source+NS+OF+R7 | - | 2026-10-14T08:26 |
| 2 | `evgendvorkin/rsna-versia-7` | 0.956 | v22 / v22 | public NS+OF+R7 | - | 2026-10-13T14:02 |
| 3 | `heliosli/rsna-knee-blend-gold-old-dataset` | 0.956 | v1 / v2 | public NS+OF+R6 | R6 | 2026-10-11T14:16 |
| 4 | `heliosli/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-11T20:01 |
| 5 | `goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` | 0.956 | v11 / v18 | public GP+NS+OF+R7 | - | 2026-10-14T14:50 |
| 6 | `degnonguidi/rsna-knee-abnormality-detection` | 0.956 | v10 / v10 | public NS+OF+R7 | - | 2026-10-13T08:55 |
| 7 | `jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T06:00 |
| 8 | `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T08:39 |
| 9 | `mayankraaz/rsna-knee-abnormality-pipeline` | 0.956 | v5 / v7 | public NS+OF+R7 | - | 2026-10-14T11:40 |
| 10 | `lavinwins/rsna-gold-medal` | 0.956 | v1 / v3 | public NS+OF+R7 | - | 2026-10-13T13:59 |
| 11 | `raihanjs/rsna-knee-fork-0957-heliosli` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T05:55 |
| 12 | `mkhamzanov/rsna-b1-ctrl` | 0.956 | v1 / v1 | public GP+NS+OF+R7 | - | 2026-10-13T01:19 |
| 13 | `varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-14T10:52 |
| 14 | `medvax/rsna-knee-r7-source-trial` | 0.956 | v1 / v2 | public NS+OF+R7 | - | 2026-10-13T15:18 |
| 15 | `yuvrajlamba01/rsna-knee-multiscale-orthofoundation-sota-0-956` | 0.956 | v2 / v2 | public NS+OF+R7 | - | 2026-10-13T10:56 |

Pending free reads (no public score yet; ran in the last 72 h; all inputs public, R7 bundle among them; a score
appearing raises SCORE_APPEARED):

- `yuvrajlamba01/rsna-knee-research-optimized` v1, ran 2026-10-14T15:15 UTC, inputs NS+OF+R7+bliverrigpdoge/rsna-knee-leg-r9all
- `rjings/rsna-r224-anatomical-mirror-20261014` v1, ran 2026-10-14T13:05 UTC, inputs NS+OF+R7
- `rjings/rsna-knee-four-public-models-rank-blend-20261014` v1, ran 2026-10-14T12:27 UTC, inputs CV1+NS+OF+R7
- `rudrpratapsingh/rsna-knee-four-public-models-one-rank-blend` v2, ran 2026-10-14T07:12 UTC, inputs CV1+NS+OF+R7
- `piyush474/rsna-knee-0-956-raptor-convnext-r7-ensemble` v1, ran 2026-10-13T09:02 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-heliosli-r7cnxfsss-0-5-blend` v4, ran 2026-10-13T06:42 UTC, inputs NS+OF+R7
- `z1358028281/0-957-latest-dataset-score-in-one-click` v2, ran 2026-10-13T02:28 UTC, inputs NS+OF+R7

Watched datasets (heliosli + nartaa lists, RESEARCH_960 baseline set; knee-related only, all in watch_state.json):

| dataset | version | last updated | bytes | via |
|---|---|---|---|---|
| `dreaddevelopment/raptor-knee-widedense` | v2 | 2026-08-23T04:12 | 543320725 | GetDatasetBasics (anonymous) |
| `heliosli/rsna-knee-compact-r6cnxfsss` | v2 | 2026-10-11T13:18 | 823081993 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-r7cnxfsss` | v1 | 2026-10-11T19:59 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-v1` | v13 | 2026-10-10T19:50 | 823082549 | datasets list --user heliosli |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | v1 | 2026-10-04T07:54 | 1343091054 | datasets list --user heliosli |
| `jessky/rsna-knee-local-cv-cnx50` | v1 | 2026-10-12T01:15 | 931795323 | GetDatasetBasics (anonymous) |
| `nartaa/rsna-knee-hpo-assets` | v9 | 2026-09-25T12:01 | 1079732 | datasets list --user nartaa |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | v1 | 2026-10-07T19:26 | 543312838 | datasets list --user nartaa |
| `nartaa/rsna24-helpers` | v8 | 2024-09-04T17:37 | 519217604 | datasets list --user nartaa |
| `nartaa/rsna24-npy1` | v1 | 2024-08-09T17:47 | 10975586069 | datasets list --user nartaa |
| `nartaa/rsna24-npy2` | v1 | 2024-08-07T18:36 | 11052914944 | datasets list --user nartaa |
| `nartaa/rsna24-npy3` | v1 | 2024-08-07T18:37 | 11004266683 | datasets list --user nartaa |
| `nartaa/rsna24-npy4` | v1 | 2024-08-07T18:37 | 3379991785 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models` | v16 | 2024-09-10T08:38 | 259425853 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models-testbed-000` | v1 | 2024-09-12T09:47 | 86077478 | datasets list --user nartaa |
| `nartaa/rsna24-train-images` | v2 | 2024-09-12T07:27 | 5877188280 | datasets list --user nartaa |

## Run 2026-10-14T17:36:49Z (10-14 23:06 IST)

- listed 384 notebooks (score pages + recency pages + key authors), known 407, read 112 (new 2), pulled -m 2, web calls 187 (errors 0), 209 s.
- max public score among public notebooks: **0.957**; score bands (count of notebooks): {'0.957': 1, '0.956': 29, '0.955': 4, '0.954': 28}; list sort inversions vs page scores: 11.
- RELEASE_CANDIDATE (score >= 0.957 and all inputs public): **none**
- browser fallback queue (page model unreadable): 0 (browser_todo.json)

Alerts this run:

- WATCH SCORE_APPEARED `rjings/rsna-r224-anatomical-mirror-20261014` public score 0.956 (v1)

Top 15 public notebooks by page score (inputs: R7 = heliosli/rsna-knee-compact-r7cnxfsss, OF = heliosli/rsna-orthofoundation-fold0-epoch7-20261004, NS = nartaa/rsna-knee-publication-swa-weights-20261007, R6 = heliosli/rsna-knee-compact-r6cnxfsss, CV1 = heliosli/rsna-knee-compact-v1, GP = goodpjw2008/rsna-knee-2-5d-convnext-reader):

| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |
|---|---|---|---|---|---|---|
| 1 | `bliverrigpdoge/r7-raptor-knee-centred-convnext` | 0.957 | v1 / v1 | NOT all public: ?/private-data-source+NS+OF+R7 | - | 2026-10-14T08:26 |
| 2 | `evgendvorkin/rsna-versia-7` | 0.956 | v22 / v22 | public NS+OF+R7 | - | 2026-10-13T14:02 |
| 3 | `heliosli/rsna-knee-blend-gold-old-dataset` | 0.956 | v1 / v2 | public NS+OF+R6 | R6 | 2026-10-11T14:16 |
| 4 | `heliosli/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-11T20:01 |
| 5 | `goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` | 0.956 | v11 / v18 | public GP+NS+OF+R7 | - | 2026-10-14T14:50 |
| 6 | `degnonguidi/rsna-knee-abnormality-detection` | 0.956 | v10 / v10 | public NS+OF+R7 | - | 2026-10-13T08:55 |
| 7 | `jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T06:00 |
| 8 | `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T08:39 |
| 9 | `mayankraaz/rsna-knee-abnormality-pipeline` | 0.956 | v5 / v7 | public NS+OF+R7 | - | 2026-10-14T11:40 |
| 10 | `lavinwins/rsna-gold-medal` | 0.956 | v1 / v3 | public NS+OF+R7 | - | 2026-10-13T13:59 |
| 11 | `mkhamzanov/rsna-b1-ctrl` | 0.956 | v1 / v1 | public GP+NS+OF+R7 | - | 2026-10-13T01:19 |
| 12 | `varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-14T10:52 |
| 13 | `medvax/rsna-knee-r7-source-trial` | 0.956 | v1 / v2 | public NS+OF+R7 | - | 2026-10-13T15:18 |
| 14 | `yuvrajlamba01/rsna-knee-multiscale-orthofoundation-sota-0-956` | 0.956 | v2 / v2 | public NS+OF+R7 | - | 2026-10-13T10:56 |
| 15 | `jashsurendra/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T02:03 |

Pending free reads (no public score yet; ran in the last 72 h; all inputs public, R7 bundle among them; a score
appearing raises SCORE_APPEARED):

- `sushanthtiruvaipati/rsna-knee-r7-raptor384-raptor224-blend` v1, ran 2026-10-14T16:20 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-r7-raptor384-raptor224-mirror-blend` v3, ran 2026-10-14T16:18 UTC, inputs NS+OF+R7
- `yuvrajlamba01/rsna-knee-research-optimized` v1, ran 2026-10-14T15:15 UTC, inputs NS+OF+R7+bliverrigpdoge/rsna-knee-leg-r9all
- `rjings/rsna-knee-four-public-models-rank-blend-20261014` v1, ran 2026-10-14T12:27 UTC, inputs CV1+NS+OF+R7
- `rudrpratapsingh/rsna-knee-four-public-models-one-rank-blend` v2, ran 2026-10-14T07:12 UTC, inputs CV1+NS+OF+R7
- `piyush474/rsna-knee-0-956-raptor-convnext-r7-ensemble` v1, ran 2026-10-13T09:02 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-heliosli-r7cnxfsss-0-5-blend` v4, ran 2026-10-13T06:42 UTC, inputs NS+OF+R7
- `z1358028281/0-957-latest-dataset-score-in-one-click` v2, ran 2026-10-13T02:28 UTC, inputs NS+OF+R7

Watched datasets (heliosli + nartaa lists, RESEARCH_960 baseline set; knee-related only, all in watch_state.json):

| dataset | version | last updated | bytes | via |
|---|---|---|---|---|
| `dreaddevelopment/raptor-knee-widedense` | v2 | 2026-08-23T04:12 | 543320725 | GetDatasetBasics (anonymous) |
| `heliosli/rsna-knee-compact-r6cnxfsss` | v2 | 2026-10-11T13:18 | 823081993 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-r7cnxfsss` | v1 | 2026-10-11T19:59 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-v1` | v13 | 2026-10-10T19:50 | 823082549 | datasets list --user heliosli |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | v1 | 2026-10-04T07:54 | 1343091054 | datasets list --user heliosli |
| `jessky/rsna-knee-local-cv-cnx50` | v1 | 2026-10-12T01:15 | 931795323 | GetDatasetBasics (anonymous) |
| `nartaa/rsna-knee-hpo-assets` | v9 | 2026-09-25T12:01 | 1079732 | datasets list --user nartaa |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | v1 | 2026-10-07T19:26 | 543312838 | datasets list --user nartaa |
| `nartaa/rsna24-helpers` | v8 | 2024-09-04T17:37 | 519217604 | datasets list --user nartaa |
| `nartaa/rsna24-npy1` | v1 | 2024-08-09T17:47 | 10975586069 | datasets list --user nartaa |
| `nartaa/rsna24-npy2` | v1 | 2024-08-07T18:36 | 11052914944 | datasets list --user nartaa |
| `nartaa/rsna24-npy3` | v1 | 2024-08-07T18:37 | 11004266683 | datasets list --user nartaa |
| `nartaa/rsna24-npy4` | v1 | 2024-08-07T18:37 | 3379991785 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models` | v16 | 2024-09-10T08:38 | 259425853 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models-testbed-000` | v1 | 2024-09-12T09:47 | 86077478 | datasets list --user nartaa |
| `nartaa/rsna24-train-images` | v2 | 2024-09-12T07:27 | 5877188280 | datasets list --user nartaa |

Errors (1): kvm raihanjs/rsna-knee-fork-0957-heliosli http 404

## Run 2026-10-14T19:10:18Z (10-15 00:40 IST)

- listed 382 notebooks (score pages + recency pages + key authors), known 413, read 116 (new 6), pulled -m 5, web calls 188 (errors 0), 216 s.
- max public score among public notebooks: **0.957**; score bands (count of notebooks): {'0.957': 2, '0.956': 31, '0.955': 4, '0.954': 28}; list sort inversions vs page scores: 15.
- RELEASE_CANDIDATE (score >= 0.957 and all inputs public): **heliosli/0-9573-latest-public-dataset-score-in-one-click**
- browser fallback queue (page model unreadable): 0 (browser_todo.json)

Alerts this run:

- FLAG RELEASE_CANDIDATE `heliosli/0-9573-latest-public-dataset-score-in-one-click` score 0.957 v1 (current v2), inputs ALL PUBLIC, DRIFT {"heliosli/rsna-knee-r6-frozen-fe055406": {"pinned_version_id": 20611930, "current_version_id": 20731557, "current_version": 2}}; next: python session_2026-10-15/watch/swap_builder.py heliosli/0-9573-latest-public-dataset-score-in-one-click/1 --name rsna-sub-rel-heliosli-0-9573-latest-public
- WATCH NEW_INPUT `heliosli/0-9573-latest-public-dataset-score-in-one-click` score 0.957 uses never-seen ['dataset:heliosli/rsna-knee-r6-frozen-fe055406']
- WATCH KEY_AUTHOR_NEW_NOTEBOOK `heliosli/0-9573-latest-public-dataset-score-in-one-click` score 0.957, inputs {'competition:rsna-knee-abnormality-detection': 'ours', 'dataset:heliosli/rsna-orthofoundation-fold0-epoch7-20261004': 'public', 'dataset:heliosli/rsna-knee-r6-frozen-fe055406': 'public', 'dataset:nartaa/rsna-knee-publication-swa-weights-20261007': 'public'}
- WATCH SCORE_APPEARED `sushanthtiruvaipati/rsna-knee-r7-raptor384-raptor224-mirror-blend` public score 0.956 (v2)
- WATCH SCORE_APPEARED `rjings/rsna-knee-four-public-models-rank-blend-20261014` public score 0.956 (v1)
- WATCH KEY_AUTHOR_NEW_NOTEBOOK `nartaa/rsna-knee-s6fall-solo` score 0.95, inputs {'competition:rsna-knee-abnormality-detection': 'ours', 'dataset:?/private-data-source': 'unresolved_owner'}
- WATCH SCORE_APPEARED `vishnusureshpbvr/vsp-agents-rsna-knee-abnormality-detection-engine` public score 0.515 (v4)
- WATCH NEW_DATASET `heliosli/rsna-knee-r6-frozen-fe055406` v2 823095598 bytes "RSNA Knee R6 Frozen fe055406"
- WATCH NEW_DATASET `nartaa/rsna-knee-s6fall-weights` v3 185910765 bytes "RSNA Knee S6 ConvNeXt all-data weights 0950"

Top 15 public notebooks by page score (inputs: R7 = heliosli/rsna-knee-compact-r7cnxfsss, OF = heliosli/rsna-orthofoundation-fold0-epoch7-20261004, NS = nartaa/rsna-knee-publication-swa-weights-20261007, R6 = heliosli/rsna-knee-compact-r6cnxfsss, CV1 = heliosli/rsna-knee-compact-v1, GP = goodpjw2008/rsna-knee-2-5d-convnext-reader):

| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |
|---|---|---|---|---|---|---|
| 1 | `bliverrigpdoge/r7-raptor-knee-centred-convnext` | 0.957 | v1 / v1 | NOT all public: ?/private-data-source+NS+OF+R7 | - | 2026-10-14T08:26 |
| 2 | `heliosli/0-9573-latest-public-dataset-score-in-one-click` | 0.957 | v1 / v2 | public NS+OF+heliosli/rsna-knee-r6-frozen-fe055406 | heliosli/rsna-knee-r6-frozen-fe055406 | 2026-10-14T18:13 |
| 3 | `evgendvorkin/rsna-versia-7` | 0.956 | v22 / v22 | public NS+OF+R7 | - | 2026-10-13T14:02 |
| 4 | `heliosli/rsna-knee-blend-gold-old-dataset` | 0.956 | v1 / v2 | public NS+OF+R6 | R6 | 2026-10-11T14:16 |
| 5 | `heliosli/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-11T20:01 |
| 6 | `goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` | 0.956 | v17 / v18 | public GP+NS+OF+R7 | - | 2026-10-14T14:50 |
| 7 | `degnonguidi/rsna-knee-abnormality-detection` | 0.956 | v10 / v10 | public NS+OF+R7 | - | 2026-10-13T08:55 |
| 8 | `jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T06:00 |
| 9 | `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T08:39 |
| 10 | `mayankraaz/rsna-knee-abnormality-pipeline` | 0.956 | v5 / v7 | public NS+OF+R7 | - | 2026-10-14T11:40 |
| 11 | `lavinwins/rsna-gold-medal` | 0.956 | v1 / v3 | public NS+OF+R7 | - | 2026-10-13T13:59 |
| 12 | `mkhamzanov/rsna-b1-ctrl` | 0.956 | v1 / v1 | public GP+NS+OF+R7 | - | 2026-10-13T01:19 |
| 13 | `varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-14T10:52 |
| 14 | `medvax/rsna-knee-r7-source-trial` | 0.956 | v1 / v2 | public NS+OF+R7 | - | 2026-10-13T15:18 |
| 15 | `yuvrajlamba01/rsna-knee-multiscale-orthofoundation-sota-0-956` | 0.956 | v2 / v2 | public NS+OF+R7 | - | 2026-10-13T10:56 |

Pending free reads (no public score yet; ran in the last 72 h; all inputs public, R7 bundle among them; a score
appearing raises SCORE_APPEARED):

- `ryugohanai/rsna-tta224-reader-20261015` v1, ran 2026-10-14T17:58 UTC, inputs GP+NS+OF+R7
- `eeshanvaghjiani2004/knee-oct15-224m` v1, ran 2026-10-14T17:50 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-r7-raptor384-raptor224-blend` v1, ran 2026-10-14T16:20 UTC, inputs NS+OF+R7
- `yuvrajlamba01/rsna-knee-research-optimized` v1, ran 2026-10-14T15:15 UTC, inputs NS+OF+R7+bliverrigpdoge/rsna-knee-leg-r9all
- `rudrpratapsingh/rsna-knee-four-public-models-one-rank-blend` v2, ran 2026-10-14T07:12 UTC, inputs CV1+NS+OF+R7
- `piyush474/rsna-knee-0-956-raptor-convnext-r7-ensemble` v1, ran 2026-10-13T09:02 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-heliosli-r7cnxfsss-0-5-blend` v4, ran 2026-10-13T06:42 UTC, inputs NS+OF+R7
- `z1358028281/0-957-latest-dataset-score-in-one-click` v2, ran 2026-10-13T02:28 UTC, inputs NS+OF+R7

Watched datasets (heliosli + nartaa lists, RESEARCH_960 baseline set; knee-related only, all in watch_state.json):

| dataset | version | last updated | bytes | via |
|---|---|---|---|---|
| `dreaddevelopment/raptor-knee-widedense` | v2 | 2026-08-23T04:12 | 543320725 | GetDatasetBasics (anonymous) |
| `heliosli/rsna-knee-compact-r6cnxfsss` | v2 | 2026-10-11T13:18 | 823081993 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-r7cnxfsss` | v1 | 2026-10-11T19:59 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-v1` | v13 | 2026-10-10T19:50 | 823082549 | datasets list --user heliosli |
| `heliosli/rsna-knee-r6-frozen-fe055406` | v2 | 2026-10-14T18:12 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | v1 | 2026-10-04T07:54 | 1343091054 | datasets list --user heliosli |
| `jessky/rsna-knee-local-cv-cnx50` | v1 | 2026-10-12T01:15 | 931795323 | GetDatasetBasics (anonymous) |
| `nartaa/rsna-knee-hpo-assets` | v9 | 2026-09-25T12:01 | 1079732 | datasets list --user nartaa |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | v1 | 2026-10-07T19:26 | 543312838 | datasets list --user nartaa |
| `nartaa/rsna-knee-s6fall-weights` | v3 | 2026-10-14T19:08 | 185910765 | datasets list --user nartaa |
| `nartaa/rsna24-helpers` | v8 | 2024-09-04T17:37 | 519217604 | datasets list --user nartaa |
| `nartaa/rsna24-npy1` | v1 | 2024-08-09T17:47 | 10975586069 | datasets list --user nartaa |
| `nartaa/rsna24-npy2` | v1 | 2024-08-07T18:36 | 11052914944 | datasets list --user nartaa |
| `nartaa/rsna24-npy3` | v1 | 2024-08-07T18:37 | 11004266683 | datasets list --user nartaa |
| `nartaa/rsna24-npy4` | v1 | 2024-08-07T18:37 | 3379991785 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models` | v16 | 2024-09-10T08:38 | 259425853 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models-testbed-000` | v1 | 2024-09-12T09:47 | 86077478 | datasets list --user nartaa |
| `nartaa/rsna24-train-images` | v2 | 2024-09-12T07:27 | 5877188280 | datasets list --user nartaa |

Errors (1): kvm raihanjs/rsna-knee-fork-0957-heliosli http 404

## Run 2026-10-14T20:43:55Z (10-15 02:13 IST)

- listed 382 notebooks (score pages + recency pages + key authors), known 415, read 112 (new 2), pulled -m 4, web calls 186 (errors 0), 197 s.
- max public score among public notebooks: **0.957**; score bands (count of notebooks): {'0.957': 2, '0.956': 32, '0.955': 4, '0.954': 28}; list sort inversions vs page scores: 16.
- RELEASE_CANDIDATE (score >= 0.957 and all inputs public): **heliosli/0-9573-latest-public-dataset-score-in-one-click**
- browser fallback queue (page model unreadable): 0 (browser_todo.json)

Alerts this run:

- WATCH SCORE_APPEARED `yuvrajlamba01/rsna-knee-research-optimized` public score 0.956 (v1)
- WATCH NEW_INPUT `mizeroluckygall/knee-mri-raptor-384-224-r7-blend` score None uses never-seen ['dataset:mizeroluckygall/rsna-knee-r6-frozen-fe055406']

Top 15 public notebooks by page score (inputs: R7 = heliosli/rsna-knee-compact-r7cnxfsss, OF = heliosli/rsna-orthofoundation-fold0-epoch7-20261004, NS = nartaa/rsna-knee-publication-swa-weights-20261007, R6 = heliosli/rsna-knee-compact-r6cnxfsss, CV1 = heliosli/rsna-knee-compact-v1, GP = goodpjw2008/rsna-knee-2-5d-convnext-reader):

| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |
|---|---|---|---|---|---|---|
| 1 | `bliverrigpdoge/r7-raptor-knee-centred-convnext` | 0.957 | v1 / v1 | NOT all public: ?/private-data-source+NS+OF+R7 | - | 2026-10-14T08:26 |
| 2 | `heliosli/0-9573-latest-public-dataset-score-in-one-click` | 0.957 | v1 / v2 | public NS+OF+heliosli/rsna-knee-r6-frozen-fe055406 | heliosli/rsna-knee-r6-frozen-fe055406 | 2026-10-14T18:13 |
| 3 | `evgendvorkin/rsna-versia-7` | 0.956 | v22 / v22 | public NS+OF+R7 | - | 2026-10-13T14:02 |
| 4 | `heliosli/rsna-knee-blend-gold-old-dataset` | 0.956 | v1 / v2 | public NS+OF+R6 | R6 | 2026-10-11T14:16 |
| 5 | `heliosli/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-11T20:01 |
| 6 | `goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` | 0.956 | v17 / v18 | public GP+NS+OF+R7 | - | 2026-10-14T14:50 |
| 7 | `degnonguidi/rsna-knee-abnormality-detection` | 0.956 | v10 / v10 | public NS+OF+R7 | - | 2026-10-13T08:55 |
| 8 | `jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T06:00 |
| 9 | `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T08:39 |
| 10 | `mayankraaz/rsna-knee-abnormality-pipeline` | 0.956 | v5 / v7 | public NS+OF+R7 | - | 2026-10-14T11:40 |
| 11 | `lavinwins/rsna-gold-medal` | 0.956 | v1 / v3 | public NS+OF+R7 | - | 2026-10-13T13:59 |
| 12 | `mkhamzanov/rsna-b1-ctrl` | 0.956 | v1 / v1 | public GP+NS+OF+R7 | - | 2026-10-13T01:19 |
| 13 | `varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-14T10:52 |
| 14 | `medvax/rsna-knee-r7-source-trial` | 0.956 | v1 / v2 | public NS+OF+R7 | - | 2026-10-13T15:18 |
| 15 | `yuvrajlamba01/rsna-knee-multiscale-orthofoundation-sota-0-956` | 0.956 | v2 / v2 | public NS+OF+R7 | - | 2026-10-13T10:56 |

Pending free reads (no public score yet; ran in the last 72 h; all inputs public, R7 bundle among them; a score
appearing raises SCORE_APPEARED):

- `ryugohanai/rsna-tta224-reader-20261015` v1, ran 2026-10-14T17:58 UTC, inputs GP+NS+OF+R7
- `eeshanvaghjiani2004/knee-oct15-224m` v1, ran 2026-10-14T17:50 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-r7-raptor384-raptor224-blend` v1, ran 2026-10-14T16:20 UTC, inputs NS+OF+R7
- `rudrpratapsingh/rsna-knee-four-public-models-one-rank-blend` v2, ran 2026-10-14T07:12 UTC, inputs CV1+NS+OF+R7
- `piyush474/rsna-knee-0-956-raptor-convnext-r7-ensemble` v1, ran 2026-10-13T09:02 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-heliosli-r7cnxfsss-0-5-blend` v4, ran 2026-10-13T06:42 UTC, inputs NS+OF+R7
- `z1358028281/0-957-latest-dataset-score-in-one-click` v2, ran 2026-10-13T02:28 UTC, inputs NS+OF+R7

Watched datasets (heliosli + nartaa lists, RESEARCH_960 baseline set; knee-related only, all in watch_state.json):

| dataset | version | last updated | bytes | via |
|---|---|---|---|---|
| `dreaddevelopment/raptor-knee-widedense` | v2 | 2026-08-23T04:12 | 543320725 | GetDatasetBasics (anonymous) |
| `heliosli/rsna-knee-compact-r6cnxfsss` | v2 | 2026-10-11T13:18 | 823081993 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-r7cnxfsss` | v1 | 2026-10-11T19:59 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-v1` | v13 | 2026-10-10T19:50 | 823082549 | datasets list --user heliosli |
| `heliosli/rsna-knee-r6-frozen-fe055406` | v2 | 2026-10-14T18:12 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | v1 | 2026-10-04T07:54 | 1343091054 | datasets list --user heliosli |
| `jessky/rsna-knee-local-cv-cnx50` | v1 | 2026-10-12T01:15 | 931795323 | GetDatasetBasics (anonymous) |
| `nartaa/rsna-knee-hpo-assets` | v9 | 2026-09-25T12:01 | 1079732 | datasets list --user nartaa |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | v1 | 2026-10-07T19:26 | 543312838 | datasets list --user nartaa |
| `nartaa/rsna-knee-s6fall-weights` | v3 | 2026-10-14T19:08 | 185910765 | datasets list --user nartaa |
| `nartaa/rsna24-helpers` | v8 | 2024-09-04T17:37 | 519217604 | datasets list --user nartaa |
| `nartaa/rsna24-npy1` | v1 | 2024-08-09T17:47 | 10975586069 | datasets list --user nartaa |
| `nartaa/rsna24-npy2` | v1 | 2024-08-07T18:36 | 11052914944 | datasets list --user nartaa |
| `nartaa/rsna24-npy3` | v1 | 2024-08-07T18:37 | 11004266683 | datasets list --user nartaa |
| `nartaa/rsna24-npy4` | v1 | 2024-08-07T18:37 | 3379991785 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models` | v16 | 2024-09-10T08:38 | 259425853 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models-testbed-000` | v1 | 2024-09-12T09:47 | 86077478 | datasets list --user nartaa |
| `nartaa/rsna24-train-images` | v2 | 2024-09-12T07:27 | 5877188280 | datasets list --user nartaa |

Errors (1): kvm raihanjs/rsna-knee-fork-0957-heliosli http 404

## Run 2026-10-14T22:17:12Z (10-15 03:47 IST)

- listed 382 notebooks (score pages + recency pages + key authors), known 415, read 110 (new 0), pulled -m 1, web calls 183 (errors 0), 192 s.
- max public score among public notebooks: **0.957**; score bands (count of notebooks): {'0.957': 2, '0.956': 32, '0.955': 4, '0.954': 28}; list sort inversions vs page scores: 16.
- RELEASE_CANDIDATE (score >= 0.957 and all inputs public): **heliosli/0-9573-latest-public-dataset-score-in-one-click**
- browser fallback queue (page model unreadable): 0 (browser_todo.json)

Top 15 public notebooks by page score (inputs: R7 = heliosli/rsna-knee-compact-r7cnxfsss, OF = heliosli/rsna-orthofoundation-fold0-epoch7-20261004, NS = nartaa/rsna-knee-publication-swa-weights-20261007, R6 = heliosli/rsna-knee-compact-r6cnxfsss, CV1 = heliosli/rsna-knee-compact-v1, GP = goodpjw2008/rsna-knee-2-5d-convnext-reader):

| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |
|---|---|---|---|---|---|---|
| 1 | `bliverrigpdoge/r7-raptor-knee-centred-convnext` | 0.957 | v1 / v1 | NOT all public: ?/private-data-source+NS+OF+R7 | - | 2026-10-14T08:26 |
| 2 | `heliosli/0-9573-latest-public-dataset-score-in-one-click` | 0.957 | v1 / v2 | public NS+OF+heliosli/rsna-knee-r6-frozen-fe055406 | heliosli/rsna-knee-r6-frozen-fe055406 | 2026-10-14T18:13 |
| 3 | `evgendvorkin/rsna-versia-7` | 0.956 | v22 / v22 | public NS+OF+R7 | - | 2026-10-13T14:02 |
| 4 | `heliosli/rsna-knee-blend-gold-old-dataset` | 0.956 | v1 / v2 | public NS+OF+R6 | R6 | 2026-10-11T14:16 |
| 5 | `heliosli/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-11T20:01 |
| 6 | `goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` | 0.956 | v17 / v18 | public GP+NS+OF+R7 | - | 2026-10-14T14:50 |
| 7 | `degnonguidi/rsna-knee-abnormality-detection` | 0.956 | v10 / v10 | public NS+OF+R7 | - | 2026-10-13T08:55 |
| 8 | `jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T06:00 |
| 9 | `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T08:39 |
| 10 | `mayankraaz/rsna-knee-abnormality-pipeline` | 0.956 | v5 / v7 | public NS+OF+R7 | - | 2026-10-14T11:40 |
| 11 | `lavinwins/rsna-gold-medal` | 0.956 | v1 / v3 | public NS+OF+R7 | - | 2026-10-13T13:59 |
| 12 | `mkhamzanov/rsna-b1-ctrl` | 0.956 | v1 / v1 | public GP+NS+OF+R7 | - | 2026-10-13T01:19 |
| 13 | `varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-14T10:52 |
| 14 | `medvax/rsna-knee-r7-source-trial` | 0.956 | v1 / v2 | public NS+OF+R7 | - | 2026-10-13T15:18 |
| 15 | `yuvrajlamba01/rsna-knee-multiscale-orthofoundation-sota-0-956` | 0.956 | v2 / v2 | public NS+OF+R7 | - | 2026-10-13T10:56 |

Pending free reads (no public score yet; ran in the last 72 h; all inputs public, R7 bundle among them; a score
appearing raises SCORE_APPEARED):

- `ryugohanai/rsna-tta224-reader-20261015` v1, ran 2026-10-14T17:58 UTC, inputs GP+NS+OF+R7
- `eeshanvaghjiani2004/knee-oct15-224m` v1, ran 2026-10-14T17:50 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-r7-raptor384-raptor224-blend` v1, ran 2026-10-14T16:20 UTC, inputs NS+OF+R7
- `rudrpratapsingh/rsna-knee-four-public-models-one-rank-blend` v2, ran 2026-10-14T07:12 UTC, inputs CV1+NS+OF+R7
- `piyush474/rsna-knee-0-956-raptor-convnext-r7-ensemble` v1, ran 2026-10-13T09:02 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-heliosli-r7cnxfsss-0-5-blend` v4, ran 2026-10-13T06:42 UTC, inputs NS+OF+R7
- `z1358028281/0-957-latest-dataset-score-in-one-click` v2, ran 2026-10-13T02:28 UTC, inputs NS+OF+R7

Watched datasets (heliosli + nartaa lists, RESEARCH_960 baseline set; knee-related only, all in watch_state.json):

| dataset | version | last updated | bytes | via |
|---|---|---|---|---|
| `dreaddevelopment/raptor-knee-widedense` | v2 | 2026-08-23T04:12 | 543320725 | GetDatasetBasics (anonymous) |
| `heliosli/rsna-knee-compact-r6cnxfsss` | v2 | 2026-10-11T13:18 | 823081993 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-r7cnxfsss` | v1 | 2026-10-11T19:59 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-v1` | v13 | 2026-10-10T19:50 | 823082549 | datasets list --user heliosli |
| `heliosli/rsna-knee-r6-frozen-fe055406` | v2 | 2026-10-14T18:12 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | v1 | 2026-10-04T07:54 | 1343091054 | datasets list --user heliosli |
| `jessky/rsna-knee-local-cv-cnx50` | v1 | 2026-10-12T01:15 | 931795323 | GetDatasetBasics (anonymous) |
| `nartaa/rsna-knee-hpo-assets` | v9 | 2026-09-25T12:01 | 1079732 | datasets list --user nartaa |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | v1 | 2026-10-07T19:26 | 543312838 | datasets list --user nartaa |
| `nartaa/rsna-knee-s6fall-weights` | v3 | 2026-10-14T19:08 | 185910765 | datasets list --user nartaa |
| `nartaa/rsna24-helpers` | v8 | 2024-09-04T17:37 | 519217604 | datasets list --user nartaa |
| `nartaa/rsna24-npy1` | v1 | 2024-08-09T17:47 | 10975586069 | datasets list --user nartaa |
| `nartaa/rsna24-npy2` | v1 | 2024-08-07T18:36 | 11052914944 | datasets list --user nartaa |
| `nartaa/rsna24-npy3` | v1 | 2024-08-07T18:37 | 11004266683 | datasets list --user nartaa |
| `nartaa/rsna24-npy4` | v1 | 2024-08-07T18:37 | 3379991785 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models` | v16 | 2024-09-10T08:38 | 259425853 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models-testbed-000` | v1 | 2024-09-12T09:47 | 86077478 | datasets list --user nartaa |
| `nartaa/rsna24-train-images` | v2 | 2024-09-12T07:27 | 5877188280 | datasets list --user nartaa |

Errors (1): kvm raihanjs/rsna-knee-fork-0957-heliosli http 404

## Run 2026-10-14T23:50:24Z (10-15 05:20 IST)

- listed 379 notebooks (score pages + recency pages + key authors), known 416, read 110 (new 1), pulled -m 1, web calls 182 (errors 0), 182 s.
- max public score among public notebooks: **0.957**; score bands (count of notebooks): {'0.957': 2, '0.956': 35, '0.955': 4, '0.954': 29}; list sort inversions vs page scores: 22.
- RELEASE_CANDIDATE (score >= 0.957 and all inputs public): **heliosli/0-9573-latest-public-dataset-score-in-one-click**
- browser fallback queue (page model unreadable): 0 (browser_todo.json)

Alerts this run:

- WATCH SCORE_APPEARED `sushanthtiruvaipati/rsna-knee-r7-raptor384-raptor224-blend` public score 0.956 (v1)
- WATCH SCORE_APPEARED `ryugohanai/rsna-tta224-reader-20261015` public score 0.956 (v1)
- WATCH SCORE_APPEARED `mizeroluckygall/knee-mri-raptor-384-224-r7-blend` public score 0.956 (v1)
- WATCH SCORE_APPEARED `isotriage/rsna-knee-quad-b2` public score 0.954 (v2)

Top 15 public notebooks by page score (inputs: R7 = heliosli/rsna-knee-compact-r7cnxfsss, OF = heliosli/rsna-orthofoundation-fold0-epoch7-20261004, NS = nartaa/rsna-knee-publication-swa-weights-20261007, R6 = heliosli/rsna-knee-compact-r6cnxfsss, CV1 = heliosli/rsna-knee-compact-v1, GP = goodpjw2008/rsna-knee-2-5d-convnext-reader):

| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |
|---|---|---|---|---|---|---|
| 1 | `bliverrigpdoge/r7-raptor-knee-centred-convnext` | 0.957 | v1 / v1 | NOT all public: ?/private-data-source+NS+OF+R7 | - | 2026-10-14T08:26 |
| 2 | `heliosli/0-9573-latest-public-dataset-score-in-one-click` | 0.957 | v1 / v2 | public NS+OF+heliosli/rsna-knee-r6-frozen-fe055406 | heliosli/rsna-knee-r6-frozen-fe055406 | 2026-10-14T18:13 |
| 3 | `evgendvorkin/rsna-versia-7` | 0.956 | v22 / v22 | public NS+OF+R7 | - | 2026-10-13T14:02 |
| 4 | `heliosli/rsna-knee-blend-gold-old-dataset` | 0.956 | v1 / v2 | public NS+OF+R6 | R6 | 2026-10-11T14:16 |
| 5 | `heliosli/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-11T20:01 |
| 6 | `goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` | 0.956 | v17 / v18 | public GP+NS+OF+R7 | - | 2026-10-14T14:50 |
| 7 | `degnonguidi/rsna-knee-abnormality-detection` | 0.956 | v10 / v10 | public NS+OF+R7 | - | 2026-10-13T08:55 |
| 8 | `jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T06:00 |
| 9 | `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T08:39 |
| 10 | `mayankraaz/rsna-knee-abnormality-pipeline` | 0.956 | v5 / v7 | public NS+OF+R7 | - | 2026-10-14T11:40 |
| 11 | `lavinwins/rsna-gold-medal` | 0.956 | v1 / v3 | public NS+OF+R7 | - | 2026-10-13T13:59 |
| 12 | `mkhamzanov/rsna-b1-ctrl` | 0.956 | v1 / v1 | public GP+NS+OF+R7 | - | 2026-10-13T01:19 |
| 13 | `varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-14T10:52 |
| 14 | `medvax/rsna-knee-r7-source-trial` | 0.956 | v1 / v2 | public NS+OF+R7 | - | 2026-10-13T15:18 |
| 15 | `yuvrajlamba01/rsna-knee-multiscale-orthofoundation-sota-0-956` | 0.956 | v2 / v2 | public NS+OF+R7 | - | 2026-10-13T10:56 |

Pending free reads (no public score yet; ran in the last 72 h; all inputs public, R7 bundle among them; a score
appearing raises SCORE_APPEARED):

- `eeshanvaghjiani2004/knee-oct15-224m` v1, ran 2026-10-14T17:50 UTC, inputs NS+OF+R7
- `rudrpratapsingh/rsna-knee-four-public-models-one-rank-blend` v2, ran 2026-10-14T07:12 UTC, inputs CV1+NS+OF+R7
- `piyush474/rsna-knee-0-956-raptor-convnext-r7-ensemble` v1, ran 2026-10-13T09:02 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-heliosli-r7cnxfsss-0-5-blend` v4, ran 2026-10-13T06:42 UTC, inputs NS+OF+R7
- `z1358028281/0-957-latest-dataset-score-in-one-click` v2, ran 2026-10-13T02:28 UTC, inputs NS+OF+R7

Watched datasets (heliosli + nartaa lists, RESEARCH_960 baseline set; knee-related only, all in watch_state.json):

| dataset | version | last updated | bytes | via |
|---|---|---|---|---|
| `dreaddevelopment/raptor-knee-widedense` | v2 | 2026-08-23T04:12 | 543320725 | GetDatasetBasics (anonymous) |
| `heliosli/rsna-knee-compact-r6cnxfsss` | v2 | 2026-10-11T13:18 | 823081993 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-r7cnxfsss` | v1 | 2026-10-11T19:59 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-v1` | v13 | 2026-10-10T19:50 | 823082549 | datasets list --user heliosli |
| `heliosli/rsna-knee-r6-frozen-fe055406` | v2 | 2026-10-14T18:12 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | v1 | 2026-10-04T07:54 | 1343091054 | datasets list --user heliosli |
| `jessky/rsna-knee-local-cv-cnx50` | v1 | 2026-10-12T01:15 | 931795323 | GetDatasetBasics (anonymous) |
| `nartaa/rsna-knee-hpo-assets` | v9 | 2026-09-25T12:01 | 1079732 | datasets list --user nartaa |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | v1 | 2026-10-07T19:26 | 543312838 | datasets list --user nartaa |
| `nartaa/rsna-knee-s6fall-weights` | v3 | 2026-10-14T19:08 | 185910765 | datasets list --user nartaa |
| `nartaa/rsna24-helpers` | v8 | 2024-09-04T17:37 | 519217604 | datasets list --user nartaa |
| `nartaa/rsna24-npy1` | v1 | 2024-08-09T17:47 | 10975586069 | datasets list --user nartaa |
| `nartaa/rsna24-npy2` | v1 | 2024-08-07T18:36 | 11052914944 | datasets list --user nartaa |
| `nartaa/rsna24-npy3` | v1 | 2024-08-07T18:37 | 11004266683 | datasets list --user nartaa |
| `nartaa/rsna24-npy4` | v1 | 2024-08-07T18:37 | 3379991785 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models` | v16 | 2024-09-10T08:38 | 259425853 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models-testbed-000` | v1 | 2024-09-12T09:47 | 86077478 | datasets list --user nartaa |
| `nartaa/rsna24-train-images` | v2 | 2024-09-12T07:27 | 5877188280 | datasets list --user nartaa |

Errors (1): kvm raihanjs/rsna-knee-fork-0957-heliosli http 404

## Run 2026-10-15T01:23:26Z (10-15 06:53 IST)

- listed 379 notebooks (score pages + recency pages + key authors), known 417, read 106 (new 1), pulled -m 3, web calls 176 (errors 0), 195 s.
- max public score among public notebooks: **0.957**; score bands (count of notebooks): {'0.957': 2, '0.956': 35, '0.955': 4, '0.954': 29}; list sort inversions vs page scores: 22.
- RELEASE_CANDIDATE (score >= 0.957 and all inputs public): **heliosli/0-9573-latest-public-dataset-score-in-one-click**
- browser fallback queue (page model unreadable): 0 (browser_todo.json)

Alerts this run:

- WATCH SCORE_APPEARED `ryeminaray/rsna-knee-lab-global96-inference` public score 0.930 (v1)

Top 15 public notebooks by page score (inputs: R7 = heliosli/rsna-knee-compact-r7cnxfsss, OF = heliosli/rsna-orthofoundation-fold0-epoch7-20261004, NS = nartaa/rsna-knee-publication-swa-weights-20261007, R6 = heliosli/rsna-knee-compact-r6cnxfsss, CV1 = heliosli/rsna-knee-compact-v1, GP = goodpjw2008/rsna-knee-2-5d-convnext-reader):

| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |
|---|---|---|---|---|---|---|
| 1 | `heliosli/0-9573-latest-public-dataset-score-in-one-click` | 0.957 | v1 / v2 | public NS+OF+heliosli/rsna-knee-r6-frozen-fe055406 | heliosli/rsna-knee-r6-frozen-fe055406 | 2026-10-14T18:13 |
| 2 | `bliverrigpdoge/r7-raptor-knee-centred-convnext` | 0.957 | v1 / v1 | NOT all public: ?/private-data-source+NS+OF+R7 | - | 2026-10-14T08:26 |
| 3 | `evgendvorkin/rsna-versia-7` | 0.956 | v22 / v22 | public NS+OF+R7 | - | 2026-10-13T14:02 |
| 4 | `heliosli/rsna-knee-blend-gold-old-dataset` | 0.956 | v1 / v2 | public NS+OF+R6 | R6 | 2026-10-11T14:16 |
| 5 | `heliosli/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-11T20:01 |
| 6 | `goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` | 0.956 | v17 / v20 | public GP+NS+OF+R7 | - | 2026-10-15T00:09 |
| 7 | `degnonguidi/rsna-knee-abnormality-detection` | 0.956 | v10 / v10 | public NS+OF+R7 | - | 2026-10-13T08:55 |
| 8 | `jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T06:00 |
| 9 | `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T08:39 |
| 10 | `mayankraaz/rsna-knee-abnormality-pipeline` | 0.956 | v5 / v7 | public NS+OF+R7 | - | 2026-10-14T11:40 |
| 11 | `lavinwins/rsna-gold-medal` | 0.956 | v1 / v4 | public NS+OF+R7 | - | 2026-10-15T01:02 |
| 12 | `mkhamzanov/rsna-b1-ctrl` | 0.956 | v1 / v1 | public GP+NS+OF+R7 | - | 2026-10-13T01:19 |
| 13 | `varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-14T10:52 |
| 14 | `medvax/rsna-knee-r7-source-trial` | 0.956 | v1 / v2 | public NS+OF+R7 | - | 2026-10-13T15:18 |
| 15 | `yuvrajlamba01/rsna-knee-multiscale-orthofoundation-sota-0-956` | 0.956 | v2 / v2 | public NS+OF+R7 | - | 2026-10-13T10:56 |

Pending free reads (no public score yet; ran in the last 72 h; all inputs public, R7 bundle among them; a score
appearing raises SCORE_APPEARED):

- `sushanthtiruvaipati/rsna-knee-perfinding-dino5-prior` v4, ran 2026-10-15T00:33 UTC, inputs NS+OF+R7
- `eeshanvaghjiani2004/knee-oct15-224m` v1, ran 2026-10-14T17:50 UTC, inputs NS+OF+R7
- `rudrpratapsingh/rsna-knee-four-public-models-one-rank-blend` v2, ran 2026-10-14T07:12 UTC, inputs CV1+NS+OF+R7
- `piyush474/rsna-knee-0-956-raptor-convnext-r7-ensemble` v1, ran 2026-10-13T09:02 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-heliosli-r7cnxfsss-0-5-blend` v4, ran 2026-10-13T06:42 UTC, inputs NS+OF+R7
- `z1358028281/0-957-latest-dataset-score-in-one-click` v2, ran 2026-10-13T02:28 UTC, inputs NS+OF+R7

Watched datasets (heliosli + nartaa lists, RESEARCH_960 baseline set; knee-related only, all in watch_state.json):

| dataset | version | last updated | bytes | via |
|---|---|---|---|---|
| `dreaddevelopment/raptor-knee-widedense` | v2 | 2026-08-23T04:12 | 543320725 | GetDatasetBasics (anonymous) |
| `heliosli/rsna-knee-compact-r6cnxfsss` | v2 | 2026-10-11T13:18 | 823081993 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-r7cnxfsss` | v1 | 2026-10-11T19:59 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-v1` | v13 | 2026-10-10T19:50 | 823082549 | datasets list --user heliosli |
| `heliosli/rsna-knee-r6-frozen-fe055406` | v2 | 2026-10-14T18:12 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | v1 | 2026-10-04T07:54 | 1343091054 | datasets list --user heliosli |
| `jessky/rsna-knee-local-cv-cnx50` | v1 | 2026-10-12T01:15 | 931795323 | GetDatasetBasics (anonymous) |
| `nartaa/rsna-knee-hpo-assets` | v9 | 2026-09-25T12:01 | 1079732 | datasets list --user nartaa |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | v1 | 2026-10-07T19:26 | 543312838 | datasets list --user nartaa |
| `nartaa/rsna-knee-s6fall-weights` | v3 | 2026-10-14T19:08 | 185910765 | datasets list --user nartaa |
| `nartaa/rsna24-helpers` | v8 | 2024-09-04T17:37 | 519217604 | datasets list --user nartaa |
| `nartaa/rsna24-npy1` | v1 | 2024-08-09T17:47 | 10975586069 | datasets list --user nartaa |
| `nartaa/rsna24-npy2` | v1 | 2024-08-07T18:36 | 11052914944 | datasets list --user nartaa |
| `nartaa/rsna24-npy3` | v1 | 2024-08-07T18:37 | 11004266683 | datasets list --user nartaa |
| `nartaa/rsna24-npy4` | v1 | 2024-08-07T18:37 | 3379991785 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models` | v16 | 2024-09-10T08:38 | 259425853 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models-testbed-000` | v1 | 2024-09-12T09:47 | 86077478 | datasets list --user nartaa |
| `nartaa/rsna24-train-images` | v2 | 2024-09-12T07:27 | 5877188280 | datasets list --user nartaa |

Errors (1): kvm raihanjs/rsna-knee-fork-0957-heliosli http 404

## Run 2026-10-15T02:56:41Z (10-15 08:26 IST)

- listed 379 notebooks (score pages + recency pages + key authors), known 419, read 107 (new 2), pulled -m 2, web calls 178 (errors 0), 185 s.
- max public score among public notebooks: **0.957**; score bands (count of notebooks): {'0.957': 2, '0.956': 36, '0.955': 4, '0.954': 29}; list sort inversions vs page scores: 24.
- RELEASE_CANDIDATE (score >= 0.957 and all inputs public): **heliosli/0-9573-latest-public-dataset-score-in-one-click**
- browser fallback queue (page model unreadable): 0 (browser_todo.json)

Alerts this run:

- WATCH SCORE_APPEARED `sushanthtiruvaipati/rsna-knee-perfinding-dino5-prior` public score 0.952 (v1)

Top 15 public notebooks by page score (inputs: R7 = heliosli/rsna-knee-compact-r7cnxfsss, OF = heliosli/rsna-orthofoundation-fold0-epoch7-20261004, NS = nartaa/rsna-knee-publication-swa-weights-20261007, R6 = heliosli/rsna-knee-compact-r6cnxfsss, CV1 = heliosli/rsna-knee-compact-v1, GP = goodpjw2008/rsna-knee-2-5d-convnext-reader):

| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |
|---|---|---|---|---|---|---|
| 1 | `heliosli/0-9573-latest-public-dataset-score-in-one-click` | 0.957 | v1 / v2 | public NS+OF+heliosli/rsna-knee-r6-frozen-fe055406 | heliosli/rsna-knee-r6-frozen-fe055406 | 2026-10-14T18:13 |
| 2 | `bliverrigpdoge/r7-raptor-knee-centred-convnext` | 0.957 | v1 / v1 | NOT all public: ?/private-data-source+NS+OF+R7 | - | 2026-10-14T08:26 |
| 3 | `evgendvorkin/rsna-versia-7` | 0.956 | v22 / v22 | public NS+OF+R7 | - | 2026-10-13T14:02 |
| 4 | `heliosli/rsna-knee-blend-gold-old-dataset` | 0.956 | v1 / v2 | public NS+OF+R6 | R6 | 2026-10-11T14:16 |
| 5 | `heliosli/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-11T20:01 |
| 6 | `goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` | 0.956 | v17 / v20 | public GP+NS+OF+R7 | - | 2026-10-15T00:09 |
| 7 | `degnonguidi/rsna-knee-abnormality-detection` | 0.956 | v10 / v10 | public NS+OF+R7 | - | 2026-10-13T08:55 |
| 8 | `jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T06:00 |
| 9 | `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T08:39 |
| 10 | `mayankraaz/rsna-knee-abnormality-pipeline` | 0.956 | v5 / v7 | public NS+OF+R7 | - | 2026-10-14T11:40 |
| 11 | `lavinwins/rsna-gold-medal` | 0.956 | v1 / v4 | public NS+OF+R7 | - | 2026-10-15T01:02 |
| 12 | `mkhamzanov/rsna-b1-ctrl` | 0.956 | v1 / v1 | public GP+NS+OF+R7 | - | 2026-10-13T01:19 |
| 13 | `varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-14T10:52 |
| 14 | `medvax/rsna-knee-r7-source-trial` | 0.956 | v1 / v2 | public NS+OF+R7 | - | 2026-10-13T15:18 |
| 15 | `yuvrajlamba01/rsna-knee-multiscale-orthofoundation-sota-0-956` | 0.956 | v2 / v2 | public NS+OF+R7 | - | 2026-10-13T10:56 |

Pending free reads (no public score yet; ran in the last 72 h; all inputs public, R7 bundle among them; a score
appearing raises SCORE_APPEARED):

- `isotriage/rsna-knee-h957` v3, ran 2026-10-15T01:58 UTC, inputs NS+OF+R7
- `eeshanvaghjiani2004/knee-oct15-224m` v1, ran 2026-10-14T17:50 UTC, inputs NS+OF+R7
- `rudrpratapsingh/rsna-knee-four-public-models-one-rank-blend` v2, ran 2026-10-14T07:12 UTC, inputs CV1+NS+OF+R7
- `piyush474/rsna-knee-0-956-raptor-convnext-r7-ensemble` v1, ran 2026-10-13T09:02 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-heliosli-r7cnxfsss-0-5-blend` v4, ran 2026-10-13T06:42 UTC, inputs NS+OF+R7
- `z1358028281/0-957-latest-dataset-score-in-one-click` v2, ran 2026-10-13T02:28 UTC, inputs NS+OF+R7

Watched datasets (heliosli + nartaa lists, RESEARCH_960 baseline set; knee-related only, all in watch_state.json):

| dataset | version | last updated | bytes | via |
|---|---|---|---|---|
| `dreaddevelopment/raptor-knee-widedense` | v2 | 2026-08-23T04:12 | 543320725 | GetDatasetBasics (anonymous) |
| `heliosli/rsna-knee-compact-r6cnxfsss` | v2 | 2026-10-11T13:18 | 823081993 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-r7cnxfsss` | v1 | 2026-10-11T19:59 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-v1` | v13 | 2026-10-10T19:50 | 823082549 | datasets list --user heliosli |
| `heliosli/rsna-knee-r6-frozen-fe055406` | v2 | 2026-10-14T18:12 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | v1 | 2026-10-04T07:54 | 1343091054 | datasets list --user heliosli |
| `jessky/rsna-knee-local-cv-cnx50` | v1 | 2026-10-12T01:15 | 931795323 | GetDatasetBasics (anonymous) |
| `nartaa/rsna-knee-hpo-assets` | v9 | 2026-09-25T12:01 | 1079732 | datasets list --user nartaa |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | v1 | 2026-10-07T19:26 | 543312838 | datasets list --user nartaa |
| `nartaa/rsna-knee-s6fall-weights` | v3 | 2026-10-14T19:08 | 185910765 | datasets list --user nartaa |
| `nartaa/rsna24-helpers` | v8 | 2024-09-04T17:37 | 519217604 | datasets list --user nartaa |
| `nartaa/rsna24-npy1` | v1 | 2024-08-09T17:47 | 10975586069 | datasets list --user nartaa |
| `nartaa/rsna24-npy2` | v1 | 2024-08-07T18:36 | 11052914944 | datasets list --user nartaa |
| `nartaa/rsna24-npy3` | v1 | 2024-08-07T18:37 | 11004266683 | datasets list --user nartaa |
| `nartaa/rsna24-npy4` | v1 | 2024-08-07T18:37 | 3379991785 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models` | v16 | 2024-09-10T08:38 | 259425853 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models-testbed-000` | v1 | 2024-09-12T09:47 | 86077478 | datasets list --user nartaa |
| `nartaa/rsna24-train-images` | v2 | 2024-09-12T07:27 | 5877188280 | datasets list --user nartaa |

Errors (1): kvm raihanjs/rsna-knee-fork-0957-heliosli http 404

## Run 2026-10-15T04:29:46Z (10-15 09:59 IST)

- listed 383 notebooks (score pages + recency pages + key authors), known 426, read 112 (new 7), pulled -m 8, web calls 183 (errors 0), 265 s.
- max public score among public notebooks: **0.957**; score bands (count of notebooks): {'0.957': 2, '0.956': 36, '0.955': 4, '0.954': 29}; list sort inversions vs page scores: 24.
- RELEASE_CANDIDATE (score >= 0.957 and all inputs public): **heliosli/0-9573-latest-public-dataset-score-in-one-click**
- browser fallback queue (page model unreadable): 0 (browser_todo.json)

Alerts this run:

- WATCH NEW_INPUT `varianceofx/rsna-knee-h957-reblend` score None uses never-seen ['kernel:isotriage/rsna-knee-h957']

Top 15 public notebooks by page score (inputs: R7 = heliosli/rsna-knee-compact-r7cnxfsss, OF = heliosli/rsna-orthofoundation-fold0-epoch7-20261004, NS = nartaa/rsna-knee-publication-swa-weights-20261007, R6 = heliosli/rsna-knee-compact-r6cnxfsss, CV1 = heliosli/rsna-knee-compact-v1, GP = goodpjw2008/rsna-knee-2-5d-convnext-reader):

| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |
|---|---|---|---|---|---|---|
| 1 | `heliosli/0-9573-latest-public-dataset-score-in-one-click` | 0.957 | v1 / v2 | public NS+OF+heliosli/rsna-knee-r6-frozen-fe055406 | heliosli/rsna-knee-r6-frozen-fe055406 | 2026-10-14T18:13 |
| 2 | `bliverrigpdoge/r7-raptor-knee-centred-convnext` | 0.957 | v1 / v1 | NOT all public: ?/private-data-source+NS+OF+R7 | - | 2026-10-14T08:26 |
| 3 | `evgendvorkin/rsna-versia-7` | 0.956 | v22 / v22 | public NS+OF+R7 | - | 2026-10-13T14:02 |
| 4 | `heliosli/rsna-knee-blend-gold-old-dataset` | 0.956 | v1 / v2 | public NS+OF+R6 | R6 | 2026-10-11T14:16 |
| 5 | `heliosli/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-11T20:01 |
| 6 | `goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` | 0.956 | v17 / v20 | public GP+NS+OF+R7 | - | 2026-10-15T00:09 |
| 7 | `degnonguidi/rsna-knee-abnormality-detection` | 0.956 | v10 / v10 | public NS+OF+R7 | - | 2026-10-13T08:55 |
| 8 | `jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T06:00 |
| 9 | `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T08:39 |
| 10 | `mayankraaz/rsna-knee-abnormality-pipeline` | 0.956 | v5 / v7 | public NS+OF+R7 | - | 2026-10-14T11:40 |
| 11 | `lavinwins/rsna-gold-medal` | 0.956 | v1 / v4 | public NS+OF+R7 | - | 2026-10-15T01:02 |
| 12 | `mkhamzanov/rsna-b1-ctrl` | 0.956 | v1 / v1 | public GP+NS+OF+R7 | - | 2026-10-13T01:19 |
| 13 | `varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-14T10:52 |
| 14 | `medvax/rsna-knee-r7-source-trial` | 0.956 | v1 / v2 | public NS+OF+R7 | - | 2026-10-13T15:18 |
| 15 | `yuvrajlamba01/rsna-knee-multiscale-orthofoundation-sota-0-956` | 0.956 | v2 / v2 | public NS+OF+R7 | - | 2026-10-13T10:56 |

Pending free reads (no public score yet; ran in the last 72 h; all inputs public, R7 bundle among them; a score
appearing raises SCORE_APPEARED):

- `rjings/rsna-knee-target-local-router-20261015` v2, ran 2026-10-15T03:59 UTC, inputs NS+OF+R7
- `oscarcarreoserpa/rsna-knee-one-unused-checkpoint-lifts-the-0-956` v1, ran 2026-10-15T03:03 UTC, inputs NS+OF+R7
- `isotriage/rsna-knee-h957` v3, ran 2026-10-15T01:58 UTC, inputs NS+OF+R7
- `eeshanvaghjiani2004/knee-oct15-224m` v1, ran 2026-10-14T17:50 UTC, inputs NS+OF+R7
- `rudrpratapsingh/rsna-knee-four-public-models-one-rank-blend` v2, ran 2026-10-14T07:12 UTC, inputs CV1+NS+OF+R7
- `piyush474/rsna-knee-0-956-raptor-convnext-r7-ensemble` v1, ran 2026-10-13T09:02 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-heliosli-r7cnxfsss-0-5-blend` v4, ran 2026-10-13T06:42 UTC, inputs NS+OF+R7
- `z1358028281/0-957-latest-dataset-score-in-one-click` v2, ran 2026-10-13T02:28 UTC, inputs NS+OF+R7

Watched datasets (heliosli + nartaa lists, RESEARCH_960 baseline set; knee-related only, all in watch_state.json):

| dataset | version | last updated | bytes | via |
|---|---|---|---|---|
| `dreaddevelopment/raptor-knee-widedense` | v2 | 2026-08-23T04:12 | 543320725 | GetDatasetBasics (anonymous) |
| `heliosli/rsna-knee-compact-r6cnxfsss` | v2 | 2026-10-11T13:18 | 823081993 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-r7cnxfsss` | v1 | 2026-10-11T19:59 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-v1` | v13 | 2026-10-10T19:50 | 823082549 | datasets list --user heliosli |
| `heliosli/rsna-knee-r6-frozen-fe055406` | v2 | 2026-10-14T18:12 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | v1 | 2026-10-04T07:54 | 1343091054 | datasets list --user heliosli |
| `jessky/rsna-knee-local-cv-cnx50` | v1 | 2026-10-12T01:15 | 931795323 | GetDatasetBasics (anonymous) |
| `nartaa/rsna-knee-hpo-assets` | v9 | 2026-09-25T12:01 | 1079732 | datasets list --user nartaa |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | v1 | 2026-10-07T19:26 | 543312838 | datasets list --user nartaa |
| `nartaa/rsna-knee-s6fall-weights` | v3 | 2026-10-14T19:08 | 185910765 | datasets list --user nartaa |
| `nartaa/rsna24-helpers` | v8 | 2024-09-04T17:37 | 519217604 | datasets list --user nartaa |
| `nartaa/rsna24-npy1` | v1 | 2024-08-09T17:47 | 10975586069 | datasets list --user nartaa |
| `nartaa/rsna24-npy2` | v1 | 2024-08-07T18:36 | 11052914944 | datasets list --user nartaa |
| `nartaa/rsna24-npy3` | v1 | 2024-08-07T18:37 | 11004266683 | datasets list --user nartaa |
| `nartaa/rsna24-npy4` | v1 | 2024-08-07T18:37 | 3379991785 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models` | v16 | 2024-09-10T08:38 | 259425853 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models-testbed-000` | v1 | 2024-09-12T09:47 | 86077478 | datasets list --user nartaa |
| `nartaa/rsna24-train-images` | v2 | 2024-09-12T07:27 | 5877188280 | datasets list --user nartaa |

Errors (2): kvm mizeroluckygall/knee-mri-raptor-384-224-r7-blend http 404; kvm raihanjs/rsna-knee-fork-0957-heliosli http 404

## Run 2026-10-15T06:04:12Z (10-15 11:34 IST)

- listed 382 notebooks (score pages + recency pages + key authors), known 427, read 112 (new 1), pulled -m 2, web calls 183 (errors 0), 192 s.
- max public score among public notebooks: **0.957**; score bands (count of notebooks): {'0.957': 2, '0.956': 38, '0.955': 4, '0.954': 29}; list sort inversions vs page scores: 25.
- RELEASE_CANDIDATE (score >= 0.957 and all inputs public): **heliosli/0-9573-latest-public-dataset-score-in-one-click**
- browser fallback queue (page model unreadable): 0 (browser_todo.json)

Alerts this run:

- WATCH SCORE_APPEARED `rudrpratapsingh/rsna-knee-four-public-models-one-rank-blend` public score 0.956 (v2)
- WATCH SCORE_APPEARED `isotriage/rsna-knee-h957` public score 0.956 (v3)

Top 15 public notebooks by page score (inputs: R7 = heliosli/rsna-knee-compact-r7cnxfsss, OF = heliosli/rsna-orthofoundation-fold0-epoch7-20261004, NS = nartaa/rsna-knee-publication-swa-weights-20261007, R6 = heliosli/rsna-knee-compact-r6cnxfsss, CV1 = heliosli/rsna-knee-compact-v1, GP = goodpjw2008/rsna-knee-2-5d-convnext-reader):

| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |
|---|---|---|---|---|---|---|
| 1 | `heliosli/0-9573-latest-public-dataset-score-in-one-click` | 0.957 | v1 / v2 | public NS+OF+heliosli/rsna-knee-r6-frozen-fe055406 | heliosli/rsna-knee-r6-frozen-fe055406 | 2026-10-14T18:13 |
| 2 | `bliverrigpdoge/r7-raptor-knee-centred-convnext` | 0.957 | v1 / v1 | NOT all public: ?/private-data-source+NS+OF+R7 | - | 2026-10-14T08:26 |
| 3 | `evgendvorkin/rsna-versia-7` | 0.956 | v22 / v22 | public NS+OF+R7 | - | 2026-10-13T14:02 |
| 4 | `heliosli/rsna-knee-blend-gold-old-dataset` | 0.956 | v1 / v2 | public NS+OF+R6 | R6 | 2026-10-11T14:16 |
| 5 | `heliosli/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-11T20:01 |
| 6 | `goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` | 0.956 | v17 / v20 | public GP+NS+OF+R7 | - | 2026-10-15T00:09 |
| 7 | `degnonguidi/rsna-knee-abnormality-detection` | 0.956 | v10 / v10 | public NS+OF+R7 | - | 2026-10-13T08:55 |
| 8 | `jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T06:00 |
| 9 | `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T08:39 |
| 10 | `lavinwins/rsna-gold-medal` | 0.956 | v1 / v4 | public NS+OF+R7 | - | 2026-10-15T01:02 |
| 11 | `mayankraaz/rsna-knee-abnormality-pipeline` | 0.956 | v5 / v7 | public NS+OF+R7 | - | 2026-10-14T11:40 |
| 12 | `mkhamzanov/rsna-b1-ctrl` | 0.956 | v1 / v1 | public GP+NS+OF+R7 | - | 2026-10-13T01:19 |
| 13 | `varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-14T10:52 |
| 14 | `medvax/rsna-knee-r7-source-trial` | 0.956 | v1 / v2 | public NS+OF+R7 | - | 2026-10-13T15:18 |
| 15 | `yuvrajlamba01/rsna-knee-multiscale-orthofoundation-sota-0-956` | 0.956 | v2 / v2 | public NS+OF+R7 | - | 2026-10-13T10:56 |

Pending free reads (no public score yet; ran in the last 72 h; all inputs public, R7 bundle among them; a score
appearing raises SCORE_APPEARED):

- `rjings/rsna-knee-target-local-router-20261015` v2, ran 2026-10-15T03:59 UTC, inputs NS+OF+R7
- `oscarcarreoserpa/rsna-knee-one-unused-checkpoint-lifts-the-0-956` v1, ran 2026-10-15T03:03 UTC, inputs NS+OF+R7
- `eeshanvaghjiani2004/knee-oct15-224m` v1, ran 2026-10-14T17:50 UTC, inputs NS+OF+R7
- `piyush474/rsna-knee-0-956-raptor-convnext-r7-ensemble` v1, ran 2026-10-13T09:02 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-heliosli-r7cnxfsss-0-5-blend` v4, ran 2026-10-13T06:42 UTC, inputs NS+OF+R7
- `z1358028281/0-957-latest-dataset-score-in-one-click` v2, ran 2026-10-13T02:28 UTC, inputs NS+OF+R7

Watched datasets (heliosli + nartaa lists, RESEARCH_960 baseline set; knee-related only, all in watch_state.json):

| dataset | version | last updated | bytes | via |
|---|---|---|---|---|
| `dreaddevelopment/raptor-knee-widedense` | v2 | 2026-08-23T04:12 | 543320725 | GetDatasetBasics (anonymous) |
| `heliosli/rsna-knee-compact-r6cnxfsss` | v2 | 2026-10-11T13:18 | 823081993 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-r7cnxfsss` | v1 | 2026-10-11T19:59 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-v1` | v13 | 2026-10-10T19:50 | 823082549 | datasets list --user heliosli |
| `heliosli/rsna-knee-r6-frozen-fe055406` | v2 | 2026-10-14T18:12 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | v1 | 2026-10-04T07:54 | 1343091054 | datasets list --user heliosli |
| `jessky/rsna-knee-local-cv-cnx50` | v1 | 2026-10-12T01:15 | 931795323 | GetDatasetBasics (anonymous) |
| `nartaa/rsna-knee-hpo-assets` | v9 | 2026-09-25T12:01 | 1079732 | datasets list --user nartaa |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | v1 | 2026-10-07T19:26 | 543312838 | datasets list --user nartaa |
| `nartaa/rsna-knee-s6fall-weights` | v3 | 2026-10-14T19:08 | 185910765 | datasets list --user nartaa |
| `nartaa/rsna24-helpers` | v8 | 2024-09-04T17:37 | 519217604 | datasets list --user nartaa |
| `nartaa/rsna24-npy1` | v1 | 2024-08-09T17:47 | 10975586069 | datasets list --user nartaa |
| `nartaa/rsna24-npy2` | v1 | 2024-08-07T18:36 | 11052914944 | datasets list --user nartaa |
| `nartaa/rsna24-npy3` | v1 | 2024-08-07T18:37 | 11004266683 | datasets list --user nartaa |
| `nartaa/rsna24-npy4` | v1 | 2024-08-07T18:37 | 3379991785 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models` | v16 | 2024-09-10T08:38 | 259425853 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models-testbed-000` | v1 | 2024-09-12T09:47 | 86077478 | datasets list --user nartaa |
| `nartaa/rsna24-train-images` | v2 | 2024-09-12T07:27 | 5877188280 | datasets list --user nartaa |

Errors (2): kvm mizeroluckygall/knee-mri-raptor-384-224-r7-blend http 404; kvm raihanjs/rsna-knee-fork-0957-heliosli http 404

## Run 2026-10-15T07:37:24Z (10-15 13:07 IST)

- listed 383 notebooks (score pages + recency pages + key authors), known 431, read 115 (new 4), pulled -m 6, web calls 188 (errors 0), 206 s.
- max public score among public notebooks: **0.957**; score bands (count of notebooks): {'0.957': 3, '0.956': 38, '0.955': 4, '0.954': 29}; list sort inversions vs page scores: 25.
- RELEASE_CANDIDATE (score >= 0.957 and all inputs public): **heliosli/0-9573-latest-public-dataset-score-in-one-click, heliosli/old-version**
- browser fallback queue (page model unreadable): 0 (browser_todo.json)

Alerts this run:

- FLAG RELEASE_CANDIDATE `heliosli/old-version` score 0.957 v1 (current v2), inputs ALL PUBLIC, DRIFT {"heliosli/rsna-knee-r6-frozen-fe055406": {"pinned_version_id": 20611930, "current_version_id": 20731557, "current_version": 2}}; next: python session_2026-10-15/watch/swap_builder.py heliosli/old-version/1 --name rsna-sub-rel-heliosli-old-version
- WATCH KEY_AUTHOR_NEW_NOTEBOOK `heliosli/old-version` score 0.957, inputs {'competition:rsna-knee-abnormality-detection': 'ours', 'dataset:heliosli/rsna-orthofoundation-fold0-epoch7-20261004': 'public', 'dataset:heliosli/rsna-knee-r6-frozen-fe055406': 'public', 'dataset:nartaa/rsna-knee-publication-swa-weights-20261007': 'public'}
- WATCH NEW_INPUT `crystalwong01/rsna-knee-helios-r6-frozen-blend-public-run` score None uses never-seen ['kernel:crystalwong01/rsna-knee-r6-frozen-fe055406']

Top 15 public notebooks by page score (inputs: R7 = heliosli/rsna-knee-compact-r7cnxfsss, OF = heliosli/rsna-orthofoundation-fold0-epoch7-20261004, NS = nartaa/rsna-knee-publication-swa-weights-20261007, R6 = heliosli/rsna-knee-compact-r6cnxfsss, CV1 = heliosli/rsna-knee-compact-v1, GP = goodpjw2008/rsna-knee-2-5d-convnext-reader):

| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |
|---|---|---|---|---|---|---|
| 1 | `heliosli/old-version` | 0.957 | v1 / v2 | public NS+OF+heliosli/rsna-knee-r6-frozen-fe055406 | heliosli/rsna-knee-r6-frozen-fe055406 | 2026-10-14T18:13 |
| 2 | `bliverrigpdoge/r7-raptor-knee-centred-convnext` | 0.957 | v1 / v1 | NOT all public: ?/private-data-source+NS+OF+R7 | - | 2026-10-14T08:26 |
| 3 | `heliosli/0-9573-latest-public-dataset-score-in-one-click` | 0.957 | v1 / v1 | public NS+OF+heliosli/rsna-knee-r6-frozen-fe055406 | heliosli/rsna-knee-r6-frozen-fe055406 | 2026-10-14T12:21 |
| 4 | `evgendvorkin/rsna-versia-7` | 0.956 | v22 / v22 | public NS+OF+R7 | - | 2026-10-13T14:02 |
| 5 | `heliosli/rsna-knee-blend-gold-old-dataset` | 0.956 | v1 / v2 | public NS+OF+R6 | R6 | 2026-10-11T14:16 |
| 6 | `heliosli/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-11T20:01 |
| 7 | `goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` | 0.956 | v17 / v20 | public GP+NS+OF+R7 | - | 2026-10-15T00:09 |
| 8 | `degnonguidi/rsna-knee-abnormality-detection` | 0.956 | v10 / v10 | public NS+OF+R7 | - | 2026-10-13T08:55 |
| 9 | `jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T06:00 |
| 10 | `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T08:39 |
| 11 | `lavinwins/rsna-gold-medal` | 0.956 | v1 / v4 | public NS+OF+R7 | - | 2026-10-15T01:02 |
| 12 | `mayankraaz/rsna-knee-abnormality-pipeline` | 0.956 | v5 / v7 | public NS+OF+R7 | - | 2026-10-14T11:40 |
| 13 | `mkhamzanov/rsna-b1-ctrl` | 0.956 | v1 / v1 | public GP+NS+OF+R7 | - | 2026-10-13T01:19 |
| 14 | `varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-14T10:52 |
| 15 | `medvax/rsna-knee-r7-source-trial` | 0.956 | v1 / v2 | public NS+OF+R7 | - | 2026-10-13T15:18 |

Pending free reads (no public score yet; ran in the last 72 h; all inputs public, R7 bundle among them; a score
appearing raises SCORE_APPEARED):

- `rjings/rsna-knee-target-local-router-20261015` v2, ran 2026-10-15T03:59 UTC, inputs NS+OF+R7
- `oscarcarreoserpa/rsna-knee-one-unused-checkpoint-lifts-the-0-956` v1, ran 2026-10-15T03:03 UTC, inputs NS+OF+R7
- `eeshanvaghjiani2004/knee-oct15-224m` v1, ran 2026-10-14T17:50 UTC, inputs NS+OF+R7
- `piyush474/rsna-knee-0-956-raptor-convnext-r7-ensemble` v1, ran 2026-10-13T09:02 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-heliosli-r7cnxfsss-0-5-blend` v4, ran 2026-10-13T06:42 UTC, inputs NS+OF+R7
- `z1358028281/0-957-latest-dataset-score-in-one-click` v2, ran 2026-10-13T02:28 UTC, inputs NS+OF+R7

Watched datasets (heliosli + nartaa lists, RESEARCH_960 baseline set; knee-related only, all in watch_state.json):

| dataset | version | last updated | bytes | via |
|---|---|---|---|---|
| `dreaddevelopment/raptor-knee-widedense` | v2 | 2026-08-23T04:12 | 543320725 | GetDatasetBasics (anonymous) |
| `heliosli/rsna-knee-compact-r6cnxfsss` | v2 | 2026-10-11T13:18 | 823081993 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-r7cnxfsss` | v1 | 2026-10-11T19:59 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-v1` | v13 | 2026-10-10T19:50 | 823082549 | datasets list --user heliosli |
| `heliosli/rsna-knee-r6-frozen-fe055406` | v2 | 2026-10-14T18:12 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | v1 | 2026-10-04T07:54 | 1343091054 | datasets list --user heliosli |
| `jessky/rsna-knee-local-cv-cnx50` | v1 | 2026-10-12T01:15 | 931795323 | GetDatasetBasics (anonymous) |
| `nartaa/rsna-knee-hpo-assets` | v9 | 2026-09-25T12:01 | 1079732 | datasets list --user nartaa |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | v1 | 2026-10-07T19:26 | 543312838 | datasets list --user nartaa |
| `nartaa/rsna-knee-s6fall-weights` | v3 | 2026-10-14T19:08 | 185910765 | datasets list --user nartaa |
| `nartaa/rsna24-helpers` | v8 | 2024-09-04T17:37 | 519217604 | datasets list --user nartaa |
| `nartaa/rsna24-npy1` | v1 | 2024-08-09T17:47 | 10975586069 | datasets list --user nartaa |
| `nartaa/rsna24-npy2` | v1 | 2024-08-07T18:36 | 11052914944 | datasets list --user nartaa |
| `nartaa/rsna24-npy3` | v1 | 2024-08-07T18:37 | 11004266683 | datasets list --user nartaa |
| `nartaa/rsna24-npy4` | v1 | 2024-08-07T18:37 | 3379991785 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models` | v16 | 2024-09-10T08:38 | 259425853 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models-testbed-000` | v1 | 2024-09-12T09:47 | 86077478 | datasets list --user nartaa |
| `nartaa/rsna24-train-images` | v2 | 2024-09-12T07:27 | 5877188280 | datasets list --user nartaa |

Errors (2): kvm mizeroluckygall/knee-mri-raptor-384-224-r7-blend http 404; kvm raihanjs/rsna-knee-fork-0957-heliosli http 404

## Run 2026-10-15T09:10:49Z (10-15 14:40 IST)

- listed 382 notebooks (score pages + recency pages + key authors), known 434, read 114 (new 3), pulled -m 3, web calls 187 (errors 0), 235 s.
- max public score among public notebooks: **0.957**; score bands (count of notebooks): {'0.957': 3, '0.956': 38, '0.955': 4, '0.954': 29}; list sort inversions vs page scores: 25.
- RELEASE_CANDIDATE (score >= 0.957 and all inputs public): **heliosli/this-one-is-old-version-0-957-new-one-you-can-use, heliosli/0-9576-latest-public-dataset-score-in-one-click**
- browser fallback queue (page model unreadable): 0 (browser_todo.json)

Alerts this run:

- FLAG RELEASE_CANDIDATE `heliosli/this-one-is-old-version-0-957-new-one-you-can-use` score 0.957 v1 (current v2), inputs ALL PUBLIC, DRIFT {"heliosli/rsna-knee-r6-frozen-fe055406": {"pinned_version_id": 20611930, "current_version_id": 20731557, "current_version": 2}}; next: python session_2026-10-15/watch/swap_builder.py heliosli/this-one-is-old-version-0-957-new-one-you-can-use/1 --name rsna-sub-rel-heliosli-this-one-is-old-versi
- WATCH KEY_AUTHOR_NEW_NOTEBOOK `heliosli/this-one-is-old-version-0-957-new-one-you-can-use` score 0.957, inputs {'competition:rsna-knee-abnormality-detection': 'ours', 'dataset:heliosli/rsna-orthofoundation-fold0-epoch7-20261004': 'public', 'dataset:heliosli/rsna-knee-r6-frozen-fe055406': 'public', 'dataset:nartaa/rsna-knee-publication-swa-weights-20261007': 'public'}
- FLAG RELEASE_CANDIDATE `heliosli/0-9576-latest-public-dataset-score-in-one-click` score 0.957 v1 (current v1), inputs ALL PUBLIC, DRIFT {"heliosli/rsna-knee-r6-frozen-fe055406": {"pinned_version_id": 20611930, "current_version_id": 20731557, "current_version": 2}}; next: python session_2026-10-15/watch/swap_builder.py heliosli/0-9576-latest-public-dataset-score-in-one-click/1 --name rsna-sub-rel-heliosli-0-9576-latest-public
- WATCH KEY_AUTHOR_NEW_NOTEBOOK `heliosli/0-9576-latest-public-dataset-score-in-one-click` score 0.957, inputs {'competition:rsna-knee-abnormality-detection': 'ours', 'dataset:heliosli/rsna-orthofoundation-fold0-epoch7-20261004': 'public', 'dataset:nartaa/rsna-knee-publication-swa-weights-20261007': 'public', 'dataset:heliosli/rsna-knee-r6-frozen-fe055406': 'public'}

Top 15 public notebooks by page score (inputs: R7 = heliosli/rsna-knee-compact-r7cnxfsss, OF = heliosli/rsna-orthofoundation-fold0-epoch7-20261004, NS = nartaa/rsna-knee-publication-swa-weights-20261007, R6 = heliosli/rsna-knee-compact-r6cnxfsss, CV1 = heliosli/rsna-knee-compact-v1, GP = goodpjw2008/rsna-knee-2-5d-convnext-reader):

| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |
|---|---|---|---|---|---|---|
| 1 | `heliosli/this-one-is-old-version-0-957-new-one-you-can-use` | 0.957 | v1 / v2 | public NS+OF+heliosli/rsna-knee-r6-frozen-fe055406 | heliosli/rsna-knee-r6-frozen-fe055406 | 2026-10-14T18:13 |
| 2 | `bliverrigpdoge/r7-raptor-knee-centred-convnext` | 0.957 | v1 / v1 | NOT all public: ?/private-data-source+NS+OF+R7 | - | 2026-10-14T08:26 |
| 3 | `heliosli/0-9576-latest-public-dataset-score-in-one-click` | 0.957 | v1 / v1 | public NS+OF+heliosli/rsna-knee-r6-frozen-fe055406 | heliosli/rsna-knee-r6-frozen-fe055406 | 2026-10-14T12:21 |
| 4 | `evgendvorkin/rsna-versia-7` | 0.956 | v22 / v22 | public NS+OF+R7 | - | 2026-10-13T14:02 |
| 5 | `heliosli/rsna-knee-blend-gold-old-dataset` | 0.956 | v1 / v2 | public NS+OF+R6 | R6 | 2026-10-11T14:16 |
| 6 | `heliosli/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-11T20:01 |
| 7 | `goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` | 0.956 | v17 / v20 | public GP+NS+OF+R7 | - | 2026-10-15T00:09 |
| 8 | `degnonguidi/rsna-knee-abnormality-detection` | 0.956 | v10 / v10 | public NS+OF+R7 | - | 2026-10-13T08:55 |
| 9 | `jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T06:00 |
| 10 | `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T08:39 |
| 11 | `lavinwins/rsna-gold-medal` | 0.956 | v1 / v4 | public NS+OF+R7 | - | 2026-10-15T01:02 |
| 12 | `mayankraaz/rsna-knee-abnormality-pipeline` | 0.956 | v5 / v7 | public NS+OF+R7 | - | 2026-10-14T11:40 |
| 13 | `mkhamzanov/rsna-b1-ctrl` | 0.956 | v1 / v1 | public GP+NS+OF+R7 | - | 2026-10-13T01:19 |
| 14 | `varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-14T10:52 |
| 15 | `medvax/rsna-knee-r7-source-trial` | 0.956 | v1 / v2 | public NS+OF+R7 | - | 2026-10-13T15:18 |

Pending free reads (no public score yet; ran in the last 72 h; all inputs public, R7 bundle among them; a score
appearing raises SCORE_APPEARED):

- `rjings/rsna-knee-target-local-router-20261015` v2, ran 2026-10-15T03:59 UTC, inputs NS+OF+R7
- `oscarcarreoserpa/rsna-knee-one-unused-checkpoint-lifts-the-0-956` v1, ran 2026-10-15T03:03 UTC, inputs NS+OF+R7
- `eeshanvaghjiani2004/knee-oct15-224m` v1, ran 2026-10-14T17:50 UTC, inputs NS+OF+R7
- `piyush474/rsna-knee-0-956-raptor-convnext-r7-ensemble` v1, ran 2026-10-13T09:02 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-heliosli-r7cnxfsss-0-5-blend` v4, ran 2026-10-13T06:42 UTC, inputs NS+OF+R7
- `z1358028281/0-957-latest-dataset-score-in-one-click` v2, ran 2026-10-13T02:28 UTC, inputs NS+OF+R7

Watched datasets (heliosli + nartaa lists, RESEARCH_960 baseline set; knee-related only, all in watch_state.json):

| dataset | version | last updated | bytes | via |
|---|---|---|---|---|
| `dreaddevelopment/raptor-knee-widedense` | v2 | 2026-08-23T04:12 | 543320725 | GetDatasetBasics (anonymous) |
| `heliosli/rsna-knee-compact-r6cnxfsss` | v2 | 2026-10-11T13:18 | 823081993 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-r7cnxfsss` | v1 | 2026-10-11T19:59 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-v1` | v13 | 2026-10-10T19:50 | 823082549 | datasets list --user heliosli |
| `heliosli/rsna-knee-r6-frozen-fe055406` | v2 | 2026-10-14T18:12 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | v1 | 2026-10-04T07:54 | 1343091054 | datasets list --user heliosli |
| `jessky/rsna-knee-local-cv-cnx50` | v1 | 2026-10-12T01:15 | 931795323 | GetDatasetBasics (anonymous) |
| `nartaa/rsna-knee-hpo-assets` | v9 | 2026-09-25T12:01 | 1079732 | datasets list --user nartaa |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | v1 | 2026-10-07T19:26 | 543312838 | datasets list --user nartaa |
| `nartaa/rsna-knee-s6fall-weights` | v3 | 2026-10-14T19:08 | 185910765 | datasets list --user nartaa |
| `nartaa/rsna24-helpers` | v8 | 2024-09-04T17:37 | 519217604 | datasets list --user nartaa |
| `nartaa/rsna24-npy1` | v1 | 2024-08-09T17:47 | 10975586069 | datasets list --user nartaa |
| `nartaa/rsna24-npy2` | v1 | 2024-08-07T18:36 | 11052914944 | datasets list --user nartaa |
| `nartaa/rsna24-npy3` | v1 | 2024-08-07T18:37 | 11004266683 | datasets list --user nartaa |
| `nartaa/rsna24-npy4` | v1 | 2024-08-07T18:37 | 3379991785 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models` | v16 | 2024-09-10T08:38 | 259425853 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models-testbed-000` | v1 | 2024-09-12T09:47 | 86077478 | datasets list --user nartaa |
| `nartaa/rsna24-train-images` | v2 | 2024-09-12T07:27 | 5877188280 | datasets list --user nartaa |

Errors (5): pull heliosli/0-9573-latest-public-dataset-score-in-one-click: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernel; pull heliosli/old-version: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernel; kvm mizeroluckygall/knee-mri-raptor-384-224-r7-blend http 404; kvm raihanjs/rsna-knee-fork-0957-heliosli http 404; pull yuvrajlamba01/rsna-knee-research-optimized: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernel

## Run 2026-10-15T10:03:23Z (10-15 15:33 IST)

- listed 431 notebooks (score pages + recency pages + key authors), known 468, read 141 (new 34), pulled -m 11, web calls 235 (errors 0), 265 s.
- max public score among public notebooks: **0.957**; score bands (count of notebooks): {'0.957': 3, '0.956': 39, '0.955': 4, '0.954': 29}; list sort inversions vs page scores: 25.
- RELEASE_CANDIDATE (score >= 0.957 and all inputs public): **heliosli/0-9576-latest-public-dataset-score-in-one-click, heliosli/this-one-is-old-version-0-957-new-one-you-can-use**
- browser fallback queue (page model unreadable): 0 (browser_todo.json)

Alerts this run:

- WATCH NEW_INPUT `basith007/model-ensembling` score None uses never-seen ['dataset:basith007/coatnet-features']
- WATCH SCORE_APPEARED `eeshanvaghjiani2004/knee-oct15-224m` public score 0.956 (v1)
- WATCH NEW_INPUT `sameerk2004/rsna-knee-sil-d5d-goldroute` score None uses never-seen ['kernel:sameerk2004/rsna-knee-sil-d5b-dual0950']
- WATCH NEW_INPUT `karttikjangid05/rsna-knee-which-llm-labels-can-you-trust` score None uses never-seen ['dataset:yunusgmsoy/rsna-knee-llm-report-labels', 'dataset:laymond/rsna-knee-abnormality-qwen3-8b-weak-labels', 'dataset:riadmohamed42/jev-knee-labels', 'dataset:karttikjangid05/rsna-knee-public-llm-label-sets-compared']

Top 15 public notebooks by page score (inputs: R7 = heliosli/rsna-knee-compact-r7cnxfsss, OF = heliosli/rsna-orthofoundation-fold0-epoch7-20261004, NS = nartaa/rsna-knee-publication-swa-weights-20261007, R6 = heliosli/rsna-knee-compact-r6cnxfsss, CV1 = heliosli/rsna-knee-compact-v1, GP = goodpjw2008/rsna-knee-2-5d-convnext-reader):

| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |
|---|---|---|---|---|---|---|
| 1 | `heliosli/this-one-is-old-version-0-957-new-one-you-can-use` | 0.957 | v1 / v2 | public NS+OF+heliosli/rsna-knee-r6-frozen-fe055406 | heliosli/rsna-knee-r6-frozen-fe055406 | 2026-10-14T18:13 |
| 2 | `bliverrigpdoge/r7-raptor-knee-centred-convnext` | 0.957 | v1 / v1 | NOT all public: ?/private-data-source+NS+OF+R7 | - | 2026-10-14T08:26 |
| 3 | `heliosli/0-9576-latest-public-dataset-score-in-one-click` | 0.957 | v1 / v1 | public NS+OF+heliosli/rsna-knee-r6-frozen-fe055406 | heliosli/rsna-knee-r6-frozen-fe055406 | 2026-10-14T12:21 |
| 4 | `evgendvorkin/rsna-versia-7` | 0.956 | v22 / v22 | public NS+OF+R7 | - | 2026-10-13T14:02 |
| 5 | `heliosli/rsna-knee-blend-gold-old-dataset` | 0.956 | v1 / v2 | public NS+OF+R6 | R6 | 2026-10-11T14:16 |
| 6 | `heliosli/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-11T20:01 |
| 7 | `goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` | 0.956 | v17 / v20 | public GP+NS+OF+R7 | - | 2026-10-15T00:09 |
| 8 | `degnonguidi/rsna-knee-abnormality-detection` | 0.956 | v10 / v10 | public NS+OF+R7 | - | 2026-10-13T08:55 |
| 9 | `jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T06:00 |
| 10 | `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T08:39 |
| 11 | `lavinwins/rsna-gold-medal` | 0.956 | v1 / v4 | public NS+OF+R7 | - | 2026-10-15T01:02 |
| 12 | `mayankraaz/rsna-knee-abnormality-pipeline` | 0.956 | v5 / v7 | public NS+OF+R7 | - | 2026-10-14T11:40 |
| 13 | `mkhamzanov/rsna-b1-ctrl` | 0.956 | v1 / v1 | public GP+NS+OF+R7 | - | 2026-10-13T01:19 |
| 14 | `varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-14T10:52 |
| 15 | `medvax/rsna-knee-r7-source-trial` | 0.956 | v1 / v2 | public NS+OF+R7 | - | 2026-10-13T15:18 |

Pending free reads (no public score yet; ran in the last 72 h; all inputs public, R7 bundle among them; a score
appearing raises SCORE_APPEARED):

- `beraterolelk/0-957-sota-rsna-knee-multi-scale-foundation` v1, ran 2026-10-15T09:39 UTC, inputs NS+OF+R7+beraterolelk/istanbul-real-estate-2024-anonymized+bliverrigpdoge/rsna-knee-leg-r9all
- `rjings/rsna-knee-target-local-router-20261015` v2, ran 2026-10-15T03:59 UTC, inputs NS+OF+R7
- `oscarcarreoserpa/rsna-knee-one-unused-checkpoint-lifts-the-0-956` v1, ran 2026-10-15T03:03 UTC, inputs NS+OF+R7
- `piyush474/rsna-knee-0-956-raptor-convnext-r7-ensemble` v1, ran 2026-10-13T09:02 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-heliosli-r7cnxfsss-0-5-blend` v4, ran 2026-10-13T06:42 UTC, inputs NS+OF+R7
- `z1358028281/0-957-latest-dataset-score-in-one-click` v2, ran 2026-10-13T02:28 UTC, inputs NS+OF+R7

Watched datasets (heliosli + nartaa lists, RESEARCH_960 baseline set; knee-related only, all in watch_state.json):

| dataset | version | last updated | bytes | via |
|---|---|---|---|---|
| `dreaddevelopment/raptor-knee-widedense` | v2 | 2026-08-23T04:12 | 543320725 | GetDatasetBasics (anonymous) |
| `heliosli/rsna-knee-compact-r6cnxfsss` | v2 | 2026-10-11T13:18 | 823081993 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-r7cnxfsss` | v1 | 2026-10-11T19:59 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-v1` | v13 | 2026-10-10T19:50 | 823082549 | datasets list --user heliosli |
| `heliosli/rsna-knee-r6-frozen-fe055406` | v2 | 2026-10-14T18:12 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | v1 | 2026-10-04T07:54 | 1343091054 | datasets list --user heliosli |
| `jessky/rsna-knee-local-cv-cnx50` | v1 | 2026-10-12T01:15 | 931795323 | GetDatasetBasics (anonymous) |
| `nartaa/rsna-knee-hpo-assets` | v9 | 2026-09-25T12:01 | 1079732 | datasets list --user nartaa |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | v1 | 2026-10-07T19:26 | 543312838 | datasets list --user nartaa |
| `nartaa/rsna-knee-s6fall-weights` | v3 | 2026-10-14T19:08 | 185910765 | datasets list --user nartaa |
| `nartaa/rsna24-helpers` | v8 | 2024-09-04T17:37 | 519217604 | datasets list --user nartaa |
| `nartaa/rsna24-npy1` | v1 | 2024-08-09T17:47 | 10975586069 | datasets list --user nartaa |
| `nartaa/rsna24-npy2` | v1 | 2024-08-07T18:36 | 11052914944 | datasets list --user nartaa |
| `nartaa/rsna24-npy3` | v1 | 2024-08-07T18:37 | 11004266683 | datasets list --user nartaa |
| `nartaa/rsna24-npy4` | v1 | 2024-08-07T18:37 | 3379991785 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models` | v16 | 2024-09-10T08:38 | 259425853 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models-testbed-000` | v1 | 2024-09-12T09:47 | 86077478 | datasets list --user nartaa |
| `nartaa/rsna24-train-images` | v2 | 2024-09-12T07:27 | 5877188280 | datasets list --user nartaa |

Errors (2): kvm mizeroluckygall/knee-mri-raptor-384-224-r7-blend http 404; kvm raihanjs/rsna-knee-fork-0957-heliosli http 404

## Run 2026-10-15T10:44:45Z (10-15 16:14 IST)

- listed 382 notebooks (score pages + recency pages + key authors), known 470, read 110 (new 2), pulled -m 2, web calls 184 (errors 0), 187 s.
- max public score among public notebooks: **0.957**; score bands (count of notebooks): {'0.957': 3, '0.956': 41, '0.955': 4, '0.954': 29}; list sort inversions vs page scores: 27.
- RELEASE_CANDIDATE (score >= 0.957 and all inputs public): **heliosli/0-9576-latest-public-dataset-score-in-one-click, heliosli/this-one-is-old-version-0-957-new-one-you-can-use**
- browser fallback queue (page model unreadable): 0 (browser_todo.json)

Alerts this run:

- WATCH SCORE_APPEARED `crystalwong01/rsna-knee-helios-r6-frozen-blend-public-run` public score 0.956 (v3)

Top 15 public notebooks by page score (inputs: R7 = heliosli/rsna-knee-compact-r7cnxfsss, OF = heliosli/rsna-orthofoundation-fold0-epoch7-20261004, NS = nartaa/rsna-knee-publication-swa-weights-20261007, R6 = heliosli/rsna-knee-compact-r6cnxfsss, CV1 = heliosli/rsna-knee-compact-v1, GP = goodpjw2008/rsna-knee-2-5d-convnext-reader):

| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |
|---|---|---|---|---|---|---|
| 1 | `heliosli/this-one-is-old-version-0-957-new-one-you-can-use` | 0.957 | v1 / v2 | public NS+OF+heliosli/rsna-knee-r6-frozen-fe055406 | heliosli/rsna-knee-r6-frozen-fe055406 | 2026-10-14T18:13 |
| 2 | `bliverrigpdoge/r7-raptor-knee-centred-convnext` | 0.957 | v1 / v1 | NOT all public: ?/private-data-source+NS+OF+R7 | - | 2026-10-14T08:26 |
| 3 | `heliosli/0-9576-latest-public-dataset-score-in-one-click` | 0.957 | v1 / v1 | public NS+OF+heliosli/rsna-knee-r6-frozen-fe055406 | heliosli/rsna-knee-r6-frozen-fe055406 | 2026-10-14T12:21 |
| 4 | `evgendvorkin/rsna-versia-7` | 0.956 | v22 / v22 | public NS+OF+R7 | - | 2026-10-13T14:02 |
| 5 | `heliosli/rsna-knee-blend-gold-old-dataset` | 0.956 | v1 / v2 | public NS+OF+R6 | R6 | 2026-10-11T14:16 |
| 6 | `heliosli/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-11T20:01 |
| 7 | `goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` | 0.956 | v17 / v20 | public GP+NS+OF+R7 | - | 2026-10-15T00:09 |
| 8 | `degnonguidi/rsna-knee-abnormality-detection` | 0.956 | v10 / v10 | public NS+OF+R7 | - | 2026-10-13T08:55 |
| 9 | `jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T06:00 |
| 10 | `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T08:39 |
| 11 | `lavinwins/rsna-gold-medal` | 0.956 | v1 / v4 | public NS+OF+R7 | - | 2026-10-15T01:02 |
| 12 | `mayankraaz/rsna-knee-abnormality-pipeline` | 0.956 | v5 / v7 | public NS+OF+R7 | - | 2026-10-14T11:40 |
| 13 | `mkhamzanov/rsna-b1-ctrl` | 0.956 | v1 / v1 | public GP+NS+OF+R7 | - | 2026-10-13T01:19 |
| 14 | `varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-14T10:52 |
| 15 | `medvax/rsna-knee-r7-source-trial` | 0.956 | v1 / v2 | public NS+OF+R7 | - | 2026-10-13T15:18 |

Pending free reads (no public score yet; ran in the last 72 h; all inputs public, R7 bundle among them; a score
appearing raises SCORE_APPEARED):

- `beraterolelk/0-957-sota-rsna-knee-multi-scale-foundation` v1, ran 2026-10-15T09:39 UTC, inputs NS+OF+R7+beraterolelk/istanbul-real-estate-2024-anonymized+bliverrigpdoge/rsna-knee-leg-r9all
- `rjings/rsna-knee-target-local-router-20261015` v2, ran 2026-10-15T03:59 UTC, inputs NS+OF+R7
- `oscarcarreoserpa/rsna-knee-one-unused-checkpoint-lifts-the-0-956` v1, ran 2026-10-15T03:03 UTC, inputs NS+OF+R7
- `piyush474/rsna-knee-0-956-raptor-convnext-r7-ensemble` v1, ran 2026-10-13T09:02 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-heliosli-r7cnxfsss-0-5-blend` v4, ran 2026-10-13T06:42 UTC, inputs NS+OF+R7
- `z1358028281/0-957-latest-dataset-score-in-one-click` v2, ran 2026-10-13T02:28 UTC, inputs NS+OF+R7

Watched datasets (heliosli + nartaa lists, RESEARCH_960 baseline set; knee-related only, all in watch_state.json):

| dataset | version | last updated | bytes | via |
|---|---|---|---|---|
| `dreaddevelopment/raptor-knee-widedense` | v2 | 2026-08-23T04:12 | 543320725 | GetDatasetBasics (anonymous) |
| `heliosli/rsna-knee-compact-r6cnxfsss` | v2 | 2026-10-11T13:18 | 823081993 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-r7cnxfsss` | v1 | 2026-10-11T19:59 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-v1` | v13 | 2026-10-10T19:50 | 823082549 | datasets list --user heliosli |
| `heliosli/rsna-knee-r6-frozen-fe055406` | v2 | 2026-10-14T18:12 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | v1 | 2026-10-04T07:54 | 1343091054 | datasets list --user heliosli |
| `jessky/rsna-knee-local-cv-cnx50` | v1 | 2026-10-12T01:15 | 931795323 | GetDatasetBasics (anonymous) |
| `nartaa/rsna-knee-hpo-assets` | v9 | 2026-09-25T12:01 | 1079732 | datasets list --user nartaa |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | v1 | 2026-10-07T19:26 | 543312838 | datasets list --user nartaa |
| `nartaa/rsna-knee-s6fall-weights` | v3 | 2026-10-14T19:08 | 185910765 | datasets list --user nartaa |
| `nartaa/rsna24-helpers` | v8 | 2024-09-04T17:37 | 519217604 | datasets list --user nartaa |
| `nartaa/rsna24-npy1` | v1 | 2024-08-09T17:47 | 10975586069 | datasets list --user nartaa |
| `nartaa/rsna24-npy2` | v1 | 2024-08-07T18:36 | 11052914944 | datasets list --user nartaa |
| `nartaa/rsna24-npy3` | v1 | 2024-08-07T18:37 | 11004266683 | datasets list --user nartaa |
| `nartaa/rsna24-npy4` | v1 | 2024-08-07T18:37 | 3379991785 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models` | v16 | 2024-09-10T08:38 | 259425853 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models-testbed-000` | v1 | 2024-09-12T09:47 | 86077478 | datasets list --user nartaa |
| `nartaa/rsna24-train-images` | v2 | 2024-09-12T07:27 | 5877188280 | datasets list --user nartaa |

Errors (2): kvm mizeroluckygall/knee-mri-raptor-384-224-r7-blend http 404; kvm raihanjs/rsna-knee-fork-0957-heliosli http 404

## Run 2026-10-15T12:17:52Z (10-15 17:47 IST)

- listed 384 notebooks (score pages + recency pages + key authors), known 475, read 111 (new 5), pulled -m 2, web calls 186 (errors 0), 197 s.
- max public score among public notebooks: **0.957**; score bands (count of notebooks): {'0.957': 3, '0.956': 41, '0.955': 4, '0.954': 29}; list sort inversions vs page scores: 28.
- RELEASE_CANDIDATE (score >= 0.957 and all inputs public): **heliosli/0-9576-latest-public-dataset-score-in-one-click, heliosli/this-one-is-old-version-0-957-new-one-you-can-use**
- browser fallback queue (page model unreadable): 0 (browser_todo.json)

Top 15 public notebooks by page score (inputs: R7 = heliosli/rsna-knee-compact-r7cnxfsss, OF = heliosli/rsna-orthofoundation-fold0-epoch7-20261004, NS = nartaa/rsna-knee-publication-swa-weights-20261007, R6 = heliosli/rsna-knee-compact-r6cnxfsss, CV1 = heliosli/rsna-knee-compact-v1, GP = goodpjw2008/rsna-knee-2-5d-convnext-reader):

| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |
|---|---|---|---|---|---|---|
| 1 | `heliosli/this-one-is-old-version-0-957-new-one-you-can-use` | 0.957 | v1 / v2 | public NS+OF+heliosli/rsna-knee-r6-frozen-fe055406 | heliosli/rsna-knee-r6-frozen-fe055406 | 2026-10-14T18:13 |
| 2 | `bliverrigpdoge/r7-raptor-knee-centred-convnext` | 0.957 | v1 / v1 | NOT all public: ?/private-data-source+NS+OF+R7 | - | 2026-10-14T08:26 |
| 3 | `heliosli/0-9576-latest-public-dataset-score-in-one-click` | 0.957 | v1 / v1 | public NS+OF+heliosli/rsna-knee-r6-frozen-fe055406 | heliosli/rsna-knee-r6-frozen-fe055406 | 2026-10-14T12:21 |
| 4 | `evgendvorkin/rsna-versia-7` | 0.956 | v22 / v22 | public NS+OF+R7 | - | 2026-10-13T14:02 |
| 5 | `heliosli/rsna-knee-blend-gold-old-dataset` | 0.956 | v1 / v2 | public NS+OF+R6 | R6 | 2026-10-11T14:16 |
| 6 | `heliosli/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-11T20:01 |
| 7 | `goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` | 0.956 | v17 / v20 | public GP+NS+OF+R7 | - | 2026-10-15T00:09 |
| 8 | `degnonguidi/rsna-knee-abnormality-detection` | 0.956 | v10 / v10 | public NS+OF+R7 | - | 2026-10-13T08:55 |
| 9 | `jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T06:00 |
| 10 | `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T08:39 |
| 11 | `lavinwins/rsna-gold-medal` | 0.956 | v1 / v4 | public NS+OF+R7 | - | 2026-10-15T01:02 |
| 12 | `mayankraaz/rsna-knee-abnormality-pipeline` | 0.956 | v5 / v7 | public NS+OF+R7 | - | 2026-10-14T11:40 |
| 13 | `mkhamzanov/rsna-b1-ctrl` | 0.956 | v1 / v1 | public GP+NS+OF+R7 | - | 2026-10-13T01:19 |
| 14 | `varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-14T10:52 |
| 15 | `yuvrajlamba01/rsna-knee-research-optimized-0-957` | 0.956 | v1 / v3 | public NS+OF+R7+bliverrigpdoge/rsna-knee-leg-r9all | - | 2026-10-15T07:35 |

Pending free reads (no public score yet; ran in the last 72 h; all inputs public, R7 bundle among them; a score
appearing raises SCORE_APPEARED):

- `dennis6886/rsna-knee-branch-playbook-20261015` v1, ran 2026-10-15T10:56 UTC, inputs CV1+GP+NS+OF+R7
- `sujanmajhisuzan/rsna-knee-apex-grandmaster-v14-dualraptor` v1, ran 2026-10-15T10:49 UTC, inputs NS+OF+R7
- `beraterolelk/0-957-sota-rsna-knee-multi-scale-foundation` v1, ran 2026-10-15T09:39 UTC, inputs NS+OF+R7+beraterolelk/istanbul-real-estate-2024-anonymized+bliverrigpdoge/rsna-knee-leg-r9all
- `rjings/rsna-knee-target-local-router-20261015` v2, ran 2026-10-15T03:59 UTC, inputs NS+OF+R7
- `oscarcarreoserpa/rsna-knee-one-unused-checkpoint-lifts-the-0-956` v1, ran 2026-10-15T03:03 UTC, inputs NS+OF+R7
- `piyush474/rsna-knee-0-956-raptor-convnext-r7-ensemble` v1, ran 2026-10-13T09:02 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-heliosli-r7cnxfsss-0-5-blend` v4, ran 2026-10-13T06:42 UTC, inputs NS+OF+R7
- `z1358028281/0-957-latest-dataset-score-in-one-click` v2, ran 2026-10-13T02:28 UTC, inputs NS+OF+R7

Watched datasets (heliosli + nartaa lists, RESEARCH_960 baseline set; knee-related only, all in watch_state.json):

| dataset | version | last updated | bytes | via |
|---|---|---|---|---|
| `dreaddevelopment/raptor-knee-widedense` | v2 | 2026-08-23T04:12 | 543320725 | GetDatasetBasics (anonymous) |
| `heliosli/rsna-knee-compact-r6cnxfsss` | v2 | 2026-10-11T13:18 | 823081993 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-r7cnxfsss` | v1 | 2026-10-11T19:59 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-v1` | v13 | 2026-10-10T19:50 | 823082549 | datasets list --user heliosli |
| `heliosli/rsna-knee-r6-frozen-fe055406` | v2 | 2026-10-14T18:12 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | v1 | 2026-10-04T07:54 | 1343091054 | datasets list --user heliosli |
| `jessky/rsna-knee-local-cv-cnx50` | v1 | 2026-10-12T01:15 | 931795323 | GetDatasetBasics (anonymous) |
| `nartaa/rsna-knee-hpo-assets` | v9 | 2026-09-25T12:01 | 1079732 | datasets list --user nartaa |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | v1 | 2026-10-07T19:26 | 543312838 | datasets list --user nartaa |
| `nartaa/rsna-knee-s6fall-weights` | v3 | 2026-10-14T19:08 | 185910765 | datasets list --user nartaa |
| `nartaa/rsna24-helpers` | v8 | 2024-09-04T17:37 | 519217604 | datasets list --user nartaa |
| `nartaa/rsna24-npy1` | v1 | 2024-08-09T17:47 | 10975586069 | datasets list --user nartaa |
| `nartaa/rsna24-npy2` | v1 | 2024-08-07T18:36 | 11052914944 | datasets list --user nartaa |
| `nartaa/rsna24-npy3` | v1 | 2024-08-07T18:37 | 11004266683 | datasets list --user nartaa |
| `nartaa/rsna24-npy4` | v1 | 2024-08-07T18:37 | 3379991785 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models` | v16 | 2024-09-10T08:38 | 259425853 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models-testbed-000` | v1 | 2024-09-12T09:47 | 86077478 | datasets list --user nartaa |
| `nartaa/rsna24-train-images` | v2 | 2024-09-12T07:27 | 5877188280 | datasets list --user nartaa |

Errors (2): kvm mizeroluckygall/knee-mri-raptor-384-224-r7-blend http 404; kvm raihanjs/rsna-knee-fork-0957-heliosli http 404

## Run 2026-10-15T13:51:10Z (10-15 19:21 IST)

- listed 383 notebooks (score pages + recency pages + key authors), known 481, read 113 (new 6), pulled -m 6, web calls 190 (errors 0), 220 s.
- max public score among public notebooks: **0.957**; score bands (count of notebooks): {'0.957': 4, '0.956': 42, '0.955': 4, '0.954': 29}; list sort inversions vs page scores: 28.
- RELEASE_CANDIDATE (score >= 0.957 and all inputs public): **heliosli/this-one-is-old-version-0-957-new-one-you-can-use, heliosli/this-is-new-version-0-9576-in-one-click, leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl**
- browser fallback queue (page model unreadable): 0 (browser_todo.json)

Alerts this run:

- FLAG RELEASE_CANDIDATE `heliosli/this-is-new-version-0-9576-in-one-click` score 0.957 v1 (current v1), inputs ALL PUBLIC, DRIFT {"heliosli/rsna-knee-r6-frozen-fe055406": {"pinned_version_id": 20611930, "current_version_id": 20731557, "current_version": 2}}; next: python session_2026-10-15/watch/swap_builder.py heliosli/this-is-new-version-0-9576-in-one-click/1 --name rsna-sub-rel-heliosli-this-is-new-version-0
- WATCH KEY_AUTHOR_NEW_NOTEBOOK `heliosli/this-is-new-version-0-9576-in-one-click` score 0.957, inputs {'competition:rsna-knee-abnormality-detection': 'ours', 'dataset:heliosli/rsna-orthofoundation-fold0-epoch7-20261004': 'public', 'dataset:nartaa/rsna-knee-publication-swa-weights-20261007': 'public', 'dataset:heliosli/rsna-knee-r6-frozen-fe055406': 'public'}
- FLAG RELEASE_CANDIDATE `leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl` score 0.957 v2 (current v10), inputs ALL PUBLIC; next: python session_2026-10-15/watch/swap_builder.py leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl/2 --name rsna-sub-rel-leoprovorov-rsna-knee-0-957-ex
- WATCH NEW_INPUT `leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl` score 0.957 uses never-seen ['dataset:nartaa/rsna-knee-s6fall-weights']
- WATCH SCORE_APPEARED `yuvrajlamba01/rsna-knee-research-optimized` public score 0.956 (v3)

Top 15 public notebooks by page score (inputs: R7 = heliosli/rsna-knee-compact-r7cnxfsss, OF = heliosli/rsna-orthofoundation-fold0-epoch7-20261004, NS = nartaa/rsna-knee-publication-swa-weights-20261007, R6 = heliosli/rsna-knee-compact-r6cnxfsss, CV1 = heliosli/rsna-knee-compact-v1, GP = goodpjw2008/rsna-knee-2-5d-convnext-reader):

| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |
|---|---|---|---|---|---|---|
| 1 | `heliosli/this-one-is-old-version-0-957-new-one-you-can-use` | 0.957 | v1 / v2 | public NS+OF+heliosli/rsna-knee-r6-frozen-fe055406 | heliosli/rsna-knee-r6-frozen-fe055406 | 2026-10-14T18:13 |
| 2 | `bliverrigpdoge/r7-raptor-knee-centred-convnext` | 0.957 | v1 / v1 | NOT all public: ?/private-data-source+NS+OF+R7 | - | 2026-10-14T08:26 |
| 3 | `heliosli/this-is-new-version-0-9576-in-one-click` | 0.957 | v1 / v1 | public NS+OF+heliosli/rsna-knee-r6-frozen-fe055406 | heliosli/rsna-knee-r6-frozen-fe055406 | 2026-10-14T12:21 |
| 4 | `leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl` | 0.957 | v2 / v10 | public NS+OF+R7+nartaa/rsna-knee-s6fall-weights | - | 2026-10-15T12:49 |
| 5 | `evgendvorkin/rsna-versia-7` | 0.956 | v22 / v22 | public NS+OF+R7 | - | 2026-10-13T14:02 |
| 6 | `heliosli/rsna-knee-blend-gold-old-dataset` | 0.956 | v1 / v2 | public NS+OF+R6 | R6 | 2026-10-11T14:16 |
| 7 | `heliosli/0-957-latest-dataset-score-in-one-click` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-11T20:01 |
| 8 | `goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` | 0.956 | v17 / v20 | public GP+NS+OF+R7 | - | 2026-10-15T00:09 |
| 9 | `degnonguidi/rsna-knee-abnormality-detection` | 0.956 | v10 / v10 | public NS+OF+R7 | - | 2026-10-13T08:55 |
| 10 | `jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T06:00 |
| 11 | `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-13T08:39 |
| 12 | `lavinwins/rsna-gold-medal` | 0.956 | v1 / v4 | public NS+OF+R7 | - | 2026-10-15T01:02 |
| 13 | `mayankraaz/rsna-knee-abnormality-pipeline` | 0.956 | v5 / v7 | public NS+OF+R7 | - | 2026-10-14T11:40 |
| 14 | `varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up` | 0.956 | v1 / v1 | public NS+OF+R7 | - | 2026-10-14T10:52 |
| 15 | `mkhamzanov/rsna-b1-ctrl` | 0.956 | v1 / v1 | public GP+NS+OF+R7 | - | 2026-10-13T01:19 |

Pending free reads (no public score yet; ran in the last 72 h; all inputs public, R7 bundle among them; a score
appearing raises SCORE_APPEARED):

- `dennis6886/rsna-knee-branch-playbook-20261015` v1, ran 2026-10-15T10:56 UTC, inputs CV1+GP+NS+OF+R7
- `sujanmajhisuzan/rsna-knee-apex-grandmaster-v14-dualraptor` v1, ran 2026-10-15T10:49 UTC, inputs NS+OF+R7
- `beraterolelk/0-957-sota-rsna-knee-multi-scale-foundation` v1, ran 2026-10-15T09:39 UTC, inputs NS+OF+R7+beraterolelk/istanbul-real-estate-2024-anonymized+bliverrigpdoge/rsna-knee-leg-r9all
- `rjings/rsna-knee-target-local-router-20261015` v2, ran 2026-10-15T03:59 UTC, inputs NS+OF+R7
- `oscarcarreoserpa/rsna-knee-one-unused-checkpoint-lifts-the-0-956` v1, ran 2026-10-15T03:03 UTC, inputs NS+OF+R7
- `piyush474/rsna-knee-0-956-raptor-convnext-r7-ensemble` v1, ran 2026-10-13T09:02 UTC, inputs NS+OF+R7
- `sushanthtiruvaipati/rsna-knee-heliosli-r7cnxfsss-0-5-blend` v4, ran 2026-10-13T06:42 UTC, inputs NS+OF+R7
- `z1358028281/0-957-latest-dataset-score-in-one-click` v2, ran 2026-10-13T02:28 UTC, inputs NS+OF+R7

Watched datasets (heliosli + nartaa lists, RESEARCH_960 baseline set; knee-related only, all in watch_state.json):

| dataset | version | last updated | bytes | via |
|---|---|---|---|---|
| `dreaddevelopment/raptor-knee-widedense` | v2 | 2026-08-23T04:12 | 543320725 | GetDatasetBasics (anonymous) |
| `heliosli/rsna-knee-compact-r6cnxfsss` | v2 | 2026-10-11T13:18 | 823081993 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-r7cnxfsss` | v1 | 2026-10-11T19:59 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-knee-compact-v1` | v13 | 2026-10-10T19:50 | 823082549 | datasets list --user heliosli |
| `heliosli/rsna-knee-r6-frozen-fe055406` | v2 | 2026-10-14T18:12 | 823095598 | datasets list --user heliosli |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | v1 | 2026-10-04T07:54 | 1343091054 | datasets list --user heliosli |
| `jessky/rsna-knee-local-cv-cnx50` | v1 | 2026-10-12T01:15 | 931795323 | GetDatasetBasics (anonymous) |
| `nartaa/rsna-knee-hpo-assets` | v9 | 2026-09-25T12:01 | 1079732 | datasets list --user nartaa |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | v1 | 2026-10-07T19:26 | 543312838 | datasets list --user nartaa |
| `nartaa/rsna-knee-s6fall-weights` | v3 | 2026-10-14T19:08 | 185910765 | datasets list --user nartaa |
| `nartaa/rsna24-helpers` | v8 | 2024-09-04T17:37 | 519217604 | datasets list --user nartaa |
| `nartaa/rsna24-npy1` | v1 | 2024-08-09T17:47 | 10975586069 | datasets list --user nartaa |
| `nartaa/rsna24-npy2` | v1 | 2024-08-07T18:36 | 11052914944 | datasets list --user nartaa |
| `nartaa/rsna24-npy3` | v1 | 2024-08-07T18:37 | 11004266683 | datasets list --user nartaa |
| `nartaa/rsna24-npy4` | v1 | 2024-08-07T18:37 | 3379991785 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models` | v16 | 2024-09-10T08:38 | 259425853 | datasets list --user nartaa |
| `nartaa/rsna24-starter-models-testbed-000` | v1 | 2024-09-12T09:47 | 86077478 | datasets list --user nartaa |
| `nartaa/rsna24-train-images` | v2 | 2024-09-12T07:27 | 5877188280 | datasets list --user nartaa |

Errors (4): pull heliosli/0-9576-latest-public-dataset-score-in-one-click: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernel; kvm mizeroluckygall/knee-mri-raptor-384-224-r7-blend http 404; kvm raihanjs/rsna-knee-fork-0957-heliosli http 404; pull yuvrajlamba01/rsna-knee-research-optimized-0-957: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernel

