# forum_lb.md: forum + LB read, 10-20 17:06-17:10 UTC (22:36-22:40 IST). READ-ONLY.

**SUBMIT-SAFE: YES.** Nothing new changes rules, data, metric, deadline or final-selection mechanics. The 10-21 slate
(r6pw20, pw25, optional lomm05) and the current picks are unaffected.
**ACTION: none required.** One FYI: a rank-50 participant reports Kaggle dataset uploads failing for ~2 h (751027,
15:17 UTC, unanswered). Our 10-21 kernels need no upload (MEASURED: `val/kernels/kernel_v224r6c4pw25j`,
`val/kernels/kernel_v224r6flips6fpw20j` and `night/c12_kernel_v224r6c4pw30lomm05j` all attach the same 5 existing
datasets as pw30). Still, check the push rc / gate line at the 05:30 IST run and allow a retry if Kaggle is degraded.

Nothing was posted, voted, followed, submitted, pushed, committed or downloaded. No PREREG / HANDOFF / PROGRESS /
a20 / a21 / val / night / scheduled-task edits. No OAI data touched. Tags: MEASURED = fetched or computed now (file
named) / CLAIMED = a poster says so / MODEL = our arithmetic / GUESS = judgement.

## 1. Brief

1. Host / staff: **no new post** since Maria Cruz's non-ruling (747455, 10-19 22:46 UTC). MEASURED: every thread
   with a post after 04:00 UTC was fetched (9 + 2 new) and none has a HOST / ADMIN / STAFF author after 04:00; topic
   lists show host last-commenters only on 733826, 736301, 749038, 744056, 747455, all dated 10-19 22:46 or earlier.
2. Rules / metadata (MEASURED `raw/comp.json`, diffed vs the 04:17 read): only counters changed. Deadline
   2026-10-22T23:59Z, numScoredSubmissions 2, maxDailySubmissions 5, leaderboardPercentage 30, truncation 3 decimals.
   Teams 6,249 -> 6,275; total submissions 147,067 -> 149,824 (+2,757 in ~12.8 h).
3. Private board: nothing. The LB API returns only `publicLeaderboard` + `teams` (MEASURED); our privateScore column
   is blank for all subs (MEASURED, `kaggle competitions submissions`). One new thread asks how private will look
   given few expert labels vs derived labels (751021, DeepUnderstanding r2506, 14:50 UTC, 0 replies). No host
   statement on private scoring, site split or label regime.
4. Our row: **r141, .959, best sid 57425970 (pw30)**, SILVER. r140 at 13:20 UTC (18:50 IST). 4th in the .959 band
   (3 above: imainjob, Civitasmass, Khayrul Islam, all unchanged since 13:20); 70 .959 teams below us.
5. Bands now: .969 top unchanged; .960+ = 137 (r1-137, was 136); .959 = 74 (r138-211, was 72); .958 = 1,399
   (r212-1610). Medals: Gold r1-22, Silver r23-313, Bronze r314-627; silver line displays .958. Margin to silver 172.

## 2. Coverage (MEASURED)

- `raw/comp.json` (GetCompetition 17:06 UTC); `raw/l_TOPIC_LIST_SORT_BY_{RECENT,NEW,HOT,TOP}.json` (3 pages each,
  108 unique topics) via `session_2026-10-16/forum/fr.py`; diff vs `session_2026-10-20/forum/raw/` (04:17 UTC) by
  `diff_topics.py`. One old topic fell off page 3 (745979, last post 10-05): pagination only.
- Messages via `session_2026-10-16/forum/msgs.py` into `raw/msgs/` (collapsed replies via batch): all threads active
  since 04:00 UTC (751027, 751021, 750334, 748181, 747658, 745861, 746792, 735304, 748151) plus watch list 747455,
  734681, 744519, 737155, 743416, 744056, 730709, 750899, 750889, 750790, 745949, 750752. `new_msgs.py` prints every
  post since 04:00 UTC plus all host posts.
- LB: `lb/api_lb_full_1020T_private.json` (raw, teams + members) and `lb/api_lb_1020T_private.json` (trimmed), fetched
  17:08:16 UTC by `fetch_lb_full.py`; analysis `lb_bands.py` -> `lb_bands.out` against 10-19 20:17, 10-20 04:17,
  05:53 and 13:20 UTC snapshots.

## 3. Posts since 10-20 04:00 UTC (MEASURED times, CLAIMED content, live rank)

| UTC | thread | who (rank) | content (paraphrase) | use |
|---|---|---|---|---|
| 06:44, 07:57 | 748151 | Yorix.um (r1327), Navneet | praise for OmerZalman's label dataset | none |
| 11:25, 13:15, 14:39 | 750334 | FOYSAL (r326), Lavin Wins (r1281), Syed Asad Ali (r587) | where are the 98-99 % label datasets; "on his profile" | none (label tables, training inputs only) |
| 13:15, 14:02, 14:10 | 748181 | Lavin Wins, Dread Development (r1060) | complaints about copied public notebooks winning medals | none |
| 13:16-13:18 | 735304, 746792, 745861, 747658 | Lavin Wins (r1281) | one-liners ("got 0.954", praise, LLM cost) | none |
| 14:50 | 751021 (NEW) | DeepUnderstanding (r2506) | asks what private will look like given few expert labels | 0 replies; no info |
| 15:17 | 751027 (NEW) | Optimo (r50, .963) | cannot upload datasets for ~2 h, asks if a Google Cloud outage; tags a Kaggle staffer | 0 replies; see ACTION |

No technical content from anyone above us. No hint on OA labels, compartment coupling, test sites or missing planes.
Context for 751027 (MEASURED, status.cloud.google.com via WebFetch): one closed multi-service incident in
europe-west2 on 10-20 (~7.9 h). Whether it touched Kaggle is unknown (GUESS: low impact on code-only kernel pushes).

## 4. Watch list

| item | thread | last post | status |
|---|---|---|---|
| host posts | all | 747455, 10-19 22:46 (staff non-ruling) | nothing new |
| rules / deadline / selection | comp metadata | n/a | unchanged (sec 1) |
| final selection / rerun rules | none | n/a | no thread |
| private scoring | 751021 | 10-20 14:50 | question only, 0 replies |
| site split | 734681 | 08-14 14:08 | no host answer, nothing new |
| missing planes | 744519 | 10-06 02:17 | no host answer, nothing new |
| Synovitis rubric | 737155 | 08-23 20:37 | no reply at all |
| external data | 743416, 747455, 750889, 750899, 750752 | 743416 10-18 23:42; 750899 10-20 03:56 | no new post after the staff non-ruling; status GRAY, Eesh's call (unchanged) |
| OA / label regime | 745949 | 10-19 15:56 | nothing new |

## 5. LB snapshot 17:08 UTC (MEASURED, `lb_bands.out`)

| snapshot (UTC) | teams | .960+ | .959 band | .958 band | us | in-band above us |
|---|---|---|---|---|---|---|
| 10-20 04:17 | 6,249 | 127 | 66 (r128-193) | 1,337 (r194-1530) | r217 .958 (c4) | 23 |
| 10-20 05:53 | 6,253 | 130 | 66 (r131-196) | 1,341 | r135 .959 (eoa20) | 4 |
| 10-20 13:20 | 6,270 | 136 | 72 (r137-208) | 1,382 | r140 .959 (pw30) | 3 |
| **10-20 17:08** | **6,276** | **137** | **74 (r138-211)** | **1,399 (r212-1610)** | **r141 .959 (pw30)** | **3** |

Top bands now: .969 1 | .968 4 (r2-5) | .967 6 (r6-11) | .966 5 | .965 5 (r17-21) | .964 13 (r22-34) | .963 16 |
.962 16 | .961 27 (r67-93) | .960 44 (r94-137).

Movement since 13:20 UTC (18:50 IST):
- One team passed us: OsakanaFish .959 (r142, below us) -> .961 (r84). Nobody fell below us.
- Three new .959 crossers, all ranked below us (jp_nerdery .954 -> .959 r199; N0.o6 .958 -> .959 r200; Vasily
  .958 -> .959 r203). Within a display band rank follows the unrounded score, so their true scores sit below pw30's.
- Moves at the top: joinT fusion .967 -> .968; Claudio Verdu Ruiz .965 -> .967; tomoon33 .964 -> .965; Ben Pepper
  .963 -> .964; Timshel .960 -> .961.
- Activity: 497 teams submitted since 13:20 (22 of them in .959, 19 in .960, 291 in .958); 6 new teams.
- Since 04:17 UTC, 18 teams crossed into .959+ (incl. us); notable: Chris Deotte .958 -> .959 (r189), Anay Chauhan
  -> .959 (r210), yu4u & tattaka -> .961 (r89). None left .959+.

Inflow (MEASURED counts, MODEL rates): .960+ per day 12.8 (10-18 -> 10-19), 15.0 overnight, 23.9 (04:17 -> 13:20),
6.3 (13:20 -> 17:08). 54.8 h remain. MODEL: +29 to +55 more .960+ teams by the deadline, most from below us, so pw30
alone would end near public r170-200; the silver line (5 % of teams) stays near r314. Public rank is a 30 % view; it
says little about private.

## 6. Our submissions (MEASURED, `kaggle competitions submissions`, read-only)

No pending entries. 10-20 UTC used all 5 slots (eoa20 / eoa10 00:01, eoa30 / pw20 06:11-06:12, pw30 06:31); all show
public .959. Scoring lag this week ~5.5-6.5 h (MODEL from 00:01 -> scored by 05:53 and 06:31 -> scored by 13:20), so
10-21 entries pushed from 00:00 UTC should read ~06:00-07:00 UTC 10-21; 10-22 entries after ~17:30 UTC may not show
a public score before the 23:59 deadline (MODEL).

## 7. Files (this folder)

`diff_topics.py`, `new_msgs.py`, `fetch_lb_full.py`, `lb_bands.py`, `lb_bands.out`, `raw/comp.json`, `raw/l_*.json`,
`raw/msgs/msgs_*.json`, `raw/msgs/batch_*.json`, `lb/api_lb_full_1020T_private.json`, `lb/api_lb_1020T_private.json`.
