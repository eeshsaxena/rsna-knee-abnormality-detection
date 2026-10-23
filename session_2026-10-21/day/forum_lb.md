# forum_lb.md: forum + LB read, 10-21 03:09-03:12 UTC (08:39-08:42 IST). READ-ONLY.

**SUBMIT-SAFE: YES.** Nothing new changes the rules, data, metric, deadline or how finals are picked. The A22 slate
(r6pw20, pw25, spl, swide, salt) and the A / B pick rule are unaffected.
**ACTION: none required.** FYI (staff, 10-20 21:33 UTC, 750790 + 730709): new GPU versions need weekly GPU quota;
submitting an existing version and the hidden rerun do not. Do not submit "Quick Save" versions (no GPU attached).
Our position: GPU quota **25.62 h left of 30 h** (MEASURED `kaggle quota` 03:10 UTC, refresh 10-24 00:00 UTC, after
the deadline). The A22 kernels are API pushes with `enable_gpu: true` (MEASURED, kernel-metadata.json) and both 08:37
pushes are running as full GPU commits (MEASURED `kaggle kernels status`: RUNNING at 03:11 UTC). The remaining ~8
commits (3 today + up to 5 on 10-22) at 0.1-0.25 h each (HANDOFF figure) need about 1-2 h. No quota risk.

Nothing was posted, voted, followed, submitted, pushed, committed or downloaded. No PREREG / HANDOFF / PROGRESS /
val / a20 / a21 / stack / scheduled-task edits. No OAI data touched, box not used. Tags: MEASURED = fetched or
computed now (file named) / CLAIMED = a poster says so / MODEL = our arithmetic / GUESS = judgement.

## 1. Brief

1. Host / staff: **3 new posts, all María Cruz (Kaggle staff)**, none a ruling. 750790 (21:33 UTC) declines GPU quota
   top-ups and gives the quota guidance above; 730709 (21:34 UTC) repeats it; 747676 (22:30 UTC) forwards a Code-tab
   voting complaint to product teams. Both quota posts restate the deadline as Thursday 10-22 23:59 UTC.
2. Rules / metadata (MEASURED `raw/comp.json` vs the 10-20 17:06 read): only counters changed. Deadline
   2026-10-22T23:59Z, 2 scored finals, 5 a day, 30 % public, 3-decimal truncation. Teams 6,275 -> 6,302; submissions
   149,824 -> 151,726 (+1,902 in ~10 h).
3. Final selection / private: nothing. The LB API still returns only the public board (MEASURED). 751021 (private
   shake-up question) got one reply from a r1453 participant, no information.
4. Watch list: no new post in site split (734681), missing planes (744519), Synovitis (737155), external data (743416,
   744056, 747455, 750752, 750889, 750899) or OA / label regime (745949). External-data status stays GRAY, Eesh's call.
5. Our row: **r146, .959, best sid 57425970 (pw30)**, SILVER. r141 at the 10-20 22:40 IST read. Still **4th in the
   .959 band** (above us: imainjob, Civitasmass, OzanM.); 73 .959 teams below us.
6. Bands now: .961 = 26 (r70-95); .960 = 47 (r96-142); **.960+ = 142** (was 137); **.959 = 77 (r143-219)** (was 74);
   .958 = 1,457 (r220-1676) (was 1,399). Top .969 now 2 teams. Medals: Gold r1-22, Silver r23-315, Bronze r316-630;
   silver line displays .958. Margin to silver 169.

## 2. Coverage (MEASURED)

- `raw/comp.json` (GetCompetition 03:09 UTC). Topic lists RECENT / NEW / HOT / TOP, 3 pages each, 109 unique topics
  (`raw/l_TOPIC_LIST_SORT_BY_*.json`) via `session_2026-10-16/forum/fr.py`; diffed against the 10-20 17:06 UTC lists
  (`session_2026-10-20/private/raw/`) by `diff_topics.py`. 1 new topic (751183), nothing fell off.
- Threads with posts after 10-20 17:00 UTC (all fetched): 751027, 751021, 748181, 750790, 730709, 747676, 751183.
  9 posts in total. Watch list also fetched: 747455, 734681, 744519, 737155, 743416 (+13 collapsed), 744056, 750899,
  750889, 750752, 745949, 750318, 748403. Messages via `session_2026-10-16/forum/msgs.py` into `raw/msgs/`;
  `new_msgs.py` -> `new_msgs.out` prints every post since 16:30 UTC plus all host / staff posts.
- LB: `lb/api_lb_full_1021T_day.json` (raw) + `lb/api_lb_1021T_day.json` (trimmed), fetched 03:11:07 UTC via
  `session_2026-10-20/private/fetch_lb_full.py`; analysis `lb_bands.py` -> `lb_bands.out` against the 10-19 20:17 and
  10-20 04:17 / 05:53 / 13:20 / 17:08 UTC snapshots.

## 3. Posts since 10-20 17:00 UTC (MEASURED times, CLAIMED content, rank as shown on the post)

| UTC | thread | who (rank) | content (paraphrase) | use |
|---|---|---|---|---|
| 17:49, 17:56 | 751027 | Angel J. Sanchez Ruiz (r1453), Optimo (r51) | dataset uploads work again; Optimo confirms his upload went through | closes the 10-20 upload FYI; our kernels need no upload anyway |
| 17:52 | 751021 | Angel J. Sanchez Ruiz (r1453) | guesses shake-up for the crowded middle, not for true top teams | opinion only |
| 21:16 | 748181 | D Chakraborty (r213) | argues with another poster about copied / LLM-written notebooks | none |
| **21:33** | **750790** | **María Cruz (STAFF)** | no GPU top-ups or early resets (reset Sat 00:00 UTC); submit / rerun use no quota; new GPU versions need quota, keep a buffer; Quick Save drops the GPU, so its rerun runs on CPU and likely fails | see ACTION FYI; no effect on us |
| **21:34** | **730709** | **María Cruz (STAFF)** | same guidance, to another user; deadline 10-22 23:59 UTC | same |
| **22:30** | **747676** | **María Cruz (STAFF)** | Code-tab downvote request forwarded to product teams | none |
| 23:38 | 747676 | D Chakraborty (r213) | wants detection / penalties for LLM-generated notebooks | none |
| 23:41 | 751183 (NEW) | M Murat Gunes (r5526) | write-up of a weak-label cascade; gold-58 macro AUC .702 (CLAIMED) | none (far below us) |

No technical content from anyone ranked above us. No hint on OA labels, compartment coupling, test sites, missing
planes or final selection.

## 4. Watch list

| item | thread | last post | status |
|---|---|---|---|
| host posts | all | 747676, 10-20 22:30 (staff) | 3 staff posts, all procedural (sec 3) |
| rules / deadline / selection | comp metadata | n/a | unchanged; staff restated 10-22 23:59 UTC |
| GPU quota for final commits | 750790, 730709 | 10-20 21:33 | staff guidance; we have 25.62 h (MEASURED) |
| final selection / rerun rules | none | n/a | no thread |
| private scoring | 751021 | 10-20 17:52 | one opinion reply, no info |
| site split | 734681 | 08-14 14:08 | nothing new |
| missing planes | 744519 | 10-06 02:17 | nothing new |
| Synovitis rubric | 737155 | 08-23 20:37 | nothing new |
| external data | 743416, 744056, 747455, 750752, 750889, 750899 | 747455 10-19 22:46 | nothing new; GRAY, Eesh's call |
| OA / label regime | 745949 | 10-19 15:56 | nothing new |
| submit queue / stuck | 750318, 748403 | 10-19 04:45 | nothing new |

## 5. LB snapshot 03:11 UTC (MEASURED, `lb_bands.out`)

| snapshot (UTC) | teams | .960+ | .959 band | .958 band | us | in-band above us |
|---|---|---|---|---|---|---|
| 10-20 04:17 | 6,249 | 127 | 66 (r128-193) | 1,337 | r217 .958 (c4) | 23 |
| 10-20 13:20 | 6,270 | 136 | 72 (r137-208) | 1,382 | r140 .959 (pw30) | 3 |
| 10-20 17:08 | 6,276 | 137 | 74 (r138-211) | 1,399 | r141 .959 (pw30) | 3 |
| **10-21 03:11** | **6,302** | **142** | **77 (r143-219)** | **1,457 (r220-1676)** | **r146 .959 (pw30)** | **3** |

Top bands: .969 2 (r1-2) | .968 4 | .967 6 | .966 5 | .965 7 (r18-24) | .964 12 (r25-36) | .963 15 | .962 18
(r52-69) | .961 26 (r70-95) | .960 47 (r96-142).

Movement since 10-20 17:08 UTC (22:38 IST):
- **5 teams passed us, nobody fell below us.** 4 crossed .959 -> .960 from below us: UEHA (r147 -> r124), AI for Good
  (r146 -> r131), Ginocchio Va(l)go (r144 -> r135), DoyMed (r153 -> r138). 1 in-band passer: OzanM. (r152 -> r145,
  still .959, 224 subs).
- **Crossings into .960+: 5** (the 4 above plus Khayrul Islam, who was already above us, .959 -> .960). All 5 came
  from the .959 band; none from .958 or lower.
- Crossings into .959+: 8, all landing below us (Cody_Null r180, eierapfelxx r183, Nguyen Kim Khanh .950 -> .959
  r185, Epoch VII r188, chill~ .954 -> .959 r206, Ray r211, a Hanbat team r218, kostopr4v r219). None left .959+.
- Top moves: CentralyDS .968 -> .969 (r2, now tied at the top display), Claudio Verdu Ruiz .967 -> .968, Ilya Kolb
  .966 -> .967, YC & NB .964 -> .966, Zyy7390 .963 -> .965; 13 more one-step moves inside .960+.
- Activity: 937 teams submitted since 17:08 (53 in .959, 29 in .960, 536 in .958); 26 new teams, none removed.

Rates and projection:
- MEASURED counts, MODEL rates: .960+ +5 in 10.0 h (~12/day; was 6-24/day on 10-20); .959+ +8 (~19/day); in-band
  passers 1 in 10 h.
- MODEL: 44.8 h remain. At 12-24/day, +22-45 more .960+ teams, mostly from the .959 band, so pw30 alone would end
  near public r170-195. The silver line (5 % of teams) should sit near r316-320. A .960 display from any A22 entry
  would land inside r96-142 now. Public rank is a 30 % view and says little about private (GUESS, as before).

## 6. Our submissions (MEASURED, read-only)

- `kaggle competitions submissions`: no 10-21 entries yet; latest are the five 10-20 entries (all COMPLETE, .959).
- A22 commits: r6pw20 (`rsna-sub-v224r6flips6fpw20j`) and pw25 (`rsna-sub-v224r6c4pw25j`) RUNNING at 03:11 UTC,
  pushed 03:07 UTC (tick_a22.log rc 0). The tick handles gate and submit.
- GPU quota: used 4.38 h, 25.62 h left, refresh 2026-10-24 00:00 UTC.

## 7. Files (this folder)

`diff_topics.py`, `new_msgs.py`, `new_msgs.out`, `lb_bands.py`, `lb_bands.out`, `raw/comp.json`,
`raw/l_TOPIC_LIST_SORT_BY_{RECENT,NEW,HOT,TOP}.json`, `raw/msgs/msgs_*.json`, `raw/msgs/batch_*.json`,
`lb/api_lb_full_1021T_day.json`, `lb/api_lb_1021T_day.json`.
