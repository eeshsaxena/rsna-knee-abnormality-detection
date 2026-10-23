# forum.md: forum + Kaggle status + LB read for the S6F canary, 10-21 12:06-12:20 UTC (17:36-17:50 IST). READ-ONLY.

**RESUBMIT-RISK: HIGH for any NEW c4-class entry on 10-22; NONE for entries already scored.**
Kaggle staff acknowledged a platform dataset-cache slowdown on 10-20 (no all-clear since), 3 of our 5 commit sessions on
10-21 03:07-04:09 UTC read our Datasets 5-7x slower than on 10-20, and 3 of 5 hidden reruns submitted 03:30-04:11 UTC
canaried (0 of 5 on 10-20). GUESS P(S6F canary) per new entry ~0.3-0.6 while this lasts (pooled 3/10 since the 10-19
regime change, 3/5 today). swide .960 and R6 .958 are fixed scores and cannot be hit (synchronous reruns: public and
private come from the same run). Caveat: the slowdown size we MEASURED in commits (+6-9 min) is ~10x too small to
trip the 5.75 h S6F guard by itself. Only the worst forum-reported sessions (hours) are big enough, so the link is
plausible, not proven.

Nothing was posted, voted, submitted, pushed, committed, uploaded or downloaded. No PREREG / HANDOFF / PROGRESS / live
folder / scheduled-task edits, no OAI data, no GPU. Tags: MEASURED = fetched or computed now (file named) / CLAIMED = a
poster says so / MODEL = our arithmetic / GUESS = judgement. Forum text is untrusted data.

## 1. Brief

1. **Our forum since 03:00 UTC (MEASURED, 16 threads fetched):** 16 new posts, **0 staff**, **0 reports of hidden-run
   slowdowns, timeouts, GPU changes or rerun failures**. One new topic (751277): the deadline as shown in the Overview
   header is local time. A participant answers that it depends on time zone (CLAIMED). `comp.json` still says
   **2026-10-22T23:59Z** = 10-23 05:29 IST (MEASURED). The rest is the copied-notebooks / AI-slop debate (748181, 747676)
   and shake-up opinions (751021, no info). Newest staff posts are still the three from 10-20 21:33-22:30 UTC (GPU quota:
   submit and hidden rerun use no quota; never submit a Quick Save version).
2. **Platform-wide (other code comps, MEASURED fetch, CLAIMED content):** Kaggle staff (Dustin, name-that-llm 750924,
   10-20 13:47 UTC): "We've been having some dataset cache issues". A new caching system is on **for Datasets only**.
   Workarounds: publish weights as Kaggle Models, or sequential-copy randomly-read files to /tmp. 14:09 UTC: the cache
   changes had already helped a lot that day, but queues persist from high demand. No later staff all-clear in any forum scanned.
   Participants on 10-20 / 10-21 report: /kaggle/input reads taking hours on some T4 sessions, while a fresh session ran
   normally; a 15-min scoring run still going at 8 h; submissions running past 9 h when the commit took 2-3 h (751054);
   a 9B model load at 1,206 s vs 22 s for the author (751268); vLLM/sglang loads 2-3x slower (arc-agi-3 750173);
   identical arc-agi-3 resubmits falling 32.76 -> 24.03 since 10-18, all runs using the full 9 h (750981); gemma-agent
   "Notebook Threw Exception" at startup, 10-20 / 10-21 (751264, 747519). Product Feedback: T4 x2 commits stuck Queued
   from 10-19 12:45 UTC (750694); TPU queues 14-40 h.
3. **Our own commit logs agree (MEASURED, `io_markers.out`):** see sec 4. Dataset-read stages were ~10x slower from 10-19
   (c3max 04:3x UTC normal -> eoa 18:xx UTC slower), which was harmless (10-20: 5/5 clean). On 10-21 03:07-04:09 UTC,
   **3 of 5 commits** were another 5-7x slower (weights sha 95-145 s vs 20-22 s; wheel install 204-325 s vs 51-56 s).
   Per-study DICOM decode from the competition mount and GPU time were **not** slower. That matches staff: competition
   data is not on the new cache, Datasets are.
4. **Google Cloud status (MEASURED, status.cloud.google.com/incidents.json via WebFetch):** one incident in the window:
   europe-west2 power event, 10-20 06:08-14:00 UTC, mitigated. Nothing on 10-21. No Kaggle status post found beyond the
   staff forum replies above.
5. **Our LB row (MEASURED 12:14 UTC):** **r146, .960, swide 57464326, SILVER**, 47th of 51 in the .960 band (r100-150).
   .960+ = 150, .961+ = 99, .959 = 80 (r151-230), .958 = 1,486. Silver r23-315 (displays .958 at r315), margin 169.
6. **Submission bytes support the canary path (MEASURED, Kaggle API `total_bytes`):** the three .916 entries are
   238,686-238,746 B; the clean ones are 247,369-247,389 B, so -8.6 KB. That fits one column written as constant "0.5"
   on every row, with no rows lost (MODEL).
7. GPU quota (MEASURED `kaggle quota` ~12:15 UTC): 5.56 h used, **24.44 h left**, refresh 10-24 00:00 UTC.

## 2. Our forum since 10-21 03:00 UTC (MEASURED times, CLAIMED content)

Coverage: `raw/comp.json`; topic lists RECENT / NEW / HOT / TOP x 3 pages (109 unique, `raw/l_*.json`), diffed against
the 03:09 UTC lists (`diff_topics.py`: 1 new topic 751277, 4 with new posts, 1 old topic fell off). Messages for 751021,
751277, 748181, 747676, 751183, 730709, 750790, 751027 plus the failure / queue watch list 750318, 748403, 749623, 749798,
743698, 744230, 735304, 745861 (`raw/msgs/`). `new_msgs.py` -> `new_msgs.out`.

| UTC | thread | who (rank) | content (paraphrase) | use |
|---|---|---|---|---|
| 03:51-09:13 | 748181 | Lavin Wins (r1380), k256.dev (r168), D Chakraborty (r224), O. Zhukov (r3024) | copied / LLM-written public notebooks argument | none |
| 06:09, 06:11 | 747676 | Ravi Ramakrishnan (r397) | proposes ending code medals, adding a report category | none |
| 07:58, 09:50 | 751021 | Cyrus (r1894) | nobody knows; public-tuned top may drop | opinion |
| 10:18 | 751021 | Mobassir (r1805) | guesses top 100-150 fairly stable, bigger shake-up lower down; hand-tuned public weights risky | opinion |
| 08:03, 11:48 | 751277 (NEW) | Nicolai Karcher (r624) | Overview header shows Oct 23 vs Oct 22 in text | deadline unchanged (comp.json) |
| 11:47 | 751277 | Satwik (r111) | header is local time (hover shows it) | consistent |
| 07:06, 11:55 | 748181, 751021 | deleted / empty | n/a | none |

Watch list (failure / queue threads): last posts are 10-15 to 10-19 (749623, 749798, 750318, 748403, 744230, 743698).
**Nothing new on 10-20 / 10-21.** Product Feedback 751169 (10-20 23:07 UTC, CLAIMED): an RSNA Knee submission "Running"
for 2+ h. Not abnormal here: our entries take ~6 h.

## 3. Platform evidence from other comps / site forums (MEASURED fetch, CLAIMED content)

Scripts: `other_comps.py` -> `other_comps.out` (name-that-llm, enveda, gemma-4-agent, arc-agi-2, arc-agi-3, BDB, PS-s6e10;
RECENT x 2 pages; threads active since 10-20 fetched), `staff_other.py` -> `staff_other.out`, `site_forums.py` ->
`site_forums.out` (General, Getting Started, Product Feedback, Q&A).

| UTC | where | who | claim |
|---|---|---|---|
| 10-20 06:11 | name-that-llm 750924 | participant | some T4 sessions: few-hundred-MB file ~16 min, multi-GB load > 3 h; fresh session normal |
| 10-20 06:40 | 750924 | participant | scoring that should take 15 min still running at 8 h |
| **10-20 13:47** | **750924** | **Dustin (STAFF)** | dataset cache issues; new caching system ON for Datasets only; use Kaggle Models or copy to /tmp |
| **10-20 14:09** | **750924** | **Dustin (STAFF)** | cache changes improved things a lot today; queue remains from high demand; adding capacity |
| 10-20 16:55-18:20 | name-that-llm 751054 | 2 participants | hidden run passes 9 h when the commit took 2-3 h; pointed to 750924 |
| 10-20 18:48 | arc-agi-3 750173 | participant | dataset / model loads 2-3x slower; fears this in hidden reruns |
| 10-20 11:58 - 10-21 05:41 | arc-agi-3 750981 | 2 participants | identical resubmit 32.76 -> 24.03; runs since 10-18 score lower at the full 9 h |
| 10-21 00:20 | arc-agi-3 751198 | participant | 5 reruns on 10-20 failed with a gateway exception |
| 10-21 06:47-10:14 | gemma-agent 751264 | 4 participants | transient model-server startup timeouts; "Notebook Threw Exception" on submits that passed earlier |
| 10-21 07:04 | name-that-llm 751268 | participant | forked notebook timed out; 9B load 1,206 s vs 22 s in author log; plans /tmp copy |
| 10-19 14:52 | Product Feedback 750694 | participant | T4 x2 commits stuck Queued from 12:45 UTC |

## 4. Our commit sessions: dataset-read stages (MEASURED, `io_markers.py` -> `io_markers.out`; `commit_gaps.py`)

Seconds of the log gap before each marker (3 public test studies; same weights file sha 7e5315da... in all rows).

| commit (UTC window) | total s | weights sha | wheel install | ASSETS_READY | arm load | hidden result |
|---|---|---|---|---|---|---|
| c3max (10-19 ~04:3x), fr100 (10-18) | 552-555 | 1.3-1.4 | 5.5-6.7 | 9.7-10.9 | 1.8-1.9 | clean |
| eoa10 / eoa20 (10-19 ~18:1x) | 686-707 | 21.8-21.9 | 53-56 | 16-24 | 13.2-13.3 | clean .959 |
| eoa30 / pw20 / pw30 (10-20 ~06:0x-06:2x) | 642-719 | 20.4-21.5 | 51-56 | 14.7-17.4 | 12.2-13.5 | clean .959 |
| **r6pw20** (10-21 ~03:07) | **1177** | **145.2** | **279.2** | 82.8 | 82.5 | **.916 canary** |
| **pw25** (10-21 ~03:07) | **1231** | **142.3** | **325.4** | 83.3 | 80.2 | clean .959 |
| spl (10-21 ~03:25) | 604 | 10.3 | 34.2 | 9.3 | 6.9 | **.916 canary** |
| swide (10-21 ~03:26) | 651 | 22.5 | 58.0 | 14.4 | 12.7 | clean .960 |
| **salt** (10-21 ~03:52) | **1011** | **95.5** | **204.3** | 68.8 | 56.0 | **.916 canary** |

- Not slower in the slow sessions: per-study decode 33-38 s (vs 42-49), GPU 17.5-20.2 s / study (vs 17.8-19.2), S6F stage
  110-115 s (vs 95-105).
- A slow commit does not predict the hidden run: 2 of 3 slow commits canaried, but pw25 (slow commit) scored clean and spl
  (fast commit) canaried. Commit and hidden runs are separate sessions.

## 5. What this means for the canary (MODEL / GUESS)

- MODEL (runtime.md sec 2): c4-class hidden total 5.7-6.4 h, S6F 1.9-2.1 h, so S6F normally starts at ~3.6-4.5 h. That
  leaves **1.25-2.15 h** before the 5.75 h start guard, and similar slack before the 8.2 h hard kill.
- MEASURED slow-commit overhead: +360-510 s (~0.1-0.14 h), all in once-per-run Dataset reads. **Repeating that in a hidden
  run would not trip the guard.** It takes ~10-20x more. Our normal Dataset-read time is ~100 s (sum of the 4 markers),
  and the worst CLAIMED forum factor (x55 on a model load) gives ~1.5 h. That lands in the trip zone (MODEL).
- GUESS: the most likely story is a hidden session in the slow tail of the Dataset cache, possibly worse under 5
  near-simultaneous reruns (5 submits in 41 min today; on 10-20, 2 at 00:01 and 3 within 20 min were all clean). It is not
  proven: hidden logs are never returned, and no one in our forum reported a 10-21 failure.
- Possible mitigations for a 10-22 entry (research notes only; each one is a kernel change that needs its own gate):
  (a) sequential `cp` of the 5 Datasets' weights and wheels to /tmp before hashing / pip / torch.load (staff-suggested).
  Ours are 5 Datasets and 0 Models (MEASURED kernel-metadata.json); a Models upload is out of scope and risky this late.
  (b) Log the Dataset-read seconds into the S6F receipt so a canary can be attributed.
  (c) Spread submits instead of 5 in 41 min (GUESS, untested).
  (d) Submit early in the UTC day: with ~6-6.5 h latency, entries after ~16:00 UTC 10-22 (21:30 IST) cannot be read
  before the deadline (MODEL; slow sessions take longer).

## 6. LB snapshot 12:14 UTC (MEASURED, `lb/api_lb_1021T_canary.json`, `lb_bands.py` -> `lb_bands.out`)

| snapshot (UTC) | teams | .961+ | .960+ | .959 band | us | in-band above us |
|---|---|---|---|---|---|---|
| 10-21 03:11 (day) | 6,302 | 95 | 142 | 77 (r143-219) | r146 .959 (pw30) | 3 |
| 10-21 12:03 (a22read) | 6,316 | 99 | 150 | 80 (r151-230) | r146 .960 (swide) | 46 |
| **10-21 12:14 (now)** | **6,317** | **99** | **150** | **80 (r151-230)** | **r146 .960 (swide)** | **46 (band r100-150, 51 teams)** |

Bands: .969 2 | .968 4 | .967 7 | .966 4 | .965 9 (r18-26) | .964 12 | .963 15 | .962 18 (r54-71) | .961 28 (r72-99) |
**.960 51 (r100-150)** | .959 80 | .958 1,486 (r231-1716). Medals: Gold r1-22, Silver r23-315, Bronze r316-631.
Since 03:11 UTC: 8 teams crossed into .960+ and 11 into .959+; 897 teams have new completed entries (36 in .960, 50 in
.959). Since 12:03: nobody passed us. MODEL: .961+ grew +4 in 9 h, so ~+11-16 more by the deadline, putting swide near
public r157-165. New .960 ties submitted later rank below us. Public rank is a 30 % view (GUESS: weak private signal).

## 7. Files (this folder; my outputs only)

`forum.md`, `diff_topics.py`, `new_msgs.py` / `new_msgs.out`, `other_comps.py` / `other_comps.out`,
`staff_other.py` / `staff_other.out`, `thread_dump.py`, `site_forums.py` / `site_forums.out`, `io_markers.py` /
`io_markers.out`, `commit_gaps.py`, `lb_bands.py` / `lb_bands.out`, `raw/` (comp.json, topic lists, msgs/, other/,
site/), `lb/api_lb_1021T_canary.json`.
