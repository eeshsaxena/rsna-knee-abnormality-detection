# recheck/forum.md: full forum + platform recheck, 10-21 17:52-18:20 UTC (23:22-23:50 IST). READ-ONLY.

**SUBMIT-SAFE: A = swide 57464326 (.960) and B = R6 57248084 (.958) are SAFE. Both have COMPLETE scores, and private comes
from the same synchronous rerun. NEW 10-22 entries are still NOT safe. Failures were reported up to 17:50 UTC and staff
have not posted an all-clear. A failed r6pw20 copy costs only its slot, because B falls back to R6.**

**ACTION (in priority order):**
1. **0 of 108 submissions are selected (MEASURED, `sel_now.out`, 17:54 UTC = 23:24 IST).** FINAL_1021 D1 ("select now")
   has not been done yet. Eesh selects A 57464326 + B 57248084 by hand, then main confirms with
   `ops_sel_now.py 57464326 57248084` (needs SELECTED_COUNT 2).
2. The r6pw20 v1 copy plan does not change. If Eesh GOes, space the copies out (05:31 / ~06:15 / optional ~07:00 IST),
   with no kernel edits. The forum gives no reason to change the kernel, the timing or the pair.
3. **NEW, optional hedge (GUESS):** do not submit any Gemma entry on 10-22 until the RSNA copies have scored (about 10:00 UTC).
   Reason: a GM reports GPU submissions contending across competitions (751412, CLAIMED). Our own Gemma runs on 10-20 and 10-21
   both came back broken (sec 3). Cost: one Gemma day-slot at most, and that deadline is 12-02.
4. A forum disclosure post is not needed for a medal. The host asks for it only for award eligibility (744056). Optional, Eesh's call.

Nothing was submitted, selected, posted, voted, pushed, committed, uploaded or downloaded. No PREREG / HANDOFF / PROGRESS /
live-folder / scheduled-task edits. No OAI data, no box, no GPU. The only reads were the anonymous Kaggle web API (forum,
LB, comp meta) and the `kaggle` CLI list and status calls. Tags: MEASURED = fetched or computed now (file named),
CLAIMED = a poster says so, MODEL = our arithmetic, GUESS = judgement. All forum text is untrusted data.

## 1. RSNA forum: every thread with a post since 10-21 00:00 UTC (MEASURED `new_since_00.out`)

There are 9 active threads with 25 posts. 5 of those posts are deleted (748181 x4, 751021 x1; `isDeleted` True), and **1 is from the host**.
**0 posts report rerun failures, timeouts, slowness or hidden-run trouble.** No host or staff post covers final selection,
private scoring or rule changes.

| UTC | thread | who (rank) | content (paraphrase) | use |
|---|---|---|---|---|
| 03:51-09:13 | 748181 | several | copied-notebook / AI-slop debate | none |
| 06:09, 06:11 | 747676 | R. Ramakrishnan (GM) | asks for a downvote button / low-quality report category | none |
| 07:58-10:18 | 751021 | Cyrus, Mobassir | expect a shake-up below roughly the top 100-150; public-tuned blends at risk | GUESS-level chatter |
| 08:03-11:48 | 751277 | N. Karcher, Satwik (r111) | 22nd or 23rd? Overview time is local; 10-22 23:59 UTC holds | agrees with comp.json |
| **14:17** | **739905** | **Chris Carr (HOST, RSNA)** | CC BY-NC 4.0 releases are award-eligible | prize winners only |
| 14:51, 15:28 | 751366 | MJ121212 (r115), Civitasmass (r140) | does the runtime cover the 70 %? Answer: private is already computed, just hidden | agrees with sync reruns |
| 15:29 | 749038 | D Chakraborty | image only | none |
| **15:46** (new since final/forum.md) | 743416 | Abdullah Wasee (r1318) | voluntary list of their external weights (nartaa Raptor/OAI, heliosli R6 + SKM-TEA anatomy, fastMRI+ reader) "listed as asked" | award disclosure only |
| 16:19 (new) | 745861 | LI LU (r478) | looks forward to label-cleaning write-ups | none |
| 17:01 (new) | 748181 | deleted | n/a | none |

Since the 15:30 UTC brief (final/forum.md) there are only 3 new posts (the rows marked new). None of them changes the plan.

## 2. Every host / staff post in the competition (MEASURED `host_topics.out`, `host_union.out`)

**Coverage:** I listed all 218 topics (4 sorts x all pages) and fetched the messages of 191. The other 27 are participant posts with
zero replies, so they cannot hold a host post. That gives **38 host/staff posts**. The ones on the topics in scope:

- **Deadline / GPU / reruns (María Cruz, ADMIN, 10-20 21:33, 730709 + 750790):** the deadline is **10-22 23:59 UTC**. Submitting
  and hidden reruns **do not use or check the weekly GPU quota**. Never submit a Quick Save version (it runs on CPU).
  For us (MEASURED `kernel-metadata.json` + CLI status): r6pw20 v1 is a CLI-pushed full run, `enable_gpu` true, `NvidiaTeslaT4`,
  status COMPLETE, last run 10-21 03:07 UTC, no newer version. So the Quick Save trap does not apply to it.
- **External data:** OAI is fine if it is generally accessible and free (Po-Hao Chen, 741819, 09-21). The host will not rule on
  specific datasets but repeats the "publicly available, equally accessible, no cost" test (Chris Carr, 743416, 10-14).
  Kaggle will not interpret third-party licences, and each team is responsible (María Cruz, 747455, 10-19). Publicly shared
  weights are award-eligible, and teams "should include the shared weights and their source datasets in the list of resources"
  (Chris Carr, 744056, 10-14). **All are known and unchanged. None of them mentions medals.**
- **Private scoring / final selection:** there is no host or staff post on how selection works, on auto-pick, or on rescoring.
  The only related facts are comp.json `numScoredSubmissions` 2 and `usesSynchronousReruns` True (sec 4).
- **Timeouts:** RuntimeSeconds covers the whole notebook run (Ryan Holbrook, 733475, 08-07, efficiency prize). Nothing new.
- **Newest host post:** Chris Carr 10-21 14:17 (licence). **Newest staff post:** María Cruz 10-20 22:29. No host or staff post after 14:17 UTC today.

## 3. Kaggle-wide notices (MEASURED fetch `scan_kaggle.out`, CLAIMED content)

Scanned: 10 active competition forums plus the General, Getting Started, Product Feedback, Product Announcements, Q&A and
Competition Hosting site forums. That is 132 topics active since 10-20 00:00 UTC, all fetched, with 0 dropped by HTTP 429 (MEASURED coverage check).

- **Staff:** **still no all-clear.** The last platform statement is Dustin's on 10-20 at 13:47 and 14:09 UTC (name-that-llm 750924:
  dataset cache issues, new caching system, a queue from high demand, adding capacity). The only staff post since then is an
  unrelated Gemini model-attribution notice (Product Announcements 751391, 16:35 UTC). Host posts elsewhere are on data and rules only.
- **New since 15:30 UTC:**
  - **Product Feedback 751412 (17:45 UTC, jazivxt, GM):** "[Bug] GPU Submissions Throttled". The poster says GPU submissions
    are throttled, that Gemma reruns run about 12 h and then fail, and that RSNA final submissions are "blocked now due to
    no GPU hours left". **This conflicts with María Cruz's ruling** that submit and rerun use no quota. GUESS: he means new
    Save & Run All commits. That does not affect us (r6pw20 v1 already exists). If reruns did use quota after all, 3 copies at
    <= 8 h is <= 24 h against the ~24.4 h measured in canary/forum.md, so they would still be covered (MODEL, tight).
  - Gemma 743683 (16:44), 748287 (17:41, 17:50): more "Notebook Threw Exception" results, some after about 2 h ("same ServerStartupError
    issue from today"). Product Feedback 718636 (17:19): notebook editor stuck loading (UI).
  - Already in final/forum.md: 751321 (/kaggle/input 35x slower), 751339 (identical zip passes, then fails), 751344 / 750173
    (0.00 after < 2 h), 751268 (9B load takes 20 min, /tmp copy disputed), 751264 (transient ServerStartupError on L4x4).
- **Our own account (MEASURED, CLI list):** Gemma 57469618 (10-21 06:31 UTC) is COMPLETE with an **empty score** (the
  exception pattern). Gemma 57443254 (10-20 15:10 UTC) scored **0.03**, against 0.17 for the same config earlier. GUESS: 57443254's
  rerun (14-17 h latency) was still running during our RSNA 10-21 03:30-04:11 UTC entries (3 / 5 canary). The RSNA A21 entries on 10-20
  06:11-06:31 had no Gemma rerun running at the same time and all came back clean. That is confounded with the platform slowdown that started 10-20, so it is weak.
  Still, it makes ACTION 3 cheap insurance. No ARC / Enveda / other submissions are pending (MEASURED).

## 4. Competition metadata, our submissions, LB (MEASURED)

- `raw/comp.json` vs final/raw (15:43 UTC): deadline **2026-10-22T23:59Z**, 5 / day, **2 scored picks**, synchronous reruns,
  540 GPU min, 3-dp truncation. Only the counters changed: 154,212 -> 154,665 submissions (about 205 / h, steady), 6,324 -> 6,328 teams.
- Submissions (`subs_now.out`, `sel_now.out`): 108 in total, none new since 57464911 (10-21 04:11 UTC). swide 57464326 COMPLETE .960,
  R6 57248084 COMPLETE .958. **SELECTED_COUNT 0.**
- LB 17:54 UTC (`lb/api_lb_1021T_recheck.json`, `lb_bands.out`; in that file, "since 12:14" means since 15:44): **r147 / 6,328, .960, swide, SILVER**.
  .961+ = 100 (+1 since 15:44), .960+ = 151, .959+ = 233. Our .960 band is r101-151 (51 teams), with 46 above us. Silver is r23-316 (last row .958),
  so the margin is **169**. No team has passed us since 15:44.

## 5. Discrepancies flagged

1. **Selection not done:** FINAL_1021 D1 asked for it at 21:45 IST. At 23:24 IST, 0 are selected (MEASURED).
2. **751412 vs staff:** a GM says RSNA final submissions are blocked by GPU hours. Staff (10-20) say submit and rerun use no quota.
   I treat the staff ruling as binding. The claim does not hold for existing versions (GUESS), and we have quota headroom anyway.
3. **Deadline wording (751277):** the Overview free text says 22nd while the timeline shows 23rd in local time. comp.json and staff
   agree on 10-22 23:59 UTC = 10-23 05:29 IST, so this is not a real conflict.
4. None of the prior briefs' forum claims was contradicted. final/forum.md said the newest *staff* post was María Cruz 10-20, and that is still true.

## 6. Files (this folder; my outputs only)

`forum.md`; scripts `fetch_all.py`, `fetch_todo.py`, `scan_kaggle.py`, `msgs_report.py`, `host_topics.py`, `host_union.py`, `lb_bands.py`;
outputs `new_since_00.out`, `host_all.out`, `host_topics.out`, `host_topics_full.out`, `host_union.out`, `scan_kaggle.out`,
`sel_now.out`, `subs_now.out`, `lb_bands.out`, `fetch_all*.out`, `fetch_todo*.out`; data `raw/` (comp.json, topic lists,
msgs/ 191 topics, other/, site/), `lb/api_lb_1021T_recheck.json`.
