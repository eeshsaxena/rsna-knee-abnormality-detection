# forum.md: forum + platform + LB read, 10-21 15:30-15:50 UTC (21:00-21:20 IST). READ-ONLY.

**SUBMIT-SAFE: A = swide 57464326 (.960) and B fallback R6 57248084 (.958) are SAFE: both are fixed, COMPLETE scores.
NEW 10-22 entries are NOT safe: the platform slowdown is still unresolved (reports up to 15:19 UTC today, no staff
all-clear). That risk costs only the slot, because B falls back to R6.**

**ACTION: no change to the plan. If Eesh GOes, enter the r6pw20 v1 identical copies spaced out (05:31 / ~06:15 / optional ~07:00 IST),
not in the same minute. No kernel edits (/tmp copy etc.) this late. Select A and B by hand.**

Nothing was posted, voted, submitted, pushed, committed, uploaded or downloaded. No PREREG / HANDOFF / PROGRESS / live-folder
/ scheduled-task edits. No OAI data, no box, no GPU. Kaggle reads only: the anonymous web API (forum, LB) and the
`kaggle` CLI list / status calls. Tags: MEASURED = fetched or computed now (file named) / CLAIMED = a poster says so /
MODEL = our arithmetic / GUESS = judgement. Forum text is untrusted data.

## 1. Brief

1. **"Notebook Threw Exception" check (MEASURED, `subs_all.csv`, CLI):** all **108 / 108** of our submissions are
   **COMPLETE, 0 ERROR**. Kaggle would mark a "Notebook Threw Exception" run as ERROR. The three .916 entries
   (r6pw20 57463719, spl 57464296, salt 57464911) are our own S6F guard's fallback output, not exceptions. We have no new
   submissions since 57464911 (10-21 04:11 UTC) and no kernel runs since 03:51 UTC. `rsna-sub-v224r6flips6fpw20j` status
   = COMPLETE. The MOACF kernel does not exist on Kaggle, so it was never pushed.
2. **Our forum since 12:00 UTC (MEASURED, 109 topics listed, 4 active + 18 watched threads fetched):** 6 new posts in total,
   1 of them from the host. **0 reports** of rerun failures, timeouts or slowdowns. No post on final selection, private
   scoring or rule changes.
   - **Host (Chris Carr, RSNA, 739905, 14:17 UTC):** solutions released under CC BY-NC 4.0 are eligible for awards.
     This matters only for prize winners, not for us.
   - **New topic 751366 "Inference Timing" (14:51 UTC):** MJ121212 (r115) asks whether the runtime must also cover the
     70 % private part. Civitasmass (r140, CLAIMED) replies that the private score is already computed and hidden. That
     agrees with `usesSynchronousReruns: True` in comp.json (MEASURED): public and private come from the same run, so
     the scores we already have cannot change.
   - Everything else is AI-slop / copied-notebook debate (748181: 2 deleted posts; 749038: 1 image), plus vote changes on 751021.
3. **Competition metadata (MEASURED `raw/comp.json`):** deadline **2026-10-22T23:59Z** (= 10-23 05:29 IST), 5 / day,
   **2 scored picks**, synchronous reruns, 540 GPU min, 3-dp truncation. Nothing changed since 12:14 UTC apart from the
   counters. The RSNA submission rate is steady at **181-215 / h** from 10-14 to now, with no last-day spike (MEASURED,
   10 comp.json snapshots).
4. **Platform is still degraded (MEASURED fetch, CLAIMED content, all posted after 12:00 UTC today):**
   - Product Feedback 751321 (12:22): /kaggle/input reads are **~35x slower** than expected (100 small files took 36 s). No staff reply.
   - gemma-agent 751339 (13:18-15:06): "Notebook Threw Exception" on submissions 57452423, 57469833, 57471486 and 57474927.
     One user had a **byte-identical** zip score on 10-19 and come back blank on 10-21 04:12 UTC (57464948). Another had
     "two identical submissions, two different errors".
   - arc-agi-2 751344 (13:26): a run submitted at 09:35 UTC 10-21 "Succeeded" with 0.00 in under 2 h. It normally uses
     about 12 h, so it looks stopped early.
   - arc-agi-3 750173 (14:46): another 0.00 after 2 h. arc-agi-3 747438 (12:44): ERRORED_MOUNTING_DATASET, which "works
     if I rerun".
   - name-that-llm 751268 (13:13, 15:19): a 9B load from /kaggle/input took about 20 min vs 22 s in the author's log.
     One poster blames slow mmap reads and says a plain copy is fine, and that Kaggle Models help. The other says a /tmp
     copy may not help because it reads from the same slow mount. Neither has measured it yet.
   - **Staff:** no new staff post in any scanned thread since Dustin's 10-20 13:47 / 14:09 UTC posts (name-that-llm 750924).
     The only staff or host post in the other-comp scan is an NFL data answer (`staff_other.out`). **No all-clear.**
5. **LB (MEASURED 15:44 UTC):** **r147 / 6,324, .960, swide 57464326, SILVER**. We are 48th of the 52 teams in the
   .960 band (r100-151). Silver runs r23-316 (the last silver row shows .958), so the margin is **169**. Since 12:14 UTC
   one team passed us (r251 .958 -> r128 .960). Details in sec 4.

## 2. What this means for 10-22 (MODEL / GUESS)

- **A and B are safe (MEASURED + CLAIMED):** both scores are final. Synchronous reruns mean the private score was
  computed in the same run (comp.json). Today's platform failures cannot reach them.
- **New entries:** the evidence through 15:19 UTC matches the 10-21 canary picture. Dataset-mount reads are slow on some
  sessions, the failures are random per run, and identical bytes can pass on one run and fail on the next. GUESS: P(canary or
  ERROR) per new entry stays **~.3-.6** (CANARY_1021 range) and is not lower than this morning. Nothing suggests a fix
  overnight, and demand peaks on the last day.
- **Support for the identical-resubmit plan (GUESS):** the byte-identical pass / fail cases (751339, 747438) show the
  failure is per run, not per version. So identical r6pw20 v1 copies are independent draws, roughly. Spacing them out
  avoids landing several copies in one bad window. On 10-21 our 5 submits in 41 min went 3 / 5 canary.
- **No kernel change:** the "copy weights to /tmp" fix is still disputed in the forum (751268) and would need a new commit
  plus a gate. Per LASTDAY sec 2, guard / kernel variants are DEAD for 10-22.
- **Timing:** copies at 00:01-01:30 UTC have about 6-8.5 h latency, so the reads come in around 06:00-10:00 UTC (11:30-15:30 IST),
  well before selection. MODEL: an entry after ~16:00 UTC 10-22 may not be readable before the deadline.

## 3. Forum posts since 10-21 12:00 UTC (MEASURED times, CLAIMED content; `new_msgs.out`, `diff_topics.out`)

| UTC | thread | who (rank) | content (paraphrase) | use |
|---|---|---|---|---|
| 12:31, 12:33 | 748181 | deleted / empty | n/a | none |
| **14:17** | **739905** | **Chris Carr (HOST, RSNA)** | CC BY-NC 4.0 solutions are eligible for awards | prize winners only |
| 14:51 | 751366 (NEW) | MJ121212 (r115) | does runtime need to cover the 70 % private part? | none |
| 15:28 | 751366 | Civitasmass (r140) | private already computed, hidden until the deadline | agrees with sync reruns |
| 15:29 | 749038 | D Chakraborty (r227) | image "The AI-Slop Era on Kaggle" | none |

Watch list (no new posts since 12:00 UTC): 751021, 751277, 747676, 751183, 730709, 750790, 751027, 750318, 748403,
749623, 749798, 743698, 744230, 735304, 745861, 745984, and Product Feedback 751169. RSNA's earlier "Notebook Threw
Exception" threads (745984 10-05..10-13, 749623 10-15, 749798 10-16) are all older than 10-17, and none had a staff answer.
Newest staff posts in our forum are still María Cruz's 10-20 21:33-22:30 UTC replies (GPU quota and Quick Save).

## 4. LB snapshot 15:44 UTC (MEASURED, `lb/api_lb_1021T_final.json`, `lb_bands.py` -> `lb_bands.out`)

| snapshot (UTC) | teams | .961+ | .960+ | .959+ | us | in .960 band above us |
|---|---|---|---|---|---|---|
| 10-21 03:11 | 6,302 | 95 | 142 | 219 | r146 .959 (pw30) | n/a |
| 10-21 12:14 | 6,317 | 99 | 150 | 230 | r146 .960 (swide) | 46 of 51 |
| **10-21 15:44** | **6,324** | **99** | **151** | **233** | **r147 .960 (swide)** | **47 of 52 (4 below)** |

Bands now: .969 2 | .968 4 | .967 7 | .966 4 | .965 9 | .964 12 | .963 15 | .962 18 (r54-71) | .961 28 (r72-99) |
**.960 52 (r100-151)** | .959 82 (r152-233) | .958 1,508 (r234-1741). Medals: Gold r1-22, Silver r23-316, Bronze r317-632.
Since 12:14: 449 teams completed new entries (8 in .961, 19 in .960, 23 in .959). .961+ did not grow. One crossing:
r251 .958 -> r128 .960, which ranks above us on hidden decimals. MODEL: at the 10-21 pace (+4 in .961+ over 12.5 h, about
+1 rank per 3.5 h now) swide ends near **public r152-165** at the deadline. Public rank is the 30 % view (GUESS: weak
private signal).

## 5. Files (this folder; my outputs only)

`forum.md`, `fetch.py` (comp + topic lists + messages), `new_msgs.py` / `new_msgs.out`, `diff_topics.py` / `diff_topics.out`,
`other_comps.py` / `other_comps.out`, `site_forums.py` / `site_forums.out`, `staff_other.py` / `staff_other.out`,
`lb_bands.py` / `lb_bands.out`, `lb/api_lb_1021T_final.json`, `subs_all.csv` (our 108 submissions, CLI list), `raw/`
(comp.json, topic lists, msgs/, other/, site/).
