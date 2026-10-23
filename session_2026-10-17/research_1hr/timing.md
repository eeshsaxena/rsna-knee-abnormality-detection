# Scoring + timing lens (research_1hr, 10-17 ~22:00-22:40 IST). READ-ONLY.

Tags: MEASURED = read from our logs / API / Kaggle metadata in this session; CLAIMED = someone says so (forum, Kaggle docs, our earlier notes); MODEL = derived from measured numbers with a stated assumption; GUESS = judgement.
Times are IST unless marked UTC. UTC day D = 05:30 IST D to 05:29 IST D+1.

## 0. Bottom line
1. **Final picks are NOT re-run.** The competition object says `usesSynchronousReruns: True` (MEASURED, GetCompetition + session_2026-10-16/forum/comp.json). In a synchronous-rerun code competition the notebook runs once on the full hidden test set (public + private) at submit time, and the private score comes from that same run (CLAIMED, Kaggle code-competition FAQ, from memory: the docs page is JS-only and WebFetch got nothing). The Efficiency Prize text backs this up: "RuntimeSeconds is the number of seconds it takes for the submission to be evaluated" (MEASURED, page_Efficiency_Prize_Evaluation.txt). So HW 57262115 and R6 57248084 have a fixed private score already, with no runtime, canary or GPU-noise risk left at the final.
2. **Submit-to-score latency is about the same as the kernel runtime. No queue showed up in scoring between 10-14 and the morning of 10-17** (MEASURED bounds, table in section 1). The R6/HW compute class (HW, S6W, hw3 and all 3 probes are weight-only changes) takes **about 6.0-6.25 h** (MODEL: intersection of R6 6h03-6h15, hw3 <= 6h03, S6W >= 5h11). The extra R7 late leg makes it 6h37-7h26 (MEASURED).
3. **HW vs the 9 h limit:** MEASURED upper bound 6h33 (the session restart left a loose read). MODEL about 6.0-6.25 h, so roughly 2.75-3 h of margin. The S6F start guard (G 5.75 h kernel age) has about 1.6 h of margin (MODEL, section 3). It did not fire in HW's scored run: HW moved the best-id, and a fired canary would have cost score (MEASURED read + MODEL). This does not matter for picks, because nothing is re-run.
4. **The only queue we have seen is on COMMIT runs (10-17 evening):** s615 sat QUEUED 2h25 and a25 about 2h58, while wr45, pushed later, started within 1 min (MEASURED, probes/out_*.log). Scoring reruns submitted 12:18-15:28 UTC on 10-17 are tonight's test of whether scoring is hit too.
5. **Timing plan:** on 10-22 the last HW-class submit that can still be read by the A15 cutoff (10-22 18:00 IST = 12:30 UTC) is **about 10:15 IST**, with a 1.5 h buffer over 6.25 h latency. If tonight's probes show a scoring queue, it moves to **08:45 IST** (3 h buffer). Commit every 10-22 kernel by **10-21 ~22:00 IST**, since commits queued up to 3 h on 10-17. Details in section 5.
6. **Tonight's reads need a snapshot between completions.** Expected completions: wr45 about 23:50-00:05, s615 about 01:20-01:40, a25 about 03:00-03:15 (MODEL, about 1.5 h apart). If the laptop sleeps through one completion and a later probe moves the best-id, the earlier probe's R1 read is lost (MODEL, section 4).

## 1. Submit-to-score latency per entry (MEASURED bounds)
Submit time comes from the API list (python `competition_submissions`, UTC, exact). The bounds come from LB snapshot files (fetched_utc + our team's `submissionCount` / `lastSubmissionDate`) plus the poller / chain logs (session_2026-10-14/api_chain.log, PROGRESS_14..17).
"Lower" = a snapshot showed it not yet complete. "Upper" = first snapshot or log line that showed it complete.

| ref | entry | submit UTC (IST) | latency bounds | tightness / note |
|---|---|---|---|---|
| 57163771 | HFR7-RR-F30 | 10-14 02:17 (07:47) | 4h51 - 4h58 | chain (180 s poll) |
| 57164843 | HFR7-RR-F40 | 10-14 03:05 (08:35) | 4h28 - 4h54 | chain |
| 57162815 | HFR7-RS-F30 | 10-14 01:37 (07:07) | 5h38 - 5h55 | chain |
| 57171955 | V224-exact (base) | 10-14 07:08 (12:38) | <= 3h16 (lower 0h51) | loose; the sim notes give <= 3.28 h (CLAIMED, s6f/sim_s6f_runtime.py) |
| 57174021 | V224-FLIP | 10-14 08:08 (13:38) | 4h26 - 4h39 | 2 pollers 4.57-4.65 h (CLAIMED, sim_s6f_runtime.py header) |
| 57200578 | FLIP + r224 mirror | 10-15 00:55 (06:25) | 4h44 - 4h48 | chain running since 00:10 UTC |
| 57199082 | FLIP + K14 fleet (FF20) | 10-15 00:07 (05:37) | 5h44 - 5h48 | chain |
| 57201874 | S6F (FLIP + S6fall) | 10-15 01:43 (07:13) | about 6h36 - 6h40 | same chain instance, continuous (prev chain_1125_2 -> chain_1354_3); MEASURED bounds 4h11-6h40 |
| 57218523 | FLIP + S student | 10-15 05:56 (11:26) | 2h27 - 5h23 | loose |
| 57222884 | S6F + S | 10-15 08:24 (13:54) | 2h55 - 7h14 | loose |
| 57248084 | R6 (J) | 10-16 00:07 (05:37) | about 6h03 - 6h15 | watch_reads 600 s poll armed 07:10 IST (CLAIMED PROGRESS_16); MEASURED upper 6h15 |
| 57249503 | KC1 (R7 base + KC leg) | 10-16 00:53 (06:23) | 5h29 - 6h29 | watcher |
| 57250689 | COMBO R6 + KC1 | 10-16 01:40 (07:10) | 5h42 - 6h43 | watcher |
| 57262115 | **HW (best)** | 10-16 08:43 (14:13) | <= 6h33 | loose (the session restart killed the waiter) |
| 57264835 | S6W (R6 + S6 .3) | 10-16 10:05 (15:35) | 5h11 - 6h09 | snapshots |
| 57291203 | hw3 (HW weights + S6 .3) | 10-17 02:01 (07:31) | 3h16 - 6h03 | snapshots |
| 57291336 | R7 late (HW + R7 leg) | 10-17 02:06 (07:36) | 6h37 - 7h26 | snapshots; about +1 h over HW |
| 57310758 / 57313454 / 57316395 | probes wr45 / s615 / a25 | 10-17 12:18 / 13:52 / 15:28 (17:48 / 19:22 / 20:58) | PENDING at 16:31 UTC (4h12 / 2h39 / 1h02 elapsed) | MEASURED API poll 16:31 UTC |

Per class (MODEL, assuming same compute gives the same runtime within GPU noise):
- V224 base about 3.0-3.3 h. FLIP about 4.6 h. FLIP + mirror about 4.75 h. FLIP + fleet about 5.75 h. S6F about 6.6 h (this matched the S6F runtime model median of 6.6 h, CLAIMED PROGRESS_15).
- **R6/HW class (HW, S6W, hw3, wr45, s615, a25): 6.0-6.25 h.** All of these change weights only, so no compute changes (CLAIMED, build notes: 1 line in cells 14 / 18). R6 is faster than the S6F R7-flip base by about 0.4-0.6 h.
- KC leg about +0.25-0.5 h. R7 late leg about +0.5-1.3 h.
- Run-to-run spread within a class looks like +-10-15 min (MODEL: R6 vs hw3 vs S6W bounds overlap only at 6.0-6.1 h).

Time-of-day trend (MEASURED within the bounds above): R6-class latency is the same for submits at 00:07, 02:01, 08:43 and 10:05 UTC, and FLIP-class is the same at 00:55 and 08:08 UTC. **No scoring-queue effect is visible for submits between 00:00 and 10:05 UTC on 10-14..10-17.** We have no measurement yet for evening UTC submits (12:00-16:00 UTC). Tonight's probes give the first one.

Useful read mechanics (MEASURED, all snapshots): the public LB `teams[].submissionCount` counts COMPLETED submissions only (85 now; the 3 pending probes are not counted). `lastSubmissionDate` is the submit time of the newest COMPLETED submission (still R7 late's 02:06:56 UTC at 16:24 UTC). So one anonymous `fetch_api_lb.py` call both detects a completion (count 85 -> 86) and gives the best-id snapshot. If two complete inside one poll, the count jumps by 2 and the order is lost.

## 2. Tonight's expected completions (MODEL: R6 class 6.0-6.25 h; a "late" flag at +7.0 h)
| probe | submit IST | expected (6.0-6.25 h) | late flag (+7.0 h) |
|---|---|---|---|
| wr45 57310758 | 17:48:55 | 23:48 - 00:03 | after 00:48 -> scoring queue present |
| s615 57313454 | 19:22:20 | 01:22 - 01:37 | after 02:22 |
| a25 57316395 | 20:58:50 | 02:58 - 03:13 | after 03:58 |
- HANDOFF's "~00:30 / ~02:00 / ~03:30" is about 30 min later than this. That fits too, as a conservative estimate (MEASURED vs MODEL).
- If wr45 is still PENDING at 00:50 IST: treat it as evidence that the 10-17 GPU crunch (commit QUEUED 2-3 h) also hits scoring, and use the 3 h buffer column in section 5 (GUESS-level rule).

## 3. HW hidden-run margin (9 h limit = maxGpuRuntimeMinutes 540, MEASURED comp.json)
- HW latency <= 6h33 (MEASURED). That makes the runtime <= 6h33, since latency = queue + runtime + scoring, so >= 2h27 of margin to 9 h (MEASURED bound). MODEL runtime 6.0-6.25 h, so about 2.75-3.0 h of margin.
- In-kernel guards (MEASURED, session_2026-10-16/hw/kernel_v224r6hws6fj):
  - V224 stage driver deadline 8.3 h (cell 7).
  - R224 skips itself if fewer than 40 min would remain before 9 h (cell 12).
  - S6F: G_START_H 5.75, BUDGET_H 3.3, DEADLINE_H 8.2, hard kill (cell 17). A skip fires the canary.
- S6F start margin: in S6F 57201874 the FLIP base ended about 4.6 h in and the total was about 6.6 h, so the S6F stage takes about 2.0 h (MODEL). If HW's S6F stage is also about 2.0 h, HW's base ends about 4.0-4.25 h in, which leaves about 1.5-1.75 h before G 5.75 h. A session would need to be about 1.35-1.4x slower to skip S6F (MODEL).
- The canary did not fire in HW's scored run. HW read positive (best-id moved over R6), and a fired canary degrades a column and would have shown as a score drop (MEASURED read + MODEL).
- Consequence: no margin risk for HW or R6 as final picks, because nothing is re-run (section 0.1). Any NEW HW-class entry carries the same about 3 h margin to 9 h and about 1.6 h to the S6F guard (MODEL).

## 4. Stuck runs, failures and queue reports (forum, CLAIMED; raw = research_1hr/forum_raw/all.txt)
- 748403 (10-11, GM vladimirsydor): one submission "stuck in a running state for over 24 hours". It was already in the submitting phase, so the user could not cancel it. Staff (herbison) replied only on 10-14 ("Can you check now?"). About 3 days to fix. This is a rare tail risk (CLAIMED, 1 report in 53 topics).
- 749623 / 749798 (10-15 / 10-16): hidden rerun "Notebook Threw Exception" while the commit run works. Deotte suggests committing a 1,300-train-study inference to catch NaNs and malformed DICOM. Not us: all our reruns completed (MEASURED, 17/17 COMPLETE in the API list).
- 730709 (10-14): a user with zero GPU quota could not save (commit), so could not submit. Commits need quota (CLAIMED). Scoring reruns are believed not to use user quota (GUESS; our latency does not depend on quota).
- Turnaround claims: nartaa 746792, 7.2 min for the efficiency entry and 26.1 min for the mirrored accuracy entry, "including queueing". kmn 735304, < 45 min. Dread Development 737696 (08-29), "around 9 hours for the full test set" for their 94-window pipeline. All CLAIMED. They mainly show that scoring queues were short when those posts were written.
- Efficiency LB notebook output (ryanholbrook, file created 10-16 21:22 UTC; pulled to pulled/timing/efflb, untrusted, read only): it lists our entry as S6F 57201874 (.957), efficiency rank 4,670 / 5,985. That fits our long runtime, since the efficiency score is mostly runtime: .001 AUC is worth about 70 s (MODEL from the formula). It lags at least 1 day (it shows S6F, not HW) (CLAIMED/MEASURED file content).
- R1 read-loss risk tonight (MODEL): the API best-id is cumulative. If wr45 completes and no snapshot is taken before s615 completes (gap about 1h20-1h35), and s615 then moves the best-id, wr45's read vs HW is lost. The same holds for s615 vs a25. An awake poller at <= 30 min (api_chain polls every 180 s) avoids this. The laptop has slept through scheduled times before: on 10-17, WakeToRun did not wake it (CLAIMED HANDOFF section 5).

## 5. Timing plan 10-18..10-22 (MODEL unless marked)
Constants:
- A15 cutoff = 10-22 18:00 IST = 12:30 UTC (CLAIMED PREREG A15).
- Deadline = 10-22 23:59 UTC = **10-23 05:29 IST** (MEASURED comp.json). Final selection is in the UI by Eesh before then.
- Daily reset 05:30 IST (MEASURED: UTC day). 5 slots per UTC day. 5 days left = 25 slots.
- Latency L by class: HW-class 6.25 h (upper of 6.0-6.25); S6F/R7-flip-base 6.7 h; R7-late class 7.45 h (MEASURED upper).
- Commit run 9-11 min plus a QUEUED wait of 0-3 h (MEASURED 10-17). Reads appear on the API LB at once (MEASURED: chain snapshots 90 s after detection already show the new count).

**Latest safe submit for a read before the A15 cutoff on 10-22:**
| class | L | buffer 1.5 h (no queue seen tonight) | buffer 3 h (queue seen tonight) |
|---|---|---|---|
| HW-class one-liner | 6.25 h | **10:15 IST** (04:45 UTC) | **08:45 IST** |
| S6F / R7-flip base | 6.7 h | 09:45 IST | 08:15 IST |
| R7-late class | 7.45 h | 09:00 IST | 07:30 IST |
- The 10-22 UTC day opens at 05:30 IST, so the 10-22 readable window is about **05:30-10:15 IST**: room for up to 5 blind HW-class submits about 45 min apart (05:30, 06:15, 07:00, 07:45, 08:30). Their reads land at about 11:45-14:45, with no follow-up possible.
- An entry submitted after about 10:15 IST on 10-22 is very likely unread at 18:00 and cannot become pick A under A15 (MODEL). Do not plan any.
- Whether a submission still PENDING at the deadline can be selected or scored is unknown (GUESS). Under A15 it does not matter.

**Decision cycles** (one read takes about 6.1-6.25 h; a follow-up needs a pre-committed kernel): about 3 per day when someone or something is awake.
- 10-18..10-21 template: submit 05:30 -> read about 11:45 -> follow-up 11:50 -> read about 18:05 -> follow-up 18:10 -> read about 00:20 -> follow-up submits at 05:30 the next UTC day.
- 10-21's last useful evening submit: by about **23:00 IST 10-21**. It reads about 05:15 IST 10-22, so its follow-up can go at 05:30 IST 10-22 (new UTC day) and read by about 11:45 IST 10-22, inside the cutoff.
- With the 3 h buffer, move that to about 21:30 IST.

**Commit timing:**
- On 10-17 commits waited up to about 3 h QUEUED, and a later push ran first (wr45), so this is Kaggle-side GPU capacity, not our 2-session cap (MEASURED order + GUESS cause).
- Push each kernel at least 4 h before its intended submit. Push all 10-22-morning candidates by about **22:00 IST 10-21**, so the 05:30-10:15 window is only gate + submit.
- GPU quota: commits use about 10-12 min of a 2xT4 session each (MEASURED), which is negligible. The weekly reset is believed to be Saturday about 05:30 IST: 10-17 was a Saturday, so the next reset is 10-24, after the deadline (CLAIMED HANDOFF "~05:50 IST" + GUESS weekly). Today's quota must cover all commits to 10-22. It easily does.

**Final selection:**
- Kaggle auto-selects if nobody selects (CLAIMED, rules 1.c "automatically selected by Kaggle"). Many entries tie at .958 (R6, HW, S6W, hw3, R7 late, maybe probes), so an auto-pick would probably take HW plus the next-highest hidden .958, not necessarily R6 (GUESS).
- **Eesh must select both picks by hand.**
- The A15 read at <= 18:00 IST leaves about 11.5 h before the 05:29 IST 10-23 deadline (MEASURED arithmetic).

**Overnight reads (tonight and on later nights):**
- Keep one poller alive across expected completion windows at <= 30 min (180 s is better), so that each completion gets its own snapshot.
- One anonymous `fetch_api_lb.py` per poll both detects completion (team submissionCount +1) and records best-id (MEASURED mechanics).

## 6. Sources used (all read only)
- API: `competition_submissions` (20 newest, statuses, UTC submit times), polled 16:31 UTC. LB snapshots session_2026-10-14/research960/lbprobe/api_lb_*.json (35) + research_1hr/api_lb_1017T_lens.json.
- Logs: session_2026-10-14/submits.log, api_chain.log, status*.log; session_2026-10-17/probes/{run_probes,finish_queued}.log, out_*_v1.log, running_*.txt; PROGRESS_14..17; HANDOFF top block; session_2026-10-15/s6f/sim_s6f_runtime.py header.
- Kaggle pages (anonymous PageService/ListPages + GetCompetition) saved to research_1hr/pulled/timing/ (page_*.txt, ListPages.json, GetCompetition.json). Efficiency LB output in pulled/timing/efflb (untrusted, parsed as text only).
- Forum: research_1hr/forum_raw/all.txt (threads 748403, 749623, 749798, 730709, 746792, 735304, 737696).
