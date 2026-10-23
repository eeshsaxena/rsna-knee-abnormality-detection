# api.md: read-only Kaggle API recheck (10-21 17:53 UTC = 23:23 IST, Claude, recheck track)

Nothing was submitted, pushed, selected, committed, downloaded, scheduled or posted. Calls: competition_submissions (ALL and
SELECTED), competition_get_submission_limits, competitions_list, kernels_status, kernels_list, kernels_list_files (listing only),
anonymous web LeaderboardService (fetch_api_lb.py), anonymous HTTP GET of two kernel pages (status code only), plus a read-only
Get-ScheduledTask. No OAI data, no GPU. Tags: MEASURED = read now from the named call; CLAIMED = older note; GUESS = judgement.

Files here: `api_recheck.py` (the read-only script), `api_recheck.out` (its output), `subs_recheck.csv` (all 108 rows),
`lb_recheck.json` (LB snapshot 17:53:54 UTC).

## 0. Flags (read these first)

1. **STILL 0 SELECTED at 17:53 UTC / 23:23 IST** (MEASURED, SubmissionGroup SELECTED = 0). FINAL_1021 D1 ("select A + B now",
   asked at ~21:45 IST) has not happened yet. If it stays at 0, Kaggle auto-picks (CLAIMED ops.md) top-2 public = swide .960 plus a
   .959 c4/E-OA twin, with no R6 hedge. Zero-cost fix: Eesh ticks swide 57464326 + R6 57248084 now; then
   `py session_2026-10-21/final/ops_sel_now.py 57464326 57248084` must print SELECTED_COUNT 2 and SELECTION_MATCHES_ARGS True.
2. **The task's clock is stale.** The computed brief says NOW ~21:45 IST; the real time at the API calls was 23:23 IST (MEASURED,
   `date -u` 17:52 UTC and API fetched_utc 17:53:25). By the ops timeline the D3 GO/NO-GO (by 23:00 IST) is now past due, and the
   main-session setup window (PREREG A23, A23_GO.txt, self-test, register RSNA_A23_tick) is 23:00-00:30 IST. MEASURED: no A23 text
   in session_2026-10-15/PREREG_10-15.md yet (last is AMENDMENT 22), no A23_GO.txt in session_2026-10-21/a23/, and no RSNA_A23
   scheduled task exists (only RSNA_A22_tick). Consistent with "not yet approved"; just note the slack is shrinking.
3. **Deadline seconds.** The API says deadline `2026-10-22 23:59:00` UTC (MEASURED competitions_list), i.e. 10-23 05:29:00 IST, not
   05:29:59 as ops.md / FINAL_1021 state. 59 s; irrelevant under the 04:30 IST hard stop, but the text should say 23:59:00.
4. **Minor wording.** "All 108 COMPLETE, 0 ERROR" is true (MEASURED), but 7 old COMPLETE rows (09-07 to 10-03, incl. 1 by swapiy)
   carry no public score. None is a pick candidate. No discrepancy on any named id.

No discrepancy found in any id, score, kernel status or day count vs FINAL_1021 / ops.md / submits.log.

## 1. All submissions (MEASURED, 17:53:25 UTC)

- ALL_COUNT 108 (identical id set to final/subs_all.csv at 15:44 UTC: nothing new since). Status counts: COMPLETE 108, 0 ERROR,
  0 PENDING. Team "Eesh saxena"; submitted_by eeshsaxena 97, swapiy 11 (swapiy's last on 10-09, none since).
- Top public: swide .960; then .959 x6 (pw25, pw30, pw20, eoa30, eoa10, eoa20); then .958 x15. Three .916 rows: r6pw20, spl, salt.
- Full table: `subs_recheck.csv`.

| name | id | date UTC | status | public | kernel (url) / desc prefix |
|---|---|---|---|---|---|
| swide (A) | 57464326 | 10-21 03:51:44 | COMPLETE | .960 | rsna-sub-v224r6c4pw25lomm05mcl10j, scriptVersionId 360633233, "STACK S_wide" |
| R6 (B) | 57248084 | 10-16 00:07:01 | COMPLETE | .958 | rsna-sub-v224r6flips6fj, scriptVersionId 358682285, "V224R6-FLIP-S6F-J" |
| pw25 | 57463742 | 10-21 03:31:45 | COMPLETE | .959 | "A22-PW25" |
| r6pw20 | 57463719 | 10-21 03:30:50 | COMPLETE | .916 | rsna-sub-v224r6flips6fpw20j, scriptVersionId 360627912, "A22-R6PW20" |
| spl | 57464296 | 10-21 03:50:51 | COMPLETE | .916 | "STACK-SKEPTIC S_pl" |
| salt | 57464911 | 10-21 04:11:01 | COMPLETE | .916 | "STACK S_alt" |
| pw30 | 57425970 | 10-20 06:31:11 | COMPLETE | .959 | "A21-PW30" |
| pw20 | 57425269 | 10-20 06:12:14 | COMPLETE | .959 | "A21-PW20" |
| eoa20 | 57413368 | 10-20 00:01:35 | COMPLETE | .959 | "A20-EOA20" |
| eoa10 | 57413380 | 10-20 00:01:40 | COMPLETE | .959 | "A20-EOA10" |
| c4 | 57343435 | 10-18 06:44:31 | COMPLETE | .958 | "A18-C4" |

All match FINAL_1021 / ops.md / submits.log (IST times there = UTC + 5:30). The .916 = S6F canary attribution is CLAIMED
(canary/CANARY_1021.md); the .916 score itself is MEASURED.

## 2. Day counts and limits (MEASURED)

- Per UTC day: 10-17 5, 10-18 5, 10-19 5, 10-20 5, **10-21 5 (5 of 5 used)**.
- `competition_get_submission_limits`: num_allowed_now **0**, num_today 5, num_total 108, limited_by_total False.
- competitions_list: max_daily_submissions 5, max_team_size 5, is_kernels_submissions_only True, submissions_disabled False,
  merger/new-entrant deadline 10-15 23:59, **deadline 10-22 23:59:00 UTC**, team_count 6328, user_rank 147.
- The 10-22 UTC day opens at 00:00 UTC = 05:30 IST; copies at 05:31 / 06:15 / 07:00 IST fit (3 of 5).

## 3. Final selection state (MEASURED)

`SELECTED_COUNT 0` (same call as ops_sel_now.py, run inside api_recheck.py at 17:53 UTC). See flag 1.

## 4. Kernel eeshsaxena/rsna-sub-v224r6flips6fpw20j

- kernels_status: **COMPLETE**, failure_message empty (MEASURED).
- kernels_list: last_run_time 2026-10-21 03:07:13 UTC (MEASURED). The fields is_private / current_version_number / id come back as
  proto defaults (False / 0 / 0 for all 100 of our kernels), so they say nothing (MEASURED). Anonymous GET of the kernel page = 404,
  so it is private (MEASURED).
- kernels_list_files: 33 output files, all created 10-21 03:07 UTC (single run), incl. submission.csv, canary_receipt.json,
  s6f_receipt.json (MEASURED, listing only, nothing downloaded).
- **Version 1:** the API exposes no version list. Evidence that v1 exists and is the only/latest version: on-disk push log
  session_2026-10-20/val/a22/push_v224r6flips6fpw20j.log = "Kernel version 1 successfully pushed" (MEASURED file, one push log);
  the 10-21 submission 57463719 points at scriptVersionId 360627912; last_run_time 03:07:13 equals the output-file timestamps, so
  no later version has run since (MEASURED). Conclusion: v1 exists, COMPLETE, and is latest (GUESS-strong, indirect). The local
  gate out_v224r6flips6fpw20j/gate = PROBE_CHECK PASS 17/17 (MEASURED file).

## 5. Leaderboard snapshot (MEASURED, anonymous LeaderboardService, 17:53:54 UTC)

- Our row: teamId 16799346, submissionId 57464326 (swide), **rank 147 / 6328, .960, SILVER** (unchanged rank vs 15:44 UTC, n was 6324).
- Bands: GOLD ranks 1-22 (.969-.965), SILVER 23-316 (.965-.958), BRONZE 317-632 (.958). Silver margin 169 ranks.
- Score ties: >.960 = 100 teams; .960 = 51 teams (ranks 101-151); .959 = 82 (152-233); .958 = 1522 (234-1755).
- Delta since 15:44 UTC: +4 teams, one team moved above .960 (99 -> 100), .958 band +14. Our rank did not move.

## 6. Scheduled task (MEASURED, read-only)

RSNA_A22_tick: Ready, LastRun 23:19:31 IST result 0, NextRun 23:29:30 IST, every 10 min, repetition duration 20 h from 08:39:30 IST
(last repeat ~04:29 IST 10-22), EndBoundary 10-22 05:25 local, DeleteExpiredTaskAfter PT0S, WakeToRun True, LogonType Interactive.
Its log (val/a22/tick_a22.log) shows the 5-slot A22 slate done by 09:41 IST; last line an SSL list error at 13:22. It cannot reach
the 10-22 UTC day (ends before 05:30 IST) and 10-21 has 0 slots, so it cannot submit anything more (MEASURED + GUESS). No RSNA_A23
task exists yet.
