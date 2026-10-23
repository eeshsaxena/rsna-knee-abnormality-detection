# ops.md: 10-22 operational audit (10-21 ~21:40 IST, Claude, ops track). RESEARCH ONLY.

Nothing was submitted, pushed, committed, downloaded, registered, scheduled or posted. PREREG, HANDOFF, PROGRESS, the live folders and
the scheduled tasks were only read. No box, no OAI data, no GPU. Kaggle calls were read-only (submission list, the SELECTED group,
submission limits, kernel status). New files are all in this folder and start with `ops_` (other tracks write here too).
**Tags.** MEASURED = read or computed now from a named source. CLAIMED = an older note or someone else says so (named). MODEL =
arithmetic on stated inputs. GUESS = judgement.

## 0. Bottom line

1. **Nothing is selected on Kaggle right now** (MEASURED 15:46 UTC: `ops_sel_now.py`, SubmissionGroup SELECTED = 0 of 108). If nobody
   selects, Kaggle auto-selects (MEASURED rules 18.c) the top-2 public (CLAIMED): swide .960 plus a .959 c4 / E-OA near-twin, with no
   R6-base hedge. That costs about -1.8 step / -1.0 smooth silver pts vs B = R6 (pw25 as B), -2.7 / -1.5 (pw30 as B) (MODEL, bpick
   2.1 regrets). **Zero-cost fix tonight: Eesh selects the standing pair A = swide 57464326 + B = R6 57248084 now**, then swaps B on
   10-22 only if an r6pw20 copy qualifies.
2. **Timeline:** 10-21 UTC day is full (MEASURED limits endpoint: num_allowed_now 0, num_today 5). The 10-22 UTC day opens **05:30:00
   IST** (00:00 UTC). Copies: c1 05:31, c2 06:15, optional c3 07:00 IST. Last submit 09:00 IST (worst read 17:30 IST at the 8.5 h
   10-21 latency). Selection 18:00-22:00 IST 10-22, hard stop 04:30 IST 10-23. **Deadline 10-23 05:29:59 IST = 10-22 23:59:59 UTC.**
3. **Mechanism: a draft tick is built and tested here** (`ops_tick_a23.py`, self-test PASS 21/21, read-only dry run at a simulated
   05:31 picks c1 with the live status COMPLETE). It pins kernel_version 1 and has no push stage. It fixes the two traps a naive copy of
   `val/tick_a22.py` would hit: (a) the A22 dedupe (slug in submits.log, prefix 'A22-R6PW20:' on Kaggle) would block every resubmit
   forever; (b) a new log dir has no push log, so the A22 logic would push v2. By hand works too (sec 2.3).
4. **Biggest operational risk is the laptop, not Kaggle.** MEASURED System log: an **unexpected shutdown at 10-21 00:09:25 IST, back
   only at 05:17:57 IST** (event 6008 / 41), i.e. 5.1 h dark across exactly tomorrow's submit window. Windows Update active hours are
   08:00-02:00 (MEASURED), so an update restart is allowed 02:00-08:00. The task is Interactive-logon (MEASURED on RSNA_A22_tick): after
   a reboot it does not run until someone logs in. Wake timers work on AC only (MEASURED). Keep it on AC, lid open, logged in, no heavy
   jobs; Eesh sets a ~06:00 IST phone check with the by-hand fallback.
5. **Value at stake (MODEL, LASTDAY_1021 sec 1):** P(>= 1 clean copy) ~.55-.64 (1 copy) / ~.75-.79 (2) / ~.83-.86 (3). B = R6: silver
   .876 / .763 (step / smooth); B = clean r6pw20: .882 / .776; 2 copies in expectation .881 / .773. A failed copy costs nothing (B
   stays R6). Forgetting to select costs more than the whole resubmit plan gains (item 1).

## 1. Timeline (machine clock MEASURED +2-3 s vs the kaggle.com Date header)

| IST | UTC | what | who | source |
|---|---|---|---|---|
| 10-21 21:40 (now) | 16:10 | 0 selected; 0 slots left today; r6pw20 kernel status COMPLETE; local nb e4d92c0a / meta 3a5b8b0f = registry | - | MEASURED |
| 10-21 22:00 | 16:30 | MOACF yes / no (default no) | Eesh | LASTDAY 3.1 |
| 10-21 by 23:00 | by 17:30 | GO on c1 + c2 (+ c3), A23 text, mechanism (tick or by hand); **standing pair selected (sec 4)** | Eesh | LASTDAY 3.1 + this file |
| 10-21 23:00-00:30 | 17:30-19:00 | main: PREREG A23, A23_GO.txt, self-test, dry run, register RSNA_A23_tick (sec 2.2) | main | - |
| 10-22 05:20 / 05:25 | 23:50 / 23:55 (10-21) | RSNA_A22_tick submit window closes / task EndBoundary, then it self-deletes (DeleteExpiredTaskAfter PT0S) | - | MEASURED task XML |
| 10-22 05:30:00 | 00:00:00 | 10-22 UTC day opens: 5 slots for the team | - | MEASURED (rules 2.2; our caps on 10-14..10-21) |
| 05:31 | 00:01 | c1 = r6pw20 v1, copy 1 | tick or Eesh | LASTDAY |
| 06:15 | 00:45 | c2 (tick: also >= 30 min after c1's actual submit) | tick or Eesh | LASTDAY |
| 07:00 | 01:30 | optional c3 | tick or Eesh | LASTDAY |
| 09:00 | 03:30 | last submit of anything (tick CLOSE) | - | LASTDAY |
| ~11:00 / 14:00 / 16:00 / 17:30 | 05:30 / 08:30 / 10:30 / 12:00 | reads (status, display, .916 canary flag) | main | LASTDAY |
| 18:00-22:00 | 12:30-16:30 | final selection (sec 4), verified by `ops_sel_now.py` | Eesh + main | - |
| 10-23 04:30 | 10-22 23:00 | hard stop: select with whatever is COMPLETE; a PENDING copy counts as failed | Eesh | LASTDAY |
| **10-23 05:29:59** | **10-22 23:59:59** | **Final Submission Deadline; selection locks** | - | MEASURED (comp Timeline: "11:59 PM UTC") |

Read windows (MODEL on latencies 4.8-7.1 h A21 CLAIMED, 5.7-6.5 h typical CLAIMED think/runtime.md, <= 8.5 h 10-21 MEASURED):

| copy | submit IST | earliest | typical | worst (8.5 h) |
|---|---|---|---|---|
| c1 | 05:31 | 10:19 | 11:13-12:01 | 14:01 |
| c2 | 06:15 | 11:03 | 11:57-12:45 | 14:45 |
| c3 | 07:00 | 11:48 | 12:42-13:30 | 15:30 |
| last allowed | 09:00 | 13:48 | 14:42-15:30 | 17:30 |

Physical last submit that could still score before the deadline: ~14:30 UTC = 20:00 IST 10-22 (9 h cap + ~0.5 h queue, MODEL). Not
in the plan: it would push the selection into the early morning of 10-23.

## 2. How the r6pw20 resubmit is executed

What is resubmitted (MEASURED): `eeshsaxena/rsna-sub-v224r6flips6fpw20j`, **version 1** (the only push: val/a22/push log
"Kernel version 1", kernel last_run_time 10-21 03:07 UTC), output `submission.csv`, gate val/a22/gate_v224r6flips6fpw20j.txt last
line `PROBE_CHECK PASS 17/17` (10-21 09:00:50 IST). Same version as 57463719 (.916 canary). No push, no commit, no GPU quota.

Same-version resubmits: never done by us (MEASURED: 0 duplicate slug+version lines in submits.log, 43 lines). Other comps show Kaggle
accepts and reruns them fresh (CLAIMED forum: an exact same-version resubmit scored differently; an accidental same-version duplicate
was scored on its own). GUESS ~.95 it is accepted. RSNA has no pending-submission limit (MEASURED: 5 entries pending at once on 10-21,
03:30-04:11 UTC, all accepted); another comp got a 1-pending limit in October (CLAIMED Gemma forum), GUESS < .05 it appears here.

### 2.1 Option A: the A23 tick (recommended; built here, not registered)

- `ops_tick_a23.py` (+ wrapper `ops_tick_a23.cmd`). Guards G1-G8 are in its header. Key facts:
  - GO file `session_2026-10-21/final/A23_GO.txt`, first line `GO c1 c2 [c3] MAX=3 Eesh <time>: "<quote>"`. Copies must be a prefix
    of c1 c2 c3 (a GO for c2 alone is invalid). Unknown key -> the whole GO is invalid.
  - PREREG must contain `AMENDMENT 23` (draft text: LASTDAY_1021 sec 3.3).
  - Static guard: local nb / meta sha = registry, A22 push log lists only v1, A22 gate last line = PASS 17/17.
  - Live guard: `kernels_status` must be COMPLETE (anything else = someone pushed again -> STATUS_ODD, no submit).
  - Dedupe per copy, from a fresh Kaggle list each tick: description prefix `A23-R6PW20C1:` / `C2:` / `C3:` or `A23 c<n> SUBMITTED`
    in submits.log. The A22 line for the same slug does not block (self-test).
  - Not-before c1 05:31:00, c2 06:15, c3 07:00, and each copy >= 30 min after the previous one's logged submit; CLOSE 09:00:00 IST.
    Strict order. A late wake (e.g. 07:40) still submits c1, then c2 not before 08:10 (self-test).
  - Cap: `competition_get_submission_limits().num_allowed_now` >= 1 (MEASURED endpoint exists; now 0 / num_today 5) AND used this UTC
    day < MAX. One submit per tick. A submit exception is retried next tick; the fresh list stops a double submit if it did land.
  - `--selftest`: offline, PASS 21/21 (`ops_a23_selftest.log`). `--dry 2026-10-22T05:31:00`: read-only API calls, never submits.
    MEASURED now: `kernel status COMPLETE | copies done - | next copy c1 | used on UTC day 2026-10-22 = 0 (MAX 3) | num_allowed_now
    (live, real clock) 0 | static ok`; at 05:30:59 it prints "before 05:31:00: nothing to do".
  - **Do not run it with `-I`**: kaggle 2.2.4 lives in the user site-packages (MEASURED: import fails under -I). The self-test is -I safe
    because it never imports kaggle.

### 2.2 What main does after Eesh's GO (in this order)

1. Append AMENDMENT 23 to PREREG (LASTDAY sec 3.3 text, plus "copies via RSNA_A23_tick, kernel_version 1, descriptions
   A23-R6PW20C<n>").
2. Write the GO file, e.g. `GO c1 c2 MAX=3 Eesh 10-21 23:00: "<quote>"` (add c3 only if Eesh wants it).
3. Run `C:\Python314\python.exe -I session_2026-10-21\final\ops_tick_a23.py --selftest`. Expect FAIL on exactly the two "currently
   absent" checks (PREREG guard now True, GO file now present) and OK on the other 19. Then `--dry 2026-10-22T05:31:00`: expect
   `next copy c1`, `static ok`, status COMPLETE.
4. Register the task (PowerShell; mirrors RSNA_A22_tick's MEASURED settings: WakeToRun, StartWhenAvailable, IgnoreNew, self-delete):
   ```
   $a  = New-ScheduledTaskAction -Execute 'C:\Users\eeshs\Downloads\rsna-knee\session_2026-10-21\final\ops_tick_a23.cmd'
   $t1 = New-ScheduledTaskTrigger -Once -At '2026-10-22 05:25:00' -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Hours 4)
   $t2 = New-ScheduledTaskTrigger -Once -At '2026-10-22 05:31:15'
   $t1.EndBoundary = '2026-10-22T09:30:00'; $t2.EndBoundary = '2026-10-22T09:30:00'
   $s  = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -DeleteExpiredTaskAfter (New-TimeSpan -Seconds 0)
   Register-ScheduledTask -TaskName RSNA_A23_tick -Action $a -Trigger $t1,$t2 -Settings $s
   ```
5. Check `Get-ScheduledTask RSNA_A23_tick | Get-ScheduledTaskInfo` shows NextRunTime 10-22 05:25. Leave RSNA_A22_tick alone: it has
   nothing left to submit (all five A22 descriptions exist) and deletes itself after 05:25.
6. Kill switch: delete `A23_GO.txt` (the tick then does nothing), or `Unregister-ScheduledTask RSNA_A23_tick -Confirm:$false`.
7. 10-22 ~05:40 IST: read `final/a23/tick_a23.log` for `A23 c1 SUBMITTED ... "ref": <id>`. The ids of c1 / c2 / c3 go into the
   selection card (sec 4).

### 2.3 Option B: by hand (fallback, or if no tick is registered)

From `C:\Users\eeshs\Downloads\rsna-knee` in PowerShell, at or after 05:31 IST:
```
C:\Python314\python.exe session_2026-10-21\final\ops_tick_a23.py --dry 2026-10-22T05:31:00
C:\Python314\python.exe -m kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-v224r6flips6fpw20j -v 1 -f submission.csv -m "A23-R6PW20C1: resubmit r6pw20 v1 unchanged (57463719 hit the S6F canary .916); B hedge copy 1"
C:\Python314\python.exe session_2026-10-21\final\ops_subs_now.py 4
```
- The dry run must say `next copy c1` and, after 05:30, `num_allowed_now 5` (or what is left). The submit prints "N submissions
  remaining today" on success (MEASURED CLI source). `ops_subs_now.py` shows the new id at the top.
- c2 / c3: the same command with `C2:` / `C3:` and "copy 2" / "copy 3".
- Always pass `-v 1`: without it the server takes the latest version (MEASURED SDK: kernel_version is only sent when given). Always
  `-f submission.csv`. Never `kaggle kernels push` this folder again (it would make v2 and a new commit).
- Use the exact `A23-R6PW20C<n>:` prefixes, so a tick that is also running sees the copy and does not submit it twice.
- Optional log line (the tick's spacing rule reads it): `[10-22 HH:MM:SS IST] A23 c1 SUBMITTED eeshsaxena/rsna-sub-v224r6flips6fpw20j
  v1: {"ref": <id>}` appended to session_2026-10-14/submits.log.
- From the browser instead (GUESS on exact labels): competition page -> Submit Prediction -> Notebook `rsna-sub-v224r6flips6fpw20j`,
  Version 1, output `submission.csv`, the same description -> Submit. This also works from another PC if the laptop is dark.

## 3. If the team selects nothing

- Rules (MEASURED comp_pages.txt): 2.b "You may select up to two (2) Final Submissions"; 18.c a Final Submission is the one "selected
  by the user, or automatically selected by Kaggle in the event not selected by the user". The Efficiency page refers to the
  auto-selection "conditions described in the My Submissions tab".
- How auto-select picks (CLAIMED, hoshinoyumemi forum 10-15 via FORUM_DIGEST_10-15b; matched Eesh's 10-15 screenshot): the two highest
  **unrounded** public scores; exact ties go to the earlier submission.
- Our state (MEASURED 15:46 UTC): 0 selected. Auto pair today = swide 57464326 (.960) + the highest unrounded .959 among pw25
  57463742, pw30 57425970, pw20 57425269, eoa20 57413368, eoa30 57425223, eoa10 57413380 (all c4 + E-OA). Which one is not visible
  (the API rounds to 3 dp). A 10-22 r6pw20 copy at .959 could also land second (GUESS: below swide always).
- Why that is bad (MODEL bpick 2.1, 19 rows): the twin B adds almost no cover for the E-OA dose tail, which is what moves silver.
  Mean silver regret (step / smooth pts): R6 .25 / .26, pw25 2.05 / 1.23, pw30 2.91 / 1.77. So auto-select costs ~-1.8 / -1.0
  (pw25) to ~-2.7 / -1.5 (pw30) vs B = R6, and another ~-0.6 / -1.3 vs a clean r6pw20 B.
- Selecting only ONE: GUESS Kaggle does not fill the second slot (auto-select applies only when nothing is selected). Always select 2.
- Memory: a wrong-notebook selection already zeroed a past competition (AI Agent Security, CLAIMED MEMORY.md).

## 4. Final selection: step by step for Eesh

**Tonight (zero cost, removes the auto-select and oversleep risks):** select the standing pair A = swide 57464326 + B = R6 57248084
with steps 2-7 below. On 10-22 only B may change.

**Selection card** (main fills the B row on 10-22 from the reads; ids MEASURED from the API list now):

| pick | id | description starts with | submitted UTC / IST | public |
|---|---|---|---|---|
| **A** | **57464326** | `STACK S_wide: pw30 with A_OA .25 + Later` | 10-21 03:51:44 / 09:21:44 | .960 |
| **B default** | **57248084** | `V224R6-FLIP-S6F-J: J twin of the best (S` | 10-16 00:07:01 / 05:37:01 | .958 |
| B if a copy qualifies | (from the tick log) | `A23-R6PW20C1:` (or C2 / C3) | 10-22 00:01 / 05:31 (c1) | >= .958, not .916 |

B rule (A22 + draft A23): B = the earliest-submitted copy (lowest id) that is COMPLETE, is not .916 and displays >= .958; fixed at the
first read where one qualifies (a later copy never replaces it); else R6 57248084. A never changes (no 10-22 entry can be A unless a
MOACF GO says otherwise).

**Do NOT select (look-alikes):** pw25 57463742 (.959, `A22-PW25:`, one row from swide); salt 57464911 (.916, `STACK S_alt:`, next to
swide); spl 57464296 (.916, `STACK-SKEPTIC S_pl:`); the OLD r6pw20 57463719 (.916, `A22-R6PW20:`); any 10-22 copy that is PENDING,
ERROR or .916; any guard-variant / MOACF row unless the card says so.

Steps (UI labels GUESS: the API exposes ALL / SUCCESSFUL / SELECTED groups, so the page likely has matching filters):
1. Wait for the selection read (main posts the filled card with both ids), from ~18:00 IST 10-22. Aim to finish by 22:00 IST.
2. Log in as eeshsaxena; open https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/submissions (API slug, MEASURED;
   the overview pages also link a `...abnormalities-detection` spelling).
3. Show all submissions. R6 is from 10-16, so it sits 25+ rows down (108 rows, 20 per API page; scroll or "load more").
4. Find A by description + date + .960. Tick its select box.
5. Find B by description + date + score from the card. Tick its select box. If a different row was ticked before (e.g. the standing
   R6), untick it so that exactly 2 are ticked.
6. Switch to the Selected filter: exactly the two card rows, both COMPLETE (no PENDING).
7. Send a screenshot. main runs `C:\Python314\python.exe session_2026-10-21\final\ops_sel_now.py 57464326 <B id>` and must see
   `SELECTED_COUNT 2` and `SELECTION_MATCHES_ARGS True`. Re-check once after ~30 min.
8. Ask swapiy and kmrsuman not to open the Submissions page or submit on 10-22 (GUESS: any member can change the team selection;
   swapiy has 11 team submissions, last 10-09, MEASURED).

## 5. What could go wrong (checklist)

| # | risk | evidence | size / P | mitigation |
|---|---|---|---|---|
| 1 | Laptop off / crashed at 05:31 | MEASURED event 6008/41: unexpected shutdown 10-21 00:09:25 IST, back 05:17:57; CLAIMED memory: thermal shutdowns (100 C), 14 surprise shutdowns in 6 days in Sept | GUESS .1-.2 for a night | AC, lid open, no heavy jobs, good airflow; StartWhenAvailable + late-wake logic submits up to 09:00; Eesh checks ~06:00 IST and submits by hand (2.3), from any PC |
| 2 | Asleep / standby | MEASURED: Sleep-after never (AC + DC); hibernate-after never on AC; wake timers AC only (DC 0); Modern Standby S0 | GUESS low on AC; S0 wake by timer not guaranteed | stay on AC; do not close the lid (lid action not readable, GUESS default sleep) |
| 3 | Windows Update restart overnight | MEASURED active hours 08:00-02:00 (restart allowed 02:00-08:00); no reboot pending now | GUESS < .05 | Eesh may pause updates for 1 day (Settings > Windows Update); after any reboot someone must log in |
| 4 | Signed out / rebooted without login | MEASURED RSNA_A22_tick LogonType Interactive (A23 registered the same way) | total for the tick | lock the screen, never sign out; by-hand fallback |
| 5 | Submit before 05:30:00 bounces | MEASURED num_allowed_now 0 today; clock +2-3 s | 0 with the tick (not-before 05:31:00) | by hand: wait for 05:31 and run the dry run first |
| 6 | Slots eaten | 5 per team per UTC day (MEASURED rules 2.2 + limits endpoint); teammates can submit | low | tell swapiy / kmrsuman; tick checks num_allowed_now; plan uses 2-3 of 5 |
| 7 | Pending-submission limit introduced | MEASURED none on 10-21 (5 pending at once); CLAIMED other comp got one | GUESS < .05 | c1 still goes; c2 / c3 would be refused (logged as SUBMIT ERR); B falls back to R6 if c1 fails |
| 8 | Same-version resubmit refused | never tried by us (MEASURED); CLAIMED accepted elsewhere | GUESS ~.05 | log shows SUBMIT ERR; then B = R6 (a push of v2 would need a commit, gate and a new A23 clause: not planned) |
| 9 | S6F canary again (.916) | 3/5 on 10-21, 0/27 before (MEASURED cause.md); per copy P(clean) ~.55 [.35-.80] (GUESS) | 2 copies P(>=1 clean) .75-.79 (MODEL) | copies 2-3; a canary costs nothing, B stays R6 |
| 10 | 9 h timeout -> ERROR | guards end S6F by NB_T0 + 8.2 h, R6 by 8.3 h, r224 skips under 40 min left (MEASURED code, cause.md); 7 timeouts 09-07..10-03, none since guards (MEASURED) | GUESS < .02 | none needed: ERROR = canary for B |
| 11 | Late read | latency <= 8.5 h on 10-21 (MEASURED) | c3 worst 15:30 IST | nothing after 09:00; a PENDING copy at the hard stop counts as failed |
| 12 | Wrong version / file | SDK drops the version when it is not given -> latest (MEASURED source); latest = v1 today | 0 if `-v 1 -f submission.csv` | tick pins VER 1 and checks kernel status; never re-push the folder |
| 13 | Double submit (tick + by hand) | tick dedupes on the Kaggle description prefix (self-test) | 0 with the same prefix | by hand: same `A23-R6PW20C<n>:` prefix; check `ops_subs_now.py` first |
| 14 | API / SSL errors | MEASURED SSLError in tick_a22.log 10-21 13:22 | per tick | the tick retries every 5 min; by hand: re-list before re-submitting |
| 15 | Naive port of tick_a22 | its dedupe would skip forever; a new dir would push v2 (MEASURED code l.137, l.143-152) | total / costly | use ops_tick_a23.py (both removed) |
| 16 | `-I` in the wrapper | MEASURED: kaggle import fails under -I | total for the tick | wrapper runs without -I |
| 17 | Nothing selected / only one selected | MEASURED 0 selected now; rules 18.c | -1.0 to -2.7 silver pts (MODEL, sec 3) | standing pair tonight; final card 10-22 |
| 18 | Wrong row selected | look-alikes: pw25 .959, salt `S_alt` .916 next to swide `S_wide` .960, old r6pw20 .916 (MEASURED list) | total for that pick | card with id + description + date + score; `ops_sel_now.py` must print SELECTION_MATCHES_ARGS True |
| 19 | Selecting a PENDING copy | Kaggle allows ticking pending rows (CLAIMED risk.md F9) | total for B if it fails | only COMPLETE rows; hard stop 04:30 IST with R6 |
| 20 | Teammate changes the selection | GUESS any member can | low, total | ask them not to touch; re-check `ops_sel_now.py` ~30 min later and once near 04:30 IST |
| 21 | Kaggle outage near the deadline | GUESS low; heavy final-day load (CLAIMED staff queue notes) | total if unselected | standing pair tonight; final selection by 22:00 IST (7.5 h buffer) |
| 22 | Identity verification banner | requiresIdentityVerification true (MEASURED 10-15 digest) | GUESS prize-only | Eesh completes it himself if a banner appears |
| 23 | Kaggle token | API calls worked at 15:46 UTC (MEASURED) | low | the dry run at 23:00 re-checks it |

## 6. Eesh's 2-minute checklist before sleeping (10-21)

1. Standing pair selected: swide 57464326 + R6 57248084 (sec 4 steps 2-7); screenshot to main.
2. GO (or NO-GO) on c1 + c2 (+ c3) and the mechanism; if by hand, set an alarm for 05:35 IST.
3. Laptop: on AC, lid open, locked (not signed out), nothing heavy running; optionally pause Windows Update for a day.
4. Phone: Kaggle logged in; at ~06:00 IST check the Submissions page for an `A23-R6PW20C1` row. If it is missing, submit by hand
   (sec 2.3) from any PC.
5. Teammates told: no submits and no selection changes on 10-22.

## 7. Files (this folder)

- `ops.md` (this file).
- `ops_tick_a23.py` (draft tick; `--selftest` offline, `--dry <ISO IST>` read-only), `ops_tick_a23.cmd` (wrapper, no -I),
  `ops_a23_selftest.log` (PASS 21/21).
- `ops_sel_now.py` (read-only: SELECTED group + id lookup; `ops_sel_now.py <A> <B>` verifies the final pair), `ops_subs_now.py`
  (read-only: newest submissions with id / date / status / score / description).
- Sources: LASTDAY_1021.md, CANARY_1021.md, canary/cause.md, canary/bpick.md, day/plan1022.md, val/tick_a22.py + .cmd + a22/*,
  session_2026-10-15/jpeg/comp_pages.txt (rules), RULES_AND_HOST_COMMENT_10-08.md, silver/FORUM_DIGEST_10-15b.md, submits.log,
  Windows Task Scheduler / powercfg / System event log (read-only).
