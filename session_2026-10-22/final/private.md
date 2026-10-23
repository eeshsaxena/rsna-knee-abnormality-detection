# private.md: the 3 private slots on 10-22 (written 10-22 ~09:15 IST, Claude, private-entries track). RESEARCH + LOCAL BUILD ONLY.

Nothing was submitted, pushed, committed, selected, downloaded or registered. PREREG / HANDOFF / PROGRESS were only read. Kaggle calls
were read-only (submission list, limits, kernel status). No OAI data, no GPU, no forum fetch (another track covers forums).
Tags: MEASURED = read or computed now from a named source. CLAIMED = an older note or a forum post says so. MODEL = arithmetic on
stated inputs. GUESS = judgement.

## 0. Bottom line

1. **Use all 3 private slots on r6pw20 v1 resubmits (design D1), via `tick_a24.py`.** No commit, no push, no new gate.
   P(at least one clean B) **.73** on cause.md's regime mixture, **.86-.90** if copies are independent at P(clean) .55-.64 (MODEL).
   Private value vs B = R6: **+0.44 step / +0.96 smooth silver pts** (base, k .86); +0.15 / +0.44 with the skeptic's k (MODEL).
2. **Do not swap a copy for r6e20 (D2).** It loses to D1 on value in all 12 model rows (3 regime models x 2 value models x 2 commit-fail
   levels): -0.04 to -0.05 step / -0.03 to -0.05 smooth silver pts (MODEL). Its only edge is +.01-.02 P(any clean), because it hedges a same-version refusal. That refusal shows up within
   minutes of c1's submit, so it can be handled after the fact. r6e20 also needs a first push, a GPU commit on the final day and the
   A20 gate key.
3. **The guard variant g85j as copy 3 (D3)** beats D1 by only +0.00 to +0.05 silver pts (MODEL). That is below the model's
   resolution, and it costs a GPU commit, a gate run, a registry entry and PREREG text. Keep it as the **contingency**, used only if
   c1 is refused as a same-version resubmit (sec 3).
4. **Tick ready:** `session_2026-10-22/final/tick_a24.py` (+ `tick_a24.cmd`). Self-test **PASS 30/30** and an offline live-path
   simulation with a fake API **PASS 27/27** (MEASURED). The read-only dry run at 08:56 IST gave `kernel status COMPLETE | copies done
   - | next copy c1 (due) | used on UTC day 2026-10-22 = 0 (MAX 5) | num_allowed_now 5 | static ok` (MEASURED). The real
   submits.log was unchanged by every test (md5 e1731db3 before and after, MEASURED).
5. **Timeline at arm ~09:45 IST:** c1 ~09:45, c2 >= 10:15, c3 >= 10:45, CLOSE 14:30 IST. Reads at the 8.5 h worst latency come by
   ~18:15 / 18:45 / 19:15 IST (MODEL on MEASURED 10-21 latency). Selection runs ~19:30-22:00 IST.
6. **Arm ONE r6pw20 mechanism only.** The final track built a parallel tick in this folder: `ops_tick_a23_1022.py` (sha 5390c3c2;
   prefix `A23-R6PW20C<n>:`, GO `A23_GO.txt`, guard AMENDMENT 23, c1 09:30 / c2 10:00 / c3 10:30, CLOSE 13:30), and forum.md's
   by-hand command uses the A23 prefix (MEASURED, read). The two prefixes do not dedupe each other in the A23 tick. To guard
   against that, tick_a24 (a) counts any `A23-R6PW20C<n>:` row or `A23 c<n> SUBMITTED` line from the 10-22 UTC day as its own
   copy c<n>, and (b) submits nothing (CONFLICT) while any `A23_GO.txt` exists in session_2026-10-22/final or session_2026-10-21/final.
   Main picks one: **A24 (this file)** = arm-time c1, a 14:30 CLOSE, the API-time gap and cross-dedupe; or **A23_1022** = fixed
   09:30 start, CLOSE 13:30. They are functionally equivalent for value. Whichever is chosen, by-hand fallbacks must use THAT
   prefix.

## 1. State now (MEASURED 10-22 08:54-08:56 IST = 03:24 UTC)

- 0 submissions on the 10-22 UTC day; `num_allowed_now` 5. The newest row is 57464911 (10-21 04:11 UTC). A23 never ran: there is no
  A23 row, no A23_GO.txt and no AMENDMENT 23. PREREG ends at AMENDMENT 22 + READ 10-21 17:33.
- r6pw20 kernel `eeshsaxena/rsna-sub-v224r6flips6fpw20j` status COMPLETE. Local nb e4d92c0a / meta 3a5b8b0f match the registry. The
  A22 push log shows v1 only, and the gate's last line is `PROBE_CHECK PASS 17/17`.
- No RSNA or Gemma scheduled task is registered (Get-ScheduledTask). RSNA_A22_tick deleted itself.
- A = swide 57464326 (.960) and B default = R6 57248084 (.958), as given in the task. Selection state was not re-read here.

## 2. Designs compared (MODEL, `private_designs.py` -> `private_designs.log`)

Inputs: per-session canary rate from cause.md sec 5, persists .57 / clears .05 / worse .80 at weights .5 / .25 / .25 (GUESS weights).
Copies share the regime and draw independently within it. Alternative: LASTDAY's independent P(clean) .55 / .64. ERROR on top of the
canary: .02 (GUESS). Same-version refusal: .05, hitting every r6pw20 v1 copy together (GUESS, ops.md). Commit fails or comes too
late: .10 / .25 for r6e20 and g85j (GUESS, sec 2.1). g85j rescues 13 % of would-be canaries (MODEL cause.md sec 6). Values vs B = R6
come from bpick 10-21 base: clean r6pw20 +0.6 step / +1.3 smooth silver pts. Skeptic k ~ N(.6, .3) gives +0.2 / +0.6. r6e20 is
r6pw20 minus 0.5 pts (VAL_1020 picks2, 33/33 rows, model-only). B = the earliest clean copy.

| design (submission order) | P(>=1 clean) mixture / indep .55 / .64 | E gain base (step / smooth) | E gain skeptic |
|---|---|---|---|
| D0 none, B = R6 | 0 | 0 / 0 | 0 / 0 |
| D5 1 x r6pw20 | .468 / .512 / .596 | +0.28 / +0.61 | +0.09 / +0.28 |
| D4 2 x r6pw20 | .636 / .748 / .818 | +0.38 / +0.83 | +0.13 / +0.38 |
| **D1 3 x r6pw20 (recommended)** | **.734 / .857 / .901** | **+0.44 / +0.96** | **+0.15 / +0.44** |
| D2 2 x r6pw20 + r6e20 (commit fail .10) | .747 / .870 / .921 | +0.39 / +0.92 | +0.09 / +0.39 |
| D2, commit fail .25 | .728 / .850 / .904 | +0.39 / +0.90 | +0.10 / +0.39 |
| D3 2 x r6pw20 + g85j (commit fail .10) | .774 / .883 / .928 | +0.46 / +1.01 | +0.16 / +0.46 |
| D3, commit fail .25 | .751 / .861 / .910 | +0.45 / +0.98 | +0.15 / +0.45 |
| D6 r6pw20 + r6e20 + g85j (2 commits) | .773 / .884 / .931 | +0.38 / +0.92 | +0.07 / +0.38 |

(E gain columns use the mixture row; the indep rows rank the designs the same way.) P(silver) with B = R6 is .876 / .763; D1 moves it to
.880-.881 / .772-.775 (MODEL, base values).

- **c3's marginal value** (D1 minus D4): +0.06 / +0.13 base, +0.02 / +0.06 skeptic. It is small but positive, it costs nothing on
  the private side, and it matches Eesh's "3 for private".
- **Why r6e20 loses:** canary risk is per session, not per kernel. r6e20 (nb 2a5ef93a) differs from r6pw20 (e4d92c0a) only in cell 18:
  cells 0-17, which hold the R6 base and the S6F stage, are identical (MEASURED now, source diff of both notebooks; R6 equality is
  MEASURED in skeptic cell_diff). The 10-21 outcomes do not follow the
  commit version: spl's T1 commit hit the canary and pw25's T2 commit ran clean (MEASURED cause.md sec 3). So r6e20 gives no canary
  diversification. When it becomes B (both r6pw20 copies failed), it is worth 0.5 pts less.
- **Version-specific failure** (r6pw20 v1 itself is broken in hidden runs): GUESS < .03. The code equals R6 (clean on 10-16) plus
  pw20's cell 18 (clean on 10-20), and pw25, built in the same T2 commit regime, ran clean.

### 2.1 Commit-throttling risk (applies to r6e20 and g85j only)

- CLAIMED (forum 751412, 10-21 17:45 UTC, grandmaster, no staff reply): GPU submissions are throttled, and RSNA final submits are
  blocked by GPU hours. Staff on 10-20 said submits and reruns use no GPU quota (CLAIMED RECHECK_1021). GUESS: the post means new
  Save & Run commits, which do use GPU.
- MEASURED: GPU quota 24.44 h on 10-21 (forum.md). A T2-regime commit took 1011-1231 s (cause.md sec 3). r6e20 has never been
  pushed, so it would be a new kernel (v1), needing the A20 gate (`run_gate.py r6e20`) after downloading its commit output.
- GUESS P(commit not COMPLETE + gated + submitted by ~12:00 IST) .10-.25, because final-day GPU queues are long across comps. D1 has
  no such step. That is the main operational reason to prefer it.

## 3. How to run it (main session, after Eesh's GO)

1. Append **AMENDMENT 24** to PREREG (draft below). The tick needs the exact text `AMENDMENT 24`.
2. Write `C:\Users\eeshs\Downloads\rsna-knee\session_2026-10-22\final\A24_GO.txt`, first line exactly
   `GO c1 c2 c3 MAX=5 Eesh 10-22 HH:MM: "<his words>"`. Never put a lower-case word or bare number right after the keys: `ok`,
   `0930` or `by eesh` voids the GO. Keep MAX=5: it caps ALL team entries of the UTC day, and the tick never takes more than its 3
   copies, so 3 copies + 2 public entries = 5 in any order. MAX=3 would starve the copies if the public entries go first.
3. From `C:\Users\eeshs\Downloads\rsna-knee`:
   - `C:\Python314\python.exe -I session_2026-10-22\final\tick_a24.py --armcheck` must print `A24_ARMCHECK PASS`. It fails now, as
     expected: no GO, no AMENDMENT 24.
   - `C:\Python314\python.exe -I session_2026-10-22\final\tick_a24.py --selftest` must print `A24_TICK_SELFTEST PASS 30/30` with
     the INFO line `-> a live tick WOULD ACT`.
   - `C:\Python314\python.exe -I -B session_2026-10-22\final\tick_a24_livesim.py` must print `A24_LIVESIM PASS 27/27` (fake API,
     temp files).
   - `C:\Python314\python.exe session_2026-10-22\final\tick_a24.py --dry` (no -I: it imports kaggle) must show `next copy c1 (due)`,
     `static ok`, status COMPLETE and `num_allowed_now` = 5 minus entries already made.
4. Register (PowerShell). The first tick runs 1 minute after registration, then every 5 min until 14:45. It self-deletes after:
   ```
   $a  = New-ScheduledTaskAction -Execute 'C:\Users\eeshs\Downloads\rsna-knee\session_2026-10-22\final\tick_a24.cmd'
   $t1 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Hours 6)
   $t1.EndBoundary = '2026-10-22T14:45:00'
   $s  = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -DeleteExpiredTaskAfter (New-TimeSpan -Seconds 0)
   Register-ScheduledTask -TaskName RSNA_A24_tick -Action $a -Trigger $t1 -Settings $s
   ```
   Or skip the task and run `session_2026-10-22\final\tick_a24.cmd` by hand at least 3 times, >= 30 min apart. Extra runs are
   harmless: the tick dedupes and enforces the gap.
5. About 5 min after arming, read `session_2026-10-22\final\a24\tick_a24.log` for `A24 c1 SUBMITTED ... "ref": <id>`. The c1 / c2 / c3
   ids go on the selection card.
6. Kill switch: delete `A24_GO.txt`, which makes the tick a no-op, or run `Unregister-ScheduledTask RSNA_A24_tick -Confirm:$false`.
7. **If the log shows `c1 SUBMIT ERR` twice with a body that reads like a duplicate or version refusal** (not SSL / 5xx): unregister,
   tell Eesh, and then either accept B = R6 or use the contingency. The contingency is to push
   `session_2026-10-21/canary/kernels/kernel_v224r6flips6fpw20g85j` (sha bb995de9, check PASS 26/26, same predictions as r6pw20, r6pw20
   gate key). It needs an A24 line and its own prefixes, and it must be submitted by 14:00 IST.
8. By hand (fallback, same dedupe prefix):
   `C:\Python314\python.exe -m kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-v224r6flips6fpw20j -v 1 -f submission.csv -m "A24-R6PW20C1: resubmit r6pw20 v1 unchanged (57463719 hit the S6F canary .916); B hedge copy 1"`
   (C2 / C3 with "copy 2" / "copy 3"). Always pass `-v 1` and `-f submission.csv`. Never `kernels push` that folder (it would make v2).

### 3.1 What the tick does (differences vs ops_tick_a23.py, all covered by tests)

- GO `final/A24_GO.txt`, guard `AMENDMENT 24`, prefixes `A24-R6PW20C1:` / `C2:` / `C3:`, log `final/a24/tick_a24.log`.
- c1 is due as soon as it is armed (not-before 05:30:00 IST, the UTC day open). c2 and c3 each come >= 30 min after the previous
  copy. CLOSE is 14:30:00 IST. Strict order, one submit per tick, 8-min lock.
- The previous copy's time is taken as the later of the Kaggle list (description prefix + date) and submits.log. A by-hand c1 with no
  log line still enforces the gap; A23 would have let c2 go at once. If a copy is done but has no known time, the tick waits
  (TIME_UNKNOWN) and never bursts.
- Cap: used today (max of API dates and submits.log) < MAX (default 5), and `num_allowed_now` != 0. None rows from the SDK are skipped.
- Cross-dedupe + interlock with the A23-prefixed path (bottom line item 6); `--armcheck` FAILs while an A23_GO.txt exists.
- Unchanged from A23: G3 static shas, G4 kernel status COMPLETE (anything else = STATUS_ODD, no submit), dedupe from a fresh list
  every tick, version pinned to 1, no push stage, a submit exception retried next tick (the list dedupes one that landed).

## 4. B rule and reads (unchanged in substance from A22 / A23)

- **B = the earliest-submitted A24 copy (lowest id) that is COMPLETE, is not .916 and displays >= .958.** It is fixed at the first read
  where one qualifies, and a later copy never replaces it. Otherwise B = R6 57248084. A = swide 57464326 always: a 10-22 entry is
  never A, and the public-push entries are never picked.
- r6pw20 displays .957 with P .006 (MODEL picks2), in which case B = R6. A copy that is PENDING at the hard stop (04:30 IST 10-23)
  counts as failed.
- Read windows (MODEL; latency 4.8-8.5 h, MEASURED / CLAIMED ops.md): arm 09:45 -> c1 earliest ~14:35, worst ~18:15; c3 (10:45)
  worst ~19:15 IST.
- Do NOT select the look-alikes: old r6pw20 57463719 (.916, `A22-R6PW20:`), pw25 57463742, salt 57464911, spl 57464296, and any
  public-push row from today.

## 5. Draft PREREG text (main pastes it, filling in the time and quote; not written here)

> ### AMENDMENT 24 (10-22 HH:MM IST): 3 private slots = r6pw20 v1 resubmits. Written by Claude BEFORE these submits.
> - Eesh 10-22: "<quote>". Entries: up to 3 copies of eeshsaxena/rsna-sub-v224r6flips6fpw20j **kernel_version 1** (nb e4d92c0a, gate
>   PROBE_CHECK PASS 17/17 on 10-21), unchanged, with no push and no commit. Descriptions `A24-R6PW20C1:` / `C2:` / `C3:`, submitted by
>   RSNA_A24_tick (session_2026-10-22/final/tick_a24.py) or by hand with the same prefixes, >= 30 min apart, none after 14:30 IST.
> - B rule, fixed before any score: B = the earliest-submitted A24 copy that is COMPLETE, not .916 and displays >= .958; else R6
>   57248084. A = swide 57464326. No 10-22 entry can be A. The public-push entries are never picked.
> - Rejected: r6e20 as a copy (lower value, needs a commit), g85j as copy 3 (+0.01-0.05 silver pts, needs a commit). Contingency only:
>   g85j if c1 is refused as a same-version resubmit. Evidence: session_2026-10-22/final/private.md.

## 6. Files (this folder)

- `tick_a24.py`: the tick. `--selftest` (offline, writes `tick_a24_selftest.log`), `--armcheck`, `--dry [ISO-IST]` (read-only API).
- `tick_a24.cmd`: the wrapper (no -I, CRLF; it logs to `a24\tick_a24_task.out`). It was run once with no GO and printed `no valid GO
  (no GO line): nothing to do`.
- `tick_a24_livesim.py`: offline live-path simulation with a fake Kaggle module, temp files only (PASS 27/27).
- `private_designs.py` -> `private_designs.log`: the design table (sec 2).
- `a24/`: created by the wrapper test; it holds only `tick_a24_task.out`.
- sha256 (MEASURED): tick_a24.py 5072c48cda73faf2...; tick_a24.cmd c50decbedf238262...; tick_a24_livesim.py 8adf0a7bf15b8efe...;
  private_designs.py 8e170e72fa159ecd.... Pinned inputs: r6pw20 nb e4d92c0a / meta 3a5b8b0f; r6e20 nb 2a5ef93a; g85j nb bb995de9.
