# mech.md: A23 submit mechanism + docs audit (recheck, 10-21 ~23:30 IST, Claude). READ-ONLY.

Nothing was submitted, pushed, committed, selected, registered or posted. PREREG, HANDOFF, PROGRESS, final/ and a23/ were only read.
Kaggle calls were read-only: submission list, kernel status, submission limits, plus CLI `-h`. Tags: MEASURED (run or read now),
CLAIMED (a note says so), GUESS (judgement).

## 0. Bottom line

1. **Use `final/ops_tick_a23.py`. Retire `a23/tick_a23.py`. Never register both.** The two scripts use different GO paths, description
   prefixes, caps and spacing (sec 3). Only the ops version matches FINAL_1021, the ops.md by-hand fallback and the selection card.
2. Both scripts submit only `eeshsaxena/rsna-sub-v224r6flips6fpw20j`, kernel_version 1, `submission.csv`. They have no push stage.
   Each copy gets its own description (MEASURED, code + harness).
3. ops self-test reproduced: **A23_TICK_SELFTEST PASS 21/21** (MEASURED 23:26, run on a copy so final/ was not touched). Its dry run
   at 05:31 gives `status COMPLETE | copies done - | next copy c1 | used 10-22 = 0 (MAX 3) | static ok` (MEASURED). At 05:30:59 it gives
   "before", and at 09:00:01 "past CLOSE".
4. The manual fallback command is **correct for kaggle CLI 2.2.4** (MEASURED help + source, sec 4).
5. **State now (MEASURED 23:2x IST):** no AMENDMENT 23 in PREREG, no A23_GO.txt in a23/ or final/, and no RSNA_A23_tick task.
   RSNA_A22_tick is still Ready (every 10 min, EndBoundary 10-22 05:25) and has nothing left to submit (CLAIMED ops.md). The 23:00 D3
   time has passed with nothing armed. **Without a GO, B = R6 57248084.**
6. **One gap to close before arming:** the documented checks (self-test, `--dry`) never parse the real GO file or PREREG, and `--dry`
   assumes `GO c1 c2 c3`. Free text after the keys can void the GO, and the tick would then print "no valid GO" every night tick and
   submit nothing (MEASURED parse cases, sec 2). Fix: after writing the GO, run the live `.cmd` once before 05:31 (sec 5).

## 1. ops_tick_a23.py, line by line (RECOMMENDED)

| guard | code | verdict |
|---|---|---|
| target | l.31 SLUG + VER = 1; l.192 the only write call, `competition_submit_code(file_name='submission.csv', message=desc, competition=COMP, kernel=SLUG, kernel_version=VER)` | MEASURED correct. The SDK sends kernel_version when it is truthy (1 is) |
| descriptions | l.38-40 `A23-R6PW20C1:` / `C2:` / `C3:`, unique, no A22 prefix | MEASURED (self-test + real list: no existing A23 rows) |
| G1 GO | l.26 `final/A23_GO.txt`; parse_go l.52-71; copies must be a prefix of c1 c2 c3; MAX clamped to 0-5 | MEASURED. Gotcha below |
| G2 AMENDMENT 23 | l.73-74 substring on the upper-cased PREREG | MEASURED False now. Any line containing "AMENDMENT 23" passes, so do not quote the phrase in PREREG before the GO |
| G3 static | l.76-88 nb sha e4d92c0a, meta sha 3a5b8b0f, meta id, push log == ['1'], gate last line PASS 17/17 | MEASURED ok |
| G4 kernel status | l.165-169 must contain COMPLETE | MEASURED `KernelWorkerStatus.COMPLETE` |
| G5 times | c1 05:31:00, c2 06:15, c3 07:00, >= 30 min after the previous copy's logged submit, CLOSE 09:00:00, strict order | MEASURED (self-test + dry runs) |
| G6 dedupe | fresh Kaggle list (page_size 50) + `A23 cN SUBMITTED` in submits.log; a failed list call means no submit | MEASURED. The A22 r6pw20 line does not block (done_keys on the real list + log = empty set) |
| G7 cap | used = max(submits.log, API dates) < MAX, and num_allowed_now != 0 | MEASURED. API dates are naive UTC, so the 10-21 count is 5 and the 10-22 count is 0. Limits now: numToday 5, numAllowedNow 0 |
| G8 | one submit per tick | MEASURED (code) |
| errors | status, list and submit errors are logged and retried next tick. A failed limits call is ignored. If the kaggle import fails, the traceback goes to task.out and the lock is removed | MEASURED (code) |
| lock | 8 min, under the 10 min ExecutionTimeLimit, so a killed run cannot block the next tick | MEASURED (code) |
| .cmd | no `-I` (kaggle is in user site-packages), creates final/a23, appends to final/a23/tick_a23_task.out | MEASURED |

**GO parse gotcha (MEASURED):** after the keys, the first token that is all lower-case letters or digits voids the GO.
- Valid: `GO c1 c2 MAX=3 Eesh 10-21 23:00: "..."` -> [c1, c2], MAX 3. `GO c1 c1 c2` -> [c1, c2]. `GO c1 c2 MAX=0` -> never submits.
- Void (nothing is submitted): `GO c1 c2 ok`, `GO c1 c2 2300`, `GO c1 c2 by eesh`, `GO c1 c2 MAX=x`.
- `GO c1 C2 Eesh` -> c1 only, because the upper-case C2 is read as free text.
- Use exactly the ops.md format: a capitalised name, then a time that contains `-` or `:`.

**Minor (no fix needed tonight):** the header says "log STATUS_ODD once", but the code logs it on every tick. Any submit response is
logged as SUBMITTED without checking `ref`. If Kaggle ever returned a soft failure without raising, the copy would count as done and
would not be retried (GUESS: unlikely, errors raise. A22's lines show `{"message": "", "ref": ...}`).

## 2. a23/tick_a23.py, line by line (DO NOT USE)

The target and guards are correct: SLUG + VERSION 1 + submission.csv. GO, AMENDMENT 23 and CLOSE are checked before any API call.
Dedupe runs against a fresh Kaggle list. The daily cap reads real API dates: a simulated 10-21 23:00 gives "CAP: 5" (MEASURED). Dry mode
writes no log, lock or submits.log (MEASURED). The weaknesses:
- **No spacing and no one-per-tick rule.** On a late wake, every due copy goes out in the same second. MEASURED harness: 06:20 gives c1 + c2
  back-to-back, and 07:10 with c3 gives all three. That throws away the time spacing that the copies are meant to add against a
  time-correlated canary.
- **Different prefix** `A23-R6PW20-C1:` (with a hyphen). The FINAL_1021 selection rule (`A23-R6PW20C<n>`) and the ops.md by-hand command
  do not match it. A by-hand c1 would not be seen by this tick, so c1 could be submitted twice.
- **Different GO path** `a23/A23_GO.txt`. ops.md tells main to write `final/A23_GO.txt`.
- MAX_DAY 5 is only Kaggle's own cap. There is no num_allowed_now check and no kernel status or static check.
- An uncaught list or import error only leaves a traceback in task.out. A 30 min lock can block about 6 ticks after a killed run.
- `GO c2` alone is accepted (no order rule). Upper-case `C2` is silently dropped.
- HANDOFF 21:30 says "dry-run OK", but `--dry` can only print "no valid GO" unless a GO file exists, and it has no simulated clock.
  The harness had to inject a GO and a clock to exercise it.

## 3. Mechanism comparison

| | ops_tick_a23.py | tick_a23.py |
|---|---|---|
| GO path | final/A23_GO.txt | a23/A23_GO.txt |
| prefix | `A23-R6PW20C1:` (matches FINAL / ops.md / card) | `A23-R6PW20-C1:` (matches nothing else) |
| spacing / per tick | >= 30 min, 1 per tick | none, burst |
| cap | MAX from GO (default 3) + num_allowed_now | 5, own count |
| extra guards | static sha, push log, gate, kernel status | none |
| tests | self-test 21/21 + `--dry <time>` | harness only (this file) |

## 4. Manual fallback (MEASURED, CLI 2.2.4)

`C:\Python314\python.exe -m kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-v224r6flips6fpw20j -v 1 -f submission.csv -m "A23-R6PW20C1: ..."`
- The help shows `[-f FILE_NAME] [-k KERNEL] -m MESSAGE [-v VERSION] [-q] [--sandbox] [competition]`. `python -m kaggle` works
  (`__main__.py` exists). The bare `kaggle` on PATH is the Python312 install, which is also 2.2.4 with the same flags.
- Small correction to ops.md 2.3: the CLI **refuses** `-k` without `-v` (ValueError in competition_submit_cli). The SDK, not the CLI,
  is what falls back to the latest version. Always pass `-v 1` anyway.
- Use the same `A23-R6PW20C<n>:` prefix, so the ops tick dedupes a hand submit.

## 5. Arming checklist (only after Eesh's GO; main's job, not done here)

1. Append AMENDMENT 23 (LASTDAY 3.3 text) to PREREG. Write `final/A23_GO.txt` in the ops.md format (sec 1 gotcha).
2. Run the self-test. Expect 19/21 OK and exactly 2 expected FAILs: the PREREG guard is now True and the GO file now exists.
3. **New check:** run `session_2026-10-21\final\ops_tick_a23.cmd` once by hand before 05:31. The tick returns at l.156-157, before
   importing kaggle, so it cannot submit. `final\a23\tick_a23_task.out` must end with `before 2026-10-22 05:31:00: nothing to do`. If it
   says `no valid GO (...)`, `STATIC_FAIL` or "no AMENDMENT 23", fix that first.
4. `--dry 2026-10-22T05:31:00` should print next copy c1, static ok, COMPLETE.
5. Register RSNA_A23_tick as in ops.md 2.2 step 4, then check NextRunTime 10-22 05:25. Do not register tick_a23.cmd.

## 6. Docs: contradictions and gaps

| # | where | finding | severity |
|---|---|---|---|
| D1 | HANDOFF 21:30 | Lists both mechanisms without picking one. FINAL_1021 D3 and ops.md name only ops_tick_a23. Pick ops (sec 0) | MEDIUM |
| D2 | PREREG READ 17:33 (l.538) | Says "Amendment 22 allowed one resubmit on ERROR". A22 as written (l.520-533) has no such clause; it was only in drafts (VAL_1020 l.77/123, PRIVATE_1020 l.21/212/215). A23 is a new permission and must stand alone, which the LASTDAY 3.3 draft does. PREREG l.14 also has "no identical resubmits" (scoped to the 10-15 lineup); A23 could say it does not apply | LOW |
| D3 | B rule wording | LASTDAY 3.3 / FINAL / CANARY: earliest id among qualifying copies. ops.md sec 4: "fixed at the first read where one qualifies (a later copy never replaces it)". These differ if c2 qualifies while c1 is still PENDING. redteam 3.4 adds an optional "higher display if displays differ". A23 should say the rule is evaluated once at the selection read (>= 18:00 IST 10-22) and settle the higher-display option in its own text | LOW-MEDIUM |
| D4 | A22 (l.532) + HANDOFF 08:45 | "Eesh may instead fix A = pw20 57425269" is still formally open. LASTDAY 4.1, FINAL ("A never changes") and the A23 draft ("A = swide fixed now") close it, but only once A23 is written. With no GO, A22's R1 still gives swide (no new entries) | LOW |
| D5 | ids | swide 57464326 (.960), R6 57248084 (.958), old r6pw20 57463719 (.916), pw25 57463742 (.959), spl 57464296 / salt 57464911 (.916) agree across PREREG, HANDOFF, FINAL, ops.md, submits.log and the live list (MEASURED) | OK |
| D6 | PROGRESS_21-10-2026 | Stops at 18:30. It does not record LASTDAY (19:55), FINAL (21:30) or the 0-selected state, and says "1-2 copies" where the current plan is 2 + an optional 3rd. Stale, not contradictory | LOW |
| D7 | ops.md 2.3 | The "-v omitted -> latest" claim holds for the SDK; the CLI errors instead (sec 4) | LOW |
| D8 | timing | 23:00 D3 has passed with no GO. ops.md planned main's setup for 23:00-00:30; the hard technical limit is the first tick at 05:25 | INFO |

## 7. Files

- `recheck/mech_harness.py`: read-only harness. It blocks every Kaggle write call, forces dry mode, uses a fake clock and temp GO /
  PREREG files. Output is in `recheck/mech_harness.out`. The ops self-test copy log is `recheck/mech_tmp/ops_a23_selftest.log`. The temp
  GO / PREREG stand-ins were deleted after the run.
