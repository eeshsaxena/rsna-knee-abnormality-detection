# FINAL_1021 (10-21 21:45 IST). Research only: nothing submitted, pushed, selected or scheduled. Sources: final/forum.md, critic.md, redteam.md, ops.md.
1. **Exception: not us.** All 108 of our submissions are COMPLETE with 0 ERROR (MEASURED, Kaggle list 15:44 UTC). The three .916 rows came from our own S6F guard writing the ACL 0.5 canary. Kaggle did not crash them. "Notebook Threw Exception" is hitting other comps today, e.g. Gemma 751339 (CLAIMED forum).
2. **Forum.** 6 new RSNA posts since 12:00 UTC, with 0 rerun-failure reports. Host post: CC BY-NC 4.0 is award-eligible (matters only for prizes). Deadline is still 10-22 23:59 UTC, 5 subs/day, 2 picks (MEASURED forum.md). Other forums report /kaggle/input 35x slower and 0.00 scores after 2 h. Staff have posted no all-clear since 10-20 (CLAIMED), so a new entry fails ~.3-.6 of the time (GUESS).
3. **LB at 15:44 UTC.** swide .960 is rank 147 / 6,324. Silver ends at rank 316 (.958), a margin of 169 (MEASURED lb_bands.out). Projected public rank at the deadline: 152-165 (MODEL).
4. **Critic: no new lever.** 18 of 20 categories closed. The best gap, DICOM protocol cues, has real within-site signal (sum z^2 191 vs null 97, P .000) but scores negative on gold58 at every dose: -.0022 / -.0049 / -.0149 (MEASURED hdr_cues.log). Tie-breaking is worth <= ~5e-5 (MODEL). Nothing to build.
5. **Red team: the pair holds.** 40-70 % of swide's lead over pw25 is read noise (MODEL). Keeping it costs at most 2.0 step / 0.9 smooth silver pts. R6 diversifies swide best (effective corr .85 vs c4 .90, r6pw20 .93). Silent partial failure in swide's run: P <= .02, bias <= .03e-4 (MODEL redteam.md). The family shock (~78 % of private variance) cannot be hedged with any entry we have.
6. **URGENT: 0 of 108 are selected right now** (MEASURED 15:46 UTC). If nothing is selected, Kaggle auto-picks swide + a .959 c4 twin, which costs 1.0-2.7 silver pts vs B = R6 (MODEL). That is more than the whole resubmit plan gains.
7. **Laptop risk.** It shut down unexpectedly 10-21 00:09-05:17 IST, the same hours as tomorrow's submit window (MEASURED event log). The tick only runs while someone is logged in.

**Decisions for Eesh (IST):**
- **D1, now:** select A = swide 57464326 + B = R6 57248084 and send a screenshot. main confirms with `ops_sel_now.py 57464326 57248084` (SELECTED_COUNT 2).
- **D2, by 22:00:** MOACF yes / no. Default and recommendation: NO (~+0.07 silver pts, and it needs a push, commit and gate tonight).
- **D3, by 23:00:** GO / NO-GO on r6pw20 v1 copies: c1 05:31, c2 06:15, optional c3 07:00. P(>= 1 clean) is .55-.64 / .75-.79 / .83-.86 for 1 / 2 / 3 copies, a clean B adds +0.3-0.4 step / +0.6-0.8 smooth silver pts, and a failed copy only costs its slot (MODEL). Also: the A23 text, and the submit method, either the `ops_tick_a23.py` tick (self-test 21/21) or by hand with `-v 1 -f submission.csv`. Optional red-team line: if two copies qualify with different scores, take the higher.
- **D4, 10-22 18:00-22:00 (hard stop 10-23 04:30):** final selection from main's filled card.

**Final selection checklist:**
- A = 57464326 (`STACK S_wide`, .960, 10-21 09:21 IST). It never changes.
- B = the earliest-id COMPLETE `A23-R6PW20C<n>` copy that shows >= .958 and is not .916. Otherwise R6 57248084 (`V224R6-FLIP-S6F-J`, .958, 10-16).
- Never select: pw25 57463742, salt 57464911, spl 57464296, old r6pw20 57463719, or any PENDING / ERROR / .916 row.
- Tick exactly 2 and check the Selected filter shows both COMPLETE. `ops_sel_now.py 57464326 <B>` must print SELECTION_MATCHES_ARGS True. Re-check after 30 min, and tell swapiy / kmrsuman not to submit or change the selection.
- Overnight: laptop on AC, lid open, locked but not signed out. Check from your phone at ~06:00 IST for the C1 row; if it is missing, submit by hand (ops.md 2.3).
