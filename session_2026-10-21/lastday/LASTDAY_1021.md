# LASTDAY_1021.md: what to do on 10-22 and the final pair (10-21 ~19:50 IST, Claude, LASTDAY synthesis)

RESEARCH + LOCAL ONLY. Nothing was submitted, pushed, committed, versioned, downloaded, scheduled or turned into a dataset. This
synthesis step did not use the box: no OAI row, id or per-knee value was read, and no GPU job was started. OAI numbers are box
AGGREGATES quoted from `oaihead.md`. PREREG, HANDOFF, PROGRESS and the live folders were only read.
**Inputs:** `oaihead.md`, `apick.md` (+ the skeptic outputs `skep_apick_{gt,dom,counts}.out`), `surface.md` (+ `work/summ.out`), all in
this folder; `canary/CANARY_1021.md`, `canary/bpick.md` (incl. sec 7 skeptic), `levers/LEVERS_1021.md`, `day/readtable.md` sec 8,
PREREG_10-15.md A22 + READ.
**Tags.** MEASURED = computed from files (named). CLAIMED = older note (named). MODEL = arithmetic on stated inputs. GUESS = judgement.
**Units.** E = private margin in 1e-4 macro (bpick / apick ruler); pub-scale private = .95802 + E x 1e-4. Silver = P(private rank <= 315),
step / smooth field curves. "pts" = percentage points. gold58 = INFO only.

## 1. Plain verdict

1. Besides the r6pw20 resubmit, enter nothing that needs a build. Resubmit = rsna-sub-v224r6flips6fpw20j v1 unchanged (nb e4d92c0a, gate PASS 17/17 on 10-21), 2 copies: 05:31 and ~06:15 IST.
2. Optional 3rd identical copy at ~07:00 IST: about +0.02-0.05 step / +0.05-0.10 smooth silver pts (MODEL, pclean.log 2 -> 3 copies). No copy after 09:00 IST.
3. MOACF(swide) (built, sim-gated, not pushed) is the only other entry with a positive expectation: ~+0.14e-4 on A after P(clean) .55, about +0.07 silver pts, P(public > swide) .57 (MODEL).
4. MOACF default: NO. It needs an exception to "a 10-22 entry can never be A" plus a push, commit and gate tonight. Only if Eesh wants an A-eligible lottery: GO by 22:00 IST.
5. DEAD for 10-22: OAI-trained OA head (no held-out OAI ruler exists; about 0 to +0.1e-4 on A; 4-5 h build), every swide re-dose, public-only twins, guard variant g85j, CVX.
6. Final A = swide 57464326 (.960, r146 / 6,316), fixed now by id. A pw20 override is allowed only by a written statement before selection, and is not recommended.
7. Final B = the earliest-submitted r6pw20 copy (by submission id) that is COMPLETE, shows no canary and displays >= .958; else R6 57248084.
8. With B = R6 (MODEL apick / bpick; A22 priors | levers weighting): E +17.33 (.95975) | +16.74 (.95969); silver .876 / .763 | .865 / .751; P(r<=150) .561 / .446 | .540 / .431.
9. With a clean r6pw20 B: E +17.86 (.95981) | +17.36 (.95976); silver .882 / .776 | .875 / .767; P(r<=150) .594 / .464 | .578 / .452.
10. Two copies, P(>=1 clean) ~.75-.79: pair +17.75, silver .881 / .773 (A22 priors). P(a clean r6pw20 beats R6 as B) ~.75-.8 (GUESS; needs E-OA transfer k >= ~.4). A family shock of ~+-10 silver pts dwarfs every gap here.
11. Eesh decides tonight: MOACF yes / no by 22:00 IST (default no). GO on the r6pw20 copies + A23 text + a submit mechanism by 23:00 IST (new tick or by hand). The A22 tick closes 05:20 and does not retry.
12. 10-22 ~18:00 IST to 10-23 05:29 IST: Eesh selects A and B by hand with both ids written down. Never let auto-selection decide.

## 2. Proposals, evidence and skeptic verdicts

"Value" is the change vs doing nothing (B = R6, A = swide). The skeptic column gives the reviewing track's verdict where one exists. Where
none was returned, it says so.

| # | proposal | evidence (tagged) | value / P | skeptic verdict | verdict |
|---|---|---|---|---|---|
| 1 | **r6pw20 v1 resubmit, copy 1, 05:31 IST** | Cells 0-17 are byte-identical to R6 57248084 and cell 18 to pw20 57425269; both scored clean (MEASURED canary/skeptic/cell_diff.out). Canary is not our code, ~.97 (MEASURED + GUESS). If clean, B beats R6 in 19/19 rows (MODEL bpick) | Clean vs R6: +0.67 step / +1.27 smooth silver, +3.3 r150 step, +0.52e-4 E (base). At P(clean) .5-.64: +0.3-0.4 / +0.6-0.8 silver (MODEL). Private downside: none, since a canary or ERROR leaves B = R6 | bpick sec 7: AGREE, conditional on E-OA transfer k >= ~.35-.45 (GUESS .75-.8 that it holds). At k N(.6, .3): +0.2 / +0.6 | **GO** |
| 2 | **copy 2, ~06:15 IST** | P(>=1 clean) .64 -> .785 at w(persist) .6, .59 -> .75 at .7 (MODEL pclean.log). The two pairs submitted a minute apart on 10-21 each split 1/1 (MEASURED) | +0.10-0.12 step / +0.18-0.23 smooth silver, +0.08e-4 E (MODEL) | bpick 7.0 #7: YES if a slot is free; worth ~3x the guard variant | **GO** |
| 3 | copy 3, ~07:00 IST | P(>=1 clean) .785 -> .857 (w .6), .75 -> .83 (w .7) (MODEL pclean.log) | ~+0.05 / +0.10 silver at k .86; ~+0.02 / +0.05 at k N(.6, .3) (MODEL, scaled from bpick) | No skeptic review. Same version as #1, so no new risk was found. It is used only if copies 1-2 both fail | OPTIONAL (free, tiny) |
| 4 | MOACF(swide) = moacfswide, slug rsna-sub-v224r6c4pw25lomm05mcl10moacf05j, nb e85ceb04 | Positive on all 4 report rulers / 6 nests / EB builds (MEASURED levers/skep_moacf*). No expert ruler: OAI has no Cont / Frac grade, and MOA sits at ceiling on gold58. Built, sim gate PASS 29/29, registry PASS 18/18, self-test 51/51 (MEASURED LEVERS 3.1). Post-read surface: P(public > swide) .57 [.40-.64], private +0.35, R1 increment +0.26, passes the P >= .6 + E >= 0 bar in 2/25 variants (MODEL surface) | R1 increment +0.26e-4 (rho0 +0.20, pess +0.11, flip -0.15). Times P(clean) .55 = ~+0.14e-4, ~+0.07 silver pts. P(it becomes A and is worse) ~.21 x .55 ~ .12 (MODEL LEVERS 2.4) | LEVERS skeptic: WEAK-positive, generic "MOA vs acute" fix, partner not specific. Surface: WEAK, closest single change, misses the bar | **OPTIONAL, NOT RECOMMENDED** |
| 5 | swide + DT OAI-trained OA head, w .10, pre E-OA | No held-out OAI ruler: all 2,581 OAI V00 knees are training rows of every OAI-trained model, and the 21 OOF files have 0 OAI rows (MEASURED box_count / box_inv). gold58 +.0009 OA-mean, P .53; DT - D placebo +.0011, P .64. RSNA OOF +.0013 R1, but DT is stronger than the weak stand-in base there, while swide is .037 above DT on gold58 (MEASURED box_oaihead) | E +0.3e-4 [-2, +2], P(>0) ~.55 (GUESS). On A after P(clean and stage used) ~.45 and the 3-dp read: ~0 to +0.1e-4 | No skeptic vote returned for this track. Self-verdict WEAK. It also adds own-OAI-training exposure to A (forum 750236, no host ruling) | **DEAD** (no kernel exists; 4-5 h build) |
| 6 | OAI head at w >= .15, post-chain, DO / DT+DO, f0-only heads | gold58: w .2 -.0068 (P .09), w .3 -.0128 (P .03), post w .1 -.0016 (P .13). OAI-free placebos are equally negative. DO has no dataset; f0 heads exist for one fold only (MEASURED) | < 0 | none returned | **DEAD** |
| 7 | OAI held-out ruler for OAI heads (track 1 as asked) | Impossible: see #5. The 99 knees outside every union have no images | n/a | n/a | **DEAD** (needs GPU cross-fit retraining) |
| 8 | swide re-doses: swide@.30 (= smid), swide@.20, MCLmm .05 / .15, lomm .03 | P(public > swide) .06-.48; private E -0.55 to -0.01; MCLmm .15 private < 0 in 22/25 variants (MODEL surface) | <= 0 | surface robustness (25 variants), plus LEVERS 4.2 "no re-dose" | **DEAD** |
| 9 | lomm .08 on swide (swide08), alone or + MOACF / MCLmm .15 + MOACF | lomm .08: P .56, private +0.03 (-0.08 at rho0). Combos P .56-.58, passing only in optimistic corners (4-7/25). Unbuilt (MODEL surface) | ~0 to +0.4 in the corners | readtable 8.0 #5: do not enter A-eligible refinements. LEVERS 4.2: no lomm .08 | **DEAD** (unbuilt, re-selects on noise) |
| 10 | **A = swide 57464326** | R1 = H2 answer at the A22 read. The read moved every stack-prior world toward swide (Bayes factor <= 2.8). Base reproduces bpick to the digit (MODEL apick, skeptic re-run MEASURED) | See sec 1 items 8-10. P(swide private > pw20) .52-.68, > pw25 .44-.67 across 6 prior weightings | apick skeptic: **KEEP**, with corrections. The robustness counts were inflated by near-copy rows: on the 12 stack-varying rows swide is best on step silver in 0/12 (within 1 pt in 2/12). swide is dominated in 6/26 rows, every one with a posterior stack <= -0.27 (MEASURED skep_apick_counts / dom) | **KEEP** |
| 11 | A = pw20 57425269 (A22 private-first clause) | vs swide: +0.8 to +2.0 step silver only when >= ~20 % of the prior sits on stack-harmful worlds. Loses r150 (-0.5 to -5.4) and E (-0.4 to -1.7e-4) under every weighting (MODEL apick) | Pair with R6 +15.62 (.95958); silver step .873-.874 in every weighting | apick: WEAK. The case rests on the prior alone, which existed before the read, and the read leans the other way | NOT RECOMMENDED. Only by a written statement before selection |
| 12 | A = pw25 / pw30 / eoa20 | pw25 wins only if the private stack is <= ~0, and pw20 beats it in those worlds. pw30 is -1.2 to -4.2 silver in every weighting. eoa20 is -2.8 to -8.6 r150 (MODEL apick) | < swide | apick: DEAD | **DEAD** |
| 13 | Free-B alternatives (eoa10, pw20, c4, eoa20, pw25, HW) | eoa10 ties R6 (+0.15 silver, range -1.35 / +1.00). The others lose silver vs R6 (MODEL bpick, 19 rows) | ~0 or < 0 | bpick 7.0 #2: B = R6 stays | **DEAD** (no amendment) |
| 14 | Guard variant g85j (S6F start guard raised) | Safe vs 9 h (end ~8.55 h). Rescue applies only on the timing path, which is ~.5 of causes (MEASURED / GUESS) | +0.04 / +0.08 silver per copy; 0 on the crash path | bpick 7.0 #8: safe, not worth building | **DEAD** |
| 15 | Public-only twins p30L / ssafe / smid | Their value was P(public .960) for pw30. swide already shows .960 at ~.96008 [.96004, .96012], and no entry can show .961 (P .000). smid = swide@.30 has P(public > swide) .30 (MODEL surface) | 0 private by construction; ~0 public now | CANARY d: zero medal value, never A or B | **DEAD** (adds auto-select hazard) |
| 16 | CONS(swide) = swide + convex E-OA (CVX) | RAK-PUB ruler killed; gold58 -12.6 on spl; value mostly selection on prior variance, +0.09 at rho0 (MEASURED / MODEL LEVERS 2.3) | +0.09 to +0.26 before P(clean) | LEVERS: runner-up, never next to MOACF | **DEAD** for 10-22 |

## 3. Timeline (GO-worthy entries: #1, #2, optional #3; #4 only on an explicit GO)

### 3.1 Tonight (10-21, now ~19:50 IST)

| by (IST) | who | step |
|---|---|---|
| 22:00 | Eesh | MOACF(swide) yes / no. Default no. Anything that needs a push and commit must start by then (CANARY_1021 (2)) |
| 23:00 | Eesh | GO / NO-GO on the r6pw20 copies (2, optional 3rd), the A23 text below, and the mechanism: a new tick that main builds, or Eesh submitting by hand at 05:31 |
| after GO, ~23:00-00:30 | main | (a) Append A23 to PREREG before any 10-22 submit. (b) Build the A23 tick from val/tick_a22.py: slate = r6pw20 v1 copies at fixed times, resubmit-same-slug allowed, gate = the 10-21 PASS 17/17 (same version, no new commit), first submit not before 05:31 IST. 10-21 UTC has 0 slots left, so a 05:2x submit would bounce. (c) Tick self-test PASS. (d) Register the Windows task RSNA_A23_tick (05:30-09:30 IST, self-deleting). (e) Dry-run log line check |
| only if MOACF GO, 22:00-~00:30 | main | (1) Re-hash nb e85ceb04 / meta f249c914 against `levers/kernels/build_receipt_*`. (2) Push `levers/kernels/kernel_v224r6c4pw25lomm05mcl10moacf05j/` as rsna-sub-v224r6c4pw25lomm05mcl10moacf05j. (3) GPU commit takes ~20 min normally, 20-40+ min on a 10-21-type I/O day (CLAIMED). Note weights_sha seconds as a free I/O probe. (4) Fetch the output, then run `py -3.12 -I session_2026-10-21/levers/kernels/moacf_commit_gate.py <OUT> --a224 0.25 --table "$C4T" --aoa 0.25 --mode pw --aloamm 0.05 --amclmm 0.1 --poa 1 --amoacf 0.05`. It must print PROBE_CHECK PASS 29/29. (5) Merge key moacfswide from `kernels/registry_entries_moacf.json` into the A23 slate. Copy its sim from `kernels/sim_moacf/` into the tick sim dir and re-run the tick self-test. (6) Kill rule: if the gate is not PASS by 04:30 IST, drop MOACF, with no retry and no edit |

Hard rule: MOACF work must never block or delay r6pw20 copy 1. Give it its own slate row and gate state. Copy 1 submits whatever
happens to MOACF.

### 3.2 10-22 (UTC day opens 05:30 IST; 5 slots)

| IST | slot | entry | read by (latency <= ~8.5 h, MEASURED 10-21) |
|---|---|---|---|
| 05:31 | 1 | r6pw20 v1, copy 1 | ~11:00-14:00 |
| 05:33 | 2 | moacfswide, only if GO + gate PASS | ~11:00-14:00 |
| ~06:15 | 2 or 3 | r6pw20 v1, copy 2 | ~11:45-14:45 |
| ~07:00 | 3 or 4 | optional r6pw20 v1, copy 3 | ~12:30-15:30 |
| 09:00 | | last allowed submit of anything | |

- Reads (read-only API list): ~11:00, ~14:00, ~16:00, ~17:30 IST. Record each copy's status, display and canary flag (.916).
- B is fixed at the first read where some copy qualifies under the earliest-COMPLETE rule. A later copy never replaces an earlier
  qualifying one.
- If MOACF ran: A = moacfswide only if it is COMPLETE, shows no canary, displays .960 and is the LB best-id at the selection read.
  Otherwise A = swide 57464326. If the best-id is any other 10-22 entry, A = swide.
- Selection: 10-22 ~18:00 IST to 10-23 05:29 IST. Do it by ~22:00 IST 10-22. Hard stop 10-23 04:30 IST with whatever is COMPLETE.
  A PENDING copy counts as failed. Check afterwards that exactly 2 entries are selected and that their ids match the note.

### 3.3 Draft A23 (text only; NOT written into PREREG; main appends it only on Eesh's GO)

> AMENDMENT 23 (10-21 evening, before any 10-22 submit): r6pw20 57463719 hit the S6F canary (.916) in its hidden run, which counts as a
> failed run. Resubmit rsna-sub-v224r6flips6fpw20j v1 unchanged (nb e4d92c0a; commit gate PASS 17/17 on 10-21): copy 1 at 10-22 ~05:31
> IST, copy 2 at ~06:15 IST [optional copy 3 at ~07:00 IST]; none after 09:00 IST. B = the earliest-submitted copy (by submission id)
> that is COMPLETE, shows no canary and displays >= .958; otherwise B = R6 57248084. A = swide 57464326, fixed now (R1 = H2 at the A22
> read). A 10-22 entry can never be A [except the MOACF clause below]. Eesh selects A and B by hand, both ids written down, before
> 10-23 05:29 IST.
> [Only if Eesh GOes MOACF by 22:00 IST: one A-eligible entry, rsna-sub-v224r6c4pw25lomm05mcl10moacf05j (nb e85ceb04) = swide 57464326's
> notebook + Medial OA <- Contusion .05 then <- Fracture .05 in cell 18, nothing else. Commit gate moacf_commit_gate.py must print
> PROBE_CHECK PASS 29/29 with --aoa 0.25 --aloamm 0.05 --amclmm 0.1 --amoacf 0.05. Submitted after r6pw20 copy 1. A = this entry only if
> it is COMPLETE, shows no canary, displays .960 and is the LB best-id at the selection read; otherwise A = swide 57464326. Never B. No
> other lever, no re-dose, no follow-up.]

## 4. What not to do

1. **No post-read fork on A.** Do not switch A to pw20 / pw25 / pw30 / eoa20 because of how the numbers feel now. The case for pw20 is
   prior-only and the read leans toward swide (apick). Any override must be written down before the selection page is opened.
2. **No OAI-head entry.** Do not build a DT / DO stage. Do not put own-OAI-trained weights into pick A. No new OAI aggregates are needed.
   The box GPU belongs to the other project, and OAI data stays on the box (aggregates only).
3. **No re-dose and no re-add:** swide@.30 / .20, MCLmm .05 / .15, lomm .03 / .08, swide08 (+ MOACF). These re-select on public noise
   already seen (surface 5.2).
4. **No public-only twins** (p30L / ssafe / smid). They have zero private value, swide already shows .960, and they create auto-select
   and best-id confusion.
5. **No second lever and no CVX.** If MOACF goes in, it is the only lever, as a single entry. Never MOACF + CVX. Never push the CVX
   control v224r6c4pw30cvx1j.
6. **No guard variant g85j.** It is safe but worth under 0.1 pt, and it needs a new slug, push, commit, gate and amendment.
7. **Do not change the B rule:** keep the .958 display gate (not .959), keep R6 as the fallback (not eoa10), and keep earliest-COMPLETE
   by submission id.
8. **No submit before 05:31 IST** (10-21 UTC has 0 slots left) **and none after 09:00 IST** (read latency up to ~8.5 h).
9. **Do not let MOACF gating, or any tick change, put r6pw20 copy 1 at risk.** Copy 1 is the only 10-22 action with material value.
10. **Do not let Kaggle auto-select.** It would take the top-2 public: swide plus a .959 pw25 / pw30-family near-twin, which gives no B hedge. A wrong-notebook
    selection has cost a past competition (CLAIMED memory).
11. **Do not read gold58 as selection evidence.** That includes the DT +.0009, the MOACF +-1 rank pair, and the CVX -12.6.
12. **Do not edit PREREG / HANDOFF / scheduled tasks without Eesh's GO.** That work belongs to main after the GO, not to research tracks.

## 5. Sources (this folder unless named)

- `oaihead.md`, `box_inv.log`, `box_count.log`, `box_oaihead.log`, `local_proxy_check.log` (OAI-head track; box aggregates only).
- `apick.md`, `apick_*.log/.json`, `bma_abs.out`, `bma.out`, `evid.log`, `apick_sweep_*.log`; skeptic `skep_apick_gt.out`,
  `skep_apick_dom.out`, `skep_apick_counts.out`.
- `surface.md`, `reads.csv`, `work/surface_*.log`, `work/summ.out`.
- `../canary/CANARY_1021.md`, `../canary/bpick.md` (sec 7 skeptic), `../canary/bpick_work/pclean.log`; `../levers/LEVERS_1021.md`;
  `../day/readtable.md` sec 8; `../../session_2026-10-15/PREREG_10-15.md` A22 + READ 10-21 17:33.
