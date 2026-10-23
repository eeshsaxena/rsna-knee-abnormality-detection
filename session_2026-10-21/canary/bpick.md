# bpick.md: B for the final pair now that A = swide 57464326 (10-21 ~18:15 IST, Claude, CANARY track B)

RESEARCH ONLY. Nothing was submitted, pushed, committed, built, downloaded, posted or voted. No dataset, no GPU job, no box, no OAI
data, no scheduled-task edit. PREREG, HANDOFF, PROGRESS and the val / day / private / stack folders were only read. One read-only
Kaggle API submission listing (`bpick_work/subs_list.py` -> `subs_1021.json`). Local CPU arithmetic only. Everything I wrote is in
`session_2026-10-21/canary/bpick_work/` plus this file.

**Tags.** MEASURED = read or computed now, file named. CLAIMED = older note or brief, named. MODEL = arithmetic on stated inputs
(`bpick_mc.py`). GUESS = judgement.
**Units.** E = private margin, 1e-4 macro over the heliosli copy block (the picks2 / dose / readtable ruler). Pub-scale private
= .95802 + E x 1e-4. "pts" = percentage points. Silver on two curves: step = the 10-19 field curve, smooth = shakeup.md's
fixed-margin curve. P(r<=150) on both. Bronze on the smooth curve.

## 0. Answer

1. **B now = R6 57248084, as Amendment 22 says. Keep it.** (MODEL, 19 variants)
   - Among the clean scored entries, no B beats R6 robustly on P(silver).
   - **eoa10 57413380 is a statistical tie.** Versus R6 it averages +0.15 step / +0.14 smooth silver (15/19 rows positive, range -1.35
     to +1.00). It is in effect a bet that c4's per-finding table beats R6's uniform W: a ±2e-4 c4-side shift swings it -1.35 / +1.00.
     That is not worth amending a pre-registered fallback.
   - **pw20 57425269** trades -0.52 step silver (4/19 rows positive) for +1.21 r150 step (19/19).
   - **c4, eoa20 and pw25 lose silver:** c4 -0.54 (1/19 rows positive), eoa20 -1.15 (0/19), pw25 -1.81 (0/19). They carry the same
     table as swide, and eoa20 / pw25 also the same E-OA dose (private corr with swide .57 / .97 / .98).
2. **Resubmit r6pw20: YES, one 10-22 slot at 05:31 IST (00:01 UTC), same kernel version (v1, gate PASS 17/17 today).**
   - It needs Eesh's GO and an addendum line. A22 has no retry clause for a canary run.
   - **If it runs clean**, B = r6pw20 beats B = R6 in 19/19 rows (MODEL):
     - silver: +0.07 to +1.63 step (base +0.67), +0.59 to +2.25 smooth (base +1.27);
     - r150: +1.97 to +4.84 step (base +3.34), +1.0 to +2.6 smooth (base +1.77);
     - E: +0.27 to +0.84e-4 (base +0.52).
   - It also beats the best free B in every row on r150 and E, and in 18/19 on silver. The one exception is the c4real stress, -0.9
     step / -0.3 smooth silver against eoa10.
   - **P(clean) = .64 [.48-.80]** (MODEL on MEASURED counts, weights GUESS; sec 3). Expected value of the slot vs B = R6 at .64:
     - base: +0.43 step / +0.81 smooth silver, +2.1 / +1.1 r150, +0.34e-4 E;
     - range over 19 variants: +0.05 to +1.04 / +0.38 to +1.44 silver, +1.3 to +3.1 / +0.6 to +1.7 r150.
   - **Private downside: none.** A canary run, an ERROR or a .957 display all leave B = R6.
   - **Opportunity cost: about zero.** Every other 10-22 use is private-neutral (public-only twins, CLAIMED readtable sec 8.0).
3. **Optional second copy of the same version (05:32 IST):** P(at least one clean) goes from .64 to .785.
   - The extra copy adds (base) +0.10 step / +0.18 smooth silver, +0.5 / +0.3 r150, +0.08e-4 E.
   - It is small but positive, and worth it only if that slot would otherwise stay empty or carry a public-only twin.
   - Rule, fixed in advance: B = the earlier-submitted clean copy. The copies are byte-identical, so their outputs should match.
4. **Base numbers (MODEL, `bpick_work/bpick_base.log`; A = swide, conditioned on the A22 read):**

| B | pair E e-4 | pub-scale | P(silver) step / smooth | P(r<=150) step / smooth | bronze | private corr with swide | P(B > A) |
|---|---|---|---|---|---|---|---|
| (swide alone) | 16.57 | .95968 | .843 / .738 | .551 / .436 | .890 | | |
| **R6 57248084 (B now)** | 17.33 | .95975 | **.876 / .763** | .561 / .446 | .924 | .31 | .12 |
| pw20 57425269 | 17.27 | .95975 | .869 / .760 | .570 / .449 | .913 | .97 | .32 |
| pw25 57463742 | 17.11 | .95973 | .857 / .752 | .569 / .448 | .901 | .98 | .33 |
| eoa20 57413368 | 17.09 | .95973 | .863 / .755 | .564 / .445 | .908 | .97 | .24 |
| c4 57343435 | 17.24 | .95974 | .870 / .759 | .558 / .444 | .920 | .57 | .12 |
| eoa10 57413380 (extra) | 17.35 | .95976 | .876 / .764 | .563 / .447 | .924 | .86 | .19 |
| HW 57262115 (extra) | 17.29 | .95975 | .874 / .761 | .554 / .444 | .926 | .38 | .11 |
| pw30 57425970 (extra) | 17.23 | .95974 | .849 / .748 | .568 / .451 | .895 | .97 | .31 |
| **r6pw20 resubmit, if clean** | **17.86** | .95981 | **.882 / .776** | **.594 / .464** | .923 | .71 | .25 |
| r6pw20 resubmit, P(clean) .64 | 17.67 | .95979 | .880 / .772 | .582 / .458 | .923 | | |
| two copies, P(>=1 clean) .785 | 17.75 | .95980 | .881 / .773 | .587 / .460 | .923 | | |

   - Expected private differences between B's are <= 0.8e-4, far below private noise (sd of swide alone 12e-4). The choice is about
     tails, as in picks2.
   - The .958 display guard costs nothing: r6pw20 "rule" and "always" match to 0.01e-4, and P(.957) = .006.

## 1. The read and what it changes (MEASURED API list 10-21, `bpick_work/subs_1021.json`; brief)

- swide 57464326 .960 (r146 of 6,316, .960 band r100-150), pw25 57463742 .959.
- r6pw20 57463719, spl 57464296 and salt 57464911 all show .916, the S6F canary.
- The 20 newest submissions are all COMPLETE. The S6F-lineage runs before 10-21 all displayed .956-.959, and none was a canary:
  - 15 in the API list (c4 10-18 to pw30 10-20);
  - about 10 more from `session_2026-10-14/submits.log` and timing.md (S6F 57201874 to a25 57316395, CLAIMED displays).
- Today's submits went in 03:30-04:11 UTC. A19 (10-19, 04:26-04:44 UTC) and A20 (10-20, 00:01 UTC) were clean, so the time of day
  alone does not explain the canaries.
- **Conditioning in the model** (`bpick_mc.py` sec 1). The world is `day/skeptic_wc.py` sections 1-2b, exec'd verbatim with the same
  seeds and its log line neutralised.
  - The model conditions on four facts:
    - swide's public > pw25's and pw30's;
    - swide shows .960;
    - pw25 shows .959;
    - pw20 / eoa10-30 / pw30 stay below .960.
  - Uniform-band rank ruler for r146 (4 teams below it in the .960 band, 46 above): t^4 (1 - t)^46.
  - The canary runs (spl, salt, r6pw20) are treated as unobserved.
  - ESS 39k of 400k. The `base_norank` variant drops the ruler (ESS 124k) and gives the same ranking.
- Posterior (MODEL, base):
  - public swide - pw25: prior +0.24, posterior +2.03e-4.
  - public MCLmm effect: prior -0.09, posterior +0.50. lomm05: +0.18 -> +0.44.
  - private stack total (lomm05 + I + MCLmm, with noise): +1.08e-4; P(< 0) .33; P(MCLmm private < 0) .41.
  - E-OA at .25: +9.9e-4 over c4 privately; P(below c4) .13. P(swide < pw20 privately) .32.

## 2. Why R6 is the free B, and what each candidate hedges (MODEL, base)

E[max - swide] (hedge), overall and conditional on the two exposures of swide:

| B | corr | hedge all | given E-OA < 0 (P .13) | given MCLmm < 0 (P .41) |
|---|---|---|---|---|
| R6 | .31 | +0.76 | +4.56 | +0.90 |
| HW | .38 | +0.72 | +4.65 | +0.85 |
| c4 | .57 | +0.67 | +5.00 | +0.82 |
| eoa10 | .86 | +0.78 | +4.89 | +1.03 |
| eoa20 | .97 | +0.51 | +2.00 | +0.83 |
| pw20 | .97 | +0.69 | +2.38 | +1.10 |
| pw25 | .98 | +0.54 | +0.56 | +0.97 |
| r6pw20 (clean) | .71 | **+1.29** | +3.05 | **+1.60** |

- **The big tail of swide is the E-OA dose / coupling tail, not MCLmm.** When E-OA fails, swide falls ~10e-4. MCLmm is a ~1-2e-4 coin
  flip.
  - R6, HW, c4 and eoa10 cover the dose tail. The c4-side ones (pw20, eoa20, pw25) barely do.
  - Silver is a threshold metric, so the dose-tail cover is what moves P(silver). That is why R6 / eoa10 beat pw20 on silver.
  - pw20 / pw25 win on r150, where only upside matters: as near-twins of swide, their max adds the upside of two close private
    draws, but they rarely rescue a silver miss.
- **R6 vs c4:** both cover the dose tail. R6 also diversifies the base (our-side private corr with c4 .54, CLAIMED picks.md), so R6 >
  c4 in 18/19 rows.
- **r6pw20** combines the base hedge (R6) with the E-OA gain (k ~ .86 transfer), so it is the best B in every row on r150 and E.

### 2.1 Robustness of the free-B choice (`bpick_summ.out`, 19 rows)

- **Rows:** base, base_norank, lskep, mzero, worst, mstack, rhoS0, rhoA0, nf15, pf5, skepall, pure, flatU, PAREA, kpess, fieldBad,
  fieldGood, c4over (c4 lineage private -2e-4, GUESS stress), c4real (+2e-4).
- **Mean / max regret among free B's** (silver step | smooth | r150 step | r150 smooth | E):
  - R6: .25/1.00 | .26/.87 | 1.27/2.68 | .66/1.53 | .08/.30
  - eoa10: .10/1.35 | .11/1.03 | 1.05/2.12 | .55/1.18 | .05/.25
  - pw20: .77/1.75 | .37/1.17 | .06/.28 | .23/.43 | .09/.33
  - c4: .79/2.04 | .69/1.57 | 1.64/3.05 | .91/1.77 | .18/.38
  - eoa20: 1.40/2.43 | .98/1.77 | .79/1.34 | .69/1.06 | .28/.51
  - pw25: 2.05/2.84 | 1.23/1.87 | .11/.42 | .26/.44 | .24/.49
  - pw30: 2.91/3.69 | 1.77/2.28 | .28/.78 | .00/.00 | .11/.37
- **Reading:**
  - eoa10 and R6 are the silver-robust pair. Neither is dominated.
  - eoa10 leads in the skeptic-component rows (nf15, rhoA0, skepall, worst: +0.56 to +0.72 step silver) and in c4real (+1.00).
  - R6 leads in c4over (+1.35), flatU, mstack and norank.
  - Switching the fallback to eoa10 buys about +0.15 pts silver on average for a ±1.2 pt bet on the c4 table. The A22 fallback (R6)
    stays (GUESS on MODEL).

## 3. P(clean) for the resubmit (MODEL on MEASURED counts; regime weights GUESS; `bpick_work/pclean.py` -> `pclean.log`)

- **Counts:** before 10-21, about 25 of 25 S6F-lineage hidden runs were clean. On 10-21, 2 of 5 were clean.
- **Mechanism:** CLAIMED timing.md / runtime.md. S6F starts about 4.0-4.3 h into a ~6.06 h run. Its start guard is at 5.75 h, so a
  session would need to be about 1.35-1.4x slower to skip it. The brief's other canary triggers (driver killed or not ok, sha, rho
  floor) are not separable from the score: any trigger shows .916.
- **Two regimes:**
  - the 10-21 regime persists: p ~ Beta(3, 4), mean .43;
  - back to normal: p ~ Beta(26, 1), mean .96.
- **Weight on "persists":** .6 (GUESS). It is above .5 because 10-22 is the deadline day, when load only rises, and Kaggle had
  incidents on 10-20/21 (CLAIMED PREREG A22 read note).
- **Results:**

  | w(persists) | 1 copy | 2 copies (>= 1 clean) | 3 copies |
  |---|---|---|---|
  | .3 | .80 | .89 | .93 |
  | .5 | .70 | .82 | .88 |
  | **.6** | **.64** | **.785** | .86 |
  | .7 | .59 | .75 | .83 |
  | .9 | .48 | .68 | .79 |

- **Copies:** treated as independent within a regime (GUESS). Today the two pairs submitted a minute apart each split one clean,
  one canary, so copies are at least not perfectly correlated (MEASURED).
- **Read timing:**
  - A 05:31 IST submit is read by about 14:00-15:00 IST 10-22. Today's latency was at most 8.5 h (MEASURED: submitted 09:00-09:41
    IST, all COMPLETE at the 17:33 read).
  - That is well before the 10-23 05:29 IST deadline.
  - No new commit is needed, so no GPU quota is used and no commit queue risk applies.
  - Submit at 00:01 UTC, the first minute of the UTC day: the A20 submits at that time were clean (MEASURED).
- **Not modelled:** a variant with a raised S6F start guard could survive moderate slowness. It is unbuilt and its cause is unproven,
  and it would need a build, a gate, an amendment and Eesh's GO. I flag it only; the canary-cause track owns it.

## 4. Draft addendum (text only; NOT written into PREREG; Eesh's GO needed)

> "A22 addendum (10-21 evening): r6pw20 57463719 hit the S6F canary (.916) in its hidden run. That counts as a failed run. One resubmit
> of rsna-sub-v224r6flips6fpw20j v1 (unchanged, commit gate PASS 17/17 on 10-21) at 10-22 ~05:31 IST [optional: a second identical
> submit at ~05:32 IST]. B = the earliest-submitted copy that is COMPLETE, shows no canary and displays >= .958; otherwise B = R6
> 57248084. A = swide 57464326 (API best-id; R1 = H2 here, since a stack won). A 10-22 entry can never be A. Eesh selects A and B by
> hand, ids written down, before 10-23 05:29 IST."

- Mechanics: the RSNA_A22 tick closes at 10-22 05:20 IST, and the tick does not retry a scoring-time failure (CLAIMED readtable sec
  1). So this is a manual submit or a new A23 tick, which main builds after the GO.

## 5. Caveats

- **One dose family** (dose_mc, OAI-proxy D5 curves fitted to public reads). The E-OA tail that drives "R6 over pw20 on silver" comes
  from it.
  - flatU and PAREA move levels but not the B ranking: R6 / eoa10 on silver, pw20 / r6pw20 on r150.
- **Stack components** rest on gold58 nominations. The read lifts the public MCLmm posterior. The private carry-over is stressed only
  by the rhoS0 / rhoA0 / skepall / worst / pure rows, and none of them flips the answer.
- **The rank ruler** assumes the .960 band is uniform in score. A denser band bottom would put swide even closer to .960 and shrink
  the stack posterior a little (norank is the opposite bound). The ranking does not change.
- **Our-side posterior** (c4 vs R6, corr .54) is the 10-19 one. The ±2e-4 c4 stress only moves R6 vs eoa10.
- **Field curves** date from 10-19 / 10-20. The fieldBad / fieldGood rows move levels by ±1-2 pts and leave the B ranking unchanged.
  The family shock of about ±10 pts (CLAIMED shakeup.md) dwarfs every gap here.
- **P(clean) regime weights are GUESS**, so value x P(clean) is quoted at .43 / .64 / .785 / .96 in `resub_value.log`.
- **Determinism:** the resubmit is assumed to reproduce the private score r6pw20 would have had if clean (same code, same hidden set).
  GPU non-determinism is assumed negligible (GUESS).
- **Not modelled:** host rulings, teams copying E-OA, and ERROR or stuck-run tails beyond the canary.

## 6. Files (`session_2026-10-21/canary/bpick_work/`)

- **Read-only API list:** `subs_list.py` -> `subs_1021.json`.
- **Model:** `bpick_mc.py`. It execs `day/skeptic_wc.py` sections 1-2b and adds the read conditioning, the B list, the resubmit mix,
  the hedge and correlation moments, and the c4shift knob. Args are as skeptic_wc plus `rank=`, `c4shift=`, `tag3=`.
- **Per-variant output:** `bpick_<tag>.log` / `.json` for 19 tags. Runners: `run_var.sh` (log `run_var.log`) and `run_var2.sh`.
- **Summaries:** `bpick_summ.py` -> `bpick_summ.out`; `resub_value.log` (value x P(clean) ranges, r6pw20 vs best free B, pairwise
  counts); `pclean.py` -> `pclean.log`.

## 7. SKEPTIC review of bpick.md and cause.md (10-21 ~19:00 IST, Claude, SKEPTIC track)

RESEARCH ONLY. Nothing was submitted, pushed, committed, built, downloaded, posted or scheduled. No OAI data, no GPU, no box. Local CPU
only (5 MC reruns of about 1 min each). Everything new is in `canary/skeptic/` (sections 0-6 above are unchanged; a backup is at
`skeptic/bpick_pre_skeptic.md.bak`). Same tags as above.

### 7.0 Verdicts

| # | claim under test | verdict | why (tag) |
|---|---|---|---|
| 1 | The canary is not our code (cause.md, ~.85) | **AGREE, stronger: ~.97 run-to-run.** The specific path (slow Dataset reads tripping the S6F timing guards) is **only ~.5** | MEASURED `skeptic/cell_diff.out`. r6pw20 cells 0-17 are byte-identical to R6 57248084 (10-15), and its cell 18 is byte-identical to pw20 57425269 (10-20). Both of those ran clean in hidden reruns. spl's cell 18 = swide's with A_MCLMM 0.1 -> 0.0. salt's = swide's minus the LOAMM block. Both are strict subsets of swide's path, and swide ran clean. The guard path itself is unproven: the measured commit slowdown (+0.1-0.17 h) is 10-20x too small to reach any timing threshold. A crash path fits the facts just as well (GUESS): one compact_infer part exits rc != 0, run_group kills its sibling, there is no retry, coverage drops below .98, and the canary fires |
| 2 | B now = R6 57248084 | **AGREE** | This is A22 as written. r6pw20 displayed .916 < .958 (MEASURED) |
| 3 | r6pw20, if clean, beats R6 as B | **AGREE, conditional on the E-OA carrying over to R6 (transfer factor k).** The margin is small | MODEL `skeptic/bpick_k0.log`, `bpick_k03.log`, `skep959_*.log`. vs R6, silver step / smooth, E: k .86 (base) +0.7 / +1.3, +0.52; k N(.6, .3) +0.2 / +0.6, +0.27; k .3 -0.3 / -0.1, 0.00; k 0 -0.5 / -0.3, -0.07. Breakeven is k ~.35-.45. The only evidence is the gold58 ratio .81 for pw .2 (P(R6 delta > 0) .96, CLAIMED picks2 1.3), from the same small set that nominated E-OA. There is no hidden-label check, because r6pw20's one hidden run was blanked. GUESS ~.75-.8 that it beats R6 in expectation |
| 4 | Display gate >= .958 (vs a stricter .959) | **KEEP .958** | MODEL `skep959_*`. The .959 gate gives up -0.3 / -0.6 silver in base and -0.0 / -0.2 at kpess, and only helps when k < ~.4 (k .3: +0.2 / +0.1). It is pre-registered; do not change it |
| 5 | Resubmit r6pw20 v1, 1 slot, 05:31 IST | **YES, worth it.** No B downside apart from a small silent-degradation tail | A canary or ERROR leaves B = R6. Residual risk (GUESS < .02): a "clean" run with R6 per-study fallbacks or S6F coverage in [.98, 1) can still display .958 and pass the gate, with a small private effect. Value at P(clean) .5-.64: about +0.3-0.4 step / +0.6-0.8 smooth silver pts (MODEL base), about 0 if k ~.35. Opportunity cost: about 0, since readtable sec 8.0 items 5-6 leave the private slots empty except for this retry |
| 6 | P(clean) .64 (bpick) vs .50 (cause) | **Both are GUESS-weighted. Use ~.55 [.35-.80]. The decision does not depend on it** | I re-derived both: .6 x 3/7 + .4 x 26/27 = .642; .5 x .57 + .25 x .8 + .25 x .05 = .498 canary. The whole gap is cause.md's "worse" scenario (.25 weight). Deadline-day load supports keeping it. Today's evidence on time of day is nil |
| 7 | Second identical copy | **YES if a slot is free. It is worth ~3x the guard variant** | MODEL. The 2nd plain copy adds +.15-.18 P(>=1 clean), about +0.10-0.12 step / +0.18-0.23 smooth silver. Spreading the copies (05:31 + ~06:00-06:30 IST) is free. The two pairs submitted a minute apart today each split 1 / 1 (MEASURED), so there is no evidence either way on correlation |
| 8 | Guard variant g85j is safe vs 9 h | **AGREE it is safe. DISAGREE it is worth building tonight** | Re-derived from the code: the kill fires at NB_T0 + 8.5 h, p.wait <= 120 s, and cell 18 then reads only /kaggle/working files (no Dataset reads), so the end is ~NB_T0 + 8.54 h. Pre-NB_T0 cells took 12-19 s even in the T2 commits (MEASURED `work/stage_marks.out`), so the typical end is ~8.55 h and the stated bound 8.87 h. The only unmeasured term is container time before cell 0. An overrun costs B nothing, because only runs that would canary under A22 can reach the new deadline. Value, though: +.065 P(clean) per copy, about +0.04 step / +0.08 smooth silver (MODEL), and that assumes the timing paths. The .13 rescue figure is 0 on crash paths. Cost: a new slug, push + GPU commit + gate + registry + A23 text tonight, all for under 0.1 pt |
| 9 | Draft addendum (sec 4) | **KEEP, with 2 edits** | (a) Add "B = the earliest-submitted copy that is COMPLETE and >= .958, by submission id", which sec 4 already implies. (b) "A 10-22 entry can never be A" matches readtable 8.4, but it is a new rule beyond A22's R1, so Eesh's GO must cover it explicitly |

### 7.1 Operational risks the documents under-weight (GUESS)

- **The 05:31 IST submit needs a mechanism.** RSNA_A22_tick closes at 05:20 IST and does not retry a canary. Overnight automation
  here works only through a Windows scheduled task (CLAIMED memory), and this track may not create one. Without a GO plus a new task
  (or Eesh submitting by hand), the slot is missed and B stays R6. That is a bigger expected loss than anything in the guard-variant
  debate.
- **Read deadline.** A slow clean run can take ~8.3 h plus queue. A 05:31 copy reads by ~14:00-15:00 IST, a 06:30 copy by ~15:00-16:00,
  both before the ~18:00 IST A / B read. Do not put a copy later than ~09:00 IST.
- **Free regime probe.** Any commit made tonight, for whatever reason, gives an I/O reading in its log (weights_sha seconds: T1 10-22,
  T2 95-145). It is information only. The resubmit is worth doing in every regime.

### 7.2 Bottom line (skeptic)

B = R6 now. Resubmit r6pw20 v1 unchanged (already gated): 1 copy at 05:31 IST, plus a 2nd at ~06:00-06:30 IST if a slot is free.
B = the first COMPLETE copy >= .958, else R6. Skip the guard variant: it is safe but worth under 0.1 pt, and its rescue rests on an
unproven timing mechanism. The value of the whole plan depends on (i) Eesh's GO with a working 05:31 submit mechanism and (ii) the
E-OA transfer to R6 being at least ~.4. If Eesh doubts (ii), doing nothing (B = R6) gives up about 0.3-0.8 silver pts and nothing else.

### 7.3 Files (`canary/skeptic/`)

- `cell_diff.py` -> `cell_diff.out`: per-cell sha for R6 10-15, pw20 10-20, the 5 A22 kernels and the variant, plus the cell 18 / cell
  14 diffs.
- `skep_bpick_mc.py`: bpick_mc.py with only HERE changed. Outputs `bpick_k0.log/.json` (k N(0, .15)) and `bpick_k03.log/.json`
  (k N(.3, .15)).
- `skep_bpick959.py`: adds a ".959-gate" B. Outputs `skep959_base / _k03 / _kpess .log/.json`.
