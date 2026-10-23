# V100 box schedule, 10-04 15:00 IST to 10-22 (own-fleet blend leg for the 0.943 parent)

Written 2026-10-04 ~15:20 IST. Plan only: nothing was launched, stopped or signalled on the box; nothing was
pushed or submitted. Tags: **MEASURED** (read from a log or file, path given), **MODEL** (derived from a stated
model), **GUESS** (judgement). `SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`,
`B` = `ssh -o BatchMode=yes training@192.168.10.30`. Box logs: `~/fleet_runs/*.log`. Closed-form tables:
`SP/jump_wf/sched_binormal.py` -> `sched_binormal.out` (equal-variance binormal, the same model family as
`bigjump/plan_numbers.py`; it is known to over-predict lift by ~.002-.004, so read it as an upper bound).

---

## 0. Verdict in ten lines

1. **Concurrency buys nothing.** One F1 job alone runs 1.08-1.09 s/it; two jobs sharing the V100 run 2.22-2.25 s/it
   each, so two jobs deliver 0.96x the work of one (MEASURED, sec 1.2). Every job should run alone, back to back.
   Sharing only doubles the time to every gate read.
2. **Stop the gated run now (Eesh's call).** It has ~4.1 alone-hours left (MODEL). It holds the TTR G-T2 read at
   ~22:45 instead of ~19:10 (MODEL). The question it answers (gated vs v4 labels) is worth about +/-.0005 of blend
   (MODEL), and at epoch 2 it tracks v4 on val (.8160 vs .8160, MEASURED).
3. **Production members are all-data seeds** chosen by a fixed epoch. Fold models are only for gates.
   One all-data model is about as good as a 5-fold cross-fit (Tian, MEASURED) at 1/4 of the GPU time.
4. **Labels: mix50_logit by default.** It is stronger than v4 even when judged on v4 labels (OOF .8811 vs .8692,
   MEASURED). The blend model says strength is the binding variable once rho_eff >= .88 (sec 3.3). Tonight's K2 solo
   read can switch this to an alternating mix50/v4 hedge (sec 5.1). Gated labels: no new runs.
5. **Recipe: TTR vs F1 is decided by G-T2** (~19:10 if gated stops, ~22:45 if not). PASS -> TTR leads and F1 adds
   recipe diversity; KILL -> F1 only, plus one EV2S diversity test; MIDDLE -> a TTR fold-0 run on mix50 decides.
6. **Capacity is not the limit.** At ~6.1 h per member (alone) the box does about 4 members a day: ~25 job slots by
   10-11 08:00, ~41 by 10-15, ~57 by 10-19 10:00 (MODEL from MEASURED rates). The limits are the serving window,
   diminishing returns and LB reads.
7. **Serving cap:** about 8 F1-equivalent members (one pass) fit if the parent takes 8.0 h, 14 at 7.7 h and 27 or
   more at <= 7.0 h (MODEL from the MEASURED T4 microbench of 296-304 ms/member/study). A TTR member costs ~0.5
   F1-equivalent (MODEL). Plan the final at <= 8 F1-equivalents behind a valve-tolerant main file.
8. **Spare capacity goes to own-OOF round 2** (TTR or F1 folds 1-4, own-OOF silent-cell refill, gated fold-0 test,
   then round-2 seeds). It is the only lever left that might add about +.01 to member strength (GUESS, optimistic
   analog). It serves the 10-22 final, not 10-12.
9. **Honest expectation:** final fleet solo S ~.925-.93 (KILL) or ~.93-.94 (PASS), blend lift +.000 to +.004, so
   public ~.943-.947 (MODEL on GUESS inputs). P(>= .950 from our own fleet) <= 5% (GUESS). The fleet is mainly a
   merge asset; a partner gets an independent ~.93 leg.
10. **Tonight's pivots:** K2 solo S1 (~17:50) sets the labels and the merge trigger; K2 blend gives rho_eff
    (inversion table, sec 5.1); G-T2 sets the recipe. Every queue row below is keyed to those three reads.

---

## 1. Measured state of every fleet run

### 1.1 Per-epoch times and trajectories

All rows MEASURED from `~/fleet_runs/<run>.log` (and process start times from `ps`, 15:00 IST). "Epoch" = the
`epoch X min` on the `ema` line, which includes both evals.

| Run (labels) | Start / end | Mode | s/it (steady) | Train min/ep | Epoch incl. eval | Best (val AUC vs v4 / gold58) |
|---|---|---|---|---|---|---|
| `f0_s42` F1 320, mix50_logit, seed 42 | 10-03 21:07 / 10-04 09:37 | shared with f0_s42_v4 | 2.22-2.28 (ep0 2.57) | 64.2-66.0 (ep0 73.4) | 73.3-75.5 (ep0 82.7) | raw ep8 (1-based) .8811 / .9143; EMA ep10 .8770 / .9111 |
| `f0_s42_v4` F1 320, v4, seed 42 | 10-03 22:08 / 10-04 10:12 | shared ep1-9 (part of ep9 alone) | 2.22-2.34 | 64.2-65.9 | 73.3-75.1; ep10 58.1 | raw ep8 .8692 / .8610; EMA ep10 .8650 / .8594 |
| `f0_s1042_ttr` TTR 224, v4, seed 1042, 24 ep | 10-04 11:09:38 / running | shared with gated | 0.82-0.83 (ep0 0.87) | 23.6-23.9 (ep0 25.1) | 28.6-29.0 (ep0 30.1) | so far raw ep8 .8597 / .8650; EMA ep8 .8338 / .8417 |
| `f0_s42_gated` F1 320, gated75_logit, seed 42 | 10-04 11:09:43 / running | shared with TTR | 2.20-2.23 | 63.7-65.0 | 72.8-73.9 | so far raw ep3 .8160 / .8243 |

(Epochs in this column are 1-based; the logs print 0-based `ep 0..9`.)

Paired trajectories, raw, val AUC vs v4 on the 867 / gold58 (MEASURED, logs):

| Epoch (1-based) | F1 mix50 | F1 v4 | F1 gated | TTR v4 (24-ep cosine) |
|---|---|---|---|---|
| 1 | .7486 / .7478 | .7238 / .6953 | .7305 / .7110 | .7207 / .6913 |
| 2 | .8079 / .8267 | .7912 / .7825 | .7950 / .8099 | .7753 / .7736 |
| 3 | .8413 / .8567 | .8160 / .8115 | .8160 / .8243 | .8155 / .8127 |
| 4 | .8613 / .8808 | .8403 / .8359 | (running) | .8245 / .8226 |
| 5 | .8742 / .9002 | .8603 / .8476 | | .8395 / .8514 |
| 6 | .8748 / .9103 | .8584 / .8588 | | .8502 / .8563 |
| 7 | .8772 / .9126 | .8646 / .8618 | | .8538 / .8588 |
| 8 | **.8811 / .9143** | **.8692** / .8610 | | .8597 / .8650 |
| 9 | .8788 / .9105 | .8666 / .8596 | | |
| 10 | .8794 / .9110 | .8673 / .8601 | | |

OOF rho vs the Raptor proxy on the 867 (the proxy is in-sample on train studies, so only the gap counts):
mix50 raw .889, v4 raw .840 (MEASURED, `~/fleet_runs/gate_history.log`, 10:13 block).
Today's gold58 (from the task brief, MEASURED there): mix50 member gold58 .9144, Spearman vs the parent .924, vs
raptor_coat_side .921.

### 1.2 Concurrency penalty (the key timing fact)

- **F1 alone = 1.08-1.09 s/it** (MEASURED, two independent windows in `f0_s42_v4.log` epoch 10):
  - it 600-800: 1.09 s/it, 09:36:30-09:40. f0_s42 had exited at 09:37 (its last checkpoint is stamped 09:37). The
    gold58 parent run started at 09:40 (`~/kmimic/runs/gold58_gpu/kaggle` stamped 09:40).
  - it 1600-1733: 1.08 s/it, 10:05:40-10:08. The parent had finished at 10:04 (`localLB_parent_preds.csv` 10:04).
  - The windows in between (1.5-2.7 s/it) are the parent sharing the GPU.
  - The log prints cumulative averages, so windowed rates come from differencing: (n2*a2 - n1*a1)/(n2-n1).
- **F1 shared = 2.22-2.25 s/it**, whether the partner is another F1 job or TTR (MEASURED, all four logs).
  That is 2.05-2.07x the alone time.
- **Aggregate:** two F1 jobs = 2/2.25 = 0.89 it/s, against 1/1.085 = 0.92 it/s for one job alone. **0.96x: sharing
  loses ~4%** (MEASURED).
- **TTR alone (MODEL):** next to TTR, F1 keeps 1.085/2.22 = 49% of its alone speed, so TTR holds ~51% of the GPU.
  TTR alone is then about 0.83 x 0.51 = **~0.42 s/it** (range 0.40-0.55 if TTR has CPU-side gaps). That gives ~12.1
  train + ~2.6 eval = **~14.7 min per fold-0 epoch**. Aggregate with sharing: ~1.0x.
- **Eval alone:** 2.0 min per tag on 867 + 74 studies for F1 (MEASURED, v4 epoch-10 lines).
- **Old assumption replaced:** ENDGAME_V3 used "alone = shared / 1.8, aggregate ~1.1x". MEASURED is shared / 2.06
  and aggregate 0.96x.
- **Memory:** TTR 3.0 GB, F1 7.0 GB, plus 0.3 GB desktop (MEASURED, `nvidia-smi` 14:59).

**Consequence:** run one job at a time. Total work is the same, but each gate reads in half the wall time.
Exception: a 20-min G-TTA bench capped at 3 GB is fine in the gap between two jobs.

### 1.3 Alone-time per job type (used everywhere below)

| Job | Steps | Per epoch, alone | Epochs | Wall, alone | Tag |
|---|---|---|---|---|---|
| F1 fold-0 (867 val) | 1,733 it | 31.3 train + 4.0 eval = 35.3 min | 10 | **5.9 h** | MODEL from MEASURED rates |
| F1 all-data (4,333 train, gold-only eval) | 2,167 it | 39.2 + 0.4 = 39.6 min | `--stop_epoch 8` (fold-0 best) | **5.3 h** | MODEL from MEASURED rates |
| TTR fold-0 | 1,733 it | ~12.1 + 2.6 = ~14.7 min | 24 | **~5.9 h** (5.4-7.4) | MODEL (TTR alone rate) |
| TTR all-data | 2,167 it | ~15.2 + 0.3 = ~15.5 min | 24 | **~6.2 h** (5.7-8.0) | MODEL |
| EV2S fold-0 (family 2) | 1,733 it | ~0.9x F1 | 10 | **~5.3 h** | MODEL (fleet2 SCHEDULE ratio) |
| G-TTA bench, passes A,B, 20-min cap | | | | **0.35 h** | BUILD.md sec 4 |

Add ~5 min per job for launch, the first slow iteration (25-65 s MEASURED) and model load.

---

## 2. Where the running jobs end

| Scenario | TTR f0 ends | G-T2 read | Gated ends | Next job starts |
|---|---|---|---|---|
| **A. Eesh stops gated at ~15:30 (recommended)** | ~19:10 (range 18:55-20:10) | ~19:15 | stopped (ep3 raw ckpt .8159 stays on disk) | ~19:45 (after G-TTA) |
| B. Both run on | ~22:45 (16 ep x 29.0 min after 15:01) | ~22:50 | ~23:05 (last ~0.5 ep alone) | ~23:10 |

- Scenario A is MODEL: the TTR epoch that starts ~15:30 runs alone, then 15 more at ~14.7 min each.
- Scenario B is MEASURED-rate projection.
- The brief's "~21:30" used 26.4 min/epoch, which is the raw-line time; the full cycle is 29.0 min (MEASURED).
- **Pre-registered check after a stop:** TTR's `s/it` at `ep N it 200` must read <= 0.55. If it reads > 0.60, TTR is
  CPU/loader-bound alone, sharing was hiding stalls, and the serial rule should be re-checked (one more data point,
  not a reversal by itself).
- Stop command (Eesh only; it checks the PID's command line first and touches nothing else):
  ```bash
  ssh -o BatchMode=yes training@192.168.10.30 'if ps -o args= -p 267280 | grep -q labels_gated75_logit; then kill 267280; sleep 3; ps -p 267280 >/dev/null && echo STILL RUNNING || echo "gated stopped $(date)" | tee -a ~/fleet_runs/gate_history.log; else echo "267280 is not the gated run: nothing killed"; fi' </dev/null
  ```

---

## 3. Decisions with numbers

### 3.1 Should the gated run continue? **No (recommended stop).**

| Item | Value | Tag |
|---|---|---|
| Remaining work | 7 epochs = ~8.5 h shared, ~4.1 h alone-equivalent | MODEL (MEASURED rates) |
| What it costs | G-T2 moves from ~19:10 to ~22:45; the whole queue shifts +3.6 h; or 0.75 of an all-data production seed | MODEL |
| What it answers | G-L: gated vs v4 at the F1 recipe, paired seed | plan |
| Evidence so far | Epoch 3 (1-based): gated val .8160 = v4 .8160; gold58 .8243 vs v4 .8115 (+.013) vs mix50 .8567 (-.032) | MEASURED |
| Expected final member | gold58 ~.876 (gated closes ~28% of the v4-to-mix50 gap at epoch 3, applied to the final .861 vs .914) | MODEL (linear interpolation) |
| Decision value | The rho step from v4 to gated to mix50 is worth about +/-.0005 of blend at fixed strength (sec 3.3 table); mix50's strength lead is worth more | MODEL |
| Caveat | gold58 favours the gated targets (their Raptor teacher was checkpoint-picked on these 58), so +.013 is an upper bound | BIG_JUMP review |

If Eesh keeps it, nothing breaks: use scenario B and shift every time below by +3.6 h. The finished gated f0 joins K3
as one more fold-0 member.

### 3.2 Fold models or all-data models? **All-data for production, fold-0 only for gates.**

| Fact | Value | Tag |
|---|---|---|
| 5 x 80% cross-fit vs 1 all-data model | equal | MEASURED (TIAN:100, via BIG_JUMP sec 1.1) |
| GPU cost | 5 folds = 5 x 5.9 = 29.5 h; 1 all-data F1 = 5.3 h; 1 all-data TTR = ~6.2 h | MODEL (MEASURED rates) |
| Serving cost | 5 folds = 5 members of T4 time, 1 all-data = 1 | MEASURED microbench, sec 4 |
| Selection without OOF | Fixed epoch from the fold-0 curve: F1 raw at epoch 8 (best for BOTH label sets); TTR = G-T2's best epoch | MEASURED curves |
| Sanity check without OOF | The trainer logs gold58 for raw and EMA every epoch in all-data mode (`fleet_train.py:377-389`); gold58 is clean (gold74 excluded) | code fact |
| ConvNeXt-T folds vs 1 | XuKong 5-fold vs 1-fold -.001 | CLAIMED (LEDGER:45) |

Fold models still earn a place only as (a) gate tests (OOF strength and rho on the 867) and (b) the own-OOF round-2
folds (sec 6, from 10-06). The fold-0 members we already have (mix50, v4, TTR) ship as free extra members while the
serving budget allows.

### 3.3 Which labels? **mix50_logit by default; K2 solo can switch to a mix50/v4 hedge; no new gated runs.**

Strength (MEASURED):
- On v4 labels (an instrument that favours the v4 student), mix50 OOF is +.012 (.8811 vs .8692).
- On gold58, mix50 is +.053 (.9143 vs .8610). This gap is inflated because the teacher was checkpoint-picked on
  gold58.

Correlation (MEASURED):
- OOF rho vs the Raptor proxy: mix50 .889 vs v4 .840 (+.05).
- gold58 rho vs the parent is .924 for mix50. v4 was not measured (sec 8).

What a gap is worth (MODEL, `sched_binormal.out`, best-W lift over .943):

| rho_eff \ S | .915 | .920 | .925 | .930 | .935 | .940 |
|---|---|---|---|---|---|---|
| .80 | +.0012 | +.0020 | +.0030 | +.0043 | +.0059 | +.0078 |
| .84 | +.0002 | +.0007 | +.0015 | +.0026 | +.0041 | +.0060 |
| .88 | .0000 | .0000 | +.0004 | +.0011 | +.0024 | +.0041 |
| .92 | .0000 | .0000 | .0000 | +.0001 | +.0008 | +.0023 |

- **Exchange rate:** -.04 of rho_eff is worth about the same as +.005 to +.010 of S. v4 gives up about .012 OOF
  (more on gold58) for about .05 of student rho. That is roughly a wash at best, and a loss if the LB strength gap
  is nearer the gold58 gap. So mix50 >= v4 for blend value (MODEL), and mix50 is clearly better as a stand-alone or
  merge asset.
- **The LB says rho_eff is high whatever the labels** (MODEL inversion of MEASURED reads): the v4-label crop leg had
  gold58 rho ~.78-.81 vs one Raptor arm, but its reads (crop15 .942, crop25 .941, K1 .942) invert to rho_eff ~.9 or
  more. That inversion is confounded by the K=24 vs K=8 serving mismatch in the old probes. Still, it does not
  support buying decorrelation with strength.
- **Gated** sits between v4 and mix50 on both axes (sec 3.1). No new gated runs.

### 3.4 Which recipe? **Decided by G-T2 (sec 5.2).**

- PASS: TTR leads. It is also cheaper to serve (~0.5 F1-equivalent per member, MODEL).
- KILL: F1 only.
- MIDDLE: one TTR fold-0 run on mix50 decides (sec 5.2).

TTR now: raw epoch 8 is .8597 vs v4-F1 at the same epoch .8692, with 2/3 of its cosine still to run. Recent gain is
about +.005 per epoch (MEASURED). GUESS: final best .872-.885; P(PASS >= .8772) ~45%, P(MIDDLE) ~40%, P(KILL) ~15%.

### 3.5 How many members by 10-12 / 10-15 / 10-22

| Horizon | Alone-hours from 19:45 10-04 (scenario A) | Job slots at ~6.1 h | Used by the queue below | Tag |
|---|---|---|---|---|
| 10-11 08:00 (last finish that a 10-11 read can carry) | ~156 h | ~25 | ~21 (4 tests, ~8 production, 4 round-2 folds, 1 round-2 test, ~4 round-2 seeds) | MODEL |
| 10-14 12:00 (last finish for a pre-merge-deadline read) | ~233 h | ~38 | +~12 seeds or merge-driven work | MODEL |
| 10-19 10:00 (checkpoints off the box for the final, compute.md sec 9) | ~350 h | ~57 | final seeds; last same-recipe start 10-18 12:00 | MODEL |

The final kernel can serve about 8 F1-equivalents (sec 4), and same-recipe seeds saturate:
- k = 1/2/3/5 equal members at pairwise rho .95 give .920 / .9226 / .9235 / .9242 (MODEL, `sched_binormal.out`).
- At pairwise rho .90 the same counts give .920 / .9253 / .9271 / .9285 (MODEL).
- Tian MEASURED +.003 for a 2nd seed and +.001 for a 3rd.

**Target final fleet:** about 8-10 all-data members across 2 recipes or label sets (3-4 seeds each), plus the best
1-2 fold-0 members as valve-droppable extras.

---

## 4. Serving in the hidden rerun (N ~1,300 placeholder)

Inputs:
- Fleet stage = setup ~2 min + max(decode D, GPU) + ~5 min safety. Decode and GPU overlap (worker design, RUNBOOK
  sec 6).
- D = 15-20 min central, ~30 min slow (0.68 s/study MEASURED by the prep kernel; the 3-study commit read
  1.48 s/study/thread, MEASURED in `SP/k2/build/out_k2v1/fleet_receipt.json`).
- GPU per F1 member per pass, per T4 = 650 studies x 0.30 s = **3.25 min** (296-304 ms/member/study MEASURED, same
  receipt).
- TTR per member: ~1.6 min (MODEL: x0.49 compute at 224 px, 24 slots).
- EV2S: ~3.0 min (MODEL: x0.91).

Window: the stage must end by the 8.6 h process-age wall, so the window is 8.6 h minus P (the parent's runtime).

| Parent P | Window | F1-equivalents, pass A | With 2-pass TTA | Note |
|---|---|---|---|---|
| 6.0 h (MEASURED turnaround, one session) | 156 min | ~45 | ~22 | decode-bound below ~5 members |
| 7.0 h | 96 min | ~27 | ~13 | |
| 7.5 h | 66 min | ~18 | ~9 | |
| 7.7 h (ENDGAME low) | 54 min | ~14 | ~7 | |
| 8.0 h (parent's own cap) | 36 min | ~8 | ~4 | slow decode (30 min) leaves ~0 margin |
| 8.05 h | 33 min | ~8 at central decode, 0 at slow decode | 0-4 | valve must drop pass B, then members |

- All rows are MODEL from the MEASURED rates above.
- Fresh MEASURED evidence of slow sessions: `rsna-sub-v9crop25` (ref 56799619) shows COMPLETE with no public
  score in today's submission list (read 15:10 IST). Earlier: 2 K=24 crop members timed out, and 3 member-add
  kernels gave no score. So P = 6.0-8.05 h stays the working range.
- **Rule for the final:** <= 8 F1-equivalents in pass A (for example TTR x8 + F1 x4 = 8.0, or F1 x6 + TTR x4 = 8.0).
  Pass B only if G-TTA passes and only behind the valve.
- The main `submission.csv` must be valve-tolerant (written with the surviving members). Solos stay fail-loud.
  This is a build item (B6), not a box item.

---

## 5. Pre-registered decision points tonight

### 5.1 K2 reads (~17:50 IST; refs 56816035 SOLO, 56816039 BLEND W .30; both PENDING at 15:10, MEASURED)

**Solo S1** (one F1 mix50 fold-0 member). Pre-registered expectation: .910-.925, central ~.918 (GUESS anchors:
Dread single CoAtNet gold58 .9167 -> LB .924; ours gold58 .9144).

| S1 read | Labels L* for every new member | Box consequence | Merge |
|---|---|---|---|
| >= .918 | **mix50_logit** | full production queue | per rho_eff below |
| .910-.917 | **alternate**: in each recipe, odd seeds mix50, even seeds v4. K3 also carries a v4 member so K3's solo moves | full production queue | post merge requests 10-05/06 |
| < .910 | v4 / mix50 alternate; gold58 did not transfer to the hidden set | production at 1 seed/day, as merge/private asset only; spare GPU to round 2 | post merge requests 10-05 (fleet cannot lift the public parent) |
| error / no score | unchanged | box unaffected; build side diagnoses runtime or decode | |

**Blend B30** gives rho_eff, inverted from (S1, B30) (MODEL, `sched_binormal.out`; displayed value, so +/-.0005
rounding plus ~.0012 read noise, about +/-.04 in rho):

| S1 \ B30 | .940 | .941 | .942 | .943 | .944 |
|---|---|---|---|---|---|
| .910 | .86 | .84 | .81 | .79 | .76 |
| .915 | .89 | .86 | .84 | .82 | .79 |
| .920 | .92 | .89 | .87 | .84 | .82 |
| .925 | .95 | .92 | .90 | .87 | .85 |

| rho_eff | Meaning (MODEL, sec 3.3 table) | Action |
|---|---|---|
| <= .86 | Route live: S .93 gives +.0026..+.0043 | Full queue; K3 main at W* (sec 5.3); target K5 >= .946 |
| .86-.90 | Lift <= +.001 at S .925, +.0011..+.0026 at S .93 | Full queue (strength is the lever); W* .15-.25 |
| >= .90 | Lift ~0 for any S < .935 | Production continues as a merge asset. Tell Eesh the merge is the only >= .945 route; post requests 10-05 |

Expected (GUESS): B30 .940-.942 (the HANDOFF pre-registration said .940-.943, flat).

### 5.2 G-T2 (TTR fold-0 vs f0_s42_v4)

Read with `gt2_read.py` (BUILD.md sec 3.5, CPU, read-only):
- PASS: TTR best (max raw/EMA over epochs) >= .8692 + .008 = **.8772**, AND no label below -.02, AND gold58 not
  lower (>= .8618, v4's best-region gold58).
- KILL: TTR best <= **.8692**.
- MIDDLE: in between.
- Selection-bias note: TTR picks from 48 candidates vs 20, worth ~+.001-.002 (GUESS). Cross-check with the
  top-3-epoch mean from `epochs/`.
- Early-warning read (no action, just calibrates expectations): at epoch 16 (2/3 of the cosine) a raw value < .866
  makes KILL likely; >= .872 makes PASS likely (GUESS).

| Verdict | J1 (first job after G-TTA) | Lead recipe for production |
|---|---|---|
| PASS | TTR fold-0, labels L*, seed 1043 (recipe test under the production labels + a member) | TTR; F1 every 3rd job for diversity |
| MIDDLE | TTR fold-0, L*, seed 1043 | Decided by J1: OOF >= .8811 + .005 = **.8861** (F1 mix50 + .005) -> TTR leads; else F1 leads and J1 ships as one member |
| KILL | F1 all-data, L*, seed 43, `--stop_epoch 8` | F1; an EV2S fold-0 test as the diversity probe |

J1 under PASS is a fold-0 test (not straight to all-data) because TTR's PASS is measured on v4 labels. Heavy
augmentation and a long schedule may help less on soft teacher-mixed targets (GUESS). One fold-0 run is a cheap
insurance at 5.9 h.

### 5.3 W* for K3 and later (MODEL, best-W from `sched_binormal.out`)

| rho_eff \ S | .920 | .925 | .930 | .935 | .940 |
|---|---|---|---|---|---|
| .84 | .16 | .23 | .30 | .38 | .45 |
| .88 | .04 | .13 | .23 | .33 | .44 |
| .92 | 0 | 0 | .09 | .24 | .40 |

- S = the K-read solo of the same fleet structure, or the previous read + the MODEL ensemble increment.
- W error costs <= .0005 across +/-.10 near W* (MODEL).

---

## 6. Day-by-day queue

- Times IST; scenario A (gated stopped ~15:30). Scenario B: add +3.6 h to everything through 10-05.
- Durations from sec 1.3 (MODEL from MEASURED rates); +5 min gap per job.
- L* = the K2 label rule (sec 5.1). Seeds: F1 all-data 43-49, TTR fold models 1043-1049, TTR all-data 1001-1019,
  EV2S 142-149.
- Every job runs alone. Overnight jobs are chained by a wait-for-PID launcher (the `auto_fold0.sh` pattern) that
  Eesh arms before bed.

### 10-04 (Sun)

| Start | End | Job | Recipe / labels / fold / seed | Gate and pre-registered read |
|---|---|---|---|---|
| running | ~19:10 (A) / ~22:45 (B) | TTR f0 | TTR 224, v4, fold 0, s1042 | G-T2 (sec 5.2) |
| ~15:30 | | **Eesh: stop gated** | | TTR `s/it` <= .55 within 10 min (sec 2) |
| ~17:50 | | K2 reads | | S1 -> L*; B30 -> rho_eff (sec 5.1) |
| ~19:15 | 19:40 | G-TTA bench | passes A,B, `--max_minutes 20 --max_gpu_gb 3`, members f0_s42 mix50 raw+EMA (+ TTR best raw) | PASS if A+B - none >= +.002 OOF with P > .9 and gold58 not lower -> serve A+B behind the valve |
| ~19:45 | ~01:45 | **J1** | PASS/MIDDLE: TTR f0, L*, s1043, 24 ep, `--save_epochs 6-24`. KILL: F1 all-data, L*, s43, `--epochs 10 --stop_epoch 8` (ends ~01:05) | TTR: G-T0 epoch 1 <= 18 min alone; G-T1 raw ep3 >= .80. F1: gold58 at ep8 >= .89 (mix50) / >= .84 (v4) |

### 10-05 (Mon)

| Start | End | Job | Recipe / labels / seed | Gate / read |
|---|---|---|---|---|
| ~01:50 | ~08:00 | **J2** (chained) | PASS: TTR all-data, L*, s1001, `--save_epochs <best-3>-24`. MIDDLE: F1 all-data L* s43 (ends ~07:10). KILL: F1 all-data L* s44 (~06:25) | gold58 at the fixed epoch within .02 of the fold-0 member of the same recipe/labels (bug guard only, n=58) |
| ~08:00 | | Pack K3 | f0 mix50 (K2), f0 v4, TTR f0 v4 (if PASS/MIDDLE), J1, J2; re-key with `pack_fleet_ds.py --tag` | `k2_check.py --kaggle_commit` |
| ~10:00 | | **Eesh: K3** push + submit main (W*, sec 5.3) + solo | 2 commits, ~0.8 T4-h | reads ~16:00-22:00: G-K3 = K3 solo minus K2 solo. The ensemble should add >= +.004 (MODEL, k 3-5 at rho_m .90-.93). Less than +.002 -> members too alike, so switch the next seeds to the other recipe or labels |
| ~08:05 | ~13:25 | **J3** | PASS: F1 all-data L* s43 (diversity). MIDDLE: lead per J1's OOF (TTR all-data s1001 or F1 s44). KILL: `f2_preflight.py` then EV2S f0 s142 (~13:30) | EV2S: G1c (within-class rho <= .86) AND OOF within .005 of the lead family's, else no EV2S seeds |
| ~13:30 | ~19:40 | **J4** | lead recipe, all-data, s1002 / s45, labels per L* rule | |
| ~19:45 | ~01:50 | **J5** (chained) | second recipe (PASS: F1 s44), or lead seed | |

### 10-06 (Tue)

| Start | End | Job | Gate / read |
|---|---|---|---|
| ~01:55 | ~08:10 | J6: lead all-data seed (TTR s1003 / F1 s46) | |
| ~08:15 | ~13:30 | J7: PASS = EV2S f0 s142 (diversity test, `f2_preflight` first). KILL = F1 all-data s47 if EV2S failed, else EV2S all-data s143 | G1c + solo parity, as above |
| ~13:35 | ~19:50 | J8: lead all-data seed (TTR s1004) | |
| ~19:55 | ~02:00 | J9: **round-2 fold 1**, lead recipe, L*, s1044 (F1 s42-style fold runs under KILL) | start of own-OOF round 2 |

### 10-07 (Wed)

| Start | End | Job | Gate / read |
|---|---|---|---|
| ~02:05 | ~08:05 | J10: round-2 fold 2 (s1045) | |
| ~08:00 | | Pack K4 = J2-J8 all-data + best fold-0s, <= 8 F1-equivalents, valve order: TTR, F1, EV2S interleaved | K4 main + solo, read ~17:00-23:00. K4 solo >= K3 solo + .002, else stop adding same-recipe seeds |
| ~08:10 | ~14:10 | J11: round-2 fold 3 (s1046) | |
| ~14:15 | ~20:15 | J12: round-2 fold 4 (s1047) | |
| ~20:20 | ~20:45 | CPU: build round-2 labels | Own 5-fold OOF fills report-silent cells at .75; L* elsewhere; same format as `labels_gated75_logit.csv`; md5 + range checks |
| ~20:50 | ~02:50 | J13: **round-2 test** = lead recipe fold 0, round-2 labels, s1048 | G-R2: OOF (vs v4) >= J1 + .005 AND gold58 >= J1's AND rho vs Raptor proxy <= J1's + .02 |

### 10-08 (Thu) to 10-10 (Sat)

| Window | Jobs | Gate / read |
|---|---|---|
| 10-08 03:00 to 10-10 03:00 | ~8 jobs. G-R2 PASS: round-2 all-data seeds s1011-s1018 (alternate with 1-2 L* seeds for diversity). G-R2 FAIL: L* seeds of the second recipe, or a decorrelation probe (V3D fold-0 to epoch 6, models3d gate: kill if AUC vs v4 < .815 or rho vs Raptor proxy >= .79; needs build work first) | G-T0 per job; gold58 bug guard per all-data seed |
| 10-09 ~20:00 | Pull checkpoints for K5 (members finished by then) | |
| 10-10 ~09:00 | **Eesh: K5 push** (pre-merge final structure) + solo; reads by ~23:00 | K5 main = the number partners see. If K5 < K4 - .001, K4's structure stands |

### 10-11 (Sun) to 10-15 (merge deadline 23:59 UTC = 10-16 05:29 IST)

- 10-11: merge decision point (Eesh). The box keeps seeding the K5 structure (about 4/day).
- If merged: the partner's needs take the box. Our fleet becomes their extra leg (send predictions/OOF only after
  the merge, per the rules).
- Last finish that a pre-deadline read can carry: ~10-14 12:00 (submit by ~17:30 for a same-day read).

### 10-15 to 10-22 (final)

- Seeds of the final structure only, no new recipes after 10-16 09:00 (compute.md sec 9).
- Last same-recipe seed start: **10-18 12:00** (alone).
- Checkpoints off the box: **10-19 10:00**. Pack + verify by 10-19 12:00. Dataset "ready" by 10-19 15:00. Final
  commits 10-19 23:00. Submit 10-20 05:30. Picks locked 10-21 23:30.
- Final kernel shape: <= 8 F1-equivalents, valve-tolerant main, same runtime shape as an already-scored K read.

---

## 7. Launch templates (Eesh runs; guards as in BUILD.md)

Common guard: refuse if any GPU process is running (serial rule) or the output dir exists. Paths MEASURED on the
box: `~/rsna_train/{labels_mix50_logit,llm_labels_v4_blend,labels_gated75_logit,folds5,gold74_eval_set}.csv`.

```bash
# guard prefix (one job at a time):
# (process_name is required: the desktop's gnome-remote-desktop holds ~0.3 GB of GPU, so count python only)
G='NP=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -c python); [ "$NP" -gt 0 ] && { echo "GPU busy: NOT LAUNCHING"; exit 1; }'
COMMON='--store ~/fleet_store --folds ~/rsna_train/folds5.csv --exclude ~/rsna_train/gold74_eval_set.csv --eval_labels ~/rsna_train/llm_labels_v4_blend.csv --gold ~/rsna_train/gold74_eval_set.csv'
# J1 TTR fold-0 on L* (L*=~/rsna_train/labels_mix50_logit.csv by default):
cd ~/bigjump_build/code && nohup ~/miniconda3/envs/rsna/bin/python fleet_train.py --model ttr --aug heavy --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs 24 --seed 1043 --save_epochs 6-24 --workers 10 $COMMON --labels <L*> --fold 0 --out ~/fleet_runs/f0_s1043_ttr > ~/fleet_runs/f0_s1043_ttr.log 2>&1 < /dev/null &
# TTR all-data seed (E = G-T2 best epoch, 1-based):
... --model ttr ... --seed 1001 --fold -1 --epochs 24 --save_epochs $((E-3))-24 --labels <L*> --out ~/fleet_runs/fA_s1001_ttr
# F1 all-data seed (fixed epoch 8 of the 10-epoch cosine; no opt-in flag = byte-identical default path):
... fleet_train.py $COMMON --labels <L*> --fold -1 --epochs 10 --stop_epoch 8 --seed 43 --out ~/fleet_runs/fA_s43
```

- `--epochs` must be given: the trainer default is 12 (`fleet_train.py:72`), and the fold-0 curves come from 10.
- Under the alternating-label rule, the member key needs `pack_fleet_ds.py --tag` when two runs share a seed
  (they should not; give every run its own seed).

---

## 8. What could not be measured here

1. **v4 / TTR / gated rho vs the parent on gold58.** The direct number for the label decision would be per-study
   gold58 predictions of the v4 raw, TTR-best and gated checkpoints, run through the same Spearman and OOS-blend code
   as `~/k2_test/gold58_oos_blend.py`.
   - The trainer logs only macro gold58, so these predictions do not exist yet.
   - A 4-study timing probe ran at 15:05: `tta_bench.py --device cpu`, 43 s, output
     `~/k2_test/analysis_wf/sched/tiny`. It measured ~10 s/study on CPU, so ~10 min per checkpoint for gold58 + 2
     val studies.
   - The full CPU job (3 checkpoints, nice 19, writes only under `~/k2_test/analysis_wf/sched/`) was **blocked by
     the session's permission policy**, so it did not run.
   - If Eesh approves, it is a ~30-min CPU job. Pre-registered use: if v4's gold58 rho vs raptor_coat_side is
     <= .86 (vs mix50 .921) AND its OOS blend into raptor_coat_side at W .15 is > mix50's -.0001, switch L* to the
     alternating hedge even if S1 >= .918.
2. **TTR alone speed** is MODEL (~0.42 s/it). It is measured within 10 min of the gated stop.
3. **The LB strength gap mix50 vs v4** is unmeasured. Only K-reads can measure it, which is why the S1 rule hedges
   in the .910-.917 band.
4. **Parent runtime P** on any given hidden session cannot be measured. The valve absorbs it.
5. **Own-OOF round 2 gain** is a target-level analog with gold-picked teachers (optimistic). G-R2 is its first real
   measurement (~10-08 03:00).

---

## 9. Expected value (honest)

| Branch | Final fleet solo S | rho_eff | Blend lift | Public | Tag |
|---|---|---|---|---|---|
| KILL, F1 mix50 x5-6 all-data + 2 fold-0 | .925-.930 | .88-.92 | +.000 to +.001 | .943-.944 | MODEL on GUESS S1 .915-.920 |
| PASS, TTR x5 + F1 x3 | .930-.940 | .88-.92 | +.001 to +.004 | .944-.947 | MODEL |
| PASS + round-2 gain +.005 | .935-.945 | .88-.92 | +.002 to +.006 | .945-.949 | GUESS |

- P(own-fleet public >= .950 by 10-12) <= 5% (GUESS). P(>= .945) ~30-40% (GUESS).
- The schedule maximizes S because S is the variable that moves the lift once rho_eff >= .88. The box cannot buy
  the jump alone. The merge (two-way, 10-15) remains the main route to >= .95, and this fleet is what we bring to it.
- The first falsifiers all land before midnight tonight: K2 S1, K2 B30 and G-T2.
