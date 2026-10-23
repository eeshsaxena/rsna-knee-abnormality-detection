# OAI_NEXT: what the V100 runs next (2026-10-09, final 13:00 IST)

Box `~/oai_build`. CPU work only. No GPU, no queue edits, no OAI byte off the box. Tags: **MEASURED** / **MODEL** / **GUESS**.

## TL;DR
1. **T1 probe verdict:** **PASS** (MEASURED 12:54, registered scorer; result paired with `f0_s1043_ttr_frac75`):

   | gate | probe | control | bar |
   |---|---|---|---|
   | G1 | .8919 | .8896 | .8876 |
   | G2 OOF T1 | .8668 | .8663 | .8653 |
   | G2 gold58 T1 | .8262 | .8192 | .8242 |
   | G3 | .9177 | .9163 | .9113 |
   | oof12 | .8856 | .8838 | |

   - The V100 idled 12:47-13:01: every old queue line was SKIP or REFUSE.
   - The coordinator installed `chain_T1pass.txt` at 12:55. **P2a LAUNCHED 13:01:39** (train 6,047 / val 867, inputs at
     the registered md5s). ETA ~18:25.
   - The selection then runs by itself: line 2's `via bash ~/oai_build/select_gate.sh`. The runner keys it by verdict
     file, runs it once after the GPU frees, CPU only, with a 90-min timeout. If `select_T1.verdict` were still missing,
     line 6 retries it.
   - Verified on the real T1 jsons: T1 report oof12 .8856 matches the registered scorer, and the selection writes
     correctly (tested into a temp dir; the real `out/` has no premature verdict).
   - Log to watch: `~/oai_build/out/select_gate.log`.
2. **T1 PASSED, so the chain is in (DONE 12:55, coordinator):** `~/oai_build/qlines2/chain_T1pass.txt`, 20 gated lines.
   It applies the pre-registered selection by itself (PROBE2_PREREG sec 6). Order:

   | step | job | MODEL time | lands (IST) |
   |---|---|---|---|
   | 1 | P2a T12-V00 probe | 5.4 h | ~18:30 10-09 |
   | 2 | automatic pick T1 vs T12v00 | CPU ~20 min | |
   | 3 | 224 folds 1-4 of the winner | ~24.5 h | ~19:30 10-10 |
   | 4 | P2b all-visits probe, once its data is READY | 11 h | |
   | 5 | 288 OAI f0 of the winner, paired with `f0_s1045_ttr288frac75` | 8.4 h | |
   | 6 | 288 f1-f4 on a 288 PASS | ~34 h | ~10-12/13 |

   - Step 3 is a 5-fold OAI 224 set for a 10-11 kit. It REPLACES set D in the N pick.
   - Step 6 is a 5-fold OAI 288 set for a 10-13 kit. It replaces set E in the N pick.
   - If P2b passes and wins the selection: 224 all-visits folds 1-4 (~51 h) still end ~10-15, before the 10-20 freeze.
3. **If T1 FAILS:** PROBE_PREREG says stop L6, and T2 is not tried on a failed T1. So no PROBE2 runs, unless Eesh
   overrides in writing (PROBE2_PREREG sec 9).
   - The V100 would otherwise idle: every remaining queue line is SKIP or REFUSE, because L5's out dir already exists.
   - Clean default: `qlines2/fail_branch_288frac75_s1046.txt` (288 frac75 reseed s1046, RSNA only, both picks, ~29 h).
4. **Data:** the V01 download started 11:25 at -wt 8, ~6 knees/min (3.3 MB/s, MEASURED). The V00 run did 37/min.
   - At 12:06 Eesh restarted it as `WT=24 bash ~/oai_work/get_oai.sh v01rest` (coordinator's rest-regex mode).
   - With WT=24 the conversion rose to ~17 knees/min (MEASURED 12:06-12:20; V01 461/2594 at 12:20), so V01 ends ~14:30.
   - After it, either:
     - three passwords: `WT=24 bash ~/oai_work/get_oai.sh v03rest` (then v06rest, v05rest); or
     - ONE password for all three: `bash ~/oai_work/get_oai_more.sh v03,v06,v05 24`. The regex is validated against
       image03 (16,425 series, 0 extra, 0 missing).
   - V03/V05/V06 are 4,102 knees, about 4 h at ~17/min. The all-visits union could be READY ~19:00-20:00 tonight
     if the next download starts right after V01 (MODEL). Either way, P2b's queue line waits for READY.
5. **Expected gain, honest:**
   - verify_D's own model: one OAI 224 set over K9N gives +.0001 to +.0004 central (range -.0005 to +.0014). A second
     (288) set roughly doubles that. Below what the LB resolves, and only in the N pick.
   - Follow-up visits have a negative public prior: nartaa's 12-month arm was "unsuccessful".
   - It is still the only lever left with a positive expected sign. Everyone at the top is adding OAI now.

## 1. T1 probe (L6) result
- Raw best ckpt: ep 15, cv .885613, vs the control .883777.
- **Where the gain is (OOF-vs-v4 per label, probe minus control):**

  | label | delta | supervised by OAI? |
  |---|---|---|
  | MCL | +.008 | no |
  | Fracture | +.008 | no |
  | Effusion | +.006 | no |
  | Lateral OA | +.006 | yes (T1) |
  | Medial OA | +.005 | no |
  | Medial Meniscus | +.004 | no |
  | Baker's | +.004 | no |
  | PF OA | +.002 | yes (T1) |
  | Lateral Meniscus | -.009 | no |
  | Synovitis | -.008 | yes (T1) |
  | Contusion | -.005 | no |

  - So most of the gain is on labels OAI does NOT supervise (G1 +.0023, against T1 OOF +.0005). The OAI knees act
    mainly as more anatomy / representation, not as label transfer (interpretation, GUESS).
  - That raises the prior for "more OAI knees" (P2u, P2b) a little, against nartaa's negative 12-month note. It makes
    T12's extra supervision a smaller, uncertain add.
- **PF OA skew did not hurt:** PF OA OOF went +.002 and gold58 +.007 despite the 77 % vs 45 % prevalence gap. No
  reweighting is needed for now.
- **Synovitis** fell on OOF (-.008) but rose on gold58 (+.004). Watch it in P2a, where Effusion is also supervised.
- gold58 T1 delta +.0070 (boot SD .0107, P>0 .75). gold74 macro .9096 vs .9083. The schedule-end OOF (last 3 epochs):
  12-label .8851 vs .8836.
- The PROBE_PREREG PASS branch says folds 1-4. PROBE2_PREREG adds P2a first: one 5.4 h probe that decides the labels of
  both following sets.
- Scoring note: two copies of `score_oai_probe.py` ran at the same time (the coordinator's watcher, pid 150502, and one
  more, 150904). They write identical outputs, so there is no harm.
- **Chain:** inserted by the coordinator at 12:55 (qctl install, 0 BAD). Do NOT also insert the old
  `qlines_oai_folds.txt`: the chain contains the T1 folds, gated by the selection.
- If you prefer to skip P2a and run T1 folds at once: insert `qlines_oai_folds.txt` instead, and lose the T12 question
  for both sets. Not recommended: P2a costs 5.4 h, and the 224 set still lands for a 10-11 kit.

## 2. What was built today (all on the box, CPU, nice 19)
| item | where | status |
|---|---|---|
| Multi-visit converter | `build_oai_store.py` (md5 a82c263a; V00 original kept as `.v00_1008`, md5 89ef97ad) | RUNNING since 11:25. The coordinator restarted it at 12:06 with `--procs 12 --poll 60`, now pid 148143. Log `~/oai_build/build_fu.log`. Never start a 2nd instance: a per-visit lock refuses it |
| Per-visit labels T1/T12/T123 | `build_oai_labels_visits.py` -> `labels_v/` | done; V00 md5s reproduced exactly |
| Union builder | `make_union2.py` (reproduces the T1 probe inputs byte for byte) | 4 unions built |
| PROBE2 pre-registration | `PROBE2_PREREG.md`: body fixed 11:5x before any PROBE2 run; sec 9 appended (dry runs, 288 control values, P2a launch); md5 7746f9d1 | done |
| Scorer | `score_oai_probe2.py` | selftest reproduces T1 control values exactly |
| Queue lines (NOT inserted) | `qlines2/` | 59 lines in 11 files, `queue_runner.sh --check` 0 BAD |
| Auto union rebuild | `watch_unions_allv.sh` | RUNNING, log `watch_unions.log` |
| Download helper | `~/oai_work/get_oai_more.sh` + `make_remaining_regex.py` (`get_oai.sh` untouched) | ready |

Converter details:
- Follow-up ids are `oai_<subj>_<R|L>_vNN`. They sort after every RSNA id and right after their own V00 id (asserted).
- Synthetic end-to-end test, all PASS:
  - old and new code give byte-identical V00 npz;
  - V01 arrays are identical to V00 for the same input;
  - stray images are dropped;
  - raw is pruned after the verified reload;
  - the lock and the idle exit work.
- On the real V00 records it is a no-op, with the same 2,680 ids.
- The first 148 V01 knees: plane / FS match 100 %, one stray per MPR dropped, views v0|v2|v3, contrast tokens as at V00,
  0 failures.

Union details:

| union | OAI rows | note |
|---|---|---|
| (a) T12v00 | 2,581 | labels md5 5235a4e0, folds md5 fd0f4fd3 (registered) |
| (b) T1allv | 8,662 when complete | pre-build |
| (c) T12allv | 9,130 when complete | pre-build |
| (d) T12uniq | 3,218 | pre-build |

- In every union, OAI rows are fold 9 only and the first 8 sorted ids are RSNA. ~/union_store (used by the running
  probe) and ~/fleet_store are never touched.
- `watch_unions_allv.sh` rebuilds (b), (c) and (d) once every visit is >= 97 % converted, then writes the READY
  verdicts that gate the all-visits queue lines.

Dry runs (CPU, the exact qline argv on the real trainer): T12v00 ALL PASS; T12allv ALL PASS (includes 4 real V01
knees). Checks:
- NaN cells are masked (finite loss).
- decode_probe is RSNA-only, and the full union's decode-probe ids equal the control's.
- No `oai_` id in the OOF or any csv.
- The ckpt args differ from the control only in train_labels + the dry overrides.

## 3. Labels: what the later visits look like (MEASURED, `labels_v/oai_labels_visits_report.txt`)
- **Rows per tier:**

  | tier | V00 | V01 | V03 | V05 | V06 |
  |---|---|---|---|---|---|
  | T1 | 2,399 | 2,594 | 1,742 | 674 | 1,253 |
  | T12 | 2,581 | 2,594 | 1,902 | 674 | 1,379 |

- **Physical knees:** 3,293 in all. V01 adds 613 knees never read at V00. V03, V05 and V06 add 0: they are re-scans of
  knees already seen.
- **Labels are very stable across visits.** Same-knee correlation vs V00:

  | visit | PF OA | Medial OA | Effusion |
  |---|---|---|---|
  | V01 | .95 | .96 | .73 |
  | V03 | .91 | .92 | |
  | V06 | | | .54 |

  So follow-ups act mostly as natural augmentation (new acquisition, same answer). They are not new information.
- **Trap fixed:** follow-up cartilage scores contain within-grade codes 0.5 / -0.5 (317 at V01; never at V00).
  - The V00 rule would turn 0.5 into "full-thickness loss" (target 1.0).
  - Now 0.5 is read as area grade .5 and -0.5 as 0 (GUESS meaning).
- **PF OA skew (honest):** OAI PF OA >= .5 in 77 % of V00 knees (V01 68 %, V03 78 %, V06 57 %), against 45 % in RSNA.
  - It is intrinsic: without the full-thickness override it is still 75 %.
  - Proposals, NOT applied: P1/P2 alternative definitions; P3 prevalence-matched masking, which would drop 75 % of the
    positive PF cells; P4 a loss weight, which needs a trainer patch.
  - The T1 probe's PF OA OOF delta (sec 1) is the evidence on whether it hurts.
- **Other label gaps:**
  - V05 has no Baker's read. V05 and V06 have no osteophyte read (at V00 only ~600 knees have one).
  - Effusion and Synovitis exist for only ~1/2 to 1/4 of the follow-up rows.

## 4. Why this V100 order (PASS branch)
- **P2a before folds.** It costs 5.4 h, and it decides the labels of BOTH sets that follow (224 and 288). Both orders
  land the 224 set in time for a 10-11 kit. T12 adds Medial OA, Effusion and Baker's, which N (the public OAI CoAtNet)
  does not supervise. That is where our own OAI members can add signal N lacks (research D, verify_D).
- **The 224 winner's folds come before P2b.** That is insurance: a complete OAI set early. P2b's data is not READY
  before ~10-10 morning anyway. Its queue line is deferred by the runner until READY, so it costs no idle time.
- **288 OAI (P2c), yes, with the V00 winner's rows (~42 h for 5 folds).**
  - 288 is our best resolution (f0 OOF .8893 vs .8838 at 224, MEASURED).
  - The f0 is itself a paired probe vs `f0_s1045_ttr288frac75` (same seed and schedule, only labels/store/folds
    differ), so nothing is wasted if it passes.
  - **All-visits at 288 is not planned:** ~84 h, and the per-row cost is 3.5x for 23 % more distinct knees.
- **All-visits rows outnumber RSNA 2.7 : 1** (9,130 OAI vs 3,466 RSNA per fold). That is the main risk of P2b: domain
  dominance and a PF OA calibration pull.
  - A cheaper middle option is pre-registered as P2u, T12uniq: one row per physical knee, +23 % rows, ~6 h probe. Use
    it if P2b cannot be built by 10-11 12:00.
  - A per-epoch subsample of the OAI rows (research D `--ext_frac`) would make all-visits cost-neutral. It needs the
    opt-in trainer patch: proposed only, not applied.

**Calendar, PASS branch (MODEL; launches stop at the 10-20 12:00 freeze, ~263 V100 h from 13:00 today):**

| step | job | hours | ends ~ |
|---|---|---|---|
| 1 | P2a T12v00 | 5.4 | 10-09 18:30 |
| 2 | 224 W f1-f4 (W = T1 or T12v00) | 24.5 | 10-10 19:30 |
| 3 | P2b T12allv (if READY) | 11.0 | 10-11 06:30 |
| 4 | 288 W f0 + gate | 8.4 | 10-11 15:00 |
| 5 | 288 W f1-f4 (on PASS) | 33.6 | 10-13 01:00 |
| 6 | if P2b PASS + selected: 224 allv f1-f4 (manual insert: `qlines2/folds_T12allv.txt`) | 50.6 | 10-15 04:00 |
| 7 | slack to the freeze | ~130 | e.g. a 2nd seed of the best OAI set |

**Kits:**
- OAI members go in the N pick only, never the clean pick.
- K9N runtime margin is only 0.65-1.2 h: an OAI set must REPLACE D (224) or E (288), not be added.
- K-builder: the new `train_labels` tags `labels_frac75oai_T1/T12v00/...` need the picker change that set D needed (research D 4.3).

## 5. FAIL branch
- Per PROBE_PREREG, a T1 FAIL closes L6, and PROBE2 must not run on top of it.
  - If the fail is only the gold58 leg (low power, SE ~.01-.02) while the OOF legs are positive, that is weak evidence.
  - Re-opening is still Eesh's explicit call, written in PROBE2_PREREG sec 9 BEFORE any launch. It would be a new
    hypothesis (and P2b carries a negative public prior).
- V100: insert `qlines2/fail_branch_288frac75_s1046.txt` (5 lines, ok / 0 BAD; Q4 argv with seed 1046), or the L5
  folds if L5 passes on Vast. Otherwise the queue idles after the probe.
- OAI downloads:
  - Let V01 finish (it is running and cheap).
  - Do not spend Eesh's time on V03-V06 until a decision.
  - The data can still serve as an external validation set for N-pick choices (verify_D lever 3; inference only).

## 6. Exact commands
```
# T1 gate (if the watcher has not already written ~/oai_build/out/gate_f0_s1043_ttr_frac75oai.json)
CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/rsna/bin/python ~/oai_build/score_oai_probe.py

# PASS branch: one block at the top (reverse order keeps line 1 first). Do NOT also insert qlines_oai_folds.txt.
for l in $(seq 20 -1 1); do bash ~/queue/qctl.sh top "$(sed -n ${l}p ~/oai_build/qlines2/chain_T1pass.txt)"; done

# FAIL branch (clean default)
for l in 5 4 3 2 1; do bash ~/queue/qctl.sh top "$(sed -n ${l}p ~/oai_build/qlines2/fail_branch_288frac75_s1046.txt)"; done

# Eesh, after V01 finishes (one password for three visits; resumable; only unconverted knees)
bash ~/oai_work/get_oai_more.sh v03,v06,v05 24      # or: WT=24 bash ~/oai_work/get_oai.sh v03rest (one visit per run)

# monitoring
tail -3 ~/oai_build/build_fu.log ~/oai_build/watch_unions.log; cat ~/oai_build/out/select_*.verdict ~/oai_build/out/p2c_*.verdict
```

## 7. Risks and caveats
- **P2b prior is negative** (nartaa's 12-month arm). P2b is a test, not an expected win.
- **gold58 legs are low-power.** The OOF legs and oof12 carry the selection.
- **The `via` gates run once per runner instance.** If `select_gate.sh` crashes, the gated lines hold 120 min and then
  defer. Look at `~/oai_build/out/select_gate.log`.
- **The runner does not survive a reboot.** There was a power cut 10-08. After a reboot, restart the converter
  (same command, it resumes) and the watcher.
- **Compliance:**
  - OAI npz and raw stay on the box: ~/oai_store and ~/oai_build are mode 700, and the union dirs hold symlinks only.
  - Asserted every time: no OAI file in ~/fleet_store; decode-probe ids RSNA only.
  - Eesh: delete the NDA keyring entry after the last download.
