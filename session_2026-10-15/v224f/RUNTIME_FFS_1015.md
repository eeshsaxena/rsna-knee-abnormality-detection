# V224-FF-S hidden-rerun runtime review (written 10-15 ~00:10 IST by a Claude subagent). No em dashes.

Scope: `build_v224fs.py --wf 0.20 --base ff --s-manifest <upload>/r7tb_manifest.json` (stand-in build `tests/standin_v224ffs20`,
nb 037f67a7, inspected). Read-only review: nothing submitted, no dataset, no push, box untouched, no frozen file edited.
Tags: MEASURED / MODEL / GUESS. Model script `sim_ffs_runtime.py` -> `sim_ffs_runtime.txt` (this dir).

## Verdict
- **Fits 9 h: YES.** Hidden kernel wall (MODEL from MEASURED latencies + per-member microbench): **central ~6.0 h** (median of a
  bimodal spread: ~5.3-5.6 h when the flip switches off early, ~6.2-6.7 h when it stays on), p99 ~7.4-7.7 h, worst clean finish
  ~8.0-8.2 h (x1.3-1.45 session), **hard worst ~8.4 h** by construction (fleet wall 8.3 h + 300 s kill + HF cell), reached only via
  the Synovitis canary. P(any time-caused canary) ~0.1-0.2 % (MODEL). P(kernel > 9 h) ~0 (every late path is guarded).
  Submit-to-score ~6.2 h central (MODEL; queue small).
- **But the as-built budgets confound the read (main finding).** The FX8 flip budget 4.4 h sits BELOW the R7-with-flip end of a
  session like the one V224-FLIP got on 10-14 (~4.2-4.4 h kernel age, MODEL from the MEASURED 4.57-4.65 h submit-to-score). The
  driver's projection (t_k + 1.05 x rate x files left) then crosses 4.4 h at study ~10 and switches both R7 workers to plain.
  MC: FF-S keeps flip on >= 90 % of studies in only **~28 %** of sessions and is mostly plain (<= 10 %) in **~59 %**; FF20
  (budget 4.9) keeps it in ~73 %. A flip loss is silent (no canary, logs not returned), so an FF-S read vs FF20 / FLIP would mostly
  measure the flip lottery (flip was the +214 frozen-team jump) rather than S (+.0001-.0003 expected).
- **FX8's premise is too pessimistic.** S = 5 TTR224 members, MEASURED 0.287-0.325 s per study per T4 each (both TTA passes, under
  the R7 wheels) = **+1.44-1.62 s/study/T4, +0.26-0.29 h central, +0.34 h slow, +0.44 h slow session** (FX8 assumed the 20-member
  average 0.49 s/member: +0.44 / +0.58 h).
- **Recommendation (main decides; needs a builder + Amendment 4 text change, not done here):** keep FF20's flip budget **4.9 h**
  (identical flip policy to the base, so S is the one change) and set the fleet start budget to **5.1 h** (MIN_START_MIN 192).
  MC: flip coverage = FF20's (>= 90 % in ~74 %), P(time canary) ~0.2 %, median 6.6 h, p99 7.7 h; slow-session corner (start at the
  guard, fleet x1.15 x 1.3) ends 8.22 h, same ~0.1 h margin class as FF20 (8.19 h) and as-built (8.12 h). 5.2 h also works
  (MC 0 skips) but its corner is 8.32 h (hard stop). If the build stays as is, pre-register that FF-S reads "V224-F + S with a
  partly-on flip", not "FF + S".

## 1. How S is added (build_v224fs.py --base ff)
**Merged into the fleet stage, not a new stage and not a replacement of F.**
- Base = commit-tested V224-FF20 (sha b86f8b9f, asserted byte-identical), 25 cells. Patched in place (revert proof = base bytes):
  cell 7 FLIP-TA `FLIP_TA_BUDGET_H` 4.9 -> **4.4**; cell 16 SAVE knobs start_budget_h 5.5 -> **5.0**, min_start_min 168 -> **198**;
  K2_CFG `MIN_START_MIN` 168 -> **198** (WALL_H 8.3 unchanged).
- New cell 20 **S-MERGE** (before CLOCK and the fleet): symlinks the 5 S checkpoints of `eeshsaxena/rsna-knee-cnxt-fleet-r7tb` into
  /tmp/k12_fleet next to the K14 files, full sha256 of every S file, writes the merged manifest. Cost: seconds (sha of 5 files).
- Cell 22 **FLEET** = the V224-F fleet cell, only its K2_CFG line differs: `MEMBERS` = K14 R 20 + S 5 (all TTA AB). One worker per
  T4, every member on every study, one start guard, one wall, all-or-nothing (P2: n_keep == 25 on both workers, no hard stop),
  worker time valve disabled (P6).
- New cell 24 **HF-S**: recomputes the fleet's equal 25-mean from the worker npz (must equal submission_fleetsolo.csv, 1e-6), then
  F = .5 mean(K14) + .5 mean(S) (two-group, PREREG_TEACHER sec 7), blends (1 - .2) pct(V224 output) + .2 pct(F).
  (`--replace` = F from S alone, MEMBERS = S only; not this entry.)

## 2. Time budget logic and canaries (in run order; kernel age from notebook start)
| stage | guard | what happens if late / broken | visible? |
|---|---|---|---|
| R7 (time-aware flip driver compact_infer_ta.py) | per worker, after every study k: k >= 10 -> switch if t_k + 1.05 x rate x files_left > NB_T0 + **4.4 h** and that work > 600 s; k < 10 -> switch if t_k > budget; budget passed at start -> plain only | flip child killed, pinned compact_infer.py WITHOUT flip scores the rest (one ~250 s restart); child crash -> plain; driver setup error -> FLIP_TA_DRIVER_ERROR -> plain V224-FLIP worker = ALWAYS-ON flip (no time awareness) | log only (FLIP_TTA_DISABLED_AT_STUDY, flip_ta_receipt.json); NOT on the LB |
| R7 (her code) | deadline NB_T0 + 8.3 h | unscored studies -> her per-study Raptor fallback (BLEND_FALLBACK) | log only (as in V224) |
| r224 | skip if age > 7.83 h; timeout min(left, 2.5 h) | V224 canary: Fracture = 0.5 (kept through the blend) | LB (display drop) |
| SAVE | logs fleet_may_start (age <= 5.0 h) | none (log) | log |
| S-MERGE | any error (dataset missing, sha, extra file, collision) | /tmp/k12_fleet removed -> fleet not clean -> Synovitis canary | LB |
| FLEET start | starts only if >= 198 min remain to the 8.3 h wall (age <= **5.0 h**) | "stage not started" -> HF_FALLBACK, Synovitis = 0.5 | LB |
| FLEET wall | no new study after 8.3 h, workers killed 300 s later | hard_stopped -> not clean -> HF_FALLBACK, Synovitis = 0.5 | LB |
| HF-S | 25 members, names == K14 + S, npz ids / shape, eq-mean == fleetsolo | any failure -> HF_FALLBACK, Synovitis = 0.5 | LB |

**If S overruns:** S has no guard of its own (it is inside the fleet stage). A slow S makes the whole fleet stage later; if that
stage misses the 8.3 h wall, both workers are killed by ~8.38 h, the fleet is not clean, and HF-S writes the V224(+flip) output with
**Synovitis = 0.5 for every row** (`HF_FALLBACK`, `CANARY_FIRED fleet not used`, hf_receipt mode H_only, v224f_receipt fired
['fleet->Synovitis']). The notebook still completes (~8.4 h) and scores, display ~.02-.04 lower = visibly not a valid pick. The K14 part
of F is lost together with S (no K14-only fallback, by P2 design). Same canary if the pre-fleet stages run past 5.0 h.

## 3. MEASURED inputs
### 3a. Hidden submit-to-COMPLETE of our scored entries (incl. queue)
Submit times from the Kaggle API (read-only `competition_submissions`, UTC -> IST); completion windows from our pollers.
| entry (ref) | stages | submit IST | COMPLETE window IST | h | source |
|---|---|---|---|---|---|
| HV (57093094) | Raptor + S6 | 10-12 08:23:19 | 11:34-11:44 | 3.18-3.35 | hv/subpoll log |
| HF30 (57097814) | HV + D fleet 5 TTR224 | 11:45:57 | 15:46-15:56 | 4.01-4.18 | hf/subpoll log |
| HF20 (57097813) | same kernel, w .20 | 11:45:53 | 16:06-16:16 | 4.35-4.52 | hf/subpoll log |
| HF-R20 (57106365) | HV + K14 R 20 | 17:25:49 | pending 21:36, done by 22:53 | 4.17-5.45 | PROGRESS_12 |
| HR7 (57137313) | R7 + r384 | 10-13 14:19:31 | 17:10-17:35 | 2.84-3.26 | PROGRESS_13 |
| HR7-RR (57140358) | + r224 | 16:17:26 | by 19:42 | <= 3.42 | PROGRESS_13 |
| HFR7-RS (57154866) | HFR7-RR + 2 Raptor arms | 10-14 01:23:38 | 07:07-07:28 | 5.73-6.08 | PROGRESS_14 |
| HFR7-RS-F30 (57162815) | same, w .30 | 07:07:38 | ~12:44-13:03 | 5.6-5.9 | api_chain.log |
| HFR7-RR-F40 (57164843) | R7 + r384 + r224 + fleet 20 | 08:35:07 | ~13:26-13:30 | 4.85-4.92 | api_chain.log |
| V224-exact (57171955) | r384 + R7 + r224 | 12:38:04 | by 15:55 | <= 3.28 | LB snapshot 15:54 |
| V224-FLIP (57174021) | + R7 flip forced | 13:38:06 | ~18:12-18:17 | **4.57-4.65** | api_chain.log (2 pollers) |

Derived: flip adds >= 1.29 h on V224 (4.57 - 3.28); fleet 20 adds >= 1.43 h on the R7 base (4.85 - 3.42; wall_model 1.66-1.80 h);
two identical kernels submitted 4 s apart (HF20 / HF30) finished 0.17-0.51 h apart (session spread ~4-12 %); the public Raptor
author's efficiency entry turned around in 7.2 min incl. queue (CLAIMED), so queue is usually small. Kernel-version run times of
the hidden reruns are not exposed by the API (only commit runtimes, already in the RESULT files).

### 3b. Fleet per-member cost (commit microbench, s per study per T4, both TTA passes; fleet_receipt.json `bench`)
| commit | K14 20 total (slower GPU) | one TTR224 member | one TTR288 member | decode s/study/thread |
|---|---|---|---|---|
| V224-F20 (R7 wheels) | 9.86 | 0.313-0.325 | 0.534-0.553 | 1.40 |
| V224-FF20 (R7 wheels) | 8.83 | 0.287-0.291 | 0.479-0.493 | 1.52 |
| V224-FF ctl (R7 wheels) | 10.18 | 0.319-0.321 | 0.497-0.572 | 1.51 |
| HF-R20 / HF20 / HFDT20 (stock) | 8.92 | 0.270-0.287 | 0.45-0.48 | 2.1-2.3 |
S (5 x TTR224, same model_mod / model_sha f141aeed / geometry ttr224-320x24 as the DT members) = **1.44-1.62 s**; the 25-member fleet is
GPU-bound (11.5 s GPU vs <= 2.3 s decode per thread), so S's hidden cost is its GPU term. Commit fleet fixed part ~1.0 min
(stage minus loop), stage 2.37-2.50 min for 3 studies.

## 4. Runtime model (MODEL; `sim_ffs_runtime.py`)
Calibration: pre-R7 (wheels + r384) 0.42 h; R7 plain 2.25-2.55 h; R7 flip = 4.62 - queue - 0.42 - r224 (so an always-on flip
reproduces the MEASURED V224-FLIP 4.62 h); r224 + stage 3 0.15-0.30 h; queue 0-0.3 h (GUESS); 1,300 hidden studies (650 per T4);
fleet = 1.5 min fixed + 650 x (9.86 K14 + 1.62 S) s x mf (1.0-1.15) x s; R7 workers simulated with the driver's exact rule
(sim_ta.py worker). Session speed s = exp(N(0, .08)) x 1.3 w.p. .05 (GUESS, consistent with the HF20/HF30 spread and FN11 1.14).

Fleet 25 alone: central 2.10 h, slow (x1.15) 2.41 h, slow session (x1.15 x 1.3) 3.12 h. S increment +0.29 / +0.34 / +0.44 h.

Grid at the calibration q .1 h, r224 .2 h (kernel h):
| session | FF-S as built (4.4 / 5.0 / 25) | FF20 (4.9 / 5.5 / 20) | rec. FF-S (4.9 / 5.1 / 25) |
|---|---|---|---|
| s .95 | flip 67 %, fleet start 4.21, end 6.20 | flip 99 %, 4.32, 6.04 | flip 100 %, 4.31, 6.31 |
| **s 1.00 (= the measured FLIP session)** | **flip 6 %**, R7 3.03, fleet 3.24, **end 5.34** | flip 100 %, R7 4.33, 4.54, 6.36 | **flip 100 %**, R7 4.33, 4.54, **6.65** |
| s 1.30, mf 1.15 | flip 2 %, 4.03, 7.16 | flip 2 %, 4.03, 6.72 | flip 2 %, 4.03, 7.16 |
| s 1.45, mf 1.15 | flip 2 %, 4.49, 8.03 (ok) | flip 2 %, 4.49, 7.53 | flip 2 %, 4.49, 8.01 (ok) |
(other calibrations in `sim_ffs_runtime.txt`: with q .3 / r224 .15 the as-built FF-S keeps flip on ~34-47 % at s 1.0; FF20 ~100 %.)

Monte Carlo (2,000 sessions):
| config | median | p90 | p99 | max | flip >= 90 % | flip <= 10 % | P(skip) | P(hard stop) |
|---|---|---|---|---|---|---|---|---|
| FF-S as built 4.4 / 5.0 | 6.02 | 6.66 | 7.40 | 7.95 | 0.28 | 0.59 | .001 | 0 |
| FF20 4.9 / 5.5 (20 members) | 6.29 | 6.87 | 7.25 | 8.20 | 0.73 | 0.18 | 0 | 0 |
| FF-S 4.9 / 5.1 (recommended) | 6.61 | 7.19 | 7.67 | 8.39 | 0.74 | 0.16 | .001 | .001 |
| FF-S 4.9 / 5.2 | 6.61 | 7.20 | 7.55 | 8.17 | 0.73 | 0.18 | 0 | 0 |
| FF-S 4.65 / 5.2 | 6.37 | 6.95 | 7.64 | 8.39 | 0.52 | 0.34 | 0 | .002 |
| FF-S as built, driver error (always-on flip) | 6.60 | 7.22 | 7.53 | 7.69 | 1.00 | 0 | **.13** | 0 |
Sensitivity (as built): S at 2.77 s/study (TTR288-like) median 6.23, p99 7.92, P(hard stop) .004; S at 4.0 s .010; 1,500 hidden
studies P(skip + hard stop) ~.014; P(kernel > 9 h) 0 in every row.

Deterministic corners (start at the guard, r224 x1.45): fleet x1.15 x 1.3 ends as built **8.12 h**, rec. 4.9 / 5.1 8.22 h,
4.9 / 5.2 8.32 h (hard stop), FF20 8.19 h; at x1.15 x 1.45 every config hits the 8.3 h wall (Synovitis canary, kernel ~8.4 h).
These corners need a late R7 AND a very slow fleet in the same session (flip kept to its limit, then a slowdown); in a uniformly
slow session the driver drops flip at study ~10 and the fleet starts by ~4.5 h.

## 5. Risks
1. **Flip-coverage confound (MODEL, high):** as built, FF-S is mostly plain R7 in sessions as fast as V224-FLIP's 10-14 run; the
   comparison base FF20 keeps flip. Expected flip share 36 % vs 78 %. Silent on the LB. Fix: flip budget 4.9 (section Verdict).
2. **All-or-nothing F:** any S problem (overrun, S-MERGE, dataset version, HF-S check) drops the whole fleet incl. K14 and
   blanks Synovitis; no K14-only fallback. Time-caused P ~0.1-0.2 % (MODEL); dataset / attach errors are a build-time risk (check
   the commit log for `[s-merge] RESULT OK ... 30 checkpoints`).
3. **Driver-error path:** FLIP_TA_DRIVER_ERROR falls back to an always-on flip; with the 5.0 h guard the fleet is then skipped in
   ~13 % of sessions (any session >= ~1.1x slower). Tested offline 75/75 + 2 commits, so the path is rare (GUESS < 2 %).
4. **Calibration rests on one FLIP latency sample** (4.57-4.65 h, two pollers) and MODEL splits (pre-R7 0.42 h, r224 0.15-0.30 h,
   queue). If the queue was large (>= 0.3 h) the as-built flip keeps ~35-50 % at s 1.0; the conclusion (FF-S flips much less than
   FF20) holds in every calibration.
5. **FF20 (05:36) itself** drops flip in ~18 % of sessions (MODEL): its read, and FF-S's base, carry the same unobservable lottery.
6. Hidden study count (~1,300, claimed) and S cost are MODEL inputs; +15 % studies or a 2.5x S cost raise P(time canary) to ~1 %.
7. The JPEG DEC insurance cell, if added to FF-S, changes runtime only on JPEG-Lossless studies (decode; MODEL small).

## 6. Checks on the FF-S commit (before the submit)
- `fleet_receipt.json` bench: each `f*_s1047r7tb_*` member <= 0.40 s per study per T4 (DT members 0.29-0.32 on the same commit);
  else re-run `python sim_ffs_runtime.py` with S_S set to 5 x the measured value. stage_min within ~+0.6 min of the FF20 commit.
- log: `[s-merge ...] RESULT OK merged fleet dir + S: 30 checkpoints`, FLIP_TTA_FORCED budget = the built value, K2 MIN_START_MIN =
  the built value, HF_BLEND_OK with groups {'K14': 20, 'S': 5}, V224FS_F2G eq-mean <= 1e-6.
- Gate: `v224ffs_check.py` (pass `--flip-budget-h` matching the build if the budget changes).

## 7. If main adopts the recommendation (not done here; edits needed)
- `build_v224fs.py`: `assert A.start_budget_h == 5.0` and the `flip_budget_h <= start_budget_h - 0.6` assert encode FX8; a 4.9 / 5.1
  build needs both relaxed (e.g. flip 4.9 with start 5.1 leaves 0.2 h, which the MC shows is enough because a flip that runs to its
  budget ends R7 by ~4.9-5.07 h only in rare non-stationary sessions; FF20 has the same structure with 0.6 h).
- `v224ffs_check.py --flip-budget-h 4.9` plus the nested `v224fs_check.py --start-budget-h 5.1`: its line 168 hard-codes
  `MIN_START == 198.0 == B['min_start_min']` (FX8) and must accept 192 for such a build; re-run the stand-in build, mock gate
  (14/14 + 7/7 negatives) and the revert proof after the change.
- PLAN_TEACHER sec 7 Amendment 4 text: "fleet start budget 5.1 h (MIN_START_MIN 192), time-aware flip budget unchanged at 4.9 h
  (identical flip policy to V224-FF20, so S is the one change); S MEASURED +0.29 h central".
