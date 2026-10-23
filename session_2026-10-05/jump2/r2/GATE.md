# Round-2 own-OOF pseudo-label gate (PRE-REGISTERED 2026-10-05 ~09:30 IST, before any round-2 training)

Tags: **MEASURED** = computed today from on-disk files (script named). **MODEL** = arithmetic on measured numbers.
**GUESS** = judgement. Box paths are on `training@192.168.10.30`; copies of every script are in
`SP/jump2/r2/` (`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`).

## 1. What runs, and when

- `~/auto_round2.sh` (NOT started). Start: `setsid nohup bash ~/auto_round2.sh > /dev/null 2>&1 < /dev/null &`.
  Status without side effects: `bash ~/auto_round2.sh --check`. Pause before launch: `touch ~/jump2/r2/PAUSE_R2`.
- It waits for: closing JSON lines in all 5 `~/fleet_runs/f{0..4}_s1043_ttr_mix50.log`, `auto_folds.sh` gone, no
  `PAUSE_CHAIN`/`PAUSE_R2`, no python process on the GPU (two checks 2 min apart). Gives up without launching after
  `2026-10-07 12:00` (env `R2_DEADLINE`).
- Then: `build_r2_targets.py` (FULL mode, aborts if any of the 4,333 training rows lacks OOF) writes
  `~/jump2/r2/labels_r2_a50.csv`; then, under `flock ~/jump2/gpu.lock` (lock held for the whole run, so the run is
  ALONE against lock-respecting jobs; `R2_HOLD_LOCK=0` releases it after the launch check), round-2 TTR fold 0:
  J1 recipe (`--model ttr --aug heavy --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs 24
  --save_epochs 6-24 --workers 10`), `--seed 1543`, `--labels labels_r2_a50.csv`, `--eval_labels` steven_v4,
  `--exclude/--gold` gold74, out `~/fleet_runs/f0_s1543_ttr_r2a50`, through `r2_capped.py` (CUDA memory fraction .75).
  (Verify 10-05: the original seed 2043 is refused by fleet_train.py BJ-5, `--model ttr` takes seeds 1000-1999 only;
  MEASURED AssertionError through r2_capped.py. Changed to 1543; the gate bars are unchanged.)
  Code: `~/bigjump_build/code/fleet_train.py`, unless `~/fast_build/SWITCH.md` holds a line `STATUS: VERIFIED` (and no
  "NOT VERIFIED"/"UNVERIFIED") AND `code2/fleet_model*.py` are byte-identical to bigjump's AND code2 has `--grad_ckpt`;
  then `~/fast_build/code2/fleet_train.py --grad_ckpt 0`. At 09:25 SWITCH.md does not exist: bigjump path.
- After the run: `r2_gate.py` evaluates this file's criteria, writes `~/jump2/r2/GATE_RESULT_a50.{txt,json}` and the
  log `~/fleet_runs/auto_round2.log`. Nothing else is launched; the operator decides all-data seeds.
- ETA (MODEL from the chain: fold ~4.6-5 h alone): folds done ~03:45 10-06, launch ~03:50, gate verdict ~08:45-09:15 10-06.

Target formula (per cell, probability space): `t = a * steven_v4 + (1 - a) * oof_mapped`, a = .50.
`oof_mapped` = best-tag OOF of the 5 TTR mix50 folds, z-scored per fold and per label on the logit scale, mapped onto
steven_v4's logit mean/std (clip .005) over the covered rows. Gold74 rows keep steven_v4 (never trained on).
OOF guarantee enforced in code: each OOF row must sit in that run's own validation fold of `folds5.csv`, never in
gold74; runs without a closing line are skipped (fold 1 has a mid-training OOF csv right now and is correctly skipped).

## 2. Reference J1 (MEASURED, `r2_gate.py` self-check reproduces all four)

J1 = `~/fleet_runs/f0_s1043_ttr_mix50`, same fold, same recipe, mix50 labels, seed 1043.
Best tag raw (closing `{"raw": .88360, "ema": .88240}`), first best epoch 16 (0-based).

| quantity | J1 | bar for round 2 |
|---|---|---|
| (a) val AUC vs steven_v4 (best tag, closing JSON) | .8836 | **>= .8806** (J1 - .003) |
| (b) gold58 at the best tag's best epoch (run log) | .9132 | **>= .9082** (J1 - .005) |
| (c) parent-likeness: mean per-label Spearman of the best-tag OOF vs the Raptor proxy, 867 fold-0 val rows | .8962 | **<= .8762** (J1 - .020) |

Raptor proxy = `~/jump2/r2/raptor_proxy_v5v8.csv` (copied from `SP/teacher_v2_out/`, sha256 d45683f4..., byte-identical
to `~/fleet/teacher_v2_raptorproxy_shard0.csv` used in blendmath/oofx, so .8962 = oofx's .896).
Rules: best tag = larger closing value; best epoch = first epoch reaching it (fleet_train selects with `cv > best`).
**Verdict PASS = (a) and (b) and (c).** Anything else is FAIL. No re-runs with other seeds to rescue a FAIL.

## 3. (d) Pre-registered INFO criterion: beat what we already own (MEASURED frontier)

(a) and (c) alone can be met by members we already have: the rank blend of J1 (raw) and TTR v4 (`f0_s1042_ttr` ema) on
the same 867 rows (MEASURED today, box CPU):

| w on TTR v4 | 0 | .2 | .4 | .5 | .6 | .7 | .8 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| AUC vs v4 | .8836 | .8848 | .8846 | .8840 | .8831 | .8818 | .8803 | .8764 |
| PL vs proxy | .8962 | .8958 | .8908 | .8866 | .8812 | .8747 | .8672 | .8496 |
| gold58 (gold74_preds) | .9132 | - | - | .9005 | .8952 | .8908 | .8858 | .8780 |

At w .7 the blend passes (a) .8818 and (c) .8747 but fails (b) .8908. So (b) is what makes the gate test something
new. `r2_gate.py` also prints (d): the candidate's AUC and gold58 minus the best blend with PL <= the candidate's PL.
(d) "beats the frontier" = AUC >= frontier + .001 and gold58 >= frontier. (d) does not change the verdict; it decides
how to read a PASS (sec 6).

## 4. Dry-run evidence on the 867 fold-0 rows (MEASURED, `build_r2_targets.py --partial 1`, `~/jump2/r2/dry/`)

Target-level parent-likeness (mean per-label Spearman vs the proxy; same 867 rows for every row of the table):

| target | PL | Spearman with steven_v4 | with mix50 |
|---|---|---|---|
| steven_v4 (base) | .6452 | 1 | - |
| mix50_logit (J1's labels) | .8643 | - | 1 |
| TTR mix50 OOF (J1, the teacher) | .8962 | - | - |
| **r2 a50 (the gate's labels)** | **.8598** | .845 | .949 |
| r2 a60 | .8472 | .866 | .954 |
| r2 a50s75 (silent cells OOF .75) | .8693 | .826 | .941 |
| r2 a70 / a80 / a90 | .8334 / .8118 / .7904 | .888 / .914 / .936 | .956 / .952 / .944 |
| v4-teacher (TTR v4 OOF) a50 / a60 / a70 | .8370 / .8256 / .8133 | .838 / .861 / .883 | .930 / .937 / .940 |

Read: **a50 from the mix50 OOF is as parent-like as mix50 itself at the target level (.860 vs .864) and correlates .949
with mix50.** The OOF teacher is a mix50 student, so Dread re-enters through it. Silence-aware weighting makes it
MORE parent-like (silent cells are where the OOF dominates). A TTR v4 teacher is only ~.02 better at equal a.

MODEL prediction for the a50 student (2 anchors: steven_v4 target .645 -> TTR v4 student .850; mix50 target .864 ->
J1 .896; plus the frontier of sec 3 as a "student = blend of what it was taught" model): PL ~.885-.895, AUC ~.882-.885,
gold58 ~.90-.91. So **(c) is likely to FAIL** (GUESS P(pass c) ~.2) unless self-distillation changes the student more
than the target. (a) likely passes; (b) is a coin flip (gold58 macro on 58 studies: paired noise sd ~.004-.006, GUESS).

## 5. Caveats

- Seed 1543 vs J1's 1043: seed noise ~.001-.003 val AUC (GUESS; forum CV bar .003). The (a) bar is inside that band.
- Weak second-order leak: the fold-1..4 teachers trained on fold-0 studies (mix50 labels), and their predictions are
  the round-2 targets of the fold-0 student's training rows. No fold-0 study or label is used directly; (a) may be
  optimistic by a little (GUESS <= .002). gold58/gold74 are clean (held out of every teacher and student) and so is the LB.
- gold58 favours parent-like students (mix50 = .926 on gold58 partly via Dread's gold-picked checkpoint), so (b) and (c)
  pull against each other by construction. That is intended: a PASS needs strength that is not the parent's.
- If the fast path is used, the run differs from J1 in code path too; only valid if SWITCH.md certifies parity.

## 6. Decisions and the LB slot plan (pre-registered)

| gate outcome | meaning | action |
|---|---|---|
| PASS and (d) beats frontier | a stronger, less parent-like member than anything we own | LB: 2 slots on the day of the verdict (10-06 from ~09:30): **(1) SOLO of r2 f0**, **(2) parent + r2 f0 at W .50** (built with the K3 kit copy: `build_k3.py` TTR member via `model_mod fleet_model_ttr`, `--members` = the r2 checkpoint only, MAIN_MODE solo / blend `--w_main` .50; operator commits and submits). Read: SOLO >= .930 AND W .50 >= .943 -> r2 member r_LB <= ~.90 at solo .93 (MODEL, blendmath sec 7 inversion) -> operator starts all-data r2 seeds (`--fold -1`, 2 seeds x ~6 h, TTR) for the 10-12 fleet; W .50 <= .941 -> parent copy after all, stop. |
| PASS, (d) does not beat frontier | same point as a J1 + TTR v4 blend we already have | no training follow-up; at most 1 LB slot (W .50) if a slot is spare after K4; otherwise none |
| FAIL on (c) only | stronger student but still a mix50-family member | no decorrelation value. LB only if val AUC >= .8866 (J1 + .003): then it is a candidate REPLACEMENT for mix50 members in the A1 fleet, tested as SOLO; else nothing |
| FAIL on (a) or (b) | the round-2 lever does not transfer with this teacher | stop this teacher. Next teacher = 5-fold TTR v4 OOF (blendmath sec 8): `build_r2_targets.py --sets '~/fleet_runs/f?_s1042_ttr*'` once folds 1-4 exist, same gate with J1 and TTR v4 as references |

K4 (A1 5-fold TTR mix50: W .20 and SOLO) keeps priority on 10-06 slots; the r2 slots only take what is left.
No Kaggle action is taken by this pipeline; dataset versions, commits and submissions are the operator's.

## 7. Files

| file | role |
|---|---|
| `~/jump2/r2/build_r2_targets.py` | target builder (CPU); `--partial 1` = dry run writing `DRY_*` files; manifest JSON with input shas |
| `~/jump2/r2/r2_gate.py` | evaluates (a)-(c) + (d) INFO; exit 0 = PASS |
| `~/jump2/r2/r2_capped.py` | runs fleet_train.py unchanged under a CUDA memory fraction (runpy) |
| `~/auto_round2.sh` | waiter + label build + gate run + gate evaluation; `--check` prints status only |
| `~/jump2/r2/test_r2_pipeline.sh` | CPU end-to-end test with fake fold runs and a fake trainer (passed 09:26) |
| `~/jump2/r2/dry/` | dry-run targets and manifests (DRY_ prefix, never used for training) |

## 8. Adversarial verification notes (10-05 ~09:45 IST; bars and verdict rules above are unchanged)

- FIXED (blocker): seed 2043 -> 1543 (see sec 1). `auto_round2.sh` now refuses at startup any R2_SEED outside
  1000-1999 or one that reuses an existing `f0_s<seed>_*` member key, and any R2_SETS pattern with `*`/`[` or without
  `f?_` (before, `R2_SETS='~/fleet_runs/f?_s1042_ttr*'` made the bash closed-check wait silently until the deadline).
  The chain check also matches `~/fast_build/auto_folds_fast.sh`. The fake trainer in `test_r2_pipeline.sh` now carries
  the BJ-5 seed guard, so the CPU test would have caught this.
- Leak caution on sec 6 row "FAIL on (c) only": its promotion rule uses val AUC (>= .8866), the one criterion the
  stacking leak inflates (pseudolabel.md sec 4.2: "never gate on it"). Read gold58 (clean) next to it before acting.
- Overlap: `pseudolabel.md` sec 4 pre-registers a near-twin probe (`labels_pl_tm5_v1` = OOF weight .75 silent /
  .50 spoken, raw tag, seed 1044, Gate S). `R2_VARIANT=a50s75` is the closest r2 variant. Only one of the two should
  get the post-chain box slot; pseudolabel.md ranks both as optional filler that must not displace a labels-track
  student. With the default R2_HOLD_LOCK=1 the run holds `~/jump2/gpu.lock` for ~5 h.
