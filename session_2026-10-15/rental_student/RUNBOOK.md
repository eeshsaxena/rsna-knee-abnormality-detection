# rental_student RUNBOOK: the T_b student on a rented GPU (RSNA-only). No em dashes.

What it trains: PREREG_TEACHER (+ Amendment A1) T_b student, seed 1047, folds 0-4, out `~/fleet_runs/f{F}_s1047_ttr_r7tb`.
argv = the frozen box sheet `~/r7t_labels/sheets/queue_r7tb.txt` (md5 6196c8e4) token for token, `~` -> `/workspace/rsna`
(`box/ops.sh argv-check`; vs the T_a sheet the box trains, only `--seed --labels --eval_labels --out` differ). fp16 AMP as on
the box (fleet_train.py has no bf16 path; changing it would change the recipe). The box trains T_a (seed 1048) itself.

## Pieces (laptop `session_2026-10-15/rental_student/`, box `~/rstud/src/`)
| file | where | role |
|---|---|---|
| `box/pack.sh REAL` | box CPU | bundle = code2 *.py + `~/fleet_store` (4,407 RSNA npz as HARD LINKS) + the T_b table from `~/r7t_labels/final/` (sha256 must match `LABELS_READY`) + folds5 + gold74 + ConvNeXt-T HF cache + kit; OAI assertion; T_b provenance (only v4 + OAI-free fleet OOF md5 3479835c + R7 OOF, else STOP); parity probe; MANIFEST.sha256; `~/rstud/bundle` |
| `launch.sh --search / --dry-run ID / ID` | laptop | 2 narrow searches (limit 25); preflight; spend projection guard; rent with the guard in onstart; box key; starts the box driver; laptop backup watcher |
| `box/driver.sh` | box (setsid) | push (8 tar streams) -> instance setup.sh (manifest, OAI assertion, pins, GPU checks, parity) -> parity compare -> EMPTY placeholders -> lanes (smoke layout) -> 10-20 min smoke + decide.py -> RUN_STARTED -> exec fetch_loop.sh. Any fatal step: logs to `~/rstud/abort_*`, `status/ABORT` (guard self-destroys), `~/rstud/ABORTED` |
| `box/fetch_loop.sh` | box | every 5 min: pull finished folds in parallel, `verify_fold.py` (sha256 of every file, closing line, ckpt args vs set D f2 recipe, receipt), place into the EMPTY placeholder, `~/fleet_runs/rstud_<N>_receipt.json`; failed / skipped folds release their placeholder (back to the box); partial pulls in the last 40 min; `FINAL_FETCH_OK` |
| `remote/guard.sh` | instance (onstart) | self-destroy on: hard deadline (cap; can only be tightened), ABORT, FINAL_FETCH_OK + 15 min, boot fuse (no RUN_STARTED 100 min after create), orphan fuse (lanes done, no box poll for 120 min) |
| `remote/rstud_lane.sh`, `start_lanes.sh`, `stop_fold.sh`, `status.sh`, `setup.sh` | instance | lanes per GPU (or "any" GPU), receipts + sha256 lists, progress |
| `watch.sh`, `destroy.sh --yes`, `status.sh` | laptop | backup destroy + spend, kill switch, one-screen status |

## Launch (after the R7 agent writes `~/r7t_labels/final/LABELS_READY`)
1. Box: `bash ~/rstud/src/box/pack.sh REAL` -> must end `BUNDLE READY (REAL)` (STOP = do not rent; the reason is printed).
2. Laptop (lab network; Vast 2FA live: `vastai show user` shows the credit; else Eesh runs `vastai tfa send-email`, then
   `vastai tfa login -t email -s <secret> -c <code>`):
   `bash launch.sh --search`, pick the top `fits=yes` row (DC first), `bash launch.sh --dry-run <id>`, then `bash launch.sh <id>`.
   Launch refuses: a DRY / stale bundle, a LABELS_READY or table that changed after packing, any existing instance on the
   account, a worst-case projection above the cap.
3. Monitor: `bash status.sh` (or box `tail ~/rstud/driver.log ~/rstud/fetch/fetch.log`).
4. End state: 5 folds in `~/fleet_runs/f{F}_s1047_ttr_r7tb` (+ `.log`, + `~/fleet_runs/rstud_<N>_receipt.json`),
   `~/rstud/FINAL_FETCH_OK`; the guard self-destroys 15 min later; `watch.sh` destroys too if it is alive. Then the student
   gate (`student_gate.py --student 'f{F}_s1047_ttr_r7tb' --infer`, main / R7 agent).
Kill switch at any time: `bash destroy.sh --yes` (writes ABORT, destroys, records the spend).

## Money
- Cap $15 TOTAL for this kit incl. failed attempts and data-in = credit drop since `ledger.env CREDIT_START` (set at the
  first rental). Budget per rental = min(cap - spent - $0.5 reserve, credit - $1.5). Guard hard deadline = min(budget / $/h, 7 h).
- After the smoke the driver re-projects with the MEASURED speed and aborts if the projected total exceeds the cap or the plan
  cannot end 10 min before the guard deadline; it then tightens the guard to now + 1.35 x remaining + 1 h.
