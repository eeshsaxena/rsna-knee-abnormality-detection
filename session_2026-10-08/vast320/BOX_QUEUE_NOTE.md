# Box queue coordination for the Vast folds

> **2026-10-09 UPDATE (288 pivot):** the placeholders are now EMPTY `~/fleet_runs/f1_s1045_ttr288frac75/` through
> `f4_s1045_ttr288frac75/`. The box runs `Q4 TTR288 s1045 f0 frac75` (launched 01:15, ends ~07:05). Then the runner
> scans as follows:
> - f0 mix50: SKIP (frac75 PASS)
> - f1 frac75: REFUSED (placeholder)
> - f1 mix50: SKIP
> - and so on for f2-f4
>
> So the box goes IDLE after ~07:05 unless the main session queues new work. A released fold is re-added as
> `Q4 TTR288 s1045 fF frac75 (back from Vast) | ~/fleet_runs/fF_s1045_ttr288frac75 | needs PASS from ~/queue/gates/frac75.verdict wait 0 | <exact argv>`.
> The 320 text below is the original note; the mechanism is identical.

## How the runner treats an existing dir

`~/queue/queue_runner.sh` takes the FIRST runnable job line. Before it looks at the gate, it checks whether
`out_dir` or `out_dir.log` exists. If either exists, it comments the line out as `#REFUSED <time> (out_dir or its
.log exists)` and never trains it. So one EMPTY directory per Vast fold is enough to reserve that fold.

## Which EMPTY dirs to create (Plan A: Vast = f2, f3, f4; box = f1)

```
~/fleet_runs/f2_s1045_ttr320frac75/
~/fleet_runs/f3_s1045_ttr320frac75/
~/fleet_runs/f4_s1045_ttr320frac75/
```

- Do NOT create f1 (the box trains it right after the probe) and do NOT create any `.log` file.
- Plan B (Vast = f3, f4): only f3 and f4.
- `vast_launch.sh` creates them automatically once the instance is running and reachable
  (`bash box_placeholders.sh create 2 3 4`). It refuses a fold the box is already running (`#RUN` line or a live
  trainer) or whose dir/log already holds anything. Manual equivalent:
  `ssh training@192.168.10.30 'mkdir ~/fleet_runs/f2_s1045_ttr320frac75 ~/fleet_runs/f3_s1045_ttr320frac75 ~/fleet_runs/f4_s1045_ttr320frac75'`
- Timing: the probe ends ~01:30 IST and the runner starts f1 about 4 min later (two free checks 120 s apart). The
  f2-f4 lines are only scanned when f1 ends (~08:50 IST), so the placeholders can be made at any point between the
  gate PASS and ~08:50. To hand f1 to Vast as well, its placeholder would have to exist before ~01:29. That is not
  the plan.
- At ~08:50 the runner marks f2-f4 `#REFUSED` and moves on to the next active lines, which are the Q4 TTR288 s1045
  lines (about 5.8 h each). The G320' note in queue.txt says to remove the Q4 lines on a PASS. That is the main
  session's decision and is unchanged by this kit.

## What fills them

`vast_fetch.sh` (called by `vast_watch.sh`) has the box pull from the instance into those exact paths, giving the
same layout a box run produces:

- `~/fleet_runs/fF_s1045_ttr320frac75/` holds the raw and ema `*_best.pt`, `oof_fF_s1045_{raw,ema}.csv` and
  `epochs/oof_*.csv`. The `epochs/*.pt` files are left out unless `FETCH_EPOCH_PT=1` is set.
- `~/fleet_runs/fF_s1045_ttr320frac75.log` is the trainer log, ending with the `{"raw": .., "ema": ..}` line. It is
  written last.
- `~/fleet_runs/vast320_fF_s1045_ttr320frac75_receipt.json` records md5 status, ckpt args, closing line and the
  instance/GPU.

## If Vast fails, or a fold is skipped or only partly trained

`vast_destroy.sh` does this automatically for every fold not fetched complete. Manual form:
`bash box_placeholders.sh release F..`. For each fold, highest number first:

1. Refuse if a box trainer is running it. Keep it untouched if its `.log` is a COMPLETE run.
2. `rmdir` the empty placeholder. Partial Vast content is moved aside to `fF_s1045_ttr320frac75_vastpartial_<time>`
   (+ `.log`), never deleted.
3. If the queue has no active line for that out_dir (the old line is `#REFUSED`, and the runner never re-reads
   commented lines), re-add the exact line at the top:
   `bash ~/queue/qctl.sh top 'Q3b TTR320 fF (s1045 frac75, 21 ep, gc0, back from Vast) | ~/fleet_runs/fF_s1045_ttr320frac75 | needs PASS from ~/queue/gates/g320.verdict wait 60 | <argv of argv_template.txt with @R@ -> ~ and @F@ -> F>'`
   This goes through qctl's validation and lock. The fold then runs next on the box.

If the release happens before ~08:50 (the lines are still active), step 2 alone is enough; the script detects this.
Status at any time: `bash box_placeholders.sh status 1 2 3 4`.
