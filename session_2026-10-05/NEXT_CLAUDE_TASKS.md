## >>> NEXT CLAUDE: 4 TASKS TO DO (handed over 2026-10-05 ~10:40 IST; Eesh switched sessions) <<<

**Conventions**
- `SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`. If Temp was wiped, the copies are in `Downloads/rsna-knee/session_2026-10-05/` (same subpaths).
- Box: `ssh -o BatchMode=yes training@192.168.10.30 '...' < /dev/null`.
- Kaggle CLI: `K=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/Scripts/kaggle.exe`. Always run it with `PYTHONIOENCODING=utf-8` and filter out the `warn` lines.

**Approvals and rules**
- Eesh APPROVED Kaggle GPU for training (10-05), and the commit -> gate -> submit flow for gated kernels.
- NEVER `pkill -f`. Kill only by PID, after checking the cmdline.
- Never put a GPU co-tenant next to a training fold.

**Two background builds were RUNNING when this session ended, and probably stopped with it:**

| Build | State |
|---|---|
| `kaggle-train-folds` | Build DONE and dry-tested 50/50 (`SP/kaggle_train/RUNBOOK.md`). Its skeptic review did not finish. |
| `k5-tta-build` | Worker TTA pass B into `SP/k5`. Status unknown, probably partial. |

Check `SP/k5/` for a `K5_RUNBOOK.md`. If it is missing or incomplete, rebuild B6 from `SP/k4` (spec: `SP/jump2/JUMP_PLAN_10-05.md` sec 4, "B6 gates").

### TASK 1: launch Kaggle training of TTR mix50 folds 3+4 (do it NOW: ~11 h run, and later = the 5-fold set lands later)

1. **Read and self-review.** Read `SP/kaggle_train/RUNBOOK.md`, then check the notebook `SP/kaggle_train/kernel/rsna-train-ttr-f34/rsna_train_ttr_f34.ipynb` and its `kernel-metadata.json`:
   - GPU T4 x2, internet ON;
   - kernel_sources = `eeshsaxena/rsna-fleet-prep-s0` and `eeshsaxena/rsna-fleet-prep-s1`;
   - dataset = `eeshsaxena/rsna-knee-fleet-train`;
   - hash asserts: pixel_sha 6f49c74fc48f3345, tensor_sha 08ec55bc155be309, select_sha 91bb8a65594a25bd, jpeg_fp 34819a6207119a3b;
   - recipe = the box chain: `--model ttr --aug heavy --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995 --seed 1043`, mix50 labels, steven_v4 eval labels, gold74 excluded;
   - self-calibrated epochs <= 24, and an 11.4 h watchdog.
2. **Create the dataset.**
   - The Windows CLI has a staging bug, so first run `mkdir -p "$TEMP/.kaggle/uploads/C_/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad/kaggle_train/ds/rsna-knee-fleet-train"`.
   - Then `$K datasets create -p "$SP/kaggle_train/ds/rsna-knee-fleet-train"` (~115 MB, including the ConvNeXt weights).
   - Poll `$K datasets status eeshsaxena/rsna-knee-fleet-train` until it reads `ready`.
3. **Push.** `$K kernels push -p "$SP/kaggle_train/kernel/rsna-train-ttr-f34"`, then check that `$K kernels status eeshsaxena/rsna-train-ttr-f34` reads RUNNING. Kaggle allows at most 2 GPU sessions; keep 1 free for submission commits.
4. **Stop the box from also training folds 3-4.** Once the Kaggle run is RUNNING (or as soon as you trust it), create placeholders: `mkdir -p ~/fleet_runs/f3_s1043_ttr_mix50 ~/fleet_runs/f4_s1043_ttr_mix50`. The chains skip any fold whose out dir exists. If the Kaggle run errors early, `rmdir` both placeholders and the box trains folds 3-4 after fold 2.
5. **When COMPLETE (~11 h):**
   - Fetch the outputs per the RUNBOOK (use `--file-pattern` to skip big files if needed).
   - Copy each fold into the box run format:
     - `~/fleet_runs/fF_s1043_ttr_mix50/` = best ckpts + `oof_fF_s1043_{raw,ema}.csv`;
     - `~/fleet_runs/fF_s1043_ttr_mix50.log`, which must end with the `{"raw": .., "ema": ..}` closing line.
   - `verify_outputs.py` checks this.
   - **Gate:** each fold's best OOF AUC vs steven_v4 must be >= .8786 (J1 .8836 - .005). If a fold misses it, ship a 3-fold set.

### TASK 2: fold-1 boundary (~13:50 IST): switch the box to the fast trainer for fold 2

`~/fleet_runs/PAUSE_CHAIN` is already SET (since 10:12), so the old chain `~/auto_folds.sh` (pid 311059) will wait after fold 1. Follow `~/fast_build/SWITCH.md` exactly:
1. **Confirm fold 1 finished:** `tail -2 ~/fleet_runs/auto_folds.log` shows `fold 1: exited rc=0`, nvidia-smi shows no python, and `~/fleet_runs/f1_s1043_ttr_mix50/` has its best ckpts + OOF.
2. **Start the new chain:** `setsid nohup bash ~/auto_folds_fast.sh > /dev/null 2>&1 < /dev/null &`.
   - It runs folds 2, 3, 4 with `--grad_ckpt 0` (bitwise-identical math, x0.78 step time).
   - It skips folds whose dir exists, so with the TASK 1 placeholders it runs only fold 2.
3. **Stop the old chain:** find its PID, check that `/proc/<pid>/cmdline` == `/bin/bash /home/training/auto_folds.sh` and that its only child is `sleep 120`, then `kill <pid>`.
4. **Release:** `rm ~/fleet_runs/PAUSE_CHAIN`. Within ~2 min, `~/fleet_runs/auto_folds_fast.log` shows `fold 2: launching (grad_ckpt 0)`.

Fold 2 then takes ~4.1 h (ends ~18:00). **Queue the next GPU job BEFORE fold 2 ends (TASK 4):** the GPU sat idle 04:00-08:25 today.

### TASK 3: ~16:30 IST: read the two interim K4 submissions and apply the pre-registered reads

| Ref | Submission |
|---|---|
| **56842828** | K4i SOLO: TTR mix50 fold-0 J1 alone, no TTA |
| **56842829** | K4i STRICT: parent + J1 at W .20 |

Command: `$K competitions submissions -c rsna-knee-abnormality-detection --csv | head -4`

**SOLO read:**
- **<= .929:** TTR is weaker than F1 (.932). K5 keeps M1, at W .15.
- **.930-.934:** as modelled. K5 at W .20.
- **>= .935:** TTR is our strongest recipe. K5 at W .25, and TTR becomes the recipe for the v4 fallback.

**STRICT W .20 read:**
- **.944:** off the tie (true lift ~+.0009).
- **.943:** ~+.0001 (a re-measure of K2/K3).
- **<= .942:** mix50 TTR hurts at W .20. K5 at W .10.
- **ERROR:** the strict fleet stage failed on the hidden set. Read the commit log in `SP/k4/build_i1/out_main_v1` and debug before K5.

The W .50 lookup table for the SOLO read is in `SP/jump2/JUMP_PLAN_10-05.md` sec 4.

Then plan K5 for 10-06:
- 5-fold TTR mix50 + TTA B6 + M1, STRICT at the W from these reads, plus a K5 SOLO.
- Build it with the `SP/k4/k4_auto.sh` / K5 kit flow.
- Gate every commit with `k2_check.py --kaggle_commit` (it must print K2CHECK PASS) before submitting.

### TASK 4: ~18:00-18:50 IST (after fold 2): second model family (F2) on pure steven_v4 = the decorrelation probe

**Why:** pure-v4 members are the only measured way to make our models less parent-like. Their OOF Spearman vs the parent's Raptor is .840-.850, against .889-.896 for mix50. A new family should also be less correlated with our ConvNeXt.

1. **Preflight** (a few minutes; the GPU must be idle):
   `cd ~/fleet2_code && ~/miniconda3/envs/rsna/bin/python f2_preflight.py --store ~/fleet_store --n_train 3466 --cases convnext_tiny.fb_in22k_ft_in1k:1e-4 tf_efficientnetv2_s.in21k_ft_in1k:2e-4 resnet50.a1_in1k:3e-4`
   - PASS on EV2S -> use EV2S: `--backbone tf_efficientnetv2_s.in21k_ft_in1k --seed 142`.
   - FAIL (known fp16 overflow risk) -> use R50: `--backbone resnet50.a1_in1k --seed 242`.
2. **Launch the fold-0 probe ALONE.** This is the family-2 path of the trainer (`--f2 1` needs `--seed >= 100`): F1 recipe, 10 epochs, pure LLM labels.
   `cd ~/bigjump_build/code && setsid nohup ~/miniconda3/envs/rsna/bin/python fleet_train.py --f2 1 --backbone <EV2S or R50> --seed <142|242> --store ~/fleet_store --labels ~/rsna_train/llm_labels_v4_blend.csv --folds ~/rsna_train/folds5.csv --fold 0 --exclude ~/rsna_train/gold74_eval_set.csv --eval_labels ~/rsna_train/llm_labels_v4_blend.csv --gold ~/rsna_train/gold74_eval_set.csv --epochs 10 --workers 10 --out ~/fleet_runs/f0_s<seed>_f2v4 > ~/fleet_runs/f0_s<seed>_f2v4.log 2>&1 < /dev/null &`
   - Check the flag names with `fleet_train.py --help` first (see `SP/fleet2/SCHEDULE.md`). R50 alone can use bs 2.
   - Expected run time ~5-6 h.
3. **Gate F2-G** (pre-registered; compare with TTR v4 fold 0 = `~/fleet_runs/f0_s1042_ttr`, which used the same labels). PASS needs ALL of:
   - (a) OOF AUC vs steven_v4 >= .8714;
   - (b) gold58 >= .868;
   - (c) mean per-label Spearman vs the Raptor proxy (`SP/teacher_v2_out/raptor_proxy_v5v8.csv`) on the 867 fold-0 val studies <= .855;
   - (d) Spearman vs the TTR v4 OOF <= .90.

   PASS -> F2 folds 1-4 (10-06/07), then the set's SOLO + blend on 10-08. FAIL -> TTR v4 folds 1-4 via the fast trainer.
4. **Before the probe ends, queue whatever comes next.** Round 2 (`~/auto_round2.sh`) is only filler for an idle GPU.

**Still Eesh's, not Claude's:** posting the merge requests (`merge_outreach.md` has the verified facts; deadline 10-15 23:59 UTC).

