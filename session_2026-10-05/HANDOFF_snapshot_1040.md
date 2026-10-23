# RSNA Knee Abnormality Detection — HANDOFF

**Competition:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
**Final deadline:** 2026-10-22 · **Team-merge deadline:** 2026-10-15 · $77k · Research CODE comp
**Metric:** macro ROC AUC over 12 binary labels. Rerun code comp, T4x2, ≤9–12h, internet OFF.
**12 labels:** ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture.

_Last updated: 2026-10-05 10:30 IST (CURRENT STATE block below; commands in NEXT_SESSION.md)._ Full detail in memory `project-rsna-knee-kaggle.md`._

---

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

## >>> CURRENT STATE (2026-10-05 10:30 IST). Read this first; the dated sections below are history. <<<

**Scores (public LB, MEASURED):**
- Best is still **0.943** (rank ~414 of ~5,100; the .943 tie holds ~950 teams).
- Our models alone: F1 mix50 fold 0 **.932**, TTR v4 fold 0 **.923**.
- Every blend so far reads **.943**: K2 W .30, K3 strict W .25.
- The crop leg is dead (.940-.942).

**Submitted today (10:27, read ~16:30):**
- `rsna-sub-k4-solo` v1: J1 = TTR mix50 fold 0 alone. Reading it: <= .929 means weaker than F1; >= .935 means TTR is our best recipe.
- `rsna-sub-k4-fleet` v1: STRICT parent + J1 at W .20. Reading it: .944 = off the tie.
- Both passed K2CHECK. 3 slots left today (shared with SwaplY).

**Plan** (`SP/jump2/JUMP_PLAN_10-05.md`, all 4 research tracks verified):
1. **MERGE = highest-EV path.** Eesh posts the requests; drafts with verified facts are in `merge_outreach.md`. Deadline 10-15 23:59 UTC; a two-way merge fits (we have 44 subs).
2. **K5 on 10-06:** parent + 5-fold TTR mix50 with TTA (A+B, +.0020 val on TTR) + M1 (F1 mix50), STRICT W .20, plus the SOLO.
3. **F2 probe on the box ~18:50:** EfficientNetV2-S (or ResNet50 if the fp16 preflight fails) on pure steven_v4 labels. It is the only measured decorrelation lever. Gate F2-G is in the plan sec 3.
- **CLOSED by measurement:** report re-extraction (no reader beats steven_v4 .8927 on gold58) and the own-OOF pseudo-label round (filler). External data: don't (`EXTERNAL_DATA_VERDICT.md`).
- Honest odds: own models only, P(>= .945) ~.05-.12; with a merge ~.25-.35.

**Box (V100, ssh training@192.168.10.30):**
- **Fold 1** TTR mix50 (`~/fleet_runs/f1_s1043_ttr_mix50`, old chain `~/auto_folds.sh`, pid 311059) ends ~13:50.
- **`~/fleet_runs/PAUSE_CHAIN` is SET**, so the old chain will NOT start fold 2 on its own.
- At the fold-1 boundary, do the fast switch (`~/fast_build/SWITCH.md`):
  1. start `~/auto_folds_fast.sh` (`--grad_ckpt 0`, x0.78 step time, bitwise-identical math);
  2. kill the old chain by PID after the cmdline check;
  3. `rm PAUSE_CHAIN`.
- If the Kaggle folds 3+4 run is live, first create the placeholders `~/fleet_runs/f3_s1043_ttr_mix50` and `f4_s1043_ttr_mix50` so the box runs only fold 2.
- **Rules:** no GPU co-tenants during a fold. Always queue the next job before a run ends (the GPU sat idle 04:00-08:25 today).

**Kaggle (Eesh approved GPU for training on 10-05):**
- Notebook `rsna-train-ttr-f34` (`SP/kaggle_train`, RUNBOOK.md) trains folds 3+4 on 2xT4. It attaches the prep-kernel store tars, asserts hashes, self-calibrates epochs and has an 11.4 h watchdog.
- Status: built, dry-test 50/50, skeptic check running. Next: create dataset `eeshsaxena/rsna-knee-fleet-train`, then push.
- Max 2 GPU sessions at once, so keep one free for submission commits.

**Background builds running:**
- `kaggle-train-folds` (verify).
- `k5-tta-build`: worker pass B (TTA) into `SP/k5`; the gate is parity with tta_bench, max diff <= 1e-3.

**Kits:**
- `SP/k3` (TTR support, strict mode, member filter).
- `SP/k4` (5-fold TTR; `k4_auto.sh` collects finished folds -> build -> box A0).
- `SP/k5` (TTA, in progress).
- K2CHECK must pass before every submit.
- Kaggle CLI: `C:/Users/eeshs/AppData/Local/Programs/Python/Python312/Scripts/kaggle.exe`; download outputs with `--file-pattern '^[^/]+$'`.

**Docs:**
- `NEXT_SESSION.md` (commands)
- `AUDIT_REPORT_2026-10-04.md`
- `EXTERNAL_DATA_VERDICT.md`
- `merge_outreach.md`
- `session_2026-10-04/` and `session_2026-10-05/` (archived kits + research: jump2 plan and tracks, k4, k5, kaggle_train, fast_build; re-copy SP/k5 and SP/kaggle_train after their builds finish)

## 10-05 10:20 IST: PLAN IN + K4 INTERIM COMMITTING
- **Plan:** `SP/jump2/JUMP_PLAN_10-05.md` (archived in session folder later).
  - Bottom line: the MERGE is the highest-EV path (Eesh posts; drafts rewritten with verified facts in `merge_outreach.md`).
  - Own-model route is worth ~+.001 true lift at most: K4 today, K5 (5-fold TTR + TTA + M1) on 10-06, and a pure-v4 second family (F2) for decorrelation.
  - **CLOSED by measurement:** report re-extraction (no text reader beats steven_v4 on gold58) and the own-OOF pseudo-label round (filler; students score below their targets).
- **Builds verified:**
  - fast trainer `~/fast_build/code2` `--grad_ckpt 0` (bitwise identical, x0.78 step time; chain `~/auto_folds_fast.sh` installed, NOT started; `SWITCH.md`);
  - K4 kit `SP/k4` (interim `build_i1` = J1; `k4_auto.sh` for folds);
  - round-2 pipeline `~/auto_round2.sh` (filler, NOT started).
- **PAUSE_CHAIN set** 10:12, so the old chain will wait after fold 1. **Do the gc0 switch at the fold-1 boundary (~13:50) for fold 2**, and create f3/f4 placeholders if the Kaggle training run is live.
- **K4 interim pushed 10:14:**
  - `rsna-sub-k4-solo` v1 = J1 solo; read: <= .929 weaker than F1, >= .935 TTR strongest.
  - `rsna-sub-k4-fleet` v1 = STRICT parent + J1 at W .20; .944 = off the tie.
  - Dataset `rsna-knee-cnxt-fleet` new version = J1 only.
- **Building:** K5 TTA worker (B6) workflow; Kaggle folds 3+4 training notebook (verify phase).

## 10-05 09:45 IST: EXECUTION MODE (Eesh: "work continuously, less time"; APPROVED Kaggle GPU for TRAINING)
- **TTA gate** (`~/jump2/tta/run1.log`, 867 fold-0 val + gold58, passes A+B = mirror + half-slice offset):
  - TTR mix50 J1: +.0020 val (P 1.00), gold58 +.0015 -> **PASS**: serve A+B for TTR members.
  - F1 mix50: +.0014 val but gold58 -.0014 -> FAIL by the rule.
  - mean(J1, F1 mix50): val .8861, **gold58 .9188** (each alone ~.913-.914).
  - The worker's pass B (B6) is NOT built yet; next after the K4 kit.
- **Workflows running:**
  - `lb-jump-research` (resumed): labels + pseudo-label tracks, verify, JUMP_PLAN.
  - `jump-execution`: (a) `--grad_ckpt 0` fast-path trainer + SWITCH.md; (b) K4 5-fold kit `SP/k4`; (c) round-2 pseudo-label pipeline `~/jump2/r2` + `~/auto_round2.sh` (not started).
  - `kaggle-train-folds`: Kaggle notebook `rsna-train-ttr-f34` trains folds 3+4 on 2xT4 in parallel. It attaches the prep kernel tars, asserts hashes, self-calibrates epochs and has an 11.4 h watchdog.
- **Plan:**
  - Box does folds 1-2 (fold 1 ends ~13:00, fold 2 ~17:40). Create placeholder dirs `~/fleet_runs/f{3,4}_s1043_ttr_mix50` once the Kaggle run is live so the chain skips them.
  - All OOF by ~22:00, then round 2.

## 10-05 09:10 IST: JUMP RESEARCH, first 2 tracks in (`SP/jump2/topsolutions.md`, `SP/jump2/blendmath.md`; workflow resumed for the labels + pseudo-label tracks, verify and synthesis)
- **Top single models (.940-.950 LB) are better STUDENTS, not better-labelled** (CLAIMED forum).
  - Their gold58 is .925-.93 vs ours .914; their label sets score ~.89 on gold58, the same as steven_v4.
  - What they do that we do not: all-data or 5-fold training, TTA, and a soft pseudo-label round from their OWN out-of-fold predictions mixed into the report labels (Archit: must be truly OOF).
  - Our mix50 teacher (Dread) is in-sample, so mix50 is not that round.
- **Blend math (MODEL on MEASURED reads):**
  - F1 mix50 r_LB .895-.945 (median .92); TTR v4 r ~.88.
  - aastik's public fleet is STRONGER (solo >= .935), not much less parent-like.
  - Any fleet from current recipes caps at ~+.0005 true lift.
  - Need ~10 members at ~.930 solo with r <= .86, i.e. strength WITHOUT the parent teacher: own-OOF self-teacher or better report labels.
  - K4 (10-06): parent + 5-fold TTR mix50 at W .20 (best expected, P(.944+) ~.40) plus its SOLO (pins r).

## 10-05 08:30 IST: K3 READS + GPU RESTARTED + JUMP RESEARCH
- **K3 reads (MEASURED):** TTR v4 SOLO = **0.923** (vs mix50 F1 solo .932). K3 STRICT parent + mean(mix50, TTR v4) at W .25 = **0.943**.
  - The STRICT read means the fleet stage ran on the hidden set.
  - mix50 labels beat v4 by ~.009 LB solo. Every blend so far reads .943.
- **J1 (TTR fold-0 on mix50, seed 1043) finished ~04:00.** Raw val .8836 (best), EMA .8824, gold58 .913.
  - It ran ALONE at ~11-12 min/epoch (2x faster than shared).
  - MIDDLE rule: J1 < .8861, so F1 nominally leads; but TTR is close, cheaper at inference and faster to train.
- **The GPU sat IDLE 04:00-08:25** (no queue after J1). Fixed: **`~/auto_folds.sh`** runs TTR mix50 folds 1-4, seed 1043, ALONE, back to back.
  - Out: `~/fleet_runs/f{1..4}_s1043_ttr_mix50`; log `~/fleet_runs/auto_folds.log`.
  - Fold 1 started 08:25; ~4.8 h per fold; done ~03:30 10-06.
  - Pause between folds: `touch ~/fleet_runs/PAUSE_CHAIN`.
  - Result = a 5-fold TTR mix50 set (with J1) plus full own-OOF.
- **Research workflow `lb-jump-research`** (wf_a24becd9-5eb) is running:
  - top-solution intel;
  - LLM relabel feasibility (reports ARE in train.csv, multilingual);
  - blend math on the measured LB points;
  - own-OOF pseudo-label plan.
  - Output: `SP/jump2/JUMP_PLAN_10-05.md`.

## 10-04 23:29 IST: K3 SUBMITTED (0 slots left 10-04; reads ~05:30 IST 10-05)
- Refs **56830643 = K3 TTR SOLO**, **56830644 = K3 STRICT** (parent + mean(M1 mix50, M2 TTR v4) at W .25).
- Both commits `K2CHECK PASS (32/32)`.
  - Manifest sha matches the build: M2 e62c9e12.
  - PARITY re-encode maxdiff 0; decode probes 8/8 and 4/4.
  - TTR fp16 on T4 ~142-154 ms/member/study; RESULT OK.
- **How to read:**
  - K3 TTR SOLO gives TTR v4's LB strength; compare it to K2 mix50 solo .932.
  - K3 STRICT vs K2 BLEND .943 shows whether a less parent-like 2nd member helps.
  - An ERROR on STRICT means the fleet stage failed on the hidden set (no silent .943).
- Model priors (gold58): TTR solo < .932; STRICT about .942-.944.
- K3 kit archived in `session_2026-10-04/k3/` (K3_RUNBOOK.md).

## 10-04 23:15 IST: K3 COMMITTING (both v1). Box: TTR + gated done, J1 running.
- **K3 kit** = `SP/k3` (`K3_RUNBOOK.md`). Box-tested and adversarially verified:
  - code verifier: GO after fixing a strict-mode hole;
  - box verifier: GO.
- **Dataset** `eeshsaxena/rsna-knee-cnxt-fleet`, NEW version (ready 23:13). It holds:
  - M1 `f0_s42_raw_cnxt320x24_cv8810_best.pt` (sha dc7e1495);
  - M2 `f0_s1042_ema_ttr224-320x24_cv8763_best.pt` (TTR v4 EMA epoch 21, sha e62c9e12; the box file of the SAME name was later overwritten by ep22/23, so go by sha);
  - the bigjump worker and `fleet_model_ttr.py`.
- **Kernels:**
  - `rsna-sub-k3-fleet` v1: STRICT, parent + mean(M1,M2) at W .25.
  - `rsna-sub-k3-ttrsolo` v1: SOLO, M2 only.
  - Next: gate each with `k3/k2_check.py --kaggle_commit --expect_mode strict|solo --expect_members ... --expect_manifest` (K3_RUNBOOK sec 5), then submit before 05:30 IST.
- **gold58 (box, real serve path):** M1 .9144, M2 .8792, mean .9081. On val vs v4: M1 .8811, M2 .8763, mean .8840.
  - Expect K3-main at or a bit below .943 and TTR solo below .932. The TTR-solo slot is the informative one.
- **Box results:**
  - TTR f0 final: G-T2 = MIDDLE (+.0072, PASS bar .008), best EMA .8764.
  - Gated f0 final: raw best .8734 (ep7), gold58 .884 at ep9.
  - `auto_j1.sh` fired 23:12. **J1 = TTR f0 mix50 seed 1043 (pid 309153, `~/fleet_runs/f0_s1043_ttr_mix50`), running ALONE.**
  - MIDDLE rule (box_schedule 5.2): J1 OOF >= .8861 -> TTR leads production, else F1 leads.

## 10-04 22:30 IST: EXTERNAL DATA VERDICT (`EXTERNAL_DATA_VERDICT.md`, sources in `session_2026-10-04/extdata/`, adversarially verified)
- **Rules 2.6(b): allowed unless the Host prohibits.** Host test (javaduke95, 08-27, thread 733965):
  - free, plus simple registration or click-through = OK;
  - institutional approval, IRB or negotiated agreements = barrier;
  - non-commercial licences OK, and prize money is not commercial.
- **Host rulings so far:** KneeCoT BANNED (734109); OAI only if no institutional sign-off (741819). There is NO per-dataset ruling: the 743416 request is unanswered.
- **No external-data disclosure thread or deadline.** No top-12 team has disclosed external data; it is forum suspicion only.
- **Verdicts:** every candidate is GRAY except MOST and KneeCoT, which are NOT ALLOWED.
  - Lowest risk: KneeMRI from the official Zenodo record (open, CC BY-NC-ND), but it is ACL-only and sagittal-only.
  - Never use the Kaggle "knee-mri-917" re-upload (false CC0).
  - fastMRI, MRNet, SKM-TEA, OAI and its derivatives are gray.
- **Default = do NOT use external data** (time cost, unproven value, removal risk). If Eesh ever opts in, keep one final pick fully external-free. Eesh's call alone.

## 10-04 21:30 IST UPDATE: K2 READS IN (MEASURED). K2 SOLO = 0.932, K2 BLEND W .30 = 0.943.
- **SOLO .932 settles the audit's fallback question** (`AUDIT_REPORT_2026-10-04.md` Finding A).
  - Solo mode deletes `submission.csv` on any failure, so a score proves the fleet stage ran cleanly on the hidden set.
  - The BLEND .943 is a real read.
- **gold58 underrated our member by ~.018** (gold58 .914 -> LB .932). Our single fold-0 ConvNeXt-T beats Dread's single CoAtNet (.924).
- **Implied correlation** (binormal model): r_LB of the mix50 member = .90-.94.
  - W* ~.15-.22 at r .90-.92; the lift is only +.0003..+.0009.
  - mix50 is parent-like, as the audit's OOF Spearman said: mix50 .889 > gated .872 > TTR v4 .856 > v4 F1 .840.
- **Tonight (slots reset 05:30 IST): K3 is being built** (workflow `k3-build`, kit in `SP/k3`, runbook `SP/k3/K3_RUNBOOK.md`).
  - K3-main = parent + mean(mix50 F1, TTR v4) at W .25, STRICT. Strict means an error instead of a silent parent fallback. Model: +.001, displays .943-.944.
  - K3-ttrsolo = TTR v4 alone. It measures TTR's LB solo and correlation, which decides production labels/recipe for the week.
  - Both are submitted after TTR finishes (~22:55) and the commits verify.

## START HERE (2026-10-04 evening). Read `NEXT_SESSION.md` first: it has the commands, pre-registered tables and the queue.

Local copies of every 10-04 scratchpad doc/script and the box-side results: `session_2026-10-04/` (see its README.md).

**State 18:30 IST 10-04:**
- **Best public is 0.943, crop leg DEAD.** K1 3-member crop ensemble scored .942; all crop variants read .940-.942.
- **K2 submitted 11:50 IST, both reads pending at 18:19.** SOLO is ref 56816035, BLEND W .30 is ref 56816039.
  - Proxy priors: SOLO ~.922, BLEND median .940 with P(>= .943) .15.
- **gold58 is in-sample for the parent's DINOv2 and RadImageNet arms.** They score .96-.996 on gold58; the parent's .951 there is inflated.
  - Only the Raptor and CoAt arms are out-of-sample: .916-.935.
  - Our fleet f0 member scores .914 out-of-sample, about one Dread arm.
  - Its blend into the out-of-sample Raptor+CoAt side is ~0. Inner injection into the parent graph is also ~0 +/- .002 (`SP/jump_wf/inner_injection.md`).
- **Honest outlook.** Our own fleet gives a final of ~.943-.947 and P(>= .950) <= 5%. The merge is the >= .945 route.
  - Post the requests 10-05/06 per the K2 table in `NEXT_SESSION.md`.
- **Box.** TTR fold-0 is at ep13 with gold58 .884; provisional G-T2 reads +.0065 vs the v4 ref (PASS needs +.008).
  - Gated fold-0 is at ep4 with gold58 .864.
  - **`~/auto_j1.sh` is ARMED** (pid 286637, log `~/fleet_runs/auto_j1.log`). After both jobs end (~23:30) it reads G-T2 and starts J1 ALONE.
    - PASS/MIDDLE: TTR fold-0 mix50 seed 1043.
    - KILL: F1 all-data mix50 seed 43.
  - **Serial rule:** 2 concurrent jobs = 0.96x of one job alone (MEASURED).
- **GPU throughput.** The V100 is thermally throttled (~20% loss: lab airflow, not ours to change).
  - Still unharvested: always-on gradient checkpointing (~30%), per-epoch dual eval (12-17%), no compile/fused AdamW.
  - The fast-path build is half done (memory benchmark only) in `~/fast_build`; finish it next session (`NEXT_SESSION.md` sec 4).
- **Public intel (MEASURED, `session_2026-10-04/jump_wf/public_intel.md`).**
  - We are rank 419 at .943 (tie spans 352-1295; public bronze cutoff ~508).
  - Nothing public above .943 is forkable: all 5 notebooks there use private datasets. DINOsaur V5 = .943.
  - The .945-.946 public notebooks = our parent's d4 stack + a private own fleet at one global W .45-.60, i.e. the K2 design.
  - The binding constraint for our fleet is correlation with the parent (r_LB prior ~.95 for the mix50 member), not strength: pick next labels by predicted lift (`NEXT_SESSION.md` sec 5).

## START HERE (2026-10-03 afternoon) — win plan after a full research sweep

**Read first:** `scratchpad/round2/WIN_PLAYBOOK.md` (every number tagged MEASURED / CLAIM / MODEL / GUESS) and `scratchpad/round2/measured_levers.csv`. (`scratchpad` = `C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad`; copy anything you want to keep out of Temp.)

Where we are: 0.943, rank ~800 of 4,957. The 0.943 tie is 880 teams (ranks 317-1196). 0.944 = rank ~259-316; silver line 0.945 (rank 247); gold line 0.957; top 0.961.

**Already done, waiting on your OK (research-only window, nothing submitted):**
- 5 probe kernels committed and verified on Kaggle (logs show the member applied / `[v9-swap] ACTIVE`): `rsna-sub-crop25`, `rsna-sub-v9`, `rsna-sub-crop15`, `rsna-sub-crop40` (diagnostic only), `rsna-sub-v9crop25`. Plan: submit all 5 right after 00:00 UTC 10-04. Pre-registered read: ≥0.944 real, 0.942-0.943 no info, ≤0.941 hurts.
- Merge drafts (rule-compliant: public scores/approach only before merging): `Downloads/rsna-knee/merge_outreach.md`. Targets: caoyupeng/Myo team 0.949, mehmetbarankrat 0.948 (/discussion/744511), prateekgroverin 0.945 (/discussion/743374). **You must post these yourself; merge closes 10-15 23:59 UTC.** This is the single biggest lever (most of the silver probability).

**Key research findings today:**
- Training targets: **0.5 × steven_v4 + 0.5 × ECDF-mapped Dread teacher** (`scratchpad/labels_mix50_ecdf.csv`) beat steven_v4 on gold74 (+0.022, partly leakage) and the same recipe measured **+0.009 solo LB on 2 seeds** for another team. My earlier "label refinement hurts" result was a bad-base bug. Most report-label noise is report-YES false positives (Contusion YES only 49% gold-positive), not silent cells.
- The only measured way above 0.943 with public assets: an **independent fleet at one global rank weight 0.45-0.6** (+0.002-0.003, aastik/pjmathematician). Per-label weight maps are noise; flattening routing costs 0.002; cnx, positional encoding, foundation models, TTA tweaks are dead.

**10-03 EVENING, USER: "do or die, want to win; increase throughput" -> all four Kaggle steps approved and running:**
- SUBMITTED 19:47 IST (10-03 UTC slots): rsna-sub-crop25, rsna-sub-v9crop25, rsna-sub-crop15, rsna-sub-v9 (all v1). 5th slot USED 20:20 IST: rsna-sub-mixsacrop25 v1 (log verified: "DEPLOY cropaug_mixSA_all_s0_enc.pt" + "DEPLOY WROTE blended (3, 13) members 1"). Paired read: mixsacrop25 vs crop25 = effect of the new all-data mixSA member at w .25. Model-expected public: crop15 .9437, crop25 .9435, v9crop25 .9435, v9 .9430.
- Dataset eeshsaxena/rsna-knee-cropaug-members NEW VERSION: 4 old files + cropaug_mixSA_all_s0 (all-data, mixSA targets, gold58 .8874 at last=best epoch vs cropaug_f1 .8860).
- Pushed (running): rsna-knee-teacher-v2 (GPU), rsna-sub-mixsacrop25 (GPU commit), rsna-fleet-prep-s0/s1 (CPU, round3 patched code via dataset eeshsaxena/rsna-knee-fleet-src). Next: download prep outputs (tar parts) -> relay to box -> `round3/FLEET_REVIEW.md` section 6(b) fold-0 run.
- Box: queue 1 (job 1 fold-1 mixSA, then gold eval + convert) and queue 2 (ConvNeXt-T 224 all-data mixSA-logit via `box_finetune_timm_crop.py`, then R50 all-data mixSA-logit seed 1) running CONCURRENTLY.
- **10-04 05:30 IST SUBMITTED K1 main** (`rsna-sub-cropfleet` v1 `submission.csv`, W .25, 3 LLM-label crop members K=8; read ~11:30). **MEASURED: Kaggle REJECTS any other output file name** (`-f submission_fleetsolo.csv` / `submission_w15.csv` -> 400 Bad Request on CreateCodeSubmission; no slot used). => one kernel VERSION = one submittable file; every variant (solo, other W) needs its own commit (~0.5 T4-h). K2 tonight = TWO commits: main (both fold-0 members, W .30) and solo (fleet only, the S1 merge pivot). 4 slots left today.
- **BIG-JUMP PLAN (`scratchpad/bigjump/BIG_JUMP_PLAN.md`, adversarially reviewed):** no single lever gives +.007; only route = strong own ensemble blended into the untouched parent ("strong-fleet stack": TTR ConvNeXt recipe 224 px / 24 ep / heavy aug / 16-of-24 slices, report-gated silent-cell labels `labels_gated75_logit.csv`, 12-14 members over 3 recipes, 2-pass TTA, W* ~.40-.47). Needs own solo >= .937 at rho_eff .80 (.9415 at .84). Reviewer odds by 10-12: P(>=.950) ~10%, P(>=.947) ~40%, P(>=.945) ~70%. PIVOT = K2 fleetsolo read tonight (S1): <= .922 -> post merge 10-05/06 with best read; ~.925 -> merge decision at K3 read 10-05 night; >= .928 -> plan for .950. Stop: crop-leg training/variants, W curves, flat teacher-mix labels for new members. Builds B1-B5 (TTR trainer/model, gated labels staged, pack --tag, TTA bench) running in workflow; TTR fold-0 to launch ~09:45 IST on lane 1 after the gold58 parent run (replacing the family-2 auto-launch if ready).
- **BIG-JUMP BUILDS B1-B5 DONE + GO (`scratchpad/bigjump/builds/BUILD.md`, box `~/bigjump_build/code`):** TTR model (`fleet_model_ttr.py`, 224 px resize in encode, model_sha f141aeed136dfc01), trainer flags (`--model ttr --aug heavy --train_slices 16 --drop_path .2 --dropout .3 --ema .9995 --epochs 24 --save_epochs 6-24`), default path byte-identical to `~/fleet` and `~/fleet2_code`; trainer/worker parity 0.0 on real box data; gated labels staged (`~/rsna_train/labels_gated75_logit.csv`, md5 a813b986...); pack `--tag`; `tta_bench.py`; `gt2_read.py`. **Scheduler replaced 06:48:** `~/auto_after_fleet2.sh` (PID 258866; old 241241 stopped so EV2S cannot OOM the v4 run) waits for mix50 (pid 212138) to exit, runs the parent on gold58 (box mimic, gpu), then launches TTR-A fold-0 (seed 1042, pure steven_v4, out `~/fleet_runs/f0_s1042_ttr`) behind memory/proc guards. TTR ~22-30 min/epoch next to one job, 24 epochs -> ~19:00-22:30. Gates: ep1 <= 32 min, ep3 raw val >= .80, G-T2 (`gt2_read.py`) TTR >= f0_s42_v4 + .008 OOF.
- **K2 SOLO MODE added + smoke PASS (10-04 06:00):** `build_k2.py ... --main_mode solo --slug rsna-sub-k2-solo` makes `submission.csv` = the fleet alone and DELETES it if the fleet stage is not clean (so a broken solo run errors instead of reading the parent's 0.943). Tonight = two builds/commits from the same ckpts: blend (`--slug rsna-sub-k2-fleet`, W .30) and solo (`--main_mode solo --slug rsna-sub-k2-solo`) -> the S1 read. Smoke: solo_ok/solo_fail 2/2, fleet_stub_ok 23/23, fastfail/lowtime 10/10.
- **K2 PIPELINE GO (`scratchpad/k2/RUNBOOK.md`):** first command when fold-0 ends: `ssh ... 'CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/kmimic/bin/python -W ignore ~/k2_test/k2/box/d1_pick.py'` (exit 0 + SHIP line = final). DECISION (overrides the one-member rule): ship BOTH fold-0 members averaged as one fleet leg via `build_k2.py --ckpt <mix ckpt> --ckpt <v4 ckpt>=f0_s42v4` (scp the v4 file into its own folder; same-name collision noted in RUNBOOK sec 7.7), W main .30 + w20/w40 + fleetsolo; FLEET-ONLY kernel unless the K1 reads (~11:30) show the K=8 crop ensemble helps (then `--combined`). Verify on the box mimic (A0) and `k2_check.py --kaggle_commit` before submitting; Wave B must be submitted by ~17:30 IST for a same-day read.
- **LB READS 10-04 ~03:40 IST (MEASURED):** crop15 0.942, crop25 0.941, mixsacrop25 0.940, v9 0.943, v9crop25 running. => the crop leg AS SERVED by the old cell 31 (K=24 vs trained K=8, single cropaug_f1) HURTS, monotone in W; the teacher-mixed member hurts most (as predicted). The gold58 proxy (+.003 at W .25) was OPTIMISTIC because every crop checkpoint was best-of-30 selected ON gold58 (fleet checkpoints are selected on val vs v4, so the fleet's gold58 is clean). K1 (K=8, 3-member ensemble) Wave A still goes at 05:30: its fleetsolo file = first clean crop-solo read at train K; if K1 main/w15 also read <= .942, the crop leg is DEAD and all effort goes to the fleet. Fleet mix50 fold-0 epoch 5: gold58 .9002, val .8742 (G1 pass).
- **D1 PROVISIONAL (epoch 4, raw OOF on fold-0 867): mix50_logit AUCvsV4 .8611 rho_raptorproxy .857 | pure v4 .8401 / .796** (gold58 .8808 vs .8359). Mix leads solo by ~.02 but is ~.06 more parent-like (rho on train studies is inflated for both, gap is what counts). Binormal: undecidable without LB solo -> keep both to epoch 10; tonight's K2 should carry BOTH members as separate files (mix W .30, v4 W .30) for a direct LB answer, solos later.
- **BOX MIMIC DONE (`scratchpad/boxmimic/`, box `~/kmimic`):** all 17 parent sources mirrored (189 files, hashes verified, versions <= 2026-09-21), all 458 H studies' DICOMs (83,119 files, 57.1 GB) on the box, env `kmimic` (Kaggle-matching numpy/pandas/scipy/timm/pydicom/opencv; deviations: py3.11, torch 2.5.1, transformers 4.48), GPU shim (2 logical T4 on the V100), every /kaggle path rewritten. CPU-emulate smokes pass 25/25 cells; Repair-v1/Global96 match published T4 gold58 preds to 6e-4/4e-3. GPU run = GO once the V100 frees (`~/kmimic/probe/run_H_gated.sh` for full H; my `~/auto_after_fleet.sh` runs gold58 first, min-free 8 GB).
- **K1 rsna-sub-cropfleet v1 COMMIT VERIFIED 02:01 IST:** log `RESULT OK fleet blended: 3 member(s) ['cropaug_f1','cropaug_f2','crop_f1']`, fp16 probes OK, crop stage 0.2 min (0.43 s/study); output files present: submission.csv (W .25), submission_w15/w20/w35/w45.csv, submission_fleetsolo.csv. APPROVED plan: at 05:30 IST (00:00 UTC) submit `-k eeshsaxena/rsna-sub-cropfleet -v 1 -f submission.csv`, `-f submission_fleetsolo.csv` (tests non-default filename scoring), `-f submission_w15.csv`; keep 2 slots for Wave B.
- **10-04 11:25 IST K2 PUSHED (both v1, committing):** d1_pick SHIP = `f0_s42_raw_cnxt320x24_cv8810_best.pt` (mix50_logit labels, raw tag, best ep 8, val cv .8811, gold58 .914). Shipped ALONE (v4 member fails G1, gold58 .86). Dataset `eeshsaxena/rsna-knee-cnxt-fleet` created + ready (117.7 MB, 11 files). Kernels `rsna-sub-k2-fleet` (MAIN_MODE blend, submission.csv = W .30) and `rsna-sub-k2-solo` (submission.csv = fleet alone, deleted if stage not clean). Box A0 CPU test PASS on both (OOF parity 5.3e-4; solo's lone FAIL was the checker expecting the blend twin, `k2_check.py` now reads MAIN_MODE). Next: `k2_check.py --work out_k2v1 --expect_fleet clean --kaggle_commit` on each, then submit `-v 1 -f submission.csv` from both (by 17:30 IST). Also running: CPU gold58 stage run (`~/k2_test/runs/C_gold58`) + `gold58_k2.py` = local gold58 score of parent / fleet solo / W grid with the REAL parent preds.
- **10-04 LB: K1 rsna-sub-cropfleet v1 (parent + 3-member crop ensemble K=8, W .25) = 0.942 (MEASURED).** With crop15 .942 / crop25 .941 / mixsacrop25 .940, the crop leg is DEAD at every tested form (pre-registered rule). No more crop work. Box 14:54: TTR ep6 raw val .854 / gold58 .859 (26.4 min/epoch, done ~21:30); gated ep2 val .816 / gold58 .824 (69 min/epoch shared). Research workflow `rsna-jump-research` (wf_96b44231-a01) running: public intel, LB proxy model, inner injection into the parent graph, box schedule; outputs `scratchpad/jump_wf/`, plan `JUMP_PLAN_10-04.md`.
- **10-04 11:50 IST K2 SUBMITTED:** both commits `K2CHECK PASS 27/27` (parity re-encode maxdiff 0, decode 4/4, ~300 ms/study/T4). Refs 56816035 (SOLO, S1 pivot read) and 56816039 (parent + fleet W .30). Reads ~17:50 IST. 2 slots left today.
- **10-04 KEY: gold58 CANNOT judge the parent or blends into it.** Per-arm gold58 from the box mimic (`scratchpad/k2/analysis/gold58_arms.py`): the parent's DINOv2 members .96-.994 (DINO stack .996), RadImageNet e10/e13 .97/.98 = trained IN-SAMPLE on gold58's gold labels. Clean (OOS) arms: Raptor arms .916-.922, Raptor+CoAt side .935, CoAt family .93. Parent gold58 .951 is inflated, so the naive "fleet W .30 = -.008" is not trustworthy. Our fleet member .914 ~= one Dread Raptor arm. Blend into OOS arms (`gold58_oos_blend.py`): into 1 Raptor arm +.003 (W .25-.50), into 4 Raptor arms +.0024 (W .25), into the full Raptor+CoAt side (.935) ~0 / -.0008 at W .25. => pre-registered K2 reads: blend W .30 ~.940-.943 (flat), solo ~.905-.915. One fold-0 member cannot lift the parent; a lift needs a much stronger AND less parent-like fleet (TTR recipe + gated labels, 5 folds).
- **LB READ LATENCY ~6 h (MEASURED, Eesh 10-04):** rsna-sub-v9 submitted 19:33 IST -> scored by ~01:50 IST. => submit Wave A right after the 05:30 IST reset (reads ~11:30); a same-day read needs submission by ~17:30 IST. Also: the full hidden rerun incl. queue finished within ~6 h, so the parent's hidden runtime is <= ~6 h (PROFILE's 6.55 h central was high) -> >= ~2.5 h to the 8.6 h wall for our stages (fleet + crop in one kernel likely fits).
- **LB READ 10-04 ~02:10 IST: rsna-sub-v9 = 0.943 (MEASURED).** Pre-registered "no information": the v9 finespacing swap into the w .10 v5-reverse arm is a visible no-op (consistent with gold58: v9 .9137 < v5 .9221). Drop v9 as a lever. Others (crop25, v9crop25, crop15, mixsacrop25) still PENDING.
- **USER 10-04 ~02:00: "use Kaggle GPU less, the V100 (box) more".** Box mimic is BUILT (`scratchpad/boxmimic/`, box `~/kmimic`, 87 GB of parent assets + H DICOMs, env `kmimic`, `run_parent.py --ids <csv> --out <dir> --device-mode gpu`); CPU-emulate smoke passed 25/25 cells (cpu-emulate numbers are NOT valid predictions). GPU smoke blocked until the two fleet runs end (~09:30 / ~10:20 IST, 14.2/16 GB used). Then on the box, in order: (1) parent on gold58 in gpu mode (~20 min) -> real-parent blend analysis for crop + fleet W; (2) parent-speedup parity test (p1-p3); (3) EV2S preflight then family-2 fold-0 (pure steven_v4); (4) pack fold-0 fleet for K2. Kaggle GPU only for submission commits.
- **10-04 00:10 IST: K1 `rsna-sub-cropfleet` v1 PUSHED (Eesh OK'd the commit).** After it completes: check its log for `RESULT OK fleet blended: 3 member(s)`, then after 05:30 IST submit 3 files (per ENDGAME_V3 Wave A): `submission.csv` (W .25), `submission_fleetsolo.csv` (also tests that Kaggle scores a non-submission.csv file), `submission_w15.csv`. Keep 2 slots for Wave B (parent + fold-0 fleet) tonight.
- **ENDGAME V3** (`scratchpad/endgame3/ENDGAME_V3.md`, adversarially verified): binding constraint = T4 time after the parent; one LB read resolves broken / solo strength / blend ~+.003 vs ~0, so no fine W curves; picks from two independent legs (fleet, crop) + their blend; label decision D1 provisional at epoch 4, final at v4 epoch 10 (~10:45 IST), ties -> pure steven_v4; new commits need per-file fail-loud outputs; KT (Kaggle local-LB commit) off by default.
- **Parent runtime study** (`scratchpad/parent_speed/`): central hidden-run estimate ~6.55 h at N=1,322 (the 1,322 is the parent's cache-size placeholder, not a measured count), window to the 8.6 h wall ~2 h (not 36-55 min); efficiency-LB inversion implies 0.943 entries run ~6.0-6.6 h. Output-identical patches p1 (v8 concurrent on cuda:1, ~13 min), p2 (fused Raptor decode in 4 procs, +4-5 min), p3 (A5 build in 8 procs, ~8 min) = ~26 min total, all GO for an offline parity test on the box mimic.
- **teacher-v2 DONE 23:26** (all 4407 studies, 0 failures): gold58 maxspan_v5 .9221, native384_v8 .9165, finespacing_v9 .9137 (v9 does not beat v5 on gold); Raptor proxy (v5 .75 + v8 .25, the parent cell-28 weights renormalised) .9246 -> `scratchpad/teacher_v2_out/raptor_proxy_v5v8.csv`. **Decisive gold58 check vs the REAL Raptor arms:** v4 crop ensemble (cropaug_f1+f2+crop_f1, rho .812) blended in: W .15 +.0022, .20 +.0024, **.25 +.0031**, .30 +.0027, .35 +.0012, .45 -.0024; mixSA ensemble (rho .843): ~0 or negative at every W. => crop-fleet probe = v4 members at W .25 is right; expect mixsacrop25 <= crop25. Fleet gates now also log rho vs maxspan_v5 + the Raptor proxy.
- **Fleet family 2 STAGED on box `~/fleet2_code`** (design `scratchpad/fleet2/FAMILY2.md`, commands `fleet2/SCHEDULE.md` sec 4c; contract files md5-identical to `~/fleet`; EV2S + R50 weights cached): pick A `tf_efficientnetv2_s.in21k_ft_in1k` (lr 2e-4, seed 142, frozen BN, ~0.9x ConvNeXt time, 0.79x memory), pick B `resnet50.a1_in1k` (lr 3e-4, seed 242). Final verify (23:45): R50 GO; EV2S CONDITIONAL GO (random-init BN overflowed fp16, peak 1.05e5 at blocks.5.14.conv_dw; run `~/fleet2_code/f2_preflight.py` on the box and require PASS before launching EV2S, only once GPU memory frees: 14.2/16 GB used now); RadImageNet init key-remap fixed; fp16 watchdog exits(2) after 25 skipped steps. Fixed code re-copied to the box. CORRECTION to its schedule: use PURE `llm_labels_v4_blend.csv` targets (decorrelation finding), not mix50_logit, unless the epoch-4 family-1 comparison says otherwise. Launch F2-A fold-0 when the losing family-1 label variant is stopped at epoch 4. Gate history: `~/fleet_runs/gate_history.log` (auto-updated by `~/fleet/gate_loop.sh`).
- **Local LB built (`scratchpad/localLB/`, GO pending a smoke run; Kaggle-GPU kernels HELD):** H_ids.csv (gold58 + 400 fold-0), fake-root parent kernel with run_check/audit gates, aligned scorer (AUC cross-checked vs sklearn + brute force). CAVEAT: the parent is IN-SAMPLE on train studies (DINO folds, A5 m_f0..f4, Rad 5 fold heads, Raptor *_full), so the 400 fold-0 studies over-rate the parent and push W too low: use them only to compare OUR members with each other; parent+member W decisions only on gold58 (parent members never trained on it, mild checkpoint-selection bias). => the box mimic only needs the parent on gold58 (~20-30 min V100).
- **Hisernberg ext kernel BUILT, GO on code, HELD (Kaggle-GPU hold):** `scratchpad/hisern_kernel/` (id eeshsaxena/rsna-sub-hisern-ext; submission.csv = C2 6-view Raptor, plus submission_c1..c5 + receipt; anchor cells byte-identical to Hisernberg's source; fixed a variants-cell bug that would have produced no C2-C5 on the hidden rerun; time guard keeps the CoAt children's budget). Risk: this anchor (repro_0946 minus Repair-v1, outer .65, Fracture excluded from Rad) has never been scored by us. One commit -> 5 submittable files.
- **teacher-v2b DONE 22:26** (`scratchpad/teacher_v2_out/teacher_v2_native384_v8_shard0.csv`, 4407 rows, 0 failures): the parent's native384-v8 CoAtNet arm on every train study. gold58 0.9165 (widedense 0.9127; rank-avg of both 0.921); Spearman vs widedense 0.906. Because these ARE the parent's arms, use them mainly as the TRUE parent-arm reference for rho (fleet OOF vs parent arms on the 867 fold-0 val studies = gate G1b), not as label teachers (teacher-mixing raises rho, see 21:50 result). teacher-v2 (maxspan-v5 + v9) still running.
- **NEXT SUBMISSION CANDIDATE (for 10-04 slots): crop-fleet probe** `scratchpad/cropfleet/kernel/rsna-sub-cropfleet/` (crop25 notebook, last cell = `crop_fleet_cell.py`: shared decode, K=8 = the shards' training K (old cell 31 served K=24, a train/serve mismatch in every crop probe so far), uint8 parity, global rank blend). Members (manifest in dataset rsna-knee-cropaug-members, uploading 22:25): v4 LLM-label cropaug_f1 + cropaug_f2 + crop_f1 (5-head), the low-rho ones. ONE commit writes submission.csv (W .35) + submission_w15/w25/w45/w60 + fleetsolo -> one commit can fill 5 slots and map the W curve. Binormal model says best W ~0.25-0.30 for ~0.92-0.93 members; gold58 proxy (parent ~ rank-avg of real v8 arm + widedense, blended with the v4 crop ensemble) peaks at W .25 (+0.0034; .15 +.0024, .35 +.0019, .45 0, .60 -.006). Cell now writes submission.csv = W .25 + w15/w20/w35/w45 + fleetsolo (kernel rebuilt 22:45). rho vs REAL v8 arm on gold58: v4 members .775-.806, mixSA members .830-.846, v4 ensemble .813. Check the commit log for `RESULT OK fleet blended: 3 member(s)` before submitting. Optional offline K=8 vs K=24 gold bench on the box once kmimic has the gold58 DICOMs.
- **21:50 KEY RESULT (gold58, same slice sampling, `scratchpad/laneA/laneA_blend_value.py`):** teacher-mixed labels did NOT improve crop members on gold58 (cropaug_f1 v4 .8904 vs mixSA same-split .8809, all-data mixSA .8802) but RAISED their correlation with the Raptor teacher (rho .79 -> .83-.84), so the crop blend model predicts they are WORSE blend legs (w.25 lift +0.0011 -> -0.0005). Use v4-label crop members (cropaug_f1/f2, crop_f1) for the crop leg. Fleet: fold-0 on mix50_logit is training; a pure-steven_v4 fold-0 auto-launches when the ConvNeXt crop job ends (~00:30) so the local LB can pick the fleet's targets on solo + rho + blend lift.
- **~21:30 USER: offline throughput, avoid Kaggle GPU.** Hisern ext commit and the Kaggle-side local-LB kernel are ON HOLD; instead `scratchpad/boxmimic/` (workflow) recreates the Kaggle rerun env on the box (`~/kmimic`) to run the unchanged parent on the local-LB holdout offline. CPU-only Kaggle kernels still OK. R50 seed-1 crop run cancelled (fleet gets the GPU); crop trainers now have `--amp 1`.
- **FLEET FOLD-0 TRAINING since 21:07 IST** on the box (`~/fleet_runs/f0_s42.log`; auto-launched by `~/fleet/auto_fold0.sh`): prep store 4407 npz (both shards status OK, contract OK, cv2 4.14.0 pinned from the official wheel), train 3466 / val 867 (= folds5 fold 0, which contains the local-LB 400) / 74 gold held out; targets labels_mix50_logit, selection on steven_v4. Gates: G0 epoch time (shared GPU, so judge leniently), G1 val AUC vs steven_v4 >= 0.87, G1b Spearman vs teacher <= 0.85.
- Throughput tricks learned: the box CAN download Kaggle kernel OUTPUTS directly via the signed `url` from `list_kernel_session_output` (6+ MB/s vs 0.6 MB/s laptop relay); PyPI is effectively blocked on this network, so fetch wheels through a tiny internet-on Kaggle kernel (rsna-wheel-fetch) and install with `pip install --no-deps --no-index`.
- Also running on Kaggle: rsna-knee-teacher-v2b (native384-v8 arm over train, completes a 3-arm Raptor teacher with teacher-v2).
- Fleet review GO (`scratchpad/round3/FLEET_REVIEW.md`, 34 bugs fixed, exact train/serve parity in synthetic dry run); I applied N-A (atomic parent write/restore) + N-B (model_sha recorded and asserted) to `round3/fleet/`; code staged on box `~/fleet/`. Fleet targets = `labels_mix50_logit.csv` (flat 0.5, logit map; gold58 .9262; on box). Still to do before fold-0: pin box opencv-python-headless to the prep manifest's `versions.cv2` (N-C), unpack the prep tars into `~/fleet_store`.
- Building: multi-member crop-fleet cell with shared decode + global rank blend + multi-file outputs (`scratchpad/cropfleet/`, workflow running).

**Later 10-03 updates (supersede the lines above where they conflict):**
- **Judge on gold58 only.** The 16 extra gold74 studies were labelled by reading reports, and they sit inside every crop member's training set. steven_v4 scores .8927 on gold58 vs .9808 on the 16. Corrected target numbers on gold58: steven_v4 .8927, widedense teacher .9127, mix50 .9241.
- **Silence-aware targets** `scratchpad/labels_mixSA_ecdf.csv` (`build_mixSA.py`: teacher 0.75 on report-silent cells, 0.5 elsewhere; Tian's pre-registered spec) = .9275 on gold58 (+0.0033, P>0 .90; Tian measured the same +0.0032). Use it for Lane A and the fleet.
- **Crop folds are whole sites** (grouped by report language): cropaug_f1 never saw the 540 Turkish studies, f2 never saw the 672 Spanish ones. All-data crop members are a clean fix.
- **Teacher echo is mild** (in-sample agreement with labels .912 vs .886 held-out). A stronger teacher matters more than OOF-ness (Tian: strong in-sample Raptor teacher +0.009 LB; OOF self-distillation -0.001). teacher-v2 kernel (maxspan-v5 + finespacing-v9 over train on 2xT4) being built locally in `scratchpad/teacher_v2/`, not pushed.
- Top-team forum intel: `scratchpad/intel/live/top_team_posts.txt`; synthesis going to `scratchpad/round3/recipes/TOP_TEAM_RECIPES.md`.
- **Box running Lane A since 17:00 IST** (`~/rsna_train/run_laneA_queue.sh`): all-data cropaug on mixSA (best + `_last`), then fold-1 paired run, then gold eval (dumps `gold74_preds_laneA.npz`) and conversion to `~/laneA_pkg`. Read-out: `scratchpad/laneA/laneA_blend_value.py` (solo AND correlation-vs-teacher -> blend lift).
- **teacher_v2 kernel GO** (`scratchpad/teacher_v2/TEACHER_V2_README.md`, ~2.5-4.2 T4 h, needs your OK to push). **Label mapping fix:** ECDF onto steven_v4 ties destroyed teacher order; logit mean/std match keeps it (`build_mixSA.py ... logit` -> `labels_mixSA_logit.csv`, gold58 .9300 vs .9275). Use for new runs.
- Playbook adversarial verify done (corrections folded into WIN_PLAYBOOK.md section 3): **merge two-way only** (rules 143(ii) cap at request date ~330; 3-way ~338 fails; merge_outreach.md updated and now leads with label work, which the Myo team says it is stuck on); **swap the crop15 probe for a cropaug_f1 SOLO read** (staged locally: `scratchpad/kernels/rsna-sub-cropaugsolo/`, needs one commit before 00:00 UTC); fleet must select on steven_v4 not mix50 (`round3/fleet_review/fleet_train_patched.py --eval_labels`), new gate G1b (OOF-vs-Raptor Spearman <= 0.85); fleet expectation lowered to +0.000-0.002 at W 0.5; confirm the ARC job (PID 37726) is off the V100 first.

**When the box is back (V100):**
- Lane A (~3h, no data transfer): **use `scratchpad/laneA/LANE_A_RUNBOOK.md`** (cropaug recipe `box_finetune_aug_mix50.py` with `--labels labels_mixSA_ecdf.csv`, fold split pinned to cropaug_f1's, converter with parity check, paired probe `rsna-sub-mix50crop25` vs crop25). The old `box_distill_crop.py` line is superseded. Decision tree depends on the 10-04 crop reads.
- Lane F (main): aastik-contract ConvNeXt-T fleet on mix50 targets. Draft code `scratchpad/round2/dissect/`, being reviewed + dry-run now → patched copy `scratchpad/round3/fleet/` + `scratchpad/round3/FLEET_REVIEW.md`. Folds: `scratchpad/folds5.csv` (gold74 = fold -1). Gates: G0 ≤45 min/epoch, G1 OOF ≥0.87, G2 solo LB ≥0.922, G3 blend W0.5 ≥0.944.

Honest odds under this plan: bronze ~45-55%, silver ~15-20% (mostly via merge), gold ~1%, win <0.2%. Final picks: never pure repro_0946; pick two different decorrelated components; only already-scored candidates.

---

## EARLIER STATE (2026-10-03 morning) — ConvNeXt climb tested + ~dead; the play is the private leapfrog + one fp32 gate

**Teammate (SwaplY) ran the ConvNeXt M448 member on the hidden test. Pipeline WORKS, but it does not climb:** cnx-solo (fold-0 alone) = **0.894**; blended at 50% it DRAGS the 0.943 parent to 0.930 (fold-0 is ~0.05 weaker). A calibrated blend simulation (rho=0.80, fit to the observed drags near-perfectly) proves fold-0@0.894 blends to a **max of 0.9430 at weight 0** — adding cnx at any weight only ties or drags. A 5-fold of 0.894 folds reaches only ~0.90-0.91 (and the 5 folds share 98.7% of their training data, so the ensemble gain is small), still below the bar; the 5-fold only climbs if the single fold is secretly ~0.924+, which it almost certainly isn't.

**THE fp32 SHOT — TESTED 2026-10-03, DEAD.** The comp runs fp16 on T4, so 0.894 was already the fp16 number. The leakage-invariant gate (`rsna-cnx-gate-1003b`) measured fp16-vs-fp32 macro delta = **-0.13m (zero)** — fp32 is NOT a lever, 0.894 is the fold's real quality (leaky-74 macro 0.8916 ≈ hidden-test 0.894 = faithful inference; montage budgets fill correctly, no truncation). **The ConvNeXt visible-climb path is comprehensively CLOSED.** No visible climb > 0.943 exists on available resources (5-fold would need a single fold ~0.924+; 0.894 is real and the 5 folds share 98.7% of their data). Don't train folds 1-4.

**THE REAL EDGE = the private leapfrog (bigger than earlier stated).** The public 0.943 bucket holds **683 teams**; a late pure-repro lands at the spike bottom (~rank 962 of 4702). Clearing the tie by epsilon on the PRIVATE split jumps to ~rank 225 (**~737 places**). So the final-2 play is a decorrelated, oracle-validated Pick 2.

**FINAL 2 PICKS (unchanged):** Pick 1 = `repro_0946` PURE (0.943 floor) · Pick 2 = `deploy_final` (0.943 base + cropaug decorrelated member). Do NOT add cnx (it only drags). Do NOT train folds 1-4 on spec (V100 blocked by your ARC job; negative EV vs ~10%).

**Staged offline tonight (ready for quota reset):** `cnx-gold74` (fp32 gate, commit run — run first), `cnx-solo/cnx-blend-*` (done, scored), `cnx-preprocess` (montage-448 train prep — ON HOLD unless the gate flips), the 5-fold recipe fully extracted (scratchpad/cnxfull). Ecosystem re-scan 2026-10-03: no public notebook > 0.943; LB top 0.959-0.961 (private, unchanged).

---

## PRIOR STATE (2026-09-30) — endgame staged + REBASED on 0.943 frontier, blocked on Kaggle GPU quota (~60h)

**Best scored = 0.942** (`eeshsaxena/rsna-knee-repro-0942`). **CORRECTION: reproducible public ceiling = 0.943, NOT 0.946.** yamadan96/d4-public0946's "0.946" needs the author's PRIVATE ConvNeXt ensemble (not in public datasets); a faithful repro of it scores 0.943 — the true public frontier (d4-blend, dinosaur-v5, coatnet-d4-0943 ALL self-claim 0.943). **Leaderboard top = 0.961 (rhoskeri), dense 0.956-0.958 = PRIVATE teams, not copyable.** So the ride-public ceiling is 0.943.

**STRATEGY = tiebreak leapfrog (the real, defensible edge).** The public 0.943 frontier is a rigid ~200 IDENTICAL-fork spike; Kaggle breaks ties by EARLIEST submission so a late pure repro = spike-BOTTOM = zero rank upside. Above the spike is sparse, below dense → a DECORRELATED member-add blend that scores >0.943 leapfrogs the entire tie (~+98 expected private places, P≈0.85).

**FINAL 2 PICKS (rebased onto the 0.943 fast base):**
- **Pick 1 (floor):** `eeshsaxena/rsna-knee-repro-0946` PURE = the 0.943 fast 2xT4 parallel base (cuda:0/cuda:1 ThreadPool + global96 member). Guaranteed frontier score.
- **Pick 2 (leapfrog):** `eeshsaxena/rsna-knee-deploy-final` = repro_0946 base + my decorrelated **cropaug_f1** RadImageNet member (deploy_blend_cell.py, Mode A, K=24, try/except→public fallback). Aim >0.943 = jump above the tie.
- deploy_A / deploy_B (old jiweiliu-0.942 base) SUPERSEDED by deploy-final.

**Best decorrelated member = cropaug_f1 (gold74 0.8977), NOT crop-f1 (0.887).** All-member human-gold74 eval (box_eval_all_gold74.py) proved the gpt56 oracle had understated the cropaug folds. Members packaged into NEW dataset `eeshsaxena/rsna-knee-cropaug-members` v2 (cropaug_f1+f2, enc + single-head). cropaug_f1 per-col: ACL .943 MCL .931 MedMen .899 LatMen .848 MedOA .961 LatOA .845 PFOA .878 Effu .952 Syno .806 Baker .971 Cont .917 Frac .884. Best gold74 ensemble {cropaug_f1×2,cropaug_f2×2,crop_f1,crop_f4}=0.9030. (Kaggle-CLI Windows upload bug fix: pre-create `%TEMP%\.kaggle\uploads\C_\Users\...` staging dirs.)

**Two novel score-levers researched hard, both HONEST NEGATIVES:** (1) post-hoc cross-label re-ranker = no-op (my member's shared head already encodes it); (2) DISTILLATION from Dread's CoAtNet teacher = net-negative on gold74 (0.8653 < crop-f1 0.8839) — the gpt56-oracle badly understated the member's real human-gold quality (already 0.953 on Effusion), so no headroom + focal targets a K=8 crop can't see HURT. **LESSON: validate on human gold74, not the report-derived gpt56 oracle.** Post-hoc can't out-SCORE the monoculture on my compute.

**BLOCKED:** Kaggle WEEKLY GPU quota EXHAUSTED (~60h to reset; deadline 10-22 = fine). Pushing a kernel triggers a GPU rerun, so repro_0946 + deploy-final HOLD until reset. Box GPU FREE (offline member-eval only).

**AT QUOTA RESET — execution checklist:**
1. `kaggle kernels push` from LAPTOP for `kernels/repro_0946` then `kernels/deploy_final` (box token is 401-expired; laptop only). Each triggers a GPU rerun.
2. When each rerun COMPLETES, submit: `kaggle competitions submit -c rsna-knee-abnormality-detection -k <kernel> -v <ver> -f submission.csv -m "..."`.
3. Verify: repro_0946 should score ~0.943 (no timeout); deploy_final should score ≥ repro_0946 (member lifts Effusion; try/except guarantees it never scores empty — if it equals the base, the member add regressed and fell back).
4. **Adaptive final-2 selection (both must be decorrelated failure modes):**
   - If deploy_final > repro_0946 → picks = {repro_0946 (0.943 floor), deploy_final (leapfrog)}.
   - If deploy_final ≤ repro_0946 (member didn't help) → picks = {repro_0946, **repro_0942** (PROVEN 0.942, already scored ref exists = zero rerun risk, decorrelated base)}. Never two picks that share a rerun failure mode.
5. deploy_A/deploy_B (0.942 jiweiliu base) are fallbacks only if repro_0946's rerun is unstable.

**Everything compiles + metadata verified offline** (appended cell py_compiles; deploy_final = 15 datasets incl cropaug-members [status: ready, 4 files]; T4/GPU/internet-off correct).

**HARDENING WORKFLOW (23 agents, 4 dims + adversarial verify) caught a leapfrog-killing bug + optimized the blend:**
- **CRITICAL (fixed): pixel normalization.** The member cell's `_slc` returns [0,1] floats but `_embed` was applying `/127.5-1` (the [0,255]->[-1,1] map), feeding the encoder a near-constant ~-1 image = member collapses to noise = silent regression below 0.943 (try/except can't catch it, no exception). Fixed to `x*2-1`. Proof it's right: the eval that measured cropaug_f1=0.8977 (box_eval_all_gold74.py) fed [0,255]/127.5-1=[-1,1]; deploy now feeds [0,1]*2-1=[-1,1], identical. This alone would have defeated the whole leapfrog.
- **Fixed: Fluid_Sensitive NaN-abort.** `int(sr['Fluid_Sensitive'])` was unguarded; an absent/NaN column on the hidden test would abort the whole member -> silent bottom-of-tie. Now try/except -> default 0 (the member's training default), clamped to {0,1}.
- **Weight map -> robustness-optimal M10** `{Medial OA 0.10, Effusion 0.15, Baker's 0.15, rest 0}` (was {Effu 0.30, Baker 0.20, MedOA 0.10, Cont 0.10}). Monte-Carlo verified: the tie-leapfrog objective is *maximize P(private delta>0)*, not max lift (any epsilon clears the tie). M10 beats the old map on realistic-worst P(delta>0) (96.8% vs 93.7%) and halves the tail downside. Dropped Contusion (0.917 vs public 0.914 = coin-flip), halved Effusion (widest gold74 CI; rank-blend gains saturate by ~0.15).
- **Mode A confirmed (keep, not B):** Mode B doubles member inference and risks an uncatchable SIGKILL timeout (fails the submission outright = worse than delta=0).
- **Lever re-scan:** no public notebook reproducibly >0.943 (leapfrog premise confirmed); no 2nd decorrelated no-training family worth adding; per-target weight-tuning is a no-op vs decorrelation. FYI only: a separate $18K Efficiency Track exists (member cell hurts efficiency standing, not the main accuracy rank).

All 3 kernels (deploy_final/A/B) rebuilt from clean bases + corrected cell, compile OK. **Residual unhedged risk: ~3% mild-regression realistic, ~36% if gold74 is a fluke — unhedgeable until a validation submit (quota-blocked).** Nothing else to build pre-quota.

---

## TL;DR — where we are and the plan (historical, superseded by CURRENT STATE above)

- **My best LB = 0.926** (sub `56198031`, K=40 GPU T4 ensemble soup(0.4)+cropFT(0.6)). Rank ~**1919 / 4260**.
- **The public LB is a 0.941 monoculture:** ~200 teams fork ONE notebook (`mattiaangeli/bend-the-knee-to-the-dinosaurs`). Top-50 all **>0.95** with genuine separate models. #1 Scott Willis (CoAtNet), Dread Development 0.944 (CoAtNet+ConvNeXt, 13min). **My hand-built RadImageNet-MIL branch (0.926) tops out UNDER the fork crowd.**
- **The whole crowd lacks an independent validation signal** and says so openly (Raymond 66th: "model beats the labels it trained on, local validation hard to trust"; Mattia: .940→.941 is just blend-search on public 30%). **I have exactly that signal: the sharp gpt56 OOF oracle (CI ~0.012, decoupled from LB).** That is my moat + the shakeup-robustness edge (public pile is overfit to the 30% public split).
- **Host CONFIRMED** (pinned thread) commercial-LLM report-labeling is legal → my whole GPT weak-label + oracle pipeline is clean.

**STRATEGY = reproduce-and-extend:**
1. **Floor:** reproduce the public stack as my own kernel → inherit their *trained* DINOv2×20 + CoAtNet members + bank ~0.941.
2. **Moat:** re-derive the per-label blend **on my oracle** instead of the public LB, and add my genuinely-strong **RadImageNet-crop-f1** member (oracle 0.8795, LB-proven).
3. **Final 2 picks:** 1 flat/robust blend (survives shakeup) + 1 oracle-optimized blend.

---

## LIVE RIGHT NOW (2026-09-25) — ALL-IN CAMPAIGN for #1

User chose **GO ALL-IN FOR #1**. Two robust 0.941 picks BANKED (repro floor `56516549`=0.941 + combined `56526314`=0.941 = repro+my-Rad diversity). Ceiling-raiser research done; campaign executing.

**RESEARCH VERDICT** (forum "Best single-model score" + Will's 0.903 ablations): a SINGLE model reaches 0.95, and **LABELS are the universal bottleneck**, unlocked by **pseudo-label DISTILLATION** (teacher → image-informed OOF pseudo-labels that beat the report labels → student trained on them). Archit(36th) 2.5D-CoAtNet@224=0.950; Raymond(69th) 0.938→0.949 all-from-labels. My per-label profile == Will's → the unspent MODEL lever = **2.5D END-TO-END FINE-TUNE** (I only ever tried 2.5D FROZEN = doomed: RadImageNet conv1 trained on grayscale-replicate is OOD for 3 real slices → must fine-tune conv1). Moat for the whole loop = the gpt56 OOF oracle everyone else lacks.

**CAMPAIGN (gated cheapest-first):**
1. Labels: `llm_labels_v4_blend.csv`(=steven_v4) already best legit static (gold74 0.9113); per-target +pilkwang ACL/MCL/Frac=0.9151 (marginal). All public label sets + JEV gold74(74 hand-verified) in scratchpad/kneelabels/. NOTE yunus_v5=0.9994 GOLD-CONTAMINATED (exclude); gpt56=0.858 weakest → cross-check members on gold74 too.
2. **2.5D FT RUNNING** (`box_finetune_25d.py --fold 1`, pipe_25d_f1.log): GROUP=3 consecutive-slice triplets as 3ch, conv1 own lr 2e-4, two-phase BN-frozen. Warm(frozen conv1) gold58 0.62-0.68 (confirms frozen-2.5D-doomed); phase2 ep3+ must climb past champ crop-f1 gold58 0.8759 / oracle 0.8795. GATE `box_eval_25d.py` (staged) → held-out gpt56-oracle + gold74, must clear ≥~0.892 (0.8795+CI).
3. If clears: 5-fold → OOF pseudo-labels → student on pseudo → oracle-confirm pseudo>static = the 0.95 distillation moat.
4. Coarse 2.5D = existing shards (~12% adjacency); if promising, dense re-preprocess (Kaggle CPU, ordered 130-150mm@288).
Scripts (scratchpad+box): box_finetune_25d.py, box_eval_25d.py. GPU FREE.

---

## KEY FINDINGS — what works, what's dead (all on the sharp gpt56 oracle)

**Oracle = the judge.** `box_labelcv.py` / `box_planecv.py`: all-4349 OOF CV, 5-fold StratifiedGroupKFold grouped by report language, truth = `labels_llm_gpt56sol.csv` hard-binary (independent LLM labels), 1000× bootstrap, CI ~0.012 (vs gold-58 CI ~0.06). ~3× oracle→LB compression but direction holds (proven: oracle picked crop over soup, LB agreed).

| Lever | Oracle (gpt56) | Verdict |
|---|---|---|
| **RadImageNet-crop-f1** (150mm crop, full-BB FT, BN-frozen) | **0.8795** | **CHAMPION single encoder.** Per-label: ACL .900 MCL .818 MedMen .925 LatMen .892 MedOA .912 LatOA .878 PFOA .871 Effu .880 Syno .771 Baker .904 Cont .898 Frac .905 |
| soup(full)+cropft(crop) ensemble | 0.8868 | = LB 0.926 (best deployed) |
| Plane-routing full-FT PlaneMIL | 0.8724 | **RETIRED** — full-BB plane-FT DEGRADES the encoder (shared head 0.8541); routing recovers +0.018 (Syno +0.061, PFOA +0.050) but still < 0.8795. Only maybe-keep Syno/PFOA columns as per-label blend inputs. |
| **Frozen DINOv2 ViT-S/14 on crop** | **0.7186** | **DECISIVE NEGATIVE** (−0.16 vs Rad). The crowd's backbone is WEAK RAW; its 0.941 value is ALL in the 20-member fine-tune + rank-blend. **DO NOT re-train DINOv2 from frozen feats.** Confirms medical-pretrain worth +0.16 + validates the oracle. |
| co-train (kneeMRI ACL aux) | 0.8540 | dead (domain gap corrupts shared enc) |
| label surgery (verdict/mask) | <0.8795 | dead (v4soft 0.25 fill is the better target) |
| per-target head, tail-SWA, partial-unfreeze, crop-soup | all ≤ champion | dead |

**Core unlock history:** RadImageNet ResNet50 medical-pretrain (+0.13 over natural-image) → 150mm physical crop via DICOM PixelSpacing (fine-tune-on-crop unlocks it) → full-backbone FT (BN running-stats FROZEN, `model.enc.eval()`, lr_bb 3e-5, two-phase warm-then-unfreeze) → K=40 GPU ensemble.

**Public 0.941 stack composition** (from forum + notebook metadata): DINOv2 ViT-S ×20 members + DINOv3/A5 ×5 folds + RadImageNet heads (they set its cal weight to 0.00 — didn't help their public LB) + Dread's "Raptor" = 3 CoAtNet checkpoints (v5 maxspan / v8 native384 / v10 native384dense, top-3 by gold58, equal-rank-mean, 130mm fov, 384px) + ConvNeXt. Per-label rank-blend: `_coatnet_weight` {default 0.60; ACL .75, MedMen .80, LatMen 1.00(=CoAt-only), LatOA .75, Frac .75} + `LEGACY_MEMBER_WEIGHT_BY_TARGET` {LatMen 15, MedOA 2.5, LatOA 15, Cont 5}. **These weights were tuned on public-30% → my oracle re-derives them (shakeup-robust).**

---

## NEXT ACTIONS (in order)

1. **[GATE] Read the repro floor score** (`kaggle competitions submissions -c rsna-knee-abnormality-detection`, ref 56516549). If ~0.941 → reproduction faithful, proceed. If ERROR → debug notebook.
2. **Phase 2 (the moat): run the repro on TRAIN images** → get the public stack's per-study prediction on all 4349 report-labeled studies → oracle-score its per-label AUC → derive the **oracle-optimal per-label blend of {public stack, my RadImageNet-crop-f1}** → deploy that blend on test. This is a clean 2-system per-label blend, NOT fragile per-member re-weighting. (Modify repro notebook: iterate `train/` studies, dump predictions npz; train images are in the competition data.)
3. **Deploy + submit** the oracle-blended kernel; compare vs 0.926 and the repro floor.
4. **Final 2 picks:** 1 flat/robust blend + 1 oracle-optimized blend (per starkhushi's "keep a flat-0.60 second pick" — the public per-label map overfits, esp. LatMen=1.00).
5. (Optional diversity) genuinely-different oracle-validated member; OAI external data is license-risky/unresolved — don't rely.

---

## ASSETS

**Box** `ssh training@192.168.10.30`, env `~/miniconda3/envs/rsna/bin/python` (torch 2.5.1+cu121, timm, HF reachable), workdir `~/rsna_train/`:
- Data: `~/rsna_shards_384_crop` (111 shards, 150mm crop, meta = [study_id, plane pid 0=Sag/1=Cor/2=Ax, fluid fb]), `~/rsna_shards_384` (full-frame).
- Labels: `llm_labels_v4_blend.csv` (soft train target), `labels_llm_gpt56sol.csv` (oracle truth), `train.csv`, `_reports_lang.csv`.
- Encoders: `ResNet50_radimagenet.pt` (base), `cropft_enc.pt`, `soup_enc.pt`, `cropaug_f2_enc.pt`, `finetune_plane_f1_radimagenet.pt` (plane-FT, retired).
- Emb caches: `emb_cache_cropft.npz` (champion), `emb_cache_soup.npz`, `emb_cache_planeft.npz`, `emb_cache_dinov2crop.npz`, + crop-aug family; full-frame `emb_cache_dinov3.npz`/`base384.npz`/`biomedclip.npz`/`densenet121.npz` (all weak, pre-crop).
- Head weights: `head_weights_soup/cropft/cropaug_f2.pt`.
- Oracle/tools: `box_labelcv.py`, `box_planecv.py` (plane-routing head), `box_ens_multi_oracle.py` (Caruana greedy), `box_finetune_radimagenet.py` (crop FT recipe), `box_embed_crop_gpu.py`, `box_embed_dinov2crop.py`, `box_embed_planeft.py`.

**Kaggle kernels (mine):**
- `eeshsaxena/rsna-knee-repro-dino941` — the public-stack reproduction (floor). ← current.
- `eeshsaxena/rsna-knee-infer-ens3` — 3-encoder K40 (soup+crop-f1+cropaug-f2, oracle 0.8886) BUILT but never confirmed submitted.
- `eeshsaxena/rsna-knee-infer-crop`, `rsna-knee-infer-ens-soupcrop` (v3 = the 0.926 sub).
- Dataset `eeshsaxena/rsna-knee-radimagenet` holds my enc + head weights.

**Laptop scratchpad** (`AppData/Local/Temp/claude/.../scratchpad/`): `rsna_pub/` (pulled Mattia + starkhushi notebooks + Mattia's exact `kernel-metadata.json` = repro source list), `repro_out/` (repro outputs: submission.csv, _coat_arm_receipt.json), `kernels/repro_dino941/`.

---

## HOW-TO / GOTCHAS

- **Kaggle push: LAPTOP ONLY.** Box `~/.kaggle/kaggle.json` token is EXPIRED (401). Laptop token is valid. Laptop had a stale `slugify` 0.0.1 shadowing python-slugify under Py312 — FIXED (uninstalled slugify, force-reinstalled python-slugify). If it recurs: `"C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe" -m pip uninstall -y slugify; ... -m pip install --force-reinstall --no-deps python-slugify`.
- **Code-comp submit needs BOTH kernel AND file:** `kaggle competitions submit -c rsna-knee-abnormality-detection -k <user/kernel> -v <ver> -f submission.csv -m "..."`. Submitting reruns the notebook on the full hidden test (commit only sees a 3-study sample).
- **GPU kernel metadata:** `"enable_gpu":true` + `"machine_shape":"NvidiaTeslaT4"` (Kaggle default P100 sm_60 is incompatible with the offline torch). "Batch GPU session count of 2 reached" = concurrency limit, retry.
- **Box V100 (sm_70):** `cudnn.enabled=False` for DINOv3 forward stability (RadImageNet ResNet50 is fine with cudnn on). Startup GPU gate pattern (`while free<9000: wait`) in FT scripts to yield to labmates.
- **Shared box etiquette:** yield GPU to other jobs (labmate ComfyUI/ARC jobs appear). **The Claude Code harness BLOCKS killing others' processes** — if GPU must be freed from someone else's job, the USER does it from their own session.
- **User handles all Kaggle secrets / HF tokens / admin passwords himself.** Never enter credentials.

---

## OPEN LEDGER
- Best deployed LB: **0.926**. Public fork baseline: 0.941. Target cluster: >0.95.
- Daily submission budget: 5/day (2 scored for final). Today: 1 used (repro), 4 left.
- Pending: repro floor score (56516549). Then Phase 2 oracle re-blend.
