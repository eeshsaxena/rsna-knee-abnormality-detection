# FOR THE NEXT CLAUDE: RSNA Knee briefing (written 2026-10-05 ~16:55 IST)

Eesh (team "Eesh saxena", teammate SwaplY) is handing this session to you. Read this file first. It is
self-contained: what is running, what to expect, what to do next (with commands), and what NOT to do.
Deeper references: `HANDOFF.md` (top sections), `NEXT_SESSION.md`, `AUDIT_REPORT_2026-10-04.md`,
`EXTERNAL_DATA_VERDICT.md`, `merge_outreach.md`, and the archived kits/research in `session_2026-10-04/` and `session_2026-10-05/`.

---

## 0. The situation in 6 lines

- **Competition.** Kaggle "RSNA Knee Abnormality Detection": 12-label macro AUC on knee MRI. It is a code competition (2xT4 hidden rerun, 9 h limit, internet off, only `submission.csv` scores).
- **Deadlines.** Merge deadline **2026-10-15 23:59 UTC**, final **2026-10-22**. 5 submissions/day, reset 05:30 IST, shared with SwaplY.
- **Score.** Best public **0.943** ("the parent" = a public ensemble notebook), rank ~414 of ~5,100. ~950 teams sit on 0.943. Silver ~0.945-0.946, gold ~0.958.
- **Our own models.** "Fleet" = ConvNeXt-T, 2.5D slice triplets, 4 views. Best single fold reads **0.932** solo (F1 recipe, mix50 labels). Every blend with the parent has read **0.943**: our models are too similar to the parent (r_LB ~0.92).
- **Plan** (`session_2026-10-05/jump2/JUMP_PLAN_10-05.md`, adversarially verified):
  1. **merge with a stronger team** (highest EV; Eesh posts);
  2. **K5** = parent + 5-fold TTR mix50 with TTA (+ M1);
  3. a **second model family on pure LLM labels** (decorrelation probe).
- **Eesh's standing approvals.** Kaggle GPU for TRAINING (10-05) and for submission commits. The build, gate, commit and submit flow for kernels that pass their gates. Box (V100) for everything else.

---

## 1. What is RUNNING right now (do not touch)

| what | where | ETA (IST) | log / check |
|---|---|---|---|
| **TTR mix50 fold 2** (fast trainer, `--grad_ckpt 0`) | box V100, python pid 430615 under chain bash `~/auto_folds_fast.sh` (pid 404321) | ends **~18:05** | `~/fleet_runs/f2_s1043_ttr_mix50.log` (ep16 at 16:47, val .887, gold58 .913) |
| **F2 probe launcher** (2nd model family) | box, `~/auto_f2probe.sh` (pid 429756) | when fold 2 ends: preflight, then probe launch **~18:10**, probe ends **~00:00-01:00 10-06** | `~/fleet_runs/auto_f2probe.log`; `bash ~/auto_f2probe.sh --check` |
| **Kaggle training: TTR mix50 folds 3+4** | Kaggle kernel `eeshsaxena/rsna-train-ttr-f34` v1 (2xT4) | done **~21:00-21:15** (hard stop by ~22:20) | `kaggle kernels status eeshsaxena/rsna-train-ttr-f34` |
| **2 LB reads pending** | refs **56842828** (K4i SOLO = TTR mix50 fold 0 "J1" alone), **56842829** (K4i STRICT = parent + J1 at W .20) | any time now (submitted 10:27) | `kaggle competitions submissions -c rsna-knee-abnormality-detection --csv` |

**Box chain notes.**
- The fast chain skips folds 3-4: empty placeholder dirs `~/fleet_runs/f3_s1043_ttr_mix50` and `f4_...` exist, because Kaggle trains those folds.
- After fold 2 it logs `chain done`. The F2 launcher then waits for a free GPU (two checks 120 s apart), takes `~/jump2/gpu.lock` and runs:
  - `~/fleet2_code/f2_preflight.py`;
  - EV2S (`tf_efficientnetv2_s.in21k_ft_in1k`, `--lr 2e-4 --seed 142`) if it PASSes, else R50 (`resnet50.a1_in1k`, `--lr 3e-4 --seed 242`), else ABORT;
  - out dir `~/fleet_runs/f0_s142_f2v4` or `f0_s242_f2v4`.

---

## 2. What to EXPECT (pre-registered reads; act on them, do not re-litigate)

### 2a. K4i reads (any time)

| K4i SOLO (56842828) | meaning | K5 setting |
|---|---|---|
| <= .929 | TTR is weaker than F1 (.932) | K5 with M1 (`--with_m1`), W .15 |
| .930-.934 | as modelled | K5 W .20 (default) |
| >= .935 | TTR is our strongest recipe | K5 W .25 |

| K4i STRICT W .20 (56842829) | meaning |
|---|---|
| .944 | off the tie (true lift ~+.0009). Good. |
| .943 | ~+.0001 (re-measure of K2/K3). Expected. |
| <= .942 | mix50 TTR hurts at W .20: K5 at W .10 |
| ERROR | the strict fleet stage failed on the hidden set: read the commit log `SP/k4/build_i1/out_main_v1/*.log` before K5 |

### 2b. Box fold 2 (~18:05)
Its log ends with a `{"raw": .., "ema": ..}` line. References:

| fold | raw val | gold58 |
|---|---|---|
| fold 0 (J1) | .8836 | .913 |
| fold 1 | .8926 | .916 |

### 2c. Kaggle folds 3+4 (~21:00)
- The notebook self-calibrated its epoch count E (likely 13-21 instead of 24).
- Gate: each fold's best val AUC vs steven_v4 >= **.8786**.
- If E < 16: still usable, but it is your call. The K5 picker refuses E < 16 unless `K5_MIN_EPOCHS=<E>`.

### 2d. F2 probe (~00:00-01:00 10-06)
**Gate F2-G.** Compare it with TTR v4 fold 0 = `~/fleet_runs/f0_s1042_ttr` (same labels). PASS needs ALL of:
- (a) OOF AUC vs steven_v4 >= **.8714**;
- (b) gold58 >= **.868**;
- (c) mean per-label Spearman vs the parent's Raptor proxy (`SP/teacher_v2_out/raptor_proxy_v5v8.csv`) on the 867 fold-0 val studies <= **.855**;
- (d) Spearman vs the TTR v4 OOF <= **.90**.

PASS: train F2 folds 1-4 (fast if possible), then the set's SOLO + blend on 10-08. FAIL: TTR v4 folds 1-4 with the fast trainer.

### 2e. Expectations overall (MODEL/GUESS, be honest with Eesh)
- K5 most likely reads **.943-.944**; P(>= .945) ~.05-.12.
- Own models alone are worth about +.001 of true lift. **The merge is the >= .945 route.**

---

## 3. What to DO next (in order)

`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`. If Temp was wiped, use `Downloads/rsna-knee/session_2026-10-05/` (same subpaths; checkpoints are NOT archived, they live on the box).

- Kaggle CLI: `K=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/Scripts/kaggle.exe`, prefix `PYTHONIOENCODING=utf-8`, filter `warn` lines.
- Python 3.12 for the kits: `C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe` (the PATH python 3.14 lacks pydicom).
- Box: `ssh -o BatchMode=yes training@192.168.10.30 '...' < /dev/null`.

1. **Read the 2 K4i scores.** Apply table 2a, then write the result and the chosen K5 W into `HANDOFF.md`.
2. **~18:10: check that the F2 launcher fired.**
   - Run `tail ~/fleet_runs/auto_f2probe.log`. It should show the preflight verdicts and `LAUNCH ...`.
   - If it says ABORT (both backbones failed the fp16 preflight), the GPU is idle: launch TTR v4 fold 1 with the fast trainer instead.
   - Command shape: `~/fast_build/code2/fleet_train.py --grad_ckpt 0` plus the TTR recipe with `--labels ~/rsna_train/llm_labels_v4_blend.csv --seed 1042 --fold 1`.
   - Recipe flags: see `~/fleet_runs/f0_s1042_ttr.log` header or `HANDOFF.md`.
3. **~21:15: Kaggle folds 3+4 are done.** Follow `SP/kaggle_train/RUNBOOK.md`:
   - sec 5: fetch with `--page-size 200`, then run `verify_outputs.py`;
   - sec 6: copy the outputs into the empty box placeholders plus `~/fleet_runs/kaggle_f34_run_receipt.json`.
   - If the Kaggle run FAILED: `rmdir` the two placeholders and re-run `~/auto_folds.sh` so the box trains folds 3-4 (RUNBOOK sec 7).
4. **Build K5** (`SP/k5/K5_RUNBOOK.md` sec 2). The kit is verified: TTA A+B matches the bench at 0.0.
   - Shortcut: `bash "$SP/k5/k5_auto.sh"` (add `K5_WITH_M1=1` if table 2a says M1). It must end `K5AUTO READY <dir>`.
   - Then the box A0 test, which must show `BOXSTAGE PASS` and the `TTA modes` line.
   - Set W_MAIN per table 2a (check `build_k5.py --help` for the W flag).
   - Upload a new version of dataset `eeshsaxena/rsna-knee-cnxt-fleet`. Pre-create the Windows staging dir first (see runbook). Wait for `ready` and check the file list.
   - Push `rsna-sub-k5-fleet` (STRICT main) and `rsna-sub-k5-solo`.
   - Gate each commit with `k2_check.py --kaggle_commit --expect_mode ... --expect_members ... --expect_manifest ... --expect_tta $BD/K5_BUILD.json`. It must print **K2CHECK PASS**.
   - Then submit `-k eeshsaxena/<slug> -v <V> -f submission.csv`.
   - Submit before 05:30 IST if slots remain for UTC day 10-05; otherwise on 10-06.
5. **~01:00 10-06: score gate F2-G** (2d) and queue the next box job immediately. Do not let the GPU idle: it sat idle 04:00-08:25 on 10-05.
6. **Keep `HANDOFF.md` current.** Its top "NEXT CLAUDE: 4 TASKS" block is partly stale now: tasks 1-2 are done, task 4 is automated.
   - Archive new scratchpad work into `session_2026-10-05/` (text files <= 5 MB; never checkpoints).
   - Update memory (`project-rsna-knee-kaggle.md`).

---

## 4. What NOT to do

**Never (rules and safety):**
- Never post, reply or vote on Kaggle discussions, and never send merge requests. Eesh posts merges himself; drafts are in `merge_outreach.md`.
- Never handle credentials.
- Never share code, data or predictions with other teams before a merge (rules line 144).
- Never use external knee MRI data. It is gray at best (`EXTERNAL_DATA_VERDICT.md`), and only Eesh can decide.
- Never `pkill -f` anything on the box: it kills your own ssh shell. Kill only by PID, after checking `/proc/<pid>/cmdline`. Never kill the running trainers, chains or launchers listed in sec 1.
- Never put a second GPU job next to a training fold. It halves the fold's speed and can OOM it: the fast trainer holds ~9 GB of 16. Benches and side jobs go at fold boundaries, under `flock ~/jump2/gpu.lock`, with a memory cap.
- Never edit `~/bigjump_build/code`, `~/fleet2_code`, `~/fleet_runs/<run>` dirs or `~/rsna_train` on the box. Never edit any `fleet_model*.py` or `fleet_common.py` (model_sha `fa35880fca963652` / `f141aeed136dfc01` and the data contract must not change). Copy instead.
- Never submit a kernel version whose commit did not end `K2CHECK PASS`.
- Never submit any file other than `submission.csv`: other names return 400.
- Never pick pure repro_0946 as a final.
- No heavy compute on the laptop (overheats). Never run `grep -r` or `find /` there.
- No em dashes in docs (Eesh's preference).

**Don't waste time on these CLOSED levers** (each measured dead; reopen only with new evidence):
- The crop leg (cropaug/crop members): every read was .940-.942.
- Re-extracting report labels with LLMs: no text reader beats steven_v4 (.8927 on gold58).
- The own-OOF pseudo-label round (`~/auto_round2.sh`): filler. Students score below their own targets. Run it only if the GPU would otherwise idle.
- More mix50 seeds of the same family: they are parent-like, so ~0 blend value.
- Inner injection of the fleet into the parent graph: ~0, no better than a smaller global W.
- W-curve fishing with slots: the difference between W .15-.30 is <= .0005 true.

**Don't trust these:**
- gold58 for judging the parent or blends into it: the parent's DINOv2 and RadImageNet arms are in-sample there. gold58 is fine for our own members, which never train on gold74.
- These claims are refuted: "independent GPT labels" (scores .849), ".927 on gold58" (it is .926 and optimistic), "+.009 solo LB" for another team's labels, "a merge reaches gold".

---

## 5. Key paths

| thing | where |
|---|---|
| kits (newest last) | `SP/k3` (TTR, strict, member filter), `SP/k4` (5-fold TTR collect/build), **`SP/k5` (TTA, K5_RUNBOOK.md)** |
| Kaggle training kit | `SP/kaggle_train` (RUNBOOK.md, verify_outputs.py) |
| fast trainer + switch doc | box `~/fast_build/code2/fleet_train.py --grad_ckpt 0`, `~/fast_build/SWITCH.md` |
| F2 launcher | box `~/auto_f2probe.sh` (+ `SP/f2probe/README.md`) |
| fold runs | box `~/fleet_runs/f{0..4}_s1043_ttr_mix50` (+ `.log`); F1 mix50 = `f0_s42`; TTR v4 = `f0_s1042_ttr`; gated = `f0_s42_gated` |
| labels | box `~/rsna_train/{llm_labels_v4_blend,labels_mix50_logit,folds5,gold74_eval_set}.csv` |
| research + plan | `SP/jump2/` (JUMP_PLAN_10-05.md, blendmath, labels, pseudolabel, topsolutions + verify_*) |
| LB history | K2 SOLO .932 / BLEND W.30 .943; K3 TTR v4 SOLO .923 / STRICT W.25 .943; K1 crop .942; crop probes .940-.942 |
