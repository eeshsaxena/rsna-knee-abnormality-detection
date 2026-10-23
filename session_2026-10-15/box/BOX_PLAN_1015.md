# BOX PLAN 10-15 12:00 -> 10-20 12:00 (V100 box, training@192.168.10.30)
Written ~12:00 IST 10-15, before the S6F read (57201874, ~14:00-14:30) and the S read (57218523, ~16:30-17:30). No em dashes.
Tags: MEASURED / MODEL / GUESS. Merges 4 lenses: compartment (KC), s6family, boxops, runtime. PREREG_10-15 Amendments 1-5 stay
in force. Main does every qctl edit, kernel push, dataset, commit and submit. Agents do CPU work on cores 28-39 only.

## 0. Verdict
- Box state 11:44 (MEASURED): T_a f2 is at ep 18 of 21 and ends ~12:10-12:15. T_a f3 (gate already PASS) auto-launches ~12:15 and
  ends ~15:45. Queued after it: T_a f4, L13 f3, L13 f4. FF20 and FLRM read negative (A5), so K14 and RM are out and the FLIP base
  has runtime room again.
- GPU is not the binding constraint: ~119 free V100-h to 10-20 11:30 against ~30-45 h of worthwhile work (MODEL). What binds is
  build time, LB read latency (~5-6 h per read) and hidden runtime.
- ONE new family starts now: KC1, a knee-centred compartment ConvNeXt-S 256 leg cut from our d96 corpus. It is the closest
  replica of bliverrigpdoge's +.001 ingredient. Shape: a fold-0 gate model, then 2 all-data members (one on T_b labels, one on
  the dread soft labels), so it reaches the LB on 10-16 instead of 10-17.
- S6-family seed: CONDITIONAL on the S6F read and a CPU PX_PARITY gate. Write its code now; heavy I/O only after GO. Its GPU slot
  is 10-16 morning, after KC1.
- KC2 (EffNetV2-S or more seeds) runs only after a positive KC1 read. T_a f3/f4 and L13 stay as fillers below the new lines.
  Nothing running is interrupted.
- Odds: SILVER_PLAN gives P(silver) ~.2 now (MODEL). If KC1 is real (P ~.3-.4, GUESS), it is worth +3 to +10e-4 public and
  +2 to +5e-4 private (MODEL, lambda .45-.75). That is the size of the gap to the cut.

## 1. Projects ranked by EV x feasibility
| # | project | GPU h (MODEL) | EV public (GUESS) | P(real) | feasibility by 10-20 | verdict |
|---|---|---|---|---|---|---|
| 1 | KC1: ConvNeXt-S 256 knee-centred compartment leg; f0 gate + 2 all-data members; .2 rank on the team best | ~13 (f0 3.9 + 2 x 4.3; range 10-17) | +0 to +1e-3. Public analogue: +.001 twice at .2 (MEASURED display steps, selection-biased) | .3-.4 | HIGH. d96 corpus is complete on the box (4,407/4,407); train and serve readers match (max diff 1 LSB on 12/12, MEASURED). Needs 1 Kaggle CPU side kernel and ~5-6 h of build. Hidden cost +0.25-0.45 h fits FLIP and FLIP+S | GO NOW |
| 2 | S6 member s43 (S6fall recipe, blendjev_sol labels), served as group 2 inside the S6F stage | ~12.5 (10-17) | +1 to +4e-4 if S6F reads >= FLIP | ~.5 given S6F positive (~.25 unconditional). Precedent is negative: HFR7-RS .955 vs .956 (MEASURED) | MEDIUM. Needs a 265 GB fetch, unzip, a 250 GB px cache, an external trainer and an HDD contiguous-read patch | CONDITIONAL: S6F read not negative (R1) AND PX_PARITY PASS |
| 3 | KC2: EffNetV2-S 256 via the f2 frozen-BN path (f0 + 1 all-data), or 2 more ConvNeXt-S all-data seeds | ~8-10 | +1 to +3e-4 | conditional on KC1 moving the best | HIGH once KC1 exists | CONDITIONAL on the KC1 read |
| 4 | T_b second seed s1049 f0-f4 (doubles the S members) | 17.5 | 0 to +1e-4 | conditional on S moving the best | HIGH (existing line format) | FILLER, only if S reads positive |
| 5 | T_a f3 / f4 | 7.0 | ~0. MEASURED: rho(T_a, T_b) .971; mean(T_a, T_b) adds +.0000/+.0003 AUC | low | f3 runs next | f3 = BRIDGE, f4 = FILLER |
| 6 | L13 f3 / f4 | 11.5 | 0 (measured null) | ~0 | queued | BOTTOM BACKSTOP only |
| - | OAI variants (S6+OAI, KC+OAI), S6 dread member before 10-17, R7 port, K14 or RM re-entry | - | - | - | - | NO-GO. OAI raw DICOMs were deleted 10-09; R7 port is no-go; K14 and RM read negative |

## 2. CPU work that starts NOW
### 2.1 Core map
- Queue trainer: unpinned, ~4.5 cores busy (MEASURED, 11.2 % of 40). Cores 0-11 stay free for it and its 10 loader workers.
- Agents: cores 28-39 only, with the prefix P below. 7 sibling probe_lat.py processes were already on 28-39 at 11:33 (MEASURED).
- Main's heavy prep (S6 unzip and px build, only on GO): cores 12-27, `nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES=
  OMP_NUM_THREADS=2`, RAM cap 60 GB. 114 GB of RAM is available (MEASURED).
- Watch the trainer's epoch time (T_a baseline 8.7-9.9 min, MEASURED). If it rises > 10 %, cut the prep process count.
```
P='taskset -c 28-39 nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/training/miniconda3/envs/rsna/bin/python'
```

### 2.2 KC build agent (cores 28-39), in this order
- **C0.** Run `mkdir -p ~/kc/code ~/kc/gate ~/kc/labels ~/kc/logs ~/kc/smoke ~/kc_store/store`.
  - The empty `~/kc_store/store` MUST exist before 15:45.
  - Why: the runner checks missing --store / --labels inputs BEFORE the gate (MEASURED in queue_runner.sh). A missing store
    DEFERs the KC f0 line without the 120-min hold, and T_a f4 would take the GPU.
- **C1.** Write the code in ~/kc/code (no GPU):
  - **kc_common.py**: a clone of fleet_common.
    - STORE_CONTRACT = 6 KC views x 12 source slices, store 256, PNG blobs in the fleet npz format, plus a contract hash.
    - Views:
      - sag_lat: SagFS 0-55 %, 110 mm box
      - sag_med: SagFS 45-100 %, 110 mm box
      - sag_pd: SagPD central 70 %, 120 mm
      - cor_med: CorFS, 90 mm box at -22 mm
      - cor_lat: CorFS, 90 mm box at +22 mm
      - ax: full 140 mm
    - train_slices means 8 of 12 PER VIEW.
  - **kc_build_store.py**:
    - Cuts from ~/raptor_store/rsna/*.npy with kc_views / canon from ~/kcrop_probe/probe_kc.py.
    - Side rule: mean_x sign if |mean_x| > 30 mm, else the DICOM tag, else the sign if > 5 mm, else U (no flip).
  - **fleet_train_kc.py**: a copy of fleet_train.py that imports kc_common as fc. New flags:
    - --mirror 0: removes the per-study mirror coin AND the heavy-aug mirror.
    - --vbias_prior 1, an anatomical init:
      - MCL / Medial Meniscus / Medial OA -> med views
      - Lateral Meniscus / Lateral OA -> lat views
      - ACL -> sag_pd
      - PF OA / Baker's / Effusion -> ax
      - all other labels flat
    - --pretrained_file: safetensors, assert sha 2457488b.
    - --gate_out: the f0 end hook, sec 4.1 G1.
    - --max_steps: smoke only.
    - Also: model_mod 'kc_ttr', seeds 2001-2099, and --grad_ckpt 1 must work (the runner's OOM fallback uses it).
  - **Other scripts**:
    - kc_parity.py
    - kc_worker.py (serve: verbatim d96 reader sha 30aa72fd -> header side -> canon -> cut -> members, fp16, 1 pass, never hflip)
    - make_dread_labels.py
    - kc_oof_kit.py (G2 metrics, the rho matrix, the C7 refs)
- **C2.** Store test on the 458 studies with known sides (~1 min):
  `$P ~/kc/code/kc_build_store.py --src ~/raptor_store/rsna --side ~/kcrop_probe/kc_side_train.csv --out ~/kc_store/test458 --size 256 --procs 12 > ~/kc/logs/build458.log 2>&1`
- **C3.** Serve parity: the verbatim d96 reader on the eda_data DICOMs -> cut, compared with the store.
  `$P ~/kc/code/kc_parity.py --dicom ~/eda_data --store ~/kc_store/test458 --procs 12 --out ~/kc/logs/parity458.json`
  - PASS = max |diff| <= 1 and mean < .05 on 458/458. Expected cost: 1.73 s/study per thread (MEASURED), ~2-3 min.
- **C4.** CPU smoke. Run it directly as fleet_train_kc.py, NEVER as fleet_train.py: the runner treats any `python ...
  fleet_train.py` process as a live trainer and waits.
  `$P ~/kc/code/fleet_train_kc.py --device cpu --max_steps 2 --model ttr --ttr_size 256 --backbone convnext_small.fb_in22k_ft_in1k --pretrained_file ~/r7port/pretrained/convnext_small_in22k_ft_in1k.safetensors --mirror 0 --vbias_prior 1 --aug heavy --train_slices 8 --bs 1 --accum 2 --seed 2000 --fold 0 --min_cover 0 --store ~/kc_store/test458 --labels ~/r7t_labels/real/labels_r7t_b.csv --folds ~/rsna_train/folds5.csv --exclude ~/rsna_train/gold74_eval_set.csv --eval_labels ~/r7t_labels/real/labels_r7t_b.csv --gold ~/rsna_train/gold74_eval_set.csv --gate_out ~/kc/smoke/test.verdict --out ~/kc/smoke/s2 --grad_ckpt 1`
  - PASS = finite loss, ckpt args recorded, mirror off, and a test verdict written.
- **C5.** Dread labels:
  `$P ~/kc/code/make_dread_labels.py --src ~/jump2/labels_in/r2_llm_soft.csv --ref ~/r7t_labels/real/labels_r7t_b.csv --folds ~/rsna_train/folds5.csv --out ~/kc/labels/labels_dread_fleet.csv`
  - Asserts: src md5 96e6723f; the same 12 columns in the same order as T_b (MEASURED: identical headers); 100 % of the folds5
    pool covered (4,349 rows, gold58 absent); values in [0, 1].
- **C6.** When ~/kc_store/side_train.csv lands (M1):
  `$P ~/kc/code/kc_build_store.py --src ~/raptor_store/rsna --side ~/kc_store/side_train.csv --out ~/kc_store/store --size 256 --procs 12 > ~/kc/logs/build_full.log 2>&1`
  - Time: ~5-10 min (0.11 s/study load + cut, MEASURED).
  - Size: ~12 GB at 256 (MODEL, from 4.31 MB/study at 320, MEASURED).
  - Asserts:
    - coverage >= 99 %
    - side U rate <= 2 % (box sample 1/458)
    - contract hash
    - folds / gold74 disjoint
    - the test458 subset byte-equal to the full build
- **C7.** Freeze the gate refs with the same code G2 will use:
  `$P ~/kc/code/kc_oof_kit.py --refs-only --out ~/kc/gate/refs.json`
  - Expect (MEASURED earlier):
    - gold74: R7 OOF .9158, S .9123, D .9136
    - gains at .2: S +.0049, D +.0054
    - rho(T_b f0, R7 OOF) ~.817
- **C8.** Copy kc_common.py and fleet_train_kc.py into ~/fast_build/code2/ (new files only; edit no existing file). Write md5s to
  ~/kc/code/RECEIPT.txt. Then main does M4. Then write KC_READY (sec 3.3).
- **C9.** After KC f0 launches, during training:
  - Write the kernel cells for main: KC stage, BLEND, CHECK.
  - Run kc_worker on 20 eda_data studies on the box CPU, compared with the store path (probs <= 1e-3), using the first saved
    f0 epoch ckpt.

### 2.3 Main, now
- **M1.** Push kcrop/kc_side_kernel.py as a private Kaggle CPU kernel: competition attached, no internet, header-only, ~10-20 min
  (MODEL). Fetch kc_side_train.csv, scp it to ~/kc_store/side_train.csv, and record its sha.
  - This is a hard prerequisite: the report names the side in only 995 / 4,407 train studies (MEASURED).
- **M2.** Paste Amendment 6 (sec 4.1) into PREREG_10-15.md BEFORE KC f0 launches. Give its sha256 to the KC agent for KC_READY.
- **M3.** Once `LAUNCH R7Ta S1048 f3` is in ~/queue/history.txt (~12:15), run the three `qctl top` commands of sec 3.2.
  - NOT before: if KC f0 were the first job line at 12:15, it would hold the free GPU for 120 min and push T_a f3 to ~14:15.
- **M4.** Install the shim (sec 3.3) after C8, run the argv regression, and tell the KC agent.
- **M5.** Box-side idle alarm (laptop loops die with the session; see memory):
  `setsid nohup bash -c 'c=0; while true; do if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then c=0; else c=$((c+1)); fi; if [ $c -ge 10 ]; then date >> ~/fleet_runs/IDLE_ALARM; c=0; fi; sleep 60; done' > /dev/null 2>&1 < /dev/null &`
  - Check `ls -la ~/fleet_runs/IDLE_ALARM` at every poll.
- **M6.** At the S6F read, decide S6 GO / NO-GO (sec 6). On GO:
  - Get a signed URL for the competition zip from the laptop Kaggle API (read-only).
  - Pass it over ssh stdin into a mode-600 file, strip the CR, and never print it.
  - Fetch with 8 parallel `curl --http1.1 -r` range requests, with retries, into ~/rsna_comp/ under `nice -n 19 ionice -c3`.
  - Size and time: 265.0 GB at 40.8 MB/s (MEASURED rate), ~1.8 h.

### 2.4 S6 prep agent (light work on cores 28-39, started after C2-C5 finish)
- S1. Kit ~/s6seed/:
  - compact_train 14767c7a, compact_infer 34f433fe, px_cache b13e38a7 and the rest of ~/r7port/src;
  - ~/r7port/code;
  - anatomy.pt 5904dd16;
  - init sha 2457488b;
  - nartaa's E_prior_s42_fall.pt (sha 0c7297d5) by laptop scp.
- S2. ct_s6v100.py, a patched copy (the original stays untouched):
  - fp16 + GradScaler at the 3 bf16 sites;
  - S6_ALL_DATA=1;
  - a contiguous per-study px read.
  - CPU unit tests: windows byte-equal on 20 studies; fp32 loss and gradient equal on 2 studies.
- S3. px_build_s6.py (S 320, v4 views, phys 3.5, slices 3, eval_stride 1, anatomy.pt, no side model) and targets_s6seed.csv
  (from ~/raptor2/labels_raptor_rsna_blendjev_sol.csv; gold rows blank).
- S4. ONLY after GO: PX_PARITY on gold58 (~/kmimic has 58/58 DICOMs, MEASURED).
  - PASS = max |dlogit| <= 1e-3 vs compact_infer end to end.
  - Record s/study and peak RSS per process; this sets the build's process count.
  - nartaa's gold58 macro AUC becomes the bar.
- S5. ONLY after GO and S4 PASS: main runs the unzip (8 procs) and px_build_s6.py on cores 12-27 (8 procs x 2 threads, ~4-5 h
  MODEL).

## 3. GPU schedule and exact qctl edits (main only)
### 3.1 Timeline (MODEL, IST)
| window | job | start mechanism | note |
|---|---|---|---|
| now -> ~12:12 10-15 | T_a f2 (running) | - | never interrupted |
| ~12:15 -> ~15:45 | T_a f3, bridge | auto (gate PASS) | |
| ~15:47 (or the KC_READY time; hold <= 120 min) -> ~19:45 | KC1 f0 s2001, T_b | qctl top (M3); gate KC_READY | 3.9 h MODEL (3.1-4.9); epoch-time check at +25 min |
| ~19:50 -> ~00:10 10-16 | KC1 AD s2002, T_b, all-data | gate kc_f0.verdict, written by the f0 end hook | ~4.3 h MODEL (x1.25 data, no val) |
| ~00:15 -> ~04:35 | KC1 AD s2003, dread, all-data | same gate | ~4.3 h MODEL |
| ~04:40 -> ~08:10 | T_a f4 filler, OR the S6 member first if GO and its cache is ready | queue / PAUSE protocol (3.5) | |
| 10-16 ~04:40 -> ~17:30 | S6 member s43 (if GO) | external job, PAUSE protocol | 12.5 h MODEL (10-17) |
| 10-16 evening -> 10-17 | KC2 if KC1 moved the best; else T_b s1049 if S moved the best; else the L13 backstop | qctl top after the reads | |
| 10-17 -> 10-19 12:00 | members / seeds of read-positive families only | qctl top | |
| 10-19 12:00 -> 10-20 04:30 | last launches; nothing whose MODEL end > 10-20 11:30 | E6 | the runner stops launches at 12:00 but never bounds end time |

If KC_READY is late (after ~17:45):
- KC f0 is DEFERRED and T_a f4 runs ~17:47-21:15.
- KC f0 then launches ~21:17, and the KC1 members are done ~09:30 10-16.
- The KC1 submit slips to ~13:00-15:00 10-16. That still fits.

### 3.2 Edit 1 (M3): three tops, run AFTER T_a f3 is #RUN, in exactly this order
`top` inserts before the first job line (then T_a f4), so after the third command the order is: KC f0, AD s2002, AD s2003,
T_a f4, L13 f3, L13 f4.
```
bash ~/queue/qctl.sh top 'KC1 AD s2003 (ConvNeXt-S KC256, dread soft labels, all-data; PREREG_10-15 A6) | ~/fleet_runs/fA_s2003_kc256_dread | needs PASS from ~/kc/gate/kc_f0.verdict wait 1 | --kc --model ttr --ttr_size 256 --backbone convnext_small.fb_in22k_ft_in1k --pretrained_file ~/r7port/pretrained/convnext_small_in22k_ft_in1k.safetensors --mirror 0 --vbias_prior 1 --aug heavy --train_slices 8 --drop_path 0.3 --dropout 0.3 --ema 0.9995 --epochs 13 --save_epochs 7-13 --bs 1 --accum 8 --lr 1e-4 --head_lr 1e-3 --wd 0.05 --workers 10 --seed 2003 --fold -1 --store ~/kc_store/store --labels ~/kc/labels/labels_dread_fleet.csv --folds ~/rsna_train/folds5.csv --exclude ~/rsna_train/gold74_eval_set.csv --eval_labels ~/kc/labels/labels_dread_fleet.csv --gold ~/rsna_train/gold74_eval_set.csv --out ~/fleet_runs/fA_s2003_kc256_dread --grad_ckpt 0'
bash ~/queue/qctl.sh top 'KC1 AD s2002 (ConvNeXt-S KC256, T_b, all-data; PREREG_10-15 A6) | ~/fleet_runs/fA_s2002_kc256_r7tb | needs PASS from ~/kc/gate/kc_f0.verdict wait 1 | --kc --model ttr --ttr_size 256 --backbone convnext_small.fb_in22k_ft_in1k --pretrained_file ~/r7port/pretrained/convnext_small_in22k_ft_in1k.safetensors --mirror 0 --vbias_prior 1 --aug heavy --train_slices 8 --drop_path 0.3 --dropout 0.3 --ema 0.9995 --epochs 13 --save_epochs 7-13 --bs 1 --accum 8 --lr 1e-4 --head_lr 1e-3 --wd 0.05 --workers 10 --seed 2002 --fold -1 --store ~/kc_store/store --labels ~/r7t_labels/real/labels_r7t_b.csv --folds ~/rsna_train/folds5.csv --exclude ~/rsna_train/gold74_eval_set.csv --eval_labels ~/r7t_labels/real/labels_r7t_b.csv --gold ~/rsna_train/gold74_eval_set.csv --out ~/fleet_runs/fA_s2002_kc256_r7tb --grad_ckpt 0'
bash ~/queue/qctl.sh top 'KC1 f0 s2001 (ConvNeXt-S KC256 6 views x 12, T_b, mirror 0; PREREG_10-15 A6) | ~/fleet_runs/f0_s2001_kc256_r7tb | needs PASS from ~/kc/KC_READY.verdict wait 120 | --kc --model ttr --ttr_size 256 --backbone convnext_small.fb_in22k_ft_in1k --pretrained_file ~/r7port/pretrained/convnext_small_in22k_ft_in1k.safetensors --mirror 0 --vbias_prior 1 --aug heavy --train_slices 8 --drop_path 0.3 --dropout 0.3 --ema 0.9995 --epochs 13 --save_epochs 7-13 --bs 1 --accum 8 --lr 1e-4 --head_lr 1e-3 --wd 0.05 --workers 10 --seed 2001 --fold 0 --store ~/kc_store/store --labels ~/r7t_labels/real/labels_r7t_b.csv --folds ~/rsna_train/folds5.csv --exclude ~/rsna_train/gold74_eval_set.csv --eval_labels ~/r7t_labels/real/labels_r7t_b.csv --gold ~/rsna_train/gold74_eval_set.csv --gate_out ~/kc/gate/kc_f0.verdict --out ~/fleet_runs/f0_s2001_kc256_r7tb --grad_ckpt 0'
bash ~/queue/qctl.sh show; bash ~/queue/queue_runner.sh --status
```
Expected --status:
- f0 holds on KC_READY.
- AD s2003 shows WOULD-DEFER (missing labels) until C5.
- T_a f4 and the L13 lines are runnable.

Mechanics (MEASURED in queue_runner.sh):
- AD "wait 1": a premature visit costs only 60 s of hold. The f0 end hook writes the verdict before the f0 process exits, so
  AD s2002 launches at the next pick. A FAIL verdict #SKIPs both AD lines.

Lines to #HOLD now: NONE.
- T_a f4 and L13 f3/f4 stay below the KC lines as fillers.
- L13 is reached only when every line above it is done or deferred, so it cannot pre-empt KC except in the rare case where
  KC_READY appears just after L13 launched (accepted; box-never-idle rule).

### 3.3 Shim (M4) and KC_READY
The runner only ever runs `python fleet_train.py` from ~/fast_build/code2 (MEASURED, TRAINER fixed at runner start). KC therefore
enters through a dispatch shim. Without `--kc`, the code path is unchanged.
1. Back up the trainer: `cp ~/fast_build/code2/fleet_train.py ~/kc/code/fleet_train.py.bak_3e62434a`.
2. Insert the shim after the docstring and any `from __future__` line, before argparse:
```python
import sys as _kc_sys, os as _kc_os, runpy as _kc_runpy
if '--kc' in _kc_sys.argv:   # 10-15 KC dispatch shim (BOX_PLAN_1015); argv without --kc runs the unchanged trainer
    _kc_sys.argv.remove('--kc')
    _kc_runpy.run_path(_kc_os.path.join(_kc_os.path.dirname(_kc_os.path.abspath(__file__)), 'fleet_train_kc.py'), run_name='__main__')
    _kc_sys.exit(0)
```
3. Record `md5sum ~/fast_build/code2/fleet_train.py >> ~/kc/code/RECEIPT.txt`.
4. Regression:
   - (a) `$P -c "import ast; ast.parse(open('/home/training/fast_build/code2/fleet_train.py').read())"`.
   - (b) T_a's argv in the FIXES_A1 CPU dry-run form (31 studies, 1 epoch, --device cpu, out under ~/kc/smoke) gives ckpt args
     identical to the pre-shim run.
   - (c) The C4 argv with `--kc` prepended runs through `fleet_train.py`.
   - (b) and (c) are fleet_train.py processes. Run them only while T_a f3 holds the GPU, and finish by 15:20.
5. The running T_a f3 process is unaffected: Python compiled its source at launch.

KC_READY. The KC agent writes it only when C3, C4, C5, C6, C7, C8, M2 and M4 have all passed:
`echo "PASS kc_ready store=<n>/4407 U=<u> parity458=<max>/<mean> smoke=ok shim_md5=<md5> a6_sha=<sha>" > ~/kc/KC_READY.verdict`.
- If the build fails for good, write `echo "FAIL kc_ready <reason>" > ~/kc/KC_READY.verdict`. The f0 line then #SKIPs and
  T_a f4 runs.

### 3.4 Conditional edits
Install method for any edit other than `top`:
```
cp ~/queue/queue.txt ~/kc/q_edit.txt
# edit ~/kc/q_edit.txt
bash ~/queue/qctl.sh install ~/kc/q_edit.txt
```
- Do it >= 20 min away from an expected job end. install replaces the whole file, so a pick between the cp and the install
  would be undone.

| edit | trigger | action |
|---|---|---|
| E2 | KC f0 epoch-1 train-only time > 30 min (i.e. > 7 h/fold) | #HOLD both AD lines and top 224 px copies (--ttr_size 224, out names ..._kc224_...). Between 20 and 30 min: keep 256 |
| E3 | S moved the best (~17:00) | Add T_b s1049 f0-f4 ABOVE L13 (install). They become S members only via a new amendment (sec 4.3). The f0 line is in the block below; f1-f4 change --fold and the out name |
| E4 | KC1 moved the best | Write Amendment 8 first, then top the KC2 lines (EffNetV2-S f0 s2011 gated like KC f0, then AD s2012), or 2 ConvNeXt-S AD seeds s2004 (T_b) / s2005 (dread) |
| E5 | kc_f0.verdict = FAIL | The AD lines #SKIP by themselves. Main then decides on at most one retry (sec 6) |
| E6 | 10-19 20:00 | #HOLD every job line whose MODEL end > 10-20 11:30. The last 7 h job launches by ~04:30 10-20 |

E3 f0 line:
```
R7Tb S1049 f0 (TTR224 s1049 T_b second seed; filler, only if S moved the best) | ~/fleet_runs/f0_s1049_ttr_r7tb | - | --model ttr --aug heavy --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs 21 --seed 1049 --save_epochs 6-21 --workers 10 --store ~/fleet_store --labels ~/r7t_labels/real/labels_r7t_b.csv --folds ~/rsna_train/folds5.csv --exclude ~/rsna_train/gold74_eval_set.csv --eval_labels ~/r7t_labels/real/labels_r7t_b.csv --gold ~/rsna_train/gold74_eval_set.csv --fold 0 --out ~/fleet_runs/f0_s1049_ttr_r7tb --grad_ckpt 0
```

### 3.5 S6 member: external-trainer protocol (only on GO)
The runner launches only fleet_train.py, but it waits while ANY python process holds the GPU (MEASURED). So:
1. While the job ahead of it is in its last epoch (normally KC1 AD s2003), run `touch ~/queue/PAUSE`. PAUSE holds new launches
   only.
2. Wait for 2 free GPU checks 120 s apart.
3. Run a ~300-step smoke (s/step, peak memory, finite loss with GradScaler).
4. Then launch:
   `cd ~/s6seed && setsid nohup env S6_ALL_DATA=1 ~/miniconda3/envs/rsna/bin/python -u ct_s6v100.py --px <px320> --targets targets_s6seed.csv --out runs/s6s43 --arms E_prior --backbone convnext_small --init ~/r7port/pretrained/convnext_small_in22k_ft_in1k.safetensors --seeds 1 --seed-base 43 --folds 5 --only-folds 0 --epochs 16 --batch 2 --train-windows 64 --crop 288 --slices 3 --slice-jitter 0 --ema 0 --phys-mm 3.5 --spatial-attn --grad-ckpt --lr 1e-4 --head-lr 1e-3 --wd 0.05 --workers 8 --eval-workers 3 --train-sampling uniform --aug crop --resume --ckpt-every 1 --eval-stride 3 --i-have-user-authorisation-to-finetune > runs/s6s43.log 2>&1 < /dev/null &`
   - The last flag is a guard in nartaa's code. Main first reads what it asserts in compact_train 14767c7a, and uses it only
     under Eesh's standing OK for training on the box.
5. Once the job shows in nvidia-smi, run `rm ~/queue/PAUSE`. The queue resumes by itself when the job ends.
- Never leave PAUSE in place without a job on the GPU; IDLE_ALARM catches that case.
- Steps 1-5 can be packaged as ~/s6seed/launch_when_free.sh (written by an agent, armed by main with setsid nohup).

## 4. Gates and pre-registration (paste before the first GPU launch of each family)
### 4.1 AMENDMENT 6 (KC1). Written before any KC GPU launch and before any KC model number
- **Family.** KC1 is a ConvNeXt-S KC-TTR:
  - Backbone: convnext_small.fb_in22k_ft_in1k, init sha 2457488b.
  - Input: 6 laterality-canonical knee-centred views x 12 slice triplets at 256 px, cut from the d96 corpus (reader sha 30aa72fd).
  - Mirror aug OFF; anatomical vbias prior.
  - Schedule: 13 epochs, bs 1 x accum 8, lr 1e-4 / head 1e-3, wd .05, drop_path .3, dropout .3, EMA .9995.
- **Members (3, fixed now).**
  - f0 s2001: T_b (labels_r7t_b), folds5 fold 0, raw-best epoch by val AUC vs T_b.
  - AD s2002: T_b, all-data.
  - AD s2003: dread soft labels (dreaddevelopment/rsna-knee-labels, md5 96e6723f), all-data.
  - All-data members use epoch E* = f0's raw-best epoch (on a tie, the later one).
  - gold74 is never trained on.
- **Leg and blend.** The leg = the mean of the members' sigmoid probabilities, per label. The entry is a per-label
  .8 rank(team-best base output) + .2 rank(KC).
  - The weight is a prior, never fitted. No weight variants and no member re-mixes go to the LB.
- **G0 integrity (KC_READY, before f0 launches):**
  - store coverage >= 99 % of 4,407;
  - side U rate <= 2 %;
  - contract hash;
  - folds / gold74 disjoint;
  - serve parity on 458/458: max |diff| <= 1 LSB and mean < .05;
  - CPU smoke OK and shim regression OK;
  - inputs RSNA-only. The OAI-string scan still runs before any dataset.
- **G1 per member.** The f0 end hook writes ~/kc/gate/kc_f0.verdict, which gates the AD lines. Criteria:
  - The run completed with a finite loss.
  - gold74 macro AUC >= .895 for T_b members (T_a single folds .906-.909, MEASURED); >= .885 for the dread member.
  - Lateral swap test: the gold74 mean AUC over Medial/Lateral Meniscus and Medial/Lateral OA, with the med and lat views
    swapped, must be LOWER than without the swap. Otherwise the side canon is broken: FAIL.
  - A member that fails G1 is dropped. The leg ships only with >= 2 passing members; else there is no entry.
- **G2 family (before packing).** G2 is a sanity floor only: the +.005 gold gains of S and D did not show on the LB (MEASURED).
  - leg gold74 macro AUC >= .9036 (D .9136 - .010);
  - gold74 gain of .8 rank(R7 OOF) + .2 rank(KC) over R7 OOF alone >= +.0024 (= max(S +.0049, D +.0054) - .003);
  - per-label-mean Spearman(KC f0 OOF, R7 OOF) on fold 0 <= rho(T_b f0, R7 OOF) + .02, both computed by the same C7 code.
- **G3 kernel (before submit):**
  - One GPU commit.
  - The commit dumps the 3 visible studies' KC crops (uint8), d96 vol shas and probs. A box CPU run of the same ckpts on those
    crops must match within max |dp| <= 1e-3. The side must match the header rule.
  - Base rows are identical with KC removed; the blend recompute is <= 1e-6.
  - The KC stage runs in its own subprocess and process group:
    - start guard G_KC, set by the runtime review;
    - hard wall at NB_T0 + 8.2 h;
    - coverage >= 98 %, else KC is dropped with the VISIBLE canary Baker's = 0.5 on every row. Unscored studies take
      rank(base).
  - Runtime review p99 <= 8.0 h, using the commit microbench.
- **Read (R1).** The entry is one change vs the team best at submit time.
  - A best-id move = KC positive.
  - A non-move on an unchanged base = negative: KC is dropped and KC2 cancelled.
  - A non-move while the base changed = UNREAD: retest once as current best + KC.
  - A canary signature = UNREAD (runtime), never a negative.
- **Final picks.** KC is eligible only after a positive read. In J-form final picks, a KC skip falls back to base rows (only KC
  is lost), never to a canary.

### 4.2 AMENDMENT 7 (S6 member). Written only on GO, before any S6 GPU use
- **Member.** The S6fall recipe:
  - compact_train 14767c7a patched to ct_s6v100.py: fp16 + GradScaler, all-data, contiguous read; math otherwise identical;
  - seed 43, labels blendjev_sol, 34,096 steps, RSNA-only.
- **Before the GPU:**
  - PX_PARITY PASS on gold58;
  - --check-targets PASS;
  - a 300-step smoke: finite loss, s/step <= 1.8, peak memory < 15.5 GB.
- **Gate:**
  - our gold58 >= nartaa gold58 - .015;
  - rank-mean(nartaa, ours) gold58 >= nartaa - .003;
  - serve-path parity on the 3 visible studies;
  - OAI-string scan.
- **Entry:**
  - Name ours E_prior_s43_fall.pt, next to nartaa's s42, so compact_infer 34f433fe averages both in one decode pass.
  - Time-aware: 2 models only if the kernel age at S6F start is <= G2 (re-sim with c x 1.25), else nartaa only.
  - One change vs a team best that contains S6F. Read under R1.

### 4.3 AMENDMENT 8 (KC2) and Amendment 9 (S seed 2). Stubs, written only after their trigger read
- **KC2.**
  - Members: EffNetV2-S 256 via the f2 frozen-BN path with an fp16-overflow watchdog (f0 s2011 gate + AD s2012), or 2 more
    ConvNeXt-S AD seeds.
  - The leg becomes the mean of all passing KC members, with the same G1-G3. The entry is one change "KC-all vs KC1" on the best.
- **S seed 2.** F = mean(S s1047, S s1049), with w_f and the guard unchanged. One change vs a best that contains S.

## 5. How each family enters a kernel
### 5.1 KC1 pack (main, 10-16 morning)
- Build ahead (overnight 10-15, while the AD members train):
  - main builds the KC cells (C9);
  - main runs one GPU commit with a private dataset holding only the f0 ckpt (serve parity + microbench).
- On 10-16 morning:
  - OAI-string scan;
  - bump the dataset to the 3 fp16 ckpts (~100 MB each), kc_common.py, kc_worker.py, a d96 reader copy, the side rule and a
    sha manifest;
  - re-commit, G3, runtime review, amendment wording, submit.

### 5.2 KC stage cost
- Placement: appended after the base's last stage.
- Pipeline: a subprocess runs the d96 decode on CPU (4 procs) feeding 1 worker per T4; 3 members, fp16, 1 pass.
- Cost (MODEL, from the MEASURED fleet microbench of 5.3-6.0 TFLOP/s per T4 and 1.63 TFLOP per member per study):
  - GPU 0.8-0.9 s/study/T4;
  - decode ~0.2-0.3 h for ~1,300 studies, in parallel with the GPU;
  - stage 0.25-0.35 h, 0.45 h in a slow session;
  - x = 1.4-2.5 s per study-equivalent.

### 5.3 Room by base
Runtime lens, MODEL, session model B, p99 <= 8.0 h; STRESS in brackets.

| team best when KC1 is submitted | x budget | KC1 fits? | entry form |
|---|---|---|---|
| FLIP-AO (S and S6F not positive) | 6.8 [1.9] s | yes | FLIP-AO + KC, G_KC ~7.2 h (set by the review) |
| FLIP-AO + S | 4.8 [0.1] s | yes in B; marginal under STRESS | FLIP-AO + S + KC, KC after S |
| FLIP + S6F (S unread or negative) | none strict; 0.9 s relaxed | only in S's slot (S ~2.0 s/study) | Built like flips6fs20 with S replaced by KC. The re-sim must show P(S6F used) >= the flips6fs20 value |
| FLIP + S6F + S | none | no | Swap S -> KC (read as "KC beats S on S6F"). Otherwise KC is not eligible for B |

Recalibrate with every new submit-to-score time. From the S6F read: c = (latency - 4.69 h) x 3600 / 650 (runtime lens table R).

### 5.4 Slots and days (UTC day = 05:30 IST to 05:29 IST; entries >= 45 min apart)
- 10-15 slot 5: per Amendment 5 (composite if S6F becomes best, else HOLD / J floor). No KC.
- 10-16 slot 1 (05:35): J floor or the S6F/S ladder step (SILVER_PLAN).
- 10-16 slots 2-3 (~10:00-14:00): the KC1 entry, in AO + J form, one change. Read ~15:30-20:00 (MODEL, 5-6 h).
- 10-17 slot 1 (05:35): the S6 member entry if gated (one change on a best that contains S6F); else the KC1 retest if UNREAD.
- 10-17 evening / 10-18: the KC2 entry if Amendment 8 applies; S seed 2 if Amendment 9 applies.
- 10-18 / 10-19: a stacking composite of the read-positive edges, with a runtime review; the B and B' candidates.
- 10-20: weights frozen 12:00; B / B' builds in AO + J form (SILVER_PLAN sec 4).
- 10-21 11:30 IST: the last risky candidate.

## 6. Stop rules
### Global
- Never interrupt a running job, never pkill, never touch ~/queue except main via qctl.
- At most 2 new families in flight: KC plus one of S6 / KC2.
- Dates: no new family after 10-18 12:00. After 10-19 12:00, only members of read-positive lines. Nothing ends after 10-20 11:30.
- A gold gate can stop a line. It can never justify an LB entry by itself. Nothing ships with a runtime p99 > 8.0 h.

### KC1
- Side table still missing at 10-15 20:00: try (a) a header pass on the full zip, if the S6 download ran; then (b) a side
  classifier on the ~1,450 header- or report-labelled studies (GPU minutes, new prereg); else (c) stop KC1. There is no
  side-agnostic fallback: mirror aug would erase the med/lat signal.
- KC_READY not PASS by 10-16 06:00: stop KC1. The GPU goes to S6 (if GO) or to fillers.
- f0 G1 FAIL:
  - One diagnosis: swap test, side table, store parity.
  - At most one retry, and only if a concrete bug is found and fixed (new seed and new out names). Else stop.
- f0 train-only time > 30 min/epoch: E2 (224 px for the AD members).
- G2 FAIL: no entry; the GPU goes to S6 or fillers.
- LB non-move on an unchanged base: drop KC and cancel KC2. No weight variants.

### S6 member
- S6F read negative under R1: no S6 member. Stop the fetch if it was started.
- S6F read shows a canary (ACL = 0.5 on every row): UNREAD. S6 GO waits for the next read.
- PX_PARITY FAIL, or smoke s/step > 1.8 (= > 17 h), or peak memory >= 15.5 GB: stop.
- Not launched by 10-17 12:00: stop. It could not be read before the picks.
- Gate FAIL: no entry.

### Fillers
- T_a f4 and L13 never sit above a KC, S6 or KC2 line.
- T_b s1049 runs only if S moved the best.
- L13 is never used in a kernel.

## 7. Decision calendar (IST)
- 10-15 ~12:15: T_a f3 LAUNCH -> M3 (three qctl tops).
- ~12:30-13:30: the side table lands (M1) -> C6.
- ~14:00-14:30: S6F read -> S6 GO / NO-GO (M6); Amendment 7 on GO.
- by 15:20: shim and regression done. No fleet_train.py processes after 15:20.
- ~15:45: T_a f3 ends -> KC f0 starts if KC_READY; otherwise the queue holds <= 120 min.
- ~16:10: KC f0 epoch-1 time check (E2).
- ~16:30-17:30: S read -> sets the KC entry base (5.3); E3 if S is positive.
- ~19:45: f0 ends -> the verdict -> the AD lines start.
- Overnight: C9 serve stage; main's f0-only commit test.
- 10-16 ~04:35: the KC1 members are done -> E* ckpts, G2, pack, commit, G3, runtime review -> submit in slot 2 or 3.
- 10-16 ~04:40: S6 member launch (3.5), if GO.
- 10-16 evening: KC1 read -> E4 or stop.
- 10-17 05:35: S6 member entry; read around midday.
- 10-19 20:00: E6 freeze holds.
- 10-20 12:00: training freeze.

## 8. Main risks
- **Side table dependency.** The side table depends on one Kaggle CPU kernel run by main.
- **Untested crops.** KC crops are centred by FOV and stack fraction, not by anatomy anchors. The overlapping halves and the
  +-22 mm boxes absorb about +-10 % centring error (MODEL); this is not measured.
- **ConvNeXt-S throughput** is a MODEL (+-25 %). The f0 epoch check and the runner's grad_ckpt OOM fallback cover it.
- **Dread member quality** cannot be pre-checked on gold58 (the table is gold-clean). G1 at .885 is a GUESS-based floor.
- **Label overlap.** T_b carries R7 OOF signal, so KC is decorrelated from the base by geometry, not by labels. The dread member
  is the label hedge.
- **gold74 noise.** gold74 has SE ~.01, so G2 is a floor only. One LB slot decides.
- **Hidden runtime on S6F bases.** KC fits there only in S's slot. S6F may be loader-bound (c ~12-14 s, MODEL), which would cut
  P(S6F used) to ~.885 (MODEL).
- **Box I/O.** S6 prep writes ~1 TB to the HDD while KC trains. kc_store (~12 GB) should stay in the ~73 GB page cache (MODEL).
  Keep the prep on cores 12-27 and watch the epoch time.
- **Kernel environment drift.** The kernel runs pydicom 3.0.1 after R7's pip. The G3 dump-and-compare covers it.
- **Operational.** A stray PAUSE file (covered by IDLE_ALARM). The install-vs-pick race (edit >= 20 min from a job end). Any
  fleet_train.py-named CPU process delays a launch.
