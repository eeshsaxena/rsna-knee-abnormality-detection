# K2 box tests: the built notebook's appended cells on real studies

`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`
`B` = `training@192.168.10.30` (always `ssh -o BatchMode=yes`). Box python = `~/miniconda3/envs/kmimic/bin/python`
(torch 2.5.1, timm 1.0.29, cv2 4.14.0 = Kaggle's cv2, pydicom 3.0.2). Everything lives under `~/k2_test/`; nothing
under `~/kmimic`, `~/fleet*` or `~/rsna_train` is written. Never run these while a `fleet_train.py` job would lose
GPU memory: the GPU tests refuse to start below their free-memory floor.

Three levels, cheapest first. Each one ends in `k2_check.py` (independent recomputation of every file) and prints a
single `PASS`/`FAIL` line.

| Level | What runs | GPU | Studies | Wall | When |
|---|---|---|---|---|---|
| A0 CPU stage test | appended cells only, real member(s), real crop members, real DICOMs | none (CUDA hidden, nice 19) | 2 fold0_400 | ~4 min | any time, also next to training. **Done 10-04 03:22 on the interim ep4 member: PASS 35/35, fleet solo vs trainer OOF max abs 1.3e-4** (`box_cpu_transcript.txt`) |
| A1 GPU stage test | same, on the V100 | >= 6 GB free | 10 gold58 | ~5-8 min | a gap between training jobs |
| B full notebook | `run_parent.py` runs parent + appended cells end to end | >= 12 GB free (idle V100) | 10 gold58 | ~15-25 min | only when the V100 is idle; optional |

## 0. Stage (laptop -> box), after `build_k2.py` has built `SP/k2/build`

```bash
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
B=training@192.168.10.30
ssh -o BatchMode=yes $B 'rm -rf ~/k2_test/ds/rsna-knee-cnxt-fleet ~/k2_test/kernel && mkdir -p ~/k2_test/ds/rsna-knee-cnxt-fleet ~/k2_test/kernel ~/k2_test/runs ~/k2_test/k2'
scp -q -o BatchMode=yes -r "$SP/k2/code" "$SP/k2/cells" "$SP/k2/box" "$SP/k2/k2_check.py" $B:k2_test/k2/
scp -q -o BatchMode=yes "$SP"/k2/build/upload/rsna-knee-cnxt-fleet/* $B:k2_test/ds/rsna-knee-cnxt-fleet/
scp -q -o BatchMode=yes "$SP"/k2/build/kernel/*/*.ipynb $B:k2_test/kernel/
# crop members (combined kernel only; already on the box since 10-04 03:20 = the K1 Cv1 version, SP/raddata_v4).
# If the members dataset got a new version, replace the folder with that version's exact files:
#   ssh $B 'rm -rf ~/k2_test/ds/rsna-knee-cropaug-members && mkdir -p ~/k2_test/ds/rsna-knee-cropaug-members'
#   scp -q -o BatchMode=yes <upload folder of that version>/* $B:k2_test/ds/rsna-knee-cropaug-members/
ssh -o BatchMode=yes $B 'cd ~/k2_test && (echo StudyInstanceUID; sed -n 2,11p ~/kmimic/ids/gold58.csv | cut -d, -f1) > gold10.csv && (echo StudyInstanceUID; sed -n 2,3p ~/kmimic/ids/H_ids.csv | cut -d, -f1) > two.csv && wc -l gold10.csv two.csv'
```

`two.csv` = the first 2 `fold0_400` studies of `H_ids.csv` (fold-0 val studies, so the trainer's OOF has them);
`gold10.csv` = the first 10 sorted gold58 studies. Both have DICOMs in the kmimic mirror.

## A0. CPU stage test (always possible)

The OOF to compare with is the one saved next to the SHIPPED checkpoint (same run dir, same tag; its mtime must equal
the checkpoint's to the second, or the comparison is against another epoch):

```bash
ssh -o BatchMode=yes $B 'cd ~/k2_test && RUN=f0_s42_v4 TAG=ema \
  && ls --time-style=full-iso -l ~/fleet_runs/$RUN/f0_s42_${TAG}_*_best.pt ~/fleet_runs/$RUN/oof_f0_s42_${TAG}.csv \
  && cp ~/fleet_runs/$RUN/oof_f0_s42_${TAG}.csv oof_shipped.csv \
  && CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/kmimic/bin/python -W ignore k2/box/k2_box_stage.py --cpu \
       --nb kernel/rsna_sub_k2_fleetcrop.ipynb --ids two.csv --fleet_ds ds/rsna-knee-cnxt-fleet \
       --crop_ds ds/rsna-knee-cropaug-members --oof oof_shipped.csv --age_h 7.9 --threads 4 \
       --out runs/A0_$(date +%H%M) > runs/A0.log 2>&1; tail -3 runs/A0.log'
```

(Set `RUN`/`TAG` to what `d1_pick.py` printed. Use `kernel/rsna_sub_k2_fleet.ipynb` and drop `--crop_ds` to test
the fleet-only notebook alone.)

Pass = the last line reads `BOXSTAGE PASS n/n`, including `fleet solo vs trainer OOF on 2 studies: max|diff| ... <
2e-2`. On 10-04 03:22 the interim member gave 1.3e-4: the serve path (DICOM -> fleet_common decode -> worker) equals
the trainer's path (store npz) on real data. What it proves: real checkpoint + K2 worker + real decode + the K2 cell's
file contract + the K1 crop cell inside the sandbox, on Linux with Kaggle's cv2. What it cannot show: CUDA/fp16, two
GPUs, T4 speed.

Expected log landmarks (`grep -v "^\[fleet 0\] [0-9]*/" runs/A0.log`): `PARITY re-encode skipped: no train_series`
(by design: only 1 of the 4 parity studies is on the box; Kaggle runs this check), `decode probe vs ckpt
(train-side decode): match 4 mismatch 0`, `RESULT OK fleet stage clean: 1 member(s)`, `[cropfleet ..] manifest
...: 3 member(s)`, `[k2-crop ..] RESULT crop ok=True fleet clean=True | C files: written | CF file: written`. With 2
studies the W files print `(byte-identical to parent)`: per-label ranks of 2 rows coincide; the fleetsolo and the
independent recomputation are what count. Since the 10-04 fixes (`FIX_LOG.md`): `_k2_crop/` holds only `k1out*.csv`
(C6 rename, checked by `k2_check.py`), and the worker runs cv2 at 1 thread (K2d), so the OOF parity line is also the
box proof that K2d leaves the decode unchanged.

Before A0, the member must come from a FINAL `d1_pick.py` call (exit 0 + `SHIP`; exit 3 = provisional, see
RUNBOOK sec 3): a provisional OOF/ckpt pair can be mid-save.

## A1. GPU stage test on 10 gold58 studies (a training gap, >= 6 GB free)

```bash
ssh -o BatchMode=yes $B 'nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv'
ssh -o BatchMode=yes $B 'cd ~/k2_test && ~/miniconda3/envs/kmimic/bin/python -W ignore k2/box/k2_box_stage.py \
     --nb kernel/rsna_sub_k2_fleetcrop.ipynb --ids gold10.csv --fleet_ds ds/rsna-knee-cnxt-fleet \
     --crop_ds ds/rsna-knee-cropaug-members --age_h 7.9 --min_free_gb 6 \
     --out runs/A1_gold10_$(date +%H%M) > runs/A1.log 2>&1; tail -3 runs/A1.log'
```

- It refuses to start (exit message `GPU 0 free ... < --min_free_gb 6`) if a training job holds the card; nothing is
  launched then.
- Checks: `BOXSTAGE PASS`; in `runs/A1_*/report.json`: `info.fleet_stage_min`, `info.workers_s`,
  `fleet_receipt_keys.worker_loop_min`, `fleet_receipt_keys.decode_s_per_study_per_thread`, `info.crop_cell_min`.
- Stage-time projection to the hidden set (~1,322 studies, 2 x T4, 4 vCPU), to be read with care (one V100 + 40 CPUs
  here): decode is CPU-bound, Kaggle measured 0.68 s/study on 4 vCPU (prep kernel), so D ~15-20 min; GPU per member
  ~3-4.5 min on T4 (MODEL); fleet stage ~3 + max(D, N x g) min (compute.md). The 3-study Kaggle commit's
  `MICROBENCH` line is the T4 number for g; this test only shows the stage is not pathologically slow.
- `--parent_csv <csv>` swaps the stand-in parent for a real one (for example a finished box parent run's
  `localLB_parent_preds.csv`); the stage outputs do not depend on it beyond row order.

## B. Full notebook through `run_parent.py` (idle V100 only, optional)

`run_parent.py` runs any notebook given by `--nb` (the sha pin applies only to the default parent). A KMIMIC_HOME
overlay adds the two K2 datasets next to the mirrored sources without touching `~/kmimic`:

```bash
ssh -o BatchMode=yes $B 'bash ~/k2_test/k2/box/make_overlay.sh | tail -2'        # idempotent; created 10-04 03:31
# dry: builds the run dir + rewritten notebook only (validated 10-04 03:31: 132 /kaggle rewrites, 0 leftovers in
# cells 31-34, both datasets linked under kaggle/input)
ssh -o BatchMode=yes $B 'KMIMIC_HOME=~/k2_test/mimic ~/miniconda3/envs/kmimic/bin/python ~/kmimic/run_parent.py \
  --nb ~/k2_test/kernel/rsna_sub_k2_fleetcrop.ipynb --ids ~/k2_test/gold10.csv --out ~/k2_test/runs/B_prep \
  --device-mode gpu --gpus 0 --prepare-only --force'
# real run: only when nvidia-smi shows the V100 idle (run_parent refuses below 12 GB free; never pass --force here)
ssh -o BatchMode=yes $B 'KMIMIC_HOME=~/k2_test/mimic nohup setsid nice -n 5 ~/miniconda3/envs/kmimic/bin/python \
  ~/kmimic/run_parent.py --nb ~/k2_test/kernel/rsna_sub_k2_fleetcrop.ipynb --ids ~/k2_test/gold10.csv \
  --out ~/k2_test/runs/B_gold10 --device-mode gpu --gpus 0 --quiet > ~/k2_test/runs/B_gold10.nohup 2>&1 < /dev/null &'
# afterwards
ssh -o BatchMode=yes $B 'R=~/k2_test/runs/B_gold10; grep -o "\"ok\": [a-z]*" $R/localLB_parts/run_check.json; \
  grep -h "k2-fleet\|k2-crop\|cropfleet.*RESULT" $R/runner/run.log | grep "RESULT\|STAGE"; \
  python3 -c "import json; t=json.load(open(\"$R/runner/cell_times.json\")); print(json.dumps(t)[:1500])"; \
  ~/miniconda3/envs/kmimic/bin/python ~/k2_test/k2/k2_check.py --work $R/kaggle/working'
```

Pass = `run_check` ok (the parent itself was unaffected by the extra datasets), both `RESULT` lines OK, `K2CHECK
PASS`, and `$R/kaggle/working/_k2_parent.csv` byte-equal to `$R/localLB_parent_preds.csv` (the parent's own file,
exported right after the parent's last cell). Box differences vs Kaggle (boxmimic README section 4): one V100 shown
as two logical T4s, so the fleet cell launches ONE worker (CUDA_VISIBLE_DEVICES is a UUID there), and no
`train_series` in the fake root, so the worker's parity re-encode is skipped.

## Laptop CPU smoke (no box)

```bash
C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe -W ignore "$SP/k2/tests/smoke_k2.py"   # ~4.5 min
```

Last run 10-04 04:09-04:15 (after the fixes): `K2 SMOKE PASS` in 5.2 min (`SP/k2/tests/smoke_transcript.txt`):
worker parity K2 vs fleet2 max abs 0.0, K2e decode-timeout test, 22 cell scenarios (16 fleet-only, 6 combined) and
`build_k2.py` on the real interim checkpoint. A0 re-run 04:15: `BOXSTAGE PASS 36/36` (`box_cpu_transcript_fix.txt`).
