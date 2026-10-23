# RUNBOOK (Eesh): frac75 folds 2, 3, 4 (+ fold 0 seed 1044) on a second teammate's Kaggle GPU

Written 2026-10-07 by Claude. Nothing here was done on Kaggle: no dataset was created or versioned, nothing was pushed,
shared or submitted. Every Kaggle step below is yours. No em dashes.

## 0. What and why

Two notebooks, run at the same time on ONE teammate account with a full 30 h weekly quota (2 GPU sessions per account):

| | slug | GPU0 | GPU1 | outputs |
|---|---|---|---|---|
| A | `rsna-train-ttr-frac75-f23` | fold 2, seed 1043 | fold 3, seed 1043 | `f2_s1043_ttr_frac75`, `f3_s1043_ttr_frac75` |
| B | `rsna-train-ttr-frac75-f4x` | fold 4, seed 1043 | fold 0, seed 1044 | `f4_s1043_ttr_frac75`, `f0_s1044_ttr_frac75` |

- **Recipe.** Identical to `rsna-train-ttr-frac75-f01` (swapiy, folds 0+1): the box TTR224 recipe with ONE change,
  `--labels labels_frac75_logit.csv` (dataset `eeshsaxena/rsna-knee-frac75-labels`, sha256 `5036425c...` baked in).
- **Folds 2-4 seed 1043** = the frac75 members of box queue **Q2 f2/f3/f4** (21 ep each, ~12 h of V100). With them
  done on Kaggle, a frac75 PASS needs no V100 time at all for Q2.
- **Fold 0 seed 1044** = a second-seed frac75 replicate. It pairs with `~/fleet_runs/f0_s1044_ttr_mix50` (Kaggle T4 x2,
  E 21, workers 3, seed 1044, mix50): same platform, seed, workers and (very likely) E, only the labels differ. That
  pairing is cleaner than the f01 probe vs J1 (E 17 vs 24, T4 vs V100). Report-only; it never changes the registered gate.
- **Only code change vs the f01 kit:** one plan line in `nb_src/c01_lib.py`
  (`NB_PLAN = dict(kernel=..., trainers=[(fold, seed), (fold, seed)])`) feeding a per-trainer seed into the run names
  and `--seed`; and the full budget below. `build.py` writes that line per notebook; nothing else differs between A and B.
- **Use only if the frac75 gate (fold 0 s1043 from the f01 run) PASSED.** On FAIL these are a record, not members.

### Budget (the full proven budget of rsna-train-ttr-f34 / -s1044-f01 / -s1044-f23)

Training is planned to END by **10.3 h**; hard wall **11.4 h**. Measured basis (Kaggle T4 x2 receipts on the box:
`~/fleet_runs/kaggle_f34_run_receipt.json`, `kaggle_s1044_f01_run_receipt.json`, `kaggle_s1044_f23_run_receipt.json`):
setup + store + checks + probe end at 0.139-0.144 h; 0.78-0.81 s/it; 27.4-28.3 min per epoch with both evals; the
slowest run (f34) overran its epoch estimate by 5.6 % (session 10.74 h); finalize < 1 min.

| per notebook | expected | bound |
|---|---|---|
| E (epochs) | **21** at 0.79-0.83 s/it (s1044 f01/f23 got 21); **22** at <= ~0.79 s/it (f34 got 22 at 0.78); 20 at 0.83-0.87; WARNING below 16 | min(24, floor((10.3 h - elapsed - 2 min) / (1733 x s/it x 1.21))) |
| `--save_epochs` | (E-2)-E (last 3 epochs: ckpt + OOF) | |
| training ends | T0 + 9.96 h (E 21 at 0.80) to 10.17 h (E 22 at 0.78); ~10.5-10.7 h with an f34-like overrun | hard wall T0 + 11.4 h, then quiet-wait <= 2 min + kill grace <= 1 min + finalize |
| session ends | ~T0 + 10.0-10.7 h | < T0 + 12 h = the push `--timeout 43200` |
| GPU quota | ~10.0-10.8 h per notebook, ~20-21.5 h for both | 24 h for both (2 x 12 h cap), so the teammate needs **>= 24 h** |
| outputs | ~1.8 GB per notebook | |

- **Push with `kaggle kernels push -p <dir> --timeout 43200`** (12 h, the Kaggle maximum; changed from 41400 on 10-07 21:40 for margin). On 10-07 a session with no `--timeout` was
  refused when the account had < ~12 h free. The cap does not bind on the normal path (ends ~10.0-10.7 h).
- **Margin on the wall path:** if the wall ever fires (epochs > ~14 % slower than the probe; worst seen +5.6 %), the
  session ends at ~11.4 h + up to ~4 min, i.e. ~31 min inside the 12 h cap (the dry test checked it against the older, tighter 11.5 h cap). If you want more slack, set `WALL_H` default to 11.3 in `nb_src/c01_lib.py`, rebuild, re-run `kt_dry.py`.
- E ~21 equals the box Q2 recipe (21 ep). The f01 run's folds 0/1 are E ~17 (K5_MIN_EPOCHS=14 covers both).

```
KT  = C:\Users\eeshs\Downloads\rsna-knee\session_2026-10-07\kaggle_frac75b
  nb_src\ build.py                    shared notebook source + builder (A and B differ only in the NB_PLAN line)
  kernel\<slug>\                      kernel folders, id eeshsaxena/... (only if you ever run them yourself)
  kernel_teammate\<slug>\             the same notebooks, id YOUR_KAGGLE_USERNAME/... (in the teammate pack)
  verify_outputs.py                   checks downloaded outputs against run_receipt.json
  box\queue_lines_f234.txt            the exact live Q2 f2/f3/f4 queue lines (restore only, sec 7)
  dry\                                box dry tests (box copies: ~/kt_frac75b/dry_src/, logs ~/kt_frac75b/*.log)
Teammate pack: Downloads\rsna-knee\friend2_frac75b_pack\ (+ .zip): README_TEAMMATE.md, TODO.md, this runbook, 2 notebooks, cli\
Inputs (all LIVE, unchanged): datasets eeshsaxena/rsna-knee-fleet-train + eeshsaxena/rsna-knee-frac75-labels,
  kernel outputs eeshsaxena/rsna-fleet-prep-s0 + -s1
B = training@192.168.10.30   (ssh -o BatchMode=yes $B '...' < /dev/null)
```

## 1. Before anything is shared

1. **Team + rules.** The teammate must ALREADY be on competition team "Eesh saxena" and have accepted the rules. Share
   nothing before that.
2. **Quota.** `kaggle quota` on the teammate's account must show **>= 24 h** left, and no other GPU session running.
3. **Gate status.** If the frac75 verdict is already **FAIL**, do not hand this out (the runs would not be used).
4. **Box: Q2 f2-f4 not started, no dirs yet:**
   ```bash
   ssh -o BatchMode=yes $B 'ls -ld ~/fleet_runs/f[234]_s1043_ttr_frac75* ~/fleet_runs/f0_s1044_ttr_frac75* 2>&1; cat ~/queue/gates/frac75.verdict 2>&1 | head -n 2; grep -hE "LAUNCH Q2|REFUSED Q2" ~/fleet_runs/queue_runner.log 2>/dev/null | tail -n 4' < /dev/null
   ```
   - Expected now: "No such file" for all four dirs, no verdict file, no `LAUNCH Q2 frac75 f[234]` line.
   - A `LAUNCH Q2 frac75 fN` line or a non-empty `fN_s1043_ttr_frac75`: that fold is already on the V100. Ask Claude for
     a rebuild without it (one `NB_PLAN` line + `python build.py` + `kt_dry.py`).

## 2. Share the inputs and hand over the pack (web UI)

1. Share with the teammate's username, **Can view**: datasets `eeshsaxena/rsna-knee-fleet-train` and
   `eeshsaxena/rsna-knee-frac75-labels`; notebooks `eeshsaxena/rsna-fleet-prep-s0` and `eeshsaxena/rsna-fleet-prep-s1`.
2. Send `Downloads\rsna-knee\friend2_frac75b_pack.zip`.
3. Say **GO** when you want both started (the README tells them to wait for it).

## 3. When both show `running` (= T0, one per notebook)

1. Write down both T0s. Each ends by ~T0 + 10.0-10.7 h, never after T0 + 11.5 h (+ output saving).
2. **Make the EMPTY placeholders right away** (and in any case BEFORE a PASS verdict is written), so the V100 queue never
   trains Q2 f2-f4 itself (the runner marks a line whose out dir exists `#REFUSED`). Fold 0 seed 1044 has no queue line:
   no placeholder.
   ```bash
   ssh -o BatchMode=yes $B 'for F in 2 3 4; do d=~/fleet_runs/f${F}_s1043_ttr_frac75; if [ -e $d ] || [ -e $d.log ]; then echo "EXISTS $d (stop: sec 1 step 4)"; else mkdir $d && echo "placeholder $d"; fi; done' < /dev/null
   ```
   If only one notebook is running, make only its seed-1043 folds (A: 2 3; B: 4).
3. With f1-f4 all refused, no Q2 line runs the gate command any more: write the verdict by hand as in the f01 RUNBOOK
   sec 7b (`F75_SCORER=... frac75_gate.sh`). Q4 then follows the verdict as before.

## 4. Monitor

```powershell
kaggle kernels status TEAMMATE/rsna-train-ttr-frac75-f23
kaggle kernels status TEAMMATE/rsna-train-ttr-frac75-f4x
```

| status | meaning / action |
|---|---|
| `error` at once, **EMPTY log** | Kaggle refused the session (quota, or 2 GPU sessions already running). Do not retry blindly: ask for a `kaggle quota` screenshot. Retry only with >= 12 h left per session and a free GPU slot. Placeholders: sec 7 |
| `error` within ~30 min | an input check failed: the last `[kt ...] ABORT` line says why (table in the f01 RUNBOOK sec 8, same notebook code). Re-run only with >= 12 h left |
| `error` later | unexpected: fetch the log (sec 5) and ask Claude |
| `complete` | sec 5 |

## 5. Fetch and verify (PowerShell, per notebook)

```powershell
$KT = "C:\Users\eeshs\Downloads\rsna-knee\session_2026-10-07\kaggle_frac75b"
foreach ($S in "f23", "f4x") {
  $OUT = "$KT\out_$S"
  kaggle kernels output TEAMMATE/rsna-train-ttr-frac75-$S -p $OUT --page-size 200 --file-pattern '^(?!_[a-z0-9]+_env/)(?!.*epochs/.*\.pt$)'
  python "$KT\verify_outputs.py" $OUT
}
```

- The pattern drops the 12 per-epoch checkpoints per notebook (~1.35 GB) and any vendored `_xxx_env/` dir. It keeps the
  4 best checkpoints, the best and per-epoch OOF csvs, the logs, `run_receipt.json` and `probe/`.
- Expected: `status DONE`, `plan E=21` (or 22) with `save_epochs=19-21` (or 20-22), `0 mismatch, 12 absent`, exit 0.
- Exit 2 = PARTIAL / ABORTED: read the warnings. A fold is usable only if its `.log` ends with `{"raw": ..., "ema": ...}`.
- Exit 1 = a file differs or a ckpt is UNUSABLE: fetch again before anything else.

## 6. Box: copy and verify (Git Bash)

```bash
KT=/c/Users/eeshs/Downloads/rsna-knee/session_2026-10-07/kaggle_frac75b; B=training@192.168.10.30
ssh -o BatchMode=yes $B 'test -e ~/fleet_runs/f0_s1044_ttr_frac75 || mkdir ~/fleet_runs/f0_s1044_ttr_frac75' < /dev/null
for X in "f23 f2_s1043_ttr_frac75 f3_s1043_ttr_frac75" "f4x f4_s1043_ttr_frac75 f0_s1044_ttr_frac75"; do
  set -- $X; S=$1; shift; OUT=$KT/out_$S
  for R in "$@"; do
    ssh -o BatchMode=yes $B "test -d ~/fleet_runs/$R && [ -z \"\$(ls -A ~/fleet_runs/$R)\" ] && [ ! -e ~/fleet_runs/$R.log ]" < /dev/null \
      || { echo "STOP: ~/fleet_runs/$R is not an empty placeholder"; break 2; }
    scp -r "$OUT/$R/"* $B:fleet_runs/$R/ && scp "$OUT/$R.log" $B:fleet_runs/$R.log
  done
  scp "$OUT/run_receipt.json" $B:fleet_runs/kaggle_frac75_${S}_run_receipt.json
  ssh -o BatchMode=yes $B "~/miniconda3/envs/rsna/bin/python ~/kt_frac75b/verify_outputs.py ~/fleet_runs --receipt ~/fleet_runs/kaggle_frac75_${S}_run_receipt.json | tail -n 2" < /dev/null
done
```

- Result: `~/fleet_runs/f{2,3,4}_s1043_ttr_frac75/`, `f0_s1044_ttr_frac75/` (+ `.log`) and
  `kaggle_frac75_{f23,f4x}_run_receipt.json`. These are the names the queue and the collectors use.
- Copy them in whatever the verdict is (that keeps Q2 f2-f4 refused). **Use them only on PASS:**
  - **PASS:** they are the K9 frac75 members for folds 2-4 (E ~21, like the box Q2 recipe); with f0/f1 from the f01 run
    (E ~17) collect with `K5_MIN_EPOCHS=14` (approved 10-06). `f0_s1044_ttr_frac75` is an extra seed for fold 0, and
    its report-only paired read is vs `f0_s1044_ttr_mix50`.
  - **FAIL:** Q2 lines become `#SKIP`; the runs stay in `~/fleet_runs` as a record, not members.
  - **INCONCLUSIVE-NARROW / tie-break pending:** keep them; the verdict from the tie-break decides (f01 RUNBOOK sec 7c).

## 7. If a run fails or is refused

- Remove ONLY that notebook's EMPTY placeholders (`rmdir` refuses non-empty dirs): A `f2 f3`, B `f4`:
  ```bash
  ssh -o BatchMode=yes $B 'for F in 2 3 4; do rmdir ~/fleet_runs/f${F}_s1043_ttr_frac75 2>/dev/null && echo "removed f$F"; done' < /dev/null
  ```
  (edit the fold list to the failed notebook's folds; for B also `rmdir ~/fleet_runs/f0_s1044_ttr_frac75` if you made it)
- **Restore a Q2 line only if the runner already refused it:** `bash ~/queue/qctl.sh show | grep "REFUSED.*frac75 f[234]"`.
  If it did, re-add from `~/kt_frac75b/box/queue_lines_f234.txt` (byte-identical to the live lines), highest fold first:
  ```bash
  ssh -o BatchMode=yes $B 'L=~/kt_frac75b/box/queue_lines_f234.txt; for F in 4 3 2; do bash ~/queue/qctl.sh top "$(grep "^Q2 frac75 f$F " $L)"; done' < /dev/null
  ```
  (only the refused folds; qctl refuses a duplicate of a still-active line, so this is safe either way)

## 8. Never

- Version `eeshsaxena/rsna-knee-fleet-train` or `eeshsaxena/rsna-knee-frac75-labels`.
- Kill a box job, or start Q2 by hand while the placeholders exist.
- Pick a run whose log lacks the closing line.

## 9. Verification done, ALL PASS (ran 2026-10-07 21:14-21:28 IST on the notebooks with sha256 d6175d4a... (A) and 687878f3... (B), box CPU, `CUDA_VISIBLE_DEVICES=` nice 19, under `~/kt_frac75b/` only)

Datasets on the box (`~/kt_frac75b/ds`, `ds_labels`) == the LIVE copies (kt_manifest 14/14 sha256; labels sha256
`5036425c...` == `~/novel/frac75`). Only the tests the changes touch were re-run (the gate scorer and queue tests of the
f01 kit are unaffected).

- `dry_src/kt_dry.py A` (log `~/kt_frac75b/dry_run_A.log`): DRY TEST 80/80 PASS (DRYEXIT 0)
- `dry_src/kt_dry.py B` (log `~/kt_frac75b/dry_run_B.log`): DRY TEST 80/80 PASS (DRYEXIT 0)
  - unit: CFG (slug, trainers, 10.3 / 11.4 h, last-3 saves), E table 0.78 -> 22, 0.80 -> 21, 0.81 -> 21, 0.85 -> 20,
    1.1 -> 15, 0.5 -> 24, 0.7996 s/it plans 9.96 h (= the s1044 f01 receipt), +10 % slower still ends 11.16 h < 11.4 h,
    wall path 41340 s < `--timeout` 41400 s (margin 60 s), per-trainer `--seed` in recipe_args
  - full flow (both mount layouts): ckpt names / args seed + fold per trainer, outputs in `f{F}_s{seed}_ttr_frac75`;
    watchdog (GPU1 trainer ignores SIGTERM -> SIGKILL), default wall vs a T0 11.4 h in the past (session end 11.401 h),
    corrupt tar, wrong contract / counts, CRLF + missing wheel + LF labels, duplicate mounts, 4 labels-file aborts
- `dry_src/kt_dry_g.py A` / `B` (logs `dry_run_A_g.log`, `dry_run_B_g.log`): PHASE G 8/8 + 8/8 PASS (fetch pattern,
  copy into empty placeholders + second copy refused, verify_outputs: complete 0 / watchdog 2 / tampered 1)
- `dry_src/kt_smoke_real.py A` (log `smoke_real_A.log`): SMOKE 22/22 PASS, the real fleet_train.py: fold 2 s1043 E 21
  (rc 0, 244 s) and fold 3 s1043 E 2 (epochs/ save path)
- `dry_src/kt_smoke_real.py B` (log `smoke_real_B.log`): SMOKE 23/23 PASS: fold 0 s1044 E 21 (rc 0, 239 s; ckpt args vs
  `~/fleet_runs/f0_s1044_ttr_mix50` differ only in train_labels, save_epochs and CPU amp) and fold 4 s1043 E 2
