# frac75 Fracture-target probe: pre-registration (written 2026-10-07 ~12:30 IST, BEFORE launch)

Source: `NOVEL_PLAN_10-07.md` sec 2 item 4 and sec 3 Q1/Q2 (gates taken from the plan as written; they are complete, so
the workstream's fallback gate set is reported but does not decide). Tags: **MEASURED**, **MODEL**, **GUESS**.
Nothing here was launched on the GPU. The bars below must not be edited after the probe starts.

## 1. The change (one input file, no code)

- `~/novel/frac75/labels_frac75_logit.csv`, md5 **ad7582b817243f8f22e8bcfdd7e3f84a** (4,407 rows, 12 labels, no NaN, CRLF
  like mix50). Built by `~/novel/frac75/build_frac75.py`, log `~/novel/frac75/build_box.log`.
- = `~/rsna_train/labels_mix50_logit.csv` with ONLY the Fracture column replaced by
  `.25 x steven_v4 + .75 x logit_match(teacher, steven_v4)`. Teacher = `~/rsna_train/teacher_train_probs.csv` (Dread
  widedense, the mix50 teacher). `logit_match` = `SP/build_mixSA.py` MAP=logit (mean/std match in logit space, clip 1e-4).
- **Method proof [MEASURED]:** the same code at w .5 reproduces all 12 mix50 columns, worst max |diff| 3.3e-16 (box),
  1.1e-16 (laptop). The new file equals the perlabel track's earlier independent `~/novel/labels_mix50_frac75.csv` with
  max |diff| 0.0 in every column. The 11 other columns, ids, header and row order are byte-identical to mix50 (the
  file is written by copying each mix50 line and replacing only the last field).
- Fracture target vs experts [MEASURED, masked]: gold58 v4 .7931 / mix50 .8431 / **frac75 .8889** / teacher .8917;
  gold74 v4 .8364 / mix50 .8818 / **frac75 .9145**. 12-label target macro gold58 .9262 -> .9300, gold74 .9354 -> .9381.
  Caveat (verify_perlabel 2.2): the teacher was epoch-selected on gold58, so the gold58 label lift is optimistic and not
  significant (CI [-.001, +.108]). The probe student never sees gold and is selected on val vs v4: it is the clean test.
- Prevalence [MEASURED, 4,333 training-pool studies]: share >= .5 v4 .067 / mix50 .074 / **frac75 .084**; mean target
  .181 / .166 / .158; experts .310 (gold58 18/58), .306 (gold74 22/72). frac75 re-ranks Fracture; it does NOT fix the
  under-call (the teacher is logit-matched onto v4's marginal by construction).

## 2. The probe (paired with J1)

- Trainer `~/fast_build/code2/fleet_train.py` (md5 3e62434a...), `--grad_ckpt 1` = the `~/bigjump_build/code` trainer
  that ran J1, bit for bit (verify_perlabel row 14). model_sha f141aeed136dfc01 (unchanged; labels never enter it).
- argv = J1's (`~/auto_j1.sh` l.30) with only `--labels` and `--out` changed, plus explicit `--grad_ckpt 1`:
  defined once in `~/novel/frac75/argv_frac75.sh`, printed by `bash ~/novel/frac75/launch_frac75.sh probe`:

```
cd ~/fast_build/code2 && ~/miniconda3/envs/rsna/bin/python fleet_train.py --model ttr --aug heavy --train_slices 16 \
  --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs 24 --seed 1043 --save_epochs 6-24 --workers 10 \
  --store ~/fleet_store --labels ~/novel/frac75/labels_frac75_logit.csv --folds ~/rsna_train/folds5.csv \
  --exclude ~/rsna_train/gold74_eval_set.csv --eval_labels ~/rsna_train/llm_labels_v4_blend.csv \
  --gold ~/rsna_train/gold74_eval_set.csv --fold 0 --out ~/fleet_runs/f0_s1043_ttr_frac75 --grad_ckpt 1 \
  > ~/fleet_runs/f0_s1043_ttr_frac75.log 2>&1
```

- Why the pairing is tight: the label file has the same 4,407 ids in the same order, so with seed 1043 the train/val
  split, shuffle order and augmentation draws are J1's; only the Fracture targets differ (up to GPU non-determinism,
  cudnn.benchmark). The scorer refuses a probe whose checkpoint args differ from J1's in anything but `train_labels`.
- Cost [MODEL]: J1 logged ~11.2-11.5 min/epoch with both evals at gc1, so ~4.6-4.8 h (+-10% thermal).
- Launch only on a free V100 (after the TTR288 chain prints `chain done`, ~10-08 10:00), by ONE queue runner or
  `LAUNCH=1 bash ~/novel/frac75/launch_frac75.sh probe` (refuses, never waits, when the GPU / PAUSE_CHAIN / chain is busy).

## 3. Scoring convention (pinned)

- **Checkpoint:** the trainer's best RAW checkpoint by val macro AUC vs steven_v4 (`f0_s1043_raw_*_best.pt` and
  `oof_f0_s1043_raw.csv`, the same selection that produced J1's cv8835). EMA is not scored. Gold never selects.
- **Gold convention: MASKED** (NaN expert cells dropped per label), as `fleet_train.macro_auc` (T9), `tta_bench` and
  `gold_infer5.py` score. gold74 has 18 NaN cells (Fracture 2), so gold74 Fracture has n 72, 22 positives.
- **Gold predictions:** `~/novel/gold_infer5.py --passes A` (fp32 CPU, the trainer's eval input). J1's are
  `~/jump2/gold74_preds/f0_s1043_ttr_mix50__raw.csv` (identical, max |diff| 0.0, to `~/novel/gold74_preds/..__raw__A.csv`;
  CPU gold74 .9092 vs the trainer log .9091).
- **Scorer:** `~/novel/frac75/score_frac75.py` (CPU). Output `~/novel/frac75/out/gate_f0_s1043_ttr_frac75.{txt,json}`.
  Exit 0 PASS / 1 FAIL / 3 not finished / 2 integrity error (ckpt args differ from J1's beyond train_labels, wrong
  label md5, OOF not the selected epoch, stale gold preds).

## 4. Gates (all three must pass)

| gate | measured on the probe | bar | J1 (same convention) [MEASURED] |
|---|---|---|---|
| **G1 no harm** | mean OOF AUC vs steven_v4 over the 11 NON-Fracture labels (Fracture vs v4 is penalised by construction) | **>= .8774** | .8803 (plan text .8804; bar = J1 - ~.003, fold-0 seed noise .0028) |
| **G2 mechanism** | gold74 Fracture AUC, masked | **>= .912** | .8936 (bar = J1 .894 + .018; 9 TTR224 members mean .8951, SD .014, max .920) |
| **G3 guard** | gold74 12-label macro AUC, masked | **>= .9062** | .9092 (bar = J1 - .003; member SD .0020) |

Report only (never decides): gold58 Fracture sign vs J1 (plan: "report the gold58 Fracture sign"), paired study
bootstraps of the Fracture deltas, the trainer-log parity line, and the workstream fallback set (Fb1 gold74 Fracture >=
J1 + .01, Fb2 gold58 macro >= J1 - .003, Fb3 OOF 12-label macro vs v4 >= J1 - .003).

Selftest [MEASURED]: the scorer run on J1 itself as a "null probe" gives G1 .8803 PASS, G2 .8936 FAIL, G3 .9092 PASS ->
FAIL (G2), as it must (`~/novel/frac75/out/gate_selftest_J1_as_null_probe.txt`).

Power [MODEL, normal approx. on the 9-member masked spread, center .8951, SD .014]: P(G2 pass) ~.11 at a true
Fracture effect of 0, ~.31 at +.01, ~.59 at +.02, ~.83 at +.03. Fold-0 members sit at or below the member mean, so
realistic fold-0 power is lower (verifier: ~.45-.55 at +.02). Joint null pass with G1 and G3 ~.08-.10. The decision
rests on mechanism + no harm, not on a significant win; the LB cannot resolve the expected +.001 either.

## 5. Decision

- **PASS (G1 and G2 and G3):** train folds 1-4 (`bash ~/novel/frac75/launch_frac75.sh folds` prints them; it refuses
  to launch unless the gate json says PASS): 21 epochs, `--grad_ckpt 0` (one early-OOM fallback to 1), seed 1043, out
  `~/fleet_runs/f{1..4}_s1043_ttr_frac75`. ~3.7-4.0 h each [MODEL: ~10.5 min/epoch with evals at gc0], ~15-16 h total.
  Then K9 (plan sec 4): the K-builder needs per-set `train_labels` (build_k7.py l.79 asserts mix50) and a labels tag in
  the member key (the ckpt names equal set A's `f{F}_s1043_raw_ttr224-...`). Ship only under the plan's K9 rule.
- **FAIL (any of G1-G3):** close target re-mixing. Fracture / Synovitis then wait on the image-aware relabel (Eesh's
  API decision). Do not re-run with another seed or a different bar.
- A crash, OOM, or a run that ends before epoch 24 is NOT a FAIL: it is "not scorable" (exit 3); re-launch once, same argv.

## 6. Validation done before launch (CPU only, 10-07 12:20-12:42 IST) [MEASURED]

- **argv vs the real trainer** (`dry_run.sh`, logs `~/novel/frac75/dry/`): the argv arrays of `argv_frac75.sh`, verbatim,
  with only `--store dry/store --out dry/... --device cpu --min_cover 0 --time_epochs 1` appended (31 symlinked
  studies: 16 train, 6+6 val, 3 gold). Probe argv (f0, gc1, 24 ep): rc 0, 5.7 min, RSS 4.3 GB; ckpt
  `f0_s1043_raw_ttr224-320x24_cv5928_best.pt` + `oof_f0_s1043_raw.csv`; ckpt args train_labels
  `labels_frac75_logit.csv`, model_sha f141aeed136dfc01, epochs 24, save_epochs 6-24, no grad_ckpt key; vs J1's ckpt
  args only `train_labels`, `amp` (fp32 on CPU) and `decode_probe` (tiny store) differ; state_dict keys and shapes
  identical. Folds argv (f1, gc0, 21 ep): rc 0, 4.8 min, prints `FB-1 opt-in: grad_ckpt 0`, args grad_ckpt False,
  epochs 21, save_epochs 6-21. `~/fast_build/code2` md5s and listing (incl. `__pycache__`) unchanged before/after.
  The live TTR288 fold-1 trainer held 0.45-0.46 s/it throughout.
- **Scorer:** J1-as-null-probe -> FAIL (G2) exit 1 (correct); synthetic PASS (J1 + Fracture preds pulled toward the
  experts) -> PASS in dry mode; the trainer-log parity check correctly REFUSED the synthetic preds outside dry mode;
  unfinished run -> exit 3; launch `folds` without a real PASS json -> refused (exit 4); busy guard -> BUSY.
  End to end on the dry checkpoint (args amp re-tagged fp16 in a deleted TEST copy, since `tta_bench.load_member`
  asserts fp16 training): gold_infer5 74 studies in 574 s at 8 threads, then G1-G3 printed. Dry results are tagged
  `DRY_` and write `"pass_": false`, so they can never unlock the folds.
- **Verify pass (10-07 ~13:00, CPU only; bars, argv, labels and scorer UNCHANGED):** labels re-derived independently
  (Fracture max |diff| 1.1e-16, 11 columns + header + ids byte-identical, gold74 rows are fold -1 and --exclude'd);
  J1's decode_probe / jpeg_fp / contract equal those of 3 later real-store runs, so the real probe should pair cleanly;
  scorer adversarial suite 16/16 (`~/novel/frac75_verify/test_score.sh`). `launch_frac75.sh` hardened (original kept as
  `launch_frac75.sh.orig_1243`): it now also refuses while any `python ... fleet_train.py` is alive (closes the
  pre-CUDA window next to a just-started trainer), while `~/queue/queue_runner.sh` is alive (the queue is then THE
  launcher), and after the 2026-10-20 12:00 IST freeze; it skips a fold whose `.log` exists (never truncates another
  launcher's log); `folds` also needs the gate json's `"run"` == the real probe dir. Sandbox suite 23/23
  (`~/novel/frac75_verify/test_launch.sh`; the original fails 5 of the new checks).
