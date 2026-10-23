# frac75 probe on Kaggle: addendum to `~/novel/frac75/PREREG.md`

Written 2026-10-07 ~14:40 IST by Claude, BEFORE the Kaggle run (`rsna-train-ttr-frac75-f01`) is pushed. Box copy:
`~/kt_frac75/PREREG_KAGGLE_ADDENDUM.md`. `PREREG.md` itself is not edited: bars G1-G3, J1 references, the MASKED gold
convention, best-RAW checkpoint selection and gold_infer5 pass A all stay exactly as registered. This file only says how
a fold-0 probe trained on Kaggle (instead of the V100 Q1 job) is scored and read. No em dashes.
Tags: MEASURED, MODEL, GUESS.

## 1. What differs from the registered box probe

| item | registered box probe (PREREG sec 2) | Kaggle probe |
|---|---|---|
| trainer, argv flags, seed 1043, folds5, gold74 excluded, steven_v4 eval, labels md5 `ad7582b8...` | J1's | identical (notebook asserts code == box `~/bigjump_build/code`, labels sha256 baked, only-Fracture-differs re-checked) |
| `--epochs` | 24 | E sized by the notebook to end training by 8.4 h: **17 expected (18 if the probe measures <= ~0.785 s/it)** |
| `--save_epochs` | 6-24 | (E-2)-E (the last 3 epochs) |
| GPU, loader workers | V100, 10 | one T4 per fold, 3 |
| ckpt args vs J1 | only `train_labels` | `train_labels`, `epochs`, `save_epochs` (MEASURED on the s1044 Kaggle run: its args differ from J1's only in epochs, save_epochs and seed; decode_probe, contract, amp fp16 equal) |
| fold 1 | Q2 on the V100 after PASS (21 ep, gc0) | trained in the same session (E epochs) |

## 2. What this does to the gate's validity

- Still paired: same split, same seed and shuffle order, same labels except Fracture, same trainer bytes.
- No longer paired:
  - augmentation draws (3 vs 10 loader workers change the per-batch worker RNG streams);
  - fp16 numerics (T4 vs V100);
  - the LR schedule (cosine compressed to E epochs).
  So probe minus J1 now carries a schedule term and seed-like noise on top of the label effect.
- Size of the schedule term:
  - MEASURED: E 21 is about equal to E 24. The s1044 Kaggle runs at E 21 vs the s1043 box runs at 24 give raw val
    +.0028 / -.0009 / +.0003 on folds 0 / 1 / 2 (mean +.0007), and that includes a seed change.
  - E 17 is outside the measured range. GUESS: 0 to -.003 on OOF and gold macro, most likely against the probe
    (fewer total steps).
- Consequences:
  - A PASS on all three registered bars stays a PASS: the deviations are not expected to favour the probe.
  - A gate missed by a wide margin stays a FAIL.
  - A gate missed by less than .003 cannot be told apart from the schedule term. Note that .003 is also the tolerance
    built into G1 and G3.
- Considered and rejected: `--epochs 24 --stop_epoch E`. It would pair epoch for epoch with J1's saved epochs 6-24,
  but it leaves both fold-0/1 models un-annealed (LR still ~20% of peak at 17/24). That makes them useless as K9
  members, and delivering usable fold 0/1 members is half the point of this session.

## 3. Scoring

- `~/kt_frac75/box/score_frac75_kaggle.py` runs the pinned `~/novel/frac75/score_frac75.py` (sha256 `f19ddd8c...`)
  unchanged, with one patch:
  - the pairing allowlist gets exactly `epochs` and `save_epochs` (any other arg difference is still an integrity
    error, exit 2);
  - plus two checks: `14 <= E <= 24` and `save_epochs == (E-2)-E`.
- Output paths, json fields and exit codes are the registered ones. So `~/queue/gates/frac75_gate.sh` can use it via
  `F75_SCORER=~/kt_frac75/box/score_frac75_kaggle.py`.
- Report only (never decides), appended to the same txt/json:
  1. Schedule-end read: OOF vs v4, mean of the last 3 epochs, probe epochs E-2..E vs J1 epochs 22-24. This is
     selection-free, and both runs sit at the end of their own cosine.
  2. The trainer-logged gold74 / gold58 / val curves at matched schedule fraction.
  3. A gold74 internal control: Fracture delta minus the 11-label non-Fracture delta. A schedule or platform shift
     moves every label; frac75 moves Fracture only.

## 4. Decision (the "KAGGLE ADDENDUM READ" line)

| read | condition | action |
|---|---|---|
| **PASS** | G1 >= .8774, G2 >= .912, G3 >= .9062 | write the queue verdict PASS; the V100 trains folds 2-4 (Q2, 21 ep, gc0); the Kaggle f0/f1 are K9 members |
| **FAIL** | at least one gate below its bar by >= .003 | write the queue verdict FAIL; close target re-mixing as registered; no re-run with another seed or bar |
| **INCONCLUSIVE-NARROW** | every failing gate is below its bar by < .003 | write NO verdict from Kaggle. Eesh decides: tie-break with the registered 24-epoch box probe (J1 argv, gc1, ~4.6-4.8 h V100, out `f0_s1043_ttr_frac75_b24`, scored by the ORIGINAL scorer, its PASS / FAIL decides), or accept FAIL |

- E 14-15 (< 16 warning): same table, but prefer the tie-break if anything is narrow.
- E < 14: not read as the probe (scorer exit 2). The registered box probe Q1 decides (out `f0_s1043_ttr_frac75_b24`).
- A Kaggle run with no closing line for fold 0 (watchdog cut, PARTIAL): not scorable (exit 3), as registered.
- Power is unchanged in spirit:
  - PREREG sec 4 puts P(G2 pass) at ~.45-.63 at a true +.02 Fracture effect.
  - The extra seed-like noise of the Kaggle pairing widens that a little.
  - The decision still rests on mechanism + no harm, not on a significant win.
