# PROBE4_PREREG: OAI T123 + pseudo-label fill (T123PL) probe f0

Written 2026-10-11 ~15:45 IST by Claude (main session), BEFORE the PL label tables or the T123PL union were built and before
any T123PL training. Nothing below may be edited after the probe starts; append results in a separate PROBE4_RESULTS.md
(keeps the sha valid). No em dashes. Box copy: ~/oai_build/PROBE4_PREREG.md (same bytes; sha256 in PROBE4_PREREG.sha256).

## Why (disclosure of what was seen first)

- T123 probe f0 (PROBE3_RESULTS.md) vs the no-OAI control D f0, OOF vs llm_labels_v4_blend, best raw ckpt:
  - the 9 labels OAI supervises all improve (ACL +.008, menisci +.003 / +.005, OA +.005 to +.012, Eff +.011);
  - the 3 labels OAI leaves NaN do not: MCL +.004, **Contusion -.0096, Fracture -.0061**;
  - gold58 the same way: **MCL -.030, Fracture -.022, Contusion -.007** (58 studies, noisy).
- So OAI rows with NaN on MCL / Contusion / Fracture dilute those heads (2,581 OAI rows train every other head, none of
  these three). Filling them with soft targets from an RSNA-only teacher should stop that.
- Teacher sanity (10-11 log): set D on OAI V00; mean teacher prob MCL .08, Contusion .28, Fracture .09.
- K14 (the 5-fold T123 set vs FN11) has NOT been read; T123 f1 is training (ep 15 cv .8921 seen at 14:55).

## Pair (one change only)

- **Probe:** `f0_s1043_ttr_frac75oaiT123PL`, the T123 probe's EXACT argv (17 ep, save 15-17, gc1, seed 1043, fold 0, 224)
  with only `--store ~/union_store_t123pl`, `--labels ~/oai_build/union_t123pl_f0/labels_frac75oai_T123PLf0.csv`,
  `--folds ~/oai_build/union_t123pl_f0/folds5_oai.csv`, `--out` changed.
- **Labels:** T123v00 OAI rows unchanged on the 9 MOAKS labels; MCL / Contusion / Fracture = raw soft probability of the
  **same-fold** set D member `f{F}_s1043_ttr_frac75` (raw best, pass A, CPU teacher run 10-11). Fold F uses only the
  member that never trained on RSNA fold F, so fold F's OOF stays leak-free. RSNA rows byte-identical (make_union3.py
  asserts it, as make_union2 did).
- **Reference:** the T123 probe `f0_s1043_ttr_frac75oaiT123v00`, numbers frozen below from
  `gate2_REPORT_T12_f0_s1043_ttr_frac75oaiT123v00.json`.

## Frozen T123 probe numbers (867 RSNA fold-0 val rows; gold58 masked macro)

| group | value |
|---|---|
| P3 OOF mean {MCL, Contusion, Fracture} | .868693 |
| R9 OOF mean (the other 9 labels) | .893982 |
| oof12 | .887660 |
| gold58 12-label macro | .919434 |
| gold58 P3 mean (report only) | .924135 |

## Gate (scorer: `score_oai_probe2.py --tier T12 --report_only --expect_labels labels_frac75oai_T123PLf0.csv`, then `gate_pl4.py`)

| gate | rule | bar |
|---|---|---|
| G1 mechanism (OOF) | P3 OOF mean >= .868693 + .003 | >= .871693 |
| G2 overall (OOF) | oof12 >= .887660 - .0003 (declared now: the PROBE3 .0001 miss was noise) | >= .887360 |
| G3 no harm (OOF) | R9 OOF mean >= .893982 - .002 | >= .891982 |
| G4 guard (gold58) | 12-label macro >= .919434 - .005 | >= .914434 |

**PASS = G1 and G2 and G3 and G4.** Verdict file `~/oai_build/out/pl4.verdict`.
- **PASS:** T123PL folds 1-4 run next (21 ep, gc0, seed 1043, out `f{F}_s1043_ttr_frac75oaiT123PL`, labels from
  `union_t123pl_f{F}`), ahead of P2b. The 5-fold set then needs its own pre-registered fleet gate (vs whatever FN11 / K14
  stands at) before any submission.
- **FAIL:** stop T123PL; P2b and the rest of the queue continue.
- **Integrity (exit 2, no verdict):** the scorer's integrity checks fail, the report is not this probe, or this file's
  sha changed.

Reported, never gating: per-label OOF and gold58, gold74, P3 gold58 mean, schedule-end means.

## Compliance

OAI data never leaves the box. Only OAI-trained WEIGHTS may go to a private Kaggle dataset, after the OAI-clean scan.
OAI-trained kernels go to the N pick only.
