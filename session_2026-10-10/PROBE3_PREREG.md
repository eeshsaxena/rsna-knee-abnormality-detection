# PROBE3_PREREG: OAI T123v00 probe f0 (does adding the MOAKS meniscus + ACL labels fix what T12 broke?)

Written 2026-10-10 ~20:55 IST by Claude (main session), BEFORE the T123v00 union was built and before any T123 training.
Nothing below may be edited after the probe starts; append a dated RESULTS section. No em dashes.
Box copy: ~/oai_build/PROBE3_PREREG.md (same bytes; sha256 in PROBE3_PREREG.sha256).

## Why (disclosure of what was seen first)

**K12 result.** The 5-fold OAI T12v00 224 set failed its fleet gate on 10-10 at 19:36 (PREREG_K12 RESULTS): OOF +.0005,
gold58 -.0018.

**The pattern.** Its own f0 probe (gate2_T12_f0_s1043_ttr_frac75oaiT12v00) and the 5-fold gate show the OAI set:
- HELPS the labels OAI supervises: PF OA, Lateral OA, Medial OA, Effusion, Baker's;
- HURTS several of the labels it leaves NaN:
  - ACL OOF -.010 at f0;
  - gold58 Lateral Meniscus, ACL, MCL and Fracture all down.

**The new label table.** `labels_v/oai_labels_T123_v00.csv` (built 10-09, never trained on) ADDS real MOAKS targets:

| label | knees |
|---|---|
| ACL | 1,654 |
| Medial Meniscus | 2,399 |
| Lateral Meniscus | 2,399 |

MCL, Contusion and Fracture stay NaN.

**External evidence (forum 748011, 15 submissions).** OOF vs LLM labels tracks the public LB at r .87; gold58 tracks it
at r .66. So this gate is OOF-led, with gold58 as a loose guard.

## Pair (one change only)

- **Probe:** `f0_s1043_ttr_frac75oaiT123v00`.
  - It uses the T12 probe's EXACT argv: 17 ep, save 15-17, gc1, seed 1043, fold 0, 224.
  - Only `--store` / `--labels` / `--folds` change, pointing at the T123v00 union (`~/union_store_t123v00`,
    `~/oai_build/union_t123v00`).
- **Reference:** the T12 probe `f0_s1043_ttr_frac75oaiT12v00`. Its numbers are frozen below, from gate2_T12_...json
  (read 20:40).

## Frozen T12 probe numbers

OOF vs `llm_labels_v4_blend`, best raw ckpt, 867 RSNA fold-0 val rows.

| group | label | OOF |
|---|---|---|
| T3 | ACL | .870314 |
| T3 | Medial Meniscus | .922917 |
| T3 | Lateral Meniscus | .918745 |
| | **T3 mean** | **.903992** |
| S12 | PF OA | .896068 |
| S12 | Lateral OA | .866835 |
| S12 | Synovitis | .854609 |
| S12 | Medial OA | .906608 |
| S12 | Effusion | .857667 |
| S12 | Baker's | .936394 |
| | **S12 mean** | **.886364** |
| all | oof12 | .887764 |
| all | gold58 12-label macro | .914173 |

## Gate

Procedure: run `score_oai_probe2.py --tier T12 --report_only` on the probe (vs control D f0), then `~/oai_build/gate_t123.py`.

| gate | rule | bar |
|---|---|---|
| G1 mechanism (OOF) | mean OOF over {ACL, Medial Meniscus, Lateral Meniscus} >= .903992 + .002 | >= .905992 |
| G2 overall (OOF) | oof12 >= the T12 probe | >= .887764 |
| G3 keep T12 gains (OOF) | mean OOF over the 6 T12 labels >= .886364 - .002 | >= .884364 |
| G4 guard (gold58) | 12-label macro >= control D f0 .9163 - .005 (the bar every OAI probe used) | >= .9113 |

**PASS = G1 and G2 and G3 and G4.** The verdict file is `~/oai_build/out/t123.verdict` (PASS / FAIL).
- **PASS:** T123v00 folds 1-4 run next (21 ep, gc0, seed 1043, out `f{F}_s1043_ttr_frac75oaiT123v00`), queued AHEAD of
  the 288 T12v00 folds. The 5-fold set then gets its own pre-registered OOF-led fleet gate before any submission.
- **FAIL:** stop T123. Nothing else changes; the 288 T12v00 chain continues as queued.
- **Integrity (no verdict, exit 2):** ckpt args differ from the T12 probe beyond store / labels / folds / out, or an OAI
  id appears in an OOF file or a checkpoint.

## Compliance

- OAI data never leaves the box.
- Only OAI-trained WEIGHTS may go to a private Kaggle dataset, and only after a K12-OAI-CLEAN-style scan.
- OAI-trained kernels go to the N pick only.
