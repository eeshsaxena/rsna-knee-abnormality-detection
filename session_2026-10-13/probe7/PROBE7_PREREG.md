# PROBE7_PREREG: T123 labels on ALL OAI visits (T123allv) probe f0

Written 2026-10-13 ~11:15 IST by Claude, BEFORE any T123allv training. Eesh 10-13: "IF FOLD 1 is not promising switch to
the next model". Results go to PROBE7_RESULTS.md. No em dashes. Box copy ~/oai_build/PROBE7_PREREG.md (+ .sha256).

## Why (seen first)
- PROBE4 (PL) fold 1 is NOT promising: oof12 .8932 vs T123 f1 .8935 (-.0003); P3 (MCL / Contusion / Fracture) -.0007; the
  2-member blend mean(DT, PL) .8944 vs mean(DT, D) .8947. Fold 0 had +.0020 (likely partly noise). PL folds 2-4 stopped.
- The OAI lever that DID work is labels on more knees: T12 -> T123 fixed ACL / menisci (+.0039 vs D on f0). The follow-up
  visits (V01, V03, V05, V06) are downloaded and converted: 9,126 OAI rows from 3,218 knees vs 2,581 at V00. Union
  `~/union_store_t123allv` + `~/oai_build/union_t123allv` built 11:05 (ALL ASSERTIONS PASSED; folds md5 = T12allv's;
  RSNA rows byte-identical). Prior: nartaa's 12-month arm "unsuccessful", so this is a genuine test, not a sure gain.

## Probe AV f0
The T123 probe argv with ONLY: `--store ~/union_store_t123allv --labels ~/oai_build/union_t123allv/labels_frac75oai_T123allv.csv
--folds ~/oai_build/union_t123allv/folds5_oai.csv`, `--epochs 9 --save_epochs 7-9` (step-matched: 2.1x rows per epoch, so 9
epochs ~= the 17 V00 epochs in optimizer steps), out `f0_s1043_ttr_frac75oaiT123allv`. Seed 1043, gc1, fold 0, TTR224.

## Gate (gate_av.py; fold-0 RSNA val rows, OOF vs llm_labels_v4_blend, best raw ckpt; gold58 pass A CPU)
| gate | rule |
|---|---|
| G1 better than T123 V00 | OOF12(AV) >= OOF12(DT f0) + .0015 = **.889160** |
| G2 gold58 guard | gold58(AV) >= gold58(DT f0) - .005 = **.914434** |
PASS -> AV folds 1-4 (11 epochs each, step-matched to 21, gc0) next; then a fleet / HF gate before any submission.
FAIL -> the queue continues (P2b T12allv line is SKIPPED as superseded; nothing else).
