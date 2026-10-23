# PREREG K14: does the OAI T123v00 224 set (DT) REPLACE set D (or join it) in the FN11 fleet?

Written 2026-10-11 ~10:40 IST by Claude (main session), BEFORE any T123 fold 1-4 checkpoint, log line, OOF or gold
prediction exists (f1 launched 10:27; f2-f4 follow, ~25 h total). Do not edit after the first fold 1-4 read; append a
dated RESULTS section. No em dashes.

## Disclosure (what was seen before writing)

- K12 (D-OAI = T12v00 set) fleet gate 10-10: R dOOF +.00052 / d58 -.0018; A dOOF +.00040 / d58 -.0007. NO K12.
- T123 f0 probe: cv .88766 (D f0 .883777), gold58 single-member .9194 (D f0 .9163); PROBE3 gate FAIL by .0001 on oof12
  vs the T12 probe; GPU override approved (PROBE3_RESULTS.md).
- Forum 748011 (colum2131, 15 submissions): OOF vs LLM labels tracks the public LB at r .87, gold58 at r .66. This gate is
  therefore OOF-led, BUT the gold58 guard is set so that K12 R (-.0018) would still FAIL it: the bars are not loosened to
  admit the result already seen.

## Candidate and base (same as PREREG_K12 sec 1, with DT in place of D-OAI)

- **DT** = `f{F}_s1043_ttr_frac75oaiT123v00`, F = 0..4. f0 = the 17-epoch probe checkpoint as is (the K12 f0 precedent);
  f1-f4 = 21 epochs.
- **B** = FN11 fleet = C + D + E + E2 (20 members).
- **R** = C + DT + E + E2 (20). **A** = C + D + DT + E + E2 (25).

## Per-fold acceptance

All PREREG_K12 sec 3 rules, with these substitutions:
- train_labels `labels_frac75oai_T123v00.csv`;
- the K12-OAI-CLEAN scan;
- the sanity bar DT_F >= D_F - .005;
- min epochs 14.

K14 needs all 5 folds accepted.

## Gate (RSNA rows only; the PREREG_K12 sec 4 machinery, `k12_gate.py` logic with DT)

Metrics:
- **G1 OOF**: fold-matched OOF macro AUC vs llm_labels_v4_blend on the 4,333 RSNA val rows, equal member mean of RAW
  probabilities.
- **G2 gold58**: pass A CPU fp32 predictions, equal mean of all fleet members, masked macro AUC on the original gold58.

Pass rules:

| variant | passes iff |
|---|---|
| R | dOOF(R - B) >= **+.0005** AND d58(R - B) >= **-.0014** (about -1 paired SE; K12 R at -.0018 would FAIL) |
| A | dOOF(A - B) >= **+.0005** AND d58(A - B) >= **-.0014** |

**VERDICT:**
1. Only R passes -> **K14 = R**.
2. Both pass -> **R**, unless OOF(A) - OOF(R) >= +.0003 AND AUC58(A) - AUC58(R) >= 0, in which case **A**.
3. Only A passes -> **A**.
4. Neither passes -> **no K14**; FN11 stays.

Reported, never gating: gold74, the paired bootstrap SE, per-label deltas, DT vs D per fold.

## If K14 passes

Same build pattern as K12 (`K12_RESULT.md` sec 3 / 5):
- a weights-only dataset `rsna-knee-cnxt-fleet-k14w`, after the OAI-clean scan;
- kernels attach k11w + k14w + N;
- F .40 / N .60 and N70 twins;
- box A0 chains;
- commit gates; submission on PASS.

OAI-trained, so N pick only.
