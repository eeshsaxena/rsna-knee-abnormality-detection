# K14 RESULTS (PREREG_K14.md sha 98bcedb8, unchanged), recorded 2026-10-12 11:50 IST

**VERDICT: K14 = R (REPLACE set D by the OAI T123v00 224 set DT).** Gate run 10:39-11:10 on box CPU; report
`k14_gate_20261012_1110.txt` (copy here; box `~/k14_test/gold/`).

| variant | dOOF vs FN11 (bar >= +.0005) | d58 vs FN11 (bar >= -.0014) | result |
|---|---|---|---|
| R = C + DT + E + E2 | **+.00106** (OOF .89276 vs .89171, 4,333 RSNA rows) | **-.0007** (SE .0013) | PASS |
| A = C + D + DT + E + E2 | +.00082 | +.0006 (SE .0014) | PASS |

Both pass; A is picked over R only if OOF(A) - OOF(R) >= +.0003 AND gold58(A) - gold58(R) >= 0: OOF -.00023 -> rule says **R**.

Report-only: DT vs D per fold OOF +.0039 / +.0025 / +.0013 / +.0048 / -.0011; single-member gold58 DT vs D +.0031 / +.0041 /
-.0016 / +.0015 / +.0068; set DO gold58 .9235 vs D .9214 (+.0021, SE .0037); gold74 R - B +.0004, A - B +.0013;
Spearman DO~D .957 (OOF). Per label R - B OOF: ACL +.0026, PF OA +.0032, Lat Men +.0017, Lat OA +.0018; Contusion / Fracture -.0002.
K14CLEAN PASS 5/5 (no OAI id in any ckpt / OOF); collect FINAL 10:38.

Next (PREREG_K14 "If K14 passes"): weights-only dataset `rsna-knee-cnxt-fleet-k14w` (after the OAI-clean scan, already PASS),
then kernels. Context 10-12: the submission vehicle is now HF (H + our fleet); HF with F = DT (instead of D) needs its own
prereg after the HF20 / HF30 reads. OAI-trained -> N pick only (the H family already contains OAI-trained Raptor).
