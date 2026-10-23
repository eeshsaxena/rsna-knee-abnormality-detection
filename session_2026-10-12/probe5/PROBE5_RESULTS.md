# PROBE5 RESULTS (Swin-T on the T123 OAI recipe, probe f0), recorded 2026-10-13 01:40 IST
PROBE5_PREREG.md (sha c336cae8) unchanged. Gate auto-run by the queue at ~23:45 10-12 (gate_sw.sh); copy gate_sw_result.txt.

**SW GATE: FAIL** (G2). Swin folds 1-4 SKIPPED by the queue; PROBE4 (PL) folds started 23:49.

| gate | value | bar | result |
|---|---|---|---|
| G1 strength OOF12(SW) | .882783 | >= .881660 | PASS (DT f0 .887660, D f0 .883777) |
| G2 diversity OOF12(mean DT,SW) - OOF12(mean DT,D) | -.000091 (.888600 vs .888691) | >= +.0010 | **FAIL** |
| G3 gold58 mean(DT,SW) - mean(DT,D) | -.0035 (.9175 vs .9210) | >= -.005 | PASS |

Report: Spearman SW~DT .948, SW~D .944, DT~D .957. Swin is a little less correlated with our ConvNeXt models but weaker, so as a
partner it adds nothing over another ConvNeXt (OOF and gold58 both slightly lower). Conclusion: a backbone swap on our
recipe does not create the diversity we need; the bottleneck is the recipe / labels, not the architecture.
