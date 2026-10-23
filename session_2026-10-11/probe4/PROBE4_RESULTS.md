# PROBE4 RESULTS (T123 + same-fold pseudo-label fill, probe f0), recorded 2026-10-12 ~17:45 IST
PROBE4_PREREG.md (sha 1aaff349) unchanged. Gate run EARLY on box CPU 17:26 (gate_pl4.sh; the queue reuses the verdict file).

**PL4 GATE: PASS** (`~/oai_build/out/pl4.verdict`; copy gate_pl4_result.txt)

| gate | value | bar | result |
|---|---|---|---|
| G1 P3 OOF mean {MCL, Contusion, Fracture} | .873202 | >= .871693 | PASS (+.0045 vs T123 probe) |
| G2 oof12 | .889698 | >= .887360 | PASS (+.0020 vs T123 probe .887660) |
| G3 R9 OOF mean | .895196 | >= .891982 | PASS (+.0012) |
| G4 gold58 12-label | .919379 | >= .914434 | PASS (T123 probe .919434) |

Report only: gold58 P3 mean .9312 (T123 .9241); per label gold58 MCL .955 (T123 .918), Contusion .951 (.957), Fracture .888 (.897).
Closing line raw .8897 / ema .8891.
PASS branch: PL folds 1-4 (L9b, 21 ep, gc0) are queued AFTER the Swin probe and (if it passes) the Swin folds.
A 5-fold PL set needs its own pre-registered fleet / HF gate before any submission.
