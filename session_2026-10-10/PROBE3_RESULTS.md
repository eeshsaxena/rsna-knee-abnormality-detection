# PROBE3 RESULTS (T123v00 probe f0), recorded 2026-10-11 10:35 IST

PROBE3_PREREG.md (sha 3534b34b...) is unchanged; this file is its RESULTS section, kept separate so the gate script's
sha check of the prereg stays valid.

## Gate read (gate_t123.py, 10:21; box ~/oai_build/out/gate_t123.txt)

| gate | value | bar | result |
|---|---|---|---|
| G1 T3 OOF mean (ACL / MedMen / LatMen) | .911297 | >= .905992 | PASS |
| G2 oof12 | .887660 | >= .887764 | **FAIL by .000104** |
| G3 S12 OOF mean | .885324 | >= .884364 | PASS |
| G4 gold58 12-label | .919434 | >= .9113 | PASS |

**T123 GATE: FAIL** (rule as written). Queue runner 10:21: t123.verdict = FAIL, so the gated f1-f4 lines would SKIP.

Per-label OOF (probe): ACL .8882 (T12 probe .8703), MCL .8146, MedMen .9231, LatMen .9226, MedOA .9081, LatOA .8626,
PFOA .8953, Eff .8561, Syn .8606, Bak .9292, Cont .8700, Frac .9215.

## Context (report-only, not part of the gate)

- vs the no-OAI control D f0 (cv .883777, gold58 .9163): T123 f0 cv .88766 (+.0039), gold58 .9194 (+.0031). It is the first
  OAI probe that beats D on gold58.
- The same day, the 288 T12v00 f0 gate (p2c) FAILED: G1 non-OAI OOF .8915 vs bar .8937; G2 gold leg .8935 vs .8960.
  Folds 1-4 SKIP.

## Deviation (explicit, approved)

Eesh approved on 10-11 ~10:25 ("DO THAT") a **GPU-allocation override only**:
- T123 folds 1-4 (21 ep, gc0, seed 1043) run anyway, as queue lines L8c, launched 10:27. Reason: G2 missed by .0001 (noise), while G1 / G3 / G4 pass and the alternatives for the GPU (P2b, negative prior; D2 filler ~+.0002) are low value.
- The protection is unchanged: **no submission** without the fleet gate `PREREG_K14.md`, written BEFORE any fold 1-4 read
  (sha in PREREG_K14.sha256).
