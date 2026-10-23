# VLM relabel pilot, in-session run (prereg_v3.json sha 5af3590d): RESULT 2026-10-12 ~11:20 IST

**DECISION (pre-registered v2 gate): NO-GO.** No scale-up.

- Run: 11 Claude Code subagents (Opus 5.5) on Eesh's plan, no API spend; blinded task folders (cc_make_tasks.py);
  text arm and img arm by different agents. 74/74 studies valid in both arms; 58/58 gold58 paired; 1 prompt sha
  (c5a72790), 1 model.
- V validity: ok. G1 sign (Effusion img - text): +0.017 (SE .030) pass.
- **S decision: mean6 AUC(0.5 logit mix50 + 0.5 logit img) - AUC(mix50) = +0.0080 (SE .0097, 90% CI [-.008, +.024]) < +0.010: FAIL.**
- Images over report-only: mean6 +0.006 (non-Effusion mean +0.0035, SE .0078): the key slices add little.

| label | n+ | text | img | mix50 | v4 | blend - mix50 (90% CI) |
|---|---|---|---|---|---|---|
| Effusion | 35 | .860 | .877 | .984 | .877 | -.035 [-.073, -.005] |
| Synovitis | 27 | .802 | .822 | .823 | .790 | +.013 [-.036, +.068] |
| Fracture | 18 | .911 | .911 | .843 | .793 | +.054 [+.007, +.112] |
| Contusion | 19 | .874 | .870 | .906 | .860 | -.004 |
| PF OA | 21 | .925 | .923 | .888 | .902 | +.019 [-.001, +.043] |
| Lateral OA | 11 | .849 | .853 | .843 | .833 | -.000 |

Exploratory only (NOT a decision; post-hoc, 18-21 positives, would need its own prereg and independent data):
the VLM labels beat mix50 on Fracture and PF OA and lose on Effusion. Files: soft_{text,img}_cc-opus-5-5.csv,
pilot_eval_cc-opus-5-5_v3.log, pilot_eval_cc-opus-5-5_v2.json (here and in kit/results/).
