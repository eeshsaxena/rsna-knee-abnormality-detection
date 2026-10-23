# PREREG_RAPTOR_RENTED (written 10-13 ~21:02 IST, while the 4 rented seeds are in epoch 0, BEFORE reading any of their gold58 numbers). No em dashes.
Companion of PREREG_RAPTOR_SEED.md (sha bc2cef53, unchanged; it governs the box seed S with OAI). This file governs the rented seeds.

## Object
- **Rr** = the RSNA-only Raptor seeds 43, 44, 45, 46 trained on Vast instance 55762630 (train_raptor.py, same recipe and targets as S
  minus the 2,399 OAI rows; bf16; 16 epochs; each seed's final = uniform average of its 3 best epochs by gold58). OAI-free.
- Ensemble **Rr** = per-label rank-mean of the SV probabilities of every seed that passes G0 and G1 below (equal weights).
- N, F, gold58, SV (serving geometry: 320 crop, 94 windows, mirror, fp16 autocast on the rental GPU / box GPU), rank blends: as in
  PREREG_RAPTOR_SEED.md. N's SV predictions come from the box (serve_check.py), the seeds' SV predictions from the rental (same
  script, same 58 studies, same order).

## Per-seed legs (a seed failing either is dropped from Rr)
| Leg | Definition | Bar |
|---|---|---|
| G0 integrity | 16 epochs completed; non-finite loss steps <= 1 %; swa.pt strict-loads at img_size 320; SV outputs finite; box-side sha256 verification of the fetched files | all true |
| G1 port sanity | seed SWA gold58 at training geometry (trainer's "=== SWA" line) | >= .9000 |

## Ensemble legs (on Rr; all must PASS)
| Leg | Definition | Bar |
|---|---|---|
| G2 quality vs N | AUC_SV(Rr) | >= AUC_SV(N) - .0150 |
| G3 Raptor-slot blend | AUC of rank-mean(N .5, Rr .5), SV | >= AUC_SV(N) - .0020 |
| G4 diversity | mean over 12 labels of Spearman(Rr, N), SV | <= .970 |
| G5 blend with our fleet | [rank-mean(N, Rr) .8 + F .2] vs [N .8 + F .2] (F = K14 R fleet, 20 members) | >= N-side - .0020 |

## Decision
- PASS: Rr may enter the Raptor slot of HR7 / HFR7 variants (e.g. Raptor_mix = N .5 / Rr .5, or N / S / Rr if S also passes its own
  prereg); exact weights fixed in a separate submission prereg BEFORE any leaderboard read. Rr weights are OAI-free (RSNA data +
  public label tables + public ImageNet weights), so they may also serve an OAI-free pick (decision for main after the 748276 ruling).
- Any ensemble leg FAILS: no submission with Rr; numbers reported to main.
- Report only: per-seed SV AUCs; AUC_SV(Rr) vs AUC_SV(S); seed-to-seed Spearman; [Rr .8 + F_B .2] with F_B = the OAI-free FN11 fleet B
  (C + D + E + E2) as the OAI-free reference.
