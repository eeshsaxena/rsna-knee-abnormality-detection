# RAPTOR_RENTED_RESULTS (gate read 10-14 00:25 IST). Governs: PREREG_RAPTOR_RENTED.md (sha 9117ee60, unchanged). No em dashes.

## Verdict: **RAPTOR_RENTED_GATE PASS** (all 4 seeds pass G0 + G1; ensemble legs G2-G5 all pass)
Raw gate output: `GATE_RENTED_RESULT.txt` (box `~/raptor2/GATE_RENTED_RESULT.txt`, script `box/gate_raptor_rented.py`).
N SV = nartaa r384 scored by serve_check.py on the box V100 in the smoke window (gold58 .922943); seeds' SV scored by the same
script on the rental GPU (4090, fp16). F = K14 R fleet (gold58 .9212).

| Seed | 16 ep | nonfinite | SWA epochs | SWA gold58 (train geometry) | SV gold58 |
|---|---|---|---|---|---|
| s43 | yes | 0 | 12, 10, 11 | .9080 | .9123 |
| s44 | yes | 0 | 13, 11, 12 | .9036 | .9127 |
| s45 | yes | 0 | 15, 11, 13 | .9097 | .9113 |
| s46 | yes | 0 | 14, 15, 13 | .9124 | .9143 |

| Leg | Value | Bar | |
|---|---|---|---|
| G2 AUC_SV(Rr, 4 seeds) vs N | .9179 vs .9229 (d -.0050) | >= -.0150 | PASS |
| G3 rank-mean(N, Rr) | .9240 (d +.0011 vs N; bootstrap SE .0033, P(>0) .64) | >= N - .0020 | PASS |
| G4 mean Spearman(Rr, N) | .9494 | <= .970 | PASS |
| G5 [mix .8 + F .2] vs [N .8 + F .2] | .9268 vs .9268 (d .0000) | >= -.0020 | PASS |

Report only:
- Seed-to-seed Spearman .933-.951 (seeds are fairly diverse among themselves too).
- Per label Rr vs N: Rr better on MCL (.956 vs .943), Lateral OA (.803 vs .776), Contusion (.955 vs .937); worse on PF OA (.831 vs
  .860), Effusion (.948 vs .973), Synovitis (.796 vs .829), Lateral Meniscus (.875 vs .896): the OAI-supervised labels, as expected
  for RSNA-only seeds.
- OAI-free reference: [Rr .8 + F_B .2] .9235 vs [N .8 + F_B .2] .9270.
- Subsets (for the kernel seed count, PREREG_RAPTOR_KERNELS Amendment 1; seeds ordered by train-geometry SWA gold58: s46, s45, s43, s44):
  k1 Rr .9143, rank-mean(N, Rr) .9215; **k2 (s46 + s45) Rr .9159, rank-mean .9231, r384 .45 / ours .30 part .9238, with F .9272**;
  k3 .9194 / .9241 / .9235 / .9276; k4 .9179 / .9240 / .9239 / .9268. All differences are inside the gold58 noise (SE ~.003).

## Spend (Vast, path b)
- Attempt 1 (A100 CA, no sshd) $0.19; attempt 2 instance 55762630 20:38-00:37 (~4.0 h x $1.898 + storage + data-in). Final figure
  in VAST_RAPTOR.md after the destroy.

## Next (per PREREG_RAPTOR_KERNELS.md + Amendment 1)
- Kernel seeds: k = 2 (s46, s45) unless a commit-log measurement allows more. Dataset folder packed with pack_seeds.py.

## Report-only experiment 10-14 00:40-01:07: weight soup (does one arm carry several seeds?)
- `box/soup_eval.py`, box CPU (4 cores, fp32), gold58 training geometry. Uniform weight average of s46 + s45 (BN buffers too):
  **gold58 .9000**, below each seed alone (.9124 / .9097) and far below their prediction ensemble (rank-mean .9157, prob-mean .9183;
  k4 .9169 / .9180). Independent fine-tunes from the same ImageNet init do NOT share a basin here: the soup fails. k4 soup not run.
- Consequence for a second Vast batch: the kernel can only afford ~2 mirrored arms (hidden runtime), the soup cannot compress more
  seeds into one arm, so 4 more same-recipe seeds would not reach a kernel. **Recommendation: no second batch now** (keep the
  ~$14.8). Only reconsider if HR7-RS shows a band-position gain over HR7-RR, and then as a diversity swap (e.g. 2 seeds with a
  different target table such as frac75 replacing s45 / s46), under its own prereg.
