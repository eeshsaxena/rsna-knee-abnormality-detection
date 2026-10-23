# PREREG_RAPTOR_SEED (written 10-13 ~18:05 IST, BEFORE the GPU smoke and BEFORE any full-run number exists). No em dashes.

## Object
- **S** = our seed of nartaa's Raptor .949 recipe: `~/raptor2/train_raptor.py` (code md5 printed on line 2 of its train.log),
  seed 42, all-data = 4,349 RSNA rows (targets `labels_raptor_rsna_blendjev_sol.csv`, md5 f1fa9d51) + 2,399 OAI V00 rows
  (`~/oai_build/oai_labels_raptorT1.csv`, md5 6b248af8; images from `~/oai_store` via build_oai_d96_approx.py), d96 / 384 corpus
  from the Kaggle prep kernels `eeshsaxena/rsna-raptor-d96-prep-s{0,1,2}` (reader sha 30aa72fd).
  Epochs: 16, or 12 if the smoke's measured s/step projects the 16-epoch run above 26 h (rule fixed by main 10-13).
  Final weights = uniform average of the 3 best epochs by gold58 (24 windows, full field 384), as nartaa.
- **N** = nartaa r384 SWA (`raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`, sha256 7e5315da...44ef).
- **F** = our K14 "R" fleet (C + DT + E + E2, 20 members), gold58 pass-A predictions, equal mean of member probabilities, exactly
  as `~/k14_test/k14/box/k14_gate.py` computes it (its logged gold58 = .9212).

## Evaluation set and geometry
- All legs use **original_gold58** (58 studies; `~/rsna_train/gold74_eval_set.csv`, source == original_gold58). Both N and S
  exclude them from gradient training, but BOTH use them for epoch selection, so they are in-sample for selection (disclosed;
  same status as nartaa's released model). JEV inside S's targets was tuned on gold58 by its author.
- **No OOF leg:** the recipe is all-data (no held-out fold) for N and for S; an OOF comparison does not exist for either.
- **SV (serving geometry)** = `~/raptor2/serve_check.py`: model built at img_size 320, 94 windows, centre crop [32:352],
  anatomical mirror TTA, fp16 autocast, mean of the 2 sigmoid vectors = the HR7 Raptor stage. Run on the box V100 in a queue
  pause window (or on CPU fp32 if no window: then N and S are both scored on CPU, never mixed).
- Rank blends = per-label average-tie percentile rank over the 58 studies.

## Legs and bars (all must PASS)
| Leg | Definition | Bar |
|---|---|---|
| G0 integrity | all planned epochs completed; non-finite loss steps <= 1 % of steps; swa.pt strict-loads at img_size 320; all SV outputs finite | all true |
| G1 port sanity | S SWA gold58 at training geometry (24 windows, full field 384; the trainer's "=== SWA" line) | >= .9050 |
| G2 quality vs N | AUC_SV(S) | >= AUC_SV(N) - .0150 |
| G3 Raptor-slot blend | AUC of rank-mean(N .5, S .5), SV | >= AUC_SV(N) - .0020 |
| G4 diversity | mean over the 12 labels of Spearman(S, N), SV predictions | <= .970 |
| G5 blend with our fleet | AUC of rank blend [Raptor_mix .8 + F .2], Raptor_mix = rank-mean(N, S), vs AUC of [N .8 + F .2] (the HFR7-R20 fleet weight; R7 has no gold58 predictions on the box, so it is left out of both sides) | >= the N-side value - .0020 |

## Decision
- **PASS (G0-G5):** build HR7 / HFR7 variants whose Raptor slot is Raptor_mix = r384 .5 / S .5 (optionally with nartaa r224 as in
  HR7-RR); exact kernel weights fixed in a separate submission prereg BEFORE any leaderboard read. S is OAI-trained: weights go
  to a new private dataset only after the OAI-clean scan (weights + manifest only), and kernels with S go to the OAI pick only.
- **G0 or G1 FAIL:** the port is broken; no submission with S; investigate (labels, corpus, loop) before any rerun.
- **Any of G2-G5 FAIL:** no submission with S; numbers reported to main.
- Report only (no decision): per-label AUCs; AUC_SV(S) vs F alone (.9212); S's training-geometry gold58 vs N's stored .918952
  and our CPU re-score of N .919111; per-epoch gold58 curve; nonfinite / skipped-update counts.

## Smoke gate (GPU smoke, before the full run; fixed by main 10-13)
- SMOKE_GO = no non-finite loss and no skipped update after the first 2 updates (GradScaler warm-up), measured s/step <= .85
  (median of the 200-step windows after the first), peak memory <= 14.5 GB, swa.pt serve-load OK, SV outputs finite.
- Full run launch also needs: all 4,407 RSNA studies unpacked with receipt sha256 OK, and the 458-study box-mirror cross-check
  passed (byte-identical, or a documented max |diff| that is explained), plus this file's sha256 recorded.
