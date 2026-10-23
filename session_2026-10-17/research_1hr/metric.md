# Metric + blending lens (research_1hr, 2026-10-17 ~22:00-22:45 IST). Read-only. No em dashes.

Tags: MEASURED (computed here or read from a file/API snapshot), CLAIMED (someone else said it), MODEL (our arithmetic on
assumptions), GUESS. Scripts + raw outputs: `research_1hr/metric_work/` (g58.py, scale.py, signs.py, max4.py, max4b.py,
g58_out.json, scale_out.json). Gold58 inputs were copied read-only from the box (gold74_eval_set.csv, R7OOF.csv,
raptor2/gold58_N_crop320_k24.npy + full384, s6seed/gold58_serve/nartaa_s42 predictions). Nothing submitted, nothing on the GPU.

## 0. Bottom line (5 lines)
1. The three pending probes are coin flips with a small tilt. Public sd of each probe's macro delta is ~3-7e-4 (MODEL, gold58
   bootstrap scaled), true effects are likely 1e-4 to 1e-3, so P(private agrees | public read positive) ~ .63-.75 (MODEL).
   They cannot tell magnitude, per-label attribution, or anything at display precision. Expected private stake of the
   whole A16 probe set: about +-3e-4 macro (MODEL), roughly 2-10 private ranks outside a clone clump.
2. Gold58 (independent of public) says HW sits near a local optimum of both knobs: W_RAPTOR .45 and .55 both lose, S6fall
   .15 and .25 both lose (MEASURED on gold58, n 58, noisy, R7 stands in for R6). Prior for each probe: slightly negative.
3. The silver question is NOT the probes. It is HW vs the heliosli clone mass: since HW read (10-16 15:17 UTC) the .958 band grew 395 -> 853 teams while the
   teams above us in that band grew only 15 -> 29 (MEASURED, API snapshots), so the clones sit BELOW HW on public. HW =
   clone + S6fall .2. Gold58 puts that S6fall add at +.0033 macro (MEASURED, sd .0016 at n 58); P(HW > clones on private)
   ~ .87-.90 with gold58, .62-.67 on the public bit alone (MODEL). That single comparison decides most of our private rank, and the evidence for it is good.
4. The weights question (pick A vs pick B) lives almost entirely in Lateral OA and Synovitis: on gold58, heliosli vs
   uniform = +.0024 macro, of which Lateral OA alone is +.0387 per-label (= +.0032 macro); PF OA and Baker's go the other
   way (MEASURED). Lateral OA and Synovitis are also two of the highest-variance labels on private (MODEL).
5. Rare labels carry the private noise: under train-report prevalence, Fracture (~62 private positives) and Synovitis
   (~115) are each ~25 % of the macro variance; with Lateral Meniscus + Lateral OA that is ~74 % (MODEL, Hanley).

## 1. Facts the analysis rests on
- Read mechanism: API best-id compares at FULL precision (MEASURED: HW 57262115 .958 replaced R6 57248084 .958 as best-id,
  PREREG_10-15 A11 read 20:47 10-16). The leaderboard CSV is also 3 decimals (MEASURED: scratchpad/lb/*.csv Score column),
  so the only sub-display signal we ever get is the binary best-id move. Rounding vs truncation of displayScore: unknown (GUESS).
- Public LB 16:21 UTC 10-17 (MEASURED, api_lb_1017T2150.json): 6,144 teams; .958 band = ranks 142-994 (853 teams); we are
  r171 (29 above us inside the band); .959 = r94-141 (48 teams); silver cut r307 and bronze cut r614 are both INSIDE the
  .958 band. Gold = 22 teams.
- Band history with our entry inside it (MEASURED, api_lb_*.json): R6 read 10-16 06:22 UTC band 99, above-in-band 41; combo
  read 08:23 band 177, above 51; HW read 15:17 band 395, above 15; 10-17 08:05 band 728, above 25; 16:21 band 853, above 29.
  About 750 entrants landed below HW. PROGRESS_16 line 110 (MEASURED there): heliosli/blend-mine-in-private-lb-luck-to-all-upvoters
  = V224 + R6 + flip, NO S6fall, per-finding Raptor weights (Lateral OA .2, PF OA .6, Synovitis .8, Baker's .35), 164+ copies.
- HW stage 3 (MEASURED, kernel cell 14/18): per label `wr*pct(Raptor slot) + (1-wr)*pct(ours)`, Raptor slot =
  `.65 pct(r384) + .35 pct(r224)`, wr = heliosli table for 5 labels, .5 for ACL, MCL, Medial Meniscus, Lateral Meniscus,
  Effusion, Contusion, Fracture; final = `.8 pct(stage3) + .2 pct(S6fall)`. wr45 touches those 7 labels; s615 and a25
  touch all 12.

## 2. Label prevalence
| label | gold58 pos/58 (MEASURED) | train-report rate, gpt56sol labels on 4,407 (MEASURED) | private positives @910 report / gold58 (MODEL) |
|---|---|---|---|
| ACL | 24 (.41) | .204 | 186 / 377 |
| MCL | 9 (.16) | .147 | 134 / 141 |
| Medial Meniscus | 26 (.45) | .394 | 359 / 408 |
| Lateral Meniscus | 23 (.40) | .151 | 137 / 361 |
| Medial OA | 15 (.26) | .359 | 327 / 235 |
| Lateral OA | 11 (.19) | .257 | 234 / 173 |
| PF OA | 21 (.36) | .450 | 410 / 329 |
| Effusion | 35 (.60) | .593 | 540 / 549 |
| Synovitis | 27 (.47) | .126 | 115 / 424 |
| Baker's | 12 (.21) | .247 | 225 / 188 |
| Contusion | 19 (.33) | .168 | 153 / 298 |
| Fracture | 18 (.31) | .068 | 62 / 282 |
- Gold58 is pathology-enriched vs reports: Fracture 4.6x, Synovitis 3.7x, Lateral Meniscus 2.6x (MEASURED ratio). Which
  regime the hidden labels follow is unknown; the data page says prevalence is "not guaranteed" equal across splits (CLAIMED,
  RESEARCH_TO_TOP_10-06 sec 3). The report regime is the conservative (noisier) one. Public positives under the report
  regime: Fracture ~27, Synovitis ~49, MCL ~57, Lateral Meniscus ~59 of 390 (MODEL).
- Other LLM label sets give similar report rates (llm_labels_v4_blend: Fracture .072, Synovitis .126; MEASURED). The old
  pseudo_labels_v1 file is not usable for prevalence (0.5 fills; Medial OA .51 at >.5).

## 3. Which labels carry the variance (single-entry AUC noise)
Hanley-McNeil SE at the gold58 HW-like per-label AUC levels (MODEL; real AUCs are higher, so true SEs are a bit smaller):
- Private, report regime: macro SE .0039 if labels were independent. Share of macro variance: Synovitis 25 %, Fracture 25 %,
  Lateral Meniscus 12 %, Lateral OA 12 %, PF OA 7 %, Contusion 6 %, MCL 5 %, Baker's 3 %, ACL 2 %, the 3 easy ones 1 % each.
- Private, gold58 regime: macro SE .0030; Lateral OA 26 %, Synovitis 14 %, PF OA 14 %, Fracture 10 %, MCL 8 %, Lateral Meniscus 8 %.
- Public: SE x ~1.5 (sqrt(910/390)); macro SE ~.005-.006 for one entry vs the truth (MODEL). That is why absolute public
  scores mean little for private order, while PAIRED deltas between near-identical entries are ~10x less noisy (sec 5).
- Macro arithmetic: a per-label move of d shifts macro by d/12; one display step (.001) needs .012 on one label, or ~.001 on
  every label (MEASURED identity).

## 4. Gold58 per-label table (MEASURED, n 58; caveats below)
| label | R7 (proxy for ours/R6) | Raptor crop320 | S6fall nartaa | HW-like blend | wr45 | s615 | uniform (R6-pick-like) |
|---|---|---|---|---|---|---|---|
| ACL | .980 | .956 | .989 | .982 | .982 | .982 | .982 |
| MCL | .952 | .952 | .966 | .969 | .972 | .968 | .969 |
| Medial Meniscus | .955 | .986 | .960 | .978 | .977 | .980 | .978 |
| Lateral Meniscus | .924 | .865 | .935 | .922 | .926 | .918 | .922 |
| Medial OA | .980 | .974 | .981 | .981 | .981 | .981 | .981 |
| Lateral OA | .878 | .768 | .851 | .865 | .865 | .867 | .826 |
| PF OA | .888 | .843 | .902 | .876 | .876 | .870 | .880 |
| Effusion | .942 | .964 | .949 | .971 | .968 | .969 | .971 |
| Synovitis | .794 | .833 | .827 | .845 | .845 | .841 | .845 |
| Baker's | .938 | .978 | .980 | .970 | .970 | .967 | .976 |
| Contusion | .940 | .943 | .934 | .958 | .956 | .957 | .958 |
| Fracture | .860 | .961 | .900 | .924 | .915 | .924 | .924 |
| macro | .9193 | .9185 | .9313 | .9369 | .9361 | .9353 | .9345 |
- Caveats: Raptor's public checkpoint was SELECTED on gold58 (stored gold_auc .918952; CLAIMED by its author, reproduced
  .9185 here), so Raptor's gold58 numbers are optimistic and anything that lowers Raptor's weight looks worse here than it
  is. R7 OOF stands in for "ours" (R6 bundle + V224); no R6 or r224 gold preds exist, so a25 cannot be checked and the
  Raptor slot here is r384-crop only. n 58: per-label AUC SE ~.02-.05.
- Within-class rank correlation, mean over labels (MEASURED): R7-Raptor .66, R7-S6fall .68, Raptor-S6fall .77,
  Raptor crop320-full384 .97. Raptor and S6fall are the closer pair; R7 is the most different leg.
- Weight curves on gold58 (MEASURED): W_RAPTOR on the 7 non-table labels .3/.4/.45/.5/.55/.6/.7 -> .9341/.9354/.9361/.9369/
  .9366/.9370/.9364 (flat between .5 and .6); W_S6F 0/.1/.15/.2/.25/.3/.4 -> .9336/.9355/.9353/.9369/.9366/.9368/.9373
  (flat .2-.4). Both curves are flat-topped around HW's values: a .05 step moves macro by <= .0008.
- Where each leg wins (MEASURED, gold58): Raptor >> R7 on Fracture (+.10), Baker's (+.04), Synovitis (+.04), Medial Meniscus
  (+.03), Effusion (+.02); R7 >> Raptor on Lateral OA (+.11), Lateral Meniscus (+.06), PF OA (+.045), ACL (+.025).
  heliosli's table agrees with gold58 on Lateral OA (.2) and Synovitis (.8) and disagrees on PF OA (.6) and Baker's (.35):
  gold58 per-label HW minus uniform = Lateral OA +.0387, PF OA -.0039, Baker's -.0063, Synovitis +.0006.

## 5. Paired-delta noise (the number that matters for reads)
Paired bootstrap over the 58 studies (1,500 resamples, ranks recomputed per resample), per-label covariance scaled to 390 /
910 studies with a Hanley 1/P + 1/Q factor per label (MODEL). "g / r" = gold58 / report prevalence regime.
| comparison (macro delta) | gold58 point | sd @58 | public sd g / r | private sd g / r | labels driving the variance (share) |
|---|---|---|---|---|---|
| wr45 - HW | -.0008 | .0013 | .00051 / .00073 | .00033 / .00048 | Fracture 33 %, Lat Meniscus 23 %, MCL 14 %, Contusion 13 % |
| s615 - HW | -.0016 | .0011 | .00041 / .00054 | .00027 / .00035 | Synovitis 19 %, Lateral OA 12 %, Fracture 12 %, PF OA 12 % |
| wr55 - HW (not submitted) | -.0003 | .0013 | .00049 / .00070 | .00032 / .00046 | same as wr45 |
| s625 - HW (not submitted) | -.0003 | .0012 | .00047 / .00060 | .00031 / .00039 | same as s615 |
| HW - clone (heliosli, no S6fall) | +.0033 | .0016 | .00061 / .00081 | .00040 / .00053 | Synovitis 19 %, Fracture 13 %, Lat OA 11 %, PF OA 11 % |
| R6 pick (uniform + S6) - clone | +.0009 | .0028 | .00108 / .00139 | .00070 / .00091 | Synovitis 41 %, Lateral OA 31 % |
| HW - R6 pick (the weights question) | +.0024 | .0023 | .00089 / .00113 | .00059 / .00074 | Synovitis 50 %, Lateral OA 37 % |
- An older binormal sim with homogeneous labels (lbprobe/sim3_070.txt, sim3_073.txt; MODEL) put a W_S6F .2 -> .15 move at
  public sd ~1.3-1.5e-4, about 3x below this table. The gold58 bootstrap includes the real per-label disagreement between
  legs (Fracture .86 vs .96, Lateral OA .88 vs .77), which inflates the delta noise; the truth is likely between. I quote both.
- a25 (no gold preds): MODEL by perturbation size, public sd ~3e-4 (r224 vs r384 are a closer pair than R7 vs Raptor, so ~0.6x
  wr45). GUESS-level.

## 6. What the three pending probes can and cannot tell us
- Each read is one bit: sign of (probe - HW) on 390 public studies at full precision. Display: a probe still shows .958 unless
  its delta crosses HW's unknown sub-display position; .957 would mean a clear loss, .959 a clear win (both unlikely at
  sd 3-7e-4 unless HW sits near a band edge; GUESS, HW's offset inside .958 is unknown).
- Per-probe (signs.py, MODEL; prior "flat" = N(0, 5e-4); prior "gold58" = gold58 point with sd58 + 1e-3 mismatch):
  - wr45: P(read positive) .50 flat / .42 gold58; P(private > 0 | positive) .63-.75; E[private delta | positive] +2 to +5e-4.
  - s615: P(read positive) .50 flat / .29 gold58; P(private > 0 | positive) .66-.74; E[private | positive] +3e-4.
  - a25: no gold58 evidence; treat as flat, P(read positive) ~.5.
- Cannot tell: (a) magnitude; (b) which labels moved (wr45's noise is half Fracture + Lateral Meniscus, the two labels with
  the fewest public positives under the report regime, ~27 and ~59); (c) whether the "R6 is stronger than R7" rationale for
  wr45 is right (no R6 gold preds); (d) curvature: a non-move on .15 says nothing about .25 (gold58 says both lose).
- Three parallel probes vs one HW: A15 makes pick A = argmax public over {HW, wr45, s615, a25}. P(A changes) .67-.88 under
  zero-mean priors, .09-.78 under a "HW is at a local optimum" prior (max4.py / max4b.py, MODEL). Expected private gain of
  that switch: +2.4 to +4.2e-4 if probe effects are zero-mean; -0.1 to -1.3e-4 if every one-line move truly loses (the
  gold58 picture). Either way |E| <= ~4e-4 macro and P(loss worse than -5e-4) <= .03. Low stakes both ways.
- Recommendation for the reads (MODEL): accept A15 mechanically; a probe move is weak evidence, not a reason for any new
  probe on the same knob. Do NOT stack two read-positive probes into a combo entry on public evidence alone (two coin flips
  with ~.7 transfer each); if Eesh wants one, gold58 says the wr45 knob is closer to neutral than s615.

## 7. Picks and the silver question through this lens
- Private rank is dominated by where we land vs the clone mass (164 copies MEASURED 10-16; GUESS several hundred now in .958; the silver
  cut r307 lies inside that mass on public). Near-identical copies move together on private (MODEL: run noise ~5e-6, sim_noise.py),
  so the question is one comparison: HW (= clone + S6fall .2, same decoder family) vs clone.
  - Evidence: public says HW > clones (MEASURED band arithmetic, sec 1); gold58 says +.0033 (MEASURED, independent).
    P(HW private > clone private) ~ .87-.90 with the gold58 prior, .62-.67 on the public bit alone (MODEL, signs.py).
  - If HW stays above the mass on private, the remaining risk is non-clone teams overtaking us; if HW falls below it, we
    drop behind the whole mass (~r900+, below even bronze r614). Downside tail ~10 % with gold58, up to ~35 % on the public bit alone.
- Pick B (R6, uniform + S6fall) vs the clones: +.0009 on gold58 with sd .0028 (P(private > clones) ~.55-.70 given the public bit, MODEL). B is a
  weaker silver bet than A; its value is the hedge on Lateral OA / Synovitis weights, where public-tuned weights (heliosli)
  could reverse. Gold58 (independent of public) favours heliosli's Lateral OA weight strongly and slightly disfavours its
  PF OA and Baker's weights, so the hedge buys little (MODEL). A15 B stays defensible because A and B share S6fall (the
  real edge over the clones) and differ only on the weights.
- If a probe becomes A, B stays R6 (A15). The A/B pair then differs in weights AND the probed knob; still fine.

## 8. Leads for the orchestrator
1. (MODEL, high value, zero cost) Keep the final-pick logic focused on "S6fall in both picks": that is the only component that
   separates us from the clone mass and both public and gold58 back it.
2. (MEASURED gap) A cleaner gold58 check of the weights question needs R6 gold58 preds (not on disk). Not worth GPU now; noted.
3. (GUESS) If the hidden labels follow the gold58 (enriched) regime, Lateral OA is the single biggest variance label; if they
   follow report rates, Fracture and Synovitis are. The A-vs-B difference is concentrated in Lateral OA + Synovitis either way.
