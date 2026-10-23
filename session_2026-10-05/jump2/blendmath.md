# Blend math on our measured LB points, and the fleet we actually need (jump2, 2026-10-05 ~09:00 IST)

Tags: **MEASURED** (source given) = an LB read, a log, or a number computed today from on-disk data.
**MODEL** = binormal two-reader model (AUC = Phi(d/sqrt2); rank blend at W: d_b = (1-W)d1 + W d2,
v = (1-W)^2 + W^2 + 2W(1-W)r, AUC_b = Phi(d_b/sqrt(2v)); k-member ensemble v_k = (1+(k-1)r_mm)/k,
solo_e = Phi(d/sqrt(2 v_k)), r_e = r_m/sqrt(v_k)). **GUESS** = judgement. Every displayed LB read x is treated as the
interval [x-.0005, x+.0005). "r" below is always the effective binormal r (the one that predicts blend reads).

Scripts and logs (all in this folder, `SP/jump2/`): `lbmath.py`/`lbmath.out` (LB inversions, laptop),
`oofx.py`/`oofx.log`/`oofx_out.json` (fold-0 OOF structure, run on the box under `~/jump2/`, CPU, nice 19),
`labx.py`/`labx.log` (label tables: gold58 vs parent-likeness, box CPU), `fleetsim.py` + `fleetsim_psd.out` /
`fleetsim_nopsd.out` (composition table and K4 value of information, laptop). Nothing on the box outside `~/jump2/`
was written; no process was touched.

---

## 0. Bottom line

1. **K2 pins our F1 mix50 member at r_LB .895-.945 (point .920)** (MODEL on MEASURED reads). At that r its best
   possible blend is +.0000 to +.0011 (W* .00-.24). K2 reading .943 was the expected outcome, not bad luck.
2. **K3 does not constrain TTR v4 by itself** (r_LB .81-.97 is consistent). Through the OOF mapping (sec 4) TTR v4 sits
   at **r_LB ~.88 (.865-.90)** (MODEL). Its weakness is strength (.923 solo), not correlation.
3. **The aastik ".79" is an artefact.** No single r fits his two reads (.945 at W .5 AND .945 at W .6) if his fleet
   solo is .927 (MODEL). Both reads hold together only if his 10-checkpoint fleet is **solo >= .935**, on the line
   **r ~ .86 at .935, .88-.89 at .937, .90-.92 at .939, .92-.94 at .941** (MODEL). His fleet is not much less
   parent-like than a v4-label member of ours; it is much **stronger** at that r than anything we have.
   The ".927" was never a read of the shipped fleet: his own script comment gives LB .927 for an EARLIER
   6-model raw mean; the shipped 10 checkpoints are a newer "pt_v5" generation (sec 10.1, MEASURED script text).
4. **Why ours is more parent-like (MEASURED on fold-0 OOF):** the labels. Same F1 recipe and seed, OOF effective r vs
   the parent's Raptor proxy: mix50 .911, gated .887, v4 .857. Recipe changes it little (TTR vs F1: +.005 to +.011).
   Ensembling raises it (F1 mix50 + TTR mix50 .924 > either member). The Dread teacher inside mix50 is the channel:
   target-level Spearman vs the Raptor proxy is .867 for mix50 vs .653 for steven_v4 (MEASURED `labx.log`).
   Pure LLM label sets are not anomalously parent-like for their quality (report_v2 .870 gold58 / .621, steven_v4
   .893 / .653).
5. **Every fleet we can build from the current recipes caps at a true lift of about +.0005** (MODEL, sec 6): 5-fold TTR
   mix50, mix50 + v4 mixes, 10-15 member fleets, a second backbone on v4. Best realistic display: .944 with
   probability ~.4-.5, .945 with ~.01-.10.
6. **What we need (MODEL):** about 10 members each **~.930 solo at member r_LB <= .86** (fleet ~.934 at r ~.88 gives
   +.0022, display .945 in every draw). Our v4 members already have roughly that r but are .007 too weak; our mix50
   members have the strength but r is ~.06 too high. The lever is **strength without the parent teacher**: better
   report labels or a self-teacher (our own OOF), not more seeds of what exists.
7. **K4 tomorrow (10-06):** best expected public score = **parent + 5-fold TTR mix50 at W .20**
   (E[display] .9434, P(.944 or better) ~.40, P(.945+) ~.03). Most information = **parent + 5-fold TTR mix50 at
   W .50** (highest value of information, halves the uncertainty of the final-fleet lift; expected display .942).
   Pair either with the **SOLO of the 5-fold TTR mix50 fleet**: SOLO + BLEND pins the fleet's r to about +-.025.

---

## 1. K2: F1 mix50 r_LB (MODEL on MEASURED reads)

Inputs: parent .943, K2 SOLO .932, K2 BLEND W .30 .943 (MEASURED, NEXT_SESSION 0b). All three as intervals.

| quantity | value |
|---|---|
| r_LB(F1 mix50), all corners | **.895-.945** |
| point (P .943, s .932, b .943) | .920 |
| W* and best true blend at r .895 / .920 / .945 | .24 -> .9441 (+.0011) / .15 -> .9433 (+.0003) / .00 -> .9430 (+.0000) |
| W .50 read at r .895 / .945 | .9427 / .9403 |
| member solo needed for +.001 / +.002 / +.003 at r .920 | .9357 / .9391 / .9416 |

After also conditioning on K3 and the OOF structure (sec 6 rejection sampler) the posterior is r_LB .908-.932
(10-90%), median .921 (MODEL). The pre-K2 prior (.95, range .90-.97) was slightly too pessimistic; the "LOW .90"
regime is at the edge, not the centre.

## 2. K3: TTR v4 (MODEL on MEASURED reads)

Inputs: TTR v4 SOLO .923, STRICT parent + mean(F1 mix50, TTR v4) at W .25 = .943 (MEASURED). Member-to-member
effective r_mm(F1 mix50, TTR v4) = .920 (MEASURED `oofx.log`, fitted to measured rank blends on 867 OOF studies).

| r_mm | 2-member fleet solo (central) | r_LB(TTR v4) consistent with K3 |
|---|---|---|
| .88 | .9337 | .816-.975 |
| .90 | .9327 | .812-.972 |
| .92 | .9317 | .809-.969 |
| .94 | .9306 | .805-.966 |

- The 2-member mean is about as strong as F1 mix50 alone (.932), so K3 mostly re-measured K2 at a lower W. K3 alone
  says only that the 2-member fleet has r_e <= ~.95 (MODEL: e .932 -> r_e .897-.953).
- Break-even for TTR v4 alone at W .25: r .871 (MODEL). TTR v4 needs r below ~.87 just to not hurt at W .25.
- With the OOF mapping of sec 4 (offset calibrated on K2): **r_LB(TTR v4) .865-.90, median .882** (MODEL).

## 3. aastik re-derived

**What his script is** (MEASURED, `SP/intel/aastik_ours_script.py`, byte-identical to the base64 payload in
knee-s75-w50/w60; local sha differs only by file, content equal):
- 10 checkpoints, `f{0..4}_convnext_tiny_320x24_pt_v5_cv{8961..9154}_best.pt`: 5 folds x 2 per fold. His own comments
  describe a "5 folds x 2 arms x best+EMA" campaign, so the 2 per fold are either 2 arms or best+EMA (if best+EMA they
  are near-duplicates, r ~.99, and the effective k is ~5). Fold CVs .896-.915 are vs HIS labels, not comparable to
  our vs-v4 numbers.
- Labels: not visible (inference only). Nothing in the script refers to a teacher; it calls OOF-vs-label "a map of
  label noise", which reads like report-derived (LLM) labels (GUESS).
- Input: 512 px JPEG store, 32 source slices, crop_mm read per checkpoint (shipped recipe 150 mm; default 130), 24
  triplets x 4 views (Ax FS, Sag non-FS, Cor FS, Sag FS = ours), 320 px, per-target heads/attention pools.
- Inference: `KNEE_LEAN=1` is injected, so **no offset TTA and no mirror TTA**; raw-probability mean of the 10
  (his note: same 6 models LB .927 raw vs .924 rank-mean).
- Blend: parent (s74-c, measured .943 with no fleet) rank + his fleet rank at one global OUR_WEIGHT .5 / .6, re-ranked.
  Both read .945 (MEASURED public).

**Inversion** (MODEL, `lbmath.out` sec 3; parent .943 and both reads as intervals):

| assumed fleet solo | r from W .5 read | r from W .6 read | r fitting BOTH |
|---|---|---|---|
| .9275 | .787-.826 | .748-.788 | none |
| .930 | .809-.849 | .775-.816 | none |
| .9325 | .831-.871 | .803-.844 | none |
| .935 | .854-.895 | .832-.874 | .858-.873 |
| .9375 | .878-.920 | .862-.905 | .878-.905 |
| .940 | .903-.946 | .893-.937 | .903-.937 |
| .9425 | .929-.972 | .925-.970 | .929-.970 |

Joint feasible region: solo .935-.947 with r .859-1.0 along a line (s .935 -> .86; .937 -> .88-.89; .939 -> .90-.92;
.941 -> .92-.94). A flat read from W .5 to W .6 means the fleet is nearly as strong as the parent; at solo .927 the
blend falls by ~.002 from W .5 to W .6 at any r (r .80: .9451 -> .9432; r .85: .9427 -> .9408), so the two reads
cannot both be .945. The old "r .79" was the least-squares compromise of an inconsistent fit at
.927 (`lb_proxy_model.md` 2.1 already flagged "no r fits both reads").

**Why his point is better than ours (MEASURED where tagged, else GUESS):**
1. **Labels, for the r axis.** Our mix50 member's r (.92) is ~.06 above his line at equal strength. The OOF says
   labels move r by .054 (mix50 vs v4, same recipe/seed, MEASURED) and recipe by .005-.011. If his labels are pure
   report labels, his single members should sit where our v4 members sit (r ~.87-.88), and a 10-checkpoint fleet at
   r_mm ~.95 lifts that to ~.89-.90 (MODEL), which puts him at **solo ~.938-.939** on his line.
2. **Strength, for the solo axis.** At r ~.88-.90 he is ~+.012-.016 stronger than our v4 TTR member (.923). About
   +.004-.005 of that is 5 folds x 2 (MODEL); the rest (~+.007-.010 per member) is recipe and label quality (GUESS:
   forum consensus that "labels are the bottleneck", Raymond .938 -> .949 and Yann +.015 from label work, CLAIMED).
3. **Not the crop, not TTA.** He serves with no TTA at all. Our crop members sat at r ~.91 on the LB
   (`lb_proxy_model.md` 2.1), so crop is not a decorrelator.

So the honest one-liner: **he has v4-like correlation with mix50-like (or better) strength; we have either one or the
other.**

## 4. Fold-0 OOF structure (MEASURED on the box, `oofx.log`, 867 val studies, vs steven_v4 >= .5)

Members (best tag), AUC vs v4, and parent-likeness vs the Raptor proxy (Spearman / within-class latent rB / effective
r fitted to measured rank blends):

| member | AUC vs v4 | sp | rB | r_eff vs Raptor proxy |
|---|---|---|---|---|
| F1 mix50 raw | .8811 | .889 | .851 | **.911** |
| TTR mix50 raw (J1) | .8836 | .896 | .862 | **.916** |
| F1 gated raw | .8734 | .872 | .823 | .887 |
| TTR v4 ema | .8764 | .850 | .801 | **.868** |
| F1 v4 raw | .8692 | .840 | .782 | .857 |

Member-to-member effective r_mm (fitted to measured blends at W .1-.5): F1m-TTRm .948, F1m-gated .947, v4F1-gated
.942, TTRv4-TTRm .934, F1m-F1v4 .927, F1m-TTRv4 .920, F1v4-TTRv4 .917, gated-TTRm .915, gated-TTRv4 .913,
F1v4-TTRm .900. Same-run raw vs EMA: .985-.995 (duplicates).

Measured ensembles (equal rank mean): best is F1m + TTRv4 + TTRm .8856 (+.0020 over best single), F1m + TTRm .8855
(+.0019); all 5 .8841. **Ensembling raises parent-likeness:** F1m + TTRm r_eff vs proxy .924 (members .911/.916).
Mixing in v4 members lowers it (F1m + TTRv4 .904).

**Translation to LB (MODEL):**
- OOF-to-LB d ratio: F1 mix50 d_LB/d_OOF = 1.263, TTR v4 1.232 (MEASURED reads / OOF). TTR mix50 via the mix50 ratio
  -> .934; gold58 says equal to F1 mix50 (.913 vs .914). I use **TTR mix50 member .9325 +- .003** (GUESS).
- The measured 2-member OOF gains (+.0019/+.0024) translate to +.0016/+.0020 LB.
- Fold-to-fold (same recipe, labels) r_mm is not measurable yet (one seed per recipe). Bounds: >= .948 (cross-recipe,
  same labels) and <= .985 (raw/EMA). Used **.94-.965** (GUESS; Tian seeds .941 external).
- k-member gains at r_mm .955, from a .932 member at r .922: k2 .9342 / r .933, k5 .9355 / .939, k10 .9360 / .941,
  k15 .9361 / .942. From TTR v4 (.923, r .882): k5 .9267 / .898, k10 .9272 / .900 (MODEL). Seeds buy ~+.003-.004
  and push r up by ~.017-.02, so a same-family fleet moves almost parallel to the iso-lift curves.
- **Offset OOF r_eff -> r_LB:** calibrated on F1 mix50 (.911 -> .920 point, .895-.945 interval), i.e. additive +.009
  (-.016..+.034), applied to every member with a per-member mapping error of .010-.025 (GUESS). Common offset, so
  member ordering is preserved.

## 5. Label tables: quality vs parent-likeness (MEASURED `labx.log`)

gold58 macro AUC vs radiologist labels; target-level Spearman vs the parent's Raptor proxy over 4,349 non-gold train
studies.

| table | gold58 | sp vs Raptor proxy | sp vs steven_v4 |
|---|---|---|---|
| steven_v4 | .8927 | .653 | 1 |
| mix50_logit | .9262 | **.867** | .872 |
| gated75_logit | .9109 | .724 | .957 |
| report_v2 | .8700 | .621 | .834 |
| gpt56sol | .8352 | .583 | .738 |
| FINAL2 | .8570 | .505 | .587 |
| gpt56sol + report_v2 (rank mean) | .8805 | .625 | .847 |
| widedense teacher | .9127 | .937 | .659 |
| qwen7b, pseudo_v1 | 1.000 (gold rows overwritten: contaminated, unusable on gold58) | .516 / .464 | .616 / .518 |

Read: among pure LLM tables, parent-likeness rises with quality (.835 -> .583, .870 -> .621, .893 -> .653). Only the
teacher-mixed tables jump off that trend (mix50 .867 at .926). So swapping steven_v4 for another public/our LLM table
buys nothing on r; the r cost is entirely the teacher. A better LLM table at ~.91-.92 gold58 would presumably sit at
sp ~.68-.70 (GUESS, extrapolating the LLM trend), far below mix50.

## 6. Fleet composition table (MODEL, `fleetsim_psd.out`; rejection sampler conditioned on K2 + K3; PSD-checked;
the no-PSD run `fleetsim_nopsd.out` agrees within .0001)

Assumptions: TTR fold ~4.8-5 h alone on the V100 (MEASURED: J1 ~5 h; fold 1 now 0.36 s/it), one job at a time,
serving cap ~16 TTR-equivalent members (NEXT_SESSION sec 3, GUESS). W* is chosen BEFORE the read (maximises
posterior-expected true blend). Display probabilities are for the public LB.

| fleet | ready | fleet solo (10/50/90) | fleet r_LB (50) | W* | true blend (50) | P(.943) / P(.944) / P(>=.945) | E[display] |
|---|---|---|---|---|---|---|---|
| F1 mix50 f0 (K2 sanity) | have | .932 | .921 | .15 | .9433 | .84 / .16 / .00 | .9432 |
| TTR v4 f0 alone | have | .923 | .882 | .10 | .9431 | .93 / .07 / .00 | .9431 |
| **A1 5-fold TTR mix50** | 10-06 ~04:00 | .932/.936/.939 | .939 | .20 | .9434 | .51 / .39 / .03 | **.9434** |
| A2 A1 + F1 mix50 | 10-06 | .936 | .942 | .20 | .9434 | .52 / .41 / .02 | .9434 |
| A3 A1 + TTR v4 | 10-06 | .935 | .937 | .20 | .9433 | .57 / .36 / .02 | .9433 |
| A5 all 9 (A1 + F1m + TTRv4 + F1v4 + gated) | 10-06 | .935 | .938 | .15 | .9433 | .73 / .26 / .00 | .9432 |
| B1 A1 + 5-fold TTR v4 | 10-08 | .933 | .927 | .15 | .9433 | .77 / .22 / .00 | .9432 |
| B3 v4-only 10 TTR v4 (2 seeds x 5f) | 10-08 | .927 | .898 | .10 | .9432 | .91 / .09 / .00 | .9431 |
| B5 mix50-only 10 TTRm + F1m | 10-08 | .937 | .943 | .20 | .9434 | .51 / .41 / .03 | .9434 |
| C1 5f TTRm + 5f TTRv4 + 5f 2nd family (v4) | 10-12 | .933 | .923 | .20 | .9435 | .49 / .47 / .01 | .9435 |
| C2 v4-only 10 TTRv4 + 5 2nd family | 10-12 | .929 | .903 | .15 | .9433 | .66 / .33 / .00 | .9433 |
| C5 mix50-heavy 15 TTRm | 10-12 | .936 | .942 | .20 | .9434 | .50 / .39 / .04 | .9434 |
| C6 5f TTRm + 10 TTR on own-OOF silent-fill labels | 10-12 | .934 | .930 | .20 | .9434 | .48 / .38 / .06 | .9434 |
| C7 10 own-OOF-label TTR + 5 2nd family | 10-12 | .932 | .914 | .20 | .9435 | .41 / .41 / .10 | .9435 |
| **H1 NEED: 10 members of (.930, r .86)** | target | .934 | .879 | .30 | .9452 | .00 / .00 / **1.00** | .9451 |
| H2 NEED: 5 H + 5f TTRm | target | .938 | .926 | .35 | .9444 | .05 / .50 / .45 | .9444 |

Second family (EffNetV2-S / ResNet-50 on v4): single .921 +- .005, r = TTRv4 - .02 +- .025, r_mm to ConvNeXt .88-.92
(all GUESS). Own-OOF silent-fill members: .927 +- .004, r = TTRv4 + .015 +- .02 (GUESS).

Requirement curve (MODEL, parent .943, best W): fleet solo needed for a TRUE +.0015 / +.0025 / +.0035:
r .84: .9250 / .9295 / .9332 · r .86: .9284 / .9325 / .9358 · r .88: .9317 / .9354 / .9384 ·
r .90: .9347 / .9381 / .9407 · r .92: .9375 / .9405 / .9427 · r .94: .9402 / .9425 / .9445.

Reading: every row we can actually build lands at a true +.0003-.0005; the differences between A, B and C rows are
smaller than display rounding. The only rows that reach .945 are the H rows, which need members we do not have.

## 7. K4 (10-06): which single submission

Value of information is measured as EVOI for the 10-12 production choice among {C1, C2, C5, B4} and as the drop in
posterior sd of the best final lift (MODEL, `fleetsim_psd.out`).

| candidate | W | display 10/50/90 | E[display] | P(>=.944) | EVOI | sd(final lift) | sd(r TTRv4) |
|---|---|---|---|---|---|---|---|
| SOLO 5f TTR mix50 | solo | .932/.936/.939 | .9357 | 0 | +.00013 | .0005 -> .0004 | no change |
| **P + 5f TTR mix50 at W .20** | .20 | .943/.943/.944 | **.9434** | **.42** | +.00014 | .0005 -> .0004 | no change |
| P + all 9 at W .15 | .15 | .943/.943/.944 | .9432 | .26 | +.00009 | .0005 -> .0004 | no change |
| P + TTR v4 alone at W .40 | .40 | .940/.941/.942 | .9410 | 0 | +.00002 | no change | **.013 -> .007** |
| **P + 5f TTR mix50 at W .50** | .50 | .940/.942/.944 | .9423 | .20 | **+.00016** | **.0005 -> .0003** | no change |

- **Best expected public score:** parent + 5-fold TTR mix50 at **W .20**. A .944 read takes us out of the 944-team
  .943 tie (.944 spans ranks ~289-351 on 10-04, MEASURED `public_intel.md`).
- **Most information:** parent + 5-fold TTR mix50 at **W .50**. At W .20 the read is nearly flat in r; at W .50 it
  moves ~.001 per .025 of r, so it measures the r of the fleet we will actually ship. If the question is only "what
  is r for v4 members", P + TTR v4 alone at W .40 halves that sd, but under this model it does not change any
  production decision.
- Practical slot plan (if 3 slots are free on 10-06 once the 5 folds land ~04:00 and a K4 kit commit runs): W .20
  blend (score), SOLO of the 5-fold fleet, and W .50 blend (r). SOLO + one blend pins r to about +-.025 (same logic as
  K2). Do not ship the 9-member mix unless SOLO shows it beats A1.
- Pre-registered reading of the W .50 read (MODEL, fleet solo .936; shift r by about +-.03 per +-.003 of solo, so use
  the SOLO read): .944 -> r .885-.926 (fleet route alive for H2-type mixes); .943 -> r .906-.948; .942 -> r .927-.969;
  <= .941 -> r >= .948 (the mix50 fleet is a parent copy; stop adding mix50 members).

## 8. What fleet we actually need, and how to get it

- Target (MODEL): fleet ~**.934-.936 at r <= .88**, i.e. ~10 members each ~.930 at member r <= .86. That is roughly
  aastik's point on his line.
- Our v4 TTR member: r ~.88 (MODEL) but .923 solo: **short by ~.007 of strength.** Our mix50 members: .932 solo but
  r ~.92: **short by ~.06 of independence.** Seeds/folds do not close either gap (sec 4).
- Levers that add strength without the parent teacher, in order of evidence (GUESS on size):
  1. **Better report labels** (forum: labels are the bottleneck; our best pure-LLM table is steven_v4 at .893 gold58).
     A table at ~.91-.92 gold58 without a teacher is the single most valuable input. Needs LLM time (box GPU is busy;
     decide with Eesh).
  2. **Self-teacher from our own 5-fold OOF** (Archit/Raymond pattern). The TTR mix50 5-fold OOF lands ~04:00 10-06,
     but it is a mix50 student (r .916 OOF), so as a teacher it re-imports Dread. A clean self-teacher needs **5-fold
     TTR v4 OOF** (4 more folds, ~19-20 h). Modelled as C6/C7: at best +.0005, P(.945+) ~.06-.10.
  3. TTA (mirror / slice offset, +.001-.003 CLAIMED/ledger) and all-data members (+.001, Tian) on the v4 side.
- **Box queue implication:** the running TTR mix50 folds 1-4 produce the best K4 blend member and a full own OOF, so
  letting them finish is reasonable. After that, **more mix50 members have ~zero value** (B5/C5 = A1). The next
  slots should go to TTR v4 folds 1-4 (gives the v4 5-fold fleet + a clean self-teacher), then students on better
  labels. Do not spend box days on mix50 seeds.
- **The merge remains the larger lever.** A partner with an independent fleet (aastik/pj style) is exactly an H-class
  fleet; our parent + their fleet + our mix50 at small W is the only path in this model to a display >= .946.

## 9. Caveats

- One global W and macro-level binormal; per-label routing not modelled (it changed predictions by <= .0009 in
  `lb_proxy_model.md`).
- r_LB for every member other than F1 mix50 is inferred through the OOF r_eff vs ONE Raptor proxy (trained on these
  train studies, so in-sample for Raptor), with a common offset calibrated on K2. If v4 members are relatively more
  DINO-like on the LB than on this proxy, their r is overstated (would help v4 rows).
- r_mm is measured against noisy v4 labels; shared label noise inflates it, so true ensemble gains may be slightly
  larger than modelled (+.001 at most, GUESS).
- aastik's 2 checkpoints per fold may be best+EMA duplicates (effective k ~5); that would make his per-member strength
  even higher on his line.
- TTR mix50 member solo is a GUESS (.9325 +- .003); the 10-06 SOLO read replaces it.

---

## 10. Continuation addendum (2026-10-05 09:15 IST)

### 10.1 aastik solo provenance (MEASURED script text / kernel metadata, `SP/intel/`)
- The ".927 solo" comes from his combine-cell comment: "the same 6 models scored LB 0.927 raw vs 0.924
  rank-averaged". That is an older 6-model set, not the 10 checkpoints his s75 kernels load. The other .928 in
  the script is an efficiency-LB lean entry dated 2026-08-13. **No measured read of the shipped fleet's solo exists**,
  so the sec 3 conclusion (both .945 reads require shipped-fleet solo >= .935) contradicts nothing measured.
- The shipped `ONLY` set is 10 `*_pt_v5_cv*` checkpoints, CV .8961-.9154 (vs his own targets). The same script calls
  ".8905 the best single model in project history" in an older comment, so pt_v5 is a later, higher-CV generation:
  either stronger models or a smoother target (for example pseudo-targets, which are easier to fit). Which one is
  GUESS; "pt" is not defined anywhere in the script.
- s75-w50 `dataset_sources` (MEASURED metadata): a blank first entry (private dataset = his weights, GUESS), the Dread
  Raptor and Mattia CoAt datasets (the parent), and `pilkwang/rsna-knee-llm-labels`. Nothing names a teacher table.
  Consistent with sec 3: LLM-type labels give his v4-like r; his solo edge is strength we have not matched.

### 10.2 gold74 as a fold-to-fold r_mm instrument (MEASURED `g74pairs.log`, box `~/jump2/g74pairs.py`, CPU nice 19)
gold74 is `--exclude`d from every fleet fold, so ANY fold's model is out-of-sample there. This is the only place a
fold-0 and a fold-1 model can be compared on the same studies. Pairwise mean per-label Spearman on gold74 for the 4
members whose gold74 preds already exist (`~/jump2/gold74_preds/`), against the fold-0 OOF numbers of sec 4:

| pair | sp gold74 (boot sd) | sp OOF | r_eff OOF |
|---|---|---|---|
| F1 mix50 / TTR mix50 | .939 (.008) | .941 | .948 |
| TTR v4 / TTR mix50 | .913 (.010) | .935 | .934 |
| F1 mix50 / F1 v4 | .912 (.011) | .931 | .927 |
| TTR v4 / F1 v4 | .904 (.011) | .917 | .917 |
| TTR v4 / F1 mix50 | .882 (.012) | .918 | .920 |
| TTR mix50 / F1 v4 | .879 (.013) | .904 | .900 |

- gold74 sp runs .020 below OOF sp on average and ranks the pairs almost the same way (corr with OOF r_eff .91).
  Map (MODEL, linear fit on these 6 pairs, residual sd .008): **r_mm ~ .924 + .663 x (sp_g74 - .905)**.
  sp_g74 .95 -> .954, .96 -> .961, .97 -> .968, .98 -> .974 (extrapolated above .939, so wider than +-.008).
- **Pre-registered step when fold 1 lands (~13:15 IST):** run `gold_infer.py` on the final
  `f1_s1043_ttr_mix50/*raw*_best.pt` (CPU, nice 19, ~5 min after decode), then `g74pairs.py`. The J1 vs fold-1 pair
  replaces the GUESS r_mm .94-.965 of sec 4 with a measurement (same seed 1043, different fold = exactly the A1
  fleet's structure).
- **Why it does not move K4 (MODEL):** parent + 5-fold TTR mix50 at W .20 reads true .94338-.94349 for r_mm
  .975-.94, k 5 or 10 (fleet solo .9345-.9378, r .931-.948). Higher r_mm makes the fleet weaker but less
  parent-like by the same proportion, so the blend barely moves. The number decides only whether more same-family
  seeds are worth box time (sec 8 says no either way).

### 10.3 Chain timing (MEASURED `~/fleet_runs/f1_s1043_ttr_mix50.log` at 09:06; MODEL for the rest)
Fold 1 at epoch 3 of 24 after 41 min (~12 min per epoch incl. eval, 0.36 s/it). ETA fold 1 ~13:15, fold 2 ~18:05,
fold 3 ~22:55, fold 4 ~03:45 on 10-06, so the A1 K4 kit can be built from ~04:00 10-06 as planned. A
`gold_infer.py` job (nice 19, 12 threads, not started by this track) is running on the box CPU alongside it.

### 10.4 Net change to the bottom line
None in direction. Sec 0 items 1, 2, 4-7 stand as written; item 3 is strengthened (the .927 input was the wrong
fleet). The K4 recommendation stays: **score = parent + 5-fold TTR mix50 at W .20; information = the same fleet at
W .50 paired with its SOLO.**
