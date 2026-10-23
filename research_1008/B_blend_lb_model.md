# B: blend and leaderboard model (research_1008, dimension B)

Written 2026-10-08 ~22:00 IST by Claude (subagent), read-only. Nothing was pushed, submitted, versioned, forked,
commented or voted on Kaggle. Kaggle reads only: `competitions leaderboard -d`, `competitions submissions --csv`,
`kernels list`, `kernels pull -m` (10 public notebooks). Box not used. Laptop: light numpy only.
Notebook and forum text was treated as data.

Tags: **MEASURED** (source given), **MODEL** (computed from measured inputs), **GUESS** (assumption).

Files (all under `C:/Users/eeshs/Downloads/rsna-knee/research_1008/B_lb/`):
- `rsna-knee-abnormality-detection-publicleaderboard-2026-10-08T15_10_00.csv`: the new LB. `hist/` holds 7 snapshots (10-01 to 10-08).
- `submissions_1008.csv`: our 50 submissions.
- `kernels_score_1008.csv` and `nb/`: top public notebooks, sorted by score, with their metadata.
- `scripts/lb_series.py`, `lb_transitions.py`: LB time series.
- `scripts/b_model.py/.out`, `b_cal2.py/.out`, `b_model2.py/.out`, `b_model3.py/.out`: blend model.
- `scripts/b_shake.py/.out`: public/private Monte Carlo, calibrated against our measured paired OOF SE.
- `scripts/b_rank.py/.out`: private-rank model (GUESS-level).

---

## 0. TL;DR

1. **We are out of the medal zone on public.**
   - At 10-08 15:10 UTC: rank **631 / 5,503** (MEASURED). Bronze (rank 550) is now **.948**, silver (rank 275) is **.950**.
   - The .950 mass point grew from **92 to 293 teams in 11.5 h**, about 20 per hour, still rising (MEASURED, two snapshots).
2. **The .950 cluster is not a P+N blend. It is N + G** (MEASURED, notebook sources).
   - N = nartaa's public OAI-trained .949 CoAtNet.
   - G = goodpjw2008's public 3-fold ConvNeXt 2.5D reader, .929 solo.
   - The blend uses per-label rank weights with mean .162. The top-voted forks (Versia-5 with 304 votes, matterhorn V2, the Apex stacks) attach only these 2 datasets.
3. **A third party measured that our parent adds nothing on top of N** (goodpjw2008, public LB 10-08, MEASURED).
   - 0.943 stack at 40% + N at 60% = **.949**, the same as N alone.
   - So r_PN is ~.945 (MODEL), not the .925 the K9N weights assumed.
   - **K9N-b (.30/.30/.40) is dominated:** about -.001 vs K9N-a at every r_FN. Do not submit it.
4. **K9N-a (.10/.30/.60, built and tested) models at .951** (raw .9513 at the one-factor r_FN .89; range .9497-.9522; subtract ~.0005 for display).
   - The optimum is P .00-.05 / F .35-.40 / N .55-.65. The minimax-regret weights are P .04 / F .36 / N .60.
   - K9N-a is within .0003 of that, which the LB cannot resolve. **Submit K9N-a as built.**
5. **Private is decided against the N band, not by split luck** (MODEL, `b_shake.out`).
   - Public-minus-private SD for one submission is **~.004**.
   - The paired private SD of K9N-a vs the N+G cluster is only **.0003-.0005**. The simulation is calibrated: it gives .0011-.0015 for two single models on 867 studies, against the MEASURED .0014.
   - P(K9N-a private > cluster private) is **1.00 at r_FN .86-.90**, **~.12 at r_FN .94**. It is 1.00 in the case where N's .949 is a selected (optimistic) read.
   - A clean K9 is ~.004 below the band on private: P(beat band) = 0.
6. **Saturday (10-10): 3 submissions, not 5.**
   - (1) **K9N-a** first, for score and the r_FN measurement. (2) **K9 SOLO**: F strength, needed to read r_FN from (1). (3) **K9 main**: the clean ban-insurance candidate.
   - Keep 2 slots in reserve. **Do not send K9N-b or K9-KF.** Weight variants differ by <= .0003 and the LB cannot resolve them.
7. **The next real lever is a parent-free kernel: F + N + G (+ optional N2)** (sec 6).
   - Modelled +.0002 to +.0008 over K9N-a.
   - It scores in ~2-3 h instead of ~7.5 h, and commits cost ~10-15 min instead of ~0.5 h.
   - It is the only safe vehicle for the 320 set (with the parent, K10N would leave a ~0.1 h slow-session margin).
8. **Own-OAI training of F is low value for the blend** (MODEL): +.0000 to +.0010, likely ~+.0003. N already carries the OAI signal on the same 3 labels (PF OA, Lateral OA, Synovitis), so an OAI-trained F would correlate more with N. Not worth 2-4 days unless F solo gains >= .004 with no r_FN rise.
9. **Final picks** (sec 5.4):
   - Pick 1 = the best N blend; when two are within .001, prefer the one with more non-N weight.
   - Pick 2 = a second N blend of a different composition if Eesh's P(OAI ban) is < ~7-12%; otherwise a clean K9.
   - Gold (.959) is out of reach: F would need ~.9535 solo for .955 (MODEL, sec 4.3).
   - The realistic goal is a private finish above the N band, i.e. silver with probability ~.65-.85 (GUESS-level rank model).

---

## 1. Leaderboard now vs the 10-08 03:36 UTC snapshot (MEASURED, `lb_series.py`)

| snapshot (UTC) | teams | top | gold cut | silver cut | bronze cut | >= .950 | at .950 | at .949 | our rank / score |
|---|---|---|---|---|---|---|---|---|---|
| 10-04 09:27 | 5,082 | .963 | .958 (r20) | .945 (r254) | .943 (r508) | 106 | 12 | 17 | 419 / .943 |
| 10-08 03:36 | 5,446 | .964 | .959 (r20) | .949 (r272) | .945 (r544) | 215 | **92** | 85 | **392** / .946 |
| **10-08 15:10** | **5,503** | .964 | .959 (r21) | **.950 (r275)** | **.948 (r550)** | **446** | **293** | 94 | **631** / .946 |

- Cut rules are Kaggle's for 1,000+ teams: gold = 10 + 0.2%, silver = 5%, bronze = 10%.
- The 03:36 numbers reproduce PROGRESS_08-10-2026.md exactly: .959 r20, .949 r272, .945 r544, 92 at .950, us at 392.

**Mass points (MEASURED, count at each score):**

| score | 10-04 | 10-08 03:36 | 10-08 15:10 |
|---|---|---|---|
| .953 | 12 | 14 | 23 |
| .952 | 10 | 18 | 23 |
| .951 | 16 | 14 | 25 |
| .950 | 12 | 92 | **293** (ranks 154-446) |
| .949 | 17 | 85 | 94 (ranks 447-540) |
| .948 | 25 | 38 | 40 (ranks 541-580) |
| .946 | 45 | 46 | 47 (ranks 612-658, us) |
| .945 | 63 | 180 | 131 |
| .944 | 63 | 232 | 193 |
| .943 | 944 | 858 | 783 |

**Where the 293 at .950 came from (03:36 -> 15:10, MEASURED, `lb_transitions.py`):**
- 87 were already at .950.
- 51 came from .945, 46 from .943, 33 from .944 and 31 from .949. 20 are new teams.
- Last-submission hours: 10-24 teams per hour, every hour from 00:00 to 13:00 UTC on 10-08. The rate is not decaying yet.
- Median submission count is 19, i.e. mostly fork users.

**Forecast to 10-22 (GUESS):**
- Teams grow ~100-120 per day (MEASURED rate), giving ~6,900-7,200 at the deadline. Silver is then about rank 345-360 and bronze about 690-720.
- The N-derived band (public .949-.951) reaches ~600-1,200 teams. Public .950 is then about the bronze line and public silver needs .951+.
- Public ranks award nothing. On **private**, the band lands within about +-.001 of one value X (its members share N), and the silver and bronze lines fall inside or just above it. That is the "cliff" in sec 5.

---

## 2. What the .950 cluster is (MEASURED, notebook sources in `nb/`)

**The top 8 public notebooks by score attach only two datasets:** `goodpjw2008/rsna-knee-2-5d-convnext-reader` (G) and `nartaa/rsna-knee-publication-swa-weights-20261007` (N).
- These are evgendvorkin Versia-5 (304 votes), matterhorn3838 V2 (90) and D4, sujanmajhisuzan Apex, hitarthjain0 / rabari9999 Apex V5, and kozykappa triple reader.
- The fusion is `rank((1-w_l) rank(N) + w_l rank(G))` with per-label G weights:
  - Baker's .35, Contusion .30, Medial OA .25, ACL .25, Medial Meniscus .20, MCL .20;
  - Lateral Meniscus, Fracture, Lateral OA and PF OA .08 each;
  - Effusion .04, Synovitis .03.
  - Mean **.162**. The notebooks label this "Official LB 0.950".
- The per-label numbers in their comments match G's gold58 per-label AUCs (Baker's .976, Contusion .966, ...). So the routing was chosen from gold58, which is in-sample for N (its author reused gold58 for selection). That suggests no private advantage beyond a global weight (INFERRED).
- The Apex V5 "Routing 2.0" variants claim .952-.955 as *targets*. These are unverified, and no matching mass point exists on the LB.

**goodpjw2008's own probes** (MEASURED, two of their notebooks, their own submissions):

| blend | public LB |
|---|---|
| N alone (nartaa) | .949 |
| 0.943 community stack (= our parent) 40% + N 60% | **.949** ("the stack adds nothing at this weight") |
| N + their own readers (Raptor 4-fold .933, ConvNeXt 3-fold .929) at 30% | .948 |
| G alone (3 folds) | .929 |
| stack + G at W .15 / .30 / .30 / .30 / .45 | .944 / .944 / .944 / .943 / .942 |

- The stack is not deterministic: one tick of run-to-run variation (.943/.944 at the same config). Our parent may wobble the same way.

---

## 3. Recalibrated correlations (MODEL from MEASURED reads; `b_model.out`, `b_cal2.out`)

**Model.** Macro single-d binormal: AUC_b = Phi(w.d / sqrt(2 w'Rw)).
- It was validated on 11 OOF pairs with error +.0000 to +.0009, one-signed optimistic (verify_blendmath sec 2).
- Every display x means [x-.0005, x+.0005).

| pair | measured reads | r interval (raw model) | point |
|---|---|---|---|
| r_PF (K7i, 10 members) | P .943, F .944, W .50 = .946 | .923-.967 | .945 |
| r_PF (K6-i7, 7 members) | F .942, W .50 = .945 | .924-.967 | .945 |
| r_PF (K5, 5 members) | F .943, W .50 = .946; W .25 = .945 | .912-.956 / .912-.970 | .934 / .941 |
| r_PF (K4i, single J1) | F .939, W .20 = .945 | .871-.939 | .905 |
| **r_PN** (goodpjw) | P .943, N .949, W_N .60 = .949 | **.921-.968** | **.945** |
| **r_GN** (cluster) | N .949, G .929, w .162 = .950 | **.797-.879** | **.838** |
| **r_PG** (goodpjw sweep, joint over W .15/.30/.45) | .944/.944/.942 | **.874-.912** | ~.89 |
| r(N, their readers ~.936 GUESS) | W .30 = .948 | .903-.955 | .929 |

**Reading.**
- Ensembles correlate with P more than single members do: .945 for 7-10 members vs .905 for J1.
- **r_PN is as high as r_PF**, so P is redundant once N is in.
- The old K9N model's r_PN .925 predicted P+N@.6 = .9498, which would display .950-.951. The MEASURED display is .949.

**r_FN is NOT measured.** A one-factor fit of the four measured r's (P ~1.0, N ~.945, F ~.945, G ~.89) gives **r_FN ~.89** and **r_FG ~.84** (MODEL).
- Our mix50 labels carry a Raptor teacher, which pulls r_FN up. Prior band .86-.94, centre ~.89-.90 (GUESS).
- PSD bounds are loose: r_FG is feasible between ~.62 and ~.98.

---

## 4. Public expectations (MODEL, raw model values; subtract ~.0005 for a display read)

### 4.1 Candidates (`b_model.out`, `b_model2.out`). Central: N .949, F .944, r_PN .945, r_PF .945

| candidate P / F / N (/ G) | r_FN .86 | .89 | .92 | .94 | max regret, 264 scen. | note |
|---|---|---|---|---|---|---|
| K9 main .50/.50 | .9460 | .9460 | .9460 | .9460 | .0079 | measured family .946 |
| **K9N-a .10/.30/.60** (built) | .9522 | **.9513** | .9503 | .9497 | .0012 | display ~.951 |
| K9N-b .30/.30/.40 (built) | .9508 | ~.9502 | .9495 | .9491 | .0028 | dominated: DROP |
| MR .05/.35/.60 | .9526 | .9515 | .9504 | .9497 | **.0006** | minimax-regret (grid optimum .04/.36/.60) |
| F+N 0/.40/.60 (parent-free) | .9529 | .9517 | .9504 | .9496 | .0008 | |
| FNG 0/.35/.55/.10 (parent-free) | .9529 | .9518 | .9508 | .9501 | n/a | G at r_FG .84 |
| N+G cluster | .9500 | .9500 | .9500 | .9500 | .0040 | the band, by calibration |
| goodpjw P+N .40/0/.60 | .9490 | .9490 | .9490 | .9490 | | display .949 (calibration point) |

- Per-scenario best weights (.05 grid): **P .00 (range 0-.30), F .40 (.05-.45), N .60 (.50-.75)**.
- **G adds** +.0000 (r_FN .86) to +.0006 (r_FN .94, r_FG .84). It is insurance against a high r_FN.
- **N2** (nartaa's second public checkpoint, the .945 r224 crop, same dataset, ~8-12 min) adds +.0000 to +.0007, depending on r(N, N2) .96 to .92 (unmeasured; `b_model3.out`).
- **Alternative N weights:** N anywhere in .55-.65 is within .0003 of the optimum for r_FN .86-.92. N .40 (K9N-b) costs ~.001. N .70 helps only if r_FN >= .94.

### 4.2 K9N-a read as an r_FN detector (`b_model2.out`, display-equivalent = raw - .0005)

| F solo (K9 SOLO read) | r_FN .84 | .86 | .88 | .90 | .92 | .94 | .96 |
|---|---|---|---|---|---|---|---|
| .943 | .9521 | .9515 | .9508 | .9502 | .9496 | .9489 | .9483 |
| .944 | .9524 | .9517 | .9511 | .9505 | .9498 | .9492 | .9486 |
| .945 | .9527 | .9520 | .9514 | .9508 | .9501 | .9495 | .9489 |
| .946 | .9529 | .9523 | .9517 | .9510 | .9504 | .9498 | .9491 |

So a display of **.952** means r_FN <= ~.86, **.951** means ~.86-.90, **.950** means ~.90-.93, and **<= .949** means >= .93 (or an N-stage problem).

### 4.3 F solo needed for a target (best weights, N .949, after removing .0005 optimism; `b_model.out` sec C)

| r_FN | for .950 | .951 | .952 | .953 | .955 |
|---|---|---|---|---|---|
| .86 | .9355 | .9400 | .9430 | .9455 | .9505 |
| .90 | .9415 | .9450 | .9475 | .9495 | .9535 |
| .94 | .9465 | .9490 | .9510 | .9525 | .9555 |

Gold (.959) and even .955 need an F solo far above anything we can train by 10-20. Every +.001 of F solo is worth ~+.0003-.0004 in the blend.

### 4.4 K10N and own-OAI F (`b_model.out` sec D; best-weight values)

| variant | modelled best | vs K9N-a best (.9513) |
|---|---|---|
| K10N, 5x320 frac75 added/replacing (F .945-.9455, r .90-.905) | .9516 | +.0003 |
| own-OAI F .947 at r .92 | .9515 | +.0002 |
| own-OAI F .948 at r .93 | .9515 | +.0002 |
| own-OAI F .948 at r .90 (OAI does not raise r_FN) | ~.9522 | +.0009 |
| own-OAI F .950 at r .93 | .9525 | +.0012 |

**K10N with the parent does not fit safely** (MODEL, from the K9N runbook runtime model).
- Replacing the 224 set with 320 adds ~+26 min central / ~+34 min slow to the fleet stage.
- K9N's 7.40 / 7.95 h then become ~7.83 / ~8.52 h against the 8.6 h wall, i.e. **~0.08 h slow margin**.
- So 320 should go only into a parent-free kernel.

---

## 5. Private, shake-up and final picks

### 5.1 Split noise (MODEL, `b_shake.py`)

**Simulation inputs.** 1,300 test studies (GUESS), 390 public / 910 private, gold58 prevalences (MEASURED), per-label AUC sd ~.03.

**Calibration.** Two single models at r .93-.96 on 867 studies give a paired-difference SD of **.0011-.0015**. That matches our MEASURED paired OOF SE of ~.0014 (v_ens V3).

| quantity | value |
|---|---|
| SD of one submission's public AUC / private AUC | .0032-.0035 / .0020-.0022 |
| **SD(public - private), one submission** | **.0038-.0042** |
| paired private SD, K9N-a vs N+G cluster | **.0003-.0004** |
| paired private SD, K9 main vs cluster | .0007-.0010 |
| paired public SD, K9N-a vs cluster | .0004-.0006 |

**Consequence.**
- Public reads of different families can be off by +-.004. But N-heavy blends move together, so **our private order vs the band is set by the true difference.**
- The public read of K9N-a vs .950 identifies that difference when it is >= .001: P(public order == private order) is .97-1.00 in those cases.

### 5.2 P(beat the band on private) (MODEL, `b_shake.out`)

| blend | r_FN .86 | r_FN .90 | r_FN .94 | CURSE (N population .9475, r .90) |
|---|---|---|---|---|
| K9N-a: E(priv - cluster) / P(>) | +.0022 / 1.00 | +.0010 / 1.00 | -.0003 / .13 | +.0014 / 1.00 |
| MR .05/.35/.60 | +.0026 / 1.00 | +.0011 / 1.00 | -.0003 / .11 | +.0015 / 1.00 |
| F+N .40/.60 | +.0029 / 1.00 | +.0013 / 1.00 | -.0004 / .07 | +.0016 / 1.00 |
| K9N-b .30/.30/.40 | +.0008 / .93 | -.0001 / .44 | -.0009 / .01 | +.0006 / .91 |
| K9 main (clean) | -.0040 / 0 | -.0040 / 0 | -.0040 / 0 | -.0027 / 0 |
| N alone | -.0010 / 0 | -.0010 / 0 | -.0010 / 0 | -.0012 / 0 |

- If N's public .949 is a selected read (winner's curse, GUESS), blends with **less** N lose less, so our edge over the band grows.
- The band's gold58-chosen per-label routing is a second reason to expect its public to be slightly optimistic (INFERRED).

### 5.3 Private rank (GUESS-level structure, `b_rank.out`)

Inputs:
- 6,900 teams: gold <= 23, silver <= 345, bronze <= 690.
- An N band of 600-1,200 teams, mixed as 50% cluster forks, 20% N-alone / P+N, 30% N + own models.
- 120 own-strong teams above the band and 600 near-miss teams below it.
- Our offset comes from the sec 5.2 curve with r_FN ~ N(.90, .025), plus model error sd .0006.

| pick(s) | median private rank | P(silver+) | P(bronze+) |
|---|---|---|---|
| K9N-a alone | 174-225 | .66-.77 | .84-.97 |
| FN / FNG (parent-free) alone | 151-185 | .75-.84 | .90-.99 |
| best of K9N-a + FNG | 141-169 | .79-.87 | .92-.99 |
| K9 clean alone | 1,026-1,622 | .00 | .00 |
| a plain cluster fork | 415-706 | .09-.34 | .48-.98 |

Gold probability is ~0 in every case.

### 5.4 Final-pick logic (to fix ~10-20, re-check 10-22)

- **Pick 1** = the N blend with the best public read.
  - Tie rule, pre-registered: any public difference < .001 is noise (sec 5.1). Within .001, prefer the candidate with **more non-N weight and the G/N2 legs**. Our edge over the band comes from the non-N legs, and less N exposure loses less if N's public is optimistic.
  - If K9N-a and a parent-free FNG tie, prefer FNG.
- **Pick 2** depends on Eesh's ban probability q (the host bans OAI-derived weights after the deadline and the selected N submissions are voided).
  - In the no-ban world (sec 5.3), a second N blend adds about +.12 silver / +.07 bronze. A clean pick adds 0, because it cannot beat the band.
  - In the ban world, a clean pick at ~.946-.947 ranks high among the remaining teams. P(medal | ban, clean) is GUESS .5-.9, depending on whether Kaggle voids submissions or whole teams (undocumented).
  - **Break-even q ~.07-.12 for bronze, ~.15-.20 for silver** (GUESS).
  - With NDA's "Institutional Approval Required: No" (PROGRESS 20:35), q looks low (GUESS .03-.08). So: **two N blends of different composition** (e.g. K10FNG + K9N-a), unless the host signals against OAI-trained weights before 10-22. Then pick 2 = clean K9 main (or K7i / K5 .946).
- KF (flat parent map) is now irrelevant as a hedge: the parent carries .00-.10 weight in any N blend.

---

## 6. Saturday 10-10 submissions (after Eesh's 05:30 IST quota reset)

Order is chosen so the longest-scoring, highest-value run starts first. Each commit is ~0.5 h of GPU (K9 runbook). Reads come ~7-7.5 h after submit.

| # | submit | kit state | purpose | expected (MODEL) | pre-registered read rule |
|---|---|---|---|---|---|
| 1 | **K9N-a** `eeshsaxena/rsna-sub-k9n-a` (P .10 / F .30 / N .60) | built + tested (K9N runbook sec 4) | **SCORE** (the jump) + **INFO** (r_FN) | ~.951 (.950-.952) | >= .952: r_FN <= .86, raise F to .40-.45 in K10. **.951**: as modelled, keep. .950: r_FN ~.90-.93, K10 must add G (+N2), N .60-.65. <= .949: r_FN >= .93 or an N-stage fault: audit `n949_receipt.json`, `rho_mean`, fallback count before any reuse |
| 2 | **K9 SOLO** `rsna-sub-k9-solo` | built + A0 PASS | **INFO**: F strength, needed to turn read 1 into r_FN (sec 4.2); fleet sanity (288 + frac75 on the LB) | .944-.946 | >= .942 keep the K9 fleet; <= .941 audit (K9 runbook) |
| 3 | **K9 main** `rsna-sub-k9-fleet` | built + A0 PASS | **HEDGE**: the newest clean final-pick candidate (ban insurance); INFO: r_PF for the new fleet | .946-.947 | must be >= .945 to replace K7i / K5 as the clean candidate |
| 4 | reserve | | resubmit after a Kaggle-side scoring error (common this week) | | |
| 5 | reserve, or **K9N-MR .05/.35/.60** only if Eesh wants a 2nd N candidate on the board | ~10 min rebuild (`build_k9n.py --w3_a .05,.35,.60`) + box stub A0 | SCORE (marginal) | +.0002 vs K9N-a, not resolvable | none: never pick weights from this pair's reads |

**Do NOT submit:**
- K9N-b (.30/.30/.40): -.001 modelled, and P(beat band) .44 at r_FN .90.
- K9-KF: it hedges a parent map that no longer matters.
- Any per-label weight variant.

**Ordering rule.** The K9 dataset version must stay the latest until all K9 and K9N commits are done (K9N runbook).

---

## 7. After Saturday: ranked by expected gain per unit effort

| rank | action | expected gain (MODEL) | effort | when |
|---|---|---|---|---|
| 1 | **Submit K9N-a** (sec 6) | +.004-.006 public vs .946; private: from no medal to P(silver+) ~.7 | 1 commit, kit ready | 10-10 05:35 |
| 2 | **Drop K9N-b and KF from the plan**; read K9 SOLO + K9N-a together to fit r_FN | avoids a -.001 candidate and 2 wasted slots | 0 | 10-10 |
| 3 | **Build the parent-free kernel K10FNG**: K5 fleet worker + N child (from K9N cell 32 with P weight 0, ids from the fleet) + G child (goodpjw2008 `infer.py`, Apache-2.0, ~20 min standalone per its author) + optional N2 (same nartaa child with the r224 checkpoint). Weights after the Saturday refit, start F .35 / N .55 / G .10 | +.0002 to +.0008 over K9N-a; reads in ~2-3 h; commits ~10-15 min | ~1 agent day + box A0 | build 10-10/11, first read 10-11/12 |
| 4 | Put the **320 set** (if G320' passes) only into K10FNG (15 members, 3 sets) | +.0002-.0003 | 0 extra once (3) exists | 10-12+ |
| 5 | **Own-OAI training of F**: only if Eesh wants it after the K9N-a read, and only if a probe shows F solo +.004 with no r_FN rise | +.0000 to +.0010, likely ~+.0003 | 2-4 days + data handling | not recommended |
| 6 | Measure nothing more on the LB with weight variants | | | |

---

## 8. OAI links (as verified and logged in PROGRESS_08-10-2026.md 10:40 and 20:35; repeated for Eesh)

- OAI = NDA Collection #2343: https://nda.nih.gov/edit_collection.html?id=2343
- NDA home (sign in): https://nda.nih.gov/
- Login.gov (NDA sign-in identity): https://login.gov/
- The .949 author's NDA study #3407: https://nda.nih.gov/study.html?id=3407
- NIH controlled-data policy (cited in the forum): https://www.cancerimagingarchive.net/nih-controlled-data-access-policy/
- The "Data Permissions" page with "Osteoarthritis Initiative ... Institutional Approval Required: No" is reached from the logged-in NDA dashboard (PROGRESS 20:35). I did not re-verify a direct URL for it, so none is given here.
- Rule context: host 09-21, thread https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819. Open question to the host: thread .../discussion/743416.
- Note for the decision: own-OAI training is modelled as low value for our blend (sec 4.4). The public N weights already bring the OAI signal.

## 9. Caveats

- r_FN is the dominant unknown. Every number that depends on it is shown as a scan, and the K9N-a + K9 SOLO reads will pin it.
- The binormal model is one-signed optimistic by ~.0002-.0009, and per-label heterogeneity between legs is only partly modelled. Real paired SDs could be ~1.5x the simulated ones (GUESS), which would still leave the band order decided by true differences >= .001.
- Third-party reads (goodpjw2008, the cluster) are their own submissions as stated in their notebooks (MEASURED as reported, not re-run by us).
- The rank model in 5.3 is structure-GUESS. Use it for the shape of the decision (the cliff), not as calibrated probabilities.
