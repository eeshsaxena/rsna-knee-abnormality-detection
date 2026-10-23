# Verify B: blend and leaderboard model (adversarial check of B_blend_lb_model.md)

Written 2026-10-08 ~22:15 IST by Claude (verify subagent). Read-only on Kaggle: one `competitions leaderboard -d`,
`datasets metadata`, `datasets files` and a `dataset_list` licence query. No push, submit, version, fork, comment
or vote. Box not used. Laptop: light numpy only (each script runs in seconds). Notebook text was treated as data.

Tags: **MEASURED** (source given), **MODEL** (computed from measured inputs), **GUESS** (assumption).

Files (all under `C:/Users/eeshs/Downloads/rsna-knee/research_1008/verify_B/`):
- `lb/rsna-knee-abnormality-detection-publicleaderboard-2026-10-08T16_34_57.csv`: a fresh LB, 85 min after B's.
- `lbcmp.py`: cut lines and mass points for B's 15:10 snapshot and the fresh one.
- `vb_checks.py`: one-factor r_FN sensitivity, r(N, readers) from goodpjw's third probe, K9N-a / F+N / K9N-b by r_FN, P(beat band) with model error.
- `vb_checks2.py`: consistency of r_GN .84 (cluster) with goodpjw's "N + readers @.30 = .948".
- `vb_checks3.py`: K9N-a and F+N display distribution under r_FN, r_PN, r_PF, rounding and optimism uncertainty.
- `vb_rank.py`: B's rank model re-run with the MEASURED count of teams already above the band.
- `meta/`: G and N dataset metadata.

---

## 0. Verdict summary

| # | B finding | verdict | corrected gain |
|---|---|---|---|
| 1 | Submit K9N-a first on Saturday (~.951) | **revise** (action stands, numbers too optimistic) | public +.004 median (display .950-.951; range +.003 to +.006). P(display >= .951) ~.25-.6. P(true > band) ~.4-.8, not 1.00 |
| 2 | Drop K9N-b and KF; Saturday = K9N-a, K9 SOLO, K9 main | **revise** (drop K9N-b: confirmed; KF reasoning is wrong) | no LB gain. K9N-b is modelled -.0006 to -.0012 vs K9N-a |
| 3 | Build parent-free K10FNG | **revise** (build it, but for speed and runtime safety, not score) | -.0002 to +.0005 vs K9N-a (MODEL). That is ~0 and the LB cannot resolve it |
| 4 | Final picks: pick 2 = second N blend unless P(ban) > ~7-20% | **revise** | P(silver+) for K9N-a ~.2-.5 (GUESS), not .66-.77. Break-even q for a clean pick 2 is ~.03-.10, not .07-.20 |
| 5 | LB dynamics: N band grows, we are out of medals on public | **keep** (re-measured) | none (context) |
| 6 | Own-OAI training of F is low value | **keep** | +.0000 to +.0010, likely ~+.0003 (MODEL) |

---

## 1. Finding 1: K9N-a first (REVISE)

**What holds.**
- The goodpjw2008 stack is our parent (MEASURED, `kernel-metadata.json`):
  - It attaches the same 15 datasets and the same 2 sofiaanjenje kernel sources as the K9 parent (PROGRESS 12:35).
  - Its markdown table (their own submissions, 10-08) reads: N alone .949; stack 40% + N 60% = .949; N + their readers @.30 = .948.
  - So the P+N calibration point is real.
- K9N-a is built and box-tested (K9N_RUNBOOK sec 4: A0n_chain_a K9NBOX PASS 85/85; MEASURED). It is still the biggest available jump. **Submit it first.**

**What is overstated.**
1. **r_FN ~.89 is weakly identified.**
   - The one-factor fit is r_FN = r_PF x r_GN / r_PG (`vb_checks.py` sec 1, MODEL). It moves one-for-one with r_GN:
     - r_GN .80 gives r_FN .85;
     - r_GN .84 gives .89;
     - r_GN .86 gives .91;
     - r_GN .88 gives .93.
   - At r_GN .838 it forces P's loading to 1.00, i.e. P *is* the factor. That is an edge solution, not a central one.
   - r_GN itself (.797-.879) is fitted with a global-weight model to a cluster that uses gold58-chosen **per-label** routing. Any gain from routing is absorbed as "low r_GN", which biases r_FN low and K9N-a high.
2. **A third-party read points to r_FN ~.90-.93.**
   - goodpjw2008 MEASURED: N + their competition-data-only 2.5D readers (Raptor fine-tune .933 + G .929) at .30 = .948.
   - That implies r(N, readers) ~.88-.95 (`vb_checks.py` sec 2) and is hard to square with r_GN .84 (`vb_checks2.py`): at r .84, a .933-.937 reader at .30 would *raise* N to .9506-.9516, but they measured .948.
   - F is the same kind of model: a competition-data-only 2.5D ConvNeXt-T trained on Raptor-teacher mix50 labels. A prior centred ~.90-.915 is better supported than .89 (GUESS).
3. **The display distribution** (`vb_checks3.py`, MODEL). It integrates r_PN .925-.965, r_PF .93-.96, display rounding of N/F/P, and model optimism U(0,.001). The binormal model's MEASURED optimism is +.0000 to +.0009, one-signed.

   | r_FN prior | E[true] | P(disp >= .952) | .951 | .950 | <= .949 | P(true > .9500) |
   |---|---|---|---|---|---|---|
   | N(.89, .025) | .9508 | .20 | .43 | .30 | .07 | .82 |
   | N(.905, .025) | .9503 | .09 | .32 | .41 | .18 | .64 |
   | N(.92, .025) | .9499 | .03 | .21 | .42 | .35 | .42 |

   - So the median display is **.950-.951**, not ".951".
   - Expected gain vs .946 is about **+.004**: +.003 in the bad tail, +.006 in the good tail.
4. **The rank claim depends on rounding** (MEASURED, fresh LB 16:35 UTC; Kaggle breaks ties by earlier submission, so a new entry lands last in its tie):
   - .951 gives rank ~157;
   - .950 gives rank ~475;
   - .949 gives rank ~575.

   B's "~130-160" is only the .951 case.
5. **"P(beat band on private) = 1.00 at r_FN <= .90" ignores model-mean error.**
   - b_shake fixes the blend mean at the model value and adds only split noise (paired SD .0003-.0004).
   - Add r_FN uncertainty, model error sd .0006 and optimism U(0,.001) (`vb_checks.py` sec 4, MODEL/GUESS):
     - r_FN ~ N(.90, .025): P ~.68-.83;
     - centre .915: P ~.51-.69;
     - centre .92 with optimism: P ~.45.
6. **Runtime risk is not priced.** K9N_RUNBOOK sec 0.2 bases the parent on **6.0 h** (one MEASURED session) and kills the N stage at 8.6 h.
   - Third parties report the same stack at **6.5-8 h** (goodpjw stackG quick-start: "6.5 to 8 hours"; goodpjw P+N table: "about 7 h"; MEASURED as reported).
   - At a 7.0 h parent: central 7.0 + .97 + .43 = 8.40 h (survives); slow 7.0 + 1.27 + .68 = 8.95 h (the N stage is killed, so ERROR, with no submission.csv by design). MODEL.
   - P(ERROR) GUESS ~10-25%. The cost is a slot, ~0.5 h quota and ~8 h of wall time, not a bad final pick (a scored run is complete).

**Corrected next action.** Submit K9N-a at 05:35 IST 10-10 as planned. Read it with the table above, not with B sec 4.2 alone. Log the submit and complete timestamps to measure the real parent runtime.

## 2. Finding 2: drop K9N-b and KF (REVISE)

**K9N-b drop: confirmed.**
- K9N-b sits -.0006 to -.0012 below K9N-a at every r_FN from .88 to .94 (MODEL, `vb_checks.py` sec 3).
- At the low end r_PN .921, where P is least redundant, the gap is still -.0006 to -.0009 (sec 3b).
- It wins only if N's true strength is ~.944 or lower (K9N_RUNBOOK stress table).
- The difference is below LB resolution, so the slot buys nothing.

**KF: the reasoning is wrong, the demotion is fine.**
- B says KF is irrelevant because "the parent carries .00-.10 in any N blend". KF was never an N-blend hedge.
- K9_RUNBOOK sec 5 plans KF as the **clean final pick 2** (the parent's outer CoAt map flattened to .60, against a public-split-tuned routing). There the parent carries .50.
- KF matters exactly in the ban world, and only if pick 2 is clean. See finding 4: a clean pick 2 is the better default. So KF should be **deferred, not dropped**: push it once the clean pick is chosen (any day before ~10-20; ~0.5 h quota).

**K9 SOLO: keep, with a smaller claim.**
- It is cheap and needed to interpret K9N-a.
- But as an r_FN detector the pair is coarse. One display tick of K9N-a spans ~.03-.04 of r_FN (B sec 4.2), the F read adds +-.0005, and model optimism adds up to .0009. So the read pins r_FN only to about +-.03 (MODEL).
- A parent-free F+N read (missed lever M1) is a cleaner measurement.

**Slots.** 5 per day for 12 more days means ~60 slots. Slots are not the binding constraint. Eesh's GPU quota (~0.5 h per parent commit) and the ~7.5 h scoring latency are. "Hold 2 in reserve" is harmless.

## 3. Finding 3: parent-free K10FNG (REVISE)

**The gain is ~0, not +.0002 to +.0008** (MODEL).
- F+N (0/.40/.60) vs K9N-a by r_FN: +.0005 at .88, +.0003 at .90, +.0001 at .92, -.0001 at .94. At r_PN .921: +.0001 at r_FN .89 and -.0002 at .92 (`vb_checks.py`).
- G adds +.0000 to +.0006 only at high r_FN, and G is the band's own second leg. It moves us toward the band, not away from it.
- **N2 evidence points to ~0.** prvsiyan's notebook reports its maintainer's private bench "V17 (E3 r224 blend) .949, ties V14 (mirror arm) .949". That is an N + r224 blend scoring the same as N alone (reported, not re-run; a weak but direct read). nartaa's dataset README names the r224 checkpoint as the .945 "efficiency" model (MEASURED, dataset metadata).
- Net: -.0002 to +.0005, within model error and below LB resolution.

**The real value is operational, and it is real.**
- No 6-8 h parent means no N-stage kill risk (finding 1, item 6).
- It reads in ~2-3 h, so the same day can be reused, and commits cost ~10-15 min of quota instead of ~0.5 h.
- It is the only design where 320 members fit with margin. B's arithmetic is right: K10N with parent gives 7.95 + .57 = 8.52 h vs the 8.6 h wall (MODEL).

**Feasibility before 10-20: yes.**
- G is Apache-2.0 (MEASURED, Kaggle dataset licence field).
- G's 3 checkpoints, infer.py and the pylibjpeg wheels are in the dataset (MEASURED, file list). Its author reports ~20 min standalone on the hidden rerun (MEASURED as reported).
- The N child already exists (K9N cell 32). Removing the parent cells is the work: ~1 agent day plus a box CPU A0.

**Corrected action.** Build "K9 SOLO + N stage" (F+N, P weight 0) first. G and N2 are optional legs added later at .05-.10 each. Do not count any score gain from them. Put 320 here only if G320' passes.

## 4. Finding 4: final-pick logic (REVISE)

**Holds.**
- SD(public - private) ~.004 for one submission (MODEL, b_shake). Test size ~1,300 with a 30/70 split is MEASURED (RESEARCH_TO_TOP sec 3), not a GUESS as B tagged it.
- N-sharing blends co-move on private (paired SD .0003-.0005, MODEL).

**Overstated.**
1. **P(silver+) .66-.77 for K9N-a** assumes ~120 teams above the band.
   - MEASURED now: **156** teams at public >= .951. That is up from 123 at 03:36 and 153 at 15:10. They are N + own-model blends doing what we do.
   - Re-running B's structure with the measured groups, our r_FN prior and optimism (`vb_rank.py`, GUESS-level) gives K9N-a P(silver+) **.49** (71+82 above, band 900, r_FN ~.90). It falls to **.18-.35** if the above-band groups grow to 150-250 + 110-130. With r_FN centred at .915: **.10-.31**. FNG gets ~+.05-.10 on top.
   - Gold is ~0 in all cases (agree).
2. **"Second N blend adds ~+.12 silver" overstates diversity.**
   - Two N blends of similar composition differ by ~.0002 in mean, with a paired private SD ~.0003 (MODEL).
   - b_rank simulates the second pick with independent noise, which inflates best-of-two.
   - A realistic best-of-two gain is GUESS ~+.02-.06 silver.
3. **Break-even ban probability.**
   - A 2nd N pick is worth (1-q) x .02-.06. A clean pick is worth q x .5-.9 (P(medal | ban, clean), B's GUESS).
   - Break-even is q ~ **.03-.10**, not .07-.20.
   - The host has been silent since 09-21. The geo-access question (743416) and the "OAI-trained public weights" question are unanswered (PUBLIC949_ASSESSMENT). A q of .05-.10 is defensible (GUESS).
   - So **pick 2 = clean (K9 main or its KF twin) is the robust default**. Switch pick 2 to a second N blend only if the host explicitly OKs OAI-trained weights before 10-22.

**Corrected action.**
- Pick 1 = the best N blend. The B tie rule is fine: within .001, prefer more non-N weight.
- Pick 2 = the clean K9 main or K9-KF, unless the host rules OAI-trained weights legal.
- Eesh states q before 10-20.

## 5. Finding 5: LB dynamics (KEEP)

Re-measured at 16:35 UTC (`lbcmp.py`, MEASURED):

| measure | 15:10 UTC (B's read) | 16:35 UTC (fresh) |
|---|---|---|
| teams | 5,503 | 5,512 |
| .950 mass point | 293 (ranks 154-446) | **318** (ranks 157-474): +25 in 85 min, ~18/h, still not decaying |
| silver line | .950 (r275) | .950 (r275) |
| bronze line | .948 | **.949** (r551) |
| our rank (.946) | 631 | **663** |

- All of B's sec 1 numbers reproduce from its CSV.
- One small correction: the top public notebooks are not *all* "N + G only".
  - romantamrazov DINOsaur V5 (134 votes) attaches the full stack + G + N + a meniscus bag. It "preserves the published Apex 0.950" routing, with a small prior-guided change.
  - rabari's Apex V5 also attaches mattiaangeli.
  - The band's core is still N + G (MEASURED, kernel metadata).

## 6. Finding 6: own-OAI training is low value (KEEP)

- The model numbers check out (B sec 4.4, MODEL): +.0000 to +.0010, likely ~+.0003.
- Feasibility makes it worse.
  - The V100 is committed to Q3r and then the Q3b 320 folds, ~40 h if G320' passes (PROGRESS 18:10).
  - An OAI F needs download, preprocessing and 5 folds at 4-7 h each. That fits before 10-20 only by displacing 320.
  - OAI data must stay on the box.
- Agree: do not start unless the K9N-a read implies a low r_FN AND a probe shows F solo >= +.004.

---

## 7. Levers B missed

1. **M1: a parent-free F+N kernel for Saturday** (K9 SOLO kernel + the K9N cell-32 N stage, P weight 0).
   - It needs no new models: same fleet dataset, same N child. Building it on Friday is ~0.5 agent day plus a box CPU A0.
   - Benefits:
     - It removes the N-stage kill risk.
     - It scores in ~2-3 h.
     - It gives the cleanest r_FN read (one unknown once K9 SOLO is read).
     - It models equal to K9N-a (-.0002 to +.0005).
   - Submit it alongside K9N-a on 10-10. If both score, the pair also measures P's marginal value directly.
2. **M2: measure the parent's real runtime before relying on 8.6 h walls.**
   - Third parties report 6.5-8 h for the stack vs the runbook's 6.0 h.
   - Record submit and complete times for every parent-carrying submission. If they run above ~6.6 h, the slow K9N path is killed.
3. **M3: the above-band race, not the band, now decides silver.**
   - 156 teams are at >= .951 today and the count is rising (MEASURED).
   - Silver at the deadline (rank ~345 of ~6,900) may sit entirely above the band.
   - Weight tuning cannot move us there. Only a stronger or less N-correlated F can: every +.001 of F solo is worth ~+.0003-.0004 in the blend (B sec 4.3, MODEL).
   - So the 320 / frac75 / PF-OA label levers (other dimensions) are the real path, and they belong in the parent-free vehicle.
4. **M4: N alone on our own pipeline as a cheap calibration read.**
   - N's .949 is the best of >= 6 author reads (K9N_RUNBOOK, winner's curse). Our N child scoring N alone costs a few minutes of quota and ~30-40 min of scoring.
   - It pins the N leg's public read in our exact code path (rank parity, fallbacks). Every K9N/FN read is interpreted relative to it.
5. **M5: prior-based per-label N weight** (GUESS +.0003 to +.001; in-sample risk).
   - Raise N on the OAI-covered labels where MOAKS gives N extra supervision (PF OA, Lateral OA, Effusion/Synovitis). Lower it where F/G are strong (Baker's, Contusion).
   - Pre-register the map from OOF and the 16 gold74-only studies, never from public reads.
   - The band's +.001 over N comes partly from per-label routing. B ruled all per-label variants out as "unresolvable", but a single pre-registered map is one submission, not a search.
