# Verify F (intel): adversarial check of F_intel_top_solutions.md (2026-10-08 ~22:00 IST)

Read-only. Kaggle CLI reads only (kernels list, one `kernels pull -m` of romantamrazov/rsna-knee-dinosaur-v5 into SP/verify_F/roman). One WebFetch (MICCAI KJD-25K page). Laptop: light Python (re-ran F_blendmodel.py plus a small extension). Box: ssh timed out at ~21:55 (same as F), nothing ran there. No writes anywhere else.
Tags: MEASURED (source named), CLAIMED (another team's number), MODEL (binormal blend arithmetic), GUESS.

## 0. Summary of verdicts

| Finding | Verdict | Corrected gain |
|---|---|---|
| Intel: LB state / .950 pile | keep (one wording fix) | n/a |
| R1 FN kernel instead of K9N-b | revise | best-sub +.0000 to +.0003 (MODEL, invisible); real value = runtime margin |
| R2 reuse freed runtime | revise | +.000 to +.0005 (MODEL/GUESS) |
| R3 N as teacher (mixN) | **reject** (as specified) | blended -.001 to +.001 (GUESS) |
| R4 own OAI members | revise | blended +.000 to +.002 (GUESS); risk LOW-MODERATE, not LOW |
| R6 VLM relabel | keep (status fix) | +.000 to +.002 blended (GUESS) |
| R5 GLC-lite | revise | +.000 to +.001 SOLO (GUESS) |
| R7/R8 filler | keep | as stated |
| Traps (KJD-25K, Apex) | keep | n/a |
| R9 final picks | revise | n/a |

## 1. What I re-checked and what held

- **LB file** (SP/lb/...2026-10-08T15_27_14.csv, MEASURED): 5,505 teams; us rank 637 at .946; rank 20 = .959, 155 = .950, 275 = .950, 550 = .948; .950 occupies ranks 155-452 (298 teams); .951+ = 154 teams; 409 of 452 teams at .950+ last submitted 10-08. All of F's numbers reproduce exactly.
- **New from the LB file (MEASURED):** our team row lists FOUR usernames: akkigupta2005, eeshsaxena, kmrsuman, swapiy. HANDOFF and RULES say the team is 3. See missed item M6.
- **Top notebooks (MEASURED, kernels list scoreDescending + metadata):** Versia 5, matterhorn V2 and D4, sujanmajhisuzan Apex, kozykappa, hitarthjain0 Apex V5 attach only nartaa weights + goodpjw2008 reader. F missed #2 in score order, romantamrazov/rsna-knee-dinosaur-v5 (135 votes): N + goodpjw reader + the full community stack datasets + a new public dataset `renta0426/rsna-knee-public0033-meniscus-bag-v1`; its own text says ".950 is a goal, not a measured fact". The claim "the 298-team pile IS this recipe" is an inference (GUESS), not measured; the notebook part is MEASURED.
- **goodpjw2008 probes (CLAIMED, notebook markdown, verified in SP/pubnb):** N alone .949; N .6 + community .943 stack .4 = .949 ("the stack adds nothing at this weight"); N .7 + their readers (.933 Raptor FT, .929 ConvNeXt) .3 = .948. Reader dataset description: .929 alone, .944 at 30% with the stack.
- **"Our parent IS that stack": now MEASURED** at the metadata level: the 14 parent datasets + 2 kernel sources + dinov2 small in our K9 kernel-metadata.json are identical to goodpjw2008's Part A sources (code byte-identity not checked).
- **Forum quotes:** hengck23 "OAI changes the competition" (746792, 12:42 UTC), Cody_Null "~450gb" (a question, not a measurement), colum2131 effusion/synovitis table (745949), Georgy 24% (742327), "Now 0.949. still label" (735304; the poster is anonymous in our dump, so the Raymond attribution is GUESS; Raymond separately says teacher pseudo-labels blended 0.5/0.5 were his main lever). All found as quoted.
- **743416 "NDA reply":** it is an image ([img]); its content is unread by us. Only the follow-up "seems reasonable to consider OAI permissible" is text. Treat as CLAIMED, weak.
- **KJD-25K = KneeCoT:** MEASURED (MICCAI page links huggingface.co/datasets/YiHui0124/KneeCoT; "controlled access for approved noncommercial research").
- **Blend model:** F_blendmodel.py re-runs and reproduces every number (MODEL).

## 2. Per-finding verdicts

### R1 FN kernel (F .35 / N .65, no parent) replacing K9N-b: REVISE
- The K9N kit's own model (k9n_weights.out) already put K9N-b (.30/.30/.40) .0006 below K9N-a and the minimax triple at P .08. goodpjw's probe pushes r_PN up (implied .926-.956 across display rounding, MODEL), so P -> 0. FN vs K9N-a: +.0000 to +.0004 (MODEL), below the .001 display step, so the two reads will very likely be identical and carry no information.
- I extended the scan (MODEL): even with a winner's-curse N_true of .9475 or .946 (the reason K9N-b existed as a hedge), K9N-b is below K9N-a in every case, so dropping it loses no real hedge. F weight between .30 and .45 changes the modelled blend by <= .0003 at every r_FN .86-.94: the choice of .35 vs .40 does not matter.
- **The real value is runtime, not score:** K9N slow case ends 7.95 h with a 0.65 h margin to the 8.6 h wall (K9N runbook, MODEL); FN is ~1.5-2.1 h (58-76 min fleet + 26-41 min N, MODEL from the same runbook). That removes the timeout tail and the 15-dataset parent dependency, and frees budget for R2.
- Caveat for private: the parent is a 20+ member diverse ensemble; at .10 it may add private robustness that public cannot show (GUESS, small).
- Expected-score caveat: goodpjw's .933/.929 comp-only readers took N DOWN by .001 at .30. Our fleet is stronger (.944 SOLO) but ConvNeXt-like; if r_FN is .92-.94, K9N-a/FN model .9497-.9505, i.e. a displayed .949-.950 after the ~.0005 optimism. F's ".950 to .951" should read **.949 to .951**.

### R2 reuse the freed runtime: REVISE
- N224 is nartaa's same recipe, same OAI rows, same labels, same seed (746792), so r(N384, N224) is likely >= .95 (GUESS). In a 3-leg blend with the fleet present, N224 adds +.0001 at r .95 and +.0000 at r .97 (MODEL, my scan). F's "+.001" was N+N224 without the fleet.
- Extra fleet sets ~+.0002 each (MEASURED k-curve analog, as F says); offset TTA on the full fleet is unmeasured (the +.003-.006 OOF was one weak member; multi-scale TTA was negative). The 320 set only exists if G320' passes (verdict ~01:30-03:00 10-09) and Q3b ends ~10-10 22:00.
- Corrected gain +.000 to +.0005. Worth doing only because it is cheap once FN exists, and because it is the vehicle for any R4 members later.

### R3 N as a teacher on our 4,407 train studies: REJECT (as specified)
1. **N is in-sample on our whole train set (MEASURED, 746792):** "We trained on 4,349 report-labeled competition studies; the 58 expert-labeled studies were excluded from training but reused for selection"; checkpoint name `raptor_ft_alldata_...`. 4,349 + 58 = 4,407. So N's predictions on our train studies mostly echo its own training targets (steven + pilkwang + JEV blend), not OAI knowledge. HANDOFF measured mild echo for in-sample teachers (.912 vs .886), so it is not worthless, but the "OAI information without downloading OAI" premise does not hold.
2. **mix50 already contains a Raptor-recipe CoAtNet teacher (Dread)**; N is the same family. mixN is mostly "a slightly better Raptor teacher", which our ledger says is a different lever from new information.
3. **Teacher-mixing raises correlation with the teacher and made blend legs worse (MEASURED, HANDOFF 21:50 result: rho .79 -> .83-.84, modelled lift +.0011 -> -.0005).** Here the teacher (N) is itself the biggest leg of the final blend, so any SOLO gain is largely cancelled by higher r_FN.
4. **Gate is compromised:** gold58 was N's selection set (optimistic) and the 16 extra gold74 studies are report-labelled and inside N's training set (HANDOFF line 786: judge on gold58 only).
5. **Compliance coupling:** a mixN fleet carries OAI-derived signal, so the "clean" insurance pick would no longer be clean.
- Corrected: blended -.001 to +.001 (GUESS). If anything is kept, it is PF OA/LatOA/Synovitis-only targets, gold58-only gate, and never in the clean pick. Low priority behind R4.

### R4 own OAI members: REVISE
- **Status is stale in F:** OAI DAR 26647 was APPROVED as an individual at 21:20 (institution empty), and nda-tools 0.7.0 is installed on the box in ~/ndaenv (PROGRESS 21:10/21:20, MEASURED). The next step is Eesh rebuilding the cart and sending the package ID, not "deciding to request access".
- **Rule risk is LOW-MODERATE, not LOW.** The host's 09-21 test (741819) has two parts: no institutional sign-off (now met) AND "generally accessible to all participants". xun1993 (China) reports being "completely blocked" after 5 emails (746792, CLAIMED), and rule 2.6 names geo restrictions explicitly. Training our own fleet on OAI puts OAI inside BOTH our legs (N and F); a ban then removes every non-clean pick.
- **Gain evidence is confounded:** nartaa's +.005 changed rows, optimizer steps and precision path at once ("not compute-matched", 746792). An OAI-trained fleet will also correlate more with N (same external rows, similar labels). Corrected blended gain +.000 to +.002 (GUESS); fleet SOLO +.002 to +.005 (GUESS).
- **Feasibility before the 10-20 freeze:** box disk UNKNOWN; download size UNKNOWN (450 GB was a question); V100 is held by 320 folds until ~10-10 22:00 if G320' passes; 5 folds at 224 with 1.5x rows ~30-45 V100-h (MODEL) plus a gated probe; OAI images may not leave the box, so T4 accounts cannot help with training; one power cut already cost 6.5 h today. Earliest set ~10-15/16 holds only if the data, store and label rules are ready by ~10-12. MOAKS reads exist only for a subset of OAI knees (GUESS), so the labelled-row count may be well under 2,399.
- Next action: Eesh rebuilds the cart with image03 (baseline, knee MRI series only) + MOAKS tables, downloads the tables first to size the job; we write the MOAKS-to-label map (pre-registered) and the converter; decide go/no-go ~10-11 once size, disk and G320' are known. Keep a fully clean fleet (current K9 sets) untouched for the insurance pick.

### Intel (LB state, .950 pile): KEEP
- All LB numbers reproduce. Change "That pile is nartaa's N plus goodpjw2008's reader" from MEASURED to "the top public notebooks are (MEASURED); the pile composition is inferred (GUESS)". Add romantamrazov dinosaur-v5 to the list.

### R6 VLM relabel: KEEP (status fix)
- The pilot is BLOCKED, not running: HTTP 401 at the first call, $0 spent (PROGRESS 21:10). It needs Eesh to run `run_pilot.py` with his own key. Gain unchanged (GUESS).

### R5 GLC-lite: REVISE
- Gold74 is not a clean trusted set: its 16 extras are report-labelled and in-sample (HANDOFF 786); gold58 alone (58 studies, 12 labels x 4 report states) gives very few cases per cell, and our ledger has the site-offset gold trap (-.015). Corrected +.000 to +.001 SOLO (GUESS). Fold only the effusion "small/equivocal = positive" part into the existing frac75-style gate, gold58 only.

### R7/R8: KEEP. Low priority, as F says. N224 leg ~+.0000-.0001 (MODEL, see R2).

### Traps: KEEP. KJD-25K = KneeCoT verified (MEASURED).

### R9 final picks: REVISE
- Add: R3 and R4 both remove the clean pick unless a separate OAI-free fleet is preserved. OAI risk is LOW-MODERATE (geo), so the clean K9 pick (or the KF twin from NOVEL_PLAN sec 5) is still worth one of the two picks unless the host rules yes. P(ban) is a GUESS; the 743416 "NDA reply" is an unread image.

## 3. Missed by F

- **M1. OAI access already approved + nda-tools on the box** (see R4); F's link table is useful but its next action is stale.
- **M2. N is all-data trained (in-sample on our train set)**: this kills the cheap version of R3 and also means N's gold58 is selection-optimistic, so any gold58 comparison of N vs our fleet is biased toward N.
- **M3. Geo-access half of the host's OAI test** is still open (China report, rule 2.6); F calls OAI "effectively cleared".
- **M4. FN's main payoff is timeout-margin and dependency reduction**, which F mentions only in passing; that is the reason to ship it, since the public gain is below display resolution.
- **M5. goodpjw's comp-only readers lowered N by .001**: the most direct evidence on r_FN for a ConvNeXt 2.5D reader; it should cap expectations for K9N/FN at .949-.951 displayed.
- **M6. Team roster:** the LB lists akkigupta2005 as a 4th member (HANDOFF says 3). Eesh should confirm. If real and willing, that account has its own weekly T4 quota (each member must run their own account, rule 3.5a). Submission slots stay 5/day per team.
- **M7. New public leg to audit:** `renta0426/rsna-knee-public0033-meniscus-bag-v1` (used by the #2 public notebook): unknown provenance and score; read its description before anyone uses it.
- **M8. Alternative R4 variant (GUESS):** if OAI is downloaded anyway, a second seed of nartaa's published CoAtNet recipe (hyperparameters are in 746792) on the box is the most likely route to a .945-.949 SOLO member, but r with N would be high (~.95, GUESS), so its blended gain is small; the ConvNeXt + MOAKS route is the more decorrelated one. Do not run both before 10-20.
