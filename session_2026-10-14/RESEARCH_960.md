# RESEARCH_960: final research plan (written 10-14 ~12:15 IST, read-only research workflow)

Inputs: 5 research angles -> synthesis -> 2 adversarial reviews (feasibility; gain/overfit). This file resolves their
disagreements and is the plan of record. No em dashes. Every number is tagged MEASURED / MODEL / GUESS / CLAIMED
(CLAIMED = a third party said it; not verified by us).

State at writing (MEASURED, kaggle CLI 06:16 UTC): 3 of 5 slots used in the 10-14 UTC day (F30-RS 57162815,
RR-F30 57163771, RR-F40 57164843, all PENDING). **2 slots left until 05:30 IST 10-15.** Team best = HFR7-RR 57146261,
public .956, API rank r449 (05:27 UTC) and r457 (05:50 UTC), MEASURED. Box: L13 T123 s1044 f1 running since 10:43 IST,
GPU 100%, 9.0 GB (MEASURED this session).

---

## 0. Bottom line

- **.960 is not reachable with the levers we have.** P(public .960 by 10-22) ~.01 (GUESS). No available lever has a
  MEASURED effect of .001 or more on top of the public .956 blends; the sum of the upper ends of every surviving lever is
  ~+.0008 (MODEL), while .960 needs +.0034 to +.0039 (MODEL).
- **What is still worth winning is medal position.** Silver cut ~r297, bronze ~r595 (MEASURED from team count). We sit at
  r449-457, below the ~125-team V224 copy pile (r290-414) and above the R7-copy pile (r555-685), MEASURED. The piles keep
  growing above us (r379 at 01:37 UTC -> r457 at 05:50 UTC, MEASURED), so doing nothing costs rank every day.
- **Top 3 actions now:** (1) build V224-exact (byte-faithful copy of the public V224 recipe, canary-armed) and submit it
  today as slot 4; (2) build V224-FLIP (R7 anatomical flip TTA switched on, nothing else) and submit it today as slot 5;
  (3) arm the late-release watch (read-only polls every 3 h) and read today's 3 pending entries with the API best-id reader.

---

## 1. Honest ceiling

| Quantity | Value | Tag |
|---|---|---|
| Display rule | scores truncated to 3 dp; .956 means unrounded in [.956, .957) | MEASURED |
| Teams above us inside the .956 band | 263-267 | MEASURED (API, 05:27-05:50 UTC) |
| Our unrounded score | ~.9561 to .9566 | MODEL |
| Needed for .957 / .958 / .960 display | +.0004 to +.0009 / +.0014 to +.0019 / +.0034 to +.0039 | MODEL |
| Sum of corrected lever gains (rebase + flip + r224 mirror) | ~+.0001 to +.0008, centre ~+.0004 | MODEL |
| Late public release (if it lands and is usable) | +.001 to +.002 display for everyone who rebases | GUESS |

Probabilities for the public LB by 10-22 (GUESS):
- **P(.957) ~.20.** About half from our own levers (~.10-.12), half from a usable late release (~.10).
- **P(.958) ~.04.** Own levers alone ~.01 (upper ends would have to stack); the rest is a release.
- **P(.960) ~.01.** Only a release from a .962-class team (David Sarrat's team is at .962, CLAIMED in the forum digest)
  could do it, and nobody above .956 has released anything (MEASURED: no public notebook above .956 as of 10-14).

Why own-lever .960 is out of reach (MODEL, from MEASURED facts): every top-50 hint (azamat1ch ConvNeXt-T 5-fold .959,
colum2131, tennogh, datnt114; CLAIMED) is an own 5-fold family trained for weeks on image-informed pseudo labels. Our own
independent fleet, after weeks, adds a positive but sub-.001 amount on top of the public blends (MEASURED: HFR7-RR beat
HR7-RR; HFR7-R20 ranked above the H copies). heliosli's own team is at .956 r203 (MEASURED), so even the author of every
past public jump does not hold a .957+ model today.

Medal outlook (GUESS, conditional on lever 1 landing us in the V224 pile): P(bronze or better) ~.75-.85, P(silver) ~.25-.35,
P(gold) ~0.

---

## 2. Disagreements resolved

| # | Topic | Synthesizer | Reviewer 1 (feasibility) | Reviewer 2 (gain/overfit) | Decision and why |
|---|---|---|---|---|---|
| D1 | R7 flip TTA gain | +.0003 to +.0007, P(>0) .7 | +.0001 to +.0005, P .55 | +.0001 to +.0004, P .65-.7 | **+.0001 to +.0004 (MODEL), P(>0) ~.6 (GUESS).** Accept R2's derivation: the +.0015-.002 prior is one member, one fold (K5_RUNBOOK sec 4); R7 is already a 4-fold mean; R7's effective blend weight is .4-.5. Accept R1's reason to lower P: heliosli wrote the v4c flip code on 10-11 and shipped R7 on 10-12 with flip_tta false while holding the OOF (MEASURED). Still the best-priced lever: unfitted, in-distribution (R7 trained with mirror .5, MEASURED), one switch. |
| D2 | Flip runtime | +0.1 to 0.6 h | +0.2 to 0.5 h, worst +1.4 h | n/a | **Accept R1.** Decode 37.1 s vs GPU 9.37 s per study (MEASURED receipt) means the flip's extra GPU (~17.5 s, CLAIMED jingshuifu) hides under decode. |
| D3 | r224 mirror TTA | separate entry with box gold58 gate | downgrade; bundle with flip or spare slot | low priority; gold58 = bug check only | **Separate spare-slot entry, no gold58 gate, commit bug check only.** Gain 0 to +.0001 (MODEL; r224 is .175 of the blend). The r224 checkpoint is not on the box (MEASURED by R1) and 58 studies cannot resolve 1e-4. Not bundled with flip, so the flip read stays clean (flip is the read that decides pick A content). |
| D4 | V224 rebase | +40 to +160 ranks, P(bronze) .9 | keep; add wheel-order diagnostic | keep; F is MEASURED positive so the gap is in the base; P(bronze) .75-.85 | **Keep as lever 1 (highest value).** Accept R1's MEASURED cell-order difference (V224 runs r224 after R7's wheel install: timm 1.0.25 / pydicom 3.0.1; ours ran it before: timm 1.0.29 / pydicom 3.0.2) and add a diagnostic commit. Accept R2: HFR7-RR displaced HR7-RR as team best (MEASURED, 10-13/10-14 handoffs), so F .2 is positive on the RR base and the gap to the V224 pile is in our base. Use R2's P(bronze) .75-.85. |
| D5 | Fleet weight w_f rule | bracket {.1,.2}, shrink formula .15 + .4(winner - .15) | n/a | shrink formula is a public-tuned parameter | **Drop the formula and the bracket.** Today's three pending reads decide w_f with a 3-outcome rule (sec 3, lever 1, step 6). w_f counts as 1 of at most 2 public-tuned parameters in pick B. |
| D6 | Release watch | P(usable) .25-.35 | .2-.3 | .15-.25 | **P(usable .957+ release before 10-20 freeze) ~.15-.25 (GUESS).** heliosli's own best is .956 (MEASURED); withheld dreaddevelopment weights are .945 solo (CLAIMED). Keep: cost near zero, value mostly defensive. |
| D7 | R7 OOF / gold58 dump | enabler, 5-8 GPU-h, box CPU fallback | full OOF ~11 h T4x2; box CPU infeasible | run gold58 on box CPU; teammate quota banned | **Accept R1 on compute, R2 on quota.** Box CPU is infeasible: 528.7 windows/study x 43 GFLOP/window, 3.5-3.95 s/window on CPU (MEASURED), so R7 gold58 x 4 folds = ~123k forwards = 10-17 h fp32 (MODEL) to ~120 h at the measured bf16 rate. R2's "gold58 on box CPU" is contradicted by those numbers. Teammate quota is banned (HANDOFF: "No teammate training packs"; "Teammates' Kaggle quota: not to be used (Eesh)"). Result: optional lever 5, only with Eesh's explicit OK, value 1e-5 to 1e-4 (MODEL). |
| D8 | Box Raptor seed on nartaa Gemini labels, then R7-distilled student | lever 5, P(pass) .3 | drop | downgrade to a Vast diagnostic | **Drop as an LB lever.** Accept R2's correction: the synthesizer's "bf16 seeds as short of N as the fp16+ckpt seed" is wrong on our own numbers (S42 SV .9069 vs rented bf16 .9113-.9143, MEASURED: a ~.005 config penalty the V100 cannot avoid). Accept R1's evidence: nartaa's hpo-assets text gives fold-0 gold58 .904 for gemlow and blend (CLAIMED), the same as our seeds, so labels likely do not explain the gap to N (.9229). Even a passing seed lands below N, and below-N Raptor seeds hurt (HFR7-RS .955, MEASURED). The step-B student needs the full R7 OOF (D7) and lands 10-16/17 at best. No Vast labels test either: it is a diagnostic with no path to the LB in 6 days. |
| D9 | jessky d50 bundle | lever 6 | drop | drop | **Drop.** MEASURED: dataset title "RSNA Knee Local CV Compact S6 d50 50-50", 29,244-byte pre-v4c compact_infer: the S6 family that is weaker than R7 (.954 vs .956 with Raptor, MEASURED). Its author's team (zentropy, 187 submissions) sits at r332 and has not converted it above the pile (MEASURED). Gate needs GPU, +2.5-3 h hidden runtime competes with F. |
| D10 | Overall P(.957 / .958 / .960) | .30 / .07 / .01-.02 | .20-.25 / .05 / <=.01 | .15-.20 / .03 / <.01 | **.20 / .04 / .01 (GUESS).** Accept R2's point that the synthesizer summed upper halves; keep a little more release tail than R2. |
| D11 | Runtime stacking | flip + F + student + d50 possible | worst case 9.6 h with student; cap | fix priority order | **Cap the stack at V224 + F + flip.** No third model stage. Expected hidden wall ~5-5.5 h, worst ~7 h, x1.3 slow session ~9.1 h (MODEL). If a measured wall ever says flip + F do not fit, keep the component with the positive public read; if both have one, keep F (MEASURED positive twice). |
| D12 | Read resolution and winner's curse | per-component pile reads | only the team-best entry is localisable | ~20-30% wrong private sign per "verified" component | **Accept both.** A read is binary: did the API best-id move to the new entry. Pick B gets at most 3 components (F, flip, r224 mirror), each needing an unfitted mechanism (or a MEASURED prior) plus a positive public read. Pick A stays a pure copy. |
| D13 | GPU quota | not counted | ~11 min per commit (MEASURED 636 s), needs a ledger | n/a | **Accept.** Plan = ~12-16 commits through 10-21, ~2.5-3 h of Eesh's own quota (MODEL), all of it commits (allowed use). Nothing else on his quota without an explicit OK. |
| D14 | Box queue | preempt L13 for s47/s48 | s47 harmless but not a lever; preempt point ~16:40 | s47 confounded on box | **No preempt.** L13 keeps the box busy (Eesh's never-idle rule) and no surviving lever needs the box GPU. |

---

## 3. Ranked levers that survived review

### Lever 1. Rebase on the exact public V224 recipe, canary-armed, read with the API best-id reader

- **What:** a byte-faithful copy of varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956 stage 3:
  R7 .5 + Raptor_mix .5, Raptor_mix = .65 * pct(r384) + .35 * pct(r224), NOT re-ranked, her stage-2b time guard, and
  **her cell order** (r224 runs after R7's global `pip install --no-index --no-deps` wheels). Our HF fleet cell is present
  behind a flag (OFF in V224-exact). Every later differentiator (flip, F, r224 mirror) sits on this base, one per entry.
- **Why first:** our HFR7-RR (r449-457) sits below all ~125 V224 runs (r290-414) although it adds a MEASURED-positive F on
  a near-identical base (MEASURED). So our base is worse than the public V224 base on the hidden run. Candidates: the
  re-rank (MODEL -3e-5, sd 8e-5, too small), a silent RR_FALLBACK on the hidden run (r384 alone, fires on >3% r224
  failures, self-Spearman < .99 or the 45 min cap; a 3-row commit cannot exercise these, MEASURED from HR7RR_RESULT.md),
  or the timm/pydicom version difference for r224 (MEASURED difference, unknown effect). V224-exact tests the sum.
- **Gain:** public 0 to +.0003 AUC (MODEL), +40 to +160 ranks (MEASURED pile positions). Private: pick A moves from the
  R7 pile into the V224 pile, +1e-4 to +7e-4 (MODEL; r224 private effect +6.7e-4, sd 2.6e-4). Inside the pile our
  position is random (no exact-tie runs longer than 7, MEASURED), centred ~r350: bronze, not silver.
- **Cost:** ~2 h builder work; commits: clean, forced-fallback control, wheel-order diagnostic = 3 x ~11 min of quota
  (MEASURED rate); 1 slot. Hidden runtime ~3.5 h (MODEL).
- **Risks:** the pile may hold variants, not pure copies (MODEL); a canary bug could blank a clean column (covered by the
  control commits); the LB API endpoint is undocumented (keep CSV snapshots too); OAI exposure unchanged (Raptor).
- **First steps (in order):**
  1. `session_2026-10-14/v224/build_v224.py`: copy varshitha cells 9-12 verbatim from scratchpad `varcells/`
     (`fd13/k/varshithakolupuri_rsna-knee-one-unused-checkpoint-lifts-the-0-956`), keep her cell order and her time guard,
     reuse `session_2026-10-13/hfr7/build_hfr7.py`'s HF cell behind `USE_F = False`. Keep every sha assert.
  2. Canary: each fallback path blanks exactly one column to 0.5 (r224 skipped/failed -> Fracture; fleet fallback ->
     Synovitis; HF fallback -> MCL). A canary hit costs ~.025-.04 display (MODEL), so a silent degradation shows up in our
     own submissions list. Safe for final picks: public and private come from the same hidden run.
  3. Gates: (a) re-implement the un-re-ranked blend independently and match the 3 commit rows to 1e-6; (b) compare the
     3-row submission.csv with varshitha's own published commit output (kernels output, read-only) within compile
     tolerance; (c) forced-fallback control commit blanks Fracture exactly; clean commit blanks nothing.
  4. Diagnostic commit (not submitted): dump r224 raw probabilities under both orders (stock timm 1.0.29 / pydicom 3.0.2
     vs the R7 wheels). Max abs diff > 1e-3 = the environment explains part of our RR gap.
  5. Write PREREG_10-14 Amendment 3 (text in sec 4) BEFORE the submit. Submit V224-exact as slot 4.
  6. w_f rule from today's pending reads (best-id after all three score; F30-RS sits on the worse RS base and is
     informative only): best stays HFR7-RR -> w_f .20; best = RR-F30 -> w_f .25 (half-step shrink); best = RR-F40 ->
     w_f .30 (shrunk; do not probe .5). No further F probes, except at most one .15 probe on a spare slot (sim P(pub>0)
     .85 at rho .73 but negative at rho .70, MODEL: model-sensitive, low value).
  7. Point `read_chain.sh` at `research960/lbprobe/fetch_api_lb.py` + `api_frozen.py` (best-submissionId per team; keeps
     .996 of teams frozen vs .67-.81 for the CSV rule, MEASURED). Snapshot after every scoring event.

### Lever 2. R7 anatomical flip TTA (the one R7 inference switch left off)

- **What:** turn on compact_infer v4c `--flip-tta` for the unchanged R7 bundle. Each of the 4 ConvNeXt-small folds also
  scores mirrored windows; mirrored logits are permuted (Medial/Lateral Meniscus swap, Medial/Lateral OA swap), MCL uses the
  plain pass only, passes are averaged in logit space. Nothing is fitted.
- **Evidence:** r7_manifest.json has flip_tta false while R7's run_config has mirror .5 (MEASURED); our compact_infer
  sha 8401812c matches the manifest pin and has v4c flip (MEASURED, R1); jingshuifu's forced-flip commit on the unchanged
  bundle passed 3/3 with logits moving up to +-1.2 on menisci, OA and Contusion (CLAIMED/observed third-party commit; it
  proves the path runs, not an AUC gain).
- **Gain:** public +.0001 to +.0004 (MODEL), P(>0) ~.6 (GUESS); private similar (unfitted, so variance reduction should
  carry). P(display .957 from flip alone) ~.05-.10 (GUESS).
- **Cost:** ~30-60 min on top of lever 1's builder; 1 commit (~11 min quota); 1 slot; hidden runtime +0.2 to 0.5 h
  (MODEL), worst +1.4 h.
- **Risks:** heliosli left it off for an unknown reason (neutral OOF, runtime, or efficiency board); the Effusion fluid
  stacker and Baker's state stacker were fit on plain-pass features; one 390-study public read has paired SD ~1-2e-4
  (MODEL), the same size as the effect.
- **First steps:**
  1. In build_v224.py, a `FLIP` flag: set `manifest['flip_tta'] = True` in memory (jingshuifu pattern, sha asserts still
     pass) or append `--flip-tta` to WORKER_ARGS, whichever v4c honours; assert it was off before; print `FLIP_TTA_FORCED`.
  2. Commit gate: worker log shows `FLIP_TTA_READY` with mcl_plain_pass_only true; receipt 3/3; per-study gpu_s ~1.8x;
     MCL column equals the V224-exact commit within compile tolerance; the other columns differ.
  3. Submit V224-FLIP as slot 5, >= 45 min after V224-exact, under Amendment 3.
  4. Read: best-id = V224-FLIP means flip > exact and > HFR7-RR on public (positive read). Best-id = V224-exact means flip
     <= exact. Neither beats HFR7-RR: flip unobserved; re-test it tomorrow on the F base (V224-F{w} vs V224-F{w}-FLIP).

### Lever 3. Late public release watch with a parameterised swap builder

- **What:** every past frontier jump (.949 -> .954 -> .956) came from a public weight release copied within hours
  (MEASURED). If one lands, rebase picks A and B on it the same day and keep F (+flip) on top.
- **Gain:** P(usable .957+ release before the 10-20 freeze) ~.15-.25 (GUESS); +.001 to +.002 display if it lands
  (GUESS); EV ~+.0002 to +.0004 (GUESS). Mostly defensive: missing one costs hundreds of ranks. A release after 10-20 is
  still usable (it is not our training).
- **Cost:** near zero (read-only CLI polls) + ~1 h to parameterise the builders by bundle slug/version; 1 rebuild
  (~1-2 h) and 1 slot if a release lands.
- **Risks:** a release may depend on private files (heliosli V1 .956 scored .954 with public files, MEASURED); may add
  OAI/fastMRI exposure; heliosli's new-notebook lock ~10-15 23:59 UTC leaves only dataset bumps or edits of existing
  notebooks (the latter CLAIMED unenforced); laptop polls die with the session.
- **First steps:**
  1. Baseline (read-only) file lists/versions of heliosli/rsna-knee-compact-r7cnxfsss,
     heliosli/rsna-orthofoundation-fold0-epoch7-20261004, nartaa/rsna-knee-publication-swa-weights-20261007,
     dreaddevelopment/raptor-knee-widedense, jessky/rsna-knee-local-cv-cnx50; `kaggle kernels list` for heliosli, nartaa,
     varshithakolupuri, jingshuifu sorted by dateRun.
  2. Poll every 3 h; log diffs to `session_2026-10-14/research960/release_watch.log`; re-arm after every session restart.
  3. Extra polls before the lock (10-15 evening IST). Watch for a public flip-TTA score (jingshuifu): a free read for lever 2.
  4. Parameterise build_v224.py by R7 bundle slug + version and Raptor slugs.

### Lever 4. r224 mirror TTA (spare-slot add-on)

- **What:** port the r384 script's `_mirror_tri` + `infer_pair_probs` into raptor224_run.py (mirror before the 0.8 crop).
  r224 runs one view today (MEASURED). Raptor was trained without flips (CLAIMED), but left and right knees make a sagittal
  mirror in-distribution.
- **Gain:** public 0 to +.0001 (MODEL), P(>0) ~.6 (GUESS). Below the paired read SD, so the read is close to a coin flip.
- **Cost:** ~1 h; 1 commit; 1 spare slot; hidden +2 to 6 min (MODEL).
- **Rule:** commit bug check only (outputs change, no NaN, runtime within budget). It enters pick B only if its entry
  becomes team best. Submit only on an otherwise idle slot, after levers 1-2 are read.

### Lever 5 (optional, needs Eesh's explicit OK). R7 plain + flip logit dump for a paired offline A/B

- **What:** compact_infer `--flip-tta --dump-logits` on studies R7 did not train on, giving z_plain and z_mirror_raw per
  fold per study; score plain vs flip offline (paired bootstrap) on gold58 and on OOF vs LLM labels (OOF vs LLM labels
  tracks LB at r .87, gold58 at r .66, MEASURED forum/our tables). Its only decision: whether flip stays in pick B when the
  public read is ambiguous.
- **Value:** 0 direct; 1e-5 to 1e-4 via a better-founded pick (MODEL). Low.
- **Routes (choose one, both need an OK):** (A) one T4x2 session on Eesh's own quota, one hash-fold holdout (~1,090
  studies) + gold58, ~3.1 h (MODEL, R1); (B) box V100 at an L13 fold boundary after downloading the public R7 bundle to
  the box (R7 fold weights are NOT on the box: only anatomy.pt and side_cls.pt, MEASURED this session), on the ~458 raw
  RSNA studies there (MEASURED by R1), ~1 h GPU + 2-3 h engineering (GUESS). Box CPU is not a route (D7).
- **If Eesh declines: skip it.** Flip is then decided by the public read plus its unfitted prior.

---

## 4. Schedule 10-14 to 10-22 (IST; the UTC day resets at 05:30 IST)

### 10-14 Wed (now ~12:15; 2 slots left until 05:30 IST 10-15)
- **12:15-14:30** build_v224.py (lever 1 steps 1-3) + FLIP flag (lever 2 step 1). Release-watch baseline (lever 3 step 1).
- **~13:00-15:00** the three pending entries score (MODEL from ~6 h latency, MEASURED on HFR7-RS). Read with api_frozen.py;
  apply the w_f rule (lever 1 step 6). Prediction (MODEL, sim3): RR-F30 and RR-F40 do not beat HFR7-RR, so w_f stays .20.
- **14:30-16:00** commits: V224-exact clean, forced-fallback control, wheel-order diagnostic, V224-FLIP clean (4 x ~11 min).
- **Before slot 4:** write PREREG_10-14 Amendment 3:
  > Amendment 3. Slot 4 = V224-exact (byte-faithful public V224 recipe, canary armed, gates pass). Slot 5 = V224-FLIP
  > (one change: R7 flip TTA). Expected signs: V224-exact lands in the V224 pile (r290-414) and becomes team best;
  > V224-FLIP >= V224-exact. A win counts only as a best-id move (API reader). Pick A becomes V224-exact once it scores
  > clean; HR7 / HR7-RR retire as pick-A candidates. Pick B = V224 base + at most 3 verified components (F, flip,
  > r224 mirror). w_f from the 10-14 reads by the 3-outcome rule. No per-label public tuning.
- **~16:00 slot 4: V224-exact. ~17:00 slot 5: V224-FLIP** (>= 45 min apart). Reads ~22:00-23:30 (MODEL).
- **One message to Eesh** (sec 7).
- **Box:** no change. L13 f1 runs to ~16:40 (f0 took 5.9 h, MEASURED), then f2. CPU work only with nice 19.
- **Vast:** $0. Laptop: release polls every 3 h (re-arm after restarts).

### 10-15 Thu (5 slots; heliosli new-notebook lock ~05:29 IST 10-16)
- **Morning:** read V224-exact / V224-FLIP (decision table):
  - best-id = V224-exact or V224-FLIP and its rank is inside r290-414: base problem confirmed; RR base retired.
  - best-id unchanged and a displayed score ~.92x: a canary fired (fallback on the hidden run); fix that path and
    resubmit the same kernel once (fallbacks are time-dependent).
  - best-id unchanged, display .956, no canary: our hidden copy sits below the pile; keep HFR7-RR as the B base candidate
    and pick A = HFR7-RR; read the wheel-order diagnostic for a cause.
- **Slots, each pre-registered, in increasing expected value:**
  1. V224-F{w} (F on the new base, w from the 10-14 rule). Re-check of a MEASURED-positive component.
  2. V224-F{w}-FLIP if flip read positive or unobserved (if flip read negative, skip).
  3. Spare: V224-F{w}(-FLIP)-M (r224 mirror, lever 4) on top of whichever of 1/2 is expected best.
  4-5. Unused unless a release lands.
- **Release watch:** extra polls in the evening before the lock.
- **Box:** L13 f3 overnight, f4 ~10:25 (MODEL from 5.9 h/fold). If Eesh OKs lever 5 route B, queue it after f4 (guarded,
  via the main session's qctl).

### 10-16 Fri
- Read the 10-15 entries. Lock the pick-B composition candidate: base + F (+ flip) (+ r224 mirror), max 3 components.
- 1-2 slots at most (e.g., the spare r224 mirror entry if not done, or a release rebase). Unused slots stay unused.
- Lever 5 results (if run): apply the flip rule: flip stays in B if public read positive AND paired offline delta >= 0;
  if public positive but offline delta < -1e-4, flip goes out of B.
- Box: guarded filler of the main session's choice after L13 (no lever depends on it).

### 10-17 Sat and 10-18 Sun
- Release rebases only (if a release lands: one entry = release-exact copy, one = release + F (+flip)).
- Runtime audit of the final B kernel: expected hidden wall < 7 h with every guard in place (MODEL budget 9 h).
- Optional single w_f .15 probe on the final base only if a slot would otherwise be idle (counts as the 2nd public-tuned
  parameter; then nothing else may be tuned).

### 10-19 Mon
- Last new-component entry day. Pre-register the final-pick rules (PREREG Amendment N, text in sec 6).
- 1 h OAI-hedge table: every scored entry with OAI / fastMRI / public-weights flags; name the best scored OAI-free entry.

### 10-20 Tue (12:00 IST own training freeze)
- No new weights from us (none planned). Build and submit any final candidate that lacks its own scored run (picks must
  be our own scored submissions). No identical resubmits. A public release after the freeze may still be used.

### 10-21 Wed
- Every final candidate has a public read. Check forum threads 749145, 748276, 744056 for an OAI / fastMRI /
  public-weights ruling. Spare slots stay unused.

### 10-22 Thu
- Choose the 2 picks before 23:59 UTC (05:29 IST 10-23) per sec 6.

GPU quota ledger (Eesh's own, commits only): 10-14 ~4 commits, 10-15 ~3, 10-16 to 10-20 ~4-8 = ~11-15 commits x ~11 min
= ~2-3 h (MODEL from MEASURED 636 s). Lever 5 route A would add ~3.1 h (only with his OK).

---

## 5. What NOT to do

- Do not chase .960 with a new model family: own 5-fold families on image-informed pseudo labels are not buildable in 6 days
  with one shared V100 and $17 (MODEL); top-50 hints point there (CLAIMED).
- Do not train more ConvNeXt-T / Swin / RadImageNet R50 seeds for the LB, and do not count L13 as a lever (MEASURED null on
  top of the public blends).
- Do not use RSNA-only Raptor seeds (HFR7-RS .955 < .956, MEASURED) or the box OAI Raptor seed line (S42 FAIL, gold58 .907
  vs N .923, MEASURED). No box Gemini-label seed or R7-distilled student (D8). No Vast labels diagnostic (D8).
- Do not add jessky d50 (S6 family) or heliosli S6/compact-v1 as an extra compact member (D9; MEASURED weaker family).
- Do not score compact ConvNeXt-S models on the box CPU (10-120 h per gold58 pass, MODEL from MEASURED per-window cost).
- Do not use teammates' Kaggle quota (Eesh's standing rule), and do not use Eesh's quota for anything but commits without
  his explicit OK.
- Do not stack a third model stage on V224 + F + flip (runtime, D11).
- Do not run identical-kernel resubmits (noise-level ~5e-6 "wins", MODEL) or large-step weight probes (F40 and up), per-label
  weights, label-group coordinate ascent, rank/probit/logit mixes, anti-tie eps, calibration, or a public R7:Raptor weight
  sweep (public-overfit traps; monotone transforms are exactly 0, MEASURED).
- Do not do more text-label work (consensus, negation fixes, VLM relabel, agreement masking, Synovitis-from-Effusion
  borrowing): MEASURED nulls or gold58-fitted single-label rules (MODEL sd 1e-4 to 3e-4 = the claimed effect).
- Do not retrain R7 or R7-lite ($40-110 > $17; private Opus state targets not reproducible).
- Do not build gold58 gates for effects near 1e-4 (58 studies, noise about +-.03 per nartaa, CLAIMED).
- Do not spend the $17.23 Vast credit (MEASURED) without a lever that needs it; none does today. Keep >= $2 reserve anyway.
- Do not let the read chain or release polls run unattended across a session restart without re-arming (they die with the
  session, MEASURED twice).

---

## 6. Final-pick strategy (2 picks, 10-22)

Principle: the private score that counts is max(A, B). A is the floor, B is the upside, and they should differ in exactly
the components whose private sign is uncertain.

- **Pick A = copy floor: V224-exact** (pure copy of the public V224 recipe, clean canary). It lands inside the V224 pile on
  private as on public (MODEL), at a random in-pile position (no long exact-tie runs, MEASURED), which is bronze territory:
  P(bronze or better) ~.75-.85 (GUESS). Fallbacks: if V224-exact never lands in the pile, A = HFR7-RR (our best measured
  entry). If a usable public release lands, A = our clean copy of the highest public pile we have a scored run of.
  Flip does NOT go into A (keeps A a hedge against flip being negative on private).
- **Pick B = differentiated: V224 base + F(w) + flip (+ r224 mirror)**, each component included only if (a) it has an
  unfitted mechanism or a MEASURED prior and (b) its entry moved the API best-id. At most 3 components; at most 2
  public-tuned parameters (w_f, and the optional .15 probe if run). B is the silver bet: silver (~r297) sits just above the
  V224 pile, so B needs a positive private effect from F and/or flip. P(silver) ~.25-.35 (GUESS).
- **OAI / fastMRI ruling risk:** every Raptor entry is OAI-exposed (nartaa's Raptor is OAI-trained; our fleet's DT set used
  OAI rows), so both picks share it. P(an enforced ruling against non-prize teams) ~.05-.15 (GUESS). Keep the 10-19 hedge
  table; switch B to the best scored OAI-free entry ONLY on an explicit host statement against OAI-trained public
  checkpoints (threads 749145, 748276, 744056). Without a statement, keep A and B as pre-registered.
- **Pre-registered final rule (Amendment N, 10-19):** A = V224-exact (or the release-exact copy). B = the best-id entry among
  V224-differentiated entries, provided it satisfies the component rule above; if the best-id is A itself, B = the entry
  with F only (MEASURED-positive component) rather than an untested mix. Never pick an entry whose display shows a canary
  hit.

---

## 7. Decisions needed from Eesh (one message today)

1. OK to use ~2-3 h of your Kaggle GPU quota this week for ~11-15 commits (no training; commits only).
2. Optional lever 5: either ~3.1 h of your quota for one R7 dump session, or OK to download the public R7 bundle
   (heliosli/rsna-knee-compact-r7cnxfsss, size ~0.8 GB, GUESS) to the box for a ~1 h queued job. Saying no is fine; the
   plan works without it.
3. No L13 preempt, no new downloads, no Vast spend, no 2FA renewal needed.

Key paths: builder `session_2026-10-14/v224/build_v224.py` (new); base builder `session_2026-10-13/hfr7/build_hfr7.py`;
gates `session_2026-10-12/hf/h_check.py`, `session_2026-10-13/hr7rr/rr_check.py`; readers
`session_2026-10-14/research960/lbprobe/{fetch_api_lb.py,api_frozen.py}`, `session_2026-10-14/read_chain.sh`; w_f sims
`research960/lbprobe/sim3_070.txt`, `sim3_073.txt`; varshitha cells in the session scratchpad `varcells/` and `fd13/k/`.
