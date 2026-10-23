# fresh.md: fresh-eyes hunt for lever categories no prior round examined, private-first (10-20 ~22:15-23:40 IST, Claude)

RESEARCH ONLY. Nothing was submitted, pushed, committed, downloaded or registered; no dataset, no scheduled task, no forum
post or vote. PREREG, HANDOFF, PROGRESS, a20 / a21 / val / night were only read. **No OAI data and no box contact** in this
track: every number below is RSNA train OOF (R7, s1044, s1045) with report labels raked to gold74 expert counts, computed on
the laptop CPU; OAI numbers are quoted from ROUND5 as CLAIMED. No GPU.
Tags: MEASURED (computed here, log named), CLAIMED (prior note named), MODEL (arithmetic on stated inputs), GUESS (judgement).
Units: e-4 = 1e-4 macro AUC (12-label). Ruler = the adj_c12 / skep_stack3 ruler: 3-mean of R7 + s1044 + s1045 OOF, report v2
labels raked to the gold74 expert OA-trio joint, which reproduced the 5 public E-OA reads at ratio 1.01-1.12 (CLAIMED
NIGHT_1020 sec 3). Site proxy = report language (9 groups: en, es, tr, hr, el, de, bg, nl, fr).

## 1. Verdict (plain)

1. **Nothing new to submit.** I read all 9 round summaries and found one lever category nobody examined that matters more for
   private than public: **is the shipped E-OA effect (and its dose choice) stable across sites?** Every prior site lens
   (unseen_sites.md, 10-19) predates E-OA and tested member robustness and offset normalisation only.
2. **Measured: yes, it is site-stable within the train acquisition envelope.** Over 9 sites x 3 models x 2 report readers,
   no heterogeneity test fires (60 Cochran Q tests, min p .125). The dose differences that drive the slate have between-site
   sd tau = 0 (point) with 90 % upper bounds of **1.7-2.3e-4 for pw30 - pw25** and **5.3-6.1e-4 for pw30 - pw20** (MEASURED).
3. **E-OA is a within-site effect, not a site-offset artifact.** On the calibrated ruler, pw30 gains +17.6e-4 on within-site
   pairs vs +8.4e-4 on between-site pairs (v2); +10.6 vs +5.1 (v4) (MEASURED). So a private set from different or more sites
   keeps a positive gain; more site diversity shrinks it by <= ~1e-4 per -.1 of within-site pair share (MODEL).
4. **Implication for the 10-21 slate: no change.** Site heterogeneity is not a reason to override R1, to move B, or to add an
   entry. Small reads: pw25 is never worse than pw30 across sites (FE pw30 - pw25 = -0.5 (sd 0.9) v2, -1.1 (0.7) v4), which
   fits pw25 as the A probe; lomm05 is site-stable (tau 0) and a second reader gives it +1.15e-4 (v2 +2.0), a small
   reader-robustness point NIGHT did not have; it does not answer the C12' skeptic's main objections.
5. **The one private-specific risk left is out-of-envelope sites** (e.g. a low-field site among the unlisted Italy /
   Mongolia contributors). No RSNA ruler exists for it. The OAI analog says model co-firing can collapse 10x more than it
   varies across train sites, which would favour lower doses, but with P(site split) x share the expected cost to pw30 vs
   pw25 is ~0.1e-4 (MODEL). Already hedged by pw25 in the slate and B at dose .2.

## 2. Coverage audit: what the 9 rounds examined vs what they did not

Examined (THINK, TRICK, EX, MORE, ROUND5, PUSH, LEVERS, VAL, NIGHT): output transforms (all 66 pair directions, E-OA shapes /
doses / placements, convex, soft-OR E1, global factor, fluid, menisci, ACL, Synovitis, MCL / Fracture, LOA <- MM), stacking
operators (logit, geometric, median, z-rank, cross-label stacker), member weights and the c4 table knobs, runtime fillers and
TTA, public models and own training, guards (missing planes, flags, NaN, failed-row fills, canary), laterality, site
normalisation / routing / per-site member robustness / atypical acquisitions (pre-E-OA), transductive (duplicates, kNN, AdaBN),
rubric levers, patient covariates, A / B pick hedging, LB slot strategy, forum.

Not examined before tonight (grep of all 10-20 notes for per-site / within-site / site-mix: only forum mentions of 734681):

| # | category | private relevance | done here | verdict |
|---|---|---|---|---|
| F1 | Site stability of the shipped E-OA dose response (and lomm05) | high: E-OA's sign is steep in model co-firing (R7 .824 reads pw30 -10.3e-4, the 3-mean .854 reads +11.0e-4, CLAIMED adj_c12.log); a site-split private could differ | MEASURED, sec 3 | stable; no lever |
| F2 | Within-site vs between-site decomposition of the E-OA gain | medium: private site mix sets the pair mix | MEASURED, sec 3.2 | within > between; robust |
| F3 | Header / OOD-gated E-OA (skip the transform on out-of-envelope studies, e.g. < 1 T or unseen vendor) | only under a site split with out-of-envelope sites | not measurable: train has no < 1.2 T study (CLAIMED unseen_sites.md sec 3); OAI flips sign with the coupling assumption (raw OAI +, gold74-raked -, CLAIMED ROUND5 A.3) | NO: sign unknown, unreadable on public, needs a kernel header pass; E ~0 (GUESS) |
| F4 | Hidden-composition probe (CPU kernel turning a header statistic of the hidden set into a score bit) | info only | not run (would need a slot) | NO: no decision depends on it at the measured tau; LB-probing grey zone |
| F5 | Silent member degradation guard (R6 / S6fall anchor misplacement on unusual anatomy or FOV) | private if unseen sites concentrate there | not measurable without GPU anchor dumps | NO |
| F6 | Rare-label stability of the pending entries | private variance | arithmetic only | nothing to do: every pending entry touches only the OA trio (LOA is the rarest touched, gold74 expert prevalence .20; MCL .15 is untouched, MEASURED gold74 means); B = r6pw20 already hedges c4's rare-label table knobs (Fracture .65, Synovitis .8) through R6's uniform stage 3 |

## 3. F1 / F2 measurement (MEASURED, local CPU, `fresh_site.py`, `fresh_site2.py`, `fresh_site3.py` and their `.log`)

Method: E-OA is applied on the pooled 2,139 (v2) / 3,467 (v4) non-gold train studies, as the kernel applies it over the whole
hidden set; AUC is then scored inside each language site with that site's own raking (site margins, gold74 expert association),
so only the model side and the case mix vary by site. Bootstrap within site (120-300 reps), DerSimonian-Laird random effects,
Q-profile 90 % upper bound on tau. Parity: pooled v2 3-mean reproduces adj_c12 exactly (eoa10 +5.7, eoa20 +9.2, eoa30 +9.3,
pw20 +10.3, pw30 +11.0; lomm05 +2.11).

### 3.1 Per-site heterogeneity (3-mean, the calibrated model)

| quantity | reader | FE mean (sd) | Q, p | tau point | tau 90 % upper | RE mean (sd) |
|---|---|---|---|---|---|---|
| pw30 gain vs no E-OA | v2 | +9.4 (3.6) | 10.1, .26 | 6.9 | 28.7 | +8.7 (5.2) |
| pw30 gain vs no E-OA | v4 | +7.6 (3.0) | 7.4, .50 | 0 | 11.5 | +7.6 (3.0) |
| pw30 - pw20 | v2 | -0.1 (1.5) | 5.0, .75 | 0 | 6.1 | -0.1 (1.5) |
| pw30 - pw20 | v4 | -0.9 (1.3) | 6.3, .61 | 0 | 5.3 | -0.9 (1.3) |
| pw30 - pw25 | v2 | -0.5 (0.9) | 4.1, .85 | 0 | 1.7 | -0.5 (0.9) |
| pw30 - pw25 | v4 | -1.1 (0.7) | 4.8, .77 | 0 | 2.3 | -1.1 (0.7) |
| lomm05 (LOA <- MM a .05 on pw30) | v2 | +2.00 (0.62) | 6.6, .58 | 0 | 2.98 | +2.00 (0.62) |
| lomm05 | v4 | +1.15 (0.48) | 4.6, .80 | 0 | 0.85 | +1.15 (0.48) |

- S6mean and R7 agree: no Q test below p .125 in any of 60 (3 models x 2 readers x 10 quantities). Their dose curves stay
  model-specific as already known (S6mean pw30 - pw20 -3.5 / -3.9; R7 -9.5 / -9.0), with tau <= 4.7 point.
- **Per-site outliers are report-reader noise, not image effects:** bg reads pw30 -54.1 (sd 25.5) on v2 but +11.7 (29.4) on
  v4; el +27.2 vs -12.0; hr +14.3 vs -7.2 (3-mean). Both readers read the same reports, so site-level report conventions are
  not removed by this check; it is an upper bound on what the images could add.
- The protocol-atypical German site (no sagittal FS in 93.5 % of studies, CLAIMED unseen_sites.md; one visible test study has
  its signature) reads pw30 -4.5 (11.2) v2 / -0.1 (9.0) v4: lower than average, inside noise.
- **Co-firing barely moves across train sites:** 3-mean mean-trio co-firing .804-.869 (sd .020); across sites
  corr(co-firing, pw30 - pw20) is -.34 / +.34 / +.28 for 3-mean / S6mean / R7, i.e. no consistent law (MEASURED). For scale,
  the OAI shift was R6 .778 -> .546 and S6fall .845 -> .531 (CLAIMED ROUND5 A.4), more than 10x the train-site spread.
- **Acquisition-atypical studies** (top 20 % / 10 % of trick/work/atypicality.csv, the earlier unseen-site stand-in): 3-mean
  pw30 +13.7 / +10.2 vs typical +11.4; pw30 - pw20 +1.2 (4.1) / -7.9 (7.3) vs +0.9 (1.4). No degradation inside the envelope.

### 3.2 Within-site vs between-site pairs (pooled raking, pooled transform)

| model, reader | pw20 within / between | pw25 within / between | pw30 within / between | pw35 within / between |
|---|---|---|---|---|
| 3-mean, v2 | +14.9 / +8.6 | +16.6 / +8.8 | +17.6 / +8.4 | +16.7 / +5.4 |
| 3-mean, v4 | +10.5 / +7.0 | +11.4 / +6.8 | +10.6 / +5.1 | |
| S6mean, v2 | +9.1 / +5.0 | +9.4 / +4.0 | +8.4 / +2.2 | +5.7 / -1.8 |
| R7, v2 | +4.2 / -3.4 | +2.2 / -8.0 | -1.2 / -14.0 | -7.0 / -22.5 |

Within-site pairs are .29 of the pair mass in train (MEASURED, PF OA). Reads:
- E-OA earns roughly twice as much inside a site as across sites, so it is not riding on site offsets (offset removal loses on
  this ruler, CLAIMED unseen_sites.md sec 6).
- Between-site pairs prefer a slightly lower dose (pw20-pw25). If private holds more between-site pairs than public (more or
  new sites), pw30 - pw25 moves by about -.14 x 1.4 = -0.2e-4 for a .14 drop in within-share (MODEL, v2; v4 -0.1e-4).

### 3.3 What this means for private (MODEL)

- **Random public / private split:** public already prices every site; nothing here changes (as unseen_sites.md sec 1.3).
- **Site split, unseen sites inside the train envelope (1.5 / 3 T, known vendors):** a private dose difference = pooled + a site
  effect with sd <= tau / sqrt(k sites). With tau <= 2.3e-4 for pw30 - pw25 and k >= 2, that is <= 1.6e-4, about the public paired
  read sd of 1.4e-4 (CLAIMED VAL_1020). R1's dose choice stays informative; the E-OA level stays positive in both RE estimates of the calibrated 3-mean (+8.7 / +7.6).
- **Site split with out-of-envelope sites:** no RSNA ruler. Using the OAI c4D5 row as the analog (a .1 +.0003, a .2 -.0014
  OA-mean, about -4e-4 macro per +.1 dose, CLAIMED ROUND5 A.3, linear extrapolation MODEL), with an out-of-envelope share s .15
  and P(site split) .3 (GUESS), E[pw30 - pw25] shifts by about -2e-4 x .15 x .3 = -0.1e-4. On raw OAI coupling the sign flips,
  so even the direction is uncertain. Not decision-relevant.

## 4. Consequences for the pending slate (no edits made; for main / Eesh)

| entry | effect of this track | change? |
|---|---|---|
| r6pw20 (B, dose .2) | dose hedge value unchanged; pw30 - pw20 site tau point 0, upper 5-6e-4 | no |
| pw25 (A probe) | never worse than pw30 per site or per reader (-0.5 / -1.1e-4 FE for pw30 - pw25); fits its minimax-regret role | no; do not override R1 |
| lomm05 (lottery) | site-stable (tau 0; upper .85-3.0e-4); v4 reader +1.15e-4 (sd .48) beside v2 +2.0. Same ruler family, so it does not touch the skeptic's H_prop / gold-like-mix / OAI-member objections | no; stays priority 3 lottery |
| any new entry | none found | no |

## 5. Caveats

- Report language is a proxy for site (English pools at least 5 contributors, CLAIMED unseen_sites.md). 9 groups give low power
  for tau, which is why the upper bounds, not the point zeros, are the useful numbers.
- The ruler has no Raptor member (no OOF exists) and uses report labels raked to gold74's association; per-site report
  conventions survive raking. The 3-mean fits public at ratio ~1 on v2 only; v4 under-reads the higher doses (pw30 +6.3 vs public
  11.3).
- Train sites are all seen by the OOF folds (random folds), so "site" here is "a different site in the mix", not "never seen".
  Out-of-envelope domains remain unmeasured by construction.

## 6. Files (`session_2026-10-20/private/`)

- `fresh.md` (this note).
- `fresh_site.py` -> `fresh_site.log`: per-site co-firing and dose curves (3 models, v2), within / between split, atypical strata,
  lomm05 per site.
- `fresh_site2.py` -> `fresh_site2.log` (+ `fresh_site2.out`): random-effects heterogeneity, 3 models x 2 readers, leave-one-site-out.
- `fresh_site3.py` -> `fresh_site3.log`: Q-profile 90 % tau upper bounds (3-mean, v2 / v4), v4 within / between split.
- Read only: the 9 round summaries, `night/adj_c12.py` / `.log`, `trick/unseen_sites.md`, `trick/work/atypicality.csv`,
  `_data/_reports_lang.csv`, `private/forum_lb.md` (sibling track).
