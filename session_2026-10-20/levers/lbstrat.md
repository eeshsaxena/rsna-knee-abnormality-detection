# lbstrat.md: LB-guided slot strategy for 10-21 / 10-22 (10-20 ~16:05 IST, levers lens "LB slot strategy", Claude)

RESEARCH + MEASUREMENT ONLY. Nothing was submitted, pushed, committed or downloaded. No dataset, no GPU, no box contact, no
scheduled-task edit. PREREG, HANDOFF, PROGRESS, a20 and a21 were only read. No OAI data: every OAI-derived number below is a box
AGGREGATE already on the laptop (push/work/box_abc2_*.log, via abc_local2.run). All new work is local CPU on competition data
(gold58 serves) and our own LB history. The A21 reads (eoa30 57425223, pw20 57425269, pw30 57425970) had NOT landed when this was
written, so the plan is conditional on them.

Tags: MEASURED (computed here or read from a named file), CLAIMED (older note, file named), MODEL (arithmetic / simulation on
stated inputs), GUESS (judgement). Units: macro AUC, "e-4" = 1e-4. "sigma" = the public (390-study) noise sd of that probe's paired
delta. "R1" = A is the API best-id (max public over all our entries).

## 0. Answer (plain)

1. **Slots.** 10-20's five are used. **8 readable slots remain:** 10-21 five (from 05:30 IST), 10-22 about three (last readable
   submit ~10:15 IST).
2. **Use 1, maybe 2. Leave the rest empty.**
   - 10-21 05:31: **R6e20** (the B hedge; push/bpick.md, worth +1.6 to +2.1 pts P(silver), MODEL).
   - At most one optional E-OA fill-in, **pw25**, and only if NO A21 entry displays .960 (P ~.22-.33, MODEL). In that branch it is
     positive in every prior variant: private +0.07 to +0.57e-4 under R1, P(.960) +.07 to +.27 (MODEL, `lbstrat_fill.log`).
     If .960 is already shown, it is worth +0.00 to +0.13e-4 and no display, so skip it (sec 6.2).
   - 10-22: nothing, except the R6e20 no-retry fallback if the 10-21 run errors.
3. **Why stop: our own LB record (MEASURED, sec 1).**
   - Probes backed by a mechanism plus an independent expert ruler: **3 / 3** positive (eoa10, eoa20, and the dose order).
   - Swaps to public-proven parts: 7 / 7 positive.
   - Knob moves (weights / table / A224 / S6 weight): **2 / 11** (P .033 vs a coin).
   - Own members / fleets / TTA / students: **1 / 9** (P .02).
   - Anything backed only by gold58 / gold74 / report OOF: **1 / 10**.
   - Nothing left to probe belongs to the two good classes. The research rounds closed every other mechanism.
4. **The E-OA axis is now in the knob regime.** After A21, every untested shape is a refinement near the optimum.
   - Public-private correlation of such a read is only **.1-.45**, and its sign matches the private sign **56-70 %** of the time
     (MODEL, sec 3).
   - That is the same regime as our knob record (2 / 11). A fill-in can only re-read the same frozen public noise.
5. **E1 soft-OR is negative-EV under our own track record** (MODEL, sec 4). It sits in the weak-ruler class: report-OOF support,
   gold58 neutral. Its R1 value is **-0.46 to +0.09e-4**, not PUSH_1020's +0.1 to +0.75. This reverses PUSH_1020's ticket ranking:
   the pw25 fill-in is the safer ticket. Do not submit E1.
6. **A rule: R1 plus two optional overrides, both free (MODEL, sec 5.3).** Write them into PREREG BEFORE the A21 reads land, or
   skip them. A rule chosen after seeing the reads is a forking path.
   - **O1:** if eoa20 stays the best-id, A = pw20. Gain +0.5 to +1.0e-4, P(private better) .75-.88 over 5 prior variants.
   - **O2:** if eoa30 is the best-id, A = pw30. Gain -0.3 to +1.5e-4, P .51-.85, and the eoa30 Medial OA ceiling haircut favours
     pw30.
   - Value averaged over outcomes: +0.1 to +0.3e-4. No slot needed.
7. **Winner's curse on private (MODEL, sec 5).** Our public score overstates our private score by about **5-14e-4 (central
   ~8e-4)**:
   - E-OA climb plus the A21 pick: 1.7-2.9e-4;
   - the knob layer HW -> a25 -> c4: ~4.6e-4 (EB primary; range 2-9e-4, the low end if part of the knob class's negative mean is
     base-selection luck);
   - heliosli's own public tuning inside HW: not modelled, GUESS 1-2e-4.
   That is about one display step. Most of it sits in the c4 knob layer, which is exactly what B = R6e20 hedges. The B case gets
   stronger, not weaker.
8. **Public rank (secondary).**
   - P(any A21 entry shows .960) is about .70 (CLAIMED, implied.md).
   - The conditional pw25 adds +.02 to +.04 to that overall (+.07 to +.27 inside its branch). A .960 display is worth ~4-9 places.
   - Nothing reaches .961 (needs ~+11e-4).

## 1. Our LB history as data (MEASURED: PREREG_10-13 / 10-14 / 10-15 READ lines, session_2026-10-14/submits.log, trick/work/lead_lb/our_subs2.csv; classes = judgement)

Clean R1 reads only. A read is clean when the base was unchanged while the probe was pending.
- UNREAD / excluded: HF20, HF30, HF-R20, HFR7-R20 (display ties before the frozen-team method), S 57218523, KC1 57249503,
  S6W 57264835, fr65, a15, pf50 (each < c4; their sign vs a25 is unknown).
- Script: `lbstrat_classes.py` -> `lbstrat_classes.log`.

| class | reads (+ = best-id move) | hit | one-sided P vs 50 % |
|---|---|---|---|
| E: mechanism + independent expert ruler (OAI) | eoa10 +, eoa20 +, eoa20 > eoa10 + | 3 / 3 | .125 |
| P: public-proven swap (other teams' notebooks / legs) | HR7 +, HR7-RR +, V224-exact +, V224-FLIP +, S6F +, R6 +, HW + | 7 / 7 | .008 |
| K: knob (weights / table / A224 / S6 weight) | hw3 -, wr45 -, s615 -, **a25 +**, **c4 +** (4-line gold58 stack), c4a15 -, fr100 -, pf10 -, c2max -, c2 -, c3max - | 2 / 11 | .033 |
| M: own members / fleets / TTA / students | HFR7-RR + (tiny), HFR7-RS -, RR-F30 -, RR-F40 -, FF20 -, FLRM -, S6F+S -, COMBO (KC1) -, R7 late - | 1 / 9 | .020 |

- **By evidence source:**
  - weak ruler only (gold58 / gold74 / report OOF): 1 / 10 (c4 is the 1);
  - independent expert ruler: 2 / 2;
  - other teams' public LB: 7 / 7;
  - pure exploration of an already-tuned knob: 1 / 5 (a25).
- **The three .957 drops** (c2max, c2, c3max) are a full display step down from two or three one-line moves that gold58 rated
  +24 to +43e-4 (MEASURED `lbstrat_noise.log`, info only). Our sharpest evidence that gold58 is not a ruler for knobs.

## 2. Public read noise (MEASURED `lbstrat_noise.py` -> `lbstrat_noise.log` / `.npz`)

- **Method:**
  - Paired study bootstrap over gold58 (B 2,000), kernel code path: R6 leg, Raptor r384 / r224 kernel serves, S6fall n42gpu,
    stage-3 table, A224, S6 weight and the E-OA block exactly as in the kernels.
  - Per-label delta covariance, scaled to 390 public / 910 private studies per label with a Hanley 1/P + 1/Q factor.
  - Two prevalence regimes: g = gold58 (expert-like, primary), r = train-report rates.
- **Paired public sd (e-4, g / r):**

| probe vs its base | public sd | probe vs its base | public sd |
|---|---|---|---|
| s615 / wr45 / hw3 (vs HW) | 2.7 / 2.8 / 4.9 (r 3.8 / 3.6 / 6.1) | eoa10 / eoa20 / eoa30 (vs c4) | 2.9 / 4.8 / 6.8 |
| a25 (vs HW), a15, c4a15 | 4.2, 3.6, 3.6 | pw20 / pw25 / pw30 (vs c4) | 6.0 / 7.0 / 7.7 |
| fr65 / pf50 / bk50 / lm40 (vs a25) | 2.3 / 2.4 / 1.0 / 2.7 | **pw30 / pw20 / eoa30 vs eoa20** | **4.5 / 3.3 / 3.8** (box D5: 3.9 / 1.3 / 3.4) |
| fr100 / pf10 / lat10 (vs c4) | 5.5 / 3.6 / 3.0 (fr100 r 10.2) | **pw25 vs pw30** | **2.0** (box 2.0) |
| c4 vs a25 (4-line stack) | 4.5 (r 6.5) | R6e20 vs R6 | 5.5 |
| c2max / c3max vs c4 | 6.8 / 7.4 (r 10.9 / 11.2) | HW vs R6 (heliosli table) | 6.4 |

- **Private sd = .655 x public sd** (median over probes, MEASURED). The two samples are disjoint, so their noises are independent
  given the population.
- **Cross-checks:**
  - report-label OOF at n 390: eoa10 2.0, eoa20 4.4 (CLAIMED r5/noise_corr.log);
  - box D5 at n 390: S20 5.3 (CLAIMED box_abc2_PFT.log).
  - Consistent with the gold58 numbers to within 1.5x.
- **Noise correlation across E-OA strengths:** eoa10 / eoa20 .80 (gold58), .94-.96 (box / OOF); pw25 / pw30 .97.
  - So any new E-OA shape re-reads mostly the same public draw.
  - fr100 / pf10 noise correlation is .06: different labels, independent noise.
- **What a read is:** the API best-id compares at full precision, so it is exact ordering, with zero repeat noise (identical resubmits
  score identically: 56158961 / 56158963 both .913). The noise is the FROZEN draw of the 390 public studies.
  - Re-submitting or re-probing the same axis cannot average it out.
  - Only an independent ruler (OAI, gold74 extra, private) can.
  - This is why LB probing works only for population effects that are large against sigma.

## 3. Public-private correlation of one-parameter moves (MODEL)

| move type | corr(public delta, private delta) | P(private sign = public sign) | source |
|---|---|---|---|
| new knob move (EB fit, sec 4) | .34-.46 | P(true > 0 \| read positive) .28-.46 | `lbstrat_eb_*.log` (4 variants) |
| E-OA vs c4 (eoa10 / eoa20, mechanism scale) | n/a (effect >> noise) | .95-1.00 | `lbstrat_a21.log` (3 ABC variants) |
| E-OA a .3 vs c4 | | .82-.91 | same |
| **A21 alternatives vs eoa20** (eoa30 / pw20 / pw30) | **.40-.44 / .08-.27 / .21-.27** | **.62-.65 / .56-.69 / .60-.68** | same |
| untested fill-ins (pw25 vs pw30, eoa25 vs eoa20) | .27-.40 | .58-.70 | same |

- The correlation of a public read with the private outcome is ~.95+ only when the move is mechanism-sized (E-OA vs c4: population
  ~10e-4 vs sigma ~5e-4).
- For anything near an optimum it is .1-.45, whether it is a table weight or an E-OA shape.
- The A21 read itself picks the private-better shape only ~60-68 % of the time. That is why R1 among A21 entries is worth only
  +1.1 to +1.8e-4 of population gain, against an oracle +2.0 to +2.7 (sec 5.3).

## 4. Empirical-Bayes class priors from our record, and the R1 value of a new probe (MODEL)

### 4.1 Knob class, full fit (`lbstrat_eb.py`)
- **Model:**
  - Each atomic move has a true public-scale effect theta = sigma (m + t z), and its read = theta + the JOINT public noise of
    sec 2 (all moves share the 390 studies).
  - Stacks are EXACT sums of single-label atoms. Each label's AUC depends only on its own column, so c4 - a25 = fr65 + pf50 + bk50
    + lm40, and c2max - c4 = fr100 + pf10, in population AND sample.
- **Data:**
  - Every clean K read with its sign.
  - Display constraints: fr100, pf10 in (-T, 0); c2max, c2, c3max < -T; T = 10 u_c4 e-4, u_c4 ~ N(.76, .07) (CLAIMED implied.md).
  - Magnitudes: a25 - HW 1.5 +- .5, c4 - a25 2.3 +- .8 (CLAIMED picks.md ruler).
- **Fit:**

| variant | m (mean, sigma units) | t (spread) | P(m < 0) | P(new probe reads +) | corr pub / priv | R1 value of a new probe | winner's curse of a win |
|---|---|---|---|---|---|---|---|
| **g noise, c4 mag 2.3 (primary)** | **-0.66 [-1.30, -0.10]** | **.82 [.30, 1.30]** | **.92** | **.32** | **.46** | **+0.02 sigma** | **.87 sigma** |
| r noise, 2.3 | -0.66 | .78 | .90 | .32 | .41 | +0.01 sigma | .89 sigma |
| g noise, c4 mag 1.0 (implied.md u-ruler) | -0.72 | .75 | .95 | .30 | .41 | -0.01 sigma | .94 sigma |
| r noise, 1.0 | -0.70 | .63 | .95 | .29 | .34 | -0.05 sigma | 1.00 sigma |

- **Reading:**
  - A knob probe off a public-selected base reads negative ~70 % of the time.
  - When it does win, about one sigma of its win is noise.
  - Its private value under R1 is about zero: -0.2 to +0.1e-4 for sigma 3.7e-4.
- **Caveat:** part of the negative mean may be BASE-SELECTION noise, not population harm. The bases (heliosli's table, S6 .2, then
  c4) were themselves picked on this public sample, so moving away loses some public-only luck. If ALL of m were public-only,
  m_private = 0 and the accepted knob moves would keep ~40 % of their public gains (shrinkage t^2 / (1 + t^2)).

### 4.2 Sign-only class priors (`lbstrat_classes.log`; hit from Beta(k+1, n-k+1), t .5-1.2)

| class | hit | m (sigma units) | R1 value of a new probe, at E1's sigma 3.7e-4 |
|---|---|---|---|
| K knob | .23 | -0.8 to -1.2 | -0.45 to +0.06e-4 |
| M own members | .18 | -1.0 to -1.4 | -0.46 to -0.05e-4 |
| weak ruler only (E1's class) | .17 | -1.1 to -1.5 | **-0.46 to -0.08e-4** |
| P public-proven | .89 | +1.4 to +1.9 | +4.7 to +6.9e-4 (none available) |
| E mechanism + ruler | .80 | +0.9 to +1.3 | +3.0 to +4.9e-4 (none available) |

- **E1** (square each member's percentile before averaging) has report-OOF support (+.0002 to +.00045, CLAIMED local_mech.log) and
  gold58 -.0001 (CLAIMED e1build.md).
  - That is exactly the weak-ruler class, whose LB record is 1 / 10. Its R1 value is -0.46 to -0.08e-4 there, and -0.2 to +0.1e-4
    under the K-class full fit.
  - e1_option.log's most pessimistic prior was mu -1e-4. Our record implies mu ~ -2.4 to -4.6e-4 for a move of E1's sigma.
  - P(E1 becomes the best-id) ~.2-.3. If it does, P(it is truly better) ~.28-.46, and ~1 sigma (~3.5e-4) of the win is public luck.
  - **Verdict: DEAD for a slot.**

## 5. Winner's curse on private (MODEL)

### 5.1 E-OA climb (`lbstrat_a21.log`; ABC over box aggregates, 3 prior variants)
- **eoa20 vs c4:** public +11.0 to +11.2e-4, population +9.7 to +10.7. Optimism **0.5-1.3e-4**. bpick.md's posterior (+8.3) implies
  ~3e-4.
- **A21 R1 winner vs c4:** public +13.7 to +14.4, population +10.8 to +12.0. Optimism **1.7-2.9e-4**.
  - Of the A21 increment over eoa20 (public +2.5 to +3.3), only +1.1 to +1.8 is population.

### 5.2 Knob layer (`lbstrat_eb_*.log`)

| variant | a25 public -> population | P(private <= 0) | c4 stack public -> population | P(private <= 0) | knob optimism |
|---|---|---|---|---|---|
| primary (g, 2.3) | 1.42 -> **+0.59** | .44 | 2.44 -> **-1.32** | .61 | **4.6e-4** |
| r, 2.3 | 1.37 -> -0.15 | .56 | 2.40 -> -4.53 | .79 | 8.5e-4 |
| g, 1.0 | 1.41 -> -0.34 | .49 | 1.18 -> -3.35 | .81 | 6.3e-4 |
| r, 1.0 | 1.36 -> -1.29 | .72 | 1.24 -> -5.24 | .84 | 9.1e-4 |

- The base-selection caveat (sec 4.1) pulls these toward "keep ~40 % of the public gain", i.e. about +0.6 / +1.0e-4. So read the
  knob layer (a25 + c4 stack) as worth **about -6.5 to +1.6e-4 privately (EB primary -0.7), not the +3.8 it showed in public**.
  GUESS central: ~0.
- **HW over R6** (+3.9e-4 public, heliosli's table) carries heliosli's OWN public-tuning luck. It is not modelled; GUESS 1-2e-4.

### 5.3 R1 vs a posterior rule among the A21 entries (population gain over eoa20, e-4; `lbstrat_a21.log`, `lbstrat_override.log`)

| prior variant | R1 | posterior rule (best-id + .960 flags) | oracle | P(posterior overrides R1) |
|---|---|---|---|---|
| PFT / N01 / logN (primary) | +1.10 | +1.40 | +2.19 | .41 |
| PAREA / N01 / logN | +1.26 | +1.36 | +2.02 | .32 |
| PFT / flat / logU | +1.77 | +2.06 | +2.65 | .25 |

- **Where it overrides, over 5 prior variants:**
  - **O1:** eoa20 stays the best-id -> pw20. pw20 - eoa20 population +0.52 to +1.03e-4, P(private > 0) .75-.88.
  - **O2:** eoa30 is the best-id -> pw30. pw30 - eoa30 -0.26 to +1.54e-4, P .51-.85. The pair weights hold Medial OA at ~.79 a, and
    the eoa30 ceiling haircut (implied.md) is not in the model, so the real O2 case is a bit stronger.
- **No override when pw20 or pw30 is the best-id:** pw20 - pw30 is -0.6 to -1.9e-4. R1 already agrees with the posterior there.

### 5.4 Totals and rank translation
- **A's public increment over the copy block overstates its private increment by about 5-14e-4 (central ~8e-4):**
  - E-OA 1.7-2.9;
  - knob layer ~2-9, central ~4.6;
  - heliosli tuning 1-2 (GUESS).
  - Near the .959 / .960 boundary that is ~4-5 public places per 1e-4 (CLAIMED rank_map.log), so ~25-60 places in absolute terms.
- **Relative to the field it is smaller:** copy teams carry heliosli's table luck too, and heavy probers carry their own. GUESS: a
  private slip of 0-20 places vs today's public order from this source.
- **Direction for B:** bpick.md's model used zero-mean knob priors. Our record says knob moves are about -0.7 sigma on average, so
  c4's private lead over R6 is smaller than bpick assumed. That raises P(R6e20 > A), which is the hedge. It is bpick's "tabOver" row
  (+2.5 pts, CLAIMED). **B = R6e20 is reinforced.**

## 6. The adaptive plan (each submit still needs Eesh's go + its PREREG amendment; I did neither)

### 6.0 Before the A21 reads land (~17:00-18:30 IST), optional, zero slots
- Write **Amendment 21b (posterior tie-break)** into PREREG now:
  - "A = API best-id (A15), except: if the best-id is eoa20 57413368, A = pw20 57425269; if it is eoa30 57425223, A = pw30 57425970.
    Both overrides are fixed before the reads. The other A21 outcomes keep R1."
- If it cannot be written before the reads, skip it. Value +0.1 to +0.3e-4 averaged, +0.5 to +1.5e-4 in the two outcomes it
  touches (P ~.2-.45 combined).

### 6.1 At the A21 read (main session, read-only)
1. Snapshot (`session_2026-10-14/research960/lbprobe/fetch_api_lb.py`).
2. Note the best-id, the 3-decimal display of every A21 entry, and the best's in-band rank.
3. Run `push/work/abc_update.py` (implied.md sec 7) for the refreshed P(.960) of pw25.
4. Apply R1 (+ 21b if written) -> A. B stays as in 6.3.

### 6.2 10-21 slate by A21 outcome (P and values MODEL, 3 ABC prior variants, `lbstrat_fill.log`; R1 values are private-population gains)

| A21 outcome (P) | A (R1) | A with 21b | 10-21 slot 1 (05:31) | optional slot 2 (05:31, needs its own go + amendment) | slots 3-5 |
|---|---|---|---|---|---|
| the best-id displays .960 (pw30, pw20 or eoa30; .67-.78) | best-id | best-id, except eoa30 -> pw30 | R6e20 | **none**: pw25 adds +0.00 to +0.13e-4 and no display | empty |
| pw30 or pw20 best, .959 (.11-.16) | best-id | same | R6e20 | **pw25**: P(moves) .33-.53, private +0.07 to +0.43e-4, P(.960) +.14 to +.27 | empty |
| eoa30 best, .959 (.03-.07) | eoa30 | **pw30** | R6e20 | **pw25**: private +0.11 to +0.57e-4 (R1), P(.960) +.07 to +.10 | empty |
| eoa20 stays (.07-.12) | eoa20 | **pw20** | R6e20 | none (Amendment 21: shape search stops). If Eesh insists: pw25, +0.08 to +0.30e-4, P(.960) +.02 to +.03 | empty |

- Never eoa25 instead of pw25 after a pw winner (R1 value -0.15 to +0.01e-4), and never pw35 (-0.27 to +0.38e-4, negative after a
  pw30 win; MODEL, `lbstrat_fill.log`).

- **pw25:** `push/shape/kernel_v224r6c4pw25j` (sha 64ed7eba). Commit gate: `a21/probe_check_eoa21.py <OUT> --a224 0.25 --table '<C4T>'
  --aoa 0.25 --mode pw` must print PASS 17/17 (CLAIMED PUSH_1020 sec 2.2).
- **R6e20:** `push/kernel_v224r6flips6feoa20j` (sha 2a5ef93a). Commit gate: `a20/probe_check_eoa.py <OUT> --a224 0.35 --table
  '{"Medial OA": 0.5, "Lateral OA": 0.5, "PF OA": 0.5}' --aoa 0.2` must print PASS 17/17 (CLAIMED bpick.md sec 5).
- **Expected value of the whole 10-21 slate** (MODEL):
  - B: +1.6 to +2.1 pts P(silver), +3.5 to +5.2 pts P(private rank <= 167) (CLAIMED bpick.md);
  - A: the optional fill-in adds +0.07 to +0.57e-4 inside its branch, i.e. +0.02 to +0.05e-4 overall (~+0.02 pt P(silver), GUESS
    slope ~0.5 pt per 1e-4 read off bpick.md's A rows), plus P(.960) +.02 to +.04 overall;
  - 21b: +0.1 to +0.3e-4.

### 6.3 B pick (unchanged rule, fixed before its score; draft A22 in bpick.md sec 6)
- B = R6e20 if it is COMPLETE and displays >= .958; otherwise B = R6 57248084.
- R6e20 is never A.
- Do not switch B to an E-OA twin of A (eoa10 / eoa20 / an A21 entry): each is 0.9 to 3.1 pts silver worse (CLAIMED bpick 3.2).
- Considered and dropped:
  - **R6pw20** (R6 + the pair-weighted shape): ~+0.2e-4 of E[max]. Unbuilt, ungated.
  - **a25e20 / HWe20** as B: hedges only part of the knob layer, so it is more correlated with A. Unbuilt. The EB result makes
    "any B without the c4 stack" better, and R6e20 already drops it.

### 6.4 10-22
- **No A probe.** Every remaining class has an R1 value <= +0.15e-4 or below zero (sec 4).
- A 10-22 read lands ~11:00-16:00 IST, with no time to recover from a mis-adoption.
- Only exception: the **R6e20 no-retry fallback at 05:31 IST** if the 10-21 run errored or was not submitted.

### 6.5 Stopping rules (fixed now)
- **S1. Probe class:** a new A probe is allowed only from class E (mechanism + independent expert ruler) or P (public-proven swap).
  None is available, so the default is STOP.
- **S2. E-OA shape:**
  - at most one fill-in after A21 (pw25 only), and only if no A21 entry displays .960 (or eoa20 stays and Eesh insists);
  - never pw35 / pw40 / eoa35 / eoa25 + pw25 pairs (reads are .97 correlated; R1 value <= +0.1e-4, slate_cond.log);
  - a fill-in non-move closes shape;
  - a move switches A once, with no follow-up step.
- **S3. Weak-ruler moves:** no E1, no knob / table move, no gold58-picked anything, no own-member leg (classes M / K / weak ruler,
  R1 value <= 0).
- **S4. Censoring:** a non-move is negative unless a parallel entry moved first (R1 censoring).
- **S5. Timing:** nothing submitted whose read cannot land before 10-22 ~16:00 IST.
- **S6. Selection:** Eesh selects A and B by hand before 10-22 23:59 UTC. Never auto-selection: it would pair two E-OA twins
  (CLAIMED bpick).

## 7. What not to do
- Do not "map the curve" with several E-OA fill-ins. They share one frozen public draw (corr .97), so a second fill-in adds
  <= +0.05e-4 (`lbstrat_a21.log` sec 4).
- Do not treat a winning knob read as evidence. P(true > 0 | it won) is only .28-.46.
- Do not read the c4 / a25 public steps as private gains when judging A vs B (sec 5.2).
- Do not override R1 with a rule chosen after seeing the reads (only 21b, pre-registered).

## 8. Caveats
- **Class assignment is judgement.**
  - HW is "public-proven", although its table was public-tuned by heliosli.
  - c4 is "knob", although it is a 4-line stack.
  - Moving either changes a hit rate by ~.1, not a verdict.
- **The EB uses the structural assumption theta proportional to sigma (m + t z)** and the MODEL magnitude rulers. The negative mean
  rests mostly on the A19 drops and the three HW-base negatives.
  - The base-selection share of that mean is not identifiable from public reads (sec 4.1).
- **The gold58 bootstrap scaled to 390 assumes Hanley scaling.** Its E-OA noise correlations (.80-.84) are lower than the box's
  (.94-.99). The box (n 390 directly) is the better guide to correlation. Both agree on the differential sds within 1.5x, except
  pw20 - eoa20 (3.3 gold58 vs 1.3 box).
- **The A21 posterior rule uses the D5 curve family (OAI-free models on OAI MOAKS knees, coupling-matched).**
  - pw20 > eoa20 holds in ~99 % of its draws. That is a family property, backed only by gold58's +1.8e-4 (n 58, info only).
  - O1's P(private better) .75-.88 is therefore model-conditional.
- **The winner's-curse totals mix MODEL and GUESS.** They are meant for direction (strengthens B), not as a forecast of private
  rank.

## 9. Files (all in session_2026-10-20/levers/)
- `lbstrat_noise.py` -> `lbstrat_noise.log`, `lbstrat_noise.npz`: gold58 joint paired-noise bootstrap, kernel path, 30 comparisons.
- `lbstrat_eb.py` -> `lbstrat_eb_{g,r}_{2p3,1p0}.log`: knob-class empirical-Bayes fit, accepted-move posteriors.
- `lbstrat_classes.py` -> `lbstrat_classes.log`: LB ledger by class, hit rates, sign-only priors, E1 value.
- `lbstrat_a21.py` -> `lbstrat_a21.log`: E-OA winner's curse, public-private correlations, R1 vs posterior rule, fill-in values.
  Reuses `push/work/abc_local2.run` (box aggregates only).
- `lbstrat_override.py` -> `lbstrat_override.log`: P(override better) per A21 outcome, 5 prior variants.
- `lbstrat_fill.py` -> `lbstrat_fill.log`: value of one pw25 / eoa25 / pw35 fill-in per A21 outcome x (.960 shown or not).
- `lbstrat.md`: this note.
