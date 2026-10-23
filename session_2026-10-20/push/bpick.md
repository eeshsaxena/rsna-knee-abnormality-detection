# bpick.md: pick B now that E-OA is LB-confirmed (10-20 ~15:00 IST, push lens, Track 3)

RESEARCH ONLY, CPU only. Nothing was submitted, pushed or committed, and no dataset was created. No download, no GPU, no box, no OAI
data (all OAI inputs are aggregate numbers already in local notes). No scheduled task, PREREG, HANDOFF, PROGRESS, a20 or a21 file was
edited (a21 was only read). One local kernel BUILD (not pushed) under `push/` (sec 5).

**Tags.** MEASURED (computed here or read from a named file), CLAIMED (an older note says so), MODEL (arithmetic on stated inputs),
GUESS (judgement).
**Units.** "Margin" = private macro AUC over the heliosli copy block, x1e-4 (the picks.md / decision.md ruler). "pts" = percentage
points. "Pub-scale" = copy block's public score (.95802, field_mc f959) + margin, i.e. the private score on the public scale with no
common public-to-private shift (that shift is shared by all teams and unknown, sd ~.0055, picks.md).
**Names.** R6 = 57248084. R6e20 = R6 + the exact eoa20 E-OA block at a .2 (built locally, NOT pushed). A21 = R1 winner over {eoa20,
eoa30 57425223, pw20 57425269, pw30 57425970}.

## 0. Answer

1. **Recommend B = R6e20, and spend ONE slot on it.** Fallback B = R6 57248084 if R6e20 errors, is not COMPLETE by selection time,
   or displays .957.
   - A is unchanged: A = API best-id (eoa20 57413368 now, or the A21 winner after tonight's reads).
   - R6e20 is the best B in 27 of 29 sensitivity rows with A = eoa20 and 26 of 29 with A = A21, once S6F's noise artifact is
     removed (sec 4).
2. **Value (MODEL, base).**
   - With A = eoa20, R6e20 vs R6 as B: P(silver) **.890 vs .869 (+2.1 pts)**.
   - P(private rank <= 167, the projected .960+ band) **.600 vs .548 (+5.2 pts)**.
   - E[max margin] **+15.4 vs +14.4e-4**.
   - Every scored entry gives less: S6F .878 is a noise artifact (sec 3.3), then HW .866, eoa10 .860, a25 .859, c4 .856, and A alone
     .848.
   - With A = A21: .898 vs .883 (+1.6), and r167 .657 vs .622 (+3.5).
3. **Why it changed from "B = R6".**
   - E-OA is now strongly confirmed. Posterior d20 is **+8.3e-4 (sd 4.4, P>0 .98)**, so plain R6 sits ~11e-4 below A on private and
     wins only 10 % of the time.
   - R6e20 gets the same E-OA gain: transfer k .9, i.e. +7.5e-4 (sd 4.3) over R6. It keeps the c4-vs-R6 tuning hedge (P(B > A) .30
     vs R6's .10).
   - It gives up only the hedge against E-OA itself, and P(E-OA hurts) is now ~.02-.06.
4. **Slot.** One submit, ideally at 10-21 05:31 IST (push any time tonight after the A21 reads). At the latest, 10-22 05:31 IST: it
   only has to be COMPLETE before selection, but that slot has no retry.
   - Cost: one probe slot for A. The A21 module values three A probes at ~+1.2 pts silver in total, so one slot for B is worth about
     4-5x a marginal A probe (MODEL, GUESS-heavy).
   - Needs Eesh's go plus a PREREG amendment (draft in sec 6). I have not done either.
5. **Headline numbers for the recommended pair (eoa20 + R6e20, MODEL base, with A = A21 winner in brackets):**

   | quantity | eoa20 + R6e20 | (A21 + R6e20) | eoa20 + R6 (status quo) |
   |---|---|---|---|
   | E[max margin] | +15.4e-4 (sd 8.5) | (+17.5e-4) | +14.4e-4 |
   | expected private score, pub-scale | .95956 | (.95977) | .95946 |
   | P(silver) | .890 | (.898) | .869 |
   | P(bronze+) | .955 | (.955) | .948 |
   | P(r <= 200) | .714 | (.751) | .670 |
   | **P(.960+ band, private rank <= 167)** | **.600** | (.657) | .548 |
   | P(r <= 130, today's .960+ band size) | .372 | (.458) | .322 |
   | P(pub-scale private >= .960) | .302 | (.395) | .256 |
   | P(r <= 100) | .117 | (.198) | .094 |
   | gold (r22) | .000 | (.000) | .000 |

   - Range across the 29 sensitivity rows: P(silver) for eoa20 + R6e20 runs .70 (neg prior) to .93 (orig prior / strong S6). Most
     rows sit at .86-.90.

## 1. What changed: the E-OA evidence

### 1.1 LB reads (MEASURED, `session_2026-10-14/research960/lbprobe/api_lb_1020_eoaread1/2.json`, `forum/api_lb_1020T0417.json`)

| snapshot (UTC) | our best | rank | .960 band | .959 band | position |
|---|---|---|---|---|---|
| 10-20 04:17 | c4 57343435 .958 | r217 / 6,249 | r87-127 | r128-193 | 23 teams above us inside .958 |
| 10-20 05:52 | eoa10 57413380 .959 | r167 / 6,253 | r87-130 | r131-196 | 29 of the 65 other .959 teams below us |
| 10-20 05:53 | eoa20 57413368 .959 | r135 / 6,253 | r87-130 | r131-196 | 61 of 65 below us (4 above) |

- Medal cuts now: gold r22, silver r312, bronze r625.
- A21 entries submitted 11:41 / 11:42 / 12:01 IST (MEASURED a21/tick_a21.log), with reads ~17:00-19:00.

### 1.2 Magnitudes (MODEL)
- In-band position u (0 = .959 floor, 1 = .960):
  - eoa10 u .40 (sd .075). That is 29.5 / 65, mapped through the density.md thinning-upward density exp(-k u), k 0-1, plus order-statistic
    noise.
  - eoa20 u .93 (sd .03).
- c4's gap under .959: g 1.5e-4 (sd .9). That is density.md's a25 gap 3.4e-4 minus the c4 - a25 ruler 2.3e-4, truncated at 0.
- So the public paired deltas vs c4 are measured as **pub10 ~ 5.5e-4 (sd 1.2) and pub20 ~ 10.8e-4 (sd .95)**, correlated through g.
- Read noise (MEASURED r5/noise_corr.log, report OOF): the paired public sd is 2.1e-4 at a .1 and 4.6e-4 at a .2, corr .96. So the
  eoa20 read is a ~2.3 sd signal, and pub20 - pub10 ~5.3e-4 is ~2 sd of its own noise (2.65e-4).

### 1.3 Posterior (MODEL, mix prior of r5/decision_mc.py; `bpick_base.json`, ESS 36.6k of 1.2M)

| quantity | prior (mix) | posterior after both magnitude reads |
|---|---|---|
| d10 (hidden macro gain of eoa10 over c4) | +2.5e-4 sd 4.4, P>0 .70 | **+5.1e-4 sd 2.3, P>0 .99** |
| d20 | +0.9e-4 sd 9.1, P>0 .53 | **+8.3e-4 sd 4.4, P>0 .98** |
| P(d20 > d10) | .37 | **.91** |
| prior weights more / ladder / orig / null / neg | .40 / .25 / .15 / .10 / .10 | .36 / .23 / .27 / .14 / .003 |

- The dose response is still rising at a .2 on the hidden labels. That refutes the 10-20 OAI / ladder view (r6_oai.md: eoa20 ~-.0001,
  P(eoa20 > eoa10) .35-.4) and the pessimist "neg" prior (its weight drops to .003).
- P(eoa20 private < c4 private) = **.06** (= P(B > A) for B = c4).

### 1.4 Transfer of E-OA to the R6 entry (MEASURED, `push/r6transfer_g58.log`, kernel-path gold58 replay, n 58)
- The R6 and c4 entries share every member (R6 bundle, Raptor r384 + r224, S6fall .2) and the same cell 18.
  - Their OA blend differs only in Lateral OA (Raptor W .5 vs .2) and A224 (.35 vs .25), CLAIMED ex_r6eoa_gate.py.
  - Final OA columns agree at Spearman **.9996 / .9886 / .9990** (MOA / LOA / PF).
- E-OA deltas on gold58 (macro):

  | a | c4 | R6 | ratio | boot corr | slope | sd(R6 - c4) | P(R6 delta > 0) |
  |---|---|---|---|---|---|---|---|
  | .2 | +.00247 | +.00218 | .88 | .75 | .83 | .0010 | .96 |
  | .1 | +.00142 | +.00083 | .58 | .47 | | | |
  | .3 | +.00201 | +.00173 | .86 | .85 | | | |

  - The E-OA rank changes themselves agree at .86-.92 Spearman for a .2.
- OAI (CLAIMED r5/r6_oai.md) measured R6 alone as zero to negative at RSNA coupling. The same ruler gave R6 + S6fall (.4 of c4) -.0016
  / -.0055 OA-mean, and c4 is now LB-positive at ~2 sd. So that ruler is pessimistic for this family and does not separate R6 from c4.
- Model input: **k ~ N(.9, .15)** (GUESS on the MEASURED ratio .88 plus near-identical OA columns), structural spread 1e-4, private
  E-OA noise corr .8 with eoa20's (gold58 .75).
  - Skeptic rows: k ~ N(.6, .3) and N(.4, .4).
  - Posterior R6e20 gain over R6: +7.5e-4 (sd 4.3). P(R6e20 beats R6 on private) is about .92 (MODEL, with private noise 3.15e-4).

## 2. Model (`push/bpick_mc.py`; MODEL unless tagged)
- **Our side:** the 10-19 think/picks posterior `post_<our>_report.npz` (200k draws), reused unchanged. It holds private margins of c4,
  R6, HW, a25 and S6F, conditioned on every public read up to 10-19.
  - c4 private margin is +5.6e-4 (sd 6.9). c4 - R6 private mean is +2.9e-4, with corr .54 (CLAIMED picks.md).
- **E-OA:** eoaX = c4 + dX + private paired noise (sd 1.46 / 3.15 / 4.9e-4 at a .1 / .2 / .3, MEASURED power law). Prior and read
  noise follow r5/decision_mc.py. Importance weights come from the sec 1.2 Gaussian measurement of (pub10, pub20).
- **R6e20** = R6 + k x d20 + eps + private noise (sec 1.4).
- **Display guard (R6e20g):**
  - R6's public sits at u .10 of .958 (density.md).
  - R6e20 displays .959 iff its public gain over R6 exceeds (1 - u) x 1e-3, and .957 iff the gain is below -u x 1e-3.
  - Policy: B = R6e20 unless it displays .957.
- **A21 module (GUESS-heavy):**
  - d30 = 3 (d20 - d10) + N(-1.5e-4, 3e-4): quadratic extrapolation, shifted down because gold58's kernel-path dose curve peaks near .2
    (MEASURED .00142 / .00247 / .00201).
  - pw = same dose + shape term N(+.3e-4, 1.5e-4), from EX: shape refinements <= +1e-4.
  - The read noise of the five entries is correlated .92-.985.
  - A = max public over {eoa20, eoa30, pw20, pw30}.
  - Model R1 odds: eoa20 stays .16, eoa30 .28, pw20 .16, pw30 .40. These are only as good as the d30 / pw GUESSES.
- **Field:** the 10-19 field_mc model, unchanged (silver x* 5.0e-4), re-run with two extra thresholds (`push/field_curves960.py`):
  - r130 = today's .960+ band size (MEASURED);
  - r167 = the .960+ band projected at the deadline (field_mc t1_bands: 166 teams at >= .960).
  - "P(.960+ band on private)" = P(private rank <= 167). The absolute version, P(pub-scale >= .960), is also given.

## 3. Results (base: our base, field base, mix prior, noise x1, k .9; `bpick_base.log` / `.json`)

### 3.1 A = eoa20 (the current API best-id)

| B | P(silver) | P(bronze+) | P(r<=200) | P(r<=167) | P(r<=130) | P(r<=100) | E[max] x1e-4 | pub-scale | P(B > A) |
|---|---|---|---|---|---|---|---|---|---|
| (A alone) | .848 | .931 | .651 | .534 | .316 | .094 | 13.96 | .95942 | |
| **R6e20** | **.890** | **.955** | **.714** | **.600** | **.372** | **.117** | **15.41** | **.95956** | .304 |
| S6F 57201874 (.957; artifact, sec 3.3) | .878 | .948 | .698 | .585 | .363 | .117 | 15.15 | .95954 | .170 |
| R6 57248084 | .869 | .948 | .670 | .548 | .322 | .094 | 14.37 | .95946 | .100 |
| HW 57262115 | .866 | .949 | .661 | .539 | .317 | .094 | 14.27 | .95945 | .086 |
| eoa10 57413380 | .860 | .940 | .661 | .541 | .319 | .094 | 14.18 | .95944 | .142 |
| a25 57316395 | .859 | .945 | .655 | .535 | .316 | .094 | 14.17 | .95944 | .069 |
| c4 57343435 | .856 | .938 | .656 | .537 | .317 | .094 | 14.10 | .95943 | .061 |

- The display guard changes nothing in base. P(R6e20 displays .957) is .001, P(.958) .44 and P(.959) .56.

### 3.2 A = A21 R1 winner (after tonight's reads)

| B | P(silver) | P(bronze+) | P(r<=200) | P(r<=167) | P(r<=130) | E[max] x1e-4 | P(B > A) |
|---|---|---|---|---|---|---|---|
| (A alone) | .860 | .928 | .702 | .609 | .419 | 16.32 | |
| **R6e20** | **.898** | **.955** | **.751** | **.657** | **.458** | **17.48** | .240 |
| S6F | .891 | .949 | .743 | .652 | .459 | 17.44 | .151 |
| eoa10 | .884 | .949 | .721 | .623 | .425 | 16.80 | .165 |
| R6 | .883 | .949 | .721 | .622 | .424 | 16.79 | .098 |
| eoa20 (if A moved; else eoa10) | .882 | .946 | .722 | .625 | .428 | 16.80 | .230 |
| HW | .881 | .951 | .712 | .614 | .420 | 16.72 | .092 |
| c4 | .875 | .943 | .711 | .615 | .421 | 16.63 | .083 |

- **By A21 outcome** (P(silver) / P(r<=167)):

  | A21 outcome | P (model) | R6e20 | R6 | eoa20 | eoa10 | c4 |
  |---|---|---|---|---|---|---|
  | eoa20 stays A | .16 | .882 / .589 | .860 / .537 | | .856 / .532 | .846 / .524 |
  | eoa30 | .28 | .887 / .649 | .869 / .615 | .878 / .628 | .879 / .622 | .868 / .611 |
  | pw20 | .16 | .906 / .629 | .889 / .589 | .875 / .580 | .882 / .581 | .877 / .578 |
  | pw30 | .40 | .910 / .702 | .898 / .676 | .898 / .680 | .900 / .678 | .892 / .670 |

- **R6e20 is the best B in every A21 outcome.** If A moves to an A21 entry, do NOT switch B to eoa20: it is a near-twin of A (-0.9 to
  -3.1 pts vs R6e20).

### 3.3 S6F is not a real alternative (MODEL, `bpick_s6fdefl.log`)
- Its edge over R6 comes only from the modelled idiosyncratic private noise of a single-fold R7 OOF stand-in (as in decision.md
  sec 0.3).
- Halving that noise (s6fdefl .5) drops it to silver .873 / r167 .559, vs R6e20 .893 / .597. Removing it gives .871 / .552.
- It is also a .957 entry. Not recommended.

## 4. Sensitivity (29 rows, rep 4; `push/summ_bpick.out`, `sens.log`; P(silver) in pts)

| row | A = eoa20: R6e20 - R6 | best B (excl. S6F) | A = A21: R6e20 - R6 | best B (excl. S6F) |
|---|---|---|---|---|
| base | +2.1 | R6e20 | +1.6 | R6e20 |
| priors more / ladder / orig / null | +2.2 / +2.2 / +1.9 / +2.1 | R6e20 | +1.6 / +2.2 / +1.2 / +1.1 | R6e20 |
| prior neg (posterior weight .003 after the read) | 0.0 | eoa10 (+0.3 vs R6e20) | -0.6 | eoa10 |
| read noise x1.5 | +0.7 | R6e20 | +0.3 | eoa10 (+0.8) |
| k ~ N(.6, .3) / N(.4, .4) / N(1, .1) | +1.2 / +0.7 / +2.5 | R6e20 | +0.8 / +0.4 / +1.9 | R6e20 |
| k N(.4, .4) + noise x1.5 (double skeptic) | -0.3 | R6 / HW | -0.7 | eoa10 |
| private E-OA noise corr .6 / .9 | +2.4 / +2.0 | R6e20 | +1.9 / +1.4 | R6e20 |
| our side skepS6 / strongS6 / tabReal / tabOver | +3.6 / +1.5 / +1.6 / +2.5 | R6e20 | +2.6 / +1.1 / +1.2 / +1.9 | R6e20 |
| our side r6better / noise1.5 / bigKnobs / smallKnobs | +2.2 / +3.4 / +2.1 / +2.4 | R6e20 | +1.7 / +2.5 / +1.5 / +1.8 | R6e20 |
| field fieldBad / fieldGood / Wbig | +1.9 / +2.4 / +2.7 | R6e20 | +1.4 / +1.8 / +2.0 | R6e20 |
| c4 gap g low / high, flat density | +2.2 / +2.1 / +2.2 | R6e20 | +1.3 / +1.8 / +1.8 | R6e20 |
| d30 optimistic / pessimistic | +2.2 / +2.2 | R6e20 | +1.3 / +1.8 | R6e20 |

- **P(r <= 167):** R6e20 - R6 is +1.5 to +6.7 pts with A = eoa20 and +1.0 to +4.8 with A = A21. It is positive in every row.
- **Where R6e20 loses:**
  - It loses only when E-OA's own hidden effect is shaky: the neg prior (which the read rejects), or 1.5x read noise combined with a
    poor transfer.
  - Then the dose hedge (eoa10) or plain R6 is the better B, by <= 0.8 pts.
- **The display guard:**
  - It costs nothing.
  - It barely helps: +0.0 to +0.1 pts in the k-skeptic rows, where P(.957) is .03-.11.
  - Keep it as a fixed rule anyway, since it is free.

## 5. Local build and offline gate (MEASURED; NOT pushed)
- `push/build_r6eoa.py` copies build_a20.py with SRC = the R6 kernel. It inserts the verbatim A20 9-line block at the same cell-18
  anchor.
- **kernel_v224r6flips6feoa20j:** sha 2a5ef93a. Kernel id eeshsaxena/rsna-sub-v224r6flips6feoa20j. Same datasets, docker image, T4 and
  GPU as R6.
- **kernel_v224r6flips6feoa00j:** a .0 control, sha 63b02589.
- **Cross-check (XREF in the build receipts):**
  - The R6e20 notebook differs from the SCORED eoa20 notebook (a20/kernel_v224r6c4eoa20j) only in cell 14, exactly as R6 differs
    from c4.
  - Its cell 18 is byte-identical to eoa20's.
  - So both code paths are proven on the hidden set: R6 57248084 and eoa20 57413368 both scored. Runtime risk is about nil.
- **`push/gate_r6eoa_offline.log`: R6EOA_OFFLINE_GATE PASS**
  - G-a: the R6e00 output is byte-identical to R6's.
  - G-b: the gold58 delta is +.00218 (target +.00218).
  - G-c: the non-OA columns are identical.
  - G-d: the OA columns equal an independent E-OA transform of R6's output.
- **Commit gate after a push (not run; it needs the committed output):**
  `python session_2026-10-19/a20/probe_check_eoa.py <out_dir> --a224 0.35 --table '{"Medial OA": 0.5, "Lateral OA": 0.5, "PF OA": 0.5}' --aoa 0.2`.
  - The table matches R6's blend_receipt `raptor_per_class` (MEASURED).
  - It must PASS 17/17 before a submit.

## 6. Slot plan and draft amendment (for Eesh; nothing done)
- **Timing:**
  - Today's UTC day has 0 slots left (eoa20, eoa10, eoa30, pw20, pw30).
  - 10-21 has 5 slots from 05:30 IST. Commits take ~20 min and scoring ~5.5-6 h.
- **Plan:**
  - On Eesh's go (any time tonight, independent of the A21 reads), push R6e20. Then run the commit gate and submit at 05:31 IST 10-21.
    It scores ~11:00-11:30 IST 10-21.
  - Then fix B by the rule below. That leaves 4 slots on 10-21 and 5 on 10-22 for A.
  - Latest fallback: submit 05:31 IST 10-22, scored ~11:30 IST, well before selection (10-22 23:59 UTC). There is no retry, so prefer
    10-21.
- **Draft PREREG Amendment 22 (text only; I did NOT write it into PREREG):**
  - "B-hedge entry R6e20 = rsna-sub-v224r6flips6feoa20j. It is R6 57248084 plus the A20 E-OA block at a .2, and its cell 18 is
    identical to eoa20 57413368.
  - Offline gate PASS (push/gate_r6eoa_offline.log). Commit gate probe_check_eoa.py (R6 params, --aoa .2) must PASS 17/17.
  - One submit.
  - Selection rule, fixed before its score: B = R6e20 if it is COMPLETE and displays >= .958; otherwise B = R6 57248084. A stays the
    API best-id (A15).
  - R6e20 is never A. It cannot reach eoa20's public score: R6 is ~7.9e-4 below c4 publicly, MODEL density.md + picks.md ruler."
- **Eesh selects A and B by hand** before 10-22 23:59 UTC.
  - Auto-selection would pair two E-OA near-twins, e.g. eoa20 + pw30: about -2 to -3 pts silver vs R6e20 (MODEL, sec 3.2).

## 7. Caveats
- **Absolute levels rest on the 10-19 field model's sharp step** (silver x* 5e-4) and on its GUESS parameters. Pair differences are more
  robust than levels. The field has not been refit since 10-19; band growth is still on the projection.
- **The magnitude read is a MODEL** (in-band density, c4's gap). The g / u rows move the R6e20 edge by at most 0.3 pts.
- **The read noise was measured on report labels with R7 OOF**, not on expert labels with c4. The x1.5 row covers that. It is the row
  that hurts R6e20 most (+0.7).
- **Transfer k comes from gold58 (n 58) plus structure.**
  - There is no hidden-label measurement of k until R6e20 scores. Its display then gives only a coarse check: .959 implies k >~ .8,
    and .957 means a failure.
  - OAI cannot measure k for this family (sec 1.4).
- **The A21 module (d30, pw) is GUESS-heavy.** Only the eoa20 rows (3.1) rest on measured reads.
- **Not modelled:**
  - the S6fall cliff beyond the picks knob prior (A and B both carry S6fall);
  - host rulings;
  - teams copying E-OA (it is not public).

## 8. Files (`session_2026-10-20/push/`, this track only)
- **Model:**
  - `bpick_mc.py` (args: our=, field=, prior=, noise=, k_mu=, k_sd=, rho=, g_mu=, u10=, u20=, d30shift=, s6fdefl=, tag=).
  - Outputs: `bpick_<tag>.json`, `bpick_base.log`, `sens.log`, `bpick_s6fdefl.log`.
  - Runner: `run_sens.sh`. Summary: `summ_bpick.py` -> `summ_bpick.out`.
- **Field:** `field_curves960.py` -> `curves_{base,fieldBad,fieldGood,Wbig}.npz` (r5/field_curves.py plus r130 / r167).
- **Transfer:** `r6transfer_g58.py` / `.log`.
- **Build (local only):**
  - `build_r6eoa.py`;
  - `kernel_v224r6flips6feoa20j/`, `kernel_v224r6flips6feoa00j/`;
  - `build_receipt_v224r6flips6feoa{20,00}j.json`;
  - `gate_r6eoa_offline.py` / `.log`.
- Other files in `push/` (census*, cbase.py, e1/, work/) belong to a sibling task and were not touched.
