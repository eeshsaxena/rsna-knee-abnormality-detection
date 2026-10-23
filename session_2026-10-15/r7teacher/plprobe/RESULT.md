# Per-label LB probing: builder + simulation (R7-teacher workflow, job 3; built 10-14 IST by a Claude subagent). NOT submitted.

No em dashes. Tags: MEASURED (we observed it), MODEL (our simulation / arithmetic), GUESS (judgement), CLAIMED (third party).
All gains are macro AUC; "e-4" = 0.0001. Directory: `session_2026-10-15/r7teacher/plprobe/`.

## 0. Bottom line

- **Builder works and is commit-proven.** `build_pl.py` turns a JSON map of per-label weights (Raptor_mix / R7 / our fleet F)
  into a private kernel on the V224 base with every base cell byte-identical, plus our canaries. Offline 21/21 (incl. bit-exact
  reproduction of V224-exact, of the HFR7-RR HF cell and of the V224-F HF cell); Kaggle commit of the example
  `eeshsaxena/rsna-sub-pl-oa65` v1: **PL_CHECK PASS 17/17** (MEASURED, 438 s, 1 commit of GPU quota). Not submitted.
- **Simulation (MODEL, 150-300 worlds per scenario, calibrated to our LB anchors):** whether per-label probing helps on PRIVATE depends
  entirely on one unknown, the true per-label strength difference between R7 and Raptor (tau, GUESS). If tau is small
  (<= .005 AUC per label) or there is no structure, coordinate ascent over 24-35 slots LOSES private (-2.7 to -7.6e-4, P(private < 0)
  .78-.99) while the public LB shows a fake +3.9 to +9.8e-4. If tau ~.010, it is roughly break-even to +3.5e-4; only at tau ~.020
  (or a real OA / trauma structure) does it pay (+11 to +25e-4 private).
- **.960 is not a realistic outcome of per-label probing:** public .960 display needs +34e-4. MODEL P(public display .960) is
  <= .015 in 6 of 8 scenarios, .04-.39 only if tau = .020 (.06-.19 if a .020 OA / trauma structure is true); prior-weighted
  ~.01 (group probing) to ~.05 (35 single-label probes) (GUESS prior). And a public .960 reached that way
  would carry ~+9 to +13e-4 of overfit on top of the real private gain.
- **What makes it defensible is the 2-pick hedge:** if pick A stays the unprobed copy, a probed pick B only needs E[max(0, gain)].
  Prior-weighted (GUESS prior) E[max(0, private gain)] = **+1.9e-4** (4 prior-directed slots) to **+3.3 / +4.6e-4**
  (24 / 35 slots, label level, final map shrunk halfway). The mean private gain of the same policies is +0.7 to +4.0e-4.
- **Recommendation (needs main + Eesh, because RESEARCH_960 sec 5 and PREREG_10-14 Amendment 3 say "no per-label public tuning"):**
  run at most the 4-slot prior-directed phase (OA Raptor share up, trauma R7 share up, each with an opposite-direction retry),
  then a decision gate; never let per-label probes touch pick A; put any accepted map into pick B only after a halfway-shrunk
  version has its own scored run. Expected private value of that phase: +0.7e-4 mean, +1.9e-4 hedged (MODEL x GUESS prior);
  quota ~0.5 h, 4-5 slots.

## 1. The builder

Files: `build_pl.py` (builder), `maps/*.json` (maps), `next_map.py` (current best map + ONE change -> next probe map),
`pl_check.py` (commit gate, no-fleet bases), `sim_offline.py` (offline execution of the inserted cells on real commit csvs).

**Blend** (rank space, per label c; the same form as the MEASURED-positive HFR7-RR HF cell):
```
H_c     = s_c * RaptorMix_c + (1 - s_c) * R7_c          her stage-3 arithmetic, fallbacks and pct_on; hers is s_c = .5
          RaptorMix_c = .65 pct(r384) + .35 pct(r224)    (A224 unchanged; r224 failed -> r384 only and the canary blanks Fracture)
final_c = (1 - f_c) * pct(H_c) + f_c * pct(F_c)         only for labels with f_c > 0 (PL-F cell); else final_c = H_c
implied weights: Raptor_mix (1 - f) s, R7 (1 - f)(1 - s), F f
```
**Map** (`maps/oa65.json`): `{"name", "note", "default": {"s": .5, "f": 0}, "groups": {"OA": {"labels": [...], "s": .65}},
"labels": {...}}`; default -> groups -> labels; any entry may instead give the triple `{"R", "R7", "F"}` (sum 1). Values on a .05
grid (`--offgrid` to override), f <= .6. The resolved map + implied weights go to `kernel_pl_<name>/pl_map_resolved.json`.

**Inserted cells** (base cells untouched; the builder asserts that deleting the inserted cells gives back the base FILE byte for
byte, and that varshitha's 13 cells occur verbatim and in her order):
- **PL-H**, right after her stage-3 cell. First a self-check: her own weights must reproduce her `out` (float64) and `sub`
  (float32) bit for bit, else it raises `PL_SELF_CHECK_FAIL` (no submission, never a silent different blend; the offline test
  shows it fires on a 1e-12 change). Then it recomputes only the columns whose s differs from .5, writes submission.csv,
  submission_PLH.csv (H before any canary) and pl_receipt.json, prints `PL_H_OK`. Our existing V224 CANARY cell follows it
  unchanged, so r224-not-used still blanks Fracture on the PL output.
- **PL-F**, appended last, only when some f > 0 (requires a base with the fleet stage, i.e. `--base-nb` = a V224-F notebook;
  the builder refuses an f > 0 map on the no-fleet bases). Reads submission_PLH.csv + submission_fleetsolo.csv; canaries
  re-applied last: F needed but missing -> H for all labels and Synovitis 0.5; r224 not used -> Fracture 0.5 (the same canaries as
  V224-F). It overwrites whatever the base's own HF cell wrote.

**Bases:** `--base v224` (default, the submitted V224-exact kernel, sha 4259ec97...), `--base v224flip` (sha 73a9d562...), or
`--base-nb NB --base-meta META` for any notebook carrying her 13 cells verbatim (checked on the parallel agent's
`session_2026-10-15/v224f/kernel_v224f20`: PL-H lands at cell 13 before the canary, PL-F at cell 23 after their HF cell; the kernel
keeps their 5 datasets).

Usage (project root):
```
python session_2026-10-15/r7teacher/plprobe/build_pl.py --map session_2026-10-15/r7teacher/plprobe/maps/oa65.json            # V224-exact base
python .../build_pl.py --map MAP --base v224flip
python .../build_pl.py --map MAP --base-nb session_2026-10-15/v224f/kernel_v224f20/rsna_sub_v224f20.ipynb --base-meta session_2026-10-15/v224f/kernel_v224f20/kernel-metadata.json
python .../next_map.py maps/oa65.json TRAUMA s=0.35 oa65-tr35      # next probe = current best map + one change
python .../pl_check.py OUT kernel_pl_<name>/pl_map_resolved.json --ref session_2026-10-14/v224/out_v224_v1   # commit gate
```
Kernel id `eeshsaxena/rsna-sub-pl-<name>`, private, GPU T4 x2, internet off, same datasets / docker pin as the base.

### 1a. Offline proof (`sim_offline.txt`: SIM_OFFLINE PASS 21/21, MEASURED)
Her stage-3 cell + PL-H + CANARY executed on the V224-exact commit csvs (`session_2026-10-14/v224/out_v224_v1`):
- identity map -> submission.csv byte-identical to the V224-exact commit (CRLF-normalised);
- oa65 and a 7-label two-direction map (incl. the R/R7/F triple form) -> == independent numpy recompute (max diff 3.3e-8), only
  mapped columns move, receipts match;
- forced r224 failure -> CANARY_FIRED after PL_H_OK, Fracture 0.5 exactly, the other 11 = recompute without r224 at the PL shares;
- tampered stage-3 `out` (+1e-12) -> PL_SELF_CHECK_FAIL raised;
- PL-F with f .2 everywhere on the HFR7-RR commit output == the HFR7-RR HF submission.csv byte for byte; fleet missing ->
  Synovitis 0.5 and H elsewhere; r224 not used -> Fracture 0.5;
- PL-F with f .2 on the parallel V224-F commit (`v224f/out_v224f20_v1`) == their HF output, max diff 0.

### 1b. Commit test (MEASURED)
| item | value |
|---|---|
| kernel | `eeshsaxena/rsna-sub-pl-oa65` **v1**, private; notebook sha256 df9f597d0fdf0088ce3d62a344caee7c4b2bbd50b0856e564f14c0c3075b4296 |
| map | `maps/oa65.json` (map sha 591ebe555f7e861e): Raptor share .65 on Medial OA, Lateral OA, PF OA; all else = V224-exact |
| run | pushed 07:40 UTC, COMPLETE 07:48 UTC, kernel log end 438 s (R7 stage 320 s, r224 23 s, blend total 422 s); 0 fallback studies |
| gate | `gate_pl_oa65_v1.txt`: **PL_CHECK PASS 17/17** (markers in order NOTEBOOK_COMPLETE < PL_H_OK < CANARY_CLEAN < V224_DONE; her blend receipt unchanged; pl_receipt = this map; submission == independent recompute max diff 3.3e-8; 9 unmapped columns bit-identical to the V224-exact commit; Medial OA and Lateral OA moved, PF OA tied on these 3 rows) |
| negative control | the V224-exact commit output under the oa65 gate: FAIL 11/16 (no PL_H_OK, no pl_receipt, recompute diff .1) |
| quota | 1 commit, ~7.3 min of Eesh's GPU quota |

`rsna-sub-pl-oa65` v1 is a valid first probe ONLY while V224-exact is the team best; on any other base rebuild with that base.

## 2. Simulation

Scripts: `calib_pl.py` (+ `calib_pl.txt`), `sim_pl.py` (policies; `simpl_main.json`, `simpl_main_summary.txt`),
`sim_probe_res.py` (single-probe resolution; `pres_main.json`, `pres_main.log`, `pres_threshold_summary.txt`),
`simpl_emax_summary.txt` (hedged value). Run on the box CPU (nice 19, 8 + 4 processes, 21 + 11 min), not the laptop.

**Model** (extends `research960/lbprobe/sim_noise.py`): per label, latent binormal scores for R7, r384, r224, F; test 1,300 =
public 390 + private 910 (MEASURED split); population 12,000 per label for the expected private value. Label prevalences = train
report-label rates (MEASURED on train, GUESS for test). Per-label component level = .949 + .5 x (gold58 N AUC - mean) (MEASURED
gold58 profile, shrinkage GUESS). Per-label truth: A(R7) - A(Raptor) = mu_c + tau z_c, swept (GUESS); "structured" = OA labels
Raptor better and Fracture / MCL / Contusion R7 better by delta. Hidden-rerun noise (MEASURED ~4e-6) ignored.

**Calibration (MODEL, `calib_pl.txt`):** the old sim_noise correlations (.62) make R7 + Raptor gain ~.016 over the parts, far more
than the LB shows. Error correlation R7 ~ Raptor **.82** (r384 ~ r224 .93) puts V224 at .956 from parts at .949-.950 (r384 .949 and
r224 .945 CLAIMED by their authors; R7 solo GUESS .950). Fleet correlation **.85** with F solo .944 reproduces both MEASURED signs:
F .2 beats F 0 (HFR7-RR > HR7-RR) and F .3 loses to F .2 (**RR-F30, 57163771, scored .956 at 12:46 IST and did not become team
best**, MEASURED via `api_chain.log`).

**Read model:** a probe = current team best + ONE change (a label group's s by +-.15, or f by +-.1); it is accepted iff its public
macro AUC beats the current best (the API best-id read). Policies: `prior2` (OA s up, TRAUMA s down, each retried in the opposite
direction if rejected, a further step if accepted; 2-4 slots), `group` (5 groups OA / TRAUMA / MEN / FLUID / ACL, prior directions
first, steps .15 -> .10 -> .05; on the F base also f per group), `label` (12 single labels), `+shrink` (final map halfway back to
the base), `pubopt` (per-label public argmax = unlimited probing), `oracle` (population argmax = best possible).

### 2a. One probe (MODEL, `pres_main.log`; e-4)
| scenario | OA s .5 -> .65: P(win) | E[private-expected gain] if won | P(gain > 0) if won | public delta sd |
|---|---|---|---|---|
| no heterogeneity (tau 0) | .22 | -1.5 | .00 | 2.1 |
| tau .005 | .31 | -1.0 | .18 | 2.5 |
| tau .010 | .33 | +0.1 | .48 | 3.2 |
| tau .020 | .33 | +2.0 | .76 | 4.7 |
| structure true (delta .010) | .78 | +2.2 | 1.00 | 2.5 |
| structure reversed (delta -.010) | .02 | (rarely accepted) | | 2.4 |

Moving a share off .5 costs ~1-1.5e-4 per group when the components are equally good (balance is worth something), so a "win"
is a coin flip or worse unless per-label differences are >= .01 AUC. A magnitude-aware read (count a win only if public moved by
> 1e-4, i.e. it passed many frozen teams) raises E[gain | win] at tau .010 from +0.1 to +0.5e-4 but does not rescue the null case
(`pres_threshold_summary.txt`). Fleet probes (group f .2 -> .1) move < 1e-4 either way.

### 2b. Policies (MODEL, `simpl_main_summary.txt`; base V224, e-4; private = realized on the 910 private studies)
| scenario (GUESS prior weight) | prior2, 4 slots: pub / priv / P(priv<0) | group, 24: pub / priv / P | group+shrink, 24: priv / P | label, 35: pub / priv / P | label+shrink, 35: priv / P | oracle priv |
|---|---|---|---|---|---|---|
| tau 0 (.175) | +1.2 / -1.7 / .61 | +3.5 / -4.1 / .93 | -1.4 / .84 | +7.3 / -7.6 / .99 | -2.4 / .93 | -0.2 |
| tau .005 (.175) | +1.8 / -1.2 / .57 | +4.8 / -2.8 / .79 | -0.2 / .61 | +9.8 / -4.4 / .87 | -0.1 / .52 | +2.1 |
| tau .010 (.30) | +2.3 / -0.2 / .46 | +7.2 / +0.5 / .43 | +2.0 / .21 | +16.0 / +3.5 / .27 | +5.2 / .09 | +9.9 |
| tau .020 (.10) | +4.6 / +2.5 / .29 | +16.5 / +10.8 / .09 | +8.8 / .02 | +34.6 / +25.1 / .00 | +17.5 / .00 | +31.7 |
| structure true .010 (.15) | +5.7 / +3.6 / .12 | +8.9 / +2.1 / .30 | +3.3 / .09 | +14.1 / +0.2 / .50 | +3.3 / .12 | +6.6 |
| structure true .020 (.03) | +16.4 / +13.7 / .00 | +23.6 / +15.2 / .02 | +12.5 / .00 | +27.9 / +14.6 / .00 | +12.3 / .00 | +20.2 |
| structure reversed (.07) | +2.9 / +1.5 / .21 | +8.2 / +2.1 / .30 | +3.1 / .11 | +13.2 / +0.1 / .48 | +3.0 / .14 | +7.1 |

On the V224-F base (f probes added) the numbers are similar
(`simpl_main_summary.txt`, V224F20 rows); the oracle there is larger (+5 to +53e-4) because F per label adds a degree of freedom.

Readings (MODEL):
- **Overfit size:** the public LB overstates the private gain by ~+6 to +15e-4 after 24-35 probes in every scenario (e.g. label/35
  under tau .010: public +16.0, private +3.5). In the no-heterogeneity world the public shows +3.9 (group/35) to +8.7 (unlimited)
  while private loses 4-8e-4; P(public display .957) .16-.69 there, all of it fake.
- **More slots only pay if tau >= .010.** With a true OA / trauma structure, 4 prior-directed slots capture almost all of it
  (+3.6e-4); 35 slots add nothing (+2.1 group, +0.2 label: overfit eats the rest).
- **Shrinking the final map halfway** roughly halves the public gain and removes most of the private downside (tau 0: -7.6 -> -2.4;
  tau .010: +3.5 -> +5.2). It costs one extra slot (the shrunk map must have its own scored run to be pickable).
- **.960:** P(public display .960) is <= .015 in every scenario except tau .020 (.04 group/35, .39 label/35) and structure .020
  (.06-.19). Prior-weighted ~.01 (group) to ~.05 (label/35) (GUESS prior), and such a display would include ~+9 to +13e-4 of overfit.

### 2c. Hedged value of a probed pick B (MODEL x GUESS prior; `simpl_emax_summary.txt`)
Kaggle scores the better of the 2 picks. With A = the unprobed base and B = base + probed map, B is worth E[max(0, private gain)]
(normal approximation from the simulated mean / sd). GUESS prior over scenarios as in 2b.

| policy (slots) | prior-weighted mean private gain | prior-weighted E[max(0, gain)] |
|---|---|---|
| prior2 (4) | +0.7 | +1.9 |
| group (24) | +1.0 | +3.0 |
| group + shrink (24) | +2.3 | +2.8 |
| label + shrink (24) | +2.8 | +3.3 |
| label (35) | +1.9 | +4.9 |
| label + shrink (35) | +4.0 | +4.6 |
| V224-F base: group + shrink (24) | +3.3 | +3.9 |

These EVs are only as good as the GUESS prior on tau; with tau <= .005 every policy has negative mean private gain. The hedge
also requires that pick A is exactly the base the probes started from, and that B is the ONLY place the probed map goes.

## 3. Proposed schedule (only if main + Eesh accept per-label public tuning; it contradicts RESEARCH_960 sec 5 + Amendment 3)

Rules for every probe: one change vs the CURRENT team best (rebuild on whatever base is best: V224-exact, V224-F, V224-F-FLIP);
submit >= 45 min apart from any other entry so scoring times separate; LB API snapshot after each scores
(`research960/lbprobe/fetch_api_lb.py` + `api_frozen.py`); a probe "wins" only if the best submissionId moves to it; record how
many frozen teams it passed (magnitude). Map names via `next_map.py`. Pick A never changes.

**Phase 1 (4 slots, 10-15 / 10-16; prior-directed, the only phase with a MODEL-positive mean):**
1. **OA s .65** (Medial OA, Lateral OA, PF OA: Raptor share up). Prior: Raptor (nartaa CoAtNet) trained with OAI d96 rows; the
   RSNA-only Raptor seeds lost a full display step on public (HFR7-RS .955 vs .956, MEASURED), consistent with OAI-supervised
   strength. If it wins, next OA step s .80; if it loses, OA s .35 once.
2. **TRAUMA s .35** (Fracture, MCL, Contusion: R7 share up). Prior: GUESS only (R7's private Opus-graded state targets and fm aux;
   no measured per-label R7 data exist). If it wins, s .20; if it loses, s .65 once.
   (Order of 1 and 2 is free; never run both on the same base in parallel, the second read would be censored.)

**Gate after phase 1:** count wins and the frozen-team jumps.
- 0 wins of 3-4 probes: evidence leans to no structure / small tau (MODEL P(0 wins) ~.3 under tau 0 vs ~.04 under a true
  structure). STOP: keep pick B without per-label weights.
- >= 1 win with a large jump (passed >= ~30 frozen teams; GUESS translation of > 1e-4): continue with phase 2.

**Phase 2 (<= 12 slots, 10-16 to 10-20):** MEN s .35 (R7 share up; R7 flip TTA is meniscus-specific) / s .65, FLUID s .65 / .35,
ACL, then on a V224-F base the fleet share per group (f .2 -> .1 first; MODEL says f .1 is mildly better everywhere, MEASURED F .3
< F .2). Step .15 only (.30 steps cost 5-8e-4 in the null world and are rarely accepted). No single-label probes unless several
group probes won with large jumps (that is the tau .020 world, where label level pays: +25e-4).

**Phase 3 (1-2 slots, by 10-21):** submit the accepted map shrunk halfway toward the base (s .65 -> .575 is off the .05 grid: use
`--offgrid`) so it has its own scored run; pick B = shrunk map (default) or the full map only if every accepted probe passed many
frozen teams. Pick A = the unprobed copy, always.

Costs: slots 4-5 (phase 1), up to ~17 total; results land ~5.5-6.5 h after submission (MEASURED HFR7-RS ~5.7 h), so at most ~4
sequential probes per UTC day and ~20-25 through 10-21, minus the other levers' slots. GPU quota: 1 commit per probe, 438 s
MEASURED on the V224 base (V224-F base ~10-11 min, MEASURED HFR7-RR 632 s): phase 1 ~0.5 h, phase 1-3 ~2-3 h, 35 probes ~4.3-6 h.
That exceeds the 2-3 h Eesh approved this week for ALL commits, so anything past phase 1 needs his explicit OK.

Proposed amendment text (for main to adopt or reject; not written into PREREG_10-14):
> Amendment PL. Per-label probing on pick B only. Phase 1 = OA s .65 and TRAUMA s .35 on the current team best, one change per entry,
> opposite-direction retry once. Continue only if >= 1 probe becomes team best by a large frozen-team jump. Pick A stays the unprobed
> V224 copy. Pick B may carry a probed map only after its halfway-shrunk version has a scored run. No single-label probes, no
> steps > .15, no probing after 10-20 12:00 IST.

## 4. Risks and caveats
- **tau is unmeasured.** No per-label R7 vs Raptor comparison exists (R7 gold58 scoring was ruled out on the box CPU, RESEARCH_960
  D7). The R7-OOF teacher run (job 1 of this workflow) would NOT measure it either (Raptor has no out-of-fold predictions on train).
  So the decision rests on the phase-1 reads, which are themselves noisy (public delta sd ~2-3e-4 per probe vs effects of ~1-2e-4).
- **Public rank becomes misleading:** in the no-heterogeneity world probing still lifts our public rank (P(.957 display) up to .16
  for 35 group probes) while private falls. Do not read a probed public rank as evidence for the final picks.
- **Slot competition:** every probe slot is a slot not used for a late public release rebase or the F / flip / mirror reads; probes
  must be sequential on the current best, so a release day freezes the probe chain (re-base, then continue).
- **Model limits:** binormal latent scores, prevalences from train reports, component levels from a shrunk gold58 profile,
  correlations fitted to 3 LB facts; per-label noise in the real public split could be larger (image-read labels, fewer positives).
- The builder's PL-F path is proven offline against both HF implementations but has no Kaggle commit of its own; commit-test the
  first f-map on the real V224-F base before any submission.

## 5. Ledger
- Kaggle: 1 commit (`rsna-sub-pl-oa65` v1, ~7.3 min), concurrency checked before the push (0 running). No submission, no dataset.
- Box: CPU only (nice 19; 8 + 4 processes, ~32 min); files in `~/plsim` (sim scripts + json). GPU queue untouched.
- Laptop: offline cell tests and summaries only (seconds).
