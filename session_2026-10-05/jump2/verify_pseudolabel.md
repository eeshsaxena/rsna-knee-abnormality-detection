# Verify: pseudo-label / self-distillation track (jump2, 2026-10-05 ~09:50 IST)

Target of the check: `SP/jump2/pseudolabel.md` and its track result. `SP` = scratchpad, `J2` = `SP/jump2`.
Tags: **MEASURED** (source) = recomputed today or read from a log; **MODEL** = arithmetic on measured inputs;
**GUESS** = judgement; **CLAIMED** = forum self-report; **LIT** = paper.

What I did (all read-only except `~/jump2/vpl/`):
- Wrote an independent re-computation, `J2/vpl/vpl_check.py` (own AUC, mapping and bootstrap code; no reuse of
  `pl_sim_b2.py` functions). Ran it on the box: CPU, `CUDA_VISIBLE_DEVICES=`, nice 19, 2 BLAS threads, ~1 min.
  Output: `~/jump2/vpl/vpl_check.out`, laptop copy `J2/vpl/vpl_check.out`.
- Read the fold-0 and fold-1 trainer logs, `~/auto_folds.sh`, `ps`, `nvidia-smi` (read-only), `~/jump2/tta/run1.log`,
  `~/fast_build/logs2/`, `~/jump2/r2/GATE.md`, `~/auto_round2.sh`.
- Re-read the forum copies (`J2/verify/t735304_flat.txt`, `t742327_flat.txt`, `t745214_flat.txt`), the Tian ledger
  (`SP/knee/tian/experiments.md` rows 77-103), `SP/round2/measured_levers.csv`, `SP/bigjump/angle_labels.md`,
  `HANDOFF.md`, `AUDIT_REPORT_2026-10-04.md`.
- Fetched the Stanton et al. 2021 arXiv abstract (2106.05945) to check one LIT claim.
- Nothing was killed, signalled, submitted or pushed. No writes outside `~/jump2/vpl/` and `J2/vpl/`.

---

## 0. Verdict

**The track's core verdict survives**: own-OOF round 2 is filler, not a lever. Every headline number reproduced
exactly. But six things need correcting before the synthesis uses this track:

1. **The chain-slowdown attribution is wrong.** `gold_infer.py` (CPU) was already running during fold 1's normal-speed
   epochs 1 and 2. The two slow epochs line up with **GPU co-tenancy** from other jobs, and the slip already exceeds the
   track's "10-20 min".
2. **The mix50 target's gold58 is optimistic.** Its Dread/Raptor teacher was epoch-selected on gold58. That biases
   "student minus target" for the mix50 and gated students, and the Dread-share line, against own-OOF teachers. The
   direction of the conclusions holds; the margins shrink.
3. **"Archit's precondition fails for us" is too strong.** It fails on Archit's overall check (OOF vs labels on all 58).
   It **holds** in report-silent cells, which is where Archit actually filled. The target-level payoff there is small
   (+.006 to +.007 gold58), and no silent-only own-OOF variant has ever been read on the LB.
4. **Gate S is noise-dominated.**
   - Its PROMOTE line sits 0.00001 from an ordinary F1 mix50 member (ws .82601 vs bar .826).
   - Its KILL lines sit about one seed-SD from J1. Tian's seed twins differ by .0098 gold58.
5. **`build_pl_tm5.py` has an unguarded footgun.** Full mode accepts a mid-training OOF csv; the r2 builder refuses
   one. Also, the "gate T PASS on fold 0" dry run actually used folds 0 and 1, with fold 1 a partial early-epoch OOF.
6. **The timeline is stale.**
   - A measured fold is 5.0 h alone, not 4.8 h.
   - HANDOFF (09:45) now plans box folds 1-2 plus Kaggle folds 3-4, all OOF by ~22:00.
   - The execution track has already built and armed a near-twin round-2 probe (`~/jump2/r2`, `auto_round2.sh`, a50).
     Run at most one of the two.

Also: one LIT claim is misread (Stanton 2021 says the opposite), and the write-up's "2 threads" line is false for
`gold_infer.py`, which ran at 12 threads. Expected public gain stays **+0.000 (range -.001 to +.002), P(>= .945) < .05**
(MODEL/GUESS, consistent with three LB nulls for OOF-derived targets).

---

## 1. Numbers re-computed independently (`vpl_check.out`, MEASURED)

| quantity | track | mine | status |
|---|---|---|---|
| gold74 in folds5 | all fold -1, excluded | 74/74 fold -1; trainer log "held-out 74, train 3466, val 867"; fold-0 val overlap with gold74 = 0; gold58 ids == `~/kmimic/ids/gold58.csv` | confirmed |
| gold58 targets: v4 / mix50_logit / gated75 | .8927 / .9262 / .9109 | .8927 / .9262 / .9109 | confirmed |
| Dread teacher alone / Raptor proxy | - / .9246 | .9127 / .9246 | confirmed |
| students gold58 (F1mix, F1v4, F1gated, TTRv4 ema, TTRmix) | .9144 .8611 .8846 .8780 .9132 | identical; logs: F1 mix50 ep7 .9143, J1 ep16 .9132 | confirmed |
| student minus target | -.012 to -.032 | -.0118 / -.0316 / -.0263 / -.0147 / -.0130 | confirmed (but see 2.2) |
| student minus v4 label | (not stated) | F1mix **+.0217**, TTRmix **+.0205**, F1gated -.008, v4 students -.015 / -.032 | new context |
| rho_rap / ws_rap, 5 students and 3 targets | table 3.1 | identical to 4 decimals | confirmed |
| member-to-member Spearman | .92-.94 | .904 to .944 (F1v4-TTRmix .904) | confirmed, range slightly wider |
| Dread-share line b = .25 / .5 / .75 / 1 | .9143 / .9204 / .9226 / .9262 | same | confirmed |
| Ev4 ALL .5 (v09t analog) | .9087 | .9087; vs line b=.75 -.0140 (sd .0075, P>0 .03) | confirmed |
| Ev4 UNK+NO .75 (DSD-own) | .9029 | .9029; vs line -.0200 (P .01) | confirmed |
| TTRmix .75/.5/.5 (pl_tm5 fold-0 analog) | .9267, rho .869, ws .813 | .9267 / .8693 / .8133; vs mix50 +.0006 (P .52); vs line +.0041 (P .72) | confirmed |
| Emix .75/.5/.5 | .9293, +.0068 vs line (P .83) | .9293; +.0068 (P .84) | confirmed |
| DSD-combo (+Raptor YES .25) | .9195 < mix50 .9262: F0 clause 2 FAILS | .9191 with my Raptor mapping; still < .9262 | confirmed |
| within-verdict gold58 AUC (silent cells: v4 .574, v4 students .70-.71, mix students .83-.84) | table | identical | confirmed |
| 10-04 F0 kill rule wording | as quoted | `bigjump/angle_labels.md` sec 5 matches | confirmed |
| r2 "a50" fold-0 analog (execution track, not this track) | - | gold58 .9240, rho .8597, ws .8037 | for comparison |

Per-fold v4 prevalence in `folds5` differs by at most .024 (Lateral Meniscus), so `build_pl_tm5.py`'s per-fold
z-scoring is safe (MEASURED, sec 7 of the output).

## 2. Refutations and corrections

### 2.1 Chain slowdown: not `gold_infer.py` (REFUTED as attributed; MEASURED)

**Timing (`ps` etimes):**
- Fold 1 started 08:25:14; `gold_infer.py` (pid 314726, ~1150% CPU, default `NTHR=12`) started 08:32:49.

**Fold-1 epochs, raw-end / full including EMA eval (min):**

| epoch | raw-end | full | what else was running |
|---|---|---|---|
| 0 | 12.9 | 14.0 | gold_infer |
| 1 | 11.5 | 12.6 | gold_infer |
| 2 | 11.5 | 12.7 | gold_infer |
| 3 | 17.2 | 19.3 | gold_infer + TTA bench on the GPU |
| 4 | 19.2 | 21.6 | gold_infer + fast_build GPU jobs |

- J1 alone ran 11.2 to 12.4 raw-end, at 0.36 s/it in-epoch.
- So `gold_infer.py` cost at most ~0.5 min per epoch while it ran with the chain.

**The slow epochs coincide with GPU co-tenancy:**
- **Epoch 3 (~09:04 to 09:23):** `~/jump2/tta/run1.log` ran `tta_bench` with `dev cuda`, "capped at 3.0 GB", for
  17.6 min, ending 09:27:46.
- **Epoch 4 (~09:23 to 09:45):**
  - `~/fast_build` GPU jobs: tiny tests 09:30 to 09:33;
  - `speed_ab.py` (pid 339781, nice 5) since 09:33:17, holding **8.9 GB** of the V100 (`nvidia-smi`, 09:47);
  - its own log records the chain's process (3.0 GB) as "other".

**Size of the slip:** already ~16 min by 09:46 (epochs 3+4 vs J1). It keeps growing while the GPU is shared, so the
track's "10 to 20 min" is already used up. `auto_folds.sh` only checks GPU idleness before each fold, not during one.

**Write-up error:** `pseudolabel.md` line 5 says box work used "2 threads". `gold_infer.py` defaults to `NTHR=12` and
ran at ~1150% CPU.

The recommendation to cap CPU threads at 4 is harmless, but it is not the measured fix. The fix is no GPU co-tenancy
during chain epochs. That concerns other tracks: the TTA bench and fast_build both used the GPU, which this workflow's
rules forbid unless the user approved it (HANDOFF 09:45 says Eesh approved Kaggle GPU for training; the box GPU is not
mentioned). Flag it to the operator. Do not act on it from here.

### 2.2 gold58 of mix50 (and the Dread-share line) is optimistic (CORRECTION; MEASURED from docs)

- `AUDIT_REPORT_2026-10-04.md` line 100: mix50's "teacher was epoch-selected on gold58".
- Tian ledger row 93: "Raptor 0.9254 (optimistic: its authors picked epochs on gold)".
- K3 receipts: the CoAtNet g96 package checkpoints are "top three official Gold58 macro-AUC epochs".
- The Raptor proxy is **in-sample on the 867** (AUDIT sec 5), so rho/ws are valid for **ordering** only.

Consequences:
- "Student minus target" for F1mix, TTRmix (-.012, -.013) and F1gated (-.026) partly reflects the target's selection
  optimism, not only student weakness.
  - GUESS optimism: about +.003 to +.005 on mix50 (half of a best-of-K epoch bias on 58 studies).
  - The v4 students' gaps (-.015, -.032) are clean.
- The "Ev4 below the Dread-share line by .010 to .019" comparison is biased against own-OOF teachers by roughly the same
  amount. Corrected guess: about -.008 to -.016.
  - Still negative.
  - The v09t LB null (-0.001) independently supports it.
  - So **recommendation 1 stands**.
- Gate T (target gold58 >= mix50 - .005) uses an optimistic comparator, which makes it conservative. Acceptable.
- Student gold58 values stay clean: gold74 never enters any student or teacher training; only teacher epoch choice saw
  gold58.

### 2.3 "Archit's precondition fails for us": too strong (PARTLY REFUTED)

What Archit actually said (CLAIMED, `t735304` 09-17 and 09-21):
- His trigger was overall OOF on the 58 (.930) beating "the extracted labels". He gave no number for his labels; the
  "~.89" in the track comes from Scott, Raymond and Tucker, so it is an inference.
- He then filled **silent cells**, because that is "where my image model was already beating the report labels".

Our data (MEASURED):
- **Overall check:** fails for the v4 students (.861 / .878 vs .893). Our mix50 students *pass* the literal check
  against the text labels (+.021), but that strength is Dread-derived.
- **Silent-cell check (Archit's fill rule):** our v4 students already beat v4 there (.70 / .71 vs .574 within-cell
  AUC).
- **Target-level payoff of a silent-only fill:**
  - Ev4 UNK .75 = .8993 (+.007, P .84);
  - TTRv4 UNK .75 = .8988;
  - below Ev4 ALL .5 and below the Dread-share line.
- **Caveat:** within-cell v4 labels are nearly constant (UNK about .25-.28), so beating them inside a cell is a low bar.

On the LB:
- Our ledger nulls (`v09t`, `v09o`) are **all-cell** 0.5 mixes. No silent-only own-OOF target has been read on the LB.
- So "two LB nulls for this idea" is accurate for flat self-distillation, not for Archit's exact variant. The gold58
  analog says the silent-only variant is weaker anyway (GUESS: LB null as well).

The track's recommendation 3 trigger ("student gold58 minus target gold58 > 0") is ambiguous for a mix50-type target,
because of 2.2 and because the students already beat the text part. Restate it as: student gold58 minus the
**gold-free** text-label target (v4 or a new LLM table) > 0 **and** the student is less parent-like than J1.

### 2.4 Gate S cannot separate the R2 effect from seed or recipe noise (REFUTED as a decision rule; MEASURED + MODEL)

Applied to the existing members (`vpl_check.out` sec 6):
- F1 mix50 gives gold58 .9144, rho .8889, **ws .82601**. PROMOTE needs ws <= .826: **missed by 0.00001**.
  - So the PROMOTE band coincides with the F1-vs-TTR **recipe** difference of plain mix50 members (rho -.007,
    ws -.010).
- Seed noise has not been measured for our TTR recipe. Tian's measured seed twins:
  - `v09r` / `v09u`: gold58 .9093 / .8995, a delta of **.0098**, with Spearman .952;
  - `v11a` / `v11b`: .9204 / .9167.
- The KILL bar gold58 < .903 (J1 - .010) therefore trips at about one seed-SD of a single model (GUESS P(KILL by noise)
  ~.15-.25).
- The KILL bars rho > .900 and ws > .840 sit .004 above J1, inside plausible seed noise.

Outcome odds: HOLD ~.6 / KILL ~.25 / PROMOTE ~.15 are mostly noise odds.
- A PROMOTE would not show that round 2 decorrelates, so it should not unlock 24 more V100-h by itself.
- Minimum fix: PROMOTE also needs the r_eff read the track logs without gating (<= .906 vs J1 .916), plus a
  parent + R2 W .50 LB read before the 4 extra folds.
- Better: let the execution track's r2 gate decide. Its bar (c) requires PL <= J1 - .020 and adds the "(d) beats
  frontier" test.

### 2.5 `build_pl_tm5.py` (BUG RISK + mis-described dry run; MEASURED)

- **Footgun:** full mode only checks that 5 OOF csvs exist. A running fold writes `oof_f{k}_s1043_raw.csv` from its
  first epochs. Fold 1's was written at 09:21 with cv .8502, mid-training.
  - So the script, run during fold 4, silently builds labels from a partial fold.
  - `~/jump2/r2/build_r2_targets.py` skips runs without the closing `{"raw":..,"ema":..}` line.
  - Port that guard if this builder is ever used.
- **Dry run mis-described:** `build_pl_tm5_dry.out` printed "folds used [0, 1] | train rows 1734". The ws/rho part of
  "gate T PASS on fold 0" therefore mixed in fold 1's partial OOF (pl_tm5 ws .7768 vs mix50 .8402 on those rows).
  - On fold 0 alone, gate T still passes: gold58 .9267 >= .9212; ws .8133 <= .8376 on the 867.
  - The conclusion is unchanged; the description is wrong.
- **New plan breaks paths:** if folds 3 and 4 come from Kaggle, their OOF csvs must land at
  `~/fleet_runs/f{3,4}_s1043_ttr_mix50/oof_f{k}_s1043_raw.csv` with exactly fold-k rows. Their checkpoints must also
  reach the box for `gold_infer.py`; otherwise gate T has no fold 3-4 gold predictions.
  - The Kaggle notebook "self-calibrates epochs", so it may not be the 24-epoch J1 recipe.
  - That makes the teacher set heterogeneous (GUESS: small effect, but say so).

### 2.6 Timeline (STALE / OPTIMISTIC; MEASURED)

- J1 ran alone from 23:12 (`auto_j1.log` launch) to 04:13 (log closing line): **5.0 h**, not 4.8 h. The track's
  11.2-12.3 min per epoch excludes the ~1.1 min EMA eval pass.
- Old plan (box does folds 1-4): 08:25 + 4 x 5.0 h + slips gives about **04:45 10-06 or later**, not 03:45. The probe
  read would then come around 10:00-10:30 10-06.
- Current plan (HANDOFF 09:45: box folds 1-2, fold 1 now ETA about 13:40 not 13:00; Kaggle folds 3-4; "all OOF by
  ~22:00, then round 2"): a probe could start around 22:30 10-05 and read around 03:30-04:00 10-06.
- Either way it lands well before 10-15 and 10-22. Feasible.
- The probe slot is contested:
  - The execution track's `auto_round2.sh` (a50: 0.5 v4 + 0.5 mix50-OOF on all cells; gate (a)-(d); `flock gpu.lock`;
    closing-line guards; seed 1543) is built and dry-tested but not started.
  - Its own `GATE.md` sec 8 flags this track's `pl_tm5` as a near-twin, and predicts its own bar (c) FAILS
    (GUESS P(pass) ~.2).
  - The two tracks agree: a mix50-OOF teacher re-teaches mix50.
  - **Run at most one.** I would use the guarded r2 pipeline (`R2_VARIANT=a50s75` is the pl_tm5 analog), not a second
    builder plus gate S.

### 2.7 Forum and literature (MEASURED against the local thread copies)

| claim in track | check |
|---|---|
| Archit Konde (53): single fold CoAtNet@224 **.950** LB; OOF on 58 **.930**; masking nothing; replacing worse than mixing; rounding hurt; "in fold ones just parrot"; 50/50 jumped, heavier helped | confirmed (09-16 to 09-21 posts). His label AUC is **not stated** (the ~.89 is inferred) |
| Raymond Yuen (48): .938 -> .94 -> .949 "still label"; multiple weak-label sources -> teachers -> pseudo-labels; best label ~.89 gold, model ~.925 gold, LB .947 (742327) | confirmed. He also says his label was "picked by LB, not by its AUC against the 58" |
| tennogh (5): .942 / .943 TTA @288; "OOF pseudo-labels pretty well correlated with LB"; own labels not better | confirmed; ambiguous as the track says |
| Scott Willis (10): Gemma-4 labels ~.89; .949 single fold, ~.930 gold | confirmed (09-05). Earlier .947 was a 5-fold. It is evidence that students beat labels, not of a pseudo-label gain |
| Chris Deotte, Nicolai Karcher (50/50 bootstrap: gold and CV up, LB 0), Tom Aindow (masking / replacing no effect), Tucker Arrants | confirmed. Tucker's "simple Qwen ... low 0.95" is in thread 745214 (rank 47 there), not 735304 |
| SpeedSci (126, 745214): label version .945 gold58 -> .910 LB | confirmed with nuance: an EffNet model scored .945 on gold with .910 LB under one label version. SpeedSci also says label changes moved DINOv2 LB .931 -> .942 |
| Tian ledger: v09t .917 vs .918; v09r +.009; v09o / v09o2 .927 / .927, pair .928 vs .930 | confirmed (`experiments.md` rows 80, 83, 101, 103). Add: v09r's +.009 is a **solo** gain; inside the 0.942 stack at beta .10 / .20 it read .942 / .941 (`measured_levers.csv` row 75). So "only Raptor moved the LB" is true for solo reads only, which strengthens the "parent-family info adds nothing to the parent blend" reading |
| NTejas silent-cell OOF repair -.0064 gold58 | confirmed (`measured_levers.csv` row 63) |
| LIT: "Students track their teachers closely (Stanton et al. 2021)" | **REFUTED (misread)**. The abstract says "there often remains a surprisingly large discrepancy between the predictive distributions of the teacher and the student" (arXiv 2106.05945). The track's own transfer data agree with Stanton, not with the sentence: target ws -.33 moved the student only -.03, a ratio of .09-.15x (v4 vs mix50 gives .13-.15x, not .09x). The track's practical conclusion (student parent-likeness is set by image content, not the mix formula) actually gets stronger |
| LIT: Reed 2015, Furlanello 2018, Mobahi 2020, Allen-Zhu and Li 2020, Xie 2020 (needs extra unlabeled data) | consistent with my knowledge; not re-fetched (uncertain only on exact wording) |

## 3. Leakage and rule checks

- **gold58 contamination of students:** none found. gold74 is fold -1, `--exclude`d (trainer log), and no overlap with
  the 867.
- **gold58 contamination of targets:** yes, via teacher epoch selection (2.2).
- **Selection on gold58 by this track:**
  - The R2 weights (.75 silent / .5 / .5) are the top TTRmix row of 7 on gold58.
  - They also equal the pre-existing mixSA spec (Tian, 10-03: teacher .75 silent, .5 elsewhere), so the selection risk
    is low.
  - Still, the +.0006 vs mix50 is a best-of-7 read.
- **Stacking leak** (fold 1-4 teachers trained on fold-0 rows): correctly flagged by the track ("never gate on val AUC
  vs v4"). It also slightly *raises* the probe's parent-likeness on the 867, which biases against PROMOTE
  (conservative).
- **Rules:**
  - The track stayed CPU, nice 19, wrote only `~/jump2/`, and touched no process.
  - Its round-2 student command writes `~/fleet_runs/f0_s1044_ttr_pltm5`; that is an operator action, not an agent one.
  - `gold_infer.py` at 12 threads contradicts its own "2 threads" note but is within the rules.
  - The GPU co-tenants (TTA bench, fast_build) belong to other tracks; see 2.1.

## 4. Corrected recommendations

1. **Keep:** no TTR v4 folds 1-4 as a self-teacher, and DSD dead.
   - Evidence after correcting for 2.2: Ev4 is still about -.008 to -.016 below the line, a TTRv4-only teacher is
     weaker still, and the v09t LB null holds.
   - This also overrides `~/jump2/r2/GATE.md` sec 6's fallback ("FAIL on (a)/(b) -> next teacher = 5-fold TTR v4
     OOF").
2. **Keep, modified:** at most one round-2 probe, through the guarded `auto_round2.sh` (a50 or a50s75), only into
   otherwise-idle V100 time after the chain or Kaggle folds and after K4.
   - Do not run `build_pl_tm5.py` as well; if it is ever used, port the closing-line guard.
   - Treat gate S's PROMOTE as non-informative unless r_eff and a W .50 LB read agree.
   - Expected: HOLD-type member, true lift about +.0000 to +.0002 (MODEL).
3. **Keep, restated trigger:** re-open pseudo-labels when a student beats a **gold-free text target** on gold58 at
   lower parent-likeness than J1; then rerun `pl_sim_b2` / `b3` (~5 min CPU).
   - A silent-only fill (Archit's variant) is the one untested-on-LB form. Do not spend a GPU slot on it with v4
     students: target-level analog +.007, below the line.
4. **Keep:** the strength problem belongs to the recipe and labels tracks.
   - Weak spots on gold58 (v4 students vs v4 label): MCL .75-.90 vs .968 (only 9 positives), ACL .90-.95 vs .987,
     Synovitis .62-.63 vs .790.
   - Text-level pooling of independent LLM tables first (Raymond's mechanism).
5. **Replace:** "cap CPU threads" with "no GPU co-tenancy while a chain fold trains".
   - Measured slip about 16 min by 09:46 and growing.
   - `speed_ab.py` held 8.9 GB of the V100 at 09:47.
   - Thread caps are fine but secondary.

**Track gain (unchanged):** public +0.000 central (-.001 to +.002), P(display >= .945) < .05 (MODEL/GUESS). Not a
route to silver.
