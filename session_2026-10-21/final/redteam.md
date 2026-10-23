# redteam.md: trying to break the final pair A = swide 57464326, B = R6 57248084 / clean r6pw20 (10-21 ~21:45 IST, Claude, FINAL red-team)

RESEARCH + LOCAL ONLY. Nothing was submitted, pushed, committed, versioned, downloaded, posted or turned into a dataset. No box use
at all (no OAI row, id or per-knee value read or printed), no GPU job, no Kaggle or LB call, no scheduled-task edit. PREREG, HANDOFF,
PROGRESS and the live folders were only read. Everything new is in `session_2026-10-21/final/`. Local CPU only (2 threads).
**Tags.** MEASURED = computed now from files (named). CLAIMED = older note (named). MODEL = arithmetic on stated inputs. GUESS = judgement.
**Units.** E = private margin in 1e-4 macro over the heliosli copy block (bpick / apick ruler); pub-scale = .95802 + E x 1e-4.
Silver = P(private rank <= 315) on the step (10-19 field) / smooth (shakeup, incl. family shock) curves. "pts" = percentage points.
gold58 here is used only to price failure paths (row-loss costs), never to rank entries.

## 0. Verdict

1. **A = swide survives.** About 40-70 % of its public lead over pw25 is read noise, mostly MCLmm (MODEL). That luck does not carry
   to private. What carries is the stack's population part: +1.08e-4 under the A22 priors, +0.41 at the central "levers" weighting,
   -0.33 if MCLmm is truly EB-negative (CLAIMED apick). The only A that beats swide in any world is pw20, and only on step silver,
   only once >= ~20-40 % of the prior sits on "stack harmful" worlds. Max regret for keeping swide: 2.0 step / 0.9 smooth silver pts.
2. **B rule survives.** R6 stays the free fallback. It is also the most diversifying scored entry once the family shock is
   included: effective private corr with swide .85, vs c4 .90, r6pw20 .93, eoa10 .96, pw20 .99 (MODEL). eoa10 ties R6. pw20, c4,
   pw25 and eoa20 lose silver. The r6pw20 resubmit stays positive: +0.3-0.4 step / +0.6-0.8 smooth silver at P(clean) .5-.64.
3. **No hidden partial failure can be large in swide's run.** Both all-or-nothing canaries are excluded by .960 (Fracture would show
   ~.93, ACL .916). There is no Synovitis fleet canary in this lineage: cell 9 has USE_F False ("no fleet cell in this kernel"),
   MEASURED. One silent path is stochastic: R6-leg rows lost to a worker crash or a 900 s loader stall take Raptor's rank, with no
   canary. Given the .960 read, P(it hit swide) <= .02 even at a 20 % prior. Expected private bias <= .03e-4 (MODEL `silent_risk.log`).
4. **The family shock is the dominant risk, and no entry of ours hedges it.** It is ~78 % of the private variance of the pair
   (sd ~19e-4 common, CLAIMED shakeup). It costs ~11 silver pts (pair .876 step -> .763 smooth). Our best non-family entry is .951,
   about 3.7 shock-sd below, so a "canary-proof shock hedge" does not exist.
5. **New side finding (MODEL):** the 10-21 data argue against a generic compact_infer crash as the S6F canary cause. LR timing:crash
   is 2.7-60, so posterior P(crash path) falls from .5 to ~.05-.27, unless the crash is S6F-specific. This raises the guard variant
   g85j to ~+.07 step / +.14 smooth silver per copy. That is still < .2 pt, and it would need a build, push, commit and gate tonight.
   No change to the plan.
6. **Changes recommended: none to A, B or the 10-22 slate.** One optional, free hygiene line for the A23 text (sec 3.4): if two or
   more r6pw20 copies qualify and their displays differ, take the higher display (byte-identical copies should match, so a lower
   one flags a silent failure); ties go to the earliest id. It is worth ~.01 pts. Skip it if it complicates the rule.

Final pair numbers (CLAIMED apick / bpick; A22 priors | levers weighting):

| pair | E (pub-scale) | silver step / smooth | P(r<=150) step / smooth |
|---|---|---|---|
| swide + R6 | +17.33 (.95975) \| +16.74 (.95969) | .876 / .763 \| .865 / .751 | .561 / .446 \| .540 / .431 |
| swide + clean r6pw20 | +17.86 (.95981) \| +17.36 (.95976) | .882 / .776 \| .875 / .767 | .594 / .464 \| .578 / .452 |
| pw20 + R6 (the only A challenger) | +15.62 \| +15.63 | .874 / .745 \| .873 / .746 | .507 / .403 \| .508 / .403 |

## 1. Attack: "swide's .960 lead over pw25 / pw30 is MCLmm luck"

**Evidence**
- Public gap swide - pw25 (posterior, e-4): +2.03 (bpick, uniform band ruler) or +1.39 [+.75, +2.11] (surface, exp band ruler).
  The prior was +0.24 (CLAIMED bpick / surface).
- Decomposition of the +2.03, base posterior (CLAIMED apick sec 1):
  - population 54 %: lomm05 +.44, I +.16, MCLmm +.50;
  - read noise 46 %: MCLmm +.78 (39 %), lomm05 +.15.
- Cross-check, Gaussian-conjugate model with tighter population priors (sd muL .8, muM 1.2; read sd 1.0 / 2.4): MCLmm read noise
  +1.16 and lomm05 +.20, so 67 % noise (MODEL, inline arithmetic).
- In the MCLmm-EB world the population part is negative: noise explains 116 % of the gap (CLAIMED apick sec 1).

**So: "partly luck", 40-70 % (more in EB worlds), and most of the luck is MCLmm.** MCLmm's public read sd (2.4) is about twice its
prior sd (CLAIMED surface sec 5).

**What it means for private** (private noise is independent of public, so only the population part carries)

| weighting (CLAIMED apick sec 3) | private stack E | P(swide priv > pw25) | P(swide priv > pw20) | swide - pw25 silver (B = R6) | swide - pw20 silver |
|---|---|---|---|---|---|
| A22 priors | +1.08 | .67 | .68 | +1.4 / +1.9 | +0.2 / +1.8 |
| levers (central) | +0.41 | .56 | .61 | +0.4 / +0.7 | -0.8 / +0.5 |
| skeptic | 0.00 | | | -0.3 / -0.2 | -1.5 / -0.3 |
| EB-heavy | -0.33 | .44 | .52 | -0.8 / -0.8 | -2.0 / -0.9 |

- Break-even (CLAIMED apick sec 4): swide >= pw25 if the true private stack >= ~0 (-0.2 to +0.25). Versus pw20: smooth silver needs
  >= ~+0.15-0.2, r150 / E need >= ~-0.6 to -0.9. Step silver vs pw20 is a tie even at the full posterior.
- pw25 is never the right fix. In the worlds where the stack hurts, pw20 beats pw25 too (CLAIMED apick sec 0.2).
- Why I would not weight EB-negative MCLmm above ~20-30 % (GUESS):
  - The report rulers that make it negative (-16 to -48 label, CLAIMED LEVERS_1021) are the same class of ruler that understated
    E-OA 2-5x in the expert regime (CLAIMED shakeup sec 6, TRICK_1019).
  - The read itself favours non-EB worlds (Bayes factor ~1.3-2.8, CLAIMED apick sec 1).
  - Against swide: gold58's MCLmm support comes from 9 MCL+ knees that also nominated it (selection).
- Model instability worth knowing (CLAIMED): dose.md (10-20, before the A22 read) said A = pw20 for the private board, +3.4 silver
  over pw30. After the read, apick finds pw20 vs swide a tie on silver and a loss on r150 / E. The pw20 edge came from pw30's wide
  dose tail. swide sits at the pw25 dose, the tail most of that edge came from is gone, and the stack is the only remaining question.

**Verdict: KEEP A = swide.** A switch to pw20 is defensible only as a written, medal-only stance (bronze / silver, not rank) with
>= ~40 % weight on EB-negative MCLmm. It must be fixed before the selection page opens (as LASTDAY sec 4.1).

## 2. Attack: pair correlation and the family shock

**MEASURED from the model** (`rt_bpick_mc.py` = bpick_mc.py base + one print block; reproduces bpick_base.log to the digit;
`rt_bpick_base.log`). sd(A = swide) 12.07e-4.

| B | sd(B) | idiosyncratic corr with swide | effective corr, shock sd 10 / 19 / 25 | B-alone E | P(B > A) |
|---|---|---|---|---|---|
| **R6** | 7.51 | .31 | .66 / **.85** / .90 | +2.74 | .12 |
| c4 | 6.89 | .57 | .78 / .90 / .93 | +5.62 | .12 |
| r6pw20 (clean) | 10.35 | .72 | .84 / .93 / .95 | +10.83 | .25 |
| eoa10 | 7.81 | .86 | .91 / .96 / .97 | +11.31 | .19 |
| pw20 | 10.25 | .97 | .98 / .99 / .99 | +15.05 | .32 |
| pw25 | 11.82 | .98 | .99 / .99 / 1.00 | +15.49 | .33 |

- The shock is a common term: the family's public luck vs the independent lineages, redrawn privately (CLAIMED shakeup sec 0.4).
  At sd 19 it is 72-79 % of each entry's private variance. B can act only on the remaining ~1/4.
- That is why B adds only +3.3 step / +2.5 smooth silver over swide alone (.843 / .738 -> .876 / .763; CLAIMED bpick). The shock
  itself costs ~11 pts (step -> smooth gap).
- **No hedge exists** (MEASURED `canary/work/subs_all.json`):
  - best non-family entries: FN11 .951 (10-10, includes a public CoAtNet member); own-only K7i SOLO .944;
  - to rescue a family-wide private drop they would need to gain ~+70e-4 relative to the family, ~3.7 shock sd.
  - A non-family B costs far more than it hedges. This agrees with shakeup sec 0.7.
- **Implication for B:** among scored entries R6 is the most diversifying on both scales, so the A22 fallback is also the
  correlation-minimising one. The E-OA dose tail is the one idiosyncratic risk R6 covers and the c4-side B's do not:
  - hedge given E-OA private < 0 (P .13): R6 +4.56, c4 +5.00, eoa10 +4.89, r6pw20 +3.05, pw20 +2.38, pw25 +0.56 (CLAIMED bpick sec 2).

## 3. Attack: should B be a different scored entry, or a canary-proof choice?

### 3.1 Scored alternatives (CLAIMED bpick base, A = swide; vs B = R6, silver step / smooth, r150 step, E)

| B | silver | r150 | E | rows positive (19) | verdict |
|---|---|---|---|---|---|
| eoa10 | +0.0 / +0.1 | +0.2 | +0.02 | 15 / 19, range -1.35 / +1.00 | tie: a bet on the c4 table vs R6's uniform W |
| pw20 | -0.7 / -0.3 | +0.9 | -0.07 | 4 / 19 on silver | rank-only bet |
| c4 | -0.5 / -0.4 | -0.3 | -0.09 | 1 / 19 | loses |
| eoa20 / pw25 | -1.3 / -0.9, -1.9 / -1.1 | +0.3 / +0.8 | < 0 | 0 / 19 | lose (near-twins of A) |

- Model dependence (CLAIMED picks2 sec 0.2 vs bpick): with A = pw30, picks2 preferred pw20 over R6 as the fallback. With A = swide,
  bpick prefers R6 by +0.7 / +0.3. The flip is driven by A's dose tail, and both gaps are < 1 pt. The pre-registered fallback (R6)
  is the right default. Changing it now would be a post-read fork for <= 0.15 pts expected.

### 3.2 "Canary-proof" B

- Every scored entry is canary-proof: picks are fixed scores, and public and private come from the same finished run (CLAIMED
  CANARY_1021). The resubmit cannot hurt B through a canary or an ERROR: B just stays R6.
- Fixing B = R6 now, with no 10-22 dependency, gives up +0.3-0.4 step / +0.6-0.8 smooth silver at P(clean) .5-.64 (CLAIMED bpick,
  skeptic). Use it only if there is no GO or no 05:31 mechanism.

### 3.3 Residual risks of a clean r6pw20 copy as B

1. **E-OA transfer to the R6 base (k).** k ~.3 gives -0.3 / -0.1. k ~0 gives -0.5 / -0.3. Breakeven is ~.35-.45. GUESS .75-.8 that
   it holds (CLAIMED bpick sec 7 #3).
2. **Silent degradation that passes the .958 gate (NEW, MODEL `silent_risk.log`).**
   - Mechanism: an R6 worker crash or a loader stall in the copy's run, with no canary (sec 4).
   - Clean r6pw20 public margin over .9580 is ~N(9.7, 3.9)e-4 (from bpick P(.957 / .958 / .959) .006 / .522 / .472).
   - gold58 row-loss cost for r6pw20 is 1.4 / 9.5 / 11.6 / 27 / 66e-4 at f .02 / .05 / .1 / .2 / .5.
   - Given a crash, the copy passes the gate with P .14-.30 and lands privately below R6 with P .06-.07.
   - Expected B shift: -.02 to -.35e-4 for crash priors .02-.2, i.e. <= .03 step / .05 smooth silver pts.
   - This confirms the skeptic's "< .02" order of magnitude. Not a reason to drop the resubmit.
3. **Selection error** (wrong id, or Kaggle auto-select = swide + a .959 near-twin, -1.9 / -1.1 silver): unchanged LASTDAY sec 4.10.

### 3.4 Optional free hardening of the A23 B line (text only, Eesh's call)

> "... B = the earliest-submitted copy that is COMPLETE, shows no canary and displays >= .958. If more than one copy qualifies at the
> selection read and their displays differ, B = the qualifying copy with the higher display (earliest id on ties); otherwise R6
> 57248084."

- Byte-identical copies run on the same hidden set, so they should display the same. A lower display flags a silent partial failure.
- Value ~.003-.03 pts (MODEL). It moves B's fixing from "first qualifying read" to "selection read" (~18:00 IST), and reads land by
  ~14-16 IST anyway.

## 4. Attack: did swide's own hidden run partially fail without a visible canary?

### 4.1 Canary inventory, swide notebook (MEASURED code: `stack/kernels/kernel_v224r6c4pw25lomm05mcl10j`, cells 8, 9, 18)

| path | canary? | effect if it fired | excluded by .960? |
|---|---|---|---|
| r224 skipped / failed / unusable (cell 9) | Fracture = .5 | ~-.03 macro, display ~.93 | yes |
| S6F any failure: guard, kill, receipt, sha, ids, coverage < .98, rho floor (cell 18) | ACL = .5 | display .916 | yes |
| Synovitis fleet | none: USE_F False, "no fleet cell in this kernel" | n/a | n/a |
| Raptor stage failure / > 3 % Raptor rows failed / > 50 % R6 rows unscored | notebook raises -> ERROR | no score | yes |

### 4.2 Silent paths (no canary) and their price (MEASURED replay `fallback_g58.py` -> `fallback_g58.log`)

The replay runs the real stage-3 fail logic + cell 18 on the gold58 serves. It reproduces the clean scores: swide .94492, pw25
.94368, c4 .94113, R6 .93960, r6pw20 .94175. Costs are mean e-4 over 300 random row sets.

| silent path | stochastic per run? | gold58 cost, swide (r6pw20) | note |
|---|---|---|---|
| R6-leg rows unscored -> Raptor rank: worker crash mid-run (probe_dual does not kill the sibling); DataLoader item timeout 900 s (compact_infer default; the R6 controller passes no `--item-timeout`), OOM | **yes** (I/O stall / crash) | f .02 -2.3 (-1.4), .05 -10.4 (-9.5), .10 -19.2 (-11.6), .20 -32.8 (-27.2), .50 -76.1 (-65.8) | predictions.csv is flushed per study, so a crash at fraction q keeps the rows before it: f ~ U(0, .5) |
| R6 controller deadline (8.3 h) cut | excluded | n/a | R6 runs before S6F. A cut needs age > 8.3 h, but S6F checks its 5.75 h start guard later, so the ACL canary would fire |
| S6F coverage in [.98, 1) (uncovered rows take the base rank in the S6fall slot) | yes | 2 % rows: -0.01 (+0.75) | negligible |
| Raptor per-study failures <= 3 % | no (same hidden data every run) | 3 %: -0.33 (-0.85) | common to every entry incl. R6 and pw25: no differential, already in all LB levels |
| Raptor blank-study garbage path (kernel_audit 4.1), compile eager fallback (dp 2.1e-4) | no / ~0 | ~0 | common |

- gold58 is a hard set (macro .94 vs LB .96), so I price LB cost at h = .5-1 x the gold58 cost (GUESS).
- Loader stalls are bounded: each costs 900 s of its worker's wall time, so a run that kept the S6F start under 5.75 h (~1.5-1.7 h
  of slack) can have lost at most ~6 studies per worker, <= ~12 (~1 %) in total. That is <= ~2e-4 on gold58 scale, and ~0 in
  expectation (MODEL).

### 4.3 Commit outputs (MEASURED `val/a22/out_*/` receipts; 3 visible rows only)

- All 5 A22 commits are clean on every flag: R6 fallback 0, worker exits {0, 0}, raptor_failed 0, ours_failed 0, V224 canary none,
  r224 used, S6F mode S6F, coverage 1.0, rho .78 (r6pw20 .77). swide's gate is PROBE_CHECK PASS 26/26.
- They prove the code paths, not the hidden per-study outcome. That is the general limit already stated in kernel_audit 3.4.

### 4.4 What .960 implies (MODEL `silent_risk.py` -> `silent_risk.log`)

- A degraded swide would have needed a clean stack gap of Dhat + c(f) to still read .960 above pw25.
- Under the stack prior (public sd 2.5-3.5), the likelihood ratio crash : no-crash is .03-.08.
- P(silent R6 crash in swide's run | read) = .001-.019 for prior pi .02-.20. Given one, its expected cost is only 1.1-2.1e-4
  (the read cuts off larger losses).
- Private bias of the no-crash model: <= -.03e-4. Any degradation is already inside the public score, and private came from the
  same run, so the read self-corrects. **Nothing to act on.**

### 4.5 Side result: timing vs crash as the 10-21 canary cause (MODEL `crash_vs_timing.py` -> `crash_vs_timing.log`)

- The R6 stage runs the same compact_infer.py (2 workers, ~2.5-3.5 h) with no canary on crash. So a generic worker-crash hazard
  large enough to cause 1.5-3 of the S6F canaries would also have silently hit the R6 stage in 36-80 % of the 10-21 runs.
- None of the 5 shows a display anomaly: swide .960 is best-id, pw25 .959 is where the surface put it, and all three canaries read
  exactly .916. P(that | generic crash) .02-.37, so LR timing:crash 2.7-60. Posterior P(crash path) falls from .5 to ~.05-.27.
- Caveat: a crash mode specific to the S6F run (its decode-bound load, or run_group's sibling kill) is not tested by this.
- Consequences:
  - (a) It supports the low silent-crash prior used in 4.4.
  - (b) P(clean) on 10-22 is unchanged (it is regime-based).
  - (c) The guard variant g85j rescues only timing-path canaries, so its value rises from +.04 / +.08 to ~+.07 / +.14 silver per copy
    (MODEL, scaled from cause.md / bpick sec 7 #8). Still < .2 pt, and it needs a new slug, push, GPU commit, gate, registry and A23
    text before ~04:30 IST. It is now ~21:40 IST. **Keep it skipped** unless Eesh wants it and main can finish by 23:00.

## 5. What would change the picks (pre-commit list)

| observation | action |
|---|---|
| Eesh writes, before selection, a medal-only stance with >= ~40 % weight on MCLmm being harmful privately | A = pw20 57425269 (not pw25) |
| No GO / no 05:31 mechanism | B = R6 57248084, final |
| r6pw20 copy COMPLETE, no canary, >= .958 | B = that copy (earliest; optional higher-display tie-break, sec 3.4) |
| r6pw20 copies all canary / ERROR / < .958 / PENDING at the hard stop | B = R6 |
| Any 10-22 entry displays .960+ | never A (A23); A stays swide |
| A host ruling against an ingredient of the family (e.g. Raptor weights) | outside this model: pause and tell Eesh (both picks contain the family) |

## 6. Files (`session_2026-10-21/final/`)

- `fallback_g58.py` -> `fallback_g58.log` / `.json`: gold58 replay of the silent R6-leg / S6F-coverage / Raptor-fail paths (real cell 8
  fail logic + cell 18 transforms; reproduces the CLAIMED gold58 scores).
- `silent_risk.py` -> `silent_risk.log`: posterior of a silent failure in swide's run given the read; leak-through of degraded
  r6pw20 copies past the .958 gate.
- `rt_bpick_mc.py` -> `rt_bpick_base.log` / `.json`: bpick_mc.py (paths to this folder) + sd / effective-corr-with-shock block.
- `crash_vs_timing.py` -> `crash_vs_timing.log`: canary cause likelihood from the 5 10-21 runs.
- Read: `lastday/{LASTDAY_1021,apick,surface}.md`, `canary/{CANARY_1021,bpick,cause}.md`, `private/{shakeup,dose}.md`,
  `val/picks2.md`, `trick/kernel_audit.md`, `val/a22/out_*` receipts + gate txt, swide / r6pw20 kernel notebooks, `canary/work/subs_all.json`.
