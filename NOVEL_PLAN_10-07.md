# RSNA Knee: novel-levers plan after the 10-07 research + verification workflow (written 2026-10-07 ~12:10 IST)

Synthesis of five research tracks (site, recipe, per-label, ensemble, forum), each re-checked by an independent
adversarial verifier. Only claims a verifier confirmed, or flagged as uncertain (marked so), are used here. Where a
verifier corrected a track, the corrected number is used.

- Tags: **MEASURED** (script + output path), **CLAIMED** (forum, thread id), **MODEL**, **GUESS**.
- Noise: paired OOF SE ~.001-.002 on 867 fold-0 studies; public LB SE ~.004-.006 absolute, ~.001-.0015 for the
  difference between two highly correlated blends, plus display rounding of +-.0005. gold58 cannot resolve macro
  effects below ~.008.
- `SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`.
  Track reports are `SP/novel/{site/site.md, recipe.md, perlabel.md, ensemble.md, forum.md}`. Verifier reports are
  `SP/novel/verify_{site,recipe,perlabel,ensemble,forum}.md`.

---

## 0. Box check (read-only ssh, 11:57 IST; nothing touched)

- **GPU:** V100 at 100%, 14.8 GB, 80 C, SM 937 MHz (thermal throttle as before). PAUSE_CHAIN is absent. [MEASURED:
  nvidia-smi]
- **Running job:** TTR288 mix50 fold 1, seed 1044 (`~/fleet_runs/f1_s1044_ttr288mix50`; trainer pid 524330 under
  `~/auto_after_ttr288.sh`, pid 464971).
  - Epoch 17 of 24 (0-based) is in eval, at 0.45 s/it and ~16.5 min per epoch with both evals.
  - Best raw so far is epoch 16: val vs v4 **.8943** (checkpoint `cv8942`), gold58 .9135, gold74 .9080 (trainer
    convention). EMA .8916.
  - For comparison, the 224 fold-1 finals are .8926 (s1043) and .8917 (s1044).
  - [MEASURED: `~/fleet_runs/f1_s1044_ttr288mix50.log`]
- **ETA [MODEL]:** fold 1 ends ~13:40. Fold 2 ~20:20, fold 3 ~03:00 10-08, fold 4 ~09:40-10:10 10-08. The chain then
  prints `chain done`.
- **Nothing is armed after the chain.** `~/novel/recipe/auto_t320.sh` is staged and was DRY-tested only. The V100 goes
  idle at ~10:00 10-08 unless one queue is armed before then (sec 3).
- **Pending:** K7i (56896918 STRICT W .50, 56896921 SOLO) reads ~13:15. Teammate fold 4 (s1044) was not visible as of
  11:25.

---

## 1. Bottom line (honest)

1. **No lever found today closes the gap to gold.**
   - Gold is ~.959 (rank ~20 of 5,366) [MEASURED: LB page 11:33 IST, `verify_forum.md` sec 7]. We are at .946, so the
     gap is +.013.
   - Every verified fleet lever is worth **+.000 to +.001 blended** on its own. The optional image-aware relabel is
     **+.001 to +.003** (GUESS, likely the low end).
   - Combined, the fleet levers give a realistic **.947-.949 public** (GUESS). With a merge into an independent
     .94-.95 family, **.949-.953** (GUESS).
   - **P(gold): GUESS below ~5% even with a good merge, ~1% without.** The realistic objective is to hold silver on
     the private split. Silver was rank ~265 at .945 on 10-06 [MEASURED], and we are inside it by a hair.
2. **The 288 px set is the one real fleet lever, and it is smaller than fold 0 suggested.**
   - On v4 labels, TTR288 leads the 224 seed by +.0069 on fold 0 at the fixed last epoch (selection-free) and by
     +.0028 on fold 1 (SE .0014, last 3 matched epochs). That puts the two-fold single-model lead D at ~.004-.005 OOF.
     [MEASURED: `~/novel/verify_ensemble/out/v_ens.txt` V3]
   - **On expert labels the lead is ~0.** On gold74, T0-A0 is -.0003 (SE .0046). A .7 x 288 / .3 x A-set mix minus
     the A set is -.0023 (SE .0029) on gold74 and -.0045 (SE .0034) on gold58. [MEASURED: same file, V4]
   - About half of the fold-0 gap over J1 is run-to-run noise. Seed-matched, the gain is +.0045 pooled, not +.0073.
     [MEASURED: `~/novel/verify_site/v3_within.log`]
   - Expected public effect of the 5-fold 288 set in K8: +.000 to +.002 SOLO, +.000 to +.001 blended. [MODEL:
     `SP/novel/verify_ensemble/v_model_out.txt`]
3. **The most valuable result today is negative: three tempting OOF wins are traps.**
   - **Per-site/language intercepts.** They gain +.0063 OOF against report labels (.8861 -> .8924). On gold58 the same
     offsets cost -.0152 (.9221 -> .9069, P>0 .00). On the same 58 studies the paired difference "vs reports minus vs
     experts" is +.0207. The offsets encode report or extraction habit, not disease. [MEASURED:
     `~/novel/site/s2_s1043.log`, `~/novel/site/s5_gold_grid.log`, `~/novel/verify_site/v2_gold.log`]
   - **Co-occurrence stacking.** It loses -.00096 OOF (SE .00014; 3 independent implementations), -.0010 on gold74 and
     -.0013 on gold58. [MEASURED: `~/novel/stack_oof.log`, `~/novel/verify_forum/v2/vf2.log`]
   - **Per-label blend weights fitted on gold58.** In-sample +.002 only, against a parent side whose checkpoints were
     picked on gold58. [MEASURED + MODEL: `verify_perlabel.md`]
4. **Our weak labels are Synovitis and Fracture, and both are mostly label-side problems.**
   - Training labels badly under-call both against the experts: Fracture is .067 in v4 vs .310 in gold58; Synovitis is
     .122 in v4 (.266 in mix50) vs .466. [MEASURED: `~/novel/verify_forum/v2/prevalence.log`; matches gchauhan 741310,
     CLAIMED]
   - Against the parent's out-of-sample arms on gold58, only the Fracture gap is robust (P(d<0) .99 vs the CoAt family
     and vs the side). Synovitis and Medial Meniscus are directional. [MEASURED: `vf2.log` sec G]
   - Per-label OOF vs v4 does not predict per-label gold (rank r .08-.09). So label-side changes must be judged on
     gold58 and the LB, never on v4 OOF. [MEASURED: `~/novel/verify_perlabel/out/vp.txt` sec 8]
5. **The merge (Eesh's call, deadline 10-15 23:59 UTC) is still the biggest single lever.**
   - The pool of .95-class partners is shrinking: tanuki + raymond (.956) and RB + Tucker + Tony (.960) have merged.
     [MEASURED: LB page; CLAIMED: 735304]
   - One new candidate: Musa Tur Farazi, ".952 with 224px single model" (735304, CLAIMED; fold count unknown).

---

## 2. Ranked actions

Expected gains are public-LB unless stated. "Blended" means parent + fleet at W .50.

| # | action | expected gain | cost | first experiment | gate (pre-registered) | ready by |
|---|---|---|---|---|---|---|
| 1 | **Team merge** (Eesh posts; we never post) | +.002 to +.007 [GUESS] | 0 GPU; Eesh's time | Eesh messages Christoffer Thimsen (746060, compute-bound, own labels and models), Nikita Ageev (743374), Musa Tur Farazi (735304) | Partner states a SOLO >= .935 for a model with no public competition weights; merge agreed by 10-13 | 10-15 23:59 UTC |
| 2 | **K8 = parent + [5x TTR288 s1044 + 5x TTR224 set A s1043]**: equal members, raw checkpoints, TTA A+B, W .50, set B dropped | SOLO +.000 to +.002, blended +.000 to +.001 [MODEL]; expected main display .946-.947 | 0 extra GPU; K8 builder ~1-2 h; 3 commits (main, SOLO, KF twin); 3 slots | fold-1 close (~13:40): selection-honest T1-A1 on the last-3 epoch files (`~/novel/verify_ensemble/v_ens.py` V3) | K8 SOLO >= .943 (a read <= .941 = build problem: audit before any use); pick-1 tie rule in sec 5 | submit ~13:30-15:00 10-08; reads ~20:00-22:00 |
| 3 | **Guard:** no per-site, per-language, per-scanner, metadata or stacking post-processing in any kernel or final pick | protects .005-.015 behind a fake +.006 OOF [MEASURED gold58; LB size GUESS -.003 to -.010] | 0 | add to HANDOFF "Settled" | n/a | now |
| 4 | **frac75 Fracture-target probe:** the trainer's existing `--labels` flag only (`~/novel/labels_mix50_frac75.csv` = mix50 with Fracture re-mixed .25 v4 + .75 teacher) | SOLO +0 to +.002, blended +0 to +.001 [GUESS]; probe power only ~45-63% at a true +.02 Fracture effect [MODEL] | ~4.6 h probe (paired with J1); ~15 h folds 1-4; K-builder edit (per-set train_labels + key tag); 2 commits + 2 slots | fold 0, seed 1043, J1's exact argv except `--labels` and a new `--out`, grad_ckpt 1, 24 epochs; then `~/novel/gold_infer5.py` on CPU | Pin ONE gold74 convention first: masked, as the trainer and gold_infer5 score. G1: mean OOF vs v4 on the 11 non-Fracture labels >= .8774. G2: gold74 Fracture >= .912. G3: gold74 macro >= .9062. Report the gold58 Fracture sign. Any fail = close target re-mixing | probe gate ~15:30 10-08; set ~06:00 10-09; K9 read ~17:00 10-09 |
| 5 | **Image-aware (VLM) relabel of Synovitis + Fracture** (+ Lateral OA, PF OA), then retrain. Needs Eesh's explicit OK to spend the API key | Synovitis label AUC +.03 to +.05 [CLAIMED 745949]; blended +.001 to +.003 [GUESS], likely the low end | pilot ~$3.8 (batch) / $7.6 (standard); scale-up ~$15-50 on the uncertain subset (up to ~$125 for all 4,407 on Opus) [MODEL, kit estimates]; 1-2 days; ~20 h V100 at 224 (~33 h at 288) | the existing pilot kit (`~/gold_prep/relabel`), **after** the gate amendment below, registered before any live call | Demote G1 (Effusion img-text >= +.04, a hard veto) to a sign check (>= 0); keep S (6-target value >= +.010) as the spend decision; report the 5-target non-Effusion mean (SE .015) as a secondary. Do NOT use a +.03 bar on a 2-label mean (a ~1 SE gate). Retrained fold 0 bars to fix before launch: gold58 Synovitis >= J1 + .02 AND gold58 macro >= J1 - .003 [bar choice GUESS] | Eesh decides by ~10-10; set ~10-12 to 10-14 |
| 6 | **TTR320 native probe** (`--ttr_size 320 --bs 1 --accum 8 --grad_ckpt 0`, 21 epochs; no code change, model_sha unchanged) | OOF +.0005 over 288 (range -.001 to +.002) [GUESS]; LB +.000 to +.001; at best roughly a second 288 seed at ~26% more GPU per fold and ~24% more serving cost | 7.3 h probe + ~29 h folds 1-4 [MODEL]; a T4 commit microbench of a 320 member (untested); a K-builder for a 288 + 320 kit | `~/novel/recipe/auto_t320.sh` with its gate patched to G320' | **G320':** best raw >= .8909 OR last-3 raw rank-mean >= .8900, AND gold58 at the best raw epoch >= .905. FAIL -> 288 seed 1045 (sec 3) | probe gate ~22:30 10-08 or ~14:00 10-09; set ~10-10 to 10-11 |
| 7 | **Within-site AUC** (same-language pairs) as a diagnostic print in a `~/novel` copy of `score_run.py` | 0 today; flags a member that learns site habit | ~20 lines, CPU seconds | copy `decomp()` from `~/novel/site/s6_within_site_metric.py`; run on TTR288 folds 1-4 as they land | diagnostic only, NOT a gate (within-site SD .0020-.0022 > pooled .0016) | now |
| 8 | **Idle filler only:** TTR288 seed 1045 reseed (5 folds, 21 epochs) or an all-24-slot probe | reseed +.0004-.0006 SOLO / ~+.0002 blended [MEASURED analog at 224]; 24-slot unknown sign (-.001 to +.003 GUESS) | ~29 h; 24-slot x1.4 per fold | only when nothing above is runnable | 24-slot: >= the same resolution's f0 + .002 | n/a |

**Parked or closed (do not spend GPU or slots):**
- Full-data refit [GUESS ~0 to +.001]. A fold -1 member needs picker, build and k2_check changes, and there is no OOF
  to gate it on. Teammate quota is likely gone this week.
- Per-site bias head during training.
- LB probe of site offsets: outcome predictable from gold58, and it breaks the byte-identical worker.
- ConvNeXt-S/B: not cached offline; a download needs Eesh's OK; +.000 to +.003 [GUESS] at x1.66 cost.
- A 384 store: needs a re-prep plus a second hidden-set decode, and the acquisition matrix makes it ~0.
- Per-label 224/288 routing: per-label effects do not replicate across folds (Spearman -.15 to -.22).
- LW-1 per-label loss weights and RK-1 pairwise rank loss.
- Multi-scale TTA (-.0005 to -.0007 on top of A+B; gold74 confirms).
- EMA twins, epoch snapshots, rank/logit averaging, OOF-weighted folds.

### 2.1 Notes behind the ranking

**K8 (item 2).**
- Use **equal members (zero code)**, not the 0.7/0.3 set weighting.
  - The two differ by at most ~.0009 OOF (~.0005 blended) [MODEL]. Equal members wins if the expert-label D is ~0,
    which gold leans toward.
  - Equal members keeps the worker and cell byte-identical to K5/K7.
  - If 0.7/0.3 is ever wanted, the change is a manifest weight `w` plus `np.average`, mapped by worker-reported names,
    with a receipt check.
- **The K8 builder is a copy of `SP/k7`**, with these edits:
  - a no-set-B switch;
  - `K7_C_FOLDS` = 0-4;
  - keys `f{F}_s1044t288`;
  - a **per-fold** sanity bar for set C instead of the single .8876. That bar was fold 0's J1 + .004. Fold 3 is a hard
    split (224 runs .8763/.8766), so a single bar would wrongly drop it.
  - Test it with the K7 suite: collect, build, box A0, k2_check, and K5/K7 rebuild byte-identical.
- **Runtime fits [MEASURED + MODEL].** T4 microbench from the K7i commit: 288 member 481-532 ms, 224 member ~293-426 ms,
  both A+B. That gives K8 ~4.0-4.3 s/study per T4, ~47 GPU-min, margin >= ~1.1 h to the 8.6 h wall.
  [`SP/k7/build_i10/out_main_v1/fleet_receipt.json`]
- **Add, not replace.** On fold 1, 288 + both 224 seeds beats 288 alone by +.0016. On fold 0 it is -.0001.
  [MEASURED, epoch-16 snapshot, `vp2.txt`] K8 keeps set A for that reason, and because its pure-288 alternative cannot
  be resolved on the LB.

**frac75 (item 4).**
- The motivating label-file lift is optimistic. The Dread teacher was epoch-selected on gold58 (AUDIT 10-04 l.100), and
  the lift is not significant: gold58 +.046, CI [-.001, +.108]. mixSA moves toward the same teacher, so it is not
  independent support. [MEASURED: `vp.txt` sec 6, `verify_perlabel.md` sec 2.2]
- The probe student itself is a clean out-of-sample test: it never sees gold and is selected on val vs v4.
- Old bars (G2 >= .900, G3 >= .9042) were set in the NaN-as-negative convention. In the masked convention J1 already
  sits at .894 Fracture, and a null probe would pass G2 ~1/3 of the time. Hence the re-based bars. [MEASURED:
  `vp2.txt`]
- **Builder edit (prep before 10-09).** `build_k7.py` l.79 asserts `train_labels == labels_mix50_logit.csv`, and a
  frac75 s1043 set would clash with set A's keys. So K9 needs:
  - per-set train_labels;
  - a labels tag in the member key.

**VLM relabel (item 5).**
- Why it ranks above TTR320 despite the API cost: it is the only lever that targets the measured label shift.
- Caveats [MEASURED, directional]:
  - Our Synovitis head trails its own mix50 label on gold58 (-.030, SE .040).
  - J1 trails it on gold74 (-.075, CI excludes 0).
  - So the model gain will be smaller than the label gain.
- The Effusion veto in the current prereg is the wrong decider, but not for the forum track's reason. G1 compares two
  LABEL arms, and text labels read Effusion at only ~.877. The real risk is a false veto: P(veto) .14 at colum2131's
  claimed +.085. A sign check cuts that to ~.02. [MEASURED SEs + MODEL: `verify_forum.md` sec 6]

**TTR320 (item 6).**
- Downgraded by the verifier. The .33 mm pixel spacing is a reconstruction grid: the median acquisition is .50 x .63 mm
  (479 series), so 288 (.486 mm) already matches most of what scanners acquired. [MEASURED:
  `~/novel/verify_recipe/v2/v2_acq.out`]
- The original G320 bar (.8895) would let a run ~.0014 worse than 288 commit ~29 GPU h. Hence G320'.
- It must yield to label-side work.

---

## 3. V100 queue: TTR288 folds 1-4 (end ~10-08 10:00) to 10-15, then to 10-22

**Mechanism.**
- Arm **exactly one** queue runner before ~09:30 10-08 (main session or Eesh; this workflow armed nothing). A copy of
  `auto_t320.sh`'s guards will do:
  - wait while PAUSE_CHAIN exists, while any python holds the GPU, and while the `auto_after_ttr288.sh` bash is alive;
  - never kill;
  - skip an existing out dir;
  - one early-OOM fallback to grad_ckpt 1.
- Better: have it read the next job from a queue file only when the GPU is free, so jobs can be inserted (e.g. VLM
  folds) without killing anything.
- **Never arm two waiting chains:** both can co-launch when the GPU frees.
- New out dirs only; key members by (fold, seed, ttr_size, labels) to avoid the `f0_s1045` probe/reseed key clash.
- Times are MODEL:
  - 224: grad_ckpt 1 11.3 min/epoch (J1), grad_ckpt 0 10.5 min/epoch.
  - 288: 16.0-16.8 min/epoch.
  - 320 at bs 1: ~21 min/epoch.
  - ~+-10% for thermal throttle.

### 3.1 To 10-15

| # | job | start -> end (IST) | then |
|---|---|---|---|
| Q0 | (running) TTR288 mix50 s1044 folds 1-4 | now -> ~09:40-10:10 10-08 | K8 collect/build (sec 4) |
| Q1 | **frac75 f0 probe** (224, s1043, J1 argv, grad_ckpt 1, 24 ep) | ~10:10 -> ~14:45 10-08 | CPU gate (G1-G3, gold_infer5, ~30 min) |
| Q2 | if frac75 PASS: **frac75 folds 1-4** (224, s1043, 21 ep, grad_ckpt 0) | ~15:20 10-08 -> ~06:00 10-09 | K9 build (sec 4) |
| Q3 | **TTR320 f0 probe** (s1045, mix50, 21 ep, G320') | ~22:30 10-08 if frac75 FAILED (started ~15:20), else ~06:00 -> ~13:20 10-09 | G320' gate |
| Q4 | G320' PASS: **TTR320 folds 1-4** (mix50, s1045). G320' FAIL: **TTR288 s1045 folds 0-4** (21 ep) using frac75 labels if Q1 passed, else mix50. The FAIL branch is then a reseed that also carries the label lever. | ~29 h, ends ~10-10 evening to ~10-11 morning | kit choice (sec 5) |
| Q5 | if Eesh approved the VLM spend and the pilot passed S: **relabeled TTR224 f0 probe** (~4.5 h), then folds 1-4 (~15 h), **inserted ahead of any remaining Q4 folds** (label-side first) | ~10-11 -> ~10-12/13 | K10 read ~10-13 |
| Q6 | otherwise: the filler of item 8 (288 reseed or 24-slot probe) | to 10-15 | |

### 3.2 10-15 to 10-22

- **10-15 -> ~10-20 12:00 IST:**
  - If merged: the V100 trains the partner's family. A compute-bound partner such as Thimsen is exactly where our V100
    adds value. Sharing within the team is allowed only after the merge.
  - If not merged: a second seed, or the 288 version, of whichever label set passed (frac75 or VLM). ~29-34 h for
    5 folds at 288.
- **Training freeze ~10-20 12:00 IST.** After that, nothing trained can ship: the kit build, A0, dataset version,
  commits and the ~6 h scoring all have to land before 10-22 23:59 UTC (10-23 05:29 IST). The GPU may idle after the
  freeze; that is not a loss.
- **Inference cap [MODEL].** The final kit holds at most **two big sets**: 5x288 + 5x224 (~4.0-4.3 s/study/T4),
  5x288 + 5x288 (~5 s), or 5x288 + 5x320 (~5.6-6.2 s, margin ~0.8-1.1 h). Three sets including 320 is too tight. So
  later sets must **replace**, not add.

---

## 4. Submission plan 10-07 to 10-10

Rules:
- Slot days are UTC, i.e. 05:30 IST to 05:29 IST next day.
- Submit `submission.csv` only, after box A0 BOXSTAGE PASS and K2CHECK PASS.
- No interim 288 subsets, no W / rank / prob variants, no per-label weights from gold58 or the LB.

Weekly Kaggle GPU quota (commits only):
- It was 10h22m on 10-06, and K7i used 2 commits. GUESS: ~9 h left, at ~0.5 h per commit.
- GUESS: it resets Saturday 10-10 ~05:30 IST, inferred from the 09-30 note "~60 h to reset". Check the Kaggle settings
  page before each push.

| UTC day | slots | submit | why / decision rule | expected [MODEL] |
|---|---|---|---|---|
| **10-07** | 2 used (K7i 07:16), 3 left | **Nothing new planned.** See the K7i and K7-full notes below. | Laptop prep today (no slot): K8 builder + tests; frac75 builder edit; queue runner + G320' patch; pin the frac75 gold74 convention in its prereg; draft the relabel gate amendment for Eesh. At fold-1 close (~13:40) run `v_ens.py` V3 and `perlabel_update.py` on CPU. | K7i main .945-.947, SOLO .943-.944 |
| **10-08** | 5 | ~13:30-15:00: **K8 main** (STRICT W .50), **K8 SOLO**, **K8-KF** (same fleet, KF parent flat .60, W .50). Reads ~20:00-22:00. | K8 SOLO >= .943 = the 288 set transfers. K8-KF exists so a scored pick-2 candidate exists early; its main use is plumbing. Before pushing: CPU gold58/gold74 read of the full 5x288 set vs the A set (the verifier's deciding read). If 5x288 < A by >= .003 on gold58, the tie rule in sec 5 stops favouring K8. 2 slots reserved. | main .946-.947; SOLO .943-.945; KF ~ main -.001 |
| **10-09** | 5 | If frac75 passed G1-G3 and folds 1-4 ended ~06:00: **K9 main + K9 SOLO** = parent + [5x288 mix50 + 5x224 frac75], ~10:00-12:00, reads ~16:00-18:00. If frac75 failed: **nothing**. | The LB cannot resolve +.001. Keep K9 for the final kit only if K9 SOLO >= K8 SOLO - .002 (a sanity bound, not a gain test) AND the probe passed its gold gates. | +0 to +.001 vs K8 |
| **10-10** | 5 (quota reset GUESS) | **Nothing planned.** The 320 set ends ~10-10 evening at the earliest, and no interim subsets are allowed. | Eesh's decision deadline for the VLM spend, so a relabeled set can land by ~10-13. Keep slots unused rather than fishing. | n/a |

**K7i reads (~13:15), pre-registered:**
- Main .945-.947 = noise-equivalent to K5 w50 (expected; the second seed is worth ~+.0002 blended).
- Main <= .944 = unexpected but most likely noise, since A0 and K2CHECK passed. Check the commit's fleet receipt before
  trusting it.
- SOLO <= .941 = build problem: audit.

**K7 full (teammate fold 4) is NOT recommended as a submission.**
- It adds one same-recipe member at E 14-15 to K7i: ~+.0001 [MODEL], below display rounding.
- Set B is dropped from K8 onward, so it cannot become a final-pick fleet.
- If fold 4 lands and Eesh still wants the read, use 1 spare 10-07/10-08 slot (main only, 1 commit). Collect it with
  `K5_MIN_EPOCHS=14` (Eesh approved 10-06).

---

## 5. Final-pick strategy (by 10-22 23:59 UTC)

- **Pick 1 ("best"):** the highest public read among the candidates.
  - Candidates: K5 w50 (.946), K7i main, K8 main, K9/K10 main if built, a merged blend if any. All on the parent's
    cell-28 map at W .50.
  - **Pre-registered tie rule:** public differences < .002 are noise (paired SE ~.001-.0015 + rounding). So pick the
    newest strongest-evidence fleet (K8, or K9 if its gates passed) when it reads **>= best other - .001**.
  - "Strongest evidence" = 5-fold pooled OOF plus no negative gold58 sign on the changed labels. Otherwise take the
    best public read.
- **Pick 2 ("shake-up hedge"):** the same final fleet on the **KF parent** (`K2_CFG['OUTER_FLAT'] = 0.60`). This
  removes the parent's single public-LB-tuned per-label map (LatMen 1.00 etc.).
  - Use W .50. W .60 only if that fleet's SOLO reads >= .945. [MODEL]
  - Expected vs pick 1: ~-.001 public, -.001 to +.001 private (GUESS). Protects at most ~.001.
    [`SP/gold_prep/parent_audit.md`]
  - **If a merge or a decorrelated strong set lands before 10-20,** pick 2 becomes the most different strong blend +
    KF. That is a better hedge than KF on a copy of pick 1 (per-label Spearman vs pick 1 ~.998).
  - k2_check cannot tell KF from the map on 3-study commits. So before submitting any KF kernel, assert
    `outer_flat=0.6` in the commit's `_k2_stage_fleet.json` and the `rec['outer_flat']` receipt.
- **Both picks must be scored submissions.** Kaggle only lets you select scored ones. Submit the KF twin of the final
  fleet by ~10-20, with K8-KF (10-08) as the fallback.
- **Never** put site, language, metadata, stacking, gold58-fitted or LB-fitted per-label weights in either pick.
- **Why a hedge at all [MEASURED + CLAIMED]:**
  - Public is ~390 studies, private ~910. The .945-.950 band is a coin toss on private.
  - Metadata-only site memorisation exists in this data (733517: .652 random folds vs .598 scanner-grouped, CLAIMED,
    re-verified).
  - Medal-line teams flip ~24% of the time (742327, CLAIMED).

---

## 6. Settled today (add to HANDOFF "Settled"; do not re-run)

| item | verdict | evidence |
|---|---|---|
| Site/language/scanner offsets, metadata stacking | **trap**: OOF +.0063 vs reports, gold58 -.0152 | MEASURED `~/novel/site/s5_gold_grid.log`, `~/novel/verify_site/v2_gold.log` |
| Site is inferable at test time (95.3% from store + series features) | true, and irrelevant: do not use | MEASURED `~/novel/verify_site/v4_obs.log` |
| Label co-occurrence stacker | -.00096 OOF, -.001 gold | MEASURED `~/novel/stack_oof.log`, `vf2.log` |
| Multi-scale TTA (256/288/320 passes) | -.0005 to -.0007 on top of A+B | MEASURED `~/novel/ens/out/ms_report.txt` |
| Averaging: prob mean, raw best-val checkpoint, equal fold weights | already optimal | MEASURED `~/novel/ens/out/oof_report.txt`, `v_ens.txt` V1 |
| 21 epochs = 24 epochs (at 224) | use 21 for new non-paired runs (saves 12.5%) | MEASURED; untested at 288/320 |
| Second seed of the same recipe | +.0004-.0006 SOLO, ~+.0002 blended | MEASURED gold74 k-curve |
| LW-1 per-label loss weights / RK-1 rank loss / per-label 224-288 routing | ~0 or negative | MODEL + MEASURED (`perlabel.md`, `vp2.txt`) |
| "Focal-finding" story for 288 | not robust (Effusion strongest; all 12 labels up selection-free) | MEASURED `~/novel/verify_recipe/v2/v2_oof.out` |
| 384 px / ConvNeXt-S/B | skip (acquisition-limited; not cached) | MEASURED `v2_acq.out`, hf cache listing |

---

## 7. Decisions only Eesh can make

1. **Merge outreach** before 10-13 (Thimsen, Nikita, Musa Tur Farazi).
2. **API spend for the VLM relabel:** pilot ~$4-8, scale-up ~$15-50. Decide by ~10-10. Approve the gate amendment
   (G1 -> sign check, S decides) before any live call.
3. **Teammate quota next week** (after the GUESS 10-10 reset): one 5-fold 224 set of whichever label variant passes
   (frac75 or VLM) would parallelise the V100. ~8 h per fold per T4 at 21 epochs [MODEL]; 288 is only feasible at
   E ~16 [GUESS].
4. **K7 full:** default skip (sec 4).

---

## 8. Sources

- Track reports: `SP/novel/site/site.md`, `SP/novel/recipe.md`, `SP/novel/perlabel.md`, `SP/novel/ensemble.md`,
  `SP/novel/forum.md`.
- Verifier reports: `SP/novel/verify_site.md`, `SP/novel/verify_recipe.md`, `SP/novel/verify_perlabel.md`,
  `SP/novel/verify_ensemble.md`, `SP/novel/verify_forum.md`.
- Box scripts and outputs: `~/novel/{site,recipe,ens,verify_*}/`, `~/novel/out/`, `~/novel/stack_oof.*`,
  `~/novel/cooc_diag.*`. Local copies in `SP/novel/box_out/` and the `verify_*` folders.
- Kits: `SP/k7` (`K7_RUNBOOK.md`), `SP/gold_prep/parent_audit.md` (KF), `~/gold_prep/relabel/` (pilot kit, prereg).
- Forum threads: 733517, 735304, 737566, 741310, 742327, 743374, 745861, 745949, 746060. All read signed out; nothing
  posted, voted or downloaded.
