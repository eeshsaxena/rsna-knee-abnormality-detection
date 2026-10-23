# Remaining no-training levers in the HW kernel (lens: remaining levers), 10-17 ~22:30 IST

Read-only audit. Nothing built, pushed, submitted or committed. Public notebooks read from the copies another lens already pulled
(`research_1hr/pulled/notebooks/watchdir/pulled/`, untrusted, read with `python -I`, never executed). R7 sources extracted for
reading into the scratchpad only. Tags: MEASURED (code / file / log read here), CLAIMED (author or forum text), MODEL (inference
from measured numbers), GUESS.

Audited kernel: `session_2026-10-16/hw/kernel_v224r6hws6fj/rsna_sub_v224r6hws6fj.ipynb` (HW 57262115, 19 cells).

## Bottom line
- **No untested lever with a measured mechanism and positive expected value is left in this kernel.** The four candidates in the
  brief are each closed by code (S6fall flip), already measured (r224 mirror), or already used correctly (heliosli table).
- **The one live option is a removal: "HW without S6fall" (cells 16-18 dropped).** It is worth a slot **only if s615 (S6 .15)
  reads POSITIVE** tonight. It is a weights-family entry, so it needs Eesh's go and a PREREG amendment first.
- Everything else (goodpjw reader at 1/10, heliosli's V2 table, S6fall side model, Raptor multi-crop) has expected gain of about
  0 at three decimals, or carries private-overfit risk. I don't recommend any of them.

| # | lever | status | expected gain | runtime | do it? |
|---|---|---|---|---|---|
| 1 | flip TTA for the S6fall stage | not available (code + training) | <= 0 (MODEL) | +1.2 to 2.4 h if forced | NO |
| 2 | Raptor r384 / r224 TTA | r384 already mirrored; r224 mirror measured negative | 0 (MEASURED read) | +~25 s | NO (closed) |
| 3 | heliosli weights table | used exactly as heliosli does; 2 cosmetic deviations, both 0 in expectation | 0 | 0 | NO |
| 3b | heliosli V2 table (.15/.65/.85/.30) | outside A11, public-tuned | unknown, private risk | 0 | NO (Eesh's call) |
| 4 | S6fall off (= heliosli blend-mine-in + our J decoders) | untested on the R6 base | -.0005 to +.0005 (MODEL) | saves ~1.2 to 2.4 h | ONLY IF s615 reads positive |
| 5 | goodpjw 2.5D reader at 1/10 | public, unused by us | ~0 (CLAIMED same display) | new leg, GUESS 20-40 min | NO |
| 6 | side model for S6fall (`--side-model`) | supported by 34f433fe, unused | ~+.0002 at most (GUESS) | ~0 | NO |

---

## 1. Flip TTA for the S6fall stage: NOT AVAILABLE
- **The worker has no flag.** S6fall runs compact_infer **34f433fe** (`s6f/src_extract/compact_infer.py.34f433fe`). Its argparse
  (lines 234-268) has no `--flip-tta`. Its only "flip" code is the side-correction re-render (lines 116-152, 266). MEASURED.
- **R7's worker would refuse the S6 checkpoint.** The flip-capable worker is v4c **8401812c** (R6 / R7 stage, cell 7 SOURCES).
  It exits when no fold model is an extent model: "`--flip-tta needs the plane column ... extent fold models only`" (line 714,
  N_EXTENT counted from `ext_u/ext_view/ext_gamma` keys, line 638). S6fall's run_config has `extent: False`
  (`silver/pulled/s6fall_meta/E_prior_s42_fall.run_config.json`). It also imports a different trainer: 04df7e2e (v9.8) instead of
  S6's 14767c7a. MEASURED.
- **The model never saw a mirrored knee.**
  - S6's trainer 14767c7a contains no mirror code at all (grep "mirror" = 0 hits).
  - Its `--aug` help text says v2 adds jitter with "no flips: laterality".
  - S6fall trained with `aug: crop`, the canonical-laterality render.
  - R7's flip works because R7 was trained with `--mirror 0.5` plus output permutation (trainer v9.4, lines 76-88 / 999-1050 of
    04df7e2e).
  - heliosli built a side classifier precisely because mirrored studies get mis-scored by these canonical models (v4c docstring /
    WORKER_ARGS comment). MEASURED code + CLAIMED comment.
  - Verdict: a forced mirror pass on S6 would be out of distribution. Expected <= 0 (MODEL). At blend weight .2, even a +.002
    S6-alone gain would be about +.0004 (MODEL).
- **Cost if forced anyway.** It needs a patched worker (new code path, new gate) and doubles S6F's GPU time: S6F wall 1.2 to 2.4 h
  at s = 1 (MODEL, `s6f/sim_s6f_runtime.txt` sec D). That pushes kernel p99 past the 8.2 h S6 hard deadline, so P(cut -> ACL
  canary) rises (MODEL). **Closed.**

## 2. Raptor TTA status: r384 mirrored; r224 mirror already measured negative
- **r384 (cell 4) already serves nartaa's anatomical mirror TTA.**
  - `_mirror_tri` reverses the 3 slice channels on sagittal slots and flips the width on coronal / axial, on uint8 before the LUT.
  - `infer_pair_probs` = 0.5 (plain + mirror).
  - No class permutation is needed: Raptor trains on un-canonicalised L and R knees, so a mirrored knee keeps its medial / lateral
    labels. MEASURED code.
  - Ablation: mirror +.001 (CLAIMED, nartaa / Danial thread 746792, FORUM_DIGEST_10-12).
- **r224 (cell 11) is plain only.** A diff vs cell 4 shows the mirror lines removed. MEASURED. We already tested the r224 mirror:
  **FLRM 57200578 = .956, non-move on the FLIP base, a valid negative under A3 R1** (PROGRESS_15-10 11:14, MEASURED). goodpjw's v25
  (224 mirror + reader) shows .958, the same display as heliosli's .958 without it (CLAIMED). It is in the closed list. **Closed.**
- **Other Raptor TTA.**
  - All 94 window centres are already used, so there is no stride or offset lever. MEASURED.
  - Serve-crop variants lose: 304 -.001, 288 -.002 (CLAIMED, nartaa ablations).
  - No public positive evidence for multi-scale or multi-crop. **Not recommended.**
- **R6 window density.** R6 already serves eval_stride 1, every centre (`session_2026-10-15/r6v1/raw/manifest.json`), and S6fall
  too (`B_GEOMETRY eval_stride 1`). There is no density lever. MEASURED.

## 3. heliosli weights table: used correctly
Diffed cell 14 against heliosli's `blend-mine-in-private-lb-luck-to-all-upvoters` v1 (cell 8).
- **Identical (MEASURED code):**
  - `W_RAPTOR_C` = {Medial OA .5, Lateral OA .2, PF OA .6, Synovitis .8, Baker's .35}, else .5.
  - Formula `wr * rr + (1 - wr) * ro`.
  - r224 at A224 .35 inside the Raptor slot.
  - Same fail rules.
  - Same R6 bundle fe055406 with flip TTA and its fluid + state stackers (WORKER_ARGS adds both from the manifest, which lists
    `fluid_stacker.json` / `state_stacker.json`).
- **Deviation (a), r224 rank source.**
  - We rank `raptor224_sub.csv`; heliosli ranks `raptor224_raw.csv`.
  - The 224 script's `rankpct` is `argsort().argsort()`, i.e. ordinal ranks with arbitrary tie order, not average ties (MEASURED,
    cell 11).
  - So tied r224 probabilities (fp16 logits make ties plausible) get arbitrary order in our r224 slot, where heliosli averages them.
  - Expected AUC effect 0 (tie order is independent of labels), tiny variance only (MODEL). Not worth an entry.
- **Deviation (b), r224 failure mask.**
  - Our r224 rank uses the r384 failure mask; heliosli uses r224's own.
  - Commit logs show 0 failures (`raptor_failed 0`, a25 commit log line 161, MEASURED), and the J decoders remove the known cause.
  - Effect 0 (MODEL).
- **S6fall sits after the table.** Final = .8 pct(V224-HW) + .2 pct(S6), so S6's effective weight is .2 on every label whatever the
  label's Raptor weight. MEASURED cell 18. Consistent with how S6F was read on the FLIP base.
- **3b, heliosli's V2 table.**
  - heliosli's newer `this-is-best-public-dataset-new-version-0-9586` v6 moves further: Lateral OA .15, PF OA .65, Synovitis .85,
    Baker's .30. It also adds goodpjw's reader (MEASURED, its cell 8 / 9).
  - It displays .958, the same as v1 (MEASURED watch flag). The ".9586" in the title is CLAIMED and cannot be checked: the API gives
    3 decimals only (MEASURED, `api_lb_1017T2150.json` displayScore).
  - goodpjw's extra weight moves beyond heliosli v1 lost: v6 .957, v7 .955 (CLAIMED, `does-s6-help-next-to-r6` v8 notes). Those
    were gold58-tuned moves, not V2.
  - V2 is public-tuned beyond what A11 allowed. **Not recommended**; Eesh's call if he wants it.
  - One-line change if ever wanted (cell 14): `W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.15, 'PF OA': 0.65, 'Synovitis': 0.85,
    "Baker's": 0.30}`.

## 4. The one live option: S6fall OFF on the R6 + heliosli base (conditional)
- **What.** HW minus the three appended S6F cells = heliosli's blend-mine-in v1 recipe plus our J decoders and the V224 r224
  canary. It is a removal, back to a published recipe; no new weight value.
- **Evidence:**
  - FOR S6 (MEASURED): S6fall .2 was a best-id move on the R7 / FLIP base (57201874, 10-15 13:54, PREREG_10-15 l.104).
  - AGAINST more S6 (MEASURED): hw3 (S6 .3 on HW) negative, 57291203.
  - AGAINST S6 next to R6 (CLAIMED):
    - goodpjw `does-s6-help-next-to-r6` v4 put "R6 flip + S6 0.3 + heliosli's weights" at **.957**, against .958 for the same recipe
      without S6. His words: "S6 next to R6 costs 0.001".
    - Caveat: his S6 is heliosli's S6 (`heliosli/rsna-knee-compact-v1`) mixed at .3 inside the compact slot. Ours is nartaa's
      all-data S6fall, mixed at .2 after the blend. Different model, different position, so this is only weak transfer.
  - UNMEASURED by us: R6 + heliosli weights with no S6. Every R6-base entry so far carries S6 .2 or more.
- **Decision rule (fixed before the read):**
  - If **s615 (S6 .15, 57313454) moves the best-id** tonight, lower S6 beats .2 on this base. "S6 off" is then the natural next
    one-change entry.
  - If s615 is negative or unread, S6 .2 is the local optimum of {.15, .2, .3}. Don't build it.
- **Expected gain (MODEL).** -.0005 to +.0005; P(best-id move | s615 positive) about .5, about .3 otherwise.
- **Runtime.** Saves the whole S6F stage: 1.2 to 2.4 h at s = 1 (MODEL, sim sec D), and removes the S6 hard-deadline / ACL-canary
  path. Kernel end about 4.5 h median (MODEL: the sim's base-end column).
- **Risk.** Low on mechanics: cells 0-15 are the already-gated V224-FLIP-R6-HW-J path, and cell 15's r224 canary stays. Private
  risk: it is another read on the S6 weight line (winner's curse, about 390 public studies).
- **Exact change.**
  - Drop cells 16, 17 and 18 of the HW notebook. `submission.csv` written by cells 14-15 is then final.
  - In `kernel-metadata.json`, remove `nartaa/rsna-knee-s6fall-weights`. Keep `glorysing/rsna-knee-offline-codecs-r09`, because
    cell 2's J decoders need it.
  - Do NOT use the one-liner `W_S6F = 0.0` in cell 18. It still runs S6F for 1-2 h, and any S6F skip or fail would fire the ACL
    canary and spoil the read.
- **Picks note.** Under A15 this entry would be on the S6 weight line, not the Raptor / R6 weights question. If it reads positive
  it becomes A. B stays per A15.

## 5. Published but unused: goodpjw 2.5D ConvNeXt reader at 1/10
- **What it is (CLAIMED).**
  - Dataset `goodpjw2008/rsna-knee-2-5d-convnext-reader`, Apache 2.0, 3 folds.
  - .929 alone, rank-blended at the end with per-finding weights .003 to .035 (Baker's .035, Contusion .03, ACL / Medial OA .025,
    the rest .02 or less).
  - heliosli's ".9586" v6 includes it.
- **Gain (CLAIMED + MODEL).** goodpjw v25 (with the reader) and heliosli v1 (without) both display .958, so the reader adds about 0
  at three decimals.
- **Cost.** A new leg with its own pip `--target` env (pylibjpeg / gdcm / timm wheels from his dataset), new code and a new gate.
  Runtime GUESS 20-40 min.
- **Not recommended.**

## 6. Partly used mechanism: side model for S6fall
- **What exists.** compact_infer 34f433fe supports `--side-model` (line 265, MEASURED). Our S6F driver does not pass it (b_cmd,
  MEASURED). R6's bundle ships `side_cls.pt`.
- **What it would do.** Re-render untagged, confidently mirrored studies to the canonical side before S6 scores them.
- **Gain.** Only a small share of studies is affected, and S6 is at .2. GUESS <= +.0002.
- **Risk.** Unknown whether nartaa's training cache used the side fix, so it could mismatch train vs test. It would also change S6F
  parity vs nartaa.
- **Not recommended.**

## Facts worth keeping
- S6fall = one checkpoint (E_prior_s42_fall.pt). nartaa's solo kernel runs `W_A, W_B = 0.0, 1.0`, so component A (the r384 model)
  has weight 0. We use everything that matters in it. MEASURED (`s6f/pulled_cur/rsna-knee-s6fall-solo.py` l.47, README).
- Measured r224 stage time on commit: 23-28 s for 3 studies. Hidden Raptor stage about 25 min (CLAIMED, nartaa 26.1 min incl.
  queue).
- An already-scored entry carries no further runtime risk. Public and private come from the same single hidden run, so HW's .958
  already proves its S6F ran clean. CLAIMED (Kaggle code-competition rules) + MODEL.
