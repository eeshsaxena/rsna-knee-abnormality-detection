# Verify: blendmath track (jump2, 2026-10-05 ~09:30 IST)

Adversarial re-check of `SP/jump2/blendmath.md` (sections 0-10). Tags: **MEASURED** (source) / **MODEL** / **GUESS**.
`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`.
New scripts and outputs: `SP/jump2/vbm/` (`vbm_lb.py/.out`, `vbm_map.py/.out`, `vbm_misc.py/.out` on the laptop, light CPU;
`vbm_formcheck.py/.log/.json` run on the box under `~/jump2/vbm/` with `CUDA_VISIBLE_DEVICES=`, `nice -n 19`, 2 threads).
Nothing was written on the box outside `~/jump2/`. No process was signalled. Kaggle was not touched.

---

## 0. Verdict

The math reproduces. Every inversion, the fleet table, the K4 map and the gold74 calibration give the same numbers
when I re-run them. I also tested the model's functional form against real rank blends, and it holds (sec 2). The
weak points are elsewhere:

1. **Box rule violation that hurts this track's timeline (MEASURED, not caused by blendmath).** Another jump2 job,
   `tta_bench.py` (pid 322072, started ~09:10, parent shell `cd ~/bigjump_build/code && nohup nice -n 5 ...`), runs
   **on the GPU** (2,394 MiB, `dev cuda`, no `CUDA_VISIBLE_DEVICES` in its environment) at **nice 5**. Both break the
   rules (GPU forbidden, nice 19 required). It slowed fold 1 epoch 3 from 12.6-12.7 min to **19.3 min**
   (`~/fleet_runs/f1_s1043_ttr_mix50.log`). Recent iterations ran ~0.73 s/it against 0.36 alone. Its own cap is
   `--max_minutes 40`. I did not touch it (no-signal rule). **The orchestrator or Eesh must stop new GPU side-jobs.**
   Also, `~/auto_folds.sh` waits for *any* python GPU app before it launches the next fold, so a side-job that is still
   running at a fold boundary stalls the chain.
2. **The chain ETAs are ~1 h optimistic (REFUTED as written).** J1 took **5 h 02 min** alone (MEASURED: launched 23:12,
   last write 04:13:46). The note uses ~12 min/epoch; the real figure is ~12.5 min plus a ~14 min first epoch.
   Corrected (MODEL): fold 1 ~13:30 with no contention, **~13:45** with today's tta_bench loss; fold 2 ~18:50;
   fold 3 ~23:50; **fold 4 ~04:50 on 10-06** (the note said 03:45), assuming no more GPU side-jobs and no PAUSE_CHAIN.
3. **One piece of sec 10.1 evidence is void (REFUTED).** `pilkwang/rsna-knee-llm-labels` is also in the
   `dataset_sources` of **s74-c**, the no-fleet parent (MEASURED `SP/intel/aastikrajan15_knee-s74-c/kernel-metadata.json`).
   It is a parent dependency, so it says nothing about the labels of aastik's fleet. The conclusion of sec 3 does not
   depend on it, because the solo >= .935 result comes from the two reads alone.
4. **"At W .20 the read is nearly flat in r" is overstated (REFUTED in degree).** At fleet solo .936, the W .20 read
   moves from .9452 at r .88 to .9428 at r .96 (MODEL `vbm_map.out`). That is about 60% of the W .50 sensitivity.
   So a W .20 read plus SOLO already carries most of the r information, and the extra W .50 slot buys little.
5. **The strength of the K4 member is the main downside risk (MODEL).** The note guesses TTR mix50 member .9325 ± .003
   from the F1 mix50 OOF-to-LB ratio (1.263, gives .934). The TTR v4 ratio is 1.232, and it gives **.929**
   (MEASURED ratios / MODEL mapping, `vbm_lb.out`). If the member is .929, the 5-fold fleet is ~.9326 and the W .20
   blend reads true **.9427**, below the parent (MODEL `vbm_misc.out`). So E[display] .9434 and P(.944) ~.40 lean
   optimistic. A fairer range is E[display] .9431-.9434 (MODEL).

Everything else is confirmed, either as the model computes it or as a measurement. The model-level conclusions
("every buildable fleet caps at ~+.0005", "we need ~10 members at .930 / r <= .86") stay MODEL with GUESS inputs.
They are internally consistent, but the LB r of every member other than F1 mix50 is inferred from an in-sample proxy
and a single calibration point, and nothing measured backs it.

---

## 1. Re-runs (all reproduce)

| quantity | blendmath | my re-run | tag |
|---|---|---|---|
| K2 r_LB(F1 mix50), round display | .895-.945 | .895-.945 | MODEL (`vbm_lb.out`) |
| same, if Kaggle truncates instead of rounding | not tested | .894-.945 (no change) | MODEL |
| aastik joint fit at solo .927 / .930 / .9325 | none | none / none / none (both display models) | MODEL |
| aastik joint fit at solo .935 / .9375 / .940 | .858-.873 / .878-.905 / .903-.937 | .858-.873 / .878-.905 / .903-.937 | MODEL |
| solo .927, r .80: W .5 -> W .6 | .9451 -> .9432 | .9451 -> .9432 | MODEL |
| TTR v4 break-even r at W .25 | .871 | .871 | MODEL |
| k-member table (.932, r .922, r_mm .955), k 5/10/15 | .9355/.939, .9360/.941, .9361/.942 | same | MODEL |
| W .50 pre-registered map at solo .936 | .944 -> r .885-.926, .943 -> .906-.948, .942 -> .927-.969 | identical | MODEL (`vbm_map.out`) |
| r_mm insensitivity of the W .20 pick (r_mm .94-.975, k 5-10) | true .94338-.94349 | .94338-.94351 | MODEL (`vbm_misc.out`) |
| gold74 map | r ~ .924 + .663 (sp - .905), corr .91, resid .008 | .924 + .663, corr .906, resid sd .0077 | MEASURED inputs, MODEL fit |
| fleetsim table / K4 EVOI | sec 6-7 tables | match `fleetsim_psd.out` (accepted N = 1,152 of 200k, so P(.944) has MC sd ~.014) | MODEL |
| OOF member / pair / label numbers | sec 4-5 | match `oofx.log`, `labx.log` byte for byte | MEASURED |

Display rounding vs truncation does not matter: all reads shift together, so the inversions move by <= .001 (MODEL).

## 2. Is the macro single-d binormal valid for rank blends? (new test, MEASURED on the box)

`~/jump2/vbm/vbm_formcheck.py`: 11 OOF pairs (867 fold-0 val studies, vs steven_v4). Each is a stronger "P" and a weaker
"F", including the parent's own Raptor/native384 arms against our members. I measured macro AUC of
`(1-W) rank(P) + W rank(F)` at W 0..1, fitted ONE macro r at W .3 (exactly how K2 is inverted), then predicted
W .5 and .6.

- Prediction error at W .5 / .6: **+.0000 to +.0009** (median ~+.0002). The measured drop from W .5 to W .6 is
  reproduced to within **.0002** in all 11 pairs (MEASURED `vbm_formcheck.log`). Per-label AUC spread in these pairs
  is sd .033-.043 (MEASURED), so real per-label heterogeneity does not break the macro model.
- The small error is one-signed: the model slightly **over**-predicts high-W blends for low-r pairs (raptor/TTRv4
  +.0006/+.0008). Real curves fall a little faster than modelled. For aastik that pushes the required fleet solo
  **up**, not down, so sec 3's conclusion gets stronger.
- Caveat: r_eff depends on which W it is fitted at. Raptor-proxy r for TTR v4 is .843-.851 at W .3 against .868 for the
  W .1-.5 fit in `oofx.py`; for F1m it is .898-.903 against .911 (MEASURED). The common offset absorbs the level, but
  the TTRv4-vs-F1m gap is .043-.055 depending on the convention. The "TTR v4 r_LB ~.88" point could equally be ~.87.

## 3. Claim-by-claim

**3.1 K2 (confirmed, with a caveat on wording).** The r interval and the "best lift <= +.0011" are correct (MODEL). But
"K2 = .943 was the expected outcome" is circular, because that r interval comes from K2 itself. Before K2, the prior
centre r ~.95 predicted a .942 read (MODEL: solo .932 at r .95 gives W .30 .9418), and NEXT_SESSION's prior was BLEND
median .940. Given the solo .932, the W .30 read spans .9414-.9477 for r .96-.80 (MODEL), so .943 carried real information.

**3.2 K3 (confirmed as a model statement; uncertain on the LB).** K3 alone gives .805-.975 (reproduced). The ".88" comes
entirely from the OOF r_eff against an **in-sample** Raptor proxy (the public Raptor arms run over train studies; their
OOF AUC vs v4 is .908 against our members' .869-.884, MEASURED). One calibration point (F1 mix50) sets the offset. mix50
labels contain the widedense Dread teacher (`build_mixSA.py`, default teacher `teacher_out/teacher_train_probs.csv`,
MEASURED), whose train-set predictions correlate .937 with that proxy (MEASURED `labx.log`). So part of the mix50-vs-v4
r gap on train studies may be in-sample echo that does not carry to the test set (GUESS on size).

**3.3 aastik (confirmed, one sub-claim refuted).**
- Confirmed (MEASURED): `aastik_ours_script.py` equals the base64 payload in s75-w50 (CRLF-normalised); the w50 and w60
  payload cells are identical; the two notebooks differ only in `OUR_WEIGHT = 0.5` vs `0.6`; s75 = the 23 s74-c cells
  unchanged + 2 appended cells (fleet stage + global rank blend). `ONLY` = 10 `pt_v5` checkpoints;
  `TARGET_WEIGHTS = None`; `KNEE_LEAN=1` is injected (no offset TTA, no mirror); lines 1061-1062 say "the same 6 models
  scored LB 0.927 raw". So the .927 was an older set. Correct.
- Confirmed (MODEL, and stronger after sec 2): no r fits both .945 reads below solo ~.934.
- Refuted: the `pilkwang/rsna-knee-llm-labels` source as evidence about his fleet labels (see 0.3).
- New caveats, unresolved: (a) the .943/.945/.945 values are Kaggle `bestPublicScore`, the max over kernel versions
  (MEASURED `SP/intel/kern_search*.jsonl`). The create and update stamps are 20 min apart (s75) and 3 s apart (s74-c),
  which suggests single versions (GUESS). (b) His third runtime valve can drop models (`N_KEEP`) if the projected time
  passes 7.5 h, so the W .5 and W .6 hidden runs could in principle have used different numbers of models. That is
  unlikely in lean mode (GUESS).

**3.4 Labels drive parent-likeness (confirmed as an OOF measurement; LB transfer uncertain).** Same seed-42 F1 runs:
mix50 .911, gated .887, v4 .857; TTR adds .005-.011; F1m+TTRm .924 (MEASURED `oofx.log`). The binormal ensemble formula
predicts .926 for that last value. All of it is measured against the in-sample proxy (see 3.2).

**3.5 r_mm .900-.948 and raw/EMA .985-.995 (confirmed, MEASURED).**

**3.6 gold74 instrument (confirmed computation; precision overstated).** gold74 really is held out of every fleet run.
Each fold-0 log shows `held-out (fold<0 or --exclude) 74`, and the fold-1 command line carries
`--exclude gold74_eval_set.csv` (MEASURED). The fit reproduces. But a same-recipe J1/fold-1 pair will sit at
sp_g74 ~.95-.97, outside the calibration range (.879-.939). Leave-one-out slopes run .53-.82, so at sp .97 the map gives
r .958-.977. At sp = 1 the map gives r .987, not 1, so the line cannot hold near the top (MODEL `vbm_lb.out`). Use
±.015, not ±.008. Clean on leakage: it compares predictions only and uses no labels.

**3.7 r_mm does not change K4 (confirmed, MODEL).** True W .20 blend .94338-.94351 over r_mm .94-.975, k 5-10.

**3.8 "Every buildable fleet caps at +.0003-.0005" (uncertain).** The table reproduces, and the A/B/C rows differ by less
than .0002 in E[true], below MC noise and display rounding (MODEL). Every non-K2 r and every second-family or
self-teacher input is GUESS. The forum evidence on self-teachers is mixed to negative: Nicolai Karcher says 50/50
soft bootstrapping "doesn't translate to LB at all" (MEASURED `SP/jump2/verify/t735304_flat.txt` line 32), and HANDOFF
records "OOF self-distillation -0.001" (Tian). So C6/C7 (own-OOF silent-fill members at .927 ± .004) are optimistic
(GUESS).

**3.9 "What a .945 needs" (confirmed as MODEL arithmetic).** H1 reproduces (true .9451). Whether such members exist is GUESS.

**3.10 Forum citations used by sec 3 (partly confirmed).**
- Raymond Yuen: confirmed. .938 LB "full train CoAtNet 288", then updates .94 and .949, "still label", built from weak
  labels -> teachers -> pseudo-labels (thread 735304, MEASURED). The .949 update does not say single model, and his
  labels came from paid APIs (under $5), which we cannot use (no keys).
- Yann Majewski: confirmed, but the context matters. The +.015 came from "my labels with some public dataset" while he
  was still below his later .936 single-fold score. A +.015 at ~.92 does not transfer to our .93 regime (GUESS).
- Archit Konde: confirmed. Single fold, CoAtNet 224, .950, "labels were the bottleneck", and his own OOF single model
  scores .930 on gold58 against our .914 (MEASURED thread). His fix was to mix his OWN model's predictions into the
  labels, which is not a parent teacher.

**3.11 Chain timing (refuted, see 0.2).** The note's claim that gold_infer (CPU, nice 19, 12 threads) runs harmlessly
alongside is **correct**. Fold 1 epochs 1-2 took 12.6/12.7 min, the same as J1's 12.3-12.6 (MEASURED). The slowdown
comes from the GPU side-job only.

## 4. Feasibility and rules

- Box LLM tooling (MEASURED): `rsna` env has transformers/accelerate/bitsandbytes, **no vllm, no llama_cpp**; the only
  local LLM is `Qwen/Qwen2.5-7B-Instruct` (15 GB HF cache). The box has internet; Kaggle reruns do not. The existing
  `pseudo_labels_qwen7b.csv` copies the 58 gold rows verbatim (gold58 = 1.000 in `labx.log`), so its quality has
  **never been measured**.
- So the note's top lever, "a pure report-label table at .91-.92 gold58", has no measured route. steven_v4 (a public
  multi-LLM blend) is .893; a forum local-LLM extraction (Scott Willis, Gemma) was "around .89" (MEASURED thread).
  The GPU is booked by the fold chain until ~04:50 10-06. Prompt iteration scored on 58 studies would also overfit
  gold58: macro-AUC sd ~.01-.015 at n = 58 (GUESS), the same size as the hoped-for gain.
- A K4 kit with parent + 5 TTR members needs ~2.5 F1-equivalents against a documented serving cap of ~8 (GUESS in
  NEXT_SESSION sec 3). It is not measured, but the margin is comfortable.
- Slots: 5/day shared with SwaplY (NEXT_SESSION). Three slots on one fleet is a large share for EVOI +.00016 (MODEL).

## 5. Corrected recommendations

| # | action | keep? | expected public gain | why |
|---|---|---|---|---|
| 1 | K4 score slot: parent + 5-fold TTR mix50 at W .20 | yes (re-timed) | E[display] .9431-.9434, true +.0000 to +.0004 (MODEL) | Model, numbers and r_mm insensitivity all reproduce. Ready ~05:00 10-06, not 04:00. The downside is a weaker member (.929 -> true .9427). W .15-.30 are within .0002 of each other. |
| 2 | SOLO of the 5-fold TTR mix50 fleet | yes | none direct; pins fleet solo ±.0005 and fixes the downside question in item 1 | Every later W and r read depends on it. |
| 3 | Extra W .50 information slot | downgrade to optional | ~0; W .50 E[display] .9423 (MODEL) | The W .20 read is ~60% as r-sensitive as W .50, so SOLO + W .20 already pins r to roughly ±.04. Spend the slot only if a slot would otherwise go unused. |
| 4 | gold_infer on the fold-1 raw best ckpt, then g74pairs, after ~13:45 | yes (CPU only, nice 19, threads <= 12) | none; replaces the r_mm GUESS | Cheap and harmless to training (MEASURED). Read the result with ±.015, not ±.008. It does not change K4. |
| 5 | Stop GPU side-jobs on the box (tta_bench-style) during the chain | **new, yes** | protects the ~04:50 10-06 K4 readiness | Rule violation (GPU, nice 5). It cost ~6.6 min on one epoch, and it can stall auto_folds at a fold boundary. Do not kill it (rule); tell the orchestrator/Eesh. |
| 6 | After the chain: TTR v4 folds 1-4 instead of more mix50 seeds | keep, low confidence | +.0000 to +.0005 true (MODEL; A/B/C rows within .0002 of each other) | Direction is fine. Self-teacher payoff evidence is negative (Nicolai null, Tian -.001), so do not count on C6/C7. |
| 7 | Pure report-label table at .91-.92 gold58 as top priority | **no, replace with a cheap probe** | unknown | No measured route (local Qwen2.5-7B via transformers only, GPU booked, no API keys), and gold58 selection risk. First, run the existing qwen7b extractor on just the 58 gold reports when the GPU is free (minutes), to see whether a local LLM is anywhere near .893. |
| 8 | Merge as the larger lever | yes | the only modelled path to >= .946 (MODEL/GUESS) | Unchanged. Note that `merge_outreach.md` says pj and aastik gain nothing from our d4, so partners must be other teams. Eesh posts requests himself. |

## 6. Files

- This note: `SP/jump2/verify_blendmath.md`
- Re-runs: `SP/jump2/vbm/vbm_lb.py` + `.out`, `vbm_map.py` + `.out`, `vbm_misc.py` + `.out` (laptop);
  `SP/jump2/vbm/vbm_formcheck.py` + `.log` + `.json` (box copy `~/jump2/vbm/`).
