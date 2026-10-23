# RSNA Knee: NEXT SESSION (written 2026-10-04 18:30 IST)

`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`
(Temp: copy anything worth keeping into `Downloads/rsna-knee/`). **All 10-04 SP docs/scripts + box results are saved in `Downloads/rsna-knee/session_2026-10-04/` (`SP/<x>` -> `session_2026-10-04/<x>`, box files under `box/`; index in its README.md).** `B` = `ssh -o BatchMode=yes training@192.168.10.30`.
Kaggle CLI = `C:/Users/eeshs/AppData/Local/Programs/Python/Python312/Scripts/kaggle.exe` (prefix `PYTHONIOENCODING=utf-8`).
Deadlines: merge **10-15 23:59 UTC**, final **10-22**. 5 slots/day shared with SwaplY, reset 05:30 IST. LB reads take ~6 h.

## 0. Where we are, in one paragraph

Best public is still **0.943**, tied with about 880 teams. The crop leg is **dead**: crop15 .942, crop25 .941,
mixsacrop25 .940, K1 3-member K=8 ensemble .942. K2 was submitted 10-04 11:50 IST and both reads were still pending
at 18:19:
- ref 56816035 = **K2 SOLO**, our ConvNeXt fleet member alone;
- ref 56816039 = **K2 BLEND**, parent + fleet at W .30.

Today's key discovery: **gold58 cannot judge the parent or any blend into it.**
- The parent's DINOv2 arms score .96-.996 on gold58, and the RadImageNet e10/e13 arms .97/.98. They were trained on those exact studies.
- Our fleet member (.914 out-of-sample) is about as good as one Dread Raptor arm (.916-.922 out-of-sample).
- Our own models alone cannot plausibly reach .95. The research says P(>= .950 from our own fleet) <= 5%, and the honest final range is .943-.947. **The merge is the >= .945 route.** Public check done: nothing above .943 is forkable (all use private datasets); DINOsaur V5 = .943. We are rank 419 (public bronze cutoff ~508).

## 00. NEXT CLAUDE: open HANDOFF.md first. Its top section NEXT CLAUDE: 4 TASKS TO DO is the work queue (Kaggle folds 3+4 launch now, fast-trainer switch ~13:50, K4 reads ~16:30, F2 probe ~18:00).

## 0a. LATEST (10-05 10:30 IST): the CURRENT STATE block at the top of HANDOFF.md supersedes everything below

Quick commands:
```bash
K=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/Scripts/kaggle.exe
PYTHONIOENCODING=utf-8 $K competitions submissions -c rsna-knee-abnormality-detection --csv | head -5
ssh -o BatchMode=yes training@192.168.10.30 'tail -n 3 ~/fleet_runs/auto_folds.log ~/fleet_runs/auto_folds_fast.log 2>/dev/null; ls ~/fleet_runs/PAUSE_CHAIN 2>/dev/null; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; for f in 1 2 3 4; do L=~/fleet_runs/f${f}_s1043_ttr_mix50.log; [ -f $L ] && echo "f$f: $(tail -n 1 $L | cut -c1-100)"; done' < /dev/null
PYTHONIOENCODING=utf-8 $K kernels status eeshsaxena/rsna-train-ttr-f34
```

## 0b. LATEST (10-04 23:30 IST): read these first

- **K2 reads:** SOLO .932, BLEND W .30 .943. The mix50 member is real but parent-like (r_LB .90-.94).
- **K3 submitted 23:29:**
  - 56830643 = TTR v4 SOLO;
  - 56830644 = STRICT parent + mean(mix50, TTR v4) at W .25.
  - Read both around 05:30 IST.
- **J1 is running alone on the box:** TTR fold-0 on mix50, seed 1043, `~/fleet_runs/f0_s1043_ttr_mix50`.
  - TTR v4's gate G-T2 came out MIDDLE (+.0072).
  - MIDDLE rule: J1 OOF >= .8861 -> TTR leads production, else F1 leads.
- **Audit** (`AUDIT_REPORT_2026-10-04.md`) says the next production members should be **TTR v4 + a 2nd model family on v4 labels** (EfficientNetV2-S / ResNet-50 in `~/fleet2_code`), not more mix50: correlation with the parent is the bottleneck.
- **External data:** `EXTERNAL_DATA_VERDICT.md`. Default = don't use.
- **K3 kit:** `SP/k3` (archived in `session_2026-10-04/k3/`) supports TTR members, STRICT mode and member filters. Reuse it for K4.

## 1. First 10 minutes (copy-paste)

```bash
K=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/Scripts/kaggle.exe
PYTHONIOENCODING=utf-8 $K competitions submissions -c rsna-knee-abnormality-detection --csv | head -4
ssh -o BatchMode=yes training@192.168.10.30 'cat ~/fleet_runs/auto_j1.log; cat ~/fleet_runs/gt2_final.txt 2>/dev/null | tail -n 4; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; for f in f0_s1042_ttr f0_s42_gated f0_s1043_ttr_mix50 all_s43_mix50; do [ -f ~/fleet_runs/$f.log ] && { echo == $f; grep -v " it " ~/fleet_runs/$f.log | grep raw | tail -n 2 | cut -c1-200; tail -n 1 ~/fleet_runs/$f.log; }; done'
```

## 2. Read K2 and act. Tables are pre-registered; do not re-litigate them.

Proxy-model priors (`B:~/k2_test/analysis_wf/proxy_k2prior.log`):
- SOLO: centred ~.922, range .908-.936.
- BLEND W .30: median .940, P(>= .943) .15, P(<= .939) .41.

**S1 = SOLO read:**

| S1 | Labels for new members | Merge |
|---|---|---|
| >= .918 | mix50_logit | decide with the BLEND read below |
| .910-.917 | alternate seeds: odd = mix50, even = steven_v4 | **post merge requests 10-05/06** |
| < .910 | v4/mix50 alternate; fleet = merge/private asset only | **post merge requests 10-05** |

**BLEND read gives rho_eff** (row = S1, column = BLEND read):

| S1 \ BLEND | .940 | .941 | .942 | .943 | .944 |
|---|---|---|---|---|---|
| .910 | .86 | .84 | .81 | .79 | .76 |
| .915 | .89 | .86 | .84 | .82 | .79 |
| .920 | .92 | .89 | .87 | .84 | .82 |
| .925 | .95 | .92 | .90 | .87 | .85 |

What rho_eff means:
- **rho_eff <= .86:** the fleet route is live. Build more members; K3 at W* (`SP/jump_wf/box_schedule.md` sec 5.3).
- **rho_eff .86-.90:** only strength helps. Keep the queue; W* .15-.25.
- **rho_eff >= .90:** the fleet cannot lift the parent. Tell Eesh the merge is the only >= .945 route.

**Merge:** Eesh posts the requests himself (never by Claude).
- Refresh `Downloads/rsna-knee/merge_outreach.md` with today's facts: our independent ~.914-out-of-sample ConvNeXt fleet member, the K2 solo LB, the gold58 contamination finding, the V100 box, and the label work (mix50).
- Targets in that file: caoyupeng/Myo .949, mehmetbarankrat .948, prateekgroverin .945.
- Re-check the leaderboard first.

## 3. Box (V100 16 GB): what is running and what is armed

- **TTR fold-0** (pid 267158, `~/fleet_runs/f0_s1042_ttr`): v4 labels, 24 epochs.
  - At ep13: raw val .8757, gold58 .8838.
  - **Provisional G-T2: +.0065 over the v4 ref** (PASS bar +.008; MIDDLE otherwise).
  - Ends ~22:45 IST.
- **Gated fold-0** (pid 267280, `~/fleet_runs/f0_s42_gated`): `labels_gated75_logit`.
  - At ep4: val .8604, gold58 .8644, already ahead of v4 (.840/.836 at ep4) and below mix50 (.861/.881).
  - Ends ~23:30.
  - Its open question is rho to the parent: measure it on gold58 with `~/k2_test/gold58_arms.py` logic after it ends.
- **ARMED: `~/auto_j1.sh`** (pid 286637, log `~/fleet_runs/auto_j1.log`).
  - Waits for BOTH pids to exit, writes `~/fleet_runs/gt2_final.txt`, then guards: no python on the GPU and >= 12 GB free.
  - Then launches J1 ALONE:
    - **PASS/MIDDLE:** TTR fold-0 on mix50_logit, seed 1043, out `~/fleet_runs/f0_s1043_ttr_mix50`;
    - **KILL:** F1 all-data mix50 seed 43, `--stop_epoch 8`, out `~/fleet_runs/all_s43_mix50`.
  - Uses the unchanged trainer `~/bigjump_build/code/fleet_train.py`.
- **Rule from now on: ONE job at a time.** Two jobs together deliver 0.96x of one job alone (MEASURED: F1 1.085 s/it
  alone vs 2.22-2.25 shared). Sharing only doubles the time to every gate read.
- Day-by-day queue to 10-22, seeds, gates and launch templates: `SP/jump_wf/box_schedule.md` sec 6-7.
  - It is the scout's plan. Its skeptic review (`verify_box-schedule.md`) confirmed the timing facts but flagged the label switch and the outlook (see sec 5).
  - Production = all-data seeds at a fixed epoch; fold-0 runs only for gates.
  - Serving cap ~8 F1-equivalent members in the 9 h rerun (a TTR member ~0.5 F1-eq).

## 4. GPU throughput work (Eesh asked "INCREASE THROUGHPUT OF GPU")

Facts (MEASURED 10-04):
- The V100 is compute-bound: SM 100%, CPU load ~5 of 40.
- It is **thermally throttled**: SM 1035-1087 of 1290 MHz at 80-81 C, reason SW thermal slowdown.
  - That is a ~20% loss.
  - The fix is physical lab airflow, a passively cooled FHHL card. Ask the lab; do not touch GPU or system settings.
- The trainer already uses fp16 AMP, channels_last and cudnn.benchmark.
- Remaining levers:
  1. **Gradient checkpointing is always on** (~+30% compute). It is only needed when jobs share memory.
  2. **Eval of raw AND EMA on val+gold every epoch** = 12-17% of epoch time.
  3. **No torch.compile / fused AdamW.**

Partial benchmark (workflow `fleet-fastpath` stopped at ~15:32 when the session ended). It lives at `B:~/fast_build/`:
`bench.py`, `run_queue.sh`, `logs/results.jsonl`, `logs/mem2.jsonl`, `logs/smoke.jsonl`. These are MEMORY numbers, measured with a 3 GB cap next to the two live jobs.

| Config | Checkpointing | MiB per image (slope) | Full-batch estimate alone |
|---|---|---|---|
| TTR 224, 64 img/sample | none | ~60 | B=2: ~8.5 GB, **fits alone without checkpointing** |
| F1 320, 96 img/sample | none | ~124 | B=2: ~24 GB, does NOT fit. B=1 + accum 8: ~13 GB, tight |
| F1 320 | stage-0 only (`s0`) | ~82 | B=1 ~8 GB, fits |
| F1 320 | stages 0-1 (`s01`) | ~61 | B=1 ~6 GB, fits |

The only TIMING number is the TTR base smoke: 0.625 s/sample with 3 jobs sharing (not useful as an absolute).

**Next:** finish the fast path in `~/fast_build/code` (copy of the trainer; never edit `~/bigjump_build` or `~/fleet`):
- add opt-in flags `--no_grad_ckpt` / `--ckpt_stages`, `--compile`, `--fused_adamw`, `--eval_every/--eval_from` (always eval the last epoch and `--save_epochs`);
- defaults must stay byte-identical in behaviour;
- saved ckpt keys / args / model_sha must stay unchanged so K2 serving works (compile must not leak `_orig_mod.`);
- time it in a gap between jobs, or on J2 directly with an A/B on the first epoch;
- the workflow script to re-run is `C:\Users\eeshs\.claude\projects\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\workflows\scripts\fleet-fastpath-wf_463f2744-366.js` (resume ids do not carry across sessions; re-launch it).

Expected 1.3-1.6x members/day (GUESS) on top of the serial rule.

## 5. Research outputs from 10-04 (workflow `rsna-jump-research`)

All 4 scouts finished. Skeptic reviews ran for box_schedule and inner_injection only. The final synthesis
(`JUMP_PLAN_10-04.md`) was never written, because the session ended. Files are in `session_2026-10-04/jump_wf/` and
`session_2026-10-04/box/k2_test/analysis_wf/`.

- **`public_intel.md`, leaderboard (MEASURED 10-04 09:27 UTC):**
  - 5,082 teams; **we are rank 419 at 0.943.** The tie runs from rank 352 to 1295, broken by earlier submission. The public bronze cutoff is rank ~508.
  - Silver line .945, gold line .958, top .963.
  - 31 teams left the .943 tie in 24 h, 78 in 3 days, mostly by +.001-.003.
- **`public_intel.md`, public notebooks:**
  - **Nothing public above .943 can be forked.** All 5 notebooks above .943 attach PRIVATE datasets.
  - The d4-family ones (pjmathematician d4-blend .946 / d4-lite .945; aastikrajan15 s75-w50/w60 .945) are **our parent's d4 stack + a private own fleet at ONE global rank weight .45-.60.** That is exactly our K2 design at a higher W.
  - analyticaobscura "RSNA Knee V4" .947 is a standalone private-member runner.
  - **DINOsaur V5 = .943** (internal re-weighting, no new weights, no lift). Its `-train` notebook has no ConvNeXt weights public yet: watch item only.
  - Unscored public weights from teams above us: tbongc (.947), 2 Raptor CoAtNet full-data retrains + 4-fold EffNet-B0 MIL; kawaharataishi (.944), CoAtNet-384 distilled SWA. Each needs integration work + an LB slot.
- **`lb_proxy_model.md` (+ box `proxy_*`):**
  - The binormal blend model fits gold58 only with an effective-r offset (+.062; LOBO MAE .0010).
  - Our fleet member is parent-like: r_eff .918 vs the out-of-sample Raptor+CoAt side, against .78-.82 for the crop members.
  - The crop LB reads imply r on the LB is even higher (.88-.94), so the prior for the fleet is **r_LB ~.95 (.90-.97).**
  - K2 priors: SOLO ~.922 +/- .008; BLEND median .940, P(>= .943) .15.
  - Ensembling one family raises solo only +.003-.006 and *raises* r.
  - **P(.950 from our own fleet) ~1-3% by 10-12, 2-5% by 10-22.** To lift the parent the ensemble must be about as strong as the parent at LOW r (+.003 needs .9255 at r .80, .9395 at r .90).
  - So the lever is LOWER correlation (labels / recipe / family), not more seeds. The aastik/pj notebooks show a d4 + low-r ConvNeXt fleet at W .45-.6 does reach .945-.946.
- **`inner_injection.md` + `verify_inner-injection.md`:**
  - The parent graph is reconstructed exactly.
  - The 5th-Raptor-arm injection gives ~0. The skeptic showed it is no better than a smaller global W in the bracket that matches the LB.
  - **Verdict: not worth a slot or the code. Dropped.**
- **`box_schedule.md` + `verify_box-schedule.md`:**
  - The serial rule is CONFIRMED (sharing = 0.93-1.01x).
  - TTR-alone speed is MODEL (0.36-0.55 s/it).
  - Skeptic flags:
    - "halves gate time" is an overclaim;
    - the S1-keyed switch to v4 labels has no direct support;
    - the .943-.947 outlook should take the model's own .002-.004 haircut.
- **Label question, the key one for the next jobs:**
  - mix50 members are strong but parent-like. The proxy says r, not strength, is the binding constraint.
  - After tonight, compare on the gold58 out-of-sample arms (`gold58_oos_blend.py` / `proxy_fit.py` logic): **TTR-v4 f0 (running) vs TTR-mix50 f0 (J1) vs gated f0 vs F1-mix50 f0**, on solo AND r_eff vs raptor_coat_side.
  - Pick production labels by the predicted blend lift, not by solo alone.
  - J1 runs mix50 so the TTR label A/B is clean.
  - If K2 BLEND <= .939, switch the next production jobs to the lowest-r labels that keep solo.

## 6. Slots

- 10-04: 2 left (nothing worth submitting until K2 reads).
- 10-05: 5. Candidates only after the reads: K3 (more members at W*), inner-injection variant (only if BLEND >= .943), a public notebook > .943 if found.
- Never submit pure repro_0946.

## 7. Rules that still hold

- Kaggle GPU only for submission commits; the V100 does everything else.
- Eesh handles credentials and posts merges himself; never sign or post for him.
- Box: never `pkill -f` (it kills the ssh shell), kill by PID only after checking its command line.
- Laptop: no heavy compute; no `grep -r` / `find` over big trees.
- Only `submission.csv` is submittable from a code kernel (other names -> 400).
- No em dashes in docs.
