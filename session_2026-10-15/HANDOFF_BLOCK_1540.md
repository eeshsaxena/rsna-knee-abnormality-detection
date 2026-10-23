# >>> HANDOFF 2026-10-15 15:45 IST (for the NEXT Claude session; read this block FIRST, then PROGRESS_15-10-2026.md bottom, session_2026-10-15/PREREG_10-15.md) <<<
**Eesh's goal (verbatim intent):** stretch the lead NOW, before .957 becomes saturated by the copy crowd. Use EVERY compute source:
- the university box;
- Vast credit (~$7.98; cap $7.40 for KC);
- a friend's 9 h 40 m of Kaggle GPU (only if the friend is a teammate);
- Eesh's Kaggle quota for commits.

Work relentlessly until he says stop. Keep the private LB in mind: read-positive DIFFERENT model families, never public-tuned weights.

Rules that never change:
- No em dashes.
- OAI data (images / labels / ids / OAI-derived preds) never leaves the university box. OAI-trained WEIGHTS go to private Kaggle datasets only after an OAI-clean scan.
- Never print tokens or signed URLs.
- Box queue only via `bash ~/queue/qctl.sh`. Never interrupt a running job. Never pkill (kill by verified PID only).
- Agents never submit or create datasets; main does.
- Pre-register every entry in session_2026-10-15/PREREG_10-15.md BEFORE its submit.
- Gate every commit. One change vs the current API team best.
- Read by API best-id with the censoring rule R1 (Amendment 3).
- Kaggle kernel-output downloads sometimes return 502 HTML pages: check for "Server Error" and merge refetches (s6f2/merge_fetch.py).
- Claude-app scheduled tasks may pause for permission. Always check session_2026-10-14/submits.log AND the Kaggle submissions list before a hand submit.

## 1. Where we are (15:45 IST 10-15)
- **TEAM BEST = V224-FLIP-AO + S6fall .2 = public .957, rank 158 / ~6,012** (silver cut ~r301), ref 57201874.
  - Kernel `eeshsaxena/rsna-sub-v224flips6f` v1 (session_2026-10-15/s6f/).
  - J twin (JPEG Lossless decoder insurance) `rsna-sub-v224flips6fj` v1 is commit-tested (jpeg/overlay/r2, J_CHECK 16/16) but NOT yet scored.
- 10-15 reads:
  - FLRM .956 not best (r224 mirror dropped);
  - FF20 .956 not best (K14 fleet .20 dropped);
  - S6F .957 BEST (+189 frozen teams).
- 10-15 slots 5/5 used. **Next slots open 05:30 IST 10-16.**
- **Pending submissions:**
  - `57218523` FLIP-AO + S (R7-teacher student) scores ~16:30-17:30. Its base FLIP is superseded, so a non-move is UNREAD.
  - `57222884` composite S6F + S (= best + student) scores ~20:30-21:30. This is the real test of S.
- Read chains die with a session. Re-arm in a new one:
  `cd /c/Users/eeshs/Downloads/rsna-knee && PREV=<latest research960/lbprobe/api_lb_chain_*.json basename> bash session_2026-10-14/api_chain.sh 57218523 57222884`
  (log session_2026-10-14/api_chain.log)
- One-off LB check: `cd session_2026-10-14/research960/lbprobe && python fetch_api_lb.py X.json && python api_frozen.py <prev>.json X.json`

## 2. Box (training@192.168.10.30, 1 x V100 16 GB, 40 cores) = KC1 knee-centred compartment family
Docs: session_2026-10-15/box/BOX_PLAN_1015.md, BOX_PLAN_CRITIQUE_1015.md, KC1_READINESS.md, KC_QCTL_1015.txt; prereg PREREG_10-15 Amendment 6.
- Queue (qctl, 0 BAD):
  - T_a f3 running (ends ~16:00);
  - L126 KC1 f0 s2001 (gate ~/kc/KC_READY.verdict wait 120);
  - L127 AD s2003 dread;
  - L128 AD s2002 T_b / L129 AD s2002 dread2 (exactly one runs; decided by ~/kc/gate/s_read.verdict);
  - L130 AD s2004 spec;
  - L131 KC2 s2011 EffNetV2-S spec;
  - L132 T_a f4 last;
  - L13 f3/f4 #HOLD.
- KC_READY was 7/8 at 15:32; only M4 (shim regression part c) was pending.
  - Check: `ssh -o BatchMode=yes training@192.168.10.30 'bash ~/kc/kc_ready.sh --status'`.
  - IF all 8 ok: `ssh ... 'bash ~/kc/kc_ready.sh'` (expect `PASS kc_ready ...`).
  - IF the regression FAILs: `bash ~/kc/code/shim/uninstall_shim.sh`, then `bash ~/kc/kc_ready.sh --fail "shim regression"`. The KC lines then SKIP and the box falls to T_a f4.
- Shim: ~/fast_build/code2/fleet_train.py md5 8d12515f (dispatches `--kc` to fleet_train_kc.py; otherwise unchanged).
  - Filed as PREREG_TEACHER A3 in ~/r7t_labels/PREREG_TEACHER_A3_SHIM.md (separate file).
  - **code2 is FROZEN** (the rental aborts on any md5 change).
- After f0 is #RUN, arm the early gate: `ssh ... 'setsid nohup bash ~/kc/code/kc_g1_early.sh > ~/kc/logs/g1_early.log 2>&1 < /dev/null &'`. Output ~/kc/gate/kc_f0_ep3_ablation.txt about 2 h after the f0 start.
- **s_read.verdict (A6 member 3), right after 57218523 scores:**
  - IF 57218523 became the team best: `echo 'PASS S positive (57218523 moved the best)' > ~/kc/gate/s_read.verdict`
  - ELSE (non-move or UNREAD): `echo 'FAIL S not positive (57218523 non-move or UNREAD)' > ~/kc/gate/s_read.verdict`
- Box helpers: idle alarm ~/fleet_runs/idle_alarm.sh (writes ~/fleet_runs/IDLE_ALARM); kc_gate_watch.sh (pid 1160489).
- Competition zip (for the S6 seed px cache): ~/rsna_comp/fetch.sh, 8 resumable parts from a signed URL in ~/rsna_comp/.u (never print it).
  - At 15:32 the parts were done and being joined into rsna-knee-abnormality-detection.zip; look for FETCH_DONE in ~/rsna_comp/fetch.log.
  - IF the URL expired: `python session_2026-10-15/box/get_comp_url.py 2>/dev/null | ssh -o BatchMode=yes training@192.168.10.30 'umask 077; cat > ~/rsna_comp/.u'`, then restart fetch.sh (it resumes).

## 3. Agents / workflows that were running in the OLD session (they die if it is closed)
- **Rental agent "rental_kc"** (kit session_2026-10-15/rental_kc/, OWNER.lock rkc-launch-1015-7f3a):
  - trains KC1 all-data members on a rented 2 x A100 (cap $7.40 new spend; credit ~$7.98);
  - rents ~45 min before the early G1 and goes ONLY on `ablation_pass=True`;
  - members in priority s2003, s2002 variant, s2004, KC2;
  - box fallback re-queue armed (KC_FALLBACK_QCTL=1).
  - IF the old session is closed: the box driver + on-instance guard (self-destroy) keep running on their own. A NEW owner must read rental_kc/RUN_LOG.md + RUNBOOK, check `vastai show instances`, and take OWNER.lock before touching Vast. Never two owners.
- **s6seed-prep workflow** -> session_2026-10-15/box/S6SEED_READINESS.md (second S6fall-family model, seed 43). Then main runs the unzip + px build (cores 12-27) and the GPU job via the PAUSE protocol (BOX_PLAN 3.5) under Amendment 7.
- **forum-sweep workflow** -> session_2026-10-15/silver/FORUM_DIGEST_10-15b.md.

## 4. Decision tree (IF / THEN, in time order)
1. **~16:00 T_a f3 ends.**
   - IF KC_READY PASS: f0 launches (look for LAUNCH KC1 f0 in ~/fleet_runs/queue_runner.log); arm kc_g1_early.sh.
   - IF not PASS by 17:45: f0 DEFERS and T_a f4 runs (acceptable).
2. **~16:30-17:30 read of 57218523 (FLIP + S).** Write s_read.verdict as above. A move is unlikely (it would mean FLIP + S > FLIP + S6fall).
3. **~17:45 early G1 (epoch-3 ablation).**
   - IF both margins are negative: main may stop f0 (a stopped f0 = G1 FAIL; see A6). The rental aborts by itself.
   - ELSE the rental trains members.
4. **~20:15 f0 ends.** Its end hook writes ~/kc/gate/kc_f0.verdict.
   - IF PASS: AD members run (box + rental).
   - IF FAIL: KC1 is dead; the AD lines SKIP. Put the GPU on the S6 seed (Amendment 7).
5. **~20:30-21:30 read of 57222884 (S6F + S).**
   - IF it moved the best: S adds on S6F; the composite is the new best.
   - ELSE: S is a valid negative on S6F (base unchanged); drop S from the final candidates.
6. **KC1 entry (10-16).** Needs >= 2 members passing G1 (f0 + the first AD).
   - **TODO, NOT BUILT YET: the in-kernel KC stage.** Append KC cells to the best kernel, like the 3 S6F cells (s6f/build_v224s6f.py pattern):
     - kc_worker.py serve path (box ~/kc/code; d96 reader sha 30aa72fd, header side, canon or both-canon for weak-zone studies, raw weights at E*, fp16);
     - leg = mean of member sigmoids; blend .8 rank(base) + .2 rank(KC leg);
     - start guard + VISIBLE canary on a free column (ACL is S6F's, Synovitis fleet/S, Fracture r224);
     - DEC wheels on PYTHONPATH.
   - Build it NOW with stand-in members so it is ready by 10-16 morning.
   - Then: pack (OAI-clean scan; members are RSNA-only) -> private dataset (main) -> GPU commit -> gate (C3 serve parity must PASS first) -> runtime review (p99 <= 8.0 h on the S6F base; S6F median ~6.3 h, so the KC stage must stay <= ~0.45 h hidden) -> amendment -> submit.
7. **10-16 05:30 slot 1 IF KC1 is not ready:** `rsna-sub-v224flips6fj` v1 (J twin of the best; commit-gated). It is needed as a scored, pick-eligible form; its public score should equal S6F's.
8. **S6 seed (10-16/17):** IF S6SEED_READINESS says ready and PX_PARITY passes: write Amendment 7, then launch on the GPU after the KC1 members (PAUSE protocol), or on a fresh Vast rental if credit is added (RSNA-only). Entry form = 2 S6fall-family checkpoints in the S6F stage (re-sim runtime with c x 1.25).
9. **Final picks (10-22; write the rule as an amendment before 10-20):**
   - Pick B = the API team best in its J form (scored).
   - Pick B' = the best entry that differs from it (e.g. without its newest edge).
   - Both must be scored and canary-free.
10. **Team merge** (deadline 10-15 23:59 UTC = 05:29 IST 10-16): Eesh decides. Draft to pand999 (r127 .957) in session_2026-10-15/teamup/DRAFT_pand999.md; Eesh sends it himself.
11. **Friend's Kaggle GPU (9 h 40 m):**
    - ONLY if the friend is a TEAMMATE (Kaggle bans private sharing outside the team). Ask Eesh.
    - If yes: a notebook pack training one KC member (e.g. dread seed s2004 or KC2), KC store 12.9 GB as a private dataset shared with the friend (RSNA-only), modelled on session_2026-10-11/kaggle_frac75d/ (swapiy pack).
    - The trained weights come back via the friend's private dataset -> box -> the same G1 gate.

## 5. Research directions (IF this, THEN that)
- IF FORUM_DIGEST_10-15b shows a new .958+ public notebook with ALL inputs public: bait check first (scored-version dataset pins vs current: watch.py drift + manifest sha compare, as for heliosli 0-9573, which was R7 in disguise). IF genuine: rebase via watch/swap_builder.py as ONE change, pre-registered.
- IF a new public model family appears (as S6fall did): port it as an appended stage on the best (s6f/build_v224s6f.py: own subprocess, parity vs the author's commit output, start guard, visible canary, coverage rule) at .2 rank.
- IF KC1 reads positive: KC2 (EffNetV2-S, queued as spec) + more dread seeds under Amendment 8; rent again if Eesh adds Vast credit.
- IF KC1 reads negative: stop KC; give the GPU to the S6 seed.
- Runtime is the binding constraint for stacking. Before adding any stage, sim it (s6f/sim_s6f_runtime.py, s6f2 sims) and keep p99 <= 8.0 h.

