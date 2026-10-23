# rental_kc RUNBOOK: KC1 all-data members (+ KC2 f0) on a rented 2 x A100 (RSNA-only). No em dashes.

What it trains: the box queue lines of `box/KC_QCTL_1015.txt` (PREREG_10-15 Amendment 6 + Amendment 8 stub), in queue priority:
AD s2003 dread `fA_s2003_kc256_dread`, AD s2002 (T_b `fA_s2002_kc256_r7tb` if `~/kc/gate/s_read.verdict` is PASS, dread2
`fA_s2002_kc256_dread2` if FAIL, none while unread), AD s2004 dread3 `fA_s2004_kc256_dread3` (speculative), KC2 f0 s2011
`f0_s2011_kc256_ev2s_r7tb` (speculative). argv = the LIVE box queue line token for token, `~` -> `/workspace/rsna`
(`box/ops.sh members` / `argv-check`); run as `python fleet_train.py <argv>` from the pushed code2 (the shim dispatches `--kc`),
exactly like `queue_runner.sh` (incl. its one early-OOM retry with `--grad_ckpt 1`). Out dirs = the queue lines' out dirs.

## Pieces (laptop `session_2026-10-15/rental_kc/`, box `~/rkc/src/`, instance `/workspace/rsna`)
| file | where | role |
|---|---|---|
| `box/pack.sh` | box CPU (nice 19, cores 30-39) | bundle = installed code2 (shim md5 8d12515f + KC files) + `~/kc_store/store` (4,407 RSNA npz, HARD LINKS) + T_b (md5 d206bbcf, sha listed in LABELS_READY) + dread table (make_dread_labels receipt, src md5 96e6723f) + folds5 + gold74 + ConvNeXt-S safetensors (2457488b) + EffNetV2-S HF cache + kit + members.tsv (from the live queue); `pack_check.py tables` + `oai` (OAI_ASSERT); parity probe; MANIFEST.sha256; BUNDLE.json (code md5s). `box/warm_cache.py` pre-hashes the store |
| `launch.sh --search / --dry-run ID / ID` | laptop | 1 narrow search (2 x A100, limit 25); OWNER.lock; preflight (kit synced, bundle kit current, box code2 == bundle, s2003 argv identical); `vastai show instances` must be []; projection guard (plan-2 WORST cost + spent <= $7.40); rent with the guard in onstart; guard ARMED check; box key + key keeper; run.env; box driver; box `kc_fallback.sh --loop`; laptop watcher |
| `box/driver.sh` | box (setsid) | D1 ssh, D2 push (8 tar streams) + PUSH-RATE GATE (< 6 MB/s after 4 min -> abort), D3 setup.sh (manifest, OAI_ASSERT, pins, GPU), D4 parity (every field), D5 smoke (phase 1: 4 trainers 2 per GPU 8 min; phase 2: 1 per GPU 6 min; outputs under /workspace/rsna/smoke, never members), D6 WAIT for `~/kc/gate/kc_f0_ep3_ablation.txt` (`ablation_pass=True` = GO; `False`, a kc_f0 FAIL, `~/rkc/STOP` or GATE_LATEST = abort + destroy), D7 GO (S read -> s2002 variant, line ACTIVE + argv identical, code2 unchanged, `decide.py --go`, EMPTY placeholders, lanes, RUN_STARTED, guard tightened), D8 fetch loop |
| `box/decide.py` | box | `--smoke`: per kind (ConvNeXt-S AD / EffNetV2-S KC2) single + paired s/it -> member hours; `--go`: time budget T = min((cap - spent_now)/$/h - fetch - tail, guard - now - 0.5 h); SEQ plan (1 per GPU, members in priority, greedily to the GPU that frees first, admitted while end x 1.10 <= T) vs PAIR (4 at once) ; more members wins |
| `box/fetch_loop.sh` + `verify_member.py` | box | every 5 min: pull finished members, verify (sha256 of every file, no extras, closing line, expected epoch files, ckpt args == queue argv + bundle + every recipe key == the box f0 ckpt except fold/seed/save_epochs/labels/decode_probe, receipt argv + labels sha), place into the EMPTY placeholder, `~/fleet_runs/rkc_<out>_receipt.json`; KC2 G1 verdict -> `~/kc/gate/kc2_f0.verdict` only if absent; failed / skipped -> placeholder released; FINAL_FETCH_OK |
| `box/kc_fallback.sh --loop` | box | qctl-free: when the rental is over, releases orphan EMPTY placeholders and writes `~/rkc/REQUEUE_CMDS.sh` (exact `qctl top` lines) for any reserved member whose box line the runner already REFUSED; runs qctl itself only if MAIN armed it with `KC_FALLBACK_QCTL=1` |
| `remote/guard.sh` | instance (onstart) | self-destroy on hard deadline (cap; only tightened), ABORT, FINAL_FETCH_OK + 15 min, boot fuse (no RUN_STARTED by GATE_LATEST + 20 min), orphan fuse; kills only verified PIDs (no pkill); wipes the data first |
| `remote/kc_lane.sh`, `start_lanes.sh`, `stop_member.sh`, `status.sh`, `setup.sh`, `parity.py` | instance | lanes (members in sequence per GPU, admission vs DEADLINE_EPOCH), receipts + sha256 lists, verified-PID stops, progress |
| `watch.sh`, `destroy.sh --yes`, `status.sh` | laptop | backup destroy + spend, kill switch, one-screen status |

## Launch (after main: C5 dread labels, C8 + M4 shim, KC lines in the queue; f0 running)
1. Box: `bash ~/rkc/src/box/pack.sh` -> must end `BUNDLE READY` (STOP = do not rent).
2. Laptop: `bash launch.sh --search`; `RKC_OWNER=<OWNER.lock owner> GATE_LATEST_HM=HH:MM bash launch.sh --dry-run <id>`; then
   `RKC_OWNER=... PIN_MACHINE=<machine> GATE_LATEST_HM=HH:MM WAIT_H=<h> bash launch.sh <id>` timed so that push + setup + smoke
   (~45 min) end near the expected early-G1 time (f0 epoch-3 ckpt + ~55 min).
3. Monitor: `bash status.sh` (or box `tail ~/rkc/driver.log ~/rkc/fetch/fetch.log`). Stop before GO: `touch ~/rkc/STOP` on the box.
4. End state: members in `~/fleet_runs/<out>` (+ `.log`, `rkc_<out>_receipt.json`), `~/rkc/FINAL_FETCH_OK`; the guard
   self-destroys 15 min later; `watch.sh` destroys too if alive. Kill switch: `bash destroy.sh --yes`.

## Money
- Cap $7.40 TOTAL new spend incl. failed attempts and data = credit drop since `ledger.env CREDIT_START` (set at the first
  rental). Budget per rental = min(cap - spent - $0.15, credit - $0.55). Guard hard deadline = min((budget - data) / $/h, 7 h).
- GO re-plans with MEASURED speeds and the spend so far; every lane re-checks admission before each member.
