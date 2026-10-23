# RUNBOOK: Vast folds in parallel with the V100 box

> **2026-10-09 01:15 UPDATE: the G320' gate FAILED (best raw .8877 < .8909), so the 320 folds are cancelled.**
> The kit now trains the box queue's **Q4 TTR288 s1045 f1-f4 frac75** lines (`remote/kit.env`: RUNTAG ttr288frac75,
> TTR 288, gate `~/queue/gates/frac75.verdict` = PASS). The argv is copied from `~/queue/queue.txt`
> (`remote/argv_template.txt`, 40 tokens, the default bs 2 x accum 4). Outputs go to `~/fleet_runs/f{1..4}_s1045_ttr288frac75`.
> The box trains Q4 f0 frac75 itself (launched 01:15, ~5.8 h, ends ~07:05). Placeholders f1-f4 must exist before
> ~07:05. Vast is authorized (Eesh: "use the Vast credit fully but judiciously"); credit $9.01; budget $8.50.
> The fold on our V100 takes ~5.8 h, so MODEL ~1.9 h on a 4090 and ~3.3 h on a 3090. The guard margin is now 0.5 h.
> Everything below still applies, with "320" read as the active family in kit.env and folds "1 2 3 4".
> Section 4 does not apply (the box runs f0, Vast runs f1-f4); the timeline table at the end is for the cancelled
> 320 plan, and the 288 numbers are in `offers.md`.
> **LIVE RUN (2026-10-09):** instance **54910062** (label rsna-ttr288frac75-1009021757), offer 29390213:
> 4x RTX 3090, Finland, verified datacenter, $1.096/h.
> - Rented at 02:18 IST. The guard self-test returned http 200 (the backstop works). Hard self-destroy at 09:41 IST.
> - The watcher cap is 09:11 IST; lanes may start a fold until 08:51.
> - Setup finished at 02:42 IST: store copied at 20 MB/s on the direct route (the proxy route ran at ~2 MB/s);
>   4407/4407 store md5 identical; all parity hashes identical.
> - S2 fix: opencv-python-headless 4.14.0.94 is installed with `--no-deps`, as on the box (its metadata wants
>   numpy 2; the box runs numpy 1.26.4). Parity proves identical decoding.
> - Training started 02:42 (f1) and 02:53 (f2-f4). MEASURED 0.28-0.29 s/it, ~8.3 min train + ~2 min eval per
>   epoch, so ~3.7 h per fold. ETA: folds done ~06:25-06:40 IST, fetched + destroyed ~06:50. Rental ~4.6 h, ~$5.0.
> - Box placeholders f1-f4 `_ttr288frac75` were created at 02:20 IST.
> - The watcher must stay alive. If the session that started it ends, re-run `bash vast_watch.sh` (Bash
>   run_in_background) from this folder; it resumes from state.env.

> **RESULT:** all 4 folds finished, were fetched and verified (md5, closing line, ckpt args; receipts at
> `~/fleet_runs/vast320_fF_s1045_ttr288frac75_receipt.json`).
> - Closing lines (raw / ema): f1 .8922/.8912, f2 .8917/.8906, f3 .8788/.8778, f4 .8914/.8903.
> - The instance was wiped and destroyed at 06:50:37 IST; the account now has 0 instances.
> - Spent ~$4.95 (credit $9.01 -> $4.06).

> OAI data never leaves the box: `box_push.sh` refuses any source resolving under `~/oai_*` or `~/union_store`,
> any symlink, and any non-npz file in the store.

Goal: train the 320 px folds f2, f3, f4 (seed 1045, frac75 labels, the box's exact argv) on one rented Vast
instance while the box trains f1, so the 5-fold 320 set is done ~08:50 IST 10-09 instead of ~06:45 IST 10-10.
Hard money rules: total spend <= $8.50 and >= $0.50 credit left. Competition data is wiped on the instance and
the instance is destroyed at the end.

Files (all in this folder; run everything from Git Bash on the laptop, from this folder):

| file | role |
|---|---|
| `offers.md` | offer shortlist snapshot + speed/cost MODEL + recommendation |
| `vast_launch.sh` | `--search` (read only) / `--dry-run OFFER` / `OFFER`: rents, waits for ssh, attaches the box PUBLIC key, creates the box placeholders |
| `vast_setup.sh` | pinned packages (= box versions), box -> instance copy run ON the box, md5 parity of all 4407 store files + code + csvs + HF weights, parity.py hash/count comparison |
| `vast_train.sh` | one `fleet_train.py` per GPU (CUDA_VISIBLE_DEVICES), lanes when folds > GPUs |
| `vast_watch.sh` | 10-min poll; enforces the caps; fetch, then destroy; before a launch it keeps the 2FA session alive |
| `vast_fetch.sh` | box pulls outputs into `~/fleet_runs/fF_s1045_ttr320frac75/` + `.log`, verifies, writes a receipt; falls back to the laptop if the box is unreachable |
| `vast_destroy.sh` | wipe + destroy + verify gone + give unfinished folds back to the box queue (manual kill switch too) |
| `box_placeholders.sh` | `status` / `create` / `release` (see `BOX_QUEUE_NOTE.md`) |
| `remote/` | box side: `box_ops.sh`, `box_push.sh`, `parity.py`; instance side: `lane.sh`, `status.sh`, `guard.sh` (self-destroy backstop, installed at create time through onstart) |
| `dryrun.sh` / `dryrun.log` | the no-rental proof |
| `state.env`, `logs/` | written at run time (nothing yet) |

## Safety layers (outermost first)

1. Budget = min($8.50, credit - $0.50). The watcher finalises when spend (max of credit drop and $/h x elapsed)
   reaches budget - $0.40, when credit reaches the floor + $0.40, or at the time cap = (budget - $0.40)/($/h) - 1 h.
2. Lanes only START a fold if it can end 20 min before the time cap (measured duration of the previous fold, else
   the MODEL mid estimate). A skipped fold goes back to the box.
3. `guard.sh` on the instance self-destroys (after wiping the data) at create + (budget - $0.40)/($/h), or 15 min
   after `FINAL_FETCH_OK` if the laptop's destroy did not land. It uses the per-instance key Vast injects, never
   printed. At start it logs `self-test ... http NNN`: 200 means the backstop works; anything else means the laptop
   watcher and the 2FA session are the only stop.
4. Any instance that never reaches `running` within 20 min is destroyed by `vast_launch.sh` itself.

## Step 0, before Eesh sleeps (~23:00 IST): Vast 2FA session (BLOCKER without it)

Every Vast call except `search offers` needs a 2FA session, including create, destroy, show instance and show user.
The session from 10-07 has expired. The session key "expires after inactivity", so the watcher keepalive below
keeps it alive overnight.

```
V=/c/Users/eeshs/AppData/Local/Programs/Python/Python312/Scripts/vastai; export PYTHONIOENCODING=utf-8
$V tfa send-email                                   # prints a short secret; Vast emails Eesh a code
$V tfa login -t email -s <secret> -c <code>         # Eesh types the code
$V show user --raw | python -c "import sys,json; t=sys.stdin.read(); print(json.loads(t[t.index('{'):])['credit'])"
$V show instances                                   # leftovers bill storage (e.g. ARC 54604662 if only stopped)
```

If the ARC instance 54604662 still exists, it bills ~$0.05/h of storage and holds the 29 GB Qwen model. Destroy it
only with Eesh's OK: `$V destroy instance 54604662 -y`.

Then start the watcher in the background. Claude: Bash tool `run_in_background`, because a plain `nohup` dies on
Git Bash. Re-run it after any session restart; it resumes from `state.env`.

```
bash vast_watch.sh          # keepalive now (credit read every 10 min), becomes the run watcher after the launch
```

Keep the laptop awake and on the lab network: the box is reachable only from 192.168.0.x, and during this prep the
laptop flapped to 10.19.x several times. The kit survives this: the box-side copy runs detached on the box, and
the fetch falls back to the laptop and pushes later.

## Step 1, ~01:30 IST: gate

```
ssh training@192.168.10.30 'cat ~/queue/gates/g320.verdict'
```

`g320_gate.py --wait` writes the verdict within a minute of the f0 probe's closing line. It must start with `PASS`.
On `FAIL`, stop: rent nothing (the box queue SKIPs the Q3b lines by itself).

## Step 2: pick the offer (read only)

```
bash vast_launch.sh --search --folds "2 3 4"
```

Take the top `fits=yes` row: verified datacenter (`V DC`) first, then `V`. At snapshot time that was 29390213
(4x RTX 3090, Finland, V DC, $1.096/h). If no V DC row fits a 3-fold plan, see the choice in `offers.md`
(recommendation 2).

## Step 3: dry run on that offer, then launch (spends money)

```
bash vast_launch.sh --dry-run 29390213 --folds "2 3 4"
bash vast_launch.sh 29390213 --folds "2 3 4"
```

The launch takes ~5-10 min. It checks the PASS verdict, credit and budget, re-checks the offer, creates the box key
`~/.ssh/vast_rsna_ed25519` if missing (only its .pub leaves the box), rents with the guard onstart, waits for
running + ssh, attaches the box key (`vastai attach ssh`), and creates EMPTY placeholders f2-f4 on the box. It can
be re-run: an instance already in `state.env` is resumed, never rented twice.

## Step 4: setup (~20-40 min, MODEL)

```
bash vast_setup.sh
```

Steps S1 (facts and gates: GPU count, CUDA, /dev/shm >= 2 GB per trainer, disk), S2 (pip pins = box: numpy
1.26.4, pandas 3.0.5, scikit-learn 1.9.0, scipy 1.17.1, opencv-python-headless 4.14.0.94, timm 1.0.29,
huggingface_hub 0.36.2, safetensors 0.8.0, PyYAML 6.0.3, pillow 12.3.0; torch 2.5.1 cu121 from the image), S3 (kit),
S4 (box_push.sh detached on the box: code, csvs, HF cache, 11.7 GB store in 4 rsync streams, then full md5 parity),
S5 (parity.py on box and instance: pixel_sha 6f49c74fc48f3345, tensor_sha 08ec55bc155be309, select_sha
91bb8a65594a25bd, jpeg_fp 34819a6207119a3b, model_sha f141aeed136dfc01, 4-study decode probe, backbone weight hash,
fold counts, all identical). Any mismatch stops with a message. Each step can be re-run.

## Step 5: train

```
bash vast_train.sh            # folds from state.env; GPU0 f2, GPU1 f3, GPU2 f4 on a >= 3-GPU offer
```

Argv per fold = the box's "Q3b TTR320 fF" line token for token, with `~/` replaced by `/workspace/rsna/` (checked
by `box_ops.sh argv-check` at launch). The only environment difference is `CUDA_VISIBLE_DEVICES=G`,
`HF_HUB_OFFLINE=1` and `OMP_NUM_THREADS=4`.

## Step 6: watch (automatic)

`tail -n 5 logs/watch.log`, or a single poll: `bash vast_watch.sh --once`. Each line shows status, spend vs budget,
credit, minutes vs cap, per-fold `fN:state epE val.. min/ep`, and GPU utilisation. At the end the watcher runs
`vast_fetch.sh --auto`, writes FINAL_FETCH_OK, runs `vast_destroy.sh --yes` (wipe, destroy, verify gone), and
releases any fold that was not fetched complete back to the box queue.

## Step 7: check

```
bash box_placeholders.sh status 1 2 3 4
ssh training@192.168.10.30 'tail -n 1 ~/fleet_runs/f{2,3,4}_s1045_ttr320frac75.log; ls ~/fleet_runs/vast320_*receipt.json'
$V show instances            # must not list the rsna320-* instance
```

## Manual levers

| situation | command |
|---|---|
| stop everything now | `bash vast_destroy.sh --yes --reason manual` |
| fetch what is finished | `bash vast_fetch.sh` (complete folds) / `--partial F` / `--auto` |
| box unreachable at fetch time | automatic `--to-laptop` (copies to `fetched/`), later `bash vast_fetch.sh --push-pending` |
| give a fold back to the box | `bash box_placeholders.sh release F` |
| watcher died (session restart) | `bash vast_watch.sh` again (Bash run_in_background) |
| ALERT_2FA in logs/ | Eesh repeats step 0 (send-email + login); then re-run `vast_destroy.sh --yes` if a destroy was refused |
| also keep epoch ckpts | `FETCH_EPOCH_PT=1 bash vast_fetch.sh ...` (~3.7 GB per fold) |

## Expected timeline and cost (MODEL; launch ~01:40 IST)

| plan / offer | create to train start | training | Vast done | 5-fold set done | rental | cost mid / worst |
|---|---|---|---|---|---|---|
| A: 29390213 4x3090 FI V DC, f2-f4 one round | ~0.5 h (02:10) | 4.2 h (worst 5.2) | 06:50-07:50 | ~08:50 10-09 (box f1) | 5.1 / 6.1 h | $5.59 / $6.73 |
| A: 38922997 2x4090 TW V, f2+f3 then f4 | ~0.5 h | 2 x 2.4 h (worst 3.0) | 07:30-08:45 | ~08:50 10-09 | 5.8 / 7.0 h | $4.83 / $5.83 |
| A: 54822069 3x4090 NO V, one round | ~0.5 h | 2.4 h (worst 3.0) | 05:00-05:40 | ~08:50 10-09 | 3.3 / 3.9 h | $5.73 / $6.74 |
| B: 36643031 2x4090 UK V DC, f3+f4 | ~0.5 h | 2.4 h | 05:00-05:40 | ~16:10 10-09 (box f1, f2) | 3.3 / 3.9 h | $4.68 / $5.53 |

Box alone: f1-f4 from ~01:34 at 7.3 h each, so the set is done ~06:45 IST 10-10. Plan A therefore saves ~22 h and
Plan B ~15 h. The 11.7 GB box upload speed is unknown (not measurable without an instance); `box_push.sh` logs the
MB/s. At 100 Mbit/s it takes ~16 min, which is inside the 0.6 h setup allowance.
