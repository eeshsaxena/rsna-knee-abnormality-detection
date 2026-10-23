# FIXES_A1: PLAN_TEACHER FX1-FX8 applied, PREREG_TEACHER Amendment A1 frozen, realistic SIM dry run, launch kit (10-14 IST)

Fix agent, R7-teacher line. No em dashes. Tags MEASURED / MODEL / GUESS. Nothing submitted, no dataset created, box GPU and
~/queue untouched by the fixes and the dry run (box CPU only: cores 34-39, nice 19, ionice idle). Run trail: `RUN_LOG.md`.

## 0. Bottom line

- **FX1-FX6 applied, PREREG_TEACHER Amendment A1 frozen 14:55** (sha256 `456f38a94723cebad89cdcd1a5eb5774e5e8183e0765493aca15c3a8b3e15035`;
  the pre-A1 part = the first 14,577 bytes, sha 5d25ff95 unchanged), copied to box `~/r7t_labels/`, laptop == box for the PREREG,
  its sha file and all 16 frozen files (MEASURED md5 diff). A REAL build on the box with the real PREREG passes both guards (sha,
  builder md5 listed); a REAL build is refused without `z|`, without the A1 code freeze, or with a wrong sha.
- **Realistic SIM dry run: PASS end to end** (MEASURED): pre-A1 path reproduced byte for byte; the realistic teacher ties 55.6 %
  of cells at the old clip (real rows 55.5 %) and the A1 builder removes the ties (M(R7) tie share <= .03 %); L0 PASS on the null
  teacher, DROP on a 1-fold leak, STOP (exit 4) on a 2-fold leak; FX2 invariance holds; trainer argv rc 0 with
  `sel_labels == train_labels ==` the table; student gate stand-ins fail exactly the legs they should (E2: G2; DT: G3 + G3').
  The dry run caught one real bug in the new DROP branch (T_a rows that fall back to v4 alone carry hard 0/1 and tripped a
  strict (0, 1) assert); fixed before the freeze.
- **FX7 + FX8 built:** `session_2026-10-15/v224f/build_v224fs.py` (two-group F = .5 mean(K14) + .5 mean(S), fleet start budget
  5.0 h), gate `v224fs_check.py`, S pack stager `make_s_pack.py`. Offline proof with 5 stand-in S members (the k14w DT checkpoints
  relabelled): new cells 15/15 scenarios PASS on the box CPU; the gate on a mock 25-member commit output PASS 32/32 and 7/7
  negative mutations FAIL (incl. "F = equal 25-member mean", the FX7 bug). Not pushed (the S checkpoints do not exist yet).
- **Eesh's "Ok R7" (relayed by main 14:50): both R7-OOF shards pushed 15:28** (`eeshsaxena/rsna-r7-oof` v3 shard 0, 2,248
  studies; `eeshsaxena/rsna-r7-oof-b` v1 shard 1, 2,159). Both COMPLETE (2.7 h / 2.8 h), verify ALL_PASS, merged
  `oof/r7_oof_ALL.csv` 4,407 / 4,407 (per fold [1043, 1146, 1137, 1081] == fold map, 0 missing, all full-precision `z|`).
- **REAL label build (18:21, PREREG A2 sha 9e987880): L0 PASS** (fold AUCs .9348 / .9344 / .9385 / .9313, spread .0071);
  **LABEL_GATE PASS** (gold58 target AUC: frac75 .9300, R7 .9217, **T_b .9338 (+.0038)**, **T_a .9298 (-.0002)**); the old clip
  would have tied 53.4 % of the REAL cells. `~/r7t_labels/final/LABELS_READY` written 18:22 (T_b sha256 7c51bb3a..., T_a
  96520e0f...) for the T_b rental agent. **Box queue 18:23: T_a (seed 1048) ahead of L13 f3/f4, T_b lines #HOLD** (0 BAD; starts
  when L13 f2 ends ~22:00). Copies of the REAL receipt / diag / logs: `labels/real_run/`.
- **Plan changes absorbed:** Amendment A2 (16:05; T_b on the Vast rental, T_a on the box, per-table labels_gate, LABELS_READY);
  V224-FF-S builder on the FLIP base (18:50; sec 2b).

## 1. What changed (md5 before -> after; laptop `r7teacher/labels/` == box `~/r7t_labels/`)

| file | pre-A1 | A1 | change |
|---|---|---|---|
| scripts/build_teacher_labels.py | 0d032723 | a7f8823d7216fb0a55d5fc7821607c78 | FX1/FX2 raw `z|` logits (unclipped) for M(R7), all 12 labels, refuse a REAL file without `z|`, assert p == sigmoid(z) on the 10 non-stacked labels; FX3 L0 gate + DROP / STOP (exit 4); tables in [0, 1]; tie report; A1 code-freeze self-check (REAL) |
| scripts/student_gate.py | a08b8ec1 | 8c0187906013dc611725d2507755e78f | FX4 G0 `sel_labels ==` table; FX2b pool-row ruler `z|`; FX5 G3' ceiling + rho vs fleet OOF reports; VERDICT includes G3'; A1 code-freeze check in G0 (REAL) |
| scripts/sim_r7_oof.py | 922fb205 | 0e61f7cc45bc2827c7b27ccee8a0d7f7 | FX6 realistic |z| (quantile map onto the 44 real rows), `z|` columns, stacker-like p| on Effusion / Baker's, L0 leak controls; plain mode byte-identical to pre-A1 |
| scripts/r7_absz_ref44.json (new) + make_absz_ref44.py (new) | - | 9a7ce040280cd448dcf854a608803cc6 / 20998d21a72f4682b66142ac2bf03297 | per-label sorted |z| of the 44 real hash-fold rows (commit v1 8 + proof v2 36) |
| kits/box/make_sheets.sh | d39ae0a7 | 772f4be0a235db7c6d18911dc75bfbb9 | FX4 `--eval_labels` = the student's own table |
| kits/box/sheets/queue_r7tb.txt | 4f76bea7 | 6196c8e48ff15687f524a4c180a4bda6 | 5 T_b lines (seed 1047), only `--eval_labels` and the name differ |
| kits/box/sheets/queue_r7ta.txt | c36177e6 | 7b95041e616597b5d9b4c73f6b97561a | same for T_a (seed 1048) |
| kits/box/sheets/queue_r7t{b,a}_SIMCHECK.txt | - | 49b6f846845a448191dcad0f9cd68aaf / 300faa2de7412d4a5e46ec828fc85ccf | SIM argv sheets |
| kits/box/on_r7oof.sh | 555f916e | 4d434327aa3ee219946d4bc50f0dba4a | prints the L0 line, stops on exit 4 |
| kits/box/dry_run_student.sh | 575c7e57 | 50e1d57d9d679f641d34b948be2decbc | cores 34-39, prints ckpt args |
| labels_gate.sh, collect_fleet_oof.py, fleet_train.py | 65442ebb, 059f79e4, 3e62434a | unchanged | |
| PREREG_TEACHER.md + .sha256 | 5d25ff95 | 456f38a9... | Amendment A1 appended (text = PLAN_TEACHER sec 7 + the FX details, dry-run results, code md5s, open item) |

New (not frozen, run kits): `kits/box/a1_dryrun.sh` (2a6c8bf0), `kits/box/a1_recheck_builder.sh` (20f0e576),
`kits/box/install_r7tb_top.sh` (a2baa7c1), `kits/box/sheets/queue_preview_r7tb_top.txt` (ccb62d0f), `oof/launch_r7oof.sh`
(a804c5d8), `oof/watch_r7oof.sh` (485c8ead), `oof/after_r7oof.sh` (933f44af). Pre-A1 copies: laptop `labels/pre_a1/`, box
`~/r7t_labels/pre_a1/`; old SIM tables box `~/r7t_labels/sim/tables_pre_a1/`.

V224-F-S (`session_2026-10-15/v224f/`, new; build_v224f.py 4c9b5f82 and v224f_check.py 7f12835d untouched):
`build_v224fs.py` 34501c56, `v224fs_check.py` 057d896f, `make_s_pack.py` 2eaf33e6, `tests/sim_fs_cells.py` d94dace2,
`tests/mock_fs_output.py` 1a24becf; stand-in builds `tests/standin_v224fs20/` (nb a6055779..., no kernel-metadata.json) and
`tests/standin_v224fsr20/` (REPLACE).

## 2. The fixes as implemented

- **FX1/FX2.** M(R7) = expit((z - mean_pool z) / sd_pool z x sd(logit v4) + mean(logit v4)) per label with the raw `z|` fold
  logits; v4 and the fleet keep logit(clip(x, 1e-4)); the ECDF branch (rank-like files) is unchanged.
- **FX3 L0.** AUC of `z|` vs confident v4 cells (<= .1 / >= .9) per R7 hash fold, non-gold pool rows, macro over labels valid in
  all 4 folds; PASS iff spread < .020; outlier = fold farthest from the median of the other three; DROP iff the other three span
  < .020 (that fold's non-gold pool rows lose R7, missing cap 27 %), else STOP. Bootstrap SE per fold reported (~.004 on SIM).
- **FX4.** Student lines: `--eval_labels ~/r7t_labels/real/labels_r7t_b.csv`; the trainer then records `sel_labels` = the table.
- **FX5 G3'.** d = rho(S, R7) - rho_ref <= .5 (1 - rho_ref); part of the VERDICT. G2 floor .10 (1 - rho_ref) unchanged.
- **FX6.** sim_r7_oof.py `--absz_ref r7_absz_ref44.json`.
- **FX7.** V224-F-S cell 22 (HF-S) reads the fleet workers' npz, requires the equal member mean == submission_fleetsolo.csv
  (1e-6), computes F = .5 mean(K14 20) + .5 mean(S 5) (failed studies -> column median, the fleet's rule), writes
  submission_fleet2g.csv, blends (1 - w_f) pct(V224) + w_f pct(F). Cell 18 (S-MERGE) adds the S checkpoints to the fleet dir
  with full sha256 checks; any failure removes the dir -> fleet not clean -> Synovitis canary. K2_CFG MEMBERS = 25 (asserts:
  "exactly 20" became K14 20 + S 5 by name in the HF cell, the receipt and the gate). The gate recomputes the two-group mean
  independently and checks the K14 group mean == the V224-F20 commit fleet solo. `--replace` = F from S alone (5 members).
- **FX8.** V224-F-S is built on build_v224f.py `--start-budget-h 5.0` (MIN_START_MIN 198). MODEL with S costed like the DT
  members (MEASURED microbench 9.63-9.86 s per study per T4 for 20 members, +1.57-1.62 s for the 5 TTR224 ones): 1,300 hidden
  studies -> fleet central 2.08 h, kernel central ~5.1 h, slow end x1.3 ~7.8 h, worst case (start at 5.0 h + slow-session fleet)
  8.11 h < the 8.3 h fleet wall (a cut fleet would show as the Synovitis canary, not a timeout). V224-FF (flip) as a base is NOT
  supported (TODO main: needs its 4.9 h flip budget cut by ~0.5 h + its own runtime review).

## 2b. V224-FF-S (the student entry on the FLIP base; main 18:30: V224-FLIP is the team best)

`python session_2026-10-15/v224f/build_v224fs.py --wf 0.20 --base ff --s-manifest <upload>/r7tb_manifest.json` (md5 fda1e28f):
base = the commit-tested V224-FF20 kernel built by build_v224ff.py (asserted byte-identical to `v224ff/kernel_v224ff20`, sha
b86f8b9f); FX8 patches in place: fleet start budget 5.5 -> 5.0 h (SAVE knobs, K2 MIN_START_MIN 168 -> 198) and time-aware flip
budget 4.9 -> 4.4 h (V224-FF's 0.6 h margin kept); then the same four S changes as V224-F-S (SAVE knob dict, S-MERGE after
K12-MERGE, K2 MEMBERS 25, HF-S two-group F); cells identical to V224-F-S modulo the tag; revert proof gives the V224-FF20 bytes.
Gate `v224f/v224ffs_check.py` (defeda80): v224ff_check with v224fs_check nested (its exact-mode V224 sub-gate fails by design and
is replaced by v224_check --flip), flip budget 4.4, END-TO-END with the two-group F. Mock FF-S output (V224-FF20 commit + 5
stand-in S rows): **V224FFS_CHECK PASS 14/14, 7/7 negatives FAIL** (`python tests/mock_fs_output.py ff`). Runtime (MODEL): the
flip adds ~1.4 h to the V224 part, the 5 S members ~0.28 h to the fleet: central ~6.6 h; the fleet starts only if the kernel is
<= 5.0 h old (else the Synovitis canary), worst start + slow-session fleet 8.1 h < 8.3 h wall. A separate runtime review before
the submit stays required (PLAN_TEACHER FX8). REPLACE: `--replace` (tag v224ffsr20).

## 3. Dry run (box CPU; logs `labels/dryrun_a1/`; MEASURED; describes our own fleet only, never R7)

| step | result |
|---|---|
| R0 A1 builder on the pre-A1 SIM file | tables byte-identical to pre-A1 (8ec090db / 8fbf5787) |
| realistic SIM teacher | 4,404 rows, 55.6 % of pool cells beyond the clip (real 44 rows 55.5 %) |
| pre-A1 builder on it (the bug) | L1 T_a - frac75 -.0042, T_b +.0002 |
| A1 builder on it | L0 PASS (.9345 / .9362 / .9350 / .9337, spread .0025); L1 T_a -.0003, T_b +.0013 -> LABEL_GATE PASS, train T_b; M(R7) ties <= .03 %; tables T_a d2502485, T_b dbfa4a52 |
| FX2 stacker-seed change | both tables byte-identical |
| L0 leak on fold 2 | fold 2 AUC .9727 (+.038) -> DROP, 1,116 rows renormalised (25.8 %), tables written, label gate PASS |
| L0 leak on folds 1 + 2 | STOP, exit 4, no table |
| REAL guards | no z| refused; PREREG without the builder md5 refused; wrong sha refused; good PREREG PASS |
| trainer argv (fleet_train.py 3e62434a, CPU, 31 studies, 1 ep) | T_b rc 0, T_a rc 0; ckpt args fold 1, seed 1047 / 1048, 21 ep, ttr 224, gc False, sel_labels == train_labels == the SIM table |
| student gate, stand-in E2 | G0 P G1 P **G2 F** G3 P G3' P (d -.0026) G4 P -> FAIL |
| student gate, stand-in DT (= teacher) | G0 P G1 P G2 P **G3 F G3' F** (d +.0470 > .0235) G4 P -> FAIL |

## 4. Launch commands (prepared; executed 15:28 on the relayed "Ok R7")

```
cd session_2026-10-15/r7teacher/oof
bash launch_r7oof.sh parallel     # checks notebook shas 358beef2 / 202d67c7, slugs, RUNNING_COUNT 0, then
#   kaggle kernels push -p kernel_r7oof_full_s0of2            (slug eeshsaxena/rsna-r7-oof,   shard 0 of 2)
#   kaggle kernels push -p kernel_r7oof_full_s1of2_parallel   (slug eeshsaxena/rsna-r7-oof-b, shard 1 of 2)
MAX_H=11 bash watch_r7oof.sh rsna-r7-oof:out_full_s0 rsna-r7-oof-b:out_full_s1     # background poll + fetch
bash after_r7oof.sh out_full_s0 out_full_s1     # verify_commit (ALL_PASS) each, merge_sessions -> r7_oof_ALL.csv,
                                                 # z| / precision pre-check, scp to the box, on_r7oof.sh (L0, REAL build, LABEL_GATE, labels_gate)
```
Sequential fallback: `launch_r7oof.sh seq0`, download, then `seq1`. Quota-refusal symptom (ERROR within ~2-3 min, empty log):
re-push with `-t 32400`; if refused again rebuild with `--budget-h 5.5` and push with `-t 21600`, resume later with `--skip-from`.

## 5. Box queue sheet (T_b ahead of L13; installed only after the label gate PASS)

`~/r7t_labels/sheets/queue_r7tb.txt` (md5 6196c8e4): 5 lines `R7Tb S1047 fF ... | ~/fleet_runs/fF_s1047_ttr_r7tb | needs PASS from
~/r7t_labels/real/labels_gate_labels_r7t_b.verdict wait 120 via bash ~/r7t_labels/labels_gate.sh labels_r7t_b.csv | --model ttr
--aug heavy --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs 21 --seed 1047 --save_epochs 6-21 --workers 10
--store ~/fleet_store --labels ~/r7t_labels/real/labels_r7t_b.csv --folds ~/rsna_train/folds5.csv --exclude
~/rsna_train/gold74_eval_set.csv --eval_labels ~/r7t_labels/real/labels_r7t_b.csv --gold ~/rsna_train/gold74_eval_set.csv --fold F
--out ~/fleet_runs/fF_s1047_ttr_r7tb --grad_ckpt 0`. Install: `bash ~/r7t_labels/install_r7tb_top.sh` (refuses unless the T_b
verdict is PASS, labels_gate passes now, the sheet md5 is the A1 one, no s1047 line / out dir exists; then `qctl.sh top` x5 in
order f4..f0 = each an edited copy validated by `queue_runner.sh --check` under the queue lock, never touching the running job;
`--dry` prints the resulting queue). Preview at prep time (`sheets/queue_preview_r7tb_top.txt`, check 8 job lines, 0 BAD):
T_b f0-f4, then L13 f2, f3, f4. Runtime 5 x 3.52 h = ~17.6 h (MEASURED rate).

## 5b. Amendment A2 (16:05, plan change: T_b on a Vast rental, T_a on the box)

PREREG sha256 now `9e987880432b3a316f9c5582f3045c91af2da3a4fd7ad59170e8038951497711` (A1 part = first 22,027 bytes, sha 456f38a9;
pre-A1 part = first 14,577 bytes, sha 5d25ff95). `labels_gate.sh` per-table (6f15f414): PASS iff REAL receipt + PREREG sha +
L0 PASS/DROP + that table passed L1 on its own + md5 unchanged. `kits/box/finalize_labels.sh` (ff4d1ec9): both verdicts, copies
the passing tables + receipt to `~/r7t_labels/final/`, writes `LABELS_READY` (sha256 / md5 per table, PREREG sha, L0, deltas)
last, only if T_b passes (else `LABELS_STOP`). `kits/box/install_r7ta_top.sh` (a921eabb): T_a f0-f4 ahead of L13, T_b f0-f4 as
`#HOLD(...)` comments (`qctl.sh top` only; temp-copy test: 8 job lines, 0 BAD). The rental kit argv is now the A1 sheet line
(`--eval_labels` = the T_b table; argv-check 0-4 IDENTICAL). Tested in an isolated temp HOME on the box (all PASS / FAIL cases
as expected). `install_r7tb_top.sh` (T_b on the box) is superseded unless the rental fails.

## 6. Open items for the main session

1. (closed by A2) T_a and T_b are each trainable on their own L1 pass.
2. V224-F-S needs, in order: student 5/5 folds, `student_gate.py --infer` VERDICT PASS (REAL, A1), `make_s_pack.py` (box) ->
   OAI-id scan -> private dataset (main) -> `build_v224fs.py --wf <team-best w_f> --base ff --s-manifest .../r7tb_manifest.json`
   (FLIP base, sec 2b; `--base f` for the non-flip base) -> commit test -> `v224ffs_check.py` (`v224fs_check.py` for f) ->
   PREREG_10-14 Amendment 4 -> runtime review -> submit. The T_b rental runs only via session_2026-10-15/rental_student/launch.sh
   (kits/vast_r7t is locked); this agent launched no rental. make_s_pack.py is not exercised yet (no
   student); its checks mirror build_k12.py `load_do`.
3. The R7-OOF output must hold full-precision `z|` for every row (the A1 builder refuses NaN logits); `after_r7oof.sh` checks
   this first; rows without them are re-run with `build_r7oof.py --skip-from`.
4. FX9 (honest framing) unchanged: gate passes are not evidence of value; only an API best-id move decides.

Addendum 10-15 00:45 IST (fix agent; details + tests in `session_2026-10-15/v224f/RESULT.md` "Fixes 10-15"). The md5 values
above are pre-fix. Current values:
- build_v224fs.py a54afad6
- make_s_pack.py 5b2df773
- v224fs_check.py b5f5d4e1
- v224ffs_check.py d91afcc3

Section 2b changes:
- `--base` is required.
- `--base ff` keeps the FF20 flip budget 4.9 h (cell 7 not patched) and defaults the fleet start budget to 5.1 h
  (MIN_START_MIN 192).
- The v224ffs_check flip default is 4.9.
- make_s_pack binds the gate file to student / table / all-pass verdict.

No frozen file changed.
