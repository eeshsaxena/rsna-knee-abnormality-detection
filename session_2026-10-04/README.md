# Session 2026-10-04: saved copies (Temp scratchpad + box)

Copied 10-04 ~18:45 IST so nothing depends on the Temp scratchpad surviving. Only text and code was copied: md, py,
sh, ipynb, json, csv, log, txt, each <= 5 MB. Checkpoints and data caches were NOT copied:
- the shipped fleet member `f0_s42_raw_cnxt320x24_cv8810_best.pt` is on the box `~/fleet_runs/f0_s42/` and in the Kaggle dataset `eeshsaxena/rsna-knee-cnxt-fleet`;
- the K2 kernels are on Kaggle as `eeshsaxena/rsna-sub-k2-fleet` and `rsna-sub-k2-solo` (v1).

Path rule: anything written `SP/<x>` in HANDOFF.md / NEXT_SESSION.md is here as `session_2026-10-04/<x>`.
Box files keep their home-relative path under `box/`.

| Folder | What it is |
|---|---|
| `jump_wf/` | Research workflow outputs (10-04). Key docs: **`box_schedule.md`** (serial rule, queue to 10-22, launch templates) and **`inner_injection.md`** (exact parent graph + fleet-injection verdict ~0). Also kernel/dataset listings `kl_*.csv`, `ds_since0925.csv` (public-intel scout unfinished), parent cells `cells/` |
| `k2/` | K2 pipeline: `RUNBOOK.md`, `build_k2.py` (`--main_mode solo`), `make_k2_kernel.py`, `k2_check.py` (solo-aware), `cells/k2_fleet_cell.py`, `code/` (fleet_common/worker/model), `box/` (BOX_TEST.md, k2_box_stage.py, d1_pick.py), `tests/` (smoke), `analysis/` (gold58_k2.py, gold58_arms.py, gold58_oos_blend.py), built kernels + commit outputs (logs, receipts, submission csvs) under `build*/` |
| `bigjump/` | BIG_JUMP_PLAN.md + builds (TTR recipe, BUILD.md, FIX_LOG.md) |
| `endgame3/` | ENDGAME_V3.md and support |
| `fastpath/` | Laptop half of the GPU fast-path benchmark (bench.py, queue) |
| `box/auto_j1.sh` | The ARMED overnight launcher (copy; the live one is `~/auto_j1.sh`) + `fleet_runs/auto_j1.log`, `gate_history.log` |
| `box/k2_test/analysis_wf/` | LB proxy model (`proxy_*.py/.log/.json`: binormal fit, K2 prior + tables) and injection scripts |
| `box/k2_test/*.py` | gold58 per-arm / OOS-blend / K2 scoring scripts |
| `box/kmimic/runs/gold58_gpu/` | Parent on gold58 (box mimic): final preds + every per-arm/stage csv (`localLB_parts/`) |
| `box/k2_test/runs/C_gold58/working/submission_fleetsolo.csv` | Fleet member on gold58 through the real K2 serve path |
| `box/fast_build/` | Partial fast-path benchmark: memory table (`logs/results.jsonl`, `mem2.jsonl`) + one TTR timing smoke |
