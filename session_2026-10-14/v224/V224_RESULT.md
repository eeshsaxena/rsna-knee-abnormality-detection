# V224-exact / V224-FLIP (built + commit-tested 10-14 by a Claude subagent). NOT submitted.

RESEARCH_960 lever 1 steps 1-4 + lever 2 steps 1-2. No em dashes.

## Result in one block
RESULT_TABLE_PLACEHOLDER

## What was built (`build_v224.py`, run from anywhere: `python session_2026-10-14/v224/build_v224.py --target all`)
Source = `src/varshitha_v224.ipynb` (sha 6406e923...), from `kaggle kernels pull varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956 -m`
on 10-14 11:58 IST. It is byte-identical to the 10-13 scratchpad pull (`fd13/k/...`); her last run is 2026-10-13 08:39 UTC, so
nothing changed since. Her metadata copy: `src/varshitha_kernel-metadata.json`.

- **Her 13 cells are kept verbatim and in her order.** The builder only INSERTS cells, and asserts that deleting the inserted cells
  re-serialises to her notebook FILE byte for byte (her serialisation: compact JSON, ensure_ascii False; checked to reproduce her
  615,181 bytes exactly). Every sha assert of hers is untouched (r384 checkpoint 7e5315da..., r224 checkpoint 3394fdd7...,
  R7 CODE_SHA / SOURCE_SHA incl. compact_infer 8401812c..., R7 bundle 8c7b666e27dc0e8f + per-file sha256).
- Her cell order kept: Raptor r384 (cells 3-4) -> R7 assets with the global `pip install --no-index --no-deps` of timm 1.0.25 /
  pydicom 3.0.1 (cell 6) -> R7 run (cell 7) -> r224 writefile + stage 2b with her 9 h time guard (cells 9-10) -> stage 3 (cell 12,
  Raptor_mix NOT re-ranked).
- kernel-metadata = hers: the same 3 datasets (heliosli/rsna-knee-compact-r7cnxfsss, nartaa/rsna-knee-publication-swa-weights-20261007,
  heliosli/rsna-orthofoundation-fold0-epoch7-20261004), the same docker pin (sha256:2757e0c7...), machine_shape NvidiaTeslaT4
  (2 x T4, as all our kernels), internet off, competition source; changed: id / title / code_file, is_private true; her `id_no` and
  `keywords` dropped. No fleet dataset, no HF cell (USE_F False = the fleet cell is simply absent).

| kernel | slug | cells | inserted cells (everything else = her cells verbatim) | notebook sha256 |
|---|---|---|---|---|
| V224-exact | `eeshsaxena/rsna-sub-v224` | 14 | 13 CANARY (after her stage 3) | 4259ec97741... |
| V224-FLIP | `eeshsaxena/rsna-sub-v224flip` | 15 | 7 FLIP (between her R7 assets cell 6 and R7 run cell 7); 14 CANARY | 73a9d5620d0... |
| control | `eeshsaxena/rsna-sub-v224ctl` | 15 | 10 CONTROL (after her r224 writefile cell 9); 14 CANARY | 4cf96fd5cce... |
| wheel diag | `eeshsaxena/rsna-sub-v224diag` | 5 | own notebook: her cell 9 verbatim + patch/run/compare cells | 30a74a119d4... |

Full inserted text: `celldiff_vs_her.txt`.

- **CANARY** (all three V224 kernels): her only silent fallback is "blend without r224" (R224_SKIPPED by the 9 h guard, R224_TIMEOUT,
  rc != 0, missing csv, R224_UNUSABLE: all leave `r224 = None` in stage 3). Then the canary sets Fracture = 0.5 for every study
  (~ -.03 macro AUC on the display), prints `CANARY_FIRED`, and re-writes submission.csv from her in-memory float32 `sub` (the
  other 11 columns serialise exactly as before). Clean path: prints `CANARY_CLEAN`, file untouched. Writes `canary_receipt.json`
  (r224_used, R224_OK, use_f False, flip_tta_forced, fired). Her other failure paths raise (no submission = visible as an error);
  her per-study fallbacks (R7 or r384 failed on some studies) print BLEND_FALLBACK and are kept as she has them (not canaried:
  they exist in her runs too).
- **FLIP** (V224-FLIP only): asserts compact_infer sha 8401812c (v4c, knows `--flip-tta` and prints FLIP_TTA_READY), asserts the pinned
  R7 manifest has `flip_tta` false and WORKER_ARGS has no `--flip-tta`, then sets `manifest['flip_tta'] = True` in memory and appends
  `--flip-tta` to WORKER_ARGS (exactly the flag her cell 6 adds when the manifest says flip_tta; probe_dual passes worker args
  through unchanged), prints `FLIP_TTA_FORCED`. Offline check (scratchpad `v224sim/flipsim.py`): her cell 6 arg builder on the real
  r7_manifest.json + this cell gives the same 31 args + `--flip-tta`.
- **CONTROL** (v224ctl, never submit): overwrites `/tmp/raptor224_run.py` with a script that exits 3, so her stage 2b prints
  `R224_STAGE rc=3 ok=False`, stage 3 blends without r224, and the canary must blank Fracture (only Fracture).

Offline (before any commit; scratchpad `v224sim/sim.py`): her stage-3 cell + the canary cell executed on HER published commit csvs
reproduce her submission.csv byte for byte (only CRLF differs on Windows); the forced r224 failure gives Fracture 0.5 and the other 11
columns = the no-r224 blend. The gate passed both sims and FAILED (as it should) on: exact gate vs control output (8 checks),
control gate vs exact output (8), flip gate vs a non-flip output (6), a submission with one cell changed by .01 (2).

DIFFS_PLACEHOLDER
