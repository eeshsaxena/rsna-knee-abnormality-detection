# HFR7-RS one-change variants for the 10-14 slots (built 10-14 ~01:25 IST by a Claude subagent). NOT submitted.

Reference = HFR7-RS (`eeshsaxena/rsna-sub-hfr7rs` v1, `session_2026-10-13/raptor/kaggle/kernel_hfr7rs/`, nb sha f374cc48...,
submitted 10-13 19:53 UTC as ref 57154866, pending at build time).

## Result in one block
| variant | kernel | version | the ONE change vs HFR7-RS | commit | gates |
|---|---|---|---|---|---|
| **HFR7-RS-F30** | `eeshsaxena/rsna-sub-hfr7rs30` | **v1** | fleet weight w_f .20 -> **.30** (HF cell) | COMPLETE 20:11 UTC (pushed 19:58), 3 rows, 0 fallbacks, total 664 s | `rs_check --hf-wf 0.3` **RS_CHECK PASS 38/38**; `h_check --kind hf --wf 0.3 --members 20` **HCHECK PASS 16/16** |
| **HFR7-RS-O45** | `eeshsaxena/rsna-sub-hfr7rso45` | **v1** | Raptor_mix r384 .45 / r224 .25 / ours .30 -> **.35 / .20 / .45** (stage 3); w_f .20 | COMPLETE 20:10 UTC (pushed 19:58), 3 rows, 0 fallbacks, total 583 s | `rs_check --w3 0.35,0.20,0.45 --hf-wf 0.2` **RS_CHECK PASS 38/38**; `h_check --kind hf --wf 0.2 --members 20` **HCHECK PASS 16/16** |

Commit attempts used: F30 1 of 2, O45 1 of 2.

## How they were built (builders re-run, no notebook hand edit)
The two downstream builders (and the sim) got optional arguments only; with default arguments they reproduce every 10-13 file byte for byte
(checked: hfr7r20 a984ae02, kernel_hr7rr 18cb5a27, kernel_hfr7rr ca3da2fb, cells_new.txt e27ffca9, kernel_hr7rs eec9e887,
kernel_hfr7rs f374cc48, and all four kernel-metadata.json files). Pre-edit copies: `builders_pre_edit/` (build_hr7rr.py da7bd2c4, build_hr7rs.py d9eaeb90, sim_stage3_rs.py f00a0467).
- `hfr7/build_hfr7.py`: unchanged (already had `--wf`).
- `hr7rr/build_hr7rr.py`: new `--hfr7-nb/--hfr7-sha` (alternative HFR7 base, full sha asserted), `--wf` (W_F the base HF cell
  must carry, default .2), `--hfr7rr-name`, `--out`.
- `raptor/kaggle/build_hr7rs.py`: new `--w3 r384,r224,ours`, `--name` (slug / dir / code file), `--base-nb/--base-sha`.
- `raptor/kaggle/sim_stage3_rs.py`: optional 3rd argument = the weights the recompute uses (default .45,.25,.30).

Commands (from the project root):
```
# V1 F30: HF base at w_f .3 -> + r224 stage -> + our seeds (W3 unchanged)
python session_2026-10-13/hfr7/build_hfr7.py --wf 0.3 --members R --fleet k14 --tag hfr7r30 --out session_2026-10-14/f30_chain/hfr7r30
python session_2026-10-13/hr7rr/build_hr7rr.py --target hfr7rr --hfr7-nb session_2026-10-14/f30_chain/hfr7r30/rsna_sub_hfr7r30.ipynb \
       --hfr7-sha 2be4bd6487e29c3f4c47df2eb54347deae03a26fa146204a2a944f23c2298d78 --wf 0.3 --hfr7rr-name hfr7rr30 --out session_2026-10-14/f30_chain
python session_2026-10-13/raptor/kaggle/build_hr7rs.py --seeds session_2026-10-13/raptor/kaggle/seeds_k2.json --target hfr7rs \
       --base-nb session_2026-10-14/f30_chain/kernel_hfr7rr30/rsna_sub_hfr7rr30.ipynb \
       --base-sha f91dfb3866812ec25e82acf36c52cdb55f35d487ffcb375b4b6cb5642d765cd6 --name hfr7rs30 --out session_2026-10-14
# V2 O45: same base as HFR7-RS (kernel_hfr7rr, ca3da2fb), only W3 changed
python session_2026-10-13/raptor/kaggle/build_hr7rs.py --seeds session_2026-10-13/raptor/kaggle/seeds_k2.json --target hfr7rs \
       --w3 0.35,0.20,0.45 --name hfr7rso45 --out session_2026-10-14
```
Notebook sha256: hfr7rs30 c903e90c..., hfr7rso45 29710fec... The `f30_chain/` intermediates (ids rsna-sub-hfr7r30 /
rsna-sub-hfr7rr30) are build steps only: do NOT push them.

## Cell diff vs kernel_hfr7rs (`celldiff.py`, full output `celldiff_<name>.txt`)
Both: 18 cells, same order, notebook-level metadata equal, every non-source cell field equal.

**F30: 2 cells differ** (0, 17):
```
cell 0 (markdown, title line only)
-# rsna-sub-hfr7r20: ... per-label rank blend (1 - 0.20) H + 0.20 F
+# rsna-sub-hfr7r30: ... per-label rank blend (1 - 0.30) H + 0.30 F
cell 17 (HF BLEND)
-W_F = 0.2
-HF_TAG = 'hfr7r20'
+W_F = 0.3
+HF_TAG = 'hfr7r30'
```
HF_TAG is only a label written into hf_receipt.json and the HF_* log lines (comes from build_hfr7 `--tag`); no effect on values.

**O45: 1 cell differs** (16, stage 3):
```
-W3 = {'r384': 0.45, 'r224': 0.25, 'ours': 0.3}    # [rs] fixed in PREREG_RAPTOR_KERNELS.md before any public read
+W3 = {'r384': 0.35, 'r224': 0.2, 'ours': 0.45}    # [rs] fixed in PREREG_RAPTOR_KERNELS.md before any public read
```
The trailing comment is the builder template (left as is to keep the diff to the constant); these O45 weights are NOT in
PREREG_RAPTOR_KERNELS, so pre-register them (see "Before submitting").

kernel-metadata.json: only `id`, `title`, `code_file` differ (private, GPU T4, internet off, same docker pin, same 6 datasets).
Unchanged by construction: the RR fallback mix (.65 / .35 if stage 1c fails), all stage caps, fleet members, R7 stage.

## Stage-3 simulation (`sim_stage3_rs.py`, synthetic csvs, `sim_stage3_summary.txt`)
| run | A (RS OK == recompute) | B (RS failed == parent bytes) | C (bad raw == parent bytes) | D (r224 failed == recompute) | verdict |
|---|---|---|---|---|---|
| F30 (weights .45/.25/.30, parent = f30_chain/kernel_hfr7rr30) | 2.9e-08 | True | True | 2.9e-08 | PASS |
| O45 (weights .35/.20/.45, parent = kernel_hfr7rr) | 2.9e-08 | True | True | 2.9e-08 | PASS |
| negative control: O45 notebook vs OLD weights | 5.1e-02 | True | True | 5.1e-02 | FAIL (as it must) |
| regression: HFR7-RS v1 vs default weights | 2.9e-08 | True | True | 2.9e-08 | PASS |

## Commit gates (outputs in `out_<name>_v1/`, gate text in `gate_<name>_v1_rs.txt` / `_h.txt`)
Commands (project root; run WITHOUT `python -I`: numpy is in user site-packages, which -I hides):
```
python session_2026-10-13/raptor/kaggle/rs_check.py session_2026-10-14/out_hfr7rs30_v1 --seeds session_2026-10-13/raptor/kaggle/seeds_k2.json --hf-wf 0.3
python session_2026-10-12/hf/h_check.py session_2026-10-14/out_hfr7rs30_v1 --kind hf --wf 0.3 --members 20
python session_2026-10-13/raptor/kaggle/rs_check.py session_2026-10-14/out_hfr7rso45_v1 --seeds session_2026-10-13/raptor/kaggle/seeds_k2.json --w3 0.35,0.20,0.45 --hf-wf 0.2
python session_2026-10-12/hf/h_check.py session_2026-10-14/out_hfr7rso45_v1 --kind hf --wf 0.2 --members 20
```
| | F30 v1 | O45 v1 |
|---|---|---|
| RS_CHECK | PASS 38/38 | PASS 38/38 |
| HCHECK | PASS 16/16 | PASS 16/16 |
| log markers | RAPTOR_STAGE_DONE, RR_STAGE_DONE, RS_STAGE_DONE, RR_MIX_OK (.65/.35), RAPTOR_SEEDS_MIX_OK (.45/.25/.30), fleet clean 20 members, HF_BLEND_OK w_f 0.3 | same, RAPTOR_SEEDS_MIX_OK **{r384 .35, r224 .2, ours .45}**, HF_BLEND_OK w_f 0.2 |
| receipt | hf_receipt mode HF, w_f 0.3, tag hfr7r30 | raptor_seeds.weights {.35, .2, .45}, used, ours_failed 0, with_r224 |
| negative control (gate run with the HFR7-RS constants) | `--hf-wf 0.2` -> FAIL 36/38 (w_f receipt + HF recompute max diff .0667) | default `--w3` -> FAIL 37/38 (receipt weights only, see note) |
| submission sha256 (3-row commit) | 13c6981dac88c3d7... | 4081bbc500e3038a... (== HFR7-RS v1, see note) |

What the 3 visible rows show vs the HFR7-RS v1 commit output:
- F30: submission_H.csv (stage-3 output) has 0 of 36 cells different from HFR7-RS (raw dumps differ only by GPU
  nondeterminism, max 1e-4, ranks unchanged); submission.csv differs in 15 of 36 cells (max .067) = the HF weight, and the
  gate's independent recompute at w_f .3 matches. So the change is confined to the HF cell, as intended.
- O45: on these 3 rows the .35/.20/.45 mix orders every label exactly like .45/.25/.30 (mix values move by up to .05, 0
  order flips), so its submission is byte-identical to HFR7-RS v1. Same situation as HR7-RR v2 (.65/.35) on 10-13. The O45
  path is proven live by RAPTOR_SEEDS_MIX_OK printing the new weights, the receipt, and the 40-row sim (A and D match the
  .35/.20/.45 recompute to 3e-08; the old-weight recompute is off by .051). On the ~1,200 hidden studies the weights will move
  ranks. That is also why the O45 negative control fails only the receipt check: the recompute cannot tell the two weightings
  apart on 3 rows.

## Runtime
- Commit (3 rows, 2 x T4): F30 664 s (r384 65 s, r224 25 s, seeds 42 s, R7 340 s); O45 583 s (47 / 22 / 37 / 330 s);
  HFR7-RS v1 was 667 s. GPU quota used: ~0.2 h each (one commit each).
- Hidden: both changes are blend constants only (HF cell / stage-3 mix), so the compute is identical to HFR7-RS; the hidden wall
  of each = HFR7-RS's. HFR7-RS (ref 57154866, submitted 19:53 UTC 10-13) was still PENDING at 20:00 UTC, so its real wall is
  not known yet. Estimate from the 10-13 docs: HFR7-RR ~4.4-5.7 h incl. queue + stage 1c (2 seed arms, <= 45 min by
  PREREG_RAPTOR_KERNELS Amendment 1, 60 min hard cap) = ~5.1-6.5 h; worst case with every cap hit ~7.3 h, under the 9 h limit.
  Read the HFR7-RS scoring time first: if it took > ~7 h, do not stack both variants on a slow day.

## Before submitting (main session)
- Pre-register first (HANDOFF ideas (3) fleet weight .3 and (4) Raptor-slot weights): neither F30's w_f .3 nor O45's
  .35/.20/.45 is in PREREG_10-13 or PREREG_RAPTOR_KERNELS (the W3 line's inherited comment says "fixed in
  PREREG_RAPTOR_KERNELS"; that is the template text, not true for O45).
- Read HFR7-RS (frozen-team count vs the 19:34 snapshot) before choosing. Each variant is one change vs HFR7-RS and is read
  against it. If both go in on the same UTC day, a team-best move cannot be attributed to one of them unless an LB snapshot
  is taken between their two scoring times (take a snapshot when the first one scores).

## Exact submit commands (NOT run)
```python
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
# V1 HFR7-RS-F30
r = api.competition_submit_code(file_name='submission.csv',
        message='HFR7-RS-F30: HFR7-RS with fleet w_f .30 (was .20); R7 .5 + Raptor_mix .5 (r384 .45/r224 .25/ours .30), rsna-sub-hfr7rs30 v1',
        competition='rsna-knee-abnormality-detection', kernel='eeshsaxena/rsna-sub-hfr7rs30', kernel_version=1)
# V2 HFR7-RS-O45
r = api.competition_submit_code(file_name='submission.csv',
        message='HFR7-RS-O45: HFR7-RS with Raptor_mix r384 .35/r224 .20/ours .45 (was .45/.25/.30), fleet .2, rsna-sub-hfr7rso45 v1',
        competition='rsna-knee-abnormality-detection', kernel='eeshsaxena/rsna-sub-hfr7rso45', kernel_version=1)
print('OK', r)   # on error print e.response.text (pattern: session_2026-10-12/hf/submit_api.py)
```

## Files (session_2026-10-14/)
`kernel_hfr7rs30/`, `kernel_hfr7rso45/` (pushed, v1 each), `f30_chain/` (build intermediates, never push), `celldiff.py` +
`celldiff_<name>.txt`, `sim_stage3.txt` (full) + `sim_stage3_summary.txt`, `push_<name>.log`, `status.log`,
`out_<name>_v1/` + `.log`, `gate_<name>_v1_rs.txt`, `gate_<name>_v1_h.txt`, `builders_pre_edit/`.
