# HFR7-RR fleet-weight variants F30 / F40 (built + commit-tested 10-14 ~07:30-07:45 IST by a Claude subagent). NOT submitted.

Reference = HFR7-RR (`eeshsaxena/rsna-sub-hfr7rr` v1, `session_2026-10-13/hr7rr/kernel_hfr7rr/`, nb sha ca3da2fb...,
ref 57146261, our best public entry .956) = R7 .5 + Raptor_mix .5 (nartaa r384 .65 / r224 .35), then HF cell
(1 - w_f) * that + w_f * F (our ConvNeXt-T fleet, K14 R, 20 members), w_f .20.

## Result in one block
| variant | kernel | version | the ONE change vs HFR7-RR | commit | gates |
|---|---|---|---|---|---|
| **HFR7-RR-F30** | `eeshsaxena/rsna-sub-hfr7rr30` | **v1** | fleet weight w_f .20 -> **.30** (HF cell) | pushed 02:03 UTC, COMPLETE 02:14 UTC, 3 rows, 0 fallbacks, total 632 s | `rr_check --hf-wf 0.3` **RR_CHECK PASS 31/31**; `h_check --kind hf --wf 0.3 --members 20` **HCHECK PASS 16/16** |
| **HFR7-RR-F40** | `eeshsaxena/rsna-sub-hfr7rr40` | **v1** | fleet weight w_f .20 -> **.40** (HF cell) | pushed 02:03 UTC, COMPLETE 02:13 UTC, 3 rows, 0 fallbacks, total 529 s | `rr_check --hf-wf 0.4` **RR_CHECK PASS 31/31**; `h_check --kind hf --wf 0.4 --members 20` **HCHECK PASS 16/16** |

Commit attempts used: F30 1 of 2, F40 1 of 2. Both kernels PRIVATE, GPU T4, internet off. GPU quota: ~0.2 h each.

## How they were built (builders re-run with the changed parameter only, no notebook hand edit)
Regression first: the same two builders with the 10-13 parameters (`--wf 0.2 --tag hfr7r20`, then `--hfr7rr-name hfr7rr`, into
the scratchpad) reproduce `hfr7/build/hfr7r20` (a984ae02...) and `hr7rr/kernel_hfr7rr` (ca3da2fb...) byte for byte, notebook,
kernel-metadata.json and cells_new.txt.

Commands (project root):
```
# F30
python session_2026-10-13/hfr7/build_hfr7.py --wf 0.3 --members R --fleet k14 --tag hfr7r30 --out session_2026-10-14/rr_chain/hfr7r30
python session_2026-10-13/hr7rr/build_hr7rr.py --target hfr7rr --hfr7-nb session_2026-10-14/rr_chain/hfr7r30/rsna_sub_hfr7r30.ipynb \
       --hfr7-sha 2be4bd6487e29c3f4c47df2eb54347deae03a26fa146204a2a944f23c2298d78 --wf 0.3 --hfr7rr-name hfr7rr30 --out session_2026-10-14
# F40
python session_2026-10-13/hfr7/build_hfr7.py --wf 0.4 --members R --fleet k14 --tag hfr7r40 --out session_2026-10-14/rr_chain/hfr7r40
python session_2026-10-13/hr7rr/build_hr7rr.py --target hfr7rr --hfr7-nb session_2026-10-14/rr_chain/hfr7r40/rsna_sub_hfr7r40.ipynb \
       --hfr7-sha fa76a7a7a868ad98daf8441c7ba0ae445127564cdc15f36aee101814e3a3883f --wf 0.4 --hfr7rr-name hfr7rr40 --out session_2026-10-14
```
Notebook sha256: hfr7rr30 **f91dfb3866812ec25e82acf36c52cdb55f35d487ffcb375b4b6cb5642d765cd6**, hfr7rr40
**65cb9c73e17511c35433e14b0646eb98d2d99f8aa35e63d0cb44f8261a5a0b98**. HFR7 bases: hfr7r30 2be4bd64... (== the 01:25
`f30_chain/hfr7r30` build), hfr7r40 fa76a7a7....
- The new `kernel_hfr7rr30/` is byte-identical (notebook + metadata) to `f30_chain/kernel_hfr7rr30/`, the intermediate the RS line
  built at 01:25 and never pushed; the pushed copy is `session_2026-10-14/kernel_hfr7rr30/`.
- `rr_chain/hfr7r30`, `rr_chain/hfr7r40` (ids rsna-sub-hfr7r30 / -hfr7r40) are build steps only: do NOT push them.
- `session_2026-10-14/cells_new.txt` was written by build_hr7rr (`--out`); it is byte-identical to `session_2026-10-13/hr7rr/cells_new.txt`.

## Cell diff vs kernel_hfr7rr (`celldiff.py`, full output `celldiff_hfr7rr30.txt` / `celldiff_hfr7rr40.txt`)
Both: 16 cells, same order, notebook-level metadata and nbformat equal, every non-source cell field equal.
**2 cells differ in each** (0, 15):
```
cell 0 (markdown, title line only)
-# rsna-sub-hfr7r20: ... per-label rank blend (1 - 0.20) H + 0.20 F
+# rsna-sub-hfr7r30: ... per-label rank blend (1 - 0.30) H + 0.30 F        (F40: hfr7r40, 0.40)
cell 15 (HF BLEND)
-W_F = 0.2
-HF_TAG = 'hfr7r20'
+W_F = 0.3                                                                 (F40: 0.4)
+HF_TAG = 'hfr7r30'                                                        (F40: 'hfr7r40')
```
HF_TAG is only a label written into hf_receipt.json and the HF_* log lines (from build_hfr7 `--tag`); no effect on values.
kernel-metadata.json: only `id`, `title`, `code_file` differ (rsna-sub-hfr7rr30 / rsna sub hfr7rr30 / rsna_sub_hfr7rr30.ipynb, and
the same for 40). Same 5 datasets, same docker pin, private, GPU T4, internet off.
Unchanged by construction: R7 stage, both Raptor stages, Raptor_mix .65 / .35 and its fallback, fleet members (K14 R, 20), all caps.

## Commit gates (outputs in `out_<name>_v1/`, gate text in `gate_<name>_v1_rr.txt` / `_h.txt`)
Commands (project root; run WITHOUT `python -I`: numpy is in user site-packages, which -I hides):
```
python session_2026-10-13/hr7rr/rr_check.py session_2026-10-14/out_hfr7rr30_v1 --hf-wf 0.3
python session_2026-10-12/hf/h_check.py session_2026-10-14/out_hfr7rr30_v1 --kind hf --wf 0.3 --members 20
python session_2026-10-13/hr7rr/rr_check.py session_2026-10-14/out_hfr7rr40_v1 --hf-wf 0.4
python session_2026-10-12/hf/h_check.py session_2026-10-14/out_hfr7rr40_v1 --kind hf --wf 0.4 --members 20
```
| | F30 v1 | F40 v1 |
|---|---|---|
| RR_CHECK | PASS 31/31 | PASS 31/31 |
| HCHECK | PASS 16/16 | PASS 16/16 |
| log markers | RAPTOR_STAGE_DONE, RR_STAGE_DONE, RR_MIX_OK (.65/.35), fleet clean 20 members, OURS_STAGE_COMPLETE PASS, HF_BLEND_OK w_f 0.3 | same, HF_BLEND_OK w_f 0.4 |
| receipt | hf_receipt mode HF, w_f 0.3, tag hfr7r30; raptor_mix used, r224_failed 0 | mode HF, w_f 0.4, tag hfr7r40; raptor_mix used, r224_failed 0 |
| negative control (gates run with the HFR7-RR constant `--hf-wf 0.2` / `--wf 0.2`) | RR FAIL 29/31 (w_f receipt + HF recompute max diff .0667), H FAIL 15/16 | RR FAIL 29/31 (max diff .133), H FAIL 15/16 |
| submission sha256 (3-row commit) | 977176b662c2a453... | 23f3d61f467989752... |

Negative-control text: `gate_hfr7rr_negctl.txt`.

What the 3 visible rows show vs the HFR7-RR v1 commit output (`session_2026-10-13/hr7rr/out_hfr7rr_v1/`):
- submission_H.csv (stage-3 output, before the HF cell): 0 of 36 cells differ, for both. submission_fleetsolo.csv and ours_sub.csv:
  0 of 36 differ. Raptor raw dumps differ only by GPU nondeterminism (max 6e-5 r384, 1.2e-4 r224), ranks unchanged.
- submission.csv: 13 of 36 cells differ (F30 max .067, F40 max .133) = the HF weight only; the gates' independent recompute at
  w_f .3 / .4 matches to 3.3e-08. So the change is confined to the HF cell, as intended.

## Runtime
- Commit (3 rows, 2 x T4): F30 632 s (r384 56 s, r224 26 s, R7 350 s); F40 529 s (48 / 21 / 320 s); HFR7-RR v1 was 637 s.
- Hidden: a blend constant is the only change, so the compute is identical to HFR7-RR; the hidden wall of each = HFR7-RR's
  (HR7RR_RESULT estimate ~4.4-5.7 h incl. queue, worst with every cap ~6.3 h; under the 9 h limit).

## Before submitting (main session)
- Pre-registration: HFR7-RR-F30 = PREREG_10-14 Amendment 1 (a) (w_f .30 on the RR base, one change vs our best). **F40 is not
  pre-registered on the RR base**: the 10-14 decision rule ("if F30 beats our best: next probe w_f .40") was written for the RS base.
  Add an amendment before submitting F40 (either after the F30 read, or as a declared F30 + F40 pair).
- Each variant is one change vs HFR7-RR and is read against it by the frozen-team method (`session_2026-10-13/frozen_read.py OLD NEW`).
  If both go in on the same UTC day, take an LB snapshot when the first one scores, else a team-best move cannot be attributed.
- HFR7-RS-F30 (57162815, RS base, submitted 07:08 IST) is a separate read; it does not test the RR base.

## Exact submit commands (NOT run; project root)
```
python session_2026-10-14/submit_at.py eeshsaxena/rsna-sub-hfr7rr30 1 "HFR7-RR-F30: HFR7-RR with fleet w_f .30 (was .20); R7 .5 + Raptor_mix .5 (r384 .65 / r224 .35), F K14 R 20, rsna-sub-hfr7rr30 v1"
python session_2026-10-14/submit_at.py eeshsaxena/rsna-sub-hfr7rr40 1 "HFR7-RR-F40: HFR7-RR with fleet w_f .40 (was .20); R7 .5 + Raptor_mix .5 (r384 .65 / r224 .35), F K14 R 20, rsna-sub-hfr7rr40 v1"
```
submit_at.py prints `OK KERNEL VERSION {..."ref": N}` or `ERR` plus the response body.

## Files (session_2026-10-14/)
`kernel_hfr7rr30/`, `kernel_hfr7rr40/` (pushed, v1 each), `rr_chain/` (HFR7 base intermediates, never push), `cells_new.txt`
(builder side output), `celldiff_hfr7rr30.txt`, `celldiff_hfr7rr40.txt`, `push_hfr7rr30.log`, `push_hfr7rr40.log`, `status_rr.log`,
`out_hfr7rr30_v1/` + `.log`, `out_hfr7rr40_v1/` + `.log`, `gate_hfr7rr30_v1_rr.txt`, `gate_hfr7rr30_v1_h.txt`,
`gate_hfr7rr40_v1_rr.txt`, `gate_hfr7rr40_v1_h.txt`, `gate_hfr7rr_negctl.txt`.
