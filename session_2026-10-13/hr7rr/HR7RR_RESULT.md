# HR7-RR and HFR7-RR: dual-Raptor variants (2026-10-13, Claude subagent). NOT submitted.

## Result in one block
| kernel | version | Raptor_mix | commit | gates |
|---|---|---|---|---|
| `eeshsaxena/rsna-sub-hr7rr` (HR7-RR) | **v2** | r384 .65 / r224 .35 | COMPLETE 10:44 UTC, 3 rows, 0 fallbacks | `rr_check.py` **PASS 27/27** (`gate_hr7rr_v2.txt`) |
| `eeshsaxena/rsna-sub-hfr7rr` (HFR7-RR) | **v1** | r384 .65 / r224 .35, F at w_f .2 | COMPLETE 10:44 UTC, 3 rows, 0 fallbacks | `rr_check.py --hf-wf 0.2` **PASS 31/31** (`gate_hfr7rr_v1_rr.txt`); `h_check.py --kind hf --wf 0.2 --members 20` **HCHECK PASS 16/16** (`gate_hfr7rr_v1_h.txt`) |
| `eeshsaxena/rsna-sub-hr7rr` | v1 (superseded) | r384 .5 / r224 .5 (first spec) | COMPLETE 10:28 UTC | PASS 27/27 with `--w384 0.5`; do NOT submit v1 |

Commit attempts used: HR7-RR 2 of 2 (v1 at .5/.5, then v2 at .65/.35 after the coordinator's change); HFR7-RR 1 of 2.

## What changed (cell diff)
Builder: `build_hr7rr.py` (asserts the sha256 of every input; `--target hr7rr|hfr7rr|all`). Base notebooks:
HR7 = `hr7/kernel/rsna_sub_hr7.ipynb` (sha cb2a0f57...), HFR7-R20 = `hfr7/build/hfr7r20/rsna_sub_hfr7r20.ipynb` (sha a984ae02...).

1. **NEW cell 2b** `%%writefile /tmp/raptor224_run.py`, inserted right after the r384 writefile cell. It is nartaa's public
   scored r224 source (`nartaa/rsna-knee-0945-efficient-224crop`, cells 3 + 4, copy in `src/`, sha b80d3b72...) with exactly
   the two edits heliosli made to the r384 source: dump raw probabilities to `raptor224_raw.csv` (float32, pandas shortest
   round-trip repr, 8 significant digits, no rounding) and write `raptor224_sub.csv` instead of `submission.csv`.
   - Why not `/tmp/raptor_run.py` itself: that file asserts the r384 sha twice, builds `coatnet_rmlp_2_rw_384` at native
     320 and runs mirror TTA with a hard (94,3,320,320) shape assert. The r224 checkpoint's scored recipe is different:
     `coatnet_rmlp_2_rw_224`, CROP_FRAC 0.8 of the 384 field then bilinear to 224, one view, no TTA.
   - The decode is identical to stage 1: the builder asserts the same `_FASTREAD_B64`, SLOTS (26/22/18/12/18), IMG 384,
     K_EVAL 94, `study_lut`, CROP_MM 140. Checkpoint sha 3394fdd7... asserted before inference (manifest in `src/`).
2. **NEW cell 3b** (stage 1b runner), right after stage 1: `Popen` in its own session, 45 min cap (`RR_TIMEOUT_S`), the
   process group is killed on a late run, any crash / non-zero exit / missing file -> `RR_FALLBACK` printed, never raises.
   It runs before the R7 assets cell on purpose: that cell pip-installs timm 1.0.25 / pydicom 3.0.1 over the image's
   1.0.29 / 3.0.2, so stage 1b sees the same stock environment as stage 1 (in HFR7-RR it also runs before ENV-SNAP and the
   fleet). Strict state_dict load of the r224 checkpoint on this image (timm 1.0.29) passed.
3. **EDITED stage 3** (HR7 cell 6 = HFR7-R20 cell 12; +43 / -2 lines): `Raptor_mix` = per label, average-tie percentile
   rank of r384 raw and r224 raw (each over its own non-failed rows), mixed .65 / .35, then fed to the UNCHANGED stage-3 rule
   in place of the Raptor raw column (so it is re-ranked, then R7 .5 / Raptor_mix .5, per-class OA weights .5 as before).
   A study r224 failed on keeps its r384 rank. Fallback to r384 alone (= HR7 byte-identical) if: stage 1b not OK, r224 csv
   ids / columns differ, non-finite values, more than 3 % r224-failed studies, or r224 raw vs its own sub Spearman < .99.
   `blend_receipt.json` gains a `raptor_mix` block (used, weights, reason, r224_seconds, r224_failed, Spearman r384 vs r224).
4. Every other cell byte-identical to its base, same order (builder asserts it). In HFR7-RR that includes cell 0 markdown,
   the input check, ENV-SNAP, FN-IDS, K12M, K2 fleet, ENV-RESTORE, R7 assets / stage 2 and the HF cell (so its receipt
   tag still reads `hfr7r20`; cosmetic).
5. kernel-metadata: private, GPU T4, internet off, same docker pin as the base; HR7-RR = HR7's 3 datasets; HFR7-RR =
   hfr7r20's 5 datasets; competition source as before.

Offline checks before any commit (scratchpad `sim_stage3.py`, synthetic csvs, CPU-trivial): mix == independent recompute;
RR_OK False, >3 % r224 failures and a bad r224 csv each give output byte-identical to the HR7 cell; partial failures handled.
`rr_check.py` passed a good fixture and failed a tampered one (both the recompute and the FALLBACK-line checks fired).

## Commit facts (3 visible studies)
- Stage 1 r384: 50-67 s. Stage 1b r224: 23-27 s. R7 stage: ~330 s. Totals: HR7-RR v2 474 s, HFR7-RR v1 637 s (fleet ~190 s).
- R7 output of HR7-RR v1 is bit-identical to HR7's; r384 raw differs from the HR7 commit by up to 6.8e-5 (GPU nondeterminism).
- On these 3 rows the .65 / .35 mix gives the same ranks as r384 alone (0 of 36 stage-3 cells differ from the HR7 rule); a .35
  weight rarely flips an order among 3 rows. The mix path is proven live by v1 (.5 / .5: 10 of 36 cells differ) and by the
  recompute matching the .65 / .35 formula. On the hidden set (~1,200 studies) the mix will move ranks.

## Hidden runtime estimate
- r224 stage: same decode as r384, one view at 224 (about a quarter of the r384 mirrored GPU work). nartaa's .945 submission
  turned around in 7.2 min including queue; the mirrored r384 stage is ~25 min hidden (Danial). Estimate **5-12 min**, upper
  bound ~25 min, hard cap 45 min then RR_FALLBACK. Stage 1b runs before R7, so it uses at most 45 min of R7's NB_T0 + 8.3 h
  deadline budget (R7 ends near 3.3 h, margin over 4 h).
- **HR7-RR:** HR7 / HV hidden ~3.3 h + 0.1-0.2 h = **~3.4-3.5 h**; worst with the cap ~4.05 h; slow session (x1.3) ~5 h.
  Inside the 6 h target.
- **HFR7-RR:** HF-R20 (H + the same 20-member fleet at .2) scored 4.2-5.5 h after submit incl. queue (10-12: pending at
  4 h 10 min, scored by 22:53). HFR7-RR ~ that + 0.1-0.2 h = **~4.4-5.7 h**; worst with the cap ~6.3 h. Internal guards
  unchanged: fleet wall 8.6 h from kernel start, R7 deadline 8.3 h. Read the HFR7-R20 (ref 57137635) hidden wall when it
  scores: if it is above ~5.5 h, HFR7-RR may pass 6 h (still well under 9 h).

## Risks / notes
- If the hidden r224 raw had many exact ties, the Spearman self-check (< .99) would trigger RR_FALLBACK, i.e. an HR7-equal
  submission under the RR name (the log prints RR_FALLBACK; the receipt is not visible after a hidden run). Unlikely: the
  raw dump is float32 and the r384 stage passes the same check.
- Both checkpoints are OAI-trained (nartaa README), same exposure as HR7 / HFR7 (thread 748276 still open).
- Submit HR7-RR **v2** only (v1 is the .5 / .5 build).

## Exact submit commands (main session decides; not run)
```python
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
# HR7-RR
r = api.competition_submit_code(file_name='submission.csv',
        message='HR7-RR: R7 .5 + Raptor_mix .5 (nartaa r384 .65 / r224 .35), rsna-sub-hr7rr v2',
        competition='rsna-knee-abnormality-detection', kernel='eeshsaxena/rsna-sub-hr7rr', kernel_version=2)
# HFR7-RR
r = api.competition_submit_code(file_name='submission.csv',
        message='HFR7-RR: HFR7-R20 with Raptor_mix (r384 .65 / r224 .35), F K14 R 20 at .2, rsna-sub-hfr7rr v1',
        competition='rsna-knee-abnormality-detection', kernel='eeshsaxena/rsna-sub-hfr7rr', kernel_version=1)
print('OK', r)   # on error print e.response.text (pattern: session_2026-10-12/hf/submit_api.py)
```

## Files (session_2026-10-13/hr7rr/)
`build_hr7rr.py` (builder, current), `build_hr7rr_v1_w50.py` + `kernel_v1_w50/` (v1 build), `kernel_hr7rr/` (sha 18cb5a27...),
`kernel_hfr7rr/` (sha ca3da2fb...), `rr_check.py` (gate: `python rr_check.py OUT [--w384 0.65] [--hf-wf 0.2]`),
`cells_new.txt` (new stage 1b + edited stage 3 text), `src/` (nartaa 0945 notebook, README, checkpoint manifest),
`out_v1/`, `out_hr7rr_v2/`, `out_hfr7rr_v1/` (fetched outputs + kernel logs), `gate_*.txt`, `push*.log`, `status*.log`.
Submission sha256: HR7-RR v2 1199c2d7..., HFR7-RR v1 d2e4318f... (3-row commit files).
