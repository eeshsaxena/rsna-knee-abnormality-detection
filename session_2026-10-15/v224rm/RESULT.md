# V224-RM: r224 anatomical mirror TTA (RESEARCH_960 lever 4). Built + commit-tested 10-14 by a Claude subagent. NOT submitted.

No em dashes. Tags: MEASURED / MODEL / GUESS / CLAIMED as in RESEARCH_960.

## Result in one block
| kernel | version | composition | commit | gates |
|---|---|---|---|---|
| `eeshsaxena/rsna-sub-v224rm` | **v1** | **V224-F20-RM** = V224-F20 (sibling `eeshsaxena/rsna-sub-v224f20` v1, base notebook byte-identical, sha 8f8b38e4) + the RM change (2 inserted cells) | pushed 08:08:23 UTC, COMPLETE 08:19:41 UTC, 3 rows, 0 fallbacks, kernel 631 s | **V224RM_CHECK PASS 15/15** (`gate_v224rm_v1.txt`), which includes the base gate **V224F_CHECK PASS 27/27** (itself `v224_check.py` 17/17 on the V224 view + `h_check.py` HCHECK 16/16) |

- w_f = .20 because the 10-14 reads set it for every fleet build (PREREG_10-14 "READS 12:46-13:30": RR-F30 and RR-F40 did not beat HFR7-RR).
- One change vs V224-F20: the sibling's V224-F builder is RUN into `base/` and its output is asserted byte-identical to
  `v224f/kernel_v224f20/rsna_sub_v224f20.ipynb` (the notebook of `rsna-sub-v224f20` v1, the kernel the sibling commit-tested).
  All 22 base cells are kept byte-identical and in order; the RM change inserts cell 10 (RM) and appends cell 23 (RM_GUARD).
- 27/27 rather than the sibling's 28/28: the dropped check is `--ref-v224` ("V224 part == the V224-exact commit"), which RM
  changes by design (the mirror moved 4 of 36 r224 rank cells on the 3 commit rows).

## The permutation map (verified against /tmp/raptor_run.py, in the kernel and locally)
- **Identity.** It is DERIVED, not assumed: `rm_verify.py` takes `infer_pair_probs` from the r384 script that her cell 3 wrote
  to `/tmp/raptor_run.py` and from the patched r224 script, calls both on stub passes (p = .01 + k/100, q = .5 + k/100) and solves
  for the class index each output came from. Both give [0..11] (`R224_MIRROR_VERIFIED`, commit log t = 450 s). A swapped
  Medial / Lateral map is a negative control that the verify rejects.
- **Combine: 0.5 * (p + q) in probability space**, same as r384 (derived with p = .9, q = .6: probability mean .75, not the logit
  mean .786). A logit-space combine is a negative control that fails.
- **Why identity is right for Raptor (and why R7's swap is not copied):** the r384 serving code re-indexes nothing. Raptor's
  preprocessing keeps the native orientation (slices sorted along the patient normal, DICOM pixel order, no side model), so a true
  L/R mirror turns a left knee into an anatomically normal right knee and every Medial / Lateral label stays what it was. R7
  (compact_infer v4c) is different: it re-renders studies into one canonical side with its orientation classifier, so in R7's
  frame a mirror moves the medial compartment to where lateral usually is, hence its Medial/Lateral Meniscus + OA swap and MCL
  from the plain pass. MEASURED in the source: `MIRROR_PAIRS` in compact_infer v4c vs `return 0.5 * (p + q)` in r384.
- **Transform:** r384's `_mirror_tri` byte-identical (sagittal slots 0, 1: reverse the 3 slice channels; coronal / axial: flip
  the width), applied to the uint8 triplets BEFORE the 0.8 crop, the resize to 224 and the normalisation (r384 mirrors before its
  320 crop). `SLOT_SIZES`, `SAGITTAL_SLOTS`, `_slot_index`, `_mirror_tri`, `LAB` byte-identical to r384 (asserted in the kernel);
  `_mirror_tri` functionally equal on all 94 window centres and an involution.

## What the commit measured (MEASURED, 3 visible studies)
| quantity | value |
|---|---|
| r224 raw, max / mean abs(plain - mirror) | .126 (Baker's) / .022; all 12 labels move by > 1e-4 (per label .016 to .126) |
| r224 rank cells changed by the mirror (3 rows) | 4 of 36 (ACL, MCL) |
| final submission vs the V224-F20 commit | 2 of 36 cells (ACL, max .133) |
| raw avg == float32 0.5 (plain + mirror) | exact (max diff 0) |
| raptor224_sub == her rankpct(avg raw) | exact (max diff 0) |
| plain pass vs her r224 raw (v224diag commit, same R7-wheel env) | max diff 3.9e-5 (GPU run-to-run floor 2.4e-4): her plain path is untouched |
| untouched stages vs the V224-F20 commit | R7 ours_sub diff 0; r384 raw 6.2e-5; fleet solo vs HFR7-RR commit 0 |
| V224 recompute (stage 3, no re-rank) / HF recompute | 3.3e-8 / 3.3e-8 (base gate) |
| canary receipt | r224 used, fired [], RM guard mirror_ok, rm_on, w_f .2; `RM_GUARD_CLEAN`, `V224RM_DONE` |

## What was built
`build_v224rm.py --wf W [--flip] [--base NB] [--slug ...]` (run from anywhere). The RM change is identical on every base:
- **cell RM** (after her r224 `%%writefile`, before her stage 2b): asserts `/tmp/raptor224_run.py` == her cell 9 (sha afb1103e)
  and `/tmp/raptor_run.py` == her cell 3 (sha 663e33fc); applies 6 literal patches (each anchor exactly once; `r224_patch.diff`):
  1. r384's `SLOT_SIZES / SAGITTAL_SLOTS / _slot_index / _mirror_tri` (verbatim) + r384's `infer_pair_probs` (only the shape
     assert 320 -> 224; it also keeps the two passes per consumer thread for the raw dump);
  2. `gpu_windows(..., mirror=False)` (r384's signature); 3. r384's two mirror lines right after the uint8 triplet gather;
  4. arrays for the plain / mirrored passes; 5. r384's consumer lines (plain + mirrored windows -> `infer_pair_probs`, XLA ->
     per-study failure exactly as r384); 6. raw dumps (`raptor224_raw.csv` = what stage 3 ranks, `_plain`, `_mirror`) +
     `raptor224_tta_receipt.json`; more than 3 % failed studies -> exit 4.
  Then `rm_verify.py` (subprocess, CPU) and the install of the patched script (sha 90b2a4eb asserted), `R224_MIRROR_PATCHED`.
  Any mismatch raises (no silent plain r224 under the RM name).
- **cell RM_GUARD** (last): see the canary map.
- Full inserted text: `celldiff_v224rm.txt`; build receipt: `build_receipt_v224rm.json`; review copies `raptor224_her.py`,
  `raptor224_rm.py`, `rm_verify.py`. Every original sha assert is untouched (her checkpoint shas 7e5315da / 3394fdd7, R7
  CODE_SHA / SOURCE_SHA / bundle, the sibling's fleet asserts).

## Canary map (every fallback is a visible column or an explicit log)
| path | what happens | visible as |
|---|---|---|
| RM cell check fails (sha, patch anchor, verify) | raises, kernel stops | submission error (no score) |
| r224-M script: > 3 % failed studies (incl. an XLA device) | exit 4 -> her stage 2b ok=False -> her stage 3 without r224 | V224 CANARY: Fracture 0.5 (the sibling HF cell keeps it blank) |
| r224 skipped by her 9 h guard / timeout / rc != 0 / unusable csv | her path, unchanged | Fracture 0.5 |
| <= 3 % failed studies | her rule: those rows stay 0.5 | logged (tta receipt `n_fail`, canary_receipt `rm_guard.n_fail`) |
| r224 used, but mirror receipt missing / wrong perm / r224 csv not the ranks of the mirror-averaged raw | RM_GUARD | Fracture 0.5 (`RM_GUARD_FIRED`) |
| fleet not clean / HF problem | sibling V224-F cells, unchanged | Synovitis 0.5 |

## Checks before the commit (offline, CPU)
- `rm_verify.py` locally on her two scripts: PASS; negative controls (swapped M/L map, logit combine, swapped mirror axes) all FAIL.
- `sim_rm.py` **SIM_RM PASS 22/22** (`sim_rm.txt`): S1 the RM cell itself (paths -> temp dir) incl. a 1-byte-changed r224 script
  that must raise; S2 her r224 script and the patched one executed end to end (`main()`) on CPU with a stand-in model / reader:
  patched plain pass == her probabilities exactly, avg == 0.5 (p + q) exactly, sub == rankpct(avg), mirrored windows == mirror
  (uint8) -> crop -> resize -> normalise, > 3 % failures -> exit 4 with no sub csv, <= 3 % -> written with the failed row 0.5;
  S3 RM_GUARD fixtures (clean -> bytes unchanged; missing receipt / wrong perm / raw not matching -> Fracture 0.5 only, the
  other 11 columns byte-equal; r224 unused -> no-op).
- Gate negative test: `v224rm_check.py` on the V224-F20 commit output (no RM) -> FAIL 3/9 (`gate_negative_on_base.txt`).

## Runtime
- Commit (MEASURED): r224 stage 26 s (V224-F20: 23 s). Warm pass .485 s per study per GPU (every mirrored pass; the first plain
  pass per GPU is 8.4-8.9 s of cudnn autotune, the same in the plain run). Kernel total 631 s (~11 min of GPU quota, 1 commit).
- Hidden r224 stage (MODEL, `rm_runtime.py` -> `runtime_v224rm_v1.txt`; N 1,200-1,300, 2 T4, her 4-proc decode <= ~7 min):
  plain ~7-10 min -> RM ~11-20 min. The GPU part doubles (x2); the stage wall grows x1.6-2 because the decode overlaps.
  **Extra +4 to +10 min, central ~+0.1 h** (pass time from the commit .485 s or FLOP-scaled from the MEASURED r384 rate .62 s;
  slow session x1.3 included in the upper end).
- **V224-F20-RM hidden wall: central ~4.9 h** (sibling V224-F20 model 4.8 h + .1), range ~4.2-5.9 h, slow session x1.3 on the
  slow end ~7.6 h (< fleet wall 8.3 h < 9 h).
- Guards: RM runs after R7, so R7's 8.3 h deadline and the 7.83 h r224 skip point are unaffected; the r224 subprocess timeout is
  min(left, 2.5 h) >> 20 min; the sibling fleet starts only if the kernel is <= 5.5 h old: fleet start central ~3.1 h, slow end
  ~3.8 h, x1.3 ~4.9 h. P(a guard trips because of RM) < 1 % (GUESS).

## Submit (the main session decides; NOT run here)
Pre-register first (one change vs V224-F20; per RESEARCH_960 lever 4 it enters pick B only if its entry becomes team best).
```
python session_2026-10-14/submit_at.py eeshsaxena/rsna-sub-v224rm 1 "V224-F20-RM: V224-F20 + r224 anatomical mirror TTA (r384 _mirror_tri, identity class map, probability mean); rsna-sub-v224rm v1"
```
Read it against the V224-F20 entry (same base bytes otherwise). Submit >= 45 min apart from other entries.

## One-flag rebuilds (same gates; each needs its own commit before a submit)
- other w_f: `python session_2026-10-15/v224rm/build_v224rm.py --wf 0.25` (the sibling builder makes V224-F25 into `base/`,
  asserted == `v224f/kernel_v224f25` if the sibling built it), push with `push_when_free2.sh kernel_v224rm push.log` (new version
  of the same slug), gate:
  `python v224rm_check.py OUT --wf 0.25 --ref-base <V224-F25 commit out> --base-gate-args "--ref-fleet session_2026-10-13/hr7rr/out_hfr7rr_v1"`.
- pure V224-RM on V224-exact (no fleet): `--wf 0`; gate `--wf 0 --ref-base session_2026-10-14/v224/out_v224_v1` (base gate =
  `v224_check.py` exact; do not pass `--her`, the r224 ranks change). NOT commit-tested.
- with flip: `--wf 0 --flip` (V224-FLIP base) or `--wf 0.20 --flip` (sibling V224-FF20, time-aware flip; base asserted ==
  `v224ff/kernel_v224ff20`); the FF gate's "submission_V224.csv == V224-FLIP commit" check differs by design (r224 changes).
  Both build offline (checked); NOT commit-tested.
- any other V224-family notebook: `--base NB.ipynb` (her 13 cells must be present verbatim; no later cell may re-bind what
  RM_GUARD reads; asserted).

## Concerns
1. Expected gain is tiny: public 0 to +1e-4 (MODEL, RESEARCH_960 lever 4), below the paired read SD; the read is near a coin flip.
2. The mirror moves r224 probabilities a lot (mean .022, max .126 on 3 rows). Raptor was trained without flips (CLAIMED); the same
   mirror is served with r384 in every public .954-.956 recipe (MEASURED) and was +.001 on r384 (CLAIMED, Danial), but it is
   unvalidated for the r224 0.8-crop checkpoint (no offline labels used, per RESEARCH_960 D3).
3. v1 was commit-tested at w_f .20 (on the sibling V224-F20 base, per the w_f rule). The pure V224-RM (`--wf 0`) and the flip
   stacks build and pass the offline checks but have no commit of their own.
4. The 0.8 crop of 384 px is 307 px at offset 38 (39 on the far side), so the mirrored view is a mirror shifted by 1 px (r384's
   320 crop is symmetric). Harmless, noted.
5. With the time-aware flip base (V224-FF), its flip budget 4.9 h leaves .6 h for r224 + fleet pre-cells before the 5.5 h fleet
   start guard; RM uses up to ~.33 h of it in a slow session (fits, tighter).
6. The slug is shared by every RM variant; track version -> composition (v1 = V224-F20-RM).
7. The r224 rank stage still treats <= 3 % per-study failures as her script does (rows at 0.5, logged, not a canary column).

## Files
`build_v224rm.py`, `v224rm_check.py`, `sim_rm.py`, `rm_runtime.py`, `push_when_free2.sh`, `kernel_v224rm/` (pushed notebook +
metadata), `base/` (sibling base builds), `out_v224rm_v1/` (commit output), `gate_v224rm_v1.txt`, `gate_negative_on_base.txt`,
`sim_rm.txt`, `runtime_v224rm_v1.txt`, `celldiff_v224rm.txt`, `r224_patch.diff`, `build_receipt_v224rm.json`, `push_v224rm.log`,
`status_v224rm.log`.
