# V224-FLIP-RM (eeshsaxena/rsna-sub-v224flrm): built + commit-tested 10-14 by a Claude subagent. NOT submitted.

No em dashes. PREREG_10-15 Amendment 1, slot 2 (~06:25 IST 10-15): V224-FLIP + r224 mirror TTA, no fleet; submit only if the gate passes.

## Result in one block
| kernel | version | composition | commit | gate |
|---|---|---|---|---|
| `eeshsaxena/rsna-sub-v224flrm` | **v1** | **V224-FLIP-RM** = the submitted V224-FLIP notebook (byte-identical base cells) + ONLY the 2 r224 mirror inserts (RM, RM_GUARD; byte-identical to the reviewed V224-F20-RM cells). No fleet. | pushed 12:56:53 UTC (both R7-OOF shards COMPLETE, 0 of our GPU sessions running), COMPLETE by 13:04:47 UTC, 3 rows, 0 fallbacks, kernel log end 432 s | **V224FLRM_CHECK PASS 10/10** (`gate_v224flrm_v1.txt`) |

- `FLIPRM_READY.txt` written (slug, version 1, gate line). Standalone recompute `recompute_flrm.py` -> RECOMPUTE_FLRM PASS
  (`recompute_v224flrm_v1.txt`). The commit submission equals the offline prediction `sim_flrm_expected_submission.csv` (0 of 36
  cells differ).
- vs the V224-FLIP commit: 4 of 36 cells differ, only ACL and MCL (max .0583), exactly the 2 columns where the mirror moved the
  r224 ranks; R7 ours_sub identical (max diff 0), r384 raw 6.9e-5 (GPU floor), every other column identical.
- Hidden wall (estimate): **~4.7 h submit-to-score central** (V224-FLIP MEASURED ~4.6 h incl. queue on 10-14, + RM ~+0.06-0.16 h).

## What was built
`python session_2026-10-15/v224rm/build_v224rm.py --wf 0 --flip --slug rsna-sub-v224flrm` (the reviewed builder's existing
`--wf 0 --flip` path; no builder change). Base = `session_2026-10-14/v224/kernel_v224flip/rsna_sub_v224flip.ipynb` (sha 73a9d562,
asserted by the builder) = the V224-FLIP notebook that was pushed as `rsna-sub-v224flip` v1 and submitted (ref 57174021, team best).
Output: `kernel_v224flrm/` (notebook sha f14f6acc..., 17 cells; kernel-metadata), `build_receipt_v224flrm.json`, `celldiff_v224flrm.txt`.
The review copies `raptor224_rm.py` (90b2a4eb) and `rm_verify.py` (8ec218fd) were rewritten byte-identical.

## Cell-diff proof (`celldiff_v224flrm_proof.py` -> `celldiff_proof_v224flrm.txt`: FLRM_CELLDIFF PASS 13/13)
Independent of the builder (imports nothing from it):
- base file sha == 73a9d562; base cells == a fresh `kaggle kernels pull eeshsaxena/rsna-sub-v224flip` (all 15 cells, type + source).
- alignment base -> new: equal [0:11], INSERT new[11] (RM, after her r224 `%%writefile`), equal [11:15] -> [12:16], INSERT new[16]
  (RM_GUARD, last, after the V224 CANARY). No replace / delete.
- new notebook minus cells 11 and 16 re-serialises to the base FILE byte for byte (cell metadata, outputs, notebook metadata).
- new[11] / new[16] == the reviewed + commit-tested `kernel_v224rm` cells 10 / 23 as whole cell dicts (sha d7a684f1 / 7361cc89);
  the RM cell carries her r224 sha afb1103e, her r384 sha 663e33fc, patched-script sha 90b2a4eb.
- every base sha / flip / canary needle (r384 7e5315da, r224 3394fdd7, compact_infer 8401812c, FLIP_TTA_FORCED, CANARY_*,
  V224_DONE) sits in the same base cells, re-indexed.
- kernel-metadata == V224-FLIP except id / title / code_file (same 3 datasets, docker pin 2757e0c7, NvidiaTeslaT4, private,
  internet off, no fleet dataset).
- After the push: `kaggle kernels pull eeshsaxena/rsna-sub-v224flrm` == the local notebook (17 cells, type + source); pulled
  metadata has the same datasets, docker pin, machine shape (`pulled_v224flrm/`).

## Offline sims (before the push)
- `sim_rm.py` re-run (the RM cell / patched r224 script / RM_GUARD; base independent): **SIM_RM PASS 22/22** (`sim_rm_flrm_rerun.txt`).
- `sim_flrm.py` (new, FLIP context; executes the notebook's own cells 11 RM, 14 her stage 3, 15 CANARY, 16 RM_GUARD in one
  namespace with FLIP_TTA_FORCED_ON as the FLIP cell sets it, on the V224-FLIP commit csvs + the V224-F20-RM commit's mirror r224
  files): **SIM_FLRM PASS 10/10** (`sim_flrm.txt`). Clean path: RM verify + patch in the FLIP namespace, CANARY_CLEAN, RM_GUARD_CLEAN,
  receipt flip_tta_forced True / rm_on True / fired []; submission == scipy recompute (3.3e-8); vs the V224-FLIP commit only ACL and
  MCL move (the columns whose r224 ranks the mirror moved). RM_GUARD fires (Fracture .5 only, other 11 columns byte-equal, flip kept)
  on a missing receipt / wrong perm / inconsistent raw; r224 failure -> CANARY_FIRED, RM_GUARD no-op. Writes the predicted commit
  submission `sim_flrm_expected_submission.csv`.

## The gate (`v224flrm_check.py OUT`; final line `V224FLRM_CHECK PASS n/m mode V224-FLIP-RM`)
- (A) V224-FLIP part: `v224_check.py OUT --flip --ref out_v224_v1` (the gate V224-FLIP passed). All checks OK with ONE by-design
  exception: "flip: submission MCL == V224-exact submission MCL" may fail only when this run's r224 MCL ranks differ from the
  V224-exact commit's (RM moves r224 ranks); then (A2) the submitted MCL must equal the V224 recipe on (this r384 raw, this r224,
  the V224-EXACT commit's R7 MCL) within 1e-6, i.e. the flip still leaves MCL alone.
- (B) RM part: `v224rm_check.py OUT --wf 0 --flip --ref-base out_v224flip_v1`. Every top-level check OK except its "base gate"
  line (it runs v224_check against the FLIP commit, where the flip-vs-exact checks cannot hold; superseded by (A)). Includes
  R224_MIRROR_VERIFIED / PATCHED, tta receipt, avg == 0.5 (plain + mirror) exactly, r224 sub == rankpct(avg), the mirror moves r224,
  plain == her r224 raw (1e-3), RM guard clean, R7 ours_sub and r384 raw == the V224-FLIP commit (1e-3).
- (C) independent recompute (scipy rankdata): r224 ranks from raptor224_raw.csv, then the V224 recipe (no re-rank) == submission.csv (1e-6).
- (D) one change vs the V224-FLIP commit: submission == recipe(FLIP commit r384 raw, FLIP commit R7, THIS r224) on every
  rank-comparable column (>= 10 of 12).
- (E) canary receipt flip_tta_forced True / use_f False / rm_on True / fired [] / w_f_global None; no fleet files or markers;
  FLIP_TTA_FORCED -> R7 -> RM -> r224 mirror stage -> stage 3 -> canary -> RM guard in log order.
- Gate tests before the commit output existed: NEGATIVE on the V224-FLIP commit (no RM) FAIL 4/8 (`gate_flrm_negative_on_flip.txt`),
  NEGATIVE on the V224-F20-RM commit (no flip, fleet) FAIL 3/10 (`gate_flrm_negative_on_f20rm.txt`); `gate_selftest_flrm.py`:
  POSITIVE on a synthetic FLIP-RM output (two real commits + the notebook's own tail cells) PASS 10/10, the same with the tta receipt
  removed FAIL 9/10 -> GATE_SELFTEST PASS (`gate_selftest_flrm.txt`).

## What the commit measured (MEASURED, 3 visible studies; `gate_v224flrm_v1.txt`)
| quantity | value |
|---|---|
| (A) v224_check --flip --ref V224-exact | 20/21 OK; the 1 fail is the by-design MCL-vs-exact (.0583; r224 MCL ranks moved by RM); (A2) MCL == recipe with the V224-EXACT R7 MCL: 3.3e-8 |
| flip | FLIP_TTA_FORCED, FLIP_TTA_READY + mcl_plain_pass_only in both worker logs, R7 receipt flip block (perm swaps M/L Meniscus + OA), R7 MCL == exact (0), 11 of 11 other R7 columns move in logit |
| RM | R224_MIRROR_VERIFIED (identity perm from both implementations, probability mean), PATCHED sha 90b2a4eb, tta receipt n 3 / n_pairs 3 / n_fail 0, avg == float32 0.5 (p + m) exactly, r224 sub == rankpct(avg) exactly, mirror moves all 12 labels (max .126, mean .022), plain == her r224 raw 1.0e-4 |
| canary receipt | r224 used, fired [], flip_tta_forced True, use_f False, rm_on True, w_f_global None, rm_guard mirror_ok; CANARY_CLEAN, RM_GUARD_CLEAN, V224RM_DONE |
| recomputes | v224_check 3.3e-8; gate (C) scipy, r224 ranks from the raw: 3.3e-8; recompute_flrm.py (own counting ranks): 3.3e-8 |
| (D) one change vs V224-FLIP commit | 12 of 12 columns rank-comparable, submission == recipe(FLIP r384 raw, FLIP R7, THIS r224) on all |
| no fleet | no fleet / HF files or markers |
| log timeline (s) | r384 done 54; FLIP_TTA_FORCED 72; R7 done 397 (R7 stage 325 s; FLIP commit 370 s); RM verify + patch 2.6 s; r224 stage 25 s (FLIP commit 23 s); stage 3 + canary + RM guard 425; log end 432 (FLIP commit 481) |
| r224 per study (s) | [total, plain, mirror] [8.80, 8.14, .46], [1.02, .46, .44], [9.20, 8.65, .43] (first plain pass per GPU = cudnn autotune) |

## Hidden wall estimate
- RM cost (MODEL, `rm_runtime.py` on this commit -> `runtime_v224flrm_v1.txt`; N 1,200-1,300, 2 T4): r224 stage plain ~7-10 min ->
  RM ~11-20 min; **extra +3.6 to +9.6 min** (commit-calibrated pass .456 s: +3.6-5.1; FLOP-scaled .623 s: +6.8-9.6; slow x1.3 in
  the upper end), central ~+0.09 h. Plus the RM cell itself (~3 s: patch + CPU verify).
- **V224-FLIP-RM submit-to-score: central ~4.7 h, range ~4.66-4.76 h** if the queue behaves as on 10-14 (V224-FLIP MEASURED ~4.6 h
  submit-to-score incl. queue). Kernel wall: V224-FLIP runtime review 3.5-5.7 h (central 4.4) -> FLIP-RM ~3.6-5.9 h (central ~4.5 h).
- Guards unaffected: RM runs after R7 (R7's 8.3 h deadline untouched); her r224 skip point is 7.83 h kernel age (worst case r224
  starts ~5.7 h, ends ~5.9 h: >= 1.9 h margin); her r224 timeout min(left, 2.5 h) >> 20 min; 9 h limit margin >= 3 h. No fleet stage
  in this kernel. P(a guard trips because of RM) < 1 % (GUESS, as for V224-F20-RM).
- A submit at ~06:25 IST 10-15 would score around ~11:05 IST (central).

## Submit (the main session decides; NOT run here)
Pre-register first (PREREG_10-15 Amendment 1 slot 2: one change vs V224-FLIP; read against V224-FLIP 57174021):
```
python session_2026-10-14/submit_at.py eeshsaxena/rsna-sub-v224flrm 1 "V224-FLIP-RM: V224-FLIP + r224 anatomical mirror TTA (r384 _mirror_tri, identity class map, probability mean); rsna-sub-v224flrm v1"
```
(`submit_at.py KERNEL VERSION "MESSAGE"`, checked; it submits submission.csv of that kernel version via the python API.)

## Concerns
1. Expected gain is tiny (RESEARCH_960 lever 4: public 0 to +1e-4 MODEL), below the paired read SD; on 3 commit rows the mirror moved
   only ACL and MCL ranks. The read is near a coin flip; RM was never scored on the LB on any base (V224-F20-RM was not submitted).
2. The gate needs one by-design exemption in the reused V224-FLIP gate (MCL vs V224-exact) and supersedes the v224rm gate's
   base-gate line; both replacements are explicit checks ((A) explained-by-RM + (A2), and (A) itself). Tested: negative on the
   FLIP and F20-RM commits FAIL, synthetic positive PASS, synthetic with the receipt removed FAIL.
3. Mirror is unvalidated for the r224 0.8-crop checkpoint on labels (same as RESULT.md concern 2); the 0.8 crop mirror is offset by
   1 px (concern 4 there).
4. The hidden latency estimate inherits the 10-14 queue; a busier queue moves the score time, not the 9 h kernel budget.
5. Every original sha assert and canary is untouched (byte proof); the RM guard adds the only new silent-path canary (Fracture .5).

## Files (all under session_2026-10-15/v224rm/)
`kernel_v224flrm/` (pushed notebook + metadata), `pulled_v224flip/` + `pulled_v224flrm/` (Kaggle pulls), `build_receipt_v224flrm.json`,
`celldiff_v224flrm.txt`, `celldiff_v224flrm_proof.py` + `celldiff_proof_v224flrm.txt`, `sim_rm_flrm_rerun.txt`, `sim_flrm.py` +
`sim_flrm.txt` + `sim_flrm_expected_submission.csv`, `v224flrm_check.py`, `gate_selftest_flrm.py` + `gate_selftest_flrm.txt`,
`gate_flrm_negative_on_flip.txt`, `gate_flrm_negative_on_f20rm.txt`, `push_v224flrm.log`, `status_v224flrm.log`, `out_v224flrm_v1/`
+ `out_v224flrm_v1.log`, `gate_v224flrm_v1.txt`, `recompute_flrm.py` + `recompute_v224flrm_v1.txt`, `runtime_v224flrm_v1.txt`,
`FLIPRM_READY.txt`.
