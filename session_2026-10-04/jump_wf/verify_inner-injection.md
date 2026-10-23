# Verification of the "inner injection" result

Date 2026-10-04, ~15:20 IST. Verifier for `inner_injection.md`. Tags: **MEASURED** (computed by me, path given), **MODEL** (follows from a stated formula), **GUESS**.

All my box work is CPU only (`nice -n 19`, `CUDA_VISIBLE_DEVICES=`), written only under `~/k2_test/analysis_wf/verify_inner/`. Training jobs were not touched. Logs are copied to `SP/jump_wf/verify_inner_logs/`.

## Verdict in one paragraph

The core numbers are real and they reproduce. I rebuilt the Raptor/CoAt side from a different source (the per-arm csvs instead of `raptor_raw.npz`) and wrote my own AUC and bootstrap. dPub +.0030, dSide -.0006, final +.0004 in-sample / -.0005 pseudo-OOS, global W .30 at -.0077 / -.0008, fleet std ratio 1.306, and the Spearman values .903/.902/.903 all match. The Rad held-out fold recovery is exactly right: 58/58 against the fold column shipped in `v52_oof.csv`. The DINO fold recovery is robust, and swapping in two other fold assignments moves the bracket by 0.001 or less. Two parts do not hold. **(1)** The claim that the LB anchors fall between the two brackets was checked against the *fleet's* brackets. Computed for the crop members that produced those anchors, the in-sample bracket overstates the harm in all 4 anchors, and the pseudo-OOS bracket is the closer one each time. **(2)** Under the pseudo-OOS bracket, the inner injection is *not* better than a plain global blend at the same movement (W ≈ .14): -.0007, CI90 [-.0036,+.0022]. So its "safer" edge only exists in the bracket the LB says is wrong. The bottom line stands: about 0 on LB, and no path to .95. The recommendation should be weaker: inner injection is not worth a slot or ~40 lines of graph code, because a smaller global W gives the same expected value.

## 1. Re-runs of the agent's own scripts (MEASURED)

- `inject_reconstruct.py` re-run gives an identical log (`verify_inner_logs/rerun_reconstruct.log`): stage2/stage3/coat_family/final are within 1.1e-16 of the stored files, raptor_public 2.6e-8 with equal ranks, and side 2.2e-16.
- `inject_stage1_probe.py` re-run (`rerun_stage1_probe.log`): 23 exact rank-sum ties, exact-integer version .0086, best of 300 orders .0086. The numbers are confirmed. That the cause is thread order is plausible but unproven: `_combine` (c16) accumulates in `per_member` list order. It does not matter downstream.
- The agent's original stage1-probe log was never written (there is no `inject_stage1_probe.log` on the box or the laptop). The re-run now supplies one.

## 2. Independent recompute (MEASURED, `verify_inner_v2_variants.py` / `.log` / `.csv`)

The own loaders read `localLB_parts/raptor_arm_*_prob.csv` and `coat_*_prob.csv`. They match `raptor_raw.npz` to 1.1e-16. My base graph gives pub .9254, fam .9315, side .9353, st3 .9734, final .9508 (stored .9509, 5 tie-flip cells) and fleet .9144.

| Quantity | Agent | Mine | Bootstrap (mine, 1000 study resamples) |
|---|---|---|---|
| 5th arm rank f=.20, dPub | +.0030 | +.0030 | CI90 [-.0001,+.0064], P>0 .94 |
| same, dSide | -.0006 | -.0006 | CI90 [-.0031,+.0017], P>0 .38 |
| same, dFinal in-sample | +.0004 | +.0004 | CI90 [-.0015,+.0023], P>0 .63 |
| same, dFinal pOOS (agent folds) | -.0005 | -.0005 | CI90 [-.0038,+.0027], P>0 .42 |
| global W .30, in-sample / pOOS | -.0077 / -.0008 | -.0077 / -.0008 | in [-.0120,-.0040]; pOOS [-.0047,+.0027] |
| global W .14, in-sample / pOOS | ≈-.0022 / ≈-.0001 (interp.) | -.0023 / +.0001 | |
| coat add f=.20, dSide | -.0028 | -.0028 | |
| Spearman fleet~pub / fleet~fam / pub~fam | .903/.902/.903 | .903/.902/.903 | |
| fleet prob std / Raptor blend std | 1.31 | 1.306 | |

## 3. Fold recovery against ground truth shipped in the inputs (MEASURED, `verify_inner_v1_folds.log`)

- **Rad:** `rsna-knee-e9-radimagenet-heads-v15/v52_oof.csv` and `rsna-knee-e11-diverse-heads-v20/v52_e11_oof.csv` both carry a `fold` column, and the two agree 58/58. The agent's argmax-loss head equals that fold on **58/58**. The head index maps to the fold (the nearest e10 head to the oof csv equals the fold column on 54/58). Held-out vs rest AUC per e10 head is .81 to .91 against .95 to .99. The Rad split agrees with the DINO split on 12/58, so the splits differ, as the agent said.
- **DINO:** the weights dataset ships `oof.npz`, but it is not a clean truth. It does not reproduce from the mimic members (median per-study abs diff .08), and its `y_derived` differs from the gold labels in 145/696 cells. Its argmin-distance fold agrees with the agent on 51/58. A label-free assignment (the member that deviates most from its sibling folds) agrees with the agent on **56/58**. The label-free check avoids the circularity of picking the max-loss member using the same labels you score with.
- The "chance ~.008" figure assumes the seed-runs are independent. They share training data, so agreement across seeds alone does not prove a member is held out. The label-free agreement (56/58) and the held-out vs rest AUC (.82 to .94 vs .994 to 1.0 by oof-match folds) do support it.
- **Effect on the bracket:**

| DINO fold assignment | T_oof | Final |
|---|---|---|
| agent | .8881 | .9306 |
| oof-match | .8891 | .9310 |
| label-free | .8904 | .9309 |
| oof.npz preds directly | .8830 | .9289 |

  The pOOS deltas move by at most ~.001 across these, e.g. inner .20: -.0005 / +.0002 / -.0003 / -.0001, and global W .30: -.0008 / -.0015 / -.0017 / -.0002.

## 4. Bracket calibration against the measured LB anchors (MEASURED, end of `verify_inner_v2_variants.log`)

The agent placed the crop-blend LB anchors against the **fleet's** brackets, which is not like for like. Here are the brackets for the crop members that actually produced those LB reads, on the same gold58 graph. Caveat: the crop checkpoints were gold58-selected, so their gold58 numbers are, if anything, optimistic.

| Anchor | LB delta (3-dp rounding ⇒ ±~.001) | In-sample bracket | pOOS bracket (4 fold variants) |
|---|---|---|---|
| cropaug_f1 W .15 | -.001 | -.0025 | +.0001 to +.0006 |
| cropaug_f1 W .25 | -.002 | -.0053 | -.0005 to -.0014 |
| mixSA_f1 W .25 | -.003 | -.0090 | -.0042 to -.0050 |
| K1 3-member W .25 | -.001 | -.0069 | -.0021 to -.0029 |

- The in-sample bracket overstates the LB harm in all 4 cases, by 1.5 to 6 times.
- The pOOS bracket is within about .002 of the LB each time and has no consistent sign error.
- Only 2 of 4 anchors lie strictly between the brackets. The other 2 sit at or just above the pOOS edge.
- So "the true effect lies between the brackets" is too weak a statement. The data say **the true effect sits near the pOOS bracket** (MODEL from 4 anchors).

## 5. What this does to the inner-vs-global comparison (MEASURED + MODEL)

- Under pOOS, the best inner variant (5th arm rank .20) is -.0005 / +.0002 / -.0003 / -.0001. Global W .14 (the same final movement) is +.0001 / +.0006 / +.0001 / +.0002.
- Paired difference, inner minus global .14, pOOS with agent folds: **-.0007, CI90 [-.0036,+.0022], P>0 .38** (MEASURED).
- Inner is clearly better only in-sample: +.0026, CI90 [+.0008,+.0048], P>0 .98. That is the bracket section 4 discredits.
- The md line "It is clearly better in the bracket that the LB anchors say is closer for the global blend" is therefore wrong, even by the agent's own anchor numbers, which are closer to its pOOS column.
- Against the K2 BLEND (global W .30), inner .20 still comes out at least as good in every bracket: pOOS margin 0 to +.0017, in-sample +.008. "Loses less than W .30" survives as a point estimate. In pOOS it is within noise, so I would call it medium, not medium-high.
- MODEL prediction for the pending K2 BLEND read, from pOOS: about -.0002 to -.0017, so .942 or .943. For K2 SOLO there is no prediction from this analysis.

## 6. Implementation snippet audit (MEASURED where stated)

- `diagnostics/btkd_parent_after_rad.npz` exists in the mimic working dir. It equals the stage-3 csv to 1.1e-16 with identical ranks, and its key is `values`.
- `_coatnet_weight` is a module-level global in c28, so it is reachable from the K2 cell.
- The c30 cleanup deletes only the `pixels_*.npy` memmaps (`_RSNA_CACHE_FILES`, c05). The receipt says 4 members and alpha .4.
- **Fragility:** c28 saves `raptor_raw.npz` *before* the neutral fill of failed Raptor studies. If any Raptor study fails on the hidden test, the rebuild gets NaN. The f=0 self-check cannot see it (`NaN > thr` is False). The K2 final finite check would then raise `_K2Fail` and drop the *whole* K2 output to the parent bytes.
- **Fix:** use `raptor_input_before_coat.csv` (post-fill, written by c28). Its ranks equal the ranks of the raw prob blend (MEASURED True on gold58). The rank-space 5th arm only needs those ranks: `pub_new = rank01(.8*rank_pct(before_coat) + .2*rank_pct(F))`. Also guard with `np.isfinite`.
- These are moot if the slot is not spent (section 7).

## 7. Corrected recommendations

1. **Do not spend a slot on the inner injection, even if K2 BLEND reads .943.** Its expected public effect is about 0 (pOOS, the LB-calibrated bracket). It is no better than a lower global W, and it adds about 40 lines of graph-rebuild code with a NaN fragility. If a low-dilution fleet entry is ever wanted for final-selection hedging, a global W of about .10 to .15 is the same expected value and needs only a literal change. Expected public change: about 0 ± .002 (GUESS). It does not reach .95.
2. **Keep the gate idea, with a fix:** screen new candidates on side-level dSide and on the **pOOS** final (not the in-sample final), using `inject_variants.py` or `verify_inner_v2_variants.py`. Note that 58 studies cannot resolve gains under ~.003. The rho < .85 and gold58 ≥ .93 targets are reasonable, but no current member meets them (fleet .914 / rho .921).
3. **Stop tuning fleet weights.** Agreed. The fleet is redundant with the CoAt family on every measured axis. A jump needs a strong member that is also diverse from it, or a better DINO side on test. Neither exists in hand today.

## Files

- Box `~/k2_test/analysis_wf/verify_inner/`:
  - `verify_inner_v1_folds.py`, `.log`, `verify_folds.npz`
  - `verify_inner_v2_variants.py`, `.log`, `.csv`
  - `verify_inner_v3_files.log`
  - `rerun_reconstruct.log`, `rerun_stage1_probe.log`
- Laptop `SP/jump_wf/`:
  - `verify_inner_v1_folds.py`, `verify_inner_v2_variants.py`
  - `verify_inner_logs/*`
  - this file
