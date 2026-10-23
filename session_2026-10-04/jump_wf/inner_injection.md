# Inner injection of the fleet member into the 0.943 parent graph

Date 2026-10-04. Scope: is it better to put our ConvNeXt-T fleet member (f0_s42 mix50, gold58 .9144) *inside* the parent's Raptor/CoAt side than to use the K2 global outer blend?

Tags: **MEASURED** = computed by the box scripts below on the gold58 box mimic run (paths given). **MODEL** = follows from the stated formula or assumption. **GUESS** = judgement.

## Bottom line

1. **The graph is reconstructed exactly** from the per-arm csv/npz files, from stage 2 onward. Max abs diff vs the stored stage files is ≤ 1.1e-16 for stage 2, stage 3, coat_family, raptor_coat_side and the final. raptor_public differs by 2.6e-8, which is float32 text rounding, and its ranks are identical. Stage 1 (the DINO stack) is off by at most one rank step. The cause is 23 exact rank-sum ties that the parent breaks by float accumulation order, and that order depends on thread completion, which is not logged. Stage 1 is not used downstream in this analysis. (MEASURED, `inject_reconstruct.log`, `inject_stage1_probe.py`)
2. **The raptor_public gain does not survive into raptor_coat_side.** Putting the fleet into the Raptor arms improves raptor_public, for example +.0030 with the 5th arm in rank space at f=.20 (90% CI [-.0004,+.0064], P>0 .93). But raptor_coat_side then *drops* by .0006 (CI [-.0032,+.0018], P>0 .35). Every inner variant tested gives a side delta between +.0002 and -.0057; none is positive beyond noise. (MEASURED, `inject_perlabel.log`, `inject_variants.log`)
3. **Why:** the side is already a .6/.4 mix of raptor_public (.9254) and the CoAt family (.9315), and those two have Spearman .903 with each other. The fleet has Spearman .903 with pub and .902 with fam, so it is no more diverse than they already are, and it is weaker (.9144). A third component that is just as correlated but weaker adds nothing at the side level. The pub gain is "information the CoAt family already brings". (MEASURED, `inject_perlabel.log`)
4. **On the final:** relative to the global blend, the inner injection is *less harmful* if the DINO side is strong (in-sample bracket). It is about *equally neutral* if the DINO side is weak (pseudo-OOS bracket). Neither bracket shows a gain beyond noise. The best inner variant, the 5th Raptor arm in rank space at f=.20, gives +.0004 in-sample and -.0005 pseudo-OOS. The global blend at the same final movement (W≈.14) gives -.0022 in-sample and about -.0001 pseudo-OOS. (MEASURED deltas; interpretation MODEL)
5. **Verdict:** the inner injection is the *safer* way to include the fleet, because it never dilutes the DINO side. But it is **not a path to ≥ 0.95**. The expected public LB change for the best inner variant is about 0 ± .002 (GUESS). It is not worth a slot ahead of the pending K2 reads unless the K2 BLEND comes back ≥ .943. A jump needs a component that is both strong (≥ .93 OOS) **and** diverse (rho < ~.85) against the Raptor/CoAt side, or something that improves the DINO side on test.

## 1. Parent combination graph (cell references are to `SP/k2/build/kernel/rsna-sub-k2-fleet/rsna_sub_k2_fleet.ipynb`)

Notation: `rank_pct` = pandas `rank(method='average', pct=True)` per label column. `rank01` = `(rank-1)/(n-1)` (`rsna_rank01`, cell 5). All ranks are taken over the test studies (58 in the mimic).

| Stage | Cell | Formula | Weights |
|---|---|---|---|
| pipeline_stage_1 (DINO stack) | c16 `_combine` + `write_submission` (called at the end of `infer_from_package`) | S1 = rank_pct( Σ_m w_m · rank_pct(p_m) / Σ_m w_m ) over the 20 "public-frontier" member probabilities | 20 members = 4 seed-runs (base-s336x12, seed2..4) × 5 folds. Every manifest `weight`/`target_weight` is absent, so all are 1.0 (MEASURED from `~/kmimic/kaggle/input/rsna-knee-weights/manifest.json`) |
| pipeline_stage_2 (+A5) | c23 `_a5_rank_available`, `_a5_blend_available`; c24 | a5 = mean over 5 folds of rank01(fold probs on A5-eligible studies), cast to float32 (`A5_PREDS`). For eligible studies: S2 = .48·rank_pct(S1) + .52·rank_pct(a5). Non-eligible: rank_pct(S1). **Not re-ranked** | A5_W = .52 (all 58 eligible in the mimic) |
| pipeline_stage_3 (+Rad +calibrator) | c26 `_rad_main` | ref = rank_pct(.5·rank_pct(mean5 e10) + .5·rank_pct(mean5 e13)); cand_l = .45·rank_pct(S2) + .55·ref for all labels except Baker's (kept as S2); branch = rank_pct(.8·rank_pct(cand) + .2·rank_pct(mean5 e13-on-e11-layout)); calibrator: x = [base, ref, second, ref-base, second-base, mean3, 4 label-group means, 12 protocol counts] (88 columns), adj = rank_pct(((x-μ)/σ)·coefᵀ + b); gate labels get .6·branch + .4·adj; S3 = rank_pct | `_RAD_ALPHA` .55, `_RAD_EXCLUDE` (Baker's), `_RAD_E13_MEMBER_WEIGHT` .5, `_RAD_V48_SECOND_ALPHA` .2; gate = ACL, Medial OA, Lateral OA, PF OA, Effusion, Baker's, Contusion (from `_RAD_CAL_PAYLOAD`) |
| raptor_public_rank (`_raptor_public.csv`) | c28 `_ke_run_raptor_arms` | pub = rank01( Σ_k w_k · clip(p_k,0,1) ), float32 | maxspan-v5 .60, native384dense-v10 .10, maxspan-v5-reverse .10 (same checkpoint, window order flipped), native384-v8 .20 (sum 1) |
| coat_family_rank | c28 `_coat_substitute` | fam = rank_pct( mean of the 4 members' probability means ) | resgated_top3 (mean of 3 raw checkpoints), global96_top3, d4_swa3, repairv1_top3, each 1/4 |
| raptor_coat_side_rank (`_raptor.csv` after the CoAt cell) | c28 `_coat_substitute` | side = .6·rank_pct(pub) + .4·fam, **not re-ranked** | `private_alpha` .4 |
| pipeline_stage_4 = final = submission.csv | c28 end (`_coatnet_weight`, `_blend_output`); published by c30 | final_l = rank_pct( (1-w_l)·rank_pct(S3_l) + w_l·side_l ) | w_l = .60, except ACL .75, Medial Meniscus .80, **Lateral Meniscus 1.00**, Lateral OA .75, Fracture .75 (mean .6875) |
| K2 global blend | c31 `_k2_stage` | sub = rank_pct( (1-W)·rank_pct(parent) + W·rank_pct(F) ), F = raw-prob mean of the fleet members | W_MAIN .30, variants .20/.40 |

**Nominal shares in the final (MODEL, ignoring rank nonlinearity):**

| Component | Mean nominal share |
|---|---|
| Transformer side (DINO+A5+Rad) | .3125 (0 for Lateral Meniscus) |
| raptor_public | .6875 × .6 = .4125 |
| maxspan-v5 | .2475 |
| v10 | .041 |
| v5-reverse | .041 |
| v8 | .0825 |
| Each CoAt member | .069 |

So the K2 global blend at W .30 gives the fleet about as much weight as the entire DINO/A5/Rad side.

## 2. Reconstruction check (MEASURED, `~/k2_test/analysis_wf/inject_reconstruct.py`, log `inject_reconstruct.log`)

| Stage rebuilt | Max abs diff vs stored file | Note |
|---|---|---|
| stage1 DINO stack | 1.72e-2 (one rank step) | 23 exact rank-sum ties. Exact-integer tie version: .0086. Best of 300 random accumulation orders: .0086. Not reproducible byte-exact (thread order) |
| stage2 +A5 from stored S1 | 1.1e-16 | needs the float32 cast of a5 (c23) |
| a5_rank_mean from speed_a5_raw.npz | 0 | |
| stage3 +Rad +calibrator from stored S2 | 1.1e-16 | |
| raptor_public from raptor_raw.npz | 2.6e-8 | float32 text rounding; ranks identical |
| coat_family from the 4 prediction npz | 1.1e-16 | |
| raptor_coat_side from stored pub/fam | 2.2e-16 | |
| final from stored S3 + side | 1.1e-16 | |
| stored final vs localLB_parent_preds.csv | 0 | |
| final fully chained from raw inputs | 1.7e-2 | only through the stage-1 tie-breaks |

For the variants, the base is my chained graph. It differs from the stored final in 5 of 696 cells, through ~1e-16 tie flips in the side; AUC is .9508 vs .9509. All deltas are like for like.

gold58 macro AUC (MEASURED):

| Stage | AUC | Sample status |
|---|---|---|
| S1 | .9956 | in-sample |
| S2 | .9497 | |
| S3 | .9734 | in-sample |
| pub | .9254 | OOS |
| fam | .9315 | OOS |
| side | .9353 | OOS |
| final | .9509 | |
| fleet | .9144 | |

## 3. Injection variants on gold58 (MEASURED, `inject_variants.py`, `inject_variants.log`, `inject_variants_results.csv`)

Columns:
- **dPub, dSide:** OOS sub-stage deltas, macro AUC on 58 studies.
- **Side CI:** paired bootstrap, 2000 reps, 90% interval.
- **MARS:** mean abs rank shift of the final vs the parent final, in pct-rank units.
- **ρ:** Spearman of final_new vs final_old, mean over labels.
- **W same MARS:** the global W that moves the final by the same amount.
- **dFinal in:** final delta with the real (in-sample) transformer side.
- **dFinal pOOS:** final delta with the pseudo-OOS transformer side (section 4).

| Variant | dPub | dSide [CI90] | MARS | ρ | W same MARS | dFinal in | dFinal pOOS [CI90] |
|---|---|---|---|---|---|---|---|
| (a) 5th arm prob f=.05 | +.0012 | +.0001 [-.0015,+.0018] | .0045 | .9993 | .09 | -.0003 | +.0003 [-.0017,+.0021] |
| (a) 5th arm prob f=.10 | +.0020 | -.0007 [-.0031,+.0018] | .0076 | .9987 | .12 | -.0002 | -.0008 |
| (a) 5th arm prob f=.25 | +.0016 | -.0010 [-.0054,+.0035] | .0164 | .9956 | .22 | +.0001 | -.0022 |
| (a) 5th arm prob f=.50 | -.0011 | -.0043 | .0270 | .9904 | .32 | -.0013 | -.0031 |
| (ar) 5th arm rank f=.10 | +.0015 | +.0002 [-.0010,+.0014] | .0038 | .9995 | .08 | -.0006 | -.0002 |
| **(ar) 5th arm rank f=.20** | **+.0030** | **-.0006 [-.0032,+.0018]** | .0093 | .9984 | .14 | **+.0004** | **-.0005 [-.0040,+.0024]** |
| (ar) 5th arm rank f=.33 | +.0028 | -.0012 | .0154 | .9962 | .21 | +.0001 | -.0013 |
| (b) replace maxspan-v5 (.60) | -.0020 | -.0049 | .0322 | .9866 | .37 | -.0010 | -.0038 |
| (b) replace v10 (.10) | +.0016 | -.0012 | .0091 | .9982 | .14 | -.0005 | -.0010 |
| (b) replace v5-reverse (.10) | +.0020 | -.0010 [-.0037,+.0019] | .0084 | .9985 | .13 | -.0004 | -.0007 |
| (b) replace v8 (.20) | +.0003 | -.0022 | .0149 | .9963 | .21 | -.0000 | -.0024 |
| (c) coat add f=.20 (5th member) | 0 (dFam -.0013) | -.0028 [-.0054,-.0004] | .0100 | .9980 | .15 | -.0014 | -.0017 |
| (c) coat add f=.33 / .50 | 0 (dFam -.0027/-.0044) | -.0025 / -.0030 | .014/.019 | | | -.0010/-.0020 | -.0024/-.0024 |
| (c) coat replace resgated / g96 / d4 / repairv1 | 0 | -.0025 / -.0016 / -.0030 / -.0027 | ~.013 | | | -.0021/-.0007/-.0010/-.0017 | -.0022/-.0017/-.0023/-.0016 |
| (c2) side-level f=.10 / .20 / .30 | 0 | -.0014 / -.0016 / -.0024 | .008/.015/.020 | | | -.0006/-.0005/-.0007 | -.0015/-.0031/-.0022 |
| (e) 5th arm .25 + coat .20 | +.0016 | -.0041 | .0223 | .9927 | .27 | -.0017 | -.0036 |
| (d) global W=.05 | 0 | 0 | .0011 | .9999 | .05 | -.0002 | -.0000 |
| (d) global W=.10 | 0 | 0 | .0055 | .9993 | .10 | -.0014 | -.0005 |
| (d) global W=.15 | 0 | 0 | .0103 | .9984 | .15 | -.0025 | -.0001 [-.0021,+.0017] |
| (d) global W=.20 | 0 | 0 | .0143 | .9973 | .20 | -.0035 | -.0003 |
| (d) global W=.30 (= K2 BLEND) | 0 | 0 | .0252 | .9927 | .30 | -.0077 | -.0008 [-.0047,+.0026] |
| (d) global W=.40 | 0 | 0 | .0346 | .9870 | .40 | -.0106 | -.0024 |

Other MEASURED facts (from `inject_variants.log` and `inject_perlabel.log`):
- The fleet's probability std is 1.31× that of the Raptor prob blend (per label .98 to 1.58). So in the parent's own prob-space arithmetic the fleet weighs more than its nominal f. That is why the prob-space variants degrade faster than the rank-space ones.
- Spearman of the fleet vs each Raptor arm is .887 to .893, and vs the CoAt members .852 to .882.
- Per label, 5th arm rank f=.20, dSide:
  - **Gains:** Lateral Meniscus +.0068 (outer weight 1.0), Lateral OA +.0048.
  - **Losses:** Synovitis -.0108, MCL -.0102 (only 9 positives).
  - Selecting labels on 58 studies would be overfitting, so per-label injection is not recommended.

### What can and cannot be concluded

- **Can (MEASURED):** on the OOS sub-stages, the fleet improves raptor_public (P>0 .93 at rank f=.20). But no placement improves raptor_coat_side, which is the only OOS input to the final. The CoAt family already supplies what the fleet adds to Raptor. Inner placement moves the final much less per unit of nominal fleet share than the global blend does: 5th arm rank f=.20 has a nominal share of .083 and moves the final like global W=.14.
- **Cannot:** score the final honestly on gold58. The DINO stack (.9956) and the Rad heads (.97/.98) are fitted on these 58 studies' gold labels, so any change that dilutes them looks bad (the in-sample bracket). The pseudo-OOS bracket in section 4 is an approximation with a known bias in the other direction.

## 4. Pseudo-OOS transformer side (bracket for the final)

Method (MEASURED, `inject_oof_probe.py`, `inject_oof_probe2.py`, logs alongside):

**Recovering the DINO holdout fold.** For each study, I took the per-study label log-loss of each DINO member. In every seed-run, the fold member with the largest loss is taken to be the one that held the study out. The 4 seed-runs agree on this fold for **54 of 58** studies; chance agreement would be ~.008. So the fold split is shared and recoverable. Max/second loss ratio medians are 1.35 to 1.73.

**Recovering the Rad holdout head.** Rad heads use a different split: agreement with the DINO fold is .16 to .19, which is chance. But e10 and e13 agree with each other on the held-out head for **.90** of studies, so I used the argmax of the summed e10+e13+second loss.

**Building the pseudo-OOS side T_oof:**
- per study, the probability mean of its 4 out-of-fold DINO members;
- rank, then the unchanged A5 stage;
- the Rad stage using only the held-out head;
- the unchanged calibrator.

Results (MEASURED):

| Component | Pseudo-OOS AUC | In-sample AUC |
|---|---|---|
| DINO-oof | .8484 | .9956 |
| +A5 | .8739 | |
| Rad-oof e10 | .8614 | |
| Rad-oof e13 | .8638 | |
| T_oof | .8881 | .9734 |
| final with T_oof | .9306 | |

**Biases (MODEL):** T_oof is too weak compared with the real test-time T. It uses 4 OOF members of one fold instead of 20 members, and 1 Rad head instead of 5. The calibrator is still whatever it was fitted on. The in-sample T is far too strong. The true test-time T lies between them, so:
- the in-sample column is pessimistic about any dilution of T (it favours inner placement);
- the pseudo-OOS column is optimistic about diluting T (it favours the global blend).

**LB anchors:** global blends of a crop member with LB .924 solo cost -.001 (W .15) and -.001 to -.002 (W .25) on the public LB. Those sit between the brackets for comparable W: in-sample -.0025/-.0056, pseudo-OOS -.0001/-.0005. The K2 BLEND read at about 17:50 IST (W .30; brackets -.0077 and -.0008) adds a fourth anchor.

**Reading:** across both brackets, the best inner variant (5th arm rank f=.20: +.0004 / -.0005) is never worse than the global blend at equal movement (W≈.14: about -.0022 / -.0001) by more than noise. It is clearly better in the bracket that the LB anchors say is closer for the global blend. But **neither is positive in both brackets beyond noise**. Bootstrap CIs are about ±.003 to ±.004 for every final delta.

## 5. If the inner injection is used: exact implementation (no Kaggle changes made)

**Recommended form:** 5th Raptor arm in **rank space**, f = .20, i.e. pub' = rank01(.8·rank_pct(Σ w_k clip p_k) + .2·rank_pct(F)). It is the only variant non-negative in-sample with the largest pub gain, and it is immune to the fleet's 1.31× probability scale. Runner-up: replace maxspan-v5-reverse (prob space).

**Where:** only in the appended K2 cell (cell 31, `_k2_stage`). Rebuild the final from the parent's persisted intermediates after `F` is computed (right after `F = fleet.to_numpy(np.float64)` and the rho gate). Moving the fleet before cell 28, as a true 5th entry of `ARMS`, would be invasive and gives the same numbers. No parent cell needs to change. The inputs survive cell 30's cleanup: `_RSNA_CACHE_FILES` only holds pixel caches.

```python
# inside _k2_stage, after F (parent row order = pub['StudyInstanceUID'])
uid = pub['StudyInstanceUID'].astype(str).tolist()
def _re(z, key):                                         # reorder an npz to the parent's study order
    pos = {str(s): i for i, s in enumerate(z['study_uids'])}; return np.asarray(z[key])[..., [pos[u] for u in uid], :]
raw = _re(np.load(f'{_K2_WORK}/raptor_raw.npz'), 'raw_probabilities').astype(np.float64)        # (4,n,12) c28 outputs
wr = np.array([.60, .10, .10, .20])                                                               # c28 ARMS w (normalised)
pb = np.tensordot(wr / wr.sum(), np.clip(raw, 0, 1), axes=(0, 0))
def _r01(a): return (pd.DataFrame(a).rank(method='average').to_numpy() - 1) / (len(a) - 1)
pub_new = _r01(0.80 * _k2_rank(pb) + 0.20 * _k2_rank(F)).astype(np.float32).astype(np.float64)
rec_ = json.load(open(f'{_K2_WORK}/_coat_raptor_blend_receipt.json'))                           # c28 receipt
fam = pd.read_csv(f'{_K2_WORK}/_coat_family_rank.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[uid, _K2_TARGETS].to_numpy()
a = float(rec_['private_alpha'])                                                                  # .4
side_new = (1 - a) * _k2_rank(pub_new) + a * fam
Tz = np.load(f'{_K2_WORK}/diagnostics/btkd_parent_after_rad.npz')                               # stage 3 (= _ke_ours)
Tr = _k2_rank(_re(Tz, 'values'))
wl = np.array([_coatnet_weight[t] for t in _K2_TARGETS])                                         # c28 global
inner = _k2_rank((1 - wl) * Tr + wl * side_new)
# self-check: the same rebuild with f=0 must reproduce the parent (allow float tie flips, as measured 5/696 cells)
side0 = (1 - a) * _k2_rank(_r01(pb).astype(np.float32).astype(np.float64)) + a * fam
chk = _k2_rank((1 - wl) * Tr + wl * side0)
if np.abs(chk - _k2_rank(pub[_K2_TARGETS].to_numpy())).max() > 2.0 / n + 1e-9: raise _K2Fail('inner rebuild != parent')
out['submission_inner20.csv'] = ...  # same frame build as the wXX files; add to _k2_loud_names(); MAIN_MODE 'inner'
```

Guards to keep:
- If the receipt shows no CoAt family members (`coat_family_unavailable`), the parent used Raptor only. Then set side_new = rank01-pub only, or raise `_K2Fail` so the slot errors loudly instead of silently differing.
- Keep the P2 fail-loud rule: the inner file exists only if the fleet stage is clean.

**Runtime cost (MODEL):** CPU-only rank operations on (n × 12) arrays, well under 1 s for a few thousand test studies. No GPU time beyond the fleet stage that K2 already runs. The 9 h budget is unchanged relative to K2.

## 6. Recommendation (GUESS, ranked by expected value)

1. **Do not plan on the inner injection for a jump.** Expected LB change is about 0 ± .002. It cannot reach .95 because, on gold58, the fleet is redundant with the CoAt family at the side level.
2. **Read K2 SOLO and K2 BLEND first** (~17:50 IST).
   - If K2 BLEND ≥ .943: the inner rank f=.20 variant is the lower-dilution version worth one slot.
   - If K2 BLEND ≤ .942: inner injection would most likely land around .942 to .943 (MODEL from the brackets), so skip it.
3. **What could move the side:** a new member must beat rho ~.85 against raptor_coat_side while staying ≥ .93 OOS. Check that on gold58 against `raptor_coat_side_rank.csv` *before* any LB slot. The ready-made test: `inject_variants.py` with `FLEET` pointed at the new member's gold58 preds, gated on dSide > 0 with P>0 ≥ .8.

## Files

- **Box** `~/k2_test/analysis_wf/`:
  - `inject_common.py`: loader plus the exact graph functions.
  - `inject_reconstruct.py` and its `.log`.
  - `inject_stage1_probe.py`.
  - `inject_oof_probe.py`, `inject_oof_probe2.py` and their logs; `oof_dino_fold_majority.npy`.
  - `inject_variants.py` and its `.log`; `inject_variants_results.csv`.
  - `inject_perlabel.py` and its `.log`.
  - `inject_radcal_payload.py`: the calibrator payload copied from cell 26.
- **Laptop** `SP/jump_wf/`: same scripts and logs, `inject_variants_results.csv`, `cells/cNN.py` (the notebook split per cell), and this file.
