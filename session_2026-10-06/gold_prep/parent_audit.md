# Parent shake-up audit and the flat hedge for final pick 2 (2026-10-06, ~22:15 IST)

Scope: the 0.943 parent = cells 0-30 of `SP/k5/build/kernel/rsna-sub-k5-fleet/rsna_sub_k5_fleet.ipynb` (verbatim
ryokucha `rsna-knee-d4-blend-0946-ours10` minus its private ConvNeXt; cell 0 says the public part reads 0.943).
Cell numbers below are 0-based notebook cells, line numbers are lines inside that cell's source (split copies in
`SP/gold_prep/k5cells/cNN.py`). Nothing was submitted, pushed or changed on Kaggle. Box work: CPU only, `nice -n 19`,
under `~/gold_prep/parent_audit/`; no training process, GPU or protected directory was touched.

Tags: **MEASURED** = computed here or a logged LB read. **CLAIMED** = written in a public notebook or forum post.
**MODEL** = arithmetic from stated inputs. **GUESS** = judgement.

`SP` = `C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad`

---

## 0. Bottom line

1. **Exactly one live per-label constant is documented as tuned on the public LB:** the outer CoAt routing map in
   cell 28 (lines 1051-1057): w = .60 for every label except ACL .75, Medial Meniscus .80, **Lateral Meniscus 1.00**,
   Lateral OA .75, Fracture .75. Its own lineage calls it "probe #22" and credits it with 0.939 (flat .60) to 0.941
   on the public split (CLAIMED, maverickss26 config cell); Tian's hedge #16 removed it from the same graph family and
   read 0.940 vs 0.942 (MEASURED LB, ref 56471784).
2. The other per-label overrides are **not** documented as LB-tuned and cannot be flattened from what the parent
   persists: the calibrator gate (cell 26, 7 labels at .4; an offline fit, probably gold58-informed), `_RAD_EXCLUDE`
   = Baker's (cell 26, inherited), and the DINO per-label window pooling (cell 16, structural). The global scalars
   A5 .52 / Rad .55 / Rad-second .20 were raised from .45 / .50 / .15 in the 0.942 "speedy" lineage (method not
   stated; 1 degree of freedom each). Several per-label maps in cell 16 (`LEGACY_*`) are **dead code**: computed or
   defined, never used in the output.
3. **Design (KF):** one config key in our K-kit cell, `K2_CFG['OUTER_FLAT'] = 0.60`. The K-kit rebuilds the parent
   final from cell 28's own persisted inputs (`btk_v32_rest_input.npz` = stage 3, `_raptor.csv` = Raptor/CoAt side)
   with a flat .60 outer weight and blends that with the fleet. Parent cells untouched. Self-check: the same rebuild
   with the parent's map must reproduce `_k2_parent.csv`, and (verifier fix 10-06) stage 3 must equal cell 26's
   saved `diagnostics/btkd_parent_after_rad.npz` on all 12 labels, because the rebuild is blind to stage 3 on
   Lateral Meniscus (map w 1.00). **Tested (MEASURED):** max abs diff **0.0** on all three real K5 Kaggle commit
   outputs (main, solo, w50) and **1.1e-16** on the 58-study box mimic; all 6 negative tests raise `_K2Fail`.
   Spliced into the real K5 and K7 K-kit cells and run end to end with a stub fleet: clean stage, every wXX equals
   the flat-parent recompute, gold58 W .50 blend with the K3 fleet .9338 (map) -> .9382 (KF), as in sec 2b.
4. **Expected public change (K5 w50 with KF vs K5 w50 = .946):** about **-.001** (range -.0018 to 0), so a display of
   .945 most likely, .946 possible (MODEL: parent-level -.002 MEASURED by Tian, times a blend dilution of .3-.6
   MEASURED on the mimic). The paired SE of that difference on ~390 public studies is only ~.0006-.0007 (mimic
   bootstrap, scaled), far below the .004-.006 SE of an absolute score; one read cannot tell the two apart.
5. **Private robustness:** the map's public +.002 came out of a probe search with a per-probe noise SD of ~.001 at the
   parent level, so a large part of it is likely selection (GUESS: true parent-level value +.0005 to +.001; some
   structural support for Lateral and Medial Meniscus, none for Lateral OA). gold58 cannot settle the sign: the two
   brackets disagree (blend level: in-sample +.004, pseudo-OOS -.002 to -.003). Expected private difference
   flat minus map at the blend level: about -.0003, range -.001 to +.001 (GUESS).
6. **The hedge is cheap but nearly identical to pick 1** (per-label Spearman vs pick 1 .987-.998 on the 5 changed
   labels, 1.000 on the other 7; mean over 12 = .998). It protects at most ~.001 of private. **Recommendation:**
   build KF as a ready, gated kernel; use OUTER_FLAT .60 on final pick 2, ideally on top of the most different
   strong member set we have by 10-20 (new family or relabel set) rather than on a copy of pick 1's members. Spend at
   most one slot on it, and only as a plumbing read. No other constant is worth flattening.

---

## 1. Inventory of weight maps, per-label overrides and tuned constants in cells 0-30

| # | constant (cell:line) | what it does | per label? | live in output? | provenance / tuned on public LB? | flatten in KF? |
|---|---|---|---|---|---|---|
| 1 | `_coatnet_weight` (c28:1051-1057), applied c28:1059-1067 | final_l = rank((1-w_l) rank(S3_l) + w_l side_l); w .60, ACL .75, MedMen .80, **LatMen 1.00**, LatOA .75, Fracture .75 (mean .6875) | **yes** | yes | **LB-tuned, documented.** "probe #22", flat .60 = 0.939, map = 0.941 public (CLAIMED: `rsna-knee/_pub0941/rsna-knee-0941-restructured.ipynb` cell 2; same wording in `SP/nbs/kolyaflexcu_rsna-knee-speedy947/*.ipynb` cell 2, which adds that LatMen 1.00 is the likeliest place to give points back privately). Thread 742050 (`SP/round2/score_mentions.md` ~L984-1004): tuned on the public split, keep a flat-.60 version as the 2nd pick. Removal measured: Tian #16 = 0.940 vs 0.942 (`SP/knee/tian/experiments.md` L1460-1490) | **YES -> .60** |
| 2 | calibrator `gate` + fixed .6/.4 mix (c26:483; list inside `_RAD_CAL_PAYLOAD`, c26:4) | on ACL, Medial OA, Lateral OA, PF OA, Effusion, Baker's, Contusion: .6 branch + .4 rank(linear 88-feature model) | **yes** (7 of 12) | yes | Fitted offline ("V18"), L1-sparse coefficients (12 x 88, absmax .157, protocol columns tiny); Tian's research calls it gold-tuned (CLAIMED/GUESS, `SP/knee/tian/research.md` ~L213, L251-254). No public statement that the gate list was LB-probed. Mattia's v38 ledger tried calibrator .40 -> .00 (`SP/round2/forum_mattia/*.ipynb` metadata); his lineage kept .40 (GUESS: v38 did not win) | no (see sec 3.3) |
| 3 | `_RAD_EXCLUDE = ("Baker's",)` (c26:35, used c26:451-453) | Baker's keeps stage 2, no Rad blend | **yes** | yes | Inherited: cf696666 left two findings out (Baker's + Fracture) (CLAIMED, credit in `SP/research/rsna-knee-e11-train.ipynb` md cell 2); our lineage re-included Fracture; aastik and romantamrazov forks re-exclude Fracture (`SP/jump_wf/public_intel.md` L46, L53). Selection method unknown | no |
| 4 | `_RAD_ALPHA` .55, `_RAD_E13_MEMBER_WEIGHT` .50, `_RAD_V48_SECOND_ALPHA` .20 (c26:34, 40, 41; used c26:447, 453, 459) | Rad reference vs stage 2; E13 vs E10; E13-on-E11 second pass | no (global) | yes | Raised from .50 / .15 of the 0.941 lineage in the 0.942 "speedy" lineage, together with 94 windows (CLAIMED, speedy config cell). Method not stated (GUESS: public probes). In a sister lineage an E10 alpha was chosen by grouped nested CV with "no visible-test tuning" (CLAIMED, e11-train ledger V41) | no |
| 5 | `_RAD_TWIN_ALT_WEIGHT` .500001 (c26:42) | none | - | **dead** (defined, never used) | - | - |
| 6 | `A5_W = 0.52` (c23:217; applied c24 via `_a5_blend_available`) | stage 2 = .48 rank(S1) + .52 rank(A5) on eligible studies | no | yes | .45 in the 0.941 lineage, .52 in speedy (CLAIMED). Method not stated | no |
| 7 | Raptor `ARMS` w .60/.10/.10/.20 (c28:50-67) and `k_eval` 62/42 -> 94 (c28:637-639) | Raptor prob blend over 4 views | no | yes | Mattia's own values; the public .55/.15 variant was an LB probe that his stack and the jiweiliu fork reverted (CLAIMED, 742050). 94 windows = speedy lineage | no |
| 8 | `private_alpha = 0.4` (c28:909, applied c28:1021) | side = .6 rank(public Raptor) + .4 CoAt family, not re-ranked | no | yes | "Retain BTKD inner blend coefficients" (code comment); receipt says `finding_specific_weights: false` | no |
| 9 | CoAt family 1/4 each, rank of the probability mean (c28:968-1015, receipt c28:1029) | 4 CoAt children | no | yes | equal by construction | already flat |
| 10 | `PUBLIC_FRONTIER_TARGET_POOL` (c16:511; used c16:274 and c16:602) | DINO window pooling: max for Fracture, Contusion, both Menisci, Baker's; top2 for ACL, MCL; mean otherwise | **yes** | yes | From the DINO baseline author (focal vs diffuse findings); structural, not stated as LB-tuned. Changing it needs a DINO re-run | no |
| 11 | `TTA_TARGET_POOL` (c16:512) | same + Synovitis `original_mean` for the jittered "primary" prediction | yes | **dead for output** (primary preds go to `per_member`, which is never combined; only `public_frontier_members` reach `_combine`, c16:826) | - | - |
| 12 | `LEGACY_FOLD_SOFTPOOL_BETA/ALPHA` (c16:516-524), `LEGACY_MEMBER_WEIGHT_BY_TARGET` = LatMen 15, MedOA 2.5, LatOA 15, Contusion 5 (c16:525), `LEGACY_WEIGHT` .5 (c16:615), `combine_public_members_by_fold`, `blend_legacy_frontier_and_soft` (c16:679-713) | an older per-label soft-pool / member-weight branch | yes | **dead** (soft pool is computed per member then dropped; the two functions and the weight map are never called; `legacy_group_members()` returns {}) | look like remnants of earlier probe maps; no effect on our bytes | - |
| 13 | DINO `_combine` member weights (c16:661-677) | rank mean of 20 members | no | yes | entries carry no `weight`/`target_weight`, so all 1.0 (MEASURED in `SP/jump_wf/inner_injection.md`) | already flat |
| 14 | `SLOT_PRIOR_TABLE` (c05:96, used c13:13) | per-label slot prior buffer inside the DINO head | yes | yes (part of trained members) | architecture prior baked into the checkpoints | no |
| 15 | ours: `K2_CFG['W_MAIN'] = .50` (c31 line 1) | outer parent + fleet rank blend | no | yes | chosen after 2 public reads (W .25 .945, W .50 .946); our binormal model says .4-.6 is flat | no (keep .50) |

Note: the parent has **no** inner CoAt/Raptor re-rank (receipt `inner_rerank: false`), unlike the 0.941 recipe.

---

## 2. Measurements behind the design (box mimic, gold58, MEASURED)

Inputs: `~/kmimic/runs/gold58_gpu` (the parent run on the 58 gold studies, 2 x T4 parity mimic). Scripts
`~/gold_prep/parent_audit/{flat_audit.py, flat_audit2.py, test_kf.py}` + copies of `inject_common.py` /
`inject_radcal_payload.py` from `~/k2_test/analysis_wf` (unchanged). Outputs `flat_audit_out.json`,
`flat_audit2_out.json`, `test_kf_mimic.log` (also copied to `SP/gold_prep/`). Runtime ~10 s CPU.

Brackets for the transformer side T (as in `inner_injection.md` sec 4): **in** = real stage 3, fitted on these 58
studies (favours T, i.e. favours flat); **oos** = pseudo-OOS stage 3 (held-out DINO fold + held-out Rad head per
study; too weak, favours the map). The Raptor/CoAt side is honestly OOS. Fleet proxies at W .50: the K2 F1 mix50
fleet (gold58 .914) and the K3 F1 + TTR-v4 fleet (.908); the K5 fleet has no stored gold58 predictions.

**2a. Rebuild exactness (the KF self-check).** Stored stage 3 + stored side with the parent map vs the stored final:
1.1e-16 (58 studies). Same check through `kf_patch_block.py` on the real Kaggle K5 commit outputs
`SP/k5/build/out_{main,solo,w50}_v1`: **0.0** (3 studies each).

**2b. Flat .60 minus map, macro AUC on gold58, paired bootstrap 90% CI:**

| level | bracket | d macro | CI90 | per-label d (ACL / MedMen / LatMen / LatOA / Fracture) |
|---|---|---|---|---|
| parent | in | +.0073 | [+.0036, +.0119] | +.001 / +.011 / +.051 / +.016 / +.008 |
| parent | oos | -.0055 | [-.0099, -.0018] | -.004 / -.007 / -.030 / -.019 / -.005 |
| blend W .50, F1 proxy | in | +.0040 | [+.0015, +.0069] | |
| blend W .50, F1 proxy | oos | -.0030 | [-.0060, -.0004] | |
| blend W .50, K3 proxy | in | +.0044 | [+.0018, +.0076] | |
| blend W .50, K3 proxy | oos | -.0016 | [-.0043, +.0007] | |

- The brackets disagree in sign, so gold58 cannot decide. The one public anchor (Tian -.002 at the parent level)
  sits inside the bracket, nearer the oos end, as for earlier dilution anchors.
- Blend dilution (blend d / parent d): .55-.60 in, .29-.55 oos. Rank movement: parent MARS .0123, blend .0077-.0088.
- About half of the effect is Lateral Meniscus alone: 46-58% at the parent level, 45-51% at the blend level
  (`map_LM_only_flat` d / `flat60` d in the json; verifier correction 10-06, the first draft said 60-70%).
- Paired SD of the difference, scaled from 58 to 390 studies: parent ~.0009-.0010, blend ~.0006-.0007.

**2c. Why the map picked those labels (component AUC on gold58, T_oos minus side):** ACL -.051, MedMen -.085, LatMen
-.126, LatOA -.024, Fracture -.045; unmapped labels -.009 to -.035, except MCL -.117 (9 positives) and PF OA +.017.
Spearman(T, side) is lowest on LatMen (.80 oos / .84 in), LatOA (.78 / .85) and Fracture (.80 / .84). So the map
partly follows where T is weakest (both menisci); Lateral OA has no such support. Moderate structural support, not
proof (58 studies).

**2d. Similarity of the hedge to pick 1 (blend W .50):** per-label Spearman .987-.998 on the 5 changed labels,
1.000 on the other 7; mean over 12 .997-.998. Max per-study rank shift at the parent level on 58 studies: LatMen .26,
LatOA .10, Fracture .10, ACL .09, MedMen .07.

---

## 3. The flat hedge (KF): design

### 3.1 What to flatten
- **Only the cell 28 outer map, to .60 for all 12 labels.** .60 is the pre-probe value (the lineage's own "parent"
  preset) and the only flat value with public reads (0.939 lineage, Tian 0.940). The mean .6875 or a "halfway" map
  would be new, unread constants; both land between flat and map on the mimic.
- Keep everything else (rows 2-15): not documented as LB-tuned per label, global 1-df scalars, structural, or not
  rebuildable from what the parent persists.

### 3.2 Implementation (K-kit cell only; parent cells 0-30 untouched)
Code ready in `SP/gold_prep/kf_patch_block.py` (tested by `SP/gold_prep/test_kf.py`). It has 4 parts:
1. **K2 default:** `'OUTER_FLAT': None`. The builder writes `'OUTER_FLAT': 0.60` into the `K2_CFG` line of the hedge
   kernel only. With None, every output is byte-identical to K5.
2. **Block `_kf_outer_flat(pub, flat, rec)`:**
   - loads `btk_v32_rest_input.npz` (cell 28's exact `_ke_ours`) and `_raptor.csv` (the side). Cell 30's cleanup
     removes only pixel caches, so both survive.
   - checks labels, ids, finiteness, and that the live cell-28 global `_coatnet_weight` equals the known map.
   - (verifier fix 10-06) requires stage 3 to equal cell 26's `diagnostics/btkd_parent_after_rad.npz` (the in-memory
     stage 3 that cell 26 saved right before writing `_pipeline_stage.csv`) to <= 1e-9 on all 12 labels. Without it,
     a wrong stage-3 Lateral Meniscus column passes the rebuild check (map w 1.00 ignores it) and feeds the flat
     parent; demonstrated by a perturbation test that passed before the fix and raises after it.
   - rebuilds with the parent map and requires max abs diff <= 1e-9 against `_k2_parent.csv`.
   - returns `rank((1-.6) rank(S3) + .6 side)` and writes a receipt `rec['outer_flat']` (flat, map, rebuild diff,
     MARS, per-label rho).
   - any failure raises `_K2Fail`. In strict mode that deletes `submission.csv` (fail-loud, never the parent).
3. **Two call lines in `_k2_stage`:**
   - right after the parent validation: `r_pub_src = pub[...]`, replaced by the KF output when OUTER_FLAT is set.
   - `r_pub = _k2_rank(r_pub_src)` replaces `_k2_rank(pub[...])`.

   It runs before the workers start, so a KF failure costs no fleet time. The file set (fleetsolo, wXX,
   submission.csv), the rho sanity gate and `_k2_parent.csv` (true parent bytes) are unchanged.
4. **Status line:** `outer_flat=K2.get('OUTER_FLAT')` in `_k2_stage_fleet.json`.

Runtime: rank operations on n x 12, under 1 s. No GPU.

**k2_check.py extension (needed before any submit):**
- `--expect_outer_flat 0.60`: the config, status and receipt must all say .60.
- Recompute independently from the commit output: rebuild with the hard-coded map must equal `_k2_parent.csv`.
- Then every `submission_wXX.csv` must equal `rank((1-W) rank(P_flat) + W rank(fleetsolo))`, with the flat parent
  in place of the parent file.
- A K5-style check on a KF kernel without this extension FAILS the blend-recompute lines only when the flat and map
  parents rank differently (e.g. the 58-study mimic). Verifier note 10-06: on the 3-study Kaggle commit (and the
  2-study A0 BOXSTAGE) they rank identically (max shift 0.0 on all three K5 outputs), so the unextended check PASSES
  a KF kernel and equally passes a kernel where KF silently did not run. The extension must therefore gate on the
  config + status + receipt (`rec['outer_flat']` present, flat .60, `rebuild_max_abs` and
  `stage3_vs_cell26_max_abs` <= 1e-9, `changed_labels` = the 5 mapped labels), not on the recompute alone.

**Kit:** copy the current pick-1 kit (k6/k7) to `SP/k8f`. Use the same MEMBERS, TTA, W_MAIN .50 and MAIN_MODE
strict, plus `'OUTER_FLAT': 0.60`. The builder post-check must assert that key on the hedge kernel and its absence
on the others.

### 3.3 Why not also flatten the calibrator gate or `_RAD_EXCLUDE`
- Both act inside cell 26 on stage 2, which the parent does not persist: `_pipeline_stage.csv` is overwritten.
- Rebuilding stage 2 from the DINO npz is not exact. Stage 1 has rank-sum ties broken by thread order: one rank step
  on 23 ties in 58 studies (`inner_injection.md` sec 2). So the strict self-check would have to be loosened, or
  cell 26 edited, which breaks the "parent cells untouched" rule.
- Neither is documented as LB-tuned.
- The calibrator is mostly a per-label reweighting dominated by each label's own rank; Tian rated it as fragile only
  to a protocol-mix shift, and its protocol coefficients are tiny.

Expected value of flattening them: ~0, with unmeasured risk. Leave both.

---

## 4. Offline checks: done and still to do

Done (MEASURED, sec 2): rebuild exactness on 3 Kaggle outputs + mimic; negative tests (map drift, parent-byte
mismatch, flat out of range, id mismatch, and after the verifier fix stage-3 LatMen perturbation and missing cell-26
diagnostics, all raise); movement and rho tables; two-bracket gold58 comparison with two fleet proxies. Verifier
re-run 10-06: `flat_audit_out.json`, `flat_audit2_out.json` and `test_kf_mimic.log` reproduce byte for byte
(`~/gold_prep/parent_audit/verify/`). Note: the first-draft `test_kf_mimic.log` shows
`untouched_labels_identical: false`; that was an exact `== 0` test against a 1.1e-16 box CSV parse difference, not a
defect (fixed to a 1e-12 tolerance; Kaggle outputs give exactly 0.0).

To do before a KF slot (box mimic / kits, CPU, ~20-30 min wall):
1. Build the KF kernel, then run A0 BOXSTAGE (2 studies, full notebook on CPU, as K5). This exercises the live
   `_coatnet_weight` check inside a real cell-28 namespace, the strict delete path and the receipt. Then run
   K2CHECK with `--expect_outer_flat 0.60`.
2. Adversarial A0 runs:
   - remove `btk_v32_rest_input.npz` before cell 31 -> must end with `submission.csv` ABSENT (strict);
   - perturb one `_raptor.csv` cell -> the rebuild self-check must fail.
3. DONE by the verifier (10-06, `SP/gold_prep/verify_parentaudit/kf_e2e_test.py`, box copy
   `~/gold_prep/parent_audit/verify/kf_e2e_mimic.log`): the KF block spliced into the real K5 w50 cell 31, stub
   fleet = the K3 G58 fleetsolo, stub torch (no GPU). Clean stage; every wXX equals the flat-parent recompute;
   submission.csv AUC .93378 (map) -> .93819 (KF), exactly `flat_audit.py`'s `blendW50_K3` in-sample numbers. The
   same splice also applies cleanly to the k6 and k7 kit cells (`kf_integration_test.py`, 11 cases, incl. wrong side
   file, shuffled side rows, duplicate side row, stage-3 LatMen perturbation, missing npz).
4. After the Kaggle commit (3 studies): K2CHECK with the extension. With 3 studies the flat and map parents rank
   identically, so the commit proves plumbing only; the numerics are covered by item 3 and sec 2.

---

## 5. Expected public change and the private argument

**Public (MODEL):** parent-level flat effect -.002 (MEASURED by Tian on the same graph family; true value -.001 to
-.003 given display rounding), times a blend dilution of .3-.6 (MEASURED) = **-.0006 to -.0012, central -.001**.
- K5 w50 reads .946, so K5 w50 + KF most likely reads .945. Verifier correction 10-06 (MODEL: K5 w50 true score
  uniform in [.9455, .9465), shift uniform over the stated range): with the central range -.0006 to -.0012 the odds are
  .945 ~84%, .946 ~13%, .944 ~3%; with the wide range -.0018 to 0 they are .945 ~54%, .946 ~28%, .944 ~18%. The first
  draft's ".946 ~35% and .944 ~5%" fits neither; .944 is a real possibility and stays inside the OK band.
- The SE of an absolute public score is .004-.006, but the two variants share ~99.8% of their ordering. The SE of
  their difference is ~.0006-.0007 (mimic bootstrap scaled to 390), plus +-.0005 display rounding on each read.
- So a .001 drop on the board is the expected price, not evidence that the hedge is worse.

**Private (GUESS, reasoned):**
- **Selection:** the map was picked from a probe series (at least #15 and #22) on ~390 public studies, where one
  variant-vs-variant difference has SD ~.001 at the parent level. The best of several probes gains ~1-2 SD from noise
  alone, so the +.002 is mostly winner's curse. The meniscus-specific weakness of T on gold58 (sec 2c) suggests part
  is real. Estimate: true parent-level value +.0005 to +.001, i.e. +.0003 to +.0006 at the blend level.
- **Tail risk of the map:** LatMen 1.00 drops three of the four stages for one label. If T's LatMen signal is
  better on the private ~910 studies than on public, that label loses up to ~.02-.03 AUC at the parent level
  (the in-sample bracket magnitude): about -.001 to -.002 macro at the blend level. The flat version cannot lose
  that way.
- **Net:** E[private(flat) - private(map)] ~ -.0003, range about -.001 to +.001 at the blend level. That is
  near-free insurance with a small premium, and it is small next to the ~.004-.006 private SE around the medal
  line.
- **Hedge value is limited by similarity.** The flat copy of pick 1 has mean rho .998 with it, so both picks medal
  or miss together in almost every scenario. The hedge only adds value if pick 2 also differs in members.

---

## 6. Recommendation for the final 2 picks

1. **Pick 1:** best public blend, parent map unchanged, W in .4-.6 (as planned).
2. **Pick 2:** OUTER_FLAT .60 with no other per-label public-tuned map, W .50, on top of the strongest member set
   that differs from pick 1:
   - if a new family or relabel set exists by ~10-20 (R50 simple-pool, TTR288, relabeled TTR), use it;
   - otherwise the K-best members (then pick 2 is the pure flat copy of pick 1: low cost, low value).
3. Build and gate KF now (sec 3.2 + sec 4: ~1-2 h laptop and box CPU, 0 GPU). Spend **at most one** public slot,
   as a plumbing read with a pre-registered band: **.944-.946 = expected / OK; <= .943 = investigate the rebuild
   before use**.
   Verifier note 10-06: a final pick must itself be a scored submission, so if pick 2 is KF on a different member set,
   build KF directly on that set and make THAT the one KF slot (its read is also the plumbing check). A separate
   KF-on-K5 plumbing read costs a second slot and is only worth it if pick 2 is the flat copy of pick 1. The builder
   must also assert MAIN_MODE 'strict' on any kernel with OUTER_FLAT set: in 'blend' mode a KF failure restores the
   parent bytes and the slot silently scores the bare parent.
4. Do not tune anything else per label on the public LB (no halfway map, no LatMen-only variant chosen from a read);
   the gold58 per-label numbers above are direction-only and contradict each other across brackets.
5. Housekeeping: `SP` is Temp. Copy `SP/gold_prep/{parent_audit.md, kf_patch_block.py, test_kf.py, flat_audit*.py,
   flat_audit*_out.json, test_kf_mimic.log}` into `rsna-knee/session_2026-10-06/` before Temp is cleaned.
