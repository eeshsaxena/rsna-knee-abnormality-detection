# Verify D: training our own models with OAI data (oai_own), 2026-10-08 ~23:00 IST

Adversarial check of `research_1008/D_oai_own_training.md` and its structured result. Read-only: nothing on Kaggle, no paid
API, no Vast, nothing written on the box. Tags: **MEASURED** (source given), **MODEL**, **GUESS**.

**Box:** still unreachable from the laptop (`ssh ... ConnectTimeout=6` timed out, MEASURED 22:55 IST). So I checked the
trainer against the local copy with the SAME md5 the report cites: `fast_build/code2/fleet_train.py` md5 `3e62434a...`
in the old scratchpad (MEASURED; `argv_frac75.sh` names this md5 as the frac75 trainer). `fleet_common.py` was checked in
`session_2026-10-08/k8/code/`. The label CSV `labels_frac75_logit.csv` md5 `ad7582b8...` is the one `argv_frac75.sh` names
(MEASURED).

## Headline corrections

1. **The claimed gain (+.001 to +.003 over K9N) is NOT supported, not even by the D author's own model.** The scratchpad file
   `oai_gain_model.out` (the D research's binormal model, same method as the K9N weight model) gives **K9N-a gain range
   -.0005 to +.0007, median +.0001** over 27 central scenarios (MODEL, MEASURED from the file). The optimistic
   "9-label, gains mostly on labels N lacks" scenarios in `oai_gain_model_upper.out` give **+.0008 to +.0014** (MODEL). The
   report's +.001 to +.003 is above even the upper scenarios.
   - Why so small: K11N swaps only 5 of the 10 fleet members, so a set-level gain is roughly halved at the fleet level; F has
     weight .30 next to N at .60; and our OAI members become more correlated with N (the `dr` term), which eats the gain
     (MODEL).
   - **Corrected gain: about +.0001 to +.0004 central, range -.0005 to +.0014 (MODEL).** This is below what the public LB
     can resolve (NOVEL_PLAN: "The LB cannot resolve +.001"; nartaa: rounded +.001 differences "do not establish
     statistical significance", MEASURED).
2. **The gain only counts in ONE of the 2 final picks.** OAI-trained members carry the same rule exposure as N, so they can
   only go into the OAI/N pick. The clean hedge pick cannot use them (MEASURED policy: PROGRESS 13:00, "keep at least one
   clean final pick"). In contrast, the queued 320 set or the Q4 288 frac75 reseed helps both picks. The report compares
   OAI with Q4 as if they were alternatives in the same pick. They are not.
3. **Probe cost is ~5.8 h, not 4.7 h.** The paired control `f0_s1043_ttr_frac75` was trained with **24 epochs and
   `--grad_ckpt 1`** (MEASURED, `argv_frac75.sh`: fold 0 = 24 ep, gc1; folds 1-4 = 21 ep, gc0). A paired probe must copy
   that argv. At 11.3 min/epoch for gc1 (MEASURED, NOVEL_PLAN l.178) x1.28 (ext_frac .5, report's own factor) x 24 epochs
   that is ~5.8 h (MODEL). The report's own TL;DR says ~5.6 h, which contradicts its table and finding (4.7 h). The
   5-fold total barely changes (~24-25 h, MODEL).
4. **`--mask_norm cells` should NOT be the probe default.** nartaa trained with "batch size 1 and gradient accumulation
   over 16 studies" (MEASURED, show746792.txt). With bs 1, a per-row masked mean (the natural implementation, and what our
   `finite` default does) gives each OAI cell 4x weight on a 3-label row. So his only LB-evidenced recipe most likely HAD
   the 4x per-cell weight (GUESS, his code for the OAI arm is not public). `cells` is an untested change. The probe should
   use the current default (`finite`, zero code change). `cells` can only be a second probe if the first one passes.
5. **Rule-risk date typo:** the structured result says "Eesh decision 10-25". The evidence is PROGRESS 10-08 20:35 (NDA
   "Institutional Approval Required: No") and 21:20 (DAR 26647 approved). The host has still not answered Taeyang's yes/no
   (743416, MEASURED FORUM_DIGEST).

## What checked out (MEASURED)

| Claim | Check | Result |
|---|---|---|
| Masked BCE at l.363-364 | local copy md5 3e62434a | `msk = torch.isfinite(y)`, sum / `msk.sum().clamp(min=1)`: exact |
| NaN-masked AUC (T9) | `macro_auc` l.288-296 | exact |
| fold 9 trains in every fold | l.132-136: `held = folds<0 | excl`; `tr_ids = folds[s] != A.fold` | exact |
| absent views zero-filled | default `__getitem__` and `fleet_aug.heavy_study` l.76 | `zeros`, `present False`, ctr 3: exact |
| decode probe = `sorted(ids)[:4]`; contract check on `sorted(ids)[:8]` | l.155-166 | exact |
| probe npz ship to Kaggle `parity/` | `fleet_common.decode_sha` docstring | exact |
| `oai_` ids sort after RSNA ids | RSNA ids all start `1.2.826...`; `'oai_x' > max(id)` | True |
| frac75 labels contain NaN? (report open item 4) | 4,407 x 12, **0 NaN**, values in [.006, 1.0] | so `--mask_norm cells` is identical on RSNA-only batches |
| `--init_ckpt` only with `--f2 1`; `--f2` banned with `--model ttr` | l.98-105 | exact |
| `*.dcm` glob in `load_series` / `_plane_from_header`; multi-frame REJECTED | fleet_common l.27, 123, 173 | exact (OAI legacy single-frame DICOM should pass, GUESS) |
| `contrast_class`: DESS (TE ~5, TR ~16) -> 0; IW TSE FS (TE 30, TR 3200) -> 1 | l.53-67 | exact |
| all 44 MOAKS/WORMS/BLOKS field names in the label map | `oai/oai_kmrisemiquant01.json` (223 elements) | 0 missing; `side` 0-3 (3 = bilateral); `visit` is a string ("Baseline"); value fields are Float with NO code notes |
| only MCL field is `w_mcl_tear` | same | exact |
| `mexamtp` series list incl. SAG IW TSE FS, COR MPR, AXIAL MPR; `qcreslt` N/NR/Y/YD | `oai/oai_mrimeta01.json` | exact |
| downloadcmd `-dp`, `-t/--txt`, `--file-regex`, `-wt`, `--verify` | `downloadcmd.py` (GitHub main copy) | exact; still re-check `-h` on the box's 0.7.0 |
| nartaa anchor .940 -> .945 with OAI, 2,399 baseline knees, T1 labels only, updates 4,336 -> 6,736, 12-month arm "unsuccessful" | show746792.txt | exact; he also says the serving precision path changed in that step, so it is a confounded recipe comparison |
| epoch-time arithmetic (x1.69, x1.35, 16.3 / 13.4 / 25.4 min) | recomputed | correct given the 80 % train-share GUESS; the 320 log (train ~16 min of ~21-25 min) suggests 65-75 %, which makes OAI slightly cheaper |

## Per-finding verdicts

### F1. Masked-label training needs no loss-code change; opt-in patch advised: **KEEP**
Every code claim checks out on the md5-matched copy. New fact: the frac75 labels have 0 NaN, so the RSNA-only batch path is
unchanged under either norm. Small corrections: the up-to-4x weight is for T1-only rows (3 labels); for T1+T2 rows (6
labels) it is 2x (arithmetic). Make `finite` (today's behaviour) the probe default (headline 4). Gain: enabler, 0 by itself.

### F2. Selective NDA download (metadata first, then `-t links.txt`): **KEEP, confidence medium (unchanged)**
Flags and table names verified. The size (0.15 vs 0.5-1.4 TB) and whether image03 rows are per-series files are still
GUESS, and the whole plan depends on the latter. Practical risk the report lists only in passing: the box is unreachable
from the laptop's current network (MEASURED tonight), and the run needs Eesh's interactive NDA password. Do step A
(metadata only, minutes) before committing any GPU plan.

### F3. Pre-registered 224 fold-0 OAI probe then folds 1-4: **REVISE**
- Gain: **+.0001 to +.0004 central, -.0005 to +.0014 range over K9N (MODEL, the D author's own model output)**, not
  +.001 to +.003. It is unresolvable on the public LB, and only in the OAI/N final pick.
- Cost: probe ~5.8 h (24 ep gc1 to pair), set ~24-25 h V100 (MODEL). Feasible before 10-20 only if NDA step A happens
  10-09 and the download is not the 1.4 TB case; it lands ~10-12 at the earliest (after Q3b ~10-10 22:00).
- Gate power is low: G2 asks gold74 mean over 6 OAI labels >= paired + .005, where the gold74 SE of that mean is
  ~.01 (GUESS). A true +.005 passes about half the time, and a null passes ~30 % (GUESS). Weight the OOF leg more, and
  pre-register `finite` norm.
- Priority: below the 320 set (Q3b) and below the Q4 frac75 reseed (both help BOTH final picks). Run it only if the V100
  would otherwise idle, or if the host says yes to OAI (then both picks may carry OAI and the value roughly doubles).

### F4. Label map T1 / T2 / T3, mask MCL, Contusion, Fracture: **REVISE (confidence medium -> low-medium)**
Field names are all real (MEASURED). But the dictionary has NO value codes for these fields, so every threshold and soft
mapping is GUESS until the value counts are read. The premise "T2 adds signal N lacks" is a hypothesis: nartaa supervised
only 3 labels and does not say why (MEASURED). `m_eff_wk` drives both Effusion and the optional Synovitis mapping, which
couples the two targets. Synovitis is our weakest label for every source (.79-.83 target AUC, MEASURED PROGRESS 20:10),
so a wrong Hoffa mapping could hurt where we are weakest. ACL tears are rare in an OA cohort (GUESS), so T3 ACL may
add almost nothing. Gain: carried by F3, so 0 here.

### F5. Compliance trap: parity npz ship to Kaggle; OAI never in ~/fleet_store, never on T4/Vast: **KEEP**
Verified line for line (probe `sorted(ids)[:4]`, contract `[:8]`, `decode_sha` docstring). The `oai_` prefix sorts after
every RSNA id (MEASURED). The assert and the symlink union dir are correct and cheap. Add one more check to the patch tests:
the ckpt `args` and any saved OOF/gold prediction file contain no OAI ids or OAI-derived per-row values.

### F6. Cheaper variant: fine-tune existing frac75 224 folds on RSNA+OAI (~7 h): **REVISE**
The `--init_ckpt` constraint is verified (needs a new code path). Gain: under the D author's own model a fleet-level
dF <= .0015 gives at most about +.0004 over K9N; with members staying near set D, use **+0 to +.0004 (MODEL/GUESS)**, not
+.0005 to +.0015. Keep on hold.

### F7. Phase 2 X-ray KL/JSN for ~9,000 knees: **KEEP (do not pursue before the freeze)**
X-ray fields verified (MEASURED). Not feasible before 10-20. Gain unknown and likely small.

## Levers this dimension MISSED

1. **Pick-level accounting.** OAI members only add value to the OAI/N pick. 320 and reseed sets add value to both picks.
   That should set the V100 order: Q3b (320) or Q4 first, OAI last, unless the host says yes.
2. **Wait for the host answer before burning GPU.** Taeyang's yes/no (743416, 10-08) is pending. Do the cheap NDA metadata
   step and the CPU prep now, but run the 24-25 V100 h only after a host yes or when the V100 would idle.
3. **OAI as a big external VALIDATION set, not a training set.** About 2,400 image-read MOAKS knees (GUESS count) could
   score our members per label (Medial OA, Effusion, Baker's, menisci) where gold74 is underpowered. It needs only
   inference on the box (~1-2 V100 h, GUESS). Caveats: domain shift, and it is still OAI use, so only for the N pick. It
   must not steer the clean pick.
4. **Free Kaggle T4 quota for the clean pick.** OAI cannot use T4 (data stays on the box), so the ~85 T4 h should go to
   clean-pick levers in parallel: the 224 "fp" PF OA relabel (PROGRESS 20:10) and 224 reseeds.
5. **Match nartaa's evidenced recipe first:** `finite` norm (bs 1 equivalent), baseline visit only (his 12-month arm was
   "unsuccessful", MEASURED), T1 labels only in the first probe. Add T2 only as a second, separately gated step, so the
   probe answers one question.
6. **N-as-teacher (mixN) labels** (F_intel R3) need no OAI download and target the same T1 labels. It is also
   OAI-dependent, so it belongs to the N pick only. It is a cheaper way to get OAI signal into 224 members than our own
   OAI pipeline.
7. **Runtime guard.** K11N must REPLACE the 224 set, not add to it. K9N has only 0.65-1.2 h margin to the 8.6 h wall
   (MEASURED, PROGRESS 11:35).
