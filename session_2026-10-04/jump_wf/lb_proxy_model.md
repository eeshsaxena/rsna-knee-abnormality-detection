# LB proxy model: what our fleet must reach to lift the 0.943 parent

Built 2026-10-04 ~15:00-15:25 IST, before the K2 reads (~17:50 IST). Box scripts and outputs are in
`~/k2_test/analysis_wf/proxy_*` (CPU, nice 19). Laptop copies are in `jump_wf/proxy_*.py` and `jump_wf/proxy_box_out/`.
Tags: **MEASURED** means computed by me or read from a log or file (path given). **MODEL** means derived from the
stated binormal model. **GUESS** means a judgment call.

---

## 0. Bottom line

1. **The binormal blend model works on gold58 once r is the *effective* correlation.** If you plug in the latent
   (Spearman-implied) correlation, the model over-predicts every gold58 blend by +.0025 on average (420 pairs, corr
   .988, MEASURED `proxy_gold58.log`). This is the same "gold58 proxy was optimistic" pattern we saw before. Using
   r_eff = r_latent + .062 removes the bias: leave-one-base-out MAE is .0010 (MEASURED `proxy_fit.log`). On the four
   context curves (fleet into the raptor v5 arm, raptor_public, coat_family and raptor_coat_side) it hits MAE .0006
   over 20 points. Every error is under 0.85 bootstrap SE (MEASURED `proxy_ctx4.log`).
2. **The fleet member is parent-like.** On gold58, its effective r against the parent's out-of-sample Raptor+CoAt
   side is .918. Against raptor_public it is .877 and against coat_family .887 (MEASURED `proxy_fit.log`). The crop
   members sit at .78-.82 against the same references. So the fleet is about .10-.14 more parent-like than the crop
   leg that already failed on the LB.
3. **On the LB, effective correlations are much higher than on gold58.** The crop LB reads imply r_LB of about .88-.94
   for the crop members (least-squares values), if their solos are .915-.925 (MODEL `proxy_lb.log` A). Tian's member gives a similar
   inflation (external). Carrying that inflation ratio over to the fleet gives fleet r_LB of about .96, with central
   values .94-.97 (MODEL `proxy_k2table.log`). The optimistic "gold58 is representative" view gives about .90.
   **Working prior: r_LB(fleet) central .95, range .90-.97 (MODEL+GUESS).**
4. **Prediction for the K2 BLEND (W .30) read: median .9400. P(>= .943) is about .15, P(>= .942) about .27 and
   P(<= .939) about .41** (MODEL+GUESS priors, `proxy_k2prior.log`). The pre-registered ".940-.943 flat" is slightly
   optimistic. The model's mode is .939-.941.
5. **K2 SOLO prior: about .922 +- .008 (GUESS).** This uses the Dread-like gold58-to-LB offset of +.007..+.011 on top
   of the fleet's gold58 .9144. If the read comes in at .93 or above, gold58 under-rates the fleet the way it
   under-rates the crop. That would make the HIGH-r regime more likely.
6. **To lift the parent, the ensemble needs to be about as strong as the parent at a low r** (MODEL `proxy_lb.log` F).

   | Target | Minimum ensemble solo |
   |---|---|
   | +.003 | .9255 at r .80 · .933 at .85 · .9395 at .90 · .944 at .94 |
   | +.007 | .938 at r .80 · .9435 at .85 · .9475 at .90 · .950 at .94 or above |

7. **Ensembling our own fleet does not fix r.** k members of one family raise solo by only +.003..+.006 (k = 5-10,
   effective member-to-member r .93-.95, MEASURED analogs). Averaging also *raises* r with the parent
   (r_ens = r_m / sqrt(v_k)). At member r_LB of .94 or above, a 10-member fleet adds +.000 at every solo row
   (MODEL `proxy_k2table.log`).
8. **A +.007 jump (to .950) from our own fleet is not plausible.** By 10-12: about 1-3% (GUESS). By 10-22: about
   2-5% (GUESS). It would need an ensemble of about .938 or better at r_eff about .80 or lower. Our best own LB solo so
   far is .924, and the fleet is the most parent-like thing we have built. The realistic own-fleet ceiling is
   +.000..+.002 (MODEL). The team merge stays the only lever with a visible path to .950 or above.
9. **What tonight's reads decide.** SOLO s plus BLEND b pins the fleet's r_LB to about +-.025 (see the inversion table
   in section 3.3).
   - b >= .943 with s .915-.925: r ~ .79-.90, the LOW regime. The own-fleet route is alive for about +.002-.004 if the
     ensemble reaches ~.935-.94.
   - b = .940-.942: r ~ .82-.97 depending on s, and the own route caps near +.001.
   - b <= .939 with s >= .915: r >= .89, and the own route is dead for lifting the parent.

---

## 1. gold58 measurements (step 1)

Data: 58 gold studies. Labels come from `~/kmimic/kaggle/input/.../train.csv` (non-NaN ACL rows that are in the fleet
serve file). Parent per-arm predictions come from `~/kmimic/runs/gold58_gpu/localLB_parts/`. Fleet = `submission_fleetsolo.csv`
(serve path). Crop members = laneA `gold74_preds_laneA.npz` (gold58-selected checkpoints), copied to
`~/k2_test/analysis_wf/proxy_crop_gold74_preds.npz`. Dread widedense = `~/rsna_train/teacher_train_probs.csv`.
Positives per label range from 9 (MCL) to 35 (Effusion).

### 1.1 Macro AUC on gold58 (MEASURED `proxy_gold58.log`)

| Arm | gold58 | Type |
|---|---|---|
| raptor_arm maxspan-v5 / v5-reverse / native384-v8 / native384dense-v10 | .9221 / .9180 / .9165 / .9190 | OOS |
| raptor_public_rank | .9254 | OOS |
| raptor_coat_side_rank | .9353 | OOS (best out-of-sample half of the parent) |
| coat_family_rank / coat_d4 / coat_global96 / coat_repairv1 / coat_resgated | .9315 / .9290 / .9303 / .9103 / .9102 | OOS |
| a5_foldmean | .8713 | OOS |
| pipeline_stage_1_dino_stack / rad e10 / rad e13 | .9956 / .9722 / .9839 | IN-SAMPLE (contaminated) |
| parent_final | .9509 | partly in-sample |
| **fleet f0 mix50 (K2 member)** | **.9144** (bootstrap SE .0142) | clean |
| crop cropaug_f1 / cropaug_f2 / crop_f1 / mixSA_f1 / K1 ens3 | .8904 / .8824 / .8730 / .8809 / .8884 | gold58-selected |
| dread_widedense (teacher file) | .9127 (SE .0136) | OOS; context gives LB .924 |

Bootstrap SE of a gold58 macro AUC is about .012-.014, and the median paired-bootstrap SE of a gold58 blend delta is
.0034 (MEASURED). Every gold58 number above is therefore noisy at the +-.01 level.

### 1.2 Correlations of the fleet member (gold58, mean over 12 labels; MEASURED `proxy_gold58.log`, `proxy_fit.log`)

rA is the latent within-class r from Monte-Carlo inversion of the mixture Spearman, using each label's AUCs and
prevalence. rB is the direct within-class Spearman (pooled Fisher-z, converted to Pearson). The two agree within
.01 everywhere. r_fit is the macro-binormal r that best fits the measured blend curve (W .10-.50).

| fleet vs | Spearman | rA | rB | r_fit (effective) |
|---|---|---|---|---|
| raptor arm maxspan-v5 | .893 | .810 | .807 | .875 |
| raptor arms v5r / v8 / v10 | .889 / .887 / .889 | .806 / .799 / .803 | | |
| raptor_public_rank | .903 | .827 | .823 | .877 |
| coat_family_rank | .902 | .828 | .823 | .887 |
| raptor_coat_side_rank | .921 | .858 | .850 | .918 |
| parent_final (contaminated) | .924 | .868 | .863 | .943 |
| dino stack (in-sample) | .815 | .660 | .666 | |
| dread_widedense | .882 | .793 | .796 | |
| crop K1 ens3 | .829 | .719 | .713 | |

For comparison, crop K1 ens3 has r_fit .820 vs raptor_coat_side, .787 vs raptor_public and .851 vs parent_final.
cropaug_f1 has .783 / .736 / .834, and mixSA_f1 has .852 / .820 / .900 (MEASURED `proxy_fit.log`). The fleet is the
most parent-like own member we have measured. It is close to Dread's own widedense (.938 vs raptor_coat_side).

### 1.3 Validation of the binormal two-reader blend model

Model: AUC_i = Phi(d_i/sqrt2), with two scores per class correlated by r. A blend at weight W has
d = (1-W)d1 + W d2 and noise variance v = (1-W)^2 + W^2 + 2W(1-W)r, giving AUC = Phi(d / sqrt(2v)). There are three
variants. The per-label analytic version uses per-label AUCs and per-label rA. The per-label Monte-Carlo version does
an exact *rank* blend under the binormal model. The macro version uses one d and the mean r. All three give the same
answer within .0005, so a rank blend is well approximated by a latent-score blend.

**The four context curves** (delta macro AUC on gold58, global rank blend, fleet weight W). Columns are: measured,
then the model with latent rA, then the model with effective r = rA + .062 (MEASURED / MODEL, `proxy_gold58.log`,
`proxy_ctx4.log`):

| Base (gold58) | W .10 | W .15 | W .25 | W .35 | W .50 |
|---|---|---|---|---|---|
| raptor v5 arm (.9221), rA .810 | +.0008 / +.0024 / +.0017 | +.0016 / +.0034 / +.0023 | +.0027 / +.0048 / +.0032 | +.0036 / +.0056 / +.0036 | +.0036 / +.0054 / +.0031 |
| raptor_public (.9254), rA .827 | +.0015 / +.0017 / +.0010 | +.0025 / +.0024 / +.0013 | +.0024 / +.0033 / +.0017 | +.0021 / +.0036 / +.0016 | +.0004 / +.0030 / +.0005 |
| coat_family (.9315), rA .828 | +.0010 / +.0012 / +.0004 | +.0012 / +.0015 / +.0005 | +.0005 / +.0019 / +.0002 | -.0009 / +.0016 / -.0005 | -.0025 / -.0000 / -.0024 |
| raptor_coat_side (.9353), rA .858 | +.0002 / +.0003 / -.0005 | -.0001 / +.0003 / -.0009 | -.0008 / -.0001 / -.0018 | -.0026 / -.0009 / -.0031 | -.0074 / -.0032 / -.0058 |

Bootstrap SE of the measured points is .0009-.0047. Over the 20 points, latent rA gives MAE .0021 with bias +.0021,
and rA + .062 gives MAE .0006 with bias -.0002. The worst error is 0.85 SE (MEASURED).

**Broad check.** I ran every ordered pair of 15 arms (12 OOS parent arms, the fleet, crop K1 and widedense) at
W .25/.50, for n = 420. With latent r: MAE .0027, bias +.0027, corr .988, sign agreement .94.
- Effective-r offset (r_fit - rA): mean +.062, sd .026, over 205 interior fits.
- Linear fit: r_fit = .833 rA + .193.
- Noise-ratio form: (1 - r_fit) = .686 (1 - rA).
- Leave-one-base-out CV: rA alone MAE .0025, rA + offset MAE .0010 (bias +.0002), noise-ratio form MAE .0011
  (MEASURED `proxy_fit.log`).

**Conclusion:** the structure holds (corr .99). The blend gain is smaller than the Spearman-implied latent r suggests,
probably because the errors that decide AUC are more shared in the tails. Always use the effective r.

---

## 2. LB calibration (step 2)

### 2.1 Anchors and implied effective r (MODEL `proxy_lb.log` A)

All values are displayed public LB to 3 decimals. The parent's true value is taken in [.9425, .9435] and reads are
+-.0005.

| Anchor | Reads | Implied r_LB by assumed solo |
|---|---|---|
| C1 cropaug_f1 single, K=24 served | W.15 .942, W.25 .941 | solo .905: .81-.86 · .910: .83-.88 · .915: .86-.91 · **.920: .89-.94 (LS .911)** · .925: .91-.96 · .930: .95-.995 |
| C2 mixSA_f1 single | W.25 .940 | .905: .84-.88 · .910: .86-.91 · **.915: .89-.94** · .920: .91-.96 · .925: .94-.99 |
| C3 K1 crop ens3, K=8 | W.25 .942 | .910: .81-.85 · .915: .83-.88 · .920: .86-.91 · **.925: .89-.94 (LS .911)** · .930: .92-.97 |
| X cnx M448 fold0, solo .894 (known) | parent W.50 .930 | **r .82-.86 (LS .837)** |
| X cnx on fast base (.942) | W.40 .935, W.50 .930 | **r .81-.83 (LS .817)**; .82-.84 if the fast base is .943 |
| external: Tian v09r .927 on a .942 fork | W.10 .942, W.20 .941 | r .91-.97 (LS .935); per notes, gold58 Spearman .835 |
| external CLAIM: aastik .928 on .943 | W.5/.6 .945 | **no r fits both reads** (LS .79). Inconsistent with the model, treated as unverified |

**Crop solo prior (GUESS):**
- cropaug_f1 as served is about .920 (.912-.926). Its sibling crop_f1 measured .924 solo LB, and K=24 serving was
  mismatched against K=8 training.
- mixSA is about .005 lower.
- K1 ens3 is about .925 (MODEL: 3 members at effective r_mm about .95 give +.003..+.004).

With those priors, all three crop anchors give r_LB of about .91 (range .84-.97). The cnx member, which was not
teacher-distilled, sits at .82-.84.

### 2.2 Effective parent strength

Section B of `proxy_lb.log`:
- The cnx curvature, with 2 reads on one base and a known solo, does **not** identify P_eff: all of P_eff .9417-.9552
  fit, with r .81-.999 (MODEL).
- Per-label heterogeneity: I re-ran the model per label, with the parent shaped like raptor_coat_side's gold58 label
  profile and the member shaped like the fleet's profile, each shifted to the target macro. This moves predicted
  deltas by a mean of +.0002, and by at most .0009 over 36 cases (MODEL).

**So use P_eff = .943, the displayed value.** Neither the anchors nor the label mix argue for a different effective
parent strength. This cannot be measured better with the anchors we have.

### 2.3 Transfer gold58 to LB for the fleet (MODEL `proxy_lb.log` C, `proxy_k2table.log`)

The fleet has no LB blend read yet, so its r_LB has to be carried over from gold58. The method is to take each crop
anchor's ratio k = (1 - r_LB(crop, parent)) / (1 - r_fit_gold58(crop, ref)), then apply it as
r_LB(fleet) = 1 - k (1 - r_fit_gold58(fleet, ref)). I used four references: raptor_coat_side, parent_final,
raptor_public and coat_family.
- At the central crop solos k is about .4. The fleet's r_LB central values are .939-.969, median .958.
- Across the crop-solo ranges they spread .89-.99.
- The additive form (r_LB = r_g58 + delta) saturates at .999, so I dropped it as ill-posed.

Independent support for k well below 1: Tian's member reads LB r about .935 against a gold58 Spearman of .835. On our
data, a Spearman of about .84 corresponds to r_fit about .82-.85, which gives k about .4 (MODEL+GUESS: his reference
arm is unknown).

Reasons the HIGH-r transfer could be too pessimistic (GUESS):
- (a) The crop members are odd on gold58: their LB-minus-gold58 offset is +.034..+.051, against +.007..+.011 for
  Dread widedense. So crop-specific gold58 failures may inflate their (1 - r_g58).
- (b) gold58 under-represents the parent's DINO half, which is in-sample there. On the LB that half is probably less
  fleet-like than Raptor+CoAt.

The optimistic "no transfer" view takes the fleet's gold58 r_fit against the OOS sides (.877-.918) at face value,
which gives r_LB about .90.

**Regimes used below:** LOW .90, MID .94, HIGH .97. The working prior is central .95, range .90-.97. I put about half
the weight on .93-.98 and half on .88-.93 (GUESS). Tonight's BLEND read replaces this prior with a measurement.

---

## 3. K2 conditional table (step 3)

### 3.1 Prior predictive of the pending reads (MODEL+GUESS, `proxy_k2prior.log`)

Priors: solo ~ N(.922, .008) (GUESS, from gold58 .9144 plus a Dread-like offset). r_LB is 50% U[.93,.98] and 50%
U[.88,.93]. The parent is U[.9425,.9435].

Displayed BLEND W.30 distribution: .937 .07 · .938 .12 · .939 .16 · **.940 .17** · .941 .16 · .942 .12 · .943 .08 ·
.944 .04. That gives P(>= .943) .15, P(>= .942) .27 and median .9400.

### 3.2 Table keyed on the K2 SOLO read (MODEL `proxy_k2table.log`; parent true .943)

| SOLO | Regime (r_LB) | Predicted BLEND W.30 | W* (lift) | 10-member fleet, r_mm .95: solo / r / lift | Ensemble solo needed at that r for +.003 / +.005 / +.007 |
|---|---|---|---|---|---|
| .905 | LOW .90 | .9372 (.937) | 0 (+.000) | .910 / .921 / +.000 | .942 / .946 / .949 |
| .905 | MID .94 | .9356 (.936) | 0 | .910 / .962 / +.000 | .9455 / .948 / .9505 |
| .905 | HIGH .97 | .9344 (.934) | 0 | .910 / .993 / +.000 | .9465 / .9485 / .9505 |
| .915 | LOW .90 | .9395 (.940) | 0 | .920 / .921 / +.000 | .942 / .946 / .949 |
| .915 | MID .94 | .9379 (.938) | 0 | .920 / .962 / +.000 | .9455 / .948 / .9505 |
| .915 | HIGH .97 | .9367 (.937) | 0 | .920 / .993 / +.000 | .9465 / .9485 / .9505 |
| .925 | LOW .90 | .9420 (.942) | .06 (+.0001) | .930 / .921 / +.0001 | .942 / .946 / .949 |
| .925 | MID .94 | .9404 (.940) | 0 | .930 / .962 / +.000 | .9455 / .948 / .9505 |
| .925 | HIGH .97 | .9392 (.939) | 0 | .930 / .993 / +.000 | .9465 / .9485 / .9505 |
| .935 | LOW .90 | .9446 (.945) | .29 (+.0016) | .939 / .921 / +.0021 (r_mm .93: +.0023) | .942 / .946 / .949 |
| .935 | MID .94 | .9430 (.943) | .15 (+.0003) | .939 / .962 / +.0004 | .9455 / .948 / .9505 |
| .935 | HIGH .97 | .9418 (.942) | 0 | .939 / .993 / +.000 | .9465 / .9485 / .9505 |

**Implied r for each row.** The prior is .95 [.90, .97] whatever the solo. One nuance (GUESS): a SOLO of .93 or above
would mean gold58 under-rates the fleet the way it under-rates the crop, which shifts weight toward HIGH. A SOLO near
.92 (Dread-like) leaves the regimes about 50/50.

Break-even r for a single member (MODEL):

| SOLO | BLEND W.30 >= parent needs r <= | Any +.001 at W* needs r <= |
|---|---|---|
| .905 | .762 | .760 |
| .915 | .815 | .805 |
| .925 | .874 | .856 |
| .935 | .940 | .915 |

### 3.3 Inversion: what r the two reads imply (MODEL `proxy_lb.log` E)

Rounding is +-.0005 on the solo, the read and the parent.

| SOLO \ BLEND | .937 | .938 | .939 | .940 | .941 | .942 | .943 | .944 | .945 |
|---|---|---|---|---|---|---|---|---|---|
| .905 | .88-.93 | .86-.91 | .83-.88 | .81-.86 | .79-.83 | .76-.81 | .74-.79 | .72-.76 | .69-.74 |
| .915 | .94-.99 | .91-.96 | .89-.94 | .86-.91 | .84-.89 | .82-.86 | .79-.84 | .77-.81 | .74-.79 |
| .925 | n/a | .98-1.0 | .95-1.0 | .92-.97 | .90-.95 | .88-.92 | .85-.90 | .83-.87 | .80-.85 |
| .935 | n/a | n/a | n/a | .99-1.0 | .97-1.0 | .94-.99 | .92-.97 | .89-.94 | .86-.91 |

One read pins r to about +-.025. That is the most valuable number tonight: the measured r_LB of a teacher-mixed fleet
member.

### 3.4 What our ensemble must reach (MODEL `proxy_lb.log` F)

Minimum ensemble solo for a TRUE lift over .943, at the optimal W:

| r_eff | +.003 (W*) | +.005 (W*) | +.007 (W*) |
|---|---|---|---|
| .75 | .9160 (.26) | .9250 (.34) | .9315 (.40) |
| .80 | .9255 (.29) | .9325 (.38) | .9380 (.44) |
| .85 | .9330 (.34) | .9390 (.43) | .9435 (.51) |
| .88 | .9370 (.38) | .9425 (.49) | .9465 (.58) |
| .90 | .9395 (.41) | .9440 (.53) | .9475 (.62) |
| .92 | .9420 (.47) | .9460 (.60) | .9490 (.70) |
| .94 | .9440 (.55) | .9470 (.68) | .9500 (.83) |
| .96 | .9455 (.67) | .9480 (.84) | .9500 (.99) |

Displayed .950 needs a true blend of .9495 or more. That is a lift of +.0065 over a parent at .943, and +.006 to +.007 across the parent's rounding interval [.9425, .9435].

### 3.5 Ensemble of our fleet (folds / seeds)

**Measured member-to-member correlation** on the fold-0 val set (867 studies, all members OOS, against v4 LLM labels
>= .5; MEASURED `proxy_oof867.log`).

AUC vs v4: mix50 raw .8811, ema .8770 · v4 raw .8692, ema .8650 · TTR raw (ep <= 8, still training) .8597, ema .8338
· gated raw (ep <= 3) .8160.

| Pair | Latent rA | Effective r_mm (inverted from the measured 2-member rank-mean ensemble) |
|---|---|---|
| mix50 raw + ema | .993 | .995 (raw and EMA are duplicates, no ensemble value) |
| mix50 + v4 (same seed and fold, different labels) | .909 | **.928** |
| mix50 + TTR | .872 | **.903** |
| v4 + TTR | .875 | **.908** |
| mix50 + gated (ep 3, weak) | .797 | .836 |

The binormal ensemble formula with latent r over-predicts these ensembles by .0002-.0036. The same effective-r offset
applies.

External (from notes): Tian, 2 seeds .927 to .930, implies effective r_mm .941.

Caveat (MODEL): r_mm measured against noisy LLM labels includes shared label noise, so the gold-label r_mm could be
somewhat lower. For seeds or folds of one recipe I use .95 (.93-.97). Cross-recipe or cross-label is about .90-.93.

**Formula (MODEL).** For k equal members with effective r_mm, v_k = (1 + (k-1) r_mm) / k. Ensemble solo is
Phi(d / sqrt(2 v_k)), and ensemble r with the parent is r_m / sqrt(v_k).

- **Solo gain:** k = 2 adds +.0025..+.003, k = 5 about +.004, k = 10 about +.005, k = 20 about +.005 (at r_mm .95).
  At r_mm .93, k = 10 adds about +.007.
- **Correlation with the parent rises:** at member r_m .967, k = 10 gives .989 (at r_mm .95). At member r_m .90,
  k = 10 gives .921 (at r_mm .95) or .930 (at r_mm .93).
- **Net:** a family ensemble moves along a line that is nearly parallel to the iso-lift curves in table 3.4. In
  every row of 3.2 except SOLO .935 / LOW, a 10-member fleet adds +.000.
- **Decorrelating labels** (v4 or gated instead of mix50) buys about -.05..-.07 in r. Evidence: gold58 crop mixSA vs
  v4-crop r_fit vs raptor_coat_side is .852 vs .783, and the gate log's Spearman vs the raptor proxy is .889 vs .840.
  It costs about .012 of solo (val vs v4 .881 vs .869). At r .90 or above that trade is about neutral (MODEL), for
  example a .928 member at r .907 times 10 gives +.0002.

### 3.6 Decision map for tonight

| Reads | Implied r_LB | Meaning |
|---|---|---|
| BLEND >= .943 (SOLO .905-.925) | .74-.90 | LOW regime. Own-fleet route alive. Next: grow solo (TTR / more folds / TTA). Target an ensemble of .935-.94 at r ~.87, worth about +.002-.004. Ship W ~.35-.45 |
| BLEND .941-.942, SOLO ~.915-.925 | .82-.95 | Ambiguous. Own route worth about +.000-.002. Merge is the lever |
| BLEND <= .940, SOLO .915-.925 | >= .86 (>= .92 at SOLO .925) | HIGH regime. A teacher-mixed fleet cannot lift the parent. Stop building more of the same. Merge is the only route to >= .945 |
| SOLO ~.905 (any BLEND) | n/a | Member too weak: any r above ~.76 hurts at W .30. Merge is the lever |
| SOLO >= .935 | depends on BLEND (row .935) | Strength surprise. Even then +.007 needs r <= .85 at an ensemble of .9435 |

---

## 4. Is +.007 (to .950) plausible from our own fleet?

**By 10-12: no. About 1-3% (GUESS).**

The requirement (MODEL, 3.4) is an ensemble solo of .938 or better at r_eff .80 or lower, or .9435 at .85, or .9475
at .90.

What we have (MEASURED or MODEL):
- One fleet member: gold58 .9144, LB prior about .922.
- 10 folds or seeds add about +.005, giving about .927.
- TTA adds about +.002 (notes, CLAIMED / ledger).
- A better recipe would add +0..+.008 (GUESS). TTR at ep 7 of 24 is not final, but it reads val .860 vs .869 for its pure-v4 reference at ep 7, and gold58 .865 vs .861 (MEASURED `~/fleet_runs/*.log`).

That puts the realistic best at about .93-.935. The fleet's r_LB is most likely .90-.97 and rises with ensembling.
Both gaps would have to close at once: about +.01..+.02 of solo and about -.1 of r. Even the optimistic
"LOW + strength surprise" corner (members at .935, r .85, k = 12) reaches only about +.0044, to .947 (MODEL
`proxy_lb.log` G).

**By 10-22: still about 2-5% (GUESS).** The obstacle is structural (parent-likeness), not time. Ten more days of the
V100 could add maybe +.005 of solo through more members and recipes, which is not enough at r .90 or above. The
team-merge deadline (10-15) also passes first. So the decision does not change: the merge is the lever, and the
own-fleet value is a +.000..+.002 hedge.

What *would* make +.007 plausible is a new member class that is both about .94 solo and about as independent as the
cnx member (r ~.82). That means not teacher-distilled and not Raptor-like, yet .05 stronger than cnx. Nothing we have
measured is close.

---

## 5. Honesty notes: what cannot be measured

- **The gold58-to-LB transfer of r is the weakest link.** It rests on 3 crop reads with unknown served solos, 1 cnx
  read with a known solo but no gold58 predictions, and 1 external read with an unclear reference arm. The ratio k of
  about .4 has a plausible range of about .2 to .8. Tonight's BLEND read is worth more than all of this modelling
  combined.
- **gold58 is contaminated for the parent** (DINO / Rad in-sample), so r(fleet, full parent) cannot be measured
  offline. No study set exists on which every parent arm is out-of-sample, except the hidden test.
- The crop members' gold58 predictions come from gold58-selected checkpoints. Their gold58 r may be biased in either
  direction.
- **Display rounding** (+-.0005 on both parent and read) limits every inversion to about +-.025 in r.
- **r_mm** was measured against noisy LLM labels and on only 2-3 truly distinct members (TTR and gated are still
  training). The seed-to-seed r_mm for our own recipe is unmeasured; I used .95 (.93-.97) from the analogs.
- **The model is macro-level, with one global W.** Per-label routing is not modelled. Per-label heterogeneity changed
  predictions by at most .0009 here.

---

## 6. Files

Box (`~/k2_test/analysis_wf/`):

| Script | Log | Output | Content |
|---|---|---|---|
| `proxy_gold58.py` | `.log` | `_out.json` | Arms, correlations, latent r, model validation |
| `proxy_fit.py` | `.log` | `_out.json`, `proxy_fit_pairs.csv` | Effective r, offset, LOBO CV |
| `proxy_ctx4.py` | `.log` | | The 4 context curves |
| `proxy_oof867.py` | `.log` | `_out.json` | Member-to-member r, ensemble check |
| `proxy_lb.py` | `.log` | `_out.json` | Anchors, P_eff, transfer, tables |
| `proxy_k2table.py` | `.log` | `_out.json` | The K2 table |
| `proxy_k2prior.py` | `.log` | | Prior predictive |

`proxy_crop_gold74_preds.npz` in the same folder is a copy of laneA's `gold74_preds_laneA.npz`.

Laptop (`scratchpad/jump_wf/`): the same scripts, `proxy_box_out/` (all logs and JSON), and this file.
