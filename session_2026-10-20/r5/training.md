# training.md: Task C, training option-value study (10-20, ~01:50-03:30 IST, Claude, r5)

STUDY ONLY. Nothing was trained, pushed, committed or submitted. No dataset was created. No scheduled task, PREREG, HANDOFF,
PROGRESS or session_2026-10-19/a20 file was touched.
- **Box use.** One inference-only GPU job (sec 2.2), started after `nvidia-smi` showed no process. It ran 10 existing fleet
  checkpoints (5 OAI-free, 5 OAI-trained) on held-out OAI follow-up knees. Predictions and id lists stay in `~/r5_oai/taskC/`
  on the box. Only aggregates came back over ssh. Nothing was killed. No pkill. No second GPU job ran beside it.
- **Governance slip (disclosed).** While scoping file formats, one `head -2` on two OAI CSVs on the box printed 1 OAI knee id,
  that knee's 12 teacher probabilities and its 12 MOAKS label values to this session's console. They were not written to any
  file or used. Every later box command printed aggregates only.
- **Tags.** MEASURED = computed tonight, or read from a named log. CLAIMED = an older note or a third party, source named.
  MODEL = arithmetic on stated inputs. GUESS = judgement.
- **Units.** Macro = 12-label macro AUC. +1e-4 of real private gain is worth about +4 pts P(silver) for the A/B pair
  (CLAIMED `session_2026-10-19/think/picks.md` l.37).

## 1. Verdict: NO-GO for every training option

1. **No model we could train before 10-22 ~10:15 IST is both plausibly additive and checkable on an independent expert ruler
   before use.**
2. **The OAI held-out ruler fails for OAI-trained models, and this is now MEASURED, not assumed.**
   - We already own an OAI-trained / OAI-free twin pair: the K14 fleet DT (TTR224 frac75 + OAI T123 V00, 5 folds) and the
     same recipe without OAI.
   - Tonight they were scored on 1,269 OAI follow-up knee-visits whose knee DT never saw (637 knees).
   - In domain, DT beats D by OA-mean **+.072** AUC (strict cohort +.091) and menisci-mean **+.031**, P 1.00. Lateral OA
     alone: **+.154**.
   - The same pair on the RSNA radiologist labels:
     - solo OA-mean +.0145 (gold58) / +.0104 (gold74), with Fracture -.026 / -.020;
     - routed into c4 on the OA trio: -.0003 / -.0010 macro (CLAIMED ours.md);
     - on the LB: 1 in-band best-id move and 2 non-moves.
   - So in-domain OAI gains overstate the RSNA solo gain by **~5-7x on OA-mean, 5-12x on menisci and ~20x on Lateral OA**,
     and they have the wrong sign once inside c4.
   - An OAI held-out fold therefore certifies "learned MOAKS on OAI images". It does not certify "adds to c4 on hidden".
3. **gold58 / gold74 is the only RSNA expert ruler, and it is too small.**
   - Paired sd of a member change is .0003-.0012 macro (CLAIMED ours.md sec 3).
   - It mis-called all 5 A19 probes on 10-19 (CLAIMED PROGRESS_19).
4. **The one design that does have an independent ruler is an RSNA-only model checked on OAI** (the way E-OA was validated).
   - Such a model can only learn the report regime.
   - Its expert-regime part (OA decoupling) is what E-OA already applies after the fact, to the much stronger final blend.
5. **Our track record on the strong base is poor.**
   - Own-trained components are 1 of 7 on the LB (one in-band move; at least two display steps down).
   - Every port of a public recipe lost .009-.016 on gold58 (sec 4).
6. **The best-mechanism candidate is C1: an R6 warm-start fine-tune with OAI MOAKS labels.**
   - It needs ~38 h nominal, or ~47 h at our usual slip, to reach one read. That is one blind shot.
   - R6's training targets are not in the bundle, and the run crosses our 10-20 12:00 IST training freeze.
   - MODEL hidden value is about -1e-4 to +1e-4 (centre -0.5e-4 to 0 after the history), P(R1 move) ~.45 and
     P(true private gain > +1e-4) ~.15.
   - EV is about -1 to 0 pts P(silver) before costs.
7. **Use the idle V100 for inference-only measurements** for other lenses (like tonight's), or leave it idle.
8. **Picks are unchanged:** A = API best-id under R1 after the eoa read, B = R6 57248084.

## 2. Is there an independent expert ruler for a model we train?

### 2.1 The rulers we have

| ruler | size / labels | independent for an RSNA-only model? | independent for an OAI-trained model? | blind spot |
|---|---|---|---|---|
| OAI V00 MOAKS | 2,399-2,581 knees. OA (cartilage proxies), menisci, ACL, effusion, Hoffa, Baker's | yes (E-OA was validated on it) | **no**. Every T12 / T123 fold trained on all 2,581 V00 knees (`union_receipt.json`: train_oai 2,581 in every fold, MEASURED) | one 3T protocol (IW TSE + DESS MPR); MOAKS coupling (OA phi ~.05) is not the RSNA expert coupling (~.24-.40, CLAIMED HANDOFF 10-19 23:30) |
| OAI follow-up visits of knees never in V00 training | 1,269 visits / 637 knees (strict, no knee of the subject in training: 1,090 / 546). Menisci and OA only: fluid labels on 4 rows (MEASURED `scope_heldout.py`) | yes | **in-domain held-out**: same scanner, protocol and label definitions as training. It measures fit to OAI | no Effusion / Synovitis / Baker's / ACL / MCL / Contusion / Fracture |
| gold58 / gold74 (RSNA radiologists) | all 12 labels, 9-48 positives per label | yes | yes | n tiny. 5 of 5 gold58-picked A19 moves read negative (CLAIMED PROGRESS_19). Its extra 16 studies have missing labels on some columns |
| RSNA report OOF | 4,333 studies, all 12 labels | wrong regime (LLM report labels) | wrong regime | rated our members +.010-.020 above R7 while they tied or lost on gold58 and the LB (CLAIMED ours.md sec 7) |
| Public LB (R1) | ~390 hidden studies | yes | yes | paired sd ~2-7e-4, 5 slots a day, ~6 h latency |

### 2.2 MEASURED tonight: in-domain gain vs RSNA transfer, same twin pair

Pair:
- **DT** = `f{0-4}_s1043_ttr_frac75oaiT123v00` raw best (OAI-trained).
- **D** = `f{0-4}_s1043_ttr_frac75` raw best (the OAI-free twin: same recipe, seed and folds).
- **D2** = `f{0-4}_s1044_ttr_frac75`. It is used on gold58 because D's gold58 predictions are not on disk.
- Set means of 5 folds.

Held-out OAI cohort:
- Follow-up knee-visits (v01-v06) of knees that are not among the 2,581 V00 training knees: 1,269 rows, 637 knees, 637
  subjects. Strict subset: 1,090 rows, 546 subjects with no knee in training.
- Labels: T123 allv soft labels binarised at >= .5 (DT's own training definition).
- Subject-cluster paired bootstrap, 1,000 resamples (`~/r5_oai/taskC/ho_eval.py`, MEASURED).

In-domain held-out OAI result (MEASURED `box_taskC/ho_eval.log`, aggregates only; 5-fold set means; sd from the subject
bootstrap):

| label | positives (knee cohort) | D (OAI-free) | DT (OAI-trained) | delta DT - D (sd, P>0) | strict cohort delta (sd, P>0) |
|---|---|---|---|---|---|
| Medial Meniscus | 228 / 1,266 | .9434 | .9709 | +.0276 (.0070, 1.00) | +.0274 (.0072, 1.00) |
| Lateral Meniscus | 120 / 1,266 | .8768 | .9114 | +.0347 (.0110, 1.00) | +.0288 (.0108, 1.00) |
| Medial OA | 38 / 1,266 | .7275 | .7228 | -.0047 (.0353, .44) | +.0357 (.0502, .77) |
| Lateral OA | 59 / 1,266 | .6806 | .8346 | **+.1539** (.0279, 1.00) | +.1726 (.0335, 1.00) |
| PF OA | 602 / 1,265 | .8414 | .9077 | **+.0662** (.0092, 1.00) | +.0643 (.0089, 1.00) |
| **OA-mean** | | .7499 | .8217 | **+.0718** (.0162, 1.00) | +.0909 (.0206, 1.00) |
| **menisci-mean** | | .9101 | .9412 | **+.0311** (.0066, 1.00) | +.0281 (.0067, 1.00) |

Fold-paired single members, OA-mean delta: +.068 / +.079 / +.087 / +.077 / +.065 (knee cohort). Every fold agrees.
GPU cost: 10 checkpoints x 1,269 visits at 0.13-0.59 s per study, ~45 min wall on the idle V100 (incl. one relaunch after
a print bug).

RSNA radiologist transfer of the same OAI training (local `work_c/g58_twin.py` -> `work_c/g58_twin.log`, MEASURED; paired
bootstrap 3,000):

| label | gold58 D2 | gold58 DT | delta (sd, P>0) | gold74 delta (sd, P>0) |
|---|---|---|---|---|
| Medial Meniscus | .9555 | .9591 | +.0036 (.0106, .59) | +.0045 (.0089, .69) |
| Lateral Meniscus | .8994 | .9081 | +.0087 (.0198, .65) | +.0008 (.0190, .52) |
| Medial OA | .9674 | .9659 | -.0016 (.0065, .30) | -.0036 (.0083, .27) |
| Lateral OA | .8066 | .8143 | +.0077 (.0147, .66) | +.0088 (.0111, .76) |
| PF OA | .8404 | .8777 | +.0373 (.0195, .99) | +.0261 (.0152, .98) |
| **OA-mean** | | | **+.0145 (.0080, .97)** | **+.0104 (.0071, .94)** |
| menisci-mean | | | +.0062 (.0109, .72) | +.0026 (.0102, .62) |
| Fracture (masked for OAI rows) | .9111 | .8847 | -.0264 (.0177, .04) | -.0200 (.0142, .05) |
| ACL | .9755 | .9865 | +.0110 (.0092, .86) | +.0092 (.0078, .90) |
| macro-12 | .9192 | .9235 | +.0043 (.0039) | +.0031 (.0035) |

Exact-twin check (CLAIMED `k14_gate_20261012_1110.txt` on the box): set DO - D gold58 +.0021 (SE .0037) and gold74 +.0020
(SE .0033) macro. Single-member gold58 deltas: +.0031 / +.0041 / -.0016 / +.0015 / +.0068.

Inside the real blend, and on the LB:
- K14 DT routed into c4 on the OA trio at .10 / .20: **-.00027 / -.00096 macro** (P>0 .16 / .22). Lateral OA -.0039 /
  -.0077 (CLAIMED ours.md sec 4, `ours_more_g58.log`).
- LB with the K14 fleet (DT inside):
  - HF-R20 on H: .954 = base (non-move).
  - HFR7-RR on R7: in-band best-id move on 10-14 (CLAIMED THINK_1019 critique 5).
  - V224-FF20 on FLIP: .956, non-move.

### 2.3 Reading

| quantity (DT - OAI-free twin) | in-domain held-out OAI | RSNA gold58 solo | RSNA gold74 solo | inside c4 (OA trio .10 / .20) | LB |
|---|---|---|---|---|---|
| OA-mean | +.072 | +.0145 | +.0104 | -.00027 / -.00096 macro | 1 in-band move, 2 non-moves (whole fleet) |
| Lateral OA | +.154 | +.0077 | +.0088 | -.0039 / -.0077 (label) | n/a |
| PF OA | +.066 | +.0373 | +.0261 | +.0006 / -.0006 (label) | n/a |
| menisci-mean | +.031 | +.0062 | +.0026 | n/a | n/a |

1. **The in-domain gain is real but mostly OAI-specific.**
   - The transfer ratio to RSNA radiologists is ~.15-.2 on OA-mean and ~.05 on Lateral OA. PF OA transfers best (~.4-.55).
   - The rest is domain fit: one protocol, DESS MPRs, and MOAKS definitions, which D never saw.
2. **Even the transferred part did not survive the blend.**
   - c4's OA labels already come from stronger legs: R6 .991 / .865 / .916 on gold58, plus the OAI-trained Raptor at .5 on
     PF OA. The LB showed that leg matters: pf10 (Raptor .1 on PF OA) read negative on 10-19, and stacked with Fracture 1.0
     it dropped a full display step.
   - Routing DT onto the OA trio in c4 is negative on gold58.
3. **So a held-out OAI fold would have PASSED this model by a mile** (+.07 OA-mean, P 1.00), and the model is LB-neutral to
   negative where it counts.
   - The same holds for any OAI-trained candidate (C1, C6). OAI held-out can serve as a plumbing check ("training worked"),
     never as a selection ruler for hidden.
4. **What OAI can still do is what it did for E-OA:** score RSNA-only models (OAI-free, independent) for a regime question.
   It already did that for the in-flight E-OA entries (G1 box check, 10-19). No trained model is needed for that.
5. **Caveats.**
   - D2 (seed 1044) stands in for the exact twin D on gold58. The exact-twin set numbers (+.0021 / +.0020 macro) agree in
     direction.
   - The OAI cohort is follow-up visits of knees that were never in training. Some subjects have the other knee in training
     (the strict cohort removes them; it gives larger deltas, not smaller).
   - Labels are binarised at >= .5 of the T123 soft scale.

## 3. Candidates

"Hours to a read" assumes Eesh's go at 10:00 IST 10-20 (sec 5 has the step list). "vs eoa base" is the hidden macro change
over whichever eoa entry (or c4) is the API best-id after today's read. P(R1 move) is the chance that the API best-id moves.

| # | candidate | what it adds that c4 / eoa lacks | ruler before use | hidden macro vs eoa base | P(true > 0) | P(R1 move) | hours to a read | verdict |
|---|---|---|---|---|---|---|---|---|
| C1 | **R6 warm-start fine-tune, 4 folds, full R6 swap.** 2-3 epochs on RSNA rows (self-distilled R6 targets, because R6's own targets are not in the bundle) plus OAI MOAKS T123 rows (9 labels; MCL / Contusion / Fracture masked); 1 OAI fold held out | OAI expert OA and meniscus grading inside the strongest leg | OAI held-out fold (in-domain, sec 2), gold58 (too small), report OOF (wrong regime) | about -1e-4 to +1e-4, centre -0.5e-4 to 0 (MODEL sec 6) | .40-.45 | ~.45 | ~38 h nominal, ~47 h slipped | **NO-GO** |
| C1-r | C1, but used only on the OA trio (R6 kept for the other 9 labels, which avoids the Fracture cost) | same | same | 0 to +1e-4 (MODEL) | ~.5 | ~.5 | same, plus a second flipped R6 stage | **NO-GO (runtime).** A second R6-flip stage is ~2.5-3 h of hidden time and pushes S6F past its 5.75 h guard (CLAIMED PROGRESS_16 00:15-00:40 "Lead 8"; THINK_1019: 5.96 h > 5.75 h) |
| C1-1f | swap 1 of R6's 4 fold models | 1/4 of C1 | same | 1/4 of C1 | ~.45 | ~.5 (coin) | ~27 h | NO-GO (below read noise) |
| C2 | **Decoupled OA head** on R6 outputs (12 logits + state heads), fitted on OAI MOAKS; constants inlined in cell 18 (no dataset) | per-label OA decoupling weights | OAI held-out folds (in-domain for the head), gold58 | <= E-OA's own effect (eoa10 ~+.0003); increment ~0 or below | ~.45 | ~.5 | ~10-12 h after R6-on-OAI predictions exist (the parallel r5 lens) | **NO-GO.** It is a multi-parameter E-OA fitted to MOAKS coupling (OA phi ~.05), which over-decouples for RSNA experts (~.24-.40). The skeptic ladder already found a .2's sign model-dependent (CLAIMED MORE_1020 row 1) |
| C3 | RSNA-only member trained on decoupled report OA targets, checked on OAI | nothing new: the target transform carries no expert information | OAI V00 (independent, like E-OA) + gold58 | ~0 over eoa (MODEL) | ~.45 | ~.5 | ~13-15 h (one all-data member) | NO-GO. E-OA already does this to the final output; member class is 1 of 7 |
| C4 | route the existing K14 DT onto the OA trio in c4 (no training) | OAI OA grading | gold58 | -.00027 / -.00096 at .10 / .20 (CLAIMED ours.md, `ours_more_g58.log`) | low | < .5 | ~7 h | NO |
| C5 | "expert-rescaled" S6-recipe seed (Synovitis broadened, Effusion narrowed, OA decoupled targets) | label regime | gold58 only. The rescale would itself be fitted on gold58 (spending the only check) or on OAI (a different regime) | about -1e-4 (MODEL: our S6 seeds lost .010-.016 to nartaa's) | ~.35 | ~.45 | ~21 h, plus a second S6 stage (closed on runtime 10-16) | NO |
| C6 | new OAI fleet trained with OAI held-out folds | = DT | in-domain (sec 2.2 measures exactly this model) | as DT in c4: below 0 | low | < .5 | ~30 h for 5 folds | NO |

Why "held-out OAI folds for effusion" is not even available: the follow-up visits carry Effusion / Synovitis / Baker's on only 4
rows (MEASURED). A fluid ruler would need a V00 split (train on 4/5 of V00), i.e. a new training run just to create the
ruler.

## 4. What our own training has done on the LB (the prior)

Weak-base era (10-04 to 10-09, base .942-.946): own fleets lifted .942 -> .946, and FN reached .951 with the public N
(CLAIMED memory / PROGRESS). That era does not transfer: the base is now public-strong. From the .954 base on:

| date | entry | own-trained part | offline gate | LB read (R1) | source |
|---|---|---|---|---|---|
| 10-12 | HF-R20 57106365 | K14 fleet (includes OAI-trained DT), .2 on H | K14 PASS: OOF +.00106, gold58 -.0007 | .954 = base, non-move | PROGRESS_12 |
| 10-14 | HFR7-RR 57146261 | K14 fleet .2 on HR7-RR | n/a | **best-id moved in band (the one win)** | THINK_1019 critique 5 |
| 10-14 | HFR7-RS 57154866 | 2 RSNA-only Raptor seeds, .30 of the Raptor slot | PASS (gold58 +.0011) | .955 < .956, a display step down | PROGRESS_14 |
| 10-14 | RR-F30 / F30-RS / RR-F40 | fleet at .3 / .4 | n/a | .956 / .955 / .955: no gain or down | PROGRESS_14 |
| 10-15 | V224-FF20 57199082 | K14 fleet .2 on FLIP | n/a | .956, non-move | ours.md sec 2 |
| 10-15 | S6F + S 57222884 | T_b student, 5 folds, .2 | student gate PASS | .957, non-move | PROGRESS_15 |
| 10-16 | COMBO 57250689 | KC1 .2 on R6 | G2 gold74 +.0068 | .957 < .958, a display step down | PROGRESS_16 |
| never entered | K12 = T12 OAI fleet (gate FAIL: gold58 -.0018; gold58 PF OA +.019 / Baker's +.025 but Lateral Meniscus -.021 / Fracture -.019), S6 seeds s43 / s44, OAI Raptor S42, Swin-T, R50-RadImageNet, PL (T123 + pseudo labels), T123allv, KC-all | gates FAIL or no gold gain | n/a | n/a | PROGRESS_10-16 |

- Public checkpoints we added read positive 3 of 3 (r224, S6fall, R6; CLAIMED ours.md sec 2).
- **Port gap** (CLAIMED ours.md sec 7, RAPTOR_RENTED_RESULTS, PROGRESS_14 / 16):

  | our port | gold58 | public original |
  |---|---|---|
  | RSNA-only CoAtNet seeds | .911-.914 | nartaa .923 |
  | OAI CoAtNet S42 ("port-broken") | .907 | nartaa .923 |
  | S6-recipe seeds | .921 / .915 | nartaa .931 |

  C1 would be our first port of the R6 trainer, with substitute targets.
- **The OAI-trained fleet specifically:** 1 in-band move and 2 non-moves. Its OA gain on gold58 did not survive c4 (sec 2.2).

## 5. Exact hours

Fixed points:
- Now 10-20 ~02:30 IST.
- Our own training freeze: 10-20 12:00 IST (HANDOFF 10-19 00:10).
- Last readable submit: 10-22 ~10:15 IST (08:45 with a queue buffer). Final: 10-22 23:59 UTC.
- Slots: 10-20 has 3 left after eoa20 / eoa10 at 05:31; 10-21 has 5; 10-22 has ~1-2 readable. Slots are not the binding
  constraint; time to a read and the ruler are.

### 5.1 C1 (R6 warm-start + OAI), step by step

| step | hours | anchor |
|---|---|---|
| 0. Eesh: go, lift the 10-20 12:00 freeze, PREREG amendment | 0.5 | his time |
| 1a. px384 render cache of the 4,407 RSNA studies from the 265 GB competition zip already on the box (R6 renders 384, crops 352: MEASURED `~/r6g58/bundle/manifest.json`) | 8.5-11 | the px320 cache took 7.7 h with 16 processes (17:52 -> 01:36, 4,407 studies, 268 GB; CLAIMED PROGRESS_15 / 16). x1.1-1.4 for 384 (MODEL); ~385 GB disk (5.4 TB free, MEASURED) |
| 1b. OAI -> R6 input adapter (plane slots, Fluid / Fat flags from `oai_series_meta.csv`) + OAI px384 build for 2,581 V00 + 1,269 held-out visits (parallel with 1a) | 3-5 dev + 2-3 build | GUESS. The parallel r5 lens is building an OAI -> R6 route for inference (`~/r5_oai/build_synth.py`); it could be reused |
| 1c. Trainer patch (parallel): compact_train v9.8 / ct_s6v100 at 384 / 352; load R6 fold weights including heads (only `--init` backbone and `--resume` own checkpoints exist today, MEASURED arg list); masked loss for OAI rows and state heads; R6 fold rule sha1(uid)[:8] % 4; OAI held-out fold; fp16 on the V100 | 4-6 incl. smoke | precedent: S6-seed kit 14:20 -> launch 03:04 (~12.7 h incl. its px build); KC1 kit ~3.75 h (CLAIMED PROGRESS_15 / 16) |
| 2. Self-distillation targets: R6 4 folds x flip on the 4,349 train studies from the cache. R6's training targets are not in the bundle (MEASURED file list: 4 fold .pt, side_cls, fluid_stacker, state_spec, state_stacker, manifest) | ~1 GPU | MODEL. Through the DICOM path R6 costs ~14 s per study (824 s / 58, CLAIMED PROGRESS_19), i.e. ~17 h, so the cache is required |
| 3. Train 4 fold models, 3 epochs on ~3,262 RSNA + ~2,065 OAI rows each | **14.7** (2 epochs: 9.8) | .44 s per study-epoch at crop 256 (S6 s43: 34,784 steps x 2 studies in 8.14 h, CLAIMED PROGRESS_16), x1.89 for crop 352 = .83 s. 5,327 x 3 x .83 s = 3.7 h per fold (MODEL) |
| 4. Validate: OAI held-out fold R6 vs R6' (needs R6 on OAI), gold58 serve path on the GPU, R6-fold OOF on report labels, analysis | 2 | 3 gold58 GPU serve runs took ~19 min (CLAIMED PROGRESS_16 11:14 -> 11:33) |
| 5. Weights off the box (4 x 200 MB) + OAI-string scan | 0.2 | 3.2-4.5 MB/s box -> laptop (CLAIMED PROGRESS_15) |
| 6. Eesh creates the private dataset | 0.3-0.5, plus waiting for him | k14w (538 MB) ~25 min, kc1-members ~20 min (CLAIMED PROGRESS_12 / 15) |
| 7. Kernel: R6 swap on the eoa base (build_r6 pattern: 2 code lines + 1 metadata source, plus a new bundle id / manifest check) and gates (r6_check-style + probe_check_eoa) | 2-3 | R6 entry built and gated in ~50 min on 10-15 (CLAIMED PROGRESS_15 18:25-18:55) |
| 8. Commit | 0.33 (+0-4 queue) | a20 commits ~20 min (CLAIMED HANDOFF 10-20 00:55); queues reached 2-3 h on 10-17 (CLAIMED RESEARCH_1017) |
| 9. Score | 6.1 (5.7-6.5) | c4 class 5h40-6h27, median 6h05 (CLAIMED runtime.md sec 1) |
| **Total** | **~37.8 h nominal** (2 epochs: ~32.9 h). **~47 h** at our usual x1.3 slip on steps 0-7 | MODEL |

Clock (MODEL):

| go at | nominal: submit / read | slipped x1.3: submit / read | slack to the 10-22 ~10:15 readable submit |
|---|---|---|---|
| 10:00 10-20 (Eesh awake) | 10-21 ~17:45 / ~23:50 (queue: ~21:45 / ~03:50 10-22) | 10-22 ~03:10 / ~09:15 (queue: ~07:10 / ~13:15) | 3-16 h; one shot, no second variant |
| 02:30 10-20 (now, not authorised) | 10-21 ~10:15 / ~16:20 | 10-21 ~19:40 / 10-22 ~01:45 | 15-24 h |

The V100 would be busy ~21:30 10-20 -> ~12:10 10-21 (the training), which competes with any inference job other lenses need.

### 5.2 The cheaper candidates

| candidate | train | validate | dataset (Eesh) | kernel + gate | commit | score | total from go | read if go 10:00 10-20 |
|---|---|---|---|---|---|---|---|---|
| C2 OA head on R6 outputs | fit 0.25 on CPU, after R6-on-OAI predictions exist (other lens, MODEL 1-3 h) | 0.35 (OAI CV + gold58) | none (inline constants) | 2.0 | 0.33 | 6.1 | ~10-12 h | 10-20 ~20:00-22:00 |
| C3 RSNA-only decoupled-target member | 1 (targets) + 3.3-3.5 (one all-data member; KC s2007 3.3 h, CLAIMED PROGRESS_15) | 0.4 (OAI V00 at 0.17-0.59 s per study, MEASURED tonight; gold58) | 0.2 + 0.5 | 2.0 (fleet stage after S6F exists, gated) | 0.33 | 6.3-7.0 (stage +0.3-0.8 h) | ~14-15 h | 10-21 ~00:00-01:00 |
| C1-1f one R6 fold | 3.7 | 2 | 0.7 | 2.5 | 0.33 | 6.1 | ~27 h (the ~10.5 h of prep is still needed) | 10-21 ~13:00 |

## 6. Expected value and the go / no-go arithmetic

### 6.1 C1 hidden gain (MODEL; inputs tagged)

1. **Headroom.** R6 on gold58 is already Medial / Lateral / PF OA .991 / .865 / .916 (MEASURED from `ours_g58.json`). That
   is above DT's .966 / .814 / .878. The OAI-trained Raptor sits below R6 on its OAI-supervised labels (PF OA .860, Lateral OA
   .776; CLAIMED edteoh 749858 for which labels Raptor took from OAI).
   - GUESS: an OAI fine-tune lifts R6 solo by +.003 to +.012 on PF OA and +.002 to +.008 on Lateral OA. DT's +.026 to +.037
     on PF OA started from .84-.86, and gains shrink near .92.
2. **Share in c4.** R6's share is (1 - W_RAPTOR) x .8: Medial OA .40, Lateral OA .64, PF OA .40, Fracture .28 (c4 table,
   CLAIMED HANDOFF). Blend damping is ~.6 (GUESS).
   - OA part: +.0015 to +.0060 of label AUC in total, i.e. **+.0001 to +.0005 macro**.
   - E-OA overlap: R6' is less coupled, so eoa over-corrects it. GUESS: 30-50 % of the OA part is lost, leaving
     **+.00005 to +.00035**.
3. **Cost on the masked labels.**
   - DT lost Fracture by -.020 / -.026 solo (MEASURED), and T12 lost Lateral Meniscus / Fracture / MCL by -.021 / -.019 /
     -.009 on gold58 (CLAIMED PROGRESS_10).
   - With warm-start and masking, GUESS -.003 to -.010 solo on Fracture, which at share .28 and damping .6 is **-.00004 to
     -.00014 macro**. MCL and Effusion are small.
4. **Raw net:** -.0001 to +.0003, centre ~+.0001.
5. **History shading** gives the final estimate, centre about -0.5e-4 to 0 (sd ~1.5e-4):
   - the same OAI effect went from solo-positive to c4-negative (sec 2.2);
   - own-trained entries are 1 of 7;
   - this would be our first R6-trainer port, on substitute targets.
6. P(true > 0) ~.40-.45. P(true > +1e-4) ~.15 (GUESS).

### 6.2 What a read can tell us

- An R6 swap is a whole-leg change. GUESS: its public paired sd is ~4-7e-4 (CLAIMED metric.md: 3-7e-4 for member changes).
  That is several times |gain|.
- So P(R1 move) is about Phi(gain / sd), i.e. ~.45 (.40-.50). A move would be mostly noise: E[gain | move] stays near the prior
  centre.
- That kills the "option value" argument. The option can only be exercised through a coin-flip read, so its value is about
  P(move) x E[gain | move] = ~.45 x (-0.3e-4), i.e. ~-0.5 pts P(silver) (range -1 to +0.5 pts) at +4 pts per 1e-4.
- Costs not counted above:
  - Eesh's go and amendment, the freeze lift, and the dataset (~1-1.5 h of his attention, some of it at night);
  - new code paths with our port-break history;
  - the GPU tied up for ~15 h;
  - own OAI-trained R6 weights inside a possible final pick A. The OAI eligibility threads 748276 / 749145 are still
    unanswered (CLAIMED HANDOFF / forum notes). Raptor is OAI-trained too, but it is public and used by thousands.

### 6.3 Decision

| candidate | EV (pts P(silver), MODEL) | decisive reason | go / no-go |
|---|---|---|---|
| C1 | -1 to +0.5 | no independent transfer ruler; 1 of 7 prior; one blind shot | **NO-GO** |
| C1-r | n/a | hidden runtime (second R6-flip stage breaks the S6F guard) | **NO-GO** |
| C2 | ~0 vs the eoa read | dominated: an E-OA strength step does the same with no fitted MOAKS coupling | **NO-GO** |
| C3 | ~0 | adds no expert information; E-OA already applied to the final output | **NO-GO** |
| C4, C5, C6 | < 0 | measured negative in c4, or a closed recipe, or = DT | **NO-GO** |

## 7. What would change the answer

- **Both eoa entries move the best-id, and eoa20 > eoa10.** That says the hidden regime wants more OA decoupling than c4 has.
  The cheap way to take more is one E-OA strength step (e.g. a .3) on the R1 winner: ~7 h to a read, no new weights, no
  freeze lift. Training is still dominated.
- **A public release of R6's training targets, or of a gold-aligned label table.** That would allow a faithful R6
  continuation, but it still could not be trained and read before 10-22 ~10:15.
- **Eesh wants the V100 busy anyway.** Use it for inference-only measurements for other lenses (R6 on OAI, gold74
  extensions).
  - If he explicitly wants training as pure research, the least-bad job is C1 on one fold with the OAI fold held out (~10 h
    prep + 3.7 h train). It would say how far OAI fine-tuning moves R6 on gold58.
  - Nothing from it could enter a readable submission. Its decision value is ~0.

## 8. Files

All under `session_2026-10-20/r5/` unless the path starts with `~` (box).
- `training.md`: this file.
- `box_taskC/`: laptop copies of the box scripts (code only). `scope_heldout.py` (counts), `make_ids.py` (box-only id list),
  `ho_infer.py` (inference, copy of `~/oai_build/oai_infer_gpu.py` without the OAI-trained refusal), `ho_eval.py`
  (aggregates).
- `~/r5_oai/taskC/`: the same scripts. The id list, predictions (`preds/`) and `ho_infer.log` stay on the box.
  `ho_eval.log` is aggregates only; a copy is in `box_taskC/ho_eval.log` (checked: no `oai_` string).
- `work_c/g58_twin.py`, `work_c/g58_twin.log`: RSNA-only gold58 / gold74 per-label DT vs D2.
