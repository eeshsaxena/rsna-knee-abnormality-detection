I have everything I need. The validation log preserved Qwen's per-label confusion matrix on gold-58, and I computed the matching rule matrix locally — so I can quantify the ensemble now (with a stated independence caveat), not just hand-wave it.

---

# LABEL PIPELINE DESIGN — report → 12 labels, target macro-AUC ~0.78 on gold-58 (from 0.7313)

## 0. The evidence I built this on (measured, gold-58, same 58 studies for both)

Rule matrix computed locally just now (`extract_rule_v0.predict_one`); Qwen matrix recovered from the validation-run log (`kernels/qwen_extract/_out/rsna-knee-qwen-extract.log`). Both reconcile to the known 0.6534 / 0.7313 macro.

| Label | rule p/r | rule AUC | Qwen p/r | Qwen AUC | winner |
|---|---|---|---|---|---|
| ACL | .89/.67 | .804 | .70/.88 | .805 | tie, opposite errors |
| MCL | .67/.67 | .803 | .47/.89 | .853 | Qwen |
| Medial Meniscus | .80/.46 | .684 | .57/1.00 | .688 | tie, opposite errors |
| Lateral Meniscus | .85/.48 | .711 | .59/1.00 | .771 | Qwen |
| Medial OA | .50/.07 | .522 | .43/.87 | .736 | Qwen (rule ≈ random) |
| Lateral OA | .75/.27 | .626 | .29/.82 | .675 | Qwen |
| **PF OA** | **.80/.38** | **.663** | **.48/.48** | **.589** | **RULE (Qwen ≈ random)** |
| Effusion | .67/.63 | .575 | .68/.97 | .638 | Qwen, both weak |
| Synovitis | .67/.30 | .584 | .71/.56 | .681 | Qwen |
| Baker's | .67/.33 | .645 | .42/.92 | .795 | Qwen |
| Contusion | .43/.53 | .596 | .70/.74 | .791 | Qwen |
| Fracture | .67/.33 | .629 | .83/.56 | .753 | Qwen |
| **macro** | | **.653** | | **.731** | |

The pattern is uniform: **rule = high-precision/low-recall (FN-heavy), Qwen = high-recall/low-precision (FP-heavy)**, and the two use unrelated mechanisms (regex clause-scan vs. generative reading), so their errors are near-independent — the ideal ensemble setup.

I then estimated each combination rule under conditional-independence of the two extractors' errors given the true label (`scratchpad/ensemble_est.py`):

| combination | macro-AUC (est.) | vs Qwen |
|---|---|---|
| Qwen alone | .731 | — |
| **OR / union** | **.732** | **+0.00 (useless)** |
| **AND / agreement** | **.652** | **−0.08 (harmful)** |
| **SOFT 3-level `(rule+Qwen)/2`** | **.789** | **+0.058** |
| SOFT + PF-OA rule-weighted | .791 | +0.060 |

Verified the 3-level math by hand on ACL (.908 ✓). This is the load-bearing finding of the whole task, so read the next section as the direct consequence.

---

## (1) THE ENSEMBLE — soft continuous fusion, NOT per-label hard OR/AND

**Do not use union or agreement as the combiner.** The numbers above kill both: OR just inherits Qwen's recall (which is already saturated) plus its false positives → no gain (.732). AND collapses to the rule's recall → loss (.652). This is the single most important design correction.

**Use a per-label weighted soft average that emits a probability, not a bit:**

```
p_label = w_r · r_label  +  w_q · q_label        (r ∈ {0,1}, q ∈ [0,1])
```

Why soft dominates: the Kaggle metric is ranking-based ROC-AUC. Averaging two complementary extractors creates an **interior rank** (the disagreement cases land at ~0.5, between confident-both-negative and confident-both-positive). Because disagreements are enriched in genuinely borderline studies, ranking them in the middle is exactly right, and AUC rewards it. That interior rank is why ACL jumps .80→.91 and MCL .80→.93 — the two labels where the rule is most precise and Qwen most recall-heavy.

**Per-label weights** (default 0.5/0.5, override where the table demands it):

| Group | Labels | w_r | w_q | reason |
|---|---|---|---|---|
| Rule trustworthy, complementary | ACL, MCL, Med Men, Lat Men, Fracture | 0.5 | 0.5 | biggest interior-rank lifts here |
| Qwen dominates, rule adds little | Lat OA, Effusion, Synovitis, Baker's, Contusion | 0.35 | 0.65 | rule recall too low to matter |
| Rule ≈ random | Medial OA | 0.2 | 0.8 | rule rec .07, prec .50 — noise |
| **Qwen ≈ random** | **PF OA** | **0.65** | **0.35** | rule .663 > Qwen .589; also fix Qwen prompt (§2) |

Emit `q` as an actual probability (§2/§4), not 0/1 — that upgrades the "3-level" estimate to a finer-grained score and lifts it further. Also emit a **separate agreement weight** per study (agreement→1.0, disagreement→0.5) for use as a training sample weight (§4) — keep it distinct from the soft value.

---

## (2) PROMPT FIXES — recover PF OA, cut Qwen's false positives

Qwen on PF OA is rec .48 / prec .48 = **essentially random** (10 TP, 11 FP, 11 FN). Root cause: the definition lumps "patellofemoral/trochlear OA / chondral loss," and Qwen reserves it for the explicit word "osteoarthritis," missing the way PF disease is actually dictated (chondromalacia grades, retropatellar/trochlear cartilage fissuring). Fix in `qwen_extract_v2`:

1. **Expand the PF-OA definition** to name the real vocabulary: *chondromalacia patellae (any grade), retropatellar or trochlear cartilage fissuring / thinning / defect / full-thickness loss, patellar or trochlear osteophytes, patellofemoral joint-space narrowing.*
2. **Evidence-span grounding (biggest FP killer):** require Qwen to quote the supporting phrase before each label — *"for each finding, output the exact phrase from the report that supports 'present', or NONE; only output present if you quoted a phrase."* Forcing a citation removes hallucinated positives that drive Lat-OA prec .29, Baker prec .42, Med-OA prec .43.
3. **3-state output → free soft labels:** change 0/1 to `{2=definite, 1=mild/possible/equivocal, 0=absent/normal}`, mapped 2→1.0, 1→0.5, 0→0.0. This absorbs the "mild effusion / trace fluid / tiny Baker's" cases that inflate FPs (Effusion prec only .68 because physiologic fluid is over-called) **and** yields a graded `q` for the soft fusion in one shot.
4. **Explicit normalcy/negation rule:** a finding on a "normal/unremarkable" template line, or qualified by *no / without / trace / physiologic*, is 0. (Directly targets Effusion and the OA over-calls.)
5. **Side-resolution rule:** *"assign a tear only to the named compartment; if the side is not stated, assign neither"* — Qwen's Med-Men rec 1.00/prec .57 shows it flags meniscal tears without resolving medial vs lateral.
6. **Few-shot, 3 exemplars** (normal knee / degenerative / acute trauma), one in **Spanish** to anchor multilingual negation. Keep short — 3B context budget on T4.

**Per-label sub-prompts: reject as the default.** 12 separate calls ≈ 12× cost (7.5 h → ~90 h, blows the 9 h limit). Single-pass all-12 JSON with evidence spans is the right cost/quality point. The one justified exception is a **cheap second pass on PF-OA only, restricted to studies where pass-1 was equivocal** (a subset, so affordable).

---

## (3) Qwen-7B on 2×T4 — YES, but as an ADJUDICATOR, not a full re-label

Reality check on the offline image: no bitsandbytes/autoawq (no 4-bit). 7B fp16 ≈ 15 GB OOMs a single T4; on 2×T4 it fits via `device_map="auto"` (naive layer split, no tensor parallelism) but runs ~2× slower → ~12–15 s/report → 4407 × 13 s ≈ **16 h > 9 h**. A single kernel can't full-label 4407 with 7B.

Recommendation:
- Keep **3B on all 4407** (the 7.5 h run already fits 1×T4).
- Run **7B (2×T4) only on the adjudication subset**: studies where rule and Qwen-3B disagree, or where 3B returned the equivocal state (≈30–40% of studies) → fits budget, targets the hardest cases, and is exactly where the **low-resource languages** (Turkish/Croatian/Greek/Bulgarian, 33.9% of corpus, where 3B is weakest) concentrate. 7B's real edge over 3B here is multilingual medical NLI on those languages.
- **Gate:** adopt 7B only if it beats 3B on gold-58 by >0.01 macro (bootstrap CI), and check the gain specifically on the non-English gold subset (tiny, n=2–6 each, so treat as directional).
- **Higher-ceiling experiment to flag:** upload `Qwen2.5-14B-Instruct` GGUF (Q4_K_M ≈ 9 GB) + a `llama-cpp-python` CUDA wheel as offline datasets. llama.cpp does its own quantization, sidestepping the missing bnb, and 14B-Q4 fits one T4. If the wheel installs offline, a 14B adjudicator is the strongest legal option. Test feasibility before committing.

---

## (4) Soft vs hard labels, confident learning, two-step training

**Soft labels: decisively yes.** The metric is ranking AUC; soft targets carry the extractor's confidence ordering into the image model, which is what AUC rewards. Hard 0/1 pseudo-labels discard the interior-rank information that produced the +0.058 above. Final targets land in a rich set `{0, .17, .33, .5, .67, .83, 1}` from the weighted 3-state fusion. Train the image model with **soft-target BCE / cross-entropy against `p`, not MSE.**

**Confident learning / OOF filtering with the 58 gold** (the "noise surgery" week):
1. **Pre-training, free signal:** rule↔Qwen agreement is a confidence label available before any image training. Agreement (both 0 or both 1) → sample weight 1.0; disagreement → 0.5. Use alongside the soft value.
2. **Calibration on gold-58:** fit a per-label monotone map (isotonic or Platt) from ensemble-score → P(true positive) on the 58; given n=58 prefer a shared logistic with per-label offsets over 12 independent fits. Report per-label AUC with bootstrap 95% CIs (MCL/Lat-OA/Baker's will be wide — quote them).
3. **Confident learning proper (Northcutt), after image v1:** get the image model's OOF predictions on all 4349 → flag studies where image prob and text pseudo-label are both confident and strongly disagree → these are candidate label errors → set to 0.5 (abstain) or down-weight → retrain v2. Do NOT derive deployment `pos_weight` from gold-58 (it's pathology-enriched); derive it from the pseudo-label distribution on the 4349 (rule priors, e.g. Fracture .047, Lat-OA .03, MCL .05) — but with AUC prefer mild focal loss (γ≈1.5) over aggressive pos_weight, which mainly hurts calibration.

**Two-step training** (RSNA-2024 playbook, per your strategy doc):
- **Step 1:** train full backbone + heads on **all** pseudo-labels (soft BCE, no gold) — noisy set teaches features.
- **Step 2:** freeze backbone, fine-tune only the 12 heads with gold-58 mixed in at high weight + confident-learning-cleaned pseudo-labels + metric-aligned loss. Gold-58 used **only here**, via leave-one-out / small-fold CV for early stopping, never in step 1 — keeps 58 examples out of the backbone.
- **Step 3 (distillation, the multimodal→unimodal move):** blend text pseudo-labels with image-v1 OOF predictions → sharper denoised targets → retrain. A good image model averages out per-study label noise, so its OOF signal genuinely improves the targets.

---

## Expected lift and the exact artifacts to build

**Expected macro-AUC of the training TARGETS on gold-58** (this is the label-pipeline objective — distinct from the eventual leaderboard AUC of the image model):
- Soft 3-level fusion: est. .789 under independence; discount for gold-58 small-n variance (the errors are complementary, not positively correlated, so little downward pressure from that) → **realistic 0.76–0.79, expected ~0.77 (+0.04)**.
- + PF-OA prompt fix + evidence-span FP reduction (lifts Qwen precision, finer ranks): **+0.01–0.02 → ~0.78–0.79**.
- + 7B/14B adjudication on the hard/low-resource subset: **+0.005–0.015 → ~0.79–0.80**.
- Ceiling: the community audit puts report-vs-gold agreement at ~82%, so text-label AUC tops out ~0.82–0.85. We're approaching the text-signal ceiling; further gains must come from the image model, not the labeler.

**Scripts/kernels to write:**
1. `kernels/qwen_extract/qwen_extract_v2.ipynb` — new prompt (evidence-span + 3-state + PF-OA vocab + few-shot incl. Spanish); **save per-study preds AND probabilities for all 4407 incl. gold-58 → `qwen_preds_full.csv`** (this is the re-run flagged below).
2. `ensemble_labels.py` (local) — load `qwen_preds_full.csv` + `extract_rule_v0` over all 4407 → per-label weighted soft fusion (§1 table) → `pseudo_labels_v1.csv` (soft) + `sample_weights.csv` (agreement weight). **Measures the real ensemble AUC on gold-58 with the true joint** (replaces my independence estimate). Gate: must beat 0.7313; target ~0.78.
3. `calibrate.py` (local) — per-label isotonic/Platt on gold-58; per-label AUC + bootstrap CIs.
4. `kernels/qwen7b_adjudicate.ipynb` (optional, 2×T4) — 7B on the disagreement subset only; gate >+0.01 on gold-58.
5. Image side (named, out of scope): `train_step1.py` (soft-BCE, all pseudo), `confident_learning.py` (OOF disagreement filter), `train_step2.py` (frozen backbone + gold-58 head FT + distillation blend).

---

## FLAG — the blocking measurement gap (as noted in the task)

The full ensemble AUC I report (~0.789) is an **estimate under conditional-independence of the two extractors' errors**, because **Qwen's per-study predictions on gold-58 are not saved** — only the marginal confusion matrix survived in the log, which is enough for the marginals but not for the true rule×Qwen joint. The fix is trivial: cell 3 of `qwen_extract.ipynb` already computes `pred` (the 58×12 per-study Qwen table); add ~3 lines to write it to `qwen_gold_preds.csv`. That re-run is ~6 minutes on T4 (355 s measured for the 58) and immediately makes `ensemble_labels.py` report the exact, not estimated, ensemble macro-AUC. Do this before trusting the 0.78 target. My independence estimate is likely a mild *under*-estimate (the mechanisms are complementary, so error correlation is ≤0), but n=58 gives it a ±0.03–0.04 CI.

Local artifacts written: `scratchpad/rule_gold.py`, `scratchpad/rule_gold_perstudy.csv` (rule per-study preds on gold-58, ready to join with the Qwen re-run), `scratchpad/ensemble_est.py`.