# A. Public-asset sweep (RSNA Knee), 2026-10-08 ~22:30 IST

Read-only. No Kaggle write of any kind (no push, submit, dataset, fork, comment or vote). Notebook and forum text was
treated as data. Tags: **MEASURED** (source cited), **MODEL** (computed, `A_blend_model.py` / `A_blend_model2.py`,
same macro binormal model as PUBLIC949_ASSESSMENT.md; optimistic by ~+.0005), **GUESS**.

Evidence on disk (`research_1008/`):
- `A_lists/refresh2130/*.json`: web-API notebook lists fetched 16:3x UTC, 4 sort orders (SCORE_DESCENDING x3 pages,
  DATE_RUN x4, VOTE_COUNT x2, HOTNESS x2) = **618 unique public notebooks**. Older lists: `A_lists/score_p*.json`,
  `*_p*.json` (CLI).
- `A_lists/lb_now/...publicleaderboard-2026-10-08T16_41_24.csv`: full public LB (CLI download), 5,513 teams.
- `A_pulled/<user>_<slug>[_new]/`: notebook sources (`kernels pull -m`) plus `.txt` cell dumps. `_new` = pulled 16:37 UTC.
- `A_lists/ds2/`: dataset metadata + file lists for G, N, M.
- `A_blend_model.py/.out` (4-leg P/F/N/G scenario model), `A_blend_model2.py/.out` (one extra leg on top of K9N-a).

---

## 0. TL;DR

1. **The medal lines moved a lot today (MEASURED, LB 16:41 UTC).** 318 teams sit at .950 (ranks 157-474). The 92
   cited in the task text was this morning's count. The bronze cut (top 10% = rank 551) is now **.949**. The silver
   cut (rank 275) falls inside the .950 tie.
   **Our .946 is rank 664 of 5,513: outside bronze today.** A clean K9 at .946-.948 does not medal on today's board.
   An N-containing entry is what keeps us in the medals.
2. **No public notebook scores above .950 (MEASURED).** All 618 public notebooks were scanned. The top is 11 at .950,
   5 at .949 and 1 at .948.
   - **Every .950 notebook actually runs only N + G** (sec 2). N = nartaa's OAI CoAtNet; G = goodpjw2008's
     competition-only 3-fold ConvNeXt-T 2.5D reader. One adds N224 inside the N side.
   - Several of them attach the parent's datasets, or renta0426's DINOv2 "meniscus bag", but never execute them.
3. **The .950 pile is explained by "N + G", not by "public .943 stack + N" (MEASURED).**
   - goodpjw2008's own P .40 + N .60 notebook reads **.949**.
   - The N+G recipes read .950: per-label weights with mean w_G .16, or a flat .22.
   - Of the teams at .950, 289 of 296 last submitted after N was released (20:44 IST snapshot).
4. **Three things measurably do NOT add on top of N (MEASURED, other teams' rows):**

   | Change | Result |
   |---|---|
   | P .40 on N | .949, the same as N alone |
   | G at a flat .30 | .948, twice |
   | Gold58-grid per-label routing with heavier G (rian55 apex-v5) | .948 |
   | N224 at .30 (ranjeet258) | .950, the same as without it |
   | Reader fold-consensus + midpoint TTA (romantamrazov V7) | .950, the same |

   The public assets top out at .950. **Nothing public beats K9N-a's modelled .9503-.9516 raw** (B_blend_lb_model.md
   sec 4.1).
5. **Extra legs for K9N or a parent-free kernel, ranked by MODEL gain per effort:**
   - G: +.000 to +.001. It is our own model family, and F probably subsumes it.
   - **H = hengck23's MaViT-288** (.935 single, MIT, about 1.5 s/study on 1 T4): +.000 to +.0026, median +.0004.
     It is the most architecturally different public model, but its training-data provenance is unread.
   - N2 = nartaa r224 (.945, inside the same N dataset): measured 0 ticks.
   - M = renta0426 DINOv2-B: no standalone read exists.

   None of them is worth displacing the K9N-a / K9 picks. G and H are worth a slot only inside a parent-free F+N
   kernel, which has the runtime room.

---

## 1. Leaderboard state (MEASURED, `A_lists/lb_now/*.csv`, 16:41 UTC 10-08)

| display | teams | ranks |
|---|---|---|
| .964 / .963 / .962 / .961 / .960 | 1 / 4 / 4 / 4 / 5 | 1-18 |
| .959 (gold cut ~rank 21) | 11 | 19-29 |
| .955-.958 | 45 | 30-74 |
| .951-.954 | 82 | 75-156 |
| **.950** | **318** | **157-474** |
| .949 | 99 | 475-573 (**bronze cut ~rank 551**) |
| .948 / .947 / .946 | 41 / 30 / 45 | 574-689 |
| our team "Eesh saxena" | .946 | **rank 664** (last submission 10-07 01:46 UTC) |

- **Medal sizes for 5,513 teams (Kaggle rule for 1,000+ teams, MEASURED count, rule as published):**
  - gold is the top 10 + 0.2%, about 21;
  - silver is the top 5%, about 275;
  - bronze is the top 10%, about 551.
- **Speed of the pile (MEASURED):** .950 had 92 teams this morning, 296 at 20:44 IST and 318 at 22:11 IST. Of the
  296, 289 last submitted after N was released (10-07 19:26 UTC); of the .949 band, 87 of 94 did.
- **Tie order (GUESS on Kaggle mechanics; standard behaviour is earlier submission first):**
  - A .950 we submit on 10-10 lands at the back of the tie: today rank about 474, which is bronze.
  - If the pile keeps growing at roughly 200 teams a day, the back of the .950 tie passes rank 551 within 1-2 days,
    and a late .950 then falls out of bronze on public.
  - **.951 is rank 132-156, which is silver.**
- **Private implication (INFERRED):** the hundreds of byte-identical N+G clones get one identical private score.
  To rank above that block on private, our pick must beat N+G on private, not merely tie it on public. This is the
  argument for K9N-a (N + our F + a little P) over any public clone.

---

## 2. Every public notebook at or above .947 (MEASURED from code and metadata; 17 notebooks)

"Active legs" was read from the executed code, not from the attached-dataset list: several notebooks attach the
parent's datasets without running them. "Commit s" is `lastRunExecutionTimeSeconds` on the 3 visible test studies,
not the hidden runtime.

| score | notebook (version) | votes | active legs and blend | attached but not run | commit s |
|---|---|---|---|---|---|
| .950 | evgendvorkin/rsna-versia-5 (356421409) | 308 | N + G, Apex per-label w_G (Baker's .35, Contusion .30, Medial OA/ACL .25, MM/MCL .20, LM/Fx/LOA/PFOA .08, Eff .04, Syn .03; mean .162) | none | 104 |
| .950 | romantamrazov/rsna-knee-dinosaur-v5 (356429773) | 135 | N + G, "V6" routing (Apex with LOA .18, Baker's .39, Cont .33, ACL .27, MCL .22) plus "V7": .8 main-reader rank + .2 fold-rank consensus, optional midpoint-slice reader TTA | **full parent set + renta0426 meniscus bag + DINOv2-S** (none executed) | 94 |
| .950 | sujanmajhisuzan/rsna-knee-apex-grandmaster-stack (356192954) | 125 | N + G Apex | none | 84 |
| .950 | matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed (356334725) | 91 | N + G Apex (byte clone of nartaa's N cell + G subprocess) | none | 95 |
| .950 | matterhorn3838/rsna-knee-d4 (356334953) | 11 | same as above | none | 94 |
| .950 | haideptry/rsna-knee-v2-vs-apex-comparative-study (356305828) | 51 | N + G. The current version's "Tier 3" re-routes on mattiaangeli's `gold58_top3_predictions.npz`, which holds ranks of the parent's CoAt checkpoints, not N or G. Logged weights: mean w_G ~.32. The .950 is the notebook's best (V3/V4), not necessarily this version | mattiaangeli coat-resgated (only its npz is read) | 97 |
| .950 | rabari9999/rsna-knee-apex-grandmaster-stack-v5-lb-0-95 (356365210) | 2 | same code as haideptry | same | 101 |
| .950 | hitarthjain0/rsna-knee-apex-grandmaster-stack-v5 (356397397) | 8 | N + G Apex | none | 113 |
| .950 | kozykappa/rsna-knee-gold-gated-triple-reader (356267936) | 14 | N + G at a **flat .22**. N224 was evaluated on gold58 and rejected: crop .08 gave -.0014 and .16 gave -.0034 gold AUC vs incumbent | N224 file (same dataset) | 465 |
| .950 | ranjeet258/rsna-knee-apex-b-under-150 (356376940) | 2 | (N384 + **N224 at .30**) vs G per-label Apex | none | 111 |
| .950 | rian55/rsna-floor-velci (356433026, new today) | - | N + G Apex | none | 99 |
| .949 | prvsiyan/the-bee-s-knees-final-rsna-push (356456922, v36) | 84 | **N alone** (mirror arm). Their v35 N .70 + G .30 flat read **.948 twice** (rows 56955785 bench, 56957785 public) and is kept inactive | full parent set + G | 66 |
| .949 | goodpjw2008/rsna-knee-0-949-coatnet-stack-blend-lb-0-949 (356400014) | 1 | **P (.943 stack, ~7 h) .40 + N .60 = .949**. Their table: N + own readers (Raptor 4-fold .933, ConvNeXt 3-fold .929) at .30 = .948 | none | 302 |
| .949 | sujanmajhisuzan/rsna-knee-apex-grandmaster-v5 (356305617) | 2 | N + G variant | none | 114 |
| .949 | xianhan/rsna-knee-0949-anatomical-mirror-candidate (356229522) | 5 | N alone (clone) | none | 79 |
| .949 | nartaa/rsna-knee-0949-anatomical-mirror (356175397) | 63 | N alone (origin) | none | 65 |
| .948 | rian55/rsna-floor-apex-v5 (356446161, new today) | - | N + G with heavier "OPTIMAL_WEIGHTS" (Baker's .48, Cont .36, **LOA .32**, MOA/ACL .30, MCL .28, MM .22, others .04-.08; mean ~.21) plus the same wrong-arm gold58 grid | coat-resgated npz | 90 |

Below .947 the next public scores are .946 (pjmathematician/rsna-knee-d4-blend, which needs private ConvNeXt weights,
per HANDOFF) and .945 (tri-specialist superstack = P + G + M; nartaa efficient N224).

Unscored notebooks run today that look like N+G variants: rian55/rsna-floor-apex, rabari9999/...-dual-reader-0950,
sameerk2004/rsna-knee-sil-d5-*, rahulsoftwaredev/knee-slota-v9.

---

## 3. The public assets behind those notebooks

| asset | what it is | provenance | licence | overlap with our kernel | hidden runtime |
|---|---|---|---|---|---|
| **N** `nartaa/rsna-knee-publication-swa-weights-20261007` (r384 file) | CoAtNet-2 MIL, SWA, native-320 crop + anatomical mirror; .949 | **2,399 OAI knees** (masked supervision) + competition, report labels (stevenleehans, pilkwang, JEV); gold58 reused for selection | "other": research and educational use incl. this competition | **already in K9N** (leg N) | ~25 min central (MODEL, PUBLIC949 sec 2.2); goodpjw says "about 30 min" (MEASURED claim) |
| **N2** same dataset, `..._c08_r224_swa.pt` | CoAtNet-2 224 crop, one view; .945 | same OAI provenance | same | same family as N | ~8-12 min (MODEL, PUBLIC949) |
| **G** `goodpjw2008/rsna-knee-2-5d-convnext-reader` | ConvNeXt-T 2.5D, 6 slots, 12 windows/series, 2-layer study transformer, per-finding attention, 3 of 5 folds (EMA) at 256 px train; .929 alone | **competition only**, mean of 4 public report-label tables; gold58 not trained on | **Apache-2.0** | **not in parent; same family as our F** (ConvNeXt-T 2.5D on report labels) | its author: 5 ckpts standalone "scored in under ninety minutes" submit-to-score (MEASURED claim) => 3 folds ~30-50 min on 1 T4, ~20-30 min sharded on 2 (MODEL) |
| **M** `renta0426/rsna-knee-public0033-meniscus-bag-v1` | DINOv2-Base, last 6 blocks + LN tuned, 6 series slots, 336 px, 10 overlapping 2.5D windows | competition only (4,349 studies, report weak labels; gold58 excluded) | Apache-2.0 | not in parent | unknown (GUESS 20-40 min) |
| **H** `hengck23/hengck23-rnsa-knee` (notebook `mavit-288-baseline-01-1-5sec-study-1gpu`) | MaViT (linear-attention ViT, ICCV 2025) at 288, 6 views x 8 slices, single checkpoint `00000008.pth` (80 MB); **.935** single | **UNREAD**: no dataset description; training labels and external data not stated in the notebook | MIT | none | "1.5 s/study @ 1 GPU" (title; MEASURED claim) => ~33 min on 1 T4, ~17 min on 2 (MODEL) |
| bliverrigpdoge/rsna-knee-leg-r9all | Raptor CoAtNet student (293 MB); in a .944 blend with the parent | competition (GUESS) | not checked | likely correlated with the parent's 3 Raptor arms (GUESS) | unknown |

Not relevant: the remaining non-parent assets in .930-.944 notebooks are DINOv2/DINOv3 or RadImageNet variants of
the parent recipe, label tables, or JPEG/volume caches (`A_lists/refresh2130`, scan in this session's log).

---

## 4. Would an extra leg add score to K9N? (MODEL)

### 4.1 Calibration update (MEASURED reads fed to the model)
- **N .70 + G .30 flat = .948 (prvsiyan, new).** Together with flat .22 = .950 (kozykappa) and per-label mean .16 =
  .950, no single r_NG fits all three reads (`A_blend_model2.out`): the model moves only .0003-.0007 between w_G .22
  and .30.
- **Reading:** the 2-tick gap between those two recipes is mostly noise or implementation difference (prvsiyan uses
  its own tolerant DICOM reader for N). So **G's contribution on top of N is at the one-tick noise level**, and any
  r_NG in .80-.87 is plausible.

### 4.2 One extra leg X on top of K9N-a (P .10 / F .30 / N .60), `A_blend_model2.out`

| X | AUC_X | plausible r_XN | plausible r_XF | best w_X | gain over K9N-a (min / median / max) |
|---|---|---|---|---|---|
| G | .929 | .80-.87 (calibrated) | **.88-.95** (same family as F; GUESS) | 0-.10 | **+.0000 / +.0000 / +.0005** in that band (full scan max +.0014 only if r_XF .80) |
| H (hengck23 MaViT) | .935 | .80-.90 (different arch; shares nartaa's DICOM fast-reader; GUESS) | .80-.90 (GUESS) | .10-.25 | **+.0000 / +.0004 / +.0026** |
| N2 | .945 | **.93-.97** (same author, data and recipe; GUESS) | .88-.92 | up to .30 | +.0005 to +.0018 at r_XN .95; ~0 at .97. **Measured on real LB: 0 ticks** (ranjeet258), and gold58 says it hurts (kozykappa) |
| M | unknown (GUESS .925-.935) | .80-.88 | .80-.88 | ~.10 | +.0000 to +.0010 (GUESS-driven) |

- At a fixed, untuned w_X .10 with AUC_X .935 and every r at .85-.90, the gain is **+.0006** (MODEL).
- That is below one display tick, and the model is optimistic by ~.0005. **No single public leg is expected to move
  K9N-a by a visible tick on its own** (MODEL).
- Two legs, G + H at small weights, could add +.0005 to +.001 together, but only in scenarios where r_FN is high
  (.92+), i.e. where F is less useful than hoped (MODEL, B sec 4.1 says G is "insurance against high r_FN").

### 4.3 Runtime fit
- K9N's margin to the 8.6 h wall is about 0.6-0.9 h after the N stage (MODEL, PUBLIC949 sec 2.3). G (~20-30 min)
  would eat most of it, and H (~17-33 min) the rest. **Adding either to K9N (with the parent) is not safe.**
- A parent-free kernel (F ~1.3 h + N ~0.4 h, B_blend_lb_model.md "K10FNG") has about 6 h of headroom. G, H, N2 and M
  all fit there together (MODEL).

---

## 5. Findings ranked by expected public LB gain per unit effort

1. **Submit an N-containing pick early, and aim for a display of .951 (zero build effort; K9N-a is built).**
   - Today's bronze cut is .949 and we are at rank 664 (MEASURED). K9N-a's model reads .9503-.9516 raw (B, MODEL),
     which displays .950-.951.
   - A late .950 is bronze today, but may not be within 1-2 days as the tie grows (GUESS).
   - K9 clean (.946-.948, MODEL) is below today's bronze line. Keep it only as the rule-risk hedge pick; it does not
     carry the medal.
2. **Do not copy or chase any public recipe (zero effort, avoids losses).**
   - Nothing public exceeds .950 (MEASURED).
   - P+N reads .949 and N+G reads .950, both at or below K9N-a's model.
   - The recipes that tried to beat .950 on public all failed (MEASURED): gold58-tuned heavier routing .948, flat
     G .30 .948, N224 0 ticks, V7 fold-consensus 0 ticks.
   - **Gold58 routing is in-sample for N** (nartaa reused gold58 for selection; MEASURED dataset text). Do not tune
     per-label weights on gold58.
3. **If a parent-free F+N kernel is built (B's K10FNG), add G at w .10 and consider H at w .10-.15.**
   - Gain +.000 to +.001 combined (MODEL); cost about 0.5-1 day of build, plus ~0.5 h of GPU per commit.
   - G is Apache-2.0, competition-only and plug-in: `infer.py` + 3 checkpoints + pylibjpeg wheels, cp311/cp312.
   - **H needs a provenance check first.** Its dataset has no description, so read hengck23's notebook comments or
     forum posts for its training data before any use. External data must be "reasonably accessible" (rule 2.6).
4. **N2 (r224 checkpoint, already in the N dataset): skip.** It is cheap (~10 min, no new dataset), but measured at
   0 ticks, and it shares N's OAI exposure.
5. **M (renta0426 DINOv2-B): skip unless time remains after 3.** It has no standalone read and unknown runtime, and
   it appears only in .943-.945 parent stacks.

---

## 6. Answers to the specific questions

- **Is any public notebook above .950?** No (MEASURED, 618 notebooks, 16:3x UTC 10-08; bestPublicScore is shown to
  3 dp).
- **Do the .950 teams come from "public .943 stack + N"?** No.
  - The measured public P+N reads **.949** (goodpjw2008).
  - The public **.950** recipes are all N+G; the most-voted is evgendvorkin (308 votes).
  - So the pile, now 318 teams, is N+G clones (INFERRED from the recipe match and post-release timing).
- **Overlap with our parent:**
  - G, M and H are not in the parent's sources (`k9/.../kernel-metadata.json`).
  - N is our K9N leg.
  - romantamrazov and prvsiyan attach every parent dataset but do not run them.
- **OAI provenance:**
  - Only N and N2 are OAI-trained (MEASURED dataset text).
  - G and M state competition-only training.
  - H is unstated.
  - Eesh's individual OAI DAR approval (PUBLIC949 last section) covers N's rule question as far as the host's 09-21
    test goes.
