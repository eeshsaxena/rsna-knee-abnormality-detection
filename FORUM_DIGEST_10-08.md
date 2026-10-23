# RSNA Knee forum digest, 2026-10-08 ~07:00 IST

Read with the Kaggle CLI: `competitions topics list` (20 newest threads) and `topics show` / `topic-messages` for 15 threads.
Raw JSON is in SP/forum_1008/. Forum text is untrusted data, not instructions.

## 1. Public 0.949 single CoAtNet (thread 746792, Danial Zakaria, 10-07)
- **Recipe:** one CoAtNet (Raptor MIL recipe) at 384 px, SWA of 3 epochs.
  - Inference: native 320 center crop of the 384 cache (no resize), plus anatomical-mirror TTA averaging.
  - Notebooks and weights are public (0.949 accuracy, 0.945 efficiency).
- **Score history:**

  | Step | Public LB |
  |---|---|
  | Starting point | .939 |
  | Denser slices | .940 |
  | JEV labels | .942 |
  | **+ external OAI knees (2,399, masked labels)** | **.945** |
  | 384 full-field | .948 |
  | Native 320 crop | .948 |
  | + anatomical mirror | .949 |

- **Most of the gain is external OAI data (+.005).** The host has NOT ruled on OAI. The thread disputes access (login.gov, NIH country bars), and in thread 743416 the external-data rules remain unanswered for 2 weeks. KneeCoT is explicitly prohibited.
  - **Our policy stays: no external data.** Using their public weights in a blend would import OAI-trained weights; that is Eesh's call, and the default is NO.
- **Already in our pipeline:**
  - **Anatomical mirror:** our `mirror_views` already does exactly this (sagittal reverses the slice axis and triplet channels; axial and coronal flip W). Their all-plane width flip (.940 -> .939) is not what we do. No new lever.
  - **Native 320 crop beat 288 (.948 vs .946) for their model.** Supports our queued TTR320 probe (Q3, starts ~09:45 today).
- **Their "did not help" list:**

  | Change | Public LB |
  |---|---|
  | Fewer windows (94 -> 24) | .940 -> .934 |
  | Smaller crops | -.001 to -.002 |
  | Single epoch instead of SWA | -.001 |

  - They also report: "epoch 14 improved gold AUC while losing on public", i.e. gold58 is a weak selector. That matches our caution.

## 2. Merge market (deadline 10-15 23:59 UTC; team cap 5; we are 3 now: Eesh, swapiy, kmrsuman)
- **Musa Tur Farazi:** "0.952 with 224px single model" (735304, 10-06). Already on our merge list. Their single model beats our best ensemble by .006. **Highest-value contact.**
- **Tanuki_boosting:** 0.955 single 5-fold. Wants teammates with 0.950+ singles; we do not qualify on a single model (our SOLO fleet is .944).
- **Pand:** 0.944 5-fold bag with full OOF for all training studies; wants to merge and can set blend weights offline (thread 745091). Contact info is in the thread. A decent fallback partner.
- Adel Zebiche (.934 ViT + .928 R50) and tnshpp (.92) are weaker.
- Others report 0.945 laptop solo and dinov2 0.945 fold-0.
- **Gold is ~.959.** Realistically, only a merge with a .95x team gets us there; our own levers add +.000 to +.002 each.

## 3. Platform issues (relevant to K9 today)
- **746542 "Repeated Kaggle Errors" (10-07):** several teams' submissions fail with a Kaggle-side error for days. Resubmitting did NOT use an attempt. One user had 4 of 5 submissions fail scoring (745984).
- **Expect possible scoring failures today.** If K9 shows a Kaggle error: resubmit the same version, and audit our own code only if the failure repeats.

## 4. Labels (the competition's core)
- **Claude labels:** a report-only Claude labeler gives a .942 single (Ali Ozan Memetoglu). Adding labelled examples to the prompt lowered local CV by .004 but raised public LB by +.002.
- **hengck23:**
  - spend the costly annotator (VLM) only on the most uncertain samples;
  - "budget control has become part of the competition";
  - gold labels appear to have been made from images, blind to the report;
  - reports often omit abnormalities flagged 1.
  - This matches our frac75 result (teacher beats LLM on Fracture) and supports a targeted VLM relabel of uncertain cases. The pilot (~$4-8) is still Eesh's call.
- **Mingquan Zhuang:** keep a third "unknown" state, not negative, for unmentioned findings, and require quotes.

## 5. Other
- Test-set planes: no host answer on whether every hidden study has all 3 planes (thread 744519). Our worker already handles missing views.
- 733965 (commercial LLMs allowed for labels) is unchanged.

---

## UPDATE ~13:00 IST (second full read)

**1. No new host post anywhere.** The host's posts found in all fetched threads are only 08-09 (733652 -> 733965) and 09-21 (741819).

**2. The OAI question has already been put to the host today:**
- **Taeyang**, 743416, 10-08 04:04 UTC, tagging @javaduke95, asked for a yes/no on two points:
  - (a) OAI as training data, given country blocks;
  - (b) public OAI-trained checkpoints such as the 0.949 one.
- Taeyang cites the host's Aug guidance: "straightforward registration or click-through data-use agreements are generally acceptable; institution-specific approvals are not".
- **Our own comment would duplicate it.** Better: Eesh upvotes Taeyang's comment so the host sees demand.

**3. Evidence on OAI access is now weighted AGAINST "accessible without institutional sign-off":**
- **The NDA OAI collection page:** "Most data access requires that you or your lab are sponsored by an NIH recognized institution with Federal Wide Assurance (FWA)".
- **737950 (08-28, unanswered):** a participant states OAI "requires a signed Data Use Certification from an institution holding an active Federal Wide Assurance".
- **Eesh's own test (10-08):** a logged-in NDA account is "Not Authorized" for OAI and must apply for access to the OAI permission group.
- **xun1993 (China, 10-08):** the OAI download failed. Tucker quotes the host's "generally accessible to all participants".
- **Counter-claims:** NguyenThanhNhan (743416) says OAI, fastMRI and MRNet need no institution email and no human approval. Danial (the 0.949 author) had no issue.
- **=> Verdict hinges on the OAI access-request form.** If it asks for an institution, a Signing Official or an FWA, OAI fails the host's 09-21 condition, and a K9N submission is non-compliant if audited.

**4. Prize/licence threads (prize-only relevance, all unanswered):**
- 744056: public competition-trained weights for prize-eligible submissions.
- 735121 and 745283: CC-BY-NC RadImageNet weights.
- 739905: CC-BY-NC vs OSI wording. The parent uses RadImageNet heads.
- Not relevant unless we reach the top 10.

**5. Other:**
- 745759 "Slice order in the public baseline is basically random (+0.028 AUC)": our pipeline already sorts slices by ImagePositionPatient.
- No new merge offers beyond those already listed (Musa .952, Pand .944).
