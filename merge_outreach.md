# RSNA Knee: team-merge outreach drafts (you post these yourself; merge deadline 2026-10-15 23:59 UTC)

Rewritten 2026-10-05 from the verified jump plan (`session` scratchpad `jump2/JUMP_PLAN_10-05.md` sec 5). Every number
below is MEASURED unless marked. The old drafts contained three claims the verification killed; they are listed at
the bottom so they don't creep back in.

**Rules check (rules line 144): "Privately sharing code or data outside of Teams is not permitted."** Before a merge,
share only PUBLIC scores and approach descriptions. No files, predictions or weights. Blend or correlation checks happen
only after merging.

**Submission cap:** we have **44** submissions (10-05 morning, plus whatever lands today). The stricter rule counts the
combined total against 5 x days running at the request date (about 340 on 10-05). A TWO-WAY merge fits with any of the
targets below; a three-way merge does not. Re-check the partner's count on their team page before posting.

**Check the live LB first.** Scores below are from 10-04/05 and may have moved.

---

## What we can honestly say (pick from this list)

- Public 0.943 (the public d4 stack, run as a 9 h-safe 2xT4 build).
- Our own model family: ConvNeXt-T on 2.5D slice triplets, 4 MRI views x 24 triplets, 140 mm crop, slices sorted by
  ImagePositionPatient, slice and view attention pooling, per-target heads. Two recipes (320 px, and a 224 px resize
  with heavy augmentation, 24 epochs, EMA).
- Single-fold solo LB: 0.932 (mix50 targets) and 0.923 (pure LLM labels). A TTR single-fold solo lands today, and a
  5-fold + TTA fleet solo on 10-06/07.
- Out-of-sample on the 58 official gold studies (all 74 gold studies are excluded from our training): 0.914 and 0.913
  for our two best single folds, 0.919 for their average.
- Training targets: 0.5 public LLM labels + 0.5 logit-matched public CoAtNet teacher. On the 58 gold studies the target
  itself scores 0.926 vs 0.893 for the LLM labels alone. Say plainly that the teacher's checkpoint was selected on those
  same 58 studies, so 0.926 is optimistic.
- Compute: a dedicated V100 16 GB (one fold in 4-5 h) plus Kaggle GPU quota.
- Infra: every submission is gate-checked; a strict mode makes a broken run error out instead of silently scoring the
  public stack.
- If they ask how much our models add: inside the public stack our members read 0.943 at weights .20-.30 (they are
  correlated with it). Their value to a partner is as an extra own-model leg, and that can only be measured after a
  merge.

---

## 1. Reply in /discussion/744511 to caoyupeng (Myo Min Htet team, ~0.949): top priority

> Hi caoyupeng, congrats on the 0.949 with your own models. Would your team consider a two-way merge before the 10-15 deadline?
>
> What we'd bring:
> - Our own ConvNeXt-T family (2.5D triplets, 4 views, slice/view attention). Single fold, it reads 0.932 solo on the public LB and 0.914 on the 58 gold studies, which we keep out of training. A 5-fold version with TTA is finishing now.
> - Label work, since your thread said labels are the bottleneck: mixing the LLM labels 50/50 with an image-teacher's predictions (logit-matched) lifted the target from 0.893 to 0.926 on the 58 gold studies. Caveat: that teacher was tuned on the same 58, so the real gain is smaller.
> - Spare compute: a dedicated V100 plus Kaggle quota, to train extra folds or seeds of your models.
>
> We're at 0.943 with 44 submissions, so the combined cap is fine for a two-way merge. Happy to compare public scores and approach details here; any blending would happen after merging, per the no-private-sharing rule.

## 2. Same thread, to mehmetbarankrat (ringbearer, ~0.948)

> Hi mehmetbarankrat, would you be open to a two-way merge before 10-15? We have our own ConvNeXt-T family (0.932 solo for a single fold, 5-fold + TTA version finishing now), a label-mixing approach that lifts the target from 0.893 to 0.926 on the 58 gold studies (optimistic, the teacher was tuned on them), and spare GPU (a dedicated V100 plus Kaggle quota). We're at 0.943 with 44 submissions. Happy to compare public scores and approaches here; blending only after merging, per the rules.

## 3. /discussion/743374 to prateekgroverin (~0.945)

> Hi Prateek, you mentioned looking for partners with labels work or a single model above 0.940. Our single-fold ConvNeXt-T reads 0.932 solo (a 5-fold + TTA version is finishing now), we have a label-mixing approach that lifts the target from 0.893 to 0.926 on the 58 gold studies (optimistic, the teacher was tuned on them), and spare GPU (V100 + Kaggle quota). Would you be open to merging before 10-15? Happy to compare public scores and approaches here first.

---

**Secondary:** aastik and pjmathematician (~0.945-0.946) already run the public stack plus a private fleet, so our
stack adds nothing for them; our fleet would only be a second private leg (+.001-.002 at most, GUESS).

**Timing:** post 1-3 now (10-05/06). Follow up around 10-06/07 with the 5-fold fleet's solo score. Aim to agree by
10-13 so there are 2 days for the mechanics. No reply by 10-10: move to the next target.

**Do NOT say (killed by verification on 10-05):**
- "Independent GPT labels on all 4,349 studies." That table scores 0.849 on the gold 58, below the public LLM labels.
- "0.927 on the 58." It is 0.926, and optimistic.
- "+0.009 solo LB for another team." That was a solo gain; inside a 0.942 stack it read 0.941-0.942.
- Any promise of gold, or a model-predicted number presented as measured.
- That SpeedSci's 0.947 is own-models only (it includes a public CoAtNet).

---

## UPDATE 2026-10-08 ~09:45 IST (use these numbers; the older ones above are stale)

- **Live LB (10-08 03:36 UTC):**
  - We are **.946, rank ~392 of 5,446** (46 teams tied at .946). That is now BRONZE.
  - Cutoffs: silver = rank 272 = **.949**; bronze = rank 544 = .945; gold = rank 20 = **.959**; top .964.
  - 92 teams sit at exactly .950. That is probably the public .949 CoAtNet notebook (OAI external data) blended into stacks.
- **We are 3 people** (Eesh, swapiy, kmrsuman), so a partner team must have <= 2. We have 53 submissions (combined cap is fine).
- **Facts we can state now (MEASURED):**
  - Own 10-member ConvNeXt-T fleet SOLO **.944** public; parent + fleet **.946**.
  - 288 px members beat 224 px on every fold so far (+.003 to +.007 OOF AUC).
  - A Fracture label fix (teacher-weighted Fracture target) lifts gold74 Fracture AUC by +.019 with the other labels flat; pre-registered gate passed.
  - Compute: a dedicated V100 plus 3 Kaggle accounts (~90 T4-hours/week).

## 4. Musa Tur Farazi (claims .952 with a single 224 px model, thread 735304, 10-06): TOP PRIORITY now
Reply in /discussion/735304 under Musa's comment (or DM):
> Hi Musa, impressive 0.952 from a single 224 px model. Would you be open to merging before 10-15? We have a 10-model ConvNeXt-T 2.5D fleet that reads 0.944 solo and 0.946 blended. It is built differently from the public CoAtNet/DINO notebooks, so it should add diversity to your model. We also have a Fracture label fix that passed a pre-registered gold check (+0.019 Fracture AUC on the gold studies), plus a dedicated V100 and ~90 Kaggle T4-hours a week to train extra folds or seeds of your model. We're a team of 3 with 53 submissions. Happy to compare public scores and approaches here first; any blending only after merging, per the rules.

## 5. Pand (0.944 5-fold bag with full OOF, thread 745091; also reachable by Discord/email per that thread)
> Hi Pand, saw you're open to merging. We're at 0.946 (own 10-model ConvNeXt-T fleet at 0.944 solo, plus a public stack). Your full OOF would let us set blend weights offline right after merging, and our fleet has full OOF too. We bring a V100 plus ~90 Kaggle T4-hours a week. We're a team of 3, so a two-way merge fits the cap of 5. Interested?

**Order:** Musa first (a single model stronger than our whole ensemble). Pand in parallel as the fallback. Myo, ringbearer and Prateek (sections 1-3) only if they already replied. Tanuki (.955) only takes 0.950+ singles, so skip. Deadline 10-15 23:59 UTC; aim to agree by 10-12.
