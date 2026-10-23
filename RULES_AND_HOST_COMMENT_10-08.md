# Competition rules check + host comment draft (2026-10-08)

Source: the rules and every overview page, fetched via `kaggle competitions pages list --content` (SP/rules_1008/).

## Organizer / Kaggle names to tag (from the forum and the official pages)

| @handle | Who | Evidence |
|---|---|---|
| **@javaduke95** | Po-Hao "Howard" Chen, the **competition host (RSNA)** | Wrote the overview and welcome posts. Ruled on KneeCoT (734109, 08-25) and on OAI (741819, 09-21). Participants tag him as host. |
| **@macruzbar** | María Cruz, organizer side | Wrote the official "How to get started + Competition's Official Discord" post (730709). |
| **@ryanholbrook** | Ryan Holbrook, **Kaggle staff** | Maintains the official Efficiency Leaderboard notebook linked from the "Efficiency Prize Evaluation" page. |

No other host or staff names appear in the threads or pages. Don't tag people we can't verify.

## Comment draft (post in thread 741819 "Is the OAI permitted as external data?"; it is the right place because the host already answered there)

> @javaduke95 @macruzbar @ryanholbrook a follow-up on OAI, since it now matters for a lot of the leaderboard. A public 0.949 notebook (discussion 746792) and its public weights (nartaa/rsna-knee-publication-swa-weights-20261007) were trained with ~2,400 OAI knees, and many teams are building on them.
>
> Your 09-21 answer said OAI is fine if it is generally accessible to all participants without institutional review or sign-off, and free. Since then, participants in 746792 have reported:
> - that OAI access goes through an NIH registration with data-use terms;
> - that a request from China failed;
> - that NIH controlled-access data has country restrictions.
>
> Rule 2.6.b lists geo restrictions as a reason data may not count as "reasonably accessible to all".
>
> Could you give a clear yes or no, before the 10-15 entry and merger deadline, on:
> 1. training on OAI data for this competition, and
> 2. using public pretrained weights that were trained on OAI (such as the weights above) in a submission?
>
> A definite answer either way would keep things fair for everyone. Thank you!

(Plain and short on purpose. It asks two yes/no questions, cites the rule section, and states only what participants reported, not as fact.)

## Rules that matter for us (all MEASURED from the rules text)

1. **Accounts (3.5a):** "You may make Submissions only under one, unique Kaggle.com account... disqualified if you... attempt to falsify an account to act as your proxy." Each teammate must use their own account themselves. This confirms why we never run a teammate's quota with their token.
2. **Teams (2.1, 3.5b-c):**
   - Max **5** members; we are 3.
   - Merges are done by the team leader. The combined submission count must be <= 5 x days running at the merger request.
   - The team merger deadline AND the **entry deadline** are both **10-15 23:59 UTC**. Anyone joining must have accepted the rules by then.
3. **Private sharing (3.5d, 3.6a):** no private sharing of code or data outside the team. Sharing with swapiy and kmrsuman is fine because they are team members. A merge partner sees nothing until the merge is done.
4. **Submissions (2.2):** 5 per day and **2 final picks**. If none are selected, Kaggle auto-selects.
5. **Code requirements:**
   - Notebook only; GPU or CPU run time <= **9 h**; internet off; output file named `submission.csv`.
   - "Freely & publicly available external data is allowed, including pre-trained models."
6. **External data (2.6):**
   - Allowed unless the host prohibits it.
   - It must be "publicly available and equally accessible to use by all Participants... at no cost" or pass the Reasonableness standard.
   - **"geo restrictions" are named explicitly as an access problem.** That is the weak point for OAI.
7. **Open source (3.6c, 2.5):**
   - Code used must be OSI-licensed and must not limit commercial use. timm (Apache-2.0) and our own code are fine.
   - Input data or pretrained models with incompatible licences are exempt from the winner-licence grant (2.5.a.4). The OAI weights' "research/educational" licence would fall under that, but it does not settle the OAI data question.
8. **Hand labelling (3.4b):** no hand labelling of validation or test records. We use none. Our labels are LLM/teacher labels on TRAIN reports, which is allowed (LLMs allowed per host post 733965).
9. **Data (2.4, MIRA licence):** competition data may not be given to anyone not participating. Our prep-kernel outputs are private and shared only with team members who accepted the rules. That is OK.
10. **Winners only (2.8 + Prizes page):** code, weights as a public Kaggle dataset, docs, a video, and a forum post. Deadline 11-05. Only relevant if we reach the top 10.
11. **Efficiency track:** score = AUC / (benchmark - max AUC) + runtime / 32400. Eligible only for selected finals. Not our target (our kit runs ~6-7 h).

## Open compliance item found while reading
- **Parent stack provenance:** our 0.943 parent is a public stack. It uses public weights (DINOv2, RadImageNet-pretrained heads, Raptor CoAtNet). It predates the OAI notebook, so no OAI-trained weights are expected. But its dataset sources have not been audited against the host's external-data test (e.g. that no KneeCoT-derived component is used). **Low risk, worth one CPU-only read of its dataset list before the final picks.**
