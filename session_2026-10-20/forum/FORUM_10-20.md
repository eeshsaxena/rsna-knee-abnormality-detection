## BRIEF 10-20 (forum read 04:17-04:25 UTC + LB snapshot 04:17 UTC; read-only, verified against raw/ and lb_bands_1020.out)
1. SUBMIT-SAFE: YES. ACTION: none required on c4 57343435, eoa20 57413368, eoa10 57413380 or the A15 picks (A=57262115, B=57248084).
2. Host/staff: one post. Maria Cruz (Kaggle admin), 747455, 10-19 22:46 UTC: a non-ruling. Kaggle will not interpret third-party licenses or pre-approve datasets or weights; she points to Chris Carr's 10-14 line (registration + license OK, institution-gated not); each team owns compliance.
3. Effect: no new ban, no new permission. OAI-trained nartaa weights in c4 and fastMRI-derived weights stay GRAY, Eesh's call (EXTERNAL_DATA_VERDICT.md). No ruling expected before the deadline. No other host post since 10-14.
4. Rule changes: none. Deadline unchanged (10-22 23:59 UTC). Nothing on final selection or rerun rules.
5. New participant posts, all unanswered: LLM-label datasets for prizes (750752), GPU quota top-up with a claimed 10-24 reset (750790), label dataset repost (750791), fastMRI / SKM-TEA weights (750889), OAI-tuned nartaa ruling request posted after the staff non-ruling (750899).
6. Useful hints: none. No technical content from anyone ranked above c4.
7. Watch list: nothing new on missing planes (744519), site split (734681) or Synovitis rubric (737155); no heliosli or nartaa post on R6 / fastMRI; colum2131 has not answered voires (745949); no OA compartment or label-regime talk.
8. LB bands (6,249 teams): top still .969; .960+ = 127 (r1-127); .959 = 66 (r128-193); .958 = 1,337 (r194-1530). .959+ went 189 -> 193 since 10-19 evening.
9. Us: c4 r217 at .958 (r199 at 10-19 10:41, r207 at 19:24, r208 at 20:17). 23 .958 teams sit above it (18 at 20:17). Silver line r312, margin 95.
10. Since 20:17, 9 teams passed c4: 4 crossers into .959+ (Yoshiki Nakamura .957 -> .961, Yann Majewski .958 -> .960, Winter Lamb and Bone Appetit -> .959) and 5 in-band .958 passers. None fell below us.
11. Pending: eoa20 and eoa10 both PENDING at ~04:20 UTC (~4.3 h after submit); expected to score ~06:30 UTC. A .959 would place inside r128-193.
12. Projection (estimate): .959+ inflow slowed to ~12-14/day (from 20-28); c4 alone would end near r250-305 vs silver ~r312. Thin margin; the ~06:30 UTC read of the E-OA scores is the one that matters.
13. Optional: if more notebook commits are planned, check our own GPU quota (the 10-24 reset is a participant claim).

---

# FORUM_10-20.md: forum read, 10-20 04:17-04:25 UTC (09:47-09:55 IST). READ-ONLY.

**SUBMIT-SAFE: YES.** Nothing in the forum changes the rules, the data, the metric or final-selection mechanics. c4 57343435,
eoa20 57413368, eoa10 57413380 and the A15 picks (A=57262115, B=57248084) are unaffected.
**ACTION: none required.** One FYI for Eesh: Kaggle staff (Maria Cruz) answered the OAI-weights question on
**10-19 22:46:13 UTC** with an explicit non-ruling (747455): no per-dataset or per-weights approval will be given, each team
is responsible for compliance. So the external-data status of our picks stays exactly where EXTERNAL_DATA_VERDICT.md put it
(GRAY, Eesh's call); no ruling is coming before the deadline. Optional: if any more notebook commits are planned, check our
own weekly GPU quota (a participant claims it resets 10-24, after the deadline; 750790, unanswered).

Nothing was posted, voted, followed, submitted, pushed or downloaded. No PREREG / HANDOFF / PROGRESS / scheduled-task edits.
Tags: MEASURED = fetched now / CLAIMED = a poster says so / GUESS = judgement.
Thread URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/<id>

## 1. Coverage (MEASURED)

- GetCompetition 04:17 UTC: deadline unchanged 2026-10-22T23:59:00Z (`raw/comp.json`).
- Topic lists RECENT / NEW / HOT / TOP, 3 pages each = 107 unique topics (`raw/l_TOPIC_LIST_SORT_BY_*.json`), via
  `session_2026-10-16/forum/fr.py`. Diffed against the 10-19 19:22 UTC lists (`session_2026-10-20/more/forum_raw/`).
- Changed since then: 4 new topics (750790, 750791, 750889, 750899), 1 thread with a new post (747455, staff), 1 new reply
  (750790); vote-only changes on 733343, 733965, 748181, 745949, 748403, 748011, 735304, 733826, 733517. Four old topics
  fell off page 3 (745826, 738156, 737950, 745823); last posts 10-04, 08-30, 10-09, 10-06, so pagination only.
- Messages fetched with `session_2026-10-16/forum/msgs.py` into `raw/msgs/` for every thread with posts since 10-19 16:00
  UTC plus the watch list: 747455, 750899, 750790, 750889, 750791, 750752, 745949, 750529, 744519, 734681, 737155, 743416
  (+13 collapsed via batch), 744056, 750236, 750257, 750258, 750259, 750260, 733965.
- Ranks: full public LB with members at 04:19 UTC (`lb/api_lb_full_1020T_forum.json`, `lb/api_lb_1020T_forum.json`).

## 2. Host / staff posts

| UTC | thread | who | content (paraphrase) | effect |
|---|---|---|---|---|
| **10-19 22:46:13** | 747455 "are public weights trained on OAI knee MRI allowed?" | Maria Cruz (@macruzbar, Kaggle STAFF, ADMIN) | apologises for the delay; Kaggle will not interpret third-party DUAs/licenses or pre-approve datasets or weights; points to Chris Carr's 10-14 RSNA post (743416 #3538535: registration + license acknowledgement is fine, institutional agreement or affiliation-gated access is not, be reasonable and disclose); each team is responsible for compliance with the rules and third-party terms; good luck with final submissions | closes 747455 as a non-ruling. No new ban, no new permission. Status of OAI-trained public weights (nartaa publication-swa in c4) and of fastMRI-derived weights unchanged: GRAY |

No other host or staff post since 10-14 (MEASURED: only 747455 has an ADMIN/HOST last commenter after 10-14 22:08 UTC).
Chris Carr's two 10-14 posts (743416 13:18, 744056 13:20: public weights trained on competition data do not block awards)
remain the operative host guidance.

## 3. Other posts since 10-19 16:00 UTC (CLAIMED content, MEASURED times, live public rank at 04:19 UTC)

| UTC | thread | who (rank) | content | use |
|---|---|---|---|---|
| 10-19 18:10 | 750752 | Abdul Wahab (r5028, .814) | asks if public CC0 LLM-label datasets (some from hosted APIs) are OK in prize-eligible entries, Rule 4.b | unanswered; already on the 10-19 watch list; prize-only question (GUESS) |
| 10-19 19:54 | 750790 | Minal kharat123 (r3394, .938) | asks for a GPU quota top-up; claims weekly reset is 10-24, after the deadline | unanswered; one reply 10-20 03:10 by D Chakraborty (r190, .959) suggesting Colab Pro, no info |
| 10-19 19:55 | 750791 | OmerZalman (r1167, .958) | complains about downvotes; re-links his label dataset omerzalman/rsna-knee-11-label-evidence-fusion | training-label table only; not usable before the deadline |
| 10-20 02:25 | 750889 | Marvin O (r1795, .956) | asks whether fastMRI (form asks for an institution) and SKM-TEA pretrained weights are allowed in a final submission | unanswered; same question as 750259 / 743416; staff 747455 says no per-dataset answers will be given |
| 10-20 03:56 | 750899 | manduinca (r2236, .950) | asks for a ruling on public checkpoints fine-tuned on OAI (nartaa publication-swa) for Leaderboard / Efficiency prizes, citing the KneeCoT ban (734109) and the unanswered 737950 | unanswered; posted 70 min after the staff non-ruling, so very unlikely to get a different answer (GUESS) |

No post since 10-19 16:00 UTC from any poster ranked above c4 with technical content. No concrete technical hint in the window.

## 4. Watch list status

| item | thread | last post | status |
|---|---|---|---|
| missing planes in test | 744519 | 10-06 02:17 (Satwik, r98) | no host answer, nothing new |
| public / private site split | 734681 | 08-14 14:08 | no host answer, nothing new |
| Synovitis rubric | 737155 | 08-23 20:37 | no reply at all |
| external data / OAI | 743416, 747455, 750236, 750257-750260, 750899 | 747455 staff 10-19 22:46 | answered with a non-ruling (sec 2); rest unanswered |
| heliosli on R6 / fastMRI | 743416, 750259 | 750259 10-17 18:49 (Dread Development) | no heliosli or nartaa post anywhere in the fetched threads (MEASURED) |
| OA compartment independence / expert-vs-report label regime | 745949 | 10-19 15:56 (voires, r614, question to colum2131 r28) | no answer from colum2131; no other thread on OA compartments or label regime |
| final selection / rerun rules | none | n/a | no thread; staff post only says choose final submissions under the rules and host guidance |

## 5. Effect on our picks and the E-OA entries

- E-OA (eoa20 57413368, eoa10 57413380): nothing in the forum touches OA labels, compartment coupling, test data or scoring.
  No effect.
- Picks (c4, A15 A/B): the only rule-relevant item is the staff non-ruling. It neither bans nor clears OAI-trained or
  fastMRI-trained public weights; risk stays GRAY and Eesh-owned, as before. Host guidance that weights trained on the
  competition data are fine for awards (744056) still stands.
- LB context (MEASURED 04:19 UTC vs 10-19 19:27): .959+ 189 -> 193, .960+ 122 -> 127, .962+ 63 -> 65, .964+ 27 -> 30,
  .966+ 13 -> 15, top .969 unchanged, 6,249 teams. c4 is still our best entry (submissionId 57343435, .958) and slipped
  r207 -> r217 (4 new .959 crossers plus 6 more .958 teams now above it). eoa20 / eoa10 have not replaced c4 as team best
  yet (consistent with not scored yet, ~6.5 h latency, expected ~06:30 UTC).

## 6. Files (new, under session_2026-10-20/forum/)

- `raw/comp.json`, `raw/l_TOPIC_LIST_SORT_BY_{RECENT,NEW,HOT,TOP}.json`
- `raw/msgs/msgs_{747455,750899,750790,750889,750791,750752,745949,750529,744519,734681,737155,743416,744056,750236,750257,750258,750259,750260,733965}.json`, `raw/msgs/batch_743416.json`
- `lb/api_lb_1020T_forum.json`, `lb/api_lb_full_1020T_forum.json`
