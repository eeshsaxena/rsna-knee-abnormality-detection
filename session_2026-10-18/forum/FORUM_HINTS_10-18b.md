# Forum hints 10-18b (read 10-18 ~15:00 UTC = 20:30 IST). READ-ONLY. No em dashes.

Window: everything new or updated since **10-17 19:45 UTC** (the 10-18 01:15 IST read, FORUM_10-18.md) up to
10-18 ~15:00 UTC. Tools: our own anonymous readers (session_2026-10-16/forum/fr.py list: RECENT 3 pages, NEW 2,
TOP 1, HOT 1; msgs.py on 30 threads) plus kgweb.py anonymous GetDatasetBasics / GetKernelViewModel for the datasets
and notebooks named in posts. Nothing posted, voted, submitted, forked, created or pushed. No public code pulled
(nothing in the window had code worth pulling), so pulled_hints/ was not created.
Raw: raw_b/ (l_*.json topic lists, msgs_*/batch_*.json), flattened new posts: new_since_b.txt.
Tags: MEASURED (we fetched it or read it in our own files) / CLAIMED (a third party said it) / GUESS (ours).
"rank N" = the poster's live competition rank from the forum API field `competitionRanking`.

## Coverage (MEASURED)
- 7 threads moved in the window, 16 messages (1 deleted): 748181 (6), 750355 (3, new), 750318 (2), 750334 (2),
  743416 (1), 750356 (1, new), 750357 (1, new). Newest topic is 750357 (10-17 22:37 UTC); last post anywhere is
  748181 at 10-18 09:25 UTC; nothing between 09:25 and 15:00 UTC.
- Top-15 most-voted threads re-checked in full (748181, 733343, 743416, 735304, 733932, 733497, 746792, 745861,
  745798, 733826, 733965, 736170, 747658, 733652, 748111, plus 737696, 749359, 730709, 750236, 750257-750260,
  750295, 748403): no new message except the 748181 and 743416 ones listed below.
- Host / Kaggle-staff posts: the complete list across all 30 fetched threads ends at **10-14 19:56 UTC** (Dustin,
  748403). Still no ruling on external data, private split, reruns, final selection or stuck runs.
- No post from anyone at .959+ in the window with any modelling content. The only posters ranked <= 170 were
  Satwik (rank 83), ForcewithMe (137, GM) and OzanM (156, GM), and none of them said anything technical about models.

## Top actionable hints, ranked by expected value for us

### 1. Commit/submit queue: do not sit on a stuck commit (CLAIMED, 750318)
- 750318: 4-6 h commit queues reported 10-17 18:07-18:27 UTC (already known). New: OzanM (GM, rank 156) at 21:38 UTC
  says 15-20 min is normal and "if it exceeds 1 hour, I cancel and start the kernel again".
- Value: protects the remaining A16/A15 slots from being lost to a queued commit in the last 4 days. Cost: none.
- IF one of our kernel COMMITS (version run, not a competition submission) sits queued > 1 h, THEN cancel and re-push
  it once (same notebook, same pinned inputs). IF a SUBMISSION (Submit to Competition) is queued, THEN leave it: we
  have no evidence that cancelling refunds the daily slot (GUESS), and 748403 shows staff clear stuck ones in ~3 days,
  too late for 10-22. Rule unchanged: push >= 4 h before any submit; last readable submit 10-22 ~08:45-10:15 IST.

### 2. External-data wave keeps growing but still no host answer; our picks are unchanged by it (MEASURED + CLAIMED)
- 10-17 22:35-22:37 UTC Karan Bhatti (rank 2219) opened three more provenance threads, all downvoted (-6 to -8):
  750355 pilkwang/rsna-knee-weights, 750356 mattiaangeli weights (knee-mri-fold-weights, coat-resgated, radimagenet
  heads, coatnet-d4/global96, cnx-m448), 750357 antoinegg1 RadImageNet e9/e11 heads. MEASURED: none of these datasets
  is in our HW kernel (session_2026-10-16/hw/kernel_v224r6hws6fj/kernel-metadata.json attaches only our R6 copy,
  nartaa publication-swa, heliosli orthofoundation, nartaa s6fall, glorysing codecs; pilkwang appears only as a
  markdown credit line).
- 743416 (10-18 04:58, nishchal jain, rank 1154): a full public disclosure table for a heliosli-blend-based team (R6
  fe055406, anatomy.pt / SKM-TEA, nartaa OAI CoAtNet, four LLM label tables). It asks heliosli whether R6 used fastMRI,
  noting "Its manifest lists no `fm_aux` entry". No reply yet. MEASURED earlier (RESEARCH_1017): our R6 v1 manifest
  has no `fm_aux`, ImageNet init.
- IF heliosli answers 743416 that R6 fe055406 used fastMRI in any way, THEN tell Eesh: R6 is in BOTH picks (A and B),
  and fastMRI needs an affiliation + intended-use request, closer to the "institutional" side of the 10-14 ruling
  than OAI is. Even then no clean replacement exists (see hint 3), so the action is information only.
- IF a host or staff post rules against OAI-trained public weights or SKM-TEA anatomy.pt for medals, THEN re-open the
  hedge question with Eesh at once. ELSE keep A15 picks (A = HW 57262115, B = R6 57248084).

### 3. A "clean" hedge pick is still far below the medal line (CLAIMED + MEASURED, 750355)
- Pilkwang (GM) answered 750355 at 10-18 07:25 UTC: the 20 checkpoints are one DINOv2-small model (4 seeds x 5 folds)
  on competition MRI only; per his run records "no external medical data" was used; README added. MEASURED via
  GetKernelViewModel: his notebook pilkwang/rsna-knee-baseline-v1 scores **0.891** public (v15, unchanged since 08-08).
- Combined with the dreaddevelopment Raptor baseline (.936-.938 public, CLAIMED in 737696), the best no-external-data
  entry we could build is ~.94 (GUESS), versus a bronze line near .956. 
- IF anyone proposes burning a final pick on a clean hedge, THEN the answer stays no unless the host rules (hint 2).

### 4. No bait or dataset swaps in the window (MEASURED)
- Anonymous GetDatasetBasics on 10-18 ~15:05 UTC, every dataset in our picks is at the version we know:
  nartaa publication-swa v1 (10-07), heliosli orthofoundation v1 (10-04), nartaa s6fall v5 (10-17 06:04, the
  training_code update already checked), glorysing codecs v1 (10-09), heliosli r6-frozen-fe055406-w1 v1 (10-15).
- heliosli "Blend mine in" notebook: still 1 version (10-15 17:03 UTC), 0.958, same three inputs. No new per-finding
  weight set published.
- IF a new kernel of ours must attach any of these after today, THEN pin the dataset VERSION ids above and assert the
  model sha as the HW kernel does (cell 7 style); ELSE nothing.

### 5. Soft-label "98-99 % accuracy" datasets: ignore (CLAIMED, 750334 / 750295)
- OmerZalman (rank 5292) 10-17 21:19 UTC: testers saw AUC "increased ... by around 0.02-0.1". Votes -11 and -12.
  MEASURED: omerzalman/rsna-knee-11-label-evidence-fusion v2 (10-17 15:49 UTC); the two newer datasets are not linked.
- Not usable: training is closed for us, the accuracy is scored on the public expert set (selection on the eval
  set), LLM/external provenance adds disclosure risk, and +0.02-0.1 AUC is not credible at .95+ (GUESS).
- IF a .959+ team (not this poster) confirms a gain from these labels, THEN still no action before 10-22 (no
  training slot); note it for the post-mortem only.

## Everything else (briefly)
- 748181 (6 new posts): community debate on public blends and medals. Amanie El-Hajj (rank 317, 20:05 UTC): no
  difference "between the 100th to 1000th position" (our band; tie-break by submission time stands, 749570).
  ForcewithMe (GM, 137) and Satwik (MASTER, 83) blame auto-upvotes on copy; Ravi Ramakrishnan (GM) says Kaggle will not
  act; D Chakraborty joke post; Mega2814s (first comp) complaint about AI text. No staff reply, no rule change.
- 750318: also a TPU/LLM-abuse rant (Muhmad KwaZulu gorilla, -1), irrelevant.
- 750355: one deleted comment (3540444, 23:55 UTC), content gone.
- Named in posts, none new to us: heliosli/blend-mine-in-private-lb-luck-to-all-upvoters, heliosli R6 w1,
  heliosli orthofoundation, nartaa publication-swa, LLM label tables (stevenleehans, pilkwang, lixin73 sol56,
  dreaddevelopment), pilkwang/rsna-knee-baseline-v1, sofiaanjenje e11/e13 training notebooks (named in 750357).
- Per-finding weighting, rank vs probability blending, label tricks, TTA, post-processing: **nothing new from any
  team** in the window. The all-time picture in session_2026-10-17/research_1hr/forum.md section 4 stands (top
  teams credit their own labels and own 5-fold CNNs; nobody disclosed fusion weights or TTA).

## Verdict
No score lever in the window. Keep A15 picks; apply the 1 h commit-queue rule; watch 743416 for a heliosli answer on
R6 + fastMRI and the forum for any host ruling (next read before the last A15 read, 10-22 ~17:30 IST).
