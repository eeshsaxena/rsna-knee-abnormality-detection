# Forum check 10-19 (read 10-19 ~05:00 UTC). READ-ONLY. No em dashes.

Window: everything new or updated since **10-18 15:00 UTC** (FORUM_HINTS_10-18b.md) up to 10-19 ~05:00 UTC.
Purpose: go / no-go before today's 5 per-finding blend-weight submits (Fracture Raptor weight .9 / 1.0, PF OA .1).
Tools: session_2026-10-16/forum/fr.py (list RECENT 3 pages, NEW 2, TOP 1, HOT 1; comp) and msgs.py on the 6 moved
threads; session_2026-10-15/watch/kgweb.py anonymous GetDatasetBasics / GetKernelViewModel; `kaggle datasets list
--mine` for our private R6 copy. Nothing posted, voted, submitted, forked or pushed.
Raw: raw/ (l_*.json, msgs_*/batch_*.json, datasets.json, comp.json); flattened new posts: new_since.txt.
Tags: MEASURED (we fetched it) / CLAIMED (a third party said it) / GUESS (ours).

## Verdict: SUBMIT-SAFE: yes
No host or staff post, no scoring or queue report, no Fracture or per-finding weighting hint, and no change to any
dataset we attach. Nothing in the window affects today's submits.

## Coverage (MEASURED)
- 6 threads moved, 9 messages: 750529 (3, new topic), 750356 (2), 743416 (1), 744230 (1), 748181 (1), 748295 (1).
  Last post anywhere: 744230 at 10-19 04:19 UTC.
- Competition deadline unchanged: 2026-10-22T23:59:00Z (GetCompetition).

## Checklist
1. Host / staff posts: **none in the window**. Latest host/staff posts in the fetched threads are still Chris Carr
   (HOST, 743416, 10-14 13:18 UTC) and Ryan Holbrook (STAFF, 748181, 10-12). No ruling on external data, OAI,
   fastMRI, SKM-TEA, private split, reruns, final selection or stuck runs.
2. Scoring / queue problems: **none new**. 750318 (submit queue) and 748403 (stuck >24 h) have no new posts.
   The 1 h commit-queue rule from 10-18b still applies.
3. Fracture / per-finding weighting: **nothing** from any team.
4. Datasets (MEASURED, anonymous GetDatasetBasics; private copy via authenticated list), all at the known version:
   - heliosli/rsna-orthofoundation-fold0-epoch7-20261004: v1, updated 10-04 07:54 UTC
   - nartaa/rsna-knee-publication-swa-weights-20261007: v1, updated 10-07 19:26 UTC
   - nartaa/rsna-knee-s6fall-weights: v5, updated 10-17 06:04 UTC (the version already checked)
   - glorysing/rsna-knee-offline-codecs-r09: v1, updated 10-09 07:03 UTC
   - heliosli/rsna-knee-r6-frozen-fe055406-w1: v1, updated 10-15 12:02 UTC
   - eeshsaxena/rsna-knee-r6-fe055406-v1 (our private R6 copy): lastUpdated 10-15 12:56 UTC, size 823098827 (same
     byte count as heliosli w1)
   - heliosli blend notebook: still 1 version (10-15 17:03 UTC), 0.958; no new weight set.

## The 9 new messages (briefly)
- 743416 (10-18 23:42, RealEclypso, rank 3699, votes -2): another request to organizers on how Section 2.6 treats
  OAI-trained public checkpoints and the SKM-TEA anatomy localiser. No answer. heliosli still has **not** answered the
  10-18 question on whether R6 used fastMRI.
- 750356 (10-18 18:34, Mattia Angeli, MASTER, rank 207): his six weight releases used competition data only plus
  public pretrained backbones; M448 backbone is under the DINOv3 licence. Not in our kernel. Karan Bhatti thanked him.
- 744230 (10-19 04:19, rank 1270): asks the thread author to release DINOv2 checkpoints. Old thread on bf16 being
  emulated on T4/P100 (slow, hits time guards); not relevant to our already-scoring kernel.
- 750529 (new topic, M. Ehtasham, rank 281): title "There seems to be no need for an external dataset"; says he
  uses a "zoomed in method" (750529), gives no detail. One reply asking what he used.
- 748181 (rank 1079): complaint about copied notebooks. 748295 (rank 281): one-line joke. No content.

## Watch list (unchanged)
- IF a host/staff post rules against OAI-trained weights or SKM-TEA anatomy.pt, THEN re-open the hedge question
  with Eesh. IF heliosli says R6 used fastMRI (743416), THEN tell Eesh (R6 is in both picks). Next read before the
  last A15 read, 10-22.
