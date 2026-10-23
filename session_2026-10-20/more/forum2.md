# forum2.md: fresh forum + public LB lens, 10-20 (read 10-19 19:22-19:45 UTC = 10-20 00:52-01:15 IST). READ-ONLY.

Nothing was posted, voted, submitted, pushed, committed, forked, downloaded (data or weights) or created on Kaggle.
No box access, no GPU, no OAI rows touched (no OAI analysis was needed: no candidate passed the screen). No PREREG /
HANDOFF / PROGRESS / scheduled-task / a20 edits. No em dashes.
Tags: MEASURED = fetched or computed by us now (file named) / CLAIMED = a third party or an older note says so (source
named) / MODEL = our arithmetic / GUESS = judgement.
Thread URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/<id>
Window: everything since 10-19 10:00 UTC. Previous lenses covered to 10-19 ~16:15 UTC (session_2026-10-19/think/forum.md
to 11:38 UTC, session_2026-10-19/trick/leaders.md sec 5 to ~16:15 UTC); this lens re-reads the whole window anyway.

## 0. Bottom line

1. **No new concrete idea from the forum or the LB. There are none to propose.** (MEASURED coverage below.)
2. **No host or Kaggle-staff post since 10-14 19:56 UTC** (MEASURED: no listed topic has a HOST/ADMIN last commenter after
   10-14 22:08 UTC, which is the Discord sticky; the open questions on missing planes 744519, site split 734681 and the
   Synovitis / Effusion rubric 737155 have no new post: last posts 10-06, 08-14, 08-23).
3. **Three threads moved in the window, none actionable** (sec 2): a rules re-ask on public LLM-label datasets and hosted
   APIs (750752, unanswered), a question to colum2131 about his 10-05 Effusion / Synovitis table (745949), and a re-post
   of dreaddevelopment's public .924 Raptor widedense description (750529).
4. **Public LB: the .959+ inflow slowed and nothing points to a shared public trick** (sec 3). .959+ went 183 -> 189 in
   8.4 h (~17/day vs 20-28/day on 10-16..10-19); .962+ flat at 63; top score unchanged (.969). The 6 new .959 crossers
   have no public knee dataset among their 13 members. c4 slipped r199 -> r207 (6 crossers + 4 in-band passers - 2 band
   teams that left upward). The public .958 copy wave continues: 63 teams rose into .958 from below, 10 new teams.
5. **No new public resource above .958** (sec 4): no new knee notebook above .958 (best still goodpjw / heliosli .958),
   no change to any dataset c4 attaches, the new label tables and models are training inputs or weak (gold58 .902-.908).

## 1. Coverage (MEASURED)

- Topic lists via the anonymous reader `session_2026-10-16/forum/fr.py` (RECENT, NEW, HOT, TOP, 3 pages each = 107
  unique topics) and GetCompetition at 19:22 UTC: `forum_raw/l_TOPIC_LIST_SORT_BY_*.json`, `forum_raw/comp.json`.
  Deadline unchanged 2026-10-22T23:59:00Z.
- Diff vs the 10-19 lists (`session_2026-10-19/think/forum_raw*/l_*.json`): only 750752 (new), 745949 (+1 post) and
  750529 (+1 post) gained posts; 748181, 750318, 748276, 750334 changed votes only.
- Messages fetched with `session_2026-10-16/forum/msgs.py`: `forum_raw/msgs/msgs_{750752,745949,750529,744519,737155}.json`.
- LB: `lb/api_lb_1020T_more.json` (fetch_api_lb.py, 19:24 UTC) and `lb/api_lb_full_1020.json` (full teams + members,
  19:27 UTC); compared with `session_2026-10-19/think/work/api_lb_1019T_think.json` (11:00 UTC) and
  `session_2026-10-19/trick/work/lead_lb/api_lb_full_1019_now.json` (15:45 UTC).
- Public assets (metadata only, no download): `kaggle datasets list -s knee / -s rsna --sort-by updated`
  (`lb/ds_knee_p1.csv`, `ds_knee_p2.csv`, `ds_rsna_p1.csv`), dataset file lists, anonymous GetDatasetBasics
  (`lb/ds_basics_1020.json`), competition notebooks by run date and by score (`lb/kn_recent.csv`, `lb/kn_score.csv`),
  anonymous GetKernelViewModel on the 5 notebooks run since 10-19 plus goodpjw (`lb/kvm_1020.json`), key-author lists
  (heliosli, nartaa, goodpjw2008, bliverrigpdoge, stefan87, dreaddevelopment), movers' member dataset lists
  (`lb/user_ds_movers.txt`), `kaggle models list -s knee / -s rsna` and one model card (`lb/modelmeta/`).

## 2. The posts in the window (CLAIMED content, MEASURED timestamps)

| UTC | thread | who | content | use |
|---|---|---|---|---|
| 10-19 14:54 | 750529 | M. Ehtasham (topic author) | pastes the description of dreaddevelopment's public notebook: one CoAtNet 2.5D model, 5 fixed series slots (64 slices; sag fluid 18, sag non-fluid 14, cor fluid 12, cor 8, ax 12), slices sampled across 6-94 % of the stack, 140 mm physical crop resized to 336, CoAtNet at 384, per-finding attention over 42 windows; LLM soft report labels; public .924 single model, gold58 .9167; weights = `dreaddevelopment/raptor-knee-widedense` | known (this is the Raptor recipe nartaa's c4 member derives from); widedense already rated in the 10-19 lens (forum.md sec 5 row 3) |
| 10-19 15:56 | 745949 | voires | asks colum2131 (now r25, .964) for architecture, label encoding of "not mentioned / negative / equivocal", gold58 use, and per-study predictions | no answer; colum2131's 10-05 table (report "absent" Effusion 10/26 expert-positive, "equivocal/small" 14/18, Synovitis "not mentioned" 11/36) was used on 10-06/10-08 (RESEARCH_TO_TOP_10-06.md l.74-78, F_intel T19) and is the basis of the gold-calibrated S6fall labels already in c4 |
| 10-19 18:10 | 750752 | Abdul Wahab | rules question: may public CC0 LLM-label datasets (some made with hosted APIs) be used as training targets in prize-eligible entries; does Rule 4.b (Data Security) matter | unanswered; host 733965 (08-09, CLAIMED in think/forum.md sec 2) already allowed commercial LLMs on report text; prize-eligibility only, not medal ranking (GUESS); watch item |

Earlier in the window (already in leaders.md sec 5): 750529 07:10-09:03 chatter ("no external data", Civitasmass r190).

## 3. Public LB since 10-19 11:00 UTC (MEASURED)

| snapshot (UTC) | teams | >= .959 | >= .960 | >= .962 | >= .964 | >= .966 | .958 band | c4 rank | band teams above c4 |
|---|---|---|---|---|---|---|---|---|---|
| 10-19 11:00 | 6,209 | 183 | 121 | 63 | 26 | 13 | 1,218 | 199 | 15 |
| 10-19 15:45 | 6,217 | 185 | 121 | 63 | 27 | 13 | 1,254 | 203 | 17 |
| 10-19 19:27 | 6,219 | 189 | 122 | 63 | 27 | 13 | 1,281 | 207 | 17 |

- Movers 11:00 -> 19:27: 6 into .959+, 7 up within .959+, 63 into .958 from below, 49 up below .958, 10 new teams.
- Into .959+: Maxim & buddies (r196 -> r160), zx2, Query&Key, Lakshya Mewara (17 subs), ryota24 (13 subs), Khanh
  Nguyen (.951 -> .959 in one step, 8 subs). Within .959+: tomoon33 .962 -> .964, five teams .962 -> .963, Effusion
  confusion .960 -> .961, Pratham Sahu .959 -> .960.
- Their 13 members own **no public knee dataset** (`lb/user_ds_movers.txt`: only old non-knee datasets). Khanh Nguyen
  (kkhanhnguyn214) and Query&Key (kimkhanhnguyen214 + 2) crossed in the same window; similar handles (GUESS: related).
- In-band passers of c4: Ezra (r1123 -> r195, 58 subs), a Hanbat University team (r329 -> r200), Chris Deotte (GM,
  r212 -> r203, 154 subs), Colhomy (r217 -> r205). All .958, so their unrounded scores now exceed c4's; no public asset
  explains it.
- Top 18 unchanged in score (.969 RB & Matheus & Tucker & Tony; .968 x3).
- Rate (MODEL): +6 at .959+ in 8.4 h = ~17/day, below the 20-28/day of 10-16..10-19 (leaders.md sec 1). Scores land
  ~6.5 h after submit, so this reflects submissions up to ~13:00 UTC; the busier 18-06 UTC submit window (leaders.md
  1.1) will show by ~10-20 06:00 UTC.
- Projection (MODEL): 76.5 h to the deadline at 17-25/day -> ~243-269 teams at .959+; with ~17-25 band teams above c4,
  c4 alone would sit at public r ~260-295. Silver cut ~r311 (5 % of ~6,230, MODEL). E-OA (in flight) at the brief's
  +.0005-.0007 on c4's unrounded ~.9587-.9589 (MODEL, brief) would display .959 and land inside the .959 group, so
  the 12:00-14:30 IST read is the event that matters, not anything here.

## 4. Public resources (MEASURED metadata; nothing downloaded)

- **Notebooks.** Run since 10-19 00:00 UTC: evgendvorkin/rsna-versia-7 (still v22 .956, NS+OF+R7; the 15:51 run did
  not create a scored version), seanmoch tri-reader (.950: goodpjw 2.5D reader + nartaa SWA), toghrulaliyev AZE (fork,
  unscored, Qwen2.5-7B + DINOv2-L, private inputs), basith007 (unscored, private inputs), shivsatyam cache builder. Top
  by score unchanged: goodpjw .958 (v27, 10-16), lavinwins copies, heliosli 0.9586 / blend-mine-in. heliosli's new
  "Try merging it" notebook (10-18 23:56) is an ARC-AGI-3 notebook, not knee.
- **Datasets c4 attaches: unchanged** (GetDatasetBasics): heliosli R6 frozen fe055406 v2 (10-14) and -w1 v1 (10-15),
  nartaa s6fall-weights v5 (10-17 06:04), nartaa publication-swa v1 (10-07), heliosli orthofoundation v1 (10-04),
  glorysing codecs v1 (10-09).
- **New knee datasets since the leaders scan (10-19 ~15:45 UTC):**
  - cyruspinto/rsna-knee-consensus-labels-public (19:08 UTC, CC0, one file SUPER_CONSENSUS_labels.parquet 630 KB).
    Owner team CYRUS PINTO .958 r634 (30 subs). A label table: training input only.
  - imran2002/knee-emb-s4-p0..p3 (18:10-18:24 UTC): per-study .npy embeddings named by StudyInstanceUID; owner not in
    the .959 group (imranxahmed .949 r2342 is the nearest name match, GUESS). Not usable as an inference member.
  - Earlier on 10-19, not in the leaders table: jorgerome/rsna-knee-weights-v2 (08:35, DINOv2 fold0, 7 downloads; team
    "Jorge Rome" .510 GUESS match), beginner080/rsna-report-context-final-20261019 (03:12, inference .py + MODEL.json;
    owner NoahRyan .949 r2373, MEASURED via the 15:45 member map).
- **New public model:** patimakornsrinamen/rsna-67knee-coatnet2 (2.5D CoAtNet-2 224, LLM-consensus labels): its card
  claims gold58 .908 (t4) / .902 (t5) and a 4-arm fusion public .936 (CLAIMED, `lb/modelmeta/model-metadata.json`).
  Far below every c4 member.

## 5. Ideas surfaced by this lens and their verdicts

None reaches PROMISING. Each was screened against an independent ruler or LB history before any compute was spent.

| # | idea (source) | ruler | evidence | expected hidden macro | P(>0) | cost | verdict |
|---|---|---|---|---|---|---|---|
| 1 | Add dreaddevelopment widedense (or finespacing) at .1-.15 in the Raptor slot (750529 re-post, 10-19 14:54) | LB history + mechanism | widedense solo gold58 .9167 / public .924 (CLAIMED 750529); same CoAtNet-2.5D recipe as nartaa's Raptor in c4, so little diversity (GUESS); our 10-04 v9 finespacing swap was a visible no-op (CLAIMED HANDOFF via think/forum.md row 3). OAI ruler not usable cheaply: needs a 543 MB weight download (Eesh's OK) plus box inference, and c4's Raptor is OAI-trained so a c4+X OAI check is contaminated | ~0 (MODEL) | ~.3 (GUESS) | download + ~25 min hidden runtime + 1 slot | DEAD |
| 2 | Expert-regime fluid coupling at inference (Effusion / Synovitis report-vs-expert table, 745949 re-surfaced 10-19 15:56) | OAI expert reads + nested gold58 (already run) | Syn<-Eff nested gold58 -.0005 (MEASURED label_regime.md m5); OAI-only fluid passes reverse on RSNA (EX_1019 C16, Syn<-Eff -.0084); fluid group decorrelation KILL (EX_1019) | < 0 | ~.3 | 0 | DEAD (closed; no new evidence) |
| 3 | Train on a new public label table (cyruspinto SUPER_CONSENSUS 10-19 19:08; srinath2495 dual-teacher; bliverrigpdoge trainkit) | LB history of owners | owners at .958 r634 (cyruspinto), .959 r184 (trainkit, crossed before publishing); our own trained members all flat or negative on LB (CLOSED list); train + commit + score does not fit before 10-22 ~10:15 IST | 0 within the window | n/a | days of V100 | DEAD (time) |
| 4 | New public weights as a small member (patimakornsrinamen CoAtNet-2, jorgerome DINOv2 fold0) | card gold58 (CLAIMED) + owner LB | gold58 .902-.908, fusion public .936; jorgerome team .510 (GUESS match) | ~0 | ~.1 | download + runtime + slot | DEAD |
| 5 | Copy whatever the 6 new .959 crossers used | LB + member asset scan | no public knee dataset among 13 members; no notebook > .958; no host hint | n/a | n/a | n/a | DEAD (nothing public to copy) |

## 6. Watch list / if-then (unchanged from 10-19 except 750752)

- IF a host or staff post appears on external data (743416, 750236, 750257-750260), missing planes (744519), the private
  split (734681) or the Synovitis rubric (737155) before 10-22, THEN tell Eesh at once.
- IF 750752 gets a ruling that hosted-LLM label datasets are not prize-eligible, THEN note it for the disclosure only
  (R6 targets are report-derived; medal ranking unaffected, GUESS).
- IF a later LB read shows a cluster of crossers after a new public release, THEN redo the sec 3 member scan; the
  10-20 ~06:00 UTC read will include the busy overnight submit window.
- Next forum + LB check: before the A15 read on 10-22 (same scripts, `forum_raw/` lists as the new baseline).

## 7. Files (all new, under session_2026-10-20/more/)

- `forum_raw/`: l_TOPIC_LIST_SORT_BY_{RECENT,NEW,HOT,TOP}.json, comp.json, msgs/msgs_{750752,745949,750529,744519,737155}.json
- `lb/`: api_lb_1020T_more.json, api_lb_full_1020.json, ds_knee_p1.csv, ds_knee_p2.csv, ds_rsna_p1.csv,
  ds_basics_1020.json, kn_recent.csv, kn_score.csv, kvm_1020.json, kvm_heliosli_trymerging.json, user_ds_movers.txt,
  modelmeta/model-metadata.json
