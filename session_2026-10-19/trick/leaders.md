# leaders.md: do the .959+ teams share a trick? (10-19, written ~21:45-22:30 IST = 16:15-17:00 UTC). RESEARCH ONLY.

Nothing was built, pushed, committed, submitted or created. No GPU, training or box job. No PREREG, HANDOFF, PROGRESS or
scheduled-task edits. No public code was downloaded or executed. No em dashes.
Tags: MEASURED = computed or read by us now (script / file named); CLAIMED = a third party or an older note says so
(source named); MODEL = our arithmetic on stated assumptions; GUESS = judgement.
All paths below are relative to `session_2026-10-19/trick/` unless they start with `session_` or `_data`.

Reads done this session (all read-only):
- 1 anonymous LB API read (full team objects incl. members + tiers): `work/lead_lb/api_lb_full_1019_now.json` (15:45 UTC).
- 1 `kaggle competitions leaderboard --download` (the public LB CSV, 199 KB zip, Kaggle itself):
  `work/lead_lb/rsna-knee-abnormality-detection-publicleaderboard-2026-10-19T15_45_06.csv`.
- `kaggle competitions submissions` (our own 98 submissions, used only to map submissionId -> submit time):
  `work/lead_lb/our_subs2.csv`.
- Kaggle API metadata only: dataset lists (`-s knee`, `-s rsna`, per-user lists for .959+ members), dataset FILE LISTS
  (names/sizes, nothing downloaded) for 4 datasets. Forum: 1 topic list + 2 threads via the anonymous reader
  (`session_2026-10-16/forum/fr.py`, `msgs.py`, run with `python -I`).
- 60 stored LB snapshots 10-01 .. 10-19 (CSV 10-01..10-08, API 10-08..10-19), merged in `work/lead_ts.parquet`
  (`work/lead_scripts/ts_build.py`).

## 0. Answer in plain language

1. **There is no shared public trick visible in the leaderboard (MEASURED).** No wave, no cluster of jump dates, no new
   public dataset whose release lines up with jumps, and every author of a recent public model (B-leg, Mycelis,
   cyruspinto, riadmohamed42, jessky, heliosli, nartaa, goodpjw) is still at .958 or below on their own team.
2. **What predicts reaching .959+ is having had an independent pipeline above the public frontier BEFORE the public
   R6 wave (MEASURED, sec 3).** Of the 70 teams that were at .953-.958 on 10-12 00:00 UTC (at or above the best public
   notebook of that day, <= .954) and are still on the board at .958+, 58 (83 %) are at .959+ now. Of the 653 teams that sat at .956 on 10-15 06:00 (the public .956 copy
   crowd), 18 (2.8 %) are. Teams arriving at .958 via the public notebooks rarely cross; teams with their own strong
   models plus the public blend do.
3. **The leaders' edge is training (labels + own k-fold CNNs), and it is cheap at inference.** The top cohort climbs
   steadily at ~.0007-.0013 macro per day with no step (MEASURED); the efficiency LB has .964-.965 entries with tiny
   runtimes (Claudio Verdu Ruiz .964, RB/Tucker team .965, MEASURED file `session_2026-10-17/research_1hr/pulled/
   timing/efflb/full_leaderboard.csv`). That is the forum's story too (own labels, image pseudo-labels, own CNNs).
4. **So there is nothing public and legal we are "missing" that we can drop in.** What the jumpers visibly have that we
   lack is their own trained models: their LB best before the public R6 wave was .946-.957 (MEASURED); that those
   entries were own models and not public copies is MODEL for the .946-.954 ones (below the public best then, so not a current public copy) and
   GUESS for the .956-.957 ones. Our own best solo is .944
   (K7i, MEASURED our_subs2.csv) and every own member we added was flat or negative on LB. Closing that ~.01 solo gap
   needs a new label + training loop, which cannot be built, committed and read before the deadline.
5. **Rank slip is structural, not a sign of a missed public trick.** .959+ grows ~20-28 teams/day since 10-16 (MEASURED);
   c4 sits ~18th inside the 1,254-team .958 band (rank 203 at 15:45 UTC, down from 149 on 10-17 02:07). At that pace
   expect rank ~250-290 public by the deadline if we stand still (MODEL: 185 + 3.3 days x 20-28 crossings, plus the
   band teams still above us). Most new .959 teams have an independent model, so they likely beat us on private too (GUESS).
6. **Verdict: DEAD for "a shared trick", WEAK overall.** The one cheap measurement left (Mycelis OOF CSVs on CPU) needs
   Eesh's OK to download, and the LB evidence (its author still .958 r1331 after 15 subs) already argues against it.

## 1. LB time series (MEASURED, `work/lead_scripts/ts_build.py`, `work/lead_ts.parquet`)

| snapshot UTC | teams | >= .959 | >= .960 | >= .962 | >= .958 | new .959 per day | new .962 per day |
|---|---|---|---|---|---|---|---|
| 10-01 06:52 | 4,702 | 2 | 2 | 0 | 8 | | |
| 10-04 09:27 | 5,082 | 9 | 4 | 2 | 20 | 2.3 | 0.6 |
| 10-08 16:41 | 5,513 | 29 | 18 | 9 | 37 | 4.6 | 1.6 |
| 10-14 05:27 | 5,956 | 71 | 55 | 27 | 104 | 7.6 | 3.3 |
| 10-15 05:47 | 6,014 | 81 | 66 | 33 | 122 | 9.9 | 5.9 |
| 10-16 06:22 | 6,065 | 105 | 71 | 37 | 204 | 23.4 | 3.9 |
| 10-17 05:18 | 6,120 | 132 | 87 | 43 | 807 | 28.3 | 6.3 |
| 10-18 06:22 | 6,164 | 159 | 102 | 51 | 1,158 | 25.9 | 7.7 |
| 10-19 10:41 | 6,209 | 183 | 121 | 63 | 1,398 | 20.3 | 10.2 |
| 10-19 15:45 | 6,217 | 185 | 121 | 63 | 1,439 | (5 h) | |

Reading:
- The .958+ count exploded 122 -> 1,439 from 10-15 to 10-19: that is the public R6 .958 notebook wave (heliosli
  R6 frozen fe055406 dataset 10-14 18:12 UTC, the .958 notebooks 10-15/16; 1,060 forks, `session_2026-10-17/
  research_1hr/notebooks.md`). It is a .958 wave, not a .959 wave.
- The .959+ rate stepped from ~8-10/day to ~23-28/day on 10-15/16, i.e. at the same moment, then stayed flat. That
  step is the public .958 base being handed to teams that already had something of their own (sec 3), not a new
  resource at .959.
- The top keeps moving: .962+ grows 4-10 teams/day and is accelerating slightly.

Our team (MEASURED): r1138 (.942, 10-01) -> r149 (.958, 10-17 02:07) -> r189 (10-18 06:22) -> r199 (10-19 10:41)
-> r203 (10-19 15:45). Band position: 185 teams at .959+, so c4 is ~18th of 1,254 in the .958 band.

### 1.1 Submit-time of the jump (MEASURED, `work/lead_scripts/events.py`, `cross_events` in `work/lead_cross_events.csv`)
- submissionId -> submit time: piecewise-linear map from our 98 own submissions (IDs strictly monotone in time;
  ~42.6k IDs/day at the end; extrapolated past 10-19 04:44). Check: over 3,162 teams the mapped best-submission time
  exceeds the team's true lastSubmissionDate by at most 2.2 h (1st percentile 1.2 h), so the error is ~1-2 h.
- For each team's FIRST crossing to .959+ we date the crossing submission. Counts per 6 h bin since 10-14 12:00 UTC run
  1-13, typically 3-9, with a clear day/night rhythm (18-06 UTC busier) and **no spike** after any release:
  - B-leg R6 fine-tunes (10-18 12:42-14:19 UTC) and Mycelis 4-fold (10-18 17:08): 17 crossings from 10-18 12:00 to
    10-19 08:00, vs 24 / 14 / 27 in the same 20 h windows starting 10-15 / 10-16 / 10-17 12:00. No change.
  - nartaa S6fall weights (10-15, v5 10-17 06:04 only added training code): no spike either.
- Before 10-14 we cannot date crossings well (no API snapshot 10-09..10-13; a later best hides the first crossing), so the
  apparent 10-13 cluster is an artefact of that gap, not a wave.

### 1.2 Mergers (MEASURED, `work/lead_scripts/mergers.py`)
- Team-merger deadline 10-15 23:59 UTC (`teamMergerExplicitDeadline`, `session_2026-10-19/think/work_forum/pages/
  GetCompetition.json`). 34 of the 185 .959+ teams have more members now than on 10-08 (CSV `TeamMemberUserNames`).
- 4 crossing events look like mergers (the new best submission predates the previous snapshot by > 7 h). The big one:
  ~~Zo3ama~~ .946 -> .963 between 10-15 19:50 and 10-16 00:07, best submission dated 10-13 18:35, team 1 -> 3 members.
  Mergers explain a few big jumps; they are closed now.

## 2. Who crossed since 10-15 (MEASURED, `work/lead_scripts/jumpers.py`, `classify.py`, `effort.py`;
   table `work/lead_crossers_since1015.csv`, full listing `work/lead_classify.txt`)

111 teams first reached .959+ in a snapshot on or after 10-15. Previous displayed score: .958 for 80, .957 for 18,
.956 for 5, .954-.955 for 4, .946-.949 for 4.

| group | n | definition | land .959 / .960+ | solo teams | median subs total | subs between first .958 and first .959 snapshot (median) |
|---|---|---|---|---|---|---|
| A | 35 | at .958 before the public wave (first .958 < 10-15 15:00) | 16 / 19 | 24 | 84 | 6.5 |
| B | 28 | .957 before 10-15 06:00, then .958, then .959 | 16 / 12 | 16 | 104 | 1.5 |
| C | 48 | reached .958 only after the public wave | 26 / 22 | 34 | 60 | 3 |

- Group C is the "public notebook + something" crowd. 22 of 48 land at .960+, which needs about +.0013 over a ~.9587
  base: far more than a lottery on small weight tweaks gives (sec 2.1).
- Typical group-C profile: a team with its own pipeline at .946-.957 before 10-15 that jumps once it combines with the
  public R6 blend. Examples (score before -> after, subs): hirokenn .948 -> .959 (10-15) -> .964 (41 subs, solo);
  MJ121212 .949 -> .960; imainjob .946 -> .959; Leslie Yeh .950/.954 -> .960 -> .963 (26 subs); Andrey Serzhenko
  .954 -> .960; Archit Konde (own 2.5D CoAtNet, .950 single fold, CLAIMED 735304) .957 -> .960; Nadeem .954 -> .960;
  66game .954 -> .960; Donato Riccio .956 -> .960.
- One timing cluster worth noting: three Contributor-only teams with 20-22 submissions each (liminhai 5 members,
  Leka ZHANG 5, liyupengli 3) all crossed with submissions made 10-18 00:00-02:30 UTC (.958 -> .959/.960). It looks like
  one shared private source (a class or group chat, GUESS). Nothing public matches that window (sec 4).

### 2.1 Is it a lottery? (MEASURED counts, MODEL arithmetic)
- Among teams with >= 10 submissions since 10-16 06:22 that are at .958 or crossed after that time: 57 crossed, 397 are
  still at .958 (12.6 %). >= 15 subs: 30 vs 179 (14.4 %).
- If it were pure tweak luck with k ~ 10 tries, 12.6 % means p ~ .013 per try, i.e. the threshold sits ~2.2 sd of the
  tweak delta away; with a ~+.0004 gap that is a tweak sd ~ .00018 (MODEL). Under that sd, a landing at .960 (+.0013)
  is a > 7 sd event, yet 22 group-C teams did it. So the .960+ landers have real gains; some of the .959 landers may be
  luck. Our own record fits the lottery arithmetic: 9 tweaks around c4, 0 crossings (MEASURED our_subs2.csv).

## 3. The predictor: an independent pipeline before the public wave (MEASURED, inline script on `work/lead_ts.parquet`)

Teams now at .958+, binned by their displayed score at 10-12 00:00 UTC (public frontier then at most ~.954: our
verbatim copy of the heliosli Raptor+S6 notebook scored .954 when submitted 10-12 02:53 UTC, the earlier public best we
copied scored .943 on 10-01; MEASURED our_subs2.csv):

| score at 10-12 00:00 | n now >= .958 | share now >= .959 | >= .960 | >= .962 |
|---|---|---|---|---|
| < .930 | 109 | .02 | .02 | .01 |
| .930-.940 | 63 | .06 | .02 | .02 |
| .940-.945 | 446 | .04 | .03 | .01 |
| .945-.950 | 194 | .16 | .07 | .03 |
| .950-.953 | 266 | .11 | .05 | .02 |
| .953-.955 | 29 | .76 | .41 | .24 |
| .955-.956 | 10 | .70 | .60 | .20 |
| .956-.957 | 15 | .93 | .80 | .40 |
| .957-.958 | 8 | .88 | .88 | .50 |
| .958-.959 | 8 | 1.00 | 1.00 | .38 |
| no score yet | 270 | .06 | | |

Same idea at 10-15 06:00 (public frontier .956/.957 then): the .956 bin (653 teams, the public .956 copy crowd) has
2.8 % at .959+ now; the .957 bin (126, where c4's ancestor S6F sat) 25 %; the .958 bin (42) 62 %.

Interpretation (MODEL): the step from the public frontier to .959+ is made by teams that bring an independent, already
strong model. Being at the public frontier with a better blend (our situation) converts at ~25 %, and we are on the
wrong side of it so far.

Our own trajectory confirms which side we are on: our own fleet solo was .932-.944 (K2-K7i), +public parent .945-.951,
and since 10-12 every best came from public members (heliosli, nartaa, R6, S6fall) re-weighted; KC1/K14/S/fleet legs
on top were flat or negative (MEASURED our_subs2.csv descriptions).

## 4. Public resources around the jump dates (MEASURED metadata, `work/lead_meta/`)

Recent `knee` datasets by downloads (10-12..10-19; `ds_knee_updated.csv`), with the owner's own LB team where it matches:

| dataset | updated (UTC) | downloads | owner team now | note |
|---|---|---|---|---|
| nartaa/rsna-knee-s6fall-weights | 10-17 06:04 (v5 = training code only) | 7,364 | Danial Zakaria .958 r389 | in c4 already |
| heliosli/rsna-knee-r6-frozen-fe055406 (+ -w1, rostigor copy) | 10-14 / 10-15 | 3,796 / 1,493 / 1,338 | Heliosli .958 r195 | in c4 already |
| stefan87/knee-abnormality-s6-render-shard-00..09 | 10-15 20:14 .. 10-16 10:02 | 428-1,829 each | **Stefan87 .959 r150** | pre-rendered S6 training rows (npz); a retrain-S6 kit |
| jessky/rsna-knee-local-cv-cnx50 | 10-12 | 1,517 | zentropy .958 r911 (206 subs) | S6 d50 5 folds, R6/S6 family |
| neronrx/mycelis-rsna-knee-4fold-hive | 10-18 17:08 | 1,036 (971 at ~11:30 UTC with 31 views) | NODUS - M.Y.C.E.L.I.S .958 r1331 (15 subs) | 4 x 210 MB ConvNeXt 384 + 4 OOF CSVs; downloads >> views = kernel attaches, not people |
| omerzalman 11-label evidence fusion / 98 % / 99 % soft labels | 10-17 | 573 / 217 / 243 | OmerZalman .958 r1075 | label tables |
| kritikabenjwal / sarthakgupta1801 / xhaitanya bleg R6 fine-tunes | 10-18 12:42-14:19 | 407 / 281 / 237 | Ghutnoon mein dimaag .958 r1185 (22 subs) | R6 family, gold58 in pixel cache (CLAIMED page) |
| dkim54/rsna-knee-models-v10/v11 | 10-14/15 | 41-495 | "This is an Agent competing" .954 r1990 | CC BY-NC 4.0 |
| cyruspinto engine8-11 | 10-18 | 5-13 | CYRUS PINTO .958 r620 | |
| riadmohamed42/rsna-knee-dino-5fold-weights | 10-16 | 38 | RIAD MOHAMED .958 r668 | |
| **bliverrigpdoge/rsna-knee-trainkit** | **10-19 11:58** | 107 | **BliverRigPåDoge .959 r181** | teacher CSVs (D1a, D1f0a), consensus_labels, labels_soft / S5 / N5 / KCS_s50 / R1s50, silver3, convnext_small_s6fall_enc.safetensors, convnext small/tiny in12k inits, judge_image.py, instrument.py (file list only) |
| haungjiawei/rsna-knee-train-v2-code-and-labels | 10-19 08:25 | 470 | haungjiawei .891 r4573 | train code + weak_labels_llm_v2 + dup_groups.csv + gold58_uids.txt |

Findings:
- **No publisher of a recent public MODEL crossed .959 with it** (B-leg team, Mycelis, cyruspinto, riadmohamed42,
  jessky, dkim54, heliosli, nartaa, goodpjw all at .958 or below). These owners had their own weights and many subs.
  That is direct LB evidence against any of these being a drop-in +.001 (MEASURED ranks; the inference is MODEL).
- The two .959 teams that publish are both "train your own on better labels" kits: Stefan87 (S6 training renders,
  crossed .957 -> .959 with a 10-17 10:00 UTC submission) and BliverRigPåDoge (trainkit with teacher / consensus /
  soft label tables and an S6fall-encoder init; crossed .958 -> .959 with a 10-17 ~23:00 UTC submission, kit published
  after). Both are training recipes, not inference add-ons.
- Download counts are not adoption: Mycelis had 971 downloads with 31 page views (session_2026-10-19/think/
  models_work/ds_basics_new.json), so most counts are kernel attaches (including the owners' own runs).

### 4.1 Per-member scan of .959+ teams (MEASURED, `work/lead_scripts/user_assets_retry.py`, `assets_report.py`,
`work/lead_meta/user_assets_top959.json`; Kaggle API dataset lists, metadata only; the API rate-limited us, so coverage
is partial)
- Coverage at writing: 226 of 332 members of the 185 .959+ teams; 164 of 183 members of the 111 crossers since 10-15
  (98 of 111 crosser teams have >= 1 member covered). Crossers were scanned first.
- **Only 5 covered .959+ teams own ANY knee-ish public dataset updated since 10-01** (plus BliverRigPåDoge, found by the
  `knee` search, not yet reached by the scan):
  - Stefan87 (r150): the S6 render shards above (Apache 2.0).
  - 66game (r115, group C, .954 -> .960 between 10-17 21:13 and 10-18 06:22 UTC): `yeshawang/model412` (10-18 03:46,
    MIT, 72 dl) is a re-upload of nartaa's S6fall (`E_prior_s42_fall.pt` + its training_code, labels_v1_blendjev),
    i.e. the same S6fall c4 already uses; `yeshawang/model231` (10-18 04:35) is one 1.6 GB `latest_epoch.pth`, their
    own checkpoint. So 66game's jump = public base + S6fall + an own model (file names only; contents not read).
  - BarryZhou (r6), kaggle & graduate trade-off (r96): Biohub (other competition) files. Nischay and Sho (r9): wheels.
- hirokenn, Leslie Yeh, the liminhai / Leka ZHANG / liyupengli cluster, Nadeem, teopinti, Donato Riccio: no public
  knee dataset at all. Their crossing inputs are private.
- So no public dataset is shared by the jumpers; the visible pattern is "S6fall + public R6/Raptor base + own model".

## 5. Forum since the 10-19 lens (MEASURED, `work/lead_forum/`)
- Only two threads moved after 11:38 UTC: 745949 (a 15:56 UTC question to colum2131) and 750529 (14:54, a .924 single
  model description by M. Ehtasham using dreaddevelopment weights). No host / staff post.
- colum2131 (now r25, .964) on 10-05 in 745949 (CLAIMED): on gold58, an LLM on reports only gives Effusion .835 /
  Synovitis .799; LLM on reports + MRI ~.91-.92 / ~.84-.86; **an image model trained on his own pseudo-labels .986 /
  .859**. He trains on soft probabilities, treats report silence as weak evidence, and ranks "explicitly present >
  equivocal > not mentioned > explicitly absent". This is the label lever, stated plainly by a .964 team.
- For scale, c4 on gold58 (MEASURED, `session_2026-10-19/think/ours_work/ours_g58.json`): macro .941, Effusion .981,
  Synovitis .840, Lateral OA .843, PF OA .901, Lateral Meniscus .928, Fracture .936. c4 already matches colum2131's
  single model on Effusion / Synovitis within gold58 noise; gold58 cannot separate .958 from .964 teams (datnt114: .961
  public with gold58 .938, CLAIMED 748011). So the leaders' advantage is not visible on gold58 and we cannot "measure our
  way" to it offline.

## 6. Can we legally use "the trick" now?
- The shared ingredient is know-how (expert-regime label construction + own k-fold CNN training), not a public file.
  Nothing to copy. Legal status is not the blocker; time is: a new label set + 4-5 fold training + a kernel + scoring
  does not fit before the last readable submit (10-22 ~10:15 IST) with our latency (~6.5 h commit + ~6.5 h score).
- Public weights that ARE legal (CC0 / Apache) and outside our families: Mycelis (CC0), riadmohamed42 DINO (CC0),
  cyruspinto engines (CC0). dkim54 is CC BY-NC 4.0 (avoid for a final pick). LB evidence from their owners is negative.

## 7. What, if anything, to do with this
- No new build is justified by the LB-dynamics evidence. It strengthens the THINK_1019 default (STOP or at most one
  label-agnostic probe) and the A = best-id / B = R6 hedge.
- Optional cheap check (needs Eesh's OK to DOWNLOAD 4 files: `neronrx/mycelis-rsna-knee-4fold-hive` oof_fold0-3.csv,
  ~181 KB each, CC0, Kaggle): put them in a new empty dir `pulled/mycelis_oof/`, read with pandas under `python -I`,
  compute per-label OOF AUC vs report labels (`_data/train.csv` + `_labels/rsna-knee-llm-labels/report_labels_v2.csv`)
  and within-class Spearman vs `session_2026-10-18/next/g58/R7OOF.csv` on shared studies, and gold58 AUC if any gold58
  study is in its OOF. ~15 min CPU. Gate to even consider a leg: OOF macro >= R7OOF's on the same studies AND
  within-class rho vs R7OOF <= .85 AND gold58 (if present) >= .925. Prior that it passes: ~.1 (GUESS), and even a pass
  could only feed a w .05-.10 leg after S6F (needs its own GPU timing: ~4 folds x 384 on T4, MODEL 0.7-1.5 h).
- If Eesh wants a GPU measurement to settle "independent own model" value instead: none fits the window with a read.
  For the record, the minimal test would be: box V100, train one ConvNeXt-T 2.5D 224 5-fold on soft labels =
  .5 x report LLM label + .5 x R7OOF/S6fall OOF probability (the colum2131 recipe shape), 5 x ~40-60 min, then OOF and
  gold58 diversity vs c4 with `session_2026-10-19/think/models_work/div_g58.py`. ~4-5 h V100. It would only inform a
  future competition, not this one.

## 8. Files
- Scripts: `work/lead_scripts/{ts_build,jumpers,classify,effort,bigjumps,events,mergers,user_assets*}.py`.
- Data: `work/lead_ts.parquet` (60 snapshots), `work/lead_top959.csv`, `work/lead_crossers_since1015.csv`,
  `work/lead_events.csv`, `work/lead_cross_events.csv`, `work/lead_bigjumps.csv`, `work/lead_classify.txt`.
- Metadata: `work/lead_meta/ds_knee_updated.csv`, `ds_rsna_updated.csv`, `user_assets_top959.json` (per-member public
  datasets of .959+ teams; partial, see 4.1; the scan was stopped at writing and can be resumed with
  `NOK=1 py -3.12 work/lead_scripts/user_assets_retry.py 0.959`).
- Forum: `work/lead_forum/{l_recent.json,msgs_745949.json,msgs_750529.json}`.
