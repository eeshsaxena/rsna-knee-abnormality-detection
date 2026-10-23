# RSNA Knee: research to top, roadmap 10-08 (synthesis of 6 verified research dimensions)

Written 2026-10-08 ~23:00 IST. Inputs: research_1008/{A,B,C,D,E,F}_*.md and their verify_*.md files, plus
PROGRESS_08-10-2026.md, HANDOFF.md, NOVEL_PLAN_10-07.md, RULES_AND_HOST_COMMENT_10-08.md. Only findings that
survived verification are used, and only with the CORRECTED gains. Nothing was written on Kaggle or on the box. The box
was unreachable from the laptop at ~22:50 (ssh timeout; the laptop is on the 10.19.x Wi-Fi).

Tags: **MEASURED** (source), **MODEL** (computed from measured inputs), **GUESS** (judgement), **CLAIMED** (forum or third
party, not re-run).

---

## 0. One-paragraph summary

N (nartaa's public OAI-trained .949 CoAtNet) is the jump. K9N-a and a new parent-free F+N kernel each model at about
**.950-.951 public** (MODEL). Everything else we can still build adds **about +.0003 to +.001 combined** (MODEL), which
is about one display tick at best. Gold (.959) is out of reach; .955 is out of reach. The realistic job from here is:
(1) put the N jump on the board on 10-10, (2) move all later work into the parent-free kernel (runtime safety, 2-3 h
reads, room for more members), (3) keep a clean OAI-free pick as ban insurance, (4) do not burn effort on levers already
measured at zero.

---

## 1. Realistic ceiling by 10-22 and the gap to .955+

### 1.1 Where we stand (MEASURED, LB CSV 16:41 UTC 10-08, A_lists/lb_now; verify_B lbcmp.py)
- Us: **.946, rank 664 of 5,513**, outside medals.
- Cut lines: gold rank 21 = .959; silver rank 275 = .950 (inside the tie); bronze rank 551 = .949. Top .964.
- .950 tie: **318 teams** (ranks 157-474), growing ~18-20 teams/h on 10-08 with no sign of slowing. .951+: 156 teams.
- No public notebook scores above .950 (MEASURED, 618 notebooks). The .950 notebooks run N + G (goodpjw2008's .929
  ConvNeXt reader) at small per-label weights (MEASURED code). That the whole 318-team pile is this recipe is a GUESS.

### 1.2 Ceiling (MODEL with ranges)

| Quantity | Value | Tag / source |
|---|---|---|
| K9N-a public display | median .950-.951; P(>= .951) .20-.43; P(<= .949) .07-.35 depending on the r_FN prior (.89-.92) | MODEL, verify_B vb_checks3 |
| Parent-free F+N public | -.0002 to +.0005 vs K9N-a, i.e. the same display | MODEL, verify_B / verify_F |
| All later verified levers combined (sec 3) | about +.0003 to +.001 on top, before the binormal model's +.0000 to +.0009 optimism | MODEL, sum of corrected gains |
| **Best realistic final public** | **.950-.952, central ~.951**; P(>= .953) <= .10 | MODEL / GUESS |
| SD(public - private), one submission | ~.004 | MODEL, B b_shake; test ~1,300 studies, 30/70 split MEASURED |
| Paired private SD vs the N band | ~.0003-.0005, so our private order against the band is set by the true difference | MODEL |
| Private P(silver+), best N pick | ~.2-.5 (falls as the >= .951 group grows) | GUESS, verify_B vb_rank |
| Private P(bronze+), best N pick, no ban | ~.5-.75 | GUESS (verify_A P(public bronze) ~.8, discounted for private) |
| Private gold | ~0 in every model | MODEL |
| Clean pick (no N), no ban | ~.004 below the band, P(medal) ~0 | MODEL |

### 1.3 What separates us from .955+ and the top
- **Single-family strength.** Our fleet F reads **.944 SOLO** (MEASURED, K7i). To reach .955 blended with N we would
  need F at ~**.9535 SOLO** at r_FN .90 (MODEL, B sec 4.3). Each +.001 of F SOLO is worth only ~+.0003 in the blend
  (MODEL, verify_E reproduced to the 4th decimal), and a one-set swap only ~.14x its set-level change (MODEL).
- **The top teams had it before N.** Gold was .958 on 10-06 (MEASURED, HANDOFF), before N was public. Other teams
  claim .949-.955 single families (CLAIMED, forum). That gap (~+.005-.010 of single-family strength) is not closable
  with any lever we have measured.
- **The only blend-side shortcut** would be a new leg that is both strong and decorrelated: SOLO >= .940 at r <= .85,
  or >= .944 at r .88, gives +.0015 or more (MODEL, verify_E vt.out). Nothing we trained or found qualifies. Own
  CoAtNet: +.0000 (too N-like, MODEL). The only unchecked candidates are two public competition-trained ConvNeXt-S knee
  weight sets (sec 3, item X1; GUESS low odds).
- **Silver is now decided above the band.** 156 teams at >= .951 (MEASURED) are N + own-model blends doing what we do.
  Weight tuning cannot clear them; only a stronger or less N-like F can.

### 1.4 Tie order inside the .950 band (conflict between verifiers)
- verify_A (MEASURED inputs, INFERRED conclusion): inside the band, 142 of 317 adjacent pairs are date-inverted, and two
  teams whose last submission predates N sit at ranks 461 and 470. So ties are most likely **ordered by the hidden
  full-precision score**, not by submission time. verify_B / F assumed earlier-submission tie-breaking.
- Use: (1) submission timing does not matter, so order slots by information value; (2) after each submission, log our
  rank inside the band and our neighbours. A rise inside the band at the same 3-dp display is a free sub-tick signal
  (INFERRED; only improvements of our best entry are visible).

---

## 2. SATURDAY 10-10: five submissions, in order

Eesh's GPU quota resets at **05:30 IST**; the wake timer fires at 05:35 (MEASURED, PROGRESS 20:35). Quota for all five
commits ~1.8 h of 30 h (MODEL: ~0.5 h per parent-carrying commit, ~0.25 h per parent-free commit). Slots are not the
constraint (~60 left to 10-22, MEASURED arithmetic); quota and scoring latency are. A Kaggle-side scoring error is
resubmittable for free (MEASURED, PROGRESS 07:00), so no slot is held in reserve.

| # | Submission | What it is | Why | Expected read (MODEL unless noted) | Read at (IST) |
|---|---|---|---|---|---|
| 1 | **K9N-a** | parent + K9 fleet + N, rank blend P/F/N .10/.30/.60 (built, 85/85 + k9n_check 80/80, MEASURED) | The medal carrier; biggest available jump | display .950-.951 (P(>= .951) .20-.43) | ~13:30-14:30 |
| 2 | **FN** (new, parent-free) | K9 SOLO cells + the K9N cell-32 N stage, P weight 0, F .40 / N .60 | Same score as K9N-a (-.0002 to +.0005) with no 6-8 h parent: removes the N-stage kill risk (GUESS P(ERROR) 10-25% for K9N if the parent runs 7 h), reads in ~2-3 h, commits cost ~10-15 min. With #1 it measures the parent's marginal value directly | same display as #1 | ~08:00-09:30 |
| 3 | **K9 SOLO** | the 10-member fleet alone | F strength, needed to read r_FN (coarse, +-.03) and to judge the clean pick | .943-.945; pre-registered audit if <= .941 (MEASURED rule) | ~07:30-08:30 |
| 4 | **K9 main** | parent + fleet, W .50, clean (no N, no OAI) | Clean pick-2 candidate (ban insurance) | .946-.948 | ~13:30-14:30 |
| 5 | **N-only** | the FN kernel at weights 0/0/1 (fleet runs, weight 0) | Anchors every N read in our exact code path; N's .949 is the author's best of >= 6 reads (winner's curse) | .948-.949 (GUESS); audit the N path if <= .947 | ~08:00-09:30 |

**Fallback if FN / N-only are not built and A0-tested by Friday night:** slot 2 = **K9-KF** (the flat-.60 clean twin,
built and A0-PASS, MEASURED) and slot 5 stays empty until FN is ready (submit it 10-11). Do NOT submit K9N-b: it models
-.0006 to -.0012 below K9N-a at every r_FN (MODEL, verify_B); the slot buys nothing.

**Pre-registered read rules (write them into PREREG_10-10.md before 05:30):**
- R1: if FN display >= K9N-a display, FN becomes the vehicle for all K10 work (parent-free).
  If FN < K9N-a by >= .002, the parent adds signal: K10 keeps the parent with at most 2 sets of <= 288 px.
  In between: tie, default to parent-free (runtime and dependency safety).
- R2: N-only <= .947 means an N-path audit (rank parity, fallbacks) before any further N submission.
- R3: K9 SOLO <= .941 means a fleet audit; >= .942 keeps K9 main as the clean candidate.
- R4: log submit and complete timestamps for #1 and #4 (real parent runtime vs the runbook's 6.0 h; >= 6.6 h means the
  slow K9N path gets killed at 8.6 h, so any later parent + N kit must shrink).
- R5: re-download the LB before submitting; after each read log our rank inside the tie band and the neighbours.

### 2.1 Must be ready by Friday 10-09 night
1. **Box reachable from the laptop** (Eesh: back on the lab network or a working path). Without it there is no FN A0
   and no queue monitoring. Deadline 10-09 18:00 IST.
2. **FN and N-only kernels built** from the K9 SOLO kernel + the K9N cell-32 N stage (P weight 0, ids from the
   fleet), same 46/46 + 7-kind negative tests as K9N, then a **box CPU A0** (~1 h). ~0.5 agent day. The K9 dataset
   version (live 11:00 10-08) must stay the latest until the K9/K9N/FN commits are done (MEASURED rule).
3. **PREREG_10-10.md**: the read table above, R1-R5, and both G320' branch layouts (sec 3.2).
4. K9N-a, K9 SOLO, K9 main kernels: built and A0-PASS already (MEASURED). k2_check / k9n_check ready.
5. Teammates told to push their T4 sessions after the 05:30 reset (sec 3.3), each on their own account (rule 3.5a).

---

## 3. Roadmap 10-09 to 10-20 (freeze 10-20 12:00 IST)

### 3.1 Value table of every surviving lever (corrected gains)

| ID | Lever | Corrected gain over K9N-a / FN | Helps which pick | Gate (pre-registered) | Decision date |
|---|---|---|---|---|---|
| L1 | FN parent-free vehicle (sat #2) | -.0002 to +.0005 (MODEL); value = runtime safety, 3x faster reads | N pick | R1 above | 10-10 |
| L2 | K10FN-wide: FN with the sets that exist (C 288 mix50, D 224 frac75, E 288 frac75 s1045 from Q4, 320 set if PASS, T4 s1044 224 set) | +.0001 to +.0005 (MODEL: ~+.0002 per extra same-family set, MEASURED k-curve analog gold74 +.0006 for 5 -> 10 members; set swap ~.14x) | N pick | composition chosen by a CPU gold58/OOF read of C+D / C+E / D+E incl. the C-E correlation, before any public read (verify_E M10); runtime slow end <= 8.1 h | 10-11 to 10-12 |
| L3 | Q4 = 288 frac75 s1045 (queued, FAIL default) | +.0000 to +.0003, central +.00015 as a set swap (MODEL) | both | per-fold bars at the 21-epoch level (not set C's 24) | auto, lands ~10-10 07:00 |
| L4 | G320' PASS branch: 320 frac75 folds 1-4 on the box only | ~+.0001 if PASS (MODEL); P(PASS) .20-.35 (MODEL from ep10 .8833) | both | G320' (best raw >= .8909 OR last-3 >= .8900) AND gold58 >= .905, auto at ~01:30 10-09 | 10-09 01:30 |
| L5 | All-24-slot 288 training, seed-matched pair with Q4 f0 (s1045, frac75, 21 ep, only --train_slices 0 differs) | -.0001 to +.0004 (MODEL on GUESS) | both | last-3 raw >= Q4 f0 last-3 + .002 AND gold58 not lower by > .003; FAIL closes slot-count work | probe read ~10-10 15:00 (FAIL branch) |
| L6 | Own OAI members, T1 labels only first (PF OA, Lateral OA, Synovitis), 224, finite norm, 24 ep gc1 paired with f0_s1043_ttr_frac75 (~5.8 h probe, ~23 h folds 1-4) | +.0001 to +.0004 central, range -.0005 to +.0014 (MODEL, D author's own model) | **N pick only** | G1 OOF vs v4 on non-OAI labels >= paired - .002; G2 OOF vs v4 mean on T1 >= paired - .001 AND gold58 T1 mean >= paired + .005 (low power; OOF leg carries the weight); G3 gold58 macro >= paired - .005. T2 labels only as a separate second gated step | go/no-go 10-11 |
| L7 | VLM relabel scale-up (only if Eesh runs the pilot with his own key and S >= .03 AND img-text >= .02) | +.0001 to +.001 blended (MODEL); must REPLACE set D, not add | both | prereg_v2.json (MEASURED sha 6ce3b3c5) then a paired student gate vs D | pilot by 10-10, scale-up decision ~10-11 |
| L8 | Per-label N weight map from priors only (raise N on PF OA / Lateral OA / Synovitis, lower on Baker's / Contusion), one submission, never fitted on gold or public | GUESS 0 to +.001; not adversarially verified (surfaced only in verify_B "missed") | N pick | map written down before the read; one submission only; adopt only if >= +.001 display | 10-12 |
| X1 | Public competition-trained ConvNeXt-S knee weights (kirilllabzin/knee-convnext-small-v17-source, ngochienphan/rsna-convnext-small-384-ep20-raw) as a decorrelated 4th leg | 0 unless it passes; +.0006 to +.0015 if SOLO >= .940 and r <= .85-.88 (MODEL) | N pick (FN-wide) | read-only check first: public standalone score >= .940, OSI-compatible licence, competition-only provenance stated; download needs Eesh's OK | check 10-09, decide 10-11 |

### 3.2 V100 (one job at a time; box-only for 320 and OAI)

Branch is set automatically at **~01:30 10-09** by the G320' verdict (MEASURED gate armed, PROGRESS 20:35).

| When (IST) | FAIL branch (P .65-.80, MODEL) | PASS branch (P .20-.35) |
|---|---|---|
| 10-09 01:30 -> | Q4: 288 frac75 s1045 folds 0-4 (~29 h) -> lands ~10-10 07:00 | Q3b: 320 frac75 folds 1-4 (~28 h at 19.9 min/epoch MEASURED) -> ~10-10 06:00 |
| 10-10 | ~07:00: L5 24-slot f0 (8.1 h) -> read ~15:00. Then L6 OAI probe (5.8 h) if Eesh gave go and the OAI store is built, else L5 folds 1-4 if L5 passed | Q4 (~29 h) -> lands ~10-11 11:00 |
| 10-11 | L5 folds 1-4 (~32 h) if PASS, else OAI folds 1-4 (~23 h) if L6 passed | ~11:00: L5 24-slot f0 (8.1 h), then L6 probe |
| 10-12 to 10-15 | the remaining one of L5 / L6 folds; then filler = extra 288 frac75 seeds for FN-wide (~+.0001-.0002 each set, MODEL) | L5 / L6 folds as gated |
| 10-16 to 10-18 | filler only; **last fold must start by ~10-17 18:00** so it lands before 10-18 night | same |
| 10-19 | idle for kit A0 support if needed; no new training | same |

Rules: if Eesh's OAI go arrives and the host also explicitly OKs OAI, L6 moves ahead of L5 (it then helps both picks,
roughly doubling its value, verify_D). OAI data stays in ~/oai (mode 700), OAI ids are prefixed `oai_` and asserted out
of the parity npz (MEASURED: probe_ids = sorted(ids)[:4] ships to Kaggle). After any power cut: restart the runner,
move partial runs aside, re-queue lines showing #RUN without DONE (MEASURED lesson, 10-08).

PASS-branch layout (pre-register now, verify_E M6): with the parent, only D+320 fits (8.08 h slow) and C+320 does not
(8.22 h). Parent-free, all of C + D + 320 fit. So on PASS the 320 set goes into **K10FN-wide**, not a parent kernel.

### 3.3 Kaggle T4 (~85 h on 3 accounts from 10-10 05:30; each session pushed by its own account owner)
OAI must never touch T4 (DUC). Model-side T4 work is worth only ~+.0001 (verify_E M7), so T4 is for label-side sets and
for cheap extra members of the parent-free fleet.

| Account | 10-10 | 10-11 to 10-14 | 10-15 to 10-19 |
|---|---|---|---|
| **Eesh** | Saturday commits (~1.8 h). Keep >= 8 h of this week's quota for commits through 10-16 | commits for K10FN-wide, KF, L8 (~1-1.5 h/day) | new week 10-17: final-kit commits only (~3-5 h) |
| **swapiy** | S1: 224 frac75 **s1044** folds 1+2 (same argv as set D; f0_s1044 already exists, MEASURED) | S3: VLM-relabel 224 folds (only if L7 approved) | spare |
| **kmrsuman** | S2: 224 frac75 s1044 folds 3+4 | S4: VLM-relabel 224 folds (only if L7 approved) | spare |

Gate for S1/S2 members: per-fold OOF >= set D same fold - .004, min epochs 14 (MEASURED K9 rule). Value: a third
224 set for FN-wide, +.0001 to +.0002 (MODEL). If L7 is not approved by 10-11, the remaining teammate quota stays
unused: no measured lever is worth it (K11 is parked, mixN is rejected, reseeds into a parent kit do not fit).

### 3.4 Vast (~$9)
Keep as the **power-cut reserve only** (rerun one lost fold). Do not spend on 320: even on PASS its value is
~+.00003 expected (MODEL, verify_E). Never put OAI on Vast. This contradicts Eesh's 20:50 "use the Vast credit", so it
needs his confirmation (sec 4).

### 3.5 CPU work (laptop light Python; box CPU nice 19)

| Date | Item | Output |
|---|---|---|
| 10-09 | FN + N-only kernels, tests, box A0 | sat #2 / #5 ready |
| 10-09 | PREREG_10-10.md (read rules, G320' branch layouts) | before 01:30 for the PASS layout, before 05:30 10-10 for the rest |
| 10-09 | OAI trainer patch in a NEW dir (defaults bit-for-bit unchanged; --mask_norm finite default; oai_ assert), default-path OOF byte-parity test, 20-row CPU dry run | ready for L6 |
| 10-09 | X1 read-only check of the two public ConvNeXt-S knee weight sets | go/no-go input |
| 10-10 | parse OAI step-A tables: value counts of the MOAKS fields, freeze the T1 map in a hashed CSV (Contusion, MCL, Fracture masked), write links_test5.txt / links_needed.txt | needs Eesh's package ID |
| 10-10 | fit r_FN from N-only, K9 SOLO, FN, K9N-a reads; recompute K10 weights | weights for K10FN-wide |
| 10-10 to 10-11 | gold58/OOF read of C+D / C+E / D+E (+ s1044 set), incl. the C-E correlation | K10FN-wide composition |
| 10-11 to 10-18 | OAI store build on the box CPU (if go), gate scoring of L5 / L6 probes, kit builds + A0 per K10 variant | |
| 10-19 | final kits: best N kit (parent-free) + clean twin; A0; commits | |

### 3.6 Submission calendar after Saturday (MODEL plan; each read changes one thing)

| Date | Submissions |
|---|---|
| 10-11 | FN (if it slipped), K9-KF (clean twin), N-only (if it slipped) |
| 10-11 / 10-12 | K10FN-wide v1 (existing sets + E or 320), L8 prior map (one sub) |
| 10-13 to 10-15 | K10FN-wide + s1044 224 set; + L5 24-slot set if PASS |
| 10-15 to 10-18 | + OAI set (N-pick kit only) if L6 PASS; clean K10 twin (parent + best 2 sets <= 288 px, W .50) |
| 10-19 to 10-20 | final kit reads; re-reads of the two final candidates if needed |
| 10-21 to 10-22 | no new builds; picks |

---

## 4. Eesh's decisions (with deadlines)

| # | Decision / action | Deadline | Recommendation |
|---|---|---|---|
| E1 | Restore laptop -> box connectivity | **10-09 18:00 IST** | Required for the FN A0 and all monitoring |
| E2 | Approve the Saturday list (K9N-a, FN, K9 SOLO, K9 main, N-only; drop K9N-b; KF deferred to 10-11) | 10-09 22:00 | Approve |
| E3 | Vast: keep the $9 as power-cut reserve, no 320 on Vast | 10-09 morning (matters only on G320' PASS) | Approve (EV of 320 on Vast ~+.00003, MODEL) |
| E4 | OAI step A (metadata only), then go/no-go on V100 time for L6 | step A by 10-10; GPU go/no-go **10-11** | Do step A now (sizing is free); GPU only after Q4 / if idle / if the host says yes |
| E5 | VLM pilot with his own API key (~$3.79 dry-run, cap $8); scale-up only with a NEW spend OK ($15-51) | pilot by 10-10; scale-up by 10-11 | Optional; low value (+.0001 to +.001) |
| E6 | Download OK for a public ConvNeXt-S knee weight set (X1), only if its read-only check passes | 10-11 | Decide after the check |
| E7 | Confirm the team roster: the LB team row lists a 4th member, akkigupta2005, while HANDOFF says 3 (verify_F) | 10-10 | If real and willing, their own T4 quota, run by them |
| E8 | State q = P(OAI-based submissions voided) | **10-20** | Default pick 2 = clean unless the host OKs OAI weights |
| E9 | Optional: upvote / add the decisive NDA evidence under Taeyang's 743416 question (host answer would double L6's value) | any time before 10-15 | Eesh's call; draft in RULES_AND_HOST_COMMENT_10-08.md |

**OAI download steps (dimension D, MEASURED from nda-tools source and NDA dictionary API; DAR 26647 approved 10-08):**
1. NDA profile page: **UPDATE PASSWORD** (nda-tools needs an NDA password, not the Login.gov one). Note the NDA username.
2. Rebuild the cart at https://nda.nih.gov from the OAI collection https://nda.nih.gov/edit_collection.html?id=2343
   with `image03`, `oai_kmrisemiquant01`, `oai_kxrsemiquant01`, `oai_mrimeta01`, `oai_inventory01`. Confirm the
   4,796 subjects now show as available (log out / in first).
3. **Create Data Package** (e.g. `oai_rsna_knee`) and tick **"include associated files"**. Send Claude the package ID.
4. In an interactive terminal on the box:
```bash
ssh training@192.168.10.30
source ~/ndaenv/bin/activate
downloadcmd -h | head -60          # confirm 0.7.0 flags
pip install keyrings.alt           # only if keyring complains (headless Linux)
mkdir -p ~/oai/pkg ~/oai/raw && chmod 700 ~/oai
downloadcmd -dp <PKG_ID> -u <nda_username> -d ~/oai/pkg --file-regex '.*\.txt$' -wt 4   # STEP A, tables only
```
5. Claude (box CPU) writes `~/oai/links_test5.txt` and `~/oai/links_needed.txt` (SAG IW TSE FS, COR MPR, AX MPR of the
   baseline MOAKS knees). With Eesh's OK:
```bash
downloadcmd -dp <PKG_ID> -u <nda_username> -t ~/oai/links_test5.txt -d ~/oai/raw -wt 8      # size and speed test
nohup downloadcmd -dp <PKG_ID> -u <nda_username> -t ~/oai/links_needed.txt -d ~/oai/raw -wt 10 > ~/oai/dl.log 2>&1 < /dev/null &
downloadcmd -dp <PKG_ID> -u <nda_username> -t ~/oai/links_needed.txt -d ~/oai/raw --verify
```
6. After the store is built: delete the keyring password
   (`python -c "import keyring; keyring.delete_password('nda-tools','<nda_username>')"`).
Size is unknown: ~150 GB if per-series, 0.5-1.4 TB if whole-visit zips (GUESS); the 5-knee test measures it.

---

## 5. Final-pick strategy for 10-22

- **Pick 1 = the best N blend** by public read. Tie rule (within .001): prefer the parent-free kernel, then more
  non-N weight, then higher rank inside the tie band (sec 1.4). OAI-own members (L6) may only enter this pick.
- **Pick 2 = the clean pick by default**: K9 main, K9-KF or a clean K10 twin, whichever reads highest; within .001
  prefer KF (no public-LB-tuned routing map, MEASURED parent audit). It is ban insurance only: on private it lands
  ~.004 below the band, P(medal | no ban) ~0 (MODEL).
- **Switch pick 2 to a second N blend** (different composition, e.g. FN-wide vs K9N-a) only if the host explicitly OKs
  OAI-trained weights before 10-22. Reason: a second similar N blend adds only ~+.02-.06 P(silver) (GUESS, paired
  private SD ~.0003), while a clean pick is worth q x .5-.9; break-even q is ~.03-.10 (MODEL / GUESS, verify_B), and
  the host has been silent since 09-21 with the geo-access and OAI-weights questions open (MEASURED).
- **Shake-up:** SD(public - private) ~.004 per submission, so public ranks inside the 300+ team tie mean little.
  Private outcome = whether our true AUC is above the N+G clone block (paired SD .0003-.0005, MODEL); P(true > band)
  for K9N-a ~.45-.83 (MODEL with model error). Kaggle fork piles shake hardest; near-line teams keep or gain a medal
  ~24% of the time (CLAIMED 742327). A scored run already holds its private score, so runtime risk affects only errored
  submissions, not picks.
- Re-read threads 741819 and 743416 and re-download the LB on 10-20 and 10-22 before picking.

---

## 6. Do not do (measured or verified traps)

- **No public N+G clones or variants, no per-label weights copied from Apex or other forks** (fitted to the public LB;
  Apex comments list unsourced AUCs, MEASURED). Flat G .30 on N read .948 twice; gold-routed G read .948 (MEASURED).
- **No per-label routing or blend weights tuned on gold58** (N used gold58 for selection, MEASURED). Judge on gold58
  only, never on the 16 extra gold74 studies (report-read, inside N's training data).
- **No K9N-b** (-.0006 to -.0012, MODEL). Never add G to K9N; G elsewhere only at <= .10 and only if free.
- **Skip N2 (r224), renta0426 DINOv2-B, and hengck23 MaViT-288** (N2 measured 0 ticks; hengck23 named KJD-25K and its
  data are unstated).
- **KJD-25K = KneeCoT = banned** (host 734109, MEASURED). gmhost/knee-corpus-crop-320 (22.6 GB, CC0, no description):
  do not touch. MRNet, fastMRI, KneeMRI: unruled, do not use.
- **No more label-column surgery** beyond frac75: +.01 of per-label target AUC is worth ~+.00005 SOLO (MEASURED
  transfer slope, verify_C corrected units). K11 parked. Effusion as a Synovitis proxy hurts (-.010 to -.081 gold58,
  MEASURED). Synovitis, Lateral OA, Effusion, Contusion, further Fracture re-mixes: closed.
- **No mixN (N as teacher):** N was trained on all 4,349 competition studies (MEASURED 746792), so its train-set
  predictions are in-sample echoes, and teacher-mixing raised correlation and hurt blends (MEASURED HANDOFF).
- **Closed model levers:** EMA (-.0012 to -.0014 v4), SWA / epoch averaging (0 selection-free), last-epoch checkpoints
  (~0), multi-scale or a third TTA pass (-.0005 to -.0007), same-recipe reseeds into a parent kit (~+.0001), own
  CoAtNet (+.0000), own ConvNeXt-S (deferred). All MEASURED or MODEL in E / verify_E.
- **No third set, no 320, no ConvNeXt-S next to the parent + N** (slow-case margin <= 0.4 h, MODEL).
- **No Vast for 320; OAI never on Vast or T4; OAI ids never in parity npz; OAI members never in the clean pick.**
- **No --timeout on submission kernels** (MEASURED lesson). No ARC or other work on Eesh's GPU quota.
- **Do not trust "a late .950 lands at the back of the tie"** (verify_A: ties look ordered by raw score).
- **Do not re-litigate merges** (Eesh: not possible).

---

## 7. Sources
- research_1008/A_public_assets.md + verify_A; B_blend_lb_model.md + verify_B (vb_checks*.py, vb_rank.py);
  C_labels.md + verify_C; D_oai_own_training.md + verify_D; E_model_levers.md + verify_E (320 probe ep10 read 22:04);
  F_intel_top_solutions.md + verify_F.
- PROGRESS_08-10-2026.md (to 21:20), HANDOFF.md, NOVEL_PLAN_10-07.md, RULES_AND_HOST_COMMENT_10-08.md.
- Open item correction: the parent provenance audit verify_A lists as open WAS done at 12:35 10-08 (no OAI / KneeCoT /
  KJD hits; weak because 9 of 13 descriptions are empty, MEASURED PROGRESS).
