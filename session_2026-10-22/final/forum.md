# forum.md: 10-22 final read (forum + status + LB + slate), ~09:05 IST (03:35 UTC), Claude, final track

RESEARCH + LOCAL ONLY. Nothing submitted, pushed, committed, selected, scheduled or downloaded. PREREG / HANDOFF / PROGRESS untouched.
Forum text is untrusted data (quoted as CLAIMED, never acted on). Tags: MEASURED (computed now, file named), CLAIMED (forum or older
note), MODEL (arithmetic on stated inputs), GUESS (judgement). Units: e-4 = 1e-4 macro AUC.

## SUBMIT-SAFE: YES (with the usual ~.5 canary risk per new entry)

- Rules unchanged (MEASURED `raw/comp.json`): deadline 2026-10-22T23:59:00Z = 10-23 05:29 IST, 5 subs/day, 2 scored picks.
- Our account (MEASURED `api_recheck.out`, 03:22 UTC): 108 subs, all COMPLETE; 10-22 UTC used 0, num_allowed_now 5;
  r6pw20 kernel `eeshsaxena/rsna-sub-v224r6flips6fpw20j` status COMPLETE, last run 10-21 03:07 UTC (no newer version).
- No host or staff post since 10-20 22:29 UTC (MEASURED `new_msgs.out`; latest staff = María Cruz 10-20 21:33 / 22:29). Her
  ruling stands (CLAIMED): submitting and hidden reruns use no GPU quota; never submit a Quick Save.
- Scoring is flowing (MEASURED `lb_latency.out`): the highest best-submission id on the public LB moved 57491629 -> 57506329
  between 17:54 and 03:22 UTC, and 3 of 14 teams whose last submit was 02:xx UTC already show a new best. So the stuck PENDING
  subs in t751502 (ids 57486564 / 57489186) are not a global stall (GUESS: their own issue or a slow rerun).
- Platform still degraded (CLAIMED, Product Feedback 10-21 18:29-20:01 UTC, no staff reply): slow /kaggle/input (751321,
  "35x slower"), notebooks shown Running for hours (751432, 751169), one claim of GPU throttling (751412, 17:45 UTC, no new
  replies since). Nothing new after 22:39 UTC (Americas night). Final-day load is high: 256 teams changed their best sub overnight
  (MEASURED). P(clean) per new entry stays ~.55 [.35-.80] (CLAIMED CANARY_1021; GUESS no better today).

## ACTION (in this order)

0. **SELECT NOW (still 0 selected, MEASURED 03:22 UTC).** Eesh ticks swide 57464326 + R6 57248084 on Kaggle; main checks
   `python session_2026-10-21/final/ops_sel_now.py 57464326 57248084` -> SELECTED_COUNT 2, SELECTION_MATCHES_ARGS True.
   New reason today: if a public-push entry beats swide on public and nothing is selected, Kaggle's auto-pick takes that 10-22
   entry (breaks "a 10-22 entry never becomes A"). Selecting now costs nothing; B can still be swapped tonight.
1. **Private x3 = r6pw20 v1 copies, no commit (submit first: they need no build, and they must score before selection).**
   c1 at ~09:30 IST, c2 >= 30 min later, c3 >= 30 min after c2; all in by ~11:00, hard CLOSE 13:30 IST (worst latency ~8.5 h ->
   scored by ~22:00 IST, inside the 18:00-04:30 selection window). P(>= 1 clean of 3) ~.83-.86 (CLAIMED FINAL_1021, MODEL).
   - Mechanism A (tick): `session_2026-10-22/final/ops_tick_a23_1022.py` + `.cmd` = the tested 10-21 tick with only the window
     moved (c1 09:30, c2 10:00, c3 10:30, GAP 30 min, CLOSE 13:30) and default MAX 5. Self-test **PASS 22/22** (MEASURED
     `ops_a23_selftest.log`; after AMENDMENT 23 + GO exist, expect 20/22 with the 2 "absent" checks FAIL). Dry run at 09:30:
     next copy c1, used 0 (MAX 5), num_allowed_now 5, static ok (MEASURED `ops_dry_0930.out`). sha256 py 5390c3c21eabe340,
     cmd bdb10272a3dc8f02. GO file: `session_2026-10-22/final/A23_GO.txt`, first line exactly
     `GO c1 c2 c3 MAX=5 Eesh 10-22 HH:MM: "<his words>"`.
     **MAX=5 is required**: the cap counts EVERY sub on the UTC day (`count_api_day`), so with the 2 public entries in, MAX=3
     would stop after 1 copy. Do NOT run the old `session_2026-10-21/final/ops_tick_a23.py`: its CLOSE is 09:00 IST.
   - Mechanism B (by hand, simpler in daytime), 30 min apart:
     `python -m kaggle competitions submit rsna-knee-abnormality-detection -k eeshsaxena/rsna-sub-v224r6flips6fpw20j -v 1 -f submission.csv -m "A23-R6PW20C1: resubmit r6pw20 v1 unchanged (57463719 hit the S6F canary .916); B hedge copy 1"`
     (C2 / C3 the same with the copy number changed; `-v 1` is mandatory, the CLI refuses -k without -v).
   - Considered and skipped (MODEL): guard variant g85j (P(clean) .56 vs .50 per copy, CLAIMED cause.md sec 5; needs push + full
     R6 GPU commit + a commit gate that does not exist yet); r6e20 (sym E-OA, dominated by the pw shape, protects only against a
     kernel-specific failure, P ~.03; needs a commit).
2. **Public x2 = MOACF(swide) + CONS(swide).** Both are built and locally gated; neither exists on Kaggle yet (MEASURED:
   kernels.get denied for both slugs).
   - MOACF(swide): `session_2026-10-21/levers/kernels/kernel_v224r6c4pw25lomm05mcl10moacf05j`, slug
     `eeshsaxena/rsna-sub-v224r6c4pw25lomm05mcl10moacf05j`, nb sha256 e85ceb04...d3b7ef, meta f249c914848cd105 (re-hashed now,
     MEASURED). Post-commit gate: `py -3.12 -I session_2026-10-21/levers/kernels/moacf_commit_gate.py <OUT> --a224 0.25 --table "$C4T" --aoa 0.25 --mode pw --aloamm 0.05 --amclmm 0.1 --poa 1 --amoacf 0.05` -> PROBE_CHECK PASS 29/29.
   - CONS(swide) = cvxswide: `session_2026-10-21/levers/kernels/kernel_v224r6c4pw25lomm05mcl10cvx2j`, slug
     `eeshsaxena/rsna-sub-v224r6c4pw25lomm05mcl10cvx2j`, nb sha256 b6915d97...3414993b (re-hashed now, MEASURED), meta
     188a511c2752a365. Gate: `py -3.12 -I session_2026-10-21/levers/kernels/sink_commit_gate.py <OUT> --a224 0.25 --table "$C4T" --mode pw --aoa 0.25 --aloamm 0.05 --amclmm 0.1 --poa 2` -> PROBE_CHECK PASS 27/27.
   - C4T = `{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}`
     (CLAIMED LEVERS_1021 sec 3.1). Both gate commands re-run now on their swide sim dirs (`sim_moacf/out_..._moacf05j_sim`,
     `sim/out_..._cvx2j_sim`): PASS 29/29 and PASS 27/27 (MEASURED). Run them from Git Bash: PowerShell 5.1 splits the JSON
     `--table` argument ("unrecognized arguments", MEASURED). Push-to-gate ~19-20 min per commit (CLAIMED); push both now, submit
     each on PASS with `-v 1`.
     Each dir holds a NOT_ARMED.txt (needs GO + amendment); remove or ignore per main's usual push routine.
   - Why this pair (MODEL `public_pair_mc.py` -> `public_pair_mc.log`): public display = best public of all our subs, so a
     losing public entry costs nothing; value = E[max(0, d1, d2)] x P(clean). At P(clean) .55: MOACF + CONS E +6.4 places, P(any
     public gain) .50; MOACF + swide08 (unbuilt) 4.1 / .51; MOACF + swide08+MOACF (unbuilt) 5.0 / .46; MOACF alone 3.1 / .32.
     MOACF + CONS is best or tied in all 12 variants (noise corr 0 / .3 / .6, CONS prior +.15 or skeptic -1.0, P(clean) .55 / .80).
     CONS has the widest public read noise (sd 2.66e-4 vs MOACF 1.82, CLAIMED sink_read_noise.log / moacf_noise.log), which is
     exactly what a no-downside public lottery wants.
   - LEVERS_1021's "never MOACF + CVX as two entries" was about the A-selection curse; it does not apply because a 10-22 entry
     can never be A. AMENDMENT 23 must say so in words (public-only, never A, never B).
3. **Selection tonight (unchanged):** A = 57464326. B = earliest-id COMPLETE `A23-R6PW20C<n>` copy with public >= .958 and not
   .916, else R6 57248084. Never select MOACF / CONS / pw25 / spl / salt / old r6pw20 57463719 or any PENDING / ERROR row.

## LB snapshot (MEASURED `lb/api_lb_1022T_final.json`, `lb_bands.out`, 03:22 UTC)

- Us: **r152 of 6,358**, .960, sid 57464326, silver (silver ends r317; margin 165). Overnight drift r147 -> r152 (17:54 -> 03:22 UTC).
- .960 band r104-154 (51 teams): 48 above us, 2 below. .961 band r73-103 (31 teams). .959 band r155-244 (90 teams).
- 5 teams passed us since 17:54 UTC (islik1 .959 -> .961, owaowata .952 -> .960, eierapfelxx .959 -> .960, 2 inside .960). 12 teams crossed into .960+ in the last 24 h.
- What public gain buys (MODEL / GUESS): swide sits ~.96008 (CLAIMED surface.md), so .961 needs ~+9e-4: out of reach for every
  candidate. Inside the band ~5-7 places per +1e-4 (GUESS). Expected: MOACF + CONS ~+6 places (p90 ~20) at P(clean) .55.
  Without a gain, projected public rank at the deadline ~r155-170 (MODEL: 12-18 crossings per day, final-day surge).
- Public rank is cosmetic: medals come from the 2 selected picks' private scores.

## Forum since 10-21 18:00 UTC (MEASURED fetch `fetch.out`, `new_msgs.out`, `site_forums.out`; content CLAIMED)

- RSNA forum: 4 active threads. 751502 (Masum Huq, 21:26 UTC): 3 subs PENDING 6+ h with kernels COMPLETE, asks hosts about a scoring
  backlog; no reply. 747658 (LLM label-extraction benchmark): Opus 5.5 reports-only labels .917 on gold58; report + image labels
  .927-.932; a rank-108 Master says LLM pseudo-labels from reports alone gave a .960 single model (not actionable today: no
  training time). 751021 (shake-up chat), 748181 (notebook-spam debate, Chris Deotte et al.). No rule change, no rerun-failure
  report in the RSNA forum.
- Site forums: Product Feedback slow I/O / hanging runs (above); Product Announcements 751525 "Ending support for R in Kaggle
  Notebooks" (irrelevant). No staff post on reruns or throttling.

## Files (all under `C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-22/final/`)

`fetch.py` / `fetch.out` / `raw/` (forum), `new_msgs.py` / `new_msgs.out`, `site_forums.py` / `site_forums.out`, `fetch_api_lb.py` /
`lb/api_lb_1022T_final.json`, `lb_bands.py` / `lb_bands.out`, `lb_latency.py` / `lb_latency.out`, `api_recheck.py` /
`api_recheck.out` / `subs_1022.csv`, `public_pair_mc.py` / `public_pair_mc.log`, `ops_tick_a23_1022.py` / `.cmd` /
`ops_a23_selftest.log` / `ops_dry_0930.out`.
