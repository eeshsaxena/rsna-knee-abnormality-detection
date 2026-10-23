# PREREG 10-13 (written 14:45 IST, BEFORE any HR7 / HFR7 read). No em dashes.

## Facts at writing
- LB 08:39 UTC 10-13: 5,896 teams; us .954 rank 822. Bands: .957 r80-119, .956 r120-453 (~334 teams, mostly copies of
  heliosli/0-957-latest-dataset-score-in-one-click, public score .956 with PUBLIC inputs incl. the R7 bundle
  heliosli/rsna-knee-compact-r7cnxfsss), .955 r454-530, .954 r531-1127. Medal cuts ~ silver r294, bronze r589.
- Identical copies tie on public AND private (ties order by submission time, so a late copy sits at the bottom of the band).
  Anything whose private score differs from the copy crowd moves ABOVE or BELOW the whole band.

## Submissions today
1. **HR7** = the heliosli R7 notebook verbatim (`eeshsaxena/rsna-sub-hr7` v1; commit PASS, 3 rows, 0 fallback). SUBMITTED 14:19, ref 57137313. Expected .956.
2. **HFR7-R20** = HR7 + our K14 R fleet (C + DT + E + E2, 20 members) at w .20, per-label rank blend; built by `hfr7/build_hfr7.py` = `session_2026-10-12/hf/build_hf.py` on the HV-R7 base (cells == the tested HF-R20 except the R7 bundle strings + 1 markdown line). Gate: `session_2026-10-12/hf/h_check.py <out> --kind hf --wf 0.2 --members 20` must PASS before submit.

## Final-pick plan (fixed now)
- Pick A = the copy floor: HR7 (or an earlier identical-score entry if one exists).
- Pick B = the differentiated bet: HFR7-R20, the R7 base plus our independent OAI-T123 fleet. Rationale: on private, B jumps the whole copy band if our fleet adds anything, and A keeps the floor if it does not.
- If HFR7-R20 displays >= .957 on public: B confirmed. If it displays .956: B still preferred over a second copy (the 3-dp display cannot resolve it). If it displays <= .955: B = HFR7 at a lower weight (w .10), one more submission, then stop.

## CORRECTION (15:55, before any HR7 / HFR7 read)
- Our rank at 08:39 UTC was **r535** (5th of the 597-team .954 band), not r822 (that number came from a time-ordered sort).
  Kaggle ranks teams INSIDE a displayed-score band by the unrounded score (FORUM_DIGEST_10-13.md: 146 id inversions in the
  .956 band; forum 748310). So the rank inside a band reveals sub-.001 differences.
- Consequence for the reads: compare HR7 vs HFR7 by which one becomes our team's best and where it sits inside the .956 band
  (the LB shows only the team's best entry). HF-R20 already ranks above almost every H copy in the .954 band = first
  rank-level evidence that our fleet adds on top of a public blend.
- Next candidate (not submitted before the HR7 / HFR7 reads): HFR7-RR = HFR7-R20 + nartaa's r224 Raptor checkpoint at .35
  of the Raptor slot (public precedent varshithakolupuri 10-13).

## AMENDMENT 1 (16:35, before any HR7 / HFR7 read)
- HR7-RR (`eeshsaxena/rsna-sub-hr7rr` v2: R7 .5 + Raptor_mix .5, Raptor_mix = rank blend r384 .65 / nartaa r224 .35; rr_check PASS 27/27) does not depend on the HFR7 read (no fleet), so it is submitted now as slot 3: it measures the dual-Raptor effect on the copy base by band position.
- HFR7-RR (`eeshsaxena/rsna-sub-hfr7rr` v1; rr_check 31/31 + HCHECK 16/16) waits for the HFR7-R20 read: submit if HFR7-R20 ranks above HR7 (our team-best moves to HFR7) or ties; if HR7 ranks above HFR7-R20, submit nothing with the fleet and keep HR7-RR as the public-family line.

## READS 19:42 + AMENDMENT 2 (written before the HFR7-RR submit)
- HR7 = .956 (alone: team rank 307). HFR7-R20 = .956. HR7-RR = .956. LB 14:12 UTC: team rank **221**, LastSubmissionDate 10:47:26 = HR7-RR's
  submit time, so **HR7-RR is our best (unrounded) of the three**; .956 band = r141-621 (481 teams). Silver cut ~r296 -> inside silver.
- The Amendment-1 test (HFR7-R20 vs HR7) is not observable any more: the LB shows only the team-best entry, and HR7-RR beats both.
- Replacement test (cleaner): submit **HFR7-RR** (= HR7-RR + our K14 R fleet at .2; same base, only F added). If our team-best moves
  to HFR7-RR (LastSubmissionDate = its submit time and/or a better rank), our fleet ADDS on the RR base -> pick B = HFR7-RR line.
  If the team-best stays HR7-RR, the fleet does not add there -> pick B = HR7-RR line, and the fleet work stops for the LB.

## READ 01:05 + AMENDMENT 3 (01:25 10-14, before the HFR7-RS submit)
- HFR7-RR = .956. The LB "LastSubmissionDate" is the team's LATEST submission, not its best, so the 19:42 inference "team best = HR7-RR" is withdrawn.
  Method that works: count teams that did NOT change between two LB snapshots ("frozen") and are ranked above us.
  12:05 -> 14:12 UTC (HFR7-R20 + HR7-RR scored): frozen above us 245 -> 155 = our best improved a lot over HR7.
  14:12 -> 19:34 UTC (HFR7-RR scored): 133 -> 132 = HFR7-RR is our best and slightly above the previous best. Team rank 300 at 19:34
  (the .956 band grew to r155-753; new entries pushed us down; silver cut ~r297).
- **Slot 5 of the 10-13 UTC day = HFR7-RS** (`eeshsaxena/rsna-sub-hfr7rs` v1 = HFR7-RR + our 2 RSNA-only Raptor seeds s46/s45 in
  the Raptor slot, mix r384 .45 / r224 .25 / ours .30 fixed in PREREG_RAPTOR_KERNELS; RS_CHECK 38/38 + HCHECK 16/16). Read: frozen-team
  count above us vs the 19:34 snapshot. Fewer -> our Raptor seeds add on top of HFR7-RR (pick-B line); same -> they do not.
