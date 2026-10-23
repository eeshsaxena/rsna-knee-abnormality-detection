# synovitis.md: Track 2, Synovitis deep dive (10-20 night, ~20:30-21:40 IST, Claude)

RESEARCH + LOCAL BUILD ONLY. Nothing was submitted, pushed or committed. No dataset, no download, no scheduled task. PREREG,
HANDOFF, PROGRESS, session_2026-10-19/a20, session_2026-10-20/a21 and session_2026-10-20/val were only read (checked: no file in
a21/ or val/ is newer than my first script). Everything I wrote is under `session_2026-10-20/night/` with a `syn_` prefix, plus
this file. No GPU job anywhere (the box V100 was at 51 % / 15.9 GB with someone else's job; my box runs were CPU, nice 5).

**OAI governance.** Two read-only CPU scripts were piped over ssh stdin into `python -I -`:
`box_syn_inspect.py` (file existence, column NAMES, MOAKS grade histograms) and `box_syn.py` (a 20-boot smoke, then the
500-boot run). They printed aggregates only: n, prevalence, a 4 x 4 grade cross-tab of counts, AUC, AUC deltas, sd, P, Spearman,
tetrachoric. Nothing was written on the box. No OAI id, row or per-knee value left the box. The only RSNA numbers sent to the box
were the four gold58 (Eff, Syn) cell counts, hard-coded in the script.

**Tags.** MEASURED = computed tonight, log named (all in `night/`). CLAIMED = an older note or a forum post says so (named).
MODEL = arithmetic on stated inputs. GUESS = judgement. "label" = single-label AUC delta; "macro" = label / 12; e-4 = 1e-4 macro.
gold58 is used as a description and a consistency check, never to pick a weight (one exception is flagged: the dose, sec 8).
"c4 Syn" below is also pw30's Synovitis column: the A21 E-OA block touches only the OA trio (MEASURED, a21/build_a21.py), and
c4's Synovitis = .64 Raptor + .16 R6 + .20 S6fall in nested rank form (MEASURED, push/cbase.py weights).

## 1. Verdict (plain language)

1. **What the experts call Synovitis.** On gold58 it is an effusion-linked, degenerative-knee finding. 22 of 27 expert positives
   have an expert effusion, and synovitis rises with the knee's overall disease burden (1 of 9 knees with 0-1 other findings,
   9 of 12 with 6 or more). Acute traumatic effusions are mostly NOT called synovitis: 9 of the 13 effusion-but-no-synovitis
   knees have an ACL tear. It is neither a plain "small effusion" label nor MOAKS Hoffa-synovitis (sec 2).
2. **What the reports and our members learned.** Report labels say Synovitis only when the word is written, and the word's use
   depends on the site's language: the LLM label's YES rate runs from .005 (Bulgarian) and .05 (Croatian, Turkish) to .18-.19
   (English, German), while effusion is reported everywhere (sec 3). Every image member already beats the report text against the experts (.78-.86 vs .68-.79). Each member's own
   Synovitis head beats its own Effusion head (sec 4).
3. **New mechanism, measured outside gold58.** On 4,333 out-of-fold train studies, report-trained Synovitis heads learned a site
   prior: among reports that say "effusion", the Synovitis score follows the language's habit of writing "synovitis" (Spearman
   over 9 languages .88 R7, .68 / .62 our S6 seeds). The Effusion head does not carry it (sec 6). Expert labels do not share
   that prior, so it is noise on the hidden set. Borrowing Effusion into Synovitis (the killed E2) dilutes it.
4. **gold58 fingerprint, pre-specified split, 6 of 6 models.** In low-report sites (bg / hr / tr / nl, chosen from the train
   rates before looking), expert synovitis is NOT rarer (8 of 15 vs 19 of 43), yet every model's Synovitis head ranks those
   positives below its own Effusion head, and the reverse in the other sites. E2's c4 gain at a .2 comes almost wholly from
   those 8 positives (+.0078 of +.0084), and it about equals an oracle site-centring of the score (+.0078) (sec 6).
5. **No hidden-label ruler exists.** OAI is one protocol (no site prior), its synovitis grades match neither expert pattern,
   and the OAI-free heads collapse Synovitis onto Effusion there. E2 on OAI is negative or zero unweighted and positive only
   when the OAI sample is reweighted to the gold58 coupling on the Hoffa >= 1 target (sec 5). gold58 cannot certify that
   Effusion is the right helper: at a .2, borrowing Fracture or Contusion also "gains" (sec 7).
6. **E2 moves from KILL to MAYBE: a lottery ticket with a mechanism, not a reasoned gain.** Honest odds for one probe,
   **E2s = pw30 + Synovitis borrows Effusion at a .2** (MODEL, sec 8): P(true > 0) about .54 (.43-.65 over my priors),
   E[true private] about +0.4e-4 (-0.3 to +1.1e-4), P(best-id move) about .54, E[private gain of probe + R1] about +1.2e-4
   (+0.8 to +1.6e-4, mostly R1's option value on a noisy label), P(it becomes A and is truly harmful) about .11.
7. **Built and gated locally, unarmed:** `night/syn_kernel_v224r6c4pw30syn20j` (notebook sha 1c6f65a4), offline gold58 gate
   PASS, commit-gate copy `syn_probe_check.py` self-test PASS 5/5 on simulated commit outputs (sec 8.2).
8. **Recommendation.** If Eesh wants to use one of the 3 free 10-21 slots, E2s is the only Synovitis entry worth it, as the
   THIRD entry after r6pw20 and pw25 (VAL_1020), never instead of them. It does not touch B and does not change the picks rule.
   Default if he says nothing: leave it unbuilt on Kaggle. It needs his go, a PREREG amendment and the commit gate.

## 2. How the experts likely grade Synovitis

**Rubric (CLAIMED, host overview thread 733343; flat_733343.txt l.52-74).** "Synovitis: Inflammation and thickening of the
synovial lining of the joint." Two subspecialty MSK readers plus an adjudicator; borderline findings graded negative. Effusion is
"a moderate or large amount of fluid distending the joint". The host also says image-review labels can disagree with the report
wording (CLAIMED 733491). The direct question "was synovitis scored through effusion-synovitis or Hoffa surrogates?" was never
answered (CLAIMED 737155).

**gold58 structure (MEASURED `syn_local.log` part 1; n 58).**

| quantity | value |
|---|---|
| prevalence: Synovitis / Effusion / Baker's | .466 (27) / .603 (35) / .207 (12) |
| Effusion x Synovitis counts (Eff 0: Syn 0 / 1; Eff 1: Syn 0 / 1) | 18 / 5; 13 / 22 |
| P(Syn given Eff) / P(Syn given no Eff) / P(Eff given Syn) | .629 / .217 / .815 |
| tetrachoric Syn-Eff | .61 (gold74, 15 extra = our reads: .64) |
| Syn x any OA / Baker's / Medial Meniscus / acute (ACL or Cont or Frac) | tet .56 / .36 / .31 / -.14 |
| Syn by number of OTHER expert findings: 0-1 / 2-3 / 4-5 / 6+ | 1 of 9 / 8 of 22 / 9 of 15 / 9 of 12 |
| Syn- Eff+ (13): other positives | ACL 9, MM 6, LM 5, Frac 5, Cont 4, PF OA 4 |
| Syn+ Eff- (5): other positives | PF OA 4, MOA 2, MM 2, LM 2 |

**What the 2x2s say (GUESS from the MEASURED pattern).**
- If experts used MOAKS-style effusion-synovitis at a lower cut than Effusion, every Effusion-positive knee would also be
  synovitis-positive. Only 22 of 35 are, so that reading is out.
- The 13 effusion-only knees are mostly acute ACL / fracture knees (hemarthrosis or reactive fluid). The 5 synovitis-only knees
  are mostly patellofemoral OA knees. So the experts appear to call synovitis when they see effusion with synovial thickening
  in a degenerative knee, and not for acute traumatic fluid.
- Hoffa fat-pad signal is not the driver. The report-trained R7 state "fat_pad_inflammation" scores AUC .336 against expert
  synovitis, i.e. it is higher in expert NEGATIVES (MEASURED `syn_local.log` part 4). Hoffa wording appears in 19 % of both
  groups' reports (AUC .496).

**OAI comparison (MEASURED `box_syn.log`, n 1,177 OAI V00 knees with both MOAKS grades and every model scored).**

| MOAKS effusion-synovitis grade (rows) x Hoffa-synovitis grade (cols) | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | 276 | 206 | 15 | 1 |
| 1 | 169 | 237 | 36 | 4 |
| 2 | 34 | 94 | 45 | 4 |
| 3 | 6 | 26 | 23 | 1 |

| target | prevalence | P(target given eff >= 2) | P(target given eff < 2) | tet with eff >= 2 |
|---|---|---|---|---|
| Hoffa >= 1 (the Raptor author's OAI map, CLAIMED 746792) | .588 | .828 | .529 | .46 |
| Hoffa >= 2 | .110 | .313 | .059 | .58 |
| RSNA expert Synovitis vs Effusion (gold58) | .466 | .629 | .217 | .61 |

Neither Hoffa cut reproduces the expert pattern: Hoffa >= 1 is far too common in knees without effusion (.53 vs .22), and
Hoffa >= 2 is far too rare (.11 vs .47). OAI also has no acute knees, so it cannot show the "acute effusion is not synovitis"
split at all.

## 3. How reports and the LLM labels assign Synovitis (MEASURED `syn_local.log` part 2; train n 4,407)

| lang | n | synovitis word | synovial thickening words | Hoffa words | effusion words | v2 Syn YES | v2 Syn UNK | v2 Eff YES | P(v2 Syn YES given v2 Eff YES) |
|---|---|---|---|---|---|---|---|---|---|
| ALL | 4,407 | .105 | .021 | .222 | .843 | .132 | .843 | .605 | .206 |
| en | 1,736 | .168 | .041 | .305 | .906 | .181 | .803 | .622 | .276 |
| es | 682 | .155 | .001 | .345 | .669 | .150 | .805 | .421 | .348 |
| de | 262 | .073 | .061 | .290 | .679 | .191 | .756 | .733 | .240 |
| el | 321 | .009* | .003 | .059 | .713 | .134 | .838 | .732 | .166 |
| fr | 81 | .111 | .012 | .198 | .988 | .123 | .864 | .605 | .204 |
| nl | 153 | .039 | .000 | .281 | .758 | .072 | .915 | .706 | .102 |
| tr | 546 | .040 | .002 | .057 | .932 | .053 | .921 | .577 | .086 |
| hr | 406 | .012 | .000 | .059 | .901 | .049 | .929 | .658 | .067 |
| bg | 220 | .005 | .000 | .027 | .959 | .005 | .968 | .609 | .007 |

\* My regex misses the Greek phrasing (v2 YES .134 there), so the Greek word rate is an undercount. Regexes are in `syn_local.py`.

- Reports with a synovitis word: 463; v2 YES .952. Reports without one: v2 YES .035. The LLM label is essentially "the word is
  written".
- v2 Synovitis is UNKNOWN for 84 % of studies, so most training rows carry a soft or default negative.
- **gold58 (same 58 studies):** expert positives have the word in 41 % of reports (v2 YES .48); expert negatives in 13 %
  (v2 YES .10). The report-label readers score .68-.79 AUC against the experts (CLAIMED trick/regime_work/m1_regime.log), BELOW
  every image member (.78-.86, sec 4). Report Synovitis prevalence on gold58 is .28-.29 vs .47 expert (CLAIMED m1).
- So the report label is narrow (explicit wording only) and its rate depends on the site's language by a factor of about 35
  (bg .005 vs de .19), while the effusion label is reported at .42-.73 everywhere.

## 4. What each member output predicts (gold58; MEASURED `syn_local.log` part 3, `syn_g74.log`)

AUC against the gold58 expert Synovitis label (27 pos / 31 neg), unless a column says otherwise.

| member | Syn head | Eff head | Baker's head | Syn head vs gold EFF | Eff head vs gold Eff | rho(Syn, Eff) | E2 a .1 (P) | E2 a .2 (P) |
|---|---|---|---|---|---|---|---|---|
| R6 (c4, OAI-free) | **.859** | .797 | .588 | .846 | .945 | .74 | +.0042 (.73) | +.0060 (.68) |
| S6fall (c4, OAI-free) | .827 | .783 | .632 | .733 | .949 | .59 | +.0048 (.71) | +.0137 (.82) |
| Raptor r384 (c4; OAI Hoffa >= 1 rows for Syn) | .829 | .767 | .622 | .837 | .973 | .78 | +.0036 (.71) | +.0048 (.68) |
| Raptor r224 (c4) | .783 | .719 | .625 | .835 | .966 | .81 | +.0006 (.50) | +.0000 (.46) |
| Raptor N crop320 | .833 | .768 | .585 | .827 | .964 | .78 | +.0036 (.68) | +.0024 (.56) |
| Raptor N full384 | .836 | .760 | .584 | .839 | .961 | .80 | +.0012 (.54) | -.0042 (.33) |
| s43 (our S6 seed) | .806 | .780 | .562 | .713 | .912 | .69 | +.0084 (.90) | +.0155 (.91) |
| R7 OOF (report-trained) | .794 | .793 | .571 | .821 | .942 | .68 | +.0102 (.87) | +.0149 (.89) |
| Dread maxspan v5 (CoAtNet, no OAI) | .832 | .817 | .657 | .957 | .965 | .99 | about 0 | +.0012 |
| Dread native384 v8 (no OAI) | .815 | .805 | .662 | .983 | .988 | .99 | about 0 | -.0012 |
| Dread finespacing v9 (no OAI) | .802 | .790 | .659 | .969 | .969 | .99 | about 0 | -.0006 |
| fleetOF mean (ours, OAI-free) | .787 | .755 | .619 | .938 | .990 | .95 | +.0012 (.42) | -.0024 (.27) |
| c4 Raptor mix (.75 / .25) | .821 | .757 | .622 | .843 | .983 | .79 | +.0006 (.53) | +.0036 (.63) |
| **c4 final = pw30 Syn / Eff columns** | **.840** | .793 | .612 | .840 | .981 | .74 | **+.0036 (.76)** | **+.0084 (.82)** |

Within expert-Effusion strata (Eff+ 35 knees = 22 Syn+ / 13 Syn-; Eff- 23 = 5 / 18), c4 Syn head .750 / .844, c4 Eff head
.643 / .833. Every member shows the same order inside Eff+ (Syn head .61-.77 vs Eff head .55-.71).

gold74 (R7 OOF and our fleet have all 74 rows out of training; MEASURED `syn_g74.log`): R7 Syn .801, Eff .781, Baker's .609,
E2 a .1 +.0064 (P .81), a .2 +.0079 (P .79); fleetOF Syn .754, Eff .739, E2 +.0008 / +.0019. The extra 16 are our own reads.

R7 report-state heads vs expert Synovitis (gold58 / gold74; MEASURED part 4): Effusion "large" .809 / .817, "moderate" .765 /
.771, "small" .314 / .323; Synovitis "synovitis" .744 / .745, "synovial_thickening" .729 / .745, "fat_pad_inflammation" .336 /
.303, "other_synovial_finding" .339 / .304.

**Reads.**
- No member's Effusion head beats its Synovitis head on expert synovitis. "Synovitis = effusion" is false for these experts. The
  Syn head carries the "which effusions are synovitis" information (inside Eff+ it beats the Eff head in every member).
- Yet Effusion carries some extra expert-synovitis signal: E2 gains in all 8 members whose two heads differ (rho < .85) at a .1,
  in 6 of 8 at a .2. Members whose Syn head already equals their Eff head (Dread, fleet, rho .95-.99) show about 0, as they must.
  These members share the same 58 studies and 27 positives, so this is one sample seen 8 times, not 8 replications.
- R6 alone (.859) beats c4 (.840) on Synovitis. That is a member weight read off gold58 and stays banned (THINK_1019 sec 5).
- OAI Hoffa supervision did not visibly help: Raptor r384 (.829) is level with Dread maxspan (.832), a CoAtNet trained without
  OAI (n 27 positives, so this is weak).

## 5. OAI: MOAKS effusion-synovitis vs Hoffa, OAI-free and served members (MEASURED `box_syn.log`, B 500)

Models: R6 and S6fall served on the store-rebuilt OAI DICOMs (`~/r5_oai`, CLAIMED r5/r6_oai.md), their c4-style .8 / .2 rank
blend, the fleet teacher D5, s1044, s1045. All OAI-free. Raptor is OAI-trained and cannot be measured.
"w" = the same delta with the OAI knees reweighted so the (target, effusion >= 2) joint matches the gold58 (Syn, Eff) joint.

| target (prevalence) | head AUCs, R6.8S6.2 blend: Syn / Eff | rho(Syn, Eff) | E2 a .1 unweighted (P) | a .1 w | a .2 unweighted (P) | a .2 w (sd, P) |
|---|---|---|---|---|---|---|
| Hoffa >= 1 (.588) | .722 / .706 | .96 | -.0004 (.20) | +.0012 | -.0010 (.09) | +.0015 (.0007, .99) |
| Hoffa >= 2 (.110) | .801 / .791 | .96 | -.0005 (.17) | -.0006 | -.0006 (.23) | -.0011 (.0009, .11) |
| effusion-synovitis >= 1 (.577) | .858 / .884 | .96 | +.0035 (1.00) | +.0017 | +.0063 (1.00) | +.0030 |
| effusion-synovitis >= 2 (.198) | .950 / .961 | .96 | +.0015 (1.00) | +.0015 | +.0029 (1.00) | +.0029 |
| eff-syn >= 2 OR Hoffa >= 2 (.246) | .907 / .913 | .96 | +.0010 (1.00) | +.0010 | +.0019 (1.00) | +.0020 (.0005, 1.00) |
| Hoffa >= 1 AND eff-syn >= 1 (.399) | .851 / .860 | .96 | +.0020 (1.00) | +.0016 | +.0034 (1.00) | +.0022 |

Per model on Hoffa >= 1, unweighted, a .1: R6 -.0001, S6fall -.0005, D5 -.0003, s1044 -.0002, s1045 -.0003 (P .06-.36).
D5 under the gold58 reweighting: -.0002 (a .1, P .15), -.0007 (a .2, P .02).

**Reads.**
- On OAI the OAI-free Synovitis heads are effusion detectors: AUC .94-.96 for effusion-synovitis >= 2 vs .70-.72 for Hoffa >= 1.
  Their Syn and Eff heads nearly coincide there (rho .95-.99) even for R6 and S6fall, whose heads differ on real RSNA DICOMs
  (rho .74 / .59). Part of that is the store rebuild, which degrades Synovitis most (R6 exact vs rebuilt Spearman .829 on gold58,
  CLAIMED r6_oai.md sec 3).
- So OAI cannot be a ruler for E2. Its sign follows the target definition: positive for any effusion-based target (trivially),
  zero or negative for Hoffa targets unweighted, positive for Hoffa >= 1 only under the gold58 coupling (and only for the c4-shaped
  blend, not D5). The RSNA expert label matches none of these definitions (sec 2).
- OAI is one MRI protocol with no language / site split, so it also cannot test the site mechanism of sec 6.

## 6. The site (language) prior: a mechanism for E2 measured outside gold58

**Out-of-fold train (non-gold studies; MEASURED `syn_more.log` part A, `syn_site2.log`).**
- Among reports that say Effusion YES, the Synovitis score's mean rank by language follows that language's synovitis-reporting
  rate: Spearman over 9 languages .88 (R7, n 4,333), .68 (s1044), .62 (s1045). R7 example: en .660, es .639, de .690 vs bg .577,
  hr .531, tr .587, nl .586.
- The Effusion head tracks the effusion-reporting rate (Spearman .95-.97), which varies far less across languages.
- Controlling for the Effusion head (Syn rank regressed on Eff rank), the residual Synovitis score of low-report sites
  (bg / hr / tr / nl) minus the others:

  | model | all studies | v2 Eff YES | v2 Eff YES and v2 Syn not YES |
  |---|---|---|---|
  | R7 OOF | -.075 | -.078 | -.057 |
  | s1044 | -.020 | -.018 | -.011 |
  | s1045 | -.021 | -.019 | -.013 |

- Reading (MODEL): the report-trained Synovitis head learned "images from this site rarely get the word synovitis". Some of the
  gap could be real case mix, but the expert labels below show no lower synovitis rate at those sites.

**Pre-specified gold58 check (MEASURED `syn_site.log`).** The split is fixed by the train reporting rates (bg / hr / tr / nl =
v2 YES .005-.072). I wrote the three predictions in the script header before running it:
- (a) Expert synovitis is not rarer at low-report sites: 8 of 15 (.53) vs 19 of 43 (.44). Holds.
- (b) The Syn head under-ranks low-site expert positives relative to the Eff head, and not at high sites. Holds in 6 of 6
  models. Mean pooled rank of expert positives, Syn head vs Eff head:

  | model | low sites | high sites |
  |---|---|---|
  | c4 | .602 vs .686 | .727 vs .657 |
  | R6 | .657 vs .718 | .719 vs .646 |
  | S6fall | .591 vs .612 | .722 vs .680 |
  | Raptor r384 | .601 vs .685 | .720 vs .637 |
  | Raptor r224 | .556 vs .664 | .703 vs .610 |
  | R7 OOF | .645 vs .744 | .674 vs .632 |

- (c) E2's gain per positive is larger at low sites in 6 of 6 models at a .2 (5 of 6 at a .1). c4: +.0078 of the +.0084 comes
  from pairs whose positive is at a low site (8 positives), +.0006 from the other 19.
- **Room check:** an oracle that subtracts each site group's mean Synovitis rank (it uses the report language, which the test
  set does not have) gains c4 +.0078, R6 +.0048, S6fall +.0143. E2 at a .2 gains about the same: +.0084, +.0060, +.0137.
  E2 also lifts c4's within-low-site AUC (.821 to .857) and leaves the high-site AUC flat (.859 to .860).
- **Caveats:** the same 58 studies, 8 low-site positives, correlated members. The "regime flip" (report-OOF loses: R7 E2 a .2
  -0.11e-4 macro on v4 labels, MEASURED `syn_more.log`; -.0003 / -.0014 label CLAIMED THINK_1019) is what the mechanism
  predicts, because the report labels carry the same site prior. But any change that departs from the report labels would also
  lose there, so it does not discriminate.

## 7. E2 re-examined: the 10-19 KILL reasons, then and now

| KILL reason (CLAIMED THINK_1019 sec 8.2) | status tonight |
|---|---|
| gold58 gain is what picking the best of 9 pairs gives by chance | **Partly answered.** The mechanism is now measured on 4,333 OOF studies that never touch gold labels. Its gold58 fingerprint used a split fixed in advance and holds in 6 of 6 models. The SIZE still rests on 58 studies |
| not Synovitis-specific | **Still true, and sharper.** On gold58 at a .2, borrowing Fracture (+.0155) or Contusion (+.0125) also beats Effusion (+.0084). 6 of 11 helpers gain (CLAIMED push/census_dirs.csv). Their gains also land at low sites, but because the low-site positives happen to be trauma-heavy: their Contusion / Fracture ranks are +.17 / +.11 higher (ACL +.34, MCL +.20), so those gains look like case-mix. Effusion is site-neutral there (+.03), so its low-site gain comes from undoing the Synovitis head's own offset (-.125) (MEASURED `syn_site3.log`). So the choice of Effusion rests on mechanism, not on gold58 rank: highest expert coupling (tet .61 vs Cont .23, Frac .08) and site-consistent reporting |
| split-half sign agreement .499 | a .1 .54, a .2 .66 (MEASURED `syn_more.log`). Still fragile: net 4 (a .1) / 8 (a .2) pair flips of 837, the top 3 studies carry 93 % / 78 % of the gain |
| report-OOF ruler negative | explained as the regime flip (sec 6), but not discriminating |
| co-occurrence stacking lost on OOF and gold74 (CLAIMED NOVEL_PLAN_10-07) | a different operator (full cross-label stacker); not re-tested |
| no independent ruler | **still true for hidden labels.** OAI cannot define the target (sec 5) |
| (new) the E-OA calibration lesson | E-OA's gold58 gain (pw30 +2.23e-4) was about 2-2.5x its public-implied gain (+0.8 to +1.2e-4), while the OAI rulers understated it 2-5x (CLAIMED val/implied2.md, night/recal_rulers.log, brief). A real cross-label effect can transfer, and gold58 overstates it about 2.5x. I shrink E2's gold58 numbers by .4 below |

## 8. The one probe: E2s (pw30 + Synovitis borrows Effusion, a .2)

### 8.1 Spec
- Base: scored pw30 57425970 (the API best-id per the brief), notebook `a21/kernel_v224r6c4pw30j`.
- One change in cell 18 (`_s6f_blend`, S6F path), right after the A21 E-OA block:
  `out['Synovitis'] = pct(pct(Syn_final) + 0.2 * pct(Eff_final))`. It runs only if neither column is blanked; the receipt
  records `a_syn` and the log prints `SYN_E2_APPLIED`. Every other column and every other cell is byte-identical to pw30.
- **Dose a .2: this is the one gold58-informed choice.** Supporting a .2: the linear theory a* .12 (CLAIMED census_dirs), OAI
  Hoffa >= 1 under the gold58 coupling peaks at .2 (+.0015), the oracle site-centring equivalence is at .2 (sec 6), and it has the
  better split-half (.66 vs .54). **Conservative substitute: a .1**, the dose pre-specified in THINK_1019 sec 2.2. It has half
  the effect and lower read noise, and it is dominated in every row of the slate model below. One dose only; never both.
- Never stack E2s with pw25 or anything else in one entry.

### 8.2 Build and gates (all MEASURED, local, unarmed)
- `syn_build.py 0.2` -> `syn_kernel_v224r6c4pw30syn20j/` (notebook sha 1c6f65a4..., metadata = pw30's with id
  `eeshsaxena/rsna-sub-v224r6c4pw30syn20j`); `syn_build.py 0` -> no-op control `..syn00j` (sha 5b87f213...). Receipts:
  `syn_build_receipt_*.json` (armed: false). Only cell 18 differs from pw30, by +6 lines.
- `syn_gate.py` -> `syn_gate.log`: **SYN_GATE PASS**. The syn00 output is byte-identical to pw30's on gold58. In syn20, 11 of
  11 other columns are identical. Synovitis equals an independent recompute (max diff 4e-8), with dAUC +.0084, gold58 macro
  .94406 vs pw30 .94336.
- `syn_sim.py` -> `syn_sim.log`: simulated commit output (pw30's REAL committed dir + syn20's cell 18 replayed; written under
  `night/syn_sim/`). The pw30 replay equals its real committed submission.csv. **SYN_SIM_GATE_SELFTEST PASS 5/5**:

  | gate run | expected | got |
  |---|---|---|
  | E2s gate on the syn20 sim | PASS | PASS 18/18 |
  | E2s gate on the real pw30 dir | FAIL | FAIL, (d3) |
  | E2s gate with asyn 0 on the real pw30 dir (the a21 gate plus a "no E2s" check) | PASS | PASS 18/18 |
  | E2s gate with asyn 0 on the syn20 sim | FAIL | FAIL, (d3) |
  | E2s gate at the wrong dose (asyn .1) on the syn20 sim | FAIL | FAIL, (d3) |

- **Caveat (MEASURED):** on the 3 public test rows E2s does not change the Synovitis ranks (pw30 and syn20 outputs are
  byte-identical there). The dose identity therefore rests on the notebook sha plus the receipt / log check (d3), as for the
  OA doses in VAL_1020.
- Commit gate command (on the fetched committed dir `<OUT>`; must print `PROBE_CHECK PASS 18/18`):
  `py -3.12 -I session_2026-10-20/night/syn_probe_check.py <OUT> --a224 0.25 --table '{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}' --aoa 0.3 --mode pw --asyn 0.2`
- **Not done (needs main and Eesh):** no registry entry. The A22 tick registry `val/slate_a22.json` is read-only for me, so E2s
  cannot be pushed or submitted by the existing tick without main adding a key. No PREREG text was written into PREREG.

### 8.3 Honest odds (MODEL; `syn_more.log` part C, `syn_slate_mc.log`)
Inputs:
- If real: gold58 label gain .0084 x .4 (E-OA shrink) = +2.8e-4 macro.
- If not real: -2.0e-4 macro, about 2x the OAI-unweighted and report-OOF losses, to be conservative.
- P(real) .5 (GUESS; range .35-.65).
- Paired public read sd 2.9e-4 macro (MEASURED: gold58 bootstrap scaled to 390 studies, 3.0e-4; R7 OOF subsample, 2.8e-4).
  The private sd is about 1.9e-4.

| P(real) | P(true > 0) | E[true] | P(best-id move) | E[private gain, probe + R1] | P(adopted and truly harmful) |
|---|---|---|---|---|---|
| .35 | .43 | -0.3e-4 | .46 | +0.8e-4 | .13 |
| .50 | .54 | +0.4e-4 | .54 | +1.2e-4 | .11 |
| .65 | .65 | +1.1e-4 | .61 | +1.6e-4 | .08 |

- At a .1 the same rows give +0.3 / +0.5 / +0.7e-4 probe value (read sd 1.8e-4).
- **Slate (MODEL, R1 = max public of pw30 and the probes):**
  - pw25 alone: +0.33e-4. Inputs: true pw25 - pw30 ~ N(+0.3, 0.8e-4), public paired sd 1.4e-4 (CLAIMED implied2 l.257).
  - E2s alone: +1.19e-4.
  - pw25 + E2s: +1.34e-4.
  - The two probes touch different columns, so their reads are independent and they add rather than compete.
- **Most of the probe value is R1's option value on a noisy label.** Any change with this much prior spread shows +0.5 to
  +1e-4 here (the census made the same point, CLAIMED push/census.md sec 6). E2s's edge over a random Synovitis tweak is only
  the mechanism behind P(real) .5 instead of about .3.
- **Display (MODEL, rough):** if pw30's unrounded public is about .9599 (CLAIMED VAL_1020), E2s shows .960 with P about .3-.4.
  That chance is driven by noise. Do not read a .960 as proof.
- **What makes this a NO:**
  - Eesh wants the minimum slate.
  - He will not let a Synovitis-only change become A.
  - He prefers no Synovitis exposure. Synovitis is about 1/4 of the private macro variance (CLAIMED THINK_1019 sec 2.2).

### 8.4 Draft PREREG text (for main, only on Eesh's go; NOT written into PREREG)
> ### AMENDMENT 23 (10-21 slate add-on: E2s, one A probe on the Synovitis column). Written <time> IST by Claude, BEFORE its commit and submit. Eesh: "<quote>".
> - e2s = rsna-sub-v224r6c4pw30syn20j (nb sha 1c6f65a4...): scored pw30 57425970 + `Synovitis = pct(pct(Syn) + 0.2 pct(Eff))` on the final output, after E-OA (cell 18, +6 lines). No other change.
> - Evidence: session_2026-10-20/night/synovitis.md. Mechanism: the report-trained Synovitis head carries a site / language prior (OOF Spearman .62-.88 over 9 languages) that expert labels do not have; Effusion is site-neutral and the experts couple it with Synovitis (tet .61). gold58 is info only (+.0084 label, fragile). No hidden-label ruler; OAI cannot define the target.
> - Gates: night/syn_gate.log SYN_GATE PASS; commit gate night/syn_probe_check.py (pw30 params + --asyn 0.2) must PASS 18/18.
> - Read by R1 vs the then API best-id. If it moves the best-id, A = e2s once; B unchanged (r6pw20 rule of A22). No retry at another dose, no stacking with pw25 or any other winner.

## 9. Other Synovitis ideas, not recommended

| idea | numbers | verdict |
|---|---|---|
| E2 at a .1 (THINK_1019 spec) | gold58 +.0036 (P .76), net 4 pair flips, top-3 studies 93 %; probe value +0.5e-4 at P(real) .5 (MODEL) | substitute only; dominated by a .2 in every modelled row |
| POST HOC E2 x (1 - ACL score) ("acute effusions are not synovitis") | gold58 +.0119 at a .2 (MEASURED `syn_more.log`) | NO: found after looking; the acute decorrelation alone hurts c4 (Syn <- ACL -.0024, CLAIMED census) |
| Borrow Fracture / Contusion into Synovitis | gold58 +.0155 / +.0125 at a .2 | NO: composition artefact of 8 low-site positives (sec 7); expert coupling .08 / .23 |
| Hoffa route: lean on the OAI Hoffa-trained Raptor for Synovitis | Raptor r384 .829 = Dread no-OAI .832; R7 fat-pad state anti-predicts (.336) | NO |
| Shift c4's Synovitis weight toward R6 (R6 .859 > c4 .840) | gold58 only | NO: member weights from gold58 are banned (THINK_1019 sec 5). Note that B (r6pw20) already sits on R6's side of the Synovitis weights |
| OR / AND member combiner for a "broad" label | gold58 OR -.0014, AND +.0038; OAI Hoffa OR -.0008 (CLAIMED expert2.md) | NO (dead) |
| Syn <- Medial OA decorrelation (best census Synovitis direction) | gold58 +.0060 (P .76) | NO: no mechanism, no ruler |
| Site-centred Synovitis score (the oracle of sec 6) | +.0078 on gold58 | NOT IMPLEMENTABLE: the test set has no report or language; metadata site priors already lost on gold58 (-.0152, CLAIMED THINK_1019) |
| Retrain Synovitis on expert-like targets | n/a | out of the window (freeze, no GPU) |

## 10. Files (all in `session_2026-10-20/night/`)
- **Local scripts and logs (RSNA data only):**
  - `syn_local.py` / `.log`: label structure, report language table, member heads, R7 states.
  - `syn_more.py` / `.log`: OOF language prior, E2 influence and pair flips, read noise, probe MC.
  - `syn_site.py` / `.log`: pre-specified site split, room check.
  - `syn_site2.py` / `.log`: OOF site prior controlling for the Eff head.
  - `syn_site3.py` / `.log`: specificity over the 11 helpers.
  - `syn_g74.py` / `.log`: gold74 members.
  - `syn_slate_mc.py` / `.log`: slate model.
- **Box scripts (aggregates only) and their logs:** `box_syn_inspect.py` / `.log`, `box_syn.py` -> `box_syn_smoke.log` (B 20),
  `box_syn.log` (B 500).
- **Build (unarmed):**
  - `syn_build.py`, `syn_kernel_v224r6c4pw30syn20j/`, `syn_kernel_v224r6c4pw30syn00j/`, `syn_build_receipt_*.json`.
  - Gates: `syn_gate.py` / `.log`, `syn_probe_check.py` (copy of a21/probe_check_eoa21.py + `--asyn`), `syn_sim.py` / `.log`,
    `syn_sim/out_v224r6c4pw30syn20j_sim/`.
- **Read only:** THINK_1019.md, EX_1019.md, push/census.md + census_dirs.csv + census_genfactor.csv + cbase.py, more/expert2.md,
  levers/LEVERS_1020.md, r5/r6_oai.md, val/VAL_1020.md + implied2.md, a21 kernels / gate, forum flat_733343 / 733491 / 737155 /
  740610 / 746792, trick/regime_work/m1_regime.log + m2_states.log.
