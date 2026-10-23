# lmfx.md: Track 3, Lateral Meniscus (.928) and Fracture (.936) deep dive (10-20 night, ~20:30-22:15 IST, Claude)

RESEARCH + LOCAL MEASUREMENT ONLY. Nothing was built, pushed, committed or submitted. No dataset, download or scheduled task.
PREREG, HANDOFF, PROGRESS, a20, a21 and val were only read. No GPU job (the box V100 was busy with someone else's job; I used
CPU only).

**OAI governance.** One read-only CPU script (`lmfx_work/o1_lm_oai.py`) was piped over ssh stdin, first as a 20-boot smoke run,
then the full 1,000-boot run. It printed aggregates only: n, prevalence, mean percentile, AUC, AUC deltas, sd, P, phi, Spearman.
It wrote nothing on the box. No OAI id, row or per-knee value left the box. The only data sent TO the box was the gold74 ACL x LM
2x2 log OR (one number). Earlier read-only `ls` / `grep` calls touched file names and the label builder's code comments, not OAI rows.

**Tags.** MEASURED = computed tonight, log named (all logs in `session_2026-10-20/night/lmfx_work/`). CLAIMED = an older note
says so (named). MODEL = arithmetic on stated inputs. GUESS = judgement. "label" = single-label AUC delta; "macro" = label / 12.
gold58 is descriptive here, never a selection ruler. e-4 = 1e-4 macro.

## 1. Verdict

1. **No probe for either label.** Every output-only fix I could tie to a mechanism either fails its only independent ruler (LM)
   or has no independent ruler at all (Fracture). Leave the 3 free 10-21 slots for nothing from this track.
2. **Lateral Meniscus.** c4 misses LM tears in acute ACL-injury knees and over-calls LM in other acute knees: c4 LM AUC is .867
   within expert-ACL+ knees vs .963 in ACL- knees on gold58 (MEASURED g1). The misses are subtle posterior-horn / root / radial
   tears that the report missed or played down. The independent ruler (OAI MOAKS, R6 and S6fall served on 1,704 OAI knees) kills
   every context or coupling fix at its pre-registered rule: all deltas < 0, P>0 <= .31 for R6, S6fall and the c4-shaped proxy
   (MEASURED o1). The OAI weak spot is intrasubstance degeneration, ranked like a half-tear (tear vs signal-only AUC .88 vs tear vs
   normal .97), and no other output column separates the two. c4's R6:S6fall balance for LM already sits on the flat OAI optimum.
3. **Fracture.** c4's misses sit in bone-bruise knees (Fracture AUC .789 inside expert-Contusion+ knees vs .960 outside;
   MEASURED g4). The experts split the impaction / osteochondral / insufficiency continuum close to a coin flip (8 of 18
   impaction-worded and 2 of 5 insufficiency-worded gold58 reports are expert-positive; MEASURED g5). OAI has no fracture grade,
   gold74's extra 16 are our own reads, and the one lever with signal (Raptor share) was already read on the LB (fr100 below c4,
   CLAIMED PREREG A19 read). Nothing output-only is left that a ruler could check.
4. **Best candidates, for the record (both NO):**
   - LM: ACL-context shrink k .4. gold58 +.0062 label (+5.2e-4 macro, P .90), but OAI -.0054 to -.0091 label (P .00), report OOF
     -.0046 / -.0104, gold74 R7 0. E private about -1.5e-4 (MODEL).
   - Fracture: Frac <- mean(OA trio) a .1. gold58 +.0042 label, but a placebo helper set gains MORE (+.0076). Report OOF -.0017 /
     -.0023, census KILL (CLAIMED census.md). E private about -0.5e-4 (MODEL).

## 2. Lateral Meniscus: what the experts respond to that we miss

### 2.1 gold58 error anatomy (MEASURED `g1_errors.log`, `g4_member_strata.log`; n 58, 23 positives)

| model | LM AUC (boot sd) | within trauma+ (ACL, Cont or MCL; 32, 11 pos) | within trauma- (26, 12 pos) | within ACL+ (24, 9 pos) |
|---|---|---|---|---|
| R6 | .916 (.036) | .844 | .976 | .867 |
| S6fall | .935 (.032) | .918 | .952 | .919 |
| Raptor (.75 r384 + .25 r224) | .894 (.046) | .870 | .923 | .800 |
| R7 OOF | .924 (.032) | .874 | .964 | .911 |
| c4 | .928 (.033) | .870 | .973 | .867 |

- **The loss is concentrated.** 58 discordant pairs of 805. The 3 / 5 worst positives carry 63 % / 85 % of them. 15 of 23
  positives have zero discordance.
- **Who the worst positives are** (my reading of the reports, competition data only). 4 of the 5 worst are acute ACL-injury knees:
  - c4 pct .34: Dutch report, "menisci normal"; complete ACL rupture with lateral impaction fracture. The experts found an LM tear
    the report missed.
  - .50: small incomplete radial tear at the posterior-horn / body junction, acute ACL (Spanish).
  - .55: Wrisberg-rip peripheral posterior-horn tear plus root avulsions, in a multi-fracture knee (English).
  - .59: a 290-character report, "both menisci torn", with ACL + MCL injury.
  - The fifth (.45) is a "suspected small horizontal anterior-horn tear" in a tricompartmental OA knee.
- **Who the worst negatives are.** Lateral-compartment trauma with a normal LM: "no lateral meniscal tear" plus a lateral
  plateau impaction fracture (.74), "menisci without particularity" plus a lateral plateau fracture (.67), and ACL tears with
  "lateral meniscus not torn" (.53, .57).
  - Also degeneration not reaching the surface: "degenerative signal throughout without surfacing tear" (.72), Turkish "grade II
    degeneration" (.66), "mild anterior-horn degeneration" (.60).
  - One is a report-stated complex posterior-horn LM tear that the experts rejected (.64).
- **Report vs expert agreement drops in trauma knees** (MEASURED g3; v4 labels): .78 in trauma+ knees vs .92 in trauma- knees.
  - On the same 58 studies the reports call LM in .39 of report-trauma knees (expert .34) and .35 of the others (expert .50).
  - Train reports couple LM with ACL strongly: P(LM | ACL) .453 vs .200 without ACL (n 1,147 / 2,732).
  - The experts do not: .375 vs .412 on gold58 and .379 vs .422 on gold74. This is the census's ACL-LM excess (+.17, CLAIMED
    census.md row 18), seen from the label side.

**Read (GUESS on mechanism, MEASURED on pattern).** In acute ACL knees the LM output tracks the lateral-compartment injury
(pivot-shift bone bruise, lateral plateau or condyle impaction) more than the meniscus itself. The experts' "surface contact on
>= 2 images" rule finds subtle posterior-horn and root tears the reports miss, and rejects some report-called tears. That costs
c4 about .10 AUC inside the trauma stratum. The second channel is the rubric's "degeneration is negative" rule against
report-trained scores that put degeneration halfway to a tear.

### 2.2 OAI MOAKS morphology with the served R6 / S6fall (MEASURED `o1_lm_oai.log`, n 1,704 knees, aggregates only)

- B1 tear (MOAKS morphology 2-8 or root tear): prevalence .209. Intrasubstance signal only (code 1): .164. Normal: .627.
- Mean percentile of the LM output by morphology:

  | category | n | R6 | S6fall | c4-shaped (D5 in the Raptor slot) |
  |---|---|---|---|---|
  | normal | 1,069 | .355 | .358 | .356 |
  | signal only (degeneration, rubric-negative) | 279 | .599 | .592 | .600 |
  | horizontal tear (code 3) | 244 | .850 | .846 | .848 |
  | code 4 (complex per MOAKS order; mapping GUESS) | 12 | .703 | .688 | .697 |
  | maceration (6-8) | 143 | .915 | .927 | .920 |
  | tear in the posterior horn | 215 | .893 | .889 | .891 |
  | tear in the anterior horn only | 33 | .783 | .779 | .773 |

- **Key AUCs (R6 / S6fall / c4-shaped):**
  - B1: .953 / .949 / .952.
  - Tear vs signal-only: .881 / .878 / .878.
  - Tear vs normal: .972 / .968 / .972.
  - Signal-only vs normal: .784 / .771 / .788.
- **So on an expert ruler the members' LM confusion is degeneration vs tear**, plus anterior-horn-only and complex tears. On OAI
  LM is not a weak label (R6 .953 vs gold58 .916). The gold58 deficit comes from the acute-trauma stratum, and OAI, an OA cohort
  with chronic ACL tears, has no such stratum. On OAI the members are better, not worse, inside the ACL-tear and
  high-trauma-output strata (R6 .973 / .959 vs .950 / .945).
- **OAI expert coupling.** ACL-LM phi +.113, P(LM | ACL) .316 vs .160: chronic ACL deficiency goes with degenerative LM tears.
  Model Spearman ACL-LM: R6 .131, S6fall .438, D5 .452. LOA-LM: expert phi .363, models .68-.74.

## 3. Fracture: what the experts respond to that we miss

### 3.1 gold58 error anatomy (MEASURED `g1_errors.log`, `g4_member_strata.log`; 18 positives)

| model | Fracture AUC (boot sd) | within Contusion+ (19, 10 pos) | within Contusion- (39, 8 pos) | within any-OA+ (29, 5 pos) |
|---|---|---|---|---|
| R6 | .887 (.054) | .744 | .875 | .967 |
| S6fall | .901 (.053) | .756 | .948 | .992 |
| Raptor | .958 (.023) | .856 | .992 | 1.000 |
| R7 OOF | .860 (.060) | .711 | .903 | .950 |
| c4 | .936 (.033) | .789 | .960 | .992 |

- **Raptor beats R6 on gold58 in both strata** (+.111 Contusion+, P .95; +.117 Contusion-, P .87). But Raptor was selected on
  gold58 (CLAIMED cell 14 comment, PREREG A19), and fr100 (Raptor only) read below c4 on the public LB. The member axis is spent.
- **Discordance is concentrated.** 46 pairs of 720. The 3 worst positives carry 71 %.
- **Misses.**
  - c4 .40, R6 .10, R7 .17: a tibial intercondylar-eminence (tibial spine) fracture with intact cruciates. It is avulsion-type
    and rare in the training reports. The report says fracture, yet R7's state head reads "none" .99.
  - .54: the report says "bone contusion posterolateral tibial plateau" only, with an ACL rupture. The experts called it fracture
    AND contusion: the pivot-shift impaction was read as a fracture.
  - .67: residual bruise with subchondral impression, bony avulsion of an ACL bundle and an ossicle at the LM root (German).
- **False alarms.**
  - .86: bone contusions plus a grade-2 MCL, no fracture in the report.
  - .82: Greek report of SONK / microtrabecular insufficiency fractures in a degenerative knee. Expert: no fracture.
  - .72: "osteochondral impaction injuries" of both posterior plateaus with an ACL tear. Expert: contusion, no fracture.
  - .69: a Bulgarian report describing an arc-shaped osteochondral fracture line of the medial condyle. Expert: no fracture.
- **Co-label strata** (MEASURED g1). Expert-Contusion+ negatives sit at c4 pct .65, vs .29 for Contusion- negatives. Contusion+
  positives sit at .82 and Contusion- positives at .80, so the stratum's problem is bruise-only knees that look like fractures.

### 3.2 Report wording (MEASURED `g5_report_words.log`; rough multilingual keywords, GUESS-level precision)

- **Train (non-gold, 290 report-YES fractures, v2).** Keyword shares overlap:

  | type | share of report-YES fractures | R7 OOF Fracture pct when YES |
  |---|---|---|
  | plateau / condyle | .52 | .906 |
  | impaction / osteochondral / notch | .34 | .874 |
  | insufficiency / SIFK / SONK / stress | .19 | .888 |
  | avulsion / Segond / spine | .16 | .827 |
  | microtrabecular | .09 | .915 |
  | old / healed | .04 | .811 |

  The report-trained head fires on every type.
- **gold58, expert Fracture+ by report keyword:**

  | keyword | expert Fracture+ |
  |---|---|
  | plateau / condyle | 11 / 14 |
  | avulsion | 2 / 2 |
  | impaction / osteochondral | 8 / 18 |
  | insufficiency | 2 / 5 |
  | microtrabecular | 0 / 1 |

- **Read.** A discrete plateau or condyle fracture or an avulsion is usually expert-positive. Impaction and osteochondral
  injuries and insufficiency fractures are near coin flips. Report vs expert agreement on Fracture is .75 in trauma knees vs .88
  otherwise (MEASURED g3; v4 sens .78, spec .82 overall). The expert boundary runs through the bone-bruise / impaction continuum,
  and only the image (fracture line yes / no) decides it. No other output column carries that information.

## 4. Candidate fixes and their rulers

Each fix is one deployable line on the final c4 columns, 0 runtime. "shrink k" = .5 + (pct f - .5)(1 - k ctx), ctx = mean pct of
the context columns; "decor a" = pct f - a ctx. All numbers are label AUC deltas, MEASURED (g2, o1; OAI rows paired bootstrap B
1,000).

| label | candidate | gold58 c4 (P) | gold58 R6 / S6fall | gold74 R7 | report OOF v2 / sol56 | OAI R6 / S6fall / c4-shaped (P>0) | verdict |
|---|---|---|---|---|---|---|---|
| LM | trauma-ctx shrink k .4 (ACL, Cont, Frac) | +.0062 (.89) | +.0025 / .0000 | +.0015 | -.0026 / -.0046 | -.0056 / -.0037 / -.0049 (.00) | KILL (OAI rule) |
| LM | ACL-ctx shrink k .4 | +.0062 (.90) | +.0037 / +.0012 | .0000 | -.0046 / -.0104 | -.0091 / -.0054 / -.0068 (.00) | KILL (OAI rule) |
| LM | LM <- ACL a .1 | +.0031 (.68) | .0000 / +.0037 | +.0027 | -.0028 / -.0054 | -.0036 / -.0011 / -.0021 (.00-.17) | KILL (also census) |
| LM | LM <- LOA a .1 / LM += LOA a -.1 | n/a | n/a | n/a | n/a | -.0013 / -.0013 / -.0005 (.08-.31); borrow -.0018 to -.0021 | KILL |
| LM | placebo shrink (Bak, MOA, PF) | -.0025 | -.0037 / +.0025 | -.0053 | -.0023 / +.0016 | -.0032 / -.0035 / -.0032 | control behaves |
| LM | S6fall share inside the OAI-free part | (CLAIMED s6_a224 table: ws .3 +.0037) | | | | w .3 vs .2 +.00034 (P .80); flat .2-.5; c4 already about .29 | no room |
| Fracture | Contusion-ctx shrink k .4 | +.0014 (.58) | -.0028 / +.0028 | -.0005 | -.0046 / -.0124 | no fracture grade | DEAD |
| Fracture | Frac <- OA trio a .1 | +.0042 (.67) | +.0069 / +.0014 | +.0041 | -.0017 / -.0023 | no fracture grade | DEAD (non-specific) |
| Fracture | placebo decor (Bak, MM, Syn) a .1 | +.0076 (.81) | +.0069 / +.0028 | +.0027 | -.0003 / -.0003 | none | the control gains more |

- **Pre-registered OAI rule** (written in the o1 header before the run): SUPPORT needs > 0 with P >= .90 for R6, S6fall and
  c4-shaped. KILL if <= 0 or P < .5 for 2+. Every LM candidate is KILL. The tilt to gold74 ACL-LM coupling (R2) does not rescue
  the k .4 shrinks (-.0028 to -.0065; k .2 about 0).
- **Caveat on the LM kill (MODEL).** OAI's ACL knees are chronic, and there OAI's members read LM *better*. So OAI can kill the
  shrink as a general rule but cannot test the acute-trauma mechanism itself. The other population-matched ruler (RSNA train,
  report labels) also says lose. Only gold58 says gain: +.0062 label, about 1.2 sd, picked from 15 context x k variants.
  That leaves no independent support.

## 5. Probes (at most one per label; both NO)

| | LM: ACL-context shrink k .4 on pw30 | Fracture: Frac <- mean pct(MOA, LOA, PF) a .1 on pw30 |
|---|---|---|
| change | cell 18, after the E-OA block: LM = .5 + (pct LM - .5)(1 - .4 pct ACL) | cell 18: Frac = pct(Frac) - .1 mean pct(OA trio) |
| offline | gold58 +.0062 label / +5.2e-4 macro (P .90). OAI KILL: c4-shaped -.0068 (P .00), R6 -.0091, S6fall -.0054. Report OOF -.0046 / -.0104. gold74 R7 .0000 | gold58 +.0042 / +3.5e-4 (P .67). Placebo helpers +.0076. gold74 R7 +.0041. Report OOF -.0017 / -.0023. No independent ruler. Census KILL (CLAIMED) |
| read noise (MEASURED g6) | public sd 4.0e-4, private 2.5e-4 macro | public 3.2e-4, private 2.0e-4 |
| calibrated public (MODEL) | pw30 -1.5e-4 (sd 4e-4); display stays .959; P(.960) about .05 | pw30 -0.5e-4 (sd 3e-4); display .959 |
| P(best-id move) | about .35 (Phi(-.4)), mostly noise | about .45, mostly noise |
| private expected (MODEL) | about -1.5e-4: P(gold58 effect real) .25 x +2e-4, else -2.5e-4 (report / OAI direction). P(>0) about .3 | about -0.5e-4, P(>0) about .45 |
| verdict | NO | NO |

**Why not a "cheap lottery" slot.** Under R1, a probe whose expected private delta is <= 0 still moves the best-id 35-45 % of the
time on public noise alone. That is the same winner's-curse problem VAL_1020 sec 4 found for the asked-five doses. Each move
replaces pw30 with something about equal or worse in expectation.

## 6. What would change this

- An expert-graded external set with fracture or acute-trauma meniscus labels (none exists before 10-22).
- An image-level signal for "fracture line vs bruise" or "tear vs degeneration": a new head or retraining on expert-like
  targets. That is out of the window (CLAIMED trick/label_regime.md sec 4).
- Post-competition: train LM targets that do not inherit the report's ACL-LM coupling, and Fracture targets that separate a
  discrete line or avulsion from impaction / insufficiency. The report census above gives the type mix to target.

## 7. Files (`session_2026-10-20/night/`)

- `lmfx.md`: this note.
- `lmfx_work/`:
  - `g1_errors.py` / `.log`: gold58 member AUCs, discordance decomposition, co-label strata, R7 state heads.
  - `g2_ctx.py` / `.log`: context shrink / decor on gold58 c4 / R6 / S6fall, gold74 R7, train report OOF.
  - `g3_regime_strata.py` / `.log`: report vs expert agreement by stratum, ACL-LM coupling.
  - `g4_member_strata.py` / `.log`: member AUCs within strata.
  - `g5_report_words.py` / `.log`: multilingual keyword census of the fracture types and LM wording.
  - `g6_noise.py` / `.log`: public / private read noise.
  - `o1_lm_oai.py` / `o1_lm_oai.log` / `o1_lm_oai_smoke.log`: box script and aggregate-only output.
- Read only: VAL_1020.md, expert2.md, census.md, cand_menisci.md, cand_global.md, label_regime.md, m2_states.log,
  s6_a224_r6_tables.md, r6_oai.md, PREREG_10-15.md (A18-A20 reads), a21 build_a21.py and the pw30 kernel cells 14 / 18.
