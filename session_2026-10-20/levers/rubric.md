# rubric.md: host rubric vs report language, label by label, and the output-only levers it suggests (10-20 ~15:40-16:30 IST, Claude)

RESEARCH + MEASUREMENT ONLY. Nothing was submitted, pushed, committed or built into a kernel. No dataset, no download, no
scheduled-task edit, no PREREG / HANDOFF / PROGRESS edit; a20 / a21 were not touched (only the c4 kernel source was read).
Box: three read-only CPU python runs piped over ssh stdin (`box_struct.py` (deleted, see sec 7), `box_struct2.py`,
`box_rubric.py`), output passed through an id-redacting `sed`; nothing written on the box. Box GPU NOT used (nvidia-smi at
~15:40 IST showed 85 % / 4.1 GB used by another job). One governance slip, sec 7.
Tags: MEASURED (computed here, log named), CLAIMED (older note or host, source named), MODEL (arithmetic on stated inputs),
GUESS (judgement). gold58 appears only as a consistency check, never to select.

## 0. Verdict (plain)

1. **No new slot-worthy lever.** The rubric-vs-report differences in the members' TRAINING LABELS are large, but they are almost
   all severity thresholds (small effusion, small Baker's, ACL / MCL low-grade sprain, low-grade chondral loss), and the members
   already rank severity monotonically. On rulers that apply the rubric threshold, the members score HIGHER than on the report
   threshold (R7 OOF Effusion .941 vs .832, ACL .960 vs .860, MCL .949 vs .843; MEASURED `local_states.log`). That
   contamination costs nothing that an output-only transform can recover.
2. **The one physical second view, R6's own fluid volume, fails on two independent rulers** (T-VOL, sec 3.1). On OAI MOAKS
   effusion it helps at the report-like cut (grade >= 1, +.0055 at b .2 for the c4-shaped proxy) and NOT at the rubric cut
   (grade >= 2, -.0000, P .49). The same pattern holds on the report text re-thresholded at the rubric (mean3 -.0021, P .12).
   That is the opposite of the regime prediction. DEAD. Baker's volume loses everywhere. DEAD.
3. **The rubric-mapped state read-out (T-STATE) does not pass the new rubric-text ruler** (sec 3.2): Effusion -.0023, menisci
   -.0015 / -.0005, ACL -.0010, Contusion -.0030, Fracture -.0042 (mix .5, R7 OOF). Only Baker's (+.0067, P .84, 68 positives)
   and MCL (+.0011, 22 positives) show the regime flip, both small-n, and R6's states are not exposed in c4 (no `--dump-state`).
   DEAD as a slot.
4. **Non-monotone subclasses (ACL mucoid degeneration, MCL chronic thickening) are real in the labels but tiny in AUC**: the
   perfect-demotion bound is +.0018 / +.0035 per label on train OOF (MEASURED `local_subclass.log`), ~+.0002 macro, before any
   real transform. DEAD.
5. **Side finding that supports the in-flight E-OA "pw" shape** (not a new entry): among studies whose LLM labels say Lateral OA =
   0, the OOF Lateral OA output is clearly higher when the report describes LATERAL PATELLAR FACET / trochlear chondral damage
   than other patellofemoral damage (AUC .745-.764, 3 OOF models, P 1.00); the medial analogue is weaker (.670-.713) (MEASURED
   `local_facet.log`). The LLM labels themselves leak little (facet-only -> Lateral OA label .13-.16 vs .09-.10 for other PF
   damage; MEASURED `census_lang.log`, `local_subclass.log`). So the PF -> Lateral OA coupling is mostly in the IMAGE MODEL, and
   it is stronger laterally, as pw's pair weights (Lateral-PF 1.416 > Medial-PF .708, CLAIMED a21/build_a21.py) assume.
   This is a tie-break input for the A21 reads, not a rule change.
6. **Recommendation: nothing from this lens enters the 10-21 slate.** The PUSH_1020 slate (R6e20, optional E1 / shape) stands.

## 1. Facts that correct or sharpen the brief

- **The members' LLM labels already follow the rubric on meniscal degeneration, degenerative marrow edema, osteochondral lesions
  and SIFK.** MEASURED `census_lang.log` (EN, n 1,736):
  - meniscal "degeneration / intrasubstance signal, no tear": labelled positive .03-.04 (MM, n 120) and .00-.04 (LM, n 74) by
    v4 / JEV / sol56 / blendjev_sol. Only our own v2 labeler calls them positive (.88 / .85); no c4 member learned v2.
    Turkish spot check: .15 (n 128; `local_ml.log`).
  - BME in a degenerative / subchondral context without "contusion": .04-.05 (n 172), v2 .59.
  - osteochondral lesion / OCD without the word fracture: .00-.05 (n 44); gold58 experts 0 / 3.
  - insufficiency / SIFK / subchondral fracture: .71-.75 (n 55); gold58 experts 5 / 6 positive, so experts count SIFK too.
- **ACL reconstruction is too rare to matter**: 1 EN report (MEASURED). Post-meniscectomy "truncated" menisci (rubric positive):
  2 + 2 EN reports.
- **The brief lists "BML" among the box MOAKS sub-grades. `oai_label_sources.csv` has no BML column** (63 columns: cartilage,
  meniscal morphology / hypertrophy / root, effusion, Hoffa, popliteal cyst, osteophytes, ACL; CLAIMED `more/box_inspect.log`).
  So contusion-vs-degenerative-BML has no OAI ruler.
- **Box R6 OAI outputs hold no state heads**: `dump.json` has z / z_plain / z_mirror_raw only (MEASURED `box_struct2.py` run;
  sec 7). R6's fluid volumes (`fluid_stacker.csv`) and Baker's state-stacker inputs are there and are used in sec 3.1.

## 2. Label by label (EN train reports; regex at sentence level with a 45-character negation window; MEASURED `census_lang.log`)

"labels+" = share labelled >= .5 by v4 / JEV / sol56 (blendjev_sol and the S6fall-style mean(v4, JEV) agree within .03).
"contam" = share of a label source's EN positives that fall in a rubric-NEGATIVE category (regex, approximate).

| label | rubric (forum 733343, CLAIMED) | report language found (EN n, share) | labels+ | contam | type | output-only lever | independent ruler | result |
|---|---|---|---|---|---|---|---|---|
| ACL | high-grade partial (>50 %) or complete; signal / degeneration / thickening negative | low-grade / sprain / grade 1: 118 (6.8 %); mucoid / degeneration, no tear: 75 (4.3 %); chronic tear 63 | .91-.97; mucoid v4 .47 / JEV .95 / sol56 .43; chronic 1.00 | .29-.35 | severity + a small non-monotone subclass (mucoid) | demote mucoid / low-grade | rubric-text OOF; OAI m_acl_tear | bound +.0018 (mucoid) / +.0034 (low-grade) on OOF; OAI code-1 bound +.0039: DEAD |
| MCL | high-grade ACUTE tear only; low-grade sprain and chronic negative | grade 1 / sprain / edema only: 114 (6.6 %); grade 2 / partial 73; chronic / thickened / Pellegrini: 58 (3.3 %); bursitis 21 | .95-.99; chronic .86-.91; bursitis .24-.33 | .53-.55 | severity + chronic subclass | demote sprain / chronic | rubric-text OOF (22 positives); no OAI | bound +.0090 (grade 1) / +.0035 (chronic); T-STATE +.0011: DEAD |
| Medial / Lateral Meniscus | surface contact on 2 images, or truncated / diminutive / displaced; intrasubstance negative | degeneration-only 120 / 74; post-op 2 / 2; extrusion only 9 / 1 | .00-.04 (v2 .85-.88) | .01 | aligned in labels; model-level elevation of signal-only knees | none without a second view (OA-meniscus closed) | OAI T123 (.2 = signal) | OAI: signal-only vs normal AUC .69-.79; perfect-demotion bound +.045 / +.025 (c4D5) but no transform; T-STATE -.0015 / -.0005: DEAD |
| Medial OA | >= ~1 cm high-grade (>50 % depth) loss in the compartment | low-grade / mild only: 148 (8.5 %); high-grade / full / grade 3-4: 401 | low-grade .45-.47 | .11 | severity | severity AND (killed 10-20 on OAI, CLAIMED expert2) | OAI MOAKS | OAI partial-vs-full separation is the big gap (bound +.08 to +.13, overstated: g1 contains >50 % partials); no output-only view: DEAD here |
| Lateral OA | same, lateral compartment only; facet chondrosis is not lateral compartment | lateral PATELLAR facet only: 67 (3.9 %) | .13-.16 (other PF damage .09-.10) | .02 | small label leak; LARGER model leak | E-OA pw (in flight) | OOF on label-negative studies | facet-only lifts the Lateral OA output: AUC .745-.764 vs medial .670-.713: supports pw |
| PF OA | same, patella + trochlea | (as Medial OA) | | | severity | E-OA (in flight) | OAI | no new lever |
| Effusion | moderate or large only | small / trace / minimal / mild only: 607 (35 %); moderate 157; large 84; unstated 218 | small .95-.99 (es 1.00, tr .90) | .55 | severity (monotone) | T-VOL (R6 volume), T-STATE | OAI m_eff_wk 0-3; rubric-text OOF; report control | T-VOL helps at grade >= 1 only; T-STATE -.0023: DEAD |
| Synovitis | inflammation and thickening of the synovium (criterion unclear, 737155 unanswered) | explicit "synovitis" 278 (16 %); synovial thickening 36; Hoffa edema without "synov" 100 (5.8 %); moderate / large effusion without "synov" 149 (8.6 %); debris / hemarthrosis 19 | explicit .98-.99; Hoffa / effusion / debris 0.00 | none (experts BROADER) | definition gap | Syn <- Eff (E2, killed), state "1 - P(none)" | none (OAI Syn = Hoffa; OAI-free Syn = Eff) | no ruler: DEAD |
| Baker's | moderate or large, typical location | small only: 183 (10.5 %); moderate / large 70; ruptured 78 | small .97 | .44 | severity | T-VOL (bak volume), T-STATE | OAI popcys / WORMS size (n 35); rubric-text OOF | T-VOL negative everywhere; T-STATE +.0067 (P .84, 68 pos): DEAD / WEAK |
| Contusion | impact edema without a fracture line; degenerative edema negative | explicit 233; degenerative-context BME 172 (9.9 %); non-specific BME 381; fracture-associated BME only 43 | .90-.94; degenerative .04-.05; non-specific .09-.10 | .03 | aligned | none needed | rubric-text OOF | T-STATE -.0030: DEAD |
| Fracture | acute cortical break or fracture line | SIFK 55; stress 15; old / healed 6; avulsion 34; osteochondral no-fracture 44; acute / impaction 58 | SIFK .71-.75 (experts 5/6 +); old 1.00 (rare); OCD .00-.05 | .06-.07 | mostly aligned; the poor gold58 agreement is not a language issue | state "acute" map | rubric-text OOF; no OAI | T-STATE -.0042: DEAD |

Report-label "contam" for ACL / MCL / Effusion / Baker's is large, yet for those four labels the members' AUC at the rubric threshold
is HIGHER than at the report threshold (sec 0 item 1). The labels where the expert read is hard (Synovitis, Lateral OA, PF OA,
Fracture, Lateral Meniscus) are hard for non-language reasons: definition (Synovitis), compartment / image coupling (OA), image
agreement (Fracture), model ranking of degenerate menisci (LM).

## 3. Measurements

### 3.1 T-VOL: out = pct(score) + b * pct(R6 fluid volume), Effusion and Baker's

- **Mechanism tested.** The rubric counts only moderate / large fluid; every member's labels count small fluid (contam .55 / .44).
  R6 measures the fluid volume (`compact_infer` v4 `fluid_d2`: eff_sag_p99_open, bak_sag_p99_open, mL) and its stacker adds
  log1p(volume) with coefficients fitted on report labels (CLAIMED from the source in c4 cell 7). If the expert threshold needs
  more weight on volume, adding pct(volume) should help at the rubric cut and less at the report cut. The volume is available
  output-only in c4 (R6 worker writes `fluid_stacker.csv` next to `predictions.csv`; CLAIMED by analogy with the OAI and gold58
  runs of the same worker args).
- **Ruler 1, OAI expert MOAKS effusion (independent; box, n 1,178 graded knees of 1,704 R6 / S6fall / D5 common; volume finite .999;
  Spearman(volume, grade) .698)** (MEASURED `box_rubric.log` part A). AUC deltas at b .1 / .2 / .3 / .5:

  | target | c4D5 (D5 in the Raptor slot) | R6 | S6fall | D5 |
  |---|---|---|---|---|
  | grade >= 1 (report-like) | +.0034 / +.0055 / +.0069 / +.0073 | +.0041 / +.0068 / +.0089 / +.0104 | +.0047 / +.0077 | +.0047 / +.0079 |
  | **grade >= 2 (RUBRIC)** | +.0005 / **-.0000** (boot P .49) / -.0009 / -.0033 | +.0006 / +.0004 (P .63) / -.0003 / -.0023 | +.0021 / +.0029 | +.0017 / +.0021 |
  | grade 3 | -.0021 / -.0051 / -.0080 / -.0131 | -.0018 / -.0039 | +.0011 / -.0006 | -.0006 / -.0032 |

  Baker's: popcys present (n 430) -.0035 / -.0078 (c4D5), every model negative; WORMS size >= 2 (n 35, 3 positives) negative.
- **Ruler 2, report text re-thresholded at the rubric (EN, regex; 1 = moderate / large stated, 0 = none or small only; n 1,202 with
  OOF, 200 positives)** (MEASURED `local_fluid.log`): R7 (already volume-stacked) -.0003 / -.0015 (P .20) / -.0033; mean3
  (R7 + s1044 + s1045 rank mean) -.0004 / -.0021 (P .12); only heads WITHOUT the stacker gain (R7 binary head +.0028 / +.0034;
  S6 seeds +.0012 / +.0007 at b .1 / .2).
- **Control, report regime (v4 / JEV >= .5, n 3,467)**: mean3 +.0012 / +.0012 (P .88 at b .2); unstacked heads +.003 to +.007.
- **gold58 (consistency only)**: c4 Effusion -.0012 / +.0025 / -.0012 at b .1 / .2 / .3; Baker's -.0009 / -.0027.
- **Read.** Volume carries REPORT-threshold information (small vs none), not rubric-threshold information: the members already
  separate moderate+ from small (OAI grade >= 2 base AUC .958-.963; rubric-text base .94-.95). Opposite of the regime prediction.
  **DEAD** (expected hidden macro about 0 to -.0001, MODEL; P(>0) about .4, GUESS).

### 3.2 T-STATE on a new ruler: rubric-mapped heliosli state read-out vs the binary head

- **Weights**: copied unchanged from `trick/regime_work/m2_states.py` (written from the rubric on 10-19 before any number). Score =
  sum of state probabilities x {0, .5, 1}; mix = .5 pct(binary) + .5 pct(state score). R7 OOF only (R6's states are not dumped in c4
  or on OAI).
- **Ruler**: report text re-thresholded at the rubric, EN, non-gold, ambiguous cases excluded (MEASURED `local_states.log`).

  | label | rubric-text n / pos | binary AUC (rubric / report ruler) | mix delta on RUBRIC (P) | mix delta on REPORT v4 (P) |
  |---|---|---|---|---|
  | Effusion | 1,491 / 241 | .941 / .832 | -.0023 (.18) | -.0103 (.00) |
  | Baker's | 1,521 / 68 | .931 / .963 | +.0067 (.84) | -.0167 (.00) |
  | ACL | 1,640 / 204 | .960 / .860 | -.0010 (.27) | +.0182 (1.00) |
  | MCL | 1,625 / 22 | .949 / .843 | +.0011 (.80) | -.0035 (.11) |
  | Medial Meniscus | 1,708 / 665 | .923 / .945 | -.0015 (.10) | -.0003 (.35) |
  | Lateral Meniscus | 1,708 / 295 | .902 / .930 | -.0005 (.34) | -.0015 (.12) |
  | Contusion | 1,648 / 222 | .895 / .917 | -.0030 (.16) | -.0003 (.44) |
  | Fracture | 1,662 / 64 | .902 / .911 | -.0042 (.13) | -.0092 (.02) |

- **Read.** The predicted regime flip (gain on rubric, loss on report) appears only for Baker's and MCL, the two smallest positive
  counts; the large-n labels lose on the rubric ruler. Not deployable without a worker-argument change (R6 `--dump-state`) and a
  read on 10-22 at best. **DEAD** as a slot; Baker's / MCL WEAK (expected hidden about +.0001 macro if carried by R6 at its c4
  weight, MODEL; P(>0) about .55, GUESS).

### 3.3 Subclass headroom (how much a perfect "demote the rubric-negative subclass" lever could buy)

AUC(g2 vs g0 + g1) is the rubric AUC; bound = the same with every g1 forced below every g2.
- **OAI, c4D5** (MEASURED `box_rubric.log` part B): MM signal-only n 667: rubric .946, AUC(g1 vs g0) .718, bound +.045; LM: .952,
  .788, +.025; Effusion grade 1: .963, .833, +.035; ACL code 1: .994, .946, +.004; OA (g1 = loss without full thickness, which
  includes rubric-positive deep partials, so overstated): Medial +.084, Lateral +.125, PF +.133.
- **RSNA train OOF, mean3, rubric-text classes** (MEASURED `local_subclass.log`): ACL mucoid +.0018, ACL low-grade +.0034, MCL
  chronic +.0035, MCL grade 1 +.0090, Effusion small +.0315, Baker's small +.0275.
- **Read.** The non-monotone label-language subclasses (ACL mucoid, MCL chronic) are worth < .004 per label even with a perfect
  demotion. The large bounds (menisci, effusion, OA) are severity / ranking gaps inside one label; the only second views that
  could close them (member disagreement, other labels, volume, state heads) are measured dead (expert2 sec 3.1, census, sec 3.1,
  sec 3.2).

### 3.4 Facet confusion behind E-OA pw (support for an in-flight entry; NOT a new entry)

- Label-negative studies (v4 AND JEV Lateral OA < .5), EN non-gold: lateral-facet-only vs other PF chondral damage without a side
  word (MEASURED `local_facet.log`):
  - Lateral OA output: R7 AUC .764 (n 55 vs 318; mean pct .542 vs .318), s1044 .747, s1045 .745; boot sd .03, P(> .5) 1.00.
  - Medial OA output with the medial facet: .670 / .683 / .713.
- Label leak is small: Lateral OA labelled positive in .13-.16 of facet-only reports vs .09-.10 of other PF reports (MEASURED).
- Caveat: facet-specified reports may describe more severe PF disease than unspecified ones; the medial analogue controls for
  part of that, and the lateral excess (+.05 to +.09 AUC) is the side-specific part (MODEL).
- Use: if the A21 reads are close between pw and sym, this is a mechanism reason to prefer pw (GUESS; the reads and R1 decide).

## 4. Proposals (also returned as structured output)

| # | lever | ruler | measured | expected hidden macro | P(>0) | verdict |
|---|---|---|---|---|---|---|
| 1 | T-VOL Effusion: + b pct(R6 effusion volume) | OAI MOAKS grade (independent); rubric-text OOF; report control; gold58 check | OAI rubric cut c4D5 -.0000 (P .49) at b .2, R6 +.0004 (P .63); report-like cut +.0055; rubric-text mean3 -.0021 (P .12) | about 0 to -.0001 (MODEL) | about .4 (GUESS) | DEAD |
| 2 | T-VOL Baker's: + b pct(R6 Baker volume) | OAI popcys / WORMS size; rubric-text OOF | negative on every ruler (OAI c4D5 -.0078 at b .2) | < 0 | about .2 | DEAD |
| 3 | T-STATE: rubric-mapped state read-out mixed .5 with the binary head | rubric-text OOF (new); report control | 6 of 8 labels lose on the rubric ruler (Effusion -.0023, Fracture -.0042); Baker's +.0067 (P .84), MCL +.0011 (P .80) | about 0 overall; Baker's / MCL alone about +.0001 | about .45 / .55 | DEAD (Baker's / MCL WEAK) |
| 4 | Demote ACL mucoid / low-grade and MCL chronic / grade-1 subclasses | rubric-text OOF; OAI ACL code 1 | perfect-demotion bound +.0018 to +.0090 per label (~+.0002 to +.0008 macro) before any real transform | about 0 | about .5 | DEAD |
| 5 | Meniscal signal-only demotion (model-level, labels already aligned) | OAI T123 signal grade | bound +.045 / +.025 (MM / LM) but no second view survives | 0 (no transform) | n/a | DEAD |
| 6 | Synovitis broadening (expert broader than report) | none independent | census: Hoffa edema / big effusion / debris labelled 0 by every source | unknown | about .4 | DEAD (no ruler) |
| 7 | Facet confusion -> prefer E-OA pw over sym (in flight) | OOF label-negative studies (report regime, mechanism only) | Lateral OA AUC .745-.764 vs medial .670-.713 | tie-break only | n/a | PROMISING as support, no new entry |

## 5. What would change this

- A host answer on the Synovitis criterion (737155) or any expert-graded set with synovitis / MCL / fracture grades.
- R6 state read-outs on OAI (`--dump-state`, ~2.75 h GPU for 1,704 knees, CLAIMED r6_oai.md timings): would give T-STATE an expert
  ruler for menisci, effusion, OA. Not worth the GPU or the risk before 10-22 given sec 3.2.

## 6. Files (this folder; scripts are mine unless named otherwise; other files here belong to sibling tasks)

- `census_lang.py` / `.log` / `.csv`: language census and contamination (EN regex, competition data).
- `local_ml.py` / `.log`: Spanish / Turkish spot check.
- `local_fluid.py` / `.log`: T-VOL on rubric-text, report and gold58 rulers.
- `box_rubric.py` / `.log`: T-VOL on OAI (part A) and subclass headroom on OAI (part B); aggregates only.
- `box_struct2.py`: column names of the R6 OAI output files (aggregates only; output not saved).
- `local_states.py` / `.log`: T-STATE on the rubric-text ruler.
- `local_subclass.py` / `.log`: subclass headroom on train OOF and facet label leakage.
- `local_facet.py` / `.log`: model-level facet confusion.
- Box command form: `ssh training@192.168.10.30 'BOOT=1000 nice -n 5 ~/miniconda3/envs/rsna/bin/python -I -' < box_rubric.py |
  sed -E 's/oai_[0-9]+_[LR]/<OAI>/g'`.

## 7. Governance log

- **Slip (10-20 ~15:42 IST).** My first structure probe (`box_struct.py`) tried to skip id-like keys of R6's OAI `dump.json` with a
  "many dots" heuristic, which fits RSNA UIDs but not OAI keys (`oai_<number>_<side>`). It printed about 30 distinct OAI knee ids as dict
  keys into this session's transcript, with no grades, labels or per-knee values (only the key names `z`, `z_plain`,
  `z_mirror_raw` and list lengths). Nothing was written to any local file; the script is deleted so it cannot be reused. Every
  later box command printed only column names, counts and aggregates, and its output went through an id-redacting `sed`.
  `grep "oai_[0-9]"` over this folder finds nothing.
- No OAI row, id or per-knee value is in any file here. No box file was written. GPU not used. No process killed.
