# readtable.md: A22 read-decision table (10-21 ~08:40-09:20 IST, Claude, DAY track)

RESEARCH ONLY. Nothing was submitted, pushed, committed, versioned, downloaded, posted or voted. No dataset, no GPU job, no
scheduled-task edit, no box, no OAI data, no Kaggle or LB call. PREREG, HANDOFF, PROGRESS and the val / a20 / a21 / stack / private
folders were only read. Scripts ran with `python -B`, so no bytecode was written into those folders. New files are only `readtable*`
in this folder; the other files here belong to sibling tracks.

**Tags.** MEASURED = read or computed now, file named. CLAIMED = older note, named. MODEL = arithmetic on stated inputs
(`readtable_mc.py`). GUESS = judgement.
**Units.** E = private margin in 1e-4 macro over the heliosli copy block (the picks2 / dose.md ruler). Public-scale private is about
.95802 + E x 1e-4. "pts" = percentage points.
**Curves.** step = the 10-19 field curve (picks2 / dose.md). smooth = shakeup.md's fixed-margin curve (it includes the family shock).
Silver is quoted on both curves; bronze and r150 on the smooth curve. Every number is a scenario mix (S0 .6 / C+ .1 / C- .1 / SITE .2,
from dose.md).

## 0. Answer

1. **Display of r6pw20 does not change the A decision.**
   - P(.957) is only .006 (MODEL). When r6pw20 shows .957, B = R6, and that costs about 0 against keeping r6pw20 anyway (-0.4 step /
     +0.2 smooth silver). The rule is fine.
   - .959 vs .958 lifts every pair by about +1.6e-4 and +2.5 silver pts (step). The PF - R1 gap stays the same.
2. **Which entry becomes best-id is what matters. One signal is robust:**
   - **If pw30 stays best-id (P .14, range .12-.22 over 8 variants), A = pw20 beats A = pw30.** Gain +2.6 to +3.1 silver pts (step) and
     +0.9 to +1.6 (smooth), +2.2 to +2.6 bronze. It costs 1.6-2.5 pts of P(r<=150) and 0.6-0.9e-4 of E. This holds in 8/8 variants.
   - **If pw25 wins (P .13): pw20 is slightly better**: +1.1 to +1.2 step, +0.2 to +0.4 smooth, 8/8.
   - **If a stack wins (spl / swide / salt, P .73 base, .57-.79 over variants): keep R1.** PF is -1.1 to -2.7 smooth silver in the
     base model.
     - PF is clearly better on smooth silver only in the most skeptical stack world (lomm05 OAI-only, MCLmm 0): +0.3 to +0.8.
     - Bronze and step silver are near a tie (-1.4 to +1.7).
3. **Whole plan (8 variants, silver step / smooth):**
   - **A22 as written (R1):** .870-.900 / .754-.793.
   - **Private-first (A = pw20 fixed):** .863-.903 / .740-.781. That is +0.0 to +1.7 step and -1.5 to +0.7 smooth vs R1. With 3
     stacks entered, PF no longer dominates R1 the way PRIVATE_1020 found without stacks.
   - **Hybrid H2: A = pw20 if the best-id is pw30 or pw25, else A = the best-id (a stack).**
     - vs R1: +0.4 to +0.9 step and +0.2 to +0.4 smooth silver, +0.4 to +0.8 bronze, -0.4 to -0.7 r150. It is >= R1 on silver in 8/8
       variants.
     - vs PF: -0.8 to +0.5 step, -0.3 to +1.7 smooth.
   - **H1 (pw20 only if pw30 stays)** gets most of that: +0.3 to +0.7 step, +0.2 to +0.4 smooth vs R1.
4. **10-22: do nothing in every row, except an r6pw20 ERROR retry.**
   - No dose or stack refinement around any winner is worth a slot. Detail (MODEL):
     - The already-built recombinations (lomm05@pw30, smid, ssafe) cost up to -0.8 to -1.0 silver in exactly the rows where they would
       become A. They re-select on read noise already seen, at dose .3.
     - Unbuilt pw20 + lomm05 (+ MCLmm) is at most +0.3 step / +0.2 smooth, with E <= 0. That is noise.
     - A22 also caps the dose at .30, and any 10-22 entry enters R1.
   - **If r6pw20 ERRORs, one resubmit on 10-22 is the only valuable action.** B = R6 instead of r6pw20 costs -0.4 to -1.2 step and
     -0.9 to -1.7 smooth silver under R1 over the 8 variants (base -0.7 / -1.2; PF base -1.1 / -1.8).
     - A22 has no retry clause.
     - The tick window closes 10-22 05:20 IST, so a retry is a manual submit after 05:30 IST (last readable ~10:15 IST).
     - It needs Eesh's GO and an A22 addendum line.
5. **Eesh's decisions:**
   - **Default: keep A22 as written (R1). It is never the worst option and it is pre-registered.**
   - Optional: write H2 (or H1) into A22 as an addendum **before the first 10-21 read (~14:30 IST)**. Its gain is small (+0.2 to +0.9
     silver pts) but positive in every variant.
   - Choosing PF or R1 after seeing the reads, without a written rule, is the cherry-pick that R1 exists to prevent.
     - The A22 wording "Eesh may instead fix A = pw20 ... before selection" allows exactly that.
     - Better: fix the rule now (R1, H1, H2 or PF) and apply it mechanically at the reads.

## 1. State (MEASURED)

- A22 text: PREREG_10-15.md "AMENDMENT 22 (10-21 ~08:36 IST)".
  - Slate: r6pw20, pw25, spl, swide, salt.
  - A = API best-id at the last read before 10-22 ~18:00 IST (R1), or pw20 57425269 if Eesh chooses private-first.
  - B = r6pw20 if COMPLETE and >= .958, else R6 57248084.
  - No dose step above .30.
- GO file: `GO r6pw20 pw25 spl swide salt MAX=5 Eesh 10-21 08:30`.
- Tick log at 08:40 IST: r6pw20 and pw25 pushed at 08:37 (rc 0). Nothing submitted yet; the stacks are queued behind the 2-commit cap.
  The CLAIMED read window is ~14:30-17:00 IST.
- The tick handles commit ERROR (COMMIT_BAD), not a scoring ERROR after submit (`val/tick_a22.py` l.161).

## 2. Model (MODEL, `readtable_mc.py`)

- **Dose world:** private/dose_work/dose_mc.py `fit()` plus skep_dose.py's draw construction, same seeds.
  - Reproduces dose_base.log: R6pw20 display .006 / .504 / .491; public pw25 - pw30 +0.11.
  - Pairs: pw30 fixed .853 step / .748 smooth, pw25 fixed .875 / .765. These equal dose.md sec 0.4 and 5.1.
  - It covers pw20 / pw25 / pw30, R6 / r6pw20 (k transfer, display guard), our-side posterior, coupling scenarios and SITE.
- **Stacks:** spl = pw25 + lomm05 + I; swide = spl + MCLmm; salt = pw25 + MCLmm. Inputs from stack/skeptic/skep_value_mc.py:
  - lomm05 scenario prior; I N(+.16, .12); MCLmm skep2 (E -0.10, sd 2.24).
  - Read sd, public / private: lomm05 1.00 / .62, MCLmm 2.40 / 1.50. The MCL column is shared by swide and salt. The lomm05 read
    correlates -.2 with pw25's read.
- **Outcomes:**
  - W = argmax public over {pw30, pw25, spl, swide, salt} (= API best-id under R1).
  - D = r6pw20 display.
  - "A shows .960" uses the band position from the fit (x20 + gap).
- **Variants (8):**
  - base;
  - lskep (lomm05 N(0, .9));
  - mzero (MCLmm = 0);
  - lskepmzero;
  - worst (lomm05 OAI-only N(-.9, .9), MCLmm = 0);
  - mstack (builder's optimistic MCLmm prior);
  - flatU (flat dose prior, log-uniform s);
  - PAREA proxy.

  Summary: `readtable_summ.out`.

## 3. Read-decision table (base variant; brackets = range over 8 variants)

A (R1) = the best-id. B = r6pw20 for D .958 / .959. Columns: P | P(A shows .960) | pair R1 {A, B}: E, silver step / smooth, bronze,
r150 | pair PF {pw20, B}: same | PF - R1 silver step / smooth | 10-22.

| # | best-id W | D | P [range] | .960 | R1 pair: E, silver step/smooth, bronze, r150 | PF pair (A = pw20) | PF - R1 silver step / smooth | 10-22 |
|---|---|---|---|---|---|---|---|---|
| 1 | none (pw30) | .958 | .077 [.063-.121] | 0 | +16.3, .839/.731, .891, .424 | +15.7, .872/.748, .919, .408 | **+3.3 [+2.8,+3.4] / +1.7 [+1.0,+1.8]** | nothing |
| 2 | none (pw30) | .959 | .063 [.051-.100] | 0 | +18.1, .871/.769, .911, .470 | +17.3, .897/.782, .933, .450 | **+2.6 [+2.3,+2.7] / +1.3 [+0.8,+1.4]** | nothing |
| 3 | pw25 | .958 | .059 [.046-.100] | .37 | +17.1, .874/.763, .918, .445 | +16.6, .887/.767, .929, .430 | +1.3 [+1.2,+1.3] / +0.4 [+0.2,+0.4] | nothing |
| 4 | pw25 | .959 | .066 [.050-.111] | .47 | +18.7, .896/.794, .931, .486 | +18.1, .907/.797, .941, .471 | +1.1 [+1.0,+1.1] / +0.3 [+0.1,+0.3] | nothing |
| 5 | spl | .958 | .133 [.050-.138] | .69 | +17.0, .871/.759, .916, .441 | +15.7, .872/.748, .920, .408 | +0.1 [-0.2,+1.9] / -1.2 [-1.5,+1.0] | nothing |
| 6 | spl | .959 | .131 [.050-.131] | .78 | +18.5, .896/.792, .932, .481 | +17.2, .897/.781, .936, .448 | +0.1 [-0.1,+1.4] / -1.1 [-1.3,+0.6] | nothing |
| 7 | swide | .958 | .142 [.062-.193] | .92 | +17.9, .882/.775, .923, .463 | +15.7, .871/.748, .919, .408 | -1.0 [-1.5,+1.8] / -2.8 [-3.4,+0.9] | nothing |
| 8 | swide | .959 | .138 [.059-.187] | .95 | +19.4, .903/.804, .936, .502 | +17.2, .895/.779, .933, .447 | -0.8 [-1.2,+1.6] / -2.5 [-3.0,+0.7] | nothing |
| 9 | salt | .958 | .093 [.093-.170] | .82 | +18.0, .885/.779, .925, .467 | +16.3, .884/.763, .927, .425 | -0.1 [-0.4,+1.3] / -1.6 [-2.0,+0.4] | nothing |
| 10 | salt | .959 | .093 [.083-.171] | .87 | +19.5, .906/.808, .937, .507 | +17.9, .905/.793, .939, .466 | -0.1 [-0.2,+1.1] / -1.5 [-1.7,+0.2] | nothing |
| 11 | any | .957 (B = R6) | .006 [.004-.008] | | +17.8, .878/.769, .923, .460 | +16.0, .881/.754, .930, .412 | +0.3 / -1.5 | nothing |
| 12 | any | r6pw20 ERROR (B = R6) | GUESS ~.01-.02 | | +17.6, .877/.767, .924, .453 | +15.9, .877/.751, .927, .409 | 0.0 / -1.6 | **one r6pw20 resubmit after 05:30 IST 10-22** (needs GO + addendum) |

- **Row 11, if B = r6pw20 anyway:** R1 +17.9, .874 / .771 (-0.4 / +0.2 vs the rule). Per-W cells at .957 are each <= .002 and keep
  the sign of their .958 row.
- **Row 12 vs the complete plan** (R1 .884 / .779, PF .888 / .769): the ERROR costs -0.7 / -1.2 (R1) and -1.1 / -1.8 (PF) silver
  pts. Across the 8 variants the R1 cost is -0.4 to -1.2 step and -0.9 to -1.7 smooth.

### 3.1 By best-id only (any display; ranges over 8 variants; "n" = variants with PF >= R1)

| W | P [range] | P(.960) | R1 pair E / silver step / smooth | PF pair | PF - R1: silver step | smooth | bronze | r150 | E | n smooth / step |
|---|---|---|---|---|---|---|---|---|---|---|
| none (pw30) | .141 [.115-.223] | 0 | +17.09 / .853 / .748 | +16.40 / .883 / .763 | +2.6..+3.1 | +0.9..+1.6 | +2.2..+2.6 | -2.5..-1.6 | -0.93..-0.63 | 8/8, 8/8 |
| pw25 | .125 [.096-.212] | .42 | +17.95 / .886 / .779 | +17.35 / .898 / .783 | +1.1..+1.2 | +0.2..+0.4 | +1.0..+1.1 | -1.6..-1.4 | -0.60..-0.56 | 8/8, 8/8 |
| spl | .265 [.101-.265] | .73 | +17.75 / .884 / .775 | +16.47 / .885 / .764 | -0.2..+1.7 | -1.4..+0.8 | +0.2..+1.4 | -3.5..-0.9 | -1.31..-0.34 | 1/8, 7/8 |
| swide | .281 [.122-.383] | .93 | +18.59 / .892 / .790 | +16.44 / .883 / .763 | -1.4..+1.7 | -3.2..+0.8 | -0.6..+1.5 | -6.2..-1.0 | -2.43..-0.38 | 1/8, 3/8 |
| salt | .188 [.184-.342] | .84 | +18.75 / .896 / .793 | +17.10 / .894 / .778 | -0.3..+1.2 | -1.9..+0.3 | +0.1..+1.1 | -4.4..-1.6 | -1.68..-0.61 | 3/8, 4/8 |

E of A alone (base): pw30 row R1 +15.18 vs PF +15.02; pw25 +16.54 vs +15.99; spl +16.53 vs +15.09; swide +17.50 vs +15.05; salt +17.54
vs +15.73. E always favours R1; P(silver) on the step curve favours PF in the pw30 / pw25 rows because pw30 and pw25 carry the wider
dose tail (dose.md sec 5.0 item 4: 63 % population spread, 37 % private noise).

### 3.2 Whole-plan policies (silver step / smooth; MEASURED `readtable_summ.out`)

| variant | R1 (A22) | PF (pw20 fixed) | H1 (pw20 if pw30 stays) | H2 (pw20 if pw30 or pw25) | pw30 fixed | pw25 fixed |
|---|---|---|---|---|---|---|
| base | .884 / .779 | .888 / .769 | .888 / .782 | **.890 / .782** | .853 / .748 | .875 / .765 |
| lskep | .880 / .774 | .888 / .769 | .885 / .776 | .887 / .777 | .853 / .748 | .875 / .765 |
| mzero | .879 / .772 | .888 / .769 | .883 / .774 | .884 / .774 | .853 / .748 | .875 / .765 |
| lskepmzero | .874 / .766 | .888 / .769 | .879 / .769 | .881 / .769 | .853 / .748 | .875 / .765 |
| worst | .870 / .761 | **.888 / .769** | .877 / .765 | .880 / .766 | .853 / .748 | .875 / .765 |
| mstack | .887 / .783 | .888 / .769 | .891 / .785 | **.892 / .785** | .853 / .748 | .875 / .765 |
| flatU | .862 / .754 | .863 / .740 | .866 / .756 | **.867 / .756** | .831 / .725 | .851 / .738 |
| PAREA | .900 / .793 | .903 / .781 | .904 / .794 | **.905 / .795** | .872 / .764 | .891 / .778 |

E (base): R1 +18.11, PF +16.68, H1 +18.01, H2 +17.94. r150 smooth: .470 / .434 / .468 / .466. Bronze: .924 / .929 / .927 / .928.

## 4. 10-22 options under R1 (MODEL; change vs "nothing" in that best-id group: silver step / smooth pts, dE e-4, P(option becomes A))

| best-id | lomm05@pw30 (built) | smid (built) | ssafe (built) | pl20 = pw20 + lomm05 (unbuilt) | wide20 = pl20 + MCLmm (unbuilt) |
|---|---|---|---|---|---|
| none (pw30) | +0.1 / +0.1, +0.06, .23 | 0.0 / 0.0, .07 | 0.0 / 0.0, .12 | +0.3 / +0.2, -0.00, .06 | +0.1 / 0.0, .02 |
| pw25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| spl | **-0.8 / -0.6**, -0.06, .31 | -0.1 / -0.1 | 0.0 | +0.2 / +0.1, -0.05, .13 | 0.0 |
| swide | -0.1 / -0.1 | **-0.9 / -0.8**, -0.08, .37 | -0.2 / -0.2, .10 | 0.0 | +0.2 / +0.1, -0.06, .15 |
| salt | 0.0 | -0.3 / -0.2, .11 | **-0.8 / -0.6**, +0.02, .38 | 0.0 | 0.0 |

- Every variant agrees: the best option anywhere is +0.3 / +0.2 (unbuilt, E <= 0), and the worst is -1.0 / -0.8 (smid when swide
  wins).
- Under PF or H1 / H2, any 10-22 A probe has no role for A at all.
- So: **nothing on 10-22 in every row**, except row 12. An unbuilt entry would also need a build, a gate, an amendment and Eesh's GO.
- An ERROR on a single A probe (pw25 or a stack) is not worth a retry: the other dose-.25 entries cover it. GUESS, not modelled.

## 5. Day procedure (GUESS on MODEL)

1. Now to ~14:30 IST: Eesh confirms the A rule.
   - Default: R1 as written.
   - Optional addendum to A22 before any 10-21 score, for example: "A = pw20 57425269 if the API best-id at the last read is pw30
     57425970 or pw25; otherwise A = the API best-id". That is H2; H1 drops "or pw25". B is unchanged.
2. Reads (~14:30-17:00 IST): note each entry's display and the API best-id (fetch_api_lb.py). Apply the fixed rule mechanically.
3. If r6pw20 shows ERROR: draft the addendum ("one resubmit of rsna-sub-v224r6flips6fpw20j on 10-22; B = that resubmit if COMPLETE
   and >= .958, else R6") and get Eesh's GO. Submit manually after 05:30 IST 10-22, no later than ~10:15 IST.
4. Otherwise submit nothing on 10-22.
5. Eesh selects A and B by hand on Kaggle before 10-23 05:29 IST (10-22 23:59 UTC). Auto-selection would pick two near-twins.

## 6. Caveats

- **One dose family.** dose_mc (OAI-proxy D5 curves fitted to 5 public reads). The PF > R1 signal in the pw30 / pw25 rows is the
  dose-tail structure that dose.md sec 5 checked against an RSNA train bootstrap (CLAIMED). Its size depends on the field curve (step
  vs smooth), so both are quoted.
- **Stacks:**
  - lomm05 and MCLmm rest on gold58 nominations (9 MCL+ knees; CLAIMED stack/build.md sec 8.1).
  - The model treats public and private as one population for both, including SITE. That is optimistic for "stack wins -> keep R1".
    The worst / lskepmzero rows show the downside.
  - The R1 value of a stack is mostly selection on prior variance (skeptic sec 8.5).
- **10-22 options:** I(pw20, lomm05) = 2 x I(pw25, lomm05) and the cross-base read correlation .9 are GUESS. They only affect sec 4.
- **Probabilities are MODEL.** The real reads will replace W and D. The table is conditional on them, so it stays usable whatever the
  read odds were.
- **Not modelled:**
  - teams copying E-OA;
  - host rulings;
  - the family shock beyond the smooth curve (about +-10 silver pts, CLAIMED shakeup.md), which no pick hedges;
  - ERROR probability per cell.
- No live LB or forum read in this track.

## 7. Files (this folder)

- `readtable_mc.py`: model (imports private/dose_work/dose_mc.py; exec's stack/skeptic/skep_value_mc.py up to its report). Args:
  lomm=, mcl=, i20=, rL=, plus dose_mc args.
- `readtable_<tag>.log` / `.json` for base, lskep, mzero, lskepmzero, worst, mstack, flatU, PAREA.
- `readtable_summ.py` -> `readtable_summ.out` (variant summary, H1 / H2, ERROR rows, 10-22 options).

## 8. SKEPTIC review of this table and plan1022.md (10-21 ~10:20 IST, Claude, SKEPTIC track)

RESEARCH ONLY. Nothing was submitted, pushed, committed, built, downloaded, posted or voted. No box, no OAI data, no GPU, no
scheduled-task change. PREREG, HANDOFF, PROGRESS and the val / a20 / a21 / stack / private folders were only read. One read-only Kaggle
API submission listing (10:07 IST). New files: `skeptic_wc*` in this folder. Sections 0-7 above are unchanged.

**State (MEASURED, API list 10:07 IST):** all 5 A22 entries PENDING (r6pw20 57463719, pw25 57463742, spl 57464296, swide 57464326,
salt 57464911). The list gives public_score to 3 decimals only ('0.959'), so inside one display step only the LB best-id ranks entries.

### 8.0 Verdicts (MODEL `skeptic_wc_summ.out`, 14 variants unless tagged)

| # | question | verdict | why |
|---|---|---|---|
| 1 | Is R1 over 5 near-identical entries a winner's-curse trap? | **Real curse, bounded price. KEEP R1 inside the stack classes; CHANGE it in the pw30 / pw25 classes (item 2)** | The R1 winner keeps .74 (base) of its public lead over pw30 on private and .49 of its lead over pw25; skeptic variants keep .21-.67 and -.21-.40. P(A_R1 privately below pw25) .25 base, .28-.45 skeptic. Versus a fixed pw25: E -0.07 to -0.26e-4 and silver -0.3 to -0.7 step / -0.2 to -0.6 smooth in the 4 null / negative-component worlds; +0.1 to +1.3e-4 elsewhere |
| 2 | A rule | **CHANGE: write H2 into A22 before the first read (~14:30 IST)**, and retire the "Eesh may instead fix A = pw20 ... before selection" option in the same line | H2 >= R1 in 14 / 14: +0.5 to +1.0 step, +0.2 to +0.5 smooth silver, bronze +0.4 to +0.9, r150 -0.3 to -0.7, E(A) -0.07 to -0.13. The pw30-class gain (PF - R1 +2.8 to +3.1 / +1.4 to +1.7) and the pw25-class gain (+1.1 to +1.2 / +0.2 to +0.4) do not move in any variant. H2s (also pw20 when a stack wins without .960) adds -0.1 to +0.2: skip |
| 3 | PF (A = pw20 fixed) instead of H2? | **Defensible, not the default** | PF - H2: -0.4 to +0.1 step / -1.7 to -0.8 smooth in the 5 as-built worlds; +0.4 to +1.0 / -0.5 to +0.5 in the 9 skeptic worlds. Equal group weight: PF +0.3 step, H2 +0.5 smooth, a tie (weights GUESS). Fix one before the reads; choosing after them is the fork |
| 4 | B = r6pw20 in every outcome? | **KEEP** | In every best-id class (plus stack&.960 / stack&no960) and all 14 variants no alternative beats the rule. R6: -0.4 to -1.0 step / -0.8 to -1.8 smooth; HW: worse than R6; pw25: -2.1 to -2.9. pw20 only ties in the pw30 class (+0.0 / -0.4), where H2 makes pw20 = A anyway. B = the A22 best-id next to A = pw20 gains smooth only in the swide class under as-built priors (up to +1.4) and loses step everywhere. The .957 -> R6 clause stays (row 11: -0.4 / +0.2) |
| 5 | 10-22 A-eligible refinements (plan1022 P1: spl08 / swide08 / pw30 twins) | **CHANGE: do not enter them as A-eligible** | R1+P1 - R1, ALL: -0.12 to +0.20e-4, -0.2 to +0.2 step, -0.2 to +0.3 smooth. In their own classes (spl / swide): -0.39 to +0.35e-4, -0.6 to +0.5 silver. Negative in 6 / 14, at most +0.05e-4 in 4 more. It doubles the gold58 bet A22 already makes and re-opens A on a read that lands 11:00-12:30 IST 10-22 |
| 6 | Eesh's "3 public / 2 private" applied to 10-22 | **KEEP-optional, in this form: up to 3 PUBLIC-ONLY slots; the private slots carry only an r6pw20 ERROR retry, else stay empty** | Public-only entries are private-neutral by construction (plan1022 P4, MODEL: P(.960) +4.2 to +6.0 pts, about 2-3 public places, no medal value). No private-push entry with robust positive value exists besides the retry. Conditions in 8.4 |
| 7 | r6pw20 ERROR retry | **KEEP** (needs GO + addendum line) | B = R6 instead of r6pw20: ALL class base -0.7 / -1.3 (rhoS0 -0.8 / -1.3); no class or variant better than -0.4 / -0.8. The only 10-22 action with material private value |
| 8 | Stack ERROR retries (plan1022 order spl > swide > salt > pw25) | **CHANGE: public-only, or skip** | Same selection-on-prior bet (about 0 in the skeptic worlds). Once 10-22 entries score, its rank vs the A22 winner cannot be read (3-decimal list). Under H2 the class comes from the completed A22 entries |

Draft H2 line (NOT written; main appends it only on Eesh's GO, before any A22 score): "A = pw20 57425269 if the API best-id after the
A22 reads is pw30 57425970 or pw25 57463742; otherwise A = that best-id. This replaces the private-first option. B unchanged."

### 8.1 The curse in numbers (MODEL; e-4; ret = private / public lead of the R1 winner)

| variant | knob | lead over pw30 pub / priv (ret) | lead over pw25 pub / priv (ret) | P(A < pw25) | E(A_R1) - E(pw25 fixed) | R1 - pw25 fixed, silver step / smooth |
|---|---|---|---|---|---|---|
| base | readtable base | +2.27 / +1.68 (.74) | +2.16 / +1.05 (.49) | .25 | +1.05 | +0.9 / +1.5 |
| mstack | MCLmm builder prior | +2.59 / +1.91 (.74) | +2.48 / +1.28 (.52) | .23 | +1.28 | +1.2 / +1.8 |
| rhoS0 | SITE: component private effect independent of public | +2.27 / +1.49 (.66) | +2.16 / +0.86 (.40) | .28 | +0.86 | +0.7 / +1.2 |
| nf15 | component read noise x1.5 | +2.79 / +1.50 (.54) | +2.68 / +0.87 (.32) | .30 | +0.87 | +0.7 / +1.2 |
| lskep | lomm05 N(0, .9) | +1.99 / +1.33 (.67) | +1.88 / +0.69 (.37) | .30 | +0.69 | +0.5 / +0.9 |
| rhoA5 | public-private component corr .5 everywhere | +2.27 / +1.10 (.49) | +2.16 / +0.47 (.22) | .35 | +0.47 | +0.3 / +0.6 |
| pf5 | component prior sd x.5 | +1.86 / +1.04 (.56) | +1.75 / +0.41 (.23) | .32 | +0.41 | +0.2 / +0.5 |
| noise | rhoS0 + nf15 + pf5 | +2.47 / +0.91 (.37) | +2.36 / +0.28 (.12) | .37 | +0.28 | +0.1 / +0.4 |
| rhoA0 | corr 0 everywhere | +2.27 / +0.72 (.32) | +2.16 / +0.08 (.04) | .41 | +0.08 | -0.1 / +0.1 |
| skepall | lskep + MCLmm 0 + rhoS0 + nf15 | +2.33 / +0.72 (.31) | +2.22 / +0.09 (.04) | .40 | +0.09 | -0.1 / +0.1 |
| pure | L = M = I = 0 (stacks = pw25 + read noise) | +1.57 / +0.56 (.36) | +1.46 / -0.07 | .44 | -0.07 | -0.3 / -0.2 |
| worst | lomm05 OAI-only N(-.9, .9), MCLmm 0 | +1.40 / +0.51 (.36) | +1.29 / -0.13 | .42 | -0.13 | -0.5 / -0.3 |
| worstsplit | worst + rhoS0 + nf15 | +2.00 / +0.42 (.21) | +1.89 / -0.21 | .44 | -0.21 | -0.6 / -0.4 |
| pureneg | L = -0.9 fixed, M = I = 0 | +1.31 / +0.38 (.29) | +1.20 / -0.26 | .45 | -0.26 | -0.7 / -0.6 |

Why the price stays small (MODEL):
- Every stack sits on pw25, so a wrong stack pick costs only that component's true effect.
- Four pw25-based entries make R1 land on the pw25 family more often than R1 over {pw30, pw25} alone, and the dose family rates pw25
  above pw30. R1 beats that two-entry R1 by +0.3 to +2.2 step / +0.1 to +2.5 smooth in 14 / 14, the pure-noise world included.
- The real trap is the dose class: when nothing beats pw30, R1 keeps the higher dose on public noise. H2 removes exactly that.

### 8.2 Knobs added (`skeptic_wc.py`; base reproduces readtable_base.log to the digit: R1 +18.108 / .884 / .779, PF +16.680 / .887 / .769)

- rhoS / rhoA: correlation of each component's private population effect (lomm05, MCLmm, I) with its public one, in the SITE
  scenario only / everywhere. readtable and plan1022 both assume 1. GUESS knob for "public 390 and private 910 are not one population"
  (P(site split) .2-.3, CLAIMED PRIVATE_1020).
- nf: component read-noise sds (public and private) x factor. pf: component prior spread x factor around its mean.
- 10-22 P1 inside the same world (spl08 / swide08 on plan1022's kap curve; pw30 twins = readtable's lomm05 / smid / ssafe). Base
  in-class spl08 +0.35e-4 matches plan1022's +0.34.

### 8.3 Per-class view of the stack classes (PF - R1, silver step / smooth)

- As-built priors (base, rhoS0, nf15, lskep, mstack): stack class -0.6 to +0.1 / -2.1 to -1.2. Keep R1 there.
- Skeptic worlds (rhoA0, skepall, worst, worstsplit, pure, pureneg): +1.1 to +1.7 / +0.1 to +0.9. Here R1 is the trap.
- Nothing observable at the read separates the two (a .960 from a stack says almost nothing about MCLmm, stack/build.md 8.4). So the
  stack-class choice is a prior bet, and H2 keeps the pre-registered side of it.

### 8.4 Conditions for a 10-22 public push (only if Eesh wants it)

1. Amendment 23, before the first A22 read: "A is fixed from the A22 reads by H2 (or PF). 10-22 entries are public-only and never A
   or B." Under H2 with no A-eligible 10-22 entry, "the last read before 10-22 ~18:00 IST" equals the A22 read, so nothing else changes.
2. Read A before any 10-22 entry scores. Once a twin scores, the LB best-id may be the twin, and the 3-decimal submission list
   (MEASURED) cannot recover the A22 order. If an A22 entry is still PENDING at 05:30 IST 10-22, hold the public-only submits until it
   is COMPLETE.
3. Built and gated entries only: p30L / ssafe / smid. Do not build spl08 / swide08 for a slot with zero private value.
4. 05:31 IST order: r6pw20 retry (if ERROR) first, then public-only entries.
5. Eesh selects A and B by hand with both submission ids written down. Auto-selection takes the top-2 public, which with twins in are
   likely two pw30 / pw25-family entries and no B hedge. CLAIMED (memory): a wrong-notebook selection already cost a past competition.

### 8.5 Corrections to the two documents

- This file, sec 0 item 5 "Optional: H2": upgrade to the default (8.0 item 2).
- This file, sec 0 item 4 ("nothing on 10-22 except the retry"): still right for A-eligible entries. Public-only entries are a separate,
  private-neutral choice.
- plan1022 sec 0 item 1, "never negative on private in any variant": false in 6 / 14 variants here. It holds only inside the priors it
  inherited (public and private one population, gold58-based components).
- plan1022 sec 0 item 4: keep r6pw20 first; drop the A-eligible stack retries.
- plan1022 P5: right in substance; it needs the read-A-before-10-22-scores mechanics (8.4 item 2), which neither document had.

### 8.6 Caveats

- One dose family (dose_mc, fitted to 5 public reads). My knobs stress only the stack components. The pw30 / pw25-class H2 signal is
  the dose family's own; its only stress is flatU / PAREA (sec 3.2), where H2 also leads.
- The 14 variants are unweighted. How much weight null / negative-component worlds deserve is a judgement (GUESS 30-50 %).
- The family shock of about +-10 silver pts (CLAIMED shakeup.md) dwarfs every difference here (all about 1 pt or less outside the
  pw30 class).
- The rho knob is ad hoc. A real site split would also move the dose family, which SITE (weight .2) already does.

### 8.7 Files (this folder)

- `skeptic_wc.py`: model (readtable_mc.py sections 1-2 verbatim plus the knobs). Args as readtable_mc.py plus rhoS=, rhoA=, nf=, pf=.
- `skeptic_wc_<tag>.log / .json` for base, rhoS0, rhoA5, rhoA0, nf15, pf5, noise, lskep, skepall, mstack, worst, worstsplit, pure,
  pureneg; runners `skeptic_wc_run.sh`, `skeptic_wc_run2.sh` (+ logs).
- `skeptic_wc_summ.py` -> `skeptic_wc_summ.out`: curse table, policies, per-class PF / H2s, 10-22 P1, B check.
