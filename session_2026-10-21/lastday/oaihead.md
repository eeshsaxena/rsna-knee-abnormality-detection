# oaihead.md: expert-trained (OAI) OA head as a 10-22 lever (10-21 ~19:30 IST, Claude, TRACK expert-trained OA head)

RESEARCH + LOCAL BUILDS ONLY. Nothing submitted, pushed, committed, versioned, created or downloaded; no scheduled task touched; PREREG /
HANDOFF / PROGRESS / live folders only read. Box: 3 CPU scripts piped over ssh stdin (`nice -n 19`, 2 threads, GPU untouched), each
printing aggregates only (counts, id SHAPES with digits masked, AUCs, deltas). No OAI CSV or receipt was headed or catted. OAI id scan
of this track's files (`oai_[0-9]{3,}`): 0 hits. One read-only `kaggle datasets list --mine`.
**Tags:** MEASURED (computed now, file named), CLAIMED (older note, named), MODEL (arithmetic on stated inputs), GUESS (judgement).
**Units:** OA-mean = mean AUC over Medial / Lateral / PF OA; macro delta = OA-mean delta x 3/12; e-4 = 1e-4 macro AUC. gold58 = INFO only.

## 0. Verdict

**Do not build or enter an OAI-head entry for 10-22. Verdict: DEAD for 10-22 (evidence WEAK, feasibility and risk poor).**

1. **The requested OAI ruler does not exist (MEASURED `box_count.log`).** Every OAI knee with images is a TRAINING row of every
   OAI-trained model: `folds5_oai.csv` puts all 2,581 OAI V00 knees in fold 9 (never validation), and the OOF files of all 21 OAI-trained
   runs hold 0 OAI rows (`box_inv.log`). The served R6 (1,704), S6fall (2,399) and D5 (2,581) OAI sets are 100 % inside the V00
   training union. The only MOAKS knees outside every union (99) have no images or predictions (raw tarballs pruned, CLAIMED 10-20).
   Follow-up visits of training knees are not held out (same knee). So no held-out OAI expert test of DT / DO is possible, with or
   without the coupling-matched reweighting. Retraining with OAI cross-fitting would need the GPU, which is not available.
2. **Substitute rulers (all RSNA, no OAI rows; `box_oaihead.log`, B 2,000 gold58 / 300 OOF):**
   - gold58 (expert labels, the real situation: swide + DT): DT's OA at w .1 before E-OA is +.0009 OA-mean (= +2.2e-4 macro, sd 3e-4,
     **P(>0) .53**). w .15: -.0007. w .2: -.0068. w .3: -.0128. Every post-chain add is negative (w .1: -.0016, P .13). OAI-free placebos
     (D, E, E2, C) behave the same: all negative at w >= .15.
   - The OAI-specific part (DT minus its OAI-free twin D at the same w) leans positive: pre w .1 +.0011 (P .64), w .15 +.0039 (P .90),
     post about 0. So OAI training adds OA information, but DT is far weaker than the c4 members on expert OA, so blending it in at a
     weight big enough to move ranks costs as much as it adds.
   - RSNA OOF (report labels, n 4,333) on an OAI-free TTR stand-in base: DT pre w .1 +.0012 (R0) / +.0013 (R1 gold74-coupling
     reweight), P 1.00; placebo D +.0001 / -.0000. Real but not transferable: on OOF DT is STRONGER than the stand-in (+.0067 OA-mean);
     on gold58 swide is .037 above DT. There is no OOF for c4's members (R6 / Raptor / S6fall saw all of RSNA train).
3. **Value (MODEL / GUESS):** swide + DT OA (pre, w .1): E private about +0.3e-4 [-2, +2], P(true gain > 0) ~.55. After P(entry
   clean with DT used) ~.45 and the 3-dp public selection coin flip, the expected change to pick A's private score is about 0 to +0.1e-4.
   Compare MOACF(swide) +0.26e-4 (CLAIMED LEVERS_1021) with no new stage.
4. **Feasibility by ~09:00 IST 10-22:** weights YES (`eeshsaxena/rsna-knee-cnxt-fleet-k14w` exists, 523.6 MB, 2026-10-12, DT 5 ckpts,
   K14CLEAN 5/5; MEASURED listing, CLAIMED scan). Stage YES in pieces (HF-DT20 10-12 ran F = DT alone, HCHECK 16/16; V224-F fleet cells
   with K12-MERGE k11w + k14w; CLAIMED). Kernel NO: no c4 / swide kernel with a DT stage exists; needs a new build + sim + GPU commit
   + gate + registry + A23 + Eesh's GO, about 4-5 h of main-session work tonight (GUESS). Runtime: DT-only stage ~0.35-0.5 h central,
   ~0.7-1.2 h on a 10-21-type day (MODEL); fits after S6F in normal sessions, but every 10-22 c4-class entry already has P(clean) ~.55
   (CLAIMED CANARY_1021). DO has no dataset (k12w never created, K12 gate failed; CLAIMED K12_RESULT).
5. **Rules and selection:** the drafted A22 addendum says a 10-22 entry can never be A (CLAIMED CANARY_1021); an OAI-head entry has
   value only as an A-candidate, and a ~0.3e-4 effect is invisible in a 3-dp public read (paired sd ~1.8e-4, CLAIMED LEVERS_1021).
   External data: our own OAI-trained weights move pick A from "public OAI checkpoint (Raptor)" to "we trained on OAI ourselves",
   which is the distinction forum 750236 asks about. No host ruling (CLAIMED FORUM_10-19). Small, unquantified extra exposure (GUESS).

## 1. Track 1: held-out OAI knees for the OAI-trained models (box, aggregates only)

| item | result | source |
|---|---|---|
| `union_t123v00/folds5_oai.csv` | 6,988 rows; RSNA folds 0-4 = 867 / 867 / 867 / 866 / 866, gold74 = -1 (74); **OAI = fold 9 for all 2,581** | MEASURED `box_inv.log` |
| OOF files of OAI-trained runs (T12v00 f0-4, T123v00 f0-4 s1043, T123 s1044 f0-2, PL f0-2, swinT f0, r50rad f0, allv f0, 288 T12 f0) | 867 / 866 rows each, **0 OAI rows** | MEASURED `box_inv.log` |
| MOAKS label-source knees | 2,680; 2,581 in the V00 unions; **99 in no union** | MEASURED `box_count.log` |
| all-visit unions (T12allv / T123allv / T1allv) | 9,126 / 9,126 / 8,659 OAI rows = the same 2,581 / 2,399 knees at more visits | MEASURED `box_count.log` |
| served R6 / S6fall / D5 OAI sets | 1,704 / 2,399 / 2,581 knees, **all inside the V00 union**; 0 outside any union | MEASURED `box_count.log` |

So the c4D5 proxy + E-OA + "add the OAI-trained OA head" test on held-out OAI expert labels (P_FT / P_AREA), with or without the
coupling-matched reweighting, cannot be run: the OAI-trained head would be scored on its own training knees. The 99 unused knees have
no image data on the box (no rows in any served or teacher set; V00 raw tarballs pruned 10-20, CLAIMED r6_oai.md).

## 2. Track 2: OAI-trained OA heads vs c4 members on gold58 / gold74 (MEASURED `box_oaihead.log` G1 / G1b)

Proxy check first (MEASURED `local_proxy_check.log`): the closed form c4 stage-3 table -> .8 / .2 S6fall -> pw E-OA -> lomm05 -> MCLmm
equals the REAL cell-18 replay of the c4 (sha b47c29c3), pw25 (64ed7eba) and swide (c9801a80) notebooks on gold58 (max |diff| 2.9e-8,
ranks identical in all 12 labels). PROXY_CHECK PASS.

| head (gold58, n 58; OA positives 15 / 11 / 21) | OA AUC M / L / PF | OA-mean | model OA rho(ns) mean | gold74 OA-mean |
|---|---|---|---|---|
| R6 (exact serve) | .991 / .865 / .916 | **.9239** | .778 | |
| S6fall | .981 / .851 / .902 | .9115 | .845 | |
| Raptor (.75 r384 + .25 r224; OAI-trained, public) | .981 / .781 / .862 | .8747 | .797 | |
| c4 (.8 V + .2 S6fall) | .989 / .843 / .901 | .9111 | .813 | |
| pw25 | .988 / .855 / .922 | .9213 | .615 | |
| **swide (A)** | .988 / .861 / .922 | **.9233** | .609 | |
| DT (OAI T123, 5f; in k14w) | .966 / .814 / .878 | .8860 | .773 | .8903 |
| DO (OAI T12, 5f; no dataset) | .969 / .830 / .873 | .8905 | .760 | .8974 |
| D (OAI-free twin of DT / DO, 5f) | .971 / .818 / .853 | .8807 | .793 | .8859 |
| E / E2 / C (OAI-free, 5f) | | .8728 / .8763 / .8858 | .796-.798 | .8819 / .8815 / .8879 |
| DT0 / PL0 / SW0 / AV0 (OAI, f0 only) | | .8938 / .8923 / .8783 / .8957 | .734-.764 | .8970 / .8986 / .8876 / .9020 |

- OAI training helps the TTR head on expert OA labels: DT - D +.005 (gold58) / +.004 (gold74); DO - D +.010 / +.012 (MEASURED). This
  matches K12 / K14 (PF OA +.019, Lat OA +.012 on gold58; OOF PF OA +.0032, Lat OA +.0018, CLAIMED).
- But every TTR head (OAI or not) is .03-.04 OA-mean below R6 and swide on gold58. Only Raptor is as weak, and c4 already carries
  Raptor (an OAI-trained OA head) at .40 / .16 / .40 of the trio (CLAIMED r6_oai.md 11.2).
- OAI training lowers trio coupling only a little: DT .773 vs D .793 (gold58), .806 vs .826 (OOF). E-OA moves swide's coupling from
  .813 to .609. The expert-regime decoupling is already delivered by E-OA; an OAI head adds little of it.

## 3. Track 1 substitute: adding the OA head to swide (MEASURED `box_oaihead.log` G2-G5, O1-O3)

`pre` = the head enters the OA trio at rank weight w after the .8 / .2 S6fall blend and BEFORE pw E-OA (a .25) and lomm05, as a
cell-18 edit would; `post` = after the whole chain. Paired study bootstrap; P = P(delta > 0).

**gold58 vs swide (OA-mean delta; INFO only)**

| head | mode | w .05 | w .10 | w .15 | w .20 | w .30 |
|---|---|---|---|---|---|---|
| DT 5f | pre | .0000 | **+.0009 (P .53)** | -.0007 (P .44) | -.0068 (P .09) | -.0128 (P .03) |
| DT 5f | post | .0000 | -.0016 (P .13) | -.0041 (P .08) | -.0055 (P .09) | -.0080 (P .08) |
| DO 5f | pre | -.0006 | +.0009 (P .60) | -.0012 | -.0069 | -.0076 |
| DT+DO 10 | pre | .0000 | +.0007 (P .60) | -.0010 | -.0062 | -.0110 |
| D (placebo) | pre | .0000 | -.0002 (P .43) | -.0046 | -.0061 | -.0115 |
| E / E2 / C (placebos) | pre | -.0006 / -.0006 / .0000 | -.0011 / -.0001 / +.0004 | -.0037 / -.0055 / -.0036 | -.0123 / -.0087 / -.0049 | all < -.0075 |
| PL0 / SW0 / AV0 / DT0 (f0) | pre | | -.0015 / -.0037 / +.0002 / -.0002 | all < -.0034 | all < -.0046 | all < -.0061 |

- At w .1 pre the trio barely moves on n 58 (DT: M 0, L 0, PF +.0026 = one rank pair). w >= .15 is negative for every head, OAI or not.
- OAI-specific part, X - D at the same w (paired): DT pre w .1 +.0011 (P .64), w .15 +.0039 (P .90), w .2 -.0007, w .3 -.0013; post
  -.0004..+.0010; DO pre +.0011 / +.0033 / -.0008 / +.0039; DT0 - D0 (f0 pair) pre +.0021 / +.0040 / +.0046 / -.0006. Positive lean,
  never clear.
- G5 control: the same DT add on the weak OAI-free stand-in base (C + E | .2 E2 -> pw25 -> lomm05; gold58 OA-mean .8839) gives +.0011
  at w .1-.15 pre, about the same as on swide. The gold58 add signal is about +.001 OA-mean at w .1 whatever the base, and sits inside
  noise (sd .0012-.0015).

**RSNA OOF, v4 report labels, n 4,333 (gold74 excluded); R1 = rows reweighted to the gold74 OA-trio joint (ESS 4,052)**

| added head (stand-in base C + E, S slot E2, pw25 + lomm05) | mode | w .05 | w .10 | w .20 | w .30 |
|---|---|---|---|---|---|
| DT | pre R0 / R1 | +.0006 / +.0007 | **+.0012 / +.0013** | +.0021 / +.0024 | +.0028 / +.0032 |
| DT | post R0 / R1 | +.0010 / +.0008 | +.0020 / +.0016 | +.0038 / +.0029 | +.0052 / +.0039 |
| D (placebo) | pre R0 / R1 | +.0001 / .0000 | +.0001 / -.0000 | -.0001 / -.0002 | -.0003 / -.0005 |
| DT - D | pre R0 / R1 | +.0006 / +.0007 | +.0011 / +.0013 | +.0022 / +.0026 | +.0031 / +.0037 |

All P 1.00 (B 300). Why it does not transfer: on OOF the added head is stronger than the base (DT .8822 vs stand-in .8755 OA-mean,
report labels; D .8763), so more weight keeps helping. On gold58 the real base (swide .9233) is .037 above DT, and the curve turns
negative at w .15. The OOF ruler is also report-regime (v4 label phi .52 / .46 / .43 vs gold58 .32 / .37 / .28), and the R1 reweight
barely moves it (ESS 4,052 of 4,333).

## 4. Value (MODEL / GUESS)

- OAI-specific signal: OOF R1 +.0013 OA-mean at w .1 = +3.3e-4 macro on a weak, same-family base (MEASURED). Transfer to c4 GUESS
  .1-.4 (c4 already has Raptor's OAI head at .40 / .16 / .40, E-OA already decouples the trio, and the c4 members are stronger) gives
  +0.3 to +1.3e-4.
- Generic dilution at w .1 pre (placebo D on gold58 -.0002 OA-mean = -0.5e-4; E -2.8e-4; C +1e-4): about 0 to -1e-4.
- Net swide + DT (pre, w .1): **E about +0.3e-4 [-2, +2], P(>0) ~.55 (GUESS).** w >= .15 and every post variant: E < 0.
- In the 10-22 world: P(S6F clean AND DT stage used) ~.45 (GUESS: .55 entry-level P(clean), CLAIMED CANARY_1021, times ~.85 for the
  extra stage's own guard / env / read risk). It moves A only through a 3-dp public read whose noise (sd ~1.8e-4) is 6x the effect,
  so it becomes A in about half the clean worlds, regardless of its true sign. Expected effect on A's private score: about 0 to +0.1e-4.
  P(it changes a medal outcome) < 1 %. Opportunity cost: a slot the r6pw20 hedge copies want (+0.10-0.12 silver step pts each, CLAIMED
  CANARY_1021).

## 5. Feasibility by ~09:00 IST 10-22 (track 3)

| item | status | source |
|---|---|---|
| OAI-clean weights dataset for DT | **exists**: `eeshsaxena/rsna-knee-cnxt-fleet-k14w`, 523.6 MB, updated 2026-10-12 06:22 UTC, 5 DT raw-best ckpts re-keyed `f{F}_s1043f75oai_*` + `k12w_manifest.json` | MEASURED `kaggle datasets list --mine`; CLAIMED PROGRESS_12 (K14CLEAN 5/5, OAI-clean scan PASS) |
| DO (T12) weights | **no dataset** (k12w never created, K12 gate FAIL) | CLAIMED K12_RESULT.md |
| PL / swinT / allv / s1044 heads | f0 or f0-f2 only, no dataset | MEASURED `box_inv.log` |
| kernel stage that runs DT | pieces exist: HF-DT20 (`rsna-sub-hfdt20` v1, F = DT 5 members, HCHECK 16/16, 10-12) and the V224-F fleet cells (ENV-SNAP, FN-IDS, K12-MERGE k11w + k14w, CLOCK, FLEET K2_CFG, ENV-RESTORE; commit-tested 10-14) | CLAIMED PROGRESS_12, session_2026-10-15/v224f/RESULT.md |
| a c4 / swide kernel with a DT stage + DT-OA blend in cell 18 | **does not exist**; needs builder, offline sim with a mocked fleet solo, negatives, GPU commit, env-parity check (the fleet has never run after R6's wheels), gate, registry, A23, GO | GUESS 4-5 h main-session work |
| runtime of a DT-only stage | ~0.35-0.5 h central, ~0.5-0.7 h slow, ~0.7-1.2 h on a 10-21-type day for ~1,300 studies on 2 x T4 (5 x 224 members = +0.27 h central / +0.39 h slow in the K12 model; fleet fixed part + decode 1.40 s / study / thread) | MODEL from CLAIMED K12_RESULT sec 2, v224f wall_model |
| placement | after S6F (S6F ends ~6.0-6.2 h normal; a guarded post-S6F stage of up to ~1.5 h fits every non-tail session) with its own start guard and a visible fallback | CLAIMED cause.md sec 2, think/runtime.md |
| 9 h risk on a slow day | the extra stage does not touch S6F's guard if placed after it; it can only skip. A 10-22 c4-class entry already has P(clean) ~.55 [.35-.80]; with the stage P(clean and DT used) ~.45 | CLAIMED CANARY_1021; GUESS |
| timing | build tonight -> commit (20-40 min, dataset reads 5-10x slower on 10-21) -> submit 05:31-07:00 IST -> read ~11:30-15:30 IST (latency up to ~8.5 h) | CLAIMED CANARY_1021 |
| external data | OAI gray, no host ruling; Raptor (in both picks) is a public OAI checkpoint; our own OAI-trained weights are a step further ("touched the data", forum 750236) | CLAIMED forum.md 10-17, FORUM_10-19 |

## 6. If Eesh still wants it (not recommended)

Only variant with non-negative evidence: **swide + DT OA trio at rank weight .10 BEFORE pw E-OA** (cell 18: after the S6fall blend,
`out[c] = .9 pct(out[c]) + .1 pct(DT_solo[c])` for the 3 OA labels; DT_solo = equal raw-prob mean of the 5 k14w members from a
post-S6F fleet stage; any stage failure -> a canary, never a silent swide copy). It needs an A-eligible clause in A23 (it conflicts
with "a 10-22 entry can never be A"). Never w >= .15, never post-chain, never DO / f0 heads.

## 7. Files (this track; all in `session_2026-10-21/lastday/`)

- `box_inv.py` / `.log`: inventory (fold counts, OOF row counts, gold74 pred files; aggregates only).
- `box_count.py` / `.log`: held-out OAI knee counts (id shapes with digits masked).
- `local_proxy_check.py` / `.log`: gold58 proxy == real cell-18 replay of c4 / pw25 / swide (laptop, RSNA only); writes
  `g58_payload.npz` (RSNA gold58 serves, embedded base64 in the box script, never copied to the box as a file).
- `box_oaihead_src.py` -> `box_oaihead.py` (payload embedded) / `box_oaihead.log` (B 2,000 / 300) / `box_oaihead_smoke.log` (B 40 / 10).
- Run pattern: `ssh training@192.168.10.30 'BOOT=2000 BOOTO=300 OMP_NUM_THREADS=2 nice -n 19 ~/miniconda3/envs/rsna/bin/python -I -' < box_oaihead.py`.
