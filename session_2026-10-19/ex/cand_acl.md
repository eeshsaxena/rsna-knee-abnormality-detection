# cand_acl.md: candidate "acl" (ACL decorrelated from the menisci / Effusion), 10-20 night, Claude. RESEARCH + MEASUREMENT ONLY.

Nothing was built, pushed, committed or submitted; no dataset, GPU job, scheduled task, PREREG, HANDOFF, PROGRESS or a20 file
was touched. Box use: two read-only CPU python runs piped over ssh stdin (smoke + full), printing aggregates only and writing
nothing on the box. No OAI id, row or per-knee value left the box; only gold74 aggregate 2x2 / joint counts went TO the box.
Tags: MEASURED (file named), CLAIMED (older note), MODEL (arithmetic on stated assumptions), GUESS (judgement).
Macro = 12-label macro AUC = ACL single-label delta / 12. ACL is not an OA column, so this transform is additive to E-OA
(eoa10 / eoa20) and does not interact with it.

## 0. Verdict: KILL

- Transform: out_ACL = pct(pct(f_ACL) - a * mean(pct(f_o) for o in group)), a in {-.1, .1, .2}.
- Groups: LM (the only ACL pair with consistent excess coupling in coupling_map.md: gap OAI +.13, gold58 +.18, gold74 +.19),
  MM (mixed gap), MM+LM, Eff (deficit gap; the rule says do not decorrelate, reference only) and MM+LM+Eff (the task group).
- **Criterion 1 fails for every group on OAI, all 3 models, both ACL binarisations** (MEASURED, `cand_acl_box.log`):
  decorrelation loses at a .1 and .2 with P>0 = 0.00; the a -.1 "control" (borrowing) is the side that gains.
- **Criterion 2 is mixed and confounded** (MEASURED): the only pass (MM under the 7-label reweight) comes from the OAI
  degenerative ACL-OA link, vanishes at +1 sd coupling and reverses for complete ACL tears.
- **Criterion 3 is against it** (MEASURED, `cand_acl_local.log`): gold58 c4 loses at a .1 for 4 of 5 groups (LM +.0012 is one
  discordant pair, P .52) and at a .2 for all 5; R7 OOF on gold74 / gold58 expert labels loses for all groups at a .1 and .2.
- Report-label OOF loses too (as the mechanism predicts), but it loses in every regime, so this read has no contrast.
- Expected hidden macro (MODEL): task group a .1 about -.0002 (range -.0006 to +.0001); LM-only a .1 about -.0002 (range -.0005
  to +.0001). P(gain > 0) about .10 for the task group, .25 for LM-only (GUESS). Not worth a slot or a stack slot.

## 1. OAI, criterion 1 (raw, paired bootstrap B 1000; MEASURED, `cand_acl_box.log`, `cand_acl_table.out`)

Cohort: OAI knees with ACL, MM, LM, Eff and OA-trio labels + all 3 OAI-free models, n 1,654. ACL B1 (T123 >= .5, partial +
complete) prevalence .089 (148 pos); B2 (complete only) .034 (56 pos). Models D5mean_s1043 / s1044_mix50 / s1045_288.

| ACL <- | base ACL AUC | rho - q | a -.1 (borrow) | a .1 [P>0 min] | a .2 [P>0 min] |
|---|---|---|---|---|---|
| LM | .938/.920/.902 | -.05/+.01/-.10 | +.0005/+.0008/+.0036 | -.0065/-.0062/-.0084 [.00] | -.0203/-.0169/-.0211 [.00] |
| MM | same | -.08/-.05/-.18 | +.0036/+.0040/+.0067 | -.0097/-.0091/-.0113 [.00] | -.0245/-.0223/-.0260 [.00] |
| MM+LM | same | -.16/-.10/-.28 | +.0035/+.0038/+.0066 | -.0064/-.0059/-.0084 [.00] | -.0159/-.0148/-.0195 [.00] |
| Eff | same | -.09/-.08/-.25 | +.0007/-.0003/+.0061 | -.0060/-.0037/-.0114 [.00] | -.0190/-.0131/-.0256 [.00] |
| MM+LM+Eff | same | -.16/-.12/-.32 | +.0032/+.0029/+.0069 | -.0056/-.0045/-.0089 [.00] | -.0149/-.0123/-.0203 [.00] |
| B2 (complete), MM+LM+Eff | .972/.963/.956 | -.06/+.01/-.20 | +.0009/+.0011/+.0055 | -.0045/-.0034/-.0093 [.00] | -.0129/-.0107/-.0213 [.00] |

- rho = corr of normal scores (ACL score vs group score); q = corr(group score, y_ACL) / corr(ACL score, y_ACL). Theory
  (skeptic, MODEL): small-a decorrelation gains iff rho > q. On OAI q > rho for every group: the OAI-free models' meniscus /
  effusion scores carry ACL signal (OAI ACL tears are degenerative and travel with meniscal tears, effusion and OA).
- Pairwise LM / MM / Eff rows reproduce merge_map.out exactly (consistency check of the new script, MEASURED).

## 2. Criterion 2: coupling-matched reweighting to gold74 (skeptic method; RBOOT 400; MEASURED)

gold74 expert coupling of the involved labels (MEASURED, local aggregates): ACL-MM tet +.10 (log OR +.24 +- .47), ACL-LM -.06
(-.17 +- .48), ACL-Eff +.29 (+.79 +- .52); the confounders ACL-MOA -.38, ACL-LOA +.04, ACL-PFOA -.46. OAI experts: ACL-MM +.52,
ACL-LM +.33, ACL-Eff +.47, ACL-MOA +.53, ACL-LOA +.43, ACL-PFOA -.10 (coupling_map.md). So OAI is the wrong population for ACL
(CLAIMED caveat, MEASURED here): its ACL tears are rare, degenerative and OA-linked; RSNA's are traumatic and anti-OA.

| scheme | what it matches | ESS | weighted phi ACL-MM / LM / Eff / MOA / PFOA |
|---|---|---|---|
| R2x4 | 4-label IPF (ACL, MM, LM, Eff), OAI marginals + gold74 log ORs | 1,381 | +.04 / -.02 / +.11 / +.21 / -.05 |
| R2x7 (primary) | 7-label IPF (+ MOA, LOA, PFOA): also moves ACL-OA to gold74 | 571 | +.04 / -.02 / +.11 / -.11 / -.16 |
| R2x7_hi | R2x7, 6 ACL-pair log ORs + 1 sd | 683 | +.10 / +.04 / +.19 / -.06 / -.10 |
| R2x7_lo | R2x7, 6 ACL-pair log ORs - 1 sd | 463 | -.03 / -.06 / +.03 / -.14 / -.21 |
| R2x7_hiI | R2x7, only ACL-{MM, LM, Eff} + 1 sd | 622 | +.10 / +.04 / +.19 / -.11 / -.16 |
| R1j4 | gold74 4-label joint (moves ACL prevalence to .41) | 164 | +.06 / -.02 / +.15 / +.25 / +.01 |

ACL AUC delta at a .1 (3 models) [P>0 min]:

| ACL <- | R2x4 | R2x7 | R2x7_hi | R2x7_lo | R1j4 | B2 (complete) R2x7 |
|---|---|---|---|---|---|---|
| LM | -.0059/-.0056/-.0060 [.00] | -.0068/-.0117/-.0136 [.00] | -.0083/-.0107/-.0132 | -.0055/-.0133/-.0147 | +.0023/+.0008/+.0004 [.61] | -.0026/+.0024/-.0046 [.05] |
| MM | +.0009/+.0008/+.0013 [.65] | **+.0121/+.0072/+.0101 [.99]** | +.0048/+.0012/+.0025 [.71] | +.0203/+.0139/+.0187 | -.0070/-.0066/-.0055 [.00] | -.0108/-.0103/-.0114 [.00] |
| MM+LM | -.0008/-.0012/-.0006 [.15] | +.0030/-.0012/+.0005 [.18] | -.0009/-.0033/-.0034 | +.0074/+.0009/+.0045 | +.0000/+.0002/+.0015 [.52] | -.0033/+.0010/-.0053 [.00] |
| Eff | -.0026/-.0003/-.0068 [.00] | +.0044/+.0034/+.0013 [.69] | -.0007/-.0000/-.0048 | +.0094/+.0065/+.0070 | -.0036/-.0009/-.0073 [.00] | -.0079/-.0042/-.0138 [.00] |
| MM+LM+Eff | -.0008/-.0005/-.0027 [.02] | +.0036/+.0004/+.0015 [.59] | -.0003/-.0021/-.0031 | +.0078/+.0027/+.0060 | -.0009/-.0005/-.0016 [.06] | -.0041/-.0017/-.0079 [.00] |

Reads:
- LM (the map-selected pair) loses under every coupling-only scheme; only R1j4 (ESS 164, marginals moved) is weakly positive.
- Matching only the involved labels (R2x4) leaves every group at about 0 or below.
- The R2x7 MM pass is a confound, not report-inherited noise (MODEL): forcing ACL-MOA from +.27 to -.11 phi down-weights the
  OAI ACL+ / OA+ knees; the base ACL AUC falls from .92-.94 to .77-.86 (the OAI ACL score leans on OA), and subtracting the
  MM score (MM-MOA model coupling) then acts as "subtract OA". It shrinks to +.0012 to +.0048 at +1 sd ACL coupling, is
  negative under R1j4, and reverses for complete tears (B2: -.0103 to -.0114, P .00), the tear type closest to RSNA trauma.
- The criterion-1-style passes on reweighted OAI are only R2x7 MM (a .1 and .2) and R2x7_lo MM, Eff and MM+LM+Eff (a .1 and
  .2), i.e. the schemes with ACL-OA forced to gold74's anti-coupling or below; every scheme at gold74 point coupling of the
  involved labels alone (R2x4) or at +1 sd (R2x7_hi, R2x7_hiI) fails (`cand_acl_table.out`).

## 3. Criterion 3 and the report regime (local CPU; MEASURED, `cand_acl_local.log`)

c4 rebuild parity: gold58 macro .94102 (= m14_gate.log). gold58 ACL prevalence .414 (24 / 58), c4 ACL AUC .9890, so the
ceiling is +.011 single-label (+.0009 macro) and one discordant pair = .0012.

| read | ACL <- LM a .1 / .2 | MM | MM+LM | Eff | MM+LM+Eff | rho - q (LM / MM / all) |
|---|---|---|---|---|---|---|
| gold58 c4 | +.0012 (P .52) / -.0074 | -.0025 (P .00) / -.0086 | -.0012 / -.0086 | -.0049 / -.0098 | -.0025 (P .00) / -.0074 | -.01 / +.01 / -.04 |
| gold58 R6 / S6fall / Raptor (a .1) | -.0037 / -.0025 / +.0006 | -.0049 / -.0037 / -.0031 | -.0037 / -.0006 / +.0006 | -.0012 / -.0012 / -.0018 | -.0037 / .0000 / -.0006 | |
| R7 OOF, gold74 expert (n 74, AUC .978) | -.0046 (P .12) / -.0111 | -.0038 / -.0115 | -.0038 / -.0092 | -.0046 / -.0199 | -.0038 (P .08) / -.0100 | -.10 / -.12 / -.22 |
| R7 OOF, gold58 expert | -.0074 / -.0147 | -.0049 / -.0165 | -.0061 / -.0159 | -.0074 / -.0196 | -.0074 / -.0135 | |
| R7 OOF train vs report YES/NO (n 3,982) | -.0078 / -.0199 | -.0093 / -.0225 | -.0081 / -.0194 | -.0084 / -.0216 | -.0077 / -.0180 | -.27 / -.28 / -.39 |

- On RSNA expert labels c4's rho - q is about 0 for the menisci (LM -.007, MM +.009) and negative with Effusion: the meniscus /
  effusion scores carry as much real ACL information as their correlation with the ACL score, so there is no shared noise to
  cancel. The coupling-map gap for ACL-LM (model rho .12 vs expert tet -.06) is therefore not removable excess (MODEL): c4's LM
  score fires on acute ACL knees beyond the LM label (GUESS: lateral bone bruise / effusion features of the pivot-shift injury).
- Report regime: report labels couple ACL with MM / LM / Eff at tet .35 / .42 / .43 (R7 train rho .13 / .23 / .10, so R7
  couples LESS than its reports). Decorrelation loses there as predicted and borrowing gains (+.0034 to +.0058, P 1.00), but
  since decorrelation also loses on every expert read, the report contrast carries no signal for this candidate.
- Borrowing (a -.1) is no candidate either: it passes on OAI raw (MM) and on report labels, but is flat on RSNA experts (c4
  -.0012 to +.0012; R7 gold74 -.0011 to +.0015). On reweighted OAI the MM borrow flips negative (R2x4, R2x7) and the LM
  borrow is positive under R2x7 (+.0036 / +.0078 / +.0101) but negative under R1j4 and for complete tears (B2 R2x7).

## 4. Bottom line (GUESS unless tagged)

- KILL the ACL decorrelation family (all 5 groups, a .1 and .2). Best a: none. Do not add it to any entry, including as an
  add-on to eoa10 / eoa20.
- Why it fails (MODEL): ACL is the most trauma-specific label; the "excess" ACL coupling the map sees (ACL-LM only) is small
  (+.13 to +.19) and is matched by real cross-predictiveness of the meniscus score, ACL AUC is already near ceiling (.98-.99 on
  gold58 / gold74), and OAI cannot stand in for RSNA here because its ACL tears are degenerative and OA-linked.
- Consistent with coupling_map.md sec 5 ("ACL borrows ... do not chase"): this extends it to ACL decorrelation.

## Files (all in session_2026-10-19/ex/)

| step | script | output |
|---|---|---|
| local reads (gold74 coupling, gold58 c4 + members, R7 OOF gold74 / gold58 / report) | `cand_acl_local.py` | `cand_acl_local.log`, `cand_acl_local.json` (aggregates) |
| box script build (embeds gold74 aggregates) | `cand_acl_make.py` + `cand_acl_box_template.py` | `cand_acl_box_run.py` |
| box run (read-only, aggregates only) | `cand_acl_box_run.py` via `ssh ... python -I - < ...` | `cand_acl_box.log` |
| table | `cand_acl_table.py` | `cand_acl_table.out` |
