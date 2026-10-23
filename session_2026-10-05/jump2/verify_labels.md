# Verify: "labels" track (adversarial check, 2026-10-05 ~10:00 IST)

Target: `jump2/labels.md`, `jump2/kx_pilot.py`, `jump2/lab2/*` and the track's summary, findings and recommendations.
Tags: **MEASURED** (re-run or read by me today, source given) | **MODEL** (arithmetic on measured data) | **GUESS** |
**CLAIMED** (forum self-report). Box work: CPU only (`CUDA_VISIBLE_DEVICES=`, `nice -n 19`, 1-2 threads), scripts and
outputs only under `~/jump2/vlab/` (laptop copies in `scratchpad/jump2/vlab/`: `vlab1.py/.out`, `vlab2.py/.out`).
No GPU use, nothing killed or signalled, nothing under `~/fleet_runs`, `~/rsna_train`, `~/fleet`, `~/bigjump_build` written.

## 0. Verdict

Numbers: almost all reproduce exactly. Reasoning: three conclusions are overstated and one recommendation contradicts the
track's own data.

1. **"Better extraction is a closed lever": CONFIRMED in substance, but two supporting arguments are wrong.**
   - Every reader scores below v4 on gold58, and second readers add at most +.0042 (P .83). Both reproduced.
   - "Most of v4's deficit is the report disagreeing with the image" is wrong. Of the .0643 "both-unanimous" deficit,
     only ~.020 is cross-state report-vs-image disagreement. Another .0226 is ties among report-silent cells, and .0215 is
     ordering inside an agreed state (U+ .0153, NEG .0062), which a graded reader could in principle fix.
     So ".9357 caps v4" is the contested-only oracle, not a cap on a better reader.
   - "The 7B lost only to binarization" is partly refuted. On ex16 at equal binarization, Qwen-7B reads .8583 vs v4-bin
     .8991 and GPT-5.6-bin .9345. On gold58 it reads .8067 vs .8187 / .8352. Reading quality also cost it.
2. **"Archit round from 5-fold OOF" (rec 1) should be DEMOTED.** Reasons:
   - The .9340 preview reproduces, but it is the best of 18 variants. Its edge over the already-built mixSA_logit is
     +.0040 (SE .0039, P .85). With build-time logit moments it is .9324 (+.0024, P .75).
   - Its pass bar (+.003 over mixSA) is under 1 SE.
   - It would be checked on the mean of 5 fold checkpoints while each train row gets one OOF. Averaging two teachers
     already adds +.0017 at label level (MEASURED), so the gate is biased toward passing.
   - The label set is the most parent-like one we have (.884 vs mix50 .864). Every LB read of teacher-mixed members was
     flat or negative (K2 blend .943 = parent; mixsacrop25 .940 vs crop25 .941).
   - The sibling verifier measured the own-OOF-only round at .9254-.9263, a tie with mix50.
   - Expected public gain ~0.
3. **KX pilot gate is not pre-registered in the sense claimed.**
   - DEFS and the grade weights were written after reading 18 gold58 error reports and gold58 convention counts (section 4
     of labels.md). Examples: MCL fluid, degenerative edema, patellar facets, low-grade grading.
   - G1 (ex16) uses JEV's plain-statement reads with ambiguous cells set to NaN, so it cannot measure ambiguous reading.
   - G0's 1.5 s/report bar fails by construction with the recommended `--spans 1`.
   - Throughput is optimistic. Measured tokens are 2,918/report (not ~2,500). The batched 0.3-0.5 s/report estimate is at
     or below the V100 FLOP floor.
4. **"Labels cannot buy decorrelation" is TOO STRONG.**
   - True for teacher mixes: a parent-free own-teacher label sits at .852, vs .864 for mix50.
   - But the label choice is the largest decorrelation lever we have measured at model level: v4-trained OOF .840-.850
     vs mix50-trained .889-.896.
   - The Spearman-to-raptor metric also mixes accuracy with similarity. Two own v4 models agree .917 with each other.
5. Tooling, leakage and hygiene findings: all confirmed (no vllm/gptq/awq/flash-attn, one Qwen2.5-7B on disk, gold74 held
   out of every fleet run, qwen7b table overwrites gold rows, micro sign, bilateral, Bulgarian shoulder report).
   Small corrections:
   - The old 7B run took 1.87 s/report, not 1.7.
   - "Raymond Qwen2.5-32B" and "Tucker simple Qwen" are not in the saved forum text.
   - SpeedSci's ".945 -> .910" is an EffNet model's gold58 score on one label version, not a label set's score.
   - jev_final2 is not an independent reader: its member weights were fit on 67 gold studies.

---

## 1. Claim-by-claim

| # | Track claim | Status | What I found | Tag |
|---|---|---|---|---|
| 1 | No text reader beats steven_v4 on gold58 (v4 .8927; pilk .8636, lixin .8487, jev .8745, steven_full .8780) | **Confirmed, with caveats** | v4 .8927, mix50 .9262, mixSA .9300 reproduced (`vlab1.out`). pilk reads .8636 / .8659 / .8700 depending on NaN handling (label_eval / decomp / label_sources; report_labels_v2 lacks 1 gold row). jev_final2 is NOT independent of gold: JEV's README says member weights come from "AUC edge over 0.5 on 67 gold studies", so it is gold-tuned and still loses. v4 provenance (stevenleehans "v4_blend") is unknown, so v4 itself could be gold58-tuned. That would bias every "vs v4 on gold58" bar, but in the direction that makes the closed-lever verdict safe. | MEASURED |
| 2 | No second reader adds to v4 at any weight (jev +.0042 P .84; pilk +.0019 P .70) | **Confirmed** | Re-run with new seed, 3000 resamples: jev w.20 +.0042 (P .83, 90% CI -.0031..+.0118), w.35 +.0020 (P .63); pilk w.20 +.0013 (P .71), w.35 +.0019 (P .71). | MEASURED |
| 3 | Most of v4's deficit (.064 of .107) is reader-unanimous cells, "meaning the report itself disagrees with the image label" | **Numbers confirmed, interpretation refuted** | Split of v4's .1073 (`vlab1.out` sec 3): contested .0430; same-state SIL .0226 (v4 has 1-2 distinct values in SIL cells for 11 of 12 labels, so these are TIES: report silent, not disagreeing); same-state U+ .0153 and NEG .0062 (ordering inside an agreed state, the part a severity- or hedge-aware soft reader targets); cross-state disagreement only NEG>SIL .0095 + (NEG/SIL pos vs U+ neg) .0081 + SIL>NEG .0026 = .0202. labels.md body hedged this ("or ordering inside an agreed state"); the summary dropped the hedge. | MEASURED |
| 4 | Perfect reader on contested cells caps v4 at .9357 (mix50 .9611); majority/vote do worse | **Confirmed as a contested-only oracle; "caps" is wrong** | Code sets contested positives to 1.001 and negatives to -.001, i.e. above or below every unanimous cell, so it is generous for contested cells. It does not bound a reader that also orders U+ / NEG cells (up to .0215 more, MODEL upper bound). Majority .8653 and vote fraction .8795 numbers stand (read from code + txt). | MODEL |
| 5 | Within every reader state the image teacher ranks better than the text | **Confirmed but near-tautological** | States are defined by the text readers, so text has almost no variance inside a state (SIL .500 is pure ties). Per-label within-state AUCs rest on tiny n (SIL: Baker 1/29, LM 1/7, Contusion 1/9 positives). "Pooled" is a misnomer: the code averages per-label AUCs, which is the right method. It says nothing about whole-label ranking: text still wins PF OA and LM. | MEASURED |
| 6 | Old local 7B lost because of binarization, not reading | **Partly refuted** | Log confirmed: gold58 bal-acc macro .8067 (Effusion .653, Synovitis .667). Binary AUC = balanced accuracy: correct. On gold58: 7B .8067 vs v4-bin .8187 vs lixin-bin .8352, so binarization explains ~.074 of the .086 gap and reading ~.012-.029. On ex16 (report-read labels): 7B .8583 vs v4-bin .8991 vs lixin-bin .9345 (-.041 / -.076). Runtime: 13:47 -> 16:06, log "all 4400/4407 8202s" = **1.86-1.87 s/report**, not ~1.7. Leak confirmed (qwen7b gold58 rows == gold: True). OOM restart confirmed (another process held 14.79 GiB). | MEASURED |
| 7 | Severity wording does not recover the image label (effusion) | **Confirmed as measured; weak test** | Rates reproduced (small .46 n26, unqualified .92 n12, mod/large .82 n11, silent .60 n5). Inside v4-present cells (n 51, 35 positive): v4 .823 vs size code .684 vs code+v4 tiebreak .720. So crude regex sizing loses to v4 even where severity should matter. Caveats: one label, n 58, regex lists written after reading gold58 reports (script's own caveat). It does not test an LLM grade read. | MEASURED |
| 8 | Gold-convention mismatches cluster on low-grade findings; P(Contusion given Fracture) .556 (n 18) | **Confirmed (spot-checked)** | P(cont given frac) .556 vs .225 reproduced. The low-grade counts are regex over gold58 and in-sample by construction. | MEASURED |
| 9 | Multilingual / structural traps | **Confirmed** | Micro sign 305 of 321 Greek reports; 27 BILATERAL NOTE; 15 reports < 80 chars; Bulgarian report on subscapularis tendinopathy at the humerus (a shoulder) found; Effusion contested rate .463. Note `lab2_lang.out`'s own shoulder search returned 2 false hits (both knees); the shoulder report came from elsewhere, and I re-found it with a Cyrillic-aware regex. | MEASURED |
| 10 | Own teacher as good as wd; v4 + wd + own TTRmix SA = .9340 vs mix50 .9262 / rebuilt mixSA .9305 | **Numbers confirmed; significance and selection refuted** | Reproduced all 18 variants exactly (`vlab1.out` sec 1). .9340 vs mixSA file .9300: +.0040, SE .0039, P .85. It is the bootstrap winner in 64% of resamples (winner's curse small, +.0007). The script computes logit-match moments on the 58 eval rows. With build-time moments (v4/wd over the table, own teacher over its 867 OOF) it reads **.9324**, P>mixSA **.75**, P>mix50 .93. "Best untuned" holds only in the sense that weights were not fit; the reported row is still the max of 18 looks at gold58. | MEASURED |
| 11 | Own gold74 predictions are clean out-of-sample | **Confirmed, mild indirect contamination** | Every fleet log (f0 x5, f1) prints "held-out 74"; fold-0 val and gold74 are disjoint; checkpoints chosen by v4-val cv (gold logged as a bug guard only). But the mix50-trained teachers learned from Dread wd, whose checkpoint was picked on a gold holdout (forum, CLAIMED), so "TTRmix" is partly a distilled wd. | MEASURED / CLAIMED |
| 12 | Labels cannot buy decorrelation from the parent (.852 vs .864) | **Partly refuted (too strong)** | Label-level numbers stand. Model-level OOF Spearman vs raptor proxy (867 val): own TTRv4 .850, F1v4 .840 vs TTRmix .896, F1mix .889, teacher_wd .935. So pure-v4 labels are the biggest decorrelation lever we have measured (-.04 to -.05); what labels cannot do is go below the v4 level. The metric also mixes accuracy with similarity: two own v4 models agree .917, TTRv4 vs wd .827. The K3 null (TTR v4 at W .25 = .943) is the real reason to doubt the value of that decorrelation. | MEASURED |
| 13 | The brief's mix50 .9241 is ECDF; the training file reads .9262 | **Confirmed** | `labels_mix50_logit.csv` .9262; `auto_folds.sh` trains on that file. | MEASURED |
| 14 | Box stack: no vllm/gptq/awq/flash-attn; transformers 4.48 + bnb 0.50.2 in rsna; only Qwen2.5-7B usable | **Confirmed** | pip lists for rsna, kmimic (torch 2.5.1, tf 4.48.0, bnb 0.50.2, accelerate 1.14), ltx-v100 (tf 5.17), base (tf 5.17); egoindustrial/fs/trellis have torch only. HF cache has Qwen2.5-7B-Instruct as the only chat LLM. GPU Tesla V100-FHHL-16GB, compute 7.0. bnb nf4 on sm70 not tested (GUESS it works). | MEASURED |
| 15 | KX on 4,407 reports ~1-1.5 GPU-h fp16; batched 0.3-0.5 s/report; gold74 pilot ~5 min | **Optimistic / batched part refuted** | Real tokenizer + chat template (`vlab2.out`): prefix 539 mean, 1,323 Q1 + 1,056 Q2 suffix tokens, **2,918 tokens/report** (el 3,295), 12.9 M total, 25 sequential forwards per report. FLOP model (2 x 7.07e9 per token, HF computes logits for all positions): 2.0 h at 25 TFLOPS, 1.26 h at 40, 0.84 h at 60, 0.45 h at the ~112 TFLOPS peak, forward-only. With per-call overhead, realistic **~1.3-2.2 GPU-h**. Batching does not cut FLOPs, so 0.3-0.5 s/report needs >= 75-120 TFLOPS sustained, at or above the card's peak. With `--spans 1` (the recommended pilot command), roughly 5-6 cells per study trigger a full-prompt `generate` of up to 32 tokens (~1-1.5 s each, MODEL from the old run's ~34 ms/step), so gold74 is ~10-15 min, not ~5. | MEASURED / MODEL |
| 16 | GPU has no room now (12.2 GB used; 8.9 GB process not this track's) | **Confirmed** | 09:45: 12,245 MiB used, 100% util; PID 339781 = `~/fast_build/speed_ab.py` (nice 5, started ~09:33). `~/jump2/gpu.lock` is referenced by `~/fast_build/bench.py`, so it belongs to the fast_build track, not labels. If that track is a subagent, it breaks the "never use the GPU" rule. Untouched. | MEASURED |
| 17 | Forum agrees (Scott, Raymond, Tucker, SpeedSci) | **Mostly confirmed, three misattributions** | Scott: local Gemma 4, "around .89", LB .949 single fold, ~.930 gold (confirmed). Raymond: "difference between expensive and cheap models are minimal", < $5 API (confirmed); **"Qwen2.5-32B" is not in any saved thread**. Tucker: "Maybe 0.89", "thousands of verified extraction failures. I correct them, no change in LB" (confirmed); **"simple Qwen prompt" not in the saved text** (he says "single prompt LLM extraction"). SpeedSci: ".945 on Gold" is "on EffNet" for "one label version", LB .910 (a model's gold58 score, not a label set's). | CLAIMED (re-read) |
| 18 | ex16 = "careful manual report reads", so readers already read at ~.98 and reading is not the bottleneck | **Weakened** | ex16 = JEV's public hand-verification: "marking a label only when the report states the finding's direction plainly". Its conventions match gold58's (small effusion = 0, suspected = 0), and 18 of 192 cells are NaN. So ex16 measures plain-statement reading only; contested cells, the ones a better reader would need, are excluded by design. jev_hybrid's .9879 there is partly its author grading his own convention. | MEASURED (gold74 csv + JEV README) |
| 19 | Archit-round labels ready 10-06 ~06:00 (chain ends ~03:30), LB 10-07 | **Late by ~1-2 h; fragile** | J1 fold 0 took 5.0 h alone (epochs 12.3-13.5 min). Fold 1 (from 08:25) ran epochs 3-4 at 17.2 / 19.2 min under contention (speed_ab GPU job + gold_infer CPU). Uncontended projection: fold 1 ends ~13:30-14:00, chain ends ~04:30-05:15 IST 10-06 (MODEL). `auto_folds.sh` also waits before each fold while ANY python process holds the GPU, so other tracks' GPU jobs can stall it further. | MEASURED / MODEL |

---

## 2. Re-run computations (box CPU, `~/jump2/vlab/vlab1.py`, `vlab2.py`)

### 2.1 Self-teach label sets on gold58 (2000 paired study resamples)

| label set | gold58 (eval-row moments, as track) | gold58 (build-time moments) | P > mixSA_file (build) | P > mix50_file (build) |
|---|---|---|---|---|
| mix50_logit file | .9262 | | | |
| mixSA_logit file | .9300 | | | |
| v4 + wd SA (= mixSA rebuilt) | .9305 | .9300 | (ref) | .94 |
| v4 + own TTRmix SA | .9272 | .9267 | .33 | .53 |
| v4 + own TTRmix+F1mix SA | .9291 | .9291 | .45 | .65 |
| v4 + own TTRv4+F1v4 SA (parent-free) | .9096 | .9109 | .01 | .03 |
| **v4 + wd + own TTRmix SA** | **.9340** (d +.0040, SE .0039, P .85) | **.9324** | **.75** | .93 |

Bootstrap win share among 18 variants: v4+wd+TTRmix SA .64, v4+wd SA .16, the rest <= .07.

Teacher-averaging bias (the proposed gold check uses the 5-ckpt mean, while train rows get one OOF each):
- Single teachers: v4 + TTRmix SA .9272, v4 + F1mix SA .9275.
- Mean of both teachers: .9291, i.e. +.0017 from averaging alone.
- Raw models: .9132 / .9144 single vs .9187 for the mean of the two.
- With 5 checkpoints the inflation is likely +.002 to +.004 (MODEL). That is the size of the +.003 bar.

### 2.2 Deficit split (steven_v4, 1 - AUC = .1073)
contested .0430 | same-state SIL (ties) .0226 | same-state U+ .0153 | cross NEG>SIL .0095 | cross NEG/SIL-pos vs U+-neg .0081
| same-state NEG .0062 | cross SIL>NEG .0026. Report-vs-image disagreement proper = ~.020 (19%).

### 2.3 ex16 (16 studies, 174 non-NaN cells, all 12 labels have both classes)
v4 .9808 | mix50 .9692 | lixin binary .9345 | v4 binarized .8991 | teacher_wd .8992 | pilk verdict .8605 | qwen7b binary .8583.

### 2.4 KX token budget (real Qwen2.5 tokenizer and chat template, all 4,407 reports)
system+template 180 tokens; prefix mean 539 (median 506, p99 1,363, max 2,901); per report 2,918 mean (median 2,885,
p99 3,742); by language median el 3,295, bg 3,010, fr 3,014, hr 3,005, en 2,864, es 2,778, tr 2,789; 13 reports truncated
at 3,500 chars. 12.86 M tokens in total.

---

## 3. Leakage and rules

- **KX prompt in-sample to gold58:**
  - labels.md section 4 records reading the 18 gold58 reports with the most v4 errors.
  - KX DEFS then encode exactly those cases: MCL bursa fluid, degenerative subchondral edema, patellar facets to PF, Baker
    vs parameniscal cyst, grade 1-2 meniscal signal.
  - The pre-registered score down-weights "low grade" (.60), following gold58 counts (small effusion .46 positive).
  - So a G2-G4 pass on gold58 would be optimistic.
  - Fix: score the gate on the ~40 gold58 studies NOT among the 18 read (reconstructable from `lab2_v4err.csv` error
    counts), and report the 18 separately.
- **Gold-tuned comparators:** jev_final2 weights were fit on 67 gold studies; mixSA's logit mapping was confirmed on gold58
  (.9300 vs .9275) after being motivated a priori; wd was checkpoint-picked on a gold holdout (CLAIMED).
- **Selection on gold58:** the .9340 row is the max of 18 variants (section 2.1); the track did not tune weights, but it
  reports the winner.
- **Leaky tables:** `pseudo_labels_qwen7b`, `pseudo_labels_v1` (gold58 AUC 1.000) confirmed. Any blend with them is leak.
- **Rules:** the labels track stayed CPU-only and wrote only under `~/jump2/`. It used the Qwen tokenizer from cache, ran
  a random 2-layer model on CPU, and made one `curl -I` to huggingface.co (a probe, no download). No GPU job, no kill.
  The GPU job on the box now is the fast_build track's `speed_ab.py`, not this track's.
- **Pending decision (KX pilot):** any KX pilot is a GPU job and needs Eesh's explicit OK. Under the current "never use the
  GPU" rule no agent may run it.

---

## 4. Corrected recommendations (ranked)

1. **Keep: close the extraction lever without spending GPU.**
   - The closure is already supported:
     - every reader is at or below v4 on gold58;
     - second readers add at most +.004 (P .83);
     - the forum agrees (Tucker: corrections gave no LB change; Raymond: cheap and expensive models give the same
       labels);
     - our K2/K3 reads are flat.
   - A KX pilot cannot be a clean closure test, because its prompt was written after reading gold58 errors.
   - Run KX only if Eesh wants it anyway, with these changes:
     - score the ~40 unread gold58 studies separately;
     - drop G0's timing bar or time `--spans 0` alone;
     - budget ~15 min of GPU for gold74 and 1.3-2.2 GPU-h for all 4,407 reports.
   - Expected public gain ~0 (GUESS).
2. **Demote the Archit round (own-OOF + wd teacher, rec 1).**
   - Label-level edge over the existing mixSA_logit: +.0024 to +.0040, P .75-.85 (MEASURED).
   - The gold check is biased upward by teacher averaging (MODEL +.002 to +.004).
   - It is the most parent-like target set (.884).
   - LB history of teacher-mixed members is flat or negative.
   - If a label change is wanted, `labels_mixSA_logit.csv` already exists.
   - If the round is built anyway: compute the gold check as the mean of the 5 single-checkpoint label scores, not the
     score of the 5-checkpoint mean; use build-time moments; require P(>mixSA) >= .90.
   - Expected public gain -.001 to +.002 (GUESS). It would cost 5-7 V100 h on the one GPU, which the 5-fold TTR member
     and a second family also need.
3. **Keep:** do not tune per-label mix weights or report rules on gold58; do not re-extract 4,407 reports. Cite SpeedSci
   correctly: an EffNet trained on one label version read gold58 .945 and LB .910.
4. **Correct the decorrelation advice.** The only label lever that lowers parent-likeness is pure steven_v4 targets (model
   OOF .84-.85 vs .89-.90). Any second family should train on pure v4, as HANDOFF 10-03 already says. Do not train it on
   the .884 Archit set.
5. **Keep the hygiene list**: no gold written into label tables; micro sign to mu; 27 two-report studies; 1 shoulder report;
   keep labels soft. Also add: jev_final2 and jev_hybrid are gold-tuned, so treat them as in-sample on gold58.
