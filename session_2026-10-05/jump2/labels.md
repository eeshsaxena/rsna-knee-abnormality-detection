# Label lever (jump2 track), 2026-10-05

Tags: **MEASURED** (source file), **MODEL** (arithmetic or bound on measured data), **GUESS**, **CLAIMED** (forum self-report).
All box work CPU only (`CUDA_VISIBLE_DEVICES=`, `nice -n 19`), inputs read-only, outputs only under `~/jump2/`.
Box copies of every output: `~/jump2/lab2_*`; laptop copies: `scratchpad/jump2/lab2/`.

---

## 0. Bottom line

1. **Better report extraction is a closed lever for us.**
   - Every independent text read we hold, including frontier API reads, scores BELOW steven_v4 on gold58:
     pilkwang (Opus-5 API) .8636, lixin (GPT-5.6) .8487, JEV 6-LLM ensemble .8745, steven_full .8780 vs v4 .8927
     (MEASURED `label_eval.txt`).
   - As a SECOND reader none adds to v4: best is jev at weight .20, +.0042 (P>v4 .84); pilk +.0019 (P .70);
     lixin and steven_full add nothing; a 4-reader rank-mean is .8899 (P>v4 .36) (MEASURED `lab2_reader2f.out`).
   - Decomposition of v4's gold58 deficit (1 - AUC = .1073): .0643 sits in pairs where all 5 readers agree
     (the report itself disagrees with the radiologist image label, or ordering inside an agreed state);
     only .0430 involves contested cells (MEASURED `lab2_decomp.txt`).
   - Hard ceiling: resolving ALL 145 contested gold58 cells perfectly lifts v4 to .9357 and mix50 to .9611
     (MODEL bound). Real readers capture far less: 5-reader vote fraction .8613, majority .8653 (MEASURED), both below v4.
   - In every reader state the image channel out-ranks the text (pooled within-state AUC, MEASURED):
     report-silent cells teacher_wd .937 / raptor .930 vs v4 .500; all-readers-positive .854 / .887 vs .657;
     contested .812 / .829 vs .724; explicitly negated .729 / .750 vs .649.
   - Our earlier local Qwen2.5-7B read scored .8067 on gold58 (binary, so AUC = balanced accuracy) (MEASURED
     `~/rsna_train/extract7b.log`), level with binarized frontier reads (v4 binarized .8187, lixin .8352, pilk
     verdict .8037, MEASURED). Its gap to v4 was binarization (soft vs binary is worth +.074 on v4 itself), not reading.
   - Forum agrees (CLAIMED, `topsolutions.md`): local-LLM labels ~.89 gold (Scott, Raymond, Tucker); fixing
     thousands of extraction errors gave no LB change (Tucker); a gold-.945 label set scored LB .910 (SpeedSci).
2. **The label lever that does move gold58 is the image teacher, now measurable cleanly with our own
   out-of-sample gold predictions.** v4 + Dread wd + own TTR-mix50 fold-0 teacher with silence-aware weights reads
   **.9340** vs mix50_logit file .9262 and the rebuilt mixSA .9305 (MEASURED `lab2_selfteach.out`). This is the
   "Archit round" and should be built from the 5-fold TTR mix50 OOF when the chain finishes (~03:30 10-06, MODEL).
3. **Labels cannot buy decorrelation from the parent.** v4 + a parent-free own teacher (TTR v4 + F1 v4) reads
   gold58 .9096 but its parent-likeness on fold-0 val is .852, vs mix50 .864 and v4 .645 (MEASURED). Any image
   teacher pulls labels toward the parent; decorrelation has to come from the model family.
4. **KX extraction design + pilot gate are written and plumbing-tested** (`kx_pilot.py`, random 2-layer model on
   CPU, real Qwen tokenizer). It needs ~5 min of free GPU for the 74 gold reports (MODEL). Prior P(pass) is low
   (GUESS 5-15% per bar). Run it only as a closure test in an idle GPU window, never displacing fleet training.

---

## 1. Every label set vs gold58 (AUC of the label value as a score vs train.csv image labels, n = 58)

Source: `~/jump2/label_eval.txt` (paired bootstrap 2000, by study) and `~/jump2/lab2_decomp.txt`. All MEASURED.

| set | gold58 | ACL | MCL | MM | LM | MOA | LOA | PFOA | Eff | Syn | Bak | Cont | Frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| steven_v4 | .8927 | .987 | .968 | .948 | .879 | .932 | .833 | .902 | .877 | .790 | .944 | .860 | .793 |
| steven_v4 binarized (>.5) | .8187 | .891 | .929 | .864 | .834 | .885 | .824 | .797 | .652 | .676 | .904 | .767 | .801 |
| steven_v2 / steven_full | .8873 / .8780 | | | | | | | | | | | | |
| pilk value (Opus-5) / verdict Y1 U.5 N0 | .8659 / .8037 | .990 | .976 | .943 | .841 | .909 | .789 | .879 | .824 | .695 | .933 | .758 | .855 |
| lixin GPT-5.6 signed conf / binary | .8487 / .8352 | .956 | .923 | .929 | .909 | .873 | .819 | .821 | .684 | .698 | .933 | .862 | .776 |
| jev_final2 (6-LLM text ens) | .8745 | .960 | .977 | .909 | .862 | .924 | .878 | .912 | .815 | .631 | .888 | .839 | .899 |
| jev_hybrid (swap chosen ON gold58: in-sample) | .8999 | | | | | | | | | | | | |
| gated75_logit | .9109 | .986 | .968 | .950 | .889 | .933 | .840 | .907 | .878 | .836 | .967 | .868 | .908 |
| mix50_logit (the training file) | .9262 | .995 | .982 | .975 | .918 | .983 | .843 | .888 | .984 | .823 | .975 | .906 | .843 |
| mix50_ecdf | .9241 | | | | | | | | | | | | |
| mixSA_logit | .9300 | | | | | | | | | | | | |
| teacher_wd (Dread CoAtNet, image) | .9127 | .979 | .973 | .960 | .839 | .981 | .843 | .820 | .984 | .777 | .971 | .934 | .892 |
| raptor_proxy (parent arm, image) | .9246 | .978 | .993 | .966 | .870 | .984 | .832 | .834 | .975 | .830 | .971 | .927 | .935 |
| own F1 mix50 fold-0 (OOS) | .9144 | .969 | .930 | .959 | .893 | .972 | .822 | .838 | .976 | .792 | .958 | .950 | .912 |
| own TTR mix50 fold-0 (OOS) | .9132 | .964 | .968 | .938 | .894 | .984 | .847 | .842 | .958 | .767 | .944 | .955 | .896 |
| own TTR v4 fold-0 (OOS) | .8780 | .945 | .902 | .912 | .876 | .984 | .814 | .839 | .865 | .628 | .967 | .962 | .840 |
| own F1 v4 fold-0 (OOS) | .8611 | .896 | .753 | .903 | .901 | .957 | .768 | .828 | .887 | .615 | .991 | .960 | .876 |

Notes:
- The task brief's "mix50 .9241" is the ECDF variant. The file the fleet trains on, `labels_mix50_logit.csv`, reads
  **.9262** (MEASURED).
- Leaky tables (gold rows overwritten with gold, gold58 reads 1.000, unusable for gold eval): `pseudo_labels_qwen7b`,
  `pseudo_labels_v1`, `yunus_v5`, `rulev0`, `qwen3b_laptop` (MEASURED leak column). Any blend with them
  (CONS_llm5 .9756, "indep4" .9744 in `labx.log`) is leak, not signal.
- Own fold-0 gold predictions: CPU fp32 inference of the fold-0 checkpoints on gold74 (`~/jump2/gold_infer.log`).
  gold74 is excluded from every fleet run, so these are clean out-of-sample.
- ex16 (16 extra gold74 studies whose labels are careful manual report reads = reading fidelity):
  v4 .9808, jev_hybrid .9879, CONS_llm4 .9878, gated75 .9791, mix50 .9692, pilk .9495, teacher_wd .8992 (MEASURED).
  Our readers already read the report at ~.98: reading is not the bottleneck.
- Noise: paired-bootstrap 90% intervals of a delta vs v4 are about +-.010 to +-.015 wide (steven_v2 -.0054
  [-.0129, +.0010]; jev_hybrid +.0072 [-.0037, +.0182]; CONS_llm4 -.0088 [-.0224, +.0044]) (MEASURED). A text gain
  must be >= ~+.012 to clear it.
- Text beats image on gold58 only for PF OA (v4 .902 vs wd .820 / raptor .834) and Lateral Meniscus (.879 vs .839 /
  .870); image wins big on Effusion (.877 vs .984), Contusion (.860 vs .934), Fracture (.793 vs .892 / .935),
  Medial OA (.932 vs .981) (MEASURED).

---

## 2. Where v4's gold58 error comes from (`~/jump2/lab2_decomp.py`, `.txt`)

Five text readers per cell: steven_v4 > .5, steven_full > .5, pilk verdict YES, lixin >= .5, jev > .5.
States: U+ (5/5 say present), C (1-4 of 5), NEG (0/5 and pilk says NO), SIL (0/5 and pilk says UNK = report silent).

| state | gold-positive / gold-negative cells (of 696) |
|---|---|
| U+ | 151 / 30 |
| C (contested) | 58 / 87 |
| NEG | 7 / 183 |
| SIL | 24 / 156 |

- 61 / 696 cells (8.8%) are reader-unanimous AND disagree with gold (U+ & gold 0: 30; NEG or SIL & gold 1: 31).
  No reader can fix these (MEASURED).
- Deficit attribution, steven_v4 .1073 = .0643 both-unanimous + .0430 contested; top pair types SIL>SIL .0226,
  C>C .0159, U+>U+ .0153 (MEASURED). mix50 .0738 = .0389 + .0349.
- Oracle bounds (MODEL on MEASURED data): contested cells set to gold: v4 .8927 -> .9357, mix50 .9262 -> .9611.
  Contested cells set to the 5-reader majority: .8653 / .8829; to vote fraction .8795 / .9006 (all BELOW the base).
  v4's own soft levels already beat any reader vote.
- 127 / 696 cells where v4 thresholded at .5 disagrees with gold: C 66, U+ 30, SIL 24, NEG 7; by label Synovitis 18,
  Effusion 16, Contusion 15, PF OA 12, Fracture 11, Lateral OA 10, LM 9, MM 8, MOA 8, ACL 7, MCL 7, Baker 6
  (MEASURED, list in `~/jump2/lab2_v4err.csv`).
- Contested-cell rate over all 4,407 studies (MEASURED `lab2_lang.out`): by label Effusion .463, PF OA .255,
  Lateral OA .230, Contusion .205, MOA .181, MM .176, LM .160, ACL .155, MCL .123, Fracture .033, Baker .032,
  Synovitis .020. By language: hr .311, de .220, tr .209, el .204, fr .203, nl .162, bg .148, en .138, es .102.

---

## 3. The earlier local 7B extraction (`~/rsna_train/box_extract7b.py`, 2026-09-03)

- What it did (MEASURED code + logs): Qwen2.5-7B-Instruct, fp16, whole model on the V100 (15.2 GB), zero-shot,
  one prompt with 12 short definitions, output = one-line JSON of 12 binary 0/1 values, greedy, 110 new tokens,
  report cut at 3,500 chars, batch 2. Gold check = per-label balanced accuracy.
- Gold58 (MEASURED `extract7b.log`): ACL .853, MCL .842, MM .903, LM .770, MOA .909, LOA .813, PF OA .786,
  Effusion .653, Synovitis .667, Baker .893, Contusion .779, Fracture .811; macro **.8067** (the earlier 3B: .7313).
- Runtime: all 4,407 reports 13:47 -> 16:06 (2 h 19 min, ~1.7 s/report incl. the gold pass) (MEASURED `chain7b.log`).
- It then wrote `pseudo_labels_qwen7b.csv` with the 58 gold rows OVERWRITTEN by gold (leak), and retrained an old
  embedding head: "ENSEMBLE gold-58 .6204" (`retrain7b.log`; old pipeline, not comparable to today's fleet).
- Train-set agreement: Spearman vs v4 .616, vs parent raptor proxy .516 (MEASURED `labx.log`).
- A first restart died with CUDA OOM because another process held 14.8 GB (`box_extract7b_restart.log`):
  a 7B fp16 run needs the whole V100.
- Lessons: binary output threw away the ranking; no severity, no hedging state, no silent state; Effusion and
  Synovitis worst (size wording and silent reports).

---

## 4. What is hard in the reports (30 read: 18 gold58 reports with the most v4 errors + 12 non-gold across 9 languages)

Gold conventions seen (gold = radiologist image labels; counts are report-level regex over gold58, MEASURED,
lab2 session, in-sample):
- Small effusion is half the time negative: "small / mild / minor / some" effusion n = 23, gold-positive .48 vs v4
  mean .90; moderate / large n = 11, gold .82.
- Low-grade cartilage is mostly not OA: chondromalacia I-II / superficial / mild chondrosis words n = 30, gold
  Medial / Lateral / PF OA .17 / .17 / .33 vs v4 .40 / .41 / .53.
- Partial / sprain / grade 1-2 ligament wording n = 26: gold ACL .42 vs v4 .62, MCL .27 vs .53.

But adjectives do not recover the image label: a size-graded effusion code with an a-priori order
(negated < silent < small < unqualified < moderate/large) scores Effusion **.7385** (.7727 with v4 tiebreak) vs v4
.8770 (MEASURED `lab2_effsize.out`): silent reports were 60% positive (n 5), unqualified 92% (n 12), small 46%
(n 26), moderate/large 82% (n 11). Severity wording is a weak proxy for the image call.

Hard cases (paraphrased):
- Report vs image: "lateral meniscus normal" with a gold LM tear (tr); a reported MM tear with gold 0 (en); a
  reported LM complex tear with gold 0 (en); medial OA gold 1 where the report says medial cartilage intact (en).
- Contusion vs degenerative edema vs fracture: OA-type subchondral marrow edema (tr, hr) gold Contusion 0; tibial
  plateau / subchondral impaction fracture with edema (en x2) gold Contusion 0; SONK-type insufficiency
  microfractures (el) gold Contusion 1, Fracture 0. Our ex16 manual note codes an insufficiency fracture as Fracture
  1, Contusion 0: an unresolved convention (n = 1 each). P(Contusion | Fracture) in gold58 = .556 (n 18) vs .225
  otherwise, so there is no exclusivity rule (MEASURED).
- Hedging: "possibly a minor partial rupture or post-distension state" of the ACL (hr) gold 0; "could be attributed
  to an incipient insufficiency fracture" (el) gold Fracture 0.
- Ligament grades: grade 1 ACL injury, interstitial ACL tear, low-grade partial MCL tear all gold 0; advanced mucoid
  ACL degeneration with swelling (nl) gold ACL 1. MCL "fluid around the MCL" (tr, de) is not a tear.
- Meniscus: degeneration with extrusion or subluxation (el, nl) gold tear 1; "grade II degeneration" (tr) v4 .42;
  Bulgarian "type 3 lesion" = tear; post-meniscectomy re-tear.
- Baker: millimetric (tr) and 3.5 cm (de) cysts; the de one is gold 0 although all 5 readers say yes.
- Compartment mapping: "medial facet of the patella" is PF, not the medial compartment.
- Structural traps (MEASURED, all 4,407): 27 studies carry two reports ("[BILATERAL NOTE: two reports filed under
  this study]", 1 in gold58); one Bulgarian report describes a shoulder (subscapularis) under a knee study; 15
  reports < 80 chars; **305 of 321 Greek reports write mu as the micro sign U+00B5** (regex and tokenizer hazard).
- Token cost (MEASURED, Qwen2.5 tokenizer, 60 reports per language): median tokens el 779 (1.17 chars/token),
  hr 465, fr 434, bg 432, en 347 (4.10 chars/token), nl 302, es 296, de 272, tr 210. 99% of reports < 3,102 chars.

---

## 5. Own image teachers as the label lever (`~/jump2/lab2_selfteach.py`, `.out`)

Recipe = mix50's: logit-match the teacher to the text table, then (1 - w) * text + w * teacher, w = .5;
"SA" = w .75 on pilk-UNK (report-silent) cells (the pre-existing mixSA rule, not tuned here). Gold58 uses the
fold-0 checkpoints' OOS gold predictions; parent-likeness = mean per-label Spearman vs the raptor proxy on the 867
fold-0 val studies (OOF of the same runs). All MEASURED.

| label set | gold58 | parent-likeness |
|---|---|---|
| v4 (text only) | .8927 | .645 |
| mix50_logit file | .9262 | .864 |
| v4 + wd (mix50 rebuilt) / SA | .9269 / .9305 | .865 / .875 |
| v4 + own TTR v4 / SA | .9063 / .9063 | .837 / .844 |
| v4 + own TTR v4 + F1 v4 / SA | .9083 / .9096 | .843 / .852 |
| v4 + own TTR mix50 / SA | .9251 / .9272 | .860 / .869 |
| v4 + own TTR mix50 + F1 mix50 / SA | .9268 / .9291 | .864 / .874 |
| v4 + wd + own TTR v4 / SA | .9228 / .9270 | .864 / .875 |
| **v4 + wd + own TTR mix50 / SA** | .9301 / **.9340** | .873 / .884 |

Reading:
- One own fold-0 TTR mix50 model is as good a teacher as Dread's wd (.9251 vs .9269); both together + SA = .9340,
  the best gold58 label set we have that was not tuned on gold58.
- Caveats: wd's checkpoint was picked on a gold holdout (Dread, forum CLAIMED), so anything containing wd is
  gold-inflated; own teachers are clean OOS. Only fold-0 teachers exist today; a 5-fold mean should be stronger
  (MODEL). Gold58 gains on labels translated into model gold58 at about 1-1.5x so far (labels v4 -> mix50 +.0335,
  models F1 +.053, TTR +.035, MEASURED), but LB transfer of label gains is weak (K2 blend .943 = parent, MEASURED;
  Tucker / Nicolai nulls, CLAIMED).
- Parent-free labels are not decorrelated labels: .852 vs .864 (MEASURED). Models trained on v4 labels are less
  parent-like (F1 v4 .840 vs F1 mix50 .889, MEASURED `oofx.log`); a v4 + own-v4-teacher model would sit in
  between (GUESS ~.86).

---

## 6. KX: a better extraction, if we ever run one (`scratchpad/jump2/kx_pilot.py`, box `~/jump2/kx/kx_pilot.py`)

### 6.1 Output and scoring
- Per study x label, two next-token probes on a shared KV-cached prefix (system + definitions + report):
  - Q1 state: A present (definite or probable) | B absent or normal/intact | C uncertain / possible / cannot exclude |
    D not mentioned -> pA pB pC pD.
  - Q2 grade (given present): 1 low (trace, minimal, small, mild, grade I-II, sprain, low-grade partial, superficial) |
    2 moderate (grade III, high-grade partial) | 3 high (large, severe, complete, full-thickness, grade IV, displaced).
  - Optional audit (`--spans 1`): greedy verbatim evidence quote (<= 12 words) for cells with pA + pC > .5, checked
    as a substring of the report (hallucination rate).
- **Pre-registered score** (fixed in the file before any gold run; never tune on gold58):
  `s = 0 * pB + .15 * pD + .45 * pC + pA * (.60 * g1 + .85 * g2 + 1.00 * g3)`.
  Order a priori: absent < silent < uncertain < present-low < present-moderate < present-high; gold58 rates agree with
  silent above negated (SIL 13% positive vs NEG 4%, MEASURED). Raw probabilities are saved so other mappings can be
  studied on ex16 / non-gold only.
- Why probes, not JSON generation: soft scores for free (binary costs .074 on v4, MEASURED); no parse failures by
  construction; no decode loop.
- Multilingual handling: NFKC + micro sign -> Greek mu; laterality synonyms for 9 languages in the system prompt;
  negation list; a-priori radiology definitions (degeneration without tear is not a tear; degenerative subchondral
  edema is not a contusion; patellar facets belong to PF; Baker is not a parameniscal cyst or ganglion; MCL bursa
  fluid alone is not a tear); whole text incl. both reports of bilateral studies; 3,500-char cap.
- Plumbing test (MEASURED 10-05): `--tiny_test 1 --limit 12 --spans 1` on CPU with a random 2-layer Qwen2 model and
  the real tokenizer: all code paths ran (prefix cache, 24 probes per report, quotes, gate printout); the Qwen chat
  template extends cleanly for the second turn (checked). Scores are meaningless by design (gold58 .47).

### 6.2 What the box has (MEASURED 10-05 ~09:07)
- GPU: one Tesla V100-FHHL-16GB (sm70: no bf16, no FlashAttention-2). Fleet chain holds 3.3 GB at 100% util at
  09:07; at 09:42 a second compute process holds 8.9 GB more (12.2 GB used, an empty `~/jump2/gpu.lock` exists; not
  this track's job). So even a 7B nf4 (~5.5 GB) does not fit right now: the pilot needs an idle window.
- Envs: `rsna` and `kmimic`: torch 2.5.1+cu121, transformers 4.48.0, bitsandbytes 0.50.2, accelerate 1.14, peft 0.20.
  `ltx-v100`: torch 2.6.0+cu124, transformers 5.17. Base: torch 2.14+cu126, transformers 5.17.
  **No vllm, auto-gptq, autoawq or flash-attn in any env.**
- Internet: `curl -I https://huggingface.co` -> HTTP/2 200 within the 10 s timeout (no download made).
- LLM weights on disk: `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct` (15 GB, 4 safetensors shards).
  Not usable: `~/arc-prize-2026/models` (NVARC Qwen3-4B with a custom 94-byte ARC vocab, 7 GB) and
  `~/ComfyUI/models/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` (15 GB, a ComfyUI
  text-encoder export, not a chat checkpoint; GUESS not loadable by transformers as is). No other LLMs found.

### 6.3 Candidate local models (V100 16 GB)

| model | on disk | memory | notes |
|---|---|---|---|
| **Qwen2.5-7B-Instruct** | yes | fp16 15.2 GB (whole GPU, MEASURED) or nf4 ~5.5 GB (GUESS, fits beside the fleet) | first choice for the pilot: zero download, measured baseline .8067 with a naive binary prompt |
| Qwen3-8B | no (16 GB download) | int8 ~9 GB / nf4 ~6 GB | needs transformers >= 4.51: make a NEW env, never change `rsna`; better multilingual (GUESS) |
| Qwen2.5-14B-Instruct | no (~30 GB download) | nf4 ~9-10 GB | ~2x slower; only with the GPU free (GUESS) |
| Aya-Expanse-8B | no | nf4 ~6 GB | multilingual focus (Greek, Turkish, Dutch, German, Spanish, French), weaker on hr/bg (GUESS); CC-BY-NC |
| skip | | | Gemma 3/4 on V100 (bf16-trained, fp16 overflow risk, GUESS); anything >= 24B; AWQ/GPTQ (kernels not installed) |

Forum (CLAIMED): Scott used local Gemma 4 (~.89 gold), Raymond Qwen2.5-32B ("cheap and expensive LLMs give about
the same labels"), Tucker a simple Qwen prompt (~.89 gold).

### 6.4 GPU-hours for 4,407 reports (MODEL, anchored to the MEASURED 1.7 s/report of the old 110-token run)
- Tokens per report: ~500 shared prefix + 12 x (~90 Q1 + ~75 Q2) ~= 2,500 (MODEL from the tokenizer counts).
- Qwen2.5-7B fp16, probes one by one: ~0.8-1.2 s/report -> **1.0-1.5 GPU-h**; with the 12 label probes batched in
  one forward: ~0.3-0.5 s -> 0.4-0.6 h.
- 7B nf4 beside the running fleet: ~1.5-3 h, and it slows the fleet chain (GUESS 10-30%).
- 14B nf4: ~2-4 h. Evidence quotes for all 4,407: +2-3 h (skip; audit gold74 only).
- Gold74 pilot: ~1-2 min compute + 1-3 min load.
- Old style (JSON with spans for every label, ~300 output tokens): ~5 s/report -> ~6 h.
- CPU fallback: 7B fp32 needs ~30 GB RAM, ~60-90 s/report -> gold74 in ~1.5 h; not while the fleet trains (it
  would evict fleet_store from the page cache).

---

## 7. Pilot gate (pre-registered; nothing run on a GPU)

Command (needs Eesh's OK and a GPU with >= 15.5 GB free, or `--load4bit 1` beside the fleet if he accepts slowing it):

```
cd ~/jump2/kx && HF_HUB_OFFLINE=1 ~/miniconda3/envs/rsna/bin/python kx_pilot.py \
   --model Qwen/Qwen2.5-7B-Instruct --device cuda --ids gold74 --spans 1 --out ~/jump2/kx/gold74_qwen25_7b.csv
```

It scores the 74 gold reports (58 gold + 16 ex16) and prints the gate (1,000 paired bootstraps by study).

| bar | requirement | reference (MEASURED) |
|---|---|---|
| G0 plumbing | 74/74 scored, no NaN; verbatim quotes >= .95 of quoted cells; fp16 <= 1.5 s/report | - |
| G1 reading fidelity | ex16 macro >= .975 | v4 .9808, mix50 .9692 |
| G2 text-only | gold58 >= .905 (v4 + .012) AND P(>v4) >= .95 AND no label below v4 - .03 | v4 .8927 |
| G3 second reader | rank-mean(KX, v4) >= .901 (v4 + .008) AND P(>v4) >= .90 | best existing 2nd reader +.0042, P .84 |
| G4 teacher mix | mix(KX, wd) logit .5 >= .934 AND P(>mix50) >= .90 | mix50 .9262, v4 + wd rebuilt .9269 |

Decision:
- Any of G2 / G3 / G4 passes (with G0, G1): run all 4,407 (~1-1.5 GPU-h) in an idle window, rebuild the text half
  (and the Archit-round mix), then only a retrain can show LB value (TTR fold-0 ~4.8 h alone; all-data ~6-7 h, MODEL).
- All fail: close the extraction lever in HANDOFF; no prompt iteration on gold58.
- Anti-overfit: prompt and score weights are frozen in `kx_pilot.py` as written today; one shot on gold58. Any
  iteration uses ex16 and non-gold reports only, then a single re-check. If a second gold58 look is ever needed,
  split gold58 by sorted UID into fixed halves A (first 29) / B (last 29), iterate on A, report B only.
- Priors (GUESS, from section 0): P(G2) 5-10%, P(G3) ~10%, P(G4) ~15%.
- When: the first idle GPU window. The TTR mix50 fold chain should end ~03:30 10-06 (MODEL from 4.8 h/fold); a
  gold74 pilot fits in 10 minutes of it.

---

## 8. Recommendations (ranked)

1. **Archit round from the 5-fold TTR mix50 OOF** (labels CPU-only once the chain ends ~03:30 10-06): text = v4;
   teacher = logit-matched mean(wd, own OOF); SA weights (.75 on pilk-UNK cells, .5 elsewhere). Before any retrain,
   score it on gold58 with the 5 fold checkpoints' gold74 predictions (CPU, same path as `gold_infer.py`, ~5-10 min
   per checkpoint at 12 threads, MEASURED for fold 0). Bar: >= .934 and >= mixSA_logit .9300 + .003. Then one
   retrain. Public gain GUESS +0 to +.004 in the blend; overlaps the sibling's rank-3 item in `topsolutions.md`.
2. **KX pilot as a closure test only** (section 7): ~10 min GPU in an idle window, no training displaced. Expected
   public gain GUESS ~0 (P(any bar) <= 20%); its value is closing the question with our own number.
3. **Do not**: re-extract 4,407 reports without passing the gate; tune per-label mix weights or report rules on
   gold58 (58 studies; SpeedSci gold .945 -> LB .910, CLAIMED); build parent-free labels hoping for decorrelation
   (.852 vs .864, MEASURED).
4. **Hygiene for any future label build**: never write gold into a label table (5 of our historical tables are
   leaky); normalize U+00B5 in Greek; handle the 27 two-report studies and the odd non-knee report; keep labels soft.

---

## 9. Files

- Box (`~/jump2/`): `lab2_decomp.py/.txt/.json`, `lab2_v4err.csv`, `lab2_selfteach.py/.out`, `lab2_reader2f.py/.out`,
  `lab2_lang.py/.out`, `lab2_effsize.py/.out`, `lab2_misc.py`, `kx/kx_pilot.py`, `kx/tiny_test.csv` (plumbing only).
  Earlier sibling outputs used here: `label_eval.txt/.json`, `labx.log`, `oofx.log`, `gold_infer.log`, `gold74_preds/`.
- `lab2_reader2.py` (slow duplicate of `lab2_reader2f.py`, its ssh session timed out) ran on at nice 19 and has
  since exited on its own with an empty `lab2_reader2.out`; superseded by `lab2_reader2f.out`. Nothing was killed.
- Laptop: `scratchpad/jump2/labels.md` (this file), `scratchpad/jump2/kx_pilot.py`, `scratchpad/jump2/lab2/` (copies).
