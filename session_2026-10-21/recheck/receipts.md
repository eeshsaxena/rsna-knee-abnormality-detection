# Receipt audit: A = swide 57464326, B = R6 57248084 (recheck, 10-21 ~23:30 IST, Claude)

READ-ONLY. Nothing submitted, pushed, committed, selected, downloaded or posted. No edits to PREREG / HANDOFF / PROGRESS / live
folders / scheduled tasks. CPU only (two local gate scripts re-run on the committed outputs already on disk; they write nothing).
One read-only Kaggle API call: `competition_submissions` (108 rows, 17:58 UTC). Tags: MEASURED (read or computed here),
CLAIMED (our older notes or Kaggle mechanics, not re-proven here), MODEL (arithmetic), GUESS.

Evidence files written here: `rerun_gate_swide.txt`, `rerun_gate_r6.txt`, `bytes_bands.txt`, `size_model.py`.

## 0. Verdict
- **No discrepancy found.** Every receipt the task lists is clean for both picks, both gates reproduce today, both notebook shas
  match their registries, and both hidden runs carry a full-size submission file (no blanked column). (MEASURED)
- **New, stronger proof than the score alone:** the Kaggle API `total_bytes` of the HIDDEN-run `submission.csv` (all public +
  private rows) puts swide (247,386) and R6 (247,294) inside the clean band (30 S6F-era entries, 247,192-247,521) and ~8.5 KB above
  the canary band (the three .916 runs, 238,686-238,746). (MEASURED, `bytes_bands.txt`)
- **Scores:** under 12-label macro AUC, any one constant-0.5 column caps the score at (11 + .5) / 12 = .9583. swide's .960 is
  therefore impossible with any canary column, V224 (Fracture) or S6F (ACL). (MODEL, exact bound; metric CLAIMED in HANDOFF.)
- Live API at 17:58 UTC: 57464326 .960 COMPLETE 247,386 B; 57248084 .958 COMPLETE 247,294 B; no new submission since 04:11 UTC 10-21. (MEASURED)

## 1. swide 57464326 (`eeshsaxena/rsna-sub-v224r6c4pw25lomm05mcl10j` v1)
Committed output: `session_2026-10-20/val/a22/out_v224r6c4pw25lomm05mcl10j` (visible commit run, 3 rows). All MEASURED.

| check | receipt / log | value |
|---|---|---|
| V224 canary | `canary_receipt.json` + log `V224_DONE` + `CANARY_CLEAN r224 used, no column blanked` | fired [], R224_OK true, r224_used true, flip forced, use_f false |
| R224 stage | log `R224_STAGE rc=0 ok=True` | ok |
| S6F | `s6f_receipt.json` + log `S6F_DONE` | mode S6F, fired [], run RAN rc 0, killed false, driver ok, coverage 1.0, n_unscored 0, base_blanked [] |
| R6 ours stage | `submission_receipt.json` + log `CONTROLLER_DONE` / `OURS_STAGE_COMPLETE` | status PASS, bundle_id fe055406fad6a12f, fallback_studies 0, worker exits 0 / 0 |
| blend | `blend_receipt.json` | raptor_failed 0, ours_failed 0, both_failed 0, mode blend |
| DEC | `dec_receipt.json` | ok true, pip_rc 0, preflight ok, 0 flagged |
| stack doses | log + S6F blend receipt | EOA_APPLIED a .25 pw, LOAMM_APPLIED .05, MCLMM_APPLIED .1, in order before S6F_BLEND_OK |
| failure markers in log | grep | Traceback 0, BLEND_FALLBACK 0, SKIPPED 0, V224FLIP_CANARY 0, S6F_NO_BASE 0; only "Error" = 4 benign `No module named 'torch_xla'` |
| receipt binding | sha256 | submission.csv 5b06fff9... == blend submission_sha256; submission_v224flip.csv dcaabfb7... == setup base_sha256 |
| gate | `gate_v224r6c4pw25lomm05mcl10j.txt` | PROBE_CHECK PASS 26/26 (= registry gate_pass); re-run today byte-identical (`rerun_gate_swide.txt`) |
| notebook sha | `val/slate_a22.json` entry swide vs `stack/kernels/kernel_v224r6c4pw25lomm05mcl10j` | nb c9801a80... MATCH, meta 6a2745e7... MATCH; file mtime 10-20 23:33 < push 10-21 09:02 |
| notebook constants vs receipts | cell 16 / 18 / 7 | DRIVER_SHA a2acdd1d == setup driver_sha256; A_OA .25, A_LOAMM .05, A_MCLMM .1 == receipt; BUNDLE_ID fe055406 == receipt |
| version | `push_*.log` + `tick_a22.log` + submits.log | pushed v1, gate on v1, submitted v1 (ref 57464326; API scriptVersionId 360633233) |

## 2. R6 57248084 (`eeshsaxena/rsna-sub-v224r6flips6fj` v1)
Committed output: `session_2026-10-15/r6entry/out_v224r6flips6fj_v1`. All MEASURED unless tagged.

| check | value |
|---|---|
| V224 canary | fired [], R224_OK true, r224_used true, flip forced; log `CANARY_CLEAN`, `R224_STAGE rc=0 ok=True` |
| S6F | mode S6F, fired [], run RAN rc 0, killed false, coverage 1.0, n_unscored 0, base_blanked [] |
| R6 ours stage | PASS, bundle_id fe055406fad6a12f, fallback_studies 0, worker exits 0 / 0 |
| blend | raptor_failed 0, ours_failed 0, both_failed 0 |
| failure markers | Traceback 0, BLEND_FALLBACK 0, SKIPPED 0, V224FLIP_CANARY 0; only benign torch_xla "Error" lines |
| receipt binding | submission.csv e1324ecc... == blend submission_sha256; submission_v224flip.csv 1ebfe53c... == setup base_sha256 |
| gate | R6_CHECK PASS 19/19 recorded in `R6_READY.txt` and PREREG A9 (line 269); re-run today PASS 19/19 (`rerun_gate_r6.txt`) |
| notebook sha | `build_receipt_v224r6flips6fj.json` out_sha256 dfee2ba3... == file on disk == PROGRESS_15 line 298 |
| version | submit_r6_at.py pins kernel_version=1; submits.log line 11 ref 57248084; API scriptVersionId 358682285 |

Small gaps (not discrepancies): R6 has no saved gate-output file (only the verdict line) and no metadata sha in a registry; the
re-run above closes the first. The pushed notebook bytes cannot be re-read without a download; the binding is the pre-push
sha guard (G3 in tick_a22 for swide; build receipt for R6) plus the constants check above. (MEASURED / CLAIMED)

## 3. Is .960 / .958 consistent with no canary in the hidden run?
1. **Canary cost (MEASURED):** the three 10-21 S6F-canary runs (r6pw20, spl, salt) all read .916 vs .959-.960 for their clean
   twins: a drop of ~.043, i.e. one column at AUC .5 when that label's true AUC is ~.99 (MODEL: (.99 - .5) / 12 = .041). A V224
   canary (Fracture blank) would cost a similar amount, both together ~.08 (CLAIMED cause.md: ~.88).
2. **Score bound (MODEL, exact):** one constant column caps macro AUC at .9583. swide .960 (>= .9595 if displayed by rounding,
   >= .960 if truncated) is out of reach with any column blanked. R6 .958 is not excluded by the bound alone, but would need the
   other 11 labels at mean AUC >= .9991 (impossible next to its .957-.959 twins), and its bytes are clean anyway.
3. **Bytes (MEASURED):** a canary writes `0.5` (3 chars) in place of ~10-char float32 ranks on every hidden row. Observed drop
   R6 -> r6pw20 = 8,548 B, swide -> salt = 8,681 B. `size_model.py` (MODEL, no ties) gives ~8.5 KB per blanked column at ~1.29k
   hidden rows, so one column exactly; two columns would be ~17 KB. swide and R6 are in the clean band. Because total_bytes covers
   the whole file, this speaks for the private rows too, not only the public split.
4. **Public = private run (CLAIMED, Kaggle code-comp mechanics):** one hidden rerun produces the file scored on both splits, and
   every canary path is whole-column, so a clean public read means a clean private file for all canary paths.
5. **What the receipts cannot rule out (GUESS, small):** silent partial paths with no byte signature: R6 ours-stage fallback on up
   to 50 % of studies (Raptor rank stands in; >50 % stops the notebook) and S6F rows unscored within the .98-1.0 coverage window
   (base rank stands in). swide reads .960 >= its twin pw25 .959, which argues against a material one; redteam put P <= .02.
   Scored entries are frozen, so this affects nothing that can still be changed.

## 4. Info (not a discrepancy)
- pydicom 3.0.2 in the DEC cell vs 3.0.1 in the S6F setup, same dist-packages path, in BOTH swide and R6 (the V224 stage installs
  a pydicom-3.0.1 wheel mid-run). R6 scored clean on hidden with the same pattern. (MEASURED)
- Registry shas also match on disk for r6pw20 (e4d92c0a, gate PASS 17/17 on file), pw25, spl, salt. (MEASURED)
- Suggestion for the 10-22 B rule: alongside "public >= .958 and not .916", require API `total_bytes` >= ~245,000 for any
  A23-R6PW20 copy (clean 247,192-247,521 vs canary ~238,700). It is cheap and catches a blanked column even if a score display were
  ambiguous. (MODEL)
