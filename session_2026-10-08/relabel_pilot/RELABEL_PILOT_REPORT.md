# Relabel pilot (VLM, item 5): report, 2026-10-08 21:10 IST

## Status: BLOCKED at the first live call by an invalid API key. Spend $0.00.

- **What failed:** the Stage A batch submission at 20:44 IST was rejected for both arms with HTTP 401 "API key is invalid".
  - No batch was created and no tokens were processed, so nothing was billed.
  - Log: `attempt_2026-10-08_2044_401.log`.
- **Why (checked with true/false only, the key was never shown):**
  - The only `ANTHROPIC_API_KEY` the agent can see is the one in the Claude Code process environment. The runbook predicted this: a Claude Code shell carries the session's own credential, which is not a valid key for direct API calls.
  - No other copy exists: none in the Windows user or machine environment, and none in the bash or PowerShell profiles.
  - A new Terminal-panel tab cannot see a key exported in Eesh's own tab.
- **So:** the gate numbers, S, the per-label deltas and the verdict do not exist yet. The scale-up plan (step 5) depends on S, so it is not written.

## Done before the block

1. **Gate amendment registered before any live call** in `prereg_v2.json`:
   - sha256 `6ce3b3c50b608fa3c4e10423469b8fccdffbc94db2913931a6694491c5905352`, written 2026-10-08 20:39:27 +0530; the same sha is in `prereg_v2.sha256`.
   - The prompt is unchanged (prompt sha `c5a72790...` = prereg_v1).
   - **Amended gate:**
     - G1 (Effusion img-text) goes from a +.040 hard veto to a sign check, >= 0.
     - S (mean over 6 targets of AUC(0.5 logit mix50 + 0.5 logit img) - AUC(mix50) >= +.010) makes the spend decision.
     - Secondary: the 5-target non-Effusion img-text mean, reported with its bootstrap SE and 90% CI.
     - The validity rules are unchanged (>= 55/58 paired, 1 sha, 1 model, 1 effort).
     - The v1 G2 per-label -.020 tolerances are printed as information only (my reading of "S decides"; flagged for Eesh).
   - **Decision:** INVALID if V fails, GO if G1-sign and S pass, otherwise NO-GO. The v1 verdict is printed alongside.
   - `pilot_eval.py --prereg prereg_v2.json` implements this. Without `--prereg`, the v1 path is unchanged and the selftest still passes. The original is kept as `_pilot_eval_v1_orig.py`.
   - **Pre-registered spend staging, $8.00 hard cap:**
     - Stage A: the first 10 gold58 studies, both arms.
     - After that: projection = actual + 1.3 x measured $/request x requests left.
     - Scope: the rest of gold74 if the projection fits the cap; else the rest of gold58 only; else stop. No model switch.
     - Chunks of <= 20 per arm when the projection is above $6. A failed request is retried at most once.
2. **Kit checked.**
   - The laptop PNGs and manifest are byte-identical to the box (871 PNGs, same combined sha256), and all scripts match the box.
   - `selftest_offline.py`: all 6 tests PASS.
   - **Dry run (no network), gold74, Opus 5.5:**

     | Arm | Batch | Standard |
     |---|---|---|
     | text | $1.67 | $3.34 |
     | img | $2.12 | $4.25 |
     | **Total** | **$3.79** | **$7.59** |

     Batch is under the cap, so the model stays `claude-opus-5-5` (recorded in prereg_v2). The output tokens (1,800 per request) are a GUESS, which is why the staging rule exists.
3. **One-command runner: `run_pilot.py`** (sha `faeeb249...`).
   - Before any spend it checks:
     - the prereg_v2, prompt, relabel_call.py and pilot_eval.py hashes;
     - the key, with ONE free `models.retrieve` call, so a bad key fails before anything is billed.
   - It then:
     - runs Stage A, polls every 5 min, fetches, tallies the actual spend from usage, and applies the staging rule;
     - runs `pilot_eval.py --prereg prereg_v2.json`;
     - copies everything into this folder.
   - It is resumable after Ctrl-C.
   - Offline test `_test_run_pilot.py`: 7/7 PASS. Covered:
     - cap 8 runs all of gold74;
     - cap 3.5 runs gold58 only;
     - cap 2 stops after Stage A;
     - chunking;
     - a bad key sends no POST;
     - a cap above $8 is refused;
     - the fake key never appears in any output.

## To run it (Eesh, in your own terminal, ~5 min of your time)

The full kit is copied here, under `kit/` (verified identical to the Temp copy, which may get cleaned).

```
cd C:\Users\eeshs\Downloads\rsna-knee\session_2026-10-08\relabel_pilot\kit
$env:ANTHROPIC_API_KEY = '<your key>'      # this terminal only
python run_pilot.py
```

- If it says CLAUDECODE is set, add `--allow_agent_shell`.
- **Expected cost:** ~$3.8, with the $8.00 cap enforced in code.
- **Expected time:** two batch cycles, usually under an hour each, up to 24 h worst case. Keep the laptop awake.
- **Outputs** land in `kit/results/` and are copied to this folder:
  - `soft_{text,img}_opus-5-5.csv`, `raw_*.jsonl`, `batches_*.json`;
  - `spend.json`, `run_pilot.log`;
  - `pilot_eval_v2.log`, `pilot_eval_opus-5-5_v2.json`.
- After that, Claude reads `pilot_eval_v2.log`. If the decision is GO, it writes the scale-up plan; that spend needs your separate OK.

## Files here

| File | What it is |
|---|---|
| `prereg_v2.json` + `prereg_v2.sha256` | the registered amendment |
| `pilot_eval.py` | v2 support added |
| `_pilot_eval_v1_orig.py` | the original v1 evaluator |
| `run_pilot.py` | the one-command runner |
| `spend_tally.py` | standalone spend tally |
| `_test_run_pilot.py`, `test_run_pilot_offline.log`, `selftest_offline.log` | tests and their output |
| `dry_run_both_arms.log` | the dry-run cost estimate |
| `attempt_2026-10-08_2044_401.log` | the failed submission |
| `SHA256SUMS.txt` | file hashes |
| `kit/` | the full runnable kit |
