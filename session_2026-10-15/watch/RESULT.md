# Public-release watcher + swap builder (RESEARCH_960 lever 3). Built 10-14 ~12:50-13:40 IST by a Claude subagent.

Nothing was pushed, committed or submitted. No dataset was created or versioned. The university box was not touched.
Everything that talks to Kaggle here only reads. No em dashes.

## Result in one block

| item | result |
|---|---|
| Watcher runs (3) | baseline 13:15 IST (386 notebooks, 732 s), incremental 13:29 and 13:33 IST (109-111 re-reads, ~160 s each). 0 errors, 0 web errors |
| Best public notebook score | **.956** (24 notebooks at .956, all on public inputs: 22 use R7 bundle + orthofoundation + nartaa swa weights, 1 adds goodpjw2008's reader, 1 uses r6 instead of R7). **No notebook >= .957, so no RELEASE_CANDIDATE.** |
| heliosli / nartaa datasets | no new dataset and no new version since 10-11. r7cnxfsss v1 (10-11 19:59 UTC), orthofoundation v1, nartaa swa-weights v1, compact-v1 v13, r6cnxfsss v2 |
| Alert raised (run 2) | `jingshuifu/rsna-knee-flip-tta` got its public score: **.956**. It is heliosli's R7 notebook with only flip TTA forced on (cell diff below). That is a free third-party read for lever 2: flip on the R7 base stays inside the .956 display band |
| Pending free reads | jingshuifu `rsna-knee-a224-flip` (= V224 + flip, the same one change as our V224-FLIP), `rsna-knee-a224-w30` (= V224 with R7 .7 / Raptor .3), `rsna-knee-a224-blend`, `rsna-knee-w30`, and rudrpratapsingh `four-public-models-one-rank-blend` (V224 + compact-v1). All ran today, all unscored so far. The watcher re-reads them every run for 72 h and alerts when a score appears |
| Swap-builder dry run on varshitha | **SWAP_V224_REPRO PASS 4/4.** Notebook sha256 `4259ec97...` and kernel-metadata sha256 `dea8a179...` are byte-identical to `session_2026-10-14/v224/kernel_v224` on disk AND to a fresh `build_v224.py --target v224` rebuild. Build gate: 24 OK, 1 WARN (the slug `rsna-sub-v224` already exists), 0 FAIL |
| Gate tests on real commit outputs | swap_check: V224-exact vs her commit **PASS 17/17** (0 of 36 cells differ, stage csvs within 1e-3). Negative controls fail as they should: v224ctl as clean FAIL 8/11, V224-FLIP vs her FAIL 12/13 (2 of 36 cells differ, max .167). v224ctl with `--expect-fired` PASS 12/12 |
| Generic canary simulation | **SIM_CANARY PASS 13/13** (clean, fired, undefined trigger, wrong ids, NaN, receipt fallbacks, missing sample) |
| Builder negative tests | dataset version drift FAIL (heliosli old-dataset: r6 pinned v1, current v2), trigger name not in their code FAIL, scored v1 of a v3 notebook pulled correctly with the pinned input list |

## Files (all in `session_2026-10-15/watch/`)

| file | what |
|---|---|
| `watch.py` | the watcher (state, log, browser fallback, dataset watch, `--loop-hours`) |
| `kgweb.py` | read-only helpers: the kaggle python API (same 2.2.4 package as the CLI) + anonymous Kaggle page endpoints |
| `swap_builder.py` | public notebook -> private canary-armed copy, ready to commit-test; `--verify-v224` proof mode |
| `swap_check.py` | commit gate for any swap kernel (clean mode, control mode, vs the public notebook's own commit output) |
| `sim_canary.py` | offline test of the generic canary cell (runs only our generated cell, never their code) |
| `watch_state.json`, `WATCH_LOG.md` | machine state (387 notebooks, inputs seen, dataset versions, dataset-id map, alerts, runs) and the human log |
| `browser_todo.json`, `browser_scores_1014.json` | browser fallback queue (empty) and the browser cross-check I did today |
| `pulled/` | `kaggle kernels pull -m` copies of 108 notebooks (31 MB; untrusted data, never executed or imported) |
| `dryrun_v224/` | the dry run: `kernel_v224/` (with `DO_NOT_PUSH.txt`), `build_v224.json`, cell diff, `src/`, `verify_v224_rebuild/` |
| `tests/` | logs of every builder / gate / simulation test listed above |
| `intel/` | cell diffs of jingshuifu's a224-flip, a224-w30, flip-tta and rudrpratapsingh's four-model blend |
| `watch_run{1,2,3}.out`, `dryrun_v224.log` | raw run output |

## watch.py: what one run does

1. **(a) Lists** the competition's public notebooks by public score (`kernels list --competition ... --sort-by scoreDescending`,
   3 pages of 100) and by recency (`--sort-by dateRun`, 2 pages), plus every competition notebook of heliosli, nartaa,
   varshithakolupuri and jingshuifu. The python API call is the same as the CLI call (the CLI table has no score column).
2. **(b) Reads** each notebook not seen before, each one whose run time changed (a new version: after the 10-15 23:59 UTC
   new-notebook lock, a release can only come this way or as a dataset bump), each unscored one that ran in the last 72 h,
   the current top 60 by score, and the key authors' notebooks. For each one:
   - **public score**: from Kaggle's own notebook page model (`GetKernelViewModel`, read anonymously: `bestPublicScore`, the
     scored version, the current version). It is the same number the page shows: 6 notebooks cross-checked in the built-in
     browser with the anchor method (navigate, wait 4.5 s, anchor texts `0.9xx Vn`): 4 scored ones agree exactly, 2 unscored
     ones agree on "no score". If the page model cannot be read, the ref goes to `browser_todo.json` (with the JS snippet);
     `--browser-scores FILE` merges browser reads back.
   - **inputs**: `kaggle kernels pull REF -m` (same API call, kept in memory so Windows text mode cannot rewrite newlines) for
     every notebook with score >= .955, an unscored one that ran in the last 7 days, or a key author. The inputs that count are
     the pinned data sources of the SCORED version's run (ListKernelVersions when the scored version is not the current one).
     Each input is tested anonymously: dataset public = GetDatasetBasics 200 (private shows as 404 or as a bare
     `private-data-source` mount); its current version id vs the version id the scored run pinned (drift); kernel sources
     public; models visible. Bare-slug mounts (seen on some notebooks) are resolved through the page's renderable URLs, a
     dataset-id map and the pulled metadata, else counted as not public.
3. **(c) Flags**: `RELEASE_CANDIDATE` = score >= .957 AND every input public AND only this competition's data. It prints
   `WATCH_FLAG`, appends to `RELEASE_FLAG.txt` with the exact swap_builder command, and the run exits 10. Also:
   `HIGH_SCORE_NOT_COPYABLE`, `SCORE_APPEARED`, `SCORE_UP`, `NEW_INPUT` (a dataset no public notebook used before, on a .955+
   or unscored notebook), `KEY_AUTHOR_NEW_VERSION`, `KEY_AUTHOR_NEW_NOTEBOOK`, `BROWSER_KVM_MISMATCH`.
4. **(d)** Writes `watch_state.json` and appends one section to `WATCH_LOG.md`: counts, score bands, flags, alerts, the top 15
   with inputs and drift, pending free reads, watched datasets.
5. **Datasets**: every dataset of heliosli and nartaa (`datasets list --user`), plus the RESEARCH_960 baseline set
   (dreaddevelopment/raptor-knee-widedense v2, jessky/rsna-knee-local-cv-cnx50 v1). Alerts: NEW_DATASET,
   DATASET_VERSION_BUMP, DATASET_SIZE_CHANGE, DATASET_GONE.

Findings worth knowing from run 1 (MEASURED):
- **Drift is real**: `heliosli/rsna-knee-blend-gold-old-dataset` (.956) scored with r6cnxfsss v1. That dataset is now at v2,
  so a CLI copy would run on v2 and would not reproduce the .956. The builder FAILS on this unless `--allow-drift` is given.
- 4 of the 24 .956 notebooks have a scored version that is older than the current version (heliosli old-dataset v1/v2,
  karttikjangid05 v1/v2, lavinwins v1/v3, medvax v1/v2). lavinwins' current v3 uses different datasets from its scored v1.
  A plain `kernels pull` would copy the wrong code; the builder pulls the scored version.
- No public notebook above .956 exists; every .956 notebook runs on the same public weights (R7 bundle or r6, orthofoundation,
  nartaa swa weights; one adds goodpjw2008's reader).

## swap_builder.py

`python session_2026-10-15/watch/swap_builder.py OWNER/SLUG[/VERSION] [--name rsna-sub-XXX] [--canary-expr EXPR]... [--canary-col Fracture]`

1. Reads the page model (score, scored version, pinned inputs, the public commit's runtime and docker digest). The version
   copied = `--version`, or `/N`, or the SCORED version, never silently the latest. (The CLI's `SLUG/N` form returns 403 on
   this API, MEASURED; `version_label 'vN'` returns version N: lavinwins v1 514,559 bytes vs v3 152,191.)
2. Pulls exactly that version into `src/` and gates the inputs: this competition's data only; each dataset public and its
   current version == the pinned version (a push attaches the LATEST version, so drift FAILS); kernel sources public; models
   visible; internet off (forced off in ours); docker pin kept (warns when not pinned or when it differs from the run's
   digest); 2 x T4.
3. Copies the notebook VERBATIM and appends ONE canary cell. The JSON serialisation is detected by an exact round trip (8
   candidates), so the proof is byte for byte: dropping the appended cell gives back their file.
4. Canary. **Profile v224** (detected by build_v224.py's 11 structure asserts): appends build_v224.py's `CANARY` text exactly,
   read from that file with `ast`, not retyped. **Generic profile** (any other notebook): `--canary-expr` triggers (each name
   must appear in their code, else FAIL; a trigger that is true OR undefined at run time fires) blank `--canary-col` to 0.5;
   always-on checks raise when submission.csv is missing, has wrong columns or ids, or holds a value that is not finite in
   [0, 1]; constant columns and per-study fallback counts in `*receipt*.json` are logged. Prints CANARY_CLEAN / CANARY_FIRED
   and SWAP_DONE, writes canary_receipt.json.
5. Metadata = theirs with build_v224.py's edits (our id, title, code_file, private, internet off, id_no and keywords
   dropped); dataset list = the scored run's pinned list (their order kept when the sets match).
6. Report `build_<suffix>.json`; a cell diff against a reference notebook (default: the V224 source); a fallback audit (46 lines
   on V224) for writing triggers; a WARN when our slug already exists; `DO_NOT_PUSH.txt` when a gate fails or in proof mode.
   It prints the push, gate and submit commands but does not run them.

### Diffs (what the builder inserts)
- Dry run: their 13 cells + **one appended cell = build_v224.py's CANARY (1,212 chars)**. Their cells are byte-verbatim (the
  proof gate) and the cell diff against the V224 source is 13 identical, 0 changed, 0 new. The resulting notebook is the
  same file as V224-exact (`4259ec97...`), so its inserted text is the one already reviewed in
  `session_2026-10-14/v224/celldiff_vs_her.txt`.
- Generic canary text: `GENERIC_CANARY` in swap_builder.py (about 3.6 kB plus 4 lines per trigger). Simulated in
  `tests/sim_canary.log`.

### Gate output (dry run, `dryrun_v224.log`)
```
OK  public page readable: score 0.956 (scored v1), current v1      OK  copied version ran COMPLETE (457 s commit, T4)
OK  pulled 615,181 bytes sha256 6406e923... (= build_v224 SRC_NB_SHA)
OK  competition data = ours only     OK  3 datasets public, pinned version id == current (20601735, 20440357, 20297367)
OK  metadata datasets == pinned (her order)   OK internet off   OK docker pinned 2757e0c7...   OK NvidiaTeslaT4
OK  serialisation reproduced: compact, ensure_ascii False   OK profile v224: all 11 structure checks hold
OK  canary compiles (1212 chars)   INFO cell diff vs V224 source: 13 identical
WARN our kernel slug already exists: a push makes a NEW VERSION of eeshsaxena/rsna-sub-v224
OK  verbatim: dropping the appended cell gives back their notebook byte for byte   OK 13 -> 14 cells
SWAP_V224_REPRO PASS 4/4    (disk + fresh rebuild; notebook 4259ec97..., metadata dea8a179...)
SWAP_BUILD PASS: 24 OK, 1 WARN, 0 FAIL
```

## Runtime
- Watcher: baseline 732 s (522 anonymous page reads + 107 pulls); incremental ~160 s (~165 reads, 9 list calls). Laptop CPU
  only, a few MB of network per run.
- A swapped kernel runs as long as its source notebook; the canary adds under 5 s. For the dry-run kernel (= V224-exact):
  hidden wall central 3.0 h (2.4-3.6), per the 10-14 runtime review. Commit test about 8 min (her public commit ran 457 s), so
  about 11-12 min of Eesh's GPU quota.

## Commands

Arm the watch (it dies with the session, so re-arm after every restart):
```
python session_2026-10-15/watch/watch.py --loop-hours 3        # run as a background task; or one-shot every 3 h
# any poller: test -f session_2026-10-15/watch/RELEASE_FLAG.txt   (one-shot runs also exit 10 on a new flag)
```
If a RELEASE_CANDIDATE appears (the flag line carries this command):
```
python session_2026-10-15/watch/swap_builder.py OWNER/SLUG/N --name rsna-sub-rel-XXX [--canary-expr "..."]
# read the fallback audit + cell diff, add a --canary-expr for every silent fallback, rebuild, then:
kaggle kernels list --mine --sort-by dateRun --page-size 10        # never more than 2 of our commits running
kaggle kernels push -p session_2026-10-15/watch/swaps/kernel_rel-XXX
kaggle kernels output eeshsaxena/rsna-sub-rel-XXX -p session_2026-10-15/watch/swaps/out_rel-XXX_v1
kaggle kernels output OWNER/SLUG -p session_2026-10-15/watch/swaps/her_rel-XXX       # their own commit output
python session_2026-10-15/watch/swap_check.py session_2026-10-15/watch/swaps/out_rel-XXX_v1 --her session_2026-10-15/watch/swaps/her_rel-XXX
# V224-family release: also python session_2026-10-14/v224/v224_check.py OUT --her HER
python session_2026-10-14/submit_at.py eeshsaxena/rsna-sub-rel-XXX 1 "REL-EXACT OWNER/SLUG vN verbatim + canary"   # only after PASS + a PREREG amendment
```
The dry-run kernel `dryrun_v224/kernel_v224` must NOT be pushed or submitted: it is byte-identical to V224-exact, which is
already submitted (57171955), and an identical resubmit is on the RESEARCH_960 do-not list. (Its exact command would be
`python session_2026-10-14/submit_at.py eeshsaxena/rsna-sub-v224 1 "V224-exact"`, the same kernel version already submitted.)

## Concerns
- The anonymous page endpoints (GetKernelViewModel, ListKernelVersions, GetDatasetBasics) are undocumented; fetch_api_lb.py
  relies on the same family. If they change, the score falls back to `browser_todo.json` and the built-in browser.
- The watcher and loop die with the session (MEASURED twice before). The main session has to re-arm it; extra polls are worth
  it on the evening of 10-15 before heliosli's new-notebook lock.
- The generic canary only knows the fallbacks it is told about. For a never-seen release, someone has to read the fallback
  audit and the cell diff and pass `--canary-expr` before the commit test. The builder WARNs when no trigger is given.
- Dataset drift is a hard FAIL by design. A heliosli "dataset bump" release (new version of an existing dataset, notebook not
  re-run) shows up as DATASET_VERSION_BUMP plus drift on every notebook that uses it. Then the old public scores no longer
  describe what a copy would run, so a rebuild needs its own commit test and a deliberate `--allow-drift`.
- `kernels pull` metadata for an older version returns the CURRENT metadata (MEASURED on lavinwins), so the builder takes
  inputs from the scored run's pinned sources instead. Kernel sources (utility scripts) attach their latest version on push;
  they are only reported, not version-checked.
- watch.py cannot run under `python -I` (the kaggle and requests packages live in user site-packages). The pulled notebooks
  sit in subdirectories, so they can never shadow an import, and nothing ever executes them.
- Free reads are weak: display truncation (3 dp) means jingshuifu's flip-tta .956 only says flip on the R7 base stays inside
  [.956, .957). It does not size lever 2.
