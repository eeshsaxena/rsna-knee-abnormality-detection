# Lever hunt 2026-10-12 ~11:00 IST (Claude, independent; nothing below is submitted without its own prereg)

## What the public frontier is (survey of the top 60 public notebooks, CLI + pages)

- Top public = heliosli H **.956**; bliverrigpdoge velci-kcd25-20 **.955**; the many ".954" notebooks.
- **Every top public notebook uses the same two model families**: nartaa Raptor CoAtNet (OAI-trained, .949) and heliosli's
  ConvNeXt-small "S6" compact bundle (heliosli/rsna-knee-compact-v1). They differ only in weights / extras.
- The only other public family is goodpjw2008/rsna-knee-2-5d-convnext-reader (2.5D ConvNeXt MIL). Adding it gives .954
  (romantamrazov DINOsaur-the-last V24, goodpjw2008), i.e. it does NOT beat H. Not a lever.
- So above .956 the only diversity available to us is **our own models** (ConvNeXt-T TTR fleet, T123 OAI set, PROBE4,
  D2 / D3 seeds). That is exactly what HF tests.

## Ranked levers

| # | Lever | Cost | Expected | Status |
|---|---|---|---|---|
| 1 | **HF20 / HF30** = H + our fleet (set D) | built, gated 16/16 | the key unknown: does an independent family add on top of H? FN11 gave +.002 over N alone | waiting on HV wall time (PREREG rule) |
| 2 | **HF with a stronger F**: K14 fleet (T123 OAI set replaces / joins D), and / or C+D+E+E2 if runtime allows | rebuild + prereg | +0 to +.001 over HF-D if F helps at all | K14 gate running; decide after HF reads |
| 3 | **S6 coverage**: if HV's hidden wall is near 8.3 h, S6 falls back to Raptor ranks on part of the test (deadline NB_T0 + 8.3 h). Then any time we add (F) costs S6 coverage, and a faster S6 (their `--compile` flag exists in the worker) would be a free gain | read HV wall first | 0 if S6 finishes early; real if it hits the deadline | HV pending |
| 4 | **Raptor second checkpoint** (nartaa dataset also ships the .945 r224 c08 SWA weights): a 3-way H' = Raptor384 + Raptor224 + S6 | ~25 min runtime, small build | small (two Raptors are highly correlated) | idea only |
| 5 | **S6-replicate on the box**: the S6 training code (compact_train.py, 230 KB) is public inside the H notebook, but its label tables (report-state targets) are not; would need px caches + our own targets | multi-day engineering, GPU busy | potentially large (S6 is the best partner for Raptor) but high risk with 10 days left | parked; revisit only if HF shows our F adds nothing |
| 6 | **Non-OAI pick** for the final two, in case the host bans OAI (748276): best clean = K7i main .946 (scored) | none now | insurance | decide when the host answers |
| 7 | D2 / D3 extra seeds in F | already trained | +.0001-.0002 (MODEL) | only through a fleet prereg |

## Decision rules (fixed now)
- If HF20 or HF30 scores > HV on the 3-dp display: F helps on top of H -> lever 2 (stronger F) is the main line.
- If both HF score <= HV: our F adds nothing over H on public LB -> stop investing in more seeds of the same family; keep
  HV / HF as picks, spend the box on PROBE4 / T123 only if they can become a DIFFERENT family, and reconsider lever 5.
