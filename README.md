# RSNA Knee Abnormality Detection (Kaggle, 2026): working repository

Team working folder for the RSNA Knee Abnormality Detection competition (12-label knee MRI, macro AUC, code competition, hidden
rerun on 2xT4 with a 9 h limit). This repository holds **code and notes only**: builders, gates, analysis scripts, kernel notebooks
and the day-by-day research log. Data, model weights, outputs and caches are deliberately not included (see "Not in this repo").

## Result

| | submission | public LB | private LB |
|---|---|---|---|
| Final pick A | `swide` 57464326 | .960 | .944 |
| Final pick B | `r6pw20` copy 3, 57523232 | .959 | .944 |

**Final private rank: 246 / 6,425, silver medal.** Public rank at the close was 157 / 6,396.

Solution writeup (figures, method, what did not work):
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/246th-place-silver-report-labels-are-not-expert
(source and figures in `writeup/`). Final notebook: https://www.kaggle.com/code/eeshsaxena/rsna-sub-v224r6c4pw25lomm05mcl10j

## What moved the score

1. **Base ensemble (c4):** Raptor CoAtNet r384 + r224 (rank mix .75 / .25), the heliosli compact ConvNeXt-S bundle (R6, 4 folds, flip
   TTA), a per-label rank blend table, and a .2 rank share of the S6fall model. Public .958.
2. **E-OA, OA decorrelation (the main lever):** the training labels come from radiology reports, but the hidden test is graded by
   experts reading the images. Report-trained models tie the three OA compartments (Medial, Lateral, Patellofemoral) together far
   more than expert graders do. Decorrelating the three OA columns on the final output,
   `out_t = pct(pct(f_t) - a/2 * sum_o w(t,o) pct(f_o))`, was validated on 2,399 expert-read OAI knees (MOAKS) with three OAI-free
   models before any submission, then confirmed on the leaderboard: c4 .958, then eoa10 .959, eoa20, pw30 (pair-weighted), then
   the `swide` stack .960. On the private board E-OA held: every E-OA entry scored .944, the plain c4 / R6 bases .943.
3. **Small stacked tweaks on top** (`swide`): pair-weighted E-OA at a .25, Lateral OA decorrelated from Medial Meniscus (.05) and MCL
   decorrelated from Medial Meniscus (.1).

Lessons: offline rulers (gold58, coupling-matched OAI) underestimated the E-OA effect on the public board by 2-5x; weight tweaks
picked on the 58 hand-labelled studies were consistently negative on the leaderboard; every entry was pre-registered
(`session_2026-10-15/PREREG_10-15.md`) and gated against its committed output before submission.

## Layout

- `writeup/`: the Kaggle solution writeup, its figures and the scripts that draw them.
- `HANDOFF.md`: running handoff, newest block first (start here).
- `PROGRESS_*.md`: daily logs. `session_2026-10-15/PREREG_10-15.md`: pre-registrations, amendments and leaderboard reads.
- `session_YYYY-MM-DD/`: each day's builders (`build_*.py`), gates (`probe_check*.py`, `*_commit_gate.py`), submit ticks
  (`tick_*.py` / `.cmd`), kernel notebooks (`kernel_*/`) and research write-ups (for example `session_2026-10-19/trick/TRICK_1019.md`
  for the E-OA discovery, `session_2026-10-20/val/VAL_1020.md`, `session_2026-10-21/lastday/LASTDAY_1021.md`).
- `kernels/`, `_nb/`, `_edge/`: kernel and notebook sources, including copies of public notebooks used as bases (third-party code,
  see their original authors on Kaggle).

## Not in this repo

Competition data (DICOM, report text, labels), any OAI-derived data (OAI rows never left the university box), model weights,
caches, kernel outputs, logs, scraped forum text and all credentials. The `.gitignore` is an allowlist for code and docs, and a
pre-push audit removed files that quoted competition report text or a third party's personal details.

Third-party code: several folders (`_nb/`, `_edge/`, `kernels/`, `research_1008/A_pulled/`, various `pulled_*` / `src/` folders) hold
copies of other Kaggle users' public notebooks that our kernels were built on. They are kept for reproducibility;
credit belongs to their original authors on Kaggle.
