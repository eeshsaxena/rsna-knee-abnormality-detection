# RSNA Knee: one public checkpoint nobody blends lifts the 0.956 recipe

**The trick in one line:** nartaa's public weights dataset contains *two* CoAtNet checkpoints. Every public
blend uses the 384-pixel one (`..._r384_swa.pt`, the "0.949 anatomical mirror"). The second one,
`..._c08_r224_swa.pt` (224 pixels from an 0.8 centre crop, 0.945 on its own), sits unused in the same
dataset. This notebook runs it as one extra stage and mixes it into the Raptor half of the blend at a
weight fixed before any submission.

**What it did on the leaderboard (same displayed score, real gain):**

| base recipe | without the 224 checkpoint | with it (35% of the Raptor slot) |
|---|---|---|
| Raptor + S6, 0.954 recipe | rank 437 | **rank 320** |
| Raptor + R7, 0.956 recipe (this notebook) | rank 233 | **rank 156** |

The displayed score stayed 0.954 and 0.956 because Kaggle truncates to three decimals, but **inside a
block of tied displayed scores, rank follows the unrounded score.** Hundreds of teams sit at exactly
0.956, so a small real gain moves you past a large part of the block. The 224 model is the same family
as the 384 one, but a different resolution and field of view make it wrong on different studies, and that is
what an ensemble needs.

**How it is built:** heliosli's public R7 + Raptor notebook, unchanged, plus two cells:

1. *Stage 2b* writes nartaa's own 0.945 inference script with the 224 checkpoint, changing only the output
   file name, and runs it in its own process **only if the 9-hour clock allows**. Any failure or timeout
   leaves the submission exactly equal to the 0.956 recipe.
2. The blend cell gives the 224 model 35% of the Raptor slot (`A224 = 0.35`), over Raptor's
   non-failed studies. Nothing is fitted on labels or on the test set.

Everything attached is public and pinned to exact dataset versions, so a copy reproduces the score.
