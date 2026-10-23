# Attribution and a correction to this notebook's name

**I did not write any of the code in this notebook.** Every code cell here is taken
verbatim from [`ryokucha/rsna-knee-d4-blend-0946-ours10`](https://www.kaggle.com/code/ryokucha/rsna-knee-d4-blend-0946-ours10);
I only removed that author's six trailing cells, which depend on private datasets that are
not attached in its public metadata. All credit for this pipeline belongs to that author and
to the upstream authors it builds on — **mattiaangeli, antoinegg1, prvsiyan, pilkwang,
marwanmath, dreaddevelopment, and sofiaanjenje** — whose notebooks, weights, heads and label
tables are the attached sources listed in this kernel.

## Correction: this scores 0.943, not 0.946

The `0946` in this notebook's URL is inherited from the parent's name and **overstates what
the public pipeline alone achieves.** I submitted exactly this notebook to the competition on
2026-09-28 and it scored **0.943** on the public leaderboard — the same as the public frontier
it descends from.

The parent notebook reaches 0.946 by blending in its author's **private** ConvNeXt ensemble at
10%. That component is not attached in the public metadata, so it cannot run here, and without
it the remaining public pipeline measures 0.943.

Please read the URL as a name, not as a score. If you are choosing what to fork, this is the
public 0.943 pipeline.
