## Stage 2 — A5 attention pooling

Five folds pool variable-length slice features. Host preparation stays bounded
and overlaps the active GPU pass; unavailable acquisitions fall back to the
base branch, while model failures remain fatal.
