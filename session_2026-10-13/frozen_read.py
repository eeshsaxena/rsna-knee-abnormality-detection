# frozen_read.py OLD_DIR NEW_DIR: our rank + count of "frozen" teams (score and LastSubmissionDate unchanged) ranked above us.
import sys, glob, pandas as pd
def load(d): return pd.read_csv(glob.glob(f"{d}/*.csv")[0])
A, B = load(sys.argv[1]), load(sys.argv[2])
m = A.merge(B, on="TeamId", suffixes=("_a", "_b"))
fz = m[(m.LastSubmissionDate_a == m.LastSubmissionDate_b) & (m.Score_a == m.Score_b)]
me = lambda X: X[X.TeamName.str.contains("Eesh saxena", na=False)].iloc[0]
ma, mb = me(A), me(B)
print(f"our rank {ma.Rank} ({ma.Score}) -> {mb.Rank} ({mb.Score}); teams {len(A)} -> {len(B)}; frozen {len(fz)}; "
      f"frozen ranked above us {(fz.Rank_a < ma.Rank).sum()} -> {(fz.Rank_b < mb.Rank).sum()}")
b = B[B.Score == mb.Score]; print(f"our band {mb.Score}: r{b.Rank.min()}-{b.Rank.max()} ({len(b)} teams); silver cut ~r{int(len(B)*0.05)}")
