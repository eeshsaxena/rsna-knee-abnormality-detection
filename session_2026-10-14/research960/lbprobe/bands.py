# bands.py: band structure + frozen-team transitions across LB snapshots. Read-only analysis.
import glob, os, pandas as pd
SP = "C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/01efcc80-088f-477c-8a8a-b504ddf28d6a/scratchpad"
names = ["lb13","lb13b","lbx","lby","lb14a","lb14b"] + sorted(os.path.basename(d) for d in glob.glob(SP+"/lb14r*")) + sorted(os.path.basename(d) for d in glob.glob(SP+"/r960lb*"))
snaps = []
for n in names:
    f = glob.glob(f"{SP}/{n}/*.csv")
    if not f: continue
    df = pd.read_csv(f[0], encoding="utf-8-sig"); ts = os.path.basename(f[0]).split("leaderboard-")[1][:19]
    snaps.append((n, ts, df))
for n, ts, df in snaps:
    me = df[df.TeamName.str.contains("Eesh saxena", na=False)].iloc[0]
    g = df.groupby("Score").Rank.agg(["min","max","size"]).sort_index(ascending=False).head(9)
    print(f"== {n} {ts} teams {len(df)}  us r{me.Rank} {me.Score}")
    print("   " + "; ".join(f"{s:.3f}: r{a}-{b} ({c})" for s,(a,b,c) in g.iterrows()))
print()
for (n1,t1,A),(n2,t2,B) in zip(snaps, snaps[1:]):
    m = A.merge(B, on="TeamId", suffixes=("_a","_b"))
    fz = m[(m.LastSubmissionDate_a == m.LastSubmissionDate_b) & (m.Score_a == m.Score_b)]
    ma = A[A.TeamName.str.contains("Eesh saxena", na=False)].iloc[0]; mb = B[B.TeamName.str.contains("Eesh saxena", na=False)].iloc[0]
    band_a = A[A.Score == 0.956]; band_b = B[B.Score == 0.956]
    fzb = fz[fz.Score_a == 0.956]
    # frozen in band, above/below us
    up_a = (fz.Rank_a < ma.Rank).sum(); up_b = (fz.Rank_b < mb.Rank).sum()
    # changed teams in .956 band (new entrants or improved)
    new_in_band = set(band_b.TeamId) - set(band_a.TeamId)
    print(f"{n1}->{n2}: frozen {len(fz)}/{len(m)}; .956 band {len(band_a)}->{len(band_b)} frozen-in-band {len(fzb)} ({len(fzb)/max(1,len(band_a)):.2f}); new in band {len(new_in_band)}; frozen above us {up_a}->{up_b}; us r{ma.Rank}->r{mb.Rank} last {ma.LastSubmissionDate}->{mb.LastSubmissionDate}")
