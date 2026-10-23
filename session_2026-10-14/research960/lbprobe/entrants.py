import json, glob, pandas as pd
SP = "C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/01efcc80-088f-477c-8a8a-b504ddf28d6a/scratchpad"
snaps = []
for n in ["lb13","lb13b","lbx","lby","lb14a","lb14b"]:
    A = pd.read_csv(glob.glob(f"{SP}/{n}/*.csv")[0], encoding="utf-8-sig"); snaps.append((n, A))
d = json.load(open("api_lb_1014T0527.json")); L = pd.DataFrame(d["lb"]).merge(pd.DataFrame(d["teams"]), on="teamId")
api = pd.DataFrame({"Rank": L["rank"], "TeamId": L.teamId, "TeamName": L.teamName, "LastSubmissionDate": L.lastSubmissionDate.str[:19].str.replace("T"," "), "Score": L.displayScore.astype(float)})
snaps.append(("api0527", api))
for (n1, A), (n2, B) in zip(snaps, snaps[1:]):
    m = B.merge(A[["TeamId","LastSubmissionDate","Score","Rank"]], on="TeamId", how="left", suffixes=("","_a"))
    changed = m[(m.LastSubmissionDate_a.isna()) | (m.LastSubmissionDate.str[:19] != m.LastSubmissionDate_a.astype(str).str[:19]) | (m.Score != m.Score_a)]
    me = B[B.TeamId == 16799346].iloc[0]
    cb = changed[changed.Score == 0.956]
    # teams that entered/improved into .956 from below or new
    ent = cb[(cb.Score_a.isna()) | (cb.Score_a < 0.956)]
    above, below = (ent.Rank < me.Rank).sum(), (ent.Rank > me.Rank).sum()
    # teams that were in band .956 already and resubmitted (score same, last changed): moved relative to us?
    re_ = cb[cb.Score_a == 0.956]
    print(f"{n1}->{n2}: us r{me.Rank}; entrants into .956 from below/new: {len(ent)} (above us {above}, below {below}); in-band resubmitters {len(re_)} (now above us {(re_.Rank<me.Rank).sum()}, below {(re_.Rank>me.Rank).sum()}); band top r{B[B.Score==0.956].Rank.min()}")
