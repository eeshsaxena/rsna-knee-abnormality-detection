import json, glob, sys, pandas as pd
SP = "C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/01efcc80-088f-477c-8a8a-b504ddf28d6a/scratchpad"
d = json.load(open("api_lb_1014T0527.json")); L = pd.DataFrame(d["lb"]).merge(pd.DataFrame(d["teams"]), on="teamId")
L["last"] = L.lastSubmissionDate.str[:19].str.replace("T"," ")
snap, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
A = pd.read_csv(glob.glob(f"{SP}/{snap}/*.csv")[0], encoding="utf-8-sig")
m = A.merge(L[["teamId","submissionId","last","rank","displayScore"]], left_on="TeamId", right_on="teamId", how="left")
m["frozen"] = (m.LastSubmissionDate.str[:19] == m["last"])
s = m[(m.Rank>=lo)&(m.Rank<=hi)].sort_values("Rank")
for _, r in s.iterrows():
    print(f"r{r.Rank} {r.Score} {'FROZEN id '+str(int(r.submissionId)) if r.frozen else '   moved now r'+str(r['rank'])+' '+str(r.displayScore)}  {r.TeamName[:22]}")
