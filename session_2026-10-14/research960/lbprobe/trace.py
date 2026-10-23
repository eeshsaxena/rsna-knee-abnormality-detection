import json, glob, pandas as pd
SP = "C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/01efcc80-088f-477c-8a8a-b504ddf28d6a/scratchpad"
d = json.load(open("api_lb_1014T0527.json")); L = pd.DataFrame(d["lb"]).merge(pd.DataFrame(d["teams"]), on="teamId")
run = L[(L['rank']>=688)&(L['rank']<=720)]
for n in ["lb13","lb13b","lbx","lby","lb14a","lb14b"]:
    A = pd.read_csv(glob.glob(f"{SP}/{n}/*.csv")[0], encoding="utf-8-sig")
    m = run.merge(A, left_on="teamId", right_on="TeamId")
    me = A[A.TeamId==16799346].iloc[0]
    m = m.sort_values("Rank")
    print(n, "us r", me.Rank, me.Score, "| cluster teams present", len(m), "ranks:", list(zip(m.Rank, m.Score.round(3), m.submissionId)) [:40])
