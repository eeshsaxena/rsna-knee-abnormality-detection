import json, sys, pandas as pd
d = json.load(open(sys.argv[1])); L = pd.DataFrame(d["lb"]); T = pd.DataFrame(d["teams"])
L = L.merge(T, on="teamId", how="left")
pats = sys.argv[2].split(",")
for p in pats:
    m = L[L.teamName.str.contains(p, case=False, na=False)]
    for _, r in m.iterrows(): print(f"{p}: r{r['rank']} {r.displayScore} sub {r.submissionId} team '{r.teamName}' n{r.submissionCount}")
b = L[L.displayScore=="0.956"].sort_values("rank")
print("--- 33-run r688-720"); print(b[(b['rank']>=686)&(b['rank']<=722)][["rank","submissionId","teamName","submissionCount"]].to_string(index=False))
