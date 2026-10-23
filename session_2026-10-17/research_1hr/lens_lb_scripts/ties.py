# Detect tie clusters: within a display band, Kaggle breaks exact ties by earlier submission, so a tied cluster shows
# as a run of consecutive ranks with increasing submissionId. Print maximal increasing runs (len>=4) in bands >= .957.
import json, sys
ME = 16799346
d = json.load(open(sys.argv[1])); L = sorted(d["lb"], key=lambda r: r["rank"])
runs = []; cur = [L[0]]
for a, b in zip(L, L[1:]):
    if b["displayScore"] == a["displayScore"] and b["submissionId"] > a["submissionId"]:
        cur.append(b)
    else:
        runs.append(cur); cur = [b]
runs.append(cur)
for r in runs:
    if float(r[0]["displayScore"]) >= 0.957 and len(r) >= 4:
        me = any(x["teamId"] == ME for x in r)
        print(f"{r[0]['displayScore']} r{r[0]['rank']}-r{r[-1]['rank']} len {len(r)} sid {r[0]['submissionId']}..{r[-1]['submissionId']}" + ("  <-- US" if me else ""))
# where are we and neighbours
i = [k for k, x in enumerate(L) if x["teamId"] == ME][0]
print("neighbours:", [(x["rank"], x["submissionId"]) for x in L[i-6:i+8]])
# random-order expectation of run lengths: mean run length ~ e-1 ~1.7; long runs => ties
import collections
c = collections.Counter(len(r) for r in runs if r[0]["displayScore"] == "0.958")
print(".958 run length histogram:", sorted(c.items()))
