import json, sys
ME = 16799346
d = json.load(open(sys.argv[1])); L = sorted(d["lb"], key=lambda r: r["rank"]); T = {t["teamId"]: t for t in d["teams"]}
runs = []; cur = [L[0]]
for a, b in zip(L, L[1:]):
    if b["displayScore"] == a["displayScore"] and b["submissionId"] > a["submissionId"]: cur.append(b)
    else: runs.append(cur); cur = [b]
runs.append(cur)
cum = 0
for r in runs:
    if r[0]["displayScore"] != "0.958" or len(r) < 4: continue
    cum += len(r)
    subs = sorted(T[x["teamId"]]["submissionCount"] or 0 for x in r)
    print(f"r{r[0]['rank']}-{r[-1]['rank']} len {len(r):3d} sid {r[0]['submissionId']}..{r[-1]['submissionId']} med subs {subs[len(subs)//2]} cum {cum}")
