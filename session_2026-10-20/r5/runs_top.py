# runs_top.py: exact-tie run structure in the top of the .958 band (read-only on the r5 snapshot)
import json, sys
d = json.load(open(sys.argv[1])); L = sorted(d["lb"], key=lambda r: r["rank"])
runs = []; cur = [L[0]]
for a, b in zip(L, L[1:]):
    if b["displayScore"] == a["displayScore"] and b["submissionId"] > a["submissionId"]: cur.append(b)
    else: runs.append(cur); cur = [b]
runs.append(cur)
for r in runs:
    if r[0]["displayScore"] == "0.958" and r[0]["rank"] < 900 and len(r) >= 4:
        print("run r%d-r%d len %d sid %d..%d" % (r[0]["rank"], r[-1]["rank"], len(r), r[0]["submissionId"], r[-1]["submissionId"]))
