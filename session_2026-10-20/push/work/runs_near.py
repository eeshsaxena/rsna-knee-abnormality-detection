# runs_near.py: exact-tie signature (increasing-submissionId runs) near the .959 boundary, 04:17 and 05:53 snapshots. Read-only.
import json
R = "C:/Users/eeshs/Downloads/rsna-knee/"; LB = R + "session_2026-10-14/research960/lbprobe/"
for f in (R + "session_2026-10-20/forum/api_lb_1020T0417.json", LB + "api_lb_1020_eoaread2.json"):
    L = sorted(json.load(open(f))["lb"], key=lambda r: r["rank"])
    for lo, hi in ((87, 230), (230, 700)):
        seg = [r for r in L if lo <= r["rank"] <= hi]
        runs = []; cur = [seg[0]]
        for a, b in zip(seg, seg[1:]):
            if b["submissionId"] > a["submissionId"] and b["displayScore"] == a["displayScore"]: cur.append(b)
            else: runs.append(cur); cur = [b]
        runs.append(cur)
        long = [(r[0]["rank"], len(r), r[0]["displayScore"]) for r in runs if len(r) >= 4]
        print(f.split("/")[-1], f"r{lo}-{hi}", "runs>=4 (start rank, len, band):", long[:25])
