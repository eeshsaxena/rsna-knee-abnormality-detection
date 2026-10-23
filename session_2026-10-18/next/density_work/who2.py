import json, glob, os
P = "C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-14/research960/lbprobe/"
files = sorted(glob.glob(P + "api_lb_*.json"), key=lambda f: json.load(open(f))["fetched_utc"])
keys = ["heliosli", "goodpjw", "kiyoc", "prvsiyan", "bees", "nartaa", "rostigor", "lavinwins", "leoprovorov", "picond", "laimer"]
for f in files[-12:]:
    d = json.load(open(f)); T = {t["teamId"]: t for t in d["teams"]}
    out = []
    for r in d["lb"]:
        nm = (T.get(r["teamId"], {}).get("teamName") or "")
        if any(k in nm.lower() for k in keys):
            out.append(f"{nm[:18]} r{r['rank']} {r['displayScore']} sid{r['submissionId']} subs{T[r['teamId']]['submissionCount']}")
    print(d["fetched_utc"][:16], " | ".join(out))
