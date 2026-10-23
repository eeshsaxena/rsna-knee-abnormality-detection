# Read-only: our team's completed-submission count + best id per LB snapshot (latency bounds).
import json, glob, os, sys
D = r"C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-14/research960/lbprobe"
extra = [r"C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/work/api_lb_1019T_think.json",
         r"C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-17/research_1hr/api_lb_1017T_lens.json"]
rows = []
for f in sorted(glob.glob(D + "/api_lb_*.json")) + extra:
    try:
        d = json.load(open(f))
    except Exception as e:
        print("ERR", f, e); continue
    t = [x for x in d["teams"] if x.get("teamName", "") == "Eesh saxena"]
    for x in t:
        lb = [y for y in d["lb"] if y["teamId"] == x["teamId"]]
        rows.append((d["fetched_utc"], os.path.basename(f), x["teamName"], x["submissionCount"], x["lastSubmissionDate"], lb[0]["submissionId"] if lb else None, lb[0]["rank"] if lb else None, lb[0]["displayScore"] if lb else None))
for r in sorted(rows):
    print(*r, sep=" | ")
