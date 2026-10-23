import json, sys
ME = 16799346
P = "C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-14/research960/lbprobe/"
d = json.load(open(P + sys.argv[1])); T = {t["teamId"]: t for t in d["teams"]}
L = sorted(d["lb"], key=lambda r: r["rank"]); me = [r for r in L if r["teamId"] == ME][0]
lo = int(sys.argv[2]) if len(sys.argv) > 2 else me["rank"] - 40
hi = int(sys.argv[3]) if len(sys.argv) > 3 else me["rank"] + 5
for r in L:
    if lo <= r["rank"] <= hi:
        t = T[r["teamId"]]
        print(r["rank"], r["displayScore"], r["submissionId"], t["submissionCount"], (t["lastSubmissionDate"] or "")[:16], "<-- US" if r["teamId"] == ME else "")
