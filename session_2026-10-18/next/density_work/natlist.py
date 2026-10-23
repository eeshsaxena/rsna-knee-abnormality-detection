import json, sys
sys.path.insert(0, "."); from sidtime import t_of
ME = 16799346
P = "C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-14/research960/lbprobe/"
cut = int(sys.argv[2])
d = json.load(open(P + sys.argv[1])); T = {t["teamId"]: t for t in d["teams"]}
L = sorted(d["lb"], key=lambda r: r["rank"]); me = [r for r in L if r["teamId"] == ME][0]
print(sys.argv[1], d["fetched_utc"][:16], "us r", me["rank"], me["submissionId"])
band = [r for r in L if r["displayScore"] == me["displayScore"]]
lo = band[0]["rank"]; n = len(band)
for r in band:
    if r["submissionId"] < cut or r["teamId"] == ME:
        print(f"  r{r['rank']} pos {r['rank']-lo+1}/{n} sid {r['submissionId']} ({t_of(r['submissionId'])}) subs {T[r['teamId']]['submissionCount']} team {r['teamId']}" + ("  <-- US" if r["teamId"] == ME else ""))
