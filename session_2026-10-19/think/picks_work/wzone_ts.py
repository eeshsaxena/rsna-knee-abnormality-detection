# wzone_ts.py: time series of the zone between our best and the dense tie (copy) zone (first run of >= 9 exact ties in .958).
import json, glob
ME = 16799346
fs = glob.glob("C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-14/research960/lbprobe/api_lb_*.json") + \
     glob.glob("C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-17/research_1hr/api_lb_*.json") + \
     ["C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/work/api_lb_1019T_think.json"]
rows = []
for f in fs:
    d = json.load(open(f)); rows.append((d["fetched_utc"][:16], d))
rows.sort(key=lambda x: x[0]); seen = set()
for ts, d in rows:
    if ts[:13] in seen or ts < "2026-10-16T06": continue
    seen.add(ts[:13])
    L = sorted(d["lb"], key=lambda r: r["rank"]); me = [r for r in L if r["teamId"] == ME][0]
    runs = []; cur = [L[0]]
    for a, b in zip(L, L[1:]):
        if b["displayScore"] == a["displayScore"] and b["submissionId"] > a["submissionId"]: cur.append(b)
        else: runs.append(cur); cur = [b]
    runs.append(cur)
    r958 = [r for r in runs if r[0]["displayScore"] == "0.958"]
    big = [r for r in r958 if len(r) >= 9]
    c0 = big[0][0]["rank"] if big else None
    ge959 = sum(1 for r in L if float(r["displayScore"]) >= 0.959)
    n958 = sum(1 for r in L if r["displayScore"] == "0.958")
    intie = sum(len(r) for r in r958 if len(r) >= 4)
    print(f"{ts} N {len(L)} us r{me['rank']} sid {me['submissionId']} | >=.959 {ge959} | .958 n {n958} | first len>=9 tie run r{c0} | W(us..c0) {c0 - me['rank'] - 1 if c0 else None} | members of runs>=4 {intie}")
