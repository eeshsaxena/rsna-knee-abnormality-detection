# traj2.py: band table per snapshot (read-only). counts per display score + first rank; our rank + in-band position.
import json, glob, os, collections
ME = 16799346
P = "C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-14/research960/lbprobe/"
files = glob.glob(P + "api_lb_*.json") + glob.glob("C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-17/research_1hr/api_lb_*.json")
snaps = {}
for f in files:
    d = json.load(open(f)); snaps.setdefault(d["fetched_utc"][:16], (os.path.basename(f), d))
for ts in sorted(snaps):
    n, d = snaps[ts]; L = sorted(d["lb"], key=lambda r: r["rank"]); N = len(L)
    me = [r for r in L if r["teamId"] == ME][0]
    c = collections.Counter(r["displayScore"] for r in L); first = {}
    for r in L: first.setdefault(r["displayScore"], r["rank"])
    ge = lambda s: sum(v for k, v in c.items() if float(k) >= s - 1e-9)
    inband = me["rank"] - first[me["displayScore"]] + 1
    print(f"{ts} N={N} us r{me['rank']} {me['displayScore']} pos {inband}/{c[me['displayScore']]} | >=.960 {ge(.960)} .959 {c['0.959']} .958 {c['0.958']} .957 {c['0.957']} .956 {c['0.956']} | .961 {c['0.961']} .962 {c['0.962']} .963 {c['0.963']} >=.964 {ge(.964)}")
