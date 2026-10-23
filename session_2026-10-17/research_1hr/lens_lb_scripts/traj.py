import json, glob, os, collections
ME = 16799346
files = glob.glob("C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-14/research960/lbprobe/api_lb_*.json") + \
        glob.glob("C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-17/research_1hr/api_lb_*.json")
snaps = []
for f in files:
    d = json.load(open(f))
    snaps.append((d["fetched_utc"], os.path.basename(f), d))
snaps.sort()
seen = set()
for ts, n, d in snaps:
    if ts[:16] in seen: continue
    seen.add(ts[:16])
    L = d["lb"]; N = len(L)
    me = [r for r in L if r["teamId"] == ME][0]
    c = collections.OrderedDict()
    first = {}
    for r in L:
        s = r["displayScore"]; c[s] = c.get(s, 0) + 1; first.setdefault(s, r["rank"])
    silver = N // 20; gold = 10 + N // 500; bronze = N // 10
    sc = {r["rank"]: r["displayScore"] for r in L}
    bands = " ".join(f"{s[1:]}:{first[s]}+{c[s]}" for s in list(c)[:14] if float(s) >= 0.955)
    print(f"{ts[:16]} N={N} us r{me['rank']} {me['displayScore']} sid{me['submissionId']} | silverR {silver} ({sc.get(silver)}) goldR {gold} ({sc.get(gold)}) bronzeR {bronze} ({sc.get(bronze)}) | {bands}")
