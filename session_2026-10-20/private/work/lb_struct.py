# lb_struct.py: band counts, exact-tie runs (copy clusters) and non-tie counts per band on a public LB snapshot. Read-only.
import json, sys, collections
ME = 16799346
f = sys.argv[1]
d = json.load(open(f, encoding="utf-8")); L = sorted(d["lb"], key=lambda r: r["rank"]); T = {t["teamId"]: t for t in d["teams"]}
print("snapshot", d["fetched_utc"], "teams", len(L))
me = [r for r in L if r["teamId"] == ME][0]; print("us", me)
# runs: same display, increasing submissionId (Kaggle breaks exact ties by submission time)
runs = []; cur = [L[0]]
for a, b in zip(L, L[1:]):
    if b["displayScore"] == a["displayScore"] and b["submissionId"] > a["submissionId"]: cur.append(b)
    else: runs.append(cur); cur = [b]
runs.append(cur)
intie = {}
for r in runs:
    for x in r: intie[x["teamId"]] = len(r)
band = collections.OrderedDict()
for r in L:
    s = r["displayScore"]; band.setdefault(s, []).append(r)
print("band  n  ranks  tie>=4  tie>=8  nontie(<4)  med_subs_nontie")
out = {}
for s, rs in band.items():
    try: v = float(s)
    except: continue
    if v < 0.945: continue
    t4 = sum(1 for x in rs if intie[x["teamId"]] >= 4); t8 = sum(1 for x in rs if intie[x["teamId"]] >= 8)
    nt = [x for x in rs if intie[x["teamId"]] < 4]
    subs = sorted((T[x["teamId"]]["submissionCount"] or 0) for x in nt)
    ms = subs[len(subs) // 2] if subs else 0
    print(f"{s} {len(rs):5d} r{rs[0]['rank']}-{rs[-1]['rank']} {t4:5d} {t8:5d} {len(nt):5d} {ms:5d}")
    out[s] = dict(n=len(rs), r0=rs[0]["rank"], r1=rs[-1]["rank"], tie4=t4, tie8=t8, nontie=len(nt), med_subs_nontie=ms)
# runs >= 8 in .958: copy clusters
big = [r for r in runs if len(r) >= 8]
print("runs >= 8:", len(big), "teams", sum(len(r) for r in big))
for r in big[:40]:
    print(f"  {r[0]['displayScore']} r{r[0]['rank']}-{r[-1]['rank']} len {len(r)}")
# submission-count distribution above us
above = [r for r in L if r["rank"] < me["rank"]]
sc = sorted((T[x["teamId"]]["submissionCount"] or 0) for x in above)
print("above us:", len(above), "subs p10/50/90", sc[len(sc)//10], sc[len(sc)//2], sc[9*len(sc)//10])
json.dump(dict(snap=d["fetched_utc"], N=len(L), us=me, bands=out), open(sys.argv[2], "w"), indent=1)
