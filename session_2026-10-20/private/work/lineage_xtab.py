# lineage_xtab.py: current band x band on 10-14 05:27 (before any public .958 notebook). Read-only, team-level aggregates only.
import json, collections
P = "C:/Users/eeshs/Downloads/rsna-knee/"
old = json.load(open(P + "session_2026-10-14/research960/lbprobe/api_lb_1014T0527.json", encoding="utf-8"))
mid = json.load(open(P + "session_2026-10-14/research960/lbprobe/api_lb_1017T2150.json", encoding="utf-8"))
now = json.load(open(P + "session_2026-10-20/private/work/api_lb_1020T_shake.json", encoding="utf-8"))
ob = {r["teamId"]: r["displayScore"] for r in old["lb"]}
mb = {r["teamId"]: r["displayScore"] for r in mid["lb"]}
L = sorted(now["lb"], key=lambda r: r["rank"])
runs = []; cur = [L[0]]
for a, b in zip(L, L[1:]):
    if b["displayScore"] == a["displayScore"] and b["submissionId"] > a["submissionId"]: cur.append(b)
    else: runs.append(cur); cur = [b]
runs.append(cur); tl = {x["teamId"]: len(r) for r in runs for x in r}
def cls(s):
    if s is None: return "new"
    v = float(s)
    if v >= 0.958: return ">=.958"
    if v >= 0.957: return ".957"
    if v >= 0.956: return ".956"
    return "<.956"
tab = collections.OrderedDict()
for r in L:
    s = r["displayScore"]; v = float(s)
    if v < 0.954: break
    key = s if v >= 0.959 else s
    tab.setdefault(key, collections.Counter())[cls(ob.get(r["teamId"]))] += 1
    if tl[r["teamId"]] >= 4: tab[key]["(tie>=4)"] += 1
print("current band | on 10-14 05:27: >=.958 / .957 / .956 / <.956 / new | in tie run>=4")
for k, c in tab.items():
    print(k, c[">=.958"], c[".957"], c[".956"], c["<.956"], c["new"], "|", c["(tie>=4)"], " total", sum(v for kk, v in c.items() if kk != "(tie>=4)"))
# where are the 104 teams that were >= .958 on 10-14 now?
cur = {r["teamId"]: r["displayScore"] for r in L}
c2 = collections.Counter(cur.get(t, "gone") for t, s in ob.items() if float(s) >= 0.958)
print("10-14 >=.958 teams now:", sorted(c2.items(), reverse=True))
# W zone: .958 teams above the first run >= 8
first = next(r for r in runs if len(r) >= 8 and r[0]["displayScore"] == "0.958")
w = [r for r in L if r["displayScore"] == "0.958" and r["rank"] < first[0]["rank"]]
cw = collections.Counter(cls(ob.get(r["teamId"])) for r in w)
print("W zone r%d-%d n %d by 10-14 band:" % (w[0]["rank"], w[-1]["rank"], len(w)), dict(cw))
