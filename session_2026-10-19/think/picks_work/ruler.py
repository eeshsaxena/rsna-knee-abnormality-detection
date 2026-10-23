# ruler.py: frozen-team ruler across our entries' read snapshots (read-only). Teams whose best submissionId is identical in every
# snapshot have a fixed public score, so their order vs each of our entries is MEASURED; counts per interval give spacing (MODEL).
import json, collections
ME = 16799346
P = "C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-14/research960/lbprobe/"
S = [("R6", P + "api_lb_1420.json"), ("HW", P + "api_lb_chain_0105_2.json"), ("a25", P + "api_lb_1018T0645.json"),
     ("c4", P + "api_lb_now.json"), ("c4_now", "C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/work/api_lb_1019T_think.json")]
snaps = {}
for nm, f in S:
    d = json.load(open(f)); L = {r["teamId"]: r for r in d["lb"]}; me = L[ME]; snaps[nm] = (d["fetched_utc"][:16], L, me)
    print(nm, d["fetched_utc"][:16], "us", me["submissionId"], "r", me["rank"], me["displayScore"])
common = set.intersection(*[set(L) for _, L, _ in snaps.values()]) - {ME}
frozen = [t for t in common if len({snaps[k][1][t]["submissionId"] for k in snaps}) == 1]
print("teams in all snapshots:", len(common), "frozen:", len(frozen))
def cls(t):
    sig = []
    for nm in ("R6", "HW", "a25", "c4"):
        _, L, me = snaps[nm]; sig.append("A" if L[t]["rank"] < me["rank"] else "b")
    return "".join(sig), snaps["c4_now"][1][t]["displayScore"]
c = collections.Counter(cls(t) for t in frozen)
for k, v in sorted(c.items(), key=lambda kv: (kv[0][1], kv[0][0]), reverse=True):
    if float(k[1]) >= 0.957: print(" pattern (above? R6,HW,a25,c4)", k[0], "display", k[1], "n", v)
# within .958 only: order consistency check (a team above c4 must be above a25 etc. if c4 > a25 > HW > R6)
bad = [t for t in frozen if cls(t)[0] not in ("AAAA", "bAAA", "bbAA", "bbbA", "bbbb")]
print("order-inconsistent frozen teams:", len(bad))
print("all patterns (any display >= .957):")
for k, v in sorted(c.items()):
    if float(k[1]) >= 0.957: print("  ", k, v)
for t in bad[:20]:
    print("  bad", t, cls(t), [snaps[k][1][t]["rank"] for k in ("R6", "HW", "a25", "c4", "c4_now")], snaps["c4_now"][1][t]["submissionId"])
