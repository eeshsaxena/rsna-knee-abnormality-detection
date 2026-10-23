# lbpos2.py (val, 10-20): in-band position of pw30 (A21 read, 13:20 UTC) and the frozen-team cross-board ruler vs eoa20 (05:53 UTC).
# Public LB API snapshots only, read-only. Prints counts.
import json, collections
R = "C:/Users/eeshs/Downloads/rsna-knee/"; LBD = R + "session_2026-10-14/research960/lbprobe/"
ME = 16799346
def load(f):
    d = json.load(open(f)); L = sorted(d["lb"], key=lambda r: r["rank"]); return d["fetched_utc"][:16], L
E1 = load(LBD + "api_lb_1020_eoaread1.json"); E2 = load(LBD + "api_lb_1020_eoaread2.json"); A = load(LBD + "api_lb_1020_a21read.json")
def bands(L):
    b = collections.OrderedDict()
    for r in L:
        x = b.setdefault(r["displayScore"], [r["rank"], r["rank"], 0]); x[1] = r["rank"]; x[2] += 1
    return b
for nm, (t, L) in (("E2", E2), ("A21", A)):
    me = [r for r in L if r["teamId"] == ME][0]; b = bands(L)
    print(nm, t, "N", len(L), "us r%d sid %d %s" % (me["rank"], me["submissionId"], me["displayScore"]))
    for s in ("0.963", "0.962", "0.961", "0.960", "0.959", "0.958", "0.957"):
        if s in b: print("   band", s, "r%d-%d n %d" % tuple(b[s]))
    lo = b[me["displayScore"]][0]; hi = b[me["displayScore"]][1]
    print("   above-in-band", me["rank"] - lo, "below-in-band", hi - me["rank"])
e2 = {r["teamId"]: r for r in E2[1]}; a = {r["teamId"]: r for r in A[1]}
e1 = {r["teamId"]: r for r in E1[1]}
# changes E2 -> A21
ch = [k for k in a if k != ME and (k not in e2 or e2[k]["submissionId"] != a[k]["submissionId"])]
print("teams whose best sid changed E2->A21:", len(ch), "| new teams:", sum(k not in e2 for k in a))
C = collections.Counter()
for k in ch:
    s0 = e2[k]["displayScore"] if k in e2 else "new"; s1 = a[k]["displayScore"]
    if s1 >= "0.958" or s0 >= "0.958": C[(s0, s1)] += 1
print("   moves among >= .958:", dict(sorted(C.items())))
# frozen ruler
fz = [k for k in a if k != ME and k in e2 and e2[k]["submissionId"] == a[k]["submissionId"]]
r20 = e2[ME]["rank"]; rpw = a[ME]["rank"]
Z = collections.Counter()
for k in fz:
    s = a[k]["displayScore"]
    if s not in ("0.958", "0.959", "0.960", "0.961"): continue
    above20 = e2[k]["rank"] < r20; abovepw = a[k]["rank"] < rpw
    Z[(s, "above eoa20" if above20 else "below eoa20", "above pw30" if abovepw else "below pw30")] += 1
for kk, v in sorted(Z.items()): print("   frozen", kk, v)
# frozen teams in .959 by zone (E2 positions: below eoa10 (E1 rank), eoa10..eoa20, above eoa20) and vs pw30
r10 = e1[ME]["rank"]
Z2 = collections.Counter()
for k in fz:
    if a[k]["displayScore"] != "0.959": continue
    z = "above eoa20" if e2[k]["rank"] < r20 else ("eoa10..eoa20" if e1[k]["rank"] < r10 else "below eoa10")
    Z2[(z, "above pw30" if a[k]["rank"] < rpw else "below pw30")] += 1
print("frozen .959 teams by E2 zone x A21 side:", dict(Z2))
# the 3 teams above pw30 in .959: were they in E2, frozen?
lo = bands(A[1])["0.959"][0]
for r in A[1]:
    if lo <= r["rank"] < rpw:
        k = r["teamId"]; print("   above pw30 in .959: r%d frozen %s E2 %s" % (r["rank"], k in fz, (e2[k]["displayScore"], e2[k]["rank"]) if k in e2 else None))
# the 4 teams above eoa20 in .959 at E2: where now?
lo2 = bands(E2[1])["0.959"][0]
for r in E2[1]:
    if lo2 <= r["rank"] < r20:
        k = r["teamId"]; print("   E2 above eoa20 in .959: r%d -> A21 %s r%d frozen %s" % (r["rank"], a[k]["displayScore"], a[k]["rank"], k in fz))
