# lbpos.py: in-band positions of c4 / eoa10 / eoa20 and the frozen-team ruler (public LB API snapshots, read-only).
import json, glob, collections
R = "C:/Users/eeshs/Downloads/rsna-knee/"; LB = R + "session_2026-10-14/research960/lbprobe/"
ME = 16799346; C4, E10, E20 = 57343435, 57413380, 57413368
def load(f):
    d = json.load(open(f)); L = sorted(d["lb"], key=lambda r: r["rank"]); return d["fetched_utc"][:16], L
S417 = load(R + "session_2026-10-20/forum/api_lb_1020T0417.json"); E1 = load(LB + "api_lb_1020_eoaread1.json"); E2 = load(LB + "api_lb_1020_eoaread2.json")
def bands(L):
    b = collections.OrderedDict()
    for r in L:
        x = b.setdefault(r["displayScore"], [r["rank"], r["rank"], 0]); x[1] = r["rank"]; x[2] += 1
    return b
for nm, (t, L) in (("04:17", S417), ("E1", E1), ("E2", E2)):
    me = [r for r in L if r["teamId"] == ME][0]; b = bands(L); lo, hi, n = b[me["displayScore"]]
    print(f"{nm} {t} N {len(L)} us r{me['rank']} sid {me['submissionId']} {me['displayScore']} band r{lo}-{hi} ({n}) above-in-band {me['rank']-lo} below-in-band {hi-me['rank']}")
# E1 vs E2: identical apart from us?
a = {r["teamId"]: r["submissionId"] for r in E1[1]}; b2 = {r["teamId"]: r["submissionId"] for r in E2[1]}
print("E1->E2 teams whose best sid changed:", [k for k in a if a.get(k) != b2.get(k)])
# 04:17 -> E1 changes in .958 strip above c4 and in .959
a417 = {r["teamId"]: (r["submissionId"], r["displayScore"], r["rank"]) for r in S417[1]}
ch = [(k, a417.get(k), (b2[k], [r for r in E2[1] if r["teamId"] == k][0]["displayScore"])) for k in b2 if a417.get(k, (None,))[0] != b2[k]]
print("04:17 -> 05:53 changed best sid:", len(ch))
ranks417 = {r["teamId"]: r for r in S417[1]}; ranksE2 = {r["teamId"]: r for r in E2[1]}
for k, x, y in ch:
    if k == ME: continue
    if (x and x[1] in ("0.958", "0.959", "0.960", "0.961")) or y[1] in ("0.958", "0.959", "0.960", "0.961"):
        print("   team", k, "04:17", x[1:] if x else None, "-> 05:53", y[1], "r", ranksE2[k]["rank"])
# frozen ruler: teams whose best sid is the same in every snapshot from c4's first read (10-18 12:52) to E2
snaps = []
for f in sorted(glob.glob(LB + "api_lb_*.json")) + glob.glob(R + "session_2026-10-19/**/api_lb_*.json", recursive=True) + glob.glob(R + "session_2026-10-20/**/api_lb_*.json", recursive=True):
    try: d = json.load(open(f))
    except Exception: continue
    if "lb" not in d: continue
    t = d["fetched_utc"][:16]
    if t >= "2026-10-18T12:52": snaps.append((t, {r["teamId"]: r["submissionId"] for r in d["lb"]}))
snaps.sort(); print("snapshots for frozen ruler:", len(snaps), snaps[0][0], "->", snaps[-1][0])
frozen = set(snaps[0][1]);
for t, m in snaps: frozen &= {k for k in m if m[k] == snaps[0][1].get(k)}
frozen.discard(ME)
def zone_counts(L, our_sid_rank):
    pass
# order: in E2, frozen teams by display band and position relative to eoa20 (r135) and eoa10 (r167 in E1)
r10 = [r for r in E1[1] if r["teamId"] == ME][0]["rank"]; r20 = [r for r in E2[1] if r["teamId"] == ME][0]["rank"]
rc4 = ranks417[ME]["rank"]
Z = collections.Counter()
for k in frozen:
    if k not in ranksE2: continue
    s = ranksE2[k]["displayScore"]; rE2 = ranksE2[k]["rank"]; rE1 = [None]
    if s in ("0.961", "0.962", "0.963"): Z[">=.961(<=.963)"] += 1
    elif s == "0.960": Z[".960"] += 1
    elif s == "0.959":
        # position vs eoa20 in E2 and eoa10 in E1 (frozen team ranks are identical in E1/E2 apart from our move)
        rk1 = [r for r in E1[1] if r["teamId"] == k][0]["rank"]
        if rE2 < r20: Z[".959 above eoa20"] += 1
        elif rk1 < r10: Z[".959 eoa10..eoa20"] += 1
        else: Z[".959 below eoa10"] += 1
    elif s == "0.958":
        if ranks417[k]["rank"] < rc4: Z[".958 above c4"] += 1
        else: Z[".958 below c4"] += 1
print("frozen teams since 10-18 12:52 (n %d):" % len(frozen), dict(Z))
# all-team counts in the same zones (E2 for .959/.960, 04:17 for the .958 strip)
b = bands(E2[1]); print("E2 bands .962/.961/.960/.959/.958:", {k: b[k] for k in ("0.963", "0.962", "0.961", "0.960", "0.959", "0.958")})
b4 = bands(S417[1]); print("04:17 bands:", {k: b4[k] for k in ("0.961", "0.960", "0.959", "0.958", "0.957")})
