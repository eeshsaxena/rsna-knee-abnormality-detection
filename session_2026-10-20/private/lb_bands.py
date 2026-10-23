# lb_bands.py: read-only band / movement analysis of public LB snapshots (10-20 04:17, 13:20, now).
import json, sys, collections
sys.stdout.reconfigure(encoding='utf-8')
R = "C:/Users/eeshs/Downloads/rsna-knee/"
snaps = [("10-19 20:17", R+"session_2026-10-20/r5/api_lb_1020T_r5.json"),
         ("10-20 04:17", R+"session_2026-10-20/forum/api_lb_1020T0417.json"),
         ("10-20 05:53", R+"session_2026-10-14/research960/lbprobe/api_lb_1020_eoaread2.json"),
         ("10-20 13:20", R+"session_2026-10-14/research960/lbprobe/api_lb_1020_a21read.json"),
         ("10-20 now", R+"session_2026-10-20/private/lb/api_lb_1020T_private.json")]
ME = 16799346
def load(f):
    d = json.load(open(f)); tm = {t["teamId"]: t for t in d["teams"]}; rows = {}
    for r in d["lb"]:
        try: s = float(r.get("displayScore"))
        except: s = None
        rows[r["teamId"]] = dict(r, score=s, **{k: tm.get(r["teamId"], {}).get(k) for k in ("teamName","submissionCount","lastSubmissionDate")})
    return d["fetched_utc"], rows
S = [(n,)+load(f) for n, f in snaps]
for n, fu, rows in S:
    sc = [r["score"] for r in rows.values() if r["score"] is not None]; me = rows[ME]
    c = lambda t: sum(1 for x in sc if x >= t - 1e-9)
    band = lambda v: [r["rank"] for r in rows.values() if r["score"] is not None and abs(r["score"]-v) < 1e-9]
    b8 = band(.958); b9 = band(.959)
    med = collections.defaultdict(list)
    for r in rows.values(): med[r.get("medal")].append(r["rank"])
    ml = {k: (min(v), max(v)) for k, v in med.items() if k}
    above = sum(1 for r in rows.values() if r["score"] == me["score"] and r["rank"] < me["rank"])
    print(f"{n} ({fu[:16]}): teams {len(rows)}; >=.959 {c(.959)}; >=.960 {c(.960)}; >=.961 {c(.961)}; >=.962 {c(.962)}; >=.964 {c(.964)}; >=.966 {c(.966)}; top {max(sc)}; "
          f".959 r{min(b9)}-{max(b9)} ({len(b9)}); .958 r{min(b8)}-{max(b8)} ({len(b8)}); us r{me['rank']} {me['displayScore']} sid {me['submissionId']} in-band-above {above}; medals {ml}")
n, fu, B = S[-1]
g = collections.defaultdict(list)
for r in B.values():
    if r["score"] is not None: g[r["score"]].append(r["rank"])
print("\nbands now")
for s in sorted(g, reverse=True)[:14]: print(f"  {s:.3f}: {len(g[s])} r{min(g[s])}-{max(g[s])}")
# silver line score
sil = max(r["rank"] for r in B.values() if r.get("medal") == "SILVER")
print("silver last rank", sil, "score there", [r["displayScore"] for r in B.values() if r["rank"] == sil])
for i in (1, 3):
    na, _, A = S[i]
    mA = A[ME]; mB = B[ME]
    cr = []
    for t, rb in B.items():
        ra = A.get(t)
        if rb["score"] is not None and rb["score"] >= .959 - 1e-9 and (ra is None or ra["score"] is None or ra["score"] < .959 - 1e-9):
            cr.append((rb["rank"], rb["teamName"], ra and ra["score"], ra and ra["rank"], rb["score"], ra and ra["submissionCount"], rb["submissionCount"], (rb["lastSubmissionDate"] or "")[:16]))
    print(f"\ncrossers into .959+ {na} -> now: {len(cr)}")
    for x in sorted(cr): print("  r%s %s | %s (r%s) -> %s | subs %s->%s | last %s" % x)
    out = [(A[t]["rank"], A[t]["teamName"], A[t]["score"], B.get(t, {}).get("score")) for t in A if A[t]["score"] and A[t]["score"] >= .959-1e-9 and (t not in B or (B[t]["score"] or 0) < .959-1e-9)]
    print("  left .959+:", out)
A = S[3][2]; mA = A[ME]; mB = B[ME]
print(f"\nus 13:20 r{mA['rank']} sid {mA['submissionId']} -> now r{mB['rank']} sid {mB['submissionId']}")
pas = [(rb["rank"], rb["teamName"], A.get(t, {}).get("rank"), A.get(t, {}).get("displayScore"), rb["displayScore"], rb["submissionCount"], (rb["lastSubmissionDate"] or "")[:16]) for t, rb in B.items()
       if t != ME and rb["rank"] < mB["rank"] and (t not in A or A[t]["rank"] > mA["rank"])]
print("passed us since 13:20:", len(pas))
for p in sorted(pas): print("  ", p)
gone = [(A[t]["rank"], A[t]["teamName"]) for t in A if t != ME and A[t]["rank"] < mA["rank"] and (t not in B or B[t]["rank"] > mB["rank"])]
print("fell below us / left:", gone)
print("score changes within .959+ since 13:20:")
for t, rb in sorted(B.items(), key=lambda kv: kv[1]["rank"]):
    ra = A.get(t)
    if ra and ra["score"] is not None and ra["score"] >= .959-1e-9 and rb["score"] != ra["score"]:
        print(f"  r{ra['rank']}->r{rb['rank']} {rb['teamName']} {ra['score']}->{rb['score']} subs {ra['submissionCount']}->{rb['submissionCount']}")
# .959 band: teams above us, frozen vs active; teams below us in band
lo = min(g[.959]); hi = max(g[.959])
ab = sorted((r["rank"], r["teamName"], r["submissionCount"], (r["lastSubmissionDate"] or "")[:16], A.get(t, {}).get("rank"), A.get(t, {}).get("displayScore")) for t, r in B.items() if r["score"] == .959 and r["rank"] < mB["rank"])
print("\n.959 band above us:", len(ab))
for x in ab: print("  ", x)
print(".959 band below us:", hi - mB["rank"])
# active since 13:20 (submission count changed) by band
act = collections.Counter()
for t, rb in B.items():
    ra = A.get(t)
    if ra and rb["submissionCount"] != ra["submissionCount"]: act[rb["displayScore"]] += 1
print("teams with new subs since 13:20 by current band (>= .957):", {k: v for k, v in sorted(act.items(), reverse=True) if k and k >= "0.957"}, "total", sum(act.values()))
print("new teams since 13:20:", len(set(B) - set(A)), "removed:", len(set(A) - set(B)))
