# Read-only analysis of public LB snapshots: bands, our row, .959 crossers.
import json, sys
R = "C:/Users/eeshs/Downloads/rsna-knee/"
snaps = [("10-18 06:46", R+"session_2026-10-14/research960/lbprobe/api_lb_1018T0645.json"),
         ("10-19 10:41", R+"session_2026-10-14/research960/lbprobe/api_lb_1019_reads.json"),
         ("10-19 19:24", R+"session_2026-10-20/more/lb/api_lb_1020T_more.json"),
         ("10-19 20:17", R+"session_2026-10-20/r5/api_lb_1020T_r5.json"),
         ("10-20 04:17", R+"session_2026-10-20/forum/api_lb_1020T0417.json")]
ME = 16799346
def load(f):
    d = json.load(open(f))
    tm = {t["teamId"]: t for t in d["teams"]}
    rows = {}
    for r in d["lb"]:
        s = r.get("displayScore")
        try: s = float(s)
        except: s = None
        rows[r["teamId"]] = dict(r, score=s, **{k: tm.get(r["teamId"], {}).get(k) for k in ("teamName","submissionCount","lastSubmissionDate")})
    return d["fetched_utc"], rows
S = [(n,)+load(f) for n,f in snaps]
print("snapshot | teams | >=.969..")
for n, fu, rows in S:
    sc = [r["score"] for r in rows.values() if r["score"] is not None]
    me = rows[ME]
    c = lambda t: sum(1 for x in sc if x >= t - 1e-9)
    band = lambda v: [r["rank"] for r in rows.values() if r["score"] is not None and abs(r["score"]-v) < 1e-9]
    b8 = band(.958); b9 = band(.959)
    above_me_band = sum(1 for r in rows.values() if r["score"] == me["score"] and r["rank"] < me["rank"])
    print(f"{n} ({fu[:16]}): teams {len(rows)}; >=.959 {c(.959)}; >=.960 {c(.960)}; >=.962 {c(.962)}; >=.964 {c(.964)}; >=.966 {c(.966)}; top {max(sc)}; "
          f".959 band {len(b9)} r{min(b9)}-{max(b9)}; .958 band {len(b8)} r{min(b8)}-{max(b8)}; us r{me['rank']} {me['displayScore']} sub {me['submissionId']} band-above-us {above_me_band}")
# final snapshot band table
n, fu, rows = S[-1]
from collections import defaultdict
g = defaultdict(list)
for r in rows.values():
    if r["score"] is not None: g[r["score"]].append(r["rank"])
print("\nbands at", n)
for s in sorted(g, reverse=True)[:16]:
    print(f"  {s:.3f}: {len(g[s])} teams r{min(g[s])}-{max(g[s])}")
print("  .960+ total:", sum(len(v) for s,v in g.items() if s >= .960-1e-9), "max rank", max(r for s,v in g.items() if s>=.960-1e-9 for r in v))
# crossers into .959+ vs each baseline
def crossers(A, B):
    out = []
    for t, rb in B.items():
        if rb["score"] is None or rb["score"] < .959 - 1e-9: continue
        ra = A.get(t)
        sa = ra["score"] if ra else None
        if sa is None or sa < .959 - 1e-9:
            out.append((rb["rank"], rb["teamName"], sa, rb["score"], ra["rank"] if ra else None, rb["submissionCount"], (ra or {}).get("submissionCount"), rb["lastSubmissionDate"], t))
    return sorted(out)
for i in (1, 3):
    na, _, A = S[i]
    cr = crossers(A, S[-1][2])
    print(f"\ncrossers into .959+ {na} -> {n}: {len(cr)}")
    for x in cr: print("  r%s %s | %s (r%s) -> %s | subs %s->%s | last %s | id %s" % (x[0], x[1], x[2], x[4], x[3], x[6], x[5], (x[7] or "")[:16], x[8]))
# within .959+ movers since 10-19 20:17
A = S[3][2]; B = S[-1][2]
print("\nscore changes within .959+ since 10-19 20:17:")
for t, rb in sorted(B.items(), key=lambda kv: kv[1]["rank"]):
    ra = A.get(t)
    if ra and ra["score"] is not None and ra["score"] >= .959-1e-9 and rb["score"] != ra["score"]:
        print(f"  r{ra['rank']}->r{rb['rank']} {rb['teamName']} {ra['score']}->{rb['score']}")
# in-band passers of us since 20:17 (same .958, rank moved from below us to above us)
mA = A[ME]; mB = B[ME]
print(f"\nus 20:17 r{mA['rank']} -> 04:17 r{mB['rank']}")
pas = [ (rb["rank"], rb["teamName"], A.get(t,{}).get("rank"), A.get(t,{}).get("score"), rb["score"], rb["submissionCount"]) for t, rb in B.items()
        if rb["rank"] < mB["rank"] and t in A and A[t]["rank"] > mA["rank"]]
pas += [ (rb["rank"], rb["teamName"], None, None, rb["score"], rb["submissionCount"]) for t, rb in B.items() if rb["rank"] < mB["rank"] and t not in A]
print("passed us since 20:17:", len(pas))
for p in sorted(pas): print("  ", p)
gone = [(A[t]["rank"], A[t]["teamName"]) for t in A if A[t]["rank"] < mA["rank"] and (t not in B or B[t]["rank"] > mB["rank"])]
print("fell below us / left:", gone)
print("new teams since 20:17:", len(set(B)-set(A)), "removed:", len(set(A)-set(B)))
