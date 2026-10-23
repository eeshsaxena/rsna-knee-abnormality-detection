# lb_bands.py (copied from canary/, 10-21 final): read-only bands + our row: 03:11 (day), 12:14 (canary), now (final). Movers since 12:14 UTC.
# and 12:03 UTC (a22read) snapshots. Prints counts per display band, our rank / in-band position, movers since 12:14 UTC.
import json, sys, collections
sys.stdout.reconfigure(encoding='utf-8')
R = "C:/Users/eeshs/Downloads/rsna-knee/"
snaps = [("10-21 03:11", R + "session_2026-10-21/day/lb/api_lb_1021T_day.json"),
         ("10-21 12:14", R + "session_2026-10-21/canary/lb/api_lb_1021T_canary.json"),
         ("now", R + "session_2026-10-21/final/lb/api_lb_1021T_final.json")]
ME = 16799346
def load(f):
    d = json.load(open(f)); tm = {t["teamId"]: t for t in d["teams"]}; rows = {}
    for r in d["lb"]:
        try: s = round(float(r.get("displayScore")), 3)
        except Exception: s = None
        rows[r["teamId"]] = dict(r, score=s, **{k: tm.get(r["teamId"], {}).get(k) for k in ("teamName", "submissionCount", "lastSubmissionDate")})
    return d["fetched_utc"], rows
S = [(n,) + load(f) for n, f in snaps]
for n, fu, rows in S:
    sc = [r["score"] for r in rows.values() if r["score"] is not None]; me = rows[ME]
    c = lambda t: sum(1 for x in sc if x >= t - 1e-9)
    band = lambda v: sorted(r["rank"] for r in rows.values() if r["score"] == v)
    b = band(me["score"])
    med = collections.defaultdict(list)
    for r in rows.values(): med[r.get("medal")].append(r["rank"])
    ml = {k: (min(v), max(v)) for k, v in med.items() if k}
    above = sum(1 for r in rows.values() if r["score"] == me["score"] and r["rank"] < me["rank"])
    print(f"{n} ({fu[:16]} UTC): teams {len(rows)}; >=.959 {c(.959)}; >=.960 {c(.960)}; >=.961 {c(.961)}; top {max(sc)}; "
          f"us r{me['rank']} {me['displayScore']} sid {me['submissionId']} subs {me['submissionCount']} band r{b[0]}-{b[-1]} ({len(b)}) in-band-above {above}; medals {ml}")
n, fu, B = S[-1]
g = collections.defaultdict(list)
for r in B.values():
    if r["score"] is not None: g[r["score"]].append(r["rank"])
print("\nbands now")
for s in sorted(g, reverse=True)[:13]: print(f"  {s:.3f}: {len(g[s])} r{min(g[s])}-{max(g[s])}")
sil = max(r["rank"] for r in B.values() if r.get("medal") == "SILVER")
print("silver last rank", sil, "display there", [r["displayScore"] for r in B.values() if r["rank"] == sil])
A = S[1][2]; mA = A[ME]; mB = B[ME]
print(f"\nus 12:14 r{mA['rank']} -> now r{mB['rank']}")
pas = sorted((rb["rank"], rb["teamName"], A.get(t, {}).get("rank"), A.get(t, {}).get("displayScore"), rb["displayScore"]) for t, rb in B.items()
             if t != ME and rb["rank"] < mB["rank"] and (t not in A or A[t]["rank"] > mA["rank"]))
print("passed us since 12:14:", pas)
act = collections.Counter(rb["displayScore"] for t, rb in B.items() if t in A and rb["submissionCount"] != A[t]["submissionCount"])
print("teams with newly completed subs since 12:14 by band (>= .958):", {k: v for k, v in sorted(act.items(), reverse=True) if k and k >= "0.958"}, "total", sum(act.values()))
A0 = S[0][2]
act0 = collections.Counter(rb["displayScore"] for t, rb in B.items() if t in A0 and rb["submissionCount"] != A0[t]["submissionCount"])
print("teams with newly completed subs since 03:11 by band (>= .958):", {k: v for k, v in sorted(act0.items(), reverse=True) if k and k >= "0.958"}, "total", sum(act0.values()))
x = [(t, A0[t]["score"], rb["score"]) for t, rb in B.items() if t in A0 and A0[t]["score"] is not None and rb["score"] is not None]
print("crossings into .960+ since 03:11:", sum(1 for _, a, b in x if a < .960 - 1e-9 and b >= .960 - 1e-9), "| into .959+:", sum(1 for _, a, b in x if a < .959 - 1e-9 and b >= .959 - 1e-9))
