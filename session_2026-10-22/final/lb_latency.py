# lb_latency.py (10-22 final, Claude): read-only on two local LB snapshots (10-21 17:54 UTC recheck, 10-22 03:22 UTC now).
# Rough scoring-flow check: teams whose public display score changed, binned by their lastSubmissionDate (UTC hour). MEASURED counts;
# latency reading is a GUESS (lastSubmissionDate is the team's latest submit, not the one that scored).
import json, collections, sys
R = "C:/Users/eeshs/Downloads/rsna-knee/"
def load(f):
    d = json.load(open(f)); tm = {t["teamId"]: t for t in d["teams"]}
    return d["fetched_utc"], {r["teamId"]: (r.get("displayScore"), r.get("submissionId"), tm.get(r["teamId"], {}).get("lastSubmissionDate"), tm.get(r["teamId"], {}).get("submissionCount")) for r in d["lb"]}
fa, A = load(R + "session_2026-10-21/recheck/lb/api_lb_1021T_recheck.json")
fb, B = load(R + "session_2026-10-22/final/lb/api_lb_1022T_final.json")
print("A", fa[:16], "B", fb[:16])
sid_changed = {t for t in B if t in A and B[t][1] != A[t][1]}
new_teams = [t for t in B if t not in A]
print("teams in B", len(B), "| new teams since A", len(new_teams), "| best-sid changed", len(sid_changed))
h = collections.Counter(); hall = collections.Counter()
for t, (s, sid, last, n) in B.items():
    if not last: continue
    k = str(last)[:13]
    if k >= "2026-10-21T17": hall[k] += 1
    if t in sid_changed: h[k] += 1
print("UTC hour of lastSubmissionDate | teams with that last-submit hour | of which best-sid changed since A")
for k in sorted(hall): print("  ", k, hall[k], h.get(k, 0))
# best-sid id range as a clock: highest submissionId now on the LB vs at A
print("max best submissionId on LB: A", max(v[1] for v in A.values() if v[1]), "B", max(v[1] for v in B.values() if v[1]))
