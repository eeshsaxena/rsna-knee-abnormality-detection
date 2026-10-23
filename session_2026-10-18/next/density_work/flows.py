# flows.py: team transitions between two snapshots, zones relative to our entry and the .959 boundary. Read-only.
import json, sys, collections
ME = 16799346
P = "C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-14/research960/lbprobe/"
def load(f):
    d = json.load(open(P + f)); L = sorted(d["lb"], key=lambda r: r["rank"]); return d["fetched_utc"][:16], L
def runs_of(L):
    # tie-run membership: maximal increasing-sid runs within the same display score, len >= 4
    mem = {}; cur = [L[0]]
    for a, b in zip(L, L[1:]):
        if b["displayScore"] == a["displayScore"] and b["submissionId"] > a["submissionId"]: cur.append(b)
        else:
            if len(cur) >= 4:
                for x in cur: mem[x["teamId"]] = len(cur)
            cur = [b]
    return mem
def zone(r, me):
    s = float(r["displayScore"])
    if s >= 0.960 - 1e-9: return "Z960+"
    if s >= 0.959 - 1e-9: return "Z959"
    if s >= 0.958 - 1e-9: return "V(above us)" if r["rank"] < me["rank"] else "below us .958"
    return "<.958"
fa, fb = sys.argv[1], sys.argv[2]
ta, A = load(fa); tb, B = load(fb)
mea = [r for r in A if r["teamId"] == ME][0]; meb = [r for r in B if r["teamId"] == ME][0]
assert mea["submissionId"] == meb["submissionId"], "our entry changed between snapshots"
ia = {r["teamId"]: r for r in A}; ib = {r["teamId"]: r for r in B}
memA = runs_of(A)
T = collections.Counter(); TC = collections.Counter(); moved_from = collections.defaultdict(list)
for t, rb in ib.items():
    ra = ia.get(t)
    if ra is None:
        T[("new team", zone(rb, meb))] += 1; continue
    if ra["submissionId"] == rb["submissionId"]: continue
    za, zb = zone(ra, mea), zone(rb, meb)
    T[(za, zb)] += 1
    if za == "below us .958" and t in memA: TC[zb] += 1
    if za == "below us .958": moved_from[zb].append(ra["rank"] - mea["rank"])
print(f"{ta} -> {tb}: us r{mea['rank']} -> r{meb['rank']}")
for k, v in sorted(T.items(), key=lambda kv: -kv[1]): print(f"  {k[0]:>15s} -> {k[1]:<15s} {v}")
print("  from tie-run (copy) teams below us, landing zone:", dict(TC))
for z, xs in moved_from.items():
    xs.sort(); print(f"  below-us movers landing in {z}: n {len(xs)}, start depth below us (ranks) p10/50/90:", [xs[int(q*(len(xs)-1))] for q in (.1, .5, .9)])
