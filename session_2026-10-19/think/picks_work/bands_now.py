# bands_now.py: band table + tie runs + zone profile on the 10-19 11:00 UTC snapshot (read-only).
import json, sys, collections, numpy as np
ME = 16799346
F = sys.argv[1]
d = json.load(open(F)); L = sorted(d["lb"], key=lambda r: r["rank"]); T = {t["teamId"]: t for t in d["teams"]}
N = len(L); me = [r for r in L if r["teamId"] == ME][0]
print("fetched", d["fetched_utc"][:16], "N", N, "us", me)
c = collections.OrderedDict(); first = {}
for r in L:
    s = r["displayScore"]; c[s] = c.get(s, 0) + 1; first.setdefault(s, r["rank"])
for s in list(c)[:20]:
    if float(s) >= 0.954: print(f"  {s}: r{first[s]}-r{first[s]+c[s]-1} n={c[s]}")
print("silver r", N // 20, "bronze r", N // 10, "gold r", 10 + N // 500)
# increasing-sid runs (exact ties) inside .958 and .957
runs = []; cur = [L[0]]
for a, b in zip(L, L[1:]):
    if b["displayScore"] == a["displayScore"] and b["submissionId"] > a["submissionId"]: cur.append(b)
    else: runs.append(cur); cur = [b]
runs.append(cur)
for band in ("0.958", "0.957"):
    rr = [r for r in runs if r[0]["displayScore"] == band]
    long = [r for r in rr if len(r) >= 4]
    print(band, "runs:", len(rr), "long(>=4):", len(long), "members in long:", sum(len(r) for r in long),
          "first long run r", long[0][0]["rank"] if long else None, "last long run end r", long[-1][-1]["rank"] if long else None)
    h = collections.Counter(min(len(r), 10) for r in rr); print("   run-length hist (10=10+):", sorted(h.items()))
# zones in .958
b0 = first["0.958"]; b1 = b0 + c["0.958"] - 1
long958 = [r for r in runs if r[0]["displayScore"] == "0.958" and len(r) >= 4]
cstart = long958[0][0]["rank"] if long958 else None
print("in-band above us:", me["rank"] - b0, "| band r", b0, "-", b1, "| first long tie run starts r", cstart)
def z(a, b, name):
    rows = [r for r in L if a <= r["rank"] <= b]
    if not rows: return
    sid = np.array([r["submissionId"] for r in rows]); sc = np.array([T[r["teamId"]]["submissionCount"] or 0 for r in rows])
    q = lambda x: np.percentile(x, [10, 50, 90]).astype(int).tolist()
    print(f"{name:24s} r{a}-{b} n={len(rows)} sid p10/50/90 {q(sid)} subs p10/50/90 {q(sc)} frac sid<57251394(heliosli pub) {np.mean(sid<57251394):.2f}")
z(1, 10 + N // 500, "gold zone"); z(11 + N // 500, first.get("0.959", 1) - 1, ">= .960"); z(first["0.959"], b0 - 1, ".959")
z(b0, me["rank"] - 1, "V (in band above us)"); z(me["rank"] + 1, N // 20, "W to silver cut"); z(N // 20 + 1, (cstart or b1) - 1, "W below cut, pre-runs")
if cstart: z(cstart, b1, "runs zone (.958)")
z(b1 + 1, b1 + c["0.957"], ".957")
json.dump(dict(N=N, me=me, bands={s: [first[s], c[s]] for s in c if float(s) >= .95}, cstart=cstart), open("bands_now.json", "w"), indent=1)
