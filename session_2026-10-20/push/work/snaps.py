# snaps.py: list LB snapshots (read-only), our row, band ranges. Aggregates of the public LB only.
import json, glob, os, collections
R = "C:/Users/eeshs/Downloads/rsna-knee/"
ME = 16799346
files = glob.glob(R+"session_2026-10-14/research960/lbprobe/api_lb_*.json") + glob.glob(R+"session_2026-10-1[89]/**/api_lb_*.json", recursive=True) + glob.glob(R+"session_2026-10-20/**/api_lb_*.json", recursive=True)
rows = []
for f in files:
    try: d = json.load(open(f))
    except Exception as e: continue
    if not isinstance(d, dict) or "lb" not in d: continue
    L = d["lb"]; me = [r for r in L if r["teamId"] == ME]
    if not me: continue
    me = me[0]
    bands = collections.OrderedDict()
    for r in sorted(L, key=lambda r: r["rank"]):
        s = r["displayScore"]; b = bands.setdefault(s, [r["rank"], r["rank"], 0]); b[1] = r["rank"]; b[2] += 1
    rows.append((d.get("fetched_utc","?")[:16], os.path.relpath(f, R), len(L), me["rank"], me["submissionId"], me["displayScore"],
                 {k: v for k, v in bands.items() if k in ("0.961","0.960","0.959","0.958")}))
rows.sort()
seen = set()
for t, f, n, rk, sid, ds, b in rows:
    if t in seen: continue
    seen.add(t)
    print(t, f.split("/")[-1][:34].ljust(34), n, "us", rk, sid, ds, " ".join(f"{k[2:]}:{v[0]}-{v[1]}({v[2]})" for k, v in b.items()))
