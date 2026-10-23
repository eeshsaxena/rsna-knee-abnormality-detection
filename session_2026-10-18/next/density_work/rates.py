# rates.py: growth rates of #>=.960, #>=.959, N, and teams above our frozen entry inside the band. Read-only.
import json, glob, datetime, numpy as np
ME = 16799346
P = "C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-14/research960/lbprobe/"
rows = {}
for f in glob.glob(P + "api_lb_*.json"):
    d = json.load(open(f)); ts = d["fetched_utc"][:16]
    if ts in rows: continue
    L = d["lb"]; me = [r for r in L if r["teamId"] == ME][0]
    t = datetime.datetime.fromisoformat(d["fetched_utc"]).timestamp() / 3600
    ge = lambda s: sum(float(r["displayScore"]) >= s - 1e-9 for r in L)
    rows[ts] = dict(t=t, N=len(L), g960=ge(.960), g959=ge(.959), g958=ge(.958), rank=me["rank"], sid=me["submissionId"], above_in_band=me["rank"] - 1 - ge(float(me["displayScore"]) + .001))
R = [rows[k] | {"ts": k} for k in sorted(rows)]
T0 = R[-1]["t"]
def fit(key, since_h, sid=None):
    xs = [(r["t"] - T0, r[key]) for r in R if r["t"] - T0 >= -since_h and (sid is None or r["sid"] == sid)]
    x, y = np.array(xs).T; b, a = np.polyfit(x, y, 1); return b * 24, len(xs)
for key in ["N", "g960", "g959", "g958"]:
    print(key, "now", R[-1][key], "| per day: last 24h %.1f (n%d), 48h %.1f (n%d), 72h %.1f (n%d)" % (*fit(key, 24), *fit(key, 48), *fit(key, 72)))
print("above-in-band, HW frozen 10-16 15:17..10-17 19:35: per day %.1f (n%d)" % fit("above_in_band", 200, 57262115))
print("above-in-band, a25 frozen 10-17 21:13..now: per day %.1f (n%d)" % fit("above_in_band", 200, 57316395))
print("rank, HW frozen: per day %.1f; a25 frozen: per day %.1f" % (fit("rank", 200, 57262115)[0], fit("rank", 200, 57316395)[0]))
for r in R[-14:]: print(r["ts"], r["N"], r["g960"], r["g959"], r["rank"], r["above_in_band"], r["sid"])
