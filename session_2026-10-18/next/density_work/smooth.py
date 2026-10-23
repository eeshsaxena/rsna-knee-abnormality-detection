# smooth.py (MODEL, cross-check): all-teams smooth-tail estimate of our in-band position per snapshot.
# Density g(s) = A exp(-k (s - .959)/.001) continued from the .959 band into the top of the .958 band.
import json, glob, numpy as np
ME = 16799346
P = "C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-14/research960/lbprobe/"
seen = set()
for f in sorted(glob.glob(P + "api_lb_*.json"), key=lambda f: json.load(open(f))["fetched_utc"]):
    d = json.load(open(f)); ts = d["fetched_utc"][:16]
    if ts in seen: continue
    seen.add(ts); L = d["lb"]; me = [r for r in L if r["teamId"] == ME][0]
    if me["submissionId"] not in (57262115, 57316395): continue
    n959 = sum(r["displayScore"] == "0.959" for r in L); n960 = sum(r["displayScore"] == "0.960" for r in L)
    above = sum(1 for r in L if r["displayScore"] == "0.958" and r["rank"] < me["rank"])
    out = []
    for k in (0.3, np.log(n959 / n960), 0.9):
        A = n959 * k / (1 - np.exp(-k)); y = np.log(1 + above * k / A) / k; out.append(1 - y)
    print(f"{ts} sid {me['submissionId']} above {above} n959 {n959} n960 {n960}  u(k .3 / k from .959:.960 / k .9) = " + " / ".join(f"{x:.2f}" for x in out))
