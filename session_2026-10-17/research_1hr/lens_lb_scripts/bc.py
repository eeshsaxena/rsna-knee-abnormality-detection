import json
ME = 16799346
P = "C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-14/research960/lbprobe/"
now = json.load(open("C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-17/research_1hr/api_lb_1017T_lens.json"))["lb"]
pos = {r["teamId"]: r for r in now}
for f in ["api_lb_r6read_1152.json", "api_lb_kc1read.json", "api_lb_comboread.json", "api_lb_1420.json"]:
    d = json.load(open(P + f)); L = d["lb"]; me = [r for r in L if r["teamId"] == ME][0]
    below = [r for r in L if r["rank"] > me["rank"] and r["displayScore"] == me["displayScore"]]
    # teams below R6 then whose best sid is unchanged now (frozen): their score unchanged, so they are below B now too
    frozen_below = [r for r in below if r["teamId"] in pos and pos[r["teamId"]]["submissionId"] == r["submissionId"]]
    rk = sorted(pos[r["teamId"]]["rank"] for r in frozen_below)
    above = [r for r in L if r["rank"] < me["rank"]]
    rka = sorted(pos[r["teamId"]]["rank"] for r in above if r["teamId"] in pos)
    print(f, d["fetched_utc"][:16], "R6/our best", me["submissionId"], "r", me["rank"], "| in-band below:", len(below), "frozen since:", len(frozen_below),
          "-> their ranks now min/med/max", (rk[0], rk[len(rk)//2], rk[-1]) if rk else None, "| above-then lowest now:", rka[-3:])
