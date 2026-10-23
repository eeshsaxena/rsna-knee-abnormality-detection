import json, numpy as np
ME = 16799346
d = json.load(open("C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-17/research_1hr/api_lb_1017T_lens.json"))
T = {t["teamId"]: t for t in d["teams"]}
L = sorted(d["lb"], key=lambda r: r["rank"])
def z(a, b, name):
    rows = [r for r in L if a <= r["rank"] <= b]
    sid = np.array([r["submissionId"] for r in rows]); sc = np.array([T[r["teamId"]]["submissionCount"] or 0 for r in rows])
    q = lambda x: np.percentile(x, [10, 50, 90]).astype(int).tolist()
    print(f"{name:22s} r{a}-{b} n={len(rows)} sid p10/50/90 {q(sid)} subs p10/50/90 {q(sc)} frac subs<=10 {np.mean(sc<=10):.2f} frac sid<57234000 {np.mean(sid<57234000):.2f}")
z(1, 21, "gold zone"); z(22, 93, ".960+"); z(94, 141, ".959"); z(142, 170, "V above us in band"); z(172, 307, "W to silver cut"); z(308, 387, "W below cut"); z(388, 700, "runs zone a"); z(701, 995, "runs zone b"); z(996, 1218, ".957")
# R6-time snapshot (B visible) 08:45 UTC 10-16
P = "C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-14/research960/lbprobe/"
d1 = json.load(open(P + "api_lb_1420.json")); L1 = sorted(d1["lb"], key=lambda r: r["rank"])
me1 = [r for r in L1 if r["teamId"] == ME][0]; print(d1["fetched_utc"][:16], me1)
above_r6 = {r["teamId"] for r in L1 if r["rank"] < me1["rank"]}
above_hw = {r["teamId"] for r in L if r["rank"] < 171}
print("B (R6) rank lower bound now:", 1 + len(above_r6 | above_hw), "| above R6 then", len(above_r6), "| union adds", len(above_r6 - above_hw))
pos_now = {r["teamId"]: r["rank"] for r in L}
x = sorted(pos_now[t] for t in above_r6 - above_hw if t in pos_now)
print("teams above R6 then but below HW now, current ranks:", x[:40])
