# fetch_lb_full.py OUT_FULL OUT_TRIM: anonymous read-only Kaggle web-API leaderboard (public board, teams + members).
# Saves the raw response (full) and the fetch_api_lb.py-compatible trimmed form; prints top-level keys (private board check).
import sys, json, time, requests, datetime
s = requests.Session(); s.get("https://www.kaggle.com/competitions", timeout=60)
h = {"x-xsrf-token": s.cookies.get("XSRF-TOKEN", "")}
def api(svc, m, b):
    for i in range(4):
        r = s.post(f"https://www.kaggle.com/api/i/{svc}/{m}", json=b, headers=h, timeout=90)
        if r.status_code == 200: return r.json()
        time.sleep(3*(i+1))
    raise SystemExit(f"{m} HTTP {r.status_code}")
lb = api("competitions.LeaderboardService", "GetLeaderboard", {"competitionId": 154281})
now = datetime.datetime.utcnow().isoformat()
print("keys:", {k: (len(v) if isinstance(v, list) else v) for k, v in lb.items() if k != "teams" or True} if False else {k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in lb.items()})
lb["_fetched_utc"] = now
json.dump(lb, open(sys.argv[1], "w"))
json.dump({"fetched_utc": now, "lb": lb["publicLeaderboard"],
           "teams": [{k: t.get(k) for k in ("teamId","teamName","submissionCount","lastSubmissionDate")} for t in lb["teams"]]}, open(sys.argv[2], "w"))
me = [r for r in lb["publicLeaderboard"] if r["teamId"] == 16799346][0]
print(now, len(lb["publicLeaderboard"]), me)
