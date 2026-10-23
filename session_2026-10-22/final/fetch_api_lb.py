# fetch_api_lb.py OUTNAME: anonymous read of the public Kaggle web API leaderboard (team best submissionId + rank). Read-only.
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
out = sys.argv[1] if len(sys.argv) > 1 else "api_lb_" + datetime.datetime.utcnow().strftime("%m%dT%H%M") + ".json"
json.dump({"fetched_utc": datetime.datetime.utcnow().isoformat(), "lb": lb["publicLeaderboard"],
           "teams": [{k: t.get(k) for k in ("teamId","teamName","submissionCount","lastSubmissionDate")} for t in lb["teams"]]}, open(out, "w"))
me = [r for r in lb["publicLeaderboard"] if r["teamId"] == 16799346][0]
print(out, len(lb["publicLeaderboard"]), me)
