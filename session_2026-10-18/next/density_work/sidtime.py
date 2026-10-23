# sidtime.py: sid -> UTC time map from the lower envelope of (best sid, team lastSubmissionDate) over all snapshots.
import json, glob, datetime, bisect
P = "C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-14/research960/lbprobe/"
pts = {}
for f in glob.glob(P + "api_lb_*.json"):
    d = json.load(open(f)); T = {t["teamId"]: t for t in d["teams"]}
    for r in d["lb"]:
        ls = T.get(r["teamId"], {}).get("lastSubmissionDate")
        if not ls: continue
        t = datetime.datetime.fromisoformat(ls.replace("Z", "+00:00")).timestamp()
        s = r["submissionId"]; pts[s] = min(pts.get(s, 9e18), t)
S = sorted(pts)
# lower envelope from the right: time(s) <= min over s' >= s of lastSub(s')
env = []; m = 9e18
for s in reversed(S):
    m = min(m, pts[s]); env.append((s, m))
env.reverse()
def t_of(sid):
    i = bisect.bisect_left([e[0] for e in env], sid)
    return datetime.datetime.utcfromtimestamp(env[min(i, len(env)-1)][1]).strftime("%m-%d %H:%M")
if __name__ == "__main__":
    for sid in [57131112, 57146261, 57171955, 57174021, 57190000, 57201874, 57222884, 57234000, 57248084, 57251394, 57262115, 57282622, 57291203, 57310758, 57316395, 57323261, 57338202]:
        print(sid, t_of(sid))
