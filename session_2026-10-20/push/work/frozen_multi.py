# frozen_multi.py: zone counts of frozen teams (best sid unchanged in every snapshot since a cutoff) for several cutoffs. Read-only.
import json, glob, collections
R = "C:/Users/eeshs/Downloads/rsna-knee/"; LB = R + "session_2026-10-14/research960/lbprobe/"
ME = 16799346
def load(f):
    d = json.load(open(f)); return d["fetched_utc"][:16], {r["teamId"]: r for r in d["lb"]}
S417 = load(R + "session_2026-10-20/forum/api_lb_1020T0417.json")[1]; E1 = load(LB + "api_lb_1020_eoaread1.json")[1]; E2 = load(LB + "api_lb_1020_eoaread2.json")[1]
snaps = {}
for f in glob.glob(LB + "api_lb_*.json") + glob.glob(R + "session_2026-10-1[89]/**/api_lb_*.json", recursive=True) + glob.glob(R + "session_2026-10-20/**/api_lb_*.json", recursive=True):
    try: d = json.load(open(f))
    except Exception: continue
    if "lb" not in d: continue
    snaps[d["fetched_utc"][:16]] = {r["teamId"]: r["submissionId"] for r in d["lb"]}
ts = sorted(snaps)
rc4 = S417[ME]["rank"]; r10 = E1[ME]["rank"]; r20 = E2[ME]["rank"]
def zones(keys):
    Z = collections.Counter()
    for k in keys:
        if k == ME or k not in E2: continue
        s = E2[k]["displayScore"]
        if s >= "0.961": Z["4_.961+"] += 1; Z["5_" + s] += 1
        elif s == "0.960": Z["3_.960"] += 1
        elif s == "0.959":
            if E2[k]["rank"] < r20: Z["2c_.959>eoa20"] += 1
            elif E1[k]["rank"] < r10: Z["2b_.959 eoa10-20"] += 1
            else: Z["2a_.959<eoa10"] += 1
        elif s == "0.958":
            if k in S417 and S417[k]["rank"] < rc4 and S417[k]["displayScore"] == "0.958": Z["1_.958>c4"] += 1
            else: Z["0_.958<c4"] += 1
    return dict(sorted(Z.items()))
print("ALL teams:", zones(E2.keys()))
for cut in ("2026-10-16T08:45", "2026-10-17T16:21", "2026-10-18T12:52", "2026-10-19T10:41", "2026-10-19T20:17", "2026-10-20T04:17"):
    sel = [t for t in ts if t >= cut]
    fz = None
    for t in sel:
        m = snaps[t]
        if fz is None: fz = dict(m); continue
        fz = {k: v for k, v in fz.items() if m.get(k) == v}
    z = zones(fz.keys())
    print(f"frozen since {cut} ({len(sel)} snaps, n {len(fz)}):", {k: v for k, v in z.items() if not k.startswith('5_') and not k.startswith('0_')}, "| .961-.963:", sum(v for k, v in z.items() if k[2:] in ('0.961','0.962','0.963')))
