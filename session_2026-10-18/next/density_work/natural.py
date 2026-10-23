# natural.py: frozen "natural" teams (best sid older than a cutoff) as a fixed-score reference sample.
# For each snapshot where our entry is X: per-band counts of natural teams, and natural teams above us in our band.
import json, sys, collections
sys.path.insert(0, "."); from sidtime import t_of
ME = 16799346
P = "C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-14/research960/lbprobe/"
snap = sys.argv[1]; cuts = [int(x) for x in sys.argv[2:]] or [57178000, 57201874, 57234000]
d = json.load(open(P + snap)); L = sorted(d["lb"], key=lambda r: r["rank"]); me = [r for r in L if r["teamId"] == ME][0]
print(snap, d["fetched_utc"][:16], "us", me["submissionId"], "r", me["rank"], me["displayScore"])
for c in cuts:
    nat = [r for r in L if r["submissionId"] < c]
    cnt = collections.Counter(r["displayScore"] for r in nat)
    band = [r for r in nat if r["displayScore"] == me["displayScore"]]
    above = sum(r["rank"] < me["rank"] for r in band)
    bands = " ".join(f"{s}:{cnt[s]}" for s in ["0.964", "0.963", "0.962", "0.961", "0.960", "0.959", "0.958", "0.957", "0.956", "0.955"])
    print(f"  cutoff sid<{c} ({t_of(c)} UTC): natural in our band {len(band)}, above us {above}, below {len(band)-above} | per band {bands}")
