# api_frozen.py OLD.json NEW.json: read our team-best change between two API snapshots (fetch_api_lb.py).
# "frozen" = team whose best submissionId did not change (stricter and larger than the CSV LastSubmissionDate rule,
# which drops every team that submitted at all). Prints our best id, rank, frozen teams jumped, and band context.
import sys, json, pandas as pd
ME = 16799346
def load(p):
    d = json.load(open(p)); L = pd.DataFrame(d["lb"]); L["fetched"] = d.get("fetched_utc", "?"); return L
A, B = load(sys.argv[1]), load(sys.argv[2])
m = A.merge(B, on="teamId", suffixes=("_a", "_b"))
fz = m[m.submissionId_a == m.submissionId_b]
ma, mb = A[A.teamId == ME].iloc[0], B[B.teamId == ME].iloc[0]
up_a, up_b = (fz["rank_a"] < ma["rank"]).sum(), (fz["rank_b"] < mb["rank"]).sum()
print(f"{A.fetched.iloc[0][:16]} -> {B.fetched.iloc[0][:16]} UTC | our best sub {ma.submissionId} -> {mb.submissionId} "
      f"({'CHANGED' if ma.submissionId != mb.submissionId else 'same'}) | rank {ma['rank']} -> {mb['rank']} | score {ma.displayScore} -> {mb.displayScore}")
print(f"frozen teams {len(fz)}/{len(m)} ({len(fz)/len(m):.3f}); frozen ranked above us {up_a} -> {up_b} (jumped {up_a - up_b})")
band = B[B.displayScore == mb.displayScore]
print(f"our band {mb.displayScore}: r{band['rank'].min()}-{band['rank'].max()} ({len(band)}); above us in band {mb['rank'] - band['rank'].min()}; "
      f"silver cut r{len(B)//20}, bronze r{len(B)//10}, gold r{10 + len(B)//500}")
# who did we jump (frozen teams that were above us and are now below): their best ids and approx times
j = fz[(fz["rank_a"] < ma["rank"]) & (fz["rank_b"] > mb["rank"])]
if len(j): print("jumped frozen teams' best ids:", sorted(j.submissionId_a.tolist())[:30])
