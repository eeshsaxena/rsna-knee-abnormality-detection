# bracket.py: public rank bracket TODAY (10-19 11:00 UTC) for our non-best entries R6, HW, a25 (MEASURED logic as research_1hr/bc.py):
# lower bound = 1 + |teams above X at X's read  U  teams above any better entry of ours at its read  U  teams above c4 now|;
# upper bound = rank just above the highest frozen team that was below X at X's read.
import json
ME = 16799346
P = "C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-14/research960/lbprobe/"
now = json.load(open("C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/work/api_lb_1019T_think.json"))["lb"]
pos = {r["teamId"]: r for r in now}
reads = [("R6", P + "api_lb_1420.json"), ("HW", P + "api_lb_chain_0105_2.json"), ("a25", P + "api_lb_1018T0645.json")]
above_c4_now = {r["teamId"] for r in now if r["rank"] < pos[ME]["rank"]}
for nm, f in reads:
    d = json.load(open(f))["lb"]; me = [r for r in d if r["teamId"] == ME][0]
    above = {r["teamId"] for r in d if r["rank"] < me["rank"]}
    lo = 1 + len(above | above_c4_now) - (1 if ME in above else 0)
    below_frozen = [pos[r["teamId"]]["rank"] for r in d if r["rank"] > me["rank"] and r["teamId"] in pos and pos[r["teamId"]]["submissionId"] == r["submissionId"] and r["teamId"] != ME]
    hi = min(below_frozen) if below_frozen else None
    print(f"{nm}: read r{me['rank']} at that time; TODAY bracket r{lo} .. r{hi} (silver r{len(now)//20})")
