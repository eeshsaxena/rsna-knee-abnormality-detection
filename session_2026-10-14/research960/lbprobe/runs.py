import json, sys, numpy as np, pandas as pd
d = json.load(open(sys.argv[1])); L = pd.DataFrame(d["lb"])
for sc in ["0.958","0.957","0.956","0.955","0.954"]:
    b = L[L.displayScore == sc].sort_values("rank"); ids = b.submissionId.values
    inc = (np.diff(ids) > 0); runs = []; cur = 1
    for x in inc:
        if x: cur += 1
        else: runs.append(cur); cur = 1
    runs.append(cur)
    print(f"{sc}: r{b['rank'].min()}-{b['rank'].max()} n={len(b)} frac_inc={inc.mean():.3f} nruns={len(runs)} longest={max(runs)} top5={sorted(runs)[-5:]}")
b = L[L.displayScore == "0.956"].sort_values("rank").reset_index(drop=True)
# print runs >= 6 with their rank spans and id ranges
ids = b.submissionId.values; start = 0
for i in range(1, len(ids)+1):
    if i == len(ids) or ids[i] < ids[i-1]:
        if i - start >= 6: print(f"  run len {i-start}: r{b['rank'][start]}-{b['rank'][i-1]} ids {ids[start]}..{ids[i-1]}")
        start = i
# block summary: median id per 25 ranks
for k in range(0, len(b), 25):
    s = b.iloc[k:k+25]; print(f"  r{s['rank'].min()}-{s['rank'].max()}: median id {int(s.submissionId.median())} frac<57137000 {(s.submissionId<57137000).mean():.2f} frac>57150000 {(s.submissionId>57150000).mean():.2f}")
