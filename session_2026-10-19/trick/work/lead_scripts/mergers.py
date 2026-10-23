import json, pandas as pd, numpy as np
R = 'C:/Users/eeshs/Downloads/rsna-knee/'; W = R + 'session_2026-10-19/trick/work/'
c08 = pd.read_csv(R + 'research_1008/A_lists/lb_now/rsna-knee-abnormality-detection-publicleaderboard-2026-10-08T16_41_24.csv', encoding='utf-8-sig')
c08['mem08'] = c08.TeamMemberUserNames.fillna('').str.split(',')
full = json.load(open(W + 'lead_lb/api_lb_full_1019_now.json'))
mem_now = {t['teamId']: [m['userName'] for m in (t.get('teamMembers') or [])] for t in full['teams']}
m08 = dict(zip(c08.TeamId, c08.mem08))
E = pd.read_csv(W + 'lead_events.csv', parse_dates=['snap', 'sub_t'])
S = pd.read_parquet(W + 'lead_ts.parquet')
snaps = sorted(S[S.src.isin(['api', 'api1008'])].t.unique())
prev_of = {snaps[i]: snaps[i - 1] for i in range(1, len(snaps))}
E['prev_snap'] = E.snap.map(prev_of)
E['stale_h'] = (E.prev_snap - E.sub_t).dt.total_seconds() / 3600   # >7 => best sub predates prev snapshot by >7h
E['nmem_now'] = E.teamId.map(lambda x: len(mem_now.get(x, [])))
E['nmem_08'] = E.teamId.map(lambda x: len(m08[x]) if x in m08 else np.nan)
E['merger_suspect'] = (E.stale_h > 7)
J = E[(E.s1 >= .959) & ((E.s0 < .959) | E.s0.isna())]
print('crossing events to .959+ :', len(J), ' merger-suspect (best sub predates previous snapshot by >7h):', J.merger_suspect.sum())
print(J[J.merger_suspect][['teamId', 'prev_snap', 'snap', 'sub_t', 's0', 's1', 'nmem_08', 'nmem_now']].to_string(index=False))
big = E[(E.s1 >= .959) & (E.d >= .003)]
print('\nbig jumps >= .003 to .959+:', len(big), ' merger suspect:', big.merger_suspect.sum())
print('team size change 10-08 -> now among .959+ teams:')
L = pd.DataFrame(full['publicLeaderboard']); L['score'] = L.displayScore.astype(float)
top = L[L.score >= .959].copy()
top['n08'] = top.teamId.map(lambda x: len(m08[x]) if x in m08 else np.nan); top['nnow'] = top.teamId.map(lambda x: len(mem_now[x]))
print(pd.crosstab(top.n08.fillna(-1), top.nnow))
E.to_csv(W + 'lead_events.csv', index=False)
